#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the cross-attention
// text-video conditioning backward. Per definition.json:
//   grad_output      : [B, N_v, D=1024]            fp32
//   video_latents    : [B, N_v, D=1024]            fp32   (saved forward state)
//   text_embeddings  : [B, N_t, D=1024]            fp32   (saved forward state)
//   query_weight     : [D=1024, D=1024]            fp32
//   query_bias       : [D=1024]                    fp32
//   key_weight       : [D=1024, D=1024]            fp32
//   key_bias         : [D=1024]                    fp32
//   value_weight     : [D=1024, D=1024]            fp32
//   value_bias       : [D=1024]                    fp32
//   output_weight    : [D=1024, D=1024]            fp32
//   output_bias      : [D=1024]                    fp32   (unused-by-backward)
//   scale            : scalar float
//   grad_video_latents  : [B, N_v, D=1024]         fp32 DPS
//   grad_text_embeddings: [B, N_t, D=1024]         fp32 DPS
//   grad_query_weight   : [D=1024, D=1024]         fp32 DPS
//   grad_query_bias     : [D=1024]                 fp32 DPS
//   grad_key_weight     : [D=1024, D=1024]         fp32 DPS
//   grad_key_bias       : [D=1024]                 fp32 DPS
//   grad_value_weight   : [D=1024, D=1024]         fp32 DPS
//   grad_value_bias     : [D=1024]                 fp32 DPS
//   grad_output_weight  : [D=1024, D=1024]         fp32 DPS
//   grad_output_bias    : [D=1024]                 fp32 DPS
//
// Const axes pinned per the spec: num_heads = 16, head_dim = 64,
// hidden_size = num_heads * head_dim = 1024. N_v (num_video_tokens)
// and N_t (num_text_tokens) are workload-driven and recovered from
// input tensor shapes at runtime.
//
// Algorithm (mirrors the reference exactly):
//   Forward recomputation:
//     queries  = F.linear(video_latents, query_weight, query_bias)
//                .view(B, N_v, H, d).transpose(1, 2)
//     keys     = F.linear(text_embeddings, key_weight, key_bias)
//                .view(B, N_t, H, d).transpose(1, 2)
//     values   = F.linear(text_embeddings, value_weight, value_bias)
//                .view(B, N_t, H, d).transpose(1, 2)
//     attn_scores = (queries @ keys.transpose(-2, -1)) * scale
//     attn_probs  = softmax(attn_scores, dim=-1)
//     context     = attn_probs @ values
//     context     = context.transpose(1, 2).contiguous()
//                          .view(B, N_v, D)
//   Output projection backward:
//     grad_context = torch.matmul(grad_output, output_weight)
//     grad_output_weight = einsum('bnd,bnk->dk', grad_output, context)
//     grad_output_bias = grad_output.sum(dim=(0, 1))
//   Head merge + attention-application backward:
//     grad_attn_probs = grad_context_heads @ values.transpose(-2, -1)
//     grad_values_heads = attn_probs.transpose(-2, -1) @ grad_context_heads
//   Softmax backward:
//     sum_grad = (grad_attn_probs * attn_probs).sum(-1, keepdim=true)
//     grad_attn_scores = attn_probs * (grad_attn_probs - sum_grad)
//   Scale backward: grad_attn_scores *= scale
//   Scaled-dot-product backward:
//     grad_queries_heads = grad_attn_scores @ keys
//     grad_keys_heads = grad_attn_scores.transpose(-2, -1) @ queries
//   Head split + Q/K/V projection backward:
//     grad_queries / grad_keys / grad_values:
//       transpose(1, 2).contiguous().view(B, *, D)
//     grad_video_latents = torch.matmul(grad_queries, query_weight)
//     grad_query_weight = einsum('bnd,bnk->dk', grad_queries, video_latents)
//     grad_query_bias = grad_queries.sum(dim=(0, 1))
//     ... analogous for K, V on text_embeddings ...
//     grad_text_embeddings = grad_text_from_keys + grad_text_from_values
//
// Dispatch surfaces:
//   * The four projection grad_input dispatches use
//     `at::matmul(g, W)` directly -- the cuBLAS layer binds
//     these as OP_N OP_N (the verified
//     `cublasSgemm_basic_fp32` family). Unlike
//     `at::linear(g, W.t())` which binds OP_N OP_T, the
//     direct `at::matmul(g, W)` call is the canonical
//     verified-family binding.
//   * The four grad_weight dispatches use
//     `at::matmul(g_2d.transpose(0, 1), x_2d)` -- binding
//     OP_T OP_N (also the verified family's transposed-A
//     grad_weight surface).
//   * The four 4D batched matmuls bind
//     `cublasSgemmStridedBatched` (fp32).
//   * Softmax / scale / sum reductions run through ATen
//     elementwise / reduction ops; the axis-list reduction
//     args are typed `std::vector<int64_t>` to bind the
//     int-axis overload unambiguously.
void run(
    torch::Tensor grad_output,
    torch::Tensor video_latents,
    torch::Tensor text_embeddings,
    torch::Tensor query_weight,
    torch::Tensor query_bias,
    torch::Tensor key_weight,
    torch::Tensor key_bias,
    torch::Tensor value_weight,
    torch::Tensor value_bias,
    torch::Tensor output_weight,
    torch::Tensor output_bias,
    double scale,
    torch::Tensor grad_video_latents,
    torch::Tensor grad_text_embeddings,
    torch::Tensor grad_query_weight,
    torch::Tensor grad_query_bias,
    torch::Tensor grad_key_weight,
    torch::Tensor grad_key_bias,
    torch::Tensor grad_value_weight,
    torch::Tensor grad_value_bias,
    torch::Tensor grad_output_weight,
    torch::Tensor grad_output_bias
) {
    check_fp32_contiguous_cuda(grad_output, "grad_output");
    check_fp32_contiguous_cuda(video_latents, "video_latents");
    check_fp32_contiguous_cuda(text_embeddings, "text_embeddings");
    check_fp32_contiguous_cuda(query_weight, "query_weight");
    check_fp32_contiguous_cuda(query_bias, "query_bias");
    check_fp32_contiguous_cuda(key_weight, "key_weight");
    check_fp32_contiguous_cuda(key_bias, "key_bias");
    check_fp32_contiguous_cuda(value_weight, "value_weight");
    check_fp32_contiguous_cuda(value_bias, "value_bias");
    check_fp32_contiguous_cuda(output_weight, "output_weight");
    check_fp32_contiguous_cuda(output_bias, "output_bias");
    check_fp32_contiguous_cuda(grad_video_latents, "grad_video_latents");
    check_fp32_contiguous_cuda(grad_text_embeddings, "grad_text_embeddings");
    check_fp32_contiguous_cuda(grad_query_weight, "grad_query_weight");
    check_fp32_contiguous_cuda(grad_query_bias, "grad_query_bias");
    check_fp32_contiguous_cuda(grad_key_weight, "grad_key_weight");
    check_fp32_contiguous_cuda(grad_key_bias, "grad_key_bias");
    check_fp32_contiguous_cuda(grad_value_weight, "grad_value_weight");
    check_fp32_contiguous_cuda(grad_value_bias, "grad_value_bias");
    check_fp32_contiguous_cuda(grad_output_weight, "grad_output_weight");
    check_fp32_contiguous_cuda(grad_output_bias, "grad_output_bias");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kHeadDim = 64;
    constexpr int64_t kHiddenSize = kNumHeads * kHeadDim;  // 1024

    // Recover variable axes from the canonical-shape inputs and
    // shape-check every other indexed tensor against them and
    // the pinned const axes.
    TORCH_CHECK(video_latents.dim() == 3 &&
                video_latents.size(2) == kHiddenSize,
                "video_latents must be [B, N_v, hidden_size=1024]");
    const int64_t B = video_latents.size(0);
    const int64_t N_v = video_latents.size(1);

    TORCH_CHECK(text_embeddings.dim() == 3 &&
                text_embeddings.size(0) == B &&
                text_embeddings.size(2) == kHiddenSize,
                "text_embeddings must be [B, N_t, hidden_size=1024]");
    const int64_t N_t = text_embeddings.size(1);

    TORCH_CHECK(grad_output.dim() == 3 &&
                grad_output.size(0) == B &&
                grad_output.size(1) == N_v &&
                grad_output.size(2) == kHiddenSize,
                "grad_output shape mismatch with [B, N_v, hidden_size]");

    TORCH_CHECK(query_weight.dim() == 2 &&
                query_weight.size(0) == kHiddenSize &&
                query_weight.size(1) == kHiddenSize,
                "query_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(query_bias.dim() == 1 &&
                query_bias.size(0) == kHiddenSize,
                "query_bias must be [hidden_size]");
    TORCH_CHECK(key_weight.dim() == 2 &&
                key_weight.size(0) == kHiddenSize &&
                key_weight.size(1) == kHiddenSize,
                "key_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(key_bias.dim() == 1 &&
                key_bias.size(0) == kHiddenSize,
                "key_bias must be [hidden_size]");
    TORCH_CHECK(value_weight.dim() == 2 &&
                value_weight.size(0) == kHiddenSize &&
                value_weight.size(1) == kHiddenSize,
                "value_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(value_bias.dim() == 1 &&
                value_bias.size(0) == kHiddenSize,
                "value_bias must be [hidden_size]");
    TORCH_CHECK(output_weight.dim() == 2 &&
                output_weight.size(0) == kHiddenSize &&
                output_weight.size(1) == kHiddenSize,
                "output_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(output_bias.dim() == 1 &&
                output_bias.size(0) == kHiddenSize,
                "output_bias must be [hidden_size]");

    // DPS output shape-checks.
    TORCH_CHECK(grad_video_latents.dim() == 3 &&
                grad_video_latents.size(0) == B &&
                grad_video_latents.size(1) == N_v &&
                grad_video_latents.size(2) == kHiddenSize,
                "grad_video_latents must be [B, N_v, hidden_size]");
    TORCH_CHECK(grad_text_embeddings.dim() == 3 &&
                grad_text_embeddings.size(0) == B &&
                grad_text_embeddings.size(1) == N_t &&
                grad_text_embeddings.size(2) == kHiddenSize,
                "grad_text_embeddings must be [B, N_t, hidden_size]");
    TORCH_CHECK(grad_query_weight.dim() == 2 &&
                grad_query_weight.size(0) == kHiddenSize &&
                grad_query_weight.size(1) == kHiddenSize,
                "grad_query_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(grad_query_bias.dim() == 1 &&
                grad_query_bias.size(0) == kHiddenSize,
                "grad_query_bias must be [hidden_size]");
    TORCH_CHECK(grad_key_weight.dim() == 2 &&
                grad_key_weight.size(0) == kHiddenSize &&
                grad_key_weight.size(1) == kHiddenSize,
                "grad_key_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(grad_key_bias.dim() == 1 &&
                grad_key_bias.size(0) == kHiddenSize,
                "grad_key_bias must be [hidden_size]");
    TORCH_CHECK(grad_value_weight.dim() == 2 &&
                grad_value_weight.size(0) == kHiddenSize &&
                grad_value_weight.size(1) == kHiddenSize,
                "grad_value_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(grad_value_bias.dim() == 1 &&
                grad_value_bias.size(0) == kHiddenSize,
                "grad_value_bias must be [hidden_size]");
    TORCH_CHECK(grad_output_weight.dim() == 2 &&
                grad_output_weight.size(0) == kHiddenSize &&
                grad_output_weight.size(1) == kHiddenSize,
                "grad_output_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(grad_output_bias.dim() == 1 &&
                grad_output_bias.size(0) == kHiddenSize,
                "grad_output_bias must be [hidden_size]");

    const std::vector<int64_t> kBatchSeqDims = {0, 1};
    const float scale_f = static_cast<float>(scale);

    // -------------------------------------------------------------
    // Forward recomputation: queries / keys / values per-head, plus
    // attention probabilities and context. Mirrors the reference
    // exactly (F.linear adds the bias, then view + transpose to
    // [B, H, *, d] layout).
    // -------------------------------------------------------------
    auto queries = at::linear(video_latents, query_weight, query_bias)
                       .view({B, N_v, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B, H, N_v, d]
    auto keys = at::linear(text_embeddings, key_weight, key_bias)
                    .view({B, N_t, kNumHeads, kHeadDim})
                    .transpose(1, 2);     // [B, H, N_t, d]
    auto values = at::linear(text_embeddings, value_weight, value_bias)
                      .view({B, N_t, kNumHeads, kHeadDim})
                      .transpose(1, 2);   // [B, H, N_t, d]

    auto attention_scores = at::matmul(queries, keys.transpose(-2, -1))
                                * scale_f;  // [B, H, N_v, N_t]
    auto attention_probs = at::softmax(attention_scores, /*dim=*/-1);

    auto context_heads = at::matmul(attention_probs, values);  // [B, H, N_v, d]
    auto context = context_heads.transpose(1, 2).contiguous()
                       .view({B, N_v, kHiddenSize});  // [B, N_v, D]

    // -------------------------------------------------------------
    // Output projection backward (output = F.linear(context,
    // output_weight, output_bias); grad_context = grad_output
    // @ output_weight at the cuBLAS layer).
    // -------------------------------------------------------------
    auto grad_context = at::matmul(grad_output, output_weight);  // [B, N_v, D]

    auto grad_output_2d = grad_output.reshape({-1, kHiddenSize});
    auto context_2d = context.reshape({-1, kHiddenSize});
    grad_output_weight.copy_(
        at::matmul(grad_output_2d.transpose(0, 1), context_2d));
    grad_output_bias.copy_(grad_output.sum(kBatchSeqDims));

    // -------------------------------------------------------------
    // Head merge -> attention-application backward -> softmax
    // backward -> scale backward -> scaled-dot-product backward.
    // -------------------------------------------------------------
    auto grad_context_heads = grad_context
                                  .view({B, N_v, kNumHeads, kHeadDim})
                                  .transpose(1, 2);  // [B, H, N_v, d]

    auto grad_attention_probs = at::matmul(
        grad_context_heads, values.transpose(-2, -1));  // [B, H, N_v, N_t]
    auto grad_values_heads = at::matmul(
        attention_probs.transpose(-2, -1),
        grad_context_heads);  // [B, H, N_t, d]

    auto sum_grad_probs = (grad_attention_probs * attention_probs)
                              .sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_attention_scores = attention_probs
                                 * (grad_attention_probs - sum_grad_probs);

    grad_attention_scores = grad_attention_scores * scale_f;

    auto grad_queries_heads = at::matmul(
        grad_attention_scores, keys);  // [B, H, N_v, d]
    auto grad_keys_heads = at::matmul(
        grad_attention_scores.transpose(-2, -1),
        queries);  // [B, H, N_t, d]

    // -------------------------------------------------------------
    // Head split: transpose(1, 2).contiguous().view back to flat
    // per-token shape, then project backward through Q / K / V.
    // -------------------------------------------------------------
    auto grad_queries = grad_queries_heads.transpose(1, 2).contiguous()
                            .view({B, N_v, kHiddenSize});
    auto grad_keys = grad_keys_heads.transpose(1, 2).contiguous()
                         .view({B, N_t, kHiddenSize});
    auto grad_values = grad_values_heads.transpose(1, 2).contiguous()
                           .view({B, N_t, kHiddenSize});

    // Q projection backward (queries = F.linear(video_latents,
    // query_weight, query_bias); the canonical verified-family
    // grad_input binding is at::matmul(g, W) -- OP_N OP_N at cuBLAS).
    grad_video_latents.copy_(at::matmul(grad_queries, query_weight));
    auto grad_queries_2d = grad_queries.reshape({-1, kHiddenSize});
    auto video_latents_2d = video_latents.reshape({-1, kHiddenSize});
    grad_query_weight.copy_(
        at::matmul(grad_queries_2d.transpose(0, 1), video_latents_2d));
    grad_query_bias.copy_(grad_queries.sum(kBatchSeqDims));

    // K projection backward (keys = F.linear(text_embeddings,
    // key_weight, key_bias)).
    auto grad_text_from_keys = at::matmul(grad_keys, key_weight);
    auto grad_keys_2d = grad_keys.reshape({-1, kHiddenSize});
    auto text_embeddings_2d = text_embeddings.reshape({-1, kHiddenSize});
    grad_key_weight.copy_(
        at::matmul(grad_keys_2d.transpose(0, 1), text_embeddings_2d));
    grad_key_bias.copy_(grad_keys.sum(kBatchSeqDims));

    // V projection backward (values = F.linear(text_embeddings,
    // value_weight, value_bias)).
    auto grad_text_from_values = at::matmul(grad_values, value_weight);
    auto grad_values_2d = grad_values.reshape({-1, kHiddenSize});
    grad_value_weight.copy_(
        at::matmul(grad_values_2d.transpose(0, 1), text_embeddings_2d));
    grad_value_bias.copy_(grad_values.sum(kBatchSeqDims));

    // grad_text_embeddings accumulates contributions from both
    // the K and V projections.
    grad_text_embeddings.copy_(grad_text_from_keys + grad_text_from_values);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Cross-attention text-video conditioning backward. Pure ATen / "
        "cuBLAS dispatch chain: four fp32 2D cublasSgemm grad_input calls "
        "via at::matmul(g, W) (OP_N OP_N) + four fp32 2D cublasSgemm "
        "grad_weight calls via at::matmul(g_2d.t(), x_2d) (OP_T OP_N) + "
        "four fp32 4D cublasSgemmStridedBatched calls (attention @ V "
        "backward and Q @ K^T backward) + softmax backward and scale "
        "backward as ATen elementwise / reduction ops. Const axes: "
        "num_heads=16, head_dim=64, hidden_size=1024. Destination-passing "
        "style.");
}
