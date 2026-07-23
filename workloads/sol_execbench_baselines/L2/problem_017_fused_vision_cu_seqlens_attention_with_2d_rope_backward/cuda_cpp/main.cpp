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

static void check_int32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kInt32,
                name, " must be int32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// rotate_half(x) = cat([-x_second_half, x_first_half], dim=-1).
// Used in the 2D RoPE forward applied to Q / K.
static torch::Tensor rotate_half(const torch::Tensor& x) {
    const int64_t last_dim = x.size(-1);
    const int64_t half = last_dim / 2;
    auto x1 = x.slice(/*dim=*/-1, /*start=*/0, /*end=*/half);
    auto x2 = x.slice(/*dim=*/-1, /*start=*/half, /*end=*/last_dim);
    return at::cat({-x2, x1}, /*dim=*/-1);
}

// rotate_half_inverse(x) = cat([x_second_half, -x_first_half], dim=-1).
// Used in the 2D RoPE backward (gradient through rotate_half).
static torch::Tensor rotate_half_inverse(const torch::Tensor& x) {
    const int64_t last_dim = x.size(-1);
    const int64_t half = last_dim / 2;
    auto x1 = x.slice(/*dim=*/-1, /*start=*/0, /*end=*/half);
    auto x2 = x.slice(/*dim=*/-1, /*start=*/half, /*end=*/last_dim);
    return at::cat({x2, -x1}, /*dim=*/-1);
}

// Destination-passing-style entry for the fused vision
// cu_seqlens variable-length attention backward with 2D RoPE.
// Per definition.json:
//   grad_output         : [total_seq_len, hidden_size=3584]   fp32
//   hidden_states       : [total_seq_len, hidden_size=3584]   fp32
//   qkv_weight          : [qkv_out_size=10752, hidden_size]   fp32
//   qkv_bias            : [qkv_out_size=10752]                fp32
//   proj_weight         : [hidden_size, hidden_size]          fp32
//   proj_bias           : [hidden_size]                       fp32   (unused-by-backward)
//   cos                 : [total_seq_len, head_dim=128]       fp32
//   sin                 : [total_seq_len, head_dim=128]       fp32
//   cu_seqlens          : [num_seqlens=num_sequences+1]       int32
//   attention_dropout   : scalar float (always 0.0 in this workload)
//   scaling             : scalar float (head_dim^-0.5)
//   grad_hidden_states  : [total_seq_len, hidden_size]        fp32 DPS
//   grad_qkv_weight     : [qkv_out_size, hidden_size]         fp32 DPS
//   grad_qkv_bias       : [qkv_out_size]                      fp32 DPS
//   grad_proj_weight    : [hidden_size, hidden_size]          fp32 DPS
//   grad_proj_bias      : [hidden_size]                       fp32 DPS
//
// Const axes pinned per the spec: hidden_size = 3584,
// num_heads = 28, head_dim = 128, qkv_out_size = 3 * hidden_size
// = 10752. total_seq_len and num_sequences are workload-driven
// and recovered from input shapes at runtime.
//
// Algorithm (mirrors the reference exactly):
//   Forward recomputation:
//     qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
//     qkv.reshape(seq_len, 3, H, d).permute(1, 0, 2, 3).unbind(0)
//       -> (Q, K, V) each [seq_len, H, d]
//     cos_expanded = cos.unsqueeze(-2)   -> [seq_len, 1, d]
//     sin_expanded = sin.unsqueeze(-2)
//     q_rope = q * cos_expanded + rotate_half(q) * sin_expanded
//     k_rope = k * cos_expanded + rotate_half(k) * sin_expanded
//     q/k/v -> transpose(0, 1).unsqueeze(0)
//              -> [1, H, seq_len, d]
//     lengths = cu_seqlens[1:] - cu_seqlens[:-1]  (host-synced)
//     Per-sequence loop:
//       attn_weights = softmax((q @ k^T) * scaling, dim=-1)
//       attn_output  = attn_weights @ v
//     cat splits; reshape to [seq_len, hidden_size].
//   Output-projection backward:
//     grad_proj_bias = grad_output.sum(dim=0)
//     grad_proj_weight = grad_output.t() @ attn_output_reshaped
//     grad_attn_output = grad_output @ proj_weight
//     Reshape to [1, H, seq_len, d].
//   Per-sequence attention backward (same lengths split):
//     grad_attn_weights = grad_attn_out @ v.transpose(2, 3)
//     grad_v = attn_weights.transpose(2, 3) @ grad_attn_out
//     sum_grad = (grad_attn_weights * attn_weights)
//                .sum(-1, keepdim=true)
//     grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad)
//     grad_attn_scores *= scaling
//     grad_q = grad_attn_scores @ k
//     grad_k = grad_attn_scores.transpose(2, 3) @ q
//   Cat splits; squeeze(0).transpose(0, 1)
//     -> [seq_len, H, d].
//   RoPE backward (rotate_half_inverse):
//     grad_q_pre_rope = grad_q * cos_expanded
//                       + rotate_half_inverse(grad_q * sin_expanded)
//     grad_k_pre_rope = grad_k * cos_expanded
//                       + rotate_half_inverse(grad_k * sin_expanded)
//     grad_v_pre_rope = grad_v (no RoPE on V).
//   Stack -> [seq_len, 3, H, d];
//     reshape -> [seq_len, qkv_out_size=10752].
//   QKV-projection backward:
//     grad_hidden_states = grad_qkv @ qkv_weight
//     grad_qkv_weight    = grad_qkv.t() @ hidden_states
//     grad_qkv_bias      = grad_qkv.sum(dim=0)
//
// Dispatch surfaces:
//   * The four projection 2D matmul dispatches
//     (grad_attn_output via at::matmul(g, W);
//     grad_proj_weight via g.t() @ x;
//     grad_hidden_states via at::matmul(g, W);
//     grad_qkv_weight via g.t() @ x) bind cublasSgemm
//     OP_N OP_N (grad_input) or OP_T OP_N (grad_weight) --
//     both in the verified `cublasSgemm_basic_fp32` family.
//   * The 4D per-sequence batched matmuls (attention-app
//     backward and scaled-dot-product backward) bind
//     cublasSgemmStridedBatched with batch = num_heads = 28.
//   * The cu_seqlens host-sync forces a brief CPU read of
//     the int32 cumulative tensor (same shape as the
//     reference). It is the load-bearing site for the
//     per-sequence iteration count.
//   * Softmax / RoPE / reductions run through ATen
//     elementwise / reduction ops.
void run(
    torch::Tensor grad_output,
    torch::Tensor hidden_states,
    torch::Tensor qkv_weight,
    torch::Tensor qkv_bias,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor cu_seqlens,
    double attention_dropout,
    double scaling,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_qkv_weight,
    torch::Tensor grad_qkv_bias,
    torch::Tensor grad_proj_weight,
    torch::Tensor grad_proj_bias
) {
    check_fp32_contiguous_cuda(grad_output, "grad_output");
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(qkv_weight, "qkv_weight");
    check_fp32_contiguous_cuda(qkv_bias, "qkv_bias");
    check_fp32_contiguous_cuda(proj_weight, "proj_weight");
    check_fp32_contiguous_cuda(proj_bias, "proj_bias");
    check_fp32_contiguous_cuda(cos, "cos");
    check_fp32_contiguous_cuda(sin, "sin");
    check_int32_contiguous_cuda(cu_seqlens, "cu_seqlens");
    check_fp32_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_fp32_contiguous_cuda(grad_qkv_weight, "grad_qkv_weight");
    check_fp32_contiguous_cuda(grad_qkv_bias, "grad_qkv_bias");
    check_fp32_contiguous_cuda(grad_proj_weight, "grad_proj_weight");
    check_fp32_contiguous_cuda(grad_proj_bias, "grad_proj_bias");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    constexpr int64_t kHiddenSize = 3584;
    constexpr int64_t kNumHeads = 28;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kQkvOutSize = 3 * kHiddenSize;  // 10752

    TORCH_CHECK(hidden_states.dim() == 2 &&
                hidden_states.size(1) == kHiddenSize,
                "hidden_states must be [total_seq_len, hidden_size=3584]");
    const int64_t seq_len = hidden_states.size(0);

    TORCH_CHECK(grad_output.dim() == 2 &&
                grad_output.size(0) == seq_len &&
                grad_output.size(1) == kHiddenSize,
                "grad_output must be [total_seq_len, hidden_size]");
    TORCH_CHECK(qkv_weight.dim() == 2 &&
                qkv_weight.size(0) == kQkvOutSize &&
                qkv_weight.size(1) == kHiddenSize,
                "qkv_weight must be [qkv_out_size=10752, hidden_size=3584]");
    TORCH_CHECK(qkv_bias.dim() == 1 &&
                qkv_bias.size(0) == kQkvOutSize,
                "qkv_bias must be [qkv_out_size]");
    TORCH_CHECK(proj_weight.dim() == 2 &&
                proj_weight.size(0) == kHiddenSize &&
                proj_weight.size(1) == kHiddenSize,
                "proj_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(proj_bias.dim() == 1 &&
                proj_bias.size(0) == kHiddenSize,
                "proj_bias must be [hidden_size]");
    TORCH_CHECK(cos.dim() == 2 &&
                cos.size(0) == seq_len &&
                cos.size(1) == kHeadDim,
                "cos must be [total_seq_len, head_dim=128]");
    TORCH_CHECK(sin.dim() == 2 &&
                sin.size(0) == seq_len &&
                sin.size(1) == kHeadDim,
                "sin must be [total_seq_len, head_dim=128]");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.size(0) >= 2,
                "cu_seqlens must be 1D with at least 2 entries");

    TORCH_CHECK(grad_hidden_states.dim() == 2 &&
                grad_hidden_states.size(0) == seq_len &&
                grad_hidden_states.size(1) == kHiddenSize,
                "grad_hidden_states must be [total_seq_len, hidden_size]");
    TORCH_CHECK(grad_qkv_weight.dim() == 2 &&
                grad_qkv_weight.size(0) == kQkvOutSize &&
                grad_qkv_weight.size(1) == kHiddenSize,
                "grad_qkv_weight must be [qkv_out_size, hidden_size]");
    TORCH_CHECK(grad_qkv_bias.dim() == 1 &&
                grad_qkv_bias.size(0) == kQkvOutSize,
                "grad_qkv_bias must be [qkv_out_size]");
    TORCH_CHECK(grad_proj_weight.dim() == 2 &&
                grad_proj_weight.size(0) == kHiddenSize &&
                grad_proj_weight.size(1) == kHiddenSize,
                "grad_proj_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(grad_proj_bias.dim() == 1 &&
                grad_proj_bias.size(0) == kHiddenSize,
                "grad_proj_bias must be [hidden_size]");

    const float scaling_f = static_cast<float>(scaling);

    // ---------------------------------------------------------
    // Forward recomputation: QKV linear + unbind, 2D RoPE on
    // Q / K, then per-cu_seqlens-split attention.
    // ---------------------------------------------------------
    auto qkv = at::linear(hidden_states, qkv_weight, qkv_bias);
    auto qkv_view = qkv.reshape({seq_len, 3, kNumHeads, kHeadDim});
    auto qkv_permuted = qkv_view.permute({1, 0, 2, 3});
    auto qkv_unbound = qkv_permuted.unbind(/*dim=*/0);
    auto query_states = qkv_unbound[0];  // [seq_len, H, d]
    auto key_states = qkv_unbound[1];
    auto value_states = qkv_unbound[2];

    auto cos_expanded = cos.unsqueeze(-2);  // [seq_len, 1, d]
    auto sin_expanded = sin.unsqueeze(-2);

    auto query_rope = query_states * cos_expanded
                      + rotate_half(query_states) * sin_expanded;
    auto key_rope = key_states * cos_expanded
                    + rotate_half(key_states) * sin_expanded;

    // Reshape for attention: [seq_len, H, d] -> [1, H, seq_len, d]
    auto query_attn = query_rope.transpose(0, 1).unsqueeze(0);
    auto key_attn = key_rope.transpose(0, 1).unsqueeze(0);
    auto value_attn = value_states.transpose(0, 1).unsqueeze(0);

    // cu_seqlens host-sync to derive the per-sequence lengths
    // for split_with_sizes. Same shape as the reference's
    // lengths.tolist() call.
    auto cu_seqlens_cpu = cu_seqlens.to(torch::kCPU);
    const int64_t num_sequences = cu_seqlens_cpu.size(0) - 1;
    const int32_t* cs_ptr = cu_seqlens_cpu.data_ptr<int32_t>();
    std::vector<int64_t> lengths;
    lengths.reserve(num_sequences);
    for (int64_t i = 0; i < num_sequences; ++i) {
        lengths.push_back(
            static_cast<int64_t>(cs_ptr[i + 1] - cs_ptr[i]));
    }

    auto query_splits = query_attn.split_with_sizes(lengths, /*dim=*/2);
    auto key_splits = key_attn.split_with_sizes(lengths, /*dim=*/2);
    auto value_splits = value_attn.split_with_sizes(lengths, /*dim=*/2);

    std::vector<torch::Tensor> attn_outputs;
    std::vector<torch::Tensor> attn_weights_list;
    attn_outputs.reserve(num_sequences);
    attn_weights_list.reserve(num_sequences);

    for (int64_t i = 0; i < num_sequences; ++i) {
        const auto& q = query_splits[i];
        const auto& k = key_splits[i];
        const auto& v = value_splits[i];
        auto attn_scores = at::matmul(q, k.transpose(2, 3)) * scaling_f;
        auto attn_weights = at::softmax(attn_scores, /*dim=*/-1);
        auto attn_output = at::matmul(attn_weights, v);
        attn_outputs.push_back(attn_output);
        attn_weights_list.push_back(attn_weights);
    }

    auto attn_output_cat = at::cat(attn_outputs, /*dim=*/2);
    auto attn_output_reshaped = attn_output_cat.squeeze(0)
                                    .transpose(0, 1)
                                    .reshape({seq_len, kHiddenSize})
                                    .contiguous();

    // ---------------------------------------------------------
    // Output-projection backward.
    // ---------------------------------------------------------
    grad_proj_bias.copy_(grad_output.sum(/*dim=*/0));
    grad_proj_weight.copy_(
        at::matmul(grad_output.transpose(0, 1), attn_output_reshaped));
    auto grad_attn_output = at::matmul(grad_output, proj_weight);

    // Reshape: [seq_len, hidden] -> [seq_len, H, d]
    //        -> [1, H, seq_len, d]
    auto grad_attn_output_heads = grad_attn_output
                                      .reshape({seq_len, kNumHeads, kHeadDim})
                                      .transpose(0, 1)
                                      .unsqueeze(0);

    auto grad_attn_output_splits =
        grad_attn_output_heads.split_with_sizes(lengths, /*dim=*/2);

    // ---------------------------------------------------------
    // Per-sequence attention backward.
    // ---------------------------------------------------------
    std::vector<torch::Tensor> grad_query_splits;
    std::vector<torch::Tensor> grad_key_splits;
    std::vector<torch::Tensor> grad_value_splits;
    grad_query_splits.reserve(num_sequences);
    grad_key_splits.reserve(num_sequences);
    grad_value_splits.reserve(num_sequences);

    for (int64_t i = 0; i < num_sequences; ++i) {
        const auto& grad_attn_out = grad_attn_output_splits[i];
        const auto& attn_weights = attn_weights_list[i];
        const auto& q = query_splits[i];
        const auto& k = key_splits[i];
        const auto& v = value_splits[i];

        auto grad_attn_weights = at::matmul(
            grad_attn_out, v.transpose(2, 3));
        auto grad_v = at::matmul(
            attn_weights.transpose(2, 3), grad_attn_out);

        // Softmax backward.
        auto sum_grad = (grad_attn_weights * attn_weights)
                            .sum(/*dim=*/-1, /*keepdim=*/true);
        auto grad_attn_scores = attn_weights
                                * (grad_attn_weights - sum_grad);

        grad_attn_scores = grad_attn_scores * scaling_f;

        auto grad_q = at::matmul(grad_attn_scores, k);
        auto grad_k = at::matmul(grad_attn_scores.transpose(2, 3), q);

        grad_query_splits.push_back(grad_q);
        grad_key_splits.push_back(grad_k);
        grad_value_splits.push_back(grad_v);
    }

    auto grad_query_states = at::cat(grad_query_splits, /*dim=*/2)
                                 .squeeze(0)
                                 .transpose(0, 1);  // [seq_len, H, d]
    auto grad_key_states = at::cat(grad_key_splits, /*dim=*/2)
                               .squeeze(0)
                               .transpose(0, 1);
    auto grad_value_states = at::cat(grad_value_splits, /*dim=*/2)
                                 .squeeze(0)
                                 .transpose(0, 1);

    // ---------------------------------------------------------
    // 2D RoPE backward (rotate_half_inverse).
    // ---------------------------------------------------------
    auto grad_query_pre_rope =
        grad_query_states * cos_expanded
        + rotate_half_inverse(grad_query_states * sin_expanded);
    auto grad_key_pre_rope =
        grad_key_states * cos_expanded
        + rotate_half_inverse(grad_key_states * sin_expanded);
    // grad_value: V has no RoPE; pass through.

    // ---------------------------------------------------------
    // Stack into [seq_len, 3, H, d]; reshape to
    // [seq_len, qkv_out_size = 10752].
    // ---------------------------------------------------------
    auto grad_qkv_stacked = at::stack(
        {grad_query_pre_rope, grad_key_pre_rope, grad_value_states},
        /*dim=*/1);
    auto grad_qkv = grad_qkv_stacked.reshape({seq_len, kQkvOutSize});

    // ---------------------------------------------------------
    // QKV-projection backward.
    // ---------------------------------------------------------
    grad_hidden_states.copy_(at::matmul(grad_qkv, qkv_weight));
    grad_qkv_weight.copy_(
        at::matmul(grad_qkv.transpose(0, 1), hidden_states));
    grad_qkv_bias.copy_(grad_qkv.sum(/*dim=*/0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused vision cu_seqlens variable-length attention backward "
        "with 2D rotary position embeddings. Pure ATen / cuBLAS "
        "dispatch chain: four fp32 2D cublasSgemm calls (output and "
        "QKV projection grad_input via at::matmul(g, W) -- OP_N OP_N "
        "verified family; grad_weight via g.t() @ x -- OP_T OP_N "
        "verified family) + N 4D cublasSgemmStridedBatched calls per "
        "cu_seqlens-split sequence (attention-application backward + "
        "scaled-dot-product backward) + 2D RoPE backward via "
        "rotate_half_inverse + softmax backward and scale backward "
        "via ATen elementwise / reduction ops. Const axes: "
        "hidden_size=3584, num_heads=28, head_dim=128, "
        "qkv_out_size=10752. Destination-passing style; 5 gradient "
        "outputs.");
}
