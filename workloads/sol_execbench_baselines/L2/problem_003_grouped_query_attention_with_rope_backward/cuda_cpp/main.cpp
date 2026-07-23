#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// rotate_half_inverse(x) = cat([x_second_half, -x_first_half], dim=-1).
// Used in the RoPE backward (gradient through rotate_half).
static torch::Tensor rotate_half_inverse(const torch::Tensor& x) {
    const int64_t last_dim = x.size(-1);
    const int64_t half = last_dim / 2;
    auto x1 = x.slice(-1, 0, half);
    auto x2 = x.slice(-1, half, last_dim);
    return at::cat({x2, -x1}, /*dim=*/-1);
}

// Destination-passing-style entry for the GQA with RoPE backward.
// Per definition.json:
//   grad_output         : [B, S, hidden_size=5120]                bf16
//   hidden_states       : [B, S, hidden_size]                     bf16
//   cos                 : [B, S, head_dim=128]                    bf16
//   sin                 : [B, S, head_dim]                        bf16
//   q_weight            : [q_proj_size=4096, hidden_size]         bf16
//   k_weight            : [kv_proj_size=1024, hidden_size]        bf16
//   v_weight            : [kv_proj_size, hidden_size]             bf16
//   o_weight            : [hidden_size, q_proj_size]              bf16
//   query_states        : [B, num_attention_heads=32, S, head_dim] bf16
//   key_states          : [B, num_attention_heads, S, head_dim]   bf16  (post-repeat_kv)
//   value_states        : [B, num_attention_heads, S, head_dim]   bf16  (post-repeat_kv)
//   attn_weights        : [B, num_attention_heads, S, S]          bf16
//   attn_output         : [B, S, q_proj_size]                     bf16
//   scaling             : scalar fp32
//   grad_hidden_states  : [B, S, hidden_size]                     bf16 DPS
//   grad_q_weight       : [q_proj_size, hidden_size]              bf16 DPS
//   grad_k_weight       : [kv_proj_size, hidden_size]             bf16 DPS
//   grad_v_weight       : [kv_proj_size, hidden_size]             bf16 DPS
//   grad_o_weight       : [hidden_size, q_proj_size]              bf16 DPS
//
// Algorithm (mirrors the reference exactly):
//   Output projection backward:
//     grad_attn_output = at::matmul(grad_output, o_weight)
//                        (2D bf16 OP_N OP_N -- secondary 2D bf16
//                         verified family surface).
//     grad_o_weight = grad_output_2d.t() @ attn_output_2d
//                     (2D bf16 OP_T OP_N).
//   Reshape grad_attn_output to [B, num_attention_heads, S, head_dim].
//   Attention-application backward (4D bf16 batched -- the
//   dominant verified cublasGemmStridedBatchedEx_4D_bf16
//   family witness):
//     grad_attn_weights = matmul(grad_attn_output,
//                                value_states.transpose(2, 3))
//                         (OP_N OP_T).
//     grad_value_states = matmul(attn_weights.transpose(2, 3),
//                                grad_attn_output)
//                         (OP_T OP_N).
//   Softmax backward (in fp32, cast back to bf16):
//     sum_grad = (grad_attn_weights * attn_weights).sum(-1, keepdim).
//     grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad).
//   Scale backward: grad_attn_scores *= scaling.
//   Scaled-dot-product backward (4D bf16 batched):
//     grad_query_states = matmul(grad_attn_scores, key_states)
//                         (OP_N OP_N -- canonical verified family
//                          binding).
//     grad_key_states = matmul(grad_attn_scores.transpose(2, 3),
//                              query_states)
//                       (OP_T OP_N).
//   GQA repeat reverse-sum:
//     grad_key_states.reshape(B, num_kv_heads, num_kv_groups,
//                             S, head_dim).sum(dim=2)
//     -> [B, num_kv_heads, S, head_dim].
//     Analogous for grad_value_states.
//   RoPE backward via rotate_half_inverse:
//     grad_query_states_pre_rope = grad_query_states * cos_expanded
//       + rotate_half_inverse(grad_query_states * sin_expanded).
//     Analogous for K. V passes through.
//   Transpose + reshape back to [B, S, *_proj_size]:
//     grad_query_proj = grad_query_states_pre_rope
//                         .transpose(1, 2).contiguous()
//                         .view(B, S, q_proj_size).
//     Analogous for K and V (kv_proj_size).
//   QKV projection backward (3 grad_input via at::matmul(g, W)
//   OP_N OP_N + 3 grad_weight via g.t() @ x OP_T OP_N).
//   grad_hidden_states accumulates Q + K + V branches.
void run(
    torch::Tensor grad_output,
    torch::Tensor hidden_states,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor o_weight,
    torch::Tensor query_states,
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor attn_weights,
    torch::Tensor attn_output,
    double scaling,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_q_weight,
    torch::Tensor grad_k_weight,
    torch::Tensor grad_v_weight,
    torch::Tensor grad_o_weight
) {
    check_bf16_contiguous_cuda(grad_output, "grad_output");
    check_bf16_contiguous_cuda(hidden_states, "hidden_states");
    check_bf16_contiguous_cuda(cos, "cos");
    check_bf16_contiguous_cuda(sin, "sin");
    check_bf16_contiguous_cuda(q_weight, "q_weight");
    check_bf16_contiguous_cuda(k_weight, "k_weight");
    check_bf16_contiguous_cuda(v_weight, "v_weight");
    check_bf16_contiguous_cuda(o_weight, "o_weight");
    check_bf16_contiguous_cuda(query_states, "query_states");
    check_bf16_contiguous_cuda(key_states, "key_states");
    check_bf16_contiguous_cuda(value_states, "value_states");
    check_bf16_contiguous_cuda(attn_weights, "attn_weights");
    check_bf16_contiguous_cuda(attn_output, "attn_output");
    check_bf16_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_bf16_contiguous_cuda(grad_q_weight, "grad_q_weight");
    check_bf16_contiguous_cuda(grad_k_weight, "grad_k_weight");
    check_bf16_contiguous_cuda(grad_v_weight, "grad_v_weight");
    check_bf16_contiguous_cuda(grad_o_weight, "grad_o_weight");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 5120;
    constexpr int64_t kNumAttentionHeads = 32;
    constexpr int64_t kNumKeyValueHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKeyValueGroups =
        kNumAttentionHeads / kNumKeyValueHeads;  // 4
    constexpr int64_t kQProjSize = kNumAttentionHeads * kHeadDim;  // 4096
    constexpr int64_t kKvProjSize = kNumKeyValueHeads * kHeadDim;  // 1024

    TORCH_CHECK(hidden_states.dim() == 3 &&
                hidden_states.size(2) == kHiddenSize,
                "hidden_states must be [B, S, hidden_size=5120]");
    const int64_t B = hidden_states.size(0);
    const int64_t S = hidden_states.size(1);

    TORCH_CHECK(grad_output.dim() == 3 &&
                grad_output.size(0) == B &&
                grad_output.size(1) == S &&
                grad_output.size(2) == kHiddenSize,
                "grad_output must be [B, S, hidden_size]");
    TORCH_CHECK(cos.dim() == 3 &&
                cos.size(0) == B &&
                cos.size(1) == S &&
                cos.size(2) == kHeadDim,
                "cos must be [B, S, head_dim]");
    TORCH_CHECK(sin.dim() == 3 &&
                sin.size(0) == B &&
                sin.size(1) == S &&
                sin.size(2) == kHeadDim,
                "sin must be [B, S, head_dim]");
    TORCH_CHECK(q_weight.dim() == 2 &&
                q_weight.size(0) == kQProjSize &&
                q_weight.size(1) == kHiddenSize,
                "q_weight must be [q_proj_size=4096, hidden_size]");
    TORCH_CHECK(k_weight.dim() == 2 &&
                k_weight.size(0) == kKvProjSize &&
                k_weight.size(1) == kHiddenSize,
                "k_weight must be [kv_proj_size=1024, hidden_size]");
    TORCH_CHECK(v_weight.dim() == 2 &&
                v_weight.size(0) == kKvProjSize &&
                v_weight.size(1) == kHiddenSize,
                "v_weight must be [kv_proj_size, hidden_size]");
    TORCH_CHECK(o_weight.dim() == 2 &&
                o_weight.size(0) == kHiddenSize &&
                o_weight.size(1) == kQProjSize,
                "o_weight must be [hidden_size, q_proj_size]");
    TORCH_CHECK(query_states.dim() == 4 &&
                query_states.size(0) == B &&
                query_states.size(1) == kNumAttentionHeads &&
                query_states.size(2) == S &&
                query_states.size(3) == kHeadDim,
                "query_states must be [B, num_attention_heads, S, head_dim]");
    TORCH_CHECK(key_states.dim() == 4 &&
                key_states.size(0) == B &&
                key_states.size(1) == kNumAttentionHeads &&
                key_states.size(2) == S &&
                key_states.size(3) == kHeadDim,
                "key_states must be post-repeat_kv [B, num_attention_heads, S, head_dim]");
    TORCH_CHECK(value_states.dim() == 4 &&
                value_states.size(0) == B &&
                value_states.size(1) == kNumAttentionHeads &&
                value_states.size(2) == S &&
                value_states.size(3) == kHeadDim,
                "value_states must be post-repeat_kv [B, num_attention_heads, S, head_dim]");
    TORCH_CHECK(attn_weights.dim() == 4 &&
                attn_weights.size(0) == B &&
                attn_weights.size(1) == kNumAttentionHeads &&
                attn_weights.size(2) == S &&
                attn_weights.size(3) == S,
                "attn_weights must be [B, num_attention_heads, S, S]");
    TORCH_CHECK(attn_output.dim() == 3 &&
                attn_output.size(0) == B &&
                attn_output.size(1) == S &&
                attn_output.size(2) == kQProjSize,
                "attn_output must be [B, S, q_proj_size]");

    TORCH_CHECK(grad_hidden_states.dim() == 3 &&
                grad_hidden_states.size(0) == B &&
                grad_hidden_states.size(1) == S &&
                grad_hidden_states.size(2) == kHiddenSize,
                "grad_hidden_states must be [B, S, hidden_size]");
    TORCH_CHECK(grad_q_weight.dim() == 2 &&
                grad_q_weight.size(0) == kQProjSize &&
                grad_q_weight.size(1) == kHiddenSize,
                "grad_q_weight must be [q_proj_size, hidden_size]");
    TORCH_CHECK(grad_k_weight.dim() == 2 &&
                grad_k_weight.size(0) == kKvProjSize &&
                grad_k_weight.size(1) == kHiddenSize,
                "grad_k_weight must be [kv_proj_size, hidden_size]");
    TORCH_CHECK(grad_v_weight.dim() == 2 &&
                grad_v_weight.size(0) == kKvProjSize &&
                grad_v_weight.size(1) == kHiddenSize,
                "grad_v_weight must be [kv_proj_size, hidden_size]");
    TORCH_CHECK(grad_o_weight.dim() == 2 &&
                grad_o_weight.size(0) == kHiddenSize &&
                grad_o_weight.size(1) == kQProjSize,
                "grad_o_weight must be [hidden_size, q_proj_size]");

    const float scaling_f = static_cast<float>(scaling);

    // ----------------------------------------------------------------
    // Output projection backward.
    // grad_attn_output = grad_output @ o_weight (2D bf16 OP_N OP_N).
    // grad_o_weight = grad_output_2d.t() @ attn_output_2d
    //                 (2D bf16 OP_T OP_N).
    // ----------------------------------------------------------------
    auto grad_attn_output_2d_flat = at::matmul(grad_output, o_weight);

    auto grad_output_2d = grad_output.reshape({-1, kHiddenSize});
    auto attn_output_2d = attn_output.reshape({-1, kQProjSize});
    grad_o_weight.copy_(
        at::matmul(grad_output_2d.transpose(0, 1), attn_output_2d));

    // ----------------------------------------------------------------
    // Reshape to [B, num_attention_heads, S, head_dim].
    // ----------------------------------------------------------------
    auto grad_attn_output_heads = grad_attn_output_2d_flat
                                      .reshape({B, S,
                                                kNumAttentionHeads,
                                                kHeadDim})
                                      .transpose(1, 2);

    // ----------------------------------------------------------------
    // Attention-application backward (4D bf16 cublasGemmStridedBatchedEx
    // -- the verified family witness).
    // ----------------------------------------------------------------
    auto grad_attn_weights = at::matmul(
        grad_attn_output_heads, value_states.transpose(2, 3));
    auto grad_value_states = at::matmul(
        attn_weights.transpose(2, 3), grad_attn_output_heads);

    // ----------------------------------------------------------------
    // Softmax backward (in fp32 for numerical stability, cast back
    // to bf16 for the downstream matmul backwards).
    // ----------------------------------------------------------------
    auto attn_weights_fp32 = attn_weights.to(torch::kFloat32);
    auto grad_attn_weights_fp32 = grad_attn_weights.to(torch::kFloat32);
    auto sum_grad = (grad_attn_weights_fp32 * attn_weights_fp32)
                        .sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_attn_scores_fp32 = attn_weights_fp32
                                 * (grad_attn_weights_fp32 - sum_grad);
    auto grad_attn_scores = grad_attn_scores_fp32
                                .to(torch::kBFloat16) * scaling_f;

    // ----------------------------------------------------------------
    // Scaled-dot-product backward (4D bf16 cublasGemmStridedBatchedEx
    // -- canonical OP_N OP_N + OP_T OP_N verified family bindings).
    // ----------------------------------------------------------------
    auto grad_query_states = at::matmul(grad_attn_scores, key_states);
    auto grad_key_states = at::matmul(grad_attn_scores.transpose(2, 3),
                                      query_states);

    // ----------------------------------------------------------------
    // GQA repeat reverse-sum: collapse num_attention_heads back to
    // num_key_value_heads by summing over the repeat group dim.
    // ----------------------------------------------------------------
    auto grad_key_states_kv = grad_key_states
                                  .reshape({B, kNumKeyValueHeads,
                                            kNumKeyValueGroups,
                                            S, kHeadDim})
                                  .sum(/*dim=*/2);
    auto grad_value_states_kv = grad_value_states
                                    .reshape({B, kNumKeyValueHeads,
                                              kNumKeyValueGroups,
                                              S, kHeadDim})
                                    .sum(/*dim=*/2);

    // ----------------------------------------------------------------
    // RoPE backward via rotate_half_inverse pattern.
    // ----------------------------------------------------------------
    auto cos_expanded = cos.unsqueeze(1);  // [B, 1, S, head_dim]
    auto sin_expanded = sin.unsqueeze(1);

    auto grad_query_states_pre_rope = grad_query_states * cos_expanded
        + rotate_half_inverse(grad_query_states * sin_expanded);
    auto grad_key_states_pre_rope = grad_key_states_kv * cos_expanded
        + rotate_half_inverse(grad_key_states_kv * sin_expanded);
    // grad_value: V has no RoPE; pass through.

    // ----------------------------------------------------------------
    // Transpose + reshape back to [B, S, *_proj_size].
    // ----------------------------------------------------------------
    auto grad_query_proj = grad_query_states_pre_rope
                               .transpose(1, 2).contiguous()
                               .view({B, S, kQProjSize});
    auto grad_key_proj = grad_key_states_pre_rope
                             .transpose(1, 2).contiguous()
                             .view({B, S, kKvProjSize});
    auto grad_value_proj = grad_value_states_kv
                               .transpose(1, 2).contiguous()
                               .view({B, S, kKvProjSize});

    // ----------------------------------------------------------------
    // Q/K/V projection backwards (3 grad_input via at::matmul(g, W)
    // OP_N OP_N -- 2D bf16 verified family secondary surface;
    // 3 grad_weight via g.t() @ x OP_T OP_N).
    // ----------------------------------------------------------------
    auto grad_hidden_from_q = at::matmul(grad_query_proj, q_weight);
    auto grad_hidden_from_k = at::matmul(grad_key_proj, k_weight);
    auto grad_hidden_from_v = at::matmul(grad_value_proj, v_weight);
    grad_hidden_states.copy_(
        grad_hidden_from_q + grad_hidden_from_k + grad_hidden_from_v);

    auto grad_query_proj_2d = grad_query_proj.reshape({-1, kQProjSize});
    auto grad_key_proj_2d = grad_key_proj.reshape({-1, kKvProjSize});
    auto grad_value_proj_2d = grad_value_proj.reshape({-1, kKvProjSize});
    auto hidden_states_2d = hidden_states.reshape({-1, kHiddenSize});

    grad_q_weight.copy_(
        at::matmul(grad_query_proj_2d.transpose(0, 1), hidden_states_2d));
    grad_k_weight.copy_(
        at::matmul(grad_key_proj_2d.transpose(0, 1), hidden_states_2d));
    grad_v_weight.copy_(
        at::matmul(grad_value_proj_2d.transpose(0, 1), hidden_states_2d));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Grouped query attention with RoPE backward. Pure ATen / "
        "cuBLAS dispatch chain: four 4D bf16 cublasGemmStridedBatchedEx "
        "dispatches (attention-application backward and "
        "scaled-dot-product backward; the verified "
        "cublasGemmStridedBatchedEx_4D_bf16_OP_N_OP_N family) + four "
        "2D bf16 cublasGemmEx projection backward dispatch pairs "
        "(O + Q/K/V grad_input via at::matmul(g, W) OP_N OP_N; "
        "grad_weight via g.t() @ x OP_T OP_N) + softmax backward "
        "(fp32 math) + RoPE backward via rotate_half_inverse + GQA "
        "repeat reverse-sum. Const axes: hidden_size=5120, "
        "num_attention_heads=32, num_key_value_heads=8, head_dim=128, "
        "num_key_value_groups=4, q_proj_size=4096, kv_proj_size=1024. "
        "Destination-passing style; 5 gradient outputs.");
}
