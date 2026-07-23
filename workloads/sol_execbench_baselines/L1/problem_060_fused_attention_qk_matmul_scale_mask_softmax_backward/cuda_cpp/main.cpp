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

// Destination-passing-style entry for the fused attention QK
// matmul + scale + mask + softmax backward. Per definition.json:
//   grad_output      : [B, num_heads=128, seq_len_q, seq_len_k] bf16
//   query            : [B, num_heads, seq_len_q, head_dim=128]  bf16
//   key              : [B, num_heads, seq_len_k, head_dim]      bf16
//   attn_weights     : [B, num_heads, seq_len_q, seq_len_k]     bf16
//   scaling          : scalar fp32 (1/sqrt(head_dim))
//   grad_query       : [B, num_heads, seq_len_q, head_dim]      bf16 DPS
//   grad_key         : [B, num_heads, seq_len_k, head_dim]      bf16 DPS
//
// Algorithm (mirrors the reference exactly):
//   Softmax backward (numerically-stable formulation, fp32 math):
//     sum_grad = sum(grad_output_f32 * attn_weights_f32, -1, keepdim)
//     grad_attn_logits = attn_weights_f32 * (grad_output_f32 - sum_grad)
//   Cast back to bf16 for the matmul backwards.
//   Scale backward: grad_scaled_logits = grad_attn_logits * scaling.
//   Q @ K^T matmul backward:
//     grad_query = grad_scaled_logits @ key
//                  (cublasGemmStridedBatchedEx 4D bf16 OP_N OP_N --
//                   the verified family canonical binding).
//     grad_key = grad_scaled_logits.transpose(-2, -1) @ query
//                (cublasGemmStridedBatchedEx 4D bf16 OP_T OP_N --
//                 same family).
void run(
    torch::Tensor grad_output,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor attn_weights,
    double scaling,
    torch::Tensor grad_query,
    torch::Tensor grad_key
) {
    check_bf16_contiguous_cuda(grad_output, "grad_output");
    check_bf16_contiguous_cuda(query, "query");
    check_bf16_contiguous_cuda(key, "key");
    check_bf16_contiguous_cuda(attn_weights, "attn_weights");
    check_bf16_contiguous_cuda(grad_query, "grad_query");
    check_bf16_contiguous_cuda(grad_key, "grad_key");

    const c10::cuda::CUDAGuard device_guard(query.device());

    constexpr int64_t kNumHeads = 128;
    constexpr int64_t kHeadDim = 128;

    TORCH_CHECK(query.dim() == 4 &&
                query.size(1) == kNumHeads &&
                query.size(3) == kHeadDim,
                "query must be [B, num_heads=128, seq_len_q, head_dim=128]");
    const int64_t B = query.size(0);
    const int64_t seq_len_q = query.size(2);

    TORCH_CHECK(key.dim() == 4 &&
                key.size(0) == B &&
                key.size(1) == kNumHeads &&
                key.size(3) == kHeadDim,
                "key must be [B, num_heads, seq_len_k, head_dim]");
    const int64_t seq_len_k = key.size(2);

    TORCH_CHECK(grad_output.dim() == 4 &&
                grad_output.size(0) == B &&
                grad_output.size(1) == kNumHeads &&
                grad_output.size(2) == seq_len_q &&
                grad_output.size(3) == seq_len_k,
                "grad_output must be [B, num_heads, seq_len_q, seq_len_k]");
    TORCH_CHECK(attn_weights.dim() == 4 &&
                attn_weights.size(0) == B &&
                attn_weights.size(1) == kNumHeads &&
                attn_weights.size(2) == seq_len_q &&
                attn_weights.size(3) == seq_len_k,
                "attn_weights must be [B, num_heads, seq_len_q, seq_len_k]");

    TORCH_CHECK(grad_query.dim() == 4 &&
                grad_query.size(0) == B &&
                grad_query.size(1) == kNumHeads &&
                grad_query.size(2) == seq_len_q &&
                grad_query.size(3) == kHeadDim,
                "grad_query must be [B, num_heads, seq_len_q, head_dim]");
    TORCH_CHECK(grad_key.dim() == 4 &&
                grad_key.size(0) == B &&
                grad_key.size(1) == kNumHeads &&
                grad_key.size(2) == seq_len_k &&
                grad_key.size(3) == kHeadDim,
                "grad_key must be [B, num_heads, seq_len_k, head_dim]");

    const float scaling_f = static_cast<float>(scaling);

    // ----------------------------------------------------------------
    // Softmax backward (numerically-stable, fp32 math; cast back to
    // bf16 for the downstream matmul backwards).
    //   d(softmax)/d(logits) = softmax * (grad_out
    //                                     - sum(grad_out * softmax))
    // ----------------------------------------------------------------
    auto grad_output_f32 = grad_output.to(torch::kFloat32);
    auto attn_weights_f32 = attn_weights.to(torch::kFloat32);

    auto sum_grad = (grad_output_f32 * attn_weights_f32)
                        .sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_attn_logits_f32 = attn_weights_f32
                                * (grad_output_f32 - sum_grad);
    auto grad_attn_logits = grad_attn_logits_f32.to(torch::kBFloat16);

    // ----------------------------------------------------------------
    // Scale backward: grad_scaled_logits = grad_attn_logits * scaling.
    // ----------------------------------------------------------------
    auto grad_scaled_logits = grad_attn_logits * scaling_f;

    // ----------------------------------------------------------------
    // Q @ K^T matmul backward.
    // grad_query = grad_scaled_logits @ key
    //              (4D bf16 cublasGemmStridedBatchedEx, OP_N OP_N --
    //               the verified family canonical binding).
    // grad_key = grad_scaled_logits.transpose(-2, -1) @ query
    //            (4D bf16 cublasGemmStridedBatchedEx, OP_T OP_N --
    //             same verified family).
    // ----------------------------------------------------------------
    grad_query.copy_(at::matmul(grad_scaled_logits, key));
    grad_key.copy_(at::matmul(grad_scaled_logits.transpose(-2, -1),
                              query));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused attention QK matmul + scale + mask + softmax backward. "
        "Pure ATen / cuBLAS dispatch chain: two 4D bf16 "
        "cublasGemmStridedBatchedEx dispatches (grad_query via "
        "at::matmul(g, K) OP_N OP_N -- the verified "
        "cublasGemmStridedBatchedEx_4D_bf16_OP_N_OP_N family "
        "canonical binding; grad_key via g.transpose(-2, -1) @ Q "
        "OP_T OP_N -- same family) + softmax backward (in fp32, "
        "cast back to bf16) + scale backward via ATen elementwise "
        "/ reduction ops. Const axes: num_heads=128, head_dim=128. "
        "Destination-passing style; 2 gradient outputs.");
}
