#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

// Destination-passing-style entry for the masked-softmax + dropout
// backward kernel. Per definition.json:
//   grad_output   : [batch_size, n_heads, seq_len, seq_len] fp32
//   p_attn        : [batch_size, n_heads, seq_len, seq_len] fp32
//   mask          : [batch_size, 1, seq_len, seq_len] bool
//   dropout_mask  : [batch_size, n_heads, seq_len, seq_len] bool
//   p_dropout     : scalar fp32 (passed as double, matching reference)
//   grad_scores   : [batch_size, n_heads, seq_len, seq_len] fp32 (DPS)
//
// No const axes pinned by definition.json — runtime checks reduce
// to cross-tensor shape coherence, following the runtime-check
// audit discipline.

void run(
    torch::Tensor grad_output,
    torch::Tensor p_attn,
    torch::Tensor mask,
    torch::Tensor dropout_mask,
    double p_dropout,
    torch::Tensor grad_scores
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(p_attn.is_cuda(), "p_attn must be CUDA");
    TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
    TORCH_CHECK(dropout_mask.is_cuda(), "dropout_mask must be CUDA");
    TORCH_CHECK(grad_scores.is_cuda(), "grad_scores must be CUDA");

    TORCH_CHECK(grad_output.dtype() == torch::kFloat32,
                "grad_output must be float32");
    TORCH_CHECK(p_attn.dtype() == torch::kFloat32,
                "p_attn must be float32");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");
    TORCH_CHECK(dropout_mask.dtype() == torch::kBool,
                "dropout_mask must be bool");
    TORCH_CHECK(grad_scores.dtype() == torch::kFloat32,
                "grad_scores must be float32");

    TORCH_CHECK(grad_output.is_contiguous(),
                "grad_output must be contiguous");
    TORCH_CHECK(p_attn.is_contiguous(), "p_attn must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(dropout_mask.is_contiguous(),
                "dropout_mask must be contiguous");
    TORCH_CHECK(grad_scores.is_contiguous(),
                "grad_scores must be contiguous");

    TORCH_CHECK(grad_output.dim() == 4,
                "grad_output must be 4D [B, H, T, T]");
    TORCH_CHECK(p_attn.dim() == 4, "p_attn must be 4D");
    TORCH_CHECK(mask.dim() == 4, "mask must be 4D [B, 1, T, T]");
    TORCH_CHECK(dropout_mask.dim() == 4, "dropout_mask must be 4D");
    TORCH_CHECK(grad_scores.dim() == 4, "grad_scores must be 4D");

    const int64_t B = grad_output.size(0);
    const int64_t H = grad_output.size(1);
    const int64_t T1 = grad_output.size(2);
    const int64_t T2 = grad_output.size(3);
    TORCH_CHECK(T1 == T2, "grad_output last two dims must be equal "
                "(both = seq_len); got ", T1, " vs ", T2);

    TORCH_CHECK(p_attn.sizes() == grad_output.sizes(),
                "p_attn must have the same shape as grad_output");
    TORCH_CHECK(dropout_mask.sizes() == grad_output.sizes(),
                "dropout_mask must have the same shape as grad_output");
    TORCH_CHECK(grad_scores.sizes() == grad_output.sizes(),
                "grad_scores must have the same shape as grad_output");
    TORCH_CHECK(mask.size(0) == B && mask.size(1) == 1 &&
                mask.size(2) == T1 && mask.size(3) == T1,
                "mask must be [B, 1, T, T] matching grad_output's "
                "B and T");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_066_masked_softmax_with_attention_dropout_backward_launcher(
        grad_output, p_attn, mask, dropout_mask, p_dropout,
        grad_scores, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 masked-softmax + dropout backward: per-row CTA "
        "(Grid (B, H, T)) with warp+block reduction over T, then "
        "per-element write with mask gating. Destination-passing "
        "style; grad_scores pre-allocated.");
}
