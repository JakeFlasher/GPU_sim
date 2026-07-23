#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kBFloat16,
        name, " must have bfloat16 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. definition.json declares:
//   attn_weights : [batch_size, num_heads, seq_len_q, seq_len_k]  bf16
//   output       : [batch_size, num_heads, seq_len_q, seq_len_k]  bf16 (DPS)
//
// num_heads is constant at 8 per definition.json. softcap=30 is
// pinned. Algorithm:
//   softcapped = tanh(attn_weights / 30) * 30
//   output     = softmax(softcapped, dim=-1, fp32 upcast)
void run(
    const torch::Tensor& attn_weights,
    torch::Tensor output
) {
    TORCH_CHECK(attn_weights.dim() == 4,
                "attn_weights must be 4D [B, H, Q, K]; got ",
                attn_weights.dim(), "D");
    TORCH_CHECK(output.dim() == 4, "output must be 4D");
    TORCH_CHECK(attn_weights.sizes() == output.sizes(),
                "output shape must match attn_weights");

    const int64_t num_heads = attn_weights.size(1);
    TORCH_CHECK(num_heads == 8,
                "num_heads must be 8 for problem_046, got ", num_heads);

    check_bf16_contiguous_cuda(attn_weights, "attn_weights");
    check_bf16_contiguous_cuda(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_046_attention_softmax_softcap_launcher(
        attn_weights, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "BF16 attention softmax with Gemma3 logit softcapping "
        "(tanh-based ±30 clamp) and fp32-upcast numerical-stability "
        "softmax over the last (K) axis. One CTA per (B, H, Q) row. "
        "Destination-passing style.");
}
