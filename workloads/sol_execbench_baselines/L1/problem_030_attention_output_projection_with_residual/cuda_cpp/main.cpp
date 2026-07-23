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

// Destination-passing-style entry. Per definition.json:
//   attn_output    : [batch_size, seq_len, hidden_size]   bf16
//   residual       : [batch_size, seq_len, hidden_size]   bf16
//   o_proj_weight  : [hidden_size, hidden_size]           bf16
//   output (DPS)   : [batch_size, seq_len, hidden_size]   bf16
//
// hidden_size is constant at 2560 per definition.json (Jamba-3B).
// Algorithm: output = (attn_output @ o_proj_weight.T) + residual.
void run(
    const torch::Tensor& attn_output,
    const torch::Tensor& residual,
    const torch::Tensor& o_proj_weight,
    torch::Tensor output
) {
    TORCH_CHECK(attn_output.dim() == 3,
                "attn_output must be 3D [B, S, H]; got ",
                attn_output.dim(), "D");
    TORCH_CHECK(residual.dim() == 3, "residual must be 3D");
    TORCH_CHECK(o_proj_weight.dim() == 2,
                "o_proj_weight must be 2D [H, H]");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");

    const int64_t hidden_size = attn_output.size(2);
    TORCH_CHECK(hidden_size == 2560,
                "hidden_size must be 2560 for problem_030, got ",
                hidden_size);
    TORCH_CHECK(o_proj_weight.size(0) == hidden_size &&
                o_proj_weight.size(1) == hidden_size,
                "o_proj_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(residual.sizes() == attn_output.sizes(),
                "residual must share shape with attn_output");
    TORCH_CHECK(output.sizes() == attn_output.sizes(),
                "output must share shape with attn_output");

    check_bf16_contiguous_cuda(attn_output, "attn_output");
    check_bf16_contiguous_cuda(residual, "residual");
    check_bf16_contiguous_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_contiguous_cuda(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_030_attn_output_proj_residual_launcher(
        attn_output, residual, o_proj_weight, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "BF16 fused attention output projection (`(attn @ "
        "o_proj_weight.T) + residual`) via CUTLASS TensorOp on "
        "Sm80; residual passed as the GEMM's C source so the "
        "epilogue adds it in-place; destination-passing style.");
}
