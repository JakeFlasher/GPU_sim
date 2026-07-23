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
//   sublayer_output : [batch_size, seq_len, hidden_size]  bf16
//   residual        : [batch_size, seq_len, hidden_size]  bf16
//   weight          : [hidden_size]                       bf16
//   eps             : fp32 scalar (Python `float`)
//   output (DPS)    : [batch_size, seq_len, hidden_size]  bf16
//
// hidden_size is constant at 4096 per definition.json (Olmo3).
// Algorithm: output = residual + RMSNorm(sublayer_output, weight, eps),
// with the bf16 → fp32 → bf16 cast inside the RMSNorm and the
// residual add applied in bf16 after the downcast.
void run(
    const torch::Tensor& sublayer_output,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps,
    torch::Tensor output
) {
    TORCH_CHECK(sublayer_output.dim() == 3,
                "sublayer_output must be 3D [B, S, H]");
    TORCH_CHECK(residual.dim() == 3, "residual must be 3D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");

    const int64_t hidden_size = sublayer_output.size(2);
    TORCH_CHECK(hidden_size == 4096,
                "hidden_size must be 4096 for problem_033, got ",
                hidden_size);
    TORCH_CHECK(weight.size(0) == hidden_size,
                "weight must have size hidden_size");
    TORCH_CHECK(residual.sizes() == sublayer_output.sizes(),
                "residual must share shape with sublayer_output");
    TORCH_CHECK(output.sizes() == sublayer_output.sizes(),
                "output must share shape with sublayer_output");

    check_bf16_contiguous_cuda(sublayer_output, "sublayer_output");
    check_bf16_contiguous_cuda(residual, "residual");
    check_bf16_contiguous_cuda(weight, "weight");
    check_bf16_contiguous_cuda(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_033_post_norm_residual_launcher(
        sublayer_output, residual, weight,
        static_cast<float>(eps), output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "BF16 post-norm residual: output = residual + "
        "RMSNorm(sublayer_output, weight, eps). RMSNorm runs in "
        "fp32 (bf16 -> fp32 -> bf16), residual add in bf16. "
        "Destination-passing style.");
}
