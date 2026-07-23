#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat32,
        name, " must have float32 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. definition.json declares:
//   x      : [batch_size, seq_len, inner_dim*2]  fp32
//   output : [batch_size, seq_len, inner_dim]    fp32 (DPS)
//
// inner_dim is constant at 5120 per definition.json; batch_size and
// seq_len are variable. The last axis of x is split into two equal
// halves (x_gate, x_linear); output is GELU_tanh(x_gate) * x_linear.
void run(
    const torch::Tensor& x,
    torch::Tensor output
) {
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, seq, inner_dim*2]");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");

    const int64_t inner_dim = output.size(2);
    TORCH_CHECK(
        inner_dim == 5120,
        "inner_dim must be 5120 for problem_085, got ", inner_dim);
    TORCH_CHECK(
        x.size(2) == 2 * inner_dim,
        "x last dim must be 2*inner_dim=", 2 * inner_dim,
        ", got ", x.size(2));
    TORCH_CHECK(
        x.size(0) == output.size(0) && x.size(1) == output.size(1),
        "x and output must share batch and seq dims");

    check_fp32_contiguous_cuda(x, "x");
    check_fp32_contiguous_cuda(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_085_geglu_activation_launcher(x, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "FP32 GEGLU activation: GELU_tanh(x_gate) * x_linear. "
        "Destination-passing style; float4-vectorized load/store "
        "along the contiguous last dim.");
}
