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
//   grad_output : [num_elements]  fp32
//   x           : [num_elements]  fp32
//   sigmoid_x   : [num_elements]  fp32
//   grad_input  : [num_elements]  fp32 (DPS, pre-alloc)
//
// All four tensors share the same flat 1D shape; num_elements is
// variable per workload. Algorithm:
//   grad_input = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))
void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& sigmoid_x,
    torch::Tensor grad_input
) {
    TORCH_CHECK(grad_output.dim() == 1, "grad_output must be 1D");
    TORCH_CHECK(x.dim() == 1, "x must be 1D");
    TORCH_CHECK(sigmoid_x.dim() == 1, "sigmoid_x must be 1D");
    TORCH_CHECK(grad_input.dim() == 1, "grad_input must be 1D");

    const int64_t n = grad_output.numel();
    TORCH_CHECK(x.numel() == n, "x must have same numel as grad_output");
    TORCH_CHECK(sigmoid_x.numel() == n, "sigmoid_x must have same numel as grad_output");
    TORCH_CHECK(grad_input.numel() == n, "grad_input must have same numel as grad_output");

    check_fp32_contiguous_cuda(grad_output, "grad_output");
    check_fp32_contiguous_cuda(x, "x");
    check_fp32_contiguous_cuda(sigmoid_x, "sigmoid_x");
    check_fp32_contiguous_cuda(grad_input, "grad_input");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_084_silu_activation_backward_launcher(
        grad_output, x, sigmoid_x, grad_input, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "FP32 SiLU backward: grad_input = grad_output * sigmoid_x * "
        "(1 + x * (1 - sigmoid_x)). Destination-passing style; "
        "float4-vectorized load/store with scalar tail fallback.");
}
