#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry. Per definition.json input order is
// grad_output, x, gate_output, up_output, activated_gate,
// gate_weight, up_weight, down_weight, followed by output args
// grad_x, grad_gate_weight, grad_up_weight, grad_down_weight.

static void check_bf16_cuda_contiguous(
    const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ",
        t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void run(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor gate_output,
    torch::Tensor up_output,
    torch::Tensor activated_gate,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight,
    torch::Tensor grad_x,
    torch::Tensor grad_gate_weight,
    torch::Tensor grad_up_weight,
    torch::Tensor grad_down_weight) {

    check_bf16_cuda_contiguous(grad_output, "grad_output");
    check_bf16_cuda_contiguous(x, "x");
    check_bf16_cuda_contiguous(gate_output, "gate_output");
    check_bf16_cuda_contiguous(up_output, "up_output");
    check_bf16_cuda_contiguous(activated_gate, "activated_gate");
    check_bf16_cuda_contiguous(gate_weight, "gate_weight");
    check_bf16_cuda_contiguous(up_weight, "up_weight");
    check_bf16_cuda_contiguous(down_weight, "down_weight");
    check_bf16_cuda_contiguous(grad_x, "grad_x");
    check_bf16_cuda_contiguous(grad_gate_weight, "grad_gate_weight");
    check_bf16_cuda_contiguous(grad_up_weight, "grad_up_weight");
    check_bf16_cuda_contiguous(grad_down_weight, "grad_down_weight");

    TORCH_CHECK(grad_output.dim() == 3, "grad_output must be 3D");
    TORCH_CHECK(
        grad_output.size(2) == 16384, "hidden_size must be 16384");
    TORCH_CHECK(
        gate_output.size(2) == 53248,
        "intermediate_size must be 53248");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_005_swiglu_mlp_backward_launcher(
        grad_output, x, gate_output, up_output, activated_gate,
        gate_weight, up_weight, down_weight,
        grad_x, grad_gate_weight, grad_up_weight, grad_down_weight,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "SwiGLU MLP backward pipeline. Six at::matmul_out projections "
        "(cuBLAS) interleaved with three custom CUDA fused-elementwise "
        "kernels for the gated_output recompute, the SiLU-derivative "
        "scaling, and the final grad_x accumulation. Destination-passing "
        "style.");
}
