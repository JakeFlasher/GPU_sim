#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry. Per definition.json input order is
// hidden_states, residual, norm_weight, gate_proj_weight,
// up_proj_weight, down_proj_weight, eps; output is `output`.

static void check_bf16_cuda_contiguous(
    const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ",
        t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void run(
    torch::Tensor hidden_states,
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor gate_proj_weight,
    torch::Tensor up_proj_weight,
    torch::Tensor down_proj_weight,
    double eps,
    torch::Tensor output) {

    check_bf16_cuda_contiguous(hidden_states, "hidden_states");
    check_bf16_cuda_contiguous(residual, "residual");
    check_bf16_cuda_contiguous(norm_weight, "norm_weight");
    check_bf16_cuda_contiguous(gate_proj_weight, "gate_proj_weight");
    check_bf16_cuda_contiguous(up_proj_weight, "up_proj_weight");
    check_bf16_cuda_contiguous(down_proj_weight, "down_proj_weight");
    check_bf16_cuda_contiguous(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D");
    TORCH_CHECK(
        hidden_states.size(2) == 16384, "hidden_size must be 16384");
    TORCH_CHECK(
        gate_proj_weight.size(0) == 53248,
        "intermediate_size must be 53248");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_004_fused_residual_rms_mlp_launcher(
        hidden_states, residual, norm_weight,
        gate_proj_weight, up_proj_weight, down_proj_weight,
        static_cast<float>(eps),
        output,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused residual + RMSNorm + SwiGLU MLP. Custom CUDA kernels "
        "for the residual+RMS and SwiGLU fusions; three at::matmul_out "
        "calls (cuBLAS on the current stream) for the projections. "
        "Destination-passing style.");
}
