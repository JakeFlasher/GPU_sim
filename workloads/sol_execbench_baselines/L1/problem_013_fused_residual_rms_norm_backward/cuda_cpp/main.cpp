#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry. Per definition.json input order is
// grad_output, x, normalized, rstd, weight, followed by output args
// grad_hidden_states, grad_residual, grad_weight. The x tensor is
// unused by the reference math (it's a saved-forward tensor that the
// gradient formula does not consume) but is kept in the signature for
// parity with the eval driver's per-workload binding.

void run(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor normalized,
    torch::Tensor rstd,
    torch::Tensor weight,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_residual,
    torch::Tensor grad_weight) {

    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(grad_output.dtype() == torch::kBFloat16,
                "grad_output must be bfloat16");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.dim() == 3,
                "grad_output must be 3D [B, S, H]");
    TORCH_CHECK(grad_output.size(2) == 2560,
                "hidden_size must be 2560");

    TORCH_CHECK(normalized.dtype() == torch::kFloat32 &&
                rstd.dtype() == torch::kFloat32 &&
                weight.dtype() == torch::kFloat32,
                "normalized/rstd/weight must be float32");

    TORCH_CHECK(grad_hidden_states.dtype() == torch::kBFloat16 &&
                grad_residual.dtype() == torch::kBFloat16,
                "grad_hidden_states/grad_residual must be bfloat16");
    TORCH_CHECK(grad_weight.dtype() == torch::kFloat32,
                "grad_weight must be float32");

    TORCH_CHECK(grad_hidden_states.is_contiguous() &&
                grad_residual.is_contiguous() &&
                grad_weight.is_contiguous(),
                "output tensors must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_013_fused_residual_rms_norm_backward_launcher(
        grad_output, x, normalized, rstd, weight,
        grad_hidden_states, grad_residual, grad_weight,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused-residual + RMSNorm backward; two-kernel decomposition "
        "(per-row grad_x + per-column grad_weight); destination-passing "
        "style.");
}
