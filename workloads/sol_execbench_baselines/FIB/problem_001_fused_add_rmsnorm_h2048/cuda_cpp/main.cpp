#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

// PyBind11 entry point. The eval driver pre-allocates the output tensor and
// passes it in destination-passing style; this function writes results
// in-place and returns void per CLAUDE.md "Submission Format".

static void check_bfloat16_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void run(torch::Tensor hidden_states, torch::Tensor residual, torch::Tensor weight, torch::Tensor output) {
    check_bfloat16_cuda_contiguous(hidden_states, "hidden_states");
    check_bfloat16_cuda_contiguous(residual, "residual");
    check_bfloat16_cuda_contiguous(weight, "weight");
    check_bfloat16_cuda_contiguous(output, "output");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2-D");
    TORCH_CHECK(residual.dim() == 2, "residual must be 2-D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
    TORCH_CHECK(output.dim() == 2, "output must be 2-D");

    const int64_t batch_size = hidden_states.size(0);
    const int64_t hidden_size = hidden_states.size(1);

    TORCH_CHECK(hidden_size == 2048, "hidden_size must be 2048, got ", hidden_size);
    TORCH_CHECK(residual.size(0) == batch_size && residual.size(1) == hidden_size,
                "residual shape mismatch");
    TORCH_CHECK(weight.size(0) == hidden_size, "weight size must equal hidden_size");
    TORCH_CHECK(output.size(0) == batch_size && output.size(1) == hidden_size,
                "output shape mismatch");

    constexpr float kEps = 1.0000000e-06f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_add_rmsnorm_h2048_launcher(
        hidden_states, residual, weight, output, kEps, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused add + RMSNorm kernel for hidden_size=2048 (BFloat16, CUDA, DPS).");
}
