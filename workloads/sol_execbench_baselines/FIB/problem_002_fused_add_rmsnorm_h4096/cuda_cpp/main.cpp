#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_bf16_contiguous(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kBFloat16,
        name, " must have bfloat16 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. The definition declares:
//   hidden_states: [batch_size, hidden_size]  bf16
//   residual     : [batch_size, hidden_size]  bf16
//   weight       : [hidden_size]              bf16
//   output       : [batch_size, hidden_size]  bf16  (DPS, pre-alloc)
//
// hidden_size is constant at 4096 per definition.json; EPS is pinned
// at 1e-5 in the kernel body (Llama-3.1-8B RMSNorm).
void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    torch::Tensor output) {
    TORCH_CHECK(
        hidden_states.dim() == 2,
        "hidden_states must be 2D [batch, hidden]; got ",
        hidden_states.dim(), "D");
    TORCH_CHECK(
        residual.sizes() == hidden_states.sizes(),
        "residual shape must match hidden_states");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");

    const int64_t hidden_size = hidden_states.size(1);
    TORCH_CHECK(
        hidden_size == 4096,
        "hidden_size must be 4096 for problem_002, got ", hidden_size);
    TORCH_CHECK(
        weight.size(0) == hidden_size,
        "weight must have size ", hidden_size, ", got ", weight.size(0));

    check_bf16_contiguous(hidden_states, "hidden_states");
    check_bf16_contiguous(residual, "residual");
    check_bf16_contiguous(weight, "weight");
    check_bf16_contiguous(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_002_fused_add_rmsnorm_h4096_launcher(
        hidden_states,
        residual,
        weight,
        output,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused residual-add + RMSNorm at hidden_size=4096; BF16 inputs / "
        "output, FP32 accumulator; EPS=1e-5 pinned by problem definition; "
        "destination-passing style.");
}
