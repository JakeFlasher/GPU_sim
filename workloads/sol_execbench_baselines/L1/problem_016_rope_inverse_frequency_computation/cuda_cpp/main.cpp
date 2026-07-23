#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

// Destination-passing-style entry. The definition declares:
//   rope_theta: scalar float32
//   inv_freq  : [half_head_dim]  float32 (DPS, pre-allocated)
//
// head_dim is pinned at 128 per definition.json, so half_head_dim = 64.

void run(double rope_theta, torch::Tensor inv_freq) {
    TORCH_CHECK(inv_freq.is_cuda(), "inv_freq must be a CUDA tensor");
    TORCH_CHECK(
        inv_freq.dtype() == torch::kFloat32,
        "inv_freq must be float32, got ", inv_freq.dtype());
    TORCH_CHECK(inv_freq.is_contiguous(), "inv_freq must be contiguous");
    TORCH_CHECK(inv_freq.dim() == 1, "inv_freq must be 1D");
    TORCH_CHECK(
        inv_freq.size(0) == 64,
        "inv_freq must have 64 elements (head_dim=128/2), got ",
        inv_freq.size(0));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_016_rope_inverse_frequency_computation_launcher(
        static_cast<float>(rope_theta),
        inv_freq,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "RoPE inverse-frequency computation; scalar rope_theta -> "
        "inv_freq[64] fp32; head_dim=128 pinned by problem definition; "
        "destination-passing style.");
}
