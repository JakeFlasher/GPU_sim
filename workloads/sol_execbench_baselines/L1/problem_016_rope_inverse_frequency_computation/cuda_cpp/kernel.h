#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the RoPE inverse-frequency-computation kernel.
// One-shot elementwise: `inv_freq[i] = 1 / rope_theta^(2*i / head_dim)`
// for i in [0, half_head_dim). Output is pre-allocated by the eval
// driver and written in-place (destination-passing style). head_dim is
// pinned at 128 per the problem definition, so half_head_dim = 64.
void problem_016_rope_inverse_frequency_computation_launcher(
    float rope_theta,
    torch::Tensor& inv_freq,
    cudaStream_t stream);
