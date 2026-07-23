#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the plain (non-fused) RMSNorm kernel at
// hidden_size=512. Inputs and outputs are BF16; intermediate
// accumulation runs in FP32 per the precision contract
// (CLAUDE.md Rule 8). Output is pre-allocated by the eval
// driver and written in-place (destination-passing style).
// EPS is a compile-time constant baked into the kernel
// (1e-6 per the problem definition).
void problem_022_rmsnorm_h512_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    torch::Tensor& output,
    cudaStream_t stream);
