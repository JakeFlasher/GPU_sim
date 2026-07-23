#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the fused add + RMSNorm kernel at hidden_size=7168.
// Inputs and outputs are BF16; intermediate accumulation runs in FP32
// per the precision contract (CLAUDE.md Rule 8). Output is pre-allocated
// by the eval driver and written in-place (destination-passing style).
// EPS is pinned at 1e-6 by the problem definition.
void problem_003_fused_add_rmsnorm_h7168_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    torch::Tensor& output,
    cudaStream_t stream);
