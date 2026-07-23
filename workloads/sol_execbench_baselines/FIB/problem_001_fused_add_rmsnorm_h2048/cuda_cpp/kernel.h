#pragma once

#ifndef FUSED_ADD_RMSNORM_001_FUSED_ADD_RMSNORM_H2048_KERNEL_H
#define FUSED_ADD_RMSNORM_001_FUSED_ADD_RMSNORM_H2048_KERNEL_H

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the fused add + RMSNorm kernel for hidden_size=2048.
// Inputs and outputs are BF16; intermediate accumulation runs in FP32 per the
// precision contract (CLAUDE.md Rule 8). Output is pre-allocated by the eval
// driver and written in-place (destination-passing style).
void fused_add_rmsnorm_h2048_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    torch::Tensor& output,
    float eps,
    cudaStream_t stream);

#endif  // FUSED_ADD_RMSNORM_001_FUSED_ADD_RMSNORM_H2048_KERNEL_H
