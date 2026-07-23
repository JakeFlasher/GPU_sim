#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the fused-residual + RMSNorm backward kernel.
// Computes grad_hidden_states, grad_residual, and grad_weight given
// grad_output, x (unused by the math, kept for signature parity),
// normalized, rstd, and weight. hidden_size is pinned at 2560 by the
// problem definition. Destination-passing style: all three output
// tensors are pre-allocated.
void problem_013_fused_residual_rms_norm_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& normalized,
    const torch::Tensor& rstd,
    const torch::Tensor& weight,
    torch::Tensor& grad_hidden_states,
    torch::Tensor& grad_residual,
    torch::Tensor& grad_weight,
    cudaStream_t stream);
