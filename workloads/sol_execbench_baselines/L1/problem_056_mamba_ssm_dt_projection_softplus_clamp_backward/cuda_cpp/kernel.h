#pragma once

#ifndef PROBLEM_056_MAMBA_SOFTPLUS_CLAMP_BACKWARD_KERNEL_H
#define PROBLEM_056_MAMBA_SOFTPLUS_CLAMP_BACKWARD_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the BF16 Mamba SSM dt-projection softplus +
// clamp backward kernel.
//
// Algorithm per (row = b*S + s, h):
//   clamp_mask    = (time_step_min < dt_activated < time_step_max)
//   softplus_grad = sigmoid(dt_with_bias.f32)
//   grad          = grad_output.f32 * clamp_mask * softplus_grad
//   grad_dt[row, h]    = bf16(grad)
//   grad_dt_bias[h]    = bf16(Σ_(b, s) grad)   # fp32 sum-then-cast
//
// Single kernel, per-head CTA layout: Grid (num_heads,);
// BLOCK_THREADS = 256 stride-loops over num_rows = B*S. Each thread
// maintains a fp32 partial_sum accumulator while writing grad_dt
// per-element; warp + 8-way cross-warp tree reduces; thread 0
// writes bf16(total_sum) to grad_dt_bias[h]. The fp32-sum-then-cast
// chain matches the reference's `.sum(dim=(0, 1)).to(torch.bfloat16)`
// exactly.
//
// num_heads = 40 is pinned by the problem definition; batch_size
// and seq_len are workload-variable.
void problem_056_mamba_softplus_clamp_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& dt_with_bias,
    const torch::Tensor& dt_activated,
    float time_step_min,
    float time_step_max,
    torch::Tensor& grad_dt,
    torch::Tensor& grad_dt_bias,
    cudaStream_t stream);

#endif  // PROBLEM_056_MAMBA_SOFTPLUS_CLAMP_BACKWARD_KERNEL_H
