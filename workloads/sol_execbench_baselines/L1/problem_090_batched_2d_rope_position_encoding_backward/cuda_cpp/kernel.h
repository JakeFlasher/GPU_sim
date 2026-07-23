#pragma once

#ifndef PROBLEM_090_BATCHED_2D_ROPE_POSITION_ENCODING_BACKWARD_KERNEL_H
#define PROBLEM_090_BATCHED_2D_ROPE_POSITION_ENCODING_BACKWARD_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the BF16/fp32 batched 2D RoPE position encoding
// backward kernel. The reference forward emits cos(idx_theta) and
// sin(idx_theta); the backward computes:
//   grad_idx_theta = -grad_cos.f32 * sin(idx_theta)
//                  +  grad_sin.f32 * cos(idx_theta)
//
// Pure per-element kernel: no reductions, no broadcasts, no
// head-axis collapse. Every output element depends on exactly the
// three input elements at the same position.
//
// One CTA per (b, s) row over head_dim elements; BLOCK_THREADS =
// 256 (each thread stride-loops over the row). batch_size, seq_len,
// and head_dim are all workload-variable.
void problem_090_batched_2d_rope_backward_launcher(
    const torch::Tensor& grad_cos,
    const torch::Tensor& grad_sin,
    const torch::Tensor& idx_theta,
    torch::Tensor& grad_idx_theta,
    cudaStream_t stream);

#endif  // PROBLEM_090_BATCHED_2D_ROPE_POSITION_ENCODING_BACKWARD_KERNEL_H
