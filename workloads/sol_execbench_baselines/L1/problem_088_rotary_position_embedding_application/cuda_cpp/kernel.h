#pragma once

#ifndef PROBLEM_088_ROPE_APPLICATION_KERNEL_H
#define PROBLEM_088_ROPE_APPLICATION_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the fp32 RoPE-application kernel.
//
// For each input tensor `x` of shape [B, H, S, head_dim=128] fp32,
// compute:
//   rotate_half(x)[..., d]   = -x[..., d + 64]   for d in [0, 64)
//                            =  x[..., d - 64]   for d in [64, 128)
//   x_rotated[..., d]        = x[..., d] * cos[s, d] +
//                              rotate_half(x)[..., d] * sin[s, d]
//
// `cos` and `sin` are both [S, head_dim] fp32 and are reused across
// all (B, H) row CTAs. The launcher is parameterized on the input
// tensor and output destination so the DPS entry can call it twice
// (once for query→query_rotated, once for key→key_rotated).
void problem_088_rotary_position_embedding_application_launcher(
    const torch::Tensor& x,
    const torch::Tensor& cos_table,
    const torch::Tensor& sin_table,
    torch::Tensor& x_rotated,
    cudaStream_t stream);

#endif  // PROBLEM_088_ROPE_APPLICATION_KERNEL_H
