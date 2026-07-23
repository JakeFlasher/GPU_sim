#pragma once

#ifndef PROBLEM_038_FLUX_MULTI_HEAD_RMSNORM_QK_KERNEL_H
#define PROBLEM_038_FLUX_MULTI_HEAD_RMSNORM_QK_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the fp32 per-head RMSNorm-QK kernel.
//
// For each input tensor `x` of shape [B, S, num_heads=48, head_dim=128]
// fp32, compute:
//   var[b, s, h] = mean_d(x[b, s, h, d]^2)
//   x_norm[b, s, h, d] = x[b, s, h, d] *
//                        rsqrt(var[b, s, h] + eps) *
//                        weight[h, d]
//
// `weight` is [num_heads, head_dim] fp32 and is reused across all
// (B, S) row CTAs at the same `h`. The launcher is parameterized on
// the (input, weight, output) triple so the DPS entry can call it
// twice (once for query+weight_q → query_norm, once for key+weight_k
// → key_norm).
void problem_038_flux_multi_head_rmsnorm_qk_launcher(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    torch::Tensor& x_norm,
    float eps,
    cudaStream_t stream);

#endif  // PROBLEM_038_FLUX_MULTI_HEAD_RMSNORM_QK_KERNEL_H
