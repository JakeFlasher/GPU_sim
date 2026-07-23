#pragma once

#ifndef PROBLEM_078_GROUP_NORM_FUSION_KERNEL_H
#define PROBLEM_078_GROUP_NORM_FUSION_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the fp32 GroupNorm kernel.
//
// For input x of shape [B, C, H, W] fp32 with num_groups = 32
// (pinned by the problem definition), C divisible by num_groups:
//
//   For each (b, g) in [0, B) x [0, num_groups):
//     N = (C / num_groups) * H * W
//     sum    = sum   over (c_in_group, h, w) of x[b, g*Cg+c_in_group, h, w]
//     sum_sq = sum   over (c_in_group, h, w) of x[...]^2
//     mean   = sum / N
//     var    = sum_sq / N - mean * mean
//     inv_s  = rsqrt(var + eps)
//     For each (c_in_group, h, w):
//       c = g * (C / num_groups) + c_in_group
//       output[b, c, h, w] = (x[b, c, h, w] - mean) * inv_s
//                          * weight[c] + bias[c]
//
// One CTA per (b, g) row over the group's full
// (channels_per_group, H, W) tile.
void problem_078_group_norm_fusion_launcher(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    float eps,
    cudaStream_t stream);

#endif  // PROBLEM_078_GROUP_NORM_FUSION_KERNEL_H
