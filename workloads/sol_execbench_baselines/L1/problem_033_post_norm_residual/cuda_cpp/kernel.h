#pragma once

#ifndef PROBLEM_033_POST_NORM_RESIDUAL_KERNEL_H
#define PROBLEM_033_POST_NORM_RESIDUAL_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the BF16 post-norm-residual kernel.
//
// Algorithm:
//   normalized = RMSNorm(sublayer_output, weight, eps)  [bf16]
//   output     = residual + normalized
//
// where the RMSNorm step computes
//   x_fp32 = sublayer_output.to(fp32)
//   inv_rms = rsqrt(mean(x_fp32^2) + eps)
//   normalized_fp32 = x_fp32 * inv_rms * weight_fp32
//   normalized = normalized_fp32.to(bf16)
//
// hidden_size = 4096 pinned. eps is a runtime fp32 scalar.
//
// One CTA per row over `batch_size * seq_len` rows; threads
// cooperatively reduce sum-of-squares in fp32 with warp shuffle +
// cross-warp shared-memory tree, then write the residual-added
// output as bf16.
void problem_033_post_norm_residual_launcher(
    const torch::Tensor& sublayer_output,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream);

#endif  // PROBLEM_033_POST_NORM_RESIDUAL_KERNEL_H
