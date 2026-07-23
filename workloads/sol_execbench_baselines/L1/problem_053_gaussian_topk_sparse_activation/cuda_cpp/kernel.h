#pragma once

#ifndef PROBLEM_053_GAUSSIAN_TOPK_SPARSE_ACTIVATION_KERNEL_H
#define PROBLEM_053_GAUSSIAN_TOPK_SPARSE_ACTIVATION_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the BF16 Gaussian top-k sparse activation kernel.
//
// For each (b, s) row of BF16 inputs[b, s, :] of length
// intermediate_size, compute in fp32:
//   mean   = mean_i(inputs[b, s, i])
//   var    = mean_i(inputs[b, s, i]^2) - mean^2
//   std    = sqrt(max(var, 0))
//   cutoff = mean + std * z_multiplier
// where `z_multiplier = ndtri(target_sparsity)` is precomputed
// ONCE on the host before the launch (Abramowitz-Stegun rational
// approximation, formula 26.2.23 from the reference's ndtri
// implementation; `target_sparsity` is a kernel-invocation constant
// scalar input). The output is the per-element fused
// `max(input.fp32 - cutoff, 0.0f)` cast back to BF16.
//
// One CTA per (b, s) row over the row's full intermediate_size
// elements via stride-loop. BF16x2 vector loads/stores in pass 2.
void problem_053_gaussian_topk_sparse_activation_launcher(
    const torch::Tensor& inputs,
    float z_multiplier,
    torch::Tensor& output,
    cudaStream_t stream);

#endif  // PROBLEM_053_GAUSSIAN_TOPK_SPARSE_ACTIVATION_KERNEL_H
