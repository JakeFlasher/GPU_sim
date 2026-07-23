#pragma once

#ifndef PROBLEM_069_RMS_NORM_KERNEL_H
#define PROBLEM_069_RMS_NORM_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for fused residual-add + RMSNorm, hidden_size=8192 (BF16).
// One CTA per (batch, seq) row; inputs are 3D
// [batch_size, seq_len, hidden_size] with the last dim contiguous, so the
// row stride is hidden_size BF16 elements. Output is pre-allocated and
// written in-place (destination-passing style).
void problem_069_rms_norm_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream);

#endif  // PROBLEM_069_RMS_NORM_KERNEL_H
