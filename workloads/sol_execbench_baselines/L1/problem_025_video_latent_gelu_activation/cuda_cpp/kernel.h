#pragma once

#ifndef PROBLEM_025_VIDEO_LATENT_GELU_ACTIVATION_KERNEL_H
#define PROBLEM_025_VIDEO_LATENT_GELU_ACTIVATION_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the FP32 GELU (tanh approximation) element-wise
// kernel. Input ``x`` is [batch_size, seq_len, intermediate_size]
// FP32 contiguous; ``output`` is pre-allocated with the same shape
// and dtype (destination-passing style). The kernel does no data
// rearrangement; it streams ``x`` once and writes ``output`` once.
void problem_025_video_latent_gelu_launcher(
    const torch::Tensor& x,
    torch::Tensor& output,
    cudaStream_t stream);

#endif  // PROBLEM_025_VIDEO_LATENT_GELU_ACTIVATION_KERNEL_H
