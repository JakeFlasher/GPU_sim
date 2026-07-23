#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the conv2d residual-block kernel set. Two
// device kernels in sequence:
//   1. conv_in: 3 -> 32 channels, kernel_size=3, padding=1,
//      producing a hidden (B, 32, H, W) intermediate the
//      launcher allocates.
//   2. conv_out_plus_residual: 32 -> 3 channels, kernel_size=3,
//      padding=1, with the input residual added on top of the
//      conv output: output = conv_out(hidden) + x.
// All inputs / outputs FP32 per definition.json. Destination-
// passing style for the final output tensor; the intermediate
// hidden is launcher-internal.
void problem_040_conv2d_residual_block_launcher(
    const torch::Tensor& x,
    const torch::Tensor& conv_in_weight,
    const torch::Tensor& conv_in_bias,
    const torch::Tensor& conv_out_weight,
    const torch::Tensor& conv_out_bias,
    torch::Tensor& output,
    cudaStream_t stream);
