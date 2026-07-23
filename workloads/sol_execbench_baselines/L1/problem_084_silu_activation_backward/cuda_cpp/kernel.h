#pragma once

#ifndef PROBLEM_084_SILU_ACTIVATION_BACKWARD_KERNEL_H
#define PROBLEM_084_SILU_ACTIVATION_BACKWARD_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the FP32 SiLU backward element-wise kernel.
//
// Inputs `grad_output`, `x`, `sigmoid_x` are all shape [num_elements]
// FP32 contiguous; `grad_input` is pre-allocated with the same shape
// (destination-passing style). Algorithm:
//
//   grad_input = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))
//
// The kernel streams the three inputs once and writes the output
// once; no shared-memory reductions, no shape coordination.
void problem_084_silu_activation_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& sigmoid_x,
    torch::Tensor& grad_input,
    cudaStream_t stream);

#endif  // PROBLEM_084_SILU_ACTIVATION_BACKWARD_KERNEL_H
