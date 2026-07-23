#pragma once

#ifndef PROBLEM_085_GEGLU_ACTIVATION_KERNEL_H
#define PROBLEM_085_GEGLU_ACTIVATION_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the FP32 GEGLU activation kernel.
//
// Input `x` is [batch_size, seq_len, inner_dim*2] FP32 contiguous;
// `output` is pre-allocated with shape [batch_size, seq_len, inner_dim]
// (destination-passing style). Algorithm:
//
//   x_gate, x_linear = x.chunk(2, dim=-1)
//   output = GELU_tanh(x_gate) * x_linear
//
// inner_dim is pinned at 5120 by the problem definition.
void problem_085_geglu_activation_launcher(
    const torch::Tensor& x,
    torch::Tensor& output,
    cudaStream_t stream);

#endif  // PROBLEM_085_GEGLU_ACTIVATION_KERNEL_H
