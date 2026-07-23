#pragma once

#ifndef PROBLEM_061_TANH_GATED_RESIDUAL_ADD_BACKWARD_KERNEL_H
#define PROBLEM_061_TANH_GATED_RESIDUAL_ADD_BACKWARD_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher set for the BF16 tanh-gated residual-addition
// backward kernels. The reference forward is:
//   output = residual + tanh(gate) * hidden_states * mask
// with mask broadcast over the final hidden_size axis. The
// backward gradients are:
//   grad_residual       = grad_output                    # identity
//   grad_hidden_states  = grad_output * tanh(gate) * mask  # per-element
//   grad_gate           = sech²(gate) *
//                         Σ_{b,s,h} grad_output * hidden_states * mask
//
// The gate is a zero-dim BF16 scalar input, so `tanh_g` and
// `sech²(g) = 1 - tanh²(g)` are evaluated ONCE on the host and
// passed to the device kernels as `float` scalar arguments.
//
// Kernel A computes grad_hidden_states + per-(b, s) row partial
// sums of `(grad_output * hidden_states * mask)` into a
// `row_partials[B*S]` fp32 buffer. Kernel B does a single-CTA
// tree reduction of that buffer, scales by `sech²(gate)`, casts
// to BF16, and writes to the zero-dim grad_gate output tensor.
//
// hidden_size = 4096 is pinned; batch_size and seq_len are
// workload-variable. The grad_residual = grad_output identity
// copy is handled in main.cpp via `output.copy_(input)` rather
// than a dedicated kernel.
void problem_061_grad_hidden_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& mask,
    float tanh_g,
    torch::Tensor& grad_hidden_states,
    torch::Tensor& row_partials,
    cudaStream_t stream);

void problem_061_grad_gate_reduce_launcher(
    const torch::Tensor& row_partials,
    float sech2,
    torch::Tensor& grad_gate,
    cudaStream_t stream);

#endif  // PROBLEM_061_TANH_GATED_RESIDUAL_ADD_BACKWARD_KERNEL_H
