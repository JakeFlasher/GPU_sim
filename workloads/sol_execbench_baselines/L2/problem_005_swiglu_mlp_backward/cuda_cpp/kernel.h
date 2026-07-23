#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the SwiGLU MLP backward pipeline. Allocates per-
// launch intermediates (grad_gated_output, gated_output workspace,
// grad_up_output, grad_gate_output, two grad_x partials) via
// torch::empty(), dispatches the six GEMMs through at::matmul_out,
// and runs three custom CUDA fused-elementwise kernels in between.
// hidden_size=16384, intermediate_size=53248 pinned by the problem
// definition. Destination-passing style on grad_x / grad_gate_weight
// / grad_up_weight / grad_down_weight.
void problem_005_swiglu_mlp_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& gate_output,
    const torch::Tensor& up_output,
    const torch::Tensor& activated_gate,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    torch::Tensor& grad_x,
    torch::Tensor& grad_gate_weight,
    torch::Tensor& grad_up_weight,
    torch::Tensor& grad_down_weight,
    cudaStream_t stream);

void swiglu_mul_kernel_launch(
    const at::BFloat16* a,
    const at::BFloat16* b,
    at::BFloat16* out,
    long num_elems,
    cudaStream_t stream);

void swiglu_backward_elementwise_launch(
    const at::BFloat16* grad_gated_output,
    const at::BFloat16* up_output,
    const at::BFloat16* activated_gate,
    const at::BFloat16* gate_output,
    at::BFloat16* grad_up_output,
    at::BFloat16* grad_gate_output,
    long num_elems,
    cudaStream_t stream);

void elementwise_add_kernel_launch(
    const at::BFloat16* a,
    const at::BFloat16* b,
    at::BFloat16* out,
    long num_elems,
    cudaStream_t stream);
