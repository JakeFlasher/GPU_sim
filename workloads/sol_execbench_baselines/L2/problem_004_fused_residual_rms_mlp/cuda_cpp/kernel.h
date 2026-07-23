#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher entry for the fused residual + RMSNorm + SwiGLU MLP
// pattern. Allocates per-launch intermediates (norm_out, gate_out,
// up_out) via torch::empty(), launches three custom CUDA kernels
// (fused residual+RMS, two SwiGLU fusions sharing memory), and three
// at::matmul_out projections (gate, up, down). hidden_size=16384,
// intermediate_size=53248 pinned by the problem definition.
// Destination-passing style: output is pre-allocated.
void problem_004_fused_residual_rms_mlp_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_proj_weight,
    const torch::Tensor& up_proj_weight,
    const torch::Tensor& down_proj_weight,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream);

// Standalone CUDA kernels (declared here for the launcher to call).
void fused_residual_rms_kernel_launch(
    const at::BFloat16* hidden_states,
    const at::BFloat16* residual,
    const at::BFloat16* norm_weight,
    at::BFloat16* norm_out,
    float eps,
    int num_rows,
    cudaStream_t stream);

void swiglu_kernel_launch(
    const at::BFloat16* gate_out,
    const at::BFloat16* up_out,
    at::BFloat16* intermediate,
    long num_elems,
    cudaStream_t stream);
