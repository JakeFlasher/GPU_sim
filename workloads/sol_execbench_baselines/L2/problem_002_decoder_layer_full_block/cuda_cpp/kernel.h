#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the LLaMA full-decoder-layer baseline.
// All tensors fp32. hidden_size=4096, intermediate_size=14336,
// num_attention_heads=32, num_key_value_heads=8, head_dim=128 are
// pinned by the problem definition. Destination-passing style on
// `output`.
void problem_002_decoder_layer_full_block_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& attention_mask,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& post_attention_layernorm_weight,
    const torch::Tensor& gate_proj_weight,
    const torch::Tensor& up_proj_weight,
    const torch::Tensor& down_proj_weight,
    float rms_norm_eps,
    torch::Tensor& output,
    cudaStream_t stream);

void rms_norm_kernel_launch(
    const float* x,
    const float* weight,
    float* out,
    float eps,
    int num_rows,
    int hidden_size,
    cudaStream_t stream);

void silu_mul_kernel_launch(
    const float* gate,
    const float* up,
    float* out,
    long num_elems,
    cudaStream_t stream);
