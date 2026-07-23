#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the FLUX feedforward (fp32 Linear -> GELU-tanh
// -> Linear):
//
//   x_pre  = hidden_states[B*S, hidden_dim=3072]
//            @ fc1_weight^T[3072, mlp_hidden_dim=12288]
//   x_act  = gelu_tanh(x_pre + fc1_bias)
//   y_pre  = x_act[B*S, 12288] @ fc2_weight^T[12288, 3072]
//   output = y_pre + fc2_bias
//
// GELU-tanh approximation:
//   gelu_tanh(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
//
// Tensors:
//   hidden_states : [B, S, hidden_dim=3072] fp32
//   fc1_weight    : [mlp_hidden_dim=12288, hidden_dim=3072] fp32
//   fc1_bias      : [mlp_hidden_dim=12288] fp32
//   fc2_weight    : [hidden_dim=3072, mlp_hidden_dim=12288] fp32
//   fc2_bias      : [hidden_dim=3072] fp32
//   output        : [B, S, hidden_dim=3072] fp32 (DPS;
//                   pre-allocated by the eval driver).
//
// hidden_dim = 3072 and mlp_hidden_dim = 12288 are pinned by the
// problem definition; B and S vary per workload.
void problem_037_flux_feedforward_gelu_approximate_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_bias,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc2_bias,
    torch::Tensor& output,
    cudaStream_t stream);
