#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the Vision MLP + GELU-tanh backward pass.
//
// Algorithm (mirroring the PyTorch reference):
//   grad_fc2_bias    = grad_output.sum(dim=0)
//   grad_fc2_weight  = grad_output.t() @ gelu_output
//   grad_gelu_output = grad_output       @ fc2_weight
//   gelu_grad        = 0.5*(1 + tanh(arg))
//                    + 0.5*x*sech^2(arg)*d(arg)/dx
//     arg            = sqrt(2/pi) * (x + 0.044715 * x^3)
//     d(arg)/dx      = sqrt(2/pi) * (1 + 3*0.044715 * x^2)
//   grad_fc1_output  = grad_gelu_output * gelu_grad
//   grad_fc1_bias    = grad_fc1_output.sum(dim=0)
//   grad_fc1_weight  = grad_fc1_output.t() @ hidden_state
//   grad_hidden_state= grad_fc1_output     @ fc1_weight
//
// Tensors:
//   grad_output       : [S, hidden_size=1280]            fp32
//   hidden_state      : [S, hidden_size=1280]            fp32
//   fc1_weight        : [intermediate_size=3584, hidden_size=1280] fp32
//   fc1_bias          : [intermediate_size=3584]         fp32 (unused
//                       in run(); accepted only to match the reference
//                       signature)
//   fc2_weight        : [hidden_size=1280, intermediate_size=3584]  fp32
//   fc2_bias          : [hidden_size=1280]               fp32 (unused
//                       in run(); accepted only to match the reference
//                       signature)
//   fc1_output        : [S, intermediate_size=3584]      fp32
//   gelu_output       : [S, intermediate_size=3584]      fp32
//   grad_hidden_state : [S, hidden_size=1280]            fp32 (DPS)
//   grad_fc1_weight   : [intermediate_size=3584, hidden_size=1280] fp32 (DPS)
//   grad_fc1_bias     : [intermediate_size=3584]         fp32 (DPS)
//   grad_fc2_weight   : [hidden_size=1280, intermediate_size=3584] fp32 (DPS)
//   grad_fc2_bias     : [hidden_size=1280]               fp32 (DPS)
//
// hidden_size = 1280 and intermediate_size = 3584 are pinned by the
// problem definition; seq_len varies.
//
// Internally the launcher allocates two fp32 workspaces:
//   grad_gelu_output [S, intermediate_size]
//   grad_fc1_output  [S, intermediate_size]
void problem_022_vision_mlp_gelu_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_state,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc1_output,
    const torch::Tensor& gelu_output,
    torch::Tensor& grad_hidden_state,
    torch::Tensor& grad_fc1_weight,
    torch::Tensor& grad_fc1_bias,
    torch::Tensor& grad_fc2_weight,
    torch::Tensor& grad_fc2_bias,
    cudaStream_t stream);
