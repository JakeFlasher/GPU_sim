#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the AdaLayerNormContinuous + modulation kernel
// chain.
//
// Algorithm:
//   modulation = F.linear(temb, linear_weight, linear_bias)
//                  # [B, inner_dim_x2 = 4864]
//   scale = modulation[:, :inner_dim]              # [B, inner_dim]
//   shift = modulation[:, inner_dim:]              # [B, inner_dim]
//   mean  = hidden_states.mean(dim=-1, keepdim=True)
//   var   = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
//   normalized = (hidden_states - mean) / sqrt(var + eps)
//   output = normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
//
// Tensors:
//   hidden_states : [B, S, inner_dim=2432] fp32
//   temb          : [B, inner_dim=2432] fp32
//   linear_weight : [inner_dim_x2=4864, inner_dim=2432] fp32
//   linear_bias   : [inner_dim_x2=4864] fp32
//   eps           : fp32 scalar (passed as double, downcast at boundary)
//   output        : [B, S, inner_dim=2432] fp32 (DPS;
//                   pre-allocated by the eval driver)
//
// `inner_dim = 2432` and `inner_dim_x2 = 4864` are pinned by the
// problem definition; `batch_size` and `seq_len` vary.
//
// Internally the launcher allocates a fp32 workspace `[B, 4864]`
// for the `modulation` intermediate.
void problem_080_adaptive_layernorm_continuous_with_modulation_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& temb,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream);
