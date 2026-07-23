#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the FLUX final-output norm + projection chain.
//
// Algorithm (mirroring the PyTorch reference):
//   mean       = hidden_states.mean(dim=-1, keepdim=True)
//   var        = hidden_states.var(dim=-1, keepdim=True,
//                                  unbiased=False)
//   hs_norm    = (hidden_states - mean) / sqrt(var + eps)
//   temb_silu  = temb * sigmoid(temb)
//   modulation = F.linear(temb_silu, linear_weight, linear_bias)
//                # [B, modulation_dim = 6144]
//   shift      = modulation[:, :inner_dim]
//   scale      = modulation[:, inner_dim:]
//   hs_mod     = hs_norm * (1 + scale.unsqueeze(1))
//                + shift.unsqueeze(1)
//   output     = F.linear(hs_mod, proj_out_weight, proj_out_bias)
//                # [B, S, output_dim = 64]
//
// Note: this slug puts shift FIRST in the modulation chunk (the
// opposite of problem_080's AdaLayerNormContinuous where scale
// was first).
//
// Tensors:
//   hidden_states    : [B, S, inner_dim=3072]                 fp32
//   temb             : [B, inner_dim=3072]                    fp32
//   linear_weight    : [modulation_dim=6144, inner_dim=3072]  fp32
//   linear_bias      : [modulation_dim=6144]                  fp32
//   proj_out_weight  : [output_dim=64, inner_dim=3072]        fp32
//   proj_out_bias    : [output_dim=64]                        fp32
//   eps              : fp32 scalar (passed as double, downcast at
//                      the launcher boundary)
//   output           : [B, S, output_dim=64]                  fp32
//                      (DPS; pre-allocated by the eval driver)
//
// `inner_dim = 3072`, `output_dim = 64`, and
// `modulation_dim = 2 * inner_dim = 6144` are pinned by the
// problem definition; `batch_size` and `seq_len` vary.
//
// Internally the launcher allocates three fp32 workspaces:
//   temb_silu   [B, inner_dim]
//   modulation  [B, modulation_dim]
//   hs_mod      [B, S, inner_dim]
void problem_036_flux_output_norm_projection_chain_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& temb,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias,
    const torch::Tensor& proj_out_weight,
    const torch::Tensor& proj_out_bias,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream);
