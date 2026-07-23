#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the AdaLayerNormZero modulation-extraction
// Linear-with-bias + 6-chunk multi-output DPS kernel:
//
//   modulation[b, n] = sum_k weight[n, k] * emb[b, k] + bias[n]
//   for n in [0, 6*inner_dim).
//   Then chunk along n into six [B, inner_dim] outputs.
//
// emb            : [B, inner_dim=3072] fp32 row-major.
// weight         : [output_dim=18432, inner_dim=3072] fp32 row-major.
// bias           : [output_dim=18432] fp32.
// outputs (DPS, pre-allocated):
//   shift_msa    : [B, inner_dim=3072] fp32 row-major.
//   scale_msa    : [B, inner_dim=3072] fp32 row-major.
//   gate_msa     : [B, inner_dim=3072] fp32 row-major.
//   shift_mlp    : [B, inner_dim=3072] fp32 row-major.
//   scale_mlp    : [B, inner_dim=3072] fp32 row-major.
//   gate_mlp     : [B, inner_dim=3072] fp32 row-major.
//
// inner_dim = 3072 and output_dim = 6 * inner_dim = 18432 are pinned
// by the problem definition. B = batch_size varies per workload.
void problem_035_flux_ada_layer_norm_zero_modulation_extraction_launcher(
    const torch::Tensor& emb,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& shift_msa,
    torch::Tensor& scale_msa,
    torch::Tensor& gate_msa,
    torch::Tensor& shift_mlp,
    torch::Tensor& scale_mlp,
    torch::Tensor& gate_mlp,
    cudaStream_t stream);
