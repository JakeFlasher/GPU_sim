#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the Stable-Diffusion-UNet ResNet block with
// time-embedding injection. Pipeline (per CTA per (batch, hw)
// position, fanned across channels):
//
//   1. GroupNorm + SiLU + conv1 (in=320, out=320, k=3, p=1)
//   2. SiLU on time_emb + linear (1280 -> 320)
//   3. Broadcast-add time-projection (B, 320, 1, 1) onto features
//   4. GroupNorm + SiLU + conv2 (320 -> 320, k=3, p=1)
//   5. Residual add (output = h + x; in_channels == out_channels)
//
// The launcher allocates three (B, 320, H, W) intermediate
// buffers (h1, h2, h3) via torch::empty so the eval harness's
// DPS contract is preserved for the final output only.
// FP32 throughout per definition.json.
void problem_066_resnet_block_with_time_embedding_launcher(
    const torch::Tensor& x,
    const torch::Tensor& time_emb,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& conv1_weight,
    const torch::Tensor& conv1_bias,
    const torch::Tensor& time_emb_proj_weight,
    const torch::Tensor& time_emb_proj_bias,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    const torch::Tensor& conv2_weight,
    const torch::Tensor& conv2_bias,
    float norm_eps,
    torch::Tensor& output,
    cudaStream_t stream);
