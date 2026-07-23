#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Generate the FLUX timestep_embed = cat([cos(args), sin(args)], -1)
// where args = (timestep * 1000)[:, None] * freqs[None, :].
// Output shape: [batch_size, time_embed_dim].
void problem_031_flux_timestep_embed_launcher(
    const torch::Tensor& timestep,           // [B] float32
    const torch::Tensor& freqs,              // [half_time_embed_dim] float32
    torch::Tensor& timestep_embed_out,       // [B, time_embed_dim] float32
    cudaStream_t stream);

// In-place SiLU: x <- x * sigmoid(x). Operates over a flat [N] view.
void problem_031_flux_silu_inplace_launcher(
    torch::Tensor& x,                        // arbitrary shape, float32
    cudaStream_t stream);
