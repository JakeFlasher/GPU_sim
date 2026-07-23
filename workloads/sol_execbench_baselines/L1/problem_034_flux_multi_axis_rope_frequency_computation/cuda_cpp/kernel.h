#pragma once

#ifndef PROBLEM_034_FLUX_MULTI_AXIS_ROPE_FREQ_KERNEL_H
#define PROBLEM_034_FLUX_MULTI_AXIS_ROPE_FREQ_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the fp32 multi-axis RoPE frequency kernel.
//
// Given ids of shape [seq_len, n_axes=3] fp32 and scalar theta,
// computes freqs_cos and freqs_sin both of shape [seq_len,
// total_dim=128] by independently running 1D RoPE per axis
// (axes_dim = [16, 56, 56]) and concatenating along the
// last dimension. For each axis a with axis_dim = 2 * half_dim_a:
//
//   freq_band[a, k] = 1.0 / theta^(k / half_dim_a)  for k in [0, half_dim_a)
//   angle[s, a, k]  = ids[s, a] * freq_band[a, k]
//   cos[s, axis_offset_a + 2*k]   = cos(angle[s, a, k])
//   cos[s, axis_offset_a + 2*k+1] = cos(angle[s, a, k])
//   sin[s, axis_offset_a + 2*k]   = sin(angle[s, a, k])
//   sin[s, axis_offset_a + 2*k+1] = sin(angle[s, a, k])
//
// (The repeat-by-2 along d-pairs comes from torch.repeat_interleave
// on the per-axis cos/sin outputs, then concatenation places the
// three axes' results contiguously: axis 0 occupies d in [0, 16),
// axis 1 occupies [16, 72), axis 2 occupies [72, 128).)
//
// One CTA per seq_len position, BLOCK_THREADS = total_dim = 128
// (one thread per output d element). Both outputs are filled by
// one kernel launch.
void problem_034_flux_multi_axis_rope_freq_launcher(
    const torch::Tensor& ids,
    float theta,
    torch::Tensor& freqs_cos,
    torch::Tensor& freqs_sin,
    cudaStream_t stream);

#endif  // PROBLEM_034_FLUX_MULTI_AXIS_ROPE_FREQ_KERNEL_H
