#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the hybrid attention-mask preparation kernel:
//
//   For each (b in [0, B), h in [0, num_heads), t in [0, T),
//                   j in [0, total_length)):
//     causal_cond = (t >= j - past_kv_length)
//     full_attention_mask[b, h, t, j]           = !causal_cond
//     sliding_window_attention_mask[b, h, t, j] = false
//
// The sliding-window mask is unconditionally False because the
// reference's `swa_mask = torch.zeros(...).masked_fill(
// valid_positions, False)` is a no-op (a zeros buffer filled
// with False at True positions yields a zeros buffer). To match
// the canonical reference bit-for-bit the kernel writes False
// for every (b, h, t, j); the `sliding_window` constant is
// accepted by this launcher for API completeness but is unused
// by the kernel.
//
// Outputs (DPS):
//   full_attention_mask           : [B, num_heads=64, T, total_length] bool
//   sliding_window_attention_mask : [B, num_heads=64, T, total_length] bool
//
// Inputs are passed as int64 host scalars. num_attention_heads = 64,
// swa_num_attention_heads = 64, and sliding_window = 128 are pinned
// by the problem definition; batch_size, seq_length,
// past_key_values_length vary.
void problem_028_hybrid_attention_mask_preparation_launcher(
    int64_t batch_size,
    int64_t seq_length,
    int64_t past_kv_length,
    int64_t sliding_window,
    torch::Tensor& full_attention_mask,
    torch::Tensor& sliding_window_attention_mask,
    cudaStream_t stream);
