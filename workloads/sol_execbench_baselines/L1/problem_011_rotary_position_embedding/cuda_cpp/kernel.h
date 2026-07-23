#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the RoPE cos/sin embedding kernel. Given
// `position_ids[batch, seq_len]` (int64), `inv_freq[half_head_dim]`
// (float32, llama3-scaled), and a scalar `attention_scaling`, write
// `cos_sin[batch, seq_len, head_dim, 2]` (bfloat16) in-place. head_dim
// is pinned at 128 per the problem definition, so half_head_dim = 64.
// Destination-passing style: cos_sin is pre-allocated by the eval
// driver and written by the kernel; no return value.
void problem_011_rotary_position_embedding_launcher(
    const torch::Tensor& position_ids,
    const torch::Tensor& inv_freq,
    float attention_scaling,
    torch::Tensor& cos_sin,
    cudaStream_t stream);
