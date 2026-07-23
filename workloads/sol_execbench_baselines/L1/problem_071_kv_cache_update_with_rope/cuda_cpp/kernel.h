#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the fused KV-cache-update + RoPE-on-K kernel.
//
// Output layout (destination-passing style):
//   updated_key_cache, updated_value_cache:
//     [batch, num_kv_heads, current_seq_len + new_seq_len, head_dim]
//   First current_seq_len positions = existing key_cache / value_cache.
//   Last new_seq_len positions = new key_states (with RoPE applied via
//   the given cos / sin tensors) / value_states (raw).
//
// All tensors are bfloat16 except where noted. The RoPE math is
// staged in fp32 to match the reference's pytorch-bf16 round-trip
// (pytorch dispatches bf16 * bf16 through an fp32 intermediate).
void problem_071_kv_cache_update_with_rope_launcher(
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    torch::Tensor& updated_key_cache,
    torch::Tensor& updated_value_cache,
    cudaStream_t stream);
