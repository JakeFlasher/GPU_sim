#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the fused RMS-norm + RoPE + KV-cache-scatter kernel.
//
// All tensors are bfloat16 except position_ids / cache_position (int64),
// inv_freq (fp32), and rms_norm_eps (fp32 scalar). The launcher:
//   1. copies key_cache → key_cache_out and value_cache → value_cache_out
//      across the full max_position_embeddings range (cudaMemcpyAsync);
//   2. applies per-head RMS norm to query / key with their respective
//      weight tensors (fp32 staging for the reduction; bf16 for the
//      output);
//   3. builds cos/sin inline from inv_freq * position_ids per (batch,
//      seq, dim) and applies RoPE rotation to the RMS-normed Q / K;
//   4. scatters the rotated key_rotated into key_cache_out at the
//      cache_position[s] slot for each output seq position s, and
//      similarly scatters value into value_cache_out.
//
// The bf16 multiply-adds in the RoPE step round each multiply result
// to bf16 before the add to bit-match the reference's pytorch
// bf16-mul + bf16-add round-trip semantics.
void problem_018_fused_rope_with_qk_norm_and_kv_cache_update_launcher(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& position_ids,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& inv_freq,
    float rms_norm_eps,
    torch::Tensor& query_rotated,
    torch::Tensor& key_rotated,
    torch::Tensor& key_cache_out,
    torch::Tensor& value_cache_out,
    cudaStream_t stream);
