#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// GQA paged decode kernel launcher. One block per
// (batch, query head); 128 threads per block.
//
//   q          : [B, H_q=32, D=128]  bf16
//   k_cache    : [num_pages, page_size=1, H_kv=4, D=128]  bf16
//   v_cache    : same shape as k_cache
//   kv_indptr  : [B+1] int32 (per-batch token range start/end)
//   kv_indices : [N] int32 (token indices into k_cache / v_cache)
//   sm_scale   : float
//   output     : [B, H_q, D] bf16 (in-place)
//   lse        : [B, H_q] float32 (in-place); base-2 log-sum-exp
//
// Algorithm:
//   per (b, h):
//     kv_head = h / gqa_ratio  (gqa_ratio = H_q / H_kv = 8)
//     for token i in kv_indices[kv_indptr[b] : kv_indptr[b+1]]:
//       logit_i = (q[b, h] @ k_cache[i, kv_head]) * sm_scale
//     attn = softmax(logits)
//     output[b, h] = attn @ v_cache[token, kv_head]
//     lse[b, h]    = log2(sum(exp(logits)))
//
// Empty-token batches (kv_indptr[b] == kv_indptr[b+1])
// produce zero output and lse = -inf (matches the
// reference's `output[b].zero_()` + initial-lse fill).
void problem_012_gqa_paged_decode_launcher(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& kv_indptr,
    const torch::Tensor& kv_indices,
    float sm_scale,
    torch::Tensor output,
    torch::Tensor lse,
    cudaStream_t stream);
