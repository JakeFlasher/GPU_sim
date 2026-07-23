#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <vector>

#include "kernel.h"

// Destination-passing-style entry point for Multi-head Latent Attention (MLA)
// paged decode. The behavioral contract is the FlashInfer-Bench reference at
// sol-execbench/data/benchmark/FlashInfer-Bench/018_mla_paged_decode_h16_ckv512_kpe64_ps1/reference.py.
//
//   q_nope     [B, H=16, ckv=512]  bf16   (query, no-positional part)
//   q_pe       [B, H=16, kpe=64]   bf16   (query, rope'd positional part)
//   ckv_cache  [num_pages, page_size=1, ckv=512]  bf16  (compressed latent KV)
//   kpe_cache  [num_pages, page_size=1, kpe=64]   bf16  (rope'd key-pe cache)
//   kv_indptr  [B+1]  int32  (per-sequence page-offset boundaries)
//   kv_indices [num_kv_indices]  int32  (page indices == token indices, ps=1)
//   sm_scale   scalar float32
//   output     [B, H, ckv]  bf16  (in-place)
//   lse        [B, H]        fp32  (in-place; base-2 log-sum-exp)
//
// MLA shares one compressed latent KV stream across all query heads. With
// page_size == 1 the page indices are token indices, so the per-sequence
// gather pulls L_tokens = kv_indptr[b+1] - kv_indptr[b] rows from the squeezed
// caches: Kc = ckv_cache[tok] of width 512 and Kp = kpe_cache[tok] of width 64.
//
// The score has two parts that share the same key axis: the no-positional part
// is the latent inner product q_nope . Kc over the 512-wide compressed dim, and
// the positional part is q_pe . Kp over the 64-wide rope dim. They sum into a
// single [H, L_tokens] logit before scaling. The attention value is read back
// from the SAME compressed latent Kc (out = softmax(logits) @ Kc), so the
// 512-wide output is the latent itself, not a separately projected V.
//
// All of the score math, the softmax, and the log-sum-exp run in fp32 (the
// reference upcasts q_nope/q_pe and the caches before the matmuls); only the
// final attention output narrows back to bf16. The log-sum-exp is reported in
// base two, matching the reference's division by log(2).
void run(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale,
    torch::Tensor output,
    torch::Tensor lse) {

    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be CUDA");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be CUDA");
    TORCH_CHECK(ckv_cache.is_cuda(), "ckv_cache must be CUDA");
    TORCH_CHECK(kpe_cache.is_cuda(), "kpe_cache must be CUDA");
    TORCH_CHECK(kv_indptr.is_cuda(), "kv_indptr must be CUDA");
    TORCH_CHECK(kv_indices.is_cuda(), "kv_indices must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lse must be CUDA");

    TORCH_CHECK(q_nope.dtype() == torch::kBFloat16, "q_nope must be bf16");
    TORCH_CHECK(q_pe.dtype() == torch::kBFloat16, "q_pe must be bf16");
    TORCH_CHECK(ckv_cache.dtype() == torch::kBFloat16, "ckv_cache must be bf16");
    TORCH_CHECK(kpe_cache.dtype() == torch::kBFloat16, "kpe_cache must be bf16");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(kv_indices.dtype() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bf16");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be fp32");

    TORCH_CHECK(q_nope.is_contiguous(), "q_nope must be contiguous");
    TORCH_CHECK(q_pe.is_contiguous(), "q_pe must be contiguous");
    TORCH_CHECK(ckv_cache.is_contiguous(), "ckv_cache must be contiguous");
    TORCH_CHECK(kpe_cache.is_contiguous(), "kpe_cache must be contiguous");
    TORCH_CHECK(kv_indptr.is_contiguous(), "kv_indptr must be contiguous");
    TORCH_CHECK(kv_indices.is_contiguous(), "kv_indices must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");

    TORCH_CHECK(q_nope.dim() == 3, "q_nope must be 3D");
    TORCH_CHECK(q_pe.dim() == 3, "q_pe must be 3D");
    TORCH_CHECK(ckv_cache.dim() == 3, "ckv_cache must be 3D");
    TORCH_CHECK(kpe_cache.dim() == 3, "kpe_cache must be 3D");
    TORCH_CHECK(kv_indptr.dim() == 1, "kv_indptr must be 1D");
    TORCH_CHECK(kv_indices.dim() == 1, "kv_indices must be 1D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");
    TORCH_CHECK(lse.dim() == 2, "lse must be 2D");

    const int64_t B = q_nope.size(0);
    const int64_t H = q_nope.size(1);
    const int64_t ckv = q_nope.size(2);
    const int64_t kpe = q_pe.size(2);
    TORCH_CHECK(H == 16, "num_qo_heads must equal 16");
    TORCH_CHECK(ckv == 512, "head_dim_ckv must equal 512");
    TORCH_CHECK(kpe == 64, "head_dim_kpe must equal 64");

    TORCH_CHECK(q_pe.size(0) == B && q_pe.size(1) == H,
                "q_pe shape mismatch with q_nope");
    TORCH_CHECK(ckv_cache.size(1) == 1, "ckv page_size must equal 1");
    TORCH_CHECK(kpe_cache.size(1) == 1, "kpe page_size must equal 1");
    TORCH_CHECK(ckv_cache.size(2) == ckv, "ckv_cache width mismatch");
    TORCH_CHECK(kpe_cache.size(2) == kpe, "kpe_cache width mismatch");
    TORCH_CHECK(kpe_cache.size(0) == ckv_cache.size(0),
                "ckv_cache and kpe_cache must share num_pages");

    const int64_t len_indptr = kv_indptr.size(0);
    TORCH_CHECK(len_indptr == B + 1,
                "kv_indptr length must equal batch_size + 1");
    TORCH_CHECK(output.size(0) == B && output.size(1) == H
                && output.size(2) == ckv,
                "output shape mismatch");
    TORCH_CHECK(lse.size(0) == B && lse.size(1) == H, "lse shape mismatch");

    const at::cuda::CUDAGuard device_guard(q_nope.device());

    // Seed the destination tensors with the reference's initial state: zeros
    // for the attention output and -inf for the base-2 log-sum-exp. Sequences
    // with no pages are left at these values, matching the reference's
    // `output[b].zero_()` + initial-lse fill on the empty branch.
    output.zero_();
    lse.fill_(-std::numeric_limits<float>::infinity());

    // Squeeze the page axis and upcast the caches once: Kc_all [num_pages, 512],
    // Kp_all [num_pages, 64].
    const auto Kc_all = ckv_cache.squeeze(1).to(torch::kFloat32);
    const auto Kp_all = kpe_cache.squeeze(1).to(torch::kFloat32);

    // The per-sequence page boundaries gate the loop and the gather extent, so
    // they are read on the host as concrete integers.
    const auto kv_cpu = kv_indptr.to(torch::kCPU);
    const int32_t* kv_ptr = kv_cpu.data_ptr<int32_t>();

    const double log2e = 1.0 / std::log(2.0);

    for (int64_t b = 0; b < B; ++b) {
        const int64_t page_beg = kv_ptr[b];
        const int64_t page_end = kv_ptr[b + 1];

        if (page_beg >= page_end) {
            // No KV cache for this sequence; leave zeros / -inf in place.
            continue;
        }

        // page_size == 1, so the page indices are token indices. Gather the
        // L_tokens rows for this sequence and cast the lookup to long.
        const auto tok_idx =
            kv_indices.slice(0, page_beg, page_end).to(torch::kLong);

        const auto Kc = Kc_all.index_select(0, tok_idx);  // [L_tokens, 512]
        const auto Kp = Kp_all.index_select(0, tok_idx);  // [L_tokens, 64]

        const auto qn = q_nope[b].to(torch::kFloat32);  // [H, 512]
        const auto qp = q_pe[b].to(torch::kFloat32);    // [H, 64]

        // Two-part score sharing the key axis: the no-positional part over the
        // 512-wide compressed latent plus the positional part over the 64-wide
        // rope dim, summed into [H, L_tokens] then scaled.
        const auto logits = qn.matmul(Kc.transpose(0, 1))
                                .add(qp.matmul(Kp.transpose(0, 1)));
        const auto logits_scaled = logits.mul(sm_scale);

        // Base-2 log-sum-exp over the key axis.
        const auto lse_b = at::logsumexp(logits_scaled, /*dim=*/-1).mul(log2e);
        lse[b].copy_(lse_b);

        const auto attn = at::softmax(logits_scaled, /*dim=*/-1);  // [H, L]
        // The value read-back reuses the compressed latent Kc directly.
        const auto out_b = attn.matmul(Kc).to(torch::kBFloat16);  // [H, 512]
        output[b].copy_(out_b);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MLA paged decode (problem_018); destination-passing-style; "
        "ATen-delegated; two-part nope/pe score over a shared compressed "
        "latent KV; fp32 score/softmax/LSE with bf16 inputs/outputs; "
        "base-2 log-sum-exp.");
}
