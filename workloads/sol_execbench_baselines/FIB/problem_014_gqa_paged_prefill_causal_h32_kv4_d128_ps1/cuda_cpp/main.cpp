#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kernel.h"

// Causal grouped-query-attention prefill over a paged KV cache, expressed
// entirely through ATen tensor operators. This faithfully reproduces the
// reference at
// sol-execbench/data/benchmark/FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1/reference.py.
//
//   q          [total_q, num_qo_heads=32, head_dim=128]  bf16  (ragged over sequences)
//   k_cache    [num_pages, page_size=1, num_kv_heads=4, head_dim=128]  bf16
//   v_cache    same shape as k_cache  bf16
//   qo_indptr  [batch_size + 1]  int32  (per-sequence query ranges)
//   kv_indptr  [batch_size + 1]  int32  (per-sequence page ranges)
//   kv_indices [num_kv_indices]  int32  (page ids into the cache)
//   sm_scale   scalar float32
//   output     [total_q, num_qo_heads, head_dim]  bf16  (in-place)
//   lse        [total_q, num_qo_heads]            fp32  (in-place)
//
// Queries are packed contiguously; sequence b owns rows qo_indptr[b] ..
// qo_indptr[b+1] and pages kv_indices[kv_indptr[b] .. kv_indptr[b+1]]. Each
// query head h reads kv head h / (num_qo_heads / num_kv_heads); with 32 query
// heads over 4 kv heads the grouping factor is eight. The causal mask is
// right-aligned: with delta = num_kv_tokens - num_q_tokens, query position
// q_idx attends only the first min(q_idx + 1 + delta, num_kv_tokens) kv tokens,
// so the final query sees the whole sequence and earlier queries see a prefix.
// Query positions whose visible window is empty leave their output rows zero
// and their lse rows at negative infinity, matching the reference's pre-fill.
// Scores, softmax, and the value-weighted reduction run in fp32, narrowing to
// bf16 only on the output write. The lse is FlashInfer's base-2 log-sum-exp:
// the natural-log logsumexp of the scaled scores divided by ln(2).
void run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale,
    torch::Tensor output,
    torch::Tensor lse) {

    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
    TORCH_CHECK(v_cache.is_cuda(), "v_cache must be CUDA");
    TORCH_CHECK(qo_indptr.is_cuda(), "qo_indptr must be CUDA");
    TORCH_CHECK(kv_indptr.is_cuda(), "kv_indptr must be CUDA");
    TORCH_CHECK(kv_indices.is_cuda(), "kv_indices must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lse must be CUDA");

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_cache.dtype() == torch::kBFloat16, "k_cache must be bf16");
    TORCH_CHECK(v_cache.dtype() == torch::kBFloat16, "v_cache must be bf16");
    TORCH_CHECK(qo_indptr.dtype() == torch::kInt32, "qo_indptr must be int32");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(kv_indices.dtype() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bf16");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be fp32");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
    TORCH_CHECK(qo_indptr.is_contiguous(), "qo_indptr must be contiguous");
    TORCH_CHECK(kv_indptr.is_contiguous(), "kv_indptr must be contiguous");
    TORCH_CHECK(kv_indices.is_contiguous(), "kv_indices must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");

    TORCH_CHECK(q.dim() == 3, "q must be 3D");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be 4D");
    TORCH_CHECK(qo_indptr.dim() == 1, "qo_indptr must be 1D");
    TORCH_CHECK(kv_indptr.dim() == 1, "kv_indptr must be 1D");
    TORCH_CHECK(kv_indices.dim() == 1, "kv_indices must be 1D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");
    TORCH_CHECK(lse.dim() == 2, "lse must be 2D");

    const int64_t total_q = q.size(0);
    const int64_t H_q = q.size(1);
    const int64_t D = q.size(2);
    TORCH_CHECK(H_q == 32, "num_qo_heads must equal 32");
    TORCH_CHECK(D == 128, "head_dim must equal 128");

    TORCH_CHECK(k_cache.size(1) == 1, "page_size must equal 1");
    const int64_t H_kv = k_cache.size(2);
    TORCH_CHECK(H_kv == 4, "num_kv_heads must equal 4");
    TORCH_CHECK(k_cache.size(3) == D, "k_cache head_dim mismatch");
    TORCH_CHECK(v_cache.size(0) == k_cache.size(0)
                && v_cache.size(1) == 1
                && v_cache.size(2) == H_kv
                && v_cache.size(3) == D,
                "v_cache shape mismatch with k_cache");

    const int64_t len_indptr = qo_indptr.size(0);
    TORCH_CHECK(len_indptr >= 1, "qo_indptr must be non-empty");
    TORCH_CHECK(kv_indptr.size(0) == len_indptr,
                "qo_indptr and kv_indptr lengths must match");
    TORCH_CHECK(output.size(0) == total_q && output.size(1) == H_q
                && output.size(2) == D,
                "output shape mismatch");
    TORCH_CHECK(lse.size(0) == total_q && lse.size(1) == H_q,
                "lse shape mismatch");

    const c10::cuda::CUDAGuard device_guard(q.device());

    const int64_t n_rep = H_q / H_kv;
    const double inv_ln2 = 1.0 / std::log(2.0);

    // Collapse the unit page dimension and promote to fp32; every score and
    // accumulation below stays in fp32 to mirror the reference.
    const auto k_flat = k_cache.squeeze(1).to(torch::kFloat32);
    const auto v_flat = v_cache.squeeze(1).to(torch::kFloat32);
    const auto q_f32 = q.to(torch::kFloat32);

    auto out_acc = torch::zeros_like(output, output.options().dtype(torch::kFloat32));
    auto lse_acc = torch::full_like(lse, -std::numeric_limits<float>::infinity());

    // Sequence boundaries are read on the host; the batch is small so this loop
    // is cheap and keeps the per-sequence gather shapes explicit.
    const auto qo_indptr_cpu = qo_indptr.to(torch::kCPU);
    const auto kv_indptr_cpu = kv_indptr.to(torch::kCPU);
    const auto qo_ptr = qo_indptr_cpu.accessor<int32_t, 1>();
    const auto kv_ptr = kv_indptr_cpu.accessor<int32_t, 1>();

    for (int64_t b = 0; b < len_indptr - 1; ++b) {
        const int64_t q_start = qo_ptr[b];
        const int64_t q_end = qo_ptr[b + 1];
        const int64_t kv_start = kv_ptr[b];
        const int64_t kv_end = kv_ptr[b + 1];
        if (q_start >= q_end || kv_start >= kv_end) {
            continue;
        }

        // page_size is one, so each page id is a single KV token; gather the
        // sequence's pages along the page dimension of the flattened cache.
        const auto page_ids =
            kv_indices.slice(0, kv_start, kv_end).to(torch::kLong);
        const int64_t num_kv_tokens = page_ids.size(0);

        const auto k_batch = k_flat.index_select(0, page_ids);  // [Tk, H_kv, D]
        const auto v_batch = v_flat.index_select(0, page_ids);  // [Tk, H_kv, D]

        // Expand each kv head across its group of query heads, yielding a
        // per-query-head view of the cache without copying the head dimension.
        const auto k_q = k_batch.unsqueeze(2)
                             .expand({num_kv_tokens, H_kv, n_rep, D})
                             .reshape({num_kv_tokens, H_q, D});  // [Tk, H_q, D]
        const auto v_q = v_batch.unsqueeze(2)
                             .expand({num_kv_tokens, H_kv, n_rep, D})
                             .reshape({num_kv_tokens, H_q, D});  // [Tk, H_q, D]

        const int64_t num_q_tokens = q_end - q_start;
        const int64_t delta = num_kv_tokens - num_q_tokens;

        for (int64_t q_idx = 0; q_idx < num_q_tokens; ++q_idx) {
            const int64_t global_q_idx = q_start + q_idx;

            // Right-aligned causal window: this query sees a prefix of the kv
            // sequence whose length grows by one per query position.
            const int64_t max_kv_idx =
                std::min<int64_t>(q_idx + 1 + delta, num_kv_tokens);
            if (max_kv_idx <= 0) {
                continue;
            }

            const auto q_pos = q_f32[global_q_idx];                  // [H_q, D]
            const auto k_vis = k_q.slice(0, 0, max_kv_idx);          // [m, H_q, D]
            const auto v_vis = v_q.slice(0, 0, max_kv_idx);          // [m, H_q, D]

            // Per-head scores q[h] . k[:, h] reduced over the head dimension.
            const auto logits = (q_pos.unsqueeze(0) * k_vis).sum(-1);  // [m, H_q]
            const auto logits_scaled = logits * sm_scale;              // [m, H_q]

            lse_acc[global_q_idx] = logits_scaled.logsumexp(0) * inv_ln2;  // [H_q]

            const auto attn = logits_scaled.softmax(0);                // [m, H_q]
            const auto out_head = (attn.unsqueeze(-1) * v_vis).sum(0);  // [H_q, D]
            out_acc[global_q_idx] = out_head;
        }
    }

    output.copy_(out_acc.to(torch::kBFloat16));
    lse.copy_(lse_acc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "GQA paged causal prefill (problem_014); ATen-delegated; "
        "destination-passing-style; fp32 softmax with bf16 inputs/outputs and "
        "base-2 log-sum-exp.");
}
