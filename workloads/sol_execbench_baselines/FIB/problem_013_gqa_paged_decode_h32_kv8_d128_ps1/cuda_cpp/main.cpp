#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <vector>

#include "kernel.h"

// Grouped-query-attention decode over a paged KV cache, expressed entirely
// through ATen tensor operators. This faithfully reproduces the reference at
// sol-execbench/data/benchmark/FlashInfer-Bench/013_gqa_paged_decode_h32_kv8_d128_ps1/reference.py.
//
//   q          [batch_size, num_qo_heads=32, head_dim=128]  bf16
//   k_cache    [num_pages, page_size=1, num_kv_heads=8, head_dim=128]  bf16
//   v_cache    same shape as k_cache  bf16
//   kv_indptr  [batch_size + 1]  int32  (per-sequence page ranges)
//   kv_indices [num_kv_indices]  int32  (page ids into the cache)
//   sm_scale   scalar float32
//   output     [batch_size, num_qo_heads, head_dim]  bf16  (in-place)
//   lse        [batch_size, num_qo_heads]            fp32  (in-place)
//
// Each query head h reads the kv head h / (num_qo_heads / num_kv_heads); with
// 32 query heads over 8 kv heads the grouping factor is four. Scores, the
// softmax, and the value-weighted reduction all run in fp32, then the context
// is narrowed back to bf16 on write. The lse is FlashInfer's base-2
// log-sum-exp: the natural-log logsumexp of the scaled scores divided by ln(2).
// Sequences with an empty page range leave their output rows zero and their
// lse rows at negative infinity, matching the reference's pre-fill.
void run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale,
    torch::Tensor output,
    torch::Tensor lse) {

    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
    TORCH_CHECK(v_cache.is_cuda(), "v_cache must be CUDA");
    TORCH_CHECK(kv_indptr.is_cuda(), "kv_indptr must be CUDA");
    TORCH_CHECK(kv_indices.is_cuda(), "kv_indices must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lse must be CUDA");

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_cache.dtype() == torch::kBFloat16, "k_cache must be bf16");
    TORCH_CHECK(v_cache.dtype() == torch::kBFloat16, "v_cache must be bf16");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(kv_indices.dtype() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bf16");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be fp32");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
    TORCH_CHECK(kv_indptr.is_contiguous(), "kv_indptr must be contiguous");
    TORCH_CHECK(kv_indices.is_contiguous(), "kv_indices must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");

    TORCH_CHECK(q.dim() == 3, "q must be 3D");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be 4D");
    TORCH_CHECK(kv_indptr.dim() == 1, "kv_indptr must be 1D");
    TORCH_CHECK(kv_indices.dim() == 1, "kv_indices must be 1D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");
    TORCH_CHECK(lse.dim() == 2, "lse must be 2D");

    const int64_t B = q.size(0);
    const int64_t H_q = q.size(1);
    const int64_t D = q.size(2);
    TORCH_CHECK(H_q == 32, "num_qo_heads must equal 32");
    TORCH_CHECK(D == 128, "head_dim must equal 128");

    TORCH_CHECK(k_cache.size(1) == 1, "page_size must equal 1");
    const int64_t H_kv = k_cache.size(2);
    TORCH_CHECK(H_kv == 8, "num_kv_heads must equal 8");
    TORCH_CHECK(k_cache.size(3) == D, "k_cache head_dim mismatch");
    TORCH_CHECK(v_cache.size(0) == k_cache.size(0)
                && v_cache.size(1) == 1
                && v_cache.size(2) == H_kv
                && v_cache.size(3) == D,
                "v_cache shape mismatch with k_cache");

    TORCH_CHECK(kv_indptr.size(0) == B + 1,
                "kv_indptr length must equal batch_size + 1");
    TORCH_CHECK(output.size(0) == B && output.size(1) == H_q
                && output.size(2) == D,
                "output shape mismatch");
    TORCH_CHECK(lse.size(0) == B && lse.size(1) == H_q,
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
    const auto kv_indptr_cpu = kv_indptr.to(torch::kCPU);
    const auto kv_ptr = kv_indptr_cpu.accessor<int32_t, 1>();

    for (int64_t b = 0; b < B; ++b) {
        const int64_t page_start = kv_ptr[b];
        const int64_t page_end = kv_ptr[b + 1];
        if (page_start >= page_end) {
            continue;
        }

        // page_size is one, so each page id is a single KV token; gather the
        // sequence's pages along the page dimension of the flattened cache.
        const auto token_indices =
            kv_indices.slice(0, page_start, page_end).to(torch::kLong);
        const int64_t num_tokens = token_indices.size(0);

        const auto k_batch = k_flat.index_select(0, token_indices);  // [T, H_kv, D]
        const auto v_batch = v_flat.index_select(0, token_indices);  // [T, H_kv, D]
        const auto q_batch = q_f32[b];                               // [H_q, D]

        // Expand each kv head across its group of query heads, yielding a
        // per-query-head view of the cache without copying the head dimension.
        const auto k_q = k_batch.unsqueeze(2)
                             .expand({num_tokens, H_kv, n_rep, D})
                             .reshape({num_tokens, H_q, D});         // [T, H_q, D]
        const auto v_q = v_batch.unsqueeze(2)
                             .expand({num_tokens, H_kv, n_rep, D})
                             .reshape({num_tokens, H_q, D});         // [T, H_q, D]

        // Per-head scores q[h] . k[:, h] reduced over the head dimension.
        const auto logits = (q_batch.unsqueeze(0) * k_q).sum(-1);    // [T, H_q]
        const auto logits_scaled = logits * sm_scale;                // [T, H_q]

        lse_acc[b] = logits_scaled.logsumexp(0) * inv_ln2;           // [H_q]

        const auto attn = logits_scaled.softmax(0);                  // [T, H_q]
        const auto out_head = (attn.unsqueeze(-1) * v_q).sum(0);     // [H_q, D]
        out_acc[b] = out_head;
    }

    output.copy_(out_acc.to(torch::kBFloat16));
    lse.copy_(lse_acc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "GQA paged decode (problem_013); ATen-delegated; destination-passing-style; "
        "fp32 softmax with bf16 inputs/outputs and base-2 log-sum-exp.");
}
