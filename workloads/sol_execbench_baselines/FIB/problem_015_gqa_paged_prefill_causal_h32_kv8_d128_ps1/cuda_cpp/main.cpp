#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <vector>

#include "kernel.h"

// Grouped-query attention prefill over a paged KV cache (page_size == 1),
// causal masked. This is an ATen-delegated port: every tensor computation runs
// through ATen ops on the current CUDA stream, with no custom device kernel.
//
// Shape contract (see the FlashInfer-Bench definition.json):
//   q          [total_q, H_q=32, D=128]  bf16
//   k_cache    [num_pages, page_size=1, H_kv=8, D=128]  bf16
//   v_cache    same shape as k_cache
//   qo_indptr  [B+1]  int32  (per-sequence query token offsets)
//   kv_indptr  [B+1]  int32  (per-sequence KV page-index offsets)
//   kv_indices [num_kv_indices]  int32  (page IDs into the KV cache)
//   sm_scale   scalar float32 (passed by value as a double)
//   output     [total_q, H_q, D]  bf16  (in-place)
//   lse        [total_q, H_q]     fp32  (in-place; base-2 log-sum-exp)
//
// The query rows are ragged over qo_indptr; the KV for sequence b is gathered
// page-by-page from the flattened cache via kv_indices[kv_indptr[b]:kv_indptr[b+1]].
// With page_size == 1 each gathered page contributes exactly one KV token.
// GQA expands the H_kv key/value heads to H_q query heads by repeating each KV
// head gqa_ratio = H_q / H_kv times along the head axis. Scores, the softmax,
// and the value accumulation all run in fp32; only the final output is cast
// back to bf16. The causal mask keeps, for query row q_idx, the KV positions in
// [0, min(q_idx + 1 + delta, num_kv_tokens)) where delta = num_kv_tokens -
// num_q_tokens, aligning the causal diagonal with the tail of the KV sequence.
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
    TORCH_CHECK(H_kv == 8, "num_kv_heads must equal 8");
    TORCH_CHECK(k_cache.size(3) == D, "k_cache head_dim mismatch");
    TORCH_CHECK(v_cache.size(0) == k_cache.size(0)
                && v_cache.size(1) == 1
                && v_cache.size(2) == H_kv
                && v_cache.size(3) == D,
                "v_cache shape mismatch with k_cache");

    TORCH_CHECK(qo_indptr.size(0) == kv_indptr.size(0),
                "qo_indptr and kv_indptr must share length");
    TORCH_CHECK(output.size(0) == total_q && output.size(1) == H_q
                && output.size(2) == D,
                "output shape mismatch");
    TORCH_CHECK(lse.size(0) == total_q && lse.size(1) == H_q,
                "lse shape mismatch");

    const c10::cuda::CUDAGuard device_guard(q.device());

    const int64_t gqa_ratio = H_q / H_kv;
    const int64_t len_indptr = qo_indptr.size(0);

    const auto q_f32 = q.to(torch::kFloat32);
    // Drop the page_size==1 axis so a single page maps to one KV token.
    const auto k_flat = k_cache.squeeze(1).to(torch::kFloat32);  // [num_pages, H_kv, D]
    const auto v_flat = v_cache.squeeze(1).to(torch::kFloat32);

    // Seed the destinations with the reference's initial fill so that empty or
    // fully-masked sequences keep zero output and -inf lse.
    output.zero_();
    lse.fill_(-std::numeric_limits<float>::infinity());

    // The indptr offsets steer host-side slicing, so pull them to the CPU once.
    const auto qo_cpu = qo_indptr.to(torch::kCPU, torch::kLong);
    const auto kv_cpu = kv_indptr.to(torch::kCPU, torch::kLong);

    const double ln2 = std::log(2.0);

    for (int64_t b = 0; b + 1 < len_indptr; ++b) {
        const int64_t q_start = qo_cpu[b].item<int64_t>();
        const int64_t q_end = qo_cpu[b + 1].item<int64_t>();
        const int64_t kv_start = kv_cpu[b].item<int64_t>();
        const int64_t kv_end = kv_cpu[b + 1].item<int64_t>();

        if (q_start >= q_end || kv_start >= kv_end) {
            continue;
        }

        // Gather this sequence's KV tokens by page id (page_size == 1 means one
        // token per page). page_ids: [num_kv_tokens].
        const auto page_ids =
            kv_indices.slice(0, kv_start, kv_end).to(torch::kLong);
        const auto k_batch = k_flat.index_select(0, page_ids);  // [num_kv, H_kv, D]
        const auto v_batch = v_flat.index_select(0, page_ids);

        const auto q_batch = q_f32.slice(0, q_start, q_end);     // [num_q, H_q, D]

        const int64_t num_q_tokens = q_batch.size(0);
        const int64_t num_kv_tokens = k_batch.size(0);
        const int64_t delta = num_kv_tokens - num_q_tokens;

        // Expand the KV heads to query heads by repeating each KV head gqa_ratio
        // times along the head axis. [num_kv, H_q, D].
        const auto k_exp = k_batch.repeat_interleave(gqa_ratio, /*dim=*/1);
        const auto v_exp = v_batch.repeat_interleave(gqa_ratio, /*dim=*/1);

        // Scores in fp32: einsum('qhd,khd->qhk'). [num_q, H_q, num_kv].
        auto logits = at::einsum("qhd,khd->qhk", {q_batch, k_exp}) * sm_scale;

        // Causal mask: keep kv_pos < q_pos + 1 + delta.
        const auto q_pos = at::arange(num_q_tokens, q_batch.options().dtype(torch::kLong));
        const auto kv_pos = at::arange(num_kv_tokens, q_batch.options().dtype(torch::kLong));
        const auto causal =
            kv_pos.unsqueeze(0) < (q_pos.unsqueeze(1) + (1 + delta));  // [num_q, num_kv]
        logits = logits.masked_fill(
            causal.logical_not().unsqueeze(1),
            -std::numeric_limits<float>::infinity());

        // A query row with no visible KV (max_kv_idx <= 0, which happens when
        // num_kv < num_q so q_pos + 1 + delta <= 0 for the leading rows) is left
        // at the seeded zero output / -inf lse by the reference, which skips it.
        // The vectorized path would instead softmax an all -inf row into NaN, so
        // mask those fully-masked rows back to the seed afterward.
        const auto row_has_kv = causal.any(/*dim=*/1);  // [num_q] bool

        // Base-2 LSE over the KV axis. [num_q, H_q].
        auto lse_batch = at::logsumexp(logits, /*dim=*/-1) / ln2;
        lse_batch = at::where(
            row_has_kv.unsqueeze(1), lse_batch,
            at::full_like(lse_batch, -std::numeric_limits<float>::infinity()));
        lse.slice(0, q_start, q_end).copy_(lse_batch);

        const auto attn = at::softmax(logits, /*dim=*/-1);             // [num_q, H_q, num_kv]
        auto out_batch = at::einsum("qhk,khd->qhd", {attn, v_exp});    // [num_q, H_q, D]
        out_batch = at::where(row_has_kv.unsqueeze(1).unsqueeze(2), out_batch,
                              at::zeros_like(out_batch));
        output.slice(0, q_start, q_end).copy_(out_batch.to(torch::kBFloat16));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "GQA paged prefill, causal (problem_015); ATen-delegated; "
        "destination-passing-style; FP32 scores/softmax with BF16 inputs/outputs; "
        "base-2 LSE.");
}
