#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <vector>

#include "kernel.h"

// Destination-passing-style entry point for ragged grouped-query-attention
// prefill with a causal mask. The behavioral contract is the FlashInfer-Bench
// reference at
// sol-execbench/data/benchmark/FlashInfer-Bench/017_gqa_ragged_prefill_causal_h32_kv8_d128/reference.py.
//
//   q          [total_q, H_q=32, D=128]   bf16
//   k          [total_kv, H_kv=8, D=128]  bf16
//   v          [total_kv, H_kv=8, D=128]  bf16
//   qo_indptr  [len_indptr]  int32   (per-sequence query token offsets)
//   kv_indptr  [len_indptr]  int32   (per-sequence key/value token offsets)
//   sm_scale   scalar float32
//   output     [total_q, H_q, D]  bf16  (in-place)
//   lse        [total_q, H_q]     fp32  (in-place; base-2 log-sum-exp)
//
// Each sequence b owns the query rows [qo_indptr[b], qo_indptr[b+1]) and the
// key/value rows [kv_indptr[b], kv_indptr[b+1]). KV is contiguous ragged (not
// paged): every sequence's KV is a plain slice. The eight KV heads broadcast
// up to the thirty-two query heads via repeat_interleave with ratio
// H_q / H_kv = 4. A query at local position i may attend to key positions
// strictly below i + 1 + delta, where delta = num_kv_tokens - num_q_tokens
// accounts for the prefix of cached keys preceding the new queries.
//
// Logits, the softmax, and the log-sum-exp run in fp32 (the reference upcasts
// q/k/v before the einsum); only the final attention output narrows back to
// bf16. The log-sum-exp is reported in base two, matching the reference's
// division by log(2).
void run(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    double sm_scale,
    torch::Tensor output,
    torch::Tensor lse) {

    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(qo_indptr.is_cuda(), "qo_indptr must be CUDA");
    TORCH_CHECK(kv_indptr.is_cuda(), "kv_indptr must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lse must be CUDA");

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bf16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bf16");
    TORCH_CHECK(qo_indptr.dtype() == torch::kInt32, "qo_indptr must be int32");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bf16");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be fp32");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(qo_indptr.is_contiguous(), "qo_indptr must be contiguous");
    TORCH_CHECK(kv_indptr.is_contiguous(), "kv_indptr must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");

    TORCH_CHECK(q.dim() == 3, "q must be 3D");
    TORCH_CHECK(k.dim() == 3, "k must be 3D");
    TORCH_CHECK(v.dim() == 3, "v must be 3D");
    TORCH_CHECK(qo_indptr.dim() == 1, "qo_indptr must be 1D");
    TORCH_CHECK(kv_indptr.dim() == 1, "kv_indptr must be 1D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");
    TORCH_CHECK(lse.dim() == 2, "lse must be 2D");

    const int64_t total_q = q.size(0);
    const int64_t H_q = q.size(1);
    const int64_t D = q.size(2);
    const int64_t total_kv = k.size(0);
    const int64_t H_kv = k.size(1);
    TORCH_CHECK(H_q == 32, "num_qo_heads must equal 32");
    TORCH_CHECK(H_kv == 8, "num_kv_heads must equal 8");
    TORCH_CHECK(D == 128, "head_dim must equal 128");
    TORCH_CHECK(k.size(2) == D && v.size(2) == D, "kv head_dim mismatch");
    TORCH_CHECK(v.size(0) == total_kv && v.size(1) == H_kv,
                "v shape mismatch with k");

    const int64_t len_indptr = qo_indptr.size(0);
    TORCH_CHECK(kv_indptr.size(0) == len_indptr,
                "qo_indptr and kv_indptr must share length");
    TORCH_CHECK(output.size(0) == total_q && output.size(1) == H_q
                && output.size(2) == D,
                "output shape mismatch");
    TORCH_CHECK(lse.size(0) == total_q && lse.size(1) == H_q,
                "lse shape mismatch");

    const at::cuda::CUDAGuard device_guard(q.device());

    const int64_t gqa_ratio = H_q / H_kv;

    // Seed the destination tensors with the reference's initial state: zeros
    // for the attention output and -inf for the base-2 log-sum-exp. Sequences
    // with no queries or no keys are left at these values, exactly as the
    // reference's `continue` branch does.
    output.zero_();
    lse.fill_(-std::numeric_limits<float>::infinity());

    const auto q_f32 = q.to(torch::kFloat32);
    const auto k_f32 = k.to(torch::kFloat32);
    const auto v_f32 = v.to(torch::kFloat32);

    // The per-sequence offsets gate the loop structure and the ragged slice
    // bounds, so they are read on the host as concrete integers.
    const auto qo_cpu = qo_indptr.to(torch::kCPU);
    const auto kv_cpu = kv_indptr.to(torch::kCPU);
    const int32_t* qo_ptr = qo_cpu.data_ptr<int32_t>();
    const int32_t* kv_ptr = kv_cpu.data_ptr<int32_t>();

    const double log2e = 1.0 / std::log(2.0);

    const auto long_opts = q.options().dtype(torch::kLong);

    for (int64_t b = 0; b + 1 < len_indptr; ++b) {
        const int64_t q_start = qo_ptr[b];
        const int64_t q_end = qo_ptr[b + 1];
        const int64_t kv_start = kv_ptr[b];
        const int64_t kv_end = kv_ptr[b + 1];

        if (q_start >= q_end || kv_start >= kv_end) {
            // No queries or KV for this sequence; leave zeros / -inf in place.
            continue;
        }

        const int64_t num_q_tokens = q_end - q_start;
        const int64_t num_kv_tokens = kv_end - kv_start;
        const int64_t delta = num_kv_tokens - num_q_tokens;

        // [num_q_tokens, H_q, D]
        const auto q_batch = q_f32.slice(0, q_start, q_end);
        // [num_kv_tokens, H_kv, D] broadcast up to H_q query heads.
        const auto k_batch = k_f32.slice(0, kv_start, kv_end);
        const auto v_batch = v_f32.slice(0, kv_start, kv_end);
        const auto k_expanded = k_batch.repeat_interleave(gqa_ratio, /*dim=*/1);
        const auto v_expanded = v_batch.repeat_interleave(gqa_ratio, /*dim=*/1);

        // logits[q, h, k] = (sum_d q[q,h,d] * k[k,h,d]) * sm_scale
        auto logits = at::einsum("qhd,khd->qhk", {q_batch, k_expanded})
                          .mul(sm_scale);

        // Causal mask: key position kk is visible to query position qq when
        // kk < qq + 1 + delta. Build the [num_q_tokens, num_kv_tokens] mask
        // and broadcast over the head axis.
        const auto q_pos = at::arange(num_q_tokens, long_opts);
        const auto kv_pos = at::arange(num_kv_tokens, long_opts);
        const auto causal = kv_pos.unsqueeze(0)
                                .lt(q_pos.unsqueeze(1).add(1 + delta));
        const auto masked_out = causal.logical_not().unsqueeze(1);
        logits = logits.masked_fill(
            masked_out, -std::numeric_limits<float>::infinity());

        // Base-2 log-sum-exp over the key axis.
        const auto lse_batch = at::logsumexp(logits, /*dim=*/-1).mul(log2e);
        lse.slice(0, q_start, q_end).copy_(lse_batch);

        const auto attn = at::softmax(logits, /*dim=*/-1);
        const auto out_batch =
            at::einsum("qhk,khd->qhd", {attn, v_expanded}).to(torch::kBFloat16);
        output.slice(0, q_start, q_end).copy_(out_batch);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "GQA ragged causal prefill (problem_017); destination-passing-style; "
        "ATen-delegated; fp32 logits/softmax/LSE with bf16 inputs/outputs; "
        "base-2 log-sum-exp.");
}
