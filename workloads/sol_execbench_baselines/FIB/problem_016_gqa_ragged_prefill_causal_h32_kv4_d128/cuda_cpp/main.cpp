#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <vector>

#include "kernel.h"

// Grouped-query attention prefill over ragged (variable-length) KV, causal
// masked. This is an ATen-delegated port: every tensor computation runs through
// ATen ops on the current CUDA stream, with no custom device kernel.
//
// Shape contract (see the FlashInfer-Bench definition.json):
//   q          [total_q, H_q=32, D=128]  bf16
//   k          [total_kv, H_kv=4, D=128]  bf16
//   v          same shape as k
//   qo_indptr  [B+1]  int32  (per-sequence query token offsets)
//   kv_indptr  [B+1]  int32  (per-sequence KV token offsets)
//   sm_scale   scalar float32 (passed by value as a double)
//   output     [total_q, H_q, D]  bf16  (in-place)
//   lse        [total_q, H_q]     fp32  (in-place; base-2 log-sum-exp)
//
// Unlike the paged variant there is no page table: the KV for sequence b is the
// contiguous slab k[kv_indptr[b]:kv_indptr[b+1]] (and the matching v). The query
// rows are ragged over qo_indptr. GQA expands the H_kv key/value heads to H_q
// query heads by repeating each KV head gqa_ratio = H_q / H_kv times along the
// head axis. Scores, the softmax, and the value accumulation all run in fp32;
// only the final output is cast back to bf16. The causal mask keeps, for query
// row q_idx, the KV positions in [0, min(q_idx + 1 + delta, num_kv_tokens))
// where delta = num_kv_tokens - num_q_tokens, aligning the causal diagonal with
// the tail of the KV sequence.
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
    TORCH_CHECK(H_q == 32, "num_qo_heads must equal 32");
    TORCH_CHECK(D == 128, "head_dim must equal 128");

    const int64_t H_kv = k.size(1);
    TORCH_CHECK(H_kv == 4, "num_kv_heads must equal 4");
    TORCH_CHECK(k.size(2) == D, "k head_dim mismatch");
    TORCH_CHECK(v.size(0) == k.size(0)
                && v.size(1) == H_kv
                && v.size(2) == D,
                "v shape mismatch with k");

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
    const auto k_f32 = k.to(torch::kFloat32);  // [total_kv, H_kv, D]
    const auto v_f32 = v.to(torch::kFloat32);

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

        // Ragged KV: this sequence's tokens are the contiguous slab
        // [kv_start, kv_end) -- no page table / gather.
        const auto q_batch = q_f32.slice(0, q_start, q_end);   // [num_q, H_q, D]
        const auto k_batch = k_f32.slice(0, kv_start, kv_end); // [num_kv, H_kv, D]
        const auto v_batch = v_f32.slice(0, kv_start, kv_end);

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

        // Base-2 LSE over the KV axis. [num_q, H_q].
        const auto lse_batch = at::logsumexp(logits, /*dim=*/-1) / ln2;
        lse.slice(0, q_start, q_end).copy_(lse_batch);

        const auto attn = at::softmax(logits, /*dim=*/-1);             // [num_q, H_q, num_kv]
        const auto out_batch = at::einsum("qhk,khd->qhd", {attn, v_exp});  // [num_q, H_q, D]
        output.slice(0, q_start, q_end).copy_(out_batch.to(torch::kBFloat16));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "GQA ragged prefill, causal (problem_016); ATen-delegated; "
        "destination-passing-style; FP32 scores/softmax with BF16 inputs/outputs; "
        "base-2 LSE.");
}
