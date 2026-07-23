#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "kernel.h"

// Multi-head Latent Attention paged PREFILL with a causal mask, captured from
// DeepSeek-V3 incremental prefill (tensor-parallel size 8 -> 16 query heads).
//
// Reference (019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1), reproduced
// op-for-op. The compressed KV cache stores a single latent vector per token
// plus a separate rope component; all 16 query heads share that one latent KV
// (the MLA latent is head-shared, so there is no per-head key/value gather).
// Each query's logits are the sum of two contractions over the same set of KV
// tokens: a "nope" part over the 512-wide compressed latent and a "pe" part
// over the 64-wide rope component. The compressed latent doubles as the value,
// so the attention-weighted output sums over that same 512-wide latent.
//
//   Kc_all = ckv_cache.squeeze(1).to(fp32)      # [num_pages, 512]
//   Kp_all = kpe_cache.squeeze(1).to(fp32)      # [num_pages, 64]
//   output = zeros([total_q, 16, 512], bf16)
//   lse    = full ([total_q, 16],  -inf, fp32)
//   for b in range(batch_size):
//       q_start, q_end   = qo_indptr[b],  qo_indptr[b+1]
//       page_beg, page_end = kv_indptr[b], kv_indptr[b+1]
//       if q_start >= q_end or page_beg >= page_end: continue   # leave -inf/zeros
//       kv_len  = page_end - page_beg
//       tok_idx = kv_indices[page_beg:page_end].to(long)        # page_size==1
//       Kc = Kc_all[tok_idx]                                    # [kv_len, 512]
//       Kp = Kp_all[tok_idx]                                    # [kv_len, 64]
//       qn = q_nope[q_start:q_end].to(fp32)                     # [q_len, 16, 512]
//       qp = q_pe  [q_start:q_end].to(fp32)                     # [q_len, 16, 64]
//       for i in range(q_len):
//           logits = (qn[i] @ Kc.T) + (qp[i] @ Kp.T)            # [16, kv_len]
//           logits = logits * sm_scale
//           prefix_len    = kv_len - q_len
//           query_abs_pos = prefix_len + i
//           logits.masked_fill_(arange(kv_len) > query_abs_pos, -inf)  # causal
//           lse[q_start+i] = logsumexp(logits, -1) / log(2)            # base-2
//           attn = softmax(logits, -1)                                 # fp32
//           output[q_start+i] = (attn @ Kc).to(bf16)                   # [16, 512]
//
// Per definition.json (insertion order):
//   inputs:
//     q_nope     : [total_q, num_qo_heads=16, head_dim_ckv=512]  bf16
//     q_pe       : [total_q, num_qo_heads=16, head_dim_kpe=64]   bf16
//     ckv_cache  : [num_pages, page_size=1, head_dim_ckv=512]    bf16
//     kpe_cache  : [num_pages, page_size=1, head_dim_kpe=64]     bf16
//     qo_indptr  : [len_indptr]                                  int32
//     kv_indptr  : [len_indptr]                                  int32
//     kv_indices : [num_kv_indices]                              int32
//     sm_scale   : scalar                                        float32 (-> double)
//   outputs (DPS, pre-alloc):
//     output     : [total_q, num_qo_heads=16, head_dim_ckv=512]  bf16
//     lse        : [total_q, num_qo_heads=16]                    fp32 (base-2 LSE)
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// compressed KV is cast to fp32 once up front and the per-token gather uses
// at::index_select. The ragged batch boundaries are read on the host through
// .item<int64_t>() (matching the reference's .item() reads), and per batch
// element the whole query block is contracted at once: a single batched matmul
// over [q_len, 16, *] handles all queries, the two-part scores are summed,
// scaled, and causally masked with a [q_len, 1, kv_len] boolean built from the
// prefix_len = kv_len - q_len alignment so column j is masked when j is strictly
// past the query's absolute position. Softmax and the base-2 log-sum-exp run in
// fp32; the latent-weighted output is cast back to bf16. Empty batch elements
// are skipped, leaving the pre-filled zeros / -inf in place exactly as the
// reference does. Both outputs are written in place (destination-passing).

namespace idx = torch::indexing;

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ",
                t.dtype());
}

static void check_int32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32, got ",
                t.dtype());
}

void run(
    const torch::Tensor& q_nope,
    const torch::Tensor& q_pe,
    const torch::Tensor& ckv_cache,
    const torch::Tensor& kpe_cache,
    const torch::Tensor& qo_indptr,
    const torch::Tensor& kv_indptr,
    const torch::Tensor& kv_indices,
    double sm_scale,
    torch::Tensor output,
    torch::Tensor lse
) {
    check_bf16_cuda(q_nope, "q_nope");
    check_bf16_cuda(q_pe, "q_pe");
    check_bf16_cuda(ckv_cache, "ckv_cache");
    check_bf16_cuda(kpe_cache, "kpe_cache");
    check_int32_cuda(qo_indptr, "qo_indptr");
    check_int32_cuda(kv_indptr, "kv_indptr");
    check_int32_cuda(kv_indices, "kv_indices");
    check_bf16_cuda(output, "output");
    TORCH_CHECK(lse.is_cuda(), "lse must be a CUDA tensor");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be fp32, got ",
                lse.dtype());

    TORCH_CHECK(q_nope.dim() == 3, "q_nope must be 3D [total_q, 16, 512]");
    TORCH_CHECK(q_pe.dim() == 3, "q_pe must be 3D [total_q, 16, 64]");
    TORCH_CHECK(ckv_cache.dim() == 3, "ckv_cache must be 3D [num_pages, 1, 512]");
    TORCH_CHECK(kpe_cache.dim() == 3, "kpe_cache must be 3D [num_pages, 1, 64]");
    TORCH_CHECK(qo_indptr.dim() == 1, "qo_indptr must be 1D");
    TORCH_CHECK(kv_indptr.dim() == 1, "kv_indptr must be 1D");
    TORCH_CHECK(kv_indices.dim() == 1, "kv_indices must be 1D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D [total_q, 16, 512]");
    TORCH_CHECK(lse.dim() == 2, "lse must be 2D [total_q, 16]");

    const c10::cuda::CUDAGuard device_guard(q_nope.device());

    const int64_t total_q = q_nope.size(0);
    const int64_t num_qo_heads = q_nope.size(1);
    const int64_t head_dim_ckv = q_nope.size(2);
    const int64_t head_dim_kpe = q_pe.size(2);
    const int64_t page_size = ckv_cache.size(1);

    TORCH_CHECK(num_qo_heads == 16, "num_qo_heads must equal 16, got ",
                num_qo_heads);
    TORCH_CHECK(head_dim_ckv == 512, "head_dim_ckv must equal 512, got ",
                head_dim_ckv);
    TORCH_CHECK(head_dim_kpe == 64, "head_dim_kpe must equal 64, got ",
                head_dim_kpe);
    TORCH_CHECK(page_size == 1, "page_size must equal 1, got ", page_size);

    TORCH_CHECK(q_pe.size(0) == total_q && q_pe.size(1) == num_qo_heads,
                "q_pe leading dims must match q_nope");
    TORCH_CHECK(ckv_cache.size(2) == head_dim_ckv,
                "ckv_cache head_dim_ckv mismatch");
    TORCH_CHECK(kpe_cache.size(0) == ckv_cache.size(0)
                    && kpe_cache.size(1) == page_size
                    && kpe_cache.size(2) == head_dim_kpe,
                "kpe_cache shape mismatch with ckv_cache");
    TORCH_CHECK(qo_indptr.size(0) == kv_indptr.size(0),
                "qo_indptr and kv_indptr must have the same length");
    TORCH_CHECK(output.size(0) == total_q && output.size(1) == num_qo_heads
                    && output.size(2) == head_dim_ckv,
                "output shape mismatch");
    TORCH_CHECK(lse.size(0) == total_q && lse.size(1) == num_qo_heads,
                "lse shape mismatch");

    const int64_t len_indptr = qo_indptr.size(0);
    const int64_t batch_size = len_indptr - 1;
    TORCH_CHECK(batch_size >= 0, "qo_indptr must have at least one element");

    // The reference leaves untouched query rows as zeros / -inf, so seed the
    // destination tensors with those values up front (every row is then either
    // overwritten by an active batch element or kept at the seeded sentinel).
    output.zero_();
    lse.fill_(-std::numeric_limits<float>::infinity());

    // Compressed latent (doubles as the value) and rope key, cast to fp32 once.
    auto Kc_all = ckv_cache.squeeze(1).to(torch::kFloat32);  // [num_pages, 512]
    auto Kp_all = kpe_cache.squeeze(1).to(torch::kFloat32);  // [num_pages, 64]

    // Ragged boundaries are read on the host, matching the reference's .item().
    auto qo_indptr_cpu = qo_indptr.to(torch::kCPU);
    auto kv_indptr_cpu = kv_indptr.to(torch::kCPU);
    auto kv_indices_long = kv_indices.to(torch::kLong);  // page_size==1 -> token ids

    const double log2 = std::log(2.0);
    const auto neg_inf = -std::numeric_limits<double>::infinity();

    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t q_start = qo_indptr_cpu[b].item<int64_t>();
        const int64_t q_end = qo_indptr_cpu[b + 1].item<int64_t>();
        const int64_t page_beg = kv_indptr_cpu[b].item<int64_t>();
        const int64_t page_end = kv_indptr_cpu[b + 1].item<int64_t>();

        if (q_start >= q_end || page_beg >= page_end) {
            // No queries or no KV for this batch element: leave zeros / -inf.
            continue;
        }

        const int64_t kv_len = page_end - page_beg;
        const int64_t q_len = q_end - q_start;

        // Gather this sequence's KV tokens (pages are token indices at ps=1).
        auto tok_idx = kv_indices_long.index(
            {idx::Slice(page_beg, page_end)});
        auto Kc = at::index_select(Kc_all, 0, tok_idx);  // [kv_len, 512]
        auto Kp = at::index_select(Kp_all, 0, tok_idx);  // [kv_len, 64]

        // The dense scores tensor is [q_len, 16, kv_len] in fp32. For the largest
        // ragged sequence (q_len and kv_len both in the tens of thousands) that
        // alone is multiple GiB and overflows device memory, even though the math
        // is per-query-row independent. Tile the query axis into fixed chunks; the
        // gathered KV (Kc / Kp) is shared across chunks, and each chunk reproduces
        // the full scores -> mask -> softmax -> latent-sum path over its own rows.
        const int64_t prefix_len = kv_len - q_len;
        constexpr int64_t kQTile = 1024;
        for (int64_t qi = 0; qi < q_len; qi += kQTile) {
            const int64_t tile_rows = std::min<int64_t>(kQTile, q_len - qi);
            const int64_t row_beg = q_start + qi;
            const int64_t row_end = row_beg + tile_rows;

            auto qn = q_nope.index({idx::Slice(row_beg, row_end)}).to(torch::kFloat32);
            auto qp = q_pe.index({idx::Slice(row_beg, row_end)}).to(torch::kFloat32);
            // qn: [tile_rows, 16, 512] ; qp: [tile_rows, 16, 64]

            // Two-part scores: q_nope . Kc^T + q_pe . Kp^T, broadcasting the shared
            // (head-independent) KV across the 16 query heads. [tile_rows,16,kv].
            auto logits = at::matmul(qn, Kc.transpose(0, 1))
                          + at::matmul(qp, Kp.transpose(0, 1));
            logits = logits * sm_scale;

            // Causal mask: a key at column j is visible to query i only when its
            // index does not exceed that query's absolute position. The prefix of
            // already-cached tokens is kv_len - q_len, so the chunk's row r sits at
            // absolute position (kv_len - q_len) + qi + r and masks columns beyond.
            auto query_abs_pos =
                at::arange(tile_rows, qo_indptr_cpu.options().dtype(torch::kLong)
                                          .device(logits.device()))
                    .add(prefix_len + qi)
                    .view({tile_rows, 1, 1});  // [tile_rows, 1, 1]
            auto key_pos =
                at::arange(kv_len, query_abs_pos.options()).view({1, 1, kv_len});
            auto causal_mask = key_pos > query_abs_pos;  // [tile_rows, 1, kv_len]
            logits = logits.masked_fill(causal_mask, neg_inf);

            // Base-2 log-sum-exp over the KV axis (fp32), then softmax + latent sum.
            auto seq_lse = at::logsumexp(logits, /*dim=*/-1) / log2;  // [tile_rows,16]
            auto attn = at::softmax(logits, /*dim=*/-1);              // [tile_rows,16,kv]
            auto out = at::matmul(attn, Kc);                          // [tile_rows,16,512]

            // Destination-passing-style writes for this query chunk's rows.
            lse.index({idx::Slice(row_beg, row_end)}).copy_(seq_lse);
            output.index({idx::Slice(row_beg, row_end)}).copy_(out.to(torch::kBFloat16));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "MLA paged causal prefill (problem_019). Pure ATen / cuBLAS dispatch: "
          "the head-shared compressed latent KV is gathered per ragged sequence "
          "via index_select, the two-part (nope + pe) scores are summed, scaled "
          "by sm_scale, and causally masked from the prefix_len = kv_len - q_len "
          "alignment; fp32 softmax and base-2 log-sum-exp drive the latent sum "
          "back to bf16. Destination-passing style (output, lse).");
}
