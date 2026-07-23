#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>
#include <cmath>
#include <limits>

// Conformer attention with Shaw's relative positional embeddings. The sequence
// is pre-normalized (LayerNorm), padded up to a multiple of the context size,
// projected to Q / K / V, reshaped into fixed-size context blocks, and attended
// block-locally. A learned relative-position bias is added to the scores before
// the softmax, the incomplete final block is masked out, and the result is
// un-padded and projected to the output.
//
// Reference (problem_046_conformer_relative_position_attention), op-for-op:
//   hidden_dim    = 1024   num_heads     = 8     dim_head     = 128
//   max_pos_emb   = 512    context_size  = 512   inner_dim    = 1024
//   inner_dim_x2  = 2048   num_rel_pos   = 1025  (== 2*max_pos_emb + 1)
//
//   bsz, num_features = hidden_states.shape[:2]
//   hidden_states = layer_norm(hidden_states, [hidden_dim], pre_norm_weight, pre_norm_bias)
//   num_blocks = ceil(num_features / context_size)
//   remainder  = num_features % context_size
//   if remainder > 0:
//       hidden_states = pad(hidden_states, (0,0,0, context_size - remainder))
//   q  = linear(hidden_states, to_q_weight)            # [B, S_pad, inner_dim]
//   kv = linear(hidden_states, to_kv_weight)           # [B, S_pad, inner_dim*2]
//   k, v = kv.chunk(2, dim=-1)
//   reshape each into [B, num_blocks, context_size, num_heads, dim_head]
//       then transpose(2,3) -> [B, num_blocks, num_heads, context_size, dim_head]
//
//   # relative position distances within one context window
//   seq = arange(context_size)
//   relpos_dist = seq[:,None] - seq[None,:]                      # [C, C]
//   attention_dists = clamp(relpos_dist, -context_size, context_size) + max_pos_emb
//   rel_pos_emb = embedding(attention_dists, rel_pos_emb_weight)  # [C, C, dim_head]
//
//   # positional attention bias, contracted over dim_head
//   pos_attn = einsum('bmhcd,crd->bmhcr', q.float(), rel_pos_emb.float()) * scale
//
//   if remainder > 0:
//       mask = ones(C, C, bool) ; mask[:remainder, :remainder] = False
//       pos_attn[:, -1, :].masked_fill_(mask, -finfo(pos_attn.dtype).max)
//   pos_attn = pos_attn.to(q.dtype)
//
//   # block-local scaled dot-product attention with the positional bias
//   out = softmax(q @ k.transpose(-2,-1) * scale + pos_attn) @ v
//   out = out.transpose(2,3).reshape(B, num_blocks*context_size, inner_dim)
//   out = out[:, :num_features, :]
//   out = linear(out, to_out_weight, to_out_bias)
//
// Per definition.json (insertion order):
//   hidden_states     : [B, S, hidden_dim]        bf16
//   pre_norm_weight   : [hidden_dim]              bf16
//   pre_norm_bias     : [hidden_dim]              bf16
//   to_q_weight       : [inner_dim, hidden_dim]   bf16
//   to_kv_weight      : [inner_dim*2, hidden_dim] bf16
//   to_out_weight     : [hidden_dim, inner_dim]   bf16
//   to_out_bias       : [hidden_dim]              bf16
//   rel_pos_emb_weight: [num_rel_pos, dim_head]   bf16
//   scale             : scalar                    fp32 -> double
//   output            : [B, S, hidden_dim]        bf16 DPS
//
// Notes on fidelity:
//   * The reference forces the MATH SDPA backend so the additive bias is applied
//     exactly. This port spells the attention out explicitly -- scores =
//     (q @ k^T) * scale, plus the positional bias, softmax, then @ v -- so the
//     relative-position bias add and the final-block masking are faithful rather
//     than left to a fused SDPA path.
//   * The einsum 'bmhcd,crd->bmhcr' contracts the head dim d between the per-
//     block query [B,M,Nh,C,D] and the relative-position embedding [C,C,D],
//     producing one bias score per (query position c, relative position r). It
//     runs in fp32 (q.float(), rel_pos_emb.float()) and is scaled by `scale`;
//     the result is cast back to the query's bf16 dtype before the softmax.
//   * attention_dists indexes rel_pos_emb_weight via clamp(i - j, -C, C) +
//     max_pos_emb. With context_size == max_pos_emb == 512 the clamp is inert on
//     the in-range distances but is kept to mirror the reference exactly; the
//     +max_pos_emb shift maps the signed distance into [0, num_rel_pos).
//   * Final-block masking only fires when seq_len is not a multiple of the
//     context size (host-side `if remainder > 0`). It fills the padded query /
//     key positions of the last block with -finfo(bf16).max before the softmax
//     so the padding contributes nothing.
//   * Precision: inputs and outputs are bf16. LayerNorm and the softmax run in
//     fp32; the positional-bias einsum runs in fp32 and is cast back to bf16.
//     The bf16 output is never upcast; bf16 is the contractual dtype, not a
//     downgrade.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& pre_norm_weight,
    const torch::Tensor& pre_norm_bias,
    const torch::Tensor& to_q_weight,
    const torch::Tensor& to_kv_weight,
    const torch::Tensor& to_out_weight,
    const torch::Tensor& to_out_bias,
    const torch::Tensor& rel_pos_emb_weight,
    double scale,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(pre_norm_weight, "pre_norm_weight");
    check_bf16_cuda(pre_norm_bias, "pre_norm_bias");
    check_bf16_cuda(to_q_weight, "to_q_weight");
    check_bf16_cuda(to_kv_weight, "to_kv_weight");
    check_bf16_cuda(to_out_weight, "to_out_weight");
    check_bf16_cuda(to_out_bias, "to_out_bias");
    check_bf16_cuda(rel_pos_emb_weight, "rel_pos_emb_weight");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden_dim], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenDim = 1024;
    constexpr int64_t kNumHeads = 8;
    constexpr int64_t kDimHead = 128;
    constexpr int64_t kMaxPosEmb = 512;
    constexpr int64_t kContextSize = 512;
    constexpr int64_t kInnerDim = 1024;

    const int64_t batch_size = hidden_states.size(0);
    const int64_t num_features = hidden_states.size(1);

    // Pre-normalization (LayerNorm over the hidden_dim), accumulated in fp32 by
    // ATen and returned in the input bf16 dtype.
    auto normed = at::layer_norm(
        hidden_states, /*normalized_shape=*/std::vector<int64_t>{kHiddenDim},
        pre_norm_weight, pre_norm_bias);

    const int64_t num_blocks = (num_features + kContextSize - 1) / kContextSize;
    const int64_t remainder = num_features % kContextSize;

    // Pad up to the block boundary when the sequence does not fill the last
    // block. F.pad((0,0,0,pad_amount)) pads the sequence axis (dim 1) on the
    // right with zeros.
    if (remainder > 0) {
        const int64_t pad_amount = kContextSize - remainder;
        normed = at::constant_pad_nd(
            normed, /*pad=*/std::vector<int64_t>{0, 0, 0, pad_amount}, /*value=*/0);
    }
    const int64_t seq_padded = num_blocks * kContextSize;

    // Project to Q / K / V. to_kv emits the concatenated K and V, split by chunk.
    auto query_states = at::matmul(normed, to_q_weight.t());      // [B, S_pad, inner_dim]
    auto key_value_states = at::matmul(normed, to_kv_weight.t()); // [B, S_pad, inner_dim*2]
    auto kv_chunks = key_value_states.chunk(/*chunks=*/2, /*dim=*/-1);
    auto key_states = kv_chunks[0];    // [B, S_pad, inner_dim]
    auto value_states = kv_chunks[1];  // [B, S_pad, inner_dim]

    // Reshape into blocks: [B, num_blocks, context_size, num_heads, dim_head]
    // then transpose to [B, num_blocks, num_heads, context_size, dim_head].
    auto to_blocks = [&](const torch::Tensor& t) {
        return t.reshape({batch_size, num_blocks, kContextSize, kNumHeads, kDimHead})
            .transpose(2, 3);
    };
    query_states = to_blocks(query_states);  // [B, M, Nh, C, D]
    key_states = to_blocks(key_states);
    value_states = to_blocks(value_states);

    // Relative-position distances within one context window: clamp(i - j) then
    // shift by max_pos_emb into the embedding-row range [0, num_rel_pos).
    auto idx_opts = hidden_states.options().dtype(torch::kLong);
    auto seq = at::arange(kContextSize, idx_opts);                 // [C]
    auto relpos_dist = seq.view({-1, 1}) - seq.view({1, -1});      // [C, C]
    auto attention_dists =
        at::clamp(relpos_dist, -kContextSize, kContextSize) + kMaxPosEmb;  // [C, C]
    auto rel_pos_emb = at::embedding(rel_pos_emb_weight, attention_dists);  // [C, C, D]

    // Positional attention bias, contracted over the head dim in fp32:
    // pos_attn[b,m,h,c,r] = sum_d query[b,m,h,c,d] * rel_pos_emb[c,r,d].
    auto pos_attn = at::einsum(
        "bmhcd,crd->bmhcr",
        {query_states.to(torch::kFloat32), rel_pos_emb.to(torch::kFloat32)});
    pos_attn = pos_attn * scale;  // [B, M, Nh, C, C]

    // Mask the incomplete final block: positions outside the real remainder are
    // filled with the most-negative bf16 value so they vanish under softmax.
    if (remainder > 0) {
        auto mask = at::ones({kContextSize, kContextSize},
                             hidden_states.options().dtype(torch::kBool));
        mask.narrow(0, 0, remainder).narrow(1, 0, remainder).fill_(false);
        const double mask_value =
            -static_cast<double>(std::numeric_limits<float>::max());
        // pos_attn[:, -1, :] -> the last block across all batches and heads.
        auto last_block = pos_attn.select(/*dim=*/1, /*index=*/num_blocks - 1);  // [B, Nh, C, C]
        last_block.masked_fill_(mask, mask_value);
    }
    pos_attn = pos_attn.to(query_states.scalar_type());

    // Block-local scaled dot-product attention with the positional bias added
    // before the softmax (MATH-backend semantics spelled out explicitly).
    auto scores = at::matmul(query_states, key_states.transpose(-2, -1)) * scale;  // [B,M,Nh,C,C]
    scores = scores + pos_attn;
    auto attn = at::softmax(scores, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                    .to(query_states.scalar_type());
    auto out = at::matmul(attn, value_states);  // [B, M, Nh, C, D]

    // Reshape back to [B, num_blocks*context_size, inner_dim], drop the padding.
    out = out.transpose(2, 3).reshape({batch_size, seq_padded, kInnerDim});
    out = out.narrow(/*dim=*/1, /*start=*/0, /*length=*/num_features);

    // Output projection (with bias).
    auto result = at::linear(out, to_out_weight, to_out_bias);  // [B, S, hidden_dim]

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Conformer block-local attention with Shaw relative-position bias via "
          "ATen (pre-LayerNorm, context-block reshape, embedding/einsum "
          "positional bias, incomplete-block masking, explicit softmax "
          "attention; no custom CUDA kernels); bf16 in/out with fp32 "
          "norm/softmax/einsum accumulation; destination-passing style "
          "(output).");
}
