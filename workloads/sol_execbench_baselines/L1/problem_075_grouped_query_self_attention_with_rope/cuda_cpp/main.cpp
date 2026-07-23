#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <string>

// Grouped Query Self-Attention with integrated RoPE position embeddings.
//
// Reference (problem_075_grouped_query_self_attention_with_rope), op-for-op:
//   num_heads            = 16
//   num_key_value_heads  = 4
//   head_dim             = 64
//   num_key_value_groups = num_heads // num_key_value_heads = 4
//   scaling              = 1.0  (applied to cos/sin, to scores, and as the
//                                softmax temperature -- all 1.0 -> no-ops, but
//                                reproduced literally for faithfulness)
//
//   query_states = hidden_states @ q_proj_weight.t()            # [B,S,q_proj]
//   key_states   = hidden_states @ k_proj_weight.t()            # [B,S,kv_proj]
//   value_states = hidden_states @ v_proj_weight.t()            # [B,S,kv_proj]
//
//   query_states = query_states.view(B,S,16,64).transpose(1,2)  # [B,16,S,64]
//   key_states   = key_states.view(B,S,4,64).transpose(1,2)     # [B, 4,S,64]
//   value_states = value_states.view(B,S,4,64).transpose(1,2)   # [B, 4,S,64]
//
//   inv_freq_expanded     = inv_freq[None,:,None].expand(B,-1,1)        # [B,32,1]
//   position_ids_expanded = position_ids[:,None,:].float()             # [B,1,S]
//   freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1,2) # [B,S,32]
//   emb = cat((freqs, freqs), dim=-1)                           # [B,S,64]
//   cos = emb.cos() * scaling ; sin = emb.sin() * scaling       # [B,S,64]
//   cos = cos.unsqueeze(1) ; sin = sin.unsqueeze(1)             # [B,1,S,64]
//
//   rotate_half(x) = cat((-x[...,32:], x[...,:32]), dim=-1)
//   query_states = query_states * cos + rotate_half(query_states) * sin
//   key_states   = key_states   * cos + rotate_half(key_states)   * sin
//
//   GQA repeat (num_key_value_groups > 1):
//     key_states   = key[:,:,None,:,:].expand(B,4,4,S,64).reshape(B,16,S,64)
//     value_states = value[:,:,None,:,:].expand(B,4,4,S,64).reshape(B,16,S,64)
//
//   attn_weights = (query_states @ key_states.transpose(2,3)) * scaling # [B,16,S,S]
//   if is_causal:
//       causal_mask = triu(ones(S,S, bool), diagonal=1)
//       attn_weights = attn_weights.masked_fill(causal_mask[None,None], -inf)
//   attn_weights = softmax(attn_weights, dim=-1, dtype=float32)
//   attn_output  = attn_weights @ value_states                  # [B,16,S,64]
//   attn_output  = attn_output.transpose(1,2).contiguous().view(B,S,-1) # [B,S,1024]
//   attn_output  = attn_output @ o_proj_weight.t()              # [B,S,1024]
//   return attn_output, attn_weights
//
// Per definition.json (insertion order):
//   hidden_states  : [batch_size, seq_len, hidden_size=1024]      fp32
//   position_ids   : [batch_size, seq_len]                        int64
//   q_proj_weight  : [q_proj_size=1024, hidden_size=1024]         fp32
//   k_proj_weight  : [kv_proj_size=256, hidden_size=1024]         fp32
//   v_proj_weight  : [kv_proj_size=256, hidden_size=1024]         fp32
//   o_proj_weight  : [hidden_size=1024, q_proj_size=1024]         fp32
//   inv_freq       : [32]  (head_dim // 2)                        fp32
//   is_causal      : scalar                                       bool
//   attn_output    : [batch_size, seq_len, hidden_size=1024]      fp32 DPS
//   attn_weights   : [batch_size, num_heads=16, seq_len, seq_len] fp32 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// full [B,16,S,S] score matrix is materialized explicitly because attn_weights
// is a contractual output (so a fused SDPA path, which never materializes the
// scores, is NOT usable here). All compute stays in fp32; softmax is taken with
// dtype=float32 to mirror the reference. is_causal is a runtime C++ bool
// branch. Both outputs are written in place (destination-passing style).

namespace idx = torch::indexing;

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& position_ids,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& inv_freq,
    bool is_causal,
    torch::Tensor attn_output,
    torch::Tensor attn_weights
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(q_proj_weight, "q_proj_weight");
    check_fp32_cuda(k_proj_weight, "k_proj_weight");
    check_fp32_cuda(v_proj_weight, "v_proj_weight");
    check_fp32_cuda(o_proj_weight, "o_proj_weight");
    check_fp32_cuda(inv_freq, "inv_freq");
    check_fp32_cuda(attn_output, "attn_output");
    check_fp32_cuda(attn_weights, "attn_weights");
    TORCH_CHECK(position_ids.is_cuda(), "position_ids must be a CUDA tensor");
    TORCH_CHECK(position_ids.dtype() == torch::kInt64,
                "position_ids must be int64, got ", position_ids.dtype());
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kNumKVHeads = 4;
    constexpr int64_t kHeadDim = 64;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 4
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 32
    constexpr double kScaling = 1.0;

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);

    // --- Q/K/V projections (no bias). matmul with explicit weight transpose. ---
    auto query_states = at::matmul(hidden_states, q_proj_weight.t());  // [B,S,q_proj]
    auto key_states = at::matmul(hidden_states, k_proj_weight.t());    // [B,S,kv_proj]
    auto value_states = at::matmul(hidden_states, v_proj_weight.t());  // [B,S,kv_proj]

    // --- Reshape to heads then transpose seq<->head. ---
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,16,S,64]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,4,S,64]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,4,S,64]

    // --- RoPE embeddings: freqs = inv_freq[B,32,1] @ position_ids[B,1,S]. ---
    auto inv_freq_expanded = inv_freq.index({idx::None, idx::Slice(), idx::None})
                                 .to(torch::kFloat32)
                                 .expand({batch_size, -1, 1});  // [B,32,1]
    auto position_ids_expanded =
        position_ids.index({idx::Slice(), idx::None, idx::Slice()})
            .to(torch::kFloat32);  // [B,1,S]
    auto freqs = at::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);  // [B,S,32]
    auto emb = at::cat({freqs, freqs}, -1);                  // [B,S,64]
    auto cos = (emb.cos() * kScaling).unsqueeze(1);          // [B,1,S,64]
    auto sin = (emb.sin() * kScaling).unsqueeze(1);          // [B,1,S,64]

    // --- Apply RoPE via rotate_half. ---
    auto q_half1 = query_states.index({"...", idx::Slice(idx::None, kHalfDim)});
    auto q_half2 = query_states.index({"...", idx::Slice(kHalfDim, idx::None)});
    auto q_rotated = at::cat({-q_half2, q_half1}, -1);
    query_states = (query_states * cos) + (q_rotated * sin);

    auto k_half1 = key_states.index({"...", idx::Slice(idx::None, kHalfDim)});
    auto k_half2 = key_states.index({"...", idx::Slice(kHalfDim, idx::None)});
    auto k_rotated = at::cat({-k_half2, k_half1}, -1);
    key_states = (key_states * cos) + (k_rotated * sin);

    // --- Repeat K/V for GQA: expand kv heads into query heads. ---
    if (kNumKVGroups > 1) {
        key_states = key_states
                         .index({idx::Slice(), idx::Slice(), idx::None,
                                 idx::Slice(), idx::Slice()})
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
        value_states = value_states
                           .index({idx::Slice(), idx::Slice(), idx::None,
                                   idx::Slice(), idx::Slice()})
                           .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                           .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    }

    // --- Attention scores (materialized; this IS the attn_weights output). ---
    auto scores = at::matmul(query_states, key_states.transpose(2, 3)) * kScaling;  // [B,16,S,S]

    // --- Optional causal mask: triu(ones, diagonal=1) -> -inf. ---
    if (is_causal) {
        auto causal_mask = at::triu(
            at::ones({seq_len, seq_len}, hidden_states.options().dtype(torch::kBool)),
            /*diagonal=*/1);
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            -std::numeric_limits<double>::infinity());
    }

    // --- Softmax (fp32) -> this is attn_weights. ---
    auto attn_probs = at::softmax(scores, /*dim=*/-1, /*dtype=*/torch::kFloat32);  // [B,16,S,S]

    // --- Attention output, unwind heads, output projection. ---
    auto out = at::matmul(attn_probs, value_states);  // [B,16,S,64]
    out = out.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});  // [B,S,1024]
    out = at::matmul(out, o_proj_weight.t());  // [B,S,hidden]

    // --- Destination-passing-style writes. ---
    TORCH_CHECK(out.sizes() == attn_output.sizes(),
                "attn_output shape ", out.sizes(),
                " does not match pre-allocated ", attn_output.sizes());
    TORCH_CHECK(attn_probs.sizes() == attn_weights.sizes(),
                "attn_weights shape ", attn_probs.sizes(),
                " does not match pre-allocated ", attn_weights.sizes());
    attn_output.copy_(out);
    attn_weights.copy_(attn_probs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Grouped Query Self-Attention with RoPE. Pure ATen / cuBLAS dispatch: "
          "Q/K/V projections, head reshape, RoPE (rotate_half) from inv_freq * "
          "position_ids, GQA expand (kv heads -> q heads), materialized QK^T "
          "scores (attn_weights is a contractual output), optional causal mask "
          "(runtime bool branch), fp32 softmax, attn @ V, output projection. "
          "fp32; destination-passing style (attn_output, attn_weights).");
}
