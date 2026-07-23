#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>
#include <cmath>

// KV-shared attention with dual RoPE for Gemma3n. 32 query heads, 8 key-value
// heads (4:1 GQA ratio), head dim 96. Q and K are each RMS-normalized in fp32
// with a learned per-head-dim norm weight; V is RMS-normalized in fp32 without
// a scale. RoPE is built from the rope_theta base and applied to Q and K on the
// pre-transpose [B, S, heads, head_dim] layout. Attention logits are tanh
// soft-capped (softcap * tanh(scores / softcap)), the additive causal mask is
// applied, softmax runs in fp32, and the value matmul is followed by the output
// projection. A host-side branch on use_shared_kv reuses the supplied shared
// K / V states (already normalized / RoPE'd by an earlier layer) instead of
// computing them from the K / V projections.
//
// Reference (problem_041_kv_shared_attention_with_dual_rope), op-for-op:
//   num_attention_heads  = 32
//   num_key_value_heads  = 8
//   head_dim             = 96
//   num_key_value_groups = 4
//
//   query_states = linear(hidden_states, q_proj_weight)        # [B,S,q_proj]
//   query_states = query_states.view(B,S,num_heads,head_dim)
//   q_var = query_states.float().pow(2).mean(-1, keepdim=True)
//   query_states = query_states * rsqrt(q_var + eps)
//   query_states = (query_states * q_norm_weight).to(bf16)
//
//   # RoPE built from rope_theta and the supplied position_ids:
//   inv_freq = 1 / (rope_theta ** (arange(0, head_dim, 2).float() / head_dim))
//   inv_freq_expanded     = inv_freq[None,:,None].expand(B,-1,1)
//   position_ids_expanded = position_ids[:,None,:].float()
//   freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
//   emb   = cat((freqs, freqs), dim=-1)
//   cos = emb.cos().to(bf16) ; sin = emb.sin().to(bf16)        # [B,S,head_dim]
//   # Applied on the [B,S,heads,head_dim] layout with cos/sin.unsqueeze(2):
//   rotate_half(x) = cat((-x[..., hd/2:], x[..., :hd/2]), dim=-1)
//   query_states = query_states*cos_u + rotate_half(query_states)*sin_u
//   query_states = query_states.transpose(1, 2)               # [B,heads,S,hd]
//
//   if use_shared_kv:
//       key_states   = shared_key_states                       # [B,kv,S,hd]
//       value_states = shared_value_states                     # [B,kv,S,hd]
//   else:
//       key_states = linear(hidden_states, k_proj_weight).view(B,S,kv,hd)
//       k_var = key_states.float().pow(2).mean(-1, keepdim=True)
//       key_states = key_states * rsqrt(k_var + eps)
//       key_states = (key_states * k_norm_weight).to(bf16)
//       key_states = key_states*cos_u + rotate_half(key_states)*sin_u
//       key_states = key_states.transpose(1, 2)                # [B,kv,S,hd]
//       value_states = linear(hidden_states, v_proj_weight).view(B,S,kv,hd)
//       v_var = value_states.float().pow(2).mean(-1, keepdim=True)
//       value_states = (value_states * rsqrt(v_var + eps)).to(bf16)  # no scale
//       value_states = value_states.transpose(1, 2)            # [B,kv,S,hd]
//
//   # repeat_kv: expand the 8 kv heads into 32 query heads.
//   attn = query_states @ key_states.transpose(2, 3)           # [B,32,S,S]
//   attn = softcap * tanh(attn / softcap)                      # logit soft-cap
//   attn = attn + attention_mask                               # additive mask
//   attn = softmax(attn, dim=-1, dtype=fp32).to(bf16)
//   attn_output = attn @ value_states                          # [B,32,S,hd]
//   attn_output = attn_output.transpose(1,2).contiguous().view(B,S,q_proj)
//   attn_output = linear(attn_output, o_proj_weight)           # [B,S,hidden]
//   return attn_output
//
// Per definition.json (insertion order):
//   hidden_states       : [B, S, hidden=3072]              bf16
//   position_ids        : [B, S]                            int64
//   attention_mask      : [B, 1, S, S]                      bf16
//   q_proj_weight       : [q_proj_dim=3072, hidden]        bf16
//   k_proj_weight       : [kv_proj_dim=768, hidden]        bf16
//   v_proj_weight       : [kv_proj_dim=768, hidden]        bf16
//   o_proj_weight       : [hidden, q_proj_dim]             bf16
//   q_norm_weight       : [head_dim=96]                    bf16
//   k_norm_weight       : [head_dim=96]                    bf16
//   shared_key_states   : [B, num_kv_heads=8, S, head_dim] bf16
//   shared_value_states : [B, num_kv_heads=8, S, head_dim] bf16
//   rope_theta          : scalar                            fp32 -> double
//   softcap             : scalar                            fp32 -> double
//   rms_norm_eps        : scalar                            fp32 -> double
//   use_shared_kv       : scalar                            bool
//   attn_output         : [B, S, hidden]                    bf16 DPS
//
// Notes on fidelity:
//   * Head counts (32 / 8 / 96) and the GQA group ratio (4) match the spec's
//     fixed constants. The q / k norm weights are per-head-dim ([head_dim]),
//     so the RMSNorm scale multiplies the [B, S, heads, head_dim] layout after
//     the view -- this mirrors the reference, which views to heads before the
//     norm and broadcasts the [head_dim] weight over the head axis.
//   * RMSNorm reduces over the last (head_dim) axis in fp32 (pow / mean /
//     rsqrt). For Q / K the bf16 cast happens after the learned-weight multiply;
//     for V there is no learned scale and the cast happens after the rsqrt.
//     Wider fp32 accumulation over bf16 inputs is expected and is not a
//     precision downgrade.
//   * RoPE is built from rope_theta via arange(0, head_dim, 2) / head_dim and
//     the supplied integer position_ids (not an internal arange), matched
//     through the inv_freq @ position_ids batched product then transpose,
//     exactly as the reference does. It is applied on the pre-transpose
//     [B, S, heads, head_dim] layout with cos / sin unsqueezed on the head
//     axis, before the transpose to [B, heads, S, head_dim]. V never receives
//     RoPE.
//   * Soft-capping is softcap * tanh(scores / softcap), applied to the raw
//     QK^T logits before the additive mask. The reference does NOT divide the
//     logits by sqrt(head_dim) here -- the only logit rescaling is the
//     soft-cap -- so no 1/sqrt(head_dim) factor is applied.
//   * use_shared_kv selects, on the host, between the computed K / V path and
//     the supplied shared_key_states / shared_value_states. The shared states
//     arrive in the post-transpose [B, num_kv_heads, S, head_dim] layout and
//     are consumed directly (no norm, no RoPE), matching the reference branch.
//   * Softmax runs in fp32 and is cast back to bf16 before the value matmul,
//     matching the reference dtype flow.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    check_cuda(t, name);
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

// rotate_half(x) = cat((-x[..., hd/2:], x[..., :hd/2]), dim=-1) over the last
// axis of a [..., head_dim] tensor.
static torch::Tensor rotate_half(const torch::Tensor& x, int64_t half_dim) {
    auto first = x.narrow(/*dim=*/-1, /*start=*/0, half_dim);
    auto second = x.narrow(/*dim=*/-1, /*start=*/half_dim, half_dim);
    return at::cat({-second, first}, -1);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& position_ids,
    const torch::Tensor& attention_mask,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& shared_key_states,
    const torch::Tensor& shared_value_states,
    double rope_theta,
    double softcap,
    double rms_norm_eps,
    bool use_shared_kv,
    torch::Tensor attn_output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_cuda(position_ids, "position_ids");
    check_bf16_cuda(attention_mask, "attention_mask");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_cuda(q_norm_weight, "q_norm_weight");
    check_bf16_cuda(k_norm_weight, "k_norm_weight");
    check_bf16_cuda(shared_key_states, "shared_key_states");
    check_bf16_cuda(shared_value_states, "shared_value_states");
    check_bf16_cuda(attn_output, "attn_output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(position_ids.dim() == 2,
                "position_ids must be 2D [B, S], got ", position_ids.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 32;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 96;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 4
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 48

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const auto qdtype = hidden_states.scalar_type();

    // Q projection + per-head RMSNorm (fp32 accumulation), scaled by the
    // per-head-dim q_norm_weight, cast back to bf16.
    auto query_states = at::linear(hidden_states, q_proj_weight);  // [B,S,q_proj]
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim});
    auto q_var = query_states.to(torch::kFloat32).pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    query_states = query_states * at::rsqrt(q_var + rms_norm_eps);
    query_states = (query_states * q_norm_weight).to(qdtype);

    // RoPE built from rope_theta and the supplied position_ids: inv_freq over
    // the even indices of head_dim, multiplied against the integer positions.
    auto exponents =
        at::arange(0, kHeadDim, 2, at::TensorOptions().dtype(torch::kFloat32)
                                       .device(hidden_states.device())) /
        static_cast<double>(kHeadDim);
    auto inv_freq = at::pow(rope_theta, exponents).reciprocal();  // [half]
    auto inv_freq_expanded =
        inv_freq.view({1, kHalfDim, 1}).expand({batch_size, kHalfDim, 1});  // [B,half,1]
    auto position_ids_expanded =
        position_ids.to(torch::kFloat32).view({batch_size, 1, seq_len});  // [B,1,S]
    auto freqs = at::matmul(inv_freq_expanded, position_ids_expanded)
                     .transpose(1, 2);  // [B,half,S] -> [B,S,half]
    auto emb = at::cat({freqs, freqs}, -1);  // [B,S,head_dim], fp32
    auto cos = emb.cos().to(qdtype);         // [B,S,head_dim]
    auto sin = emb.sin().to(qdtype);

    // Apply RoPE to Q on the [B,S,heads,head_dim] layout with cos / sin
    // unsqueezed on the head axis, then transpose to [B,heads,S,head_dim].
    auto cos_u = cos.unsqueeze(2);  // [B,S,1,head_dim]
    auto sin_u = sin.unsqueeze(2);
    query_states = (query_states * cos_u) + (rotate_half(query_states, kHalfDim) * sin_u);
    query_states = query_states.transpose(1, 2);  // [B,32,S,96]

    // K / V come either from the supplied shared states or from fresh
    // projections, selected on the host by use_shared_kv.
    torch::Tensor key_states;
    torch::Tensor value_states;
    if (use_shared_kv) {
        // Reuse the earlier layer's K / V; they are already normalized and
        // RoPE'd and arrive in the [B, num_kv_heads, S, head_dim] layout.
        key_states = shared_key_states;
        value_states = shared_value_states;
    } else {
        // K projection + per-head RMSNorm with k_norm_weight, then RoPE.
        key_states = at::linear(hidden_states, k_proj_weight);  // [B,S,kv_proj]
        key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});
        auto k_var = key_states.to(torch::kFloat32).pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
        key_states = key_states * at::rsqrt(k_var + rms_norm_eps);
        key_states = (key_states * k_norm_weight).to(qdtype);
        key_states = (key_states * cos_u) + (rotate_half(key_states, kHalfDim) * sin_u);
        key_states = key_states.transpose(1, 2);  // [B,8,S,96]

        // V projection + RMSNorm without a learned scale; no RoPE.
        value_states = at::linear(hidden_states, v_proj_weight);  // [B,S,kv_proj]
        value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});
        auto v_var = value_states.to(torch::kFloat32).pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
        value_states = (value_states * at::rsqrt(v_var + rms_norm_eps)).to(qdtype);
        value_states = value_states.transpose(1, 2);  // [B,8,S,96]
    }

    // repeat_kv: expand the 8 kv heads into 32 query heads.
    auto key_rep = key_states
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // QK^T, tanh logit soft-cap (no 1/sqrt(head_dim) factor here), additive
    // mask, fp32 softmax cast back to bf16.
    auto attn_weights = at::matmul(query_states, key_rep.transpose(2, 3));  // [B,32,S,S]
    attn_weights = at::tanh(attn_weights / softcap) * softcap;
    attn_weights = attn_weights + attention_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32).to(qdtype);
    auto attn_out = at::matmul(attn_weights, value_rep);  // [B,32,S,96]

    // Unwind heads and apply the output projection.
    attn_out = attn_out.transpose(1, 2).contiguous()
                   .reshape({batch_size, seq_len, kNumHeads * kHeadDim});
    auto result = at::linear(attn_out, o_proj_weight);  // [B,S,hidden]

    // Destination-passing-style write.
    TORCH_CHECK(result.sizes() == attn_output.sizes(),
                "attn_output shape ", result.sizes(),
                " does not match pre-allocated ", attn_output.sizes());
    attn_output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "KV-shared attention with dual RoPE via ATen (fp32-accumulated Q/K/V "
          "RMSNorm with per-head-dim Q/K scale and scale-free V, RoPE from "
          "rope_theta and supplied position_ids, rotate-half, GQA KV repeat, "
          "tanh logit soft-capping, additive causal mask, fp32 softmax; "
          "host-side use_shared_kv branch reuses the supplied shared K/V; no "
          "custom CUDA kernels); bf16 in/out with fp32 norm/softmax "
          "accumulation; destination-passing style (attn_output).");
}
