#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>
#include <cmath>

// KV-sharing attention with RoPE for Gemma3n. 8 query heads, 1 key-value head
// (8:1 GQA ratio), head dim 256. Q, K, and V are each RMS-normalized in fp32;
// Q and K carry a learned per-head-dim norm weight while V is normalized
// without a scale. RoPE is built from the rope_theta base, attention logits
// are tanh soft-capped, the additive causal mask is applied, and softmax runs
// in fp32. The pre-repeat K and V states are returned alongside the attention
// output for the KV-sharing mechanism, so they are cloned before the GQA head
// repeat to capture the unrepeated [B, num_kv_heads, S, head_dim] layout.
//
// Reference (problem_039_kv_shared_attention_with_rope), op-for-op:
//   num_attention_heads  = 8
//   num_key_value_heads  = 1
//   head_dim             = 256
//   num_key_value_groups = num_attention_heads // num_key_value_heads = 8
//
//   query_states = linear(hidden_states, q_proj_weight)   # [B,S,qkv_out]
//   query_states = query_states.view(B,S,num_heads,head_dim)
//   q_var = query_states.float().pow(2).mean(-1, keepdim=True)
//   query_states = query_states * rsqrt(q_var + eps)
//   query_states = (query_states * q_norm_weight).to(bf16)
//   # K identically with k_norm_weight (view to num_kv_heads).
//   # V: RMSNorm without scale -- (v * rsqrt(v_var + eps)).to(bf16).
//
//   # RoPE built from rope_theta:
//   inv_freq = 1 / (rope_theta ** (arange(0, head_dim, 2).float() / head_dim))
//   inv_freq_expanded     = inv_freq[None,:,None].expand(B,-1,1)
//   position_ids_expanded = position_ids[:,None,:].float()
//   freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
//   emb   = cat((freqs, freqs), dim=-1)
//   cos = emb.cos().to(bf16) ; sin = emb.sin().to(bf16)   # [B,S,head_dim]
//   # Applied on the [B,S,heads,head_dim] layout with cos/sin.unsqueeze(2):
//   rotate_half(x) = cat((-x[..., hd/2:], x[..., :hd/2]), dim=-1)
//   query_states = query_states*cos_u + rotate_half(query_states)*sin_u
//   key_states   = key_states*cos_u   + rotate_half(key_states)*sin_u
//
//   # Transpose to [B,heads,S,head_dim].
//   query_states = query_states.transpose(1, 2)
//   key_states   = key_states.transpose(1, 2)
//   value_states = value_states.transpose(1, 2)
//
//   # KV-sharing outputs: snapshot K / V before the GQA repeat.
//   key_states_out   = key_states.clone()
//   value_states_out = value_states.clone()
//
//   # repeat_kv: expand the single kv head into 8 query heads, then reshape.
//   attn = query_states @ key_states.transpose(2,3)
//   attn = attn / sqrt(head_dim)
//   attn = softcap * tanh(attn / softcap)     # logit soft-capping
//   attn = attn + attention_mask              # additive causal mask
//   attn = softmax(attn, dim=-1, dtype=fp32).to(bf16)
//   attn_output = attn @ value_states
//   attn_output = attn_output.transpose(1,2).contiguous().view(B,S,qkv_out)
//   attn_output = linear(attn_output, o_proj_weight)        # [B,S,hidden]
//   return attn_output, key_states_out, value_states_out
//
// Per definition.json (insertion order):
//   hidden_states    : [B, S, hidden=2304]               bf16
//   position_ids     : [B, S]                             int64
//   attention_mask   : [B, 1, S, S]                       bf16
//   q_proj_weight    : [qkv_out=2048, hidden]            bf16
//   k_proj_weight    : [kv_out=256, hidden]              bf16
//   v_proj_weight    : [kv_out=256, hidden]              bf16
//   o_proj_weight    : [hidden, qkv_out]                 bf16
//   q_norm_weight    : [head_dim=256]                    bf16
//   k_norm_weight    : [head_dim=256]                    bf16
//   rope_theta       : scalar                            fp32 -> double
//   softcap          : scalar                            fp32 -> double
//   rms_norm_eps     : scalar                            fp32 -> double
//   attn_output      : [B, S, hidden]                    bf16 DPS
//   key_states_out   : [B, num_kv_heads=1, S, head_dim]  bf16 DPS
//   value_states_out : [B, num_kv_heads=1, S, head_dim]  bf16 DPS
//
// Notes on fidelity:
//   * Head counts (8 / 1 / 256) and the GQA group ratio (8) match the spec's
//     fixed constants. The q / k norm weights are per-head-dim ([head_dim]),
//     so the RMSNorm scale multiplies the [B, S, heads, head_dim] layout after
//     the view -- this mirrors the reference, which views to heads before the
//     norm and broadcasts the [head_dim] weight over the head axis.
//   * RMSNorm reduces over the last (head_dim) axis in fp32 (pow / mean /
//     rsqrt). The bf16 cast happens after the weight multiply for Q / K, and
//     for V after the rsqrt scale (V has no learned norm weight). Wider fp32
//     accumulation over bf16 inputs is expected and is not a precision
//     downgrade.
//   * RoPE is built from rope_theta via arange(0, head_dim, 2) / head_dim, and
//     applied on the pre-transpose [B, S, heads, head_dim] layout with cos /
//     sin unsqueezed on the head axis, exactly as the reference does, before
//     the transpose to [B, heads, S, head_dim].
//   * Soft-capping is softcap * tanh(scores / softcap), applied after the
//     1/sqrt(head_dim) scale and before the additive mask, matching the
//     reference order.
//   * key_states_out / value_states_out are the post-RoPE / post-norm K / V
//     in their pre-GQA-repeat [B, num_kv_heads, S, head_dim] layout. They are
//     cloned before the repeat so the returned KV are the unrepeated states
//     the KV-sharing mechanism consumes.
//   * Softmax runs in fp32 and is cast back to bf16 before the value matmul,
//     matching the reference dtype flow.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. All
// three outputs are written in place into pre-allocated destinations
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
    double rope_theta,
    double softcap,
    double rms_norm_eps,
    torch::Tensor attn_output,
    torch::Tensor key_states_out,
    torch::Tensor value_states_out
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
    check_bf16_cuda(attn_output, "attn_output");
    check_bf16_cuda(key_states_out, "key_states_out");
    check_bf16_cuda(value_states_out, "value_states_out");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(position_ids.dim() == 2,
                "position_ids must be 2D [B, S], got ", position_ids.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 8;
    constexpr int64_t kNumKVHeads = 1;
    constexpr int64_t kHeadDim = 256;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 8
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 128

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const auto qdtype = hidden_states.scalar_type();
    const double attn_scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    // Q projection + per-head RMSNorm (fp32 accumulation), scaled by the
    // per-head-dim q_norm_weight, cast back to bf16.
    auto query_states = at::linear(hidden_states, q_proj_weight);  // [B,S,qkv_out]
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim});
    auto q_var = query_states.to(torch::kFloat32).pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    query_states = query_states * at::rsqrt(q_var + rms_norm_eps);
    query_states = (query_states * q_norm_weight).to(qdtype);

    // K projection + per-head RMSNorm identically with k_norm_weight.
    auto key_states = at::linear(hidden_states, k_proj_weight);  // [B,S,kv_out]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});
    auto k_var = key_states.to(torch::kFloat32).pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    key_states = key_states * at::rsqrt(k_var + rms_norm_eps);
    key_states = (key_states * k_norm_weight).to(qdtype);

    // V projection + RMSNorm without a learned scale.
    auto value_states = at::linear(hidden_states, v_proj_weight);  // [B,S,kv_out]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});
    auto v_var = value_states.to(torch::kFloat32).pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    value_states = (value_states * at::rsqrt(v_var + rms_norm_eps)).to(qdtype);

    // RoPE built from rope_theta: inv_freq over the even indices of head_dim.
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

    // Apply RoPE on the [B,S,heads,head_dim] layout with cos / sin unsqueezed
    // on the head axis.
    auto cos_u = cos.unsqueeze(2);  // [B,S,1,head_dim]
    auto sin_u = sin.unsqueeze(2);
    query_states = (query_states * cos_u) + (rotate_half(query_states, kHalfDim) * sin_u);
    key_states = (key_states * cos_u) + (rotate_half(key_states, kHalfDim) * sin_u);

    // Transpose to [B,heads,S,head_dim].
    query_states = query_states.transpose(1, 2);  // [B,8,S,256]
    key_states = key_states.transpose(1, 2);      // [B,1,S,256]
    value_states = value_states.transpose(1, 2);  // [B,1,S,256]

    // KV-sharing snapshots: clone the pre-repeat K / V so the returned states
    // are the unrepeated [B, num_kv_heads, S, head_dim] layout.
    auto key_snapshot = key_states.clone();
    auto value_snapshot = value_states.clone();

    // repeat_kv: expand the single kv head into 8 query heads.
    auto key_rep = key_states
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // QK^T, 1/sqrt(head_dim) scale, tanh soft-cap, additive mask, fp32 softmax.
    auto attn_weights = at::matmul(query_states, key_rep.transpose(2, 3));  // [B,8,S,S]
    attn_weights = attn_weights * attn_scale;
    attn_weights = at::tanh(attn_weights / softcap) * softcap;
    attn_weights = attn_weights + attention_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32).to(qdtype);
    auto attn_out = at::matmul(attn_weights, value_rep);  // [B,8,S,256]

    // Unwind heads and apply the output projection.
    attn_out = attn_out.transpose(1, 2).contiguous()
                   .reshape({batch_size, seq_len, kNumHeads * kHeadDim});
    auto result = at::linear(attn_out, o_proj_weight);  // [B,S,hidden]

    // Destination-passing-style writes for all three outputs.
    TORCH_CHECK(result.sizes() == attn_output.sizes(),
                "attn_output shape ", result.sizes(),
                " does not match pre-allocated ", attn_output.sizes());
    TORCH_CHECK(key_snapshot.sizes() == key_states_out.sizes(),
                "key_states_out shape ", key_snapshot.sizes(),
                " does not match pre-allocated ", key_states_out.sizes());
    TORCH_CHECK(value_snapshot.sizes() == value_states_out.sizes(),
                "value_states_out shape ", value_snapshot.sizes(),
                " does not match pre-allocated ", value_states_out.sizes());
    attn_output.copy_(result);
    key_states_out.copy_(key_snapshot);
    value_states_out.copy_(value_snapshot);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "KV-sharing attention with RoPE via ATen (fp32-accumulated Q/K/V "
          "RMSNorm with per-head-dim Q/K scale and scale-free V, RoPE from "
          "rope_theta, rotate-half, GQA KV repeat, tanh logit soft-capping, "
          "additive causal mask, fp32 softmax; pre-repeat K/V cloned for the "
          "KV-sharing outputs; no custom CUDA kernels); bf16; "
          "destination-passing style (attn_output, key_states_out, "
          "value_states_out).");
}
