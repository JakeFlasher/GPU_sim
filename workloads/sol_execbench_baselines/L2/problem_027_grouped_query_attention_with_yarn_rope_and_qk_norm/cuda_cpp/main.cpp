#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Grouped Query Attention with YARN RoPE scaling and Q/K RMS normalization.
// 40 query heads, 8 key-value heads (5:1 GQA ratio), head dim 128. The YARN
// inverse frequencies arrive pre-computed as the inv_freq input; this port
// builds the cos / sin tables from them, applies the YARN attention_factor
// to the tables, normalizes Q and K with fp32-accumulated RMSNorm, rotates Q
// and K, repeats the KV heads for GQA, then runs scaled dot-product attention
// with the additive sliding-window mask and an fp32-accumulated softmax.
//
// Reference (problem_027_grouped_query_attention_with_yarn_rope_and_qk_norm),
// op-for-op:
//   num_attention_heads  = 40
//   num_key_value_heads  = 8
//   head_dim             = 128
//   num_key_value_groups = num_attention_heads // num_key_value_heads = 5
//
//   query_states = linear(hidden_states, q_proj_weight)   # [B,S,hidden]
//   key_states   = linear(hidden_states, k_proj_weight)   # [B,S,kv_hidden]
//   value_states = linear(hidden_states, v_proj_weight)   # [B,S,kv_hidden]
//
//   # Q RMSNorm in fp32, scaled by the full-width q_norm_weight:
//   q32 = query_states.float()
//   q_var = q32.pow(2).mean(-1, keepdim=True)
//   query_states = (q_norm_weight * (q32 * rsqrt(q_var + eps))).to(bf16)
//   # K RMSNorm identically with k_norm_weight.
//
//   reshape to [B, heads, S, head_dim] via view + transpose(1, 2).
//
//   # YARN RoPE: inv_freq is the pre-computed YARN frequency vector.
//   inv_freq_expanded     = inv_freq[None,:,None].float().expand(B,-1,1)
//   position_ids_expanded = position_ids[:,None,:].float()
//   freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
//   emb   = cat((freqs, freqs), dim=-1)
//   cos   = (emb.cos() * attention_factor).unsqueeze(1).to(bf16)  # [B,1,S,hd]
//   sin   = (emb.sin() * attention_factor).unsqueeze(1).to(bf16)
//   rotate_half(x) = cat((-x[..., hd/2:], x[..., :hd/2]), dim=-1)
//   query_states = query_states*cos + rotate_half(query_states)*sin
//   key_states   = key_states*cos   + rotate_half(key_states)*sin
//
//   repeat_kv: expand the 8 kv heads into 40 query heads, then reshape.
//   attn = (query_states @ key_states.transpose(2,3)) * scaling
//   attn = attn + attention_mask              # additive sliding-window mask
//   attn = softmax(attn, dim=-1, dtype=fp32).to(bf16)
//   attn_output = attn @ value_states
//   attn_output = attn_output.transpose(1,2).contiguous().view(B,S,hidden)
//   output = linear(attn_output, o_proj_weight)
//
// Per definition.json (insertion order):
//   hidden_states    : [B, S, hidden=5120]               bf16
//   position_ids     : [B, S]                             int64
//   attention_mask   : [B, 1, S, S]                       bf16
//   q_proj_weight    : [hidden, hidden]                   bf16
//   k_proj_weight    : [kv_hidden=1024, hidden]           bf16
//   v_proj_weight    : [kv_hidden=1024, hidden]           bf16
//   o_proj_weight    : [hidden, hidden]                   bf16
//   q_norm_weight    : [hidden]                           bf16
//   k_norm_weight    : [kv_hidden]                        bf16
//   inv_freq         : [half_head_dim=64]                 fp32
//   rms_norm_eps     : scalar                             fp32 -> double
//   attention_factor : scalar                             fp32 -> double
//   scaling          : scalar                             fp32 -> double
//   output           : [B, S, hidden]                     bf16 DPS
//
// Notes on fidelity:
//   * Head counts (40 / 8 / 128) and the GQA group ratio (5) match the
//     spec's fixed constants. q_norm_weight / k_norm_weight span the whole
//     projection width (hidden / kv_hidden), so the RMSNorm scale multiplies
//     the flat [B, S, width] layout before the per-head reshape -- this
//     mirrors the reference, which normalizes over the last (full-width) axis
//     and applies the full-width weight before the view.
//   * RMSNorm reduces over the last axis in fp32 (pow / mean / rsqrt) and the
//     bf16 cast happens only after the weight multiply, matching the
//     reference precision profile. Wider fp32 accumulation over bf16 inputs
//     is expected and is not a precision downgrade.
//   * The YARN attention_factor multiplies the cos / sin tables (it is the
//     YARN magnitude-scaling term), applied before the bf16 cast just as the
//     reference does.
//   * inv_freq arrives as fp32 and is matmul-ed against the fp32 position ids
//     to form freqs; the cos / sin tables are taken in fp32 and cast to bf16
//     once, matching the reference.
//   * The sliding-window behavior lives entirely in the additive
//     attention_mask input (-inf outside the window); no extra masking is
//     synthesized here.
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

// RMSNorm over the last (full-width) axis with fp32 accumulation, scaled by a
// full-width weight, then cast back to the input dtype. x is the bf16
// projection; weight is the full-width bf16 norm weight.
static torch::Tensor rms_norm_qk(
    const torch::Tensor& x, const torch::Tensor& weight, double eps
) {
    auto x32 = x.to(torch::kFloat32);
    auto variance = x32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto normed = x32 * at::rsqrt(variance + eps);
    return (weight * normed).to(x.dtype());
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
    const torch::Tensor& inv_freq,
    double rms_norm_eps,
    double attention_factor,
    double scaling,
    torch::Tensor output
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
    check_cuda(inv_freq, "inv_freq");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(position_ids.dim() == 2,
                "position_ids must be 2D [B, S], got ", position_ids.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 40;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 5
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 64

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const auto qdtype = hidden_states.scalar_type();

    // Q/K/V projections (no bias) via linear.
    auto query_states = at::linear(hidden_states, q_proj_weight);  // [B,S,hidden]
    auto key_states = at::linear(hidden_states, k_proj_weight);    // [B,S,kv_hidden]
    auto value_states = at::linear(hidden_states, v_proj_weight);  // [B,S,kv_hidden]

    // Q / K RMSNorm over the full projection width (fp32 accumulation), scaled
    // by the full-width norm weights, cast back to bf16.
    query_states = rms_norm_qk(query_states, q_norm_weight, rms_norm_eps);
    key_states = rms_norm_qk(key_states, k_norm_weight, rms_norm_eps);

    // Reshape to heads then transpose seq<->head.
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,40,S,128]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,8,S,128]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,8,S,128]

    // YARN RoPE: build cos / sin from the pre-computed inv_freq and the
    // position ids. inv_freq is [half], position_ids is [B, S].
    auto inv_freq_expanded = inv_freq.to(torch::kFloat32)
                                 .view({1, kHalfDim, 1})
                                 .expand({batch_size, kHalfDim, 1});  // [B,half,1]
    auto position_ids_expanded =
        position_ids.to(torch::kFloat32).view({batch_size, 1, seq_len});  // [B,1,S]
    auto freqs = at::matmul(inv_freq_expanded, position_ids_expanded)
                     .transpose(1, 2);  // [B,half,S] -> [B,S,half]
    auto emb = at::cat({freqs, freqs}, -1);  // [B,S,head_dim], fp32

    // The YARN attention_factor scales the cos / sin tables; cast to bf16
    // once and unsqueeze the head axis so the tables broadcast over heads.
    auto cos = (emb.cos() * attention_factor).unsqueeze(1).to(qdtype);  // [B,1,S,hd]
    auto sin = (emb.sin() * attention_factor).unsqueeze(1).to(qdtype);

    query_states = (query_states * cos) + (rotate_half(query_states, kHalfDim) * sin);
    key_states = (key_states * cos) + (rotate_half(key_states, kHalfDim) * sin);

    // repeat_kv: expand the 8 kv heads into 40 query heads.
    auto key_rep = key_states
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // Scaled QK^T, additive sliding-window mask, fp32 softmax, attention @ V.
    auto attn_weights = at::matmul(query_states, key_rep.transpose(2, 3)) * scaling;  // [B,40,S,S]
    attn_weights = attn_weights + attention_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32).to(qdtype);
    auto attn_output = at::matmul(attn_weights, value_rep);  // [B,40,S,128]

    // Unwind heads and apply the output projection.
    attn_output = attn_output.transpose(1, 2).contiguous()
                      .reshape({batch_size, seq_len, kNumHeads * kHeadDim});
    auto result = at::linear(attn_output, o_proj_weight);  // [B,S,hidden]

    // Destination-passing-style write.
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Grouped query attention with YARN RoPE and Q/K RMSNorm via ATen "
          "(fp32-accumulated Q/K RMSNorm, YARN cos/sin from pre-computed "
          "inv_freq scaled by attention_factor, rotate-half, GQA KV repeat, "
          "scaled dot-product attention with additive sliding-window mask and "
          "fp32 softmax; no custom CUDA kernels); bf16; destination-passing "
          "style (output).");
}
