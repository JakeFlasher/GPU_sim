#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>
#include <cmath>
#include <limits>

// Complete decoder layer: pre-norm GQA self-attention with RoPE and a causal
// mask feeding a residual, then pre-norm multi-head cross-attention over the
// encoder states feeding a second residual, then a pre-norm SwiGLU MLP feeding
// a third residual. Three RMSNorms, three residual connections.
//
// Reference (problem_062_decoder_complete_layer), op-for-op:
//   num_attention_heads        = 16   num_key_value_heads        = 4
//   head_dim                   = 128
//   cross_num_attention_heads  = 16   cross_num_key_value_heads  = 16
//   cross_head_dim             = 128
//
//   rms_norm(x, weight, eps):
//     x_fp32   = x.to(float32)
//     variance = x_fp32.pow(2).mean(-1, keepdim=True)
//     x_normed = x_fp32 * rsqrt(variance + eps)
//     return (weight * x_normed).to(input_dtype)   # scale in fp32, then bf16
//
//   rotate_half(x) = cat([-x[..., d/2:], x[..., :d/2]], dim=-1)
//
//   apply_rope(q, k, position_ids, head_dim, rope_theta):
//     inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2, fp32) / head_dim))
//     freqs    = outer(position_ids.float().flatten(), inv_freq)   # [B*S, d/2]
//     emb      = cat([freqs, freqs], dim=-1)                       # [B*S, d]
//     cos = emb.cos().to(bf16).view(B,1,S,d) ; sin = emb.sin().to(bf16).view(...)
//     q_embed = q*cos + rotate_half(q)*sin ; k_embed = k*cos + rotate_half(k)*sin
//
//   --- Self-attention block (GQA + RoPE + causal mask) ---
//   residual = hidden_states
//   normed   = rms_norm(hidden_states, self_attn_norm_weight, eps)
//   q = linear(normed, self_attn_q_weight).view(B,S,16,128).transpose(1,2)
//   k = linear(normed, self_attn_k_weight).view(B,S, 4,128).transpose(1,2)
//   v = linear(normed, self_attn_v_weight).view(B,S, 4,128).transpose(1,2)
//   q, k = apply_rope(q, k, arange(S), 128, rope_theta)
//   k = repeat_kv(k, 4) ; v = repeat_kv(v, 4)            # [B,16,S,128]
//   attn = (q @ k.transpose(2,3)) / sqrt(128)
//   causal = triu(full((S,S), -inf), diagonal=1)
//   attn = attn + causal.unsqueeze(0).unsqueeze(0)
//   attn = softmax(attn, dim=-1, dtype=float32).to(bf16)
//   o    = (attn @ v).transpose(1,2).contiguous().view(B,S,16*128)
//   o    = linear(o, self_attn_o_weight)
//   hidden_states = residual + o
//
//   --- Cross-attention block (plain MHA over encoder states, no RoPE/mask) ---
//   residual = hidden_states
//   normed   = rms_norm(hidden_states, cross_attn_norm_weight, eps)
//   q = linear(normed, cross_attn_q_weight).view(B,S,16,128).transpose(1,2)
//   k = linear(encoder_hidden_states, cross_attn_k_weight)
//         .view(B,E,16,128).transpose(1,2)
//   v = linear(encoder_hidden_states, cross_attn_v_weight)
//         .view(B,E,16,128).transpose(1,2)
//   # 16 query heads == 16 kv heads, so no repeat_kv
//   attn = (q @ k.transpose(2,3)) / sqrt(128)
//   attn = softmax(attn, dim=-1, dtype=float32).to(bf16)
//   o    = (attn @ v).transpose(1,2).contiguous().view(B,S,16*128)
//   o    = linear(o, cross_attn_o_weight)
//   hidden_states = residual + o
//
//   --- MLP block (SwiGLU) ---
//   residual = hidden_states
//   normed   = rms_norm(hidden_states, mlp_norm_weight, eps)
//   gate = linear(normed, mlp_gate_weight) ; up = linear(normed, mlp_up_weight)
//   hidden_states = linear(silu(gate) * up, mlp_down_weight)
//   output = residual + hidden_states
//
// Per definition.json (insertion order):
//   hidden_states          : [B, S, hidden]               bf16
//   encoder_hidden_states  : [B, E, cross_hidden]         bf16
//   self_attn_norm_weight  : [hidden]                     bf16
//   self_attn_q_weight     : [self_attn_q_dim, hidden]    bf16
//   self_attn_k_weight     : [self_attn_kv_dim, hidden]   bf16
//   self_attn_v_weight     : [self_attn_kv_dim, hidden]   bf16
//   self_attn_o_weight     : [hidden, self_attn_q_dim]    bf16
//   cross_attn_norm_weight : [hidden]                     bf16
//   cross_attn_q_weight    : [cross_attn_q_dim, hidden]   bf16
//   cross_attn_k_weight    : [cross_attn_kv_dim, cross_hidden] bf16
//   cross_attn_v_weight    : [cross_attn_kv_dim, cross_hidden] bf16
//   cross_attn_o_weight    : [hidden, cross_attn_q_dim]   bf16
//   mlp_norm_weight        : [hidden]                     bf16
//   mlp_gate_weight        : [intermediate, hidden]       bf16
//   mlp_up_weight          : [intermediate, hidden]       bf16
//   mlp_down_weight        : [hidden, intermediate]       bf16
//   norm_eps               : scalar                       fp32 -> double
//   rope_theta             : scalar                       fp32 -> double
//   output                 : [B, S, hidden]               bf16 DPS
//
// Notes on fidelity:
//   * RoPE here is the rotate-half variant built internally (not from supplied
//     cos / sin): inv_freq is derived from rope_theta, freqs come from an
//     arange(seq_len) position grid, emb = cat([freqs, freqs]), and the
//     rotation is q*cos + rotate_half(q)*sin where rotate_half swaps and negates
//     the two halves: cat([-x2, x1]). cos / sin are computed in fp32 then cast
//     to bf16 before the rotation, matching emb.cos().to(q.dtype).
//   * The causal mask is built internally as an upper-triangular -inf matrix
//     (triu, diagonal=1) and added to the scaled self-attention scores. Cross-
//     attention is unmasked.
//   * Cross-attention has equal query and kv head counts (16 == 16), so there
//     is no repeat_kv there. Self-attention is GQA (16 query, 4 kv) and uses
//     repeat_kv to expand the 4 kv heads into 16.
//   * Precision: inputs and outputs are bf16. The RMSNorm variance / rsqrt and
//     the two softmaxes run in fp32; the normalized value is cast back to bf16
//     after the weight scale and the softmax weights are cast back to bf16
//     before the value matmul. The bf16 output is never upcast; bf16 is the
//     contractual dtype, not a downgrade.
//   * SwiGLU: gate = silu(gate_proj); the gate is multiplied by the up
//     projection, then the down projection is applied.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

// RMSNorm over the last dim. The variance / rsqrt run in fp32; the fp32
// normalized value is scaled by the weight and cast back to bf16, matching the
// reference's (weight * x_normed).to(input_dtype).
static torch::Tensor rms_norm(
    const torch::Tensor& x, const torch::Tensor& weight, double eps
) {
    auto x_fp32 = x.to(torch::kFloat32);
    auto variance = x_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto x_normed = x_fp32 * at::rsqrt(variance + eps);
    return (weight * x_normed).to(x.scalar_type());
}

// rotate_half: split the last dim in half, then return cat([-second, first]).
static torch::Tensor rotate_half(const torch::Tensor& x) {
    const int64_t half = x.size(-1) / 2;
    auto x1 = x.narrow(/*dim=*/-1, /*start=*/0, half);
    auto x2 = x.narrow(/*dim=*/-1, /*start=*/half, half);
    return at::cat({-x2, x1}, -1);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& self_attn_norm_weight,
    const torch::Tensor& self_attn_q_weight,
    const torch::Tensor& self_attn_k_weight,
    const torch::Tensor& self_attn_v_weight,
    const torch::Tensor& self_attn_o_weight,
    const torch::Tensor& cross_attn_norm_weight,
    const torch::Tensor& cross_attn_q_weight,
    const torch::Tensor& cross_attn_k_weight,
    const torch::Tensor& cross_attn_v_weight,
    const torch::Tensor& cross_attn_o_weight,
    const torch::Tensor& mlp_norm_weight,
    const torch::Tensor& mlp_gate_weight,
    const torch::Tensor& mlp_up_weight,
    const torch::Tensor& mlp_down_weight,
    double norm_eps,
    double rope_theta,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(encoder_hidden_states, "encoder_hidden_states");
    check_bf16_cuda(self_attn_norm_weight, "self_attn_norm_weight");
    check_bf16_cuda(self_attn_q_weight, "self_attn_q_weight");
    check_bf16_cuda(self_attn_k_weight, "self_attn_k_weight");
    check_bf16_cuda(self_attn_v_weight, "self_attn_v_weight");
    check_bf16_cuda(self_attn_o_weight, "self_attn_o_weight");
    check_bf16_cuda(cross_attn_norm_weight, "cross_attn_norm_weight");
    check_bf16_cuda(cross_attn_q_weight, "cross_attn_q_weight");
    check_bf16_cuda(cross_attn_k_weight, "cross_attn_k_weight");
    check_bf16_cuda(cross_attn_v_weight, "cross_attn_v_weight");
    check_bf16_cuda(cross_attn_o_weight, "cross_attn_o_weight");
    check_bf16_cuda(mlp_norm_weight, "mlp_norm_weight");
    check_bf16_cuda(mlp_gate_weight, "mlp_gate_weight");
    check_bf16_cuda(mlp_up_weight, "mlp_up_weight");
    check_bf16_cuda(mlp_down_weight, "mlp_down_weight");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(encoder_hidden_states.dim() == 3,
                "encoder_hidden_states must be 3D [B, E, cross_hidden], got ",
                encoder_hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kNumKVHeads = 4;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 4
    constexpr int64_t kCrossNumHeads = 16;
    constexpr int64_t kCrossNumKVHeads = 16;
    constexpr int64_t kCrossHeadDim = 128;

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t encoder_seq_len = encoder_hidden_states.size(1);
    const double self_scaling = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    const double cross_scaling = 1.0 / std::sqrt(static_cast<double>(kCrossHeadDim));

    // --- Self-attention block: GQA + RoPE + causal mask. ---
    auto residual = hidden_states;
    auto normed = rms_norm(residual, self_attn_norm_weight, norm_eps);

    auto query_states = at::matmul(normed, self_attn_q_weight.t());  // [B,S,q_dim]
    auto key_states = at::matmul(normed, self_attn_k_weight.t());    // [B,S,kv_dim]
    auto value_states = at::matmul(normed, self_attn_v_weight.t());  // [B,S,kv_dim]

    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,16,S,128]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,4,S,128]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,4,S,128]

    // Build RoPE cos / sin from an arange position grid and rope_theta, the way
    // the reference does. inv_freq is [head_dim/2]; freqs = outer(positions,
    // inv_freq) is [S, head_dim/2]; emb = cat([freqs, freqs]) is [S, head_dim].
    auto fp32_opts = hidden_states.options().dtype(torch::kFloat32);
    auto position_ids = at::arange(seq_len, fp32_opts);  // [S]
    auto exponents = at::arange(/*start=*/0, /*end=*/kHeadDim, /*step=*/2, fp32_opts)
                         / static_cast<double>(kHeadDim);  // [head_dim/2]
    auto inv_freq = at::pow(rope_theta, exponents).reciprocal();  // [head_dim/2]
    auto freqs = at::outer(position_ids, inv_freq);  // [S, head_dim/2]
    auto emb = at::cat({freqs, freqs}, -1);          // [S, head_dim]
    // Shape to [1,1,S,head_dim] so it broadcasts across batch and head axes;
    // every batch row shares the same arange positions.
    auto cos = emb.cos().to(query_states.scalar_type())
                   .view({1, 1, seq_len, kHeadDim});
    auto sin = emb.sin().to(query_states.scalar_type())
                   .view({1, 1, seq_len, kHeadDim});

    auto q_embed = query_states * cos + rotate_half(query_states) * sin;
    auto k_embed = key_states * cos + rotate_half(key_states) * sin;

    // repeat_kv: expand the 4 kv heads into 16 query heads.
    auto key_rep = k_embed
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    auto self_attn = at::matmul(q_embed, key_rep.transpose(2, 3)) * self_scaling;  // [B,16,S,S]

    // Internal causal mask: upper triangle (above the diagonal) set to -inf.
    auto causal_mask = at::triu(
        at::full({seq_len, seq_len},
                 -std::numeric_limits<float>::infinity(), fp32_opts),
        /*diagonal=*/1);
    self_attn = self_attn + causal_mask.unsqueeze(0).unsqueeze(0);

    self_attn = at::softmax(self_attn, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                    .to(query_states.scalar_type());
    auto self_out = at::matmul(self_attn, value_rep);  // [B,16,S,128]

    self_out = self_out.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
    self_out = at::matmul(self_out, self_attn_o_weight.t());  // [B,S,hidden]

    auto hidden_post_self = residual + self_out;

    // --- Cross-attention block: plain MHA over the encoder states. ---
    residual = hidden_post_self;
    normed = rms_norm(residual, cross_attn_norm_weight, norm_eps);

    auto cross_q = at::matmul(normed, cross_attn_q_weight.t());                 // [B,S,q_dim]
    auto cross_k = at::matmul(encoder_hidden_states, cross_attn_k_weight.t());  // [B,E,kv_dim]
    auto cross_v = at::matmul(encoder_hidden_states, cross_attn_v_weight.t());  // [B,E,kv_dim]

    cross_q = cross_q.view({batch_size, seq_len, kCrossNumHeads, kCrossHeadDim})
                  .transpose(1, 2);  // [B,16,S,128]
    cross_k = cross_k.view({batch_size, encoder_seq_len, kCrossNumKVHeads, kCrossHeadDim})
                  .transpose(1, 2);  // [B,16,E,128]
    cross_v = cross_v.view({batch_size, encoder_seq_len, kCrossNumKVHeads, kCrossHeadDim})
                  .transpose(1, 2);  // [B,16,E,128]

    // 16 query heads == 16 kv heads -> no repeat_kv. No RoPE, no mask.
    auto cross_attn = at::matmul(cross_q, cross_k.transpose(2, 3)) * cross_scaling;  // [B,16,S,E]
    cross_attn = at::softmax(cross_attn, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                     .to(cross_q.scalar_type());
    auto cross_out = at::matmul(cross_attn, cross_v);  // [B,16,S,128]

    cross_out = cross_out.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
    cross_out = at::matmul(cross_out, cross_attn_o_weight.t());  // [B,S,hidden]

    auto hidden_post_cross = residual + cross_out;

    // --- MLP block: pre-norm SwiGLU. ---
    residual = hidden_post_cross;
    normed = rms_norm(residual, mlp_norm_weight, norm_eps);

    auto gate = at::matmul(normed, mlp_gate_weight.t());  // [B,S,intermediate]
    auto up = at::matmul(normed, mlp_up_weight.t());      // [B,S,intermediate]
    auto mlp_out = at::matmul(at::silu(gate) * up, mlp_down_weight.t());  // [B,S,hidden]

    auto result = residual + mlp_out;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Complete decoder layer via ATen (RMSNorm, GQA self-attention with "
          "rotate-half RoPE and causal mask, MHA cross-attention, SwiGLU MLP; "
          "no custom CUDA kernels); bf16 in/out with fp32 norm/softmax "
          "accumulation; destination-passing style (output).");
}
