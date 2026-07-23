#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>
#include <cmath>

// Complete Mllama text decoder layer: pre-norm GQA self-attention with RoPE and
// an additive causal mask feeding a residual, then a pre-norm SwiGLU gated MLP
// feeding a second residual. Two RMSNorms, two residual connections. 32 query
// heads, 8 key-value heads (4:1 GQA ratio), head dim 128.
//
// Reference (problem_053_text_decoder_layer_with_self_attention_and_mlp),
// op-for-op:
//   num_heads            = 32   num_key_value_heads = 8
//   head_dim             = 128  num_key_value_groups = 4
//   intermediate_size    = 14336
//   scaling              = head_dim ** -0.5
//
//   rms_norm(x, weight):
//     x_fp32   = x.to(float32)
//     variance = x_fp32.pow(2).mean(-1, keepdim=True)
//     x        = x_fp32 * rsqrt(variance + eps)
//     return weight * x.to(input_dtype)   # normalize in fp32, cast, then scale
//
//   rotate_half(x) = cat([-x[..., d/2:], x[..., :d/2]], dim=-1)
//
//   apply_rope(q, k, seq_len):
//     inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2, fp32) / head_dim))
//     position_ids = arange(seq_len).unsqueeze(0).expand(bsz, -1)
//     freqs = (inv_freq[None,:,None] @ position_ids[:,None,:].float()).transpose(1,2)
//     emb   = cat([freqs, freqs], dim=-1)                 # [B,S,d], fp32
//     cos = emb.cos().unsqueeze(1) ; sin = emb.sin().unsqueeze(1)  # [B,1,S,d] fp32
//     q_embed = (q*cos + rotate_half(q)*sin).to(q.dtype)  # fp32 math, cast bf16
//     k_embed = (k*cos + rotate_half(k)*sin).to(k.dtype)
//
//   --- Self-attention block (GQA + RoPE + causal mask) ---
//   residual = hidden_states
//   normed   = rms_norm(hidden_states, input_layernorm_weight)
//   q = linear(normed, q_proj_weight).view(B,S,32,128).transpose(1,2)
//   k = linear(normed, k_proj_weight).view(B,S, 8,128).transpose(1,2)
//   v = linear(normed, v_proj_weight).view(B,S, 8,128).transpose(1,2)
//   q, k = apply_rope(q, k, seq_len)
//   k = repeat_kv(k) ; v = repeat_kv(v)                   # [B,32,S,128]
//   attn = (q @ k.transpose(2,3)) * scaling
//   attn = attn + attention_mask[:, :, :S, :S]            # additive causal mask
//   attn = softmax(attn, dim=-1, dtype=float32).to(bf16)
//   o    = (attn @ v).transpose(1,2).contiguous().view(B,S,32*128)
//   o    = linear(o, o_proj_weight)
//   hidden_states = residual + o
//
//   --- MLP block (SwiGLU) ---
//   residual = hidden_states
//   normed   = rms_norm(hidden_states, post_attention_layernorm_weight)
//   gate = silu(linear(normed, gate_proj_weight))
//   up   = linear(normed, up_proj_weight)
//   hidden_states = linear(gate * up, down_proj_weight)
//   output = residual + hidden_states
//
// Per definition.json (insertion order):
//   hidden_states                    : [B, S, hidden=4096]            bf16
//   attention_mask                   : [B, 1, S, S]                   bf16
//   q_proj_weight                    : [qkv_head_dim=4096, hidden]    bf16
//   k_proj_weight                    : [kv_head_dim=1024, hidden]     bf16
//   v_proj_weight                    : [kv_head_dim=1024, hidden]     bf16
//   o_proj_weight                    : [hidden, qkv_head_dim]         bf16
//   gate_proj_weight                 : [intermediate=14336, hidden]   bf16
//   up_proj_weight                   : [intermediate, hidden]         bf16
//   down_proj_weight                 : [hidden, intermediate]         bf16
//   input_layernorm_weight           : [hidden]                       bf16
//   post_attention_layernorm_weight  : [hidden]                       bf16
//   rms_norm_eps                     : scalar                         fp32 -> double
//   rope_theta                       : scalar                         fp32 -> double
//   output                           : [B, S, hidden]                 bf16 DPS
//
// Notes on fidelity:
//   * The two RMSNorms (input and post-attention) normalize over the last
//     (hidden) axis in fp32: variance / rsqrt in fp32, the normalized value
//     cast back to bf16, then the per-hidden weight applied. This matches the
//     reference's weight * x.to(input_dtype). Wider fp32 accumulation over
//     bf16 inputs is expected and is not a precision downgrade.
//   * RoPE is built from rope_theta via arange(0, head_dim, 2) / head_dim and
//     an internal arange(seq_len) position grid (this problem has no supplied
//     position_ids input). It is applied AFTER the transpose, on the
//     [B, heads, S, head_dim] layout, with cos / sin shaped [B, 1, S, head_dim]
//     so they broadcast over the head axis. The rotation runs in fp32 (cos /
//     sin stay fp32, bf16 q / k promote) and the result is cast back to bf16,
//     mirroring the reference's q_embed.to(q.dtype). RoPE applies to Q and K
//     only, never V.
//   * Attention logits are scaled by head_dim ** -0.5 before the additive mask,
//     matching the reference order. The mask is sliced to [:, :, :S, :S] as in
//     the reference.
//   * Softmax runs in fp32 and is cast back to bf16 before the value matmul.
//   * SwiGLU: gate = silu(gate_proj); the gate is multiplied by the up
//     projection, then the down projection is applied.
//   * bf16 is the contractual in / out dtype and is never upcast; the only
//     wider-precision work is the fp32 norm / RoPE / softmax accumulation.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

// RMSNorm over the last dim. The variance / rsqrt run in fp32; the fp32
// normalized value is cast back to bf16 and then scaled by the weight, matching
// the reference's weight * x.to(input_dtype).
static torch::Tensor rms_norm(
    const torch::Tensor& x, const torch::Tensor& weight, double eps
) {
    auto x_fp32 = x.to(torch::kFloat32);
    auto variance = x_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto x_normed = (x_fp32 * at::rsqrt(variance + eps)).to(x.scalar_type());
    return weight * x_normed;
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
    const torch::Tensor& attention_mask,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& gate_proj_weight,
    const torch::Tensor& up_proj_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& post_attention_layernorm_weight,
    double rms_norm_eps,
    double rope_theta,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(attention_mask, "attention_mask");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_cuda(gate_proj_weight, "gate_proj_weight");
    check_bf16_cuda(up_proj_weight, "up_proj_weight");
    check_bf16_cuda(down_proj_weight, "down_proj_weight");
    check_bf16_cuda(input_layernorm_weight, "input_layernorm_weight");
    check_bf16_cuda(post_attention_layernorm_weight, "post_attention_layernorm_weight");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 32;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 4
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 64

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const auto qdtype = hidden_states.scalar_type();
    const double scaling = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    // --- Self-attention block: GQA + RoPE + causal mask. ---
    auto residual = hidden_states;
    auto normed = rms_norm(residual, input_layernorm_weight, rms_norm_eps);

    auto query_states = at::linear(normed, q_proj_weight);  // [B,S,qkv_head_dim]
    auto key_states = at::linear(normed, k_proj_weight);    // [B,S,kv_head_dim]
    auto value_states = at::linear(normed, v_proj_weight);  // [B,S,kv_head_dim]

    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,32,S,128]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,8,S,128]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,8,S,128]

    // RoPE cos / sin built from rope_theta and an internal arange position grid.
    // inv_freq is [head_dim/2]; freqs come from the batched inv_freq @ positions
    // product; emb = cat([freqs, freqs]) is [B,S,head_dim]; cos / sin reshape to
    // [B,1,S,head_dim] to broadcast over the head axis. Kept in fp32 so the
    // rotation accumulates in fp32 before casting back to bf16.
    auto fp32_opts = hidden_states.options().dtype(torch::kFloat32);
    auto exponents = at::arange(/*start=*/0, /*end=*/kHeadDim, /*step=*/2, fp32_opts)
                         / static_cast<double>(kHeadDim);  // [head_dim/2]
    auto inv_freq = at::pow(rope_theta, exponents).reciprocal();  // [head_dim/2]
    auto positions = at::arange(seq_len, fp32_opts);             // [S]
    auto inv_freq_expanded =
        inv_freq.view({1, kHalfDim, 1}).expand({batch_size, kHalfDim, 1});  // [B,half,1]
    auto positions_expanded =
        positions.view({1, 1, seq_len}).expand({batch_size, 1, seq_len});  // [B,1,S]
    auto freqs = at::matmul(inv_freq_expanded, positions_expanded)
                     .transpose(1, 2);  // [B,half,S] -> [B,S,half]
    auto emb = at::cat({freqs, freqs}, -1);  // [B,S,head_dim], fp32
    auto cos = emb.cos().view({batch_size, 1, seq_len, kHeadDim});  // fp32
    auto sin = emb.sin().view({batch_size, 1, seq_len, kHeadDim});  // fp32

    // Rotation runs in fp32 (bf16 q / k promote against fp32 cos / sin) then
    // casts back to bf16, mirroring q_embed.to(q.dtype).
    auto q_embed = (query_states * cos + rotate_half(query_states) * sin).to(qdtype);
    auto k_embed = (key_states * cos + rotate_half(key_states) * sin).to(qdtype);

    // repeat_kv: expand the 8 kv heads into 32 query heads.
    auto key_rep = k_embed
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // QK^T, head_dim ** -0.5 scale, additive causal mask sliced to [:,:,:S,:S],
    // fp32 softmax cast back to bf16, value matmul.
    auto attn_weights = at::matmul(q_embed, key_rep.transpose(2, 3)) * scaling;  // [B,32,S,S]
    auto causal_mask = attention_mask
                           .narrow(/*dim=*/2, /*start=*/0, seq_len)
                           .narrow(/*dim=*/3, /*start=*/0, seq_len);
    attn_weights = attn_weights + causal_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32).to(qdtype);
    auto attn_out = at::matmul(attn_weights, value_rep);  // [B,32,S,128]

    attn_out = attn_out.transpose(1, 2).contiguous()
                   .reshape({batch_size, seq_len, kNumHeads * kHeadDim});
    attn_out = at::linear(attn_out, o_proj_weight);  // [B,S,hidden]

    auto hidden_post_attn = residual + attn_out;

    // --- MLP block: pre-norm SwiGLU. ---
    residual = hidden_post_attn;
    normed = rms_norm(residual, post_attention_layernorm_weight, rms_norm_eps);

    auto gate = at::silu(at::linear(normed, gate_proj_weight));  // [B,S,intermediate]
    auto up = at::linear(normed, up_proj_weight);                // [B,S,intermediate]
    auto mlp_out = at::linear(gate * up, down_proj_weight);      // [B,S,hidden]

    auto result = residual + mlp_out;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Mllama text decoder layer via ATen (RMSNorm, GQA self-attention with "
          "rotate-half RoPE and additive causal mask, SwiGLU MLP, two residuals; "
          "no custom CUDA kernels); bf16 in/out with fp32 norm/RoPE/softmax "
          "accumulation; destination-passing style (output).");
}
