#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>

// Complete Qwen3 decoder layer block: pre-norm grouped-query self-attention
// with per-head Q/K RMSNorm and RoPE feeding a residual, then a pre-norm
// SwiGLU MLP feeding a second residual. Three families of RMSNorm (the input
// norm, the per-head Q/K norms, the post-attention norm) and two residual
// connections.
//
// Reference (problem_059_decoder_layer_full_block), op-for-op:
//   hidden_size           = 5120   num_attention_heads = 40
//   num_key_value_heads   = 8      head_dim            = 128
//   intermediate_size     = 17408  num_key_value_groups = 40 / 8 = 5
//
//   rms_norm(x, weight, eps):
//     x_fp32   = x.to(float32)
//     variance = x_fp32.pow(2).mean(-1, keepdim=True)
//     x_normed = x_fp32 * rsqrt(variance + eps)
//     return (weight * x_normed).to(input_dtype)   # scale in fp32, then bf16
//
//   rotate_half(x) = cat([-x[..., d/2:], x[..., :d/2]], dim=-1)
//
//   --- Self-attention block (GQA + per-head QK norm + RoPE + additive mask) ---
//   residual = hidden_states
//   hidden_states = rms_norm(hidden_states, input_layernorm_weight, eps)
//   q = linear(hidden_states, q_proj_weight).view(B,S,40,128)
//   k = linear(hidden_states, k_proj_weight).view(B,S, 8,128)
//   v = linear(hidden_states, v_proj_weight).view(B,S, 8,128)
//   q = rms_norm(q, q_norm_weight, eps)    # norm over the trailing head_dim
//   k = rms_norm(k, k_norm_weight, eps)
//   q = q.transpose(1,2) ; k = k.transpose(1,2) ; v = v.transpose(1,2)
//   # RoPE cos / sin from the supplied inv_freq and integer position_ids:
//   freqs = (inv_freq[None,:,None] @ position_ids[:,None,:].float()).transpose(1,2)
//   emb   = cat([freqs, freqs], dim=-1)
//   cos = emb.cos().to(bf16).unsqueeze(1) ; sin = emb.sin().to(bf16).unsqueeze(1)
//   q = q*cos + rotate_half(q)*sin ; k = k*cos + rotate_half(k)*sin
//   k = repeat_kv(k, 5) ; v = repeat_kv(v, 5)            # [B,40,S,128]
//   attn = (q @ k.transpose(2,3)) * attention_scale
//   attn = attn + attention_mask                         # additive bf16 mask
//   attn = softmax(attn, dim=-1, dtype=float32).to(bf16)
//   o    = (attn @ v).transpose(1,2).contiguous().view(B,S,40*128)
//   o    = linear(o, o_proj_weight)
//   hidden_states = residual + o
//
//   --- MLP block (SwiGLU) ---
//   residual = hidden_states
//   hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight, eps)
//   gate = linear(hidden_states, gate_proj_weight) ; up = linear(hidden_states, up_proj_weight)
//   hidden_states = linear(silu(gate) * up, down_proj_weight)
//   output = residual + hidden_states
//
// Per definition.json (insertion order):
//   hidden_states                    : [B, S, hidden]                bf16
//   position_ids                     : [B, S]                        int64
//   attention_mask                   : [B, 1, S, S]                  bf16
//   input_layernorm_weight           : [hidden]                      bf16
//   q_proj_weight                    : [q_dim, hidden]               bf16
//   k_proj_weight                    : [kv_dim, hidden]              bf16
//   v_proj_weight                    : [kv_dim, hidden]              bf16
//   q_norm_weight                    : [head_dim]                    bf16
//   k_norm_weight                    : [head_dim]                    bf16
//   o_proj_weight                    : [hidden, q_dim]               bf16
//   post_attention_layernorm_weight  : [hidden]                      bf16
//   gate_proj_weight                 : [intermediate, hidden]        bf16
//   up_proj_weight                   : [intermediate, hidden]        bf16
//   down_proj_weight                 : [hidden, intermediate]        bf16
//   inv_freq                         : [head_dim/2]                  fp32
//   rms_norm_eps                     : scalar                        fp32 -> double
//   attention_scale                  : scalar                        fp32 -> double
//   output                           : [B, S, hidden]                bf16 DPS
//
// Notes on fidelity:
//   * RoPE here is built from the supplied inv_freq input and the integer
//     position_ids input (not from an internal theta / arange grid). freqs is
//     the outer product of inv_freq with the per-position ids, computed in
//     fp32; emb = cat([freqs, freqs]); cos / sin are taken in fp32 then cast to
//     bf16 before the rotation, matching emb.cos().to(bf16). rotate_half swaps
//     and negates the two halves: cat([-x2, x1]).
//   * The per-head Q/K RMSNorm runs over the trailing head_dim while q / k are
//     still in [B,S,heads,head_dim] layout, before the transpose to
//     [B,heads,S,head_dim].
//   * The attention mask is the supplied bf16 [B,1,S,S] tensor, added to the
//     scaled scores (it broadcasts across the head axis); it is not an internal
//     causal triangle.
//   * GQA: 40 query heads, 8 kv heads (5 groups), so repeat_kv expands the 8 kv
//     heads to 40 before the score / value matmuls.
//   * Precision: inputs and outputs are bf16. Every RMSNorm variance / rsqrt and
//     the softmax run in fp32; the normalized value is cast back to bf16 after
//     the weight scale and the softmax weights are cast back to bf16 before the
//     value matmul. The bf16 output is never upcast; bf16 is the contractual
//     dtype, not a downgrade.
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
    const torch::Tensor& position_ids,
    const torch::Tensor& attention_mask,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& post_attention_layernorm_weight,
    const torch::Tensor& gate_proj_weight,
    const torch::Tensor& up_proj_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& inv_freq,
    double rms_norm_eps,
    double attention_scale,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(attention_mask, "attention_mask");
    check_bf16_cuda(input_layernorm_weight, "input_layernorm_weight");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(q_norm_weight, "q_norm_weight");
    check_bf16_cuda(k_norm_weight, "k_norm_weight");
    check_bf16_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_cuda(post_attention_layernorm_weight, "post_attention_layernorm_weight");
    check_bf16_cuda(gate_proj_weight, "gate_proj_weight");
    check_bf16_cuda(up_proj_weight, "up_proj_weight");
    check_bf16_cuda(down_proj_weight, "down_proj_weight");
    check_bf16_cuda(output, "output");
    TORCH_CHECK(position_ids.is_cuda(), "position_ids must be a CUDA tensor");
    TORCH_CHECK(position_ids.dtype() == torch::kInt64,
                "position_ids must be int64, got ", position_ids.dtype());
    TORCH_CHECK(inv_freq.is_cuda(), "inv_freq must be a CUDA tensor");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 40;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 5

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);

    // --- Self-attention block: GQA + per-head QK norm + RoPE + additive mask. ---
    auto residual = hidden_states;
    auto normed = rms_norm(residual, input_layernorm_weight, rms_norm_eps);

    auto query_states = at::matmul(normed, q_proj_weight.t());  // [B,S,q_dim]
    auto key_states = at::matmul(normed, k_proj_weight.t());    // [B,S,kv_dim]
    auto value_states = at::matmul(normed, v_proj_weight.t());  // [B,S,kv_dim]

    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim});
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});

    // Per-head Q/K RMSNorm over the trailing head_dim, in [B,S,heads,head_dim].
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps);
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps);

    query_states = query_states.transpose(1, 2);  // [B,40,S,128]
    key_states = key_states.transpose(1, 2);      // [B,8,S,128]
    value_states = value_states.transpose(1, 2);  // [B,8,S,128]

    // RoPE cos / sin from the supplied inv_freq and the integer position_ids.
    // inv_freq_expanded is [B, head_dim/2, 1]; position_ids_expanded is
    // [B, 1, S]; their batched product is [B, head_dim/2, S], transposed to
    // [B, S, head_dim/2]. emb = cat([freqs, freqs]) is [B, S, head_dim].
    auto inv_freq_expanded =
        inv_freq.view({1, -1, 1}).to(torch::kFloat32)
            .expand({batch_size, -1, 1});
    auto position_ids_expanded =
        position_ids.unsqueeze(1).to(torch::kFloat32);  // [B,1,S]
    auto freqs = at::matmul(inv_freq_expanded, position_ids_expanded)
                     .transpose(1, 2);              // [B,S,head_dim/2]
    auto emb = at::cat({freqs, freqs}, -1);         // [B,S,head_dim]
    // Add the head axis so cos / sin broadcast across the 40 / 8 heads.
    auto cos = emb.cos().to(query_states.scalar_type()).unsqueeze(1);  // [B,1,S,head_dim]
    auto sin = emb.sin().to(query_states.scalar_type()).unsqueeze(1);

    auto q_embed = query_states * cos + rotate_half(query_states) * sin;
    auto k_embed = key_states * cos + rotate_half(key_states) * sin;

    // repeat_kv: expand the 8 kv heads into 40 query heads (5 groups).
    auto key_rep = k_embed
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    auto attn = at::matmul(q_embed, key_rep.transpose(2, 3)) * attention_scale;  // [B,40,S,S]
    attn = attn + attention_mask;  // [B,1,S,S] broadcasts across the head axis
    attn = at::softmax(attn, /*dim=*/-1, /*dtype=*/torch::kFloat32)
               .to(query_states.scalar_type());
    auto attn_out = at::matmul(attn, value_rep);  // [B,40,S,128]

    attn_out = attn_out.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
    attn_out = at::matmul(attn_out, o_proj_weight.t());  // [B,S,hidden]

    auto hidden_post_attn = residual + attn_out;

    // --- MLP block: pre-norm SwiGLU. ---
    residual = hidden_post_attn;
    normed = rms_norm(residual, post_attention_layernorm_weight, rms_norm_eps);

    auto gate = at::matmul(normed, gate_proj_weight.t());  // [B,S,intermediate]
    auto up = at::matmul(normed, up_proj_weight.t());      // [B,S,intermediate]
    auto mlp_out = at::matmul(at::silu(gate) * up, down_proj_weight.t());  // [B,S,hidden]

    auto result = residual + mlp_out;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Complete Qwen3 decoder layer via ATen (input RMSNorm, GQA "
          "self-attention with per-head QK RMSNorm, inv_freq/position_ids "
          "RoPE, additive mask, SwiGLU MLP; no custom CUDA kernels); bf16 "
          "in/out with fp32 norm/softmax accumulation; destination-passing "
          "style (output).");
}
