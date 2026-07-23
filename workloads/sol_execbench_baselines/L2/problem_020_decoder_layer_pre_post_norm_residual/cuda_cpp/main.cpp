#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>
#include <cmath>

// Pre/post-norm decoder layer: pre-norm GQA self-attention with RoPE and an
// additive attention mask feeding a residual, then a pre-norm SwiGLU MLP
// feeding a second residual. Pattern: x = x + attn(norm(x)); x = x + mlp(norm(x)).
//
// Reference (problem_020_decoder_layer_pre_post_norm_residual), op-for-op:
//   num_attention_heads  = 64
//   num_key_value_heads  = 8
//   head_dim             = 96
//   num_key_value_groups = num_attention_heads // num_key_value_heads = 8
//   scaling              = head_dim ** -0.5
//   half_dim             = head_dim // 2 = 48
//
//   rms_norm(x, weight, eps):
//     x_fp32   = x.to(float32)
//     variance = x_fp32.pow(2).mean(-1, keepdim=True)
//     x_normed = x_fp32 * rsqrt(variance + eps)
//     return weight * x_normed.to(input_dtype)   # downcast to bf16 before scale
//
//   --- Attention block ---
//   residual1     = hidden_states
//   hidden_states = rms_norm(hidden_states, input_layernorm_weight, eps)
//   q = linear(hidden_states, q_proj_weight)        # [B,S,qkv_hidden]
//   k = linear(hidden_states, k_proj_weight)        # [B,S,kv_hidden]
//   v = linear(hidden_states, v_proj_weight)        # [B,S,kv_hidden]
//   q = q.view(B,S,64,96).transpose(1,2)            # [B,64,S,96]
//   k = k.view(B,S, 8,96).transpose(1,2)            # [B, 8,S,96]
//   v = v.view(B,S, 8,96).transpose(1,2)            # [B, 8,S,96]
//   cos_e = cos.unsqueeze(1) ; sin_e = sin.unsqueeze(1)   # [B,1,S,48]
//   # interleaved-half RoPE on each of q and k:
//   #   x1 = x[..., :48] ; x2 = x[..., 48:]
//   #   x  = cat([x1*cos_e - x2*sin_e, x1*sin_e + x2*cos_e], dim=-1)
//   k = repeat_kv(k, 8) ; v = repeat_kv(v, 8)       # [B,64,S,96]
//   attn = (q @ k.transpose(2,3)) * scaling         # [B,64,S,S]
//   attn = attn + attention_mask                    # additive mask, [B,1,S,S]
//   attn = softmax(attn, dim=-1, dtype=float32).to(bf16)
//   attn_output = attn @ v                          # [B,64,S,96]
//   attn_output = attn_output.transpose(1,2).contiguous().view(B,S,-1)
//   attn_output = linear(attn_output, o_proj_weight)
//   hidden_states = residual1 + attn_output
//
//   --- MLP block ---
//   residual2     = hidden_states
//   hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight, eps)
//   gate = silu(linear(hidden_states, gate_proj_weight))
//   up   = linear(hidden_states, up_proj_weight)
//   hidden_states = linear(gate * up, down_proj_weight)
//   output = residual2 + hidden_states
//
// Per definition.json (insertion order):
//   hidden_states                    : [B, S, hidden]            bf16
//   cos                              : [B, S, half_head_dim]     bf16
//   sin                              : [B, S, half_head_dim]     bf16
//   attention_mask                   : [B, 1, S, S]              bf16
//   input_layernorm_weight           : [hidden]                  bf16
//   q_proj_weight                    : [qkv_hidden, hidden]      bf16
//   k_proj_weight                    : [kv_hidden, hidden]       bf16
//   v_proj_weight                    : [kv_hidden, hidden]       bf16
//   o_proj_weight                    : [hidden, qkv_hidden]      bf16
//   post_attention_layernorm_weight  : [hidden]                  bf16
//   gate_proj_weight                 : [intermediate, hidden]    bf16
//   up_proj_weight                   : [intermediate, hidden]    bf16
//   down_proj_weight                 : [hidden, intermediate]    bf16
//   rms_norm_eps                     : scalar                    fp32 -> double
//   output                           : [B, S, hidden]            bf16 DPS
//
// Notes on fidelity:
//   * RoPE here is the interleaved-half variant: the head dim splits into a
//     first half and a second half, and the two output halves are built
//     explicitly as [x1*cos - x2*sin, x1*sin + x2*cos] from the supplied
//     cos / sin (each of width half_head_dim). This is distinct from the
//     rotate_half(cat(-x2, x1)) style used elsewhere.
//   * cos / sin arrive as [B, S, half_head_dim] and get unsqueeze(1) so they
//     broadcast across the head axis; each is half the head dim and applies to
//     both halves of the split.
//   * The attention mask is an additive bf16 tensor of shape [B, 1, S, S]; it
//     is added to the scaled scores before softmax (no internal mask
//     construction).
//   * repeat_kv expands the 8 kv heads into 64 query heads via an inserted
//     group axis, expand, and reshape -- the same path the reference takes.
//   * Precision: inputs and outputs are bf16. The RMSNorm variance / rsqrt and
//     the softmax run in fp32, then the normalized value is cast back to bf16
//     before the weight scale (matching weight * x_normed.to(input_dtype)) and
//     the softmax weights are cast back to bf16 before the value matmul. The
//     bf16 output is never upcast; bf16 is the contractual dtype, not a
//     downgrade.
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

// RMSNorm over the last dim. The variance / rsqrt run in fp32; the normalized
// value is downcast to bf16 before the bf16 weight scale, matching the
// reference's weight * x_normed.to(input_dtype).
static torch::Tensor rms_norm(
    const torch::Tensor& x, const torch::Tensor& weight, double eps
) {
    auto x_fp32 = x.to(torch::kFloat32);
    auto variance = x_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto x_normed = x_fp32 * at::rsqrt(variance + eps);
    return weight * x_normed.to(x.scalar_type());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& attention_mask,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& post_attention_layernorm_weight,
    const torch::Tensor& gate_proj_weight,
    const torch::Tensor& up_proj_weight,
    const torch::Tensor& down_proj_weight,
    double rms_norm_eps,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(cos, "cos");
    check_bf16_cuda(sin, "sin");
    check_bf16_cuda(attention_mask, "attention_mask");
    check_bf16_cuda(input_layernorm_weight, "input_layernorm_weight");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_cuda(post_attention_layernorm_weight, "post_attention_layernorm_weight");
    check_bf16_cuda(gate_proj_weight, "gate_proj_weight");
    check_bf16_cuda(up_proj_weight, "up_proj_weight");
    check_bf16_cuda(down_proj_weight, "down_proj_weight");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(cos.dim() == 3, "cos must be 3D [B, S, half_head_dim], got ", cos.dim(), "D");
    TORCH_CHECK(sin.dim() == 3, "sin must be 3D [B, S, half_head_dim], got ", sin.dim(), "D");
    TORCH_CHECK(attention_mask.dim() == 4,
                "attention_mask must be 4D [B, 1, S, S], got ", attention_mask.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 64;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 96;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 8
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 48

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const double scaling = std::pow(static_cast<double>(kHeadDim), -0.5);

    // --- Attention block. ---
    auto residual1 = hidden_states;
    auto normed = rms_norm(residual1, input_layernorm_weight, rms_norm_eps);

    // Q/K/V projections (no bias). matmul with explicit weight transpose.
    auto query_states = at::matmul(normed, q_proj_weight.t());  // [B,S,qkv_hidden]
    auto key_states = at::matmul(normed, k_proj_weight.t());    // [B,S,kv_hidden]
    auto value_states = at::matmul(normed, v_proj_weight.t());  // [B,S,kv_hidden]

    // Reshape to heads then transpose seq<->head.
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,64,S,96]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,8,S,96]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,8,S,96]

    // RoPE from the supplied cos / sin: unsqueeze(1) broadcasts over heads.
    // Interleaved-half rotation: split into first / second half, then build
    // [x1*cos - x2*sin, x1*sin + x2*cos].
    auto cos_e = cos.unsqueeze(1);  // [B,1,S,48]
    auto sin_e = sin.unsqueeze(1);  // [B,1,S,48]

    auto q1 = query_states.narrow(/*dim=*/-1, /*start=*/0, kHalfDim);
    auto q2 = query_states.narrow(/*dim=*/-1, /*start=*/kHalfDim, kHalfDim);
    auto q_embed = at::cat({q1 * cos_e - q2 * sin_e, q1 * sin_e + q2 * cos_e}, -1);

    auto k1 = key_states.narrow(/*dim=*/-1, /*start=*/0, kHalfDim);
    auto k2 = key_states.narrow(/*dim=*/-1, /*start=*/kHalfDim, kHalfDim);
    auto k_embed = at::cat({k1 * cos_e - k2 * sin_e, k1 * sin_e + k2 * cos_e}, -1);

    // repeat_kv: expand the 8 kv heads into 64 query heads.
    auto key_rep = k_embed
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // Scaled QK^T, additive mask, softmax (fp32) cast back to bf16, attention @ V.
    auto attn_weights = at::matmul(q_embed, key_rep.transpose(2, 3)) * scaling;  // [B,64,S,S]
    attn_weights = attn_weights + attention_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                       .to(query_states.scalar_type());
    auto attn_output = at::matmul(attn_weights, value_rep);  // [B,64,S,96]

    // Unwind heads and apply the output projection.
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
    attn_output = at::matmul(attn_output, o_proj_weight.t());  // [B,S,hidden]

    auto hidden_post_attn = residual1 + attn_output;

    // --- MLP block: pre-norm SwiGLU. ---
    auto residual2 = hidden_post_attn;
    auto normed_mlp = rms_norm(residual2, post_attention_layernorm_weight, rms_norm_eps);

    auto gate = at::silu(at::matmul(normed_mlp, gate_proj_weight.t()));  // [B,S,intermediate]
    auto up = at::matmul(normed_mlp, up_proj_weight.t());                // [B,S,intermediate]
    auto mlp_out = at::matmul(gate * up, down_proj_weight.t());          // [B,S,hidden]

    auto result = residual2 + mlp_out;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Pre/post-norm decoder layer via ATen (RMSNorm, GQA self-attention "
          "with interleaved-half RoPE and additive mask, SwiGLU MLP; no custom "
          "CUDA kernels); bf16 in/out with fp32 norm/softmax accumulation; "
          "destination-passing style (output).");
}
