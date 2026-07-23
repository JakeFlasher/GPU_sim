#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>
#include <cmath>
#include <limits>

// Fused decoder layer: pre-attention RMSNorm, GQA self-attention with
// multi-modal 3D RoPE and an internally-built causal mask, a residual, a
// post-attention RMSNorm, a SwiGLU MLP, and a second residual. Pattern:
// x = x + attn(norm(x)); x = x + mlp(norm(x)). All math is fp32.
//
// Reference (problem_019_decoder_layer_fused_attention_mlp), op-for-op:
//   hidden_size    = 3584
//   num_heads      = 28
//   num_kv_heads   = 4
//   head_dim       = 128
//   num_kv_groups  = num_heads // num_kv_heads = 7
//   scaling        = head_dim ** -0.5
//   mrope_section          = [16, 24, 24]   # temporal, height, width
//   mrope_section_doubled  = [32, 48, 48]   # cos/sin pair widths
//
//   rms_norm(x, weight, eps):
//     x_fp32   = x.to(float32)
//     variance = x_fp32.pow(2).mean(-1, keepdim=True)
//     x_normed = x_fp32 * rsqrt(variance + eps)
//     return weight * x_normed.to(input_dtype)   # input is fp32 here
//
//   --- Attention block ---
//   residual      = hidden_states
//   hidden_states = rms_norm(hidden_states, input_layernorm_weight, eps)
//   q = linear(hidden_states, q_proj_weight, q_proj_bias)   # [B,S,28*128]
//   k = linear(hidden_states, k_proj_weight, k_proj_bias)   # [B,S, 4*128]
//   v = linear(hidden_states, v_proj_weight, v_proj_bias)   # [B,S, 4*128]
//   q = q.view(B,S,28,128).transpose(1,2)                   # [B,28,S,128]
//   k = k.view(B,S, 4,128).transpose(1,2)                   # [B, 4,S,128]
//   v = v.view(B,S, 4,128).transpose(1,2)                   # [B, 4,S,128]
//
//   # rope_cos / rope_sin arrive as [3, B, kv_seq_len, 128]. The combined
//   # plane gathers section i (width doubled[i], starting at sum(doubled[:i]))
//   # from rope plane i % 3, then trims to seq_len. With three sections and
//   # i % 3 == i: plane 0 -> [0:32], plane 1 -> [32:80], plane 2 -> [80:128].
//   cos_combined = cat([ cos[i][..., start_i:start_i+doubled[i]] ], -1)[:, :S].unsqueeze(1)
//   sin_combined = cat([ sin[i][..., start_i:start_i+doubled[i]] ], -1)[:, :S].unsqueeze(1)
//
//   rotate_half(x) = cat((-x[..., 64:], x[..., :64]), dim=-1)
//   q = q * cos_combined + rotate_half(q) * sin_combined
//   k = k * cos_combined + rotate_half(k) * sin_combined
//
//   k = repeat_kv(k, 7) ; v = repeat_kv(v, 7)               # [B,28,S,128]
//   attn = (q @ k.transpose(2,3)) * scaling                  # [B,28,S,S]
//   causal_mask = triu(full((S,S), -inf), diagonal=1)        # internal mask
//   attn = attn + causal_mask
//   attn = softmax(attn, dim=-1, dtype=float32).to(fp32)
//   attn_output = attn @ v                                   # [B,28,S,128]
//   attn_output = attn_output.transpose(1,2).contiguous().view(B,S,hidden)
//   attn_output = linear(attn_output, o_proj_weight)
//   hidden_states = residual + attn_output
//
//   --- MLP block ---
//   residual      = hidden_states
//   hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight, eps)
//   gate = silu(linear(hidden_states, gate_proj_weight))
//   up   = linear(hidden_states, up_proj_weight)
//   hidden_states = linear(gate * up, down_proj_weight)
//   output = residual + hidden_states
//
// Per definition.json (insertion order):
//   hidden_states                    : [B, S, hidden]            fp32
//   input_layernorm_weight           : [hidden]                  fp32
//   q_proj_weight                    : [28*128, hidden]          fp32
//   q_proj_bias                      : [28*128]                  fp32
//   k_proj_weight                    : [4*128, hidden]           fp32
//   k_proj_bias                      : [4*128]                   fp32
//   v_proj_weight                    : [4*128, hidden]           fp32
//   v_proj_bias                      : [4*128]                   fp32
//   o_proj_weight                    : [hidden, 28*128]          fp32
//   rope_cos                         : [3, B, kv_seq_len, 128]   fp32
//   rope_sin                         : [3, B, kv_seq_len, 128]   fp32
//   post_attention_layernorm_weight  : [hidden]                  fp32
//   gate_proj_weight                 : [intermediate, hidden]    fp32
//   up_proj_weight                   : [intermediate, hidden]    fp32
//   down_proj_weight                 : [hidden, intermediate]    fp32
//   rms_norm_eps                     : scalar                    fp32 -> double
//   output                           : [B, S, hidden]            fp32 DPS
//
// Notes on fidelity:
//   * The mrope combine is a section gather, not an interleave: each of the
//     three head-dim sections is read from its own rope plane, stitched back
//     along the head dim, trimmed to seq_len along the sequence axis, then
//     broadcast over heads via unsqueeze(1). The rotation itself is the
//     rotate-half form cat((-x2, x1)) applied with the gathered cos / sin.
//   * The causal mask is built internally with triu of a [-inf]-filled
//     [S, S] matrix at diagonal 1 -- there is no mask input. The fill value
//     is -inf so masked positions vanish under softmax.
//   * repeat_kv expands the 4 kv heads into 28 query heads via an inserted
//     group axis, expand, and reshape -- the same path the reference takes.
//   * Precision: everything is fp32 -- inputs, weights, the RMSNorm variance /
//     rsqrt, the softmax, and the output. No casts are introduced; fp32 is the
//     contractual dtype on every surface.
//   * SwiGLU: gate = silu(gate_proj); the gate is multiplied by the up
//     projection, then the down projection is applied.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// RMSNorm over the last dim, computed entirely in fp32 (inputs are already
// fp32). variance = mean(x^2); x *= rsqrt(variance + eps); scale by weight.
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
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& q_proj_bias,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& k_proj_bias,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& v_proj_bias,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& rope_cos,
    const torch::Tensor& rope_sin,
    const torch::Tensor& post_attention_layernorm_weight,
    const torch::Tensor& gate_proj_weight,
    const torch::Tensor& up_proj_weight,
    const torch::Tensor& down_proj_weight,
    double rms_norm_eps,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(input_layernorm_weight, "input_layernorm_weight");
    check_fp32_cuda(q_proj_weight, "q_proj_weight");
    check_fp32_cuda(q_proj_bias, "q_proj_bias");
    check_fp32_cuda(k_proj_weight, "k_proj_weight");
    check_fp32_cuda(k_proj_bias, "k_proj_bias");
    check_fp32_cuda(v_proj_weight, "v_proj_weight");
    check_fp32_cuda(v_proj_bias, "v_proj_bias");
    check_fp32_cuda(o_proj_weight, "o_proj_weight");
    check_fp32_cuda(rope_cos, "rope_cos");
    check_fp32_cuda(rope_sin, "rope_sin");
    check_fp32_cuda(post_attention_layernorm_weight, "post_attention_layernorm_weight");
    check_fp32_cuda(gate_proj_weight, "gate_proj_weight");
    check_fp32_cuda(up_proj_weight, "up_proj_weight");
    check_fp32_cuda(down_proj_weight, "down_proj_weight");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(rope_cos.dim() == 4,
                "rope_cos must be 4D [3, B, kv_seq_len, head_dim], got ", rope_cos.dim(), "D");
    TORCH_CHECK(rope_sin.dim() == 4,
                "rope_sin must be 4D [3, B, kv_seq_len, head_dim], got ", rope_sin.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 3584;
    constexpr int64_t kNumHeads = 28;
    constexpr int64_t kNumKVHeads = 4;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 7
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 64

    // mrope section widths doubled for the cos/sin pairs: [32, 48, 48].
    const std::vector<int64_t> mrope_doubled{32, 48, 48};

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const double scaling = std::pow(static_cast<double>(kHeadDim), -0.5);

    // --- Attention block. ---
    auto residual = hidden_states;
    auto normed = rms_norm(residual, input_layernorm_weight, rms_norm_eps);

    // Q/K/V projections with bias via at::linear (x @ weight.t() + bias).
    auto query_states = at::linear(normed, q_proj_weight, q_proj_bias);  // [B,S,28*128]
    auto key_states = at::linear(normed, k_proj_weight, k_proj_bias);    // [B,S, 4*128]
    auto value_states = at::linear(normed, v_proj_weight, v_proj_bias);  // [B,S, 4*128]

    // Reshape to heads then transpose seq<->head.
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,28,S,128]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,4,S,128]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,4,S,128]

    // Multi-modal 3D RoPE: gather each head-dim section from its own rope
    // plane, stitch them back along the head dim, trim to seq_len, then
    // broadcast over heads.
    std::vector<torch::Tensor> cos_parts;
    std::vector<torch::Tensor> sin_parts;
    cos_parts.reserve(mrope_doubled.size());
    sin_parts.reserve(mrope_doubled.size());
    int64_t start = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(mrope_doubled.size()); ++i) {
        const int64_t width = mrope_doubled[i];
        const int64_t plane = i % 3;
        cos_parts.push_back(rope_cos[plane].narrow(/*dim=*/-1, /*start=*/start, /*length=*/width));
        sin_parts.push_back(rope_sin[plane].narrow(/*dim=*/-1, /*start=*/start, /*length=*/width));
        start += width;
    }
    auto cos_combined = at::cat(cos_parts, -1)
                            .narrow(/*dim=*/1, /*start=*/0, seq_len)
                            .unsqueeze(1);  // [B,1,S,128]
    auto sin_combined = at::cat(sin_parts, -1)
                            .narrow(/*dim=*/1, /*start=*/0, seq_len)
                            .unsqueeze(1);  // [B,1,S,128]

    // rotate_half: cat((-x2, x1)) where x1/x2 are the two head-dim halves.
    auto q_half1 = query_states.narrow(/*dim=*/-1, /*start=*/0, kHalfDim);
    auto q_half2 = query_states.narrow(/*dim=*/-1, /*start=*/kHalfDim, kHalfDim);
    auto q_rotated = at::cat({-q_half2, q_half1}, -1);
    auto q_embed = (query_states * cos_combined) + (q_rotated * sin_combined);

    auto k_half1 = key_states.narrow(/*dim=*/-1, /*start=*/0, kHalfDim);
    auto k_half2 = key_states.narrow(/*dim=*/-1, /*start=*/kHalfDim, kHalfDim);
    auto k_rotated = at::cat({-k_half2, k_half1}, -1);
    auto k_embed = (key_states * cos_combined) + (k_rotated * sin_combined);

    // repeat_kv: expand the 4 kv heads into 28 query heads.
    auto key_rep = k_embed
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // Scaled QK^T, internally-built causal mask, fp32 softmax, attention @ V.
    auto attn_weights = at::matmul(q_embed, key_rep.transpose(2, 3)) * scaling;  // [B,28,S,S]
    auto neg_inf = -std::numeric_limits<float>::infinity();
    auto causal_mask = at::triu(
        at::full({seq_len, seq_len}, neg_inf, hidden_states.options()),
        /*diagonal=*/1
    );
    attn_weights = attn_weights + causal_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                       .to(query_states.scalar_type());
    auto attn_output = at::matmul(attn_weights, value_rep);  // [B,28,S,128]

    // Unwind heads and apply the output projection (no bias).
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, kHiddenSize});
    attn_output = at::linear(attn_output, o_proj_weight);  // [B,S,hidden]

    auto hidden_post_attn = residual + attn_output;

    // --- MLP block: post-attention norm then SwiGLU. ---
    auto residual_mlp = hidden_post_attn;
    auto normed_mlp = rms_norm(residual_mlp, post_attention_layernorm_weight, rms_norm_eps);

    auto gate = at::silu(at::linear(normed_mlp, gate_proj_weight));  // [B,S,intermediate]
    auto up = at::linear(normed_mlp, up_proj_weight);                // [B,S,intermediate]
    auto mlp_out = at::linear(gate * up, down_proj_weight);          // [B,S,hidden]

    auto result = residual_mlp + mlp_out;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Fused decoder layer via ATen (pre-attn RMSNorm, GQA self-attention "
          "with multi-modal 3D RoPE and an internal triu causal mask, SwiGLU "
          "MLP, two residuals; no custom CUDA kernels); fp32 throughout; "
          "destination-passing style (output).");
}
