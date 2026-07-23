#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <cmath>

// Encoder layer with two sequential residual-norm patterns: pre-norm GQA
// self-attention with a residual, then a pre-norm gated MLP with a residual.
//
// Reference (problem_063_encoder_layer_dual_residual_norm_chain), op-for-op:
//   num_heads            = 16
//   num_key_value_heads  = 4
//   head_dim             = 64
//   num_key_value_groups = num_heads // num_key_value_heads = 4
//
//   rms_norm(x, weight, eps):
//     x = x.to(float32)
//     variance = x.pow(2).mean(-1, keepdim=True)
//     x = x * rsqrt(variance + eps)
//     return weight * x        # input is already fp32 here
//
//   rotate_half(x) = cat((-x[..., d/2:], x[..., :d/2]), dim=-1)
//
//   --- First residual-norm pattern: self-attention ---
//   residual_1 = hidden_states
//   normed_1   = rms_norm(residual_1, pre_sa_norm_weight, norm_eps)
//   q = normed_1 @ q_proj_weight.t()          # [B,S,hidden]
//   k = normed_1 @ k_proj_weight.t()          # [B,S,256]
//   v = normed_1 @ v_proj_weight.t()          # [B,S,256]
//   q = q.view(B,S,16,64).transpose(1,2)      # [B,16,S,64]
//   k = k.view(B,S, 4,64).transpose(1,2)      # [B, 4,S,64]
//   v = v.view(B,S, 4,64).transpose(1,2)      # [B, 4,S,64]
//   cos_e = cos.unsqueeze(1) ; sin_e = sin.unsqueeze(1)   # [B,1,S,64]
//   q_embed = q * cos_e + rotate_half(q) * sin_e
//   k_embed = k * cos_e + rotate_half(k) * sin_e
//   k = repeat_kv(k_embed, 4) ; v = repeat_kv(v, 4)       # [B,16,S,64]
//   attn = (q_embed @ k.transpose(2,3)) / sqrt(head_dim)  # [B,16,S,S], non-causal
//   attn = softmax(attn, dim=-1, dtype=float32)
//   attn_output = attn @ v                                # [B,16,S,64]
//   attn_output = attn_output.transpose(1,2).contiguous().view(B,S,-1)
//   attn_out = attn_output @ o_proj_weight.t()            # [B,S,hidden]
//   hidden_states = residual_1 + attn_out
//
//   --- Second residual-norm pattern: gated MLP ---
//   residual_2 = hidden_states
//   normed_2   = rms_norm(residual_2, post_sa_norm_weight, norm_eps)
//   up = normed_2 @ gate_up_proj_weight.t()               # [B,S,intermediate*2]
//   gate, up = up.chunk(2, dim=-1)                        # each [B,S,intermediate]
//   up = up * silu(gate)
//   mlp_out = up @ down_proj_weight.t()                   # [B,S,hidden]
//   output  = residual_2 + mlp_out
//
// Per definition.json (insertion order):
//   hidden_states       : [B, S, hidden]                 fp32
//   cos                 : [B, S, head_dim]               fp32
//   sin                 : [B, S, head_dim]               fp32
//   pre_sa_norm_weight  : [hidden]                       fp32
//   q_proj_weight       : [hidden, hidden]               fp32
//   k_proj_weight       : [256, hidden]                  fp32
//   v_proj_weight       : [256, hidden]                  fp32
//   o_proj_weight       : [hidden, hidden]               fp32
//   post_sa_norm_weight : [hidden]                       fp32
//   gate_up_proj_weight : [intermediate*2, hidden]       fp32
//   down_proj_weight    : [hidden, intermediate]         fp32
//   norm_eps            : scalar                         fp32 -> double
//   output              : [B, S, hidden]                 fp32 DPS
//
// Notes on fidelity:
//   * Head counts (16 / 4 / 64) match the spec's fixed constants. The GQA
//     group count is num_heads / num_key_value_heads. k/v project to 256 =
//     num_key_value_heads * head_dim. Per-head dims are derived from the q
//     projection width so the reshape follows the actual tensor shape.
//   * RoPE is applied with the supplied cos / sin (no inv_freq derivation
//     here). cos / sin arrive as [B, S, head_dim] and get an unsqueeze(1) so
//     they broadcast across the head axis.
//   * repeat_kv expands the 4 kv heads into 16 query heads via an inserted
//     group axis, expand, and reshape -- the same path the reference takes.
//   * The encoder layer is non-causal: there is no attention mask in the
//     reference run(), so no masking is applied here.
//   * All math stays in fp32. The variance reduction, rsqrt, and softmax are
//     fp32, matching the reference dtype profile (fp32 in, fp32 out, no
//     intermediate downcast).
//   * The fused gate_up projection is split with chunk(2) on the last dim;
//     the first half is the gate, the second half is the up path, and the
//     SwiGLU combine is up * silu(gate).
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
    auto variance = x.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto normed = x * at::rsqrt(variance + eps);
    return weight * normed;
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& pre_sa_norm_weight,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& post_sa_norm_weight,
    const torch::Tensor& gate_up_proj_weight,
    const torch::Tensor& down_proj_weight,
    double norm_eps,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(cos, "cos");
    check_fp32_cuda(sin, "sin");
    check_fp32_cuda(pre_sa_norm_weight, "pre_sa_norm_weight");
    check_fp32_cuda(q_proj_weight, "q_proj_weight");
    check_fp32_cuda(k_proj_weight, "k_proj_weight");
    check_fp32_cuda(v_proj_weight, "v_proj_weight");
    check_fp32_cuda(o_proj_weight, "o_proj_weight");
    check_fp32_cuda(post_sa_norm_weight, "post_sa_norm_weight");
    check_fp32_cuda(gate_up_proj_weight, "gate_up_proj_weight");
    check_fp32_cuda(down_proj_weight, "down_proj_weight");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(cos.dim() == 3, "cos must be 3D [B, S, head_dim], got ", cos.dim(), "D");
    TORCH_CHECK(sin.dim() == 3, "sin must be 3D [B, S, head_dim], got ", sin.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kNumKVHeads = 4;
    constexpr int64_t kHeadDim = 64;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 4
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 32

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const double scaling = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    // --- First residual-norm pattern: self-attention. ---
    auto residual_1 = hidden_states;
    auto normed_1 = rms_norm(residual_1, pre_sa_norm_weight, norm_eps);

    // Q/K/V projections (no bias). matmul with explicit weight transpose.
    auto query_states = at::matmul(normed_1, q_proj_weight.t());  // [B,S,hidden]
    auto key_states = at::matmul(normed_1, k_proj_weight.t());    // [B,S,256]
    auto value_states = at::matmul(normed_1, v_proj_weight.t());  // [B,S,256]

    // Reshape to heads then transpose seq<->head.
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,16,S,64]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,4,S,64]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,4,S,64]

    // RoPE from the supplied cos / sin: unsqueeze(1) broadcasts over heads.
    auto cos_e = cos.unsqueeze(1);  // [B,1,S,64]
    auto sin_e = sin.unsqueeze(1);  // [B,1,S,64]

    auto q_half1 = query_states.narrow(/*dim=*/-1, /*start=*/0, kHalfDim);
    auto q_half2 = query_states.narrow(/*dim=*/-1, /*start=*/kHalfDim, kHalfDim);
    auto q_rotated = at::cat({-q_half2, q_half1}, -1);
    auto q_embed = (query_states * cos_e) + (q_rotated * sin_e);

    auto k_half1 = key_states.narrow(/*dim=*/-1, /*start=*/0, kHalfDim);
    auto k_half2 = key_states.narrow(/*dim=*/-1, /*start=*/kHalfDim, kHalfDim);
    auto k_rotated = at::cat({-k_half2, k_half1}, -1);
    auto k_embed = (key_states * cos_e) + (k_rotated * sin_e);

    // repeat_kv: expand the 4 kv heads into 16 query heads.
    auto key_rep = k_embed
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // Scaled QK^T, non-causal softmax (fp32), attention @ V.
    auto attn_weights = at::matmul(q_embed, key_rep.transpose(2, 3)) * scaling;  // [B,16,S,S]
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    auto attn_output = at::matmul(attn_weights, value_rep);  // [B,16,S,64]

    // Unwind heads and apply the output projection.
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
    auto attn_out = at::matmul(attn_output, o_proj_weight.t());  // [B,S,hidden]

    auto hidden_post_attn = residual_1 + attn_out;

    // --- Second residual-norm pattern: gated MLP. ---
    auto residual_2 = hidden_post_attn;
    auto normed_2 = rms_norm(residual_2, post_sa_norm_weight, norm_eps);

    auto up_states = at::matmul(normed_2, gate_up_proj_weight.t());  // [B,S,intermediate*2]
    auto chunks = up_states.chunk(2, /*dim=*/-1);
    auto gate = chunks[0];
    auto up = chunks[1];
    up = up * at::silu(gate);
    auto mlp_out = at::matmul(up, down_proj_weight.t());  // [B,S,hidden]

    auto result = residual_2 + mlp_out;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Encoder layer with dual residual-norm chain via ATen (RMSNorm, GQA "
          "self-attention with RoPE from supplied cos/sin, SwiGLU gated MLP; no "
          "custom CUDA kernels); fp32; destination-passing style (output).");
}
