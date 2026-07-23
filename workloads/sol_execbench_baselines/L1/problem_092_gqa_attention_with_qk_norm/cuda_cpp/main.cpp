#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>
#include <string>

// GLM-4 grouped-query attention with query/key RMSNorm. Includes Q/K/V
// projections (with bias), RMSNorm on Q/K over head_dim, rotary position
// embeddings, grouped-query attention (8 KV heads expanded to 96 query heads,
// 12x), scaled dot-product attention with a causal mask, and an output
// projection (no bias).
//
// This reproduces the reference run() op-for-op:
//   num_attention_heads  = 96
//   num_key_value_heads  = 8
//   head_dim             = 128
//   num_key_value_groups = 12
//   scaling              = head_dim ** -0.5
//
// GLM-4 RMSNorm uses the plain (weight * x) scaling form (no 1 + weight); the
// variance is computed in fp32 and the result is cast back to the input (bf16)
// dtype:
//   x       = x.to(float32)
//   variance = x.pow(2).mean(-1, keepdim)
//   x       = x * rsqrt(variance + rms_norm_eps)
//   output  = (weight * x).to(input_dtype)
//
//   query_states = linear(hidden_states, q_proj_weight, q_proj_bias)
//   key_states   = linear(hidden_states, k_proj_weight, k_proj_bias)
//   value_states = linear(hidden_states, v_proj_weight, v_proj_bias)
//
//   query_states = query_states.view(B,S,96,128)                # heads
//   key_states   = key_states.view(B,S,8,128)
//   value_states = value_states.view(B,S,8,128)
//
//   query_states = rms_norm(query_states, q_norm_weight)        # over head_dim
//   key_states   = rms_norm(key_states,   k_norm_weight)
//
//   query_states = query_states.transpose(1,2)                  # [B,96,S,128]
//   key_states   = key_states.transpose(1,2)                    # [B, 8,S,128]
//   value_states = value_states.transpose(1,2)                  # [B, 8,S,128]
//
//   RoPE (cos/sin provided, unsqueezed on the head axis, split at half=64):
//     cos = cos.unsqueeze(1) ; sin = sin.unsqueeze(1)           # [B,1,S,128]
//     q1,q2 = query[...,:64], query[...,64:]
//     q_rot_half = cat((-q2, q1), dim=-1)
//     query_states = query_states*cos + q_rot_half*sin
//     k1,k2 = key[...,:64], key[...,64:]
//     k_rot_half = cat((-k2, k1), dim=-1)
//     key_states = key_states*cos + k_rot_half*sin
//
//   GQA repeat (num_key_value_groups = 12):
//     key_states   = key[:,:,None,:,:].expand(B,8,12,S,128).reshape(B,96,S,128)
//     value_states = value[:,:,None,:,:].expand(B,8,12,S,128).reshape(B,96,S,128)
//
//   attn_weights = (query_states @ key_states.transpose(2,3)) * scaling
//   causal mask (triu of a full -inf matrix, diagonal=1, added):
//     causal_mask  = triu(full((S,S), -inf, dtype=attn_weights.dtype), diagonal=1)
//     attn_weights = attn_weights + causal_mask
//   softmax in fp32, cast back to the query dtype:
//     attn_weights = softmax(attn_weights, dim=-1, dtype=float32).to(q.dtype)
//   attn_output  = attn_weights @ value_states                  # [B,96,S,128]
//   attn_output  = attn_output.transpose(1,2).reshape(B,S,96*128)
//   output       = linear(attn_output, o_proj_weight, None)     # no bias
//
// Per definition.json (insertion order):
//   hidden_states  : [batch_size, seq_len, hidden_size=4096]      bf16
//   q_proj_weight  : [q_out_features=12288, hidden_size=4096]      bf16
//   q_proj_bias    : [q_out_features=12288]                        bf16
//   k_proj_weight  : [kv_out_features=1024, hidden_size=4096]      bf16
//   k_proj_bias    : [kv_out_features=1024]                        bf16
//   v_proj_weight  : [kv_out_features=1024, hidden_size=4096]      bf16
//   v_proj_bias    : [kv_out_features=1024]                        bf16
//   o_proj_weight  : [hidden_size=4096, q_out_features=12288]      bf16
//   q_norm_weight  : [head_dim=128]                                bf16
//   k_norm_weight  : [head_dim=128]                                bf16
//   cos            : [batch_size, seq_len, head_dim=128]           bf16
//   sin            : [batch_size, seq_len, head_dim=128]           bf16
//   rms_norm_eps   : scalar                                        fp32
//   output         : [batch_size, seq_len, hidden_size=4096]       bf16 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- no custom CUDA kernels. The
// scores are materialized through an explicit matmul (not a fused SDPA) so the
// score scaling, the causal-mask op order, and the softmax dtype match the
// reference precisely. RMSNorm keeps its fp32 variance and casts back to bf16
// exactly where the reference does; the causal mask is a triangular -inf matrix
// added to the scores; softmax is taken in fp32 and cast back to the query
// (bf16) dtype. Output is written in place (DPS).

namespace idx = torch::indexing;

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

// GLM-4 RMSNorm over the trailing head_dim axis with the plain (weight * x)
// scaling form. Variance is accumulated in fp32; the result is cast back to the
// input dtype, matching (weight * x).to(input_dtype) in the reference.
static torch::Tensor glm_rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps
) {
    auto input_dtype = x.scalar_type();
    auto x_float = x.to(torch::kFloat32);
    auto variance = x_float.pow(2).mean(-1, /*keepdim=*/true);
    auto x_normed = x_float * at::rsqrt(variance + eps);
    auto output = weight.to(torch::kFloat32) * x_normed;
    return output.to(input_dtype);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& q_proj_bias,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& k_proj_bias,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& v_proj_bias,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    double rms_norm_eps,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(q_proj_bias, "q_proj_bias");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(k_proj_bias, "k_proj_bias");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(v_proj_bias, "v_proj_bias");
    check_bf16_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_cuda(q_norm_weight, "q_norm_weight");
    check_bf16_cuda(k_norm_weight, "k_norm_weight");
    check_bf16_cuda(cos, "cos");
    check_bf16_cuda(sin, "sin");
    check_bf16_cuda(output, "output");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 96;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = 12;
    constexpr int64_t kHalfDim = kHeadDim / 2;  // 64
    const double scaling = std::pow(static_cast<double>(kHeadDim), -0.5);

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);

    // Q/K/V projections (with bias).
    auto query_states = at::linear(hidden_states, q_proj_weight, q_proj_bias);  // [B,S,q_out]
    auto key_states = at::linear(hidden_states, k_proj_weight, k_proj_bias);    // [B,S,kv_out]
    auto value_states = at::linear(hidden_states, v_proj_weight, v_proj_bias);  // [B,S,kv_out]

    // Reshape into heads (RMSNorm runs before the seq<->head transpose).
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim});
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});

    // Query/key RMSNorm over head_dim (GLM-4 plain (weight * x) form, fp32 var).
    query_states = glm_rms_norm(query_states, q_norm_weight, rms_norm_eps);
    key_states = glm_rms_norm(key_states, k_norm_weight, rms_norm_eps);

    // Transpose seq<->head.
    query_states = query_states.transpose(1, 2);  // [B,96,S,128]
    key_states = key_states.transpose(1, 2);      // [B,8,S,128]
    value_states = value_states.transpose(1, 2);  // [B,8,S,128]

    // RoPE: cos/sin provided; unsqueeze on the head axis and apply rotate_half
    // by splitting at half. rot_half(x) = cat((-x[..., half:], x[..., :half])).
    auto cos_e = cos.unsqueeze(1);  // [B,1,S,128]
    auto sin_e = sin.unsqueeze(1);  // [B,1,S,128]

    auto q1 = query_states.index({"...", idx::Slice(idx::None, kHalfDim)});
    auto q2 = query_states.index({"...", idx::Slice(kHalfDim, idx::None)});
    auto q_rot_half = at::cat({-q2, q1}, -1);
    query_states = (query_states * cos_e) + (q_rot_half * sin_e);

    auto k1 = key_states.index({"...", idx::Slice(idx::None, kHalfDim)});
    auto k2 = key_states.index({"...", idx::Slice(kHalfDim, idx::None)});
    auto k_rot_half = at::cat({-k2, k1}, -1);
    key_states = (key_states * cos_e) + (k_rot_half * sin_e);

    // GQA repeat: expand the 8 KV heads into 96 query heads (12x).
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

    // Attention scores with scaling (materialized; needed for the causal mask).
    auto attn_weights = at::matmul(query_states, key_states.transpose(2, 3)) * scaling;  // [B,96,S,S]

    // Causal mask: triu of a full -inf matrix (diagonal=1), added to the scores.
    auto causal_mask = at::triu(
        at::full({seq_len, seq_len},
                 -std::numeric_limits<double>::infinity(),
                 attn_weights.options()),
        /*diagonal=*/1);
    attn_weights = attn_weights + causal_mask;

    // Softmax in fp32, cast back to the query (bf16) dtype.
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                       .to(query_states.scalar_type());

    // Attention output, unwind heads, output projection (no bias).
    auto out = at::matmul(attn_weights, value_states);  // [B,96,S,128]
    out = out.transpose(1, 2).contiguous()
              .reshape({batch_size, seq_len, kNumHeads * kHeadDim});
    out = at::linear(out, o_proj_weight, /*bias=*/{});  // [B,S,hidden]

    // Destination-passing-style write.
    TORCH_CHECK(out.sizes() == output.sizes(),
                "output shape ", out.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "GLM-4 grouped-query attention with QK RMSNorm. Pure ATen / cuBLAS "
          "dispatch: Q/K/V projections (with bias), head reshape, GLM-4 RMSNorm "
          "over head_dim (plain weight * x form, fp32 variance), seq<->head "
          "transpose, RoPE (rotate_half) from the provided cos/sin, GQA expand "
          "(8 KV heads -> 96 query heads), materialized QK^T scores, triangular "
          "-inf causal mask, fp32 softmax cast back to bf16, attn @ V, head "
          "merge, output projection (no bias). bf16; destination-passing "
          "style.");
}
