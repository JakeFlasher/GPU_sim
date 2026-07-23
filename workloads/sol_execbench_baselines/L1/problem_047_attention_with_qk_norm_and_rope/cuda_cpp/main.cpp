#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <string>

// Gemma3-style attention with query/key RMSNorm, rotary position embeddings,
// grouped-query attention, attention logit softcapping, and an additive
// (causal) mask. Includes Q/K/V/O projections.
//
// This reproduces the reference run() op-for-op:
//   num_attention_heads  = 24
//   num_key_value_heads  = 8
//   head_dim             = 128
//   num_key_value_groups = num_attention_heads / num_key_value_heads = 3
//   scaling              = head_dim ** -0.5
//
// Gemma3 RMSNorm uses the (1 + weight) scaling form; the variance is computed
// in fp32 and the result is cast back to the input (bf16) dtype:
//   x_float  = x.float()
//   variance = x_float.pow(2).mean(-1, keepdim)
//   x_normed = x_float * rsqrt(variance + eps)
//   output   = (x_normed * (1.0 + weight.float())).type_as(x)
//
//   query_states = linear(hidden_states, q_proj_weight)         # no bias
//   key_states   = linear(hidden_states, k_proj_weight)
//   value_states = linear(hidden_states, v_proj_weight)
//   query_states = query_states.view(B,S,24,128).transpose(1,2) # [B,24,S,128]
//   key_states   = key_states.view(B,S,8,128).transpose(1,2)    # [B, 8,S,128]
//   value_states = value_states.view(B,S,8,128).transpose(1,2)  # [B, 8,S,128]
//
//   query_states = rms_norm(query_states, q_norm_weight, eps)   # over head_dim
//   key_states   = rms_norm(key_states,   k_norm_weight, eps)
//
//   RoPE (cos/sin provided, unsqueezed on the head axis):
//     cos = cos.unsqueeze(1) ; sin = sin.unsqueeze(1)           # [B,1,S,128]
//     rotate_half(x) = cat((-x[...,64:], x[...,:64]), dim=-1)
//     query_states = query_states*cos + rotate_half(query_states)*sin
//     key_states   = key_states*cos   + rotate_half(key_states)*sin
//
//   GQA repeat (num_key_value_groups = 3):
//     key_states   = key[:,:,None,:,:].expand(B,8,3,S,128).reshape(B,24,S,128)
//     value_states = value[:,:,None,:,:].expand(B,8,3,S,128).reshape(B,24,S,128)
//
//   attn_weights = (query_states @ key_states.transpose(2,3)) * scaling
//   softcap:
//     attn_weights = attn_weights / attn_logit_softcapping
//     attn_weights = tanh(attn_weights)
//     attn_weights = attn_weights * attn_logit_softcapping
//   additive mask, sliced to the key length:
//     causal_mask  = attention_mask[:,:,:,:key_len]
//     attn_weights = attn_weights + causal_mask
//   softmax in fp32, cast back to the query dtype:
//     attn_weights = softmax(attn_weights, dim=-1, dtype=float32).to(q.dtype)
//   attn_output  = attn_weights @ value_states                  # [B,24,S,128]
//   attn_output  = attn_output.transpose(1,2).reshape(B,S,24*128)
//   attn_output  = linear(attn_output, o_proj_weight)           # no bias
//
// Per definition.json (insertion order):
//   hidden_states          : [batch_size, seq_len, hidden_size=3072]  bf16
//   cos                    : [batch_size, seq_len, head_dim=128]      bf16
//   sin                    : [batch_size, seq_len, head_dim=128]      bf16
//   attention_mask         : [batch_size, 1, seq_len, seq_len]        bf16
//   q_proj_weight          : [qkv_out_dim=3072, hidden_size=3072]     bf16
//   k_proj_weight          : [kv_out_dim=1024, hidden_size=3072]      bf16
//   v_proj_weight          : [kv_out_dim=1024, hidden_size=3072]      bf16
//   o_proj_weight          : [hidden_size=3072, qkv_out_dim=3072]     bf16
//   q_norm_weight          : [head_dim=128]                           bf16
//   k_norm_weight          : [head_dim=128]                           bf16
//   attn_logit_softcapping : scalar                                   fp32
//   rms_norm_eps           : scalar                                   fp32
//   attn_output            : [batch_size, seq_len, hidden_size=3072]  bf16 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- no custom CUDA kernels. The
// scores are materialized through an explicit matmul (not a fused SDPA) because
// the logit softcapping and the additive mask cannot be expressed through the
// fused attention path; the score-then-softcap-then-mask op order must match
// the reference precisely. RMSNorm keeps its fp32 variance and casts back to
// bf16 exactly where the reference does; softmax is taken in fp32 and cast
// back to the query (bf16) dtype. Output is written in place (DPS).

namespace idx = torch::indexing;

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

// Gemma3 RMSNorm over the trailing head_dim axis with the (1 + weight) scaling
// form. Variance is accumulated in fp32; the result is cast back to the input
// dtype, matching rms_norm(...).type_as(x) in the reference.
static torch::Tensor gemma_rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps
) {
    auto input_dtype = x.scalar_type();
    auto x_float = x.to(torch::kFloat32);
    auto variance = x_float.pow(2).mean(-1, /*keepdim=*/true);
    auto x_normed = x_float * at::rsqrt(variance + eps);
    auto output = x_normed * (1.0 + weight.to(torch::kFloat32));
    return output.to(input_dtype);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& attention_mask,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    double attn_logit_softcapping,
    double rms_norm_eps,
    torch::Tensor attn_output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(cos, "cos");
    check_bf16_cuda(sin, "sin");
    check_bf16_cuda(attention_mask, "attention_mask");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_cuda(q_norm_weight, "q_norm_weight");
    check_bf16_cuda(k_norm_weight, "k_norm_weight");
    check_bf16_cuda(attn_output, "attn_output");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 24;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 3
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 64
    const double scaling = std::pow(static_cast<double>(kHeadDim), -0.5);

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);

    // Q/K/V projections (no bias).
    auto query_states = at::linear(hidden_states, q_proj_weight);  // [B,S,qkv_out]
    auto key_states = at::linear(hidden_states, k_proj_weight);    // [B,S,kv_out]
    auto value_states = at::linear(hidden_states, v_proj_weight);  // [B,S,kv_out]

    // Reshape into heads then transpose seq<->head.
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,24,S,128]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,8,S,128]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,8,S,128]

    // Query/key RMSNorm over head_dim (Gemma3 (1 + weight) form, fp32 variance).
    query_states = gemma_rms_norm(query_states, q_norm_weight, rms_norm_eps);
    key_states = gemma_rms_norm(key_states, k_norm_weight, rms_norm_eps);

    // RoPE: cos/sin are provided; unsqueeze on the head axis and apply
    // rotate_half. rotate_half(x) = cat((-x[..., half:], x[..., :half])).
    auto cos_e = cos.unsqueeze(1);  // [B,1,S,128]
    auto sin_e = sin.unsqueeze(1);  // [B,1,S,128]

    auto q_half1 = query_states.index({"...", idx::Slice(idx::None, kHalfDim)});
    auto q_half2 = query_states.index({"...", idx::Slice(kHalfDim, idx::None)});
    auto q_rotated = at::cat({-q_half2, q_half1}, -1);
    query_states = (query_states * cos_e) + (q_rotated * sin_e);

    auto k_half1 = key_states.index({"...", idx::Slice(idx::None, kHalfDim)});
    auto k_half2 = key_states.index({"...", idx::Slice(kHalfDim, idx::None)});
    auto k_rotated = at::cat({-k_half2, k_half1}, -1);
    key_states = (key_states * cos_e) + (k_rotated * sin_e);

    // GQA repeat: expand the 8 KV heads into 24 query heads (3x).
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

    // Attention scores with scaling (materialized; needed for softcap + mask).
    auto attn_weights = at::matmul(query_states, key_states.transpose(2, 3)) * scaling;  // [B,24,S,S]

    // Attention logit softcapping: x/cap -> tanh -> *cap.
    attn_weights = attn_weights / attn_logit_softcapping;
    attn_weights = at::tanh(attn_weights);
    attn_weights = attn_weights * attn_logit_softcapping;

    // Additive mask, sliced to the key length.
    const int64_t key_len = key_states.size(-2);
    auto causal_mask = attention_mask.index(
        {idx::Slice(), idx::Slice(), idx::Slice(), idx::Slice(idx::None, key_len)});
    attn_weights = attn_weights + causal_mask;

    // Softmax in fp32, cast back to the query (bf16) dtype.
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                       .to(query_states.scalar_type());

    // Attention output, unwind heads, output projection (no bias).
    auto out = at::matmul(attn_weights, value_states);  // [B,24,S,128]
    out = out.transpose(1, 2).contiguous()
              .reshape({batch_size, seq_len, kNumHeads * kHeadDim});
    out = at::linear(out, o_proj_weight);  // [B,S,hidden]

    // Destination-passing-style write.
    TORCH_CHECK(out.sizes() == attn_output.sizes(),
                "attn_output shape ", out.sizes(),
                " does not match pre-allocated ", attn_output.sizes());
    attn_output.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Gemma3-style attention with QK RMSNorm, RoPE, grouped-query "
          "attention, logit softcapping, and an additive mask. Pure ATen / "
          "cuBLAS dispatch: Q/K/V projections, head reshape, Gemma3 RMSNorm "
          "over head_dim (1 + weight form, fp32 variance), RoPE (rotate_half) "
          "from the provided cos/sin, GQA expand (8 KV heads -> 24 query "
          "heads), materialized QK^T scores, softcap (x/cap -> tanh -> *cap), "
          "additive mask, fp32 softmax cast back to bf16, attn @ V, head "
          "merge, output projection. bf16; destination-passing style.");
}
