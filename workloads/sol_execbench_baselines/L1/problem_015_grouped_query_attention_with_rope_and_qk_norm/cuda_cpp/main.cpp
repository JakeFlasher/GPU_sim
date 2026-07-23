#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Grouped Query Attention with RoPE and per-head Query/Key RMSNorm.
//
// Reference (problem_015_grouped_query_attention_with_rope_and_qk_norm),
// reproduced op-for-op. This is the Qwen3-8B attention block: 32 query heads,
// 8 KV heads (4x grouping), per-head RMSNorm on Q/K before RoPE, an additive
// attention mask, and a float32 softmax.
//
//   num_attention_heads  = 32
//   num_key_value_heads  = 8
//   head_dim             = 128
//   num_key_value_groups = 4
//
//   query_states = linear(hidden_states, q_proj_weight)   # [B,S,4096]
//   key_states   = linear(hidden_states, k_proj_weight)   # [B,S,1024]
//   value_states = linear(hidden_states, v_proj_weight)   # [B,S,1024]
//
//   query_states = query_states.view(B,S,32,128)
//   key_states   = key_states.view(B,S,8,128)
//   value_states = value_states.view(B,S,8,128)
//
//   rms_norm(x, weight, eps):
//       x_fp32   = x.to(float32)
//       variance = x_fp32.pow(2).mean(-1, keepdim=True)
//       x_normed = x_fp32 * rsqrt(variance + eps)
//       return weight * x_normed.to(input_dtype)        # bf16 before weight mul
//   query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
//   key_states   = rms_norm(key_states,   k_norm_weight, rms_norm_eps)
//
//   query_states = query_states.transpose(1,2)          # [B,32,S,128]
//   key_states   = key_states.transpose(1,2)            # [B, 8,S,128]
//   value_states = value_states.transpose(1,2)          # [B, 8,S,128]
//
//   cos = cos.unsqueeze(1) ; sin = sin.unsqueeze(1)     # [B,1,S,128]
//   rotate_half(x) = cat((-x[...,64:], x[...,:64]), dim=-1)
//   query_states = query_states * cos + rotate_half(query_states) * sin
//   key_states   = key_states   * cos + rotate_half(key_states)   * sin
//
//   GQA repeat (8 -> 32 heads via expand + reshape):
//     key_states   = key[:,:,None,:,:].expand(B,8,4,S,128).reshape(B,32,S,128)
//     value_states = value[:,:,None,:,:].expand(B,8,4,S,128).reshape(B,32,S,128)
//
//   attn_weights = (query_states @ key_states.transpose(2,3)) * scaling # [B,32,S,S]
//   attn_weights = attn_weights + attention_mask        # additive, broadcast over heads
//   attn_weights = softmax(attn_weights, dim=-1, dtype=float32).to(bf16)
//   attn_output  = attn_weights @ value_states          # [B,32,S,128]
//   attn_output  = attn_output.transpose(1,2).contiguous().reshape(B,S,4096)
//   attn_output  = linear(attn_output, o_proj_weight)   # [B,S,hidden]
//   return attn_output
//
// Per definition.json (insertion order):
//   hidden_states  : [batch_size, seq_len, hidden_size=4096]      bf16
//   cos            : [batch_size, seq_len, head_dim=128]          bf16
//   sin            : [batch_size, seq_len, head_dim=128]          bf16
//   attention_mask : [batch_size, 1, seq_len, seq_len]            bf16
//   q_proj_weight  : [q_proj_out_dim=4096, hidden_size=4096]      bf16
//   k_proj_weight  : [kv_proj_out_dim=1024, hidden_size=4096]     bf16
//   v_proj_weight  : [kv_proj_out_dim=1024, hidden_size=4096]     bf16
//   o_proj_weight  : [hidden_size=4096, q_proj_out_dim=4096]      bf16
//   q_norm_weight  : [head_dim=128]                               bf16
//   k_norm_weight  : [head_dim=128]                               bf16
//   rms_norm_eps   : scalar                                       fp32 (by-value double)
//   scaling        : scalar                                       fp32 (by-value double)
//   attn_output    : [batch_size, seq_len, hidden_size=4096]      bf16 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// precision profile matches the reference exactly: the only float32 excursions
// are the RMSNorm variance (cast back to bf16 before the weight multiply) and
// the softmax (taken with dtype=float32, then cast back to bf16). Everything
// else -- projections, RoPE, the QK^T and attn @ V matmuls, the output
// projection -- stays bf16, mirroring the reference and avoiding an over-precise
// fp32 chain that would diverge from the bf16 reference. The additive mask is a
// straight add that broadcasts over the head axis. Output written in place
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", t.dtype());
}

// Per-head RMSNorm matching the reference: variance accumulated in float32, the
// normalized value cast back to bf16 before the weight multiply, so the
// elementwise weight scaling happens in bf16 exactly as the reference does.
static torch::Tensor rms_norm(
    const torch::Tensor& x, const torch::Tensor& weight, double eps) {
    auto x_fp32 = x.to(torch::kFloat32);
    auto variance = x_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto x_normed = x_fp32 * at::rsqrt(variance + eps);
    return weight * x_normed.to(torch::kBFloat16);
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
    double rms_norm_eps,
    double scaling,
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

    constexpr int64_t kNumHeads = 32;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 4
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 64

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);

    // Q/K/V projections (no bias) via linear -- bf16 matmul.
    auto query_states = at::linear(hidden_states, q_proj_weight);  // [B,S,4096]
    auto key_states = at::linear(hidden_states, k_proj_weight);    // [B,S,1024]
    auto value_states = at::linear(hidden_states, v_proj_weight);  // [B,S,1024]

    // Reshape to heads.
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim});
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim});

    // Per-head RMSNorm on Q and K, applied before the seq<->head transpose
    // exactly as the reference does.
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps);
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps);

    // Transpose seq<->head.
    query_states = query_states.transpose(1, 2);  // [B,32,S,128]
    key_states = key_states.transpose(1, 2);      // [B, 8,S,128]
    value_states = value_states.transpose(1, 2);  // [B, 8,S,128]

    // cos / sin broadcast over the head axis via unsqueeze(1).
    auto cos_expanded = cos.unsqueeze(1);  // [B,1,S,128]
    auto sin_expanded = sin.unsqueeze(1);  // [B,1,S,128]

    // Apply RoPE via rotate_half: cat((-x2, x1)).
    auto q_half1 = query_states.slice(/*dim=*/-1, /*start=*/0, /*end=*/kHalfDim);
    auto q_half2 = query_states.slice(/*dim=*/-1, /*start=*/kHalfDim, /*end=*/kHeadDim);
    auto q_rotated = at::cat({-q_half2, q_half1}, /*dim=*/-1);
    query_states = (query_states * cos_expanded) + (q_rotated * sin_expanded);

    auto k_half1 = key_states.slice(/*dim=*/-1, /*start=*/0, /*end=*/kHalfDim);
    auto k_half2 = key_states.slice(/*dim=*/-1, /*start=*/kHalfDim, /*end=*/kHeadDim);
    auto k_rotated = at::cat({-k_half2, k_half1}, /*dim=*/-1);
    key_states = (key_states * cos_expanded) + (k_rotated * sin_expanded);

    // Repeat K/V for GQA: expand 8 kv heads into 32 query heads.
    key_states = key_states
                     .index({at::indexing::Slice(), at::indexing::Slice(),
                             at::indexing::None, at::indexing::Slice(),
                             at::indexing::Slice()})
                     .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                     .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    value_states = value_states
                       .index({at::indexing::Slice(), at::indexing::Slice(),
                               at::indexing::None, at::indexing::Slice(),
                               at::indexing::Slice()})
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // Attention scores in bf16, scaled, then the additive mask (broadcasts over
    // the head axis from [B,1,S,S]).
    auto attn_weights = at::matmul(query_states, key_states.transpose(2, 3)) * scaling;  // [B,32,S,S]
    attn_weights = attn_weights + attention_mask;

    // Softmax taken in float32 for stability, cast back to bf16 -- the only
    // float32 excursion in the score path.
    auto attn_probs = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                          .to(torch::kBFloat16);  // [B,32,S,S]

    // Attention output, unwind heads, output projection.
    auto out = at::matmul(attn_probs, value_states);  // [B,32,S,128]
    out = out.transpose(1, 2).contiguous().reshape({batch_size, seq_len, kNumHeads * kHeadDim});  // [B,S,4096]
    out = at::linear(out, o_proj_weight);  // [B,S,hidden]

    TORCH_CHECK(out.sizes() == attn_output.sizes(),
                "attn_output result shape ", out.sizes(),
                " does not match pre-allocated ", attn_output.sizes());
    attn_output.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Grouped Query Attention with RoPE and per-head Q/K RMSNorm "
          "(Qwen3-8B attention). Pure ATen / cuBLAS dispatch: Q/K/V "
          "projections, head reshape, per-head RMSNorm (float32 variance, bf16 "
          "weight scale), RoPE via rotate_half, GQA expand (8 kv heads -> 32 q "
          "heads), scaled QK^T scores, additive attention mask, float32 softmax "
          "cast to bf16, attn @ V, output projection. bf16 throughout except "
          "the RMSNorm variance and softmax; destination-passing style "
          "(attn_output).");
}
