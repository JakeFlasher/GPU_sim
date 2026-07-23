#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>
#include <string>

// Flash Attention for Grouped Query Attention with ultra-long context.
//
// Reference (problem_067_flash_attention_gqa_ultralong), op-for-op. Fuses QKV
// projection, RoPE application, GQA KV repetition, scaled dot-product attention
// with causal masking, and output projection. All compute is float32.
//
//   num_attention_heads  = 32
//   num_key_value_heads  = 8
//   head_dim             = 128
//   num_key_value_groups = 4
//   scaling              = head_dim ** -0.5
//
//   query_states = linear(hidden_states, q_proj_weight)        # [B,S,q_proj]
//   key_states   = linear(hidden_states, k_proj_weight)        # [B,S,kv_proj]
//   value_states = linear(hidden_states, v_proj_weight)        # [B,S,kv_proj]
//
//   query_states = query.view(B,S,32,128).transpose(1,2)       # [B,32,S,128]
//   key_states   = key.view(B,S,8,128).transpose(1,2)          # [B, 8,S,128]
//   value_states = value.view(B,S,8,128).transpose(1,2)        # [B, 8,S,128]
//
//   RoPE (cos / sin supplied as inputs, broadcast over the head axis):
//     cos_expanded = cos.unsqueeze(1) ; sin_expanded = sin.unsqueeze(1)
//     rotate_half(x) = cat((-x[...,64:], x[...,:64]), dim=-1)
//     query_states = query * cos_expanded + rotate_half(query) * sin_expanded
//     key_states   = key   * cos_expanded + rotate_half(key)   * sin_expanded
//
//   GQA repeat (kv heads broadcast into attention heads):
//     key_states   = key[:,:,None].expand(B,8,4,S,128).reshape(B,32,S,128)
//     value_states = value[:,:,None].expand(B,8,4,S,128).reshape(B,32,S,128)
//
//   attn_weights = (query @ key.transpose(2,3)) * scaling       # [B,32,S,S]
//   causal_mask  = triu(full((S,S), -inf), diagonal=1)
//   attn_weights = attn_weights + causal_mask                   # additive mask
//   attn_weights = softmax(attn_weights, dim=-1, dtype=float32)
//   attn_output  = attn_weights @ value_states                  # [B,32,S,128]
//   attn_output  = attn_output.transpose(1,2).contiguous().reshape(B,S,4096)
//   output       = linear(attn_output, o_proj_weight)           # [B,S,hidden]
//
// Per definition.json (insertion order):
//   inputs (all fp32):
//     hidden_states : [batch, seq_len, hidden_size=4096]
//     cos           : [batch, seq_len, head_dim=128]
//     sin           : [batch, seq_len, head_dim=128]
//     q_proj_weight : [q_proj_out=4096, hidden_size=4096]
//     k_proj_weight : [kv_proj_out=1024, hidden_size=4096]
//     v_proj_weight : [kv_proj_out=1024, hidden_size=4096]
//     o_proj_weight : [hidden_size=4096, q_proj_out=4096]
//   output (fp32, DPS pre-alloc, written in place):
//     output        : [batch, seq_len, hidden_size=4096]
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// reference MATERIALIZES the full [B,32,S,S] score matrix (a fused SDPA path
// that never materializes the scores would change the numerics), so the scores
// are materialized here too. The causal mask is the reference's additive
// triu(-inf) form (NOT masked_fill). All compute stays in fp32; softmax is
// taken with dtype=float32 and the subsequent .to(query dtype) is an fp32
// no-op. The single output is written in place (destination-passing style).
//
// Memory note: materializing [B,32,S,S] fp32 scores is the peak allocation; on
// a 24 GiB host the longest-context workloads (large seq_len) are OOM-watch.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(cos, "cos");
    check_fp32_cuda(sin, "sin");
    check_fp32_cuda(q_proj_weight, "q_proj_weight");
    check_fp32_cuda(k_proj_weight, "k_proj_weight");
    check_fp32_cuda(v_proj_weight, "v_proj_weight");
    check_fp32_cuda(o_proj_weight, "o_proj_weight");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(cos.dim() == 3, "cos must be 3D, got ", cos.dim(), "D");
    TORCH_CHECK(sin.dim() == 3, "sin must be 3D, got ", sin.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 32;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 4
    constexpr int64_t kHalfDim = kHeadDim / 2;                 // 64
    const double scaling = std::pow(static_cast<double>(kHeadDim), -0.5);

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);

    // Q/K/V projections (no bias). linear(x, W) = x @ W^T.
    auto query_states = at::linear(hidden_states, q_proj_weight);  // [B,S,q_proj]
    auto key_states = at::linear(hidden_states, k_proj_weight);    // [B,S,kv_proj]
    auto value_states = at::linear(hidden_states, v_proj_weight);  // [B,S,kv_proj]

    // Reshape to heads then transpose seq<->head.
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,32,S,128]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,8,S,128]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,8,S,128]

    // RoPE: cos / sin broadcast over the head axis via unsqueeze(1).
    auto cos_expanded = cos.unsqueeze(1);  // [B,1,S,128]
    auto sin_expanded = sin.unsqueeze(1);  // [B,1,S,128]

    // rotate_half(x) = cat((-x2, x1), dim=-1) where x1/x2 are the head halves.
    auto q1 = query_states.slice(/*dim=*/-1, /*start=*/0, /*end=*/kHalfDim);
    auto q2 = query_states.slice(/*dim=*/-1, /*start=*/kHalfDim, /*end=*/kHeadDim);
    auto query_rotated = at::cat({-q2, q1}, -1);
    query_states = (query_states * cos_expanded) + (query_rotated * sin_expanded);

    auto k1 = key_states.slice(/*dim=*/-1, /*start=*/0, /*end=*/kHalfDim);
    auto k2 = key_states.slice(/*dim=*/-1, /*start=*/kHalfDim, /*end=*/kHeadDim);
    auto key_rotated = at::cat({-k2, k1}, -1);
    key_states = (key_states * cos_expanded) + (key_rotated * sin_expanded);

    // GQA repeat: broadcast each kv head across its group of attention heads.
    key_states = key_states.unsqueeze(2)
                     .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                     .reshape({batch_size, kNumHeads, seq_len, kHeadDim});
    value_states = value_states.unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, seq_len, kHeadDim});

    // Attention scores (materialized to match the reference numerics).
    auto attn_weights = at::matmul(query_states, key_states.transpose(2, 3)) * scaling;  // [B,32,S,S]

    // Causal mask: triu(full((S,S), -inf), diagonal=1) added to the scores.
    auto causal_mask = at::triu(
        at::full({seq_len, seq_len},
                 -std::numeric_limits<double>::infinity(),
                 hidden_states.options()),
        /*diagonal=*/1);
    attn_weights = attn_weights + causal_mask;

    // Softmax in fp32; the subsequent cast back to the query dtype is an fp32
    // no-op here (all inputs are fp32).
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32);

    // Attention output, unwind heads, output projection.
    auto attn_output = at::matmul(attn_weights, value_states);  // [B,32,S,128]
    attn_output = attn_output.transpose(1, 2).contiguous()
                      .reshape({batch_size, seq_len, kNumHeads * kHeadDim});  // [B,S,4096]
    auto out = at::linear(attn_output, o_proj_weight);  // [B,S,hidden]

    TORCH_CHECK(out.sizes() == output.sizes(),
                "output result shape ", out.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Flash Attention for GQA with ultra-long context. Pure ATen / cuBLAS "
        "dispatch: Q/K/V projections (no bias), head reshape, RoPE (rotate_half "
        "with supplied cos/sin broadcast over heads), GQA repeat (kv heads -> "
        "attention heads), materialized QK^T scores scaled by head_dim**-0.5, "
        "additive triu(-inf) causal mask, fp32 softmax, attn @ V, output "
        "projection. fp32 throughout; destination-passing style (output).");
}
