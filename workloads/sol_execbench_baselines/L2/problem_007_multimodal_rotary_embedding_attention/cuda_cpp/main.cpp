#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>
#include <cmath>

// Grouped-query attention with multi-modal 3D rotary position embeddings.
// 28 query heads share 4 key/value heads (7:1 GQA). The rotary embedding is
// gathered from three section-specific cos/sin planes (temporal / height /
// width) before a standard rotate-half rotation, the scores carry an explicit
// additive mask, and the softmax accumulates in fp32.
//
// Reference (problem_007_multimodal_rotary_embedding_attention), op-for-op:
//   num_heads      = 28
//   num_kv_heads   = 4
//   num_kv_groups  = 7
//   head_dim       = 128
//   scaling        = head_dim ** -0.5
//   mrope_section          = [16, 24, 24]   # temporal, height, width
//   mrope_section_doubled  = [32, 48, 48]   # cos/sin pair widths
//
//   q = linear(hidden_states, q_weight, q_bias)     # [B,S,28*128]
//   k = linear(hidden_states, k_weight, k_bias)     # [B,S, 4*128]
//   v = linear(hidden_states, v_weight, v_bias)     # [B,S, 4*128]
//   q = q.view(B,S,28,128).transpose(1,2)           # [B,28,S,128]
//   k = k.view(B,S, 4,128).transpose(1,2)           # [B, 4,S,128]
//   v = v.view(B,S, 4,128).transpose(1,2)           # [B, 4,S,128]
//
//   # cos / sin arrive as [3, B, S, 128]. The combined plane gathers section i
//   # (width doubled[i], starting at sum(doubled[:i])) from rope plane i % 3.
//   # With three sections and i % 3 == i this is: plane 0 contributes channels
//   # [0:32], plane 1 contributes [32:80], plane 2 contributes [80:128].
//   cos_combined = cat([ cos[i][..., start_i:start_i+doubled[i]] ], -1).unsqueeze(1)
//   sin_combined = cat([ sin[i][..., start_i:start_i+doubled[i]] ], -1).unsqueeze(1)
//
//   rotate_half(x) = cat((-x[..., 64:], x[..., :64]), dim=-1)
//   q = q * cos_combined + rotate_half(q) * sin_combined
//   k = k * cos_combined + rotate_half(k) * sin_combined
//
//   k = repeat_kv(k, 7) ; v = repeat_kv(v, 7)       # [B,28,S,128]
//   attn = (q @ k.transpose(2,3)) * scaling          # [B,28,S,S]
//   attn = attn + attention_mask[..., :S]            # additive mask, [B,1,S,S]
//   attn = softmax(attn, dim=-1, dtype=float32).to(bf16)
//   attn_output = attn @ v                           # [B,28,S,128]
//   attn_output = attn_output.transpose(1,2).contiguous().view(B,S,-1)
//   output = linear(attn_output, o_weight)           # [B,S,hidden]
//
// Per definition.json (insertion order):
//   hidden_states   : [B, S, hidden]          bf16
//   q_weight        : [28*128, hidden]         bf16
//   q_bias          : [28*128]                 bf16
//   k_weight        : [4*128, hidden]          bf16
//   k_bias          : [4*128]                  bf16
//   v_weight        : [4*128, hidden]          bf16
//   v_bias          : [4*128]                  bf16
//   o_weight        : [hidden, 28*128]         bf16
//   cos             : [3, B, S, head_dim]      bf16
//   sin             : [3, B, S, head_dim]      bf16
//   attention_mask  : [B, 1, S, S]             bf16
//   output          : [B, S, hidden]           bf16 DPS
//
// Notes on fidelity:
//   * The mrope combine is a section gather, not an interleave: each of the
//     three head-dim sections is read from its own rope plane and stitched
//     back together along the head dim, then broadcast across heads via
//     unsqueeze(1). The rotation itself is the rotate-half form
//     cat((-x2, x1)) applied with the gathered cos / sin.
//   * The mask is an explicit additive bf16 tensor of shape [B, 1, S, S],
//     sliced along its last dim to the key length before being added to the
//     scaled scores. There is no internally-built causal mask here.
//   * repeat_kv expands the 4 kv heads into 28 query heads via an inserted
//     group axis, expand, and reshape -- the same path the reference takes.
//   * Precision: inputs and outputs are bf16. Only the softmax accumulates in
//     fp32, then casts back to bf16 before the value matmul. The bf16 output
//     is never upcast; bf16 is the contractual dtype, not a downgrade.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_weight,
    const torch::Tensor& q_bias,
    const torch::Tensor& k_weight,
    const torch::Tensor& k_bias,
    const torch::Tensor& v_weight,
    const torch::Tensor& v_bias,
    const torch::Tensor& o_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& attention_mask,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(q_weight, "q_weight");
    check_bf16_cuda(q_bias, "q_bias");
    check_bf16_cuda(k_weight, "k_weight");
    check_bf16_cuda(k_bias, "k_bias");
    check_bf16_cuda(v_weight, "v_weight");
    check_bf16_cuda(v_bias, "v_bias");
    check_bf16_cuda(o_weight, "o_weight");
    check_bf16_cuda(cos, "cos");
    check_bf16_cuda(sin, "sin");
    check_bf16_cuda(attention_mask, "attention_mask");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(cos.dim() == 4, "cos must be 4D [3, B, S, head_dim], got ", cos.dim(), "D");
    TORCH_CHECK(sin.dim() == 4, "sin must be 4D [3, B, S, head_dim], got ", sin.dim(), "D");
    TORCH_CHECK(attention_mask.dim() == 4,
                "attention_mask must be 4D [B, 1, S, S], got ", attention_mask.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

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

    // Q/K/V projections with bias via at::linear (x @ weight.t() + bias).
    auto query_states = at::linear(hidden_states, q_weight, q_bias);  // [B,S,28*128]
    auto key_states = at::linear(hidden_states, k_weight, k_bias);    // [B,S, 4*128]
    auto value_states = at::linear(hidden_states, v_weight, v_bias);  // [B,S, 4*128]

    // Reshape to heads then transpose seq<->head.
    query_states = query_states.view({batch_size, seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,28,S,128]
    key_states = key_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,4,S,128]
    value_states = value_states.view({batch_size, seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,4,S,128]

    // Multi-modal 3D RoPE: gather each head-dim section from its own rope
    // plane, stitch them back along the head dim, then broadcast over heads.
    std::vector<torch::Tensor> cos_parts;
    std::vector<torch::Tensor> sin_parts;
    cos_parts.reserve(mrope_doubled.size());
    sin_parts.reserve(mrope_doubled.size());
    int64_t start = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(mrope_doubled.size()); ++i) {
        const int64_t width = mrope_doubled[i];
        const int64_t plane = i % 3;
        cos_parts.push_back(cos[plane].narrow(/*dim=*/-1, /*start=*/start, /*length=*/width));
        sin_parts.push_back(sin[plane].narrow(/*dim=*/-1, /*start=*/start, /*length=*/width));
        start += width;
    }
    auto cos_combined = at::cat(cos_parts, -1).unsqueeze(1);  // [B,1,S,128]
    auto sin_combined = at::cat(sin_parts, -1).unsqueeze(1);  // [B,1,S,128]

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

    // Scaled QK^T, additive mask (sliced to key length), fp32 softmax cast
    // back to bf16, attention @ V.
    auto attn_weights = at::matmul(q_embed, key_rep.transpose(2, 3)) * scaling;  // [B,28,S,S]
    auto causal_mask = attention_mask.narrow(/*dim=*/-1, /*start=*/0, seq_len);
    attn_weights = attn_weights + causal_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                       .to(query_states.scalar_type());
    auto attn_output = at::matmul(attn_weights, value_rep);  // [B,28,S,128]

    // Unwind heads and apply the output projection (no bias).
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
    auto result = at::linear(attn_output, o_weight);  // [B,S,hidden]

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "GQA attention with multi-modal 3D RoPE via ATen (sectioned cos/sin "
          "gather, rotate-half, additive mask, fp32 softmax; no custom CUDA "
          "kernels); bf16 in/out with fp32 softmax accumulation; "
          "destination-passing style (output).");
}
