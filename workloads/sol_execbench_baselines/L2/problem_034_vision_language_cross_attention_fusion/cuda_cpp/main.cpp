#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <vector>
#include <string>
#include <cmath>
#include <utility>

// Qwen3-VL cross-attention fusion: language queries attend to vision
// key/values under grouped-query attention (32 query heads, 8 key/value heads,
// 4:1 grouping). Queries carry a standard 1D rotary embedding driven by the
// language position ids; keys carry a 3D rotary embedding whose temporal /
// height / width positions are read out of the vision_grid_thw index tensor.
// Reproduced op-for-op from the reference run().
//
// Fixed constants from the spec:
//   hidden_size    = 4096
//   num_heads      = 32
//   num_kv_heads   = 8
//   num_kv_groups  = 4
//   head_dim       = 128
//   scaling        = 1 / sqrt(head_dim)
//   rope_theta     = 10000.0
//   3D split       = (dim_t, dim_h, dim_w) = (42, 42, 44)  (sums to head_dim)
//
// Rotary inverse frequencies, exactly as the reference derives them:
//   inv_freq_1d = 1 / theta ^ (arange(0, head_dim, 2) / head_dim)   # 64 freqs
//   inv_freq_t  = 1 / theta ^ (arange(0, dim_t, 2)    / dim_t)      # 21 freqs
//   inv_freq_h  = 1 / theta ^ (arange(0, dim_h, 2)    / dim_h)      # 21 freqs
//   inv_freq_w  = 1 / theta ^ (arange(0, dim_w, 2)    / dim_w)      # 22 freqs
//
// rotate_half(x) = cat((-x2, x1)) where x1/x2 are the two halves of the part
// being rotated. The 1D embedding rotates the whole 128-wide head; each 3D
// section rotates only its own slice (dim_t / dim_h / dim_w wide) and the three
// rotated slices are stitched back along the head dim.
//
// 1D RoPE on Q (positions from language_position_ids, [B, lang_seq]):
//   freqs = position_ids[:,None,:,None].float() * inv_freq_1d        # [B,1,Sq,64]
//   emb   = cat([freqs, freqs], -1)                                  # [B,1,Sq,128]
//   cos/sin = emb.cos()/emb.sin() cast to the query dtype
//   q = q * cos + rotate_half(q) * sin
//
// 3D RoPE on K (positions gathered per section from vision_grid_thw[...,axis]):
//   for each section (t / h / w):
//     k_part      = k[..., section_slice]                            # [B,8,Sk,dim]
//     pos         = vision_grid_thw[..., axis].float()               # [B,Sk]
//     freqs       = pos[:,None,:,None] * inv_freq_axis               # [B,1,Sk,dim/2]
//     emb         = cat([freqs, freqs], -1)                          # [B,1,Sk,dim]
//     cos/sin     = emb.cos()/emb.sin() cast to the key dtype
//     k_part_rot  = k_part * cos + rotate_half(k_part) * sin
//   k = cat([k_t_rot, k_h_rot, k_w_rot], -1)
//
// GQA: the 8 kv heads are expanded to 32 query heads via an inserted group
// axis, expand, and reshape -- the same path the reference's repeat_kv takes.
// Then scaled QK^T, fp32 softmax cast back to bf16, attn @ V, head unwind, and
// the (bias-free) output projection.
//
// Per definition.json (insertion order):
//   language_hidden_states : [B, lang_seq, hidden]     bf16
//   vision_hidden_states   : [B, vision_seq, hidden]   bf16
//   language_position_ids  : [B, lang_seq]             int64
//   vision_grid_thw        : [B, vision_seq, 3]        int64
//   q_proj_weight          : [32*128, hidden]          bf16
//   q_proj_bias            : [32*128]                  bf16
//   k_proj_weight          : [8*128, hidden]           bf16
//   k_proj_bias            : [8*128]                   bf16
//   v_proj_weight          : [8*128, hidden]           bf16
//   v_proj_bias            : [8*128]                   bf16
//   o_proj_weight          : [hidden, 32*128]          bf16
//   output                 : [B, lang_seq, hidden]     bf16 DPS
//
// Notes on fidelity:
//   * vision_grid_thw and language_position_ids are int64 data tensors. The
//     per-token positions are read on-device (select on the position axis, then
//     float()); there is no host-side loop over tokens, so the pointer-shifting
//     harness sees a fresh gather every call.
//   * The 3D split is the reference's literal 42/42/44, not an even thirds
//     split; the width imbalance is contractual.
//   * Precision: inputs and outputs are bf16, the rope cos/sin are cast to the
//     query/key dtype before the multiply (matching the reference's .to()), and
//     only the softmax accumulates in fp32 before casting back to bf16. The
//     bf16 output is never upcast; bf16 is the contractual dtype.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

static void check_int64_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt64, name, " must be int64, got ", t.dtype());
}

// Build the [B, 1, S, dim] cos/sin pair from float positions [B, S] and the
// per-axis inverse frequencies, then return cos and sin cast to out_dtype.
// Mirrors the reference helper: emb = cat([freqs, freqs]); cos/sin = emb.cos()/sin().
static std::pair<torch::Tensor, torch::Tensor> rope_cos_sin(
    const torch::Tensor& positions_f32,  // [B, S], float32
    const torch::Tensor& inv_freq,       // [dim/2], float32
    torch::ScalarType out_dtype) {
    // [B, 1, S, 1] * [dim/2] -> [B, 1, S, dim/2].
    auto pos_expanded = positions_f32.unsqueeze(1).unsqueeze(-1);
    auto freqs = pos_expanded * inv_freq;
    auto emb = at::cat({freqs, freqs}, -1);  // [B, 1, S, dim]
    auto cos = emb.cos().to(out_dtype);
    auto sin = emb.sin().to(out_dtype);
    return {cos, sin};
}

// rotate_half(x) = cat((-x2, x1)) over the last dim.
static torch::Tensor rotate_half(const torch::Tensor& x) {
    const int64_t last = x.size(-1);
    const int64_t half = last / 2;
    auto x1 = x.narrow(-1, 0, half);
    auto x2 = x.narrow(-1, half, last - half);
    return at::cat({-x2, x1}, -1);
}

void run(
    const torch::Tensor& language_hidden_states,
    const torch::Tensor& vision_hidden_states,
    const torch::Tensor& language_position_ids,
    const torch::Tensor& vision_grid_thw,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& q_proj_bias,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& k_proj_bias,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& v_proj_bias,
    const torch::Tensor& o_proj_weight,
    torch::Tensor output
) {
    check_bf16_cuda(language_hidden_states, "language_hidden_states");
    check_bf16_cuda(vision_hidden_states, "vision_hidden_states");
    check_int64_cuda(language_position_ids, "language_position_ids");
    check_int64_cuda(vision_grid_thw, "vision_grid_thw");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(q_proj_bias, "q_proj_bias");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(k_proj_bias, "k_proj_bias");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(v_proj_bias, "v_proj_bias");
    check_bf16_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(language_hidden_states.dim() == 3,
                "language_hidden_states must be 3D [B, lang_seq, hidden], got ",
                language_hidden_states.dim(), "D");
    TORCH_CHECK(vision_hidden_states.dim() == 3,
                "vision_hidden_states must be 3D [B, vision_seq, hidden], got ",
                vision_hidden_states.dim(), "D");
    TORCH_CHECK(vision_grid_thw.dim() == 3 && vision_grid_thw.size(-1) == 3,
                "vision_grid_thw must be 3D [B, vision_seq, 3], got ",
                vision_grid_thw.sizes());

    const c10::cuda::CUDAGuard device_guard(language_hidden_states.device());

    constexpr int64_t kNumHeads = 32;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKVGroups = kNumHeads / kNumKVHeads;  // 4
    constexpr double kRopeTheta = 10000.0;
    constexpr int64_t kDimT = 42;
    constexpr int64_t kDimH = 42;
    constexpr int64_t kDimW = 44;  // 42 + 42 + 44 == 128

    const int64_t batch_size = language_hidden_states.size(0);
    const int64_t lang_seq_len = language_hidden_states.size(1);
    const int64_t vision_seq_len = vision_hidden_states.size(1);
    const double scaling = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    const auto f32_opts = language_hidden_states.options().dtype(torch::kFloat32);
    const auto query_dtype = language_hidden_states.scalar_type();

    // inv_freq for a rope segment: 1 / theta ^ (arange(0, dim, 2) / dim).
    auto make_inv_freq = [&](int64_t dim) {
        auto idx = at::arange(0, dim, 2, f32_opts);  // [dim/2]
        return at::pow(kRopeTheta, idx / static_cast<double>(dim)).reciprocal();
    };
    auto inv_freq_1d = make_inv_freq(kHeadDim);
    auto inv_freq_t = make_inv_freq(kDimT);
    auto inv_freq_h = make_inv_freq(kDimH);
    auto inv_freq_w = make_inv_freq(kDimW);

    // Q/K/V projections with bias (x @ weight.t() + bias).
    auto query_states = at::linear(language_hidden_states, q_proj_weight, q_proj_bias);
    auto key_states = at::linear(vision_hidden_states, k_proj_weight, k_proj_bias);
    auto value_states = at::linear(vision_hidden_states, v_proj_weight, v_proj_bias);

    query_states = query_states.view({batch_size, lang_seq_len, kNumHeads, kHeadDim})
                       .transpose(1, 2);  // [B,32,Sq,128]
    key_states = key_states.view({batch_size, vision_seq_len, kNumKVHeads, kHeadDim})
                     .transpose(1, 2);    // [B,8,Sk,128]
    value_states = value_states.view({batch_size, vision_seq_len, kNumKVHeads, kHeadDim})
                       .transpose(1, 2);  // [B,8,Sk,128]

    // --- 1D RoPE on the queries, driven by the language position ids. ---
    auto q_positions = language_position_ids.to(torch::kFloat32);  // [B, Sq]
    auto q_cos_sin = rope_cos_sin(q_positions, inv_freq_1d, query_dtype);
    auto q_embed = (query_states * q_cos_sin.first) +
                   (rotate_half(query_states) * q_cos_sin.second);

    // --- 3D RoPE on the keys, gathering t/h/w positions from vision_grid_thw. ---
    // The three head-dim sections are rotated independently then re-stitched.
    const int64_t starts[3] = {0, kDimT, kDimT + kDimH};
    const int64_t widths[3] = {kDimT, kDimH, kDimW};
    const torch::Tensor* inv_freqs[3] = {&inv_freq_t, &inv_freq_h, &inv_freq_w};

    std::vector<torch::Tensor> k_rotated_parts;
    k_rotated_parts.reserve(3);
    for (int64_t axis = 0; axis < 3; ++axis) {
        auto k_part = key_states.narrow(-1, starts[axis], widths[axis]);  // [B,8,Sk,dim]
        // Position for this axis: vision_grid_thw[..., axis] -> [B, Sk], float.
        auto pos = vision_grid_thw.select(-1, axis).to(torch::kFloat32);  // [B, Sk]
        auto cos_sin = rope_cos_sin(pos, *inv_freqs[axis], query_dtype);
        auto k_part_rot = (k_part * cos_sin.first) + (rotate_half(k_part) * cos_sin.second);
        k_rotated_parts.push_back(k_part_rot);
    }
    auto k_embed = at::cat(k_rotated_parts, -1);  // [B,8,Sk,128]

    // GQA: expand the 8 kv heads into 32 query heads.
    auto key_rep = k_embed
                       .unsqueeze(2)
                       .expand({batch_size, kNumKVHeads, kNumKVGroups, vision_seq_len, kHeadDim})
                       .reshape({batch_size, kNumHeads, vision_seq_len, kHeadDim});
    auto value_rep = value_states
                         .unsqueeze(2)
                         .expand({batch_size, kNumKVHeads, kNumKVGroups, vision_seq_len, kHeadDim})
                         .reshape({batch_size, kNumHeads, vision_seq_len, kHeadDim});

    // Scaled QK^T, fp32 softmax cast back to bf16, attn @ V.
    auto attn_weights = at::matmul(q_embed, key_rep.transpose(2, 3)) * scaling;  // [B,32,Sq,Sk]
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                       .to(query_dtype);
    auto attn_output = at::matmul(attn_weights, value_rep);  // [B,32,Sq,128]

    // Unwind heads and apply the (bias-free) output projection.
    attn_output = attn_output.transpose(1, 2).contiguous()
                      .view({batch_size, lang_seq_len, kNumHeads * kHeadDim});
    auto result = at::linear(attn_output, o_proj_weight);  // [B, Sq, hidden]

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Qwen3-VL vision-language cross-attention fusion via pure ATen "
          "dispatch: GQA (32 query / 8 kv heads), 1D RoPE on queries from "
          "language position ids, 3D RoPE on keys with temporal/height/width "
          "positions gathered on-device from vision_grid_thw (42/42/44 split), "
          "scaled QK^T, fp32 softmax cast back to bf16, attn @ V, and a "
          "bias-free output projection; no custom CUDA kernels; bf16 in/out "
          "with fp32 softmax accumulation; destination-passing style (output).");
}
