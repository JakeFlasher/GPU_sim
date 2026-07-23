#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <vector>
#include <string>
#include <cmath>

// FIBO dual-stream attention block with conditional cross-attention. A single
// boolean (is_joint_block) selects between joint attention -- where text
// context key/values are concatenated onto the image stream -- and plain
// self-attention. AdaLN modulation driven by a SiLU-gated timestep embedding,
// 3D rotary embeddings gathered from precomputed cos/sin tables, scaled
// dot-product attention, an output projection, and a gated residual add,
// reproduced op-for-op from the reference run().
//
// Fixed constants from the spec:
//   num_heads          = 24
//   head_dim           = 128
//   hidden_size        = 3072
//   joint_attention_dim= 4096      (context / encoder stream width)
//   modulation_dim     = 6 * 3072  (adaLN linear output, chunked into 6)
//   rope_dim_per_axis  = 128 // 3 = 42  (three rotated axes; 2 dims left un-rotated)
//   scale              = 1 / sqrt(head_dim)
//
// AdaLN modulation: the timestep embedding is gated by its own sigmoid
// (SiLU: temb * sigmoid(temb)), passed through a single linear, then chunk(6)
// splits the result into scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp,
// gate_mlp. The image stream is layer-normed (affine-free, default eps) and
// modulated as normed * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1).
// Only scale_msa / shift_msa / gate_msa participate here; the _mlp chunks are
// computed (matching the reference's chunk) but unused on this attention-only
// path, and gate_msa scales the attention residual.
//
// 3D RoPE: q and k are viewed as [B, S, heads, head_dim]. The first three
// 42-wide slices are rotated independently against axis-0/1/2 position indices;
// the trailing 2 dims are left unrotated. Each axis gathers cos/sin rows by its
// pos_idx (an int64 index tensor, gathered on-device) from the precomputed
// [max_pos, 42] tables. The rotation uses the interleaved-half form:
//   half     = rope_dim_per_axis // 2 = 21
//   x_first  = x[..., :21] ; x_second = x[..., 21:]
//   cos_first/sin_first = cos/sin gathered[..., :21]
//   rotated_first  = x_first * cos_first - x_second * sin_first
//   rotated_second = x_first * sin_first + x_second * cos_first
//   out = cat([rotated_first, rotated_second], -1)
// (Note: only the first 21 columns of the 42-wide cos/sin rows are consulted,
// matching the reference exactly.)
//
// Conditional cross-attention (HOST branch on is_joint_block):
//   if is_joint_block != 0:
//     the text encoder states are layer-normed over joint_attention_dim, then
//     projected to context K/V (with bias) and reshaped to heads; these are
//     concatenated onto the image-stream K/V along the sequence axis, so the
//     key/value length becomes seq_len + text_seq_len.
//   else:
//     self-attention only -- K/V are the image-stream projections alone.
// Both arms appear in the workload set (is_joint_block in {0, 1}), so the host
// branch handles both. is_joint_block arrives as a plain scalar (definition
// shape is null / int32); pybind binds it to int64_t.
//
// Per definition.json (insertion order):
//   hidden_states          : [B, seq, hidden]               fp32
//   timestep_embedding     : [B, hidden]                    fp32
//   encoder_hidden_states  : [B, text_seq, joint_attn_dim]  fp32
//   adaln_linear_weight    : [6*hidden, hidden]             fp32
//   adaln_linear_bias      : [6*hidden]                     fp32
//   to_q/k/v_weight/bias   : image-stream QKV projections   fp32
//   to_k/v_context_weight/bias : context KV projections     fp32
//   to_out_weight/bias     : attention output projection    fp32
//   pos_idx_axis0/1/2      : [seq]                          int64
//   rope_cos/sin_axis0/1/2 : [max_pos, 42]                  fp32
//   is_joint_block         : scalar                         int32 -> int64_t
//   output                 : [B, seq, hidden]               fp32 DPS
//
// Notes on fidelity:
//   * The output keeps only the image-stream tokens (seq_len). When joint, the
//     concatenated text K/V lengthen the key axis only; attn @ V then collapses
//     the key axis, so attn_output stays [B, heads, seq, head_dim] and the text
//     tokens never appear in the output -- matching the reference's reshape to
//     [B, seq, hidden].
//   * Precision is native fp32 throughout, including the softmax; there is no
//     widened accumulator and no downgrade.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

static void check_int64_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt64, name, " must be int64, got ", t.dtype());
}

// Rotate one 42-wide axis slice using the interleaved-half convention against
// cos/sin rows gathered by pos_idx. x_part is [B, S, heads, rope_dim].
static torch::Tensor apply_rope_axis(
    const torch::Tensor& x_part,
    const torch::Tensor& pos_idx,    // [S], int64
    const torch::Tensor& rope_cos,   // [max_pos, rope_dim]
    const torch::Tensor& rope_sin) {
    // Gather per-position rows then broadcast over batch and heads:
    // [S, rope_dim] -> [1, S, 1, rope_dim].
    auto cos_gathered = at::index_select(rope_cos, 0, pos_idx).unsqueeze(0).unsqueeze(2);
    auto sin_gathered = at::index_select(rope_sin, 0, pos_idx).unsqueeze(0).unsqueeze(2);

    const int64_t rope_dim = x_part.size(-1);
    const int64_t half_dim = rope_dim / 2;

    auto x_first = x_part.narrow(-1, 0, half_dim);
    auto x_second = x_part.narrow(-1, half_dim, rope_dim - half_dim);

    auto cos_first = cos_gathered.narrow(-1, 0, half_dim);
    auto sin_first = sin_gathered.narrow(-1, 0, half_dim);

    auto rotated_first = x_first * cos_first - x_second * sin_first;
    auto rotated_second = x_first * sin_first + x_second * cos_first;
    return at::cat({rotated_first, rotated_second}, -1);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& timestep_embedding,
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& adaln_linear_weight,
    const torch::Tensor& adaln_linear_bias,
    const torch::Tensor& to_q_weight,
    const torch::Tensor& to_q_bias,
    const torch::Tensor& to_k_weight,
    const torch::Tensor& to_k_bias,
    const torch::Tensor& to_v_weight,
    const torch::Tensor& to_v_bias,
    const torch::Tensor& to_k_context_weight,
    const torch::Tensor& to_k_context_bias,
    const torch::Tensor& to_v_context_weight,
    const torch::Tensor& to_v_context_bias,
    const torch::Tensor& to_out_weight,
    const torch::Tensor& to_out_bias,
    const torch::Tensor& pos_idx_axis0,
    const torch::Tensor& pos_idx_axis1,
    const torch::Tensor& pos_idx_axis2,
    const torch::Tensor& rope_cos_axis0,
    const torch::Tensor& rope_sin_axis0,
    const torch::Tensor& rope_cos_axis1,
    const torch::Tensor& rope_sin_axis1,
    const torch::Tensor& rope_cos_axis2,
    const torch::Tensor& rope_sin_axis2,
    int64_t is_joint_block,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(timestep_embedding, "timestep_embedding");
    check_fp32_cuda(encoder_hidden_states, "encoder_hidden_states");
    check_fp32_cuda(adaln_linear_weight, "adaln_linear_weight");
    check_fp32_cuda(adaln_linear_bias, "adaln_linear_bias");
    check_fp32_cuda(to_q_weight, "to_q_weight");
    check_fp32_cuda(to_q_bias, "to_q_bias");
    check_fp32_cuda(to_k_weight, "to_k_weight");
    check_fp32_cuda(to_k_bias, "to_k_bias");
    check_fp32_cuda(to_v_weight, "to_v_weight");
    check_fp32_cuda(to_v_bias, "to_v_bias");
    check_fp32_cuda(to_k_context_weight, "to_k_context_weight");
    check_fp32_cuda(to_k_context_bias, "to_k_context_bias");
    check_fp32_cuda(to_v_context_weight, "to_v_context_weight");
    check_fp32_cuda(to_v_context_bias, "to_v_context_bias");
    check_fp32_cuda(to_out_weight, "to_out_weight");
    check_fp32_cuda(to_out_bias, "to_out_bias");
    check_int64_cuda(pos_idx_axis0, "pos_idx_axis0");
    check_int64_cuda(pos_idx_axis1, "pos_idx_axis1");
    check_int64_cuda(pos_idx_axis2, "pos_idx_axis2");
    check_fp32_cuda(rope_cos_axis0, "rope_cos_axis0");
    check_fp32_cuda(rope_sin_axis0, "rope_sin_axis0");
    check_fp32_cuda(rope_cos_axis1, "rope_cos_axis1");
    check_fp32_cuda(rope_sin_axis1, "rope_sin_axis1");
    check_fp32_cuda(rope_cos_axis2, "rope_cos_axis2");
    check_fp32_cuda(rope_sin_axis2, "rope_sin_axis2");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, seq, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(timestep_embedding.dim() == 2,
                "timestep_embedding must be 2D [B, hidden], got ", timestep_embedding.dim(), "D");
    TORCH_CHECK(encoder_hidden_states.dim() == 3,
                "encoder_hidden_states must be 3D [B, text_seq, joint_attn_dim], got ",
                encoder_hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 24;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kHiddenSize = 3072;
    constexpr int64_t kJointAttnDim = 4096;
    constexpr int64_t kRopeDimPerAxis = kHeadDim / 3;  // 42

    const int64_t batch = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    const std::vector<int64_t> hidden_norm_shape{kHiddenSize};
    const std::vector<int64_t> context_norm_shape{kJointAttnDim};

    // ===================== AdaLN modulation =====================
    // SiLU gate on the timestep embedding, then a single linear and chunk(6).
    auto timestep_activated = timestep_embedding * at::sigmoid(timestep_embedding);
    auto modulation = at::linear(timestep_activated, adaln_linear_weight, adaln_linear_bias);
    auto mod_chunks = modulation.chunk(6, /*dim=*/-1);
    auto scale_msa = mod_chunks[0];
    auto shift_msa = mod_chunks[1];
    auto gate_msa = mod_chunks[2];
    // mod_chunks[3..5] (scale_mlp / shift_mlp / gate_mlp) are unused on this
    // attention-only path; chunk(6) is kept to match the reference split.

    // Affine-free layer_norm over the hidden dim, then msa modulation.
    auto hidden_norm = at::layer_norm(hidden_states, hidden_norm_shape);
    auto hidden_mod = hidden_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1);

    // ===================== QKV projections + 3D RoPE =====================
    auto q = at::linear(hidden_mod, to_q_weight, to_q_bias)
                 .view({batch, seq_len, kNumHeads, kHeadDim});
    auto k = at::linear(hidden_mod, to_k_weight, to_k_bias)
                 .view({batch, seq_len, kNumHeads, kHeadDim});
    auto v = at::linear(hidden_mod, to_v_weight, to_v_bias)
                 .view({batch, seq_len, kNumHeads, kHeadDim});

    // Rotate the three 42-wide axis slices; leave the trailing 2 dims as-is.
    const int64_t rope_total = 3 * kRopeDimPerAxis;  // 126
    const torch::Tensor* pos_idx[3] = {&pos_idx_axis0, &pos_idx_axis1, &pos_idx_axis2};
    const torch::Tensor* rope_cos[3] = {&rope_cos_axis0, &rope_cos_axis1, &rope_cos_axis2};
    const torch::Tensor* rope_sin[3] = {&rope_sin_axis0, &rope_sin_axis1, &rope_sin_axis2};

    auto rotate_qk = [&](const torch::Tensor& x) {
        std::vector<torch::Tensor> parts;
        parts.reserve(4);
        for (int64_t axis = 0; axis < 3; ++axis) {
            auto slice = x.narrow(-1, axis * kRopeDimPerAxis, kRopeDimPerAxis);
            parts.push_back(apply_rope_axis(slice, *pos_idx[axis], *rope_cos[axis], *rope_sin[axis]));
        }
        // Trailing un-rotated remainder (head_dim - 3 * rope_dim_per_axis).
        parts.push_back(x.narrow(-1, rope_total, kHeadDim - rope_total));
        return at::cat(parts, -1);
    };
    q = rotate_qk(q);
    k = rotate_qk(k);

    // ===================== Conditional cross-attention (host branch) =====================
    if (is_joint_block != 0) {
        const int64_t text_seq_len = encoder_hidden_states.size(1);
        auto encoder_norm = at::layer_norm(encoder_hidden_states, context_norm_shape);
        auto k_context = at::linear(encoder_norm, to_k_context_weight, to_k_context_bias)
                             .view({batch, text_seq_len, kNumHeads, kHeadDim});
        auto v_context = at::linear(encoder_norm, to_v_context_weight, to_v_context_bias)
                             .view({batch, text_seq_len, kNumHeads, kHeadDim});
        // Concatenate text key/values onto the image stream along the seq axis.
        k = at::cat({k, k_context}, /*dim=*/1);
        v = at::cat({v, v_context}, /*dim=*/1);
    }

    // ===================== Scaled dot-product attention =====================
    // Move heads in front of the sequence axis.
    q = q.transpose(1, 2);  // [B, heads, seq, head_dim]
    k = k.transpose(1, 2);  // [B, heads, kv_len, head_dim]
    v = v.transpose(1, 2);  // [B, heads, kv_len, head_dim]

    auto attn_scores = at::matmul(q, k.transpose(-2, -1)) * scale;  // [B, heads, seq, kv_len]
    auto attn_probs = at::softmax(attn_scores, /*dim=*/-1);
    auto attn_output = at::matmul(attn_probs, v);  // [B, heads, seq, head_dim]

    // Unwind heads back to [B, seq, hidden] (only image tokens survive).
    attn_output = attn_output.transpose(1, 2).contiguous()
                      .view({batch, seq_len, kHiddenSize});
    attn_output = at::linear(attn_output, to_out_weight, to_out_bias);

    // Gated residual add.
    auto result = hidden_states + gate_msa.unsqueeze(1) * attn_output;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "FIBO dual-stream attention with conditional cross-attention via pure "
          "ATen dispatch: SiLU-gated AdaLN modulation (chunk-6, msa applied), "
          "affine-free layer_norm, 3D RoPE gathered from cos/sin tables by int64 "
          "position indices (interleaved-half rotation over three 42-wide axes, "
          "2 dims left un-rotated), a host is_joint_block branch that "
          "concatenates context key/values for joint attention, scaled QK^T, "
          "fp32 softmax, attn @ V, output projection, and a gated residual add; "
          "no custom CUDA kernels; fp32; destination-passing style (output).");
}
