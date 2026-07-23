#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <vector>
#include <string>
#include <cmath>

// SD3.5 JointTransformerBlock residual path: a dual-stream (image + text)
// transformer block with AdaLayerNormZero modulation driven by a timestep
// embedding, joint multi-head attention over the concatenated streams, and a
// per-stream GELU feed-forward, reproduced op-for-op from the reference run().
//
// Fixed constants from the spec:
//   inner_dim    = 1536   (image / hidden stream width, "dim" in the reference)
//   context_dim  = 1152   (text / encoder stream width)
//   num_heads    = 24
//   head_dim     = 64
//   scale        = head_dim ** -0.5
//
// AdaLayerNormZero: the two normalization passes use F.layer_norm with NO
// affine (no weight / bias) and a hardcoded eps of 1e-6 -- this is true
// at::layer_norm over the last dim, distinct from problem 070's manual affine
// LayerNorm. The supplied norm1_weight / norm1_bias (and the _context pair) are
// NOT norm affine parameters: they are linear projection weights that map the
// SiLU-activated timestep embedding to six modulation chunks. temb is passed
// through SiLU once, then a single linear per stream yields a [B, 6*width]
// tensor that chunk(6) splits into shift_msa, scale_msa, gate_msa, shift_mlp,
// scale_mlp, gate_mlp. The shift / scale modulate the normalized states as
// normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1); the gates scale the
// residual contributions. The same scheme runs independently for the context
// stream with its own c_* chunks.
//
// Joint attention: each stream projects its modulated, normalized states to
// Q/K/V with bias (image stream via to_q/k/v, text stream via add_q/k/v). The
// two streams are concatenated along the sequence axis (image first, then
// text) before the reshape to [B, heads, total_seq, head_dim]. A single scaled
// dot-product attention runs over the joint sequence: scores = (Q @ K^T) *
// scale with the scale on the raw product before the softmax over the key axis;
// softmax stays in native fp32; attn @ V. The result is reshaped back and split
// along the sequence axis into the image part (image_seq_len) and the text part
// (context_seq_len). Each part gets its own output projection (to_out for the
// image stream, to_add_out for the text stream) and a gated residual add.
//
// Feed-forward: each stream re-normalizes (layer_norm, eps 1e-6, no affine),
// applies its _mlp modulation, runs linear1 -> gelu(approximate="tanh") ->
// linear2, and adds a gated residual.
//
// Per definition.json (insertion order):
//   hidden_states          : [B, seq_len, inner_dim]          fp32
//   encoder_hidden_states  : [B, context_len, context_dim]    fp32
//   temb                   : [B, inner_dim]                    fp32
//   norm1_weight / bias            : modulation proj for the image stream
//   norm1_context_weight / bias    : modulation proj for the text stream
//   to_q/k/v_weight / bias         : image-stream QKV projections
//   add_q/k/v_proj_weight / bias   : text-stream QKV projections
//   to_out_weight / bias           : image-stream attention output projection
//   to_add_out_weight / bias       : text-stream attention output projection
//   ff_linear1/2_weight / bias         : image-stream feed-forward
//   ff_context_linear1/2_weight / bias : text-stream feed-forward
//   output_encoder_hidden_states : [B, context_len, context_dim]  fp32 DPS
//   output_hidden_states         : [B, seq_len, inner_dim]        fp32 DPS
//
// The two outputs are written in the definition's order: the encoder (text)
// stream first, then the hidden (image) stream -- matching the reference's
// `return encoder_hidden_states, hidden_states`.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels.
// at::layer_norm / at::silu / at::linear / at::matmul / at::softmax / at::gelu
// / at::cat match the reference ops with the native fp32 precision profile. The
// outputs are written in place into the pre-allocated destinations
// (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& temb,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& norm1_context_weight,
    const torch::Tensor& norm1_context_bias,
    const torch::Tensor& to_q_weight,
    const torch::Tensor& to_q_bias,
    const torch::Tensor& to_k_weight,
    const torch::Tensor& to_k_bias,
    const torch::Tensor& to_v_weight,
    const torch::Tensor& to_v_bias,
    const torch::Tensor& add_q_proj_weight,
    const torch::Tensor& add_q_proj_bias,
    const torch::Tensor& add_k_proj_weight,
    const torch::Tensor& add_k_proj_bias,
    const torch::Tensor& add_v_proj_weight,
    const torch::Tensor& add_v_proj_bias,
    const torch::Tensor& to_out_weight,
    const torch::Tensor& to_out_bias,
    const torch::Tensor& to_add_out_weight,
    const torch::Tensor& to_add_out_bias,
    const torch::Tensor& ff_linear1_weight,
    const torch::Tensor& ff_linear1_bias,
    const torch::Tensor& ff_linear2_weight,
    const torch::Tensor& ff_linear2_bias,
    const torch::Tensor& ff_context_linear1_weight,
    const torch::Tensor& ff_context_linear1_bias,
    const torch::Tensor& ff_context_linear2_weight,
    const torch::Tensor& ff_context_linear2_bias,
    torch::Tensor output_encoder_hidden_states,
    torch::Tensor output_hidden_states
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(encoder_hidden_states, "encoder_hidden_states");
    check_fp32_cuda(temb, "temb");
    check_fp32_cuda(norm1_weight, "norm1_weight");
    check_fp32_cuda(norm1_bias, "norm1_bias");
    check_fp32_cuda(norm1_context_weight, "norm1_context_weight");
    check_fp32_cuda(norm1_context_bias, "norm1_context_bias");
    check_fp32_cuda(to_q_weight, "to_q_weight");
    check_fp32_cuda(to_q_bias, "to_q_bias");
    check_fp32_cuda(to_k_weight, "to_k_weight");
    check_fp32_cuda(to_k_bias, "to_k_bias");
    check_fp32_cuda(to_v_weight, "to_v_weight");
    check_fp32_cuda(to_v_bias, "to_v_bias");
    check_fp32_cuda(add_q_proj_weight, "add_q_proj_weight");
    check_fp32_cuda(add_q_proj_bias, "add_q_proj_bias");
    check_fp32_cuda(add_k_proj_weight, "add_k_proj_weight");
    check_fp32_cuda(add_k_proj_bias, "add_k_proj_bias");
    check_fp32_cuda(add_v_proj_weight, "add_v_proj_weight");
    check_fp32_cuda(add_v_proj_bias, "add_v_proj_bias");
    check_fp32_cuda(to_out_weight, "to_out_weight");
    check_fp32_cuda(to_out_bias, "to_out_bias");
    check_fp32_cuda(to_add_out_weight, "to_add_out_weight");
    check_fp32_cuda(to_add_out_bias, "to_add_out_bias");
    check_fp32_cuda(ff_linear1_weight, "ff_linear1_weight");
    check_fp32_cuda(ff_linear1_bias, "ff_linear1_bias");
    check_fp32_cuda(ff_linear2_weight, "ff_linear2_weight");
    check_fp32_cuda(ff_linear2_bias, "ff_linear2_bias");
    check_fp32_cuda(ff_context_linear1_weight, "ff_context_linear1_weight");
    check_fp32_cuda(ff_context_linear1_bias, "ff_context_linear1_bias");
    check_fp32_cuda(ff_context_linear2_weight, "ff_context_linear2_weight");
    check_fp32_cuda(ff_context_linear2_bias, "ff_context_linear2_bias");
    check_fp32_cuda(output_encoder_hidden_states, "output_encoder_hidden_states");
    check_fp32_cuda(output_hidden_states, "output_hidden_states");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, seq_len, inner_dim], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(encoder_hidden_states.dim() == 3,
                "encoder_hidden_states must be 3D [B, context_len, context_dim], got ",
                encoder_hidden_states.dim(), "D");
    TORCH_CHECK(temb.dim() == 2, "temb must be 2D [B, inner_dim], got ", temb.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kInnerDim = 1536;
    constexpr int64_t kContextDim = 1152;
    constexpr int64_t kNumHeads = 24;
    constexpr int64_t kHeadDim = 64;
    constexpr double kLayerNormEps = 1e-6;

    const int64_t batch = hidden_states.size(0);
    const int64_t image_seq_len = hidden_states.size(1);
    const int64_t context_seq_len = encoder_hidden_states.size(1);

    // head_dim ** -0.5; multiplies the raw QK^T before the softmax.
    const double scale = std::pow(static_cast<double>(kHeadDim), -0.5);

    const std::vector<int64_t> image_norm_shape{kInnerDim};
    const std::vector<int64_t> context_norm_shape{kContextDim};

    // ===================== AdaLayerNormZero modulation =====================
    // Affine-free layer_norm over the last dim for both streams.
    auto norm_hidden = at::layer_norm(hidden_states, image_norm_shape,
                                      /*weight=*/{}, /*bias=*/{}, kLayerNormEps);
    auto norm_encoder = at::layer_norm(encoder_hidden_states, context_norm_shape,
                                       /*weight=*/{}, /*bias=*/{}, kLayerNormEps);

    // Modulation parameters from the SiLU-activated timestep embedding.
    auto temb_silu = at::silu(temb);
    auto mod = at::linear(temb_silu, norm1_weight, norm1_bias);            // [B, 6*inner_dim]
    auto mod_chunks = mod.chunk(6, /*dim=*/-1);
    auto shift_msa = mod_chunks[0];
    auto scale_msa = mod_chunks[1];
    auto gate_msa = mod_chunks[2];
    auto shift_mlp = mod_chunks[3];
    auto scale_mlp = mod_chunks[4];
    auto gate_mlp = mod_chunks[5];

    auto mod_ctx = at::linear(temb_silu, norm1_context_weight, norm1_context_bias);  // [B, 6*context_dim]
    auto mod_ctx_chunks = mod_ctx.chunk(6, /*dim=*/-1);
    auto c_shift_msa = mod_ctx_chunks[0];
    auto c_scale_msa = mod_ctx_chunks[1];
    auto c_gate_msa = mod_ctx_chunks[2];
    auto c_shift_mlp = mod_ctx_chunks[3];
    auto c_scale_mlp = mod_ctx_chunks[4];
    auto c_gate_mlp = mod_ctx_chunks[5];

    // Apply the msa modulation to the normalized states.
    norm_hidden = norm_hidden * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1);
    norm_encoder = norm_encoder * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1);

    // ===================== Joint dual-stream attention =====================
    // Image-stream QKV (with bias).
    auto q = at::linear(norm_hidden, to_q_weight, to_q_bias);
    auto k = at::linear(norm_hidden, to_k_weight, to_k_bias);
    auto v = at::linear(norm_hidden, to_v_weight, to_v_bias);

    // Text-stream QKV (with bias).
    auto c_q = at::linear(norm_encoder, add_q_proj_weight, add_q_proj_bias);
    auto c_k = at::linear(norm_encoder, add_k_proj_weight, add_k_proj_bias);
    auto c_v = at::linear(norm_encoder, add_v_proj_weight, add_v_proj_bias);

    // Concatenate along the sequence axis: image first, then text.
    q = at::cat({q, c_q}, /*dim=*/1);
    k = at::cat({k, c_k}, /*dim=*/1);
    v = at::cat({v, c_v}, /*dim=*/1);

    const int64_t total_seq_len = q.size(1);

    // Reshape to [B, heads, total_seq, head_dim].
    q = q.view({batch, total_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    k = k.view({batch, total_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    v = v.view({batch, total_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);

    // Scaled QK^T, softmax over the key axis (fp32), attn @ V.
    auto attn_weights = at::matmul(q, k.transpose(-2, -1)) * scale;  // [B, heads, T, T]
    attn_weights = at::softmax(attn_weights, /*dim=*/-1);
    auto attn_output = at::matmul(attn_weights, v);  // [B, heads, T, head_dim]

    // Reshape back to [B, total_seq, inner_dim].
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch, total_seq_len, -1});

    // Split back into image and text streams along the sequence axis.
    auto attn_split = attn_output.split({image_seq_len, context_seq_len}, /*dim=*/1);
    auto attn_output_img = attn_split[0];
    auto c_attn_output = attn_split[1];

    // Per-stream attention output projections.
    attn_output_img = at::linear(attn_output_img, to_out_weight, to_out_bias);
    c_attn_output = at::linear(c_attn_output, to_add_out_weight, to_add_out_bias);

    // Gated residual add for the attention path.
    auto hidden = hidden_states + gate_msa.unsqueeze(1) * attn_output_img;
    auto encoder = encoder_hidden_states + c_gate_msa.unsqueeze(1) * c_attn_output;

    // ===================== Per-stream feed-forward =====================
    norm_hidden = at::layer_norm(hidden, image_norm_shape,
                                 /*weight=*/{}, /*bias=*/{}, kLayerNormEps);
    norm_encoder = at::layer_norm(encoder, context_norm_shape,
                                  /*weight=*/{}, /*bias=*/{}, kLayerNormEps);

    // Apply the mlp modulation.
    norm_hidden = norm_hidden * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1);
    norm_encoder = norm_encoder * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1);

    // Image-stream feed-forward: linear -> gelu(tanh) -> linear.
    auto ff_hidden = at::linear(norm_hidden, ff_linear1_weight, ff_linear1_bias);
    ff_hidden = at::gelu(ff_hidden, /*approximate=*/"tanh");
    auto ff_output = at::linear(ff_hidden, ff_linear2_weight, ff_linear2_bias);

    // Text-stream feed-forward.
    auto c_ff_hidden = at::linear(norm_encoder, ff_context_linear1_weight, ff_context_linear1_bias);
    c_ff_hidden = at::gelu(c_ff_hidden, /*approximate=*/"tanh");
    auto c_ff_output = at::linear(c_ff_hidden, ff_context_linear2_weight, ff_context_linear2_bias);

    // Gated residual add for the feed-forward path.
    hidden = hidden + gate_mlp.unsqueeze(1) * ff_output;
    encoder = encoder + c_gate_mlp.unsqueeze(1) * c_ff_output;

    // ===================== Destination-passing-style writes =====================
    // Definition order: encoder (text) stream first, then hidden (image) stream.
    TORCH_CHECK(encoder.sizes() == output_encoder_hidden_states.sizes(),
                "output_encoder_hidden_states shape ", encoder.sizes(),
                " does not match pre-allocated ", output_encoder_hidden_states.sizes());
    TORCH_CHECK(hidden.sizes() == output_hidden_states.sizes(),
                "output_hidden_states shape ", hidden.sizes(),
                " does not match pre-allocated ", output_hidden_states.sizes());
    output_encoder_hidden_states.copy_(encoder);
    output_hidden_states.copy_(hidden);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "SD3.5 JointTransformerBlock residual path via pure ATen dispatch: "
          "AdaLayerNormZero modulation (affine-free layer_norm + SiLU-driven "
          "shift/scale/gate projected from temb), joint dual-stream attention "
          "(image + text Q/K/V concatenated, softmax(QK^T * head_dim^-0.5) @ V, "
          "split back with per-stream output projections and gated residuals), "
          "and per-stream GELU(tanh) feed-forward with gated residuals; no "
          "custom CUDA kernels; fp32; destination-passing style "
          "(output_encoder_hidden_states then output_hidden_states).");
}
