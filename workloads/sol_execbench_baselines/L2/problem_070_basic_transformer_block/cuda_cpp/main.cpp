#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <vector>
#include <string>
#include <cmath>

// SDXL-Refiner BasicTransformerBlock: self-attention, cross-attention, and a
// GEGLU feed-forward network, each wrapped in a pre-norm + residual pattern,
// reproduced op-for-op from the reference run().
//
// Fixed constants from the spec:
//   dim                 = 1280
//   num_attention_heads = 160
//   attention_head_dim  = 24
//   inner_dim           = num_attention_heads * attention_head_dim = 3840
//   scale               = attention_head_dim ** -0.5
//
// The three LayerNorms are computed manually exactly as the reference writes
// them, NOT via at::layer_norm: a per-row mean over the last dim, a biased
// variance (mean of squared deviation, i.e. var(unbiased=false)), then
// (x - mean) / sqrt(var + eps) followed by an affine scale-and-shift. This
// matches the reference's explicit mean / variance / rsqrt expression and its
// fp32 accumulation; no extra casts are introduced.
//
// Self-attention (attn1): Q/K/V projections from the normed states have NO
// bias; reshape to [B, heads, S, head_dim]; scores = (Q @ K^T) * scale with
// the scale applied to the raw product before the softmax over the key axis;
// softmax stays in native fp32; attn @ V; reshape back; output projection with
// bias; residual add.
//
// Cross-attention (attn2): Q comes from the normed hidden states, K/V from the
// encoder_hidden_states (the conditioning sequence). K/V therefore carry the
// encoder sequence length. Same scaling / softmax / projection structure.
//
// GEGLU feed-forward: the first linear (with bias) produces 2x the
// intermediate width; chunk(2) on the last dim splits it into the value half
// and the gate half; the activation is value * gelu(gate, approximate="tanh");
// the second linear (with bias) projects back; residual add.
//
// Per definition.json (insertion order):
//   hidden_states          : [B, spatial_seq_len, dim]               fp32
//   encoder_hidden_states  : [B, encoder_seq_len, cross_attn_dim]    fp32
//   norm1_weight / bias    : [dim]                                   fp32
//   attn1_to_q/k/v_weight  : [inner_dim, dim]                        fp32
//   attn1_to_out_weight    : [dim, inner_dim]                        fp32
//   attn1_to_out_bias      : [dim]                                   fp32
//   norm2_weight / bias    : [dim]                                   fp32
//   attn2_to_q_weight      : [inner_dim, dim]                        fp32
//   attn2_to_k/v_weight    : [inner_dim, cross_attention_dim]        fp32
//   attn2_to_out_weight    : [dim, inner_dim]                        fp32
//   attn2_to_out_bias      : [dim]                                   fp32
//   norm3_weight / bias    : [dim]                                   fp32
//   ff_linear1_weight      : [ff_gate_dim, dim]                      fp32
//   ff_linear1_bias        : [ff_gate_dim]                           fp32
//   ff_linear2_weight      : [dim, ff_intermediate]                  fp32
//   ff_linear2_bias        : [dim]                                   fp32
//   norm_eps               : scalar                                  fp32 -> double
//   output                 : [B, spatial_seq_len, dim]               fp32 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels.
// at::linear / at::matmul / at::softmax / at::gelu match the reference ops with
// the native fp32 precision profile. The output is written in place into the
// pre-allocated destination (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// Manual LayerNorm over the last dim, matching the reference's explicit
// formulation: per-row mean, biased variance (mean of squared deviation),
// normalize by 1/sqrt(var + eps), then the per-channel affine. All fp32.
static torch::Tensor manual_layer_norm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
    auto mean = x.mean(/*dim=*/-1, /*keepdim=*/true);
    auto centered = x - mean;
    auto var = centered.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto normed = centered / at::sqrt(var + eps);
    return normed * weight + bias;
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& attn1_to_q_weight,
    const torch::Tensor& attn1_to_k_weight,
    const torch::Tensor& attn1_to_v_weight,
    const torch::Tensor& attn1_to_out_weight,
    const torch::Tensor& attn1_to_out_bias,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    const torch::Tensor& attn2_to_q_weight,
    const torch::Tensor& attn2_to_k_weight,
    const torch::Tensor& attn2_to_v_weight,
    const torch::Tensor& attn2_to_out_weight,
    const torch::Tensor& attn2_to_out_bias,
    const torch::Tensor& norm3_weight,
    const torch::Tensor& norm3_bias,
    const torch::Tensor& ff_linear1_weight,
    const torch::Tensor& ff_linear1_bias,
    const torch::Tensor& ff_linear2_weight,
    const torch::Tensor& ff_linear2_bias,
    double norm_eps,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(encoder_hidden_states, "encoder_hidden_states");
    check_fp32_cuda(norm1_weight, "norm1_weight");
    check_fp32_cuda(norm1_bias, "norm1_bias");
    check_fp32_cuda(attn1_to_q_weight, "attn1_to_q_weight");
    check_fp32_cuda(attn1_to_k_weight, "attn1_to_k_weight");
    check_fp32_cuda(attn1_to_v_weight, "attn1_to_v_weight");
    check_fp32_cuda(attn1_to_out_weight, "attn1_to_out_weight");
    check_fp32_cuda(attn1_to_out_bias, "attn1_to_out_bias");
    check_fp32_cuda(norm2_weight, "norm2_weight");
    check_fp32_cuda(norm2_bias, "norm2_bias");
    check_fp32_cuda(attn2_to_q_weight, "attn2_to_q_weight");
    check_fp32_cuda(attn2_to_k_weight, "attn2_to_k_weight");
    check_fp32_cuda(attn2_to_v_weight, "attn2_to_v_weight");
    check_fp32_cuda(attn2_to_out_weight, "attn2_to_out_weight");
    check_fp32_cuda(attn2_to_out_bias, "attn2_to_out_bias");
    check_fp32_cuda(norm3_weight, "norm3_weight");
    check_fp32_cuda(norm3_bias, "norm3_bias");
    check_fp32_cuda(ff_linear1_weight, "ff_linear1_weight");
    check_fp32_cuda(ff_linear1_bias, "ff_linear1_bias");
    check_fp32_cuda(ff_linear2_weight, "ff_linear2_weight");
    check_fp32_cuda(ff_linear2_bias, "ff_linear2_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, spatial_seq_len, dim], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(encoder_hidden_states.dim() == 3,
                "encoder_hidden_states must be 3D [B, encoder_seq_len, cross_attention_dim], got ",
                encoder_hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 160;
    constexpr int64_t kHeadDim = 24;
    constexpr int64_t kInnerDim = kNumHeads * kHeadDim;  // 3840

    const int64_t batch = hidden_states.size(0);
    const int64_t spatial_seq_len = hidden_states.size(1);
    const int64_t encoder_seq_len = encoder_hidden_states.size(1);

    // attention_head_dim ** -0.5; multiplies the raw QK^T before the softmax.
    const double scale = std::pow(static_cast<double>(kHeadDim), -0.5);

    auto h = hidden_states;

    // ===================== Self-attention block =====================
    auto normed = manual_layer_norm(h, norm1_weight, norm1_bias, norm_eps);

    // Q/K/V projections (no bias) from the normed spatial states.
    auto query = at::linear(normed, attn1_to_q_weight);  // [B, S, inner_dim]
    auto key = at::linear(normed, attn1_to_k_weight);
    auto value = at::linear(normed, attn1_to_v_weight);

    // Reshape to [B, heads, S, head_dim].
    query = query.view({batch, spatial_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    key = key.view({batch, spatial_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    value = value.view({batch, spatial_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);

    // Scaled QK^T, softmax over the key axis (fp32), attn @ V.
    auto scores = at::matmul(query, key.transpose(-2, -1)) * scale;  // [B, heads, S, S]
    auto probs = at::softmax(scores, /*dim=*/-1);
    auto attn_out = at::matmul(probs, value);  // [B, heads, S, head_dim]

    // Reshape back to [B, S, inner_dim], output projection with bias, residual.
    attn_out = attn_out.transpose(1, 2).contiguous().view({batch, spatial_seq_len, kInnerDim});
    attn_out = at::linear(attn_out, attn1_to_out_weight, attn1_to_out_bias);
    h = h + attn_out;

    // ===================== Cross-attention block =====================
    normed = manual_layer_norm(h, norm2_weight, norm2_bias, norm_eps);

    // Q from the normed spatial states; K/V from the encoder conditioning.
    query = at::linear(normed, attn2_to_q_weight);                  // [B, S, inner_dim]
    key = at::linear(encoder_hidden_states, attn2_to_k_weight);     // [B, E, inner_dim]
    value = at::linear(encoder_hidden_states, attn2_to_v_weight);   // [B, E, inner_dim]

    query = query.view({batch, spatial_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    key = key.view({batch, encoder_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    value = value.view({batch, encoder_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);

    scores = at::matmul(query, key.transpose(-2, -1)) * scale;  // [B, heads, S, E]
    probs = at::softmax(scores, /*dim=*/-1);
    attn_out = at::matmul(probs, value);  // [B, heads, S, head_dim]

    attn_out = attn_out.transpose(1, 2).contiguous().view({batch, spatial_seq_len, kInnerDim});
    attn_out = at::linear(attn_out, attn2_to_out_weight, attn2_to_out_bias);
    h = h + attn_out;

    // ===================== GEGLU feed-forward block =====================
    normed = manual_layer_norm(h, norm3_weight, norm3_bias, norm_eps);

    // First linear produces 2x the intermediate width for the GEGLU gate.
    auto ff = at::linear(normed, ff_linear1_weight, ff_linear1_bias);  // [B, S, ff_gate_dim]
    auto ff_chunks = ff.chunk(2, /*dim=*/-1);
    auto ff_value = ff_chunks[0];
    auto ff_gate = ff_chunks[1];
    ff = ff_value * at::gelu(ff_gate, /*approximate=*/"tanh");

    // Second linear projects back, then the residual add.
    ff = at::linear(ff, ff_linear2_weight, ff_linear2_bias);  // [B, S, dim]
    auto result = h + ff;

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "SDXL-Refiner BasicTransformerBlock via pure ATen dispatch: pre-norm "
          "self-attention (no QKV bias) with softmax(QK^T * head_dim^-0.5) @ V "
          "and a residual, pre-norm cross-attention over the encoder states "
          "with a residual, and a pre-norm GEGLU feed-forward "
          "(value * gelu(gate, tanh)) with a residual; manual LayerNorm matches "
          "the reference's mean / biased-variance / rsqrt / affine; no custom "
          "CUDA kernels; fp32; destination-passing style (output).");
}
