#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <string>
#include <vector>

// Llama-3.2 Vision encoder layer with tanh-gated residual connections.
// Reproduces the reference run() op-for-op via pure ATen / cuBLAS dispatch --
// no custom device kernels. Two blocks, each a LayerNorm -> sublayer ->
// tanh-gated residual:
//   hidden = residual + tanh(gate_attn) * self_attention(LayerNorm(hidden))
//   output = residual + tanh(gate_ffn)  * mlp(LayerNorm(hidden))
//
// Pinned axes from the problem definition:
//   hidden_size       = 1280
//   num_heads         = 16
//   head_dim          = hidden_size / num_heads = 80
//   intermediate_size = 5120
//   scaling           = head_dim ** -0.5
//
// Per definition.json (insertion order):
//   hidden_state                     : [B, S, hidden=1280]            bf16
//   input_layernorm_weight           : [hidden]                        bf16
//   input_layernorm_bias             : [hidden]                        bf16
//   q_proj_weight                    : [hidden, hidden]                bf16
//   k_proj_weight                    : [hidden, hidden]                bf16
//   v_proj_weight                    : [hidden, hidden]                bf16
//   o_proj_weight                    : [hidden, hidden]                bf16
//   post_attention_layernorm_weight  : [hidden]                        bf16
//   post_attention_layernorm_bias    : [hidden]                        bf16
//   fc1_weight                       : [intermediate=5120, hidden]     bf16
//   fc1_bias                         : [intermediate]                  bf16
//   fc2_weight                       : [hidden, intermediate]          bf16
//   fc2_bias                         : [hidden]                        bf16
//   gate_attn                        : [1]                             bf16
//   gate_ffn                         : [1]                             bf16
//   norm_eps                         : scalar                          fp32 (by value)
//   output                           : [B, S, hidden]                  bf16 DPS
//
// Algorithm details that have to track the reference exactly:
//   * LayerNorm: the reference computes the biased (population) variance
//     ((x-mean)^2).mean(-1) in fp32, normalizes in fp32, casts back to bf16,
//     then applies weight and bias. at::layer_norm over the trailing hidden
//     axis matches this (fp32 internal accumulation, biased variance, affine
//     weight + bias) and is the canonical dispatch.
//   * The Q/K/V/O projections carry no bias (matmul with the transposed
//     weight, mirroring weight.t()); they reshape into [B, heads, S, head_dim].
//   * Attention scores are materialized (Q @ K^T) * scaling; softmax is taken
//     in fp32 and cast back to the query (bf16) dtype, exactly as the reference
//     does F.softmax(..., dtype=float32).to(query.dtype).
//   * The gates are [1] bf16 tensors. The reference multiplies by
//     tanh(gate) directly, so tanh runs in bf16 and the result broadcasts over
//     the sublayer output; this port keeps the tensor form (no .item()) so the
//     bf16 rounding of tanh(gate) matches the reference.
//   * The MLP is fc1 (with bias) -> GELU -> fc2 (with bias).
// Output is written in place (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_state,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& input_layernorm_bias,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& post_attention_layernorm_weight,
    const torch::Tensor& post_attention_layernorm_bias,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_bias,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc2_bias,
    const torch::Tensor& gate_attn,
    const torch::Tensor& gate_ffn,
    double norm_eps,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_state, "hidden_state");
    check_bf16_cuda(input_layernorm_weight, "input_layernorm_weight");
    check_bf16_cuda(input_layernorm_bias, "input_layernorm_bias");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_cuda(post_attention_layernorm_weight, "post_attention_layernorm_weight");
    check_bf16_cuda(post_attention_layernorm_bias, "post_attention_layernorm_bias");
    check_bf16_cuda(fc1_weight, "fc1_weight");
    check_bf16_cuda(fc1_bias, "fc1_bias");
    check_bf16_cuda(fc2_weight, "fc2_weight");
    check_bf16_cuda(fc2_bias, "fc2_bias");
    check_bf16_cuda(gate_attn, "gate_attn");
    check_bf16_cuda(gate_ffn, "gate_ffn");
    check_bf16_cuda(output, "output");
    TORCH_CHECK(hidden_state.dim() == 3,
                "hidden_state must be 3D [B, S, hidden], got ",
                hidden_state.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_state.device());

    constexpr int64_t kHiddenSize = 1280;
    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kHeadDim = kHiddenSize / kNumHeads;  // 80
    const double scaling = std::pow(static_cast<double>(kHeadDim), -0.5);

    const int64_t batch_size = hidden_state.size(0);
    const int64_t seq_len = hidden_state.size(1);
    const int64_t hidden_size = hidden_state.size(2);
    TORCH_CHECK(hidden_size == kHiddenSize,
                "hidden_size must be ", kHiddenSize, ", got ", hidden_size);

    const std::vector<int64_t> norm_shape = {kHiddenSize};

    // ---- Self-attention block with tanh-gated residual ----
    auto residual = hidden_state;

    // Input LayerNorm over the trailing hidden axis (biased variance, fp32
    // accumulation, affine weight + bias).
    auto normed = at::layer_norm(hidden_state, norm_shape,
                                 input_layernorm_weight, input_layernorm_bias,
                                 norm_eps);

    // Q / K / V projections (no bias); weight.t() is expressed via matmul with
    // the transposed weight.
    auto query = at::matmul(normed, q_proj_weight.t());  // [B, S, hidden]
    auto key = at::matmul(normed, k_proj_weight.t());
    auto value = at::matmul(normed, v_proj_weight.t());

    // Reshape into [B, heads, S, head_dim].
    query = query.view({batch_size, seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    key = key.view({batch_size, seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    value = value.view({batch_size, seq_len, kNumHeads, kHeadDim}).transpose(1, 2);

    // Scaled dot-product attention with the scores materialized; softmax in
    // fp32 cast back to the query (bf16) dtype.
    auto attn_weights = at::matmul(query, key.transpose(-2, -1)) * scaling;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                       .to(query.scalar_type());
    auto attn_output = at::matmul(attn_weights, value);  // [B, heads, S, head_dim]

    // Merge heads and apply the output projection (no bias).
    attn_output = attn_output.transpose(1, 2).contiguous()
                      .view({batch_size, seq_len, hidden_size});
    attn_output = at::matmul(attn_output, o_proj_weight.t());

    // Gated residual: tanh(gate_attn) runs in bf16 and broadcasts.
    auto hidden = residual + at::tanh(gate_attn) * attn_output;

    // ---- MLP block with tanh-gated residual ----
    residual = hidden;

    auto post_normed = at::layer_norm(hidden, norm_shape,
                                      post_attention_layernorm_weight,
                                      post_attention_layernorm_bias,
                                      norm_eps);

    // fc1 (with bias) -> GELU -> fc2 (with bias).
    auto mlp = at::matmul(post_normed, fc1_weight.t()) + fc1_bias;
    mlp = at::gelu(mlp);
    mlp = at::matmul(mlp, fc2_weight.t()) + fc2_bias;

    auto out = residual + at::tanh(gate_ffn) * mlp;

    // Destination-passing-style write.
    TORCH_CHECK(out.sizes() == output.sizes(),
                "output shape ", out.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Vision encoder layer with tanh-gated residuals. Pure ATen / cuBLAS "
          "dispatch: input LayerNorm (biased variance, affine), bias-free "
          "Q/K/V projections, head reshape, materialized scaled QK^T scores, "
          "fp32 softmax cast back to bf16, attn @ V, head merge, bias-free "
          "output projection, tanh(gate_attn) gated residual; then "
          "post-attention LayerNorm, fc1+bias -> GELU -> fc2+bias MLP, and a "
          "tanh(gate_ffn) gated residual. bf16; destination-passing style.");
}
