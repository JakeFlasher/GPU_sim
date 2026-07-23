#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <vector>

// Whisper / Voxtral-style audio encoder front block: two strided 1D
// convolutions with GELU, learned positional embeddings, then the first
// encoder layer's self-attention and FFN sublayers.
//
// Reference (055_audio_encoder_conv_positional_layer_stack), op-for-op:
//   x = F.conv1d(input_features, conv1_weight, conv1_bias, padding=1)
//   x = F.gelu(x)
//   x = F.conv1d(x, conv2_weight, conv2_bias, stride=2, padding=1)
//   x = F.gelu(x)
//   x = x.permute(0, 2, 1)                            # (B, seq, d_model)
//   hidden = x + embed_positions_weight
//   residual = hidden
//   # LayerNorm in fp32, fold weight/bias in fp32, cast back to bf16
//   q = F.linear(hidden, q_proj_weight, q_proj_bias) * (head_dim ** -0.5)
//   k = F.linear(hidden, k_proj_weight, None)         # k_proj has NO bias
//   v = F.linear(hidden, v_proj_weight, v_proj_bias)
//   reshape each to (B, num_heads, seq, head_dim)
//   attn = softmax(q @ k^T, fp32) -> bf16
//   out  = (attn @ v) -> (B, seq, d_model) -> F.linear(out_proj_weight, out_proj_bias)
//   hidden = residual + out
//   residual = hidden
//   # second LayerNorm (same fp32 recipe)
//   hidden = F.linear(fc1_weight, fc1_bias); F.gelu; F.linear(fc2_weight, fc2_bias)
//   hidden = residual + hidden
//
// Per definition.json (insertion order):
//   input_features              : [batch, num_mel_bins=80, input_seq_len=3000]  bf16
//   conv1_weight                : [d_model=5120, 80, 3]                          bf16
//   conv1_bias                  : [d_model]                                      bf16
//   conv2_weight                : [d_model, d_model, 3]                          bf16
//   conv2_bias                  : [d_model]                                      bf16
//   embed_positions_weight      : [output_seq_len=1500, d_model]                 bf16
//   self_attn_layer_norm_weight : [d_model]                                      bf16
//   self_attn_layer_norm_bias   : [d_model]                                      bf16
//   q_proj_weight               : [d_model, d_model]                             bf16
//   q_proj_bias                 : [d_model]                                      bf16
//   k_proj_weight               : [d_model, d_model]                             bf16   (no bias)
//   v_proj_weight               : [d_model, d_model]                             bf16
//   v_proj_bias                 : [d_model]                                      bf16
//   out_proj_weight             : [d_model, d_model]                             bf16
//   out_proj_bias               : [d_model]                                      bf16
//   final_layer_norm_weight     : [d_model]                                      bf16
//   final_layer_norm_bias       : [d_model]                                      bf16
//   fc1_weight                  : [encoder_ffn_dim=20480, d_model]               bf16
//   fc1_bias                    : [encoder_ffn_dim]                              bf16
//   fc2_weight                  : [d_model, encoder_ffn_dim]                     bf16
//   fc2_bias                    : [d_model]                                      bf16
//   hidden_states               : [batch, output_seq_len, d_model]              bf16  (DPS)
//
// Constants are fixed by the workload table (d_model=5120, num_heads=20,
// head_dim=256). The two convolutions use padding 1; the second adds stride 2.
// Both convs and the FFN use the default exact-erf GELU (F.gelu with no
// approximate= argument). k_proj is bias-free, matching the reference's
// F.linear(..., None). The query is scaled by head_dim**-0.5 before the
// attention score matmul.
//
// Precision: inputs and output are bfloat16. The reference upcasts only inside
// the two LayerNorms (mean / variance / normalize in fp32, with the affine
// weight/bias themselves promoted to fp32) and inside the attention softmax
// (computed in fp32, then cast back to bf16); every other op stays bf16. We
// reproduce exactly those casts and no others -- the q/k/v projections,
// matmuls, and FFN linears run in bf16, leaving wider accumulation to the
// ATen / cuBLAS / cuDNN dispatch.
//
// Implementation: pure ATen / cuDNN / cuBLAS dispatch -- no custom CUDA
// kernels. Output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

// LayerNorm computed in fp32 over the last dimension (the reference promotes
// the affine weight/bias to fp32 before folding), result cast back to bf16.
static torch::Tensor layer_norm_fp32(
    const torch::Tensor& x_bf16,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
    auto x = x_bf16.to(torch::kFloat32);
    auto mean = x.mean(/*dim=*/{-1}, /*keepdim=*/true);
    auto centered = x - mean;
    auto var = (centered * centered).mean(/*dim=*/{-1}, /*keepdim=*/true);
    auto normed = centered / torch::sqrt(var + eps);
    auto affine = normed * weight.to(torch::kFloat32) + bias.to(torch::kFloat32);
    return affine.to(torch::kBFloat16);
}

void run(
    const torch::Tensor& input_features,
    const torch::Tensor& conv1_weight,
    const torch::Tensor& conv1_bias,
    const torch::Tensor& conv2_weight,
    const torch::Tensor& conv2_bias,
    const torch::Tensor& embed_positions_weight,
    const torch::Tensor& self_attn_layer_norm_weight,
    const torch::Tensor& self_attn_layer_norm_bias,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& q_proj_bias,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& v_proj_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    const torch::Tensor& final_layer_norm_weight,
    const torch::Tensor& final_layer_norm_bias,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_bias,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc2_bias,
    torch::Tensor hidden_states
) {
    check_bf16_cuda(input_features, "input_features");
    check_bf16_cuda(conv1_weight, "conv1_weight");
    check_bf16_cuda(conv1_bias, "conv1_bias");
    check_bf16_cuda(conv2_weight, "conv2_weight");
    check_bf16_cuda(conv2_bias, "conv2_bias");
    check_bf16_cuda(embed_positions_weight, "embed_positions_weight");
    check_bf16_cuda(self_attn_layer_norm_weight, "self_attn_layer_norm_weight");
    check_bf16_cuda(self_attn_layer_norm_bias, "self_attn_layer_norm_bias");
    check_bf16_cuda(q_proj_weight, "q_proj_weight");
    check_bf16_cuda(q_proj_bias, "q_proj_bias");
    check_bf16_cuda(k_proj_weight, "k_proj_weight");
    check_bf16_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_cuda(v_proj_bias, "v_proj_bias");
    check_bf16_cuda(out_proj_weight, "out_proj_weight");
    check_bf16_cuda(out_proj_bias, "out_proj_bias");
    check_bf16_cuda(final_layer_norm_weight, "final_layer_norm_weight");
    check_bf16_cuda(final_layer_norm_bias, "final_layer_norm_bias");
    check_bf16_cuda(fc1_weight, "fc1_weight");
    check_bf16_cuda(fc1_bias, "fc1_bias");
    check_bf16_cuda(fc2_weight, "fc2_weight");
    check_bf16_cuda(fc2_bias, "fc2_bias");
    check_bf16_cuda(hidden_states, "hidden_states");
    TORCH_CHECK(input_features.dim() == 3,
                "input_features must be 3D [batch, mel, time], got ", input_features.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(input_features.device());

    constexpr int64_t d_model = 5120;
    constexpr int64_t num_heads = 20;
    constexpr int64_t head_dim = 256;
    const double scaling = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const double ln_eps = 1e-5;

    // Stage 1: two 1D convolutions, each followed by exact-erf GELU. The first
    // keeps the sequence length (padding 1); the second halves it (stride 2).
    auto x = at::conv1d(input_features, conv1_weight, conv1_bias,
                        /*stride=*/std::vector<int64_t>{1},
                        /*padding=*/std::vector<int64_t>{1});
    x = at::gelu(x);
    x = at::conv1d(x, conv2_weight, conv2_bias,
                   /*stride=*/std::vector<int64_t>{2},
                   /*padding=*/std::vector<int64_t>{1});
    x = at::gelu(x);

    // (batch, d_model, seq) -> (batch, seq, d_model).
    x = x.permute({0, 2, 1});

    const int64_t bsz = x.size(0);
    const int64_t seq_len = x.size(1);

    // Stage 2: add the learned positional embeddings (broadcast over batch).
    auto hidden = x + embed_positions_weight;

    // Stage 3: self-attention sublayer.
    auto residual = hidden;
    hidden = layer_norm_fp32(hidden, self_attn_layer_norm_weight,
                             self_attn_layer_norm_bias, ln_eps);

    // Q is scaled by head_dim**-0.5; K has no bias.
    auto query_states = at::linear(hidden, q_proj_weight, q_proj_bias) * scaling;
    auto key_states = at::linear(hidden, k_proj_weight);  // k_proj is bias-free
    auto value_states = at::linear(hidden, v_proj_weight, v_proj_bias);

    // (batch, seq, d_model) -> (batch, num_heads, seq, head_dim).
    query_states = query_states.view({bsz, seq_len, num_heads, head_dim})
                       .transpose(1, 2).contiguous();
    key_states = key_states.view({bsz, seq_len, num_heads, head_dim})
                     .transpose(1, 2).contiguous();
    value_states = value_states.view({bsz, seq_len, num_heads, head_dim})
                       .transpose(1, 2).contiguous();

    // Scaled-dot-product attention with no mask; softmax computed in fp32 then
    // cast back to the query dtype, matching the reference.
    auto attn_weights = at::matmul(query_states, key_states.transpose(2, 3));
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32)
                       .to(query_states.scalar_type());

    auto attn_output = at::matmul(attn_weights, value_states);
    attn_output = attn_output.transpose(1, 2).contiguous();
    attn_output = attn_output.reshape({bsz, seq_len, d_model});

    attn_output = at::linear(attn_output, out_proj_weight, out_proj_bias);
    hidden = residual + attn_output;

    // Stage 4: FFN sublayer.
    residual = hidden;
    hidden = layer_norm_fp32(hidden, final_layer_norm_weight,
                             final_layer_norm_bias, ln_eps);
    hidden = at::linear(hidden, fc1_weight, fc1_bias);
    hidden = at::gelu(hidden);
    hidden = at::linear(hidden, fc2_weight, fc2_bias);
    hidden = residual + hidden;

    TORCH_CHECK(hidden.sizes() == hidden_states.sizes(),
                "computed output shape ", hidden.sizes(),
                " does not match pre-allocated ", hidden_states.sizes());
    hidden_states.copy_(hidden);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Audio encoder conv + positional + first-layer attention/FFN stack. "
          "Pure ATen / cuDNN / cuBLAS dispatch: conv1d(pad1)+GELU, "
          "conv1d(stride2,pad1)+GELU, permute, add positional embeddings, "
          "fp32 LayerNorm, 20-head self-attention (q scaled, k bias-free, "
          "fp32 softmax), out_proj + residual, fp32 LayerNorm, GELU FFN + "
          "residual. bfloat16 I/O; destination-passing style.");
}
