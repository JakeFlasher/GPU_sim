#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <vector>

// GIT vision encoder layer: pre-LayerNorm multi-head self-attention with a
// residual, then pre-LayerNorm MLP (quick-GELU) with a residual.
//
// Reference (problem_079_git_vision_encoder_layer), op-for-op:
//   num_heads = 12 ; head_dim = hidden_size // num_heads ; scale = head_dim ** -0.5
//
//   residual = hidden_states
//   # LayerNorm1 (mean / biased var over the last axis, then affine)
//   m = hidden_states.mean(-1, keepdim); v = ((hidden_states - m) ** 2).mean(-1, keepdim)
//   hs = (hidden_states - m) / sqrt(v + eps) * layer_norm1_weight + layer_norm1_bias
//   q = hs @ q_proj_weight.t() + q_proj_bias
//   k = hs @ k_proj_weight.t() + k_proj_bias
//   v = hs @ v_proj_weight.t() + v_proj_bias
//   q = q.view(B, S, H, D).transpose(1, 2)        # [B, H, S, D]
//   k = k.view(B, S, H, D).transpose(1, 2)
//   v = v.view(B, S, H, D).transpose(1, 2)
//   attn = (q @ k.transpose(-1, -2)) * scale       # scale AFTER QK^T
//   attn = softmax(attn, dim=-1, dtype=float32)
//   out = attn @ v                                 # [B, H, S, D]
//   out = out.transpose(1, 2).contiguous().reshape(B, S, hidden_size)
//   out = out @ out_proj_weight.t() + out_proj_bias
//   hidden_states = residual + out
//
//   residual = hidden_states
//   # LayerNorm2 (same form)
//   hs = (hidden_states - m) / sqrt(v + eps) * layer_norm2_weight + layer_norm2_bias
//   hs = hs @ fc1_weight.t() + fc1_bias
//   hs = hs * sigmoid(1.702 * hs)                  # quick-GELU
//   hs = hs @ fc2_weight.t() + fc2_bias
//   output = residual + hs
//
// Per definition.json (insertion order):
//   hidden_states      : [batch_size, seq_len, hidden_size=768]   fp32
//   layer_norm1_weight : [768]                                    fp32
//   layer_norm1_bias   : [768]                                    fp32
//   q_proj_weight      : [768, 768]                               fp32
//   q_proj_bias        : [768]                                    fp32
//   k_proj_weight      : [768, 768]                               fp32
//   k_proj_bias        : [768]                                    fp32
//   v_proj_weight      : [768, 768]                               fp32
//   v_proj_bias        : [768]                                    fp32
//   out_proj_weight    : [768, 768]                               fp32
//   out_proj_bias      : [768]                                    fp32
//   layer_norm2_weight : [768]                                    fp32
//   layer_norm2_bias   : [768]                                    fp32
//   fc1_weight         : [intermediate_size=3072, 768]            fp32
//   fc1_bias           : [3072]                                   fp32
//   fc2_weight         : [768, 3072]                              fp32
//   fc2_bias           : [768]                                    fp32
//   layer_norm_eps     : scalar                                   fp32
//   output             : [batch_size, seq_len, hidden_size=768]   fp32  (DPS)
//
// The reference's manual LayerNorm uses a biased (population) variance over the
// last axis and the standard (x - mean) / sqrt(var + eps) * weight + bias
// affine; at::layer_norm({hidden_size}, weight, bias, eps) computes exactly
// that. Attention scores are materialized through an explicit matmul (not a
// fused SDPA) so the scale ordering -- applied to the scores AFTER QK^T -- and
// the op order match the reference precisely. num_heads is fixed at 12 by the
// reference; head_dim is derived from hidden_size. Softmax is in fp32, which is
// the native dtype here. quick-GELU is x * sigmoid(1.702 * x).
//
// Precision: everything stays float32; the reference never casts. Implementation
// is pure ATen / cuBLAS / cuDNN dispatch -- no custom CUDA kernels. Output is
// written in place into the pre-allocated destination (destination-passing
// style).

static void check_fp32_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& layer_norm1_weight,
    const torch::Tensor& layer_norm1_bias,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& q_proj_bias,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& k_proj_bias,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& v_proj_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    const torch::Tensor& layer_norm2_weight,
    const torch::Tensor& layer_norm2_bias,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_bias,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc2_bias,
    double layer_norm_eps,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(layer_norm1_weight, "layer_norm1_weight");
    check_fp32_cuda(layer_norm1_bias, "layer_norm1_bias");
    check_fp32_cuda(q_proj_weight, "q_proj_weight");
    check_fp32_cuda(q_proj_bias, "q_proj_bias");
    check_fp32_cuda(k_proj_weight, "k_proj_weight");
    check_fp32_cuda(k_proj_bias, "k_proj_bias");
    check_fp32_cuda(v_proj_weight, "v_proj_weight");
    check_fp32_cuda(v_proj_bias, "v_proj_bias");
    check_fp32_cuda(out_proj_weight, "out_proj_weight");
    check_fp32_cuda(out_proj_bias, "out_proj_bias");
    check_fp32_cuda(layer_norm2_weight, "layer_norm2_weight");
    check_fp32_cuda(layer_norm2_bias, "layer_norm2_bias");
    check_fp32_cuda(fc1_weight, "fc1_weight");
    check_fp32_cuda(fc1_bias, "fc1_bias");
    check_fp32_cuda(fc2_weight, "fc2_weight");
    check_fp32_cuda(fc2_bias, "fc2_bias");
    check_fp32_cuda(output, "output");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 12;
    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_size = hidden_states.size(2);
    const int64_t head_dim = hidden_size / kNumHeads;
    const double scale = std::pow(static_cast<double>(head_dim), -0.5);  // head_dim ** -0.5

    const std::vector<int64_t> norm_shape = {hidden_size};

    // Attention block: pre-LayerNorm, then materialized multi-head attention.
    auto residual = hidden_states;
    auto hs = at::layer_norm(hidden_states, norm_shape,
                             layer_norm1_weight, layer_norm1_bias, layer_norm_eps);

    auto q = at::linear(hs, q_proj_weight, q_proj_bias);
    auto k = at::linear(hs, k_proj_weight, k_proj_bias);
    auto v = at::linear(hs, v_proj_weight, v_proj_bias);

    q = q.view({batch_size, seq_len, kNumHeads, head_dim}).transpose(1, 2);  // [B, H, S, D]
    k = k.view({batch_size, seq_len, kNumHeads, head_dim}).transpose(1, 2);
    v = v.view({batch_size, seq_len, kNumHeads, head_dim}).transpose(1, 2);

    auto attn = at::matmul(q, k.transpose(-1, -2)) * scale;  // scale applied after QK^T
    attn = at::softmax(attn, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    auto attn_out = at::matmul(attn, v);  // [B, H, S, D]

    attn_out = attn_out.transpose(1, 2).contiguous().reshape({batch_size, seq_len, hidden_size});
    attn_out = at::linear(attn_out, out_proj_weight, out_proj_bias);
    auto hidden = residual + attn_out;

    // MLP block: pre-LayerNorm, fc1, quick-GELU, fc2, residual.
    residual = hidden;
    hs = at::layer_norm(hidden, norm_shape,
                        layer_norm2_weight, layer_norm2_bias, layer_norm_eps);
    hs = at::linear(hs, fc1_weight, fc1_bias);
    hs = hs * at::sigmoid(1.702 * hs);  // quick-GELU
    hs = at::linear(hs, fc2_weight, fc2_bias);
    auto result = residual + hs;

    TORCH_CHECK(result.sizes() == output.sizes(),
                "computed output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "GIT vision encoder layer. Pure ATen / cuBLAS / cuDNN dispatch: "
          "pre-LayerNorm multi-head self-attention (12 heads, materialized "
          "QK^T scaled by head_dim^-0.5 after the matmul, fp32 softmax) with a "
          "residual, then pre-LayerNorm MLP (fc1, quick-GELU x*sigmoid(1.702*x), "
          "fc2) with a residual. float32; destination-passing style.");
}
