#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <string>

// Scaled dot-product attention with QK LayerNorm normalization (SD3.5 block).
//
// Reference (problem_082_qk_norm_scaled_dot_product_attention), op-for-op:
//   num_heads = 24 ; head_dim = 64 ; scale = head_dim ** -0.5
//
//   qkv = (hidden_states @ qkv_weight.t()) + qkv_bias           # [B,S,3*dim]
//   qkv = qkv.reshape(B,S,3,24,64).permute(2,0,3,1,4)           # [3,B,24,S,64]
//   q, k, v = qkv[0], qkv[1], qkv[2]                            # each [B,24,S,64]
//
//   # LayerNorm over the last dim (head_dim=64), unbiased=False variance:
//   q_mean = q.mean(-1, keepdim); q_var = q.var(-1, unbiased=False, keepdim)
//   q = (q - q_mean) / sqrt(q_var + eps) * q_norm_weight + q_norm_bias
//   k_mean = k.mean(-1, keepdim); k_var = k.var(-1, unbiased=False, keepdim)
//   k = (k - k_mean) / sqrt(k_var + eps) * k_norm_weight + k_norm_bias
//
//   q = q * scale                                              # scale on Q, post-norm
//   attn_scores = q @ k.transpose(-2,-1)                       # [B,24,S,S]
//   attn_probs  = softmax(attn_scores, dim=-1)                 # fp32 (native dtype)
//   attn_output = attn_probs @ v                               # [B,24,S,64]
//   attn_output = attn_output.transpose(1,2).reshape(B,S,24*64)# [B,S,dim]
//   output = (attn_output @ out_proj_weight.t()) + out_proj_bias
//   return output
//
// Per definition.json (insertion order):
//   hidden_states   : [batch_size, seq_len, dim=1536]            fp32
//   qkv_weight      : [dim_times_3=4608, dim=1536]               fp32
//   qkv_bias        : [dim_times_3=4608]                         fp32
//   q_norm_weight   : [head_dim=64]                              fp32
//   q_norm_bias     : [head_dim=64]                              fp32
//   k_norm_weight   : [head_dim=64]                              fp32
//   k_norm_bias     : [head_dim=64]                              fp32
//   out_proj_weight : [dim=1536, dim=1536]                       fp32
//   out_proj_bias   : [dim=1536]                                 fp32
//   eps             : scalar                                     fp32
//   output          : [batch_size, seq_len, dim=1536]            fp32 DPS
//
// Implementation: pure ATen / cuBLAS / cuDNN dispatch -- NO custom CUDA
// kernels. The Q/K LayerNorm over the trailing head_dim axis is delegated to
// at::layer_norm({head_dim}, weight, bias, eps), which computes exactly the
// reference's (x-mean)/sqrt(var+eps)*w+b with biased (unbiased=False) variance
// over the last axis -- matching q.var(unbiased=False). The scores are
// materialized through an explicit matmul (not a fused SDPA) so the scale,
// softmax dtype, and op order match the reference precisely; scale is applied
// to Q AFTER the LayerNorm and BEFORE the QK^T matmul, exactly as written.
// Softmax stays in the native fp32 dtype. Output is written in place (DPS).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& q_norm_bias,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& k_norm_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    double eps,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(qkv_weight, "qkv_weight");
    check_fp32_cuda(qkv_bias, "qkv_bias");
    check_fp32_cuda(q_norm_weight, "q_norm_weight");
    check_fp32_cuda(q_norm_bias, "q_norm_bias");
    check_fp32_cuda(k_norm_weight, "k_norm_weight");
    check_fp32_cuda(k_norm_bias, "k_norm_bias");
    check_fp32_cuda(out_proj_weight, "out_proj_weight");
    check_fp32_cuda(out_proj_bias, "out_proj_bias");
    check_fp32_cuda(output, "output");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, dim], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 24;
    constexpr int64_t kHeadDim = 64;
    const double scale = std::pow(static_cast<double>(kHeadDim), -0.5);  // head_dim ** -0.5

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t dim = hidden_states.size(2);

    // --- QKV projection + bias. ---
    auto qkv = at::matmul(hidden_states, qkv_weight.t()) + qkv_bias;  // [B,S,3*dim]

    // --- Reshape and split into q, k, v ([B,24,S,64] each). ---
    qkv = qkv.reshape({batch_size, seq_len, 3, kNumHeads, kHeadDim});
    qkv = qkv.permute({2, 0, 3, 1, 4});  // [3,B,24,S,64]
    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    // --- LayerNorm over head_dim (last axis) on Q and K, affine, biased var. ---
    q = at::layer_norm(q, /*normalized_shape=*/{kHeadDim}, q_norm_weight, q_norm_bias, eps);
    k = at::layer_norm(k, /*normalized_shape=*/{kHeadDim}, k_norm_weight, k_norm_bias, eps);

    // --- Scale Q (post-norm, pre-matmul), exactly as the reference orders it. ---
    q = q * scale;

    // --- Attention scores (materialized), native-dtype softmax, attn @ V. ---
    auto attn_scores = at::matmul(q, k.transpose(-2, -1));  // [B,24,S,S]
    auto attn_probs = at::softmax(attn_scores, /*dim=*/-1);  // fp32 (native)
    auto attn_output = at::matmul(attn_probs, v);            // [B,24,S,64]

    // --- Merge heads, output projection + bias. ---
    attn_output = attn_output.transpose(1, 2).reshape({batch_size, seq_len, kNumHeads * kHeadDim});
    auto out = at::matmul(attn_output, out_proj_weight.t()) + out_proj_bias;  // [B,S,dim]

    // --- Destination-passing-style write. ---
    TORCH_CHECK(out.sizes() == output.sizes(),
                "output shape ", out.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "QK-norm scaled dot-product attention (SD3.5). Pure ATen / cuBLAS / "
          "cuDNN dispatch: QKV projection + bias, reshape/permute to heads, "
          "at::layer_norm over head_dim (affine, biased variance) on Q and K, "
          "scale Q by head_dim^-0.5 (post-norm, pre-matmul), materialized QK^T, "
          "fp32 softmax, attn @ V, head merge, output projection + bias. fp32; "
          "destination-passing style.");
}
