#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <string>

// VAE self-attention block with group normalization.
//
// Reference (problem_089_vae_attention_block_with_groupnorm), reproduced
// op-for-op:
//   batch, channels, height, width = x.shape ; num_groups = 32 ; seq_len = H*W
//
//   residual = x
//
//   # Group normalization over each group's (channels_per_group, H, W) block
//   # with biased (unbiased=False) variance, then the per-channel affine:
//   x_norm = group_norm(x, num_groups, group_norm_weight, group_norm_bias, eps)
//
//   # Flatten spatial dims and move channels last: (B, C, H, W) -> (B, H*W, C)
//   x_seq = x_norm.view(B, C, seq_len).permute(0, 2, 1).contiguous()
//
//   # Per-position Q/K/V projections with bias (weights in [C, C] layout):
//   q = x_seq @ query_weight.t() + query_bias
//   k = x_seq @ key_weight.t()   + key_bias
//   v = x_seq @ value_weight.t() + value_bias
//
//   # Single-head attention; the channels^-0.5 scale multiplies the raw
//   # QK^T scores BEFORE the softmax over the key axis:
//   scale = channels ** -0.5
//   attn_scores  = bmm(q, k.transpose(1, 2)) * scale
//   attn_weights = softmax(attn_scores, dim=-1)
//   attn_output  = bmm(attn_weights, v)
//
//   # Output projection with bias, reshape back to spatial, residual add:
//   attn_output = attn_output @ proj_out_weight.t() + proj_out_bias
//   attn_output = attn_output.permute(0, 2, 1).contiguous().view(B, C, H, W)
//   output = residual + attn_output
//
// Per definition.json (insertion order):
//   x                 : [batch_size, channels, spatial_size, spatial_size]  fp32
//   group_norm_weight : [channels]                                          fp32
//   group_norm_bias   : [channels]                                          fp32
//   query_weight      : [channels, channels]                                fp32
//   query_bias        : [channels]                                          fp32
//   key_weight        : [channels, channels]                                fp32
//   key_bias          : [channels]                                          fp32
//   value_weight      : [channels, channels]                                fp32
//   value_bias        : [channels]                                          fp32
//   proj_out_weight   : [channels, channels]                                fp32
//   proj_out_bias     : [channels]                                          fp32
//   eps               : scalar                                              fp32
//   output            : [batch_size, channels, spatial_size, spatial_size]  fp32 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// group norm is at::group_norm, which computes exactly the reference's
// per-group (x - mean) / sqrt(var + eps) with biased variance followed by the
// per-channel weight/bias affine. The Q/K/V and output projections are
// at::linear (x @ weight.t() + bias). Attention scores are materialized through
// at::bmm so the channels^-0.5 scale, softmax dtype, and op order match the
// reference precisely: the scale is applied to the QK^T product BEFORE the
// softmax over the last axis. Softmax stays in the native fp32 dtype. The
// residual is the original (un-normalized) x. Output is written in place (DPS).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& group_norm_weight,
    const torch::Tensor& group_norm_bias,
    const torch::Tensor& query_weight,
    const torch::Tensor& query_bias,
    const torch::Tensor& key_weight,
    const torch::Tensor& key_bias,
    const torch::Tensor& value_weight,
    const torch::Tensor& value_bias,
    const torch::Tensor& proj_out_weight,
    const torch::Tensor& proj_out_bias,
    double eps,
    torch::Tensor output
) {
    check_fp32_cuda(x, "x");
    check_fp32_cuda(group_norm_weight, "group_norm_weight");
    check_fp32_cuda(group_norm_bias, "group_norm_bias");
    check_fp32_cuda(query_weight, "query_weight");
    check_fp32_cuda(query_bias, "query_bias");
    check_fp32_cuda(key_weight, "key_weight");
    check_fp32_cuda(key_bias, "key_bias");
    check_fp32_cuda(value_weight, "value_weight");
    check_fp32_cuda(value_bias, "value_bias");
    check_fp32_cuda(proj_out_weight, "proj_out_weight");
    check_fp32_cuda(proj_out_bias, "proj_out_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W], got ", x.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    constexpr int64_t kNumGroups = 32;

    const int64_t batch = x.size(0);
    const int64_t channels = x.size(1);
    const int64_t height = x.size(2);
    const int64_t width = x.size(3);
    const int64_t seq_len = height * width;
    const double scale = std::pow(static_cast<double>(channels), -0.5);  // channels ** -0.5

    // The residual branch carries the original (un-normalized) input forward.
    auto residual = x;

    // Group normalization over each group's spatial+channel block (biased
    // variance), followed by the per-channel affine. at::group_norm matches the
    // reference's manual mean/var/normalize/scale/shift exactly.
    auto x_norm = at::group_norm(x, kNumGroups, group_norm_weight, group_norm_bias, eps);

    // Flatten spatial dimensions and move channels to the last axis so each
    // spatial position becomes a sequence token: (B, C, H, W) -> (B, H*W, C).
    auto x_seq = x_norm.view({batch, channels, seq_len}).permute({0, 2, 1}).contiguous();

    // Per-position query/key/value projections with bias.
    auto q = at::linear(x_seq, query_weight, query_bias);  // [B, H*W, C]
    auto k = at::linear(x_seq, key_weight, key_bias);      // [B, H*W, C]
    auto v = at::linear(x_seq, value_weight, value_bias);  // [B, H*W, C]

    // Single-head attention. The channels^-0.5 scale multiplies the raw QK^T
    // scores before the softmax over the key axis, exactly as the reference
    // orders it.
    auto attn_scores = at::bmm(q, k.transpose(1, 2)) * scale;  // [B, H*W, H*W]
    auto attn_weights = at::softmax(attn_scores, /*dim=*/-1);  // fp32 (native)
    auto attn_output = at::bmm(attn_weights, v);               // [B, H*W, C]

    // Output projection with bias.
    attn_output = at::linear(attn_output, proj_out_weight, proj_out_bias);  // [B, H*W, C]

    // Reshape back to spatial layout and add the residual.
    attn_output = attn_output.permute({0, 2, 1}).contiguous().view({batch, channels, height, width});
    auto result = residual + attn_output;

    // Destination-passing-style write.
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "VAE self-attention block with group normalization via pure ATen "
          "dispatch: at::group_norm (32 groups), flatten to sequence, Q/K/V "
          "linear projections, single-head softmax(QK^T * channels^-0.5) @ V, "
          "output projection, reshape to spatial, residual add. fp32; "
          "destination-passing style.");
}
