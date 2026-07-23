#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <vector>
#include <string>

// VAE decoder mid-block: ResNet -> single-head self-attention -> ResNet,
// reproduced op-for-op from the reference run(). The bottleneck carries 512
// channels, GroupNorm uses 32 groups throughout, and attention is single-head
// (num_heads = in_channels // attention_head_dim with attention_head_dim = 1,
// so head_dim equals the full channel count). The definition.json description
// text mentions "4 groups", but the axes table and the reference code both fix
// num_groups = 32, so 32 is the contractual value used here.
//
// Each ResNet block:
//   residual = h_in
//   h = group_norm(h_in, 32, norm1_w, norm1_b, eps)
//   h = silu(h)
//   h = conv2d(h, conv1_w, conv1_b, padding=1)
//   temb_proj = linear(silu(temb), time_emb_proj_w, time_emb_proj_b)
//   h = h + temb_proj[:, :, None, None]               # broadcast over H, W
//   h = group_norm(h, 32, norm2_w, norm2_b, eps)
//   h = silu(h)
//   h = conv2d(h, conv2_w, conv2_b, padding=1)
//   h_out = h + residual
//
// Attention block:
//   attn_residual = h_in
//   h = group_norm(h_in, 32, attn_group_norm_w, attn_group_norm_b, eps)
//   h = h.view(B, C, H*W).transpose(1, 2)             # [B, H*W, C]
//   q = linear(h, to_q_w, to_q_b)
//   k = linear(h, to_k_w, to_k_b)
//   v = linear(h, to_v_w, to_v_b)
//   scale = head_dim ** -0.5  (head_dim == C, single head)
//   scores = (q @ k^T) * scale                        # scale on raw scores,
//   probs  = softmax(scores, dim=-1)                  # before the softmax
//   h = probs @ v
//   h = linear(h, to_out_w, to_out_b)
//   h = h.transpose(1, 2).view(B, C, H, W)
//   h_out = h + attn_residual
//
// Per definition.json (insertion order):
//   hidden_states                : [batch_size, 512, height, width]  fp32
//   temb                         : [batch_size, 512]                 fp32
//   resnet1_norm1_weight/bias    : [512]                             fp32
//   resnet1_conv1_weight         : [512, 512, 3, 3]                  fp32
//   resnet1_conv1_bias           : [512]                             fp32
//   resnet1_time_emb_proj_weight : [512, 512]                        fp32
//   resnet1_time_emb_proj_bias   : [512]                             fp32
//   resnet1_norm2_weight/bias    : [512]                             fp32
//   resnet1_conv2_weight         : [512, 512, 3, 3]                  fp32
//   resnet1_conv2_bias           : [512]                             fp32
//   attn_group_norm_weight/bias  : [512]                             fp32
//   attn_to_q/k/v/out_weight     : [512, 512]                        fp32
//   attn_to_q/k/v/out_bias       : [512]                             fp32
//   resnet2_* (same layout as resnet1_*)                             fp32
//   eps                          : scalar                            fp32
//   output                       : [batch_size, 512, height, width]  fp32 DPS
//
// Implementation: pure ATen / cuDNN / cuBLAS dispatch -- NO custom CUDA
// kernels. at::group_norm matches the reference's per-group biased-variance
// normalize plus per-channel affine; at::silu / at::conv2d / at::linear /
// at::bmm / at::softmax match the remaining ops with the native fp32 precision
// profile. The time-embedding projection is broadcast over the spatial dims via
// an unsqueeze to [B, C, 1, 1]. The attention scale multiplies the raw QK^T
// product before the softmax over the key axis, exactly as the reference orders
// it; softmax stays in native fp32. Output is written in place (DPS).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// One ResNet sub-block: group_norm + silu + conv, time-embedding broadcast add,
// group_norm + silu + conv, then the input residual.
static torch::Tensor resnet_block(
    const torch::Tensor& h_in,
    const torch::Tensor& temb,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& conv1_weight,
    const torch::Tensor& conv1_bias,
    const torch::Tensor& time_emb_proj_weight,
    const torch::Tensor& time_emb_proj_bias,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    const torch::Tensor& conv2_weight,
    const torch::Tensor& conv2_bias,
    int64_t num_groups,
    double eps,
    const std::vector<int64_t>& stride1,
    const std::vector<int64_t>& pad1
) {
    auto residual = h_in;

    auto h = at::group_norm(h_in, num_groups, norm1_weight, norm1_bias, eps);
    h = at::silu(h);
    h = at::conv2d(h, conv1_weight, conv1_bias, /*stride=*/stride1, /*padding=*/pad1);

    // Time embedding: silu then linear, broadcast over the spatial dimensions.
    auto temb_proj = at::silu(temb);
    temb_proj = at::linear(temb_proj, time_emb_proj_weight, time_emb_proj_bias);  // [B, C]
    h = h + temb_proj.unsqueeze(-1).unsqueeze(-1);                                // [B, C, 1, 1]

    h = at::group_norm(h, num_groups, norm2_weight, norm2_bias, eps);
    h = at::silu(h);
    h = at::conv2d(h, conv2_weight, conv2_bias, /*stride=*/stride1, /*padding=*/pad1);

    return h + residual;
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& temb,
    const torch::Tensor& resnet1_norm1_weight,
    const torch::Tensor& resnet1_norm1_bias,
    const torch::Tensor& resnet1_conv1_weight,
    const torch::Tensor& resnet1_conv1_bias,
    const torch::Tensor& resnet1_time_emb_proj_weight,
    const torch::Tensor& resnet1_time_emb_proj_bias,
    const torch::Tensor& resnet1_norm2_weight,
    const torch::Tensor& resnet1_norm2_bias,
    const torch::Tensor& resnet1_conv2_weight,
    const torch::Tensor& resnet1_conv2_bias,
    const torch::Tensor& attn_group_norm_weight,
    const torch::Tensor& attn_group_norm_bias,
    const torch::Tensor& attn_to_q_weight,
    const torch::Tensor& attn_to_q_bias,
    const torch::Tensor& attn_to_k_weight,
    const torch::Tensor& attn_to_k_bias,
    const torch::Tensor& attn_to_v_weight,
    const torch::Tensor& attn_to_v_bias,
    const torch::Tensor& attn_to_out_weight,
    const torch::Tensor& attn_to_out_bias,
    const torch::Tensor& resnet2_norm1_weight,
    const torch::Tensor& resnet2_norm1_bias,
    const torch::Tensor& resnet2_conv1_weight,
    const torch::Tensor& resnet2_conv1_bias,
    const torch::Tensor& resnet2_time_emb_proj_weight,
    const torch::Tensor& resnet2_time_emb_proj_bias,
    const torch::Tensor& resnet2_norm2_weight,
    const torch::Tensor& resnet2_norm2_bias,
    const torch::Tensor& resnet2_conv2_weight,
    const torch::Tensor& resnet2_conv2_bias,
    double eps,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(temb, "temb");
    check_fp32_cuda(resnet1_norm1_weight, "resnet1_norm1_weight");
    check_fp32_cuda(resnet1_norm1_bias, "resnet1_norm1_bias");
    check_fp32_cuda(resnet1_conv1_weight, "resnet1_conv1_weight");
    check_fp32_cuda(resnet1_conv1_bias, "resnet1_conv1_bias");
    check_fp32_cuda(resnet1_time_emb_proj_weight, "resnet1_time_emb_proj_weight");
    check_fp32_cuda(resnet1_time_emb_proj_bias, "resnet1_time_emb_proj_bias");
    check_fp32_cuda(resnet1_norm2_weight, "resnet1_norm2_weight");
    check_fp32_cuda(resnet1_norm2_bias, "resnet1_norm2_bias");
    check_fp32_cuda(resnet1_conv2_weight, "resnet1_conv2_weight");
    check_fp32_cuda(resnet1_conv2_bias, "resnet1_conv2_bias");
    check_fp32_cuda(attn_group_norm_weight, "attn_group_norm_weight");
    check_fp32_cuda(attn_group_norm_bias, "attn_group_norm_bias");
    check_fp32_cuda(attn_to_q_weight, "attn_to_q_weight");
    check_fp32_cuda(attn_to_q_bias, "attn_to_q_bias");
    check_fp32_cuda(attn_to_k_weight, "attn_to_k_weight");
    check_fp32_cuda(attn_to_k_bias, "attn_to_k_bias");
    check_fp32_cuda(attn_to_v_weight, "attn_to_v_weight");
    check_fp32_cuda(attn_to_v_bias, "attn_to_v_bias");
    check_fp32_cuda(attn_to_out_weight, "attn_to_out_weight");
    check_fp32_cuda(attn_to_out_bias, "attn_to_out_bias");
    check_fp32_cuda(resnet2_norm1_weight, "resnet2_norm1_weight");
    check_fp32_cuda(resnet2_norm1_bias, "resnet2_norm1_bias");
    check_fp32_cuda(resnet2_conv1_weight, "resnet2_conv1_weight");
    check_fp32_cuda(resnet2_conv1_bias, "resnet2_conv1_bias");
    check_fp32_cuda(resnet2_time_emb_proj_weight, "resnet2_time_emb_proj_weight");
    check_fp32_cuda(resnet2_time_emb_proj_bias, "resnet2_time_emb_proj_bias");
    check_fp32_cuda(resnet2_norm2_weight, "resnet2_norm2_weight");
    check_fp32_cuda(resnet2_norm2_bias, "resnet2_norm2_bias");
    check_fp32_cuda(resnet2_conv2_weight, "resnet2_conv2_weight");
    check_fp32_cuda(resnet2_conv2_bias, "resnet2_conv2_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 4, "hidden_states must be 4D [B, C, H, W], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(temb.dim() == 2, "temb must be 2D [B, temb_channels], got ", temb.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumGroups = 32;

    const int64_t batch = hidden_states.size(0);
    const int64_t channels = hidden_states.size(1);
    const int64_t height = hidden_states.size(2);
    const int64_t width = hidden_states.size(3);
    const int64_t seq_len = height * width;

    // Single head: head_dim == channels, so the scale is channels ** -0.5.
    const double scale = std::pow(static_cast<double>(channels), -0.5);

    const std::vector<int64_t> stride1{1, 1};
    const std::vector<int64_t> pad1{1, 1};

    // ResNet block 1.
    auto h = resnet_block(
        hidden_states, temb,
        resnet1_norm1_weight, resnet1_norm1_bias,
        resnet1_conv1_weight, resnet1_conv1_bias,
        resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias,
        resnet1_norm2_weight, resnet1_norm2_bias,
        resnet1_conv2_weight, resnet1_conv2_bias,
        kNumGroups, eps, stride1, pad1);

    // Attention block. The residual branch carries the post-ResNet feature map.
    auto attn_residual = h;
    auto a = at::group_norm(h, kNumGroups, attn_group_norm_weight, attn_group_norm_bias, eps);

    // Flatten spatial dims and move channels last: [B, C, H, W] -> [B, H*W, C].
    a = a.view({batch, channels, seq_len}).transpose(1, 2).contiguous();

    // Per-position query/key/value projections with bias.
    auto q = at::linear(a, attn_to_q_weight, attn_to_q_bias);  // [B, H*W, C]
    auto k = at::linear(a, attn_to_k_weight, attn_to_k_bias);  // [B, H*W, C]
    auto v = at::linear(a, attn_to_v_weight, attn_to_v_bias);  // [B, H*W, C]

    // Single-head attention. The channels^-0.5 scale multiplies the raw QK^T
    // scores before the softmax over the key axis.
    auto scores = at::bmm(q, k.transpose(1, 2)) * scale;  // [B, H*W, H*W]
    auto probs = at::softmax(scores, /*dim=*/-1);         // fp32 (native)
    auto attn_out = at::bmm(probs, v);                    // [B, H*W, C]

    // Output projection, reshape back to spatial, residual add.
    attn_out = at::linear(attn_out, attn_to_out_weight, attn_to_out_bias);  // [B, H*W, C]
    attn_out = attn_out.transpose(1, 2).contiguous().view({batch, channels, height, width});
    h = attn_out + attn_residual;

    // ResNet block 2.
    auto result = resnet_block(
        h, temb,
        resnet2_norm1_weight, resnet2_norm1_bias,
        resnet2_conv1_weight, resnet2_conv1_bias,
        resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias,
        resnet2_norm2_weight, resnet2_norm2_bias,
        resnet2_conv2_weight, resnet2_conv2_bias,
        kNumGroups, eps, stride1, pad1);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "VAE decoder mid-block (ResNet -> single-head self-attention -> "
          "ResNet) via pure ATen dispatch: each ResNet block is group_norm(32) "
          "+ silu + conv3x3, a time-embedding broadcast add, group_norm + silu "
          "+ conv3x3, and an input residual; the attention block is group_norm, "
          "flatten to sequence, Q/K/V linear projections, single-head "
          "softmax(QK^T * channels^-0.5) @ V, output projection, reshape to "
          "spatial, residual add. fp32; destination-passing style.");
}
