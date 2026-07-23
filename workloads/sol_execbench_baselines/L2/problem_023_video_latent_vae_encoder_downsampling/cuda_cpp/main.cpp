#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Video-latent VAE encoder downsampling block, reproduced op-for-op from the
// reference run(). A stack of 3D convolutions, group normalizations, SiLU
// activations, and 1x1x1 residual projections compresses a video tensor in
// time and space. num_groups is fixed at 32; the three downsampling blocks use
// 32//4, 32//2, and 32 groups respectively (the latent projection also uses 32):
//
//   h = conv3d(x, input_conv_w, input_conv_b, stride=1, padding=1)
//
//   # block 1 (temporal + spatial downsample, stride 2 on all three axes)
//   id1 = conv3d(h, down1_res_w, down1_res_b, stride=2)            # 1x1x1, pad 0
//   h   = group_norm(h, 32//4, down1_gn_w, down1_gn_b, eps)
//   h   = silu(h)
//   h   = conv3d(h, down1_conv_w, down1_conv_b, stride=2, padding=1)
//   h   = h + id1
//
//   # block 2 (same shape pattern as block 1)
//   id2 = conv3d(h, down2_res_w, down2_res_b, stride=2)
//   h   = group_norm(h, 32//2, down2_gn_w, down2_gn_b, eps)
//   h   = silu(h)
//   h   = conv3d(h, down2_conv_w, down2_conv_b, stride=2, padding=1)
//   h   = h + id2
//
//   # block 3 (spatial-only downsample, stride (1, 2, 2))
//   id3 = conv3d(h, down3_res_w, down3_res_b, stride=(1, 2, 2))
//   h   = group_norm(h, 32, down3_gn_w, down3_gn_b, eps)
//   h   = silu(h)
//   h   = conv3d(h, down3_conv_w, down3_conv_b, stride=(1, 2, 2), padding=1)
//   h   = h + id3
//
//   # latent projection
//   h      = group_norm(h, 32, latent_gn_w, latent_gn_b, eps)
//   h      = silu(h)
//   output = conv3d(h, latent_conv_w, latent_conv_b, stride=1, padding=1)
//
// The three residual projections are 1x1x1 convolutions carrying the matching
// stride with the default zero padding, so they downsample without mixing
// neighbors. The first two blocks halve all three spatial axes (stride 2 on
// time, height, width); the third halves only height and width (stride 1 on
// time). group_norm runs before the SiLU in every block, with eps as the
// single fp32 scalar (passed as double).
//
// Per definition.json (insertion order):
//   x                  : [batch_size, in_channels, frames, height, width]  fp32
//   input_conv_weight  : [hidden_div4, in_channels, 3, 3, 3]               fp32
//   input_conv_bias    : [hidden_div4]                                     fp32
//   down1_gn_weight    : [hidden_div4]                                     fp32
//   down1_gn_bias      : [hidden_div4]                                     fp32
//   down1_conv_weight  : [hidden_div2, hidden_div4, 3, 4, 4]               fp32
//   down1_conv_bias    : [hidden_div2]                                     fp32
//   down1_res_weight   : [hidden_div2, hidden_div4, 1, 1, 1]               fp32
//   down1_res_bias     : [hidden_div2]                                     fp32
//   down2_gn_weight    : [hidden_div2]                                     fp32
//   down2_gn_bias      : [hidden_div2]                                     fp32
//   down2_conv_weight  : [hidden_size, hidden_div2, 3, 4, 4]               fp32
//   down2_conv_bias    : [hidden_size]                                     fp32
//   down2_res_weight   : [hidden_size, hidden_div2, 1, 1, 1]               fp32
//   down2_res_bias     : [hidden_size]                                     fp32
//   down3_gn_weight    : [hidden_size]                                     fp32
//   down3_gn_bias      : [hidden_size]                                     fp32
//   down3_conv_weight  : [hidden_size, hidden_size, 3, 4, 4]               fp32
//   down3_conv_bias    : [hidden_size]                                     fp32
//   down3_res_weight   : [hidden_size, hidden_size, 1, 1, 1]               fp32
//   down3_res_bias     : [hidden_size]                                     fp32
//   latent_gn_weight   : [hidden_size]                                     fp32
//   latent_gn_bias     : [hidden_size]                                     fp32
//   latent_conv_weight : [latent_channels, hidden_size, 3, 3, 3]           fp32
//   latent_conv_bias   : [latent_channels]                                 fp32
//   eps                : scalar                                            fp32
//   output             : [batch_size, latent_channels, frames_div4,
//                         height_div8, width_div8]            fp32 DPS (pre-alloc)
//
// Implementation: pure ATen / cuDNN dispatch -- NO custom CUDA kernels. Each
// at::conv3d routes through cuDNN and at::group_norm / at::silu reproduce the
// normalization and activation, all in the reference's native fp32 precision
// profile (no extra casts). Output is written in place (destination-passing
// style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& input_conv_weight,
    const torch::Tensor& input_conv_bias,
    const torch::Tensor& down1_gn_weight,
    const torch::Tensor& down1_gn_bias,
    const torch::Tensor& down1_conv_weight,
    const torch::Tensor& down1_conv_bias,
    const torch::Tensor& down1_res_weight,
    const torch::Tensor& down1_res_bias,
    const torch::Tensor& down2_gn_weight,
    const torch::Tensor& down2_gn_bias,
    const torch::Tensor& down2_conv_weight,
    const torch::Tensor& down2_conv_bias,
    const torch::Tensor& down2_res_weight,
    const torch::Tensor& down2_res_bias,
    const torch::Tensor& down3_gn_weight,
    const torch::Tensor& down3_gn_bias,
    const torch::Tensor& down3_conv_weight,
    const torch::Tensor& down3_conv_bias,
    const torch::Tensor& down3_res_weight,
    const torch::Tensor& down3_res_bias,
    const torch::Tensor& latent_gn_weight,
    const torch::Tensor& latent_gn_bias,
    const torch::Tensor& latent_conv_weight,
    const torch::Tensor& latent_conv_bias,
    double eps,
    torch::Tensor output
) {
    check_fp32_cuda(x, "x");
    check_fp32_cuda(input_conv_weight, "input_conv_weight");
    check_fp32_cuda(input_conv_bias, "input_conv_bias");
    check_fp32_cuda(down1_gn_weight, "down1_gn_weight");
    check_fp32_cuda(down1_gn_bias, "down1_gn_bias");
    check_fp32_cuda(down1_conv_weight, "down1_conv_weight");
    check_fp32_cuda(down1_conv_bias, "down1_conv_bias");
    check_fp32_cuda(down1_res_weight, "down1_res_weight");
    check_fp32_cuda(down1_res_bias, "down1_res_bias");
    check_fp32_cuda(down2_gn_weight, "down2_gn_weight");
    check_fp32_cuda(down2_gn_bias, "down2_gn_bias");
    check_fp32_cuda(down2_conv_weight, "down2_conv_weight");
    check_fp32_cuda(down2_conv_bias, "down2_conv_bias");
    check_fp32_cuda(down2_res_weight, "down2_res_weight");
    check_fp32_cuda(down2_res_bias, "down2_res_bias");
    check_fp32_cuda(down3_gn_weight, "down3_gn_weight");
    check_fp32_cuda(down3_gn_bias, "down3_gn_bias");
    check_fp32_cuda(down3_conv_weight, "down3_conv_weight");
    check_fp32_cuda(down3_conv_bias, "down3_conv_bias");
    check_fp32_cuda(down3_res_weight, "down3_res_weight");
    check_fp32_cuda(down3_res_bias, "down3_res_bias");
    check_fp32_cuda(latent_gn_weight, "latent_gn_weight");
    check_fp32_cuda(latent_gn_bias, "latent_gn_bias");
    check_fp32_cuda(latent_conv_weight, "latent_conv_weight");
    check_fp32_cuda(latent_conv_bias, "latent_conv_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(x.dim() == 5, "x must be 5D [B, C, T, H, W], got ", x.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    constexpr int64_t kNumGroups = 32;

    // Volumetric stride / padding triples. The downsampling convolutions pad by
    // 1 on every spatial axis; the 1x1x1 residual projections take no padding.
    const std::vector<int64_t> stride1{1, 1, 1};
    const std::vector<int64_t> stride2{2, 2, 2};
    const std::vector<int64_t> stride_spatial{1, 2, 2};
    const std::vector<int64_t> pad1{1, 1, 1};
    const std::vector<int64_t> pad0{0, 0, 0};

    // Initial projection: in_channels -> hidden_div4, shape preserved.
    auto h = at::conv3d(x, input_conv_weight, input_conv_bias,
                        /*stride=*/stride1, /*padding=*/pad1);

    // First downsampling block: stride-2 on all three axes.
    auto identity1 = at::conv3d(h, down1_res_weight, down1_res_bias,
                                /*stride=*/stride2, /*padding=*/pad0);
    h = at::group_norm(h, kNumGroups / 4, down1_gn_weight, down1_gn_bias, eps);
    h = at::silu(h);
    h = at::conv3d(h, down1_conv_weight, down1_conv_bias,
                   /*stride=*/stride2, /*padding=*/pad1);
    h = h + identity1;

    // Second downsampling block: stride-2 on all three axes.
    auto identity2 = at::conv3d(h, down2_res_weight, down2_res_bias,
                                /*stride=*/stride2, /*padding=*/pad0);
    h = at::group_norm(h, kNumGroups / 2, down2_gn_weight, down2_gn_bias, eps);
    h = at::silu(h);
    h = at::conv3d(h, down2_conv_weight, down2_conv_bias,
                   /*stride=*/stride2, /*padding=*/pad1);
    h = h + identity2;

    // Third downsampling block: spatial-only stride (1, 2, 2).
    auto identity3 = at::conv3d(h, down3_res_weight, down3_res_bias,
                                /*stride=*/stride_spatial, /*padding=*/pad0);
    h = at::group_norm(h, kNumGroups, down3_gn_weight, down3_gn_bias, eps);
    h = at::silu(h);
    h = at::conv3d(h, down3_conv_weight, down3_conv_bias,
                   /*stride=*/stride_spatial, /*padding=*/pad1);
    h = h + identity3;

    // Latent projection: group_norm + silu + conv3d, shape preserved.
    h = at::group_norm(h, kNumGroups, latent_gn_weight, latent_gn_bias, eps);
    h = at::silu(h);
    auto result = at::conv3d(h, latent_conv_weight, latent_conv_bias,
                             /*stride=*/stride1, /*padding=*/pad1);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Video-latent VAE encoder downsampling via pure ATen dispatch: an "
          "initial conv3d projection, then three downsampling blocks (each a "
          "stride-matched 1x1x1 residual conv3d alongside group_norm + silu + "
          "stride-2 conv3d, summed) -- the first two halving all three spatial "
          "axes, the third halving height and width only -- and a latent "
          "projection of group_norm + silu + conv3d. group_norm uses 8/16/32/32 "
          "groups across the four normalizations. fp32; destination-passing "
          "style.");
}
