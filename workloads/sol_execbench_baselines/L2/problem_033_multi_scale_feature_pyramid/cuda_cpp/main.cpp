#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Multi-scale feature pyramid (U-Net style encoder/decoder with skip
// connections), reproduced op-for-op from the reference run():
//
//   enc0  = conv2d(x,    encoder_conv0_w,  encoder_conv0_b,  padding=1)
//   down0 = conv2d(enc0, downsample_conv0_w, downsample_conv0_b, stride=2, padding=1)
//   enc1  = conv2d(down0,encoder_conv1_w,  encoder_conv1_b,  padding=1)
//   down1 = conv2d(enc1, downsample_conv1_w, downsample_conv1_b, stride=2, padding=1)
//   enc2  = conv2d(down1,encoder_conv2_w,  encoder_conv2_b,  padding=1)
//   feat  = conv2d(enc2, bottleneck_w,     bottleneck_b,     padding=1)
//   up0   = conv_transpose2d(feat, upsample_conv0_w, upsample_conv0_b, stride=2)
//   skip0 = cat([up0, enc1], dim=1)              # 64 + 64 -> 128 channels
//   dec0  = conv2d(skip0,decoder_conv0_w,  decoder_conv0_b,  padding=1)
//   up1   = conv_transpose2d(dec0, upsample_conv1_w, upsample_conv1_b, stride=2)
//   skip1 = cat([up1, enc0], dim=1)              # 32 + 32 -> 64 channels
//   dec1  = conv2d(skip1,decoder_conv1_w,  decoder_conv1_b,  padding=1)
//   output= conv2d(dec1, output_conv_w,    output_conv_b,    padding=1)
//
// The reference applies NO activation between stages -- every stage is a bare
// convolution / transposed convolution, with the two skip branches carried
// forward as the channel-wise concatenation of the upsampled feature with the
// matching-resolution encoder feature (enc1 at the first skip, enc0 at the
// second). The same-resolution 3x3 convolutions all use padding=1; the two
// downsample convolutions use stride=2 padding=1; the two upsample transposed
// convolutions use stride=2 with the default zero padding / output padding.
//
// Per definition.json (insertion order):
//   x                       : [batch_size, 3, height, width]   fp32
//   encoder_conv0_weight    : [32, 3, 3, 3]                    fp32
//   encoder_conv0_bias      : [32]                             fp32
//   downsample_conv0_weight : [64, 32, 3, 3]                   fp32
//   downsample_conv0_bias   : [64]                             fp32
//   encoder_conv1_weight    : [64, 64, 3, 3]                   fp32
//   encoder_conv1_bias      : [64]                             fp32
//   downsample_conv1_weight : [128, 64, 3, 3]                  fp32
//   downsample_conv1_bias   : [128]                            fp32
//   encoder_conv2_weight    : [128, 128, 3, 3]                 fp32
//   encoder_conv2_bias      : [128]                            fp32
//   bottleneck_weight       : [128, 128, 3, 3]                 fp32
//   bottleneck_bias         : [128]                            fp32
//   upsample_conv0_weight   : [128, 64, 2, 2]                  fp32
//   upsample_conv0_bias     : [64]                             fp32
//   decoder_conv0_weight    : [64, 128, 3, 3]                  fp32
//   decoder_conv0_bias      : [64]                             fp32
//   upsample_conv1_weight   : [64, 32, 2, 2]                   fp32
//   upsample_conv1_bias     : [32]                             fp32
//   decoder_conv1_weight    : [32, 64, 3, 3]                   fp32
//   decoder_conv1_bias      : [32]                             fp32
//   output_conv_weight      : [3, 32, 3, 3]                    fp32
//   output_conv_bias        : [3]                              fp32
//   output                  : [batch_size, 3, height, width]   fp32 DPS (pre-alloc)
//
// Implementation: pure ATen / cuDNN dispatch -- NO custom CUDA kernels. Each
// at::conv2d / at::conv_transpose2d routes through cuDNN with the same fp32
// precision profile the reference's F.conv2d / F.conv_transpose2d uses. The
// transposed-convolution weights keep the PyTorch [in_channels, out_channels,
// kh, kw] layout, so they are passed through unchanged. Output is written in
// place into the pre-allocated destination (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& encoder_conv0_weight,
    const torch::Tensor& encoder_conv0_bias,
    const torch::Tensor& downsample_conv0_weight,
    const torch::Tensor& downsample_conv0_bias,
    const torch::Tensor& encoder_conv1_weight,
    const torch::Tensor& encoder_conv1_bias,
    const torch::Tensor& downsample_conv1_weight,
    const torch::Tensor& downsample_conv1_bias,
    const torch::Tensor& encoder_conv2_weight,
    const torch::Tensor& encoder_conv2_bias,
    const torch::Tensor& bottleneck_weight,
    const torch::Tensor& bottleneck_bias,
    const torch::Tensor& upsample_conv0_weight,
    const torch::Tensor& upsample_conv0_bias,
    const torch::Tensor& decoder_conv0_weight,
    const torch::Tensor& decoder_conv0_bias,
    const torch::Tensor& upsample_conv1_weight,
    const torch::Tensor& upsample_conv1_bias,
    const torch::Tensor& decoder_conv1_weight,
    const torch::Tensor& decoder_conv1_bias,
    const torch::Tensor& output_conv_weight,
    const torch::Tensor& output_conv_bias,
    torch::Tensor output
) {
    check_fp32_cuda(x, "x");
    check_fp32_cuda(encoder_conv0_weight, "encoder_conv0_weight");
    check_fp32_cuda(encoder_conv0_bias, "encoder_conv0_bias");
    check_fp32_cuda(downsample_conv0_weight, "downsample_conv0_weight");
    check_fp32_cuda(downsample_conv0_bias, "downsample_conv0_bias");
    check_fp32_cuda(encoder_conv1_weight, "encoder_conv1_weight");
    check_fp32_cuda(encoder_conv1_bias, "encoder_conv1_bias");
    check_fp32_cuda(downsample_conv1_weight, "downsample_conv1_weight");
    check_fp32_cuda(downsample_conv1_bias, "downsample_conv1_bias");
    check_fp32_cuda(encoder_conv2_weight, "encoder_conv2_weight");
    check_fp32_cuda(encoder_conv2_bias, "encoder_conv2_bias");
    check_fp32_cuda(bottleneck_weight, "bottleneck_weight");
    check_fp32_cuda(bottleneck_bias, "bottleneck_bias");
    check_fp32_cuda(upsample_conv0_weight, "upsample_conv0_weight");
    check_fp32_cuda(upsample_conv0_bias, "upsample_conv0_bias");
    check_fp32_cuda(decoder_conv0_weight, "decoder_conv0_weight");
    check_fp32_cuda(decoder_conv0_bias, "decoder_conv0_bias");
    check_fp32_cuda(upsample_conv1_weight, "upsample_conv1_weight");
    check_fp32_cuda(upsample_conv1_bias, "upsample_conv1_bias");
    check_fp32_cuda(decoder_conv1_weight, "decoder_conv1_weight");
    check_fp32_cuda(decoder_conv1_bias, "decoder_conv1_bias");
    check_fp32_cuda(output_conv_weight, "output_conv_weight");
    check_fp32_cuda(output_conv_bias, "output_conv_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(x.dim() == 4, "x must be 4D [N, C, H, W], got ", x.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    // Same-resolution 3x3 convolutions preserve spatial size with padding=1;
    // the two downsample convolutions halve it with stride=2 padding=1; the two
    // transposed convolutions double it with stride=2 and default padding.
    const std::vector<int64_t> stride1{1, 1};
    const std::vector<int64_t> stride2{2, 2};
    const std::vector<int64_t> pad1{1, 1};
    const std::vector<int64_t> pad0{0, 0};

    // Encoder scale 0: 3 -> 32 channels at full resolution.
    auto enc0 = at::conv2d(x, encoder_conv0_weight, encoder_conv0_bias,
                           /*stride=*/stride1, /*padding=*/pad1);

    // Downsample 0: 32 -> 64 channels, spatial halved.
    auto down0 = at::conv2d(enc0, downsample_conv0_weight, downsample_conv0_bias,
                            /*stride=*/stride2, /*padding=*/pad1);

    // Encoder scale 1: 64 -> 64 channels.
    auto enc1 = at::conv2d(down0, encoder_conv1_weight, encoder_conv1_bias,
                           /*stride=*/stride1, /*padding=*/pad1);

    // Downsample 1: 64 -> 128 channels, spatial halved again.
    auto down1 = at::conv2d(enc1, downsample_conv1_weight, downsample_conv1_bias,
                            /*stride=*/stride2, /*padding=*/pad1);

    // Encoder scale 2: 128 -> 128 channels.
    auto enc2 = at::conv2d(down1, encoder_conv2_weight, encoder_conv2_bias,
                           /*stride=*/stride1, /*padding=*/pad1);

    // Bottleneck: 128 -> 128 channels.
    auto feat = at::conv2d(enc2, bottleneck_weight, bottleneck_bias,
                           /*stride=*/stride1, /*padding=*/pad1);

    // Upsample 0 (transposed conv, stride=2): 128 -> 64 channels, spatial
    // doubled to match enc1's resolution.
    auto up0 = at::conv_transpose2d(feat, upsample_conv0_weight, upsample_conv0_bias,
                                    /*stride=*/stride2, /*padding=*/pad0);

    // Skip connection with the matching-resolution encoder feature enc1.
    auto skip0 = at::cat({up0, enc1}, /*dim=*/1);  // 64 + 64 -> 128 channels

    // Decoder conv 0: 128 -> 64 channels.
    auto dec0 = at::conv2d(skip0, decoder_conv0_weight, decoder_conv0_bias,
                           /*stride=*/stride1, /*padding=*/pad1);

    // Upsample 1 (transposed conv, stride=2): 64 -> 32 channels, spatial
    // doubled to match enc0's resolution.
    auto up1 = at::conv_transpose2d(dec0, upsample_conv1_weight, upsample_conv1_bias,
                                    /*stride=*/stride2, /*padding=*/pad0);

    // Skip connection with the matching-resolution encoder feature enc0.
    auto skip1 = at::cat({up1, enc0}, /*dim=*/1);  // 32 + 32 -> 64 channels

    // Decoder conv 1: 64 -> 32 channels.
    auto dec1 = at::conv2d(skip1, decoder_conv1_weight, decoder_conv1_bias,
                           /*stride=*/stride1, /*padding=*/pad1);

    // Output projection: 32 -> 3 channels at full resolution.
    auto result = at::conv2d(dec1, output_conv_weight, output_conv_bias,
                             /*stride=*/stride1, /*padding=*/pad1);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Multi-scale feature pyramid (U-Net) via pure ATen dispatch: encoder "
          "convolutions with two stride-2 downsamples, a bottleneck convolution, "
          "two stride-2 transposed-convolution upsamples each followed by a "
          "channel-wise skip concat with the matching encoder feature, decoder "
          "convolutions, and an output projection. No activations between stages. "
          "fp32; destination-passing style.");
}
