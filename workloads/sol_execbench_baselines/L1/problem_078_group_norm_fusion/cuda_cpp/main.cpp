#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat32,
        name, " must have float32 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. Per definition.json:
//   x       : [batch_size, num_channels, height, width] fp32
//   weight  : [num_channels] fp32   (per-channel scale, gamma)
//   bias    : [num_channels] fp32   (per-channel shift, beta)
//   eps     : scalar fp32
//   output  : [batch_size, num_channels, height, width] fp32 (DPS)
//
// num_groups = 32 is pinned. num_channels must be divisible by 32
// (320 / 640 / 1280 -> 10 / 20 / 40 channels per group).
void run(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps,
    torch::Tensor output
) {
    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D [C]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [C]");
    TORCH_CHECK(output.dim() == 4, "output must be 4D [B, C, H, W]");

    constexpr int64_t kNumGroups = 32;
    const int64_t num_channels = x.size(1);
    TORCH_CHECK(num_channels % kNumGroups == 0,
                "num_channels must be divisible by num_groups=32, got ",
                num_channels);
    TORCH_CHECK(weight.size(0) == num_channels,
                "weight must have ", num_channels, " elements");
    TORCH_CHECK(bias.size(0) == num_channels,
                "bias must have ", num_channels, " elements");
    TORCH_CHECK(output.sizes() == x.sizes(),
                "output shape must match x");

    check_fp32_contiguous_cuda(x, "x");
    check_fp32_contiguous_cuda(weight, "weight");
    check_fp32_contiguous_cuda(bias, "bias");
    check_fp32_contiguous_cuda(output, "output");

    const float eps_f32 = static_cast<float>(eps);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_078_group_norm_fusion_launcher(
        x, weight, bias, output, eps_f32, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 GroupNorm: normalizes across (channels_per_group, H, W) "
        "within each of num_groups=32 channel groups; applies "
        "per-channel weight/bias affine transform. num_groups=32 pinned.");
}
