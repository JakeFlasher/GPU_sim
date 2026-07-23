#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_fp32_contiguous(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat32,
        name, " must have float32 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. Definition.json declares:
//   x                : [batch, in_channels=3, height, width]      fp32
//   conv_in_weight   : [hidden_channels=32, in_channels=3, 3, 3]  fp32
//   conv_in_bias     : [hidden_channels=32]                       fp32
//   conv_out_weight  : [out_channels=3, hidden_channels=32, 3, 3] fp32
//   conv_out_bias    : [out_channels=3]                           fp32
//   output           : [batch, out_channels=3, height, width]     fp32 (DPS)
//
// Block math: out = conv_out(conv_in(x)) + x. kernel_size=3,
// padding=1 on both convs.
void run(
    const torch::Tensor& x,
    const torch::Tensor& conv_in_weight,
    const torch::Tensor& conv_in_bias,
    const torch::Tensor& conv_out_weight,
    const torch::Tensor& conv_out_bias,
    torch::Tensor output) {

    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W]; got ", x.dim(), "D");
    TORCH_CHECK(x.size(1) == 3, "in_channels must be 3, got ", x.size(1));
    TORCH_CHECK(
        conv_in_weight.dim() == 4
            && conv_in_weight.size(0) == 32
            && conv_in_weight.size(1) == 3
            && conv_in_weight.size(2) == 3
            && conv_in_weight.size(3) == 3,
        "conv_in_weight must be [32, 3, 3, 3]");
    TORCH_CHECK(
        conv_in_bias.dim() == 1 && conv_in_bias.size(0) == 32,
        "conv_in_bias must be [32]");
    TORCH_CHECK(
        conv_out_weight.dim() == 4
            && conv_out_weight.size(0) == 3
            && conv_out_weight.size(1) == 32
            && conv_out_weight.size(2) == 3
            && conv_out_weight.size(3) == 3,
        "conv_out_weight must be [3, 32, 3, 3]");
    TORCH_CHECK(
        conv_out_bias.dim() == 1 && conv_out_bias.size(0) == 3,
        "conv_out_bias must be [3]");
    TORCH_CHECK(
        output.sizes() == x.sizes(),
        "output must match x shape [B, 3, H, W]");

    check_fp32_contiguous(x, "x");
    check_fp32_contiguous(conv_in_weight, "conv_in_weight");
    check_fp32_contiguous(conv_in_bias, "conv_in_bias");
    check_fp32_contiguous(conv_out_weight, "conv_out_weight");
    check_fp32_contiguous(conv_out_bias, "conv_out_bias");
    check_fp32_contiguous(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_040_conv2d_residual_block_launcher(
        x,
        conv_in_weight,
        conv_in_bias,
        conv_out_weight,
        conv_out_bias,
        output,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "conv2d residual block (3->32->3 channels, kernel_size=3, "
        "padding=1) with element-wise residual add: "
        "out = conv_out(conv_in(x)) + x. FP32 throughout; "
        "destination-passing style.");
}
