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
//   x                   : [batch, in_channels=320, H, W]               fp32
//   time_emb            : [batch, time_emb_channels=1280]              fp32
//   norm1_weight        : [in_channels=320]                             fp32
//   norm1_bias          : [in_channels=320]                             fp32
//   conv1_weight        : [out_channels=320, in_channels=320, 3, 3]    fp32
//   conv1_bias          : [out_channels=320]                            fp32
//   time_emb_proj_weight: [out_channels=320, time_emb_channels=1280]   fp32
//   time_emb_proj_bias  : [out_channels=320]                            fp32
//   norm2_weight        : [out_channels=320]                            fp32
//   norm2_bias          : [out_channels=320]                            fp32
//   conv2_weight        : [out_channels=320, out_channels=320, 3, 3]   fp32
//   conv2_bias          : [out_channels=320]                            fp32
//   norm_eps            : scalar                                        fp32
//   output              : [batch, out_channels=320, H, W]        (DPS) fp32
void run(
    const torch::Tensor& x,
    const torch::Tensor& time_emb,
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
    double norm_eps,
    torch::Tensor output) {

    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W]");
    TORCH_CHECK(x.size(1) == 320, "in_channels must be 320");
    TORCH_CHECK(
        conv1_weight.dim() == 4
            && conv1_weight.size(0) == 320
            && conv1_weight.size(1) == 320
            && conv1_weight.size(2) == 3
            && conv1_weight.size(3) == 3,
        "conv1_weight must be [320, 320, 3, 3]");
    TORCH_CHECK(
        conv2_weight.dim() == 4
            && conv2_weight.size(0) == 320
            && conv2_weight.size(1) == 320
            && conv2_weight.size(2) == 3
            && conv2_weight.size(3) == 3,
        "conv2_weight must be [320, 320, 3, 3]");
    TORCH_CHECK(
        time_emb.dim() == 2 && time_emb.size(1) == 1280,
        "time_emb must be [B, 1280]");
    TORCH_CHECK(
        time_emb_proj_weight.dim() == 2
            && time_emb_proj_weight.size(0) == 320
            && time_emb_proj_weight.size(1) == 1280,
        "time_emb_proj_weight must be [320, 1280]");
    TORCH_CHECK(output.sizes() == x.sizes(), "output must match x shape");

    check_fp32_contiguous(x, "x");
    check_fp32_contiguous(time_emb, "time_emb");
    check_fp32_contiguous(norm1_weight, "norm1_weight");
    check_fp32_contiguous(norm1_bias, "norm1_bias");
    check_fp32_contiguous(conv1_weight, "conv1_weight");
    check_fp32_contiguous(conv1_bias, "conv1_bias");
    check_fp32_contiguous(time_emb_proj_weight, "time_emb_proj_weight");
    check_fp32_contiguous(time_emb_proj_bias, "time_emb_proj_bias");
    check_fp32_contiguous(norm2_weight, "norm2_weight");
    check_fp32_contiguous(norm2_bias, "norm2_bias");
    check_fp32_contiguous(conv2_weight, "conv2_weight");
    check_fp32_contiguous(conv2_bias, "conv2_bias");
    check_fp32_contiguous(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_066_resnet_block_with_time_embedding_launcher(
        x,
        time_emb,
        norm1_weight,
        norm1_bias,
        conv1_weight,
        conv1_bias,
        time_emb_proj_weight,
        time_emb_proj_bias,
        norm2_weight,
        norm2_bias,
        conv2_weight,
        conv2_bias,
        static_cast<float>(norm_eps),
        output,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Stable Diffusion ResNet block with time-embedding injection: "
        "GroupNorm + SiLU + conv1 + (silu(time_emb) @ proj.T + bias) "
        "broadcast-add + GroupNorm + SiLU + conv2 + residual add. "
        "FP32 throughout; destination-passing style.");
}
