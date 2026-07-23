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

// Destination-passing-style entry. The upstream definition.json
// declares:
//   u                    : [batch, inner_width=768, seq_len] fp32
//   short_filter_weight  : [inner_width, 1, short_filter_order=3] fp32
//   short_filter_bias    : [inner_width] fp32
//   v_gated              : [batch, d_model=256, seq_len] fp32 (DPS)
//   x0                   : [batch, d_model=256, seq_len] fp32 (DPS)
//   x1                   : [batch, d_model=256, seq_len] fp32 (DPS)
//
// Hyena order = 2; inner_width = d_model * (order + 1) = 768.
void run(
    const torch::Tensor& u,
    const torch::Tensor& short_filter_weight,
    const torch::Tensor& short_filter_bias,
    torch::Tensor v_gated,
    torch::Tensor x0,
    torch::Tensor x1) {

    TORCH_CHECK(u.dim() == 3, "u must be 3D [batch, inner, seq]; got ",
                u.dim(), "D");
    const int64_t batch = u.size(0);
    const int64_t inner = u.size(1);
    const int64_t seq_len = u.size(2);
    TORCH_CHECK(inner == 768, "inner_width must be 768, got ", inner);
    TORCH_CHECK(
        short_filter_weight.dim() == 3
            && short_filter_weight.size(0) == 768
            && short_filter_weight.size(1) == 1
            && short_filter_weight.size(2) == 3,
        "short_filter_weight must be [768, 1, 3]");
    TORCH_CHECK(
        short_filter_bias.dim() == 1 && short_filter_bias.size(0) == 768,
        "short_filter_bias must be [768]");
    TORCH_CHECK(
        v_gated.sizes() == torch::IntArrayRef({batch, 256, seq_len})
            && x0.sizes() == torch::IntArrayRef({batch, 256, seq_len})
            && x1.sizes() == torch::IntArrayRef({batch, 256, seq_len}),
        "v_gated / x0 / x1 must be [batch, 256, seq_len]");

    check_fp32_contiguous(u, "u");
    check_fp32_contiguous(short_filter_weight, "short_filter_weight");
    check_fp32_contiguous(short_filter_bias, "short_filter_bias");
    check_fp32_contiguous(v_gated, "v_gated");
    check_fp32_contiguous(x0, "x0");
    check_fp32_contiguous(x1, "x1");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_006_hyena_depthwise_conv1d_split_gate_launcher(
        u,
        short_filter_weight,
        short_filter_bias,
        v_gated,
        x0,
        x1,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Hyena depthwise 1D conv (kernel_size=3, padding=2 then "
        "truncate to seq_len) + channel split + element-wise gate. "
        "FP32 throughout per definition.json; destination-passing "
        "style.");
}
