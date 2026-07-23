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
//   ids        : [seq_len, n_axes=3] fp32  (per-axis positions)
//   theta      : scalar fp32  (RoPE base frequency)
//   freqs_cos  : [seq_len, total_dim=128] fp32 (DPS)
//   freqs_sin  : [seq_len, total_dim=128] fp32 (DPS)
//
// n_axes, axes_dim = [16, 56, 56], total_dim = 128 are all pinned by
// the problem definition. The single launcher invocation populates
// both outputs in parallel.
void run(
    const torch::Tensor& ids,
    double theta,
    torch::Tensor freqs_cos,
    torch::Tensor freqs_sin
) {
    TORCH_CHECK(ids.dim() == 2, "ids must be 2D [seq_len, n_axes]");
    TORCH_CHECK(freqs_cos.dim() == 2, "freqs_cos must be 2D");
    TORCH_CHECK(freqs_sin.dim() == 2, "freqs_sin must be 2D");

    constexpr int64_t kNumAxes = 3;
    constexpr int64_t kTotalDim = 128;
    TORCH_CHECK(ids.size(1) == kNumAxes,
                "ids must have ", kNumAxes,
                " axes, got ", ids.size(1));
    TORCH_CHECK(freqs_cos.size(1) == kTotalDim,
                "freqs_cos last dim must be ", kTotalDim,
                ", got ", freqs_cos.size(1));
    TORCH_CHECK(freqs_sin.size(1) == kTotalDim,
                "freqs_sin last dim must be ", kTotalDim,
                ", got ", freqs_sin.size(1));
    TORCH_CHECK(freqs_cos.size(0) == ids.size(0),
                "freqs_cos seq_len must match ids seq_len");
    TORCH_CHECK(freqs_sin.size(0) == ids.size(0),
                "freqs_sin seq_len must match ids seq_len");

    check_fp32_contiguous_cuda(ids, "ids");
    check_fp32_contiguous_cuda(freqs_cos, "freqs_cos");
    check_fp32_contiguous_cuda(freqs_sin, "freqs_sin");

    const float theta_f32 = static_cast<float>(theta);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_034_flux_multi_axis_rope_freq_launcher(
        ids, theta_f32, freqs_cos, freqs_sin, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 multi-axis RoPE frequency computation: generates cos/sin "
        "tables for FLUX 3D position encoding (axes_dim=[16,56,56], "
        "total_dim=128 pinned). Two distinct DPS outputs filled by "
        "one kernel launch.");
}
