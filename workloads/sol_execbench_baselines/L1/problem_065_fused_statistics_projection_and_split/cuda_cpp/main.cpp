#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <vector>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the fused Conv1D projection +
// channel-wise split (OpenVoice TTS encoder statistics head).
// Per definition.json:
//   x       : [B, in_channels=192, T]                          fp32
//   weight  : [out_channels_2=384, in_channels=192, kernel_size=1] fp32
//   bias    : [out_channels_2=384]                             fp32
//   mean    : [B, out_channels=192, T]                         fp32 (DPS)
//   logs    : [B, out_channels=192, T]                         fp32 (DPS)
//
// Algorithm (mirroring the PyTorch reference):
//   stats        = F.conv1d(x, weight, bias, padding=0)
//                  # [B, 384, T]
//   mean, logs   = torch.split(stats, 192, dim=1)
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   in_channels     = 192
//   out_channels    = 192
//   out_channels_2  = 384
//   kernel_size     = 1
//
// Implementation: pure ATen/cuBLAS dispatch — NO custom CUDA
// kernels. For kernel_size=1 fp32, at::conv1d reduces to a
// cublasSgemm at (M=B*T, K=192, N=384) which matches the
// reference's torch.F.conv1d dispatch exactly.
void run(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mean,
    torch::Tensor logs
) {
    check_fp32_contiguous_cuda(x, "x");
    check_fp32_contiguous_cuda(weight, "weight");
    check_fp32_contiguous_cuda(bias, "bias");
    check_fp32_contiguous_cuda(mean, "mean");
    check_fp32_contiguous_cuda(logs, "logs");

    constexpr int64_t kInChannels = 192;
    constexpr int64_t kOutChannels = 192;
    constexpr int64_t kOutChannels2 = 2 * kOutChannels;  // 384
    constexpr int64_t kKernelSize = 1;

    TORCH_CHECK(x.dim() == 3,
                "x must be 3D [B, in_channels, T]");
    TORCH_CHECK(weight.dim() == 3,
                "weight must be 3D [out_channels_2, in_channels, "
                "kernel_size]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [out_channels_2]");
    TORCH_CHECK(mean.dim() == 3, "mean must be 3D [B, out_channels, T]");
    TORCH_CHECK(logs.dim() == 3, "logs must be 3D [B, out_channels, T]");

    TORCH_CHECK(x.size(1) == kInChannels,
                "x.size(1) must equal in_channels=192, got ",
                x.size(1));
    TORCH_CHECK(weight.size(0) == kOutChannels2,
                "weight.size(0) must equal out_channels_2=384, got ",
                weight.size(0));
    TORCH_CHECK(weight.size(1) == kInChannels,
                "weight.size(1) must equal in_channels=192, got ",
                weight.size(1));
    TORCH_CHECK(weight.size(2) == kKernelSize,
                "weight.size(2) must equal kernel_size=1, got ",
                weight.size(2));
    TORCH_CHECK(bias.size(0) == kOutChannels2,
                "bias.size(0) must equal out_channels_2=384, got ",
                bias.size(0));
    TORCH_CHECK(mean.size(0) == x.size(0),
                "mean.size(0) must equal x.size(0)");
    TORCH_CHECK(mean.size(1) == kOutChannels,
                "mean.size(1) must equal out_channels=192, got ",
                mean.size(1));
    TORCH_CHECK(mean.size(2) == x.size(2),
                "mean.size(2) must equal x.size(2)");
    TORCH_CHECK(logs.size(0) == x.size(0),
                "logs.size(0) must equal x.size(0)");
    TORCH_CHECK(logs.size(1) == kOutChannels,
                "logs.size(1) must equal out_channels=192, got ",
                logs.size(1));
    TORCH_CHECK(logs.size(2) == x.size(2),
                "logs.size(2) must equal x.size(2)");

    // (1) at::conv1d projects to 2x channels. For kernel_size=1
    // fp32 this dispatches to a cublasSgemm-equivalent kernel via
    // cuDNN; the result matches the reference's F.conv1d
    // bit-for-bit. Use IntArrayRef literals for stride / padding /
    // dilation to disambiguate from the string-padding overload.
    const std::vector<int64_t> stride_vec = {1};
    const std::vector<int64_t> padding_vec = {0};
    const std::vector<int64_t> dilation_vec = {1};
    auto stats = at::conv1d(
        x, weight, bias,
        at::IntArrayRef(stride_vec),
        at::IntArrayRef(padding_vec),
        at::IntArrayRef(dilation_vec),
        /*groups=*/1);

    // (2)(3) Channel-wise split into the two pre-allocated DPS
    // outputs.
    mean.copy_(stats.narrow(1, 0, kOutChannels));
    logs.copy_(stats.narrow(1, kOutChannels, kOutChannels));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 fused Conv1D projection to 2x channels + channel-wise "
        "split into mean and logs (OpenVoice TTS encoder statistics "
        "head). Pure ATen/cuBLAS dispatch — at::conv1d for the "
        "projection (kernel_size=1 fp32 -> cublasSgemm equivalent) "
        "followed by two narrow().copy_() calls into pre-allocated "
        "DPS outputs. in_channels=192, out_channels=192, "
        "out_channels_2=384, kernel_size=1 pinned. Destination-"
        "passing style.");
}
