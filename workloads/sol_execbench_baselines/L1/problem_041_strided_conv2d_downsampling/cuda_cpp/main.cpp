#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Strided 2D convolution for spatial downsampling (NAFNet encoder path).
//
// Reference (problem_041_strided_conv2d_downsampling):
//     stride = 2
//     padding = 1
//     output = F.conv2d(x, weight, bias, stride=stride, padding=padding)
//
// Per definition.json (insertion order):
//   x      : [batch_size, in_channels, height, width]                 fp32
//   weight : [out_channels, in_channels, kernel_size, kernel_size]    fp32
//   bias   : [out_channels]                                           fp32
//   output : [batch_size, out_channels, out_height, out_width]        fp32  (DPS, pre-alloc)
//
// stride / padding are fixed constants in the reference run() (the workload
// table's stride/padding columns describe this fixed configuration), so they
// are baked in here rather than passed as inputs.
//
// Implementation: pure ATen / cuDNN dispatch -- NO custom CUDA kernels. The
// single at::conv2d call routes through cuDNN with the same fp32 precision
// profile the reference's F.conv2d uses. Output is written in place into the
// pre-allocated destination (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor output
) {
    check_fp32_cuda(x, "x");
    check_fp32_cuda(weight, "weight");
    check_fp32_cuda(bias, "bias");
    check_fp32_cuda(output, "output");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N, C, H, W], got ", x.dim(), "D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [Co, Ci, kh, kw], got ", weight.dim(), "D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [Co], got ", bias.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    constexpr int64_t kStride = 2;
    constexpr int64_t kPadding = 1;

    // at::conv2d(input, weight, bias, stride, padding) -- the same cuDNN
    // dispatch F.conv2d(x, weight, bias, stride=2, padding=1) issues.
    torch::Tensor result = at::conv2d(
        x, weight, bias,
        /*stride=*/{kStride, kStride},
        /*padding=*/{kPadding, kPadding});

    TORCH_CHECK(result.sizes() == output.sizes(),
                "conv2d output shape ", result.sizes(),
                " does not match pre-allocated output ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Strided (stride=2, pad=1) 2D conv downsampling via at::conv2d (cuDNN); "
          "fp32; destination-passing style.");
}
