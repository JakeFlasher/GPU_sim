#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>

// Fused VAE residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 ->
// GroupNorm -> SiLU -> add(input). The fundamental building block of Sana's
// VAE encoder/decoder.
//
// Reference (problem_002_vae_conv3x3_groupnorm_silu_residual_fused):
//   num_groups = 32
//   residual = x
//   # First path: Conv3x3 -> GroupNorm -> SiLU
//   out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
//   out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
//   out = F.silu(out)
//   # Second path: Conv3x3 -> GroupNorm -> SiLU
//   out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
//   out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
//   out = F.silu(out)
//   # Residual connection
//   out = out + residual
//   return out
//
// Per definition.json (insertion order):
//   x            : [batch_size, channels, height, width]                 fp32
//   conv1_weight : [channels, channels, kernel_size, kernel_size]        fp32
//   norm1_weight : [channels]                                            fp32
//   norm1_bias   : [channels]                                            fp32
//   conv2_weight : [channels, channels, kernel_size, kernel_size]        fp32
//   norm2_weight : [channels]                                            fp32
//   norm2_bias   : [channels]                                            fp32
//   eps          : scalar                                                fp32 (-> C++ double)
//   output       : [batch_size, channels, height, width]                 fp32  (DPS, pre-alloc)
//
// Notes on op-for-op fidelity:
//   * num_groups is the fixed constant 32 from the reference run() (and every
//     workload row in the spec carries num_groups = 32). It is NOT a tensor
//     input, so it is baked in here.
//   * Both convolutions are bias-free (F.conv2d(..., bias=None)) with stride=1,
//     padding=1 -- a same-spatial-size 3x3 conv. The undefined Tensor
//     (default-constructed torch::Tensor{}) is passed as the bias argument so
//     ATen takes the no-bias path, exactly matching bias=None.
//   * group_norm normalises over (channels_per_group * H * W) within each of the
//     32 groups, then applies the per-channel affine (weight, bias). eps is the
//     scalar run() argument (passed by the harness as a Python float -> double).
//   * SiLU via at::silu == x * sigmoid(x), matching F.silu.
//   * The residual is the ORIGINAL x input (residual = x is captured before any
//     conv), added to the post-second-SiLU activation.
//   * All math stays in fp32; ATen's conv2d / group_norm accumulate internally
//     and emit fp32 outputs, matching the reference's fp32 cast profile (no
//     intermediate dtype change in run()).
//
// Implementation: pure ATen / cuDNN dispatch -- NO custom CUDA kernels. Output
// is written in place into the pre-allocated destination (destination-passing
// style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& conv1_weight,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& conv2_weight,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    double eps,
    torch::Tensor output
) {
    check_fp32_cuda(x, "x");
    check_fp32_cuda(conv1_weight, "conv1_weight");
    check_fp32_cuda(norm1_weight, "norm1_weight");
    check_fp32_cuda(norm1_bias, "norm1_bias");
    check_fp32_cuda(conv2_weight, "conv2_weight");
    check_fp32_cuda(norm2_weight, "norm2_weight");
    check_fp32_cuda(norm2_bias, "norm2_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(x.dim() == 4, "x must be 4D [N, C, H, W], got ", x.dim(), "D");
    TORCH_CHECK(conv1_weight.dim() == 4,
                "conv1_weight must be 4D [C, C, k, k], got ", conv1_weight.dim(), "D");
    TORCH_CHECK(conv2_weight.dim() == 4,
                "conv2_weight must be 4D [C, C, k, k], got ", conv2_weight.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    // num_groups is the reference constant 32 (every workload row uses 32).
    constexpr int64_t kNumGroups = 32;

    // Bias-free conv (F.conv2d(..., bias=None)). A default-constructed Tensor is
    // undefined -> ATen takes the no-bias path.
    const torch::Tensor no_bias;

    // stride=1, padding=1 for a same-size 3x3 conv. Explicitly-typed int64
    // vectors so the IntArrayRef conv2d overload is the unique match (a
    // braced-init {1, 1} can be ambiguous against the string_view padding
    // overload).
    const std::vector<int64_t> stride{1, 1};
    const std::vector<int64_t> padding{1, 1};

    // Residual is the ORIGINAL x captured before any conv.
    const torch::Tensor& residual = x;

    // First path: Conv3x3 -> GroupNorm -> SiLU.
    torch::Tensor out = at::conv2d(x, conv1_weight, no_bias, stride, padding);
    out = at::group_norm(out, kNumGroups, norm1_weight, norm1_bias, eps);
    out = at::silu(out);

    // Second path: Conv3x3 -> GroupNorm -> SiLU.
    out = at::conv2d(out, conv2_weight, no_bias, stride, padding);
    out = at::group_norm(out, kNumGroups, norm2_weight, norm2_bias, eps);
    out = at::silu(out);

    // Residual connection.
    torch::Tensor result = out + residual;

    TORCH_CHECK(result.sizes() == output.sizes(),
                "fused block output shape ", result.sizes(),
                " does not match pre-allocated output ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Fused VAE residual block (Conv3x3 -> GroupNorm(32) -> SiLU x2 + "
          "residual) via ATen (cuDNN conv2d / group_norm / silu; no custom CUDA "
          "kernels); fp32; destination-passing style.");
}
