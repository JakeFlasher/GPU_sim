#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <array>
#include <string>
#include <vector>

// Residual coupling block for a normalizing flow with affine coupling
// transforms. The channel axis is split in half; one half is transformed
// conditioned on the other, then the two halves are recombined. Supports a
// forward and a reverse pass.
//
// Reference (problem_057_residual_coupling_flow_block), op-for-op:
//   half_channels = x.shape[1] // 2
//   transforms    = [transform_0, transform_1, transform_2, transform_3]
//
//   apply_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b):
//     padding = conv0_w.shape[2] // 2          # kernel_size // 2
//     h = conv1d(x0, conv0_w, conv0_b, padding=padding)
//     h = relu(h)
//     h = conv1d(h,  conv1_w, conv1_b, padding=padding)
//     h = relu(h)
//     h = conv1d(h,  conv2_w, conv2_b, padding=padding)
//     return h
//
//   forward (reverse == False), transforms in order:
//     for each transform:
//       x0 = x[:, :half_channels, :]
//       x1 = x[:, half_channels:, :]
//       h  = apply_transform(x0, ...) * x_mask
//       x1 = x1 + h
//       x  = cat([x0, x1], dim=1)
//       x  = x * x_mask
//
//   reverse (reverse == True), transforms in reversed order:
//     for each transform (reversed):
//       x0 = x[:, :half_channels, :]
//       x1 = x[:, half_channels:, :]
//       h  = apply_transform(x0, ...) * x_mask
//       x1 = x1 - h
//       x  = cat([x0, x1], dim=1)
//       x  = x * x_mask
//
//   return x
//
// Per definition.json (insertion order):
//   x                       : [B, channels, time]                       fp32
//   x_mask                  : [B, 1, time]                              fp32
//   reverse                 : scalar                                    bool
//   transform_{i}_conv0_weight : [hidden_channels, half_channels, k]    fp32
//   transform_{i}_conv0_bias   : [hidden_channels]                      fp32
//   transform_{i}_conv1_weight : [hidden_channels, hidden_channels, k]  fp32
//   transform_{i}_conv1_bias   : [hidden_channels]                      fp32
//   transform_{i}_conv2_weight : [half_channels, hidden_channels, k]    fp32
//   transform_{i}_conv2_bias   : [half_channels]                        fp32
//     ... for i in {0, 1, 2, 3} ...
//   output                  : [B, channels, time]                       fp32 DPS
//
// Notes on fidelity:
//   * half_channels is read off x (channels // 2), not hard-coded.
//   * The conv padding is kernel_size // 2 taken from conv0's weight, matching
//     the reference's padding = conv0_w.shape[2] // 2. With an odd kernel and
//     stride 1 this same-length padding keeps the time axis at its input size,
//     so cat back onto x0 stays shape-consistent across all transforms.
//   * The transform body is conv -> relu -> conv -> relu -> conv (only the
//     first two convs are followed by ReLU; the third is the raw conditioner
//     output h).
//   * The mask multiply happens twice per transform: once on h before the
//     coupling add/sub, and once on the recombined x.
//   * The affine coupling adds h in the forward pass and subtracts h in the
//     reverse pass, and the reverse pass walks the transforms in reverse
//     order. x0 is never modified by a transform, so cat([x0, x1]) simply
//     reattaches the untouched conditioning half.
//   * reverse is a C++ bool scalar driving a host-side branch.
//   * All math stays in fp32 (fp32 in, fp32 out, no intermediate downcast).
//
// Implementation: pure ATen ops -- NO custom CUDA kernels. The output is
// written in place into the pre-allocated destination (destination-passing
// style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// conv0 -> ReLU -> conv1 -> ReLU -> conv2. Padding is kernel_size // 2 from
// conv0's weight (same-length, stride 1). Explicitly-typed int64 vectors keep
// the IntArrayRef conv1d overload the unique match.
static torch::Tensor apply_transform(
    const torch::Tensor& x0,
    const torch::Tensor& conv0_w, const torch::Tensor& conv0_b,
    const torch::Tensor& conv1_w, const torch::Tensor& conv1_b,
    const torch::Tensor& conv2_w, const torch::Tensor& conv2_b
) {
    const int64_t pad = conv0_w.size(2) / 2;
    const std::vector<int64_t> stride{1}, padding{pad}, dilation{1};

    auto h = at::conv1d(x0, conv0_w, conv0_b, stride, padding, dilation, /*groups=*/1);
    h = at::relu(h);
    h = at::conv1d(h, conv1_w, conv1_b, stride, padding, dilation, /*groups=*/1);
    h = at::relu(h);
    h = at::conv1d(h, conv2_w, conv2_b, stride, padding, dilation, /*groups=*/1);
    return h;
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& x_mask,
    bool reverse,
    const torch::Tensor& transform_0_conv0_weight,
    const torch::Tensor& transform_0_conv0_bias,
    const torch::Tensor& transform_0_conv1_weight,
    const torch::Tensor& transform_0_conv1_bias,
    const torch::Tensor& transform_0_conv2_weight,
    const torch::Tensor& transform_0_conv2_bias,
    const torch::Tensor& transform_1_conv0_weight,
    const torch::Tensor& transform_1_conv0_bias,
    const torch::Tensor& transform_1_conv1_weight,
    const torch::Tensor& transform_1_conv1_bias,
    const torch::Tensor& transform_1_conv2_weight,
    const torch::Tensor& transform_1_conv2_bias,
    const torch::Tensor& transform_2_conv0_weight,
    const torch::Tensor& transform_2_conv0_bias,
    const torch::Tensor& transform_2_conv1_weight,
    const torch::Tensor& transform_2_conv1_bias,
    const torch::Tensor& transform_2_conv2_weight,
    const torch::Tensor& transform_2_conv2_bias,
    const torch::Tensor& transform_3_conv0_weight,
    const torch::Tensor& transform_3_conv0_bias,
    const torch::Tensor& transform_3_conv1_weight,
    const torch::Tensor& transform_3_conv1_bias,
    const torch::Tensor& transform_3_conv2_weight,
    const torch::Tensor& transform_3_conv2_bias,
    torch::Tensor output
) {
    check_fp32_cuda(x, "x");
    check_fp32_cuda(x_mask, "x_mask");
    check_fp32_cuda(transform_0_conv0_weight, "transform_0_conv0_weight");
    check_fp32_cuda(transform_0_conv0_bias, "transform_0_conv0_bias");
    check_fp32_cuda(transform_0_conv1_weight, "transform_0_conv1_weight");
    check_fp32_cuda(transform_0_conv1_bias, "transform_0_conv1_bias");
    check_fp32_cuda(transform_0_conv2_weight, "transform_0_conv2_weight");
    check_fp32_cuda(transform_0_conv2_bias, "transform_0_conv2_bias");
    check_fp32_cuda(transform_1_conv0_weight, "transform_1_conv0_weight");
    check_fp32_cuda(transform_1_conv0_bias, "transform_1_conv0_bias");
    check_fp32_cuda(transform_1_conv1_weight, "transform_1_conv1_weight");
    check_fp32_cuda(transform_1_conv1_bias, "transform_1_conv1_bias");
    check_fp32_cuda(transform_1_conv2_weight, "transform_1_conv2_weight");
    check_fp32_cuda(transform_1_conv2_bias, "transform_1_conv2_bias");
    check_fp32_cuda(transform_2_conv0_weight, "transform_2_conv0_weight");
    check_fp32_cuda(transform_2_conv0_bias, "transform_2_conv0_bias");
    check_fp32_cuda(transform_2_conv1_weight, "transform_2_conv1_weight");
    check_fp32_cuda(transform_2_conv1_bias, "transform_2_conv1_bias");
    check_fp32_cuda(transform_2_conv2_weight, "transform_2_conv2_weight");
    check_fp32_cuda(transform_2_conv2_bias, "transform_2_conv2_bias");
    check_fp32_cuda(transform_3_conv0_weight, "transform_3_conv0_weight");
    check_fp32_cuda(transform_3_conv0_bias, "transform_3_conv0_bias");
    check_fp32_cuda(transform_3_conv1_weight, "transform_3_conv1_weight");
    check_fp32_cuda(transform_3_conv1_bias, "transform_3_conv1_bias");
    check_fp32_cuda(transform_3_conv2_weight, "transform_3_conv2_weight");
    check_fp32_cuda(transform_3_conv2_bias, "transform_3_conv2_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, channels, time], got ", x.dim(), "D");
    TORCH_CHECK(x_mask.dim() == 3, "x_mask must be 3D [B, 1, time], got ", x_mask.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    const int64_t half_channels = x.size(1) / 2;

    // Bundle the six conv tensors per transform in definition order so the
    // forward / reverse iteration can index them as a group of four.
    struct Transform {
        const torch::Tensor& conv0_w;
        const torch::Tensor& conv0_b;
        const torch::Tensor& conv1_w;
        const torch::Tensor& conv1_b;
        const torch::Tensor& conv2_w;
        const torch::Tensor& conv2_b;
    };
    const std::array<Transform, 4> transforms{{
        {transform_0_conv0_weight, transform_0_conv0_bias,
         transform_0_conv1_weight, transform_0_conv1_bias,
         transform_0_conv2_weight, transform_0_conv2_bias},
        {transform_1_conv0_weight, transform_1_conv0_bias,
         transform_1_conv1_weight, transform_1_conv1_bias,
         transform_1_conv2_weight, transform_1_conv2_bias},
        {transform_2_conv0_weight, transform_2_conv0_bias,
         transform_2_conv1_weight, transform_2_conv1_bias,
         transform_2_conv2_weight, transform_2_conv2_bias},
        {transform_3_conv0_weight, transform_3_conv0_bias,
         transform_3_conv1_weight, transform_3_conv1_bias,
         transform_3_conv2_weight, transform_3_conv2_bias},
    }};

    auto current = x;
    for (int step = 0; step < 4; ++step) {
        // Forward walks 0..3 adding h; reverse walks 3..0 subtracting h.
        const int idx = reverse ? (3 - step) : step;
        const Transform& t = transforms[idx];

        auto x0 = current.narrow(/*dim=*/1, /*start=*/0, half_channels);
        auto x1 = current.narrow(/*dim=*/1, /*start=*/half_channels,
                                 current.size(1) - half_channels);

        auto h = apply_transform(x0, t.conv0_w, t.conv0_b, t.conv1_w, t.conv1_b,
                                 t.conv2_w, t.conv2_b);
        h = h * x_mask;

        auto x1_coupled = reverse ? (x1 - h) : (x1 + h);

        current = at::cat({x0, x1_coupled}, /*dim=*/1);
        current = current * x_mask;
    }

    // --- Destination-passing-style write. ---
    TORCH_CHECK(current.sizes() == output.sizes(),
                "output shape ", current.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(current);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Residual coupling flow block via ATen (channel split, conv1d -> ReLU "
          "-> conv1d -> ReLU -> conv1d conditioner, masked affine coupling with a "
          "forward add / reverse subtract bool branch; no custom CUDA kernels); "
          "fp32; destination-passing style (output).");
}
