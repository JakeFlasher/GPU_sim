#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>

// Backward pass for Adaptive Instance Normalization 2D.
//
// Reference (problem_039_adaptive_instance_normalization_2d_backward):
//   Per-channel (instance) normalization backward across spatial dims.
//   The forward normalization is y = gamma * (x - mean) / std + beta, where
//   mean/std are per (N, C) over the spatial (H, W) plane. The reference
//   supplies the forward `std` directly (NOT var); every formula below uses
//   `std` and `torch.pow(std, -3)` exactly as the reference writes it -- no
//   var/eps reconstruction is performed.
//
//   N, C, H, W = x.shape
//   spatial_size = H * W
//
//   x_centered   = x - mean
//   x_normalized = x_centered / std
//
//   grad_bias   = grad_output.sum(dim=(0, 2, 3))                          # (C,)
//   grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))         # (C,)
//
//   weight_reshaped    = weight.view(1, C, 1, 1)
//   grad_output_scaled = grad_output * weight_reshaped
//
//   grad_var  = (grad_output_scaled * x_centered).sum(dim=(2,3), keepdim=True)
//                 * (-0.5) * torch.pow(std, -3)                           # (N,C,1,1)
//
//   grad_mean = (grad_output_scaled / (-std)).sum(dim=(2,3), keepdim=True)
//   grad_mean = grad_mean
//                 + grad_var * (-2.0 * x_centered).sum(dim=(2,3), keepdim=True)
//                   / spatial_size                                        # (N,C,1,1)
//
//   grad_input = grad_output_scaled / std
//   grad_input = grad_input + grad_var * 2.0 * x_centered / spatial_size
//   grad_input = grad_input + grad_mean / spatial_size                   # (N,C,H,W)
//
// Per definition.json (insertion order):
//   grad_output : [N, C, H, W]   fp32
//   x           : [N, C, H, W]   fp32
//   weight      : [C]            fp32
//   mean        : [N, C, 1, 1]   fp32
//   std         : [N, C, 1, 1]   fp32
//   grad_input  : [N, C, H, W]   fp32  (DPS, pre-alloc)
//   grad_weight : [C]            fp32  (DPS, pre-alloc)
//   grad_bias   : [C]            fp32  (DPS, pre-alloc)
//
// Implementation: pure ATen tensor algebra -- NO custom CUDA kernels and NO
// autograd. The expression is transcribed op-for-op from the reference run()
// so the fp32 precision profile and the exact reduction dims / keepdim
// semantics match. Outputs are written in place into the pre-allocated
// destinations (destination-passing style). The reduction broadcast shapes
// (grad_var / grad_mean are (N,C,1,1) and broadcast against the (N,C,H,W)
// tensors) are preserved exactly as the reference relies on.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& mean,
    const torch::Tensor& std,
    torch::Tensor grad_input,
    torch::Tensor grad_weight,
    torch::Tensor grad_bias
) {
    check_fp32_cuda(grad_output, "grad_output");
    check_fp32_cuda(x, "x");
    check_fp32_cuda(weight, "weight");
    check_fp32_cuda(mean, "mean");
    check_fp32_cuda(std, "std");
    check_fp32_cuda(grad_input, "grad_input");
    check_fp32_cuda(grad_weight, "grad_weight");
    check_fp32_cuda(grad_bias, "grad_bias");

    TORCH_CHECK(x.dim() == 4, "x must be 4D [N, C, H, W], got ", x.dim(), "D");
    TORCH_CHECK(grad_output.dim() == 4,
                "grad_output must be 4D [N, C, H, W], got ", grad_output.dim(), "D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D [C], got ", weight.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const double spatial_size = static_cast<double>(H * W);

    // x_centered = x - mean ; x_normalized = x_centered / std
    torch::Tensor x_centered = x - mean;
    torch::Tensor x_normalized = x_centered / std;

    // grad_bias = grad_output.sum(dim=(0, 2, 3))
    torch::Tensor gb = grad_output.sum(/*dim=*/{0, 2, 3});

    // grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))
    torch::Tensor gw = (grad_output * x_normalized).sum(/*dim=*/{0, 2, 3});

    // grad_output_scaled = grad_output * weight.view(1, C, 1, 1)
    torch::Tensor weight_reshaped = weight.view({1, C, 1, 1});
    torch::Tensor grad_output_scaled = grad_output * weight_reshaped;

    // grad_var = (grad_output_scaled * x_centered).sum(dim=(2,3), keepdim=True)
    //              * (-0.5) * torch.pow(std, -3)
    torch::Tensor grad_var =
        (grad_output_scaled * x_centered).sum(/*dim=*/{2, 3}, /*keepdim=*/true)
        * (-0.5) * torch::pow(std, -3);

    // grad_mean = (grad_output_scaled / (-std)).sum(dim=(2,3), keepdim=True)
    //             + grad_var * (-2.0 * x_centered).sum(dim=(2,3), keepdim=True)
    //               / spatial_size
    torch::Tensor grad_mean =
        (grad_output_scaled / (-std)).sum(/*dim=*/{2, 3}, /*keepdim=*/true);
    grad_mean = grad_mean
        + grad_var * (-2.0 * x_centered).sum(/*dim=*/{2, 3}, /*keepdim=*/true) / spatial_size;

    // grad_input = grad_output_scaled / std
    //            + grad_var * 2.0 * x_centered / spatial_size
    //            + grad_mean / spatial_size
    torch::Tensor gi = grad_output_scaled / std;
    gi = gi + grad_var * 2.0 * x_centered / spatial_size;
    gi = gi + grad_mean / spatial_size;

    TORCH_CHECK(gi.sizes() == grad_input.sizes(),
                "grad_input shape ", gi.sizes(),
                " does not match pre-allocated output ", grad_input.sizes());
    TORCH_CHECK(gw.sizes() == grad_weight.sizes(),
                "grad_weight shape ", gw.sizes(),
                " does not match pre-allocated output ", grad_weight.sizes());
    TORCH_CHECK(gb.sizes() == grad_bias.sizes(),
                "grad_bias shape ", gb.sizes(),
                " does not match pre-allocated output ", grad_bias.sizes());

    grad_input.copy_(gi);
    grad_weight.copy_(gw);
    grad_bias.copy_(gb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Adaptive InstanceNorm2d backward via ATen tensor algebra (no autograd, "
          "no custom CUDA kernels); fp32; destination-passing style.");
}
