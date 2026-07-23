#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <cmath>
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kBFloat16,
        name, " must have bfloat16 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Host-side inverse normal CDF (Abramowitz-Stegun rational
// approximation, formula 26.2.23). Mirrors the reference's
// _ndtri implementation exactly so that the kernel-side
// z_multiplier matches the PyTorch reference's per-row cutoff
// derivation.
static double host_ndtri(double p) {
    constexpr double a1 = -3.969683028665376e+01;
    constexpr double a2 =  2.209460984245205e+02;
    constexpr double a3 = -2.759285104469687e+02;
    constexpr double a4 =  1.383577518672690e+02;
    constexpr double a5 = -3.066479806614716e+01;
    constexpr double a6 =  2.506628277459239e+00;
    constexpr double b1 = -5.447609879822406e+01;
    constexpr double b2 =  1.615858368580409e+02;
    constexpr double b3 = -1.556989798598866e+02;
    constexpr double b4 =  6.680131188771972e+01;
    constexpr double b5 = -1.328068155288572e+01;
    constexpr double c1 = -7.784894002430293e-03;
    constexpr double c2 = -3.223964580411365e-01;
    constexpr double c3 = -2.400758277161838e+00;
    constexpr double c4 = -2.549732539343734e+00;
    constexpr double c5 =  4.374664141464968e+00;
    constexpr double c6 =  2.938163982698783e+00;
    constexpr double d1 =  7.784695709041462e-03;
    constexpr double d2 =  3.224671290700398e-01;
    constexpr double d3 =  2.445134137142996e+00;
    constexpr double d4 =  3.754408661907416e+00;
    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    } else if (p <= p_high) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
               (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    }
}

// Destination-passing-style entry. Per definition.json:
//   inputs          : [batch_size, seq_len, intermediate_size] bf16
//   target_sparsity : scalar fp32 in [0, 1]; 0.0 returns inputs unchanged
//   output          : [batch_size, seq_len, intermediate_size] bf16 (DPS)
//
// intermediate_size is workload-variable (4096, 8192, 12288, 16384).
void run(
    const torch::Tensor& inputs,
    double target_sparsity,
    torch::Tensor output
) {
    TORCH_CHECK(inputs.dim() == 3, "inputs must be 3D [B, S, I]");
    TORCH_CHECK(output.dim() == 3, "output must be 3D [B, S, I]");
    TORCH_CHECK(output.sizes() == inputs.sizes(),
                "output shape must match inputs");
    TORCH_CHECK(inputs.size(2) % 2 == 0,
                "intermediate_size must be even for BF16x2 vector access, got ",
                inputs.size(2));

    check_bf16_contiguous_cuda(inputs, "inputs");
    check_bf16_contiguous_cuda(output, "output");

    // Early return: target_sparsity == 0.0 means no sparsity; the
    // reference returns inputs unchanged. DPS contract requires the
    // output tensor to be written, so copy inputs -> output in-place.
    if (target_sparsity == 0.0) {
        output.copy_(inputs);
        return;
    }

    // Host-side ndtri evaluation (one-shot per kernel invocation).
    const float z_multiplier =
        static_cast<float>(host_ndtri(target_sparsity));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_053_gaussian_topk_sparse_activation_launcher(
        inputs, z_multiplier, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "BF16 Gaussian top-k sparse activation: per-row mean/std + "
        "host-side ndtri(target_sparsity) + ReLU(x - cutoff). Single "
        "DPS output. intermediate_size workload-variable.");
}
