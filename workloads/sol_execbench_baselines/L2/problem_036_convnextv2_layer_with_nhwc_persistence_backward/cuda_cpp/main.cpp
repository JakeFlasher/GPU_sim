#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Backward pass for the ConvNeXtV2 layer with NHWC persistence -- the analytic
// inverse of the forward block (residual add, drop path, depthwise 7x7
// convolution, channels-last LayerNorm, pointwise expansion, GELU, Global
// Response Normalization, pointwise projection).
//
// Reference (problem_036_convnextv2_layer_with_nhwc_persistence_backward): the
// gradient chain written out by hand as explicit tensor algebra (no
// torch.autograd). Because the forward saved every intermediate, no forward
// recompute is needed -- the gradient walks straight back through each stage
// using the persisted activations. This port reproduces that op-for-op.
//
// Precision profile (matched exactly): every tensor is float32 and the
// reference stays in float32 throughout (the GELU tanh-approximation
// derivative, the GRN norm/division algebra, and the LayerNorm
// variance/mean reductions are all native float32). There are no cast points
// to track.
//
// Per definition.json (insertion order):
//   inputs (all float32 except the two trailing scalars):
//     grad_output      : [B, C, H, W]            (NCHW)
//     residual         : [B, C, H, W]
//     x_dwconv         : [B, C, H, W]
//     x_nhwc           : [B, H, W, C]
//     mean             : [B, H, W, 1]
//     var              : [B, H, W, 1]
//     x_normalized     : [B, H, W, C]
//     x_ln             : [B, H, W, C]
//     x_expanded       : [B, H, W, C4]
//     x_gelu           : [B, H, W, C4]
//     global_features  : [B, 1, 1, C4]
//     gf_mean          : [B, 1, 1, 1]
//     norm_features    : [B, 1, 1, C4]
//     x_grn_scaled     : [B, H, W, C4]
//     x_grn            : [B, H, W, C4]
//     dwconv_weight    : [C, 1, 7, 7]
//     layernorm_weight : [C]
//     pwconv1_weight   : [C4, C]
//     grn_weight       : [1, 1, 1, C4]
//     pwconv2_weight   : [C, C4]
//     drop_mask        : [B, 1, 1, 1]
//     drop_path_prob   : scalar float32 -> double
//     eps              : scalar float32 -> double
//   outputs (all float32, DPS pre-alloc, written in place):
//     grad_x               : [B, C, H, W]
//     grad_dwconv_weight   : [C, 1, 7, 7]
//     grad_dwconv_bias     : [C]
//     grad_layernorm_weight: [C]
//     grad_layernorm_bias  : [C]
//     grad_pwconv1_weight  : [C4, C]
//     grad_pwconv1_bias    : [C4]
//     grad_grn_weight      : [1, 1, 1, C4]
//     grad_grn_bias        : [1, 1, 1, C4]
//     grad_pwconv2_weight  : [C, C4]
//     grad_pwconv2_bias    : [C]
//
// Implementation: pure ATen / cuDNN / cuBLAS dispatch -- no custom CUDA
// kernels. at::linear routes the pointwise-layer grads through cuBLAS,
// at::conv_transpose2d gives the depthwise input gradient, and the depthwise
// weight gradient uses the grouped-convolution cross-correlation
// reformulation (one grouped at::conv2d of the residual against grad_x_dwconv
// as the kernel, groups == B*C) instead of the reference's per-channel host
// loop. Every reduction orientation, the GRN backward, the GELU
// tanh-approximation derivative, and the LayerNorm backward algebra match the
// reference. C is the per-problem constant 128.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& residual,
    const torch::Tensor& x_dwconv,
    const torch::Tensor& x_nhwc,
    const torch::Tensor& mean,
    const torch::Tensor& var,
    const torch::Tensor& x_normalized,
    const torch::Tensor& x_ln,
    const torch::Tensor& x_expanded,
    const torch::Tensor& x_gelu,
    const torch::Tensor& global_features,
    const torch::Tensor& gf_mean,
    const torch::Tensor& norm_features,
    const torch::Tensor& x_grn_scaled,
    const torch::Tensor& x_grn,
    const torch::Tensor& dwconv_weight,
    const torch::Tensor& layernorm_weight,
    const torch::Tensor& pwconv1_weight,
    const torch::Tensor& grn_weight,
    const torch::Tensor& pwconv2_weight,
    const torch::Tensor& drop_mask,
    double drop_path_prob,
    double eps,
    torch::Tensor grad_x,
    torch::Tensor grad_dwconv_weight,
    torch::Tensor grad_dwconv_bias,
    torch::Tensor grad_layernorm_weight,
    torch::Tensor grad_layernorm_bias,
    torch::Tensor grad_pwconv1_weight,
    torch::Tensor grad_pwconv1_bias,
    torch::Tensor grad_grn_weight,
    torch::Tensor grad_grn_bias,
    torch::Tensor grad_pwconv2_weight,
    torch::Tensor grad_pwconv2_bias
) {
    check_fp32_cuda(grad_output, "grad_output");
    check_fp32_cuda(residual, "residual");
    check_fp32_cuda(x_dwconv, "x_dwconv");
    check_fp32_cuda(x_nhwc, "x_nhwc");
    check_fp32_cuda(mean, "mean");
    check_fp32_cuda(var, "var");
    check_fp32_cuda(x_normalized, "x_normalized");
    check_fp32_cuda(x_ln, "x_ln");
    check_fp32_cuda(x_expanded, "x_expanded");
    check_fp32_cuda(x_gelu, "x_gelu");
    check_fp32_cuda(global_features, "global_features");
    check_fp32_cuda(gf_mean, "gf_mean");
    check_fp32_cuda(norm_features, "norm_features");
    check_fp32_cuda(x_grn_scaled, "x_grn_scaled");
    check_fp32_cuda(x_grn, "x_grn");
    check_fp32_cuda(dwconv_weight, "dwconv_weight");
    check_fp32_cuda(layernorm_weight, "layernorm_weight");
    check_fp32_cuda(pwconv1_weight, "pwconv1_weight");
    check_fp32_cuda(grn_weight, "grn_weight");
    check_fp32_cuda(pwconv2_weight, "pwconv2_weight");
    check_fp32_cuda(drop_mask, "drop_mask");

    TORCH_CHECK(grad_output.dim() == 4,
                "grad_output must be 4D [B, C, H, W], got ",
                grad_output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    const int64_t B = grad_output.size(0);
    const int64_t C = grad_output.size(1);
    const int64_t H = grad_output.size(2);
    const int64_t W = grad_output.size(3);

    // Gradient through the residual addition: one branch flows straight to the
    // input residual, the other into the block body.
    auto grad_residual = grad_output;
    auto grad_x_nchw = grad_output;

    // Gradient through drop path (drop_path_prob is 0.1 here, so this scaling
    // branch is always active).
    if (drop_path_prob > 0.0) {
        const double keep_prob = 1.0 - drop_path_prob;
        grad_x_nchw = grad_x_nchw * drop_mask / keep_prob;
    }

    // Gradient through NHWC -> NCHW permutation: back to channels-last.
    auto grad_x_projected = grad_x_nchw.permute({0, 2, 3, 1});

    // Gradient through pwconv2: x_projected = x_grn @ pwconv2_weight.T + bias.
    auto grad_x_grn = at::linear(grad_x_projected, pwconv2_weight.t());

    auto grad_x_projected_flat =
        grad_x_projected.reshape({-1, grad_x_projected.size(-1)});
    auto x_grn_flat = x_grn.reshape({-1, x_grn.size(-1)});
    auto grad_pwconv2_weight_result = at::matmul(grad_x_projected_flat.t(), x_grn_flat);
    auto grad_pwconv2_bias_result =
        grad_x_projected.sum(/*dim=*/std::vector<int64_t>{0, 1, 2});

    // Gradient through GRN: x_grn = grn_weight * x_grn_scaled + grn_bias + x_gelu.
    auto grad_x_gelu_from_grn = grad_x_grn;
    auto grad_x_grn_scaled = grad_x_grn * grn_weight;

    auto grad_grn_weight_result =
        (grad_x_grn * x_grn_scaled).sum(/*dim=*/std::vector<int64_t>{0, 1, 2}, /*keepdim=*/true);
    auto grad_grn_bias_result =
        grad_x_grn.sum(/*dim=*/std::vector<int64_t>{0, 1, 2}, /*keepdim=*/true);

    // Gradient through x_grn_scaled = x_gelu * norm_features.
    auto grad_x_gelu_from_scaled = grad_x_grn_scaled * norm_features;
    auto grad_norm_features =
        (grad_x_grn_scaled * x_gelu).sum(/*dim=*/std::vector<int64_t>{1, 2}, /*keepdim=*/true);

    auto grad_x_gelu = grad_x_gelu_from_grn + grad_x_gelu_from_scaled;

    // Gradient through norm_features = global_features / (gf_mean + eps).
    auto grad_global_features = grad_norm_features / (gf_mean + eps);
    auto grad_gf_mean =
        -grad_norm_features * global_features / (gf_mean + eps).pow(2);

    // Gradient through gf_mean = global_features.mean(dim=-1, keepdim=True).
    const int64_t C_expanded = global_features.size(-1);
    grad_global_features =
        grad_global_features + grad_gf_mean / static_cast<double>(C_expanded);

    // Gradient through global_features = ||x_gelu||_2 over the spatial dims:
    // d/dx ||x||_2 = x / ||x||_2.
    grad_x_gelu =
        grad_x_gelu + x_gelu * grad_global_features / (global_features + eps);

    // Gradient through GELU (tanh approximation):
    //   inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    //   gelu(x) = 0.5 * x * (1 + tanh(inner))
    //   d/dx gelu(x) = cdf + x * pdf, with cdf/pdf as below.
    const double sqrt_2_over_pi = 0.7978845608028654;
    const double cdf_coeff = 0.044715;
    auto inner = sqrt_2_over_pi * (x_expanded + cdf_coeff * x_expanded.pow(3));
    auto tanh_inner = at::tanh(inner);
    auto cdf_approx = 0.5 * (1.0 + tanh_inner);
    auto pdf_approx = 0.5 * (1.0 - tanh_inner.pow(2)) * sqrt_2_over_pi *
                      (1.0 + 3.0 * cdf_coeff * x_expanded.pow(2));
    auto gelu_grad = cdf_approx + x_expanded * pdf_approx;
    auto grad_x_expanded = grad_x_gelu * gelu_grad;

    // Gradient through pwconv1: x_expanded = x_ln @ pwconv1_weight.T + bias.
    auto grad_x_ln = at::linear(grad_x_expanded, pwconv1_weight.t());

    auto grad_x_expanded_flat =
        grad_x_expanded.reshape({-1, grad_x_expanded.size(-1)});
    auto x_ln_flat = x_ln.reshape({-1, x_ln.size(-1)});
    auto grad_pwconv1_weight_result = at::matmul(grad_x_expanded_flat.t(), x_ln_flat);
    auto grad_pwconv1_bias_result =
        grad_x_expanded.sum(/*dim=*/std::vector<int64_t>{0, 1, 2});

    // Gradient through LayerNorm: x_ln = x_normalized * layernorm_weight + bias.
    auto grad_x_normalized = grad_x_ln * layernorm_weight;
    auto grad_layernorm_weight_result =
        (grad_x_ln * x_normalized).sum(/*dim=*/std::vector<int64_t>{0, 1, 2});
    auto grad_layernorm_bias_result =
        grad_x_ln.sum(/*dim=*/std::vector<int64_t>{0, 1, 2});

    // Gradient through normalization: x_normalized = (x_nhwc - mean) / sqrt(var + eps).
    auto std_dev = at::sqrt(var + eps);
    const int64_t N = x_nhwc.size(-1);
    auto x_centered = x_nhwc - mean;

    auto grad_x_nhwc = grad_x_normalized / std_dev;
    auto grad_var = -(grad_x_normalized * x_centered).sum(/*dim=*/-1, /*keepdim=*/true) /
                    (2.0 * (var + eps) * std_dev);
    auto grad_mean =
        -(grad_x_normalized / std_dev).sum(/*dim=*/-1, /*keepdim=*/true);
    grad_mean = grad_mean +
                grad_var * (-2.0 * x_centered.sum(/*dim=*/-1, /*keepdim=*/true) /
                            static_cast<double>(N));

    grad_x_nhwc =
        grad_x_nhwc + grad_var * (2.0 * x_centered / static_cast<double>(N));
    grad_x_nhwc = grad_x_nhwc + grad_mean / static_cast<double>(N);

    // Gradient through NCHW -> NHWC permutation: back to channels-first.
    auto grad_x_dwconv = grad_x_nhwc.permute({0, 3, 1, 2}).contiguous();

    // Gradient through the depthwise convolution input: transposed convolution
    // with groups == C and the same padding.
    const std::vector<int64_t> dw_stride{1, 1};
    const std::vector<int64_t> dw_pad{3, 3};
    const std::vector<int64_t> dw_output_pad{0, 0};
    const std::vector<int64_t> dw_dilation{1, 1};
    auto grad_x_from_conv = at::conv_transpose2d(
        grad_x_dwconv, dwconv_weight, /*bias=*/{},
        /*stride=*/dw_stride, /*padding=*/dw_pad,
        /*output_padding=*/dw_output_pad, /*groups=*/C, /*dilation=*/dw_dilation);
    auto grad_x_result = grad_x_from_conv + grad_residual;

    // Depthwise conv weight gradient via the grouped-convolution
    // cross-correlation reformulation: treat each (batch, channel) pair as its
    // own group so one grouped conv computes every per-channel correlation at
    // once. The residual is the input, grad_x_dwconv is the per-group kernel,
    // and the output 7x7 maps are summed over the batch.
    const int64_t kH = dwconv_weight.size(2);
    const int64_t kW = dwconv_weight.size(3);
    auto residual_grouped = residual.reshape({1, B * C, H, W});
    auto grad_kernel = grad_x_dwconv.reshape({B * C, 1, H, W});
    const std::vector<int64_t> wg_stride{1, 1};
    const std::vector<int64_t> wg_pad{3, 3};
    const std::vector<int64_t> wg_dilation{1, 1};
    auto weight_grad_grouped = at::conv2d(
        residual_grouped, grad_kernel, /*bias=*/{},
        /*stride=*/wg_stride, /*padding=*/wg_pad,
        /*dilation=*/wg_dilation, /*groups=*/B * C);
    auto grad_dwconv_weight_result =
        weight_grad_grouped.reshape({B, C, kH, kW})
            .sum(/*dim=*/0)
            .reshape({C, 1, kH, kW});

    // Depthwise conv bias gradient.
    auto grad_dwconv_bias_result =
        grad_x_dwconv.sum(/*dim=*/std::vector<int64_t>{0, 2, 3});

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_x_result.sizes() == grad_x.sizes(),
                "grad_x shape ", grad_x_result.sizes(),
                " does not match pre-allocated ", grad_x.sizes());
    TORCH_CHECK(grad_dwconv_weight_result.sizes() == grad_dwconv_weight.sizes(),
                "grad_dwconv_weight shape ", grad_dwconv_weight_result.sizes(),
                " does not match pre-allocated ", grad_dwconv_weight.sizes());
    TORCH_CHECK(grad_dwconv_bias_result.sizes() == grad_dwconv_bias.sizes(),
                "grad_dwconv_bias shape ", grad_dwconv_bias_result.sizes(),
                " does not match pre-allocated ", grad_dwconv_bias.sizes());
    TORCH_CHECK(grad_layernorm_weight_result.sizes() == grad_layernorm_weight.sizes(),
                "grad_layernorm_weight shape ", grad_layernorm_weight_result.sizes(),
                " does not match pre-allocated ", grad_layernorm_weight.sizes());
    TORCH_CHECK(grad_layernorm_bias_result.sizes() == grad_layernorm_bias.sizes(),
                "grad_layernorm_bias shape ", grad_layernorm_bias_result.sizes(),
                " does not match pre-allocated ", grad_layernorm_bias.sizes());
    TORCH_CHECK(grad_pwconv1_weight_result.sizes() == grad_pwconv1_weight.sizes(),
                "grad_pwconv1_weight shape ", grad_pwconv1_weight_result.sizes(),
                " does not match pre-allocated ", grad_pwconv1_weight.sizes());
    TORCH_CHECK(grad_pwconv1_bias_result.sizes() == grad_pwconv1_bias.sizes(),
                "grad_pwconv1_bias shape ", grad_pwconv1_bias_result.sizes(),
                " does not match pre-allocated ", grad_pwconv1_bias.sizes());
    TORCH_CHECK(grad_grn_weight_result.sizes() == grad_grn_weight.sizes(),
                "grad_grn_weight shape ", grad_grn_weight_result.sizes(),
                " does not match pre-allocated ", grad_grn_weight.sizes());
    TORCH_CHECK(grad_grn_bias_result.sizes() == grad_grn_bias.sizes(),
                "grad_grn_bias shape ", grad_grn_bias_result.sizes(),
                " does not match pre-allocated ", grad_grn_bias.sizes());
    TORCH_CHECK(grad_pwconv2_weight_result.sizes() == grad_pwconv2_weight.sizes(),
                "grad_pwconv2_weight shape ", grad_pwconv2_weight_result.sizes(),
                " does not match pre-allocated ", grad_pwconv2_weight.sizes());
    TORCH_CHECK(grad_pwconv2_bias_result.sizes() == grad_pwconv2_bias.sizes(),
                "grad_pwconv2_bias shape ", grad_pwconv2_bias_result.sizes(),
                " does not match pre-allocated ", grad_pwconv2_bias.sizes());

    grad_x.copy_(grad_x_result);
    grad_dwconv_weight.copy_(grad_dwconv_weight_result);
    grad_dwconv_bias.copy_(grad_dwconv_bias_result);
    grad_layernorm_weight.copy_(grad_layernorm_weight_result);
    grad_layernorm_bias.copy_(grad_layernorm_bias_result);
    grad_pwconv1_weight.copy_(grad_pwconv1_weight_result);
    grad_pwconv1_bias.copy_(grad_pwconv1_bias_result);
    grad_grn_weight.copy_(grad_grn_weight_result);
    grad_grn_bias.copy_(grad_grn_bias_result);
    grad_pwconv2_weight.copy_(grad_pwconv2_weight_result);
    grad_pwconv2_bias.copy_(grad_pwconv2_bias_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for the ConvNeXtV2 layer with NHWC persistence. Pure "
        "ATen / cuDNN / cuBLAS dispatch over the persisted forward "
        "intermediates: residual / drop-path / permutation grads, pwconv2 "
        "grad, GRN backward (grn affine grads, response-norm division and L2 "
        "norm grads), GELU tanh-approximation derivative, pwconv1 grad, manual "
        "LayerNorm backward, depthwise-conv input grad via conv_transpose2d, "
        "and the depthwise weight grad via the grouped-convolution "
        "cross-correlation reformulation. Native float32 throughout. "
        "Destination-passing style (11 outputs).");
}
