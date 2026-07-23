#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// ConvNeXtV2 block with Global Response Normalization (GRN), reproduced
// op-for-op from the reference run():
//
//   residual = x                                          # [B, C, H, W]
//   out = conv2d(x, dwconv_w, dwconv_b, padding=3, groups=C)   # depthwise 7x7
//   out = out.permute(0, 2, 3, 1)                         # [B, H, W, C]
//   out = layer_norm(out, (C,), ln_w, ln_b, layer_norm_eps)
//   out = matmul(out, pwconv1_w.T) + pwconv1_b            # [B, H, W, 4C]
//   out = gelu(out)
//   global_features = vector_norm(out, ord=2, dim=(1,2), keepdim=True) # [B,1,1,4C]
//   norm_features   = global_features / (global_features.mean(-1, keepdim) + eps)
//   out = grn_w * (out * norm_features) + grn_b + out
//   out = matmul(out, pwconv2_w.T) + pwconv2_b            # [B, H, W, C]
//   out = out.permute(0, 3, 1, 2)                         # [B, C, H, W]
//   out = residual + out
//
// The depthwise convolution carries one filter per channel (groups == dim) with
// padding 3 so the 7x7 kernel preserves spatial size. The two pointwise layers
// are channel-mixing matmuls against the transposed weights with a bias add,
// applied in the channels-last [B, H, W, *] layout. The GRN aggregates an L2
// norm over the two spatial axes per channel, divides by the channel-wise mean
// of those norms (offset by eps), then applies the learnable affine and adds the
// pre-GRN feature back. LayerNorm normalizes over the channel axis with its own
// epsilon. eps and layer_norm_eps arrive as fp32 scalars (passed as double).
//
// Per definition.json (insertion order):
//   x                : [batch_size, dim, height, width]        fp32
//   dwconv_weight    : [dim, 1, kernel_size, kernel_size]      fp32
//   dwconv_bias      : [dim]                                   fp32
//   layernorm_weight : [dim]                                   fp32
//   layernorm_bias   : [dim]                                   fp32
//   pwconv1_weight   : [expansion_dim, dim]                    fp32
//   pwconv1_bias     : [expansion_dim]                         fp32
//   grn_weight       : [1, 1, 1, expansion_dim]                fp32
//   grn_bias         : [1, 1, 1, expansion_dim]                fp32
//   pwconv2_weight   : [dim, expansion_dim]                    fp32
//   pwconv2_bias     : [dim]                                   fp32
//   eps              : scalar                                  fp32
//   layer_norm_eps   : scalar                                  fp32
//   output           : [batch_size, dim, height, width]        fp32 DPS (pre-alloc)
//
// Implementation: pure ATen / cuDNN / cuBLAS dispatch -- NO custom CUDA kernels.
// at::conv2d routes the depthwise convolution through cuDNN, at::layer_norm /
// at::gelu / at::linalg_vector_norm reproduce the normalization and activation,
// and the pointwise layers reuse at::linear (matmul against the transposed
// weight plus bias). Every op keeps the reference's native fp32 precision
// profile -- no extra casts. Output is written in place (destination-passing
// style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& dwconv_weight,
    const torch::Tensor& dwconv_bias,
    const torch::Tensor& layernorm_weight,
    const torch::Tensor& layernorm_bias,
    const torch::Tensor& pwconv1_weight,
    const torch::Tensor& pwconv1_bias,
    const torch::Tensor& grn_weight,
    const torch::Tensor& grn_bias,
    const torch::Tensor& pwconv2_weight,
    const torch::Tensor& pwconv2_bias,
    double eps,
    double layer_norm_eps,
    torch::Tensor output
) {
    check_fp32_cuda(x, "x");
    check_fp32_cuda(dwconv_weight, "dwconv_weight");
    check_fp32_cuda(dwconv_bias, "dwconv_bias");
    check_fp32_cuda(layernorm_weight, "layernorm_weight");
    check_fp32_cuda(layernorm_bias, "layernorm_bias");
    check_fp32_cuda(pwconv1_weight, "pwconv1_weight");
    check_fp32_cuda(pwconv1_bias, "pwconv1_bias");
    check_fp32_cuda(grn_weight, "grn_weight");
    check_fp32_cuda(grn_bias, "grn_bias");
    check_fp32_cuda(pwconv2_weight, "pwconv2_weight");
    check_fp32_cuda(pwconv2_bias, "pwconv2_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W], got ", x.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    const int64_t channels = x.size(1);

    auto residual = x;

    // Depthwise 7x7 convolution: groups == channels, padding 3 preserves H, W.
    const std::vector<int64_t> dw_stride{1, 1};
    const std::vector<int64_t> dw_pad{3, 3};
    const std::vector<int64_t> dw_dilation{1, 1};
    auto out = at::conv2d(x, dwconv_weight, dwconv_bias,
                          /*stride=*/dw_stride, /*padding=*/dw_pad,
                          /*dilation=*/dw_dilation, /*groups=*/channels);

    // Move to channels-last for the per-channel LayerNorm and pointwise matmuls.
    out = out.permute({0, 2, 3, 1});

    // LayerNorm over the channel axis with its own epsilon.
    out = at::layer_norm(out, /*normalized_shape=*/{channels},
                         layernorm_weight, layernorm_bias, layer_norm_eps);

    // Pointwise expansion: matmul against pwconv1_weight.T plus bias -> 4C.
    out = at::linear(out, pwconv1_weight, pwconv1_bias);

    out = at::gelu(out);

    // GRN: L2 norm over the two spatial axes (channels-last dims 1 and 2),
    // keepdim so the result is [B, 1, 1, 4C].
    auto global_features = at::linalg_vector_norm(
        out, /*ord=*/2, /*dim=*/std::vector<int64_t>{1, 2}, /*keepdim=*/true);

    // Divide by the channel-wise mean of those norms (offset by eps).
    auto norm_features = global_features /
                         (global_features.mean(/*dim=*/-1, /*keepdim=*/true) + eps);

    // Learnable affine on the response-normalized feature, with the pre-GRN
    // feature added back.
    out = grn_weight * (out * norm_features) + grn_bias + out;

    // Pointwise projection: matmul against pwconv2_weight.T plus bias -> C.
    out = at::linear(out, pwconv2_weight, pwconv2_bias);

    // Back to channels-first and add the input residual.
    out = out.permute({0, 3, 1, 2});
    auto result = residual + out;

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "ConvNeXtV2 block with Global Response Normalization via pure ATen "
          "dispatch: depthwise 7x7 convolution (groups=dim, padding=3), "
          "channels-last LayerNorm, pointwise expansion to 4x channels, GELU, "
          "GRN (spatial L2 norm divided by its channel-wise mean plus a "
          "learnable affine and pre-GRN add), pointwise projection back to dim, "
          "and an input residual. fp32; destination-passing style.");
}
