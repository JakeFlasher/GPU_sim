#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the fused MLP block with
// GELU + Global Response Normalization (GRN). Per definition.json:
//   hidden_states   : [B, H, W, dim=128]           fp32
//   pwconv1_weight  : [hidden_dim=512, dim=128]    fp32
//   pwconv1_bias    : [hidden_dim=512]             fp32
//   grn_weight      : [1, 1, 1, hidden_dim=512]    fp32
//   grn_bias        : [1, 1, 1, hidden_dim=512]    fp32
//   pwconv2_weight  : [dim=128, hidden_dim=512]    fp32
//   pwconv2_bias    : [dim=128]                    fp32
//   eps             : scalar float
//   output          : [B, H, W, dim=128]           fp32 DPS
//
// Algorithm (mirrors the reference exactly):
//   x = F.linear(hidden_states, pwconv1_weight, pwconv1_bias)
//     # NHWC 4D input + 2D weight; F.linear flattens to
//     # [-1, dim] for the matmul and reshapes back. Dispatches
//     # cublasSgemm at the `cublasSgemm_2D_fp32_OP_N_OP_T`
//     # family (input @ weight.T because weight is stored
//     # [out_features, in_features]).
//     # shape: [B, H, W, hidden_dim=512]
//   x = F.gelu(x)
//     # GELU with the default "none" approximation
//     # (erf-based: 0.5 * x * (1 + erf(x / sqrt(2)))).
//   global_features = linalg.vector_norm(x, ord=2, dim=(1, 2),
//                                        keepdim=True)
//     # L2 norm across the spatial (H, W) axes. Shape:
//     # [B, 1, 1, hidden_dim].
//   norm_features = global_features /
//                   (global_features.mean(dim=-1, keepdim=True)
//                    + eps)
//     # Channel-wise mean across the hidden_dim axis; divide.
//   x = grn_weight * (x * norm_features) + grn_bias + x
//     # Affine multiply with grn_weight on the
//     # (x * norm_features) product, add grn_bias, add the
//     # pre-GRN x as a residual.
//   output = F.linear(x, pwconv2_weight, pwconv2_bias)
//     # Second cublasSgemm 2D fp32 OP_N OP_T dispatch (same
//     # family) at a different shape.
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   dim         = 128
//   hidden_dim  = 512   (= 4 * dim, the ConvNextV2 expansion factor)
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA
// kernels. The two 2D fp32 matmul dispatches route through
// cublasSgemm (legacy fp32 API) at transA=N transB=T orientation
// (the `cublasSgemm_2D_fp32_OP_N_OP_T` family pinned by
// batch-041 + batch-042). The GELU activation and the GRN
// elementwise / reduction chain run through ATen ops directly;
// the dominant dispatch cost is the two cublasSgemm calls.
//
// The axis-list arguments to `linalg_vector_norm` and `mean`
// are wrapped in `std::vector<int64_t>` so the int-axis
// overload binds unambiguously (a bare brace-init like
// `{1, 2}` can resolve to either the int-axis or named-axis
// overload under current Torch headers).
void run(
    torch::Tensor hidden_states,
    torch::Tensor pwconv1_weight,
    torch::Tensor pwconv1_bias,
    torch::Tensor grn_weight,
    torch::Tensor grn_bias,
    torch::Tensor pwconv2_weight,
    torch::Tensor pwconv2_bias,
    double eps,
    torch::Tensor output
) {
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(pwconv1_weight, "pwconv1_weight");
    check_fp32_contiguous_cuda(pwconv1_bias, "pwconv1_bias");
    check_fp32_contiguous_cuda(grn_weight, "grn_weight");
    check_fp32_contiguous_cuda(grn_bias, "grn_bias");
    check_fp32_contiguous_cuda(pwconv2_weight, "pwconv2_weight");
    check_fp32_contiguous_cuda(pwconv2_bias, "pwconv2_bias");
    check_fp32_contiguous_cuda(output, "output");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kDim = 128;
    constexpr int64_t kHiddenDim = 512;

    TORCH_CHECK(hidden_states.dim() == 4 &&
                hidden_states.size(3) == kDim,
                "hidden_states must be [B, H, W, dim=128]");
    const int64_t B = hidden_states.size(0);
    const int64_t H = hidden_states.size(1);
    const int64_t W = hidden_states.size(2);
    TORCH_CHECK(pwconv1_weight.dim() == 2 &&
                pwconv1_weight.size(0) == kHiddenDim &&
                pwconv1_weight.size(1) == kDim,
                "pwconv1_weight must be [hidden_dim=512, dim=128]");
    TORCH_CHECK(pwconv1_bias.dim() == 1 &&
                pwconv1_bias.size(0) == kHiddenDim,
                "pwconv1_bias must be [hidden_dim=512]");
    TORCH_CHECK(grn_weight.dim() == 4 &&
                grn_weight.size(0) == 1 && grn_weight.size(1) == 1 &&
                grn_weight.size(2) == 1 && grn_weight.size(3) == kHiddenDim,
                "grn_weight must be [1, 1, 1, hidden_dim=512]");
    TORCH_CHECK(grn_bias.dim() == 4 &&
                grn_bias.size(0) == 1 && grn_bias.size(1) == 1 &&
                grn_bias.size(2) == 1 && grn_bias.size(3) == kHiddenDim,
                "grn_bias must be [1, 1, 1, hidden_dim=512]");
    TORCH_CHECK(pwconv2_weight.dim() == 2 &&
                pwconv2_weight.size(0) == kDim &&
                pwconv2_weight.size(1) == kHiddenDim,
                "pwconv2_weight must be [dim=128, hidden_dim=512]");
    TORCH_CHECK(pwconv2_bias.dim() == 1 &&
                pwconv2_bias.size(0) == kDim,
                "pwconv2_bias must be [dim=128]");
    TORCH_CHECK(output.dim() == 4 && output.size(0) == B &&
                output.size(1) == H && output.size(2) == W &&
                output.size(3) == kDim,
                "output must be [B, H, W, dim=128]");

    // Expansion Linear. F.linear with bias dispatches through
    // cublasSgemm 2D fp32 OP_N OP_T + a fused bias-add.
    auto x = at::linear(hidden_states, pwconv1_weight, pwconv1_bias);

    // GELU activation. PyTorch's default (no `approximate` kwarg)
    // is the erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
    x = at::gelu(x);

    // Global Response Normalization (GRN). The dim-list args are
    // typed `std::vector<int64_t>` to bind the int-axis overload
    // unambiguously.
    const std::vector<int64_t> kSpatialDims = {1, 2};
    auto global_features = at::linalg_vector_norm(
        x,
        /*ord=*/at::Scalar(2.0),
        /*dim=*/kSpatialDims,
        /*keepdim=*/true,
        /*dtype=*/c10::nullopt);

    // Channel-wise mean across the hidden_dim axis (last dim).
    auto channel_mean = global_features.mean(/*dim=*/-1, /*keepdim=*/true);
    auto norm_features = global_features / (channel_mean + eps);

    // Affine multiply + residual add.
    x = grn_weight * (x * norm_features) + grn_bias + x;

    // Projection Linear. Same family at a different shape.
    output.copy_(at::linear(x, pwconv2_weight, pwconv2_bias));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused MLP block with GELU and Global Response Normalization "
        "(ConvNextV2 pattern). Pure ATen / cuBLAS dispatch chain: two "
        "F.linear-equivalent 2D fp32 cublasSgemm calls at transA=N "
        "transB=T orientation (the cublasSgemm_2D_fp32_OP_N_OP_T family "
        "pinned by batch-041 + batch-042), a GELU activation, and a GRN "
        "block that L2-norms across the spatial dims, channel-divides "
        "with eps, affine-multiplies with a residual add, all via ATen "
        "elementwise / reduction ops. Const axes: dim=128, hidden_dim=512 "
        "(ConvNextV2 4x expansion). Destination-passing style.");
}
