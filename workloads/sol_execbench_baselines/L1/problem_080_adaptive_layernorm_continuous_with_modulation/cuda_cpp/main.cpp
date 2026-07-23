#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the AdaLayerNormContinuous +
// modulation kernel chain. Per definition.json:
//   hidden_states : [B, S, inner_dim=2432] fp32
//   temb          : [B, inner_dim=2432] fp32
//   linear_weight : [inner_dim_x2=4864, inner_dim=2432] fp32
//   linear_bias   : [inner_dim_x2=4864] fp32
//   eps           : scalar fp32 (passed as double, matching reference signature)
//   output        : [B, S, inner_dim=2432] fp32 (DPS)
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   inner_dim     = 2432
//   inner_dim_x2  = 4864

void run(
    torch::Tensor hidden_states,
    torch::Tensor temb,
    torch::Tensor linear_weight,
    torch::Tensor linear_bias,
    double eps,
    torch::Tensor output
) {
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(temb, "temb");
    check_fp32_contiguous_cuda(linear_weight, "linear_weight");
    check_fp32_contiguous_cuda(linear_bias, "linear_bias");
    check_fp32_contiguous_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, inner_dim]");
    TORCH_CHECK(temb.dim() == 2, "temb must be 2D [B, inner_dim]");
    TORCH_CHECK(linear_weight.dim() == 2,
                "linear_weight must be 2D [inner_dim_x2, inner_dim]");
    TORCH_CHECK(linear_bias.dim() == 1,
                "linear_bias must be 1D [inner_dim_x2]");
    TORCH_CHECK(output.dim() == 3,
                "output must be 3D [B, S, inner_dim]");

    constexpr int64_t kInnerDim = 2432;
    constexpr int64_t kInnerDimX2 = 2 * kInnerDim;  // 4864

    TORCH_CHECK(hidden_states.size(2) == kInnerDim,
                "hidden_states.size(2) must equal inner_dim=2432, got ",
                hidden_states.size(2));
    TORCH_CHECK(temb.size(0) == hidden_states.size(0),
                "temb.size(0) must equal hidden_states.size(0)");
    TORCH_CHECK(temb.size(1) == kInnerDim,
                "temb.size(1) must equal inner_dim=2432, got ",
                temb.size(1));
    TORCH_CHECK(linear_weight.size(0) == kInnerDimX2,
                "linear_weight.size(0) must equal inner_dim_x2=4864, got ",
                linear_weight.size(0));
    TORCH_CHECK(linear_weight.size(1) == kInnerDim,
                "linear_weight.size(1) must equal inner_dim=2432, got ",
                linear_weight.size(1));
    TORCH_CHECK(linear_bias.size(0) == kInnerDimX2,
                "linear_bias.size(0) must equal inner_dim_x2=4864, got ",
                linear_bias.size(0));
    TORCH_CHECK(output.sizes() == hidden_states.sizes(),
                "output must have the same shape as hidden_states");

    const float eps_f32 = static_cast<float>(eps);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_080_adaptive_layernorm_continuous_with_modulation_launcher(
        hidden_states, temb, linear_weight, linear_bias,
        eps_f32, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 AdaLayerNormContinuous with timestep-conditioned "
        "modulation. Two-kernel chain: small-M Linear "
        "(temb -> modulation[B, 4864]) + per-(b, s) LN + "
        "modulation kernel. inner_dim=2432, inner_dim_x2=4864 "
        "pinned. Destination-passing style.");
}
