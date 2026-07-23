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

// Destination-passing-style entry for the FLUX final-output norm
// + projection chain. Per definition.json:
//   hidden_states   : [B, S, inner_dim=3072]                  fp32
//   temb            : [B, inner_dim=3072]                     fp32
//   linear_weight   : [modulation_dim=6144, inner_dim=3072]   fp32
//   linear_bias     : [modulation_dim=6144]                   fp32
//   proj_out_weight : [output_dim=64, inner_dim=3072]         fp32
//   proj_out_bias   : [output_dim=64]                         fp32
//   eps             : fp32 scalar (passed as double, downcast at
//                     the launcher boundary)
//   output          : [B, S, output_dim=64]                   fp32 (DPS)
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   inner_dim       = 3072
//   output_dim      = 64
//   modulation_dim  = 2 * inner_dim = 6144
void run(
    torch::Tensor hidden_states,
    torch::Tensor temb,
    torch::Tensor linear_weight,
    torch::Tensor linear_bias,
    torch::Tensor proj_out_weight,
    torch::Tensor proj_out_bias,
    double eps,
    torch::Tensor output
) {
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(temb, "temb");
    check_fp32_contiguous_cuda(linear_weight, "linear_weight");
    check_fp32_contiguous_cuda(linear_bias, "linear_bias");
    check_fp32_contiguous_cuda(proj_out_weight, "proj_out_weight");
    check_fp32_contiguous_cuda(proj_out_bias, "proj_out_bias");
    check_fp32_contiguous_cuda(output, "output");

    constexpr int64_t kInnerDim = 3072;
    constexpr int64_t kOutputDim = 64;
    constexpr int64_t kModulationDim = 2 * kInnerDim;  // 6144

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, inner_dim]");
    TORCH_CHECK(temb.dim() == 2, "temb must be 2D [B, inner_dim]");
    TORCH_CHECK(linear_weight.dim() == 2,
                "linear_weight must be 2D [modulation_dim, inner_dim]");
    TORCH_CHECK(linear_bias.dim() == 1,
                "linear_bias must be 1D [modulation_dim]");
    TORCH_CHECK(proj_out_weight.dim() == 2,
                "proj_out_weight must be 2D [output_dim, inner_dim]");
    TORCH_CHECK(proj_out_bias.dim() == 1,
                "proj_out_bias must be 1D [output_dim]");
    TORCH_CHECK(output.dim() == 3,
                "output must be 3D [B, S, output_dim]");

    TORCH_CHECK(hidden_states.size(2) == kInnerDim,
                "hidden_states.size(2) must equal inner_dim=3072, got ",
                hidden_states.size(2));
    TORCH_CHECK(temb.size(0) == hidden_states.size(0),
                "temb.size(0) must equal hidden_states.size(0)");
    TORCH_CHECK(temb.size(1) == kInnerDim,
                "temb.size(1) must equal inner_dim=3072");
    TORCH_CHECK(linear_weight.size(0) == kModulationDim,
                "linear_weight.size(0) must equal modulation_dim=6144");
    TORCH_CHECK(linear_weight.size(1) == kInnerDim,
                "linear_weight.size(1) must equal inner_dim=3072");
    TORCH_CHECK(linear_bias.size(0) == kModulationDim,
                "linear_bias.size(0) must equal modulation_dim=6144");
    TORCH_CHECK(proj_out_weight.size(0) == kOutputDim,
                "proj_out_weight.size(0) must equal output_dim=64");
    TORCH_CHECK(proj_out_weight.size(1) == kInnerDim,
                "proj_out_weight.size(1) must equal inner_dim=3072");
    TORCH_CHECK(proj_out_bias.size(0) == kOutputDim,
                "proj_out_bias.size(0) must equal output_dim=64");
    TORCH_CHECK(output.size(0) == hidden_states.size(0),
                "output.size(0) must equal hidden_states.size(0)");
    TORCH_CHECK(output.size(1) == hidden_states.size(1),
                "output.size(1) must equal hidden_states.size(1)");
    TORCH_CHECK(output.size(2) == kOutputDim,
                "output.size(2) must equal output_dim=64");

    const float eps_f32 = static_cast<float>(eps);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_036_flux_output_norm_projection_chain_launcher(
        hidden_states, temb,
        linear_weight, linear_bias,
        proj_out_weight, proj_out_bias,
        eps_f32, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 FLUX final-output norm + projection chain. Four-launch "
        "chain: per-element SiLU + small-M Linear (temb -> "
        "modulation[B, 6144]) + fused LN + modulation per-(b, s) "
        "row + cuBLAS fp32 GEMM via at::linear_out for the final "
        "small-N=64 output projection. inner_dim=3072, output_dim="
        "64, modulation_dim=6144 pinned. Destination-passing style.");
}
