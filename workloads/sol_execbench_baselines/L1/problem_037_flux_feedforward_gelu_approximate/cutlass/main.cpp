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

// Destination-passing-style entry for the FLUX feedforward.
// Per definition.json:
//   hidden_states : [B, S, hidden_dim=3072] fp32
//   fc1_weight    : [mlp_hidden_dim=12288, hidden_dim=3072] fp32
//   fc1_bias      : [mlp_hidden_dim=12288] fp32
//   fc2_weight    : [hidden_dim=3072, mlp_hidden_dim=12288] fp32
//   fc2_bias      : [hidden_dim=3072] fp32
//   output        : [B, S, hidden_dim=3072] fp32 (DPS)
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_dim     = 3072
//   mlp_hidden_dim = 12288

void run(
    torch::Tensor hidden_states,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias,
    torch::Tensor output
) {
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(fc1_weight, "fc1_weight");
    check_fp32_contiguous_cuda(fc1_bias, "fc1_bias");
    check_fp32_contiguous_cuda(fc2_weight, "fc2_weight");
    check_fp32_contiguous_cuda(fc2_bias, "fc2_bias");
    check_fp32_contiguous_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden_dim]");
    TORCH_CHECK(fc1_weight.dim() == 2, "fc1_weight must be 2D");
    TORCH_CHECK(fc1_bias.dim() == 1, "fc1_bias must be 1D");
    TORCH_CHECK(fc2_weight.dim() == 2, "fc2_weight must be 2D");
    TORCH_CHECK(fc2_bias.dim() == 1, "fc2_bias must be 1D");
    TORCH_CHECK(output.dim() == 3,
                "output must be 3D [B, S, hidden_dim]");

    constexpr int64_t kHiddenDim = 3072;
    constexpr int64_t kMlpHiddenDim = 12288;

    TORCH_CHECK(hidden_states.size(2) == kHiddenDim,
                "hidden_states.size(2) must equal hidden_dim=3072, got ",
                hidden_states.size(2));
    TORCH_CHECK(fc1_weight.size(0) == kMlpHiddenDim,
                "fc1_weight.size(0) must equal mlp_hidden_dim=12288, got ",
                fc1_weight.size(0));
    TORCH_CHECK(fc1_weight.size(1) == kHiddenDim,
                "fc1_weight.size(1) must equal hidden_dim=3072, got ",
                fc1_weight.size(1));
    TORCH_CHECK(fc1_bias.size(0) == kMlpHiddenDim,
                "fc1_bias.size(0) must equal mlp_hidden_dim=12288, got ",
                fc1_bias.size(0));
    TORCH_CHECK(fc2_weight.size(0) == kHiddenDim,
                "fc2_weight.size(0) must equal hidden_dim=3072, got ",
                fc2_weight.size(0));
    TORCH_CHECK(fc2_weight.size(1) == kMlpHiddenDim,
                "fc2_weight.size(1) must equal mlp_hidden_dim=12288, got ",
                fc2_weight.size(1));
    TORCH_CHECK(fc2_bias.size(0) == kHiddenDim,
                "fc2_bias.size(0) must equal hidden_dim=3072, got ",
                fc2_bias.size(0));

    TORCH_CHECK(output.size(0) == hidden_states.size(0),
                "output.size(0) must equal hidden_states.size(0)");
    TORCH_CHECK(output.size(1) == hidden_states.size(1),
                "output.size(1) must equal hidden_states.size(1)");
    TORCH_CHECK(output.size(2) == kHiddenDim,
                "output.size(2) must equal hidden_dim=3072, got ",
                output.size(2));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_037_flux_feedforward_gelu_approximate_launcher(
        hidden_states, fc1_weight, fc1_bias,
        fc2_weight, fc2_bias, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 FLUX feedforward: CUTLASS Sm80 TensorOp GEMM 1 "
        "(fp32 tf32 mma) + GELU-tanh+bias kernel + CUTLASS Sm80 "
        "TensorOp GEMM 2 + bias-add kernel. hidden_dim=3072, "
        "mlp_hidden_dim=12288 pinned. Destination-passing style.");
}
