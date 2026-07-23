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

// Destination-passing-style entry for the Vision MLP + GELU-tanh
// backward chain. Per definition.json:
//   grad_output       : [S, hidden_size=1280]                 fp32
//   hidden_state      : [S, hidden_size=1280]                 fp32
//   fc1_weight        : [intermediate_size=3584, hidden_size] fp32
//   fc1_bias          : [intermediate_size=3584]              fp32 (unused
//                       by the run; signature included only to match the
//                       reference)
//   fc2_weight        : [hidden_size=1280, intermediate_size] fp32
//   fc2_bias          : [hidden_size=1280]                    fp32 (unused
//                       by the run; signature included only to match the
//                       reference)
//   fc1_output        : [S, intermediate_size=3584]           fp32
//   gelu_output       : [S, intermediate_size=3584]           fp32
//   grad_hidden_state : [S, hidden_size=1280]                 fp32 DPS
//   grad_fc1_weight   : [intermediate_size=3584, hidden_size] fp32 DPS
//   grad_fc1_bias     : [intermediate_size=3584]              fp32 DPS
//   grad_fc2_weight   : [hidden_size=1280, intermediate_size] fp32 DPS
//   grad_fc2_bias     : [hidden_size=1280]                    fp32 DPS
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_size       = 1280
//   intermediate_size = 3584
void run(
    torch::Tensor grad_output,
    torch::Tensor hidden_state,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias,
    torch::Tensor fc1_output,
    torch::Tensor gelu_output,
    torch::Tensor grad_hidden_state,
    torch::Tensor grad_fc1_weight,
    torch::Tensor grad_fc1_bias,
    torch::Tensor grad_fc2_weight,
    torch::Tensor grad_fc2_bias
) {
    check_fp32_contiguous_cuda(grad_output, "grad_output");
    check_fp32_contiguous_cuda(hidden_state, "hidden_state");
    check_fp32_contiguous_cuda(fc1_weight, "fc1_weight");
    check_fp32_contiguous_cuda(fc1_bias, "fc1_bias");
    check_fp32_contiguous_cuda(fc2_weight, "fc2_weight");
    check_fp32_contiguous_cuda(fc2_bias, "fc2_bias");
    check_fp32_contiguous_cuda(fc1_output, "fc1_output");
    check_fp32_contiguous_cuda(gelu_output, "gelu_output");
    check_fp32_contiguous_cuda(grad_hidden_state, "grad_hidden_state");
    check_fp32_contiguous_cuda(grad_fc1_weight, "grad_fc1_weight");
    check_fp32_contiguous_cuda(grad_fc1_bias, "grad_fc1_bias");
    check_fp32_contiguous_cuda(grad_fc2_weight, "grad_fc2_weight");
    check_fp32_contiguous_cuda(grad_fc2_bias, "grad_fc2_bias");

    constexpr int64_t kHiddenSize = 1280;
    constexpr int64_t kIntermediateSize = 3584;

    TORCH_CHECK(grad_output.dim() == 2,
                "grad_output must be 2D [S, hidden_size]");
    TORCH_CHECK(hidden_state.dim() == 2,
                "hidden_state must be 2D [S, hidden_size]");
    TORCH_CHECK(fc1_weight.dim() == 2,
                "fc1_weight must be 2D [intermediate_size, hidden_size]");
    TORCH_CHECK(fc1_bias.dim() == 1, "fc1_bias must be 1D");
    TORCH_CHECK(fc2_weight.dim() == 2,
                "fc2_weight must be 2D [hidden_size, intermediate_size]");
    TORCH_CHECK(fc2_bias.dim() == 1, "fc2_bias must be 1D");
    TORCH_CHECK(fc1_output.dim() == 2,
                "fc1_output must be 2D [S, intermediate_size]");
    TORCH_CHECK(gelu_output.dim() == 2,
                "gelu_output must be 2D [S, intermediate_size]");
    TORCH_CHECK(grad_hidden_state.dim() == 2,
                "grad_hidden_state must be 2D [S, hidden_size]");
    TORCH_CHECK(grad_fc1_weight.dim() == 2,
                "grad_fc1_weight must be 2D [intermediate_size, hidden_size]");
    TORCH_CHECK(grad_fc1_bias.dim() == 1, "grad_fc1_bias must be 1D");
    TORCH_CHECK(grad_fc2_weight.dim() == 2,
                "grad_fc2_weight must be 2D [hidden_size, intermediate_size]");
    TORCH_CHECK(grad_fc2_bias.dim() == 1, "grad_fc2_bias must be 1D");

    const int64_t S = grad_output.size(0);

    TORCH_CHECK(grad_output.size(1) == kHiddenSize,
                "grad_output.size(1) must equal hidden_size=1280, got ",
                grad_output.size(1));
    TORCH_CHECK(hidden_state.size(0) == S,
                "hidden_state.size(0) must equal grad_output.size(0)");
    TORCH_CHECK(hidden_state.size(1) == kHiddenSize,
                "hidden_state.size(1) must equal hidden_size=1280");
    TORCH_CHECK(fc1_weight.size(0) == kIntermediateSize,
                "fc1_weight.size(0) must equal intermediate_size=3584");
    TORCH_CHECK(fc1_weight.size(1) == kHiddenSize,
                "fc1_weight.size(1) must equal hidden_size=1280");
    TORCH_CHECK(fc1_bias.size(0) == kIntermediateSize,
                "fc1_bias.size(0) must equal intermediate_size=3584");
    TORCH_CHECK(fc2_weight.size(0) == kHiddenSize,
                "fc2_weight.size(0) must equal hidden_size=1280");
    TORCH_CHECK(fc2_weight.size(1) == kIntermediateSize,
                "fc2_weight.size(1) must equal intermediate_size=3584");
    TORCH_CHECK(fc2_bias.size(0) == kHiddenSize,
                "fc2_bias.size(0) must equal hidden_size=1280");
    TORCH_CHECK(fc1_output.size(0) == S,
                "fc1_output.size(0) must equal grad_output.size(0)");
    TORCH_CHECK(fc1_output.size(1) == kIntermediateSize,
                "fc1_output.size(1) must equal intermediate_size=3584");
    TORCH_CHECK(gelu_output.size(0) == S,
                "gelu_output.size(0) must equal grad_output.size(0)");
    TORCH_CHECK(gelu_output.size(1) == kIntermediateSize,
                "gelu_output.size(1) must equal intermediate_size=3584");
    TORCH_CHECK(grad_hidden_state.sizes() == hidden_state.sizes(),
                "grad_hidden_state must have the same shape as hidden_state");
    TORCH_CHECK(grad_fc1_weight.sizes() == fc1_weight.sizes(),
                "grad_fc1_weight must have the same shape as fc1_weight");
    TORCH_CHECK(grad_fc1_bias.size(0) == kIntermediateSize,
                "grad_fc1_bias.size(0) must equal intermediate_size=3584");
    TORCH_CHECK(grad_fc2_weight.sizes() == fc2_weight.sizes(),
                "grad_fc2_weight must have the same shape as fc2_weight");
    TORCH_CHECK(grad_fc2_bias.size(0) == kHiddenSize,
                "grad_fc2_bias.size(0) must equal hidden_size=1280");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_022_vision_mlp_gelu_backward_launcher(
        grad_output, hidden_state,
        fc1_weight, fc2_weight,
        fc1_output, gelu_output,
        grad_hidden_state,
        grad_fc1_weight, grad_fc1_bias,
        grad_fc2_weight, grad_fc2_bias,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 Vision MLP + GELU-tanh backward (Qwen3-VL-8B vision "
        "encoder). 7-kernel chain: 4 CUTLASS Sm80 TensorOp fp32 "
        "GEMMs + 1 per-element GELU-tanh backward + 2 per-column "
        "bias-sum reductions. hidden_size=1280, intermediate_size="
        "3584 pinned. Destination-passing style.");
}
