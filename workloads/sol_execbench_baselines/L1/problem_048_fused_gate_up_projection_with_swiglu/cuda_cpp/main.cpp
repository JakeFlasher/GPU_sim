#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the Gemma3 fused gate + up
// projection with GELU-tanh chain. Per definition.json:
//   x         : [B, S, hidden_size = 3072]            bf16
//   gate_proj : [intermediate_size = 24576, hidden_size = 3072] bf16
//   up_proj   : [intermediate_size = 24576, hidden_size = 3072] bf16
//   output    : [B, S, intermediate_size = 24576]     bf16 (DPS)
//
// Algorithm (mirroring the PyTorch reference):
//   gate_output    = torch.matmul(x, gate_proj.t())
//   up_output      = torch.matmul(x, up_proj.t())
//   gate_fp32      = gate_output.to(fp32)
//   inner          = sqrt(2/pi) * (gate_fp32 + 0.044715 * gate_fp32**3)
//   activated_gate = 0.5 * gate_fp32 * (1 + tanh(inner))
//   activated_gate = activated_gate.to(bf16)   # bf16 downcast
//   output         = activated_gate * up_output   # bf16 multiply
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_size       = 3072
//   intermediate_size = 24576
void run(
    torch::Tensor x,
    torch::Tensor gate_proj,
    torch::Tensor up_proj,
    torch::Tensor output
) {
    check_bf16_contiguous_cuda(x, "x");
    check_bf16_contiguous_cuda(gate_proj, "gate_proj");
    check_bf16_contiguous_cuda(up_proj, "up_proj");
    check_bf16_contiguous_cuda(output, "output");

    constexpr int64_t kHiddenSize = 3072;
    constexpr int64_t kIntermediateSize = 24576;

    TORCH_CHECK(x.dim() == 3,
                "x must be 3D [B, S, hidden_size]");
    TORCH_CHECK(gate_proj.dim() == 2,
                "gate_proj must be 2D [intermediate_size, hidden_size]");
    TORCH_CHECK(up_proj.dim() == 2,
                "up_proj must be 2D [intermediate_size, hidden_size]");
    TORCH_CHECK(output.dim() == 3,
                "output must be 3D [B, S, intermediate_size]");

    TORCH_CHECK(x.size(2) == kHiddenSize,
                "x.size(2) must equal hidden_size=3072, got ",
                x.size(2));
    TORCH_CHECK(gate_proj.size(0) == kIntermediateSize,
                "gate_proj.size(0) must equal intermediate_size=24576");
    TORCH_CHECK(gate_proj.size(1) == kHiddenSize,
                "gate_proj.size(1) must equal hidden_size=3072");
    TORCH_CHECK(up_proj.size(0) == kIntermediateSize,
                "up_proj.size(0) must equal intermediate_size=24576");
    TORCH_CHECK(up_proj.size(1) == kHiddenSize,
                "up_proj.size(1) must equal hidden_size=3072");
    TORCH_CHECK(output.size(0) == x.size(0),
                "output.size(0) must equal x.size(0)");
    TORCH_CHECK(output.size(1) == x.size(1),
                "output.size(1) must equal x.size(1)");
    TORCH_CHECK(output.size(2) == kIntermediateSize,
                "output.size(2) must equal intermediate_size=24576");

    // bf16 cuBLAS GEMMs via at::matmul. cuBLAS dispatches a bf16
    // GEMM using Sm80 bf16 tensor cores with fp32 accumulator. The
    // reference also uses torch.matmul for these projections, so
    // the cuBLAS path matches exactly.
    auto gate_output = at::matmul(x, gate_proj.t());
    auto up_output = at::matmul(x, up_proj.t());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_048_fused_gelu_tanh_mul_launcher(
        gate_output, up_output, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "bf16 Gemma3 fused gate + up projection with GELU-tanh. "
        "Three-launch chain: 2 cuBLAS at::matmul calls + 1 fused "
        "per-element GELU-tanh + bf16 multiply. hidden_size=3072, "
        "intermediate_size=24576 pinned. Destination-passing style.");
}
