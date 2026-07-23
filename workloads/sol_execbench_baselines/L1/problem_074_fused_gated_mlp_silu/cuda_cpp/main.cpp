#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the fused gated MLP with
// SiLU activation. Per definition.json:
//   hidden_states     : [B, S, hidden_size=1024]                 fp32
//   gate_up_weight    : [intermediate_size_x2=8192,
//                        hidden_size=1024]                       fp32
//   down_weight       : [hidden_size=1024,
//                        intermediate_size=4096]                 fp32
//   output            : [B, S, hidden_size=1024]                 fp32 DPS
//
// Algorithm (mirrors the reference exactly):
//   up_states = F.linear(hidden_states, gate_up_weight)
//     # F.linear is `hidden_states @ gate_up_weight.T + 0`. On
//     # 2D fp32 contiguous operands, this dispatches cublasSgemm
//     # with transA = N, transB = T -- the novel
//     # `cublasSgemm_2D_fp32_OP_N_OP_T` family pinned by
//     # batch-041.
//     # shape: [B, S, intermediate_size_x2=8192]
//   gate, up_states = up_states.chunk(2, dim=-1)
//     # each: [B, S, intermediate_size=4096]
//   silu_gate = gate * sigmoid(gate)     # SiLU activation
//   up_states = up_states * silu_gate    # gated multiply
//   output = F.linear(up_states, down_weight)
//     # 2D fp32 cublasSgemm OP_N OP_T at a different shape
//     # ([B*S, intermediate_size=4096] x [hidden_size=1024,
//     # intermediate_size=4096].T -> [B*S, hidden_size=1024]).
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_size          = 1024
//   intermediate_size    = 4096
//   intermediate_size_x2 = 8192 (= 2 * intermediate_size)
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA
// kernels. The two 2D fp32 matmul dispatches route through
// cublasSgemm (legacy fp32 API) at transA=N transB=T orientation
// because PyTorch's F.linear stores weight row-major as
// [out_features, in_features] and the forward computes
// `input @ weight.T`. The SiLU-gate fusion is expressed as
// elementwise ATen ops (chunk + sigmoid + multiply); the
// dominant dispatch cost is the two cublasSgemm calls.
void run(
    torch::Tensor hidden_states,
    torch::Tensor gate_up_weight,
    torch::Tensor down_weight,
    torch::Tensor output
) {
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(gate_up_weight, "gate_up_weight");
    check_fp32_contiguous_cuda(down_weight, "down_weight");
    check_fp32_contiguous_cuda(output, "output");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 1024;
    constexpr int64_t kIntermediateSize = 4096;
    constexpr int64_t kIntermediateSizeX2 = 2 * kIntermediateSize;  // 8192

    TORCH_CHECK(hidden_states.dim() == 3 &&
                hidden_states.size(2) == kHiddenSize,
                "hidden_states must be [B, S, hidden_size=1024]");
    const int64_t B = hidden_states.size(0);
    const int64_t S = hidden_states.size(1);
    TORCH_CHECK(gate_up_weight.dim() == 2 &&
                gate_up_weight.size(0) == kIntermediateSizeX2 &&
                gate_up_weight.size(1) == kHiddenSize,
                "gate_up_weight must be [intermediate_size_x2=8192, hidden_size=1024]");
    TORCH_CHECK(down_weight.dim() == 2 &&
                down_weight.size(0) == kHiddenSize &&
                down_weight.size(1) == kIntermediateSize,
                "down_weight must be [hidden_size=1024, intermediate_size=4096]");
    TORCH_CHECK(output.dim() == 3 &&
                output.size(0) == B && output.size(1) == S &&
                output.size(2) == kHiddenSize,
                "output must be [B, S, hidden_size=1024]");

    // Fused gate + up projection. The optional bias argument is
    // omitted (matches the reference `F.linear(hidden_states,
    // gate_up_weight)` with no bias). The dispatch is the
    // novel `cublasSgemm_2D_fp32_OP_N_OP_T` family pinned by
    // batch-041.
    auto up_states = at::linear(hidden_states, gate_up_weight);

    // Split into gate (first half) and up (second half) along
    // the last dim. The split returns a tuple of VIEWS; the
    // subsequent elementwise ops produce contiguous tensors.
    auto split = up_states.chunk(2, /*dim=*/-1);
    auto& gate = split[0];
    auto& up_split = split[1];

    // SiLU activation on gate + gated multiply on up_split.
    auto silu_gate = gate * at::sigmoid(gate);
    auto gated = up_split * silu_gate;

    // Down projection -- the second cublasSgemm 2D fp32 OP_N
    // OP_T dispatch. Note `at::linear` accepts a 3D input and
    // implicitly flattens to 2D for the matmul (PyTorch
    // semantics); the dispatch shape is
    // [B*S, intermediate_size] x [hidden_size, intermediate_size].T
    // -> [B*S, hidden_size] which is then viewed back to
    // [B, S, hidden_size].
    output.copy_(at::linear(gated, down_weight));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused gated MLP with SiLU activation. Pure ATen / cuBLAS "
        "dispatch chain: two F.linear-equivalent 2D fp32 cublasSgemm "
        "calls at transA=N transB=T orientation (the novel "
        "cublasSgemm_2D_fp32_OP_N_OP_T family pinned by batch-041), "
        "with a SiLU-gate fusion expressed as ATen elementwise ops "
        "(chunk + sigmoid + multiply). Const axes: hidden_size=1024, "
        "intermediate_size=4096, intermediate_size_x2=8192. "
        "Destination-passing style.");
}
