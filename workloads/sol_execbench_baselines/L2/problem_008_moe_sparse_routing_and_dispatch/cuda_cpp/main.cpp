#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry. Per definition.json input order is
// hidden_states, gate_weight, expert_gate_proj, expert_up_proj,
// expert_down_proj, norm_topk_prob (scalar bool), output.

static void check_bf16_cuda_contiguous(
    const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ",
        t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void run(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    torch::Tensor expert_gate_proj,
    torch::Tensor expert_up_proj,
    torch::Tensor expert_down_proj,
    bool norm_topk_prob,
    torch::Tensor output) {

    check_bf16_cuda_contiguous(hidden_states, "hidden_states");
    check_bf16_cuda_contiguous(gate_weight, "gate_weight");
    check_bf16_cuda_contiguous(expert_gate_proj, "expert_gate_proj");
    check_bf16_cuda_contiguous(expert_up_proj, "expert_up_proj");
    check_bf16_cuda_contiguous(expert_down_proj, "expert_down_proj");
    check_bf16_cuda_contiguous(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D");
    TORCH_CHECK(
        hidden_states.size(2) == 2048, "hidden_size must be 2048");
    TORCH_CHECK(
        gate_weight.size(0) == 128, "num_experts must be 128");
    TORCH_CHECK(
        expert_gate_proj.size(1) == 768,
        "intermediate_size must be 768");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_008_moe_sparse_routing_and_dispatch_launcher(
        hidden_states,
        gate_weight,
        expert_gate_proj,
        expert_up_proj,
        expert_down_proj,
        norm_topk_prob,
        output,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Sparse MoE routing + per-expert dispatch (top-8 of 128). "
        "One custom CUDA kernel for the SiLU * up fusion; gate / "
        "softmax / topk / per-expert MLP matmuls dispatch through "
        "at:: ops. Destination-passing style.");
}
