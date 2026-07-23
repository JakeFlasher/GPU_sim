#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the sparse MoE routing + dispatch baseline.
// num_experts=128, num_experts_per_tok=8, hidden_size=2048,
// intermediate_size=768 pinned by the problem definition.
// Destination-passing style on `output`.
void problem_008_moe_sparse_routing_and_dispatch_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& gate_weight,
    const torch::Tensor& expert_gate_proj,
    const torch::Tensor& expert_up_proj,
    const torch::Tensor& expert_down_proj,
    bool norm_topk_prob,
    torch::Tensor& output,
    cudaStream_t stream);

void silu_mul_bf16_kernel_launch(
    const at::BFloat16* gate,
    const at::BFloat16* up,
    at::BFloat16* out,
    long num_elems,
    cudaStream_t stream);
