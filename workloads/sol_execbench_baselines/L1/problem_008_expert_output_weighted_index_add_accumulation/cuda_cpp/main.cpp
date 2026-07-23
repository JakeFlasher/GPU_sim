#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry. Per definition.json:
//   final_hidden_states: [batch_seq_len, 3072] bf16
//   expert_outputs:      [num_selected_tokens, 3072] bf16
//   token_indices:       [num_selected_tokens] int64
//   output (DPS):        [batch_seq_len, 3072] bf16
// hidden_size is pinned at 3072.

void run(
    torch::Tensor final_hidden_states,
    torch::Tensor expert_outputs,
    torch::Tensor token_indices,
    torch::Tensor output) {

    TORCH_CHECK(final_hidden_states.is_cuda(),
                "final_hidden_states must be CUDA");
    TORCH_CHECK(expert_outputs.is_cuda(),
                "expert_outputs must be CUDA");
    TORCH_CHECK(token_indices.is_cuda(),
                "token_indices must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");

    TORCH_CHECK(
        final_hidden_states.dtype() == torch::kBFloat16,
        "final_hidden_states must be bfloat16");
    TORCH_CHECK(
        expert_outputs.dtype() == torch::kBFloat16,
        "expert_outputs must be bfloat16");
    TORCH_CHECK(
        token_indices.dtype() == torch::kInt64,
        "token_indices must be int64");
    TORCH_CHECK(
        output.dtype() == torch::kBFloat16,
        "output must be bfloat16");

    TORCH_CHECK(final_hidden_states.is_contiguous() &&
                expert_outputs.is_contiguous() &&
                token_indices.is_contiguous() &&
                output.is_contiguous(),
                "all tensors must be contiguous");

    TORCH_CHECK(final_hidden_states.dim() == 2,
                "final_hidden_states must be 2D");
    TORCH_CHECK(expert_outputs.dim() == 2,
                "expert_outputs must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(
        final_hidden_states.size(1) == 3072,
        "hidden_size must be 3072");
    TORCH_CHECK(
        expert_outputs.size(1) == 3072,
        "expert_outputs last dim must be 3072");
    TORCH_CHECK(
        token_indices.size(0) == expert_outputs.size(0),
        "token_indices length must equal num_selected_tokens");
    TORCH_CHECK(
        output.size(0) == final_hidden_states.size(0) &&
        output.size(1) == final_hidden_states.size(1),
        "output must match final_hidden_states shape");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_008_expert_output_weighted_index_add_accumulation_launcher(
        final_hidden_states,
        expert_outputs,
        token_indices,
        output,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE expert-output atomic scatter-add accumulation. Phase 1 "
        "memcpy final_hidden_states → output, Phase 2 atomic bf16 "
        "scatter-add over (num_selected_tokens, hidden_size). "
        "Destination-passing style.");
}
