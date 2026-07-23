#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the MoE atomic-accumulation kernel.
// final_hidden_states: [batch_seq_len, hidden_size] bf16 — copied to output.
// expert_outputs:      [num_selected_tokens, hidden_size] bf16 — atomically
//                      added back into output[token_indices[i]] per row.
// token_indices:       [num_selected_tokens] int64.
// output:              [batch_seq_len, hidden_size] bf16 (DPS, pre-allocated).
// hidden_size is pinned at 3072 by the problem definition.
void problem_008_expert_output_weighted_index_add_accumulation_launcher(
    const torch::Tensor& final_hidden_states,
    const torch::Tensor& expert_outputs,
    const torch::Tensor& token_indices,
    torch::Tensor& output,
    cudaStream_t stream);
