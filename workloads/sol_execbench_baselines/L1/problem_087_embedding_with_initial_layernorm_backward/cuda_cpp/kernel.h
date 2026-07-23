#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the embedding+RMSNorm backward pipeline.
// Pipeline (one fused per-row kernel + two cast kernels):
//
//   Kernel 1 (per (batch, seq) row):
//     a. RMSNorm-backward chain-rule on the row's 4096-element
//        hidden vector; reduce mean(grad_h_norm * h_norm) in
//        shared memory.
//     b. atomicAdd grad_h_fp32 into grad_embed_weight_fp32 at the
//        token's vocab row.
//     c. atomicAdd grad_out * h_norm into grad_norm_weight_fp32.
//
//   Kernel 2: cast grad_embed_weight_fp32 (vocab*hidden FP32) to
//             BF16 grad_embed_weight (DPS output).
//
//   Kernel 3: cast grad_norm_weight_fp32 (hidden FP32) to BF16
//             grad_norm_weight (DPS output).
//
// FP32 accumulators internally; BF16 DPS outputs match the
// definition.json contract.
void problem_087_embedding_layernorm_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_ids,
    const torch::Tensor& hidden_states_fp32,
    const torch::Tensor& rstd,
    const torch::Tensor& norm_weight,
    torch::Tensor& grad_embed_weight,
    torch::Tensor& grad_norm_weight,
    cudaStream_t stream);
