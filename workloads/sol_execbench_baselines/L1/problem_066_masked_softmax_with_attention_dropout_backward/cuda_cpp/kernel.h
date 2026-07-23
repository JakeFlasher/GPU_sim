#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the masked-softmax + dropout backward kernel:
//   grad_softmax_output[j] = grad_output[j] * dropout_mask[j] / (1 - p_dropout)
//   sum_term = sum_j (p_attn[j] * grad_softmax_output[j])
//   grad_softmax_input[j] = p_attn[j] * (grad_softmax_output[j] - sum_term)
//   grad_scores[j] = mask[j] ? grad_softmax_input[j] : 0
// where j indexes the last (softmax) dim of length T = seq_len.
//
// Tensors:
//   grad_output  : [B, H, T, T] fp32
//   p_attn       : [B, H, T, T] fp32
//   mask         : [B, 1, T, T] bool
//   dropout_mask : [B, H, T, T] bool
//   p_dropout    : scalar fp32 (0.0 means "no dropout"; passes grad unchanged)
//   grad_scores  : [B, H, T, T] fp32 (DPS; pre-allocated)
//
// Per-row CTA topology: Grid (B, H, T); each CTA owns one row of T
// columns. No const axes are pinned by definition.json.
void problem_066_masked_softmax_with_attention_dropout_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& p_attn,
    const torch::Tensor& mask,
    const torch::Tensor& dropout_mask,
    double p_dropout,
    torch::Tensor& grad_scores,
    cudaStream_t stream);
