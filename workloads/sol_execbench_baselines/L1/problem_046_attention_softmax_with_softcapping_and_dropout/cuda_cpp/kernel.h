#pragma once

#ifndef PROBLEM_046_ATTENTION_SOFTMAX_SOFTCAP_KERNEL_H
#define PROBLEM_046_ATTENTION_SOFTMAX_SOFTCAP_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the BF16 attention-softmax-with-softcapping kernel.
//
// Input `attn_weights` is [batch_size, num_heads, seq_len_q,
// seq_len_k] BF16 contiguous; `output` is pre-allocated with the
// same shape (destination-passing style). Algorithm:
//
//   softcapped = tanh(attn_weights / 30) * 30
//   output = softmax(softcapped, dim=-1, fp32 upcast)
//
// One CTA per row over batch_size * num_heads * seq_len_q rows;
// each row spans seq_len_k columns. The reduction (max, then sum)
// runs in fp32 for numerical stability and downcasts to BF16 on
// store. num_heads=8 is pinned; softcap=30 is pinned.
void problem_046_attention_softmax_softcap_launcher(
    const torch::Tensor& attn_weights,
    torch::Tensor& output,
    cudaStream_t stream);

#endif  // PROBLEM_046_ATTENTION_SOFTMAX_SOFTCAP_KERNEL_H
