#pragma once

#ifndef PROBLEM_012_FUSED_COS_SIN_EMBEDDING_GENERATION_KERNEL_H
#define PROBLEM_012_FUSED_COS_SIN_EMBEDDING_GENERATION_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the fused cos/sin embedding generation.
// freqs is [batch_size, seq_len, half_head_dim] FP32 contiguous.
// cos / sin are pre-allocated [batch_size, seq_len, head_dim] BF16
// outputs (destination-passing style). head_dim = 2 * half_head_dim;
// the emb = cat(freqs, freqs) replication is folded into the index
// math so freqs is read only once per element.
void problem_012_fused_cos_sin_launcher(
    const torch::Tensor& freqs,
    float attention_scaling,
    torch::Tensor& cos_out,
    torch::Tensor& sin_out,
    cudaStream_t stream);

#endif  // PROBLEM_012_FUSED_COS_SIN_EMBEDDING_GENERATION_KERNEL_H
