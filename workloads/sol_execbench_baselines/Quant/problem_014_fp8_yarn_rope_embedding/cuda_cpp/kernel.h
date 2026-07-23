#pragma once

#ifndef PROBLEM_014_FP8_YARN_ROPE_EMBEDDING_KERNEL_H
#define PROBLEM_014_FP8_YARN_ROPE_EMBEDDING_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the fused FP8 YaRN-RoPE embedding kernel.
//
// Inputs:
//   position_ids   [seq_len]            int64
//   inv_freq       [HALF_DIM=32]        fp32 (precomputed on host)
//   mscale_combined                     float (precomputed on host)
// Outputs (DPS):
//   cos_emb        [seq_len, 64]        bf16
//   sin_emb        [seq_len, 64]        bf16
//
// HALF_DIM (=32) is the number of base frequencies (qk_rope_head_dim / 2).
// PAD_BLOCK (=128) is the FP8 BlockWise1x128 block size; the per-row
// frequency vector is padded with 96 zeros so the FP8 row scale is
// computed on a 128-wide vector but only the first 32 entries hold
// non-zero values, so amax over the padded row equals amax over the
// 32 real values.
void problem_014_fp8_yarn_rope_embedding_launcher(
    const torch::Tensor& position_ids,
    const float* inv_freq_host,
    float mscale_combined,
    torch::Tensor& cos_emb,
    torch::Tensor& sin_emb,
    cudaStream_t stream);

#endif  // PROBLEM_014_FP8_YARN_ROPE_EMBEDDING_KERNEL_H
