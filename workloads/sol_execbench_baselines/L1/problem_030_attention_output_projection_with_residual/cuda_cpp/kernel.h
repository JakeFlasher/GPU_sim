#pragma once

#ifndef PROBLEM_030_ATTN_OUTPUT_PROJ_RESIDUAL_KERNEL_H
#define PROBLEM_030_ATTN_OUTPUT_PROJ_RESIDUAL_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the fused BF16 attention output projection +
// residual add kernel.
//
//   output = (attn_output @ o_proj_weight.T) + residual
//
// All BF16 contiguous. attn_output is [B, S, H]; o_proj_weight is
// [H, H]; residual is [B, S, H]; output (DPS) is [B, S, H].
// hidden_size H=2560 is pinned by the problem definition.
//
// Implemented as a 2D GEMM over the flattened M = B*S row dim with
// the CUTLASS Sm80 TensorOp BF16 template; the residual is passed
// as the C source so the epilogue does `D = α·(A·B) + β·C` with
// α=1, β=1.
void problem_030_attn_output_proj_residual_launcher(
    const torch::Tensor& attn_output,
    const torch::Tensor& residual,
    const torch::Tensor& o_proj_weight,
    torch::Tensor& output,
    cudaStream_t stream);

#endif  // PROBLEM_030_ATTN_OUTPUT_PROJ_RESIDUAL_KERNEL_H
