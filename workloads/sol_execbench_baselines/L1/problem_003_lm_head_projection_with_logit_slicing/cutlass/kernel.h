#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the language-model head projection GEMM:
//   logits = hidden_states_slice @ weight^T
//
// hidden_states_slice: [M, K=2048] bf16 row-major
//                      (flattened from [B, logits_to_keep, hidden_size]
//                      with M = B * logits_to_keep; all canonical
//                      workloads have logits_to_keep == seq_len so the
//                      slice is a no-op).
// weight:              [N=102400, K=2048] bf16 row-major.
// logits:              [M, N=102400] bf16 row-major (DPS;
//                      pre-allocated by the eval driver).
//
// N = vocab_size = 102400 and K = hidden_size = 2048 are pinned by
// the problem definition; M = B * logits_to_keep varies per workload.
// 102400 mod 8 = 0, so AlignmentC = 8 (default) is correct.
void problem_003_lm_head_projection_with_logit_slicing_launcher(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& weight,
    torch::Tensor& logits_2d,
    cudaStream_t stream);
