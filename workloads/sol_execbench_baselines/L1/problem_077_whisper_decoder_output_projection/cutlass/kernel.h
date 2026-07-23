#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the Whisper decoder output projection GEMM:
//   logits = hidden_states @ weight^T
//
// hidden_states: [M, K=1280] fp16 row-major (flattened from
//                [B, S, d_model] with M = B*S).
// weight:        [N=51866, K=1280] fp16 row-major.
// logits:        [M, N=51866] fp16 row-major
//                (destination-passing-style; pre-allocated by the
//                eval driver, flattened from [B, S, vocab_size]).
//
// N = vocab_size = 51866 and K = d_model = 1280 are pinned by the
// problem definition; M varies per workload. N=51866 is NOT 8-element
// aligned (51866 mod 8 = 2), so the epilogue uses AlignmentC = 2
// instead of the default 8.
void problem_077_whisper_decoder_output_projection_launcher(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& weight,
    torch::Tensor& logits_2d,
    cudaStream_t stream);
