#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// FLUX dual-stream linear projection. The reference concatenates the
// encoder + image streams along the sequence dimension, applies a
// single linear projection, and slices the result back into two
// streams. This launcher uses the per-stream strategy: two direct
// at::matmul_out calls, one per output stream, sharing the
// transposed weight view. Avoids the reference's concat + slice
// memcpy chain at the cost of one additional cuBLAS launch.
void problem_030_flux_concat_split_launcher(
    const torch::Tensor& hidden_states,          // [B, img_seq_len, 3072]
    const torch::Tensor& encoder_hidden_states,  // [B, text_seq_len, 3072]
    const torch::Tensor& process_weight,         // [3072, 3072]
    torch::Tensor& processed_encoder,            // [B, text_seq_len, 3072] DPS
    torch::Tensor& processed_hidden,             // [B, img_seq_len, 3072] DPS
    cudaStream_t stream);
