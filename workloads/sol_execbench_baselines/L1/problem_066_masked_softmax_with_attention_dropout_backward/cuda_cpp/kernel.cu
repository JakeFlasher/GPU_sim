#include "kernel.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// fp32 masked-softmax + dropout backward in one per-row CTA kernel.
//
// Algorithm (matching the reference's three-link chain):
//   (a) dropout backward:
//       if p_dropout > 0: grad_softmax_output[j] = grad_output[j]
//                         * dropout_mask[j] * (1 / (1 - p_dropout))
//       else: grad_softmax_output[j] = grad_output[j]
//   (b) softmax backward:
//       sum_term = sum_j (p_attn[j] * grad_softmax_output[j])
//       grad_softmax_input[j] = p_attn[j] * (grad_softmax_output[j] - sum_term)
//   (c) masked_fill backward:
//       grad_scores[j] = mask[j] ? grad_softmax_input[j] : 0
//
// Kernel topology:
//   * Grid (B, H, T). One CTA per (b, h, t) row.
//   * BLOCK_THREADS = 256 (8 warps of 32 lanes); stride-loop over
//     the row length T (which can exceed 256).
//   * Each thread accumulates a fp32 partial sum across its stride
//     of (b, h, t, j). Warp-shuffle reduce within each warp; cross-
//     warp tree reduce via shared memory; thread 0 broadcasts the
//     final sum_term back through shared memory.
//   * Pass 1 reads grad_output[j] + dropout_mask[j] + p_attn[j]
//     and accumulates partial sum; pass 2 reads grad_output[j] +
//     dropout_mask[j] + p_attn[j] + mask[j] again to compute the
//     final grad_scores[j]. The two passes are necessary because
//     sum_term must be known before computing grad_scores[j].
//
// Note: mask is `[B, 1, T, T]`, which broadcasts across the H axis.
// The mask offset for (b, h, t, j) is `(b * 1 + 0) * T * T + t * T + j`
// = `b * T * T + t * T + j`.

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;  // 8

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__global__ void __launch_bounds__(kBlockThreads, 4) k_masked_softmax_dropout_bwd(
    const float* __restrict__ grad_output,
    const float* __restrict__ p_attn,
    const bool* __restrict__ mask,
    const bool* __restrict__ dropout_mask,
    float inv_keep_prob,        // 1.0 / (1.0 - p_dropout) if p_dropout > 0 else 1.0
    bool dropout_active,
    float* __restrict__ grad_scores,
    int H,
    int T
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    const int tid = threadIdx.x;

    const size_t row_offset = ((size_t)b * H + h) * T * T + (size_t)t * T;
    const size_t mask_row_offset = ((size_t)b * T + t) * T;  // mask is [B, 1, T, T]

    const float* __restrict__ grad_out_row = grad_output + row_offset;
    const float* __restrict__ p_attn_row = p_attn + row_offset;
    const bool* __restrict__ drop_row = dropout_mask + row_offset;
    const bool* __restrict__ mask_row = mask + mask_row_offset;
    float* __restrict__ grad_score_row = grad_scores + row_offset;

    float partial = 0.0f;
    for (int j = tid; j < T; j += kBlockThreads) {
        float gso = grad_out_row[j];
        if (dropout_active) {
            gso *= (drop_row[j] ? inv_keep_prob : 0.0f);
        }
        partial += p_attn_row[j] * gso;
    }

    partial = warp_reduce_sum(partial);

    __shared__ float warp_sums[kNumWarps];
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    if (lane == 0) {
        warp_sums[warp_id] = partial;
    }
    __syncthreads();

    __shared__ float s_sum_term;
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < kNumWarps; ++w) {
            total += warp_sums[w];
        }
        s_sum_term = total;
    }
    __syncthreads();

    const float sum_term = s_sum_term;

    for (int j = tid; j < T; j += kBlockThreads) {
        float gso = grad_out_row[j];
        if (dropout_active) {
            gso *= (drop_row[j] ? inv_keep_prob : 0.0f);
        }
        const float grad_softmax_input = p_attn_row[j] * (gso - sum_term);
        grad_score_row[j] = mask_row[j] ? grad_softmax_input : 0.0f;
    }
}

}  // namespace

void problem_066_masked_softmax_with_attention_dropout_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& p_attn,
    const torch::Tensor& mask,
    const torch::Tensor& dropout_mask,
    double p_dropout,
    torch::Tensor& grad_scores,
    cudaStream_t stream
) {
    const int B = static_cast<int>(grad_output.size(0));
    const int H = static_cast<int>(grad_output.size(1));
    const int T = static_cast<int>(grad_output.size(2));

    const float p_drop_f32 = static_cast<float>(p_dropout);
    const bool dropout_active = p_drop_f32 > 0.0f;
    const float inv_keep_prob = dropout_active ? (1.0f / (1.0f - p_drop_f32)) : 1.0f;

    dim3 grid(B, H, T);
    dim3 block(kBlockThreads);

    k_masked_softmax_dropout_bwd<<<grid, block, 0, stream>>>(
        grad_output.data_ptr<float>(),
        p_attn.data_ptr<float>(),
        mask.data_ptr<bool>(),
        dropout_mask.data_ptr<bool>(),
        inv_keep_prob,
        dropout_active,
        grad_scores.data_ptr<float>(),
        H, T
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "problem_066 masked-softmax dropout backward kernel launch failed");
}
