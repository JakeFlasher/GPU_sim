#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// BF16 Gaussian top-k sparse activation.
//
// Per (b, s) row of intermediate_size BF16 elements:
//   Pass 1 (reduction): each thread stride-loops over the row,
//     reads BF16x2 pairs, casts to fp32, accumulates partial sum
//     and partial sum-of-squares. Warp-shuffle reduces within each
//     warp; 8-way cross-warp reduction via shared memory. Thread 0
//     computes mean = sum/N, var = sum_sq/N - mean^2,
//     std = sqrt(max(var, 0)), cutoff = mean + std * z_multiplier,
//     and broadcasts `cutoff` through a shared float.
//   Pass 2 (output): each thread stride-loops again, reads BF16x2
//     pairs as fp32, computes max(x - cutoff, 0.0f), casts back
//     to BF16 and stores as BF16x2 pairs.
//
// One CTA per (b, s) row. Grid: (batch_size * seq_len,).
// BLOCK_THREADS = 256 (8 warps).
//
// intermediate_size is workload-variable (4096, 8192, 12288,
// 16384 in the workload sweep). The kernel handles any even
// intermediate_size via BF16x2 strides.

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;  // 8

__global__ void __launch_bounds__(kBlockThreads, 4)
problem_053_gaussian_topk_sparse_kernel(
    const __nv_bfloat16* __restrict__ inputs,
    __nv_bfloat16* __restrict__ output,
    int intermediate_size,
    float z_multiplier) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = row * intermediate_size;

    // BF16x2 view over the row. intermediate_size is even (every
    // workload uses a power-of-two times something divisible by 2).
    const __nv_bfloat162* in_row_vec =
        reinterpret_cast<const __nv_bfloat162*>(inputs + row_offset);
    __nv_bfloat162* out_row_vec =
        reinterpret_cast<__nv_bfloat162*>(output + row_offset);
    const int row_pairs = intermediate_size >> 1;

    // Pass 1: partial sum + partial sum-of-squares in fp32.
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < row_pairs; i += kBlockThreads) {
        const __nv_bfloat162 v_pair = __ldg(in_row_vec + i);
        const float2 v_f2 = __bfloat1622float2(v_pair);
        thread_sum    += v_f2.x + v_f2.y;
        thread_sum_sq += v_f2.x * v_f2.x + v_f2.y * v_f2.y;
    }

    // Warp-shuffle reduction.
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        thread_sum    += __shfl_xor_sync(0xffffffff, thread_sum, off);
        thread_sum_sq += __shfl_xor_sync(0xffffffff, thread_sum_sq, off);
    }

    // Cross-warp reduction via shared memory.
    __shared__ float warp_sums[kNumWarps];
    __shared__ float warp_sum_sqs[kNumWarps];
    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
        warp_sum_sqs[warp_id] = thread_sum_sq;
    }
    __syncthreads();

    __shared__ float row_cutoff;
    if (tid == 0) {
        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;
        #pragma unroll
        for (int w = 0; w < kNumWarps; ++w) {
            total_sum    += warp_sums[w];
            total_sum_sq += warp_sum_sqs[w];
        }
        const float n_f = static_cast<float>(intermediate_size);
        const float mean = total_sum / n_f;
        const float var = fmaxf(total_sum_sq / n_f - mean * mean, 0.0f);
        const float std_val = sqrtf(var);
        row_cutoff = mean + std_val * z_multiplier;
    }
    __syncthreads();

    const float cutoff = row_cutoff;

    // Pass 2: per-element max(x - cutoff, 0) in fp32; cast back to BF16.
    for (int i = tid; i < row_pairs; i += kBlockThreads) {
        const __nv_bfloat162 v_pair = __ldg(in_row_vec + i);
        const float2 v_f2 = __bfloat1622float2(v_pair);
        const float y0 = fmaxf(v_f2.x - cutoff, 0.0f);
        const float y1 = fmaxf(v_f2.y - cutoff, 0.0f);
        out_row_vec[i] = __float22bfloat162_rn(make_float2(y0, y1));
    }
}

}  // namespace

void problem_053_gaussian_topk_sparse_activation_launcher(
    const torch::Tensor& inputs,
    float z_multiplier,
    torch::Tensor& output,
    cudaStream_t stream) {

    // inputs is [B, S, intermediate_size] BF16.
    const int64_t batch_size = inputs.size(0);
    const int64_t seq_len = inputs.size(1);
    const int64_t intermediate_size = inputs.size(2);
    const int rows = static_cast<int>(batch_size * seq_len);
    const int int_sz = static_cast<int>(intermediate_size);

    dim3 grid(rows);
    dim3 block(kBlockThreads);

    problem_053_gaussian_topk_sparse_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(inputs.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        int_sz,
        z_multiplier);
}
