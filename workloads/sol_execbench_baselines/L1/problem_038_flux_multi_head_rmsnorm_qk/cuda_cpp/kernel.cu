#include "kernel.h"
#include <cuda_runtime.h>

// fp32 per-head RMSNorm applied to Q (or K) attention tensor.
//
//   var[b, s, h] = mean_d(x[b, s, h, d]^2)
//   x_norm[..., d] = x[..., d] * rsqrt(var + eps) * weight[h, d]
//
// num_heads = 48 and head_dim = 128 (both pinned by the problem
// definition). One CTA per (b, s, h) row of 128 elements;
// BLOCK_THREADS = head_dim = 128 so each thread handles exactly one
// element. The per-head weight table is indexed by (h, d), so the
// (B, S) row CTAs at a fixed `h` all share the same weight slice
// — strong inter-CTA reuse on the weight loads.

namespace {

constexpr int kHeadDim = 128;
constexpr int kBlockThreads = kHeadDim;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;  // 4

__global__ void __launch_bounds__(kBlockThreads, 8)
problem_038_rmsnorm_qk_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ x_norm,
    int num_heads,
    int seq_len,
    float eps) {

    // Grid layout: (h, s, B). Each CTA handles one (b, s, h) row of
    // head_dim elements.
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int d = threadIdx.x;

    const int row_offset =
        ((b * seq_len + s) * num_heads + h) * kHeadDim;
    const int weight_offset = h * kHeadDim;

    const float xv = x[row_offset + d];
    const float wv = weight[weight_offset + d];

    // Warp-shuffle reduction of xv^2 across 32 lanes.
    float sumsq = xv * xv;
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
    }

    // Cross-warp reduction via shared memory: lane 0 of each warp
    // contributes one partial sum; thread 0 finalises.
    __shared__ float warp_sums[kNumWarps];
    const int warp_id = d / kWarpSize;
    const int lane_id = d % kWarpSize;
    if (lane_id == 0) {
        warp_sums[warp_id] = sumsq;
    }
    __syncthreads();

    __shared__ float row_inv_rms;
    if (d == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < kNumWarps; ++w) {
            total += warp_sums[w];
        }
        const float mean_sq = total / static_cast<float>(kHeadDim);
        row_inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    x_norm[row_offset + d] = xv * row_inv_rms * wv;
}

}  // namespace

void problem_038_flux_multi_head_rmsnorm_qk_launcher(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    torch::Tensor& x_norm,
    float eps,
    cudaStream_t stream) {

    // x is [B, S, num_heads, head_dim] fp32.
    const int batch_size = static_cast<int>(x.size(0));
    const int seq_len = static_cast<int>(x.size(1));
    const int num_heads = static_cast<int>(x.size(2));

    dim3 grid(num_heads, seq_len, batch_size);
    dim3 block(kBlockThreads);

    problem_038_rmsnorm_qk_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        x_norm.data_ptr<float>(),
        num_heads,
        seq_len,
        eps);
}
