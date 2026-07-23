#include "kernel.h"
#include <cuda_runtime.h>

// fp32 GroupNorm for Stable Diffusion UNet.
//
// Reduces over (channels_per_group, H, W) within each of num_groups = 32
// channel groups; applies a per-channel affine transform
// (weight[c], bias[c]) to the normalized output. Two-pass kernel:
//   Pass 1: stride-loop over the group's N = (C/G)*H*W elements;
//           each thread accumulates partial sum + partial sum-of-squares
//           in fp32; warp-shuffle + cross-warp shared-memory tree
//           reduce to thread 0; thread 0 computes mean + inv_stddev;
//           broadcasts both through shared memory.
//   Pass 2: stride-loop again over the same N elements; each thread
//           re-reads x, looks up weight[c] / bias[c] for its current
//           channel, and writes (x - mean) * inv_stddev * weight[c]
//           + bias[c] to the output.
//
// num_groups = 32 is pinned by the problem definition.

namespace {

constexpr int kNumGroups = 32;
constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;  // 8

__global__ void __launch_bounds__(kBlockThreads, 4)
problem_078_group_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int num_channels,
    int height,
    int width,
    float eps) {

    const int g = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;

    const int channels_per_group = num_channels / kNumGroups;
    const int hw = height * width;
    const int group_count = channels_per_group * hw;

    // Per-batch base offsets.
    const int batch_offset = b * num_channels * hw;
    const int group_base_channel = g * channels_per_group;
    const int group_offset = batch_offset + group_base_channel * hw;

    // Pass 1: sum + sum-of-squares reduction across the group's
    // (channels_per_group, H, W) tile via a stride-loop.
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < group_count; i += kBlockThreads) {
        const float v = x[group_offset + i];
        thread_sum += v;
        thread_sum_sq += v * v;
    }

    // Warp-shuffle reduction of (sum, sum_sq) across 32 lanes.
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

    __shared__ float row_mean;
    __shared__ float row_inv_stddev;
    if (tid == 0) {
        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;
        #pragma unroll
        for (int w = 0; w < kNumWarps; ++w) {
            total_sum    += warp_sums[w];
            total_sum_sq += warp_sum_sqs[w];
        }
        const float n_f = static_cast<float>(group_count);
        const float mean = total_sum / n_f;
        const float var = total_sum_sq / n_f - mean * mean;
        row_mean = mean;
        row_inv_stddev = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = row_mean;
    const float inv_stddev = row_inv_stddev;

    // Pass 2: normalize, affine, write output.
    for (int i = tid; i < group_count; i += kBlockThreads) {
        const int c_in_group = i / hw;
        const int c = group_base_channel + c_in_group;
        const float v = x[group_offset + i];
        const float wv = weight[c];
        const float bv = bias[c];
        output[group_offset + i] =
            (v - mean) * inv_stddev * wv + bv;
    }
}

}  // namespace

void problem_078_group_norm_fusion_launcher(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    float eps,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(x.size(0));
    const int num_channels = static_cast<int>(x.size(1));
    const int height = static_cast<int>(x.size(2));
    const int width = static_cast<int>(x.size(3));

    dim3 grid(kNumGroups, batch_size);
    dim3 block(kBlockThreads);

    problem_078_group_norm_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_channels,
        height,
        width,
        eps);
}
