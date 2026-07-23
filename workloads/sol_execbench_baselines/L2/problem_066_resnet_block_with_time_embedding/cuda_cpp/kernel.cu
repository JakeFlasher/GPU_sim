#include "kernel.h"
#include <cuda_runtime.h>

// Stable Diffusion ResNet block with time-embedding injection.
// All shape constants are pinned by the upstream definition.json:
//
//   in_channels=320, out_channels=320, time_emb_channels=1280,
//   norm_num_groups=32, conv kernel_size=3, padding=1.
//
// Layout: tensors carry [batch, channels, height, width] in
// channels-first contiguous order. GroupNorm groups the 320
// channels into 32 groups of 10 channels each; the per-group
// mean + variance are computed over (channels_per_group, H, W).
//
// Pipeline (4 sequential __global__ kernel launches on the
// default stream so phase i+1 sees phase i's writes; K1 and K3
// reuse the same groupnorm_silu_kernel symbol with different
// per-launch arguments):
//
//   K1 (groupnorm_silu over x):  h1[b, c, hw] = silu(groupnorm(x; norm1_*))
//   K2 (conv1_with_time):        h2[b, oc, hw] = conv2d(h1; conv1_*) +
//                                    time_proj_bcast[b, oc]
//   K3 (groupnorm_silu over h2): h3[b, c, hw] = silu(groupnorm(h2; norm2_*))
//   K4 (conv2_plus_residual):    output[b, oc, hw] = conv2d(h3; conv2_*)
//                                    + x[b, oc, hw]
//
// time projection (silu(time_emb) @ proj.T + bias) is computed
// inside K2 by each thread for its own (b, oc) destination
// element (the projection is small: 1280 -> 320 dense matmul per
// (b, oc); the entire time_emb row is reloaded per thread but
// the reuse pattern is cache-friendly on a single CTA).

namespace {

constexpr int IN_CHANNELS = 320;
constexpr int OUT_CHANNELS = 320;
constexpr int TIME_EMB_CHANNELS = 1280;
constexpr int NUM_GROUPS = 32;
constexpr int CHANNELS_PER_GROUP = IN_CHANNELS / NUM_GROUPS;  // 10
constexpr int KERNEL_SIZE = 3;
constexpr int PAD = 1;
constexpr int BLOCK_THREADS = 256;

__device__ inline float silu(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ inline float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

// K1: GroupNorm + SiLU on the input tensor. One CTA per
// (batch, group); the CTA owns CHANNELS_PER_GROUP * spatial
// elements and computes the group mean / variance via warp +
// shared-memory reduction, then writes silu(normalized) into the
// h1 buffer.
__global__ void __launch_bounds__(BLOCK_THREADS, 4)
problem_066_resnet_block_with_time_embedding_groupnorm_silu_kernel(
    const float* __restrict__ src,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    float* __restrict__ dst,
    int batch_size,
    int height,
    int width,
    float norm_eps) {

    const int b = blockIdx.x;
    const int g = blockIdx.y;
    const int spatial = height * width;
    const int group_elements = CHANNELS_PER_GROUP * spatial;
    const int tid = threadIdx.x;

    // Per-CTA pointer into the (b, group) channel block.
    const int channel_base = g * CHANNELS_PER_GROUP;
    const float* src_g =
        src + (b * IN_CHANNELS + channel_base) * spatial;

    // Pass 1: sum and sum-of-squares over the group.
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < group_elements; i += BLOCK_THREADS) {
        const float v = __ldg(src_g + i);
        sum += v;
        sum_sq += v * v;
    }
    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);

    constexpr int NUM_WARPS = BLOCK_THREADS / 32;  // 8
    __shared__ float s_sum[NUM_WARPS];
    __shared__ float s_sum_sq[NUM_WARPS];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        s_sum[warp] = sum;
        s_sum_sq[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float w_sum = (lane < NUM_WARPS) ? s_sum[lane] : 0.0f;
        float w_sum_sq = (lane < NUM_WARPS) ? s_sum_sq[lane] : 0.0f;
        w_sum = warp_reduce_sum(w_sum);
        w_sum_sq = warp_reduce_sum(w_sum_sq);
        if (lane == 0) {
            s_sum[0] = w_sum;
            s_sum_sq[0] = w_sum_sq;
        }
    }
    __syncthreads();
    const float inv_n = 1.0f / static_cast<float>(group_elements);
    const float mean = s_sum[0] * inv_n;
    const float var = s_sum_sq[0] * inv_n - mean * mean;
    const float inv_std = rsqrtf(var + norm_eps);

    // Pass 2: apply per-channel affine, SiLU.
    float* dst_g = dst + (b * IN_CHANNELS + channel_base) * spatial;
    for (int i = tid; i < group_elements; i += BLOCK_THREADS) {
        const int local_c = i / spatial;
        const int c = channel_base + local_c;
        const float v = __ldg(src_g + i);
        const float normalized = (v - mean) * inv_std;
        const float scaled = normalized * __ldg(norm_weight + c)
            + __ldg(norm_bias + c);
        dst_g[i] = silu(scaled);
    }
}

// Helper: project time_emb + add bias for a given (b, oc). The
// time-projection is silu(time_emb[b]) @ time_emb_proj_weight[oc, :].T
// + time_emb_proj_bias[oc].
__device__ inline float compute_time_proj_element(
    int b,
    int oc,
    const float* __restrict__ time_emb,
    const float* __restrict__ time_proj_weight,
    const float* __restrict__ time_proj_bias) {
    float acc = __ldg(time_proj_bias + oc);
    const float* time_row = time_emb + b * TIME_EMB_CHANNELS;
    const float* weight_row =
        time_proj_weight + oc * TIME_EMB_CHANNELS;
    #pragma unroll 4
    for (int k = 0; k < TIME_EMB_CHANNELS; ++k) {
        const float te = __ldg(time_row + k);
        const float silu_te = te / (1.0f + __expf(-te));
        acc += silu_te * __ldg(weight_row + k);
    }
    return acc;
}

// K2: conv1 (in=320, out=320, k=3, p=1) + time-projection
// broadcast add. One thread per (b, oc, h, w) output. Each
// thread independently computes the time-projection element for
// its (b, oc) -- naive (recomputed per spatial position) for
// simplicity; the 1280-deep dot product per thread is fine for
// the workload sizes in the spec (largest spatial 131*131 ~= 17K
// threads per (b, oc) channel; per-thread cost ~= 1280 muladds).
__global__ void __launch_bounds__(BLOCK_THREADS, 4)
problem_066_resnet_block_with_time_embedding_conv1_with_time_kernel(
    const float* __restrict__ src,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ time_emb,
    const float* __restrict__ time_proj_weight,
    const float* __restrict__ time_proj_bias,
    float* __restrict__ dst,
    int batch_size,
    int height,
    int width) {

    const int linear = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    const int spatial = height * width;
    const int total = batch_size * OUT_CHANNELS * spatial;
    if (linear >= total) return;

    const int hw = linear % spatial;
    const int boc = linear / spatial;
    const int oc = boc % OUT_CHANNELS;
    const int b = boc / OUT_CHANNELS;
    const int h = hw / width;
    const int w = hw % width;

    const float* src_b = src + b * (IN_CHANNELS * spatial);
    float acc = __ldg(conv_bias + oc);

    #pragma unroll
    for (int ic = 0; ic < IN_CHANNELS; ++ic) {
        const float* src_ic = src_b + ic * spatial;
        #pragma unroll
        for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
            const int ih = h + ki - PAD;
            if (ih < 0 || ih >= height) continue;
            #pragma unroll
            for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                const int iw = w + kj - PAD;
                if (iw < 0 || iw >= width) continue;
                const float wv = __ldg(
                    conv_weight
                    + ((oc * IN_CHANNELS + ic) * KERNEL_SIZE + ki)
                        * KERNEL_SIZE + kj);
                const float xv = __ldg(src_ic + ih * width + iw);
                acc += wv * xv;
            }
        }
    }
    // Broadcast time-projection (b, oc) is constant across hw;
    // we recompute it per-thread because the per-thread cost
    // (1280 muladds) is small relative to the conv cost
    // (320 * 9 = 2880 muladds per output) and the access
    // pattern is cache-friendly.
    const float t = compute_time_proj_element(
        b, oc, time_emb, time_proj_weight, time_proj_bias);
    dst[linear] = acc + t;
}

// K3: conv2 (320 -> 320, k=3, p=1) + residual add against the
// ORIGINAL x (which the reference adds back at the end). Same
// shape as K2 but no time-emb; reads from the h3 (post-second-
// groupnorm-silu) buffer and the original x.
__global__ void __launch_bounds__(BLOCK_THREADS, 4)
problem_066_resnet_block_with_time_embedding_conv2_plus_residual_kernel(
    const float* __restrict__ src_h3,
    const float* __restrict__ residual_x,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    float* __restrict__ dst,
    int batch_size,
    int height,
    int width) {

    const int linear = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    const int spatial = height * width;
    const int total = batch_size * OUT_CHANNELS * spatial;
    if (linear >= total) return;

    const int hw = linear % spatial;
    const int boc = linear / spatial;
    const int oc = boc % OUT_CHANNELS;
    const int b = boc / OUT_CHANNELS;
    const int h = hw / width;
    const int w = hw % width;

    const float* src_b = src_h3 + b * (IN_CHANNELS * spatial);
    float acc = __ldg(conv_bias + oc);

    #pragma unroll
    for (int ic = 0; ic < IN_CHANNELS; ++ic) {
        const float* src_ic = src_b + ic * spatial;
        #pragma unroll
        for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
            const int ih = h + ki - PAD;
            if (ih < 0 || ih >= height) continue;
            #pragma unroll
            for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                const int iw = w + kj - PAD;
                if (iw < 0 || iw >= width) continue;
                const float wv = __ldg(
                    conv_weight
                    + ((oc * IN_CHANNELS + ic) * KERNEL_SIZE + ki)
                        * KERNEL_SIZE + kj);
                const float xv = __ldg(src_ic + ih * width + iw);
                acc += wv * xv;
            }
        }
    }
    const float resv = __ldg(residual_x + linear);
    dst[linear] = acc + resv;
}

}  // namespace

void problem_066_resnet_block_with_time_embedding_launcher(
    const torch::Tensor& x,
    const torch::Tensor& time_emb,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& conv1_weight,
    const torch::Tensor& conv1_bias,
    const torch::Tensor& time_emb_proj_weight,
    const torch::Tensor& time_emb_proj_bias,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    const torch::Tensor& conv2_weight,
    const torch::Tensor& conv2_bias,
    float norm_eps,
    torch::Tensor& output,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(x.size(0));
    const int height = static_cast<int>(x.size(2));
    const int width = static_cast<int>(x.size(3));

    auto h1 = torch::empty_like(x);
    auto h2 = torch::empty_like(x);
    auto h3 = torch::empty_like(x);

    // K1: GroupNorm + SiLU on x -> h1.
    {
        dim3 grid(batch_size, NUM_GROUPS);
        dim3 block(BLOCK_THREADS);
        problem_066_resnet_block_with_time_embedding_groupnorm_silu_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            norm1_weight.data_ptr<float>(),
            norm1_bias.data_ptr<float>(),
            h1.data_ptr<float>(),
            batch_size,
            height,
            width,
            norm_eps);
    }

    // K2: conv1 + broadcast-add time projection -> h2.
    {
        const int total = batch_size * OUT_CHANNELS * height * width;
        const int blocks = (total + BLOCK_THREADS - 1) / BLOCK_THREADS;
        problem_066_resnet_block_with_time_embedding_conv1_with_time_kernel<<<blocks, BLOCK_THREADS, 0, stream>>>(
            h1.data_ptr<float>(),
            conv1_weight.data_ptr<float>(),
            conv1_bias.data_ptr<float>(),
            time_emb.data_ptr<float>(),
            time_emb_proj_weight.data_ptr<float>(),
            time_emb_proj_bias.data_ptr<float>(),
            h2.data_ptr<float>(),
            batch_size,
            height,
            width);
    }

    // K3: GroupNorm + SiLU on h2 -> h3.
    {
        dim3 grid(batch_size, NUM_GROUPS);
        dim3 block(BLOCK_THREADS);
        problem_066_resnet_block_with_time_embedding_groupnorm_silu_kernel<<<grid, block, 0, stream>>>(
            h2.data_ptr<float>(),
            norm2_weight.data_ptr<float>(),
            norm2_bias.data_ptr<float>(),
            h3.data_ptr<float>(),
            batch_size,
            height,
            width,
            norm_eps);
    }

    // K4: conv2 + residual add -> output.
    {
        const int total = batch_size * OUT_CHANNELS * height * width;
        const int blocks = (total + BLOCK_THREADS - 1) / BLOCK_THREADS;
        problem_066_resnet_block_with_time_embedding_conv2_plus_residual_kernel<<<blocks, BLOCK_THREADS, 0, stream>>>(
            h3.data_ptr<float>(),
            x.data_ptr<float>(),
            conv2_weight.data_ptr<float>(),
            conv2_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            height,
            width);
    }
}
