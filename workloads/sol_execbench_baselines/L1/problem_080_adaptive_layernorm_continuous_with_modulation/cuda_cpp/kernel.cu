#include "kernel.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Two-kernel chain for fp32 AdaLayerNormContinuous with timestep-
// conditioned modulation.
//
//   Kernel 1 — small-M Linear (temb -> modulation[B, 4864])
//     Per-(b, n_tile) CTA; each warp computes one output element
//     via K = inner_dim = 2432 reduction over `temb[b, :]`; lane 0
//     writes `partial + linear_bias[n]` to `modulation[b, n]`.
//     Uses the small-M Linear pattern with shared-memory timestep
//     embedding cache and per-warp K-reduction.
//
//   Kernel 2 — LN + modulation per-(b, s) row
//     Per-(b, s) CTA over Grid `(B, S)`; BLOCK_THREADS = 256
//     stride-loops over `inner_dim`. Pass A: each thread
//     accumulates per-row partial sum and partial sum-of-squares
//     of `hidden_states[b, s, :]`; warp-shuffle + cross-warp tree
//     reduces to `mean` and `var`. Pass B: each thread `d` reads
//     `hidden_states[b, s, d]`, `modulation[b, d]` (scale), and
//     `modulation[b, d + inner_dim]` (shift); writes
//     `output[b, s, d] = ((hidden_states - mean) * rstd) *
//     (1 + scale) + shift`.
//
// Both passes use per-row reductions only (within a single CTA);
// no cross-CTA reduction tree avoids accumulation-order drift
// against PyTorch's reduction order.

namespace {

constexpr int kInnerDim = 2432;
constexpr int kInnerDimX2 = 2 * kInnerDim;  // 4864
constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;  // 8
constexpr int kWarpsPerCtaLinear = kNumWarps;
constexpr int kNTilesPerCtaLinear = kWarpsPerCtaLinear;

static_assert(kInnerDimX2 % kWarpsPerCtaLinear == 0,
              "kInnerDimX2 must be divisible by kWarpsPerCtaLinear");

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(
    float v, float* warp_sums
) {
    v = warp_reduce_sum(v);
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    if (lane == 0) warp_sums[warp_id] = v;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x < kNumWarps) {
        total = warp_sums[threadIdx.x];
    }
    total = warp_reduce_sum(total);
    return total;
}

// Kernel 1: small-M Linear. Grid (B, N_tiles = kInnerDimX2 / 8 = 608).
__global__ void __launch_bounds__(kBlockThreads, 4) k_small_m_linear(
    const float* __restrict__ temb,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ modulation
) {
    const int b = blockIdx.x;
    const int n_tile_base = blockIdx.y * kWarpsPerCtaLinear;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);

    __shared__ float s_temb[kInnerDim];
    #pragma unroll 4
    for (int k = tid; k < kInnerDim; k += kBlockThreads) {
        s_temb[k] = temb[b * kInnerDim + k];
    }
    __syncthreads();

    const int n = n_tile_base + warp_id;

    float partial = 0.0f;
    const float* w_row = weight + (size_t)n * kInnerDim;
    for (int k = lane; k < kInnerDim; k += kWarpSize) {
        partial += w_row[k] * s_temb[k];
    }
    partial = warp_reduce_sum(partial);

    if (lane == 0) {
        modulation[b * kInnerDimX2 + n] = partial + bias[n];
    }
}

// Kernel 2: LN + modulation. Grid (B, S).
__global__ void __launch_bounds__(kBlockThreads, 4)
    k_layernorm_with_modulation(
        const float* __restrict__ hidden_states,
        const float* __restrict__ modulation,
        float eps,
        float* __restrict__ output,
        int S
) {
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t row_offset = ((size_t)b * S + s) * kInnerDim;
    const float* __restrict__ x_row = hidden_states + row_offset;
    float* __restrict__ out_row = output + row_offset;
    const float* __restrict__ scale_row = modulation + (size_t)b * kInnerDimX2;
    const float* __restrict__ shift_row =
        modulation + (size_t)b * kInnerDimX2 + kInnerDim;

    // Pass A: compute mean and variance.
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int d = tid; d < kInnerDim; d += kBlockThreads) {
        const float x = x_row[d];
        sum += x;
        sum_sq += x * x;
    }

    __shared__ float warp_sums[kNumWarps];
    sum = block_reduce_sum(sum, warp_sums);
    __syncthreads();
    sum_sq = block_reduce_sum(sum_sq, warp_sums);

    __shared__ float s_mean;
    __shared__ float s_rstd;
    if (tid == 0) {
        const float mean = sum / (float)kInnerDim;
        const float var = sum_sq / (float)kInnerDim - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = s_mean;
    const float rstd = s_rstd;

    // Pass B: write output with modulation.
    for (int d = tid; d < kInnerDim; d += kBlockThreads) {
        const float normalized = (x_row[d] - mean) * rstd;
        const float scale = scale_row[d];
        const float shift = shift_row[d];
        out_row[d] = normalized * (1.0f + scale) + shift;
    }
}

}  // namespace

void problem_080_adaptive_layernorm_continuous_with_modulation_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& temb,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream
) {
    const int B = static_cast<int>(hidden_states.size(0));
    const int S = static_cast<int>(hidden_states.size(1));

    auto modulation = torch::empty(
        {B, kInnerDimX2},
        torch::TensorOptions()
            .dtype(hidden_states.dtype())
            .device(hidden_states.device()));

    // Kernel 1: small-M Linear temb -> modulation.
    {
        dim3 grid(B, kInnerDimX2 / kWarpsPerCtaLinear);
        dim3 block(kBlockThreads);
        k_small_m_linear<<<grid, block, 0, stream>>>(
            temb.data_ptr<float>(),
            linear_weight.data_ptr<float>(),
            linear_bias.data_ptr<float>(),
            modulation.data_ptr<float>());
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_080 small-M Linear kernel launch failed");
    }

    // Kernel 2: LN + modulation per-(b, s) row.
    {
        dim3 grid(B, S);
        dim3 block(kBlockThreads);
        k_layernorm_with_modulation<<<grid, block, 0, stream>>>(
            hidden_states.data_ptr<float>(),
            modulation.data_ptr<float>(),
            eps,
            output.data_ptr<float>(),
            S);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_080 LN+modulation kernel launch failed");
    }
}
