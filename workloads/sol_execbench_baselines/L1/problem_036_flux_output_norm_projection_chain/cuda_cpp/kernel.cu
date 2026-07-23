#include "kernel.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Four-launch chain for the FLUX final-output norm + projection.
//
//   Kernel 1 — per-element SiLU on temb
//     temb_silu[b, k] = temb[b, k] * sigmoid(temb[b, k])
//     Grid-stride loop over B * inner_dim.
//
//   Kernel 2 — small-M Linear: temb_silu -> modulation[B, 6144]
//     Per-(b, n_tile) CTA over Grid `(B, modulation_dim /
//     warps_per_cta = 6144 / 8 = 768)`. BLOCK_THREADS = 256
//     (8 warps). Each CTA shared-memory-caches temb_silu[b, :K =
//     3072] once (12 KiB fp32). One warp per output element with
//     `__shfl_xor_sync` K-reduction; lane 0 writes
//     `partial + linear_bias[n]`.
//
//   Kernel 3 — fused LN + modulation per-(b, s) row
//     Per-(b, s) CTA over Grid `(B, S)`; BLOCK_THREADS = 256.
//     Pass A: per-row in-CTA reduction over inner_dim=3072 to
//     produce mean, rstd. Pass B: write
//     `hs_mod[b, s, d] = ((x - mean) * rstd) * (1 + scale) +
//     shift`, where shift = modulation[b, :inner_dim] and
//     scale = modulation[b, inner_dim:].
//
//   Kernel 4 — cuBLAS fp32 GEMM via at::linear (dispatched from
//     main.cpp): output = hs_mod @ proj_out_weight.t() +
//     proj_out_bias.
//
// All custom-kernel reductions are in-CTA only; no cross-CTA
// reduction tree avoids accumulation-order drift against
// PyTorch's reduction order.

namespace {

constexpr int kInnerDim = 3072;
constexpr int kModulationDim = 2 * kInnerDim;  // 6144
constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;  // 8
constexpr int kWarpsPerCtaLinear = kNumWarps;

static_assert(kModulationDim % kWarpsPerCtaLinear == 0,
              "kModulationDim must be divisible by warps-per-CTA");

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

// Kernel 1: per-element SiLU. Grid-stride over [B, inner_dim].
__global__ void __launch_bounds__(kBlockThreads, 4) k_silu(
    const float* __restrict__ temb,
    float* __restrict__ temb_silu,
    size_t total_elements
) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements; idx += stride) {
        const float v = temb[idx];
        const float sig = 1.0f / (1.0f + __expf(-v));
        temb_silu[idx] = v * sig;
    }
}

// Kernel 2: small-M Linear. Grid (B, modulation_dim / 8 = 768).
__global__ void __launch_bounds__(kBlockThreads, 4) k_small_m_linear(
    const float* __restrict__ temb_silu,
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
        s_temb[k] = temb_silu[b * kInnerDim + k];
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
        modulation[b * kModulationDim + n] = partial + bias[n];
    }
}

// Kernel 3: fused LN + modulation. Grid (B, S).
__global__ void __launch_bounds__(kBlockThreads, 4)
    k_layernorm_with_modulation(
        const float* __restrict__ hidden_states,
        const float* __restrict__ modulation,
        float eps,
        float* __restrict__ hs_mod,
        int S
) {
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t row_offset = ((size_t)b * S + s) * kInnerDim;
    const float* __restrict__ x_row = hidden_states + row_offset;
    float* __restrict__ out_row = hs_mod + row_offset;
    // problem_036 chunk order: shift first, scale second
    // (modulation[:, :inner_dim] -> shift,
    //  modulation[:, inner_dim:] -> scale).
    const float* __restrict__ shift_row =
        modulation + (size_t)b * kModulationDim;
    const float* __restrict__ scale_row =
        modulation + (size_t)b * kModulationDim + kInnerDim;

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

    // Pass B: write hs_mod with modulation.
    for (int d = tid; d < kInnerDim; d += kBlockThreads) {
        const float normalized = (x_row[d] - mean) * rstd;
        const float scale = scale_row[d];
        const float shift = shift_row[d];
        out_row[d] = normalized * (1.0f + scale) + shift;
    }
}

}  // namespace

void problem_036_flux_output_norm_projection_chain_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& temb,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias,
    const torch::Tensor& proj_out_weight,
    const torch::Tensor& proj_out_bias,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream
) {
    const int B = static_cast<int>(hidden_states.size(0));
    const int S = static_cast<int>(hidden_states.size(1));

    auto opts = hidden_states.options();
    auto temb_silu = torch::empty({B, kInnerDim}, opts);
    auto modulation = torch::empty({B, kModulationDim}, opts);
    auto hs_mod = torch::empty({B, S, kInnerDim}, opts);

    // Kernel 1: SiLU on temb -> temb_silu.
    {
        const size_t total = (size_t)B * kInnerDim;
        const int max_blocks = 65535;
        int blocks = static_cast<int>(
            (total + kBlockThreads - 1) / kBlockThreads);
        if (blocks > max_blocks) blocks = max_blocks;
        dim3 grid(blocks);
        dim3 block(kBlockThreads);
        k_silu<<<grid, block, 0, stream>>>(
            temb.data_ptr<float>(),
            temb_silu.data_ptr<float>(),
            total);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_036 SiLU kernel launch failed");
    }

    // Kernel 2: small-M Linear temb_silu -> modulation.
    {
        dim3 grid(B, kModulationDim / kWarpsPerCtaLinear);
        dim3 block(kBlockThreads);
        k_small_m_linear<<<grid, block, 0, stream>>>(
            temb_silu.data_ptr<float>(),
            linear_weight.data_ptr<float>(),
            linear_bias.data_ptr<float>(),
            modulation.data_ptr<float>());
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_036 small-M Linear kernel launch failed");
    }

    // Kernel 3: fused LN + modulation per-(b, s) row.
    {
        dim3 grid(B, S);
        dim3 block(kBlockThreads);
        k_layernorm_with_modulation<<<grid, block, 0, stream>>>(
            hidden_states.data_ptr<float>(),
            modulation.data_ptr<float>(),
            eps,
            hs_mod.data_ptr<float>(),
            S);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_036 LN+modulation kernel launch failed");
    }

    // Kernel 4: cuBLAS fp32 GEMM via at::linear, dispatched from
    // main.cpp. The launcher returns; main.cpp follows up with
    //   at::linear(hs_mod, proj_out_weight, proj_out_bias)
    // and writes the result to the pre-allocated `output` via
    // at::linear_out. (Modeled at the launcher boundary so the
    // hs_mod tensor allocated here can be passed by value to the
    // caller — see main.cpp for the final at::linear_out call.)
    // To keep ownership of hs_mod simple, the launcher itself
    // dispatches the cuBLAS GEMM here.
    at::linear_out(
        output, hs_mod, proj_out_weight, proj_out_bias);
}
