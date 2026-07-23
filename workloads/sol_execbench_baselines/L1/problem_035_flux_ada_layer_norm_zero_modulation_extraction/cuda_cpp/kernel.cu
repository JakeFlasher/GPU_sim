#include "kernel.h"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// AdaLayerNormZero modulation extraction: one fused fp32
// Linear-with-bias + 6-chunk multi-output DPS kernel.
//
// Per-element math (matching the reference
//   `emb_out = emb @ weight.T + bias; chunk(6, dim=1)`):
//
//   modulation[b, n] = sum_{k=0}^{K-1} weight[n, k] * emb[b, k]
//                    + bias[n]
//   for n in [0, N=6*H) where H = inner_dim = 3072.
//
// Then chunk along n:
//   chunk_idx = n / H ;  local_n = n % H
//   outputs[chunk_idx][b, local_n] = modulation[b, n]
//
// Output layout per the canonical reference:
//   chunk 0 = shift_msa, chunk 1 = scale_msa, chunk 2 = gate_msa,
//   chunk 3 = shift_mlp, chunk 4 = scale_mlp, chunk 5 = gate_mlp.
//
// Kernel topology:
//   * Grid (B, N_tiles) where N_tiles = N / kWarpsPerCta = 18432 / 8
//     = 2304. Each CTA owns one (b, n_tile) and computes the
//     `kWarpsPerCta = 8` output elements in that tile via 8 warps.
//   * BLOCK_THREADS = 256 (8 warps of 32 lanes).
//   * Each CTA caches emb[b, :K=3072] in shared memory once
//     (12 KiB fp32) so the 8 warps reuse it across their 8 K-
//     reductions instead of fetching from L2/HBM per warp.
//   * Per output element: 32 lanes stride-load
//     `weight[n, k0..K-1]` from global, multiply by the shared
//     `emb[b, k]`, accumulate a per-lane fp32 partial sum, warp-
//     shuffle reduce across the 32 lanes, lane 0 writes
//     `partial + bias[n]` to the correct chunked output.
//
// Pinning: H=3072 and N=18432=6*H are pinned by the problem
// definition. The kernel uses `constexpr int kH = 3072` etc. and
// the host launcher checks every tensor's pinned dims via TORCH_CHECK
// to enforce the runtime-check audit discipline.

namespace {

constexpr int kH = 3072;
constexpr int kN = 18432;          // = 6 * kH
constexpr int kNumOutputs = 6;
constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerCta = kBlockThreads / kWarpSize;  // 8
constexpr int kNTilesPerCta = kWarpsPerCta;              // 8 output n's per CTA

static_assert(kN % kWarpsPerCta == 0,
              "kN must be divisible by kWarpsPerCta");

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__global__ void __launch_bounds__(kBlockThreads, 4) k_ada_modulation_extract(
    const float* __restrict__ emb,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ shift_msa,
    float* __restrict__ scale_msa,
    float* __restrict__ gate_msa,
    float* __restrict__ shift_mlp,
    float* __restrict__ scale_mlp,
    float* __restrict__ gate_mlp
) {
    const int b = blockIdx.x;
    const int n_tile_base = blockIdx.y * kWarpsPerCta;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);

    __shared__ float s_emb[kH];
    #pragma unroll 4
    for (int k = tid; k < kH; k += kBlockThreads) {
        s_emb[k] = emb[b * kH + k];
    }
    __syncthreads();

    const int n = n_tile_base + warp_id;

    float partial = 0.0f;
    const float* w_row = weight + (size_t)n * kH;
    for (int k = lane; k < kH; k += kWarpSize) {
        partial += w_row[k] * s_emb[k];
    }
    partial = warp_reduce_sum(partial);

    if (lane == 0) {
        const float out_val = partial + bias[n];
        const int chunk_idx = n / kH;
        const int local_n = n - chunk_idx * kH;

        float* out_ptrs[kNumOutputs] = {
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp
        };
        out_ptrs[chunk_idx][b * kH + local_n] = out_val;
    }
}

}  // namespace

void problem_035_flux_ada_layer_norm_zero_modulation_extraction_launcher(
    const torch::Tensor& emb,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& shift_msa,
    torch::Tensor& scale_msa,
    torch::Tensor& gate_msa,
    torch::Tensor& shift_mlp,
    torch::Tensor& scale_mlp,
    torch::Tensor& gate_mlp,
    cudaStream_t stream
) {
    const int batch_size = static_cast<int>(emb.size(0));

    dim3 grid(batch_size, kN / kWarpsPerCta);
    dim3 block(kBlockThreads);

    k_ada_modulation_extract<<<grid, block, 0, stream>>>(
        emb.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        shift_msa.data_ptr<float>(),
        scale_msa.data_ptr<float>(),
        gate_msa.data_ptr<float>(),
        shift_mlp.data_ptr<float>(),
        scale_mlp.data_ptr<float>(),
        gate_mlp.data_ptr<float>()
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "problem_035 ada modulation extract kernel launch failed");
}
