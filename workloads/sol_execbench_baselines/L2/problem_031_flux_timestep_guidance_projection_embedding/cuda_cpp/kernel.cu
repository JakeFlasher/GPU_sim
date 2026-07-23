#include "kernel.h"
#include <cuda_runtime.h>
#include <cstdint>

// Two small fused kernels for the FLUX timestep + guidance projection
// reference. The matmuls themselves stay on cuBLAS (via at::linear from
// main.cpp); these kernels collapse the reference's surrounding
// elementwise work (timestep*1000 -> broadcast mul -> sin -> cos ->
// concat, then sigmoid + multiply for SiLU) into one launch each.
//
// Constants per definition.json:
//   inner_dim              = 3072
//   pooled_projection_dim  = 768
//   time_embed_dim         = 768
//   half_time_embed_dim    = 384
//
// The timestep_embed kernel issues 1 CTA per batch row. Each thread
// handles two output columns (the cos slot and the matching sin slot
// in the second half), so 384 threads cover the 768-wide output row.
//
// The SiLU kernel is a stock element-wise fp32 kernel sized for the
// [B * inner_dim] flat layout.

namespace {

constexpr int HALF_TIME_EMBED_DIM = 384;       // freq count = half of 768
constexpr int TIME_EMBED_DIM = 768;            // cos + sin halves concatenated
constexpr float TIMESTEP_SCALE = 1000.0f;      // FLUX convention

__global__ void __launch_bounds__(HALF_TIME_EMBED_DIM, 4)
problem_031_flux_timestep_embed_kernel(
    const float* __restrict__ timestep,
    const float* __restrict__ freqs,
    float* __restrict__ out,
    int batch_size) {

    const int row = blockIdx.x;
    if (row >= batch_size) return;
    const int tid = threadIdx.x;   // 0 .. HALF_TIME_EMBED_DIM - 1

    const float ts_scaled = timestep[row] * TIMESTEP_SCALE;
    const float arg = ts_scaled * freqs[tid];

    // Reference layout: timestep_embed = cat([cos_embed, sin_embed], -1)
    //   timestep_embed[row, k]                       = cos(arg_k)   for k in [0, half)
    //   timestep_embed[row, k + half]                = sin(arg_k)   for k in [0, half)
    float c, s;
    __sincosf(arg, &s, &c);

    const int row_base = row * TIME_EMBED_DIM;
    out[row_base + tid] = c;
    out[row_base + tid + HALF_TIME_EMBED_DIM] = s;
}

constexpr int SILU_BLOCK_THREADS = 256;

__global__ void __launch_bounds__(SILU_BLOCK_THREADS, 4)
problem_031_flux_silu_kernel(float* __restrict__ x, int64_t n) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const float v = x[i];
        // SiLU: v * sigmoid(v) = v / (1 + exp(-v)).
        const float sig = 1.0f / (1.0f + __expf(-v));
        x[i] = v * sig;
    }
}

}  // namespace

void problem_031_flux_timestep_embed_launcher(
    const torch::Tensor& timestep,
    const torch::Tensor& freqs,
    torch::Tensor& timestep_embed_out,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(timestep.size(0));
    dim3 grid(batch_size);
    dim3 block(HALF_TIME_EMBED_DIM);
    problem_031_flux_timestep_embed_kernel<<<grid, block, 0, stream>>>(
        timestep.data_ptr<float>(),
        freqs.data_ptr<float>(),
        timestep_embed_out.data_ptr<float>(),
        batch_size);
}

void problem_031_flux_silu_inplace_launcher(
    torch::Tensor& x,
    cudaStream_t stream) {
    const int64_t n = x.numel();
    const int64_t blocks_full = (n + SILU_BLOCK_THREADS - 1) / SILU_BLOCK_THREADS;
    const int blocks = static_cast<int>(blocks_full < 65535 ? blocks_full : 65535);
    if (blocks <= 0) return;
    dim3 grid(blocks);
    dim3 block(SILU_BLOCK_THREADS);
    problem_031_flux_silu_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), n);
}
