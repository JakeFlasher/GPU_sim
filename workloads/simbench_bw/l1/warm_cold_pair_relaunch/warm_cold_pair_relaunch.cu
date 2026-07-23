// slug: warm_cold_pair_relaunch
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Warm/cold cache pair: two launches over the same allocation.
// The first launch (cold) reads through the input fresh from
// DRAM; the second launch (warm) re-reads the same window after
// an optional eviction step. The HRD-profiler criterion compares
// per-launch hit-rate distributions to recover the warm/cold
// delta — the per-line miss rate should drop on the warm pass
// proportional to the L2 capacity vs. allocation-size ratio.
//
// `eviction_bytes` shape axis controls how many bytes the
// scratch sweep touches between the two launches. When 0, the
// warm launch hits whatever survived in cache. When sized at
// 2× L2, the warm launch is forced cold.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void warm_cold_pair_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    float *__restrict__ sink,
    int32_t allocation_floats, int32_t accesses)
{
    // Global thread ID across the multi-block grid. Previously the
    // kernel used `threadIdx.x` only, which caused every block in a
    // multi-warp (multi-block) launch to replay the same warm/cold
    // access stream and race writing the same output values. Codex
    // Round-18 review P2 fix: switch to global thread indexing so
    // each thread gets a unique slot across
    // `gridDim.x * blockDim.x` threads.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lanes = gridDim.x * blockDim.x;
    float acc = 0.0f;
    int idx = tid % allocation_floats;
    for (int i = 0; i < accesses; ++i) {
        acc += in[idx];
        idx = (idx + 32) % allocation_floats;
    }
    if (tid == 0) sink[0] = acc;
    __syncthreads();
    for (int i = tid; i < allocation_floats; i += lanes) {
        out[i] = in[i];
    }
}

extern "C" __global__ void warm_cold_pair_eviction_kernel(
    float *__restrict__ scratch, int32_t scratch_floats)
{
    // Global thread ID across the multi-block grid (same Codex
    // Round-18 review P2 fix as warm_cold_pair_kernel above).
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lanes = gridDim.x * blockDim.x;
    // Force every line of `scratch` through the cache so the
    // primary allocation is partially evicted between launches.
    for (int i = tid; i < scratch_floats; i += lanes) {
        scratch[i] = scratch[i] + 1.0f;
    }
}

extern "C" void warm_cold_pair_relaunch_launch(
    const float *d_in, float *d_out, float *d_sink,
    float *d_scratch,
    int allocation_floats, int accesses,
    int eviction_floats, int warps, cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps > 0 ? warps : 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    // Cold launch.
    warm_cold_pair_kernel<<<grid, block, 0, stream>>>(
        d_in, d_out, d_sink, allocation_floats, accesses);
    if (eviction_floats > 0 && d_scratch != nullptr) {
        // Eviction sweep.
        warm_cold_pair_eviction_kernel<<<grid, block, 0, stream>>>(
            d_scratch, eviction_floats);
    }
    // Warm launch (re-read same allocation).
    warm_cold_pair_kernel<<<grid, block, 0, stream>>>(
        d_in, d_out, d_sink, allocation_floats, accesses);
}
