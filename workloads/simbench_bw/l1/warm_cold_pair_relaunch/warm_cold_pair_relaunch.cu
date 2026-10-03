// slug: warm_cold_pair_relaunch
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Warm/cold cache pair: two launches over the same allocation.
// The two passes read the same input, with an optional scratch sweep between
// them. Neither a cold first pass nor eviction of every input line is assured.
// Separate output buffers and sinks allow checking both passes after all
// launches without inserting cache-perturbing host readbacks between them.

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
    // Touch every scratch element. How much input data this evicts is measured,
    // not assumed from the ratio of scratch size to nominal L2 capacity.
    for (int i = tid; i < scratch_floats; i += lanes) {
        scratch[i] = scratch[i] + 1.0f;
    }
}

extern "C" cudaError_t warm_cold_pair_relaunch_launch(
    const float *d_in, float *d_first_out, float *d_repeat_out, float *d_sinks,
    float *d_scratch,
    int allocation_floats, int accesses,
    int eviction_floats, int warps, cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    // Cold launch.
    warm_cold_pair_kernel<<<grid, block, 0, stream>>>(
        d_in, d_first_out, d_sinks, allocation_floats, accesses);
    cudaError_t rc = cudaGetLastError();
    if (rc != cudaSuccess) return rc;
    if (eviction_floats > 0 && d_scratch != nullptr) {
        // Eviction sweep.
        warm_cold_pair_eviction_kernel<<<grid, block, 0, stream>>>(
            d_scratch, eviction_floats);
        rc = cudaGetLastError();
        if (rc != cudaSuccess) return rc;
    }
    // Warm launch (re-read same allocation).
    warm_cold_pair_kernel<<<grid, block, 0, stream>>>(
        d_in, d_repeat_out, d_sinks + 1, allocation_floats, accesses);
    return cudaGetLastError();
}
