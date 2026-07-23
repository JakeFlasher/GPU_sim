// slug: l2_prefetch_hint_sweep
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// L2 prefetch hint sweep microbench: each warp issues a
// `prefetch.global.L2 [addr]` PTX hint `prefetch_distance` slots
// ahead of each read in a sequential stream. Sweeping the
// `prefetch_distance` shape axis exposes how the L2-prefetch hint
// changes the effective hit-rate distribution observed by the
// HRD-profiler criterion.
//
// The PTX `prefetch.global.L2` instruction is supported on
// CC >= 8.0 (Ampere onwards); it is hardware-advisory and the
// optimizer may legally elide it. The kernel still runs and emits
// trace evidence consumable by the sector-stream / divergence-
// proxy / HRD-profiler / sharing-locality / order-projection
// feature extractors regardless of whether the prefetch retires;
// the microbench-suite-coverage criterion's prefetch-hint family
// covers the structural expression of the hint, not a guaranteed
// SASS marker. R46 extension to close the prefetch-hint family.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void l2_prefetch_hint_sweep_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    float *__restrict__ sink,
    int32_t accesses, int32_t prefetch_distance, int32_t warps)
{
    const int total = accesses * warps;
    // Global thread ID across the multi-block grid. Previously the
    // kernel used `threadIdx.x` only, which caused every block in a
    // multi-warp (multi-block) launch to replay the same prefetch /
    // load stream and race writing the same output values. Codex
    // Round-18 review P2 fix: switch to global thread indexing so
    // each thread gets a unique slot across `gridDim.x * blockDim.x`
    // threads.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lanes = gridDim.x * blockDim.x;
    float acc = 0.0f;
    int idx = tid % total;
    for (int i = 0; i < accesses; ++i) {
        int prefetch_idx = (idx + prefetch_distance) % total;
        // Issue the L2 prefetch hint for the address `prefetch_distance`
        // slots ahead of the current read. The optimizer may legally
        // elide the hint; that is fine — the family covers the
        // structural expression, not a guaranteed SASS marker.
        asm volatile(
            "prefetch.global.L2 [%0];"
            :
            : "l"(in + prefetch_idx)
        );
        acc += in[idx];
        idx = (idx + 1) % total;
    }
    if (tid == 0) sink[0] = acc;
    __syncthreads();
    // Identity output for correctness.
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void l2_prefetch_hint_sweep_launch(
    const float *d_in, float *d_out, float *d_sink,
    int accesses, int prefetch_distance, int warps,
    cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps > 0 ? warps : 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    l2_prefetch_hint_sweep_kernel<<<grid, block, 0, stream>>>(
        d_in, d_out, d_sink, accesses, prefetch_distance, warps);
}
