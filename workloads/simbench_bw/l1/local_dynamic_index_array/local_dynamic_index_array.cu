// slug: local_dynamic_index_array
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Local-memory microbench (DYNAMIC-INDEX mechanism). Each thread holds a
// per-thread automatic array `buf[K]` and accesses it at DATA-DEPENDENT
// (runtime) indices derived from the seeded input values. The compiler cannot
// resolve a data-dependent index at compile time, so it CANNOT register-promote
// the array -- it is placed in `.local` memory and every access becomes a real
// LDL / STL in SASS (STACK-frame bytes > 0 in `cuobjdump -res-usage`).
//
// This forces local memory by DYNAMIC INDEXING alone: there is NO
// `-maxrregcount` cap (the default uncapped build still spills here), so it is a
// genuinely slug-distinct local-memory family member, NOT a register-cap variant
// of register_spill_local_mem (whose local traffic is forced by a build-time
// register cap on a statically-indexed live set). It is also distinct from the
// volatile-array member local_volatile_array (whose local traffic is forced by
// volatile semantics, not by a dynamic index).
//
// Correctness is the identity round-trip on the seeded input/output buffer
// (out = in) -- byte-for-byte the same reference as the other
// identity-round-trip microbenches. The local-memory work folds into a SEPARATE
// scratch sink that is NOT part of the verified output.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void local_dynamic_index_array_kernel(
    const float *__restrict__ in, float *__restrict__ out, float *__restrict__ sink,
    int32_t samples, int32_t bins)
{
    const int total = samples * bins;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    // Per-thread automatic array. K is a power of two so the index mask `& (K-1)`
    // keeps every dynamic index in-bounds.
    const int K = 64;
    float buf[K];
    for (int r = 0; r < K; ++r) buf[r] = in[(tid * 7 + r) % total];
    float acc = 0.0f;
    for (int i = tid; i < total; i += lanes) {
        // Indices derived from input VALUES + the runtime loop variable -> NOT
        // compile-time-resolvable -> `buf` cannot live in registers (LDL/STL).
        int widx = ((int)(in[i] * 131.0f)) & (K - 1);
        int ridx = ((int)(buf[i % K] * 251.0f)) & (K - 1);
        buf[widx] = buf[ridx] * 1.0001f + buf[widx];
        acc += buf[(widx + ridx) & (K - 1)];
    }
    if (tid == 0) sink[0] = acc;
    __syncthreads();
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void local_dynamic_index_array_launch(
    const float *d_in, float *d_out, float *d_sink,
    int samples, int bins, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    local_dynamic_index_array_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_sink, samples, bins);
}
