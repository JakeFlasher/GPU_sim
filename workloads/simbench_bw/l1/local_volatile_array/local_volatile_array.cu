// slug: local_volatile_array
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Local-memory microbench (VOLATILE mechanism). Each thread holds a per-thread
// `volatile` automatic array `buf[K]`. The `volatile` qualifier forces every
// access to be a REAL memory access -- the compiler may not keep the values in
// registers or elide/reorder the loads and stores -- so the array is placed in
// `.local` memory and each read/write is an actual LDL / STL in SASS
// (STACK-frame bytes > 0 in `cuobjdump -res-usage`).
//
// This forces local memory by VOLATILE semantics alone: the indices are static
// and there is NO `-maxrregcount` cap, so it is a genuinely slug-distinct
// local-memory family member, NOT a register-cap variant of
// register_spill_local_mem (register-pressure-forced spill on a statically
// indexed live set) and NOT the dynamic-index member local_dynamic_index_array
// (data-dependent-index-forced local memory).
//
// Correctness is the identity round-trip on the seeded input/output buffer
// (out = in) -- byte-for-byte the same reference as the other
// identity-round-trip microbenches. The local-memory work folds into a SEPARATE
// scratch sink that is NOT part of the verified output.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void local_volatile_array_kernel(
    const float *__restrict__ in, float *__restrict__ out, float *__restrict__ sink,
    int32_t samples, int32_t bins)
{
    const int total = samples * bins;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    // Per-thread VOLATILE automatic array. K is a power of two so the wrap mask
    // `& (K-1)` keeps the neighbor index in-bounds.
    const int K = 64;
    volatile float buf[K];
    for (int r = 0; r < K; ++r) buf[r] = in[(tid * 7 + r) % total];
    // `bins` passes of a volatile read-modify-write stencil. Each buf access is
    // a real LDL/STL because of the volatile qualifier.
    for (int pass = 0; pass < bins; ++pass) {
        for (int r = 0; r < K; ++r) {
            buf[r] = buf[r] * 1.0001f + buf[(r + 1) & (K - 1)];
        }
    }
    float acc = 0.0f;
    for (int r = 0; r < K; ++r) acc += buf[r];
    if (tid == 0) sink[0] = acc;
    __syncthreads();
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void local_volatile_array_launch(
    const float *d_in, float *d_out, float *d_sink,
    int samples, int bins, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    local_volatile_array_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_sink, samples, bins);
}
