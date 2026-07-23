// slug: global_scalar_32b_copy
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Vector-width request/sector-formation family (32-bit scalar member). A
// grid-strided scalar float copy out[i] = in[i] where each thread's successive
// accesses are stride-`lanes` apart (NOT contiguous per-thread), so the
// compiler cannot fuse them into a wide request and emits the scalar 32-bit
// SASS load/store form (LDG.E / STG.E, no .128). Paired with the 128-bit member
// global_vectorized_128b_copy (LDG.E.128 / STG.E.128): the ONLY difference is
// the per-access request width, the vector-width observation-point contrast for
// request/sector formation.
//
// Correctness is the identity round-trip on the seeded buffer (out = in) --
// the same reference shape as global_vectorized_128b_copy and the other
// identity-round-trip microbenches.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void global_scalar_32b_copy_kernel(
    const float *__restrict__ in, float *__restrict__ out, int32_t n)
{
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    // Grid-strided scalar copy: per-thread accesses are `lanes` apart, so each
    // is a single 32-bit request -> LDG.E / STG.E (never fused to .128).
    for (int i = tid; i < n; i += lanes) out[i] = in[i];
}

extern "C" void global_scalar_32b_copy_launch(
    const float *d_in, float *d_out, int n, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    global_scalar_32b_copy_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n);
}
