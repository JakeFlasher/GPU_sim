// slug: global_vectorized_128b_copy
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Vector-width request/sector-formation family (128-bit member). A contiguous
// float4 (128-bit) global copy out4[i] = in4[i] over a 16-byte-aligned buffer,
// so each per-thread access forms a single 128-bit request and the compiler
// emits the vectorized SASS load/store form (LDG.E.128 / STG.E.128). Paired
// with the scalar 32-bit member global_scalar_32b_copy (LDG.E / STG.E, no
// .128): the ONLY difference is the per-access request width, the vector-width
// observation-point contrast for request/sector formation.
//
// Correctness is the identity round-trip on the seeded buffer (out = in) --
// the same reference shape as the other identity-round-trip microbenches. Every
// registered shape has samples*bins divisible by 4 (the host driver fails
// closed otherwise), so the whole buffer is copied through the 128-bit path.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void global_vectorized_128b_copy_kernel(
    const float4 *__restrict__ in4, float4 *__restrict__ out4, int32_t n4)
{
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    // float4 copy: a single 128-bit request per access -> LDG.E.128 / STG.E.128.
    for (int i = tid; i < n4; i += lanes) out4[i] = in4[i];
}

extern "C" void global_vectorized_128b_copy_launch(
    const float *d_in, float *d_out, int n, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    int n4 = n / 4;  // n is a multiple of 4 for every registered shape
    global_vectorized_128b_copy_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float4 *>(d_in),
        reinterpret_cast<float4 *>(d_out), n4);
}
