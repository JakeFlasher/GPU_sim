// slug: global_vectorized_64b_copy
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Vector-width request/sector-formation family (64-bit member). A grid-strided
// float2 (64-bit) global copy out2[i] = in2[i] where each per-thread access
// forms a single 64-bit request, so the compiler emits the 64-bit vectorized
// SASS load/store form (LDG.E.64 / STG.E.64). This is the opcode-distinct
// middle member of the vector-width sweep: the scalar 32-bit member
// global_scalar_32b_copy emits bare LDG.E / STG.E (no width suffix) and the
// 128-bit member global_vectorized_128b_copy emits LDG.E.128 / STG.E.128. The
// ONLY difference across the three is the per-access request width -- the
// vector-width observation-point contrast for request/sector formation.
//
// Correctness is the identity round-trip on the seeded buffer (out = in) --
// the same reference shape as the other identity-round-trip microbenches. Every
// registered shape has samples*bins divisible by 2 (the host driver fails
// closed otherwise), so the whole buffer is copied through the 64-bit path
// with no scalar tail that would pollute the LDG.E.64 / STG.E.64 contract.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void global_vectorized_64b_copy_kernel(
    const float2 *__restrict__ in2, float2 *__restrict__ out2, int32_t n2)
{
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    // float2 copy: a single 64-bit request per access -> LDG.E.64 / STG.E.64.
    for (int i = tid; i < n2; i += lanes) out2[i] = in2[i];
}

extern "C" void global_vectorized_64b_copy_launch(
    const float *d_in, float *d_out, int n, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    int n2 = n / 2;  // n is a multiple of 2 for every registered shape
    global_vectorized_64b_copy_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float2 *>(d_in),
        reinterpret_cast<float2 *>(d_out), n2);
}
