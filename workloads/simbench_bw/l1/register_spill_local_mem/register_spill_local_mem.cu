// slug: register_spill_local_mem
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Register-spill microbench. Each thread holds a FIXED 64-wide live FP32 set
// (`regs[64]`) updated across an `iterations`-length dependency chain. That
// live set FITS in registers at an uncapped build; the SPILL phenomenon is
// controlled by the BUILD-TIME per-thread register cap (Makefile SPILL_MAXREG;
// nvcc -maxrregcount): a tight cap forces the live set to
// `.local` memory (LDL / STL in SASS, STACK-frame bytes in cuobjdump
// -res-usage). The RUNTIME axes (warps / live_regs / iterations) set the WORK
// ENVELOPE only -- grid blocks, dependency-chain length, and the seeded buffer
// size (`total = warps * live_regs * iterations`); `live_regs` does NOT control
// the live register set or register pressure. Identity round-trip on the seeded
// buffer; per-thread accumulator folded into a sink.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void register_spill_local_mem_kernel(
    const float *__restrict__ in, float *__restrict__ out, float *__restrict__ sink,
    int32_t warps, int32_t live_regs, int32_t iterations)
{
    const int total = warps * live_regs * iterations;
    // Global thread ID across the multi-block grid. Previously the
    // kernel used `threadIdx.x` only, which caused every block in a
    // multi-warp (multi-block) launch to replay the same register-
    // spill initialization stream and race writing the same output
    // values. Codex Round-18 review P2 fix: switch to global thread
    // indexing so each thread gets a unique slot across
    // `gridDim.x * blockDim.x` threads.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lanes = gridDim.x * blockDim.x;
    // Allocate a per-thread "register file" wider than the SM's allocation
    // budget per lane to force spill. 64 floats per thread is well above
    // the typical compile-time max of 64 registers per thread.
    float regs[64];
    for (int r = 0; r < 64; ++r) regs[r] = in[(tid * 7 + r) % total];
    for (int it = 0; it < iterations; ++it) {
        for (int r = 0; r < 64; ++r) {
            regs[r] = regs[r] * 1.0001f + regs[(r + 1) % 64];
        }
    }
    float acc = 0.0f;
    for (int r = 0; r < 64; ++r) acc += regs[r];
    if (tid == 0) sink[0] = acc;
    __syncthreads();
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void register_spill_local_mem_launch(
    const float *d_in, float *d_out, float *d_sink,
    int warps, int live_regs, int iterations, cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps > 0 ? warps : 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    register_spill_local_mem_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_sink, warps, live_regs, iterations);
}
