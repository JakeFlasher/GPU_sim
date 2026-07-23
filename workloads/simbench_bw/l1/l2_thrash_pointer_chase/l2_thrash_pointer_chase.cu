// slug: l2_thrash_pointer_chase
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Pointer-chase microbench: each warp follows a chain of dependent
// global loads, exercising L2 hit/miss and MSHR serialization. The
// chain is built from the seeded input data — each thread reads the
// next index from the previous load. Identity-output round-trip;
// SASS contains LDG.E.SYS or LDG.E.STRONG.GPU markers.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void l2_thrash_pointer_chase_kernel(
    const float *__restrict__ in, float *__restrict__ out, float *__restrict__ sink,
    int32_t chain_length, int32_t warps)
{
    const int total = chain_length * warps;
    // Global thread ID across the multi-block grid. Previously the
    // kernel used `threadIdx.x` only, which caused every block in a
    // multi-warp (multi-block) launch to replay the same pointer-
    // chase stream and race writing the same output values. Codex
    // Round-18 review P2 fix: switch to global thread indexing so
    // each thread gets a unique slot across `gridDim.x * blockDim.x`
    // threads.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lanes = gridDim.x * blockDim.x;
    // Pointer chase: build a hash from each loaded value, fold into sink.
    float acc = 0.0f;
    int idx = tid % total;
    for (int i = 0; i < chain_length; ++i) {
        float v = in[idx];
        acc += v;
        // Next index is derived from the bit pattern of the loaded value
        // — dependent load on the previous result.
        idx = ((__float_as_int(v) >> 3) ^ idx) % total;
    }
    if (tid == 0) sink[0] = acc;
    __syncthreads();
    // Identity output.
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void l2_thrash_pointer_chase_launch(
    const float *d_in, float *d_out, float *d_sink,
    int chain_length, int warps, cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps > 0 ? warps : 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    l2_thrash_pointer_chase_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_sink, chain_length, warps);
}
