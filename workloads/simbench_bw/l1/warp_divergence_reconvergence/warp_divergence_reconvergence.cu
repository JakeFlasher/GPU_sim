// slug: warp_divergence_reconvergence
// tier: P
// arch_gate: sm_90a sm_120
//
// Warp-divergence microbench: each thread takes one of `branches`
// independent code paths derived from its lane id, iterates
// `iterations` rounds of arithmetic per branch, and reconverges via
// __syncwarp(). Identity round-trip on the seeded buffer; the
// per-branch accumulator is folded into a sink.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void warp_divergence_reconvergence_kernel(
    const float *__restrict__ in, float *__restrict__ out, float *__restrict__ sink,
    int32_t warps, int32_t branches, int32_t iterations)
{
    const int total = warps * branches * iterations;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    float acc = 0.0f;
    int branch_id = tid % branches;
    for (int it = 0; it < iterations; ++it) {
        if (branch_id == 0) {
            acc += in[(tid + it) % total];
        } else if (branch_id == 1) {
            acc -= in[(tid + it * 2) % total];
        } else {
            acc *= 1.0f + in[(tid + it * 3) % total];
        }
        __syncwarp();
    }
    if (tid == 0) sink[0] = acc;
    __syncthreads();
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void warp_divergence_reconvergence_launch(
    const float *d_in, float *d_out, float *d_sink,
    int warps, int branches, int iterations, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    warp_divergence_reconvergence_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_sink, warps, branches, iterations);
}
