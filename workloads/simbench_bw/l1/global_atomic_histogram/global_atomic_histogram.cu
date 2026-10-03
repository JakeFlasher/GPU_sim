// slug: global_atomic_histogram
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Global atomic histogram: each thread issues a global atomicAdd into
// one of `bins` slots. The kernel has both contended (all threads ->
// same bin) and dispersed (each thread -> distinct bin) paths.
// Identity round-trip on the seeded input/output buffer; the atomic
// counters are copied back and checked against an independent host histogram.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void global_atomic_histogram_kernel(
    const float *__restrict__ in, float *__restrict__ out, int32_t *__restrict__ histo,
    int32_t samples, int32_t bins)
{
    const int total = samples * bins;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    // Dispersed atomic: every sample increments a bin derived from its index.
    for (int i = tid; i < total; i += lanes) {
        int bin = (i * 2654435769u) % bins;
        atomicAdd(&histo[bin], 1);
    }
    // Contended atomic: all threads pile onto bin 0.
    for (int i = tid; i < samples; i += lanes) atomicAdd(&histo[0], 1);
    __syncthreads();
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void global_atomic_histogram_launch(
    const float *d_in, float *d_out, int32_t *d_histo,
    int samples, int bins, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    global_atomic_histogram_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_histo, samples, bins);
}
