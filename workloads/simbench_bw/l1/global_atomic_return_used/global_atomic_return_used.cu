// slug: global_atomic_return_used
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Paired ATOM partner of global_atomic_histogram. It issues the SAME global
// atomicAdd traffic, but USES the returned old value (accumulates it and stores
// the accumulator to global memory) so the compiler cannot discard the return
// and must emit the atomic-with-return SASS form (ATOM / ATOMG) rather than the
// reduction form (RED / REDG) that global_atomic_histogram (return ignored)
// emits. The pair is the ATOM-vs-RED observation-point contrast: same
// algorithm, the ONLY difference is whether the atomic's return value is live.
//
// Correctness is the identity round-trip on the seeded input/output buffer
// (out = in) -- byte-for-byte the same reference as global_atomic_histogram.
// The accumulated returned values escape into `histo` (a scratch counter buffer
// that is NOT part of the verified output), which keeps the return live without
// changing `out`.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void global_atomic_return_used_kernel(
    const float *__restrict__ in, float *__restrict__ out, int32_t *__restrict__ histo,
    int32_t samples, int32_t bins)
{
    const int total = samples * bins;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    int acc = 0;  // accumulates the OLD values RETURNED by each atomicAdd
    // Dispersed atomic: RETURN VALUE USED -> atomic-with-return (ATOM/ATOMG).
    for (int i = tid; i < total; i += lanes) {
        int bin = (i * 2654435769u) % bins;
        acc += atomicAdd(&histo[bin], 1);
    }
    // Contended atomic: RETURN VALUE USED.
    for (int i = tid; i < samples; i += lanes) acc += atomicAdd(&histo[0], 1);
    // Make `acc` live: store it into `histo` (NOT the verified `out`) so the
    // returned values cannot be eliminated and the atomic stays ATOM, not RED.
    if (bins > 0) histo[(tid + bins - 1) % bins] = acc;
    __syncthreads();
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void global_atomic_return_used_launch(
    const float *d_in, float *d_out, int32_t *d_histo,
    int samples, int bins, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    global_atomic_return_used_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_histo, samples, bins);
}
