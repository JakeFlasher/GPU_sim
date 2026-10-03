// slug: global_atomic_return_used
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Paired ATOM partner of global_atomic_histogram. It issues the SAME global
// atomicAdd traffic, but USES the returned old value (accumulates it and stores
// the accumulator to global memory) so the compiler cannot discard the return
// and must emit the atomic-with-return SASS form (ATOM / ATOMG) rather than the
// reduction form (RED / REDG) that global_atomic_histogram (return ignored)
// emits. Keeping the result live also adds dependent accumulation and sink
// stores; the pair is NOT a pure opcode-latency intervention.
//
// Each lane owns one separate 64-bit sum output. Never write the sums into
// histogram bins: colliding ordinary stores race with each other and may race
// with still-running atomics. The host checks the copy, every histogram bin,
// and sum(returned old values) == sum_b count[b]*(count[b]-1)/2, independent of
// the legal (nondeterministic) lane order of atomic updates.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void global_atomic_return_used_kernel(
    const float *__restrict__ in, float *__restrict__ out, int32_t *__restrict__ histo,
    uint64_t *__restrict__ returned_sums, int32_t samples, int32_t bins)
{
    const int total = samples * bins;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    uint64_t acc = 0;  // a 32-bit counter does not bound the sum of old values
    // Dispersed atomic: RETURN VALUE USED -> atomic-with-return (ATOM/ATOMG).
    for (int i = tid; i < total; i += lanes) {
        int bin = (i * 2654435769u) % bins;
        acc += (uint32_t)atomicAdd(&histo[bin], 1);
    }
    // Contended atomic: RETURN VALUE USED.
    for (int i = tid; i < samples; i += lanes) acc += (uint32_t)atomicAdd(&histo[0], 1);
    returned_sums[tid] = acc;
    __syncthreads();
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void global_atomic_return_used_launch(
    const float *d_in, float *d_out, int32_t *d_histo, uint64_t *d_returned_sums,
    int samples, int bins, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    global_atomic_return_used_kernel<<<grid, block, 0, stream>>>(
        d_in, d_out, d_histo, d_returned_sums, samples, bins);
}
