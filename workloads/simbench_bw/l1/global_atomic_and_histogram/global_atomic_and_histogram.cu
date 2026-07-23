// slug: global_atomic_and_histogram
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Global atomic-AND histogram: each thread issues a global atomicAnd into one
// of `bins` slots, RETURN IGNORED, so the compiler emits the reduction SASS
// form (RED / REDG .AND) rather than the atomic-with-return form. This is the
// opcode-distinct AND-operation member of the atomics family: the add-based
// global_atomic_histogram emits REDG.E.ADD, this one emits RED.E.AND.S32 — a
// distinct slug exercising the global-reduction op-class. Paired with
// global_atomic_and_return_used (ATOMG.E.AND, return used), the ONLY difference
// is whether the atomic's return value is live (the ATOM-vs-RED contrast).
//
// Correctness is the identity round-trip on the seeded input/output buffer
// (out = in) -- byte-for-byte the same reference as the other identity-round-trip
// microbenches. The atomicAnd traffic targets a SEPARATE scratch counter buffer
// `histo` that is NOT part of the verified output.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void global_atomic_and_histogram_kernel(
    const float *__restrict__ in, float *__restrict__ out, int32_t *__restrict__ histo,
    int32_t samples, int32_t bins)
{
    const int total = samples * bins;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    // Dispersed atomicAnd: RETURN IGNORED -> reduction (RED / REDG .AND).
    for (int i = tid; i < total; i += lanes) {
        int bin = (i * 2654435769u) % bins;
        atomicAnd(&histo[bin], (int)i);
    }
    // Contended atomicAnd: all threads onto bin 0, return ignored.
    for (int i = tid; i < samples; i += lanes) atomicAnd(&histo[0], (int)i);
    __syncthreads();
    for (int i = tid; i < total; i += lanes) out[i] = in[i];
}

extern "C" void global_atomic_and_histogram_launch(
    const float *d_in, float *d_out, int32_t *d_histo,
    int samples, int bins, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    global_atomic_and_histogram_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_histo, samples, bins);
}
