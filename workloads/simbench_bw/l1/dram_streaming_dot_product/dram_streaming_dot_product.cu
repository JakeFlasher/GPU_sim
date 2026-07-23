// slug: dram_streaming_dot_product
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Streaming dot product with deterministic L2-resident working set:
// each thread accumulates a*b across a grid-strided window over two
// equal-sized buffers; the per-block partial sum is written to a
// separate output slot (no atomics). With the default --n 131072
// (= 128K floats per buffer = 512KB each, 1MB total working set on
// disk) and --passes 4, the kernel performs four sequential sweeps
// of the same allocation. The first sweep is a cold compulsory fill
// of L2 with the two buffers (1MB fits comfortably in the sm_86 L2
// cache of 6MB); the remaining three sweeps re-read the same
// addresses from a warmed L2. The aggregate L2 sector hit rate is
// dominated by spatial-locality misses (1 sector miss per 4 sectors
// on the cold sweep) PLUS three warm sweeps at ~100% sector hit
// rate, producing a stable per-launch hit rate near 94% that is
// independent of cache state at launch start. This stability is the
// admission-relevant property: 5-repeat NCU labeling under the
// schema-pinned stddev_l2_sector <= 0.03 gate should reproduce the
// hit rate within tens of basis points across separate processes.
//
// SASS surface: the kernel emits canonical LDG.E reads of a and b
// (plus one ST.E.SY for the per-block partial sum); no HMMA, IMMA,
// TMA, cp.async, atomic, or shared-bank-conflicting opcodes are
// used. The NVBit tracer's existing parser admits this pattern
// without any modification to tools/nvbit_tracer/.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void dram_streaming_dot_product_kernel(
    const float *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ partial_sums, int32_t n, int32_t passes)
{
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    float partial = 0.0f;
    for (int p = 0; p < passes; ++p) {
        for (int i = gid; i < n; i += stride) {
            partial += a[i] * b[i];
        }
    }
    sdata[tid] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

extern "C" void dram_streaming_dot_product_launch(
    const float *d_a, const float *d_b, float *d_partial_sums,
    int n, int passes, int blocks, int threads, cudaStream_t stream)
{
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    size_t shmem = (size_t)threads * sizeof(float);
    dram_streaming_dot_product_kernel<<<grid, block, shmem, stream>>>(
        d_a, d_b, d_partial_sums, n, passes);
}
