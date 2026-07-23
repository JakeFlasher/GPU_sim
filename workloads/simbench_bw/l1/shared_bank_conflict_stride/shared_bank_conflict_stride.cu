// slug: shared_bank_conflict_stride
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Stages a (rows * cols * stride) float tile through shared memory.
// The kernel performs two read passes: a strided read (bank-conflict
// stress) followed by an identity read; the strided value is XOR'd
// into a sink to prevent DCE while the identity read produces the
// output. Cubin contains LDS/STS instructions per per-row contract.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void shared_bank_conflict_stride_kernel(
    const float *__restrict__ in, float *__restrict__ out, float *__restrict__ sink,
    int32_t rows, int32_t cols, int32_t stride)
{
    extern __shared__ float smem[];
    const int total = rows * cols * stride;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;

    for (int i = tid; i < total; i += lanes) smem[i] = in[i];
    __syncthreads();
    // Strided read (bank-conflict stress); fold into the sink so the
    // compiler can't DCE the access.
    float local = 0.0f;
    for (int i = tid; i < total; i += lanes) {
        int permuted = (i * stride) % total;
        local += smem[permuted];
    }
    if (tid == 0) sink[0] = local;
    // Identity read for the output.
    for (int i = tid; i < total; i += lanes) {
        out[i] = smem[i];
    }
}

extern "C" void shared_bank_conflict_stride_launch(
    const float *d_in, float *d_out, float *d_sink,
    int rows, int cols, int stride, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    size_t shmem = (size_t)rows * cols * stride * sizeof(float);
    shared_bank_conflict_stride_kernel<<<grid, block, shmem, stream>>>(d_in, d_out, d_sink, rows, cols, stride);
}
