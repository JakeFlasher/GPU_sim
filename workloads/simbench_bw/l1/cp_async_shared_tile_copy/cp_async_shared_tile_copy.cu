// slug: cp_async_shared_tile_copy
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Each CTA copies a [rows, cols] tile of FP32 from global into shared
// memory via `cp.async.ca.shared.global` (4-byte loads), waits via
// `cp.async.commit_group` + `cp.async.wait_group 0`, then stores the
// shared-memory tile back to a global output buffer.
//
// Per BL-20260424-kernel-contract-vs-shape-match the inline PTX is
// the contract enforcement; cubin must contain LDGSTS / cp.async
// SASS markers.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void cp_async_shared_tile_copy_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    int32_t rows, int32_t cols)
{
    extern __shared__ float smem[];
    const int total = rows * cols;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;

    // Stage: cp.async global -> shared, 4 bytes per thread per iteration.
    for (int i = tid; i < total; i += lanes) {
        float *smem_dst = smem + i;
        const float *gmem_src = in + i;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :
            : "r"((unsigned)__cvta_generic_to_shared(smem_dst)),
              "l"(gmem_src));
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // Store: shared -> global.
    for (int i = tid; i < total; i += lanes) {
        out[i] = smem[i];
    }
}

extern "C" void cp_async_shared_tile_copy_launch(
    const float *d_in, float *d_out, int rows, int cols, cudaStream_t stream)
{
    int threads = 128;
    dim3 grid(1, 1, 1);
    dim3 block(threads, 1, 1);
    size_t shmem = (size_t)rows * cols * sizeof(float);
    cp_async_shared_tile_copy_kernel<<<grid, block, shmem, stream>>>(d_in, d_out, rows, cols);
}
