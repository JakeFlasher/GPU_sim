// slug: fp32_fma_dep_chain
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// FP32 dependent-FMA chain. Each thread accumulates a serial dot product
// over K via a chain of FFMAs: c[m, n] += a[m, k] * b[k, n]. The reference
// is references/mma_microbench.py::_fp32_matmul over the same (m, n, k).
// The kernel is a 1-thread-per-output FFMA loop — the simplest pattern
// that exercises the FP32 FFMA dependency chain on Hopper and Blackwell.
// Tolerance: (FP32, fma_chain, mma_microbench).

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void fp32_fma_dep_chain_kernel(
    const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
    int32_t M, int32_t N, int32_t K)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(A[row * K + k], B[k * N + col], acc);
    }
    C[row * N + col] = acc;
}

extern "C" void fp32_fma_dep_chain_launch(
    const float *d_A, const float *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    int threads = 32;
    dim3 grid((M + threads - 1) / threads, N, 1);
    dim3 block(threads, 1, 1);
    fp32_fma_dep_chain_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}
