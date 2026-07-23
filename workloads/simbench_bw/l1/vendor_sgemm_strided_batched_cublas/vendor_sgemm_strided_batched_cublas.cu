// slug: vendor_sgemm_strided_batched_cublas
// tier: control
// arch_gate: sm_86 sm_90a sm_120
//
// Vendor-control row: cuBLAS FP32 strided-batched GEMM
// C_i = A_i @ B_i (i in [0, batch_count)) via `cublasSgemmStridedBatched`
// under `CUBLAS_PEDANTIC_MATH`. The math-mode lock disables TF32 /
// HMMA / tensor-core dispatch, forcing FP32 FFMA SIMT kernels. This
// makes the trace clean under the canonical NVBit tracer's tensor-core
// opcode parser (no HMMA shape variants are emitted).
//
// Distinct from the existing `vendor_sgemm_cublas` row by virtue of
// the batched API call shape: cublasSgemmStridedBatched dispatches a
// distinct cuBLAS internal kernel from cublasSgemm at the same
// (M, N, K), giving the vendor_baselines family-bucket a second
// admittable slug under split-by-slug. The strided variant keeps
// each batch's matrices contiguous via stride parameters so the
// batched call is real (not just a back-to-back loop on the host).

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" int vendor_sgemm_strided_batched_cublas_launch(
    const float *d_A, const float *d_B, float *d_C,
    int M, int N, int K, int batch_count, cudaStream_t stream)
{
    cublasHandle_t handle = nullptr;
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed: %d\n", st);
        return 1;
    }
    cublasSetStream(handle, stream);

    // Lock to FP32 FFMA SIMT path. Without this, cuBLAS Sgemm-family
    // calls may dispatch to TF32 tensor-core kernels when
    // CUBLAS_DEFAULT_MATH is in effect on Ampere/Hopper/Blackwell.
    // CUBLAS_PEDANTIC_MATH forbids any reduced-precision math,
    // including TF32, ensuring the kernel uses only FFMA instructions
    // whose trace is clean under the canonical NVBit tracer's HMMA
    // opcode parser.
    st = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSetMathMode pedantic failed: %d\n", st);
        cublasDestroy(handle);
        return 1;
    }

    const float alpha = 1.0f, beta = 0.0f;
    const long long int stride_A = (long long int)M * K;
    const long long int stride_B = (long long int)K * N;
    const long long int stride_C = (long long int)M * N;

    // C_i[m, n] = sum_k A_i[m, k] * B_i[k, n] for i in [0, batch_count).
    // cublasSgemmStridedBatched is column-major; with A row-major
    // (M x K) and B row-major (K x N) per batch, we ask cuBLAS to
    // compute B^T @ A^T = C^T (col-major), and each batch's result
    // C is then row-major.
    st = cublasSgemmStridedBatched(handle,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, M, K,
                                   &alpha,
                                   d_B, N, stride_B,
                                   d_A, K, stride_A,
                                   &beta,
                                   d_C, N, stride_C,
                                   batch_count);
    cublasDestroy(handle);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
