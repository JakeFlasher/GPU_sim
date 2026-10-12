// slug: vendor_gemm_cublas
// tier: control
// arch_gate: sm_86 sm_90a sm_120
//
// Vendor-control row: cuBLAS BF16 GEMM C = A @ B via `cublasGemmEx`.
// Per AC17 / AC34 this row is `tier=control` and excluded from
// headline subsystem scoring. The wrapper exists so the suite can
// run real cuBLAS as a provenance sanity-check against its
// non-vendor rows.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" int vendor_gemm_cublas_launch(
    const __nv_bfloat16 *d_A, const __nv_bfloat16 *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    cublasHandle_t handle = nullptr;
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed: %d\n", st);
        return 1;
    }
    st = cublasSetStream(handle, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSetStream failed: %d\n", st);
        cublasDestroy(handle);
        return 1;
    }
    const float alpha = 1.0f, beta = 0.0f;

    // C[m, n] = sum_k A[m, k] * B[k, n].
    // cublasGemmEx is column-major; with A row-major (M x K) and B row-major (K x N),
    // we ask cublas to compute B^T @ A^T = C^T (col-major), and the result C is then
    // row-major. Equivalent expression:
    //   cublasGemmEx(handle, OP_N, OP_N, N, M, K, &alpha,
    //                d_B, R_16BF, N, d_A, R_16BF, K, &beta, d_C, R_32F, N, ...);
    st = cublasGemmEx(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      N, M, K,
                      &alpha,
                      d_B, CUDA_R_16BF, N,
                      d_A, CUDA_R_16BF, K,
                      &beta,
                      d_C, CUDA_R_32F, N,
                      CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
