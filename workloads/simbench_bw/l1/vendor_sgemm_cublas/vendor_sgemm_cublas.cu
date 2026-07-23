// slug: vendor_sgemm_cublas
// tier: control
// arch_gate: sm_86 sm_90a sm_120
//
// Vendor-control row: cuBLAS FP32 GEMM C = A @ B via `cublasSgemm`
// under `CUBLAS_PEDANTIC_MATH`. The math-mode lock disables TF32 /
// HMMA / tensor-core dispatch, forcing FP32 FFMA SIMT kernels. This
// makes the trace clean under the canonical NVBit tracer's tensor-core
// opcode parser (no HMMA shape variants are emitted).
//
// Per AC17 / AC34 this row is `tier=control` and excluded from
// headline subsystem scoring. The wrapper exists so the suite can
// run a real cuBLAS call against its non-vendor rows AND satisfy the
// vendor_baselines family-bucket minimum without modifying
// `tools/nvbit_tracer/`.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" int vendor_sgemm_cublas_launch(
    const float *d_A, const float *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    cublasHandle_t handle = nullptr;
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed: %d\n", st);
        return 1;
    }
    cublasSetStream(handle, stream);

    // Lock to FP32 FFMA SIMT path. Without this, cuBLAS Sgemm may
    // dispatch to TF32 tensor-core kernels when CUBLAS_DEFAULT_MATH
    // is in effect on Ampere/Hopper/Blackwell. CUBLAS_PEDANTIC_MATH
    // forbids any reduced-precision math, including TF32, ensuring
    // the kernel uses only FFMA instructions whose trace is clean
    // under the canonical NVBit tracer's HMMA opcode parser.
    st = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSetMathMode pedantic failed: %d\n", st);
        cublasDestroy(handle);
        return 1;
    }

    const float alpha = 1.0f, beta = 0.0f;

    // C[m, n] = sum_k A[m, k] * B[k, n].
    // cublasSgemm is column-major; with A row-major (M x K) and B row-major (K x N),
    // we ask cublas to compute B^T @ A^T = C^T (col-major), and the result C is then
    // row-major.
    st = cublasSgemm(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B, N,
                     d_A, K,
                     &beta,
                     d_C, N);
    cublasDestroy(handle);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
