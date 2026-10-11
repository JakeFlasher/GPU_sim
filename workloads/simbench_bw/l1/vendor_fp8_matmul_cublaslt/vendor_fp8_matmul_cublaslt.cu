// slug: vendor_fp8_matmul_cublaslt
// tier: control
// arch_gate: sm_90a sm_120
//
// Vendor-control row: FP8 (E4M3) matmul via cuBLASLt's `cublasLtMatmul`.
// Per AC17 / AC34 this row is `tier=control` and excluded from
// headline subsystem scoring. The wrapper exists so the suite has a
// vendor FP8 matmul kernel that can be referenced for sanity checks
// at the same tile sizes as the in-tree FP8 mma rows.
//
// The cuBLASLt FP8 path uses E4M3 inputs and FP32 output. Per-tensor
// scaling factors are set to 1.0 (the host driver pre-quantizes to FP8
// matching the seeded reference range so no extra scaling is required).

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" int vendor_fp8_matmul_cublaslt_launch(
    const __nv_fp8_e4m3 *d_A, const __nv_fp8_e4m3 *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    cublasLtHandle_t handle = nullptr;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;

    cublasLtMatmulDesc_t opdesc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulDescCreate(&opdesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opdesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(opdesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Logical A[M,K] row-major; logical B[K,N] has been packed as [N,K]
    // row-major by the driver. GeForce FP8 requires TN (CUDA 13.2 cuBLAS).
    // Compute (A*B)^T = B^T*A^T with swapped device operands:
    // left stored COL[K,N] (logical B), transposed -> [N,K];
    // right stored COL[K,M] (logical A^T), not transposed -> [K,M];
    // output COL[N,M] has exactly the requested ROW[M,N] byte layout.
    cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F,     N, M, N);

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasLtMatmul(handle, opdesc,
                                        &alpha,
                                        d_B, a_desc,
                                        d_A, b_desc,
                                        &beta,
                                        d_C, c_desc,
                                        d_C, c_desc,
                                        nullptr, nullptr, 0, stream);

    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatmulDescDestroy(opdesc);
    cublasLtDestroy(handle);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
