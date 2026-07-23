// Host driver for vendor_fp8_matmul_cublaslt. FP8 E4M3 GEMM via cuBLASLt.

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" int vendor_fp8_matmul_cublaslt_launch(const __nv_fp8_e4m3 *d_A, const __nv_fp8_e4m3 *d_B,
                                                  float *d_C, int M, int N, int K, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "vendor_fp8_matmul_cublaslt_run: %s (rc=%d)\n", msg, rc);
    exit(1);
}

int main(int argc, char **argv) {
    int M = 16, N = 16, K = 32;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "vendor_fp8_matmul_cublaslt_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n_a = (size_t)M * K, n_b = (size_t)K * N, n_c = (size_t)M * N;
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    __nv_fp8_e4m3 *h_A = (__nv_fp8_e4m3 *)malloc(n_a * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 *h_B = (__nv_fp8_e4m3 *)malloc(n_b * sizeof(__nv_fp8_e4m3));
    for (size_t i = 0; i < n_a; ++i) h_A[i] = __nv_fp8_e4m3(A_f[i]);
    for (size_t i = 0; i < n_b; ++i) h_B[i] = __nv_fp8_e4m3(B_f[i]);

    __nv_fp8_e4m3 *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A, n_a * sizeof(__nv_fp8_e4m3)); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, n_b * sizeof(__nv_fp8_e4m3)); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, n_c * sizeof(float));         if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, h_A, n_a * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B, n_b * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    int rc2 = vendor_fp8_matmul_cublaslt_launch(d_A, d_B, d_C, M, N, K, 0); if (rc2) fail("cublasLt", rc2);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"vendor_fp8_matmul_cublaslt\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int m = 0; m < M; ++m) {
        if (m > 0) printf(",");
        printf("[");
        for (int n = 0; n < N; ++n) {
            if (n > 0) printf(",");
            printf("%.6g", h_C[m * N + n]);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A_f); free(B_f); free(h_A); free(h_B); free(h_C);
    return 0;
}
