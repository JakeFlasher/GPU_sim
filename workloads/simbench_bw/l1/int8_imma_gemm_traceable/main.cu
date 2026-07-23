// Host driver for int8_imma_gemm_traceable. Seeds:
//   A_fp32 = seeded((M, K), seed,     [-0.5, 0.5))
//   B_fp32 = seeded((K, N), seed + 1, [-0.5, 0.5))
// Host quantizes each input by SCALE=64 (int8 range fits since |x| <= 0.5
// gives |x * 64| <= 32). The kernel's INT32 accumulator is dequantized
// by 1 / SCALE^2 to recover an FP32 estimate of A @ B.
// Reference: references/mma_microbench.py::_fp32_matmul (FP32).

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void int8_imma_gemm_traceable_launch(const int8_t *d_A, const int8_t *d_B, float *d_C,
                                                 int M, int N, int K, float dequant_scale, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static int8_t quantize_int8(float v, float scale) {
    float q = v * scale;
    int qi = (int)lroundf(q);
    if (qi > 127) qi = 127;
    if (qi < -128) qi = -128;
    return (int8_t)qi;
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "int8_imma_gemm_traceable_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int M = 32, N = 16, K = 64;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "int8_imma_gemm_traceable_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const float SCALE = 64.0f;
    const float dequant = 1.0f / (SCALE * SCALE);
    const size_t n_a = (size_t)M * K, n_b = (size_t)K * N, n_c = (size_t)M * N;
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    int8_t *h_A = (int8_t *)malloc(n_a * sizeof(int8_t));
    int8_t *h_B = (int8_t *)malloc(n_b * sizeof(int8_t));
    for (size_t i = 0; i < n_a; ++i) h_A[i] = quantize_int8(A_f[i], SCALE);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            h_B[n * K + k] = quantize_int8(B_f[k * N + n], SCALE);

    int8_t *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A, n_a * sizeof(int8_t)); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, n_b * sizeof(int8_t)); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, n_c * sizeof(float));  if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, h_A, n_a * sizeof(int8_t), cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B, n_b * sizeof(int8_t), cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    int8_imma_gemm_traceable_launch(d_A, d_B, d_C, M, N, K, dequant, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"int8_imma_gemm_traceable\",\n");
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
