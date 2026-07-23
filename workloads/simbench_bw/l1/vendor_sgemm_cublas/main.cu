// Host driver for vendor_sgemm_cublas. FP32 GEMM via cublasSgemm
// under CUBLAS_PEDANTIC_MATH (no TF32, no tensor cores).

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" int vendor_sgemm_cublas_launch(const float *d_A, const float *d_B,
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
    fprintf(stderr, "vendor_sgemm_cublas_run: %s (rc=%d)\n", msg, rc);
    exit(1);
}

int main(int argc, char **argv) {
    int M = 32, N = 32, K = 32;
    uint32_t seed = 42;
    int warmup = 0;
    int reps = 1;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        // Legacy workload-mode aliases for compatibility with
        // tools/local_labels/label_driver.sh's shape_point=timing_shape /
        // correctness_shape special-cases.
        else if (strcmp(argv[i], "--timing_shape") == 0) shape_point = "timing_shape";
        else if (strcmp(argv[i], "--correctness_shape") == 0) shape_point = "correctness_shape";
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        else { fprintf(stderr, "vendor_sgemm_cublas_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n_a = (size_t)M * K, n_b = (size_t)K * N, n_c = (size_t)M * N;
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A, n_a * sizeof(float)); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, n_b * sizeof(float)); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, n_c * sizeof(float)); if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, A_f, n_a * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, B_f, n_b * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    // Optional warmup launches before the timed/profiled invocation.
    // label_driver.sh's shape_point=timing_shape auto-appends --warmup <count>
    // so the runner must accept the flag.
    for (int w = 0; w < warmup; ++w) {
        int wc = vendor_sgemm_cublas_launch(d_A, d_B, d_C, M, N, K, 0);
        if (wc) fail("cublas warmup", wc);
    }
    if (warmup > 0) {
        rc = cudaDeviceSynchronize();
        if (rc) fail("sync warmup", rc);
    }

    // Timed/profiled invocations. --reps controls how many measured
    // launches the binary issues (default 1 preserves the existing
    // single-launch behavior).
    for (int r = 0; r < reps; ++r) {
        int rc2 = vendor_sgemm_cublas_launch(d_A, d_B, d_C, M, N, K, 0);
        if (rc2) fail("cublas", rc2);
    }
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"vendor_sgemm_cublas\",\n");
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
    free(A_f); free(B_f); free(h_C);
    return 0;
}
