// Host driver for vendor_sgemm_strided_batched_cublas. FP32
// strided-batched GEMM via cublasSgemmStridedBatched under
// CUBLAS_PEDANTIC_MATH (no TF32, no tensor cores).

#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" int vendor_sgemm_strided_batched_cublas_launch(
    const float *d_A, const float *d_B, float *d_C,
    int M, int N, int K, int batch_count, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static uint32_t parse_u32(const char *name, const char *text, uint32_t maximum, bool allow_zero) {
    char *end = nullptr; errno = 0; unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || !end || *end || value > maximum || (!allow_zero && value == 0)) {
        fprintf(stderr, "vendor_sgemm_strided_batched_cublas_run: out_of_domain: invalid %s: %s\n", name, text); exit(2);
    }
    return (uint32_t)value;
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "vendor_sgemm_strided_batched_cublas_run: %s (rc=%d)\n", msg, rc);
    exit(1);
}

int main(int argc, char **argv) {
    int M = 32, N = 32, K = 32;
    int batch_count = 4;
    uint32_t seed = 42;
    int warmup = 0;
    int reps = 1;
    const char *shape_point = "correctness_00";
    bool summary_only = false;
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = (int)parse_u32("m", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = (int)parse_u32("n", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = (int)parse_u32("k", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--batch-count") == 0 && i + 1 < argc) batch_count = (int)parse_u32("batch-count", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = parse_u32("seed", argv[++i], UINT32_MAX, true);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        // Legacy workload-mode aliases for compatibility with
        // tools/local_labels/label_driver.sh's shape_point=timing_shape /
        // correctness_shape special-cases.
        else if (strcmp(argv[i], "--timing_shape") == 0) shape_point = "timing_shape";
        else if (strcmp(argv[i], "--correctness_shape") == 0) shape_point = "correctness_shape";
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = (int)parse_u32("warmup", argv[++i], INT_MAX, true);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (int)parse_u32("reps", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--summary") == 0) summary_only = true;
        else { fprintf(stderr, "vendor_sgemm_strided_batched_cublas_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (!*shape_point || strspn(shape_point, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape_point)) { fprintf(stderr, "vendor_sgemm_strided_batched_cublas_run: out_of_domain: invalid shape-point token\n"); return 2; }
    const uint64_t n_a64=(uint64_t)M*K*batch_count,n_b64=(uint64_t)K*N*batch_count,n_c64=(uint64_t)M*N*batch_count;
    if (n_a64>SIZE_MAX/sizeof(float)||n_b64>SIZE_MAX/sizeof(float)||n_c64>SIZE_MAX/sizeof(float)||n_a64>INT_MAX||n_b64>INT_MAX||n_c64>INT_MAX) { fprintf(stderr,"vendor_sgemm_strided_batched_cublas_run: out_of_domain: shape overflows allocation/index\n"); return 2; }
    const size_t n_a = (size_t)M * K * batch_count;
    const size_t n_b = (size_t)K * N * batch_count;
    const size_t n_c = (size_t)M * N * batch_count;
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    if (!A_f || !B_f || !h_C) { fprintf(stderr,"vendor_sgemm_strided_batched_cublas_run: out of memory: host allocation\n"); return 1; }
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
        int wc = vendor_sgemm_strided_batched_cublas_launch(d_A, d_B, d_C, M, N, K, batch_count, 0);
        if (wc) fail("cublas warmup", wc); rc = cudaGetLastError(); if (rc) fail("warmup launch", rc);
    }
    if (warmup > 0) {
        rc = cudaDeviceSynchronize();
        if (rc) fail("sync warmup", rc);
    }

    // Timed/profiled invocations. --reps controls how many measured
    // launches the binary issues (default 1 preserves the single-
    // launch behavior of the non-batched vendor row).
    for (int r = 0; r < reps; ++r) {
        int rc2 = vendor_sgemm_strided_batched_cublas_launch(d_A, d_B, d_C, M, N, K, batch_count, 0);
        if (rc2) fail("cublas", rc2); rc = cudaGetLastError(); if (rc) fail("launch", rc2);
    }
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    size_t mismatch_count = 0;
    double max_abs_error = 0.0;
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float reference = 0.0f;
        for (int k = 0; k < K; ++k) reference += A_f[(size_t)m * K + k] * B_f[(size_t)k * N + n];
        const double error = fabs((double)h_C[(size_t)m * N + n] - (double)reference);
        if (!isfinite(h_C[(size_t)m * N + n]) || error > 1.0e-3) ++mismatch_count;
        if (error > max_abs_error) max_abs_error = error;
    }
    const bool correct = mismatch_count == 0;
    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"vendor_sgemm_strided_batched_cublas\",\n");
    printf("  \"shape_set\": \"%s\",\n  \"shape_point\": \"%s\",\n", strncmp(shape_point, "timing_", 7) == 0 ? "timing_shapes" : "correctness_shapes", shape_point);
    printf("  \"batch_count\": %d, \"correct\": %s, \"mismatch_count\": %zu, \"max_abs_error\": %.9g", batch_count, correct ? "true" : "false", mismatch_count, max_abs_error);
    // Print just batch 0's output matrix to keep the correctness payload
    // size bounded (the existing non-batched vendor row prints only one
    // matrix; we mirror that here for parity). The matrix is emitted under
    // the canonical `output` field the correctness harness reads
    // (handle_correctness -> kernel_result.get("output")); the control
    // reference compares against batch 0's single-matmul reference.
    if (!summary_only) printf(",\n  \"output\": [");
    if (!summary_only) {
    for (int m = 0; m < M; ++m) {
        if (m > 0) printf(",");
        printf("[");
        for (int n = 0; n < N; ++n) {
            if (n > 0) printf(",");
            printf("%.6g", h_C[m * N + n]);
        }
        printf("]");
    }
    if (!summary_only) printf("]");
    printf("\n}\n");
    }

    if (summary_only) printf("\n}\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A_f); free(B_f); free(h_C);
    return correct ? 0 : 1;
}
