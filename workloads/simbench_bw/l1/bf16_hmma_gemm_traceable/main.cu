// Host driver for bf16_hmma_gemm_traceable. Seeds:
//   A_fp32 = seeded((M, K), seed,     [-0.5, 0.5))   row-major
//   B_fp32 = seeded((K, N), seed + 1, [-0.5, 0.5))   row-major
// Host casts to BF16; B is transposed to col-major for MMA's .col layout.
// Output C[M, N] = A @ B in FP32 (the kernel accumulates FP32 from BF16
// inputs). Reference: references/mma_microbench.py::_fp32_matmul.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void bf16_hmma_gemm_traceable_launch(const __nv_bfloat16 *d_A, const __nv_bfloat16 *d_B,
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
    fprintf(stderr, "bf16_hmma_gemm_traceable_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

static uint32_t parse_u32(const char *name, const char *text, uint32_t maximum, bool allow_zero) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || !end || *end || value > maximum ||
        (!allow_zero && value == 0)) {
        fprintf(stderr, "bf16_hmma_gemm_traceable_run: out_of_domain: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

int main(int argc, char **argv) {
    int M = 32, N = 16, K = 64;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    bool summary_only = false;
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = (int)parse_u32("m", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = (int)parse_u32("n", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = (int)parse_u32("k", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = parse_u32("seed", argv[++i], UINT32_MAX, true);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--summary") == 0) summary_only = true;
        else { fprintf(stderr, "bf16_hmma_gemm_traceable_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (strspn(shape_point, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape_point)) {
        fprintf(stderr, "bf16_hmma_gemm_traceable_run: out_of_domain: invalid shape-point token\n");
        return 2;
    }
    if (M % 16 != 0 || N % 8 != 0 || K % 16 != 0) {
        fprintf(stderr, "bf16_hmma_gemm_traceable_run: out_of_domain: m%%16, n%%8, and k%%16 must be zero\n");
        return 2;
    }
    const uint64_t n_a64 = (uint64_t)M * (uint64_t)K;
    const uint64_t n_b64 = (uint64_t)K * (uint64_t)N;
    const uint64_t n_c64 = (uint64_t)M * (uint64_t)N;
    if (n_a64 > SIZE_MAX / sizeof(__nv_bfloat16) || n_b64 > SIZE_MAX / sizeof(__nv_bfloat16) ||
        n_c64 > SIZE_MAX / sizeof(float) || n_a64 > SIZE_MAX / sizeof(float) ||
        n_b64 > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "bf16_hmma_gemm_traceable_run: out_of_domain: shape overflows allocation\n");
        return 2;
    }
    const size_t n_a = (size_t)n_a64, n_b = (size_t)n_b64, n_c = (size_t)n_c64;
    const size_t bytes_a = n_a * sizeof(__nv_bfloat16), bytes_b = n_b * sizeof(__nv_bfloat16), bytes_c = n_c * sizeof(float);
    if (bytes_a > SIZE_MAX - bytes_b || bytes_a + bytes_b > SIZE_MAX - bytes_c) {
        fprintf(stderr, "bf16_hmma_gemm_traceable_run: out_of_domain: shape byte count overflows\n");
        return 2;
    }
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc) fail("memory query", rc);
    const size_t device_bytes = bytes_a + bytes_b + bytes_c;
    if (device_bytes > free_bytes / 2) {
        fprintf(stderr, "bf16_hmma_gemm_traceable_run: out of memory: requested shape exceeds free device memory\n");
        return 1;
    }
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    __nv_bfloat16 *h_A = (__nv_bfloat16 *)malloc(bytes_a);
    __nv_bfloat16 *h_B = (__nv_bfloat16 *)malloc(bytes_b);
    if (!A_f || !B_f || !h_C || !h_A || !h_B) {
        fprintf(stderr, "bf16_hmma_gemm_traceable_run: out of memory: host allocation failed\n");
        free(A_f); free(B_f); free(h_C); free(h_A); free(h_B);
        return 1;
    }
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    for (size_t i = 0; i < n_a; ++i) h_A[i] = __float2bfloat16(A_f[i]);
    // Transpose B from [K, N] row-major to [N, K] col-major for the MMA `.col` layout.
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            h_B[n * K + k] = __float2bfloat16(B_f[k * N + n]);

    __nv_bfloat16 *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    rc = cudaMalloc(&d_A, bytes_a); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, bytes_b); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, bytes_c); if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    bf16_hmma_gemm_traceable_launch(d_A, d_B, d_C, M, N, K, 0);
    rc = cudaGetLastError(); if (rc) fail("launch", rc);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, bytes_c, cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    const float tolerance = 1.0e-2f;
    double max_abs_error = 0.0;
    size_t mismatch_count = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float reference = 0.0f;
            for (int k = 0; k < K; ++k) {
                reference += __bfloat162float(h_A[(size_t)m * K + k]) * __bfloat162float(h_B[(size_t)n * K + k]);
            }
            const double error = fabs((double)h_C[(size_t)m * N + n] - (double)reference);
            if (!isfinite((double)h_C[(size_t)m * N + n]) || error > tolerance) ++mismatch_count;
            if (error > max_abs_error) max_abs_error = error;
        }
    }
    const bool correct = mismatch_count == 0;

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"bf16_hmma_gemm_traceable\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u,\n", M, N, K, seed);
    printf("  \"correct\": %s, \"mismatch_count\": %zu, \"max_abs_error\": %.9g, \"tolerance\": %.9g",
           correct ? "true" : "false", mismatch_count, max_abs_error, tolerance);
    if (!summary_only) {
    printf(",\n");
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
    printf("]");
    }
    printf("\n}\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A_f); free(B_f); free(h_A); free(h_B); free(h_C);
    return correct ? 0 : 1;
}
