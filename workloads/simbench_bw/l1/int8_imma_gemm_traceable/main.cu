// Host driver for int8_imma_gemm_traceable. Seeds:
//   A_fp32 = seeded((M, K), seed,     [-0.5, 0.5))
//   B_fp32 = seeded((K, N), seed + 1, [-0.5, 0.5))
// Host quantizes each input by SCALE=64 (int8 range fits since |x| <= 0.5
// gives |x * 64| <= 32). The kernel's INT32 accumulator is dequantized
// by 1 / SCALE^2 to recover an FP32 estimate of A @ B.
// Reference: references/mma_microbench.py::_fp32_matmul (FP32).

#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
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

static uint32_t parse_u32(const char *name, const char *text, uint32_t maximum, bool allow_zero) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || !end || *end || value > maximum ||
        (!allow_zero && value == 0)) {
        fprintf(stderr, "int8_imma_gemm_traceable_run: out_of_domain: invalid %s: %s\n", name, text);
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
        else { fprintf(stderr, "int8_imma_gemm_traceable_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (!*shape_point || strspn(shape_point, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape_point)) {
        fprintf(stderr, "int8_imma_gemm_traceable_run: out_of_domain: invalid shape-point token\n");
        return 2;
    }
    if (M % 16 != 0 || N % 8 != 0 || K % 32 != 0) {
        fprintf(stderr, "int8_imma_gemm_traceable_run: out_of_domain: m%%16, n%%8, and k%%32 must be zero\n");
        return 2;
    }
    const uint64_t n_a64 = (uint64_t)M * (uint64_t)K;
    const uint64_t n_b64 = (uint64_t)K * (uint64_t)N;
    const uint64_t n_c64 = (uint64_t)M * (uint64_t)N;
    if (n_a64 > SIZE_MAX / sizeof(int8_t) || n_b64 > SIZE_MAX / sizeof(int8_t) ||
        n_c64 > SIZE_MAX / sizeof(float) || n_a64 > SIZE_MAX / sizeof(float) ||
        n_b64 > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "int8_imma_gemm_traceable_run: out_of_domain: shape overflows allocation\n");
        return 2;
    }
    const size_t n_a = (size_t)n_a64, n_b = (size_t)n_b64, n_c = (size_t)n_c64;
    const size_t bytes_a = n_a * sizeof(int8_t), bytes_b = n_b * sizeof(int8_t), bytes_c = n_c * sizeof(float);
    if (bytes_a > SIZE_MAX - bytes_b || bytes_a + bytes_b > SIZE_MAX - bytes_c) {
        fprintf(stderr, "int8_imma_gemm_traceable_run: out_of_domain: shape byte count overflows\n");
        return 2;
    }
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc) fail("memory query", rc);
    const size_t device_bytes = bytes_a + bytes_b + bytes_c;
    if (device_bytes > free_bytes / 2) {
        fprintf(stderr, "int8_imma_gemm_traceable_run: out of memory: requested shape exceeds free device memory\n");
        return 1;
    }
    const float SCALE = 64.0f;
    const float dequant = 1.0f / (SCALE * SCALE);
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    int8_t *h_A = (int8_t *)malloc(n_a * sizeof(int8_t));
    int8_t *h_B = (int8_t *)malloc(n_b * sizeof(int8_t));
    if (!A_f || !B_f || !h_C || !h_A || !h_B) {
        fprintf(stderr, "int8_imma_gemm_traceable_run: out of memory: host allocation failed\n");
        free(A_f); free(B_f); free(h_C); free(h_A); free(h_B);
        return 1;
    }
    for (size_t i = 0; i < n_a; ++i) h_A[i] = quantize_int8(A_f[i], SCALE);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            h_B[n * K + k] = quantize_int8(B_f[k * N + n], SCALE);

    int8_t *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    rc = cudaMalloc(&d_A, bytes_a); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, bytes_b); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, bytes_c); if (rc) fail("mc C", rc);
    rc = cudaMemset(d_C, 0xff, bytes_c); if (rc) fail("memset C", rc);
    rc = cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    int8_imma_gemm_traceable_launch(d_A, d_B, d_C, M, N, K, dequant, 0);
    rc = cudaGetLastError(); if (rc) fail("launch", rc);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, bytes_c, cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);
    double max_abs_error = 0.0;
    size_t mismatch_count = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t accum = 0;
            for (int k = 0; k < K; ++k) {
                accum += (int32_t)h_A[(size_t)m * K + k] * (int32_t)h_B[(size_t)n * K + k];
            }
            const double reference = (double)accum * (double)dequant;
            const double error = fabs((double)h_C[(size_t)m * N + n] - reference);
            if (!isfinite((double)h_C[(size_t)m * N + n]) || error > 1.0e-6) ++mismatch_count;
            if (error > max_abs_error) max_abs_error = error;
        }
    }
    const bool correct = mismatch_count == 0;

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"int8_imma_gemm_traceable\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u,\n", M, N, K, seed);
    printf("  \"correct\": %s, \"mismatch_count\": %zu, \"max_abs_error\": %.9g, \"tolerance\": 1e-6",
           correct ? "true" : "false", mismatch_count, max_abs_error);
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
