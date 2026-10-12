// Host driver for vendor_gemm_cublas. BF16 GEMM via cublasGemmEx.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" int vendor_gemm_cublas_launch(const __nv_bfloat16 *d_A, const __nv_bfloat16 *d_B,
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
    fprintf(stderr, "vendor_gemm_cublas_run: %s (rc=%d)\n", msg, rc);
    exit(1);
}
static uint32_t parse_u32(const char *name, const char *text,
                          uint32_t max_value, bool zero_ok) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || *end ||
        value > max_value || (!zero_ok && value == 0)) {
        fprintf(stderr, "vendor_gemm_cublas_run: out_of_domain: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}
static bool valid_token(const char *text) {
    return text && *text && strspn(text, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == strlen(text);
}

int main(int argc, char **argv) {
    int M = 32, N = 32, K = 32;
    uint32_t seed = 42;
    int warmup = 0;
    int reps = 1;
    const char *shape_point = "correctness_00";
    bool summary = false;
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = (int)parse_u32("m", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = (int)parse_u32("n", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = (int)parse_u32("k", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = parse_u32("seed", argv[++i], UINT32_MAX, true);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        // Mode aliases label the run; explicit size arguments determine the workload.
        else if (strcmp(argv[i], "--timing_shape") == 0) shape_point = "timing_shape";
        else if (strcmp(argv[i], "--correctness_shape") == 0) shape_point = "correctness_shape";
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = (int)parse_u32("warmup", argv[++i], INT_MAX, true);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (int)parse_u32("reps", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--summary") == 0) summary = true;
        else { fprintf(stderr, "vendor_gemm_cublas_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (!valid_token(shape_point)) {
        fprintf(stderr, "vendor_gemm_cublas_run: out_of_domain: invalid shape-point token\n");
        return 2;
    }
    const uint64_t n_a64 = (uint64_t)M * (uint64_t)K;
    const uint64_t n_b64 = (uint64_t)K * (uint64_t)N;
    const uint64_t n_c64 = (uint64_t)M * (uint64_t)N;
    if (n_a64 == 0 || n_b64 == 0 || n_c64 == 0 ||
        n_a64 > SIZE_MAX / sizeof(__nv_bfloat16) ||
        n_b64 > SIZE_MAX / sizeof(__nv_bfloat16) ||
        n_c64 > SIZE_MAX / sizeof(float) || n_a64 > INT_MAX || n_b64 > INT_MAX || n_c64 > INT_MAX) {
        fprintf(stderr, "vendor_gemm_cublas_run: out_of_domain: shape overflows allocation/index\n");
        return 2;
    }
    const size_t n_a = (size_t)n_a64, n_b = (size_t)n_b64, n_c = (size_t)n_c64;
    const size_t bytes_a = n_a * sizeof(__nv_bfloat16), bytes_b = n_b * sizeof(__nv_bfloat16), bytes_c = n_c * sizeof(float);
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc) fail("memory query", rc);
    if (bytes_a > free_bytes / 2 || bytes_b > free_bytes / 2 || bytes_c > free_bytes / 2 ||
        bytes_a + bytes_b + bytes_c > free_bytes) {
        fprintf(stderr, "vendor_gemm_cublas_run: out of memory: requested shape exceeds free device memory\n");
        return 1;
    }
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    if (!A_f || !B_f || !h_C) {
        fprintf(stderr, "vendor_gemm_cublas_run: out of memory: FP32 host allocation\n");
        free(A_f); free(B_f); free(h_C);
        return 1;
    }
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    __nv_bfloat16 *h_A = (__nv_bfloat16 *)malloc(n_a * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_B = (__nv_bfloat16 *)malloc(n_b * sizeof(__nv_bfloat16));
    float *h_reference = nullptr;
    const bool reference_checked = n_c <= (1u << 20);
    if (!h_A || !h_B) {
        fprintf(stderr, "vendor_gemm_cublas_run: out of memory: BF16 host allocation\n");
        free(A_f); free(B_f); free(h_C); free(h_A); free(h_B);
        return 1;
    }
    if (reference_checked) {
        h_reference = (float *)malloc(n_c * sizeof(float));
        if (!h_reference) {
            fprintf(stderr, "vendor_gemm_cublas_run: out of memory: reference allocation\n");
            free(A_f); free(B_f); free(h_C); free(h_A); free(h_B);
            return 1;
        }
    }
    for (size_t i = 0; i < n_a; ++i) h_A[i] = __float2bfloat16(A_f[i]);
    for (size_t i = 0; i < n_b; ++i) h_B[i] = __float2bfloat16(B_f[i]);

    __nv_bfloat16 *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    rc = cudaMalloc(&d_A, bytes_a); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, bytes_b); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, bytes_c); if (rc) fail("mc C", rc);
    rc = cudaMemset(d_C, 0xff, bytes_c); if (rc) fail("initialize C", rc);
    rc = cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    if (reference_checked) {
        for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += (double)__bfloat162float(h_A[(size_t)m * K + k]) *
                       (double)__bfloat162float(h_B[(size_t)k * N + n]);
            h_reference[(size_t)m * N + n] = (float)sum;
        }
    }

    // Optional warmup launches before the timed/profiled invocation.
    // label_driver.sh's shape_point=timing_shape auto-appends --warmup <count>
    // so the runner must accept the flag. Warmup kernel launches use the
    // same axes as the measured launch.
    for (int w = 0; w < warmup; ++w) {
        int wc = vendor_gemm_cublas_launch(d_A, d_B, d_C, M, N, K, 0);
        if (wc) fail("cublas warmup", wc);
    }
    if (warmup > 0) {
        rc = cudaDeviceSynchronize();
        if (rc) fail("sync warmup", rc);
    }

    // Timed/profiled invocations. The NCU profiler picks one launch per
    // --launch-skip/--launch-count window; --reps controls how many
    // measured launches the binary issues (default 1 preserves the
    // existing single-launch behavior).
    for (int r = 0; r < reps; ++r) {
        int rc2 = vendor_gemm_cublas_launch(d_A, d_B, d_C, M, N, K, 0);
        if (rc2) fail("cublas", rc2);
    }
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, bytes_c, cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    size_t mismatches = 0;
    double max_abs_error = 0.0;
    if (reference_checked) {
        for (size_t i = 0; i < n_c; ++i) {
            const double error = fabs((double)h_C[i] - (double)h_reference[i]);
            if (!isfinite(h_C[i]) || error > 2.0e-2 + 2.0e-2 * fabs((double)h_reference[i])) ++mismatches;
            max_abs_error = fmax(max_abs_error, error);
        }
    } else {
        for (size_t i = 0; i < n_c; ++i) if (!isfinite(h_C[i])) ++mismatches;
    }
    const bool correct = mismatches == 0;

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"vendor_gemm_cublas\",\n");
    const char *shape_set = !strncmp(shape_point, "timing_", 7) ? "timing_shapes" :
                            !strncmp(shape_point, "trace_", 6) ? "trace_shapes" : "correctness_shapes";
    printf("  \"shape_set\": \"%s\",\n  \"shape_point\": \"%s\",\n", shape_set, shape_point);
    printf("  \"m\": %d, \"n\": %d, \"k\": %d, \"seed\": %u, \"warmup\": %d, \"reps\": %d,\n", M, N, K, seed, warmup, reps);
    printf("  \"reference_checked\": %s, \"correct\": %s, \"mismatch_count\": %zu, \"max_abs_error\": %.9g", reference_checked ? "true" : "false", correct ? "true" : "false", mismatches, max_abs_error);
    if (!summary) {
        printf(",\n  \"output\": [");
        for (int m = 0; m < M; ++m) {
            if (m > 0) printf(",");
            printf("[");
            for (int n = 0; n < N; ++n) {
                if (n > 0) printf(",");
                printf("%.6g", h_C[(size_t)m * N + n]);
            }
            printf("]");
        }
        printf("]");
    }
    printf("\n}\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A_f); free(B_f); free(h_A); free(h_B); free(h_C); free(h_reference);
    return correct ? 0 : 1;
}
