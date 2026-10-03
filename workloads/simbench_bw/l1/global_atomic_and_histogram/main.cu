// Host driver for global_atomic_and_histogram.

#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void global_atomic_and_histogram_launch(const float *d_in, float *d_out, int32_t *d_histo,
                                                int samples, int bins, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "global_atomic_and_histogram_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}
static uint32_t parse_u32(const char *name, const char *text,
                          uint32_t max_value, bool zero_ok) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || *end ||
        value > max_value || (!zero_ok && value == 0)) {
        fprintf(stderr, "global_atomic_and_histogram_run: out_of_domain: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

int main(int argc, char **argv) {
    int samples = 32, bins = 8;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    bool summary = false;
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--samples")     == 0 && i + 1 < argc) samples = (int)parse_u32("samples", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--bins")        == 0 && i + 1 < argc) bins = (int)parse_u32("bins", argv[++i], INT_MAX - 32, false);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed = parse_u32("seed", argv[++i], UINT32_MAX, true);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--summary")     == 0) summary = true;
        else { fprintf(stderr, "global_atomic_and_histogram_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (!*shape_point || strspn(shape_point, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape_point)) {
        fprintf(stderr, "global_atomic_and_histogram_run: out_of_domain: invalid shape-point token\n");
        return 2;
    }
    const uint64_t n64 = (uint64_t)samples * (uint64_t)bins;
    if (n64 > (uint64_t)INT_MAX - 32 || n64 > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "global_atomic_and_histogram_run: out_of_domain: shape overflows index/allocation\n");
        return 2;
    }
    const size_t n = (size_t)n64;
    const size_t bytes = n * sizeof(float);
    const size_t histo_bytes = (size_t)bins * sizeof(int32_t);
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc) fail("memory query", rc);
    if (bytes > free_bytes / 2 || histo_bytes > free_bytes) {
        fprintf(stderr, "global_atomic_and_histogram_run: out of memory: requested shape exceeds free device memory\n");
        return 1;
    }
    float *h_in = (float *)malloc(n * sizeof(float));
    float *h_out = (float *)malloc(n * sizeof(float));
    uint32_t *h_expected = (uint32_t *)malloc((size_t)bins * sizeof(uint32_t));
    int32_t *h_histo = (int32_t *)malloc(histo_bytes);
    if (!h_in || !h_out || !h_expected || !h_histo) {
        fprintf(stderr, "global_atomic_and_histogram_run: out of memory: host allocation\n");
        free(h_in); free(h_out); free(h_expected); free(h_histo);
        return 1;
    }
    seed_tensor(h_in, n, seed, -1.0, 1.0);
    for (int b = 0; b < bins; ++b) h_expected[b] = UINT32_MAX;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t bin = ((uint32_t)i * UINT32_C(2654435769)) % (uint32_t)bins;
        h_expected[bin] &= (uint32_t)i;
    }
    for (int i = 0; i < samples; ++i) h_expected[0] &= (uint32_t)i;

    float *d_in = nullptr, *d_out = nullptr;
    int32_t *d_histo = nullptr;
    rc = cudaMalloc(&d_in, bytes); if (rc) fail("mc in", rc);
    rc = cudaMalloc(&d_out, bytes); if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_histo, histo_bytes); if (rc) fail("mc histo", rc);
    rc = cudaMemset(d_out, 0xff, bytes); if (rc) fail("memset output", rc);
    rc = cudaMemset(d_histo, 0xff, histo_bytes); if (rc) fail("memset histo", rc);
    rc = cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice); if (rc) fail("H2D", rc);

    global_atomic_and_histogram_launch(d_in, d_out, d_histo, samples, bins, 0);
    rc = cudaGetLastError(); if (rc) fail("launch", rc);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);
    rc = cudaMemcpy(h_histo, d_histo, histo_bytes, cudaMemcpyDeviceToHost); if (rc) fail("histogram D2H", rc);
    size_t copy_mismatches = 0, histogram_mismatches = 0;
    for (size_t i = 0; i < n; ++i) if (!isfinite(h_out[i]) || h_out[i] != h_in[i]) ++copy_mismatches;
    for (int b = 0; b < bins; ++b) if ((uint32_t)h_histo[b] != h_expected[b]) ++histogram_mismatches;
    const bool correct = copy_mismatches == 0 && histogram_mismatches == 0;

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"global_atomic_and_histogram\",\n");
    const char *shape_set = !strncmp(shape_point, "timing_", 7) ? "timing_shapes" :
                            !strncmp(shape_point, "trace_", 6) ? "trace_shapes" : "correctness_shapes";
    printf("  \"shape_set\": \"%s\",\n  \"shape_point\": \"%s\",\n", shape_set, shape_point);
    printf("  \"samples\": %d, \"bins\": %d, \"seed\": %u,\n", samples, bins, seed);
    printf("  \"correct\": %s, \"copy_mismatches\": %zu, \"histogram_mismatches\": %zu", correct ? "true" : "false", copy_mismatches, histogram_mismatches);
    if (!summary) {
        printf(",\n  \"output\": [");
        for (int s = 0; s < samples; ++s) {
            if (s > 0) printf(",");
            printf("[");
            for (int b = 0; b < bins; ++b) {
                if (b > 0) printf(",");
                printf("%.6g", h_out[(size_t)s * bins + b]);
            }
            printf("]");
        }
        printf("]");
    }
    printf("\n}\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_histo);
    free(h_in); free(h_out); free(h_expected); free(h_histo);
    return correct ? 0 : 1;
}
