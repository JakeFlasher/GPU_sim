// Host driver for global_scalar_32b_copy (32-bit scalar member of the
// vector-width request/sector-formation pair). Same I/O + seeding + verified
// output (identity round-trip on `out`) as the 128-bit partner; only the
// launched kernel's per-access request width differs (scalar float -> LDG.E).

#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void global_scalar_32b_copy_launch(const float *d_in, float *d_out,
                                              int n, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "global_scalar_32b_copy_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

static uint32_t parse_u32(const char *name, const char *text, uint32_t maximum) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || !end || *end || value == 0 || value > maximum) {
        fprintf(stderr, "global_scalar_32b_copy_run: out_of_domain: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

int main(int argc, char **argv) {
    int samples = 32, bins = 8;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    bool summary_only = false;
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--samples")     == 0 && i + 1 < argc) samples = (int)parse_u32("samples", argv[++i], INT_MAX);
        else if (strcmp(argv[i], "--bins")        == 0 && i + 1 < argc) bins = (int)parse_u32("bins", argv[++i], INT_MAX);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed = parse_u32("seed", argv[++i], UINT32_MAX);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--summary") == 0) summary_only = true;
        else { fprintf(stderr, "global_scalar_32b_copy_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (strspn(shape_point, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape_point)) {
        fprintf(stderr, "global_scalar_32b_copy_run: out_of_domain: invalid shape-point token\n");
        return 2;
    }
    const uint64_t elements = (uint64_t)samples * (uint64_t)bins;
    if (elements > (uint64_t)INT_MAX - 32 || elements > SIZE_MAX / (2 * sizeof(float))) {
        fprintf(stderr, "global_scalar_32b_copy_run: out_of_domain: shape overflows index or allocation\n");
        return 2;
    }
    const size_t n = (size_t)elements;
    const size_t bytes = n * sizeof(float);
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc) fail("memory query", rc);
    if (bytes > free_bytes / 2) {
        fprintf(stderr, "global_scalar_32b_copy_run: out of memory: requested shape exceeds free device memory\n");
        return 1;
    }
    float *h_in = (float *)malloc(n * sizeof(float));
    float *h_out = (float *)malloc(n * sizeof(float));
    if (!h_in || !h_out) {
        fprintf(stderr, "global_scalar_32b_copy_run: out of memory: host allocation failed\n");
        free(h_in); free(h_out);
        return 1;
    }
    seed_tensor(h_in, n, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr;
    rc = cudaMalloc(&d_in,  bytes); if (rc) fail("mc in",  rc);
    rc = cudaMalloc(&d_out, bytes); if (rc) fail("mc out", rc);
    rc = cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice); if (rc) fail("H2D", rc);

    global_scalar_32b_copy_launch(d_in, d_out, (int)n, 0);
    rc = cudaGetLastError(); if (rc) fail("launch", rc);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);
    size_t mismatch_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(h_out[i]) || h_out[i] != h_in[i]) ++mismatch_count;
    }
    const bool correct = mismatch_count == 0;

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"global_scalar_32b_copy\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"samples\": %d, \"bins\": %d, \"seed\": %u,\n", samples, bins, seed);
    printf("  \"correct\": %s, \"mismatch_count\": %zu", correct ? "true" : "false", mismatch_count);
    if (!summary_only) {
    printf(",\n");
    printf("  \"output\": [");
    for (int s = 0; s < samples; ++s) {
        if (s > 0) printf(",");
        printf("[");
        for (int b = 0; b < bins; ++b) {
            if (b > 0) printf(",");
            printf("%.6g", h_out[s * bins + b]);
        }
        printf("]");
    }
    printf("]");
    }
    printf("\n}\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return correct ? 0 : 1;
}
