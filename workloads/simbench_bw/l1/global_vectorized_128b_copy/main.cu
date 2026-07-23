// Host driver for global_vectorized_128b_copy (128-bit member of the
// vector-width request/sector-formation pair). Same I/O + seeding + verified
// output (identity round-trip on `out`) as the scalar partner; only the
// launched kernel's per-access request width differs (float4 -> LDG.E.128).

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void global_vectorized_128b_copy_launch(const float *d_in, float *d_out,
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
    fprintf(stderr, "global_vectorized_128b_copy_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int samples = 32, bins = 8;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--samples")     == 0 && i + 1 < argc) samples = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bins")        == 0 && i + 1 < argc) bins = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "global_vectorized_128b_copy_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n = (size_t)samples * bins;
    // The 128-bit path copies float4 elements; require a 4-divisible length so
    // the whole buffer round-trips through the vectorized path (no scalar tail
    // that would pollute the LDG.E.128 / STG.E.128 contract).
    if (n % 4 != 0) { fprintf(stderr, "global_vectorized_128b_copy_run: samples*bins must be divisible by 4 (got %zu)\n", n); return 2; }
    float *h_in = (float *)malloc(n * sizeof(float));
    float *h_out = (float *)malloc(n * sizeof(float));
    seed_tensor(h_in, n, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_in,  n * sizeof(float)); if (rc) fail("mc in",  rc);
    rc = cudaMalloc(&d_out, n * sizeof(float)); if (rc) fail("mc out", rc);
    rc = cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D", rc);

    global_vectorized_128b_copy_launch(d_in, d_out, (int)n, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"global_vectorized_128b_copy\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
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
    printf("]\n}\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
