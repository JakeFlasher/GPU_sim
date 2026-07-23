// Host driver for dsmem_pairwise_latency.
//   in_fp32 = seeded((tiles, elements), seed, [-1.0, 1.0))
// Reference: references/cluster_dsmem.py::dsmem_pairwise_latency.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" cudaError_t dsmem_pairwise_latency_launch(const float *d_in, float *d_out,
                                                      int tiles, int elements, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "dsmem_pairwise_latency_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int tiles = 2, elements = 16;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--tiles")    == 0 && i + 1 < argc) tiles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--elements") == 0 && i + 1 < argc) elements = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")     == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "dsmem_pairwise_latency_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n = (size_t)tiles * elements;
    float *h_in = (float *)malloc(n * sizeof(float));
    float *h_out = (float *)malloc(n * sizeof(float));
    seed_tensor(h_in, n, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_in,  n * sizeof(float)); if (rc) fail("mc in",  rc);
    rc = cudaMalloc(&d_out, n * sizeof(float)); if (rc) fail("mc out", rc);
    rc = cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D", rc);

    rc = dsmem_pairwise_latency_launch(d_in, d_out, tiles, elements, 0); if (rc) fail("launch", rc);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"dsmem_pairwise_latency\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int t = 0; t < tiles; ++t) {
        if (t > 0) printf(",");
        printf("[");
        for (int e = 0; e < elements; ++e) {
            if (e > 0) printf(",");
            printf("%.6g", h_out[t * elements + e]);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
