// Host driver for warp_divergence_reconvergence.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../sm120_event_timing.h"

extern "C" void warp_divergence_reconvergence_launch(const float *d_in, float *d_out, float *d_sink,
                                                      int warps, int branches, int iterations,
                                                      cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "warp_divergence_reconvergence_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int warps = 4, branches = 2, iterations = 4;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--warps")      == 0 && i + 1 < argc) warps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--branches")   == 0 && i + 1 < argc) branches = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")       == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "warp_divergence_reconvergence_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n = (size_t)warps * branches * iterations;
    float *h_in = (float *)malloc(n * sizeof(float));
    float *h_out = (float *)malloc(n * sizeof(float));
    seed_tensor(h_in, n, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr, *d_sink = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_in,   n * sizeof(float)); if (rc) fail("mc in",   rc);
    rc = cudaMalloc(&d_out,  n * sizeof(float)); if (rc) fail("mc out",  rc);
    rc = cudaMalloc(&d_sink, sizeof(float));     if (rc) fail("mc sink", rc);
    rc = cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D", rc);

    Sm120CudaEventScope event_scope;
    const int event_reps = sm120_event_reps();
    for (int rep = 0; rep < event_reps; ++rep) {
        warp_divergence_reconvergence_launch(d_in, d_out, d_sink, warps, branches, iterations, 0);
    }
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    event_scope.finish("warp_divergence_reconvergence");
    rc = cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"warp_divergence_reconvergence\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int w = 0; w < warps; ++w) {
        if (w > 0) printf(",");
        printf("[");
        for (int b = 0; b < branches; ++b) {
            if (b > 0) printf(",");
            printf("[");
            for (int it = 0; it < iterations; ++it) {
                if (it > 0) printf(",");
                printf("%.6g", h_out[(w * branches + b) * iterations + it]);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_sink);
    free(h_in); free(h_out);
    return 0;
}
