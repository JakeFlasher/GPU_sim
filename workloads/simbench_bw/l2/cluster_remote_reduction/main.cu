// Host driver for cluster_remote_reduction. Seeds the per-cluster-member
// tiles via the same PRNG as references/_common.py::deterministic_seeded_tensor
// (each cluster member c uses seed + c), runs the cross-member atomic
// reduction on GPU, and emits `[elements]` floats.
//
// Matches references/cluster_dsmem.py::build_reference for slug ==
// "cluster_remote_reduction":
//   for c in cluster_size:  tile_c = seeded((tiles, elements), seed + c)
//   out[e] = sum_c sum_t tile_c[t][e]

#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" cudaError_t cluster_remote_reduction_launch(
    const float *d_tiles,
    float *d_out,
    int cluster_size, int n_tiles, int elements, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *state) {
    *state = (*state) * 1103515245u + 12345u;
    return *state;
}

static inline float next_float(uint32_t *state, double low, double high) {
    double r = (double)next_u32(state) / 4294967296.0;
    return (float)(low + (high - low) * r);
}

static void seed_tensor(float *flat, size_t n, uint32_t seed, double low, double high) {
    uint32_t state = (seed * 2654435761u + 1u);
    for (size_t i = 0; i < n; ++i) flat[i] = next_float(&state, low, high);
}

static void fail(const char *msg, int rc) {
    fprintf(stderr, "cluster_remote_reduction_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int cluster_size = 2;
    int tiles = 4;
    int elements = 64;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "cluster_remote_reduction";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cluster-size") == 0 && i + 1 < argc) cluster_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tiles") == 0 && i + 1 < argc) tiles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--elements") == 0 && i + 1 < argc) elements = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "cluster_remote_reduction_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_per_tile = (size_t)tiles * elements;
    const size_t n_total = (size_t)cluster_size * n_per_tile;
    float *h_tiles = (float *)malloc(n_total * sizeof(float));
    float *h_out = (float *)malloc(elements * sizeof(float));
    for (int c = 0; c < cluster_size; ++c) {
        seed_tensor(h_tiles + c * n_per_tile, n_per_tile, seed + c, -1.0, 1.0);
    }

    float *d_tiles = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_tiles, n_total * sizeof(float));    if (rc) fail("cudaMalloc d_tiles", rc);
    rc = cudaMalloc(&d_out, elements * sizeof(float));     if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_tiles, h_tiles, n_total * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D tiles", rc);
    rc = cudaMemset(d_out, 0, elements * sizeof(float));   if (rc) fail("memset d_out", rc);

    cudaError_t launch_rc = cluster_remote_reduction_launch(d_tiles, d_out, cluster_size, tiles, elements, 0);
    if (launch_rc) fail("cluster launch", launch_rc);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, elements * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int e = 0; e < elements; ++e) {
        if (e > 0) printf(",");
        printf("%.6g", h_out[e]);
    }
    printf("]\n}\n");

    cudaFree(d_tiles); cudaFree(d_out);
    free(h_tiles); free(h_out);
    return 0;
}
