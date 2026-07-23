// Host driver for warm_cold_pair_relaunch.
//   in_fp32 = seeded(allocation_floats, seed, [-1.0, 1.0))
// Reference: identity round-trip from `in` to `out` (the warm
// launch overwrites `out` with the same identity copy as the
// cold launch); correctness shape's hash digest verifies
// determinism.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void warm_cold_pair_relaunch_launch(
    const float *d_in, float *d_out, float *d_sink,
    float *d_scratch,
    int allocation_floats, int accesses,
    int eviction_floats, int warps, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) {
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}

static void seed_tensor(float *flat, size_t n, uint32_t seed,
                        double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}

static void fail(const char *msg, int rc) {
    fprintf(stderr, "warm_cold_pair_relaunch_run: %s "
            "(rc=%d: %s)\n", msg, rc,
            cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int allocation_bytes = 4096;
    int accesses = 64;
    int eviction_bytes = 0;
    int warps = 1;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--allocation-bytes") == 0 && i + 1 < argc)
            allocation_bytes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--accesses") == 0 && i + 1 < argc)
            accesses = atoi(argv[++i]);
        else if (strcmp(argv[i], "--eviction-bytes") == 0 && i + 1 < argc)
            eviction_bytes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc)
            warps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc)
            shape_point = argv[++i];
        else {
            fprintf(stderr, "warm_cold_pair_relaunch_run: "
                    "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    int allocation_floats = allocation_bytes / 4;
    if (allocation_floats < 32) allocation_floats = 32;
    int eviction_floats = eviction_bytes / 4;

    float *h_in = (float *)malloc(allocation_floats * sizeof(float));
    float *h_out = (float *)malloc(allocation_floats * sizeof(float));
    seed_tensor(h_in, allocation_floats, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr;
    float *d_sink = nullptr, *d_scratch = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_in, allocation_floats * sizeof(float));
    if (rc) fail("mc in", rc);
    rc = cudaMalloc(&d_out, allocation_floats * sizeof(float));
    if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_sink, sizeof(float));
    if (rc) fail("mc sink", rc);
    if (eviction_floats > 0) {
        rc = cudaMalloc(&d_scratch, eviction_floats * sizeof(float));
        if (rc) fail("mc scratch", rc);
        rc = cudaMemset(d_scratch, 0, eviction_floats * sizeof(float));
        if (rc) fail("memset scratch", rc);
    }
    rc = cudaMemcpy(d_in, h_in, allocation_floats * sizeof(float),
                    cudaMemcpyHostToDevice);
    if (rc) fail("H2D", rc);

    warm_cold_pair_relaunch_launch(
        d_in, d_out, d_sink, d_scratch,
        allocation_floats, accesses, eviction_floats, warps, 0);
    rc = cudaDeviceSynchronize();
    if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out,
                    allocation_floats * sizeof(float),
                    cudaMemcpyDeviceToHost);
    if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n");
    printf("  \"slug\": \"warm_cold_pair_relaunch\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    int report = allocation_floats < 64 ? allocation_floats : 64;
    for (int i = 0; i < report; ++i) {
        if (i > 0) printf(",");
        printf("%.6g", h_out[i]);
    }
    printf("]\n}\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_sink);
    if (d_scratch) cudaFree(d_scratch);
    free(h_in); free(h_out);
    return 0;
}
