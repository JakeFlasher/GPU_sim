// Host driver for working_set_capacity_sweep.
//   in_fp32 = seeded(working_set_floats, seed, [-1.0, 1.0))
// Reference: identity round-trip from `in` to `out` over the
// first `warps * 32` floats; correctness shape's hash digest
// verifies determinism.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void working_set_capacity_sweep_launch(
    const float *d_in, float *d_out, float *d_sink,
    int working_set_floats, int accesses, int warps,
    cudaStream_t stream);

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
    fprintf(stderr, "working_set_capacity_sweep_run: %s "
            "(rc=%d: %s)\n", msg, rc,
            cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int working_set_bytes = 8192;
    int accesses = 64;
    int warps = 4;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--working-set-bytes") == 0 && i + 1 < argc)
            working_set_bytes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--accesses") == 0 && i + 1 < argc)
            accesses = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc)
            warps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc)
            shape_point = argv[++i];
        else {
            fprintf(stderr, "working_set_capacity_sweep_run: "
                    "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    int working_set_floats = working_set_bytes / 4;
    if (working_set_floats < 32) working_set_floats = 32;
    int output_count = warps * 32;

    float *h_in = (float *)malloc(working_set_floats * sizeof(float));
    float *h_out = (float *)malloc(output_count * sizeof(float));
    seed_tensor(h_in, working_set_floats, seed, -1.0, 1.0);

    float *d_in = nullptr, *d_out = nullptr, *d_sink = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_in, working_set_floats * sizeof(float));
    if (rc) fail("mc in", rc);
    rc = cudaMalloc(&d_out, output_count * sizeof(float));
    if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_sink, sizeof(float));
    if (rc) fail("mc sink", rc);
    rc = cudaMemcpy(d_in, h_in, working_set_floats * sizeof(float),
                    cudaMemcpyHostToDevice);
    if (rc) fail("H2D", rc);

    working_set_capacity_sweep_launch(
        d_in, d_out, d_sink,
        working_set_floats, accesses, warps, 0);
    rc = cudaDeviceSynchronize();
    if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, output_count * sizeof(float),
                    cudaMemcpyDeviceToHost);
    if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n");
    printf("  \"slug\": \"working_set_capacity_sweep\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int w = 0; w < warps; ++w) {
        if (w > 0) printf(",");
        printf("[");
        for (int l = 0; l < 32; ++l) {
            if (l > 0) printf(",");
            printf("%.6g", h_out[w * 32 + l]);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_sink);
    free(h_in); free(h_out);
    return 0;
}
