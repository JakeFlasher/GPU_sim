// Host driver for rmsnorm_fwd_bf16. Allocates seeded input + gamma on GPU,
// runs the kernel, and dumps the output as JSON on stdout so the orchestrator
// (`run_suite.py --mode correctness --kernel rmsnorm_fwd_bf16`) can read it
// and compare against the Python reference under the BF16 norm tolerance.
//
// Input seeding mirrors references/_common.py::deterministic_seeded_tensor:
//
//     state = (seed * 2654435761u + 1u)
//     next_u32: state = state * 1103515245u + 12345u; return state
//     float32 = low + (high - low) * (double(next_u32()) / 2**32)
//
// Using double in the scaling keeps us byte-identical with the Python
// reference's seeded tensor before the BF16 cast.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" void rmsnorm_fwd_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_gamma,
    __nv_bfloat16 *d_out,
    int rows, int hidden, float eps, cudaStream_t stream);

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
    for (size_t i = 0; i < n; ++i) {
        flat[i] = next_float(&state, low, high);
    }
}


static void fail(const char *msg, int rc) {
    fprintf(stderr, "rmsnorm_fwd_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int rows = 32;
    int hidden = 512;
    uint32_t seed = 42;
    float eps = 1e-5f;
    const char *shape_point = "correctness_00";
    const char *slug = "rmsnorm_fwd_bf16";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) eps = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "rmsnorm_fwd_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_x = (size_t)rows * hidden;
    const size_t n_g = (size_t)hidden;
    float *h_x_f32 = (float *)malloc(n_x * sizeof(float));
    float *h_g_f32 = (float *)malloc(n_g * sizeof(float));
    seed_tensor(h_x_f32, n_x, seed, -1.0, 1.0);
    seed_tensor(h_g_f32, n_g, seed + 1, 0.5, 1.5);

    __nv_bfloat16 *h_x = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_g = (__nv_bfloat16 *)malloc(n_g * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x; ++i) h_x[i] = __float2bfloat16(h_x_f32[i]);
    for (size_t i = 0; i < n_g; ++i) h_g[i] = __float2bfloat16(h_g_f32[i]);

    __nv_bfloat16 *d_x = nullptr, *d_g = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x, n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_x", rc);
    rc = cudaMalloc(&d_g, n_g * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_g", rc);
    rc = cudaMalloc(&d_out, n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_x, h_x, n_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x", rc);
    rc = cudaMemcpy(d_g, h_g, n_g * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D g", rc);

    // Bracket the kernel launch with cudaEvents so the runner reports a
    // cudaEventElapsedTime-derived wall-clock measurement. This is the only
    // live timing observable for this workload. The historical CUPTI
    // Activity dual-source reconciliation contract was retired by D8 in
    // S4-C1; it survives only in frozen campaign evidence, interpreted via
    // the build witness and this directory's PRESERVED_BINARY.md.
    cudaEvent_t kernel_start, kernel_stop;
    rc = cudaEventCreate(&kernel_start); if (rc) fail("eventCreate start", rc);
    rc = cudaEventCreate(&kernel_stop); if (rc) fail("eventCreate stop", rc);
    rc = cudaEventRecord(kernel_start, 0); if (rc) fail("eventRecord start", rc);
    rmsnorm_fwd_bf16_launch(d_x, d_g, d_out, rows, hidden, eps, 0);
    rc = cudaEventRecord(kernel_stop, 0); if (rc) fail("eventRecord stop", rc);
    rc = cudaEventSynchronize(kernel_stop); if (rc) fail("eventSynchronize", rc);
    float elapsed_ms = 0.0f;
    rc = cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop);
    if (rc) fail("eventElapsedTime", rc);
    double wall_time_ms = (double)elapsed_ms;
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    // Emit JSON: the "output" field matches the Python reference's
    // list-of-lists of FP32 values (bfloat16 promoted back to float). Use
    // enough precision that bfloat16's 7-bit mantissa round-trips cleanly.
    // wall_time_ms is the cudaEventElapsedTime-derived measurement that
    // run_suite.py picks up via its existing stdout-JSON parse.
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"wall_time_ms\": %.9f,\n", wall_time_ms);
    printf("  \"output\": [");
    for (int r = 0; r < rows; ++r) {
        if (r > 0) printf(",");
        printf("[");
        for (int c = 0; c < hidden; ++c) {
            if (c > 0) printf(",");
            float v = __bfloat162float(h_out[r * hidden + c]);
            printf("%.6g", v);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_g); cudaFree(d_out);
    free(h_x_f32); free(h_g_f32); free(h_x); free(h_g); free(h_out);
    return 0;
}
