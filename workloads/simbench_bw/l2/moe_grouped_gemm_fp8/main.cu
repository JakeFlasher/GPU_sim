// Host driver for moe_grouped_gemm_fp8. Seeds logits (FP32) / x / w_experts
// via the same MWC PRNG as references/_common.py::deterministic_seeded_tensor,
// quantizes x and w_experts to FP8 E4M3 before upload. Compares against the
// Python reference under (FP8_E4M3, fma_chain, grouped_gemm) tolerance.
//
// Seeding matches references/grouped_gemm.py for slug "moe_grouped_gemm_fp8":
//   logits       = seeded((tokens, experts),              seed,            [-2.0,  2.0))
//   x            = seeded((tokens, hidden),               seed + 1,        [-0.5,  0.5))
//   w_experts[e] = seeded((hidden, intermediate),         seed + 100 + e,  [-0.1,  0.1))
//
// Output is [tokens, intermediate] FP8. Dumped as a JSON nested list of
// floats with the standard schema_version=1 envelope.

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void moe_grouped_gemm_fp8_launch(
    const float         *d_logits,
    const __nv_fp8_e4m3 *d_x,
    const __nv_fp8_e4m3 *d_w_experts,
    __nv_fp8_e4m3 *d_out,
    int tokens, int experts, int top_k, int hidden, int intermediate,
    cudaStream_t stream);


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
    fprintf(stderr, "moe_grouped_gemm_fp8_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int tokens = 64;
    int experts = 4;
    int top_k = 2;
    int hidden = 512;
    int intermediate = 1024;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "moe_grouped_gemm_fp8";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) experts = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--intermediate") == 0 && i + 1 < argc) intermediate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "moe_grouped_gemm_fp8_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_logits = (size_t)tokens * experts;
    const size_t n_x      = (size_t)tokens * hidden;
    const size_t n_we     = (size_t)experts * hidden * intermediate;
    const size_t n_out    = (size_t)tokens * intermediate;

    float *h_l_f32  = (float *)malloc(n_logits * sizeof(float));
    float *h_x_f32  = (float *)malloc(n_x      * sizeof(float));
    float *h_we_f32 = (float *)malloc(n_we     * sizeof(float));
    seed_tensor(h_l_f32, n_logits, seed,     -2.0, 2.0);
    seed_tensor(h_x_f32, n_x,      seed + 1, -0.5, 0.5);
    const size_t per_expert = (size_t)hidden * intermediate;
    for (int e = 0; e < experts; ++e) {
        seed_tensor(h_we_f32 + (size_t)e * per_expert, per_expert, seed + 100 + e, -0.1, 0.1);
    }

    __nv_fp8_e4m3 *h_x  = (__nv_fp8_e4m3 *)malloc(n_x   * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 *h_we = (__nv_fp8_e4m3 *)malloc(n_we  * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 *h_out = (__nv_fp8_e4m3 *)malloc(n_out * sizeof(__nv_fp8_e4m3));
    for (size_t i = 0; i < n_x;  ++i) h_x[i]  = (__nv_fp8_e4m3)h_x_f32[i];
    for (size_t i = 0; i < n_we; ++i) h_we[i] = (__nv_fp8_e4m3)h_we_f32[i];

    float         *d_logits = nullptr;
    __nv_fp8_e4m3 *d_x = nullptr, *d_we = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_logits, n_logits * sizeof(float));         if (rc) fail("cudaMalloc d_logits", rc);
    rc = cudaMalloc(&d_x,      n_x      * sizeof(__nv_fp8_e4m3)); if (rc) fail("cudaMalloc d_x", rc);
    rc = cudaMalloc(&d_we,     n_we     * sizeof(__nv_fp8_e4m3)); if (rc) fail("cudaMalloc d_we", rc);
    rc = cudaMalloc(&d_out,    n_out    * sizeof(__nv_fp8_e4m3)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_logits, h_l_f32,  n_logits * sizeof(float),         cudaMemcpyHostToDevice); if (rc) fail("H2D logits", rc);
    rc = cudaMemcpy(d_x,      h_x,      n_x      * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D x", rc);
    rc = cudaMemcpy(d_we,     h_we,     n_we     * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D w_experts", rc);

    moe_grouped_gemm_fp8_launch(d_logits, d_x, d_we, d_out, tokens, experts, top_k, hidden, intermediate, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n_out * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int t = 0; t < tokens; ++t) {
        if (t > 0) printf(",");
        printf("[");
        for (int c = 0; c < intermediate; ++c) {
            if (c > 0) printf(",");
            float v = (float)h_out[(size_t)t * intermediate + c];
            printf("%.6g", v);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_logits); cudaFree(d_x); cudaFree(d_we); cudaFree(d_out);
    free(h_l_f32); free(h_x_f32); free(h_we_f32);
    free(h_x); free(h_we); free(h_out);
    return 0;
}
