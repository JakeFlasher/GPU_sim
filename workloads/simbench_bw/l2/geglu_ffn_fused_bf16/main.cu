// Host driver for geglu_ffn_fused_bf16. Seeds inputs via the same PRNG as
// references/_common.py::deterministic_seeded_tensor so the GPU output can
// be compared against the Python reference under the BF16 activation
// tolerance.
//
// Matches references/activation.py::build_reference:
//   x      = seeded((tokens, hidden),         seed,     [-1, 1))
//   w_gate = seeded((hidden, intermediate),   seed + 1, [-0.1, 0.1))
//   w_up   = seeded((hidden, intermediate),   seed + 2, [-0.1, 0.1))
//   w_down = seeded((intermediate, hidden),   seed + 3, [-0.1, 0.1))

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void geglu_ffn_fused_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_w_gate,
    const __nv_bfloat16 *d_w_up,
    const __nv_bfloat16 *d_w_down,
    __nv_bfloat16 *d_out,
    int tokens, int hidden, int intermediate, cudaStream_t stream);


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
    fprintf(stderr, "geglu_ffn_fused_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int tokens = 128;
    int hidden = 512;
    int intermediate = 1024;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "geglu_ffn_fused_bf16";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--intermediate") == 0 && i + 1 < argc) intermediate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "geglu_ffn_fused_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_x = (size_t)tokens * hidden;
    const size_t n_wg = (size_t)hidden * intermediate;
    const size_t n_wd = (size_t)intermediate * hidden;
    const size_t n_out = (size_t)tokens * hidden;

    float *h_x_f32    = (float *)malloc(n_x  * sizeof(float));
    float *h_wg_f32   = (float *)malloc(n_wg * sizeof(float));
    float *h_wu_f32   = (float *)malloc(n_wg * sizeof(float));
    float *h_wd_f32   = (float *)malloc(n_wd * sizeof(float));
    seed_tensor(h_x_f32,  n_x,  seed,     -1.0, 1.0);
    seed_tensor(h_wg_f32, n_wg, seed + 1, -0.1, 0.1);
    seed_tensor(h_wu_f32, n_wg, seed + 2, -0.1, 0.1);
    seed_tensor(h_wd_f32, n_wd, seed + 3, -0.1, 0.1);

    __nv_bfloat16 *h_x  = (__nv_bfloat16 *)malloc(n_x  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_wg = (__nv_bfloat16 *)malloc(n_wg * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_wu = (__nv_bfloat16 *)malloc(n_wg * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_wd = (__nv_bfloat16 *)malloc(n_wd * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x;  ++i) h_x[i]  = __float2bfloat16(h_x_f32[i]);
    for (size_t i = 0; i < n_wg; ++i) h_wg[i] = __float2bfloat16(h_wg_f32[i]);
    for (size_t i = 0; i < n_wg; ++i) h_wu[i] = __float2bfloat16(h_wu_f32[i]);
    for (size_t i = 0; i < n_wd; ++i) h_wd[i] = __float2bfloat16(h_wd_f32[i]);

    __nv_bfloat16 *d_x = nullptr, *d_wg = nullptr, *d_wu = nullptr, *d_wd = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x,  n_x  * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_x",  rc);
    rc = cudaMalloc(&d_wg, n_wg * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_wg", rc);
    rc = cudaMalloc(&d_wu, n_wg * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_wu", rc);
    rc = cudaMalloc(&d_wd, n_wd * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_wd", rc);
    rc = cudaMalloc(&d_out, n_out * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_x,  h_x,  n_x  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x",  rc);
    rc = cudaMemcpy(d_wg, h_wg, n_wg * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D wg", rc);
    rc = cudaMemcpy(d_wu, h_wu, n_wg * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D wu", rc);
    rc = cudaMemcpy(d_wd, h_wd, n_wd * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D wd", rc);

    geglu_ffn_fused_bf16_launch(d_x, d_wg, d_wu, d_wd, d_out, tokens, hidden, intermediate, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int t = 0; t < tokens; ++t) {
        if (t > 0) printf(",");
        printf("[");
        for (int h = 0; h < hidden; ++h) {
            if (h > 0) printf(",");
            float v = __bfloat162float(h_out[t * hidden + h]);
            printf("%.6g", v);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_wg); cudaFree(d_wu); cudaFree(d_wd); cudaFree(d_out);
    free(h_x_f32); free(h_wg_f32); free(h_wu_f32); free(h_wd_f32);
    free(h_x); free(h_wg); free(h_wu); free(h_wd); free(h_out);
    return 0;
}
