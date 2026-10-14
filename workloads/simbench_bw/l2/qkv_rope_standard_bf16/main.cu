// Host driver for qkv_rope_standard_bf16.
//
// Allocates a seeded QKV tensor [tokens, heads, head_dim] on GPU, precomputes
// cos/sin tables host-side with the same formula as references/rope.py
// (base=10000.0, yarn_scale=1.0 for the standard variant), copies them to the
// device, runs the rotation kernel, copies the rotated QKV back, and emits
// the result as JSON on stdout matching the update_goldens.py candidate
// format.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../sm120_event_timing.h"

extern "C" void qkv_rope_standard_bf16_launch(
    __nv_bfloat16 *d_qkv, const __nv_bfloat16 *d_cos, const __nv_bfloat16 *d_sin,
    int tokens, int heads, int head_dim, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *state) {
    *state = (*state) * 1103515245u + 12345u;
    return *state;
}


static void seed_tensor(float *flat, size_t n, uint32_t seed, double low, double high) {
    uint32_t state = (seed * 2654435761u + 1u);
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&state) / 4294967296.0;
        flat[i] = (float)(low + (high - low) * r);
    }
}


static void fail(const char *msg, int rc) {
    fprintf(stderr, "qkv_rope_standard_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int tokens = 128;
    int heads = 8;
    int head_dim = 64;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "qkv_rope_standard_bf16";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "qkv_rope_standard_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const int half = head_dim / 2;
    const size_t n_qkv = (size_t)tokens * heads * head_dim;
    const size_t n_ct = (size_t)tokens * half;

    // Precompute cos/sin tables in FP32 (matches references/rope.py).
    float *h_cos_f32 = (float *)malloc(n_ct * sizeof(float));
    float *h_sin_f32 = (float *)malloc(n_ct * sizeof(float));
    const double base = 10000.0;
    for (int p = 0; p < tokens; ++p) {
        for (int k = 0; k < half; ++k) {
            double inv_freq = 1.0 / pow(base, (double)(2 * k) / (double)head_dim);
            h_cos_f32[p * half + k] = (float)cos(p * inv_freq);
            h_sin_f32[p * half + k] = (float)sin(p * inv_freq);
        }
    }

    float *h_qkv_f32 = (float *)malloc(n_qkv * sizeof(float));
    seed_tensor(h_qkv_f32, n_qkv, seed, -1.0, 1.0);

    __nv_bfloat16 *h_qkv = (__nv_bfloat16 *)malloc(n_qkv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_cos = (__nv_bfloat16 *)malloc(n_ct * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_sin = (__nv_bfloat16 *)malloc(n_ct * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_qkv; ++i) h_qkv[i] = __float2bfloat16(h_qkv_f32[i]);
    for (size_t i = 0; i < n_ct; ++i) h_cos[i] = __float2bfloat16(h_cos_f32[i]);
    for (size_t i = 0; i < n_ct; ++i) h_sin[i] = __float2bfloat16(h_sin_f32[i]);

    __nv_bfloat16 *d_qkv = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_qkv, n_qkv * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_qkv", rc);
    rc = cudaMalloc(&d_cos, n_ct * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_cos", rc);
    rc = cudaMalloc(&d_sin, n_ct * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_sin", rc);
    rc = cudaMemcpy(d_qkv, h_qkv, n_qkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D qkv", rc);
    rc = cudaMemcpy(d_cos, h_cos, n_ct * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D cos", rc);
    rc = cudaMemcpy(d_sin, h_sin, n_ct * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D sin", rc);

    sm120_measure_launches("qkv_rope_standard_bf16", [&]() { qkv_rope_standard_bf16_launch(d_qkv, d_cos, d_sin, tokens, heads, head_dim, 0); },
        [&]() { rc = cudaMemcpy(d_qkv, h_qkv, n_qkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("restore in-place QKV", rc); });

    rc = cudaMemcpy(h_qkv, d_qkv, n_qkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H qkv", rc);

    // Emit JSON: output is [tokens][heads][head_dim] (matches references/rope.py)
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int p = 0; p < tokens; ++p) {
        if (p > 0) printf(",");
        printf("[");
        for (int h = 0; h < heads; ++h) {
            if (h > 0) printf(",");
            printf("[");
            for (int d = 0; d < head_dim; ++d) {
                if (d > 0) printf(",");
                float v = __bfloat162float(h_qkv[(p * heads + h) * head_dim + d]);
                printf("%.6g", v);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_qkv); cudaFree(d_cos); cudaFree(d_sin);
    free(h_qkv_f32); free(h_cos_f32); free(h_sin_f32);
    free(h_qkv); free(h_cos); free(h_sin);
    return 0;
}
