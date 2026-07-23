// Host driver for kv_cache_update_rope_bf16 (R39 repair).
// Seeds the K cache at seed+100 with shape [batch, kv_heads, context,
// head_dim], precomputes standard RoPE cos/sin tables, applies them
// per position, emits the rotated K as JSON output.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void kv_cache_update_rope_bf16_launch(
    const __nv_bfloat16 *d_k_in, const __nv_bfloat16 *d_cos, const __nv_bfloat16 *d_sin,
    __nv_bfloat16 *d_k_out,
    int batch, int kv_heads, int context, int head_dim,
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
    fprintf(stderr, "kv_cache_update_rope_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 1, context = 32, head_dim = 32;
    int q_heads = 4, kv_heads = 2, page_size = 16;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) context = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--q-heads") == 0 && i + 1 < argc) q_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) kv_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--page-size") == 0 && i + 1 < argc) page_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "kv_cache_update_rope_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }
    (void)q_heads; (void)page_size;
    if (head_dim % 2 != 0) {
        fprintf(stderr, "kv_cache_update_rope_bf16_run: head_dim must be even\n");
        return 2;
    }

    const int half = head_dim / 2;
    const size_t n_k = (size_t)batch * kv_heads * context * head_dim;
    const size_t n_ct = (size_t)context * half;

    // Seed the K cache at seed+100 (matching reference).
    float *k_f32 = (float *)malloc(n_k * sizeof(float));
    seed_tensor(k_f32, n_k, seed + 100, -0.5, 0.5);

    // Precompute cos/sin tables (standard theta schedule).
    float *cos_f = (float *)malloc(n_ct * sizeof(float));
    float *sin_f = (float *)malloc(n_ct * sizeof(float));
    const double base = 10000.0;
    for (int p = 0; p < context; ++p) {
        for (int k = 0; k < half; ++k) {
            double inv_freq = 1.0 / pow(base, (double)(2 * k) / (double)head_dim);
            cos_f[p * half + k] = (float)cos((double)p * inv_freq);
            sin_f[p * half + k] = (float)sin((double)p * inv_freq);
        }
    }

    __nv_bfloat16 *h_k = (__nv_bfloat16 *)malloc(n_k * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_cos = (__nv_bfloat16 *)malloc(n_ct * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_sin = (__nv_bfloat16 *)malloc(n_ct * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_k * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_k; ++i) h_k[i] = __float2bfloat16(k_f32[i]);
    for (size_t i = 0; i < n_ct; ++i) { h_cos[i] = __float2bfloat16(cos_f[i]); h_sin[i] = __float2bfloat16(sin_f[i]); }

    __nv_bfloat16 *d_k = nullptr, *d_cos = nullptr, *d_sin = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_k,   n_k  * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_k", rc);
    rc = cudaMalloc(&d_cos, n_ct * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_cos", rc);
    rc = cudaMalloc(&d_sin, n_ct * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_sin", rc);
    rc = cudaMalloc(&d_out, n_k  * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_k,   h_k,   n_k  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D k", rc);
    rc = cudaMemcpy(d_cos, h_cos, n_ct * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D cos", rc);
    rc = cudaMemcpy(d_sin, h_sin, n_ct * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D sin", rc);

    kv_cache_update_rope_bf16_launch(d_k, d_cos, d_sin, d_out, batch, kv_heads, context, head_dim, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n_k * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"kv_cache_update_rope_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int h = 0; h < kv_heads; ++h) {
            if (h > 0) printf(",");
            printf("[");
            for (int p = 0; p < context; ++p) {
                if (p > 0) printf(",");
                printf("[");
                for (int d = 0; d < head_dim; ++d) {
                    if (d > 0) printf(",");
                    float v = __bfloat162float(h_out[(((b * kv_heads) + h) * context + p) * head_dim + d]);
                    printf("%.6g", v);
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_k); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_out);
    free(k_f32); free(cos_f); free(sin_f); free(h_k); free(h_cos); free(h_sin); free(h_out);
    return 0;
}
