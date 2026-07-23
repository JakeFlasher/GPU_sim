// Host driver for flash_attn_fwd_bf16_noncausal. Same per-batch seeds
// as the causal variant per references/attention.py:
//   q = seeded(heads, seq, head_dim) at seed+b        in [-0.5, 0.5)
//   k = seeded(heads, seq, head_dim) at seed+b+100    in [-0.5, 0.5)
//   v = seeded(heads, seq, head_dim) at seed+b+200    in [-0.5, 0.5)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void flash_attn_fwd_bf16_noncausal_launch(
    const __nv_bfloat16 *d_q, const __nv_bfloat16 *d_k, const __nv_bfloat16 *d_v,
    __nv_bfloat16 *d_out,
    int batch, int heads, int seq, int head_dim,
    float scale, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static inline float next_float(uint32_t *s, double lo, double hi) {
    double r = (double)next_u32(s) / 4294967296.0; return (float)(lo + (hi - lo) * r);
}
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) flat[i] = next_float(&st, lo, hi);
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "flash_attn_fwd_bf16_noncausal_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 1, seq = 64, head_dim = 32, heads = 4;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "flash_attn_fwd_bf16_noncausal_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t per_batch = (size_t)heads * seq * head_dim;
    const size_t n_q = (size_t)batch * per_batch;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float *q32 = (float *)malloc(n_q * sizeof(float));
    float *k32 = (float *)malloc(n_q * sizeof(float));
    float *v32 = (float *)malloc(n_q * sizeof(float));
    for (int b = 0; b < batch; ++b) {
        seed_tensor(q32 + (size_t)b * per_batch, per_batch, seed + b,       -0.5, 0.5);
        seed_tensor(k32 + (size_t)b * per_batch, per_batch, seed + b + 100, -0.5, 0.5);
        seed_tensor(v32 + (size_t)b * per_batch, per_batch, seed + b + 200, -0.5, 0.5);
    }
    __nv_bfloat16 *hq = (__nv_bfloat16 *)malloc(n_q * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hk = (__nv_bfloat16 *)malloc(n_q * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hv = (__nv_bfloat16 *)malloc(n_q * sizeof(__nv_bfloat16));
    __nv_bfloat16 *ho = (__nv_bfloat16 *)malloc(n_q * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_q; ++i) { hq[i] = __float2bfloat16(q32[i]); hk[i] = __float2bfloat16(k32[i]); hv[i] = __float2bfloat16(v32[i]); }

    __nv_bfloat16 *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&dq,   n_q * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc q", rc);
    rc = cudaMalloc(&dk,   n_q * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc k", rc);
    rc = cudaMalloc(&dv,   n_q * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc v", rc);
    rc = cudaMalloc(&dout, n_q * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc out", rc);
    rc = cudaMemcpy(dq, hq, n_q * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D q", rc);
    rc = cudaMemcpy(dk, hk, n_q * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D k", rc);
    rc = cudaMemcpy(dv, hv, n_q * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D v", rc);

    flash_attn_fwd_bf16_noncausal_launch(dq, dk, dv, dout, batch, heads, seq, head_dim, scale, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(ho, dout, n_q * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"flash_attn_fwd_bf16_noncausal\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int h = 0; h < heads; ++h) {
            if (h > 0) printf(",");
            printf("[");
            for (int s = 0; s < seq; ++s) {
                if (s > 0) printf(",");
                printf("[");
                for (int d = 0; d < head_dim; ++d) {
                    if (d > 0) printf(",");
                    printf("%.6g", __bfloat162float(ho[(((b * heads) + h) * seq + s) * head_dim + d]));
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    free(q32); free(k32); free(v32); free(hq); free(hk); free(hv); free(ho);
    return 0;
}
