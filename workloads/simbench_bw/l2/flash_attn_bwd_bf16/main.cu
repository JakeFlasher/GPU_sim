// Host driver for flash_attn_bwd_bf16. Seeds Q / K / V / dO via the same
// MWC PRNG as references/_common.py::deterministic_seeded_tensor.
// Matches references/attention.py::_compute_dq for slug
// "flash_attn_bwd_bf16". Non-causal: attention.py:95's resolver picks
// `causal = "causal" in slug and "noncausal" not in slug`, and the
// slug name `flash_attn_bwd_bf16` has no "causal" substring, so the
// reference runs the non-causal branch. The causal backward variant
// would ship as a separate `flash_attn_bwd_bf16_causal` row.
// Per-batch seeds:
//   Q  = seeded((heads, seq, head_dim), seed + b,       [-0.5, 0.5))
//   K  = seeded((heads, seq, head_dim), seed + b + 100, [-0.5, 0.5))
//   V  = seeded((heads, seq, head_dim), seed + b + 200, [-0.5, 0.5))
//   dO = seeded((heads, seq, head_dim), seed + b + 300, [-0.5, 0.5))
//
// Emits dQ as a nested [batch][heads][seq][head_dim] list in JSON — same
// shape as Q, which is what references/attention.py::_compute_dq returns.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void flash_attn_bwd_bf16_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    const __nv_bfloat16 *d_dO,
    __nv_bfloat16 *d_dq,
    int batch, int heads, int seq, int head_dim,
    float scale, cudaStream_t stream);


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
    fprintf(stderr, "flash_attn_bwd_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 1;
    int seq = 256;
    int head_dim = 64;
    int heads = 4;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "flash_attn_bwd_bf16";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "flash_attn_bwd_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_per_qkv_per_batch = (size_t)heads * seq * head_dim;
    const size_t n = (size_t)batch * n_per_qkv_per_batch;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float *h_q_f32  = (float *)malloc(n * sizeof(float));
    float *h_k_f32  = (float *)malloc(n * sizeof(float));
    float *h_v_f32  = (float *)malloc(n * sizeof(float));
    float *h_dO_f32 = (float *)malloc(n * sizeof(float));
    for (int b = 0; b < batch; ++b) {
        seed_tensor(h_q_f32  + (size_t)b * n_per_qkv_per_batch, n_per_qkv_per_batch, seed + b,       -0.5, 0.5);
        seed_tensor(h_k_f32  + (size_t)b * n_per_qkv_per_batch, n_per_qkv_per_batch, seed + b + 100, -0.5, 0.5);
        seed_tensor(h_v_f32  + (size_t)b * n_per_qkv_per_batch, n_per_qkv_per_batch, seed + b + 200, -0.5, 0.5);
        seed_tensor(h_dO_f32 + (size_t)b * n_per_qkv_per_batch, n_per_qkv_per_batch, seed + b + 300, -0.5, 0.5);
    }

    __nv_bfloat16 *h_q  = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_k  = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_v  = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_dO = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_dq = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n; ++i) h_q[i]  = __float2bfloat16(h_q_f32[i]);
    for (size_t i = 0; i < n; ++i) h_k[i]  = __float2bfloat16(h_k_f32[i]);
    for (size_t i = 0; i < n; ++i) h_v[i]  = __float2bfloat16(h_v_f32[i]);
    for (size_t i = 0; i < n; ++i) h_dO[i] = __float2bfloat16(h_dO_f32[i]);

    __nv_bfloat16 *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_dO = nullptr, *d_dq = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_q,  n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_q",  rc);
    rc = cudaMalloc(&d_k,  n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_k",  rc);
    rc = cudaMalloc(&d_v,  n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_v",  rc);
    rc = cudaMalloc(&d_dO, n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_dO", rc);
    rc = cudaMalloc(&d_dq, n * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_dq", rc);
    rc = cudaMemcpy(d_q,  h_q,  n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D q",  rc);
    rc = cudaMemcpy(d_k,  h_k,  n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D k",  rc);
    rc = cudaMemcpy(d_v,  h_v,  n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D v",  rc);
    rc = cudaMemcpy(d_dO, h_dO, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D dO", rc);

    flash_attn_bwd_bf16_launch(d_q, d_k, d_v, d_dO, d_dq, batch, heads, seq, head_dim, scale, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_dq, d_dq, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H dq", rc);

    // Reference returns [batch][heads][seq][head_dim] (shape of Q) — emit dQ identically.
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int h = 0; h < heads; ++h) {
            if (h > 0) printf(",");
            printf("[");
            for (int i = 0; i < seq; ++i) {
                if (i > 0) printf(",");
                printf("[");
                for (int d = 0; d < head_dim; ++d) {
                    if (d > 0) printf(",");
                    float v = __bfloat162float(h_dq[(((b * heads) + h) * seq + i) * head_dim + d]);
                    printf("%.6g", v);
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_dO); cudaFree(d_dq);
    free(h_q_f32); free(h_k_f32); free(h_v_f32); free(h_dO_f32);
    free(h_q); free(h_k); free(h_v); free(h_dO); free(h_dq);
    return 0;
}
