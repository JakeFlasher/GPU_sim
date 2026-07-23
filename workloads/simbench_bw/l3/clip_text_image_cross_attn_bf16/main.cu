// Host driver for clip_text_image_cross_attn_bf16.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void clip_text_image_cross_attn_bf16_launch(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K, const __nv_bfloat16 *d_V,
    __nv_bfloat16 *d_out,
    int B, int H, int St, int Si, int D, float scale, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "clip_text_image_cross_attn_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int batch = 2, heads = 2, seq_text = 3, seq_image = 4, head_dim = 4;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq-text") == 0 && i + 1 < argc) seq_text = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq-image") == 0 && i + 1 < argc) seq_image = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "clip_text_image_cross_attn_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t n_Q  = (size_t)batch * heads * seq_text  * head_dim;
    const size_t n_KV = (size_t)batch * heads * seq_image * head_dim;
    const size_t n_out = n_Q;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float *Qf = (float *)malloc(n_Q * sizeof(float));
    float *Kf = (float *)malloc(n_KV * sizeof(float));
    float *Vf = (float *)malloc(n_KV * sizeof(float));
    seed_tensor(Qf, n_Q,  seed,     -0.5, 0.5);
    seed_tensor(Kf, n_KV, seed + 1, -0.5, 0.5);
    seed_tensor(Vf, n_KV, seed + 2, -0.5, 0.5);

    __nv_bfloat16 *hQ = (__nv_bfloat16 *)malloc(n_Q * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hK = (__nv_bfloat16 *)malloc(n_KV * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hV = (__nv_bfloat16 *)malloc(n_KV * sizeof(__nv_bfloat16));
    __nv_bfloat16 *ho = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_Q;  ++i) hQ[i] = __float2bfloat16(Qf[i]);
    for (size_t i = 0; i < n_KV; ++i) hK[i] = __float2bfloat16(Kf[i]);
    for (size_t i = 0; i < n_KV; ++i) hV[i] = __float2bfloat16(Vf[i]);

    __nv_bfloat16 *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dout = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&dQ,   n_Q  * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc Q",   rc);
    rc = cudaMalloc(&dK,   n_KV * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc K",   rc);
    rc = cudaMalloc(&dV,   n_KV * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc V",   rc);
    rc = cudaMalloc(&dout, n_out * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc out", rc);
    rc = cudaMemcpy(dQ, hQ, n_Q  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Q", rc);
    rc = cudaMemcpy(dK, hK, n_KV * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D K", rc);
    rc = cudaMemcpy(dV, hV, n_KV * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D V", rc);

    clip_text_image_cross_attn_bf16_launch(dQ, dK, dV, dout, batch, heads, seq_text, seq_image, head_dim, scale, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(ho, dout, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"clip_text_image_cross_attn_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int h = 0; h < heads; ++h) {
            if (h > 0) printf(",");
            printf("[");
            for (int s = 0; s < seq_text; ++s) {
                if (s > 0) printf(",");
                printf("[");
                for (int d = 0; d < head_dim; ++d) {
                    if (d > 0) printf(",");
                    printf("%.6g", __bfloat162float(ho[(((b * heads) + h) * seq_text + s) * head_dim + d]));
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dout);
    free(Qf); free(Kf); free(Vf); free(hQ); free(hK); free(hV); free(ho);
    return 0;
}
