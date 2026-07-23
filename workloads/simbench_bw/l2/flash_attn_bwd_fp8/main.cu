// Host driver for flash_attn_bwd_fp8 (R39 FP8 repair). Seeds
// Q/K/V/dO in FP32 then quantizes to __nv_fp8_e4m3 before H2D. The
// kernel operates on FP8 storage; output dQ is FP8 too and the host
// driver converts back to float for JSON emission (matches the
// Python reference's FP32 dQ output under the FP8 tolerance band).

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void flash_attn_bwd_fp8_launch(
    const __nv_fp8_e4m3 *d_q,
    const __nv_fp8_e4m3 *d_k,
    const __nv_fp8_e4m3 *d_v,
    const __nv_fp8_e4m3 *d_dO,
    __nv_fp8_e4m3 *d_dq,
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
    fprintf(stderr, "flash_attn_bwd_fp8_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
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
        else { fprintf(stderr, "flash_attn_bwd_fp8_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t per_batch = (size_t)heads * seq * head_dim;
    const size_t n = (size_t)batch * per_batch;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float *q32 = (float *)malloc(n * sizeof(float));
    float *k32 = (float *)malloc(n * sizeof(float));
    float *v32 = (float *)malloc(n * sizeof(float));
    float *dO32 = (float *)malloc(n * sizeof(float));
    for (int b = 0; b < batch; ++b) {
        seed_tensor(q32  + (size_t)b * per_batch, per_batch, seed + b,       -0.5, 0.5);
        seed_tensor(k32  + (size_t)b * per_batch, per_batch, seed + b + 100, -0.5, 0.5);
        seed_tensor(v32  + (size_t)b * per_batch, per_batch, seed + b + 200, -0.5, 0.5);
        seed_tensor(dO32 + (size_t)b * per_batch, per_batch, seed + b + 300, -0.5, 0.5);
    }

    __nv_fp8_e4m3 *hq = (__nv_fp8_e4m3 *)malloc(n * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 *hk = (__nv_fp8_e4m3 *)malloc(n * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 *hv = (__nv_fp8_e4m3 *)malloc(n * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 *hdO = (__nv_fp8_e4m3 *)malloc(n * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 *hdq = (__nv_fp8_e4m3 *)malloc(n * sizeof(__nv_fp8_e4m3));
    for (size_t i = 0; i < n; ++i) {
        hq[i]  = __nv_fp8_e4m3(q32[i]);
        hk[i]  = __nv_fp8_e4m3(k32[i]);
        hv[i]  = __nv_fp8_e4m3(v32[i]);
        hdO[i] = __nv_fp8_e4m3(dO32[i]);
    }

    __nv_fp8_e4m3 *dq = nullptr, *dk = nullptr, *dv = nullptr, *ddO = nullptr, *ddq = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&dq,  n * sizeof(__nv_fp8_e4m3)); if (rc) fail("cudaMalloc q",  rc);
    rc = cudaMalloc(&dk,  n * sizeof(__nv_fp8_e4m3)); if (rc) fail("cudaMalloc k",  rc);
    rc = cudaMalloc(&dv,  n * sizeof(__nv_fp8_e4m3)); if (rc) fail("cudaMalloc v",  rc);
    rc = cudaMalloc(&ddO, n * sizeof(__nv_fp8_e4m3)); if (rc) fail("cudaMalloc dO", rc);
    rc = cudaMalloc(&ddq, n * sizeof(__nv_fp8_e4m3)); if (rc) fail("cudaMalloc dq", rc);
    rc = cudaMemcpy(dq,  hq,  n * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D q",  rc);
    rc = cudaMemcpy(dk,  hk,  n * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D k",  rc);
    rc = cudaMemcpy(dv,  hv,  n * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D v",  rc);
    rc = cudaMemcpy(ddO, hdO, n * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D dO", rc);

    flash_attn_bwd_fp8_launch(dq, dk, dv, ddO, ddq, batch, heads, seq, head_dim, scale, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(hdq, ddq, n * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"flash_attn_bwd_fp8\",\n");
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
                    float v = float(hdq[(((b * heads) + h) * seq + s) * head_dim + d]);
                    printf("%.6g", v);
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(ddO); cudaFree(ddq);
    free(q32); free(k32); free(v32); free(dO32);
    free(hq); free(hk); free(hv); free(hdO); free(hdq);
    return 0;
}
