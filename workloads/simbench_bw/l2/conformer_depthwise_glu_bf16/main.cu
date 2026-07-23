// Host driver for conformer_depthwise_glu_bf16. Seeds:
//   x       = seeded((B, T, C),       seed,        [-0.5, 0.5))
//   W_in    = seeded((C, 2*C),         seed + 1,    [-0.1, 0.1))
//   W_dw    = seeded((C, Ks),          seed + 2,    [-0.1, 0.1))
//   W_out   = seeded((C, C),           seed + 3,    [-0.1, 0.1))
// Matches references/audio.py::_conformer_depthwise_glu.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void conformer_depthwise_glu_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_W_in, const __nv_bfloat16 *d_W_dw,
    const __nv_bfloat16 *d_W_out, __nv_bfloat16 *d_glu_scratch,
    __nv_bfloat16 *d_out,
    int B, int T, int C, int Ks, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "conformer_depthwise_glu_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 1, time = 6, channels = 4, kernel_size = 3;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc) time = atoi(argv[++i]);
        else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) channels = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kernel-size") == 0 && i + 1 < argc) kernel_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "conformer_depthwise_glu_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t n_x   = (size_t)batch * time * channels;
    const size_t n_Win = (size_t)channels * 2 * channels;
    const size_t n_Wdw = (size_t)channels * kernel_size;
    const size_t n_Wou = (size_t)channels * channels;
    const size_t n_out = n_x;

    float *x_f32 = (float *)malloc(n_x * sizeof(float));
    float *Win_f32 = (float *)malloc(n_Win * sizeof(float));
    float *Wdw_f32 = (float *)malloc(n_Wdw * sizeof(float));
    float *Wou_f32 = (float *)malloc(n_Wou * sizeof(float));
    seed_tensor(x_f32,   n_x,   seed,     -0.5, 0.5);
    seed_tensor(Win_f32, n_Win, seed + 1, -0.1, 0.1);
    seed_tensor(Wdw_f32, n_Wdw, seed + 2, -0.1, 0.1);
    seed_tensor(Wou_f32, n_Wou, seed + 3, -0.1, 0.1);

    __nv_bfloat16 *h_x   = (__nv_bfloat16 *)malloc(n_x   * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Win = (__nv_bfloat16 *)malloc(n_Win * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wdw = (__nv_bfloat16 *)malloc(n_Wdw * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wou = (__nv_bfloat16 *)malloc(n_Wou * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x;   ++i) h_x[i]   = __float2bfloat16(x_f32[i]);
    for (size_t i = 0; i < n_Win; ++i) h_Win[i] = __float2bfloat16(Win_f32[i]);
    for (size_t i = 0; i < n_Wdw; ++i) h_Wdw[i] = __float2bfloat16(Wdw_f32[i]);
    for (size_t i = 0; i < n_Wou; ++i) h_Wou[i] = __float2bfloat16(Wou_f32[i]);

    __nv_bfloat16 *d_x = nullptr, *d_Win = nullptr, *d_Wdw = nullptr, *d_Wou = nullptr,
                  *d_scratch = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x,       n_x   * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc x",   rc);
    rc = cudaMalloc(&d_Win,     n_Win * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc Win", rc);
    rc = cudaMalloc(&d_Wdw,     n_Wdw * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc Wdw", rc);
    rc = cudaMalloc(&d_Wou,     n_Wou * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc Wou", rc);
    rc = cudaMalloc(&d_scratch, n_x   * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc scratch", rc);
    rc = cudaMalloc(&d_out,     n_out * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc out", rc);
    rc = cudaMemcpy(d_x,   h_x,   n_x   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x",   rc);
    rc = cudaMemcpy(d_Win, h_Win, n_Win * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Win", rc);
    rc = cudaMemcpy(d_Wdw, h_Wdw, n_Wdw * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Wdw", rc);
    rc = cudaMemcpy(d_Wou, h_Wou, n_Wou * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Wou", rc);

    conformer_depthwise_glu_bf16_launch(d_x, d_Win, d_Wdw, d_Wou, d_scratch, d_out,
                                         batch, time, channels, kernel_size, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"conformer_depthwise_glu_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int t = 0; t < time; ++t) {
            if (t > 0) printf(",");
            printf("[");
            for (int c = 0; c < channels; ++c) {
                if (c > 0) printf(",");
                printf("%.6g", __bfloat162float(h_out[(b * time + t) * channels + c]));
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_Win); cudaFree(d_Wdw); cudaFree(d_Wou); cudaFree(d_scratch); cudaFree(d_out);
    free(x_f32); free(Win_f32); free(Wdw_f32); free(Wou_f32);
    free(h_x); free(h_Win); free(h_Wdw); free(h_Wou); free(h_out);
    return 0;
}
