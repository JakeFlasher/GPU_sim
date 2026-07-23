// Host driver for diffusion_resnet_time_block_bf16. Seeds:
//   x       = seeded((B, C, H, W),    seed,     [-0.5,  0.5))
//   t       = seeded((B, T),          seed + 1, [-0.5,  0.5))
//   W_time  = seeded((T, C),          seed + 2, [-0.25, 0.25))
//   W_conv1 = seeded((C, C, 3, 3),    seed + 3, [-0.1,  0.1))   3x3 stage-1 conv
//   W_conv2 = seeded((C, C, 3, 3),    seed + 4, [-0.1,  0.1))   3x3 stage-2 conv
// Matches references/diffusion.py::_resnet_time_block (plan-row 60 contract:
// "time embedding + conv/norm/SiLU").

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void diffusion_resnet_time_block_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_t,
    const __nv_bfloat16 *d_W_time,
    const __nv_bfloat16 *d_W_conv1,
    const __nv_bfloat16 *d_W_conv2,
    float *d_time_bias,
    float *d_h_pre1,
    float *d_h1,
    float *d_h_pre2,
    __nv_bfloat16 *d_out,
    int B, int C, int H, int W, int T,
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
    fprintf(stderr, "diffusion_resnet_time_block_bf16_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int batch = 1, channels = 4, height = 4, width = 4, time_dim = 8;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--batch")       == 0 && i + 1 < argc) batch    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--channels")    == 0 && i + 1 < argc) channels = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height")      == 0 && i + 1 < argc) height   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width")       == 0 && i + 1 < argc) width    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--time-dim")    == 0 && i + 1 < argc) time_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed     = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "diffusion_resnet_time_block_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t n_x  = (size_t)batch * channels * height * width;
    const size_t n_t  = (size_t)batch * time_dim;
    const size_t n_Wt = (size_t)time_dim * channels;
    const size_t n_Wc = (size_t)channels * channels * 3 * 3;
    const size_t n_tb = (size_t)batch * channels;

    float *x_f   = (float *)malloc(n_x  * sizeof(float));
    float *t_f   = (float *)malloc(n_t  * sizeof(float));
    float *Wt_f  = (float *)malloc(n_Wt * sizeof(float));
    float *Wc1_f = (float *)malloc(n_Wc * sizeof(float));
    float *Wc2_f = (float *)malloc(n_Wc * sizeof(float));
    seed_tensor(x_f,   n_x,  seed,     -0.5,  0.5);
    seed_tensor(t_f,   n_t,  seed + 1, -0.5,  0.5);
    seed_tensor(Wt_f,  n_Wt, seed + 2, -0.25, 0.25);
    seed_tensor(Wc1_f, n_Wc, seed + 3, -0.1,  0.1);
    seed_tensor(Wc2_f, n_Wc, seed + 4, -0.1,  0.1);

    __nv_bfloat16 *h_x   = (__nv_bfloat16 *)malloc(n_x  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_t   = (__nv_bfloat16 *)malloc(n_t  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wt  = (__nv_bfloat16 *)malloc(n_Wt * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wc1 = (__nv_bfloat16 *)malloc(n_Wc * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wc2 = (__nv_bfloat16 *)malloc(n_Wc * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_x  * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x;  ++i) h_x[i]   = __float2bfloat16(x_f[i]);
    for (size_t i = 0; i < n_t;  ++i) h_t[i]   = __float2bfloat16(t_f[i]);
    for (size_t i = 0; i < n_Wt; ++i) h_Wt[i]  = __float2bfloat16(Wt_f[i]);
    for (size_t i = 0; i < n_Wc; ++i) h_Wc1[i] = __float2bfloat16(Wc1_f[i]);
    for (size_t i = 0; i < n_Wc; ++i) h_Wc2[i] = __float2bfloat16(Wc2_f[i]);

    __nv_bfloat16 *d_x = nullptr, *d_t = nullptr, *d_Wt = nullptr;
    __nv_bfloat16 *d_Wc1 = nullptr, *d_Wc2 = nullptr, *d_out = nullptr;
    float *d_time_bias = nullptr, *d_h_pre1 = nullptr, *d_h1 = nullptr, *d_h_pre2 = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x,   n_x  * sizeof(__nv_bfloat16)); if (rc) fail("mc x",   rc);
    rc = cudaMalloc(&d_t,   n_t  * sizeof(__nv_bfloat16)); if (rc) fail("mc t",   rc);
    rc = cudaMalloc(&d_Wt,  n_Wt * sizeof(__nv_bfloat16)); if (rc) fail("mc Wt",  rc);
    rc = cudaMalloc(&d_Wc1, n_Wc * sizeof(__nv_bfloat16)); if (rc) fail("mc Wc1", rc);
    rc = cudaMalloc(&d_Wc2, n_Wc * sizeof(__nv_bfloat16)); if (rc) fail("mc Wc2", rc);
    rc = cudaMalloc(&d_out, n_x  * sizeof(__nv_bfloat16)); if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_time_bias, n_tb * sizeof(float)); if (rc) fail("mc tb", rc);
    rc = cudaMalloc(&d_h_pre1,    n_x * sizeof(float));  if (rc) fail("mc h_pre1", rc);
    rc = cudaMalloc(&d_h1,        n_x * sizeof(float));  if (rc) fail("mc h1", rc);
    rc = cudaMalloc(&d_h_pre2,    n_x * sizeof(float));  if (rc) fail("mc h_pre2", rc);

    rc = cudaMemcpy(d_x,   h_x,   n_x  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x",   rc);
    rc = cudaMemcpy(d_t,   h_t,   n_t  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D t",   rc);
    rc = cudaMemcpy(d_Wt,  h_Wt,  n_Wt * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Wt",  rc);
    rc = cudaMemcpy(d_Wc1, h_Wc1, n_Wc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Wc1", rc);
    rc = cudaMemcpy(d_Wc2, h_Wc2, n_Wc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Wc2", rc);

    diffusion_resnet_time_block_bf16_launch(d_x, d_t, d_Wt, d_Wc1, d_Wc2,
                                             d_time_bias, d_h_pre1, d_h1, d_h_pre2, d_out,
                                             batch, channels, height, width, time_dim, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"diffusion_resnet_time_block_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int c = 0; c < channels; ++c) {
            if (c > 0) printf(",");
            printf("[");
            for (int h = 0; h < height; ++h) {
                if (h > 0) printf(",");
                printf("[");
                for (int w = 0; w < width; ++w) {
                    if (w > 0) printf(",");
                    size_t idx = ((size_t)b * channels + c) * height * width + (size_t)h * width + w;
                    printf("%.6g", __bfloat162float(h_out[idx]));
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_t); cudaFree(d_Wt); cudaFree(d_Wc1); cudaFree(d_Wc2); cudaFree(d_out);
    cudaFree(d_time_bias); cudaFree(d_h_pre1); cudaFree(d_h1); cudaFree(d_h_pre2);
    free(x_f); free(t_f); free(Wt_f); free(Wc1_f); free(Wc2_f);
    free(h_x); free(h_t); free(h_Wt); free(h_Wc1); free(h_Wc2); free(h_out);
    return 0;
}
