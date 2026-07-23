// Host driver for rope2d_mm_qkv_bf16 (R39 2D positional repair).
// Precomputes separate cos/sin tables for h and w coordinates per
// references/rope.py::build_reference for `rope2d_mm_qkv_bf16`:
//   theta_k = 10000^{-2k / (head_dim / 2)} for k in [0, head_dim/4)
//   cos_h / sin_h indexed by p_h = token // grid_w
//   cos_w / sin_w indexed by p_w = token %  grid_w
// Token outputs depend on BOTH coordinates (semantic test).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void rope2d_mm_qkv_bf16_launch(
    __nv_bfloat16 *d_qkv,
    const __nv_bfloat16 *d_cos_h, const __nv_bfloat16 *d_sin_h,
    const __nv_bfloat16 *d_cos_w, const __nv_bfloat16 *d_sin_w,
    int tokens, int heads, int head_dim, int grid_h, int grid_w,
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
    fprintf(stderr, "rope2d_mm_qkv_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int tokens = 64, heads = 4, head_dim = 32;
    int grid_h = 0, grid_w = 0;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--grid-h") == 0 && i + 1 < argc) grid_h = atoi(argv[++i]);
        else if (strcmp(argv[i], "--grid-w") == 0 && i + 1 < argc) grid_w = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "rope2d_mm_qkv_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (grid_h == 0 || grid_w == 0) {
        int side = (int)sqrt((double)tokens);
        if (side * side != tokens) {
            fprintf(stderr, "rope2d_mm_qkv_bf16_run: tokens (%d) must be a perfect square or pass --grid-h/--grid-w\n", tokens);
            return 2;
        }
        grid_h = grid_w = side;
    }
    if (head_dim % 4 != 0) {
        fprintf(stderr, "rope2d_mm_qkv_bf16_run: head_dim (%d) must be divisible by 4 for 2D split\n", head_dim);
        return 2;
    }

    const int quarter = head_dim / 4;
    const size_t n_qkv = (size_t)tokens * heads * head_dim;

    // cos_h / sin_h / cos_w / sin_w tables under the head_dim/2 theta
    // schedule (each axis uses half the head dim).
    float *cos_h_f = (float *)malloc((size_t)grid_h * quarter * sizeof(float));
    float *sin_h_f = (float *)malloc((size_t)grid_h * quarter * sizeof(float));
    float *cos_w_f = (float *)malloc((size_t)grid_w * quarter * sizeof(float));
    float *sin_w_f = (float *)malloc((size_t)grid_w * quarter * sizeof(float));
    const double base = 10000.0;
    const int axis_head = head_dim / 2;
    for (int p = 0; p < grid_h; ++p) {
        for (int k = 0; k < quarter; ++k) {
            double inv_freq = 1.0 / pow(base, (double)(2 * k) / (double)axis_head);
            cos_h_f[p * quarter + k] = (float)cos((double)p * inv_freq);
            sin_h_f[p * quarter + k] = (float)sin((double)p * inv_freq);
        }
    }
    for (int p = 0; p < grid_w; ++p) {
        for (int k = 0; k < quarter; ++k) {
            double inv_freq = 1.0 / pow(base, (double)(2 * k) / (double)axis_head);
            cos_w_f[p * quarter + k] = (float)cos((double)p * inv_freq);
            sin_w_f[p * quarter + k] = (float)sin((double)p * inv_freq);
        }
    }

    float *qkv_f32 = (float *)malloc(n_qkv * sizeof(float));
    seed_tensor(qkv_f32, n_qkv, seed, -1.0, 1.0);

    __nv_bfloat16 *h_qkv   = (__nv_bfloat16 *)malloc(n_qkv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_cos_h = (__nv_bfloat16 *)malloc((size_t)grid_h * quarter * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_sin_h = (__nv_bfloat16 *)malloc((size_t)grid_h * quarter * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_cos_w = (__nv_bfloat16 *)malloc((size_t)grid_w * quarter * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_sin_w = (__nv_bfloat16 *)malloc((size_t)grid_w * quarter * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_qkv; ++i) h_qkv[i] = __float2bfloat16(qkv_f32[i]);
    for (size_t i = 0; i < (size_t)grid_h * quarter; ++i) { h_cos_h[i] = __float2bfloat16(cos_h_f[i]); h_sin_h[i] = __float2bfloat16(sin_h_f[i]); }
    for (size_t i = 0; i < (size_t)grid_w * quarter; ++i) { h_cos_w[i] = __float2bfloat16(cos_w_f[i]); h_sin_w[i] = __float2bfloat16(sin_w_f[i]); }

    __nv_bfloat16 *d_qkv = nullptr, *d_cos_h = nullptr, *d_sin_h = nullptr, *d_cos_w = nullptr, *d_sin_w = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_qkv,   n_qkv * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc qkv", rc);
    rc = cudaMalloc(&d_cos_h, (size_t)grid_h * quarter * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc cos_h", rc);
    rc = cudaMalloc(&d_sin_h, (size_t)grid_h * quarter * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc sin_h", rc);
    rc = cudaMalloc(&d_cos_w, (size_t)grid_w * quarter * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc cos_w", rc);
    rc = cudaMalloc(&d_sin_w, (size_t)grid_w * quarter * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc sin_w", rc);
    rc = cudaMemcpy(d_qkv,   h_qkv,   n_qkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D qkv", rc);
    rc = cudaMemcpy(d_cos_h, h_cos_h, (size_t)grid_h * quarter * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D cos_h", rc);
    rc = cudaMemcpy(d_sin_h, h_sin_h, (size_t)grid_h * quarter * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D sin_h", rc);
    rc = cudaMemcpy(d_cos_w, h_cos_w, (size_t)grid_w * quarter * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D cos_w", rc);
    rc = cudaMemcpy(d_sin_w, h_sin_w, (size_t)grid_w * quarter * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D sin_w", rc);

    rope2d_mm_qkv_bf16_launch(d_qkv, d_cos_h, d_sin_h, d_cos_w, d_sin_w,
                               tokens, heads, head_dim, grid_h, grid_w, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_qkv, d_qkv, n_qkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"rope2d_mm_qkv_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int p = 0; p < tokens; ++p) {
        if (p > 0) printf(",");
        printf("[");
        for (int h = 0; h < heads; ++h) {
            if (h > 0) printf(",");
            printf("[");
            for (int d = 0; d < head_dim; ++d) {
                if (d > 0) printf(",");
                printf("%.6g", __bfloat162float(h_qkv[(p * heads + h) * head_dim + d]));
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_qkv); cudaFree(d_cos_h); cudaFree(d_sin_h); cudaFree(d_cos_w); cudaFree(d_sin_w);
    free(qkv_f32); free(cos_h_f); free(sin_h_f); free(cos_w_f); free(sin_w_f);
    free(h_qkv); free(h_cos_h); free(h_sin_h); free(h_cos_w); free(h_sin_w);
    return 0;
}
