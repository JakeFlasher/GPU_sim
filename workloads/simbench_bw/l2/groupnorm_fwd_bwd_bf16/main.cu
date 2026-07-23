// Host driver for groupnorm_fwd_bwd_bf16 (R39 backward repair).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void groupnorm_fwd_bwd_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_gamma, const __nv_bfloat16 *d_dO,
    __nv_bfloat16 *d_dX,
    int rows, int hidden, int num_groups, float eps, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static inline float next_float(uint32_t *s, double lo, double hi) {
    double r = (double)next_u32(s) / 4294967296.0; return (float)(lo + (hi - lo) * r);
}
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) flat[i] = next_float(&st, lo, hi);
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "groupnorm_fwd_bwd_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int rows = 8, hidden = 64, num_groups = 8;
    uint32_t seed = 42;
    float eps = 1e-5f;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-groups") == 0 && i + 1 < argc) num_groups = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) eps = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "groupnorm_fwd_bwd_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (hidden % num_groups != 0) {
        fprintf(stderr, "groupnorm_fwd_bwd_bf16_run: hidden must be divisible by num_groups\n");
        return 2;
    }

    const size_t n_x = (size_t)rows * hidden, n_g = (size_t)hidden;
    float *x_f32 = (float *)malloc(n_x * sizeof(float));
    float *g_f32 = (float *)malloc(n_g * sizeof(float));
    float *dO_f32 = (float *)malloc(n_x * sizeof(float));
    seed_tensor(x_f32,  n_x, seed,     -1.0, 1.0);
    seed_tensor(g_f32,  n_g, seed + 1, 0.5, 1.5);
    seed_tensor(dO_f32, n_x, seed + 3, -0.5, 0.5);

    __nv_bfloat16 *h_x = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_g = (__nv_bfloat16 *)malloc(n_g * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_dO = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_dX = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x; ++i) h_x[i]  = __float2bfloat16(x_f32[i]);
    for (size_t i = 0; i < n_g; ++i) h_g[i]  = __float2bfloat16(g_f32[i]);
    for (size_t i = 0; i < n_x; ++i) h_dO[i] = __float2bfloat16(dO_f32[i]);

    __nv_bfloat16 *d_x = nullptr, *d_g = nullptr, *d_dO = nullptr, *d_dX = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x,  n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_x", rc);
    rc = cudaMalloc(&d_g,  n_g * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_g", rc);
    rc = cudaMalloc(&d_dO, n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_dO", rc);
    rc = cudaMalloc(&d_dX, n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_dX", rc);
    rc = cudaMemcpy(d_x,  h_x,  n_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x", rc);
    rc = cudaMemcpy(d_g,  h_g,  n_g * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D g", rc);
    rc = cudaMemcpy(d_dO, h_dO, n_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D dO", rc);

    groupnorm_fwd_bwd_bf16_launch(d_x, d_g, d_dO, d_dX, rows, hidden, num_groups, eps, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_dX, d_dX, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"groupnorm_fwd_bwd_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int r = 0; r < rows; ++r) {
        if (r > 0) printf(",");
        printf("[");
        for (int c = 0; c < hidden; ++c) {
            if (c > 0) printf(",");
            printf("%.6g", __bfloat162float(h_dX[r * hidden + c]));
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_g); cudaFree(d_dO); cudaFree(d_dX);
    free(x_f32); free(g_f32); free(dO_f32); free(h_x); free(h_g); free(h_dO); free(h_dX);
    return 0;
}
