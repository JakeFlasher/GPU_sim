// Host driver for cluster_rmsnorm_dsmem_bf16. Seeds per-cluster segments
// at seed+c (low=-1.0, high=1.0) matching references/cluster_dsmem.py,
// concatenates into [rows, hidden] on host, runs the kernel.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void cluster_rmsnorm_dsmem_bf16_launch(
    const __nv_bfloat16 *d_x, __nv_bfloat16 *d_out,
    int rows, int hidden, float eps, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "cluster_rmsnorm_dsmem_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int cluster_size = 2, rows = 4, hidden = 16;
    uint32_t seed = 42;
    float eps = 1e-5f;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cluster-size") == 0 && i + 1 < argc) cluster_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "cluster_rmsnorm_dsmem_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (hidden % cluster_size != 0) {
        fprintf(stderr, "cluster_rmsnorm_dsmem_bf16_run: hidden must be divisible by cluster_size\n");
        return 2;
    }
    const int per_segment = hidden / cluster_size;
    const size_t n_x = (size_t)rows * hidden;
    float *x_f32 = (float *)calloc(n_x, sizeof(float));

    // Seed each cluster member's segment at seed + c, concatenate in order.
    for (int c = 0; c < cluster_size; ++c) {
        float *seg = (float *)malloc((size_t)rows * per_segment * sizeof(float));
        seed_tensor(seg, (size_t)rows * per_segment, seed + c, -1.0, 1.0);
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < per_segment; ++k) {
                x_f32[r * hidden + c * per_segment + k] = seg[r * per_segment + k];
            }
        }
        free(seg);
    }

    __nv_bfloat16 *h_x = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_o = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x; ++i) h_x[i] = __float2bfloat16(x_f32[i]);

    __nv_bfloat16 *d_x = nullptr, *d_o = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x, n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_x", rc);
    rc = cudaMalloc(&d_o, n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_o", rc);
    rc = cudaMemcpy(d_x, h_x, n_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x", rc);

    cluster_rmsnorm_dsmem_bf16_launch(d_x, d_o, rows, hidden, eps, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_o, d_o, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"cluster_rmsnorm_dsmem_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int r = 0; r < rows; ++r) {
        if (r > 0) printf(",");
        printf("[");
        for (int c = 0; c < hidden; ++c) {
            if (c > 0) printf(",");
            printf("%.6g", __bfloat162float(h_o[r * hidden + c]));
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_o);
    free(x_f32); free(h_x); free(h_o);
    return 0;
}
