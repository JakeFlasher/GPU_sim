// Host driver for embedding_gather_vocab128k. Seeds:
//   table = seeded((vocab_size, hidden), seed, [-0.5, 0.5))
//   indices via direct MWC PRNG (seed+1) matching references/embedding.py::_build_indices.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern "C" void embedding_gather_vocab128k_launch(
    const __nv_bfloat16 *d_table, const int32_t *d_indices,
    __nv_bfloat16 *d_out,
    int batch, int width, int hidden, cudaStream_t stream);


static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void build_indices(int *flat, int batch, int width, int vocab_size, uint32_t seed) {
    uint32_t st = seed * 2654435761u + 1u;
    for (int b = 0; b < batch; ++b) {
        for (int w = 0; w < width; ++w) {
            st = st * 1103515245u + 12345u;
            flat[b * width + w] = (int)(st % (uint32_t)(vocab_size > 0 ? vocab_size : 1));
        }
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "embedding_gather_vocab128k_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 4, indices_w = 8, hidden = 32, vocab_size = 64;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--indices") == 0 && i + 1 < argc) indices_w = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) vocab_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "embedding_gather_vocab128k_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t n_E = (size_t)vocab_size * hidden;
    const size_t n_idx = (size_t)batch * indices_w;
    const size_t n_out = (size_t)batch * indices_w * hidden;

    float *E_f32 = (float *)malloc(n_E * sizeof(float));
    seed_tensor(E_f32, n_E, seed, -0.5, 0.5);

    int *h_idx = (int *)malloc(n_idx * sizeof(int));
    build_indices(h_idx, batch, indices_w, vocab_size, seed + 1);

    __nv_bfloat16 *h_E = (__nv_bfloat16 *)malloc(n_E * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_o = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_E; ++i) h_E[i] = __float2bfloat16(E_f32[i]);

    __nv_bfloat16 *d_E = nullptr, *d_o = nullptr;
    int32_t *d_idx = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_E, n_E * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc E", rc);
    rc = cudaMalloc(&d_idx, n_idx * sizeof(int32_t));   if (rc) fail("cudaMalloc idx", rc);
    rc = cudaMalloc(&d_o, n_out * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc o", rc);
    rc = cudaMemcpy(d_E, h_E, n_E * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D E", rc);
    rc = cudaMemcpy(d_idx, h_idx, n_idx * sizeof(int32_t), cudaMemcpyHostToDevice); if (rc) fail("H2D idx", rc);

    embedding_gather_vocab128k_launch(d_E, d_idx, d_o, batch, indices_w, hidden, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_o, d_o, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"embedding_gather_vocab128k\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int w = 0; w < indices_w; ++w) {
            if (w > 0) printf(",");
            printf("[");
            for (int d = 0; d < hidden; ++d) {
                if (d > 0) printf(",");
                printf("%.6g", __bfloat162float(h_o[(b * indices_w + w) * hidden + d]));
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_E); cudaFree(d_idx); cudaFree(d_o);
    free(E_f32); free(h_idx); free(h_E); free(h_o);
    return 0;
}
