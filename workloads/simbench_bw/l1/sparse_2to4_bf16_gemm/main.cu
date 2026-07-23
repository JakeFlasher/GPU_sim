// Host driver for sparse_2to4_bf16_gemm.
//
// Seeds A (M x K) and B (K x N) via the same MWC PRNG as
// references/_common.py::deterministic_seeded_tensor (low=-0.5,
// high=0.5), with A seeded via `seed` and B via `seed + 1` — matching
// references/mma_microbench.py::build_reference for the GEMM-like
// branch.
//
// Applies the deterministic 2:4 mask (keep K-indices {0, 1} of every
// 4-group; zero {2, 3}) on A and packs the kept non-zero values into
// a compressed `A_sparse[M][K/2]` BF16 buffer, then casts B to BF16
// col-major ([N][K] layout, K innermost per column) so the kernel's
// `.col` MMA sees the right layout.  Emits the resulting [M x N] FP32
// output as a nested `[[...]]` list.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void sparse_2to4_bf16_gemm_launch(
    const __nv_bfloat16 *d_A_sparse,
    const __nv_bfloat16 *d_B,
    float               *d_C,
    int                  M,
    int                  N,
    int                  K,
    cudaStream_t         stream);


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
    fprintf(stderr, "sparse_2to4_bf16_gemm_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int M = 64, N = 64, K = 256;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "sparse_2to4_bf16_gemm";

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m")    == 0 && i + 1 < argc) M    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n")    == 0 && i + 1 < argc) N    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k")    == 0 && i + 1 < argc) K    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "sparse_2to4_bf16_gemm_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    if ((M % 16) != 0 || (N % 8) != 0 || (K % 32) != 0) {
        fprintf(stderr, "sparse_2to4_bf16_gemm_run: "
                "M must be a multiple of 16, N of 8, K of 32 (got M=%d N=%d K=%d)\n",
                M, N, K);
        return 2;
    }

    const size_t nA = (size_t)M * K;
    const size_t nB = (size_t)K * N;
    const size_t nC = (size_t)M * N;
    const size_t nAs = (size_t)M * (K / 2);   // compressed A: half the K of dense

    float *h_A_f32 = (float *)malloc(nA * sizeof(float));
    float *h_B_f32 = (float *)malloc(nB * sizeof(float));
    float *h_C     = (float *)malloc(nC * sizeof(float));

    seed_tensor(h_A_f32, nA, seed,     -0.5, 0.5);
    seed_tensor(h_B_f32, nB, seed + 1, -0.5, 0.5);

    // Build compressed A BF16 in row-major [M][K/2] by packing the
    // kept positions (indices {0, 1} of each 4-group in K). The
    // compressed column index is `(k / 4) * 2 + (k % 2)` for
    // original dense column `k` with `k % 4 in {0, 1}`.
    __nv_bfloat16 *h_A_sparse = (__nv_bfloat16 *)malloc(nAs * sizeof(__nv_bfloat16));
    for (int i = 0; i < M; ++i) {
        for (int g = 0; g < K / 4; ++g) {
            // Kept: dense K = 4g + 0 and 4g + 1.
            const float v0 = h_A_f32[(size_t)i * K + (4 * g + 0)];
            const float v1 = h_A_f32[(size_t)i * K + (4 * g + 1)];
            h_A_sparse[(size_t)i * (K / 2) + (2 * g + 0)] = __float2bfloat16(v0);
            h_A_sparse[(size_t)i * (K / 2) + (2 * g + 1)] = __float2bfloat16(v1);
        }
    }

    // Build col-major B in BF16: [N][K], with K innermost per column.
    // (Input h_B_f32 is row-major [K][N]; transpose at host side.)
    __nv_bfloat16 *h_B_col = (__nv_bfloat16 *)malloc(nB * sizeof(__nv_bfloat16));
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            h_B_col[(size_t)j * K + k] = __float2bfloat16(h_B_f32[(size_t)k * N + j]);
        }
    }

    __nv_bfloat16 *d_A_sparse = nullptr;
    __nv_bfloat16 *d_B        = nullptr;
    float         *d_C        = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A_sparse, nAs * sizeof(__nv_bfloat16));  if (rc) fail("cudaMalloc A_sparse", rc);
    rc = cudaMalloc(&d_B,        nB  * sizeof(__nv_bfloat16));  if (rc) fail("cudaMalloc B", rc);
    rc = cudaMalloc(&d_C,        nC  * sizeof(float));          if (rc) fail("cudaMalloc C", rc);
    rc = cudaMemcpy(d_A_sparse, h_A_sparse, nAs * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice); if (rc) fail("H2D A_sparse", rc);
    rc = cudaMemcpy(d_B, h_B_col, nB * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    sparse_2to4_bf16_gemm_launch(d_A_sparse, d_B, d_C, M, N, K, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, nC * sizeof(float), cudaMemcpyDeviceToHost);
    if (rc) fail("D2H C", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int i = 0; i < M; ++i) {
        if (i > 0) printf(",");
        printf("[");
        for (int j = 0; j < N; ++j) {
            if (j > 0) printf(",");
            printf("%.6g", h_C[i * N + j]);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_A_sparse); cudaFree(d_B); cudaFree(d_C);
    free(h_A_f32); free(h_B_f32); free(h_C);
    free(h_A_sparse); free(h_B_col);
    return 0;
}
