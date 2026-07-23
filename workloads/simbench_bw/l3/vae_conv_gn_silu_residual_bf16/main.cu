// Host driver for vae_conv_gn_silu_residual_bf16.
//
// Seeds X, W, b, gamma, beta via the same MWC PRNG as
// references/_common.py::deterministic_seeded_tensor (low=-0.5,
// high=0.5), with seed offsets matching references/conv.py:
//   X     = seed        : (batch, channels, height, width)
//   W     = seed + 1    : (channels, channels, 3, 3)
//   b     = seed + 2    : (channels,)
//   gamma = seed + 3    : (channels,)
//   beta  = seed + 4    : (channels,)
// Casts to BF16 on the host, uploads, invokes the 3-stage kernel, and
// emits the final BF16 output Y as a nested [batch][channels][height]
// [width] FP32 list — same shape as the Python reference produces.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void vae_conv_gn_silu_residual_bf16_launch(
    const __nv_bfloat16 *d_X,
    const __nv_bfloat16 *d_W,
    const __nv_bfloat16 *d_b,
    const __nv_bfloat16 *d_gamma,
    const __nv_bfloat16 *d_beta,
    float               *d_conv_out,
    float               *d_mean,
    float               *d_var,
    __nv_bfloat16       *d_Y,
    int N, int C, int H, int Width,
    cudaStream_t stream);


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
    fprintf(stderr, "vae_conv_gn_silu_residual_bf16_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int N = 1, C = 32, H = 16, Width = 16;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "vae_conv_gn_silu_residual_bf16";

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--batch")    == 0 && i + 1 < argc) N     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) C     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height")   == 0 && i + 1 < argc) H     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width")    == 0 && i + 1 < argc) Width = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")     == 0 && i + 1 < argc) seed  = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "vae_conv_gn_silu_residual_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    if (N <= 0 || C <= 0 || H <= 0 || Width <= 0) {
        fprintf(stderr, "vae_conv_gn_silu_residual_bf16_run: all dims must be positive (got N=%d C=%d H=%d W=%d)\n",
                N, C, H, Width);
        return 2;
    }

    // GroupNorm partitions channels into fixed 32-channel groups (VAE
    // convention; see references/conv.py). Channels-not-divisible-by-32
    // would floor-divide num_groups and leave the tail out-of-group,
    // producing wrong-by-tail-channels output. Refuse upfront and match
    // the Python reference's ValueError contract.
    if (C >= 32 && (C % 32) != 0) {
        fprintf(stderr,
                "vae_conv_gn_silu_residual_bf16_run: "
                "groupnorm_channels_not_divisible_by_32: channels=%d "
                "(must be a multiple of 32 when channels >= 32)\n",
                C);
        return 2;
    }

    const size_t nX = (size_t)N * C * H * Width;
    const size_t nW = (size_t)C * C * 3 * 3;
    const size_t nb = (size_t)C;
    const int group_size = (C >= 32) ? 32 : C;
    const int num_groups = (group_size > 0) ? (C / group_size) : 1;

    float *h_X_f32 = (float *)malloc(nX * sizeof(float));
    float *h_W_f32 = (float *)malloc(nW * sizeof(float));
    float *h_b_f32 = (float *)malloc(nb * sizeof(float));
    float *h_gamma_f32 = (float *)malloc(nb * sizeof(float));
    float *h_beta_f32  = (float *)malloc(nb * sizeof(float));

    seed_tensor(h_X_f32,     nX, seed,     -0.5, 0.5);
    seed_tensor(h_W_f32,     nW, seed + 1, -0.5, 0.5);
    seed_tensor(h_b_f32,     nb, seed + 2, -0.5, 0.5);
    seed_tensor(h_gamma_f32, nb, seed + 3, -0.5, 0.5);
    seed_tensor(h_beta_f32,  nb, seed + 4, -0.5, 0.5);

    __nv_bfloat16 *h_X     = (__nv_bfloat16 *)malloc(nX * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_W     = (__nv_bfloat16 *)malloc(nW * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_b     = (__nv_bfloat16 *)malloc(nb * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_gamma = (__nv_bfloat16 *)malloc(nb * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_beta  = (__nv_bfloat16 *)malloc(nb * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Y     = (__nv_bfloat16 *)malloc(nX * sizeof(__nv_bfloat16));

    for (size_t i = 0; i < nX; ++i) h_X[i]     = __float2bfloat16(h_X_f32[i]);
    for (size_t i = 0; i < nW; ++i) h_W[i]     = __float2bfloat16(h_W_f32[i]);
    for (size_t i = 0; i < nb; ++i) h_b[i]     = __float2bfloat16(h_b_f32[i]);
    for (size_t i = 0; i < nb; ++i) h_gamma[i] = __float2bfloat16(h_gamma_f32[i]);
    for (size_t i = 0; i < nb; ++i) h_beta[i]  = __float2bfloat16(h_beta_f32[i]);

    __nv_bfloat16 *d_X = nullptr, *d_W = nullptr, *d_b = nullptr;
    __nv_bfloat16 *d_gamma = nullptr, *d_beta = nullptr, *d_Y = nullptr;
    float *d_conv_out = nullptr, *d_mean = nullptr, *d_var = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_X,        nX * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc X", rc);
    rc = cudaMalloc(&d_W,        nW * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc W", rc);
    rc = cudaMalloc(&d_b,        nb * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc b", rc);
    rc = cudaMalloc(&d_gamma,    nb * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc gamma", rc);
    rc = cudaMalloc(&d_beta,     nb * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc beta", rc);
    rc = cudaMalloc(&d_Y,        nX * sizeof(__nv_bfloat16));   if (rc) fail("cudaMalloc Y", rc);
    rc = cudaMalloc(&d_conv_out, nX * sizeof(float));           if (rc) fail("cudaMalloc conv_out", rc);
    rc = cudaMalloc(&d_mean,     (size_t)N * num_groups * sizeof(float));  if (rc) fail("cudaMalloc mean", rc);
    rc = cudaMalloc(&d_var,      (size_t)N * num_groups * sizeof(float));  if (rc) fail("cudaMalloc var", rc);
    rc = cudaMemcpy(d_X,     h_X,     nX * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D X", rc);
    rc = cudaMemcpy(d_W,     h_W,     nW * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D W", rc);
    rc = cudaMemcpy(d_b,     h_b,     nb * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D b", rc);
    rc = cudaMemcpy(d_gamma, h_gamma, nb * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D gamma", rc);
    rc = cudaMemcpy(d_beta,  h_beta,  nb * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D beta", rc);

    vae_conv_gn_silu_residual_bf16_launch(d_X, d_W, d_b, d_gamma, d_beta,
                                          d_conv_out, d_mean, d_var, d_Y,
                                          N, C, H, Width, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_Y, d_Y, nX * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    if (rc) fail("D2H Y", rc);

    // Emit [[[[ ... ]]]] matching conv.py::build_reference's FP32 output shape.
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int n = 0; n < N; ++n) {
        if (n > 0) printf(",");
        printf("[");
        for (int c = 0; c < C; ++c) {
            if (c > 0) printf(",");
            printf("[");
            for (int y = 0; y < H; ++y) {
                if (y > 0) printf(",");
                printf("[");
                for (int x = 0; x < Width; ++x) {
                    if (x > 0) printf(",");
                    const float v = __bfloat162float(h_Y[((n * C + c) * H + y) * Width + x]);
                    printf("%.6g", v);
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_X); cudaFree(d_W); cudaFree(d_b); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_conv_out); cudaFree(d_mean); cudaFree(d_var); cudaFree(d_Y);
    free(h_X_f32); free(h_W_f32); free(h_b_f32); free(h_gamma_f32); free(h_beta_f32);
    free(h_X); free(h_W); free(h_b); free(h_gamma); free(h_beta); free(h_Y);
    return 0;
}
