// Host driver for tts_vocoder_upsample_silu_bf16. Seeds:
//   X          = seeded((B, C, L),  seed,     [-0.5, 0.5))
//   W_upsample = seeded((C, UF),    seed + 1, [-1.0, 1.0))    learned upsample
// Matches references/tts.py::_vocoder_upsample_silu (plan-row 68
// contract: "transposed / upsample conv + SiLU").

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void tts_vocoder_upsample_silu_bf16_launch(
    const __nv_bfloat16 *d_X,
    const __nv_bfloat16 *d_W_upsample,
    __nv_bfloat16 *d_out,
    int B, int C, int L, int UF,
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
    fprintf(stderr, "tts_vocoder_upsample_silu_bf16_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int batch = 2, channels = 4, length = 8, upsample_factor = 2;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--batch")           == 0 && i + 1 < argc) batch    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--channels")        == 0 && i + 1 < argc) channels = atoi(argv[++i]);
        else if (strcmp(argv[i], "--length")          == 0 && i + 1 < argc) length   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--upsample-factor") == 0 && i + 1 < argc) upsample_factor = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")            == 0 && i + 1 < argc) seed     = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point")     == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "tts_vocoder_upsample_silu_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t n_in  = (size_t)batch * channels * length;
    const size_t n_out = n_in * upsample_factor;
    const size_t n_W   = (size_t)channels * upsample_factor;

    float *X_f = (float *)malloc(n_in * sizeof(float));
    float *W_f = (float *)malloc(n_W  * sizeof(float));
    seed_tensor(X_f, n_in, seed,     -0.5, 0.5);
    seed_tensor(W_f, n_W,  seed + 1, -1.0, 1.0);

    __nv_bfloat16 *h_X   = (__nv_bfloat16 *)malloc(n_in  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_W   = (__nv_bfloat16 *)malloc(n_W   * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_in; ++i) h_X[i] = __float2bfloat16(X_f[i]);
    for (size_t i = 0; i < n_W;  ++i) h_W[i] = __float2bfloat16(W_f[i]);

    __nv_bfloat16 *d_X = nullptr, *d_W = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_X,   n_in  * sizeof(__nv_bfloat16)); if (rc) fail("mc X",   rc);
    rc = cudaMalloc(&d_W,   n_W   * sizeof(__nv_bfloat16)); if (rc) fail("mc W",   rc);
    rc = cudaMalloc(&d_out, n_out * sizeof(__nv_bfloat16)); if (rc) fail("mc out", rc);
    rc = cudaMemcpy(d_X, h_X, n_in * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D X", rc);
    rc = cudaMemcpy(d_W, h_W, n_W  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D W", rc);

    tts_vocoder_upsample_silu_bf16_launch(d_X, d_W, d_out, batch, channels, length, upsample_factor, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    const int Lout = length * upsample_factor;
    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"tts_vocoder_upsample_silu_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int c = 0; c < channels; ++c) {
            if (c > 0) printf(",");
            printf("[");
            for (int i = 0; i < Lout; ++i) {
                if (i > 0) printf(",");
                size_t idx = ((size_t)b * channels + c) * Lout + i;
                printf("%.6g", __bfloat162float(h_out[idx]));
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_X); cudaFree(d_W); cudaFree(d_out);
    free(X_f); free(W_f); free(h_X); free(h_W); free(h_out);
    return 0;
}
