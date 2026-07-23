// Host driver for tts_duration_predictor_ln_bf16. Seeds:
//   X      = seeded((B, S, H), seed,     [-0.5, 0.5))
//   W_conv = seeded((H, H, 3), seed + 1, [-0.3, 0.3))     conv1d kernel along seq
//   b1     = seeded((H,),      seed + 2, [-0.5, 0.5))
//   gamma  = seeded((H,),      seed + 3, [-0.5, 0.5))
//   beta   = seeded((H,),      seed + 4, [-0.5, 0.5))
//   W2     = seeded((H,),      seed + 5, [-0.5, 0.5))
//   b2     = seeded((1,),      seed + 6, [-0.5, 0.5))
// Matches references/tts.py::_duration_predictor (plan-row 67 contract:
// "duration predictor conv + LN").

#define DURATION_CONV_K 3

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void tts_duration_predictor_ln_bf16_launch(
    const __nv_bfloat16 *d_X,
    const __nv_bfloat16 *d_W_conv, const __nv_bfloat16 *d_b1,
    const __nv_bfloat16 *d_gamma, const __nv_bfloat16 *d_beta,
    const __nv_bfloat16 *d_W2, const __nv_bfloat16 *d_b2_vec,
    float *d_hidden1,
    __nv_bfloat16 *d_out,
    int B, int S, int H,
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
    fprintf(stderr, "tts_duration_predictor_ln_bf16_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int batch = 2, seq = 4, hidden = 8;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--batch")       == 0 && i + 1 < argc) batch  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq")         == 0 && i + 1 < argc) seq    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden")      == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed   = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "tts_duration_predictor_ln_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t n_X    = (size_t)batch * seq * hidden;
    const size_t n_W_conv = (size_t)hidden * hidden * DURATION_CONV_K;
    const size_t n_H    = (size_t)hidden;
    const size_t n_out  = (size_t)batch * seq;
    const size_t n_h1   = (size_t)batch * seq * hidden;

    float *X_f  = (float *)malloc(n_X  * sizeof(float));
    float *W_conv_f = (float *)malloc(n_W_conv * sizeof(float));
    float *b1_f = (float *)malloc(n_H  * sizeof(float));
    float *gm_f = (float *)malloc(n_H  * sizeof(float));
    float *bt_f = (float *)malloc(n_H  * sizeof(float));
    float *W2_f = (float *)malloc(n_H  * sizeof(float));
    float b2_f[1];
    seed_tensor(X_f,  n_X,  seed,     -0.5, 0.5);
    seed_tensor(W_conv_f, n_W_conv, seed + 1, -0.3, 0.3);
    seed_tensor(b1_f, n_H,  seed + 2, -0.5, 0.5);
    seed_tensor(gm_f, n_H,  seed + 3, -0.5, 0.5);
    seed_tensor(bt_f, n_H,  seed + 4, -0.5, 0.5);
    seed_tensor(W2_f, n_H,  seed + 5, -0.5, 0.5);
    seed_tensor(b2_f, 1,    seed + 6, -0.5, 0.5);

    __nv_bfloat16 *h_X  = (__nv_bfloat16 *)malloc(n_X  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_W_conv = (__nv_bfloat16 *)malloc(n_W_conv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_b1 = (__nv_bfloat16 *)malloc(n_H  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_gm = (__nv_bfloat16 *)malloc(n_H  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_bt = (__nv_bfloat16 *)malloc(n_H  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_W2 = (__nv_bfloat16 *)malloc(n_H  * sizeof(__nv_bfloat16));
    __nv_bfloat16 h_b2[1] = { __float2bfloat16(b2_f[0]) };
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_X;  ++i) h_X[i]  = __float2bfloat16(X_f[i]);
    for (size_t i = 0; i < n_W_conv; ++i) h_W_conv[i] = __float2bfloat16(W_conv_f[i]);
    for (size_t i = 0; i < n_H;  ++i) h_b1[i] = __float2bfloat16(b1_f[i]);
    for (size_t i = 0; i < n_H;  ++i) h_gm[i] = __float2bfloat16(gm_f[i]);
    for (size_t i = 0; i < n_H;  ++i) h_bt[i] = __float2bfloat16(bt_f[i]);
    for (size_t i = 0; i < n_H;  ++i) h_W2[i] = __float2bfloat16(W2_f[i]);

    __nv_bfloat16 *d_X = nullptr, *d_W_conv = nullptr, *d_b1 = nullptr;
    __nv_bfloat16 *d_gm = nullptr, *d_bt = nullptr, *d_W2 = nullptr, *d_b2 = nullptr, *d_out = nullptr;
    float *d_h1 = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_X,  n_X  * sizeof(__nv_bfloat16)); if (rc) fail("mc X",  rc);
    rc = cudaMalloc(&d_W_conv, n_W_conv * sizeof(__nv_bfloat16)); if (rc) fail("mc W_conv", rc);
    rc = cudaMalloc(&d_b1, n_H  * sizeof(__nv_bfloat16)); if (rc) fail("mc b1", rc);
    rc = cudaMalloc(&d_gm, n_H  * sizeof(__nv_bfloat16)); if (rc) fail("mc gm", rc);
    rc = cudaMalloc(&d_bt, n_H  * sizeof(__nv_bfloat16)); if (rc) fail("mc bt", rc);
    rc = cudaMalloc(&d_W2, n_H  * sizeof(__nv_bfloat16)); if (rc) fail("mc W2", rc);
    rc = cudaMalloc(&d_b2, 1    * sizeof(__nv_bfloat16)); if (rc) fail("mc b2", rc);
    rc = cudaMalloc(&d_out, n_out * sizeof(__nv_bfloat16)); if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_h1,  n_h1  * sizeof(float));         if (rc) fail("mc h1",  rc);

    rc = cudaMemcpy(d_X,  h_X,  n_X  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D X",  rc);
    rc = cudaMemcpy(d_W_conv, h_W_conv, n_W_conv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D W_conv", rc);
    rc = cudaMemcpy(d_b1, h_b1, n_H  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D b1", rc);
    rc = cudaMemcpy(d_gm, h_gm, n_H  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D gm", rc);
    rc = cudaMemcpy(d_bt, h_bt, n_H  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D bt", rc);
    rc = cudaMemcpy(d_W2, h_W2, n_H  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D W2", rc);
    rc = cudaMemcpy(d_b2, h_b2, 1    * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D b2", rc);

    tts_duration_predictor_ln_bf16_launch(d_X, d_W_conv, d_b1, d_gm, d_bt, d_W2, d_b2,
                                           d_h1, d_out, batch, seq, hidden, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"tts_duration_predictor_ln_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int s = 0; s < seq; ++s) {
            if (s > 0) printf(",");
            printf("%.6g", __bfloat162float(h_out[b * seq + s]));
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_X); cudaFree(d_W_conv); cudaFree(d_b1); cudaFree(d_gm); cudaFree(d_bt);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_out); cudaFree(d_h1);
    free(X_f); free(W_conv_f); free(b1_f); free(gm_f); free(bt_f); free(W2_f);
    free(h_X); free(h_W_conv); free(h_b1); free(h_gm); free(h_bt); free(h_W2); free(h_out);
    return 0;
}
