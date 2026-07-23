// Host driver for whisper_encoder_block_bf16. Seeds:
//   x      = seeded((B, S, hidden),       seed,     [-0.5, 0.5))
//   W_qkv  = seeded((hidden, 3 * hidden), seed + 1, [-0.1, 0.1))
//   W_out  = seeded((hidden, hidden),     seed + 2, [-0.1, 0.1))
//   W_ff1  = seeded((hidden, mlp_dim),    seed + 3, [-0.1, 0.1))
//   W_ff2  = seeded((mlp_dim, hidden),    seed + 4, [-0.1, 0.1))
//   W_conv = seeded((hidden, hidden, 3),  seed + 5, [-0.1, 0.1))   audio conv front-end
// Matches references/audio.py::_whisper_encoder_block (pre-conv front-end + transformer).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WHISPER_CONV_K 3

extern "C" void whisper_encoder_block_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_W_qkv, const __nv_bfloat16 *d_W_out,
    const __nv_bfloat16 *d_W_ff1, const __nv_bfloat16 *d_W_ff2,
    const __nv_bfloat16 *d_W_conv,
    __nv_bfloat16 *d_x_conv,
    float *d_x_ln, float *d_qkv, float *d_attn, float *d_h1, float *d_h1_ln, float *d_ff_inner,
    __nv_bfloat16 *d_out,
    int B, int S, int heads, int head_dim, int M,
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
    fprintf(stderr, "whisper_encoder_block_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int batch = 1, seq = 3, heads = 2, head_dim = 4, mlp_dim = 8;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mlp-dim") == 0 && i + 1 < argc) mlp_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "whisper_encoder_block_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const int hidden = heads * head_dim;
    const size_t n_x     = (size_t)batch * seq * hidden;
    const size_t n_Wqkv  = (size_t)hidden * 3 * hidden;
    const size_t n_Wout  = (size_t)hidden * hidden;
    const size_t n_Wff1  = (size_t)hidden * mlp_dim;
    const size_t n_Wff2  = (size_t)mlp_dim * hidden;
    const size_t n_Wconv = (size_t)hidden * hidden * WHISPER_CONV_K;

    float *x_f     = (float *)malloc(n_x     * sizeof(float));
    float *Wqkv_f  = (float *)malloc(n_Wqkv  * sizeof(float));
    float *Wout_f  = (float *)malloc(n_Wout  * sizeof(float));
    float *Wff1_f  = (float *)malloc(n_Wff1  * sizeof(float));
    float *Wff2_f  = (float *)malloc(n_Wff2  * sizeof(float));
    float *Wconv_f = (float *)malloc(n_Wconv * sizeof(float));
    seed_tensor(x_f,     n_x,     seed,     -0.5, 0.5);
    seed_tensor(Wqkv_f,  n_Wqkv,  seed + 1, -0.1, 0.1);
    seed_tensor(Wout_f,  n_Wout,  seed + 2, -0.1, 0.1);
    seed_tensor(Wff1_f,  n_Wff1,  seed + 3, -0.1, 0.1);
    seed_tensor(Wff2_f,  n_Wff2,  seed + 4, -0.1, 0.1);
    seed_tensor(Wconv_f, n_Wconv, seed + 5, -0.1, 0.1);

    __nv_bfloat16 *h_x     = (__nv_bfloat16 *)malloc(n_x     * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wqkv  = (__nv_bfloat16 *)malloc(n_Wqkv  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wout  = (__nv_bfloat16 *)malloc(n_Wout  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wff1  = (__nv_bfloat16 *)malloc(n_Wff1  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wff2  = (__nv_bfloat16 *)malloc(n_Wff2  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wconv = (__nv_bfloat16 *)malloc(n_Wconv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out   = (__nv_bfloat16 *)malloc(n_x     * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x;     ++i) h_x[i]     = __float2bfloat16(x_f[i]);
    for (size_t i = 0; i < n_Wqkv;  ++i) h_Wqkv[i]  = __float2bfloat16(Wqkv_f[i]);
    for (size_t i = 0; i < n_Wout;  ++i) h_Wout[i]  = __float2bfloat16(Wout_f[i]);
    for (size_t i = 0; i < n_Wff1;  ++i) h_Wff1[i]  = __float2bfloat16(Wff1_f[i]);
    for (size_t i = 0; i < n_Wff2;  ++i) h_Wff2[i]  = __float2bfloat16(Wff2_f[i]);
    for (size_t i = 0; i < n_Wconv; ++i) h_Wconv[i] = __float2bfloat16(Wconv_f[i]);

    __nv_bfloat16 *d_x = nullptr, *d_Wqkv = nullptr, *d_Wout = nullptr, *d_Wff1 = nullptr, *d_Wff2 = nullptr;
    __nv_bfloat16 *d_Wconv = nullptr, *d_x_conv = nullptr, *d_out = nullptr;
    float *d_x_ln = nullptr, *d_qkv = nullptr, *d_attn = nullptr, *d_h1 = nullptr, *d_h1_ln = nullptr, *d_ff_inner = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x,     n_x     * sizeof(__nv_bfloat16)); if (rc) fail("mc x", rc);
    rc = cudaMalloc(&d_Wqkv,  n_Wqkv  * sizeof(__nv_bfloat16)); if (rc) fail("mc qkv", rc);
    rc = cudaMalloc(&d_Wout,  n_Wout  * sizeof(__nv_bfloat16)); if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_Wff1,  n_Wff1  * sizeof(__nv_bfloat16)); if (rc) fail("mc ff1", rc);
    rc = cudaMalloc(&d_Wff2,  n_Wff2  * sizeof(__nv_bfloat16)); if (rc) fail("mc ff2", rc);
    rc = cudaMalloc(&d_Wconv, n_Wconv * sizeof(__nv_bfloat16)); if (rc) fail("mc conv", rc);
    rc = cudaMalloc(&d_x_conv, n_x    * sizeof(__nv_bfloat16)); if (rc) fail("mc x_conv", rc);
    rc = cudaMalloc(&d_out,   n_x     * sizeof(__nv_bfloat16)); if (rc) fail("mc out_bf16", rc);
    rc = cudaMalloc(&d_x_ln,    n_x * sizeof(float));      if (rc) fail("mc x_ln", rc);
    rc = cudaMalloc(&d_qkv,     (size_t)batch * seq * 3 * hidden * sizeof(float)); if (rc) fail("mc qkv_scr", rc);
    rc = cudaMalloc(&d_attn,    n_x * sizeof(float));      if (rc) fail("mc attn", rc);
    rc = cudaMalloc(&d_h1,      n_x * sizeof(float));      if (rc) fail("mc h1", rc);
    rc = cudaMalloc(&d_h1_ln,   n_x * sizeof(float));      if (rc) fail("mc h1_ln", rc);
    rc = cudaMalloc(&d_ff_inner, (size_t)batch * seq * mlp_dim * sizeof(float)); if (rc) fail("mc ff_inner", rc);

    rc = cudaMemcpy(d_x,     h_x,     n_x     * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x", rc);
    rc = cudaMemcpy(d_Wqkv,  h_Wqkv,  n_Wqkv  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D qkv", rc);
    rc = cudaMemcpy(d_Wout,  h_Wout,  n_Wout  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D out", rc);
    rc = cudaMemcpy(d_Wff1,  h_Wff1,  n_Wff1  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D ff1", rc);
    rc = cudaMemcpy(d_Wff2,  h_Wff2,  n_Wff2  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D ff2", rc);
    rc = cudaMemcpy(d_Wconv, h_Wconv, n_Wconv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D conv", rc);

    whisper_encoder_block_bf16_launch(d_x, d_Wqkv, d_Wout, d_Wff1, d_Wff2,
                                       d_Wconv, d_x_conv,
                                       d_x_ln, d_qkv, d_attn, d_h1, d_h1_ln, d_ff_inner,
                                       d_out, batch, seq, heads, head_dim, mlp_dim, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"whisper_encoder_block_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int s = 0; s < seq; ++s) {
            if (s > 0) printf(",");
            printf("[");
            for (int d = 0; d < hidden; ++d) {
                if (d > 0) printf(",");
                printf("%.6g", __bfloat162float(h_out[(b * seq + s) * hidden + d]));
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_Wqkv); cudaFree(d_Wout); cudaFree(d_Wff1); cudaFree(d_Wff2);
    cudaFree(d_Wconv); cudaFree(d_x_conv); cudaFree(d_out);
    cudaFree(d_x_ln); cudaFree(d_qkv); cudaFree(d_attn); cudaFree(d_h1); cudaFree(d_h1_ln); cudaFree(d_ff_inner);
    free(x_f); free(Wqkv_f); free(Wout_f); free(Wff1_f); free(Wff2_f); free(Wconv_f);
    free(h_x); free(h_Wqkv); free(h_Wout); free(h_Wff1); free(h_Wff2); free(h_Wconv); free(h_out);
    return 0;
}
