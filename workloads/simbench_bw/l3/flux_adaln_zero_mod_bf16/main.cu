// Host driver for flux_adaln_zero_mod_bf16. Seeds:
//   x       = seeded((B, S, hidden), seed,     [-0.5, 0.5))
//   cond    = seeded((B, Cd),        seed + 1, [-0.5, 0.5))
//   W_scale = seeded((Cd, hidden),   seed + 2, [-0.1, 0.1))
//   W_shift = seeded((Cd, hidden),   seed + 3, [-0.1, 0.1))
//   W_gate  = seeded((Cd, hidden),   seed + 4, [-0.1, 0.1))
// Matches references/diffusion.py::_adaln_zero.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void flux_adaln_zero_mod_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_cond,
    const __nv_bfloat16 *d_W_scale, const __nv_bfloat16 *d_W_shift, const __nv_bfloat16 *d_W_gate,
    float *d_scale, float *d_shift, float *d_gate,
    __nv_bfloat16 *d_out,
    int B, int S, int hidden, int Cd,
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
    fprintf(stderr, "flux_adaln_zero_mod_bf16_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int batch = 1, seq = 3, hidden = 8, cond_dim = 4;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--batch")       == 0 && i + 1 < argc) batch    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq")         == 0 && i + 1 < argc) seq      = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden")      == 0 && i + 1 < argc) hidden   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cond-dim")    == 0 && i + 1 < argc) cond_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed     = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "flux_adaln_zero_mod_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t n_x   = (size_t)batch * seq * hidden;
    const size_t n_c   = (size_t)batch * cond_dim;
    const size_t n_Wsc = (size_t)cond_dim * hidden;
    const size_t n_bh  = (size_t)batch * hidden;

    float *x_f   = (float *)malloc(n_x   * sizeof(float));
    float *c_f   = (float *)malloc(n_c   * sizeof(float));
    float *Wsc_f = (float *)malloc(n_Wsc * sizeof(float));
    float *Wsh_f = (float *)malloc(n_Wsc * sizeof(float));
    float *Wg_f  = (float *)malloc(n_Wsc * sizeof(float));
    seed_tensor(x_f,   n_x,   seed,     -0.5,  0.5);
    seed_tensor(c_f,   n_c,   seed + 1, -0.5,  0.5);
    seed_tensor(Wsc_f, n_Wsc, seed + 2, -0.1,  0.1);
    seed_tensor(Wsh_f, n_Wsc, seed + 3, -0.1,  0.1);
    seed_tensor(Wg_f,  n_Wsc, seed + 4, -0.1,  0.1);

    __nv_bfloat16 *h_x   = (__nv_bfloat16 *)malloc(n_x   * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_c   = (__nv_bfloat16 *)malloc(n_c   * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wsc = (__nv_bfloat16 *)malloc(n_Wsc * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wsh = (__nv_bfloat16 *)malloc(n_Wsc * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Wg  = (__nv_bfloat16 *)malloc(n_Wsc * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x;   ++i) h_x[i]   = __float2bfloat16(x_f[i]);
    for (size_t i = 0; i < n_c;   ++i) h_c[i]   = __float2bfloat16(c_f[i]);
    for (size_t i = 0; i < n_Wsc; ++i) h_Wsc[i] = __float2bfloat16(Wsc_f[i]);
    for (size_t i = 0; i < n_Wsc; ++i) h_Wsh[i] = __float2bfloat16(Wsh_f[i]);
    for (size_t i = 0; i < n_Wsc; ++i) h_Wg[i]  = __float2bfloat16(Wg_f[i]);

    __nv_bfloat16 *d_x = nullptr, *d_c = nullptr, *d_Wsc = nullptr, *d_Wsh = nullptr, *d_Wg = nullptr, *d_out = nullptr;
    float *d_scale = nullptr, *d_shift = nullptr, *d_gate = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x,   n_x   * sizeof(__nv_bfloat16)); if (rc) fail("mc x",   rc);
    rc = cudaMalloc(&d_c,   n_c   * sizeof(__nv_bfloat16)); if (rc) fail("mc c",   rc);
    rc = cudaMalloc(&d_Wsc, n_Wsc * sizeof(__nv_bfloat16)); if (rc) fail("mc Wsc", rc);
    rc = cudaMalloc(&d_Wsh, n_Wsc * sizeof(__nv_bfloat16)); if (rc) fail("mc Wsh", rc);
    rc = cudaMalloc(&d_Wg,  n_Wsc * sizeof(__nv_bfloat16)); if (rc) fail("mc Wg",  rc);
    rc = cudaMalloc(&d_out, n_x   * sizeof(__nv_bfloat16)); if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_scale, n_bh * sizeof(float)); if (rc) fail("mc scale", rc);
    rc = cudaMalloc(&d_shift, n_bh * sizeof(float)); if (rc) fail("mc shift", rc);
    rc = cudaMalloc(&d_gate,  n_bh * sizeof(float)); if (rc) fail("mc gate",  rc);

    rc = cudaMemcpy(d_x,   h_x,   n_x   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x",   rc);
    rc = cudaMemcpy(d_c,   h_c,   n_c   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D c",   rc);
    rc = cudaMemcpy(d_Wsc, h_Wsc, n_Wsc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Wsc", rc);
    rc = cudaMemcpy(d_Wsh, h_Wsh, n_Wsc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Wsh", rc);
    rc = cudaMemcpy(d_Wg,  h_Wg,  n_Wsc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D Wg",  rc);

    flux_adaln_zero_mod_bf16_launch(d_x, d_c, d_Wsc, d_Wsh, d_Wg,
                                     d_scale, d_shift, d_gate, d_out,
                                     batch, seq, hidden, cond_dim, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"flux_adaln_zero_mod_bf16\",\n");
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

    cudaFree(d_x); cudaFree(d_c); cudaFree(d_Wsc); cudaFree(d_Wsh); cudaFree(d_Wg); cudaFree(d_out);
    cudaFree(d_scale); cudaFree(d_shift); cudaFree(d_gate);
    free(x_f); free(c_f); free(Wsc_f); free(Wsh_f); free(Wg_f);
    free(h_x); free(h_c); free(h_Wsc); free(h_Wsh); free(h_Wg); free(h_out);
    return 0;
}
