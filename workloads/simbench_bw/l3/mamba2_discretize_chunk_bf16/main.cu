// Host driver for mamba2_discretize_chunk_bf16. Seeds:
//   x      = seeded((B, seq, channels),    seed,     [-1.0, 1.0))
//   a      = seeded((B, seq, state_dim),   seed + 1, [ 0.9,  0.999))
//   b      = seeded((B, seq, state_dim),   seed + 2, [-0.1,  0.1))
//   dt     = seeded((B, seq, state_dim),   seed + 3, [ 0.01, 0.1))   timestep
//   c_proj = seeded((state_dim, channels), seed + 4, [-0.2, 0.2))    output projection
// Matches references/ssm.py::_mamba2_discretize_chunk.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void mamba2_discretize_chunk_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_a, const __nv_bfloat16 *d_b,
    const __nv_bfloat16 *d_dt, const __nv_bfloat16 *d_c_proj,
    __nv_bfloat16 *d_out,
    int B, int seq, int state_dim, int channels,
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
    fprintf(stderr, "mamba2_discretize_chunk_bf16_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int batch = 1, seq = 16, state_dim = 8, channels = 16;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--batch")       == 0 && i + 1 < argc) batch     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq")         == 0 && i + 1 < argc) seq       = atoi(argv[++i]);
        else if (strcmp(argv[i], "--state-dim")   == 0 && i + 1 < argc) state_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--channels")    == 0 && i + 1 < argc) channels  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed      = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "mamba2_discretize_chunk_bf16_run: unknown arg %s\n", argv[i]); return 2; }
    }

    const size_t n_x  = (size_t)batch * seq * channels;
    const size_t n_s  = (size_t)batch * seq * state_dim;
    const size_t n_cp = (size_t)state_dim * channels;

    float *x_f  = (float *)malloc(n_x  * sizeof(float));
    float *a_f  = (float *)malloc(n_s  * sizeof(float));
    float *b_f  = (float *)malloc(n_s  * sizeof(float));
    float *dt_f = (float *)malloc(n_s  * sizeof(float));
    float *cp_f = (float *)malloc(n_cp * sizeof(float));
    seed_tensor(x_f,  n_x,  seed,     -1.0,  1.0);
    seed_tensor(a_f,  n_s,  seed + 1,  0.9,  0.999);
    seed_tensor(b_f,  n_s,  seed + 2, -0.1,  0.1);
    seed_tensor(dt_f, n_s,  seed + 3,  0.01, 0.1);
    seed_tensor(cp_f, n_cp, seed + 4, -0.2,  0.2);

    __nv_bfloat16 *h_x   = (__nv_bfloat16 *)malloc(n_x  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_a   = (__nv_bfloat16 *)malloc(n_s  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_b   = (__nv_bfloat16 *)malloc(n_s  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_dt  = (__nv_bfloat16 *)malloc(n_s  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_cp  = (__nv_bfloat16 *)malloc(n_cp * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_x  * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x;  ++i) h_x[i]  = __float2bfloat16(x_f[i]);
    for (size_t i = 0; i < n_s;  ++i) h_a[i]  = __float2bfloat16(a_f[i]);
    for (size_t i = 0; i < n_s;  ++i) h_b[i]  = __float2bfloat16(b_f[i]);
    for (size_t i = 0; i < n_s;  ++i) h_dt[i] = __float2bfloat16(dt_f[i]);
    for (size_t i = 0; i < n_cp; ++i) h_cp[i] = __float2bfloat16(cp_f[i]);

    __nv_bfloat16 *d_x = nullptr, *d_a = nullptr, *d_b = nullptr;
    __nv_bfloat16 *d_dt = nullptr, *d_cp = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x,   n_x  * sizeof(__nv_bfloat16)); if (rc) fail("mc x",   rc);
    rc = cudaMalloc(&d_a,   n_s  * sizeof(__nv_bfloat16)); if (rc) fail("mc a",   rc);
    rc = cudaMalloc(&d_b,   n_s  * sizeof(__nv_bfloat16)); if (rc) fail("mc b",   rc);
    rc = cudaMalloc(&d_dt,  n_s  * sizeof(__nv_bfloat16)); if (rc) fail("mc dt",  rc);
    rc = cudaMalloc(&d_cp,  n_cp * sizeof(__nv_bfloat16)); if (rc) fail("mc cp",  rc);
    rc = cudaMalloc(&d_out, n_x  * sizeof(__nv_bfloat16)); if (rc) fail("mc out", rc);

    rc = cudaMemcpy(d_x,  h_x,  n_x  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x",  rc);
    rc = cudaMemcpy(d_a,  h_a,  n_s  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D a",  rc);
    rc = cudaMemcpy(d_b,  h_b,  n_s  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D b",  rc);
    rc = cudaMemcpy(d_dt, h_dt, n_s  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D dt", rc);
    rc = cudaMemcpy(d_cp, h_cp, n_cp * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D cp", rc);

    mamba2_discretize_chunk_bf16_launch(d_x, d_a, d_b, d_dt, d_cp, d_out,
                                         batch, seq, state_dim, channels, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"mamba2_discretize_chunk_bf16\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int b = 0; b < batch; ++b) {
        if (b > 0) printf(",");
        printf("[");
        for (int t = 0; t < seq; ++t) {
            if (t > 0) printf(",");
            printf("[");
            for (int c = 0; c < channels; ++c) {
                if (c > 0) printf(",");
                printf("%.6g", __bfloat162float(h_out[(b * seq + t) * channels + c]));
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_a); cudaFree(d_b); cudaFree(d_dt); cudaFree(d_cp); cudaFree(d_out);
    free(x_f); free(a_f); free(b_f); free(dt_f); free(cp_f);
    free(h_x); free(h_a); free(h_b); free(h_dt); free(h_cp); free(h_out);
    return 0;
}
