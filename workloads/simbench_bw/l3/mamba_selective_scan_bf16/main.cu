// Host driver for mamba_selective_scan_bf16. Seeds x / a / b via the same
// PRNG as references/_common.py::deterministic_seeded_tensor, launches the
// kernel, and emits out[batch, seq, channels] as JSON.
//
// Matches references/ssm.py::build_reference for
// slug == "mamba_selective_scan_bf16":
//   x = seeded((batch, seq, channels), seed,         [-1, 1))
//   a = seeded((batch, seq, state_dim), seed + 1,    [0.9, 0.999))
//   b = seeded((batch, seq, state_dim), seed + 2,    [-0.1, 0.1))

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void mamba_selective_scan_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_a,
    const __nv_bfloat16 *d_b,
    __nv_bfloat16 *d_out,
    int batch, int seq, int channels, int state_dim, cudaStream_t stream);


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
    fprintf(stderr, "mamba_selective_scan_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 1;
    int seq = 128;
    int channels = 64;
    int state_dim = 16;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "mamba_selective_scan_bf16";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) channels = atoi(argv[++i]);
        else if (strcmp(argv[i], "--state-dim") == 0 && i + 1 < argc) state_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "mamba_selective_scan_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_x = (size_t)batch * seq * channels;
    const size_t n_ab = (size_t)batch * seq * state_dim;
    const size_t n_out = (size_t)batch * seq * channels;

    float *h_x_f32 = (float *)malloc(n_x  * sizeof(float));
    float *h_a_f32 = (float *)malloc(n_ab * sizeof(float));
    float *h_b_f32 = (float *)malloc(n_ab * sizeof(float));
    seed_tensor(h_x_f32, n_x,  seed,     -1.0, 1.0);
    seed_tensor(h_a_f32, n_ab, seed + 1, 0.9, 0.999);
    seed_tensor(h_b_f32, n_ab, seed + 2, -0.1, 0.1);

    __nv_bfloat16 *h_x = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_a = (__nv_bfloat16 *)malloc(n_ab * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_b = (__nv_bfloat16 *)malloc(n_ab * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x;  ++i) h_x[i] = __float2bfloat16(h_x_f32[i]);
    for (size_t i = 0; i < n_ab; ++i) h_a[i] = __float2bfloat16(h_a_f32[i]);
    for (size_t i = 0; i < n_ab; ++i) h_b[i] = __float2bfloat16(h_b_f32[i]);

    __nv_bfloat16 *d_x = nullptr, *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x,  n_x  * sizeof(__nv_bfloat16));  if (rc) fail("cudaMalloc d_x", rc);
    rc = cudaMalloc(&d_a,  n_ab * sizeof(__nv_bfloat16));  if (rc) fail("cudaMalloc d_a", rc);
    rc = cudaMalloc(&d_b,  n_ab * sizeof(__nv_bfloat16));  if (rc) fail("cudaMalloc d_b", rc);
    rc = cudaMalloc(&d_out, n_out * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_x, h_x, n_x  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x", rc);
    rc = cudaMemcpy(d_a, h_a, n_ab * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D a", rc);
    rc = cudaMemcpy(d_b, h_b, n_ab * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D b", rc);

    mamba_selective_scan_bf16_launch(d_x, d_a, d_b, d_out, batch, seq, channels, state_dim, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int ba = 0; ba < batch; ++ba) {
        if (ba > 0) printf(",");
        printf("[");
        for (int t = 0; t < seq; ++t) {
            if (t > 0) printf(",");
            printf("[");
            for (int c = 0; c < channels; ++c) {
                if (c > 0) printf(",");
                float v = __bfloat162float(h_out[((ba * seq) + t) * channels + c]);
                printf("%.6g", v);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_x_f32); free(h_a_f32); free(h_b_f32);
    free(h_x); free(h_a); free(h_b); free(h_out);
    return 0;
}
