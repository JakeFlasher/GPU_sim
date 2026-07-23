// Host driver for hyena_fft_conv_block_fp32. Seeds x + filt via the same
// PRNG as references/_common.py::deterministic_seeded_tensor so the GPU
// output can be compared against the Python reference under the FP32 SSM
// tolerance.
//
// Matches references/ssm.py::build_reference for slug == "hyena_fft_conv_block_fp32":
//   x    = seeded((batch, seq, channels),  seed,      [-1, 1))
//   filt = seeded((channels, kernel),      seed + 1,  [-0.1, 0.1))
//   expected_output = causal _direct_conv(x, filt)

#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void hyena_fft_conv_block_fp32_launch(
    const float *d_x,
    const float *d_filt,
    float *d_out,
    int batch, int seq, int channels, int kernel_size, cudaStream_t stream);


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
    fprintf(stderr, "hyena_fft_conv_block_fp32_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int batch = 1;
    int seq = 128;
    int channels = 64;
    int kernel_size = 8;  // matches ssm.py default when shape axes omit `kernel`
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "hyena_fft_conv_block_fp32";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) channels = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) kernel_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr, "hyena_fft_conv_block_fp32_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    const size_t n_x = (size_t)batch * seq * channels;
    const size_t n_f = (size_t)channels * kernel_size;
    const size_t n_out = n_x;

    float *h_x = (float *)malloc(n_x * sizeof(float));
    float *h_f = (float *)malloc(n_f * sizeof(float));
    float *h_out = (float *)malloc(n_out * sizeof(float));
    seed_tensor(h_x, n_x, seed, -1.0, 1.0);
    seed_tensor(h_f, n_f, seed + 1, -0.1, 0.1);

    float *d_x = nullptr, *d_f = nullptr, *d_out = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_x, n_x * sizeof(float));   if (rc) fail("cudaMalloc d_x", rc);
    rc = cudaMalloc(&d_f, n_f * sizeof(float));   if (rc) fail("cudaMalloc d_f", rc);
    rc = cudaMalloc(&d_out, n_out * sizeof(float)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemcpy(d_x, h_x, n_x * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D x", rc);
    rc = cudaMemcpy(d_f, h_f, n_f * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D f", rc);

    hyena_fft_conv_block_fp32_launch(d_x, d_f, d_out, batch, seq, channels, kernel_size, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

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
                printf("%.9g", h_out[((ba * seq) + t) * channels + c]);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_x); cudaFree(d_f); cudaFree(d_out);
    free(h_x); free(h_f); free(h_out);
    return 0;
}
