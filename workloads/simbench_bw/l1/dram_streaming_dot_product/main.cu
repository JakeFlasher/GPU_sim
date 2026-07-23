// Host driver for dram_streaming_dot_product. See
// dram_streaming_dot_product.cu for the kernel-side documentation
// of the L2-resident two-buffer reduction pattern.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void dram_streaming_dot_product_launch(
    const float *d_a, const float *d_b, float *d_partial_sums,
    int n, int passes, int blocks, int threads, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) {
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}

static void seed_tensor(float *flat, size_t n, uint32_t seed,
                        double lo, double hi)
{
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}

static void fail(const char *msg, int rc) {
    fprintf(stderr,
            "dram_streaming_dot_product_run: %s (rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

int main(int argc, char **argv) {
    int n = 131072;
    int passes = 4;
    int blocks = 128;
    int threads = 256;
    int warmup = 4;
    int reps = 1;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--n")           == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--passes")      == 0 && i + 1 < argc) passes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--blocks")      == 0 && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads")     == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup")      == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps")        == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr,
                    "dram_streaming_dot_product_run: unknown arg %s\n",
                    argv[i]);
            return 2;
        }
    }
    if (n < 1 || passes < 1 || blocks < 1 || threads < 1
        || warmup < 0 || reps < 1) {
        fprintf(stderr,
                "dram_streaming_dot_product_run: --n / --passes / "
                "--blocks / --threads / --reps must all be >= 1; "
                "--warmup must be >= 0\n");
        return 2;
    }
    if ((threads & (threads - 1)) != 0) {
        fprintf(stderr,
                "dram_streaming_dot_product_run: --threads must be "
                "a power of two (for the block reduction)\n");
        return 2;
    }

    const size_t buf_n = (size_t)n;
    float *h_a = (float *)malloc(buf_n * sizeof(float));
    float *h_b = (float *)malloc(buf_n * sizeof(float));
    seed_tensor(h_a, buf_n, seed,     -1.0, 1.0);
    seed_tensor(h_b, buf_n, seed + 1, -1.0, 1.0);

    float *d_a = nullptr, *d_b = nullptr, *d_partial = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_a, buf_n * sizeof(float));
    if (rc) fail("mc a", rc);
    rc = cudaMalloc(&d_b, buf_n * sizeof(float));
    if (rc) fail("mc b", rc);
    rc = cudaMalloc(&d_partial, (size_t)blocks * sizeof(float));
    if (rc) fail("mc partial", rc);
    rc = cudaMemcpy(d_a, h_a, buf_n * sizeof(float),
                    cudaMemcpyHostToDevice);
    if (rc) fail("H2D a", rc);
    rc = cudaMemcpy(d_b, h_b, buf_n * sizeof(float),
                    cudaMemcpyHostToDevice);
    if (rc) fail("H2D b", rc);

    for (int w = 0; w < warmup; ++w) {
        dram_streaming_dot_product_launch(d_a, d_b, d_partial, n,
                                          passes, blocks, threads, 0);
    }
    if (warmup > 0) {
        rc = cudaDeviceSynchronize();
        if (rc) fail("sync warmup", rc);
    }

    for (int r = 0; r < reps; ++r) {
        dram_streaming_dot_product_launch(d_a, d_b, d_partial, n, passes,
                                          blocks, threads, 0);
    }
    rc = cudaDeviceSynchronize();
    if (rc) fail("sync", rc);

    float *h_partial = (float *)malloc((size_t)blocks * sizeof(float));
    rc = cudaMemcpy(h_partial, d_partial,
                    (size_t)blocks * sizeof(float),
                    cudaMemcpyDeviceToHost);
    if (rc) fail("D2H partial", rc);
    double total = 0.0;
    for (int i = 0; i < blocks; ++i) total += (double)h_partial[i];

    printf("{\n  \"schema_version\": 1,\n");
    printf("  \"slug\": \"dram_streaming_dot_product\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"n\": %d,\n", n);
    printf("  \"passes\": %d,\n", passes);
    printf("  \"blocks\": %d,\n", blocks);
    printf("  \"threads\": %d,\n", threads);
    // Emit the scalar dot product under the canonical `output` field the
    // correctness harness reads (handle_correctness -> kernel_result["output"]);
    // the mma_microbench reference compares against the analytic
    // passes * sum_i a[i]*b[i] scalar for this row.
    printf("  \"output\": %.9g\n}\n", total);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_partial);
    free(h_a); free(h_b); free(h_partial);
    return 0;
}
