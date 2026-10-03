// Host driver for dram_streaming_dot_product. See
// dram_streaming_dot_product.cu for the kernel-side documentation
// of the L2-resident two-buffer reduction pattern.

#include <cuda_runtime.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
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

static uint32_t parse_u32(const char *name, const char *text,
                          uint32_t max_value, bool zero_ok)
{
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno != 0 ||
        end == text || *end != '\0' || value > max_value ||
        (!zero_ok && value == 0)) {
        fprintf(stderr,
                "dram_streaming_dot_product_run: out_of_domain: invalid %s: %s\n",
                name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static bool valid_token(const char *text)
{
    if (!text || !*text) return false;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (!(isalnum(*p) || *p == '_' || *p == '-')) return false;
    }
    return true;
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
        if      (strcmp(argv[i], "--n")           == 0 && i + 1 < argc) n = (int)parse_u32("n", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--passes")      == 0 && i + 1 < argc) passes = (int)parse_u32("passes", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--blocks")      == 0 && i + 1 < argc) blocks = (int)parse_u32("blocks", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--threads")     == 0 && i + 1 < argc) threads = (int)parse_u32("threads", argv[++i], 1024, false);
        else if (strcmp(argv[i], "--warmup")      == 0 && i + 1 < argc) warmup = (int)parse_u32("warmup", argv[++i], INT_MAX, true);
        else if (strcmp(argv[i], "--reps")        == 0 && i + 1 < argc) reps = (int)parse_u32("reps", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--seed")        == 0 && i + 1 < argc) seed = parse_u32("seed", argv[++i], UINT32_MAX, true);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else {
            fprintf(stderr,
                    "dram_streaming_dot_product_run: unknown arg %s\n",
                    argv[i]);
            return 2;
        }
    }
    if ((threads & (threads - 1)) != 0) {
        fprintf(stderr,
                "dram_streaming_dot_product_run: out_of_domain: --threads must be "
                "a power of two (for the block reduction)\n");
        return 2;
    }
    const uint64_t element_count = (uint64_t)n;
    const uint64_t stride = (uint64_t)blocks * (uint64_t)threads;
    if (element_count + stride - 1 > INT_MAX ||
        element_count > SIZE_MAX / sizeof(float) || !valid_token(shape_point)) {
        fprintf(stderr, "dram_streaming_dot_product_run: out_of_domain: invalid label or shape overflows index/allocation\n");
        return 2;
    }
    // A product plus sum can round twice without FMA. Include the block's
    // pairwise reduction depth. Reject a shape whose roundoff bound is vacuous.
    unsigned depth = 0;
    for (int t = threads; t > 1; t /= 2) ++depth;
    const double steps = 2.0 * ((element_count + stride - 1) / stride) * passes + depth;
    const double ru = steps * (FLT_EPSILON / 2.0);
    if (ru >= 1.0) {
        fprintf(stderr, "dram_streaming_dot_product_run: out_of_domain: accumulation too long for meaningful FP32 validation\n");
        return 2;
    }

    const size_t buf_n = (size_t)n;
    const size_t buf_bytes = buf_n * sizeof(float);
    const size_t partial_bytes = (size_t)blocks * sizeof(float);
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t memrc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (memrc) fail("memory query", memrc);
    if (buf_bytes > free_bytes || buf_bytes > free_bytes - buf_bytes ||
        partial_bytes > free_bytes - 2 * buf_bytes) {
        fprintf(stderr,
                "dram_streaming_dot_product_run: out of memory: requested shape exceeds free device memory\n");
        return 1;
    }
    float *h_a = (float *)malloc(buf_n * sizeof(float));
    float *h_b = (float *)malloc(buf_n * sizeof(float));
    if (!h_a || !h_b) {
        fprintf(stderr, "dram_streaming_dot_product_run: out of memory: host allocation\n");
        free(h_a); free(h_b);
        return 1;
    }
    seed_tensor(h_a, buf_n, seed,     -1.0, 1.0);
    seed_tensor(h_b, buf_n, seed + 1, -1.0, 1.0);
    double *expected = (double *)calloc((size_t)blocks, sizeof(double));
    double *absolute_sum = (double *)calloc((size_t)blocks, sizeof(double));
    if (!expected || !absolute_sum) {
        fprintf(stderr, "dram_streaming_dot_product_run: out of memory: reference allocation\n");
        free(h_a); free(h_b); free(expected); free(absolute_sum);
        return 1;
    }
    for (size_t i = 0; i < buf_n; ++i) {
        const size_t block = (i % stride) / (size_t)threads;
        const double product = (double)h_a[i] * (double)h_b[i];
        expected[block] += product;
        absolute_sum[block] += fabs(product);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_partial = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_a, buf_bytes);
    if (rc) fail("mc a", rc);
    rc = cudaMalloc(&d_b, buf_bytes);
    if (rc) fail("mc b", rc);
    rc = cudaMalloc(&d_partial, partial_bytes);
    if (rc) fail("mc partial", rc);
    rc = cudaMemset(d_partial, 0xff, partial_bytes);
    if (rc) fail("initialize partials", rc);
    rc = cudaMemcpy(d_a, h_a, buf_bytes,
                    cudaMemcpyHostToDevice);
    if (rc) fail("H2D a", rc);
    rc = cudaMemcpy(d_b, h_b, buf_bytes,
                    cudaMemcpyHostToDevice);
    if (rc) fail("H2D b", rc);

    for (int w = 0; w < warmup; ++w) {
        dram_streaming_dot_product_launch(d_a, d_b, d_partial, n,
                                          passes, blocks, threads, 0);
        rc = cudaGetLastError(); if (rc) fail("warmup launch", rc);
    }
    if (warmup > 0) {
        rc = cudaDeviceSynchronize();
        if (rc) fail("sync warmup", rc);
    }

    for (int r = 0; r < reps; ++r) {
        dram_streaming_dot_product_launch(d_a, d_b, d_partial, n, passes,
                                          blocks, threads, 0);
        rc = cudaGetLastError(); if (rc) fail("launch", rc);
    }
    rc = cudaDeviceSynchronize();
    if (rc) fail("sync", rc);

    float *h_partial = (float *)malloc((size_t)blocks * sizeof(float));
    if (!h_partial) {
        fprintf(stderr, "dram_streaming_dot_product_run: out of memory: partial host allocation\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_partial);
        free(h_a); free(h_b); free(expected); free(absolute_sum);
        return 1;
    }
    rc = cudaMemcpy(h_partial, d_partial,
                    (size_t)blocks * sizeof(float),
                    cudaMemcpyDeviceToHost);
    if (rc) fail("D2H partial", rc);
    double total = 0.0, reference = 0.0, max_block_error = 0.0;
    size_t partial_mismatches = 0;
    // Account also for double reference summation and multiplication by passes.
    const double du = (buf_n + 1.0) * (DBL_EPSILON / 2.0);
    const double gamma = ru / (1.0 - ru) + du / (1.0 - du);
    for (int i = 0; i < blocks; ++i) {
        expected[i] *= passes;
        const double block_error = fabs((double)h_partial[i] - expected[i]);
        const double bound = gamma * absolute_sum[i] * passes;
        if (!isfinite(h_partial[i]) || block_error > bound) ++partial_mismatches;
        max_block_error = fmax(max_block_error, block_error);
        total += (double)h_partial[i];
        reference += expected[i];
    }
    const double error = fabs(total - reference);
    const bool correct = isfinite(total) && isfinite(reference) && partial_mismatches == 0;

    printf("{\n  \"schema_version\": 1,\n");
    printf("  \"slug\": \"dram_streaming_dot_product\",\n");
    const char *shape_set = !strncmp(shape_point, "timing_", 7) ? "timing_shapes" :
                            !strncmp(shape_point, "trace_", 6) ? "trace_shapes" : "correctness_shapes";
    printf("  \"shape_set\": \"%s\",\n", shape_set);
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"n\": %d,\n", n);
    printf("  \"passes\": %d,\n", passes);
    printf("  \"blocks\": %d,\n", blocks);
    printf("  \"threads\": %d,\n", threads);
    printf("  \"warmup\": %d, \"reps\": %d, \"seed\": %u,\n", warmup, reps, seed);
    printf("  \"partial_mismatches\": %zu, \"max_block_abs_error\": %.9g,\n", partial_mismatches, max_block_error);
    printf("  \"correct\": %s, \"reference\": %.9g, \"abs_error\": %.9g,\n", correct ? "true" : "false", reference, error);
    // Emit the scalar dot product under the canonical `output` field the
    // correctness harness reads (handle_correctness -> kernel_result["output"]);
    // the mma_microbench reference compares against the analytic
    // passes * sum_i a[i]*b[i] scalar for this row.
    printf("  \"output\": %.9g\n}\n", total);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_partial);
    free(h_a); free(h_b); free(h_partial); free(expected); free(absolute_sum);
    return correct ? 0 : 1;
}
