// Host driver for shared_bank_conflict_stride.
// The output is an identity copy; the separate sink validates the strided read.

#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" cudaError_t shared_bank_conflict_stride_launch(
    const float *d_in, float *d_out, float *d_sink,
    int rows, int cols, int stride, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "shared_bank_conflict_stride_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}
static uint32_t parse_u32(const char *name, const char *text, uint32_t maximum, bool zero_ok = false) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || !end || *end || (!zero_ok && value == 0) || value > maximum) {
        fprintf(stderr, "shared_bank_conflict_stride_run: out_of_domain: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

int main(int argc, char **argv) {
    int rows = 8, cols = 8, stride = 2;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    bool summary_only = false;
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--rows")   == 0 && i + 1 < argc) rows   = (int)parse_u32("rows", argv[++i], INT_MAX);
        else if (strcmp(argv[i], "--cols")   == 0 && i + 1 < argc) cols   = (int)parse_u32("cols", argv[++i], INT_MAX);
        else if (strcmp(argv[i], "--stride") == 0 && i + 1 < argc) stride = (int)parse_u32("stride", argv[++i], INT_MAX);
        else if (strcmp(argv[i], "--seed")   == 0 && i + 1 < argc) seed   = parse_u32("seed", argv[++i], UINT32_MAX, true);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--summary") == 0) summary_only = true;
        else { fprintf(stderr, "shared_bank_conflict_stride_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (!*shape_point || strspn(shape_point, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape_point)) {
        fprintf(stderr, "shared_bank_conflict_stride_run: out_of_domain: invalid shape-point token\n");
        return 2;
    }
    const uint64_t plane = (uint64_t)rows * (uint64_t)cols;
    if (plane > ((uint64_t)INT_MAX - 32) / (uint64_t)stride) {
        fprintf(stderr, "shared_bank_conflict_stride_run: out_of_domain: shape product exceeds kernel index range\n");
        return 2;
    }
    const uint64_t elements = plane * (uint64_t)stride;
    if (elements > (uint64_t)INT_MAX - 32 || elements > SIZE_MAX / (2 * sizeof(float))) {
        fprintf(stderr, "shared_bank_conflict_stride_run: out_of_domain: shape overflows index or allocation\n");
        return 2;
    }
    if (elements - 1 > (uint64_t)INT_MAX / (uint64_t)stride) {
        fprintf(stderr, "shared_bank_conflict_stride_run: out_of_domain: strided index would overflow\n");
        return 2;
    }
    const size_t n = (size_t)elements, bytes = n * sizeof(float);
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc) fail("memory query", rc);
    int max_shared = 0, default_shared = 0, device = 0;
    rc = cudaGetDevice(&device); if (rc) fail("device query", rc);
    rc = cudaDeviceGetAttribute(&default_shared, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (rc) fail("default shared-memory query", rc);
    rc = cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (rc) fail("shared-memory query", rc);
    if (bytes > (size_t)max_shared) {
        fprintf(stderr, "shared_bank_conflict_stride_run: out_of_domain: requested %zu shared bytes exceeds opt-in limit %d\n", bytes, max_shared);
        return 2;
    }
    if (bytes > free_bytes / 3) {
        fprintf(stderr, "shared_bank_conflict_stride_run: out of memory: shape exceeds free device memory\n");
        return 1;
    }
    float *h_in = (float *)malloc(bytes), *h_out = (float *)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "shared_bank_conflict_stride_run: out of memory: host allocation failed\n");
        free(h_in); free(h_out); return 1;
    }
    seed_tensor(h_in, n, seed, -1.0, 1.0);
    float *d_in = nullptr, *d_out = nullptr, *d_sink = nullptr, h_sink = 0.0f;
    rc = cudaMalloc(&d_in, bytes); if (rc) fail("mc in", rc);
    rc = cudaMalloc(&d_out, bytes); if (rc) fail("mc out", rc);
    rc = cudaMalloc(&d_sink, sizeof(float)); if (rc) fail("mc sink", rc);
    rc = cudaMemset(d_out, 0xff, bytes); if (rc) fail("memset out", rc);
    rc = cudaMemset(d_sink, 0xff, sizeof(float)); if (rc) fail("memset sink", rc);
    rc = cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice); if (rc) fail("H2D", rc);
    rc = shared_bank_conflict_stride_launch(d_in, d_out, d_sink, rows, cols, stride, 0);
    if (rc) fail("configure/launch", rc);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);
    rc = cudaMemcpy(&h_sink, d_sink, sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("sink D2H", rc);
    float expected_sink = 0.0f;
    for (size_t i = 0; i < n; i += 32) expected_sink += h_in[(i * (size_t)stride) % n];
    size_t mismatch_count = 0;
    for (size_t i = 0; i < n; ++i) if (!isfinite(h_out[i]) || h_out[i] != h_in[i]) ++mismatch_count;
    const double sink_error = fabs((double)h_sink - (double)expected_sink);
    const bool correct = mismatch_count == 0 && isfinite(h_sink) && sink_error <= 1.0e-4;
    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"shared_bank_conflict_stride\",\n");
    const char *shape_set = strncmp(shape_point, "trace_", 6) == 0 ? "trace_shapes" : strncmp(shape_point, "timing_", 7) == 0 ? "timing_shapes" : "correctness_shapes";
    printf("  \"shape_set\": \"%s\",\n  \"shape_point\": \"%s\",\n", shape_set, shape_point);
    printf("  \"rows\": %d, \"cols\": %d, \"stride\": %d, \"seed\": %u,\n", rows, cols, stride, seed);
    printf("  \"shared_bytes\": %zu, \"default_shared_limit\": %d, \"optin_shared_limit\": %d,\n", bytes, default_shared, max_shared);
    printf("  \"correct\": %s, \"mismatch_count\": %zu, \"sink\": %.9g, \"expected_sink\": %.9g, \"sink_abs_error\": %.9g", correct ? "true" : "false", mismatch_count, h_sink, expected_sink, sink_error);
    if (!summary_only) {
        printf(",\n  \"output\": [");
        for (int r = 0; r < rows; ++r) { if (r > 0) printf(","); printf("["); for (int c = 0; c < cols; ++c) { if (c > 0) printf(","); printf("["); for (int s = 0; s < stride; ++s) { if (s > 0) printf(","); printf("%.6g", h_out[((size_t)r * cols + c) * stride + s]); } printf("]"); } printf("]"); }
        printf("]");
    }
    printf("\n}\n");
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_sink); free(h_in); free(h_out);
    return correct ? 0 : 1;
}
