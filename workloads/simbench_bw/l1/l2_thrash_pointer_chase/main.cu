// Host driver for l2_thrash_pointer_chase. The pointer-chase sink and identity
// output are independently checked; unsafe shapes fail before allocation.
#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void l2_thrash_pointer_chase_launch(const float *d_in, float *d_out, float *d_sink,
                                                int chain_length, int warps, cudaStream_t stream);
static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) { uint32_t st = seed * 2654435761u + 1u; for (size_t i = 0; i < n; ++i) { double r = (double)next_u32(&st) / 4294967296.0; flat[i] = (float)(lo + (hi - lo) * r); } }
static void fail(const char *msg, int rc) { fprintf(stderr, "l2_thrash_pointer_chase_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc)); exit(1); }
static uint32_t parse_u32(const char *name, const char *text, uint32_t maximum) { char *end = nullptr; errno = 0; const unsigned long long value = strtoull(text, &end, 10); if (text[0] < '0' || text[0] > '9' || errno || !end || *end || value == 0 || value > maximum) { fprintf(stderr, "l2_thrash_pointer_chase_run: out_of_domain: invalid %s: %s\n", name, text); exit(2); } return (uint32_t)value; }

int main(int argc, char **argv) {
    int chain_length = 16, warps = 4; uint32_t seed = 42; const char *shape_point = "correctness_00"; bool summary_only = false;
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--chain-length") == 0 && i + 1 < argc) chain_length = (int)parse_u32("chain-length", argv[++i], INT_MAX);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc) warps = (int)parse_u32("warps", argv[++i], INT_MAX);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = parse_u32("seed", argv[++i], UINT32_MAX);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--summary") == 0) summary_only = true;
        else { fprintf(stderr, "l2_thrash_pointer_chase_run: unknown arg %s\n", argv[i]); return 2; }
    }
    if (!*shape_point || strspn(shape_point, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape_point)) { fprintf(stderr, "l2_thrash_pointer_chase_run: out_of_domain: invalid shape-point token\n"); return 2; }
    const uint64_t total64 = (uint64_t)chain_length * (uint64_t)warps;
    if (total64 > (uint64_t)INT_MAX - 32 || total64 > SIZE_MAX / (2 * sizeof(float))) { fprintf(stderr, "l2_thrash_pointer_chase_run: out_of_domain: shape overflows index or allocation\n"); return 2; }
    const size_t n = (size_t)total64, bytes = n * sizeof(float); size_t free_bytes = 0, total_bytes = 0; cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes); if (rc) fail("memory query", rc); if (bytes > free_bytes / 3) { fprintf(stderr, "l2_thrash_pointer_chase_run: out of memory: requested shape exceeds free device memory\n"); return 1; }
    float *h_in = (float *)malloc(bytes), *h_out = (float *)malloc(bytes); if (!h_in || !h_out) { fprintf(stderr, "l2_thrash_pointer_chase_run: out of memory: host allocation failed\n"); free(h_in); free(h_out); return 1; }
    seed_tensor(h_in, n, seed, -1.0, 1.0);
    float expected_sink = 0.0f; int idx = 0;
    for (int i = 0; i < chain_length; ++i) { const float v = h_in[idx]; expected_sink += v; const uint32_t bits = *(const uint32_t *)&v; idx = (int)(((bits >> 3) ^ (uint32_t)idx) % (uint32_t)n); }
    float *d_in = nullptr, *d_out = nullptr, *d_sink = nullptr, h_sink = 0.0f; rc = cudaMalloc(&d_in, bytes); if (rc) fail("mc in", rc); rc = cudaMalloc(&d_out, bytes); if (rc) fail("mc out", rc); rc = cudaMalloc(&d_sink, sizeof(float)); if (rc) fail("mc sink", rc); rc = cudaMemset(d_out, 0xff, bytes); if (rc) fail("memset out", rc); rc = cudaMemset(d_sink, 0xff, sizeof(float)); if (rc) fail("memset sink", rc); rc = cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice); if (rc) fail("H2D", rc);
    l2_thrash_pointer_chase_launch(d_in, d_out, d_sink, chain_length, warps, 0); rc = cudaGetLastError(); if (rc) fail("launch", rc); rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc); rc = cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc); rc = cudaMemcpy(&h_sink, d_sink, sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("sink D2H", rc);
    size_t mismatch_count = 0; for (size_t i = 0; i < n; ++i) if (!isfinite(h_out[i]) || h_out[i] != h_in[i]) ++mismatch_count; const double sink_error = fabs((double)h_sink - (double)expected_sink); const bool correct = mismatch_count == 0 && isfinite(h_sink) && sink_error <= 1.0e-4;
    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"l2_thrash_pointer_chase\",\n"); const char *shape_set = strncmp(shape_point, "trace_", 6) == 0 ? "trace_shapes" : strncmp(shape_point, "timing_", 7) == 0 ? "timing_shapes" : "correctness_shapes"; printf("  \"shape_set\": \"%s\",\n  \"shape_point\": \"%s\",\n", shape_set, shape_point); printf("  \"chain_length\": %d, \"warps\": %d, \"seed\": %u,\n", chain_length, warps, seed); printf("  \"correct\": %s, \"mismatch_count\": %zu, \"sink\": %.9g, \"expected_sink\": %.9g, \"sink_abs_error\": %.9g", correct ? "true" : "false", mismatch_count, h_sink, expected_sink, sink_error);
    if (!summary_only) { printf(",\n  \"output\": ["); for (int c = 0; c < chain_length; ++c) { if (c) printf(","); printf("["); for (int w = 0; w < warps; ++w) { if (w) printf(","); printf("%.6g", h_out[(size_t)c * warps + w]); } printf("]"); } printf("]"); }
    printf("\n}\n"); cudaFree(d_in); cudaFree(d_out); cudaFree(d_sink); free(h_in); free(h_out); return correct ? 0 : 1;
}
