// Validate both passes and eviction after the uninterrupted launch sequence.
// The first/repeat names specify order, not a guarantee of cache temperature.
#include <cuda_runtime.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" cudaError_t warm_cold_pair_relaunch_launch(
    const float *, float *, float *, float *, float *, int, int, int, int, cudaStream_t);

static void check(cudaError_t rc, const char *operation) {
    if (rc != cudaSuccess) {
        fprintf(stderr, "warm_cold_pair_relaunch: %s: %s\n", operation, cudaGetErrorString(rc));
        exit(1);
    }
}

static uint32_t parse_u32(const char *name, const char *text, uint32_t maximum, bool zero_ok) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || !end || *end ||
        value > maximum || (!zero_ok && value == 0)) {
        fprintf(stderr, "warm_cold_pair_relaunch: out_of_domain: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static void seed_input(float *data, size_t count, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1103515245u + 12345u;
        data[i] = (float)(-1.0 + 2.0 * (double)state / 4294967296.0);
    }
}

int main(int argc, char **argv) {
    int allocation_bytes = 4096, eviction_bytes = 0, accesses = 64, warps = 1;
    uint32_t seed = 42;
    const char *shape = "correctness_00";
    bool summary = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--summary")) { summary = true; continue; }
        const char *name = argv[i];
        if (++i >= argc) { fprintf(stderr, "warm_cold_pair_relaunch: missing value for %s\n", name); return 2; }
        if (!strcmp(name, "--allocation-bytes")) allocation_bytes = (int)parse_u32(name, argv[i], INT_MAX, false);
        else if (!strcmp(name, "--eviction-bytes")) eviction_bytes = (int)parse_u32(name, argv[i], INT_MAX, true);
        else if (!strcmp(name, "--accesses")) accesses = (int)parse_u32(name, argv[i], INT_MAX, false);
        else if (!strcmp(name, "--warps")) warps = (int)parse_u32(name, argv[i], INT_MAX / 32, false);
        else if (!strcmp(name, "--seed")) seed = parse_u32(name, argv[i], UINT32_MAX, true);
        else if (!strcmp(name, "--shape-point")) shape = argv[i];
        else { fprintf(stderr, "warm_cold_pair_relaunch: unknown argument %s\n", name); return 2; }
    }
    if (!*shape || strspn(shape, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape) ||
        allocation_bytes % 4 || eviction_bytes % 4) {
        fprintf(stderr, "warm_cold_pair_relaunch: out_of_domain: invalid label or non-float-aligned byte count\n");
        return 2;
    }
    const size_t n = (size_t)allocation_bytes / sizeof(float);
    const size_t scratch_count = (size_t)eviction_bytes / sizeof(float);
    const uint64_t threads = (uint64_t)warps * 32;
    if (n > INT_MAX - threads || scratch_count > INT_MAX - threads) {
        fprintf(stderr, "warm_cold_pair_relaunch: out_of_domain: grid-stride index overflow\n");
        return 2;
    }
    const uint64_t device_bytes = 3 * (uint64_t)allocation_bytes + eviction_bytes + 2 * sizeof(float);
    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "memory query");
    if (device_bytes > free_bytes || device_bytes > SIZE_MAX) {
        fprintf(stderr, "warm_cold_pair_relaunch: out of memory: need %llu bytes, free %zu\n",
                (unsigned long long)device_bytes, free_bytes);
        return 1;
    }
    float *input = (float *)malloc(allocation_bytes);
    float *first = (float *)malloc(allocation_bytes);
    float *repeat = (float *)malloc(allocation_bytes);
    float *scratch = eviction_bytes ? (float *)malloc(eviction_bytes) : nullptr;
    if (!input || !first || !repeat || (eviction_bytes && !scratch)) {
        fprintf(stderr, "warm_cold_pair_relaunch: out of memory: host allocation\n");
        free(input); free(first); free(repeat); free(scratch); return 1;
    }
    seed_input(input, n, seed);
    // Independent higher-precision reference for global thread zero's reads.
    double expected_sink = 0.0, sum_abs = 0.0;
    for (int a = 0; a < accesses; ++a) {
        const double value = input[((uint64_t)a * 32) % n];
        expected_sink += value;
        sum_abs += fabs(value);
    }
    // Standard sequential-summation error bound, valid only while n*u < 1.
    const double nu = (double)accesses * (FLT_EPSILON / 2);
    if (nu >= 0.5) {
        fprintf(stderr, "warm_cold_pair_relaunch: out_of_domain: accesses too large for useful sink error bound\n");
        free(input); free(first); free(repeat); free(scratch); return 2;
    }
    const double sink_tolerance = nu / (1.0 - nu) * sum_abs + 1e-6;
    float *d_input = nullptr, *d_first = nullptr, *d_repeat = nullptr;
    float *d_sinks = nullptr, *d_scratch = nullptr;
    check(cudaMalloc(&d_input, allocation_bytes), "allocate input");
    check(cudaMalloc(&d_first, allocation_bytes), "allocate first output");
    check(cudaMalloc(&d_repeat, allocation_bytes), "allocate repeat output");
    check(cudaMalloc(&d_sinks, 2 * sizeof(float)), "allocate sinks");
    if (eviction_bytes) {
        check(cudaMalloc(&d_scratch, eviction_bytes), "allocate scratch");
        check(cudaMemset(d_scratch, 0, eviction_bytes), "initialize scratch");
    }
    check(cudaMemset(d_first, 0xff, allocation_bytes), "initialize first output");
    check(cudaMemset(d_repeat, 0xff, allocation_bytes), "initialize repeat output");
    check(cudaMemset(d_sinks, 0xff, 2 * sizeof(float)), "initialize sinks");
    check(cudaMemcpy(d_input, input, allocation_bytes, cudaMemcpyHostToDevice), "copy input");
    check(warm_cold_pair_relaunch_launch(d_input, d_first, d_repeat, d_sinks, d_scratch,
                                        (int)n, accesses, (int)scratch_count, warps, 0), "launch sequence");
    check(cudaDeviceSynchronize(), "synchronize sequence");
    check(cudaMemcpy(first, d_first, allocation_bytes, cudaMemcpyDeviceToHost), "first output");
    check(cudaMemcpy(repeat, d_repeat, allocation_bytes, cudaMemcpyDeviceToHost), "repeat output");
    float sinks[2];
    check(cudaMemcpy(sinks, d_sinks, sizeof(sinks), cudaMemcpyDeviceToHost), "sinks");
    if (eviction_bytes) check(cudaMemcpy(scratch, d_scratch, eviction_bytes, cudaMemcpyDeviceToHost), "scratch");
    size_t first_mismatches = 0, repeat_mismatches = 0, eviction_mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(first[i]) || first[i] != input[i]) ++first_mismatches;
        if (!isfinite(repeat[i]) || repeat[i] != input[i]) ++repeat_mismatches;
    }
    for (size_t i = 0; i < scratch_count; ++i) if (scratch[i] != 1.0f) ++eviction_mismatches;
    const bool first_sink_ok = isfinite(sinks[0]) && fabs(sinks[0] - expected_sink) <= sink_tolerance;
    const bool repeat_sink_ok = isfinite(sinks[1]) && fabs(sinks[1] - expected_sink) <= sink_tolerance;
    const bool correct = !first_mismatches && !repeat_mismatches && !eviction_mismatches && first_sink_ok && repeat_sink_ok;
    const char *shape_set = !strncmp(shape, "trace_", 6) ? "trace_shapes" :
                            !strncmp(shape, "timing_", 7) ? "timing_shapes" : "correctness_shapes";
    printf("{\n  \"schema_version\": 1, \"slug\": \"warm_cold_pair_relaunch\",\n");
    printf("  \"shape_set\": \"%s\", \"shape_point\": \"%s\",\n", shape_set, shape);
    printf("  \"allocation_bytes\": %d, \"eviction_bytes\": %d, \"accesses\": %d, \"warps\": %d, \"seed\": %u,\n",
           allocation_bytes, eviction_bytes, accesses, warps, seed);
    printf("  \"correct\": %s, \"first_mismatches\": %zu, \"repeat_mismatches\": %zu, \"eviction_mismatches\": %zu,\n",
           correct ? "true" : "false", first_mismatches, repeat_mismatches, eviction_mismatches);
    printf("  \"first_sink_correct\": %s, \"repeat_sink_correct\": %s, \"sink_tolerance\": %.12g",
           first_sink_ok ? "true" : "false", repeat_sink_ok ? "true" : "false", sink_tolerance);
    if (!summary && correct) {
        printf(",\n  \"output\": [");
        for (size_t i = 0; i < n && i < 64; ++i) printf("%s%.6g", i ? "," : "", repeat[i]);
        printf("]");
    }
    printf("\n}\n");
    check(cudaFree(d_input), "free input");
    check(cudaFree(d_first), "free first output");
    check(cudaFree(d_repeat), "free repeat output");
    check(cudaFree(d_sinks), "free sinks");
    if (d_scratch) check(cudaFree(d_scratch), "free scratch");
    free(input); free(first); free(repeat); free(scratch);
    return correct ? 0 : 1;
}
