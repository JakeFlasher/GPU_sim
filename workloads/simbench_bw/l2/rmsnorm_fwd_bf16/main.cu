// Host driver for rmsnorm_fwd_bf16. Allocates seeded input + gamma on GPU,
// runs the kernel, and dumps the output as JSON on stdout so the orchestrator
// (`run_suite.py --mode correctness --kernel rmsnorm_fwd_bf16`) can read it
// and compare against the Python reference under the BF16 norm tolerance.
//
// Input seeding mirrors references/_common.py::deterministic_seeded_tensor:
//
//     state = (seed * 2654435761u + 1u)
//     next_u32: state = state * 1103515245u + 12345u; return state
//     float32 = low + (high - low) * (double(next_u32()) / 2**32)
//
// Using double in the scaling keeps us byte-identical with the Python
// reference's seeded tensor before the BF16 cast.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" void rmsnorm_fwd_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_gamma,
    __nv_bfloat16 *d_out,
    int rows, int hidden, float eps, cudaStream_t stream);

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
    for (size_t i = 0; i < n; ++i) {
        flat[i] = next_float(&state, low, high);
    }
}


static void fail(const char *msg, int rc) {
    fprintf(stderr, "rmsnorm_fwd_bf16_run: %s (cuda rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}

static uint32_t parse_u32(const char *name, const char *text,
                          uint32_t max_value, bool zero_ok) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] < '0' || text[0] > '9' || errno || *end ||
        value > max_value || (!zero_ok && value == 0)) {
        fprintf(stderr, "rmsnorm_fwd_bf16_run: out_of_domain: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static float parse_eps(const char *text) {
    char *end = nullptr;
    errno = 0;
    const float value = strtof(text, &end);
    if (end == text || *end || errno || !isfinite(value) || value < 1.17549435e-38f) {
        fprintf(stderr, "rmsnorm_fwd_bf16_run: out_of_domain: eps must be a positive normal float\n");
        exit(2);
    }
    return value;
}


int main(int argc, char **argv) {
    int rows = 32;
    int hidden = 512;
    uint32_t seed = 42;
    float eps = 1e-5f;
    const char *shape_point = "correctness_00";
    const char *slug = "rmsnorm_fwd_bf16";
    bool summary = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = (int)parse_u32("rows", argv[++i], INT_MAX, false);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = (int)parse_u32("hidden", argv[++i], INT_MAX - 256, false);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = parse_u32("seed", argv[++i], UINT32_MAX, true);
        else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) eps = parse_eps(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--summary") == 0) summary = true;
        else {
            fprintf(stderr, "rmsnorm_fwd_bf16_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    if (!*shape_point || strspn(shape_point, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != strlen(shape_point)) {
        fprintf(stderr, "rmsnorm_fwd_bf16_run: out_of_domain: invalid shape-point token\n");
        return 2;
    }
    const uint64_t count = (uint64_t)rows * (uint64_t)hidden;
    // The kernel uses signed row*hidden+c and c+=256 indexing.
    if (count > INT_MAX || count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "rmsnorm_fwd_bf16_run: out_of_domain: shape overflows index/allocation\n");
        return 2;
    }
    const size_t n_x = (size_t)count;
    const size_t n_g = (size_t)hidden;
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (rc) fail("memory query", rc);
    const uint64_t device_bytes = (2 * count + n_g) * sizeof(__nv_bfloat16);
    if (device_bytes > free_bytes) {
        fprintf(stderr, "rmsnorm_fwd_bf16_run: out of memory: requested shape exceeds free device memory\n");
        return 1;
    }
    float *h_x_f32 = (float *)malloc(n_x * sizeof(float));
    float *h_g_f32 = (float *)malloc(n_g * sizeof(float));
    if (!h_x_f32 || !h_g_f32) {
        fprintf(stderr, "rmsnorm_fwd_bf16_run: out of memory: FP32 host allocation\n");
        free(h_x_f32); free(h_g_f32);
        return 1;
    }
    seed_tensor(h_x_f32, n_x, seed, -1.0, 1.0);
    seed_tensor(h_g_f32, n_g, seed + 1, 0.5, 1.5);

    __nv_bfloat16 *h_x = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_g = (__nv_bfloat16 *)malloc(n_g * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    if (!h_x || !h_g || !h_out) {
        fprintf(stderr, "rmsnorm_fwd_bf16_run: out of memory: BF16 host allocation\n");
        free(h_x_f32); free(h_g_f32); free(h_x); free(h_g); free(h_out);
        return 1;
    }
    for (size_t i = 0; i < n_x; ++i) h_x[i] = __float2bfloat16(h_x_f32[i]);
    for (size_t i = 0; i < n_g; ++i) h_g[i] = __float2bfloat16(h_g_f32[i]);

    __nv_bfloat16 *d_x = nullptr, *d_g = nullptr, *d_out = nullptr;
    rc = cudaMalloc(&d_x, n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_x", rc);
    rc = cudaMalloc(&d_g, n_g * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_g", rc);
    rc = cudaMalloc(&d_out, n_x * sizeof(__nv_bfloat16)); if (rc) fail("cudaMalloc d_out", rc);
    rc = cudaMemset(d_out, 0xff, n_x * sizeof(__nv_bfloat16)); if (rc) fail("initialize output", rc);
    rc = cudaMemcpy(d_x, h_x, n_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D x", rc);
    rc = cudaMemcpy(d_g, h_g, n_g * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D g", rc);

    // CUDA-event duration is a separate observation, not the NCU cycle target.
    cudaEvent_t kernel_start, kernel_stop;
    rc = cudaEventCreate(&kernel_start); if (rc) fail("eventCreate start", rc);
    rc = cudaEventCreate(&kernel_stop); if (rc) fail("eventCreate stop", rc);
    rc = cudaEventRecord(kernel_start, 0); if (rc) fail("eventRecord start", rc);
    rmsnorm_fwd_bf16_launch(d_x, d_g, d_out, rows, hidden, eps, 0);
    rc = cudaGetLastError(); if (rc) fail("launch", rc);
    rc = cudaEventRecord(kernel_stop, 0); if (rc) fail("eventRecord stop", rc);
    rc = cudaEventSynchronize(kernel_stop); if (rc) fail("eventSynchronize", rc);
    float elapsed_ms = 0.0f;
    rc = cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop);
    if (rc) fail("eventElapsedTime", rc);
    double wall_time_ms = (double)elapsed_ms;
    if (!isfinite(wall_time_ms) || wall_time_ms < 0) {
        fprintf(stderr, "rmsnorm_fwd_bf16_run: invalid event duration\n");
        return 1;
    }
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);

    rc = cudaMemcpy(h_out, d_out, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost); if (rc) fail("D2H out", rc);

    // Compare every element with a double reference using the actual BF16
    // inputs (not the unquantized random numbers). The declared tolerance is
    // one BF16 relative ULP plus a small near-zero absolute allowance; it is
    // a numerical acceptance criterion, not a bitwise-equivalence claim.
    const double rtol = 1.0 / 128.0, atol = 1.0e-6;
    size_t mismatches = 0;
    double max_abs_error = 0.0, max_relative_error = 0.0;
    for (int r = 0; r < rows; ++r) {
        const size_t base = (size_t)r * hidden;
        double sum_sq = 0.0;
        for (int c = 0; c < hidden; ++c) {
            const double x = __bfloat162float(h_x[base + c]);
            sum_sq += x * x;
        }
        const double inv = 1.0 / sqrt(sum_sq / hidden + eps);
        for (int c = 0; c < hidden; ++c) {
            const double reference = __bfloat162float(h_x[base + c]) * inv * __bfloat162float(h_g[c]);
            const double actual = __bfloat162float(h_out[base + c]);
            const double error = fabs(actual - reference);
            if (!isfinite(actual) || !isfinite(reference) || error > atol + rtol * fabs(reference)) ++mismatches;
            max_abs_error = fmax(max_abs_error, error);
            max_relative_error = fmax(max_relative_error, error / fmax(atol, fabs(reference)));
        }
    }
    const bool correct = mismatches == 0;

    // Emit JSON: the "output" field matches the Python reference's
    // list-of-lists of FP32 values (bfloat16 promoted back to float). Use
    // enough precision that bfloat16's 7-bit mantissa round-trips cleanly.
    // wall_time_ms is the cudaEventElapsedTime-derived measurement that
    // run_suite.py picks up via its existing stdout-JSON parse.
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    const char *shape_set = !strncmp(shape_point, "timing_", 7) ? "timing_shapes" :
                            !strncmp(shape_point, "trace_", 6) ? "trace_shapes" : "correctness_shapes";
    printf("  \"shape_set\": \"%s\",\n", shape_set);
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"wall_time_ms\": %.9f,\n", wall_time_ms);
    printf("  \"rows\": %d, \"hidden\": %d, \"seed\": %u, \"eps\": %.9g,\n", rows, hidden, seed, eps);
    printf("  \"grid\": [%d,1,1], \"block\": [256,1,1],\n", rows);
    printf("  \"correct\": %s, \"mismatch_count\": %zu,\n", correct ? "true" : "false", mismatches);
    printf("  \"max_abs_error\": %.9g, \"max_relative_error\": %.9g,\n", max_abs_error, max_relative_error);
    printf("  \"rtol\": %.9g, \"atol\": %.9g", rtol, atol);
    if (!summary) {
    printf(",\n  \"output\": [");
    for (int r = 0; r < rows; ++r) {
        if (r > 0) printf(",");
        printf("[");
        for (int c = 0; c < hidden; ++c) {
            if (c > 0) printf(",");
            float v = __bfloat162float(h_out[r * hidden + c]);
            printf("%.6g", v);
        }
        printf("]");
    }
    printf("]");
    }
    printf("\n}\n");

    cudaFree(d_x); cudaFree(d_g); cudaFree(d_out);
    free(h_x_f32); free(h_g_f32); free(h_x); free(h_g); free(h_out);
    return correct ? 0 : 1;
}
