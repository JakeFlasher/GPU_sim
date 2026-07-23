// Host runner for softmax_safe_bf16_sm80, an LCD-portable
// numerically-stable BF16 softmax-along-cols kernel.
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`. Correctness
// mode runs a small shape (rows=8, cols=128) and compares against an
// in-process FP32 safe-softmax reference; the absolute tolerance is
// 5e-2 to admit BF16 quantization through the exp + normalize chain.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" void softmax_safe_bf16_sm80_launch(
    const __nv_bfloat16 *d_x, __nv_bfloat16 *d_out,
    int rows, int cols, cudaStream_t stream);

static const char *PROG = "softmax_safe_bf16_sm80_run";
static const char *SLUG = "softmax_safe_bf16_sm80";

static void cpu_softmax_safe_fp32(const float *x, float *out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        double m = -DBL_MAX;
        for (int c = 0; c < cols; ++c) {
            double v = (double)x[r * cols + c];
            if (v > m) m = v;
        }
        double z = 0.0;
        for (int c = 0; c < cols; ++c) {
            z += exp((double)x[r * cols + c] - m);
        }
        double inv_z = 1.0 / z;
        for (int c = 0; c < cols; ++c) {
            out[r * cols + c] = (float)(exp((double)x[r * cols + c] - m) * inv_z);
        }
    }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    require_accepted_overrides(A, PROG, {});

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    int rows, cols;
    const char *shape_point;
    if (A.mode == Mode::Correctness) { rows = 8;  cols = 128;  shape_point = "correctness_00"; }
    else                             { rows = 64; cols = 4096; shape_point = "timing_00"; }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    const size_t n = (size_t)rows * cols;
    float *X_f = (float *)malloc(n * sizeof(float));
    float *O_f = (float *)malloc(n * sizeof(float));
    float *Ref = (A.mode == Mode::Correctness) ? (float *)malloc(n * sizeof(float)) : nullptr;
    seed_tensor(X_f, n, A.seed, -2.0, 2.0);

    __nv_bfloat16 *h_X = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n; ++i) h_X[i] = __float2bfloat16(X_f[i]);

    __nv_bfloat16 *d_X = nullptr, *d_O = nullptr;
    cudaMalloc(&d_X, n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O, n * sizeof(__nv_bfloat16));
    cudaMemcpy(d_X, h_X, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        softmax_safe_bf16_sm80_launch(d_X, d_O, rows, cols, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        softmax_safe_bf16_sm80_launch(d_X, d_O, rows, cols, 0);
    });

    __nv_bfloat16 *h_O = (__nv_bfloat16 *)malloc(n * sizeof(__nv_bfloat16));
    cudaMemcpy(h_O, d_O, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i) O_f[i] = __bfloat162float(h_O[i]);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_softmax_safe_fp32(X_f, Ref, rows, cols);
        for (size_t i = 0; i < n; ++i) {
            double e = fabs((double)O_f[i] - (double)Ref[i]);
            if (e > max_abs_err) max_abs_err = e;
        }
        const double tol = 5e-2;
        correctness_ok = (max_abs_err <= tol);
        if (!correctness_ok) {
            fprintf(stderr, "%s: correctness_failed: max_abs_err=%.6g tol=%.6g\n", PROG, max_abs_err, tol);
        }
    }

    double mean_ms, cv_pct;
    summarize_reps(samples_ms, A.reps, &mean_ms, &cv_pct);

    char manifest_path[1280];
    const char *mode_str = (A.mode == Mode::Correctness) ? "correctness" : "timing";
    emit_manifest_v2(PROG, SLUG, A, dp, mode_str, shape_point, cv_pct, manifest_path);

    printf("{\n");
    printf("  \"slug\": \"%s\",\n", SLUG);
    printf("  \"mode\": \"%s\",\n", mode_str);
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"rows\": %d, \"cols\": %d,\n", rows, cols);
    printf("  \"reps\": %d, \"warmup\": %d,\n", A.reps, A.warmup);
    printf("  \"mean_ms\": %.6f, \"cv_pct\": %.6f,\n", mean_ms, cv_pct);
    if (A.mode == Mode::Correctness) {
        printf("  \"max_abs_err\": %.6g, \"correctness_ok\": %s,\n", max_abs_err, correctness_ok ? "true" : "false");
    }
    printf("  \"manifest\": \"%s\"\n", manifest_path);
    printf("}\n");

    cudaFree(d_X); cudaFree(d_O);
    free(X_f); free(O_f); free(Ref);
    free(h_X); free(h_O); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
