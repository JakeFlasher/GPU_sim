// Host runner for layernorm_fwd_bf16_sm80, an LCD-portable BF16
// LayerNorm forward kernel using a one-pass Welford reduction.
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`.
//
// Correctness mode: rows=8, hidden=128, seeded inputs, in-process
// FP32 LayerNorm reference, per-element absolute tolerance 5e-2
// (BF16 quantization through (x-mean)*inv_std*scale + bias).
// Timing mode: rows=64, hidden=4096.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" void layernorm_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_scale,
    const __nv_bfloat16 *d_bias, __nv_bfloat16 *d_out,
    int rows, int hidden, float eps, cudaStream_t stream);

static const char *PROG = "layernorm_fwd_bf16_sm80_run";
static const char *SLUG = "layernorm_fwd_bf16_sm80";

static void cpu_layernorm_fwd(const float *x, const float *scale, const float *bias,
                              float *out, int rows, int hidden, float eps)
{
    for (int r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (int c = 0; c < hidden; ++c) mean += (double)x[r * hidden + c];
        mean /= (double)hidden;
        double var = 0.0;
        for (int c = 0; c < hidden; ++c) {
            double d = (double)x[r * hidden + c] - mean;
            var += d * d;
        }
        var /= (double)hidden;
        double inv_std = 1.0 / sqrt(var + (double)eps);
        for (int c = 0; c < hidden; ++c) {
            double y = ((double)x[r * hidden + c] - mean) * inv_std
                       * (double)scale[c] + (double)bias[c];
            out[r * hidden + c] = (float)y;
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

    int rows, hidden;
    const char *shape_point;
    const float eps = 1e-5f;
    if (A.mode == Mode::Correctness) { rows = 8;  hidden = 128;  shape_point = "correctness_00"; }
    else                             { rows = 64; hidden = 4096; shape_point = "timing_00"; }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    const size_t n_x = (size_t)rows * hidden;
    const size_t n_h = (size_t)hidden;

    float *X_f = (float *)malloc(n_x * sizeof(float));
    float *S_f = (float *)malloc(n_h * sizeof(float));
    float *B_f = (float *)malloc(n_h * sizeof(float));
    float *O_f = (float *)malloc(n_x * sizeof(float));
    float *Ref = (A.mode == Mode::Correctness) ? (float *)malloc(n_x * sizeof(float)) : nullptr;
    seed_tensor(X_f, n_x, A.seed,     -1.0, 1.0);
    seed_tensor(S_f, n_h, A.seed + 1,  0.5, 1.5);
    seed_tensor(B_f, n_h, A.seed + 2, -0.1, 0.1);

    __nv_bfloat16 *h_X = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_S = (__nv_bfloat16 *)malloc(n_h * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_B = (__nv_bfloat16 *)malloc(n_h * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x; ++i) h_X[i] = __float2bfloat16(X_f[i]);
    for (size_t i = 0; i < n_h; ++i) { h_S[i] = __float2bfloat16(S_f[i]); h_B[i] = __float2bfloat16(B_f[i]); }

    __nv_bfloat16 *d_X = nullptr, *d_S = nullptr, *d_B = nullptr, *d_O = nullptr;
    cudaMalloc(&d_X, n_x * sizeof(__nv_bfloat16));
    cudaMalloc(&d_S, n_h * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, n_h * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O, n_x * sizeof(__nv_bfloat16));
    cudaMemcpy(d_X, h_X, n_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, h_S, n_h * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_h * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        layernorm_fwd_bf16_sm80_launch(d_X, d_S, d_B, d_O, rows, hidden, eps, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        layernorm_fwd_bf16_sm80_launch(d_X, d_S, d_B, d_O, rows, hidden, eps, 0);
    });

    __nv_bfloat16 *h_O = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    cudaMemcpy(h_O, d_O, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_x; ++i) O_f[i] = __bfloat162float(h_O[i]);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_layernorm_fwd(X_f, S_f, B_f, Ref, rows, hidden, eps);
        for (size_t i = 0; i < n_x; ++i) {
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
    printf("  \"rows\": %d, \"hidden\": %d,\n", rows, hidden);
    printf("  \"reps\": %d, \"warmup\": %d,\n", A.reps, A.warmup);
    printf("  \"mean_ms\": %.6f, \"cv_pct\": %.6f,\n", mean_ms, cv_pct);
    if (A.mode == Mode::Correctness) {
        printf("  \"max_abs_err\": %.6g, \"correctness_ok\": %s,\n", max_abs_err, correctness_ok ? "true" : "false");
    }
    printf("  \"manifest\": \"%s\"\n", manifest_path);
    printf("}\n");

    cudaFree(d_X); cudaFree(d_S); cudaFree(d_B); cudaFree(d_O);
    free(X_f); free(S_f); free(B_f); free(O_f); free(Ref);
    free(h_X); free(h_S); free(h_B); free(h_O); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
