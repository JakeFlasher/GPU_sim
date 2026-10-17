// Host runner for silu_fwd_bf16_sm80, an LCD-portable element-wise
// BF16 SiLU activation kernel.
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`. Correctness
// mode runs a small shape (n = 8 * 128 = 1024) and compares against
// an in-process FP32 SiLU reference; tolerance 5e-2 (BF16 quantization
// through the SiLU activation).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" void silu_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_x,
    __nv_bfloat16 *d_out, int64_t n, cudaStream_t stream);

static const char *PROG = "silu_fwd_bf16_sm80_run";
static const char *SLUG = "silu_fwd_bf16_sm80";

static void cpu_silu_fwd_fp32(const float *x, float *out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        double v = (double)x[i];
        out[i] = (float)(v / (1.0 + exp(-v)));
    }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    require_accepted_overrides(A, PROG, {});

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    int64_t n;
    const char *shape_point;
    if (A.mode == Mode::Correctness) { n = 8LL * 128;          shape_point = "correctness_00"; }
    else                             { n = 64LL * 4096;        shape_point = "timing_00"; }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    float *X_f = (float *)malloc((size_t)n * sizeof(float));
    float *O_f = (float *)malloc((size_t)n * sizeof(float));
    float *Ref = (A.mode == Mode::Correctness) ? (float *)malloc((size_t)n * sizeof(float)) : nullptr;
    seed_tensor(X_f, (size_t)n, A.seed, -2.0, 2.0);

    __nv_bfloat16 *h_X = (__nv_bfloat16 *)malloc((size_t)n * sizeof(__nv_bfloat16));
    for (int64_t i = 0; i < n; ++i) h_X[i] = __float2bfloat16(X_f[i]);

    __nv_bfloat16 *d_X = nullptr, *d_O = nullptr;
    cudaMalloc(&d_X, (size_t)n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O, (size_t)n * sizeof(__nv_bfloat16));
    cudaMemcpy(d_X, h_X, (size_t)n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        silu_fwd_bf16_sm80_launch(d_X, d_O, n, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        silu_fwd_bf16_sm80_launch(d_X, d_O, n, 0);
    });

    __nv_bfloat16 *h_O = (__nv_bfloat16 *)malloc((size_t)n * sizeof(__nv_bfloat16));
    cudaMemcpy(h_O, d_O, (size_t)n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; ++i) O_f[i] = __bfloat162float(h_O[i]);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_silu_fwd_fp32(X_f, Ref, n);
        for (int64_t i = 0; i < n; ++i) {
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
    printf("  \"n\": %lld,\n", (long long)n);
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
