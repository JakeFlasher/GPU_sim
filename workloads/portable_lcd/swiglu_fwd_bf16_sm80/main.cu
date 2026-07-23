// Host runner for swiglu_fwd_bf16_sm80, an LCD-portable element-wise
// BF16 SwiGLU activation kernel.
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`.
//
// Correctness: n = 8 * 128 = 1024; FP32 SwiGLU reference; tolerance 5e-2.
// Timing: n = 64 * 4096.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../runner_common.h"

extern "C" void swiglu_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_gate, const __nv_bfloat16 *d_up,
    __nv_bfloat16 *d_out, int64_t n, cudaStream_t stream);

static const char *PROG = "swiglu_fwd_bf16_sm80_run";
static const char *SLUG = "swiglu_fwd_bf16_sm80";

static void cpu_swiglu_fwd_fp32(const float *gate, const float *up, float *out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        double g = (double)gate[i];
        double u = (double)up[i];
        double silu = g / (1.0 + exp(-g));
        out[i] = (float)(silu * u);
    }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    // This activation runner repurposes only `--n` as the element count
    // (R42); `--m` / `--k` are NOT consumed, so accept the narrow `--n`-only
    // override mode rather than the full GEMM family (which would silently
    // accept-and-ignore `--m` / `--k`).
    require_accepted_overrides(A, PROG, {.activation_n = true});

    int64_t n;
    const char *shape_point;
    if (A.mode == Mode::Correctness) { n = 8LL * 128;   shape_point = "correctness_00"; }
    else                             { n = 64LL * 4096; shape_point = "timing_00"; }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    // R42 shape-CLI extension: the runner_common.h --n override
    // (added in R40 for GEMM) is repurposed here as the swiglu
    // element count, honored in Mode::Timing only; correctness
    // mode rejects via the presence-boolean discipline established
    // in R26. Override validation runs BEFORE acquire_device()
    // per the R27 + R41 ordering invariant.
    if (A.has_override_n) {
        if (A.mode == Mode::Correctness) {
            fprintf(stderr, "%s: shape_override_in_correctness_mode_rejected\n", PROG);
            return 2;
        }
        if (A.override_n <= 0) {
            fprintf(stderr, "%s: --n must be a positive integer\n", PROG);
            return 2;
        }
        n = A.override_n;
    }

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    float *G_f = (float *)malloc((size_t)n * sizeof(float));
    float *U_f = (float *)malloc((size_t)n * sizeof(float));
    float *O_f = (float *)malloc((size_t)n * sizeof(float));
    float *Ref = (A.mode == Mode::Correctness) ? (float *)malloc((size_t)n * sizeof(float)) : nullptr;
    seed_tensor(G_f, (size_t)n, A.seed,     -2.0, 2.0);
    seed_tensor(U_f, (size_t)n, A.seed + 1, -2.0, 2.0);

    __nv_bfloat16 *h_G = (__nv_bfloat16 *)malloc((size_t)n * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_U = (__nv_bfloat16 *)malloc((size_t)n * sizeof(__nv_bfloat16));
    for (int64_t i = 0; i < n; ++i) {
        h_G[i] = __float2bfloat16(G_f[i]);
        h_U[i] = __float2bfloat16(U_f[i]);
    }

    __nv_bfloat16 *d_G, *d_U, *d_O;
    cudaMalloc(&d_G, (size_t)n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_U, (size_t)n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O, (size_t)n * sizeof(__nv_bfloat16));
    cudaMemcpy(d_G, h_G, (size_t)n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, (size_t)n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        swiglu_fwd_bf16_sm80_launch(d_G, d_U, d_O, n, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        swiglu_fwd_bf16_sm80_launch(d_G, d_U, d_O, n, 0);
    });

    __nv_bfloat16 *h_O = (__nv_bfloat16 *)malloc((size_t)n * sizeof(__nv_bfloat16));
    cudaMemcpy(h_O, d_O, (size_t)n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; ++i) O_f[i] = __bfloat162float(h_O[i]);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_swiglu_fwd_fp32(G_f, U_f, Ref, n);
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

    cudaFree(d_G); cudaFree(d_U); cudaFree(d_O);
    free(G_f); free(U_f); free(O_f); free(Ref);
    free(h_G); free(h_U); free(h_O); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
