// Host runner for rmsnorm_bwd_bf16_sm80, an LCD-portable BF16
// RMSNorm-backward (input gradient) kernel; sister of
// rmsnorm_fwd_bf16_sm80.
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`.
//
// Correctness mode: rows=8, hidden=128, seeded inputs, in-process
// FP64 RMSNorm-backward reference, per-element absolute
// tolerance 5e-2 (BF16 quantization through the
// inv_rms / inv_rms^3 / S-term chain).
// Timing mode: rows=64, hidden=4096.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" void rmsnorm_bwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_gamma,
    const __nv_bfloat16 *d_grad_y, __nv_bfloat16 *d_dx,
    int rows, int hidden, float eps, cudaStream_t stream);

static const char *PROG = "rmsnorm_bwd_bf16_sm80_run";
static const char *SLUG = "rmsnorm_bwd_bf16_sm80";

static void cpu_rmsnorm_bwd(const float *x, const float *gamma, const float *grad_y,
                            float *dx, int rows, int hidden, float eps)
{
    const int H = hidden;
    for (int r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int c = 0; c < H; ++c) {
            double xv = (double)x[r * H + c];
            ss += xv * xv;
        }
        double ms      = ss / (double)H;
        double inv_rms = 1.0 / sqrt(ms + (double)eps);
        double S = 0.0;
        for (int c = 0; c < H; ++c) {
            S += (double)grad_y[r * H + c] * (double)gamma[c] * (double)x[r * H + c];
        }
        double coeff = inv_rms * inv_rms * inv_rms / (double)H;
        for (int c = 0; c < H; ++c) {
            double dxv = (double)grad_y[r * H + c] * (double)gamma[c] * inv_rms
                       - (double)x[r * H + c] * S * coeff;
            dx[r * H + c] = (float)dxv;
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
    const float eps = 1e-6f;
    if (A.mode == Mode::Correctness) { rows = 8;  hidden = 128;  shape_point = "correctness_00"; }
    else                             { rows = 64; hidden = 4096; shape_point = "timing_00"; }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    const size_t n_x = (size_t)rows * hidden;
    const size_t n_h = (size_t)hidden;

    float *X_f  = (float *)malloc(n_x * sizeof(float));
    float *G_f  = (float *)malloc(n_h * sizeof(float));
    float *Gy_f = (float *)malloc(n_x * sizeof(float));
    float *Dx_f = (float *)malloc(n_x * sizeof(float));
    float *Ref  = (A.mode == Mode::Correctness) ? (float *)malloc(n_x * sizeof(float)) : nullptr;
    seed_tensor(X_f,  n_x, A.seed,     -1.0, 1.0);
    seed_tensor(G_f,  n_h, A.seed + 1,  0.5, 1.5);
    seed_tensor(Gy_f, n_x, A.seed + 2, -0.5, 0.5);

    __nv_bfloat16 *h_X  = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_G  = (__nv_bfloat16 *)malloc(n_h * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_Gy = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_x; ++i) { h_X[i] = __float2bfloat16(X_f[i]); h_Gy[i] = __float2bfloat16(Gy_f[i]); }
    for (size_t i = 0; i < n_h; ++i) h_G[i] = __float2bfloat16(G_f[i]);

    __nv_bfloat16 *d_X = nullptr, *d_G = nullptr, *d_Gy = nullptr, *d_Dx = nullptr;
    cudaMalloc(&d_X,  n_x * sizeof(__nv_bfloat16));
    cudaMalloc(&d_G,  n_h * sizeof(__nv_bfloat16));
    cudaMalloc(&d_Gy, n_x * sizeof(__nv_bfloat16));
    cudaMalloc(&d_Dx, n_x * sizeof(__nv_bfloat16));
    cudaMemcpy(d_X,  h_X,  n_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_G,  h_G,  n_h * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gy, h_Gy, n_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        rmsnorm_bwd_bf16_sm80_launch(d_X, d_G, d_Gy, d_Dx, rows, hidden, eps, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        rmsnorm_bwd_bf16_sm80_launch(d_X, d_G, d_Gy, d_Dx, rows, hidden, eps, 0);
    });

    __nv_bfloat16 *h_Dx = (__nv_bfloat16 *)malloc(n_x * sizeof(__nv_bfloat16));
    cudaMemcpy(h_Dx, d_Dx, n_x * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_x; ++i) Dx_f[i] = __bfloat162float(h_Dx[i]);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_rmsnorm_bwd(X_f, G_f, Gy_f, Ref, rows, hidden, eps);
        for (size_t i = 0; i < n_x; ++i) {
            double e = fabs((double)Dx_f[i] - (double)Ref[i]);
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

    cudaFree(d_X); cudaFree(d_G); cudaFree(d_Gy); cudaFree(d_Dx);
    free(X_f); free(G_f); free(Gy_f); free(Dx_f); free(Ref);
    free(h_X); free(h_G); free(h_Gy); free(h_Dx); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
