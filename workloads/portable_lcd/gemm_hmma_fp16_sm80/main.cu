// Host runner for gemm_hmma_fp16_sm80, an LCD-portable FP16 HMMA GEMM
// (sister of gemm_hmma_bf16_sm80; the two kernels jointly cover the
// Ampere FP16/BF16 HMMA encodings).
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`.
//
// Correctness mode:
//   M=32, N=16, K=64 FP16 inputs seeded by --seed; in-process FP32 matmul
//   reference; per-element absolute tolerance 5e-2 (FP16 has ~3 decimal
//   digits of precision; the cumulative error across 64 FP16
//   multiply-accumulates is bounded under this tolerance).
//
// Timing mode:
//   M=256, N=256, K=256 FP16 inputs; cudaEventRecord-bracketed warmup
//   and repeat loops; emits cv_pct over the repeat samples.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" void gemm_hmma_fp16_sm80_launch(
    const __half *d_A, const __half *d_B,
    float *d_C, int M, int N, int K, cudaStream_t stream);

static const char *PROG = "gemm_hmma_fp16_sm80_run";
static const char *SLUG = "gemm_hmma_fp16_sm80";

static void cpu_matmul_fp32(const float *A_rm, const float *B_rm,
                            float *C_rm, int M, int N, int K)
{
    // A_rm: [M, K] row-major. B_rm: [K, N] row-major. C_rm: [M, N] row-major.
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A_rm[m * K + k] * (double)B_rm[k * N + n];
            C_rm[m * N + n] = (float)acc;
        }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    require_accepted_overrides(A, PROG, {.gemm_mnk = true});

    int M, N, K;
    const char *shape_point;
    if (A.mode == Mode::Correctness) { M = 32; N = 16; K = 64;  shape_point = "correctness_00"; }
    else                             { M = 256; N = 256; K = 256; shape_point = "timing_00"; }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    // R40 shape-CLI extension: --m / --n / --k overrides honored
    // in Mode::Timing only; correctness mode rejects via the
    // presence-boolean discipline established in R26. R41 hardens
    // the validation ordering (R26/R27 ordering-bug class):
    // override validation runs BEFORE acquire_device() so an invalid
    // override under CUDA_VISIBLE_DEVICES=-1 returns rc=2 with the
    // typed stderr fragment instead of failing at cudaGetDevice.
    bool gemm_override_presence =
        A.has_override_m || A.has_override_n || A.has_override_k;
    if (gemm_override_presence) {
        if (A.mode == Mode::Correctness) {
            fprintf(stderr, "%s: shape_override_in_correctness_mode_rejected\n", PROG);
            return 2;
        }
        if (A.has_override_m) {
            if (A.override_m <= 0) {
                fprintf(stderr, "%s: --m must be a positive integer\n", PROG);
                return 2;
            }
            M = A.override_m;
        }
        if (A.has_override_n) {
            if (A.override_n <= 0) {
                fprintf(stderr, "%s: --n must be a positive integer\n", PROG);
                return 2;
            }
            N = A.override_n;
        }
        if (A.has_override_k) {
            if (A.override_k <= 0) {
                fprintf(stderr, "%s: --k must be a positive integer\n", PROG);
                return 2;
            }
            K = A.override_k;
        }
        // Tile-alignment validation: the FP16 HMMA kernel launches
        // with grid=((M+15)/16, (N+7)/8, 1) and writes C[M*N]
        // unguarded. The kernel requires M%16==0, N%8==0, K%16==0.
        if (M % 16 != 0 || N % 8 != 0 || K % 16 != 0) {
            fprintf(stderr,
                "%s: shape_override_tile_alignment_violation: "
                "kernel requires M%%16==0, N%%8==0, K%%16==0; "
                "got M=%d, N=%d, K=%d\n",
                PROG, M, N, K);
            return 2;
        }
    }

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    const size_t n_a = (size_t)M * K, n_b = (size_t)K * N, n_c = (size_t)M * N;

    float *A_f  = (float *)malloc(n_a * sizeof(float));
    float *B_f  = (float *)malloc(n_b * sizeof(float));
    float *C_f  = (float *)malloc(n_c * sizeof(float));
    float *Ref  = (A.mode == Mode::Correctness) ? (float *)malloc(n_c * sizeof(float)) : nullptr;
    seed_tensor(A_f, n_a, A.seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, A.seed + 1, -0.5, 0.5);

    __half *h_A = (__half *)malloc(n_a * sizeof(__half));
    __half *h_B = (__half *)malloc(n_b * sizeof(__half));
    for (size_t i = 0; i < n_a; ++i) h_A[i] = __float2half(A_f[i]);
    // Transpose B from [K, N] row-major to [N, K] col-major (== [N, K] row-major
    // of the transposed tensor) for the MMA's `.col` layout.
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            h_B[n * K + k] = __float2half(B_f[k * N + n]);

    __half *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc(&d_A, n_a * sizeof(__half));
    cudaMalloc(&d_B, n_b * sizeof(__half));
    cudaMalloc(&d_C, n_c * sizeof(float));
    cudaMemcpy(d_A, h_A, n_a * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_b * sizeof(__half), cudaMemcpyHostToDevice);

    // Warmup
    for (int w = 0; w < A.warmup; ++w) {
        gemm_hmma_fp16_sm80_launch(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    // Repeat loop with timing
    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        gemm_hmma_fp16_sm80_launch(d_A, d_B, d_C, M, N, K, 0);
    });

    cudaMemcpy(C_f, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost);

    // Correctness check
    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_matmul_fp32(A_f, B_f, Ref, M, N, K);
        for (size_t i = 0; i < n_c; ++i) {
            double e = fabs((double)C_f[i] - (double)Ref[i]);
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
    printf("  \"M\": %d, \"N\": %d, \"K\": %d,\n", M, N, K);
    printf("  \"reps\": %d, \"warmup\": %d,\n", A.reps, A.warmup);
    printf("  \"mean_ms\": %.6f, \"cv_pct\": %.6f,\n", mean_ms, cv_pct);
    if (A.mode == Mode::Correctness) {
        printf("  \"max_abs_err\": %.6g, \"correctness_ok\": %s,\n", max_abs_err, correctness_ok ? "true" : "false");
    }
    printf("  \"manifest\": \"%s\"\n", manifest_path);
    printf("}\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A_f); free(B_f); free(C_f); free(Ref);
    free(h_A); free(h_B); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
