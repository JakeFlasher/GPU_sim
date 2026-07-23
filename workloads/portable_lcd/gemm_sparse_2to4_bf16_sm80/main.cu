// Host runner for gemm_sparse_2to4_bf16_sm80, an LCD-portable
// BF16 GEMM with 2:4 structured sparsity on the A operand via
// the warp-level mma.sp.sync.aligned.m16n8k32 instruction.
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`.
//
// Correctness mode:
//   M=16, N=8, K=32 BF16 inputs seeded by --seed; A is generated
//   by sampling a dense matrix and zeroing every odd-position
//   element of every 4-K-block (kept indices = {0, 2}). The
//   compact A passed to the kernel contains only the kept
//   columns (K_compact = K/2 = 16). The CPU FP32 reference
//   matmul uses the original dense A_full (post-zeroing) so the
//   sparse kernel and reference produce mathematically identical
//   sums.
//   Per-element absolute tolerance 5e-2 (BF16 matmul through 32
//   K-elements with half retained -> 16 effective accumulations).
//
// Timing mode:
//   M=128, N=128, K=128 (K_compact=64); same sparse pattern.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" void gemm_sparse_2to4_bf16_sm80_launch(
    const __nv_bfloat16 *d_A_compact, const __nv_bfloat16 *d_B,
    float *d_C, int M, int N, int K, cudaStream_t stream);

static const char *PROG = "gemm_sparse_2to4_bf16_sm80_run";
static const char *SLUG = "gemm_sparse_2to4_bf16_sm80";

// FP32 reference matmul over a dense [M, K] x [K, N] = [M, N].
// A_full has the 2:4 sparsity already applied (odd K-positions
// of every 4-block are zero), so the same dense matmul produces
// the reference for the sparse kernel.
static void cpu_matmul_fp32(const float *A_full, const float *B,
                            float *C, int M, int N, int K)
{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A_full[m * K + k] * (double)B[k * N + n];
            C[m * N + n] = (float)acc;
        }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    require_accepted_overrides(A, PROG, {});

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    int M, N, K;
    const char *shape_point;
    if (A.mode == Mode::Correctness) { M = 16;  N = 8;   K = 32;   shape_point = "correctness_00"; }
    else                             { M = 128; N = 128; K = 128;  shape_point = "timing_00"; }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    // K must be a multiple of 4 (2:4 pattern) and a multiple of K_TILE=32.
    if (K % 32 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 32\n", PROG, K);
        return 2;
    }
    const int K_HALF = K / 2;

    const size_t n_a_full = (size_t)M * K;
    const size_t n_a_comp = (size_t)M * K_HALF;
    const size_t n_b      = (size_t)K * N;
    const size_t n_c      = (size_t)M * N;

    float *A_full = (float *)malloc(n_a_full * sizeof(float));
    float *B_f    = (float *)malloc(n_b * sizeof(float));
    float *B_col  = (float *)malloc(n_b * sizeof(float)); // [N, K] col-major (B is .col)
    float *C_f    = (float *)malloc(n_c * sizeof(float));
    float *Ref    = (A.mode == Mode::Correctness) ? (float *)malloc(n_c * sizeof(float)) : nullptr;
    seed_tensor(A_full, n_a_full, A.seed,     -0.5, 0.5);
    seed_tensor(B_f,    n_b,      A.seed + 1, -0.5, 0.5);

    // Apply 2:4 sparsity on A_full: keep cols 0,2 of every 4-K-block,
    // zero cols 1,3. Build A_compact by stripping the kept columns
    // into a row-major [M, K/2] array.
    float *A_compact = (float *)malloc(n_a_comp * sizeof(float));
    for (int m = 0; m < M; ++m) {
        int comp_idx = 0;
        for (int kb = 0; kb < K; kb += 4) {
            // Zero the discarded columns in the dense reference.
            A_full[m * K + kb + 1] = 0.0f;
            A_full[m * K + kb + 3] = 0.0f;
            // Pack the kept columns into the compact array.
            A_compact[m * K_HALF + comp_idx++] = A_full[m * K + kb + 0];
            A_compact[m * K_HALF + comp_idx++] = A_full[m * K + kb + 2];
        }
    }

    // Transpose B from [K, N] row-major to [N, K] col-major (so the
    // .col operand layout matches dense m16n8k32 B's expectation).
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            B_col[n * K + k] = B_f[k * N + n];

    __nv_bfloat16 *h_A = (__nv_bfloat16 *)malloc(n_a_comp * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_B = (__nv_bfloat16 *)malloc(n_b      * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_a_comp; ++i) h_A[i] = __float2bfloat16(A_compact[i]);
    for (size_t i = 0; i < n_b;      ++i) h_B[i] = __float2bfloat16(B_col[i]);

    __nv_bfloat16 *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc(&d_A, n_a_comp * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, n_b      * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, n_c      * sizeof(float));
    cudaMemcpy(d_A, h_A, n_a_comp * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_b      * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        gemm_sparse_2to4_bf16_sm80_launch(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        gemm_sparse_2to4_bf16_sm80_launch(d_A, d_B, d_C, M, N, K, 0);
    });

    cudaMemcpy(C_f, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_matmul_fp32(A_full, B_f, Ref, M, N, K);
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
    free(A_full); free(B_f); free(B_col); free(C_f); free(Ref);
    free(A_compact); free(h_A); free(h_B); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
