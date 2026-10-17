// Host runner for gemm_imma_int8_sm80, an LCD-portable INT8/INT32
// IMMA GEMM (sister of gemm_hmma_bf16_sm80 / gemm_hmma_fp16_sm80;
// extends suite tensor-core opcode coverage to the integer pipeline).
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`.
//
// Correctness mode:
//   M=32, N=16, K=64 INT8 inputs seeded by --seed; in-process INT32
//   reference matmul; expected `max_abs_err = 0` (pure integer
//   arithmetic; bit-exact). Inputs bounded to [-8, 7] to keep the
//   accumulator (sum of 64 INT8*INT8 products) well within INT32.
//
// Timing mode:
//   M=256, N=256, K=256 INT8 inputs; cudaEventRecord-bracketed warmup
//   and repeat loops; emits cv_pct over the repeat samples.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" void gemm_imma_int8_sm80_launch(
    const int8_t *d_A, const int8_t *d_B,
    int32_t *d_C, int M, int N, int K, cudaStream_t stream);

static const char *PROG = "gemm_imma_int8_sm80_run";
static const char *SLUG = "gemm_imma_int8_sm80";

// Slug-specific helper: seed INT8 inputs in [lo, hi] (inclusive).
static void seed_tensor_int8(int8_t *flat, size_t n, uint32_t seed, int lo, int hi) {
    uint32_t st = seed * 2654435761u + 1u;
    const int range = hi - lo + 1;
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = runner_common::next_u32(&st);
        flat[i] = (int8_t)(lo + (int)(r % (uint32_t)range));
    }
}

static void cpu_matmul_int32(const int8_t *A_rm, const int8_t *B_rm,
                             int32_t *C_rm, int M, int N, int K)
{
    // A_rm: [M, K] row-major. B_rm: [K, N] row-major. C_rm: [M, N] row-major.
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int32_t)A_rm[m * K + k] * (int32_t)B_rm[k * N + n];
            C_rm[m * N + n] = acc;
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
        // Tile-alignment validation: the INT8 IMMA kernel launches
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

    int8_t  *A_i  = (int8_t *)malloc(n_a * sizeof(int8_t));
    int8_t  *B_i  = (int8_t *)malloc(n_b * sizeof(int8_t));
    int32_t *C_i  = (int32_t *)malloc(n_c * sizeof(int32_t));
    int32_t *Ref  = (A.mode == Mode::Correctness) ? (int32_t *)malloc(n_c * sizeof(int32_t)) : nullptr;
    seed_tensor_int8(A_i, n_a, A.seed,     -8, 7);
    seed_tensor_int8(B_i, n_b, A.seed + 1, -8, 7);

    // Transpose B from [K, N] row-major to [N, K] col-major (== [N, K] row-major
    // of the transposed tensor) for the MMA's `.col` layout.
    int8_t *h_B = (int8_t *)malloc(n_b * sizeof(int8_t));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            h_B[n * K + k] = B_i[k * N + n];

    int8_t  *d_A = nullptr, *d_B = nullptr;
    int32_t *d_C = nullptr;
    cudaMalloc(&d_A, n_a * sizeof(int8_t));
    cudaMalloc(&d_B, n_b * sizeof(int8_t));
    cudaMalloc(&d_C, n_c * sizeof(int32_t));
    cudaMemcpy(d_A, A_i, n_a * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n_b * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Warmup
    for (int w = 0; w < A.warmup; ++w) {
        gemm_imma_int8_sm80_launch(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    // Repeat loop with timing
    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        gemm_imma_int8_sm80_launch(d_A, d_B, d_C, M, N, K, 0);
    });

    cudaMemcpy(C_i, d_C, n_c * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Correctness check (bit-exact: pure integer arithmetic).
    bool correctness_ok = true;
    int32_t max_abs_err = 0;
    if (A.mode == Mode::Correctness) {
        cpu_matmul_int32(A_i, B_i, Ref, M, N, K);
        for (size_t i = 0; i < n_c; ++i) {
            int32_t e = C_i[i] - Ref[i];
            if (e < 0) e = -e;
            if (e > max_abs_err) max_abs_err = e;
        }
        correctness_ok = (max_abs_err == 0);
        if (!correctness_ok) {
            fprintf(stderr, "%s: correctness_failed: max_abs_err=%d expected=0\n", PROG, max_abs_err);
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
        printf("  \"max_abs_err\": %d, \"correctness_ok\": %s,\n", max_abs_err, correctness_ok ? "true" : "false");
    }
    printf("  \"manifest\": \"%s\"\n", manifest_path);
    printf("}\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A_i); free(B_i); free(C_i); free(Ref);
    free(h_B); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
