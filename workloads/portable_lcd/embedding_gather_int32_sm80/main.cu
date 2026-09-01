// Host runner for embedding_gather_int32_sm80, an LCD-portable
// integer-indexed BF16 embedding lookup.
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`. Correctness
// mode runs (B=2, S=8, V=64, D=128) and compares against an
// in-process FP32-equivalent reference (reference is exact equality
// since BF16 lookup has no numerical noise; tol kept at 1e-6).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include "../runner_common.h"

extern "C" void embedding_gather_int32_sm80_launch(
    const int32_t *d_indices, const __nv_bfloat16 *d_table,
    __nv_bfloat16 *d_out, int B, int S, int D, cudaStream_t stream);

static const char *PROG = "embedding_gather_int32_sm80_run";
static const char *SLUG = "embedding_gather_int32_sm80";

// Slug-specific helper: integer indices into a vocab. Uses the same
// scrambled-LCG seed as runner_common::seed_tensor for determinism.
static void seed_indices(int32_t *flat, size_t n, uint32_t seed, int32_t vocab_size) {
    uint32_t st = seed * 2654435761u + 7u;
    for (size_t i = 0; i < n; ++i) {
        flat[i] = (int32_t)(runner_common::next_u32(&st) % (uint32_t)vocab_size);
    }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    require_accepted_overrides(A, PROG, {.flash_attn_bhsqsk = true});

    int B, S, V, D;
    const char *shape_point;
    bool overrides_set = A.has_override_bh || A.has_override_sq
                      || A.has_override_sk || A.has_override_d;
    if (A.mode == Mode::Correctness) {
        if (overrides_set) {
            fprintf(stderr, "%s: shape_override_in_correctness_mode_rejected\n", PROG);
            return 2;
        }
        B = 2; S = 8; V = 64; D = 128; shape_point = "correctness_00";
    } else {
        B = 4; S = 128; V = 4096; D = 2048; shape_point = "timing_00";
        if (overrides_set) {
            B = A.has_override_bh ? A.override_bh : B;
            S = A.has_override_sq ? A.override_sq : S;
            V = A.has_override_sk ? A.override_sk : V;
            D = A.has_override_d ? A.override_d : D;
            if (B <= 0 || S <= 0 || V <= 0 || D <= 0) {
                fprintf(stderr, "%s: shape_override_positive_dimensions_required (B=%d S=%d V=%d D=%d)\n",
                        PROG, B, S, V, D);
                return 2;
            }
            int64_t pairs = (int64_t)B * S;
            if (pairs > INT_MAX || (int64_t)V * D > INT_MAX || pairs * D > INT_MAX) {
                fprintf(stderr, "%s: shape_override_element_count_exceeds_int32\n", PROG);
                return 2;
            }
            shape_point = "timing_override";
        }
    }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    const size_t n_idx   = (size_t)B * S;
    const size_t n_table = (size_t)V * D;
    const size_t n_out   = (size_t)B * S * D;

    int32_t *Idx_h = (int32_t *)malloc(n_idx   * sizeof(int32_t));
    float   *T_f   = (float   *)malloc(n_table * sizeof(float));
    float   *O_f   = (float   *)malloc(n_out   * sizeof(float));
    float   *Ref   = (A.mode == Mode::Correctness) ? (float *)malloc(n_out * sizeof(float)) : nullptr;
    seed_indices(Idx_h, n_idx, A.seed,     V);
    seed_tensor(T_f,    n_table, A.seed + 1, -1.0, 1.0);

    __nv_bfloat16 *h_T = (__nv_bfloat16 *)malloc(n_table * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_table; ++i) h_T[i] = __float2bfloat16(T_f[i]);

    int32_t       *d_Idx = nullptr;
    __nv_bfloat16 *d_T   = nullptr;
    __nv_bfloat16 *d_O   = nullptr;
    cudaMalloc(&d_Idx, n_idx   * sizeof(int32_t));
    cudaMalloc(&d_T,   n_table * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O,   n_out   * sizeof(__nv_bfloat16));
    cudaMemcpy(d_Idx, Idx_h, n_idx   * sizeof(int32_t),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_T,   h_T,   n_table * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        embedding_gather_int32_sm80_launch(d_Idx, d_T, d_O, B, S, D, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        embedding_gather_int32_sm80_launch(d_Idx, d_T, d_O, B, S, D, 0);
    });

    __nv_bfloat16 *h_O = (__nv_bfloat16 *)malloc(n_out * sizeof(__nv_bfloat16));
    cudaMemcpy(h_O, d_O, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_out; ++i) O_f[i] = __bfloat162float(h_O[i]);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        // Reference: out[b,s,d] = table[indices[b*S+s]][d] in BF16.
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                int32_t idx = Idx_h[b * S + s];
                for (int d = 0; d < D; ++d) {
                    float gold_bf16 = __bfloat162float(h_T[(size_t)idx * D + d]);
                    Ref[(size_t)((b * S + s) * D + d)] = gold_bf16;
                }
            }
        }
        for (size_t i = 0; i < n_out; ++i) {
            double e = fabs((double)O_f[i] - (double)Ref[i]);
            if (e > max_abs_err) max_abs_err = e;
        }
        // Pure gather: BF16 → BF16, no arithmetic, so exact equality
        // is expected. Allow a small tolerance for safety.
        const double tol = 1e-6;
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
    printf("  \"B\": %d, \"S\": %d, \"V\": %d, \"D\": %d,\n", B, S, V, D);
    printf("  \"reps\": %d, \"warmup\": %d,\n", A.reps, A.warmup);
    printf("  \"mean_ms\": %.6f, \"cv_pct\": %.6f,\n", mean_ms, cv_pct);
    if (A.mode == Mode::Correctness) {
        printf("  \"max_abs_err\": %.6g, \"correctness_ok\": %s,\n", max_abs_err, correctness_ok ? "true" : "false");
    }
    printf("  \"manifest\": \"%s\"\n", manifest_path);
    printf("}\n");

    cudaFree(d_Idx); cudaFree(d_T); cudaFree(d_O);
    free(Idx_h); free(T_f); free(O_f); free(Ref);
    free(h_T); free(h_O); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
