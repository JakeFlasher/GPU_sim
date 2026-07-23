// Host runner for cross_entropy_fwd_fp32_sm80, an LCD-portable
// FP32 per-sample cross-entropy kernel.
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`.
// Correctness mode runs B=8, V=128 and compares against an
// in-process FP32 reference; tolerance is 1e-4 (FP32 throughout, so
// the dominant noise is just the reduction-order difference between
// the GPU tree-reduce and the host scalar accumulation).

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" void cross_entropy_fwd_fp32_sm80_launch(
    const float *d_logits, const int32_t *d_labels,
    float *d_loss, int B, int V, cudaStream_t stream);

static const char *PROG = "cross_entropy_fwd_fp32_sm80_run";
static const char *SLUG = "cross_entropy_fwd_fp32_sm80";

// Slug-specific helper: integer label indices into a vocab.
static void seed_indices(int32_t *flat, size_t n, uint32_t seed, int32_t vocab_size) {
    uint32_t st = seed * 2654435761u + 7u;
    for (size_t i = 0; i < n; ++i) {
        flat[i] = (int32_t)(runner_common::next_u32(&st) % (uint32_t)vocab_size);
    }
}

static void cpu_cross_entropy_fwd(const float *logits, const int32_t *labels,
                                  float *loss, int B, int V)
{
    for (int b = 0; b < B; ++b) {
        double m = -DBL_MAX;
        for (int v = 0; v < V; ++v) {
            double x = (double)logits[b * V + v];
            if (x > m) m = x;
        }
        double z = 0.0;
        for (int v = 0; v < V; ++v) {
            z += exp((double)logits[b * V + v] - m);
        }
        double lse = m + log(z);
        loss[b] = (float)(lse - (double)logits[b * V + labels[b]]);
    }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    require_accepted_overrides(A, PROG, {});

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    int B, V;
    const char *shape_point;
    if (A.mode == Mode::Correctness) { B = 8;  V = 128;  shape_point = "correctness_00"; }
    else                             { B = 64; V = 4096; shape_point = "timing_00"; }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    const size_t n_logits = (size_t)B * V;
    const size_t n_labels = (size_t)B;
    const size_t n_loss   = (size_t)B;

    float   *L_h   = (float   *)malloc(n_logits * sizeof(float));
    int32_t *Lbl_h = (int32_t *)malloc(n_labels * sizeof(int32_t));
    float   *O_h   = (float   *)malloc(n_loss   * sizeof(float));
    float   *Ref   = (A.mode == Mode::Correctness) ? (float *)malloc(n_loss * sizeof(float)) : nullptr;
    seed_tensor(L_h, n_logits, A.seed,     -2.0, 2.0);
    seed_indices(Lbl_h, n_labels, A.seed + 1, V);

    float   *d_L   = nullptr;
    int32_t *d_Lbl = nullptr;
    float   *d_O   = nullptr;
    cudaMalloc(&d_L,   n_logits * sizeof(float));
    cudaMalloc(&d_Lbl, n_labels * sizeof(int32_t));
    cudaMalloc(&d_O,   n_loss   * sizeof(float));
    cudaMemcpy(d_L,   L_h,   n_logits * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lbl, Lbl_h, n_labels * sizeof(int32_t), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        cross_entropy_fwd_fp32_sm80_launch(d_L, d_Lbl, d_O, B, V, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        cross_entropy_fwd_fp32_sm80_launch(d_L, d_Lbl, d_O, B, V, 0);
    });

    cudaMemcpy(O_h, d_O, n_loss * sizeof(float), cudaMemcpyDeviceToHost);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_cross_entropy_fwd(L_h, Lbl_h, Ref, B, V);
        for (size_t i = 0; i < n_loss; ++i) {
            double e = fabs((double)O_h[i] - (double)Ref[i]);
            if (e > max_abs_err) max_abs_err = e;
        }
        const double tol = 1e-4;
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
    printf("  \"B\": %d, \"V\": %d,\n", B, V);
    printf("  \"reps\": %d, \"warmup\": %d,\n", A.reps, A.warmup);
    printf("  \"mean_ms\": %.6f, \"cv_pct\": %.6f,\n", mean_ms, cv_pct);
    if (A.mode == Mode::Correctness) {
        printf("  \"max_abs_err\": %.6g, \"correctness_ok\": %s,\n", max_abs_err, correctness_ok ? "true" : "false");
    }
    printf("  \"manifest\": \"%s\"\n", manifest_path);
    printf("}\n");

    cudaFree(d_L); cudaFree(d_Lbl); cudaFree(d_O);
    free(L_h); free(Lbl_h); free(O_h); free(Ref);
    free(samples_ms);

    return correctness_ok ? 0 : 1;
}
