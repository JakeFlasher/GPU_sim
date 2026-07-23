// Host runner for flash_attn_fwd_bf16_sm80, an LCD-portable
// FlashAttention v2 forward kernel using the LCD-glue header at
// `../flash_attn/lcd_glue/fwd_sm80.cuh` (verbatim upstream Tri Dao
// FA v2 source lives alongside under `../flash_attn/upstream/`;
// see `../flash_attn/{VERSION, NOTES.md}`).
//
// Implements the suite-uniform CLI surface and per-row schema-v2
// measurement-manifest emission via `../runner_common.h`.
//
// Correctness: BH=1, Sq=16, Sk=32, D=16. Sk/BLOCK_K = 2 exercises
// the multi-K-block online-softmax accumulator update path.
//   FP64 attention reference; tolerance 5e-2 (BF16 quantization
//   through 4 BF16 MMAs + the FP32->BF16 round-trip on P).
// Timing: BH=4, Sq=64, Sk=128, D=16.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../runner_common.h"

extern "C" void flash_attn_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V_T, __nv_bfloat16 *d_O,
    int BH, int Sq, int Sk, float scale, cudaStream_t stream);

static const char *PROG = "flash_attn_fwd_bf16_sm80_run";
static const char *SLUG = "flash_attn_fwd_bf16_sm80";

// FP64 reference attention forward: O = softmax(Q @ K^T / sqrt(D)) @ V.
static void cpu_attn_fwd(const float *Q, const float *K, const float *V,
                         float *O, int BH, int Sq, int Sk, int D, float scale)
{
    for (int b = 0; b < BH; ++b) {
        for (int sq = 0; sq < Sq; ++sq) {
            double scores[1024];
            double m = -INFINITY;
            for (int sk = 0; sk < Sk; ++sk) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += (double)Q[(b*Sq + sq) * D + d] * (double)K[(b*Sk + sk) * D + d];
                }
                scores[sk] = dot * (double)scale;
                if (scores[sk] > m) m = scores[sk];
            }
            double sum_exp = 0.0;
            for (int sk = 0; sk < Sk; ++sk) {
                scores[sk] = exp(scores[sk] - m);
                sum_exp += scores[sk];
            }
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (int sk = 0; sk < Sk; ++sk) {
                    acc += (scores[sk] / sum_exp) * (double)V[(b*Sk + sk) * D + d];
                }
                O[(b*Sq + sq) * D + d] = (float)acc;
            }
        }
    }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    require_accepted_overrides(A, PROG, {.flash_attn_bhsqsk = true});

    // CLI validation runs BEFORE acquire_device so invalid inputs
    // exit with the typed validation rc=2 regardless of device
    // availability. acquire_device's cudaGetDevice rc=1 must not
    // mask shape_override_* validation failures under
    // CUDA_VISIBLE_DEVICES=-1 or other no-device environments.
    int BH, Sq, Sk;
    constexpr int D = 16;
    const char *shape_point;
    bool overrides_set = (A.has_override_bh || A.has_override_sq
                       || A.has_override_sk || A.has_override_d);
    if (A.mode == Mode::Correctness) {
        if (overrides_set) {
            fprintf(stderr, "%s: shape_override_in_correctness_mode_rejected\n", PROG);
            return 2;
        }
        BH = 1; Sq = 16; Sk = 32;  shape_point = "correctness_00";
    } else {
        BH = 4; Sq = 64; Sk = 128;
        shape_point = "timing_00";
        if (overrides_set) {
            int o_d  = A.has_override_d  ? A.override_d  : D;
            int o_bh = A.has_override_bh ? A.override_bh : BH;
            int o_sq = A.has_override_sq ? A.override_sq : Sq;
            int o_sk = A.has_override_sk ? A.override_sk : Sk;
            if (o_d != D) {
                fprintf(stderr, "%s: shape_override_d_must_equal_%d (got %d); the LCD glue kernel pins D_TILE=%d\n",
                        PROG, D, o_d, D);
                return 2;
            }
            if (o_bh <= 0 || o_sq <= 0 || o_sk <= 0) {
                fprintf(stderr, "%s: shape_override_positive_dimensions_required (BH=%d Sq=%d Sk=%d)\n",
                        PROG, o_bh, o_sq, o_sk);
                return 2;
            }
            if ((o_sq % 16) != 0 || (o_sk % 16) != 0) {
                fprintf(stderr, "%s: shape_override_sq_sk_must_be_multiple_of_16 (Sq=%d Sk=%d)\n",
                        PROG, o_sq, o_sk);
                return 2;
            }
            BH = o_bh; Sq = o_sq; Sk = o_sk;
        }
    }
    if (A.shape_point != nullptr) shape_point = A.shape_point;

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    const float scale = 1.0f / sqrtf((float)D);
    const size_t n_q = (size_t)BH * Sq * D;
    const size_t n_kv = (size_t)BH * Sk * D;
    const size_t n_o = (size_t)BH * Sq * D;

    float *Q_f = (float *)malloc(n_q * sizeof(float));
    float *K_f = (float *)malloc(n_kv * sizeof(float));
    float *V_f = (float *)malloc(n_kv * sizeof(float));
    float *O_f = (float *)malloc(n_o * sizeof(float));
    float *Ref = (A.mode == Mode::Correctness) ? (float *)malloc(n_o * sizeof(float)) : nullptr;
    seed_tensor(Q_f, n_q,  A.seed,     -0.5, 0.5);
    seed_tensor(K_f, n_kv, A.seed + 1, -0.5, 0.5);
    seed_tensor(V_f, n_kv, A.seed + 2, -0.5, 0.5);

    // Pre-transpose V: [BH, Sk, D] row-major -> V_T [BH, D, Sk] row-major,
    // so the kernel's MMA2 .col B operand reads V_T as contiguous BF16 pairs.
    float *V_T_f = (float *)malloc(n_kv * sizeof(float));
    for (int b = 0; b < BH; ++b)
        for (int s = 0; s < Sk; ++s)
            for (int d = 0; d < D; ++d)
                V_T_f[(b * D + d) * Sk + s] = V_f[(b * Sk + s) * D + d];

    __nv_bfloat16 *h_Q   = (__nv_bfloat16 *)malloc(n_q * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_K   = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_V_T = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_q;  ++i) h_Q[i]   = __float2bfloat16(Q_f[i]);
    for (size_t i = 0; i < n_kv; ++i) h_K[i]   = __float2bfloat16(K_f[i]);
    for (size_t i = 0; i < n_kv; ++i) h_V_T[i] = __float2bfloat16(V_T_f[i]);

    __nv_bfloat16 *d_Q, *d_K, *d_V_T, *d_O;
    cudaMalloc(&d_Q,   n_q  * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K,   n_kv * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V_T, n_kv * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O,   n_o  * sizeof(__nv_bfloat16));
    cudaMemcpy(d_Q,   h_Q,   n_q  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,   h_K,   n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_T, h_V_T, n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int w = 0; w < A.warmup; ++w) {
        flash_attn_fwd_bf16_sm80_launch(d_Q, d_K, d_V_T, d_O, BH, Sq, Sk, scale, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        flash_attn_fwd_bf16_sm80_launch(d_Q, d_K, d_V_T, d_O, BH, Sq, Sk, scale, 0);
    });

    __nv_bfloat16 *h_O = (__nv_bfloat16 *)malloc(n_o * sizeof(__nv_bfloat16));
    cudaMemcpy(h_O, d_O, n_o * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_o; ++i) O_f[i] = __bfloat162float(h_O[i]);

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_attn_fwd(Q_f, K_f, V_f, Ref, BH, Sq, Sk, D, scale);
        for (size_t i = 0; i < n_o; ++i) {
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
    printf("  \"BH\": %d, \"Sq\": %d, \"Sk\": %d, \"D\": %d,\n", BH, Sq, Sk, D);
    printf("  \"reps\": %d, \"warmup\": %d,\n", A.reps, A.warmup);
    printf("  \"mean_ms\": %.6f, \"cv_pct\": %.6f,\n", mean_ms, cv_pct);
    if (A.mode == Mode::Correctness) {
        printf("  \"max_abs_err\": %.6g, \"correctness_ok\": %s,\n", max_abs_err, correctness_ok ? "true" : "false");
    }
    printf("  \"manifest\": \"%s\"\n", manifest_path);
    printf("}\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V_T); cudaFree(d_O);
    free(Q_f); free(K_f); free(V_f); free(V_T_f); free(O_f); free(Ref);
    free(h_Q); free(h_K); free(h_V_T); free(h_O); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
