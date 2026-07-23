// Host runner for flash_attn_bwd_bf16_sm80 (LCD-glue header at
// ../flash_attn/lcd_glue/bwd_sm80.cuh). Suite-uniform CLI + schema-v2
// manifest via ../runner_common.h. Correctness: BH=1, Sq=16, Sk=32,
// D=16; FP64 reference; tolerance 5e-2.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../runner_common.h"

extern "C" void flash_attn_bwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V_T, const __nv_bfloat16 *d_dO,
    __nv_bfloat16 *d_dQ, __nv_bfloat16 *d_dK, __nv_bfloat16 *d_dV,
    int BH, int Sq, int Sk, float scale, cudaStream_t stream);

static const char *PROG = "flash_attn_bwd_bf16_sm80_run";
static const char *SLUG = "flash_attn_bwd_bf16_sm80";

// FP64 reference attention backward: given Q, K, V, dO, compute
// dQ, dK, dV via the standard chain rule. O = softmax(Q@K^T*scale) @ V.
static void cpu_attn_bwd(const float *Q, const float *K, const float *V, const float *dO,
                         float *dQ, float *dK, float *dV,
                         int BH, int Sq, int Sk, int D, float scale)
{
    for (int b = 0; b < BH; ++b) {
        for (int sq = 0; sq < Sq; ++sq) {
            // Forward (recompute scores, P, then dP/dS).
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
            double P[1024];
            for (int sk = 0; sk < Sk; ++sk) {
                P[sk] = exp(scores[sk] - m);
                sum_exp += P[sk];
            }
            for (int sk = 0; sk < Sk; ++sk) P[sk] /= sum_exp;

            // dP[sk] = sum_d dO[sq, d] * V[sk, d].
            double dP[1024];
            for (int sk = 0; sk < Sk; ++sk) {
                double dp = 0.0;
                for (int d = 0; d < D; ++d) {
                    dp += (double)dO[(b*Sq + sq) * D + d] * (double)V[(b*Sk + sk) * D + d];
                }
                dP[sk] = dp;
            }

            // row_dot = sum_sk dP[sk] * P[sk].
            double row_dot = 0.0;
            for (int sk = 0; sk < Sk; ++sk) row_dot += dP[sk] * P[sk];

            // dS[sk] = (dP[sk] - row_dot) * P[sk] * scale.
            double dS[1024];
            for (int sk = 0; sk < Sk; ++sk) dS[sk] = (dP[sk] - row_dot) * P[sk] * (double)scale;

            // dQ[sq, d] += sum_sk dS[sk] * K[sk, d].
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (int sk = 0; sk < Sk; ++sk) acc += dS[sk] * (double)K[(b*Sk + sk) * D + d];
                dQ[(b*Sq + sq) * D + d] = (float)acc;
            }
            // dV[sk, d] += P[sk] * dO[sq, d]; dK[sk, d] += dS[sk] * Q[sq, d].
            for (int sk = 0; sk < Sk; ++sk) {
                for (int d = 0; d < D; ++d) {
                    dV[(b*Sk + sk) * D + d] += (float)(P[sk] * (double)dO[(b*Sq + sq) * D + d]);
                    dK[(b*Sk + sk) * D + d] += (float)(dS[sk] * (double)Q[(b*Sq + sq) * D + d]);
                }
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
        BH = 4; Sq = 16; Sk = 128;
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
    const size_t n_q  = (size_t)BH * Sq * D;
    const size_t n_kv = (size_t)BH * Sk * D;

    auto malloc_f = [](size_t n) { return (float *)malloc(n * sizeof(float)); };
    float *Q_f = malloc_f(n_q),  *K_f = malloc_f(n_kv), *V_f = malloc_f(n_kv), *dO_f = malloc_f(n_q);
    float *dQ_f = malloc_f(n_q), *dK_f = malloc_f(n_kv), *dV_f = malloc_f(n_kv);
    float *RefQ = (A.mode == Mode::Correctness) ? (float *)calloc(n_q,  sizeof(float)) : nullptr;
    float *RefK = (A.mode == Mode::Correctness) ? (float *)calloc(n_kv, sizeof(float)) : nullptr;
    float *RefV = (A.mode == Mode::Correctness) ? (float *)calloc(n_kv, sizeof(float)) : nullptr;
    seed_tensor(Q_f,  n_q,  A.seed,     -0.5, 0.5);
    seed_tensor(K_f,  n_kv, A.seed + 1, -0.5, 0.5);
    seed_tensor(V_f,  n_kv, A.seed + 2, -0.5, 0.5);
    seed_tensor(dO_f, n_q,  A.seed + 3, -0.5, 0.5);

    // V_T pre-transposed.
    float *V_T_f = (float *)malloc(n_kv * sizeof(float));
    for (int b = 0; b < BH; ++b)
        for (int s = 0; s < Sk; ++s)
            for (int d = 0; d < D; ++d)
                V_T_f[(b * D + d) * Sk + s] = V_f[(b * Sk + s) * D + d];

    __nv_bfloat16 *h_Q   = (__nv_bfloat16 *)malloc(n_q  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_K   = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_V_T = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_dO  = (__nv_bfloat16 *)malloc(n_q  * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_q;  ++i) { h_Q[i]  = __float2bfloat16(Q_f[i]);  h_dO[i] = __float2bfloat16(dO_f[i]); }
    for (size_t i = 0; i < n_kv; ++i) { h_K[i]  = __float2bfloat16(K_f[i]);  h_V_T[i] = __float2bfloat16(V_T_f[i]); }

    __nv_bfloat16 *d_Q, *d_K, *d_V_T, *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q,   n_q  * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K,   n_kv * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V_T, n_kv * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dO,  n_q  * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dQ,  n_q  * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dK,  n_kv * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dV,  n_kv * sizeof(__nv_bfloat16));
    cudaMemcpy(d_Q,   h_Q,   n_q  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,   h_K,   n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_T, h_V_T, n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO,  h_dO,  n_q  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    auto zero_grads = [&]() {
        cudaMemset(d_dQ, 0, n_q  * sizeof(__nv_bfloat16));
        cudaMemset(d_dK, 0, n_kv * sizeof(__nv_bfloat16));
        cudaMemset(d_dV, 0, n_kv * sizeof(__nv_bfloat16));
    };
    for (int w = 0; w < A.warmup; ++w) {
        zero_grads();
        flash_attn_bwd_bf16_sm80_launch(d_Q, d_K, d_V_T, d_dO, d_dQ, d_dK, d_dV, BH, Sq, Sk, scale, 0);
    }
    cudaDeviceSynchronize();

    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    // FA-bwd accumulates dQ/dK/dV in place; reset BEFORE each timed
    // launch so the final launch's result lands in the device
    // buffers (the original loop had `zero_grads(); launch();` per
    // rep — preserved here by ordering inside the lambda).
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        zero_grads();
        flash_attn_bwd_bf16_sm80_launch(d_Q, d_K, d_V_T, d_dO, d_dQ, d_dK, d_dV, BH, Sq, Sk, scale, 0);
    });

    __nv_bfloat16 *h_dQ = (__nv_bfloat16 *)malloc(n_q  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_dK = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_dV = (__nv_bfloat16 *)malloc(n_kv * sizeof(__nv_bfloat16));
    cudaMemcpy(h_dQ, d_dQ, n_q  * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK, d_dK, n_kv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV, d_dV, n_kv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_q;  ++i) dQ_f[i] = __bfloat162float(h_dQ[i]);
    for (size_t i = 0; i < n_kv; ++i) { dK_f[i] = __bfloat162float(h_dK[i]); dV_f[i] = __bfloat162float(h_dV[i]); }

    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_attn_bwd(Q_f, K_f, V_f, dO_f, RefQ, RefK, RefV, BH, Sq, Sk, D, scale);
        for (size_t i = 0; i < n_q;  ++i) { double e = fabs((double)dQ_f[i] - (double)RefQ[i]); if (e > max_abs_err) max_abs_err = e; }
        for (size_t i = 0; i < n_kv; ++i) { double e = fabs((double)dK_f[i] - (double)RefK[i]); if (e > max_abs_err) max_abs_err = e; }
        for (size_t i = 0; i < n_kv; ++i) { double e = fabs((double)dV_f[i] - (double)RefV[i]); if (e > max_abs_err) max_abs_err = e; }
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

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V_T); cudaFree(d_dO);
    cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
    void *to_free[] = {Q_f, K_f, V_f, V_T_f, dO_f, dQ_f, dK_f, dV_f,
                       RefQ, RefK, RefV, h_Q, h_K, h_V_T, h_dO, h_dQ, h_dK, h_dV, samples_ms};
    for (void *p : to_free) free(p);
    return correctness_ok ? 0 : 1;
}
