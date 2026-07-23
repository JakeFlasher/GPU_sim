// Host runner for conv2d_implicit_gemm_bf16_sm80, an LCD-portable
// CUTLASS Conv2dFprop sm_80 implicit-GEMM convolution.
//
// Suite-uniform CLI surface via runner_common.h:
//   --correctness_shape | --timing_shape
//   --seed N         | --warmup N        | --reps N
//   --output_dir P   | --allow-any-device
//
// Correctness shape:
//   N=4, H=W=8, C=16, K=16, R=S=3, pad=1, stride=1, dilation=1
//   -> P=Q=8 output spatial. Channel counts are multiples of 8 to
//   satisfy the BF16 alignment expected by the CUTLASS Sm80
//   tensor-op iterator (ElementsPerAccess=8).
//
// Timing shape:
//   N=8, H=W=16, C=32, K=32, R=S=3, pad=1, stride=1, dilation=1.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../runner_common.h"

extern "C" int conv2d_implicit_gemm_bf16_sm80_launch(
    const __nv_bfloat16 *d_act,
    const __nv_bfloat16 *d_filter,
    __nv_bfloat16 *d_out,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int P, int Q,
    cudaStream_t stream);

static const char *PROG = "conv2d_implicit_gemm_bf16_sm80_run";
static const char *SLUG = "conv2d_implicit_gemm_bf16_sm80";

// CPU FP32 reference: NHWC activations, KRSC filters, NHWC output,
// cross-correlation (= CUTLASS Conv2dFprop semantics with
// Mode::kCrossCorrelation), padding pad_h/pad_w on each side,
// stride stride_h/stride_w, dilation dilation_h/dilation_w.
static void cpu_conv2d_fprop_fp32(
    const float *act, const float *filt, float *out,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int P, int Q)
{
    for (int n = 0; n < N; ++n)
    for (int p = 0; p < P; ++p)
    for (int q = 0; q < Q; ++q)
    for (int k = 0; k < K; ++k) {
        double acc = 0.0;
        for (int r = 0; r < R; ++r)
        for (int s = 0; s < S; ++s)
        for (int c = 0; c < C; ++c) {
            int h = p * stride_h - pad_h + r * dilation_h;
            int w = q * stride_w - pad_w + s * dilation_w;
            if (h < 0 || h >= H || w < 0 || w >= W) continue;
            double a = (double)act[((n * H + h) * W + w) * C + c];
            double f = (double)filt[((k * R + r) * S + s) * C + c];
            acc += a * f;
        }
        out[((n * P + p) * Q + q) * K + k] = (float)acc;
    }
}

int main(int argc, char **argv) {
    using namespace runner_common;
    Args A = parse_args(argc, argv, PROG);
    require_accepted_overrides(A, PROG, {});

    cudaDeviceProp dp;
    int rc = acquire_device(PROG, A, dp);
    if (rc) return rc;

    int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w;
    const char *shape_point;
    if (A.mode == Mode::Correctness) {
        N = 4; H = 8;  W = 8;  C = 16; K = 16; R = 3; S = 3;
        pad_h = pad_w = 1; stride_h = stride_w = 1; dil_h = dil_w = 1;
        shape_point = "correctness_00";
    } else {
        N = 8; H = 16; W = 16; C = 32; K = 32; R = 3; S = 3;
        pad_h = pad_w = 1; stride_h = stride_w = 1; dil_h = dil_w = 1;
        shape_point = "timing_00";
    }
    if (A.shape_point != nullptr) shape_point = A.shape_point;
    int P = (H + 2 * pad_h - dil_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) / stride_w + 1;

    const size_t n_act  = (size_t)N * H * W * C;
    const size_t n_filt = (size_t)K * R * S * C;
    const size_t n_out  = (size_t)N * P * Q * K;

    float *act_f  = (float *)malloc(n_act  * sizeof(float));
    float *filt_f = (float *)malloc(n_filt * sizeof(float));
    float *out_f  = (float *)malloc(n_out  * sizeof(float));
    float *Ref    = (A.mode == Mode::Correctness)
                        ? (float *)malloc(n_out * sizeof(float)) : nullptr;
    seed_tensor(act_f,  n_act,  A.seed,     -0.5, 0.5);
    seed_tensor(filt_f, n_filt, A.seed + 1, -0.5, 0.5);

    __nv_bfloat16 *h_act  = (__nv_bfloat16 *)malloc(n_act  * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_filt = (__nv_bfloat16 *)malloc(n_filt * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_out  = (__nv_bfloat16 *)malloc(n_out  * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_act;  ++i) h_act[i]  = __float2bfloat16(act_f[i]);
    for (size_t i = 0; i < n_filt; ++i) h_filt[i] = __float2bfloat16(filt_f[i]);

    __nv_bfloat16 *d_act = nullptr, *d_filt = nullptr, *d_out = nullptr;
    cudaMalloc(&d_act,  n_act  * sizeof(__nv_bfloat16));
    cudaMalloc(&d_filt, n_filt * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out,  n_out  * sizeof(__nv_bfloat16));
    cudaMemcpy(d_act,  h_act,  n_act  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filt, h_filt, n_filt * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Warmup
    for (int w = 0; w < A.warmup; ++w) {
        int rc = conv2d_implicit_gemm_bf16_sm80_launch(
            d_act, d_filt, d_out,
            N, H, W, C, K, R, S,
            pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, P, Q,
            0);
        if (rc) {
            fprintf(stderr, "%s: launch_failed_during_warmup rc=%d\n", PROG, rc);
            return rc;
        }
    }
    cudaDeviceSynchronize();

    // Repeat loop with cudaEvent timing
    double *samples_ms = (double *)malloc((size_t)A.reps * sizeof(double));
    int conv_launch_rc = 0;
    runner_common::run_timed_reps(A, samples_ms, [&]() {
        if (conv_launch_rc) return;
        conv_launch_rc = conv2d_implicit_gemm_bf16_sm80_launch(
            d_act, d_filt, d_out,
            N, H, W, C, K, R, S,
            pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, P, Q,
            0);
    });
    if (conv_launch_rc) {
        fprintf(stderr, "%s: launch_failed_during_rep rc=%d\n", PROG, conv_launch_rc);
        return conv_launch_rc;
    }

    cudaMemcpy(h_out, d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_out; ++i) out_f[i] = __bfloat162float(h_out[i]);

    // Correctness
    bool correctness_ok = true;
    double max_abs_err = 0.0;
    if (A.mode == Mode::Correctness) {
        cpu_conv2d_fprop_fp32(
            act_f, filt_f, Ref,
            N, H, W, C, K, R, S,
            pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, P, Q);
        for (size_t i = 0; i < n_out; ++i) {
            double e = fabs((double)out_f[i] - (double)Ref[i]);
            if (e > max_abs_err) max_abs_err = e;
        }
        // BF16 conv tolerance: each output is the sum of R*S*C BF16
        // multiply-accumulates accumulated in FP32 then narrowed to
        // BF16. R*S*C=144 for the correctness shape, so a 5e-2
        // tolerance follows the suite's BF16 conv band.
        const double tol = 5e-2;
        correctness_ok = (max_abs_err <= tol);
        if (!correctness_ok) {
            fprintf(stderr, "%s: correctness_failed: max_abs_err=%.6g tol=%.6g\n",
                    PROG, max_abs_err, tol);
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
    printf("  \"N\": %d, \"H\": %d, \"W\": %d, \"C\": %d, \"K\": %d, \"R\": %d, \"S\": %d,\n",
           N, H, W, C, K, R, S);
    printf("  \"P\": %d, \"Q\": %d,\n", P, Q);
    printf("  \"reps\": %d, \"warmup\": %d,\n", A.reps, A.warmup);
    printf("  \"mean_ms\": %.6f, \"cv_pct\": %.6f,\n", mean_ms, cv_pct);
    if (A.mode == Mode::Correctness) {
        printf("  \"max_abs_err\": %.6g, \"correctness_ok\": %s,\n",
               max_abs_err, correctness_ok ? "true" : "false");
    }
    printf("  \"manifest\": \"%s\"\n", manifest_path);
    printf("}\n");

    cudaFree(d_act); cudaFree(d_filt); cudaFree(d_out);
    free(act_f); free(filt_f); free(out_f); free(Ref);
    free(h_act); free(h_filt); free(h_out); free(samples_ms);

    return correctness_ok ? 0 : 1;
}
