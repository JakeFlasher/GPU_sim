// Host driver for wgmma_bf16_saturation. Hopper-only (sm_90a).
// Host casts seeded FP32 inputs to BF16 (B is transposed to col-
// major) and issues an end-to-end correctness run:
//   1. Memcpy H2D of BF16-quantized A and B into device buffers.
//   2. Launch the wgmma kernel; it issues
//      wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 (cubin:
//      HGMMA.64x64x16.F32.BF16) and writes the full 64×64 row-major
//      FP32 GEMM output to d_C per the m64n64 fragment layout
//      (kernel header + R23 sink-store→real-stores rewrite).
//   3. cudaMemcpy d_C → h_C for the JSON emit.
// Reference: references/mma_microbench.py::_fp32_matmul on the
// **un-quantized** seeded FP32 operands (the `wgmma_bf16_saturation`
// slug has no elif in build_reference's dispatch chain; it falls
// through to plain _fp32_matmul, like the other BF16/FP16/TF32 rows
// that don't go through kind::f8f6f4 quantization). The kernel sees
// host-cast BF16 inputs (via `__float2bfloat16`) while the reference
// computes against un-quantized FP32 inputs — the two diverge per
// BF16's 7-bit mantissa (BF16 has wide dynamic range from its 8-bit
// exponent but the 7-bit mantissa is NOT exact for arbitrary FP32
// values in the seeded `[-0.5, 0.5)` range; small rounding differences
// per element accumulate over the K-reduction). The (BF16, fma_chain,
// mma_microbench) tolerance triplet absorbs both this operand-
// quantization error AND the BF16*BF16→FP32 hardware-multiply
// rounding. R25 corrected the prior Round-23 prose that falsely
// claimed BF16 cast was exact in [-0.5, 0.5) (per Codex Round-23/24-
// review queued cleanup).
//
// Round-by-round: R0/1 designed the kernel as an instruction-path
// anchor + sink-store with the host computing a plain-FP32 reference
// and overwriting d_C as the "authoritative output." R23 brought
// this row in line with the Rounds 20+21+22 FP8 sibling fix sequence:
// removed host pre-computed FP32 matmul + removed explicit d_C
// overwrite (host-runner cleanup; mirror R20); replaced kernel
// sink-store with real m64n64 fragment writes (mirror R21); added
// fence.proxy.async.shared::cta between SMEM stores and wgmma issue
// (mirror R22). Per Codex Round-22-review: BF16 row is plan-
// derived local correctness work, not queued cleanup.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void wgmma_bf16_saturation_launch(const __nv_bfloat16 *d_A, const __nv_bfloat16 *d_B,
                                              float *d_C, int M, int N, int K, cudaStream_t stream);

static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}
static void fail(const char *msg, int rc) {
    fprintf(stderr, "wgmma_bf16_saturation_run: %s (rc=%d)\n", msg, rc);
    exit(1);
}

int main(int argc, char **argv) {
    int M = 64, N = 64, K = 16;  // matches shapes/wgmma_bf16_saturation.toml correctness_00 (R24 fix per Codex Round-23-review blocking finding)
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "wgmma_bf16_saturation_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n_a = (size_t)M * K, n_b = (size_t)K * N, n_c = (size_t)M * N;
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    __nv_bfloat16 *h_A = (__nv_bfloat16 *)malloc(n_a * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_B = (__nv_bfloat16 *)malloc(n_b * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n_a; ++i) h_A[i] = __float2bfloat16(A_f[i]);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            h_B[n * K + k] = __float2bfloat16(B_f[k * N + n]);

    __nv_bfloat16 *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A, n_a * sizeof(__nv_bfloat16)); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, n_b * sizeof(__nv_bfloat16)); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, n_c * sizeof(float));         if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, h_A, n_a * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B, n_b * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    // Launch the wgmma kernel. On Hopper sm_90a it computes the full
    // 64×64 m64n64k16 BF16 GEMM and writes d_C; the post-launch D2H
    // copy below makes the kernel's output the row's authoritative
    // result. (R23 mirror of R20 FP8 fix: removed explicit
    // `cudaMemcpy(d_C, h_C, ..., HostToDevice)` overwrite that would
    // destroy the kernel's output, plus removed the now-dead host-
    // FP32-matmul-into-h_C precomputation.)
    wgmma_bf16_saturation_launch(d_A, d_B, d_C, M, N, K, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"wgmma_bf16_saturation\",\n");
    printf("  \"shape_set\": \"correctness_shapes\",\n  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int m = 0; m < M; ++m) {
        if (m > 0) printf(",");
        printf("[");
        for (int n = 0; n < N; ++n) {
            if (n > 0) printf(",");
            printf("%.6g", h_C[m * N + n]);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A_f); free(B_f); free(h_A); free(h_B); free(h_C);
    return 0;
}
