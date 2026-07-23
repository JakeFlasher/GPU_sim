// Host driver for wgmma_fp8_saturation. Hopper-only (sm_90a).
// Host casts seeded FP32 inputs to FP8 E4M3 (B is transposed to col-
// major) and issues an end-to-end correctness run:
//   1. Memcpy H2D of FP8-quantized A and B into device buffers.
//   2. Launch the wgmma kernel; it issues
//      wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 (cubin:
//      QGMMA.64x64x32.F32.E4M3.E4M3) and writes the full 64×64
//      row-major FP32 GEMM output to d_C per the m64n64 fragment
//      layout (kernel header + R21 drift-recovery rewrite).
//   3. cudaMemcpy d_C → h_C for the JSON emit.
// Reference: references/mma_microbench.py::_quantize_matrix_fp8_e4m3 + _fp32_matmul.
//   (R19 added the FP8 E4M3 dispatch so the reference round-trips
//    FP32 → FP8 E4M3 code → FP32 representable bit-exactly the same way
//    the host's `__nv_fp8_e4m3` cast does, before the FP32 matmul.)
// On Hopper hardware: kernel d_C output matches reference within the
// (FP8_E4M3, fma_chain, mma_microbench) tolerance triplet at
// rtol=atol=0.05, max_ulp=8.
//
// Round-by-round: R0/1 designed the kernel as an instruction-path
// anchor + sink-store, with the host computing a plain-FP32 reference
// and overwriting d_C as the "authoritative output." R19 added
// FP8 E4M3 reference dispatch (host quantizes ⇒ reference must too).
// R20 fixed this main.cu to remove the explicit d_C overwrite.
// R21 (drift-recovery) rewrote the kernel's sink-store as real
// per-fragment row-major stores so the row's correctness contract is
// finally end-to-end coherent: kernel computes ⇒ host emits kernel
// output ⇒ reference matches via the same FP8 round-trip.

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void wgmma_fp8_saturation_launch(const __nv_fp8_e4m3 *d_A, const __nv_fp8_e4m3 *d_B,
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
    fprintf(stderr, "wgmma_fp8_saturation_run: %s (rc=%d)\n", msg, rc);
    exit(1);
}

int main(int argc, char **argv) {
    int M = 64, N = 64, K = 32;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "wgmma_fp8_saturation_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n_a = (size_t)M * K, n_b = (size_t)K * N, n_c = (size_t)M * N;
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    float *h_C = (float *)malloc(n_c * sizeof(float));
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    __nv_fp8_e4m3 *h_A = (__nv_fp8_e4m3 *)malloc(n_a * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 *h_B = (__nv_fp8_e4m3 *)malloc(n_b * sizeof(__nv_fp8_e4m3));
    for (size_t i = 0; i < n_a; ++i) h_A[i] = __nv_fp8_e4m3(A_f[i]);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            h_B[n * K + k] = __nv_fp8_e4m3(B_f[k * N + n]);

    __nv_fp8_e4m3 *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A, n_a * sizeof(__nv_fp8_e4m3)); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, n_b * sizeof(__nv_fp8_e4m3)); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, n_c * sizeof(float));         if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, h_A, n_a * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B, n_b * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);

    // Launch the wgmma kernel. On Hopper sm_90a it computes the full
    // 64×64 m64n64k32 FP8 E4M3 GEMM and writes d_C; the post-launch
    // D2H copy below makes the kernel's output the row's authoritative
    // result. (R21 drift-recovery rewrote the kernel from a
    // sink-store to real per-fragment stores; see kernel header.
    // R20 fix per Codex Round-19-review removed an obsolete
    // explicit `cudaMemcpy(d_C, h_C, ..., HostToDevice)` that would
    // destroy the kernel's output before the D2H. Pre-Round-20 dry-run-
    // cpu missed this because dry-run never invokes the runner binary.)
    wgmma_fp8_saturation_launch(d_A, d_B, d_C, M, N, K, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"wgmma_fp8_saturation\",\n");
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
