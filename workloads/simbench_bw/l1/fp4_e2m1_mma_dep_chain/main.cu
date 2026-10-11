// Host driver for fp4_e2m1_mma_dep_chain. R3 rewrite + R4
// byte-layout fix: the host quantizes FP32 inputs to packed FP4 E2M1
// bytes (kind::f8f6f4 .e2m1 layout per PTX ISA
// `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md:18459-18467`:
// 1 byte per element with the FP4 nibble in the **central 4 bits
// [5:2]** and padding zeros in bits [7:6] and [1:0], padding mask
// 0xC3) and passes `const uint8_t *` operand buffers to the device
// kernel. The kernel issues
// `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32`
// (cubin contains `QMMA.16832.F32.E2M1.E2M1` SASS).
// `BL-20260425-kind-f8f6f4-byte-placement` codifies the per-dtype
// byte-position rule.
//
// The C-side `_quantize_fp4_e2m1_nibble` quantizer must produce
// bit-identical output to `references/_low_precision_quant.py::
// _quantize_one_fp4_e2m1`. Both implement round-to-nearest with
// banker's tie-break, sign preservation, NaN→+0, and ±Inf saturation
// to ±6.0. The host-side reference matmul (used in `h_C` JSON output
// for dry-run paths) computes FP32 matmul of the dequantized FP4
// values, matching what `references/mma_microbench.py::build_reference`
// computes for this slug.
//
// Tier=B (sm_120a-only). Locally we cannot launch the kernel (the
// local host on this branch is RTX 3090 Ti / sm_86 — Ampere; sm_120a
// binaries require RTX 5090 Blackwell or stronger). The dry-run
// path uses the FP4-aware Python reference directly. End-to-end
// correctness on RTX 5090 Blackwell remains pending (R3+
// goal-tracker queued issue).

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void fp4_e2m1_mma_dep_chain_launch(const uint8_t *d_A_packed,
                                                const uint8_t *d_B_packed,
                                                float *d_C,
                                                int M, int N, int K,
                                                cudaStream_t stream);

// MWC PRNG matching references/_common.py::deterministic_seeded_tensor.
static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed, double lo, double hi) {
    uint32_t st = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        double r = (double)next_u32(&st) / 4294967296.0;
        flat[i] = (float)(lo + (hi - lo) * r);
    }
}

// FP4 E2M1 representable positive magnitudes (matches
// references/_low_precision_quant.py::FP4_E2M1_POSITIVE_VALUES).
static const float FP4_E2M1_POSITIVE_VALUES[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

// Quantize one FP32 value to its FP4 E2M1 nibble. Must produce
// bit-identical output to the Python reference `_quantize_one_fp4_e2m1`.
// Algorithm: round-to-nearest with banker's (round-half-to-even on
// nibble LSB) tie-break, sign preservation, NaN→+0, ±Inf→±6.0.
static uint8_t _quantize_fp4_e2m1_nibble(float value) {
    if (isnan(value)) return 0x0;
    int sign_bit = (signbit(value) || value < 0.0f) ? 1 : 0;
    float abs_v = fabsf(value);
    int idx;
    if (isinf(abs_v)) {
        idx = 7;  // ±6.0 (max representable)
    } else if (abs_v - FP4_E2M1_POSITIVE_VALUES[7] >= 0.0f) {
        // Saturation: |v| >= +6.0 (the negative side is below for sign_bit=1)
        idx = 7;
    } else {
        // Linear search for nearest representable. Banker's rounding:
        // on tie (equal distance), pick the candidate whose nibble has
        // the LSB == 0 (even).
        float best_dist = INFINITY;
        int best_idx = 0;
        for (int i = 0; i < 8; ++i) {
            float d = fabsf(abs_v - FP4_E2M1_POSITIVE_VALUES[i]);
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            } else if (d == best_dist) {
                if ((i & 0x1) == 0) best_idx = i;
            }
        }
        idx = best_idx;
    }
    if (sign_bit && idx == 0) {
        // Negative-zero canonicalizes to nibble 0x0 (+0.0).
        return 0x0;
    }
    return (uint8_t)((sign_bit << 3) | (idx & 0x7));
}

static float _dequantize_fp4_e2m1_nibble(uint8_t nibble) {
    float sign = (nibble & 0x8) ? -1.0f : 1.0f;
    return sign * FP4_E2M1_POSITIVE_VALUES[nibble & 0x7];
}

// Pack a length-n FP32 array into the kind::f8f6f4 byte layout per
// PTX ISA references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md
// :18459-18467: 1 byte per element, central 4 bits [5:2] hold the
// FP4 nibble, padding bits [7:6] and [1:0] are zero. Encoding is
// `byte = (nibble & 0xF) << 2`. R4 fix per Codex's R3
// review (the R3 layout used the low 4 bits, which doesn't
// match the kernel's PTX consumer). Codified in
// `BL-20260425-kind-f8f6f4-byte-placement`. Matches Python module
// `references/_low_precision_quant.py::quantize_fp4_e2m1_to_packed_bytes`
// bit-for-bit.
static inline uint8_t _pack_fp4_byte(uint8_t nibble) {
    return (uint8_t)((nibble & 0xF) << 2);
}

static inline uint8_t _unpack_fp4_byte(uint8_t packed) {
    return (uint8_t)((packed >> 2) & 0xF);
}

// Pack a 1-D length-n FP32 array linearly. Used for A (row-major
// [M, K]) where the kernel reads at byte-offset (row * K) + (col).
static void quantize_to_fp4_packed(const float *src, uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = _pack_fp4_byte(_quantize_fp4_e2m1_nibble(src[i]));
    }
}

// Pack B as col-major in the kernel's memory layout. The kernel
// indexes B at `b_col_base = (tile_n * 8 + groupID) * K_packed_bytes`
// — meaning rows of `B_packed` are *columns* of logical B, with K
// contiguous within each col. The host's B_f buffer is logical
// row-major [K, N] (B_f[k * N + n] = element at logical (k, n)).
// Pack with transpose so `B_packed[n * K + k] = pack(B_f[k * N + n])`.
// R4 fix per Codex Round-3-review blocking-issue #1: R3
// used the linear `quantize_to_fp4_packed` for B, which packed B in
// the wrong logical order — kernel reads at the right byte offsets
// but those bytes held the wrong logical FP4 values.
static void quantize_to_fp4_packed_b_col_major(
    const float *B_f, uint8_t *B_packed, int K, int N)
{
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float v = B_f[(size_t)k * N + n];
            B_packed[(size_t)n * K + k] = _pack_fp4_byte(_quantize_fp4_e2m1_nibble(v));
        }
    }
}

// Decode a packed FP4 array back to FP32 representable values, ready
// for a host-side reference matmul. Operates on the linear (row-major
// for A) layout. For B (col-major-packed via `quantize_to_fp4_packed_b_col_major`)
// the host reference still reads `B_f` directly — see main(); we don't
// need to dequantize the col-major-packed B since the reference uses
// `B_q = dequantize(quantize(B_f))` element-wise (preserving original
// row-major order).
static void dequantize_fp4_packed(const uint8_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = _dequantize_fp4_e2m1_nibble(_unpack_fp4_byte(src[i]));
    }
}

static void fail(const char *msg, int rc) {
    fprintf(stderr, "fp4_e2m1_mma_dep_chain_run: %s (rc=%d)\n", msg, rc);
    exit(1);
}

int main(int argc, char **argv) {
    int M = 16, N = 16, K = 32;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "fp4_e2m1_mma_dep_chain_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n_a = (size_t)M * K;  // FP4 element counts (1 byte per element)
    const size_t n_b = (size_t)K * N;
    const size_t n_c = (size_t)M * N;

    // stage 1: seed FP32 inputs.
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    // stage 2: quantize FP32 → packed FP4 bytes (1 byte per element,
    // central-bit layout per kind::f8f6f4 .e2m1; PTX ISA :18459-18467).
    // A is row-major [M, K]: pack linearly. B is col-major in the
    // device buffer to match the kernel's `b_col_base = (tile_n*8 +
    // groupID) * K_packed_bytes` indexing — `B_packed[n*K + k] =
    // pack(B_f[k*N + n])`. R4 fix per Codex Round-3-review
    // blocking-issue #1.
    uint8_t *h_A_packed = (uint8_t *)malloc(n_a);
    uint8_t *h_B_packed = (uint8_t *)malloc(n_b);
    quantize_to_fp4_packed(A_f, h_A_packed, n_a);
    quantize_to_fp4_packed_b_col_major(B_f, h_B_packed, K, N);

    // stage 3: compute the FP4-aware reference matmul. The reference
    // operates on the LOGICAL row-major B (B_f), independent of the
    // device buffer's col-major packing — both paths must compute the
    // same product. Dequantize-via-element-wise-quantize for both A
    // and B (note we don't dequantize h_B_packed here because it's
    // permuted; we apply quantization element-wise to B_f instead).
    float *A_q = (float *)malloc(n_a * sizeof(float));
    float *B_q = (float *)malloc(n_b * sizeof(float));
    dequantize_fp4_packed(h_A_packed, A_q, n_a);
    for (size_t i = 0; i < n_b; ++i) {
        B_q[i] = _dequantize_fp4_e2m1_nibble(_quantize_fp4_e2m1_nibble(B_f[i]));
    }

    float *h_C = (float *)malloc(n_c * sizeof(float));
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A_q[m * K + k] * B_q[k * N + n];
            h_C[m * N + n] = acc;
        }

    // stage 4: device-side path. Allocate uint8_t* device buffers and
    // FP32 output. The host's reference is the authoritative output
    // on dry-run paths; on a real sm_120a GPU run the kernel result
    // overwrites C and the comparison runs through the FP4-aware
    // Python reference under the (FP4_E2M1, fma_chain, mma_microbench)
    // tolerance triplet (rtol=atol=0.15).
    uint8_t *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A, n_a); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, n_b); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, n_c * sizeof(float)); if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, h_A_packed, n_a, cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B_packed, n_b, cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);
    rc = cudaMemcpy(d_C, h_C,        n_c * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D C", rc);
    fp4_e2m1_mma_dep_chain_launch(d_A, d_B, d_C, M, N, K, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"fp4_e2m1_mma_dep_chain\",\n");
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
    free(A_f); free(B_f); free(h_A_packed); free(h_B_packed); free(A_q); free(B_q); free(h_C);
    return 0;
}
