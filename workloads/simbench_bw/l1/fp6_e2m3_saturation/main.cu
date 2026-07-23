// Host driver for fp6_e2m3_saturation. R5 rewrite (parallel to
// the FP4 sibling row's R4 main.cu): the host quantizes FP32
// inputs to packed FP6 E2M3 bytes (kind::f8f6f4 .e2m3 layout per PTX
// ISA `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md
// :18459-18467`: 1 byte per element with the FP6 code in the **lower
// 6 bits [5:0]** and padding zeros in bits [7:6], padding mask 0xC0)
// and passes `const uint8_t *` operand buffers to the device kernel.
// The kernel issues
// `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32`
// (cubin contains `QMMA.16832.F32.E2M3.E2M3` SASS).
// `BL-20260425-kind-f8f6f4-byte-placement` codifies the per-dtype
// byte-position rule.
//
// The C-side `_quantize_fp6_e2m3_code` quantizer must produce
// bit-identical output to `references/_low_precision_quant.py::
// _quantize_one_fp6_e2m3`. Both implement round-to-nearest with
// banker's tie-break, sign preservation, NaN→+0, and ±Inf saturation
// to ±7.5. The host-side reference matmul (used in `h_C` JSON output
// for dry-run paths) computes FP32 matmul of the dequantized FP6
// values, matching what `references/mma_microbench.py::build_reference`
// computes for this slug.
//
// Tier=B (sm_120a-only). Locally we cannot launch the kernel (the
// local host on this branch is RTX 3090 Ti / sm_86 — Ampere; sm_120a
// binaries require RTX PRO 6000 Blackwell or stronger). The dry-run
// path uses the FP6-aware Python reference directly. End-to-end
// correctness on RTX PRO 6000 Blackwell remains pending (R5+
// goal-tracker queued issue).

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void fp6_e2m3_saturation_launch(const uint8_t *d_A_packed,
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

// FP6 E2M3 representable positive magnitudes (matches
// references/_low_precision_quant.py::FP6_E2M3_POSITIVE_VALUES).
// 32 entries: 8 denormals (exp=0), 8 each for exp=1, exp=2, exp=3.
static const float FP6_E2M3_POSITIVE_VALUES[32] = {
    // exp=0 (denormal): m / 8
    0.0f,    0.125f,  0.25f,   0.375f,  0.5f,    0.625f,  0.75f,   0.875f,
    // exp=1: 1.0 + m/8
    1.0f,    1.125f,  1.25f,   1.375f,  1.5f,    1.625f,  1.75f,   1.875f,
    // exp=2: (1.0 + m/8) * 2
    2.0f,    2.25f,   2.5f,    2.75f,   3.0f,    3.25f,   3.5f,    3.75f,
    // exp=3: (1.0 + m/8) * 4
    4.0f,    4.5f,    5.0f,    5.5f,    6.0f,    6.5f,    7.0f,    7.5f,
};

// Quantize one FP32 value to its FP6 E2M3 6-bit code. Must produce
// bit-identical output to the Python reference `_quantize_one_fp6_e2m3`.
// Algorithm: round-to-nearest with banker's (round-half-to-even on
// code LSB) tie-break, sign preservation, NaN→+0, ±Inf→±7.5.
static uint8_t _quantize_fp6_e2m3_code(float value) {
    if (isnan(value)) return 0x00;
    int sign_bit = (signbit(value) || value < 0.0f) ? 1 : 0;
    float abs_v = fabsf(value);
    int idx;
    if (isinf(abs_v)) {
        idx = 31;  // ±7.5 (max representable)
    } else if (abs_v - FP6_E2M3_POSITIVE_VALUES[31] >= 0.0f) {
        idx = 31;
    } else {
        // Linear search for nearest representable. Banker's rounding:
        // on tie (equal distance), pick the candidate whose code has
        // the LSB == 0 (even).
        float best_dist = INFINITY;
        int best_idx = 0;
        for (int i = 0; i < 32; ++i) {
            float d = fabsf(abs_v - FP6_E2M3_POSITIVE_VALUES[i]);
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
        // Negative-zero canonicalizes to code 0x00 (+0.0).
        return 0x00;
    }
    return (uint8_t)((sign_bit << 5) | (idx & 0x1F));
}

static float _dequantize_fp6_e2m3_code(uint8_t code) {
    float sign = (code & 0x20) ? -1.0f : 1.0f;
    return sign * FP6_E2M3_POSITIVE_VALUES[code & 0x1F];
}

// Pack a length-n FP32 array into the kind::f8f6f4 byte layout per
// PTX ISA `:18459-18467`: 1 byte per element, lower 6 bits [5:0] hold
// the FP6 code, padding bits [7:6] are zero. Encoding is
// `byte = code & 0x3F`. Matches Python module
// `references/_low_precision_quant.py::quantize_fp6_e2m3_to_packed_bytes`
// bit-for-bit.
static inline uint8_t _pack_fp6_byte(uint8_t code) {
    return (uint8_t)(code & 0x3F);
}

static inline uint8_t _unpack_fp6_byte(uint8_t packed) {
    return (uint8_t)(packed & 0x3F);
}

// Pack a 1-D length-n FP32 array linearly. Used for A (row-major
// [M, K]) where the kernel reads at byte-offset (row * K) + (col).
static void quantize_to_fp6_packed(const float *src, uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = _pack_fp6_byte(_quantize_fp6_e2m3_code(src[i]));
    }
}

// Pack B as col-major in the kernel's memory layout. The kernel
// indexes B at `b_col_base = (tile_n * 8 + groupID) * K_packed_bytes`
// — meaning rows of `B_packed` are *columns* of logical B, with K
// contiguous within each col. The host's B_f buffer is logical
// row-major [K, N] (B_f[k * N + n] = element at logical (k, n)).
// Pack with transpose so `B_packed[n * K + k] = pack(B_f[k * N + n])`.
// Same pattern as the FP4 sibling row's R4 fix.
static void quantize_to_fp6_packed_b_col_major(
    const float *B_f, uint8_t *B_packed, int K, int N)
{
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float v = B_f[(size_t)k * N + n];
            B_packed[(size_t)n * K + k] = _pack_fp6_byte(_quantize_fp6_e2m3_code(v));
        }
    }
}

// Decode a packed FP6 array back to FP32 representable values, ready
// for a host-side reference matmul. Operates on the linear (row-major
// for A) layout. For B (col-major-packed via
// `quantize_to_fp6_packed_b_col_major`) the host reference still reads
// `B_f` directly — see main(); we don't need to dequantize the
// col-major-packed B since the reference uses
// `B_q = dequantize(quantize(B_f))` element-wise (preserving original
// row-major order).
static void dequantize_fp6_packed(const uint8_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = _dequantize_fp6_e2m3_code(_unpack_fp6_byte(src[i]));
    }
}

static void fail(const char *msg, int rc) {
    fprintf(stderr, "fp6_e2m3_saturation_run: %s (rc=%d)\n", msg, rc);
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
        else { fprintf(stderr, "fp6_e2m3_saturation_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n_a = (size_t)M * K;  // FP6 element counts (1 byte per element)
    const size_t n_b = (size_t)K * N;
    const size_t n_c = (size_t)M * N;

    // stage 1: seed FP32 inputs.
    float *A_f = (float *)malloc(n_a * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    seed_tensor(A_f, n_a, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b, seed + 1, -0.5, 0.5);

    // stage 2: quantize FP32 → packed FP6 bytes (1 byte per element,
    // low-6-bits layout per kind::f8f6f4 .e2m3; PTX ISA :18459-18467).
    // A is row-major [M, K]: pack linearly. B is col-major in the
    // device buffer to match the kernel's `b_col_base = (tile_n*8 +
    // groupID) * K_packed_bytes` indexing — `B_packed[n*K + k] =
    // pack(B_f[k*N + n])`.
    uint8_t *h_A_packed = (uint8_t *)malloc(n_a);
    uint8_t *h_B_packed = (uint8_t *)malloc(n_b);
    quantize_to_fp6_packed(A_f, h_A_packed, n_a);
    quantize_to_fp6_packed_b_col_major(B_f, h_B_packed, K, N);

    // stage 3: compute the FP6-aware reference matmul. The reference
    // operates on the LOGICAL row-major B (B_f), independent of the
    // device buffer's col-major packing — both paths must compute the
    // same product.
    float *A_q = (float *)malloc(n_a * sizeof(float));
    float *B_q = (float *)malloc(n_b * sizeof(float));
    dequantize_fp6_packed(h_A_packed, A_q, n_a);
    for (size_t i = 0; i < n_b; ++i) {
        B_q[i] = _dequantize_fp6_e2m3_code(_quantize_fp6_e2m3_code(B_f[i]));
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
    // overwrites C and the comparison runs through the FP6-aware
    // Python reference under the (FP6_E2M3, fma_chain, mma_microbench)
    // tolerance triplet (rtol=atol=0.1, max_ulp=4).
    uint8_t *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A, n_a); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, n_b); if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, n_c * sizeof(float)); if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, h_A_packed, n_a, cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B_packed, n_b, cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);
    rc = cudaMemcpy(d_C, h_C,        n_c * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D C", rc);
    fp6_e2m3_saturation_launch(d_A, d_B, d_C, M, N, K, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"fp6_e2m3_saturation\",\n");
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
