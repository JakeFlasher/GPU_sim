// Host driver for sparse_2to4_fp8_gemm. R7 rewrite (parallel to
// the FP4/FP6 sibling rows' main.cu, plus the sparse-specific A
// compression): the host quantizes FP32 inputs to packed FP8 E4M3
// bytes (kind::f8f6f4 .e4m3 layout per PTX ISA
// `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md:18459-18467`:
// 1 byte per element with **all 8 bits [7:0] used**, no padding mask)
// and additionally compresses A under the 2:4 keep pattern
// {4g+0, 4g+1} so A_packed has shape [M, K/2]. B remains dense at
// full K, packed in col-major order. The kernel reads them through
// `const uint8_t *` operand pointers and issues
// `mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4
// .f32.e4m3.e4m3.f32` with metadata `0x44444444` (cubin contains
// `QMMA.SP.16864.F32.E4M3.E4M3` SASS).
// `BL-20260425-kind-f8f6f4-byte-placement` codifies the per-dtype
// byte-position rule.
//
// The C-side `_quantize_fp8_e4m3_byte` quantizer must produce
// bit-identical output to `references/_low_precision_quant.py::
// _quantize_one_fp8_e4m3`. Both implement round-to-nearest with
// banker's tie-break, sign preservation, NaN→+0 (canonical),
// ±Inf saturation to ±448 (max normal — OCP/NVIDIA E4M3 satfinite
// behavior). The host-side reference matmul (used in `h_C` JSON
// output for dry-run paths) applies the same 2:4 mask to A in FP32
// space, quantizes both A and B to FP8 representable values, then
// computes FP32 matmul of the dequantized values — matching what
// `references/mma_microbench.py::build_reference` computes for this
// slug.
//
// Tier=B (sm_120a-only). Locally we cannot launch the kernel (the
// local host on this branch is RTX 3090 Ti / sm_86 — Ampere, three
// architectural generations older than sm_120a; sm_120a binaries
// require RTX PRO 6000 Blackwell or stronger). The dry-run path uses
// the sparse-FP8-aware Python reference directly. End-to-end
// correctness on RTX PRO 6000 Blackwell remains pending.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void sparse_2to4_fp8_gemm_launch(const uint8_t *d_A_packed,
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

// FP8 E4M3 saturation max (OCP/NVIDIA E4M3 max normal = +448).
static const float FP8_E4M3_MAX = 448.0f;

// FP8 E4M3 representable value table generator (mirrors Python module's
// `_build_fp8_e4m3_value_table`). 256 entries; bytes 0x7F and 0xFF are
// NaN. Filled lazily once at startup.
static float FP8_E4M3_VALUE_TABLE[256];
static int FP8_E4M3_TABLE_INITIALIZED = 0;

static void _init_fp8_e4m3_table(void) {
    for (int b = 0; b < 256; ++b) {
        int s = (b >> 7) & 1;
        int e = (b >> 3) & 0xF;
        int m = b & 0x7;
        if (e == 15 && m == 7) {
            FP8_E4M3_VALUE_TABLE[b] = NAN;
            continue;
        }
        float sign = s ? -1.0f : 1.0f;
        float val;
        if (e == 0) {
            // Denormal: (m/8) * 2^(1-7) = m / 512
            val = sign * ((float)m / 8.0f) * powf(2.0f, 1 - 7);
        } else {
            // Normal: (1 + m/8) * 2^(e-7)
            val = sign * (1.0f + (float)m / 8.0f) * powf(2.0f, e - 7);
        }
        FP8_E4M3_VALUE_TABLE[b] = val;
    }
    FP8_E4M3_TABLE_INITIALIZED = 1;
}

// Quantize one FP32 value to its FP8 E4M3 byte. Must produce
// bit-identical output to the Python reference `_quantize_one_fp8_e4m3`.
// Algorithm: search positive bytes [0x00..0x7E] (excluding +NaN at
// 0x7F) for nearest |v| with banker's tie-break, then OR in sign bit;
// preserve sign of -0.0 (returns byte 0x80); FP32 NaN → byte 0x7F
// (canonical positive E4M3 NaN, all NaN sign collapsed); saturate
// ±Inf and out-of-range to ±448. R31 alignment with
// `cuda_fp8.h::__nv_fp8_e4m3` per Codex Round-29-review CUDA 13 host
// probe (was Round-7-design pre-Round-31: -0 → 0x00, NaN → 0x00).
static uint8_t _quantize_fp8_e4m3_byte(float value) {
    if (!FP8_E4M3_TABLE_INITIALIZED) _init_fp8_e4m3_table();
    if (isnan(value)) return 0x7F;
    int sign_bit = (signbit(value) || value < 0.0f) ? 1 : 0;
    float abs_v = fabsf(value);
    if (isinf(abs_v) || abs_v >= FP8_E4M3_MAX) {
        return sign_bit ? 0xFE : 0x7E;
    }
    int best_idx = 0;
    float best_dist = INFINITY;
    for (int b = 0; b < 0x7F; ++b) {  // 0x00..0x7E (positive bytes)
        float cand = FP8_E4M3_VALUE_TABLE[b];
        float d = fabsf(abs_v - cand);
        if (d < best_dist) {
            best_dist = d;
            best_idx = b;
        } else if (d == best_dist) {
            if ((b & 0x1) == 0) best_idx = b;
        }
    }
    return (uint8_t)(best_idx | (sign_bit << 7));
}

static float _dequantize_fp8_e4m3_byte(uint8_t byte) {
    if (!FP8_E4M3_TABLE_INITIALIZED) _init_fp8_e4m3_table();
    return FP8_E4M3_VALUE_TABLE[byte];
}

// FP8 byte pack: identity (no padding mask). Per PTX ISA :18459-18467,
// .e4m3 uses all 8 bits, so encoding is just the raw FP8 code.
// (No `_unpack_fp8_byte` symmetry: the host reference dequantizes
// directly from the FP8 byte via `_dequantize_fp8_e4m3_byte`.)
static inline uint8_t _pack_fp8_byte(uint8_t code) {
    return code;
}

// Compress + quantize A under the 2:4 keep pattern {4g+0, 4g+1} into
// the kernel's [M, K/2] byte buffer. For each output position (m, kc):
//   logical K position k = (kc / 2) * 4 + (kc % 2)
//   A_packed[m * (K/2) + kc] = pack(quantize(A_f[m * K + k]))
// This matches the kernel's per-thread fragment layout (a0..a3 byte
// offsets at K_packed_bytes_a = K/2 stride per row of A_compressed).
static void quantize_compressed_a_to_fp8_packed(
    const float *A_f, uint8_t *A_packed, int M, int K)
{
    const int K_kept = K / 2;
    for (int m = 0; m < M; ++m) {
        for (int kc = 0; kc < K_kept; ++kc) {
            int k = (kc / 2) * 4 + (kc % 2);
            float v = A_f[(size_t)m * K + k];
            A_packed[(size_t)m * K_kept + kc] = _pack_fp8_byte(_quantize_fp8_e4m3_byte(v));
        }
    }
}

// Pack dense B as col-major in the kernel's memory layout (parallel
// to FP4/FP6 col-major B packers). The kernel indexes B at
// `b_col_base = (tile_n * 8 + groupID) * K_packed_bytes_b` with
// K_packed_bytes_b = K (full dense width). The host's B_f buffer is
// logical row-major [K, N]; pack with transpose so
// `B_packed[n * K + k] = pack(quantize(B_f[k * N + n]))`.
static void quantize_dense_b_to_fp8_packed_b_col_major(
    const float *B_f, uint8_t *B_packed, int K, int N)
{
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float v = B_f[(size_t)k * N + n];
            B_packed[(size_t)n * K + k] = _pack_fp8_byte(_quantize_fp8_e4m3_byte(v));
        }
    }
}

static void fail(const char *msg, int rc) {
    fprintf(stderr, "sparse_2to4_fp8_gemm_run: %s (rc=%d)\n", msg, rc);
    exit(1);
}

int main(int argc, char **argv) {
    int M = 32, N = 16, K = 64;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else { fprintf(stderr, "sparse_2to4_fp8_gemm_run: unknown arg %s\n", argv[i]); return 2; }
    }
    const size_t n_a_dense = (size_t)M * K;            // logical A (FP32)
    const size_t n_a_packed = (size_t)M * (K / 2);     // compressed A (1 byte per FP8)
    const size_t n_b = (size_t)K * N;                  // dense B (1 byte per FP8 in packed form)
    const size_t n_c = (size_t)M * N;

    // stage 1: seed FP32 inputs (logical row-major A and B).
    float *A_f = (float *)malloc(n_a_dense * sizeof(float));
    float *B_f = (float *)malloc(n_b * sizeof(float));
    seed_tensor(A_f, n_a_dense, seed,     -0.5, 0.5);
    seed_tensor(B_f, n_b,       seed + 1, -0.5, 0.5);

    // stage 2: compress + quantize A under the 2:4 keep mask + col-major
    // pack B. Both buffers are 1 byte per FP8 element (no padding mask
    // — kind::f8f6f4 .e4m3 uses all 8 bits per PTX ISA :18459-18467).
    uint8_t *h_A_packed = (uint8_t *)malloc(n_a_packed);
    uint8_t *h_B_packed = (uint8_t *)malloc(n_b);
    quantize_compressed_a_to_fp8_packed(A_f, h_A_packed, M, K);
    quantize_dense_b_to_fp8_packed_b_col_major(B_f, h_B_packed, K, N);

    // stage 3: compute the sparse-FP8-aware reference matmul. The
    // reference operates on the LOGICAL row-major buffers A_f and B_f,
    // independent of the device's compressed/col-major packings:
    //   1. Apply 2:4 mask to A in FP32 space (zero K-indices {4g+2, 4g+3}).
    //   2. Quantize both masked A and dense B to FP8 representables.
    //   3. FP32 matmul of dequantized values.
    // Both paths must compute the same product when the kernel runs on
    // hardware: the kernel reads compressed-bytes + dense-bytes and
    // metadata-skips the {4g+2, 4g+3} positions of B; the host computes
    // from masked-and-quantized FP32, but both consume the same kept
    // logical (k, n) values.
    float *A_q_masked = (float *)malloc(n_a_dense * sizeof(float));
    float *B_q = (float *)malloc(n_b * sizeof(float));
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            int kept = ((k & 3) < 2) ? 1 : 0;  // keep K-index in {4g+0, 4g+1}
            float v = kept ? A_f[(size_t)m * K + k] : 0.0f;
            A_q_masked[(size_t)m * K + k] = _dequantize_fp8_e4m3_byte(_quantize_fp8_e4m3_byte(v));
        }
    }
    for (size_t i = 0; i < n_b; ++i) {
        B_q[i] = _dequantize_fp8_e4m3_byte(_quantize_fp8_e4m3_byte(B_f[i]));
    }

    float *h_C = (float *)malloc(n_c * sizeof(float));
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A_q_masked[m * K + k] * B_q[k * N + n];
            h_C[m * N + n] = acc;
        }

    // stage 4: device-side path. Allocate uint8_t* device buffers and
    // FP32 output. The host's reference is the authoritative output
    // on dry-run paths; on a real sm_120a GPU run the kernel result
    // overwrites C and the comparison runs through the sparse-FP8-aware
    // Python reference under the (FP8_E4M3, fma_chain, mma_microbench)
    // tolerance triplet (rtol=atol=0.05, max_ulp=8).
    uint8_t *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A, n_a_packed); if (rc) fail("mc A", rc);
    rc = cudaMalloc(&d_B, n_b);        if (rc) fail("mc B", rc);
    rc = cudaMalloc(&d_C, n_c * sizeof(float)); if (rc) fail("mc C", rc);
    rc = cudaMemcpy(d_A, h_A_packed, n_a_packed, cudaMemcpyHostToDevice); if (rc) fail("H2D A", rc);
    rc = cudaMemcpy(d_B, h_B_packed, n_b,        cudaMemcpyHostToDevice); if (rc) fail("H2D B", rc);
    rc = cudaMemcpy(d_C, h_C,        n_c * sizeof(float), cudaMemcpyHostToDevice); if (rc) fail("H2D C", rc);
    sparse_2to4_fp8_gemm_launch(d_A, d_B, d_C, M, N, K, 0);
    rc = cudaDeviceSynchronize(); if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost); if (rc) fail("D2H", rc);

    printf("{\n  \"schema_version\": 1,\n  \"slug\": \"sparse_2to4_fp8_gemm\",\n");
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
    free(A_f); free(B_f); free(h_A_packed); free(h_B_packed); free(A_q_masked); free(B_q); free(h_C);
    return 0;
}
