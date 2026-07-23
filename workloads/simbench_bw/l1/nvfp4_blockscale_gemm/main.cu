// Host driver for nvfp4_blockscale_gemm.
//
// Seeds A (M x K) and B (K x N) via the same MWC PRNG as
// references/_common.py::deterministic_seeded_tensor, computes a
// per-16-element-block UE4M3 scale for every row of A and every
// column of B, quantizes to NVFP4 (E2M1), invokes the kernel, and
// emits the resulting FP32 [M x N] matrix as a nested JSON list.
//
// Reference contract: references/mma_microbench.py::build_reference
// returns an FP32 matmul of A (M x K, seed)  x B (K x N, seed+1), each
// seeded via `deterministic_seeded_tensor(..., low=-0.5, high=0.5)`.
// The kernel's quantized output is compared under the
// `(NVFP4, fma_chain, mma_microbench)` tolerance triplet
// (rtol=atol=0.1, max_ulp=4).  Per-block scaling gives near-FP8
// precision per 16-element stripe, which is what the NVFP4 tolerance
// band is calibrated against (in contrast to pure FP4 at ~0.15 rtol).

#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void nvfp4_blockscale_gemm_launch(
    const uint8_t *d_A_packed,
    const uint8_t *d_B_packed,
    const uint8_t *d_scale_A,
    const uint8_t *d_scale_B,
    float         *d_C,
    int            M,
    int            N,
    int            K,
    cudaStream_t   stream);


// -----------------------------------------------------------------------------
// MWC PRNG matching references/_common.py::deterministic_seeded_tensor
// -----------------------------------------------------------------------------

static inline uint32_t next_u32(uint32_t *state) {
    *state = (*state) * 1103515245u + 12345u;
    return *state;
}

static inline float next_float(uint32_t *state, double low, double high) {
    double r = (double)next_u32(state) / 4294967296.0;
    return (float)(low + (high - low) * r);
}

static void seed_tensor(float *flat, size_t n, uint32_t seed, double low, double high) {
    uint32_t state = (seed * 2654435761u + 1u);
    for (size_t i = 0; i < n; ++i) flat[i] = next_float(&state, low, high);
}


// -----------------------------------------------------------------------------
// FP32 -> E2M1 quantization (signed 4-bit FP: sign + 2 exp + 1 mantissa).
// Representable positive values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
//   0000=+0  0001=+0.5  0010=+1.0  0011=+1.5
//   0100=+2  0101=+3    0110=+4    0111=+6
//   (negatives mirror with bit 3 = sign)
// -----------------------------------------------------------------------------

static inline uint8_t fp32_to_e2m1(float x) {
    if (isnan(x) || x == 0.0f) return 0;
    const uint8_t sign_bit = (x < 0.0f) ? 0x8 : 0x0;
    const float ax = fabsf(x);
    uint8_t mag;
    if      (ax < 0.25f) mag = 0x0;
    else if (ax < 0.75f) mag = 0x1;
    else if (ax < 1.25f) mag = 0x2;
    else if (ax < 1.75f) mag = 0x3;
    else if (ax < 2.5f)  mag = 0x4;
    else if (ax < 3.5f)  mag = 0x5;
    else if (ax < 5.0f)  mag = 0x6;
    else                 mag = 0x7;
    return sign_bit | mag;
}


// -----------------------------------------------------------------------------
// FP32 -> UE4M3 (unsigned 7-bit FP: 4 exp bits bias=7, 3 mantissa bits).
// Subnormal: 0 <= x < 2^-6, encoded with E_field=0, M in [0,7].
// Normal:    2^-6 <= x <= 480, encoded as (E_field << 3) | M,
//            value = (1 + M/8) * 2^(E_field - 7).
// No Inf/NaN in UE4M3; saturate.
// -----------------------------------------------------------------------------

static uint8_t fp32_to_ue4m3(float x) {
    if (x <= 0.0f || isnan(x)) return 0;
    if (x >= 480.0f) return 0x7e;        // largest normal

    union { float f; uint32_t u; } pun;
    pun.f = x;
    int      exp_f32  = (int)((pun.u >> 23) & 0xff) - 127;
    uint32_t mant_f32 = pun.u & 0x7fffffu;

    if (exp_f32 < -9) return 0;          // underflow to zero

    if (exp_f32 < -6) {
        // Subnormal: x ~= M/8 * 2^-6.  Effective f32 mantissa bits
        // (with implicit 1) shifted right to align to M's 3-bit field
        // at the subnormal boundary.
        int shift = -6 - exp_f32;        // 1..3
        uint32_t m_full = (mant_f32 | 0x800000u);          // add implicit 1
        int M = (int)(m_full >> (20 + shift));             // top 3 bits after shift
        // Round: check the bit immediately below.
        uint32_t round_mask = m_full & ((1u << (20 + shift)) - 1u);
        uint32_t half = 1u << (19 + shift);
        if (round_mask > half || (round_mask == half && (M & 1))) M += 1;
        if (M > 7) M = 7;
        return (uint8_t)(M & 0x7);
    }

    int E_field = exp_f32 + 7;
    int M       = (int)(mant_f32 >> 20);         // top 3 bits
    uint32_t remainder = mant_f32 & 0xfffffu;    // remaining 20 bits
    if (remainder > 0x80000u || (remainder == 0x80000u && (M & 1))) {
        M += 1;
        if (M > 7) { M = 0; E_field += 1; }
    }
    if (E_field >= 16) return 0x7e;
    return (uint8_t)((E_field << 3) | M);
}


// -----------------------------------------------------------------------------
// UE4M3 -> FP32 (for host-side correctness inspection, currently unused).
// -----------------------------------------------------------------------------

static inline float ue4m3_to_fp32(uint8_t b) {
    int E_field = (b >> 3) & 0xf;
    int M       = b & 0x7;
    if (E_field == 0) return ldexpf((float)M / 8.0f, -6);
    return ldexpf(1.0f + (float)M / 8.0f, E_field - 7);
}


// -----------------------------------------------------------------------------
// Per-16-element block absmax -> UE4M3 scale encoding.  Dequantized FP4
// value range is [-6, 6], so scale = absmax / 6 keeps the quantized
// representation within range without saturation.
// -----------------------------------------------------------------------------

#define BLOCK_K 16
#define FP4_MAX 6.0f

static float block_absmax_row(const float *row, int k_start, int k_end) {
    float m = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
        float a = fabsf(row[k]);
        if (a > m) m = a;
    }
    return m;
}

static float block_absmax_col(const float *B, int K, int N, int j, int k_start, int k_end) {
    float m = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
        float a = fabsf(B[k * N + j]);
        if (a > m) m = a;
    }
    return m;
}

static uint8_t absmax_to_ue4m3_scale(float absmax) {
    if (absmax < 1e-30f) return fp32_to_ue4m3(1.0f / 6.0f); // never zero scale
    return fp32_to_ue4m3(absmax / FP4_MAX);
}


// -----------------------------------------------------------------------------
// Packing.  A_packed is [M][K/2] row-major; B_packed is [N][K/2]
// col-major (col contiguous).  Within each byte: bits [3:0] hold the
// even-K FP4, bits [7:4] hold the odd-K FP4 — giving per-nibble
// little-endian ordering when loaded as a uint32.
// -----------------------------------------------------------------------------

static void pack_A(uint8_t *A_packed, uint8_t *scale_A,
                   const float *A_f32, int M, int K) {
    int num_blocks = K / BLOCK_K;
    for (int i = 0; i < M; ++i) {
        const float *row = A_f32 + (size_t)i * K;
        for (int b = 0; b < num_blocks; ++b) {
            float amax = block_absmax_row(row, b * BLOCK_K, (b + 1) * BLOCK_K);
            scale_A[i * num_blocks + b] = absmax_to_ue4m3_scale(amax);
        }
        for (int k = 0; k < K; k += 2) {
            int b0 = k / BLOCK_K;
            int b1 = (k + 1) / BLOCK_K;
            float s0 = ue4m3_to_fp32(scale_A[i * num_blocks + b0]);
            float s1 = ue4m3_to_fp32(scale_A[i * num_blocks + b1]);
            uint8_t even = fp32_to_e2m1(row[k    ] / s0);
            uint8_t odd_ = fp32_to_e2m1(row[k + 1] / s1);
            A_packed[i * (K / 2) + k / 2] = (odd_ << 4) | even;
        }
    }
}

static void pack_B(uint8_t *B_packed, uint8_t *scale_B,
                   const float *B_f32, int K, int N) {
    int num_blocks = K / BLOCK_K;
    for (int j = 0; j < N; ++j) {
        for (int b = 0; b < num_blocks; ++b) {
            float amax = block_absmax_col(B_f32, K, N, j, b * BLOCK_K, (b + 1) * BLOCK_K);
            scale_B[j * num_blocks + b] = absmax_to_ue4m3_scale(amax);
        }
        for (int k = 0; k < K; k += 2) {
            int b0 = k / BLOCK_K;
            int b1 = (k + 1) / BLOCK_K;
            float s0 = ue4m3_to_fp32(scale_B[j * num_blocks + b0]);
            float s1 = ue4m3_to_fp32(scale_B[j * num_blocks + b1]);
            uint8_t even = fp32_to_e2m1(B_f32[(k    ) * N + j] / s0);
            uint8_t odd_ = fp32_to_e2m1(B_f32[(k + 1) * N + j] / s1);
            B_packed[j * (K / 2) + k / 2] = (odd_ << 4) | even;
        }
    }
}


static void fail(const char *msg, int rc) {
    fprintf(stderr, "nvfp4_blockscale_gemm_run: %s (cuda rc=%d: %s)\n",
            msg, rc, cudaGetErrorString((cudaError_t)rc));
    exit(1);
}


int main(int argc, char **argv) {
    int M = 64, N = 64, K = 256;
    uint32_t seed = 42;
    const char *shape_point = "correctness_00";
    const char *slug = "nvfp4_blockscale_gemm";

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m")    == 0 && i + 1 < argc) M    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n")    == 0 && i + 1 < argc) N    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k")    == 0 && i + 1 < argc) K    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--shape-point") == 0 && i + 1 < argc) shape_point = argv[++i];
        else if (strcmp(argv[i], "--scale-vec") == 0 && i + 1 < argc) { (void)argv[++i]; /* NVFP4 canonical block_scale=16; kernel uses scale_vec::4X */ }
        else {
            fprintf(stderr, "nvfp4_blockscale_gemm_run: unknown arg %s\n", argv[i]);
            return 2;
        }
    }

    if ((M % 16) != 0 || (N % 8) != 0 || (K % 64) != 0) {
        fprintf(stderr, "nvfp4_blockscale_gemm_run: "
                "M must be a multiple of 16, N of 8, K of 64 (got M=%d N=%d K=%d)\n",
                M, N, K);
        return 2;
    }

    const size_t nA = (size_t)M * K;
    const size_t nB = (size_t)K * N;
    const size_t nC = (size_t)M * N;
    const int num_blocks = K / BLOCK_K;

    float *h_A = (float *)malloc(nA * sizeof(float));
    float *h_B = (float *)malloc(nB * sizeof(float));
    float *h_C = (float *)malloc(nC * sizeof(float));

    seed_tensor(h_A, nA, seed,     -0.5, 0.5);
    seed_tensor(h_B, nB, seed + 1, -0.5, 0.5);

    uint8_t *h_A_packed = (uint8_t *)malloc((size_t)M * (K / 2));
    uint8_t *h_B_packed = (uint8_t *)malloc((size_t)N * (K / 2));
    uint8_t *h_scale_A  = (uint8_t *)malloc((size_t)M * num_blocks);
    uint8_t *h_scale_B  = (uint8_t *)malloc((size_t)N * num_blocks);

    pack_A(h_A_packed, h_scale_A, h_A, M, K);
    pack_B(h_B_packed, h_scale_B, h_B, K, N);

    uint8_t *d_A_packed = nullptr;
    uint8_t *d_B_packed = nullptr;
    uint8_t *d_scale_A  = nullptr;
    uint8_t *d_scale_B  = nullptr;
    float   *d_C        = nullptr;
    cudaError_t rc;
    rc = cudaMalloc(&d_A_packed, (size_t)M * (K / 2));             if (rc) fail("cudaMalloc A",  rc);
    rc = cudaMalloc(&d_B_packed, (size_t)N * (K / 2));             if (rc) fail("cudaMalloc B",  rc);
    rc = cudaMalloc(&d_scale_A,  (size_t)M * num_blocks);          if (rc) fail("cudaMalloc sA", rc);
    rc = cudaMalloc(&d_scale_B,  (size_t)N * num_blocks);          if (rc) fail("cudaMalloc sB", rc);
    rc = cudaMalloc(&d_C,        nC * sizeof(float));              if (rc) fail("cudaMalloc C",  rc);
    rc = cudaMemcpy(d_A_packed, h_A_packed, (size_t)M * (K / 2),
                    cudaMemcpyHostToDevice);                       if (rc) fail("H2D A",  rc);
    rc = cudaMemcpy(d_B_packed, h_B_packed, (size_t)N * (K / 2),
                    cudaMemcpyHostToDevice);                       if (rc) fail("H2D B",  rc);
    rc = cudaMemcpy(d_scale_A,  h_scale_A,  (size_t)M * num_blocks,
                    cudaMemcpyHostToDevice);                       if (rc) fail("H2D sA", rc);
    rc = cudaMemcpy(d_scale_B,  h_scale_B,  (size_t)N * num_blocks,
                    cudaMemcpyHostToDevice);                       if (rc) fail("H2D sB", rc);

    nvfp4_blockscale_gemm_launch(d_A_packed, d_B_packed,
                                 d_scale_A, d_scale_B,
                                 d_C, M, N, K, 0);
    rc = cudaDeviceSynchronize();                                  if (rc) fail("sync", rc);
    rc = cudaMemcpy(h_C, d_C, nC * sizeof(float), cudaMemcpyDeviceToHost);
    if (rc) fail("D2H C", rc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"slug\": \"%s\",\n", slug);
    printf("  \"shape_set\": \"correctness_shapes\",\n");
    printf("  \"shape_point\": \"%s\",\n", shape_point);
    printf("  \"output\": [");
    for (int i = 0; i < M; ++i) {
        if (i > 0) printf(",");
        printf("[");
        for (int j = 0; j < N; ++j) {
            if (j > 0) printf(",");
            printf("%.6g", h_C[i * N + j]);
        }
        printf("]");
    }
    printf("]\n}\n");

    cudaFree(d_A_packed); cudaFree(d_B_packed);
    cudaFree(d_scale_A);  cudaFree(d_scale_B);
    cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    free(h_A_packed); free(h_B_packed);
    free(h_scale_A);  free(h_scale_B);
    return 0;
}
