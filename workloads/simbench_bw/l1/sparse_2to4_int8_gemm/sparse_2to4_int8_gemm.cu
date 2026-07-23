// slug: sparse_2to4_int8_gemm
// tier: P
// arch_gate: sm_90a sm_120
//
// INT8 sparse 2:4 GEMM via warp-level
//   mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32
//
// Per BL-20260424-kernel-contract-vs-shape-match the inline PTX is the
// contract enforcement; cubin must contain *.SP.* sparse IMMA SASS.
//
// 2:4 keep pattern: keep K-indices {4g+0, 4g+1}; zero {4g+2, 4g+3}.
// Per-thread metadata: every 4-bit group is `(1 << 2) | 0 = 0x4`,
// so meta = 0x44444444.
//
// Per-thread fragment layout (m16n8k64, .row x .col, INT8):
//   lane = 0..31; groupID = lane / 4; tidg = lane & 3.
//
//   A_sparse is compressed [M=16, K_compressed=K/2] (the host driver
//   packs A's kept K-positions only, mirroring the 2:4 mask).
//     A_sparse: 4 x b32 per thread = 16 INT8 per thread = 16 rows × 32 K_compressed / 32 threads.
//     a0: row=groupID,   K_compressed-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     a1: row=groupID+8, K_compressed-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     a2: row=groupID,   K_compressed-byte=[tidg*4 + 16 .. tidg*4 + 19]
//     a3: row=groupID+8, K_compressed-byte=[tidg*4 + 16 .. tidg*4 + 19]
//
//   B is **dense** at full K (col-major, K-byte stride = K). The
//   metadata register tells the hardware which K-positions of B to
//   pair with A's kept positions; B is NOT host-side compressed.
//   R13 fix per Codex Round-9-review (the original R1
//   kernel comment + the K_packed_bytes_b = K/2 + 2-b32-for-B
//   reads were wrong; ptxas reported "Argument vector size mismatch
//   for instruction 'Sparse mma'" because B requires 4 b32 per thread,
//   not 2). This mirrors the R7 sparse_2to4_fp8_gemm fix.
//     B: 4 x b32 per thread = 16 INT8 per thread = 8 cols × 64 K_dense / 32 threads.
//     b0: col=groupID, K-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     b1: col=groupID, K-byte=[tidg*4 + 16 .. tidg*4 + 19]
//     b2: col=groupID, K-byte=[tidg*4 + 32 .. tidg*4 + 35]
//     b3: col=groupID, K-byte=[tidg*4 + 48 .. tidg*4 + 51]
//
//   D (4 x s32 per thread): row=groupID/+8, col=tidg*2/+1.
//
// Launch: grid = (M/16, N/8, 1), block = (32, 1, 1).

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void sparse_2to4_int8_gemm_kernel(
    const int8_t *__restrict__ A_sparse, const int8_t *__restrict__ B, float *__restrict__ C,
    int32_t M, int32_t N, int32_t K, float dequant_scale)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    const uint32_t meta = 0x44444444u;

    // A is K-compressed (K_packed_bytes_a per row of A_sparse = K / 2).
    // B is dense at full K (K_packed_bytes_b per col of B = K).
    const int K_packed_bytes_a = K >> 1;
    const int K_packed_bytes_b = K;
    const int a_row0_base = (tile_m * 16 + groupID    ) * K_packed_bytes_a;
    const int a_row1_base = (tile_m * 16 + groupID + 8) * K_packed_bytes_a;
    const int b_col_base  = (tile_n * 8  + groupID    ) * K_packed_bytes_b;

    for (int k0 = 0; k0 < K; k0 += 64) {
        const int k_byte_a = k0 >> 1;
        const int k_byte_b = k0;
        const int a_off = k_byte_a + tidg * 4;
        const int b_off = k_byte_b + tidg * 4;

        uint32_t a0 = *reinterpret_cast<const uint32_t *>(A_sparse + a_row0_base + a_off);
        uint32_t a1 = *reinterpret_cast<const uint32_t *>(A_sparse + a_row1_base + a_off);
        uint32_t a2 = *reinterpret_cast<const uint32_t *>(A_sparse + a_row0_base + a_off + 16);
        uint32_t a3 = *reinterpret_cast<const uint32_t *>(A_sparse + a_row1_base + a_off + 16);
        uint32_t b0 = *reinterpret_cast<const uint32_t *>(B + b_col_base + b_off);
        uint32_t b1 = *reinterpret_cast<const uint32_t *>(B + b_col_base + b_off + 16);
        uint32_t b2 = *reinterpret_cast<const uint32_t *>(B + b_col_base + b_off + 32);
        uint32_t b3 = *reinterpret_cast<const uint32_t *>(B + b_col_base + b_off + 48);

        asm volatile(
            "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "r"(meta));
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col  = tile_n * 8 + tidg * 2;
    C[c_row0 * N + c_col + 0] = (float)d0 * dequant_scale;
    C[c_row0 * N + c_col + 1] = (float)d1 * dequant_scale;
    C[c_row1 * N + c_col + 0] = (float)d2 * dequant_scale;
    C[c_row1 * N + c_col + 1] = (float)d3 * dequant_scale;
}

extern "C" void sparse_2to4_int8_gemm_launch(
    const int8_t *d_A_sparse, const int8_t *d_B, float *d_C,
    int M, int N, int K, float dequant_scale, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    sparse_2to4_int8_gemm_kernel<<<grid, block, 0, stream>>>(d_A_sparse, d_B, d_C, M, N, K, dequant_scale);
}
