// slug: sparse_2to4_fp8_gemm
// tier: B
// arch_gate: sm_120a
//
// Blackwell sparse 2:4 FP8 (E4M3) GEMM via warp-level inline PTX
//   mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col
//                          .kind::f8f6f4.f32.e4m3.e4m3.f32
//
// Per BL-20260424-kernel-contract-vs-shape-match the inline PTX is
// the contract enforcement; cubin must contain QMMA.SP.16864.F32.E4M3.E4M3
// SASS (the Blackwell-consumer class for sparse 2:4 FP8 warp-level MMA
// via the kind::f8f6f4 universal-fragment + sparse path).
//
// Tier=B (sm_120a-only). Cannot run on a sm_120 baseline device or on
// Hopper. R7 rewrite: the host driver pre-quantizes seeded FP32
// inputs to packed FP8 E4M3 bytes (kind::f8f6f4 .e4m3 layout per PTX
// ISA `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md
// :18459-18467`: 1 byte per element with **all 8 bits [7:0] used**;
// no padding mask). A is also 2:4-sparse-compressed to [M, K/2]
// keeping K-indices `{4g+0, 4g+1}`; B remains dense at full K. The
// kernel reads them directly through `const uint8_t *` operand
// pointers. `BL-20260425-kind-f8f6f4-byte-placement` codifies the
// per-dtype byte-position rule (FP8 = no padding mask, distinct from
// FP4 / FP6).
//
// Per the sparse mma contract (PTX ISA
// `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md:19397-19522`,
// section 9.7.14.6 "Matrix multiply-accumulate operation using mma.sp
// instruction with sparse matrix A"): the metadata
// register tells the hardware which K-positions of B to pair with
// A's kept positions. For keep pattern `{4g+0, 4g+1}`, the metadata
// nibble per 4-group is `0x4`; the full 32-bit metadata is
// `0x44444444` (covers 8 4-groups = 32 K-positions = the per-thread
// K-coverage in this fragment).
//
// 2:4 keep pattern: keep K-indices {4g+0, 4g+1}; zero {4g+2, 4g+3}.
// Per-thread metadata: every 4-bit group is `(1 << 2) | 0 = 0x4`,
// so meta = 0x44444444.
//
// Per-thread fragment layout (m16n8k64 .e4m3 sparse via kind::f8f6f4):
//   lane = 0..31; groupID = lane / 4; tidg = lane & 3.
//   A_sparse (4 x b32 per thread; A is compressed to K_compressed = K/2
//   along the K axis, so per-thread payload is the kept-positions of
//   16 rows × 32 K-compressed FP8 = 512 FP8 / 32 = 16 FP8 = 4 b32):
//     a0: row=groupID,   K_compressed-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     a1: row=groupID+8, K_compressed-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     a2: row=groupID,   K_compressed-byte=[tidg*4 + 16 .. tidg*4 + 19]
//     a3: row=groupID+8, K_compressed-byte=[tidg*4 + 16 .. tidg*4 + 19]
//   B (4 x b32 per thread; B is dense at full K=64 with K-byte stride
//   matching the 2:4-mask kept positions — the metadata register tells
//   the hardware which K-positions of B to pair with A's kept
//   positions): 8 cols × 64 FP8 = 512 FP8 / 32 = 16 FP8 = 4 b32:
//     b0: col=groupID, K-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     b1: col=groupID, K-byte=[tidg*4 + 16 .. tidg*4 + 19]
//     b2: col=groupID, K-byte=[tidg*4 + 32 .. tidg*4 + 35]
//     b3: col=groupID, K-byte=[tidg*4 + 48 .. tidg*4 + 51]
//   D (4 x f32 per thread): row=groupID/+8, col=tidg*2/+1.
//
// Launch: grid = (M/16, N/8, 1), block = (32, 1, 1). One warp per
// [16 x 8] output tile.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void sparse_2to4_fp8_gemm_kernel(
    const uint8_t *__restrict__ A_packed,  // [M, K/2] row-major; 1 byte per FP8 (compressed: kept K-positions only)
    const uint8_t *__restrict__ B_packed,  // [N, K] col-major; 1 byte per FP8 (dense full K)
    float         *__restrict__ C,          // [M, N] row-major (FP32 sink)
    int32_t M, int32_t N, int32_t K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const uint32_t meta = 0x44444444u;

    // A is K-compressed (K_packed_bytes per row of A = K / 2).
    // B is dense at full K (B_packed_bytes per col of B = K).
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

        uint32_t a0 = *reinterpret_cast<const uint32_t *>(A_packed + a_row0_base + a_off);
        uint32_t a1 = *reinterpret_cast<const uint32_t *>(A_packed + a_row1_base + a_off);
        uint32_t a2 = *reinterpret_cast<const uint32_t *>(A_packed + a_row0_base + a_off + 16);
        uint32_t a3 = *reinterpret_cast<const uint32_t *>(A_packed + a_row1_base + a_off + 16);
        uint32_t b0 = *reinterpret_cast<const uint32_t *>(B_packed + b_col_base  + b_off);
        uint32_t b1 = *reinterpret_cast<const uint32_t *>(B_packed + b_col_base  + b_off + 16);
        uint32_t b2 = *reinterpret_cast<const uint32_t *>(B_packed + b_col_base  + b_off + 32);
        uint32_t b3 = *reinterpret_cast<const uint32_t *>(B_packed + b_col_base  + b_off + 48);

        asm volatile(
            "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "r"(meta));
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col  = tile_n * 8  + tidg * 2;
    if (c_row0 < M && c_col + 0 < N) C[c_row0 * N + c_col + 0] = d0;
    if (c_row0 < M && c_col + 1 < N) C[c_row0 * N + c_col + 1] = d1;
    if (c_row1 < M && c_col + 0 < N) C[c_row1 * N + c_col + 0] = d2;
    if (c_row1 < M && c_col + 1 < N) C[c_row1 * N + c_col + 1] = d3;
}

extern "C" void sparse_2to4_fp8_gemm_launch(
    const uint8_t *d_A_packed, const uint8_t *d_B_packed, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    sparse_2to4_fp8_gemm_kernel<<<grid, block, 0, stream>>>(d_A_packed, d_B_packed, d_C, M, N, K);
}
