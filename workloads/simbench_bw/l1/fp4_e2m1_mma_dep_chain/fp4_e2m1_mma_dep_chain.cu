// slug: fp4_e2m1_mma_dep_chain
// tier: B
// arch_gate: sm_120a
//
// Blackwell FP4 (E2M1) dependent-MMA chain via warp-level inline PTX
//   mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32
//
// Per BL-20260424-kernel-contract-vs-shape-match the inline PTX is
// the contract enforcement; cubin must contain QMMA.16832.F32.E2M1.E2M1
// SASS (the Blackwell-consumer class for FP4 warp-level MMA without
// block_scale; the block-scaled NVFP4 path is the separate
// `nvfp4_blockscale_gemm` row).
//
// Tier=B (sm_120a-only). Cannot run on a sm_120 baseline device or on
// Hopper. The host driver pre-quantizes seeded FP32 inputs to packed
// FP4 E2M1 bytes (R3 + R4: kind::f8f6f4 .e2m1 layout = 1
// byte per element with the FP4 nibble in the **central 4 bits
// [5:2]** and padding zeros in bits [7:6] and [1:0], padding mask
// 0xC3 per PTX ISA `references/NVIDIA_Manuals/ptx_isa/
// 09_instruction-set.md:18459-18467`) and the kernel reads them
// directly through a `const uint8_t *` operand pointer.
// `BL-20260425-kind-f8f6f4-byte-placement` codifies the per-dtype
// byte-position rule.
//
// kind::f8f6f4 fragment layout (R3 fix; the R1 layout
// assumed 2 FP4 per byte, which causes 4-byte OOB reads at K=32 and
// does not match the universal-fragment encoding the PTX ISA defines):
//
//   The kind::f8f6f4 universal fragment is sized for FP8 — 4 b32 per
//   thread for A, 2 b32 per thread for B. When used with .e2m1, each
//   byte holds one FP4 value (in bits [5:2], padding bits [7:6] and
//   [1:0] zero). So per-thread A reads 16 bytes = 16 FP4 elements per
//   K-block, distributed across 32 threads = 512 elements = exactly
//   the m16k32 fragment size. No oversubscription.
//
// Per-thread fragment layout (m16n8k32 .e2m1):
//   lane = 0..31; groupID = lane / 4; tidg = lane & 3.
//   A (4 x b32 per thread, 16 bytes / 16 FP4 elements per thread):
//     a0: row=groupID,   K-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     a1: row=groupID+8, K-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     a2: row=groupID,   K-byte=[tidg*4 + 16 .. tidg*4 + 19]
//     a3: row=groupID+8, K-byte=[tidg*4 + 16 .. tidg*4 + 19]
//   B (2 x b32 per thread, 8 bytes / 8 FP4 elements per thread; B is .col):
//     b0: col=groupID, K-byte=[tidg*4 + 0  .. tidg*4 + 3]
//     b1: col=groupID, K-byte=[tidg*4 + 16 .. tidg*4 + 19]
//   D (4 x f32 per thread): row=groupID/+8, col=tidg*2/+1.
//
//   With K-byte stride = K (1 byte per element), per row of A spans K
//   bytes. For K=32: byte offsets 0..31 are in range; tidg=3 reads
//   a2 at byte offset 28..31 (from `tidg*4 + 16` plus 4 bytes), in
//   range. For K=64: a2 at byte offset 28..31 + the next K-iter
//   advances by 32 bytes.
//
// Launch: grid = (M/16, N/8, 1), block = (32, 1, 1). One warp per
// [16 x 8] output tile. Requires M % 16 == 0, N % 8 == 0, K % 32 == 0.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void fp4_e2m1_mma_dep_chain_kernel(
    const uint8_t *__restrict__ A_packed,  // [M, K] row-major; 1 byte per FP4 (bits [5:2])
    const uint8_t *__restrict__ B_packed,  // [N, K] col-major; 1 byte per FP4 (bits [5:2])
    float         *__restrict__ C,          // [M, N] row-major (FP32 sink)
    int32_t M, int32_t N, int32_t K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // 1 byte per FP4 element under the kind::f8f6f4 layout.
    const int K_packed_bytes = K;
    const int a_row0_base = (tile_m * 16 + groupID    ) * K_packed_bytes;
    const int a_row1_base = (tile_m * 16 + groupID + 8) * K_packed_bytes;
    const int b_col_base  = (tile_n * 8  + groupID    ) * K_packed_bytes;

    for (int k0 = 0; k0 < K; k0 += 32) {
        const int k_byte = k0;
        const int a_off  = k_byte + tidg * 4;
        const int b_off  = k_byte + tidg * 4;

        uint32_t a0 = *reinterpret_cast<const uint32_t *>(A_packed + a_row0_base + a_off);
        uint32_t a1 = *reinterpret_cast<const uint32_t *>(A_packed + a_row1_base + a_off);
        uint32_t a2 = *reinterpret_cast<const uint32_t *>(A_packed + a_row0_base + a_off + 16);
        uint32_t a3 = *reinterpret_cast<const uint32_t *>(A_packed + a_row1_base + a_off + 16);
        uint32_t b0 = *reinterpret_cast<const uint32_t *>(B_packed + b_col_base  + b_off);
        uint32_t b1 = *reinterpret_cast<const uint32_t *>(B_packed + b_col_base  + b_off + 16);

        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1));
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col  = tile_n * 8  + tidg * 2;
    if (c_row0 < M && c_col + 0 < N) C[c_row0 * N + c_col + 0] = d0;
    if (c_row0 < M && c_col + 1 < N) C[c_row0 * N + c_col + 1] = d1;
    if (c_row1 < M && c_col + 0 < N) C[c_row1 * N + c_col + 0] = d2;
    if (c_row1 < M && c_col + 1 < N) C[c_row1 * N + c_col + 1] = d3;
}

extern "C" void fp4_e2m1_mma_dep_chain_launch(
    const uint8_t *d_A_packed, const uint8_t *d_B_packed, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    fp4_e2m1_mma_dep_chain_kernel<<<grid, block, 0, stream>>>(d_A_packed, d_B_packed, d_C, M, N, K);
}
