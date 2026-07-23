// slug: nvfp4_blockscale_gemm
// tier: B
// arch_gate: sm_120f
//
// Block-scaled NVFP4 GEMM via warp-level
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
//        .scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
//
// This is the real hardware path per BL-20260424-kernel-contract-vs-shape-match.
// The built cubin contains `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`
// (the Blackwell rename of the pre-Blackwell `HMMA.*.BS.*` pattern),
// which is the delivery test for this row.  A scalar FP4 simulation
// is explicitly rejected by the R18-review directive.
//
// Layout conventions
// ==================
//
// Operand A: row-major, M rows x K columns (K innermost). Packed into
//   A_packed[M][K/2] bytes.  Within each byte, bits [3:0] hold the FP4
//   at the even K-position and bits [7:4] hold the FP4 at the odd
//   K-position; a uint32 loaded from 4 contiguous bytes therefore has
//   FP4(k=0) in bits [3:0], FP4(k=1) in [7:4], ..., FP4(k=7) in [31:28]
//   (little-endian-per-nibble).
//
// Operand B: column-major (since the PTX MMA is `.col`), N rows x K cols
//   stored as B_packed[N][K/2] — each column's K-stream is contiguous.
//
// Operand C/D: row-major, M rows x N cols, FP32.
//
// Block-scaling (per-16-element block, UE4M3 scales)
// ==================================================
//
// For `.scale_vec::4X.ue4m3`, the PTX ISA (9.7.14.5.14) requires
// `{byte-id-a, thread-id-a, byte-id-b, thread-id-b}` = `{0, 0, 0, 0}`.
// Within each quad of 4 threads (groupID × 4 lanes), threads with
// `lane % 4 in {0, 1}` contribute the A scale bytes, and the thread
// with `lane % 4 == 0` contributes the B scale byte.  Each contributing
// thread's full 4-byte `scaleAData` (resp. `scaleBData`) b32 register
// holds 4 scale values — one per K-block of the m16n8k64 MMA
// (K=64 split into four 16-element blocks).  Non-contributing threads'
// scale registers are not read but must carry some legal value (zero is
// fine).
//
// The per-(row, block) scale assignment used here:
//
//   lane=(groupID*4 + 0).scaleAData[b]  <-  scale_A[row=groupID  ][block=k0/16 + b]
//   lane=(groupID*4 + 1).scaleAData[b]  <-  scale_A[row=groupID+8][block=k0/16 + b]
//   lane=(groupID*4 + 0).scaleBData[b]  <-  scale_B[col=groupID  ][block=k0/16 + b]
//
// for b = 0..3. That gives 8 scale values per (groupID) quad for A and
// 4 per quad for B, totalling 64 bytes across the warp for A (16 rows x
// 4 blocks) and 32 bytes for B (8 cols x 4 blocks) — matching exactly
// the scale budget for a single m16n8k64 MMA.
//
// Per-matrix scale computation (host side): for each 16-element block,
// `scale = max(|x_i|) / 6.0` (FP4's absolute maximum is 6.0).  This
// gives per-block tightness close to FP8 precision, matching the NVFP4
// tolerance band (rtol=atol=0.1).
//
// Round-19 authoring note on PTX scale-vec size
// =============================================
//
// The R18-review directive wrote
// `mma.sync.kind::mxf4nvf4.block_scale.scale_vec::1X`.  Per PTX ISA
// 9.7.14.5.14, `.scale_vec::1X` is defined only for `.kind::mxf8f6f4`;
// for `.kind::mxf4nvf4` the valid sizes are `.scale_vec::2X` (block32)
// and `.scale_vec::4X` (block16).  NVFP4's canonical block16 + ue4m3
// scale pairs with `.scale_vec::4X`, which this kernel uses.  Either
// choice lowers to the same `OMMA.SF.*` SASS class that the delivery
// contract (BL-20260424-kernel-contract-vs-shape-match) requires.
//
// Per-thread fragment layout (m16n8k64, .e2m1)
// ============================================
//
// Warp = 32 threads.  Let `groupID = lane >> 2`, `tidg = lane & 3`.
//
//   A (4 x b32 per thread, 32 FP4 = 16 bytes):
//     a0: rows={groupID}  ,  K={tidg*8 .. tidg*8 + 7}
//     a1: rows={groupID+8},  K={tidg*8 .. tidg*8 + 7}
//     a2: rows={groupID}  ,  K={tidg*8 + 32 .. tidg*8 + 39}
//     a3: rows={groupID+8},  K={tidg*8 + 32 .. tidg*8 + 39}
//
//   B (2 x b32 per thread, 16 FP4 = 8 bytes):  (B is `.col`)
//     b0: col={groupID},  K={tidg*8 .. tidg*8 + 7}
//     b1: col={groupID},  K={tidg*8 + 32 .. tidg*8 + 39}
//
//   D (4 x f32 per thread, 4 output cells):
//     d0: row={groupID}  ,  col={tidg*2 + 0}
//     d1: row={groupID}  ,  col={tidg*2 + 1}
//     d2: row={groupID+8},  col={tidg*2 + 0}
//     d3: row={groupID+8},  col={tidg*2 + 1}
//
// Launch geometry
// ===============
//
//   grid  = (ceil(M/16), ceil(N/8), 1)    (one warp per [16 x 8] output tile)
//   block = (32, 1, 1)                    (one warp)
//
// M, N, K must satisfy M % 16 == 0, N % 8 == 0, K % 64 == 0 on shape
// points that actually launch the kernel (correctness_00, timing_00).

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void nvfp4_blockscale_gemm_kernel(
    const uint8_t *__restrict__ A_packed,   // [M, K/2] row-major
    const uint8_t *__restrict__ B_packed,   // [N, K/2] col-major (by N)
    const uint8_t *__restrict__ scale_A,    // [M, K/16] UE4M3 per-block scales, row-major
    const uint8_t *__restrict__ scale_B,    // [N, K/16] UE4M3 per-block scales, col-major-by-N
    float         *__restrict__ C,          // [M, N] row-major
    int32_t        M,
    int32_t        N,
    int32_t        K)
{
    const int tile_m  = blockIdx.x;     // 0 .. M/16 - 1
    const int tile_n  = blockIdx.y;     // 0 .. N/8  - 1
    const int lane    = threadIdx.x;    // 0 .. 31
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    const int num_blocks_K = K >> 4;    // K / 16

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    const int K_half = K >> 1;
    const int a_row0 = tile_m * 16 + groupID;
    const int a_row1 = tile_m * 16 + groupID + 8;
    const int b_col  = tile_n * 8  + groupID;
    const int a_row0_base = a_row0 * K_half;
    const int a_row1_base = a_row1 * K_half;
    const int b_col_base  = b_col  * K_half;

    for (int k0 = 0; k0 < K; k0 += 64) {
        const int k_byte    = k0 >> 1;          // byte offset into K-stream
        const int block_k0  = k0 >> 4;          // base block index: blocks {block_k0 .. block_k0+3}
        const int a_off     = k_byte + tidg * 4;
        const int b_off     = k_byte + tidg * 4;

        // A, B fragment loads.
        uint32_t a0, a1, a2, a3, b0, b1;
        a0 = *reinterpret_cast<const uint32_t *>(A_packed + a_row0_base + a_off       );
        a1 = *reinterpret_cast<const uint32_t *>(A_packed + a_row1_base + a_off       );
        a2 = *reinterpret_cast<const uint32_t *>(A_packed + a_row0_base + a_off + 16  );
        a3 = *reinterpret_cast<const uint32_t *>(A_packed + a_row1_base + a_off + 16  );
        b0 = *reinterpret_cast<const uint32_t *>(B_packed + b_col_base  + b_off       );
        b1 = *reinterpret_cast<const uint32_t *>(B_packed + b_col_base  + b_off + 16  );

        // Per-thread scale loads: tidg=0 provides A-scales for row=groupID and
        // B-scales for col=groupID; tidg=1 provides A-scales for row=groupID+8;
        // tidg=2, 3 write zero (they don't contribute under the PTX scale-vec::4X
        // distribution with {byte-id-a=0, thread-id-a=0, ...}).
        // Experimental byte → block mapping: the fragment register
        // order (a0, a1, a2, a3) interleaves K-block indices as
        // [0, 0, 2, 2] (a0/a1 cover K_offset=0 = block k0/16 + 0,
        // a2/a3 cover K_offset=32 = block k0/16 + 2).  Byte 0 → block 0,
        // byte 1 → block 1, byte 2 → block 2, byte 3 → block 3 is the
        // "linear" mapping we try first.
        uint32_t sA = 0u, sB = 0u;
        if (tidg == 0) {
            const uint8_t *p = scale_A + a_row0 * num_blocks_K + block_k0;
            sA = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            const uint8_t *q = scale_B + b_col * num_blocks_K + block_k0;
            sB = (uint32_t)q[0] | ((uint32_t)q[1] << 8) |
                 ((uint32_t)q[2] << 16) | ((uint32_t)q[3] << 24);
        } else if (tidg == 1) {
            const uint8_t *p = scale_A + a_row1 * num_blocks_K + block_k0;
            sA = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        }

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, "
            "{%4,%5,%6,%7}, "
            "{%8,%9}, "
            "{%0,%1,%2,%3}, "
            "%10, {0, 0}, "
            "%11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sA), "r"(sB));
    }

    const int c_col = tile_n * 8 + tidg * 2;
    C[a_row0 * N + c_col + 0] = d0;
    C[a_row0 * N + c_col + 1] = d1;
    C[a_row1 * N + c_col + 0] = d2;
    C[a_row1 * N + c_col + 1] = d3;
}

extern "C" void nvfp4_blockscale_gemm_launch(
    const uint8_t *d_A_packed,
    const uint8_t *d_B_packed,
    const uint8_t *d_scale_A,
    const uint8_t *d_scale_B,
    float         *d_C,
    int            M,
    int            N,
    int            K,
    cudaStream_t   stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    nvfp4_blockscale_gemm_kernel<<<grid, block, 0, stream>>>(
        d_A_packed, d_B_packed, d_scale_A, d_scale_B, d_C, M, N, K);
}
