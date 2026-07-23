// slug: sparse_2to4_bf16_gemm
// tier: P
// arch_gate: sm_90a sm_120
//
// Sparse 2:4 BF16 GEMM via warp-level
//   mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32
//
// This is the real hardware path per BL-20260424-kernel-contract-vs-shape-match.
// The built cubin MUST contain an `*.SP.*` sparse-MMA SASS class
// (Blackwell `HMMA.*.SP.*` or equivalent).  A scalar "2:4-masked
// dense GEMM" placeholder is explicitly rejected.
//
// 2:4 pattern
// ===========
//
// For every K-group of 4 consecutive positions {4g+0, 4g+1, 4g+2,
// 4g+3} in matrix A, the pattern keeps positions {4g+0, 4g+1} and
// zeros positions {4g+2, 4g+3}.  The host packs the kept non-zeros
// into a compressed buffer `A_sparse[M][K/2]` (row-major) and
// generates a metadata register where every 4-bit group encodes the
// pair of kept indices `(1 << 2) | 0 = 0b0100 = 0x4` for this
// particular pattern — same bits for every group, for every thread.
//
// Per-thread fragment layout (m16n8k32, .bf16)
// ============================================
//
// Warp = 32 threads.  Let `groupID = lane >> 2`, `tidg = lane & 3`.
//
//   A (compressed) (4 x b32 per thread, 8 BF16 total):
//     a0: row={groupID},   K_compressed={tidg*2, tidg*2+1}
//     a1: row={groupID+8}, K_compressed={tidg*2, tidg*2+1}
//     a2: row={groupID},   K_compressed={tidg*2+8, tidg*2+9}
//     a3: row={groupID+8}, K_compressed={tidg*2+8, tidg*2+9}
//
//   B (dense, `.col`) (4 x b32 per thread, 8 BF16 total):
//     b0: col={groupID}, rows={tidg*2, tidg*2+1}
//     b1: col={groupID}, rows={tidg*2+8, tidg*2+9}
//     b2: col={groupID}, rows={tidg*2+16, tidg*2+17}
//     b3: col={groupID}, rows={tidg*2+24, tidg*2+25}
//
//   D (4 x f32 per thread):
//     d0: row={groupID},   col={tidg*2 + 0}
//     d1: row={groupID},   col={tidg*2 + 1}
//     d2: row={groupID+8}, col={tidg*2 + 0}
//     d3: row={groupID+8}, col={tidg*2 + 1}
//
// Sparse metadata
// ===============
//
// The PTX `mma.sp` takes a 32-bit `%Re` register per thread.  Each
// 4-bit nibble encodes the pair of kept indices for one K-group
// the thread touches: `(idx1 << 2) | idx0`.  For our uniform {0, 1}
// pattern, every nibble is `0b0100 = 0x4`, so the full per-thread
// metadata is `0x44444444`.  The sparse selector immediate (0x1
// for m16n8k32 BF16 per the PTX example) indicates which half of a
// thread-pair supplies the metadata; the uniform broadcast pattern
// tolerates either value.
//
// Launch geometry
// ===============
//
//   grid  = (ceil(M/16), ceil(N/8), 1)   one warp per [16 x 8] tile
//   block = (32, 1, 1)                   one warp
//
// Must satisfy M % 16 == 0, N % 8 == 0, K % 32 == 0 on shape points
// that actually launch the kernel.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void sparse_2to4_bf16_gemm_kernel(
    const __nv_bfloat16 *__restrict__ A_sparse,   // [M, K/2] compressed (kept non-zeros), row-major
    const __nv_bfloat16 *__restrict__ B,          // [N, K] col-major (B `.col`)
    float               *__restrict__ C,          // [M, N] row-major
    int32_t              M,
    int32_t              N,
    int32_t              K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // Every 4-bit group encodes kept-index pair (1 << 2) | 0 = 0x4.
    // The per-thread metadata register is the same for every thread
    // under the uniform {0, 1} keep mask.
    const uint32_t meta = 0x44444444u;

    const int K_half = K >> 1;                                      // compressed A stride
    const int a_row0_base = (tile_m * 16 + groupID    ) * K_half;   // row for a0, a2
    const int a_row1_base = (tile_m * 16 + groupID + 8) * K_half;   // row for a1, a3
    const int b_col_base  = (tile_n * 8  + groupID    ) * K;         // col-major B column start

    // K_tile = 32 per MMA; K_compressed = 16 per MMA over compressed A.
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int k_half = k0 >> 1;          // compressed A byte offset (K=2 per compressed column)
        // Per-thread A compressed loads.
        uint32_t a0 = *reinterpret_cast<const uint32_t *>(A_sparse + a_row0_base + k_half + tidg * 2);
        uint32_t a1 = *reinterpret_cast<const uint32_t *>(A_sparse + a_row1_base + k_half + tidg * 2);
        uint32_t a2 = *reinterpret_cast<const uint32_t *>(A_sparse + a_row0_base + k_half + tidg * 2 + 8);
        uint32_t a3 = *reinterpret_cast<const uint32_t *>(A_sparse + a_row1_base + k_half + tidg * 2 + 8);

        // Per-thread B dense loads (col-major, dense K=32 slice).
        uint32_t b0 = *reinterpret_cast<const uint32_t *>(B + b_col_base + k0 + tidg * 2);
        uint32_t b1 = *reinterpret_cast<const uint32_t *>(B + b_col_base + k0 + tidg * 2 + 8);
        uint32_t b2 = *reinterpret_cast<const uint32_t *>(B + b_col_base + k0 + tidg * 2 + 16);
        uint32_t b3 = *reinterpret_cast<const uint32_t *>(B + b_col_base + k0 + tidg * 2 + 24);

        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, "
            "{%4,%5,%6,%7}, "
            "{%8,%9,%10,%11}, "
            "{%0,%1,%2,%3}, "
            "%12, 0x0;"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "r"(meta));
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col  = tile_n * 8  + tidg * 2;
    C[c_row0 * N + c_col + 0] = d0;
    C[c_row0 * N + c_col + 1] = d1;
    C[c_row1 * N + c_col + 0] = d2;
    C[c_row1 * N + c_col + 1] = d3;
}

extern "C" void sparse_2to4_bf16_gemm_launch(
    const __nv_bfloat16 *d_A_sparse,
    const __nv_bfloat16 *d_B,
    float               *d_C,
    int                  M,
    int                  N,
    int                  K,
    cudaStream_t         stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    sparse_2to4_bf16_gemm_kernel<<<grid, block, 0, stream>>>(d_A_sparse, d_B, d_C, M, N, K);
}
