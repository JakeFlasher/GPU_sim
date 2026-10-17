// slug: gemm_sparse_2to4_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// BF16 GEMM with 2:4 structured sparsity on operand A via the
// warp-level sparse tensor-core instruction
//
//   mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32
//
// Shape: M=16, N=8, K_eff=32 per warp tile. Sparse A holds the
// non-zero half (same register footprint as dense m16n8k16: 4
// .b32 per thread) plus a 1-.b32 metadata register and a
// selector constant. Dense B stores the full K=32 strip (4 .b32
// per thread). D / C use the same 4 .f32 layout as the dense
// HMMA siblings.
//
// Sparse 2:4 metadata format: the kept-index pattern is uniform
// across every K=4 group — kept indices = {0, 2}, metadata nibble
// = (2 << 2) | 0 = 0x8, and the per-thread metadata register
// holds 0x88888888 (eight nibbles, each covering one 4-K-block).
// The selector imm is 0x0; all 32 lanes contribute the same
// metadata value, so the warp-collective metadata used by the
// instruction reduces to the uniform sm_80 layout.
//
// Same `cp.async`-staged shared-memory load and pipeline
// scaffolding as the dense BF16 sister kernel
// (gemm_hmma_bf16_sm80). On sm_120 SASS this is expected to
// emit the `HMMA.SP` family of instructions, broadening the
// suite's tensor-core opcode-mix coverage.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int K_TILE = 32;

__device__ inline uint32_t cp_async_smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
} // namespace

extern "C" __global__ void gemm_sparse_2to4_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ A_compact, // [M, K/2] row-major (sparse half)
    const __nv_bfloat16 *__restrict__ B,         // [N, K] col-major
    float               *__restrict__ C,         // [M, N] row-major
    int32_t M, int32_t N, int32_t K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    // Compact A in shared mem: 16 rows x K/2 BF16 (each row holds the
    // non-zero half of the dense row).
    __shared__ __nv_bfloat16 sA[16 * (K_TILE / 2)];
    __shared__ __nv_bfloat16 sB[8  * K_TILE];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    const int gA_row_base = tile_m * 16;
    const int gB_row_base = tile_n * 8;

    // Uniform 2:4 metadata across the warp: every 4-K-block keeps
    // indices {0, 2}; nibble = (2<<2)|0 = 0x8; eight nibbles per
    // .b32 register.
    const uint32_t sp_meta = 0x88888888u;

    const int K_HALF = K / 2;

    for (int k0 = 0; k0 < K_HALF; k0 += K_TILE / 2) {
        // Stage A_compact: 16 rows * (K_TILE/2 = 16) BF16 = 32 bytes/row,
        // 16 ops * 16 bytes = 256 bytes -> use the half-row strided pattern
        // mirroring the dense HMMA siblings (one cp.async per (row, half)).
        if (lane < 32) {
            const int row     = lane >> 1;
            const int col     = (lane & 1) << 3; // 0 or 8 BF16
            const int gmem_el = (gA_row_base + row) * K_HALF + k0 + col;
            const int smem_el = row * (K_TILE / 2) + col;
            uint32_t s_addr = cp_async_smem_addr(&sA[smem_el]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(s_addr), "l"(A_compact + gmem_el));
        }

        // Stage B: 8 rows * K_TILE BF16 = 64 bytes/row, 32 ops total.
        {
            const int row     = lane >> 2;            // 0..7
            const int col     = (lane & 3) << 3;       // 0,8,16,24 BF16
            const int gmem_el = (gB_row_base + row) * K + (k0 * 2) + col;
            const int smem_el = row * K_TILE + col;
            uint32_t s_addr = cp_async_smem_addr(&sB[smem_el]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(s_addr), "l"(B + gmem_el));
        }

        asm volatile("cp.async.commit_group;\n" :: );
        asm volatile("cp.async.wait_all;\n"     :: );
        __syncthreads();

        // m16n8k32.sp A operand: 4 .b32 per thread, holding the sparse
        // half of the original dense m16n8k32 A. Layout mirrors the
        // dense m16n8k16 A pattern over the compact half-K array.
        const int a_row0 = groupID;
        const int a_row1 = groupID + 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(&sA[a_row0 * (K_TILE / 2) + tidg * 2 + 0]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t *>(&sA[a_row1 * (K_TILE / 2) + tidg * 2 + 0]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t *>(&sA[a_row0 * (K_TILE / 2) + tidg * 2 + 8]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t *>(&sA[a_row1 * (K_TILE / 2) + tidg * 2 + 8]);

        // m16n8k32 dense B operand: 4 .b32 per thread (covers full K=32 strip).
        const int b_row = groupID;
        const uint32_t b0 = *reinterpret_cast<const uint32_t *>(&sB[b_row * K_TILE + tidg * 2 + 0]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t *>(&sB[b_row * K_TILE + tidg * 2 + 8]);
        const uint32_t b2 = *reinterpret_cast<const uint32_t *>(&sB[b_row * K_TILE + tidg * 2 + 16]);
        const uint32_t b3 = *reinterpret_cast<const uint32_t *>(&sB[b_row * K_TILE + tidg * 2 + 24]);

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
              "r"(sp_meta));

        __syncthreads();
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col  = tile_n * 8  + tidg * 2;
    C[c_row0 * N + c_col + 0] = d0;
    C[c_row0 * N + c_col + 1] = d1;
    C[c_row1 * N + c_col + 0] = d2;
    C[c_row1 * N + c_col + 1] = d3;
}

extern "C" void gemm_sparse_2to4_bf16_sm80_launch(
    const __nv_bfloat16 *d_A_compact, const __nv_bfloat16 *d_B,
    float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    gemm_sparse_2to4_bf16_sm80_kernel<<<grid, block, 0, stream>>>(
        d_A_compact, d_B, d_C, M, N, K);
}
