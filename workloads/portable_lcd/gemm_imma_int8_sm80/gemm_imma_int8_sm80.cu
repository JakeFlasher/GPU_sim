// slug: gemm_imma_int8_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// INT8 IMMA microbench via warp-level
//   mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
//
// Sister kernel of `gemm_hmma_bf16_sm80` and `gemm_hmma_fp16_sm80`.
// Same single-warp CTA, same 16x8 output tile, same cp.async-staged
// shared-memory pipeline; only the MMA datatype changes from
// {BF16, FP16} to INT8/INT32.
//
// SASS encoding on sm_80+ targets is `IMMA.16816.S8.S8` — distinct
// from the HMMA tensor-core mnemonics emitted by the BF16 / FP16
// sibling GEMMs — and extends suite tensor-core opcode-mix coverage
// to the integer pipeline.
//
// Per-thread MMA fragment sizes for m16n8k16 INT8:
//   A operand: 2 .b32 (8 INT8 elements per thread)
//   B operand: 1 .b32 (4 INT8 elements per thread)
//   C/D operand: 4 .s32 (4 INT32 accumulators per thread)

#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int K_TILE = 16;

__device__ inline uint32_t cp_async_smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
} // namespace

extern "C" __global__ void gemm_imma_int8_sm80_kernel(
    const int8_t  *__restrict__ A, // [M, K] row-major
    const int8_t  *__restrict__ B, // [N, K] col-major (B is .col)
    int32_t       *__restrict__ C, // [M, N] row-major
    int32_t M, int32_t N, int32_t K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    __shared__ int8_t sA[16 * K_TILE];
    __shared__ int8_t sB[8  * K_TILE];

    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    const int gA_row_base = tile_m * 16;
    const int gB_row_base = tile_n * 8;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        // Stage A: 32 cp.async ops, 8 bytes each, covering 16x16 INT8 = 256 B.
        // Each row of A is 16 INT8 = 16 B; lane (row<<1 | half) loads 8 B.
        {
            const int row     = lane >> 1;
            const int col     = (lane & 1) << 3;
            const int gmem_el = (gA_row_base + row) * K + k0 + col;
            const int smem_el = row * K_TILE + col;
            uint32_t s_addr = cp_async_smem_addr(&sA[smem_el]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 8;\n"
                :: "r"(s_addr), "l"(A + gmem_el));
        }

        // Stage B: first 16 lanes load 8x16 INT8 = 128 B / 8 B per op = 16 ops.
        if (lane < 16) {
            const int row     = lane >> 1;
            const int col     = (lane & 1) << 3;
            const int gmem_el = (gB_row_base + row) * K + k0 + col;
            const int smem_el = row * K_TILE + col;
            uint32_t s_addr = cp_async_smem_addr(&sB[smem_el]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 8;\n"
                :: "r"(s_addr), "l"(B + gmem_el));
        }

        asm volatile("cp.async.commit_group;\n" :: );
        asm volatile("cp.async.wait_all;\n"     :: );
        __syncthreads();

        // m16n8k16 INT8 A fragment: 2 .b32 per thread.
        //   a0: row=groupID,    cols=4*tidg..4*tidg+3
        //   a1: row=groupID+8,  cols=4*tidg..4*tidg+3
        const int a_row0 = groupID;
        const int a_row1 = groupID + 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(&sA[a_row0 * K_TILE + tidg * 4]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t *>(&sA[a_row1 * K_TILE + tidg * 4]);

        // m16n8k16 INT8 B fragment: 1 .b32 per thread.
        //   b0: col=groupID, ks=4*tidg..4*tidg+3 (col-major layout)
        const int b_row  = groupID;
        const uint32_t b0 = *reinterpret_cast<const uint32_t *>(&sB[b_row * K_TILE + tidg * 4]);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(b0));

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

extern "C" void gemm_imma_int8_sm80_launch(
    const int8_t *d_A, const int8_t *d_B, int32_t *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    gemm_imma_int8_sm80_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}
