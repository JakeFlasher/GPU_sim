// slug: tma_tensor_copy_2d
// tier: P
// arch_gate: sm_90a sm_120
//
// 2-D tile copy via TMA (`cp.async.bulk.tensor`). The reference
// (`references/tma.py`) is an identity copy of a seeded [rows, cols]
// BF16 tensor, so this kernel loads a single (rows, cols) TMA tile into
// shared memory via a `cp.async.bulk.tensor.2d.shared::cta.global.tile`
// PTX, synchronizes on the completion mbarrier, fences the async proxy,
// and stores it back with `cp.async.bulk.tensor.2d.global.shared::cta.tile`.
// The two tensor-maps are constructed host-side via `cuTensorMapEncodeTiled`
// and passed in as `__grid_constant__` parameters so the kernel observes
// the tensor-map via the tensormap proxy; see the PTX ISA sections on
// `cp.async.bulk.tensor` and tensor maps.
//
// SASS-level evidence: the built cubin contains `UTMALDG` / `UTMASTG`
// mnemonics (or the sm_120 equivalents) — checked via `cuobjdump
// --dump-sass <slug>_run`. That is the proof required by R11 that
// the binary actually uses TMA, not a plain global load/store fallback.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Inline-PTX helpers for mbarrier + TMA. We use raw PTX (rather than the
// <cuda/barrier> C++ helpers) so the instruction shape is fully visible in
// the built cubin for the SASS evidence check.
static __device__ __forceinline__ void mbarrier_init(uint64_t *bar, uint32_t expected_arrivals) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(bar_addr), "r"(expected_arrivals));
}

static __device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *bar, uint32_t tx_count) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar_addr), "r"(tx_count));
}

static __device__ __forceinline__ void mbarrier_wait(uint64_t *bar, uint32_t phase) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier_wait_loop: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra.uni mbarrier_wait_done;\n"
        "bra.uni mbarrier_wait_loop;\n"
        "mbarrier_wait_done:\n"
        "}"
        :: "r"(bar_addr), "r"(phase));
}

static __device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;");
}

static __device__ __forceinline__ void tma_load_2d(
    __nv_bfloat16 *smem, const CUtensorMap *tensor_map, int32_t x, int32_t y, uint64_t *bar)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t bar_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"(tensor_map), "r"(x), "r"(y), "r"(bar_addr)
        : "memory");
}

static __device__ __forceinline__ void tma_store_2d(
    const CUtensorMap *tensor_map, int32_t x, int32_t y, const __nv_bfloat16 *smem)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
        "[%0, {%1, %2}], [%3];"
        :: "l"(tensor_map), "r"(x), "r"(y), "r"(smem_addr)
        : "memory");
}

static __device__ __forceinline__ void tma_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}

static __device__ __forceinline__ void tma_wait_group_read() {
    asm volatile("cp.async.bulk.wait_group.read 0;");
}

// Tile size. BF16 = 2 bytes × 64 × 64 = 8192 bytes per tile. The
// CUtensorMap's box dims at the host must match these; the TMA box size
// is encoded into the map and is what `cp.async.bulk.tensor` transfers
// per issue regardless of the larger tensor shape. Larger correctness /
// trace / timing surfaces are covered by launching multiple blocks, one
// per 64×64 tile — each block issues a single TMA load + store against
// its own (blockIdx.x, blockIdx.y) tile coordinate.
constexpr int kTileRows = 64;
constexpr int kTileCols = 64;

extern "C" __global__ void tma_tensor_copy_2d_kernel(
    const __grid_constant__ CUtensorMap tensor_map_in,
    const __grid_constant__ CUtensorMap tensor_map_out)
{
    __shared__ alignas(128) __nv_bfloat16 smem[kTileRows * kTileCols];
    __shared__ alignas(8) uint64_t bar;

    // Element coordinates in the full input / output tensors. The
    // CUtensorMap encodes per-box transfer size (64 × 64 BF16); these
    // coords select which box this block transfers.
    const int tile_col = blockIdx.x * kTileCols;
    const int tile_row = blockIdx.y * kTileRows;

    if (threadIdx.x == 0) {
        mbarrier_init(&bar, 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_expect_tx(&bar, kTileRows * kTileCols * sizeof(__nv_bfloat16));
        tma_load_2d(smem, &tensor_map_in, tile_col, tile_row, &bar);
    }
    mbarrier_wait(&bar, 0);
    __syncthreads();

    if (threadIdx.x == 0) {
        fence_proxy_async_shared_cta();
        tma_store_2d(&tensor_map_out, tile_col, tile_row, smem);
        tma_commit_group();
        tma_wait_group_read();
    }
}

extern "C" void tma_tensor_copy_2d_launch(
    const CUtensorMap *tensor_map_in,
    const CUtensorMap *tensor_map_out,
    int rows, int cols,
    cudaStream_t stream)
{
    // One block per 64×64 tile. 32 threads (one warp) is enough since
    // only thread 0 issues the TMA instructions. Callers MUST pass
    // (rows, cols) that are multiples of kTileRows / kTileCols — the
    // host driver enforces that before building the CUtensorMap.
    dim3 grid(cols / kTileCols, rows / kTileRows, 1);
    dim3 block(32, 1, 1);
    tma_tensor_copy_2d_kernel<<<grid, block, 0, stream>>>(*tensor_map_in, *tensor_map_out);
}
