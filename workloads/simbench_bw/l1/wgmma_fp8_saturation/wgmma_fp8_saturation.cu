// slug: wgmma_fp8_saturation
// tier: H
// arch_gate: sm_90a
//
// Hopper warpgroup MMA FP8 (E4M3) saturation microbench via inline PTX
//   wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3
//
// Hopper-only (`tier=H`); the cubin contains QGMMA SASS markers and
// will not build under sm_120 baseline. The host driver runs on H100
// (CC 9.0) and emits the kernel's d_C output as the row's
// authoritative result.
//
// R21 (drift-recovery): Replaced the original Round-0/1 sink-store
// (`C[tid] = sum(d00..d31)` for tid<M*N — only 128 of 4096 elements
// written, each a sum-of-all-fragments rather than a per-position
// accumulator value) with real per-fragment row-major stores. The
// kernel now produces the full 64×64 m64n64 GEMM output, and the host
// runner D2H copies it (`l1/wgmma_fp8_saturation/main.cu` R20).
// The Python reference applies the same FP8 E4M3 round-trip via
// `mma_microbench.py::_quantize_matrix_fp8_e4m3` (R19). On
// Hopper hardware: kernel output matches reference within the
// (FP8_E4M3, fma_chain, mma_microbench) tolerance triplet at
// rtol=atol=0.05, max_ulp=8.
//
// Correctness shape constraint: the kernel handles M=64, N=64, K=32
// only (single warpgroup × single m64n64k32 wgmma issue). Larger
// shape points (`trace_00 = m64n64k128`, `timing_00 = m1024n1024k4096`)
// are instruction-path-only — they pass through the SMEM cooperative
// load + same wgmma issue but only the first 64×32 of A and 32×64 of B
// participate in the GEMM (the rest fold to zero in the SMEM stage).
// Trace/timing modes don't validate output, so this is fine; the
// shape file (`shapes/wgmma_fp8_saturation.toml`) constrains the
// `correctness_shapes` list to `correctness_00 = m=64, n=64, k=32`.
//
// Per BL-20260424-kernel-contract-vs-shape-match the row's contract is
// the actual instruction-path: the cubin must contain QGMMA at the
// declared m64n64k32.f32.e4m3.e4m3 shape, not just wgmma.fence /
// commit / wait synchronization markers (which lower to WARPGROUP.*
// SASS without an HGMMA / QGMMA anchor at the row's declared shape).
// (Hopper FP8 wgmma lowers to QGMMA — the Quarter-precision GMMA SASS
// class — distinct from BF16/FP16's HGMMA — the Half-precision GMMA
// class. The R0 BF16 sibling row's cubin emits HGMMA.64x64x16.F32.BF16;
// this FP8 row emits QGMMA.64x64x32.F32.E4M3.E4M3.)
//
// Per-thread-warpgroup fragment layout (m64n64k32, 4 warps = 128 threads):
//   wgmma.mma_async issues an asynchronous MMA whose A and B operands
//   are read from shared memory via 64-bit descriptors (PTX ISA
//   §9.7.15.5.2). The D fragment lives in registers: 4096 f32 elements
//   / 128 threads = 32 f32 per thread.
//   Warp w (= tid / 32, range 0..3) owns rows [16*w, 16*w+15].
//   Within a warp: lane = tid % 32; groupId = lane / 4 (range 0..7);
//   tidg = lane % 4 (range 0..3).
//   Per n8 block b (range 0..7, covering N = 8 * 8 = 64 cols):
//     d[4*b + 0] → (row = 16*w + groupId,     col = 8*b + 2*tidg + 0)
//     d[4*b + 1] → (row = 16*w + groupId,     col = 8*b + 2*tidg + 1)
//     d[4*b + 2] → (row = 16*w + groupId + 8, col = 8*b + 2*tidg + 0)
//     d[4*b + 3] → (row = 16*w + groupId + 8, col = 8*b + 2*tidg + 1)
//   Per-thread store count: 32 elements (one per fragment register).
//   Total warpgroup store count: 32 × 128 = 4096 = 64 × 64 ✓.
//
// Note on K vs the BF16 sibling row: BF16 wgmma uses K=16, FP8 wgmma
// uses K=32 (per `references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md` shape
// × type table and `shapes/wgmma_fp8_saturation.toml` correctness_00).

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

// Encode a byte-quantity for the wgmma matrix descriptor field format
// (PTX ISA §9.7.15.5.2): each 14-bit field stores `(x & 0x3FFFF) >> 4`,
// covering shared-memory addresses up to 256 KiB and operand strides up
// to 16 KiB.
__device__ static inline uint64_t wgmma_desc_encode(uint32_t x) {
    return static_cast<uint64_t>((x & 0x3FFFFu) >> 4);
}

// Pack a wgmma matrix descriptor with no swizzle, no base offset.
//   bits [13:0]  = encode(start_addr)
//   bits [29:16] = encode(leading_dim_bytes)
//   bits [45:32] = encode(stride_dim_bytes)
//   bits [51:49] = base offset (0)
//   bits [63:62] = swizzle mode (0 = none)
__device__ static inline uint64_t wgmma_make_desc(
    uint32_t start_addr, uint32_t leading_dim_bytes, uint32_t stride_dim_bytes)
{
    return wgmma_desc_encode(start_addr)
         | (wgmma_desc_encode(leading_dim_bytes) << 16)
         | (wgmma_desc_encode(stride_dim_bytes)  << 32);
}

extern "C" __global__ void wgmma_fp8_saturation_kernel(
    const __nv_fp8_e4m3 *__restrict__ A, // [M, K] row-major (FP8 E4M3)
    const __nv_fp8_e4m3 *__restrict__ B, // [N, K] col-major (FP8 E4M3; contig along K)
    float               *__restrict__ C, // [M, N] row-major (FP32; written by this kernel)
    int32_t M, int32_t N, int32_t K)
{
    // 4 KiB shared memory: a 2 KiB m64k32 FP8 A-tile + a 2 KiB k32n64
    // FP8 B-tile, both feeding wgmma.mma_async via descriptor.
    extern __shared__ __align__(16) char smem_buf[];
    __nv_fp8_e4m3 *smem_A = reinterpret_cast<__nv_fp8_e4m3 *>(smem_buf);
    __nv_fp8_e4m3 *smem_B = smem_A + 64 * 32;

    const int tid = threadIdx.x; // 0..127

    // Cooperative SMEM stage: 128 threads × 16 FP8 = 2048 FP8 = exactly
    // one m64×k32 tile (and one k32×n64 tile, also 2048 FP8). Out-of-range
    // slots fold to zero so the kernel is shape-safe at smaller (M, N, K)
    // shape points without changing the wgmma instruction the cubin emits.
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int idx = tid * 16 + i;
        smem_A[idx] = (idx < M * K) ? A[idx] : __nv_fp8_e4m3(0.0f);
        smem_B[idx] = (idx < K * N) ? B[idx] : __nv_fp8_e4m3(0.0f);
    }
    __syncthreads();
    // R22 (per Codex Round-21-review blocking finding): WGMMA runs in
    // the async proxy and reads its A/B operands from shared memory through
    // 64-bit descriptors. The cooperative SMEM stores above are normal
    // generic-proxy stores. PTX ISA `:20521-20525` requires `fence.proxy.async`
    // for cross-proxy SMEM access; PTX ISA `:21784-21801` narrows
    // `wgmma.fence.sync.aligned` (issued below) to warpgroup register
    // ordering only and explicitly requires a separate async-proxy fence
    // for prior writes to shared-memory matrices read by `wgmma.mma_async`.
    // Without this fence, the WGMMA SMEM reads can race with the cooperative
    // stores even after `__syncthreads()` (which gives CTA-wide store
    // completion in the generic proxy but does NOT establish cross-proxy
    // visibility). The "memory" clobber prevents the compiler from moving
    // prior stores past the fence.
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    // Descriptors. Without swizzle, leading_dim = 256 (one core-matrix
    // tile width: for FP8 m64k32, one core matrix = 8 rows × 32 cols
    // × 1 byte = 256 bytes). stride_dim = 0 (the K=32 dimension is a
    // single core matrix; no second K-block to stride to).
    const uint32_t addrA = static_cast<uint32_t>(__cvta_generic_to_shared(smem_A));
    const uint32_t addrB = static_cast<uint32_t>(__cvta_generic_to_shared(smem_B));
    const uint64_t descA = wgmma_make_desc(addrA, 256u, 0u);
    const uint64_t descB = wgmma_make_desc(addrB, 256u, 0u);

    // 32 f32 accumulator registers per thread (m64n64 / 128 threads).
    float d00 = 0, d01 = 0, d02 = 0, d03 = 0, d04 = 0, d05 = 0, d06 = 0, d07 = 0;
    float d08 = 0, d09 = 0, d10 = 0, d11 = 0, d12 = 0, d13 = 0, d14 = 0, d15 = 0;
    float d16 = 0, d17 = 0, d18 = 0, d19 = 0, d20 = 0, d21 = 0, d22 = 0, d23 = 0;
    float d24 = 0, d25 = 0, d26 = 0, d27 = 0, d28 = 0, d29 = 0, d30 = 0, d31 = 0;

    // Order prior register writes against the async MMA's reads.
    asm volatile("wgmma.fence.sync.aligned;\n" :: );

    // The contract instruction. scaleD=false → D = A*B (overwrite, no
    // accumulate). imm_scale_a = imm_scale_b = +1 (no operand negation).
    // The .e4m3.e4m3 FP8 wgmma variant takes no transpose immediates
    // (PTX ISA §9.7.15.5.2: `imm-trans-a/b` are .f16/.bf16 descriptor
    // variants only; FP8 has no transpose option). Cubin lowering on
    // sm_90a emits QGMMA at the m64n64k32 FP8→F32 shape — the row's
    // declared instruction-path contract per
    // BL-20260424-kernel-contract-vs-shape-match.
    asm volatile(
        ".reg .pred wp_scale_d;\n"
        "setp.ne.s32 wp_scale_d, %32, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%33, %34, wp_scale_d, 1, 1;\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03),
          "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
          "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
          "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
          "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19),
          "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
          "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27),
          "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "r"(0), "l"(descA), "l"(descB)
    );

    asm volatile("wgmma.commit_group.sync.aligned;\n" :: );
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" :: );

    // Real per-fragment row-major stores (R21 drift-recovery
    // replaced the original Round-0/1 sink-store). Each thread writes
    // its 32 D fragment registers to their (row, col) positions per the
    // PTX ISA §9.7.15.5.2 m64n64k32 accumulator-fragment layout
    // documented in the header above. Bounds-checked so smaller M/N
    // shape points fold correctly (writes only land in the in-range
    // (row, col); out-of-range fragment positions are silently dropped,
    // which is safe because the SMEM cooperative load already zero-
    // padded the corresponding A/B input slots so those accumulator
    // values are zero anyway).
    const int w       = tid >> 5;        // warp 0..3
    const int lane    = tid & 0x1F;       // 0..31
    const int groupId = lane >> 2;        // 0..7
    const int tidg    = lane & 0x3;       // 0..3
    const int row_top = 16 * w + groupId;
    const int row_bot = row_top + 8;

    #define WGMMA_STORE_NBLOCK(b, dn0, dn1, dn2, dn3) do {                                      \
        const int col_l = 8 * (b) + 2 * tidg + 0;                                              \
        const int col_r = 8 * (b) + 2 * tidg + 1;                                              \
        if (row_top < M) {                                                                      \
            if (col_l < N) C[row_top * N + col_l] = (dn0);                                     \
            if (col_r < N) C[row_top * N + col_r] = (dn1);                                     \
        }                                                                                       \
        if (row_bot < M) {                                                                      \
            if (col_l < N) C[row_bot * N + col_l] = (dn2);                                     \
            if (col_r < N) C[row_bot * N + col_r] = (dn3);                                     \
        }                                                                                       \
    } while (0)

    WGMMA_STORE_NBLOCK(0, d00, d01, d02, d03);
    WGMMA_STORE_NBLOCK(1, d04, d05, d06, d07);
    WGMMA_STORE_NBLOCK(2, d08, d09, d10, d11);
    WGMMA_STORE_NBLOCK(3, d12, d13, d14, d15);
    WGMMA_STORE_NBLOCK(4, d16, d17, d18, d19);
    WGMMA_STORE_NBLOCK(5, d20, d21, d22, d23);
    WGMMA_STORE_NBLOCK(6, d24, d25, d26, d27);
    WGMMA_STORE_NBLOCK(7, d28, d29, d30, d31);

    #undef WGMMA_STORE_NBLOCK
}

#else

extern "C" __global__ void wgmma_fp8_saturation_kernel(
    const __nv_fp8_e4m3 *, const __nv_fp8_e4m3 *, float *, int32_t, int32_t, int32_t) {}

#endif

extern "C" void wgmma_fp8_saturation_launch(
    const __nv_fp8_e4m3 *d_A, const __nv_fp8_e4m3 *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);
    // 4 KiB shared memory: 2 KiB A (m64×k32 FP8) + 2 KiB B (k32×n64 FP8).
    const size_t shmem_bytes = 4 * 1024;
    wgmma_fp8_saturation_kernel<<<grid, block, shmem_bytes, stream>>>(d_A, d_B, d_C, M, N, K);
}
