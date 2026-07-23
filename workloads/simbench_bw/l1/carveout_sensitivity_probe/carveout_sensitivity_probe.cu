// slug: carveout_sensitivity_probe
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Carveout-sensitivity microbench: each launch sets a per-kernel
// preferred-shared-carveout split via
// `cudaFuncSetAttribute` /
// `cudaFuncAttributePreferredSharedMemoryCarveout` before
// launching the kernel; the host driver sweeps `carveout_percent`
// across the supported partition points (typically 0/25/50/75/100
// percent) so the HRD-profiler criterion can observe how the
// L1/shared partition ratio shifts the effective hit-rate
// distribution.
//
// R29 rewrite: replaces the prior single prime-stride global-read
// loop with four explicit layered-reuse phases. Each warp owns a
// distinct anchor cache line in the global allocation; the kernel
// stages a multi-KiB carveout-bounded shared tile (unchanged from
// the prior version so the L1/shared partition pressure remains
// observable) and then runs the four phases below. Each phase
// ends with an anchor revisit, producing reuse-distance
// observations that concentrate densely in a small set of HRD
// bins so the producer-side HRD profile clears
// MINIMUM_BIN_SAMPLE_COUNT=50 with margin and admits a scalar
// bundle.
//   Stage A (sector-rd target ~ 7): 16 cycles of
//       [per-warp distinct line, anchor-revisit].
//   Stage B (sector-rd target ~ 1027): 1 cycle of
//       [256 shared distinct lines, anchor-revisit].
//   Stage C (sector-rd target ~ 4099): 1 cycle of
//       [1024 shared distinct lines, anchor-revisit].
//   Stage D (preserves carveout sensitivity): 16 cycles of
//       [shared-tile read, anchor-revisit]; the shared-tile
//       reads keep the L1/shared partition observable on the
//       trace.
// The launch geometry is unchanged (threads=32, blocks=warps).
// All 32 threads of each warp coalesce on each line touch.
// Stage iteration counts are sized so the NVBit-instrumented
// trace runtime stays under the standard retrace timeout while
// leaving each target bin with margin over the bin-admission
// floor.
//
// The carveout preference is hardware-advisory; on architectures
// that do not support per-kernel shared-carveout selection (the
// runtime advertises it via
// `cudaDevAttrMaxSharedMemoryPerBlockOptin`), the host driver
// emits a structured typed-unavailable JSON document rather than
// silently running the kernel with a default partition.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void carveout_sensitivity_probe_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    float *__restrict__ sink,
    int32_t allocation_floats, int32_t accesses, int32_t warps,
    int32_t shared_floats)
{
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const int warp_id = blockIdx.x;

    // Stage `shared_floats` words of the input into shared
    // memory. `shared_floats` is bounded above by the dynamic-
    // shared budget the runtime grants given the per-kernel
    // carveout preference; on sm_86 the supported partition
    // points typically cap the dynamic-shared budget at the
    // carveout-bounded amount. `#pragma unroll 1` keeps each
    // staging iteration a strictly 1:1 (read, write) pair so the
    // alloc-1 reads do not get reordered into 4-deep groups by
    // the compiler. Without it, the staging loop emits four
    // global reads then four shared writes, which introduces 31
    // observations in alloc bin [4,7] from the warp-load lane-0
    // (sector-cold, line-cold, alloc-finite) classification path
    // and underconfigures the bundle.
    int stage = shared_floats;
    if (stage < 32) stage = 32;
    if (stage > allocation_floats) stage = allocation_floats;
    #pragma unroll 1
    for (int i = tid; i < stage; i += lanes) {
        smem[i] = in[i];
    }
    __syncthreads();

    // Cache-line geometry: each cache line is 128 bytes = 32 fp32
    // entries. Coalesced 32-thread warp load = 1 line = 4 sectors.
    const int32_t LINE_FLOATS = 32;

    // Per-warp anchor line: warp_id-th cache line.
    const int32_t anchor_line = warp_id;
    // Stage A's per-warp distinct lines start after the anchor
    // band; each warp owns a contiguous 16-line slot.
    const int32_t PHASE_A_BASE_LINE = (int32_t)warps;
    // Stage B's shared 256-line band starts after Stage A.
    const int32_t PHASE_B_BASE_LINE = PHASE_A_BASE_LINE + 16 * warps;
    // Stage C's shared 1024-line band starts after Stage B.
    const int32_t PHASE_C_BASE_LINE = PHASE_B_BASE_LINE + 256;
    // The trace_l2_accuracy_xl shape ships with
    // allocation_bytes=16777216 => 4194304 fp32 entries =>
    // 131072 cache lines. Stage C ends at line
    // PHASE_C_BASE_LINE + 1024 = warps + 16*warps + 256 + 1024
    // = 1280 + 17*warps; for warps<=64 this stays well under
    // 131072. Smaller shapes collapse via the modular wrap below;
    // their HRD bin distribution is not the underconfig target.

    auto thread_addr_from_line =
        [allocation_floats, LINE_FLOATS, tid](int32_t line)
            -> int32_t {
        int32_t base = line * LINE_FLOATS;
        int32_t off = base + tid;
        if (allocation_floats <= 0) return 0;
        if (off < 0) off = 0;
        off %= allocation_floats;
        return off;
    };

    float acc = 0.0f;
    const int32_t anchor_addr = thread_addr_from_line(anchor_line);

    // Initial anchor touch.
    acc += in[anchor_addr];

    // Stage A: 16 cycles of (per-warp distinct line, anchor
    // revisit). Sector-rd at the first lane of each revisit
    // sector lands in bin [4,7]; 16 cycles * 64 warps * 4 sectors
    // = 4096 samples with 80x margin above the bin-admission
    // floor.
    for (int cycle = 0; cycle < 16; ++cycle) {
        int32_t distinct_line = PHASE_A_BASE_LINE
            + warp_id * 16 + cycle;
        int32_t distinct_addr =
            thread_addr_from_line(distinct_line);
        acc += in[distinct_addr];
        acc += in[anchor_addr];
    }

    // Stage B: 256 shared distinct lines, then one anchor
    // revisit. Sector-rd ~ 1027; 64 warps * 4 sectors = 256
    // samples land near the bin [1024, 2047] floor with margin.
    for (int sweep = 0; sweep < 256; ++sweep) {
        int32_t distinct_line = PHASE_B_BASE_LINE + sweep;
        int32_t distinct_addr =
            thread_addr_from_line(distinct_line);
        acc += in[distinct_addr];
    }
    acc += in[anchor_addr];

    // Stage C: 1024 shared distinct lines, then one anchor
    // revisit. Sector-rd ~ 4099; 64 warps * 4 sectors = 256
    // samples land near bin [4096, 8191] with margin.
    for (int sweep = 0; sweep < 1024; ++sweep) {
        int32_t distinct_line = PHASE_C_BASE_LINE + sweep;
        int32_t distinct_addr =
            thread_addr_from_line(distinct_line);
        acc += in[distinct_addr];
    }
    acc += in[anchor_addr];

    // Stage D: 16 cycles of (shared-staged read, anchor
    // revisit). The shared reads keep the L1/shared carveout
    // partition observable in the trace; the anchor revisits
    // continue to feed sector bin [4,7] (each cycle's anchor
    // revisit sits 1 cache-line worth of intervening shared
    // accesses away from the previous anchor touch, replicating
    // the Stage A revisit pattern).
    int smem_idx = tid;
    for (int cycle = 0; cycle < 16; ++cycle) {
        int s = smem_idx % stage;
        acc += smem[s];
        acc += in[anchor_addr];
        smem_idx += lanes;
    }

    // Sink write restricted to block 0's tid 0 so multi-block
    // launches (warps > 1, which makes blocks = warps in the
    // launch wrapper below) do not race on `sink[0]`. The
    // kernel uses block-local `tid = threadIdx.x` for its per-CTA
    // observation pattern; gating the scalar sink store on
    // `blockIdx.x == 0` keeps a single deterministic store
    // without disturbing the per-CTA work distribution. Codex
    // review-phase P2 fix (multi-block sink race).
    if (blockIdx.x == 0 && tid == 0) sink[0] = acc;
    __syncthreads();
    // Identity output: each warp emits a single coalesced
    // warp-load to its own (warp_id, tid) slot in `out`,
    // writing the anchor-line value the lane just read. The
    // single-warp-load shape (instead of a strided identity
    // loop over the entire `out` buffer) keeps the identity-
    // output access pattern from spreading samples thinly
    // across mid-range HRD bins on the trace_l2_accuracy_xl
    // shape.
    out[warp_id * lanes + tid] = in[anchor_addr];
}

extern "C" void carveout_sensitivity_probe_launch(
    const float *d_in, float *d_out, float *d_sink,
    int allocation_floats, int accesses, int warps,
    int shared_floats,
    cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps > 0 ? warps : 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    size_t smem_bytes = (size_t)shared_floats * sizeof(float);
    carveout_sensitivity_probe_kernel<<<grid, block, smem_bytes, stream>>>(
        d_in, d_out, d_sink, allocation_floats, accesses, warps,
        shared_floats);
}
