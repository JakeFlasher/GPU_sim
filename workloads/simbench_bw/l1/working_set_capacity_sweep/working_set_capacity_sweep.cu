// slug: working_set_capacity_sweep
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// Working-set sweep microbench: each launch streams through a
// fresh allocation sized by `working_set_bytes`. Sweeping the
// shape axis across the L1 / L2 / DRAM capacity knees lets the
// HRD-profiler criterion expose per-layer hit-rate distributions
// that change shape at each capacity threshold.
//
// R29 rewrite: replaces the prior prime-stride-only access stream
// with explicit layered-reuse phases. Each warp owns a distinct
// anchor cache line; the kernel runs three layered phases that
// each end with the anchor revisit, producing reuse-distance
// observations that concentrate densely in a small set of HRD
// bins so the producer-side HRD profile clears
// MINIMUM_BIN_SAMPLE_COUNT=50 with margin and admits a scalar
// bundle. The phase iteration counts are sized so the
// NVBit-instrumented trace runtime stays under the standard
// retrace timeout while leaving each target bin with margin
// over the bin-admission floor:
//   Stage 1 (sector-rd target ~ 7): 16 cycles of
//       [distinct-line, anchor-revisit].
//   Stage 2 (sector-rd target ~ 4099): 1 cycle of
//       [1024 distinct lines, anchor-revisit].
//   Stage 3 (R34 resize: sector-rd target ~ 1027): 1 cycle of
//       [256 distinct lines, anchor-revisit].
// The launch geometry is unchanged (threads=32, blocks=warps).
// All 32 threads of each warp coalesce on each line touch.
//
// R31 rollback: the R30 L2-cache-wash preamble (768 lines/warp x
// 64 warps = 6 MiB = exactly sm_86 L2 capacity) was REVERTED
// after a bounded 1-cell probe FALSIFIED its label-stability
// hypothesis (best stddev rose from R29's 0.041 to R30's 0.1357,
// a 3.3x regression; 3 of 8 NCU configs produced metric-out-of-
// range failures). The falsification evidence is preserved in
// BL-20260513-l2-wash-preamble-at-l2-capacity-amplifies-label-
// variance.
//
// R34 main-pattern resize: Stage 3 shrinks from 4096 to 256
// distinct lines so the kernel's intra-launch L2 footprint
// drops from ~680 KiB to ~210 KiB (down from ~11% to ~3% of
// the sm_86 L2 capacity). Stage 3's anchor revisit now lands
// in sector_rd bin [1024, 2047] (256 distinct lines * 4
// sectors + 3 residual = 1027 sectors); this is a different
// bin from Stage 2's [4096, 8191] revisit, so Stage 2 and
// Stage 3 each contribute 256 samples to their respective
// bins. The hypothesis is that a smaller intra-launch
// footprint reduces the label-side stddev's sensitivity to
// inter-launch L2 carryover. This is the Solution path (a)
// main-pattern-resize arm from the R30 BL-20260513 lesson.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void working_set_capacity_sweep_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    float *__restrict__ sink,
    int32_t working_set_floats, int32_t accesses, int32_t warps)
{
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const int warp_id = blockIdx.x;

    // Cache-line geometry: each cache line is 128 bytes = 32 fp32
    // entries. Coalesced 32-thread warp load = 1 line = 4 sectors
    // (32-byte sectors).
    const int32_t LINE_FLOATS = 32;

    // Per-warp anchor line: warp_id-th cache line. The first
    // `warps` lines of the allocation are reserved for anchors so
    // each warp's anchor is a distinct identity.
    const int32_t anchor_line = warp_id;
    // Stage 1's per-warp distinct lines live after the anchor
    // band; each warp owns a contiguous 16-line slot so its
    // Stage-1 distinct sweep does not collide with neighbours.
    const int32_t PHASE1_BASE_LINE = (int32_t)warps;
    // Stage 2's shared distinct-line band starts after Stage 1's
    // per-warp slots: warps * 16 lines + warps for anchors.
    const int32_t PHASE2_BASE_LINE = PHASE1_BASE_LINE + 16 * warps;
    // Stage 3's shared distinct-line band starts after Stage 2.
    const int32_t PHASE3_BASE_LINE = PHASE2_BASE_LINE + 1024;

    // Compute a thread's address from a target line index.
    auto thread_addr_from_line =
        [working_set_floats, LINE_FLOATS, tid](int32_t line)
            -> int32_t {
        int32_t base = line * LINE_FLOATS;
        int32_t off = base + tid;
        if (working_set_floats <= 0) return 0;
        if (off < 0) off = 0;
        off %= working_set_floats;
        return off;
    };

    float acc = 0.0f;
    const int32_t anchor_addr = thread_addr_from_line(anchor_line);

    // Initial anchor touch (cold at every layer for warp 0; for
    // warps 1..N-1 the allocation/region trackers see the
    // allocation as already finite at this point in the per-TB
    // walk).
    acc += in[anchor_addr];

    // Stage 1: 16 cycles of (per-warp distinct line, anchor
    // revisit). Each revisit produces a sector-rd of 7 (3 other
    // sectors from the previous anchor touch + 4 sectors from
    // the distinct line) at the first lane of each sector. The
    // bin admission requires >=50 pooled samples; 64 warps * 16
    // cycles * 4 sectors = 4096 sector-layer samples land in
    // bin [4,7] with 80x margin over the floor.
    for (int cycle = 0; cycle < 16; ++cycle) {
        int32_t distinct_line = PHASE1_BASE_LINE
            + warp_id * 16 + cycle;
        int32_t distinct_addr =
            thread_addr_from_line(distinct_line);
        acc += in[distinct_addr];
        acc += in[anchor_addr];
    }

    // Stage 2: 1024 distinct lines (shared across warps), then
    // one anchor revisit. The revisit's sector-rd ~ 4099 (3
    // residual + 1024 * 4 = 4096 + 3); 64 warps * 4 sectors =
    // 256 samples in sector bin [4096, 8191] (5x margin over
    // the MINIMUM_BIN_SAMPLE_COUNT=50 floor).
    for (int sweep = 0; sweep < 1024; ++sweep) {
        int32_t distinct_line = PHASE2_BASE_LINE + sweep;
        int32_t distinct_addr =
            thread_addr_from_line(distinct_line);
        acc += in[distinct_addr];
    }
    acc += in[anchor_addr];

    // Stage 3: 256 distinct lines (shared across warps), then
    // one anchor revisit. The revisit's sector-rd ~ 1027
    // (256 distinct lines * 4 sectors + 3 residual);
    // 64 warps * 4 sectors = 256 samples in sector bin
    // [1024, 2047], well above the MINIMUM_BIN_SAMPLE_COUNT=50
    // floor. The R34 main-pattern resize shrinks Stage 3 from
    // 4096 to 256 distinct lines (kernel L2 footprint drops
    // from ~680 KiB to ~210 KiB, well within L2 capacity) to
    // test whether the smaller intra-launch footprint reduces
    // label-side stddev_l2_sector below the 0.03 admission
    // threshold (R31 baseline reproduced at 0.0560 with the
    // 4096-line Stage 3; R34 is the kernel-pattern-remediation
    // arm of BL-20260513-l2-wash-preamble-at-l2-capacity-
    // amplifies-label-variance Solution path (a) main-pattern
    // resize).
    for (int sweep = 0; sweep < 256; ++sweep) {
        int32_t distinct_line = PHASE3_BASE_LINE + sweep;
        int32_t distinct_addr =
            thread_addr_from_line(distinct_line);
        acc += in[distinct_addr];
    }
    acc += in[anchor_addr];

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
    // correctness shape's hash digest reads back warps * 32
    // floats; using a single warp-load (instead of a strided
    // identity loop over the entire `out` buffer) keeps the
    // identity-output access pattern from spreading samples
    // thinly across mid-range HRD bins on the
    // trace_l2_accuracy_xl shape (the bin-admission floor is
    // 50; the prior strided identity loop produced 2-48
    // samples in bins [2,2], [3,3], [4,7], [8,15], [16,31],
    // forcing the bundle to be underconfigured).
    out[warp_id * lanes + tid] = in[anchor_addr];
}

extern "C" void working_set_capacity_sweep_launch(
    const float *d_in, float *d_out, float *d_sink,
    int working_set_floats, int accesses, int warps,
    cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps > 0 ? warps : 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    working_set_capacity_sweep_kernel<<<grid, block, 0, stream>>>(
        d_in, d_out, d_sink, working_set_floats, accesses, warps);
}
