// slug: l2_persisting_window_sweep
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// L2 persisting-window sweep microbench: the launching CUDA
// stream has an L2 persistence access policy window set via
// `cudaStreamSetAttribute` /
// `cudaStreamAttributeAccessPolicyWindow` over the input
// allocation `d_in`. The persistence hint is hardware-advisory
// and the runtime may legally clip the requested window to the
// device's actual L2 persisting capacity (queryable via
// `cudaDeviceGetAttribute` with
// `cudaDevAttrMaxAccessPolicyWindowSize`). On architectures
// that do not advertise the attribute, the host driver emits a
// typed-unavailable sidecar rather than silently skipping.
//
// R35 rewrite: replaces the prior per-thread stride-1 streaming
// loop with the warp-coalesced layered-reuse pattern proven on
// `working_set_capacity_sweep` (R34 main-pattern resize landed
// stddev_l2_sector=0.0136 on that cell). Each warp owns a
// distinct anchor cache line inside the persistence window; the
// kernel runs two layered phases that end with anchor revisits,
// producing reuse-distance observations that concentrate
// densely in a small set of HRD bins so the producer-side HRD
// profile clears MINIMUM_BIN_SAMPLE_COUNT=50 with margin and
// admits a scalar bundle.
//
//   Stage 1 (sector-rd target ~ 7): 4 cycles of
//       [per-warp distinct line, anchor revisit].
//   Stage 2 (sector-rd target ~ 1027): 1 cycle of
//       [256 shared distinct lines, anchor revisit].
//
// The launch geometry is unchanged (threads=32, blocks=warps).
// All 32 threads of each warp coalesce on each line touch.
//
// The host-side persistence-window setup in `main.cu` is
// preserved unchanged: the cell's identity as a
// persistence-window exerciser is retained because all reads
// land in the persistence window (the warp anchor + Stage 1 +
// Stage 2 lines occupy the first 1536 cache lines = 49152
// floats = 192 KiB of the allocation, well inside the 8 MiB
// trace_l2_accuracy_xl persistence-window setting). The R35
// rewrite applies the Solution path (a) main-pattern resize
// arm from BL-20260513-l2-wash-preamble-at-l2-capacity-
// amplifies-label-variance, which working_set R34 validated.
//
// R35 intra-launch L2 footprint: 256 anchor lines + Stage 1
// distinct lines (warps * 4 = 1024 lines) + Stage 2 distinct
// lines (256 lines) = 1536 lines = 192 KiB ≈ 3.2% of sm_86 L2
// capacity. Comparable to working_set R34's ~210 KiB
// (3.5% of L2).
//
// R35 label-stability outcome: documented-debug-finding. Three
// kernel-pattern iterations were measured against the 0.03 L2
// stddev admission threshold (and 0.045 partial-success
// threshold):
//   (i)  Stage 1 (4 cycles) + Stage 2 (256 shared distinct
//        lines): best stddev_l2_sector = 0.0585 at
//        warmup=5/cache=all (the iteration committed here).
//   (ii) Stage 1 only (16 cycles, no shared phase): best
//        stddev_l2_sector = 0.0602 at warmup=10/cache=none.
//   (iii) Stage 1 (16 cycles) with active-warp gate
//        (warp_id < 64): bundle_underconfigured = TRUE
//        (the R35 contract's blocking side issue 1: the
//        kernel rewrite MUST preserve
//        bundle_underconfigured = false; this iteration
//        violates that and is reverted).
// All three iterations stayed above the 0.045 partial-success
// threshold. The variance source is the hardware-advisory
// persistence-window hint:
// cudaAccessPropertyPersisting biases L2 replacement against
// eviction inside the window, but the actual eviction order
// is hardware-determined and varies launch-to-launch. The
// kernel-pattern change cannot fully eliminate this variance
// because the persistence window is set host-side and
// preserved as part of the cell's identity. Next-round
// recovery paths are documented in
// `.humanize/rlcr/2026-05-12_13-03-00/round-35-summary.md`.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void l2_persisting_window_sweep_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    float *__restrict__ sink,
    int32_t allocation_floats, int32_t accesses, int32_t warps)
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
    // band; each warp owns a contiguous 4-line slot so its
    // Stage-1 distinct sweep does not collide with neighbours.
    const int32_t PHASE1_BASE_LINE = (int32_t)warps;
    // Stage 2's shared distinct-line band starts after Stage 1's
    // per-warp slots: warps * 4 lines + warps for anchors.
    const int32_t PHASE2_BASE_LINE = PHASE1_BASE_LINE + 4 * warps;

    // Compute a thread's address from a target line index.
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

    // Initial anchor touch (cold at every layer for warp 0; for
    // warps 1..N-1 the allocation/region trackers see the
    // allocation as already finite at this point in the per-TB
    // walk).
    acc += in[anchor_addr];

    // Stage 1: 4 cycles of (per-warp distinct line, anchor
    // revisit). Each revisit produces a sector-rd of 7 (3 other
    // sectors from the previous anchor touch + 4 sectors from
    // the distinct line) at the first lane of each sector. The
    // bin admission requires >=50 pooled samples; 256 warps * 4
    // cycles * 4 sectors = 4096 sector-layer samples land in
    // bin [4,7] with 81x margin over the floor.
    for (int cycle = 0; cycle < 4; ++cycle) {
        int32_t distinct_line = PHASE1_BASE_LINE
            + warp_id * 4 + cycle;
        int32_t distinct_addr =
            thread_addr_from_line(distinct_line);
        acc += in[distinct_addr];
        acc += in[anchor_addr];
    }

    // Stage 2: 256 distinct lines (shared across warps), then
    // one anchor revisit. The revisit's sector-rd ~ 1027
    // (256 distinct lines * 4 sectors + 3 residual);
    // 256 warps * 4 sectors = 1024 samples in sector bin
    // [1024, 2047], 20x margin over the
    // MINIMUM_BIN_SAMPLE_COUNT=50 floor.
    for (int sweep = 0; sweep < 256; ++sweep) {
        int32_t distinct_line = PHASE2_BASE_LINE + sweep;
        int32_t distinct_addr =
            thread_addr_from_line(distinct_line);
        acc += in[distinct_addr];
    }
    acc += in[anchor_addr];

    // Sink write restricted to block 0's tid 0 so multi-block
    // launches (warps > 1, which makes blocks = warps in the
    // launch wrapper below; trace_l2_accuracy_xl uses warps=256)
    // do not race on `sink[0]`. The kernel uses block-local
    // `tid = threadIdx.x` for its per-CTA observation pattern;
    // gating the scalar sink store on `blockIdx.x == 0` keeps a
    // single deterministic store without disturbing the per-CTA
    // work distribution. Codex review-phase P2 fix (multi-block
    // sink race).
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
    // trace_l2_accuracy_xl shape (per
    // BL-20260513-coalesced-warpload-identity-output-floors-
    // hrd-bins).
    out[warp_id * lanes + tid] = in[anchor_addr];
    (void)accesses;
}

extern "C" void l2_persisting_window_sweep_launch(
    const float *d_in, float *d_out, float *d_sink,
    int allocation_floats, int accesses, int warps,
    cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps > 0 ? warps : 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    l2_persisting_window_sweep_kernel<<<grid, block, 0, stream>>>(
        d_in, d_out, d_sink, allocation_floats, accesses, warps);
}
