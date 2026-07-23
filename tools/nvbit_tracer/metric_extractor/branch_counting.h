// nvbit_tracer/metric_extractor/branch_counting.h
//
// Pure host + device helper: map a 32-bit warp-collective predicate mask
// to the histogram bucket index in `branch_active_lanes_hist[0..32]`.
// Used by the `count_branch_active_mask_cb` injected device callback and
// by the host-only regression in test_classifier.cc.
//
// Semantics:
//   - Each warp-instance of a branch-class instruction (every opcode the
//     classifier routes to `INSTR_CLASS_BRA` — BRA / JMP / EXIT / BRX /
//     CALL / RET / etc.) increments exactly one bucket in
//     `branch_active_lanes_hist`.
//   - The bucket is indexed by the popcount of the warp-collective
//     predicate mask, which by construction is in `[0, 32]` (the mask is
//     32 bits wide, one bit per lane).
//   - Bucket 0 = predicate-false everywhere (no active lane took the
//     branch this warp-instance); a true observation, NOT a placeholder.
//   - Bucket 32 = fully-active branch (unconditional or always-true
//     predicate); also a true observation.
//   - Bucket 16 = AC-4 canonical "predicated branch with 16 active
//     lanes" — the deterministic regression test in test_classifier.cc
//     drives three different 16-active lane patterns through this
//     helper and asserts `hist[16] == 3`.
//
// The histogram is divergence-shape blind: two warp-instances with
// different lane patterns (e.g. `0x0000FFFF` vs `0xFFFF0000`) but the
// same popcount land in the same bucket. This is intentional — the
// metric describes branch divergence intensity, not which specific
// lanes diverged.

#ifndef NVBIT_TRACER_METRIC_EXTRACTOR_BRANCH_COUNTING_H_
#define NVBIT_TRACER_METRIC_EXTRACTOR_BRANCH_COUNTING_H_

#include <stdint.h>

// __host__ __device__ so the device callback in inject_funcs.cu and the
// host-only test in test_classifier.cc call the same function. Marked
// inline so multiple translation units that include the header do not
// generate duplicate symbols at link time.
//
// On the device this resolves to `__popc(mask)`; on the host it
// resolves to `__builtin_popcount(mask)`. Both return an integer in
// `[0, 32]`.
//
// `mask`: warp-collective predicate mask, typically built via
//         `__ballot_sync(__activemask(), predicate)`. The mask bit
//         layout is bit `i` = lane `i`'s predicate-true status (with
//         inactive lanes contributing 0 by construction of
//         `__ballot_sync`).
//
// Returns: the bucket index in `branch_active_lanes_hist`, i.e. the
//          count of predicate-true active lanes.
#ifdef __CUDACC__
__host__ __device__
#endif
inline int count_active_lanes(uint32_t mask) {
#ifdef __CUDA_ARCH__
    return __popc(mask);
#else
    return __builtin_popcount(mask);
#endif
}

#endif  // NVBIT_TRACER_METRIC_EXTRACTOR_BRANCH_COUNTING_H_
