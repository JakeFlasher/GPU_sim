// nvbit_tracer/metric_extractor/saturating_counter.h
//
// Saturating-add primitive for per-launch ``LaunchMetrics`` counters.
//
// Counters in the per-launch slot are 64-bit unsigned (``uint64_t``). A
// raw ``atomicAdd`` on these counters wraps silently at ``UINT64_MAX``,
// which the bounded-memory contract rejects: under a long-running
// fanout (or a future Tier-2 per-PC table with denser increments), the
// JSON-L record must show ``saturated_counters > 0`` instead of letting
// the histogram value wrap to a small number and look like a normal
// launch.
//
// The helper here splits cleanly into two parts:
//
// * ``saturating_add_u64_compute`` — pure ``__host__ __device__``
//   function that takes a ``(old_value, delta)`` pair and returns the
//   clamped ``new_value`` plus an ``overflowed`` flag. No atomics, no
//   memory access. The host-side tests exercise this directly via a
//   host-only translation unit (no CUDA dependency) so the saturation
//   logic is verifiable without needing a CUDA runtime.
// * ``saturating_atomic_add_u64`` — ``__device__``-only wrapper that
//   composes the compute helper with an ``atomicCAS``-based retry loop.
//   On the iteration that wins the CAS, it bumps a per-launch
//   ``saturated_counters`` field iff that iteration clamped at
//   ``UINT64_MAX``. The 8-byte aligned ``uint64_t`` load that seeds the
//   loop is atomic on every SM ≥ 6.0 device the tool targets (sm_86
//   local + sm_100/sm_100a B200), so the seed read does not need its
//   own atomic primitive.
//
// Every per-launch counter atomicAdd in ``inject_funcs.cu`` routes
// through ``saturating_atomic_add_u64``: histogram buckets, the TMA
// counter, the four sector-byte counters, the branch-active-lanes
// histogram, and the bank-conflict total. The ``count_distinct_sectors``
// and ``count_shared_bank_conflicts`` helpers themselves stay pure
// because their per-warp results feed a single saturating add at the
// call site, not a ladder of per-bank atomics.

#ifndef NVBIT_TRACER_METRIC_EXTRACTOR_SATURATING_COUNTER_H_
#define NVBIT_TRACER_METRIC_EXTRACTOR_SATURATING_COUNTER_H_

#include <stdint.h>

// ``__CUDACC__`` is defined by nvcc on every CUDA compilation pass
// (device and host paths). Plain ``g++`` / ``clang++`` host
// compilation does not define it, which is the signal we use to
// stub out the CUDA attribute macros so the compute helper still
// compiles cleanly in a host-only translation unit (e.g. the
// ``test_saturating_counter.cc`` regression test).
#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__ inline
#define ME_SAT_HOST_ONLY_BUILD 1
#endif

// Pure compute helper. Returns the clamped new value and whether the
// addition would have overflowed ``UINT64_MAX``. Both outputs are
// returned via pointers so the host-side test can call the same
// function the device wrapper does — no need to maintain two
// definitions in lockstep.
//
// Contract: if ``delta == 0`` the new value equals the old value and
// ``overflowed`` is false (a zero-add never saturates). Otherwise
// ``overflowed`` is true iff ``old_value + delta`` would exceed
// ``UINT64_MAX``; in that case the new value clamps to ``UINT64_MAX``.
__host__ __device__ __forceinline__ void saturating_add_u64_compute(
    uint64_t old_value,
    uint64_t delta,
    uint64_t* new_value_out,
    bool* overflowed_out) {
    if (delta == 0ULL) {
        *new_value_out = old_value;
        *overflowed_out = false;
        return;
    }
    const uint64_t headroom = UINT64_MAX - old_value;
    if (delta > headroom) {
        *new_value_out = UINT64_MAX;
        *overflowed_out = true;
    } else {
        *new_value_out = old_value + delta;
        *overflowed_out = false;
    }
}

#ifndef ME_SAT_HOST_ONLY_BUILD

// Device-only wrapper. Atomically clamps ``*addr`` at ``UINT64_MAX``
// on overflow and increments ``*saturated_counter`` once per iteration
// of the CAS loop that wins the CAS AND clamped. ``saturated_counter``
// may be ``nullptr`` for callers that do not need overflow accounting
// (e.g. dedicated test paths). Concurrent threads each retry the CAS;
// the per-thread retry that succeeds is the one that decides whether
// to bump the saturation counter, which means simultaneous overflows
// from N threads bump ``saturated_counter`` by exactly N (one per
// successful saturating add).
//
// Why CAS instead of a vanilla ``atomicAdd`` + post-check: a vanilla
// ``atomicAdd`` would wrap silently, and the post-hoc detection
// (``returned_old + delta < returned_old``) would race with concurrent
// adds and double-count saturation events. The CAS loop is the only
// correct pattern that emits a saturation event for exactly the right
// iteration.
__device__ __forceinline__ void saturating_atomic_add_u64(
    uint64_t* addr,
    uint64_t delta,
    uint64_t* saturated_counter) {
    if (delta == 0ULL) return;
    unsigned long long* ull_addr =
        reinterpret_cast<unsigned long long*>(addr);
    // The 8-byte aligned load below is atomic on every SM ≥ 6.0 device
    // we target. It seeds the CAS loop; concurrent updates land via
    // the CAS retry path.
    unsigned long long old_val = *ull_addr;
    while (true) {
        uint64_t new_val_u64;
        bool overflowed = false;
        saturating_add_u64_compute(
            static_cast<uint64_t>(old_val), delta,
            &new_val_u64, &overflowed);
        const unsigned long long new_val =
            static_cast<unsigned long long>(new_val_u64);
        const unsigned long long observed =
            atomicCAS(ull_addr, old_val, new_val);
        if (observed == old_val) {
            if (overflowed && saturated_counter != nullptr) {
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(saturated_counter),
                    1ULL);
            }
            return;
        }
        old_val = observed;
    }
}

#endif  // !ME_SAT_HOST_ONLY_BUILD

#ifdef ME_SAT_HOST_ONLY_BUILD
#undef __host__
#undef __device__
#undef __forceinline__
#undef ME_SAT_HOST_ONLY_BUILD
#endif

#endif  // NVBIT_TRACER_METRIC_EXTRACTOR_SATURATING_COUNTER_H_
