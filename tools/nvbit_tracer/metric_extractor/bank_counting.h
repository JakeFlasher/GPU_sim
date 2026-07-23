// nvbit_tracer/metric_extractor/bank_counting.h
//
// Pure host + device helper: count the number of 32-bank shared-memory
// bank conflicts produced by a set of per-lane shared loads/stores.
// Used by the `count_shared_bank_conflicts_cb` injected device callback
// and by the host-only regression in test_classifier.cc.
//
// Canonical 32-bank shared-memory model (Volta+):
//   - Bank id = (addr / 4) & 31 — i.e. each 4-byte word lives in a bank
//     numbered by `word_idx & 31`, with adjacent words in adjacent banks.
//   - Within a bank, multiple lanes hitting the SAME word is a
//     broadcast (no conflict).
//   - Within a bank, N lanes hitting DIFFERENT words causes serialization
//     and adds N-1 conflicts to the warp's tally.
//   - Conflicts are symmetric between loads and stores; the metric does
//     not distinguish LDS from STS in the per-launch counter (the
//     JSON-L schema has a single `bank_conflicts` field).
//
// Algorithm:
//   For each lane address (predicate-true only; callers pre-filter):
//     word_idx = addr >> 2
//     bank     = word_idx & 31
//     Dedup the (bank, word_idx) tuple; track distinct_words_per_bank[b].
//   Total conflicts = sum over b of max(0, distinct_words_per_bank[b] - 1).
//
// Distributed-shared (DSMEM) loads/stores have different banking semantics
// and MUST NOT be processed by this model — the instrument-pass filter
// `getMemorySpace() == InstrType::MemorySpace::SHARED` excludes DSMEM by
// positive selection (DSMEM has `DISTRIBUTED_SHARED`), so this helper
// never sees DSMEM addresses. The Round-4 `is_distributed_shared(Instr*)`
// bypass marker is the documented supporting contract for any future
// callback installer that wants to layer additional shared-memory
// analysis without re-deriving the filter.

#ifndef NVBIT_TRACER_METRIC_EXTRACTOR_BANK_COUNTING_H_
#define NVBIT_TRACER_METRIC_EXTRACTOR_BANK_COUNTING_H_

#include <stdint.h>

// __host__ __device__ so the device callback in inject_funcs.cu and the
// host-only test in test_classifier.cc call the same function. Marked
// inline so multiple translation units that include the header do not
// generate duplicate symbols at link time.
//
// `addresses`: array of `n` lane addresses (callers pre-filter to
//              predicate-true lanes).
// `n`: count of addresses (0..32 for typical warps).
//
// Returns the total number of warp-level bank conflicts. Bounded by
// `n - distinct_banks_used` plus broadcast accounting, but always
// non-negative.
#ifdef __CUDACC__
__host__ __device__
#endif
inline int count_shared_bank_conflicts(const uint64_t* addresses, int n) {
    if (n <= 0) {
        return 0;
    }
    // Per-warp dedup of (bank, word_idx) tuples. Worst case is 32 unique
    // tuples for a 32-lane warp where every lane targets a different
    // (bank, word) — i.e. perfect distribution with no conflicts. The
    // arrays are sized to 32 because we only deduplicate within a single
    // warp instance and `n <= 32` for any realistic warp.
    uint64_t seen_words[32];
    uint8_t  seen_banks[32];
    int      n_seen = 0;
    int      distinct_per_bank[32];
    for (int b = 0; b < 32; ++b) distinct_per_bank[b] = 0;
    for (int i = 0; i < n; ++i) {
        const uint64_t word_idx = addresses[i] >> 2;
        const uint8_t  bank     = (uint8_t)(word_idx & 31u);
        bool already = false;
        for (int j = 0; j < n_seen; ++j) {
            if (seen_banks[j] == bank && seen_words[j] == word_idx) {
                already = true;
                break;
            }
        }
        if (!already) {
            if (n_seen >= 32) {
                // Defensive: should never reach here for n<=32, but the
                // bound prevents out-of-bounds write on pathological
                // input.
                continue;
            }
            seen_banks[n_seen] = bank;
            seen_words[n_seen] = word_idx;
            n_seen++;
            distinct_per_bank[bank]++;
        }
    }
    int conflicts = 0;
    for (int b = 0; b < 32; ++b) {
        if (distinct_per_bank[b] > 0) {
            conflicts += distinct_per_bank[b] - 1;
        }
    }
    return conflicts;
}

#endif  // NVBIT_TRACER_METRIC_EXTRACTOR_BANK_COUNTING_H_
