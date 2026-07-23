// nvbit_tracer/metric_extractor/sector_counting.h
//
// Pure host + device helper: count the distinct 32-byte memory sectors
// touched by a set of per-lane global memory accesses. Used by the
// `count_global_mem_access` injected device callback and by the host-only
// regression test that locks in the algorithm against the Round-5 single-
// sector-only bug Codex flagged (an access at `addr=24, bytes=16` touches
// BOTH sector 0 and sector 1; the pre-Round-6 algorithm counted only
// sector 0, undercounting transferred bytes for any access that crosses a
// 32-byte boundary).
//
// Algorithm:
//   For each lane access (addr, bytes):
//     first_sector = addr >> 5
//     last_sector  = (addr + bytes - 1) >> 5
//     Insert every sector in [first_sector, last_sector] into a per-warp
//     set of unique sector ids.
//   Returns the size of the set, capped at a documented cap.
//
// Capacity rationale:
//   Worst-case per-warp distinct sectors = 32 lanes * ceil(bytes/32) + 1
//   for boundary-crossing misalignment. For typical access sizes
//   (1, 2, 4, 8, 16, 32 bytes per lane), the worst case is 32 lanes *
//   2 sectors-per-lane = 64. For LDG.E.128 (16 B per lane) misaligned by
//   24 bytes, every lane spans 2 sectors, and adjacent lanes share
//   boundaries so distinct = 17 across the warp. For 128 B per lane
//   (uncommon, hypothetical 4x uint4 super-vector), distinct could be
//   32 * 5 = 160. The cap is set to 128 (covering 2x safety margin
//   beyond realistic worst-case access sizes) — if a future ISA / access
//   pattern exceeds 128 distinct sectors per warp instruction, the
//   helper returns the cap value and a future bounded-memory saturation
//   flag (T38 follow-up) will surface the saturation.

#ifndef NVBIT_TRACER_METRIC_EXTRACTOR_SECTOR_COUNTING_H_
#define NVBIT_TRACER_METRIC_EXTRACTOR_SECTOR_COUNTING_H_

#include <stdint.h>

#define METRIC_EXTRACTOR_DISTINCT_SECTOR_CAP 128

// __host__ __device__ so the device callback in inject_funcs.cu calls the
// same function the host-only regression in test_classifier.cc exercises.
// Marked inline so multiple translation units that include the header do
// not generate duplicate symbols at link time.
//
// `addresses`: array of `n` lane base addresses.
// `n`: number of lanes whose access should be counted. Callers filter by
//      predicate / active mask before passing the array.
// `bytes_per_access`: per-lane access size in bytes (from `Instr::getSize()`).
//
// Returns the count of distinct 32-byte sectors touched by any lane.
// When n <= 0 or bytes_per_access <= 0, returns 0 (defensive — neither
// the device callback nor the host test should reach this path with
// pathological inputs, but the early-return prevents a spurious
// out-of-range write to the local sector array).
#ifdef __CUDACC__
__host__ __device__
#endif
inline int count_distinct_sectors(const uint64_t* addresses,
                                  int n,
                                  int bytes_per_access) {
    if (n <= 0 || bytes_per_access <= 0) {
        return 0;
    }
    uint64_t sectors[METRIC_EXTRACTOR_DISTINCT_SECTOR_CAP];
    int distinct = 0;
    for (int i = 0; i < n; ++i) {
        const uint64_t addr = addresses[i];
        const uint64_t first_sec = addr >> 5;
        const uint64_t last_sec  =
            (addr + (uint64_t)bytes_per_access - 1ULL) >> 5;
        for (uint64_t s = first_sec; s <= last_sec; ++s) {
            bool seen = false;
            for (int j = 0; j < distinct; ++j) {
                if (sectors[j] == s) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                if (distinct >= METRIC_EXTRACTOR_DISTINCT_SECTOR_CAP) {
                    // Cap reached. Return the cap so the caller's
                    // numerator/denominator math stays consistent; a
                    // future bounded-memory saturation flag (T38
                    // follow-up) will surface the cap-hit signal.
                    return distinct;
                }
                sectors[distinct++] = s;
            }
        }
    }
    return distinct;
}

#endif  // NVBIT_TRACER_METRIC_EXTRACTOR_SECTOR_COUNTING_H_
