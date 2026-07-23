// Host-side regression test for ``saturating_add_u64_compute``, the
// pure compute kernel that ``saturating_atomic_add_u64`` (in
// ``saturating_counter.h``) wraps with an atomicCAS retry loop on the
// device. The compute kernel is the part the saturation contract
// actually hinges on — the CAS wrapper is straightforward retry
// scaffolding around it. Locking down the compute behaviour here gives
// the AC-15 saturation contract a real test surface without needing
// CUDA / NVBit at test time.
//
// Build via the metric_extractor Makefile:
//   make -C tools/nvbit_tracer/metric_extractor test_saturating_counter
//   make -C tools/nvbit_tracer/metric_extractor run-test-saturating-counter
//
// Exits 0 on success, non-zero on any assertion failure.

#include <cstdio>
#include <cstdint>

#include "saturating_counter.h"

namespace {

int failures = 0;

void report_fail(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "FAIL  %s:%d  %s\n", file, line, expr);
    ++failures;
}

#define ME_EXPECT(cond)                                          \
    do {                                                         \
        if (!(cond)) report_fail(#cond, __FILE__, __LINE__);     \
    } while (0)

void test_zero_delta_does_not_saturate() {
    uint64_t out = 0;
    bool overflowed = true;
    saturating_add_u64_compute(/*old=*/0, /*delta=*/0, &out, &overflowed);
    ME_EXPECT(out == 0);
    ME_EXPECT(!overflowed);

    saturating_add_u64_compute(/*old=*/UINT64_MAX, /*delta=*/0,
                               &out, &overflowed);
    ME_EXPECT(out == UINT64_MAX);
    ME_EXPECT(!overflowed);
}

void test_normal_add_does_not_saturate() {
    uint64_t out = 0;
    bool overflowed = true;
    saturating_add_u64_compute(/*old=*/100, /*delta=*/1, &out, &overflowed);
    ME_EXPECT(out == 101);
    ME_EXPECT(!overflowed);

    saturating_add_u64_compute(/*old=*/(UINT64_MAX / 2),
                               /*delta=*/1ULL, &out, &overflowed);
    ME_EXPECT(out == (UINT64_MAX / 2) + 1ULL);
    ME_EXPECT(!overflowed);
}

void test_exact_max_does_not_saturate() {
    // Hitting UINT64_MAX exactly is NOT saturation; the helper only
    // counts an event when the addition WOULD have overflowed past
    // UINT64_MAX. This matches the spec: saturated_counters is the
    // count of LOST increments, not the count of times the counter
    // touched its ceiling.
    uint64_t out = 0;
    bool overflowed = true;
    saturating_add_u64_compute(/*old=*/(UINT64_MAX - 1),
                               /*delta=*/1, &out, &overflowed);
    ME_EXPECT(out == UINT64_MAX);
    ME_EXPECT(!overflowed);
}

void test_one_past_max_saturates() {
    uint64_t out = 0;
    bool overflowed = false;
    saturating_add_u64_compute(/*old=*/UINT64_MAX, /*delta=*/1,
                               &out, &overflowed);
    ME_EXPECT(out == UINT64_MAX);
    ME_EXPECT(overflowed);
}

void test_partial_overflow_clamps_at_max() {
    // The canonical overflow case: the addition would land somewhere
    // past UINT64_MAX. The helper clamps at UINT64_MAX and reports
    // overflow, NOT some intermediate "wrap mod 2^64" value the way a
    // raw atomicAdd would.
    uint64_t out = 0;
    bool overflowed = false;
    saturating_add_u64_compute(/*old=*/(UINT64_MAX - 5),
                               /*delta=*/100ULL,
                               &out, &overflowed);
    ME_EXPECT(out == UINT64_MAX);
    ME_EXPECT(overflowed);

    // Stress: the largest possible delta against a non-zero old value.
    saturating_add_u64_compute(/*old=*/1, /*delta=*/UINT64_MAX,
                               &out, &overflowed);
    ME_EXPECT(out == UINT64_MAX);
    ME_EXPECT(overflowed);
}

void test_zero_plus_max_does_not_saturate() {
    // Old = 0 with delta = UINT64_MAX lands EXACTLY at UINT64_MAX
    // without overflowing, mirroring test_exact_max_does_not_saturate
    // above. This case matters because a Tier-2 per-PC table that
    // resets each launch needs the very first increment to land
    // cleanly, not get spuriously flagged as a saturation event.
    uint64_t out = 0;
    bool overflowed = true;
    saturating_add_u64_compute(/*old=*/0, /*delta=*/UINT64_MAX,
                               &out, &overflowed);
    ME_EXPECT(out == UINT64_MAX);
    ME_EXPECT(!overflowed);
}

}  // namespace

int main() {
    test_zero_delta_does_not_saturate();
    test_normal_add_does_not_saturate();
    test_exact_max_does_not_saturate();
    test_one_past_max_saturates();
    test_partial_overflow_clamps_at_max();
    test_zero_plus_max_does_not_saturate();
    if (failures == 0) {
        std::printf("PASS test_saturating_counter\n");
        return 0;
    }
    std::fprintf(stderr, "FAIL %d assertion(s) in test_saturating_counter\n",
                 failures);
    return 1;
}
