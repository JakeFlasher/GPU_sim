"""Warp-pool walker fixtures: exact designed hit rates for the
interleaving, latency-proportional, MSHR-merge, MSHR-exhaustion, and
two-layer sector/line families, plus the negative arms (the
concatenated walk fails the interleaving fixture; the unlimited-MSHR
walk fails the exhaustion fixture; the flat single-granularity walk
fails the sectored fixtures) and the card-sourced configuration
resolvers (carveout-aware capacity; replacement/set-mapping as
documented approximations ONLY where the card types unresolved).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from l1_warp_pool_walker import (  # noqa: E402
    DOCUMENTED_DEFAULT_MSHR_COUNT,
    LINE_BYTES,
    WalkerConfigError,
    co_resident_warp_limit,
    resolve_latencies,
    resolve_line_capacity,
    resolve_mshr_count,
    resolve_replacement_policy,
    resolve_set_mapping,
    resolve_walk_geometry,
    walk_warp_pool,
)

HIT = 33.0
MISS = 250.0


def _stream(line_ids):
    """Explicit 3-tuples (line_id, sector_id, l1_eligible): one access
    to each line's first sector — the unambiguous sector form (the
    legacy 2-tuple form means (line_id, l1_eligible) and is covered by
    its own contract arms below)."""
    return [(line, line * 4, True) for line in line_ids]


def _interleaving_fixture():
    """Two warps, disjoint 6-line working sets, two passes each;
    capacity 8 lines. Designed expectations (exact):
    - concatenated: each warp's second pass fully hits -> 12 hits,
      12 misses (the block-by-block overestimate);
    - interleaved (warp_pool or round_robin with symmetric latencies):
      the combined 12-line set thrashes the 8-line LRU -> 0 hits,
      24 misses."""
    w0 = _stream(list(range(0, 6)) * 2)
    w1 = _stream(list(range(100, 106)) * 2)
    return [("w0", w0), ("w1", w1)]


class LegacyAccessContractTest(unittest.TestCase):
    """The adapter still emits legacy 2-tuples (line_id, l1_eligible)
    when data_width is unavailable; the warp-pool walker must honor
    the SAME contract as the existing walker (the review's blocking
    reproducer: bypass records silently became modeled misses)."""

    def test_legacy_two_tuple_miss_then_hit(self) -> None:
        result = walk_warp_pool(
            [("w0", [(10, True), (10, True)])], line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            mode="warp_pool")
        self.assertEqual(result.miss_count, 1)
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.bypass_count, 0)

    def test_legacy_bypass_leaves_state_untouched(self) -> None:
        """The reviewer's exact reproducer: one eligible miss + two
        bypasses; bypasses change neither hit/miss nor MSHR state."""
        result = walk_warp_pool(
            [("w0", [(10, True), (10, False), (20, False)])],
            line_capacity=8, hit_latency=HIT, miss_latency=MISS,
            mshr_count=2, mode="warp_pool")
        self.assertEqual(result.miss_count, 1)
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.bypass_count, 2)
        self.assertEqual(result.line_fill_count, 1)
        self.assertEqual(result.mshr_stall_events, 0)

    def test_adapter_legacy_shape_regression(self) -> None:
        """Regression pinned to the trace adapter's documented legacy
        emission: when data_width is None the adapter emits ONE
        2-tuple (line_id, l1_eligible) per active lane address and the
        walk degrades to line-level LRU. Two warps re-reading the same
        two lines must produce line-level hits on the second pass."""
        legacy_stream = [(100, True), (101, True),
                         (100, True), (101, True)]
        result = walk_warp_pool(
            [("w0", legacy_stream)], line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            mode="warp_pool")
        self.assertEqual(result.miss_count, 2)
        self.assertEqual(result.hit_count, 2)

    def test_invalid_arity_raises(self) -> None:
        from l1_warp_pool_walker import WarpPoolAccessInvalid

        with self.assertRaises(WarpPoolAccessInvalid):
            walk_warp_pool(
                [("w0", [(1, 2, 3, 4)])], line_capacity=8,
                hit_latency=HIT, miss_latency=MISS, mshr_count=None,
                mode="warp_pool")

    def test_contract_matches_existing_walker(self) -> None:
        """The two normalizers must agree on both tuple forms."""
        import l1_walker
        from l1_warp_pool_walker import _normalize_access as new_norm

        for access in ((7, True), (7, False), (7, 30, True),
                       (7, 30, False)):
            self.assertEqual(
                new_norm(access),
                l1_walker._normalize_access(access),
                access)


class InterleavingFixtureTest(unittest.TestCase):
    def test_warp_pool_corrects_the_block_by_block_overestimate(self) -> None:
        result = walk_warp_pool(
            _interleaving_fixture(), line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            mode="warp_pool")
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.miss_count, 24)

    def test_concatenated_walk_fails_the_interleaving_fixture(self) -> None:
        """The negative arm: the old block-by-block behavior
        reproduces the documented overestimate (50% hits where the
        interleaved truth is 0%)."""
        result = walk_warp_pool(
            _interleaving_fixture(), line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            mode="concatenated")
        self.assertEqual(result.hit_count, 12)
        self.assertEqual(result.miss_count, 12)
        self.assertNotEqual(result.hit_count, 0,
                            "concatenation must NOT reproduce the "
                            "interleaved hit rate")


class LatencyProportionalTest(unittest.TestCase):
    """One warp cycles over 2 lines (hits once warm); another streams
    distinct lines (always misses). Latency-proportional scheduling
    lets the hitting warp re-reference its lines ~MISS/HIT times more
    often than the streamer evicts, so its locality SURVIVES a tight
    cache; the latency-blind round-robin destroys it."""

    def _fixture(self):
        hot = _stream([0, 1] * 60)
        streamer = _stream(range(1000, 1030))
        return [("hot", hot), ("streamer", streamer)]

    def test_warp_pool_preserves_hot_warp_locality(self) -> None:
        """EXACT designed arithmetic: the hot warp misses once cold,
        then once per streamer fill (each fill inserts a line at MRU
        of the capacity-3 LRU, evicting exactly one hot line) -> 31
        misses, 89 hits of 120."""
        result = walk_warp_pool(
            self._fixture(), line_capacity=3,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            mode="warp_pool")
        self.assertEqual(result.per_warp_hits.get("hot", 0), 89)
        self.assertEqual(result.per_warp_misses.get("hot", 0), 31)

    def test_round_robin_destroys_hot_warp_locality(self) -> None:
        result = walk_warp_pool(
            self._fixture(), line_capacity=3,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            mode="round_robin")
        hot_hits = result.per_warp_hits.get("hot", 0)
        self.assertLess(hot_hits / 120.0, 0.55)


class MshrMergeFixtureTest(unittest.TestCase):
    def test_same_line_in_flight_merges_one_fill(self) -> None:
        """Two warps miss the SAME line back-to-back: one MSHR entry,
        one fill, the second access merges (counts as a miss, becomes
        ready at the fill)."""
        fixture = [("w0", _stream([7])), ("w1", _stream([7]))]
        result = walk_warp_pool(
            fixture, line_capacity=4,
            hit_latency=HIT, miss_latency=MISS, mshr_count=4,
            mode="warp_pool")
        self.assertEqual(result.line_fill_count, 1)
        self.assertEqual(result.mshr_merge_count, 1)
        self.assertEqual(result.miss_count, 2)
        self.assertEqual(result.hit_count, 0)

    def test_post_fill_access_is_an_lru_hit_not_a_merge(self) -> None:
        """A third access to the line AFTER its fill time is a real
        hit (fills are applied lazily at issue time)."""
        # w0 misses line 7; w1's stream starts far in the future via
        # a long miss chain, then touches line 7 after the fill.
        fixture = [("w0", _stream([7])),
                   ("w1", _stream([100, 101, 7]))]
        result = walk_warp_pool(
            fixture, line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=8,
            mode="warp_pool")
        self.assertEqual(result.per_warp_hits.get("w1", 0), 1)
        self.assertEqual(result.line_fill_count, 3)


class MshrExhaustionFixtureTest(unittest.TestCase):
    def _fixture(self):
        return [("w0", _stream([10])), ("w1", _stream([20])),
                ("w2", _stream([30]))]

    def test_exhaustion_stalls_the_third_warp(self) -> None:
        """mshr_count=2 with three simultaneous distinct-line misses:
        the third warp stalls and retries when the earliest fill
        frees an entry."""
        result = walk_warp_pool(
            self._fixture(), line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=2,
            mode="warp_pool")
        self.assertEqual(result.mshr_stall_events, 1)
        self.assertEqual(result.miss_count, 3)
        self.assertEqual(result.line_fill_count, 3)
        # the stalled fill completes one full miss latency after the
        # earliest fill freed its entry
        self.assertGreaterEqual(result.final_time, 2 * MISS)

    def test_unlimited_mshr_walk_fails_the_exhaustion_fixture(self) -> None:
        """The negative arm: with unlimited MSHRs the fixture's stall
        expectation does NOT hold."""
        result = walk_warp_pool(
            self._fixture(), line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            mode="warp_pool")
        self.assertEqual(result.mshr_stall_events, 0)
        self.assertLess(result.final_time, 2 * MISS)
        self.assertNotEqual(
            result.mshr_stall_events, 1,
            "an unlimited-MSHR walk must fail the exhaustion fixture")


class CoResidencyTest(unittest.TestCase):
    def test_wave_admission_respects_residency(self) -> None:
        """Four warps, residency 2: the combined working set of waves
        never mixes — wave 2's warps only run after wave 1 retires, so
        wave 1's interleaving thrash does not touch wave 2."""
        w = {f"w{i}": _stream(list(range(i * 10, i * 10 + 6)) * 2)
             for i in range(4)}
        # residency 2, capacity 8: each WAVE of two warps thrashes
        # (12 lines > 8) but waves are independent.
        result = walk_warp_pool(
            list(w.items()), line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            co_resident_warps=2, mode="warp_pool")
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.miss_count, 48)
        # full residency: same outcome here, but the admission
        # machinery is what's pinned — with residency 1 each warp runs
        # alone and its second pass HITS (working set 6 < capacity 8).
        solo = walk_warp_pool(
            list(w.items()), line_capacity=8,
            hit_latency=HIT, miss_latency=MISS, mshr_count=None,
            co_resident_warps=1, mode="warp_pool")
        self.assertEqual(solo.hit_count, 24)

    def test_geometry_limit_helper(self) -> None:
        self.assertEqual(
            co_resident_warp_limit(warps_per_block=4,
                                   blocks_per_sm_limit=8), 32)
        self.assertEqual(
            co_resident_warp_limit(warps_per_block=32,
                                   blocks_per_sm_limit=4,
                                   occupancy_warp_limit=48), 48)
        with self.assertRaises(WalkerConfigError):
            co_resident_warp_limit(warps_per_block=0,
                                   blocks_per_sm_limit=1)


class SectorVsLineFixtureTest(unittest.TestCase):
    """Two-layer 32B-sector / 128B-line accounting. Sector residency
    dies with the evicted line (partial-line eviction), so the
    re-walk of a previously fully-fetched line must re-fetch its
    sectors; the flat single-granularity walk cannot see that."""

    # Capacity 2 lines. Line 10's sectors {40, 41} are fetched, the
    # 20/30 allocations evict line 10 (its sector bits die), and the
    # re-walk of line 10 re-allocates (sector 40) then takes a sector
    # miss on the resident line (sector 41).
    _PARTIAL_EVICTION_STREAM = [
        (10, 40, True), (10, 41, True), (20, 80, True),
        (30, 120, True), (10, 40, True), (10, 41, True)]

    def test_sector_miss_on_resident_line_classified(self) -> None:
        """A second sector of an already-resident line is a SECTOR
        miss (no new line fill); the sector then hits when
        re-touched."""
        result = walk_warp_pool(
            [("w0", [(10, 40, True), (10, 41, True), (10, 41, True)])],
            line_capacity=4, hit_latency=HIT, miss_latency=MISS,
            mshr_count=None, mode="warp_pool")
        self.assertEqual(result.line_alloc_miss_count, 1)
        self.assertEqual(result.sector_miss_resident_line_count, 1)
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.miss_count, 2)
        self.assertEqual(result.line_fill_count, 1)

    def test_partial_line_eviction_loses_sector_residency(self) -> None:
        """EXACT designed arithmetic: 0 hits, 4 line-allocation misses
        (10, 20, 30, then 10 again after its eviction) + 2 sector
        misses on the resident line 10 (sector 41 both times: once
        cold, once because the eviction dropped it)."""
        result = walk_warp_pool(
            [("w0", list(self._PARTIAL_EVICTION_STREAM))],
            line_capacity=2, hit_latency=HIT, miss_latency=MISS,
            mshr_count=None, mode="warp_pool")
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.miss_count, 6)
        self.assertEqual(result.line_alloc_miss_count, 4)
        self.assertEqual(result.sector_miss_resident_line_count, 2)
        self.assertEqual(result.line_fill_count, 4)
        self.assertEqual(result.mshr_merge_count, 0)

    def test_flat_single_granularity_walk_fails_the_sectored_fixture(
            self) -> None:
        """The negative arm: the line_flat ablation treats every
        access to a resident line as a hit, so it overpredicts the
        two partial-line sector misses as hits."""
        flat = walk_warp_pool(
            [("w0", list(self._PARTIAL_EVICTION_STREAM))],
            line_capacity=2, hit_latency=HIT, miss_latency=MISS,
            mshr_count=None, mode="warp_pool",
            sector_accounting="line_flat")
        self.assertEqual(flat.hit_count, 2)
        self.assertEqual(flat.miss_count, 4)
        self.assertEqual(flat.sector_miss_resident_line_count, 0)
        self.assertNotEqual(
            flat.hit_count, 0,
            "a flat single-granularity walk must FAIL the sectored "
            "fixture (the two-layer truth is 0 hits)")

    def test_miss_class_invariant_holds_with_merges(self) -> None:
        """miss_count == line_alloc + sector_resident + merge on the
        merge fixture (two warps miss the SAME line back-to-back)."""
        fixture = [("w0", _stream([7])), ("w1", _stream([7]))]
        result = walk_warp_pool(
            fixture, line_capacity=4,
            hit_latency=HIT, miss_latency=MISS, mshr_count=4,
            mode="warp_pool")
        self.assertEqual(
            result.miss_count,
            result.line_alloc_miss_count
            + result.sector_miss_resident_line_count
            + result.mshr_merge_count)
        self.assertEqual(result.line_alloc_miss_count, 1)
        self.assertEqual(result.mshr_merge_count, 1)

    def test_invalid_sector_accounting_rejected(self) -> None:
        with self.assertRaises(ValueError):
            walk_warp_pool(
                [("w0", _stream([1]))], line_capacity=2,
                hit_latency=HIT, miss_latency=MISS, mshr_count=None,
                mode="warp_pool", sector_accounting="three_layer")


class ConcurrentSectorFillTest(unittest.TestCase):
    """A resident-line sector fetch is keyed by (line, sector): it
    must not shadow hits to the line's other already-resident
    sectors, the SAME missing sector merges, and DISTINCT missing
    sectors are independent sector-resident misses — each consuming
    real MSHR capacity (the review's blocking reproducer family)."""

    def test_resident_sector_hits_while_other_sector_in_flight(
            self) -> None:
        """The reviewer's exact reproducer: w1's final (10, 40)
        access lands while sector 41 of line 10 is being fetched —
        line 10 is resident and sector 40 is resident, so it HITS
        (the whole-line in-flight shadow was the bug: it produced
        hit=0/miss=4/merge=1)."""
        result = walk_warp_pool(
            [
                ("w0", [(10, 40, True), (10, 41, True)]),
                ("w1", [(20, 80, True), (10, 40, True)]),
            ],
            line_capacity=4, hit_latency=HIT, miss_latency=MISS,
            mshr_count=8, mode="warp_pool")
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.miss_count, 3)
        self.assertEqual(result.mshr_merge_count, 0)
        self.assertEqual(result.line_alloc_miss_count, 2)
        self.assertEqual(result.sector_miss_resident_line_count, 1)
        self.assertEqual(result.line_fill_count, 2)

    def test_same_missing_resident_sector_merges_without_line_fill(
            self) -> None:
        """Two warps fetch the SAME missing sector of a resident
        line back-to-back: one sector entry, the second access
        merges, and NO new line fill appears."""
        result = walk_warp_pool(
            [
                ("w0", [(10, 40, True), (10, 41, True)]),
                ("w1", [(20, 80, True), (10, 41, True)]),
            ],
            line_capacity=4, hit_latency=HIT, miss_latency=MISS,
            mshr_count=8, mode="warp_pool")
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.miss_count, 4)
        self.assertEqual(result.mshr_merge_count, 1)
        self.assertEqual(result.line_alloc_miss_count, 2)
        self.assertEqual(result.sector_miss_resident_line_count, 1)
        self.assertEqual(result.line_fill_count, 2)
        self.assertEqual(
            result.miss_count,
            result.line_alloc_miss_count
            + result.sector_miss_resident_line_count
            + result.mshr_merge_count)

    def test_distinct_missing_resident_sectors_are_independent(
            self) -> None:
        """Two DIFFERENT missing sectors of the same resident line:
        two sector-resident misses, no merge, no false
        line-allocation fill; the miss-class invariant holds."""
        result = walk_warp_pool(
            [
                ("w0", [(10, 40, True), (10, 41, True)]),
                ("w1", [(20, 80, True), (10, 42, True)]),
            ],
            line_capacity=4, hit_latency=HIT, miss_latency=MISS,
            mshr_count=8, mode="warp_pool")
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.miss_count, 4)
        self.assertEqual(result.mshr_merge_count, 0)
        self.assertEqual(result.line_alloc_miss_count, 2)
        self.assertEqual(result.sector_miss_resident_line_count, 2)
        self.assertEqual(result.line_fill_count, 2)
        self.assertEqual(
            result.miss_count,
            result.line_alloc_miss_count
            + result.sector_miss_resident_line_count
            + result.mshr_merge_count)

    def test_sector_fills_consume_mshr_capacity(self) -> None:
        """mshr_count=1: w0's resident-line sector fetch holds the
        single entry, so w1's line-allocation miss stalls a second
        time against the SECTOR fill (first stall was against the
        line-allocation fill) — outstanding sector fetches consume
        real capacity."""
        result = walk_warp_pool(
            [
                ("w0", [(10, 40, True), (10, 41, True)]),
                ("w1", [(20, 80, True)]),
            ],
            line_capacity=4, hit_latency=HIT, miss_latency=MISS,
            mshr_count=1, mode="warp_pool")
        self.assertEqual(result.mshr_stall_events, 2)
        self.assertEqual(result.line_alloc_miss_count, 2)
        self.assertEqual(result.sector_miss_resident_line_count, 1)
        self.assertEqual(result.mshr_merge_count, 0)
        self.assertEqual(result.miss_count, 3)

    def test_lazy_sector_fill_then_hit(self) -> None:
        """After a resident-line sector fill completes, the sector
        is resident: a later access HITS (the lazy-fill mirror of
        the post-line-fill hit arm)."""
        result = walk_warp_pool(
            [("w0", [(10, 40, True), (10, 41, True), (10, 41, True),
                     (10, 40, True)])],
            line_capacity=4, hit_latency=HIT, miss_latency=MISS,
            mshr_count=8, mode="warp_pool")
        self.assertEqual(result.hit_count, 2)
        self.assertEqual(result.line_alloc_miss_count, 1)
        self.assertEqual(result.sector_miss_resident_line_count, 1)
        self.assertEqual(result.mshr_merge_count, 0)

    def test_stale_sector_fill_drops_at_its_own_time(self) -> None:
        """A sector fill whose line is absent AT ITS OWN FILL TIME
        drops THEN — it must never land into a LATER re-allocation
        of the same line. The sparse schedule here leaves no access
        between the stale fill's due time (333) and the
        re-allocation fill's due time (400): kind-batched
        application would land the re-allocation first and then
        resurrect the stale sector (counting w2's final access as a
        false hit); chronological replay drops it, so the final
        access is a sector-resident miss."""
        result = walk_warp_pool(
            [
                ("w0", [(3, 15, True)]),
                ("w1", [(2, 8, True), (3, 15, True), (3, 12, True),
                        (3, 14, True)]),
                ("w2", [(2, 8, True), (3, 12, True), (4, 18, True),
                        (3, 15, True), (3, 14, True)]),
            ],
            line_capacity=1, hit_latency=HIT, miss_latency=100.0,
            mshr_count=10, mode="warp_pool")
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.miss_count, 9)
        self.assertEqual(result.mshr_merge_count, 2)
        self.assertEqual(result.line_alloc_miss_count, 5)
        self.assertEqual(result.sector_miss_resident_line_count, 2)
        self.assertEqual(result.per_warp_hits, {"w1": 1})
        self.assertEqual(result.final_time, 500.0)
        self.assertEqual(
            result.miss_count,
            result.line_alloc_miss_count
            + result.sector_miss_resident_line_count
            + result.mshr_merge_count)


def _temp_card(td: str, properties: dict) -> Path:
    path = Path(td) / "card.json"
    path.write_text(json.dumps({"properties": properties}),
                    encoding="utf-8")
    return path


class CarveoutCapacityTest(unittest.TestCase):
    """Carveout-aware capacity sourced from the committed card's
    swept `l1_capacity_carveout_{pct}_pct_bytes` properties (the
    values pinned here are the committed sm_86 card's cross-run
    stable evidence: 96K default, 96/80/48/32/32K across the sweep —
    stable across driver 595 unlocked, and driver 610 under the
    fully-pinned v2.1 protocol with BOTH clocks locked; the interim
    core-only lock left carveout-100 unresolved because the
    free-running memory clock wobbled the L2 plateau).
    """

    def test_default_capacity_resolves_from_card(self) -> None:
        cfg = resolve_line_capacity()
        self.assertEqual(cfg.property_name, "l1_capacity_bytes")
        self.assertEqual(cfg.capacity_bytes, 98304)
        self.assertEqual(cfg.line_capacity, 98304 // LINE_BYTES)
        self.assertEqual(cfg.source, "card")

    def test_swept_carveout_points_resolve(self) -> None:
        expected_lines = {0: 768, 25: 640, 50: 384, 75: 256, 100: 256}
        seen = []
        for pct, lines in expected_lines.items():
            cfg = resolve_line_capacity(carveout_pct=pct)
            self.assertEqual(cfg.line_capacity, lines, pct)
            self.assertEqual(
                cfg.property_name,
                f"l1_capacity_carveout_{pct}_pct_bytes")
            self.assertEqual(cfg.capacity_bytes,
                             lines * LINE_BYTES)
            seen.append(cfg.line_capacity)
        # capacity never grows as the shared-memory carveout grows
        self.assertEqual(seen, sorted(seen, reverse=True))

    def test_carveout_fixture_tracks_swept_capacity(self) -> None:
        """Two-pass cyclic walk over N distinct lines: the second
        pass fully hits when N fits the resolved capacity and
        zero-hit LRU-thrashes when it does not. N=500 separates
        carveout 0 (768 lines) from carveout 50 (384); N=300
        separates carveout 50 from carveout 100 (256)."""
        def total_hits(n_lines: int, line_capacity: int) -> int:
            stream = _stream(list(range(n_lines)) * 2)
            result = walk_warp_pool(
                [("w0", stream)], line_capacity=line_capacity,
                hit_latency=HIT, miss_latency=MISS, mshr_count=None,
                mode="warp_pool")
            return result.hit_count

        cap_0 = resolve_line_capacity(carveout_pct=0).line_capacity
        cap_50 = resolve_line_capacity(carveout_pct=50).line_capacity
        cap_100 = resolve_line_capacity(carveout_pct=100).line_capacity
        self.assertEqual(total_hits(500, cap_0), 500)
        self.assertEqual(total_hits(500, cap_50), 0)
        self.assertEqual(total_hits(300, cap_50), 300)
        self.assertEqual(total_hits(300, cap_100), 0)

    def test_unswept_carveout_point_rejected(self) -> None:
        """The committed card did not sweep 60%: no card entry, no
        typed default -> rejected."""
        with self.assertRaises(WalkerConfigError):
            resolve_line_capacity(carveout_pct=60)

    def test_unsourced_capacity_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = _temp_card(td, {})
            with self.assertRaises(WalkerConfigError):
                resolve_line_capacity(bad)
            with self.assertRaises(WalkerConfigError):
                resolve_line_capacity(bad, carveout_pct=50)

    def test_typed_capacity_has_no_documented_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            card = _temp_card(td, {"l1_capacity_bytes": {
                "kind": "typed",
                "typed_reason": "geometry_unresolved"}})
            with self.assertRaises(WalkerConfigError):
                resolve_line_capacity(card)

    def test_misaligned_capacity_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            card = _temp_card(td, {"l1_capacity_bytes": {
                "kind": "value", "value": 1000}})
            with self.assertRaises(WalkerConfigError):
                resolve_line_capacity(card)

    def test_value_marker_without_value_rejected(self) -> None:
        """kind=value with no value field is a typed failure, not a
        raw KeyError."""
        with tempfile.TemporaryDirectory() as td:
            card = _temp_card(td, {"l1_capacity_bytes": {
                "kind": "value"}})
            with self.assertRaises(WalkerConfigError):
                resolve_line_capacity(card)

    def test_non_integer_carveout_pct_rejected(self) -> None:
        """A fractional or bool pct must not silently snap to a
        swept point's property name."""
        with self.assertRaises(WalkerConfigError):
            resolve_line_capacity(carveout_pct=50.5)
        with self.assertRaises(WalkerConfigError):
            resolve_line_capacity(carveout_pct=True)


class MappingReplacementTest(unittest.TestCase):
    """Replacement / set mapping per card evidence: the committed
    card types both `geometry_unresolved`, so LRU and the
    fully-associative walk are DOCUMENTED approximations with
    provenance notes — and ONLY then; everything else fails closed."""

    def test_replacement_documented_approximation_when_card_typed(
            self) -> None:
        cfg = resolve_replacement_policy()
        self.assertEqual(cfg.modeled, "lru")
        self.assertEqual(cfg.source, "documented_approximation")
        self.assertIn("geometry_unresolved", cfg.note)

    def test_set_mapping_documented_approximation_when_card_typed(
            self) -> None:
        cfg = resolve_set_mapping()
        self.assertEqual(cfg.modeled, "fully_associative")
        self.assertEqual(cfg.source, "documented_approximation")
        self.assertIn("geometry_unresolved", cfg.note)

    def test_absent_properties_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = _temp_card(td, {})
            with self.assertRaises(WalkerConfigError):
                resolve_replacement_policy(bad)
            with self.assertRaises(WalkerConfigError):
                resolve_set_mapping(bad)

    def test_card_valued_lru_accepted_as_card_sourced(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            card = _temp_card(td, {"l1_replacement_policy": {
                "kind": "value", "value": "lru"}})
            cfg = resolve_replacement_policy(card)
            self.assertEqual((cfg.modeled, cfg.source), ("lru", "card"))
            self.assertIsNone(cfg.note)

    def test_card_valued_non_lru_replacement_rejected(self) -> None:
        """A card-VALUED non-LRU policy must never silently degrade
        to the LRU approximation."""
        with tempfile.TemporaryDirectory() as td:
            card = _temp_card(td, {"l1_replacement_policy": {
                "kind": "value", "value": "pseudo_random"}})
            with self.assertRaises(WalkerConfigError):
                resolve_replacement_policy(card)

    def test_card_valued_ways_rejected_until_modeled(self) -> None:
        """A card-VALUED ways count needs a set-indexed pool walk the
        core does not model yet — fail closed, never flatten."""
        with tempfile.TemporaryDirectory() as td:
            card = _temp_card(td, {"l1_associativity_ways": {
                "kind": "value", "value": 4}})
            with self.assertRaises(WalkerConfigError):
                resolve_set_mapping(card)

    def test_malformed_replacement_value_rejected(self) -> None:
        """A non-string policy value is a malformed card, not an
        unmodeled policy — typed rejection either way."""
        with tempfile.TemporaryDirectory() as td:
            for bad in (0, True, None, ["lru"]):
                card = _temp_card(td, {"l1_replacement_policy": {
                    "kind": "value", "value": bad}})
                with self.assertRaises(WalkerConfigError, msg=bad):
                    resolve_replacement_policy(card)


class WalkGeometryTest(unittest.TestCase):
    def test_composed_geometry_resolves_and_walks(self) -> None:
        geo = resolve_walk_geometry(carveout_pct=50)
        self.assertEqual(geo.capacity.line_capacity, 384)
        self.assertEqual(geo.replacement.modeled, "lru")
        self.assertEqual(geo.set_mapping.modeled, "fully_associative")
        self.assertEqual(geo.mshr.source, "documented_default")
        self.assertLess(geo.latencies.hit_latency,
                        geo.latencies.miss_latency)
        result = walk_warp_pool(
            [("w0", _stream([1, 1]))],
            line_capacity=geo.capacity.line_capacity,
            hit_latency=geo.latencies.hit_latency,
            miss_latency=geo.latencies.miss_latency,
            mshr_count=geo.mshr.count, mode="warp_pool")
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.miss_count, 1)
        self.assertEqual(result.line_alloc_miss_count, 1)


class CardSourcedConfigTest(unittest.TestCase):
    def test_mshr_resolver_uses_documented_default_with_note(self) -> None:
        """The REAL committed card types the MSHR property, so the
        resolver must return the documented default WITH the
        sensitivity note (the AC's positive arm)."""
        cfg = resolve_mshr_count()
        self.assertEqual(cfg.source, "documented_default")
        self.assertEqual(cfg.count, DOCUMENTED_DEFAULT_MSHR_COUNT)
        self.assertIsNotNone(cfg.sensitivity_note)
        self.assertIn("sensitivity", cfg.sensitivity_note)

    def test_latency_resolver_reads_card_values(self) -> None:
        cfg = resolve_latencies()
        self.assertEqual(cfg.source, "card")
        self.assertLess(cfg.hit_latency, cfg.miss_latency)

    def test_unsourced_geometry_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = _temp_card(td, {})
            with self.assertRaises(WalkerConfigError):
                resolve_mshr_count(bad)
            with self.assertRaises(WalkerConfigError):
                resolve_latencies(bad)

    def test_card_valued_mshr_value_is_audited(self) -> None:
        """The kind=value marker alone is never trusted: the VALUE
        must be a positive integer (a fractional, zero, negative,
        bool, string, or absent value fails closed instead of
        silently entering the walk as the MSHR depth)."""
        def mshr_card(td, value):
            return _temp_card(td, {"mshr_outstanding_misses": {
                "kind": "value", "value": value}})

        with tempfile.TemporaryDirectory() as td:
            good = resolve_mshr_count(mshr_card(td, 48))
            self.assertEqual((good.count, good.source), (48, "card"))
            self.assertIsNone(good.sensitivity_note)
            for bad in (31.7, 0, -5, True, "48", None):
                with self.assertRaises(WalkerConfigError, msg=bad):
                    resolve_mshr_count(mshr_card(td, bad))
            with self.assertRaises(WalkerConfigError):
                resolve_mshr_count(_temp_card(
                    td, {"mshr_outstanding_misses": {"kind": "value"}}))

    def test_card_latency_values_are_audited(self) -> None:
        """Latencies drive the whole latency-proportional schedule:
        non-positive, non-finite, non-numeric, or hit >= miss values
        fail closed instead of inverting the model."""
        def lat_card(td, hit, miss):
            return _temp_card(td, {
                "l1_hit_latency_cycles": {
                    "kind": "value", "value": hit},
                "l2_hit_latency_cycles": {
                    "kind": "value", "value": miss}})

        with tempfile.TemporaryDirectory() as td:
            cfg = resolve_latencies(lat_card(td, 33.0, 250.0))
            self.assertEqual(cfg.source, "card")
            for hit, miss in ((-7.0, 250.0), (0.0, 250.0),
                              (250.0, 33.0), (33.0, 33.0),
                              ("33", 250.0), (33.0, float("inf"))):
                with self.assertRaises(WalkerConfigError,
                                       msg=(hit, miss)):
                    resolve_latencies(lat_card(td, hit, miss))


if __name__ == "__main__":
    unittest.main()
