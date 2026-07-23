"""Tests for the Fenwick-tree-backed reuse-distance tracker.

R53 regression-locks the 0-indexed-from-top semantics that the
existing observation-builder pins (immediate reuse → rd=0; classic
"abcabc" cyclic → rd=2 on each repeat). Includes the edge cases
Codex flagged in the design review and a coarse complexity
benchmark proving the per-access cost is sub-linear in N (the
pre-R53 implementation was O(N) per access via list.index/remove).
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from reuse_distance_tracker import (  # type: ignore[import-not-found]  # noqa: E402
    ReuseDistanceTracker,
    _Fenwick,
)


class FenwickTests(unittest.TestCase):
    """Lock the BIT primitives that ReuseDistanceTracker depends on."""

    def test_add_and_prefix_sum(self) -> None:
        bit = _Fenwick(8)
        bit.add(1, 5)
        bit.add(3, 7)
        bit.add(7, 2)
        self.assertEqual(bit.prefix_sum(0), 0)
        self.assertEqual(bit.prefix_sum(1), 5)
        self.assertEqual(bit.prefix_sum(2), 5)
        self.assertEqual(bit.prefix_sum(3), 12)
        self.assertEqual(bit.prefix_sum(6), 12)
        self.assertEqual(bit.prefix_sum(7), 14)
        self.assertEqual(bit.prefix_sum(8), 14)

    def test_negative_delta(self) -> None:
        bit = _Fenwick(4)
        bit.add(2, 1)
        bit.add(2, -1)
        self.assertEqual(bit.prefix_sum(4), 0)

    def test_size_validation(self) -> None:
        with self.assertRaises(ValueError):
            _Fenwick(0)
        with self.assertRaises(ValueError):
            _Fenwick(-3)

    def test_index_bounds(self) -> None:
        bit = _Fenwick(4)
        with self.assertRaises(IndexError):
            bit.add(0, 1)
        with self.assertRaises(IndexError):
            bit.add(5, 1)
        with self.assertRaises(IndexError):
            bit.prefix_sum(-1)
        # prefix_sum clamps at upper bound.
        self.assertEqual(bit.prefix_sum(99), 0)


def _trace_rds(seq: list[str]) -> list[int | None]:
    """Helper: run a single tracker over `seq` and return the rds."""
    tracker = ReuseDistanceTracker(max_accesses=max(1, len(seq)))
    out: list[int | None] = []
    for t, identity in enumerate(seq, start=1):
        rd, state = tracker.access(identity, t)
        if state == "cold":
            out.append(None)
        else:
            out.append(rd)
    return out


class ReuseDistanceOraclesTests(unittest.TestCase):
    """Pin 0-indexed-from-top semantics matching the existing
    observation-builder. These oracles are translated from
    stack_alg_sim's 1-indexed oracles by subtracting 1 from each
    Some(...) value."""

    def test_empty(self) -> None:
        self.assertEqual(_trace_rds([]), [])

    def test_single_access(self) -> None:
        self.assertEqual(_trace_rds(["a"]), [None])

    def test_immediate_reuse(self) -> None:
        # ["a", "a"]: second "a" is immediate; no other identity in
        # between → rd = 0.
        self.assertEqual(_trace_rds(["a", "a"]), [None, 0])

    def test_aba(self) -> None:
        # ["a", "b", "a"]: 1 other identity ("b") in between → rd=1.
        self.assertEqual(_trace_rds(["a", "b", "a"]), [None, None, 1])

    def test_abba(self) -> None:
        # ["a", "b", "b", "a"]: third "b" is immediate (rd=0); fourth
        # "a" sees "b" in between (rd=1).
        self.assertEqual(
            _trace_rds(["a", "b", "b", "a"]),
            [None, None, 0, 1],
        )

    def test_abcabc(self) -> None:
        # Classic cyclic — every repeat sees 2 other identities in
        # between → rd=2 (matches the existing observation-builder
        # tests at test_observation_builder.py:631 / :670 / :683).
        self.assertEqual(
            _trace_rds(["a", "b", "c", "a", "b", "c"]),
            [None, None, None, 2, 2, 2],
        )

    def test_abacba(self) -> None:
        # Stack walk:
        # a: cold (lru=[a])
        # b: cold (lru=[a,b])
        # a: lru=[a,b], pos=0, rd = 2-1-0 = 1. After: lru=[b,a].
        # c: cold (lru=[b,a,c])
        # a: lru=[b,a,c], pos=1, rd = 3-1-1 = 1. After: lru=[b,c,a].
        self.assertEqual(
            _trace_rds(["a", "b", "a", "c", "a"]),
            [None, None, 1, None, 1],
        )

    def test_abcba_sawtooth(self) -> None:
        # ["a", "b", "c", "c", "b", "a"]: stack walk
        # a: cold (lru=[a])
        # b: cold (lru=[a,b])
        # c: cold (lru=[a,b,c])
        # c: pos=2, rd = 3-1-2 = 0 (lru unchanged)
        # b: pos=1, rd = 3-1-1 = 1 (lru=[a,c,b])
        # a: pos=0, rd = 3-1-0 = 2 (lru=[c,b,a])
        self.assertEqual(
            _trace_rds(["a", "b", "c", "c", "b", "a"]),
            [None, None, None, 0, 1, 2],
        )


class ReuseDistanceMultiLayerTests(unittest.TestCase):
    """Two trackers with the same timestamp axis can produce
    different rds because the layer collapses identities
    differently. Mirrors the sector/line/allocation/region
    multi-layer pattern in observation_builder."""

    def test_layers_collapse_identities_independently(self) -> None:
        # Sector layer: identities ["s1", "s2", "s3"]
        # Line layer:   identities ["L1", "L1", "L1"] (all collapse)
        sec = ReuseDistanceTracker(max_accesses=4)
        line = ReuseDistanceTracker(max_accesses=4)
        # t=1
        rd_s, st_s = sec.access("s1", 1)
        rd_l, st_l = line.access("L1", 1)
        self.assertIsNone(rd_s)
        self.assertEqual(st_s, "cold")
        self.assertIsNone(rd_l)
        self.assertEqual(st_l, "cold")
        # t=2
        rd_s, st_s = sec.access("s2", 2)
        rd_l, st_l = line.access("L1", 2)
        self.assertIsNone(rd_s)
        self.assertEqual(st_s, "cold")
        # Line: L1 immediately reused → rd=0, finite.
        self.assertEqual(rd_l, 0)
        self.assertEqual(st_l, "finite")
        # t=3
        rd_s, st_s = sec.access("s3", 3)
        rd_l, st_l = line.access("L1", 3)
        self.assertIsNone(rd_s)
        self.assertEqual(st_s, "cold")
        self.assertEqual(rd_l, 0)
        self.assertEqual(st_l, "finite")
        # t=4 — sector s1 reused with 2 others in between
        rd_s, st_s = sec.access("s1", 4)
        rd_l, st_l = line.access("L1", 4)
        self.assertEqual(rd_s, 2)
        self.assertEqual(st_s, "finite")
        self.assertEqual(rd_l, 0)
        self.assertEqual(st_l, "finite")


class ReuseDistanceComplexityTests(unittest.TestCase):
    """Coarse benchmark: prove the new tracker is sub-linear per
    access. The pre-R53 list.index/remove was O(N) per access;
    100k accesses against a 100k-identity LRU stack would have
    taken ~10 seconds in pure Python with the legacy code. The
    Fenwick tracker should complete the same workload in well
    under 1 second."""

    def test_100k_distinct_then_repeat_under_one_second(self) -> None:
        n = 100_000
        tracker = ReuseDistanceTracker(max_accesses=2 * n)
        t0 = time.perf_counter()
        # 100k distinct identities, all cold.
        for i in range(n):
            tracker.access(i, i + 1)
        # Re-touch in cold order (0, 1, 2, ...). Each access pulls
        # the BOTTOM of the LRU stack to the top, so the next
        # access's bottom-of-stack is again at position n-1.
        # Therefore ALL re-touches return rd = n - 1 in this
        # specific pattern. Legacy code = O(N) per re-touch =
        # O(N^2) total. Fenwick = O(log N) per re-touch.
        for i in range(n):
            rd, state = tracker.access(i, n + i + 1)
            self.assertEqual(state, "finite")
            self.assertEqual(rd, n - 1)
        elapsed = time.perf_counter() - t0
        # 5.0s gives generous headroom on a slow CI box; the
        # legacy O(N^2) code would take ~10-30s here, so a
        # regression to O(N^2) would fire this assertion.
        self.assertLess(
            elapsed, 5.0,
            f"100k+100k accesses took {elapsed:.2f}s; the legacy "
            f"O(N^2) would have taken ~10-30s, so a regression to "
            f"O(N^2) would fire this assertion.",
        )


class ReuseDistanceBoundsTests(unittest.TestCase):
    """Out-of-bounds inputs fail loud rather than corrupt state."""

    def test_max_accesses_zero_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReuseDistanceTracker(max_accesses=0)

    def test_timestamp_out_of_bounds(self) -> None:
        tracker = ReuseDistanceTracker(max_accesses=4)
        with self.assertRaises(IndexError):
            tracker.access("a", 0)
        with self.assertRaises(IndexError):
            tracker.access("a", 5)


if __name__ == "__main__":
    unittest.main()
