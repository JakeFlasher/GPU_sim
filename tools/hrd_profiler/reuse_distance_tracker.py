"""Fenwick-tree-backed LRU stack reuse-distance tracker.

R53 replacement for the O(N^2) per-launch LRU loop in
`observation_builder._walk_launch` (lines 206-210), which used
`lru_list.index() / .remove() / .append()` and produced a hang on
multi-launch traces where the per-launch sector-stream expansion
exceeded a few hundred thousand records (warm_cold_pair_relaunch
trace_00 reaches 6.4 MB sector_stream.jsonl across 3 launches and
hung 40+ minutes during R52's canonical capture attempt).

Algorithm: Fenwick tree (Binary Indexed Tree) over a monotonically
increasing per-launch timestamp axis. For each identity we keep
only its latest timestamp; the BIT stores a `1` at each identity's
latest timestamp slot and `0` elsewhere. The reuse distance for
an access is the count of OTHER identities whose latest timestamp
is strictly greater than this identity's previous timestamp:

    rd = (live_count) - prefix_sum(prev_timestamp)

This matches the EXISTING observation-builder semantics
(0-indexed-from-top: an immediate re-touch returns rd=0; the
classic "abcabc" cyclic returns rd=2 on each repeat). It does NOT
match the stack_alg_sim/Olken Rust crate's 1-indexed semantics
(its oracle is `Some(distance + 1)`); we preserve our existing
contract because every `tools/hrd_profiler/test_observation_builder.py`
test pins the 0-indexed form.

Complexity: O(log N) per access, O(N log N) total per launch. The
BIT is backed by `array("q")` (signed int64 array) for compact
storage and predictable memory; this is meaningful at N up to 10^7
where a Python `list[int]` would inflate by ~28x.

Per Codex (R53 design review): use 4 independent trackers (one
per layer: sector, line, allocation, region) with a shared
timestamp axis. The live-set state must be independent because
sector/line/allocation/region collapse identities differently.
"""

from __future__ import annotations

from array import array


class _Fenwick:
    """Minimal 1-indexed Fenwick tree over int64 storage."""

    __slots__ = ("_n", "_tree")

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(
                f"Fenwick size must be positive; got {n}"
            )
        self._n = n
        # 1-indexed: index 0 is unused. Allocate n+1 slots.
        self._tree = array("q", [0] * (n + 1))

    def add(self, i: int, delta: int) -> None:
        """Add `delta` to position `i` (1-indexed)."""
        if i < 1 or i > self._n:
            raise IndexError(
                f"Fenwick index {i} out of bounds [1, {self._n}]"
            )
        tree = self._tree
        while i <= self._n:
            tree[i] += delta
            i += i & -i

    def prefix_sum(self, i: int) -> int:
        """Sum of values at positions [1, i] (1-indexed). i=0 → 0."""
        if i < 0:
            raise IndexError(f"Fenwick prefix_sum index {i} < 0")
        if i > self._n:
            i = self._n
        tree = self._tree
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s


class ReuseDistanceTracker:
    """LRU-stack reuse-distance tracker with O(log N) access.

    Use one instance per identity layer (sector/line/allocation/
    region in the HRD producer). All instances may share the same
    timestamp axis (the producer's monotonic counter); the BIT
    state must be per-instance.

    Returns `(rd, state)` where state in {"finite", "cold"}.
    `rd` is 0-indexed: rd=0 means immediate reuse with one other
    identity touched in between (matches the legacy
    `len(lru_list) - 1 - pos` semantics that pre-R53 observation
    builders pinned).
    """

    __slots__ = ("_bit", "_last_seen", "_live", "_max_t")

    def __init__(self, max_accesses: int) -> None:
        if max_accesses <= 0:
            raise ValueError(
                f"max_accesses must be positive; got {max_accesses}"
            )
        self._bit = _Fenwick(max_accesses)
        # Maps identity → its latest timestamp (1-indexed).
        self._last_seen: dict[object, int] = {}
        # Number of distinct identities currently alive (each has
        # exactly one `1` in the BIT).
        self._live = 0
        self._max_t = max_accesses

    def access(self, identity: object, t: int) -> tuple[int | None, str]:
        """Record an access of `identity` at timestamp `t` (>= 1).

        Returns `(rd, state)`:
        - first sight of `identity`: `(None, "cold")`.
        - re-touch: `(rd, "finite")` where `rd` is the number of
          OTHER distinct identities whose latest pre-`t`
          timestamp is greater than `identity`'s previous
          timestamp. rd=0 → no other identity was touched between.
        """
        if t < 1 or t > self._max_t:
            raise IndexError(
                f"timestamp {t} out of bounds [1, {self._max_t}]"
            )
        prev = self._last_seen.get(identity)
        if prev is None:
            self._bit.add(t, 1)
            self._last_seen[identity] = t
            self._live += 1
            return None, "cold"

        # rd = # of OTHER identities whose latest is > prev.
        # Equivalently: live - count of latest-timestamps in [1, prev].
        # Since `identity` itself has its `1` at `prev` right now,
        # the prefix sum at `prev` includes that `1`; so the
        # subtraction nets out to (live - 1 - latest_in_(prev, ...))
        # which is exactly the count of OTHER identities with
        # latest-timestamp > prev. That's the 0-indexed rd.
        rd = self._live - self._bit.prefix_sum(prev)
        # Move identity's bit from prev → t.
        self._bit.add(prev, -1)
        self._bit.add(t, 1)
        self._last_seen[identity] = t
        return rd, "finite"
