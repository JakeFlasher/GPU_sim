"""Warp-pool L1 walker core: scheduler-aware interleaving + finite
MSHR.

Replaces the block-by-block sequential walk (`_walk_one_launch_fa`)
with the Nugteren-style warp-pool schedule: each RESIDENT warp issues
ONE memory request per turn and re-enters the pool delayed by the
request's latency (hit -> hit_latency, miss -> miss_latency), so a
fast-hitting warp naturally issues more requests per unit time than a
missing streamer — the interleaving the sequential walk cannot
express. Co-residency comes from launch geometry + occupancy: warps
beyond the residency limit are admitted in arrival order as resident
warps retire (wave-sequenced admission).

Finite-MSHR model with TWO distinguishable in-flight fill kinds, both
consuming MSHR capacity:

- line-ALLOCATION fills, keyed by line: the line is not resident
  until its fill time; any access to the line meanwhile MERGES by
  line (no new entry, no second fill — the access still counts as a
  miss and becomes ready at the fill time, and the fill services
  every sector the merged accesses touched);
- resident-line SECTOR fills, keyed by (line, sector): only an access
  to the SAME missing sector merges; the line's other,
  already-resident sectors still HIT (a per-sector fetch never
  shadows the rest of its line — sector-present hit testing precedes
  sector-merge handling). Distinct missing sectors of the same
  resident line are each their own sector-resident miss with their
  own entry, never a collapsed line fill.

When every entry is busy the warp STALLS and retries the same access
when the earliest fill (of either kind) frees an entry. Fills are
applied lazily at issue time: an in-flight line is NOT in the LRU —
and an in-flight sector is NOT in its line's residency set — until
the fill time passes, so accesses-under-fill are merges, never LRU
hits; a sector fill whose line was evicted mid-flight is dropped
(sector residency dies with the line).

Two-layer (32B sector / 128B line) accounting: the hit decision is
sector-with-line-allocation (the same contract as the existing walker
and the hrd second-layer machinery — a hit needs the LINE resident in
the LRU AND the touched SECTOR already fetched into it), and the miss
stream is SPLIT into the two classes the flat single-granularity walk
conflates:

- ``line_alloc_miss_count``   — the line was not resident: a fresh
  line allocation (plus the LRU eviction it forces; the evicted
  line's sector-residency bits die with it — partial-line eviction);
- ``sector_miss_resident_line_count`` — the line was resident but the
  touched 32B sector had not been fetched (a partial-line sector
  fetch: no allocation, no eviction, but real L2 traffic and a full
  miss latency).

Invariant: ``miss_count == line_alloc + sector_resident + merge``.

Configuration is CARD-SOURCED: hit/miss latencies, the MSHR count,
and the L1 capacity (default or per-carveout, in 128B lines) come
from the committed P-chase static card via resolvers; a card property
typed ``geometry_unresolved`` resolves to a DOCUMENTED default or
approximation with a mandatory note (MSHR count -> the documented
default + sensitivity note; replacement -> LRU; set mapping ->
fully-associative). Everything else fails closed: the walker never
invents an unsourced geometry constant, never silently approximates a
card-VALUED non-LRU policy, and capacity has NO default at all.

Scheduling modes (the fixtures' negative arms):
- ``warp_pool``      — the real model (latency-proportional).
- ``round_robin``    — strict 1:1 turns, latency-blind (ablation).
- ``concatenated``   — the old block-by-block behavior (ablation).

Sector-accounting modes:
- ``two_layer``      — the real model (sector-with-line-allocation).
- ``line_flat``      — single-granularity ablation: any access to a
  resident line hits, blind to partial-line sector state.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

DEFAULT_CARD_PATH = _REPO_ROOT / "data/pchase/sm_86_static_card.json"

# Documented default when the card types the MSHR property
# geometry_unresolved: the card's own probe bounds outstanding misses
# from BELOW at 32 (per-load throughput still improving at k=32), and
# the dossier's walker line uses 32 as the per-SM working default.
DOCUMENTED_DEFAULT_MSHR_COUNT = 32
MSHR_SENSITIVITY_COUNTS = (16, 32, 64)


class WalkerConfigError(Exception):
    """An unsourced geometry constant (no card entry, no typed
    default) — fail closed."""


class MshrConfig(NamedTuple):
    count: int | None          # None = unlimited (ablation only)
    source: str                # "card" | "documented_default" | "ablation"
    sensitivity_note: str | None


class LatencyConfig(NamedTuple):
    hit_latency: float
    miss_latency: float
    source: str


class LineCapacityConfig(NamedTuple):
    line_capacity: int         # capacity in 128B lines
    capacity_bytes: int
    property_name: str         # the card property the value came from
    source: str                # always "card" — capacity has no default


class GeometryPolicyConfig(NamedTuple):
    modeled: str               # "lru" | "fully_associative"
    source: str                # "card" | "documented_approximation"
    note: str | None


class WalkGeometry(NamedTuple):
    capacity: LineCapacityConfig
    replacement: GeometryPolicyConfig
    set_mapping: GeometryPolicyConfig
    mshr: MshrConfig
    latencies: LatencyConfig


def _load_card(card_path: Path | None) -> dict:
    path = card_path or DEFAULT_CARD_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _require_value(prop: dict, prop_name: str):
    """The ``kind=value`` MARKER alone is never trusted: the value
    field must exist (and the per-resolver audits below must pass)."""
    if "value" not in prop:
        raise WalkerConfigError(
            f"the card property {prop_name} has kind=value but no "
            "value field; fail closed")
    return prop["value"]


def _card_int(prop: dict, prop_name: str) -> int:
    """Audit a card-VALUED integer: positive, integral, finite, and
    not a bool/string masquerading through coercion."""
    raw = _require_value(prop, prop_name)
    if prop_name.endswith("_bytes") and prop.get("unit") not in (None, "bytes"):
        raise WalkerConfigError(
            f"{prop_name} unit {prop.get('unit')!r} is not bytes; fail closed"
        )
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
            or not math.isfinite(raw) or raw != int(raw) or raw < 1:
        raise WalkerConfigError(
            f"{prop_name} value {raw!r} is not a positive integer; "
            "the marker said value but the VALUE fails the audit — "
            "fail closed")
    return int(raw)


def _card_positive_float(prop: dict | None, prop_name: str) -> float:
    if not prop or prop.get("kind") != "value":
        raise WalkerConfigError(
            "card latencies unavailable; an unsourced latency constant "
            "is rejected")
    raw = _require_value(prop, prop_name)
    if prop.get("unit") not in (None, "cycles_per_dependent_load"):
        raise WalkerConfigError(
            f"{prop_name} unit {prop.get('unit')!r} is not cycles_per_dependent_load; fail closed"
        )
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
            or not math.isfinite(raw) or raw <= 0:
        raise WalkerConfigError(
            f"{prop_name} value {raw!r} is not a positive finite "
            "latency; fail closed")
    return float(raw)


def resolve_mshr_count(card_path: Path | None = None) -> MshrConfig:
    card = _load_card(card_path)
    prop = card.get("properties", {}).get("mshr_outstanding_misses")
    if prop is None:
        raise WalkerConfigError(
            "the card has no mshr_outstanding_misses property; an "
            "unsourced MSHR constant is rejected")
    if prop.get("kind") == "value":
        return MshrConfig(
            _card_int(prop, "mshr_outstanding_misses"), "card", None)
    if prop.get("kind") == "typed":
        return MshrConfig(
            DOCUMENTED_DEFAULT_MSHR_COUNT,
            "documented_default",
            (
                "the card types mshr_outstanding_misses "
                "geometry_unresolved (the single-thread probe bounds "
                "outstanding misses from below without a saturation "
                "knee), so the walk uses the documented default "
                f"{DOCUMENTED_DEFAULT_MSHR_COUNT}; report results at "
                f"counts {list(MSHR_SENSITIVITY_COUNTS)} for "
                "sensitivity"
            ),
        )
    raise WalkerConfigError(
        f"unrecognized card property kind {prop.get('kind')!r}")


def resolve_line_capacity(
    card_path: Path | None = None,
    *,
    carveout_pct: int | None = None,
) -> LineCapacityConfig:
    """Carveout-aware L1 capacity from the card's swept properties.

    ``carveout_pct=None`` reads the default-carveout
    ``l1_capacity_bytes``; an integer percentage reads the matching
    ``l1_capacity_carveout_{pct}_pct_bytes`` sweep point. Capacity has
    NO documented default: a missing property (unswept point), a typed
    property, or a value that is not a positive multiple of the 128B
    line all fail closed.
    """
    card = _load_card(card_path)
    if carveout_pct is None:
        prop_name = "l1_capacity_bytes"
    else:
        if isinstance(carveout_pct, bool) \
                or not isinstance(carveout_pct, int):
            raise WalkerConfigError(
                f"carveout_pct must be an integer percentage matching "
                f"a swept card point; got {carveout_pct!r}")
        prop_name = f"l1_capacity_carveout_{carveout_pct}_pct_bytes"
    prop = card.get("properties", {}).get(prop_name)
    if prop is None:
        raise WalkerConfigError(
            f"the card has no {prop_name} property (carveout point not "
            "swept); an unsourced capacity constant is rejected")
    if prop.get("kind") != "value":
        raise WalkerConfigError(
            f"the card types {prop_name} "
            f"{prop.get('typed_reason')!r} and capacity has no "
            "documented default — fail closed")
    capacity_bytes = _card_int(prop, prop_name)
    if capacity_bytes % LINE_BYTES:
        raise WalkerConfigError(
            f"{prop_name}={capacity_bytes} is not a positive multiple "
            f"of the {LINE_BYTES}B line")
    return LineCapacityConfig(capacity_bytes // LINE_BYTES,
                              capacity_bytes, prop_name, "card")


def resolve_replacement_policy(
    card_path: Path | None = None,
) -> GeometryPolicyConfig:
    """Replacement policy per card evidence. The card's typed
    ``geometry_unresolved`` resolves to LRU as the DOCUMENTED
    approximation (note carries the provenance); a card-VALUED policy
    other than lru is never silently approximated."""
    card = _load_card(card_path)
    prop = card.get("properties", {}).get("l1_replacement_policy")
    if prop is None:
        raise WalkerConfigError(
            "the card has no l1_replacement_policy property; an "
            "unsourced replacement constant is rejected")
    if prop.get("kind") == "value":
        policy = _require_value(prop, "l1_replacement_policy")
        if not isinstance(policy, str):
            raise WalkerConfigError(
                f"l1_replacement_policy value {policy!r} is malformed "
                "(not a policy name); fail closed")
        if policy == "lru":
            return GeometryPolicyConfig("lru", "card", None)
        raise WalkerConfigError(
            f"card-sourced replacement policy {policy!r} is not "
            "modeled; fail closed rather than silently approximating")
    if prop.get("kind") == "typed":
        return GeometryPolicyConfig(
            "lru", "documented_approximation",
            "the card types l1_replacement_policy "
            f"{prop.get('typed_reason')!r}; LRU is the documented "
            "approximation pending card evidence")
    raise WalkerConfigError(
        f"unrecognized card property kind {prop.get('kind')!r}")


def resolve_set_mapping(
    card_path: Path | None = None,
) -> GeometryPolicyConfig:
    """Set mapping per card evidence. The card's typed
    ``geometry_unresolved`` ways resolve to a fully-associative
    DOCUMENTED approximation; a card-VALUED ways count would require a
    set-indexed pool walk this core does not model yet, so it fails
    closed instead of silently flattening real sets."""
    card = _load_card(card_path)
    prop = card.get("properties", {}).get("l1_associativity_ways")
    if prop is None:
        raise WalkerConfigError(
            "the card has no l1_associativity_ways property; an "
            "unsourced set-mapping constant is rejected")
    if prop.get("kind") == "value":
        raise WalkerConfigError(
            f"card-sourced associativity ways={prop.get('value')!r} "
            "needs a set-indexed pool walk that is not modeled yet; "
            "fail closed rather than silently treating sets as "
            "fully-associative")
    if prop.get("kind") == "typed":
        return GeometryPolicyConfig(
            "fully_associative", "documented_approximation",
            "the card types l1_associativity_ways "
            f"{prop.get('typed_reason')!r}; the fully-associative "
            "walk is the documented approximation pending card "
            "evidence")
    raise WalkerConfigError(
        f"unrecognized card property kind {prop.get('kind')!r}")


def resolve_walk_geometry(
    card_path: Path | None = None,
    *,
    carveout_pct: int | None = None,
) -> WalkGeometry:
    """Resolve every card-sourced input the pool walk needs in one
    call (capacity is carveout-aware; the rest are carveout-blind)."""
    return WalkGeometry(
        resolve_line_capacity(card_path, carveout_pct=carveout_pct),
        resolve_replacement_policy(card_path),
        resolve_set_mapping(card_path),
        resolve_mshr_count(card_path),
        resolve_latencies(card_path),
    )


def resolve_latencies(card_path: Path | None = None) -> LatencyConfig:
    card = _load_card(card_path)
    props = card.get("properties", {})
    hit_latency = _card_positive_float(
        props.get("l1_hit_latency_cycles"), "l1_hit_latency_cycles")
    miss_latency = _card_positive_float(
        props.get("l2_hit_latency_cycles"), "l2_hit_latency_cycles")
    if hit_latency >= miss_latency:
        raise WalkerConfigError(
            f"card hit latency {hit_latency} is not strictly below "
            f"the miss latency {miss_latency}; the "
            "latency-proportional schedule would invert — fail closed")
    return LatencyConfig(hit_latency, miss_latency, "card")


class WarpPoolResult(NamedTuple):
    hit_count: int
    miss_count: int
    bypass_count: int
    mshr_merge_count: int
    mshr_stall_events: int
    line_fill_count: int
    line_alloc_miss_count: int
    sector_miss_resident_line_count: int
    per_warp_hits: dict
    per_warp_misses: dict
    final_time: float


SECTORS_PER_LINE = 4  # 128B line / 32B sector — mirrors l1_walker
SECTOR_BYTES = 32
LINE_BYTES = SECTORS_PER_LINE * SECTOR_BYTES  # the 128B line


class WarpPoolAccessInvalid(Exception):
    """Unsupported access tuple arity (mirrors the existing walker's
    fail-closed arm)."""


def _normalize_access(access: tuple) -> tuple[int, int, bool]:
    """Coerce to the canonical ``(line_id, sector_id, l1_eligible)``
    shape under the EXISTING walker's contract (the adapter still
    emits both forms):

    - ``(line_id, l1_eligible)`` legacy 2-tuple: a line-level access;
      the sector id is synthesized as ``line_id * SECTORS_PER_LINE``
      (the line's first sector), degrading to line-level LRU for
      call sites that have not learned the sector contract;
    - ``(line_id, sector_id, l1_eligible)`` 3-tuple: the
      sector-with-line-allocation form.

    Treating a legacy 2-tuple's second element as a sector id would
    silently convert bypass records (``l1_eligible=False``) into
    modeled misses — the contract bug the independent review caught.
    """
    if len(access) == 2:
        line_id, l1_eligible = access
        return (int(line_id), int(line_id) * SECTORS_PER_LINE,
                bool(l1_eligible))
    if len(access) == 3:
        line_id, sector_id, l1_eligible = access
        return int(line_id), int(sector_id), bool(l1_eligible)
    raise WarpPoolAccessInvalid(
        f"access must be (line_id, l1_eligible) or "
        f"(line_id, sector_id, l1_eligible); got {access!r}")


def walk_warp_pool(
    per_warp_accesses: Sequence[tuple[object, Sequence[tuple]]],
    *,
    line_capacity: int,
    hit_latency: float,
    miss_latency: float,
    mshr_count: int | None,
    co_resident_warps: int | None = None,
    mode: str = "warp_pool",
    sector_accounting: str = "two_layer",
    warmup_launches: int = 0,
) -> WarpPoolResult:
    """Walk one SM's resident warps under the requested schedule mode.

    ``per_warp_accesses`` is an ordered sequence of (warp_key,
    access_stream) — block-then-warp order defines admission order.
    ``co_resident_warps`` caps simultaneous residency (None = all
    resident); retiring warps admit the next pending warp at their
    retirement time. ``sector_accounting`` selects the two-layer
    sector-with-line-allocation hit decision (the real model) or the
    ``line_flat`` single-granularity ablation that is blind to
    partial-line sector state (the sectored fixtures' negative arm).

    ``warmup_launches`` (default 0) mirrors the existing walker's
    NCU-style warmup semantics: the SAME traversal replays that many
    times with persistent line/sector cache state and DISCARDED
    counters (in-flight fills drain at each launch boundary; warps
    re-admit fresh each launch), then ONE measured launch runs
    against the warm cache and its counters are returned. When
    warmup is requested the access streams are materialized so they
    can be replayed (single-use iterators are only safe at
    warmup_launches=0).
    """
    if mode not in ("warp_pool", "round_robin", "concatenated"):
        raise ValueError(f"unknown mode {mode!r}")
    if sector_accounting not in ("two_layer", "line_flat"):
        raise ValueError(
            f"unknown sector_accounting {sector_accounting!r}")
    if line_capacity <= 0:
        raise WalkerConfigError("line_capacity must be positive")
    if warmup_launches < 0:
        raise WalkerConfigError(
            f"warmup_launches={warmup_launches} must be >= 0")
    flat = sector_accounting == "line_flat"

    # Persistent cache state (lines + sector residency) carried
    # across warmup launches into the measured launch.
    lru: list[int] = []
    sectors: dict[int, set[int]] = {}
    if warmup_launches:
        per_warp_accesses = [
            (key, list(stream)) for key, stream in per_warp_accesses
        ]
    for _ in range(warmup_launches):
        _walk_launch(
            per_warp_accesses, line_capacity=line_capacity,
            hit_latency=hit_latency, miss_latency=miss_latency,
            mshr_count=mshr_count,
            co_resident_warps=co_resident_warps, mode=mode,
            flat=flat, lru=lru, sectors=sectors)
    return _walk_launch(
        per_warp_accesses, line_capacity=line_capacity,
        hit_latency=hit_latency, miss_latency=miss_latency,
        mshr_count=mshr_count, co_resident_warps=co_resident_warps,
        mode=mode, flat=flat, lru=lru, sectors=sectors)


def _walk_launch(
    per_warp_accesses: Sequence[tuple[object, Sequence[tuple]]],
    *,
    line_capacity: int,
    hit_latency: float,
    miss_latency: float,
    mshr_count: int | None,
    co_resident_warps: int | None,
    mode: str,
    flat: bool,
    lru: list[int],
    sectors: dict[int, set[int]],
) -> WarpPoolResult:
    """One launch of the pool walk against caller-owned persistent
    cache state (``lru`` + ``sectors``). In-flight fill state and
    counters are per-launch (fills drain at the launch boundary via
    the final inf-flush)."""
    # --- per-launch in-flight + counter state ---
    # Two DISTINGUISHABLE in-flight fill kinds (both consume MSHR
    # capacity):
    # - line_fills: line-ALLOCATION fetches keyed by line — the line
    #   is not resident until its fill time; any access to the line
    #   meanwhile MERGES by line (the fill services every sector the
    #   merged accesses touched).
    # - sector_fills: resident-line SECTOR fetches keyed by
    #   (line, sector) — only the SAME missing sector merges; other
    #   already-resident sectors of the line still HIT (a per-sector
    #   fetch must never shadow the rest of its line).
    line_fills: dict[int, float] = {}              # line -> fill time
    sector_fills: dict[tuple[int, int], float] = {}

    hit_count = miss_count = bypass_count = 0
    merge_count = stall_events = fill_count = 0
    line_alloc_misses = sector_resident_misses = 0
    per_warp_hits: dict = {}
    per_warp_misses: dict = {}

    def _mshr_in_use() -> int:
        return len(line_fills) + len(sector_fills)

    def _earliest_fill() -> float:
        return min(list(line_fills.values())
                   + list(sector_fills.values()))

    def _apply_fills(now: float) -> None:
        # Replay due fills in CHRONOLOGICAL order: each fill's effect
        # is decided by the cache state AT ITS OWN FILL TIME, not by
        # its kind. Batching by kind is only correct when callers
        # poll between every pair of due times — under a sparse
        # schedule it would apply a LATER line re-allocation before
        # an EARLIER stale sector fill and resurrect a sector whose
        # line was evicted (the sector fill must drop at its own
        # timestamp instead). Line fills land before sector fills
        # ONLY at exactly equal timestamps, and the stable sort keeps
        # same-kind same-timestamp fills in issue order.
        due = [(fill_time, 0, line)
               for line, fill_time in line_fills.items()
               if fill_time <= now]
        due += [(fill_time, 1, key)
                for key, fill_time in sector_fills.items()
                if fill_time <= now]
        due.sort(key=lambda item: (item[0], item[1]))
        for fill_time, kind, key in due:
            if kind == 0:
                line = key
                del line_fills[line]
                if line in lru:
                    lru.remove(line)
                lru.insert(0, line)
                if len(lru) > line_capacity:
                    for evicted in lru[line_capacity:]:
                        sectors.pop(evicted, None)
                    del lru[line_capacity:]
            else:
                line, sector = key
                del sector_fills[key]
                if line in lru:
                    # the sector arrives into its (still-resident)
                    # line and re-references it at MRU
                    sectors.setdefault(line, set()).add(sector)
                    lru.remove(line)
                    lru.insert(0, line)
                # else: the line was evicted before this fill's own
                # timestamp — the fill is DROPPED (sector residency
                # dies with the line); a later re-allocation of the
                # same line never receives it.

    def _access_once(warp_key, line_id: int, sector_id: int,
                     now: float) -> tuple[str, float]:
        """Returns (outcome, ready_time). Outcomes: hit, miss, merge,
        stall (retry needed)."""
        nonlocal hit_count, miss_count, merge_count
        nonlocal stall_events, fill_count
        nonlocal line_alloc_misses, sector_resident_misses
        _apply_fills(now)
        if line_id in line_fills:
            # access while the line ALLOCATION is in flight: a MERGE
            # by line (counts as a miss, ready at the fill, no new
            # entry; the fill services this sector too).
            merge_count += 1
            miss_count += 1
            per_warp_misses[warp_key] = per_warp_misses.get(
                warp_key, 0) + 1
            sectors.setdefault(line_id, set()).add(sector_id)
            return "merge", line_fills[line_id]
        line_resident = line_id in lru and lru.index(
            line_id) < line_capacity
        # Sector-present HIT testing precedes sector-merge handling:
        # an in-flight fetch of one missing sector must not shadow
        # hits to the line's other, already-resident sectors.
        sector_present = line_resident and (
            flat or sector_id in sectors.get(line_id, set()))
        if sector_present:
            hit_count += 1
            per_warp_hits[warp_key] = per_warp_hits.get(warp_key, 0) + 1
            lru.remove(line_id)
            lru.insert(0, line_id)
            return "hit", now + hit_latency
        if line_resident and (line_id, sector_id) in sector_fills:
            # the SAME missing sector is already being fetched: MERGE
            # (counts as a miss, ready at that sector's fill).
            merge_count += 1
            miss_count += 1
            per_warp_misses[warp_key] = per_warp_misses.get(
                warp_key, 0) + 1
            return "merge", sector_fills[(line_id, sector_id)]
        # miss path: needs an MSHR entry (both fill kinds consume
        # capacity)
        if mshr_count is not None and _mshr_in_use() >= mshr_count:
            stall_events += 1
            return "stall", _earliest_fill()
        miss_count += 1
        per_warp_misses[warp_key] = per_warp_misses.get(warp_key, 0) + 1
        if line_resident:
            # two-layer split: the line is allocated but the touched
            # 32B sector has not been fetched — a partial-line sector
            # fetch (no new line fill, no eviction). The sector
            # becomes resident at its FILL time (lazy, mirroring
            # line fills), so a concurrent same-sector access merges
            # instead of falsely hitting.
            sector_resident_misses += 1
            sector_fills[(line_id, sector_id)] = now + miss_latency
        else:
            line_alloc_misses += 1
            fill_count += 1
            line_fills[line_id] = now + miss_latency
            sectors.setdefault(line_id, set()).add(sector_id)
        return "miss", now + miss_latency

    if mode == "concatenated":
        # The old behavior: each warp's full stream end-to-end.
        now = 0.0
        for warp_key, stream in per_warp_accesses:
            for access in stream:
                line_id, sector_id, eligible = _normalize_access(access)
                if not eligible:
                    bypass_count += 1
                    continue
                while True:
                    outcome, ready = _access_once(
                        warp_key, line_id, sector_id, now)
                    if outcome != "stall":
                        now = ready if outcome != "hit" else now
                        break
                    now = ready
        _apply_fills(float("inf"))
        return WarpPoolResult(hit_count, miss_count, bypass_count,
                              merge_count, stall_events, fill_count,
                              line_alloc_misses, sector_resident_misses,
                              per_warp_hits, per_warp_misses, now)

    # --- pooled modes (warp_pool / round_robin) ---
    streams = {key: iter(stream) for key, stream in per_warp_accesses}
    admission = [key for key, _ in per_warp_accesses]
    residency = (len(admission) if co_resident_warps is None
                 else max(1, co_resident_warps))
    heap: list[tuple[float, int, object]] = []
    seq = 0
    resident = 0
    next_admit = 0
    now = 0.0

    def _admit(at: float) -> None:
        nonlocal next_admit, resident, seq
        while resident < residency and next_admit < len(admission):
            key = admission[next_admit]
            next_admit += 1
            resident += 1
            heapq.heappush(heap, (at, seq, key))
            seq += 1

    _admit(0.0)
    rr_turn = 0.0
    while heap:
        ready, _, warp_key = heapq.heappop(heap)
        now = max(now, ready)
        access = next(streams[warp_key], None)
        if access is None:
            resident -= 1
            _admit(now)
            continue
        line_id, sector_id, eligible = _normalize_access(access)
        if not eligible:
            bypass_count += 1
            heapq.heappush(heap, (now, seq, warp_key))
            seq += 1
            continue
        outcome, next_ready = _access_once(
            warp_key, line_id, sector_id, now)
        if outcome == "stall":
            # retry the SAME access when an entry frees
            streams[warp_key] = _chain_front(
                (line_id, sector_id, True), streams[warp_key])
            heapq.heappush(heap, (next_ready, seq, warp_key))
            seq += 1
            continue
        if mode == "round_robin":
            # latency-blind: strict turn order regardless of outcome.
            rr_turn += 1.0
            next_ready = rr_turn
        heapq.heappush(heap, (next_ready, seq, warp_key))
        seq += 1
    _apply_fills(float("inf"))
    return WarpPoolResult(hit_count, miss_count, bypass_count,
                          merge_count, stall_events, fill_count,
                          line_alloc_misses, sector_resident_misses,
                          per_warp_hits, per_warp_misses, now)


def _chain_front(item: tuple, rest: Iterable[tuple]):
    yield item
    yield from rest


def co_resident_warp_limit(
    *,
    warps_per_block: int,
    blocks_per_sm_limit: int,
    occupancy_warp_limit: int = 48,
) -> int:
    """Co-residency from launch geometry + occupancy: resident warps
    per SM = min(blocks-per-SM cap x warps per block, the occupancy
    warp limit)."""
    if warps_per_block <= 0 or blocks_per_sm_limit <= 0:
        raise WalkerConfigError("geometry values must be positive")
    return min(warps_per_block * blocks_per_sm_limit,
               occupancy_warp_limit)
