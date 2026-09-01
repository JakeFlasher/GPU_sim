"""Proto-backed reuse-distance observation builder for the HRD
profiler.

R19 closure of Codex's R18 review directive #1: replace the JSONL-
backed synthetic-identity builder with a builder that walks the
NVBit threadblock proto records via
``tools.nvbit_tracer.import_trace_contract`` and emits one HRD
observation per active+predicated lane × per touched 32B sector.

Per-access identities are byte-address-derived per the R16 task10
artifact §1 hierarchy:

* **Sector**: ``byte_addr // 32``
* **Line**:   ``byte_addr // 128``
* **Allocation**: looked up from the
  ``extra_info/allocations.json`` sidecar; the matching entry's
  ``allocation_id`` is the identity. When no allocation entry's
  ``[base_address, base_address + size_bytes)`` range contains the
  address (e.g., a stack reference, a constant-memory access, or a
  registered allocation that has been freed before the trace
  capture), the task10 §1 default applies and the line address
  doubles as the allocation identity.
* **Region**: ``byte_addr // 4096`` (4 KiB).

Reuse distance uses **same-launch LRU-stack-distance semantics**.
The launch identity is the
``threadblocks/device_<id>/stream_<id>/kernel_<id>/`` directory:
each ``kernel_<id>`` subdirectory is one launch. ``cold`` fires for
the first observed identity within a launch; ``infinite`` is
reserved (the typed reason stays in the enum, but R19 does not
synthesize an infinity from in-launch evidence — every same-launch
observation is either ``cold`` or finite).

Hit boolean: the modeled hit derivation is the
**sector-layer-LRU-distance-vs-capacity** rule. A sector access is
modeled as a hit iff its sector reuse distance (in 32B-sector
units) is strictly less than
``cache_model_spec["capacity_bytes"] // 32``. Cold accesses are
misses; finite accesses with reuse distance ≥ capacity are misses;
all other finite accesses are hits. The modeled hit is the one the
profiler aggregates; downstream consumers get the per-bin hit-rate
fraction. This derivation is intentionally explicit per Codex's
R18 directive #4: there is no authoritative hit signal in the
trace today (NVBit Activity captures memory transactions, not L2
hits), so a documented modeled-hit rule is the correct pragmatic
contract.

The capability-backed builder consumes only the paths, immutable payload
snapshots, allocations mapping, and admission metadata exposed by
``AdmittedTraceSources``. ``build_launches_from_trace_dir`` remains a thin
compatibility wrapper that admits and opens the capability before invoking the
core. Admission/resource/filesystem failures propagate to the caller.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.nvbit_tracer import import_trace_contract  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    AdmittedTraceSources,
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)

# Use a relative-style import that works whether the package is
# discovered as `tools.hrd_profiler` (test invocation from repo
# root) or via the `_HERE`-on-sys.path idiom that
# `tools/hrd_profiler/profiler.py` already uses (run_suite.py
# spawning).
if __package__:
    from .reuse_distance_tracker import ReuseDistanceTracker
else:  # pragma: no cover — exercised only when imported as a
        # top-level module via sys.path insertion.
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from reuse_distance_tracker import ReuseDistanceTracker  # type: ignore[import-not-found]  # noqa: E402


# Default fallback allocation identity when the address is outside
# every known allocation range. Per R16 task10 §1: "one 128B line
# for task11 unless a device-specific allocation unit is
# available". Encoding the fallback as the line-address keeps the
# hierarchy descent rule consistent — the allocation identity
# never collides with a real allocation_id (which are positive
# ints from the allocator-registry sidecar).
_FALLBACK_ALLOCATION_KEY_PREFIX = "line:"


@dataclass(frozen=True)
class LaunchSurvivingState:
    """Surviving sector-LRU state captured at the end of a launch's
    Pass-2 walk. Used to seed the next launch's trackers when
    ``cross_launch_state_carryover=True``: the surviving identities
    pre-populate the next launch's LRU stack at synthetic timestamps
    so the per-slice (and optional flat HRD-second-layer) tracker
    starts warm rather than cold.

    Shape:
      - ``sec_rd_per_slice_survivors``: one inner list per L2 slice,
        each containing the sector ids touched in that slice during
        the prior launch, sorted by ascending last-touch timestamp
        (LRU oldest first). The outer list length equals the prior
        launch's ``num_l2_slices``.
      - ``sec_rd_flat_survivors``: when the prior launch ran with
        ``hrd_second_layer_enabled=True``, the surviving sector ids
        from the flat first-layer tracker, sorted LRU-oldest-first;
        otherwise ``None``.

    Coarser-layer trackers (line / allocation / region) are NOT
    carried: they are diagnostic reuse-distance signals, not part
    of the L2 hit decision the carryover targets. Carrying them
    would inflate the synthetic timestamps without affecting any
    consumer's hit-rate computation.
    """

    sec_rd_per_slice_survivors: list[list[int]]
    sec_rd_flat_survivors: list[int] | None


@dataclass
class _BuildResult:
    """Diagnostic shape returned to the controller. ``launches`` is
    a list of per-launch record dicts; ``contract_ok`` /
    ``contract_typed_reason`` / ``contract_detail`` mirror the
    `import_trace_contract.ValidationResult` semantics so the
    controller can surface the correct ``hrd_skip_reason`` typed
    reason.

    Per-launch record shape (closed mapping; one entry per
    threadblocks/device_*/stream_*/kernel_*/ directory):

      - ``observations``: ``list[dict]`` carrying the per-access
        observation dicts (`hit` / `sector_state` / `sector_rd` /
        `line_state` / `line_rd` / `allocation_state` /
        `allocation_rd` / `region_state` / `region_rd`); the
        per-access dict shape is unchanged from pre-Round-11.
      - ``cold_count`` / ``finite_miss_count`` /
        ``unique_sector_count`` / ``working_set_sector_count`` /
        ``reuse_distance_finite_p50`` / ``_p90`` / ``_p99`` /
        ``per_op_class_counts``: per-launch aggregated
        diagnostics produced by ``summarize_launch``.

    The diagnostics propagate on the launch record itself rather
    than via a parallel side-list so downstream consumers
    (`cache_predictor.py::_aggregate_launches_from_observations`
    and `profiler.py::compute_hrd_profile`) see them on the same
    boundary they already read."""

    launches: list[dict[str, Any]]
    contract_ok: bool
    contract_typed_reason: str | None
    contract_detail: str | None
    total_observations: int
    total_threadblocks: int
    cross_launch_state_carryover_applied_launch_count: int = 0


def _nearest_rank_percentile(
    sorted_values: list[int], percentile: float,
) -> int:
    """Nearest-rank percentile on a pre-sorted list of ints. Returns
    the value at rank `ceil(percentile/100 * N) - 1` (0-indexed).
    `percentile` is 50/90/99 in this module; the helper does not
    interpolate, so the return is always an int from the input
    list. Assumes the input is non-empty (caller filters)."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile requires non-empty input")
    import math
    rank = max(1, math.ceil(percentile / 100.0 * n))
    return sorted_values[rank - 1]


# MemoryOpAccessKind enum mirror. Values MUST stay 1:1 with the
# proto3 enum in
# `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto`.
# The dimension is orthogonal to MemoryOpClass: the kind tags the
# write-policy direction (load / store / atomic / async-copy),
# the class tags the memory space (global / shared / local / etc.).
_MEMORY_OP_ACCESS_KIND_UNSPECIFIED = 0
_MEMORY_OP_ACCESS_KIND_LOAD = 1
_MEMORY_OP_ACCESS_KIND_STORE = 2
_MEMORY_OP_ACCESS_KIND_ATOMIC = 3
_MEMORY_OP_ACCESS_KIND_ASYNC_COPY = 4


def summarize_launch(
    observations: list[dict[str, Any]],
    *,
    access_sector_ids: list[int],
    memory_op_class_counts: dict[int, int],
    memory_op_access_kind_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Aggregate per-access observations + the walker's running
    sector-id and op-class sequences into per-launch diagnostics.

    The aggregator does NOT change the modeled-hit rule, the
    reuse-distance tracker, or the per-access observation dict
    shape; it only emits new SUMMARY fields on the per-launch HRD
    output. Existing per-access tests on
    ``test_observation_builder.py`` stay green.

    Fields returned (closed shape):
      - ``cold_count``: count of observations with
        ``sector_state == "cold"``.
      - ``finite_miss_count``: count of observations with
        ``sector_state == "finite"`` AND ``hit is False``.
      - ``unique_sector_count``: cardinality of the
        ``access_sector_ids`` set (the launch's sector footprint).
      - ``working_set_sector_count``: count of distinct sectors
        that appear ≥2 times in ``access_sector_ids`` (i.e.,
        sectors that experienced at least one finite-state
        access — the "reuse-eligible" set).
      - ``reuse_distance_finite_p50`` / ``_p90`` / ``_p99``:
        nearest-rank percentiles over the sector-rd values of
        every finite-state observation. ``None`` when no finite
        observations occurred (e.g., an all-cold launch).
      - ``per_op_class_counts``: passthrough of the walker's
        per-instruction ``memory_op_class`` accumulator (keyed by
        ``MemoryOpClass`` enum int, valued by access count).
      - ``per_access_kind_counts``: passthrough of the walker's
        per-instruction ``memory_op_access_kind`` accumulator
        (keyed by ``MemoryOpAccessKind`` enum int, valued by
        access count). Absent from the input dict means the
        walker observed no instructions with the kind present.
      - ``load_count`` / ``store_count`` / ``atomic_count`` /
        ``async_copy_count`` / ``unspecified_kind_count``:
        convenience scalar aliases for the per-access-kind
        buckets. ``unspecified_kind_count`` carries the legacy
        bucket for traces emitted before the producer-side
        access-kind classifier landed.

    ``access_sector_ids`` length matches ``observations`` length
    when the walker passes both. The aggregator does not enforce
    that invariant (defensive: an empty observations list with an
    empty sector-id list returns the zero tally).

    ``memory_op_access_kind_counts`` defaults to an empty dict
    when omitted (back-compat for legacy callers that don't yet
    pass the kwarg)."""
    cold_count = 0
    finite_miss_count = 0
    finite_rds: list[int] = []
    for obs in observations:
        state = obs.get("sector_state")
        if state == "cold":
            cold_count += 1
            continue
        if state == "finite":
            if obs.get("hit") is False:
                finite_miss_count += 1
            rd = obs.get("sector_rd")
            if rd is not None:
                finite_rds.append(int(rd))

    unique_sector_count = len(set(access_sector_ids))
    sector_id_counts: dict[int, int] = {}
    for sid in access_sector_ids:
        sector_id_counts[sid] = sector_id_counts.get(sid, 0) + 1
    working_set_sector_count = sum(
        1 for c in sector_id_counts.values() if c >= 2
    )

    if finite_rds:
        finite_rds.sort()
        p50 = _nearest_rank_percentile(finite_rds, 50.0)
        p90 = _nearest_rank_percentile(finite_rds, 90.0)
        p99 = _nearest_rank_percentile(finite_rds, 99.0)
    else:
        p50 = None
        p90 = None
        p99 = None

    kind_counts = (
        dict(memory_op_access_kind_counts)
        if memory_op_access_kind_counts else {}
    )

    return {
        "cold_count": cold_count,
        "finite_miss_count": finite_miss_count,
        "unique_sector_count": unique_sector_count,
        "working_set_sector_count": working_set_sector_count,
        "reuse_distance_finite_p50": p50,
        "reuse_distance_finite_p90": p90,
        "reuse_distance_finite_p99": p99,
        "per_op_class_counts": dict(memory_op_class_counts),
        "per_access_kind_counts": kind_counts,
        "load_count": kind_counts.get(
            _MEMORY_OP_ACCESS_KIND_LOAD, 0,
        ),
        "store_count": kind_counts.get(
            _MEMORY_OP_ACCESS_KIND_STORE, 0,
        ),
        "atomic_count": kind_counts.get(
            _MEMORY_OP_ACCESS_KIND_ATOMIC, 0,
        ),
        "async_copy_count": kind_counts.get(
            _MEMORY_OP_ACCESS_KIND_ASYNC_COPY, 0,
        ),
        "unspecified_kind_count": kind_counts.get(
            _MEMORY_OP_ACCESS_KIND_UNSPECIFIED, 0,
        ),
    }


def _allocation_identity_for_address(
    addr: int,
    allocations: Mapping[int, Any] | None,
) -> str:
    """Look up the allocation identity for a byte address. When the
    sidecar is absent or no entry's range contains ``addr``, fall
    back to the line-address per R16 task10 §1.

    Returns a string identity (allocations live in the same key-
    space as the line fallback because they are interchangeable —
    a downstream consumer computes hit rates over identity
    equality, not numeric ordering)."""
    if allocations:
        for alloc_id, entry in allocations.items():
            base = entry.base_address
            if base <= addr < base + entry.size_bytes:
                return f"alloc:{alloc_id}"
    return f"{_FALLBACK_ALLOCATION_KEY_PREFIX}{addr // 128}"


def _round_robin_merge_tb_accesses(
    per_tb_accesses: list[list[tuple]],
) -> list[tuple]:
    """Round-robin merge per-TB access sequences.

    Given ``per_tb_accesses == [[a, b, c], [x, y, z], [p, q]]``,
    returns ``[a, x, p, b, y, q, c, z]`` (access[i] from each TB
    in order, then access[i+1], stopping when a TB runs out).

    This approximates concurrent-CTA execution where N CTAs on
    N SMs issue accesses in lockstep through the shared L2.
    Used by ``_walk_launch`` when
    ``cta_interleave_enabled=True``. When the per-TB list has
    one entry, the function reduces to the legacy sequential
    walk (a no-op).
    """
    if not per_tb_accesses:
        return []
    if len(per_tb_accesses) == 1:
        return list(per_tb_accesses[0])
    max_len = max(len(seq) for seq in per_tb_accesses)
    merged: list[tuple] = []
    for i in range(max_len):
        for seq in per_tb_accesses:
            if i < len(seq):
                merged.append(seq[i])
    return merged


def _extract_lru_order(tracker: ReuseDistanceTracker) -> list:
    """Return the tracker's currently-live identities sorted by
    last-touch timestamp ascending (LRU oldest first). Used at the
    end of a launch to capture the surviving state that the next
    launch will receive when `cross_launch_state_carryover=True`."""
    last_seen = tracker._last_seen
    return [
        identity for identity, _ts in sorted(
            last_seen.items(), key=lambda kv: kv[1],
        )
    ]


def _seed_tracker_with_survivors(
    tracker: ReuseDistanceTracker,
    survivors: list,
) -> int:
    """Pre-seed a fresh tracker with surviving identities from a
    prior launch at synthetic timestamps `1..k` (oldest first ->
    smallest timestamp). Returns `k`, the next-available timestamp
    offset the caller must add to its walk-time counter.

    The synthetic timestamps preserve LRU ordering: a re-touch in
    the next launch of an identity that was the OLDEST survivor
    receives the largest reuse distance (proportional to how many
    OTHER survivors sit above it in the stack); a re-touch of the
    NEWEST survivor receives the smallest. This is the exact
    invariant the L2-cache-carryover physical model targets: warm
    cache lines should be hits in the next launch, with the hit
    likelihood ordered by LRU recency.

    Mutates `tracker` in place. The tracker's `max_accesses` MUST
    already be sized to admit `len(survivors) + next-launch access
    count`.
    """
    for synthetic_t, identity in enumerate(survivors, start=1):
        tracker._bit.add(synthetic_t, 1)
        tracker._last_seen[identity] = synthetic_t
        tracker._live += 1
    return len(survivors)


def _walk_launch(
    threadblock_paths: list[str],
    *,
    sources: AdmittedTraceSources,
    allocations: Mapping[int, Any] | None,
    capacity_bytes: int,
    threadblock_pb2_mod: Any,
    num_l2_slices: int | None = None,
    cta_interleave_enabled: bool = False,
    hrd_second_layer_enabled: bool = False,
    inbound_state: LaunchSurvivingState | None = None,
) -> dict[str, Any]:
    """Walk every threadblock proto in one kernel-launch directory
    and emit per active-lane × touched-sector observations under
    the LRU-stack reuse-distance state. Returns the launch's
    observation list (in walk order — proto file lexicographic
    sort, then warp index, then instruction index, then lane index,
    then sector offset).

    The LRU-stack state is per-layer per-launch; identities cross
    threadblock boundaries within one launch (Codex's R18 review
    directive #4: same-launch LRU-stack-distance over real
    identities).

    R53: rewritten as a two-pass walk. Pass 1 collects per-access
    `(sector_id, line_id, allocation_id, region_id)` tuples in
    walk order. Pass 2 feeds them into 4 independent
    `ReuseDistanceTracker` instances (Fenwick-tree-backed,
    O(log N) per access), preserving the legacy 0-indexed-from-top
    `rd` semantics that the existing tests pin
    (test_observation_builder.py:613/631/670/683). The pre-R53
    implementation used `lru_list.index/remove/append` which was
    O(N) per access, producing an O(N²) hang on multi-launch
    traces — confirmed at R52 on `warm_cold_pair_relaunch
    trace_00` (3 kernel launches → 6.4 MB sector_stream.jsonl,
    HRD producer CPU-bound 40+ minutes).

    Per-launch diagnostics: pass 1 ALSO tallies per-instruction
    ``memory_op_class`` counts (incrementing by the number of
    sector-spanning accesses produced per instruction, NOT by
    instruction count). After pass 2, ``summarize_launch`` folds
    those plus the walked sector-id sequence into the per-launch
    diagnostics dict, which is merged with the ``observations``
    list into the closed-shape launch record dict returned by
    this function.

    Slice-aware sector capacity: when ``num_l2_slices > 1``, the
    sector layer uses ``num_l2_slices`` independent
    ``ReuseDistanceTracker`` instances (one per slice). Each sector
    routes to slice ``sector_id % num_l2_slices``. The per-access
    ``sector_rd`` is the reuse distance within that slice's stack;
    the hit boolean uses the per-slice sector capacity
    ``max(1, capacity_bytes // num_l2_slices // 32)``. The line /
    allocation / region layers stay fully-associative (they are
    coarser identities — slice partitioning targets the L2 sector
    cache specifically). When ``num_l2_slices == 1`` (the default),
    the walker collapses to a single sector tracker — byte-for-byte
    equivalent to the pre-Round-12 fully-associative LRU.
    """
    if not isinstance(sources, AdmittedTraceSources):
        raise TypeError("sources must be an AdmittedTraceSources capability")
    sources.verify_unchanged()
    if num_l2_slices is None:
        raise ValueError(
            "cache_model_geometry_missing: num_l2_slices must be resolved "
            "from an identified architecture before walking"
        )
    if num_l2_slices < 1:
        raise ValueError(
            f"num_l2_slices must be >= 1, got {num_l2_slices!r}"
        )
    sectors_per_capacity = max(
        1, capacity_bytes // num_l2_slices // 32
    )
    # Maeda 2017 Algorithm 1 second HRD layer: when enabled, a
    # flat first-layer sector tracker (single instance, NOT
    # per-slice) checks against the total-L2 sector capacity;
    # the existing per-slice second-layer trackers stay as-is.
    # Hit decision becomes `first_layer_hit AND second_layer_hit`.
    # When disabled (the default), the walker collapses to the
    # legacy per-slice-only hit decision — byte-for-byte
    # equivalent to the pre-flag walker output.
    flat_sector_capacity = max(1, capacity_bytes // 32)

    # Pass 1: walk every threadblock proto and collect per-access
    # identity tuples in walk order. The number of accesses is the
    # sum over all (warp × instruction × lane × sector) emissions;
    # this matches the canonical sector-stream record count.
    #
    # Each access tuple is
    # ``(sector_id, line_id, allocation_id, region_id,
    #    access_kind, memory_op_class)`` — the last two are the
    # producer-side direction + memory-space tags, propagated onto
    # the per-access observation dict so the predictor-side
    # write-policy aggregator can distinguish write-allocate paths
    # (counted into the L2 hit/miss tally) from no-write-allocate
    # paths (shared store, async-copy destination — skipped
    # because the access never reaches L2). Legacy instructions
    # without an `memory_op_access_kind` wire field surface as
    # UNSPECIFIED and preserve current behavior on the consumer
    # side.
    # Pass-1 per-TB accumulators. Each entry is the access
    # sequence emitted by one threadblock proto. Pass 2's LRU
    # walk consumes EITHER the legacy concatenation (sequential
    # per-TB walk, the byte-for-byte pre-Round-19 behavior) OR
    # the round-robin merge across per-TB sequences when
    # `cta_interleave_enabled=True` is in effect (the coarse
    # CTA/SM concurrent-execution approximation: each CTA issues
    # access[i] in lockstep, so sectors touched by one CTA appear
    # near the other CTAs' adjacent accesses in the LRU stack).
    # The four contract cells:
    #
    #   - sequential-cross-cta-reuse: legacy default; the entire
    #     other-CTA working set sits between same-sector touches
    #     across CTAs, producing larger reuse distances.
    #   - interleaved-cross-cta-reuse: round-robin merge brings
    #     same-sector accesses closer in time across CTAs, so
    #     reuse distances drop and more cross-CTA reuse is
    #     captured.
    #   - disjoint-no-change: when per-CTA working sets share no
    #     sectors, interleaving does not invent reuse — every
    #     access stays cold.
    #   - single-cta-no-effect: round-robin over one TB is a no-
    #     op.
    #
    # Regression-locked by `CtaInterleaveAggregationTest` in
    # `test_observation_builder.py`.
    per_tb_accesses: list[list[tuple[int, int, str, int, int, int]]] = []
    memory_op_class_counts: dict[int, int] = {}
    # Write-policy classification tally. The
    # `memory_op_access_kind` field is orthogonal to
    # `memory_op_class`: the kind tags the direction (load /
    # store / atomic / async-copy), the class tags the memory
    # space. The aggregator emits per-launch convenience scalars
    # (`load_count` / `store_count` / `atomic_count` /
    # `async_copy_count` / `unspecified_kind_count`) so consumers
    # can distinguish write-allocate paths from no-write-allocate
    # paths without re-deriving the kind from the opcode. For
    # legacy traces emitted before the producer-side classifier
    # landed, the field is absent on the wire and every access
    # routes to the unspecified-kind bucket so the back-compat
    # path stays admissible.
    memory_op_access_kind_counts: dict[int, int] = {}
    # Sort tb_paths by numeric CTA block coordinates so the
    # per-TB sequence walks CTAs in execution order. The
    # canonical .pb filename suffix is `..._X,Y,Z.pb` where the
    # `X,Y,Z` triple is the block coordinate. Lexicographic
    # sort of `sorted(tb_paths)` puts `..._10,0,0.pb` before
    # `..._2,0,0.pb` on real kernels with >=10 CTAs, which
    # under `cta_interleave_enabled=True` causes the
    # round-robin merge to interleave blocks out of CTA order
    # and produces incorrect modeled reuse distances /
    # hit rates. The numeric-by-block-coord sort matches the
    # CTA execution order; fall back to lexicographic name
    # when the suffix doesn't parse as `X,Y,Z` (legacy traces
    # or non-canonical names).
    def _tb_path_block_coord_sort_key(relative_path: str):
        name = PurePosixPath(relative_path).stem
        last_underscore = name.rfind("_")
        if last_underscore == -1:
            return (1, name)
        suffix = name[last_underscore + 1:]
        parts = suffix.split(",")
        if len(parts) != 3:
            return (1, name)
        try:
            x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
            return (0, x, y, z)
        except ValueError:
            return (1, name)

    ordered_paths = sorted(
        threadblock_paths, key=_tb_path_block_coord_sort_key
    )
    for relative_path in ordered_paths:
        snapshot = sources.read_threadblock_snapshot(relative_path)
        tb_accesses: list[tuple[int, int, str, int, int, int]] = []
        tb = threadblock_pb2_mod.threadblock()
        tb.ParseFromString(snapshot.payload)
        for warp_index in sorted(tb.warps.keys()):
            for inst in tb.warps[warp_index].instructions:
                if not inst.HasField("memory_op_class"):
                    continue
                if not inst.addresses:
                    continue
                and_mask = inst.active_mask & inst.predicate_mask
                if and_mask == 0:
                    continue
                # Per-instruction direction + memory-space tags
                # propagated onto each per-access observation
                # produced by this instruction. Legacy
                # instructions without an access-kind wire field
                # route to UNSPECIFIED.
                inst_op_class = int(inst.memory_op_class)
                if inst.HasField("memory_op_access_kind"):
                    inst_access_kind = int(inst.memory_op_access_kind)
                else:
                    inst_access_kind = _MEMORY_OP_ACCESS_KIND_UNSPECIFIED
                accesses_before_inst = len(tb_accesses)
                for addr_record in inst.addresses:
                    address_mask = import_trace_contract._address_lane_mask(
                        addr_record, and_mask
                    )
                    recon = import_trace_contract._reconstruct_lane_vector(
                        addr_record, address_mask
                    )
                    lane_vector = recon[0]
                    if lane_vector is None:
                        # Defensive: admission already accepted this
                        # record. A reconstruction failure here is
                        # a producer-side drift; skip so the
                        # observation list stays well-formed
                        # rather than crashing.
                        continue
                    data_width = (
                        addr_record.data_width
                        if addr_record.data_width
                        else inst.data_width
                    )
                    if data_width <= 0:
                        continue
                    for lane_index, lane_addr in enumerate(lane_vector):
                        if not ((address_mask >> lane_index) & 1):
                            continue
                        if lane_addr == import_trace_contract.LANE_INACTIVE_SENTINEL:
                            # Active+predicated lane that the
                            # producer left as a sentinel — skip
                            # rather than emit a synthetic 0-address
                            # observation.
                            continue
                        access_start = int(lane_addr)
                        access_end = access_start + int(data_width)
                        # Sector-edge accounting at 32B
                        # granularity: every memory access
                        # contributes one observation per 32B
                        # sector touched by its
                        # ``[access_start, access_end)`` byte
                        # range. The four contract cells:
                        #
                        #   - aligned-fits: a 32B-aligned
                        #     access whose width <= 32B records
                        #     exactly one sector.
                        #   - aligned-crosses: a 32B access at
                        #     a 16B-aligned-but-not-32B-aligned
                        #     address (e.g. byte 16) crosses
                        #     one 32B boundary and records two
                        #     sectors.
                        #   - sub-32B-fits: an access shorter
                        #     than 32B whose range stays inside
                        #     one sector records one sector
                        #     touch.
                        #   - sub-32B-crosses: an access
                        #     shorter than 32B whose range
                        #     crosses a 32B boundary records
                        #     two sector touches.
                        #
                        # Regression-locked by
                        # `SectorEdgeAccountingTest` in
                        # `test_observation_builder.py`.
                        first_sector = access_start // 32
                        last_sector = (access_end - 1) // 32
                        for sector_n in range(first_sector, last_sector + 1):
                            sector_byte_addr = sector_n * 32
                            line_id = sector_byte_addr // 128
                            region_id = sector_byte_addr // 4096
                            allocation_id = _allocation_identity_for_address(
                                sector_byte_addr, allocations
                            )
                            tb_accesses.append((
                                sector_n,
                                line_id,
                                allocation_id,
                                region_id,
                                inst_access_kind,
                                inst_op_class,
                            ))
                accesses_added = len(tb_accesses) - accesses_before_inst
                if accesses_added > 0:
                    memory_op_class_counts[inst_op_class] = (
                        memory_op_class_counts.get(inst_op_class, 0)
                        + accesses_added
                    )
                    memory_op_access_kind_counts[inst_access_kind] = (
                        memory_op_access_kind_counts.get(
                            inst_access_kind, 0,
                        )
                        + accesses_added
                    )
        per_tb_accesses.append(tb_accesses)

    # Merge per-TB sequences. Round-robin interleave approximates
    # concurrent-CTA execution; legacy concatenation preserves the
    # pre-Round-19 sequential per-TB walk byte-for-byte.
    if cta_interleave_enabled:
        accesses = _round_robin_merge_tb_accesses(per_tb_accesses)
    else:
        accesses = [a for tb_seq in per_tb_accesses for a in tb_seq]

    n = len(accesses)
    observations: list[dict[str, Any]] = []
    if n == 0:
        diagnostics = summarize_launch(
            observations,
            access_sector_ids=[],
            memory_op_class_counts=memory_op_class_counts,
            memory_op_access_kind_counts=memory_op_access_kind_counts,
        )
        # Empty-launch record carries an empty surviving state when
        # carryover is in play (no new identities touched; any
        # inbound survivors are passed through unchanged).
        empty_state = LaunchSurvivingState(
            sec_rd_per_slice_survivors=(
                list(inbound_state.sec_rd_per_slice_survivors)
                if inbound_state is not None
                else [[] for _ in range(num_l2_slices)]
            ),
            sec_rd_flat_survivors=(
                list(inbound_state.sec_rd_flat_survivors)
                if inbound_state is not None
                and inbound_state.sec_rd_flat_survivors is not None
                else ([] if hrd_second_layer_enabled else None)
            ),
        )
        result = {
            "observations": observations,
            "surviving_state": empty_state,
            **diagnostics,
        }
        sources.verify_unchanged()
        return result

    # Pass 2: per-slice sector trackers + 3 coarser-layer
    # trackers (line / allocation / region stay
    # fully-associative — slice partitioning targets the sector
    # cache specifically per the slice-geometry refinement). With
    # `num_l2_slices=1` the per-slice list degenerates to a
    # single tracker — byte-for-byte equivalent to the
    # pre-slice-aware fully-associative LRU.
    #
    # Cross-launch carryover: when `inbound_state` is provided, each
    # per-slice tracker (and the flat second-layer tracker, when
    # enabled) is sized to admit BOTH the inbound survivors AND this
    # launch's accesses, then pre-seeded with the survivors at
    # synthetic timestamps `1..k_slice`. The walk-time counter
    # starts at `k_slice + 1` per slice (per-tracker offset, not a
    # single global offset, because different slices receive
    # different survivor counts). When `inbound_state is None`
    # (default), the tracker behaves byte-for-byte identically to
    # the pre-Round-7 cold-start walker.
    if inbound_state is not None:
        slice_survivors = inbound_state.sec_rd_per_slice_survivors
        if len(slice_survivors) != num_l2_slices:
            raise ValueError(
                "inbound_state.sec_rd_per_slice_survivors length "
                f"({len(slice_survivors)}) must match num_l2_slices "
                f"({num_l2_slices})"
            )
        flat_survivors = inbound_state.sec_rd_flat_survivors
        if (flat_survivors is None) != (not hrd_second_layer_enabled):
            raise ValueError(
                "inbound_state.sec_rd_flat_survivors presence must "
                "match hrd_second_layer_enabled flag"
            )
    else:
        slice_survivors = [[] for _ in range(num_l2_slices)]
        flat_survivors = [] if hrd_second_layer_enabled else None

    sec_rd_per_slice = [
        ReuseDistanceTracker(max_accesses=n + len(slice_survivors[i]))
        for i in range(num_l2_slices)
    ]
    sec_rd_flat = (
        ReuseDistanceTracker(
            max_accesses=n + (len(flat_survivors) if flat_survivors else 0)
        )
        if hrd_second_layer_enabled else None
    )
    line_rd = ReuseDistanceTracker(max_accesses=n)
    alloc_rd = ReuseDistanceTracker(max_accesses=n)
    region_rd = ReuseDistanceTracker(max_accesses=n)

    # Per-slice timestamp offsets after seeding.
    slice_t_offset = [
        _seed_tracker_with_survivors(sec_rd_per_slice[i], slice_survivors[i])
        for i in range(num_l2_slices)
    ]
    # Per-slice walk-time counters (each slice has its own offset).
    slice_t_counter = list(slice_t_offset)
    flat_t_offset = (
        _seed_tracker_with_survivors(sec_rd_flat, flat_survivors)
        if (hrd_second_layer_enabled and flat_survivors)
        else 0
    )
    flat_t_counter = flat_t_offset

    for t, (
        sector_id,
        line_id,
        allocation_id,
        region_id,
        access_kind,
        op_class,
    ) in enumerate(accesses, start=1):
        slice_id = sector_id % num_l2_slices
        slice_t_counter[slice_id] += 1
        s_rd, s_state = sec_rd_per_slice[slice_id].access(
            sector_id, slice_t_counter[slice_id],
        )
        l_rd, l_state = line_rd.access(line_id, t)
        a_rd, a_state = alloc_rd.access(allocation_id, t)
        r_rd, r_state = region_rd.access(region_id, t)
        if hrd_second_layer_enabled:
            # The flat tracker is the first HRD layer (total-L2
            # sector capacity check); the per-slice tracker
            # above is the second layer (per-slice capacity
            # check). An access hits only if BOTH layers
            # admit it. The per-slice tracker's `s_rd` /
            # `s_state` is retained in the observation dict
            # unchanged so downstream consumers continue to
            # see the slice-level reuse-distance signal.
            flat_t_counter += 1
            flat_rd, flat_state = sec_rd_flat.access(
                sector_id, flat_t_counter,
            )
            first_layer_hit = bool(
                flat_state == "finite"
                and flat_rd is not None
                and flat_rd < flat_sector_capacity
            )
            second_layer_hit = bool(
                s_state == "finite"
                and s_rd is not None
                and s_rd < sectors_per_capacity
            )
            hit = first_layer_hit and second_layer_hit
        else:
            hit = bool(
                s_state == "finite"
                and s_rd is not None
                and s_rd < sectors_per_capacity
            )
        # Per-access write-policy + memory-space tags propagated
        # onto the observation dict so the predictor-side
        # aggregator can filter no-write-allocate paths (shared
        # stores, async-copy destination side) out of the L2
        # hit/miss tally without re-deriving the kind from the
        # opcode. Legacy traces (kind absent on wire) route to
        # UNSPECIFIED here, preserving current consumer-side
        # behavior on the existing committed corpus.
        observations.append({
            "hit": hit,
            "sector_rd": s_rd,
            "sector_state": s_state,
            "line_rd": l_rd,
            "line_state": l_state,
            "allocation_rd": a_rd,
            "allocation_state": a_state,
            "region_rd": r_rd,
            "region_state": r_state,
            "access_kind": access_kind,
            "memory_op_class": op_class,
        })

    diagnostics = summarize_launch(
        observations,
        access_sector_ids=[acc[0] for acc in accesses],
        memory_op_class_counts=memory_op_class_counts,
        memory_op_access_kind_counts=memory_op_access_kind_counts,
    )
    # Capture surviving sector-LRU state for cross-launch carryover.
    # The state is the per-slice (and optional flat-layer) tracker
    # contents in LRU oldest-first order. The next launch's walker
    # seeds its trackers from this state when
    # `cross_launch_state_carryover=True`. When carryover is
    # disabled at the build level, callers ignore this field.
    surviving_state = LaunchSurvivingState(
        sec_rd_per_slice_survivors=[
            _extract_lru_order(t) for t in sec_rd_per_slice
        ],
        sec_rd_flat_survivors=(
            _extract_lru_order(sec_rd_flat)
            if (hrd_second_layer_enabled and sec_rd_flat is not None)
            else None
        ),
    )
    result = {
        "observations": observations,
        "surviving_state": surviving_state,
        **diagnostics,
    }
    sources.verify_unchanged()
    return result


def build_launches_from_admitted_sources(
    *,
    sources: AdmittedTraceSources,
    capacity_bytes: int,
    num_l2_slices: int | None = None,
    cta_interleave_enabled: bool = False,
    hrd_second_layer_enabled: bool = False,
    cross_launch_state_carryover: bool = False,
) -> _BuildResult:
    """Walk every admitted threadblock payload and emit launch observations.

    The capability has already enforced the wire contract and immutable source
    generation. This core never validates or inventories a raw trace path.

    Launch grouping: each
    ``threadblocks/device_<id>/stream_<id>/kernel_<id>/``
    directory is one launch. Observations within a launch carry
    LRU-stack-distance reuse semantics over real byte-address-
    derived identities; observations from different launches are
    in separate lists.

    ``capacity_bytes`` is the resolved cache-model spec capacity;
    ``num_l2_slices`` (default 1) controls the slice-aware
    effective capacity. The per-launch hit boolean derives from
    ``per_slice_sector_rd < (capacity_bytes // num_l2_slices //
    32)``; the per-launch records' per-access dict shape is
    unchanged.

    ``cta_interleave_enabled`` (default False) selects whether
    multi-TB launches use the coarse concurrent-CTA round-robin
    merge model (True) or the legacy sequential per-TB walk
    (False — pre-Round-19 behavior). The default preserves
    byte-for-byte legacy output on the committed corpus; the
    production paths enable interleave for sm_86 to approximate
    cross-CTA L2 contention.

    ``cross_launch_state_carryover`` (default False) selects
    whether multi-launch traces share L2 state across kernel
    launches within a given device/stream. When True, the
    surviving sector-LRU state from launch N (per-slice + flat
    second-layer when enabled) seeds launch N+1's trackers at
    synthetic timestamps so the next launch starts warm rather
    than cold. State is scoped per (device, stream): cross-stream
    state does not bleed. The default (False) preserves the
    pre-Round-7 cold-start-per-launch behavior byte-for-byte and
    keeps multi-launch admitted-cell predictions stable in the
    refreshed sweep.
    """
    if not isinstance(sources, AdmittedTraceSources):
        raise TypeError("sources must be an AdmittedTraceSources capability")
    sources.verify_unchanged()
    if num_l2_slices is None:
        result = _BuildResult(
            launches=[],
            contract_ok=False,
            contract_typed_reason="cache_model_geometry_missing",
            contract_detail=(
                "num_l2_slices must be resolved from an identified "
                "architecture before walking"
            ),
            total_observations=0,
            total_threadblocks=0,
        )
        sources.verify_unchanged()
        return result
    admission = sources.admission
    threadblock_paths = sources.threadblock_paths
    if len(threadblock_paths) != admission.threadblock_count:
        raise TraceAdmissionError(
            "trace_snapshot_changed",
            "admitted threadblock path count disagrees with admission",
        )
    allocations = sources.allocations
    threadblock_pb2_mod = import_trace_contract._import_pb_modules()[0]
    launches: list[dict[str, Any]] = []
    total_threadblocks = 0
    # Independent build-path diagnostic for the bundle-provenance
    # scanner at
    # `tools/sm_86_predictor_accuracy_recovery/test_bundle_provenance_scanner.py`:
    # counts launches that actually received a non-None
    # inbound_state (i.e., second-and-later launches in a stream
    # when carryover=True). Stays 0 when carryover=False, when a
    # stream has only one kernel, OR when the call's first launch
    # would receive None state. This is what proves the walker
    # actually exercised the carryover seeding path, independent
    # of the predictor's `cache_model_cross_launch_state_carryover`
    # claim — guards against producer-side bugs that set the
    # bundle field without threading the kwarg.
    applied_launch_count = 0
    grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
    for relative_path in threadblock_paths:
        parts = PurePosixPath(relative_path).parts
        if (
            len(parts) != 5
            or parts[0] != "threadblocks"
            or not parts[1].startswith("device_")
            or not parts[2].startswith("stream_")
            or not parts[3].startswith("kernel_")
        ):
            sources.read_threadblock_snapshot(relative_path)
            raise TraceAdmissionError(
                "trace_source_not_admitted",
                f"threadblock source has an invalid launch path: {relative_path}",
            )
        grouped.setdefault((parts[1], parts[2]), {}).setdefault(
            parts[3], []
        ).append(relative_path)

    def _kernel_name_sort_key(name: str) -> tuple[int, str]:
        suffix = name.removeprefix("kernel_")
        try:
            return (0, f"{int(suffix):020d}")
        except ValueError:
            return (1, name)

    for stream_key in sorted(grouped):
        stream_state: LaunchSurvivingState | None = None
        for kernel_name in sorted(
            grouped[stream_key], key=_kernel_name_sort_key
        ):
            launch_paths = grouped[stream_key][kernel_name]
            total_threadblocks += len(launch_paths)
            if stream_state is not None:
                applied_launch_count += 1
            record = _walk_launch(
                launch_paths,
                sources=sources,
                allocations=allocations,
                capacity_bytes=capacity_bytes,
                threadblock_pb2_mod=threadblock_pb2_mod,
                num_l2_slices=num_l2_slices,
                cta_interleave_enabled=cta_interleave_enabled,
                hrd_second_layer_enabled=hrd_second_layer_enabled,
                inbound_state=stream_state,
            )
            if cross_launch_state_carryover:
                stream_state = record.get("surviving_state")
            launches.append(record)
    result = _BuildResult(
        launches=launches,
        contract_ok=True,
        contract_typed_reason=None,
        contract_detail=None,
        total_observations=sum(
            len(record["observations"]) for record in launches
        ),
        total_threadblocks=total_threadblocks,
        cross_launch_state_carryover_applied_launch_count=(
            applied_launch_count
        ),
    )
    sources.verify_unchanged()
    return result


def build_launches_from_trace_dir(
    trace_dir: Path,
    *,
    capacity_bytes: int,
    num_l2_slices: int | None = None,
    cta_interleave_enabled: bool = False,
    hrd_second_layer_enabled: bool = False,
    cross_launch_state_carryover: bool = False,
) -> _BuildResult:
    """Path-compatible wrapper around the admitted-source core."""
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        return build_launches_from_admitted_sources(
            sources=sources,
            capacity_bytes=capacity_bytes,
            num_l2_slices=num_l2_slices,
            cta_interleave_enabled=cta_interleave_enabled,
            hrd_second_layer_enabled=hrd_second_layer_enabled,
            cross_launch_state_carryover=cross_launch_state_carryover,
        )


def emit_canonical_pre_calibration_features(
    *,
    launch_summary: dict,
    cache_model_capacity_bytes: int,
    raw_prediction: float,
    sharing_locality: dict | None = None,
    hrd_profile: dict | None = None,
    sector_stream_records: object | None = None,
    grid_volume: int | None = None,
    device: str | None = None,
    sector_size_bytes: int = 32,
) -> dict:
    """Producer-side canonical pre-calibration feature emission.

    Computes the 6-feature block from the walker's launch summary +
    cache model spec + optional auxiliary sidecars. Wraps
    workloads.simbench_bw.canonical_features.assemble_canonical_features
    so observation_builder owns the producer-surface contract: every new
    bundle this module emits MAY include the canonical block by calling
    this helper with the walker's existing internal state.

    ``launch_summary`` is the output of ``summarize_launch`` (carries
    ``unique_sector_count``). Caller supplies ``raw_prediction`` (the
    pre-correction walker scalar; typically the arithmetic mean of
    per-launch hit rates) + ``cache_model_capacity_bytes`` (from the
    cache_model_spec). When ``sharing_locality`` / ``hrd_profile`` /
    sector_stream are not staged on the host, the helper passes through
    the canonical_features assembler's typed-error path (caller emits
    typed-unavailable rather than a partial block).
    """
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from workloads.simbench_bw.canonical_features import (
        assemble_canonical_features,
    )
    unique_sector_count = int(launch_summary.get("unique_sector_count", 0))
    # Build a minimal synthetic bundle shape that the assembler expects.
    synthetic_bundle = {
        "evidence": {"predicted_hit_rate_uncorrected": float(raw_prediction)},
        "cache_model_capacity_bytes": int(cache_model_capacity_bytes),
        "sector_size_bytes": int(sector_size_bytes),
        "device": device,
    }
    if grid_volume is not None:
        synthetic_bundle["cache_model_grid_volume"] = int(grid_volume)
    return assemble_canonical_features(
        predictor_bundle=synthetic_bundle,
        sharing_locality=sharing_locality or {},
        hrd_profile=hrd_profile or {},
        unique_sector_count=unique_sector_count,
        sector_stream_records=sector_stream_records,
        grid_volume=grid_volume,
    )


__all__ = [
    "build_launches_from_admitted_sources",
    "build_launches_from_trace_dir",
    "emit_canonical_pre_calibration_features",
    "summarize_launch",
    "_BuildResult",
]
