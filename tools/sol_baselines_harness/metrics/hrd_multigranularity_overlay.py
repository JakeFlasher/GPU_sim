"""Maeda 2017 Algorithm 1 multi-granularity HRD
overlay (deployed-scope upper-bound enhancement).

Profiles reuse distance simultaneously at four
granularities — sector (32 B), line (128 B),
allocation, region (4 KiB) — in a single
O(N log M) pass over an access stream. Per
Maeda 2017 (HPCA Algorithm 1), four independent
Fenwick-tree LRU stack-distance trackers share
the timestamp axis; each tracker's identity
history is the per-granularity identity from
the access record.

The conditional-descent hit decision per access
is:

    sector_hit AND (line_hit OR allocation_hit OR region_hit)

where each per-granularity hit fires when the
per-access reuse distance at that granularity
is non-cold AND strictly less than that
granularity's capacity. The strict less-than
convention matches the established sibling HRD
profiler: a cache of capacity ``C`` admits an
access whose stack distance is ``< C``; cold
accesses always miss; finite accesses with
``rd >= C`` miss; all other finite accesses
hit. Zero capacity at any granularity therefore
admits zero hits at that granularity, including
on RD=0 immediate re-touches. The aggregate
hit rate over the stream is the number of
conditional-descent hits divided by the total
access count; an empty stream returns ``0.0``.

Public surface:

  * ``GRANULARITY_NAMES`` — pinned tuple
    ``("sector", "line", "allocation", "region")``
    declaring iteration order.
  * ``GRANULARITIES`` — closed frozenset of the
    four granularity strings.
  * ``HRD_MULTIGRANULARITY_OVERLAY_TYPED_REASONS``
    — closed frozenset of typed-reason strings.
  * ``HRDMultiGranularityOverlayError(typed_reason,
    message)`` — typed exception.
  * ``MultiGranularityAccessRecord`` — frozen
    dataclass carrying the per-access identity
    tuple ``(sector_id, line_id, allocation_id,
    region_id)``.
  * ``PerGranularityRDHistograms`` — frozen
    dataclass with ``histograms: Mapping[str,
    RDHistogram]`` keyed by ``GRANULARITY_NAMES``.
    Each per-granularity ``RDHistogram``
    populates the ``finite_rds`` field with the
    exact per-access finite reuse distances at
    that granularity in stream order, so a
    downstream consumer can audit per-layer
    hit counts under the strict less-than
    convention without re-walking the stream.
  * ``MultiGranularityHRDOverlay`` class with
    ``walk(records, *, bin_edges)`` returning
    ``PerGranularityRDHistograms``.
  * ``conditional_descent_hit_rate(records, *,
    capacities_in_lines_at_granularity)`` —
    per-access conditional-descent evaluation
    under the strict less-than capacity
    convention; returns a float in
    ``[0.0, 1.0]``.

The module imports the deployed
``_FenwickReuseDistanceTracker`` and
``RDHistogram`` types from the sibling
``sharing_classifier`` deployed module to avoid
duplication. No calibration-subtree imports;
NCU quarantine preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from baselines_harness.metrics.sharing_classifier import (
    RD_BIN_EDGES_DEFAULT,
    RDHistogram,
    _FenwickReuseDistanceTracker,
    _assign_bin,
    _validate_bin_edges,
)


GRANULARITY_SECTOR: str = "sector"
GRANULARITY_LINE: str = "line"
GRANULARITY_ALLOCATION: str = "allocation"
GRANULARITY_REGION: str = "region"


GRANULARITY_NAMES: tuple[str, ...] = (
    GRANULARITY_SECTOR,
    GRANULARITY_LINE,
    GRANULARITY_ALLOCATION,
    GRANULARITY_REGION,
)


GRANULARITIES: frozenset[str] = frozenset(GRANULARITY_NAMES)


HRD_MULTIGRANULARITY_OVERLAY_TYPED_REASONS: frozenset[str] = frozenset({
    "hrd_overlay_input_invalid",
    "hrd_overlay_capacity_invalid",
    "hrd_overlay_bin_edges_invalid",
})


class HRDMultiGranularityOverlayError(Exception):
    """Raised by the multi-granularity HRD overlay
    on input or capacity contract violations.
    Mirrors the deployed
    ``SharingClassifierError`` shape.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class MultiGranularityAccessRecord:
    sector_id: int
    line_id: int
    allocation_id: int
    region_id: int


@dataclass(frozen=True)
class PerGranularityRDHistograms:
    """Per-granularity reuse-distance histograms
    produced by a single
    ``MultiGranularityHRDOverlay.walk`` pass.

    ``histograms`` maps each granularity name in
    ``GRANULARITY_NAMES`` to its ``RDHistogram``;
    each histogram is independently keyed by the
    corresponding identity field in the per-access
    records.
    """

    histograms: Mapping[str, RDHistogram]
    total_accesses: int


def _validate_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HRDMultiGranularityOverlayError(
            "hrd_overlay_input_invalid",
            f"{name} must be int; got {type(value).__name__}",
        )
    if value < 0:
        raise HRDMultiGranularityOverlayError(
            "hrd_overlay_input_invalid",
            f"{name} must be non-negative; got {value}",
        )
    return value


def _validate_record(record: MultiGranularityAccessRecord) -> None:
    if not isinstance(record, MultiGranularityAccessRecord):
        raise HRDMultiGranularityOverlayError(
            "hrd_overlay_input_invalid",
            f"record must be MultiGranularityAccessRecord; got "
            f"{type(record).__name__}",
        )
    _validate_int(record.sector_id, "sector_id")
    _validate_int(record.line_id, "line_id")
    _validate_int(record.allocation_id, "allocation_id")
    _validate_int(record.region_id, "region_id")


def _validate_capacities(
    capacities: Mapping[str, int],
) -> Mapping[str, int]:
    if not isinstance(capacities, Mapping):
        raise HRDMultiGranularityOverlayError(
            "hrd_overlay_capacity_invalid",
            f"capacities_in_lines_at_granularity must be a "
            f"mapping; got {type(capacities).__name__}",
        )
    for name in GRANULARITY_NAMES:
        if name not in capacities:
            raise HRDMultiGranularityOverlayError(
                "hrd_overlay_capacity_invalid",
                f"capacities_in_lines_at_granularity missing "
                f"required granularity {name!r}; got "
                f"{sorted(capacities.keys())}",
            )
        value = capacities[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise HRDMultiGranularityOverlayError(
                "hrd_overlay_capacity_invalid",
                f"capacities_in_lines_at_granularity[{name!r}] "
                f"must be int; got {type(value).__name__}",
            )
        if value < 0:
            raise HRDMultiGranularityOverlayError(
                "hrd_overlay_capacity_invalid",
                f"capacities_in_lines_at_granularity[{name!r}] "
                f"must be non-negative; got {value}",
            )
    return capacities


def _identity_for_granularity(
    record: MultiGranularityAccessRecord, granularity: str,
) -> int:
    if granularity == GRANULARITY_SECTOR:
        return record.sector_id
    if granularity == GRANULARITY_LINE:
        return record.line_id
    if granularity == GRANULARITY_ALLOCATION:
        return record.allocation_id
    if granularity == GRANULARITY_REGION:
        return record.region_id
    raise HRDMultiGranularityOverlayError(
        "hrd_overlay_input_invalid",
        f"unknown granularity {granularity!r}; valid names are "
        f"{sorted(GRANULARITIES)}",
    )


def _bin_counts_to_tuple(
    counts: Sequence[int],
) -> tuple[int, ...]:
    return tuple(counts)


def _wrap_bin_edges_error(error: Exception) -> Exception:
    """Translate a ``SharingClassifierError`` raised
    by the shared ``_validate_bin_edges`` helper
    into this module's typed-reason vocabulary so
    callers see a consistent exception type.
    """
    message = getattr(error, "message", str(error))
    return HRDMultiGranularityOverlayError(
        "hrd_overlay_bin_edges_invalid",
        message,
    )


class MultiGranularityHRDOverlay:
    """Single-pass Maeda 2017 Algorithm 1 walker.

    The walker materializes four independent
    Fenwick-tree LRU stack-distance trackers
    (one per granularity) sharing the global
    timestamp axis. For each access in the
    stream, every tracker observes the
    per-granularity identity at the current
    timestamp; the returned per-access RD is
    binned into the granularity's histogram.

    The walker is stateful only during ``walk``;
    a fresh ``MultiGranularityHRDOverlay`` is
    constructed per call (instance-level state is
    only the bin edges + capacity bookkeeping).
    """

    def __init__(self) -> None:
        pass

    def walk(
        self,
        records: Iterable[MultiGranularityAccessRecord],
        *,
        bin_edges: Sequence[int] = RD_BIN_EDGES_DEFAULT,
    ) -> PerGranularityRDHistograms:
        try:
            edges = _validate_bin_edges(bin_edges)
        except Exception as error:
            raise _wrap_bin_edges_error(error) from error
        records_seq = tuple(records)
        for record in records_seq:
            _validate_record(record)
        n = len(records_seq)
        if n == 0:
            empty = RDHistogram(
                bin_edges=edges,
                bin_counts=tuple([0] * (len(edges) + 1)),
                total=0,
                cold=0,
                finite_rds=(),
            )
            return PerGranularityRDHistograms(
                histograms={name: empty for name in GRANULARITY_NAMES},
                total_accesses=0,
            )

        trackers: dict[str, _FenwickReuseDistanceTracker] = {
            name: _FenwickReuseDistanceTracker(n)
            for name in GRANULARITY_NAMES
        }
        bin_counts_per_granularity: dict[str, list[int]] = {
            name: [0] * (len(edges) + 1)
            for name in GRANULARITY_NAMES
        }
        finite_rds_per_granularity: dict[str, list[int]] = {
            name: [] for name in GRANULARITY_NAMES
        }
        finite_count_per_granularity: dict[str, int] = {
            name: 0 for name in GRANULARITY_NAMES
        }
        cold_count_per_granularity: dict[str, int] = {
            name: 0 for name in GRANULARITY_NAMES
        }

        for timestamp, record in enumerate(records_seq, start=1):
            for name in GRANULARITY_NAMES:
                identity = _identity_for_granularity(record, name)
                rd, state = trackers[name].access(identity, timestamp)
                if state == "cold":
                    cold_count_per_granularity[name] += 1
                else:
                    assert rd is not None
                    bin_index = _assign_bin(rd, edges)
                    bin_counts_per_granularity[name][bin_index] += 1
                    finite_rds_per_granularity[name].append(rd)
                    finite_count_per_granularity[name] += 1

        histograms: dict[str, RDHistogram] = {
            name: RDHistogram(
                bin_edges=edges,
                bin_counts=_bin_counts_to_tuple(bin_counts_per_granularity[name]),
                total=finite_count_per_granularity[name],
                cold=cold_count_per_granularity[name],
                finite_rds=tuple(finite_rds_per_granularity[name]),
            )
            for name in GRANULARITY_NAMES
        }
        return PerGranularityRDHistograms(
            histograms=histograms,
            total_accesses=n,
        )


def conditional_descent_hit_rate(
    records: Iterable[MultiGranularityAccessRecord],
    *,
    capacities_in_lines_at_granularity: Mapping[str, int],
) -> float:
    """Compute the per-access conditional-descent
    hit rate per Maeda 2017.

    The hit decision per access is:

        sector_hit AND (line_hit OR allocation_hit
                        OR region_hit)

    where each per-granularity hit fires when the
    per-access reuse distance at that granularity
    is non-cold AND strictly less than the
    granularity's capacity (cache-capacity
    convention: a cache of capacity ``C_S``
    sectors admits an access whose stack
    distance is ``< C_S``). A cold per-
    granularity access is by convention a miss
    at that granularity, and zero capacity at
    any granularity precludes a hit at that
    granularity even on RD=0 immediate
    re-touches.

    Returns a float in ``[0.0, 1.0]``; an empty
    stream returns ``0.0``; any zero per-
    granularity capacity that participates in
    the decision drops that granularity to a
    miss for every access.
    """
    capacities = _validate_capacities(capacities_in_lines_at_granularity)
    records_seq = tuple(records)
    for record in records_seq:
        _validate_record(record)
    n = len(records_seq)
    if n == 0:
        return 0.0

    trackers: dict[str, _FenwickReuseDistanceTracker] = {
        name: _FenwickReuseDistanceTracker(n)
        for name in GRANULARITY_NAMES
    }
    hit_count = 0
    for timestamp, record in enumerate(records_seq, start=1):
        per_granularity_hit: dict[str, bool] = {}
        for name in GRANULARITY_NAMES:
            identity = _identity_for_granularity(record, name)
            rd, state = trackers[name].access(identity, timestamp)
            if state == "cold":
                per_granularity_hit[name] = False
            else:
                assert rd is not None
                per_granularity_hit[name] = (
                    rd < capacities[name]
                )
        sector_hit = per_granularity_hit[GRANULARITY_SECTOR]
        descent_hit = (
            per_granularity_hit[GRANULARITY_LINE]
            or per_granularity_hit[GRANULARITY_ALLOCATION]
            or per_granularity_hit[GRANULARITY_REGION]
        )
        if sector_hit and descent_hit:
            hit_count += 1
    return hit_count / n


__all__ = [
    "GRANULARITIES",
    "GRANULARITY_ALLOCATION",
    "GRANULARITY_LINE",
    "GRANULARITY_NAMES",
    "GRANULARITY_REGION",
    "GRANULARITY_SECTOR",
    "HRDMultiGranularityOverlayError",
    "HRD_MULTIGRANULARITY_OVERLAY_TYPED_REASONS",
    "MultiGranularityAccessRecord",
    "MultiGranularityHRDOverlay",
    "PerGranularityRDHistograms",
    "conditional_descent_hit_rate",
]
