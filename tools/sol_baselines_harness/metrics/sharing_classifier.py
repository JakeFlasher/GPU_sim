"""Jiang 2022 Shalom per-access sharing classifier
(deployed-scope upper-bound overlay).

Each cache-line access is classified into one of
four GPU-adapted sharing buckets per Jiang 2022
"Shalom" (TPDS):

  * ``intra_warp`` — the identity was previously
    touched by the same warp.
  * ``intra_cta`` — previously touched by a
    different warp in the same CTA.
  * ``inter_cta`` — previously touched by a
    different CTA.
  * ``cold`` — first sight of the identity.

Per-bucket reuse-distance (RD) histograms feed
the analytical hit-rate formula

    m = sum_bucket weight_bucket * P[RD_bucket < C_S]

under the strict cache-capacity convention: a
cache of capacity ``C_S`` sectors admits an
access whose reuse distance is strictly less
than ``C_S``. ``C_S`` is the cache capacity in
sectors (equivalent identity units); the
per-bucket weight is the bucket's share of total
accesses (finite + cold) across all four
buckets. Cold accesses always miss (their stack
distance is ``+infinity`` by convention); finite
accesses with ``rd >= C_S`` miss; all other
finite accesses hit. Zero capacity therefore
admits zero hits even on RD=0 immediate
re-touches.

Public surface:

  * ``SHARING_CLASSES`` — closed frozenset of the
    four bucket strings.
  * ``SHARING_CLASS_NAMES`` — pinned tuple
    declaring iteration order
    (``intra_warp``, ``intra_cta``,
    ``inter_cta``, ``cold``).
  * ``SHARING_CLASSIFIER_TYPED_REASONS`` —
    closed frozenset of typed-reason strings.
  * ``SharingClassifierError(typed_reason,
    message)`` — typed exception following the
    project's calibrator-error pattern.
  * ``SharingAccessRecord`` — frozen dataclass
    carrying ``(identity, warp_id, cta_id)``
    per access.
  * ``ClassifiedAccess`` — frozen dataclass
    carrying the input fields plus
    ``sharing_class``.
  * ``RDHistogram`` — frozen dataclass with
    ``bin_edges``, ``bin_counts``, ``total``,
    ``cold``, and ``finite_rds``. The
    ``finite_rds`` tuple is the exact list of
    per-bucket finite reuse distances in
    classified-stream order and is the source
    of truth for analytical hit counting; the
    ``bin_counts`` are intentionally lossy
    (multiple distinct RDs share a bin) and are
    retained only for inspection / debugging.
  * ``RD_BIN_EDGES_DEFAULT`` — pinned default
    histogram bin edges (powers-of-two ladder
    mirroring the sibling project's HRD
    profiler).
  * ``classify_stream(records)`` — returns a
    tuple of ``ClassifiedAccess`` records.
  * ``compute_per_bucket_rd_histograms(classified,
    *, bin_edges=RD_BIN_EDGES_DEFAULT)`` — runs
    a SINGLE full-stream Fenwick walk over the
    classified sequence in original order,
    computes the per-access ``(rd, state)``
    against the FULL identity history, and
    appends the outcome to the histogram of
    ``access.sharing_class``. An internal
    cold/finite consistency check fails closed
    when ``classify_stream()``'s cold/non-cold
    label disagrees with the full-stream
    Fenwick state at the same timestamp.
  * ``analytical_hit_rate(per_bucket_histograms,
    *, cache_capacity_in_sectors)`` — weighted
    sum of per-bucket hit probabilities under
    the strict less-than convention; hits are
    counted exactly against each bucket's
    ``finite_rds`` field with no bin-edge
    interpolation.

The module imports only stdlib + the deployed
``baselines_harness.metrics.framework_contract``
module surface; no calibration-subtree imports
and no NCU dependency.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


SHARING_CLASS_INTRA_WARP: str = "intra_warp"
SHARING_CLASS_INTRA_CTA: str = "intra_cta"
SHARING_CLASS_INTER_CTA: str = "inter_cta"
SHARING_CLASS_COLD: str = "cold"


SHARING_CLASS_NAMES: tuple[str, ...] = (
    SHARING_CLASS_INTRA_WARP,
    SHARING_CLASS_INTRA_CTA,
    SHARING_CLASS_INTER_CTA,
    SHARING_CLASS_COLD,
)


SHARING_CLASSES: frozenset[str] = frozenset(SHARING_CLASS_NAMES)


SHARING_CLASSIFIER_TYPED_REASONS: frozenset[str] = frozenset({
    "sharing_classifier_input_invalid",
    "sharing_classifier_capacity_invalid",
    "sharing_classifier_bin_edges_invalid",
})


# Powers-of-two RD bin upper bounds (inclusive).
# A bin at index `i` contains accesses whose RD
# satisfies ``BIN_EDGES[i-1] < rd <= BIN_EDGES[i]``
# (with ``BIN_EDGES[-1]`` treated as ``-1`` so the
# first bin includes ``rd == 0``). The final bin
# is ``+infinity`` semantically (anything beyond
# the last finite edge falls into the catch-all
# bin appended internally). Mirrors the sibling
# project's HRD profiler ladder.
RD_BIN_EDGES_DEFAULT: tuple[int, ...] = (
    0, 1, 3, 7, 15, 31, 63, 127, 255, 511,
    1023, 2047, 4095, 8191, 16383, 32767,
    65535, 131071, 262143, 524287, 1048575,
)


class SharingClassifierError(Exception):
    """Raised by the sharing classifier on input
    or capacity contract violations. Carries the
    typed-reason string so callers can compose
    typed-unavailable bundles or report-table
    errors without parsing message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class SharingAccessRecord:
    identity: int
    warp_id: int
    cta_id: int


@dataclass(frozen=True)
class ClassifiedAccess:
    identity: int
    warp_id: int
    cta_id: int
    sharing_class: str


@dataclass(frozen=True)
class RDHistogram:
    """Per-bucket reuse-distance histogram.

    ``bin_counts[i]`` is the number of finite-RD
    accesses whose RD falls into bin ``i``.
    ``total`` is the sum across all finite-RD
    bins; ``cold`` is the count of cold accesses
    (no prior identity sight). ``bin_edges``
    pins the upper bound of each finite bin; an
    access with RD strictly greater than the
    last edge falls into the final catch-all
    bin which is appended internally.

    ``finite_rds`` carries the exact per-bucket
    finite reuse distances in the original
    classified-stream order. The histogram bin
    counts are intentionally lossy (multiple
    distinct RDs share a bin); the
    ``finite_rds`` tuple is the source of truth
    for analytical hit-rate computation, so the
    helper can compare each RD exactly against
    the caller-supplied cache capacity without
    inflating or deflating counts via bin-edge
    interpolation.
    """

    bin_edges: tuple[int, ...]
    bin_counts: tuple[int, ...]
    total: int
    cold: int
    finite_rds: tuple[int, ...] = ()


class _FenwickReuseDistanceTracker:
    """Fenwick-tree-backed LRU stack-distance
    tracker, O(log N) per access.

    Returns ``(rd, state)`` per access where
    ``state`` is ``"cold"`` for the first sight of
    an identity and ``"finite"`` for re-touches.
    The finite RD counts OTHER identities whose
    latest pre-touch timestamp is strictly
    greater than this identity's previous
    timestamp (0-indexed-from-top: an immediate
    re-touch with no other identity in between
    returns ``rd == 0``).

    Per the sibling project's R53 design, the
    BIT is backed by ``array("q")`` (signed
    int64) for compact storage at large N. The
    deployed-scope copy follows the same
    semantics so the new sharing classifier and
    multi-granularity HRD walker can share a
    well-tested algorithm without crossing the
    sibling boundary at runtime.
    """

    __slots__ = ("_bit_size", "_tree", "_last_seen", "_live", "_max_t")

    def __init__(self, max_accesses: int) -> None:
        if max_accesses <= 0:
            raise SharingClassifierError(
                "sharing_classifier_input_invalid",
                f"_FenwickReuseDistanceTracker max_accesses must "
                f"be positive; got {max_accesses}",
            )
        self._bit_size = max_accesses
        # 1-indexed: index 0 is unused.
        self._tree = array("q", [0] * (max_accesses + 1))
        self._last_seen: dict[object, int] = {}
        self._live = 0
        self._max_t = max_accesses

    def _bit_add(self, i: int, delta: int) -> None:
        tree = self._tree
        n = self._bit_size
        while i <= n:
            tree[i] += delta
            i += i & -i

    def _bit_prefix_sum(self, i: int) -> int:
        if i < 0:
            return 0
        if i > self._bit_size:
            i = self._bit_size
        tree = self._tree
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    def access(self, identity: object, t: int) -> tuple[int | None, str]:
        if t < 1 or t > self._max_t:
            raise SharingClassifierError(
                "sharing_classifier_input_invalid",
                f"timestamp {t} out of bounds [1, {self._max_t}]",
            )
        prev = self._last_seen.get(identity)
        if prev is None:
            self._bit_add(t, 1)
            self._last_seen[identity] = t
            self._live += 1
            return None, "cold"
        rd = self._live - self._bit_prefix_sum(prev)
        self._bit_add(prev, -1)
        self._bit_add(t, 1)
        self._last_seen[identity] = t
        return rd, "finite"


def _validate_int(value: int, name: str, *, allow_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SharingClassifierError(
            "sharing_classifier_input_invalid",
            f"{name} must be int; got {type(value).__name__}",
        )
    if not allow_negative and value < 0:
        raise SharingClassifierError(
            "sharing_classifier_input_invalid",
            f"{name} must be non-negative; got {value}",
        )
    return value


def _validate_record(record: SharingAccessRecord) -> None:
    if not isinstance(record, SharingAccessRecord):
        raise SharingClassifierError(
            "sharing_classifier_input_invalid",
            f"record must be SharingAccessRecord; got "
            f"{type(record).__name__}",
        )
    _validate_int(record.identity, "identity")
    _validate_int(record.warp_id, "warp_id")
    _validate_int(record.cta_id, "cta_id")


def classify_stream(
    records: Iterable[SharingAccessRecord],
) -> tuple[ClassifiedAccess, ...]:
    """Classify each access into one of the four
    Jiang 2022 sharing buckets.

    The classification consults a per-identity
    history of the last-seen ``(warp_id, cta_id)``
    pair. Cold accesses do not update the history
    of any other identity; they only register the
    identity as seen for future re-touch
    classification.
    """
    last_seen: dict[int, tuple[int, int]] = {}
    classified: list[ClassifiedAccess] = []
    for record in records:
        _validate_record(record)
        prior = last_seen.get(record.identity)
        if prior is None:
            sharing_class = SHARING_CLASS_COLD
        else:
            prev_warp, prev_cta = prior
            if record.cta_id != prev_cta:
                sharing_class = SHARING_CLASS_INTER_CTA
            elif record.warp_id != prev_warp:
                sharing_class = SHARING_CLASS_INTRA_CTA
            else:
                sharing_class = SHARING_CLASS_INTRA_WARP
        last_seen[record.identity] = (record.warp_id, record.cta_id)
        classified.append(
            ClassifiedAccess(
                identity=record.identity,
                warp_id=record.warp_id,
                cta_id=record.cta_id,
                sharing_class=sharing_class,
            )
        )
    return tuple(classified)


def _validate_bin_edges(bin_edges: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(bin_edges, (list, tuple)):
        raise SharingClassifierError(
            "sharing_classifier_bin_edges_invalid",
            f"bin_edges must be a sequence; got "
            f"{type(bin_edges).__name__}",
        )
    if len(bin_edges) == 0:
        raise SharingClassifierError(
            "sharing_classifier_bin_edges_invalid",
            "bin_edges must be non-empty",
        )
    edges = tuple(bin_edges)
    for index, edge in enumerate(edges):
        if isinstance(edge, bool) or not isinstance(edge, int):
            raise SharingClassifierError(
                "sharing_classifier_bin_edges_invalid",
                f"bin_edges[{index}] must be int; got "
                f"{type(edge).__name__}",
            )
        if edge < 0:
            raise SharingClassifierError(
                "sharing_classifier_bin_edges_invalid",
                f"bin_edges[{index}] must be non-negative; "
                f"got {edge}",
            )
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            raise SharingClassifierError(
                "sharing_classifier_bin_edges_invalid",
                f"bin_edges must be strictly increasing; got "
                f"edges[{index - 1}]={edges[index - 1]} and "
                f"edges[{index}]={edges[index]}",
            )
    return edges


def _assign_bin(rd: int, bin_edges: tuple[int, ...]) -> int:
    """Return the bin index in ``[0, len(bin_edges)]``.

    Each finite bin ``i`` contains accesses whose
    RD satisfies ``rd <= bin_edges[i]`` and (for
    ``i > 0``) ``rd > bin_edges[i - 1]``. The
    catch-all bin at index ``len(bin_edges)``
    contains accesses whose RD exceeds the last
    edge.
    """
    for index, edge in enumerate(bin_edges):
        if rd <= edge:
            return index
    return len(bin_edges)


def compute_per_bucket_rd_histograms(
    classified: Sequence[ClassifiedAccess],
    *,
    bin_edges: Sequence[int] = RD_BIN_EDGES_DEFAULT,
) -> Mapping[str, RDHistogram]:
    """Bin per-bucket per-access RDs into the
    caller-supplied bin edges using a single
    full-stream Fenwick tracker.

    The classified stream is walked in original
    order with one
    ``_FenwickReuseDistanceTracker`` sized to
    ``len(classified)``. For each access the
    tracker computes ``(rd, state)`` once against
    the FULL identity history; the outcome is
    then appended to the histogram of
    ``access.sharing_class``. This guarantees
    that the first non-cold reuse of an identity
    contributes a finite RD to its sharing
    bucket — the bucket does NOT reset its own
    cold state per a fresh per-bucket tracker.

    ``classify_stream()`` and the full-stream
    Fenwick tracker must agree on cold vs.
    finite for every access: a first sight of an
    identity is cold in both, and any re-touch
    is finite in both. If a caller-constructed
    ``ClassifiedAccess`` sequence violates this
    invariant (e.g. a non-first sight carrying
    ``sharing_class == "cold"`` or a first
    sight carrying a non-cold class), the helper
    fails closed with
    ``sharing_classifier_input_invalid`` naming
    the offending timestamp.

    Returns a mapping from each name in
    ``SHARING_CLASS_NAMES`` to its
    ``RDHistogram``. Empty buckets (no accesses
    of that class in the stream) produce a
    histogram with zero counts and empty
    ``finite_rds``. The histograms'
    ``bin_edges`` field carries the validated
    edge tuple verbatim so a downstream consumer
    can reproduce binning; the ``finite_rds``
    tuple carries exact per-access finite RDs
    in classified-stream order so analytical
    hit counting does not depend on bin-edge
    granularity.
    """
    edges = _validate_bin_edges(bin_edges)
    classified_seq = tuple(classified)
    if not classified_seq:
        empty = RDHistogram(
            bin_edges=edges,
            bin_counts=tuple([0] * (len(edges) + 1)),
            total=0,
            cold=0,
            finite_rds=(),
        )
        return {name: empty for name in SHARING_CLASS_NAMES}

    for access in classified_seq:
        if access.sharing_class not in SHARING_CLASSES:
            raise SharingClassifierError(
                "sharing_classifier_input_invalid",
                f"unknown sharing_class "
                f"{access.sharing_class!r}; valid names are "
                f"{sorted(SHARING_CLASSES)}",
            )

    tracker = _FenwickReuseDistanceTracker(len(classified_seq))
    per_bucket_bin_counts: dict[str, list[int]] = {
        name: [0] * (len(edges) + 1) for name in SHARING_CLASS_NAMES
    }
    per_bucket_finite_rds: dict[str, list[int]] = {
        name: [] for name in SHARING_CLASS_NAMES
    }
    per_bucket_finite_count: dict[str, int] = {
        name: 0 for name in SHARING_CLASS_NAMES
    }
    per_bucket_cold_count: dict[str, int] = {
        name: 0 for name in SHARING_CLASS_NAMES
    }

    for timestamp, access in enumerate(classified_seq, start=1):
        rd, state = tracker.access(access.identity, timestamp)
        classified_is_cold = access.sharing_class == SHARING_CLASS_COLD
        full_stream_is_cold = state == "cold"
        if classified_is_cold != full_stream_is_cold:
            raise SharingClassifierError(
                "sharing_classifier_input_invalid",
                f"classified sequence inconsistent at "
                f"timestamp {timestamp} (identity "
                f"{access.identity!r}): classified "
                f"sharing_class={access.sharing_class!r} "
                f"implies "
                f"classified_is_cold={classified_is_cold} "
                f"but the full-stream Fenwick tracker "
                f"reports state={state!r}",
            )
        if full_stream_is_cold:
            per_bucket_cold_count[access.sharing_class] += 1
        else:
            assert rd is not None
            bin_index = _assign_bin(rd, edges)
            per_bucket_bin_counts[access.sharing_class][bin_index] += 1
            per_bucket_finite_rds[access.sharing_class].append(rd)
            per_bucket_finite_count[access.sharing_class] += 1

    return {
        name: RDHistogram(
            bin_edges=edges,
            bin_counts=tuple(per_bucket_bin_counts[name]),
            total=per_bucket_finite_count[name],
            cold=per_bucket_cold_count[name],
            finite_rds=tuple(per_bucket_finite_rds[name]),
        )
        for name in SHARING_CLASS_NAMES
    }


def _validate_capacity(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SharingClassifierError(
            "sharing_classifier_capacity_invalid",
            f"cache_capacity_in_sectors must be int; got "
            f"{type(value).__name__}",
        )
    if value < 0:
        raise SharingClassifierError(
            "sharing_classifier_capacity_invalid",
            f"cache_capacity_in_sectors must be non-negative; "
            f"got {value}",
        )
    return value


def _bucket_hit_count(
    histogram: RDHistogram, cache_capacity_in_sectors: int,
) -> int:
    """Count finite-RD accesses in ``histogram``
    whose reuse distance is strictly less than
    ``cache_capacity_in_sectors``.

    The cache-capacity convention is strict
    less-than (matching the established sibling
    convention recorded in the cross-study HRD
    profiler): a cache of capacity ``C_S``
    sectors admits an access whose stack
    distance is ``< C_S`` — equivalently, the
    cache must hold at least ``rd + 1``
    distinct sectors to retain the target's
    most recent occurrence. Cold accesses are
    always misses (their stack distance is
    ``+infinity`` by convention); finite
    accesses with ``rd >= capacity`` are
    misses; all other finite accesses are hits.
    Zero capacity therefore admits zero hits.

    The count is computed exactly against
    ``histogram.finite_rds`` — no bin-edge
    interpolation — so capacities falling
    inside a coarse bin interval still produce
    the correct count.
    """
    return sum(
        1 for rd in histogram.finite_rds
        if rd < cache_capacity_in_sectors
    )


def analytical_hit_rate(
    per_bucket_histograms: Mapping[str, RDHistogram],
    *,
    cache_capacity_in_sectors: int,
) -> float:
    """Compute the weighted analytical hit rate
    ``m = sum_bucket weight_bucket * P[RD_bucket < C_S]``.

    The per-bucket weight is the bucket's share
    of total accesses across all four buckets
    (finite + cold). Cold accesses always count
    as misses; a finite access in any bucket
    hits when its reuse distance is strictly
    less than the cache capacity (sibling
    convention). The function returns a float
    in ``[0.0, 1.0]``; an empty stream returns
    ``0.0``; zero capacity returns ``0.0`` even
    when the stream contains finite reuses.
    """
    capacity = _validate_capacity(cache_capacity_in_sectors)
    total_accesses = 0
    for name in SHARING_CLASS_NAMES:
        if name not in per_bucket_histograms:
            raise SharingClassifierError(
                "sharing_classifier_input_invalid",
                f"per_bucket_histograms missing required "
                f"bucket {name!r}; got "
                f"{sorted(per_bucket_histograms.keys())}",
            )
        histogram = per_bucket_histograms[name]
        if not isinstance(histogram, RDHistogram):
            raise SharingClassifierError(
                "sharing_classifier_input_invalid",
                f"per_bucket_histograms[{name!r}] must be "
                f"RDHistogram; got {type(histogram).__name__}",
            )
        total_accesses += histogram.total + histogram.cold
    if total_accesses == 0:
        return 0.0
    total_hits = 0
    for name in SHARING_CLASS_NAMES:
        histogram = per_bucket_histograms[name]
        total_hits += _bucket_hit_count(histogram, capacity)
    return total_hits / total_accesses


__all__ = [
    "ClassifiedAccess",
    "RDHistogram",
    "RD_BIN_EDGES_DEFAULT",
    "SHARING_CLASSES",
    "SHARING_CLASSIFIER_TYPED_REASONS",
    "SHARING_CLASS_COLD",
    "SHARING_CLASS_INTER_CTA",
    "SHARING_CLASS_INTRA_CTA",
    "SHARING_CLASS_INTRA_WARP",
    "SHARING_CLASS_NAMES",
    "SharingAccessRecord",
    "SharingClassifierError",
    "analytical_hit_rate",
    "classify_stream",
    "compute_per_bucket_rd_histograms",
]
