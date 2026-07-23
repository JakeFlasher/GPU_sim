"""Deterministic admissibility-gate classifier.

Routes each still-breaching cell to EXACTLY ONE typed method before any sampler
runs. A cell reaches block-subset sampling only if every forbidding gate passes;
otherwise it is routed to a matched reduced-shape or typed `unavailable`. Every
threshold is a named constant here (provenance-recorded); the gate is a pure
function of the counting-pass `CellStats` — no randomness, no counter/label input.

The `wave_gate` floor is stated in MEASURED WAVES (not raw waves): the estimator
needs >= MIN_SEGMENTS segments and a segment spans SEGMENT_LEN waves of which
GHOST_DEPTH are warm-up, so a cell must expose >= MIN_SEGMENTS*(SEGMENT_LEN-
GHOST_DEPTH) MEASURED waves. The `homogeneity` gate turns per-SM block-count skew
and per-wave record-total phases into strata, and forbids subset when the grid is
too uneven (underfill) or the phases cannot fill the per-stratum + segment floors.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from counting_pass import CellStats

# Segment geometry (a segment is the primary sampling unit).
SEGMENT_LEN = 3          # pseudo-waves per segment
GHOST_DEPTH = 1          # leading warm-up waves (metrics muted)
MEASURED_PER_SEGMENT = SEGMENT_LEN - GHOST_DEPTH
MIN_SEGMENTS = 8         # estimator floor
MIN_PER_STRATUM = 2      # n_h >= 2
# wave_gate: MEASURED waves required to form MIN_SEGMENTS segments.
MIN_MEASURED_WAVES = MIN_SEGMENTS * MEASURED_PER_SEGMENT  # 16 at defaults

# atomic_gate: below BOTH is fine; above BOTH forbids subset for L2/cycles/atomic.
ATOMIC_DENSITY_FORBID = 0.005          # atomic records / total records
ATOMIC_TOP256_SHARE_FORBID = 0.10      # top-256 sectors / atomic traffic

# skew_gate: a head this heavy becomes certainty includes.
SKEW_TOP1PCT_FORBID = 0.15
SKEW_GINI_FORBID = 0.5

# homogeneity_gate: uneven SM occupancy or fragmented phases.
HOMOGENEITY_CV_FORBID = 0.5            # CV of per-SM block counts
PHASE_DELTA = 0.5                      # successive per-wave record-total rel. jump

# Wire method names (must match the sidecar schema enum).
M_SUBSET = "wave_segment_stratified_subset"
M_REDUCED = "reduced_shape_matched_label"
M_UNAVAILABLE = "unavailable"


@dataclass
class GateResult:
    coop: bool
    wave_count: int
    n_measured_waves: int
    atomic_density: float
    hotline_share: float
    skew_top1pct_share: float
    gini: float
    per_sm_block_count_cv: float
    n_phases: int
    tail_measured_waves: int
    skew_heavy_block_count: int
    certainty_bytes: int | None
    max_trace_bytes: int | None
    certainty_budget_exceeded: bool
    verdict: str          # matches schema gates.verdict enum
    method: str           # the ONE routed method
    forbidding_gates: list[str]
    force_include_blocks: tuple[int, ...] = ()

    def to_sidecar_gates(self) -> dict:
        return {
            "coop": self.coop,
            "wave_count": self.wave_count,
            "n_measured_waves": self.n_measured_waves,
            "atomic_density": self.atomic_density,
            "hotline_share": self.hotline_share,
            "skew_top1pct_share": self.skew_top1pct_share,
            "gini": self.gini,
            "per_sm_block_count_cv": self.per_sm_block_count_cv,
            "n_phases": self.n_phases,
            "tail_measured_waves": self.tail_measured_waves,
            "skew_heavy_block_count": self.skew_heavy_block_count,
            "certainty_bytes": self.certainty_bytes,
            "max_trace_bytes": self.max_trace_bytes,
            "certainty_budget_exceeded": self.certainty_budget_exceeded,
            "verdict": self.verdict,
        }


def _cv(values: list[int]) -> float:
    """Coefficient of variation (std/mean) of a non-negative distribution."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    mean = statistics.fmean(vals)
    if mean == 0:
        return 0.0
    return statistics.pstdev(vals) / mean


def _count_phases(per_wave_totals: dict[int, int]) -> int:
    """Number of phases = 1 + count of successive per-wave record-total jumps
    exceeding PHASE_DELTA (relative). Waves are ordered by index."""
    if not per_wave_totals:
        return 1
    ordered = [per_wave_totals[w] for w in sorted(per_wave_totals)]
    changes = 0
    for prev, cur in zip(ordered, ordered[1:]):
        base = max(prev, 1)
        if abs(cur - prev) / base > PHASE_DELTA:
            changes += 1
    return changes + 1


def _heavy_block_ids(stats: CellStats) -> tuple[int, ...]:
    """Return the deterministic top-1% head used when either skew trigger fires.

    The count is ``ceil(.01*N)`` and record-count ties break by flattened block
    id.  Merely crossing a skew threshold does not forbid subset sampling: these
    blocks become offline force-includes, expanded to whole segments by the
    selector.  The exact raw-PB certainty budget is evaluated on pass 2.
    """
    if not stats.blocks:
        return ()
    k = max(1, math.ceil(0.01 * len(stats.blocks)))
    ordered = sorted(stats.blocks, key=lambda block: (-block.records, block.block_id))
    return tuple(sorted(block.block_id for block in ordered[:k]))


def _ordinary_phase_layout(
        stats: CellStats,
        force_include_blocks: tuple[int, ...],
        ) -> tuple[int, dict[tuple[int, int], int], int, int, int, bool]:
    """Mirror the selector's tail-free phase/tercile accounting cheaply.

    Returns ``(n_phases, ordinary_stratum_sizes, certainty_count,
    n_measured_waves, tail_measured_waves, residual_tail)``. Detected phase
    changes are computed only over non-tail complete waves, so a folded residual
    tail cannot invent a phase. A segment crossing a real boundary is certainty,
    as are cold start and heavy-block enclosing segments.
    """
    if not stats.per_sm_block_ids or not stats.per_sm_resident:
        segment_count = math.ceil(stats.n_waves / SEGMENT_LEN)
        residual_tail = stats.n_waves % SEGMENT_LEN != 0
        tail_wave_count = (stats.n_waves % SEGMENT_LEN
                           if residual_tail else min(SEGMENT_LEN, stats.n_waves))
        return (_count_phases(stats.per_wave_record_totals), {}, 0,
                max(0, stats.n_waves - GHOST_DEPTH * segment_count),
                max(0, tail_wave_count - GHOST_DEPTH), residual_tail)
    complete_waves = min(
        len(stats.per_sm_block_ids[sm]) // stats.per_sm_resident[sm]
        for sm in stats.per_sm_block_ids
    )
    complete_cutoff = (complete_waves // SEGMENT_LEN) * SEGMENT_LEN
    residual_tail = any(block.wave >= complete_cutoff for block in stats.blocks)
    tail_wave_count = (stats.n_waves - complete_cutoff
                       if residual_tail else min(SEGMENT_LEN, complete_cutoff))
    tail_measured_waves = max(0, tail_wave_count - GHOST_DEPTH)
    segment_count = (complete_cutoff // SEGMENT_LEN
                     + (1 if residual_tail else 0))
    n_measured_waves = max(
        0, stats.n_waves - GHOST_DEPTH * segment_count)
    tail_start = complete_cutoff if residual_tail else max(0, complete_cutoff - SEGMENT_LEN)
    phase_totals = {
        wave: stats.per_wave_record_totals[wave]
        for wave in range(tail_start)
        if wave in stats.per_wave_record_totals
    }
    n_phases = _count_phases(phase_totals)
    if tail_start <= 0 or len(phase_totals) != tail_start:
        return (n_phases, {}, 1 if stats.blocks else 0,
                n_measured_waves, tail_measured_waves, residual_tail)

    phase_by_wave: dict[int, int] = {}
    phase = 0
    previous: int | None = None
    for wave in range(tail_start):
        current = phase_totals[wave]
        if previous is not None and abs(current - previous) / max(previous, 1) > PHASE_DELTA:
            phase += 1
        phase_by_wave[wave] = phase
        previous = current

    forced_blocks = set(force_include_blocks)
    forced_segment_ids = {
        block.wave // SEGMENT_LEN
        for block in stats.blocks
        if block.block_id in forced_blocks and block.wave < tail_start
    }
    n_chronological = tail_start // SEGMENT_LEN
    strata: dict[tuple[int, int], int] = {}
    certainty_ids = {0, n_chronological}  # cold + the separately folded tail
    for segment_id in range(n_chronological):
        waves = range(segment_id * SEGMENT_LEN, (segment_id + 1) * SEGMENT_LEN)
        phases = {phase_by_wave[wave] for wave in waves}
        if len(phases) != 1 or segment_id in forced_segment_ids:
            certainty_ids.add(segment_id)
            continue
        if segment_id == 0:
            continue
        tercile = min(2, segment_id * 3 // n_chronological)
        key = (next(iter(phases)), tercile)
        strata[key] = strata.get(key, 0) + 1
    return (n_phases, strata, len(certainty_ids), n_measured_waves,
            tail_measured_waves, residual_tail)


def classify(stats: CellStats, *, sample_size: int = MIN_SEGMENTS,
             force_include_block_ids: tuple[int, ...] = (),
             certainty_bytes: int | None = None,
             max_trace_bytes: int | None = None) -> GateResult:
    """Route a cell to exactly one method. The first forbidding gate (in the
    fixed order coop -> wave -> atomic -> skew -> homogeneity) sets the verdict;
    only a cell that clears all of them is subset-admissible."""
    cv = _cv(list(stats.per_sm_block_counts.values()))
    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if certainty_bytes is not None and (type(certainty_bytes) is not int
                                        or certainty_bytes < 0):
        raise ValueError("certainty_bytes must be a non-negative integer")
    if max_trace_bytes is not None and (type(max_trace_bytes) is not int
                                        or max_trace_bytes <= 0):
        raise ValueError("max_trace_bytes must be a positive integer")
    if certainty_bytes is not None and max_trace_bytes is None:
        raise ValueError("certainty_bytes requires max_trace_bytes")
    if not isinstance(force_include_block_ids, tuple) or any(
            type(block_id) is not int or block_id < 0
            for block_id in force_include_block_ids):
        raise ValueError(
            "force_include_block_ids must be a tuple of non-negative integers")

    coop = bool(stats.cooperative)
    atomic_fires = (stats.atomic_density > ATOMIC_DENSITY_FORBID
                    and stats.atomic_top256_sector_share > ATOMIC_TOP256_SHARE_FORBID)
    skew_fires = (stats.skew_top1pct_share >= SKEW_TOP1PCT_FORBID
                  or stats.gini > SKEW_GINI_FORBID)
    detected_heavy_blocks = _heavy_block_ids(stats) if skew_fires else ()
    heavy_blocks = tuple(sorted(set(force_include_block_ids).union(
        detected_heavy_blocks)))
    (n_phases, crossed_strata, certainty_count, n_measured,
     tail_measured_waves, residual_tail) = _ordinary_phase_layout(
        stats, heavy_blocks)
    wave_ok = n_measured >= MIN_MEASURED_WAVES
    tail_measurement_fires = residual_tail and tail_measured_waves == 0
    certainty_budget_exceeded = bool(
        certainty_bytes is not None
        and 5 * certainty_bytes > 2 * max_trace_bytes)

    # homogeneity: uneven SM occupancy (underfill) hard-forbids; detected phases
    # are admissible only if they can fill MIN_SEGMENTS with n_h >= 2 per phase.
    total_segments = certainty_count + sum(crossed_strata.values())
    phase_feasible = (
        total_segments >= MIN_SEGMENTS
        and bool(crossed_strata)
        and all(size >= MIN_PER_STRATUM for size in crossed_strata.values())
        and certainty_count + MIN_PER_STRATUM * len(crossed_strata) <= sample_size
    )
    homogeneity_fires = cv > HOMOGENEITY_CV_FORBID or (n_phases > 1 and not phase_feasible)
    # A single detected phase still crosses launch terciles.  Refuse if those
    # binding strata cannot meet the n_h floor at the declared sample size.
    homogeneity_fires = homogeneity_fires or (wave_ok and not phase_feasible)

    forbidding: list[str] = []
    if coop:
        forbidding.append("coop_gate")
    if not wave_ok:
        forbidding.append("wave_gate")
    if atomic_fires:
        forbidding.append("atomic_gate")
    if tail_measurement_fires:
        forbidding.append("tail_measurement_gate")
    if certainty_budget_exceeded:
        forbidding.append("skew_gate")
    if homogeneity_fires:
        forbidding.append("homogeneity_gate")

    if coop:
        verdict, method = "coop_forbid", M_REDUCED
    elif not wave_ok:
        verdict, method = "wave_forbid", M_REDUCED
    elif atomic_fires:
        verdict, method = "atomic_forbid", M_REDUCED
    elif tail_measurement_fires:
        verdict, method = "tail_measurement_forbid", M_REDUCED
    elif certainty_budget_exceeded:
        verdict = ("skew_to_reduced_shape" if heavy_blocks
                   else "certainty_budget_to_reduced_shape")
        method = M_REDUCED
    elif homogeneity_fires:
        verdict, method = "homogeneity_to_reduced_shape", M_REDUCED
    else:
        verdict, method = "subset_admissible", M_SUBSET

    return GateResult(
        coop=coop, wave_count=stats.n_waves, n_measured_waves=n_measured,
        atomic_density=stats.atomic_density,
        hotline_share=stats.atomic_top256_sector_share,
        skew_top1pct_share=stats.skew_top1pct_share, gini=stats.gini,
        per_sm_block_count_cv=cv, n_phases=n_phases,
        tail_measured_waves=tail_measured_waves,
        skew_heavy_block_count=len(heavy_blocks),
        certainty_bytes=certainty_bytes, max_trace_bytes=max_trace_bytes,
        certainty_budget_exceeded=certainty_budget_exceeded,
        verdict=verdict, method=method, forbidding_gates=forbidding,
        force_include_blocks=heavy_blocks)
