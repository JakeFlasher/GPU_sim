"""Binding wave-segment population, stratification, and seeded selection.

The counting pass scopes ``CellStats`` to the first measured launch and assigns
each block a pseudo-wave.  This module consumes that exact population.  It does
not walk trace files or reconstruct launch scope a second time.

Ordinary primary sampling units contain three complete global pseudo-waves.
Cold start, the folded residual/underfilled tail, segments crossing a detected
record-total phase boundary, and segments enclosing explicitly force-included
blocks are probability-one units.  Remaining segments are crossed by detected
phase and chronological launch tercile, then allocated by capped Neyman weights
from the counting pass's per-block sector-access pilot.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from admissibility_gate import (
    MIN_PER_STRATUM,
    MIN_SEGMENTS,
    PHASE_DELTA,
    SEGMENT_LEN,
)

if TYPE_CHECKING:
    from counting_pass import CellStats


N_LAUNCH_TERCILES = 3


class SelectionPlanError(ValueError):
    """Typed refusal from population construction or allocation."""

    def __init__(self, reason: str, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def encode_stratum_id(phase_id: int, launch_tercile: int) -> int:
    """Encode the crossed stratum in the schema's integer ``id`` field."""
    if type(phase_id) is not int or phase_id < 0:
        raise SelectionPlanError("invalid_phase_id", repr(phase_id))
    if (type(launch_tercile) is not int
            or not 0 <= launch_tercile < N_LAUNCH_TERCILES):
        raise SelectionPlanError(
            "invalid_launch_tercile", repr(launch_tercile))
    return phase_id * N_LAUNCH_TERCILES + launch_tercile


def decode_stratum_id(stratum_id: int) -> tuple[int, int]:
    """Return ``(phase_id, launch_tercile)`` for a sidecar stratum id."""
    if type(stratum_id) is not int or stratum_id < 0:
        raise SelectionPlanError("invalid_stratum_id", repr(stratum_id))
    return divmod(stratum_id, N_LAUNCH_TERCILES)


@dataclass(frozen=True)
class PopulationSegment:
    """One exhaustive whole-block segment in the first-launch population."""

    id: int
    wave_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    sector_accesses: int = 0
    serialized_bytes: int = 0
    phase_id: int | None = None
    launch_tercile: int | None = None
    certainty: bool = False
    tail: bool = False
    certainty_reasons: tuple[str, ...] = ()

    @property
    def stratum_id(self) -> int:
        if self.phase_id is None or self.launch_tercile is None:
            raise SelectionPlanError(
                "segment_without_ordinary_stratum", f"segment={self.id}")
        return encode_stratum_id(self.phase_id, self.launch_tercile)

    def sidecar_dict(self) -> dict:
        return {"id": self.id, "block_ids": list(self.block_ids)}


@dataclass(frozen=True)
class PopulationStratum:
    """One detected-phase by launch-tercile ordinary population."""

    id: int
    phase_id: int
    launch_tercile: int
    segments: tuple[PopulationSegment, ...]

    @property
    def N_h(self) -> int:
        return len(self.segments)

    @property
    def sector_stddev(self) -> float:
        if not self.segments:
            raise SelectionPlanError(
                "empty_ordinary_stratum", f"stratum={self.id}")
        return float(statistics.pstdev(
            segment.sector_accesses for segment in self.segments))

    @property
    def neyman_weight(self) -> float:
        return self.N_h * self.sector_stddev


@dataclass(frozen=True)
class SegmentPopulation:
    """An exhaustive population with explicit certainty and ordinary strata."""

    segments: tuple[PopulationSegment, ...]
    segment_len: int
    phase_boundary_waves: tuple[int, ...] = ()
    force_include_block_ids: tuple[int, ...] = ()

    @property
    def certainty_segments(self) -> tuple[PopulationSegment, ...]:
        chronological = tuple(
            segment for segment in self.segments
            if segment.certainty and not segment.tail)
        tails = tuple(segment for segment in self.segments if segment.tail)
        return chronological + tuple(segment for segment in tails
                                     if segment not in chronological)

    @property
    def ordinary_segments(self) -> tuple[PopulationSegment, ...]:
        return tuple(segment for segment in self.segments
                     if not segment.certainty)

    @property
    def ordinary_strata(self) -> tuple[PopulationStratum, ...]:
        groups: dict[tuple[int, int], list[PopulationSegment]] = {}
        for segment in self.ordinary_segments:
            if segment.phase_id is None:
                raise SelectionPlanError(
                    "ordinary_segment_missing_phase", f"segment={segment.id}")
            if segment.launch_tercile is None:
                raise SelectionPlanError(
                    "ordinary_segment_missing_tercile", f"segment={segment.id}")
            key = (segment.phase_id, segment.launch_tercile)
            groups.setdefault(key, []).append(segment)
        if not groups:
            raise SelectionPlanError("empty_ordinary_population")

        strata = tuple(
            PopulationStratum(
                id=encode_stratum_id(phase_id, tercile),
                phase_id=phase_id,
                launch_tercile=tercile,
                segments=tuple(sorted(segments, key=lambda segment: segment.id)),
            )
            for (phase_id, tercile), segments in sorted(groups.items())
        )
        accounted = [segment.id for stratum in strata
                     for segment in stratum.segments]
        expected = [segment.id for segment in self.ordinary_segments]
        if (sorted(accounted) != sorted(expected)
                or len(accounted) != len(set(accounted))):
            raise SelectionPlanError("ordinary_stratum_accounting_mismatch")
        return strata

    @property
    def tail_segment(self) -> PopulationSegment:
        tails = [segment for segment in self.segments if segment.tail]
        if len(tails) != 1:
            raise SelectionPlanError(
                "tail_count_mismatch", f"observed={len(tails)}, expected=1")
        return tails[0]

    @property
    def block_ids(self) -> tuple[int, ...]:
        return tuple(block_id for segment in self.segments
                     for block_id in segment.block_ids)


@dataclass(frozen=True)
class SelectedStratum:
    """One ordinary population stratum and its PCG64 prefix draw."""

    population: PopulationStratum
    sampled_segments: tuple[PopulationSegment, ...]

    @property
    def id(self) -> int:
        return self.population.id

    @property
    def N_h(self) -> int:
        return self.population.N_h

    @property
    def n_h(self) -> int:
        return len(self.sampled_segments)

    @property
    def inclusion_probability(self) -> float:
        if self.N_h <= 0:
            raise SelectionPlanError(
                "empty_ordinary_stratum", f"stratum={self.id}")
        return self.n_h / self.N_h

    def sidecar_dict(self) -> dict:
        return {
            "id": self.id,
            "phase_id": self.population.phase_id,
            "launch_tercile": self.population.launch_tercile,
            "segment_ids": [segment.id for segment in self.sampled_segments],
            "N_h": self.N_h,
            "n_h": self.n_h,
            "pi": self.inclusion_probability,
            "sector_stddev": self.population.sector_stddev,
            "neyman_weight": self.population.neyman_weight,
        }


@dataclass(frozen=True)
class SegmentSelectionPlan:
    """Probability-one certainty units plus stratified ordinary draws."""

    population: SegmentPopulation
    seed: int
    requested_segment_count: int
    requested_ordinary_count: int
    strata: tuple[SelectedStratum, ...]

    @property
    def certainty_segments(self) -> tuple[PopulationSegment, ...]:
        return self.population.certainty_segments

    @property
    def sampled_ordinary_segments(self) -> tuple[PopulationSegment, ...]:
        return tuple(sorted(
            (segment for stratum in self.strata
             for segment in stratum.sampled_segments),
            key=lambda segment: segment.id))

    @property
    def selected_segments(self) -> tuple[PopulationSegment, ...]:
        return self.certainty_segments + self.sampled_ordinary_segments

    @property
    def ordinary_population_size(self) -> int:
        return len(self.population.ordinary_segments)

    @property
    def ordinary_sample_size(self) -> int:
        return sum(stratum.n_h for stratum in self.strata)

    @property
    def ordinary_inclusion_probability(self) -> float:
        """Compatibility accessor for designs whose stratum rates are equal."""
        probabilities = {stratum.inclusion_probability for stratum in self.strata}
        if len(probabilities) != 1:
            raise SelectionPlanError("heterogeneous_ordinary_probabilities")
        return next(iter(probabilities))

    def inclusion_probability(self, segment_id: int) -> float:
        if any(segment.id == segment_id for segment in self.certainty_segments):
            return 1.0
        for stratum in self.strata:
            if any(segment.id == segment_id
                   for segment in stratum.population.segments):
                return stratum.inclusion_probability
        raise SelectionPlanError("unknown_segment_id", str(segment_id))

    def sidecar_segments(self) -> list[dict]:
        return [segment.sidecar_dict() for segment in self.selected_segments]

    def sidecar_strata(self) -> list[dict]:
        return [stratum.sidecar_dict() for stratum in self.strata]


def _require_nonnegative_int(value: Any, reason: str, detail: str) -> int:
    if type(value) is not int or value < 0:
        raise SelectionPlanError(reason, f"{detail}, value={value!r}")
    return value


def _validate_stats(stats: "CellStats") -> tuple[dict[int, Any], dict[int, int]]:
    """Validate the exact first-launch items consumed by the selector."""
    if not stats.blocks:
        raise SelectionPlanError("empty_block_population")
    if stats.n_blocks != len(stats.blocks):
        raise SelectionPlanError(
            "block_count_mismatch",
            f"declared={stats.n_blocks}, rows={len(stats.blocks)}")
    expected_grid_blocks = stats.grid[0] * stats.grid[1] * stats.grid[2]
    if stats.n_blocks != expected_grid_blocks:
        raise SelectionPlanError(
            "incomplete_first_launch",
            f"observed={stats.n_blocks}, grid={expected_grid_blocks}")

    block_by_id: dict[int, Any] = {}
    wave_record_totals: dict[int, int] = {}
    observed_records = 0
    for block in stats.blocks:
        block_id = _require_nonnegative_int(
            block.block_id, "invalid_block_id", "block id")
        if block_id in block_by_id:
            raise SelectionPlanError("duplicate_block_id", str(block_id))
        wave = _require_nonnegative_int(
            block.wave, "invalid_wave_id", f"block={block_id}")
        records = _require_nonnegative_int(
            block.records, "invalid_record_count", f"block={block_id}")
        if not hasattr(block, "sector_accesses"):
            raise SelectionPlanError(
                "missing_sector_accesses", f"block={block_id}")
        _require_nonnegative_int(
            block.sector_accesses, "invalid_sector_accesses", f"block={block_id}")
        if hasattr(block, "serialized_bytes"):
            _require_nonnegative_int(
                block.serialized_bytes, "invalid_serialized_bytes",
                f"block={block_id}")
        block_by_id[block_id] = block
        observed_records += records
        wave_record_totals[wave] = wave_record_totals.get(wave, 0) + records

    if stats.total_records != observed_records:
        raise SelectionPlanError(
            "record_count_mismatch",
            f"declared={stats.total_records}, rows={observed_records}")
    if stats.per_wave_record_totals != wave_record_totals:
        raise SelectionPlanError(
            "wave_record_totals_mismatch",
            f"declared={stats.per_wave_record_totals}, rows={wave_record_totals}")

    if set(stats.per_sm_block_ids) != set(stats.per_sm_resident):
        raise SelectionPlanError("per_sm_map_key_mismatch")
    listed: list[int] = []
    for sm, block_ids in stats.per_sm_block_ids.items():
        resident = stats.per_sm_resident[sm]
        if type(resident) is not int or resident <= 0:
            raise SelectionPlanError(
                "invalid_resident_cohort", f"sm={sm}, resident={resident!r}")
        for block_id in block_ids:
            block = block_by_id.get(block_id)
            if block is None:
                raise SelectionPlanError(
                    "per_sm_unknown_block", f"sm={sm}, block={block_id}")
            if block.sm_id != sm:
                raise SelectionPlanError(
                    "per_sm_placement_mismatch",
                    f"block={block_id}, map_sm={sm}, block_sm={block.sm_id}")
        listed.extend(block_ids)
    if len(listed) != len(set(listed)) or set(listed) != set(block_by_id):
        raise SelectionPlanError("per_sm_block_partition_mismatch")
    return block_by_id, wave_record_totals


def _normalize_force_include_ids(
        force_include_block_ids: Iterable[int] | None,
        block_by_id: dict[int, Any]) -> tuple[int, ...]:
    if force_include_block_ids is None:
        return ()
    try:
        raw_ids = list(force_include_block_ids)
    except TypeError as exc:
        raise SelectionPlanError("invalid_force_include_collection") from exc
    normalized: set[int] = set()
    for block_id in raw_ids:
        value = _require_nonnegative_int(
            block_id, "invalid_force_include_block_id", "force include")
        if value not in block_by_id:
            raise SelectionPlanError("unknown_force_include_block", str(value))
        normalized.add(value)
    return tuple(sorted(normalized))


def _detected_wave_phases(
        wave_ids: tuple[int, ...], wave_record_totals: dict[int, int],
        *, phase_delta: float = PHASE_DELTA,
        ) -> tuple[dict[int, int], tuple[int, ...]]:
    if not wave_ids:
        raise SelectionPlanError("empty_phase_wave_population")
    expected = tuple(range(wave_ids[0], wave_ids[-1] + 1))
    if wave_ids != expected or wave_ids[0] != 0:
        raise SelectionPlanError(
            "noncontiguous_complete_wave_population", repr(wave_ids))

    phase_by_wave: dict[int, int] = {}
    boundaries: list[int] = []
    phase_id = 0
    previous: int | None = None
    for wave in wave_ids:
        if wave not in wave_record_totals:
            raise SelectionPlanError("missing_wave_record_total", f"wave={wave}")
        current = wave_record_totals[wave]
        if previous is not None:
            relative_change = abs(current - previous) / max(previous, 1)
            if relative_change > phase_delta:
                phase_id += 1
                boundaries.append(wave)
        phase_by_wave[wave] = phase_id
        previous = current
    return phase_by_wave, tuple(boundaries)


def build_segment_population(
        stats: "CellStats", *, segment_len: int = SEGMENT_LEN,
        force_include_block_ids: Iterable[int] | None = None) -> SegmentPopulation:
    """Freeze exhaustive segments, phase strata, and certainty expansions."""
    if type(segment_len) is not int or segment_len != SEGMENT_LEN:
        raise SelectionPlanError(
            "unsupported_segment_length",
            f"requested={segment_len!r}, required={SEGMENT_LEN}")
    block_by_id, wave_record_totals = _validate_stats(stats)
    force_ids = _normalize_force_include_ids(force_include_block_ids, block_by_id)

    complete_waves = min(
        len(stats.per_sm_block_ids[sm]) // stats.per_sm_resident[sm]
        for sm in stats.per_sm_block_ids
    )
    complete_wave_cutoff = (complete_waves // segment_len) * segment_len

    grouped: dict[int, list[Any]] = {}
    for block in stats.blocks:
        segment_id = (block.wave // segment_len
                      if block.wave < complete_wave_cutoff
                      else complete_wave_cutoff // segment_len)
        grouped.setdefault(segment_id, []).append(block)
    if not grouped:
        raise SelectionPlanError("empty_segment_population")

    tail_id = complete_wave_cutoff // segment_len
    if tail_id not in grouped:
        tail_id = max(grouped)

    base_segments: list[PopulationSegment] = []
    first_id = min(grouped)
    for segment_id in sorted(grouped):
        blocks = grouped[segment_id]
        wave_ids = tuple(sorted({block.wave for block in blocks}))
        base_segments.append(PopulationSegment(
            id=segment_id,
            wave_ids=wave_ids,
            block_ids=tuple(sorted(block.block_id for block in blocks)),
            sector_accesses=sum(block.sector_accesses for block in blocks),
            serialized_bytes=sum(
                getattr(block, "serialized_bytes", 0) for block in blocks),
            certainty=segment_id in (first_id, tail_id),
            tail=segment_id == tail_id,
        ))

    wave_sm_counts: dict[tuple[int, int], int] = {}
    for block in stats.blocks:
        key = (block.sm_id, block.wave)
        wave_sm_counts[key] = wave_sm_counts.get(key, 0) + 1
    for segment in base_segments:
        if segment.tail:
            continue
        if len(segment.wave_ids) != segment_len:
            raise SelectionPlanError(
                "incomplete_ordinary_segment",
                f"segment={segment.id}, waves={segment.wave_ids}")
        for wave in segment.wave_ids:
            for sm, resident in stats.per_sm_resident.items():
                observed = wave_sm_counts.get((sm, wave), 0)
                if observed != resident:
                    raise SelectionPlanError(
                        "incomplete_ordinary_cohort",
                        f"segment={segment.id}, wave={wave}, sm={sm}, "
                        f"observed={observed}, expected={resident}")

    phase_segments = tuple(segment for segment in base_segments if not segment.tail)
    phase_waves = tuple(wave for segment in phase_segments
                        for wave in segment.wave_ids)
    phase_by_wave, phase_boundaries = _detected_wave_phases(
        phase_waves, wave_record_totals)
    forced_segment_ids = {
        segment.id for segment in base_segments
        if set(segment.block_ids).intersection(force_ids)
    }

    classified: list[PopulationSegment] = []
    n_chronological = len(phase_segments)
    rank_by_id = {segment.id: rank
                  for rank, segment in enumerate(phase_segments)}
    for segment in base_segments:
        reasons: list[str] = []
        if segment.id == first_id:
            reasons.append("cold_start")
        if segment.tail:
            reasons.append("folded_tail")
        if segment.id in forced_segment_ids:
            reasons.append("force_include")

        phase_id: int | None = None
        launch_tercile: int | None = None
        if not segment.tail:
            phases = {phase_by_wave[wave] for wave in segment.wave_ids}
            if len(phases) > 1:
                reasons.append("phase_boundary")
            else:
                phase_id = next(iter(phases))
            rank = rank_by_id[segment.id]
            launch_tercile = min(
                N_LAUNCH_TERCILES - 1,
                rank * N_LAUNCH_TERCILES // n_chronological)

        classified.append(replace(
            segment,
            phase_id=phase_id,
            launch_tercile=launch_tercile,
            certainty=bool(reasons),
            certainty_reasons=tuple(reasons),
        ))

    population = SegmentPopulation(
        segments=tuple(classified),
        segment_len=segment_len,
        phase_boundary_waves=phase_boundaries,
        force_include_block_ids=force_ids,
    )
    if len(population.block_ids) != len(set(population.block_ids)):
        raise SelectionPlanError("duplicate_population_block")
    if set(population.block_ids) != set(block_by_id):
        raise SelectionPlanError("population_block_accounting_mismatch")
    population.tail_segment
    # Force construction now so missing metadata cannot hide until a later user.
    population.ordinary_strata
    return population


def _continuous_capped_neyman_targets(
        strata: tuple[PopulationStratum, ...], total: int,
        neyman_weights: dict[int, float]) -> dict[int, float]:
    """Continuous lower-bounded/capped Neyman targets for one house size."""
    targets = {stratum.id: float(MIN_PER_STRATUM) for stratum in strata}
    capacity = {
        stratum.id: stratum.N_h - MIN_PER_STRATUM for stratum in strata}
    remaining = float(total - MIN_PER_STRATUM * len(strata))
    active = {stratum.id for stratum in strata if capacity[stratum.id] > 0}

    while remaining > 1e-12:
        if not active:
            raise SelectionPlanError("neyman_capacity_exhausted")
        active_ids = sorted(active)
        positive_weight = sum(neyman_weights[sid] for sid in active_ids)
        if positive_weight > 0:
            weights = {sid: neyman_weights[sid] for sid in active_ids}
        else:
            # When every remaining stratum has zero dispersion, Neyman is
            # indifferent. Residual capacity is the deterministic census path.
            weights = {sid: float(capacity[sid]) for sid in active_ids}
        weight_total = sum(weights[sid] for sid in active_ids)
        if weight_total <= 0:
            raise SelectionPlanError("zero_neyman_allocation_weight")

        shares = {sid: remaining * weights[sid] / weight_total
                  for sid in active_ids}
        saturated = [sid for sid in active_ids
                     if shares[sid] >= capacity[sid] - 1e-12]
        if not saturated:
            for sid, share in shares.items():
                targets[sid] += share
            remaining = 0.0
            break
        for sid in sorted(saturated):
            targets[sid] += capacity[sid]
            remaining -= capacity[sid]
            capacity[sid] = 0
            active.remove(sid)
    return targets


def _allocate_capped_neyman(
        strata: tuple[PopulationStratum, ...], ordinary_budget: int) -> dict[int, int]:
    """Allocate the final capped Neyman targets by largest remainder.

    The continuous targets already enforce the per-stratum floor and census
    cap.  Hamilton rounding then preserves each stratum's final quota (floor or
    ceiling of its continuous target), with lower stratum id as the recorded
    deterministic tie-break.  Repeated one-seat apportionment can be
    house-monotone but violate that final quota, so it is not used here.
    """
    if not strata:
        raise SelectionPlanError("empty_ordinary_population")
    for stratum in strata:
        if stratum.N_h < MIN_PER_STRATUM:
            raise SelectionPlanError(
                "stratum_population_below_floor",
                f"stratum={stratum.id}, N_h={stratum.N_h}, "
                f"floor={MIN_PER_STRATUM}")
    floor_total = MIN_PER_STRATUM * len(strata)
    population_total = sum(stratum.N_h for stratum in strata)
    if ordinary_budget < floor_total:
        raise SelectionPlanError(
            "ordinary_budget_below_strata_floor",
            f"budget={ordinary_budget}, required={floor_total}")
    if ordinary_budget > population_total:
        raise SelectionPlanError(
            "ordinary_budget_exceeds_population",
            f"budget={ordinary_budget}, population={population_total}")
    if ordinary_budget == population_total:
        return {stratum.id: stratum.N_h for stratum in strata}

    neyman_weights = {
        stratum.id: stratum.neyman_weight for stratum in strata}
    targets = _continuous_capped_neyman_targets(
        strata, ordinary_budget, neyman_weights)
    allocation = {
        stratum.id: min(stratum.N_h, max(
            MIN_PER_STRATUM, math.floor(targets[stratum.id] + 1e-12)))
        for stratum in strata
    }
    remaining = ordinary_budget - sum(allocation.values())
    eligible = sorted(
        (stratum for stratum in strata
         if allocation[stratum.id] < stratum.N_h),
        key=lambda stratum: (
            -(targets[stratum.id] - allocation[stratum.id]),
            stratum.id,
        ),
    )
    if remaining < 0 or remaining > len(eligible):
        raise SelectionPlanError(
            "neyman_rounding_infeasible",
            f"remaining={remaining}, eligible={len(eligible)}")
    for stratum in eligible[:remaining]:
        allocation[stratum.id] += 1
    if sum(allocation.values()) != ordinary_budget:
        raise SelectionPlanError(
            "ordinary_allocation_count_mismatch",
            f"allocated={sum(allocation.values())}, requested={ordinary_budget}")
    return allocation


def _requested_counts(
        population: SegmentPopulation, *, sample_size: int | None,
        fraction: float | None, ordinary_sample_size: int | None,
        min_segments: int) -> tuple[int, int]:
    supplied = sum(value is not None
                   for value in (sample_size, fraction, ordinary_sample_size))
    if supplied > 1:
        raise SelectionPlanError(
            "ambiguous_sample_budget",
            "provide only one of sample_size, fraction, ordinary_sample_size")
    population_size = len(population.segments)
    certainty_count = len(population.certainty_segments)

    if ordinary_sample_size is not None:
        ordinary = _require_nonnegative_int(
            ordinary_sample_size, "invalid_ordinary_sample_size",
            "ordinary budget")
        if ordinary > len(population.ordinary_segments):
            raise SelectionPlanError(
                "ordinary_budget_exceeds_population",
                f"budget={ordinary}, "
                f"population={len(population.ordinary_segments)}")
        requested = certainty_count + ordinary
    elif fraction is not None:
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise SelectionPlanError("invalid_sample_fraction", repr(fraction))
        if not math.isfinite(float(fraction)) or not 0 < float(fraction) <= 1:
            raise SelectionPlanError("invalid_sample_fraction", repr(fraction))
        requested = round(float(fraction) * population_size)
        ordinary = requested - certainty_count
    elif sample_size is not None:
        requested = _require_nonnegative_int(
            sample_size, "invalid_sample_size", "total budget")
        ordinary = requested - certainty_count
    else:
        requested = min_segments
        ordinary = requested - certainty_count

    if requested < min_segments:
        raise SelectionPlanError(
            "total_budget_below_minimum",
            f"requested={requested}, minimum={min_segments}")
    if requested > population_size:
        raise SelectionPlanError(
            "total_budget_exceeds_population",
            f"requested={requested}, population={population_size}")
    if ordinary <= 0:
        raise SelectionPlanError(
            "empty_ordinary_sample", f"ordinary_budget={ordinary}")
    return requested, ordinary


def build_selection_plan(
        population: SegmentPopulation, *, seed: int = 0,
        sample_size: int | None = None, fraction: float | None = None,
        ordinary_sample_size: int | None = None,
        min_segments: int = MIN_SEGMENTS) -> SegmentSelectionPlan:
    """Allocate and draw certainty plus stratified ordinary segments.

    ``sample_size`` and ``fraction`` describe the total selected count, including
    certainty units. ``ordinary_sample_size`` is the exact ordinary budget and
    is useful when certainty expansion is planned separately. No request is
    silently clamped to a floor or population size.
    """
    if type(seed) is not int or seed < 0:
        raise SelectionPlanError("invalid_seed", repr(seed))
    if type(min_segments) is not int or min_segments <= 0:
        raise SelectionPlanError("invalid_minimum_segments", repr(min_segments))
    requested, ordinary_budget = _requested_counts(
        population,
        sample_size=sample_size,
        fraction=fraction,
        ordinary_sample_size=ordinary_sample_size,
        min_segments=min_segments,
    )
    strata = population.ordinary_strata
    allocation = _allocate_capped_neyman(strata, ordinary_budget)

    selected_strata: list[SelectedStratum] = []
    for stratum in strata:
        # A stratum-specific seed produces one stable random permutation. Any
        # two draws are therefore conditional prefixes: when Hamilton assigns a
        # nondecreasing n_h, the smaller draw is a prefix of the larger one.
        # Hamilton itself is intentionally not claimed to be house-monotone.
        seed_sequence = np.random.SeedSequence([seed, stratum.id])
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        permutation = rng.permutation(stratum.N_h)
        n_h = allocation[stratum.id]
        sampled = tuple(stratum.segments[int(index)]
                        for index in permutation[:n_h])
        selected_strata.append(SelectedStratum(stratum, sampled))

    plan = SegmentSelectionPlan(
        population=population,
        seed=seed,
        requested_segment_count=requested,
        requested_ordinary_count=ordinary_budget,
        strata=tuple(selected_strata),
    )
    if plan.ordinary_sample_size != ordinary_budget:
        raise SelectionPlanError(
            "ordinary_draw_count_mismatch",
            f"drawn={plan.ordinary_sample_size}, requested={ordinary_budget}")
    if len(plan.selected_segments) != requested:
        raise SelectionPlanError(
            "total_draw_count_mismatch",
            f"drawn={len(plan.selected_segments)}, requested={requested}")
    return plan


def plan_from_stats(
        stats: "CellStats", *, seed: int = 0, sample_size: int | None = None,
        fraction: float | None = None, ordinary_sample_size: int | None = None,
        min_segments: int = MIN_SEGMENTS, segment_len: int = SEGMENT_LEN,
        force_include_block_ids: Iterable[int] | None = None,
) -> SegmentSelectionPlan:
    """Consumer entry: validate stats, freeze population, allocate, and draw."""
    population = build_segment_population(
        stats,
        segment_len=segment_len,
        force_include_block_ids=force_include_block_ids,
    )
    return build_selection_plan(
        population,
        seed=seed,
        sample_size=sample_size,
        fraction=fraction,
        ordinary_sample_size=ordinary_sample_size,
        min_segments=min_segments,
    )
