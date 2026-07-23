"""Exhaustive phase-by-tercile populations and deterministic Neyman draws."""
from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

from admissibility_gate import MIN_SEGMENTS, SEGMENT_LEN
from counting_pass import BlockStat, CellStats
from selection import (
    PopulationSegment,
    PopulationStratum,
    SelectionPlanError,
    _allocate_capped_neyman,
    build_segment_population,
    build_selection_plan,
    decode_stratum_id,
    encode_stratum_id,
    plan_from_stats,
)


BlockValue = Callable[[int, int, int, int], int]


def _stats(
        per_sm_counts: dict[int, int], resident: int = 2, *,
        records_for_block: BlockValue | None = None,
        sectors_for_block: BlockValue | None = None,
        serialized_for_block: BlockValue | None = None) -> CellStats:
    records_for_block = records_for_block or (
        lambda wave, sm, seq, block_id: 100)
    sectors_for_block = sectors_for_block or (
        lambda wave, sm, seq, block_id: 10)
    serialized_for_block = serialized_for_block or (
        lambda wave, sm, seq, block_id: 7)

    blocks = []
    per_sm_ids: dict[int, list[int]] = {}
    per_wave_totals: dict[int, int] = {}
    total_records = 0
    next_id = 0
    n_waves = 0
    fields = BlockStat.__dataclass_fields__
    for sm, count in sorted(per_sm_counts.items()):
        ids = []
        for seq in range(count):
            wave = seq // resident
            records = records_for_block(wave, sm, seq, next_id)
            sectors = sectors_for_block(wave, sm, seq, next_id)
            serialized = serialized_for_block(wave, sm, seq, next_id)
            kwargs = {
                "block_id": next_id,
                "sm_id": sm,
                "records": records,
            }
            if "sector_accesses" in fields:
                kwargs["sector_accesses"] = sectors
            if "serialized_bytes" in fields:
                kwargs["serialized_bytes"] = serialized
            block = BlockStat(**kwargs)
            if "sector_accesses" not in fields:
                block.sector_accesses = sectors
            if "serialized_bytes" not in fields:
                block.serialized_bytes = serialized
            block.wave = wave
            blocks.append(block)
            ids.append(next_id)
            total_records += records
            per_wave_totals[wave] = per_wave_totals.get(wave, 0) + records
            n_waves = max(n_waves, wave + 1)
            next_id += 1
        per_sm_ids[sm] = ids
    return CellStats(
        cell_id="c", kernel_name="k", grid=(next_id, 1, 1),
        block=(32, 1, 1), n_blocks=next_id, total_records=total_records,
        op_class_totals={}, blocks=blocks,
        per_sm_resident={sm: resident for sm in per_sm_counts},
        per_sm_block_ids=per_sm_ids, n_waves=n_waves, atomic_records=0,
        atomic_top256_sector_share=0.0,
        per_wave_record_totals=per_wave_totals,
        per_sm_block_counts=dict(per_sm_counts),
    )


def _assert_reason(reason: str, operation: Callable[[], object]) -> None:
    with pytest.raises(SelectionPlanError) as exc_info:
        operation()
    assert exc_info.value.reason == reason


def _segment(population, segment_id: int):
    return next(segment for segment in population.segments
                if segment.id == segment_id)


def test_population_is_exhaustive_and_crosses_phase_with_launch_tercile():
    stats = _stats({0: 120})  # 60 full waves -> 20 exhaustive segments.
    population = build_segment_population(stats)

    assert len(population.segments) == 20
    assert sorted(population.block_ids) == list(range(120))
    assert len(population.block_ids) == len(set(population.block_ids))
    assert [segment.id for segment in population.certainty_segments] == [0, 19]
    assert population.tail_segment.id == 19
    assert population.phase_boundary_waves == ()
    assert all(len(segment.wave_ids) == SEGMENT_LEN
               for segment in population.ordinary_segments)
    assert all(len(segment.block_ids) == 2 * SEGMENT_LEN
               for segment in population.ordinary_segments)
    assert [stratum.id for stratum in population.ordinary_strata] == [0, 1, 2]
    assert [stratum.N_h for stratum in population.ordinary_strata] == [6, 6, 6]
    assert [(stratum.phase_id, stratum.launch_tercile)
            for stratum in population.ordinary_strata] == [(0, 0), (0, 1), (0, 2)]
    assert _segment(population, 1).serialized_bytes == 6 * 7


def test_population_refuses_incomplete_first_launch_accounting():
    stats = _stats({0: 120})
    stats.grid = (121, 1, 1)
    _assert_reason("incomplete_first_launch",
                   lambda: build_segment_population(stats))


def test_underfilled_waves_fold_once_and_cannot_create_a_phase_boundary():
    # Waves 0..44 are ordinary complete cohorts. The record jump at wave 50 is
    # wholly inside the folded underfilled tail and must not affect phase strata.
    stats = _stats(
        {0: 120, 1: 90},
        records_for_block=lambda wave, sm, seq, block_id:
            300 if wave >= 50 else 100,
    )
    population = build_segment_population(stats)
    tail = population.tail_segment

    assert tail.wave_ids == tuple(range(45, 60))
    assert tail.certainty_reasons == ("folded_tail",)
    assert population.phase_boundary_waves == ()
    assert len([segment for segment in population.segments if segment.tail]) == 1
    assert all(wave < 45 for segment in population.ordinary_segments
               for wave in segment.wave_ids)
    assert {segment.phase_id for segment in population.ordinary_segments} == {0}
    assert sorted(population.block_ids) == list(range(210))


def test_vacuous_and_malformed_sector_evidence_refuses_on_consumer_path():
    _assert_reason("empty_block_population", lambda: plan_from_stats(_stats({})))

    missing = _stats({0: 120})
    first = missing.blocks[0]
    missing.blocks[0] = SimpleNamespace(
        block_id=first.block_id,
        sm_id=first.sm_id,
        records=first.records,
        wave=first.wave,
        serialized_bytes=first.serialized_bytes,
    )
    _assert_reason("missing_sector_accesses", lambda: plan_from_stats(missing))

    for malformed in (True, -1, 1.5):
        stats = _stats({0: 120})
        stats.blocks[0].sector_accesses = malformed
        _assert_reason("invalid_sector_accesses",
                       lambda stats=stats: plan_from_stats(stats))


def test_reconstructed_wave_totals_are_validated_on_consumer_path():
    stats = _stats({0: 120})
    stats.per_wave_record_totals[0] += 1
    _assert_reason("wave_record_totals_mismatch",
                   lambda: plan_from_stats(stats))


def test_aligned_phase_boundary_produces_exhaustive_crossed_strata():
    stats = _stats(
        {0: 240},
        records_for_block=lambda wave, sm, seq, block_id:
            300 if wave >= 60 else 100,
    )
    population = build_segment_population(stats)

    assert population.phase_boundary_waves == (60,)
    assert [stratum.id for stratum in population.ordinary_strata] == [0, 1, 4, 5]
    assert [stratum.N_h for stratum in population.ordinary_strata] == [12, 7, 6, 13]
    assert [(stratum.phase_id, stratum.launch_tercile)
            for stratum in population.ordinary_strata] == [
                (0, 0), (0, 1), (1, 1), (1, 2)]
    assert sum(stratum.N_h for stratum in population.ordinary_strata) == 38
    for stratum in population.ordinary_strata:
        assert decode_stratum_id(stratum.id) == (
            stratum.phase_id, stratum.launch_tercile)
        assert encode_stratum_id(*decode_stratum_id(stratum.id)) == stratum.id


def test_boundary_crossing_segment_is_probability_one_not_ordinary():
    stats = _stats(
        {0: 240},
        records_for_block=lambda wave, sm, seq, block_id:
            300 if wave >= 61 else 100,
    )
    population = build_segment_population(stats)
    crossing = _segment(population, 20)

    assert population.phase_boundary_waves == (61,)
    assert crossing.wave_ids == (60, 61, 62)
    assert crossing.certainty
    assert crossing.certainty_reasons == ("phase_boundary",)
    assert crossing.phase_id is None
    assert crossing.launch_tercile == 1
    assert all(crossing not in stratum.segments
               for stratum in population.ordinary_strata)


def test_force_include_expands_to_enclosing_segment_and_validates_ids():
    stats = _stats({0: 240})
    population = build_segment_population(
        stats, force_include_block_ids=[63, 63])
    forced = _segment(population, 10)

    assert population.force_include_block_ids == (63,)
    assert 63 in forced.block_ids
    assert forced.certainty
    assert forced.certainty_reasons == ("force_include",)
    assert all(forced not in stratum.segments
               for stratum in population.ordinary_strata)
    assert [population.certainty_segments[0].id,
            population.certainty_segments[-1].id] == [0, 39]

    _assert_reason(
        "unknown_force_include_block",
        lambda: build_segment_population(stats, force_include_block_ids=[240]),
    )
    _assert_reason(
        "invalid_force_include_block_id",
        lambda: build_segment_population(stats, force_include_block_ids=[True]),
    )


def test_crossed_strata_floor_has_typed_budget_and_population_refusals():
    aligned = build_segment_population(_stats(
        {0: 240},
        records_for_block=lambda wave, sm, seq, block_id:
            300 if wave >= 60 else 100,
    ))
    _assert_reason(
        "ordinary_budget_below_strata_floor",
        lambda: build_selection_plan(aligned, seed=0, ordinary_sample_size=7),
    )

    singleton = build_segment_population(_stats(
        {0: 120},
        records_for_block=lambda wave, sm, seq, block_id:
            300 if wave >= 36 else 100,
    ))
    assert 1 in [stratum.N_h for stratum in singleton.ordinary_strata]
    _assert_reason(
        "stratum_population_below_floor",
        lambda: build_selection_plan(
            singleton, seed=0,
            ordinary_sample_size=len(singleton.ordinary_segments)),
    )


def test_capped_neyman_prefers_high_sector_dispersion_and_spills_at_cap():
    population = build_segment_population(_stats(
        {0: 240},
        sectors_for_block=lambda wave, sm, seq, block_id:
            (1000 if (wave // 3) % 2 else 0) if wave // 3 < 13 else 100,
    ))

    preferred = build_selection_plan(
        population, seed=11, ordinary_sample_size=10)
    assert [stratum.n_h for stratum in preferred.strata] == [6, 2, 2]
    assert preferred.strata[0].population.sector_stddev > 0
    assert [stratum.population.sector_stddev
            for stratum in preferred.strata[1:]] == [0.0, 0.0]
    assert preferred.ordinary_sample_size == 10

    capped = build_selection_plan(
        population, seed=11, ordinary_sample_size=24)
    assert [stratum.n_h for stratum in capped.strata] == [12, 6, 6]
    assert capped.strata[0].n_h == capped.strata[0].N_h
    assert capped.ordinary_sample_size == 24


def test_zero_dispersion_tie_break_and_census_are_deterministic():
    population = build_segment_population(_stats({0: 120}))
    one_remainder = build_selection_plan(
        population, seed=4, ordinary_sample_size=7)
    assert [stratum.n_h for stratum in one_remainder.strata] == [3, 2, 2]

    census = build_selection_plan(
        population, seed=4,
        ordinary_sample_size=len(population.ordinary_segments))
    assert [stratum.n_h for stratum in census.strata] == [6, 6, 6]
    assert all(stratum.inclusion_probability == 1.0
               for stratum in census.strata)
    assert len(census.selected_segments) == len(population.segments)


def test_final_budget_hamilton_quota_wins_over_house_monotonicity():
    # Hamilton's Alabama paradox is the explicit adjudication boundary.  The
    # final budget gets quota-respecting largest-remainder allocation even when
    # one stratum's n_h shrinks relative to the preceding budget.
    strata = []
    for stratum_id, amplitude in enumerate((1, 3, 3)):
        segments = tuple(
            PopulationSegment(
                id=100 * stratum_id + index,
                wave_ids=(index,),
                block_ids=(100 * stratum_id + index,),
                sector_accesses=(0 if index % 2 == 0 else 2 * amplitude),
                phase_id=stratum_id,
                launch_tercile=0,
            )
            for index in range(10)
        )
        strata.append(PopulationStratum(
            id=stratum_id,
            phase_id=stratum_id,
            launch_tercile=0,
            segments=segments,
        ))

    at_nine = _allocate_capped_neyman(tuple(strata), 9)
    at_ten = _allocate_capped_neyman(tuple(strata), 10)
    assert [at_nine[index] for index in range(3)] == [3, 3, 3]
    assert [at_ten[index] for index in range(3)] == [2, 4, 4]
    assert at_ten[0] < at_nine[0]


def test_pcg64_draws_repeat_and_are_conditional_stratum_prefixes():
    population = build_segment_population(_stats(
        {0: 240},
        sectors_for_block=lambda wave, sm, seq, block_id:
            (1000 if (wave // 3) % 2 else 0) if wave // 3 < 13 else 100,
    ))
    budgets = [6, 10, 14, 20, 38]
    plans = [build_selection_plan(
        population, seed=17, ordinary_sample_size=budget)
        for budget in budgets]

    for smaller, larger in zip(plans, plans[1:]):
        for small_stratum, large_stratum in zip(smaller.strata, larger.strata):
            small_ids = [segment.id for segment in small_stratum.sampled_segments]
            large_ids = [segment.id for segment in large_stratum.sampled_segments]
            assert small_stratum.id == large_stratum.id
            if small_stratum.n_h <= large_stratum.n_h:
                assert small_ids == large_ids[:small_stratum.n_h]
            else:
                assert large_ids == small_ids[:large_stratum.n_h]

    repeat = build_selection_plan(
        population, seed=17, ordinary_sample_size=10)
    other = build_selection_plan(
        population, seed=18, ordinary_sample_size=10)
    assert repeat.sidecar_strata() == plans[1].sidecar_strata()
    assert [row["segment_ids"] for row in repeat.sidecar_strata()] != [
        row["segment_ids"] for row in other.sidecar_strata()]


def test_sidecar_strata_exactly_report_allocation_provenance():
    population = build_segment_population(_stats(
        {0: 240},
        records_for_block=lambda wave, sm, seq, block_id:
            300 if wave >= 60 else 100,
    ))
    plan = build_selection_plan(
        population, seed=3, ordinary_sample_size=8)
    rows = plan.sidecar_strata()

    assert len(plan.selected_segments) == len(population.certainty_segments) + 8
    assert [segment.id for segment in plan.selected_segments[:2]] == [0, 39]
    assert sum(row["n_h"] for row in rows) == 8
    assert {row["id"] for row in rows} == {0, 1, 4, 5}
    for selected, row in zip(plan.strata, rows):
        assert set(row) == {
            "id", "phase_id", "launch_tercile", "segment_ids", "N_h",
            "n_h", "pi", "sector_stddev", "neyman_weight",
        }
        assert row["phase_id"] == selected.population.phase_id
        assert row["launch_tercile"] == selected.population.launch_tercile
        assert row["N_h"] == selected.N_h
        assert row["n_h"] == selected.n_h
        assert row["pi"] == pytest.approx(selected.n_h / selected.N_h)
        assert row["sector_stddev"] == selected.population.sector_stddev
        assert row["neyman_weight"] == selected.population.neyman_weight
        assert row["segment_ids"] == [
            segment.id for segment in selected.sampled_segments]


def test_total_budget_requests_are_exact_and_never_silently_clamped():
    population = build_segment_population(_stats({0: 240}))  # 40 segments.
    _assert_reason(
        "total_budget_below_minimum",
        lambda: build_selection_plan(population, seed=0, fraction=0.15),
    )
    _assert_reason(
        "total_budget_below_minimum",
        lambda: build_selection_plan(
            population, seed=0, sample_size=MIN_SEGMENTS - 1),
    )
    _assert_reason(
        "ambiguous_sample_budget",
        lambda: build_selection_plan(
            population, seed=0, sample_size=8, ordinary_sample_size=6),
    )

    plan = build_selection_plan(population, seed=0, fraction=0.225)
    assert plan.requested_segment_count == 9
    assert plan.requested_ordinary_count == 7
    assert len(plan.selected_segments) == 9


def test_plan_from_stats_applies_force_certainty_before_exact_ordinary_draw():
    plan = plan_from_stats(
        _stats({0: 240}), seed=23, ordinary_sample_size=6,
        force_include_block_ids=[63],
    )

    forced = _segment(plan.population, 10)
    assert forced in plan.certainty_segments
    assert forced.certainty_reasons == ("force_include",)
    assert plan.requested_ordinary_count == 6
    assert plan.ordinary_sample_size == 6
    assert len(plan.selected_segments) == 9
    assert plan.inclusion_probability(forced.id) == 1.0
