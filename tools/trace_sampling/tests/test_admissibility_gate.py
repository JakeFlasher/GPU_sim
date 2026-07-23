"""Admissibility gate: measured-wave gating, homogeneity, and per-gate routing."""
from admissibility_gate import (MIN_MEASURED_WAVES, M_REDUCED, M_SUBSET,
                                 classify)
from counting_pass import BlockStat, CellStats


def _stats(*, n_waves, cooperative=False, atomic_density=0.0, atomic_top256=0.0,
           block_records=None, per_sm_counts=None, per_wave=None):
    if block_records is not None:
        recs = list(block_records)
        assert len(recs) == n_waves
    elif per_wave is not None:
        recs = [per_wave[wave] for wave in range(n_waves)]
    else:
        recs = [100] * n_waves
    blocks = [BlockStat(
        block_id=i, sm_id=0, records=r, sector_accesses=max(0, r // 32),
        serialized_bytes=1, wave=i) for i, r in enumerate(recs)]
    total = sum(recs)
    atomic = int(atomic_density * total)
    psc = per_sm_counts if per_sm_counts is not None else {0: len(recs)}
    pwt = per_wave if per_wave is not None else {
        wave: recs[wave] for wave in range(n_waves)}
    return CellStats(
        cell_id="c", kernel_name="k", grid=(len(recs), 1, 1), block=(32, 1, 1),
        n_blocks=len(recs), total_records=total, op_class_totals={},
        blocks=blocks, per_sm_resident={0: 1}, per_sm_block_ids={0: list(range(len(recs)))},
        n_waves=n_waves, atomic_records=atomic, atomic_top256_sector_share=atomic_top256,
        per_wave_record_totals=pwt, per_sm_block_counts=psc, cooperative=cooperative)


# --- wave_gate: measured, not raw (Codex R15 blocking #1) ---

def test_raw_16_waves_10_measured_is_forbidden():
    # Five complete segments + a one-wave tail consume six ghosts, leaving 10.
    g = classify(_stats(n_waves=16))
    assert g.n_measured_waves == 10
    assert g.verdict == "wave_forbid"
    assert g.method == M_REDUCED


def test_raw_24_waves_16_measured_is_admissible():
    # 24 raw waves -> 24 - 24//3 = 16 measured == MIN -> admit.
    g = classify(_stats(n_waves=24))
    assert g.n_measured_waves == MIN_MEASURED_WAVES == 16
    assert g.verdict == "subset_admissible"
    assert g.method == M_SUBSET
    assert g.forbidding_gates == []


def test_residual_tail_consumes_its_own_ghost_wave():
    # Seven complete segments + one two-wave tail: 23 - 8 ghosts = 15.
    g = classify(_stats(n_waves=23))
    assert g.n_measured_waves == 15
    assert g.verdict == "wave_forbid"


def test_one_wave_residual_tail_has_a_typed_measurement_refusal():
    # Eight complete segments meet the global 16-wave floor, but the separate
    # one-wave tail would be all ghost and has no measurable contribution.
    g = classify(_stats(n_waves=25))
    assert g.n_measured_waves == MIN_MEASURED_WAVES == 16
    assert g.tail_measured_waves == 0
    assert g.verdict == "tail_measurement_forbid"
    assert g.method == M_REDUCED
    assert "tail_measurement_gate" in g.forbidding_gates


def test_two_wave_residual_tail_keeps_one_measured_wave():
    g = classify(_stats(n_waves=26))
    assert g.tail_measured_waves == 1
    assert g.verdict == "subset_admissible"


# --- other structural gates ---

def test_cooperative_forbids_subset():
    g = classify(_stats(n_waves=100, cooperative=True))
    assert g.verdict == "coop_forbid" and g.method == M_REDUCED


def test_atomic_hotline_forbids_subset():
    g = classify(_stats(n_waves=100, atomic_density=0.02, atomic_top256=0.9))
    assert g.verdict == "atomic_forbid" and g.method == M_REDUCED


def test_atomic_low_density_does_not_forbid():
    g = classify(_stats(n_waves=100, atomic_density=0.001, atomic_top256=0.9))
    assert "atomic_gate" not in g.forbidding_gates


def test_heavy_skew_becomes_a_deterministic_force_include_provisionally():
    g = classify(_stats(n_waves=99, block_records=[10000] + [1] * 98))
    assert g.verdict == "subset_admissible" and g.method == M_SUBSET
    assert g.force_include_blocks == (0,)
    assert g.skew_heavy_block_count == 1


def test_heavy_skew_exactly_at_forty_percent_is_admissible():
    stats = _stats(n_waves=99, block_records=[10000] + [1] * 98)
    g = classify(stats, certainty_bytes=4, max_trace_bytes=10)
    assert g.verdict == "subset_admissible" and not g.certainty_budget_exceeded


def test_heavy_skew_one_byte_over_forty_percent_routes_out():
    stats = _stats(n_waves=99, block_records=[10000] + [1] * 98)
    g = classify(stats, certainty_bytes=5, max_trace_bytes=10)
    assert g.verdict == "skew_to_reduced_shape" and g.method == M_REDUCED
    assert g.certainty_budget_exceeded


def test_non_skew_certainty_over_budget_routes_out_typed():
    stats = _stats(n_waves=24)
    g = classify(stats, certainty_bytes=5, max_trace_bytes=10)
    assert g.verdict == "certainty_budget_to_reduced_shape"
    assert g.method == M_REDUCED


# --- homogeneity gate (Codex R15 mainline gap #2) ---

def test_homogeneous_grid_is_subset_admissible():
    g = classify(_stats(n_waves=24))  # uniform per-SM, flat per-wave
    assert g.per_sm_block_count_cv == 0.0
    assert g.n_phases == 1
    assert g.verdict == "subset_admissible"


def test_uneven_sm_occupancy_forbids_subset():
    # One SM carries almost all blocks -> high CV -> structural underfill.
    g = classify(_stats(n_waves=99, per_sm_counts={0: 99, 1: 1, 2: 1}))
    assert g.per_sm_block_count_cv > 0.5
    assert g.verdict == "homogeneity_to_reduced_shape"
    assert g.method == M_REDUCED


def test_fragmented_phases_forbid_subset():
    # Many change-points -> phases cannot fill n_h>=2 per stratum at 8 segments.
    alternating = {w: (100 if w % 2 == 0 else 1000) for w in range(24)}
    g = classify(_stats(n_waves=24, per_wave=alternating))
    assert g.n_phases > 4
    assert g.verdict == "homogeneity_to_reduced_shape"


def test_folded_tail_jump_does_not_create_an_ordinary_phase():
    totals = {wave: 100 for wave in range(24)}
    totals[23] = 10000
    g = classify(_stats(n_waves=24, per_wave=totals))
    assert g.n_phases == 1
    assert g.verdict == "subset_admissible"


def test_crossed_strata_floor_uses_declared_sample_size():
    totals = {wave: (100 if wave < 24 else 1000) for wave in range(48)}
    stats = _stats(n_waves=48, per_wave=totals)
    assert classify(stats).verdict == "homogeneity_to_reduced_shape"
    assert classify(stats, sample_size=14).verdict == "subset_admissible"
