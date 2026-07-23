"""Golden-set fidelity: accuracy/CI gates, negative controls, ablation."""
from admissibility_gate import M_SUBSET, classify
from golden_set import (CI_COVERAGE_MIN, COUNT_TOL, _gate_stats, run_controlled_smoke,
                        validate_estimator_cell, _streaming_cell)


def test_golden_set_passes_overall():
    r = run_controlled_smoke()
    assert r["pass"], r["failures"]
    assert "rate_and_error_budget" in r["smoke_notes"]  # rates/budget live in the real harness


def test_admissible_cells_meet_accuracy_and_coverage_gates():
    r = run_controlled_smoke()
    assert r["admissible_cells"]
    for c in r["admissible_cells"]:
        assert c["accuracy_ok"], (c["cell_id"], c["p95_rel_error"], COUNT_TOL)
        assert c["ci_coverage_ok"], (c["cell_id"], c["ci_coverage"], CI_COVERAGE_MIN)


def test_all_negative_controls_are_gate_forbidden():
    # The single most important check: every negative control MUST be rejected.
    r = run_controlled_smoke()
    assert len(r["negative_controls"]) >= 3
    for n in r["negative_controls"]:
        assert n["correctly_forbidden"], (n["cell_id"], n["verdict"])
        assert n["method"] != M_SUBSET


def test_gate_actually_discriminates_not_forbid_everything():
    # A clean large-grid cell IS admitted, so the negative controls' rejection is
    # meaningful (the gate is not trivially forbidding everything).
    # Use an exact ordinary-wave tail.  A one-wave residual tail is
    # intentionally routed away from subset sampling because ghost accounting
    # leaves it with no measured waves.
    clean = _gate_stats(n_waves=99, block_records=[100] * 200)
    assert classify(clean).method == M_SUBSET
    # ... while a histogram-class cell is forbidden.
    histo = _gate_stats(n_waves=100, atomic_density=0.02, atomic_top256=0.9)
    assert classify(histo).method != M_SUBSET


def test_ablation_wave_segment_not_worse_than_independent_block():
    r = run_controlled_smoke()
    a = r["ablation"]
    # For additive counts both are ~unbiased; wave-segment must not be worse.
    assert a["wave_segment_p95"] <= a["independent_block_p95"] + 0.02


def test_estimator_cell_scoring_is_deterministic():
    cell = _streaming_cell("det", 256, 0.1, seed=3)
    a = validate_estimator_cell(cell)
    b = validate_estimator_cell(cell)
    assert a["p95_rel_error"] == b["p95_rel_error"]
    assert a["ci_coverage"] == b["ci_coverage"]
