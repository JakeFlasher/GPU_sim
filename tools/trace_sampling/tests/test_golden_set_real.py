"""Real golden-set: fail-closed on missing inputs + the real run over the manifest."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import golden_set_real as gsr
from admissibility_gate import classify
from counting_pass import BlockStat, CellStats
from golden_set_real import (
    COUNT_TOL,
    FRACTIONS,
    GOLDEN_ABLATION_EXPECTATIONS,
    GOLDEN_MANIFEST_SHA256,
    N_SEEDS,
    RATE_TOL,
    GoldenSetError,
    _evaluate_ablation_composite,
    _fraction_design,
    _full_shape_rate_interval,
    _manifest_policy_status,
    _segment_membership,
    run_golden_set,
)
from golden_provenance import GoldenProvenanceError
from selection import PopulationSegment, SegmentPopulation

_REPO = Path(__file__).resolve().parents[3]
_MANIFEST = _REPO / "data" / "local_labels" / "campaign" / "golden_set_manifest.json"


def _ablation_estimate(count_error: float, rate_error: float) -> dict:
    sectors = 100.0 * (1.0 + count_error)
    rate = 0.5 + rate_error
    return {"sectors": sectors, "hits": sectors * rate, "l2_hit_rate": rate}


def _ablation_trial(fraction: float, seed: int, *,
                    wave_count: float, block_count: float,
                    wave_rate: float, block_rate: float) -> dict:
    wave = _ablation_estimate(wave_count, wave_rate)
    block = _ablation_estimate(block_count, block_rate)
    return {
        "requested_fraction": fraction,
        "seed": seed,
        "wave_segment_payload": {"blocks": 4},
        "independent_block_payload": {"blocks": 4},
        "independent_draw_partially_retained_segment": True,
        "independent_partial_segment_witness": {
            "segment_id": 7,
            "segment_block_ids": [0, 1, 2],
            "selected_block_ids": [0],
        },
        "wave_segment_estimate": wave,
        "independent_block_estimate": block,
        "wave_segment_count_error_rel": abs(wave["sectors"] - 100.0) / 100.0,
        "independent_block_count_error_rel": (
            abs(block["sectors"] - 100.0) / 100.0),
        "wave_segment_rate_error_abs": abs(wave["l2_hit_rate"] - 0.5),
        "independent_block_rate_error_abs": (
            abs(block["l2_hit_rate"] - 0.5)),
    }


def _ablation_cell(slug: str, expectation: str, *,
                   wave_count: float = 0.005, block_count: float = 0.01,
                   wave_rate: float = 0.001, block_rate: float = 0.01) -> dict:
    return {
        "slug": slug,
        "expectation": expectation,
        "common_production_reference": {
            "sectors": 100.0, "hits": 50.0, "rate": 0.5},
        "trials": [
            _ablation_trial(
                fraction, seed,
                wave_count=wave_count, block_count=block_count,
                wave_rate=wave_rate, block_rate=block_rate)
            for fraction in FRACTIONS for seed in range(N_SEEDS)
        ],
        # These deliberately untrusted summaries prove the evaluator uses trials.
        "wave_segment_count_p95": 99.0,
        "independent_block_count_p95": 0.0,
        "wave_segment_rate_p95": 99.0,
        "independent_block_rate_p95": 0.0,
        "both_arms_accuracy_ok": False,
    }


def _passing_composite_cells() -> list[dict]:
    return [
        _ablation_cell(
            "sensitive", "co_residency_sensitive",
            wave_rate=0.001, block_rate=0.01),
        _ablation_cell(
            "neutral", "neutral",
            wave_rate=0.006, block_rate=0.004),
    ]


_TEST_ABLATION_POLICY = {
    "sensitive": "co_residency_sensitive",
    "neutral": "neutral",
}


def _policy_for(cells: list[dict]) -> dict[str, str]:
    return {cell["slug"]: cell["expectation"] for cell in cells}


def _population(n_segments: int) -> SegmentPopulation:
    return SegmentPopulation(tuple(
        PopulationSegment(
            id=i, wave_ids=(i,), block_ids=(i,),
            sector_accesses=i + 1,
            phase_id=None if i == n_segments - 1 else 0,
            launch_tercile=(None if i == n_segments - 1 else
                            min(2, i * 3 // (n_segments - 1))),
            certainty=i in (0, n_segments - 1), tail=i == n_segments - 1)
        for i in range(n_segments)
    ), segment_len=3)


def _manifest_ready() -> bool:
    if not _MANIFEST.is_file():
        return False
    m = json.loads(_MANIFEST.read_text())
    return all((_REPO / c["trace_dir"] / "dynamic_trace.pb").is_file() for c in m["cells"])


def test_fail_closed_on_missing_trace(tmp_path):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"cells": [
        {"slug": "x", "role": "positive", "ncu_l2_hit_pct": 50.0,
         "trace_dir": "does/not/exist"}]}))
    with pytest.raises(GoldenSetError, match="trace dir missing"):
        run_golden_set(m)


def test_positive_without_ncu_label_fails_closed(tmp_path):
    # A positive cell lacking the error-budget denominator (NCU label) must fail
    # closed — the gate is never silently skipped. Use a real admissible trace.
    if not _manifest_ready():
        pytest.skip("golden traces absent")
    m = json.loads(_MANIFEST.read_text())
    pos = next(c for c in m["cells"] if c["role"] == "positive")
    pos = {k: v for k, v in pos.items() if k != "ncu_l2_hit_pct"}
    mm = tmp_path / "m.json"
    mm.write_text(json.dumps({"cells": [pos]}))
    with pytest.raises(GoldenSetError, match="ncu_l2_hit_pct"):
        run_golden_set(mm)


def test_positive_without_ncu_kernel_identity_fails_closed(tmp_path):
    if not _manifest_ready():
        pytest.skip("golden traces absent")
    manifest = json.loads(_MANIFEST.read_text())
    positive = next(c for c in manifest["cells"] if c["role"] == "positive")
    positive = {k: v for k, v in positive.items() if k != "ncu_kernel_name"}
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps({"cells": [positive]}))
    with pytest.raises(GoldenSetError, match="ncu_kernel_name"):
        run_golden_set(candidate)


def test_positive_with_wrong_ncu_kernel_identity_fails_closed(tmp_path):
    if not _manifest_ready():
        pytest.skip("golden traces absent")
    manifest = json.loads(_MANIFEST.read_text())
    positive = dict(next(c for c in manifest["cells"] if c["role"] == "positive"))
    positive["ncu_kernel_name"] = "different_same_grid_kernel()"
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps({"cells": [positive]}))
    with pytest.raises(GoldenProvenanceError, match="ncu_kernel_mismatch"):
        run_golden_set(candidate)


def test_fraction_ladder_rejects_floor_clamped_population():
    with pytest.raises(GoldenSetError, match="insufficient_segment_population"):
        _fraction_design(_population(9))


def test_fraction_ladder_exercises_all_nominal_budgets():
    design = _fraction_design(_population(128))
    assert [row["selected_segments"] for row in design] == [8, 16, 32]
    assert [row["realized_fraction"] for row in design] == list(FRACTIONS)


def test_full_shape_rate_interval_is_predeclared_and_halfwidth_gated():
    lo, hi, halfwidth = _full_shape_rate_interval(0.40, 0.41)
    assert (lo, hi) == pytest.approx((0.39, 0.42))
    assert halfwidth == pytest.approx(0.015)
    assert halfwidth <= 0.02
    assert _full_shape_rate_interval(0.40, 0.43)[2] > 0.02


def test_subset_helpers_scope_and_preserve_the_first_launch(tmp_path):
    trace = tmp_path / "trace"
    (trace / "dynamic_trace.pb").parent.mkdir(parents=True)
    (trace / "dynamic_trace.pb").write_bytes(b"launch")
    first = trace / "threadblocks" / "device_0" / "stream_0" / "kernel_3"
    later = trace / "threadblocks" / "device_0" / "stream_0" / "kernel_8"
    first.mkdir(parents=True)
    later.mkdir(parents=True)
    (first / "first.pb").write_bytes(b"first")
    (later / "later.pb").write_bytes(b"later")
    assert gsr._pb_files(trace) == [first / "first.pb"]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with gsr._subset_trace_dir(trace, [first / "first.pb"], scratch) as subset:
        assert (subset / first.relative_to(trace) / "first.pb").is_symlink()
        assert not (subset / later.relative_to(trace)).exists()


def test_underfilled_waves_fold_into_one_certainty_tail():
    blocks = [SimpleNamespace(
        block_id=i, wave=i, sm_id=0, records=1, sector_accesses=1)
              for i in range(10)]
    blocks += [SimpleNamespace(
        block_id=100 + i, wave=i, sm_id=1, records=1, sector_accesses=1)
               for i in range(8)]
    stats = SimpleNamespace(
        blocks=blocks,
        grid=(18, 1, 1), n_blocks=18, total_records=18,
        per_sm_resident={0: 1, 1: 1},
        per_sm_block_ids={0: list(range(10)), 1: list(range(100, 108))},
        per_wave_record_totals={
            wave: (2 if wave < 8 else 1) for wave in range(10)},
    )
    membership = _segment_membership(stats)
    assert set(membership.values()) == {0, 1, 2}
    assert sum(segment == 2 for segment in membership.values()) == 6


def test_one_wave_tail_is_refused_before_golden_measurement(
        monkeypatch, tmp_path):
    blocks = [BlockStat(
        block_id=index, sm_id=0, records=100, sector_accesses=4,
        serialized_bytes=10, wave=index)
        for index in range(25)]
    stats = CellStats(
        cell_id="tail", kernel_name="k", grid=(25, 1, 1),
        block=(32, 1, 1), n_blocks=25, total_records=2500,
        op_class_totals={}, blocks=blocks, per_sm_resident={0: 1},
        per_sm_block_ids={0: list(range(25))}, n_waves=25,
        atomic_records=0, atomic_top256_sector_share=0.0,
        per_wave_record_totals={wave: 100 for wave in range(25)},
        per_sm_block_counts={0: 25},
    )
    assert classify(stats).verdict == "tail_measurement_forbid"

    trace = tmp_path / "one_wave_tail"
    trace.mkdir()
    (trace / "dynamic_trace.pb").write_bytes(b"present")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cells": [{
        "slug": "one_wave_tail",
        "role": "positive",
        "trace_dir": str(trace),
        "ncu_l2_hit_pct": 50.0,
        "ncu_csv": "unused.csv",
        "ncu_kernel_name": "k",
    }]}))
    monkeypatch.setattr(gsr, "analyze_trace_dir", lambda *_args: stats)
    monkeypatch.setattr(
        gsr, "_segment_measurements",
        lambda *_args: pytest.fail("typed tail refusal must precede measurement"),
    )

    report = run_golden_set(manifest)
    [row] = report["positive_cells"]
    assert row == {
        "slug": "one_wave_tail",
        "admissible": False,
        "verdict": "tail_measurement_forbid",
    }


def test_segment_measurement_uses_ghost_differencing(monkeypatch, tmp_path):
    # Union decoding would CTA-sort roles.  Give each of two segments one path
    # per wave and require explicit wave-major replay.  The second segment is the
    # fixed-population depth-sensitivity witness and prepends wave 2 for depth 2.
    paths = [tmp_path / f"kernel_1_{i},0,0.pb" for i in range(6)]
    layouts = []
    for segment_id in range(2):
        segment_paths = paths[3 * segment_id:3 * segment_id + 3]
        waves = list(range(3 * segment_id, 3 * segment_id + 3))
        layouts.append({
            "id": segment_id,
            "ghost_paths": segment_paths[:1],
            "measured_paths": segment_paths[1:],
            "ghost_records": 4,
            "measured_records": 9,
            "waves": waves,
            "paths_by_wave": dict(zip(waves, ([path] for path in segment_paths))),
            "records_by_wave": {waves[0]: 4, waves[1]: 5, waves[2]: 4},
        })
    decode_calls = []
    replay_calls = []
    monkeypatch.setattr(gsr, "_segment_layout", lambda _trace, _stats: layouts)

    def decode(_trace, selected, _scratch):
        decode_calls.append(list(selected))
        value = paths.index(selected[0]) + 1
        return [value], [value % 4]

    def replay(lines, _offsets):
        replay_calls.append((list(lines), list(_offsets)))
        return float(len(lines)), float(sum(line % 2 == 0 for line in lines))

    monkeypatch.setattr(gsr, "_decode_access_subset", decode)
    monkeypatch.setattr(gsr, "_replay_accesses", replay)
    depth1, returned_layouts, oracle, sensitivity = gsr._segment_measurements(
        tmp_path, object(), tmp_path)
    assert decode_calls == [[paths[i]] for i in range(6)] + [[paths[2]]]
    assert replay_calls == [
        ([1], [1]), ([1, 2, 3], [1, 2, 3]),
        ([4], [0]), ([4, 5, 6], [0, 1, 2]),
        ([5, 6], [1, 2]), ([3, 4], [3, 0]),
        ([3, 4, 5, 6], [3, 0, 1, 2]),
    ]
    assert returned_layouts == layouts
    assert depth1 == [
        {"id": 0, "sectors": 2.0, "hits": 1.0, "records": 9.0},
        {"id": 1, "sectors": 2.0, "hits": 1.0, "records": 9.0},
    ]
    assert oracle == {"sectors": 4.0, "hits": 2.0, "records": 18.0}
    assert sensitivity["movement_0_to_1"] == 0.0
    assert sensitivity["movement_1_to_2"] == 0.0


def test_ablation_composite_accepts_neutral_cell_rate_reversal():
    cells = _passing_composite_cells()
    before = deepcopy(cells)
    result = _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY)

    assert result["composite_acceptance_ok"]
    assert result["complete_fair_evidence_ok"]
    assert result["per_cell_absolute_accuracy_ok"]
    assert result["pooled_selection_unit_noninferiority_ok"]
    assert result["co_residency_sensitive_witness_ok"]
    assert result["strict_advantage_cell_slugs"] == ["sensitive"]
    assert not result["universal_per_cell_noninferiority"]
    assert not result["per_cell_metrics"]["neutral"][
        "wave_segment_rate_error_le_independent"]
    assert result["typed_failure_reasons"] == []
    assert cells == before
    assert _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY) == result


def test_ablation_composite_rejects_post_hoc_expectation_relabel():
    cells = _passing_composite_cells()
    cells[0]["expectation"] = "neutral"
    cells[1]["expectation"] = "co_residency_sensitive"
    result = _evaluate_ablation_composite(cells, _TEST_ABLATION_POLICY)
    assert not result["complete_fair_evidence_ok"]
    assert not result["composite_acceptance_ok"]
    assert any("ablation_expectation_policy_mismatch" in reason
               for reason in result["typed_failure_reasons"])


def test_authoritative_manifest_bytes_are_policy_pinned():
    raw = _MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == GOLDEN_MANIFEST_SHA256
    assert _manifest_policy_status(raw) == (GOLDEN_MANIFEST_SHA256, True)
    relabeled = raw.replace(
        b'"ablation_expectation": "co_residency_sensitive"',
        b'"ablation_expectation": "neutral"',
        1,
    )
    assert hashlib.sha256(relabeled).hexdigest() != GOLDEN_MANIFEST_SHA256
    assert not _manifest_policy_status(relabeled)[1]


def test_ablation_composite_accepts_neutral_cell_count_reversal():
    cells = [
        _ablation_cell(
            "sensitive", "co_residency_sensitive",
            wave_count=0.005, block_count=0.02,
            wave_rate=0.001, block_rate=0.01),
        _ablation_cell(
            "neutral", "neutral",
            wave_count=0.015, block_count=0.004,
            wave_rate=0.001, block_rate=0.01),
    ]
    result = _evaluate_ablation_composite(cells, _TEST_ABLATION_POLICY)
    assert result["composite_acceptance_ok"]
    assert result["pooled_selection_unit_noninferiority_ok"]
    assert not result["universal_per_cell_noninferiority"]
    assert not result["per_cell_metrics"]["neutral"][
        "wave_segment_count_error_le_independent"]


def test_ablation_composite_rejects_missing_duplicate_and_empty_sensitive_coverage():
    sensitive, neutral = _passing_composite_cells()

    missing = _evaluate_ablation_composite(
        [sensitive], _TEST_ABLATION_POLICY)
    assert not missing["composite_acceptance_ok"]
    assert any(reason.startswith("incomplete_cell_coverage:")
               for reason in missing["typed_failure_reasons"])

    duplicate = _evaluate_ablation_composite(
        [sensitive, deepcopy(sensitive)], _TEST_ABLATION_POLICY)
    assert not duplicate["composite_acceptance_ok"]
    assert "duplicate_cell_coverage:sensitive" in duplicate[
        "typed_failure_reasons"]

    neutral_only = [
        _ablation_cell("neutral_a", "neutral"),
        _ablation_cell("neutral_b", "neutral"),
    ]
    empty_sensitive = _evaluate_ablation_composite(
        neutral_only, _policy_for(neutral_only))
    assert not empty_sensitive["complete_fair_evidence_ok"]
    assert not empty_sensitive["composite_acceptance_ok"]
    assert "co_residency_sensitive_witness_set_empty" in empty_sensitive[
        "typed_failure_reasons"]


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"wave_count": COUNT_TOL + 0.001},
     "sensitive:wave_segment_count_accuracy_failed"),
    ({"block_count": COUNT_TOL + 0.001},
     "sensitive:independent_block_count_accuracy_failed"),
    ({"wave_rate": RATE_TOL + 0.001},
     "sensitive:wave_segment_rate_accuracy_failed"),
    ({"block_rate": RATE_TOL + 0.001},
     "sensitive:independent_block_rate_accuracy_failed"),
])
def test_ablation_composite_rejects_each_inaccurate_arm_metric(
        overrides, reason):
    cells = _passing_composite_cells()
    cells[0] = _ablation_cell(
        "sensitive", "co_residency_sensitive", **overrides)
    result = _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY)
    assert not result["composite_acceptance_ok"]
    assert not result["per_cell_absolute_accuracy_ok"]
    assert reason in result["typed_failure_reasons"]


def test_ablation_composite_rejects_pooled_count_regression():
    cells = [
        _ablation_cell(
            "sensitive", "co_residency_sensitive",
            wave_count=0.02, block_count=0.01,
            wave_rate=0.001, block_rate=0.01),
        _ablation_cell(
            "neutral", "neutral",
            wave_count=0.02, block_count=0.01,
            wave_rate=0.001, block_rate=0.01),
    ]
    result = _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY)
    assert not result["composite_acceptance_ok"]
    assert not result["pooled_selection_unit_noninferiority_ok"]
    assert "pooled_wave_segment_count_regression" in result[
        "typed_failure_reasons"]


def test_ablation_composite_rejects_pooled_rate_regression():
    cells = [
        _ablation_cell(
            "sensitive", "co_residency_sensitive",
            wave_rate=0.001, block_rate=0.002),
        _ablation_cell(
            "neutral", "neutral",
            wave_rate=0.015, block_rate=0.001),
    ]
    result = _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY)
    assert not result["composite_acceptance_ok"]
    assert not result["pooled_selection_unit_noninferiority_ok"]
    assert "pooled_wave_segment_rate_regression" in result[
        "typed_failure_reasons"]


def test_ablation_composite_rejects_masked_sensitive_rate_regression():
    cells = [
        _ablation_cell(
            "sensitive", "co_residency_sensitive",
            wave_rate=0.01, block_rate=0.005),
        _ablation_cell(
            "neutral_a", "neutral",
            wave_rate=0.001, block_rate=0.015),
        _ablation_cell(
            "neutral_b", "neutral",
            wave_rate=0.001, block_rate=0.015),
    ]
    result = _evaluate_ablation_composite(cells, _policy_for(cells))
    assert result["pooled_selection_unit_noninferiority_ok"]
    assert not result["co_residency_sensitive_witness_ok"]
    assert not result["composite_acceptance_ok"]
    assert "sensitive:co_residency_sensitive_rate_regression" in result[
        "typed_failure_reasons"]


def test_ablation_composite_requires_strict_sensitive_advantage():
    cells = [
        _ablation_cell(
            "sensitive", "co_residency_sensitive",
            wave_rate=0.005, block_rate=0.005),
        _ablation_cell(
            "neutral", "neutral",
            wave_rate=0.001, block_rate=0.01),
    ]
    result = _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY)
    assert result["pooled_selection_unit_noninferiority_ok"]
    assert not result["co_residency_sensitive_witness_ok"]
    assert not result["composite_acceptance_ok"]
    assert "co_residency_sensitive_strict_advantage_missing" in result[
        "typed_failure_reasons"]


@pytest.mark.parametrize(("mutation", "reason"), [
    ("block_budget", "sensitive:ablation_block_budget_mismatch"),
    ("whole_segment", "sensitive:independent_block_draw_preserved_whole_segments"),
    ("error_ledger", (
        "sensitive:inconsistent_trial_error:0:wave_segment_rate_error_abs")),
])
def test_ablation_composite_rejects_unfair_or_inconsistent_trial(
        mutation, reason):
    cells = _passing_composite_cells()
    trial = cells[0]["trials"][0]
    if mutation == "block_budget":
        trial["independent_block_payload"]["blocks"] += 1
    elif mutation == "whole_segment":
        trial["independent_partial_segment_witness"]["selected_block_ids"] = [0, 1, 2]
    else:
        trial["wave_segment_rate_error_abs"] += 0.001
    result = _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY)
    assert not result["complete_fair_evidence_ok"]
    assert not result["composite_acceptance_ok"]
    assert reason in result["typed_failure_reasons"]


def test_ablation_composite_rejects_duplicate_fraction_seed_grid_point():
    missing_cells = _passing_composite_cells()
    missing_cells[0]["trials"].pop()
    missing = _evaluate_ablation_composite(
        missing_cells, _TEST_ABLATION_POLICY)
    assert len(missing_cells[0]["trials"]) == len(FRACTIONS) * N_SEEDS - 1
    assert not missing["complete_fair_evidence_ok"]
    assert "sensitive:incomplete_trial_grid" in missing[
        "typed_failure_reasons"]

    cells = _passing_composite_cells()
    cells[0]["trials"][-1]["requested_fraction"] = FRACTIONS[0]
    cells[0]["trials"][-1]["seed"] = 0
    result = _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY)
    assert not result["complete_fair_evidence_ok"]
    assert "sensitive:incomplete_trial_grid" in result["typed_failure_reasons"]
    assert "sensitive:duplicate_trial_grid_point" in result[
        "typed_failure_reasons"]


@pytest.mark.parametrize("supplied_error", [-0.001, float("nan")])
def test_ablation_composite_rejects_negative_or_nonfinite_error_ledger(
        supplied_error):
    cells = _passing_composite_cells()
    cells[0]["trials"][0]["wave_segment_rate_error_abs"] = supplied_error
    result = _evaluate_ablation_composite(
        cells, _TEST_ABLATION_POLICY)
    assert not result["complete_fair_evidence_ok"]
    assert not result["composite_acceptance_ok"]
    assert (
        "sensitive:inconsistent_trial_error:0:wave_segment_rate_error_abs"
        in result["typed_failure_reasons"])


@pytest.mark.skipif(not _manifest_ready(), reason="golden traces absent")
def test_real_golden_set_passes_all_gates():
    r = run_golden_set(_MANIFEST)
    assert r["pass"], r["failures"]
    assert r["schema_version"] == 2
    assert r["manifest_policy_ok"]
    assert r["manifest_sha256"] == r["manifest_policy_sha256"]
    # >=2 real subset-admissible positives, all admissible.
    pos = r["positive_cells"]
    assert len(pos) >= 2 and all(p["admissible"] for p in pos)
    # Every declared gate holds per positive cell.
    for p in pos:
        assert p["measurement_protocol"] == "per_segment_ghost_differencing"
        for g in ("count_accuracy_ok", "rate_accuracy_ok",
                  "count_ci_coverage_ok", "rate_ci_coverage_ok",
                  "rate_ci_halfwidth_ok", "count_ci_halfwidth_ok",
                  "min_segments_ok", "calibration_ok", "error_budget_ok",
                  "ghost_depth_sensitivity_ok", "estimator_status_ok"):
            assert p[g], (p["slug"], g)
        assert len(p["trials"]) == len(FRACTIONS) * N_SEEDS
        assert all(t["status"] == "converged" for t in p["trials"])
        assert all(t["full_shape_rate_ci_halfwidth_bound_abs"] <= 0.02
                   for t in p["trials"])
        assert len(p["provenance"]["ncu"]["csv_sha256"]) == 64
        assert p["provenance"]["trace"]["first_launch"]["block_count"] == 36864
    assert r["manifest_cell_count_ok"]
    assert all(r["fidelity_aggregate"][key] for key in (
        "count_accuracy_ok", "rate_accuracy_ok",
        "count_ci_coverage_ok", "rate_ci_coverage_ok",
        "raw_ghost_rate_ci_coverage_ok"))
    assert r["error_budget_aggregate"]["pass"]
    assert r["count_validation"]["pass"]
    assert "gs_var_stride" in (
        r["count_validation"]["independent_block_variation_cells"])
    # negative controls all gate-forbidden.
    assert r["negative_controls"] and all(n["forbidden"] for n in r["negative_controls"])
    histogram = next(n for n in r["negative_controls"]
                     if n["slug"] == "ac4_global_atomic_histogram")
    assert "atomic_gate" in histogram["forbidding_gates"]
    # the wave-segment-vs-independent-block ablation is present + gated (real, not
    # only in the synthetic smoke).
    abl = r["ablation"]
    assert r["versions"]["ablation_policy"] == "fixed-design-composite-v1"
    assert abl["policy_version"] == r["versions"]["ablation_policy"]
    assert abl["ablation_present"]
    assert not abl["causal_co_residency_claim"]
    assert abl["wave_segment_not_worse"]
    assert abl["complete_fair_evidence_ok"]
    assert abl["per_cell_absolute_accuracy_ok"]
    assert abl["pooled_selection_unit_noninferiority_ok"]
    assert abl["co_residency_sensitive_witness_ok"]
    assert abl["composite_acceptance_ok"]
    assert not abl["universal_per_cell_noninferiority"]
    assert not abl["per_cell_not_worse"]
    assert abl["sensitive_cell_slugs"] == ["gs_stream_reuse"]
    assert abl["strict_advantage_cell_slugs"] == ["gs_stream_reuse"]
    assert abl["expected_expectations"] == GOLDEN_ABLATION_EXPECTATIONS
    assert abl["typed_failure_reasons"] == []
    assert len(abl["per_cell"]) == len(pos)
    for a in abl["per_cell"]:
        assert a["n_trials"] == len(FRACTIONS) * N_SEEDS
        assert a["protocol"] == (
            "production_ghost_differenced_estimator_vs_"
            "independent_block_cold_subtrace")
        assert not a["causal_co_residency_claim"]
        assert a["both_arms_accuracy_ok"]
        assert a["co_residency_effect_gate_ok"]
        assert a["per_cell_absolute_accuracy_ok"]
        assert isinstance(a["wave_segment_count_error_le_independent"], bool)
        assert isinstance(a["wave_segment_rate_error_le_independent"], bool)
        assert a["typed_failure_reasons"] == []
        assert "rate_noninferiority_margin" not in a
        design = a["fraction_design"]
        selected = [d["selected_segments"] for d in design]
        assert all(b > a for a, b in zip(selected, selected[1:]))
        assert all(abs(d["realized_fraction"] - d["requested_fraction"])
                   <= 0.5 / d["population_segments"] + 1e-12 for d in design)
        trials = a["trials"]
        assert all(
            trial["wave_segment_payload"]["blocks"]
            == trial["independent_block_payload"]["blocks"]
            for trial in trials)
        assert all(
            trial["independent_draw_partially_retained_segment"]
            for trial in trials)
        assert all(
            0 < len(trial["independent_partial_segment_witness"]["selected_block_ids"])
            < len(trial["independent_partial_segment_witness"]["segment_block_ids"])
            for trial in trials)
        assert all(
            trial["independent_block_estimate"]["l2_hit_rate"]
            == pytest.approx(
                trial["independent_block_estimate"]["hits"]
                / trial["independent_block_estimate"]["sectors"])
            for trial in trials)
        realized_by_fraction = [
            min(t["wave_segment_payload"]["realized_block_fraction"]
                for t in trials if t["requested_fraction"] == fraction)
            for fraction in FRACTIONS
        ]
        assert all(b > a for a, b in zip(realized_by_fraction,
                                         realized_by_fraction[1:]))
        assert a["certainty_tail"]["kind"] == "probability_one_residual_wave_tail"
        for metric in ("wave_segment_rate_p95", "independent_block_rate_p95",
                       "wave_segment_count_p95", "independent_block_count_p95"):
            assert metric in a
    neutral_reversal = next(
        a for a in abl["per_cell"] if a["slug"] == "gs_var_stride")
    assert neutral_reversal["expectation"] == "neutral"
    assert not neutral_reversal["wave_segment_rate_error_le_independent"]
    assert not neutral_reversal["universal_per_cell_noninferiority"]
    assert neutral_reversal["typed_diagnostic_reasons"] == [
        "wave_segment_rate_regression"]
    sensitive = next(
        a for a in abl["per_cell"] if a["slug"] == "gs_stream_reuse")
    assert sensitive["expectation"] == "co_residency_sensitive"
    assert sensitive["wave_segment_rate_error_le_independent"]
    assert {
        "wave_segment_rate_p95", "independent_block_rate_p95",
        "wave_segment_count_p95", "independent_block_count_p95",
    } <= set(abl["aggregate"])
    assert abl["aggregate"]["both_arms_accuracy_ok"]
    assert abl["aggregate"]["wave_segment_count_error_le_independent"]
    assert abl["aggregate"]["wave_segment_rate_error_le_independent"]
    assert all(abl["aggregate"][key] is not None for key in (
        "wave_segment_rate_p95", "independent_block_rate_p95",
        "wave_segment_count_p95", "independent_block_count_p95"))
    # No required golden-set gate is omitted from the verdict.
    assert "sim_vs_ncu_error_budget" not in r["typed_out_of_scope_gaps"]
    assert "hit_rate_ci" not in r["typed_out_of_scope_gaps"]
