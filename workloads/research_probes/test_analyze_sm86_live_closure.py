from __future__ import annotations

from copy import deepcopy
import importlib.util
import csv
import json
import shutil
from pathlib import Path

import pytest

from workloads.research_probes import analyze_sm86_live_closure as ANALYZER
from workloads.research_probes.analyze_sm86_live_closure import _derive_decision


def _rows() -> list[dict[str, object]]:
    return [
        {
            "cell": f"cell-{index}",
            "comparable_to_simulator_timing": True,
            "live": {"stable_within_five_percent": True},
            "simulator": {
                "readiness_schema": 6,
                "top_readiness_causes": [{"cause": "control_stall", "limiting_cycles": 1}],
                "tails": {"dirty_writeback": True, "coherence": True},
            },
            "residual": {
                "baseline_cycles": index + 1,
                "selection_vector": {
                    "sm_completion_gap_cycles": index + 1,
                    "gpc_completion_gap_cycles": 2,
                    "gpc_launch_envelope_cycles": 3,
                },
            },
        }
        for index in range(2)
    ]


def test_decision_is_derived_from_validated_rows() -> None:
    ranking, validation, decision = _derive_decision(_rows())
    assert decision["decision"] == "no-go"
    assert decision["selected_diagnostic"] == "dirty_writeback_coherence_retirement"
    assert validation["decision_eligible"] is True
    assert validation["aggregate_score_used"] is False
    assert validation["selection_rule"] == "mechanism_local_typed_tail_precedence"
    assert "residual_score_cycles" not in validation
    assert all("selection_score" not in candidate for candidate in ranking)
    assert all(
        candidate["evidence_basis"]["aggregate_score_used"] is False
        for candidate in ranking
    )


def test_frontier_vector_is_preserved_without_scalar_ranking() -> None:
    rows = _rows()
    rows[0]["residual"]["selection_vector"] = {
        "sm_completion_gap_cycles": 10_000_000,
        "gpc_completion_gap_cycles": 1,
        "gpc_launch_envelope_cycles": 2,
    }
    ranking, validation, decision = _derive_decision(rows)
    assert decision["selection_rule"] == "mechanism_local_typed_tail_precedence"
    assert validation["selection_dimensions"] == [
        "sm_completion_gap_cycles",
        "gpc_completion_gap_cycles",
        "gpc_launch_envelope_cycles",
    ]
    assert validation["frontiers_remain_independent"] is True
    assert all("selection_score" not in candidate for candidate in ranking)
    assert rows[0]["residual"]["selection_vector"]["sm_completion_gap_cycles"] == 10_000_000


def test_unstable_repeat_is_not_silently_converted_to_no_go() -> None:
    rows = _rows()
    rows[0]["live"]["stable_within_five_percent"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="insufficient_evidence: live repeat is unstable"):
        _derive_decision(rows)


def test_missing_retirement_tails_is_not_silently_converted_to_no_go() -> None:
    rows = deepcopy(_rows())
    rows[1]["simulator"]["tails"]["dirty_writeback"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="insufficient_evidence: dirty_writeback"):
        _derive_decision(rows)


def _repeat_fixture(tmp_path):
    slug = "gemm_hmma_bf16_sm80"
    cell = {
        "id": slug,
        "binary": {
            "path": "workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run"
        },
    }
    stage = {
        "tool_path": ANALYZER.TRUSTED_NCU_IDENTITY["path"],
        "argv": [
            ".venv/bin/python",
            "tools/local_labels/collect_ncu.py",
            "--output",
            "evidence/raw/ncu/gemm_hmma_bf16_sm80",
            "--metrics",
            ",".join(ANALYZER.METRICS),
            "--launch-skip",
            "1",
            "--launch-count",
            "1",
            "--replay-mode",
            "application",
            "--clock-control",
            "none",
            "--",
            cell["binary"]["path"],
            "--timing_shape",
            "--warmup",
            "1",
            "--reps",
            "1",
        ],
    }
    repeat_dir = "evidence/raw/ncu_repeat/rep-2/gemm_hmma_bf16_sm80"
    report = tmp_path / repeat_dir / "ncu-report.ncu-rep"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"report")
    expected_command, _target = ANALYZER._repeat_command_contract(
        tmp_path, cell, stage, repeat_dir, ANALYZER.TRUSTED_NCU_IDENTITY
    )
    metadata = {
        "command": expected_command,
        "return_code": 0,
        "csv": repeat_dir + "/ncu.csv",
        "report": repeat_dir + "/ncu-report.ncu-rep",
    }
    return cell, stage, repeat_dir, metadata


@pytest.mark.parametrize("return_code", [False, 0.0])
def test_repeat_metadata_rejects_non_integer_success_code(
    tmp_path, return_code
) -> None:
    cell, stage, repeat_dir, metadata = _repeat_fixture(tmp_path)
    metadata["return_code"] = return_code
    with pytest.raises(ValueError, match="return_code is malformed or nonzero"):
        ANALYZER._validate_repeat_metadata(
            tmp_path,
            cell,
            stage,
            repeat_dir,
            metadata,
            ANALYZER.TRUSTED_NCU_IDENTITY,
        )


def test_repeat_metadata_requires_report_artifact(tmp_path) -> None:
    cell, stage, repeat_dir, metadata = _repeat_fixture(tmp_path)
    (tmp_path / repeat_dir / "ncu-report.ncu-rep").unlink()
    with pytest.raises(ValueError, match="file is missing"):
        ANALYZER._validate_repeat_metadata(
            tmp_path,
            cell,
            stage,
            repeat_dir,
            metadata,
            ANALYZER.TRUSTED_NCU_IDENTITY,
        )


def test_replay_summary_accepts_complete_frontier(tmp_path: Path) -> None:
    helper_path = Path(__file__).with_name("test_replay_resume.py")
    spec = importlib.util.spec_from_file_location("replay_summary_fixture", helper_path)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    expectation = helper.REPLAY.ReplayExpectation(
        launches=(helper._launch(),), carded_dram=False
    )
    document = helper._report(expectation)
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    summary = ANALYZER._replay_summary(path)
    assert summary["completion_status"] == "complete"
    assert summary["completion_kind"] == "exact"
    assert summary["completion_cycles"] == 1
    assert summary["completion_lower_bound_cycles"] is None
    assert summary["completion_reference_cycles"] == 1
    assert summary["tails"] == {
        "reduction_read_modify_write": False,
        "dirty_writeback": False,
        "coherence": False,
        "async_copy_shared_visibility": False,
    }


def test_ncu_repeat_csv_cannot_be_substituted_across_same_geometry_cells(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        Path(
            "results/captures/sm86_simulator_live_baseline_20260829_v2/campaign_manifest.json"
        ).read_text(encoding="utf-8")
    )
    cell = next(c for c in manifest["cells"] if c["id"] == "gemm_hmma_bf16_sm80")
    source = Path(
        "results/captures/sm86_simulator_live_baseline_20260829_v2/evidence/raw/ncu_repeat/rep-2/gemm_hmma_fp16_sm80/ncu.csv"
    )
    substituted = tmp_path / "substituted.csv"
    shutil.copyfile(source, substituted)
    with pytest.raises(ValueError, match="process identity|kernel identity"):
        ANALYZER._validate_ncu_csv(
            substituted, cell, "cross-cell-repeat", expected_kernel="gemm_hmma_bf16_sm80_kernel"
        )


def test_ncu_identity_rejects_another_compute_capability_86_gpu() -> None:
    manifest = json.loads(
        Path(
            "results/captures/sm86_simulator_live_baseline_20260829_v2/campaign_manifest.json"
        ).read_text(encoding="utf-8")
    )
    cell = next(c for c in manifest["cells"] if c["id"] == "gemm_hmma_bf16_sm80")
    path = Path(
        "results/captures/sm86_simulator_live_baseline_20260829_v2/evidence/raw/ncu/gemm_hmma_bf16_sm80/ncu.csv"
    )
    row = next(row for row in csv.DictReader(path.open(newline="")) if row.get("Kernel Name"))
    altered = dict(row)
    altered["device__attribute_display_name"] = "NVIDIA A40"
    with pytest.raises(ValueError, match="device/compute-capability identity"):
        ANALYZER._validate_ncu_row_identity(altered, cell, "hardware-fixture")
