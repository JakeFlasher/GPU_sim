from __future__ import annotations

from copy import deepcopy

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
            "residual": {"baseline_cycles": index + 1},
        }
        for index in range(2)
    ]


def test_decision_is_derived_from_validated_rows() -> None:
    ranking, validation, decision = _derive_decision(_rows())
    assert decision["decision"] == "no-go"
    assert decision["selected_diagnostic"] == "dirty_writeback_coherence_retirement"
    assert validation["decision_eligible"] is True
    assert ranking[0]["selection_score"] == 3


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
