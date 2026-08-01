from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.contracts.validation import loads_json_strict, validate_report
from eval.labels import known_synthetic_authority_pin, load_synthetic_matched_labels
from eval.score import (
    EvaluatorScoringError,
    ResultSnapshot,
    score_synthetic_fixture,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).parent / "fixtures/evaluator_mvp"
FIXTURE_ID = "evaluator-mvp-canary-v1"
RESULT_PATH = FIXTURE_DIR / "sim_result.v2.json"
LABELS_PATH = FIXTURE_DIR / "matched_labels.v1.json"
LABEL_AUTHORITY_PATH = FIXTURE_DIR / "synthetic_matched_label_authority.v1.json"
EVALUATOR_IDENTITY_PATH = (
    FIXTURE_DIR / "synthetic_evaluator_build_identity.v1.json"
)


def _score() -> dict[str, object]:
    return score_synthetic_fixture(
        RESULT_PATH,
        LABELS_PATH,
        LABEL_AUTHORITY_PATH,
        EVALUATOR_IDENTITY_PATH,
        fixture_id=FIXTURE_ID,
    )


def _rows(report: dict[str, object]) -> dict[str, dict[str, object]]:
    outcome = report["outcome"]
    assert isinstance(outcome, dict)
    rows = outcome["per_metric_family"]
    assert isinstance(rows, list)
    return {row["metric_key"]: row for row in rows}


def test_scores_exact_seven_metrics_and_keeps_cycles_blocked() -> None:
    report = _score()
    labels = load_synthetic_matched_labels(
        LABELS_PATH,
        LABEL_AUTHORITY_PATH,
        authority_pin=known_synthetic_authority_pin(FIXTURE_ID),
    )
    validate_report(report, label_authority=labels.authority)

    outcome = report["outcome"]
    assert outcome["status"] == "admitted"
    assert outcome["comparison_failures"] == []
    assert outcome["coverage"] == {
        "total_cells": 8,
        "labeled_cells": 8,
        "unlabeled_cells": 0,
        "scored_cells": 7,
        "unavailable_cells": 1,
        "comparison_failure_cells": 0,
        "silent_absence_count": 0,
        "unavailable_by_type": [
            {
                "status": "required_unimplemented",
                "blocking": "aggregation_semantics_study",
                "count": 1,
            }
        ],
    }

    rows = _rows(report)
    assert tuple(rows) == (
        "smsp_inst_executed_sum",
        "traffic_gmem_ld_sectors",
        "traffic_gmem_st_sectors",
        "traffic_dram_read_sectors",
        "cache_hit_rate_l1tex_read_sector",
        "cache_hit_rate_l1tex_sector",
        "cache_hit_rate_l2_sector",
        "exec_cycles_gpc_max",
    )
    assert rows["smsp_inst_executed_sum"]["summary"]["exact_count"] == 1
    assert rows["traffic_gmem_ld_sectors"]["summary"]["relative_error"] == {
        "n": 1,
        "mean_absolute_error": 0.875,
        "max_absolute_error": 0.875,
        "signed_bias": -0.875,
    }
    assert rows["cache_hit_rate_l1tex_read_sector"]["summary"][
        "error_percentage_points"
    ] == {
        "n": 1,
        "mean_absolute_error": 25,
        "max_absolute_error": 25,
        "signed_bias": 25,
    }
    assert rows["cache_hit_rate_l1tex_sector"]["within_label_noise"][
        "within_count"
    ] == 1
    cycles = rows["exec_cycles_gpc_max"]
    assert cycles["n_scored"] == 0
    assert cycles["n_unavailable"] == 1
    assert cycles["summary"] == {
        "kind": "cycle",
        "status": "required_unimplemented",
        "blocking": "aggregation_semantics_study",
    }


def test_report_is_deterministic_and_fixture_build_is_unclaimable() -> None:
    first = _score()
    second = _score()

    assert first == second
    assert first["report_sha256"] == (
        "f992da35ef55f0e3c7d77bf1101b08829017901fdc6cb92c3f2bda30162f2f15"
    )
    assert first["identities"]["evaluator_build_claimability"] == {
        "status": "unclaimable",
        "reason": "dirty_build",
    }
    assert first["identities"]["report_id"].startswith("synthetic:")


def test_result_snapshot_rechecks_retained_bytes() -> None:
    snapshot = ResultSnapshot(raw_bytes=RESULT_PATH.read_bytes(), file_sha256="0" * 64)

    with pytest.raises(EvaluatorScoringError, match="RESULT_SNAPSHOT_HASH_MISMATCH"):
        snapshot.validated_document()


def test_cli_emits_one_canonical_report_line() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.score",
            FIXTURE_ID,
            str(RESULT_PATH),
            str(LABELS_PATH),
            str(LABEL_AUTHORITY_PATH),
            str(EVALUATOR_IDENTITY_PATH),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    report = loads_json_strict(completed.stdout, source="score stdout")
    assert completed.stdout == (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def test_cli_input_failure_is_typed_and_channel_exclusive(tmp_path: Path) -> None:
    result = loads_json_strict(RESULT_PATH.read_text(encoding="utf-8"))
    result["result_sha256"] = "0" * 64
    invalid_result = tmp_path / "invalid-result.json"
    invalid_result.write_text(
        json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.score",
            FIXTURE_ID,
            str(invalid_result),
            str(LABELS_PATH),
            str(LABEL_AUTHORITY_PATH),
            str(EVALUATOR_IDENTITY_PATH),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    error = loads_json_strict(completed.stderr, source="score stderr")
    assert error["code"] == "CANONICAL_SEAL_MISMATCH"
    assert completed.stderr.endswith("\n")


@pytest.mark.parametrize(
    "material",
    [
        b"[" * 10_000 + b"]" * 10_000,
        b'{"value":' + b"9" * 5_000 + b"}\n",
    ],
    ids=["excessive-depth", "excessive-integer-digits"],
)
def test_cli_malformed_json_resource_failure_is_typed(
    tmp_path: Path, material: bytes
) -> None:
    invalid_result = tmp_path / "resource-invalid-result.json"
    invalid_result.write_bytes(material)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.score",
            FIXTURE_ID,
            str(invalid_result),
            str(LABELS_PATH),
            str(LABEL_AUTHORITY_PATH),
            str(EVALUATOR_IDENTITY_PATH),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    error = loads_json_strict(completed.stderr, source="score stderr")
    assert error == {
        "code": "EVALUATOR_INPUT_JSON_RESOURCE_INVALID",
        "detail": "simulator result JSON exceeds deterministic parser limits",
    }
    assert completed.stderr.endswith("\n")


def test_cli_usage_failure_is_typed_and_channel_exclusive() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "eval.score"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    error = loads_json_strict(completed.stderr, source="score stderr")
    assert error["code"] == "CLI_USAGE_FAILED"
    assert completed.stderr.endswith("\n")
