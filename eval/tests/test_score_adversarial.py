from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

import pytest

import eval.labels as labels_module
from eval.contracts.validation import (
    METRIC_KEYS,
    ContractValidationError,
    canonical_launch_key_token,
    canonical_sha256,
    loads_json_strict,
)
from eval.score import load_result_snapshot, score_synthetic_fixture

FIXTURE_DIR = Path(__file__).parent / "fixtures/evaluator_mvp"
FIXTURE_ID = "evaluator-mvp-canary-v1"
RESULT_PATH = FIXTURE_DIR / "sim_result.v2.json"
LABELS_PATH = FIXTURE_DIR / "matched_labels.v1.json"
LABEL_AUTHORITY_PATH = FIXTURE_DIR / "synthetic_matched_label_authority.v1.json"
EVALUATOR_IDENTITY_PATH = (
    FIXTURE_DIR / "synthetic_evaluator_build_identity.v1.json"
)
COUNT_KEYS = METRIC_KEYS[:4]
RATE_KEYS = METRIC_KEYS[4:7]


def _load(path: Path) -> dict[str, object]:
    value = loads_json_strict(path.read_text(encoding="utf-8"), source=str(path))
    assert isinstance(value, dict)
    return value


def _encode(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reseal(value: dict[str, object], member: str) -> None:
    value[member] = "0" * 64
    value[member] = canonical_sha256(value, member)


def _write_result(
    tmp_path: Path, mutate: Callable[[dict[str, object]], None]
) -> Path:
    result = deepcopy(_load(RESULT_PATH))
    mutate(result)
    _reseal(result, "result_sha256")
    path = tmp_path / "sim_result.v2.json"
    path.write_bytes(_encode(result))
    return path


def _write_label_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
    *,
    reseal_labels: bool = True,
) -> tuple[Path, Path]:
    labels = deepcopy(_load(LABELS_PATH))
    mutate(labels)
    if reseal_labels:
        _reseal(labels, "matched_labels_sha256")
    labels_bytes = _encode(labels)
    labels_path = tmp_path / "matched_labels.v1.json"
    labels_path.write_bytes(labels_bytes)

    authority = deepcopy(_load(LABEL_AUTHORITY_PATH))
    source = labels["source_label_identity"]
    crosswalk = labels["crosswalk_identity"]
    authority["matched_labels_file_sha256"] = hashlib.sha256(labels_bytes).hexdigest()
    authority["matched_labels_sha256"] = labels["matched_labels_sha256"]
    authority["label_set_id"] = source["label_set_id"]
    authority["label_set_sha256"] = source["label_set_sha256"]
    authority["crosswalk_id"] = crosswalk["crosswalk_id"]
    authority["crosswalk_sha256"] = crosswalk["crosswalk_sha256"]
    _reseal(authority, "authority_sha256")
    authority_bytes = _encode(authority)
    authority_path = tmp_path / "synthetic_matched_label_authority.v1.json"
    authority_path.write_bytes(authority_bytes)
    monkeypatch.setattr(
        labels_module,
        "KNOWN_SYNTHETIC_AUTHORITY_FILE_SHA256",
        {FIXTURE_ID: hashlib.sha256(authority_bytes).hexdigest()},
    )
    return labels_path, authority_path


def _score_with_labels(labels_path: Path, authority_path: Path) -> dict[str, object]:
    return score_synthetic_fixture(
        RESULT_PATH,
        labels_path,
        authority_path,
        EVALUATOR_IDENTITY_PATH,
        fixture_id=FIXTURE_ID,
    )


def _row(report: dict[str, object], metric_key: str, family: str = "vector") -> dict:
    return next(
        row
        for row in report["outcome"]["per_metric_family"]
        if row["metric_key"] == metric_key and row["kernel_family"] == family
    )


def test_label_only_launch_uses_full_outer_join_and_separate_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def add_label_launch(labels: dict[str, object]) -> None:
        launch = deepcopy(labels["launches"][0])
        launch["launch_key"]["matched_execution_id"] = "labels-only-execution"
        launch["launch_key"]["process_launch_ordinal"] = 1
        launch["launch_key"]["stream_id"] = 8
        launch["kernel_family"] = "matrix"
        labels["launches"].append(launch)

    labels_path, authority_path = _write_label_bundle(
        tmp_path, monkeypatch, add_label_launch
    )
    report = _score_with_labels(labels_path, authority_path)
    outcome = report["outcome"]

    assert outcome["coverage"] == {
        "total_cells": 16,
        "labeled_cells": 16,
        "unlabeled_cells": 0,
        "scored_cells": 7,
        "unavailable_cells": 2,
        "comparison_failure_cells": 7,
        "silent_absence_count": 0,
        "unavailable_by_type": [
            {
                "status": "required_unimplemented",
                "blocking": "aggregation_semantics_study",
                "count": 2,
            }
        ],
    }
    assert len(outcome["per_metric_family"]) == 16
    assert [
        (row["metric_key"], row["kernel_family"])
        for row in outcome["per_metric_family"]
    ] == [(key, family) for key in METRIC_KEYS for family in ("matrix", "vector")]
    assert all(
        failure["failure_type"] == "MISSING_PREDICTION"
        for failure in outcome["comparison_failures"]
    )
    assert _row(report, "smsp_inst_executed_sum", "matrix")["n_scored"] == 0
    assert _row(report, "exec_cycles_gpc_max", "matrix")["n_unavailable"] == 1


def test_prediction_only_launch_counts_every_metric_as_missing_label(
    tmp_path: Path,
) -> None:
    def add_prediction_launch(result: dict[str, object]) -> None:
        launch = deepcopy(result["launches"][0])
        launch["launch_key"]["matched_execution_id"] = "prediction-only-execution"
        launch["launch_key"]["process_launch_ordinal"] = 1
        launch["launch_key"]["stream_id"] = 8
        result["launches"].append(launch)

        diagnostic = deepcopy(result["diagnostics"]["per_launch"][0])
        diagnostic["launch_key"] = deepcopy(launch["launch_key"])
        result["diagnostics"]["per_launch"].append(diagnostic)
        result["aggregate"]["launch_count"] = 2
        for key in COUNT_KEYS:
            result["aggregate"]["metrics"][key]["value"]["value"] *= 2
        for key in RATE_KEYS:
            value = result["aggregate"]["metrics"][key]["value"]
            value["numerator"] *= 2
            value["denominator"] *= 2

    result_path = _write_result(tmp_path, add_prediction_launch)
    report = score_synthetic_fixture(
        result_path,
        LABELS_PATH,
        LABEL_AUTHORITY_PATH,
        EVALUATOR_IDENTITY_PATH,
        fixture_id=FIXTURE_ID,
    )
    outcome = report["outcome"]

    assert outcome["coverage"]["total_cells"] == 16
    assert outcome["coverage"]["labeled_cells"] == 8
    assert outcome["coverage"]["unlabeled_cells"] == 8
    assert outcome["coverage"]["scored_cells"] == 7
    assert outcome["coverage"]["unavailable_cells"] == 1
    assert outcome["coverage"]["comparison_failure_cells"] == 8
    assert all(
        failure["failure_type"] == "MISSING_LABEL"
        for failure in outcome["comparison_failures"]
    )
    observed_order = [
        (
            canonical_launch_key_token(failure["context"]["launch_key"]),
            METRIC_KEYS.index(failure["context"]["metric_key"]),
        )
        for failure in outcome["comparison_failures"]
    ]
    assert observed_order == sorted(observed_order)


def test_zero_label_has_no_epsilon_relative_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def zero_instruction_label(labels: dict[str, object]) -> None:
        labels["launches"][0]["metrics"][0]["outcome"]["value"] = 0.0

    labels_path, authority_path = _write_label_bundle(
        tmp_path, monkeypatch, zero_instruction_label
    )
    row = _row(
        _score_with_labels(labels_path, authority_path),
        "smsp_inst_executed_sum",
    )

    assert row["summary"]["zero_label_nonzero_prediction_count"] == 1
    assert row["summary"]["absolute_error"] == {
        "n": 1,
        "mean_absolute_error": 64,
        "max_absolute_error": 64,
        "signed_bias": 64,
    }
    assert row["summary"]["relative_error"] == {
        "n": 0,
        "mean_absolute_error": None,
        "max_absolute_error": None,
        "signed_bias": None,
    }


def test_unavailable_label_is_explicit_comparison_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make_label_unavailable(labels: dict[str, object]) -> None:
        labels["launches"][0]["metrics"][1]["outcome"] = {
            "status": "unavailable",
            "value": None,
            "reason": "source_null",
        }

    labels_path, authority_path = _write_label_bundle(
        tmp_path, monkeypatch, make_label_unavailable
    )
    report = _score_with_labels(labels_path, authority_path)
    outcome = report["outcome"]

    assert outcome["coverage"]["labeled_cells"] == 7
    assert outcome["coverage"]["unlabeled_cells"] == 1
    assert outcome["coverage"]["comparison_failure_cells"] == 1
    assert outcome["comparison_failures"][0]["failure_type"] == "UNSUPPORTED_OUTCOME"
    assert outcome["comparison_failures"][0]["context"]["observed_status"] == (
        "label_unavailable"
    )


def test_zero_denominator_fraction_is_typed_unavailable(tmp_path: Path) -> None:
    def make_fraction_undefined(result: dict[str, object]) -> None:
        for parent in (result["launches"][0], result["aggregate"]):
            parent["metrics"]["cache_hit_rate_l1tex_read_sector"] = {
                "status": "predicted",
                "value": {
                    "kind": "fraction",
                    "numerator": 0,
                    "denominator": 0,
                    "value": None,
                    "semantics": "simulator_scope",
                },
            }

    result_path = _write_result(tmp_path, make_fraction_undefined)
    report = score_synthetic_fixture(
        result_path,
        LABELS_PATH,
        LABEL_AUTHORITY_PATH,
        EVALUATOR_IDENTITY_PATH,
        fixture_id=FIXTURE_ID,
    )
    row = _row(report, "cache_hit_rate_l1tex_read_sector")

    assert row["n_scored"] == 0
    assert row["n_unavailable"] == 1
    assert row["unavailable_by_type"] == [
        {
            "status": "unavailable",
            "reason": "undefined_fraction",
            "detail": "zero_denominator",
            "count": 1,
        }
    ]
    assert report["outcome"]["coverage"]["scored_cells"] == 6
    assert report["outcome"]["coverage"]["unavailable_cells"] == 2


def test_schema_valid_memory_mode_unavailability_is_preserved(
    tmp_path: Path,
) -> None:
    def use_l1_const_mode(result: dict[str, object]) -> None:
        configuration = result["resolved_configuration"]
        configuration["replay_mode"] = {"mode": "l1_const"}
        unmodeled_unit = {
            "status": "unmodeled",
            "reason": "replay_mode_without_shared_l2",
        }
        configuration["l2"] = deepcopy(unmodeled_unit)
        configuration["dram"] = deepcopy(unmodeled_unit)

        unavailable_l2 = {
            "status": "unavailable",
            "detail": {
                "reason": "mem_mode_lacks_unit",
                "needs": "lts",
            },
        }
        unavailable_dram = {
            "status": "unavailable",
            "detail": {
                "reason": "mem_mode_lacks_unit",
                "needs": "dram",
            },
        }
        for parent in (result["launches"][0], result["aggregate"]):
            metrics = parent["metrics"]
            metrics["cache_hit_rate_l2_sector"] = deepcopy(unavailable_l2)
            metrics["traffic_dram_read_sectors"] = deepcopy(unavailable_dram)

    result_path = _write_result(tmp_path, use_l1_const_mode)
    report = score_synthetic_fixture(
        result_path,
        LABELS_PATH,
        LABEL_AUTHORITY_PATH,
        EVALUATOR_IDENTITY_PATH,
        fixture_id=FIXTURE_ID,
    )
    row = _row(report, "cache_hit_rate_l2_sector")

    assert row["n_scored"] == 0
    assert row["n_unavailable"] == 1
    assert row["unavailable_by_type"] == [
        {
            "status": "unavailable",
            "reason": "mem_mode_lacks_unit",
            "needs": "lts",
            "count": 1,
        }
    ]


def test_nonzero_fraction_scores_exact_operands_not_rounded_value(
    tmp_path: Path,
) -> None:
    numerator = 2**53
    denominator = 2**54 + 1

    def make_fraction_lossy(result: dict[str, object]) -> None:
        for parent in (result["launches"][0], result["aggregate"]):
            value = parent["metrics"]["cache_hit_rate_l1tex_read_sector"]["value"]
            value["numerator"] = numerator
            value["denominator"] = denominator
            value["value"] = 0.5

    result_path = _write_result(tmp_path, make_fraction_lossy)
    report = score_synthetic_fixture(
        result_path,
        LABELS_PATH,
        LABEL_AUTHORITY_PATH,
        EVALUATOR_IDENTITY_PATH,
        fixture_id=FIXTURE_ID,
    )
    row = _row(report, "cache_hit_rate_l1tex_read_sector")
    error = row["summary"]["error_percentage_points"]

    assert row["summary"]["exact_count"] == 0
    assert error["n"] == 1
    assert error["signed_bias"] < 0
    assert error["mean_absolute_error"] > 0


def test_noise_threshold_is_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def set_exact_threshold(labels: dict[str, object]) -> None:
        labels["launches"][0]["metrics"][1]["repeat_metadata"]["dispersion"][
            "scalar"
        ] = 0.875

    labels_path, authority_path = _write_label_bundle(
        tmp_path, monkeypatch, set_exact_threshold
    )
    row = _row(
        _score_with_labels(labels_path, authority_path),
        "traffic_gmem_ld_sectors",
    )

    assert row["within_label_noise"]["evaluated_count"] == 1
    assert row["within_label_noise"]["within_count"] == 1


def test_unavailable_dispersion_is_noise_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def remove_dispersion(labels: dict[str, object]) -> None:
        labels["launches"][0]["metrics"][0]["repeat_metadata"]["dispersion"] = {
            "status": "unavailable",
            "rule": "relative_tolerance",
            "reason": "source_dispersion_null",
        }

    labels_path, authority_path = _write_label_bundle(
        tmp_path, monkeypatch, remove_dispersion
    )
    row = _row(
        _score_with_labels(labels_path, authority_path),
        "smsp_inst_executed_sum",
    )

    assert row["within_label_noise"] == {
        "comparison_semantics": (
            "absolute_error_native_unit_lte_declared_dispersion_threshold"
        ),
        "evaluated_count": 0,
        "within_count": 0,
        "not_applicable_count": 1,
        "not_applicable_reasons": ["dispersion_unavailable"],
    }


def test_label_unit_drift_fails_strict_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def change_unit(labels: dict[str, object]) -> None:
        labels["launches"][0]["metrics"][0]["unit"] = "cycles"

    labels_path, authority_path = _write_label_bundle(
        tmp_path, monkeypatch, change_unit
    )

    with pytest.raises(ContractValidationError, match="SCHEMA_VALIDATION_FAILED"):
        _score_with_labels(labels_path, authority_path)


def test_label_seal_drift_survives_file_binding_but_fails_contract_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalidate_seal(labels: dict[str, object]) -> None:
        labels["matched_labels_sha256"] = "0" * 64

    labels_path, authority_path = _write_label_bundle(
        tmp_path,
        monkeypatch,
        invalidate_seal,
        reseal_labels=False,
    )

    with pytest.raises(ContractValidationError, match="CANONICAL_SEAL_MISMATCH"):
        _score_with_labels(labels_path, authority_path)


def test_duplicate_result_member_fails_before_scoring(tmp_path: Path) -> None:
    material = RESULT_PATH.read_text(encoding="utf-8")
    duplicate = material.replace(
        '{"aggregate":',
        '{"schema_version":"sim_result.v2","aggregate":',
        1,
    )
    path = tmp_path / "duplicate-result.json"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ContractValidationError, match="DUPLICATE_JSON_KEY"):
        load_result_snapshot(path)


def test_result_seal_drift_fails_before_candidate_enumeration(tmp_path: Path) -> None:
    result = deepcopy(_load(RESULT_PATH))
    result["result_sha256"] = "0" * 64
    path = tmp_path / "bad-seal-result.json"
    path.write_bytes(_encode(result))

    with pytest.raises(ContractValidationError, match="CANONICAL_SEAL_MISMATCH"):
        load_result_snapshot(path)


def test_evaluator_identity_substitution_fails_code_pin(tmp_path: Path) -> None:
    changed = tmp_path / "evaluator-identity.json"
    changed.write_bytes(EVALUATOR_IDENTITY_PATH.read_bytes() + b" ")

    with pytest.raises(
        ContractValidationError,
        match="EVALUATOR_BUILD_IDENTITY_FILE_HASH_MISMATCH",
    ):
        score_synthetic_fixture(
            RESULT_PATH,
            LABELS_PATH,
            LABEL_AUTHORITY_PATH,
            changed,
            fixture_id=FIXTURE_ID,
        )
