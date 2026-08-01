from __future__ import annotations

from pathlib import Path

import pytest

from eval.contracts._common import (
    AuthorityBlockerCode,
    EvaluatorAuthorityUnavailable,
)
from eval.contracts.validation import ContractValidationError, METRIC_KEYS
from eval.labels import (
    known_synthetic_authority_pin,
    load_production_matched_labels,
    load_synthetic_matched_labels,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures/evaluator_mvp"
LABELS_PATH = FIXTURE_DIR / "matched_labels.v1.json"
AUTHORITY_PATH = FIXTURE_DIR / "synthetic_matched_label_authority.v1.json"
FIXTURE_ID = "evaluator-mvp-canary-v1"


def test_loads_code_pinned_synthetic_labels_and_preserves_namespaces() -> None:
    labels = load_synthetic_matched_labels(
        LABELS_PATH,
        AUTHORITY_PATH,
        authority_pin=known_synthetic_authority_pin(FIXTURE_ID),
    )

    assert labels.synthetic_fixture_id == FIXTURE_ID
    assert labels.binary_architecture == "sm_86"
    assert labels.gpu_sku == "rtx_3090_ti"
    assert len(labels.launches) == 1

    launch = labels.launches[0]
    assert launch.identity.cohort.workload_id == "fixture/vector-add:v1"
    assert launch.identity.cohort.matched_execution_id == "matched-execution-0001"
    assert launch.identity.cohort.process_launch_ordinal == 0
    assert (
        launch.identity.replay.kernel_mangled_name
        == "_Z10vector_addPKfS0_Pfi"
    )
    assert launch.identity.replay.binary_architecture == "sm_86"
    assert tuple(metric.key for metric in launch.metrics) == METRIC_KEYS
    assert launch.metric_by_key("exec_cycles_gpc_max").value == 1000.0


def test_production_loader_refuses_before_payload_access() -> None:
    with pytest.raises(EvaluatorAuthorityUnavailable) as caught:
        load_production_matched_labels("/path/that/must/not/be/opened")

    assert caught.value.code is AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE


def test_unknown_synthetic_authority_is_typed_unavailable() -> None:
    with pytest.raises(EvaluatorAuthorityUnavailable) as caught:
        known_synthetic_authority_pin("not-code-pinned")

    assert caught.value.code is AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE


def test_authority_bytes_must_match_independent_code_pin(tmp_path: Path) -> None:
    changed = tmp_path / "authority.json"
    changed.write_bytes(AUTHORITY_PATH.read_bytes() + b" ")

    with pytest.raises(
        ContractValidationError, match="FIXTURE_AUTHORITY_FILE_HASH_MISMATCH"
    ):
        load_synthetic_matched_labels(
            LABELS_PATH,
            changed,
            authority_pin=known_synthetic_authority_pin(FIXTURE_ID),
        )


def test_label_bytes_must_match_sealed_authority(tmp_path: Path) -> None:
    changed = tmp_path / "labels.json"
    changed.write_bytes(LABELS_PATH.read_bytes() + b" ")

    with pytest.raises(ContractValidationError, match="FIXTURE_LABEL_FILE_HASH_MISMATCH"):
        load_synthetic_matched_labels(
            changed,
            AUTHORITY_PATH,
            authority_pin=known_synthetic_authority_pin(FIXTURE_ID),
        )
