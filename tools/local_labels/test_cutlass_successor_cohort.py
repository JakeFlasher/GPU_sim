from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.local_labels import build_cutlass_successor_cohort as cohort


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATHS = (
    ROOT / cohort.VALIDATION_SCHEMA_REL,
    ROOT / cohort.BUDGETS_SCHEMA_REL,
    ROOT / cohort.ROSTER_SCHEMA_REL,
    ROOT / cohort.MATRIX_SCHEMA_REL,
)


def _load_if_present(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f"generated successor evidence is not present yet: {path}")
    return json.loads(path.read_text())


def test_successor_schema_family_is_draft_2020_12_and_closed():
    for path in SCHEMA_PATHS:
        document = json.loads(path.read_text())
        Draft202012Validator.check_schema(document)
        assert document["$schema"].endswith("draft/2020-12/schema")
        assert document["type"] == "object"
        assert document["additionalProperties"] is False


def test_successor_case_set_is_exact_and_namespaced():
    assert len(cohort.CASES) == 3
    assert [case.matrix_key for case in cohort.CASES] == [
        "cutlass_successor:vendor_gemm_cublas:vendor_gemm_cutlass",
        "cutlass_successor:vendor_sgemm_cublas:vendor_sgemm_cutlass",
        "cutlass_successor:vendor_sgemm_strided_batched_cublas:vendor_sgemm_strided_batched_cutlass",
    ]
    assert {case.slug for case in cohort.CASES}.isdisjoint(
        {case.base_slug for case in cohort.CASES}
    )
    assert all(case.slug.startswith("vendor_") and case.slug.endswith("_cutlass") for case in cohort.CASES)


def test_frozen_reference_rechecks_all_base_and_input_pins():
    frozen = cohort._frozen_reference()
    assert frozen["matrix_file_sha256"] == cohort.FROZEN_MATRIX_SHA256
    assert frozen["matrix_seal_sha256"] == cohort.FROZEN_MATRIX_SEAL
    assert frozen["authority_sha256"] == cohort.FROZEN_AUTHORITY_SHA256
    assert frozen["metric_registry_sha256"] == cohort.FROZEN_REGISTRY_SHA256
    assert frozen["metric_identity_sha256"] == cohort.FROZEN_METRIC_IDENTITY_SHA256
    assert frozen["trace_budgets_sha256"] == cohort.FROZEN_TRACE_BUDGETS_SHA256
    assert frozen["source_alias_sha256"] == cohort.FROZEN_ALIAS_SHA256
    assert frozen["base_rows"] == {
        slug: {"authority_cell_sha256": digest, "disposition": "runnable"}
        for slug, digest in sorted(cohort.FROZEN_BASE_CELL_SHA256.items())
    }


def test_source_input_set_closes_successor_contract_files():
    # The current contract is intentionally checked against the current HEAD
    # only after its narrow contract commit; before that commit this is a
    # structural assertion over the closed path table.
    expected = {
        cohort.GENERATOR_REL.as_posix(),
        cohort.GUARD_REL.as_posix(),
        cohort.VALIDATION_SCHEMA_REL.as_posix(),
        cohort.BUDGETS_SCHEMA_REL.as_posix(),
        cohort.ROSTER_SCHEMA_REL.as_posix(),
        cohort.MATRIX_SCHEMA_REL.as_posix(),
        "workloads/simbench_bw/schemas/cutlass_successor_build_witness.schema.json",
        "workloads/simbench_bw/schemas/cutlass_sgemm_successor_build_witness.schema.json",
        "workloads/simbench_bw/schemas/cutlass_strided_batched_successor_build_witness.schema.json",
    }
    # The implementation uses a closed literal table; inspect the source
    # rather than accepting an ambient directory walk.
    source = (ROOT / cohort.GENERATOR_REL).read_text()
    for path in expected:
        assert path in source
    assert "rglob" not in source
    assert '"cat-file", "blob"' in source


def test_generated_bundle_is_sealed_and_capture_pending_when_present():
    validation = _load_if_present(ROOT / cohort.VALIDATION_REL)
    budgets = _load_if_present(ROOT / cohort.BUDGETS_REL)
    roster = _load_if_present(ROOT / cohort.ROSTER_REL)
    matrix = _load_if_present(ROOT / cohort.MATRIX_REL)
    cohort.validate_bundle(validation, budgets, roster, matrix)
    assert matrix["dispatch"]["status"] == "not_admitted"
    assert matrix["dispatch"]["m6_schedule_created"] is False
    assert matrix["counts"] == {
        "total": 3,
        "capture_pending": 3,
        "hardware_capture_runnable": 0,
        "excluded": 0,
    }
    assert all(row["capture_admission"]["status"] == "pending" for row in matrix["rows"])
    assert matrix["chunking"]["chunks"] == []
    assert matrix["chunks"] == []


def test_resealed_pending_row_does_not_change_admission_state_when_present():
    matrix = _load_if_present(ROOT / cohort.MATRIX_REL)
    mutated = copy.deepcopy(matrix)
    mutated["rows"][0]["capture_admission"]["status"] = "ready"
    mutated["matrix_seal_sha256"] = cohort._canonical_sha(
        {key: value for key, value in mutated.items() if key != "matrix_seal_sha256"}
    )
    with pytest.raises(cohort.CohortError):
        cohort.validate_bundle(
            _load_if_present(ROOT / cohort.VALIDATION_REL),
            _load_if_present(ROOT / cohort.BUDGETS_REL),
            _load_if_present(ROOT / cohort.ROSTER_REL),
            mutated,
        )
