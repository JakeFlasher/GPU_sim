from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import admit_cutlass_successor_capture as admission
from tools.local_labels import build_cutlass_successor_m6_determinism_schedule as schedule
from tools.local_labels import m6_determinism_campaign as m6
from tools.local_labels.test_cutlass_successor_capture_admission import (
    MATRIX,
    _REPORT_CSV,
    _all_pairs,
    _receipt,
)

GENERATED_AT = "2026-07-23T01:00:00Z"
FROZEN_MATRIX = Path("data/local_labels/campaign/sm86_reference_capture_matrix.json")
FROZEN_MATRIX_SHA256 = "af29d72e955cf42e281036190f2d55d72af0af92a632c78c8d159073a072fb0e"


@pytest.fixture(autouse=True)
def _mock_ncu_report_import(monkeypatch):
    monkeypatch.setattr(
        admission,
        "_import_ncu_report_csv",
        lambda path: (_REPORT_CSV, None),
    )


def _build(root: Path, seed: int = 42) -> dict:
    return schedule.build_schedule(
        root,
        seed=seed,
        generated_at_utc=GENERATED_AT,
    )


def _reseal(document: dict) -> None:
    document["schedule_sha256"] = m6.canonical_sha256(
        {key: value for key, value in document.items() if key != "schedule_sha256"}
    )


def test_schedule_schema_is_closed():
    schema_path = schedule.ROOT / schedule.SCHEMA_REL
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)


def test_absent_receipts_refuse_without_output(tmp_path: Path, capsys):
    output = tmp_path.parent / f"{tmp_path.name}-schedule.json"
    result = schedule.main(
        [
            "--receipt-root",
            str(tmp_path),
            "--output",
            str(output),
            "--seed",
            "42",
            "--generated-at-utc",
            GENERATED_AT,
        ]
    )
    assert result == 1
    assert "SUCCESSOR_M6_CAPTURE_NOT_READY" in capsys.readouterr().err
    assert not output.exists()


def test_partial_receipts_refuse(tmp_path: Path):
    row = MATRIX["rows"][0]
    _receipt(tmp_path, row, "ncu", 0)
    _receipt(tmp_path, row, "nvbit", 0)
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_CAPTURE_NOT_READY"):
        _build(tmp_path)


def test_all_pairs_build_exact_deterministic_schedule(tmp_path: Path, monkeypatch):
    _all_pairs(tmp_path)
    monkeypatch.setattr(
        m6,
        "_build_slots",
        lambda *args, **kwargs: pytest.fail("canonical slot builder must not be used"),
    )
    first = _build(tmp_path)
    second = _build(tmp_path)
    assert first == second
    assert first["counts"] == {
        "cells": 3,
        "configs": 6,
        "tier1_slots": 210,
        "tier2_candidate_slots": 30,
        "maximum_slots": 240,
        "tier1_fresh_processes": 210,
        "maximum_fresh_processes": 240,
    }
    assert [slot["ordinal"] for slot in first["slots"] + first["tier2_candidates"]] == list(
        range(240)
    )
    assert all(
        slot["process_isolation"] == "fresh_process"
        for slot in first["slots"] + first["tier2_candidates"]
    )
    assert first["execution_dispatch"] == {
        "status": "required_unimplemented",
        "typed_reason": schedule.EXECUTION_BLOCK_REASON,
        "schedule_is_executable": False,
        "tier1_dispatch_allowed": False,
        "tier2_activation_allowed": False,
        "tier2_candidates_are_contract_only": True,
    }
    assert first["claims"] == {
        "canonical_v1_impact": "none_additive_supplement_only",
        "canonical_m6_completed": False,
        "m6_done_created": False,
        "m7_handoff_ready": False,
        "publication_claim": False,
        "hardware_values_are_targets_only": True,
    }

    for cell in first["cells"]:
        tier1 = Counter(
            slot["config_id"]
            for slot in first["slots"]
            if slot["cell_id"] == cell["cell_id"]
        )
        assert tier1[schedule.REFERENCE_CONFIG] == 20
        assert sorted(
            count
            for key, count in tier1.items()
            if key != schedule.REFERENCE_CONFIG
        ) == [
            10,
            10,
            10,
            10,
            10,
        ]
        assert sum(
            slot["cell_id"] == cell["cell_id"] for slot in first["tier2_candidates"]
        ) == 10


def test_seed_changes_order_not_membership(tmp_path: Path):
    _all_pairs(tmp_path)
    first = _build(tmp_path, seed=42)
    second = _build(tmp_path, seed=43)
    assert first["schedule_sha256"] != second["schedule_sha256"]
    assert first["slots_sha256"] != second["slots_sha256"]
    first_members = Counter(
        (
            slot["tier"],
            slot["sweep_index"],
            slot["cell_id"],
            slot["config_id"],
        )
        for slot in first["slots"] + first["tier2_candidates"]
    )
    second_members = Counter(
        (
            slot["tier"],
            slot["sweep_index"],
            slot["cell_id"],
            slot["config_id"],
        )
        for slot in second["slots"] + second["tier2_candidates"]
    )
    assert first_members == second_members


def test_seed_boundaries_are_fail_closed(tmp_path: Path):
    _all_pairs(tmp_path)
    for invalid in (True, -1, 9007199254740992):
        with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_SEED"):
            _build(tmp_path, seed=invalid)
    assert _build(tmp_path, seed=0)["seed"] == 0
    assert _build(tmp_path, seed=9007199254740991)["seed"] == 9007199254740991


def test_slot_mutation_cannot_be_resealed(tmp_path: Path):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    mutated = copy.deepcopy(document)
    mutated["slots"][0]["cell_id"] = mutated["cells"][1]["cell_id"]
    mutated["slots_sha256"] = m6.canonical_sha256(mutated["slots"])
    _reseal(mutated)
    with pytest.raises(
        schedule.ScheduleError,
        match="SUCCESSOR_M6_SLOT_(DUPLICATE|DERIVATION)",
    ):
        schedule.validate_schedule(mutated)


def test_semantic_duplicates_cannot_be_resealed(tmp_path: Path):
    _all_pairs(tmp_path)
    document = _build(tmp_path)

    duplicate_cell = copy.deepcopy(document)
    first_cell = duplicate_cell["cells"][0]
    second_cell = duplicate_cell["cells"][1]
    for key in ("cell_id", "matrix_key", "variant_id"):
        second_cell[key] = first_cell[key]
    second_cell["cell_identity_sha256"] = m6.canonical_sha256(
        {
            key: value
            for key, value in second_cell.items()
            if key != "cell_identity_sha256"
        }
    )
    duplicate_cell["cell_identity_sha256"] = m6.canonical_sha256(
        duplicate_cell["cells"]
    )
    duplicate_cell["schedule_id"] = schedule._schedule_id(
        duplicate_cell["source_identity_sha256"],
        duplicate_cell["config_identity_sha256"],
        duplicate_cell["cell_identity_sha256"],
        duplicate_cell["seed"],
    )
    _reseal(duplicate_cell)
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_CELL_DUPLICATE"):
        schedule.validate_schedule(duplicate_cell)

    duplicate_slot = copy.deepcopy(document)
    first_slot = duplicate_slot["slots"][0]
    second_slot = duplicate_slot["slots"][1]
    for key in (
        "tier",
        "sweep_index",
        "cell_id",
        "cell_identity_sha256",
        "config_id",
        "config_sha256",
    ):
        second_slot[key] = first_slot[key]
    second_slot["slot_sha256"] = m6.canonical_sha256(
        {
            key: value
            for key, value in second_slot.items()
            if key != "slot_sha256"
        }
    )
    duplicate_slot["slots_sha256"] = m6.canonical_sha256(duplicate_slot["slots"])
    _reseal(duplicate_slot)
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_SLOT_DUPLICATE"):
        schedule.validate_schedule(duplicate_slot)


def test_source_and_count_mutations_are_rejected(tmp_path: Path):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    source_mutation = copy.deepcopy(document)
    source_mutation["source_identity"]["pinned_bundle_commit"] = "0" * 40
    source_mutation["source_identity_sha256"] = m6.canonical_sha256(
        source_mutation["source_identity"]
    )
    _reseal(source_mutation)
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_SCHEMA"):
        schedule.validate_schedule(source_mutation)

    count_mutation = copy.deepcopy(document)
    count_mutation["counts"]["maximum_slots"] = 239
    _reseal(count_mutation)
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_SCHEMA"):
        schedule.validate_schedule(count_mutation)


def test_source_file_drift_cannot_be_resealed(tmp_path: Path):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    mutated = copy.deepcopy(document)
    mutated["source_identity"]["producer_files"]["schedule_builder"]["sha256"] = "0" * 64
    mutated["source_identity_sha256"] = m6.canonical_sha256(mutated["source_identity"])
    mutated["schedule_id"] = schedule._schedule_id(
        mutated["source_identity_sha256"],
        mutated["config_identity_sha256"],
        mutated["cell_identity_sha256"],
        mutated["seed"],
    )
    _reseal(mutated)
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_SOURCE_DRIFT"):
        schedule.validate_schedule(mutated)


def test_validation_keeps_pinned_producer_commit_after_head_advances(
    tmp_path: Path, monkeypatch
):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    producer_commit = document["source_identity"]["producer_commit"]
    monkeypatch.setattr(schedule, "_git_head", lambda: "f" * 40)
    schedule.validate_schedule(document)
    assert document["source_identity"]["producer_commit"] == producer_commit


def test_write_validates_exact_staged_snapshot(tmp_path: Path, monkeypatch):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    expected = copy.deepcopy(document)
    output_root = tmp_path.parent / f"{tmp_path.name}-snapshot-output"
    output_root.mkdir()
    output = output_root / "schedule.json"
    real_stage = schedule._stage_payload

    def mutate_after_staging(path, payload):
        stage = real_stage(path, payload)
        document["seed"] = -1
        return stage

    monkeypatch.setattr(schedule, "_stage_payload", mutate_after_staging)
    monkeypatch.setattr(schedule, "_require_publication_source", lambda snapshot: None)
    schedule.write_schedule(document, output, tmp_path)
    assert json.loads(output.read_text()) == expected


def test_write_failure_leaves_no_partial_artifact(tmp_path: Path, monkeypatch):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    output_root = tmp_path.parent / f"{tmp_path.name}-partial-output"
    output_root.mkdir()
    output = output_root / "schedule.json"
    real_write = schedule.os.write
    calls = 0

    def fail_after_partial_write(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, payload[:8])
        raise OSError("injected staged write failure")

    monkeypatch.setattr(schedule.os, "write", fail_after_partial_write)
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_OUTPUT"):
        schedule.write_schedule(document, output, tmp_path)
    assert not output.exists()
    assert not list(output_root.glob(".*.tmp"))


def test_stage_cleanup_failure_does_not_report_published_output_as_refused(
    tmp_path: Path, monkeypatch
):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    output_root = tmp_path.parent / f"{tmp_path.name}-cleanup-output"
    output_root.mkdir()
    output = output_root / "schedule.json"
    real_unlink = Path.unlink

    def fail_stage_unlink(path, *args, **kwargs):
        if path.name.endswith(".tmp"):
            raise OSError("injected stage cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(schedule, "_require_publication_source", lambda snapshot: None)
    monkeypatch.setattr(Path, "unlink", fail_stage_unlink)
    schedule.write_schedule(document, output, tmp_path)
    assert json.loads(output.read_text()) == document
    for stage in output_root.glob(".*.tmp"):
        real_unlink(stage)


def test_write_refuses_dirty_publication_source(tmp_path: Path, monkeypatch):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    output_root = tmp_path.parent / f"{tmp_path.name}-dirty-output"
    output_root.mkdir()
    output = output_root / "schedule.json"
    monkeypatch.setattr(
        schedule,
        "_git_blob_bytes",
        lambda commit, relative: b"different committed bytes",
    )
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_SOURCE_DIRTY"):
        schedule.write_schedule(document, output, tmp_path)
    assert not output.exists()
    assert not list(output_root.glob(".*.tmp"))


def test_write_refuses_overlap_and_overwrite(tmp_path: Path, monkeypatch):
    _all_pairs(tmp_path)
    document = _build(tmp_path)
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_OUTPUT_OVERLAP"):
        schedule.write_schedule(document, tmp_path / "schedule.json", tmp_path)

    output_root = tmp_path.parent / f"{tmp_path.name}-output"
    output_root.mkdir()
    output = output_root / "schedule.json"
    monkeypatch.setattr(schedule, "_require_publication_source", lambda document: None)
    schedule.write_schedule(document, output, tmp_path)
    original = output.read_bytes()
    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_NO_OVERWRITE"):
        schedule.write_schedule(document, output, tmp_path)
    assert output.read_bytes() == original


def test_frozen_matrix_remains_unchanged_and_no_synthetic_cli():
    path = schedule.ROOT / FROZEN_MATRIX
    assert admission.sha256_bytes(path.read_bytes()) == FROZEN_MATRIX_SHA256
    source = (schedule.ROOT / schedule.BUILDER_REL).read_text()
    assert "allow-synthetic" not in source
    assert "rehash_sources" not in source
