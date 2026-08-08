from __future__ import annotations

import copy
import json
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import admit_cutlass_successor_capture as admission
from tools.local_labels import build_cutlass_successor_m6_determinism_schedule as schedule
from tools.local_labels import m6_determinism_campaign as m6
from tools.local_labels.test_cutlass_successor_capture_admission import (
    MATRIX,
    _REPORT_IDENTITY,
    _all_pairs,
    _receipt,
    _row_for_report_path,
    _row_report_csv,
)

GENERATED_AT = "2026-07-23T01:00:00Z"
FROZEN_MATRIX = Path("data/local_labels/campaign/sm86_reference_capture_matrix.json")
FROZEN_MATRIX_SHA256 = "af29d72e955cf42e281036190f2d55d72af0af92a632c78c8d159073a072fb0e"


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Give production-gated receipt tests a fresh external root."""
    with tempfile.TemporaryDirectory(
        prefix="successor-schedule-pytest-", dir="/tmp",
    ) as temporary_root:
        root = Path(temporary_root) / "receipts"
        root.mkdir()
        yield root


@pytest.fixture(autouse=True)
def _mock_ncu_report_import(monkeypatch):
    monkeypatch.setattr(
        admission,
        "_import_ncu_report_csv",
        lambda _identity, path: (
            _row_report_csv(_row_for_report_path(path)),
            admission.sha256_bytes(path.read_bytes()),
            None,
        ),
    )
    monkeypatch.setattr(
        admission,
        "_sass_kernel_names",
        lambda _row: {_REPORT_IDENTITY["Kernel Name"]},
    )


def _build(root: Path, seed: int = 42) -> dict:
    return schedule.build_schedule(
        root,
        seed=seed,
        generated_at_utc=GENERATED_AT,
    )


def _schema_authority(version: int) -> dict:
    relative = admission.RECEIPT_SCHEMA_PATHS_BY_VERSION[version]
    path = schedule.ROOT / relative
    return {
        "schema_version": version,
        "schema_id": admission.RECEIPT_SCHEMA_IDS_BY_VERSION[version],
        "path": relative.as_posix(),
        "sha256": admission.RECEIPT_SCHEMA_SHA256_BY_VERSION[version],
        "size_bytes": path.stat().st_size,
    }


def _reseal(document: dict) -> None:
    document["schedule_sha256"] = m6.canonical_sha256(
        {key: value for key, value in document.items() if key != "schedule_sha256"}
    )


def test_schedule_schema_registry_preserves_v1_and_dispatches_active_v2():
    assert set(schedule.SCHEMA_PATHS_BY_VERSION) == {1, 2}
    assert schedule.ACTIVE_SCHEMA_VERSION == 2
    assert schedule.SCHEMA_REL == schedule.SCHEMA_PATHS_BY_VERSION[2]
    assert schedule.SCHEMA_SHA256_BY_VERSION[1] == (
        "52af5bfe5a6e88393546ad5f5718b3a49dc93e126f070287c7c49b75d28547ac"
    )
    for version, relative in schedule.SCHEMA_PATHS_BY_VERSION.items():
        schema_path = schedule.ROOT / relative
        payload = schema_path.read_bytes()
        assert admission.sha256_bytes(payload) == (
            schedule.SCHEMA_SHA256_BY_VERSION[version]
        )
        schema_document = json.loads(payload)
        assert schema_document["$id"] == schedule.SCHEMA_IDS_BY_VERSION[version]
        assert schema_document["properties"]["schema_version"]["const"] == version
        jsonschema.Draft202012Validator.check_schema(schema_document)

        def visit(value):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    assert value.get("additionalProperties") is False
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(schema_document)


def test_historical_v1_schedule_shape_dispatches_read_only(tmp_path: Path):
    _all_pairs(tmp_path)
    historical = copy.deepcopy(_build(tmp_path))
    historical["schema_version"] = 1
    source = historical["source_identity"]
    source["admission_overlay"]["overlay_id"] = (
        admission.OVERLAY_IDS_BY_VERSION[1]
    )
    source.pop("receipt_schema_authorities")
    source.pop("nvbit_tools")
    producer_files = source["producer_files"]
    producer_files["receipt_schema"] = schedule._file_ref(
        admission.RECEIPT_SCHEMA_PATHS_BY_VERSION[2],
        role="historical receipt schema",
    )
    producer_files["nvbit_tracer"] = schedule._file_ref(
        schedule.TRACER_TOOL_REL,
        role="historical NVBit tracer",
    )
    for cell in historical["cells"]:
        cell["ncu_receipt"].pop("schema_authority")
        cell["nvbit_receipt"].pop("schema_authority")
        cell["nvbit_receipt"].pop("nvbit_tool")

    schedule._validate_schema(historical)
    with pytest.raises(
        schedule.ScheduleError,
        match="SUCCESSOR_M6_HISTORICAL_SCHEDULE",
    ):
        schedule.validate_schedule(historical)


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
    assert first["schema_version"] == schedule.ACTIVE_SCHEMA_VERSION == 2
    assert first["source_identity"]["admission_overlay"]["overlay_id"] == (
        "cutlass_successor_capture_admission_sm86_v2"
    )
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


def test_historical_v2_receipts_cannot_build_active_schedule(tmp_path: Path):
    _all_pairs(tmp_path, schema_version=2)

    with pytest.raises(
        admission.AdmissionError,
        match="schema_version 2 is historical-read-only",
    ):
        _build(tmp_path)


def test_historical_v2_receipts_cannot_publish_active_schedule(
    tmp_path: Path,
    capsys,
):
    _all_pairs(tmp_path, schema_version=2)
    output = tmp_path.parent / f"{tmp_path.name}-historical-v2-schedule.json"

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
    assert "schema_version 2 is historical-read-only" in capsys.readouterr().err
    assert not output.exists()


def test_schedule_consumer_refuses_historical_admission_overlay(
    tmp_path: Path,
):
    _all_pairs(tmp_path)
    overlay = admission.build_overlay(
        tmp_path,
        generated_at_utc=GENERATED_AT,
    )
    overlay["schema_version"] = 1
    overlay["overlay_id"] = admission.OVERLAY_IDS_BY_VERSION[1]

    with pytest.raises(
        schedule.ScheduleError,
        match="SUCCESSOR_M6_ADMISSION_SCHEMA",
    ):
        schedule._require_capture_ready(overlay, MATRIX)


def test_schedule_consumer_refuses_synthetic_historical_receipt_overlay(
    tmp_path: Path,
):
    _all_pairs(tmp_path)
    overlay = admission.build_overlay(
        tmp_path,
        generated_at_utc=GENERATED_AT,
    )
    overlay["rows"][0]["ncu_receipt"]["schema_authority"][
        "schema_version"
    ] = 2

    with pytest.raises(
        schedule.ScheduleError,
        match="SUCCESSOR_M6_HISTORICAL_RECEIPT",
    ):
        schedule._dispatched_receipt_authorities(overlay)


def test_v3_schedule_validates_rich_nvbit_identity_then_projects_phase_tools(
    tmp_path: Path,
):
    _all_pairs(tmp_path, schema_version=3)
    overlay = admission.build_overlay(
        tmp_path,
        generated_at_utc=GENERATED_AT,
    )
    for row in overlay["rows"]:
        rich_tool = row["nvbit_receipt"]["nvbit_tool"]
        assert rich_tool["identity_kind"] == (
            "retained_reviewed_external_build_bundle.v1"
        )
        assert set(rich_tool) == {
            "identity_kind",
            "path",
            "sha256",
            "size_bytes",
            "runtime_manifest",
            "external_build_authority",
            "source_manifest",
            "build_receipt",
            "artifact_identity",
        }
        assert schedule._project_nvbit_tool_identity(
            row,
            rich_tool,
            receipt_schema_version=3,
        ) == {
            "identity_kind": "retained_external_phase_tool",
            "path": rich_tool["path"],
            "sha256": rich_tool["sha256"],
            "size_bytes": rich_tool["size_bytes"],
        }

    document = _build(tmp_path)
    source = document["source_identity"]
    assert source["receipt_schema_authorities"] == [_schema_authority(3)]
    assert len(source["nvbit_tools"]) == len(MATRIX["rows"])
    repository_tracer_sha256 = admission.sha256_bytes(
        (schedule.ROOT / schedule.TRACER_TOOL_REL).read_bytes()
    )
    expected_paths = {
        f"capture_tools/{row['variant_id']}.nvbit_tracer_tool.so"
        for row in MATRIX["rows"]
    }
    assert {tool["path"] for tool in source["nvbit_tools"]} == expected_paths
    assert all(
        set(tool) == {"identity_kind", "path", "sha256", "size_bytes"}
        and tool["identity_kind"] == "retained_external_phase_tool"
        and tool["sha256"] != repository_tracer_sha256
        for tool in source["nvbit_tools"]
    )
    tools_by_path = {tool["path"]: tool for tool in source["nvbit_tools"]}
    for cell in document["cells"]:
        assert cell["ncu_receipt"]["schema_authority"] == _schema_authority(3)
        assert cell["nvbit_receipt"]["schema_authority"] == _schema_authority(3)
        projected_tool = cell["nvbit_receipt"]["nvbit_tool"]
        assert projected_tool == tools_by_path[projected_tool["path"]]


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    (
        (("identity_kind",), "retained_external_phase_tool"),
        (("path",), "capture_tools/wrong.nvbit_tracer_tool.so"),
        (("sha256",), "0" * 64),
        (("size_bytes",), 0),
        (
            ("external_build_authority", "authority_kind"),
            "unreviewed_candidate_sha256.v1",
        ),
        (("external_build_authority", "expected_tracer_sha256"), "0" * 64),
        (("source_manifest", "sha256"), "0" * 64),
        (("runtime_manifest", "size_bytes"), 0),
    ),
)
def test_schedule_consumer_refuses_mutated_rich_nvbit_tool_identity(
    tmp_path: Path,
    field_path: tuple[str, ...],
    replacement,
):
    _all_pairs(tmp_path)
    overlay = admission.build_overlay(
        tmp_path,
        generated_at_utc=GENERATED_AT,
    )
    tool = overlay["rows"][0]["nvbit_receipt"]["nvbit_tool"]
    target = tool
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement

    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_NVBIT_TOOL"):
        schedule._dispatched_receipt_authorities(overlay)


@pytest.mark.parametrize(
    "missing_field",
    (
        "runtime_manifest",
        "external_build_authority",
        "source_manifest",
        "build_receipt",
        "artifact_identity",
    ),
)
def test_schedule_consumer_requires_complete_rich_nvbit_authority(
    tmp_path: Path,
    missing_field: str,
):
    _all_pairs(tmp_path)
    overlay = admission.build_overlay(
        tmp_path,
        generated_at_utc=GENERATED_AT,
    )
    overlay["rows"][0]["nvbit_receipt"]["nvbit_tool"].pop(missing_field)

    with pytest.raises(schedule.ScheduleError, match="SUCCESSOR_M6_NVBIT_TOOL"):
        schedule._dispatched_receipt_authorities(overlay)


def test_schedule_consumer_refuses_repository_tracer_substitution(tmp_path: Path):
    _all_pairs(tmp_path)
    overlay = admission.build_overlay(
        tmp_path,
        generated_at_utc=GENERATED_AT,
    )
    tool = overlay["rows"][0]["nvbit_receipt"]["nvbit_tool"]
    repository_tracer_sha256 = admission.NVBIT_TRACER_SHA256_BY_RECEIPT_VERSION[2]
    tool["sha256"] = repository_tracer_sha256
    tool["external_build_authority"][
        "expected_tracer_sha256"
    ] = repository_tracer_sha256

    with pytest.raises(
        schedule.ScheduleError,
        match="repository tracer substitution is forbidden",
    ):
        schedule._dispatched_receipt_authorities(overlay)


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
