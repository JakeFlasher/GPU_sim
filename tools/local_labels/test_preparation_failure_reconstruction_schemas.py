from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "workloads" / "simbench_bw" / "schemas"
SCHEMA_NAMES = (
    "runner_refusal_reconstruction.schema.json",
    "phase_reconstruction.schema.json",
    "gpu_process_observation_late.schema.json",
    "preparation_failure_no_gpu_attestation_v2.schema.json",
    "preparation_failure_ledger_entry.schema.json",
    "legacy_comparator_reproduction.schema.json",
)

SHA = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
HEAD = "d" * 40
CELL = "sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation"
CHUNK = "sol_execbench-000-bootstrap"
AMENDMENT = "ANNEX_C_AMEND_R28_PREP070_RECON_V2"
WITHDRAWN_AMENDMENT = "ANNEX_C_AMEND_R28_PREP070_RECON_V1"
LEGACY_AUTHORITY = (
    "deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes"
)
WITHDRAWN_AUTHORITY = "deterministic_v1_comparator_over_sealed_bytes"
STAMP = "2026-07-11T12:34:56.123456Z"


def _schema(name: str) -> dict:
    return json.loads((SCHEMA_ROOT / name).read_text())


def _validate(name: str, instance: object) -> None:
    validator = jsonschema.Draft202012Validator(
        _schema(name), format_checker=jsonschema.FormatChecker()
    )
    validator.validate(instance)


def _rejects(name: str, instance: object) -> None:
    with pytest.raises(jsonschema.ValidationError):
        _validate(name, instance)


def _reproduction() -> dict:
    source = (
        ".humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/"
        "scratch/source_definition.json"
    )
    staged = (
        ".humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/"
        "scratch/staged_definition.json"
    )
    emit = (
        ".humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/"
        "legacy_comparator_reproduction.json"
    )
    source_sha = "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa"
    staged_sha = "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387"
    return {
        "record_type": "legacy_comparator_reproduction",
        "amendment_id": AMENDMENT,
        "rule_id": "SOL_DEF_LEGACY_COMPARE_9B8E0E40",
        "executed_at": STAMP,
        "no_gpu": True,
        "pure_cpu": True,
        "capture_code_imported": False,
        "tool": {
            "path": "tools/local_labels/legacy_compare_9b8e0e40.py",
            "sha256": SHA,
            "head": HEAD,
        },
        "inputs": {
            "source_scratch_path": source,
            "source_sha256_expected": source_sha,
            "source_sha256_observed": source_sha,
            "staged_scratch_path": staged,
            "staged_sha256_expected": staged_sha,
            "staged_sha256_observed": staged_sha,
            "source_origin": {
                "kind": "external_prepare_manifest_pinned",
                "prepare_manifest_journal_relative_path": (
                    "cells/sol_execbench__problem_070_mamba2_fused_intra_chunk_"
                    "diagonal_computation/sol_stage/capture_prepare_manifest.json"
                ),
                "prepare_manifest_sha256": (
                    "d9c2fd5e55a9edf860ff23c2b2b018e4c63800adb47d26174ff664c0bf20c7e7"
                ),
                "source_path": (
                    "/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/"
                    "tools/sol_execbench/data/benchmark/L1/"
                    "070_mamba2_fused_intra_chunk_diagonal_computation/definition.json"
                ),
            },
            "staged_origin": {
                "kind": "historical_journal_member",
                "journal_relative_path": (
                    "cells/sol_execbench__problem_070_mamba2_fused_intra_chunk_"
                    "diagonal_computation/sol_stage/definition.json"
                ),
            },
        },
        "verdict": {
            "equal": False,
            "differing_top_level_keys": ["op_type"],
            "difference_pointers": ["/op_type"],
            "maps_to_refusal_code": "sol_stage_source_mismatch",
        },
        "python_version": "3.12.11",
        "command_argv": [
            str(ROOT / "tools/local_labels/legacy_compare_9b8e0e40.py"),
            "--source",
            source,
            "--staged",
            staged,
            "--expect-source-sha",
            source_sha,
            "--expect-staged-sha",
            staged_sha,
            "--tool-head",
            HEAD,
            "--emit",
            emit,
        ],
    }


def _runner() -> dict:
    return {
        "record_type": "runner_refusal_reconstruction",
        "amendment_id": AMENDMENT,
        "reconstruction_late": True,
        "contemporaneous": False,
        "reconstructed_at": STAMP,
        "cell": CELL,
        "chunk": CHUNK,
        "original_head": "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a",
        "session": "f10de1cc-612e-43e3-b51f-ce1c9985e281",
        "matrix_file_sha256": "fd3a1868845ceac309e718da6519f4779824b7fd69eddf1bfa0e7e7847d4e297",
        "preflight_sha256": "3810d2b29a06163f41d6aab73375b273e87cf462a7ad22032088802eef85b63a",
        "historical_refusal": {
            "code": "sol_stage_source_mismatch",
            "message": "staged definition differs from pinned source",
            "reason": "STAGE_SEMANTIC_VALIDATION_FAILED",
        },
        "sealed_inputs": {
            "source_definition_sha256": "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa",
            "staged_definition_sha256": "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387",
            "staged_module_sha256": "017ec9c8c977078c141e8d7d4fb58a67501550aa5540c220dbd97a94b8e1ee96",
        },
        "legacy_comparator_identity": {
            "rule_id": "SOL_DEF_LEGACY_COMPARE_9B8E0E40",
            "head": "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a",
            "path": "tools/local_labels/sm86_capture_chunk.py",
            "file_sha256": "1dfbe3ec5ed19d6dff234616141fbd3c601adc88d20709860055380a47ea5447",
            "line_span": [1306, 1317],
            "line_span_sha256": "cbc95368ce58e4f7454e4f39a87a85c04fce2baa83546ca5af08eb38025d124c",
            "function": "normalized_definition",
            "source_excerpt_file": "legacy_comparator_source_excerpt.py.txt",
            "semantics": (
                "For top-level fields hf_id, op_type, custom_inputs_entrypoint: a "
                "PRESENT empty-string value is rewritten to null; missing keys are not "
                "inserted and are never equated with null or empty string; all other "
                "content is compared by strict dict equality; inequality raises "
                "ChunkRefusal sol_stage_source_mismatch."
            ),
        },
        "rederivation": {
            "equal": False,
            "differing_top_level_keys": ["op_type"],
            "difference_pointers": ["/op_type"],
            "source_op_type_state": "missing",
            "staged_op_type_state": "null",
            "maps_to_refusal_code": "sol_stage_source_mismatch",
            "reproduction_record": "legacy_comparator_reproduction.json",
            "reproduction_record_sha256": SHA,
        },
        "authority_basis": ["annex_c_adjudication", LEGACY_AUTHORITY],
        "authority_citations": {
            "annex_c_adjudication_sha256": "b9bc56d06ddbed14ca28df02012077a533c9b846a90f2f08ffdcfc9c6c52b568",
            "corrected_adjudication_sha256": SHA,
            "superseded_adjudication_sha256": SHA_B,
            "supersession_status": "superseded_by_v2",
        },
        "non_certifying_context": {
            "sol_def_equiv_v1": {
                "verdict": "equal",
                "difference_pointers": [],
                "artifact_sha256": "b6dd22ad6e7a87033374262aa8cd6a8a3ea97ac901a5e1c266a3a6d7341cdcd0",
            },
            "sol_def_equiv_v2": {
                "verdict": "equal",
                "audit_sha256": "ea51b0c6da89f5d339cab365ed9bfc69cb642a44dd73dc2fabe9d5dafafcb252",
            },
            "note": "V1/V2 equality is the intended repair semantics and is never a basis for the historical refusal.",
        },
    }


def _phase() -> dict:
    return {
        "record_type": "phase_reconstruction",
        "amendment_id": AMENDMENT,
        "interval_type": "non_certifying_reconstruction",
        "reconstruction_late": True,
        "contemporaneous": False,
        "reconstructed_at": STAMP,
        "cell": CELL,
        "chunk": CHUNK,
        "anchors": [
            {
                "name": "prepare_stdout_mtime",
                "value": "2026-07-10T13:34:52.699087829Z",
                "basis": "filesystem_stat_mtime",
                "evidence_sha256": "ca3249b38c8c5dc3c8d428aed3aa8c650978d629903640befb46517aed7d8cdc",
                "certainty": "exact_observed",
            },
            {
                "name": "journal_start",
                "value": None,
                "basis": "journal_start_sha256",
                "evidence_sha256": "6b6dcce8cb7f4a21385c6e52c693f7c398258877ec702090438ee67bea8faa4f",
                "certainty": "content_sealed_time_unknown",
            },
        ],
        "phases": [
            {
                "phase": "admission",
                "occurred": True,
                "start": None,
                "end": None,
                "certainty": "unknown_interval",
                "evidence_basis": [
                    "admission_run:0ff22c451824e77fa4ec20392866c50a7356b6de4e1e986ba788bd1b5ee1a658"
                ],
            },
            {
                "phase": "prepare_execution",
                "occurred": True,
                "start": None,
                "end": "2026-07-10T13:34:52.699087829Z",
                "certainty": "end_bounded_by_mtime",
                "evidence_basis": [
                    "prepare_stdout_sha256:ca3249b38c8c5dc3c8d428aed3aa8c650978d629903640befb46517aed7d8cdc"
                ],
            },
            {
                "phase": "stage_semantic_validation_refusal",
                "occurred": True,
                "start": None,
                "end": None,
                "certainty": "unknown_interval",
                "evidence_basis": ["annex_c_adjudication", LEGACY_AUTHORITY],
                "refusal_code": "sol_stage_source_mismatch",
                "reason": "STAGE_SEMANTIC_VALIDATION_FAILED",
            },
            {
                "phase": "gpu_execution",
                "occurred": False,
                "start": None,
                "end": None,
                "certainty": "certified_absent",
                "evidence_basis": ["journal_forensic_absence_scan", "annex_c_adjudication"],
            },
            {
                "phase": "finalization",
                "occurred": False,
                "start": None,
                "end": None,
                "certainty": "certified_absent",
                "evidence_basis": ["journal_complete_json_absent"],
            },
        ],
    }


def _late_observation() -> dict:
    return {
        "record_type": "gpu_process_observation_late",
        "amendment_id": AMENDMENT,
        "observed_at": STAMP,
        "certifying": False,
        "late": True,
        "method": "nvidia-smi_readonly_query",
        "command": "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv",
        "raw_output_file": "gpu_process_observation_late.raw.txt",
        "raw_output_sha256": SHA,
        "interpretation": "Late observation only; asserts nothing about the historical 2026-07-10 attempt.",
    }


def _attestation() -> dict:
    return {
        "record_type": "preparation_failure_no_gpu_attestation",
        "record_version": 2,
        "amendment_id": AMENDMENT,
        "cell": CELL,
        "chunk": CHUNK,
        "attestation": "No GPU execution occurred in the historical problem-070 preparation attempt; the attempt terminated at pre-GPU staged-definition semantic validation.",
        "certifying_elements": [
            "annex_c_adjudication",
            LEGACY_AUTHORITY,
            "journal_forensic_absence_scan",
        ],
        "journal_forensic_absence_scan": {
            "forbidden_artifact_scan": [],
            "journal_complete_present": False,
            "prepare_stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "prepare_stderr_empty": True,
            "inventory_files": 22,
            "inventory_bytes": 3855797,
            "inventory_domain_sha256": "ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36",
        },
        "non_certifying_elements": ["gpu_process_observation_late"],
        "late_observation_reference": {
            "file": "gpu_process_observation_late.json",
            "sha256": SHA,
            "certifying": False,
        },
        "reproduction_reference": {
            "file": "legacy_comparator_reproduction.json",
            "sha256": SHA_B,
        },
        "campaign_rows": 0,
        "capture_supersession_attempts_consumed": 0,
    }


def _entry_base(no_gpu_attestation: dict) -> dict:
    evidence = {"evidence_path": "/tmp/evidence", "sha256": SHA}
    return {
        "schema_version": 1,
        "entry_id": "12345678-1234-4123-8123-123456789abc",
        "entry_type": "preparation_failure_archive",
        "created_at_utc": STAMP,
        "rlcr_session_id": "fixture-session",
        "rlcr_dir": "/tmp/rlcr",
        "campaign_round": 28,
        "chunk_id": CHUNK,
        "cell_id": CELL,
        "reason_code": "STAGE_SEMANTIC_VALIDATION_FAILED",
        "runner_refusal_code": "sol_stage_source_mismatch",
        "original_head": "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a",
        "original_matrix_file_sha256": "fd3a1868845ceac309e718da6519f4779824b7fd69eddf1bfa0e7e7847d4e297",
        "original_matrix_seal_sha256": "544938301f40a46713dd056af9d349c4f59848179212ef4a7826f12d9418d447",
        "original_preflight_sha256": "3810d2b29a06163f41d6aab73375b273e87cf462a7ad22032088802eef85b63a",
        "original_admission_run_id": "0ff22c451824e77fa4ec20392866c50a7356b6de4e1e986ba788bd1b5ee1a658",
        "original_session_id": "f10de1cc-612e-43e3-b51f-ce1c9985e281",
        "replacement_head": HEAD,
        "source_journal_path": "/tmp/journal",
        "archive_path": "/tmp/archive",
        "inventory_file": "/tmp/archive/inventory.json",
        "inventory_sha256": SHA,
        "no_gpu_attestation": no_gpu_attestation,
        "pinned_source_definition_sha256": "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa",
        "staged_definition_sha256": "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387",
        "staged_module_sha256": "017ec9c8c977078c141e8d7d4fb58a67501550aa5540c220dbd97a94b8e1ee96",
        "defect_summary": "comparator lacked absent≡null for optional fields",
        "adjudication_ref": "/tmp/adjudication.md",
        "operator_flag": "--archive-preparation-failure",
    }


def _contemporaneous_entry() -> dict:
    def item(**extra: object) -> dict:
        return {"evidence_path": "/tmp/evidence", "sha256": SHA, **extra}

    attestation = {
        "journal_state": item(state="capture_started", journal_complete_present=False),
        "prepare_stdout": item(contains_only_validated_prepare_manifest=True),
        "prepare_stderr": item(empty=True),
        "forbidden_artifact_scan": item(
            checked_patterns=["*.ncu-rep"], matches=[], scanned_file_count=22
        ),
        "runner_refusal": item(
            runner_refusal_code="sol_stage_source_mismatch", refused_at_utc=STAMP
        ),
        "phase_log": item(last_phase="prepare_validate", post_prepare_phases=[]),
        "problem_packager_execute": item(
            source_head=HEAD,
            function="ProblemPackager.execute",
            citation="driver/capture_adapter.py:1",
            writes_driver_and_returns_command_without_execution=True,
        ),
        "gpu_process_observation": item(remaining_gpu_processes=[]),
    }
    return _entry_base(attestation)


def _reconstructed_entry() -> dict:
    entry = _entry_base(_attestation())
    entry.update(
        {
            "supersedes": None,
            "reconstruction_records": [
                {
                    "path": "chunk_context/reconstructed/legacy_comparator_reproduction.json",
                    "sha256": SHA,
                },
                {
                    "path": "chunk_context/reconstructed/runner_refusal_reconstruction.json",
                    "sha256": SHA,
                },
                {
                    "path": "chunk_context/reconstructed/phase_reconstruction.json",
                    "sha256": SHA,
                },
                {
                    "path": "chunk_context/reconstructed/gpu_process_observation_late.json",
                    "sha256": SHA,
                },
                {
                    "path": "chunk_context/reconstructed/preparation_failure_no_gpu_attestation_v2.json",
                    "sha256": SHA,
                },
            ],
            "authority": {
                "amendment_id": AMENDMENT,
                "supersession_status": "superseded_by_v2",
                "corrected_adjudication_sha256": SHA,
                "superseded_adjudication_sha256": SHA_B,
                "correction_id": "R28_PREP070_RECON_V2_CORR1",
                "correction_adjudication_sha256": SHA_C,
            },
        }
    )
    return entry


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_reconstruction_schema_is_valid_draft_2020_12(name: str) -> None:
    schema = _schema(name)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("runner_refusal_reconstruction.schema.json", _runner),
        ("phase_reconstruction.schema.json", _phase),
        ("gpu_process_observation_late.schema.json", _late_observation),
        ("preparation_failure_no_gpu_attestation_v2.schema.json", _attestation),
        ("legacy_comparator_reproduction.schema.json", _reproduction),
    ],
)
def test_reconstruction_record_golden_fixture_validates(name: str, factory) -> None:
    _validate(name, factory())


def test_runner_authority_basis_is_order_fixed_and_exhaustive() -> None:
    runner = _runner()
    _validate("runner_refusal_reconstruction.schema.json", runner)
    for bad in (
        [LEGACY_AUTHORITY, "annex_c_adjudication"],
        ["annex_c_adjudication", LEGACY_AUTHORITY, "extra"],
        ["annex_c_adjudication"],
    ):
        mutated = copy.deepcopy(runner)
        mutated["authority_basis"] = bad
        _rejects("runner_refusal_reconstruction.schema.json", mutated)


@pytest.mark.parametrize(
    ("name", "factory", "mutate"),
    [
        (
            "runner_refusal_reconstruction.schema.json",
            _runner,
            lambda value: value.update(authority_basis=["annex_c_adjudication", WITHDRAWN_AUTHORITY]),
        ),
        (
            "phase_reconstruction.schema.json",
            _phase,
            lambda value: value["phases"][2].update(
                evidence_basis=["annex_c_adjudication", WITHDRAWN_AUTHORITY]
            ),
        ),
        (
            "preparation_failure_no_gpu_attestation_v2.schema.json",
            _attestation,
            lambda value: value.update(
                certifying_elements=[
                    "annex_c_adjudication",
                    WITHDRAWN_AUTHORITY,
                    "journal_forensic_absence_scan",
                ]
            ),
        ),
    ],
)
def test_withdrawn_v1_refusal_authority_is_unshippable(name: str, factory, mutate) -> None:
    value = factory()
    mutate(value)
    _rejects(name, value)


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("runner_refusal_reconstruction.schema.json", _runner),
        ("phase_reconstruction.schema.json", _phase),
        ("gpu_process_observation_late.schema.json", _late_observation),
        ("preparation_failure_no_gpu_attestation_v2.schema.json", _attestation),
        ("legacy_comparator_reproduction.schema.json", _reproduction),
    ],
)
def test_retired_amendment_id_is_unshippable(name: str, factory) -> None:
    value = factory()
    value["amendment_id"] = WITHDRAWN_AMENDMENT
    _rejects(name, value)


def test_placeholder_values_are_unshippable() -> None:
    runner = _runner()
    runner["reconstructed_at"] = "TO_BE_SEALED:wall_clock_utc"
    _rejects("runner_refusal_reconstruction.schema.json", runner)

    reproduction = _reproduction()
    reproduction["tool"]["sha256"] = "TO_BE_SEALED:sha256"
    _rejects("legacy_comparator_reproduction.schema.json", reproduction)


def test_reproduction_argv_is_exactly_13_elements_in_bound_order() -> None:
    record = _reproduction()
    assert len(record["command_argv"]) == 13
    _validate("legacy_comparator_reproduction.schema.json", record)
    for argv in (
        " ".join(record["command_argv"]),
        record["command_argv"][:-1],
        record["command_argv"] + ["extra"],
        [record["command_argv"][0], *reversed(record["command_argv"][1:])],
    ):
        mutated = copy.deepcopy(record)
        mutated["command_argv"] = argv
        _rejects("legacy_comparator_reproduction.schema.json", mutated)


def test_ledger_contemporaneous_arm_preserves_h2_shape() -> None:
    entry = _contemporaneous_entry()
    _validate("preparation_failure_ledger_entry.schema.json", entry)
    for extra in ("authority", "reconstruction_records", "supersedes"):
        mutated = copy.deepcopy(entry)
        mutated[extra] = None
        _rejects("preparation_failure_ledger_entry.schema.json", mutated)


def test_ledger_reconstructed_arm_requires_closed_v2_shape() -> None:
    entry = _reconstructed_entry()
    _validate("preparation_failure_ledger_entry.schema.json", entry)
    for required in ("authority", "reconstruction_records", "supersedes"):
        mutated = copy.deepcopy(entry)
        del mutated[required]
        _rejects("preparation_failure_ledger_entry.schema.json", mutated)
    mutated = copy.deepcopy(entry)
    mutated["prior_preparation_failures"] = []
    _rejects("preparation_failure_ledger_entry.schema.json", mutated)


def test_ledger_reconstruction_records_are_order_fixed() -> None:
    entry = _reconstructed_entry()
    entry["reconstruction_records"][0], entry["reconstruction_records"][1] = (
        entry["reconstruction_records"][1],
        entry["reconstruction_records"][0],
    )
    _rejects("preparation_failure_ledger_entry.schema.json", entry)


def test_ledger_reconstructed_authority_is_v2_corr1_only() -> None:
    entry = _reconstructed_entry()
    for key, value in (
        ("amendment_id", WITHDRAWN_AMENDMENT),
        ("correction_id", "R28_PREP070_RECON_V1"),
        ("supersession_status", "binding"),
    ):
        mutated = copy.deepcopy(entry)
        mutated["authority"][key] = value
        _rejects("preparation_failure_ledger_entry.schema.json", mutated)
