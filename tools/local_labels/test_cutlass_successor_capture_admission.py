from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.local_labels import admit_cutlass_successor_capture as admit
from tools.local_labels import build_cutlass_successor_cohort as cohort


ROOT = Path(__file__).resolve().parents[2]
RECEIPT_SCHEMA = ROOT / admit.RECEIPT_SCHEMA_REL
OVERLAY_SCHEMA = ROOT / admit.OVERLAY_SCHEMA_REL
MATRIX = json.loads((ROOT / cohort.MATRIX_REL).read_text())


_REPORT_CSV = """Kernel Name,Metric Name,Metric Value
_Zsynthetic_kernel,gpc__cycles_elapsed.max,1
_Zsynthetic_kernel,gpu__time_duration.sum,1
"""


@pytest.fixture(autouse=True)
def _mock_ncu_report_import(monkeypatch):
    monkeypatch.setattr(admit, "_import_ncu_report_csv", lambda path: (_REPORT_CSV, None))




def _report_sidecar() -> dict:
    return {
        "unit_tagged_metrics": [
            {"ncu_metric": "gpc__cycles_elapsed.max", "unit": "cycles", "value": 1.0},
            {"ncu_metric": "gpu__time_duration.sum", "unit": "ns", "value": 1.0},
        ]
    }


def _report_observation() -> dict:
    observation = {
        "identity_columns": ["Kernel Name"],
        "identity": {"Kernel Name": "_Zsynthetic_kernel"},
        "report_sha256": "unused",
    }
    return observation


def test_ncu_report_import_failure_is_not_sidecar_only(tmp_path: Path, monkeypatch):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    monkeypatch.setattr(
        admit, "_import_ncu_report_csv", lambda path: (None, type("E", (), {"detail": "ncu unavailable"})())
    )
    with pytest.raises(admit.AdmissionError, match="import failed"):
        admit._verify_ncu_report(report, admit.sha256_bytes(report.read_bytes()), _report_sidecar(), _report_observation())


def test_ncu_report_metric_disagreement_is_rejected(tmp_path: Path, monkeypatch):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    csv = "Kernel Name,Metric Name,Metric Value\n_Zsynthetic_kernel,gpc__cycles_elapsed.max,2\n_Zsynthetic_kernel,gpu__time_duration.sum,1\n"
    monkeypatch.setattr(admit, "_import_ncu_report_csv", lambda path: (csv, None))
    with pytest.raises(admit.AdmissionError, match="metric disagreement"):
        admit._verify_ncu_report(report, admit.sha256_bytes(report.read_bytes()), _report_sidecar(), _report_observation())


def test_ncu_report_multiple_launches_are_rejected(tmp_path: Path, monkeypatch):
    report = tmp_path / "run.ncu-rep"
    report.write_bytes(b"report")
    csv = "ID,Kernel Name,Metric Name,Metric Value\n0,_Zsynthetic_kernel,gpc__cycles_elapsed.max,1\n1,_Zsynthetic_kernel,gpc__cycles_elapsed.max,1\n0,_Zsynthetic_kernel,gpu__time_duration.sum,1\n1,_Zsynthetic_kernel,gpu__time_duration.sum,1\n"
    monkeypatch.setattr(admit, "_import_ncu_report_csv", lambda path: (csv, None))
    with pytest.raises(admit.AdmissionError, match="metric extraction failed"):
        admit._verify_ncu_report(report, admit.sha256_bytes(report.read_bytes()), _report_sidecar(), _report_observation())


def _seal(document: dict) -> dict:
    document = copy.deepcopy(document)
    document["receipt_seal_sha256"] = admit.canonical_sha(
        {key: value for key, value in document.items() if key != "receipt_seal_sha256"}
    )
    return document


def _process(pid: int, command: list[str]) -> dict:
    process = {
        "host_identity_sha256": "a" * 64,
        "boot_id_sha256": "b" * 64,
        "pid": pid,
        "started_at_utc": f"2026-07-23T00:0{pid % 10}:00Z",
        "command_sha256": admit.canonical_sha(command),
    }
    process["process_identity_sha256"] = admit.canonical_sha(process)
    return process


def _receipt(root: Path, row: dict, kind: str, index: int) -> Path:
    slug = row["successor_slug"]
    identity = {
        "slug": slug,
        "shape_point": row["shape"]["shape_point"],
        "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
        "source_commit": admit.PINNED_BUNDLE_COMMIT,
        "kernel_signature_sha256": hashlib.sha256(
            f"{slug}|{row['static_launch_inventory']['binary_path']}".encode()
        ).hexdigest(),
        "arch": row["arch"],
        "ncu_metric_version": admit.DEFAULT_METRIC_VERSION,
    }
    launch = {
        "launch_index": 0,
        "kernel_name": slug,
        "kernel_signature_sha256": identity["kernel_signature_sha256"],
        "grid_dim": row["shape"]["trace_shape"]["launch_config"]["grid"],
        "block_dim": row["shape"]["trace_shape"]["launch_config"]["block"],
    }
    launch["launch_identity_sha256"] = admit.canonical_sha(launch)
    command = ["ncu" if kind == "ncu" else "nvbit", slug, "--shape-point", "trace_00"]
    process = _process(100 + index * 2 + (kind == "nvbit"), command)
    roles = (
        ["ncu_report", "label_sidecar", "label_run_manifest", "ncu_launch_observation", "preflight", "environment"]
        if kind == "ncu"
        else ["dynamic_trace", "stats", "threadblock", "trace_summary", "preflight", "environment"]
    )
    if kind == "nvbit":
        from tools.nvbit_tracer.test_trace_admission import _write_trace
        trace_root = root / "nvbit" / f"{index:02d}" / slug
        trace_root.parent.mkdir(parents=True, exist_ok=True)
        _write_trace(trace_root)
    artifacts = []
    preflight_path = None
    preflight_sha = None
    environment_sha = None
    for role_index, role in enumerate(roles):
        relative = (
            f"artifacts/{index:02d}_{kind}_{role_index}.bin"
            if kind == "ncu"
            else (
                f"nvbit/{index:02d}/{slug}/threadblocks/device_0/stream_0/kernel_0/0.pb"
                if role == "threadblock"
                else (
                    f"nvbit/{index:02d}/{slug}/{'dynamic_trace.pb' if role == 'dynamic_trace' else 'stats.csv' if role == 'stats' else 'trace_summary.json' if role == 'trace_summary' else f'{role_index}_{role}.bin'}"
                )
            )
        )
        payload = f"synthetic-{index}-{kind}-{role}".encode()
        if kind == "nvbit" and role in {"dynamic_trace", "stats", "threadblock"}:
            payload = (root / relative).read_bytes()
        if kind == "ncu" and role == "label_sidecar":
            payload = json.dumps({
                "schema_version": 5,
                "device": row["arch"],
                "slug": slug,
                "shape_point": row["shape"]["shape_point"],
                "repeat_index": 0,
                "launch_index": 0,
                "binary_sha256": identity["binary_sha256"],
                "profiler_family": "nsight_compute",
                "profiler_version": "2026.1.0",
                "cuda_version": "13.2",
                "driver_version": "610.43.02",
                "gpu_uuid_sha256": "c" * 64,
                "replay_mode": "kernel",
                "aggregation": "kernel",
                "warmup_count": 0,
                "cache_hit_rate_l2_sector": 0.5,
                "confidence_interval_l2_sector": {
                    "lower": 0.5, "upper": 0.5, "confidence_level": 0.95,
                    "units": "sector_hit_rate_fraction", "aggregation": "kernel",
                },
                "measurement_protocol": {
                    "ncu_clock_control": "none", "ncu_cache_control": "all",
                    "ncu_replay_mode": "kernel",
                },
                "unit_tagged_metrics": [
                    {"ncu_metric": "gpc__cycles_elapsed.max", "unit": "cycles", "value": 1},
                    {"ncu_metric": "gpu__time_duration.sum", "unit": "ns", "value": 1},
                ],
            }, sort_keys=True).encode()
        elif kind == "ncu" and role == "label_run_manifest":
            payload = json.dumps({
                "schema_version": 1, "source_commit": admit.PINNED_BUNDLE_COMMIT,
                "binary_sha256": identity["binary_sha256"], "kernel_signature_sha256": identity["kernel_signature_sha256"], "arch": row["arch"],
                "slug": slug, "shape_point": row["shape"]["shape_point"],
                "repeat_index": 0, "launch_index": 0,
            }, sort_keys=True).encode()
        elif kind == "ncu" and role == "ncu_launch_observation":
            report_payload = f"synthetic-{index}-{kind}-ncu_report".encode()
            observation = {
                "schema_version": 1, "kind": "ncu_details_csv_launch_observation",
                "kernel_name_base": "mangled", "requested_launch_skip": 0,
                "requested_launch_count": 1, "report_sha256": hashlib.sha256(report_payload).hexdigest(),
                "identity_columns": ["Kernel Name"], "identity": {"Kernel Name": "_Zsynthetic_kernel"},
            }
            observation["observation_sha256"] = admit.canonical_sha(observation)
            payload = json.dumps(observation, sort_keys=True).encode()
        elif kind == "nvbit" and role == "trace_summary":
            payload = json.dumps({
                "schema_version": 1, "producer": "nvbit_tracer",
                "artifact_kind": "cutlass_successor_nvbit_trace_summary",
                "trace_contract_id": admit.TRACE_CONTRACT_ID,
                "memory_payload_classes": admit.TRACE_MEMORY_PAYLOAD_CLASSES,
                "binary_version": 86, "nvbit_version": "1.8",
                "trace_contract_status": "pass", "expected_record_count": 1,
                "admitted_record_count": 1, "threadblock_count": 1,
                "dynamic_trace": "dynamic_trace.pb", "stats": "stats.csv",
            }, sort_keys=True).encode()
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        artifacts.append({"role": role, "path": relative, "sha256": digest, "size_bytes": len(payload)})
        if role == "preflight":
            preflight_path, preflight_sha = relative, digest
        if role == "environment":
            environment_sha = digest
    receipt = {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_receipt",
        "receipt": {
            "capture_kind": kind,
            "status": "pass",
            "matrix_key": row["matrix_key"],
            "variant_id": row["variant_id"],
            "identity": identity,
            "launch_identity": launch,
            "process": process,
            "provenance": {
                "capture_producer_sha256": admit.sha256_bytes(
                    (ROOT / "tools/local_labels/capture_cutlass_successor_receipts.py").read_bytes()
                ),
                "phase_tool_sha256": admit.sha256_bytes(
                    (ROOT / ("tools/local_labels/label_driver.sh" if kind == "ncu" else "tools/nvbit_tracer/tracer_tool/tracer_tool.so")).read_bytes()
                ),
            },
            "captured_at_utc": "2026-07-23T01:00:00Z",
            "preflight_path": preflight_path,
            "preflight_sha256": preflight_sha,
            "environment_sha256": environment_sha,
            "command": command,
            "command_sha256": admit.canonical_sha(command),
            "semantic_validation": (
                {
                    "semantic_kind": "ncu_metric_validation",
                    "primary_metric": "gpc__cycles_elapsed.max",
                    "secondary_metric": "gpu__time_duration.sum",
                    "primary_metric_present": True,
                    "secondary_metric_present": True,
                    "target_only": True,
                    "simulator_config_present": False,
                }
                if kind == "ncu"
                else {
                    "semantic_kind": "nvbit_trace_validation",
                    "trace_contract_status": "pass",
                    "dynamic_trace_role": "dynamic_trace",
                    "summary_role": "trace_summary",
                    "target_only": True,
                    "simulator_config_present": False,
                }
            ),
            "artifacts": artifacts,
        },
    }
    path = root / f"{index:02d}_{kind}{admit.RECEIPT_SUFFIX}"
    path.write_text(json.dumps(_seal(receipt), indent=2, sort_keys=True) + "\n")
    return path


def _all_pairs(root: Path) -> None:
    for index, row in enumerate(MATRIX["rows"]):
        _receipt(root, row, "ncu", index)
        _receipt(root, row, "nvbit", index)


def _rewrite(path: Path, document: dict) -> None:
    path.write_text(json.dumps(_seal(document), indent=2, sort_keys=True) + "\n")


def test_new_schemas_are_closed_draft_2020_12():
    for path in (RECEIPT_SCHEMA, OVERLAY_SCHEMA):
        schema = json.loads(path.read_text())
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"].endswith("draft/2020-12/schema")
        assert schema["additionalProperties"] is False


def test_valid_joined_pairs_produce_capture_ready_chunk(tmp_path: Path):
    _all_pairs(tmp_path)
    overlay = admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")
    assert overlay["counts"] == {"total": 3, "pending": 0, "capture_ready": 3, "chunks": 1}
    assert all(row["status"] == "capture_ready" for row in overlay["rows"])
    assert len(overlay["chunks"]) == 1
    Draft202012Validator(json.loads(OVERLAY_SCHEMA.read_text()), format_checker=None).validate(overlay)


def test_absent_receipts_are_typed_pending_and_have_zero_chunks(tmp_path: Path):
    overlay = admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")
    assert overlay["dispatch"]["typed_reason"] == "capture_receipts_absent"
    assert overlay["counts"] == {"total": 3, "pending": 3, "capture_ready": 0, "chunks": 0}
    assert overlay["chunks"] == []
    assert {row["typed_reason"] for row in overlay["rows"]} == {"ncu_and_nvbit_receipts_absent"}


def test_one_missing_variant_stays_pending(tmp_path: Path):
    row = MATRIX["rows"][0]
    _receipt(tmp_path, row, "ncu", 0)
    overlay = admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")
    assert overlay["rows"][0]["typed_reason"] == "nvbit_receipt_absent"
    assert overlay["rows"][0]["status"] == "pending"
    assert overlay["counts"]["chunks"] == 0


@pytest.mark.parametrize("case", ["hash", "identity", "launch", "unknown", "duplicate", "extra"])
def test_receipt_validation_rejects_negative_cases(tmp_path: Path, case: str):
    _all_pairs(tmp_path)
    target = tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}"
    document = json.loads(target.read_text())
    if case == "hash":
        (tmp_path / "artifacts/00_ncu_0.bin").write_bytes(b"drift")
    elif case == "identity":
        document["receipt"]["identity"]["arch"] = "sm_120"
        _rewrite(target, document)
    elif case == "launch":
        document["receipt"]["launch_identity"]["grid_dim"][0] += 1
        _rewrite(target, document)
    elif case == "unknown":
        document["receipt"]["unexpected"] = True
        _rewrite(target, document)
    elif case == "duplicate":
        shutil.copy2(target, tmp_path / f"duplicate_ncu{admit.RECEIPT_SUFFIX}")
    elif case == "extra":
        (tmp_path / "unreferenced.bin").write_bytes(b"extra")
    with pytest.raises(admit.AdmissionError):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_receipt_path_escape_is_rejected_by_schema(tmp_path: Path):
    row = MATRIX["rows"][0]
    path = _receipt(tmp_path, row, "ncu", 0)
    document = json.loads(path.read_text())
    document["receipt"]["artifacts"][0]["path"] = "../escape.bin"
    _rewrite(path, document)
    with pytest.raises(admit.AdmissionError):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_symlink_in_external_receipt_inventory_is_rejected(tmp_path: Path):
    _all_pairs(tmp_path)
    link = tmp_path / "symlink.bin"
    link.symlink_to(tmp_path / "artifacts/00_ncu_0.bin")
    with pytest.raises(admit.AdmissionError, match="symlink"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_same_process_receipts_cannot_promote(tmp_path: Path):
    _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    nvbit = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    document = json.loads(nvbit.read_text())
    ncu = json.loads((tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}").read_text())
    document["receipt"]["process"] = ncu["receipt"]["process"]
    document["receipt"]["command"] = ncu["receipt"]["command"]
    document["receipt"]["command_sha256"] = ncu["receipt"]["command_sha256"]
    _rewrite(nvbit, document)
    with pytest.raises(admit.AdmissionError, match="separate processes"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_extra_receipt_kind_is_rejected(tmp_path: Path):
    _all_pairs(tmp_path)
    target = tmp_path / f"00_nvbit{admit.RECEIPT_SUFFIX}"
    document = json.loads(target.read_text())
    document["receipt"]["capture_kind"] = "ncu"
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_process_command_hash_must_bind_receipt_command(tmp_path: Path):
    _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    target = tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}"
    document = json.loads(target.read_text())
    document["receipt"]["command"] = ["tampered"]
    document["receipt"]["command_sha256"] = admit.canonical_sha(["tampered"])
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="process command"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_kernel_name_must_bind_successor_identity(tmp_path: Path):
    _receipt(tmp_path, MATRIX["rows"][0], "ncu", 0)
    target = tmp_path / f"00_ncu{admit.RECEIPT_SUFFIX}"
    document = json.loads(target.read_text())
    document["receipt"]["launch_identity"]["kernel_name"] = "wrong_successor"
    launch = document["receipt"]["launch_identity"]
    launch["launch_identity_sha256"] = admit.canonical_sha(
        {key: value for key, value in launch.items() if key != "launch_identity_sha256"}
    )
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError, match="kernel_name"):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")


def test_nvbit_receipt_requires_stats_and_threadblock_payloads(tmp_path: Path):
    target = _receipt(tmp_path, MATRIX["rows"][0], "nvbit", 0)
    document = json.loads(target.read_text())
    document["receipt"]["artifacts"] = [
        artifact
        for artifact in document["receipt"]["artifacts"]
        if artifact["role"] not in {"stats", "threadblock"}
    ]
    _rewrite(target, document)
    with pytest.raises(admit.AdmissionError):
        admit.build_overlay(tmp_path, generated_at_utc="2026-07-23T01:00:00Z")
