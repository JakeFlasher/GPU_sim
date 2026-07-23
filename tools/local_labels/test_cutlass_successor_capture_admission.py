from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.local_labels import admit_cutlass_successor_capture as admit
from tools.local_labels import build_cutlass_successor_cohort as cohort


ROOT = Path(__file__).resolve().parents[2]
RECEIPT_SCHEMA = ROOT / admit.RECEIPT_SCHEMA_REL
OVERLAY_SCHEMA = ROOT / admit.OVERLAY_SCHEMA_REL
MATRIX = json.loads((ROOT / cohort.MATRIX_REL).read_text())


def _seal(document: dict) -> dict:
    document = copy.deepcopy(document)
    document["receipt_seal_sha256"] = admit.canonical_sha(
        {key: value for key, value in document.items() if key != "receipt_seal_sha256"}
    )
    return document


def _process(pid: int, command_byte: str) -> dict:
    process = {
        "host_identity_sha256": "a" * 64,
        "boot_id_sha256": "b" * 64,
        "pid": pid,
        "started_at_utc": f"2026-07-23T00:0{pid % 10}:00Z",
        "command_sha256": ("d" if command_byte == "a" else "e") * 64,
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
        "kernel_name": f"{slug}_kernel",
        "kernel_signature_sha256": identity["kernel_signature_sha256"],
        "grid_dim": row["shape"]["trace_shape"]["launch_config"]["grid"],
        "block_dim": row["shape"]["trace_shape"]["launch_config"]["block"],
    }
    launch["launch_identity_sha256"] = admit.canonical_sha(launch)
    process = _process(100 + index * 2 + (kind == "nvbit"), "a" if kind == "ncu" else "f")
    roles = (
        ["ncu_report", "label_sidecar", "label_run_manifest", "label_capture_manifest", "ncu_launch_observation"]
        if kind == "ncu"
        else ["dynamic_trace", "trace_summary"]
    )
    artifacts = []
    for role_index, role in enumerate(roles):
        relative = f"artifacts/{index:02d}_{kind}_{role_index}.bin"
        payload = f"synthetic-{index}-{kind}-{role}".encode()
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(payload)
        artifacts.append({"role": role, "path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
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
