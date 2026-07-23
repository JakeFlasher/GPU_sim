#!/usr/bin/env python3
"""Validate external CUTLASS successor receipts and emit an additive overlay.

This tool never edits the pinned successor cohort.  It reads the external
receipt directory, validates the immutable 237f17c bundle, and writes one
explicitly requested overlay.  A row is capture-ready only after an exact NCU
and NVBit join, including the runner's seven-field identity and launch identity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import build_cutlass_successor_cohort as cohort  # noqa: E402


PINNED_BUNDLE_COMMIT = "237f17cd34728b31ea752062e7ea7843a3a2603a"
RECEIPT_SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/cutlass_successor_capture_receipt.schema.json"
)
OVERLAY_SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/cutlass_successor_capture_admission.schema.json"
)
OVERLAY_ID = "cutlass_successor_capture_admission_sm86_v1"
DEFAULT_METRIC_VERSION = "ncu_2026.1.x"
RECEIPT_SUFFIX = ".capture_receipt.json"

BUNDLE_FILES = (
    cohort.VALIDATION_REL,
    cohort.BUDGETS_REL,
    cohort.ROSTER_REL,
    cohort.MATRIX_REL,
    cohort.VALIDATION_SCHEMA_REL,
    cohort.BUDGETS_SCHEMA_REL,
    cohort.ROSTER_SCHEMA_REL,
    cohort.MATRIX_SCHEMA_REL,
)
BUNDLE_HASHES = {
    cohort.VALIDATION_REL.as_posix(): "2682e21a0c4874f104a0480673fb965ef2e1cb16a4d8a7ef92c34a7b476ccb0e",
    cohort.BUDGETS_REL.as_posix(): "374abe73648f53c5c4e249c11a84f6b0b1ab3eb8ce9e331220919a462d01f551",
    cohort.ROSTER_REL.as_posix(): "210b8678c64208ea741c91cdd6ce1dce5d55bd29a169dbb0a5a9aa2274ce7185",
    cohort.MATRIX_REL.as_posix(): "ec0dc06429299371ec08adf9132eb1f843d99a7ba846d0ada712b747bc7a1879",
}


class AdmissionError(RuntimeError):
    """External capture admission failed closed."""


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular(path: Path, label: str) -> None:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise AdmissionError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink():
        raise AdmissionError(f"{label} is a symlink: {path}")
    if not path.is_file() or stat.st_nlink != 1:
        raise AdmissionError(f"{label} is not a single regular file: {path}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"invalid JSON {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"JSON document is not an object: {label}")
    return value


def _check_dt(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise AdmissionError(f"{label} is not an ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdmissionError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise AdmissionError(f"{label} must include a timezone")


def _validate_schema(path: Path, document: dict[str, Any], label: str) -> None:
    schema = _load_json(path, label)
    try:
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(document)
    except jsonschema.ValidationError as exc:
        raise AdmissionError(f"{label} schema validation failed: {exc.message}") from exc


def _git_blob(path: Path) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{PINNED_BUNDLE_COMMIT}:{path.as_posix()}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AdmissionError(f"pinned successor bundle is unavailable: {path}")
    return result.stdout


def validate_immutable_bundle() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Require the current four manifests and four source schemas to equal 237f17c."""
    for path in BUNDLE_FILES:
        current = ROOT / path
        _regular(current, f"pinned bundle file {path}")
        expected = _git_blob(path)
        actual = current.read_bytes()
        if actual != expected:
            raise AdmissionError(f"pinned successor bundle drifted: {path}")
        expected_hash = BUNDLE_HASHES.get(path.as_posix())
        if expected_hash is not None and sha256_bytes(actual) != expected_hash:
            raise AdmissionError(f"pinned successor manifest hash drifted: {path}")
    try:
        documents = cohort.validate_existing_bundle()
    except Exception as exc:  # normalize helper failures at this boundary
        raise AdmissionError(f"pinned successor bundle validation failed: {exc}") from exc
    return documents


def _external_root(raw: Path) -> Path:
    if not raw.is_absolute():
        raise AdmissionError("receipt root must be an absolute external path")
    candidate = Path(raw.anchor)
    for component in raw.parts[1:]:
        candidate /= component
        if candidate.is_symlink():
            raise AdmissionError(f"receipt root path contains a symlink: {candidate}")
    if raw.is_symlink():
        raise AdmissionError("receipt root is a symlink")
    root = raw.resolve()
    try:
        root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise AdmissionError("receipt root must be outside the repository")
    if not root.is_dir():
        raise AdmissionError(f"receipt root is not a directory: {raw}")
    return root


def _inventory(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(dirs):
            path = current_path / name
            if path.is_symlink():
                raise AdmissionError(f"receipt directory contains a symlink: {path}")
            if not path.is_dir():
                raise AdmissionError(f"receipt directory entry is not a directory: {path}")
        for name in names:
            path = current_path / name
            _regular(path, "receipt directory entry")
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _safe_receipt_path(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or "\\" in raw:
        raise AdmissionError(f"{label} is not a safe relative path")
    lexical = root / raw
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AdmissionError(f"{label} escapes receipt root: {raw!r}") from exc
    _regular(resolved, label)
    if resolved != lexical:
        raise AdmissionError(f"{label} resolves through a symlink: {raw!r}")
    return resolved


def _receipt_ref(root: Path, relative: str, seal: str) -> dict[str, Any]:
    path = _safe_receipt_path(root, relative, "receipt path")
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "receipt_seal_sha256": seal,
    }


def _expected_rows(matrix: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in matrix["rows"]:
        key = (row["matrix_key"], row["variant_id"])
        if key in rows:
            raise AdmissionError(f"duplicate matrix row identity: {key}")
        rows[key] = row
    if len(rows) != 3:
        raise AdmissionError("successor matrix must contain exactly three rows")
    return rows


def _verify_process(process: dict[str, Any], label: str) -> None:
    expected = canonical_sha({k: v for k, v in process.items() if k != "process_identity_sha256"})
    if process["process_identity_sha256"] != expected:
        raise AdmissionError(f"{label} process identity seal mismatch")
    _check_dt(process["started_at_utc"], f"{label} process started_at_utc")


def _verify_launch(launch: dict[str, Any], identity: dict[str, Any], row: dict[str, Any], label: str) -> None:
    expected = canonical_sha({k: v for k, v in launch.items() if k != "launch_identity_sha256"})
    if launch["launch_identity_sha256"] != expected:
        raise AdmissionError(f"{label} launch identity seal mismatch")
    if launch["kernel_signature_sha256"] != identity["kernel_signature_sha256"]:
        raise AdmissionError(f"{label} launch/kernel signature mismatch")
    shape = row["shape"]["trace_shape"]
    if launch["grid_dim"] != shape["launch_config"]["grid"] or launch["block_dim"] != shape["launch_config"]["block"]:
        raise AdmissionError(f"{label} launch geometry does not match successor matrix")


def _verify_receipt(
    root: Path,
    path: Path,
    expected_rows: dict[tuple[str, str], dict[str, Any]],
    used_artifacts: set[str],
) -> tuple[tuple[str, str], str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rel = path.relative_to(root).as_posix()
    document = _load_json(path, f"receipt {rel}")
    _validate_schema(ROOT / RECEIPT_SCHEMA_REL, document, f"receipt {rel}")
    seal = document["receipt_seal_sha256"]
    if seal != canonical_sha({k: v for k, v in document.items() if k != "receipt_seal_sha256"}):
        raise AdmissionError(f"receipt seal mismatch: {rel}")
    receipt = document["receipt"]
    pair_key = (receipt["matrix_key"], receipt["variant_id"])
    if pair_key not in expected_rows:
        raise AdmissionError(f"receipt has unknown matrix_key+variant_id: {pair_key}")
    row = expected_rows[pair_key]
    identity = receipt["identity"]
    expected_binary = row["static_launch_inventory"]["binary_sha256"]
    expected_kernel = hashlib.sha256(
        f"{row['successor_slug']}|{row['static_launch_inventory']['binary_path']}".encode("utf-8")
    ).hexdigest()
    expected_identity = {
        "slug": row["successor_slug"],
        "shape_point": row["shape"]["shape_point"],
        "binary_sha256": expected_binary,
        "source_commit": PINNED_BUNDLE_COMMIT,
        "kernel_signature_sha256": expected_kernel,
        "arch": row["arch"],
        "ncu_metric_version": DEFAULT_METRIC_VERSION,
    }
    if identity != expected_identity:
        raise AdmissionError(f"receipt identity does not match matrix: {rel}")
    _verify_launch(receipt["launch_identity"], identity, row, rel)
    _verify_process(receipt["process"], rel)
    expected_roles = (
        {"ncu_report", "label_sidecar", "label_run_manifest", "label_capture_manifest", "ncu_launch_observation"}
        if receipt["capture_kind"] == "ncu"
        else {"dynamic_trace", "trace_summary"}
    )
    seen_roles: set[str] = set()
    for artifact in receipt["artifacts"]:
        role = artifact["role"]
        if role in seen_roles:
            raise AdmissionError(f"duplicate artifact role in receipt: {rel}:{role}")
        seen_roles.add(role)
        if role not in expected_roles:
            raise AdmissionError(f"unexpected artifact role in receipt: {rel}:{role}")
        artifact_path = _safe_receipt_path(root, artifact["path"], "artifact path")
        artifact_rel = artifact_path.relative_to(root).as_posix()
        if artifact_rel.endswith(RECEIPT_SUFFIX):
            raise AdmissionError(f"receipt listed as its own artifact: {artifact_rel}")
        if artifact_rel in used_artifacts:
            raise AdmissionError(f"duplicate artifact path: {artifact_rel}")
        used_artifacts.add(artifact_rel)
        payload = artifact_path.read_bytes()
        if sha256_bytes(payload) != artifact["sha256"] or len(payload) != artifact["size_bytes"]:
            raise AdmissionError(f"artifact hash or size drifted: {artifact_rel}")
    if seen_roles != expected_roles:
        raise AdmissionError(f"receipt artifact set incomplete: {rel}")
    return pair_key, receipt["capture_kind"], receipt["identity"], receipt["launch_identity"], receipt["process"], _receipt_ref(root, rel, seal)


def _row_key(pair_key: tuple[str, str]) -> dict[str, str]:
    return {"matrix_key": pair_key[0], "variant_id": pair_key[1]}


def _process_separation(ncu: dict[str, Any], nvbit: dict[str, Any]) -> dict[str, Any]:
    if ncu["host_identity_sha256"] != nvbit["host_identity_sha256"] or ncu["boot_id_sha256"] != nvbit["boot_id_sha256"]:
        raise AdmissionError("NCU and NVBit receipts do not prove the same host boot")
    same_tuple = (ncu["pid"], ncu["started_at_utc"]) == (nvbit["pid"], nvbit["started_at_utc"])
    if ncu["process_identity_sha256"] == nvbit["process_identity_sha256"] or same_tuple:
        raise AdmissionError("NCU and NVBit receipts do not prove separate processes")
    if ncu["command_sha256"] == nvbit["command_sha256"]:
        raise AdmissionError("NCU and NVBit receipts have the same process command")
    return {
        "host_identity_sha256": ncu["host_identity_sha256"],
        "boot_id_sha256": ncu["boot_id_sha256"],
        "ncu_process_identity_sha256": ncu["process_identity_sha256"],
        "nvbit_process_identity_sha256": nvbit["process_identity_sha256"],
        "separate": True,
    }


def _bundle_overlay_meta(matrix: dict[str, Any]) -> dict[str, Any]:
    base = {
        "source_commit": PINNED_BUNDLE_COMMIT,
        "validation": {"path": cohort.VALIDATION_REL.as_posix(), "sha256": BUNDLE_HASHES[cohort.VALIDATION_REL.as_posix()]},
        "trace_budgets": {"path": cohort.BUDGETS_REL.as_posix(), "sha256": BUNDLE_HASHES[cohort.BUDGETS_REL.as_posix()]},
        "roster": {"path": cohort.ROSTER_REL.as_posix(), "sha256": BUNDLE_HASHES[cohort.ROSTER_REL.as_posix()]},
        "capture_matrix": {"path": cohort.MATRIX_REL.as_posix(), "sha256": BUNDLE_HASHES[cohort.MATRIX_REL.as_posix()]},
    }
    return {**base, "bundle_identity_sha256": canonical_sha(base)}


def build_overlay(receipt_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = _external_root(receipt_root)
    _check_dt(generated_at_utc, "generated_at_utc")
    _, _, _, matrix = validate_immutable_bundle()
    expected = _expected_rows(matrix)
    files = _inventory(root)
    receipt_files = [path for path in files if path.name.endswith(RECEIPT_SUFFIX)]
    extras = [path for path in files if not path.name.endswith(RECEIPT_SUFFIX)]
    used_artifacts: set[str] = set()
    receipts: dict[tuple[str, str], dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]] = {}
    for path in receipt_files:
        pair_key, kind, identity, launch, process, ref = _verify_receipt(root, path, expected, used_artifacts)
        by_kind = receipts.setdefault(pair_key, {})
        if kind in by_kind:
            raise AdmissionError(f"duplicate {kind} receipt for {pair_key}")
        by_kind[kind] = (identity, launch, process, ref)
    referenced = set(used_artifacts) | {path.relative_to(root).as_posix() for path in receipt_files}
    extra_rel = [path.relative_to(root).as_posix() for path in extras if path.relative_to(root).as_posix() not in referenced]
    if extra_rel:
        raise AdmissionError(f"unreferenced extra receipt files: {extra_rel}")
    inventory = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_bytes(path.read_bytes()), "size_bytes": path.stat().st_size}
        for path in files
    ]
    rows: list[dict[str, Any]] = []
    ready_keys: list[tuple[str, str]] = []
    for pair_key in expected:
        by_kind = receipts.get(pair_key, {})
        ncu = by_kind.get("ncu")
        nvbit = by_kind.get("nvbit")
        if ncu is None and nvbit is None:
            rows.append({"matrix_key": pair_key[0], "variant_id": pair_key[1], "status": "pending", "typed_reason": "ncu_and_nvbit_receipts_absent", "ncu_receipt": None, "nvbit_receipt": None, "identity": None, "launch_identity": None, "process_separation": None})
            continue
        if ncu is None or nvbit is None:
            rows.append({"matrix_key": pair_key[0], "variant_id": pair_key[1], "status": "pending", "typed_reason": "ncu_receipt_absent" if ncu is None else "nvbit_receipt_absent", "ncu_receipt": ncu[3] if ncu else None, "nvbit_receipt": nvbit[3] if nvbit else None, "identity": None, "launch_identity": None, "process_separation": None})
            continue
        identity, launch, ncu_process, ncu_ref = ncu
        nv_identity, nv_launch, nv_process, nv_ref = nvbit
        if identity != nv_identity or launch != nv_launch:
            raise AdmissionError(f"NCU/NVBit identity or launch join mismatch: {pair_key}")
        separation = _process_separation(ncu_process, nv_process)
        rows.append({"matrix_key": pair_key[0], "variant_id": pair_key[1], "status": "capture_ready", "typed_reason": "valid_joined_receipt_pair", "ncu_receipt": ncu_ref, "nvbit_receipt": nv_ref, "identity": identity, "launch_identity": launch, "process_separation": separation})
        ready_keys.append(pair_key)
    row_keys = [_row_key(pair_key) for pair_key in ready_keys]
    chunks = []
    if row_keys:
        chunks = [{"chunk_id": "cutlass-successor-capture-000", "rows": row_keys, "membership_sha256": canonical_sha(row_keys)}]
    if not receipt_files:
        dispatch_status, dispatch_reason = "not_admitted", "capture_receipts_absent"
    elif ready_keys and len(ready_keys) == len(expected):
        dispatch_status, dispatch_reason = "capture_ready", "all_successor_receipt_pairs_valid"
    elif ready_keys:
        dispatch_status, dispatch_reason = "partially_admitted", "capture_receipts_partially_ready"
    else:
        dispatch_status, dispatch_reason = "not_admitted", "capture_receipts_partially_ready"
    document = {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_admission_overlay",
        "overlay_id": OVERLAY_ID,
        "generated_at_utc": generated_at_utc,
        "base_bundle": _bundle_overlay_meta(matrix),
        "receipt_root": {"path": str(root), "inventory_sha256": canonical_sha(inventory)},
        "dispatch": {"status": dispatch_status, "m6_schedule_created": False, "typed_reason": dispatch_reason},
        "chunking": {"policy": "capture_ready_rows_only", "max_rows_per_chunk": 8, "runnable_order": row_keys, "runnable_membership_sha256": canonical_sha(row_keys)},
        "chunks": chunks,
        "rows": rows,
        "counts": {"total": len(rows), "pending": len(rows) - len(ready_keys), "capture_ready": len(ready_keys), "chunks": len(chunks)},
        "integrity": {"base_bundle_unchanged": True, "unique_matrix_keys": True, "unique_variant_ids": True, "receipt_inventory_closed": True, "only_ready_rows_chunked": True, "separate_ncu_nvbit_processes": True},
    }
    document["overlay_seal_sha256"] = canonical_sha(document)
    _validate_schema(ROOT / OVERLAY_SCHEMA_REL, document, "capture admission overlay")
    return document


def write_overlay(document: dict[str, Any], output: Path, receipt_root: Path) -> None:
    if not output.is_absolute():
        raise AdmissionError("overlay output must be an absolute path")
    resolved_root = _external_root(receipt_root)
    target = output.resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        pass
    else:
        raise AdmissionError("overlay output must not be inside receipt root")
    if output.exists() or output.is_symlink():
        raise AdmissionError(f"refusing to overwrite existing overlay: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AdmissionError(f"overlay output parent is unavailable: {output.parent}")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with output.open("xb") as stream:
            stream.write(payload)
    except OSError as exc:
        raise AdmissionError(f"cannot write overlay {output}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at-utc", default="2026-07-23T00:00:00Z")
    args = parser.parse_args(argv)
    try:
        document = build_overlay(args.receipt_root, generated_at_utc=args.generated_at_utc)
        write_overlay(document, args.output, args.receipt_root)
    except (AdmissionError, OSError, ValueError, jsonschema.SchemaError) as exc:
        print(f"CUTLASS successor capture admission failed: {exc}", file=sys.stderr)
        return 1
    print(f"CUTLASS successor capture admission: pass=True -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
