#!/usr/bin/env python3
"""Validate or emit the staged-tree metric-extractor artifact pin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_external_output,
)
from tools.nvbit_tracer.test_nvbit_release_manifest import (  # noqa: E402
    SDK_PATHS,
    parse_manifest,
)

PIN_REL = Path("tools/nvbit_tracer/metric_extractor/artifact_pin.json")
SDK_MANIFEST_REL = "tools/nvbit_tracer/nvbit_release_manifest.sha256"
LOCK_REL = "tools/nvbit_tracer/toolchain/pb_explicit.lock"
ARTIFACT_REL = "tools/nvbit_tracer/metric_extractor/metric_extractor.so"
STAMP_REL = Path(
    "tools/nvbit_tracer/metric_extractor/metric_extractor.provenance.json"
)
ARTIFACT_SHA256 = "73045a303ac25f99135290468da20aae1c50d275a8e9aca134869cb0229a7e5f"
LIBNVBIT_SHA256 = "db221829106673bcd69d1e05766136f80e2df4497fb8415d8c2f298b96c302f7"
SCRATCH_SHA256 = "3431c6a16457777493808c6fb519628119ca4cc9663d5d7574c6d2791b2e2c9f"
LOCK_SHA256 = "d522b3ee1560e98512cd460b213adee5a901f77aef0e4fae3ba94699e3b1e222"
SOURCE_CLOSURE_PATHS = tuple(sorted((
    "tools/nvbit_tracer/metric_extractor/Makefile",
    "tools/nvbit_tracer/metric_extractor/metric_extractor.cu",
    "tools/nvbit_tracer/metric_extractor/inject_funcs.cu",
    "tools/nvbit_tracer/metric_extractor/bank_counting.h",
    "tools/nvbit_tracer/metric_extractor/branch_counting.h",
    "tools/nvbit_tracer/metric_extractor/classifier.h",
    "tools/nvbit_tracer/metric_extractor/common.h",
    "tools/nvbit_tracer/metric_extractor/module_origin_resolver.h",
    "tools/nvbit_tracer/metric_extractor/saturating_counter.h",
    "tools/nvbit_tracer/metric_extractor/sector_counting.h",
    "tools/nvbit_tracer/metric_extractor/tier2_frame_serializer.h",
    "tools/nvbit_tracer/metric_extractor/tier2_sha256.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/ampere_opcode.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/blackwell_opcode.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/hopper_opcode.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/kepler_opcode.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/operation_type.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/pascal_opcode.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/trace_opcode.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/turing_opcode.h",
    "tools/nvbit_tracer/metric_extractor/isa_def/volta_opcode.h",
)))
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS = {
    "schema_version", "artifact_path", "artifact_sha256", "artifact_status",
    "built_under_current_contract", "legacy_commit", "h3_hold",
    "allowed_build_arches", "artifact_embedded_arches", "source_closure",
    "sdk_manifest_sha256", "libnvbit_sha256", "scratch_witness",
    "replacement_rule",
}


class MissingTreeInput(RuntimeError):
    pass


class PinMismatch(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, indent=2,
    ) + "\n").encode("utf-8")


def _git_blob(tree: str, path: str) -> bytes:
    spec = f":{path}" if tree == "INDEX" else f"HEAD:{path}"
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "blob", spec],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise MissingTreeInput(f"missing {tree} blob: {path}")
    return proc.stdout


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sdk_manifest(payload: bytes, tree: str) -> None:
    rows = parse_manifest(payload)
    if len(rows) != 53:
        raise PinMismatch(f"SDK manifest must contain 53 rows, got {len(rows)}")
    for path, digest in rows:
        if path not in SDK_PATHS:
            raise PinMismatch(f"unexpected SDK path: {path}")
        if _sha(_git_blob(tree, path)) != digest:
            raise PinMismatch(f"SDK manifest digest mismatch: {path}")


def _validate_lock(payload: bytes) -> None:
    if _sha(payload) != LOCK_SHA256:
        raise PinMismatch("protobuf explicit-lock bytes mismatch")
    lines = payload.decode("utf-8").splitlines()
    if len(lines) != 8 or lines[0] != "@EXPLICIT":
        raise PinMismatch("protobuf explicit-lock grammar mismatch")
    for row in lines[1:]:
        if not row.startswith("https://") or "#" not in row:
            raise PinMismatch("protobuf explicit-lock row malformed")
        digest = row.rsplit("#", 1)[1]
        if not _HEX64.fullmatch(digest):
            raise PinMismatch("protobuf explicit-lock digest malformed")


def generate_pin(tree: str) -> dict:
    sdk_payload = _git_blob(tree, SDK_MANIFEST_REL)
    lock_payload = _git_blob(tree, LOCK_REL)
    _validate_sdk_manifest(sdk_payload, tree)
    _validate_lock(lock_payload)
    return {
        "schema_version": 1,
        "artifact_path": ARTIFACT_REL,
        "artifact_sha256": ARTIFACT_SHA256,
        "artifact_status": "grandfathered",
        "built_under_current_contract": False,
        "legacy_commit": "deae005734d3119ee5fe105bd4f852b9e64f92ac",
        "h3_hold": True,
        "allowed_build_arches": ["sm_86", "sm_100a"],
        "artifact_embedded_arches": ["sm_86"],
        "source_closure": [
            {"path": path, "sha256": _sha(_git_blob(tree, path))}
            for path in SOURCE_CLOSURE_PATHS
        ],
        "sdk_manifest_sha256": _sha(sdk_payload),
        "libnvbit_sha256": LIBNVBIT_SHA256,
        "scratch_witness": {
            "artifact_sha256": SCRATCH_SHA256,
            "status": "pre-contract feasibility witness",
            "pb_lock_sha256": _sha(lock_payload),
        },
        "replacement_rule": {
            "operator_wait_required": True,
            "replacement_must_pass_current_contract": True,
            "canonical_worktree_rebuild_forbidden_while_h3": True,
            "repin_forbidden_while_h3": True,
        },
    }


def validate_pin(document: dict) -> None:
    if set(document) != _TOP_KEYS:
        raise PinMismatch("artifact pin top-level key set mismatch")
    constants = {
        "schema_version": 1,
        "artifact_path": ARTIFACT_REL,
        "artifact_sha256": ARTIFACT_SHA256,
        "artifact_status": "grandfathered",
        "built_under_current_contract": False,
        "legacy_commit": "deae005734d3119ee5fe105bd4f852b9e64f92ac",
        "h3_hold": True,
        "allowed_build_arches": ["sm_86", "sm_100a"],
        "artifact_embedded_arches": ["sm_86"],
        "libnvbit_sha256": LIBNVBIT_SHA256,
    }
    for key, expected in constants.items():
        if document.get(key) != expected:
            raise PinMismatch(f"artifact pin constant mismatch: {key}")
    closure = document.get("source_closure")
    if not isinstance(closure, list) or len(closure) != 21:
        raise PinMismatch("source closure must contain exactly 21 entries")
    if tuple(row.get("path") for row in closure) != SOURCE_CLOSURE_PATHS:
        raise PinMismatch("source closure path set/order mismatch")
    if any(set(row) != {"path", "sha256"} for row in closure):
        raise PinMismatch("source closure entries must be closed")
    hashes = [row["sha256"] for row in closure]
    if any(not isinstance(value, str) or not _HEX64.fullmatch(value)
           for value in hashes):
        raise PinMismatch("source closure contains malformed digest")
    scratch = document.get("scratch_witness")
    if scratch != {
        "artifact_sha256": SCRATCH_SHA256,
        "status": "pre-contract feasibility witness",
        "pb_lock_sha256": LOCK_SHA256,
    }:
        raise PinMismatch("scratch witness mismatch")
    replacement = document.get("replacement_rule")
    if replacement != {
        "operator_wait_required": True,
        "replacement_must_pass_current_contract": True,
        "canonical_worktree_rebuild_forbidden_while_h3": True,
        "repin_forbidden_while_h3": True,
    }:
        raise PinMismatch("replacement rule mismatch")
    for key in ("sdk_manifest_sha256",):
        value = document.get(key)
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise PinMismatch(f"malformed digest: {key}")


def validate_local_artifact(document: dict) -> None:
    artifact = REPO_ROOT / ARTIFACT_REL
    if artifact.exists():
        if not artifact.is_file() or artifact.is_symlink():
            raise PinMismatch("present grandfathered artifact is not a regular file")
        if _sha(artifact.read_bytes()) != ARTIFACT_SHA256:
            raise PinMismatch("present grandfathered artifact digest mismatch")
    stamp = REPO_ROOT / STAMP_REL
    if not stamp.exists():
        return
    if not stamp.is_file() or stamp.is_symlink():
        raise PinMismatch("optional provenance stamp is not a regular file")
    try:
        stamp_doc = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinMismatch(f"optional provenance stamp is invalid: {exc}") from exc
    expected = {
        "schema_version": 1,
        "artifact_path": ARTIFACT_REL,
        "artifact_sha256": ARTIFACT_SHA256,
        "artifact_pin_sha256": _sha(canonical_json_bytes(document)),
        "status": "grandfathered-legacy",
        "observed_embedded_arches": ["sm_86"],
    }
    if stamp_doc != expected:
        raise PinMismatch("optional provenance stamp content mismatch")


def _atomic_emit(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, choices=("INDEX", "HEAD"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--emit", type=Path)
    action.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    try:
        document = generate_pin(args.tree)
        validate_pin(document)
        # Authority emission never masks a damaged present grandfathered
        # artifact or malformed optional stamp.  Absence remains valid and
        # still performs complete structural/tree validation.
        validate_local_artifact(document)
        payload = canonical_json_bytes(document)
        selected = args.emit if args.emit is not None else args.check
        assert selected is not None
        if args.emit is not None:
            destination = assert_external_output(
                selected,
                repo_root=REPO_ROOT,
                canonical_paths=(PIN_REL,),
                operation="emit artifact pin",
            )
            _atomic_emit(destination, payload)
        else:
            try:
                actual = selected.read_bytes()
            except FileNotFoundError as exc:
                raise MissingTreeInput(
                    f"missing check input: {selected}"
                ) from exc
            decoded = json.loads(actual)
            validate_pin(decoded)
            if actual != canonical_json_bytes(decoded) or actual != payload:
                raise PinMismatch("artifact pin bytes differ from selected tree")
            validate_local_artifact(decoded)
    except MissingTreeInput as exc:
        print(f"metric_extractor_artifact_pin: prerequisite missing: {exc}", file=sys.stderr)
        return 2
    except (PinMismatch, ProtectedPathError, OSError, json.JSONDecodeError) as exc:
        print(f"metric_extractor_artifact_pin: mismatch: {exc}", file=sys.stderr)
        return 1
    return 0


def _worktree_document() -> dict:
    payload = (REPO_ROOT / PIN_REL).read_bytes()
    document = json.loads(payload)
    validate_pin(document)
    assert payload == canonical_json_bytes(document)
    return document


def test_artifact_pin_is_closed_and_canonical() -> None:
    document = _worktree_document()
    assert len(document) == 14
    assert len(document["source_closure"]) == 21


def test_source_closure_matches_current_bytes() -> None:
    document = _worktree_document()
    for row in document["source_closure"]:
        assert _sha((REPO_ROOT / row["path"]).read_bytes()) == row["sha256"]


def test_sdk_manifest_and_lock_bindings_match_current_bytes() -> None:
    document = _worktree_document()
    sdk = (REPO_ROOT / SDK_MANIFEST_REL).read_bytes()
    lock = (REPO_ROOT / LOCK_REL).read_bytes()
    _validate_lock(lock)
    assert document["sdk_manifest_sha256"] == _sha(sdk)
    assert document["scratch_witness"]["pb_lock_sha256"] == _sha(lock)


def test_local_artifact_and_optional_stamp() -> None:
    validate_local_artifact(_worktree_document())


def test_emit_refuses_canonical_worktree_path() -> None:
    with pytest.raises(ProtectedPathError):
        assert_external_output(
            REPO_ROOT / PIN_REL,
            repo_root=REPO_ROOT,
            canonical_paths=(PIN_REL,),
            operation="emit artifact pin",
        )


def test_emit_validates_present_local_artifact_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    document = _worktree_document()
    monkeypatch.setattr(
        sys.modules[__name__], "generate_pin", lambda _tree: document,
    )

    def reject(_document: dict) -> None:
        raise PinMismatch("present grandfathered artifact digest mismatch")

    monkeypatch.setattr(sys.modules[__name__], "validate_local_artifact", reject)
    destination = tmp_path / "artifact_pin.json"
    assert main(["--tree", "HEAD", "--emit", str(destination)]) == 1
    assert not destination.exists()


def test_missing_check_input_is_prerequisite_exit_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    document = _worktree_document()
    monkeypatch.setattr(
        sys.modules[__name__], "generate_pin", lambda _tree: document,
    )
    missing = tmp_path / "missing-artifact-pin.json"
    assert main(["--tree", "HEAD", "--check", str(missing)]) == 2


def test_metric_make_default_and_read_only_goal_are_explicit() -> None:
    makefile = (
        REPO_ROOT / "tools/nvbit_tracer/metric_extractor/Makefile"
    ).read_text(encoding="utf-8")
    assert ".DEFAULT_GOAL := all" in makefile
    assert "filter-out verify-libnvbit" in makefile


if __name__ == "__main__":
    raise SystemExit(main())
