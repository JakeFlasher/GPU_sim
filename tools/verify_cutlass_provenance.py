#!/usr/bin/env python3
"""Verify the checked-in CUTLASS vendor tree against an upstream Git commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_PATH = Path("third_party/cutlass")
DEFAULT_PROVENANCE_PATH = Path("third_party/cutlass.provenance.json")
EXPECTED_URL = "https://github.com/NVIDIA/cutlass.git"
DIGEST_ALGORITHM = "sha256-cutlass-scoped-tree-v1"
_ALLOWED_MODES = {"100644", "100755"}
_VERSION_RE = re.compile(
    (
        rb"^#define CUTLASS_MAJOR (\d+)\s*$.*?"
        rb"^#define CUTLASS_MINOR (\d+)\s*$.*?"
        rb"^#define CUTLASS_PATCH (\d+)\s*$"
    ),
    re.MULTILINE | re.DOTALL,
)


class ProvenanceError(ValueError):
    """A CUTLASS provenance contract is not satisfied."""


def _fail(detail: str) -> None:
    raise ProvenanceError(detail)


def _run_git(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        _fail(f"git invocation failed: {type(exc).__name__}: {exc}")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        _fail(f"git {' '.join(args)} failed: {detail or result.returncode}")
    return result.stdout


def _digest_records(records: Iterable[tuple[str, bytes, bytes]]) -> str:
    """Hash mode, raw POSIX path, and payload records in canonical order."""
    digest = hashlib.sha256()
    for mode, path_bytes, payload in sorted(records, key=lambda item: item[1]):
        if mode not in _ALLOWED_MODES:
            _fail(f"unsupported Git mode {mode!r} for {path_bytes!r}")
        digest.update(mode.encode("ascii"))
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _git_blob_payloads(
    upstream_repo: Path,
    object_ids: Iterable[str],
) -> dict[str, bytes]:
    requested = list(object_ids)
    if not requested:
        return {}
    input_payload = b"".join(object_id.encode("ascii") + b"\n" for object_id in requested)
    try:
        result = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=upstream_repo,
            input=input_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        _fail(f"git blob batch failed: {type(exc).__name__}: {exc}")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        _fail(f"git cat-file --batch failed: {detail or result.returncode}")
    output = result.stdout
    cursor = 0
    payloads: dict[str, bytes] = {}
    for expected_id in requested:
        end_header = output.find(b"\n", cursor)
        if end_header < 0:
            _fail("truncated Git blob batch header")
        header = output[cursor:end_header].split(b" ")
        cursor = end_header + 1
        if len(header) != 3 or header[0].decode("ascii") != expected_id:
            _fail("Git blob batch object identity differs")
        object_type = header[1].decode("ascii")
        if object_type == "missing":
            _fail(f"missing upstream blob {expected_id}")
        if object_type != "blob":
            _fail(f"upstream object is not a blob: {expected_id}")
        try:
            size = int(header[2])
        except ValueError:
            _fail(f"invalid Git blob size for {expected_id}")
        payload = output[cursor : cursor + size]
        if len(payload) != size or output[cursor + size : cursor + size + 1] != b"\n":
            _fail(f"truncated Git blob payload for {expected_id}")
        cursor += size + 1
        payloads[expected_id] = payload
    if cursor != len(output):
        _fail("unexpected trailing Git blob batch output")
    return payloads


def _git_records(
    upstream_repo: Path,
    commit: str,
    scope: tuple[str, ...],
) -> dict[bytes, tuple[str, bytes]]:
    raw = _run_git(
        upstream_repo,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        commit,
        "--",
        *scope,
    )
    records: dict[bytes, tuple[str, bytes]] = {}
    metadata_rows: list[tuple[bytes, str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            metadata, path_bytes = item.split(b"\t", 1)
            mode_bytes, object_type, object_id = metadata.split(b" ", 2)
        except ValueError:
            _fail(f"malformed Git tree record: {item!r}")
        mode = mode_bytes.decode("ascii")
        object_name = object_id.decode("ascii")
        if object_type != b"blob" or mode not in _ALLOWED_MODES:
            _fail(f"unsupported upstream entry: {mode} {object_type.decode()} {path_bytes!r}")
        if path_bytes in records or any(row[0] == path_bytes for row in metadata_rows):
            _fail(f"duplicate upstream path: {path_bytes!r}")
        metadata_rows.append((path_bytes, mode, object_name))
    payloads = _git_blob_payloads(upstream_repo, (row[2] for row in metadata_rows))
    for path_bytes, mode, object_name in metadata_rows:
        records[path_bytes] = (mode, payloads[object_name])
    if not records:
        _fail("upstream scope contains no files")
    return records


def _filesystem_records(
    vendor_root: Path,
    scope: tuple[str, ...],
) -> dict[bytes, tuple[str, bytes]]:
    if vendor_root.is_symlink() or not vendor_root.is_dir():
        _fail(f"vendor path is not a regular directory: {vendor_root}")
    expected_scope = set(scope)
    observed_scope: set[str] = set()
    records: dict[bytes, tuple[str, bytes]] = {}

    def visit(directory: Path, relative_prefix: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: os.fsencode(entry.name))
        except OSError as exc:
            _fail(f"cannot enumerate {directory}: {exc}")
        for entry in entries:
            path = Path(entry.path)
            relative = relative_prefix / entry.name
            if entry.is_symlink():
                _fail(f"symlink in vendor tree: {relative}")
            if entry.is_dir(follow_symlinks=False):
                visit(path, relative)
                continue
            if not entry.is_file(follow_symlinks=False):
                _fail(f"non-regular entry in vendor tree: {relative}")
            path_bytes = os.fsencode(relative.as_posix())
            mode = f"10{stat.S_IMODE(entry.stat(follow_symlinks=False).st_mode):04o}"
            if mode not in _ALLOWED_MODES:
                _fail(f"unsupported vendor mode {mode!r} for {relative}")
            if path_bytes in records:
                _fail(f"duplicate vendor path: {relative}")
            try:
                payload = path.read_bytes()
            except OSError as exc:
                _fail(f"cannot read vendor file {relative}: {exc}")
            records[path_bytes] = (mode, payload)

    for entry in sorted(os.scandir(vendor_root), key=lambda item: os.fsencode(item.name)):
        if entry.is_symlink():
            _fail(f"symlink in vendor tree: {entry.name}")
        if not entry.is_dir(follow_symlinks=False):
            _fail(f"unexpected top-level vendor entry: {entry.name}")
        observed_scope.add(entry.name)
        if entry.name not in expected_scope:
            _fail(f"vendor tree contains out-of-scope directory: {entry.name}")
        visit(Path(entry.path), Path(entry.name))
    if observed_scope != expected_scope:
        _fail(
            "vendor scope differs: "
            f"expected={sorted(expected_scope)} observed={sorted(observed_scope)}"
        )
    if not records:
        _fail("vendor tree contains no files")
    return records


def _validate_version(payload: bytes, expected: tuple[int, int, int]) -> None:
    match = _VERSION_RE.search(payload)
    if match is None:
        _fail("CUTLASS version header is missing or malformed")
    observed = tuple(int(value) for value in match.groups())
    if observed != expected:
        _fail(f"CUTLASS version differs: expected={expected} observed={observed}")


def _strict_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail("provenance must be a JSON object")
    expected_keys = {
        "$id",
        "schema_version",
        "upstream_url",
        "upstream_tag",
        "upstream_commit",
        "upstream_tree",
        "repository_path",
        "materialization_type",
        "scope",
        "version",
        "digest",
    }
    if set(value) != expected_keys:
        _fail("provenance keys differ")
    if value["$id"] != "simbench_bw/third_party/cutlass.provenance.json":
        _fail("provenance $id differs")
    if value["schema_version"] != 1:
        _fail("unsupported provenance schema version")
    if value["upstream_url"] != EXPECTED_URL or value["upstream_tag"] != "v4.6.1":
        _fail("upstream identity differs")
    for field in ("upstream_commit", "upstream_tree"):
        raw = value[field]
        if not isinstance(raw, str) or not re.fullmatch(r"[0-9a-f]{40}", raw):
            _fail(f"invalid {field}")
    if value["repository_path"] != DEFAULT_VENDOR_PATH.as_posix():
        _fail("repository path differs")
    if value["materialization_type"] != "scoped_regular_tracked_tree":
        _fail("materialization type differs")
    scope = value["scope"]
    if (
        not isinstance(scope, list)
        or any(not isinstance(item, str) for item in scope)
        or tuple(scope) != ("examples", "include", "tools")
    ):
        _fail("CUTLASS scope differs")
    version = value["version"]
    if version != {"major": 4, "minor": 6, "patch": 1}:
        _fail("CUTLASS version contract differs")
    digest = value["digest"]
    if not isinstance(digest, dict) or set(digest) != {"algorithm", "sha256"}:
        _fail("digest contract is malformed")
    if digest["algorithm"] != DIGEST_ALGORITHM or not isinstance(digest["sha256"], str):
        _fail("digest algorithm or value differs")
    if not re.fullmatch(r"[0-9a-f]{64}", digest["sha256"]):
        _fail("digest is not lowercase SHA-256")
    return value


def verify(
    *,
    repo_root: Path = REPO_ROOT,
    upstream_repo: Path,
    provenance_path: Path = REPO_ROOT / DEFAULT_PROVENANCE_PATH,
) -> str:
    """Verify and return the deterministic scoped vendor digest."""
    provenance = _strict_provenance(
        json.loads(provenance_path.read_text(encoding="utf-8"))
    )
    commit = str(provenance["upstream_commit"])
    tree = str(provenance["upstream_tree"])
    scope = tuple(str(item) for item in provenance["scope"])
    observed_commit = _run_git(upstream_repo, "rev-parse", commit).decode().strip()
    if observed_commit != commit:
        _fail("upstream commit cannot be resolved exactly")
    observed_tree = _run_git(upstream_repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if observed_tree != tree:
        _fail(f"upstream tree differs: expected={tree} observed={observed_tree}")
    upstream_records = _git_records(upstream_repo, commit, scope)
    vendor_root = repo_root / DEFAULT_VENDOR_PATH
    vendor_records = _filesystem_records(vendor_root, scope)
    if set(upstream_records) != set(vendor_records):
        missing = sorted(set(upstream_records) - set(vendor_records))
        extra = sorted(set(vendor_records) - set(upstream_records))
        _fail(f"vendor file set differs: missing={missing[:3]} extra={extra[:3]}")
    for path_bytes, (upstream_mode, upstream_payload) in upstream_records.items():
        vendor_mode, vendor_payload = vendor_records[path_bytes]
        if vendor_mode != upstream_mode:
            _fail(f"mode differs for {path_bytes!r}: {vendor_mode} != {upstream_mode}")
        if vendor_payload != upstream_payload:
            _fail(f"payload differs for {path_bytes!r}")
    digest = _digest_records(
        (mode, path_bytes, payload)
        for path_bytes, (mode, payload) in upstream_records.items()
    )
    if digest != provenance["digest"]["sha256"]:
        _fail(f"vendor digest differs: expected={provenance['digest']['sha256']} observed={digest}")
    version_payload = vendor_records[os.fsencode("include/cutlass/version.h")][1]
    _validate_version(version_payload, (4, 6, 1))
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=REPO_ROOT / DEFAULT_PROVENANCE_PATH,
    )
    args = parser.parse_args(argv)
    try:
        digest = verify(
            repo_root=args.repo_root.resolve(),
            upstream_repo=args.upstream_repo.resolve(),
            provenance_path=args.provenance.resolve(),
        )
    except (OSError, json.JSONDecodeError, ProvenanceError) as exc:
        print(f"CUTLASS_PROVENANCE_INVALID: {exc}", file=sys.stderr)
        return 1
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
