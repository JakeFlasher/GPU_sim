#!/usr/bin/env python3
"""Build a current-HEAD live v1 authority for an external capture session.

This is deliberately separate from the durable-v2 projection.  The durable
producer remains the host-independent roster authority; this producer starts
from its SHA-verified frozen-v1 projection, re-verifies every runnable binary
or solution against the frozen v1 identity, and stamps the exact current Git
HEAD for consumption by ``build_sm86_capture_matrix.py --authority``.

The output is session-local evidence.  ``--out`` is restricted to the exact
HEAD-versioned ``inputs`` path under the external campaign root, and an
existing path is never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tools.local_labels import build_capture_roster_authority as durable  # noqa: E402
from tools.local_labels.build_simbench_build_witness import (  # noqa: E402
    canonical_json_bytes,
)
ARCH = "sm_86"
SKU = "rtx_3090_ti"
ROSTER_ID = "sm86_capture_roster_authority_v1"
CAPTURE_ROOT = Path("/home/jakeshea/sm86_reference_capture_r27")
SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/capture_roster_authority.schema.json"
)
HISTORICAL_REL = durable.HISTORICAL_REL
EXPECTED_COUNTS = durable.EXPECTED_COUNTS

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_LOCAL_ENV = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_WORK_TREE",
)


class LiveAuthorityError(ValueError):
    """A frozen identity, live artifact, or output invariant failed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in _GIT_LOCAL_ENV:
        env.pop(name, None)
    return env


def _git_head(repo_root: Path) -> str:
    root = Path(repo_root).resolve(strict=True)
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_git_env(),
        )
    except OSError as exc:
        raise LiveAuthorityError(f"cannot execute Git to resolve HEAD: {exc}") from exc
    head = completed.stdout.strip()
    if completed.returncode != 0 or not _HEX40.fullmatch(head):
        detail = completed.stderr.strip() or repr(head)
        raise LiveAuthorityError(f"cannot resolve an exact 40-hex HEAD: {detail}")
    return head


def _count_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(cell.get("resolution_status") for cell in cells)
    by_source: dict[str, Counter[str]] = {}
    for cell in cells:
        source = cell.get("roster_source")
        status = cell.get("resolution_status")
        if not isinstance(source, str) or not isinstance(status, str):
            raise LiveAuthorityError("cell source/status identity is invalid")
        by_source.setdefault(source, Counter())[status] += 1
    return {
        "universe": len(cells),
        "runnable": statuses["runnable"],
        "pending_build": statuses["pending_build"],
        "pending_reconstitution": statuses["pending_reconstitution"],
        "pending_shrink": statuses["pending_shrink"],
        "excluded": statuses["excluded"],
        "by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(by_source.items())
        },
    }


def _identity(cell: dict[str, Any]) -> tuple[str, str]:
    source = cell.get("roster_source")
    slug = cell.get("slug")
    if not isinstance(source, str) or not isinstance(slug, str) or not slug:
        raise LiveAuthorityError(f"invalid cell identity: {(source, slug)!r}")
    return source, slug


def _durable_form(cell: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(cell)
    projected.pop("binary_available_now", None)
    projected.pop("binary_sha256_matches_now", None)
    if projected.get("resolution_status") == "runnable":
        projected["binary_present"] = True
        if projected.get("roster_source") == "sol_execbench":
            projected["binary_sha256"] = None
    return projected


def _verified_runnable_identity(
    cell: dict[str, Any], *, repo_root: Path,
) -> str:
    """Return the current SHA after a safe-path and frozen-identity check."""

    ident = _identity(cell)
    relative = cell.get("binary")
    expected = cell.get("binary_sha256")
    if not isinstance(relative, str) or not relative:
        raise LiveAuthorityError(f"runnable cell lacks a binary path: {ident!r}")
    if (
        relative.startswith("/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise LiveAuthorityError(
            f"runnable binary path is not safe repo-relative POSIX: {relative!r}"
        )
    if not isinstance(expected, str) or not _HEX64.fullmatch(expected):
        raise LiveAuthorityError(f"runnable cell lacks a frozen SHA-256: {ident!r}")

    root = Path(repo_root).resolve(strict=True)
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise LiveAuthorityError(
            f"runnable binary is absent or escapes the repository: {relative}"
        ) from exc
    if not path.is_file():
        raise LiveAuthorityError(f"runnable binary is not a regular file: {relative}")
    if ident[0] != "sol_execbench" and not os.access(path, os.X_OK):
        raise LiveAuthorityError(
            f"runnable direct binary is not executable: {relative}"
        )
    actual = _sha256_file(path)
    if actual != expected:
        raise LiveAuthorityError(
            f"runnable binary identity mismatch: {relative} "
            f"expected={expected} actual={actual}"
        )
    return actual


def _validate_v1_schema(authority: dict[str, Any], repo_root: Path) -> None:
    schema_path = Path(repo_root) / SCHEMA_REL
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveAuthorityError(f"cannot load v1 authority schema: {exc}") from exc
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(authority),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        path = "/".join(str(part) for part in errors[0].absolute_path)
        raise LiveAuthorityError(
            f"live-v1 schema rejection at /{path}: {errors[0].message}"
        )


def build_authority(
    repo_root: Path = REPO_ROOT, *, generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build, but do not write, a current live-v1 execution authority."""

    root = Path(repo_root).resolve(strict=True)
    try:
        durable_authority = durable.build_authority(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LiveAuthorityError(
            f"durable projection/source verification failed: {exc}"
        ) from exc

    historical_path = root / HISTORICAL_REL
    try:
        historical_bytes = historical_path.read_bytes()
        historical = json.loads(historical_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveAuthorityError(f"cannot load frozen v1 authority: {exc}") from exc
    actual_historical_sha = _sha256_bytes(historical_bytes)
    if actual_historical_sha != durable.HISTORICAL_SHA256:
        raise LiveAuthorityError(
            "frozen v1 authority SHA-256 mismatch: "
            f"{actual_historical_sha} != {durable.HISTORICAL_SHA256}"
        )
    if not isinstance(historical, dict):
        raise LiveAuthorityError("frozen v1 authority root is not an object")
    if historical.get("roster_id") != ROSTER_ID:
        raise LiveAuthorityError("frozen v1 roster_id mismatch")
    if historical.get("arch") != ARCH or historical.get("sku") != SKU:
        raise LiveAuthorityError("frozen v1 arch/sku mismatch")

    historical_cells = historical.get("cells")
    durable_cells = durable_authority.get("cells")
    if not isinstance(historical_cells, list) or not isinstance(durable_cells, list):
        raise LiveAuthorityError("frozen/durable cell arrays are absent")
    durable_by_id = {_identity(cell): cell for cell in durable_cells}
    if len(durable_by_id) != len(durable_cells):
        raise LiveAuthorityError("durable projection contains duplicate identities")

    live_cells: list[dict[str, Any]] = []
    verified = 0
    for frozen_cell in historical_cells:
        ident = _identity(frozen_cell)
        projected = durable_by_id.get(ident)
        if projected is None or _durable_form(frozen_cell) != projected:
            raise LiveAuthorityError(
                f"frozen v1 and durable projection diverge at {ident!r}"
            )
        live_cell = copy.deepcopy(frozen_cell)
        if live_cell.get("resolution_status") == "runnable":
            _verified_runnable_identity(live_cell, repo_root=root)
            live_cell["binary_present"] = True
            live_cell["binary_available_now"] = True
            if "binary_sha256_matches_now" in live_cell:
                live_cell["binary_sha256_matches_now"] = True
            verified += 1
        live_cells.append(live_cell)
    live_cells.sort(key=_identity)

    if len(live_cells) != len(durable_by_id):
        raise LiveAuthorityError("frozen/durable cell identity cardinality mismatch")
    counts = _count_cells(live_cells)
    if counts != EXPECTED_COUNTS or historical.get("counts") != EXPECTED_COUNTS:
        raise LiveAuthorityError(
            f"live-v1 count mismatch: observed={counts!r} expected={EXPECTED_COUNTS!r}"
        )
    if verified != EXPECTED_COUNTS["runnable"]:
        raise LiveAuthorityError(
            f"live runnable verification count mismatch: {verified}"
        )

    authority = copy.deepcopy(historical)
    authority["schema_version"] = 1
    authority["roster_id"] = ROSTER_ID
    authority["arch"] = ARCH
    authority["sku"] = SKU
    authority["source_commit"] = _git_head(root)
    authority["generated_at_utc"] = generated_at_utc or datetime.now(
        timezone.utc
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    authority["counts"] = counts
    authority["accounting_identity_holds"] = True
    authority["cells"] = live_cells
    _validate_v1_schema(authority, root)
    return authority


def _resolve_output(
    value: str,
    *,
    expected_head: str,
    repo_root: Path = REPO_ROOT,
    capture_root: Path | None = None,
) -> Path:
    if not _HEX40.fullmatch(expected_head):
        raise LiveAuthorityError("--expected-head must be exact lowercase 40-hex")
    requested = Path(value)
    if not requested.is_absolute():
        raise LiveAuthorityError("--out must be an explicit absolute campaign path")
    root = Path(capture_root or CAPTURE_ROOT)
    if not root.is_absolute():
        raise LiveAuthorityError("campaign root must be an explicit absolute path")
    resolved_root = root.resolve(strict=True)
    if root != resolved_root or root.is_symlink() or not root.is_dir():
        raise LiveAuthorityError(
            f"campaign root must be a canonical non-symlink directory: {root}"
        )
    target = requested
    expected = (
        root
        / "inputs"
        / f"r28_{expected_head[:8]}"
        / "capture_roster_authority_sm86.json"
    )
    if target != expected:
        raise LiveAuthorityError(
            "--out must be the exact current-HEAD campaign input path "
            f"{expected}; observed {target}"
        )
    if target.is_relative_to(Path(repo_root).resolve(strict=True)):
        raise LiveAuthorityError("campaign authority output cannot be repository-local")
    current = root
    for component in target.parent.relative_to(root).parts:
        current = current / component
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise LiveAuthorityError(
                    f"campaign output ancestor must be a real directory: {current}"
                )
    if target.exists() or target.is_symlink():
        raise LiveAuthorityError(f"refusing to overwrite existing output: {target}")
    return target


def _open_output_parent(
    *, capture_root: Path, expected_head: str,
) -> int:
    """Create/open the exact output parent without following symlinks."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    descriptor = os.open(capture_root, directory_flags)
    try:
        for component in ("inputs", f"r28_{expected_head[:8]}"):
            try:
                os.mkdir(component, mode=0o777, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_exclusive(
    target: Path,
    payload: bytes,
    *,
    expected_head: str,
    repo_root: Path = REPO_ROOT,
    capture_root: Path | None = None,
) -> None:
    checked = _resolve_output(
        str(target),
        expected_head=expected_head,
        repo_root=repo_root,
        capture_root=capture_root,
    )
    root = Path(capture_root or CAPTURE_ROOT)
    parent_fd = _open_output_parent(
        capture_root=root, expected_head=expected_head,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(checked.name, flags, 0o666, dir_fd=parent_fd)
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_fd)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(checked.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    try:
        current_head = _git_head(REPO_ROOT)
        if args.expected_head != current_head:
            raise LiveAuthorityError(
                f"HEAD mismatch: expected={args.expected_head} observed={current_head}"
            )
        output = _resolve_output(
            args.out,
            expected_head=current_head,
            repo_root=REPO_ROOT,
        )
        authority = build_authority(REPO_ROOT)
        final_head = _git_head(REPO_ROOT)
        if authority["source_commit"] != current_head or final_head != current_head:
            raise LiveAuthorityError(
                "HEAD changed while building the live authority: "
                f"start={current_head} authority={authority['source_commit']} "
                f"final={final_head}"
            )
        _write_exclusive(
            output,
            canonical_json_bytes(authority),
            expected_head=current_head,
            repo_root=REPO_ROOT,
        )
    except (
        LiveAuthorityError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"build_sm86_live_capture_authority: failed: {exc}", file=sys.stderr)
        return 2
    counts = authority["counts"]
    print(
        "live-v1 roster authority: "
        f"universe={counts['universe']} runnable={counts['runnable']} "
        f"excluded={counts['excluded']} head={authority['source_commit']} "
        f"-> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
