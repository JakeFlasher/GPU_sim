#!/usr/bin/env python3
"""Validate or emit the exact 53-row NVBit SDK tree manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_external_output,
)

MANIFEST_REL = Path("tools/nvbit_tracer/nvbit_release_manifest.sha256")
SDK_PATHS = (
    "tools/nvbit_tracer/nvbit_release/EULA.txt",
    "tools/nvbit_tracer/nvbit_release/LICENSE",
    "tools/nvbit_tracer/nvbit_release/README.md",
    "tools/nvbit_tracer/nvbit_release/core/cuda.h",
    "tools/nvbit_tracer/nvbit_release/core/generated_cuda_meta.h",
    "tools/nvbit_tracer/nvbit_release/core/instr_types.h",
    "tools/nvbit_tracer/nvbit_release/core/libnvbit.a",
    "tools/nvbit_tracer/nvbit_release/core/nvbit.h",
    "tools/nvbit_tracer/nvbit_release/core/nvbit_reg_rw.h",
    "tools/nvbit_tracer/nvbit_release/core/nvbit_tool.h",
    "tools/nvbit_tracer/nvbit_release/core/tools_cuda_api_meta.h",
    "tools/nvbit_tracer/nvbit_release/core/utils/channel.hpp",
    "tools/nvbit_tracer/nvbit_release/core/utils/utils.h",
    "tools/nvbit_tracer/nvbit_release/test-apps/Makefile",
    "tools/nvbit_tracer/nvbit_release/test-apps/vectoradd/Makefile",
    "tools/nvbit_tracer/nvbit_release/test-apps/vectoradd/vectoradd.cu",
    "tools/nvbit_tracer/nvbit_release/tools/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count/instr_count.cu",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count_bb/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count_bb/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count_bb/instr_count.cu",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count_cuda_graph/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count_cuda_graph/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/instr_count_cuda_graph/instr_count_cuda_graph.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_printf/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/mem_printf/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_printf/mem_printf.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_read_shared/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/mem_read_shared/common.h",
    "tools/nvbit_tracer/nvbit_release/tools/mem_read_shared/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_read_shared/mem_read_shared.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace/common.h",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace/mem_trace.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace/tool_func/flush_channel.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace_tma/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace_tma/common.h",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace_tma/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mem_trace_tma/mem_trace_tma.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mov_replace/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/mov_replace/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/mov_replace/mov_replace.cu",
    "tools/nvbit_tracer/nvbit_release/tools/opcode_hist/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/opcode_hist/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/opcode_hist/opcode_hist.cu",
    "tools/nvbit_tracer/nvbit_release/tools/record_reg_vals/Makefile",
    "tools/nvbit_tracer/nvbit_release/tools/record_reg_vals/common.h",
    "tools/nvbit_tracer/nvbit_release/tools/record_reg_vals/inject_funcs.cu",
    "tools/nvbit_tracer/nvbit_release/tools/record_reg_vals/record_reg_vals.cu",
    "tools/nvbit_tracer/nvbit_release/tools/record_reg_vals/tool_func/flush_channel.cu",
)
_LINE_RE = re.compile(rb"^([0-9a-f]{64})  ([^\r\n]+)$")


class MissingTreeInput(RuntimeError):
    pass


class ManifestMismatch(RuntimeError):
    pass


def _git(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=os.environ.copy(),
    )


def _tree_names(tree: str) -> tuple[str, ...]:
    if tree == "INDEX":
        proc = _git([
            "ls-files", "--stage", "-z", "--",
            "tools/nvbit_tracer/nvbit_release",
        ])
    else:
        proc = _git([
            "ls-tree", "-r", "-z", "HEAD", "--",
            "tools/nvbit_tracer/nvbit_release",
        ])
    if proc.returncode != 0:
        raise MissingTreeInput(proc.stderr.decode(errors="replace"))
    names: list[str] = []
    try:
        for raw in proc.stdout.split(b"\0"):
            if not raw:
                continue
            metadata, encoded_path = raw.split(b"\t", 1)
            fields = metadata.decode("ascii").split()
            path = encoded_path.decode("utf-8")
            mode = fields[0]
            if mode not in {"100644", "100755"}:
                raise ManifestMismatch(f"SDK path is not a regular blob: {path}")
            if tree == "INDEX" and (len(fields) != 3 or fields[2] != "0"):
                raise ManifestMismatch(f"SDK index path has a nonzero stage: {path}")
            if tree == "HEAD" and (len(fields) != 3 or fields[1] != "blob"):
                raise ManifestMismatch(f"SDK tree path is not a blob: {path}")
            names.append(path)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestMismatch("SDK tree entry grammar is invalid") from exc
    return tuple(names)


def _tree_blob(tree: str, path: str) -> bytes:
    spec = f":{path}" if tree == "INDEX" else f"HEAD:{path}"
    proc = _git(["cat-file", "blob", spec])
    if proc.returncode != 0:
        raise MissingTreeInput(f"missing {tree} blob: {path}")
    return proc.stdout


def _safe_path(value: str) -> bool:
    path = PurePosixPath(value)
    raw_parts = value.split("/")
    return bool(
        value and not path.is_absolute() and "\\" not in value
        and not value.endswith("/") and "//" not in value
        and all(part not in {"", ".", ".."} for part in raw_parts)
    )


def parse_manifest(payload: bytes) -> tuple[tuple[str, str], ...]:
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ManifestMismatch("manifest must end in exactly one LF")
    rows: list[tuple[str, str]] = []
    for raw in payload[:-1].split(b"\n"):
        match = _LINE_RE.fullmatch(raw)
        if match is None:
            raise ManifestMismatch("malformed manifest row")
        digest = match.group(1).decode("ascii")
        try:
            path = match.group(2).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestMismatch("manifest path is not UTF-8") from exc
        if not _safe_path(path):
            raise ManifestMismatch(f"unsafe manifest path: {path!r}")
        rows.append((path, digest))
    names = tuple(path for path, _digest in rows)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ManifestMismatch("manifest paths are duplicate or out of order")
    if names != SDK_PATHS:
        raise ManifestMismatch("manifest path set differs from the exact 53-row SDK set")
    return tuple(rows)


def generate_manifest(tree: str) -> bytes:
    names = _tree_names(tree)
    if names != SDK_PATHS:
        raise ManifestMismatch(
            f"{tree} SDK path set differs: expected {len(SDK_PATHS)}, got {len(names)}"
        )
    payload = "".join(
        f"{hashlib.sha256(_tree_blob(tree, path)).hexdigest()}  {path}\n"
        for path in SDK_PATHS
    ).encode("utf-8")
    parse_manifest(payload)
    return payload


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
        expected = generate_manifest(args.tree)
        selected = args.emit if args.emit is not None else args.check
        assert selected is not None
        if args.emit is not None:
            destination = assert_external_output(
                selected,
                repo_root=REPO_ROOT,
                canonical_paths=(MANIFEST_REL,),
                operation="emit SDK manifest",
            )
            _atomic_emit(destination, expected)
        else:
            try:
                actual = selected.read_bytes()
            except FileNotFoundError as exc:
                raise MissingTreeInput(
                    f"missing check input: {selected}"
                ) from exc
            parse_manifest(actual)
            if actual != expected:
                raise ManifestMismatch("manifest bytes differ from selected tree")
    except MissingTreeInput as exc:
        print(f"nvbit_release_manifest: prerequisite missing: {exc}", file=sys.stderr)
        return 2
    except (ManifestMismatch, ProtectedPathError, OSError) as exc:
        print(f"nvbit_release_manifest: mismatch: {exc}", file=sys.stderr)
        return 1
    return 0


def test_sdk_path_set_is_exact() -> None:
    assert len(SDK_PATHS) == len(set(SDK_PATHS)) == 53
    assert SDK_PATHS == tuple(sorted(SDK_PATHS))


def test_worktree_manifest_matches_head_tree() -> None:
    payload = (REPO_ROOT / MANIFEST_REL).read_bytes()
    assert parse_manifest(payload)
    assert payload == generate_manifest("HEAD")


def test_parser_rejects_malformed_or_unsafe_rows() -> None:
    good = generate_manifest("HEAD")
    with pytest.raises(ManifestMismatch):
        parse_manifest(good.rstrip(b"\n"))
    with pytest.raises(ManifestMismatch):
        parse_manifest(good.replace(b"  tools/", b" tools/", 1))
    with pytest.raises(ManifestMismatch):
        parse_manifest(good.replace(b"tools/nvbit", b"tools/../nvbit", 1))


def test_emit_refuses_canonical_worktree_path(tmp_path: Path) -> None:
    assert main(["--tree", "HEAD", "--emit", str(REPO_ROOT / MANIFEST_REL)]) == 1
    destination = tmp_path / "manifest.sha256"
    assert main(["--tree", "HEAD", "--emit", str(destination)]) == 0
    assert destination.read_bytes() == generate_manifest("HEAD")


def test_missing_check_input_is_prerequisite_exit_two(tmp_path: Path) -> None:
    missing = tmp_path / "missing-sdk-manifest.sha256"
    assert main(["--tree", "HEAD", "--check", str(missing)]) == 2


if __name__ == "__main__":
    raise SystemExit(main())
