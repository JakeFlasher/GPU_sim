#!/usr/bin/env python3
"""Fail-closed source and ELF checks for project-owned CUTLASS artifacts.

A CUTLASS successor must be a direct executable whose build inputs do not
reference cuBLAS/cuBLASLt.  This module deliberately checks both the source
contract and the produced ELF: a source-only check cannot catch an accidental
link flag, while an ELF-only check cannot catch a forbidden include or API
that happened not to be reached by the linker.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_LIBRARY_RE = re.compile(r"cublas(?:lt|xt)?", re.IGNORECASE)
FORBIDDEN_LINK_RE = re.compile(r"-l(?:cublas(?:lt|xt)?)\b", re.IGNORECASE)
SOURCE_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cuh", ".cu", ".h", ".hpp", ".mk", ".toml",
})


class CutlassArtifactError(RuntimeError):
    """A source, binary, or dependency contract failed closed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _strip_c_comments(text: str) -> str:
    """Blank C/C++ comments while preserving line numbers and literals."""
    output: list[str] = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "line_comment"
                continue
            if char == "/" and following == "*":
                output.extend((" ", " "))
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                output.append(char)
                index += 1
                state = "string"
                continue
            if char == "'":
                output.append(char)
                index += 1
                state = "char"
                continue
            output.append(char)
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                output.append(char)
                state = "code"
            else:
                output.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "code"
            else:
                output.append("\n" if char == "\n" else " ")
                index += 1
            continue
        # Keep literals intact: a runtime dlopen/string reference must be
        # rejected just like an identifier or linker flag.
        output.append(char)
        if char == "\\" and index + 1 < len(text):
            output.append(text[index + 1])
            index += 2
            continue
        if (state == "string" and char == '"') or (
            state == "char" and char == "'"
        ):
            state = "code"
        index += 1
    if state == "block_comment":
        raise CutlassArtifactError("unterminated C/C++ block comment")
    return "".join(output)


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise CutlassArtifactError(f"{label} is not a regular file: {path}")


def verify_source_inputs(
    paths: Iterable[Path], *, repo_root: Path,
) -> dict[str, Any]:
    """Hash and scan exactly the source/build files used by a successor."""
    root = repo_root.resolve()
    records: dict[str, str] = {}
    violations: list[str] = []
    seen: set[Path] = set()
    for supplied in paths:
        path = Path(supplied)
        if path.is_symlink():
            raise CutlassArtifactError(f"source input is not a regular file: {path}")
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _regular_file(resolved, label="source input")
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise CutlassArtifactError(
                f"source input escapes repository: {path}"
            ) from exc
        if resolved.suffix not in SOURCE_SUFFIXES and resolved.name != "Makefile":
            raise CutlassArtifactError(f"unsupported source input type: {relative}")
        raw_payload = resolved.read_bytes()
        payload = raw_payload.decode("utf-8")
        normalized = _strip_c_comments(payload)
        for line_number, line in enumerate(normalized.splitlines(), start=1):
            if FORBIDDEN_LIBRARY_RE.search(line) or FORBIDDEN_LINK_RE.search(line):
                violations.append(f"{relative}:{line_number}")
        records[relative] = sha256_bytes(raw_payload)
    if not records:
        raise CutlassArtifactError("successor source input set is empty")
    if violations:
        raise CutlassArtifactError(
            "forbidden cuBLAS reference in successor source inputs: "
            + ", ".join(sorted(violations))
        )
    return {
        "files": dict(sorted(records.items())),
        "input_set_sha256": canonical_sha256(dict(sorted(records.items()))),
        "forbidden_library_policy": "reject_cublas_cublaslt_cublasxt_source_and_link_references",
    }


def _run_readelf(binary: Path, arguments: list[str]) -> tuple[bytes, bytes]:
    command = ["readelf", *arguments, str(binary)]
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=120, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CutlassArtifactError(f"readelf failed: {exc}") from exc
    if result.returncode != 0 or result.stderr:
        raise CutlassArtifactError(
            f"readelf contract failed ({' '.join(arguments)}): "
            f"returncode={result.returncode}, stderr={result.stderr[-300:]!r}"
        )
    return result.stdout, result.stderr


def verify_elf_no_cublas(binary: Path) -> dict[str, Any]:
    """Reject direct NEEDED entries and dynamic symbols naming cuBLAS."""
    binary = Path(binary).resolve()
    _regular_file(binary, label="successor binary")
    if not binary.stat().st_mode & 0o111:
        raise CutlassArtifactError(f"successor binary is not executable: {binary}")
    dynamic, _ = _run_readelf(binary, ["--dynamic", "--wide"])
    symbols, _ = _run_readelf(binary, ["--dyn-syms", "--wide"])
    dynamic_text = dynamic.decode("utf-8", errors="strict")
    symbol_text = symbols.decode("utf-8", errors="strict")
    needed = sorted({
        match.group(1)
        for match in re.finditer(r"Shared library:\s*\[([^\]]+)\]", dynamic_text)
    })
    dependency_violations = [name for name in needed if FORBIDDEN_LIBRARY_RE.search(name)]
    symbol_violations = sorted({
        line.strip() for line in symbol_text.splitlines()
        if FORBIDDEN_LIBRARY_RE.search(line)
    })
    if dependency_violations or symbol_violations:
        raise CutlassArtifactError(
            "forbidden cuBLAS ELF dependency or symbol: "
            f"dependencies={dependency_violations!r}, symbols={symbol_violations!r}"
        )
    return {
        "binary_sha256": sha256_file(binary),
        "binary_bytes": binary.stat().st_size,
        "readelf_dynamic_sha256": sha256_bytes(dynamic),
        "readelf_dynamic_bytes": len(dynamic),
        "readelf_dynamic_symbols_sha256": sha256_bytes(symbols),
        "readelf_dynamic_symbols_bytes": len(symbols),
        "needed_libraries": needed,
        "forbidden_dependency_count": 0,
        "forbidden_dynamic_symbol_count": 0,
    }


def verify_cutlass_artifact(
    *, binary: Path, source_inputs: Iterable[Path], repo_root: Path,
) -> dict[str, Any]:
    source = verify_source_inputs(source_inputs, repo_root=repo_root)
    elf = verify_elf_no_cublas(binary)
    return {"source": source, "elf": elf, "pass": True}


__all__ = [
    "CutlassArtifactError", "canonical_sha256", "sha256_bytes", "sha256_file",
    "verify_cutlass_artifact", "verify_elf_no_cublas", "verify_source_inputs",
]
