#!/usr/bin/env python3
"""Read-only byte gate for the 15 portable-LCD preserved binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SOURCE_AUTHORITY_PATH = "data/local_labels/campaign/capture_roster_authority_sm86.json"
SOURCE_AUTHORITY_SHA256 = (
    "d02c817091b0a9f06c2106c0987c59a481910afcc3b78ab1de7a7d5bd5a6b4b4"
)
SOURCE_BASELINE_COMMIT = "c4b76c0ae3b49378f6e3d4cabf7830dbe3eaa432"
DEFAULT_MANIFEST = Path("workloads/portable_lcd/preserved_binaries_manifest.json")
RUNNER_COMMON = "workloads/portable_lcd/runner_common.h"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_PINS = {
    "workloads/portable_lcd/conv2d_implicit_gemm_bf16_sm80/conv2d_implicit_gemm_bf16_sm80_run": "9883fbaf8d08dee40cc5af25bf14b5afdddf17b7d0a6c1f254a91606683a4558",
    "workloads/portable_lcd/cross_entropy_fwd_fp32_sm80/cross_entropy_fwd_fp32_sm80_run": "1ee5a2ea3e35f5b30e13f9e08c7aaa9595d7f22b8a4f0538edbe7db47618fe01",
    "workloads/portable_lcd/embedding_gather_int32_sm80/embedding_gather_int32_sm80_run": "55fb08b61852bee89d102852da8a3ddf55115b38611dce0c201517112c7a2372",
    "workloads/portable_lcd/flash_attn_bwd_bf16_sm80/flash_attn_bwd_bf16_sm80_run": "419a658fbc47da8e4b360c4de6619329c9de84b00885818c12ca60550865acd1",
    "workloads/portable_lcd/flash_attn_fwd_bf16_sm80/flash_attn_fwd_bf16_sm80_run": "acaae23ed7a6705c887bbff159e0a664b941ba6df42f6bdc25b60d6c77009ce1",
    "workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run": "bcf504cd975df9588e75c6726d68888a5a975950a6358b1486b60c0900627acb",
    "workloads/portable_lcd/gemm_hmma_fp16_sm80/gemm_hmma_fp16_sm80_run": "829927c7f57d7c46f0a8e51c7de6bb8c7eb7d8de1e963a33e5e9afee7839a750",
    "workloads/portable_lcd/gemm_imma_int8_sm80/gemm_imma_int8_sm80_run": "44b362cf4d503e1eb4b2791046d2084288b12fc1fc7fde9674a6d538a4b16e4a",
    "workloads/portable_lcd/gemm_sparse_2to4_bf16_sm80/gemm_sparse_2to4_bf16_sm80_run": "abf792ec7d7e7042ca4a4c21ae611577b50b94fad5c1a6ee534b45b13a22e254",
    "workloads/portable_lcd/layernorm_fwd_bf16_sm80/layernorm_fwd_bf16_sm80_run": "edfee9f9588d72a00caf401b2d667ba3dbbb90965fe11acf4d906f7ebc1b33dc",
    "workloads/portable_lcd/rmsnorm_bwd_bf16_sm80/rmsnorm_bwd_bf16_sm80_run": "229b653219e8186632fee1e784ff85dd4b729d59f0881f7a7e484412e2113cba",
    "workloads/portable_lcd/rmsnorm_fwd_bf16_sm80/rmsnorm_fwd_bf16_sm80_run": "2897b18afbd4eafdb517abe9764376697535b2251f69cc4d2fea34461849c4d4",
    "workloads/portable_lcd/silu_fwd_bf16_sm80/silu_fwd_bf16_sm80_run": "48d65e6fda1322f0a6a2f4d4ca67a5ef74c587cf373b52d6acc676288eacd4df",
    "workloads/portable_lcd/softmax_safe_bf16_sm80/softmax_safe_bf16_sm80_run": "e798f2cb4720b46a32c19158c9ce7fc64ae670ee63787cd4fe3bb1b86f8e1782",
    "workloads/portable_lcd/swiglu_fwd_bf16_sm80/swiglu_fwd_bf16_sm80_run": "a9deef77dc5c117e904770ac2f815ba5b87d9273071184a0490301797e4128d8",
}


class GatePrerequisiteError(RuntimeError):
    """The manifest, repository, or a preserved binary is absent."""


class GateValidationError(RuntimeError):
    """Present manifest or binary state violates the closed contract."""


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _safe_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateValidationError(f"{field} must be a nonempty string")
    parts = value.split("/")
    if (
        value.startswith("/")
        or "\\" in value
        or any(part in ("", ".", "..") for part in parts)
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise GateValidationError(f"{field} is not a safe repository path: {value!r}")
    return value


def _exact_keys(value: object, expected: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateValidationError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise GateValidationError(
            f"{field} keys mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        raise GatePrerequisiteError(f"manifest is missing: {path}")
    if path.is_symlink() or not path.is_file():
        raise GateValidationError(f"manifest must be a regular nonsymlink file: {path}")
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GateValidationError(f"manifest is unreadable: {path}: {exc}") from exc
    try:
        value = json.loads(payload, object_pairs_hook=_object_no_duplicates)
    except json.JSONDecodeError as exc:
        raise GateValidationError(f"manifest JSON is invalid: {exc}") from exc
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if payload != canonical:
        raise GateValidationError("manifest bytes are not canonical JSON")
    return _exact_keys(
        value,
        {"schema_version", "source_authority", "source_baseline_commit", "binaries"},
        field="manifest",
    )


def validate_manifest(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value["schema_version"] != 1:
        raise GateValidationError("manifest.schema_version must equal 1")
    if value["source_baseline_commit"] != SOURCE_BASELINE_COMMIT:
        raise GateValidationError("manifest.source_baseline_commit mismatch")
    source = _exact_keys(
        value["source_authority"], {"path", "sha256"}, field="source_authority"
    )
    if source != {"path": SOURCE_AUTHORITY_PATH, "sha256": SOURCE_AUTHORITY_SHA256}:
        raise GateValidationError("manifest.source_authority mismatch")
    rows = value["binaries"]
    if not isinstance(rows, list) or len(rows) != 15:
        raise GateValidationError("manifest.binaries must contain exactly 15 rows")

    observed_paths: list[str] = []
    observed_cells: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _exact_keys(
            raw,
            {"cells", "binary_path", "binary_sha256", "diverged_sources"},
            field=f"binaries[{index}]",
        )
        binary = _safe_path(row["binary_path"], field=f"binaries[{index}].binary_path")
        digest = row["binary_sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise GateValidationError(f"binaries[{index}].binary_sha256 is invalid")
        if EXPECTED_PINS.get(binary) != digest:
            raise GateValidationError(f"unexpected binary path or pin: {binary} {digest}")
        cells = row["cells"]
        if (
            not isinstance(cells, list)
            or not cells
            or any(not isinstance(item, str) or not item for item in cells)
            or cells != sorted(set(cells))
        ):
            raise GateValidationError(f"binaries[{index}].cells must be sorted and unique")
        if observed_cells.intersection(cells):
            raise GateValidationError(f"cell appears in multiple binary rows: {cells}")
        observed_cells.update(cells)
        expected_slug = PurePosixPath(binary).parent.name
        if cells != [expected_slug]:
            raise GateValidationError(
                f"binary {binary} must own exactly its frozen cell {expected_slug}"
            )
        sources = row["diverged_sources"]
        if (
            not isinstance(sources, list)
            or not sources
            or sources != sorted(set(sources))
        ):
            raise GateValidationError(
                f"binaries[{index}].diverged_sources must be nonempty, sorted, unique"
            )
        for source_path in sources:
            _safe_path(source_path, field=f"binaries[{index}].diverged_sources")
        expected_sources = [RUNNER_COMMON]
        if expected_slug == "conv2d_implicit_gemm_bf16_sm80":
            expected_sources.append(
                "workloads/portable_lcd/conv2d_implicit_gemm_bf16_sm80/main.cu"
            )
        if expected_slug == "gemm_hmma_bf16_sm80":
            expected_sources.append(
                "workloads/portable_lcd/gemm_hmma_bf16_sm80/main.cu"
            )
        if sources != sorted(expected_sources):
            raise GateValidationError(
                f"binary {binary} has incorrect source attribution: {sources}"
            )
        observed_paths.append(binary)
        normalized.append(row)

    if observed_paths != sorted(EXPECTED_PINS):
        raise GateValidationError("manifest binary paths are not the exact sorted 15-path set")
    if any("/ac4_mixed_long_sm80/" in item for item in observed_paths):
        raise GateValidationError("ac4 must remain excluded from the portable manifest")
    return normalized


def gate_preserved_binaries(*, repo_root: Path, manifest_path: Path) -> None:
    rows = validate_manifest(load_manifest(manifest_path))
    mismatches: list[str] = []
    missing: list[str] = []
    for row in rows:
        relative = row["binary_path"]
        lexical = repo_root / relative
        try:
            canonical = lexical.resolve(strict=False)
            canonical.relative_to(repo_root)
        except (OSError, RuntimeError, ValueError):
            mismatches.append(f"binary path escapes repository: {relative}")
            continue
        if lexical.is_symlink():
            mismatches.append(f"binary is a symlink: {relative}")
            continue
        if not lexical.exists():
            missing.append(f"binary is missing: {relative}")
            continue
        if not lexical.is_file():
            mismatches.append(f"binary is not a regular file: {relative}")
            continue
        try:
            actual = hashlib.sha256(lexical.read_bytes()).hexdigest()
        except OSError as exc:
            mismatches.append(f"binary is unreadable: {relative}: {exc}")
            continue
        if actual != row["binary_sha256"]:
            mismatches.append(
                f"binary SHA-256 mismatch: {relative}: "
                f"expected={row['binary_sha256']} actual={actual}"
            )
    if mismatches:
        raise GateValidationError("; ".join(sorted(mismatches)))
    if missing:
        raise GatePrerequisiteError("; ".join(sorted(missing)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        repo_root = (
            args.repo_root.resolve(strict=True)
            if args.repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        if not (repo_root / ".git").exists():
            raise GatePrerequisiteError(f"repository root is unavailable: {repo_root}")
        manifest = args.manifest or (repo_root / DEFAULT_MANIFEST)
        if not manifest.is_absolute():
            manifest = repo_root / manifest
        gate_preserved_binaries(repo_root=repo_root, manifest_path=manifest)
    except GatePrerequisiteError as exc:
        print(f"preserved_binary_gate: prerequisite missing: {exc}", file=sys.stderr)
        return 2
    except GateValidationError as exc:
        print(f"preserved_binary_gate: FAIL {exc}", file=sys.stderr)
        return 1
    print("preserved_binary_gate: ok (15/15 pins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
