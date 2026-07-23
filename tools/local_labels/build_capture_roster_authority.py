#!/usr/bin/env python3
"""Generate the durable-v2 sm_86 capture-roster projection.

The protected v1 authority remains the historical campaign authority.  This
generator verifies every source pin and the registered SimBench witness, then
projects v1 without consulting host-local binaries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_simbench_build_witness import (  # noqa: E402
    canonical_json_bytes,
    validate_witness_file,
)

ARCH = "sm_86"
ROSTER_ID = "sm86_capture_roster_authority_durable_v2"
SOURCE_COMMIT = "c4b76c0ae3b49378f6e3d4cabf7830dbe3eaa432"
CANONICAL_OUT_REL = Path(
    "tools/local_labels/authorities/"
    "capture_roster_authority_sm86.durable.v2.json")
SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/"
    "capture_roster_authority.durable.v2.schema.json")
HISTORICAL_REL = Path(
    "data/local_labels/campaign/capture_roster_authority_sm86.json")
REGISTRY_REL = Path("tools/local_labels/witness_compat_registry.json")
WITNESS_REL = Path(
    "data/local_labels/campaign/simbench_build_witness_sm86.json")

HISTORICAL_SHA256 = (
    "d02c817091b0a9f06c2106c0987c59a481910afcc3b78ab1de7a7d5bd5a6b4b4")
FIXED_SOURCE_PINS = {
    "portable_lcd_roster_sha256": (
        "tools/local_labels/repeatability_matrix_roster_full.json",
        "7d075f8f07a14094e475699e3caaac9baea256b0c16b34f8fe911ac17ff6713f"),
    "simbench_build_witness_sha256": (
        WITNESS_REL.as_posix(),
        "d577b3850f5ed04c798b1475d06cd6f29e48f857c0dc5735f923157e26cc7750"),
    "simbench_roster_sha256": (
        "workloads/simbench_bw/roster.toml",
        "d79b1c6b3f1cddb47776405abb503f7c0f8705f7da62bb28719448417ccdbd0e"),
    "sol_port_scope_sha256": (
        "tools/sol_baselines_harness/calibration/sm86_port_scope_v1.toml",
        "22c2792c9b100663afed77b3220944b07b4a6871b51f32fe83c71f40c227baa3"),
    "sol_runnability_sha256": (
        "data/local_labels/campaign/sol_runnability_sm86.json",
        "760853e5916b78ea4e233b5995f6db49d4d8942ec09d240e02f2e072ed8bbe4f"),
}
EXPECTED_COUNTS = {
    "universe": 344,
    "runnable": 122,
    "pending_build": 0,
    "pending_reconstitution": 0,
    "pending_shrink": 0,
    "excluded": 222,
    "by_source": {
        "portable_lcd": {"runnable": 17},
        "simbench_bw": {"runnable": 31, "excluded": 61},
        "sol_execbench": {"runnable": 74, "excluded": 161},
    },
}
REASON_VOCABULARY = frozenset({
    "arch_gate", "semantic_mismatch", "trace_unsupported",
    "repeatability_failed", "correctness_failed",
    "calibration_inconclusive", "host_capacity_deferred",
})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _verify_file(path: Path, expected: str, label: str) -> bytes:
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return payload


def _count_cells(cells: list[dict]) -> dict:
    status_counts = Counter(cell["resolution_status"] for cell in cells)
    source_counts: dict[str, Counter] = {}
    for cell in cells:
        source_counts.setdefault(cell["roster_source"], Counter())[
            cell["resolution_status"]] += 1
    return {
        "universe": len(cells),
        "runnable": status_counts["runnable"],
        "pending_build": status_counts["pending_build"],
        "pending_reconstitution": status_counts["pending_reconstitution"],
        "pending_shrink": status_counts["pending_shrink"],
        "excluded": status_counts["excluded"],
        "by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_counts.items())
        },
    }


def _verify_projection_identity(historical: dict, cells: list[dict]) -> None:
    if historical.get("roster_id") != "sm86_capture_roster_authority_v1":
        raise ValueError("historical roster_id mismatch")
    if historical.get("arch") != ARCH:
        raise ValueError("historical arch mismatch")
    if historical.get("source_commit") != (
            "124ce358733f97ba6e0af808d10ab69ffe4d5a78"):
        raise ValueError("historical source_commit mismatch")
    expected_historical_pins = {
        field: expected for field, (_path, expected) in FIXED_SOURCE_PINS.items()
    }
    if historical.get("source_pins") != expected_historical_pins:
        raise ValueError("historical source_pins mismatch")
    historical_ids = {
        (cell.get("roster_source"), cell.get("slug"))
        for cell in historical.get("cells", [])
    }
    projected_ids = {
        (cell.get("roster_source"), cell.get("slug")) for cell in cells
    }
    if len(projected_ids) != len(cells) or projected_ids != historical_ids:
        raise ValueError("projected cell identity set mismatch")


def build_authority(repo_root: Path = REPO_ROOT) -> dict:
    repo_root = Path(repo_root).resolve()
    historical_bytes = _verify_file(
        repo_root / HISTORICAL_REL, HISTORICAL_SHA256, "historical authority")
    historical = json.loads(historical_bytes)

    source_pins: dict[str, str] = {}
    for field, (relative, expected) in FIXED_SOURCE_PINS.items():
        _verify_file(repo_root / relative, expected, field)
        source_pins[field] = expected

    registry = repo_root / REGISTRY_REL
    registry_bytes = registry.read_bytes()
    registry_doc = json.loads(registry_bytes)
    if registry_bytes != canonical_json_bytes(registry_doc):
        raise ValueError("witness compatibility registry is not canonical JSON")
    witness_problems = validate_witness_file(
        WITNESS_REL, repo_root=repo_root, registry_path=REGISTRY_REL)
    if witness_problems:
        raise ValueError(
            "registered SimBench witness rejected: " + repr(witness_problems))
    source_pins["witness_compat_registry_sha256"] = _sha256_bytes(registry_bytes)

    cells: list[dict] = []
    for source in historical.get("cells", []):
        cell = copy.deepcopy(source)
        cell.pop("binary_available_now", None)
        cell.pop("binary_sha256_matches_now", None)
        if cell.get("resolution_status") == "runnable":
            cell["binary_present"] = True
            if cell.get("roster_source") == "sol_execbench":
                cell["binary_sha256"] = None
        cells.append(cell)
    cells.sort(key=lambda cell: (cell["roster_source"], cell["slug"]))
    _verify_projection_identity(historical, cells)
    counts = _count_cells(cells)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"durable count mismatch: {counts!r}")

    authority = {
        "schema_version": 2,
        "roster_id": ROSTER_ID,
        "arch": ARCH,
        "resolution_semantics": "durable",
        "accounting_identity_holds": True,
        "source_commit": SOURCE_COMMIT,
        "historical_authority": {
            "path": HISTORICAL_REL.as_posix(),
            "roster_id": "sm86_capture_roster_authority_v1",
            "sha256": HISTORICAL_SHA256,
        },
        "source_pins": source_pins,
        "counts": counts,
        "cells": cells,
    }
    schema = json.loads((repo_root / SCHEMA_REL).read_text())
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(authority),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise ValueError(f"durable-v2 schema rejection: {errors[0].message}")
    return authority


def _resolve_output(value: str | None, repo_root: Path) -> Path:
    if value is None:
        return repo_root / CANONICAL_OUT_REL
    path = Path(value)
    path = path if path.is_absolute() else repo_root / path
    resolved = path.resolve(strict=False)
    canonical = (repo_root / CANONICAL_OUT_REL).resolve(strict=False)
    if resolved == canonical:
        return resolved
    protected = [
        repo_root / ".humanize", repo_root / "data/annex",
        repo_root / "data/local_labels/campaign",
        repo_root / "tools/sol_execbench/data", repo_root / "references",
        repo_root / ".toolchain", repo_root / "evidence/round28",
        repo_root / "workloads/portable_lcd/results",
    ]
    if any(resolved == root.resolve() or root.resolve() in resolved.parents
           for root in protected):
        raise ValueError(f"--out enters protected root: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        authority = build_authority(REPO_ROOT)
        output = _resolve_output(args.out, REPO_ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(authority))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"build_capture_roster_authority: failed: {exc}", file=sys.stderr)
        return 2
    counts = authority["counts"]
    print(
        "durable-v2 roster authority: "
        f"universe={counts['universe']} runnable={counts['runnable']} "
        f"excluded={counts['excluded']} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
