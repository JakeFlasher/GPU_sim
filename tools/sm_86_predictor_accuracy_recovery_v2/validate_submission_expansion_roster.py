"""Validator for the sm_86 submission expansion roster.

The roster is a preflight planning artifact: it proves the candidate
pool is wide enough before any local-label sweep is launched. It does
not check NCU labels, binary provenance, or trace completeness; those
belong to the later candidate-preflight and live-admission tools.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

try:
    from publication_gate import (  # noqa: E402
        DEPTH_FIRST_MIN_SHAPE_POINTS,
        DEPTH_FIRST_REQUIRED_SLUGS,
        FAMILY_BUCKET_MINIMA,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - host env fallback
    if exc.name != "jsonschema":
        raise
    FAMILY_BUCKET_MINIMA = {
        "dl_primitives": 4,
        "gemm_family": 4,
        "attention_conv": 2,
        "memory_stress": 3,
        "irregular_hostile": 4,
        "vendor_baselines": 1,
    }
    DEPTH_FIRST_REQUIRED_SLUGS = 6
    DEPTH_FIRST_MIN_SHAPE_POINTS = 3

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROSTER_PATH = (
    REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "submission_expansion_roster.json"
)
PORTABLE_LCD_ROOT = REPO_ROOT / "workloads" / "portable_lcd"
PORTABLE_LCD_ROSTER_PATH = PORTABLE_LCD_ROOT / "roster.toml"
LCD_ARCH_GATE_EXPANSION = ("sm_86", "sm_90", "sm_120")
ARCH_GATE_PATTERN = re.compile(r"^\s*ARCH_GATE\s*[:?+]?=\s*(.+?)\s*(?:#.*)?$")
SOURCE_ARCH_GATE_PATTERN = re.compile(r"^\s*//\s*arch_gate:\s*(.+?)\s*$")
SOURCE_ARCH_GATE_SCAN_LINES = 40

MIN_PRIMARY_CELLS = 42
MIN_PRIMARY_UNIQUE_SLUGS = 35
MIN_REPLACEMENT_CELLS = 10

ALLOWED_ROLES = {"primary", "replacement"}
ALLOWED_COMPATIBILITY = {"compatible", "typed_excluded"}
REQUIRED_SOURCE_PREFIXES = (
    "workloads/portable_lcd/",
    "workloads/simbench_bw/l1/",
    "workloads/simbench_bw/shapes/",
)
FORBIDDEN_SM86_TOKENS = (
    "wgmma",
    "tma",
    "cluster",
    "dsmem",
    "nvfp4",
    "fp4",
    "fp6",
    "sm_90a",
    "sm_120a",
    "sm_120f",
)

TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_kind",
    "scope",
    "generated_on",
    "source_artifacts",
    "thresholds",
    "rows",
}
THRESHOLD_FIELDS = {
    "min_primary_cells",
    "min_primary_unique_slugs",
    "min_replacement_cells",
    "family_bucket_minima",
    "depth_first_required_slugs",
    "depth_first_min_shape_points",
}
ROW_FIELDS = {
    "role",
    "slug",
    "shape_point",
    "family_bucket",
    "workload_root",
    "shape_source",
    "sm86_compatibility_status",
    "unsupported_feature_reason",
    "replacement_for",
}


class SubmissionExpansionRosterError(Exception):
    """Base class for roster validation failures."""


class SubmissionExpansionRosterSchemaError(SubmissionExpansionRosterError):
    """Raised when JSON shape validation fails."""


class SubmissionExpansionRosterGateError(SubmissionExpansionRosterError):
    """Raised when the roster is well-formed but fails a planning gate."""


@dataclass(frozen=True)
class RosterValidationSummary:
    primary_cells: int
    primary_unique_slugs: int
    replacement_cells: int
    per_family_primary_unique_slugs: Mapping[str, int]
    depth_first_slugs: tuple[str, ...]


def _repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _arch_gate_tokens(raw_gate: str) -> tuple[str, ...]:
    return tuple(token.strip("'\"") for token in raw_gate.split() if token.strip("'\""))


def _read_makefile_arch_gate(
    workload_root: Path,
    workload_root_text: str,
) -> tuple[str, ...]:
    makefile = workload_root / "Makefile"
    if not makefile.is_file():
        _raise_gate(f"workload_makefile_missing: {workload_root_text}")
    for line in makefile.read_text().splitlines():
        match = ARCH_GATE_PATTERN.match(line)
        if match:
            return _arch_gate_tokens(match.group(1))
    _raise_gate(f"workload_arch_gate_missing: {workload_root_text}")
    raise AssertionError("unreachable")


def _read_source_arch_gates(
    workload_root: Path,
) -> tuple[tuple[Path, int, tuple[str, ...]], ...]:
    source_gates: list[tuple[Path, int, tuple[str, ...]]] = []
    for source_path in sorted(workload_root.glob("*.cu")):
        lines = source_path.read_text().splitlines()
        for line_number, line in enumerate(lines[:SOURCE_ARCH_GATE_SCAN_LINES], start=1):
            match = SOURCE_ARCH_GATE_PATTERN.match(line)
            if match:
                source_gates.append(
                    (source_path, line_number, _arch_gate_tokens(match.group(1)))
                )
    return tuple(source_gates)


@lru_cache(maxsize=1)
def _portable_lcd_roster_arch_gates() -> dict[str, tuple[str, ...]]:
    if not PORTABLE_LCD_ROSTER_PATH.is_file():
        return {}
    payload = tomllib.loads(PORTABLE_LCD_ROSTER_PATH.read_text())
    gates: dict[str, tuple[str, ...]] = {}
    for row in payload.get("rows", []):
        if row.get("status") != "active":
            continue
        slug = row.get("slug")
        arch_gate = row.get("arch_gate", [])
        if isinstance(slug, str) and isinstance(arch_gate, list):
            gates[slug] = tuple(str(token) for token in arch_gate)
    return gates


def _expanded_arch_gate(
    row: Mapping[str, Any],
    workload_root: Path,
    literal_gate: tuple[str, ...],
) -> tuple[str, ...]:
    expanded: list[str] = []
    is_portable_lcd = _is_relative_to(workload_root, PORTABLE_LCD_ROOT)
    for token in literal_gate:
        if token == "lcd" and is_portable_lcd:
            expanded.extend(LCD_ARCH_GATE_EXPANSION)
        else:
            expanded.append(token)
    if is_portable_lcd:
        expanded.extend(_portable_lcd_roster_arch_gates().get(str(row["slug"]), ()))
    return tuple(dict.fromkeys(expanded))


def _validate_source_arch_gates(
    row: Mapping[str, Any],
    workload_root: Path,
    expanded_gate: tuple[str, ...],
) -> None:
    if not _is_relative_to(workload_root, REPO_ROOT / "workloads" / "simbench_bw"):
        return
    expected = set(expanded_gate)
    for source_path, line_number, source_gate in _read_source_arch_gates(workload_root):
        relative_source = source_path.relative_to(REPO_ROOT)
        if "sm_86" not in source_gate:
            _raise_gate(
                f"source_arch_gate_excludes_sm86: {row['slug']} "
                f"{relative_source}:{line_number} gate={source_gate!r}"
            )
        if set(source_gate) != expected:
            _raise_gate(
                f"source_arch_gate_disagrees_with_makefile: {row['slug']} "
                f"{relative_source}:{line_number} source={source_gate!r} "
                f"makefile={expanded_gate!r}"
            )


def _validate_workload_arch_gate(row: Mapping[str, Any]) -> None:
    workload_root_text = str(row["workload_root"])
    workload_root = _repo_path(workload_root_text)
    if workload_root.name != row["slug"]:
        _raise_gate(
            f"workload_root_not_slug_specific: {row['slug']} -> {workload_root_text}"
        )
    literal_gate = _read_makefile_arch_gate(workload_root, workload_root_text)
    expanded_gate = _expanded_arch_gate(row, workload_root, literal_gate)
    if "sm_86" not in expanded_gate:
        _raise_gate(
            f"workload_arch_gate_excludes_sm86: {row['slug']} "
            f"gate={literal_gate!r} expanded={expanded_gate!r}"
        )
    _validate_source_arch_gates(row, workload_root, expanded_gate)


def _forbidden_token(row: Mapping[str, Any]) -> str | None:
    haystack = " ".join(
        str(row.get(field, "")).lower()
        for field in ("slug", "shape_point", "workload_root", "shape_source")
    )
    word_tokens = set(re.split(r"[^a-z0-9]+", haystack))
    for token in FORBIDDEN_SM86_TOKENS:
        if token.startswith("sm_"):
            if token in haystack:
                return token
        elif token in word_tokens:
            return token
    return None


def _raise_gate(message: str) -> None:
    raise SubmissionExpansionRosterGateError(message)


def _validate_schema(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise SubmissionExpansionRosterSchemaError("top-level payload must be an object")
    missing = TOP_LEVEL_FIELDS - set(payload)
    extra = set(payload) - TOP_LEVEL_FIELDS
    if missing or extra:
        raise SubmissionExpansionRosterSchemaError(
            f"top-level field mismatch: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    if payload["schema_version"] != 1:
        raise SubmissionExpansionRosterSchemaError("schema_version must be 1")
    if payload["artifact_kind"] != "sm86_submission_expansion_roster":
        raise SubmissionExpansionRosterSchemaError("unexpected artifact_kind")
    if payload["scope"] != "sm_86_only":
        raise SubmissionExpansionRosterSchemaError("scope must be sm_86_only")
    if not isinstance(payload["source_artifacts"], list) or not payload["source_artifacts"]:
        raise SubmissionExpansionRosterSchemaError("source_artifacts must be a non-empty list")
    if not all(isinstance(item, str) and item for item in payload["source_artifacts"]):
        raise SubmissionExpansionRosterSchemaError("source_artifacts entries must be strings")

    thresholds = payload["thresholds"]
    if not isinstance(thresholds, dict):
        raise SubmissionExpansionRosterSchemaError("thresholds must be an object")
    missing_thresholds = THRESHOLD_FIELDS - set(thresholds)
    extra_thresholds = set(thresholds) - THRESHOLD_FIELDS
    if missing_thresholds or extra_thresholds:
        raise SubmissionExpansionRosterSchemaError(
            "threshold field mismatch: "
            f"missing={sorted(missing_thresholds)!r} extra={sorted(extra_thresholds)!r}"
        )
    integer_floors = {
        "min_primary_cells": MIN_PRIMARY_CELLS,
        "min_primary_unique_slugs": MIN_PRIMARY_UNIQUE_SLUGS,
        "min_replacement_cells": MIN_REPLACEMENT_CELLS,
        "depth_first_required_slugs": DEPTH_FIRST_REQUIRED_SLUGS,
        "depth_first_min_shape_points": DEPTH_FIRST_MIN_SHAPE_POINTS,
    }
    for field, floor in integer_floors.items():
        value = thresholds[field]
        if not isinstance(value, int) or value < floor:
            raise SubmissionExpansionRosterSchemaError(
                f"{field} must be an integer >= {floor}"
            )
    if not isinstance(thresholds["family_bucket_minima"], dict):
        raise SubmissionExpansionRosterSchemaError(
            "family_bucket_minima must be an object"
        )

    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise SubmissionExpansionRosterSchemaError("rows must be a non-empty list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SubmissionExpansionRosterSchemaError(f"rows[{index}] must be an object")
        missing_row_fields = ROW_FIELDS - set(row)
        extra_row_fields = set(row) - ROW_FIELDS
        if missing_row_fields or extra_row_fields:
            raise SubmissionExpansionRosterSchemaError(
                f"row field mismatch at {index}: "
                f"missing={sorted(missing_row_fields)!r} "
                f"extra={sorted(extra_row_fields)!r}"
            )
        if row["role"] not in ALLOWED_ROLES:
            raise SubmissionExpansionRosterSchemaError(f"invalid role at row {index}")
        if row["family_bucket"] not in FAMILY_BUCKET_MINIMA:
            raise SubmissionExpansionRosterSchemaError(
                f"invalid family_bucket at row {index}"
            )
        if row["sm86_compatibility_status"] not in ALLOWED_COMPATIBILITY:
            raise SubmissionExpansionRosterSchemaError(
                f"invalid sm86_compatibility_status at row {index}"
            )
        for field in ("slug", "shape_point", "workload_root", "shape_source"):
            if not isinstance(row[field], str) or not row[field]:
                raise SubmissionExpansionRosterSchemaError(
                    f"{field} must be a non-empty string at row {index}"
                )
        for field in ("unsupported_feature_reason", "replacement_for"):
            if row[field] is not None and not isinstance(row[field], str):
                raise SubmissionExpansionRosterSchemaError(
                    f"{field} must be a string or null at row {index}"
                )


def _validate_thresholds(payload: Mapping[str, Any]) -> None:
    minima = payload["thresholds"]["family_bucket_minima"]
    if dict(minima) != dict(FAMILY_BUCKET_MINIMA):
        _raise_gate(
            "family_bucket_minima must match publication_gate.FAMILY_BUCKET_MINIMA"
        )


def _validate_row_contract(rows: list[Mapping[str, Any]]) -> None:
    seen: Counter[tuple[str, str, str]] = Counter()
    source_roots_seen: set[str] = set()
    for row in rows:
        role = row["role"]
        key = (role, row["slug"], row["shape_point"])
        seen[key] += 1
        if seen[key] > 1:
            _raise_gate(f"duplicate roster row: {key!r}")

        workload_root = str(row["workload_root"])
        shape_source = str(row["shape_source"])
        if not _repo_path(workload_root).exists():
            _raise_gate(f"workload_root_missing: {workload_root}")
        if not _repo_path(shape_source).exists():
            _raise_gate(f"shape_source_missing: {shape_source}")
        for prefix in REQUIRED_SOURCE_PREFIXES:
            if workload_root.startswith(prefix) or shape_source.startswith(prefix):
                source_roots_seen.add(prefix)

        token = _forbidden_token(row)
        status = row["sm86_compatibility_status"]
        reason = row["unsupported_feature_reason"]
        if token and status != "typed_excluded":
            _raise_gate(
                f"unsupported_sm86_candidate_promoted: {row['slug']} contains {token}"
            )
        if status == "typed_excluded" and not reason:
            _raise_gate(f"typed_exclusion_missing_reason: {row['slug']}")
        if status == "compatible" and reason is not None:
            _raise_gate(f"compatible_row_has_unsupported_reason: {row['slug']}")
        if status == "compatible":
            _validate_workload_arch_gate(row)
        if role == "primary" and status != "compatible":
            _raise_gate(f"primary_row_not_compatible: {row['slug']}")
        if role == "replacement" and not row["replacement_for"]:
            _raise_gate(f"replacement_missing_target: {row['slug']}")

    missing_sources = tuple(
        prefix for prefix in REQUIRED_SOURCE_PREFIXES if prefix not in source_roots_seen
    )
    if missing_sources:
        _raise_gate(f"required_source_roots_missing: {missing_sources!r}")


def _summarize(payload: Mapping[str, Any]) -> RosterValidationSummary:
    thresholds = payload["thresholds"]
    rows = list(payload["rows"])
    primary = [
        row for row in rows
        if row["role"] == "primary"
        and row["sm86_compatibility_status"] == "compatible"
    ]
    replacements = [
        row for row in rows
        if row["role"] == "replacement"
        and row["sm86_compatibility_status"] == "compatible"
    ]

    primary_slugs = {row["slug"] for row in primary}
    if len(primary) < thresholds["min_primary_cells"]:
        _raise_gate(
            f"primary_cell_count_below_floor: {len(primary)} < "
            f"{thresholds['min_primary_cells']}"
        )
    if len(primary_slugs) < thresholds["min_primary_unique_slugs"]:
        _raise_gate(
            f"primary_unique_slug_count_below_floor: {len(primary_slugs)} < "
            f"{thresholds['min_primary_unique_slugs']}"
        )
    if len(replacements) < thresholds["min_replacement_cells"]:
        _raise_gate(
            f"replacement_cell_count_below_floor: {len(replacements)} < "
            f"{thresholds['min_replacement_cells']}"
        )

    family_slugs: dict[str, set[str]] = {
        bucket: set() for bucket in FAMILY_BUCKET_MINIMA
    }
    slug_shapes: dict[str, set[str]] = defaultdict(set)
    for row in primary:
        family_slugs[row["family_bucket"]].add(row["slug"])
        slug_shapes[row["slug"]].add(row["shape_point"])

    per_family = {
        bucket: len(slugs) for bucket, slugs in family_slugs.items()
    }
    deficient = {
        bucket: count
        for bucket, count in per_family.items()
        if count < FAMILY_BUCKET_MINIMA[bucket]
    }
    if deficient:
        _raise_gate(f"family_bucket_slug_count_below_floor: {deficient!r}")

    depth_first = tuple(
        sorted(
            slug for slug, shapes in slug_shapes.items()
            if len(shapes) >= thresholds["depth_first_min_shape_points"]
        )
    )
    if len(depth_first) < thresholds["depth_first_required_slugs"]:
        _raise_gate(
            f"depth_first_slug_count_below_floor: {len(depth_first)} < "
            f"{thresholds['depth_first_required_slugs']}"
        )

    return RosterValidationSummary(
        primary_cells=len(primary),
        primary_unique_slugs=len(primary_slugs),
        replacement_cells=len(replacements),
        per_family_primary_unique_slugs=per_family,
        depth_first_slugs=depth_first,
    )


def validate_submission_expansion_roster(path: Path) -> RosterValidationSummary:
    payload = json.loads(path.read_text())
    _validate_schema(payload)
    _validate_thresholds(payload)
    rows = list(payload["rows"])
    _validate_row_contract(rows)
    return _summarize(payload)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(DEFAULT_ROSTER_PATH))
    args = parser.parse_args(argv)
    summary = validate_submission_expansion_roster(Path(args.path))
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
