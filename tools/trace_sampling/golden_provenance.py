"""Fail-closed provenance helpers for the captured real golden set.

The NCU label is validation evidence, not a trace/model input.  This module
only authenticates that evidence and the trace payload consumed by the real
golden-set runner.  Hashes are over the bytes on disk; no CSV or JSON
normalization is applied before hashing.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


NCU_L2_HIT_RATE_METRIC = "lts__t_sector_hit_rate.pct"
PROVENANCE_SCHEMA_VERSION = 1

_NCU_REQUIRED_COLUMNS = {
    "Metric Name",
    "Metric Unit",
    "Metric Value",
    "Grid Size",
}
_DIM3_RE = re.compile(
    r"^\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$"
)
_HASH_CHUNK_BYTES = 1024 * 1024
_AGGREGATE_DOMAIN = b"real-golden-trace-provenance-v1\0"


class GoldenProvenanceError(RuntimeError):
    """An input cannot be admitted into real-golden-set provenance."""


def _read_required_bytes(path: Path, role: str) -> bytes:
    path = Path(path)
    if not path.is_file():
        raise GoldenProvenanceError(f"{role}_missing: {path}")
    payload = path.read_bytes()
    if not payload:
        raise GoldenProvenanceError(f"{role}_empty: {path}")
    return payload


def _expected_dim3(value: Sequence[int]) -> tuple[int, int, int]:
    try:
        grid = tuple(value)
    except TypeError as exc:
        raise GoldenProvenanceError(
            f"ncu_expected_grid_invalid: expected three integers, got {value!r}"
        ) from exc
    if isinstance(value, (str, bytes)) or len(grid) != 3:
        raise GoldenProvenanceError(
            f"ncu_expected_grid_invalid: expected three integers, got {value!r}"
        )
    if any(type(component) is not int or component <= 0 for component in grid):
        raise GoldenProvenanceError(
            f"ncu_expected_grid_invalid: expected three positive integers, got {value!r}"
        )
    return grid  # type: ignore[return-value]


def _parse_ncu_dim3(raw: str) -> tuple[int, int, int]:
    match = _DIM3_RE.fullmatch(raw.strip())
    if match is None:
        raise GoldenProvenanceError(
            f"ncu_grid_malformed: Grid Size value {raw!r} is not '(x, y, z)'"
        )
    grid = tuple(int(component) for component in match.groups())
    if any(component <= 0 for component in grid):
        raise GoldenProvenanceError(
            f"ncu_grid_malformed: Grid Size must be positive, got {grid}"
        )
    return grid  # type: ignore[return-value]


def _ncu_table(text: str) -> tuple[list[str], list[dict[str, str]]]:
    """Find the NCU long-form table after any profiler/target preamble."""
    lines = text.splitlines()
    header_index: int | None = None
    header: list[str] | None = None
    for index, line in enumerate(lines):
        try:
            cells = next(csv.reader([line]))
        except csv.Error:
            continue
        normalized = [cell.strip() for cell in cells]
        if _NCU_REQUIRED_COLUMNS.issubset(normalized):
            header_index = index
            header = normalized
            break
    if header_index is None or header is None:
        raise GoldenProvenanceError(
            "ncu_csv_header_missing: no long-form NCU Metric Name table found"
        )

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    if reader.fieldnames is None:
        raise GoldenProvenanceError("ncu_csv_header_missing: CSV header is empty")
    normalized_names = [name.strip() for name in reader.fieldnames]
    if len(normalized_names) != len(set(normalized_names)):
        raise GoldenProvenanceError(
            f"ncu_csv_header_duplicate_column: {normalized_names}"
        )

    rows: list[dict[str, str]] = []
    for raw_row in reader:
        if None in raw_row:
            raise GoldenProvenanceError(
                "ncu_csv_row_malformed: row has more values than the header"
            )
        row = {
            str(key).strip(): (value or "").strip()
            for key, value in raw_row.items()
        }
        rows.append(row)
    return header, rows


def parse_ncu_l2_csv(
    path: Path,
    *,
    expected_grid: Sequence[int],
    expected_kernel_name: str | None = None,
) -> dict[str, Any]:
    """Parse and authenticate one shape-matched NCU L2 hit-rate label.

    NCU may prefix its quoted CSV header with ``==PROF==``/``==WARNING==``
    lines and target stdout.  The parser scans for the required long-form header,
    then requires exactly one row for :data:`NCU_L2_HIT_RATE_METRIC`.
    """
    path = Path(path)
    payload = _read_required_bytes(path, "ncu_csv")
    csv_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GoldenProvenanceError(
            f"ncu_csv_encoding_invalid: {path} is not UTF-8"
        ) from exc

    header, rows = _ncu_table(text)
    matches = [
        row for row in rows
        if row.get("Metric Name") == NCU_L2_HIT_RATE_METRIC
    ]
    if len(matches) != 1:
        observed = sorted({
            row.get("Metric Name", "") for row in rows
            if row.get("Metric Name", "")
        })
        raise GoldenProvenanceError(
            "ncu_metric_row_count: expected exactly one "
            f"{NCU_L2_HIT_RATE_METRIC!r} row, found {len(matches)}; "
            f"observed={observed}"
        )

    row = matches[0]
    unit = row.get("Metric Unit", "")
    if unit != "%":
        raise GoldenProvenanceError(
            f"ncu_metric_unit_mismatch: expected '%', got {unit!r}"
        )

    value_text = row.get("Metric Value", "")
    try:
        value_pct = float(value_text)
    except ValueError as exc:
        raise GoldenProvenanceError(
            f"ncu_metric_non_numeric: Metric Value {value_text!r}"
        ) from exc
    if not math.isfinite(value_pct):
        raise GoldenProvenanceError(
            f"ncu_metric_nonfinite: Metric Value {value_text!r}"
        )
    if not 0.0 <= value_pct <= 100.0:
        raise GoldenProvenanceError(
            f"ncu_metric_out_of_range: {value_pct} is outside [0, 100]"
        )

    expected = _expected_dim3(expected_grid)
    observed_grid = _parse_ncu_dim3(row.get("Grid Size", ""))
    if observed_grid != expected:
        raise GoldenProvenanceError(
            f"ncu_grid_mismatch: expected {expected}, got {observed_grid}"
        )

    kernel_name = row.get("Kernel Name")
    if expected_kernel_name is not None:
        if "Kernel Name" not in header:
            raise GoldenProvenanceError(
                "ncu_kernel_column_missing: expected-kernel check requested"
            )
        if kernel_name != expected_kernel_name:
            raise GoldenProvenanceError(
                f"ncu_kernel_mismatch: expected {expected_kernel_name!r}, "
                f"got {kernel_name!r}"
            )

    return {
        "csv_sha256": csv_sha256,
        "metric_name": NCU_L2_HIT_RATE_METRIC,
        "metric_unit": "%",
        "value_pct": value_pct,
        "grid": list(observed_grid),
        "kernel_name": kernel_name,
    }


def _hash_file(path: Path, relative_path: str, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise GoldenProvenanceError(f"{role}_missing: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    if byte_count == 0:
        raise GoldenProvenanceError(f"{role}_empty: {path}")
    return {
        "relative_path": relative_path,
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
    }


def _aggregate_file_records(records: Sequence[dict[str, Any]]) -> str:
    """Hash a sorted file set with unambiguous path/size/content framing."""
    digest = hashlib.sha256(_AGGREGATE_DOMAIN)
    for record in sorted(records, key=lambda item: item["relative_path"]):
        label = record["relative_path"].encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(int(record["bytes"]).to_bytes(8, "big"))
        digest.update(bytes.fromhex(record["sha256"]))
    return digest.hexdigest()


def _explicit_block_labels(
    paths: list[Path],
) -> tuple[list[tuple[str, Path]], Path | None]:
    if not paths:
        return [], None
    absolute = [path.absolute() for path in paths]
    try:
        common = Path(os.path.commonpath([str(path.parent) for path in absolute]))
    except ValueError as exc:
        raise GoldenProvenanceError(
            "first_launch_paths_incompatible: paths have no common root"
        ) from exc
    labelled = [
        (path.relative_to(common).as_posix(), original)
        for path, original in zip(absolute, paths)
    ]
    labels = [label for label, _ in labelled]
    if len(labels) != len(set(labels)):
        raise GoldenProvenanceError(
            f"first_launch_path_duplicate: relative paths are not unique: {labels}"
        )
    return sorted(labelled, key=lambda item: item[0]), common


def _trace_relative_identity(path: Path, trace_dir: Path) -> str:
    try:
        relative = path.absolute().relative_to(trace_dir.absolute())
    except ValueError as exc:
        raise GoldenProvenanceError(
            f"first_launch_outside_trace_dir: {path} is not under {trace_dir}"
        ) from exc
    return relative.as_posix() or "."


def _first_launch_records(
    *,
    trace_dir: Path,
    first_launch_dir: Path | None,
    first_launch_paths: Iterable[Path] | None,
) -> tuple[str, list[dict[str, Any]]]:
    if (first_launch_dir is None) == (first_launch_paths is None):
        raise GoldenProvenanceError(
            "first_launch_source_invalid: provide exactly one of "
            "first_launch_dir or first_launch_paths"
        )

    if first_launch_dir is not None:
        root = Path(first_launch_dir)
        if not root.is_dir():
            raise GoldenProvenanceError(f"first_launch_dir_missing: {root}")
        labelled = sorted(
            (
                (path.relative_to(root).as_posix(), path)
                for path in root.rglob("*.pb")
                if path.is_file()
            ),
            key=lambda item: item[0],
        )
    else:
        paths = [Path(path) for path in first_launch_paths or ()]
        labelled, root = _explicit_block_labels(paths)

    if not labelled:
        raise GoldenProvenanceError(
            "first_launch_payload_missing: no .pb payloads were supplied"
        )
    records = []
    for relative_path, path in labelled:
        if path.suffix != ".pb":
            raise GoldenProvenanceError(
                f"first_launch_payload_not_pb: {path}"
            )
        records.append(
            _hash_file(path, f"first_launch/{relative_path}", "first_launch_payload")
        )
    if root is None:  # Only possible for an empty explicit path set, rejected above.
        raise GoldenProvenanceError("first_launch_payload_missing: no payload root")
    return _trace_relative_identity(root, trace_dir), records


def trace_provenance(
    trace_dir: Path,
    *,
    first_launch_dir: Path | None = None,
    first_launch_paths: Iterable[Path] | None = None,
    include_files: bool = False,
) -> dict[str, Any]:
    """Build deterministic provenance for one trace and its first launch.

    ``first_launch_dir`` is convenient for the shared launch locator; callers
    that already selected payloads may pass ``first_launch_paths`` instead.
    Explicit path order is immaterial.
    """
    trace_dir = Path(trace_dir)
    if not trace_dir.is_dir():
        raise GoldenProvenanceError(f"trace_dir_missing: {trace_dir}")

    dynamic = _hash_file(
        trace_dir / "dynamic_trace.pb", "dynamic_trace.pb", "dynamic_trace_pb"
    )
    stats = _hash_file(trace_dir / "stats.csv", "stats.csv", "stats_csv")
    allocations_path = trace_dir / "extra_info" / "allocations.json"
    allocations = (
        _hash_file(
            allocations_path,
            "extra_info/allocations.json",
            "allocations_json",
        )
        if allocations_path.exists()
        else None
    )
    launch_relative_path, blocks = _first_launch_records(
        trace_dir=trace_dir,
        first_launch_dir=first_launch_dir,
        first_launch_paths=first_launch_paths,
    )

    metadata_records = [dynamic, stats]
    if allocations is not None:
        metadata_records.append(allocations)
    all_records = metadata_records + blocks
    first_launch: dict[str, Any] = {
        "relative_path": launch_relative_path,
        "content_sha256": _aggregate_file_records(blocks),
        "block_count": len(blocks),
        "byte_count": sum(int(record["bytes"]) for record in blocks),
    }
    if include_files:
        first_launch["files"] = blocks
    return {
        "dynamic_trace_pb": dynamic,
        "stats_csv": stats,
        "allocations_json": allocations,
        "first_launch": first_launch,
        "content_sha256": _aggregate_file_records(all_records),
    }


def _version_string(value: str, role: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GoldenProvenanceError(
            f"{role}_version_invalid: expected a non-empty trimmed string"
        )
    return value


def build_golden_provenance(
    *,
    ncu_csv_path: Path,
    trace_dir: Path,
    expected_grid: Sequence[int],
    tracer_version: str,
    stats_version: str,
    estimator_version: str,
    expected_kernel_name: str | None = None,
    first_launch_dir: Path | None = None,
    first_launch_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Build the JSON-ready NCU + trace provenance envelope.

    Producer versions are required keyword arguments so a caller cannot emit an
    apparently complete record with implicit or environment-derived versions.
    """
    versions = {
        "tracer": _version_string(tracer_version, "tracer"),
        "stats": _version_string(stats_version, "stats"),
        "estimator": _version_string(estimator_version, "estimator"),
    }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "versions": versions,
        "ncu": parse_ncu_l2_csv(
            ncu_csv_path,
            expected_grid=expected_grid,
            expected_kernel_name=expected_kernel_name,
        ),
        "trace": trace_provenance(
            trace_dir,
            first_launch_dir=first_launch_dir,
            first_launch_paths=first_launch_paths,
        ),
    }
