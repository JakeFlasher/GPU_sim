#!/usr/bin/env python3
"""Reduce one retained LDGSTS burst-release campaign.

The frozen reducer in :mod:`analyze` deliberately accepts only clean paired
rows.  This driver is the small file-system adapter around it: it streams one
raw process at a time, retains any raw gap outliers as a diagnostic, reads the
comparison-only NCU records, and writes one compact terminal report.  NCU
values never become simulator inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - the repository venv has orjson
    orjson = None


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze as frozen  # noqa: E402


REPORT_SCHEMA = 1
NCU_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_executed_op_ldgsts.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__sectors_read.sum",
)
INSTANCE_BASES = (
    "smsp__inst_executed_op_ldgsts",
    "smsp__sass_inst_executed_op_global_ld",
)
BUNDLES = ("route_zero_gap", "spacing_g16", "spacing_g64", "placement", "clock_assay")


class LiveAnalysisError(RuntimeError):
    """The retained campaign cannot be reduced safely."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LiveAnalysisError(f"cannot read JSON: {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_line(line: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line.decode("utf-8"))


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveAnalysisError(f"{label}: expected a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise LiveAnalysisError(f"{label}: expected a finite number")
    return result


def number_from_csv(row: Mapping[str, str], name: str) -> float:
    value = row.get(name)
    if value is None or value.strip() == "":
        raise LiveAnalysisError(f"NCU CSV is missing {name}")
    try:
        return float(value.replace(",", ""))
    except ValueError as error:
        raise LiveAnalysisError(f"NCU value {name} is not numeric: {value!r}") from error


def expected_bundle_keys() -> dict[str, tuple[str, ...]]:
    by_bundle: dict[str, list[str]] = {bundle: [] for bundle in BUNDLES}
    for spec in frozen.EXPECTED_CELLS:
        if spec.role in {"identification", "holdout"}:
            bundle = "route_zero_gap"
        elif spec.role == "spacing":
            bundle = f"spacing_g{spec.gap}"
        elif spec.role == "placement":
            bundle = "placement"
        elif spec.role == "clock_assay":
            bundle = "clock_assay"
        else:  # pragma: no cover - frozen matrix validation catches this
            raise LiveAnalysisError(f"unknown role for {spec.key}: {spec.role}")
        by_bundle[bundle].append(spec.key)
    # The placement launch deliberately repeats nine W1 references before its
    # nine W4 diagnostic cells.  They are validated as part of that launch but
    # the route-zero-gap copy remains the canonical W1 reduction source.
    by_bundle["placement"] = list(by_bundle["placement"])
    by_bundle["placement"][:0] = [
        spec.key
        for spec in frozen.EXPECTED_CELLS
        if spec.warps == 1
        and spec.assay == "ldgsts"
        and spec.role in {"identification", "holdout"}
        and spec.depth in frozen.KNEE_DEPTHS
    ]
    return {key: tuple(value) for key, value in by_bundle.items()}


def _gap_normalized_rows(
    rows: Sequence[Mapping[str, Any]],
    spec: frozen.CellSpec,
    outliers: list[dict[str, Any]],
) -> list[Mapping[str, Any]]:
    """Preserve a diagnostic reduction when a scheduler pause hit one edge.

    The original value is retained in ``outliers``.  Only the diagnostic copy
    passed to the strict reducer is bounded, so a pause can never silently
    enter the scientific result or be mistaken for a valid achieved delay.
    """

    if spec.gap == 0:
        return list(rows)
    low, high = frozen._gap_bounds(spec.gap)
    normalized: list[Mapping[str, Any]] = []
    for row_index, row in enumerate(rows):
        copy = dict(row)
        for field in ("gaps_on", "gaps_off"):
            values: list[float] = []
            raw_values = row.get(field)
            if not isinstance(raw_values, list):
                normalized.append(copy)
                break
            for edge, raw in enumerate(raw_values):
                value = as_float(raw, f"{spec.key}:{field}[{edge}]")
                if not low <= value <= high:
                    outliers.append(
                        {
                            "cell": spec.key,
                            "row": row_index,
                            "field": field,
                            "edge": edge,
                            "value": value,
                            "expected_range": [low, high],
                        }
                    )
                    value = min(max(value, low), high)
                values.append(value)
            copy[field] = values
        normalized.append(copy)
    return normalized


def parse_direct_process(
    path: Path,
    repetition: int,
    expected_keys: Sequence[str],
    outliers: list[dict[str, Any]],
) -> tuple[dict[str, frozen.ProcessReduction], dict[str, Any]]:
    """Stream one raw JSONL process, reducing contiguous cells one at a time."""

    expected = set(expected_keys)
    reductions: dict[str, frozen.ProcessReduction] = {}
    seen: set[str] = set()
    current: str | None = None
    rows: list[Mapping[str, Any]] = []
    summary: Mapping[str, Any] | None = None
    parse_errors: list[str] = []

    def add_error(message: str) -> None:
        # A malformed bundle can otherwise produce hundreds of thousands of
        # repeated per-line messages and drown the compact report.  Keep a
        # bounded sample and add one explicit truncation marker.
        if len(parse_errors) < 32:
            parse_errors.append(message)
        elif len(parse_errors) == 32:
            parse_errors.append(f"{path}: additional parse errors omitted")

    def finish() -> None:
        nonlocal rows, current
        if current is None:
            return
        if current in reductions:
            add_error(f"{path}: cell repeated non-contiguously: {current}")
            rows = []
            return
        spec = frozen.EXPECTED_BY_KEY.get(current)
        if spec is None:
            add_error(f"{path}: unknown cell {current}")
            rows = []
            return
        try:
            clean_rows = _gap_normalized_rows(rows, spec, outliers)
            reductions[current] = frozen.reduce_process_rows(clean_rows, spec, repetition)
        except (frozen.AnalysisError, LiveAnalysisError) as error:
            add_error(f"{path}:{current}: {error}")
        rows = []

    try:
        handle = path.open("rb")
    except OSError as error:
        raise LiveAnalysisError(f"cannot open direct raw stream: {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = decode_line(line)
            except (ValueError, UnicodeDecodeError) as error:
                add_error(f"{path}:{line_number}: malformed JSON: {error}")
                continue
            if not isinstance(value, Mapping):
                add_error(f"{path}:{line_number}: JSON value is not an object")
                continue
            row_type = value.get("type")
            if row_type == "paired":
                key = value.get("cell")
                if not isinstance(key, str) or key not in expected:
                    add_error(f"{path}:{line_number}: unexpected cell {key!r}")
                    continue
                if current is None:
                    current = key
                elif key != current:
                    finish()
                    seen.add(current)
                    current = key
                rows.append(value)
            elif row_type == "summary":
                summary = value
            else:
                add_error(f"{path}:{line_number}: unexpected record type {row_type!r}")
    finish()
    if current is not None:
        seen.add(current)
    missing = sorted(expected - set(reductions))
    if missing:
        add_error(f"{path}: missing reduced cells: {', '.join(missing)}")
    if summary is None:
        add_error(f"{path}: no summary record")
    else:
        observed = summary.get("cell_keys")
        if not isinstance(observed, list) or set(observed) != expected:
            add_error(f"{path}: summary cell set differs from frozen bundle")
        if summary.get("correct") is not True:
            add_error(f"{path}: summary correctness is not true")
        expected_records = frozen.BURSTS * sum(
            frozen.EXPECTED_BY_KEY[key].warps for key in expected_keys
        )
        if summary.get("record_count") != expected_records:
            add_error(
                f"{path}: record_count {summary.get('record_count')} != {expected_records}"
            )
    return reductions, {
        "path": str(path),
        "repetition": repetition,
        "summary": dict(summary) if isinstance(summary, Mapping) else None,
        "errors": parse_errors,
        "outlier_count": sum(1 for item in outliers if item.get("cell") in expected),
    }


def reduce_direct(root: Path) -> tuple[dict[str, list[frozen.ProcessReduction]], dict[str, Any]]:
    bundle_keys = expected_bundle_keys()
    reductions: dict[str, list[frozen.ProcessReduction]] = defaultdict(list)
    process_records: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    errors: list[str] = []
    for bundle, keys in bundle_keys.items():
        for repetition in range(1, frozen.DIRECT_REPETITIONS + 1):
            path = root / "direct" / bundle / f"repeat-{repetition}" / "raw.jsonl"
            if not path.is_file():
                errors.append(f"missing direct stream: {path}")
                continue
            before = len(outliers)
            reduced, record = parse_direct_process(path, repetition, keys, outliers)
            record["raw_outlier_count"] = len(outliers) - before
            process_records.append(record)
            errors.extend(record["errors"])
            for key, value in reduced.items():
                # Placement replays the W1 references in the same CTA as W4;
                # keep that copy as a cross-check, not as an eighth-through-
                # fourteenth pseudo-repetition of the canonical W1 stream.
                if bundle == "placement" and frozen.EXPECTED_BY_KEY[key].warps == 1:
                    continue
                reductions[key].append(value)
    for key in list(reductions):
        reductions[key].sort(key=lambda value: value.repetition)
    missing = sorted(set(frozen.EXPECTED_BY_KEY) - set(reductions))
    if missing:
        errors.append("direct reduction missing cells: " + ", ".join(missing))
    incomplete = sorted(
        key
        for key, values in reductions.items()
        if len(values) != frozen.DIRECT_REPETITIONS
    )
    if incomplete:
        errors.append("direct reduction missing repetitions: " + ", ".join(incomplete))
    gap_report: dict[str, Any] = {"admitted": False, "failures": ["spacing reductions unavailable"], "differences": {}}
    spacing = {
        key: values
        for key, values in reductions.items()
        if frozen.EXPECTED_BY_KEY[key].role == "spacing"
    }
    if not errors or spacing:
        try:
            gap_admission = frozen.admit_gap_matching(spacing)
            gap_report = {
                "admitted": gap_admission.admitted,
                "failures": list(gap_admission.failures),
                "differences": {
                    key: {
                        "values": list(value.values),
                        "median": value.median,
                        "mad": value.mad,
                    }
                    for key, value in gap_admission.differences.items()
                },
            }
            if not gap_admission.admitted:
                errors.extend(gap_admission.failures)
        except frozen.AnalysisError as error:
            gap_report = {"admitted": False, "failures": [str(error)], "differences": {}}
            errors.append(str(error))
    return reductions, {
        "bundles": bundle_keys,
        "processes": process_records,
        "raw_gap_outliers": outliers,
        "raw_gap_outlier_count": len(outliers),
        "errors": sorted(set(errors)),
        "strict_raw_gap_admission": not outliers and not errors,
        "gap_admission": gap_report,
    }


def marker_complete(path: Path, kind: str, key: str) -> tuple[bool, list[str], Mapping[str, Any] | None]:
    marker_path = path / "complete.json"
    if not marker_path.is_file():
        return False, [f"missing complete marker: {path}"], None
    try:
        marker = load_json(marker_path)
    except LiveAnalysisError as error:
        return False, [str(error)], None
    failures: list[str] = []
    if not isinstance(marker, Mapping):
        return False, [f"malformed complete marker: {path}"], None
    if marker.get("complete") is not True:
        failures.append(f"{path}: complete=false")
    if marker.get("kind") != kind or marker.get("key") != key:
        failures.append(f"{path}: marker identity mismatch")
    required = marker.get("required_files")
    if not isinstance(required, list):
        failures.append(f"{path}: required_files is not a list")
    else:
        for relative in required:
            candidate = path / str(relative)
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                failures.append(f"{path}: missing/empty required file {relative}")
    return not failures, failures, marker


def read_ncu_cell(root: Path, spec: frozen.CellSpec, repetition: int) -> dict[str, Any]:
    path = root / "ncu" / spec.key / f"repeat-{repetition}"
    ok, failures, marker = marker_complete(path, "ncu", f"{spec.key}:repeat-{repetition}")
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        failures.append(f"missing NCU metadata: {metadata_path}")
        return {"cell": spec.key, "repetition": repetition, "valid": False, "failures": failures}
    try:
        metadata = load_json(metadata_path)
    except LiveAnalysisError as error:
        failures.append(str(error))
        return {"cell": spec.key, "repetition": repetition, "valid": False, "failures": failures}
    observed = metadata.get("observed") if isinstance(metadata, Mapping) else None
    if not isinstance(observed, Mapping):
        failures.append("NCU metadata has no observed metrics")
        observed = {}
    values: dict[str, float] = {}
    for metric in NCU_METRICS:
        try:
            values[metric] = as_float(observed[metric], f"{spec.key}:{metric}")
        except (KeyError, LiveAnalysisError) as error:
            failures.append(str(error))
    csv_path = path / "ncu.csv"
    csv_row: Mapping[str, str] | None = None
    if csv_path.is_file():
        try:
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row.get("Kernel Name")]
            if len(rows) != 1:
                failures.append(f"NCU CSV kernel-row count is {len(rows)}, expected 1")
            else:
                csv_row = rows[0]
        except (OSError, csv.Error) as error:
            failures.append(f"cannot parse NCU CSV: {error}")
    else:
        failures.append(f"missing NCU CSV: {csv_path}")
    instance: dict[str, Any] = {}
    if csv_row is not None:
        for base in INSTANCE_BASES:
            stats: dict[str, float] = {}
            for suffix in ("avg", "max", "min", "sum"):
                name = f"{base}.{suffix}"
                try:
                    stats[suffix] = number_from_csv(csv_row, name)
                except LiveAnalysisError as error:
                    failures.append(str(error))
            instance[base] = stats
    identity = metadata.get("identity", {}) if isinstance(metadata, Mapping) else {}
    if isinstance(metadata, Mapping) and metadata.get("return_code") != 0:
        failures.append(f"NCU return_code={metadata.get('return_code')}")
    if isinstance(metadata, Mapping) and metadata.get("timed_out") is True:
        failures.append("NCU profile timed out")
    if isinstance(identity, Mapping):
        if identity.get("CC") not in {"8.6", 8.6}:
            failures.append(f"unexpected compute capability {identity.get('CC')!r}")
        if identity.get("Device") not in {"0", 0}:
            failures.append(f"unexpected profiler device {identity.get('Device')!r}")
    # Q is the source-sector denominator for route counters.  The LDGSTS
    # destination-sector and shared-wavefront counters are zero for the
    # independent LDG clock control, even though its global-load route work
    # uses the same 16-sector-per-position denominator.
    expected_q = 16 * spec.depth * spec.warps * frozen.BURSTS
    expected_ldgsts_q = expected_q if spec.assay == "ldgsts" else 0
    expected_wf = 4 * spec.depth * spec.warps * frozen.BURSTS if spec.assay == "ldgsts" else 0
    expected_ldgsts = (
        (8 if spec.gap == 0 else spec.depth) * spec.warps * frozen.BURSTS
        if spec.assay == "ldgsts"
        else 0
    )
    expected_ld = spec.depth * spec.warps * frozen.BURSTS if spec.assay == "ldg" else 0
    qobs = values.get(NCU_METRICS[3])
    wf = values.get(NCU_METRICS[4])
    ldgsts_count = values.get(NCU_METRICS[1])
    ld_count = values.get(NCU_METRICS[2])
    h1 = values.get(NCU_METRICS[5])
    m1 = values.get(NCU_METRICS[6])
    h2 = values.get(NCU_METRICS[7])
    m2 = values.get(NCU_METRICS[8])
    checks: dict[str, Any] = {
        "expected_q": expected_q,
        "expected_ldgsts_q": expected_ldgsts_q,
        "expected_wavefronts": expected_wf,
    }
    if qobs is not None:
        checks["qobs_exact"] = qobs == expected_ldgsts_q
    if wf is not None:
        checks["wavefronts_exact"] = wf == expected_wf
    if ldgsts_count is not None:
        checks["ldgsts_count_exact"] = ldgsts_count == expected_ldgsts
    if ld_count is not None:
        checks["global_ld_count_exact"] = ld_count == expected_ld
    if h1 is not None and m1 is not None:
        checks["l1_conserved"] = h1 + m1 == expected_q
    if h2 is not None and m2 is not None and m1 is not None:
        checks["lts_conserved"] = abs((h2 + m2) - m1) <= max(16.0, expected_q / 1000.0)
    route_label: str | None = None
    if h1 is not None and m1 is not None and expected_q > 0:
        if h1 / (h1 + m1) >= 0.95:
            route_label = "l1"
        elif m1 / expected_q >= 0.95 and h2 is not None and m2 is not None and h2 + m2 > 0:
            if h2 / (h2 + m2) >= 0.95:
                route_label = "l2"
            elif m2 / (h2 + m2) >= 0.95:
                route_label = "dram"
    checks["route_label"] = route_label
    if route_label != spec.route:
        failures.append(f"route label {route_label!r} != expected {spec.route!r}")
    active_cycles = values.get(NCU_METRICS[0])
    return {
        "cell": spec.key,
        "repetition": repetition,
        # Per-profile counter conservation can lose a small sector bucket on
        # one replay.  Keep that observation below and decide admission from
        # matched-repeat medians in ``reduce_ncu``; metadata/identity/route
        # failures remain hard per-profile failures.
        "valid": ok and not failures,
        "failures": failures,
        "observed": values,
        "checks": checks,
        "identity": dict(identity) if isinstance(identity, Mapping) else {},
        "instance": instance,
        "active_cycles": active_cycles,
        "marker": dict(marker) if isinstance(marker, Mapping) else None,
    }


def reduce_ncu(root: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    errors: list[str] = []
    for spec in frozen.EXPECTED_CELLS:
        repeats = [read_ncu_cell(root, spec, repetition) for repetition in range(1, frozen.NCU_REPEATS_MIN + 1)]
        cycle_values = [item["active_cycles"] for item in repeats if isinstance(item.get("active_cycles"), (int, float))]
        cycle_mean = statistics.mean(cycle_values) if cycle_values else None
        cycle_cv = (
            (max(cycle_values) - min(cycle_values)) / cycle_mean
            if cycle_mean and len(cycle_values) >= 2
            else None
        )
        for item in repeats:
            if not item["valid"]:
                errors.extend(f"{spec.key} r{item['repetition']}: {failure}" for failure in item["failures"])
        aggregate: dict[str, Any] = {}
        aggregate_failures: list[str] = []
        for metric in NCU_METRICS:
            values = [item.get("observed", {}).get(metric) for item in repeats]
            if not all(isinstance(value, (int, float)) for value in values):
                aggregate_failures.append(f"missing metric median: {metric}")
            else:
                aggregate[f"median:{metric}"] = statistics.median(values)
        q = 16 * spec.depth * spec.warps * frozen.BURSTS
        ldgsts_q = q if spec.assay == "ldgsts" else 0
        expected_wf = 4 * spec.depth * spec.warps * frozen.BURSTS if spec.assay == "ldgsts" else 0
        expected_ldgsts = ((8 if spec.gap == 0 else spec.depth) * spec.warps * frozen.BURSTS
                           if spec.assay == "ldgsts" else 0)
        expected_ld = spec.depth * spec.warps * frozen.BURSTS if spec.assay == "ldg" else 0
        qobs = aggregate.get(f"median:{NCU_METRICS[3]}")
        wf = aggregate.get(f"median:{NCU_METRICS[4]}")
        ldgsts_count = aggregate.get(f"median:{NCU_METRICS[1]}")
        ld_count = aggregate.get(f"median:{NCU_METRICS[2]}")
        h1 = aggregate.get(f"median:{NCU_METRICS[5]}")
        m1 = aggregate.get(f"median:{NCU_METRICS[6]}")
        h2 = aggregate.get(f"median:{NCU_METRICS[7]}")
        m2 = aggregate.get(f"median:{NCU_METRICS[8]}")
        aggregate_checks: dict[str, Any] = {}
        if qobs is not None:
            aggregate_checks["qobs_exact"] = qobs == ldgsts_q
        if wf is not None:
            aggregate_checks["wavefronts_exact"] = wf == expected_wf
        if ldgsts_count is not None:
            aggregate_checks["ldgsts_count_exact"] = ldgsts_count == expected_ldgsts
        if ld_count is not None:
            aggregate_checks["global_ld_count_exact"] = ld_count == expected_ld
        if h1 is not None and m1 is not None:
            aggregate_checks["l1_conserved"] = h1 + m1 == q
        if h2 is not None and m2 is not None and m1 is not None:
            aggregate_checks["lts_conserved"] = abs((h2 + m2) - m1) <= max(16.0, q / 1000.0)
        aggregate_route: str | None = None
        if h1 is not None and m1 is not None and h1 + m1 > 0:
            if h1 / (h1 + m1) >= 0.95:
                aggregate_route = "l1"
            elif m1 / q >= 0.95 and h2 is not None and m2 is not None and h2 + m2 > 0:
                aggregate_route = "l2" if h2 / (h2 + m2) >= 0.95 else "dram" if m2 / (h2 + m2) >= 0.95 else None
        aggregate_checks["route_label"] = aggregate_route
        if aggregate_route != spec.route:
            aggregate_failures.append(f"aggregate route label {aggregate_route!r} != {spec.route!r}")
        for name, value in aggregate_checks.items():
            if name != "route_label" and value is not True:
                aggregate_failures.append(f"aggregate {name} failed")
        expected_instance_count = spec.warps if spec.assay == "ldgsts" else 1
        instance_counts: list[int | None] = []
        for item in repeats:
            stats = item.get("instance", {}).get("smsp__inst_executed_op_ldgsts", {})
            maximum = stats.get("max")
            total = stats.get("sum")
            if maximum and maximum > 0 and total is not None:
                rounded = int(round(total / maximum))
                instance_counts.append(rounded if abs(total - rounded * maximum) <= 1.0 else None)
            elif spec.assay == "ldg":
                instance_counts.append(1)
            else:
                instance_counts.append(None)
        instance_ok = all(value == expected_instance_count for value in instance_counts)
        if not instance_ok:
            errors.append(f"{spec.key}: SMSP instance count {instance_counts} != {expected_instance_count}")
        if aggregate_failures:
            errors.extend(f"{spec.key}: {failure}" for failure in aggregate_failures)
        cells[spec.key] = {
            "expected_route": spec.route,
            "expected_instance_count": expected_instance_count,
            "repeats": repeats,
            "aggregate": aggregate,
            "aggregate_checks": aggregate_checks,
            "aggregate_failures": aggregate_failures,
            "active_cycle_cv": cycle_cv,
            "high_cv_requires_extension": cycle_cv is not None and cycle_cv > 0.02,
            "instance_counts": instance_counts,
            "instance_admission": instance_ok,
            "admitted": all(item["valid"] for item in repeats) and instance_ok and not aggregate_failures,
        }
    # DRAM-vs-L2 is a matched route corroboration, never a timing scalar.
    route_deltas: list[dict[str, Any]] = []
    for warps in (1, 4):
        for depth in frozen.KNEE_DEPTHS:
            dram_key = frozen._cell_key("ldgsts", warps, "dram", depth, 0)
            l2_key = frozen._cell_key("ldgsts", warps, "l2", depth, 0)
            if dram_key not in cells or l2_key not in cells:
                continue
            dram_values = [item["observed"].get(NCU_METRICS[9]) for item in cells[dram_key]["repeats"]]
            l2_values = [item["observed"].get(NCU_METRICS[9]) for item in cells[l2_key]["repeats"]]
            dram_median = statistics.median(dram_values)
            l2_median = statistics.median(l2_values)
            q = 16 * depth * warps * frozen.BURSTS
            delta = dram_median - l2_median
            route_deltas.append(
                {
                    "warps": warps,
                    "depth": depth,
                    "dram_minus_l2_median": dram_median - l2_median,
                    "threshold": 0.5 * q,
                    "admitted": delta >= 0.5 * q,
                }
            )
            if delta < 0.5 * q:
                errors.append(f"{dram_key} vs {l2_key}: DRAM excess {delta} below {0.5 * q}")
    high_cv = sorted(key for key, value in cells.items() if value["high_cv_requires_extension"])
    return {
        "cells": cells,
        "route_deltas": route_deltas,
        "high_cv_cells": high_cv,
        "errors": sorted(set(errors)),
        "all_cells_admitted": all(value["admitted"] for value in cells.values()) and not errors,
    }


def series_json(series: object) -> dict[str, Any]:
    if not isinstance(series, (frozen.EffectSeries, frozen.ContrastSeries)):
        raise LiveAnalysisError(f"cannot serialize non-series value: {series!r}")
    return {
        "name": series.name if isinstance(series, frozen.ContrastSeries) else None,
        "values": [float(value) for value in series.values],
        "median": float(series.median),
        "mad": float(series.mad),
    }


def effects_json(effects: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    return {
        family: {key: series_json(value) for key, value in values.items()}
        for family, values in effects.items()
    }


def reduction_json(value: frozen.ProcessReduction) -> dict[str, Any]:
    return {
        "cell": value.cell,
        "repetition": value.repetition,
        "smid": value.smid,
        "issue_by_slot": list(value.issue_by_slot),
        "completion_by_slot": list(value.completion_by_slot),
        "gap_on_by_slot": [list(edges) for edges in value.gap_on_by_slot],
        "gap_off_by_slot": [list(edges) for edges in value.gap_off_by_slot],
        "arm_order_pattern": list(value.arm_order_pattern),
    }


def static_admission(root: Path) -> tuple[bool, dict[str, Any]]:
    path = root / "static" / "static-report.json"
    if not path.is_file():
        return False, {"errors": [f"missing static report: {path}"]}
    report = load_json(path)
    errors: list[str] = []
    findings = report.get("findings", {}) if isinstance(report, Mapping) else {}
    if findings.get("unknown_control_encodings") != 0:
        errors.append("unknown control encodings are present")
    if findings.get("zero_gap_target_endpoint_clean_observed_count") != 5:
        errors.append("not all zero-gap variants have clean endpoints")
    if findings.get("gapped_clock_wait_masks_observed") != []:
        errors.append(f"gapped wait masks are nonzero: {findings.get('gapped_clock_wait_masks_observed')}")
    variants = report.get("target_variants", []) if isinstance(report, Mapping) else []
    for item in variants:
        if not isinstance(item, Mapping) or item.get("assay") != "ldgsts":
            continue
        depth = item.get("depth")
        gap = item.get("gap")
        expected_targets = 8 if gap == 0 else depth
        if item.get("target_in_function") != expected_targets:
            errors.append(f"{item.get('function')}: target count mismatch")
        if item.get("issue_clock_wait_masks") != [0] or item.get("drain_clock_wait_masks") != [0]:
            errors.append(f"{item.get('function')}: endpoint wait mask mismatch")
        if item.get("cta_barriers_in_timed_window"):
            errors.append(f"{item.get('function')}: CTA barrier in timed window")
    return not errors, {
        "errors": sorted(set(errors)),
        "findings": findings,
        "target_variant_count": len(variants) if isinstance(variants, list) else None,
    }


def pipeline_status(root: Path) -> dict[str, Any]:
    """Check the bounded marker-scoped NVBit -> simulator smoke, if present."""

    full_capture = root / "nvbit" / "ldgsts_w1_l1_d4_g0"
    trace = root / "nvbit" / "smoke5_ldgsts_w1_l1_d4_b64"
    simulator = root / "simulator" / "smoke5_ldgsts_w1_l1_d4_b64"
    errors: list[str] = []
    trace_complete, trace_marker_errors, _ = marker_complete(
        trace, "ldgsts_nvbit_capture", "ldgsts_w1_l1_d4_g0"
    )
    if not trace_complete:
        errors.extend(trace_marker_errors)
    trace_files = [trace / name for name in ("dynamic_trace.pb", "stats.csv")]
    payloads = list((trace / "threadblocks").rglob("*.pb")) if (trace / "threadblocks").is_dir() else []
    if not all(path.is_file() and path.stat().st_size > 0 for path in trace_files):
        errors.append("marker-scoped NVBit smoke has no complete primary trace")
    if not payloads:
        errors.append("marker-scoped NVBit smoke has no threadblock payload")
    replays = [simulator / "replay-1.json", simulator / "replay-2.json"]
    replay_hashes: list[str] = []
    replay_docs: list[Mapping[str, Any]] = []
    for path in replays:
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing simulator replay: {path}")
            continue
        replay_hashes.append(sha256_file(path))
        try:
            value = load_json(path)
            if not isinstance(value, Mapping):
                errors.append(f"simulator replay is not an object: {path}")
            else:
                replay_docs.append(value)
        except LiveAnalysisError as error:
            errors.append(str(error))
    if len(replay_hashes) == 2 and replay_hashes[0] != replay_hashes[1]:
        errors.append("simulator replay twins differ")
    for document in replay_docs:
        launches = document.get("launches")
        if document.get("schema_version") != 1 or not isinstance(launches, list) or len(launches) != 1:
            errors.append("simulator replay launch/schema contract failed")
        elif launches[0].get("coverage", {}).get("kind") != "full":
            errors.append("simulator replay is not full coverage")
    return {
        "full_capture_root": str(full_capture),
        "full_capture_failure": load_json(full_capture / "failure.json") if (full_capture / "failure.json").is_file() else None,
        "trace_root": str(trace),
        "simulator_root": str(simulator),
        "simulator_command": [
            "sim/target/release/sim-run",
            "--trace", str(trace),
            "--control", str(root / "static" / "control_bits.json"),
            "--arch-card", "arch_profiles/cards/sm_86.json",
            "--pchase-card", "data/pchase/sm_86_static_card.json",
            "--mode", "l1-l2",
        ],
        "trace_sha256": sha256_file(trace / "dynamic_trace.pb") if (trace / "dynamic_trace.pb").is_file() else None,
        "trace_payload_count": len(payloads),
        "replay_sha256": replay_hashes,
        "passed": not errors,
        "errors": errors,
        "comparison_only": True,
    }


def clock_assay(
    reductions: Mapping[str, Sequence[frozen.ProcessReduction]],
) -> tuple[object | None, dict[str, Any]]:
    keys = {key: value for key, value in reductions.items() if frozen.EXPECTED_BY_KEY[key].assay == "ldg"}
    if set(keys) != {frozen._cell_key("ldg", 1, "l1", depth, 0) for depth in (4, 5, 6, 8)}:
        return None, {"errors": ["clock assay reductions are incomplete"]}
    try:
        p = frozen.contrast_series(keys, "ldg_w1_l1_d5_g0", "ldg_w1_l1_d4_g0", name="LDG:P:l1:g0")
        s = frozen.contrast_series(keys, "ldg_w1_l1_d6_g0", "ldg_w1_l1_d5_g0", name="LDG:S:l1:g0")
        knee = frozen.difference_series(s, p, name="LDG:K:l1:g0")
    except frozen.AnalysisError as error:
        return None, {"errors": [str(error)]}
    return knee, {
        "status": "valid" if frozen.pos(knee) else "invalid_clock_assay",
        "P": series_json(p),
        "S": series_json(s),
        "K": series_json(knee),
        "positive_material": frozen.pos(knee),
        "errors": [],
    }


def build_stages(
    root: Path,
    reductions: Mapping[str, Sequence[frozen.ProcessReduction]],
    direct: Mapping[str, Any],
    ncu: Mapping[str, Any],
    static_ok: bool,
    static: Mapping[str, Any],
) -> dict[str, Any]:
    w1 = {key: value for key, value in reductions.items() if frozen.EXPECTED_BY_KEY[key].warps == 1}
    w4 = {key: value for key, value in reductions.items() if frozen.EXPECTED_BY_KEY[key].warps == 4}
    clock, clock_report = clock_assay(reductions)
    try:
        pskr = frozen.reduce_pskr(w1, active_warps=1)
    except frozen.AnalysisError as error:
        return {"clock_assay": clock_report, "errors": [f"W1 PSKR reduction failed: {error}"]}
    effects: dict[str, Mapping[str, object]] = {
        "K": pskr.k,
        "R": pskr.r,
        "E": pskr.e,
        "C": pskr.c,
    }
    if clock is not None:
        effects["LDG_K"] = {"l1:g0": clock}
    # Keep clock-lock availability separate from the mathematical reducer.
    lock_record = load_json(root / "metadata" / "clock-session-direct.json") if (root / "metadata" / "clock-session-direct.json").is_file() else {}
    lock_ok = lock_record.get("lock_applied") is True if isinstance(lock_record, Mapping) else False
    w1_keys = [key for key in w1 if frozen.EXPECTED_BY_KEY[key].role in {"identification", "spacing"}]
    ncu_w1_ok = all(ncu.get("cells", {}).get(key, {}).get("admitted") for key in w1_keys)
    identification_admission = (
        static_ok
        and not direct.get("errors")
        and not direct.get("raw_gap_outliers")
        and ncu_w1_ok
        and clock is not None
    )
    identification_failures = (
        tuple(static.get("errors", []))
        + tuple(direct.get("errors", []))
        + (("raw gap outliers retained",) if direct.get("raw_gap_outliers") else ())
        + (("NCU W1 route/work admission failed",) if not ncu_w1_ok else ())
    )
    try:
        identification = frozen.classify_identification(
            admission_ok=identification_admission,
            clock_knee=clock,
            effects=effects,
            admission_failures=identification_failures,
        )
    except frozen.AnalysisError as error:
        identification = frozen.StageResult("invalid_experiment", (str(error),))

    placement_effects: dict[int, frozen.PSKR] = {}
    placement_errors: list[str] = []
    try:
        for slot in range(4):
            placement_effects[slot] = frozen.reduce_slot_pskr(w4, slot)
    except frozen.AnalysisError as error:
        placement_errors.append(str(error))
    placement_ncu_ok = all(
        ncu.get("cells", {}).get(key, {}).get("admitted")
        for key in w4
    )
    if len(placement_effects) == 4:
        placement_failures = tuple(placement_errors) + (
            ("NCU W4 admission failed",) if not placement_ncu_ok else ()
        )
        placement = frozen.classify_placement(
            identification=identification,
            admission_ok=not placement_errors and placement_ncu_ok and static_ok,
            effects_by_slot=placement_effects,
            admission_failures=placement_failures,
            w1_effects=pskr,
        )
    else:
        placement = frozen.StageResult("invalid", tuple(placement_errors or ("W4 reductions incomplete",)))
    holdout_keys = {key: value for key, value in w1.items() if frozen.EXPECTED_BY_KEY[key].role in {"identification", "spacing", "holdout"}}
    holdout_ncu_ok = all(
        ncu.get("cells", {}).get(key, {}).get("admitted")
        for key in holdout_keys
    )
    holdout = frozen.classify_l2(
        identification=identification,
        admission_ok=holdout_ncu_ok and static_ok and not placement_errors,
        effects=effects,
        w1_total_route=pskr.r.get(frozen.route_key(6, 0)),
        admission_failures=("NCU holdout admission failed",) if not holdout_ncu_ok else (),
    )
    staged = frozen.classify_staged(
        identification=identification,
        placement=placement,
        holdout=holdout,
    )
    # An unavailable external clock lock is an explicit unknown gate; it does
    # not relabel the measured W1 result, but it prevents successor promotion.
    successor_eligible = staged.successor_eligible and lock_ok and not direct.get("raw_gap_outliers")
    completion_route: dict[str, Any] = {}
    for depth in frozen.IDENTIFICATION_DEPTHS:
        l1_key = frozen._cell_key("ldgsts", 1, "l1", depth, 0)
        l2_key = frozen._cell_key("ldgsts", 1, "l2", depth, 0)
        dram_key = frozen._cell_key("ldgsts", 1, "dram", depth, 0)
        if all(key in pskr.c for key in (l1_key, l2_key, dram_key)):
            a = frozen.difference_series(pskr.c[l2_key], pskr.c[l1_key], name=f"C:A:n{depth}")
            b = frozen.difference_series(pskr.c[dram_key], pskr.c[l2_key], name=f"C:B:n{depth}")
            completion_route[f"n{depth}:g0"] = {"L2_minus_L1": series_json(a), "DRAM_minus_L2": series_json(b)}
    return {
        "clock_assay": clock_report,
        "clock_lock": {
            "lock_applied": lock_ok,
            "limitation": lock_record.get("lock_limitation") if isinstance(lock_record, Mapping) else "clock session unavailable",
        },
        "identification": {
            "status": identification.status,
            "failures": list(identification.failures),
            "predicates": dict(identification.predicates),
            "effects": effects_json({"K": pskr.k, "R": pskr.r}),
            "label": identification.identification_label,
        },
        "placement": {
            "status": placement.status,
            "failures": list(placement.failures),
            "predicates": dict(placement.predicates),
            "identification_label": placement.identification_label,
        },
        "holdout": {
            "status": holdout.status,
            "failures": list(holdout.failures),
            "predicates": dict(holdout.predicates),
            "identification_label": holdout.identification_label,
        },
        "staged_successor_eligible_before_external_gates": staged.successor_eligible,
        "successor_eligible": successor_eligible,
        "issue_frontier": effects_json({"P": pskr.p, "S": pskr.s, "K": pskr.k, "R": pskr.r}),
        "completion_frontier": effects_json({"E": pskr.e, "C": pskr.c}),
        "completion_route_contrasts": completion_route,
        "pskr_missing": list(pskr.missing),
        "errors": placement_errors,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    outcome = report.get("outcome", {})
    stages = report.get("stages", {})
    issue = stages.get("issue_frontier", {})
    completion = stages.get("completion_route_contrasts", {})
    zero_gap_k = {
        route: issue.get("K", {}).get(f"{route}:g0", {}).get("median")
        for route in ("l1", "l2", "dram")
    }
    completion_depth6 = completion.get("n6:g0", {})
    lines = [
        "# SM86 LDGSTS burst-release live reduction",
        "",
        f"Generated: `{report.get('generated_at')}`",
        f"Evidence root: `{report.get('evidence_root')}`",
        f"Binary SHA-256: `{report.get('binary_sha256')}`",
        "",
        "## Outcome",
        "",
        f"- Terminal status: **{outcome.get('status')}**",
        f"- Identification: `{stages.get('identification', {}).get('status')}`",
        f"- Placement: `{stages.get('placement', {}).get('status')}`",
        f"- L2 holdout: `{stages.get('holdout', {}).get('status')}`",
        f"- Successor eligible: `{stages.get('successor_eligible')}`",
        "",
        "Issue and post-issue completion frontiers are retained separately; no scalar timing score is formed.",
        "",
        "## Frontier snapshot",
        "",
        f"- Zero-gap issue K medians (L1/L2/DRAM): `{zero_gap_k}`",
        f"- Zero-gap issue R medians (depth 1/4/5/6/8): `{[issue.get('R', {}).get(f'n{depth}:g0', {}).get('median') for depth in (1, 4, 5, 6, 8)]}`",
        f"- Depth-6 completion contrasts (L2−L1, DRAM−L2): `{ {name: value.get('median') for name, value in completion_depth6.items()} }`",
        f"- LDG control K median: `{stages.get('clock_assay', {}).get('K', {}).get('median')}`",
        "",
        "## Admission",
        "",
        f"- Static: `{report.get('admission', {}).get('static')}`",
        f"- Direct reduction: `{report.get('admission', {}).get('direct')}`",
        f"- Achieved-gap matching (diagnostic medians): `{report.get('direct', {}).get('gap_admission', {}).get('admitted')}`",
        f"- NCU comparison labels: `{report.get('admission', {}).get('ncu')}`",
        f"- Raw achieved-gap outliers retained: `{report.get('direct', {}).get('raw_gap_outlier_count')}`",
        f"- Clock lock applied: `{stages.get('clock_lock', {}).get('lock_applied')}`",
        f"- Marker-scoped NVBit → simulator smoke: `{report.get('pipeline', {}).get('passed')}`",
        "",
        "## Scientific boundary",
        "",
        "NCU observations are comparison-only route/work/placement evidence. The trace, static control sidecar, architecture card, and pchase card remain the only simulator inputs. A failed or unknown gate leaves simulator timing unchanged; this assay does not identify physical queue size or release location.",
        "",
    ]
    if stages.get("clock_lock", {}).get("limitation"):
        lines.extend([f"Clock limitation: `{stages['clock_lock']['limitation']}`", ""])
    return "\n".join(lines)


def run(root: Path, *, force: bool) -> int:
    root = root.resolve()
    matrix_path = root / "metadata" / "matrix.json"
    matrix = load_json(matrix_path)
    matrix_cells = matrix.get("cells") if isinstance(matrix, Mapping) else None
    matrix_errors: list[str] = []
    try:
        frozen.validate_matrix(matrix_cells)
    except frozen.AnalysisError as error:
        matrix_errors.append(str(error))
    binary = HERE / "ldgsts_burst_release"
    if not binary.is_file() or binary.stat().st_size <= 0:
        matrix_errors.append(f"missing target binary: {binary}")
    direct_reductions, direct = reduce_direct(root)
    ncu = reduce_ncu(root)
    static_ok, static = static_admission(root)
    pipeline = pipeline_status(root)
    stages = build_stages(root, direct_reductions, direct, ncu, static_ok, static)
    if matrix_errors:
        outcome_status = "invalid_experiment"
    elif not static_ok or direct.get("errors") or direct.get("raw_gap_outliers"):
        outcome_status = "invalid_experiment"
    elif ncu.get("high_cv_cells"):
        outcome_status = "unknown_requires_ncu_extension"
    elif stages.get("successor_eligible"):
        outcome_status = "identified_successor_eligible"
    elif stages.get("identification", {}).get("status") in frozen.IDENTIFICATION_LABELS:
        outcome_status = "identified_but_no_successor"
    else:
        outcome_status = stages.get("identification", {}).get("status", "invalid_experiment")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "generated_at": now(),
        "evidence_root": str(root),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary) if binary.is_file() else None,
        "matrix_errors": matrix_errors,
        "admission": {
            "static": static_ok,
            "direct": not direct.get("errors") and not direct.get("raw_gap_outliers"),
            "ncu": ncu.get("all_cells_admitted"),
        },
        "static": static,
        "direct": direct,
        "ncu": ncu,
        "pipeline": pipeline,
        "stages": stages,
        "outcome": {"status": outcome_status},
        "claim_ceiling": [
            "GA102 burst issue/completion discriminator only",
            "no physical queue-size, release-event, or pipeline-stage claim",
            "NCU values are not simulator inputs",
            "JADE/HETR and CPU-GPU co-simulation remain deferred",
        ],
    }
    analysis_dir = root / "analysis"
    report_path = analysis_dir / "report.json"
    if report_path.exists() and not force:
        raise LiveAnalysisError(f"analysis report exists; pass --force to replace: {report_path}")
    write_json(report_path, report)
    (analysis_dir / "README.md").write_text(markdown_report(report), encoding="utf-8")
    write_json(
        analysis_dir / "complete.json",
        {
            "complete": True,
            "kind": "ldgsts_live_analysis",
            "key": root.name,
            "required_files": ["report.json", "README.md"],
            "completed_at": now(),
            "terminal_status": outcome_status,
        },
    )
    print(json.dumps({"analysis": str(report_path), "status": outcome_status}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args.root, force=args.force)
    except (LiveAnalysisError, OSError, ValueError, frozen.AnalysisError) as error:
        print(f"ldgsts live analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
