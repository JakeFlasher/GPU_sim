#!/usr/bin/env python3
"""Summarize exact shadow locality/MRC metrics from a full replay artifact.

The report is deliberately NCU-independent.  It aggregates the exact
trace-locality reference emitted by ``sector_mrc_observer_v1`` and checks the
same structural invariants as the launch validator.  It never treats an MRC
point as a silicon or NCU target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonl_io import open_jsonl_reader


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open_jsonl_reader(path) as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: root must be an object")
            rows.append(value)
    return rows


def _aggregate(path: Path) -> dict[str, Any]:
    rows = _load(path)
    statuses: dict[str, int] = {}
    successful_captures = 0
    launches = 0
    origins: dict[str, dict[str, Any]] = {}
    hotl: dict[str, Any] = {
        "launches": 0,
        "line_miss_references": [0] * 28,
        "sector_miss_references": [0] * 28,
        "exact_line_miss_references": [0] * 28,
        "exact_sector_miss_references": [0] * 28,
        "line_references": 0,
        "sector_references": 0,
    }
    conditional: dict[str, dict[str, Any]] = {}
    capacity_lines: list[int] | None = None
    monotonic_violations = 0
    checked_launches = 0
    for capture in rows:
        status = str(capture.get("status", "missing"))
        statuses[status] = statuses.get(status, 0) + 1
        if status != "ok":
            continue
        successful_captures += 1
        for launch in capture.get("launches", []):
            if not isinstance(launch, dict):
                continue
            launches += 1
            trace = launch.get("trace_metrics", {})
            mrc = trace.get("locality_mrc") if isinstance(trace, dict) else None
            if not isinstance(mrc, dict):
                continue
            checked_launches += 1
            reuse = trace.get("reuse_interval") if isinstance(trace, dict) else None
            if isinstance(reuse, dict) and isinstance(reuse.get("hotl_sector_miss_references"), list):
                hotl["launches"] += 1
                hotl["line_references"] += int(reuse.get("line_references", 0))
                hotl["sector_references"] += int(reuse.get("sector_references", 0))
                for index, value in enumerate(reuse["hotl_line_miss_references"]):
                    hotl["line_miss_references"][index] += int(value)
                for index, value in enumerate(reuse["hotl_sector_miss_references"]):
                    hotl["sector_miss_references"][index] += int(value)
            for key in ("sm_l1_locality_observer", "l2_system_locality_observer"):
                observer = trace.get(key) if isinstance(trace, dict) else None
                if not isinstance(observer, dict):
                    continue
                row = conditional.setdefault(
                    key,
                    {
                        "schema": observer.get("schema"),
                        "ordering_scope": observer.get("ordering_scope"),
                        "capacity_lines": observer.get("capacity_lines", []),
                        "references": 0,
                        "lower_hits": [0] * len(observer.get("capacity_lines", [])),
                        "upper_hits": [0] * len(observer.get("capacity_lines", [])),
                        "lower_misses": [0] * len(observer.get("capacity_lines", [])),
                        "upper_misses": [0] * len(observer.get("capacity_lines", [])),
                        "ordering_unavailable_references": 0,
                    },
                )
                row["references"] += int(observer.get("references", 0))
                row["ordering_unavailable_references"] += int(
                    observer.get("ordering_unavailable_references", 0)
                )
                for field in ("lower_hits", "upper_hits", "lower_misses", "upper_misses"):
                    for index, value in enumerate(observer.get(field, [])):
                        if index < len(row[field]):
                            row[field][index] += int(value)
            grid = mrc.get("capacity_lines")
            if not isinstance(grid, list) or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in grid
            ):
                raise ValueError(f"{path}: invalid locality MRC capacity grid")
            if capacity_lines is None:
                capacity_lines = grid
            elif capacity_lines != grid:
                raise ValueError(f"{path}: capacity grid changed between launches")
            for origin, row in mrc.get("by_origin", {}).items():
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: invalid MRC row for {origin!r}")
                n = len(grid)
                total = origins.setdefault(
                    str(origin),
                    {
                        "launches": 0,
                        "line_references": 0,
                        "sector_references": 0,
                        "cold_line_references": 0,
                        "cold_sector_references": 0,
                        "line_miss_references": [0] * n,
                        "line_miss_references_upper": [0] * n,
                        "sector_miss_references": [0] * n,
                        "sector_miss_references_upper": [0] * n,
                        "sector_line_tag_miss_references": [0] * n,
                        "sector_resident_fill_miss_references": [0] * n,
                        "sector_line_tag_miss_references_upper": [0] * n,
                        "sector_resident_fill_miss_references_upper": [0] * n,
                        "same_tick_max_batch_lines": 0,
                        "maximum_line_rd_envelope_width": 0,
                    },
                )
                total["launches"] += 1
                for field in (
                    "line_references",
                    "sector_references",
                    "cold_line_references",
                    "cold_sector_references",
                ):
                    total[field] += int(row[field])
                for field in (
                    "line_miss_references",
                    "sector_miss_references",
                    "sector_line_tag_miss_references",
                    "sector_resident_fill_miss_references",
                    "line_miss_references_upper",
                    "sector_miss_references_upper",
                    "sector_line_tag_miss_references_upper",
                    "sector_resident_fill_miss_references_upper",
                ):
                    for index, value in enumerate(row[field]):
                        total[field][index] += int(value)
                total["same_tick_max_batch_lines"] = max(
                    total["same_tick_max_batch_lines"], int(row["same_tick_max_batch_lines"])
                )
                total["maximum_line_rd_envelope_width"] = max(
                    total["maximum_line_rd_envelope_width"],
                    int(row["maximum_line_rd_envelope_width"]),
                )
                line_misses = row["line_miss_references"]
                sector_misses = row["sector_miss_references"]
                if any(a < b for a, b in zip(line_misses, line_misses[1:])):
                    monotonic_violations += 1
                if any(a < b for a, b in zip(sector_misses, sector_misses[1:])):
                    monotonic_violations += 1
                for index, value in enumerate(line_misses):
                    hotl["exact_line_miss_references"][index] += int(value)
                for index, value in enumerate(sector_misses):
                    hotl["exact_sector_miss_references"][index] += int(value)

    if capacity_lines is None:
        capacity_lines = []
    line_den = sum(hotl["exact_line_miss_references"])
    sector_den = sum(hotl["exact_sector_miss_references"])
    hotl["line_wape_pct"] = (
        100.0
        * sum(
            abs(a - b)
            for a, b in zip(
                hotl["line_miss_references"], hotl["exact_line_miss_references"]
            )
        )
        / line_den
        if line_den
        else None
    )
    hotl["sector_wape_pct"] = (
        100.0
        * sum(
            abs(a - b)
            for a, b in zip(
                hotl["sector_miss_references"], hotl["exact_sector_miss_references"]
            )
        )
        / sector_den
        if sector_den
        else None
    )
    for row in origins.values():
        line_refs = row["line_references"]
        sector_refs = row["sector_references"]
        row["line_miss_ratio"] = [
            value / line_refs if line_refs else None for value in row["line_miss_references"]
        ]
        row["sector_miss_ratio"] = [
            value / sector_refs if sector_refs else None
            for value in row["sector_miss_references"]
        ]
        row["line_miss_ratio_upper"] = [
            value / line_refs if line_refs else None
            for value in row["line_miss_references_upper"]
        ]
        row["sector_miss_ratio_upper"] = [
            value / sector_refs if sector_refs else None
            for value in row["sector_miss_references_upper"]
        ]
        row["sector_envelope_width_percentage_points"] = [
            (upper - lower) * 100.0 / sector_refs if sector_refs else None
            for lower, upper in zip(
                row["sector_miss_references"], row["sector_miss_references_upper"]
            )
        ]
        finite_widths = [
            value
            for value in row["sector_envelope_width_percentage_points"]
            if value is not None
        ]
        row["maximum_sector_envelope_width_percentage_points"] = (
            max(finite_widths) if finite_widths else None
        )
        row["mean_sector_envelope_width_percentage_points"] = (
            sum(finite_widths) / len(finite_widths) if finite_widths else None
        )
        row["cold_sector_fraction"] = (
            row["cold_sector_references"] / sector_refs if sector_refs else None
        )
    return {
        "path": str(path),
        "captures": len(rows),
        "statuses": statuses,
        "successful_captures": successful_captures,
        "launches": launches,
        "launches_with_locality_mrc": checked_launches,
        "capacity_lines": capacity_lines,
        "origins": origins,
        "reuse_interval_hotl": hotl,
        "conditional_locality_observers": conditional,
        "monotonicity_violations": monotonic_violations,
        "ncu_used": False,
        "hardware_mrc_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = _aggregate(args.full_jsonl)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
