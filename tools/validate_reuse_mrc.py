#!/usr/bin/env python3
"""Validate the report-only reuse-interval observer in full replay output.

This validator intentionally consumes one ``full.jsonl.zst`` artifact.  It
does not read NCU profiles, does not run fast replay, and does not select any
simulator parameter.  The observed curve is a bounded reuse-interval proxy,
not a claim about undocumented set-associative hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonl_io import open_jsonl_reader


def _int(row: dict[str, Any], key: str, errors: list[str], context: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{context}: {key} must be a non-negative integer")
        return 0
    return value


def _check_launch(launch: dict[str, Any], context: str, errors: list[str]) -> bool:
    reuse = launch.get("trace_metrics", {}).get("reuse_interval")
    if not isinstance(reuse, dict):
        errors.append(f"{context}: missing trace_metrics.reuse_interval")
        return False
    if reuse.get("schema") != "reuse_interval_observer_v1":
        errors.append(f"{context}: unexpected reuse observer schema")
    if reuse.get("ordering_scope") != "within_warp_trace_order_only":
        errors.append(f"{context}: unexpected ordering scope")
    if reuse.get("within_record_order") != "same_logical_tick_no_order_available":
        errors.append(f"{context}: unexpected within-record order declaration")
    capacities = reuse.get("proxy_capacity_records")
    if not isinstance(capacities, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in capacities
    ):
        errors.append(f"{context}: invalid proxy capacity vector")
        return False
    if capacities != sorted(set(capacities)):
        errors.append(f"{context}: proxy capacities must be sorted and unique")

    line_refs = _int(reuse, "line_references", errors, context)
    sector_refs = _int(reuse, "sector_references", errors, context)
    line_cold = _int(reuse, "cold_line_references", errors, context)
    line_reused = _int(reuse, "reused_line_references", errors, context)
    sector_cold = _int(reuse, "cold_sector_references", errors, context)
    sector_reused = _int(reuse, "reused_sector_references", errors, context)
    if line_refs != line_cold + line_reused:
        errors.append(f"{context}: line cold/reuse conservation failed")
    if sector_refs != sector_cold + sector_reused:
        errors.append(f"{context}: sector cold/reuse conservation failed")

    line_hist = reuse.get("line_reuse_interval_histogram")
    sector_hist = reuse.get("sector_reuse_interval_histogram")
    if not isinstance(line_hist, list) or not isinstance(sector_hist, list):
        errors.append(f"{context}: missing reuse histograms")
        return False
    line_overflow = _int(reuse, "line_reuse_interval_overflow", errors, context)
    sector_overflow = _int(reuse, "sector_reuse_interval_overflow", errors, context)
    if sum(line_hist) + line_overflow != line_reused:
        errors.append(f"{context}: line histogram mass failed")
    if sum(sector_hist) + sector_overflow != sector_reused:
        errors.append(f"{context}: sector histogram mass failed")

    for field, refs in (
        ("line_proxy_miss_references", line_refs),
        ("sector_proxy_miss_references", sector_refs),
    ):
        misses = reuse.get(field)
        if not isinstance(misses, list) or len(misses) != len(capacities):
            errors.append(f"{context}: {field} length does not match capacities")
            continue
        values = [_int({"value": value}, "value", errors, context) for value in misses]
        if any(value > refs for value in values):
            errors.append(f"{context}: {field} exceeds reference count")
        if values != sorted(values, reverse=True):
            errors.append(f"{context}: {field} is not monotone non-increasing")
        if capacities and capacities[0] == 0 and values[0] != refs:
            errors.append(f"{context}: zero-capacity proxy must miss every reference")

    for field in (
        "streams",
        "stream_unique_lines",
        "maximum_stream_working_set_lines",
        "records_without_decoded_span",
        "records_with_empty_span",
    ):
        _int(reuse, field, errors, context)
    return True


def validate(path: Path, expected_statuses: dict[str, int] | None) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    launches = 0
    checked = 0
    errors: list[str] = []
    with open_jsonl_reader(path) as reader:
        for index, raw in enumerate(reader):
            try:
                capture = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"capture[{index}]: invalid JSON: {exc}")
                continue
            status = str(capture.get("status", "missing"))
            statuses[status] = statuses.get(status, 0) + 1
            for launch_index, launch in enumerate(capture.get("launches") or []):
                launches += 1
                if isinstance(launch, dict) and _check_launch(
                    launch, f"capture[{index}].launch[{launch_index}]", errors
                ):
                    checked += 1
    if expected_statuses is not None and statuses != expected_statuses:
        errors.append(f"status profile mismatch: expected {expected_statuses}, got {statuses}")
    return {
        "path": str(path),
        "captures": sum(statuses.values()),
        "statuses": statuses,
        "launches": launches,
        "launches_checked": checked,
        "ok": not errors,
        "errors": errors[:100],
        "error_count": len(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_jsonl", type=Path, help="one full.jsonl.zst artifact")
    parser.add_argument(
        "--expected-status",
        action="append",
        metavar="STATUS=COUNT",
        help="repeat to enforce a complete-corpus status profile",
    )
    args = parser.parse_args()
    expected = None
    if args.expected_status:
        expected = {}
        for token in args.expected_status:
            try:
                status, count = token.split("=", 1)
                expected[status] = int(count)
            except ValueError as exc:
                parser.error(f"invalid --expected-status {token!r}: {exc}")
    result = validate(args.full_jsonl, expected)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
