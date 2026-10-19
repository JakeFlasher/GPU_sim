#!/usr/bin/env python3
"""Validate report-only reuse-interval and exact locality-MRC observers.

This validator intentionally consumes one ``full.jsonl.zst`` artifact.  It
does not read NCU profiles, does not run fast replay, and does not select any
simulator parameter.  The reuse curve is a bounded interval proxy and the MRC
is an exact fully-associative trace-locality reference, not a claim about
undocumented set-associative hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonl_io import open_jsonl_reader


MRC_CAPACITY_LINES = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    224,
    256,
    512,
    768,
    896,
    960,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    49152,
    65536,
    131072,
    262144,
    524288,
    786432,
    1048576,
]


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
    histogram_valid = True
    for name, histogram in (("line", line_hist), ("sector", sector_hist)):
        for index, value in enumerate(histogram):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{context}: {name} histogram[{index}] is invalid")
                histogram_valid = False
    line_overflow = _int(reuse, "line_reuse_interval_overflow", errors, context)
    sector_overflow = _int(reuse, "sector_reuse_interval_overflow", errors, context)
    if histogram_valid:
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
    hotl_fields = (
        "hotl_line_interval_count",
        "hotl_line_interval_sum",
        "hotl_sector_interval_count",
        "hotl_sector_interval_sum",
        "hotl_line_censored_tail_count",
        "hotl_line_censored_tail_sum",
        "hotl_sector_censored_tail_count",
        "hotl_sector_censored_tail_sum",
    )
    for field in hotl_fields:
        values = reuse.get(field)
        if not isinstance(values, list) or len(values) != 80:
            errors.append(f"{context}: {field} must have 80 bins")
            continue
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{context}: {field}[{index}] is invalid")
    for field in ("hotl_line_miss_references", "hotl_sector_miss_references"):
        values = reuse.get(field)
        if not isinstance(values, list) or len(values) != len(MRC_CAPACITY_LINES):
            errors.append(f"{context}: {field} length mismatch")
            continue
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            errors.append(f"{context}: {field} contains an invalid value")
        if any(a < b for a, b in zip(values, values[1:])):
            errors.append(f"{context}: {field} is not monotone non-increasing")
    return True


def _check_locality_mrc(
    launch: dict[str, Any], context: str, errors: list[str]
) -> bool:
    mrc = launch.get("trace_metrics", {}).get("locality_mrc")
    if not isinstance(mrc, dict):
        errors.append(f"{context}: missing trace_metrics.locality_mrc")
        return False
    if mrc.get("schema") != "sector_mrc_observer_v1":
        errors.append(f"{context}: unexpected locality MRC schema")
    if mrc.get("ordering_scope") != "within_warp_trace_order_only":
        errors.append(f"{context}: unexpected locality MRC ordering scope")
    if mrc.get("same_tick_semantics") != "batch_query_before_insert_with_interval_envelope":
        errors.append(f"{context}: unexpected locality MRC same-tick semantics")
    if mrc.get("capacity_unit") != "distinct_128B_lines":
        errors.append(f"{context}: unexpected locality MRC capacity unit")
    if mrc.get("sector_unit") != "32B_sectors":
        errors.append(f"{context}: unexpected locality MRC sector unit")
    capacities = mrc.get("capacity_lines")
    if capacities != MRC_CAPACITY_LINES:
        errors.append(f"{context}: locality MRC capacity grid mismatch")
        return False
    origins = mrc.get("by_origin")
    if not isinstance(origins, dict):
        errors.append(f"{context}: locality MRC by_origin must be an object")
        return False
    ok = True
    for origin, row in origins.items():
        origin_context = f"{context}.locality_mrc.by_origin[{origin!r}]"
        if not isinstance(row, dict):
            errors.append(f"{origin_context}: row must be an object")
            ok = False
            continue
        line_refs = _int(row, "line_references", errors, origin_context)
        sector_refs = _int(row, "sector_references", errors, origin_context)
        cold_lines = _int(row, "cold_line_references", errors, origin_context)
        cold_sectors = _int(row, "cold_sector_references", errors, origin_context)
        if cold_lines > line_refs or cold_sectors > sector_refs:
            errors.append(f"{origin_context}: cold references exceed totals")
            ok = False
        arrays: dict[str, list[int]] = {}
        for field in (
            "line_miss_references",
            "line_miss_references_upper",
            "sector_miss_references",
            "sector_miss_references_upper",
            "sector_line_tag_miss_references",
            "sector_resident_fill_miss_references",
            "sector_line_tag_miss_references_upper",
            "sector_resident_fill_miss_references_upper",
        ):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != len(MRC_CAPACITY_LINES):
                errors.append(f"{origin_context}: {field} length mismatch")
                ok = False
                continue
            parsed: list[int] = []
            for index, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    errors.append(f"{origin_context}: {field}[{index}] invalid")
                    ok = False
                    parsed.append(0)
                else:
                    parsed.append(value)
            arrays[field] = parsed
        line_misses = arrays.get("line_miss_references", [])
        line_misses_upper = arrays.get("line_miss_references_upper", [])
        sector_misses = arrays.get("sector_miss_references", [])
        sector_misses_upper = arrays.get("sector_miss_references_upper", [])
        tag_misses = arrays.get("sector_line_tag_miss_references", [])
        fill_misses = arrays.get("sector_resident_fill_miss_references", [])
        tag_misses_upper = arrays.get("sector_line_tag_miss_references_upper", [])
        fill_misses_upper = arrays.get("sector_resident_fill_miss_references_upper", [])
        if line_misses:
            if line_misses[0] != line_refs:
                errors.append(f"{origin_context}: MRC(0) line endpoint is not one")
                ok = False
            if any(a < b for a, b in zip(line_misses, line_misses[1:])):
                errors.append(f"{origin_context}: line MRC is not monotone")
                ok = False
            if any(value > line_refs for value in line_misses):
                errors.append(f"{origin_context}: line MRC exceeds references")
                ok = False
        if line_misses_upper:
            if line_misses_upper[0] != line_refs:
                errors.append(f"{origin_context}: upper MRC(0) line endpoint is not one")
                ok = False
            if any(a < b for a, b in zip(line_misses_upper, line_misses_upper[1:])):
                errors.append(f"{origin_context}: upper line MRC is not monotone")
                ok = False
            if any(value > line_refs for value in line_misses_upper):
                errors.append(f"{origin_context}: upper line MRC exceeds references")
                ok = False
            if line_misses and any(
                lower > upper for lower, upper in zip(line_misses, line_misses_upper)
            ):
                errors.append(f"{origin_context}: upper line MRC is below lower curve")
                ok = False
        if sector_misses:
            if sector_misses[0] != sector_refs:
                errors.append(f"{origin_context}: MRC(0) sector endpoint is not one")
                ok = False
            if any(a < b for a, b in zip(sector_misses, sector_misses[1:])):
                errors.append(f"{origin_context}: sector MRC is not monotone")
                ok = False
            if any(value > sector_refs for value in sector_misses):
                errors.append(f"{origin_context}: sector MRC exceeds references")
                ok = False
        if sector_misses_upper:
            if sector_misses_upper[0] != sector_refs:
                errors.append(f"{origin_context}: upper MRC(0) sector endpoint is not one")
                ok = False
            if any(a < b for a, b in zip(sector_misses_upper, sector_misses_upper[1:])):
                errors.append(f"{origin_context}: upper sector MRC is not monotone")
                ok = False
            if any(value > sector_refs for value in sector_misses_upper):
                errors.append(f"{origin_context}: upper sector MRC exceeds references")
                ok = False
            if sector_misses and any(
                lower > upper for lower, upper in zip(sector_misses, sector_misses_upper)
            ):
                errors.append(f"{origin_context}: upper sector MRC is below lower curve")
                ok = False
        if sector_misses and tag_misses and fill_misses:
            for index, (miss, tag, fill) in enumerate(
                zip(sector_misses, tag_misses, fill_misses)
            ):
                if miss != tag + fill:
                    errors.append(
                        f"{origin_context}: sector decomposition failed at capacity {MRC_CAPACITY_LINES[index]}"
                    )
                    ok = False
                if tag > miss or fill > miss:
                    errors.append(f"{origin_context}: sector miss component exceeds total")
                    ok = False
        if sector_misses_upper and tag_misses_upper and fill_misses_upper:
            for index, (miss, tag, fill) in enumerate(
                zip(sector_misses_upper, tag_misses_upper, fill_misses_upper)
            ):
                if miss != tag + fill:
                    errors.append(
                        f"{origin_context}: upper sector decomposition failed at capacity {MRC_CAPACITY_LINES[index]}"
                    )
                    ok = False
                if tag > miss or fill > miss:
                    errors.append(f"{origin_context}: upper sector miss component exceeds total")
                    ok = False
        for field in ("same_tick_max_batch_lines", "maximum_line_rd_envelope_width"):
            _int(row, field, errors, origin_context)
        max_batch = row.get("same_tick_max_batch_lines", 0)
        max_width = row.get("maximum_line_rd_envelope_width", 0)
        if (
            isinstance(max_batch, int)
            and not isinstance(max_batch, bool)
            and isinstance(max_width, int)
            and not isinstance(max_width, bool)
            and max_batch >= 0
            and max_width >= 0
            and max_width > 2 * max(max_batch - 1, 0)
        ):
            errors.append(
                f"{origin_context}: same-tick RD envelope exceeds two-batch bound"
            )
            ok = False
    return ok


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
            if not isinstance(capture, dict):
                errors.append(f"capture[{index}]: root must be an object")
                continue
            status = str(capture.get("status", "missing"))
            statuses[status] = statuses.get(status, 0) + 1
            launch_rows = capture.get("launches", [])
            if not isinstance(launch_rows, list):
                errors.append(f"capture[{index}]: launches must be a list")
                continue
            for launch_index, launch in enumerate(launch_rows):
                launches += 1
                if isinstance(launch, dict):
                    context = f"capture[{index}].launch[{launch_index}]"
                    reuse_ok = _check_launch(launch, context, errors)
                    locality_ok = _check_locality_mrc(launch, context, errors)
                    if reuse_ok and locality_ok:
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
