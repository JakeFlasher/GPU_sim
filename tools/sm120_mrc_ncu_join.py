#!/usr/bin/env python3
"""Create a diagnostic MRC/service versus NCU-sector table.

This is an observation join, not a calibration loop.  The MRC is a logical
fully-associative scenario, while NCU counters are collected under their own
counter/replay protocol.  The output therefore keeps unit/scope labels and
reports zero-reference cases explicitly instead of tuning service parameters
to the scored rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _value(metrics: dict[str, Any], name: str) -> float | None:
    row = metrics.get(name)
    if isinstance(row, (int, float)):
        return float(row)
    if not isinstance(row, dict):
        return None
    try:
        return float(row.get("value"))
    except (TypeError, ValueError):
        return None


def _sum_matching(metrics: dict[str, Any], predicate) -> float | None:
    values: list[float] = []
    for name, row in metrics.items():
        if not predicate(str(name)):
            continue
        if isinstance(row, dict):
            if row.get("unit") != "sector":
                continue
            try:
                values.append(float(row.get("value")))
            except (TypeError, ValueError):
                continue
        elif isinstance(row, (int, float)):
            # sm120_memory_service_model already reduces profile rows to
            # sector-valued numeric observations.
            values.append(float(row))
    return sum(values) if values else None


def ncu_sector_views(metrics: dict[str, Any]) -> dict[str, float | None]:
    # Direct L1 request op totals avoid lookup/eviction submetrics, which are
    # different domains and would double count the same request.
    l1_requests = _sum_matching(
        metrics,
        lambda name: name.startswith("l1tex__t_sectors_")
        and "_op_" in name
        and "_lookup_" not in name
        and "evict" not in name
        and name.endswith(".sum"),
    )
    l1_lookup_misses = _sum_matching(
        metrics,
        lambda name: name.startswith("l1tex__t_sectors_")
        and "_lookup_miss" in name
        and name.endswith(".sum"),
    )
    l2_lookup_misses = _sum_matching(
        metrics,
        lambda name: name.startswith("lts__t_sectors_")
        and "lookup_miss" in name
        and "evict" not in name
        and name.endswith(".sum"),
    )
    dram_sectors = _sum_matching(
        metrics,
        lambda name: name.startswith("dram__sectors_op_") and name.endswith(".sum"),
    )
    return {
        "l1_request_sectors_observed": l1_requests,
        "l1_lookup_miss_sectors_observed": l1_lookup_misses,
        "l2_lookup_miss_sectors_observed": l2_lookup_misses,
        "dram_sectors_observed": dram_sectors,
    }


def comparison(predicted: float, observed: float | None) -> dict[str, Any]:
    if observed is None:
        return {"status": "unavailable_observation", "predicted": predicted, "observed": None}
    absolute = abs(predicted - observed)
    if observed == 0:
        return {
            "status": "zero_reference_absolute_only",
            "predicted": predicted,
            "observed": observed,
            "absolute_error": absolute,
        }
    return {
        "status": "diagnostic_relative_error",
        "predicted": predicted,
        "observed": observed,
        "absolute_error": absolute,
        "signed_relative_error": (predicted - observed) / observed,
        "absolute_relative_error": absolute / abs(observed),
    }


def build(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for service in payload.get("prediction_rows", []):
        metrics = (service.get("ncu_diagnostic") or {}).get("metrics") or {}
        observed = ncu_sector_views(metrics)
        for record in service.get("records", []):
            demand = float(record.get("trace_demand_sectors", 0) or 0)
            l1_misses = float(record.get("l1_misses", 0) or 0)
            dram_misses = float(record.get("logical_dram_misses", 0) or 0)
            rows.append(
                {
                    "capture_name": service.get("capture_name"),
                    "launch_index": service.get("launch_index"),
                    "kernel_id": service.get("kernel_id"),
                    "capacity_sectors": record.get("capacity_sectors"),
                    "mrc_scenario": service.get("mrc_scenario"),
                    "mrc_available": service.get("mrc_available", False),
                    "ncu_join_status": (service.get("ncu_diagnostic") or {}).get("status"),
                    "ncu_sector_views": observed,
                    "comparisons": {
                        "modeled_demand_vs_l1_requests": comparison(
                            demand, observed["l1_request_sectors_observed"]
                        ),
                        "logical_l1_misses_vs_l1_lookup_misses": comparison(
                            l1_misses, observed["l1_lookup_miss_sectors_observed"]
                        ),
                        "logical_dram_misses_vs_dram_sectors": comparison(
                            dram_misses, observed["dram_sectors_observed"]
                        ),
                    },
                    "scope_note": "diagnostic only; MRC is logical trace-order demand and NCU counters use an independent collection domain",
                }
            )
    ape: dict[str, list[float]] = {}
    for row in rows:
        for name, item in row["comparisons"].items():
            value = item.get("absolute_relative_error")
            if isinstance(value, (float, int)):
                ape.setdefault(name, []).append(float(value))
    summary = {
        "row_count": len(rows),
        "ncu_joined_rows": sum(row["ncu_join_status"] == "separate_observation_join" for row in rows),
        "mrc_rows": sum(bool(row["mrc_available"]) for row in rows),
        "median_absolute_relative_error_by_comparison": {
            name: statistics.median(values) if values else None for name, values in sorted(ape.items())
        },
    }
    return {
        "schema_version": 1,
        "model": "sm120_mrc_ncu_diagnostic_join_v1",
        "service_model": payload.get("model"),
        "mrc_input": payload.get("mrc_input"),
        "mrc_scenario": payload.get("mrc_scenario"),
        "summary": summary,
        "rows": rows,
        "interpretation": [
            "NCU metrics are independent observations, never MRC inputs or fitted parameters.",
            "Counter-name aggregation is a diagnostic sector view and may not share the trace launch population.",
            "No row identifies cache set hashing, physical associativity, MSHR count, or replacement policy.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.service.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
