#!/usr/bin/env python3
"""Reduce replay metric predictions against independent NCU profiles.

This is a post-replay report path. It consumes predictions already emitted by
the simulator, never feeds NCU values back into replay, and writes a compact
compressed comparison stream plus a family-grouped summary.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics
from typing import Any

from jsonl_io import jsonl_compression_description, open_jsonl_reader, open_jsonl_writer
from sm120_corpus_compare import predictions as timing_predictions
from sm120_replay_bench import join_ncu_rows, parse_ncu_raw


def distribution(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None, "p90": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p90": values[int((len(values) - 1) * 0.90)],
    }


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def family(metric: str) -> str:
    if metric.startswith("dram__"):
        return "DRAM"
    if metric.startswith("lts__"):
        return "L2"
    if metric.startswith("l1tex__"):
        return "L1/L1TEX"
    if metric.startswith(("sm__", "smsp__", "gpc__", "pmsampling:")):
        return "SM/compute"
    if metric.startswith("derived__"):
        return "derived"
    return metric.split("__", 1)[0]


def allowed_units(unit: str) -> set[str]:
    return {
        "pct": {"%"},
        "count": {"inst", "sector", "request", "line", "wavefront", "byte", "warp"},
        "cycles": {"cycle"},
        "inst/cycle": {"inst/cycle"},
        "warp": {"warp"},
        "inst": {"inst"},
        "sector": {"sector"},
        "byte": {"byte"},
        "request": {"request"},
    }.get(unit, set())


def metric_entries(launch: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize native metric and cache-observer predictions."""
    entries: list[dict[str, Any]] = []
    for prediction in launch.get("metric_predictions", []):
        if not isinstance(prediction, dict) or not prediction.get("ncu_metric"):
            continue
        outcome = prediction.get("outcome")
        if not isinstance(outcome, dict):
            continue
        value = outcome.get("value")
        estimate = value.get("value") if isinstance(value, dict) else None
        estimate = finite(estimate)
        if isinstance(value, dict) and value.get("kind") == "fraction" and estimate is not None:
            estimate *= 100.0
        entries.append({
            "metric": str(prediction["ncu_metric"]),
            "unit": str(prediction.get("unit", "")),
            "status": str(outcome.get("status", "unavailable")),
            "estimate": estimate,
            "reason": (outcome.get("detail") or {}).get("reason")
            if isinstance(outcome.get("detail"), dict) else None,
        })
    for observer in launch.get("cache_lookup_predictions", []):
        if not isinstance(observer, dict) or observer.get("status") != "predicted":
            continue
        counts = observer.get("observer")
        if not isinstance(counts, dict):
            continue
        for field, metric_key in (("hits", "ncu_hit_metric"), ("misses", "ncu_miss_metric")):
            metric = observer.get(metric_key)
            estimate = finite(counts.get(field))
            if not metric:
                continue
            entries.append({
                "metric": str(metric),
                "unit": "count",
                "status": "predicted" if estimate is not None else "unavailable",
                "estimate": estimate,
                "reason": None if estimate is not None else "observer_count_unavailable",
            })
    native = launch.get("native_metrics", {})
    native_metrics = native.get("metrics", {}) if isinstance(native, dict) else {}
    for key, metric in native_metrics.items():
        if not isinstance(metric, dict) or not metric.get("ncu_metric"):
            continue
        estimate = finite(metric.get("value"))
        status = metric.get("status")
        entries.append({
            "metric": str(metric["ncu_metric"]),
            "unit": "cycles" if metric.get("unit") == "cycle" else str(metric.get("unit", "")),
            "status": "predicted" if status in {"modeled", "approximate", "observed_external"} and estimate is not None else "unavailable",
            "estimate": estimate,
            "reason": metric.get("semantics") if estimate is None else None,
            "source_key": key,
        })
    return entries


def add_observation(
    table: dict[str, dict[str, Any]],
    entry: dict[str, Any],
    target_metrics: dict[str, Any],
    mode: str,
    capture: str,
) -> dict[str, Any]:
    name = entry["metric"]
    row = table.setdefault(name, {
        "family": family(name),
        "modes": set(),
        "prediction_status": collections.Counter(),
        "reasons": collections.Counter(),
        "predicted": 0,
        "compared": 0,
        "target_available": 0,
        "unit_mismatch": 0,
        "target_missing": 0,
        "zero_opportunity": 0,
        "errors": [],
        "exact": 0,
        "captures": set(),
    })
    row["modes"].add(mode)
    row["captures"].add(capture)
    status = entry["status"]
    row["prediction_status"][status] += 1
    if entry.get("reason"):
        row["reasons"][entry["reason"]] += 1
    estimate = entry.get("estimate")
    target = target_metrics.get(name, {})
    truth = finite(target.get("value")) if isinstance(target, dict) else None
    if status != "predicted" or estimate is None:
        if status == "predicted":
            row["zero_opportunity"] += 1
        return {"metric": name, "status": "model_unavailable", "reason": entry.get("reason")}
    row["predicted"] += 1
    if truth is None:
        row["target_missing"] += 1
        return {"metric": name, "status": "target_missing"}
    row["target_available"] += 1
    if target.get("unit") not in allowed_units(entry["unit"]):
        row["unit_mismatch"] += 1
        return {"metric": name, "status": "unit_mismatch", "target_unit": target.get("unit")}
    error = abs(estimate - truth)
    row["compared"] += 1
    row["errors"].append(error)
    row["exact"] += int(error == 0.0)
    return {"metric": name, "status": "compared", "absolute_error": error}


def reduce_mode(
    path: Path,
    mode: str,
    profiles: Path,
    output,
    table: dict[str, dict[str, Any]],
) -> dict[str, int]:
    key = "launches" if mode == "full" else "fast_frontier"
    counts = collections.Counter()
    with open_jsonl_reader(path) as stream:
        for line_number, line in enumerate(stream, 1):
            capture = json.loads(line)
            rel = str(capture.get("canonical_capture", ""))
            if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError(f"{path}:{line_number}: invalid capture identity")
            ncu = parse_ncu_raw(profiles / rel / "raw.csv")
            launches = capture.get(key, [])
            if not isinstance(launches, list):
                raise ValueError(f"{path}:{line_number}: {key} is not an array")
            counts["captures"] += 1
            counts["launches"] += len(launches)
            for join in join_ncu_rows(launches, ncu):
                li, ni = join.get("launch_index"), join.get("ncu_row_index")
                if li is None or ni is None:
                    output.write(json.dumps({"capture": rel, "mode": mode, **join}) + "\n")
                    continue
                launch = launches[li]
                target_metrics = ncu[ni].get("metrics", {})
                results = []
                for entry in metric_entries(launch):
                    results.append(add_observation(table, entry, target_metrics, mode, rel))
                for metric, estimate in timing_predictions(launch, mode).items():
                    results.append(add_observation(
                        table,
                        {"metric": metric, "unit": "cycles", "status": "predicted", "estimate": estimate},
                        target_metrics,
                        mode,
                        rel,
                    ))
                output.write(json.dumps({"capture": rel, "mode": mode, **join, "results": results}, separators=(",", ":")) + "\n")
    return dict(counts)


def render_summary(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    unavailable: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for name, row in sorted(table.items()):
        compared = row["compared"]
        errors = row["errors"]
        item = {
            "family": row["family"],
            "modes": sorted(row["modes"]),
            "predicted": row["predicted"],
            "compared": compared,
            "target_available": row["target_available"],
            "target_missing": row["target_missing"],
            "unit_mismatch": row["unit_mismatch"],
            "zero_opportunity": row["zero_opportunity"],
            "exact": row["exact"],
            "prediction_status": dict(row["prediction_status"]),
            "reasons": dict(row["reasons"]),
            "absolute_error": distribution(errors),
        }
        metrics[name] = item
        if row["predicted"] == 0:
            unavailable[row["family"]].append({
                "metric": name,
                "rows": sum(row["prediction_status"].values()),
                "reasons": dict(row["reasons"]),
            })
    return {
        "metrics": metrics,
        "unavailable_by_family": {family_name: names for family_name, names in sorted(unavailable.items())},
        "available_metric_count": sum(item["predicted"] > 0 for item in metrics.values()),
        "compared_metric_count": sum(item["compared"] > 0 for item in metrics.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict[str, Any]] = {}
    with open_jsonl_writer(args.output / "metric_comparisons.jsonl") as output:
        full_counts = reduce_mode(args.full, "full", args.profiles, output, table)
        fast_counts = reduce_mode(args.fast, "fast", args.profiles, output, table)
    summary = {
        "scope": "post-replay metric predictions; NCU values never replay/model inputs",
        "sources": {"full": str(args.full), "fast": str(args.fast), "profiles": str(args.profiles)},
        "jsonl_compression": jsonl_compression_description(),
        "full": full_counts,
        "fast": fast_counts,
        **render_summary(table),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary.md").open("w") as report:
        report.write("# SM120 metric coverage\n\n")
        report.write("Post-replay reduction of emitted predictions; NCU values were not replay inputs.\n\n")
        report.write("## Compared metrics\n\n| metric | family | modes | compared | median absolute error | P90 | exact |\n|---|---|---|---:|---:|---:|---:|\n")
        for name, item in summary["metrics"].items():
            if item["compared"] == 0:
                continue
            error = item["absolute_error"]
            report.write(f"| `{name}` | {item['family']} | {','.join(item['modes'])} | {item['compared']} | {error['median'] if error['median'] is not None else '—'} | {error['p90'] if error['p90'] is not None else '—'} | {item['exact']} |\n")
        report.write("\n## Currently unavailable model families\n\n")
        report.write("Metrics with no simulator prediction in either retained mode are listed once by family; raw NCU distributions are intentionally omitted.\n\n")
        for family_name, entries in summary["unavailable_by_family"].items():
            names = ", ".join(
                f"`{entry['metric']}` ({entry['rows']} rows)" for entry in entries
            )
            report.write(f"- **{family_name}:** {names}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
