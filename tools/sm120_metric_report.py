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
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
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
        "wavefront": {"wavefront"},
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
            "source_key": f"metric_predictions.{prediction.get('key', prediction['ncu_metric'])}",
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
                "source_key": f"cache_lookup_predictions.{metric_key}.{metric}",
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
            "status": "predicted" if status in {"modeled", "approximate"} and estimate is not None else str(status or "unavailable"),
            "estimate": estimate,
            "reason": metric.get("semantics") if estimate is None else None,
            "source_key": f"native_metrics.{key}",
            "semantics": metric.get("semantics"),
        })
    return entries


def selected_entries(launch: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """One prediction per metric/launch, selected without inspecting NCU.

    Prefer a finite declared metric, then cache observer, then native metric,
    then the existing timing fallback. Preserve other representations in the
    row, but do not count aliases as independent launches. External measured
    values are not simulator predictions.
    """
    candidates = metric_entries(launch)
    for metric, estimate in timing_predictions(launch, mode).items():
        candidates.append({
            "metric": metric, "unit": "cycles",
            "status": "predicted" if estimate is not None else "unavailable",
            "estimate": estimate, "source_key": f"{mode}_timing.{metric}",
            "semantics": "model-relative timing endpoint; not an identified hardware clock",
        })
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in candidates:
        groups[entry["metric"]].append(entry)
    result = []
    for entries in groups.values():
        chosen = next((entry for entry in entries
                       if entry["status"] == "predicted" and entry["estimate"] is not None), entries[0])
        row = dict(chosen)
        alternatives = [entry for entry in entries if entry is not chosen]
        if alternatives:
            row["alternative_predictions"] = alternatives
        predicted = [entry for entry in entries
                     if entry["status"] == "predicted" and entry.get("estimate") is not None]
        # A post-replay reducer must not silently choose between incompatible
        # endpoints.  Equal numeric aliases remain compatible; differing
        # values or units are retained as evidence but excluded from scoring.
        if len(predicted) > 1:
            values = {float(entry["estimate"]) for entry in predicted}
            units = {entry.get("unit", "") for entry in predicted}
            if len(values) > 1 or len(units) > 1:
                row["status"] = "conflict"
                row["estimate"] = None
                row["reason"] = "incompatible_prediction_alternatives"
                row["alternative_conflict"] = True
        result.append(row)
    return result


def add_observation(
    table: dict[str, dict[str, Any]],
    entry: dict[str, Any],
    target_metrics: dict[str, Any],
    mode: str,
    capture: str,
    join_status: str,
) -> dict[str, Any]:
    name = entry["metric"]
    workload_family = capture.split("_", 1)[0]
    row = table.setdefault(name, {
        # `family` is retained as a compatibility alias for the metric family.
        "family": family(name),
        "metric_family": family(name),
        "workload_families": set(),
        "modes": set(),
        "prediction_status": collections.Counter(),
        "reasons": collections.Counter(),
        "predicted": 0,
        "compared": 0,
        "target_available": 0,
        "unit_mismatch": 0,
        "target_missing": 0,
        "zero_opportunity": 0,
        "prediction_value_missing": 0,
        "predicted_zero_on_zero_target": 0,
        "predicted_nonzero_on_zero_target": 0,
        "errors": [],
        "relative_errors": [],
        "errors_by_capture": collections.defaultdict(list),
        "relative_by_family": collections.defaultdict(list),
        "relative_by_workload_family": collections.defaultdict(list),
        "relative_by_join": collections.defaultdict(list),
        "source_counts": collections.Counter(),
        "aliases_not_scored": 0,
        "different_numeric_alternatives": 0,
        "alternative_conflicts": 0,
        "zero_targets": 0,
        "exact": 0,
        "captures": set(),
    })
    row["modes"].add(mode)
    row["captures"].add(capture)
    row["workload_families"].add(workload_family)
    status = entry["status"]
    row["prediction_status"][status] += 1
    if entry.get("reason"):
        row["reasons"][entry["reason"]] += 1
    estimate = entry.get("estimate")
    row["source_counts"][entry["source_key"]] += 1
    alternatives = entry.get("alternative_predictions", [])
    row["aliases_not_scored"] += len(alternatives)
    row["different_numeric_alternatives"] += sum(
        candidate["status"] == "predicted"
        and candidate.get("estimate") is not None
        and estimate is not None
        and candidate["estimate"] != estimate
        for candidate in alternatives
    )
    if entry.get("alternative_conflict"):
        row["alternative_conflicts"] += 1
    result = {"metric": name, "source_key": entry["source_key"],
              "estimate": estimate, "model_status": status, "model_unit": entry["unit"]}
    if entry.get("semantics"):
        result["semantics"] = entry["semantics"]
    if alternatives:
        result["alternative_predictions"] = alternatives
    if entry.get("alternative_conflict"):
        result["alternative_conflict"] = True
    target = target_metrics.get(name, {})
    truth = finite(target.get("value")) if isinstance(target, dict) else None
    if status != "predicted" or estimate is None:
        if status == "predicted" and estimate is None:
            row["prediction_value_missing"] += 1
        return {**result, "status": "model_unavailable", "reason": entry.get("reason")}
    row["predicted"] += 1
    if truth is None:
        row["target_missing"] += 1
        return {**result, "status": "target_missing"}
    row["target_available"] += 1
    result.update(target=truth, target_unit=target.get("unit"))
    if truth == 0:
        row["zero_opportunity"] += 1
        if estimate == 0:
            row["predicted_zero_on_zero_target"] += 1
        else:
            row["predicted_nonzero_on_zero_target"] += 1
    # NCU exports these documented count metrics with an empty unit on SM120.
    # This is a name-qualified count relation, not permission to equate arbitrary
    # unitless metrics with wavefronts or to convert timing domains.
    shared_wavefront = name in {
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum",
    } and entry["unit"] == "wavefront" and target.get("unit") == ""
    if target.get("unit") not in allowed_units(entry["unit"]) and not shared_wavefront:
        row["unit_mismatch"] += 1
        return {**result, "status": "unit_mismatch"}
    error = abs(estimate - truth)
    row["compared"] += 1
    row["errors"].append(error)
    row["errors_by_capture"][capture].append(error)
    relative = error / abs(truth) if truth != 0 else None
    if relative is not None:
        row["relative_errors"].append(relative)
        row["relative_by_family"][workload_family].append(relative)
        row["relative_by_workload_family"][workload_family].append(relative)
        row["relative_by_join"][join_status].append(relative)
    else:
        row["zero_targets"] += 1
    row["exact"] += int(error == 0.0)
    return {**result, "status": "compared", "absolute_error": error,
            "relative_error_nonzero_target": relative}


def reduce_mode(
    path: Path,
    mode: str,
    profiles: Path,
    output,
    table: dict[str, dict[str, Any]],
    target_inventory: collections.Counter,
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
            counts[f"capture_status_{capture.get('status', 'unspecified')}"] += 1
            counts["launches"] += len(launches)
            for join in join_ncu_rows(launches, ncu):
                counts[f"join_{join['join_status']}"] += 1
                li, ni = join.get("launch_index"), join.get("ncu_row_index")
                if li is None or ni is None:
                    output.write(json.dumps({"capture": rel, "mode": mode, **join}) + "\n")
                    continue
                launch = launches[li]
                target_metrics = ncu[ni].get("metrics", {})
                target_inventory.update(target_metrics.keys())
                results = []
                for entry in selected_entries(launch, mode):
                    results.append(add_observation(
                        table, entry, target_metrics, mode, rel, join["join_status"]
                    ))
                output.write(json.dumps({"capture": rel, "mode": mode, **join, "results": results}, separators=(",", ":"), allow_nan=False) + "\n")
    return dict(counts)


def render_summary(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    unavailable: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for name, row in sorted(table.items()):
        compared = row["compared"]
        errors = row["errors"]
        workload_relative = {
            name: distribution(values)
            for name, values in sorted(row["relative_by_workload_family"].items())
        }
        item = {
            "family": row["family"],
            "metric_family": row["metric_family"],
            "workload_families": sorted(row["workload_families"]),
            "modes": sorted(row["modes"]),
            "predicted": row["predicted"],
            "compared": compared,
            "target_available": row["target_available"],
            "target_missing": row["target_missing"],
            "unit_mismatch": row["unit_mismatch"],
            "zero_opportunity": row["zero_opportunity"],
            "prediction_value_missing": row["prediction_value_missing"],
            "predicted_zero_on_zero_target": row["predicted_zero_on_zero_target"],
            "predicted_nonzero_on_zero_target": row["predicted_nonzero_on_zero_target"],
            "exact": row["exact"],
            "prediction_status": dict(row["prediction_status"]),
            "reasons": dict(row["reasons"]),
            "absolute_error": distribution(errors),
            "relative_error_nonzero_targets": distribution(row["relative_errors"]),
            "equal_capture_median_absolute_error": distribution([
                statistics.median(values) for values in row["errors_by_capture"].values()
            ]),
            "family_relative_error_nonzero_targets": {
                name: distribution(values) for name, values in sorted(row["relative_by_family"].items())
            },
            # The old key is metric-family stratification.  Keep it for
            # compatibility and expose the workload-family stratification
            # under an unambiguous name.
            "workload_family_relative_error_nonzero_targets": workload_relative,
            "join_relative_error_nonzero_targets": {
                name: distribution(values) for name, values in sorted(row["relative_by_join"].items())
            },
            "source_counts": dict(row["source_counts"]),
            "aliases_not_scored": row["aliases_not_scored"],
            "different_numeric_alternatives": row["different_numeric_alternatives"],
            "alternative_conflicts": row["alternative_conflicts"],
            "zero_targets": row["zero_targets"],
        }
        metrics[name] = item
        if row["predicted"] == 0:
            unavailable[row["metric_family"]].append({
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
    tables: dict[str, dict[str, Any]] = {"full": {}, "fast": {}}
    target_inventory: collections.Counter = collections.Counter()
    with open_jsonl_writer(args.output / "metric_comparisons.jsonl") as output:
        full_counts = reduce_mode(args.full, "full", args.profiles, output, tables["full"], target_inventory)
        fast_counts = reduce_mode(args.fast, "fast", args.profiles, output, tables["fast"], target_inventory)
    by_mode = {mode: render_summary(table) for mode, table in tables.items()}
    available = {name for table in tables.values() for name, row in table.items() if row["predicted"]}
    declared = {name for table in tables.values() for name in table}
    unavailable: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for name in sorted((set(target_inventory) | declared) - available):
        unavailable[family(name)].append({"metric": name, "target_rows": target_inventory[name]})
    summary = {
        "schema_version": 4,
        "scope": "post-replay metric predictions; NCU values never replay/model inputs",
        "selection": "one prediction per metric/launch/mode; finite declared metric, cache observer, native metric, then timing fallback; alternatives retained but not scored; no NCU-based selection",
        "sources": {"full": str(args.full), "fast": str(args.fast), "profiles": str(args.profiles)},
        "jsonl_compression": jsonl_compression_description(),
        "full": full_counts,
        "fast": fast_counts,
        "by_mode": by_mode,
        "available_metric_count": len(available),
        "unavailable_metric_count": sum(map(len, unavailable.values())),
        "unavailable_by_family": dict(sorted(unavailable.items())),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary.md").open("w") as report:
        report.write("# SM120 metric coverage\n\n")
        report.write("Post-replay reduction of emitted predictions; NCU values were not replay inputs.\n\n")
        report.write("Each launch contributes at most once per metric and mode. Full and fast errors are not pooled; aliases and differing alternatives are retained in compressed rows, not counted as extra observations. Relative errors exclude zero targets.\n\n")
        for mode, mode_summary in by_mode.items():
            report.write(f"## {mode.capitalize()} compared metrics\n\n| metric | family | compared | median absolute error | P90 | mean APE | median APE | P90 APE | exact |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for name, item in mode_summary["metrics"].items():
                if item["compared"] == 0:
                    continue
                error = item["absolute_error"]
                relative = item["relative_error_nonzero_targets"]
                mean_ape = relative.get("mean")
                report.write(f"| `{name}` | {item['family']} | {item['compared']} | {error['median']} | {error['p90']} | {mean_ape} | {relative['median']} | {relative['p90']} | {item['exact']} |\n")
        report.write("\n## Currently unavailable model families\n\n")
        report.write("Metrics with no simulator prediction in either retained mode are listed once by family; raw NCU distributions are intentionally omitted.\n\n")
        for family_name, entries in summary["unavailable_by_family"].items():
            names = ", ".join(
                f"`{entry['metric']}` ({entry['target_rows']} target rows)" for entry in entries
            )
            report.write(f"- **{family_name}:** {names}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
