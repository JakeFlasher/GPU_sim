#!/usr/bin/env python3
"""Reduce one retained SM86 demo campaign into JSON and Markdown tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.sm86_ncu_demo_benchmark import validate_spec


STAGES = ("correctness", "ncu", "nvbit", "static", "simulator")
COUNTER_IDS = (
    "instructions",
    "global_load_sectors",
    "global_store_sectors",
    "dram_read_sectors",
)
HIT_RATE_IDS = (
    "l1_read_hit_rate",
    "l1_unified_hit_rate",
    "l2_unified_hit_rate",
)


class ReportError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"{path}: root must be an object")
    return value


def _verify_artifact(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise ReportError(f"{label}: artifact is missing: {path}")
    size = record.get("size_bytes")
    if size is not None and path.stat().st_size != size:
        raise ReportError(f"{label}: artifact size differs")
    if record.get("sha256") != _sha256(path):
        raise ReportError(f"{label}: artifact SHA-256 differs")
    return path


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{label}: expected a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ReportError(f"{label}: expected {'positive ' if positive else ''}nonnegative finite value")
    return result


def _ape(simulator: float | None, ncu: float | None) -> float | None:
    if simulator is None or ncu is None or ncu <= 0:
        return None
    return abs(simulator - ncu) / ncu * 100.0


def _absolute_error(simulator: float | None, ncu: float | None) -> float | None:
    if simulator is None or ncu is None:
        return None
    return abs(simulator - ncu)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ReportError("cannot average an empty value sequence")
    return statistics.fmean(values)


def _mean_optional(values: Sequence[float | None]) -> float | None:
    covered = [value for value in values if value is not None]
    return None if not covered else _mean(covered)


def _input_document(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    record = manifest.get("inputs", {}).get(name)
    if not isinstance(record, Mapping):
        raise ReportError(f"campaign input {name!r} is missing")
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if record.get("sha256") != _sha256(path):
        raise ReportError(f"campaign input {name!r} SHA-256 differs")
    return _load(path)


def _evidence_archive(campaign_path: Path) -> dict[str, Any] | None:
    manifest_path = campaign_path.parent / "evidence" / "archive_manifest.json"
    if not manifest_path.is_file():
        return None
    document = _load(manifest_path)
    if document.get("schema_version") != 1 or document.get("member_root") != "evidence/raw":
        raise ReportError("evidence archive manifest contract differs")
    archive = document.get("archive")
    if not isinstance(archive, Mapping):
        raise ReportError("evidence archive identity is missing")
    _verify_artifact(archive, "evidence archive")
    return {
        "manifest_path": _portable_path(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        **document,
    }


def _record_index(
    manifest: Mapping[str, Any], workload_ids: Sequence[str]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ReportError("campaign records are missing")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ReportError("campaign contains a non-object stage record")
        key = (str(record.get("workload_id")), str(record.get("stage")))
        if key in index:
            raise ReportError(f"duplicate campaign stage record: {key}")
        index[key] = record
    expected = {(workload, stage) for workload in workload_ids for stage in STAGES}
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise ReportError(f"campaign stage matrix differs; missing={missing}, extra={extra}")
    return index


def _host_identity(records: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "index",
        "name",
        "uuid",
        "compute_capability",
        "driver_version",
        "graphics_clock_mhz",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    identities = {
        tuple(record[boundary].get(field) for field in fields)
        for record in records.values()
        for boundary in ("host_pre", "host_post")
    }
    if len(identities) != 1:
        raise ReportError("pre/post hardware identity changed within the campaign")
    identity = next(iter(identities))
    return dict(zip(fields, identity))


def _verify_stage_artifacts(
    workload_id: str, records: Mapping[tuple[str, str], Mapping[str, Any]]
) -> None:
    correctness = records[(workload_id, "correctness")]["result"]
    _verify_artifact(correctness["manifest"], f"{workload_id}.correctness")
    nvbit = records[(workload_id, "nvbit")]["result"]
    for name in ("trace", "stats", "environment"):
        _verify_artifact(nvbit[name], f"{workload_id}.nvbit.{name}")
    if nvbit.get("sidecars_empty") is not True:
        raise ReportError(f"{workload_id}: NVBit exclusion sidecars are not empty")
    static = records[(workload_id, "static")]["result"]
    _verify_artifact(static["control"], f"{workload_id}.static.control")
    simulator = records[(workload_id, "simulator")]["result"]
    _verify_artifact(simulator["simulator_binary"], f"{workload_id}.simulator.binary")
    outputs = [
        _verify_artifact(record, f"{workload_id}.simulator.output")
        for record in simulator["outputs"]
    ]
    if len(outputs) < 3 or any(
        output.read_bytes() != outputs[0].read_bytes() for output in outputs[1:]
    ):
        raise ReportError(
            f"{workload_id}: at least three byte-identical simulator repeats are required"
        )
    ncu = records[(workload_id, "ncu")]["result"]
    for repeat in ncu["repeats"]:
        _verify_artifact(repeat["report"], f"{workload_id}.ncu.report")
        _verify_artifact(repeat["csv"], f"{workload_id}.ncu.csv")


def _ncu_metric(
    repeats: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, object]:
    values: list[float] = []
    statuses: list[str] = []
    for repeat in repeats:
        record = repeat["metrics"].get(metric)
        if not isinstance(record, Mapping):
            raise ReportError(f"NCU metric {metric!r} is missing")
        status = str(record.get("status", "measured"))
        statuses.append(status)
        value = record.get("value")
        if value is not None:
            values.append(_finite(value, f"NCU metric {metric}"))
    if values and len(values) != len(repeats):
        raise ReportError(f"NCU metric {metric!r} has inconsistent repeat coverage")
    mean = None if not values else _mean(values)
    median = None if not values else float(statistics.median(values))
    relative_range = (
        None
        if mean in (None, 0.0)
        else (max(values) - min(values)) / float(mean) * 100.0
    )
    return {
        "status": "measured" if values else statuses[0],
        "value": median,
        "reduction": "median",
        "mean": mean,
        "median": median,
        "repeat_values": values,
        "repeat_count": len(repeats),
        "coverage": len(values),
        "minimum": None if not values else min(values),
        "maximum": None if not values else max(values),
        "relative_range_pct": relative_range,
    }


def _simulator_document(record: Mapping[str, Any], workload_id: str) -> dict[str, Any]:
    result = record["result"]
    path = _verify_artifact(result["outputs"][0], f"{workload_id}.simulator.output")
    document = _load(path)
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1 or not isinstance(launches[0], dict):
        raise ReportError(f"{workload_id}: simulator output must contain one launch")
    return document


def _projection_map(launch: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = launch.get("metric_predictions")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ReportError("simulator output does not contain the 8-row metric projection")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("ncu_metric"), str):
            raise ReportError("simulator metric projection is malformed")
        metric = row["ncu_metric"]
        if metric in result:
            raise ReportError(f"duplicate simulator metric projection {metric!r}")
        result[metric] = row
    return result


def _simulator_projection(row: Mapping[str, Any], metric_id: str) -> dict[str, object]:
    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping):
        raise ReportError(f"{metric_id}: simulator outcome is malformed")
    status = str(outcome.get("status"))
    if status != "predicted":
        return {"status": status, "value": None, "detail": dict(outcome)}
    value = outcome.get("value")
    if not isinstance(value, Mapping):
        raise ReportError(f"{metric_id}: predicted simulator value is malformed")
    kind = value.get("kind")
    numeric = value.get("value")
    if numeric is None:
        return {"status": "unavailable_zero_denominator", "value": None, "kind": kind}
    numeric = _finite(numeric, f"{metric_id}.simulator")
    if kind == "fraction":
        numeric *= 100.0
    elif kind != "exact":
        raise ReportError(f"{metric_id}: unsupported simulator value kind {kind!r}")
    return {"status": "predicted", "value": numeric, "kind": kind}


def _metric_comparison(
    metric: Mapping[str, Any],
    ncu: Mapping[str, object],
    simulator: Mapping[str, object],
) -> dict[str, object]:
    ncu_value = ncu["value"]
    simulator_value = simulator["value"]
    ncu_numeric = None if ncu_value is None else float(ncu_value)
    simulator_numeric = None if simulator_value is None else float(simulator_value)
    result: dict[str, object] = {
        "ncu": dict(ncu),
        "simulator": dict(simulator),
        "ape_pct": _ape(simulator_numeric, ncu_numeric),
        "absolute_error": _absolute_error(simulator_numeric, ncu_numeric),
    }
    if metric.get("hit_rate_member"):
        result["absolute_percentage_point_error"] = result["absolute_error"]
    return result


def _scalar_comparison(
    ncu: float | None,
    simulator: float | None,
    *,
    unit: str,
    simulator_status: str,
) -> dict[str, object]:
    return {
        "unit": unit,
        "ncu": ncu,
        "simulator": simulator,
        "simulator_status": simulator_status,
        "ape_pct": _ape(simulator, ncu),
        "absolute_error": _absolute_error(simulator, ncu),
    }


def _timing_envelopes(crosswalk: Mapping[str, Any] | None) -> tuple[float, float]:
    if crosswalk is None:
        return 0.0, 0.0
    active = crosswalk["sm_active_projection"]
    elapsed = crosswalk["gpc_elapsed_projection"]
    return (
        _finite(active["fixed_envelope_cycles"], "SM-active envelope"),
        _finite(elapsed["profiler_envelope_cycles"], "GPC profiler envelope"),
    )


def _row(
    workload: Mapping[str, Any],
    metric_schema: Mapping[str, Any],
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    hardware: Mapping[str, Any],
    timing_crosswalk: Mapping[str, Any] | None,
) -> dict[str, Any]:
    workload_id = str(workload["id"])
    _verify_stage_artifacts(workload_id, records)
    ncu_result = records[(workload_id, "ncu")]["result"]
    repeats = ncu_result["repeats"]
    if not isinstance(repeats, list) or not repeats:
        raise ReportError(f"{workload_id}: NCU repeats are missing")
    simulator_record = records[(workload_id, "simulator")]
    simulator_result = simulator_record["result"]
    document = _simulator_document(simulator_record, workload_id)
    launch = document["launches"][0]
    projections = _projection_map(launch)
    timing = launch.get("timing")
    if not isinstance(timing, Mapping):
        raise ReportError(f"{workload_id}: simulator timing is missing")
    frontier = timing.get("completion_frontier")
    if not isinstance(frontier, Mapping):
        raise ReportError(f"{workload_id}: completion frontier is missing")
    terminal = timing.get("terminal_modeled_event_time")
    raw_issue = _finite(timing.get("diagnostic_last_issue_cycles"), f"{workload_id}.issue")
    raw_terminal = None if terminal is None else _finite(terminal, f"{workload_id}.terminal")
    active_envelope, profiler_envelope = _timing_envelopes(timing_crosswalk)
    issue = raw_issue + active_envelope
    terminal_value = (
        None
        if raw_terminal is None
        else raw_terminal + active_envelope + profiler_envelope
    )

    comparisons: dict[str, object] = {}
    for metric in metric_schema["metrics"]:
        metric_id = str(metric["id"])
        ncu = _ncu_metric(repeats, str(metric["ncu_metric"]))
        if metric_id == "gpc_elapsed_cycles":
            simulator = {
                "status": (
                    "single_kernel_ncu_crosswalk_lower_bound"
                    if frontier.get("status") != "complete"
                    else "single_kernel_ncu_crosswalk"
                ),
                "value": terminal_value,
                "raw_frontier_value": raw_terminal,
                "sm_active_envelope_cycles": active_envelope,
                "profiler_envelope_cycles": profiler_envelope,
                "frontier": dict(frontier),
            }
        elif metric_id == "sm_active_cycles":
            simulator = {
                "status": "single_kernel_ncu_crosswalk",
                "value": issue,
                "raw_frontier_value": raw_issue,
                "sm_active_envelope_cycles": active_envelope,
            }
        elif metric_id == "gpu_kernel_duration":
            simulator = {"status": "not_scored", "value": None}
        else:
            projection = projections.get(str(metric["ncu_metric"]))
            if projection is None:
                raise ReportError(f"{workload_id}: simulator projection {metric['ncu_metric']!r} is missing")
            simulator = _simulator_projection(projection, metric_id)
        comparisons[metric_id] = _metric_comparison(metric, ncu, simulator)

    gpu_ns = comparisons["gpu_kernel_duration"]["ncu"]["value"]
    gpu_seconds = _finite(gpu_ns, f"{workload_id}.gpu_duration") * 1e-9
    sm_clock_mhz = _finite(
        hardware.get("sm_clock_mhz"), f"{workload_id}.sm_clock", positive=True
    )
    simulator_runtime_ns = None if terminal_value is None else terminal_value * 1000.0 / sm_clock_mhz
    ncu_instructions = _finite(
        comparisons["instructions"]["ncu"]["value"],
        f"{workload_id}.ncu_instructions",
        positive=True,
    )
    simulator_instructions = _finite(
        comparisons["instructions"]["simulator"]["value"],
        f"{workload_id}.simulator_instructions",
        positive=True,
    )
    ncu_issue_cycles = _finite(
        comparisons["sm_active_cycles"]["ncu"]["value"],
        f"{workload_id}.ncu_issue_cycles",
        positive=True,
    )
    ncu_terminal_cycles = _finite(
        comparisons["gpc_elapsed_cycles"]["ncu"]["value"],
        f"{workload_id}.ncu_terminal_cycles",
        positive=True,
    )
    derived = {
        "runtime_equivalent": _scalar_comparison(
            float(gpu_ns),
            simulator_runtime_ns,
            unit="ns",
            simulator_status=(
                "single_kernel_ncu_crosswalk_lower_bound"
                if frontier.get("status") != "complete"
                else "single_kernel_ncu_crosswalk"
            ),
        ),
        "issue_instruction_throughput": _scalar_comparison(
            ncu_instructions / ncu_issue_cycles,
            simulator_instructions / issue,
            unit="warp_instructions_per_max_active_cycle",
            simulator_status="single_kernel_ncu_crosswalk",
        ),
        "terminal_instruction_throughput": _scalar_comparison(
            ncu_instructions / ncu_terminal_cycles,
            None
            if terminal_value in (None, 0.0)
            else simulator_instructions / terminal_value,
            unit="warp_instructions_per_terminal_cycle",
            simulator_status=(
                "single_kernel_ncu_crosswalk_lower_bound_derived"
                if frontier.get("status") != "complete"
                else "single_kernel_ncu_crosswalk"
            ),
        ),
    }
    ncu_walls = [
        _finite(repeat["ncu_profile_wall_seconds"], f"{workload_id}.ncu_wall", positive=True)
        for repeat in repeats
    ]
    simulator_walls = [
        _finite(value, f"{workload_id}.sim_wall", positive=True)
        for value in simulator_result["repeat_wall_seconds"]
    ]
    simulator_wall = _finite(
        simulator_result["simulator_wall_seconds_median"],
        f"{workload_id}.simulator_wall_seconds_median",
        positive=True,
    )
    if not math.isclose(
        simulator_wall,
        statistics.median(simulator_walls),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ReportError(f"{workload_id}: simulator wall is not the repeat median")
    counter_apes = [comparisons[metric]["ape_pct"] for metric in COUNTER_IDS]
    repeat_spreads = [
        (comparison["ncu"]["relative_range_pct"], metric_id)
        for metric_id, comparison in comparisons.items()
        if comparison["ncu"]["relative_range_pct"] is not None
    ]
    max_spread, max_spread_metric = max(repeat_spreads, default=(None, None))
    return {
        "workload_id": workload_id,
        "display_name": workload["display_name"],
        "functional_analogue": workload["functional_analogue"],
        "target_launches_per_process": int(workload["target_launch_count"]),
        "ncu_launch_samples": len(repeats),
        "simulator_process_samples": len(simulator_walls),
        "timing": {
            "gpu_kernel_seconds": gpu_seconds,
            "ncu_profile_wall_seconds_mean": _mean(ncu_walls),
            "ncu_profile_wall_seconds_total": sum(ncu_walls),
            "simulator_wall_seconds": simulator_wall,
            "simulator_process_wall_seconds_total": sum(simulator_walls),
            "simulator_wall_seconds_mean": _mean(simulator_walls),
            "simulator_wall_seconds_minimum": min(simulator_walls),
            "simulator_wall_seconds_maximum": max(simulator_walls),
            "simulator_wall_relative_range_pct": simulator_result[
                "simulator_wall_relative_range_pct"
            ],
            "gpu_execution_over_simulator_wall": gpu_seconds / simulator_wall,
            "ncu_profile_wall_over_simulator_wall": _mean(ncu_walls) / simulator_wall,
        },
        "accuracy": {
            "terminal_lower_bound_mape_pct": comparisons["gpc_elapsed_cycles"]["ape_pct"],
            "issue_frontier_mape_pct": comparisons["sm_active_cycles"]["ape_pct"],
            "raw_terminal_lower_bound_mape_pct": _ape(raw_terminal, ncu_terminal_cycles),
            "raw_issue_frontier_mape_pct": _ape(raw_issue, ncu_issue_cycles),
            "runtime_equivalent_mape_pct": derived["runtime_equivalent"]["ape_pct"],
            "instruction_throughput_mape_pct": derived[
                "issue_instruction_throughput"
            ]["ape_pct"],
            "counter_mape_pct": _mean_optional(counter_apes),
            "counter_coverage": sum(value is not None for value in counter_apes),
            "counter_metric_count": len(counter_apes),
        },
        "completion_frontier": dict(frontier),
        "timing_projection": {
            "scope": (
                "raw_frontiers"
                if timing_crosswalk is None
                else str(timing_crosswalk["observation_scope"])
            ),
            "raw_issue_cycles": raw_issue,
            "projected_sm_active_cycles": issue,
            "raw_terminal_cycles": raw_terminal,
            "projected_gpc_elapsed_cycles": terminal_value,
            "sm_active_envelope_cycles": active_envelope,
            "profiler_envelope_cycles": profiler_envelope,
        },
        "derived_metrics": derived,
        "ncu_repeatability": {
            "repeat_count": len(repeats),
            "max_relative_range_pct": max_spread,
            "max_relative_range_metric": max_spread_metric,
        },
        "simulator_repeatability": {
            "repeat_count": len(simulator_walls),
            "wall_relative_range_pct": simulator_result[
                "simulator_wall_relative_range_pct"
            ],
            "deterministic_predictions": simulator_result.get(
                "deterministic_repeats"
            )
            is True,
        },
        "metrics": comparisons,
    }


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gpu = sum(float(row["timing"]["gpu_kernel_seconds"]) for row in rows)
    simulator = sum(float(row["timing"]["simulator_wall_seconds"]) for row in rows)
    ncu_mean = sum(float(row["timing"]["ncu_profile_wall_seconds_mean"]) for row in rows)
    ncu_total = sum(float(row["timing"]["ncu_profile_wall_seconds_total"]) for row in rows)
    terminal = [row["accuracy"]["terminal_lower_bound_mape_pct"] for row in rows]
    issue = [row["accuracy"]["issue_frontier_mape_pct"] for row in rows]
    raw_terminal = [
        row["accuracy"].get("raw_terminal_lower_bound_mape_pct") for row in rows
    ]
    raw_issue = [row["accuracy"].get("raw_issue_frontier_mape_pct") for row in rows]
    runtime = [row["accuracy"]["runtime_equivalent_mape_pct"] for row in rows]
    throughput = [
        row["accuracy"]["instruction_throughput_mape_pct"] for row in rows
    ]
    counter = [
        row["metrics"][metric]["ape_pct"]
        for row in rows
        for metric in COUNTER_IDS
    ]
    hit_summaries: dict[str, object] = {}
    for metric in HIT_RATE_IDS:
        mape = [row["metrics"][metric]["ape_pct"] for row in rows]
        pp = [row["metrics"][metric]["absolute_percentage_point_error"] for row in rows]
        hit_summaries[metric] = {
            "mape_pct": _mean_optional(mape),
            "mean_absolute_percentage_point_error": _mean_optional(pp),
            "coverage": sum(value is not None for value in mape),
            "percentage_point_coverage": sum(value is not None for value in pp),
            "workloads": len(rows),
        }
    worst_repeatability = max(
        rows,
        key=lambda row: float(row["ncu_repeatability"]["max_relative_range_pct"] or -1.0),
    )
    return {
        "workloads": len(rows),
        "target_launches_per_process": sum(
            int(row["target_launches_per_process"]) for row in rows
        ),
        "ncu_launch_samples": sum(int(row["ncu_launch_samples"]) for row in rows),
        "simulator_process_samples": sum(
            int(row["simulator_process_samples"]) for row in rows
        ),
        "timing": {
            "gpu_kernel_seconds": gpu,
            "ncu_profile_wall_seconds_mean_sum": ncu_mean,
            "ncu_profile_wall_seconds_campaign_total": ncu_total,
            "simulator_wall_seconds": simulator,
            "gpu_execution_over_simulator_wall": gpu / simulator,
            "ncu_profile_wall_over_simulator_wall": ncu_mean / simulator,
        },
        "accuracy": {
            "terminal_lower_bound_mape_pct": _mean_optional(terminal),
            "terminal_lower_bound_coverage": sum(value is not None for value in terminal),
            "issue_frontier_mape_pct": _mean_optional(issue),
            "issue_frontier_coverage": sum(value is not None for value in issue),
            "raw_terminal_lower_bound_mape_pct": _mean_optional(raw_terminal),
            "raw_terminal_lower_bound_coverage": sum(
                value is not None for value in raw_terminal
            ),
            "raw_issue_frontier_mape_pct": _mean_optional(raw_issue),
            "raw_issue_frontier_coverage": sum(value is not None for value in raw_issue),
            "runtime_equivalent_mape_pct": _mean_optional(runtime),
            "runtime_equivalent_coverage": sum(
                value is not None for value in runtime
            ),
            "instruction_throughput_mape_pct": _mean_optional(throughput),
            "instruction_throughput_coverage": sum(
                value is not None for value in throughput
            ),
            "counter_mape_pct": _mean_optional(counter),
            "counter_coverage": sum(value is not None for value in counter),
            "counter_metric_launch_pairs": len(counter),
            "hit_rates": hit_summaries,
        },
        "ncu_repeatability": {
            "max_relative_range_pct": worst_repeatability["ncu_repeatability"]["max_relative_range_pct"],
            "workload_id": worst_repeatability["workload_id"],
            "metric_id": worst_repeatability["ncu_repeatability"]["max_relative_range_metric"],
        },
        "simulator_repeatability": {
            "maximum_wall_relative_range_pct": max(
                float(row["simulator_repeatability"]["wall_relative_range_pct"])
                for row in rows
            ),
            "all_predictions_deterministic": all(
                row["simulator_repeatability"]["deterministic_predictions"] is True
                for row in rows
            ),
        },
    }


def _evaluate_accuracy_gate(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    total: Mapping[str, Any],
) -> dict[str, Any]:
    accuracy = total["accuracy"]
    hit_rates = accuracy["hit_rates"]
    observed = {
        "terminal_cycle_mape_pct": (
            accuracy["terminal_lower_bound_mape_pct"],
            accuracy["terminal_lower_bound_coverage"],
        ),
        "runtime_equivalent_mape_pct": (
            accuracy["runtime_equivalent_mape_pct"],
            accuracy["runtime_equivalent_coverage"],
        ),
        "issue_cycle_mape_pct": (
            accuracy["issue_frontier_mape_pct"],
            accuracy["issue_frontier_coverage"],
        ),
        "instruction_throughput_mape_pct": (
            accuracy["instruction_throughput_mape_pct"],
            accuracy["instruction_throughput_coverage"],
        ),
        "l1_unified_hit_rate_mape_pct": (
            hit_rates["l1_unified_hit_rate"]["mape_pct"],
            hit_rates["l1_unified_hit_rate"]["coverage"],
        ),
        "l2_unified_hit_rate_mape_pct": (
            hit_rates["l2_unified_hit_rate"]["mape_pct"],
            hit_rates["l2_unified_hit_rate"]["coverage"],
        ),
        "counter_traffic_mape_pct": (
            accuracy["counter_mape_pct"],
            accuracy["counter_coverage"],
        ),
    }
    checks: dict[str, Any] = {}
    for name, contract in gate["checks"].items():
        value, coverage = observed[name]
        passed = (
            value is not None
            and float(value) <= float(contract["maximum"])
            and int(coverage) >= int(contract["required_coverage"])
        )
        result: dict[str, Any] = {
            "value": value,
            "maximum": contract["maximum"],
            "coverage": coverage,
            "required_coverage": contract["required_coverage"],
        }
        if "maximum_mean_absolute_percentage_point_error" in contract:
            metric_id = (
                "l1_unified_hit_rate"
                if name.startswith("l1_")
                else "l2_unified_hit_rate"
            )
            summary = hit_rates[metric_id]
            pp_value = summary["mean_absolute_percentage_point_error"]
            pp_coverage = summary["percentage_point_coverage"]
            pp_passed = (
                pp_value is not None
                and float(pp_value)
                <= float(contract["maximum_mean_absolute_percentage_point_error"])
                and int(pp_coverage)
                >= int(contract["required_percentage_point_coverage"])
            )
            result.update(
                {
                    "mean_absolute_percentage_point_error": pp_value,
                    "maximum_mean_absolute_percentage_point_error": contract[
                        "maximum_mean_absolute_percentage_point_error"
                    ],
                    "percentage_point_coverage": pp_coverage,
                    "required_percentage_point_coverage": contract[
                        "required_percentage_point_coverage"
                    ],
                }
            )
            passed = passed and pp_passed
        result["passed"] = passed
        checks[name] = result

    repeat_checks = {
        "workloads": {
            "value": len(rows),
            "required": gate["required_workloads"],
            "passed": len(rows) >= int(gate["required_workloads"]),
        },
        "ncu_repeats_per_workload": {
            "value": min(int(row["ncu_launch_samples"]) for row in rows),
            "required": gate["required_ncu_repeats"],
            "passed": min(int(row["ncu_launch_samples"]) for row in rows)
            >= int(gate["required_ncu_repeats"]),
        },
        "simulator_repeats_per_workload": {
            "value": min(int(row["simulator_process_samples"]) for row in rows),
            "required": gate["required_simulator_repeats"],
            "passed": min(int(row["simulator_process_samples"]) for row in rows)
            >= int(gate["required_simulator_repeats"]),
        },
        "simulator_prediction_determinism": {
            "value": total["simulator_repeatability"][
                "all_predictions_deterministic"
            ],
            "required": True,
            "passed": total["simulator_repeatability"][
                "all_predictions_deterministic"
            ]
            is True,
        },
    }
    technical_pass = all(check["passed"] for check in checks.values()) and all(
        check["passed"] for check in repeat_checks.values()
    )
    return {
        "gate_id": gate["gate_id"],
        "technical_status": "pass" if technical_pass else "fail",
        "metric_checks": checks,
        "repeat_checks": repeat_checks,
        "manual_approval_required": gate["manual_approval_required"],
        "manual_approval_status": "pending",
        "candidate_promotable_for_user_review": technical_pass,
    }


def _caveats(
    status: str,
    ncu_repeats: int,
    simulator_repeats: int,
    timing_crosswalk: Mapping[str, Any] | None = None,
) -> list[str]:
    phase = status
    return [
        "NCU is comparison-only ground truth and is absent from simulator commands and profile inputs.",
        "Cache rates are launch-global weighted rates derived from NCU hit/miss sector sums; direct per-subunit NCU .pct ratios are not used.",
        f"Every {phase} raw completion frontier remains explicit, including retained unmodeled retirement tails.",
        (
            "Timing accuracy uses raw simulator frontiers without a measurement crosswalk."
            if timing_crosswalk is None
            else "Single-kernel timing accuracy applies the independently calibrated SM-active and NCU GPC-profiler envelopes after replay; those envelopes are never passed to the simulator and do not erase raw frontiers."
        ),
        "GPU/simulator and NCU/simulator values are named time ratios, not unqualified speedups.",
        f"Results contain {ncu_repeats} independent application-replay NCU launches and {simulator_repeats} independent simulator processes per workload; raw samples remain in JSON.",
        "Runtime-equivalent values convert the single-kernel GPC-cycle projection with the independently recorded SM clock; incomplete retirement tails remain typed on the raw frontier.",
    ]


def build_report(campaign_path: Path) -> dict[str, Any]:
    campaign = _load(campaign_path)
    if campaign.get("schema_version") != 1:
        raise ReportError("campaign schema_version must be 1")
    spec = _input_document(campaign, "spec")
    metric_schema = _input_document(campaign, "metrics")
    profile = _input_document(campaign, "profile")
    gate = _input_document(campaign, "gate")
    timing_crosswalk = (
        _input_document(campaign, "timing_crosswalk")
        if "timing_crosswalk" in campaign.get("inputs", {})
        else None
    )
    if spec.get("timing_crosswalk") is not None and timing_crosswalk is None:
        raise ReportError("campaign omitted the specification's timing crosswalk identity")
    if timing_crosswalk is not None:
        validate_spec.validate_timing_crosswalk(timing_crosswalk)
    validate_spec.validate_spec(spec, metric_schema, profile)
    validate_spec.validate_accuracy_gate(gate)
    workload_ids = [str(workload["id"]) for workload in spec["workloads"]]
    records_present = campaign.get("records", [])
    selected_ids = sorted(
        {
            str(record.get("workload_id"))
            for record in records_present
            if isinstance(record, Mapping)
        }
    )
    selected = [workload for workload in spec["workloads"] if workload["id"] in selected_ids]
    admitted_counts = (
        {len(workload_ids)}
        if spec.get("matrix_role") == "independent_holdout"
        else {4, len(workload_ids)}
    )
    if len(selected) not in admitted_counts:
        raise ReportError("report admits exactly the four-row smoke or full frozen matrix")
    records = _record_index(campaign, [str(workload["id"]) for workload in selected])
    host = _host_identity(records)
    rows = [
        _row(workload, metric_schema, records, host, timing_crosswalk)
        for workload in selected
    ]
    simulator_hashes = {
        records[(str(workload["id"]), "simulator")]["result"]["simulator_binary"]["sha256"]
        for workload in selected
    }
    if len(simulator_hashes) != 1:
        raise ReportError("simulator binary identity changed within the campaign")
    status = (
        "holdout"
        if spec.get("matrix_role") == "independent_holdout"
        else "smoke"
        if len(rows) == 4
        else "final"
    )
    total = _totals(rows)
    gate_result = _evaluate_accuracy_gate(gate, rows, total)
    ncu_repeats = min(int(row["ncu_launch_samples"]) for row in rows)
    simulator_repeats = min(int(row["simulator_process_samples"]) for row in rows)
    return {
        "schema_version": 1,
        "report_id": f"{campaign['benchmark_id']}_{status}_report_v1",
        "status": status,
        "campaign": {
            "path": _portable_path(campaign_path),
            "sha256": _sha256(campaign_path),
            "repository": campaign["repository"],
            "simulator_binary_sha256": next(iter(simulator_hashes)),
        },
        "evidence_archive": _evidence_archive(campaign_path),
        "hardware": host,
        "timing_crosswalk": timing_crosswalk,
        "formulas": {
            "ncu_repeat_reduction": "median of independent NCU repeats; raw values, mean, min, max, and relative range retained",
            "ape_pct": "abs(simulator - ncu) / ncu * 100 when ncu > 0",
            "hit_rate_pct": "100 * NCU hit sectors / (hit sectors + miss sectors)",
            "hit_rate_pp_error": "abs(simulator_percent - ncu_percent)",
            "workload_mape": "arithmetic mean over covered launch-metric APE values",
            "total_mape": "launch-weighted arithmetic mean over covered launch-metric APE values",
            "gpu_execution_over_simulator_wall": "sum NCU gpu__time_duration.sum seconds / sum simulator median process wall seconds",
            "ncu_profile_wall_over_simulator_wall": "sum mean NCU capture-process wall seconds / sum simulator median process wall seconds",
            "projected_sm_active_cycles": "raw issue frontier + independently calibrated single-kernel SM-active envelope",
            "projected_gpc_elapsed_cycles": "raw terminal frontier + SM-active envelope + independently calibrated NCU GPC profiler envelope",
            "runtime_equivalent_ns": "projected GPC elapsed cycles * 1000 / observed_sm_clock_mhz",
            "issue_instruction_throughput": "smsp instructions / projected maximum SM-active cycles",
        },
        "rows": rows,
        "total": total,
        "accuracy_gate": gate_result,
        "caveats": _caveats(
            status, ncu_repeats, simulator_repeats, timing_crosswalk
        ),
    }


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if 0 < abs(numeric) < 0.01:
        return f"{numeric:.3e}"
    return f"{numeric:.{digits}f}"


def _pct(value: object, digits: int = 2) -> str:
    return "n/a" if value is None else f"{_fmt(value, digits)}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = report["rows"]
    total = report["total"]
    lines = [
        f"# SM86 NCU-grounded simulator {report['status']} report",
        "",
        f"Revision: `{report['campaign']['repository']['revision']}` on `{report['campaign']['repository']['branch']}`.",
        f"GPU: `{report['hardware']['name']}` (`{report['hardware']['uuid']}`, CC {report['hardware']['compute_capability']}, driver {report['hardware']['driver_version']}).",
        f"Simulator: `{report['campaign']['simulator_binary_sha256']}`.",
        "",
        "## Primary accuracy and cost",
        "",
        "| Workload | NCU launches / sim runs | GPU kernel ms | NCU profile s | Simulator s | GPU / simulator | Runtime MAPE | GPC-cycle MAPE | SM-active MAPE | Instruction-throughput MAPE | Counter MAPE |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {samples} | {gpu} | {ncu} | {sim} | {gpu_ratio} | {runtime}% | {terminal}% | {issue}% | {throughput}% | {counter}% |".format(
                name=row["display_name"],
                samples=f"{row['ncu_launch_samples']} / {row['simulator_process_samples']}",
                gpu=_fmt(row["timing"]["gpu_kernel_seconds"] * 1000.0, 4),
                ncu=_fmt(row["timing"]["ncu_profile_wall_seconds_mean"], 4),
                sim=_fmt(row["timing"]["simulator_wall_seconds"], 4),
                gpu_ratio=_fmt(row["timing"]["gpu_execution_over_simulator_wall"], 3),
                runtime=_fmt(row["accuracy"]["runtime_equivalent_mape_pct"]),
                terminal=_fmt(row["accuracy"]["terminal_lower_bound_mape_pct"]),
                issue=_fmt(row["accuracy"]["issue_frontier_mape_pct"]),
                throughput=_fmt(
                    row["accuracy"]["instruction_throughput_mape_pct"]
                ),
                counter=_fmt(row["accuracy"]["counter_mape_pct"]),
            )
        )
    lines.append(
        "| **Total / launch-weighted** | {samples} | {gpu} | {ncu} | {sim} | {gpu_ratio} | {runtime}% | {terminal}% | {issue}% | {throughput}% | {counter}% |".format(
            samples=f"{total['ncu_launch_samples']} / {total['simulator_process_samples']}",
            gpu=_fmt(total["timing"]["gpu_kernel_seconds"] * 1000.0, 4),
            ncu=_fmt(total["timing"]["ncu_profile_wall_seconds_mean_sum"], 4),
            sim=_fmt(total["timing"]["simulator_wall_seconds"], 4),
            gpu_ratio=_fmt(total["timing"]["gpu_execution_over_simulator_wall"], 3),
            runtime=_fmt(total["accuracy"]["runtime_equivalent_mape_pct"]),
            terminal=_fmt(total["accuracy"]["terminal_lower_bound_mape_pct"]),
            issue=_fmt(total["accuracy"]["issue_frontier_mape_pct"]),
            throughput=_fmt(total["accuracy"]["instruction_throughput_mape_pct"]),
            counter=_fmt(total["accuracy"]["counter_mape_pct"]),
        )
    )
    lines.extend(
        [
            "",
            f"NCU profiler campaign wall across retained repeats: {_fmt(total['timing']['ncu_profile_wall_seconds_campaign_total'], 4)} s. The table shows the mean capture-process wall per workload profile.",
            "",
            "## Raw simulator frontiers",
            "",
            "| Workload | Raw issue cycles | Projected SM-active cycles | Raw issue MAPE | Raw terminal cycles | Projected GPC cycles | Raw terminal MAPE |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        projection = row["timing_projection"]
        lines.append(
            "| {name} | {raw_issue} | {active} | {issue_mape} | {raw_terminal} | {gpc} | {terminal_mape} |".format(
                name=row["display_name"],
                raw_issue=_fmt(projection["raw_issue_cycles"]),
                active=_fmt(projection["projected_sm_active_cycles"]),
                issue_mape=_pct(row["accuracy"]["raw_issue_frontier_mape_pct"]),
                raw_terminal=_fmt(projection["raw_terminal_cycles"]),
                gpc=_fmt(projection["projected_gpc_elapsed_cycles"]),
                terminal_mape=_pct(
                    row["accuracy"]["raw_terminal_lower_bound_mape_pct"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "The scored timing columns apply only the bound single-kernel measurement crosswalk. Raw issue and terminal frontiers remain separate simulator evidence.",
            "",
            "## NCU repeatability",
            "",
            "| Workload | NCU repeats | NCU maximum relative range | Metric | Simulator repeats | Simulator wall relative range |",
            "| :--- | ---: | ---: | :--- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['ncu_repeatability']['repeat_count']} | {_pct(row['ncu_repeatability']['max_relative_range_pct'])} | {row['ncu_repeatability']['max_relative_range_metric']} | {row['simulator_repeatability']['repeat_count']} | {_pct(row['simulator_repeatability']['wall_relative_range_pct'])} |"
        )
    lines.extend(
        [
            "",
            "## Cache and traffic accuracy",
            "",
            "| Workload | L1 read MAPE / pp | L1 unified MAPE / pp | L2 MAPE / pp | Instructions APE | Global load APE | Global store APE | DRAM read APE |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        metrics = row["metrics"]
        pair = lambda metric: f"{_pct(metrics[metric]['ape_pct'])} / {_fmt(metrics[metric]['absolute_percentage_point_error'])} pp"
        lines.append(
            "| {name} | {l1r} | {l1} | {l2} | {inst}% | {ld}% | {st}% | {dram}% |".format(
                name=row["display_name"],
                l1r=pair("l1_read_hit_rate"),
                l1=pair("l1_unified_hit_rate"),
                l2=pair("l2_unified_hit_rate"),
                inst=_fmt(metrics["instructions"]["ape_pct"]),
                ld=_fmt(metrics["global_load_sectors"]["ape_pct"]),
                st=_fmt(metrics["global_store_sectors"]["ape_pct"]),
                dram=_fmt(metrics["dram_read_sectors"]["ape_pct"]),
            )
        )
    lines.extend(["", "## Accuracy gate", ""])
    lines.extend(
        [
            f"Technical status: **{report['accuracy_gate']['technical_status'].upper()}**. Manual approval: **{report['accuracy_gate']['manual_approval_status']}**.",
            "",
            "| Check | Observed | Maximum | Coverage | Required | Pass |",
            "| :--- | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for name, check in report["accuracy_gate"]["metric_checks"].items():
        lines.append(
            f"| {name} | {_fmt(check['value'])}% | {_fmt(check['maximum'])}% | {check['coverage']} | {check['required_coverage']} | {'yes' if check['passed'] else 'no'} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    lines.extend(
        [
            "",
            "Regenerate and validate:",
            "",
            "```bash",
            f".venv/bin/python workloads/research_probes/sm86_ncu_demo_benchmark/report.py --campaign {report['campaign']['path']}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _json_bytes(report: Mapping[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        campaign = args.campaign.resolve()
        report = build_report(campaign)
        prefix = report["status"]
        output_json = (args.output_json or campaign.parent / f"{prefix}_report.json").resolve()
        output_md = (args.output_md or campaign.parent / f"{prefix}_report.md").resolve()
        json_bytes = _json_bytes(report)
        markdown = render_markdown(report).encode()
        if args.validate_only:
            if output_json.read_bytes() != json_bytes or output_md.read_bytes() != markdown:
                raise ReportError("retained report differs from deterministic regeneration")
        else:
            output_json.write_bytes(json_bytes)
            output_md.write_bytes(markdown)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "workloads": len(report["rows"]),
                    "ncu_launch_samples": report["total"]["ncu_launch_samples"],
                    "simulator_process_samples": report["total"][
                        "simulator_process_samples"
                    ],
                    "output_json": str(output_json),
                    "output_md": str(output_md),
                },
                sort_keys=True,
            )
        )
        return 0
    except (KeyError, OSError, TypeError, ValueError, ReportError) as error:
        print(f"sm86_ncu_demo_report: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
