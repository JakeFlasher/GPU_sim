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
    fields = ("index", "name", "uuid", "compute_capability", "driver_version")
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
    if len(outputs) != 2 or outputs[0].read_bytes() != outputs[1].read_bytes():
        raise ReportError(f"{workload_id}: simulator twins are not byte-identical")
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
    return {
        "status": "measured" if values else statuses[0],
        "value": None if not values else _mean(values),
        "repeat_values": values,
        "repeat_count": len(repeats),
        "coverage": len(values),
        "minimum": None if not values else min(values),
        "maximum": None if not values else max(values),
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


def _row(
    workload: Mapping[str, Any],
    metric_schema: Mapping[str, Any],
    records: Mapping[tuple[str, str], Mapping[str, Any]],
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
    issue = _finite(timing.get("diagnostic_last_issue_cycles"), f"{workload_id}.issue")
    terminal_value = None if terminal is None else _finite(terminal, f"{workload_id}.terminal")

    comparisons: dict[str, object] = {}
    for metric in metric_schema["metrics"]:
        metric_id = str(metric["id"])
        ncu = _ncu_metric(repeats, str(metric["ncu_metric"]))
        if metric_id == "gpc_elapsed_cycles":
            simulator = {
                "status": "lower_bound" if frontier.get("status") != "complete" else "modeled",
                "value": terminal_value,
                "frontier": dict(frontier),
            }
        elif metric_id == "sm_active_cycles":
            simulator = {"status": "issue_only_diagnostic", "value": issue}
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
    ncu_walls = [
        _finite(repeat["ncu_profile_wall_seconds"], f"{workload_id}.ncu_wall", positive=True)
        for repeat in repeats
    ]
    twin_walls = [
        _finite(value, f"{workload_id}.sim_wall", positive=True)
        for value in simulator_result["twin_wall_seconds"]
    ]
    simulator_wall = _finite(
        simulator_result["simulator_wall_seconds"],
        f"{workload_id}.simulator_wall_seconds",
        positive=True,
    )
    if not math.isclose(simulator_wall, statistics.median(twin_walls), rel_tol=0, abs_tol=1e-12):
        raise ReportError(f"{workload_id}: simulator wall is not the twin median")
    counter_apes = [comparisons[metric]["ape_pct"] for metric in COUNTER_IDS]
    return {
        "workload_id": workload_id,
        "display_name": workload["display_name"],
        "functional_analogue": workload["functional_analogue"],
        "launches": 1,
        "timing": {
            "gpu_kernel_seconds": gpu_seconds,
            "ncu_profile_wall_seconds_mean": _mean(ncu_walls),
            "ncu_profile_wall_seconds_total": sum(ncu_walls),
            "simulator_wall_seconds": simulator_wall,
            "simulator_twin_wall_seconds_total": sum(twin_walls),
            "gpu_execution_over_simulator_wall": gpu_seconds / simulator_wall,
            "ncu_profile_wall_over_simulator_wall": _mean(ncu_walls) / simulator_wall,
        },
        "accuracy": {
            "terminal_lower_bound_mape_pct": comparisons["gpc_elapsed_cycles"]["ape_pct"],
            "issue_frontier_mape_pct": comparisons["sm_active_cycles"]["ape_pct"],
            "counter_mape_pct": _mean_optional(counter_apes),
            "counter_coverage": sum(value is not None for value in counter_apes),
            "counter_metric_count": len(counter_apes),
        },
        "completion_frontier": dict(frontier),
        "metrics": comparisons,
    }


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gpu = sum(float(row["timing"]["gpu_kernel_seconds"]) for row in rows)
    simulator = sum(float(row["timing"]["simulator_wall_seconds"]) for row in rows)
    ncu_mean = sum(float(row["timing"]["ncu_profile_wall_seconds_mean"]) for row in rows)
    ncu_total = sum(float(row["timing"]["ncu_profile_wall_seconds_total"]) for row in rows)
    terminal = [row["accuracy"]["terminal_lower_bound_mape_pct"] for row in rows]
    issue = [row["accuracy"]["issue_frontier_mape_pct"] for row in rows]
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
            "launches": len(rows),
        }
    return {
        "workloads": len(rows),
        "launches": sum(int(row["launches"]) for row in rows),
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
            "counter_mape_pct": _mean_optional(counter),
            "counter_coverage": sum(value is not None for value in counter),
            "counter_metric_launch_pairs": len(counter),
            "hit_rates": hit_summaries,
        },
    }


def build_report(campaign_path: Path) -> dict[str, Any]:
    campaign = _load(campaign_path)
    if campaign.get("schema_version") != 1:
        raise ReportError("campaign schema_version must be 1")
    spec = _input_document(campaign, "spec")
    metric_schema = _input_document(campaign, "metrics")
    profile = _input_document(campaign, "profile")
    validate_spec.validate_spec(spec, metric_schema, profile)
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
    if len(selected) not in {4, len(workload_ids)}:
        raise ReportError("report admits exactly the four-row smoke or full frozen matrix")
    records = _record_index(campaign, [str(workload["id"]) for workload in selected])
    host = _host_identity(records)
    rows = [_row(workload, metric_schema, records) for workload in selected]
    simulator_hashes = {
        records[(str(workload["id"]), "simulator")]["result"]["simulator_binary"]["sha256"]
        for workload in selected
    }
    if len(simulator_hashes) != 1:
        raise ReportError("simulator binary identity changed within the campaign")
    return {
        "schema_version": 1,
        "report_id": f"{campaign['benchmark_id']}_{'smoke' if len(rows) == 4 else 'final'}_report_v1",
        "status": "smoke" if len(rows) == 4 else "final",
        "campaign": {
            "path": str(campaign_path.resolve()),
            "sha256": _sha256(campaign_path),
            "repository": campaign["repository"],
            "simulator_binary_sha256": next(iter(simulator_hashes)),
        },
        "hardware": host,
        "formulas": {
            "ape_pct": "abs(simulator - ncu) / ncu * 100 when ncu > 0",
            "hit_rate_pct": "100 * NCU hit sectors / (hit sectors + miss sectors)",
            "hit_rate_pp_error": "abs(simulator_percent - ncu_percent)",
            "workload_mape": "arithmetic mean over covered launch-metric APE values",
            "total_mape": "launch-weighted arithmetic mean over covered launch-metric APE values",
            "gpu_execution_over_simulator_wall": "sum NCU gpu__time_duration.sum seconds / sum simulator median process wall seconds",
            "ncu_profile_wall_over_simulator_wall": "sum mean NCU capture-process wall seconds / sum simulator median process wall seconds",
        },
        "rows": rows,
        "total": _totals(rows),
        "caveats": [
            "NCU is comparison-only ground truth and is absent from simulator commands and profile inputs.",
            "Cache rates are launch-global weighted rates derived from NCU hit/miss sector sums; direct per-subunit NCU .pct ratios are not used.",
            "Every smoke completion frontier is an explicit lower bound with retained unmodeled retirement tails.",
            "GPU/simulator and NCU/simulator values are named time ratios, not unqualified speedups.",
            "Smoke results contain one NCU repeat per workload; the final campaign requires three independent repeats.",
        ],
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
        f"Revision: `{report['campaign']['repository']['revision']}` on `{report['campaign']['repository']['branch']}`.  ",
        f"GPU: `{report['hardware']['name']}` (`{report['hardware']['uuid']}`, CC {report['hardware']['compute_capability']}, driver {report['hardware']['driver_version']}).  ",
        f"Simulator: `{report['campaign']['simulator_binary_sha256']}`.",
        "",
        "## Primary accuracy and cost",
        "",
        "| Workload | Launches | GPU kernel ms | NCU profile s | Simulator s | GPU / simulator | NCU / simulator | Terminal lower-bound MAPE | Issue-frontier MAPE | Counter MAPE |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {launches} | {gpu} | {ncu} | {sim} | {gpu_ratio} | {ncu_ratio} | {terminal}% | {issue}% | {counter}% |".format(
                name=row["display_name"],
                launches=row["launches"],
                gpu=_fmt(row["timing"]["gpu_kernel_seconds"] * 1000.0, 4),
                ncu=_fmt(row["timing"]["ncu_profile_wall_seconds_mean"], 4),
                sim=_fmt(row["timing"]["simulator_wall_seconds"], 4),
                gpu_ratio=_fmt(row["timing"]["gpu_execution_over_simulator_wall"], 3),
                ncu_ratio=_fmt(row["timing"]["ncu_profile_wall_over_simulator_wall"], 3),
                terminal=_fmt(row["accuracy"]["terminal_lower_bound_mape_pct"]),
                issue=_fmt(row["accuracy"]["issue_frontier_mape_pct"]),
                counter=_fmt(row["accuracy"]["counter_mape_pct"]),
            )
        )
    lines.append(
        "| **Total / launch-weighted** | {launches} | {gpu} | {ncu} | {sim} | {gpu_ratio} | {ncu_ratio} | {terminal}% | {issue}% | {counter}% |".format(
            launches=total["launches"],
            gpu=_fmt(total["timing"]["gpu_kernel_seconds"] * 1000.0, 4),
            ncu=_fmt(total["timing"]["ncu_profile_wall_seconds_mean_sum"], 4),
            sim=_fmt(total["timing"]["simulator_wall_seconds"], 4),
            gpu_ratio=_fmt(total["timing"]["gpu_execution_over_simulator_wall"], 3),
            ncu_ratio=_fmt(total["timing"]["ncu_profile_wall_over_simulator_wall"], 3),
            terminal=_fmt(total["accuracy"]["terminal_lower_bound_mape_pct"]),
            issue=_fmt(total["accuracy"]["issue_frontier_mape_pct"]),
            counter=_fmt(total["accuracy"]["counter_mape_pct"]),
        )
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
        prefix = "smoke" if report["status"] == "smoke" else "final"
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
                    "launches": report["total"]["launches"],
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
