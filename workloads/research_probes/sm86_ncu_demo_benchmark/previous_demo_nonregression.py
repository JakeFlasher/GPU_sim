#!/usr/bin/env python3
"""Replay the approved eight-workload demo and reduce strict aggregate non-regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_BASELINE = (
    REPO_ROOT
    / "results/captures/sm86_cache_count_targets_20260901_single_sim_final_v1/final_report.json"
)
DEFAULT_SIMULATOR = REPO_ROOT / "sim/target/release/sim-run"
DEFAULT_FEEDBACK = (
    REPO_ROOT
    / "workloads/research_probes/l1tex_admission_backpressure/"
    "memory_write_completion_feedback_store_201.json"
)
DEFAULT_CROSSWALK = HERE / "timing_crosswalk.json"
BASELINE_EVIDENCE = (
    REPO_ROOT
    / "results/captures/sm86_cache_count_targets_20260901_single_sim_final_v1/evidence/raw"
)
AGGREGATE_KEYS = (
    "cache_lookup_count_normalized_error_pct",
    "counter_traffic_mape_pct",
    "instruction_throughput_mape_pct",
    "issue_cycle_mape_pct",
    "runtime_equivalent_mape_pct",
    "terminal_cycle_mape_pct",
)
CACHE_PAIRS = (
    ("l1_read_lookup_hits", "l1_read_lookup_misses", "l1_global_read"),
    ("l1_unified_lookup_hits", "l1_unified_lookup_misses", "l1_unified"),
    ("l2_tex_lookup_hits", "l2_tex_lookup_misses", "l2_l1tex_source"),
)
COUNTERS = {
    "instructions": "smsp__inst_executed.sum",
    "global_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "global_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "dram_read_sectors": "dram__sectors_read.sum",
}


class NonRegressionError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NonRegressionError(f"{path}: root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _ape(simulator: float, ncu: float) -> float:
    if ncu <= 0:
        if simulator == 0:
            return 0.0
        raise NonRegressionError("nonzero simulator value has an exact-zero NCU target")
    return abs(simulator - ncu) / ncu * 100.0


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise NonRegressionError("cannot average an empty metric sequence")
    return statistics.fmean(values)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise NonRegressionError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _projection_map(launch: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in launch.get("metric_predictions", []):
        outcome = row.get("outcome", {})
        if outcome.get("status") != "predicted":
            continue
        result[str(row["ncu_metric"])] = float(outcome["value"]["value"])
    return result


def _candidate_row(
    baseline: Mapping[str, Any],
    document: Mapping[str, Any],
    active_envelope: float,
    profiler_envelope: float,
    sm_clock_mhz: float,
) -> dict[str, Any]:
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise NonRegressionError(f"{baseline['workload_id']}: expected one simulator launch")
    if document.get("memory_admission_feedback_candidate_id") != 201:
        raise NonRegressionError(f"{baseline['workload_id']}: STORE feedback identity differs")
    launch = launches[0]
    timing = launch["timing"]
    raw_issue = float(timing["diagnostic_last_issue_cycles"])
    raw_terminal = float(timing["terminal_modeled_event_time"])
    projected_issue = raw_issue + active_envelope
    projected_terminal = raw_terminal + active_envelope + profiler_envelope
    ncu_issue = float(baseline["metrics"]["sm_active_cycles"]["ncu"]["value"])
    ncu_terminal = float(baseline["metrics"]["gpc_elapsed_cycles"]["ncu"]["value"])
    ncu_runtime = float(baseline["derived_metrics"]["runtime_equivalent"]["ncu"])
    simulator_runtime = projected_terminal * 1000.0 / sm_clock_mhz
    ncu_instructions = float(baseline["metrics"]["instructions"]["ncu"]["value"])
    simulator_instructions = float(launch["counters"]["instructions_issued"])
    if simulator_instructions != ncu_instructions:
        raise NonRegressionError(f"{baseline['workload_id']}: instruction conservation differs")

    projections = _projection_map(launch)
    counter_errors = [
        _ape(
            projections[ncu_metric],
            float(baseline["metrics"][metric_id]["ncu"]["value"]),
        )
        for metric_id, ncu_metric in COUNTERS.items()
    ]
    cache = {str(row["scope"]): row for row in launch["cache_lookup_predictions"]}
    cache_errors: list[float] = []
    for hit_id, miss_id, scope in CACHE_PAIRS:
        ncu_hits = float(baseline["metrics"][hit_id]["ncu"]["value"])
        ncu_misses = float(baseline["metrics"][miss_id]["ncu"]["value"])
        denominator = ncu_hits + ncu_misses
        observer = cache[scope]["observer"]
        cache_errors.extend(
            (
                abs(float(observer["hits"]) - ncu_hits) / denominator * 100.0,
                abs(float(observer["misses"]) - ncu_misses) / denominator * 100.0,
            )
        )
    return {
        "workload_id": baseline["workload_id"],
        "raw_issue_cycles": raw_issue,
        "raw_terminal_cycles": raw_terminal,
        "projected_issue_cycles": projected_issue,
        "projected_terminal_cycles": projected_terminal,
        "issue_cycle_mape_pct": _ape(projected_issue, ncu_issue),
        "terminal_cycle_mape_pct": _ape(projected_terminal, ncu_terminal),
        "runtime_equivalent_mape_pct": _ape(simulator_runtime, ncu_runtime),
        "instruction_throughput_mape_pct": _ape(
            simulator_instructions / projected_issue,
            ncu_instructions / ncu_issue,
        ),
        "counter_traffic_mape_pct": _mean(counter_errors),
        "cache_lookup_count_normalized_error_pct": _mean(cache_errors),
        "cache_lookup_count_maximum_error_pct": max(cache_errors),
    }


def _baseline_aggregate(report: Mapping[str, Any]) -> dict[str, float]:
    accuracy = report["total"]["accuracy"]
    return {
        "cache_lookup_count_normalized_error_pct": float(
            accuracy["cache_lookup_count_normalized_error_pct"]
        ),
        "counter_traffic_mape_pct": float(accuracy["counter_mape_pct"]),
        "instruction_throughput_mape_pct": float(
            accuracy["instruction_throughput_mape_pct"]
        ),
        "issue_cycle_mape_pct": float(accuracy["issue_frontier_mape_pct"]),
        "runtime_equivalent_mape_pct": float(accuracy["runtime_equivalent_mape_pct"]),
        "terminal_cycle_mape_pct": float(accuracy["terminal_lower_bound_mape_pct"]),
    }


def _candidate_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        name: _mean([float(row[name]) for row in rows])
        for name in AGGREGATE_KEYS
    }


def _attach_row_transitions(
    baseline: Mapping[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    baseline_cache = _mean(
        [
            float(baseline["metrics"][metric_id]["pair_normalized_absolute_error_pct"])
            for hit_id, miss_id, _ in CACHE_PAIRS
            for metric_id in (hit_id, miss_id)
        ]
    )
    comparisons = (
        (
            "cache_lookup_count_normalized_error_pct",
            baseline_cache,
            float(candidate["cache_lookup_count_normalized_error_pct"]),
        ),
        (
            "counter_traffic_mape_pct",
            float(baseline["accuracy"]["counter_mape_pct"]),
            float(candidate["counter_traffic_mape_pct"]),
        ),
        (
            "instruction_throughput_mape_pct",
            float(baseline["accuracy"]["instruction_throughput_mape_pct"]),
            float(candidate["instruction_throughput_mape_pct"]),
        ),
        (
            "issue_cycle_mape_pct",
            float(baseline["accuracy"]["issue_frontier_mape_pct"]),
            float(candidate["issue_cycle_mape_pct"]),
        ),
        (
            "runtime_equivalent_mape_pct",
            float(baseline["accuracy"]["runtime_equivalent_mape_pct"]),
            float(candidate["runtime_equivalent_mape_pct"]),
        ),
        (
            "terminal_cycle_mape_pct",
            float(baseline["accuracy"]["terminal_lower_bound_mape_pct"]),
            float(candidate["terminal_cycle_mape_pct"]),
        ),
    )
    transitions = {
        name: {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
            "numerically_non_regressed": candidate_value <= baseline_value + 1e-12,
        }
        for name, baseline_value, candidate_value in comparisons
    }
    candidate["metric_transitions"] = transitions
    return [
        {
            "workload_id": candidate["workload_id"],
            "metric": name,
            **transition,
        }
        for name, transition in transitions.items()
        if not transition["numerically_non_regressed"]
    ]


def _render(report: Mapping[str, Any]) -> str:
    lines = [
        "# Previous eight-workload demo non-regression",
        "",
        f"Revision: `{report['repository_revision']}`.",
        f"Simulator: `{report['simulator_binary_sha256']}`.",
        "",
        "| Aggregate metric | Approved baseline | Candidate | Non-regressed |",
        "| :--- | ---: | ---: | :---: |",
    ]
    for name in AGGREGATE_KEYS:
        transition = report["aggregate_transitions"][name]
        lines.append(
            f"| {name} | {transition['baseline']:.4f}% | "
            f"{transition['candidate']:.4f}% | "
            f"{'yes' if transition['numerically_non_regressed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Strict all-metric result: **{'PASS' if report['strict_all_aggregate_metrics_numerically_non_regressed'] else 'FAIL'}**.",
            f"All candidate aggregate metrics below 20%: **{'yes' if report['all_aggregate_metrics_under_20_pct'] else 'no'}**.",
            "",
            "One deterministic simulator process is used per workload. The approved report's seven NCU repeats remain comparison-only targets and no NCU value appears in a simulator command.",
        ]
    )
    regressions = report["row_level_numeric_regressions"]
    lines.extend(
        [
            "",
            "## Row-level tradeoffs",
            "",
            f"Universal per-cell numeric non-regression: **{'yes' if report['strict_all_row_metrics_numerically_non_regressed'] else 'no'}** ({len(regressions)} regressed workload/metric cells). The approved contract is aggregate-level; these cells are retained for manual review.",
            "",
            "| Workload | Metric | Baseline | Candidate | Delta |",
            "| :--- | :--- | ---: | ---: | ---: |",
        ]
    )
    for row in regressions:
        lines.append(
            f"| {row['workload_id']} | {row['metric']} | {row['baseline']:.4f}% | "
            f"{row['candidate']:.4f}% | {row['delta']:+.4f} pp |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _load(args.baseline.resolve())
    crosswalk = _load(args.crosswalk.resolve())
    feedback = _load(args.feedback.resolve())
    if feedback.get("candidates", [{}])[0].get("candidate_id") != 201:
        raise NonRegressionError("feedback candidate 201 is required")
    active = float(crosswalk["sm_active_projection"]["fixed_envelope_cycles"])
    profiler = float(crosswalk["gpc_elapsed_projection"]["profiler_envelope_cycles"])
    sm_clock = float(baseline["hardware"]["sm_clock_mhz"])
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise NonRegressionError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    simulator_root = output / "simulator"
    simulator_root.mkdir()

    commands: list[list[str]] = []
    rows: list[dict[str, Any]] = []
    walls: list[float] = []
    for baseline_row in baseline["rows"]:
        workload = str(baseline_row["workload_id"])
        destination = simulator_root / workload
        destination.mkdir()
        command = [
            str(args.simulator.resolve()),
            "--trace",
            str(BASELINE_EVIDENCE / "nvbit" / workload),
            "--control",
            str(BASELINE_EVIDENCE / "static" / workload / f"{workload}.control_bits.json"),
            "--arch-card",
            "arch_profiles/cards/sm_86.json",
            "--pchase-card",
            "data/pchase/sm_86_static_card.json",
            "--mode",
            "l1-l2",
            "--cache-scope",
            "launch",
            "--memory-admission-feedback",
            _portable(args.feedback.resolve()),
            "--ncu-shaped-metrics",
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_seconds,
        )
        walls.append(time.monotonic() - started)
        if completed.returncode != 0:
            raise NonRegressionError(f"{workload}: simulator failed: {completed.stderr.strip()}")
        document = json.loads(completed.stdout)
        (destination / "replay-1.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (destination / "replay-1.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        (destination / "command.txt").write_text(
            shlex.join(command) + "\n", encoding="utf-8"
        )
        candidate_row = _candidate_row(
            baseline_row, document, active, profiler, sm_clock
        )
        rows.append(candidate_row)
        commands.append(command)

    baseline_rows = {str(row["workload_id"]): row for row in baseline["rows"]}
    row_level_regressions = [
        regression
        for row in rows
        for regression in _attach_row_transitions(
            baseline_rows[str(row["workload_id"])], row
        )
    ]

    baseline_aggregate = _baseline_aggregate(baseline)
    candidate_aggregate = _candidate_aggregate(rows)
    transitions = {
        name: {
            "baseline": baseline_aggregate[name],
            "candidate": candidate_aggregate[name],
            "numerically_non_regressed": candidate_aggregate[name]
            <= baseline_aggregate[name] + 1e-12,
        }
        for name in AGGREGATE_KEYS
    }
    ncu_values_passed = any(
        str(value) in token
        for baseline_row in baseline["rows"]
        for metric in baseline_row["metrics"].values()
        for value in [metric.get("ncu", {}).get("value")]
        if isinstance(value, (int, float)) and math.isfinite(float(value))
        for command in commands
        for token in command
    )
    result = {
        "schema_version": 1,
        "status": "strict_aggregate_non_regression_pass"
        if all(row["numerically_non_regressed"] for row in transitions.values())
        else "strict_aggregate_non_regression_fail",
        "repository_revision": _git("rev-parse", "HEAD"),
        "repository_branch": _git("branch", "--show-current"),
        "simulator_binary_sha256": _sha256(args.simulator.resolve()),
        "simulator_runs_per_workload": 1,
        "simulator_wall_seconds": sum(walls),
        "baseline_report": {"path": _portable(args.baseline), "sha256": _sha256(args.baseline)},
        "timing_crosswalk": {"path": _portable(args.crosswalk), "sha256": _sha256(args.crosswalk)},
        "memory_admission_feedback": {"path": _portable(args.feedback), "sha256": _sha256(args.feedback)},
        "ncu_values_passed_to_simulator": ncu_values_passed,
        "baseline_aggregate": baseline_aggregate,
        "candidate_aggregate": candidate_aggregate,
        "aggregate_transitions": transitions,
        "strict_all_aggregate_metrics_numerically_non_regressed": all(
            row["numerically_non_regressed"] for row in transitions.values()
        ),
        "strict_all_row_metrics_numerically_non_regressed": not row_level_regressions,
        "row_level_numeric_regressions": row_level_regressions,
        "non_regression_contract": {
            "level": "aggregate",
            "row_level_tradeoffs_reported": True,
        },
        "all_aggregate_metrics_under_20_pct": all(
            value < 20.0 for value in candidate_aggregate.values()
        ),
        "rows": rows,
    }
    if ncu_values_passed:
        raise NonRegressionError("an NCU target value appeared in a simulator command")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--simulator", type=Path, default=DEFAULT_SIMULATOR)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
        (args.output.resolve() / "nonregression_report.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output.resolve() / "nonregression_report.md").write_text(
            _render(result), encoding="utf-8"
        )
        print(json.dumps({"status": result["status"], "candidate": result["candidate_aggregate"]}, sort_keys=True))
        return 0 if result["strict_all_aggregate_metrics_numerically_non_regressed"] else 1
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, NonRegressionError, subprocess.TimeoutExpired) as error:
        print(f"previous demo non-regression: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
