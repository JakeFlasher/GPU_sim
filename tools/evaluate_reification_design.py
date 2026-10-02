#!/usr/bin/env python3
"""Evaluate a trace-safe model intervention without consuming NCU targets.

The harness compares one detailed replay with opt-in latent service-profile
variants. Native replay fields are checked for invariance; only the declared
model-local service frontier is ranked. NCU/CUPTI/host measurements are
rejected as inputs and are not used for selection.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


def _first_launch(document: dict[str, Any]) -> dict[str, Any]:
    launches = document.get("launches")
    if not isinstance(launches, list) or not launches or not isinstance(launches[0], dict):
        raise ValueError("simulator output has no launch report")
    return launches[0]


def _native_terminal(document: dict[str, Any]) -> int | None:
    timing = _first_launch(document).get("timing")
    if not isinstance(timing, dict):
        raise ValueError("simulator output has no timing report")
    value = timing.get("terminal_modeled_event_time")
    return value if isinstance(value, int) else None


def _native_projection(document: dict[str, Any]) -> dict[str, Any]:
    """Return the declared timing projection used for invariance checks.

    A terminal timestamp alone can stay equal while issue/readiness or the
    modeled system-event frontier changes.  Fields absent from an older output
    remain ``None`` rather than being silently treated as equal to an
    unobserved value.
    """
    timing = _first_launch(document).get("timing")
    if not isinstance(timing, dict):
        raise ValueError("simulator output has no timing report")
    keys = (
        "diagnostic_last_issue_cycles",
        "last_processed_modeled_system_event_time",
        "terminal_modeled_event_time",
        "completion_frontier",
        "readiness",
    )
    return {key: timing.get(key) for key in keys}


def _native_projection_fields_present(document: dict[str, Any]) -> bool:
    timing = _first_launch(document).get("timing")
    return isinstance(timing, dict) and all(
        key in timing
        for key in (
            "diagnostic_last_issue_cycles",
            "last_processed_modeled_system_event_time",
            "terminal_modeled_event_time",
            "completion_frontier",
            "readiness",
        )
    )


def _interval_relations(rows: list[dict[str, Any]]) -> dict[str, list[list[str]]]:
    ordered_pairs: list[list[str]] = []
    indeterminate_pairs: list[list[str]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            left_upper = left.get("simulator_upper")
            right_upper = right.get("simulator_upper")
            if isinstance(left_upper, int) and left_upper < right["simulator_lower"]:
                ordered_pairs.append([left["variant"], right["variant"]])
            elif isinstance(right_upper, int) and right_upper < left["simulator_lower"]:
                ordered_pairs.append([right["variant"], left["variant"]])
            else:
                indeterminate_pairs.append([left["variant"], right["variant"]])
    return {"ordered_pairs": ordered_pairs, "indeterminate_pairs": indeterminate_pairs}


def _latent_projection(document: dict[str, Any]) -> dict[str, Any]:
    envelope = document.get("latent_envelope")
    if not isinstance(envelope, dict):
        raise ValueError("variant output has no latent envelope")
    launches = envelope.get("launches")
    if not isinstance(launches, list) or not launches or not isinstance(launches[0], dict):
        raise ValueError("variant latent envelope has no launch")
    launch = launches[0]
    frontiers = launch.get("frontiers")
    projection = launch.get("ncu_projection")
    service = launch.get("resource_service_projection")
    if not isinstance(frontiers, dict) or not isinstance(projection, dict):
        raise ValueError("variant latent envelope is missing frontiers/projection")
    return {
        "service_lower": service.get("lower_cycles") if isinstance(service, dict) else None,
        "service_upper": service.get("upper_cycles") if isinstance(service, dict) else None,
        "resource_frontier": frontiers.get("resource_service"),
        "simulator_lower": projection.get("simulator_lower_cycles"),
        "simulator_upper": projection.get("simulator_upper_cycles"),
        "predicted_lower": projection.get("predicted_lower_cycles"),
        "predicted_upper": projection.get("predicted_upper_cycles"),
        "role": projection.get("role"),
    }


def summarize_selective_coverage(summary: dict[str, Any]) -> dict[str, Any]:
    """Compute record coverage and execution-step reduction separately."""
    coverage = summary.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("fast-frontier summary has no coverage object")
    required = ("trace_records", "candidate_records", "summarized_records", "reified_records", "uncovered_records")
    if any(not isinstance(coverage.get(key), int) or coverage[key] < 0 for key in required):
        raise ValueError("fast-frontier coverage counts are invalid")
    if coverage["trace_records"] != coverage["candidate_records"] + coverage["uncovered_records"]:
        raise ValueError("fast-frontier coverage denominator does not balance")
    if coverage["candidate_records"] != coverage["summarized_records"] + coverage["reified_records"]:
        raise ValueError("fast-frontier candidate coverage does not balance")
    exact_steps = summary.get("exact_execution_steps")
    summarized_steps = summary.get("summarized_execution_steps")
    if not isinstance(exact_steps, int) or exact_steps < 0:
        exact_steps = None
    if not isinstance(summarized_steps, int) or summarized_steps < 0:
        summarized_steps = None
    macro_dispatches = summary.get("macro_dispatches")
    if not isinstance(macro_dispatches, int) or macro_dispatches < 0:
        macro_dispatches = None
    step_reduction = (
        exact_steps / summarized_steps
        if exact_steps is not None and summarized_steps not in (None, 0)
        else None
    )
    return {
        **coverage,
        "summarized_fraction": (
            coverage["summarized_records"] / coverage["trace_records"]
            if coverage["trace_records"] else 0.0
        ),
        # This is a record statistic, not an execution compression claim.
        "record_to_summary_ratio": (
            coverage["trace_records"] / coverage["summarized_records"]
            if coverage["summarized_records"] else None
        ),
        "exact_execution_steps": exact_steps,
        "summarized_execution_steps": summarized_steps,
        "step_reduction": step_reduction,
        "macro_dispatches": macro_dispatches,
        "macro_execution_observed": macro_dispatches is not None and macro_dispatches > 0,
        "accepted_records": coverage["summarized_records"],
        "refused_records": coverage["reified_records"] + coverage["uncovered_records"],
        "accepted_fraction": (
            coverage["summarized_records"] / coverage["trace_records"]
            if coverage["trace_records"] else 0.0
        ),
        "refusal_fraction": (
            (coverage["reified_records"] + coverage["uncovered_records"])
            / coverage["trace_records"]
            if coverage["trace_records"] else 0.0
        ),
        "cost_basis": "dispatch_counts_are_not_runtime_speedup",
    }


def summarize_design_runs(
    detailed: dict[str, Any],
    variants: Sequence[tuple[str, dict[str, Any]]],
    fast_frontier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic model-local design report from JSON outputs."""
    detailed_terminal = _native_terminal(detailed)
    rows = []
    for name, document in variants:
        projection = _latent_projection(document)
        rows.append(
            {
                "variant": name,
                "native_terminal_modeled_event": _native_terminal(document),
                "native_projection": _native_projection(document),
                "runtime_ms": document.get("__evaluation_runtime_ms"),
                "memory_bytes": document.get("__evaluation_memory_bytes"),
                **projection,
            }
        )
    detailed_projection = _native_projection(detailed)
    native_projection_complete = bool(rows) and _native_projection_fields_present(detailed) and all(
        _native_projection_fields_present(document)
        for _, document in variants
    )
    native_invariant = native_projection_complete and all(
        row["native_projection"] == detailed_projection for row in rows
    )
    finite = [row for row in rows if isinstance(row["simulator_lower"], int)]
    relations = _interval_relations(finite)
    selective_coverage = None
    if fast_frontier is not None:
        rows_fast = fast_frontier.get("fast_frontier")
        if not isinstance(rows_fast, list) or not rows_fast or not isinstance(rows_fast[0], dict):
            raise ValueError("fast-frontier report has no launch summary")
        selective_coverage = summarize_selective_coverage(rows_fast[0].get("summary", {}))
    detailed_runtime = detailed.get("__evaluation_runtime_ms")
    detailed_memory = detailed.get("__evaluation_memory_bytes")
    for row in rows:
        runtime = row.get("runtime_ms")
        memory = row.get("memory_bytes")
        row["runtime_delta_ms_vs_detailed"] = (
            runtime - detailed_runtime
            if isinstance(runtime, (int, float)) and isinstance(detailed_runtime, (int, float))
            else None
        )
        row["memory_delta_bytes_vs_detailed"] = (
            memory - detailed_memory
            if isinstance(memory, (int, float)) and isinstance(detailed_memory, (int, float))
            else None
        )
    return {
        "schema": "sm86_reification_design_evaluation_v2",
        "classification": "model_local_service_intervention",
        "native_replay_terminal_invariant": native_invariant,
        "detailed_native_terminal_modeled_event": detailed_terminal,
        "native_projection_complete": native_projection_complete,
        "ranking_basis": "simulator_elapsed_interval_partial_order",
        "ranking": relations["ordered_pairs"],
        "ranking_relations": relations,
        "ranking_stability": {
            "native_terminal_invariant": native_invariant,
            "declared_projection_invariant": native_invariant,
            "model_lower_bound_ordered_pairs": relations["ordered_pairs"],
            "indeterminate_pairs": relations["indeterminate_pairs"],
        },
        "variants": rows,
        "detailed_cost": {
            "runtime_ms": detailed_runtime,
            "memory_bytes": detailed_memory,
            "scope": "child_wall_clock_and_peak_rss_include_parse_replay_planning_effect_compare_output",
        },
        "cost_components": [
            "trace_parse",
            "detailed_replay",
            "candidate_planning",
            "effect_materialization",
            "boundary_compare",
            "report_output",
        ],
        "native_speedup_claimed": False,
        "selective_coverage": selective_coverage,
        "hardware_comparison": "not_run_ncu_remains_independent_target",
        "runtime_measurement": "wall_clock_child_process",
        "memory_measurement": "child_peak_rss_usr_bin_time_or_proc_status",
        "architectural_conclusion": (
            "native_projection_unknown_exact_replay_required"
            if not native_projection_complete
            else (
                "model_local_partial_order_only_declared_native_projection_unchanged"
                if native_invariant
                else "native_replay_changed_reify_or_investigate"
            )
        ),
    }


def _run(command: Sequence[str], timeout: float) -> dict[str, Any]:
    if any(argument.startswith("--ncu") or argument.startswith("--cupti") for argument in command):
        raise ValueError("NCU/CUPTI inputs are forbidden in this intervention evaluation")
    started = time.perf_counter()
    time_binary = shutil.which("/usr/bin/time")
    wrapped = [time_binary, "-f", "__EVAL_MAXRSS_KB__=%M", *command] if time_binary else list(command)
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(wrapped, stdout=stdout_file, stderr=stderr_file)
        peak_rss_kb = 0
        while process.poll() is None:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                values = [
                    int(match.group(1))
                    for match in re.finditer(r"Vm(?:HWM|RSS):\s+(\d+)\s+kB", status)
                ]
                peak_rss_kb = max([peak_rss_kb, *values]) if values else peak_rss_kb
            except (FileNotFoundError, PermissionError):
                pass
            if time.perf_counter() - started > timeout:
                process.kill()
                process.wait()
                raise TimeoutError(f"simulator timed out after {timeout}s")
            time.sleep(0.005)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode()
        stderr = stderr_file.read().decode()
    if process.returncode != 0:
        raise RuntimeError(f"simulator failed with {process.returncode}: {stderr[-2000:]}")
    document = json.loads(stdout)
    if not isinstance(document, dict):
        raise ValueError("simulator output must be a JSON object")
    # Keep measurement metadata outside the simulator schema. `/usr/bin/time`
    # supplies child peak RSS when available; Python/process RSS is never used
    # as a proxy.
    document["__evaluation_runtime_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    memory_match = re.search(r"__EVAL_MAXRSS_KB__=(\d+)", stderr)
    document["__evaluation_memory_bytes"] = (
        int(memory_match.group(1)) * 1024 if memory_match else (peak_rss_kb * 1024 or None)
    )
    return document


def _scaled_profile(base: dict[str, Any], scale: int, mode: str = "throughput") -> dict[str, Any]:
    if scale <= 0:
        raise ValueError("service scale must be positive")
    if mode not in {"throughput", "latency", "both"}:
        raise ValueError("service scale mode must be throughput, latency, or both")
    profile = json.loads(json.dumps(base))
    for rate in profile.get("pools", {}).values():
        if mode in {"throughput", "both"}:
            rate["initiation_interval_cycles"] *= scale
        if mode in {"latency", "both"}:
            rate["latency_lower_cycles"] *= scale
            rate["latency_upper_cycles"] *= scale
    profile["source"] = f"{profile.get('source', 'profile')}:{mode}_scale_{scale}"
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--arch-card", type=Path, required=True)
    parser.add_argument("--pchase-card", type=Path, required=True)
    parser.add_argument("--scale", type=int, action="append", required=True)
    parser.add_argument("--scale-mode", choices=("throughput", "latency", "both"), default="throughput")
    parser.add_argument("--fast-frontier-report", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base = json.loads(args.profile.read_text())
    common = [
        str(args.sim_run),
        "--trace", str(args.trace),
        "--control", str(args.control),
        "--arch-card", str(args.arch_card),
        "--pchase-card", str(args.pchase_card),
        "--mode", "l1-l2",
        "--cache-scope", "launch",
    ]
    detailed = _run(common, args.timeout)
    variants = []
    with tempfile.TemporaryDirectory(prefix="gpusim_design_profiles_") as directory:
        for scale in args.scale:
            profile_path = Path(directory) / f"scale_{scale}.json"
            profile_path.write_text(json.dumps(_scaled_profile(base, scale, args.scale_mode)))
            variants.append(
                (
                    f"service_{args.scale_mode}_scale_{scale}",
                    _run(common + [
                        "--latent-envelope",
                        "--predictive-service-profile", str(profile_path),
                    ], args.timeout),
                )
            )
    fast_frontier = (
        json.loads(args.fast_frontier_report.read_text())
        if args.fast_frontier_report is not None
        else None
    )
    print(json.dumps(summarize_design_runs(detailed, variants, fast_frontier), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
