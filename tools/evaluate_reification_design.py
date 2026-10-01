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
    """Compute coverage and compression metrics from a fast-frontier summary."""
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
    return {
        **coverage,
        "summarized_fraction": (
            coverage["summarized_records"] / coverage["trace_records"]
            if coverage["trace_records"] else 0.0
        ),
        "compression_ratio": (
            coverage["trace_records"] / coverage["summarized_records"]
            if coverage["summarized_records"] else None
        ),
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
                "runtime_ms": document.get("__evaluation_runtime_ms"),
                "memory_bytes": document.get("__evaluation_memory_bytes"),
                **projection,
            }
        )
    native_invariant = all(row["native_terminal_modeled_event"] == detailed_terminal for row in rows)
    finite = [row for row in rows if isinstance(row["simulator_lower"], int)]
    ranking = [row["variant"] for row in sorted(finite, key=lambda row: (row["simulator_lower"], row["variant"]))]
    selective_coverage = None
    if fast_frontier is not None:
        rows_fast = fast_frontier.get("fast_frontier")
        if not isinstance(rows_fast, list) or not rows_fast or not isinstance(rows_fast[0], dict):
            raise ValueError("fast-frontier report has no launch summary")
        selective_coverage = summarize_selective_coverage(rows_fast[0].get("summary", {}))
    return {
        "schema": "sm86_reification_design_evaluation_v1",
        "classification": "model_local_service_intervention",
        "native_replay_terminal_invariant": native_invariant,
        "detailed_native_terminal_modeled_event": detailed_terminal,
        "ranking_basis": "simulator_elapsed_interval_lower_bound",
        "ranking": ranking,
        "ranking_stability": {
            "native_terminal_invariant": native_invariant,
            "model_lower_bound_monotone": all(
                left["simulator_lower"] <= right["simulator_lower"]
                for left, right in zip(
                    sorted(finite, key=lambda row: row["variant"]),
                    sorted(finite, key=lambda row: row["variant"])[1:],
                )
            ),
        },
        "variants": rows,
        "selective_coverage": selective_coverage,
        "hardware_comparison": "not_run_ncu_remains_independent_target",
        "runtime_measurement": "wall_clock_child_process",
        "memory_measurement": "child_peak_rss_usr_bin_time_or_proc_status",
        "architectural_conclusion": (
            "model_local_ranking_only_native_replay_unchanged"
            if native_invariant
            else "native_replay_changed_reify_or_investigate"
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


def _scaled_profile(base: dict[str, Any], scale: int) -> dict[str, Any]:
    if scale <= 0:
        raise ValueError("service scale must be positive")
    profile = json.loads(json.dumps(base))
    for rate in profile.get("pools", {}).values():
        rate["initiation_interval_cycles"] *= scale
        rate["latency_lower_cycles"] *= scale
        rate["latency_upper_cycles"] *= scale
    profile["source"] = f"{profile.get('source', 'profile')}:scale_{scale}"
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
            profile_path.write_text(json.dumps(_scaled_profile(base, scale)))
            variants.append(
                (
                    f"service_scale_{scale}",
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
