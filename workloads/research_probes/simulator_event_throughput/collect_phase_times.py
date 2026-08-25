#!/usr/bin/env python3
"""Collect feature-gated decode/replay/serialization phase timing."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
from pathlib import Path
from typing import Sequence

import benchmark


LINE = re.compile(r"^SIM_PHASE_PERF (?P<fields>.+)$")
RUN_FIELDS = (
    "setup_ns",
    "trace_ns",
    "exec_ns",
    "control_ns",
    "schedule_ns",
    "replay_ns",
    "run_total_ns",
)
PHASE_FIELDS = {*RUN_FIELDS, "serialization_ns"}


def parse_line(line: str) -> dict[str, int] | None:
    matched = LINE.fullmatch(line)
    if matched is None:
        return None
    result = {}
    for item in matched.group("fields").split():
        if "=" not in item:
            raise RuntimeError(f"malformed phase-performance field {item!r}")
        key, value = item.split("=", 1)
        if key not in PHASE_FIELDS:
            raise RuntimeError(f"unknown phase-performance field {key!r}")
        if key in result:
            raise RuntimeError(f"duplicate phase-performance field {key!r}")
        parsed = int(value)
        if parsed < 0:
            raise RuntimeError(f"negative phase-performance field {key!r}")
        result[key] = parsed
    return result


def median(values: Sequence[int]) -> float:
    return float(statistics.median(values))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=benchmark.RESULT_ROOT)
    result.add_argument("--binary", type=Path, default=benchmark.BINARY)
    result.add_argument("--cpu", type=int, default=8)
    result.add_argument("--repeats", type=int, default=7)
    result.add_argument("--cases")
    result.add_argument("--disable-batching", action="store_true")
    return result


def summarize_runs(runs: Sequence[dict[str, int]]) -> dict[str, float]:
    if not runs:
        raise RuntimeError("cannot summarize an empty phase run set")
    fields = (*RUN_FIELDS, "serialization_ns")
    summary = {
        f"{field}_median": median([run[field] for run in runs]) for field in fields
    }
    accounted_fractions = []
    trace_fractions = []
    replay_fractions = []
    for run in runs:
        total = run["run_total_ns"]
        if total <= 0:
            raise RuntimeError("phase run_total_ns must be positive")
        accounted = sum(
            run[field]
            for field in (
                "setup_ns",
                "trace_ns",
                "exec_ns",
                "control_ns",
                "schedule_ns",
                "replay_ns",
            )
        )
        if accounted > total:
            raise RuntimeError("phase components exceed their matched run total")
        accounted_fractions.append(accounted / total)
        trace_fractions.append(run["trace_ns"] / total)
        replay_fractions.append(run["replay_ns"] / total)
    summary["accounted_run_fraction"] = float(
        statistics.median(accounted_fractions)
    )
    summary["trace_fraction"] = float(statistics.median(trace_fractions))
    summary["replay_fraction"] = float(statistics.median(replay_fractions))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    binary = args.binary.resolve()
    binary_identity = benchmark.require_performance_feature(
        binary, args.cpu, 120.0
    )
    cases = benchmark.selected_cases(args.cases)
    runs: dict[str, list[dict[str, int]]] = {case.key: [] for case in cases}
    orders = []
    rng = random.Random(13_086)
    for repeat in range(1, args.repeats + 1):
        order = list(cases)
        rng.shuffle(order)
        orders.append([case.key for case in order])
        for case in order:
            completed = subprocess.run(
                benchmark.command(case, args.cpu, binary),
                cwd=benchmark.REPO_ROOT,
                env=benchmark.simulator_environment(
                    performance_counters=True,
                    disable_batching=args.disable_batching,
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{case.key}: instrumented replay failed: {completed.stderr[-1000:]}"
                )
            phase_lines = [
                parsed
                for line in completed.stderr.splitlines()
                if (parsed := parse_line(line)) is not None
            ]
            run_lines = [line for line in phase_lines if set(line) == set(RUN_FIELDS)]
            serialization_lines = [
                line for line in phase_lines if set(line) == {"serialization_ns"}
            ]
            if len(run_lines) != 1 or len(serialization_lines) != 1:
                raise RuntimeError(
                    f"{case.key}: expected one run and one serialization phase line, "
                    f"got {len(run_lines)} and {len(serialization_lines)}"
                )
            run = run_lines[0]
            serialization = serialization_lines[0]
            material = {**run, **serialization, "repeat": repeat}
            runs[case.key].append(material)
            print(
                f"repeat {repeat}/{args.repeats} {case.key}: "
                f"trace {run['trace_ns'] / 1e9:.3f}s, "
                f"replay {run['replay_ns'] / 1e9:.3f}s",
                flush=True,
            )

    cases_document = {}
    for case in cases:
        summary = summarize_runs(runs[case.key])
        cases_document[case.key] = {"summary": summary, "runs": runs[case.key]}

    benchmark.verify_binary_identity(binary_identity)
    document = {
        "schema_version": 1,
        "method": {
            "counter_build_feature": "sim-run/performance-counters",
            "measured_repeats": args.repeats,
            "order_seed": 13_086,
            "round_orders": orders,
            "stdout_discarded": True,
            "binary": binary_identity,
            "same_cycle_batching_disabled": args.disable_batching,
        },
        "cases": cases_document,
    }
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase_times.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "phase_times.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
