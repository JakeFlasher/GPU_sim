#!/usr/bin/env python3
"""Collect feature-gated exact event counts and join clean wall-time medians."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence

import benchmark


LINE = re.compile(r"^SIM_EVENT_PERF (?P<fields>.+)$")
COUNTERS = (
    "events_processed",
    "queue_events_processed",
    "sm_wake",
    "l2_access",
    "l2_access_batches",
    "l2_access_batched_requests",
    "l2_fill",
    "atomic_completion",
    "mem_response",
    "same_cycle_after_first",
    "queue_same_cycle_after_first",
    "maximum_queue_depth",
)


def parse_line(line: str) -> dict[str, int] | None:
    matched = LINE.fullmatch(line)
    if matched is None:
        return None
    result = {}
    for item in matched.group("fields").split():
        if "=" not in item:
            raise RuntimeError(f"malformed event-performance field {item!r}")
        key, value = item.split("=", 1)
        if key not in COUNTERS:
            raise RuntimeError(f"unknown event-performance counter {key!r}")
        if key in result:
            raise RuntimeError(f"duplicate event-performance counter {key!r}")
        parsed = int(value)
        if parsed < 0:
            raise RuntimeError(f"negative event-performance counter {key!r}")
        result[key] = parsed
    missing = set(COUNTERS) - set(result)
    if missing:
        raise RuntimeError(f"missing event-performance counters: {sorted(missing)}")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--timing",
        type=Path,
        default=benchmark.RESULT_ROOT / "optimized/baseline.json",
    )
    result.add_argument("--binary", type=Path, default=benchmark.BINARY)
    result.add_argument("--output-root", type=Path, default=benchmark.RESULT_ROOT)
    result.add_argument("--cpu", type=int, default=8)
    result.add_argument("--cases")
    result.add_argument("--disable-batching", action="store_true")
    result.add_argument(
        "--retime-existing",
        action="store_true",
        help="rejoin retained exact counts to clean timing without executing sim-run",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    timing = json.loads(args.timing.read_text(encoding="utf-8"))
    cases = benchmark.selected_cases(args.cases)
    output = args.output_root.resolve()
    if args.retime_existing:
        path = output / "event_counts.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        for case in cases:
            row = document["cases"][case.key]
            elapsed = float(timing["cases"][case.key]["summary"]["elapsed_seconds_median"])
            row["clean_elapsed_seconds_median"] = elapsed
            row["events_per_second"] = int(row["events_processed"]) / elapsed
            row["queue_events_per_second"] = int(row["queue_events_processed"]) / elapsed
        document["method"]["timing_source"] = str(args.timing.resolve())
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(path)
        return 0
    binary = args.binary.resolve()
    binary_identity = benchmark.require_performance_feature(
        binary, args.cpu, 120.0
    )
    rows = []
    for case in cases:
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
        launches = [
            parsed
            for line in completed.stderr.splitlines()
            if (parsed := parse_line(line)) is not None
        ]
        if not launches:
            raise RuntimeError(
                f"{case.key}: no SIM_EVENT_PERF output; build sim-run with "
                "--features performance-counters"
            )
        summary = timing["cases"][case.key]["summary"]
        expected_launches = int(summary["launches"])
        if len(launches) != expected_launches:
            raise RuntimeError(
                f"{case.key}: diagnostic launch count {len(launches)} differs "
                f"from clean timing count {expected_launches}"
            )
        elapsed = float(summary["elapsed_seconds_median"])
        totals = {
            key: (
                max(launch[key] for launch in launches)
                if key == "maximum_queue_depth"
                else sum(launch[key] for launch in launches)
            )
            for key in COUNTERS
        }
        row = {
            "case": case.key,
            "launches": len(launches),
            **totals,
            "clean_elapsed_seconds_median": elapsed,
            "events_per_second": totals["events_processed"] / elapsed,
            "queue_events_per_second": totals["queue_events_processed"] / elapsed,
            "instructions_per_event": (
                int(summary["instructions_issued"]) / totals["events_processed"]
                if totals["events_processed"]
                else None
            ),
        }
        rows.append(row)
        print(
            f"{case.key}: {totals['events_processed']} events, "
            f"{row['events_per_second'] / 1_000_000:.3f} M events/s",
            flush=True,
        )

    benchmark.verify_binary_identity(binary_identity)
    document = {
        "schema_version": 1,
        "method": {
            "counter_build_feature": "sim-run/performance-counters",
            "timing_source": str(args.timing.resolve()),
            "binary": binary_identity,
            "timing_and_counting_runs_are_separate": True,
            "maximum_queue_depth_is_maximum_across_launches": True,
            "other_counters_are_summed_across_launches": True,
            "same_cycle_batching_disabled": args.disable_batching,
        },
        "cases": {row["case"]: row for row in rows},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "event_counts.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "event_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output / "event_counts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
