#!/usr/bin/env python3
"""Reduce typed synchronization and readiness attribution from replay rows.

This report consumes model/trace diagnostics already emitted by replay. It is
not a timing calibration, does not use NCU, and never turns an async frontier
into an engine-completion timestamp.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from jsonl_io import jsonl_compression_description, open_jsonl_reader, open_jsonl_writer


def add_counter(dst: collections.Counter, values: Any) -> None:
    if isinstance(values, dict):
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                dst[str(key)] += value


def add_nested(dst: dict[str, collections.Counter], values: Any) -> None:
    if isinstance(values, dict):
        for key, nested in values.items():
            if isinstance(nested, dict):
                add_counter(dst.setdefault(str(key), collections.Counter()), nested)


def launch_record(capture: str, mode: str, launch: dict[str, Any]) -> dict[str, Any]:
    trace = launch.get("trace_metrics", {})
    obligation = trace.get("async_obligations") or {}
    timing = launch.get("timing") or {}
    readiness = timing.get("readiness") or {}
    record = {
        "capture": capture,
        "mode": mode,
        "kernel_id": launch.get("kernel_id"),
        "async": {
            key: obligation.get(key)
            for key in (
                "tma_load_obligations",
                "tma_store_obligations",
                "expected_arrival_obligations",
                "expected_transaction_bytes",
                "captured_query_instructions",
                "captured_successful_queries",
                "captured_unsuccessful_queries",
                "qualified_phase_observations",
                "protocol_scoped_phase_candidates",
                "qualified_satisfied_generations",
                "possible_unresolved_generations",
                "pre_result_frontiers",
                "post_result_narrowed_frontiers",
                "current_query_only_frontiers",
                "deferred_arrival_obligations",
                "frontier_precision_lost",
                "interpretation_conflict_frontiers",
            )
            if key in obligation
        },
        "unresolved_obligations": obligation.get("unresolved_obligations", []),
        "engine_completion_clock": obligation.get("engine_completion_clock", "unavailable"),
        "limiting_cycles": readiness.get("limiting_cycles", {}),
        "constraint_exposure_cycles": readiness.get("constraint_exposure_cycles", {}),
        "delayed_ready_issues": readiness.get("delayed_ready_issues", 0),
        "async_groups": readiness.get("async_groups", {}),
        "scalar_execution_pipeline": readiness.get("scalar_execution_pipeline"),
        "cta_barriers": readiness.get("cta_barriers", {}),
        "barrier_pipeline": readiness.get("barrier_pipeline", {}),
    }
    return record


def reduce(path: Path, mode: str, output, summary: dict[str, Any]) -> None:
    key = "launches" if mode == "full" else "fast_frontier"
    with open_jsonl_reader(path) as stream:
        for line_number, line in enumerate(stream, 1):
            capture = json.loads(line)
            name = str(capture.get("canonical_capture", ""))
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"{path}:{line_number}: invalid capture identity")
            launches = capture.get(key, [])
            if not isinstance(launches, list):
                raise ValueError(f"{path}:{line_number}: {key} must be an array")
            summary[mode]["captures"] += 1
            summary[mode]["launches"] += len(launches)
            for launch in launches:
                record = launch_record(name, mode, launch)
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                summary[mode]["async_launches"] += bool(record["async"])
                add_counter(summary["async_totals"], record["async"])
                add_counter(summary["limiting_cycles"], record["limiting_cycles"])
                add_counter(summary["constraint_exposure_cycles"], record["constraint_exposure_cycles"])
                add_counter(summary["async_groups"], record["async_groups"])
                add_counter(summary["cta_barriers"], record["cta_barriers"])
                add_counter(summary["barrier_pipeline"], record["barrier_pipeline"])
                summary["delayed_ready_issues"] += int(record["delayed_ready_issues"] or 0)
                for reason in record["unresolved_obligations"]:
                    summary["unresolved_obligations"][str(reason)] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scope": "post-replay synchronization attribution; NCU not used",
        "jsonl_compression": jsonl_compression_description(),
        "full": {"captures": 0, "launches": 0, "async_launches": 0},
        "fast": {"captures": 0, "launches": 0, "async_launches": 0},
        "async_totals": collections.Counter(),
        "limiting_cycles": collections.Counter(),
        "constraint_exposure_cycles": collections.Counter(),
        "async_groups": collections.Counter(),
        "cta_barriers": collections.Counter(),
        "barrier_pipeline": collections.Counter(),
        "unresolved_obligations": collections.Counter(),
        "delayed_ready_issues": 0,
    }
    with open_jsonl_writer(args.output / "sync_attribution.jsonl") as output:
        reduce(args.full, "full", output, summary)
        reduce(args.fast, "fast", output, summary)
    for key, value in list(summary.items()):
        if isinstance(value, collections.Counter):
            summary[key] = dict(value)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary.md").open("w") as report:
        report.write("# SM120 synchronization attribution\n\n")
        report.write("Post-replay trace/model attribution; no NCU value is a model input.\n\n")
        report.write("| mode | captures | launches | launches with async evidence |\n|---|---:|---:|---:|\n")
        for mode in ("full", "fast"):
            row = summary[mode]
            report.write(f"| {mode} | {row['captures']} | {row['launches']} | {row['async_launches']} |\n")
        report.write("\n## Limiting readiness cycles\n\n| cause | cycles |\n|---|---:|\n")
        for key, value in sorted(summary["limiting_cycles"].items()):
            report.write(f"| `{key}` | {value} |\n")
        report.write("\n## Async obligation totals\n\n| field | total |\n|---|---:|\n")
        for key, value in sorted(summary["async_totals"].items()):
            report.write(f"| `{key}` | {value} |\n")
        report.write("\n## Unresolved obligation reasons\n\n")
        for key, value in sorted(summary["unresolved_obligations"].items()):
            report.write(f"- `{key}`: {value}\n")
        report.write("\nEngine completion remains unavailable; these rows are attribution/model evidence, not completion timestamps.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
