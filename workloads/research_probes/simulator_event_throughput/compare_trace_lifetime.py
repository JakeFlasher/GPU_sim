#!/usr/bin/env python3
"""Matched A/B timing of raw-retaining versus compact direct trace loading."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Sequence

import benchmark


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-root", type=Path, default=benchmark.RESULT_ROOT / "trace_lifetime_ab"
    )
    result.add_argument("--binary", type=Path, default=benchmark.BINARY)
    result.add_argument("--cpu", type=int, default=8)
    result.add_argument("--repeats", type=int, default=7)
    result.add_argument("--timeout-seconds", type=float, default=120.0)
    result.add_argument("--cases")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    binary = args.binary.resolve()
    binary_identity = benchmark.require_performance_feature(
        binary, args.cpu, args.timeout_seconds
    )
    cases = benchmark.selected_cases(args.cases)
    canonical = {}
    for case in cases:
        _, raw = benchmark.run_once(
            case,
            args.cpu,
            args.timeout_seconds,
            retain_raw_trace_payload=True,
            binary=binary,
        )
        _, compact = benchmark.run_once(
            case, args.cpu, args.timeout_seconds, binary=binary
        )
        if raw != compact:
            raise RuntimeError(f"{case.key}: compact loading changes normal JSON")
        canonical[case.key] = raw

    runs = {case.key: {"raw": [], "compact": []} for case in cases}
    orders = []
    rng = random.Random(13_086)
    for repeat in range(1, args.repeats + 1):
        order = list(cases)
        rng.shuffle(order)
        round_order = []
        for position, case in enumerate(order):
            conditions = ("raw", "compact")
            if (repeat + position) % 2:
                conditions = tuple(reversed(conditions))
            for condition in conditions:
                run, output = benchmark.run_once(
                    case,
                    args.cpu,
                    args.timeout_seconds,
                    retain_raw_trace_payload=condition == "raw",
                    binary=binary,
                )
                if output != canonical[case.key]:
                    raise RuntimeError(
                        f"{case.key}/{condition}: output differs from matched canonical bytes"
                    )
                run["repeat"] = repeat
                runs[case.key][condition].append(run)
                round_order.append(f"{case.key}:{condition}")
                print(
                    f"repeat {repeat}/{args.repeats} {case.key}/{condition}: "
                    f"{run['elapsed_seconds']:.3f}s, {run['peak_rss_kib'] / 1024:.1f} MiB",
                    flush=True,
                )
        orders.append(round_order)

    summaries = {}
    for case in cases:
        raw = runs[case.key]["raw"]
        compact = runs[case.key]["compact"]
        raw_seconds = median([run["elapsed_seconds"] for run in raw])
        compact_seconds = median([run["elapsed_seconds"] for run in compact])
        raw_rss = median([run["peak_rss_kib"] for run in raw])
        compact_rss = median([run["peak_rss_kib"] for run in compact])
        elapsed_changes = benchmark.paired_percent_changes(
            raw, compact, "elapsed_seconds"
        )
        rss_changes = benchmark.paired_percent_changes(raw, compact, "peak_rss_kib")
        summaries[case.key] = {
            "raw_elapsed_seconds_median": raw_seconds,
            "compact_elapsed_seconds_median": compact_seconds,
            "speedup": raw_seconds / compact_seconds,
            "elapsed_change_percent": median(elapsed_changes),
            "ratio_of_medians_elapsed_change_percent": 100
            * (compact_seconds / raw_seconds - 1),
            "raw_peak_rss_kib_median": raw_rss,
            "compact_peak_rss_kib_median": compact_rss,
            "peak_rss_change_percent": median(rss_changes),
            "ratio_of_medians_peak_rss_change_percent": 100
            * (compact_rss / raw_rss - 1),
            "paired_elapsed_change_percent_range": [
                min(elapsed_changes),
                max(elapsed_changes),
            ],
            "paired_peak_rss_change_percent_range": [
                min(rss_changes),
                max(rss_changes),
            ],
            "raw_elapsed_range": [
                min(run["elapsed_seconds"] for run in raw),
                max(run["elapsed_seconds"] for run in raw),
            ],
            "compact_elapsed_range": [
                min(run["elapsed_seconds"] for run in compact),
                max(run["elapsed_seconds"] for run in compact),
            ],
        }
    benchmark.verify_binary_identity(binary_identity)
    output = args.output_root.resolve()
    canonical_root = output / "canonical"
    canonical_root.mkdir(parents=True, exist_ok=True)
    case_metadata = {}
    for case in cases:
        canonical_path = canonical_root / f"{case.key}.json"
        canonical_path.write_bytes(canonical[case.key])
        case_metadata[case.key] = {
            "trace": str(case.trace.relative_to(benchmark.REPO_ROOT)),
            "control": str(case.control.relative_to(benchmark.REPO_ROOT)),
            "extra_arguments": list(case.extra),
            "trace_bytes": benchmark.tree_bytes(case.trace),
            "canonical_output": str(canonical_path.relative_to(output)),
            "canonical_result_bytes": len(canonical[case.key]),
            "expected_readiness_schema": case.expected_readiness_schema,
            "expected_feedback_candidate_id": case.expected_feedback_candidate_id,
        }
    document = {
        "schema_version": 1,
        "method": {
            "feature_binary": "sim-run/performance-counters",
            "performance_logging_enabled": False,
            "cpu": args.cpu,
            "warmups_per_case_and_condition": 1,
            "measured_repeats": args.repeats,
            "order_seed": 13_086,
            "condition_order_alternates": True,
            "round_orders": orders,
            "cross_condition_byte_equality_required": True,
            "binary": binary_identity,
        },
        "summaries": summaries,
        "cases": case_metadata,
        "runs": runs,
    }
    (output / "comparison.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
