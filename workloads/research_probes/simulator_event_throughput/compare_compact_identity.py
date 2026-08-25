#!/usr/bin/env python3
"""Matched A/B timing of 32-byte versus 20-byte validated instructions."""

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
        "--output-root",
        type=Path,
        default=benchmark.RESULT_ROOT / "compact_identity_ab",
    )
    result.add_argument("--baseline-binary", type=Path, required=True)
    result.add_argument("--candidate-binary", type=Path, required=True)
    result.add_argument("--cpu", type=int, default=8)
    result.add_argument("--repeats", type=int, default=7)
    result.add_argument("--timeout-seconds", type=float, default=120.0)
    result.add_argument("--cases")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    baseline_binary = args.baseline_binary.resolve()
    candidate_binary = args.candidate_binary.resolve()
    cases = benchmark.selected_cases(args.cases)
    canonical = {}
    baseline_versions = set()
    candidate_versions = set()
    for case in cases:
        baseline_run, baseline = benchmark.run_once(
            case, args.cpu, args.timeout_seconds, binary=baseline_binary
        )
        candidate_run, candidate = benchmark.run_once(
            case, args.cpu, args.timeout_seconds, binary=candidate_binary
        )
        if baseline != candidate:
            raise RuntimeError(f"{case.key}: compact identity changes normal JSON")
        baseline_versions.add(baseline_run["simulator_version"])
        candidate_versions.add(candidate_run["simulator_version"])
        canonical[case.key] = baseline
    if len(baseline_versions) != 1 or baseline_versions != candidate_versions:
        raise RuntimeError("simulator versions differ across matched binaries")
    version = next(iter(baseline_versions))
    baseline_identity = benchmark.capture_binary_identity(baseline_binary, version)
    candidate_identity = benchmark.capture_binary_identity(candidate_binary, version)

    runs = {case.key: {"baseline32": [], "compact20": []} for case in cases}
    orders = []
    rng = random.Random(13_086)
    for repeat in range(1, args.repeats + 1):
        order = list(cases)
        rng.shuffle(order)
        round_order = []
        for position, case in enumerate(order):
            conditions = ("baseline32", "compact20")
            if (repeat + position) % 2:
                conditions = tuple(reversed(conditions))
            for condition in conditions:
                binary = (
                    baseline_binary if condition == "baseline32" else candidate_binary
                )
                run, output = benchmark.run_once(
                    case, args.cpu, args.timeout_seconds, binary=binary
                )
                if output != canonical[case.key]:
                    raise RuntimeError(
                        f"{case.key}/{condition}: output differs from canonical bytes"
                    )
                run["repeat"] = repeat
                runs[case.key][condition].append(run)
                round_order.append(f"{case.key}:{condition}")
                print(
                    f"repeat {repeat}/{args.repeats} {case.key}/{condition}: "
                    f"{run['elapsed_seconds']:.3f}s, "
                    f"{run['peak_rss_kib'] / 1024:.1f} MiB",
                    flush=True,
                )
        orders.append(round_order)

    summaries = {}
    for case in cases:
        baseline = runs[case.key]["baseline32"]
        compact = runs[case.key]["compact20"]
        baseline_seconds = median([run["elapsed_seconds"] for run in baseline])
        compact_seconds = median([run["elapsed_seconds"] for run in compact])
        baseline_rss = median([run["peak_rss_kib"] for run in baseline])
        compact_rss = median([run["peak_rss_kib"] for run in compact])
        elapsed_changes = benchmark.paired_percent_changes(
            baseline, compact, "elapsed_seconds"
        )
        rss_changes = benchmark.paired_percent_changes(
            baseline, compact, "peak_rss_kib"
        )
        summaries[case.key] = {
            "baseline_elapsed_seconds_median": baseline_seconds,
            "compact_elapsed_seconds_median": compact_seconds,
            "speedup": baseline_seconds / compact_seconds,
            "elapsed_change_percent": median(elapsed_changes),
            "ratio_of_medians_elapsed_change_percent": 100
            * (compact_seconds / baseline_seconds - 1),
            "baseline_peak_rss_kib_median": baseline_rss,
            "compact_peak_rss_kib_median": compact_rss,
            "peak_rss_change_percent": median(rss_changes),
            "ratio_of_medians_peak_rss_change_percent": 100
            * (compact_rss / baseline_rss - 1),
            "paired_elapsed_change_percent_range": [
                min(elapsed_changes),
                max(elapsed_changes),
            ],
            "paired_peak_rss_change_percent_range": [
                min(rss_changes),
                max(rss_changes),
            ],
            "baseline_elapsed_range": [
                min(run["elapsed_seconds"] for run in baseline),
                max(run["elapsed_seconds"] for run in baseline),
            ],
            "compact_elapsed_range": [
                min(run["elapsed_seconds"] for run in compact),
                max(run["elapsed_seconds"] for run in compact),
            ],
        }

    benchmark.verify_binary_identity(baseline_identity)
    benchmark.verify_binary_identity(candidate_identity)
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
            "binary_mode": "matched_ordinary_binaries",
            "baseline_binary": baseline_identity,
            "candidate_binary": candidate_identity,
            "cpu": args.cpu,
            "warmups_per_case_and_condition": 1,
            "measured_repeats": args.repeats,
            "order_seed": 13_086,
            "condition_order_alternates": True,
            "round_orders": orders,
            "cross_condition_byte_equality_required": True,
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

