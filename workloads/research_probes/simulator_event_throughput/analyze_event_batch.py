#!/usr/bin/env python3
"""Reduce the deterministic same-cycle event-batch milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import benchmark


TARGETS = {
    "retirement_store_reuse_d64",
    "arrival_store_w8",
    "admission_store_d128",
    "admission_red_d128",
    "admission_red_gap64",
}
FEEDBACK = {
    "admission_red_d128_schema10",
    "admission_red_d128_schema11",
    "admission_red_d128_schema13",
}


def load(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError(f"{path}: expected object")
    return document


def analyze(root: Path) -> dict[str, object]:
    before = load(root / "summary.json")
    comparison = load(root / "batch_ab/comparison.json")
    byte_identity = benchmark.validate_comparison_evidence(
        root / "batch_ab", comparison, ("legacy", "batched")
    )
    counts = load(root / "batched/event_counts.json")
    before_phases = load(root / "phase_times.json")
    after_phases = load(root / "batched/phase_times.json")
    expected = {case.key for case in benchmark.CASES}
    for label, keys in (
        ("before", before["absolute_optimized"]),
        ("comparison", comparison["summaries"]),
        ("counts", counts["cases"]),
        ("before phases", before_phases["cases"]),
        ("after phases", after_phases["cases"]),
    ):
        if set(keys) != expected:
            raise RuntimeError(f"{label}: case set differs: {sorted(set(keys) ^ expected)}")

    cases = {}
    for case in sorted(expected):
        old = before["absolute_optimized"][case]
        measured = comparison["summaries"][case]
        event = counts["cases"][case]
        if int(old["events_processed"]) != int(event["events_processed"]):
            raise RuntimeError(f"{case}: logical event count changed")
        physical = int(event["queue_events_processed"])
        logical = int(event["events_processed"])
        old_depth = int(old["maximum_queue_depth"])
        new_depth = int(event["maximum_queue_depth"])
        old_replay = float(
            before_phases["cases"][case]["summary"]["replay_ns_median"]
        )
        new_replay = float(
            after_phases["cases"][case]["summary"]["replay_ns_median"]
        )
        cases[case] = {
            **measured,
            "logical_events_processed": logical,
            "physical_queue_events_processed": physical,
            "physical_queue_event_reduction_percent": 100 * (1 - physical / logical),
            "l2_access_batches": int(event["l2_access_batches"]),
            "l2_access_batched_requests": int(event["l2_access_batched_requests"]),
            "legacy_maximum_queue_depth": old_depth,
            "batched_maximum_queue_depth": new_depth,
            "maximum_queue_depth_reduction_percent": (
                100 * (1 - new_depth / old_depth) if old_depth else 0.0
            ),
            "legacy_replay_ns_median": old_replay,
            "batched_replay_ns_median": new_replay,
            "replay_phase_change_percent": 100 * (new_replay / old_replay - 1),
        }

    maximum_regression = max(float(item["elapsed_change_percent"]) for item in cases.values())
    feedback_is_legacy = all(
        cases[case]["l2_access_batches"] == 0
        and cases[case]["physical_queue_events_processed"]
        == cases[case]["logical_events_processed"]
        for case in FEEDBACK
    )
    targets_improve = all(
        float(cases[case]["elapsed_change_percent"]) < 0
        and float(cases[case]["peak_rss_change_percent"]) < 0
        for case in TARGETS
    )
    deep_depth_reduction = min(
        float(cases[case]["maximum_queue_depth_reduction_percent"])
        for case in ("admission_store_d128", "admission_red_d128")
    )
    retain = (
        feedback_is_legacy
        and targets_improve
        and deep_depth_reduction >= 50
        and maximum_regression <= 3
        and byte_identity
    )
    return {
        "schema_version": 1,
        "method": comparison["method"],
        "cases": cases,
        "decision": {
            "retain_same_cycle_replyless_write_batching": retain,
            "feedback_path_keeps_legacy_events": feedback_is_legacy,
            "all_targeted_cases_improve_time_and_memory": targets_improve,
            "minimum_deep_write_queue_depth_reduction_percent": deep_depth_reduction,
            "maximum_cohort_elapsed_regression_percent": maximum_regression,
            "normal_json_byte_identical_across_conditions": byte_identity,
            "scope": (
                "Consecutive same-SM/same-time reply-less STORE/RED requests from one "
                "emission batch; timing-active feedback remains individual-event."
            ),
        },
    }


def markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Deterministic same-cycle L2-access batching",
        "",
        "Logical modeled requests are unchanged. Physical queue events combine only",
        "consecutive same-SM/same-time reply-less STORE/RED requests emitted by one",
        "actor advance. Feedback replay retains its legacy individual-event path.",
        "",
        "| case | logical events | queue events | queue reduction | max queue before→after | time change | RSS change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, item in result["cases"].items():
        lines.append(
            f"| {case} | {item['logical_events_processed']} | "
            f"{item['physical_queue_events_processed']} | "
            f"{item['physical_queue_event_reduction_percent']:.1f}% | "
            f"{item['legacy_maximum_queue_depth']}→{item['batched_maximum_queue_depth']} | "
            f"{item['elapsed_change_percent']:+.1f}% | "
            f"{item['peak_rss_change_percent']:+.1f}% |"
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Retain batching: **{decision['retain_same_cycle_replyless_write_batching']}**.",
            "Deep STORE/RED queue depth falls by at least "
            f"{decision['minimum_deep_write_queue_depth_reduction_percent']:.1f}%. "
            "All targeted cases improve both time and memory; maximum cohort timing "
            f"regression is {decision['maximum_cohort_elapsed_regression_percent']:.1f}%.",
            "",
            "The representation does not remove logical memory work. Exact counters report",
            "logical requests and physical queue events separately, and matched A/B runs",
            "require byte-identical normal JSON in both conditions.",
            "",
            "Machine-readable reductions are in `event_batch_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=benchmark.RESULT_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.output_root.resolve()
    result = analyze(root)
    (root / "event_batch_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "event_batch.md").write_text(markdown(result), encoding="utf-8")
    print(root / "event_batch.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
