#!/usr/bin/env python3
"""Reduce the compact direct trace-payload lifetime experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import benchmark


def load(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError(f"{path}: expected object")
    return document


def analyze(root: Path) -> dict[str, object]:
    comparison = load(root / "trace_lifetime_ab/comparison.json")
    byte_identity = benchmark.validate_comparison_evidence(
        root / "trace_lifetime_ab", comparison, ("raw", "compact")
    )
    expected = {case.key for case in benchmark.CASES}
    if set(comparison["summaries"]) != expected:
        raise RuntimeError(
            "trace lifetime comparison case set differs: "
            f"{sorted(set(comparison['summaries']) ^ expected)}"
        )
    cases = comparison["summaries"]
    maximum_timing_regression = max(
        float(item["elapsed_change_percent"]) for item in cases.values()
    )
    maximum_memory_regression = max(
        float(item["peak_rss_change_percent"]) for item in cases.values()
    )
    largest_memory_reduction = max(
        -float(item["peak_rss_change_percent"]) for item in cases.values()
    )
    gap = cases["admission_red_gap64"]
    retain = (
        maximum_timing_regression <= 3
        and maximum_memory_regression <= 1
        and float(gap["peak_rss_change_percent"]) < -10
        and byte_identity
    )
    return {
        "schema_version": 1,
        "method": comparison["method"],
        "cases": cases,
        "decision": {
            "retain_compact_direct_loader": retain,
            "campaign_and_raw_loader_unchanged": True,
            "normal_json_byte_identical_across_conditions": byte_identity,
            "maximum_cohort_elapsed_regression_percent": maximum_timing_regression,
            "maximum_cohort_peak_rss_regression_percent": maximum_memory_regression,
            "largest_peak_rss_reduction_percent": largest_memory_reduction,
            "gap64_peak_rss_reduction_percent": -float(gap["peak_rss_change_percent"]),
            "scope": (
                "Direct sim-run releases raw threadblock protobuf bytes after the "
                "existing full validation/compaction pass; raw and campaign loaders retain bytes."
            ),
        },
    }


def markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Compact direct trace-payload lifetime",
        "",
        "Both conditions execute the same complete validation/compaction pass. The",
        "compact condition releases raw threadblock protobuf bytes before replay;",
        "campaign and explicit raw-retaining APIs remain unchanged.",
        "",
        "| case | raw MiB | compact MiB | RSS change | time change |",
        "|---|---:|---:|---:|---:|",
    ]
    for case, item in result["cases"].items():
        lines.append(
            f"| {case} | {item['raw_peak_rss_kib_median'] / 1024:.1f} | "
            f"{item['compact_peak_rss_kib_median'] / 1024:.1f} | "
            f"{item['peak_rss_change_percent']:+.1f}% | "
            f"{item['elapsed_change_percent']:+.1f}% |"
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Retain compact direct loading: **{decision['retain_compact_direct_loader']}**.",
            "The gap-heavy case peak RSS falls "
            f"{decision['gap64_peak_rss_reduction_percent']:.1f}%; maximum cohort "
            "timing regression is "
            f"{decision['maximum_cohort_elapsed_regression_percent']:.1f}%.",
            "",
            "This is a lifetime optimization, not sampling or weakened admission.",
            "Every payload is still decoded and semantically validated before raw bytes",
            "are released, and matched conditions require byte-identical normal JSON.",
            "",
            "Machine-readable reductions are in `trace_lifetime_summary.json`.",
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
    (root / "trace_lifetime_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "trace_lifetime.md").write_text(markdown(result), encoding="utf-8")
    print(root / "trace_lifetime.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
