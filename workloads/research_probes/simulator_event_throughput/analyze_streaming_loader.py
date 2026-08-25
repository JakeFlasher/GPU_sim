#!/usr/bin/env python3
"""Reduce the descriptor-rostered streaming direct-loader experiment."""

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
    comparison = load(root / "streaming_loader_ab/comparison.json")
    byte_identity = benchmark.validate_comparison_evidence(
        root / "streaming_loader_ab",
        comparison,
        ("full_compact", "streaming"),
    )
    expected = {case.key for case in benchmark.CASES}
    if set(comparison["summaries"]) != expected:
        raise RuntimeError("streaming comparison case set differs")
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
        byte_identity
        and maximum_timing_regression <= 3
        and maximum_memory_regression <= 1
        and float(gap["peak_rss_change_percent"]) < -10
    )
    return {
        "schema_version": 1,
        "method": comparison["method"],
        "cases": cases,
        "decision": {
            "retain_streaming_direct_loader": retain,
            "raw_and_campaign_loaders_unchanged": True,
            "normal_json_byte_identical_across_conditions": byte_identity,
            "maximum_cohort_elapsed_regression_percent": maximum_timing_regression,
            "maximum_cohort_peak_rss_regression_percent": maximum_memory_regression,
            "largest_peak_rss_reduction_percent": largest_memory_reduction,
            "gap64_peak_rss_reduction_percent": -float(
                gap["peak_rss_change_percent"]
            ),
            "scope": (
                "Complete bounded descriptor-relative roster followed by exact-generation "
                "one-file-at-a-time read, validation, and compaction."
            ),
        },
    }


def markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Descriptor-rostered streaming direct loading",
        "",
        "Both conditions retain the same compact validated trace. The streaming",
        "condition snapshots the complete bounded roster, then reads and compacts",
        "one exact file generation at a time instead of retaining the full raw map.",
        "",
        "| case | full-map MiB | streaming MiB | RSS change | time change |",
        "|---|---:|---:|---:|---:|",
    ]
    for case, item in result["cases"].items():
        lines.append(
            f"| {case} | {item['full_compact_peak_rss_kib_median'] / 1024:.1f} | "
            f"{item['streaming_peak_rss_kib_median'] / 1024:.1f} | "
            f"{item['peak_rss_change_percent']:+.1f}% | "
            f"{item['elapsed_change_percent']:+.1f}% |"
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Retain streaming loading: "
            f"**{decision['retain_streaming_direct_loader']}**.",
            "Gap-heavy peak RSS changes by "
            f"-{decision['gap64_peak_rss_reduction_percent']:.1f}%; maximum cohort "
            "timing regression is "
            f"{decision['maximum_cohort_elapsed_regression_percent']:.1f}%.",
            "",
            "Machine-readable reductions are in `streaming_loader_summary.json`.",
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
    (root / "streaming_loader_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "streaming_loader.md").write_text(markdown(result), encoding="utf-8")
    print(root / "streaming_loader.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

