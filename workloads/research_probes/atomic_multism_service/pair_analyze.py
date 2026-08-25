#!/usr/bin/env python3
"""Reduce the sm_86 pair-collision and absolute-phase sweeps."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_COLLISION = REPO_ROOT / "results/captures/sm86_atomic_pair_collision_20260818_v1"
DEFAULT_PHASE = REPO_ROOT / "results/captures/sm86_atomic_pair_phase_20260818_v1"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def median_max(payload: dict) -> float:
    return statistics.median(max(trial) for trial in payload["samples_cycles"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-root", type=Path, default=DEFAULT_COLLISION)
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE)
    args = parser.parse_args()
    collision = args.collision_root.resolve()
    phase = args.phase_root.resolve()

    collision_cells = read(collision / "metadata/matrix.json")["cells"]
    phase_cells = read(phase / "metadata/matrix.json")["cells"]
    if len(collision_cells) != 288 or len(phase_cells) != 256:
        raise RuntimeError("pair matrices have unexpected size")

    rows = []
    maximum_range = 0
    for root, cells, kind in ((collision, collision_cells, "offset"), (phase, phase_cells, "phase")):
        for cell in cells:
            payload = read(root / "correctness" / cell["key"] / "result.json")["measurement"]
            if payload["correct"] is not True or len(payload["samples_cycles"]) != 7:
                raise RuntimeError(f"{cell['key']}: invalid correctness evidence")
            cycles = median_max(payload)
            maximum_range = max(
                maximum_range,
                max(max(trial) for trial in payload["samples_cycles"])
                - min(max(trial) for trial in payload["samples_cycles"]),
            )
            if kind == "offset":
                base = 0
                delta = cell["offset_bytes"]
            else:
                base = cell["base_bytes"]
                delta = cell["delta_bytes"]
            same_region = base // 256 == (base + delta) // 256
            observed_collision = cycles > 100_000
            rows.append(
                {
                    "kind": kind,
                    "key": cell["key"],
                    "base_bytes": base,
                    "delta_bytes": delta,
                    "cycles": cycles,
                    "same_256b_region": same_region,
                    "observed_collision": observed_collision,
                    "classification_correct": same_region == observed_collision,
                }
            )

    errors = [row for row in rows if not row["classification_correct"]]
    if errors:
        raise RuntimeError(f"256-byte collision rule has {len(errors)} errors")
    collision_cycles = [row["cycles"] for row in rows if row["observed_collision"]]
    independent_cycles = [row["cycles"] for row in rows if not row["observed_collision"]]
    summary = {
        "schema_version": 1,
        "campaign": {
            "offset_cells": 288,
            "phase_cells": 256,
            "correctness_executions": (288 + 256) * 7,
            "maximum_seven_trial_range_cycles": maximum_range,
        },
        "classification": {
            "rule": "floor(address_a / 256) == floor(address_b / 256)",
            "threshold_cycles": 100_000,
            "errors": 0,
            "tested_pairs": len(rows),
            "collision_median_cycles": statistics.median(collision_cycles),
            "collision_min_cycles": min(collision_cycles),
            "collision_max_cycles": max(collision_cycles),
            "independent_median_cycles": statistics.median(independent_cycles),
            "independent_min_cycles": min(independent_cycles),
            "independent_max_cycles": max(independent_cycles),
        },
        "rows": rows,
        "decision": {
            "status": "aligned_256b_cross_warp_contention_region_identified",
            "promote_region_bytes": 256,
            "timing_queue": False,
            "reason": (
                "the pair relation is exact, but many-region throughput still depends "
                "on an unidentified shared scheduling/banking network"
            ),
        },
    }
    analysis = collision / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    c = summary["classification"]
    report = "\n".join(
        [
            "# sm_86 pairwise atomic collision oracle",
            "",
            "## Decision",
            "",
            "Cross-warp/CTA global RED contention follows one exact aligned-256-byte",
            "region relation across all 544 tested address pairs:",
            "",
            "```text",
            "floor(address_a / 256) == floor(address_b / 256)",
            "```",
            "",
            "Pairs in one region form the high-contention cluster; pairs in different",
            "regions form the independent two-address cluster. Absolute base phase is",
            "fully covered at four-byte alignment for deltas 128, 256, 260, and 4096.",
            "",
            "## Integrity and separation",
            "",
            f"- 544 cells and {summary['campaign']['correctness_executions']:,} correct executions",
            f"- zero classification errors",
            f"- collision cluster median {c['collision_median_cycles']:.0f} cycles, range {c['collision_min_cycles']:.0f}–{c['collision_max_cycles']:.0f}",
            f"- independent cluster median {c['independent_median_cycles']:.0f} cycles, range {c['independent_min_cycles']:.0f}–{c['independent_max_cycles']:.0f}",
            "",
            "The result identifies a serialization/contention region, not a fixed global",
            "atomic-unit hash. Many-region B84 throughput still varies with spacing, so",
            "the shared scheduling/banking network remains unmodeled and no timing queue",
            "is activated.",
            "",
        ]
    )
    (analysis / "README.md").write_text(report, encoding="utf-8")
    print(analysis / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
