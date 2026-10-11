#!/usr/bin/env python3
"""Run tiny locality scenarios through the SM120 MRC implementation.

This is a microbenchmark/observability driver, not a test suite.  It records
the model's semantic bins, MRC values and wall time for small sequences that
exercise cold references, zero-distance reuse, capacity boundaries, warm
passes, and bounded-prefix sampling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sm120_mrc_model import analyze_sequence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "all_cold_unique": list(range(64)),
        "single_sector_reuse": [7] * 64,
        "working_set_boundary": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        "distance_zero_and_one": [0, 0, 1, 1, 0, 1],
        "warm_pass_source": [0, 1, 2, 3, 0, 1, 2, 3],
        "bounded_prefix_source": list(range(80)) + [0, 1, 2, 3] * 20,
    }
    rows = []
    for name, sequence in cases.items():
        max_references = 32 if name == "bounded_prefix_source" else max(1, len(sequence))
        start = time.perf_counter_ns()
        row = analyze_sequence(
            sequence[:max_references],
            [1, 2, 3, 4, 8, 32, 64],
            32,
            max(0, len(sequence) - max_references),
            "toy_microbenchmark",
        )
        elapsed_ns = time.perf_counter_ns() - start
        rows.append(
            {
                "name": name,
                "source_references": len(sequence),
                "modeled_references": row["references"],
                "cold": row["cold_references"],
                "overflow": row["overflow_references"],
                "sample_bin": row["sample_bin_references"],
                "histogram": row["histogram"],
                "mrc": row["mrc"],
                "wall_ns": elapsed_ns,
            }
        )
    result = {
        "schema_version": 1,
        "model": "sm120_mrc_toy_microbenchmark_v1",
        "cases": rows,
        "interpretation": "semantic observations of logical LRU scenarios; no hardware-cache claim",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "wall_ns": sum(row["wall_ns"] for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
