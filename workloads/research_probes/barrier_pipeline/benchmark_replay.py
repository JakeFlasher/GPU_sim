#!/usr/bin/env python3
"""Measure process-level direct replay on matched predecessor sentinels."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_barrier_pipeline_coupled_20260817_v1"
    / "metadata"
    / "host_replay_final.json"
)
SOURCE = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_async_barrier_readiness_20260817_v4"
)


def command(key: str) -> list[str]:
    return [
        str(REPO_ROOT / "sim" / "target" / "release" / "sim-run"),
        "--trace",
        str(SOURCE / "nvbit" / key),
        "--control",
        str(SOURCE / "sass" / "async_barrier_overlap_trace.control_bits.json"),
        "--arch-card",
        str(REPO_ROOT / "arch_profiles" / "cards" / "sm_86.json"),
        "--pchase-card",
        str(REPO_ROOT / "data" / "pchase" / "sm_86_static_card.json"),
        "--mode",
        "l1-l2",
        "--kernel-id",
        "1",
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def measure(key: str, samples: int, warmups: int) -> dict[str, float | int]:
    invocation = command(key)
    for _ in range(warmups):
        completed = subprocess.run(
            invocation,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{key}: warmup failed with {completed.returncode}")
    durations: list[float] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        completed = subprocess.run(
            invocation,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        stop = time.perf_counter_ns()
        if completed.returncode != 0:
            raise RuntimeError(f"{key}: replay failed with {completed.returncode}")
        durations.append((stop - start) / 1_000_000.0)
    return {
        "samples": samples,
        "warmups": warmups,
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.mean(durations),
        "min_ms": min(durations),
        "p95_ms": percentile(durations, 0.95),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--samples", type=int, default=200)
    result.add_argument("--warmups", type=int, default=10)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.samples < 1 or args.warmups < 0:
            raise RuntimeError("sample and warmup counts are invalid")
        result = {
            key: measure(key, args.samples, args.warmups)
            for key in ("w4_c00_na_g0", "w4_c11_l1_g0")
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, RuntimeError) as error:
        print(f"benchmark replay: {error}", file=__import__("sys").stderr)
        return 1
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
