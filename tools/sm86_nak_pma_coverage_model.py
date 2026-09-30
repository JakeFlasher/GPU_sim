#!/usr/bin/env python3
"""Summarize direct NAK PMA sampling coverage by static body size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time


def one(directory: Path, shader: Path | None = None, pma: Path | None = None) -> dict:
    shader = shader or directory / "nak_shader.bin"
    if pma is None:
        candidates = sorted(directory.glob("pma_*.bin"))
        if not candidates:
            raise RuntimeError(f"no PMA blob in {directory}")
        pma = candidates[0]
    blob = pma.read_bytes()
    shader_bytes = shader.read_bytes()
    code_bytes = max(len(shader_bytes) - 728, 0)
    records = len(blob) // 8
    raw = [blob[index:index + 8] for index in range(0, records * 8, 8)]
    return {
        "directory": str(directory),
        "shader": str(shader),
        "pma": str(pma),
        "shader_bytes": len(shader_bytes),
        "code_bytes_after_info_header": code_bytes,
        "static_instruction_words": code_bytes // 16,
        "pma_records": records,
        "records_per_static_instruction": records / max(code_bytes // 16, 1),
        "unique_raw_records": len(set(raw)),
        "unique_raw_record_fraction": len(set(raw)) / max(records, 1),
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    if args.shader and len(args.shader) != len(args.capture):
        raise ValueError("--shader count must match --capture count")
    if args.pma and len(args.pma) != len(args.capture):
        raise ValueError("--pma count must match --capture count")
    samples = [
        one(path, (args.shader[index] if args.shader else None),
            (args.pma[index] if args.pma else None))
        for index, path in enumerate(args.capture)
    ]
    densities = [item["records_per_static_instruction"] for item in samples]
    report = {
        "schema": "sm86_nak_pma_coverage_model_v1",
        "samples": samples,
        "coverage_summary": {
            "median_records_per_static_instruction": statistics.median(densities),
            "min_records_per_static_instruction": min(densities),
            "max_records_per_static_instruction": max(densities),
        },
        "runtime_seconds": time.perf_counter() - started,
        "contract": "coverage-density observation only; no physical pipe/slot/issue inference",
        "limitations": [
            "raw PMA record encoding is not interpreted as a physical route witness",
            "sample density depends on body duration, PMA interval, warp population, and driver state",
            "three bodies do not establish universal sampling law",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--shader", type=Path, action="append")
    parser.add_argument("--pma", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"coverage_summary": report["coverage_summary"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
