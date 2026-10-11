#!/usr/bin/env python3
"""Inventory launch/timing scope for the retained SM120 corpus.

This is an observation-oriented companion to ``sm120_replay_bench.py``.  It
does not infer missing warm-up or epilogue launches and it never feeds timing
labels into prediction.  Every launch is classified as declared, measured, or
unknown from the files that actually exist under the local evidence root.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RUNTIME_RE = re.compile(r"Avg\s+runtime\s*:?[ \t]*([0-9.eE+-]+)\s*ms", re.I)


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def stats_rows(path: Path) -> list[dict[str, Any]]:
    rows = csv_rows(path)
    output: list[dict[str, Any]] = []
    for row in rows:
        raw = str(row.get("kernel id", "")).strip()
        if not raw or raw.lower() == "kernel id":
            continue
        match = re.search(r"kernel-(\d+)", raw)
        kernel_id = int(match.group(1)) if match else None
        output.append(
            {
                "kernel_id": kernel_id,
                "device_id": row.get("device_id"),
                "stream_id": row.get("stream_id"),
                "kernel_name": row.get("kernel mangled name"),
                "grid": [row.get("grid_dimX"), row.get("grid_dimY"), row.get("grid_dimZ")],
                "block": [row.get("block_dimX"), row.get("block_dimY"), row.get("block_dimZ")],
                "total_insts": row.get("total_insts"),
            }
        )
    return output


def timing_for(timing: Any, capture: str) -> dict[str, Any] | None:
    if not isinstance(timing, dict):
        return None
    value: Any = timing.get("captures", timing)
    if isinstance(value, dict) and "captures" in value:
        value = value["captures"]
    if isinstance(value, dict) and capture in value:
        value = value[capture]
    return value if isinstance(value, dict) else None


def inventory(work_root: Path, timing_path: Path | None) -> dict[str, Any]:
    timing: Any = None
    if timing_path is not None and timing_path.is_file():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            timing = None
    captures = work_root / "captures"
    rows: list[dict[str, Any]] = []
    for capture in sorted(captures.iterdir() if captures.is_dir() else []):
        if not capture.is_dir() or not (capture / "dynamic_trace.pb").is_file():
            continue
        launches = stats_rows(capture / "stats.csv")
        stdout = (capture / "program.stdout").read_text(encoding="utf-8", errors="replace") if (capture / "program.stdout").is_file() else ""
        runtimes = [float(match.group(1)) for match in RUNTIME_RE.finditer(stdout)]
        profile_rows = csv_rows(work_root / "profiles" / capture.name / "raw.csv")
        hardware = timing_for(timing, capture.name)
        coverage = str((hardware or {}).get("coverage", "unknown")).lower()
        scope = str((hardware or {}).get("scope", "unknown"))
        declared_full = bool((hardware or {}).get("coverage_full")) or coverage == "full"
        phase_labels = []
        for launch in launches:
            phase_labels.append(
                {
                    "kernel_id": launch.get("kernel_id"),
                    "phase": "measured_region" if declared_full else "unknown",
                    "evidence": "uninstrumented_timing_contract" if declared_full else "no_phase_marker",
                }
            )
        rows.append(
            {
                "capture_name": capture.name,
                "capture_dir": str(capture),
                "launch_count_from_stats": len(launches),
                "launches": launches,
                "ncu_row_count": len(profile_rows),
                "tracer_runtime_ms_values": runtimes,
                "hardware_timing": hardware,
                "scope": {
                    "status": "full_declared" if declared_full else "unknown",
                    "coverage": coverage,
                    "scope": scope,
                    "phase_labels": phase_labels,
                    "initialization": "unknown_without_source_marker",
                    "warmup": "unknown_without_source_marker",
                    "measured_iteration": "measured_region" if declared_full else "unknown",
                    "epilogue": "unknown_without_source_marker",
                },
            }
        )
    return {
        "schema_version": 1,
        "model": "sm120_launch_scope_inventory_v1",
        "work_root": str(work_root),
        "timing_input": str(timing_path) if timing_path else None,
        "capture_count": len(rows),
        "launch_count": sum(int(row["launch_count_from_stats"]) for row in rows),
        "full_declared_captures": sum(row["scope"]["status"] == "full_declared" for row in rows),
        "rows": rows,
        "prediction_inputs": "none; scope table is reporting-only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/dev/shm/sm120_rtx_5090_work"))
    parser.add_argument("--hardware-timing", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.work_root, args.hardware_timing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("capture_count", "launch_count", "full_declared_captures")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
