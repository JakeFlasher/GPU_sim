#!/usr/bin/env python3
"""Reduce the bounded store-free async-source live slice."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


ROUTES = ("l1", "dram")
DEPTHS = (4, 8)
REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__sectors_read.sum",
)


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> dict[str, str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [
                {str(key).strip(): str(value).strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
                if row.get(" kernel mangled name", row.get("kernel mangled name", row.get("Kernel Name")))
            ]
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot read NCU CSV {path}: {error}") from error
    if len(rows) != 1:
        raise AnalysisError(f"{path}: expected one kernel row, got {len(rows)}")
    return rows[0]


def ncu_cell(root: Path, route: str, depth: int) -> dict[str, Any]:
    base = root / "ncu" / f"async_source_only_{route}_d{depth}_g64"
    paths = sorted(base.glob("repeat-*/ncu.csv"))
    if len(paths) != 3:
        raise AnalysisError(f"{base}: expected three NCU repeats, got {len(paths)}")
    repeats: list[dict[str, float]] = []
    for path in paths:
        row = read_csv(path)
        try:
            repeats.append({metric: float(row[metric]) for metric in METRICS})
        except (KeyError, ValueError) as error:
            raise AnalysisError(f"{path}: missing/non-numeric metric: {error}") from error
    medians = {metric: statistics.median(item[metric] for item in repeats) for metric in METRICS}
    active = [item["sm__cycles_active.max"] for item in repeats]
    cv = (max(active) - min(active)) / statistics.mean(active)
    return {
        "route": route,
        "depth": depth,
        "repeats": repeats,
        "median": medians,
        "active_cycle_relative_range": cv,
        "csv_sha256": [sha256(path) for path in paths],
    }


def simulator_cell(root: Path, route: str, depth: int) -> dict[str, Any]:
    base = root / "simulator" / f"async_source_only_{route}_d{depth}_g64"
    first = base / "replay-1.json"
    second = base / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise AnalysisError(f"{base}: simulator replay twins are incomplete")
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    if first_bytes != second_bytes:
        raise AnalysisError(f"{base}: simulator replay twins differ")
    document = json.loads(first_bytes)
    launches = document.get("launches")
    if document.get("schema_version") != 1 or not isinstance(launches, list) or len(launches) != 1:
        raise AnalysisError(f"{first}: launch/schema contract failed")
    launch = launches[0]
    if launch.get("coverage", {}).get("kind") != "full":
        raise AnalysisError(f"{first}: coverage is not full")
    return {
        "replay_sha256": sha256(first),
        "instructions_issued": launch["counters"]["instructions_issued"],
        "records_seen": launch["counters"]["records_seen"],
        "counters": launch["counters"],
        "timing": launch["timing"],
        "typed_tails": launch["counters"]["unmodeled_tails"],
        "coverage": launch["coverage"],
    }


def run(root: Path, output: Path) -> int:
    root = root.resolve()
    cells: list[dict[str, Any]] = []
    errors: list[str] = []
    for route in ROUTES:
        for depth in DEPTHS:
            try:
                ncu = ncu_cell(root, route, depth)
                sim = simulator_cell(root, route, depth)
                cells.append({"route": route, "depth": depth, "ncu": ncu, "simulator": sim})
            except AnalysisError as error:
                errors.append(str(error))
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "probe_binary": str(REPO_ROOT / "workloads/research_probes/async_source_only/async_source_only"),
        "cells": cells,
        "errors": errors,
        "status": "pass_pipeline_no_timing_patch" if not errors else "invalid_slice",
        "decision": "no_bypass_destination_timing_promotion",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "cells": len(cells)}, sort_keys=True))
    return 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.root, args.output)
    except (AnalysisError, OSError, ValueError, KeyError) as error:
        print(f"async-source-only analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
