#!/usr/bin/env python3
"""Replay the retained matched STORE corpus through dirty-L2 observability."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOURCE_ROOT = REPO_ROOT / "results/captures/sm86_memory_retirement_service_20260818_v1"
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_l2_dirty_eviction_topology_20260901_v1"
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
CONTROL = SOURCE_ROOT / "sass/memory_retirement_service_trace.control_bits.json"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
TOPOLOGY_FIELDS = (
    "l2_evicted_lines",
    "l2_evicted_valid_sectors",
    "l2_evicted_pending_lines",
    "l2_evicted_pending_sectors",
    "l2_dirty_eviction_lines",
    "l2_dirty_evicted_sectors",
    "maximum_terminal_l2_dirty_lines",
    "maximum_terminal_l2_dirty_sectors",
)


@dataclass(frozen=True, slots=True)
class Cell:
    key: str
    store_instructions: int
    store_sectors: int
    expected_dirty_evicted_sectors: int
    expected_terminal_dirty_sectors: int


CELLS = (
    Cell("b84_store_none_line_stream_d16_l32", 1344, 43008, 0, 43008),
    Cell("b84_store_none_line_stream_d64_l32", 5376, 172032, 122880, 49152),
    Cell("b84_store_none_line_reuse_d16_l32", 1344, 43008, 0, 2688),
    Cell("b84_store_none_line_reuse_d64_l32", 5376, 172032, 0, 2688),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def command(cell: Cell) -> list[str]:
    return [
        str(SIM_RUN),
        "--trace",
        str(SOURCE_ROOT / "nvbit" / cell.key),
        "--control",
        str(CONTROL),
        "--arch-card",
        str(ARCH_CARD),
        "--pchase-card",
        str(PCHASE_CARD),
        "--mode",
        "l1-l2",
    ]


def validate_trace(cell: Cell) -> dict[str, Any]:
    trace = SOURCE_ROOT / "nvbit" / cell.key
    with (trace / "stats.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"{cell.key}: expected one trace stats row")
    row = {key.strip(): value.strip() for key, value in rows[0].items()}
    require(
        row["total_insts"] == row["total_reported_insts"],
        f"{cell.key}: total/reported trace mismatch",
    )
    for name in (
        "unsupported_memory_path.json",
        "unmapped_memory_range.json",
        "cross_allocation_lane_span.json",
    ):
        value = json.loads((trace / "extra_info" / name).read_text(encoding="utf-8"))
        require(value == [], f"{cell.key}: {name} is nonempty")
    return {
        "total_instructions": int(row["total_insts"]),
        "dynamic_trace_sha256": sha256(trace / "dynamic_trace.pb"),
    }


def ncu_dram_write_samples(cell: Cell) -> list[float]:
    samples: list[float] = []
    directory = SOURCE_ROOT / "ncu" / cell.key
    for path in sorted(directory.glob("repeat-*/ncu.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        material = [
            row
            for row in rows
            if row.get("Kernel Name", "").strip()
            and row.get("dram__sectors_write.sum", "").strip()
        ]
        require(len(material) == 1, f"{path}: expected one NCU kernel row")
        samples.append(float(material[0]["dram__sectors_write.sum"]))
    require(samples, f"{cell.key}: no NCU DRAM-write samples")
    return samples


def topology(launch: dict[str, Any]) -> dict[str, int]:
    dram = launch["counters"]["dram"]
    return {field: int(dram.get(field, 0)) for field in TOPOLOGY_FIELDS}


def normalized_without_topology(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for launch in result["launches"]:
        dram = launch["counters"]["dram"]
        for field in TOPOLOGY_FIELDS:
            dram.pop(field, None)
    return result


def validate_cell(
    cell: Cell, first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    require(first == second, f"{cell.key}: replay twins differ")
    predecessor = json.loads(
        (SOURCE_ROOT / "simulator" / cell.key / "replay-1.json").read_text(
            encoding="utf-8"
        )
    )
    require(
        normalized_without_topology(first) == predecessor,
        f"{cell.key}: non-topology replay output changed",
    )
    require(len(first["launches"]) == 1, f"{cell.key}: expected one launch")
    launch = first["launches"][0]
    issue = launch["timing"]["readiness"]["global_memory_issue"]
    require(issue["store_instructions"] == cell.store_instructions, f"{cell.key}: STORE instructions")
    require(issue["store_sectors"] == cell.store_sectors, f"{cell.key}: STORE sectors")
    observed = topology(launch)
    require(
        observed["l2_dirty_evicted_sectors"] == cell.expected_dirty_evicted_sectors,
        f"{cell.key}: dirty victim sectors",
    )
    require(
        observed["maximum_terminal_l2_dirty_sectors"]
        == cell.expected_terminal_dirty_sectors,
        f"{cell.key}: terminal dirty sectors",
    )
    require(
        observed["l2_dirty_evicted_sectors"]
        + observed["maximum_terminal_l2_dirty_sectors"]
        <= cell.store_sectors,
        f"{cell.key}: dirty topology exceeds STORE sectors",
    )
    tails = launch["timing"]["completion_frontier"]["unmodeled_tails"]
    require(tails["dirty_writeback"] is True, f"{cell.key}: dirty tail cleared")
    require(tails["coherence"] is True, f"{cell.key}: coherence tail cleared")
    ncu = ncu_dram_write_samples(cell)
    return {
        "trace": validate_trace(cell),
        "store_instructions": cell.store_instructions,
        "store_sectors": cell.store_sectors,
        "topology": observed,
        "dirty_conservation_sum": observed["l2_dirty_evicted_sectors"]
        + observed["maximum_terminal_l2_dirty_sectors"],
        "normalized_predecessor_equal": True,
        "deterministic_twins": True,
        "ncu_dram_write_sectors": {
            "role": "independent directional comparison only",
            "samples": ncu,
            "minimum": min(ncu),
            "median": statistics.median(ncu),
            "maximum": max(ncu),
        },
        "completion_status": launch["timing"]["completion_frontier"]["status"],
        "unmodeled_tails": tails,
    }


def run(output: Path) -> dict[str, Any]:
    for path in (SIM_RUN, CONTROL, ARCH_CARD, PCHASE_CARD):
        require(path.is_file() and path.stat().st_size > 0, f"missing input {path}")
    reports: dict[str, Any] = {}
    for index, cell in enumerate(CELLS, 1):
        destination = output / "simulator" / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        outputs: list[bytes] = []
        for twin in (1, 2):
            completed = subprocess.run(
                command(cell),
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
                check=False,
            )
            (destination / f"replay-{twin}.stderr.txt").write_bytes(completed.stderr)
            require(
                completed.returncode == 0,
                f"{cell.key}: replay {twin} failed: {completed.stderr.decode(errors='replace')}",
            )
            json.loads(completed.stdout)
            (destination / f"replay-{twin}.json").write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        first, second = map(json.loads, outputs)
        reports[cell.key] = validate_cell(cell, first, second)
        write_json(
            destination / "identity.json",
            {
                "schema_version": 1,
                "command": command(cell),
                "sim_run_sha256": sha256(SIM_RUN),
                "stdout_sha256": hashlib.sha256(outputs[0]).hexdigest(),
                "deterministic_twins": True,
            },
        )
        print(f"dirty-topology {index}/{len(CELLS)} {cell.key}", flush=True)
    unique_d16 = reports["b84_store_none_line_stream_d16_l32"]
    unique_d64 = reports["b84_store_none_line_stream_d64_l32"]
    reuse_d16 = reports["b84_store_none_line_reuse_d16_l32"]
    reuse_d64 = reports["b84_store_none_line_reuse_d64_l32"]
    require(unique_d16["topology"]["l2_dirty_evicted_sectors"] == 0, "D16 unique crossed capacity")
    require(unique_d64["topology"]["l2_dirty_evicted_sectors"] > 0, "D64 unique emitted no victims")
    require(reuse_d16["topology"]["l2_dirty_evicted_sectors"] == 0, "D16 reuse evicted dirty data")
    require(reuse_d64["topology"]["l2_dirty_evicted_sectors"] == 0, "D64 reuse evicted dirty data")
    return {
        "schema_version": 1,
        "milestone": "sm86_l2_dirty_eviction_topology",
        "status": "complete_exact_dirty_topology_no_write_service",
        "generated_at": now(),
        "revision": revision(),
        "inputs": {
            str(path.relative_to(REPO_ROOT)): {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (SIM_RUN, CONTROL, ARCH_CARD, PCHASE_CARD)
        },
        "cells": reports,
        "decision": (
            "Exact dirty-sector state, replacement victims, and terminal residency pass "
            "the capacity/reuse discriminator without changing predecessor timing or lookup "
            "outputs. NCU DRAM writes are directionally consistent but composite; no write "
            "service timing or tail clearing is authorized."
        ),
        "scientific_boundary": {
            "ncu_used_as_implementation_input": False,
            "writeback_event_scheduled": False,
            "writeback_latency_identified": False,
            "dirty_writeback_tail_cleared": False,
            "coherence_tail_cleared": False,
        },
    }


def validate_existing(output: Path) -> dict[str, Any]:
    report = json.loads((output / "analysis/report.json").read_text(encoding="utf-8"))
    require(report["status"] == "complete_exact_dirty_topology_no_write_service", "status drift")
    require(report["inputs"][str(SIM_RUN.relative_to(REPO_ROOT))]["sha256"] == sha256(SIM_RUN), "sim-run drift")
    for cell in CELLS:
        first = json.loads((output / "simulator" / cell.key / "replay-1.json").read_text())
        second = json.loads((output / "simulator" / cell.key / "replay-2.json").read_text())
        validate_cell(cell, first, second)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--validate-only", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        if args.validate_only:
            report = validate_existing(output)
        else:
            report = run(output)
            write_json(output / "analysis/report.json", report)
        print(json.dumps({"status": report["status"]}, sort_keys=True))
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError, KeyError) as error:
        print(f"dirty-eviction analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
