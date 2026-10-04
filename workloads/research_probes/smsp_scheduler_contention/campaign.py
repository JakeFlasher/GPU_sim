#!/usr/bin/env python3
"""Run the bounded sm_86 SMSP scheduler-contention campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_smsp_scheduler_contention_20260817_v1"
)
LONG_BINARY = HERE / "smsp_scheduler_contention"
TRACE_BINARY = LONG_BINARY


def load_base_campaign():
    path = REPO_ROOT / "workloads" / "research_probes" / "barrier_pipeline" / "campaign.py"
    spec = importlib.util.spec_from_file_location("scheduler_base_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.LONG_BINARY = LONG_BINARY
    module.TRACE_BINARY = TRACE_BINARY
    return module


BASE = load_base_campaign()

METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__cycles_active.sum",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_ipa.sum",
    "smsp__inst_executed_pipe_cbu.sum",
    "smsp__inst_executed_pipe_adu.sum",
    "smsp__inst_executed_pipe_xu.sum",
    "smsp__inst_executed_pipe_uniform.sum",
    "smsp__pipe_alu_cycles_active.sum",
    "smsp__pipe_fma_cycles_active.sum",
    "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
)
BASE.METRICS = METRICS

SAME_PROTOCOLS = ("q00", "q10", "q11", "q12", "q13", "q14", "q15")
CROSS_PROTOCOLS = ("q20", "q21", "q22", "q23", "q24", "q25")
ALL_PROTOCOLS = SAME_PROTOCOLS + CROSS_PROTOCOLS


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    protocol: str
    phase: int

    @property
    def key(self) -> str:
        return f"w{self.warps}_{self.protocol}_f{self.phase}"

    def target(self, binary: Path, trials: int) -> list[str]:
        return [
            str(binary),
            self.protocol,
            str(self.phase),
            str(self.warps),
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    cells: list[Cell] = []
    for protocol in SAME_PROTOCOLS:
        cells.append(Cell("no_contention_control", 4, protocol, 0))
    for phase in (0, 8):
        for protocol in ALL_PROTOCOLS:
            cells.append(Cell("two_warp_identification", 8, protocol, phase))
    for phase in (4, 16):
        for protocol in SAME_PROTOCOLS:
            cells.append(Cell("two_warp_phase_transfer", 8, protocol, phase))
    for phase in (0, 4):
        for protocol in ALL_PROTOCOLS:
            cells.append(Cell("three_warp_held_out", 12, protocol, phase))
    for phase in (0, 16):
        for protocol in ALL_PROTOCOLS:
            cells.append(Cell("four_warp_held_out", 16, protocol, phase))
    if len(cells) != 99 or len({cell.key for cell in cells}) != 99:
        raise RuntimeError("scheduler matrix must contain 99 unique cells")
    return cells


def selected_cells(selector: str | None) -> list[Cell]:
    cells = configured_cells()
    if selector is None:
        return cells
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [cell for cell in cells if cell.key in requested]
    missing = requested - {cell.key for cell in selected}
    if missing:
        raise RuntimeError(f"unknown cell key(s): {', '.join(sorted(missing))}")
    return selected


def emit_manifest(output: Path) -> None:
    BASE.write_json(
        output / "metadata" / "matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(cell), "key": cell.key} for cell in configured_cells()],
            "ncu_metrics": list(METRICS),
            "long_generations": 8192,
            "generations": 8192,
            "modeled_smsps_per_sm": 4,
            "identification": {
                "warps": [4, 8],
                "phases": [0, 8],
                "same_protocols": list(SAME_PROTOCOLS),
                "cross_protocols": list(CROSS_PROTOCOLS),
            },
            "held_out": {
                "warps": [12, 16],
                "phases": [4, 16],
                "cross_family_order_reversals": [
                    ["q20", "q21"],
                    ["q22", "q23"],
                    ["q24", "q25"],
                ],
            },
        },
    )
    environment: dict[str, object] = {"captured_at": BASE.now()}
    for name, command in {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "nvcc": ["/usr/local/cuda-13.2/bin/nvcc", "--version"],
        "ncu": [os.environ.get("NCU", "ncu"), "--version"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        environment[name] = {
            "command": command,
            "return_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    BASE.write_json(output / "metadata" / "environment.json", environment)


def source_cells(cells: Sequence[Cell]) -> list[Cell]:
    keys = {
        *(f"w4_{protocol}_f0" for protocol in ("q10", "q12", "q14")),
        *(f"w8_{protocol}_f0" for protocol in ("q10", "q11", "q12", "q13", "q14", "q15")),
        "w8_q00_f8",
        *(f"w8_{protocol}_f8" for protocol in CROSS_PROTOCOLS),
        *(f"w12_{protocol}_f4" for protocol in ("q10", "q12", "q14", "q20", "q21")),
    }
    return [cell for cell in cells if cell.key in keys]


def run_source(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = source_cells(cells)
    for index, cell in enumerate(sentinels, 1):
        destination = output / "ncu-source" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(f"source {index:02d}/{len(sentinels):02d} {cell.key} (existing)")
            continue
        command = [
            ncu,
            "--force-overwrite",
            "--export",
            str(report_base),
            "--profile-from-start",
            "off",
            "--launch-count",
            "1",
            "--replay-mode",
            "application",
            "--clock-control",
            "base",
            "--cache-control",
            "none",
            "--section",
            "SourceCounters",
            *cell.target(LONG_BINARY, 1),
        ]
        BASE.run_captured(command, destination, timeout=1200)
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        completed = subprocess.run(
            [
                ncu,
                "--import",
                str(report),
                "--csv",
                "--page",
                "source",
                "--print-source",
                "sass",
                "--log-file",
                str(destination / "source.csv"),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (destination / "source-export.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (destination / "source-export.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not (destination / "source.csv").is_file():
            raise RuntimeError(f"NCU source export failed for {report}")
        print(f"source {index:02d}/{len(sentinels):02d} {cell.key}", flush=True)


def run_simulator(
    output: Path,
    cells: Sequence[Cell],
    force: bool,
    directory_name: str,
    simulator_binary: Path,
) -> None:
    if not directory_name or "/" in directory_name or directory_name in {".", ".."}:
        raise RuntimeError("simulator directory must be one safe path component")
    binary = simulator_binary.resolve()
    control = output / "sass" / "smsp_scheduler_contention.control_bits.json"
    if not binary.is_file() or not control.is_file():
        raise RuntimeError("build sim-run --release and run static first")
    for index, cell in enumerate(cells, 1):
        destination = output / directory_name / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        first = destination / "replay-1.json"
        second = destination / "replay-2.json"
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(control),
            "--arch-card",
            str(REPO_ROOT / "arch_profiles" / "cards" / "sm_86.json"),
            "--mode",
            "flat",
        ]
        if not (first.is_file() and second.is_file()) or force:
            outputs: list[bytes] = []
            for repeat in (1, 2):
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                (destination / f"replay-{repeat}.stderr.txt").write_bytes(
                    completed.stderr
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{cell.key}: replay {repeat} failed with {completed.returncode}"
                    )
                (destination / f"replay-{repeat}.json").write_bytes(completed.stdout)
                outputs.append(completed.stdout)
            if outputs[0] != outputs[1]:
                raise RuntimeError(f"{cell.key}: replay is nondeterministic")
        if first.read_bytes() != second.read_bytes():
            raise RuntimeError(f"{cell.key}: saved replays differ")
        launch = json.loads(first.read_bytes())["launches"][0]
        counters = launch["counters"]
        if counters["instructions_issued"] != counters["records_seen"]:
            raise RuntimeError(f"{cell.key}: replay cardinality mismatch")
        if launch["coverage"]["kind"] != "full":
            raise RuntimeError(f"{cell.key}: replay coverage is not full")
        BASE.write_json(
            destination / "replay.json",
            {
                "cell": {**asdict(cell), "key": cell.key},
                "command": command,
                "byte_identical": True,
                "coverage": launch["coverage"],
                "instructions_issued": counters["instructions_issued"],
                "cycles": launch["timing"]["diagnostic_last_issue_cycles"],
                "readiness_schema_version": launch["timing"]["readiness"][
                    "schema_version"
                ],
            },
        )
        print(f"simulator {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=("manifest", "correctness", "ncu", "source", "nvbit", "static", "simulator"),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=7)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--simulator-directory", default="simulator")
    result.add_argument(
        "--simulator-binary",
        type=Path,
        default=REPO_ROOT / "sim" / "target" / "release" / "sim-run",
    )
    result.add_argument("--force", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        cells = selected_cells(args.cells)
        emit_manifest(output)
        if args.action == "manifest":
            print(output / "metadata" / "matrix.json")
        elif args.action == "correctness":
            BASE.run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            BASE.run_ncu(
                output,
                cells,
                repeat_start=args.repeat_start,
                repeats=args.repeats,
                force=args.force,
            )
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "nvbit":
            BASE.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            BASE.run_static(output)
        elif args.action == "simulator":
            run_simulator(
                output,
                cells,
                args.force,
                args.simulator_directory,
                args.simulator_binary,
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"scheduler campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
