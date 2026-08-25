#!/usr/bin/env python3
"""Run fixed-eight-warp multi-CTA barrier contention validation."""

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
    / "sm86_barrier_pipeline_multicta_20260817_v1"
)
LONG_BINARY = HERE / "multi_cta_barrier"
TRACE_BINARY = HERE / "multi_cta_barrier_trace"


def load_barrier_campaign():
    path = HERE / "campaign.py"
    spec = importlib.util.spec_from_file_location("barrier_base_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.LONG_BINARY = LONG_BINARY
    module.TRACE_BINARY = TRACE_BINARY
    return module


BASE = load_barrier_campaign()


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    ctas_per_sm: int
    warps_per_cta: int
    protocol: str

    @property
    def key(self) -> str:
        return f"c{self.ctas_per_sm}_w{self.warps_per_cta}_{self.protocol}"

    def target(self, binary: Path, trials: int) -> list[str]:
        return [
            str(binary),
            self.protocol,
            str(self.ctas_per_sm),
            str(self.warps_per_cta),
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    cells = [
        Cell("fixed_eight_warp_multicta", ctas, 8 // ctas, protocol)
        for protocol in ("m00", "m10")
        for ctas in (1, 2, 4)
    ]
    if len(cells) != 6 or len({cell.key for cell in cells}) != 6:
        raise RuntimeError("multi-CTA matrix must contain six unique cells")
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
            "ncu_metrics": list(BASE.METRICS),
            "long_generations": 16384,
            "trace_generations": 64,
            "total_warps_per_sm": 8,
            "shared_bytes_per_sm": 96 * 1024,
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


def run_source(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = [cell for cell in cells if cell.protocol == "m10"]
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
            "--launch-skip",
            "0",
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
    directory_name: str = "simulator",
) -> None:
    binary = REPO_ROOT / "sim" / "target" / "release" / "sim-run"
    control = output / "sass" / "multi_cta_barrier_trace.control_bits.json"
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
        expected_blocks = 84 * cell.ctas_per_sm
        if launch["coverage"]["blocks_observed"] != expected_blocks:
            raise RuntimeError(f"{cell.key}: replay block coverage mismatch")
        counters = launch["counters"]
        if counters["instructions_issued"] != counters["records_seen"]:
            raise RuntimeError(f"{cell.key}: replay cardinality mismatch")
        BASE.write_json(
            destination / "replay.json",
            {
                "cell": {**asdict(cell), "key": cell.key},
                "byte_identical": True,
                "coverage": launch["coverage"],
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
        choices=(
            "manifest",
            "correctness",
            "ncu",
            "source",
            "nvbit",
            "static",
            "simulator",
            "simulator-scalar",
            "simulator-iadd",
            "simulator-carry",
            "simulator-carry3",
            "simulator-branch",
            "simulator-branch-forward",
        ),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=7)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
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
            run_simulator(output, cells, args.force)
        elif args.action == "simulator-scalar":
            run_simulator(output, cells, args.force, "simulator-scalar")
        elif args.action == "simulator-iadd":
            run_simulator(output, cells, args.force, "simulator-iadd")
        elif args.action == "simulator-carry":
            run_simulator(output, cells, args.force, "simulator-carry")
        elif args.action == "simulator-carry3":
            run_simulator(output, cells, args.force, "simulator-carry3")
        elif args.action == "simulator-branch":
            run_simulator(output, cells, args.force, "simulator-branch")
        elif args.action == "simulator-branch-forward":
            run_simulator(output, cells, args.force, "simulator-branch-forward")
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"multi-CTA campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
