#!/usr/bin/env python3
"""Run the bounded sm_86 same-bank fixed-result producer-order campaign."""

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
    REPO_ROOT / "results" / "captures" / "sm86_fixed_latency_result_same_bank_20260817_v1"
)
DRIVER = HERE / "fixed_latency_result_same_bank_driver"
LONG_NATIVE = HERE / "fixed_latency_result_same_bank_native.cubin"
LONG_CANONICAL = HERE / "fixed_latency_result_same_bank.cubin"
# One 8,192-generation cubin serves all active stages; short trace twins are
# historical and are not execution inputs.
TRACE_NATIVE = LONG_NATIVE
TRACE_CANONICAL = LONG_CANONICAL


def load_order_campaign():
    path = HERE.parent / "fixed_latency_result_order" / "campaign.py"
    spec = importlib.util.spec_from_file_location("same_bank_order_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORDER = load_order_campaign()
BASE = ORDER.BASE
METRICS = ORDER.METRICS
WIDTHS = (4, 8, 12, 16)
ORDERS = ("fh", "hf")


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    order: str

    @property
    def key(self) -> str:
        return f"w{self.warps}_{self.order}"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(TRACE_CANONICAL if trace else LONG_CANONICAL),
            self.order,
            str(self.warps),
            "8192",
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    cells = [
        Cell("identification" if warps in {4, 8} else "held_out_width", warps, order)
        for warps in WIDTHS
        for order in ORDERS
    ]
    if len(cells) != 8 or len({cell.key for cell in cells}) != 8:
        raise RuntimeError("same-bank matrix must contain eight unique cells")
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


def require_binaries() -> None:
    for path in (DRIVER, LONG_NATIVE, LONG_CANONICAL):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required binary is missing: {path}")


def emit_manifest(output: Path) -> None:
    require_binaries()
    BASE.write_json(
        output / "metadata" / "matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(cell), "key": cell.key} for cell in configured_cells()],
            "ncu_metrics": list(METRICS),
            "long_generations": 8192,
            "trace_generations": 8192,
            "identification_widths": [4, 8],
            "held_out_widths": [12, 16],
            "destination_bank": 0,
            "pair_invariants": [
                "48 instructions and 16 registers per function",
                "same-bank R4/R8 destination stream",
                "equal per-opcode source-bank demand and byte-identical timed consumers",
                "identical controls and code outside producers plus a post-clock commutative oracle",
            ],
        },
    )
    environment: dict[str, object] = {"captured_at": BASE.now()}
    for name, command in {
        "nvidia_smi": [
            "nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "nvcc": ["/usr/local/cuda-13.2/bin/nvcc", "--version"],
        "ncu": [os.environ.get("NCU", "ncu"), "--version"],
    }.items():
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        environment[name] = {
            "command": command,
            "return_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    BASE.write_json(output / "metadata" / "environment.json", environment)


def run_source(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = list(cells)
    for index, cell in enumerate(sentinels, 1):
        destination = output / "ncu-source" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(f"source {index:02d}/{len(sentinels):02d} {cell.key} (existing)")
            continue
        BASE.run_captured(
            [
                ncu, "--force-overwrite", "--export", str(report_base),
                "--profile-from-start", "on", "--launch-count", "1",
                "--replay-mode", "application", "--clock-control", "base",
                "--cache-control", "none", "--section", "SourceCounters",
                *cell.target(trace=False, trials=1),
            ],
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        completed = subprocess.run(
            [
                ncu, "--import", str(report), "--csv", "--page", "source",
                "--print-source", "sass", "--log-file", str(destination / "source.csv"),
            ],
            cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        (destination / "source-export.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (destination / "source-export.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not (destination / "source.csv").is_file():
            raise RuntimeError(f"NCU source export failed for {report}")
        print(f"source {index:02d}/{len(sentinels):02d} {cell.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = {
        "fixed_latency_result_same_bank_native": LONG_NATIVE,
        "fixed_latency_result_same_bank": LONG_CANONICAL,
    }
    sidecars = []
    for name, binary in binaries.items():
        for suffix, option in (("sass", "--dump-sass"), ("resources", "--dump-resource-usage")):
            completed = subprocess.run(
                ["/usr/local/cuda-13.2/bin/cuobjdump", option, str(binary)],
                cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            (destination / f"{name}.{suffix}.txt").write_text(completed.stdout, encoding="utf-8")
            (destination / f"{name}.{suffix}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"cuobjdump {option} failed for {binary}")
        sidecar = destination / f"{name}.control_bits.json"
        BASE.run_captured(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(REPO_ROOT / "tools" / "sass_control" / "emit_control_bits_sidecar.py"),
                "--binary", str(binary), "--arch", "sm_86", "--out", str(sidecar),
                "--toolkit-bindir", "/usr/local/cuda-13.2/bin",
            ],
            destination / f"{name}.control-extraction",
        )
        sidecars.append(sidecar)
    for source, name in (
        (HERE / "fixed_latency_result_same_bank.patch.json", "long_same_bank_patch.json"),
    ):
        BASE.write_json(output / "metadata" / name, json.loads(source.read_text(encoding="utf-8")))

    def audit(paths: list[Path], out_name: str, run_name: str) -> None:
        BASE.run_captured(
            [
                "cargo", "run", "--quiet", "--release", "--manifest-path",
                str(REPO_ROOT / "sim" / "Cargo.toml"), "-p", "sim-sm", "--example",
                "operand_projection_audit", "--", "--out",
                str(output / "metadata" / out_name), *map(str, paths),
            ],
            output / "metadata" / run_name,
            timeout=900,
        )

    audit(sidecars, "operand_projection_audit.json", "operand-projection-run")
    # Historical predecessor sidecars are not prerequisites for this current
    # same-binary vertical capture.
    print(f"static artifacts: {destination}")


def run_simulator(output: Path, cells: Sequence[Cell], force: bool, simulator_binary: Path) -> None:
    binary = simulator_binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"simulator missing: {binary}")
    for index, cell in enumerate(cells, 1):
        destination = output / "simulator" / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary), "--trace", str(output / "nvbit" / cell.key),
            "--control", str(output / "sass" / "fixed_latency_result_same_bank.control_bits.json"),
            "--arch-card", str(REPO_ROOT / "arch_profiles" / "cards" / "sm_86.json"),
            "--mode", "flat",
        ]
        outputs = []
        for repeat in (1, 2):
            path = destination / f"replay-{repeat}.json"
            if path.is_file() and not force:
                outputs.append(path.read_bytes())
                continue
            completed = subprocess.run(
                command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{cell.key}: replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: replay is nondeterministic")
        launch = json.loads(outputs[0])["launches"][0]
        if launch["coverage"]["kind"] != "full":
            raise RuntimeError(f"{cell.key}: incomplete replay")
        print(f"simulator {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action", choices=("manifest", "correctness", "ncu", "source", "nvbit", "static", "simulator")
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=7)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--repeats", type=int, default=7)
    result.add_argument(
        "--simulator-binary", type=Path,
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
            ORDER.run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            ORDER.run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force, args.simulator_binary)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"same-bank campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
