#!/usr/bin/env python3
"""Run the sm_86 fixed-result bypass-frontier campaign."""

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
    / "sm86_fixed_latency_result_bypass_20260817_v1"
)
DRIVER = HERE / "fixed_latency_result_bypass_driver"
CANARY_DRIVER = HERE / "fixed_latency_result_bypass_canary_driver"
LONG_NATIVE = HERE / "fixed_latency_result_bypass_native.cubin"
VARIANTS = ("fixed_bypass", "fixed_retired", "global_frontier", "global_late")
WIDTHS = (1, 4, 8, 12, 16)


def load_priority_campaign():
    path = HERE.parent / "fixed_latency_result_load_priority" / "campaign.py"
    spec = importlib.util.spec_from_file_location("bypass_priority_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRIORITY = load_priority_campaign()
ORDER = PRIORITY.ORDER
BASE = PRIORITY.BASE
METRICS = PRIORITY.METRICS


def cubin(variant: str, trace: bool) -> Path:
    return HERE / f"fixed_latency_result_bypass_{variant}.cubin"


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    variant: str
    kind: str
    distance: int

    @property
    def key(self) -> str:
        return f"w{self.warps}_{self.variant}"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(cubin(self.variant, trace)),
            self.kind,
            str(self.warps),
            str(1025),
            str(trials),
            self.variant,
        ]


def configured_cells() -> list[Cell]:
    metadata = {
        "fixed_bypass": ("fixed", 4),
        "fixed_retired": ("fixed", 5),
        "global_frontier": ("global", 5),
        "global_late": ("global", 6),
    }
    cells = []
    for warps in WIDTHS:
        stage = "identification" if warps in {1, 4} else "held_out_width"
        for variant in VARIANTS:
            kind, distance = metadata[variant]
            cells.append(Cell(stage, warps, variant, kind, distance))
    if len(cells) != 20 or len({cell.key for cell in cells}) != 20:
        raise RuntimeError("bypass matrix must contain twenty unique cells")
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
    paths = [DRIVER, CANARY_DRIVER, LONG_NATIVE]
    paths.extend(cubin(variant, False) for variant in VARIANTS)
    paths.extend(
        HERE / f"fixed_latency_result_bypass_{name}.cubin"
        for name in ("fixed_invalid", "canary_d4", "canary_d5")
    )
    for path in paths:
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
            "long_generations": 1025,
            "generations": 1025,
            "identification_widths": [1, 4],
            "held_out_widths": [8, 12, 16],
            "fixed_boundary": {"invalid": 3, "bypass": 4, "control": 5},
            "global_boundary": {"invalid": 4, "frontier": 5, "control": 6},
            "pair_invariants": [
                "identical instruction work within each fixed/global pair",
                "equal selected static stall sums within each pair",
                "fixed d4/d5 changes only producer and checksum controls",
                "global d5/d6 changes only spacing and checksum controls",
                "host checksum validates every valid treatment",
            ],
        },
    )
    environment = {"captured_at": BASE.now()}
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


def run_calibration(output: Path) -> None:
    destination = output / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    commands = {
        "fixed_d3": [
            str(DRIVER),
            str(HERE / "fixed_latency_result_bypass_fixed_invalid.cubin"),
            "fixed",
            "1",
            "1025",
            "7",
            "fixed_invalid",
        ],
        "global_d5": [
            str(CANARY_DRIVER),
            str(HERE / "fixed_latency_result_bypass_canary_d5.cubin"),
            "7",
        ],
    }
    results = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (destination / f"{name}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (destination / f"{name}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        results[name] = {
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    d4_runs = []
    d4_command = [
        str(CANARY_DRIVER),
        str(HERE / "fixed_latency_result_bypass_canary_d4.cubin"),
        "1",
    ]
    for repeat in range(1, 8):
        completed = subprocess.run(
            d4_command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        d4_runs.append(
            {
                "repeat": repeat,
                "return_code": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
    results["global_d4"] = {"command": d4_command, "runs": d4_runs}
    if results["fixed_d3"]["return_code"] != 1 or '"correct":false' not in results["fixed_d3"]["stdout"]:
        raise RuntimeError("fixed d3 did not fail its checksum")
    if results["global_d5"]["return_code"] != 0 or '"correct":true' not in results["global_d5"]["stdout"]:
        raise RuntimeError("global d5 did not pass")
    if any(
        row["return_code"] != 1 or "CUDA_ERROR_ILLEGAL_ADDRESS" not in row["stderr"]
        for row in d4_runs
    ):
        raise RuntimeError("global d4 did not fail with the expected CUDA error")
    BASE.write_json(destination / "summary.json", results)
    print(destination / "summary.json")


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = {
        "fixed_latency_result_bypass_native": LONG_NATIVE,
    }
    for variant in VARIANTS:
        binaries[f"fixed_latency_result_bypass_{variant}"] = cubin(variant, False)
    for name in ("fixed_invalid", "canary_d4", "canary_d5"):
        binaries[f"fixed_latency_result_bypass_{name}"] = (
            HERE / f"fixed_latency_result_bypass_{name}.cubin"
        )
    sidecars = []
    for name, binary in binaries.items():
        for suffix, option in (("sass", "--dump-sass"), ("resources", "--dump-resource-usage")):
            completed = subprocess.run(
                ["/usr/local/cuda-13.2/bin/cuobjdump", option, str(binary)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"{name}.{suffix}.txt").write_text(completed.stdout, encoding="utf-8")
            (destination / f"{name}.{suffix}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"cuobjdump {option} failed for {binary}")
        sidecar = destination / f"{name}.control_bits.json"
        BASE.run_captured(
            [
                str(REPO_ROOT / ".venv/bin/python"),
                str(REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"),
                "--binary",
                str(binary),
                "--arch",
                "sm_86",
                "--out",
                str(sidecar),
                "--toolkit-bindir",
                "/usr/local/cuda-13.2/bin",
            ],
            destination / f"{name}.control-extraction",
        )
        sidecars.append(sidecar)
    for source, name in (
        (HERE / "fixed_latency_result_bypass.patch.json", "long_bypass_patch.json"),
        (HERE / "fixed_latency_result_bypass_calibration.patch.json", "calibration_bypass_patch.json"),
    ):
        BASE.write_json(output / "metadata" / name, json.loads(source.read_text(encoding="utf-8")))

    def audit(paths: list[Path], out_name: str, run_name: str) -> None:
        BASE.run_captured(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(REPO_ROOT / "sim/Cargo.toml"),
                "-p",
                "sim-sm",
                "--example",
                "operand_projection_audit",
                "--",
                "--out",
                str(output / "metadata" / out_name),
                *map(str, paths),
            ],
            output / "metadata" / run_name,
            timeout=900,
        )

    audit(sidecars, "operand_projection_audit.json", "operand-projection-run")
    predecessor_patterns = (
        "results/captures/sm86_async_barrier_readiness_20260817_v*/sass/*.control_bits.json",
        "results/captures/sm86_barrier_pipeline_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_barrier_pipeline_coupled_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_barrier_pipeline_multicta_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_cycle_residual_20260817_v1/control/*.control_bits.json",
        "results/captures/sm86_cycle_residual_20260817_v1/research-probes/async-cross-warp/*.control_bits.json",
        "results/captures/sm86_fixed_latency_allocate_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_fixed_latency_completion_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_fixed_latency_reuse_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_issue_port_control_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_smsp_scheduler_contention_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_fixed_latency_result_queue_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_fixed_latency_result_order_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_fixed_latency_result_same_bank_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_fixed_latency_result_capacity_20260817_v1/sass/*.control_bits.json",
        "results/captures/sm86_fixed_latency_result_load_priority_20260817_v1/sass/*.control_bits.json",
    )
    predecessors = sorted({path for pattern in predecessor_patterns for path in REPO_ROOT.glob(pattern)})
    print(destination)


def run_simulator(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = REPO_ROOT / "sim/target/release/sim-run"
    for index, cell in enumerate(cells, 1):
        destination = output / "simulator" / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(output / "sass" / f"fixed_latency_result_bypass_{cell.variant}.control_bits.json"),
            "--arch-card",
            str(REPO_ROOT / "arch_profiles/cards/sm_86.json"),
            "--pchase-card",
            str(REPO_ROOT / "data/pchase/sm_86_static_card.json"),
            "--mode",
            "l1",
        ]
        outputs = []
        for repeat in (1, 2):
            path = destination / f"replay-{repeat}.json"
            if path.is_file() and not force:
                outputs.append(path.read_bytes())
                continue
            completed = subprocess.run(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{cell.key}: replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: nondeterministic replay")
        print(f"simulator {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=("manifest", "calibration", "correctness", "ncu", "source", "nvbit", "static", "simulator"),
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
            print(output / "metadata/matrix.json")
        elif args.action == "calibration":
            run_calibration(output)
        elif args.action == "correctness":
            ORDER.run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            PRIORITY.run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            PRIORITY.run_source(output, cells, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"bypass campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
