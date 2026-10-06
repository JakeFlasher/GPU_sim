#!/usr/bin/env python3
"""Run the dense sm_86 producer-shift/SMSP-occupancy campaign."""

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
    / "sm86_fixed_latency_producer_shift_20260817_v1"
)
WIDTHS = tuple(range(4, 17))
VARIANTS = ("fixed_bypass", "fixed_retired", "global_frontier", "global_late")
SOURCE_WIDTHS = {4, 5, 6, 8, 9, 12, 13, 16}


def load_bypass_campaign():
    path = HERE.parent / "fixed_latency_result_bypass" / "campaign.py"
    spec = importlib.util.spec_from_file_location("shift_bypass_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BYPASS = load_bypass_campaign()
ORDER = BYPASS.ORDER
PRIORITY = BYPASS.PRIORITY
BASE = BYPASS.BASE
METRICS = BYPASS.METRICS
DRIVER = BYPASS.DRIVER


def cubin(variant: str, trace: bool) -> Path:
    return BYPASS.cubin(variant, trace)


def stage(warps: int) -> str:
    if warps == 4:
        return "one_warp_per_smsp_baseline"
    if warps <= 8:
        return "second_warp_partition_fill"
    if warps <= 12:
        return "third_warp_partition_fill"
    return "fourth_warp_partition_fill"


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
    cells = [
        Cell(stage(warps), warps, variant, *metadata[variant])
        for warps in WIDTHS
        for variant in VARIANTS
    ]
    if len(cells) != 52 or len({cell.key for cell in cells}) != 52:
        raise RuntimeError("producer-shift matrix must contain 52 unique cells")
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
    if not DRIVER.is_file():
        raise RuntimeError(f"driver missing: {DRIVER}")
    for variant in VARIANTS:
        for trace in (False, True):
            path = cubin(variant, trace)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"binary missing: {path}")


def emit_manifest(output: Path) -> None:
    require_binaries()
    BASE.write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(cell), "key": cell.key} for cell in configured_cells()],
            "ncu_metrics": list(METRICS),
            "long_generations": 1025,
            "trace_generations": 65,
            "widths": list(WIDTHS),
            "source_counter_widths": sorted(SOURCE_WIDTHS),
            "smsp_assignment": "physical_warp_slot_modulo_4",
            "pair_invariants": [
                "same valid cubins and host oracles as the bypass campaign",
                "equal work and equal selected stall sums within each pair",
                "only warp count changes across the dense occupancy sweep",
                "W5/W9/W13 add a new warp to SMSP0; subsequent widths fill SMSP1-3",
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
    BASE.write_json(output / "metadata/environment.json", environment)


def run_source(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = [cell for cell in cells if cell.warps in SOURCE_WIDTHS]
    for index, cell in enumerate(sentinels, 1):
        destination = output / "ncu-source" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(f"source {index:02d}/{len(sentinels):02d} {cell.key} (existing)")
            continue
        BASE.run_captured(
            [
                ncu,
                "--force-overwrite",
                "--export",
                str(report_base),
                "--profile-from-start",
                "on",
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
        (destination / "source-export.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (destination / "source-export.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not (destination / "source.csv").is_file():
            raise RuntimeError(f"source export failed for {cell.key}")
        print(f"source {index:02d}/{len(sentinels):02d} {cell.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = {}
    for variant in VARIANTS:
        binaries[f"fixed_latency_producer_shift_{variant}"] = cubin(variant, False)
        binaries[f"fixed_latency_producer_shift_trace_{variant}"] = cubin(variant, True)
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
                raise RuntimeError(f"cuobjdump failed for {binary}")
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
        "results/captures/sm86_fixed_latency_result_bypass_20260817_v1/sass/*.control_bits.json",
    )
    predecessors = sorted({path for pattern in predecessor_patterns for path in REPO_ROOT.glob(pattern)})
    if len(predecessors) != 87:
        raise RuntimeError(f"expected 87 predecessor sidecars, got {len(predecessors)}")
    audit(predecessors + sidecars, "retained_corpus_operand_projection_audit.json", "retained-corpus-operand-projection-run")
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
            str(output / "sass" / f"fixed_latency_producer_shift_trace_{cell.variant}.control_bits.json"),
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
    result.add_argument("action", choices=("manifest", "correctness", "ncu", "source", "nvbit", "static", "simulator"))
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
        elif args.action == "correctness":
            ORDER.run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            PRIORITY.run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"producer-shift campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
