#!/usr/bin/env python3
"""Run the sm_86 fixed-result versus L1TEX writeback-priority campaign."""

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
    / "sm86_fixed_latency_result_load_priority_20260817_v1"
)
DRIVER = HERE / "fixed_latency_result_load_priority_driver"
LONG_NATIVE = HERE / "fixed_latency_result_load_priority_native.cubin"
TRACE_NATIVE = LONG_NATIVE
VARIANTS = (
    "same_noncollision",
    "same_collision",
    "opposite_noncollision",
    "opposite_shifted",
)
WIDTHS = (1, 4, 8, 12, 16)


def load_order_campaign():
    path = HERE.parent / "fixed_latency_result_order" / "campaign.py"
    spec = importlib.util.spec_from_file_location("load_priority_order_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORDER = load_order_campaign()
BASE = ORDER.BASE
METRICS = ORDER.METRICS + (
    "smsp__inst_executed_pipe_lsu.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "dram__sectors_read.sum",
)


def cubin(variant: str, trace: bool) -> Path:
    return HERE / f"fixed_latency_result_load_priority_{variant}.cubin"


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    variant: str
    bank_relation: str
    producer_cycle: int

    @property
    def key(self) -> str:
        return f"w{self.warps}_{self.variant}"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(cubin(self.variant, trace)),
            self.variant,
            str(self.warps),
            str(1024),
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    cells = []
    for warps in WIDTHS:
        stage = (
            "single_warp_identification"
            if warps == 1
            else (
                "single_warp_per_smsp_transfer"
                if warps == 4
                else "held_out_width"
            )
        )
        for variant in VARIANTS:
            cells.append(
                Cell(
                    stage=stage,
                    warps=warps,
                    variant=variant,
                    bank_relation=(
                        "same" if variant.startswith("same_") else "opposite"
                    ),
                    producer_cycle=(
                        23
                        if variant in {"same_collision", "opposite_shifted"}
                        else 22
                    ),
                )
            )
    if len(cells) != 20 or len({cell.key for cell in cells}) != 20:
        raise RuntimeError("load-priority matrix must contain twenty unique cells")
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
    paths = [DRIVER, LONG_NATIVE, TRACE_NATIVE]
    paths.extend(cubin(variant, trace) for variant in VARIANTS for trace in (False, True))
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required binary is missing: {path}")


def emit_manifest(output: Path) -> None:
    require_binaries()
    BASE.write_json(
        output / "metadata" / "matrix.json",
        {
            "schema_version": 1,
            "cells": [
                {**asdict(cell), "key": cell.key} for cell in configured_cells()
            ],
            "ncu_metrics": list(METRICS),
            "long_generations": 1024,
            "trace_generations": 1024,
            "identification_widths": [1, 4],
            "held_out_widths": [8, 12, 16],
            "load_consumer_cycle": 29,
            "fixed_producer_cycles": {"noncollision": 22, "collision": 23},
            "pair_invariants": [
                "48 instructions and 18 compiler-declared registers",
                "one prewarmed per-lane LDG per generation with barrier-2 wait",
                "equal instruction work and selected static stall sum 46",
                "collision pairs differ only at IADD/NOP stall fields",
                "bank controls differ only by complete R11-to-R14 load-result rename",
                "load-only consumer precedes a safely delayed fixed-result consumer",
            ],
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


def run_ncu(
    output: Path,
    cells: Sequence[Cell],
    repeat_start: int,
    repeats: int,
    force: bool,
) -> None:
    ncu = os.environ.get("NCU", "ncu")
    total = len(cells) * repeats
    progress = 0
    for cell in cells:
        for repeat in range(repeat_start, repeat_start + repeats):
            progress += 1
            destination = output / "ncu" / cell.key / f"repeat-{repeat}"
            report_base = destination / "ncu-report"
            report = destination / "ncu-report.ncu-rep"
            if report.is_file() and not force:
                print(
                    f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)",
                    flush=True,
                )
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
                    "--metrics",
                    ",".join(METRICS),
                    *cell.target(trace=False, trials=1),
                ],
                destination,
                timeout=1200,
            )
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            BASE.export_ncu(ncu, report, destination)
            print(
                f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat}",
                flush=True,
            )


def run_source(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = [cell for cell in cells if cell.warps in {1, 4}]
    for index, cell in enumerate(sentinels, 1):
        destination = output / "ncu-source" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(
                f"source {index:02d}/{len(sentinels):02d} {cell.key} (existing)",
                flush=True,
            )
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
        (destination / "source-export.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (destination / "source-export.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not (destination / "source.csv").is_file():
            raise RuntimeError(f"NCU source export failed for {report}")
        print(
            f"source {index:02d}/{len(sentinels):02d} {cell.key}", flush=True
        )


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = {"fixed_latency_result_load_priority_native": LONG_NATIVE}
    binaries.update(
        {f"fixed_latency_result_load_priority_{v}": cubin(v, False)
         for v in VARIANTS}
    )
    sidecars = []
    for name, binary in binaries.items():
        for suffix, option in (
            ("sass", "--dump-sass"),
            ("resources", "--dump-resource-usage"),
        ):
            completed = subprocess.run(
                ["/usr/local/cuda-13.2/bin/cuobjdump", option, str(binary)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"{name}.{suffix}.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (destination / f"{name}.{suffix}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(f"cuobjdump {option} failed for {binary}")
        sidecar = destination / f"{name}.control_bits.json"
        BASE.run_captured(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(
                    REPO_ROOT
                    / "tools"
                    / "sass_control"
                    / "emit_control_bits_sidecar.py"
                ),
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

    BASE.write_json(
        output / "metadata" / "long_result_load_priority_patch.json",
        json.loads(
            (HERE / "fixed_latency_result_load_priority.patch.json")
            .read_text(encoding="utf-8")
        ),
    )

    def audit(paths: list[Path], out_name: str, run_name: str) -> None:
        BASE.run_captured(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(REPO_ROOT / "sim" / "Cargo.toml"),
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
    )
    predecessors = sorted(
        {path for pattern in predecessor_patterns for path in REPO_ROOT.glob(pattern)}
    )
    if len(predecessors) != 64:
        raise RuntimeError(
            f"expected 64 retained predecessor sidecars, got {len(predecessors)}"
        )
    audit(
        predecessors + sidecars,
        "retained_corpus_operand_projection_audit.json",
        "retained-corpus-operand-projection-run",
    )
    print(f"static artifacts: {destination}")


def run_simulator(
    output: Path,
    cells: Sequence[Cell],
    force: bool,
    simulator_binary: Path,
    simulator_mode: str,
    simulator_directory: str,
) -> None:
    if simulator_mode not in {"flat", "l1"}:
        raise RuntimeError("simulator mode must be flat or l1")
    if (
        not simulator_directory
        or "/" in simulator_directory
        or simulator_directory in {".", ".."}
    ):
        raise RuntimeError("simulator directory must be one safe path component")
    binary = simulator_binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"simulator missing: {binary}")
    for index, cell in enumerate(cells, 1):
        destination = output / simulator_directory / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        control = (
            output
            / "sass"
            / f"fixed_latency_result_load_priority_{cell.variant}.control_bits.json"
        )
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(control),
            "--arch-card",
            str(REPO_ROOT / "arch_profiles" / "cards" / "sm_86.json"),
            "--mode",
            simulator_mode,
        ]
        if simulator_mode == "l1":
            command[command.index("--mode"):command.index("--mode")] = [
                "--pchase-card",
                str(REPO_ROOT / "data" / "pchase" / "sm_86_static_card.json"),
            ]
        outputs = []
        for repeat in (1, 2):
            path = destination / f"replay-{repeat}.json"
            if path.is_file() and not force:
                outputs.append(path.read_bytes())
                continue
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
                raise RuntimeError(f"{cell.key}: replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: replay is nondeterministic")
        launch = json.loads(outputs[0])["launches"][0]
        if launch["coverage"]["kind"] != "full":
            raise RuntimeError(f"{cell.key}: incomplete replay")
        if launch["counters"]["instructions_issued"] != launch["counters"][
            "records_seen"
        ]:
            raise RuntimeError(f"{cell.key}: cardinality mismatch")
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
        ),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=7)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument(
        "--simulator-binary",
        type=Path,
        default=REPO_ROOT / "sim" / "target" / "release" / "sim-run",
    )
    result.add_argument("--simulator-mode", choices=("flat", "l1"), default="flat")
    result.add_argument("--simulator-directory", default="simulator")
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
            run_ncu(
                output,
                cells,
                args.repeat_start,
                args.repeats,
                args.force,
            )
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(
                output,
                cells,
                args.force,
                args.simulator_binary,
                args.simulator_mode,
                args.simulator_directory,
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"result/load priority campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
