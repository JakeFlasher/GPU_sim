#!/usr/bin/env python3
"""Run the bounded sm_86 fixed-latency result/writeback-queue campaign."""

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
    / "sm86_fixed_latency_result_queue_20260817_v1"
)
DRIVER = HERE / "fixed_latency_result_queue_driver"
LONG_NATIVE = HERE / "fixed_latency_result_queue.cubin"
LONG_COLLISION = HERE / "fixed_latency_result_queue_collision.cubin"
LONG_SEPARATED = HERE / "fixed_latency_result_queue_separated.cubin"


def load_base_campaign():
    path = REPO_ROOT / "workloads" / "research_probes" / "barrier_pipeline" / "campaign.py"
    spec = importlib.util.spec_from_file_location("result_queue_base_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
    "smsp__inst_executed_pipe_fma_type_fp16.sum",
    "smsp__inst_executed_pipe_cbu.sum",
    "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
)
VARIANTS = ("collision", "separated")
WIDTHS = (4, 8, 12, 16)


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    variant: str

    @property
    def key(self) -> str:
        return f"w{self.warps}_{self.variant}"

    def cubin(self, trace: bool) -> Path:
        # One patched cubin per treatment is used for correctness, NVBit,
        # replay, and NCU. The former 64-generation trace cubins made the
        # campaign compare different programs.
        if self.variant == "collision":
            return LONG_COLLISION
        return LONG_SEPARATED

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(self.cubin(trace)),
            self.variant,
            str(self.warps),
            str(8192),
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    cells = [
        Cell("identification" if warps in {4, 8} else "held_out_width", warps, variant)
        for warps in WIDTHS
        for variant in VARIANTS
    ]
    if len(cells) != 8 or len({cell.key for cell in cells}) != 8:
        raise RuntimeError("result-queue matrix must contain eight unique cells")
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
    for path in (
        DRIVER,
        LONG_NATIVE,
        LONG_COLLISION,
        LONG_SEPARATED,
    ):
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
            "generations": 8192,
            "identification_widths": [4, 8],
            "held_out_widths": [12, 16],
            "pair_invariants": [
                "instruction PCs, opcodes, operands, predicates, and reuse bits",
                "barrier and wait-mask control fields",
                "resource declarations and instruction counts",
                "runtime inputs, arithmetic results, and host oracle",
            ],
            "pair_treatment": (
                "four HADD2 controls use stall=1 for collision and stall=3 for "
                "separated; both clear yield and reuse"
            ),
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


def run_correctness(output: Path, cells: Sequence[Cell], trials: int) -> None:
    for index, cell in enumerate(cells, 1):
        destination = output / "correctness" / cell.key
        completed = BASE.run_captured(cell.target(trace=False, trials=trials), destination)
        payload = BASE.measurement_json_line(completed.stdout, cell.key)
        BASE.write_json(
            destination / "result.json",
            {"cell": {**asdict(cell), "key": cell.key}, "measurement": payload},
        )
        print(f"correctness {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_ncu(
    output: Path, cells: Sequence[Cell], repeat_start: int, repeats: int, force: bool
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
                print(f"ncu {progress:02d}/{total:02d} {cell.key} r{repeat} (existing)")
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
                timeout=900,
            )
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            BASE.export_ncu(ncu, report, destination)
            print(f"ncu {progress:02d}/{total:02d} {cell.key} r{repeat}", flush=True)


def source_cells(cells: Sequence[Cell]) -> list[Cell]:
    return [cell for cell in cells if cell.warps in {4, 16}]


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
            raise RuntimeError(f"NCU source export failed for {report}")
        print(f"source {index:02d}/{len(sentinels):02d} {cell.key}", flush=True)


def run_nvbit(output: Path, cells: Sequence[Cell], force: bool) -> None:
    tracer = REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
    if not tracer.is_file():
        raise RuntimeError(f"NVBit tracer is missing: {tracer}")
    base_environment = dict(os.environ)
    protobuf_lib = REPO_ROOT / ".toolchain" / "pb" / "lib"
    existing_library_path = base_environment.get("LD_LIBRARY_PATH", "")
    base_environment.update(
        {
            "CUDA_INJECTION64_PATH": str(tracer),
            "USER_DEFINED_FOLDERS": "1",
            "ACTIVE_FROM_START": "1",
            "SKIP_ENHANCED_TRACER": "0",
            "ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S": "30",
            "ENHANCED_TRACER_MAX_PARSER_INPUT_KB": "64",
            "LD_LIBRARY_PATH": str(protobuf_lib)
            + ((":" + existing_library_path) if existing_library_path else ""),
        }
    )
    for index, cell in enumerate(cells, 1):
        destination = output / "nvbit" / cell.key
        trace = destination / "dynamic_trace.pb"
        if trace.is_file() and not force:
            print(f"nvbit {index:02d}/{len(cells):02d} {cell.key} (existing)")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        environment = dict(base_environment)
        environment["TRACES_FOLDER"] = str(destination.resolve())
        completed = BASE.run_captured(
            cell.target(trace=True, trials=1),
            destination,
            environment=environment,
            timeout=900,
        )
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError(f"NVBit did not produce a nonempty trace for {cell.key}")
        BASE.write_json(
            destination / "capture.json",
            {
                "cell": {**asdict(cell), "key": cell.key},
                "stdout_bytes": len(completed.stdout.encode()),
                "stderr_bytes": len(completed.stderr.encode()),
                "dynamic_trace_bytes": trace.stat().st_size,
            },
        )
        print(f"nvbit {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = {
        "fixed_latency_result_queue_native": LONG_NATIVE,
        "fixed_latency_result_queue_collision": LONG_COLLISION,
        "fixed_latency_result_queue_separated": LONG_SEPARATED,
    }
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
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(REPO_ROOT / "tools" / "sass_control" / "emit_control_bits_sidecar.py"),
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
        (HERE / "fixed_latency_result_queue.patch.json", "long_result_pair_patch.json"),
            ):
        BASE.write_json(output / "metadata" / name, json.loads(source.read_text(encoding="utf-8")))
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
            str(output / "metadata" / "operand_projection_audit.json"),
            *map(str, sidecars),
        ],
        output / "metadata" / "operand-projection-run",
        timeout=900,
    )
    print(f"static artifacts: {destination}")


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
    if not binary.is_file():
        raise RuntimeError(f"simulator missing: {binary}")
    for index, cell in enumerate(cells, 1):
        destination = output / directory_name / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(
                output
                / "sass"
                / f"fixed_latency_result_queue_{cell.variant}.control_bits.json"
            ),
            "--arch-card",
            str(REPO_ROOT / "arch_profiles" / "cards" / "sm_86.json"),
            "--mode",
            "flat",
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
        if launch["counters"]["instructions_issued"] != launch["counters"]["records_seen"]:
            raise RuntimeError(f"{cell.key}: cardinality mismatch")
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
            run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "nvbit":
            run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(
                output,
                cells,
                args.force,
                args.simulator_directory,
                args.simulator_binary,
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"result-queue campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
