#!/usr/bin/env python3
"""Run the sm_86 memory-retirement service campaign."""

from __future__ import annotations

import argparse
import csv
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
    REPO_ROOT / "results/captures/sm86_memory_retirement_service_20260818_v1"
)
LONG_BINARY = HERE / "memory_retirement_service"
TRACE_BINARY = HERE / "memory_retirement_service_trace"
TAILS = ("control", "load", "store", "red")
FENCES = ("none", "device", "system")

METRICS = (
    "gpc__cycles_elapsed.avg",
    "gpc__cycles_elapsed.min",
    "gpc__cycles_elapsed.max",
    "gpc__cycles_elapsed.sum",
    "gpc__cycles_elapsed.max.per_second",
    "gpu__time_duration.sum",
    "sm__cycles_active.avg",
    "sm__cycles_active.min",
    "sm__cycles_active.max",
    "sm__cycles_active.sum",
    "smsp__cycles_active.sum",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_requests_srcunit_tex_op_red.sum",
    "lts__t_sectors_op_red.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
)


def load_helpers():
    path = HERE.parent / "launch_cycle_semantics/campaign.py"
    spec = importlib.util.spec_from_file_location("retirement_campaign_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load campaign helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = load_helpers()
BASE = HELPERS.HELPERS
REPLAY = HELPERS.REPLAY
latest_input_mtime_ns = REPLAY.latest_input_mtime_ns


def ensure_release_simulator() -> Path:
    completed = subprocess.run(
        ["cargo", "build", "--release", "-p", "sim-run"],
        cwd=REPO_ROOT / "sim",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"release simulator build failed:\n{completed.stdout}")
    binary = REPO_ROOT / "sim/target/release/sim-run"
    if not binary.is_file():
        raise RuntimeError(f"release simulator missing after build: {binary}")
    return binary


def complete_csv(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, strict=True))
    except (OSError, csv.Error, UnicodeDecodeError):
        return False
    return len(rows) >= 3 and all(row for row in rows) and any(len(row) > 1 for row in rows)


def complete_ncu_export(report: Path, *exports: Path) -> bool:
    return (
        report.is_file()
        and report.stat().st_size > 0
        and all(complete_csv(path) for path in exports)
        and all(path.stat().st_mtime_ns >= report.stat().st_mtime_ns for path in exports)
    )


def complete_nvbit_capture(destination: Path) -> bool:
    required_nonempty = (
        destination / "dynamic_trace.pb",
        destination / "stats.csv",
        destination / "capture.json",
        destination / "command.txt",
        destination / "run.json",
        destination / "stdout.txt",
        destination / "extra_info/allocations.json",
        destination / "extra_info/enhanced_execution_info.json",
        destination / "extra_info/traced_kernel_modules.json",
        destination / "extra_info/unsupported_memory_path.json",
        destination / "extra_info/unmapped_memory_range.json",
        destination / "extra_info/cross_allocation_lane_span.json",
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_nonempty):
        return False
    if not (destination / "stderr.txt").is_file():
        return False
    try:
        structured = (
            json.loads((destination / "capture.json").read_text(encoding="utf-8")),
            json.loads((destination / "run.json").read_text(encoding="utf-8")),
            json.loads(
                (destination / "extra_info/allocations.json").read_text(encoding="utf-8")
            ),
            json.loads(
                (destination / "extra_info/enhanced_execution_info.json").read_text(
                    encoding="utf-8"
                )
            ),
            json.loads(
                (destination / "extra_info/traced_kernel_modules.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        return all(isinstance(value, (dict, list)) for value in structured) and all(
            json.loads((destination / "extra_info" / name).read_text(encoding="utf-8")) == []
            for name in (
                "unsupported_memory_path.json",
                "unmapped_memory_range.json",
                "cross_allocation_lane_span.json",
            )
        )
    except (OSError, json.JSONDecodeError):
        return False


def preserve_json_snapshot(path: Path, value: object) -> None:
    if path.is_file():
        existing = REPLAY.load_json_strict(path)
        if existing != value:
            raise RuntimeError(
                f"retained configuration differs from current input: {path}; use a new output root"
            )
        return
    BASE.write_json(path, value)


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    tail: str
    fence: str
    layout: str
    blocks: int
    depth: int
    active_lanes: int

    @property
    def key(self) -> str:
        return (
            f"b{self.blocks}_{self.tail}_{self.fence}_{self.layout}_"
            f"d{self.depth}_l{self.active_lanes}"
        )

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(TRACE_BINARY if trace else LONG_BINARY),
            self.tail,
            self.fence,
            self.layout,
            str(self.blocks),
            str(self.depth),
            str(self.active_lanes),
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    cells: dict[str, Cell] = {}

    def add(cell: Cell) -> None:
        cells.setdefault(cell.key, cell)

    for tail in TAILS:
        for blocks in (1, 84):
            for depth in (1, 4, 16, 64):
                add(
                    Cell(
                        "depth_identification",
                        tail,
                        "none",
                        "line_stream",
                        blocks,
                        depth,
                        32,
                    )
                )

    layouts = {
        "load": ("line_stream", "word_stream", "same_stream"),
        "store": ("line_stream", "word_stream"),
        "red": ("line_stream", "word_stream", "same_stream"),
    }
    for tail, tail_layouts in layouts.items():
        for blocks in (1, 84):
            for layout in tail_layouts:
                for active_lanes in (1, 8, 32):
                    add(
                        Cell(
                            "lane_layout_transfer",
                            tail,
                            "none",
                            layout,
                            blocks,
                            16,
                            active_lanes,
                        )
                    )

    for tail in ("load", "store", "red"):
        for blocks in (1, 84):
            for depth in (16, 64):
                add(
                    Cell(
                        "address_reuse_transfer",
                        tail,
                        "none",
                        "line_reuse",
                        blocks,
                        depth,
                        32,
                    )
                )

    for tail in TAILS:
        for blocks in (1, 84):
            for fence in ("device", "system"):
                add(
                    Cell(
                        "explicit_fence_discriminator",
                        tail,
                        fence,
                        "line_stream",
                        blocks,
                        16,
                        32,
                    )
                )

    material = list(cells.values())
    if len(material) != 102:
        raise RuntimeError(f"memory-retirement matrix must contain 102 cells, got {len(material)}")
    return material


def source_cells() -> list[Cell]:
    return [
        cell
        for cell in configured_cells()
        if cell.blocks == 84
        and cell.depth == 16
        and cell.active_lanes == 32
        and cell.layout == "line_stream"
    ]


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
    for path in (LONG_BINARY, TRACE_BINARY):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required binary missing: {path}")


def emit_manifest(output: Path, *, refresh_environment: bool = False) -> None:
    require_binaries()
    cells = configured_cells()
    matrix = {
            "schema_version": 1,
            "cells": [{**asdict(cell), "key": cell.key} for cell in cells],
            "source_counter_cells": [cell.key for cell in source_cells()],
            "ncu_metrics": list(METRICS),
            "ncu_repeats": 3,
            "correctness_trials": 3,
            "work_iterations": 256,
            "invariants": [
                "one warp per CTA",
                "one compile-time tail and fence specialization per launch",
                "line layouts use 128-byte lane spacing",
                "matched control preserves grid, depth loop, work, and fence scope",
                "device and system fences compile to MEMBAR.SC.GPU/SYS plus compiler epilogue",
                "NCU and NVBit execute in separate processes",
            ],
    }
    if refresh_environment:
        BASE.write_json(output / "metadata/matrix.json", matrix)
    else:
        preserve_json_snapshot(output / "metadata/matrix.json", matrix)
    environment_path = output / "metadata/environment.json"
    if environment_path.is_file() and not refresh_environment:
        return
    environment: dict[str, object] = {
        "observed_at": BASE.now(),
        "role": "tool/host observation; raw artifact mtimes define capture chronology",
    }
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "nvcc": ["/usr/local/cuda-13.2/bin/nvcc", "--version"],
        "ncu": [os.environ.get("NCU", "ncu"), "--version"],
    }
    for name, command in commands.items():
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
    BASE.write_json(environment_path, environment)


def result_payload(completed: subprocess.CompletedProcess[str], cell: Cell) -> dict:
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"{cell.key}: target emitted no JSON")
    value = json.loads(lines[-1])
    if value.get("correct") is not True:
        raise RuntimeError(f"{cell.key}: correctness failed")
    return value


def run_correctness(output: Path, cells: Sequence[Cell], trials: int) -> None:
    for index, cell in enumerate(cells, 1):
        destination = output / "correctness" / cell.key
        completed = BASE.run_captured(
            cell.target(trace=False, trials=trials), destination, timeout=300
        )
        BASE.write_json(
            destination / "result.json",
            {"cell": {**asdict(cell), "key": cell.key}, "measurement": result_payload(completed, cell)},
        )
        print(f"correctness {index:03d}/{len(cells):03d} {cell.key}", flush=True)


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
            if HELPERS.complete_ncu_export(report, destination / "ncu.csv") and not force:
                print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)", flush=True)
                continue
            BASE.run_captured(
                [
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
                    "all",
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
            if not HELPERS.complete_ncu_export(report, destination / "ncu.csv"):
                raise RuntimeError(f"NCU export is incomplete for {cell.key} repeat {repeat}")
            print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat}", flush=True)


def run_source(output: Path, force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    cells = source_cells()
    for index, cell in enumerate(cells, 1):
        destination = output / "ncu-source" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if (
            HELPERS.complete_ncu_export(
                report, destination / "ncu.csv", destination / "source.csv"
            )
            and not force
        ):
            print(f"source {index:02d}/{len(cells):02d} {cell.key} (existing)", flush=True)
            continue
        BASE.run_captured(
            [
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
                "all",
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
        if completed.returncode != 0 or not HELPERS.complete_ncu_export(
            report, destination / "ncu.csv", destination / "source.csv"
        ):
            raise RuntimeError(f"source export failed for {cell.key}")
        print(f"source {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_nvbit(output: Path, cells: Sequence[Cell], force: bool) -> None:
    tracer = REPO_ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    base_environment = dict(os.environ)
    protobuf_lib = REPO_ROOT / ".toolchain/pb/lib"
    old_library_path = base_environment.get("LD_LIBRARY_PATH", "")
    base_environment.update(
        {
            "CUDA_INJECTION64_PATH": str(tracer),
            "USER_DEFINED_FOLDERS": "1",
            "ACTIVE_FROM_START": "0",
            "SKIP_ENHANCED_TRACER": "0",
            "LD_LIBRARY_PATH": str(protobuf_lib)
            + ((":" + old_library_path) if old_library_path else ""),
        }
    )
    for index, cell in enumerate(cells, 1):
        destination = output / "nvbit" / cell.key
        trace = destination / "dynamic_trace.pb"
        if complete_nvbit_capture(destination) and not force:
            print(f"nvbit {index:03d}/{len(cells):03d} {cell.key} (existing)", flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        environment = dict(base_environment)
        environment["TRACES_FOLDER"] = str(destination.resolve())
        completed = BASE.run_captured(
            cell.target(trace=True, trials=1),
            destination,
            environment=environment,
            timeout=1200,
        )
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError(f"NVBit did not produce a trace for {cell.key}")
        BASE.write_json(
            destination / "capture.json",
            {
                "cell": {**asdict(cell), "key": cell.key},
                "dynamic_trace_bytes": trace.stat().st_size,
                "stdout_bytes": len(completed.stdout.encode()),
                "stderr_bytes": len(completed.stderr.encode()),
            },
        )
        print(f"nvbit {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    for binary in (LONG_BINARY, TRACE_BINARY):
        for suffix, option in (("sass", "--dump-sass"), ("resources", "--dump-resource-usage")):
            completed = subprocess.run(
                ["/usr/local/cuda-13.2/bin/cuobjdump", option, str(binary)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"{binary.name}.{suffix}.txt").write_text(completed.stdout, encoding="utf-8")
            (destination / f"{binary.name}.{suffix}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"cuobjdump failed for {binary}")
        BASE.run_captured(
            [
                str(REPO_ROOT / ".venv/bin/python"),
                str(REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"),
                "--binary",
                str(binary),
                "--arch",
                "sm_86",
                "--out",
                str(destination / f"{binary.name}.control_bits.json"),
                "--toolkit-bindir",
                "/usr/local/cuda-13.2/bin",
            ],
            destination / f"{binary.name}.control-extraction",
        )
    print(destination)


def run_simulator(output: Path, cells: Sequence[Cell], force: bool, carded: bool) -> None:
    binary = ensure_release_simulator()
    control = output / "sass/memory_retirement_service_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    directory = "simulator-carded-dram" if carded else "simulator"
    for index, cell in enumerate(cells, 1):
        destination = output / directory / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(control),
            "--arch-card",
            str(arch_card),
            "--pchase-card",
            str(pchase_card),
            "--mode",
            "l1-l2",
        ]
        if carded:
            command.append("--carded-dram")
        outputs: list[bytes] = []
        dependencies = [
            binary,
            control,
            arch_card,
            pchase_card,
            output / "nvbit" / cell.key,
        ]
        expectation = REPLAY.ReplayExpectation(
            launches=REPLAY.expected_launches_from_trace(
                output / "nvbit" / cell.key,
                grid=(cell.blocks, 1, 1),
                block=(32, 1, 1),
            ),
            carded_dram=carded,
        )
        for repeat in (1, 2):
            path = destination / f"replay-{repeat}.json"
            if (
                not force
                and REPLAY.reusable_simulator_output(
                    path,
                    dependencies,
                    expectation,
                )
            ):
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
            raise RuntimeError(f"{cell.key}: nondeterministic replay")
        print(f"simulator {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=("manifest", "correctness", "ncu", "source", "nvbit", "static", "simulator", "simulator-carded-dram", "keys"),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=3)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--force", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        if args.action == "keys":
            print(",".join(cell.key for cell in configured_cells()))
            return 0
        emit_manifest(
            output,
            refresh_environment=args.action == "manifest" and args.force,
        )
        cells = selected_cells(args.cells)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
        elif args.action == "correctness":
            run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            run_source(output, args.force)
        elif args.action == "nvbit":
            run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force, False)
        elif args.action == "simulator-carded-dram":
            run_simulator(output, cells, args.force, True)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"memory-retirement campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
