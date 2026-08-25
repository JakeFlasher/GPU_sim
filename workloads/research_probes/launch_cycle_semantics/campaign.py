#!/usr/bin/env python3
"""Run the sm_86 launch-cycle, completion, and range-composition campaign."""

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
    REPO_ROOT / "results/captures/sm86_launch_cycle_semantics_20260818_v1"
)
LONG_BINARY = HERE / "launch_cycle_semantics"
TRACE_BINARY = HERE / "launch_cycle_semantics_trace"
TAILS = ("control", "load_discard", "load_use", "store", "red", "atom_use")
CHAIN_TAILS = ("control", "load_discard", "red", "atom_use")

KERNEL_METRICS = (
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
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
)

RANGE_METRICS = (
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
)


def load_helpers():
    path = HERE.parent / "async_barrier_overlap/campaign.py"
    spec = importlib.util.spec_from_file_location("launch_cycle_campaign_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load campaign helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = load_helpers()


def load_replay_resume():
    path = HERE.parent / "replay_resume.py"
    spec = importlib.util.spec_from_file_location("launch_cycle_replay_resume", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load replay helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPLAY = load_replay_resume()


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


def complete_ncu_export(report: Path, export: Path) -> bool:
    return (
        report.is_file()
        and report.stat().st_size > 0
        and complete_csv(export)
        and export.stat().st_mtime_ns >= report.stat().st_mtime_ns
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
            json.loads((destination / "extra_info/allocations.json").read_text(encoding="utf-8")),
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
    HELPERS.write_json(path, value)


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    tail: str
    blocks: int
    iterations: int
    chain_length: int

    @property
    def key(self) -> str:
        return (
            f"b{self.blocks}_{self.tail}_i{self.iterations}_c{self.chain_length}"
        )

    def target(
        self,
        *,
        trace: bool,
        trials: int,
        sync_placement: str | None = None,
    ) -> list[str]:
        command = [
            str(TRACE_BINARY if trace else LONG_BINARY),
            self.tail,
            str(self.blocks),
            str(self.iterations),
            str(self.chain_length),
            str(trials),
        ]
        if sync_placement is not None:
            if sync_placement not in {"SYNC_INSIDE_RANGE", "SYNC_OUTSIDE_RANGE"}:
                raise RuntimeError(f"invalid sync placement {sync_placement}")
            command.append(sync_placement)
        return command


def configured_cells() -> list[Cell]:
    cells: list[Cell] = []
    for tail in TAILS:
        for blocks in (1, 7, 42, 84):
            for iterations in (0, 256):
                cells.append(
                    Cell("single_launch_tail_factorial", tail, blocks, iterations, 1)
                )
    for blocks in (1, 84):
        for iterations in (16, 64, 1024, 4096):
            cells.append(
                Cell("single_launch_work_scaling", "control", blocks, iterations, 1)
            )
    for tail in CHAIN_TAILS:
        for blocks in (1, 84):
            for chain_length in (2, 4, 8):
                cells.append(
                    Cell(
                        "same_stream_chain_composition",
                        tail,
                        blocks,
                        256,
                        chain_length,
                    )
                )
    if len(cells) != 80 or len({cell.key for cell in cells}) != 80:
        raise RuntimeError("launch-cycle matrix must contain 80 unique cells")
    return cells


def range_cells() -> list[Cell]:
    by_key = {cell.key: cell for cell in configured_cells()}
    cells = []
    for tail in CHAIN_TAILS:
        for blocks in (1, 84):
            for chain_length in (1, 2, 4, 8):
                key = f"b{blocks}_{tail}_i256_c{chain_length}"
                cells.append(by_key[key])
    if len(cells) != 32:
        raise RuntimeError("range matrix must contain 32 cells")
    return cells


def app_range_cells() -> list[Cell]:
    return [cell for cell in range_cells() if cell.tail in {"control", "red"}]


def selected_cells(selector: str | None, *, only_range: bool = False) -> list[Cell]:
    cells = range_cells() if only_range else configured_cells()
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
            "range_cells": [cell.key for cell in range_cells()],
            "application_range_sentinels": [cell.key for cell in app_range_cells()],
            "kernel_metrics": list(KERNEL_METRICS),
            "range_metrics": list(RANGE_METRICS),
            "ncu_repeats": 3,
            "correctness_trials": 7,
            "tail_stride_bytes": 128,
            "sm_count": 84,
            "hypothesized_gpc_count_from_prior_instance_rollups": 7,
            "invariants": [
                "one warp per CTA",
                "runtime dependent IMAD chain retained by a volatile shared store",
                "CTA barrier immediately precedes the compile-time tail",
                "tail addresses occupy distinct aligned 128-byte lines",
                "all chain launches use CUDA stream 0 without an intervening host sync",
                "NCU application replay reports kernels; Range Replay reports the complete profiler range",
            ],
        }
    if refresh_environment:
        HELPERS.write_json(output / "metadata/matrix.json", matrix)
    else:
        preserve_json_snapshot(output / "metadata/matrix.json", matrix)
    environment_path = output / "metadata/environment.json"
    if environment_path.is_file() and not refresh_environment:
        return
    environment: dict[str, object] = {
        "observed_at": HELPERS.now(),
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
    HELPERS.write_json(environment_path, environment)


def measurement(completed: subprocess.CompletedProcess[str], cell: Cell) -> dict:
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"{cell.key}: target emitted no JSON result")
    payload = json.loads(lines[-1])
    if payload.get("correct") is not True:
        raise RuntimeError(f"{cell.key}: correctness failed")
    return payload


def run_correctness(output: Path, cells: Sequence[Cell], trials: int) -> None:
    for index, cell in enumerate(cells, 1):
        destination = output / "correctness" / cell.key
        completed = HELPERS.run_captured(
            cell.target(trace=False, trials=trials), destination, timeout=180
        )
        HELPERS.write_json(
            destination / "result.json",
            {"cell": {**asdict(cell), "key": cell.key}, "measurement": measurement(completed, cell)},
        )
        print(f"correctness {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def export_ncu(ncu: str, report: Path, destination: Path) -> None:
    HELPERS.export_ncu(ncu, report, destination)


def run_ncu_kernel(
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
            destination = output / "ncu-kernel" / cell.key / f"repeat-{repeat}"
            report_base = destination / "ncu-report"
            report = destination / "ncu-report.ncu-rep"
            if complete_ncu_export(report, destination / "ncu.csv") and not force:
                print(
                    f"ncu-kernel {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)",
                    flush=True,
                )
                continue
            command = [
                ncu,
                "--force-overwrite",
                "--export",
                str(report_base),
                "--profile-from-start",
                "off",
                "--launch-count",
                str(cell.chain_length),
                "--replay-mode",
                "application",
                "--app-replay-mode",
                "strict",
                "--clock-control",
                "base",
                "--cache-control",
                "all",
                "--metrics",
                ",".join(KERNEL_METRICS),
                *cell.target(trace=False, trials=1),
            ]
            HELPERS.run_captured(command, destination, timeout=1200)
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            export_ncu(ncu, report, destination)
            if not complete_ncu_export(report, destination / "ncu.csv"):
                raise RuntimeError(f"NCU export is incomplete for {cell.key} repeat {repeat}")
            print(
                f"ncu-kernel {progress:03d}/{total:03d} {cell.key} r{repeat}",
                flush=True,
            )


def run_ncu_range(
    output: Path,
    cells: Sequence[Cell],
    repeat_start: int,
    repeats: int,
    force: bool,
    *,
    replay_mode: str = "range",
    sync_placement: str = "SYNC_INSIDE_RANGE",
) -> None:
    if replay_mode not in {"range", "app-range"}:
        raise RuntimeError(f"unsupported range replay mode {replay_mode}")
    ncu = os.environ.get("NCU", "ncu")
    base_directory = "ncu-range" if replay_mode == "range" else "ncu-app-range"
    directory = (
        base_directory
        if sync_placement == "SYNC_INSIDE_RANGE"
        else f"{base_directory}-outside-sync"
    )
    total = len(cells) * repeats
    progress = 0
    for cell in cells:
        for repeat in range(repeat_start, repeat_start + repeats):
            progress += 1
            destination = output / directory / cell.key / f"repeat-{repeat}"
            report_base = destination / "ncu-report"
            report = destination / "ncu-report.ncu-rep"
            if complete_ncu_export(report, destination / "ncu.csv") and not force:
                print(
                    f"ncu-{replay_mode} {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)",
                    flush=True,
                )
                continue
            # --profile-from-start is intentionally absent: Range Replay uses
            # the explicit cudaProfilerStart/Stop range and rejects that flag.
            command = [
                ncu,
                "--force-overwrite",
                "--export",
                str(report_base),
                "--launch-count",
                "1",
                "--replay-mode",
                replay_mode,
                "--clock-control",
                "base",
                "--cache-control",
                "all",
                "--metrics",
                ",".join(RANGE_METRICS),
                *cell.target(
                    trace=False, trials=1, sync_placement=sync_placement
                ),
            ]
            HELPERS.run_captured(command, destination, timeout=1200)
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            export_ncu(ncu, report, destination)
            if not complete_ncu_export(report, destination / "ncu.csv"):
                raise RuntimeError(f"NCU export is incomplete for {cell.key} repeat {repeat}")
            print(
                f"ncu-{replay_mode} {progress:03d}/{total:03d} {cell.key} r{repeat}",
                flush=True,
            )


def run_nvbit(output: Path, cells: Sequence[Cell], force: bool) -> None:
    tracer = REPO_ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    if not tracer.is_file():
        raise RuntimeError(f"NVBit tracer is missing: {tracer}")
    base_environment = dict(os.environ)
    protobuf_lib = REPO_ROOT / ".toolchain/pb/lib"
    existing_library_path = base_environment.get("LD_LIBRARY_PATH", "")
    base_environment.update(
        {
            "CUDA_INJECTION64_PATH": str(tracer),
            "USER_DEFINED_FOLDERS": "1",
            "ACTIVE_FROM_START": "0",
            "SKIP_ENHANCED_TRACER": "0",
            "LD_LIBRARY_PATH": str(protobuf_lib)
            + ((":" + existing_library_path) if existing_library_path else ""),
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
        completed = HELPERS.run_captured(
            cell.target(trace=True, trials=1),
            destination,
            environment=environment,
            timeout=1200,
        )
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError(f"NVBit did not produce a nonempty trace for {cell.key}")
        HELPERS.write_json(
            destination / "capture.json",
            {
                "cell": {**asdict(cell), "key": cell.key},
                "stdout_bytes": len(completed.stdout.encode()),
                "stderr_bytes": len(completed.stderr.encode()),
                "dynamic_trace_bytes": trace.stat().st_size,
            },
        )
        print(f"nvbit {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    cuobjdump = "/usr/local/cuda-13.2/bin/cuobjdump"
    for binary in (LONG_BINARY, TRACE_BINARY):
        for suffix, option in (
            ("sass", "--dump-sass"),
            ("resources", "--dump-resource-usage"),
        ):
            completed = subprocess.run(
                [cuobjdump, option, str(binary)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"{binary.name}.{suffix}.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (destination / f"{binary.name}.{suffix}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(f"cuobjdump failed for {binary}")
        sidecar = destination / f"{binary.name}.control_bits.json"
        HELPERS.run_captured(
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
            destination / f"{binary.name}.control-extraction",
        )
    print(destination)


def run_simulator(
    output: Path,
    cells: Sequence[Cell],
    force: bool,
    *,
    carded_dram: bool = False,
) -> None:
    binary = ensure_release_simulator()
    control = output / "sass/launch_cycle_semantics_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    if not control.is_file():
        raise RuntimeError("trace control sidecar is missing")
    for index, cell in enumerate(cells, 1):
        directory = "simulator-carded-dram" if carded_dram else "simulator"
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
            "--cache-scope",
            "stream-l2",
        ]
        if carded_dram:
            command.append("--carded-dram")
        outputs: list[bytes] = []
        dependencies = [
            binary,
            control,
            arch_card,
            pchase_card,
            output / "nvbit" / cell.key,
        ]
        expected_launches = REPLAY.expected_launches_from_trace(
            output / "nvbit" / cell.key,
            grid=(cell.blocks, 1, 1),
            block=(32, 1, 1),
        )
        if len(expected_launches) != cell.chain_length:
            raise RuntimeError(
                f"{cell.key}: trace has {len(expected_launches)} launches, "
                f"expected {cell.chain_length}"
            )
        expectation = REPLAY.ReplayExpectation(
            launches=expected_launches,
            carded_dram=carded_dram,
            cache_scope="stream-l2",
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
                raise RuntimeError(f"{cell.key}: simulator replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: simulator replay is nondeterministic")
        print(f"simulator {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=(
            "manifest",
            "correctness",
            "ncu-kernel",
            "ncu-range",
            "ncu-app-range",
            "ncu-range-outside-sync",
            "ncu-app-range-outside-sync",
            "nvbit",
            "static",
            "simulator",
            "simulator-carded-dram",
            "keys",
            "range-keys",
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
        if args.action == "keys":
            print(",".join(cell.key for cell in configured_cells()))
            return 0
        if args.action == "range-keys":
            print(",".join(cell.key for cell in range_cells()))
            return 0
        emit_manifest(
            output,
            refresh_environment=args.action == "manifest" and args.force,
        )
        if args.action in {"ncu-app-range", "ncu-range-outside-sync", "ncu-app-range-outside-sync"}:
            available = app_range_cells()
            if args.cells is None:
                cells = available
            else:
                requested = {item.strip() for item in args.cells.split(",") if item.strip()}
                cells = [cell for cell in available if cell.key in requested]
                missing = requested - {cell.key for cell in cells}
                if missing:
                    raise RuntimeError(
                        f"unknown application-range cell key(s): {', '.join(sorted(missing))}"
                    )
        else:
            cells = selected_cells(args.cells, only_range=args.action == "ncu-range")
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
        elif args.action == "correctness":
            run_correctness(output, cells, args.trials)
        elif args.action == "ncu-kernel":
            run_ncu_kernel(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "ncu-range":
            run_ncu_range(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "ncu-app-range":
            run_ncu_range(
                output,
                cells,
                args.repeat_start,
                args.repeats,
                args.force,
                replay_mode="app-range",
            )
        elif args.action == "ncu-range-outside-sync":
            run_ncu_range(
                output,
                cells,
                args.repeat_start,
                args.repeats,
                args.force,
                sync_placement="SYNC_OUTSIDE_RANGE",
            )
        elif args.action == "ncu-app-range-outside-sync":
            run_ncu_range(
                output,
                cells,
                args.repeat_start,
                args.repeats,
                args.force,
                replay_mode="app-range",
                sync_placement="SYNC_OUTSIDE_RANGE",
            )
        elif args.action == "nvbit":
            run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
        elif args.action == "simulator-carded-dram":
            run_simulator(output, cells, args.force, carded_dram=True)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"launch-cycle campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
