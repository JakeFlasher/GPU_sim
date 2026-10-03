#!/usr/bin/env python3
"""Run the bounded sm_86 async-copy / CTA-barrier experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_async_barrier_readiness_20260817_v4"
)
LONG_BINARY = HERE / "async_barrier_overlap"
TRACE_BINARY = HERE / "async_barrier_overlap_trace"
# Keep correctness, NCU, NVBit, and replay on the same 64-group executable.
# The legacy 16,384-group binary remains available for manual stress runs but
# must not be mixed into a scored trace/NCU comparison.
MATCHED_BINARY = TRACE_BINARY

METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__sectors_read.sum",
)


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    protocol: str
    locality: str
    gap: int

    @property
    def key(self) -> str:
        return f"w{self.warps}_{self.protocol}_{self.locality}_g{self.gap}"

    def target(self, binary: Path, trials: int) -> list[str]:
        return [
            str(binary),
            self.protocol,
            self.locality,
            str(self.gap),
            str(self.warps),
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    cells: list[Cell] = []
    for protocol in ("c00", "c01"):
        for gap in (0, 64):
            cells.append(Cell("identification_controls", 4, protocol, "na", gap))
    for protocol in ("c10", "c11"):
        for locality in ("l1", "l2", "dram"):
            for gap in (0, 64):
                cells.append(
                    Cell("identification_async_factorial", 4, protocol, locality, gap)
                )
    for locality in ("l1", "dram"):
        for gap in (0, 64):
            cells.append(Cell("asymmetric_diagnosis", 4, "c12", locality, gap))
    for warps in (2, 8):
        for protocol in ("c00", "c01"):
            cells.append(Cell("held_out_cta_shapes", warps, protocol, "na", 0))
    for warps in (2, 8):
        for protocol in ("c10", "c11"):
            for locality in ("l1", "dram"):
                cells.append(
                    Cell("held_out_async_extremes", warps, protocol, locality, 0)
                )
    for warps in (2, 8):
        for locality in ("l1", "dram"):
            cells.append(Cell("held_out_asymmetry", warps, "c12", locality, 0))
    if len(cells) != 36 or len({cell.key for cell in cells}) != 36:
        raise RuntimeError("the configured identification matrix must have 36 unique cells")
    return cells


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_captured(
    command: Sequence[str],
    output: Path,
    *,
    environment: dict[str, str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    output.mkdir(parents=True, exist_ok=True)
    started_at = now()
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    finished_at = now()
    (output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (output / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    write_json(
        output / "run.json",
        {
            "command": list(command),
            "return_code": completed.returncode,
            "started_at": started_at,
            "finished_at": finished_at,
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}; see {output / 'stderr.txt'}"
        )
    return completed


def selected_cells(selector: str | None) -> list[Cell]:
    cells = configured_cells()
    if selector is None:
        return cells
    requested = {value.strip() for value in selector.split(",") if value.strip()}
    selected = [cell for cell in cells if cell.key in requested]
    missing = requested - {cell.key for cell in selected}
    if missing:
        raise RuntimeError(f"unknown cell key(s): {', '.join(sorted(missing))}")
    return selected


def emit_manifest(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "metadata" / "matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(cell), "key": cell.key} for cell in configured_cells()],
            "ncu_metrics": list(METRICS),
            "matched_binary": str(MATCHED_BINARY),
            "matched_groups": 64,
            "legacy_long_binary_not_used": True,
        },
    )
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "nvcc": ["/usr/local/cuda-13.2/bin/nvcc", "--version"],
        "ncu": [os.environ.get("NCU", "ncu"), "--version"],
    }
    environment: dict[str, object] = {"captured_at": now()}
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
    write_json(output / "metadata" / "environment.json", environment)


def run_correctness(output: Path, cells: Sequence[Cell], trials: int) -> None:
    for index, cell in enumerate(cells, 1):
        destination = output / "correctness" / cell.key
        completed = run_captured(cell.target(LONG_BINARY, trials), destination)
        lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        if not lines:
            raise RuntimeError(f"{cell.key}: target did not emit a JSON result")
        payload = json.loads(lines[-1])
        if payload.get("correct") is not True:
            raise RuntimeError(f"{cell.key}: correctness check failed")
        write_json(
            destination / "result.json",
            {"cell": {**asdict(cell), "key": cell.key}, "measurement": payload},
        )
        print(f"correctness {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def ncu_launch_skip(cell: Cell) -> int:
    del cell
    # Profiling starts at the explicit cudaProfilerStart() immediately before
    # the target launch, so the prewarm kernel is outside the captured range.
    return 0


def export_ncu(ncu: str, report: Path, destination: Path) -> None:
    raw = subprocess.run(
        [
            ncu,
            "--import",
            str(report),
            "--csv",
            "--page",
            "raw",
            "--log-file",
            str(destination / "ncu.csv"),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (destination / "export.stdout.txt").write_text(raw.stdout, encoding="utf-8")
    (destination / "export.stderr.txt").write_text(raw.stderr, encoding="utf-8")
    if raw.returncode != 0 or not (destination / "ncu.csv").is_file():
        raise RuntimeError(f"NCU export failed for {report}")


def run_ncu(
    output: Path,
    cells: Sequence[Cell],
    *,
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
                print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)")
                continue
            command = [
                ncu,
                "--force-overwrite",
                "--export",
                str(report_base),
                "--profile-from-start",
                "off",
                "--launch-skip",
                str(ncu_launch_skip(cell)),
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
                *cell.target(LONG_BINARY, 1),
            ]
            run_captured(command, destination, timeout=900)
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            export_ncu(ncu, report, destination)
            print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat}", flush=True)


def source_cells(cells: Sequence[Cell]) -> list[Cell]:
    return [
        cell
        for cell in cells
        if cell.warps == 4
        and cell.gap == 0
        and (
            (cell.protocol == "c01" and cell.locality == "na")
            or (
                cell.protocol in {"c10", "c11", "c12"}
                and cell.locality in {"l1", "dram"}
            )
        )
    ]


def run_source_counters(output: Path, cells: Sequence[Cell], force: bool) -> None:
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
            "--launch-skip",
            str(ncu_launch_skip(cell)),
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
            *cell.target(MATCHED_BINARY, 1),
        ]
        run_captured(command, destination, timeout=1200)
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        export_ncu(ncu, report, destination)
        source = subprocess.run(
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
            source.stdout, encoding="utf-8"
        )
        (destination / "source-export.stderr.txt").write_text(
            source.stderr, encoding="utf-8"
        )
        if source.returncode != 0 or not (destination / "source.csv").is_file():
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
            "ACTIVE_FROM_START": "0",
            "SKIP_ENHANCED_TRACER": "0",
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
        completed = run_captured(
            cell.target(TRACE_BINARY, 1),
            destination,
            environment=environment,
            timeout=900,
        )
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError(f"NVBit did not produce a nonempty trace for {cell.key}")
        write_json(
            destination / "capture.json",
            {
                "cell": {**asdict(cell), "key": cell.key},
                "environment": {
                    key: environment[key]
                    for key in (
                        "CUDA_INJECTION64_PATH",
                        "USER_DEFINED_FOLDERS",
                        "ACTIVE_FROM_START",
                        "SKIP_ENHANCED_TRACER",
                        "TRACES_FOLDER",
                    )
                },
                "stdout_bytes": len(completed.stdout.encode()),
                "stderr_bytes": len(completed.stderr.encode()),
                "dynamic_trace_bytes": trace.stat().st_size,
            },
        )
        print(f"nvbit {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    cuobjdump = "/usr/local/cuda-13.2/bin/cuobjdump"
    for binary in (LONG_BINARY, TRACE_BINARY):
        name = binary.name
        for suffix, option in (("sass", "--dump-sass"), ("resources", "--dump-resource-usage")):
            completed = subprocess.run(
                [cuobjdump, option, str(binary)],
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
        command = [
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
        ]
        run_captured(command, destination / f"{name}.control-extraction")
    print(f"static artifacts: {destination}")


def run_simulator(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = REPO_ROOT / "sim" / "target" / "release" / "sim-run"
    control = output / "sass" / "async_barrier_overlap_trace.control_bits.json"
    if not binary.is_file() or not control.is_file():
        raise RuntimeError("build sim-run --release and run the static stage first")
    for index, cell in enumerate(cells, 1):
        destination = output / "simulator" / cell.key
        first_path = destination / "replay-1.json"
        second_path = destination / "replay-2.json"
        if first_path.is_file() and second_path.is_file() and not force:
            if first_path.read_bytes() != second_path.read_bytes():
                raise RuntimeError(f"existing simulator outputs differ for {cell.key}")
            print(f"simulator {index:02d}/{len(cells):02d} {cell.key} (existing)")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(control),
            "--arch-card",
            str(REPO_ROOT / "arch_profiles" / "cards" / "sm_86.json"),
            "--pchase-card",
            str(REPO_ROOT / "data" / "pchase" / "sm_86_static_card.json"),
            "--mode",
            "l1-l2",
            "--kernel-id",
            "1",
        ]
        outputs: list[bytes] = []
        for repeat in (1, 2):
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"simulator failed for {cell.key} replay {repeat} with "
                    f"status {completed.returncode}"
                )
            (destination / f"replay-{repeat}.json").write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"simulator outputs are not byte-identical for {cell.key}")
        report = json.loads(outputs[0])
        launches = report.get("launches", [])
        if len(launches) != 1:
            raise RuntimeError(f"{cell.key}: expected exactly one replayed launch")
        launch = launches[0]
        counters = launch["counters"]
        if counters["instructions_issued"] != counters["records_seen"]:
            raise RuntimeError(f"{cell.key}: replay instruction cardinality mismatch")
        if launch["coverage"].get("kind") != "full":
            raise RuntimeError(f"{cell.key}: replay does not have full block coverage")
        readiness = launch.get("timing", {}).get("readiness", {})
        # The simulator's readiness schema is versioned and has advanced past
        # the probe's original v1 check.  Require a present positive version,
        # not one stale literal, while retaining the recorded version in the
        # replay summary below.
        readiness_version = readiness.get("schema_version")
        if not isinstance(readiness_version, int) or readiness_version < 1:
            raise RuntimeError(f"{cell.key}: readiness schema is missing or invalid")
        write_json(
            destination / "replay.json",
            {
                "cell": {**asdict(cell), "key": cell.key},
                "command": command,
                "byte_identical": True,
                "instructions_issued": counters["instructions_issued"],
                "records_seen": counters["records_seen"],
                "coverage": launch["coverage"],
                "readiness_schema_version": readiness["schema_version"],
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
        ),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells", help="comma-separated exact cell keys")
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
        # The helpers below refer to module globals; point both names to the
        # exact bounded binary before any action so long/trace generations
        # cannot silently diverge.
        globals()["LONG_BINARY"] = MATCHED_BINARY
        globals()["TRACE_BINARY"] = MATCHED_BINARY
        emit_manifest(output)
        if args.action == "manifest":
            print(output / "metadata" / "matrix.json")
        elif args.action == "correctness":
            run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            run_ncu(
                output,
                cells,
                repeat_start=args.repeat_start,
                repeats=args.repeats,
                force=args.force,
            )
        elif args.action == "source":
            run_source_counters(output, cells, args.force)
        elif args.action == "nvbit":
            run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
