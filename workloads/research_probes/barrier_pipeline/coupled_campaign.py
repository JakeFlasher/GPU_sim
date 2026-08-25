#!/usr/bin/env python3
"""Run coupled async/barrier validation with traced prewarm launch chains."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ASYNC_DIR = REPO_ROOT / "workloads" / "research_probes" / "async_barrier_overlap"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_barrier_pipeline_coupled_20260817_v1"
)


def load_async_campaign():
    path = ASYNC_DIR / "campaign.py"
    spec = importlib.util.spec_from_file_location("async_barrier_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ASYNC = load_async_campaign()


def configured_cells() -> list[object]:
    cells = [
        ASYNC.Cell("coupled_barrier_cache", warps, protocol, locality, 0)
        for warps in (2, 4, 8)
        for protocol in ("c10", "c11")
        for locality in ("l1", "l2", "dram")
    ]
    if len(cells) != 18 or len({cell.key for cell in cells}) != 18:
        raise RuntimeError("coupled matrix must contain 18 unique cells")
    return cells


def selected_cells(selector: str | None) -> list[object]:
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
    ASYNC.write_json(
        output / "metadata" / "matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(cell), "key": cell.key} for cell in configured_cells()],
            "ncu_metrics": list(ASYNC.METRICS),
            "long_groups": 16384,
            "trace_groups": 64,
            "nvbit_capture_scope": "active_from_process_start",
            "simulator_scopes": ["launch", "stream-l2"],
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
    environment: dict[str, object] = {"captured_at": ASYNC.now()}
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
    ASYNC.write_json(output / "metadata" / "environment.json", environment)


def run_nvbit(output: Path, cells: Sequence[object], force: bool) -> None:
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
        completed = ASYNC.run_captured(
            cell.target(ASYNC.TRACE_BINARY, 1),
            destination,
            environment=environment,
            timeout=900,
        )
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError(f"NVBit did not produce a nonempty trace for {cell.key}")
        expected_launches = 2 if cell.locality in {"l1", "l2"} else 1
        modules = json.loads(
            (destination / "extra_info" / "traced_kernel_modules.json").read_text(
                encoding="utf-8"
            )
        )
        accepted = modules["selection"]["accepted_active_launches"]
        if accepted != expected_launches:
            raise RuntimeError(
                f"{cell.key}: expected {expected_launches} accepted launches, got {accepted}"
            )
        ASYNC.write_json(
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
                "expected_launches": expected_launches,
                "accepted_launches": accepted,
                "stdout_bytes": len(completed.stdout.encode()),
                "stderr_bytes": len(completed.stderr.encode()),
                "dynamic_trace_bytes": trace.stat().st_size,
            },
        )
        print(f"nvbit {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_simulator(
    output: Path,
    cells: Sequence[object],
    force: bool,
    directory_name: str = "simulator",
) -> None:
    binary = REPO_ROOT / "sim" / "target" / "release" / "sim-run"
    control = output / "sass" / "async_barrier_overlap_trace.control_bits.json"
    if not binary.is_file() or not control.is_file():
        raise RuntimeError("build sim-run --release and run static first")
    for index, cell in enumerate(cells, 1):
        destination = output / directory_name / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        base_command = [
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
        ]
        metadata: dict[str, object] = {"cell": {**asdict(cell), "key": cell.key}}
        for scope in ("launch", "stream-l2"):
            first = destination / f"{scope}-1.json"
            second = destination / f"{scope}-2.json"
            if first.is_file() and second.is_file() and not force:
                if first.read_bytes() != second.read_bytes():
                    raise RuntimeError(f"{cell.key}: existing {scope} outputs differ")
            else:
                command = [*base_command, "--cache-scope", scope]
                outputs: list[bytes] = []
                for repeat in (1, 2):
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    (destination / f"{scope}-{repeat}.stderr.txt").write_bytes(
                        completed.stderr
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"{cell.key}: {scope} replay {repeat} failed with {completed.returncode}"
                        )
                    (destination / f"{scope}-{repeat}.json").write_bytes(completed.stdout)
                    outputs.append(completed.stdout)
                if outputs[0] != outputs[1]:
                    raise RuntimeError(f"{cell.key}: {scope} replay is nondeterministic")
            report = json.loads(first.read_bytes())
            launches = report["launches"]
            expected_launches = 2 if cell.locality in {"l1", "l2"} else 1
            if len(launches) != expected_launches:
                raise RuntimeError(
                    f"{cell.key}: {scope} expected {expected_launches} launches, got {len(launches)}"
                )
            for launch in launches:
                counters = launch["counters"]
                if counters["instructions_issued"] != counters["records_seen"]:
                    raise RuntimeError(f"{cell.key}: {scope} replay count mismatch")
                if launch["coverage"]["kind"] != "full":
                    raise RuntimeError(f"{cell.key}: {scope} replay coverage is not full")
            if scope == "launch":
                if "cache_session" in report:
                    raise RuntimeError(f"{cell.key}: cold scope emitted a cache session")
            else:
                session = report.get("cache_session")
                if session is None or session["completed_launches"] != expected_launches:
                    raise RuntimeError(f"{cell.key}: stream-L2 session metadata mismatch")
            metadata[scope] = {
                "byte_identical": True,
                "launches": len(launches),
                "target_kernel_id": launches[-1]["kernel_id"],
                "target_cycles": launches[-1]["timing"]["diagnostic_last_issue_cycles"],
                "target_l2": launches[-1]["counters"]["l2"],
                "target_dram": launches[-1]["counters"]["dram"],
                "target_readiness_schema": launches[-1]["timing"]["readiness"][
                    "schema_version"
                ],
            }
        ASYNC.write_json(destination / "replay.json", metadata)
        print(f"simulator {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_production(output: Path, force: bool) -> None:
    binary = REPO_ROOT / "sim" / "target" / "release" / "sim-run"
    source = REPO_ROOT / "results" / "captures" / "sm86_cycle_residual_20260817_v1"
    names = ("canary", "bf16", "fp16", "int8", "sparse-bf16", "attention")
    destination = output / "production"
    destination.mkdir(parents=True, exist_ok=True)
    predicted: dict[str, int] = {}
    for index, name in enumerate(names, 1):
        first = destination / f"{name}-1.json"
        second = destination / f"{name}-2.json"
        command = [
            str(binary),
            "--trace",
            str(source / "nvbit" / name),
            "--control",
            str(source / "control" / f"{name}.control_bits.json"),
            "--arch-card",
            str(REPO_ROOT / "arch_profiles" / "cards" / "sm_86.json"),
            "--pchase-card",
            str(REPO_ROOT / "data" / "pchase" / "sm_86_static_card.json"),
            "--mode",
            "l1-l2",
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
                (destination / f"{name}-{repeat}.stderr.txt").write_bytes(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"production {name} replay {repeat} failed with {completed.returncode}"
                    )
                (destination / f"{name}-{repeat}.json").write_bytes(completed.stdout)
                outputs.append(completed.stdout)
            if outputs[0] != outputs[1]:
                raise RuntimeError(f"production {name} replay is nondeterministic")
        if first.read_bytes() != second.read_bytes():
            raise RuntimeError(f"production {name} saved replays differ")
        report = json.loads(first.read_bytes())
        if len(report["launches"]) != 1:
            raise RuntimeError(f"production {name} expected one launch")
        launch = report["launches"][0]
        counters = launch["counters"]
        if counters["instructions_issued"] != counters["records_seen"]:
            raise RuntimeError(f"production {name} replay count mismatch")
        if launch["coverage"]["kind"] != "full":
            raise RuntimeError(f"production {name} coverage is not full")
        predicted[name] = launch["timing"]["diagnostic_last_issue_cycles"]
        print(f"production {index:02d}/06 {name}", flush=True)
    ASYNC.write_json(destination / "predicted.json", predicted)


def run_predecessor(output: Path, force: bool) -> None:
    binary = REPO_ROOT / "sim" / "target" / "release" / "sim-run"
    source = (
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_async_barrier_readiness_20260817_v4"
    )
    control = source / "sass" / "async_barrier_overlap_trace.control_bits.json"
    cells = json.loads((source / "metadata" / "matrix.json").read_text(encoding="utf-8"))[
        "cells"
    ]
    if len(cells) != 36:
        raise RuntimeError("predecessor matrix must contain 36 cells")
    destination = output / "predecessor"
    destination.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells, 1):
        key = cell["key"]
        first = destination / f"{key}-1.json"
        second = destination / f"{key}-2.json"
        command = [
            str(binary),
            "--trace",
            str(source / "nvbit" / key),
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
                (destination / f"{key}-{repeat}.stderr.txt").write_bytes(
                    completed.stderr
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"predecessor {key} replay {repeat} failed with {completed.returncode}"
                    )
                (destination / f"{key}-{repeat}.json").write_bytes(completed.stdout)
                outputs.append(completed.stdout)
            if outputs[0] != outputs[1]:
                raise RuntimeError(f"predecessor {key} replay is nondeterministic")
        if first.read_bytes() != second.read_bytes():
            raise RuntimeError(f"predecessor {key} saved replays differ")
        launch = json.loads(first.read_bytes())["launches"][0]
        if launch["counters"]["instructions_issued"] != launch["counters"]["records_seen"]:
            raise RuntimeError(f"predecessor {key} replay count mismatch")
        if launch["coverage"]["kind"] != "full":
            raise RuntimeError(f"predecessor {key} coverage is not full")
        print(f"predecessor {index:02d}/36 {key}", flush=True)


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
            "production",
            "predecessor",
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
            ASYNC.run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            ASYNC.run_ncu(
                output,
                cells,
                repeat_start=args.repeat_start,
                repeats=args.repeats,
                force=args.force,
            )
        elif args.action == "source":
            ASYNC.run_source_counters(output, cells, args.force)
        elif args.action == "nvbit":
            run_nvbit(output, cells, args.force)
        elif args.action == "static":
            ASYNC.run_static(output)
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
        elif args.action == "production":
            run_production(output, args.force)
        elif args.action == "predecessor":
            run_predecessor(output, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"coupled campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
