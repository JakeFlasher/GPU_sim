#!/usr/bin/env python3
"""Run the sm_86 L1TEX arrival/output-replay service campaign."""

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
    REPO_ROOT / "results/captures/sm86_l1tex_arrival_replay_service_20260818_v1"
)
LONG_BINARY = HERE / "l1tex_arrival_replay_service"
TRACE_BINARY = HERE / "l1tex_arrival_replay_service_trace"
CANDIDATE_FILE = HERE / "memory_service_candidates.json"
TAILS = ("control", "load", "store", "red")

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
    "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_red.sum",
    "l1tex__lsuin_requests.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_requests_srcunit_tex_op_red.sum",
    "lts__t_sectors_op_red.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
)

L2_REQUEST_METRICS = (
    "lts__t_requests_srcunit_tex_op_read.sum",
    "lts__t_requests_srcunit_tex_op_write.sum",
    "lts__t_requests_srcunit_tex_op_red.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
    "lts__t_sectors_op_red.sum",
)


def load_helpers():
    path = HERE.parent / "memory_retirement_service/campaign.py"
    spec = importlib.util.spec_from_file_location("l1tex_campaign_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load campaign helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = load_helpers()
BASE = HELPERS.BASE
REPLAY = HELPERS.REPLAY


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    tail: str
    layout: str
    blocks: int
    warps: int
    depth: int
    active_lanes: int
    gap_steps: int

    @property
    def key(self) -> str:
        return (
            f"b{self.blocks}_w{self.warps}_{self.tail}_{self.layout}_"
            f"d{self.depth}_l{self.active_lanes}_g{self.gap_steps}"
        )

    @property
    def ideal_wavefronts(self) -> int:
        per_instruction = (
            (self.active_lanes + 3) // 4 if self.layout == "line" else 1
        )
        return self.blocks * self.warps * self.depth * per_instruction

    @property
    def ideal_wavefronts_per_nominal_sm(self) -> int:
        if self.blocks % 84 != 0:
            raise RuntimeError(f"{self.key}: grid is not an 84-SM multiple")
        return self.ideal_wavefronts // 84

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(TRACE_BINARY if trace else LONG_BINARY),
            self.tail,
            self.layout,
            str(self.blocks),
            str(self.warps),
            str(self.depth),
            str(self.active_lanes),
            str(self.gap_steps),
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    cells: dict[str, Cell] = {}

    def add(cell: Cell) -> None:
        cells.setdefault(cell.key, cell)

    for tail in TAILS:
        for warps, depth in ((1, 64), (2, 32), (4, 16), (8, 8)):
            add(Cell("fixed_ideal_concurrency", tail, "line", 84, warps, depth, 32, 0))

    for tail in TAILS:
        for depth in (2, 4, 8, 16, 32):
            add(Cell("w4_work_scaling", tail, "line", 84, 4, depth, 32, 0))

    for tail in ("control", "store"):
        for depth in (8, 16, 32, 64, 128):
            add(Cell("w1_work_scaling", tail, "line", 84, 1, depth, 32, 0))

    for tail in TAILS:
        add(Cell("cta_shape_transfer", tail, "line", 168, 4, 8, 32, 0))
        add(Cell("cta_shape_transfer", tail, "line", 336, 2, 8, 32, 0))

    for tail in ("control", "load", "store"):
        for gap in (16, 64, 256):
            add(Cell("arrival_skew_w8", tail, "line", 84, 8, 8, 32, gap))
        for gap in (64, 256):
            add(Cell("arrival_skew_w4", tail, "line", 84, 4, 16, 32, gap))

    for tail in ("control", "load", "store"):
        for active_lanes, depth in ((4, 64), (8, 32), (16, 16), (32, 8)):
            add(
                Cell(
                    "fixed_ideal_lane_topology",
                    tail,
                    "line",
                    84,
                    8,
                    depth,
                    active_lanes,
                    0,
                )
            )
        add(Cell("fixed_ideal_word_topology", tail, "word", 84, 8, 64, 32, 0))

    material = list(cells.values())
    if len(material) != 75:
        raise RuntimeError(f"L1TEX arrival matrix must contain 75 cells, got {len(material)}")
    return material


def source_cells() -> list[Cell]:
    wanted = {
        "b84_w1_store_line_d64_l32_g0",
        "b84_w2_store_line_d32_l32_g0",
        "b84_w4_store_line_d16_l32_g0",
        "b84_w8_store_line_d8_l32_g0",
        "b84_w1_load_line_d64_l32_g0",
        "b84_w4_load_line_d16_l32_g0",
        "b84_w8_load_line_d8_l32_g0",
        "b168_w4_store_line_d8_l32_g0",
        "b336_w2_store_line_d8_l32_g0",
        "b84_w1_store_line_d128_l32_g0",
        "b84_w8_store_word_d64_l32_g0",
        "b84_w8_store_line_d8_l32_g256",
    }
    cells = [cell for cell in configured_cells() if cell.key in wanted]
    if len(cells) != len(wanted):
        raise RuntimeError("source sentinel set is incomplete")
    return cells


def l2_request_cells() -> list[Cell]:
    return [
        cell
        for cell in configured_cells()
        if (
            cell.stage in {"fixed_ideal_concurrency", "fixed_ideal_lane_topology", "fixed_ideal_word_topology"}
            and cell.tail in {"load", "store", "red"}
        )
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
            "cells": [
                {
                    **asdict(cell),
                    "key": cell.key,
                    "ideal_wavefronts": cell.ideal_wavefronts,
                    "ideal_wavefronts_per_nominal_sm": cell.ideal_wavefronts_per_nominal_sm,
                }
                for cell in cells
            ],
            "source_counter_cells": [cell.key for cell in source_cells()],
            "l2_request_cells": [cell.key for cell in l2_request_cells()],
            "ncu_metrics": list(METRICS),
            "ncu_repeats": 3,
            "correctness_trials": 3,
            "sm_count": 84,
            "memory_service_candidate_file": str(CANDIDATE_FILE.relative_to(REPO_ROOT)),
            "invariants": [
                "one to eight warps per CTA and no local/stack allocation",
                "fixed-work concurrency cells carry 512 ideal wavefronts per nominal SM",
                "CTA shapes 1x8, 2x4, and 4x2 preserve eight warps and 512 ideal wavefronts per nominal SM",
                "arrival skew changes warp-uniform dependent IMAD work and the derived STORE payload; compression sensitivity remains explicit",
                "lane topology cells preserve 512 ideal wavefronts per nominal SM",
                "NCU output wavefronts remain observations; replay consumes only traced addresses",
            ],
    }
    if refresh_environment:
        BASE.write_json(output / "metadata/matrix.json", matrix)
    else:
        HELPERS.preserve_json_snapshot(output / "metadata/matrix.json", matrix)
    REPLAY.candidate_configurations(CANDIDATE_FILE)
    candidate_document = REPLAY.load_json_strict(CANDIDATE_FILE)
    HELPERS.preserve_json_snapshot(
        output / "metadata/memory_service_candidates.json", candidate_document
    )
    environment_path = output / "metadata/environment.json"
    if environment_path.is_file() and not refresh_environment:
        return
    environment: dict[str, object] = {
        "observed_at": BASE.now(),
        "role": "tool/host observation; raw artifact mtimes define capture chronology",
    }
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
    BASE.write_json(environment_path, environment)


def payload(completed: subprocess.CompletedProcess[str], cell: Cell) -> dict:
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"{cell.key}: target emitted no JSON")
    value = json.loads(lines[-1])
    if value.get("correct") is not True or value.get("ideal_wavefronts") != cell.ideal_wavefronts:
        raise RuntimeError(f"{cell.key}: correctness/work invariant failed")
    return value


def run_correctness(output: Path, cells: Sequence[Cell], trials: int) -> None:
    for index, cell in enumerate(cells, 1):
        destination = output / "correctness" / cell.key
        completed = BASE.run_captured(cell.target(trace=False, trials=trials), destination, timeout=300)
        BASE.write_json(
            destination / "result.json",
            {"cell": {**asdict(cell), "key": cell.key}, "measurement": payload(completed, cell)},
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
        exported = subprocess.run(
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
        (destination / "source-export.stdout.txt").write_text(exported.stdout, encoding="utf-8")
        (destination / "source-export.stderr.txt").write_text(exported.stderr, encoding="utf-8")
        if exported.returncode != 0 or not HELPERS.complete_ncu_export(
            report, destination / "ncu.csv", destination / "source.csv"
        ):
            raise RuntimeError(f"source export failed for {cell.key}")
        print(f"source {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_l2_requests(output: Path, force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    cells = l2_request_cells()
    for index, cell in enumerate(cells, 1):
        destination = output / "ncu-l2-requests" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if HELPERS.complete_ncu_export(report, destination / "ncu.csv") and not force:
            print(f"l2-requests {index:02d}/{len(cells):02d} {cell.key} (existing)", flush=True)
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
                ",".join(L2_REQUEST_METRICS),
                *cell.target(trace=False, trials=1),
            ],
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        if not HELPERS.complete_ncu_export(report, destination / "ncu.csv"):
            raise RuntimeError(f"L2-request export is incomplete for {cell.key}")
        print(f"l2-requests {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_nvbit(output: Path, cells: Sequence[Cell], force: bool) -> None:
    tracer = REPO_ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    environment_base = dict(os.environ)
    protobuf_lib = REPO_ROOT / ".toolchain/pb/lib"
    old_library_path = environment_base.get("LD_LIBRARY_PATH", "")
    environment_base.update(
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
        if HELPERS.complete_nvbit_capture(destination) and not force:
            print(f"nvbit {index:03d}/{len(cells):03d} {cell.key} (existing)", flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        environment = dict(environment_base)
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
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_arrival_replay_service_trace.control_bits.json"
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
                block=(cell.warps * 32, 1, 1),
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


def run_memory_service_candidates(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_arrival_replay_service_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    if not CANDIDATE_FILE.is_file():
        raise RuntimeError(f"memory-service candidate file missing: {CANDIDATE_FILE}")
    candidates = REPLAY.candidate_configurations(CANDIDATE_FILE)
    for index, cell in enumerate(cells, 1):
        destination = output / "simulator-memory-service-candidates" / cell.key
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
            "--memory-service-candidates",
            str(CANDIDATE_FILE),
        ]
        outputs: list[bytes] = []
        dependencies = [
            binary,
            control,
            arch_card,
            pchase_card,
            CANDIDATE_FILE,
            output / "metadata/memory_service_candidates.json",
            output / "nvbit" / cell.key,
        ]
        expectation = REPLAY.ReplayExpectation(
            launches=REPLAY.expected_launches_from_trace(
                output / "nvbit" / cell.key,
                grid=(cell.blocks, 1, 1),
                block=(cell.warps * 32, 1, 1),
            ),
            carded_dram=False,
            service_candidates=candidates,
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
                raise RuntimeError(f"{cell.key}: candidate replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: nondeterministic candidate replay")
        print(f"candidate-replay {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=("manifest", "correctness", "ncu", "source", "l2-requests", "nvbit", "static", "simulator", "simulator-carded-dram", "memory-service-candidates", "keys"),
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
        elif args.action == "l2-requests":
            run_l2_requests(output, args.force)
        elif args.action == "nvbit":
            run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force, False)
        elif args.action == "simulator-carded-dram":
            run_simulator(output, cells, args.force, True)
        elif args.action == "memory-service-candidates":
            run_memory_service_candidates(output, cells, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"L1TEX arrival campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
