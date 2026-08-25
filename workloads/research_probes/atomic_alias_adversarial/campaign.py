#!/usr/bin/env python3
"""Run the sm_86 adversarial atomic alias/spacing/width campaign."""

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
    REPO_ROOT / "results/captures/sm86_atomic_alias_adversarial_20260818_v1"
)
DRIVER = HERE / "atomic_alias_adversarial_driver"
ALIASES = (1, 2, 4, 8, 16, 32)
WIDTHS_32 = (1, 4, 8, 16)
WIDTHS_64 = (1, 16)
CROSS_WIDTHS = (4, 8, 16)
CROSS_SPACINGS = (0, 128, 256, 4096, 4100)
GROUP_SPACINGS = (128, 256)
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
    "smsp__inst_executed_pipe_lsu.sum",
    "smsp__inst_executed_op_generic_atom_dot_alu.sum",
    "smsp__inst_executed_op_global_red.sum",
    "sm__sass_inst_executed_op_global_atom.sum",
    "sm__sass_inst_executed_op_global_red.sum",
    "smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_red.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu_lookup_hit.sum",
    "lts__t_requests_srcunit_tex_op_red.sum",
    "lts__t_requests_srcunit_tex_op_red_lookup_hit.sum",
    "lts__t_sectors_op_atom.sum",
    "lts__t_sectors_op_red.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
)
RETURN_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_executed_op_generic_atom_dot_alu.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_atom.sum",
    "l1tex__m_xbar2l1tex_read_sectors_mem_global_op_atom.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_atom.sum",
    "l1tex__m_l1tex2xbar_write_sectors_mem_global_op_atom.sum",
    "l1tex__lsu_writeback_active.sum",
    "lts__cycles_active.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
    "lts__t_sectors_op_atom.sum",
)


def load_predecessor_campaign():
    path = HERE.parent / "atomic_completion_serialization/campaign.py"
    spec = importlib.util.spec_from_file_location("alias_atomic_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREVIOUS = load_predecessor_campaign()
ORDER = PREVIOUS.ORDER
BASE = PREVIOUS.BASE


def binary_variant(width: int, returning: bool) -> str:
    return f"{'ret' if returning else 'red'}{width}"


def cubin(width: int, returning: bool, *, trace: bool) -> Path:
    infix = "_trace" if trace else ""
    return HERE / f"atomic_alias_adversarial{infix}_{binary_variant(width, returning)}.cubin"


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    width: int
    returning: bool
    pattern: str
    pattern_index: int
    alias: int
    stride_bytes: int
    scattered: bool

    @property
    def operation(self) -> str:
        return "ATOMG" if self.returning else "RED"

    @property
    def variant(self) -> str:
        return binary_variant(self.width, self.returning)

    @property
    def key(self) -> str:
        operation = "ret" if self.returning else "red"
        if self.pattern == "cross_warp":
            suffix = f"cross{self.stride_bytes}"
        elif self.pattern == "group_stride":
            suffix = f"stride{self.stride_bytes}_a{self.alias}"
        else:
            suffix = f"dense_a{self.alias}_{'s' if self.scattered else 'g'}"
        return f"w{self.warps}_{operation}{self.width}_{suffix}"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(cubin(self.width, self.returning, trace=trace)),
            str(self.warps),
            str(64 if trace else 1024),
            str(trials),
            self.key,
            str(self.width),
            "1" if self.returning else "0",
            str(self.pattern_index),
            str(self.alias),
            str(self.stride_bytes),
            "1" if self.scattered else "0",
        ]


def configured_cells() -> list[Cell]:
    cells = []
    for returning in (True, False):
        for warps in WIDTHS_32:
            for alias in ALIASES:
                cells.append(
                    Cell(
                        "dense_alias_width_transfer",
                        warps,
                        32,
                        returning,
                        "dense",
                        0,
                        alias,
                        4,
                        False,
                    )
                )
        for alias in ALIASES:
            cells.append(
                Cell(
                    "lane_assignment_holdout",
                    16,
                    32,
                    returning,
                    "dense",
                    0,
                    alias,
                    4,
                    True,
                )
            )
            for stride in GROUP_SPACINGS:
                cells.append(
                    Cell(
                        "equal_sector_intra_warp_spacing",
                        16,
                        32,
                        returning,
                        "group_stride",
                        1,
                        alias,
                        stride,
                        False,
                    )
                )
        for warps in WIDTHS_64:
            for alias in ALIASES:
                cells.append(
                    Cell(
                        "word_width_transfer",
                        warps,
                        64,
                        returning,
                        "dense",
                        0,
                        alias,
                        8,
                        False,
                    )
                )
        for warps in CROSS_WIDTHS:
            for spacing in CROSS_SPACINGS:
                cells.append(
                    Cell(
                        "cross_warp_spacing",
                        warps,
                        32,
                        returning,
                        "cross_warp",
                        2,
                        32,
                        spacing,
                        False,
                    )
                )
    if len(cells) != 138 or len({entry.key for entry in cells}) != 138:
        raise RuntimeError("adversarial atomic matrix must contain 138 unique cells")
    return cells


def selected_cells(selector: str | None) -> list[Cell]:
    cells = configured_cells()
    if selector is None:
        return cells
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [entry for entry in cells if entry.key in requested]
    missing = requested - {entry.key for entry in selected}
    if missing:
        raise RuntimeError(f"unknown cell key(s): {', '.join(sorted(missing))}")
    return selected


def expanded_ncu_cells() -> list[Cell]:
    return [entry for entry in configured_cells() if entry.warps == 16]


def source_cells(cells: Sequence[Cell]) -> list[Cell]:
    selected = []
    for entry in cells:
        if entry.warps != 16:
            continue
        if entry.pattern == "dense" and entry.width == 32 and not entry.scattered:
            selected.append(entry)
        elif entry.pattern == "dense" and entry.width == 32 and entry.alias in {1, 32}:
            selected.append(entry)
        elif entry.pattern == "group_stride" and entry.alias in {1, 32}:
            selected.append(entry)
        elif entry.pattern == "dense" and entry.width == 64 and entry.alias in {1, 32}:
            selected.append(entry)
        elif entry.pattern == "cross_warp" and entry.stride_bytes in {0, 128, 256}:
            selected.append(entry)
    return selected


def return_cells(cells: Sequence[Cell]) -> list[Cell]:
    return [
        entry
        for entry in cells
        if entry.returning
        and entry.warps == 16
        and entry.pattern == "dense"
        and not entry.scattered
    ]


def require_binaries() -> None:
    paths = [DRIVER]
    paths.extend(
        cubin(width, returning, trace=trace)
        for width in (32, 64)
        for returning in (True, False)
        for trace in (False, True)
    )
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required binary missing: {path}")


def emit_manifest(output: Path) -> None:
    require_binaries()
    cells = configured_cells()
    BASE.write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(entry), "operation": entry.operation, "variant": entry.variant, "key": entry.key} for entry in cells],
            "ncu_metrics": list(METRICS),
            "return_path_ncu_metrics": list(RETURN_METRICS),
            "long_generations": 1024,
            "trace_generations": 64,
            "ncu_repeats": {"all_cells": 1, "w16_cells": 3},
            "source_counter_cells": [entry.key for entry in source_cells(cells)],
            "research_basis": [
                "NVIDIA Nsight Compute Profiling Guide: request/sector/wavefront/L2 work-package definitions",
                "Maucher et al., Are Your GPU Atomics Secretly Contending?, PLOS 2025",
                "NVIDIA CUDA Pro Tip: NVCC automatic warp aggregation in many compiler-known cases",
            ],
            "pair_invariants": [
                "one static cubin per operation and word width across every address treatment",
                "volatile runtime index load and CTA barrier precede the first CLOCK sample",
                "one ATOMG or RED warp instruction per generation",
                "all active lanes perform exactly one atomic add per generation",
                "grouped/scattered pairs preserve the same address multiset",
                "128/256-byte group-stride pairs preserve 32 projected atomic sectors per instruction",
                "final memory and returned-old-value sums are deterministic correctness oracles",
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
    BASE.write_json(output / "metadata/environment.json", environment)


def run_correctness(output: Path, cells: Sequence[Cell], trials: int, *, trace: bool) -> None:
    directory = "trace-correctness" if trace else "correctness"
    for index, entry in enumerate(cells, 1):
        destination = output / directory / entry.key
        completed = BASE.run_captured(
            entry.target(trace=trace, trials=trials), destination, timeout=120
        )
        payload = BASE.measurement_json_line(completed.stdout, entry.key)
        BASE.write_json(
            destination / "result.json",
            {"cell": {**asdict(entry), "key": entry.key}, "measurement": payload},
        )
        print(f"{directory} {index:03d}/{len(cells):03d} {entry.key}", flush=True)


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
    for entry in cells:
        for repeat in range(repeat_start, repeat_start + repeats):
            progress += 1
            destination = output / "ncu" / entry.key / f"repeat-{repeat}"
            report_base = destination / "ncu-report"
            report = destination / "ncu-report.ncu-rep"
            if report.is_file() and not force:
                print(f"ncu {progress:03d}/{total:03d} {entry.key} r{repeat} (existing)", flush=True)
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
                    *entry.target(trace=False, trials=1),
                ],
                destination,
                timeout=1200,
            )
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            BASE.export_ncu(ncu, report, destination)
            print(f"ncu {progress:03d}/{total:03d} {entry.key} r{repeat}", flush=True)


def run_source(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = source_cells(cells)
    for index, entry in enumerate(sentinels, 1):
        destination = output / "ncu-source" / entry.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(f"source {index:02d}/{len(sentinels):02d} {entry.key} (existing)", flush=True)
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
                *entry.target(trace=False, trials=1),
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
            raise RuntimeError(f"source export failed for {entry.key}")
        print(f"source {index:02d}/{len(sentinels):02d} {entry.key}", flush=True)


def run_return_path(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = return_cells(cells)
    for index, entry in enumerate(sentinels, 1):
        destination = output / "ncu-return" / entry.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(f"return {index:02d}/{len(sentinels):02d} {entry.key} (existing)", flush=True)
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
                ",".join(RETURN_METRICS),
                *entry.target(trace=False, trials=1),
            ],
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        print(f"return {index:02d}/{len(sentinels):02d} {entry.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = [
        cubin(width, returning, trace=trace)
        for width in (32, 64)
        for returning in (True, False)
        for trace in (False, True)
    ]
    sidecars = []
    for binary in binaries:
        name = binary.stem
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

    def audit(paths: list[Path], output_name: str, run_name: str) -> None:
        BASE.run_captured(
            [
                "cargo", "run", "--quiet", "--release", "--manifest-path",
                str(REPO_ROOT / "sim/Cargo.toml"), "-p", "sim-sm", "--example",
                "operand_projection_audit", "--", "--out",
                str(output / "metadata" / output_name), *map(str, paths),
            ],
            output / "metadata" / run_name,
            timeout=900,
        )

    audit(sidecars, "operand_projection_audit.json", "operand-projection-run")
    output_sass = destination.resolve()
    retained = sorted(
        path
        for path in REPO_ROOT.glob("results/captures/**/*.control_bits.json")
        if path.parent.resolve() != output_sass
    )
    audit(retained + sidecars, "retained_corpus_operand_projection_audit.json", "retained-corpus-operand-projection-run")
    print(destination)


def run_simulator(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = REPO_ROOT / "sim/target/release/sim-run"
    if not binary.is_file():
        raise RuntimeError(f"simulator missing: {binary}")
    for index, entry in enumerate(cells, 1):
        destination = output / "simulator" / entry.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / entry.key),
            "--control",
            str(output / "sass" / f"atomic_alias_adversarial_trace_{entry.variant}.control_bits.json"),
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
                raise RuntimeError(f"{entry.key}: replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{entry.key}: nondeterministic replay")
        launch = json.loads(outputs[0])["launches"][0]
        if launch["coverage"]["kind"] != "full":
            raise RuntimeError(f"{entry.key}: incomplete replay")
        print(f"simulator {index:03d}/{len(cells):03d} {entry.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=("manifest", "correctness", "trace-correctness", "ncu", "source", "return-path", "nvbit", "static", "simulator", "expanded-keys"),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=7)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--repeats", type=int, default=1)
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
        elif args.action == "expanded-keys":
            print(",".join(entry.key for entry in expanded_ncu_cells()))
        elif args.action == "correctness":
            run_correctness(output, cells, args.trials, trace=False)
        elif args.action == "trace-correctness":
            run_correctness(output, cells, args.trials, trace=True)
        elif args.action == "ncu":
            run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "return-path":
            run_return_path(output, cells, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"adversarial atomic campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
