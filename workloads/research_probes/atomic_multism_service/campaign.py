#!/usr/bin/env python3
"""Run the sm_86 multi-SM atomic service/placement campaign."""

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
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_atomic_multism_service_20260818_v1"
DRIVER = HERE / "atomic_multism_service_driver"
BLOCKS = (4, 16, 42, 84, 168)
SPACINGS = (0, 128, 256, 4096, 4100)
FINE_SPACINGS = (
    4, 8, 16, 32, 64, 96, 124, 128, 132, 160, 192, 224, 252, 256,
    260, 288, 320, 384, 512, 1024, 2048, 4064, 4092, 4096, 4100, 4128, 8192,
)


def load_predecessor_campaign():
    path = HERE.parent / "atomic_alias_adversarial/campaign.py"
    spec = importlib.util.spec_from_file_location("multism_alias_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ALIAS = load_predecessor_campaign()
ORDER = ALIAS.ORDER
BASE = ALIAS.BASE
METRICS = ALIAS.METRICS


def cubin(returning: bool, *, trace: bool) -> Path:
    infix = "_trace" if trace else ""
    variant = "ret32" if returning else "red32"
    return HERE / f"atomic_multism_service{infix}_{variant}.cubin"


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    blocks: int
    returning: bool
    spacing_bytes: int
    permuted: bool

    @property
    def warps(self) -> int:
        return self.blocks

    @property
    def operation(self) -> str:
        return "ATOMG" if self.returning else "RED"

    @property
    def variant(self) -> str:
        return "ret32" if self.returning else "red32"

    @property
    def key(self) -> str:
        op = "ret" if self.returning else "red"
        return f"b{self.blocks}_{op}_s{self.spacing_bytes}_{'p' if self.permuted else 'd'}"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(cubin(self.returning, trace=trace)),
            str(self.blocks),
            str(64 if trace else 1024),
            str(trials),
            self.key,
            "1" if self.returning else "0",
            str(self.spacing_bytes),
            "1" if self.permuted else "0",
        ]


def configured_cells() -> list[Cell]:
    cells = []
    for returning in (True, False):
        cells.append(Cell("single_cta_baseline", 1, returning, 0, False))
        for blocks in BLOCKS:
            for spacing in SPACINGS:
                cells.append(
                    Cell(
                        "one_wave_transfer" if blocks <= 84 else "two_wave_transfer",
                        blocks,
                        returning,
                        spacing,
                        False,
                    )
                )
        for spacing in SPACINGS[1:]:
            cells.append(Cell("block_address_permutation", 84, returning, spacing, True))
        for spacing in FINE_SPACINGS:
            if spacing not in SPACINGS:
                cells.append(Cell("fine_atomic_unit_spacing", 84, returning, spacing, False))
    if len(cells) != 106 or len({entry.key for entry in cells}) != 106:
        raise RuntimeError("multi-SM atomic matrix must contain 106 unique cells")
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
    return [
        entry
        for entry in configured_cells()
        if entry.blocks == 168
        or (entry.blocks == 84 and (entry.spacing_bytes in SPACINGS or entry.permuted))
    ]


def fine_spacing_cells() -> list[Cell]:
    return [entry for entry in configured_cells() if entry.stage == "fine_atomic_unit_spacing"]


def source_cells(cells: Sequence[Cell]) -> list[Cell]:
    return [
        entry
        for entry in cells
        if (entry.blocks == 84 and not entry.permuted and entry.spacing_bytes in SPACINGS)
        or (entry.blocks == 168 and entry.spacing_bytes in {0, 256})
    ]


def require_binaries() -> None:
    paths = [DRIVER]
    paths.extend(cubin(returning, trace=trace) for returning in (True, False) for trace in (False, True))
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
            "cells": [{**asdict(entry), "warps": entry.warps, "operation": entry.operation, "variant": entry.variant, "key": entry.key} for entry in cells],
            "ncu_metrics": list(METRICS),
            "long_generations": 1024,
            "trace_generations": 64,
            "sm_count": 84,
            "ncu_repeats": {"all_cells": 1, "coarse_blocks_84_or_168": 3},
            "fine_spacing_bytes": list(FINE_SPACINGS),
            "source_counter_cells": [entry.key for entry in source_cells(cells)],
            "pair_invariants": [
                "one warp per CTA and one ATOMG or RED instruction per generation",
                "volatile runtime index load and CTA barrier precede CLOCK timing",
                "every warp has full 32-lane same-word aliasing",
                "direct/permuted pairs preserve the complete block-address multiset",
                "84 blocks target one nominal CTA per physical SM; 168 blocks require at least two waves",
                "final memory and returned-old-value sums are deterministic",
            ],
        },
    )
    environment: dict[str, object] = {"captured_at": BASE.now()}
    for name, command in {
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
        "nvcc": ["/usr/local/cuda-13.2/bin/nvcc", "--version"],
        "ncu": [os.environ.get("NCU", "ncu"), "--version"],
    }.items():
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        environment[name] = {"command": command, "return_code": completed.returncode, "output": completed.stdout.strip()}
    BASE.write_json(output / "metadata/environment.json", environment)


def run_correctness(output: Path, cells: Sequence[Cell], trials: int, *, trace: bool) -> None:
    directory = "trace-correctness" if trace else "correctness"
    for index, entry in enumerate(cells, 1):
        destination = output / directory / entry.key
        completed = BASE.run_captured(entry.target(trace=trace, trials=trials), destination, timeout=180)
        payload = BASE.measurement_json_line(completed.stdout, entry.key)
        BASE.write_json(destination / "result.json", {"cell": {**asdict(entry), "key": entry.key}, "measurement": payload})
        print(f"{directory} {index:02d}/{len(cells):02d} {entry.key}", flush=True)


def run_ncu(output: Path, cells: Sequence[Cell], repeat_start: int, repeats: int, force: bool) -> None:
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
                [ncu, "--force-overwrite", "--export", str(report_base), "--profile-from-start", "on", "--launch-count", "1", "--replay-mode", "application", "--clock-control", "base", "--cache-control", "none", "--metrics", ",".join(METRICS), *entry.target(trace=False, trials=1)],
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
            [ncu, "--force-overwrite", "--export", str(report_base), "--profile-from-start", "on", "--launch-count", "1", "--replay-mode", "application", "--clock-control", "base", "--cache-control", "none", "--section", "SourceCounters", *entry.target(trace=False, trials=1)],
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        completed = subprocess.run(
            [ncu, "--import", str(report), "--csv", "--page", "source", "--print-source", "sass", "--log-file", str(destination / "source.csv")],
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


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    sidecars = []
    for returning in (True, False):
        for trace in (False, True):
            binary = cubin(returning, trace=trace)
            name = binary.stem
            for suffix, option in (("sass", "--dump-sass"), ("resources", "--dump-resource-usage")):
                completed = subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", option, str(binary)], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                (destination / f"{name}.{suffix}.txt").write_text(completed.stdout, encoding="utf-8")
                (destination / f"{name}.{suffix}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(f"cuobjdump failed for {binary}")
            sidecar = destination / f"{name}.control_bits.json"
            BASE.run_captured(
                [str(REPO_ROOT / ".venv/bin/python"), str(REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"), "--binary", str(binary), "--arch", "sm_86", "--out", str(sidecar), "--toolkit-bindir", "/usr/local/cuda-13.2/bin"],
                destination / f"{name}.control-extraction",
            )
            sidecars.append(sidecar)

    def audit(paths: list[Path], output_name: str, run_name: str) -> None:
        BASE.run_captured(
            ["cargo", "run", "--quiet", "--release", "--manifest-path", str(REPO_ROOT / "sim/Cargo.toml"), "-p", "sim-sm", "--example", "operand_projection_audit", "--", "--out", str(output / "metadata" / output_name), *map(str, paths)],
            output / "metadata" / run_name,
            timeout=900,
        )

    audit(sidecars, "operand_projection_audit.json", "operand-projection-run")
    output_sass = destination.resolve()
    retained = sorted(path for path in REPO_ROOT.glob("results/captures/**/*.control_bits.json") if path.parent.resolve() != output_sass)
    audit(retained + sidecars, "retained_corpus_operand_projection_audit.json", "retained-corpus-operand-projection-run")
    print(destination)


def run_simulator(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = REPO_ROOT / "sim/target/release/sim-run"
    for index, entry in enumerate(cells, 1):
        destination = output / "simulator" / entry.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [str(binary), "--trace", str(output / "nvbit" / entry.key), "--control", str(output / "sass" / f"atomic_multism_service_trace_{entry.variant}.control_bits.json"), "--arch-card", str(REPO_ROOT / "arch_profiles/cards/sm_86.json"), "--pchase-card", str(REPO_ROOT / "data/pchase/sm_86_static_card.json"), "--mode", "l1"]
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
        print(f"simulator {index:02d}/{len(cells):02d} {entry.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("manifest", "correctness", "trace-correctness", "ncu", "source", "nvbit", "static", "simulator", "expanded-keys", "fine-keys"))
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
        elif args.action == "fine-keys":
            print(",".join(entry.key for entry in fine_spacing_cells()))
        elif args.action == "correctness":
            run_correctness(output, cells, args.trials, trace=False)
        elif args.action == "trace-correctness":
            run_correctness(output, cells, args.trials, trace=True)
        elif args.action == "ncu":
            run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"multi-SM atomic campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
