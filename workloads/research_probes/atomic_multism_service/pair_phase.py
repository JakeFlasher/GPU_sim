#!/usr/bin/env python3
"""Sweep absolute 256-byte address phase for two-address RED pairs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_atomic_pair_phase_20260818_v1"
DRIVER = HERE / "atomic_multism_service_driver"
CUBIN = HERE / "atomic_multism_service_trace_red32.cubin"
BASES = tuple(range(0, 256, 4))
DELTAS = (128, 256, 260, 4096)


def load_base():
    path = HERE / "campaign.py"
    spec = importlib.util.spec_from_file_location("phase_multism_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base().BASE


@dataclass(frozen=True, slots=True)
class Cell:
    base_bytes: int
    delta_bytes: int

    @property
    def key(self) -> str:
        return f"base{self.base_bytes}_delta{self.delta_bytes}"

    def target(self, trials: int) -> list[str]:
        return [
            str(DRIVER), str(CUBIN), "84", "64", str(trials), self.key,
            "0", str(self.delta_bytes), "0", "2", str(self.base_bytes),
        ]


def cells() -> list[Cell]:
    result = [Cell(base, delta) for delta in DELTAS for base in BASES]
    if len(result) != 256 or len({cell.key for cell in result}) != 256:
        raise RuntimeError("pair phase sweep must contain 256 cells")
    return result


def manifest(output: Path) -> None:
    for path in (DRIVER, CUBIN):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing binary: {path}")
    BASE.write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(cell), "key": cell.key} for cell in cells()],
            "blocks": 84,
            "address_groups": 2,
            "generations": 64,
            "operation": "RED.E.ADD.STRONG.GPU",
        },
    )


def run(output: Path, trials: int) -> None:
    for index, cell in enumerate(cells(), 1):
        destination = output / "correctness" / cell.key
        completed = BASE.run_captured(cell.target(trials), destination, timeout=180)
        payload = BASE.measurement_json_line(completed.stdout, cell.key)
        if payload["correct"] is not True or payload["base_bytes"] != cell.base_bytes:
            raise RuntimeError(f"{cell.key}: incorrect phase execution")
        BASE.write_json(destination / "result.json", {"cell": asdict(cell), "measurement": payload})
        print(f"phase {index:03d}/256 {cell.key}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("manifest", "correctness"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        output = args.output_root.resolve()
        manifest(output)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
        else:
            run(output, args.trials)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"pair phase campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
