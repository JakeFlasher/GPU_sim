#!/usr/bin/env python3
"""Sweep a two-address, 84-CTA RED collision oracle on sm_86."""

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
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_atomic_pair_collision_20260818_v1"
DRIVER = HERE / "atomic_multism_service_driver"
CUBIN = HERE / "atomic_multism_service_trace_red32.cubin"
OFFSETS = (0, *range(4, 128, 4), *range(128, 32_768 + 1, 128))


def load_base():
    path = HERE / "campaign.py"
    spec = importlib.util.spec_from_file_location("pair_multism_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAMPAIGN = load_base()
BASE = CAMPAIGN.BASE


@dataclass(frozen=True, slots=True)
class Cell:
    offset_bytes: int

    @property
    def key(self) -> str:
        return f"pair_s{self.offset_bytes}"

    def target(self, trials: int) -> list[str]:
        return [
            str(DRIVER), str(CUBIN), "84", "64", str(trials), self.key,
            "0", str(self.offset_bytes), "0", "2",
        ]


def cells() -> list[Cell]:
    result = [Cell(offset) for offset in OFFSETS]
    if len(result) != 288 or len({cell.key for cell in result}) != 288:
        raise RuntimeError("pair collision sweep must contain 288 unique offsets")
    return result


def selected(selector: str | None) -> list[Cell]:
    configured = cells()
    if selector is None:
        return configured
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    result = [cell for cell in configured if cell.key in requested]
    missing = requested - {cell.key for cell in result}
    if missing:
        raise RuntimeError(f"unknown pair cells: {', '.join(sorted(missing))}")
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
            "blocks_per_group": 42,
            "generations": 64,
            "operation": "RED.E.ADD.STRONG.GPU",
            "interpretation": (
                "offset zero is one shared atomic unit; a different-unit pair should "
                "approach the 42-block one-address time while a same-unit pair should "
                "approach the 84-block one-address time"
            ),
        },
    )


def run_correctness(output: Path, selected_cells: Sequence[Cell], trials: int) -> None:
    for index, cell in enumerate(selected_cells, 1):
        destination = output / "correctness" / cell.key
        completed = BASE.run_captured(cell.target(trials), destination, timeout=180)
        payload = BASE.measurement_json_line(completed.stdout, cell.key)
        if payload["correct"] is not True or payload["address_groups"] != 2:
            raise RuntimeError(f"{cell.key}: incorrect pair execution")
        BASE.write_json(destination / "result.json", {"cell": asdict(cell), "measurement": payload})
        print(f"pair {index:03d}/{len(selected_cells):03d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("manifest", "correctness"))
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=7)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        manifest(output)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
        else:
            run_correctness(output, selected(args.cells), args.trials)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"pair collision campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
