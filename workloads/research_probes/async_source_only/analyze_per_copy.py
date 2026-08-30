#!/usr/bin/env python3
"""Validate bypass per-copy observations against retained visibility replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROUTES = ("l1", "dram")
LAYOUTS = ("compact", "conflict")
DEPTHS = (4, 8)
REPO_ROOT = Path(__file__).resolve().parents[3]


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def cell_name(route: str, layout: str, depth: int) -> str:
    return f"async_source_visibility_{route}_{layout}_d{depth}_g64"


def replay_pair(root: Path, cell: str) -> tuple[dict[str, Any], str]:
    first = root / "simulator" / cell / "replay-1.json"
    second = root / "simulator" / cell / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise AnalysisError(f"{root}: missing replay twins for {cell}")
    first_bytes = first.read_bytes()
    if first_bytes != second.read_bytes():
        raise AnalysisError(f"{root}: replay twins differ for {cell}")
    return read_json(first), sha256(first)


def normalized_without_observation(document: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    normalized = deepcopy(document)
    readiness = normalized["launches"][0]["timing"]["readiness"]
    observation = readiness.pop("bypass_async_copies", None)
    if observation is not None:
        readiness["schema_version"] = 3
    return normalized, observation


def expected_observation(layout: str, depth: int) -> dict[str, int]:
    copies = 64 * depth
    wavefronts_per_copy = 8 if layout == "compact" else 16
    return {
        "copies_observed": copies,
        "destination_wavefronts_observed": copies * wavefronts_per_copy,
        "copy_source_spread_cycles": 1024 if depth == 4 else 2304,
        "per_copy_completion_cycles": 27840,
        "group_aggregate_excess_cycles": 3072 if depth == 4 else 7168,
    }


def run(root: Path, baseline_root: Path, output: Path) -> int:
    root = root.resolve()
    baseline_root = baseline_root.resolve()
    cells: list[dict[str, Any]] = []
    errors: list[str] = []
    for route in ROUTES:
        for depth in DEPTHS:
            for layout in LAYOUTS:
                cell = cell_name(route, layout, depth)
                try:
                    current, current_hash = replay_pair(root, cell)
                    baseline, baseline_hash = replay_pair(baseline_root, cell)
                    normalized, observation = normalized_without_observation(current)
                    unchanged = normalized == baseline
                    if not unchanged:
                        raise AnalysisError(f"{cell}: non-observational output changed")
                    if route == "l1":
                        if observation is not None or current != baseline:
                            raise AnalysisError(f"{cell}: cache-all output is not byte-compatible")
                    else:
                        expected = expected_observation(layout, depth)
                        if observation != expected:
                            raise AnalysisError(
                                f"{cell}: observation mismatch: {observation!r} != {expected!r}"
                            )
                    launch = current["launches"][0]
                    cells.append(
                        {
                            "cell": cell,
                            "route": route,
                            "layout": layout,
                            "depth": depth,
                            "terminal_modeled_event_time": launch["timing"][
                                "terminal_modeled_event_time"
                            ],
                            "completion_frontier": launch["timing"]["completion_frontier"],
                            "observation": observation,
                            "non_observational_output_unchanged": unchanged,
                            "replay_sha256": current_hash,
                            "baseline_replay_sha256": baseline_hash,
                        }
                    )
                except (AnalysisError, KeyError, IndexError, TypeError) as error:
                    errors.append(str(error))

    criteria = {
        "all_eight_cells_present": len(cells) == 8,
        "all_non_observational_output_unchanged": len(cells) == 8
        and all(cell["non_observational_output_unchanged"] for cell in cells),
        "cache_all_byte_compatible": len(cells) == 8
        and all(
            cell["replay_sha256"] == cell["baseline_replay_sha256"]
            for cell in cells
            if cell["route"] == "l1"
        ),
        "bypass_observations_complete": len(cells) == 8
        and all(cell["observation"] is not None for cell in cells if cell["route"] == "dram"),
    }
    if not all(criteria.values()):
        errors.append("one or more per-copy observation criteria failed")

    binary = REPO_ROOT / "sim/target/release/sim-run"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "baseline_root": str(baseline_root),
        "simulator_binary": {"path": str(binary), "sha256": sha256(binary)},
        "cells": cells,
        "criteria": criteria,
        "errors": errors,
        "status": "pass_observational_per_copy_no_timing_change"
        if not errors
        else "invalid_per_copy_observation",
        "decision": "retain_typed_tail_and_evaluate_bypass_projection",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "status": report["status"], "cells": len(cells)},
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.root, args.baseline_root, args.output)
    except (AnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"per-copy observation analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
