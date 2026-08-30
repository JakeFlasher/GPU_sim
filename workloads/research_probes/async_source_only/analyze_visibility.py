#!/usr/bin/env python3
"""Reduce the fixed-source compact/conflict LDGSTS visibility slice."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROUTES = ("l1", "dram")
LAYOUTS = ("compact", "conflict")
DEPTHS = (4, 8)
REPEATS = 5
GROUPS = 64
REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__sectors_read.sum",
)
RESOURCE_RE = re.compile(
    r"^ Function (?P<name>\S+):\n"
    r"  REG:(?P<registers>\d+) STACK:(?P<stack>\d+) "
    r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)",
    re.MULTILINE,
)


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_name(route: str, layout: str, depth: int) -> str:
    return f"async_source_visibility_{route}_{layout}_d{depth}_g{GROUPS}"


def function_name(route: str, layout: str, depth: int) -> str:
    dram = 1 if route == "dram" else 0
    conflict = 1 if layout == "conflict" else 0
    return f"_Z18source_only_kernelILi{depth}ELb{dram}ELb{conflict}EEvPKci"


def read_ncu_csv(path: Path) -> dict[str, str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("Kernel Name")]
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot read NCU CSV {path}: {error}") from error
    if len(rows) != 1:
        raise AnalysisError(f"{path}: expected one kernel row, got {len(rows)}")
    return {str(key).strip(): str(value).strip() for key, value in rows[0].items()}


def ncu_cell(root: Path, route: str, layout: str, depth: int) -> dict[str, Any]:
    base = root / "ncu" / cell_name(route, layout, depth)
    paths = sorted(base.glob("repeat-*/ncu.csv"))
    if len(paths) != REPEATS:
        raise AnalysisError(f"{base}: expected {REPEATS} NCU repeats, got {len(paths)}")
    repeats: list[dict[str, float]] = []
    for path in paths:
        row = read_ncu_csv(path)
        try:
            repeats.append({metric: float(row[metric]) for metric in METRICS})
        except (KeyError, ValueError) as error:
            raise AnalysisError(f"{path}: missing/non-numeric metric: {error}") from error
    medians = {
        metric: statistics.median(repeat[metric] for repeat in repeats)
        for metric in METRICS
    }
    active = [repeat["sm__cycles_active.max"] for repeat in repeats]
    relative_range = (max(active) - min(active)) / statistics.mean(active)
    return {
        "route": route,
        "layout": layout,
        "depth": depth,
        "repeats": repeats,
        "median": medians,
        "active_cycle_relative_range": relative_range,
        "csv_sha256": [sha256(path) for path in paths],
    }


def simulator_cell(root: Path, route: str, layout: str, depth: int) -> dict[str, Any]:
    base = root / "simulator" / cell_name(route, layout, depth)
    first = base / "replay-1.json"
    second = base / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise AnalysisError(f"{base}: simulator replay twins are incomplete")
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    if first_bytes != second_bytes:
        raise AnalysisError(f"{base}: simulator replay twins differ")
    document = json.loads(first_bytes)
    launches = document.get("launches")
    if document.get("schema_version") != 1 or not isinstance(launches, list) or len(launches) != 1:
        raise AnalysisError(f"{first}: launch/schema contract failed")
    launch = launches[0]
    if launch.get("coverage", {}).get("kind") != "full":
        raise AnalysisError(f"{first}: coverage is not full")
    timing = launch["timing"]
    readiness = timing["readiness"]
    return {
        "replay_sha256": sha256(first),
        "instructions_issued": launch["counters"]["instructions_issued"],
        "records_seen": launch["counters"]["records_seen"],
        "counters": launch["counters"],
        "timing": timing,
        "terminal_modeled_event_time": timing["terminal_modeled_event_time"],
        "async_groups": readiness["async_groups"],
        "typed_tails": launch["counters"]["unmodeled_tails"],
        "coverage": launch["coverage"],
    }


def static_report(root: Path) -> dict[str, Any]:
    control_path = root / "static/control_bits.json"
    resource_path = root / "static/resources.txt"
    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
        resources_text = resource_path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot load static evidence: {error}") from error
    summary = control.get("summary")
    functions = control.get("functions")
    if not isinstance(summary, Mapping) or not isinstance(functions, Mapping):
        raise AnalysisError("control sidecar is missing functions/summary")
    if summary.get("total_unknown_encoding") != 0:
        raise AnalysisError("control sidecar contains unknown instruction encodings")
    resources = {
        match.group("name"): {
            "registers": int(match.group("registers")),
            "stack": int(match.group("stack")),
            "shared": int(match.group("shared")),
            "local": int(match.group("local")),
        }
        for match in RESOURCE_RE.finditer(resources_text)
    }
    selected: dict[str, Any] = {}
    forbidden: dict[str, list[str]] = {}
    for route in ROUTES:
        for layout in LAYOUTS:
            for depth in DEPTHS:
                name = function_name(route, layout, depth)
                rows = functions.get(name)
                stats = summary.get("functions", {}).get(name)
                resource = resources.get(name)
                if not isinstance(rows, list) or not isinstance(stats, Mapping) or resource is None:
                    raise AnalysisError(f"static evidence is missing {name}")
                opcodes = [str(row.get("opcode_text", "")) for row in rows if isinstance(row, Mapping)]
                bad = [
                    opcode
                    for opcode in opcodes
                    if opcode.startswith(("STG", "ATOM", "RED", "BAR.SYNC"))
                ]
                if bad:
                    forbidden[name] = bad
                selected[name] = {
                    "instructions": stats.get("instructions"),
                    "stall_cycles_static_sum": stats.get("stall_cycles_static_sum"),
                    **resource,
                }
    pair_resources_equal = True
    pair_static_shape_equal = True
    for route in ROUTES:
        for depth in DEPTHS:
            compact = selected[function_name(route, "compact", depth)]
            conflict = selected[function_name(route, "conflict", depth)]
            pair_resources_equal &= all(
                compact[field] == conflict[field]
                for field in ("registers", "stack", "shared", "local")
            )
            pair_static_shape_equal &= all(
                compact[field] == conflict[field]
                for field in ("instructions", "stall_cycles_static_sum")
            )
    return {
        "control_sha256": sha256(control_path),
        "resources_sha256": sha256(resource_path),
        "function_count": len(functions),
        "decoded": summary.get("total_decoded"),
        "unknown_encoding": summary.get("total_unknown_encoding"),
        "selected_functions": selected,
        "forbidden_target_opcodes": forbidden,
        "pair_resources_equal": pair_resources_equal,
        "pair_static_shape_equal": pair_static_shape_equal,
    }


def paired_delta(
    compact: Mapping[str, Any], conflict: Mapping[str, Any], metric: str
) -> dict[str, Any]:
    deltas = [
        conflict_repeat[metric] - compact_repeat[metric]
        for compact_repeat, conflict_repeat in zip(
            compact["repeats"], conflict["repeats"], strict=True
        )
    ]
    percentages = [
        100.0 * delta / compact_repeat[metric]
        for compact_repeat, delta in zip(compact["repeats"], deltas, strict=True)
    ]
    return {
        "per_repeat": deltas,
        "median": statistics.median(deltas),
        "minimum": min(deltas),
        "maximum": max(deltas),
        "percent_per_repeat": percentages,
        "median_percent": statistics.median(percentages),
        "all_positive": all(delta > 0 for delta in deltas),
    }


def expected_live(route: str, layout: str, depth: int) -> dict[str, float]:
    instructions = GROUPS * depth
    source_sectors = instructions * 32
    if layout == "compact":
        destination_wavefronts_per_instruction = 8
    elif route == "l1":
        destination_wavefronts_per_instruction = 16
    else:
        destination_wavefronts_per_instruction = 32
    first_group_sectors = depth * 32
    return {
        "smsp__inst_executed_op_ldgsts.sum": float(instructions),
        "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum": float(source_sectors),
        "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum": float(
            instructions * destination_wavefronts_per_instruction
        ),
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum": float(
            source_sectors - first_group_sectors if route == "l1" else 0
        ),
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum": float(
            first_group_sectors if route == "l1" else source_sectors
        ),
    }


def run(root: Path, output: Path) -> int:
    root = root.resolve()
    errors: list[str] = []
    try:
        static = static_report(root)
    except AnalysisError as error:
        static = {"error": str(error)}
        errors.append(str(error))

    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    for route in ROUTES:
        for depth in DEPTHS:
            for layout in LAYOUTS:
                try:
                    cells[(route, layout, depth)] = {
                        "route": route,
                        "layout": layout,
                        "depth": depth,
                        "ncu": ncu_cell(root, route, layout, depth),
                        "simulator": simulator_cell(root, route, layout, depth),
                    }
                except (AnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    errors.append(str(error))

    pairs: list[dict[str, Any]] = []
    for route in ROUTES:
        for depth in DEPTHS:
            compact = cells.get((route, "compact", depth))
            conflict = cells.get((route, "conflict", depth))
            if compact is None or conflict is None:
                continue
            compact_ncu = compact["ncu"]
            conflict_ncu = conflict["ncu"]
            live_active = paired_delta(compact_ncu, conflict_ncu, "sm__cycles_active.max")
            live_gpc = paired_delta(compact_ncu, conflict_ncu, "gpc__cycles_elapsed.max")
            simulator_delta = (
                conflict["simulator"]["terminal_modeled_event_time"]
                - compact["simulator"]["terminal_modeled_event_time"]
            )
            transfer_error = (
                abs(simulator_delta - live_active["median"]) / live_active["median"]
                if live_active["median"] > 0
                else None
            )
            source_metrics_equal = all(
                compact_repeat[metric] == conflict_repeat[metric]
                for metric in (
                    "smsp__inst_executed_op_ldgsts.sum",
                    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
                )
                for compact_repeat, conflict_repeat in zip(
                    compact_ncu["repeats"], conflict_ncu["repeats"], strict=True
                )
            )
            expected_exact = True
            for layout, cell in (("compact", compact), ("conflict", conflict)):
                expected = expected_live(route, layout, depth)
                expected_exact &= all(
                    all(repeat[metric] == value for repeat in cell["ncu"]["repeats"])
                    for metric, value in expected.items()
                )
            pair = {
                "route": route,
                "depth": depth,
                "source_metrics_equal": source_metrics_equal,
                "expected_live_work_exact": expected_exact,
                "destination_wavefront_ratio": (
                    conflict_ncu["median"][
                        "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
                    ]
                    / compact_ncu["median"][
                        "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
                    ]
                ),
                "live_sm_active_delta": live_active,
                "live_gpc_elapsed_delta": live_gpc,
                "simulator_terminal_delta": simulator_delta,
                "simulator_to_live_active_delta_relative_error": transfer_error,
                "compact_typed_tails": compact["simulator"]["typed_tails"],
                "conflict_typed_tails": conflict["simulator"]["typed_tails"],
            }
            pairs.append(pair)

    criteria = {
        "static_pairs_equal": bool(
            static.get("pair_resources_equal")
            and static.get("pair_static_shape_equal")
            and not static.get("forbidden_target_opcodes")
        ),
        "source_work_equal_and_exact": bool(pairs)
        and all(pair["source_metrics_equal"] and pair["expected_live_work_exact"] for pair in pairs),
        "live_layout_delta_positive": bool(pairs)
        and all(
            pair["live_sm_active_delta"]["all_positive"]
            and pair["live_gpc_elapsed_delta"]["all_positive"]
            for pair in pairs
        ),
        "l1_delta_transfers_within_five_percent": all(
            pair["simulator_to_live_active_delta_relative_error"] <= 0.05
            for pair in pairs
            if pair["route"] == "l1"
        ),
        "bypass_layout_delta_unmodeled": all(
            pair["simulator_terminal_delta"] == 0
            and pair["compact_typed_tails"].get("async_copy_shared_visibility") is True
            and pair["conflict_typed_tails"].get("async_copy_shared_visibility") is True
            for pair in pairs
            if pair["route"] == "dram"
        ),
    }
    if not all(criteria.values()):
        errors.append("one or more discriminator acceptance criteria failed")

    binary = REPO_ROOT / "workloads/research_probes/async_source_only/async_source_only"
    simulator = REPO_ROOT / "sim/target/release/sim-run"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "probe_binary": {"path": str(binary), "sha256": sha256(binary)},
        "simulator_binary": {"path": str(simulator), "sha256": sha256(simulator)},
        "static": static,
        "cells": [cells[key] for key in sorted(cells)],
        "pairs": pairs,
        "criteria": criteria,
        "errors": errors,
        "status": "pass_bypass_visibility_discriminator_no_timing_promotion"
        if not errors
        else "invalid_visibility_slice",
        "decision": "retain_bypass_typed_tail_pending_issue_vs_completion_split",
        "rationale": (
            "The fixed-source layout effect transfers across depth for cache-all copies, but "
            "the bypass live delta does not identify whether the extra cycles arise at issue, "
            "source/destination replay, or post-source shared visibility."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "cells": len(cells),
                "pairs": len(pairs),
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.root, args.output)
    except (AnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"async-source visibility analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
