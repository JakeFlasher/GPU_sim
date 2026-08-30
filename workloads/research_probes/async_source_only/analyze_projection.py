#!/usr/bin/env python3
"""Reduce the compact/conflict/severe bypass destination projection assay."""

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


ROUTES = ("l1", "l2")
LAYOUTS = ("compact", "conflict", "severe")
LAYOUT_INDEX = {"compact": 0, "conflict": 1, "severe": 2}
REPEATS = 9
GROUPS = 8192
REPO_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_RE = re.compile(
    r"^ Function (?P<name>\S+):\n"
    r"  REG:(?P<registers>\d+) STACK:(?P<stack>\d+) "
    r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)",
    re.MULTILINE,
)
NCU_METRICS = (
    "smsp__inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
)


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


def function_name(route: str, layout: str) -> str:
    bypass = 0 if route == "l1" else 1
    return (
        f"_Z23visibility_phase_kernelILi4ELb{bypass}ELb0"
        f"ELi{LAYOUT_INDEX[layout]}EEvPKciP11PhaseResult"
    )


def direct_cell(root: Path, route: str, layout: str) -> dict[str, Any]:
    base = root / "direct" / f"async_bypass_projection_{route}_{layout}_d4_g{GROUPS}"
    paths = [base / f"repeat-{repeat}/result.json" for repeat in range(1, REPEATS + 1)]
    if not all(path.is_file() for path in paths):
        raise AnalysisError(f"{base}: direct repeat set is incomplete")
    repeats: list[dict[str, Any]] = []
    for path in paths:
        value = read_json(path)
        expected = {
            "workload": "async_visibility_phase",
            "route": route,
            "layout": layout,
            "depth": 4,
            "groups": GROUPS,
            "sum_consistent": True,
            "result_store_after_intervals": True,
            "correct": True,
        }
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise AnalysisError(f"{path}: result contract mismatch")
        repeats.append(value)
    return {
        "route": route,
        "layout": layout,
        "repeats": repeats,
        "median_per_group": {
            field: statistics.median(value[field] for value in repeats) / GROUPS
            for field in ("issue_cycles", "wait_cycles", "total_cycles")
        },
        "relative_range": {
            field: (
                max(value[field] for value in repeats)
                - min(value[field] for value in repeats)
            )
            / statistics.mean(value[field] for value in repeats)
            for field in ("issue_cycles", "wait_cycles", "total_cycles")
        },
        "result_sha256": [sha256(path) for path in paths],
    }


def ncu_cell(root: Path, route: str, layout: str) -> dict[str, Any]:
    path = (
        root
        / "ncu"
        / f"async_bypass_projection_{route}_{layout}_d4_g64"
        / "repeat-1/ncu.csv"
    )
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("Kernel Name")]
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if len(rows) != 1:
        raise AnalysisError(f"{path}: expected one kernel row, got {len(rows)}")
    try:
        metrics = {metric: float(rows[0][metric]) for metric in NCU_METRICS}
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"{path}: missing/non-numeric metric: {error}") from error
    return {"route": route, "layout": layout, "metrics": metrics, "csv_sha256": sha256(path)}


def static_report(root: Path) -> dict[str, Any]:
    control_path = root / "static/control_bits.json"
    resources_path = root / "static/resources.txt"
    control = read_json(control_path)
    try:
        resources_text = resources_path.read_text(encoding="utf-8")
    except OSError as error:
        raise AnalysisError(f"cannot read {resources_path}: {error}") from error
    summary = control.get("summary")
    functions = control.get("functions")
    if not isinstance(summary, Mapping) or not isinstance(functions, Mapping):
        raise AnalysisError("static sidecar is missing functions/summary")
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
    contracts: dict[str, Any] = {}
    for route in ROUTES:
        for layout in LAYOUTS:
            name = function_name(route, layout)
            rows = functions.get(name)
            stats = summary.get("functions", {}).get(name)
            resource = resources.get(name)
            if not isinstance(rows, list) or not isinstance(stats, Mapping) or resource is None:
                raise AnalysisError(f"static evidence is missing {name}")
            clocks = [row for row in rows if "CLOCKLO" in str(row.get("opcode_text", ""))]
            copies = [row for row in rows if str(row.get("opcode_text", "")).startswith("LDGSTS")]
            commit = [row for row in rows if row.get("opcode_text") == "LDGDEPBAR"]
            wait = [row for row in rows if str(row.get("opcode_text", "")).startswith("DEPBAR.LE")]
            stores = [row for row in rows if str(row.get("opcode_text", "")).startswith("STG")]
            if not (len(clocks) == 3 and len(copies) == 4 and len(commit) == 1 and len(wait) == 1 and stores):
                raise AnalysisError(f"{name}: static phase cardinality mismatch")
            read_mask = 1 << copies[-1]["control"]["read_barrier"]
            contracts[name] = {
                "ordered": clocks[0]["pc"] < copies[0]["pc"]
                and copies[-1]["pc"] < clocks[1]["pc"] < commit[0]["pc"]
                and commit[0]["pc"] < wait[0]["pc"] < clocks[2]["pc"]
                and clocks[2]["pc"] < min(store["pc"] for store in stores),
                "clock_masks_read_barrier_only": all(
                    clock["control"]["wait_mask"] in (0, read_mask)
                    for clock in clocks[:2]
                ),
                "post_wait_mask_zero": clocks[2]["control"]["wait_mask"] == 0,
            }
            selected[name] = {
                "instructions": stats.get("instructions"),
                "stall_cycles_static_sum": stats.get("stall_cycles_static_sum"),
                **resource,
            }
    shapes_equal = all(
        selected[function_name(route, layout)] == selected[function_name(route, "compact")]
        for route in ROUTES
        for layout in LAYOUTS
    )
    return {
        "control_sha256": sha256(control_path),
        "resources_sha256": sha256(resources_path),
        "function_count": len(functions),
        "decoded": summary.get("total_decoded"),
        "unknown_encoding": summary.get("total_unknown_encoding"),
        "selected_functions": selected,
        "phase_contracts": contracts,
        "selected_shapes_equal": shapes_equal,
        "phase_contracts_valid": all(all(value.values()) for value in contracts.values()),
    }


def pair_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    deltas = [
        (r["wait_cycles"] - l["wait_cycles"]) / GROUPS
        for l, r in zip(left["repeats"], right["repeats"], strict=True)
    ]
    return {
        "per_repeat_per_group": deltas,
        "median_per_group": statistics.median(deltas),
        "minimum_per_group": min(deltas),
        "maximum_per_group": max(deltas),
        "positive_repeats": sum(delta > 0 for delta in deltas),
    }


def run(root: Path, output: Path) -> int:
    root = root.resolve()
    errors: list[str] = []
    try:
        static = static_report(root)
    except AnalysisError as error:
        static = {"error": str(error)}
        errors.append(str(error))
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    ncu: dict[tuple[str, str], dict[str, Any]] = {}
    for route in ROUTES:
        for layout in LAYOUTS:
            try:
                cells[(route, layout)] = direct_cell(root, route, layout)
                ncu[(route, layout)] = ncu_cell(root, route, layout)
            except AnalysisError as error:
                errors.append(str(error))
    pairs: list[dict[str, Any]] = []
    for route in ROUTES:
        for left, right in (("compact", "conflict"), ("conflict", "severe"), ("compact", "severe")):
            if (route, left) in cells and (route, right) in cells:
                pairs.append(
                    {
                        "route": route,
                        "left": left,
                        "right": right,
                        **pair_delta(cells[(route, left)], cells[(route, right)]),
                    }
                )

    def metric(route: str, layout: str, name: str) -> float:
        return ncu[(route, layout)]["metrics"][name]

    destination_metric = "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
    criteria = {
        "static_equal_and_ordered": bool(static.get("selected_shapes_equal") and static.get("phase_contracts_valid")),
        "source_instruction_and_sector_work_exact": len(ncu) == 6
        and all(
            item["metrics"]["smsp__inst_executed_op_ldgsts.sum"] == 256
            and item["metrics"]["sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum"] == 8192
            for item in ncu.values()
        ),
        "cache_all_wavefront_projection_transfers": len(ncu) == 6
        and [metric("l1", layout, destination_metric) for layout in LAYOUTS]
        == [2048, 4096, 8192],
        "bypass_wavefront_counter_saturates": len(ncu) == 6
        and [metric("l2", layout, destination_metric) for layout in LAYOUTS]
        == [2048, 8192, 8192],
        "issue_frontier_layout_invariant": len(cells) == 6
        and len(
            {
                value["issue_cycles"]
                for cell in cells.values()
                for value in cell["repeats"]
            }
        )
        == 1,
        "all_drain_steps_positive": len(pairs) == 6
        and all(pair["positive_repeats"] == REPEATS for pair in pairs),
        "equal_bypass_wavefront_count_has_nonzero_drain_delta": any(
            pair["route"] == "l2"
            and pair["left"] == "conflict"
            and pair["right"] == "severe"
            and pair["median_per_group"] > 3.0
            for pair in pairs
        ),
    }
    if not all(criteria.values()):
        errors.append("one or more projection-discriminator criteria failed")

    binary = REPO_ROOT / "workloads/research_probes/async_source_only/async_visibility_phase"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "probe_binary": {"path": str(binary), "sha256": sha256(binary)},
        "static": static,
        "direct_cells": [cells[key] for key in sorted(cells)],
        "ncu_cells": [ncu[key] for key in sorted(ncu)],
        "pairs": pairs,
        "criteria": criteria,
        "errors": errors,
        "status": "pass_projection_discriminator_no_timing_promotion"
        if not errors
        else "invalid_projection_discriminator",
        "decision": "retain_typed_tail_projection_requires_multiple_dimensions",
        "rationale": (
            "Cache-all follows the source-cohort bank projection. Bypass conflict and "
            "severe have the same hardware destination-wavefront count but different "
            "drain time, so neither that counter nor the cache-all projection is a "
            "sufficient scalar bypass completion coordinate."
        ),
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args.root, args.output)
    except (AnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"bypass projection analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
