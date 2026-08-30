#!/usr/bin/env python3
"""Reduce the three-frontier async visibility phase assay."""

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


ROUTES = ("l1", "l2", "dram")
LAYOUTS = ("compact", "conflict")
DEPTHS = (1, 2, 4, 8)
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


def direct_cell_name(route: str, layout: str, depth: int) -> str:
    return f"async_visibility_phase_{route}_{layout}_d{depth}_g{GROUPS}"


def function_name(route: str, layout: str, depth: int) -> str:
    bypass = 0 if route == "l1" else 1
    streaming = 1 if route == "dram" else 0
    conflict = 1 if layout == "conflict" else 0
    return (
        f"_Z23visibility_phase_kernelILi{depth}ELb{bypass}ELb{streaming}"
        f"ELb{conflict}EEvPKciP11PhaseResult"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def direct_cell(root: Path, route: str, layout: str, depth: int) -> dict[str, Any]:
    base = root / "direct" / direct_cell_name(route, layout, depth)
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
            "depth": depth,
            "groups": GROUPS,
            "sum_consistent": True,
            "result_store_after_intervals": True,
            "correct": True,
        }
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise AnalysisError(f"{path}: result identity/contract mismatch")
        for field in ("issue_cycles", "wait_cycles", "total_cycles"):
            if not isinstance(value.get(field), int) or value[field] <= 0:
                raise AnalysisError(f"{path}: invalid {field}")
        if value["issue_cycles"] + value["wait_cycles"] != value["total_cycles"]:
            raise AnalysisError(f"{path}: phase sum mismatch")
        repeats.append(value)
    medians = {
        field: statistics.median(value[field] for value in repeats)
        for field in ("issue_cycles", "wait_cycles", "total_cycles")
    }
    relative_ranges = {
        field: (
            max(value[field] for value in repeats) - min(value[field] for value in repeats)
        )
        / statistics.mean(value[field] for value in repeats)
        for field in ("issue_cycles", "wait_cycles", "total_cycles")
    }
    return {
        "route": route,
        "layout": layout,
        "depth": depth,
        "repeats": repeats,
        "median": medians,
        "median_per_group": {field: value / GROUPS for field, value in medians.items()},
        "relative_range": relative_ranges,
        "result_sha256": [sha256(path) for path in paths],
    }


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
    if summary.get("total_unknown_encoding") != 0:
        raise AnalysisError("static sidecar contains unknown encodings")
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
    phase_contracts: dict[str, Any] = {}
    for route in ROUTES:
        for layout in LAYOUTS:
            for depth in DEPTHS:
                name = function_name(route, layout, depth)
                rows = functions.get(name)
                stats = summary.get("functions", {}).get(name)
                resource = resources.get(name)
                if not isinstance(rows, list) or not isinstance(stats, Mapping) or resource is None:
                    raise AnalysisError(f"static evidence is missing {name}")
                clocks = [row for row in rows if "CLOCKLO" in str(row.get("opcode_text", ""))]
                copies = [row for row in rows if str(row.get("opcode_text", "")).startswith("LDGSTS")]
                commits = [row for row in rows if row.get("opcode_text") == "LDGDEPBAR"]
                waits = [row for row in rows if str(row.get("opcode_text", "")).startswith("DEPBAR.LE")]
                stores = [row for row in rows if str(row.get("opcode_text", "")).startswith("STG")]
                if not (len(clocks) == 3 and len(copies) == depth and len(commits) == 1 and len(waits) == 1 and stores):
                    raise AnalysisError(f"{name}: phase instruction cardinality mismatch")
                last_copy_barrier = copies[-1]["control"]["read_barrier"]
                expected_issue_wait = 1 << last_copy_barrier
                ordered = (
                    clocks[0]["pc"] < copies[0]["pc"]
                    and copies[-1]["pc"] < clocks[1]["pc"] < commits[0]["pc"]
                    and commits[0]["pc"] < waits[0]["pc"] < clocks[2]["pc"]
                    and clocks[2]["pc"] < min(store["pc"] for store in stores)
                )
                phase_contracts[name] = {
                    "ordered": ordered,
                    "pre_issue_clock_wait_mask": clocks[0]["control"]["wait_mask"],
                    "post_issue_clock_wait_mask": clocks[1]["control"]["wait_mask"],
                    "post_issue_expected_read_barrier_mask": expected_issue_wait,
                    "clock_bracket_masks_read_barrier_only": (
                        clocks[0]["control"]["wait_mask"] in (0, expected_issue_wait)
                        and clocks[1]["control"]["wait_mask"] in (0, expected_issue_wait)
                    ),
                    "post_wait_clock_wait_mask": clocks[2]["control"]["wait_mask"],
                    "stores_after_post_wait_clock": all(
                        store["pc"] > clocks[2]["pc"] for store in stores
                    ),
                }
                selected[name] = {
                    "instructions": stats.get("instructions"),
                    "stall_cycles_static_sum": stats.get("stall_cycles_static_sum"),
                    **resource,
                }
    pair_shape_equal = True
    for route in ROUTES:
        for depth in DEPTHS:
            compact = selected[function_name(route, "compact", depth)]
            conflict = selected[function_name(route, "conflict", depth)]
            pair_shape_equal &= compact == conflict
    phase_contract_valid = all(
        item["ordered"]
        and item["clock_bracket_masks_read_barrier_only"]
        and item["post_wait_clock_wait_mask"] == 0
        and item["stores_after_post_wait_clock"]
        for item in phase_contracts.values()
    )
    return {
        "control_sha256": sha256(control_path),
        "resources_sha256": sha256(resources_path),
        "function_count": len(functions),
        "decoded": summary.get("total_decoded"),
        "unknown_encoding": summary.get("total_unknown_encoding"),
        "selected_functions": selected,
        "phase_contracts": phase_contracts,
        "pair_shape_equal": pair_shape_equal,
        "phase_contract_valid": phase_contract_valid,
    }


def ncu_l2_control(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layout in LAYOUTS:
        path = (
            root
            / "ncu"
            / f"async_visibility_phase_l2_{layout}_d4_g64"
            / "repeat-1/ncu.csv"
        )
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row.get("Kernel Name")]
        except (OSError, csv.Error) as error:
            raise AnalysisError(f"cannot read NCU CSV {path}: {error}") from error
        if len(rows) != 1:
            raise AnalysisError(f"{path}: expected one kernel row, got {len(rows)}")
        try:
            metrics = {metric: float(rows[0][metric]) for metric in NCU_METRICS}
        except (KeyError, ValueError) as error:
            raise AnalysisError(f"{path}: missing/non-numeric metric: {error}") from error
        result[layout] = {"metrics": metrics, "csv_sha256": sha256(path)}
    compact = result["compact"]["metrics"]
    conflict = result["conflict"]["metrics"]
    result["contract"] = {
        "source_work_equal": all(
            compact[metric] == conflict[metric]
            for metric in (
                "smsp__inst_executed_op_ldgsts.sum",
                "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
                "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
                "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
            )
        ),
        "all_source_requests_hit_l2": all(
            item["metrics"]["lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"] == 8192
            and item["metrics"]["lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"] == 0
            for item in (result["compact"], result["conflict"])
        ),
        "destination_wavefronts_separated": (
            compact["sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"]
            == 2048
            and conflict[
                "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
            ]
            == 8192
        ),
    }
    return result


def paired_delta(compact: Mapping[str, Any], conflict: Mapping[str, Any], field: str) -> dict[str, Any]:
    values = [
        right[field] - left[field]
        for left, right in zip(compact["repeats"], conflict["repeats"], strict=True)
    ]
    return {
        "per_repeat": values,
        "per_group": [value / GROUPS for value in values],
        "median": statistics.median(values),
        "median_per_group": statistics.median(values) / GROUPS,
        "minimum": min(values),
        "maximum": max(values),
        "positive_repeats": sum(value > 0 for value in values),
        "zero_repeats": sum(value == 0 for value in values),
    }


def run(root: Path, output: Path) -> int:
    root = root.resolve()
    errors: list[str] = []
    try:
        static = static_report(root)
    except AnalysisError as error:
        static = {"error": str(error)}
        errors.append(str(error))
    try:
        ncu = ncu_l2_control(root)
    except AnalysisError as error:
        ncu = {"error": str(error)}
        errors.append(str(error))

    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    for route in ROUTES:
        for depth in DEPTHS:
            for layout in LAYOUTS:
                try:
                    cells[(route, layout, depth)] = direct_cell(
                        root, route, layout, depth
                    )
                except AnalysisError as error:
                    errors.append(str(error))

    pairs: list[dict[str, Any]] = []
    for route in ROUTES:
        for depth in DEPTHS:
            compact = cells.get((route, "compact", depth))
            conflict = cells.get((route, "conflict", depth))
            if compact is None or conflict is None:
                continue
            pairs.append(
                {
                    "route": route,
                    "depth": depth,
                    "issue_delta": paired_delta(compact, conflict, "issue_cycles"),
                    "wait_delta": paired_delta(compact, conflict, "wait_cycles"),
                    "total_delta": paired_delta(compact, conflict, "total_cycles"),
                }
            )

    l1_pairs = [pair for pair in pairs if pair["route"] == "l1"]
    l2_pairs = [pair for pair in pairs if pair["route"] == "l2"]
    criteria = {
        "static_pair_and_clock_contract": bool(
            static.get("pair_shape_equal") and static.get("phase_contract_valid")
        ),
        "l2_source_control_exact": bool(ncu.get("contract"))
        and all(ncu["contract"].values()),
        "issue_layout_delta_zero": bool(pairs)
        and all(pair["issue_delta"]["zero_repeats"] == REPEATS for pair in pairs),
        "drain_layout_delta_positive": bool(pairs)
        and all(pair["wait_delta"]["positive_repeats"] == REPEATS for pair in pairs),
        "total_delta_equals_drain_delta": bool(pairs)
        and all(
            pair["total_delta"]["per_repeat"] == pair["wait_delta"]["per_repeat"]
            for pair in pairs
        ),
        "cache_all_depth_transfer": len(l1_pairs) == len(DEPTHS)
        and all(
            abs(pair["wait_delta"]["median_per_group"] - 16.0 * pair["depth"])
            <= 0.02
            for pair in l1_pairs
        ),
        "bypass_l2_not_group_linear": len(l2_pairs) == len(DEPTHS)
        and max(pair["wait_delta"]["median_per_group"] for pair in l2_pairs)
        < 2.0 * min(pair["wait_delta"]["median_per_group"] for pair in l2_pairs),
    }
    if not all(criteria.values()):
        errors.append("one or more phase-assay acceptance criteria failed")

    binary = REPO_ROOT / "workloads/research_probes/async_source_only/async_visibility_phase"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "probe_binary": {"path": str(binary), "sha256": sha256(binary)},
        "static": static,
        "ncu_l2_control": ncu,
        "cells": [cells[key] for key in sorted(cells)],
        "pairs": pairs,
        "criteria": criteria,
        "errors": errors,
        "status": "pass_bypass_drain_localized_no_group_timing_promotion"
        if not errors
        else "invalid_phase_assay",
        "decision": "retain_typed_tail_pending_per_copy_bypass_delivery_model",
        "rationale": (
            "The layout delta is absent at the LDGSTS operand-read/issue frontier and "
            "present only after DEPBAR. Cache-all scales linearly with group depth, but "
            "L2-resident bypass remains approximately constant across an eightfold depth "
            "change, so the existing group-aggregate destination rule is the wrong shape."
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
        print(f"async visibility phase analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
