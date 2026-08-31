#!/usr/bin/env python3
"""Reduce the four-layout multidimensional bypass-topology milestone."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


LAYOUTS = ("compact", "conflict", "skew", "severe")
LAYOUT_INDEX = {layout: index for index, layout in enumerate(LAYOUTS)}
DEPTHS = (1, 2, 4, 8)
REPEATS = 9
GROUPS = 8192
REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY = {
    "compact": {"lines": 1, "bank": 1, "bypass": 1, "penalty": 0},
    "conflict": {"lines": 2, "bank": 2, "bypass": 4, "penalty": 14},
    "skew": {"lines": 3, "bank": 1, "bypass": 3, "penalty": 8},
    "severe": {"lines": 4, "bank": 4, "bypass": 4, "penalty": 18},
}
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


def phase_function(route: str, layout: str) -> str:
    bypass = 0 if route == "l1" else 1
    return (
        f"_Z23visibility_phase_kernelILi4ELb{bypass}ELb0"
        f"ELi{LAYOUT_INDEX[layout]}EEvPKciP11PhaseResult"
    )


def source_function(route: str, layout: str) -> str:
    bypass = 0 if route == "l1" else 1
    return f"_Z18source_only_kernelILi4ELb{bypass}ELb0ELi{LAYOUT_INDEX[layout]}EEvPKci"


def direct_cell(root: Path, route: str, layout: str, depth: int) -> dict[str, Any]:
    base = root / "direct" / f"async_bypass_multidim_{route}_{layout}_d{depth}_g{GROUPS}"
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
            raise AnalysisError(f"{path}: direct result contract mismatch")
        repeats.append(value)
    return {
        "route": route,
        "layout": layout,
        "depth": depth,
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
        / f"async_bypass_multidim_{route}_{layout}_d4_g64"
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


def load_resources(path: Path) -> dict[str, dict[str, int]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    return {
        match.group("name"): {
            "registers": int(match.group("registers")),
            "stack": int(match.group("stack")),
            "shared": int(match.group("shared")),
            "local": int(match.group("local")),
        }
        for match in RESOURCE_RE.finditer(text)
    }


def static_report(root: Path) -> dict[str, Any]:
    phase_path = root / "static/phase/control_bits.json"
    source_path = root / "static/source_only/control_bits.json"
    phase = read_json(phase_path)
    source = read_json(source_path)
    phase_resources = load_resources(root / "static/phase/resources.txt")
    source_resources = load_resources(root / "static/source_only/resources.txt")
    selected_phase: dict[str, Any] = {}
    selected_source: dict[str, Any] = {}
    phase_contracts: dict[str, Any] = {}
    for route in ("l1", "l2"):
        for layout in LAYOUTS:
            phase_name = phase_function(route, layout)
            source_name = source_function(route, layout)
            phase_rows = phase["functions"].get(phase_name)
            source_rows = source["functions"].get(source_name)
            if not isinstance(phase_rows, list) or not isinstance(source_rows, list):
                raise AnalysisError(f"static evidence missing {route}/{layout}")
            phase_stats = phase["summary"]["functions"][phase_name]
            source_stats = source["summary"]["functions"][source_name]
            selected_phase[phase_name] = {
                "instructions": phase_stats["instructions"],
                "stalls": phase_stats["stall_cycles_static_sum"],
                **phase_resources[phase_name],
            }
            selected_source[source_name] = {
                "instructions": source_stats["instructions"],
                "stalls": source_stats["stall_cycles_static_sum"],
                **source_resources[source_name],
            }
            clocks = [row for row in phase_rows if "CLOCKLO" in str(row.get("opcode_text", ""))]
            copies = [row for row in phase_rows if str(row.get("opcode_text", "")).startswith("LDGSTS")]
            commit = [row for row in phase_rows if row.get("opcode_text") == "LDGDEPBAR"]
            wait = [row for row in phase_rows if str(row.get("opcode_text", "")).startswith("DEPBAR.LE")]
            stores = [row for row in phase_rows if str(row.get("opcode_text", "")).startswith("STG")]
            if not (len(clocks) == 3 and len(copies) == 4 and len(commit) == 1 and len(wait) == 1 and stores):
                raise AnalysisError(f"phase static cardinality failed for {route}/{layout}")
            read_mask = 1 << copies[-1]["control"]["read_barrier"]
            phase_contracts[phase_name] = {
                "ordered": clocks[0]["pc"] < copies[0]["pc"]
                and copies[-1]["pc"] < clocks[1]["pc"] < commit[0]["pc"]
                and commit[0]["pc"] < wait[0]["pc"] < clocks[2]["pc"]
                and clocks[2]["pc"] < min(store["pc"] for store in stores),
                "issue_clock_masks_read_only": all(
                    clock["control"]["wait_mask"] in (0, read_mask)
                    for clock in clocks[:2]
                ),
                "post_wait_mask_zero": clocks[2]["control"]["wait_mask"] == 0,
            }
            forbidden = [
                row["opcode_text"]
                for row in source_rows
                if str(row.get("opcode_text", "")).startswith(("STG", "ATOM", "RED", "BAR.SYNC"))
            ]
            if forbidden:
                raise AnalysisError(f"source-only target has forbidden ops: {route}/{layout}: {forbidden}")
    phase_shapes_equal = all(
        selected_phase[phase_function(route, layout)]
        == selected_phase[phase_function(route, "compact")]
        for route in ("l1", "l2")
        for layout in LAYOUTS
    )
    source_resources_equal = all(
        all(
            selected_source[source_function(route, layout)][field]
            == selected_source[source_function(route, "compact")][field]
            for field in ("instructions", "registers", "stack", "shared", "local")
        )
        for route in ("l1", "l2")
        for layout in LAYOUTS
    )
    return {
        "phase_control_sha256": sha256(phase_path),
        "source_control_sha256": sha256(source_path),
        "phase_function_count": len(phase["functions"]),
        "source_function_count": len(source["functions"]),
        "phase_decoded": phase["summary"]["total_decoded"],
        "source_decoded": source["summary"]["total_decoded"],
        "phase_unknown": phase["summary"]["total_unknown_encoding"],
        "source_unknown": source["summary"]["total_unknown_encoding"],
        "phase_shapes_equal": phase_shapes_equal,
        "source_resources_equal": source_resources_equal,
        "source_static_stall_delta_from_compact": {
            f"{route}_{layout}": selected_source[source_function(route, layout)]["stalls"]
            - selected_source[source_function(route, "compact")]["stalls"]
            for route in ("l1", "l2")
            for layout in LAYOUTS
        },
        "phase_contracts_valid": all(all(value.values()) for value in phase_contracts.values()),
        "selected_phase": selected_phase,
        "selected_source": selected_source,
    }


def replay_pair(root: Path, cell: str) -> tuple[dict[str, Any], str]:
    first = root / "simulator" / cell / "replay-1.json"
    second = root / "simulator" / cell / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise AnalysisError(f"missing replay twins for {cell}")
    if first.read_bytes() != second.read_bytes():
        raise AnalysisError(f"replay twins differ for {cell}")
    return read_json(first), sha256(first)


def expected_observation(layout: str) -> dict[str, int]:
    copies = 64 * 4
    cohorts = copies * 8
    topology = TOPOLOGY[layout]
    penalty = topology["penalty"]
    return {
        "copies_observed": copies,
        "destination_wavefronts_observed": cohorts * topology["bank"],
        "bypass_wavefronts_observed": cohorts * topology["bypass"],
        "source_cohorts_observed": cohorts,
        "destination_lines_observed": cohorts * topology["lines"],
        "max_bank_wavefronts_sum": copies * topology["bank"],
        "max_bypass_wavefronts_sum": copies * topology["bypass"],
        "topology_candidate_penalty_cycles": copies * penalty,
    }


def simulator_cell(root: Path, route: str, layout: str) -> dict[str, Any]:
    cell = f"async_bypass_multidim_{route}_{layout}_d4_g64"
    document, digest = replay_pair(root, cell)
    launch = document["launches"][0]
    readiness = launch["timing"]["readiness"]
    observation = readiness.get("bypass_async_copies")
    if route == "l1":
        if readiness["schema_version"] != 3 or observation is not None:
            raise AnalysisError(f"{cell}: cache-all observation/schema changed")
    else:
        if readiness["schema_version"] != 15 or not isinstance(observation, dict):
            raise AnalysisError(f"{cell}: missing schema-15 observation")
        for field, expected in expected_observation(layout).items():
            if observation.get(field) != expected:
                raise AnalysisError(
                    f"{cell}: {field}={observation.get(field)!r}, expected {expected}"
                )
        if launch["counters"]["unmodeled_tails"].get("async_copy_shared_visibility") is not True:
            raise AnalysisError(f"{cell}: typed bypass visibility tail was cleared")
    return {
        "cell": cell,
        "route": route,
        "layout": layout,
        "terminal_modeled_event_time": launch["timing"]["terminal_modeled_event_time"],
        "completion_frontier": launch["timing"]["completion_frontier"],
        "observation": observation,
        "replay_sha256": digest,
    }


def regression_report(root: Path, baseline_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for depth in (4, 8):
        for route in ("l1", "dram"):
            for layout in ("compact", "conflict"):
                cell = f"async_source_visibility_{route}_{layout}_d{depth}_g64"
                current_path = root / "regression" / cell / "replay-1.json"
                twin_path = root / "regression" / cell / "replay-2.json"
                baseline_path = baseline_root / "simulator" / cell / "replay-1.json"
                if current_path.read_bytes() != twin_path.read_bytes():
                    raise AnalysisError(f"{cell}: regression twins differ")
                current = read_json(current_path)
                baseline = read_json(baseline_path)
                normalized = deepcopy(current)
                readiness = normalized["launches"][0]["timing"]["readiness"]
                observation = readiness.pop("bypass_async_copies", None)
                if observation is not None:
                    readiness["schema_version"] = baseline["launches"][0]["timing"][
                        "readiness"
                    ]["schema_version"]
                if normalized != baseline:
                    raise AnalysisError(f"{cell}: non-observational regression output changed")
                cells.append(
                    {
                        "cell": cell,
                        "cache_all_byte_identical": route == "l1"
                        and current_path.read_bytes() == baseline_path.read_bytes(),
                        "observation_present": observation is not None,
                    }
                )
    return {"cells": cells, "all_normalized_unchanged": len(cells) == 8}


def paired_wait_delta(
    compact: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    values = [
        (right["wait_cycles"] - left["wait_cycles"]) / GROUPS
        for left, right in zip(compact["repeats"], treatment["repeats"], strict=True)
    ]
    return {
        "per_repeat_per_group": values,
        "median_per_group": statistics.median(values),
        "minimum_per_group": min(values),
        "maximum_per_group": max(values),
        "positive_repeats": sum(value > 0 for value in values),
    }


def run(root: Path, baseline_root: Path, output: Path) -> int:
    root = root.resolve()
    baseline_root = baseline_root.resolve()
    errors: list[str] = []
    try:
        static = static_report(root)
    except (AnalysisError, KeyError) as error:
        static = {"error": str(error)}
        errors.append(str(error))
    direct: dict[tuple[str, str, int], dict[str, Any]] = {}
    for depth in DEPTHS:
        for layout in LAYOUTS:
            try:
                direct[("l2", layout, depth)] = direct_cell(root, "l2", layout, depth)
            except AnalysisError as error:
                errors.append(str(error))
    for layout in LAYOUTS:
        try:
            direct[("l1", layout, 4)] = direct_cell(root, "l1", layout, 4)
        except AnalysisError as error:
            errors.append(str(error))
    ncu: dict[tuple[str, str], dict[str, Any]] = {}
    simulator: list[dict[str, Any]] = []
    for route in ("l1", "l2"):
        for layout in LAYOUTS:
            try:
                ncu[(route, layout)] = ncu_cell(root, route, layout)
                simulator.append(simulator_cell(root, route, layout))
            except (AnalysisError, KeyError, IndexError, TypeError) as error:
                errors.append(str(error))
    try:
        regression = regression_report(root, baseline_root)
    except (AnalysisError, OSError, KeyError, IndexError) as error:
        regression = {"error": str(error), "all_normalized_unchanged": False}
        errors.append(str(error))

    live_pairs: list[dict[str, Any]] = []
    candidate_errors: list[dict[str, Any]] = []
    for depth in DEPTHS:
        compact = direct.get(("l2", "compact", depth))
        if compact is None:
            continue
        for layout in LAYOUTS[1:]:
            treatment = direct.get(("l2", layout, depth))
            if treatment is None:
                continue
            delta = paired_wait_delta(compact, treatment)
            error = delta["median_per_group"] - TOPOLOGY[layout]["penalty"]
            live_pairs.append({"depth": depth, "layout": layout, **delta})
            candidate_errors.append(
                {
                    "depth": depth,
                    "layout": layout,
                    "candidate_penalty": TOPOLOGY[layout]["penalty"],
                    "observed_delta": delta["median_per_group"],
                    "signed_error": error,
                    "absolute_error": abs(error),
                }
            )

    destination_metric = "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
    expected_ncu = {
        "l1": {"compact": 2048, "conflict": 4096, "skew": 2048, "severe": 8192},
        "l2": {"compact": 2048, "conflict": 8192, "skew": 6144, "severe": 8192},
    }
    criteria = {
        "static_contracts_equal": bool(
            static.get("phase_shapes_equal")
            and static.get("source_resources_equal")
            and static.get("phase_contracts_valid")
            and static.get("phase_unknown") == 0
            and static.get("source_unknown") == 0
        ),
        "live_issue_frontier_invariant": len(direct) == 20
        and all(
            [repeat["issue_cycles"] for repeat in direct[(route, layout, depth)]["repeats"]]
            == [
                repeat["issue_cycles"]
                for repeat in direct[(route, "compact", depth)]["repeats"]
            ]
            for route, depth in (("l1", 4), *(("l2", depth) for depth in DEPTHS))
            for layout in LAYOUTS
        ),
        "live_layout_deltas_positive": len(live_pairs) == 12
        and all(pair["positive_repeats"] == REPEATS for pair in live_pairs),
        "ncu_source_work_exact": len(ncu) == 8
        and all(
            cell["metrics"]["smsp__inst_executed_op_ldgsts.sum"] == 256
            and cell["metrics"]["sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum"]
            == 8192
            for cell in ncu.values()
        ),
        "trace_proxy_matches_ncu_wavefronts": len(ncu) == 8
        and all(
            ncu[(route, layout)]["metrics"][destination_metric]
            == expected_ncu[route][layout]
            for route in ("l1", "l2")
            for layout in LAYOUTS
        ),
        "schema15_topology_complete": len(simulator) == 8
        and all(
            cell["observation"] is not None
            for cell in simulator
            if cell["route"] == "l2"
        ),
        "replay_regression_unchanged": regression.get("all_normalized_unchanged") is True,
        "candidate_fails_depth_transfer": any(
            item["depth"] != 4 and item["absolute_error"] > 3.0
            for item in candidate_errors
        ),
    }
    if not all(criteria.values()):
        errors.append("one or more multidimensional acceptance criteria failed")

    simulator_binary = REPO_ROOT / "sim/target/release/sim-run"
    phase_binary = REPO_ROOT / "workloads/research_probes/async_source_only/async_visibility_phase"
    source_binary = REPO_ROOT / "workloads/research_probes/async_source_only/async_source_only"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "baseline_root": str(baseline_root),
        "binaries": {
            "simulator": {"path": str(simulator_binary), "sha256": sha256(simulator_binary)},
            "phase": {"path": str(phase_binary), "sha256": sha256(phase_binary)},
            "source_only": {"path": str(source_binary), "sha256": sha256(source_binary)},
        },
        "topology": TOPOLOGY,
        "static": static,
        "direct_cells": [direct[key] for key in sorted(direct)],
        "ncu_cells": [ncu[key] for key in sorted(ncu)],
        "simulator_cells": simulator,
        "regression": regression,
        "live_pairs": live_pairs,
        "candidate_errors": candidate_errors,
        "criteria": criteria,
        "errors": errors,
        "status": "pass_multidimensional_observation_no_timing_promotion"
        if not errors
        else "invalid_multidimensional_milestone",
        "decision": "retain_typed_tail_candidate_requires_pipelined_service_evidence",
        "rationale": (
            "The trace-derived line-fanout/bank product proxy matches all four live "
            "wavefront counters and the depth-4 local penalty, but a latency-only "
            "per-copy candidate fails untouched depths. Timing therefore remains "
            "underidentified beyond the schema-15 observation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "status": report["status"], "cells": len(direct)},
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
        print(f"multidimensional bypass analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
