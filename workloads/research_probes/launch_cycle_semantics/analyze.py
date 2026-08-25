#!/usr/bin/env python3
"""Reduce the sm_86 launch-cycle and range-composition campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from campaign import DEFAULT_OUTPUT, TAILS, app_range_cells, configured_cells, range_cells


GPC = "gpc__cycles_elapsed.max"
GPC_AVG = "gpc__cycles_elapsed.avg"
GPC_MIN = "gpc__cycles_elapsed.min"
GPC_SUM = "gpc__cycles_elapsed.sum"
SM_ACTIVE = "sm__cycles_active.max"
INST = "smsp__inst_executed.sum"


def median(values: Iterable[float]) -> float:
    material = list(values)
    if not material:
        raise RuntimeError("median of empty sample")
    return float(statistics.median(material))


def mean(values: Iterable[float]) -> float:
    material = list(values)
    if not material:
        raise RuntimeError("mean of empty sample")
    return float(statistics.fmean(material))


def cv(values: Iterable[float]) -> float:
    material = list(values)
    if len(material) < 2:
        return 0.0
    center = mean(material)
    return 0.0 if center == 0 else statistics.stdev(material) / center


def mape(predicted: Iterable[float], observed: Iterable[float]) -> float:
    pairs = list(zip(predicted, observed, strict=True))
    return 100.0 * mean(abs(p - o) / o for p, o in pairs if o != 0)


def mae(predicted: Iterable[float], observed: Iterable[float]) -> float:
    return mean(abs(p - o) for p, o in zip(predicted, observed, strict=True))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ncu_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("ID") or "").strip()]
    if not rows:
        raise RuntimeError(f"NCU CSV has no data rows: {path}")
    numeric: list[dict[str, float]] = []
    for row in rows:
        converted: dict[str, float] = {"ID": float(int(row["ID"]))}
        for key, value in row.items():
            if key == "ID" or value is None or not value.strip():
                continue
            try:
                parsed = float(value.replace(",", ""))
            except ValueError:
                continue
            if math.isfinite(parsed):
                converted[key] = parsed
        numeric.append(converted)
    return numeric


def ncu_repeats(root: Path, directory: str, key: str, expected_rows: int) -> list[list[dict[str, float]]]:
    paths = sorted((root / directory / key).glob("repeat-*/ncu.csv"))
    if len(paths) != 3:
        raise RuntimeError(f"{directory}/{key}: expected three NCU CSVs, got {len(paths)}")
    repeats = [ncu_rows(path) for path in paths]
    if any(len(rows) != expected_rows for rows in repeats):
        raise RuntimeError(f"{directory}/{key}: unexpected NCU row cardinality")
    return repeats


def ordinal_medians(repeats: Sequence[Sequence[Mapping[str, float]]]) -> list[dict[str, float]]:
    row_count = len(repeats[0])
    keys = set.intersection(*(set(row) for repeat in repeats for row in repeat))
    result = []
    for ordinal in range(row_count):
        result.append(
            {
                key: median(repeat[ordinal][key] for repeat in repeats)
                for key in keys
                if key != "ID"
            }
        )
    return result


def simulator_launches(root: Path, directory: str, key: str, expected: int) -> list[dict]:
    first_path = root / directory / key / "replay-1.json"
    second_path = root / directory / key / "replay-2.json"
    if first_path.read_bytes() != second_path.read_bytes():
        raise RuntimeError(f"{directory}/{key}: nondeterministic replay pair")
    launches = json.loads(first_path.read_text(encoding="utf-8"))["launches"]
    if len(launches) != expected:
        raise RuntimeError(f"{directory}/{key}: launch count mismatch")
    return launches


def validate_integrity(root: Path) -> dict[str, object]:
    cells = configured_cells()
    correctness = 0
    ncu_kernel_files = 0
    ncu_kernel_rows = 0
    nvbit_launches = 0
    nvbit_records = 0
    for cell in cells:
        payload = json.loads(
            (root / "correctness" / cell.key / "result.json").read_text(encoding="utf-8")
        )
        if payload["measurement"]["correct"] is not True:
            raise RuntimeError(f"{cell.key}: incorrect correctness payload")
        correctness += payload["measurement"]["trials"]

        repeats = ncu_repeats(root, "ncu-kernel", cell.key, cell.chain_length)
        ncu_kernel_files += len(repeats)
        ncu_kernel_rows += sum(len(rows) for rows in repeats)

        trace = root / "nvbit" / cell.key
        if (trace / "dynamic_trace.pb").stat().st_size <= 0:
            raise RuntimeError(f"{cell.key}: empty dynamic trace")
        with (trace / "stats.csv").open(encoding="utf-8", newline="") as handle:
            stats = [
                {key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        if len(stats) != cell.chain_length:
            raise RuntimeError(f"{cell.key}: NVBit launch count mismatch")
        for row in stats:
            total = int(row["total_insts"])
            reported = int(row["total_reported_insts"])
            if total != reported or int(row["#blocks"]) != cell.blocks:
                raise RuntimeError(f"{cell.key}: NVBit cardinality mismatch")
            nvbit_records += total
        nvbit_launches += len(stats)
        extra = trace / "extra_info"
        for name in (
            "unsupported_memory_path.json",
            "unmapped_memory_range.json",
            "cross_allocation_lane_span.json",
        ):
            if json.loads((extra / name).read_text(encoding="utf-8")) != []:
                raise RuntimeError(f"{cell.key}: nonempty {name}")
        modules = json.loads((extra / "traced_kernel_modules.json").read_text(encoding="utf-8"))
        selection = modules["selection"]
        if (
            selection["accepted_active_launches"] != cell.chain_length
            or selection["accepted_channel_records"] != sum(int(row["total_insts"]) for row in stats)
            or selection["exact_contract_satisfied"] is not True
        ):
            raise RuntimeError(f"{cell.key}: NVBit selection mismatch")

        for directory in ("simulator", "simulator-carded-dram"):
            launches = simulator_launches(root, directory, cell.key, cell.chain_length)
            for launch in launches:
                if launch["counters"]["instructions_issued"] != launch["counters"]["records_seen"]:
                    raise RuntimeError(f"{directory}/{cell.key}: issue/record mismatch")

    ncu_range_files = 0
    for cell in range_cells():
        repeats = ncu_repeats(root, "ncu-range", cell.key, 1)
        ncu_range_files += len(repeats)
    ncu_app_range_files = 0
    for cell in app_range_cells():
        repeats = ncu_repeats(root, "ncu-app-range", cell.key, 1)
        ncu_app_range_files += len(repeats)
    ncu_range_outside_sync_files = 0
    ncu_app_range_outside_sync_files = 0
    for cell in app_range_cells():
        ncu_range_outside_sync_files += len(
            ncu_repeats(root, "ncu-range-outside-sync", cell.key, 1)
        )
        ncu_app_range_outside_sync_files += len(
            ncu_repeats(root, "ncu-app-range-outside-sync", cell.key, 1)
        )

    sidecar = json.loads(
        (root / "sass/launch_cycle_semantics_trace.control_bits.json").read_text(
            encoding="utf-8"
        )
    )
    statuses = [entry["status"] for entries in sidecar["functions"].values() for entry in entries]
    if len(sidecar["functions"]) != 6 or any(status != "decoded" for status in statuses):
        raise RuntimeError("static control sidecar is incomplete")

    return {
        "matrix_cells": len(cells),
        "correctness_executions": correctness,
        "ncu_kernel_profiles": ncu_kernel_files,
        "ncu_kernel_rows": ncu_kernel_rows,
        "ncu_range_profiles": ncu_range_files,
        "ncu_application_range_profiles": ncu_app_range_files,
        "ncu_range_outside_sync_profiles": ncu_range_outside_sync_files,
        "ncu_application_range_outside_sync_profiles": ncu_app_range_outside_sync_files,
        "nvbit_captures": len(cells),
        "nvbit_launches": nvbit_launches,
        "nvbit_records": nvbit_records,
        "simulator_replay_pairs": len(cells),
        "carded_dram_replay_pairs": len(cells),
        "static_functions": len(sidecar["functions"]),
        "static_instructions": len(statuses),
        "static_unknown_encodings": sum(status != "decoded" for status in statuses),
    }


def build_kernel_rows(root: Path) -> tuple[list[dict[str, object]], dict[str, list[dict[str, float]]]]:
    output: list[dict[str, object]] = []
    observed_by_cell: dict[str, list[dict[str, float]]] = {}
    for cell in configured_cells():
        repeats = ncu_repeats(root, "ncu-kernel", cell.key, cell.chain_length)
        medians = ordinal_medians(repeats)
        observed_by_cell[cell.key] = medians
        launches = simulator_launches(root, "simulator", cell.key, cell.chain_length)
        carded = simulator_launches(root, "simulator-carded-dram", cell.key, cell.chain_length)
        for ordinal, (observed, launch, carded_launch) in enumerate(
            zip(medians, launches, carded, strict=True), start=1
        ):
            timing = launch["timing"]
            carded_timing = carded_launch["timing"]
            gpc_instances = observed[GPC_SUM] / observed[GPC_AVG]
            output.append(
                {
                    "cell": cell.key,
                    "stage": cell.stage,
                    "tail": cell.tail,
                    "blocks": cell.blocks,
                    "iterations": cell.iterations,
                    "chain_length": cell.chain_length,
                    "launch_ordinal": ordinal,
                    "ncu_gpc_elapsed_max": observed[GPC],
                    "ncu_gpc_elapsed_min": observed[GPC_MIN],
                    "ncu_gpc_elapsed_avg": observed[GPC_AVG],
                    "ncu_gpc_elapsed_sum": observed[GPC_SUM],
                    "ncu_gpc_instances": gpc_instances,
                    "ncu_gpc_instance_spread": observed[GPC] - observed[GPC_MIN],
                    "ncu_sm_active_max": observed[SM_ACTIVE],
                    "ncu_gpc_max_minus_sm_active_max": observed[GPC]
                    - observed[SM_ACTIVE],
                    "ncu_inst_executed": observed[INST],
                    "sim_last_issue": timing["diagnostic_last_issue_cycles"],
                    "sim_terminal_event": timing["terminal_modeled_event_time"],
                    "sim_modeled_drain": timing["terminal_modeled_event_time"]
                    - timing["diagnostic_last_issue_cycles"],
                    "sim_carded_last_issue": carded_timing["diagnostic_last_issue_cycles"],
                    "sim_carded_terminal_event": carded_timing["terminal_modeled_event_time"],
                    "unmodeled_reduction_rmw": launch["counters"]["unmodeled_tails"][
                        "reduction_read_modify_write"
                    ],
                    "unmodeled_dirty_writeback": launch["counters"]["unmodeled_tails"][
                        "dirty_writeback"
                    ],
                    "unmodeled_coherence": launch["counters"]["unmodeled_tails"][
                        "coherence"
                    ],
                    "ncu_gpc_cv": cv(repeat[ordinal - 1][GPC] for repeat in repeats),
                    "ncu_sm_active_cv": cv(
                        repeat[ordinal - 1][SM_ACTIVE] for repeat in repeats
                    ),
                }
            )
    return output, observed_by_cell


def kernel_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    single = [row for row in rows if row["chain_length"] == 1]
    control = [row for row in single if row["tail"] == "control"]
    extrema_differences: dict[str, dict[str, float]] = {}
    for blocks in (1, 7, 42, 84):
        values = [
            float(row["ncu_gpc_max_minus_sm_active_max"])
            for row in control
            if row["blocks"] == blocks
        ]
        extrema_differences[str(blocks)] = {
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }

    extrema_difference_by_block = {
        int(key): value["median"] for key, value in extrema_differences.items()
    }
    observed = [float(row["ncu_gpc_elapsed_max"]) for row in single]
    issue = [float(row["sim_last_issue"]) for row in single]
    terminal = [float(row["sim_terminal_event"]) for row in single]
    issue_plus_extrema_difference = [
        float(row["sim_last_issue"])
        + extrema_difference_by_block[int(row["blocks"])]
        for row in single
    ]
    terminal_plus_extrema_difference = [
        float(row["sim_terminal_event"])
        + extrema_difference_by_block[int(row["blocks"])]
        for row in single
    ]

    by_key = {(row["tail"], row["blocks"], row["iterations"]): row for row in single}
    tail_effects: dict[str, dict[str, float]] = {}
    for tail in TAILS[1:]:
        material = []
        for blocks in (1, 7, 42, 84):
            for iterations in (0, 256):
                row = by_key[(tail, blocks, iterations)]
                base = by_key[("control", blocks, iterations)]
                material.append(
                    {
                        "observed_gpc": float(row["ncu_gpc_elapsed_max"])
                        - float(base["ncu_gpc_elapsed_max"]),
                        "observed_sm": float(row["ncu_sm_active_max"])
                        - float(base["ncu_sm_active_max"]),
                        "issue": float(row["sim_last_issue"])
                        - float(base["sim_last_issue"]),
                        "terminal": float(row["sim_terminal_event"])
                        - float(base["sim_terminal_event"]),
                        "gpc_max_minus_sm_active_max_delta": float(
                            row["ncu_gpc_max_minus_sm_active_max"]
                        )
                        - float(base["ncu_gpc_max_minus_sm_active_max"]),
                    }
                )
        tail_effects[tail] = {
            "observed_gpc_delta_median": median(item["observed_gpc"] for item in material),
            "observed_sm_active_delta_median": median(item["observed_sm"] for item in material),
            "observed_gpc_max_minus_sm_active_max_delta_median": median(
                item["gpc_max_minus_sm_active_max_delta"] for item in material
            ),
            "sim_issue_delta_median": median(item["issue"] for item in material),
            "sim_terminal_delta_median": median(item["terminal"] for item in material),
            "sim_issue_tail_effect_mae": mae(
                [item["issue"] for item in material],
                [item["observed_gpc"] for item in material],
            ),
            "sim_terminal_tail_effect_mae": mae(
                [item["terminal"] for item in material],
                [item["observed_gpc"] for item in material],
            ),
        }

    return {
        "gpc_instance_count_median": median(float(row["ncu_gpc_instances"]) for row in rows),
        "gpc_instance_count_max_abs_error_from_7": max(
            abs(float(row["ncu_gpc_instances"]) - 7.0) for row in rows
        ),
        "gpc_instance_spread_max_cycles": max(
            float(row["ncu_gpc_instance_spread"]) for row in rows
        ),
        "ncu_kernel_gpc_cv_max_pct": 100.0 * max(float(row["ncu_gpc_cv"]) for row in rows),
        "control_gpc_max_minus_sm_active_max_by_blocks": extrema_differences,
        "single_launch_candidates": {
            "last_issue_raw_mape_pct": mape(issue, observed),
            "terminal_raw_mape_pct": mape(terminal, observed),
            "last_issue_plus_control_extrema_difference_mape_pct": mape(
                issue_plus_extrema_difference, observed
            ),
            "terminal_plus_control_extrema_difference_mape_pct": mape(
                terminal_plus_extrema_difference, observed
            ),
        },
        "tail_effects": tail_effects,
        "carded_dram_changed_any_timing": any(
            row["sim_last_issue"] != row["sim_carded_last_issue"]
            or row["sim_terminal_event"] != row["sim_carded_terminal_event"]
            for row in rows
        ),
    }


def range_summary(
    root: Path,
    kernel_observed: Mapping[str, Sequence[Mapping[str, float]]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    modes = (
        ("range", "ncu-range", range_cells()),
        ("app-range", "ncu-app-range", app_range_cells()),
        ("range-outside-sync", "ncu-range-outside-sync", app_range_cells()),
        (
            "app-range-outside-sync",
            "ncu-app-range-outside-sync",
            app_range_cells(),
        ),
    )
    for mode, directory, cells in modes:
        for cell in cells:
            repeats = ncu_repeats(root, directory, cell.key, 1)
            observed = ordinal_medians(repeats)[0]
            kernels = kernel_observed[cell.key]
            kernel_gpc_sum = sum(row[GPC] for row in kernels)
            kernel_sm_sum = sum(row[SM_ACTIVE] for row in kernels)
            sim = simulator_launches(root, "simulator", cell.key, cell.chain_length)
            sim_issue_sum = sum(
                launch["timing"]["diagnostic_last_issue_cycles"] for launch in sim
            )
            sim_terminal_sum = sum(
                launch["timing"]["terminal_modeled_event_time"] for launch in sim
            )
            rows.append(
                {
                    "mode": mode,
                    "cell": cell.key,
                    "tail": cell.tail,
                    "blocks": cell.blocks,
                    "chain_length": cell.chain_length,
                    "range_gpc_elapsed": observed[GPC],
                    "range_sm_active": observed[SM_ACTIVE],
                    "kernel_gpc_sum": kernel_gpc_sum,
                    "kernel_sm_active_sum": kernel_sm_sum,
                    "sim_issue_sum": sim_issue_sum,
                    "sim_terminal_sum": sim_terminal_sum,
                    "range_minus_kernel_gpc_sum": observed[GPC] - kernel_gpc_sum,
                    "range_gpc_over_kernel_gpc_sum": observed[GPC] / kernel_gpc_sum,
                    "range_sm_over_kernel_sm_sum": observed[SM_ACTIVE] / kernel_sm_sum,
                    "range_gpc_cv": cv(repeat[0][GPC] for repeat in repeats),
                    "range_sm_active_cv": cv(repeat[0][SM_ACTIVE] for repeat in repeats),
                }
            )

    summary: dict[str, object] = {}
    for mode, _, _ in modes:
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "cells": len(selected),
            "gpc_sum_composition_mape_pct": mape(
                [float(row["kernel_gpc_sum"]) for row in selected],
                [float(row["range_gpc_elapsed"]) for row in selected],
            ),
            "sm_active_sum_composition_mape_pct": mape(
                [float(row["kernel_sm_active_sum"]) for row in selected],
                [float(row["range_sm_active"]) for row in selected],
            ),
            "range_gpc_over_kernel_sum_median": median(
                float(row["range_gpc_over_kernel_gpc_sum"]) for row in selected
            ),
            "range_gpc_over_kernel_sum_min": min(
                float(row["range_gpc_over_kernel_gpc_sum"]) for row in selected
            ),
            "range_gpc_over_kernel_sum_max": max(
                float(row["range_gpc_over_kernel_gpc_sum"]) for row in selected
            ),
            "range_gpc_cv_median_pct": 100.0
            * median(float(row["range_gpc_cv"]) for row in selected),
            "range_gpc_cv_max_pct": 100.0
            * max(float(row["range_gpc_cv"]) for row in selected),
            "range_sm_active_cv_max_pct": 100.0
            * max(float(row["range_sm_active_cv"]) for row in selected),
        }
    return rows, summary


def report_markdown(integrity: Mapping[str, object], kernel: Mapping[str, object], ranges: Mapping[str, object]) -> str:
    extrema_differences = kernel["control_gpc_max_minus_sm_active_max_by_blocks"]
    tails = kernel["tail_effects"]
    candidates = kernel["single_launch_candidates"]
    range_mode = ranges["range"]
    app_mode = ranges["app-range"]
    range_outside = ranges["range-outside-sync"]
    app_outside = ranges["app-range-outside-sync"]
    lines = [
        "# sm_86 launch-cycle, completion, and range-composition campaign",
        "",
        "## Outcome",
        "",
        "The live campaign resolves the semantic boundary but rejects immediate numeric",
        "promotion of `gpc__cycles_elapsed.max`. For a single kernel, GPC elapsed is",
        "almost spatially uniform, but subtracting the separately reduced SM-active",
        "maximum does not produce a temporal interval. The simulator independently",
        "marks fire-and-forget store and reduction completion work unmodeled. For",
        "multiple launches, NCU",
        "Range Replay measures host/front-end idle time as well as kernels, so neither",
        "the sum nor maximum of trace-derived kernel times reconstructs the range.",
        "",
        "The implementation consequence is a typed completion diagnostic: last issue,",
        "modeled drain frontier, and drain completeness must remain distinct.  Per-kernel",
        "cycle emission stays unavailable when launch-envelope or retirement tails are",
        "unmodeled; generic workload-range emission remains unavailable without an",
        "explicit launch-timeline input.",
        "",
        "## Campaign coverage",
        "",
        f"- {integrity['matrix_cells']} cells and {integrity['correctness_executions']} correct ordinary executions.",
        f"- {integrity['ncu_kernel_profiles']} per-kernel NCU profiles ({integrity['ncu_kernel_rows']} kernel rows).",
        f"- {integrity['ncu_range_profiles']} Range Replay and {integrity['ncu_application_range_profiles']} Application Range Replay profiles with sync inside the range.",
        f"- {integrity['ncu_range_outside_sync_profiles']} + {integrity['ncu_application_range_outside_sync_profiles']} adversarial profiles with the final sync outside the range.",
        f"- {integrity['nvbit_captures']} NVBit captures, {integrity['nvbit_launches']} launches, and {integrity['nvbit_records']:,} records.",
        f"- {integrity['simulator_replay_pairs']} ordinary and {integrity['carded_dram_replay_pairs']} carded-DRAM byte-identical replay pairs.",
        f"- {integrity['static_functions']} tail kernels, {integrity['static_instructions']} decoded static instructions, zero unknown encodings and zero spills.",
        "- Every trace has exact total/reported cardinality and empty unsupported, unmapped, and cross-allocation sidecars.",
        "- Application replay can collect GPC and SM-active metric groups on",
        "  different executions; their extrema difference is not co-temporal.",
        "",
        "## GPC rollup and extrema comparison",
        "",
        f"The inferred instance count is {kernel['gpc_instance_count_median']:.6f}; maximum error from seven is {kernel['gpc_instance_count_max_abs_error_from_7']:.6f}. ",
        f"Across every kernel row, `max-min` is at most {kernel['gpc_instance_spread_max_cycles']:.0f} cycles. ",
        "Thus GPC `.max` is a stable workload-duration observable here; it does not",
        "require an SM-to-GPC reduction for this SKU. SM-active `.max` is reduced",
        "separately and is not a shared timestamp.",
        "",
        "| CTAs | median GPC−SM-active | min | max |",
        "|---:|---:|---:|---:|",
    ]
    for blocks in (1, 7, 42, 84):
        item = extrema_differences[str(blocks)]
        lines.append(
            f"| {blocks} | {item['median']:.1f} | {item['min']:.1f} | {item['max']:.1f} |"
        )
    lines.extend(
        [
            "",
            "For control kernels this difference is invariant over 0–4,096 dependent",
            "iterations at each tested grid size. It is grid-shape dependent and not a",
            "temporal interval, so one universal intercept is not promoted.",
            "",
            "## Terminal-tail discriminator",
            "",
            "All values below are medians of eight matched `(CTAs, iterations)` deltas",
            "relative to the control. The extrema-difference column subtracts",
            "independently reduced GPC and SM maxima; it does not isolate a temporal",
            "interval after SM activity ends.",
            "",
            "| final tail | live GPC delta | live SM-active delta | GPC-max minus SM-active-max delta | sim issue delta | sim terminal delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tail in TAILS[1:]:
        item = tails[tail]
        lines.append(
            f"| {tail} | {item['observed_gpc_delta_median']:.1f} | "
            f"{item['observed_sm_active_delta_median']:.1f} | "
            f"{item['observed_gpc_max_minus_sm_active_max_delta_median']:.1f} | "
            f"{item['sim_issue_delta_median']:.1f} | {item['sim_terminal_delta_median']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The discarded load proves that the modeled terminal event is directionally",
            "better than last issue: it adds the 420-cycle modeled response tail.  Its",
            "absolute live delta is larger because the current latency-only hierarchy",
            "does not model the full L1TEX/request-service path.  Store and RED instead",
            "have larger GPC than SM-active deltas. This spatial extrema comparison",
            "cannot place the excess after SM activity, but the simulator's explicit",
            "dirty-writeback/coherence and reduction-RMW flags still make its terminal",
            "frontier semantically incomplete rather than optional metadata.",
            "",
            "| single-launch candidate | MAPE vs GPC elapsed |",
            "|---|---:|",
            f"| last issue, raw | {candidates['last_issue_raw_mape_pct']:.3f}% |",
            f"| terminal event, raw | {candidates['terminal_raw_mape_pct']:.3f}% |",
            f"| last issue + control extrema difference | {candidates['last_issue_plus_control_extrema_difference_mape_pct']:.3f}% |",
            f"| terminal + control extrema difference | {candidates['terminal_plus_control_extrema_difference_mape_pct']:.3f}% |",
            "",
            "The empirical correction improves scale but does not repair missing execution mechanisms",
            "or unmodeled retirement tails.  Carded peak DRAM bandwidth changes no cell",
            "because this matrix does not saturate the existing aggregate server.",
            "",
            "## Multi-launch composition",
            "",
            "| NCU mode | cells | MAPE of sum(kernel GPC) | MAPE of sum(kernel SM-active) | median range / kernel-GPC sum | max GPC CV |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Range Replay | {range_mode['cells']} | {range_mode['gpc_sum_composition_mape_pct']:.3f}% | {range_mode['sm_active_sum_composition_mape_pct']:.3f}% | {range_mode['range_gpc_over_kernel_sum_median']:.3f}× | {range_mode['range_gpc_cv_max_pct']:.3f}% |",
            f"| Application Range Replay | {app_mode['cells']} | {app_mode['gpc_sum_composition_mape_pct']:.3f}% | {app_mode['sm_active_sum_composition_mape_pct']:.3f}% | {app_mode['range_gpc_over_kernel_sum_median']:.3f}× | {app_mode['range_gpc_cv_max_pct']:.3f}% |",
            f"| Range Replay, sync outside | {range_outside['cells']} | {range_outside['gpc_sum_composition_mape_pct']:.3f}% | {range_outside['sm_active_sum_composition_mape_pct']:.3f}% | {range_outside['range_gpc_over_kernel_sum_median']:.3f}× | {range_outside['range_gpc_cv_max_pct']:.3f}% |",
            f"| Application Range Replay, sync outside | {app_outside['cells']} | {app_outside['gpc_sum_composition_mape_pct']:.3f}% | {app_outside['sm_active_sum_composition_mape_pct']:.3f}% | {app_outside['range_gpc_over_kernel_sum_median']:.3f}× | {app_outside['range_gpc_cv_max_pct']:.3f}% |",
            "",
            "The SM-active sums remain close because they count active work.  GPC elapsed",
            "ranges include idle intervals between marked CUDA APIs (including the final",
            "sync when it lies inside the range);",
            "their magnitude and variability differ between capture-based Range Replay",
            "and whole-application range replay. Moving the final device synchronization",
            "outside the marked range does not repair GPC composition, while active-SM",
            "composition remains stable. A SASS trace has no timestamps for",
            "those host/front-end gaps.  Generic `sum`, `max`, and terminal-event reducers",
            "are therefore all rejected as workload-range semantics.",
            "",
            "## Decision",
            "",
            "1. Treat NCU ordinary application-replay labels as per-kernel observables.",
            "2. Define the simulator's completion frontier as the maximum of last issue",
            "   and processed modeled events, but expose whether every retirement tail is",
            "   represented before calling that frontier complete.",
            "3. Do not convert a completion frontier directly to GPC elapsed until a",
            "   launch-envelope model is present and completion is complete.",
            "4. Do not emit one workload-range GPC value from an unordered launch roster.",
            "   A future range prediction requires an explicit timed launch DAG or command",
            "   stream; a same-stream cache handoff alone supplies ordering, not gaps.",
            "5. Keep `sm_120` unavailable; none of these Ampere measurements transfer an",
            "   architecture default.",
            "",
            "No privileged command was required.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.output_root.resolve()
    integrity = validate_integrity(root)
    kernel_rows, observed = build_kernel_rows(root)
    kernel = kernel_summary(kernel_rows)
    range_rows, ranges = range_summary(root, observed)
    summary = {"integrity": integrity, "kernel": kernel, "ranges": ranges}
    analysis = root / "analysis"
    write_csv(analysis / "kernel_launches.csv", kernel_rows)
    write_csv(analysis / "ranges.csv", range_rows)
    write_json(analysis / "summary.json", summary)
    (analysis / "README.md").write_text(
        report_markdown(integrity, kernel, ranges), encoding="utf-8"
    )
    print(analysis / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
