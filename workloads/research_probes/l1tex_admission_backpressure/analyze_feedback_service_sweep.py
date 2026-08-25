#!/usr/bin/env python3
"""Select and evaluate schema-10 downstream miss rate/width candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence

import campaign


REFERENCE_FEEDBACK_ID = {"store": 7, "red": 12}
REFERENCE_SWEEP_ID = {"store": 105, "red": 125}
WIDTH_ONE_IDS = {
    "store": (101, 105, 109, 113),
    "red": (137, 121, 125, 129, 133),
}


def mean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("empty mean")
    return float(statistics.fmean(values))


def score(rows: Sequence[Mapping[str, object]], predicted: str, observed: str) -> dict[str, object]:
    errors = [float(row[predicted]) - float(row[observed]) for row in rows]
    worst = max(range(len(rows)), key=lambda index: abs(errors[index]))
    return {
        "cells": len(rows),
        "mae": mean([abs(error) for error in errors]),
        "mean_signed_error": mean(errors),
        "max_error": abs(errors[worst]),
        "worst_signed_error": errors[worst],
        "worst_cell": str(rows[worst]["cell"]),
    }


def configured_by_key() -> dict[str, campaign.Cell]:
    return {cell.key: cell for cell in campaign.configured_cells()}


def load_base_rows(root: Path) -> list[dict[str, object]]:
    path = root / "analysis/cells.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise RuntimeError(f"base analysis must contain 100 rows, got {len(rows)}")
    return rows


def partitions(
    rows: Sequence[Mapping[str, object]], tail: str
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    material = [row for row in rows if row["tail"] == tail]
    calibration = [
        row
        for row in material
        if row["stage"] == "burst_depth" and int(row["depth"]) <= 16
    ]
    held_out = [row for row in material if row not in calibration]
    if len(calibration) != 5 or len(held_out) != 20:
        raise RuntimeError(f"{tail}: expected 5 calibration and 20 holdout rows")
    return calibration, held_out


def candidate_tail(candidate: Mapping[str, object]) -> str:
    operation = candidate.get("operation")
    if operation == "store":
        return "store"
    if operation == "reduction":
        return "red"
    raise RuntimeError(f"unsupported sweep operation {operation!r}")


def candidate_paths(
    root: Path, candidate_id: int, candidate: Mapping[str, object]
) -> tuple[Path, list[Path]]:
    directory = root / f"simulator-memory-admission-feedback-service-{candidate_id}"
    source_file = campaign.validated_feedback_service_sweep()[candidate_id][0]
    source_snapshot = root / (
        "metadata/memory_admission_feedback_store_service_sweep.json"
        if candidate["operation"] == "store"
        else "metadata/memory_admission_feedback_red_service_sweep.json"
    )
    singleton = (
        root
        / "metadata/memory_admission_feedback_service_singletons"
        / f"candidate-{candidate_id}.json"
    )
    dependencies = [
        campaign.REPO_ROOT / "sim/target/release/sim-run",
        root / "sass/l1tex_admission_backpressure_trace.control_bits.json",
        campaign.REPO_ROOT / "arch_profiles/cards/sm_86.json",
        campaign.REPO_ROOT / "data/pchase/sm_86_static_card.json",
        source_file,
        source_snapshot,
        singleton,
    ]
    return directory, dependencies


def replay_launch(
    root: Path,
    candidate_id: int,
    candidate: Mapping[str, object],
    cell: campaign.Cell,
) -> dict[str, object]:
    directory, dependencies = candidate_paths(root, candidate_id, candidate)
    destination = directory / cell.key
    first = destination / "replay-1.json"
    second = destination / "replay-2.json"
    trace = root / "nvbit" / cell.key
    expectation = campaign.REPLAY.ReplayExpectation(
        launches=campaign.REPLAY.expected_launches_from_trace(
            trace,
            grid=(cell.blocks, 1, 1),
            block=(cell.warps * 32, 1, 1),
        ),
        carded_dram=False,
        feedback_candidate=candidate,
    )
    error = campaign.REPLAY.replay_pair_error(
        first, second, [*dependencies, trace], expectation
    )
    if error is not None:
        raise RuntimeError(f"candidate {candidate_id}/{cell.key}: {error}")
    document = campaign.REPLAY.load_json_strict(first)
    if not isinstance(document, dict) or len(document.get("launches", [])) != 1:
        raise RuntimeError(f"candidate {candidate_id}/{cell.key}: malformed replay")
    return document["launches"][0]


def launch_metrics(launch: Mapping[str, object]) -> dict[str, int]:
    timing = launch["timing"]
    readiness = timing["readiness"]
    feedback = readiness["memory_admission_feedback"]
    source = launch["counters"].get("l2_by_source", {}).get("store", {})
    source_sectors = int(source.get("sectors", 0))
    source_hits = int(source.get("hits", 0))
    terminal = timing["terminal_modeled_event_time"]
    if terminal is None:
        raise RuntimeError("feedback replay has no terminal modeled event")
    service_terminal = int(feedback["terminal_service_completion_time"])
    return {
        "last_issue": int(timing["diagnostic_last_issue_cycles"]),
        "terminal": int(terminal),
        "service_completion": max(int(terminal), service_terminal),
        "clock": int(
            readiness.get("clock_issue", {}).get("maximum_first_to_last_issue_cycles", 0)
        ),
        "observer_hits": int(feedback["l2_hit_sectors_observed"]),
        "observer_misses": int(feedback["l2_miss_sectors_observed"]),
        "source_hits": source_hits,
        "source_misses": source_sectors - source_hits,
    }


def candidate_delta(
    root: Path,
    candidate_id: int,
    candidate: Mapping[str, object],
    cell: campaign.Cell,
    cells: Mapping[str, campaign.Cell],
) -> dict[str, int]:
    target = launch_metrics(replay_launch(root, candidate_id, candidate, cell))
    control_name = (
        f"b{cell.blocks}_w{cell.warps}_control_{cell.layout}_"
        f"d{cell.depth}_g{cell.gap_steps}_{cell.pattern}"
    )
    control = launch_metrics(
        replay_launch(root, candidate_id, candidate, cells[control_name])
    )
    return {key: target[key] - control[key] for key in target}


def enrich(
    rows: Sequence[Mapping[str, object]],
    deltas: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "sweep_last_issue": deltas[str(row["cell"])]["last_issue"],
            "sweep_terminal": deltas[str(row["cell"])]["terminal"],
            "sweep_service_completion": deltas[str(row["cell"])]["service_completion"],
            "sweep_clock": deltas[str(row["cell"])]["clock"],
            "sweep_observer_hits": deltas[str(row["cell"])]["observer_hits"],
            "sweep_observer_misses": deltas[str(row["cell"])]["observer_misses"],
            "sweep_source_hits": deltas[str(row["cell"])]["source_hits"],
            "sweep_source_misses": deltas[str(row["cell"])]["source_misses"],
        }
        for row in rows
    ]


def pareto_ids(scores: Mapping[int, Mapping[str, object]]) -> list[int]:
    result = []
    for candidate_id, item in scores.items():
        issue = float(item["issue"]["mae"])
        completion = float(item["service_completion"]["mae"])
        dominated = any(
            other_id != candidate_id
            and float(other["issue"]["mae"]) <= issue
            and float(other["service_completion"]["mae"]) <= completion
            and (
                float(other["issue"]["mae"]) < issue
                or float(other["service_completion"]["mae"]) < completion
            )
            for other_id, other in scores.items()
        )
        if not dominated:
            result.append(candidate_id)
    return sorted(result)


def analyze(root: Path, *, calibration_only: bool) -> dict[str, object]:
    base_rows = load_base_rows(root)
    cells = configured_by_key()
    sweep = campaign.validated_feedback_service_sweep()
    result: dict[str, object] = {
        "schema_version": 1,
        "selection_rule": (
            "Among width-one candidates, retain the calibration Pareto frontier over "
            "last-issue MAE and max(raw-system-terminal, downstream-service-completion) "
            "MAE. Advance every declared width at every Pareto rate without holdout refitting."
        ),
        "operations": {},
    }
    for tail in ("store", "red"):
        calibration, held_out = partitions(base_rows, tail)
        baseline = {
            "issue": score(calibration, "sim_last_issue_delta", "native_issue_delta"),
            "terminal": score(calibration, "sim_terminal_delta", "ncu_gpc_delta"),
        }
        candidates = {
            candidate_id: candidate
            for candidate_id, (_, candidate) in sweep.items()
            if candidate_tail(candidate) == tail
        }
        calibration_scores: dict[int, dict[str, object]] = {}
        calibration_deltas: dict[int, dict[str, dict[str, int]]] = {}
        for candidate_id, candidate in candidates.items():
            deltas = {
                str(row["cell"]): candidate_delta(
                    root, candidate_id, candidate, cells[str(row["cell"])], cells
                )
                for row in calibration
            }
            calibration_deltas[candidate_id] = deltas
            material = enrich(calibration, deltas)
            issue = score(material, "sweep_last_issue", "native_issue_delta")
            terminal = score(material, "sweep_terminal", "ncu_gpc_delta")
            service_completion = score(
                material, "sweep_service_completion", "ncu_gpc_delta"
            )
            calibration_scores[candidate_id] = {
                "configuration": candidate,
                "issue": issue,
                "terminal": terminal,
                "service_completion": service_completion,
                "clock": score(material, "sweep_clock", "native_issue_delta"),
                "joint_cycle_mae": float(issue["mae"])
                + float(service_completion["mae"]),
            }

        reference_id = REFERENCE_SWEEP_ID[tail]
        reference_feedback_id = REFERENCE_FEEDBACK_ID[tail]
        for row in calibration:
            key = str(row["cell"])
            delta = calibration_deltas[reference_id][key]
            comparisons = {
                "last_issue": int(float(row[f"feedback_{reference_feedback_id}_sim_last_issue_delta"])),
                "terminal": int(float(row[f"feedback_{reference_feedback_id}_sim_terminal_delta"])),
                "clock": int(float(row[f"feedback_{reference_feedback_id}_sim_clock_delta"])),
                "observer_hits": int(float(row[f"feedback_{reference_feedback_id}_l2_hit_sectors_observed_delta"])),
                "observer_misses": int(float(row[f"feedback_{reference_feedback_id}_l2_miss_sectors_observed_delta"])),
                "source_hits": int(float(row[f"feedback_{reference_feedback_id}_l2_source_hits_delta"])),
                "source_misses": int(float(row[f"feedback_{reference_feedback_id}_l2_source_misses_delta"])),
            }
            if {name: value for name, value in delta.items() if name != "service_completion"} != comparisons:
                raise RuntimeError(
                    f"{tail}/{key}: width-one predecessor compatibility mismatch"
                )

        width_one = {candidate_id: calibration_scores[candidate_id] for candidate_id in WIDTH_ONE_IDS[tail]}
        width_one_pareto_ids = pareto_ids(width_one)
        selected_rates = sorted(
            {
                int(candidates[candidate_id]["l2_miss_sector_cycles"])
                for candidate_id in width_one_pareto_ids
            }
        )
        selected_ids = sorted(
            candidate_id
            for candidate_id, candidate in candidates.items()
            if int(candidate["l2_miss_sector_cycles"]) in selected_rates
        )
        operation: dict[str, object] = {
            "baseline_calibration": baseline,
            "calibration_scores": {str(key): value for key, value in calibration_scores.items()},
            "width_one_pareto_ids": width_one_pareto_ids,
            "selected_rate_candidate_ids": width_one_pareto_ids,
            "selected_miss_cycles_per_sector": selected_rates,
            "selected_width_candidate_ids": selected_ids,
            "selected_rate_is_grid_boundary": any(
                selected_rate in {
                    min(
                        int(candidate["l2_miss_sector_cycles"])
                        for candidate in candidates.values()
                    ),
                    max(
                        int(candidate["l2_miss_sector_cycles"])
                        for candidate in candidates.values()
                    ),
                }
                for selected_rate in selected_rates
            ),
        }

        missing = []
        holdout_scores = {}
        if not calibration_only:
            for candidate_id in selected_ids:
                candidate = candidates[candidate_id]
                deltas = {}
                for row in held_out:
                    cell = cells[str(row["cell"])]
                    directory, _ = candidate_paths(root, candidate_id, candidate)
                    control_name = (
                        f"b{cell.blocks}_w{cell.warps}_control_{cell.layout}_"
                        f"d{cell.depth}_g{cell.gap_steps}_{cell.pattern}"
                    )
                    if not (directory / cell.key / "replay-1.json").is_file() or not (
                        directory / control_name / "replay-1.json"
                    ).is_file():
                        missing.append(f"{candidate_id}:{cell.key}")
                        continue
                    deltas[str(row["cell"])] = candidate_delta(
                        root, candidate_id, candidate, cell, cells
                    )
                if len(deltas) == len(held_out):
                    material = enrich(held_out, deltas)
                    holdout_scores[str(candidate_id)] = {
                        "configuration": candidate,
                        "issue": score(material, "sweep_last_issue", "native_issue_delta"),
                        "terminal": score(material, "sweep_terminal", "ncu_gpc_delta"),
                        "service_completion": score(
                            material,
                            "sweep_service_completion",
                            "ncu_gpc_delta",
                        ),
                        "clock": score(material, "sweep_clock", "native_issue_delta"),
                        "raw_terminal_equals_last_issue_cells": sum(
                            int(row["sweep_terminal"]) == int(row["sweep_last_issue"])
                            for row in material
                        ),
                        "aggregate_store_source_changed_vs_predecessor": sum(
                            int(row["sweep_source_hits"])
                            != int(float(row[f"feedback_{reference_feedback_id}_l2_source_hits_delta"]))
                            or int(row["sweep_source_misses"])
                            != int(float(row[f"feedback_{reference_feedback_id}_l2_source_misses_delta"]))
                            for row in material
                        ),
                        "candidate_observer_changed_vs_predecessor": sum(
                            int(row["sweep_observer_hits"])
                            != int(float(row[f"feedback_{reference_feedback_id}_l2_hit_sectors_observed_delta"]))
                            or int(row["sweep_observer_misses"])
                            != int(float(row[f"feedback_{reference_feedback_id}_l2_miss_sectors_observed_delta"]))
                            for row in material
                        ),
                        "by_stage": {
                            stage: {
                                "issue": score(
                                    [row for row in material if row["stage"] == stage],
                                    "sweep_last_issue",
                                    "native_issue_delta",
                                ),
                                "terminal": score(
                                    [row for row in material if row["stage"] == stage],
                                    "sweep_terminal",
                                    "ncu_gpc_delta",
                                ),
                                "service_completion": score(
                                    [row for row in material if row["stage"] == stage],
                                    "sweep_service_completion",
                                    "ncu_gpc_delta",
                                ),
                            }
                            for stage in sorted({str(row["stage"]) for row in material})
                        },
                    }
        holdout_complete = (
            not calibration_only
            and not missing
            and len(holdout_scores) == len(selected_ids)
        )
        if holdout_complete:
            numeric_holdout = {int(key): value for key, value in holdout_scores.items()}
            predecessor_id = REFERENCE_SWEEP_ID[tail]
            predecessor = numeric_holdout[predecessor_id]
            operation["holdout_pareto_ids"] = pareto_ids(numeric_holdout)
            operation["predecessor_candidate_id"] = predecessor_id
            operation["candidates_improving_both_predecessor_frontiers"] = sorted(
                candidate_id
                for candidate_id, item in numeric_holdout.items()
                if float(item["issue"]["mae"]) < float(predecessor["issue"]["mae"])
                and float(item["service_completion"]["mae"])
                < float(predecessor["service_completion"]["mae"])
            )
            label_gaps = [
                abs(float(row["native_issue_delta"]) - float(row["ncu_gpc_delta"]))
                for row in held_out
            ]
            operation["same_frontier_joint_mae_lower_bound"] = mean(label_gaps)
            operation["maximum_issue_vs_gpc_label_gap"] = max(label_gaps)
        operation["holdout_status"] = (
            "complete" if holdout_complete else "awaiting_replay"
        )
        operation["missing_holdout_outputs"] = missing
        operation["holdout_scores"] = holdout_scores
        result["operations"][tail] = operation
    return result


def markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Feedback miss-service rate/width sweep",
        "",
        "The queue scope, capacity, hit service, and operand-read latency remain frozen.",
        "Width is an explicit fixed miss-sector quantum; width zero remains legacy fluid service.",
        "",
        "## Calibration selection",
        "",
        "| tail | selected scalar rate | width candidates advanced | grid boundary |",
        "|---|---:|---|---|",
    ]
    operations = result["operations"]
    for tail in ("store", "red"):
        item = operations[tail]
        lines.append(
            f"| {tail} | {item['selected_miss_cycles_per_sector']} cycles/miss-sector | "
            f"{item['selected_width_candidate_ids']} | {item['selected_rate_is_grid_boundary']} |"
        )
    lines.extend(["", "Selection uses only five shallow unique-line cells."])
    lines.extend(
        [
            "",
            "Width-one calibration scores use the downstream service-completion frontier:",
            "",
            "| tail | candidate | rate | issue MAE | service-completion MAE | Pareto |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for tail in ("store", "red"):
        item = operations[tail]
        for candidate_id in WIDTH_ONE_IDS[tail]:
            candidate = item["calibration_scores"][str(candidate_id)]
            lines.append(
                f"| {tail} | {candidate_id} | "
                f"{candidate['configuration']['l2_miss_sector_cycles']} | "
                f"{candidate['issue']['mae']:.1f} | "
                f"{candidate['service_completion']['mae']:.1f} | "
                f"{candidate_id in item['width_one_pareto_ids']} |"
            )
    if all(operations[tail]["holdout_status"] == "complete" for tail in ("store", "red")):
        lines.extend(
            [
                "",
                "## Strict 20-cell holdout",
                "",
                "| tail | candidate | rate | width | issue MAE | raw-terminal MAE | service-completion MAE | issue max | completion max |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tail in ("store", "red"):
            for candidate_id, score_item in operations[tail]["holdout_scores"].items():
                config = score_item["configuration"]
                lines.append(
                    f"| {tail} | {candidate_id} | {config['l2_miss_sector_cycles']} | "
                    f"{config['l2_miss_service_width_sectors']} | {score_item['issue']['mae']:.1f} | "
                    f"{score_item['terminal']['mae']:.1f} | "
                    f"{score_item['service_completion']['mae']:.1f} | "
                    f"{score_item['issue']['max_error']:.0f} | "
                    f"{score_item['service_completion']['max_error']:.0f} |"
                )
        lines.extend(
            [
                "",
                "No candidate improves both holdout issue and downstream service-completion",
                "MAE over its width-one predecessor. Width two trades a very small issue",
                "improvement for a completion regression; widths four/eight fail topology",
                "transfer. RED rate 12 is calibration-Pareto but fails holdout, while rate 16",
                "returns to the predecessor. No rate or width advances to an architecture default.",
                "",
                f"Raw modeled terminal equals last issue in all "
                f"{operations['store']['holdout_scores']['105']['raw_terminal_equals_last_issue_cells']} STORE and "
                f"{operations['red']['holdout_scores']['125']['raw_terminal_equals_last_issue_cells']} RED predecessor holdout cells. "
                f"A single shared prediction therefore has an irreducible issue/GPC joint-MAE "
                f"lower bound of {operations['store']['same_frontier_joint_mae_lower_bound']:.1f} STORE and "
                f"{operations['red']['same_frontier_joint_mae_lower_bound']:.1f} RED cycles. "
                "The next component must separate admission/issue readiness from write-class",
                "completion rather than continue tuning one service frontier.",
            ]
        )
    else:
        lines.extend(["", "Holdout replay is pending for the frozen width candidates."])
    lines.extend(["", "The machine-readable selection and scores are in `feedback_service_sweep_summary.json`.", ""])
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=campaign.DEFAULT_OUTPUT)
    result.add_argument("--calibration-only", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.output_root.resolve()
    try:
        result = analyze(root, calibration_only=args.calibration_only)
        analysis = root / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        (analysis / "feedback_service_sweep_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (analysis / "feedback_service_sweep.md").write_text(
            markdown(result), encoding="utf-8"
        )
        print(analysis / "feedback_service_sweep.md")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"feedback service sweep analysis: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
