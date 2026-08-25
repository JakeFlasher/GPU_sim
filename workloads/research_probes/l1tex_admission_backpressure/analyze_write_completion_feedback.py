#!/usr/bin/env python3
"""Reduce schema-11 per-warp write-completion MEMBAR feedback."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

import campaign


PREDECESSOR = {"store": 7, "red": 12}
COMPLETION = {"store": 201, "red": 202}


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


def cells_by_key() -> dict[str, campaign.Cell]:
    return {cell.key: cell for cell in campaign.configured_cells()}


def control_key(cell: campaign.Cell) -> str:
    return (
        f"b{cell.blocks}_w{cell.warps}_control_{cell.layout}_"
        f"d{cell.depth}_g{cell.gap_steps}_{cell.pattern}"
    )


def base_rows(root: Path) -> list[dict[str, object]]:
    with (root / "analysis/cells.csv").open(encoding="utf-8", newline="") as handle:
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
    holdout = [row for row in material if row not in calibration]
    if len(calibration) != 5 or len(holdout) != 20:
        raise RuntimeError(f"{tail}: expected 5 calibration and 20 holdout cells")
    return calibration, holdout


def completion_launch(
    root: Path,
    candidate_id: int,
    candidate: Mapping[str, object],
    cell: campaign.Cell,
) -> dict[str, object]:
    directory = root / f"simulator-memory-write-completion-feedback-{candidate_id}" / cell.key
    first = directory / "replay-1.json"
    second = directory / "replay-2.json"
    source = campaign.WRITE_COMPLETION_FEEDBACK_FILES[candidate_id]
    snapshot = root / f"metadata/memory_write_completion_feedback_{candidate_id}.json"
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
    dependencies = [
        campaign.REPO_ROOT / "sim/target/release/sim-run",
        root / "sass/l1tex_admission_backpressure_trace.control_bits.json",
        campaign.REPO_ROOT / "arch_profiles/cards/sm_86.json",
        campaign.REPO_ROOT / "data/pchase/sm_86_static_card.json",
        source,
        snapshot,
        trace,
    ]
    error = campaign.REPLAY.replay_pair_error(first, second, dependencies, expectation)
    if error is not None:
        raise RuntimeError(f"completion {candidate_id}/{cell.key}: {error}")
    document = campaign.REPLAY.load_json_strict(first)
    if not isinstance(document, dict) or len(document.get("launches", [])) != 1:
        raise RuntimeError(f"completion {candidate_id}/{cell.key}: malformed replay")
    launch = document["launches"][0]
    expected_schema = 12 if candidate.get("write_completion_latency_cycles", 0) else 11
    if launch["timing"]["readiness"].get("schema_version") != expected_schema:
        raise RuntimeError(
            f"completion {candidate_id}/{cell.key}: expected schema {expected_schema}"
        )
    return launch


def predecessor_launch(root: Path, candidate_id: int, cell: campaign.Cell) -> dict[str, object]:
    directory = (
        root
        / f"simulator-memory-admission-feedback-{candidate_id}"
        / cell.key
    )
    first = directory / "replay-1.json"
    second = directory / "replay-2.json"
    try:
        source = campaign.FEEDBACK_CANDIDATE_FILES[candidate_id]
    except KeyError as error:
        raise RuntimeError(f"unknown predecessor candidate {candidate_id}") from error
    candidates = campaign.REPLAY.candidate_configurations(
        source, require_exactly_one=True
    )
    trace = root / "nvbit" / cell.key
    expectation = campaign.REPLAY.ReplayExpectation(
        launches=campaign.REPLAY.expected_launches_from_trace(
            trace,
            grid=(cell.blocks, 1, 1),
            block=(cell.warps * 32, 1, 1),
        ),
        carded_dram=False,
        feedback_candidate=candidates[0],
    )
    dependencies = [
        campaign.REPO_ROOT / "sim/target/release/sim-run",
        root / "sass/l1tex_admission_backpressure_trace.control_bits.json",
        campaign.REPO_ROOT / "arch_profiles/cards/sm_86.json",
        campaign.REPO_ROOT / "data/pchase/sm_86_static_card.json",
        source,
        root / f"metadata/memory_admission_feedback_{candidate_id}.json",
        trace,
    ]
    try:
        document = campaign.REPLAY.load_replay_pair(
            first, second, dependencies, expectation
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"predecessor {candidate_id}/{cell.key}: {error}"
        ) from error
    return document["launches"][0]


def metrics(launch: Mapping[str, object], *, completion: bool) -> dict[str, int]:
    timing = launch["timing"]
    readiness = timing["readiness"]
    feedback = readiness["memory_admission_feedback"]
    raw_terminal = int(timing["terminal_modeled_event_time"] or 0)
    service_terminal = int(feedback["terminal_service_completion_time"])
    source = launch["counters"].get("l2_by_source", {}).get("store", {})
    source_sectors = int(source.get("sectors", 0))
    source_hits = int(source.get("hits", 0))
    clock = readiness.get("clock_issue", {})
    return {
        "phase_issue": int(clock.get("maximum_first_to_second_issue_cycles", 0)),
        "phase_fence": int(clock.get("maximum_second_to_third_issue_cycles", 0)),
        "last_issue": int(timing["diagnostic_last_issue_cycles"]),
        "raw_terminal": raw_terminal,
        "service_completion": max(raw_terminal, service_terminal),
        "write_completion_terminal": int(
            feedback.get("terminal_write_completion_time", service_terminal)
        ),
        "write_completion_requests": int(
            feedback.get("write_completion_requests_observed", 0)
        ),
        "write_completion_regions": int(
            feedback.get("write_completion_regions_observed", 0)
        ),
        "write_completion_busy": int(feedback.get("write_completion_busy_cycles", 0)),
        "write_completion_limiting": int(
            readiness["limiting_cycles"].get("memory_write_completion", 0)
        ),
        "write_completion_exposure": int(
            readiness["constraint_exposure_cycles"].get("memory_write_completion", 0)
        ),
        "observer_hits": int(feedback["l2_hit_sectors_observed"]),
        "observer_misses": int(feedback["l2_miss_sectors_observed"]),
        "source_hits": source_hits,
        "source_misses": source_sectors - source_hits,
        "warps_with_three_reads": int(clock.get("warps_with_three_reads", 0))
        if completion
        else 0,
    }


def subtract(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    return {key: int(left[key]) - int(right[key]) for key in left}


def analyze(root: Path) -> dict[str, object]:
    rows = base_rows(root)
    cells = cells_by_key()
    candidates = campaign.validated_write_completion_feedback()
    result: dict[str, object] = {"schema_version": 1, "operations": {}}
    for tail in ("store", "red"):
        candidate_id = COMPLETION[tail]
        predecessor_id = PREDECESSOR[tail]
        candidate = candidates[candidate_id][1]
        calibration, holdout = partitions(rows, tail)
        enriched = []
        for row in [*calibration, *holdout]:
            cell = cells[str(row["cell"])]
            control = cells[control_key(cell)]
            target_metrics = metrics(
                completion_launch(root, candidate_id, candidate, cell), completion=True
            )
            control_metrics = metrics(
                completion_launch(root, candidate_id, candidate, control), completion=True
            )
            delta = subtract(target_metrics, control_metrics)
            old_target = metrics(
                predecessor_launch(root, predecessor_id, cell), completion=False
            )
            old_control = metrics(
                predecessor_launch(root, predecessor_id, control), completion=False
            )
            old_delta = subtract(old_target, old_control)
            enriched.append(
                {
                    **row,
                    **{f"completion_{key}": value for key, value in delta.items()},
                    **{f"predecessor_{key}": value for key, value in old_delta.items()},
                    "zero_fence": 0,
                }
            )
        calibration_rows = enriched[: len(calibration)]
        holdout_rows = enriched[len(calibration) :]

        def reductions(material: Sequence[Mapping[str, object]]) -> dict[str, object]:
            return {
                "phase_issue": score(material, "completion_phase_issue", "native_issue_delta"),
                "phase_fence": score(material, "completion_phase_fence", "native_fence_delta"),
                "zero_fence": score(material, "zero_fence", "native_fence_delta"),
                "raw_terminal_gpc": score(
                    material, "completion_raw_terminal", "ncu_gpc_delta"
                ),
                "service_completion_gpc": score(
                    material, "completion_service_completion", "ncu_gpc_delta"
                ),
                "predecessor_service_completion_gpc": score(
                    material, "predecessor_service_completion", "ncu_gpc_delta"
                ),
                "target_observer_changed_cells": sum(
                    int(row["completion_observer_hits"])
                    != int(row["predecessor_observer_hits"])
                    or int(row["completion_observer_misses"])
                    != int(row["predecessor_observer_misses"])
                    for row in material
                ),
                "aggregate_store_source_changed_cells": sum(
                    int(row["completion_source_hits"])
                    != int(row["predecessor_source_hits"])
                    or int(row["completion_source_misses"])
                    != int(row["predecessor_source_misses"])
                    for row in material
                ),
                "three_clock_warp_delta": sum(
                    int(row["completion_warps_with_three_reads"])
                    for row in material
                ),
                "by_stage": {
                    stage: {
                        "fence": score(
                            [row for row in material if row["stage"] == stage],
                            "completion_phase_fence",
                            "native_fence_delta",
                        ),
                        "gpc": score(
                            [row for row in material if row["stage"] == stage],
                            "completion_service_completion",
                            "ncu_gpc_delta",
                        ),
                    }
                    for stage in sorted({str(row["stage"]) for row in material})
                },
            }

        result["operations"][tail] = {
            "candidate_id": candidate_id,
            "predecessor_id": predecessor_id,
            "configuration": candidate,
            "calibration": reductions(calibration_rows),
            "holdout": reductions(holdout_rows),
        }
    return result


def markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Write-class completion feedback",
        "",
        "Schema 11 keeps admission queue release at service start and makes each",
        "warp's `MEMBAR.SC.GPU` wait for that warp's own latest matching service",
        "completion. Three CLOCK reads expose issue and fence phases separately.",
        "",
        "## Strict 20-cell holdout",
        "",
        "| tail | zero-fence MAE | schema-11 fence MAE | fence max | predecessor service/GPC MAE | schema-11 service/GPC MAE | GPC max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tail in ("store", "red"):
        item = result["operations"][tail]["holdout"]
        lines.append(
            f"| {tail} | {item['zero_fence']['mae']:.1f} | "
            f"{item['phase_fence']['mae']:.1f} | {item['phase_fence']['max_error']:.0f} | "
            f"{item['predecessor_service_completion_gpc']['mae']:.1f} | "
            f"{item['service_completion_gpc']['mae']:.1f} | "
            f"{item['service_completion_gpc']['max_error']:.0f} |"
        )
    lines.extend(
        [
            "",
            "The mechanism strongly identifies STORE fence waiting and improves RED",
            "fence direction, while preserving the predecessor GPC completion scale.",
            "RED deep/CTA/spacing fence residuals remain large because non-returning RED",
            "RMW completion is still outside the modeled event stream. STORE/RED retirement",
            "therefore remains typed incomplete and no numerical GPC result is promoted.",
            "",
            "Machine-readable reductions are in `write_completion_feedback_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=campaign.DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.output_root.resolve()
    try:
        result = analyze(root)
        analysis = root / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        (analysis / "write_completion_feedback_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (analysis / "write_completion_feedback.md").write_text(
            markdown(result), encoding="utf-8"
        )
        print(analysis / "write_completion_feedback.md")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"write-completion analysis: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
