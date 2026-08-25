#!/usr/bin/env python3
"""Select request/region RED RMW completion work and bounded parallelism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import analyze_write_completion_feedback as base
import campaign


CANDIDATE_IDS = (202, *range(211, 229))


def pareto_ids(scores: Mapping[int, Mapping[str, object]]) -> list[int]:
    result = []
    for candidate_id, item in scores.items():
        fence = float(item["fence"]["mae"])
        gpc = float(item["gpc"]["mae"])
        dominated = any(
            other_id != candidate_id
            and float(other["fence"]["mae"]) <= fence
            and float(other["gpc"]["mae"]) <= gpc
            and (
                float(other["fence"]["mae"]) < fence
                or float(other["gpc"]["mae"]) < gpc
            )
            for other_id, other in scores.items()
        )
        if not dominated:
            result.append(candidate_id)
    return sorted(result)


def rmw_launch(
    root: Path,
    candidate_id: int,
    candidate: Mapping[str, object],
    cell: campaign.Cell,
) -> dict[str, object]:
    directory = root / f"simulator-red-rmw-completion-{candidate_id}" / cell.key
    first = directory / "replay-1.json"
    second = directory / "replay-2.json"
    source = campaign.validated_red_rmw_completion_sweep()[candidate_id][0]
    source_snapshot = root / "metadata" / source.name
    singleton = root / "metadata/red_rmw_completion_singletons" / f"candidate-{candidate_id}.json"
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
        source_snapshot,
        singleton,
        trace,
    ]
    error = campaign.REPLAY.replay_pair_error(first, second, dependencies, expectation)
    if error is not None:
        raise RuntimeError(f"RED RMW {candidate_id}/{cell.key}: {error}")
    document = campaign.REPLAY.load_json_strict(first)
    launch = document["launches"][0]
    if launch["timing"]["readiness"].get("schema_version") != 13:
        raise RuntimeError(f"RED RMW {candidate_id}/{cell.key}: expected schema 13")
    return launch


def candidate_launch(
    root: Path,
    candidate_id: int,
    candidate: Mapping[str, object],
    cell: campaign.Cell,
) -> dict[str, object]:
    if candidate_id == 202:
        return base.completion_launch(root, 202, candidate, cell)
    return rmw_launch(root, candidate_id, candidate, cell)


def delta(
    root: Path,
    candidate_id: int,
    candidate: Mapping[str, object],
    cell: campaign.Cell,
    cells: Mapping[str, campaign.Cell],
) -> dict[str, int]:
    target = base.metrics(
        candidate_launch(root, candidate_id, candidate, cell), completion=True
    )
    control = base.metrics(
        candidate_launch(root, candidate_id, candidate, cells[base.control_key(cell)]),
        completion=True,
    )
    return base.subtract(target, control)


def enrich(
    rows: Sequence[Mapping[str, object]], deltas: Mapping[str, Mapping[str, int]]
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "rmw_fence": deltas[str(row["cell"])]["phase_fence"],
            "rmw_gpc": deltas[str(row["cell"])]["service_completion"],
            "rmw_requests": deltas[str(row["cell"])]["write_completion_requests"],
            "rmw_regions": deltas[str(row["cell"])]["write_completion_regions"],
            "rmw_busy": deltas[str(row["cell"])]["write_completion_busy"],
        }
        for row in rows
    ]


def repeat_sensitivity(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        "fence_envelope_low": base.score(
            rows, "rmw_fence", "native_fence_delta_repeat_envelope_low"
        ),
        "fence_envelope_high": base.score(
            rows, "rmw_fence", "native_fence_delta_repeat_envelope_high"
        ),
        "gpc_balanced_three": base.score(
            rows, "rmw_gpc", "ncu_gpc_delta_balanced_three"
        ),
        "gpc_envelope_low": base.score(
            rows, "rmw_gpc", "ncu_gpc_delta_repeat_envelope_low"
        ),
        "gpc_envelope_high": base.score(
            rows, "rmw_gpc", "ncu_gpc_delta_repeat_envelope_high"
        ),
    }


def analyze(root: Path, *, calibration_only: bool) -> dict[str, object]:
    rows = base.base_rows(root)
    cells = base.cells_by_key()
    _, calibration_candidates = campaign.validated_admission_documents()
    candidates = {
        202: {
            **calibration_candidates[12],
            "candidate_id": 202,
            "membar_waits_for_service_completion": True,
        },
        **{
            candidate_id: candidate
            for candidate_id, (_, candidate) in campaign.validated_red_rmw_completion_sweep().items()
        },
    }
    calibration, holdout = base.partitions(rows, "red")
    calibration_scores: dict[int, dict[str, object]] = {}
    for candidate_id in CANDIDATE_IDS:
        candidate = candidates[candidate_id]
        deltas = {
            str(row["cell"]): delta(
                root, candidate_id, candidate, cells[str(row["cell"])], cells
            )
            for row in calibration
        }
        material = enrich(calibration, deltas)
        calibration_scores[candidate_id] = {
            "configuration": candidate,
            "fence": base.score(material, "rmw_fence", "native_fence_delta"),
            "gpc": base.score(material, "rmw_gpc", "ncu_gpc_delta"),
        }
    selected_ids = pareto_ids(calibration_scores)
    holdout_scores = {}
    missing = []
    if not calibration_only:
        for candidate_id in selected_ids:
            candidate = candidates[candidate_id]
            directory = (
                root / "simulator-memory-write-completion-feedback-202"
                if candidate_id == 202
                else root / f"simulator-red-rmw-completion-{candidate_id}"
            )
            deltas = {}
            for row in holdout:
                cell = cells[str(row["cell"])]
                control = cells[base.control_key(cell)]
                if not (directory / cell.key / "replay-1.json").is_file() or not (
                    directory / control.key / "replay-1.json"
                ).is_file():
                    missing.append(f"{candidate_id}:{cell.key}")
                    continue
                deltas[str(row["cell"])] = delta(
                    root, candidate_id, candidate, cell, cells
                )
            if len(deltas) == len(holdout):
                material = enrich(holdout, deltas)
                holdout_scores[str(candidate_id)] = {
                    "configuration": candidate,
                    "fence": base.score(material, "rmw_fence", "native_fence_delta"),
                    "gpc": base.score(material, "rmw_gpc", "ncu_gpc_delta"),
                    "requests_observed_delta": sum(int(row["rmw_requests"]) for row in material),
                    "regions_observed_delta": sum(int(row["rmw_regions"]) for row in material),
                    "busy_cycles_delta": sum(int(row["rmw_busy"]) for row in material),
                    "repeat_sensitivity": repeat_sensitivity(material),
                    "by_stage": {
                        stage: {
                            "fence": base.score(
                                [row for row in material if row["stage"] == stage],
                                "rmw_fence",
                                "native_fence_delta",
                            ),
                            "gpc": base.score(
                                [row for row in material if row["stage"] == stage],
                                "rmw_gpc",
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
    numeric_holdout = {int(key): value for key, value in holdout_scores.items()}
    holdout_pareto = pareto_ids(numeric_holdout) if holdout_complete else []
    improving_both = []
    repeat_robust_improving_both = []
    if numeric_holdout and 202 in numeric_holdout:
        predecessor = numeric_holdout[202]
        improving_both = sorted(
            candidate_id
            for candidate_id, item in numeric_holdout.items()
            if float(item["fence"]["mae"]) < float(predecessor["fence"]["mae"])
            and float(item["gpc"]["mae"]) < float(predecessor["gpc"]["mae"])
        )
        repeat_robust_improving_both = sorted(
            candidate_id
            for candidate_id in improving_both
            if all(
                float(numeric_holdout[candidate_id]["repeat_sensitivity"][label]["mae"])
                < float(predecessor["repeat_sensitivity"][label]["mae"])
                for label in predecessor["repeat_sensitivity"]
            )
        )
    return {
        "schema_version": 1,
        "selection_rule": "Advance the five-cell fence/GPC Pareto frontier without holdout refitting.",
        "calibration_scores": {
            str(candidate_id): item for candidate_id, item in calibration_scores.items()
        },
        "calibration_pareto_ids": selected_ids,
        "holdout_status": "complete" if holdout_complete else "awaiting_replay",
        "missing_holdout_outputs": missing,
        "holdout_scores": holdout_scores,
        "holdout_pareto_ids": holdout_pareto,
        "candidates_improving_both_predecessor_targets": improving_both,
        "candidates_improving_both_all_repeat_sensitivity_labels": (
            repeat_robust_improving_both
        ),
        "decision": {
            "retain_schema_13_mechanism": True,
            "promoted_candidate_id": (
                repeat_robust_improving_both[0]
                if len(repeat_robust_improving_both) == 1
                else None
            ),
            "reason": (
                "No candidate improves both predecessor targets across every "
                "retained repeat-sensitivity label."
            ),
        },
    }


def label(config: Mapping[str, object]) -> str:
    if config.get("write_completion_request_cycles", 0):
        return f"request/{config['write_completion_request_cycles']}"
    if config.get("write_completion_region_cycles", 0):
        return f"region/{config['write_completion_region_cycles']}"
    return "none/0"


def markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# RED RMW completion work/parallelism sweep",
        "",
        "Independent completion begins after admission service done and cannot change",
        "queue release. Work is exact L2 requests or unique 256-byte regions.",
        "",
        "## Calibration",
        "",
        "| candidate | basis/rate | lanes | fence MAE | GPC MAE | Pareto |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for candidate_id in CANDIDATE_IDS:
        item = result["calibration_scores"][str(candidate_id)]
        config = item["configuration"]
        lines.append(
            f"| {candidate_id} | {label(config)} | "
            f"{config.get('write_completion_parallel_lanes', 0)} | "
            f"{item['fence']['mae']:.1f} | {item['gpc']['mae']:.1f} | "
            f"{candidate_id in result['calibration_pareto_ids']} |"
        )
    if result["holdout_status"] == "complete":
        lines.extend(
            [
                "",
                "## Strict 20-cell holdout",
                "",
                "| candidate | basis/rate | lanes | fence MAE | GPC MAE | fence max | GPC max |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate_id in sorted(result["holdout_scores"], key=int):
            item = result["holdout_scores"][candidate_id]
            config = item["configuration"]
            lines.append(
                f"| {candidate_id} | {label(config)} | "
                f"{config.get('write_completion_parallel_lanes', 0)} | "
                f"{item['fence']['mae']:.1f} | {item['gpc']['mae']:.1f} | "
                f"{item['fence']['max_error']:.0f} | {item['gpc']['max_error']:.0f} |"
            )
        improving = result["candidates_improving_both_predecessor_targets"]
        robust = result["candidates_improving_both_all_repeat_sensitivity_labels"]
        lines.extend(
            [
                "",
                "The calibration frontier was frozen before holdout scoring. "
                f"Retrospective holdout Pareto IDs: {result['holdout_pareto_ids']}. "
                "Candidates improving both predecessor targets: "
                f"{improving if improving else 'none'}. Candidates retaining both "
                "improvements across the observed repeat-sensitivity labels: "
                f"{robust if robust else 'none'}.",
                "Repeat envelopes are sensitivity bounds, not confidence intervals.",
                "Schema 13 remains an opt-in discriminator; no request/region rate "
                "or completion-lane count is promoted.",
            ]
        )
    else:
        lines.extend(["", "Holdout replay is pending for the calibration-Pareto candidates."])
    lines.extend(["", "Machine-readable reductions are in `red_rmw_completion_sweep_summary.json`.", ""])
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
        (analysis / "red_rmw_completion_sweep_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (analysis / "red_rmw_completion_sweep.md").write_text(
            markdown(result), encoding="utf-8"
        )
        print(analysis / "red_rmw_completion_sweep.md")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"RED RMW completion analysis: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
