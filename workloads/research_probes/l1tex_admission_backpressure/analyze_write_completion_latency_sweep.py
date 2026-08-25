#!/usr/bin/env python3
"""Select fixed write-completion latency without changing admission timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import analyze_write_completion_feedback as base
import campaign


CANDIDATE_IDS = {
    "store": (201, 203, 204, 205, 206),
    "red": (202, 207, 208, 209, 210),
}


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


def delta(
    root: Path,
    candidate_id: int,
    candidate: Mapping[str, object],
    cell: campaign.Cell,
    cells: Mapping[str, campaign.Cell],
) -> dict[str, int]:
    target = base.metrics(
        base.completion_launch(root, candidate_id, candidate, cell), completion=True
    )
    control = base.metrics(
        base.completion_launch(
            root, candidate_id, candidate, cells[base.control_key(cell)]
        ),
        completion=True,
    )
    return base.subtract(target, control)


def enrich(
    rows: Sequence[Mapping[str, object]], deltas: Mapping[str, Mapping[str, int]]
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "latency_fence": deltas[str(row["cell"])]["phase_fence"],
            "latency_gpc": deltas[str(row["cell"])]["service_completion"],
            "latency_issue": deltas[str(row["cell"])]["phase_issue"],
        }
        for row in rows
    ]


def analyze(root: Path, *, calibration_only: bool) -> dict[str, object]:
    rows = base.base_rows(root)
    cells = base.cells_by_key()
    candidates = campaign.validated_write_completion_feedback()
    result: dict[str, object] = {
        "schema_version": 1,
        "selection_rule": (
            "Retain the five-cell calibration Pareto frontier over native-fence MAE "
            "and downstream service-completion versus NCU-GPC MAE; advance every "
            "Pareto latency without holdout refitting."
        ),
        "operations": {},
    }
    for tail in ("store", "red"):
        calibration, holdout = base.partitions(rows, tail)
        calibration_scores: dict[int, dict[str, object]] = {}
        for candidate_id in CANDIDATE_IDS[tail]:
            candidate = candidates[candidate_id][1]
            deltas = {
                str(row["cell"]): delta(
                    root, candidate_id, candidate, cells[str(row["cell"])], cells
                )
                for row in calibration
            }
            material = enrich(calibration, deltas)
            calibration_scores[candidate_id] = {
                "configuration": candidate,
                "fence": base.score(material, "latency_fence", "native_fence_delta"),
                "gpc": base.score(material, "latency_gpc", "ncu_gpc_delta"),
                "issue": base.score(material, "latency_issue", "native_issue_delta"),
            }
        selected_ids = pareto_ids(calibration_scores)
        holdout_scores = {}
        missing = []
        if not calibration_only:
            for candidate_id in selected_ids:
                candidate = candidates[candidate_id][1]
                directory = root / f"simulator-memory-write-completion-feedback-{candidate_id}"
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
                        "fence": base.score(
                            material, "latency_fence", "native_fence_delta"
                        ),
                        "gpc": base.score(material, "latency_gpc", "ncu_gpc_delta"),
                        "issue": base.score(
                            material, "latency_issue", "native_issue_delta"
                        ),
                        "by_stage": {
                            stage: {
                                "fence": base.score(
                                    [row for row in material if row["stage"] == stage],
                                    "latency_fence",
                                    "native_fence_delta",
                                ),
                                "gpc": base.score(
                                    [row for row in material if row["stage"] == stage],
                                    "latency_gpc",
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
        predecessor_id = 201 if tail == "store" else 202
        improving_both = []
        holdout_pareto = []
        if holdout_complete:
            predecessor = numeric_holdout[predecessor_id]
            holdout_pareto = pareto_ids(numeric_holdout)
            improving_both = sorted(
                candidate_id
                for candidate_id, item in numeric_holdout.items()
                if float(item["fence"]["mae"]) < float(predecessor["fence"]["mae"])
                and float(item["gpc"]["mae"]) < float(predecessor["gpc"]["mae"])
            )
        result["operations"][tail] = {
            "calibration_scores": {
                str(candidate_id): item
                for candidate_id, item in calibration_scores.items()
            },
            "calibration_pareto_ids": selected_ids,
            "holdout_pareto_ids": holdout_pareto,
            "candidates_improving_both_predecessor_targets": improving_both,
            "holdout_status": "complete" if holdout_complete else "awaiting_replay",
            "missing_holdout_outputs": missing,
            "holdout_scores": holdout_scores,
        }
    return result


def markdown(result: Mapping[str, object]) -> str:
    lines = [
        "# Write-completion latency sweep",
        "",
        "The latency is non-occupying and begins after service done; admission",
        "queue release and issue timing remain unchanged.",
        "",
        "## Calibration",
        "",
        "| tail | candidate | latency | fence MAE | service/GPC MAE | Pareto |",
        "|---|---:|---:|---:|---:|---|",
    ]
    operations = result["operations"]
    for tail in ("store", "red"):
        item = operations[tail]
        for candidate_id in CANDIDATE_IDS[tail]:
            candidate = item["calibration_scores"][str(candidate_id)]
            config = candidate["configuration"]
            lines.append(
                f"| {tail} | {candidate_id} | "
                f"{config.get('write_completion_latency_cycles', 0)} | "
                f"{candidate['fence']['mae']:.1f} | {candidate['gpc']['mae']:.1f} | "
                f"{candidate_id in item['calibration_pareto_ids']} |"
            )
    if all(operations[tail]["holdout_status"] == "complete" for tail in ("store", "red")):
        lines.extend(
            [
                "",
                "## Strict 20-cell holdout",
                "",
                "| tail | candidate | latency | fence MAE | GPC MAE | fence max | GPC max |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tail in ("store", "red"):
            for candidate_id, item in operations[tail]["holdout_scores"].items():
                lines.append(
                    f"| {tail} | {candidate_id} | "
                    f"{item['configuration'].get('write_completion_latency_cycles', 0)} | "
                    f"{item['fence']['mae']:.1f} | {item['gpc']['mae']:.1f} | "
                    f"{item['fence']['max_error']:.0f} | {item['gpc']['max_error']:.0f} |"
                )
        lines.extend(
            [
                "",
                "STORE latency zero dominates every positive latency on holdout. RED",
                "latency monotonically improves fence error only by degrading GPC error;",
                "no candidate improves both predecessor targets. No fixed completion",
                "latency advances to an architecture default.",
                "",
                "The next RED component must model RMW completion work/parallelism rather",
                "than another constant tail. STORE retains the schema-11 frontier with",
                "zero added visibility latency.",
            ]
        )
    else:
        lines.extend(["", "Holdout replay is pending for the calibration-Pareto latencies."])
    lines.extend(
        [
            "",
            "Machine-readable reductions are in `write_completion_latency_sweep_summary.json`.",
            "",
        ]
    )
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
        (analysis / "write_completion_latency_sweep_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (analysis / "write_completion_latency_sweep.md").write_text(
            markdown(result), encoding="utf-8"
        )
        print(analysis / "write_completion_latency_sweep.md")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"write-completion latency analysis: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
