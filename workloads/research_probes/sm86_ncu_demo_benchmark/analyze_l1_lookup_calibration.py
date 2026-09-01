#!/usr/bin/env python3
"""Replay an independent campaign and score the two L1TEX lookup observers."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.sm86_ncu_demo_benchmark import (
    analyze_l2_lookup_calibration as common,
)
from workloads.research_probes.sm86_ncu_demo_benchmark import validate_spec


UNIFIED_METRIC = "l1tex__t_sector_hit_rate.pct"
READ_METRIC = "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"
L1_METRICS = (UNIFIED_METRIC, READ_METRIC)
CalibrationError = common.CalibrationError


def _metric_row(document: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise CalibrationError("simulator output must contain exactly one launch")
    rows = launches[0].get("metric_predictions")
    if not isinstance(rows, list):
        raise CalibrationError("simulator metric projections are missing")
    matched = [row for row in rows if row.get("ncu_metric") == metric]
    if len(matched) != 1:
        raise CalibrationError(f"simulator metric projection is not unique: {metric}")
    return matched[0]


def _prediction_percent(document: Mapping[str, Any], metric: str) -> float:
    outcome = _metric_row(document, metric).get("outcome")
    if not isinstance(outcome, Mapping) or outcome.get("status") != "predicted":
        raise CalibrationError(f"simulator metric is unavailable: {metric}")
    value = outcome.get("value")
    if not isinstance(value, Mapping) or value.get("kind") != "fraction":
        raise CalibrationError(f"simulator metric is not a fraction: {metric}")
    fraction = value.get("value")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise CalibrationError(f"simulator fraction is nonnumeric: {metric}")
    result = float(fraction) * 100.0
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise CalibrationError(f"simulator percentage is outside [0, 100]: {metric}")
    return result


def _without_l1_outcomes(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(document)
    for metric in L1_METRICS:
        row = _metric_row(normalized, metric)
        assert isinstance(row, dict)
        row["outcome"] = {"status": "normalized_l1_observer"}
    return normalized


def _ncu_median(
    raw_root: Path, workload_id: str, metric: str
) -> tuple[float, list[float]]:
    values: list[float] = []
    repeat = 1
    while True:
        path = raw_root / "ncu" / f"rep-{repeat}" / workload_id / "normalized.json"
        if not path.is_file():
            break
        record = common._load(path)["metrics"][metric]
        if record.get("status") != "measured":
            raise CalibrationError(f"{workload_id}: NCU metric unavailable: {metric}")
        value = float(record["value"])
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise CalibrationError(f"{workload_id}: NCU metric invalid: {metric}")
        values.append(value)
        repeat += 1
    if len(values) < 3:
        raise CalibrationError(f"{workload_id}: at least three NCU repeats are required")
    return float(statistics.median(values)), values


def _score(ncu: float, simulator: float) -> dict[str, float | None]:
    pp = abs(simulator - ncu)
    return {
        "absolute_percentage_point_error": pp,
        "ape_pct": None if ncu == 0 else pp / ncu * 100.0,
    }


def build_report(
    campaign_root: Path,
    predecessor_root: Path,
    output_root: Path,
    simulator: Path,
    repeats: int,
) -> dict[str, Any]:
    campaign = common._load(campaign_root / "campaign_manifest.json")
    spec_record = campaign.get("inputs", {}).get("spec")
    metrics_record = campaign.get("inputs", {}).get("metrics")
    if not isinstance(spec_record, Mapping) or not isinstance(metrics_record, Mapping):
        raise CalibrationError("campaign input identities are missing")
    spec_path = REPO_ROOT / str(spec_record["path"])
    metrics_path = REPO_ROOT / str(metrics_record["path"])
    if (
        common._sha256(spec_path) != spec_record["sha256"]
        or common._sha256(metrics_path) != metrics_record["sha256"]
    ):
        raise CalibrationError("campaign input identity drift")
    spec = common._load(spec_path)
    metrics = common._load(metrics_path)
    profile = common._load(HERE / "host_profile.json")
    validate_spec.validate_spec(spec, metrics, profile)

    raw_root = campaign_root / "evidence" / "raw"
    rows: list[dict[str, Any]] = []
    for workload in spec["workloads"]:
        workload_id = str(workload["id"])
        old_document = common._load(
            predecessor_root / "simulator" / workload_id / "replay-1.json"
        )
        new_document, walls, artifacts = common._run_one(
            simulator,
            raw_root,
            workload_id,
            repeats,
            output_root / "simulator" / workload_id,
        )
        if _without_l1_outcomes(old_document) != _without_l1_outcomes(new_document):
            raise CalibrationError(
                f"{workload_id}: candidate changed output outside the two L1 observers"
            )

        measurements: dict[str, Any] = {}
        for name, metric in (("unified", UNIFIED_METRIC), ("read", READ_METRIC)):
            ncu, ncu_repeats = _ncu_median(raw_root, workload_id, metric)
            predicted = _prediction_percent(new_document, metric)
            measurements[name] = {
                "metric": metric,
                "ncu_percent": ncu,
                "ncu_repeat_values": ncu_repeats,
                "simulator_percent": predicted,
                **_score(ncu, predicted),
            }
        rows.append(
            {
                "workload_id": workload_id,
                "measurements": measurements,
                "simulator_repeat_wall_seconds": walls,
                "simulator_outputs": artifacts,
                "simulator_repeats_byte_identical": True,
                "non_l1_output_equal_to_predecessor": True,
            }
        )

    def summary_for(name: str) -> dict[str, float | int]:
        measurements = [row["measurements"][name] for row in rows]
        covered = [float(row["ape_pct"]) for row in measurements if row["ape_pct"] is not None]
        pp = [float(row["absolute_percentage_point_error"]) for row in measurements]
        return {
            "workloads": len(measurements),
            "coverage": len(covered),
            "mape_pct": statistics.fmean(covered),
            "mean_absolute_percentage_point_error": statistics.fmean(pp),
            "maximum_ape_pct": max(covered),
        }

    return {
        "schema_version": 1,
        "milestone": "sm86_l1_lookup_observer_calibration",
        "status": "calibration_only_not_gate",
        "source_campaign": {
            "path": common._portable(campaign_root / "campaign_manifest.json"),
            "sha256": common._sha256(campaign_root / "campaign_manifest.json"),
        },
        "predecessor": {
            "path": common._portable(predecessor_root / "report.json"),
            "sha256": common._sha256(predecessor_root / "report.json"),
        },
        "simulator": {
            "path": common._portable(simulator),
            "sha256": common._sha256(simulator),
            "repeats_per_workload": repeats,
        },
        "observer_rule": {
            "read": "resident sector hits + cache-all LDGSTS merges into allocations predating the warp instruction",
            "write": "64 B allocation within each ideal L1TEX write-class wavefront",
            "excluded": "same-instruction sector coalescing",
        },
        "scientific_boundary": {
            "scored_gate_rows_used": False,
            "cache_state_changed": False,
            "timing_changed": False,
            "l2_changed": False,
            "dram_traffic_changed": False,
        },
        "rows": rows,
        "summary": {
            "unified": summary_for("unified"),
            "read": summary_for("read"),
        },
    }


def validate_retained(report: Mapping[str, Any]) -> None:
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CalibrationError("retained report has no rows")
    for row in rows:
        outputs = row.get("simulator_outputs")
        if not isinstance(outputs, list) or len(outputs) < 3:
            raise CalibrationError("retained row has fewer than three simulator outputs")
        payloads: list[bytes] = []
        for artifact in outputs:
            path = REPO_ROOT / str(artifact["path"])
            if (
                not path.is_file()
                or path.stat().st_size != artifact["size_bytes"]
                or common._sha256(path) != artifact["sha256"]
            ):
                raise CalibrationError(f"retained simulator artifact drift: {path}")
            payloads.append(path.read_bytes())
        if any(payload != payloads[0] for payload in payloads[1:]):
            raise CalibrationError("retained simulator outputs are not byte-identical")
        for measurement in row["measurements"].values():
            expected = _score(
                float(measurement["ncu_percent"]),
                float(measurement["simulator_percent"]),
            )
            for key, value in expected.items():
                retained = measurement[key]
                if value is None:
                    if retained is not None:
                        raise CalibrationError("retained zero-target APE must be null")
                elif not math.isclose(float(retained), value, rel_tol=0, abs_tol=1e-12):
                    raise CalibrationError(f"retained measurement differs: {key}")

    for name in ("unified", "read"):
        measurements = [row["measurements"][name] for row in rows]
        covered = [float(row["ape_pct"]) for row in measurements if row["ape_pct"] is not None]
        pp = [float(row["absolute_percentage_point_error"]) for row in measurements]
        summary = report["summary"][name]
        expected = {
            "workloads": len(measurements),
            "coverage": len(covered),
            "mape_pct": statistics.fmean(covered),
            "mean_absolute_percentage_point_error": statistics.fmean(pp),
            "maximum_ape_pct": max(covered),
        }
        for key, value in expected.items():
            retained = summary[key]
            if isinstance(value, int):
                if retained != value:
                    raise CalibrationError(f"retained {name} summary differs: {key}")
            elif not math.isclose(float(retained), value, rel_tol=0, abs_tol=1e-12):
                raise CalibrationError(f"retained {name} summary differs: {key}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, default=Path("sim/target/release/sim-run"))
    parser.add_argument("--simulator-repeats", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not 3 <= args.simulator_repeats <= 16:
            raise CalibrationError("--simulator-repeats must be within 3..16")
        campaign = args.campaign.resolve()
        predecessor = args.predecessor.resolve()
        output = args.output.resolve()
        report_path = output / "report.json"
        if args.validate_only:
            report = common._load(report_path)
            validate_retained(report)
        else:
            report = build_report(
                campaign,
                predecessor,
                output,
                args.simulator.resolve(),
                args.simulator_repeats,
            )
            common._write(report_path, report)
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    except (
        CalibrationError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"l1 lookup calibration: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
