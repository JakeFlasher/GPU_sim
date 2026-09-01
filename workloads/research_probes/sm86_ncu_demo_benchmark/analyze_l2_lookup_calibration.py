#!/usr/bin/env python3
"""Replay an independent campaign and score the L2 lookup observer only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.sm86_ncu_demo_benchmark import validate_spec


L2_METRIC = "lts__t_sector_hit_rate.pct"


class CalibrationError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationError(f"{path}: root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metric_row(document: Mapping[str, Any]) -> Mapping[str, Any]:
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise CalibrationError("simulator output must contain exactly one launch")
    rows = launches[0].get("metric_predictions")
    if not isinstance(rows, list):
        raise CalibrationError("simulator metric projections are missing")
    matched = [row for row in rows if row.get("ncu_metric") == L2_METRIC]
    if len(matched) != 1:
        raise CalibrationError("simulator L2 metric projection is not unique")
    return matched[0]


def _prediction_percent(document: Mapping[str, Any]) -> float:
    outcome = _metric_row(document).get("outcome")
    if not isinstance(outcome, Mapping) or outcome.get("status") != "predicted":
        raise CalibrationError("simulator L2 metric is unavailable")
    value = outcome.get("value")
    if not isinstance(value, Mapping) or value.get("kind") != "fraction":
        raise CalibrationError("simulator L2 metric is not a fraction")
    fraction = value.get("value")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise CalibrationError("simulator L2 fraction is nonnumeric")
    result = float(fraction) * 100.0
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise CalibrationError("simulator L2 percentage is outside [0, 100]")
    return result


def _without_l2_outcome(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(document)
    row = _metric_row(normalized)
    assert isinstance(row, dict)
    row["outcome"] = {"status": "normalized_l2_observer"}
    return normalized


def _ncu_median(raw_root: Path, workload_id: str) -> tuple[float, list[float]]:
    values: list[float] = []
    repeat = 1
    while True:
        path = raw_root / "ncu" / f"rep-{repeat}" / workload_id / "normalized.json"
        if not path.is_file():
            break
        record = _load(path)["metrics"][L2_METRIC]
        if record.get("status") != "measured":
            raise CalibrationError(f"{workload_id}: NCU L2 metric is unavailable")
        value = float(record["value"])
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise CalibrationError(f"{workload_id}: NCU L2 metric is invalid")
        values.append(value)
        repeat += 1
    if len(values) < 3:
        raise CalibrationError(f"{workload_id}: at least three NCU repeats are required")
    return float(statistics.median(values)), values


def _run_one(
    simulator: Path,
    raw_root: Path,
    workload_id: str,
    repeats: int,
    destination: Path,
) -> tuple[dict[str, Any], list[float], list[dict[str, object]]]:
    trace = raw_root / "nvbit" / workload_id
    control = raw_root / "static" / workload_id / f"{workload_id}.control_bits.json"
    for path in (simulator, trace, control):
        if not path.exists():
            raise CalibrationError(f"missing replay input: {path}")
    command = [
        str(simulator),
        "--trace",
        str(trace),
        "--control",
        str(control),
        "--arch-card",
        "arch_profiles/cards/sm_86.json",
        "--pchase-card",
        "data/pchase/sm_86_static_card.json",
        "--mode",
        "l1-l2",
        "--ncu-shaped-metrics",
    ]
    documents: list[dict[str, Any]] = []
    walls: list[float] = []
    destination.mkdir(parents=True, exist_ok=True)
    for repeat in range(1, repeats + 1):
        started = time.perf_counter_ns()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
            check=False,
        )
        wall = (time.perf_counter_ns() - started) / 1_000_000_000
        (destination / f"replay-{repeat}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise CalibrationError(
                f"{workload_id}: simulator repeat {repeat} exited {completed.returncode}"
            )
        document = json.loads(completed.stdout)
        if not isinstance(document, dict):
            raise CalibrationError("simulator output root is not an object")
        (destination / f"replay-{repeat}.json").write_text(
            completed.stdout, encoding="utf-8"
        )
        documents.append(document)
        walls.append(wall)
    first_bytes = (destination / "replay-1.json").read_bytes()
    if any(
        (destination / f"replay-{repeat}.json").read_bytes() != first_bytes
        for repeat in range(2, repeats + 1)
    ):
        raise CalibrationError(f"{workload_id}: simulator repeats differ")
    artifacts = [
        {
            "path": _portable(destination / f"replay-{repeat}.json"),
            "sha256": _sha256(destination / f"replay-{repeat}.json"),
            "size_bytes": (destination / f"replay-{repeat}.json").stat().st_size,
        }
        for repeat in range(1, repeats + 1)
    ]
    return documents[0], walls, artifacts


def build_report(
    campaign_root: Path,
    output_root: Path,
    simulator: Path,
    repeats: int,
) -> dict[str, Any]:
    campaign = _load(campaign_root / "campaign_manifest.json")
    spec_record = campaign.get("inputs", {}).get("spec")
    metrics_record = campaign.get("inputs", {}).get("metrics")
    if not isinstance(spec_record, Mapping) or not isinstance(metrics_record, Mapping):
        raise CalibrationError("campaign input identities are missing")
    spec_path = REPO_ROOT / str(spec_record["path"])
    metrics_path = REPO_ROOT / str(metrics_record["path"])
    if _sha256(spec_path) != spec_record["sha256"] or _sha256(metrics_path) != metrics_record["sha256"]:
        raise CalibrationError("campaign input identity drift")
    spec = _load(spec_path)
    metrics = _load(metrics_path)
    profile = _load(HERE / "host_profile.json")
    validate_spec.validate_spec(spec, metrics, profile)
    raw_root = campaign_root / "evidence" / "raw"
    rows: list[dict[str, Any]] = []
    for workload in spec["workloads"]:
        workload_id = str(workload["id"])
        old_document = _load(raw_root / "simulator" / workload_id / "replay-1.json")
        new_document, walls, artifacts = _run_one(
            simulator,
            raw_root,
            workload_id,
            repeats,
            output_root / "simulator" / workload_id,
        )
        if _without_l2_outcome(old_document) != _without_l2_outcome(new_document):
            raise CalibrationError(
                f"{workload_id}: candidate changed output outside the L2 observer"
            )
        ncu, ncu_repeats = _ncu_median(raw_root, workload_id)
        predicted = _prediction_percent(new_document)
        ape = None if ncu == 0 else abs(predicted - ncu) / ncu * 100.0
        rows.append(
            {
                "workload_id": workload_id,
                "ncu_percent": ncu,
                "ncu_repeat_values": ncu_repeats,
                "simulator_percent": predicted,
                "absolute_percentage_point_error": abs(predicted - ncu),
                "ape_pct": ape,
                "simulator_repeat_wall_seconds": walls,
                "simulator_outputs": artifacts,
                "simulator_repeats_byte_identical": True,
                "non_l2_output_equal_to_predecessor": True,
            }
        )
    covered = [float(row["ape_pct"]) for row in rows if row["ape_pct"] is not None]
    pp = [float(row["absolute_percentage_point_error"]) for row in rows]
    return {
        "schema_version": 1,
        "milestone": "sm86_l2_lookup_observer_calibration",
        "status": "calibration_only_not_gate",
        "source_campaign": {
            "path": _portable(campaign_root / "campaign_manifest.json"),
            "sha256": _sha256(campaign_root / "campaign_manifest.json"),
        },
        "simulator": {
            "path": _portable(simulator),
            "sha256": _sha256(simulator),
            "repeats_per_workload": repeats,
        },
        "observer_rule": "valid data hits + pending-read merges + all write-class sector lookups",
        "scientific_boundary": {
            "scored_gate_rows_used": False,
            "cache_state_changed": False,
            "timing_changed": False,
            "dram_traffic_changed": False,
        },
        "rows": rows,
        "summary": {
            "workloads": len(rows),
            "coverage": len(covered),
            "mape_pct": statistics.fmean(covered),
            "mean_absolute_percentage_point_error": statistics.fmean(pp),
            "maximum_ape_pct": max(covered),
        },
    }


def validate_retained(report: Mapping[str, Any]) -> None:
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CalibrationError("retained report has no rows")
    apes: list[float] = []
    pp: list[float] = []
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
                or _sha256(path) != artifact["sha256"]
            ):
                raise CalibrationError(f"retained simulator artifact drift: {path}")
            payloads.append(path.read_bytes())
        if any(payload != payloads[0] for payload in payloads[1:]):
            raise CalibrationError("retained simulator outputs are not byte-identical")
        ncu = float(row["ncu_percent"])
        simulator = float(row["simulator_percent"])
        expected_pp = abs(simulator - ncu)
        if not math.isclose(
            expected_pp,
            float(row["absolute_percentage_point_error"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise CalibrationError("retained percentage-point error differs")
        pp.append(expected_pp)
        if ncu > 0:
            expected_ape = expected_pp / ncu * 100.0
            if not math.isclose(
                expected_ape, float(row["ape_pct"]), rel_tol=0, abs_tol=1e-12
            ):
                raise CalibrationError("retained APE differs")
            apes.append(expected_ape)
    summary = report.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("workloads") != len(rows)
        or summary.get("coverage") != len(apes)
        or not math.isclose(
            float(summary["mape_pct"]), statistics.fmean(apes), rel_tol=0, abs_tol=1e-12
        )
        or not math.isclose(
            float(summary["mean_absolute_percentage_point_error"]),
            statistics.fmean(pp),
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise CalibrationError("retained summary differs from rows")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
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
        output = args.output.resolve()
        report_path = output / "report.json"
        if args.validate_only:
            report = _load(report_path)
            validate_retained(report)
        else:
            report = build_report(
                campaign,
                output,
                args.simulator.resolve(),
                args.simulator_repeats,
            )
            _write(report_path, report)
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    except (CalibrationError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"l2 lookup calibration: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
