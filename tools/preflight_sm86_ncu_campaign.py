#!/usr/bin/env python3
"""Preflight an accuracy-first NCU campaign without profiling a workload.

The result is a diagnostic report. It does not create an approval gate or
prevent unrelated simulator execution. A campaign should not collect NCU data
until ``ready_for_collection`` is true.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from prediction_contract import validate_prediction_document, validate_target_contract
from validate_sm86_ncu_accuracy_protocol import (
    validate_manifest_sources,
    validate_manifest_splits,
    validate_protocol,
)


def summarize_repetitions(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("repetition values must be nonempty")
    ordered = sorted(samples)
    q1, median, q3 = (
        statistics.quantiles(ordered, n=4, method="inclusive")[0]
        if len(ordered) > 1
        else ordered[0],
        statistics.median(ordered),
        statistics.quantiles(ordered, n=4, method="inclusive")[2]
        if len(ordered) > 1
        else ordered[0],
    )
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "all_samples": ordered,
    }


def validate_run_records(
    records: Iterable[dict[str, Any]], *, minimum: int, timeout_seconds: int
) -> dict[str, Any]:
    """Summarize already-recorded repetition process outcomes.

    This consumes metadata/fixtures only; it does not launch NCU. A future
    campaign runner can write the same record shape after each bounded run.
    """

    rows = list(records)
    failures: list[str] = []
    values: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"record_{index}_not_object")
            continue
        if row.get("returncode") != 0:
            failures.append(f"record_{index}_returncode_{row.get('returncode')}")
        if row.get("timed_out") is True:
            failures.append(f"record_{index}_timed_out")
        elapsed = row.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or elapsed < 0 or elapsed > timeout_seconds:
            failures.append(f"record_{index}_timeout_budget")
        value = row.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            failures.append(f"record_{index}_value_missing")
        else:
            values.append(float(value))
    if len(rows) < minimum:
        failures.append(f"repetition_count_{len(rows)}_below_{minimum}")
    summary = summarize_repetitions(values) if not failures and values else None
    return {
        "termination_ok": not failures,
        "failures": failures,
        "repetitions": summary,
    }


def _same_contract(protocol: dict[str, Any], target: dict[str, Any]) -> list[str]:
    measurement = protocol["measurement_contract"]
    expected = {
        "metric": measurement["target_metric"],
        "endpoint": measurement["endpoint"],
        "clock_domain": measurement["clock_domain"],
        "cache_policy": measurement["cache_policy"],
        "launch_identity": measurement["launch_identity"],
        "repetitions": measurement["repetitions"],
    }
    return [
        f"target_contract.{field}={target.get(field)!r}, expected {value!r}"
        for field, value in expected.items()
        if target.get(field) != value
    ]


def preflight_campaign(
    protocol: dict[str, Any],
    identity: dict[str, Any],
    manifest: dict[str, Any],
    *,
    prediction: dict[str, Any] | None = None,
    observation_path: Path | None = None,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol)
    mismatches: list[str] = []
    validate_manifest_splits(protocol, manifest)
    try:
        validate_manifest_sources(manifest, require_paths=True)
    except ValueError as error:
        mismatches.append(f"manifest_sources:{error}")
    measurement = protocol["measurement_contract"]
    if identity.get("schema") != "sm86_ncu_identity_v1":
        mismatches.append("identity_schema")
    if identity.get("status") != "independently_verified" or not identity.get("eligible_for_accuracy"):
        mismatches.append("identity_ineligible")
    if identity.get("build", {}).get("working_tree_dirty") is not False:
        mismatches.append("working_tree_dirty")
    if identity.get("metric_availability", {}).get("available") is not True:
        mismatches.append("target_metric_unavailable")
    if identity.get("measurement_contract") != measurement:
        mismatches.append("measurement_contract_mismatch")
    target = manifest.get("target_contract")
    if not isinstance(target, dict):
        mismatches.append("manifest_target_contract_missing")
    else:
        try:
            validate_target_contract(target, "campaign manifest", require_hardware=True)
            mismatches.extend(f"manifest_{item}" for item in _same_contract(protocol, target))
        except ValueError as error:
            mismatches.append(f"manifest_target_contract:{error}")
    rows = manifest.get("rows", [])
    for index, row in enumerate(rows):
        for field in ("trace", "control", "arch_card", "pchase_card"):
            if not isinstance(row.get(field), str) or not Path(row[field]).exists():
                mismatches.append(f"manifest_row_{index}_missing_{field}")
    prediction_status = "not_checked"
    if prediction is not None:
        try:
            validate_prediction_document(prediction)
            prediction_status = "valid"
            mismatches.extend(f"prediction_{item}" for item in _same_contract(protocol, prediction["target_contract"]))
        except ValueError as error:
            prediction_status = "invalid"
            mismatches.append(f"prediction:{error}")
    observation_status = "not_present"
    if observation_path is not None and observation_path.exists():
        observation_status = "present_before_collection"
        mismatches.append("observation_already_present_before_collection")
    return {
        "schema": "sm86_ncu_campaign_preflight_v1",
        "ready_for_collection": not mismatches and prediction_status == "valid",
        "mismatches": mismatches,
        "prediction_status": prediction_status,
        "observation_status": observation_status,
        "repetition_policy": {
            "minimum": measurement["repetitions"],
            "warmup_launches": measurement["warmup_launches"],
            "prediction_timeout_seconds": measurement["prediction_timeout_seconds"],
            "ncu_timeout_seconds": measurement["ncu_timeout_seconds"],
            "summary": "median_iqr_and_all_samples",
        },
        "measurement_contract": measurement,
        "collection_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("identity", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = preflight_campaign(
        json.loads(args.protocol.read_text()),
        json.loads(args.identity.read_text()),
        json.loads(args.manifest.read_text()),
        prediction=json.loads(args.prediction.read_text()) if args.prediction else None,
        observation_path=args.observation,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ready_for_collection": report["ready_for_collection"], "mismatches": report["mismatches"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
