#!/usr/bin/env python3
"""Build a post-prediction NCU observation document for a corpus manifest.

The command reads only profiler CSVs and a workload roster.  It does not read
predictions and cannot influence prediction materialization or model choice.
Prediction and observation documents therefore remain separate artifacts until
``evaluate_prediction_splits.py`` performs the explicit document-identity-consistent join.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

from prediction_contract import validate_observation_document, validate_target_contract


TARGET_METRIC = "gpc__cycles_elapsed.max"


def _numeric_values(path: Path, metric: str) -> list[float]:
    lines = path.read_text().splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.startswith('"ID"'))
    except StopIteration as error:
        raise ValueError(f"{path}: NCU CSV header row is missing") from error
    rows = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    values: list[float] = []
    for row in rows:
        raw = row.get(metric)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            # NCU emits a units row immediately after the header.
            continue
    return values


def build_observation_set(
    manifest: dict[str, Any],
    campaign_root: Path,
    *,
    source: str = "ncu_csv",
) -> dict[str, Any]:
    if manifest.get("schema") != "reification_prediction_manifest_v1":
        raise ValueError("manifest schema must be reification_prediction_manifest_v1")
    contract = validate_target_contract(
        manifest.get("target_contract"), "observation builder", require_hardware=True
    )
    if contract["metric"] != TARGET_METRIC:
        raise ValueError(f"observation metric must be {TARGET_METRIC!r}")
    # The CSV parser observes values, not GPU UUID/driver/clock provenance.
    # There is intentionally no caller-selectable "verified" status.
    identity_status = "manifest_labeled_historical"
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest rows must be nonempty")
    workloads: set[str] = set()
    observations: list[dict[str, Any]] = []
    for index, roster_row in enumerate(rows):
        workload = roster_row.get("workload") if isinstance(roster_row, dict) else None
        if not isinstance(workload, str) or not workload.strip():
            raise ValueError(f"manifest row {index} needs a workload")
        if workload in workloads:
            raise ValueError(f"manifest repeats workload {workload!r}")
        workloads.add(workload)
        workload_dir = campaign_root / workload
        if not workload_dir.is_dir():
            raise ValueError(f"workload {workload!r} has no campaign directory: {workload_dir}")
        csv_paths = sorted(workload_dir.glob("ncu_*_csv.stdout"))
        if len(csv_paths) != contract["repetitions"]:
            raise ValueError(
                f"workload {workload!r} has {len(csv_paths)} NCU CSV files; "
                f"contract declares {contract['repetitions']} repetitions"
            )
        values: list[float] = []
        for path in csv_paths:
            values.extend(_numeric_values(path, contract["metric"]))
        if len(values) != contract["repetitions"]:
            raise ValueError(
                f"workload {workload!r} has {len(values)} numeric {contract['metric']} rows; "
                f"contract declares {contract['repetitions']} repetitions"
            )
        source_binding = None
        if all(
            isinstance(roster_row.get(field), str) and roster_row[field].strip()
            for field in ("trace", "control", "arch_card", "pchase_card")
        ):
            source_binding = {
                "workload": workload,
                "trace": roster_row["trace"],
                "control": roster_row["control"],
                "arch_card": roster_row["arch_card"],
                "pchase_card": roster_row["pchase_card"],
                "launch_identity": contract["launch_identity"],
            }
        for value in values:
            row = {
                    "workload": workload,
                    "value": value,
                    "metric": contract["metric"],
                    "endpoint": contract["endpoint"],
                    "clock_domain": contract["clock_domain"],
                    "hardware": contract["hardware"],
                    "source": source,
                    "identity_status": identity_status,
                }
            if source_binding is not None:
                row["source_binding"] = source_binding
            observations.append(row)
    document = {
        "schema": "ncu_observation_v2",
        "source": source,
        "identity_status": identity_status,
        "target_contract": contract,
        "rows": observations,
    }
    return validate_observation_document(document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source", default="ncu_csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_observation_set(
        json.loads(args.manifest.read_text()),
        args.campaign_root,
        source=args.source,
    )
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
