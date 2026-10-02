#!/usr/bin/env python3
"""Validate the accuracy-first SM86 NCU contract before data collection.

This validates a declared measurement contract and a family/workload roster.
It does not inspect NCU values, select workloads, or create an execution gate.
Historical manifests may use a different repetition/launch contract and must
remain labeled historical rather than being silently upgraded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FAMILIES = {"calibration", "development", "held_out"}


def validate_protocol(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != "sm86_ncu_accuracy_protocol_v1":
        raise ValueError("protocol schema must be sm86_ncu_accuracy_protocol_v1")
    if document.get("status") != "pre_measurement_contract":
        raise ValueError("protocol status must be pre_measurement_contract")
    measurement = document.get("measurement_contract")
    if not isinstance(measurement, dict):
        raise ValueError("measurement_contract must be an object")
    expected = {
        "target_metric": "gpc__cycles_elapsed.max",
        "ncu_chip": "ga102",
        "endpoint": "ncu_grid_completion_semaphore",
        "clock_domain": "gpc_cycles",
        "unit": "cycles",
        "cache_policy": "all",
        "replay_mode": "application",
        "clock_control": "none",
    }
    for field, value in expected.items():
        if measurement.get(field) != value:
            raise ValueError(f"measurement_contract.{field} must be {value!r}")
    if not isinstance(measurement.get("launch_scope"), str) or not measurement["launch_scope"].strip():
        raise ValueError("measurement_contract.launch_scope must be nonempty")
    if not isinstance(measurement.get("launch_identity"), str) or not measurement["launch_identity"].strip():
        raise ValueError("measurement_contract.launch_identity must be nonempty")
    if type(measurement.get("device_index")) is not int or measurement["device_index"] < 0:
        raise ValueError("measurement_contract.device_index must be nonnegative")
    if type(measurement.get("repetitions")) is not int or measurement["repetitions"] < 5:
        raise ValueError("measurement_contract.repetitions must be at least five")
    if type(measurement.get("warmup_launches")) is not int or measurement["warmup_launches"] < 0:
        raise ValueError("measurement_contract.warmup_launches must be nonnegative")
    for field in ("prediction_timeout_seconds", "ncu_timeout_seconds"):
        if type(measurement.get(field)) is not int or measurement[field] <= 0:
            raise ValueError(f"measurement_contract.{field} must be positive")

    identity = document.get("identity_requirements")
    if not isinstance(identity, dict) or identity.get("architecture") != "sm_86":
        raise ValueError("identity_requirements.architecture must be sm_86")
    if not isinstance(identity.get("expected_gpu_name"), str) or not identity["expected_gpu_name"].strip():
        raise ValueError("identity_requirements.expected_gpu_name must be nonempty")
    if identity.get("independent_check_required") is not True:
        raise ValueError("independent device identity checking is required")
    required_identity = identity.get("required_fields")
    if not isinstance(required_identity, list) or not required_identity:
        raise ValueError("identity_requirements.required_fields must be nonempty")
    if identity.get("mismatch_policy") != "exclude_or_label_historical":
        raise ValueError("identity mismatch policy must exclude_or_label_historical")

    artifacts = document.get("artifact_contract")
    if not isinstance(artifacts, dict):
        raise ValueError("artifact_contract must be an object")
    if artifacts.get("prediction_before_observation") is not True:
        raise ValueError("predictions must precede observation collection")
    if artifacts.get("ncu_values_forbidden_in_prediction") is not True:
        raise ValueError("NCU values must be forbidden in prediction artifacts")

    splits = document.get("split_contract", {})
    families = splits.get("families") if isinstance(splits, dict) else None
    if not isinstance(families, dict) or not families:
        raise ValueError("split_contract.families must be nonempty")
    if any(value not in FAMILIES for value in families.values()):
        raise ValueError("split_contract.families contains an unsupported split")
    if not all(splits.get(field) is True for field in ("family_held_out", "no_workload_crossing", "no_family_crossing")):
        raise ValueError("split contract closure flags must all be true")
    planned = splits.get("planned_expansion_families")
    if not isinstance(planned, dict) or not planned:
        raise ValueError("split_contract.planned_expansion_families must be nonempty")
    for family, spec in planned.items():
        if not isinstance(spec, dict) or spec.get("split") not in FAMILIES:
            raise ValueError(f"planned expansion family {family!r} has an invalid split")
        if not isinstance(spec.get("case_types"), list) or not spec["case_types"]:
            raise ValueError(f"planned expansion family {family!r} needs case_types")
    return document


def validate_manifest_splits(protocol: dict[str, Any], manifest: Any) -> None:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("rows"), list) or not manifest["rows"]:
        raise ValueError("manifest rows must be a nonempty array")
    family_splits = protocol["split_contract"]["families"]
    workload_split: dict[str, str] = {}
    workload_family: dict[str, str] = {}
    for index, row in enumerate(manifest["rows"]):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} must be an object")
        workload, family, split = row.get("workload"), row.get("family"), row.get("split")
        if not all(isinstance(value, str) and value.strip() for value in (workload, family)):
            raise ValueError(f"manifest row {index} needs workload and family")
        if split not in FAMILIES or family not in family_splits:
            raise ValueError(f"manifest row {index} has unsupported family/split")
        if family_splits[family] != split:
            raise ValueError(f"manifest row {index} family {family!r} violates protocol split")
        if workload in workload_split and workload_split[workload] != split:
            raise ValueError(f"workload {workload!r} crosses splits")
        if workload in workload_family and workload_family[workload] != family:
            raise ValueError(f"workload {workload!r} crosses families")
        workload_split[workload] = split
        workload_family[workload] = family


def validate_manifest_sources(manifest: Any, *, require_paths: bool = True) -> None:
    """Check source identity fields for every roster row before collection."""

    rows = manifest.get("rows") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest rows must be a nonempty array")
    required = ("trace", "control", "arch_card", "pchase_card")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} must be an object")
        workload = row.get("workload")
        if not isinstance(workload, str) or not workload.strip():
            raise ValueError(f"manifest row {index} needs workload")
        if workload in seen:
            raise ValueError(f"manifest repeats workload {workload!r}")
        seen.add(workload)
        for field in required:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"manifest row {index}.{field} must be nonempty")
            if require_paths and not Path(value).exists():
                raise ValueError(f"manifest row {index}.{field} does not exist: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    protocol = validate_protocol(json.loads(args.protocol.read_text()))
    if args.manifest is not None:
        validate_manifest_splits(protocol, json.loads(args.manifest.read_text()))
    print(json.dumps({"schema": protocol["schema"], "status": "valid"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
