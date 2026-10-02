"""Shared validation for pre-observation predictions and observations.

The target contract describes the quantity being compared.  It is a data
boundary, not an execution, approval, or provenance gate: a caller may choose
another declared run contract, but prediction and observation documents must
agree before a join is reported.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


TARGET_METRIC = "gpc__cycles_elapsed.max"
SPLITS = {"calibration", "development", "held_out"}
PREDICTION_STATUSES = {"predicted", "refused"}
REQUIRED_CONTRACT_FIELDS = {
    "metric",
    "endpoint",
    "clock_domain",
    "cache_policy",
    "launch_identity",
    "repetitions",
}
REQUIRED_HARDWARE_FIELDS = {
    "gpu_uuid",
    "compute_capability",
    "sm_clock_mhz",
    "device_index",
}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def validate_target_contract(
    contract: Any, label: str, *, require_hardware: bool = True
) -> dict[str, Any]:
    """Validate the declared metric, observer, repetition, and hardware key."""

    if not isinstance(contract, dict):
        raise ValueError(f"{label} target_contract must be an object")
    missing = REQUIRED_CONTRACT_FIELDS.difference(contract)
    if missing:
        raise ValueError(f"{label} target_contract missing {sorted(missing)}")
    for field in ("metric", "endpoint", "clock_domain", "cache_policy", "launch_identity"):
        value = contract.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} target_contract.{field} must be a nonempty string")
    if contract["metric"] != TARGET_METRIC:
        raise ValueError(
            f"{label} target_contract.metric must be {TARGET_METRIC!r}, "
            f"got {contract['metric']!r}"
        )
    repetitions = contract.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError(f"{label} target_contract.repetitions must be positive")
    hardware = contract.get("hardware")
    if require_hardware:
        if not isinstance(hardware, dict):
            raise ValueError(f"{label} target_contract.hardware must be an object")
        missing_hardware = REQUIRED_HARDWARE_FIELDS.difference(hardware)
        if missing_hardware:
            raise ValueError(
                f"{label} target_contract.hardware missing {sorted(missing_hardware)}"
            )
        if not isinstance(hardware["gpu_uuid"], str) or not hardware["gpu_uuid"].strip():
            raise ValueError(f"{label} target_contract.hardware.gpu_uuid must be nonempty")
        if (
            not isinstance(hardware["compute_capability"], str)
            or not hardware["compute_capability"].strip()
        ):
            raise ValueError(
                f"{label} target_contract.hardware.compute_capability must be nonempty"
            )
        if (
            isinstance(hardware["sm_clock_mhz"], bool)
            or not isinstance(hardware["sm_clock_mhz"], int)
            or hardware["sm_clock_mhz"] <= 0
        ):
            raise ValueError(
                f"{label} target_contract.hardware.sm_clock_mhz must be positive"
            )
        if (
            isinstance(hardware["device_index"], bool)
            or not isinstance(hardware["device_index"], int)
            or hardware["device_index"] < 0
        ):
            raise ValueError(
                f"{label} target_contract.hardware.device_index must be nonnegative"
            )
    return contract


def validate_split_roster(rows: Any, label: str) -> tuple[dict[str, str], dict[str, str]]:
    """Validate workload/family split closure and return split maps.

    Family-held-out evaluation is meaningful only if a family occurs in one
    split.  A workload may have multiple interventions, but it may not move
    between splits or families.
    """

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} rows must be a nonempty array")
    workload_splits: dict[str, str] = {}
    workload_families: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} must be an object")
        workload = row.get("workload")
        family = row.get("family")
        split = row.get("split")
        if not isinstance(workload, str) or not workload.strip():
            raise ValueError(f"{label} row {index} needs a nonempty workload")
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f"{label} row {index} needs a nonempty family")
        if split not in SPLITS:
            raise ValueError(f"{label} row {index} has unsupported split {split!r}")
        prior_split = workload_splits.setdefault(workload, split)
        if prior_split != split:
            raise ValueError(f"workload {workload!r} crosses prediction splits")
        prior_family = workload_families.setdefault(workload, family)
        if prior_family != family:
            raise ValueError(f"workload {workload!r} appears with multiple families")
        prior_family_split = family_splits.setdefault(family, split)
        if prior_family_split != split:
            raise ValueError(f"family {family!r} crosses prediction splits")
    return workload_splits, workload_families


def validate_prediction_row(row: Any, index: int) -> None:
    """Validate one pre-observation prediction row without reading targets."""

    if not isinstance(row, dict):
        raise ValueError(f"prediction row {index} must be an object")
    observed_forbidden = {"ncu", "observed", "observation", "target", "target_value"}
    if observed_forbidden.intersection(row):
        raise ValueError("prediction rows must not contain observed target fields")
    status = row.get("status")
    if status not in PREDICTION_STATUSES:
        raise ValueError(f"prediction row {index} has unsupported status {status!r}")
    intervention = row.get("intervention", "default")
    if not isinstance(intervention, str) or not intervention.strip():
        raise ValueError(f"prediction row {index}.intervention must be nonempty")
    if status == "refused":
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"refused prediction row {index} needs a nonempty reason")
        if "prediction" in row:
            raise ValueError(f"refused prediction row {index} must not contain prediction")
        return
    prediction = row.get("prediction")
    if not isinstance(prediction, dict):
        raise ValueError(f"predicted row {index} lacks prediction interval")
    point = _finite_number(prediction.get("point"), f"prediction row {index}.point")
    lower = _finite_number(prediction.get("lower"), f"prediction row {index}.lower")
    upper_raw = prediction.get("upper")
    upper = None if upper_raw is None else _finite_number(
        upper_raw, f"prediction row {index}.upper"
    )
    if lower > (upper if upper is not None else math.inf):
        raise ValueError(f"prediction row {index} interval is inverted")
    if point < lower or (upper is not None and point > upper):
        raise ValueError(f"prediction row {index} point is outside its interval")
    projection = row.get("observer_projection")
    if not isinstance(projection, dict):
        raise ValueError(f"predicted row {index} lacks observer_projection")
    for field in ("target_metric", "role", "conversion"):
        value = projection.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prediction row {index}.observer_projection.{field} must be nonempty")
    if projection["target_metric"] != TARGET_METRIC:
        raise ValueError(
            f"prediction row {index}.observer_projection.target_metric must be {TARGET_METRIC!r}"
        )
    projected_lower = projection.get("predicted_lower_cycles")
    if isinstance(projected_lower, bool) or not isinstance(projected_lower, int):
        raise ValueError(
            f"prediction row {index}.observer_projection.predicted_lower_cycles must be an integer"
        )
    projected_upper = projection.get("predicted_upper_cycles")
    if projected_upper is not None and (
        isinstance(projected_upper, bool) or not isinstance(projected_upper, int)
    ):
        raise ValueError(
            f"prediction row {index}.observer_projection.predicted_upper_cycles must be an integer or null"
        )
    if projected_lower != lower or projected_upper != upper:
        raise ValueError(
            f"prediction row {index} interval disagrees with observer_projection"
        )
    for field in ("simulator_lower_cycles", "simulator_upper_cycles"):
        value = projection.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(
                f"prediction row {index}.observer_projection.{field} must be an integer or null"
            )
    for field in ("runtime_ms", "memory_bytes", "step_reduction"):
        if row.get(field) is not None:
            _finite_number(row[field], f"prediction row {index}.{field}")


def validate_prediction_document(document: Any) -> dict[str, Any]:
    """Validate the executable output document before it is materialized."""

    if not isinstance(document, dict) or document.get("schema") != "reification_prediction_v2":
        raise ValueError("prediction document schema must be reification_prediction_v2")
    validate_target_contract(document.get("target_contract"), "prediction", require_hardware=True)
    rows = document.get("rows")
    validate_split_roster(rows, "prediction")
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        validate_prediction_row(row, index)
        identity = (row["workload"], row.get("intervention", "default"))
        if identity in identities:
            raise ValueError(f"prediction repeats workload/intervention {identity!r}")
        identities.add(identity)
    return document


def validate_observation_row(
    row: Any, index: int, target_contract: dict[str, Any]
) -> None:
    """Require row-level observer and hardware identity, not caller discipline."""

    if (
        not isinstance(row, dict)
        or not isinstance(row.get("workload"), str)
        or not row["workload"].strip()
    ):
        raise ValueError(f"observation row {index} needs a nonempty workload")
    for field in ("metric", "endpoint", "clock_domain"):
        if row.get(field) != target_contract[field]:
            raise ValueError(f"observation row {index}.{field} disagrees with target_contract")
    if row.get("hardware") != target_contract["hardware"]:
        raise ValueError(f"observation row {index}.hardware disagrees with target_contract")
    _finite_number(row.get("value"), f"observation row {index}.value")


def validate_observation_document(document: Any, *, allow_empty: bool = False) -> dict[str, Any]:
    """Validate post-prediction observations and their row-level identity."""

    if not isinstance(document, dict) or document.get("schema") != "ncu_observation_v2":
        raise ValueError("observation document schema must be ncu_observation_v2")
    target_contract = validate_target_contract(
        document.get("target_contract"), "observation", require_hardware=True
    )
    rows = document.get("rows")
    if not isinstance(rows, list) or (not rows and not allow_empty):
        raise ValueError("observation rows must be a nonempty array")
    for index, row in enumerate(rows):
        validate_observation_row(row, index, target_contract)
    return document


def validate_observation_workloads(
    rows: Iterable[dict[str, Any]], prediction_workloads: set[str]
) -> None:
    """Reject observations that cannot be joined to a declared workload."""

    unknown = sorted({row["workload"] for row in rows} - prediction_workloads)
    if unknown:
        raise ValueError(f"observation contains workloads absent from predictions: {unknown}")


def compare_target_contracts(prediction: dict[str, Any], observation: dict[str, Any]) -> None:
    """Reject mismatches in the declared observer and hardware contract."""

    if prediction != observation:
        differences = sorted(
            key
            for key in set(prediction).union(observation)
            if prediction.get(key) != observation.get(key)
        )
        raise ValueError(f"prediction/observation target contract mismatch in {differences}")
