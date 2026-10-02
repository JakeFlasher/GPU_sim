"""Shared validation for pre-NCU predictions and post-NCU observations.

The contract identifies what is being compared.  It is descriptive metadata,
not an execution or approval gate; a caller may choose a different declared
hardware/run contract, but the prediction and observation must agree before a
join is reported.
"""

from __future__ import annotations

from typing import Any


TARGET_METRIC = "gpc__cycles_elapsed.max"
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


def validate_target_contract(contract: Any, label: str, *, require_hardware: bool = True) -> dict[str, Any]:
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
            f"{label} target_contract.metric must be {TARGET_METRIC!r}, got {contract['metric']!r}"
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
            raise ValueError(f"{label} target_contract.hardware missing {sorted(missing_hardware)}")
        if not isinstance(hardware["gpu_uuid"], str) or not hardware["gpu_uuid"].strip():
            raise ValueError(f"{label} target_contract.hardware.gpu_uuid must be nonempty")
        if not isinstance(hardware["compute_capability"], str) or not hardware["compute_capability"].strip():
            raise ValueError(f"{label} target_contract.hardware.compute_capability must be nonempty")
        if isinstance(hardware["sm_clock_mhz"], bool) or not isinstance(hardware["sm_clock_mhz"], int) or hardware["sm_clock_mhz"] <= 0:
            raise ValueError(f"{label} target_contract.hardware.sm_clock_mhz must be positive")
        if isinstance(hardware["device_index"], bool) or not isinstance(hardware["device_index"], int) or hardware["device_index"] < 0:
            raise ValueError(f"{label} target_contract.hardware.device_index must be nonnegative")
    return contract


def compare_target_contracts(prediction: dict[str, Any], observation: dict[str, Any]) -> None:
    """Reject mismatches in the declared observer and hardware contract."""
    if prediction != observation:
        differences = sorted(
            key for key in set(prediction).union(observation) if prediction.get(key) != observation.get(key)
        )
        raise ValueError(f"prediction/observation target contract mismatch in {differences}")
