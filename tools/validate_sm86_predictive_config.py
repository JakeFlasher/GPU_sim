#!/usr/bin/env python3
"""Validate the compact SM86 predictive-closure configuration.

This is a structural parser, not an approval or admission workflow.  It keeps
configuration errors close to the source while leaving NCU labels outside the
model input contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "sm86_predictive_closure_config_v1"
SCOPE_POLICY = "analytical_or_statistical_modeling_with_unobservable_internals_black_box"
REQUIRED = (
    "schema",
    "hardware",
    "input_contract",
    "route_quotient",
    "cache_quotient",
    "endpoint_projection",
    "validation",
)


def _nonempty(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def validate_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")
    missing = [field for field in REQUIRED if field not in raw]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if raw["schema"] != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")

    hardware = raw["hardware"]
    if not isinstance(hardware, dict):
        raise ValueError("hardware must be an object")
    for field in ("gpu", "compute_capability", "driver"):
        _nonempty(hardware.get(field), f"hardware.{field}")
    for field in ("sm_clock_mhz", "memory_clock_mhz"):
        value = hardware.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"hardware.{field} must be a positive integer")

    inputs = raw["input_contract"]
    if not isinstance(inputs, dict):
        raise ValueError("input_contract must be an object")
    if inputs.get("ncu_labels") != "score_only_after_prediction":
        raise ValueError("input_contract.ncu_labels must keep NCU score-only")

    route = raw["route_quotient"]
    if not isinstance(route, dict) or route.get("mode") != "abstract_candidate_resource_pools":
        raise ValueError("route_quotient.mode must be abstract_candidate_resource_pools")
    if route.get("physical_route_labels") != "unknown":
        raise ValueError("route_quotient must preserve physical route identity as unknown")

    cache = raw["cache_quotient"]
    if not isinstance(cache, dict):
        raise ValueError("cache_quotient must be an object")
    if cache.get("hardware_dram_completion") != "unobserved":
        raise ValueError("cache_quotient must keep hardware DRAM completion unobserved")

    endpoint = raw["endpoint_projection"]
    if not isinstance(endpoint, dict):
        raise ValueError("endpoint_projection must be an object")
    if endpoint.get("ncu_target") != "gpc__cycles_elapsed.max":
        raise ValueError("endpoint_projection.ncu_target must be gpc__cycles_elapsed.max")
    if endpoint.get("cross_domain_cycle_conversion") != "forbidden_without_declared_contract":
        raise ValueError("endpoint projection must forbid undeclared cross-domain conversion")

    validation = raw["validation"]
    if not isinstance(validation, dict):
        raise ValueError("validation must be an object")
    if validation.get("score_split") != "family_held_out":
        raise ValueError("validation.score_split must be family_held_out")
    if validation.get("ncu_same_workload_features") is not False:
        raise ValueError("validation must forbid same-workload NCU features")
    interval_target = validation.get("interval_target")
    if not isinstance(interval_target, (int, float)) or not 0 < interval_target < 1:
        raise ValueError("validation.interval_target must be between 0 and 1")
    _nonempty(raw.get("service_rate_profile"), "service_rate_profile")
    if raw.get("scope_policy") != SCOPE_POLICY:
        raise ValueError("scope_policy must keep unobservable internals black-box")

    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        raw = json.loads(args.config.read_text())
        validate_config(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"valid {SCHEMA}: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
