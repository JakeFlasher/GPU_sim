#!/usr/bin/env python3
"""Report finite/open completion-frontier eligibility from simulator output.

This is a diagnostic projection, not an execution gate.  It never consumes
NCU values and never turns an observation into a simulator upper bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "sm86_completion_frontier_report_v1"
TAIL_KEYS = (
    "reduction_read_modify_write",
    "dirty_writeback",
    "coherence",
    "async_copy_shared_visibility",
)


def _bool_tail_map(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise ValueError("completion frontier unmodeled_tails must be an object")
    result: dict[str, bool] = {}
    for key in TAIL_KEYS:
        raw = value.get(key)
        if type(raw) is not bool:
            raise ValueError(f"completion frontier tail {key} must be boolean")
        result[key] = raw
    return result


def report_frontiers(document: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "sm86_completion_frontier_contract_v1":
        raise ValueError("unsupported completion frontier contract")
    endpoint = contract.get("target_endpoint")
    if not isinstance(endpoint, Mapping):
        raise ValueError("completion frontier contract endpoint is missing")
    launches = document.get("launches")
    if not isinstance(launches, list) or not launches:
        raise ValueError("simulator document has no launches")
    rows: list[dict[str, Any]] = []
    for index, launch in enumerate(launches):
        if not isinstance(launch, Mapping):
            raise ValueError(f"launch {index} is not an object")
        timing = launch.get("timing")
        if not isinstance(timing, Mapping):
            raise ValueError(f"launch {index} has no timing report")
        completion = timing.get("completion_frontier")
        if not isinstance(completion, Mapping):
            raise ValueError(f"launch {index} has no completion frontier")
        status = completion.get("status")
        if status == "complete":
            lower = completion.get("cycles")
            if type(lower) is not int or lower < 0:
                raise ValueError(f"launch {index} complete frontier lacks cycles")
            tails = {key: False for key in TAIL_KEYS}
        elif status == "incomplete_unmodeled_retirement_tails":
            lower = completion.get("lower_bound_cycles")
            if type(lower) is not int or lower < 0:
                raise ValueError(f"launch {index} incomplete frontier lacks lower bound")
            tails = _bool_tail_map(completion.get("unmodeled_tails"))
        else:
            raise ValueError(f"launch {index} has unsupported completion status {status!r}")
        gpc = timing.get("gpc_observation_frontier")
        gpc_lower = gpc.get("lower_bound_cycles") if isinstance(gpc, Mapping) else None
        gpc_point = gpc.get("cycles") if isinstance(gpc, Mapping) else None
        open_dimensions = [key for key, active in tails.items() if active]
        # This field is a point (terminal model time + launch envelope), not
        # an upper bound. A complete model event stream is not an NCU upper
        # either. This report has no endpoint-error or statistical model input.
        open_dimensions.append("ncu_endpoint_upper_not_established")
        finite_upper = False
        rows.append({
            "launch_index": index,
            "kernel_name": launch.get("kernel_name"),
            "completion_status": status,
            "modeled_lower_bound_cycles": lower,
            "gpc_observation_lower_bound_cycles": gpc_lower,
            "gpc_observation_cycles": gpc_point,
            "finite_upper_eligible": finite_upper,
            "open_dimensions": open_dimensions,
            "ncu_target_is_not_used": True,
        })
    return {
        "schema": SCHEMA,
        "contract": "sm86_completion_frontier_contract_v1",
        "target_endpoint": dict(endpoint),
        "launch_count": len(rows),
        "finite_upper_eligible_launches": sum(row["finite_upper_eligible"] for row in rows),
        "open_launches": sum(not row["finite_upper_eligible"] for row in rows),
        "rows": rows,
        "interpretation": "diagnostic_model_frontier_only_ncu_values_not_consumed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator_json", type=Path)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = report_frontiers(
        json.loads(args.simulator_json.read_text()),
        json.loads(args.contract.read_text()),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "open_launches": report["open_launches"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
