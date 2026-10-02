#!/usr/bin/env python3
"""Convert an NCU CSV metric into a post-prediction observation document.

This utility has no prediction input and performs no model selection. It is a
small boundary adapter so NCU values can only enter through the explicit
post-prediction join evaluator.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

from prediction_contract import TARGET_METRIC, validate_observation_document, validate_target_contract


def parse_ncu_csv(
    path: Path,
    metric: str,
    workload: str,
    source: str = "ncu_csv",
    *,
    endpoint: str | None = None,
    clock_domain: str | None = None,
    cache_policy: str | None = None,
    launch_identity: str | None = None,
    repetitions: int = 1,
    gpu_uuid: str | None = None,
    compute_capability: str | None = None,
    sm_clock_mhz: int | None = None,
    device_index: int | None = None,
) -> dict[str, Any]:
    if metric != TARGET_METRIC:
        raise ValueError(f"NCU observation metric must be {TARGET_METRIC!r}")
    target_contract = validate_target_contract(
        {
            "metric": metric,
            "endpoint": endpoint,
            "clock_domain": clock_domain,
            "cache_policy": cache_policy,
            "launch_identity": launch_identity,
            "repetitions": repetitions,
            "hardware": {
                "gpu_uuid": gpu_uuid,
                "compute_capability": compute_capability,
                "sm_clock_mhz": sm_clock_mhz,
                "device_index": device_index,
            },
        },
        "NCU observation",
        require_hardware=True,
    )
    lines = path.read_text().splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.startswith('"ID"'))
    except StopIteration as error:
        raise ValueError("NCU CSV header row is missing") from error
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header_index:]))))
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        try:
            parsed = float(value)
        except ValueError:
            # The first row after the header carries units.
            continue
        values.append(parsed)
    if not values:
        raise ValueError(f"NCU metric {metric!r} has no numeric data rows")
    if len(values) != repetitions:
        raise ValueError(
            f"NCU metric {metric!r} has {len(values)} numeric rows, "
            f"but target contract declares {repetitions} repetitions"
        )
    return validate_observation_document({
        "schema": "ncu_observation_v2",
        "identity_status": "caller_labeled_unverified",
        "target_contract": target_contract,
        "source": source,
        "rows": [
            {
                "workload": workload,
                "value": value,
                "metric": metric,
                "endpoint": target_contract["endpoint"],
                "clock_domain": target_contract["clock_domain"],
                "hardware": target_contract["hardware"],
                "source": source,
                "identity_status": "caller_labeled_unverified",
            }
            for value in values
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--source", default="ncu_csv")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--clock-domain", required=True)
    parser.add_argument("--cache-policy", required=True)
    parser.add_argument("--launch-identity", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--compute-capability", required=True)
    parser.add_argument("--sm-clock-mhz", type=int, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(parse_ncu_csv(
        args.csv,
        args.metric,
        args.workload,
        args.source,
        endpoint=args.endpoint,
        clock_domain=args.clock_domain,
        cache_policy=args.cache_policy,
        launch_identity=args.launch_identity,
        repetitions=args.repetitions,
        gpu_uuid=args.gpu_uuid,
        compute_capability=args.compute_capability,
        sm_clock_mhz=args.sm_clock_mhz,
        device_index=args.device_index,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
