#!/usr/bin/env python3
"""Read-only readiness probe for a locked-clock SM86 campaign.

The probe never changes clocks. It records host identity, current clocks, max
clocks, and whether non-interactive privilege is available for a future
operator-approved lock command. ``ready`` therefore means only that the
privilege prerequisite is present; it is not a locked-clock measurement.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from typing import Any


def _run(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return completed.returncode, completed.stdout, completed.stderr


def _sample_summary(path: Path, expected_sm: str | None, expected_mem: str | None) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"status": "sample_file_unreadable", "detail": str(exc)}
    rows = []

    def _number(value: str, *, integer: bool = False) -> float | int:
        parsed = float(value.strip())
        return int(parsed) if integer else parsed

    # Accept both the original five-column sampler format
    # (timestamp,sm_mhz,mem_mhz,power_w,temp_c[,pstate]) and the native
    # nvidia-smi/NVML format used by the campaign helper
    # (sample,sm_mhz,mem_mhz,pstate,throttle,temp_c,power_mw).  The old
    # positional-only parser silently discarded the latter because ``P0`` was
    # parsed as a power value, turning a real active-window failure into an
    # apparent zero-sample probe.
    header = [field.strip().lower() for field in lines[0].split(",")] if lines else []
    has_named_clock_columns = "sm_mhz" in header and "mem_mhz" in header
    for line in lines[1:] if has_named_clock_columns else lines[:100_000]:
        fields = [field.strip() for field in line.split(",")]
        try:
            if has_named_clock_columns:
                values = dict(zip(header, fields))
                power_key = "power_mw" if "power_mw" in values else "power_w"
                power = _number(values[power_key])
                if power_key == "power_mw":
                    power /= 1000.0
                row = {
                    "sm_mhz": _number(values["sm_mhz"], integer=True),
                    "mem_mhz": _number(values["mem_mhz"], integer=True),
                    "power_w": power,
                    "temp_c": _number(values["temp_c"], integer=True),
                }
                if "pstate" in values:
                    row["pstate"] = values["pstate"]
                if "throttle" in values:
                    row["throttle"] = values["throttle"]
            else:
                if len(fields) < 5:
                    continue
                row = {
                    "sm_mhz": _number(fields[1], integer=True),
                    "mem_mhz": _number(fields[2], integer=True),
                    "power_w": _number(fields[3]),
                    "temp_c": _number(fields[4], integer=True),
                }
                if len(fields) >= 6:
                    row["pstate"] = fields[5]
            rows.append(row)
        except (KeyError, ValueError):
            continue
    sm_values = sorted({row["sm_mhz"] for row in rows})
    mem_values = sorted({row["mem_mhz"] for row in rows})
    stable = len(rows) > 0 and len(sm_values) == 1 and len(mem_values) == 1
    target_match = (
        stable
        and expected_sm is not None
        and expected_mem is not None
        and sm_values[0] == int(expected_sm)
        and mem_values[0] == int(expected_mem)
    )
    return {
        "path": str(path),
        "sample_count": len(rows),
        "sm_values_mhz": sm_values,
        "memory_values_mhz": mem_values,
        "stable": stable,
        "target_match": target_match,
        "power_range_w": [
            min((row["power_w"] for row in rows), default=None),
            max((row["power_w"] for row in rows), default=None),
        ],
        "temperature_range_c": [
            min((row["temp_c"] for row in rows), default=None),
            max((row["temp_c"] for row in rows), default=None),
        ],
        "pstate_values": sorted({row["pstate"] for row in rows if row.get("pstate")}),
        "throttle_values": sorted({row["throttle"] for row in rows if row.get("throttle")}),
    }


def probe(
    expected_sm: str | None = None,
    expected_mem: str | None = None,
    samples_path: Path | None = None,
) -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,compute_cap,driver_version,pci.bus_id,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem",
        "--format=csv,noheader,nounits",
    ]
    rc, stdout, stderr = _run(query)
    if rc != 0:
        return {
            "schema": "sm86_clock_lock_readiness_v1",
            "status": "refused",
            "reason": "nvidia_smi_query_failed",
            "returncode": rc,
            "stderr_tail": stderr[-2000:],
        }
    rows = list(csv.reader(io.StringIO(stdout)))
    if not rows or len(rows[0]) != 10:
        return {
            "schema": "sm86_clock_lock_readiness_v1",
            "status": "refused",
            "reason": "nvidia_smi_identity_shape_invalid",
        }
    identity = dict(
        zip(
            [
                "index",
                "name",
                "uuid",
                "compute_capability",
                "driver_version",
                "pci_bus_id",
                "current_sm_mhz",
                "current_mem_mhz",
                "max_sm_mhz",
                "max_mem_mhz",
            ],
            (value.strip() for value in rows[0]),
        )
    )
    sudo_rc, _sudo_out, sudo_err = _run(["sudo", "-n", "true"])
    targets_observed = (
        expected_sm is not None
        and expected_mem is not None
        and identity["current_sm_mhz"] == expected_sm
        and identity["current_mem_mhz"] == expected_mem
    )
    sample_summary = (
        _sample_summary(samples_path, expected_sm, expected_mem)
        if samples_path is not None
        else None
    )
    status = "ready" if sudo_rc == 0 else ("target_clocks_observed" if targets_observed else "privilege_required")
    if sample_summary is not None:
        if sample_summary.get("target_match"):
            status = "target_clocks_stable_observed"
        elif sample_summary.get("sample_count", 0):
            status = "target_clocks_not_stable"
    return {
        "schema": "sm86_clock_lock_readiness_v1",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "clock_control": {
            "lock_command_run": False,
            "expected_sm_mhz": expected_sm,
            "expected_mem_mhz": expected_mem,
            "target_clocks_observed": targets_observed,
            "samples": sample_summary,
            "noninteractive_privilege": sudo_rc == 0,
            "sudo_stderr_tail": sudo_err[-1000:],
            "required_future_verification": [
                "application_clock_lock_acknowledged",
                "stable_sm_clock_samples_under_workload",
                "stable_memory_clock_samples_under_workload",
                "trace_card_control_output_content_identity",
            ],
        },
        "scientific_claim": "readiness_only_no_locked_clock_claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--expected-sm-mhz")
    parser.add_argument("--expected-mem-mhz")
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(
        probe(args.expected_sm_mhz, args.expected_mem_mhz, args.samples), indent=2, sort_keys=True
    ) + "\n"
    if args.output:
        with open(args.output, "x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
