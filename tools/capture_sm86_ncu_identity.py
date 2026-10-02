#!/usr/bin/env python3
"""Capture and validate metadata for an accuracy-first SM86 NCU campaign.

This command performs metadata queries only. It never launches a workload or
profiles a kernel. A mismatch/unavailable result is labeled and ineligible for
accuracy scoring; it is not an execution or approval gate.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from validate_sm86_ncu_accuracy_protocol import validate_protocol


Runner = Callable[[Sequence[str]], tuple[int, str, str]]


def _run(command: Sequence[str]) -> tuple[int, str, str]:
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    return process.returncode, process.stdout, process.stderr


def _version(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _clock(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else None


def _query_smi(runner: Runner) -> tuple[dict[str, Any] | None, str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version,index,clocks.current.graphics,clocks.current.memory,pstate",
        "--format=csv,noheader,nounits",
    ]
    code, stdout, stderr = runner(command)
    if code != 0:
        return None, stderr.strip() or "nvidia-smi query failed"
    rows = list(csv.reader(io.StringIO(stdout)))
    if len(rows) != 1 or len(rows[0]) != 8:
        return None, "nvidia-smi returned an unexpected GPU row count/shape"
    name, uuid, cc, driver, index, graphics, memory, pstate = [value.strip() for value in rows[0]]
    try:
        device_index = int(index)
    except ValueError:
        return None, "nvidia-smi device index is not an integer"
    return {
        "gpu_name": name,
        "gpu_uuid": uuid,
        "compute_capability": cc,
        "driver_version": driver,
        "device_index": device_index,
        "graphics_clock_mhz": _clock(graphics),
        "memory_clock_mhz": _clock(memory),
        "power_state": pstate,
    }, None


def capture_identity(
    protocol: dict[str, Any],
    *,
    simulator_binary: Path | None = None,
    trace: Path | None = None,
    control: Path | None = None,
    arch_card: Path | None = None,
    pchase_card: Path | None = None,
    nvbit_tracer_build: str | None = None,
    simulator_build_profile: str | None = None,
    trace_schema_version: str | None = None,
    runner: Runner = _run,
) -> dict[str, Any]:
    validate_protocol(protocol)
    measurement = protocol["measurement_contract"]
    mismatches: list[str] = []
    hardware, smi_error = _query_smi(runner)
    if hardware is None:
        hardware = {field: None for field in (
            "gpu_name", "gpu_uuid", "compute_capability", "driver_version",
            "device_index", "graphics_clock_mhz", "memory_clock_mhz", "power_state",
        )}
        mismatches.append(f"hardware_identity_unavailable:{smi_error}")
    else:
        expected_name = protocol["identity_requirements"]["expected_gpu_name"]
        if hardware["gpu_name"] != expected_name:
            mismatches.append(f"gpu_name:{hardware['gpu_name']}")
        if hardware["compute_capability"] != "8.6":
            mismatches.append(f"compute_capability:{hardware['compute_capability']}")
        if hardware["device_index"] != measurement["device_index"]:
            mismatches.append(f"device_index:{hardware['device_index']}")

    metric_base = measurement["target_metric"].split(".", 1)[0]
    metric_query_command = ["ncu", "--query-metrics", "--chip", measurement["ncu_chip"]]
    code, ncu_stdout, ncu_stderr = runner(metric_query_command)
    metric_available = code == 0 and any(
        line.strip().split()[0:1] == [metric_base]
        for line in ncu_stdout.splitlines()
    )
    if not metric_available:
        mismatches.append("target_metric_unavailable_or_query_failed")

    def query_version(command: Sequence[str], pattern: str, label: str) -> str | None:
        code, stdout, stderr = runner(command)
        value = _version(stdout + "\n" + stderr, pattern) if code == 0 else None
        if value is None:
            mismatches.append(f"{label}_unavailable")
        return value

    cuda_version = query_version(["nvcc", "--version"], r"release\s+([0-9.]+)", "cuda_version")
    ncu_version = query_version(["ncu", "--version"], r"Version\s+([^\s(]+)", "ncu_version")
    rustc_version = query_version(["rustc", "--version"], r"(rustc\s+[^\n]+)", "rustc_version")
    commit_code, commit_out, _ = runner(["git", "rev-parse", "HEAD"])
    commit = commit_out.strip() if commit_code == 0 else None
    if commit is None:
        mismatches.append("simulator_git_commit_unavailable")
    status_code, status_out, _ = runner(["git", "status", "--porcelain", "--untracked-files=no"])
    all_status_code, all_status_out, _ = runner(["git", "status", "--porcelain", "--untracked-files=all"])
    untracked_paths = []
    unexpected_untracked = []
    if all_status_code == 0:
        for line in all_status_out.splitlines():
            if line.startswith("?? "):
                path = line[3:]
                untracked_paths.append(path)
                if not path.startswith("research_audit_20261001/"):
                    unexpected_untracked.append(path)
    working_tree_dirty = status_code != 0 or bool(status_out.strip()) or bool(unexpected_untracked)
    if working_tree_dirty:
        mismatches.append("working_tree_dirty")

    paths = {
        "simulator_binary": str(simulator_binary) if simulator_binary else None,
        "trace": str(trace) if trace else None,
        "control": str(control) if control else None,
        "arch_card": str(arch_card) if arch_card else None,
        "pchase_card": str(pchase_card) if pchase_card else None,
        "nvbit_tracer_build": nvbit_tracer_build,
    }
    for field, value in paths.items():
        if field != "nvbit_tracer_build" and (value is None or not Path(value).exists()):
            mismatches.append(f"input_missing:{field}")
        if field == "nvbit_tracer_build" and not value:
            mismatches.append("nvbit_tracer_build_unavailable")
    if not simulator_build_profile:
        mismatches.append("simulator_build_profile_unavailable")
    if not trace_schema_version:
        mismatches.append("trace_schema_version_unavailable")

    status = "independently_verified" if not mismatches else "mismatch"
    return {
        "schema": "sm86_ncu_identity_v1",
        "status": status,
        "eligible_for_accuracy": status == "independently_verified",
        "protocol": "sm86_ncu_accuracy_protocol_v1",
        "measurement_contract": measurement,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware": hardware,
        "software": {"cuda_version": cuda_version, "ncu_version": ncu_version, "rustc_version": rustc_version},
        "build": {
            "simulator_git_commit": commit,
            "simulator_build_profile": simulator_build_profile,
            "simulator_binary": paths["simulator_binary"],
            "trace_schema_version": trace_schema_version,
            "nvbit_tracer_build": nvbit_tracer_build,
            "working_tree_dirty": working_tree_dirty,
            "untracked_paths": untracked_paths,
            "ignored_untracked_paths": [path for path in untracked_paths if path.startswith("research_audit_20261001/")],
        },
        "inputs": {key: value for key, value in paths.items() if key != "nvbit_tracer_build"},
        "metric_availability": {
            "chip": measurement["ncu_chip"],
            "target_metric": measurement["target_metric"],
            "available": metric_available,
            "query_command": metric_query_command,
        },
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--simulator-binary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--arch-card", type=Path, required=True)
    parser.add_argument("--pchase-card", type=Path, required=True)
    parser.add_argument("--nvbit-tracer-build", required=True)
    parser.add_argument("--simulator-build-profile", required=True)
    parser.add_argument("--trace-schema-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture_identity(
        json.loads(args.protocol.read_text()),
        simulator_binary=args.simulator_binary,
        trace=args.trace,
        control=args.control,
        arch_card=args.arch_card,
        pchase_card=args.pchase_card,
        nvbit_tracer_build=args.nvbit_tracer_build,
        simulator_build_profile=args.simulator_build_profile,
        trace_schema_version=args.trace_schema_version,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "eligible_for_accuracy": result["eligible_for_accuracy"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
