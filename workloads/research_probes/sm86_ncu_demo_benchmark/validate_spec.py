#!/usr/bin/env python3
"""Validate the frozen SM86 NCU demo benchmark specification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_SPEC = HERE / "benchmark_spec.json"
DEFAULT_METRICS = HERE / "metric_schema.json"
DEFAULT_PROFILE = HERE / "host_profile.json"


class ValidationError(RuntimeError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: top-level JSON value must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_artifact(record: Mapping[str, Any], context: str) -> None:
    path = REPO_ROOT / str(record["path"])
    require(path.is_file(), f"{context}: missing file {path}")
    require(sha256(path) == record["sha256"], f"{context}: SHA-256 drift")


def validate_profile(profile: Mapping[str, Any], live_host: bool) -> dict[str, Any] | None:
    require(profile.get("schema_version") == 1, "host profile schema_version must be 1")
    expected = profile["expected"]
    require(expected["sm_count"] == 84, "host profile must bind 84 SMs")
    require(expected["l2_capacity_bytes"] == 6291456, "host profile L2 capacity drift")
    for name, artifact in profile["artifacts"].items():
        validate_artifact(artifact, f"profile artifact {name}")
    if not live_host:
        return None
    command = [
        "nvidia-smi",
        "-i",
        str(profile["device_index"]),
        "--query-gpu=index,name,uuid,compute_cap,driver_version,persistence_mode,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(completed.returncode == 0, f"nvidia-smi failed: {completed.stderr.strip()}")
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    require(len(fields) == 10, "nvidia-smi returned an unexpected field count")
    observed = {
        "index": int(fields[0]),
        "name": fields[1],
        "uuid": fields[2],
        "compute_capability": fields[3],
        "driver_version": fields[4],
        "persistence_mode": fields[5],
        "pstate": fields[6],
        "graphics_clock_mhz": int(fields[7]),
        "sm_clock_mhz": int(fields[8]),
        "memory_clock_mhz": int(fields[9]),
    }
    for field in ("name", "uuid", "compute_capability", "driver_version"):
        require(observed[field] == expected[field], f"live host {field} mismatch")
    return observed


def validate_metrics(metrics: Mapping[str, Any], requested: Sequence[str]) -> None:
    require(metrics.get("schema_version") == 1, "metric schema_version must be 1")
    rows = metrics.get("metrics")
    require(isinstance(rows, list) and len(rows) == 10, "metric schema must contain 10 rows")
    ids = [row["id"] for row in rows]
    ncu = [row["ncu_metric"] for row in rows]
    require(len(ids) == len(set(ids)), "metric ids must be unique")
    require(len(ncu) == len(set(ncu)), "NCU metric names must be unique")
    require(ncu == list(requested), "benchmark NCU metric order differs from metric schema")
    hit_rates = [row for row in rows if row.get("hit_rate_member")]
    require(len(hit_rates) == 3, "exactly three hit-rate metrics are required")
    require(
        all("percentage_point" in row["error"] for row in hit_rates),
        "hit-rate rows must require percentage-point error",
    )
    ratios = metrics.get("timing_ratios")
    require(isinstance(ratios, list) and len(ratios) == 2, "two timing ratios are required")
    ratio_ids = {row["id"] for row in ratios}
    require(
        ratio_ids
        == {"gpu_execution_over_simulator_wall", "ncu_profile_wall_over_simulator_wall"},
        "timing ratio ids drift",
    )


def validate_spec(spec: Mapping[str, Any], metrics: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    require(spec.get("schema_version") == 1, "benchmark schema_version must be 1")
    require(spec.get("architecture") == "sm_86", "benchmark architecture must be sm_86")
    workloads = spec.get("workloads")
    require(isinstance(workloads, list) and 6 <= len(workloads) <= 8, "workload count must be 6..8")
    ids = [row["id"] for row in workloads]
    require(len(ids) == len(set(ids)), "workload ids must be unique")
    require(sum(row["stage"] == "smoke" for row in workloads) >= 4, "at least four smoke rows required")
    forbidden_names = ("GPT-2", "BERT", "BFS", "HotSpot", "Pathfinder", "SRAD", "Streamcluster")
    for row in workloads:
        binary = REPO_ROOT / row["binary"]
        source = REPO_ROOT / row["source"]
        require(binary.is_file() and os.access(binary, os.X_OK), f"{row['id']}: binary missing/not executable")
        require(source.is_file(), f"{row['id']}: source missing")
        require(sha256(binary) == row["binary_sha256"], f"{row['id']}: binary SHA-256 drift")
        require(binary.stat().st_size == row["binary_size_bytes"], f"{row['id']}: binary size drift")
        require(row["target_launch_count"] == 1, f"{row['id']}: v1 admits one target launch")
        require(row["correctness_args"][0] == "--correctness_shape", f"{row['id']}: correctness args")
        require(row["ncu_args"] == ["--timing_shape", "--warmup", "1", "--reps", "1"], f"{row['id']}: NCU args")
        require(row["trace_args"] == ["--timing_shape", "--warmup", "0", "--reps", "1"], f"{row['id']}: trace args")
        require(not any(name in row["display_name"] for name in forbidden_names), f"{row['id']}: misleading display name")
        require("kernel" in row["functional_analogue"], f"{row['id']}: analogue must say kernel")
    validate_metrics(metrics, spec["ncu"]["metrics"])
    validate_profile(profile, live_host=False)
    require(spec["simulator"]["multi_launch_policy"].startswith("not_admitted"), "v1 multi-launch policy must fail closed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    result.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    result.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    result.add_argument("--validate-live-host", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        spec = load_object(args.spec)
        metrics = load_object(args.metrics)
        profile = load_object(args.profile)
        validate_spec(spec, metrics, profile)
        observed = validate_profile(profile, live_host=args.validate_live_host)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "benchmark_id": spec["benchmark_id"],
                    "workloads": len(spec["workloads"]),
                    "smoke_workloads": sum(row["stage"] == "smoke" for row in spec["workloads"]),
                    "metrics": len(metrics["metrics"]),
                    "live_host": observed,
                },
                sort_keys=True,
            )
        )
    except (OSError, KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as error:
        print(f"benchmark spec validation: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
