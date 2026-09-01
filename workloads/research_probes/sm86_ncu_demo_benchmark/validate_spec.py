#!/usr/bin/env python3
"""Validate the frozen SM86 NCU demo benchmark specification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_SPEC = HERE / "benchmark_spec.json"
DEFAULT_METRICS = HERE / "metric_schema_counts.json"
DEFAULT_PROFILE = HERE / "host_profile.json"
DEFAULT_GATE = HERE / "accuracy_gate_counts_single_sim.json"


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
    schema_version = metrics.get("schema_version")
    require(schema_version in {1, 2}, "metric schema_version must be 1 or 2")
    rows = metrics.get("metrics")
    expected_rows = 10 if schema_version == 1 else 13
    require(
        isinstance(rows, list) and len(rows) == expected_rows,
        f"metric schema must contain {expected_rows} rows",
    )
    ids = [row["id"] for row in rows]
    ncu = [row["ncu_metric"] for row in rows]
    collected = [
        source
        for row in rows
        for source in row.get("ncu_sources", [row["ncu_metric"]])
    ]
    require(len(ids) == len(set(ids)), "metric ids must be unique")
    require(len(ncu) == len(set(ncu)), "NCU metric names must be unique")
    require(collected == list(requested), "benchmark NCU collection order differs from metric schema")
    if schema_version == 1:
        hit_rates = [row for row in rows if row.get("hit_rate_member")]
        require(len(hit_rates) == 3, "exactly three hit-rate metrics are required")
        require(
            all(
                len(row.get("ncu_sources", [])) == 2
                and row.get("ncu_formula")
                == "100 * hit_sectors / (hit_sectors + miss_sectors)"
                for row in hit_rates
            ),
            "hit-rate rows must derive from one NCU hit/miss pair",
        )
        require(
            all("percentage_point" in row["error"] for row in hit_rates),
            "hit-rate rows must require percentage-point error",
        )
    else:
        count_rows = [row for row in rows if row.get("cache_lookup_count_member")]
        require(len(count_rows) == 6, "count-target schema requires six cache count rows")
        require(
            {(row["cache_lookup_scope"], row["cache_lookup_component"]) for row in count_rows}
            == {
                ("l1_global_read", "hits"),
                ("l1_global_read", "misses"),
                ("l1_unified", "hits"),
                ("l1_unified", "misses"),
                ("l2_l1tex_source", "hits"),
                ("l2_l1tex_source", "misses"),
            },
            "cache lookup count scopes/components drift",
        )
        derived = metrics.get("derived_hit_rates")
        require(
            isinstance(derived, list)
            and {row["id"] for row in derived}
            == {"l1_read_hit_rate", "l1_unified_hit_rate", "l2_unified_hit_rate"},
            "three derived hit-rate reports are required",
        )
        by_id = {row["id"] for row in count_rows}
        require(
            all(
                row["hit_metric_id"] in by_id and row["miss_metric_id"] in by_id
                for row in derived
            ),
            "derived hit rates must reference cache count targets",
        )
    ratios = metrics.get("timing_ratios")
    require(isinstance(ratios, list) and len(ratios) == 2, "two timing ratios are required")
    ratio_ids = {row["id"] for row in ratios}
    require(
        ratio_ids
        == {"gpu_execution_over_simulator_wall", "ncu_profile_wall_over_simulator_wall"},
        "timing ratio ids drift",
    )


def validate_accuracy_gate(gate: Mapping[str, Any]) -> None:
    schema_version = gate.get("schema_version")
    require(
        schema_version in {1, 2, 3},
        "accuracy gate schema_version must be 1, 2, or 3",
    )
    require(gate.get("required_workloads") == 8, "accuracy gate must require 8 workloads")
    require(gate.get("required_ncu_repeats") == 7, "accuracy gate must require 7 NCU repeats")
    expected_simulator_repeats = 1 if schema_version == 3 else 7
    require(
        gate.get("required_simulator_repeats") == expected_simulator_repeats,
        f"accuracy gate must require {expected_simulator_repeats} simulator repeat(s)",
    )
    expected = {
        "terminal_cycle_mape_pct": (20.0, 8),
        "runtime_equivalent_mape_pct": (20.0, 8),
        "issue_cycle_mape_pct": (20.0, 8),
        "instruction_throughput_mape_pct": (20.0, 8),
        "counter_traffic_mape_pct": (20.0, 32),
    }
    if schema_version == 1:
        expected.update(
            {
                "l1_unified_hit_rate_mape_pct": (20.0, 8),
                "l2_unified_hit_rate_mape_pct": (20.0, 8),
            }
        )
    else:
        expected["cache_lookup_count_normalized_error_pct"] = (20.0, 48)
    checks = gate.get("checks")
    require(isinstance(checks, Mapping) and set(checks) == set(expected), "accuracy gate checks drift")
    for name, (maximum, coverage) in expected.items():
        require(checks[name].get("maximum") == maximum, f"{name}: maximum drift")
        require(
            checks[name].get("required_coverage") == coverage,
            f"{name}: coverage drift",
        )
    if schema_version == 1:
        for name in ("l1_unified_hit_rate_mape_pct", "l2_unified_hit_rate_mape_pct"):
            require(
                checks[name].get("maximum_mean_absolute_percentage_point_error") == 10.0
                and checks[name].get("required_percentage_point_coverage") == 8,
                f"{name}: percentage-point gate drift",
            )
    require(gate.get("manual_approval_required") is True, "manual approval must remain required")


def validate_timing_crosswalk(crosswalk: Mapping[str, Any]) -> None:
    require(crosswalk.get("schema_version") == 1, "timing crosswalk schema_version must be 1")
    require(crosswalk.get("architecture") == "sm_86", "timing crosswalk architecture drift")
    require(crosswalk.get("sku") == "rtx_3090_ti", "timing crosswalk SKU drift")
    require(
        crosswalk.get("observation_scope") == "single_kernel_ncu_application_replay",
        "timing crosswalk observation scope drift",
    )
    active = crosswalk.get("sm_active_projection")
    elapsed = crosswalk.get("gpc_elapsed_projection")
    require(isinstance(active, Mapping), "timing crosswalk SM-active projection is missing")
    require(isinstance(elapsed, Mapping), "timing crosswalk GPC projection is missing")
    cells = crosswalk.get("calibration", {}).get("cells")
    require(isinstance(cells, list) and len(cells) == 2, "timing crosswalk requires two controls")
    require(
        {cell.get("id") for cell in cells}
        == {"b84_control_i0_c1", "b84_control_i256_c1"},
        "timing crosswalk control identities drift",
    )
    sm_residuals: list[float] = []
    gpc_residuals: list[float] = []
    for cell in cells:
        ncu_sm = float(cell["ncu_sm_active_cycles_median"])
        ncu_gpc = float(cell["ncu_gpc_elapsed_cycles_median"])
        sim_issue = float(cell["simulator_issue_cycles"])
        sm_residual = float(cell["sm_active_minus_issue_cycles"])
        gpc_residual = float(cell["gpc_elapsed_minus_sm_active_cycles"])
        require(ncu_sm - sim_issue == sm_residual, f"{cell['id']}: SM-active residual drift")
        require(ncu_gpc - ncu_sm == gpc_residual, f"{cell['id']}: GPC residual drift")
        sm_residuals.append(sm_residual)
        gpc_residuals.append(gpc_residual)
    require(
        float(active.get("fixed_envelope_cycles")) == statistics.median(sm_residuals),
        "timing crosswalk SM-active envelope is not the control median",
    )
    require(
        float(elapsed.get("sm_active_envelope_cycles"))
        == float(active.get("fixed_envelope_cycles")),
        "timing crosswalk SM-active envelopes differ",
    )
    require(
        float(elapsed.get("profiler_envelope_cycles"))
        == statistics.median(gpc_residuals),
        "timing crosswalk profiler envelope is not the control median",
    )
    boundary = crosswalk.get("scientific_boundary")
    require(isinstance(boundary, Mapping), "timing crosswalk scientific boundary is missing")
    for field in (
        "scored_workload_ncu_values_used",
        "passed_to_simulator_command",
        "issue_and_terminal_frontiers_collapsed",
        "multi_launch_ranges_supported",
    ):
        require(boundary.get(field) is False, f"timing crosswalk boundary {field} must be false")
    for field in ("raw_frontiers_preserved", "incomplete_retirement_tails_remain_typed"):
        require(boundary.get(field) is True, f"timing crosswalk boundary {field} must be true")


def validate_spec(spec: Mapping[str, Any], metrics: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    require(spec.get("schema_version") == 1, "benchmark schema_version must be 1")
    require(spec.get("architecture") == "sm_86", "benchmark architecture must be sm_86")
    require(spec.get("correctness") == {"timeout_seconds": 180}, "correctness contract drift")
    require(
        spec.get("static_control")
        == {"toolkit_bindir": "/usr/local/cuda-13.2/bin", "timeout_seconds": 300},
        "static-control contract drift",
    )
    workloads = spec.get("workloads")
    matrix_role = spec.get("matrix_role", "benchmark")
    if matrix_role == "independent_holdout":
        require(
            isinstance(workloads, list) and 1 <= len(workloads) <= 3,
            "independent holdout workload count must be 1..3",
        )
    else:
        require(
            isinstance(workloads, list) and 6 <= len(workloads) <= 8,
            "workload count must be 6..8",
        )
    ids = [row["id"] for row in workloads]
    require(len(ids) == len(set(ids)), "workload ids must be unique")
    if matrix_role != "independent_holdout":
        require(
            sum(row["stage"] == "smoke" for row in workloads) >= 4,
            "at least four smoke rows required",
        )
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
        overrides = row.get("timing_overrides", [])
        require(
            row["ncu_args"]
            == ["--timing_shape", "--warmup", "1", "--reps", "1", *overrides],
            f"{row['id']}: NCU args",
        )
        require(
            row["trace_args"]
            == ["--timing_shape", "--warmup", "0", "--reps", "1", *overrides],
            f"{row['id']}: trace args",
        )
        require(not any(name in row["display_name"] for name in forbidden_names), f"{row['id']}: misleading display name")
        require("kernel" in row["functional_analogue"], f"{row['id']}: analogue must say kernel")
    validate_metrics(metrics, spec["ncu"]["metrics"])
    require(
        spec.get("accuracy_gate")
        in {
            "workloads/research_probes/sm86_ncu_demo_benchmark/accuracy_gate.json",
            "workloads/research_probes/sm86_ncu_demo_benchmark/accuracy_gate_counts.json",
            "workloads/research_probes/sm86_ncu_demo_benchmark/accuracy_gate_counts_single_sim.json",
        },
        "accuracy gate path drift",
    )
    timing_crosswalk = spec.get("timing_crosswalk")
    if timing_crosswalk is not None:
        require(
            timing_crosswalk
            == "workloads/research_probes/sm86_ncu_demo_benchmark/timing_crosswalk.json",
            "timing crosswalk path drift",
        )
        validate_timing_crosswalk(load_object(REPO_ROOT / str(timing_crosswalk)))
    require(
        spec["ncu"].get("repeats_smoke") == 3
        and spec["ncu"].get("repeats_final") == 7,
        "NCU repeat contract drift",
    )
    expected_simulator_repeats = (
        (1, 1)
        if spec.get("accuracy_gate")
        == "workloads/research_probes/sm86_ncu_demo_benchmark/accuracy_gate_counts_single_sim.json"
        else (3, 7)
    )
    require(
        (
            spec["simulator"].get("repeats_smoke"),
            spec["simulator"].get("repeats_final"),
        )
        == expected_simulator_repeats
        and "twins" not in spec["simulator"],
        "simulator repeat contract drift",
    )
    validate_profile(profile, live_host=False)
    require(spec["simulator"]["multi_launch_policy"].startswith("not_admitted"), "v1 multi-launch policy must fail closed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    result.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    result.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    result.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    result.add_argument("--validate-live-host", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        spec = load_object(args.spec)
        metrics = load_object(args.metrics)
        profile = load_object(args.profile)
        gate = load_object(args.gate)
        validate_spec(spec, metrics, profile)
        validate_accuracy_gate(gate)
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
