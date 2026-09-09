#!/usr/bin/env python3
"""Validate the frozen SM86 NCU demo benchmark specification."""

from __future__ import annotations

import argparse
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
# ``nvidia-smi`` normally returns in milliseconds, but a wedged driver can
# block indefinitely.  Live campaign pre/post snapshots are part of the
# bounded evidence path, so keep the probe itself bounded as well.
NVIDIA_SMI_TIMEOUT_SECONDS = 30
EXPECTED_PERSISTENCE_MODE = "Enabled"
EXPECTED_PSTATES = frozenset({"P0", "P2"})


class ValidationError(RuntimeError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: top-level JSON value must be an object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_artifact(record: Mapping[str, Any], context: str) -> None:
    path = REPO_ROOT / str(record["path"])
    require(path.is_file(), f"{context}: missing file {path}")


def validate_profile(
    profile: Mapping[str, Any],
    live_host: bool,
    *,
    require_reference_clocks: bool = False,
) -> dict[str, Any] | None:
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
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ValidationError(
            "nvidia-smi timed out after "
            f"{NVIDIA_SMI_TIMEOUT_SECONDS} seconds"
        ) from error
    require(completed.returncode == 0, f"nvidia-smi failed: {completed.stderr.strip()}")
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    require(len(fields) == 10, "nvidia-smi returned an unexpected field count")
    try:
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
    except ValueError as error:
        raise ValidationError("nvidia-smi returned non-numeric identity/clock fields") from error
    require(
        observed["index"] == profile["device_index"],
        "live host device index mismatch",
    )
    for field in ("name", "uuid", "compute_capability", "driver_version"):
        require(observed[field] == expected[field], f"live host {field} mismatch")
    if require_reference_clocks:
        require(
            observed["persistence_mode"] == EXPECTED_PERSISTENCE_MODE,
            "live host persistence mode must remain enabled for the reference-clock contract",
        )
        require(
            observed["pstate"] in EXPECTED_PSTATES,
            "live host pstate is outside the allowed locked idle-state set",
        )
        clock_policy = profile.get("clock_policy")
        require(
            isinstance(clock_policy, Mapping),
            "clock policy is required when reference clocks are enforced",
        )
        expected_sm = clock_policy.get("expected_reference_sm_clock_mhz")
        expected_memory = clock_policy.get("expected_reference_memory_clock_mhz")
        require(
            isinstance(expected_sm, int)
            and not isinstance(expected_sm, bool)
            and isinstance(expected_memory, int)
            and not isinstance(expected_memory, bool),
            "reference clock values must be integer MHz values",
        )
        require(
            observed["graphics_clock_mhz"] == expected_sm
            and observed["sm_clock_mhz"] == expected_sm
            and observed["memory_clock_mhz"] == expected_memory,
            "live host clocks differ from the configured reference clocks: "
            f"observed graphics/sm/memory="
            f"{observed['graphics_clock_mhz']}/{observed['sm_clock_mhz']}/"
            f"{observed['memory_clock_mhz']} MHz, expected "
            f"{expected_sm}/{expected_sm}/{expected_memory} MHz",
        )
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
        schema_version in {1, 2, 3, 4, 5, 6},
        "accuracy gate schema_version must be 1, 2, 3, 4, 5, or 6",
    )
    required_workloads = gate.get("required_workloads")
    if schema_version in {4, 6}:
        require(
            isinstance(required_workloads, int) and 6 <= required_workloads <= 32,
            "large-workload accuracy gate must require 6..32 workloads",
        )
    else:
        require(required_workloads == 8, "accuracy gate must require 8 workloads")
    require(gate.get("required_ncu_repeats") == 7, "accuracy gate must require 7 NCU repeats")
    expected_simulator_repeats = 1 if schema_version in {3, 4, 5, 6} else 7
    require(
        gate.get("required_simulator_repeats") == expected_simulator_repeats,
        f"accuracy gate must require {expected_simulator_repeats} simulator repeat(s)",
    )
    aggregate_maximum = 10.0 if schema_version in {5, 6} else 20.0
    expected = {
        "terminal_cycle_mape_pct": (aggregate_maximum, required_workloads),
        "runtime_equivalent_mape_pct": (aggregate_maximum, required_workloads),
        "issue_cycle_mape_pct": (aggregate_maximum, required_workloads),
        "instruction_throughput_mape_pct": (aggregate_maximum, required_workloads),
        "counter_traffic_mape_pct": (aggregate_maximum, 4 * required_workloads),
    }
    if schema_version == 1:
        expected.update(
            {
                "l1_unified_hit_rate_mape_pct": (20.0, 8),
                "l2_unified_hit_rate_mape_pct": (20.0, 8),
            }
        )
    else:
        expected["cache_lookup_count_normalized_error_pct"] = (
            aggregate_maximum,
            6 * required_workloads,
        )
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
    if schema_version in {4, 6}:
        require(
            gate.get("minimum_hardware_cycles_per_workload") == 8000,
            "large-workload hardware cycle floor must be 8000",
        )
        for name in expected:
            require(
                checks[name].get("maximum_per_workload") == 20.0,
                f"{name}: per-workload maximum drift",
            )
        policy = gate.get("scientific_policy")
        require(isinstance(policy, Mapping), "large-workload scientific policy missing")
        for field in (
            "scored_ncu_values_as_simulator_inputs",
            "per_workload_parameters",
            "aggregate_only_gate",
        ):
            require(policy.get(field) is False, f"scientific policy {field} must be false")
        require(
            policy.get("previous_demo_non_regression_required") is True,
            "previous demo non-regression must remain required",
        )
    if schema_version == 5:
        policy = gate.get("scientific_policy")
        require(isinstance(policy, Mapping), "small 10% scientific policy missing")
        require(
            policy.get("scored_ncu_values_as_simulator_inputs") is False
            and policy.get("per_workload_parameters") is False,
            "small 10% gate must prohibit scored inputs and per-workload parameters",
        )
        require(
            policy.get("metric_families_gated_independently") is True
            and policy.get("per_workload_errors_reported") is True
            and policy.get("raw_frontier_non_regression_required") is True,
            "small 10% gate must retain family, row, and raw-frontier evidence",
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
    projection_mode = active.get("projection_mode", "fixed_observation_residual")
    require(
        projection_mode in {"fixed_observation_residual", "causal_raw_frontier"},
        "timing crosswalk SM-active projection mode is unsupported",
    )
    cells = crosswalk.get("calibration", {}).get("cells")
    require(isinstance(cells, list) and len(cells) == 2, "timing crosswalk requires two controls")
    require(
        {cell.get("id") for cell in cells}
        == {"b84_control_i0_warm_c1", "b84_control_i256_warm_c1"},
        "timing crosswalk control identities drift",
    )
    sm_residuals: list[float] = []
    for cell in cells:
        repeats = cell.get("ncu_sm_active_cycle_repeats")
        require(
            isinstance(repeats, list) and len(repeats) == 7,
            f"{cell['id']}: requires seven SM-active repeats",
        )
        ncu_sm = float(cell["ncu_sm_active_cycles_median"])
        require(
            ncu_sm == statistics.median(float(value) for value in repeats),
            f"{cell['id']}: SM-active median drift",
        )
        sim_issue = float(cell["simulator_issue_cycles"])
        sm_residual = float(cell["sm_active_minus_issue_cycles"])
        require(ncu_sm - sim_issue == sm_residual, f"{cell['id']}: SM-active residual drift")
        sm_residuals.append(sm_residual)
    profiler_cells = crosswalk.get("calibration", {}).get("profiler_cells")
    require(
        isinstance(profiler_cells, list) and len(profiler_cells) == 3,
        "timing crosswalk requires three profiler controls",
    )
    require(
        {cell.get("id") for cell in profiler_cells}
        == {"control_b84_t32_i0", "control_b84_t256_i0", "control_b84_t256_i256"},
        "timing crosswalk profiler control identities drift",
    )
    gpc_residuals: list[float] = []
    for cell in profiler_cells:
        repeats = cell.get("gpc_minus_sm_cycle_repeats")
        require(
            isinstance(repeats, list) and len(repeats) == 7,
            f"{cell['id']}: requires seven GPC-minus-SM repeats",
        )
        residual = float(cell["gpc_elapsed_minus_sm_active_cycles_median"])
        require(
            residual == statistics.median(float(value) for value in repeats),
            f"{cell['id']}: GPC-minus-SM median drift",
        )
        gpc_residuals.append(residual)
    if projection_mode == "fixed_observation_residual":
        require(
            float(active.get("fixed_envelope_cycles")) == statistics.median(sm_residuals),
            "timing crosswalk SM-active envelope is not the control median",
        )
    else:
        require(
            float(active.get("fixed_envelope_cycles")) == 0.0,
            "causal raw-frontier crosswalk must have zero additive SM envelope",
        )
        require(
            active.get("formula") == "raw_issue_frontier",
            "causal raw-frontier crosswalk formula drift",
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


def validate_simulator_feedback(simulator: Mapping[str, Any]) -> None:
    feedback = simulator.get("memory_admission_feedback")
    if feedback is None:
        return
    require(
        feedback
        == "workloads/research_probes/l1tex_admission_backpressure/memory_write_completion_feedback_store_201.json",
        "simulator feedback path drift",
    )
    document = load_object(REPO_ROOT / str(feedback))
    require(document.get("schema_version") == 1, "simulator feedback schema drift")
    candidates = document.get("candidates")
    require(
        isinstance(candidates, list)
        and candidates
        == [
            {
                "candidate_id": 201,
                "scope": "smsp",
                "operation": "store",
                "queue_capacity_instructions": 16,
                "admission_latency_cycles": 8,
                "l2_hit_sector_units": 1,
                "l2_hit_sector_cycles": 1,
                "l2_miss_sector_units": 1,
                "l2_miss_sector_cycles": 10,
                "membar_waits_for_service_completion": True,
            }
        ],
        "simulator STORE completion feedback configuration drift",
    )


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
            "workloads/research_probes/sm86_ncu_demo_benchmark/accuracy_gate_large_workloads.json",
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
        in {
            "workloads/research_probes/sm86_ncu_demo_benchmark/accuracy_gate_counts_single_sim.json",
            "workloads/research_probes/sm86_ncu_demo_benchmark/accuracy_gate_large_workloads.json",
        }
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
    validate_simulator_feedback(spec["simulator"])
    validate_profile(profile, live_host=False)
    require(spec["simulator"]["multi_launch_policy"].startswith("not_admitted"), "v1 multi-launch policy must fail closed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    result.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    result.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    result.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    result.add_argument("--validate-live-host", action="store_true")
    result.add_argument(
        "--require-reference-clocks",
        action="store_true",
        help="when validating the live host, require the profile's configured SM and memory clocks",
    )
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
        if args.require_reference_clocks and not args.validate_live_host:
            raise ValidationError(
                "--require-reference-clocks requires --validate-live-host"
            )
        observed = validate_profile(
            profile,
            live_host=args.validate_live_host,
            require_reference_clocks=args.require_reference_clocks,
        )
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
