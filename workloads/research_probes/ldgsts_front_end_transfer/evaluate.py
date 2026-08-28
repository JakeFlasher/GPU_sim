#!/usr/bin/env python3
"""Evaluate the opt-in SM86 LDGSTS front end on retained CPU replay data."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAPTURES = REPO_ROOT / "results" / "captures"
V4_ROOT = CAPTURES / "sm86_async_barrier_readiness_20260817_v4"
COUPLED_ROOT = CAPTURES / "sm86_barrier_pipeline_coupled_20260817_v1"
CYCLE_ROOT = CAPTURES / "sm86_cycle_residual_20260817_v1"
ISSUE_PORT_ROOT = CAPTURES / "sm86_issue_port_control_20260817_v1"
PRODUCTION_ROOT = CAPTURES / "sm86_fixed_latency_result_load_priority_20260817_v1"

TRACE_GROUPS = 64
LIVE_GROUPS = 16_384
FRONT_END_MODE = "ampere_queue_v1"
FRONT_END_CLI = "ampere-queue-v1"

V4_CONTROL_KEYS = {
    *(f"w4_c{protocol}_na_g{gap}" for protocol in ("00", "01") for gap in (0, 64)),
    *(f"w{width}_c{protocol}_na_g0" for width in (2, 8) for protocol in ("00", "01")),
}
V4_SIGN_KEYS = {
    *V4_CONTROL_KEYS,
    *(
        f"w4_c{protocol}_{locality}_g{gap}"
        for protocol in ("10", "11")
        for locality in ("l1", "l2", "dram")
        for gap in (0, 64)
    ),
    *(
        f"w{width}_c{protocol}_{locality}_g0"
        for width in (2, 8)
        for protocol in ("10", "11")
        for locality in ("l1", "dram")
    ),
}
V4_TRANSFER_KEYS = {
    f"w{width}_c{protocol}_{locality}_g0"
    for width in (2, 8)
    for protocol in ("10", "11", "12")
    for locality in ("l1", "dram")
}
V4_REQUIRED_KEYS = frozenset(V4_SIGN_KEYS | V4_TRANSFER_KEYS)
V4_MANIFEST_KEYS = frozenset(
    {
        *(f"w4_c{protocol}_na_g{gap}" for protocol in ("00", "01") for gap in (0, 64)),
        *(
            f"w4_c{protocol}_{locality}_g{gap}"
            for protocol in ("10", "11", "12")
            for locality in ("l1", "dram")
            for gap in (0, 64)
        ),
        *(
            f"w4_c{protocol}_l2_g{gap}"
            for protocol in ("10", "11")
            for gap in (0, 64)
        ),
        *(f"w{width}_c{protocol}_na_g0" for width in (2, 8) for protocol in ("00", "01")),
        *(
            f"w{width}_c{protocol}_{locality}_g0"
            for width in (2, 8)
            for protocol in ("10", "11", "12")
            for locality in ("l1", "dram")
        ),
    }
)
COUPLED_KEYS = frozenset(
    f"w{width}_c{protocol}_{locality}_g0"
    for width in (2, 4, 8)
    for protocol in ("10", "11")
    for locality in ("l1", "l2", "dram")
)
GUARD_KEYS = frozenset(f"w{width}" for width in range(1, 9))
PRODUCTION_KEYS = frozenset(
    ("attention", "bf16", "canary", "fp16", "int8", "sparse-bf16")
)
EXPECTED_PRODUCTION_BASELINE = {
    "attention": 12_551,
    "bf16": 8_915,
    "canary": 1_102,
    "fp16": 8_915,
    "int8": 5_855,
    "sparse-bf16": 2_375,
}
LAUNCH_IDENTITY_FIELDS = (
    "device_id",
    "stream_id",
    "kernel_id",
    "function_unique_id",
    "kernel_name",
    "grid",
    "block",
    "cluster",
)
STABLE_COUNTER_FIELDS = (
    "instructions_issued",
    "records_seen",
    "l1",
    "l2",
    "dram",
    "unmodeled_tails",
)
L2_SOURCE_LANES = frozenset(("l1_fill", "bypass", "untagged", "store"))


class EvaluationError(RuntimeError):
    """Retained evidence or replay output violates the frozen contract."""


@dataclass(frozen=True)
class ReplaySpec:
    campaign: str
    key: str
    trace: Path
    control: Path
    baseline: Path
    cache_scope: str = "launch"
    kernel_id: int | None = None


@dataclass(frozen=True)
class FrozenEvidence:
    specs: Mapping[str, tuple[ReplaySpec, ...]]
    live_v4: Mapping[str, float]
    live_coupled: Mapping[str, float]
    live_guard: Mapping[str, float]
    live_production: Mapping[str, float]
    baseline_v4: Mapping[str, float]
    baseline_coupled: Mapping[str, float]
    baseline_guard: Mapping[str, float]
    baseline_production: Mapping[str, float]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{path}: cannot read JSON: {error}") from error


def finite_number(value: object, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}: expected number")
    result = float(value)
    require(math.isfinite(result), f"{label}: expected finite number")
    return result


def nonnegative_integer(value: object, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label}: expected nonnegative integer",
    )
    return value


def require_exact_keys(name: str, values: Mapping[str, object], expected: frozenset[str]) -> None:
    require(isinstance(values, Mapping), f"{name}: expected keyed values")
    actual = frozenset(values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    require(not missing and not extra, f"{name}: key mismatch; missing={missing}, extra={extra}")
    for key, value in values.items():
        finite_number(value, f"{name}:{key}")


def manifest_cells(path: Path, expected: frozenset[str]) -> dict[str, dict[str, Any]]:
    document = read_json(path)
    require(isinstance(document, dict), f"{path}: expected object")
    require(document.get("schema_version") == 1, f"{path}: expected schema 1")
    require(document.get("long_groups") == LIVE_GROUPS, f"{path}: long-group mismatch")
    require(document.get("trace_groups") == TRACE_GROUPS, f"{path}: trace-group mismatch")
    cells = document.get("cells")
    require(isinstance(cells, list), f"{path}: cells must be a list")
    by_key: dict[str, dict[str, Any]] = {}
    for cell in cells:
        require(isinstance(cell, dict) and isinstance(cell.get("key"), str), f"{path}: malformed cell")
        key = cell["key"]
        require(key not in by_key, f"{path}: duplicate cell {key}")
        by_key[key] = cell
    require_exact_keys(str(path), {key: 0 for key in by_key}, expected)
    return by_key


def wide_metric(path: Path, metric: str) -> float:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
    except OSError as error:
        raise EvaluationError(f"{path}: cannot read CSV: {error}") from error
    require(len(rows) == 3, f"{path}: expected header, units, and one data row")
    header, units, data = rows
    require(len(header) == len(units) == len(data), f"{path}: row-width mismatch")
    require(len(set(header)) == len(header), f"{path}: duplicate columns")
    require(metric in header, f"{path}: missing {metric}")
    raw = data[header.index(metric)].replace(",", "")
    try:
        value = float(raw)
    except ValueError as error:
        raise EvaluationError(f"{path}: {metric} is not numeric") from error
    require(math.isfinite(value), f"{path}: {metric} is not finite")
    return value


def ncu_active_median(root: Path, key: str, expected_repeats: int) -> float:
    repeat_root = root / "ncu" / key
    paths = sorted(repeat_root.glob("repeat-*/ncu.csv"))
    expected_names = [f"repeat-{index}/ncu.csv" for index in range(1, expected_repeats + 1)]
    actual_names = [str(path.relative_to(repeat_root)) for path in paths]
    require(actual_names == expected_names, f"{key}: NCU repeat mismatch; got {actual_names}")
    return statistics.median(wide_metric(path, "sm__cycles_active.max") for path in paths) / LIVE_GROUPS


def target_launch(document: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    launches = document.get("launches")
    require(isinstance(launches, list) and launches, f"{label}: no launches")
    launch = launches[-1]
    require(isinstance(launch, dict), f"{label}: malformed target launch")
    return launch


def launch_cycles(launch: Mapping[str, Any], label: str) -> float:
    timing = launch.get("timing")
    require(isinstance(timing, dict), f"{label}: malformed timing")
    value = finite_number(
        timing.get("diagnostic_last_issue_cycles"), f"{label}: cycles"
    )
    require(value >= 0.0, f"{label}: cycles must be nonnegative")
    return value


def validated_launch_identity(launch: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    for field in ("device_id", "stream_id", "kernel_id", "function_unique_id"):
        nonnegative_integer(launch.get(field), f"{label}: {field}")
    require(
        isinstance(launch.get("kernel_name"), str) and bool(launch["kernel_name"]),
        f"{label}: kernel_name",
    )
    for field in ("grid", "block", "cluster"):
        geometry = launch.get(field)
        require(
            isinstance(geometry, list)
            and len(geometry) == 3
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in geometry
            ),
            f"{label}: {field}",
        )
    return tuple(
        tuple(launch[field]) if field in {"grid", "block", "cluster"} else launch[field]
        for field in LAUNCH_IDENTITY_FIELDS
    )


def validate_coverage(coverage: object, label: str) -> None:
    require(
        isinstance(coverage, dict) and coverage.get("kind") == "full",
        f"{label}: coverage",
    )
    observed = nonnegative_integer(
        coverage.get("blocks_observed"), f"{label}: blocks observed"
    )
    population = nonnegative_integer(
        coverage.get("population_blocks"), f"{label}: population blocks"
    )
    require(observed == population and observed > 0, f"{label}: block coverage")


def validate_counter_shape(counters: Mapping[str, Any], label: str) -> None:
    require(
        frozenset(counters) == frozenset(STABLE_COUNTER_FIELDS),
        f"{label}: retained counter schema",
    )
    nonnegative_integer(counters.get("instructions_issued"), f"{label}: instructions")
    nonnegative_integer(counters.get("records_seen"), f"{label}: records")
    for component in ("l1", "l2", "dram"):
        values = counters.get(component)
        require(isinstance(values, dict) and values, f"{label}: {component} counters")
        for name, value in values.items():
            nonnegative_integer(value, f"{label}: {component}.{name}")
    tails = counters.get("unmodeled_tails")
    require(
        isinstance(tails, dict)
        and tails
        and all(isinstance(value, bool) for value in tails.values()),
        f"{label}: unmodeled-tail flags",
    )


def stable_candidate_counters(counters: Mapping[str, Any], label: str) -> dict[str, Any]:
    require(
        frozenset(counters) == frozenset((*STABLE_COUNTER_FIELDS, "l2_by_source")),
        f"{label}: candidate counter schema",
    )
    stable = {field: counters[field] for field in STABLE_COUNTER_FIELDS}
    validate_counter_shape(stable, label)

    split = counters.get("l2_by_source")
    require(
        isinstance(split, dict) and frozenset(split) == L2_SOURCE_LANES,
        f"{label}: L2 source split",
    )
    sectors = 0
    hits = 0
    for lane in sorted(L2_SOURCE_LANES):
        values = split.get(lane)
        require(
            isinstance(values, dict) and frozenset(values) == {"sectors", "hits"},
            f"{label}: L2 source lane {lane}",
        )
        sectors += nonnegative_integer(values.get("sectors"), f"{label}: {lane}.sectors")
        hits += nonnegative_integer(values.get("hits"), f"{label}: {lane}.hits")
    l2 = stable["l2"]
    require(
        sectors == l2["hits"] + l2["misses"] and hits == l2["hits"],
        f"{label}: L2 source split does not reconcile",
    )
    return stable


def validate_baseline(document: Mapping[str, Any], spec: ReplaySpec) -> None:
    require(document.get("architecture") == "sm_86", f"{spec.campaign}:{spec.key}: baseline architecture")
    require(document.get("memory_mode") == "l1-l2", f"{spec.campaign}:{spec.key}: baseline memory mode")
    require(
        document.get("ldgsts_front_end", "disabled") == "disabled",
        f"{spec.campaign}:{spec.key}: baseline unexpectedly enables the candidate",
    )
    retained_scope = document.get("cache_scope")
    require(
        retained_scope is None or retained_scope == spec.cache_scope,
        f"{spec.campaign}:{spec.key}: baseline cache scope",
    )
    launches = document.get("launches")
    expected_launches = (
        2
        if spec.campaign == "coupled"
        and ("_l1_" in spec.key or "_l2_" in spec.key)
        else 1
    )
    require(
        isinstance(launches, list) and len(launches) == expected_launches,
        f"{spec.campaign}:{spec.key}: baseline launch count",
    )
    for index, launch in enumerate(launches):
        require(isinstance(launch, dict), f"{spec.campaign}:{spec.key}: baseline launch {index}")
        for field in LAUNCH_IDENTITY_FIELDS:
            require(field in launch, f"{spec.campaign}:{spec.key}: baseline launch {index} missing {field}")
        validated_launch_identity(
            launch, f"{spec.campaign}:{spec.key}: baseline launch {index}"
        )
        counters = launch.get("counters")
        coverage = launch.get("coverage")
        require(isinstance(counters, dict), f"{spec.campaign}:{spec.key}: baseline counters")
        validate_counter_shape(counters, f"{spec.campaign}:{spec.key}: baseline launch {index}")
        require(counters.get("instructions_issued") == counters.get("records_seen"), f"{spec.campaign}:{spec.key}: baseline instruction cardinality")
        validate_coverage(coverage, f"{spec.campaign}:{spec.key}: baseline launch {index}")
    identities = [
        validated_launch_identity(
            launch, f"{spec.campaign}:{spec.key}: baseline launch {index}"
        )
        for index, launch in enumerate(launches)
    ]
    require(len(identities) == len(set(identities)), f"{spec.campaign}:{spec.key}: duplicate baseline launch identity")
    if spec.kernel_id is not None:
        require(launches[-1].get("kernel_id") == spec.kernel_id, f"{spec.campaign}:{spec.key}: baseline target kernel")
    session = document.get("cache_session")
    if spec.cache_scope == "stream-l2":
        require(isinstance(session, dict), f"{spec.campaign}:{spec.key}: baseline cache session")
        require(session.get("scope") == "stream-l2", f"{spec.campaign}:{spec.key}: baseline cache-session scope")
        require(
            nonnegative_integer(
                session.get("completed_launches"),
                f"{spec.campaign}:{spec.key}: baseline completed launches",
            )
            == expected_launches,
            f"{spec.campaign}:{spec.key}: baseline cache-session accounting",
        )
    else:
        require(session is None, f"{spec.campaign}:{spec.key}: unexpected baseline cache session")


def validate_candidate(
    document: Mapping[str, Any], baseline: Mapping[str, Any], spec: ReplaySpec
) -> tuple[Mapping[str, Any], list[str]]:
    validate_baseline(baseline, spec)
    require(document.get("schema_version") == 1, f"{spec.campaign}:{spec.key}: candidate schema")
    require(document.get("architecture") == "sm_86", f"{spec.campaign}:{spec.key}: candidate architecture")
    require(document.get("memory_mode") == "l1-l2", f"{spec.campaign}:{spec.key}: candidate memory mode")
    require(document.get("cache_scope") == spec.cache_scope, f"{spec.campaign}:{spec.key}: candidate cache scope")
    require(document.get("ldgsts_front_end") == FRONT_END_MODE, f"{spec.campaign}:{spec.key}: candidate selector identity")
    candidate_launches = document.get("launches")
    baseline_launches = baseline.get("launches")
    require(isinstance(candidate_launches, list) and isinstance(baseline_launches, list), f"{spec.campaign}:{spec.key}: launch shape")
    require(len(candidate_launches) == len(baseline_launches) and candidate_launches, f"{spec.campaign}:{spec.key}: launch-count mismatch")
    accounting_failures: list[str] = []
    for index, (candidate, retained) in enumerate(zip(candidate_launches, baseline_launches)):
        require(isinstance(candidate, dict) and isinstance(retained, dict), f"{spec.campaign}:{spec.key}: launch {index} shape")
        require(
            validated_launch_identity(
                candidate, f"{spec.campaign}:{spec.key}: candidate launch {index}"
            )
            == validated_launch_identity(
                retained, f"{spec.campaign}:{spec.key}: baseline launch {index}"
            ),
            f"{spec.campaign}:{spec.key}: launch {index} identity mismatch",
        )
        counters = candidate.get("counters")
        coverage = candidate.get("coverage")
        require(isinstance(counters, dict), f"{spec.campaign}:{spec.key}: candidate counters")
        validate_coverage(coverage, f"{spec.campaign}:{spec.key}: candidate launch {index}")
        stable_counters = stable_candidate_counters(
            counters, f"{spec.campaign}:{spec.key}: candidate launch {index}"
        )
        if counters.get("instructions_issued") != counters.get("records_seen"):
            accounting_failures.append(f"launch_{index}:instruction_cardinality")
        if stable_counters != retained.get("counters"):
            accounting_failures.append(f"launch_{index}:traffic_or_counter_drift")
        if coverage != retained.get("coverage"):
            accounting_failures.append(f"launch_{index}:coverage_drift")
    if spec.cache_scope == "stream-l2":
        session = document.get("cache_session")
        require(isinstance(session, dict), f"{spec.campaign}:{spec.key}: missing cache session")
        require(session.get("scope") == "stream-l2", f"{spec.campaign}:{spec.key}: cache-session scope")
        require(
            nonnegative_integer(
                session.get("completed_launches"),
                f"{spec.campaign}:{spec.key}: completed launches",
            )
            == len(candidate_launches),
            f"{spec.campaign}:{spec.key}: cache-session accounting",
        )
        if session != baseline.get("cache_session"):
            accounting_failures.append("cache_session_drift")
    else:
        require(document.get("cache_session") is None, f"{spec.campaign}:{spec.key}: unexpected cache session")
    launch = candidate_launches[-1]
    launch_cycles(launch, f"{spec.campaign}:{spec.key}: candidate")
    return launch, accounting_failures


def replay_specs() -> dict[str, tuple[ReplaySpec, ...]]:
    v4_control = V4_ROOT / "sass" / "async_barrier_overlap_trace.control_bits.json"
    v4_baseline = ISSUE_PORT_ROOT / "regression-branch-forward" / "predecessor"
    # c00/c01 contain no effective LDGSTS instruction. Their current retained
    # replay is therefore the candidate value as well as the baseline value;
    # do not spend a replay process on those interaction controls.
    v4 = tuple(
        ReplaySpec(
            campaign="v4",
            key=key,
            trace=V4_ROOT / "nvbit" / key,
            control=v4_control,
            baseline=v4_baseline / f"{key}-1.json",
            kernel_id=1,
        )
        for key in sorted(V4_REQUIRED_KEYS - V4_CONTROL_KEYS)
    )
    coupled = tuple(
        ReplaySpec(
            campaign="coupled",
            key=key,
            trace=COUPLED_ROOT / "nvbit" / key,
            control=COUPLED_ROOT / "sass" / "async_barrier_overlap_trace.control_bits.json",
            baseline=COUPLED_ROOT / "simulator-branch-forward" / key / "stream-l2-1.json",
            cache_scope="stream-l2",
        )
        for key in sorted(COUPLED_KEYS)
    )
    cross_warp = CYCLE_ROOT / "research-probes" / "async-cross-warp"
    guard = tuple(
        ReplaySpec(
            campaign="guard",
            key=f"w{width}",
            trace=cross_warp / "nvbit-small" / f"warps-{width}",
            control=cross_warp / "async_cross_warp_trace_all.control_bits.json",
            baseline=cross_warp / "nvbit-small" / f"warps-{width}" / "sim.group-queue.json",
            kernel_id=1,
        )
        for width in range(1, 9)
    )
    current_production = PRODUCTION_ROOT / "regression-priority" / "production"
    production = tuple(
        ReplaySpec(
            campaign="production",
            key=key,
            trace=CYCLE_ROOT / "nvbit" / key,
            control=CYCLE_ROOT / "control" / f"{key}.control_bits.json",
            baseline=current_production / f"{key}-1.json",
        )
        for key in sorted(PRODUCTION_KEYS)
    )
    return {"v4": v4, "coupled": coupled, "guard": guard, "production": production}


def guard_live_labels() -> dict[str, float]:
    path = CYCLE_ROOT / "research-probes" / "async-cross-warp" / "direct.full.live.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{path}: cannot read guard labels: {error}") from error
    require(len(rows) == 40, f"{path}: expected 40 retained measurements")
    samples: dict[str, list[float]] = {key: [] for key in GUARD_KEYS}
    repetitions: dict[str, int] = {key: 0 for key in GUARD_KEYS}
    for row in rows:
        require(isinstance(row, dict), f"{path}: malformed measurement")
        width = row.get("warps")
        key = f"w{width}"
        require(key in samples and row.get("iterations") == LIVE_GROUPS and row.get("batch") == 8, f"{path}: guard identity mismatch")
        trials = row.get("samples_cycles")
        require(isinstance(trials, list) and len(trials) == 7, f"{key}: expected seven guard trials")
        for trial in trials:
            require(isinstance(trial, list) and len(trial) == width, f"{key}: guard warp count mismatch")
            samples[key].append(max(finite_number(value, f"{key}: guard sample") for value in trial))
        repetitions[key] += 1
    require(all(count == 5 for count in repetitions.values()), f"{path}: expected five repetitions per width")
    return {key: statistics.median(values) / LIVE_GROUPS for key, values in samples.items()}


def load_frozen_evidence() -> FrozenEvidence:
    manifest_cells(V4_ROOT / "metadata" / "matrix.json", V4_MANIFEST_KEYS)
    manifest_cells(COUPLED_ROOT / "metadata" / "matrix.json", COUPLED_KEYS)
    specs = replay_specs()
    baselines: dict[str, dict[str, float]] = {name: {} for name in specs}
    for campaign, campaign_specs in specs.items():
        for spec in campaign_specs:
            baseline = read_json(spec.baseline)
            require(isinstance(baseline, dict), f"{spec.baseline}: expected object")
            validate_baseline(baseline, spec)
            cycles = launch_cycles(
                target_launch(baseline, f"{campaign}:{spec.key}"),
                f"{campaign}:{spec.key}: baseline",
            )
            baselines[campaign][spec.key] = cycles / (
                1 if campaign == "production" else TRACE_GROUPS
            )
    for key in sorted(V4_CONTROL_KEYS):
        spec = ReplaySpec(
            campaign="v4",
            key=key,
            trace=V4_ROOT / "nvbit" / key,
            control=V4_ROOT / "sass" / "async_barrier_overlap_trace.control_bits.json",
            baseline=ISSUE_PORT_ROOT
            / "regression-branch-forward"
            / "predecessor"
            / f"{key}-1.json",
            kernel_id=1,
        )
        baseline = read_json(spec.baseline)
        require(isinstance(baseline, dict), f"{spec.baseline}: expected object")
        validate_baseline(baseline, spec)
        cycles = launch_cycles(
            target_launch(baseline, f"v4:{key}"), f"v4:{key}: baseline"
        )
        baselines["v4"][key] = cycles / TRACE_GROUPS
    require_exact_keys("baseline v4", baselines["v4"], V4_REQUIRED_KEYS)
    require_exact_keys("baseline coupled", baselines["coupled"], COUPLED_KEYS)
    require_exact_keys("baseline guard", baselines["guard"], GUARD_KEYS)
    require_exact_keys("baseline production", baselines["production"], PRODUCTION_KEYS)
    require(
        dict(baselines["production"]) == EXPECTED_PRODUCTION_BASELINE,
        "production baseline does not match the retained current model",
    )

    live_v4 = {
        key: ncu_active_median(
            V4_ROOT,
            key,
            5 if key in {"w4_c10_l2_g64", "w4_c11_l2_g64"} else 3,
        )
        for key in V4_REQUIRED_KEYS
    }
    live_coupled = {
        key: ncu_active_median(
            COUPLED_ROOT,
            key,
            5 if key in {
                "w4_c10_l2_g0",
                "w4_c11_l2_g0",
                "w8_c10_l2_g0",
                "w8_c11_l2_g0",
            } else 3,
        )
        for key in COUPLED_KEYS
    }
    production_live_raw = read_json(CYCLE_ROOT / "analysis" / "observed_sm_active_median.json")
    require(isinstance(production_live_raw, dict), "production labels must be keyed")
    production_live = {key: finite_number(value, f"production live:{key}") for key, value in production_live_raw.items()}
    require_exact_keys("production live", production_live, PRODUCTION_KEYS)
    return FrozenEvidence(
        specs=specs,
        live_v4=live_v4,
        live_coupled=live_coupled,
        live_guard=guard_live_labels(),
        live_production=production_live,
        baseline_v4=baselines["v4"],
        baseline_coupled=baselines["coupled"],
        baseline_guard=baselines["guard"],
        baseline_production=baselines["production"],
    )


def normalized_rows(
    cells: Sequence[tuple[str, str]],
    live: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    baseline_errors: list[float] = []
    candidate_errors: list[float] = []
    for campaign, key in cells:
        observed = live[campaign][key]
        before = baseline[campaign][key]
        after = candidate[campaign][key]
        baseline_error = abs(before - observed)
        candidate_error = abs(after - observed)
        baseline_errors.append(baseline_error)
        candidate_errors.append(candidate_error)
        rows.append(
            {
                "campaign": campaign,
                "key": key,
                "live_cycles_per_group": observed,
                "baseline_cycles_per_group": before,
                "candidate_cycles_per_group": after,
                "baseline_absolute_error": baseline_error,
                "candidate_absolute_error": candidate_error,
            }
        )
    return rows, statistics.mean(baseline_errors), statistics.mean(candidate_errors)


def nonzero_sign(value: float, label: str) -> int:
    require(value != 0.0, f"{label}: retained live contrast is zero")
    return 1 if value > 0.0 else -1


def normalized_accounting_failures(
    failures_by_cell: Mapping[str, Sequence[str]],
    allowed_cells: set[str],
) -> list[dict[str, Any]]:
    require(
        set(failures_by_cell) <= allowed_cells,
        "accounting failures contain an unexpected cell",
    )
    rows: list[dict[str, Any]] = []
    for composite_key in sorted(failures_by_cell):
        failures = failures_by_cell[composite_key]
        require(
            isinstance(composite_key, str)
            and isinstance(failures, Sequence)
            and not isinstance(failures, (str, bytes))
            and all(isinstance(failure, str) and failure for failure in failures),
            "accounting failures are malformed",
        )
        if failures:
            rows.append(
                {"cell": composite_key, "failures": sorted(set(failures))}
            )
    return rows


def reduce_core(
    *,
    live_v4: Mapping[str, float],
    live_coupled: Mapping[str, float],
    live_guard: Mapping[str, float],
    baseline_v4: Mapping[str, float],
    baseline_coupled: Mapping[str, float],
    baseline_guard: Mapping[str, float],
    candidate_v4: Mapping[str, float],
    candidate_coupled: Mapping[str, float],
    candidate_guard: Mapping[str, float],
    accounting_failures: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    for name, values, expected in (
        ("live v4", live_v4, V4_REQUIRED_KEYS),
        ("baseline v4", baseline_v4, V4_REQUIRED_KEYS),
        ("candidate v4", candidate_v4, V4_REQUIRED_KEYS),
        ("live coupled", live_coupled, COUPLED_KEYS),
        ("baseline coupled", baseline_coupled, COUPLED_KEYS),
        ("candidate coupled", candidate_coupled, COUPLED_KEYS),
        ("live guard", live_guard, GUARD_KEYS),
        ("baseline guard", baseline_guard, GUARD_KEYS),
        ("candidate guard", candidate_guard, GUARD_KEYS),
    ):
        require_exact_keys(name, values, expected)
    accounting_failures = accounting_failures or {}
    allowed_accounting_cells = {
        *(f"v4:{key}" for key in V4_REQUIRED_KEYS - V4_CONTROL_KEYS),
        *(f"coupled:{key}" for key in COUPLED_KEYS),
        *(f"guard:{key}" for key in GUARD_KEYS),
    }
    accounting_rows = normalized_accounting_failures(
        accounting_failures, allowed_accounting_cells
    )

    transfer_cells = [
        *(('coupled', key) for key in sorted(COUPLED_KEYS)),
        *(('v4', key) for key in sorted(V4_TRANSFER_KEYS)),
    ]
    transfer_rows, transfer_before, transfer_after = normalized_rows(
        transfer_cells,
        {"v4": live_v4, "coupled": live_coupled},
        {"v4": baseline_v4, "coupled": baseline_coupled},
        {"v4": candidate_v4, "coupled": candidate_coupled},
    )
    require(len(transfer_rows) == 30, "transfer matrix must contain 30 composite-key cells")

    guard_rows, guard_before, guard_after = normalized_rows(
        [("guard", key) for key in sorted(GUARD_KEYS)],
        {"guard": live_guard},
        {"guard": baseline_guard},
        {"guard": candidate_guard},
    )

    route_rows: list[dict[str, Any]] = []
    for width in (2, 4, 8):
        for protocol in ("c10", "c11"):
            keys = [f"w{width}_{protocol}_{locality}_g0" for locality in ("l1", "l2", "dram")]
            live_values = [live_coupled[key] for key in keys]
            candidate_values = [candidate_coupled[key] for key in keys]
            live_strict = live_values[0] < live_values[1] < live_values[2]
            require(live_strict, f"W{width} {protocol}: retained live route order is not strict")
            route_rows.append(
                {
                    "width": width,
                    "protocol": protocol,
                    "keys": keys,
                    "live_cycles_per_group": live_values,
                    "candidate_cycles_per_group": candidate_values,
                    "passed": candidate_values[0] < candidate_values[1] < candidate_values[2],
                }
            )

    v4_sign_rows: list[dict[str, Any]] = []
    sign_cells = [
        *((4, locality, gap) for locality in ("l1", "l2", "dram") for gap in (0, 64)),
        *((width, locality, 0) for width in (2, 8) for locality in ("l1", "dram")),
    ]
    for width, locality, gap in sign_cells:
        c00 = f"w{width}_c00_na_g{gap}"
        c01 = f"w{width}_c01_na_g{gap}"
        c10 = f"w{width}_c10_{locality}_g{gap}"
        c11 = f"w{width}_c11_{locality}_g{gap}"
        live_value = (live_v4[c11] - live_v4[c10]) - (live_v4[c01] - live_v4[c00])
        candidate_value = (candidate_v4[c11] - candidate_v4[c10]) - (
            candidate_v4[c01] - candidate_v4[c00]
        )
        expected_sign = nonzero_sign(live_value, f"v4 W{width} {locality} gap {gap}")
        v4_sign_rows.append(
            {
                "width": width,
                "locality": locality,
                "gap": gap,
                "live_corrected_interaction": live_value,
                "candidate_corrected_interaction": candidate_value,
                "live_sign": expected_sign,
                "candidate_sign": 0 if candidate_value == 0 else (1 if candidate_value > 0 else -1),
                "passed": candidate_value != 0 and (1 if candidate_value > 0 else -1) == expected_sign,
            }
        )
    require(len(v4_sign_rows) == 10, "v4 sign matrix must contain ten cells")

    coupled_sign_rows: list[dict[str, Any]] = []
    for width in (2, 4, 8):
        for locality in ("l1", "l2", "dram"):
            c10 = f"w{width}_c10_{locality}_g0"
            c11 = f"w{width}_c11_{locality}_g0"
            live_value = live_coupled[c11] - live_coupled[c10]
            candidate_value = candidate_coupled[c11] - candidate_coupled[c10]
            expected_sign = nonzero_sign(live_value, f"coupled W{width} {locality}")
            require(expected_sign == 1, f"coupled W{width} {locality}: retained contrast is not positive")
            coupled_sign_rows.append(
                {
                    "width": width,
                    "locality": locality,
                    "live_c11_minus_c10": live_value,
                    "candidate_c11_minus_c10": candidate_value,
                    "passed": candidate_value > 0.0,
                }
            )
    require(len(coupled_sign_rows) == 9, "coupled sign matrix must contain nine cells")

    gates = [
        {"name": "transfer_mae_strictly_below_baseline", "passed": transfer_after < transfer_before},
        {"name": "warmed_source_mae_not_above_baseline", "passed": guard_after <= guard_before},
        {"name": "coupled_route_order_strict", "passed": all(row["passed"] for row in route_rows)},
        {"name": "v4_corrected_interaction_signs_match", "passed": all(row["passed"] for row in v4_sign_rows)},
        {"name": "coupled_contrast_signs_match", "passed": all(row["passed"] for row in coupled_sign_rows)},
        {
            "name": "traffic_and_instruction_accounting_exact",
            "passed": not accounting_rows,
        },
    ]
    failed = [gate["name"] for gate in gates if not gate["passed"]]
    return {
        "schema_version": 1,
        "candidate": FRONT_END_MODE,
        "normalization": {"simulator_groups": TRACE_GROUPS, "live_groups": LIVE_GROUPS},
        "transfer": {
            "cell_count": len(transfer_rows),
            "baseline_mae_cycles_per_group": transfer_before,
            "candidate_mae_cycles_per_group": transfer_after,
            "rows": transfer_rows,
        },
        "warmed_source_guard": {
            "cell_count": len(guard_rows),
            "baseline_mae_cycles_per_group": guard_before,
            "candidate_mae_cycles_per_group": guard_after,
            "rows": guard_rows,
        },
        "route_order": {"group_count": len(route_rows), "rows": route_rows},
        "v4_corrected_interaction_signs": {"cell_count": len(v4_sign_rows), "rows": v4_sign_rows},
        "coupled_contrast_signs": {"cell_count": len(coupled_sign_rows), "rows": coupled_sign_rows},
        "accounting": {
            "exact": not accounting_rows,
            "failed_cells": accounting_rows,
            "probe_replay_cells": len(V4_REQUIRED_KEYS - V4_CONTROL_KEYS)
            + len(COUPLED_KEYS)
            + len(GUARD_KEYS),
            "retained_control_cells": len(V4_CONTROL_KEYS),
        },
        "gates": gates,
        "failed_gates": failed,
        "verdict": "retained_opt_in" if not failed else "not_retained",
        "timing_transfer_gates_frozen_before_production": True,
    }


def score_production(observed: Mapping[str, float], predicted: Mapping[str, float]) -> dict[str, Any]:
    require_exact_keys("production observed", observed, PRODUCTION_KEYS)
    require_exact_keys("production predicted", predicted, PRODUCTION_KEYS)
    rows = []
    absolute_errors = []
    percentage_errors = []
    squared_errors = []
    for key in sorted(PRODUCTION_KEYS):
        live = observed[key]
        prediction = predicted[key]
        require(live > 0.0, f"production {key}: live value must be positive")
        require(prediction >= 0.0, f"production {key}: prediction must be nonnegative")
        error = finite_number(prediction - live, f"production {key}: error")
        absolute_error = finite_number(abs(error), f"production {key}: absolute error")
        percentage_error = finite_number(
            absolute_error / live, f"production {key}: percentage error"
        )
        squared_error = finite_number(error * error, f"production {key}: squared error")
        absolute_errors.append(absolute_error)
        percentage_errors.append(percentage_error)
        squared_errors.append(squared_error)
        rows.append({"key": key, "live_cycles": live, "predicted_cycles": prediction, "absolute_error": absolute_error})
    mae = finite_number(statistics.mean(absolute_errors), "production MAE")
    rmse = finite_number(statistics.mean(squared_errors) ** 0.5, "production RMSE")
    mape = finite_number(statistics.mean(percentage_errors), "production MAPE")
    return {
        "coverage": len(rows),
        "mae_cycles": mae,
        "rmse_cycles": rmse,
        "mape": mape,
        "rows": rows,
    }


def attach_production(
    core: Mapping[str, Any],
    *,
    live: Mapping[str, float],
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    accounting_failures: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    before = score_production(live, baseline)
    after = score_production(live, candidate)
    baseline_errors = {row["key"]: row["absolute_error"] for row in before["rows"]}
    candidate_errors = {row["key"]: row["absolute_error"] for row in after["rows"]}
    result = copy.deepcopy(dict(core))
    gates = result.get("gates")
    require(isinstance(gates, list), "core gates are malformed")
    timing_gate_snapshot = copy.deepcopy(
        [
            gate
            for gate in gates
            if gate.get("name") != "traffic_and_instruction_accounting_exact"
        ]
    )
    result["probe_verdict_before_production_replay"] = result.get("verdict")
    production_accounting = normalized_accounting_failures(
        accounting_failures or {},
        {f"production:{key}" for key in PRODUCTION_KEYS},
    )
    accounting = result.get("accounting")
    require(isinstance(accounting, dict), "core accounting is malformed")
    prior_rows = accounting.get("failed_cells")
    require(isinstance(prior_rows, list), "core accounting failures are malformed")
    all_accounting_rows = sorted(
        [*prior_rows, *production_accounting], key=lambda row: row["cell"]
    )
    accounting["failed_cells"] = all_accounting_rows
    accounting["exact"] = not all_accounting_rows
    accounting["production_cells"] = len(PRODUCTION_KEYS)
    accounting["total_replay_cells"] = accounting["probe_replay_cells"] + len(
        PRODUCTION_KEYS
    )
    for gate in gates:
        if gate.get("name") == "traffic_and_instruction_accounting_exact":
            gate["passed"] = not all_accounting_rows
            break
    else:
        raise EvaluationError("core accounting gate is missing")
    result["failed_gates"] = [gate["name"] for gate in gates if not gate["passed"]]
    result["verdict"] = (
        "retained_opt_in" if not result["failed_gates"] else "not_retained"
    )
    result["production"] = {
        "timing_labels_verdict_inert": True,
        "accounting_participates_in_exactness_gate": True,
        "baseline": before,
        "candidate": after,
        "all_six_absolute_errors_improved": all(
            candidate_errors[key] < baseline_errors[key] for key in PRODUCTION_KEYS
        ),
        "accounting_failures": production_accounting,
    }
    require(
        timing_gate_snapshot
        == [
            gate
            for gate in result["gates"]
            if gate.get("name") != "traffic_and_instruction_accounting_exact"
        ],
        "production scoring changed a frozen timing-transfer gate",
    )
    return result


def replay_command(binary: Path, spec: ReplaySpec) -> list[str]:
    require(binary.name == "sim-run", f"unexpected simulator executable: {binary}")
    command = [
        str(binary),
        "--trace",
        str(spec.trace),
        "--control",
        str(spec.control),
        "--arch-card",
        str(REPO_ROOT / "arch_profiles" / "cards" / "sm_86.json"),
        "--pchase-card",
        str(REPO_ROOT / "data" / "pchase" / "sm_86_static_card.json"),
        "--mode",
        "l1-l2",
        "--cache-scope",
        spec.cache_scope,
        "--ldgsts-front-end",
        FRONT_END_CLI,
    ]
    if spec.kernel_id is not None:
        command.extend(("--kernel-id", str(spec.kernel_id)))
    return command


def build_simulator(binary: Path, skip_build: bool) -> None:
    if not skip_build:
        completed = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(REPO_ROOT / "sim" / "Cargo.toml"),
                "-p",
                "sim-run",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=1_800,
        )
        require(completed.returncode == 0, f"sim-run build failed:\n{completed.stderr[-4000:]}")
    require(binary.is_file(), f"sim-run is missing: {binary}")


def run_campaign(
    binary: Path,
    specs: Sequence[ReplaySpec],
    timeout: int,
) -> tuple[
    dict[str, float],
    dict[str, Mapping[str, Any]],
    dict[str, list[str]],
]:
    cycles: dict[str, float] = {}
    launches: dict[str, Mapping[str, Any]] = {}
    accounting_failures: dict[str, list[str]] = {}
    identities = [(spec.campaign, spec.key) for spec in specs]
    require(
        len(identities) == len(set(identities)),
        "replay specification contains duplicate campaign cells",
    )
    for index, spec in enumerate(specs, 1):
        require(spec.trace.is_dir(), f"{spec.campaign}:{spec.key}: trace directory missing")
        require(spec.control.is_file(), f"{spec.campaign}:{spec.key}: control sidecar missing")
        baseline = read_json(spec.baseline)
        require(isinstance(baseline, dict), f"{spec.baseline}: expected object")
        command = replay_command(binary, spec)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        require(
            completed.returncode == 0,
            f"{spec.campaign}:{spec.key}: replay failed with {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace')[-2000:]}",
        )
        try:
            document = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"{spec.campaign}:{spec.key}: malformed replay JSON") from error
        require(isinstance(document, dict), f"{spec.campaign}:{spec.key}: replay must be an object")
        launch, failures = validate_candidate(document, baseline, spec)
        value = launch_cycles(launch, f"{spec.campaign}:{spec.key}: candidate")
        cycles[spec.key] = value if spec.campaign == "production" else value / TRACE_GROUPS
        launches[spec.key] = launch
        if failures:
            accounting_failures[f"{spec.campaign}:{spec.key}"] = failures
        print(f"{spec.campaign} {index:02d}/{len(specs):02d} {spec.key}", flush=True)
    return cycles, launches, accounting_failures


def mechanism_sanity(launches: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = [f"w4_c10_{locality}_g{gap}" for locality in ("l1", "l2", "dram") for gap in (0, 64)]
    rows = []
    for key in keys:
        launch = launches.get(key)
        require(isinstance(launch, dict), f"mechanism sanity missing {key}")
        timing = launch.get("timing")
        require(isinstance(timing, dict), f"mechanism sanity {key}: timing")
        readiness = timing.get("readiness")
        require(isinstance(readiness, dict), f"mechanism sanity {key}: readiness")
        require(readiness.get("schema_version") == 14, f"mechanism sanity {key}: readiness schema")
        diagnostic = readiness.get("ldgsts_front_end")
        require(isinstance(diagnostic, dict), f"mechanism sanity {key}: missing diagnostics")
        require(set(diagnostic) == {"issued_tokens", "queue_blocked_issues", "excess_delay_cycles", "maximum_occupancy"}, f"mechanism sanity {key}: diagnostic keys")
        for name, value in diagnostic.items():
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"mechanism sanity {key}: {name}")
        require(diagnostic["issued_tokens"] > 0, f"mechanism sanity {key}: no issued tokens")
        require(1 <= diagnostic["maximum_occupancy"] <= 5, f"mechanism sanity {key}: occupancy")
        rows.append({"key": key, **diagnostic})
    return rows


def write_json(path: Path, value: object) -> None:
    try:
        encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    except (OSError, ValueError) as error:
        raise EvaluationError(f"cannot write {path}: {error}") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-root",
        type=Path,
        default=CAPTURES / "sm86_ldgsts_front_end_transfer_20260828_v1",
    )
    result.add_argument(
        "--sim-run",
        type=Path,
        default=REPO_ROOT / "sim" / "target" / "release" / "sim-run",
    )
    result.add_argument("--no-build", action="store_true")
    result.add_argument("--timeout", type=int, default=900)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        require(args.timeout > 0, "timeout must be positive")
        output_root = args.output_root.resolve()
        result_path = output_root / "evaluation.json"
        require(not result_path.exists(), f"result already exists: {result_path}")
        evidence = load_frozen_evidence()
        binary = args.sim_run.resolve()
        build_simulator(binary, args.no_build)

        candidate: dict[str, dict[str, float]] = {}
        launch_reports: dict[str, dict[str, Mapping[str, Any]]] = {}
        accounting_failures: dict[str, list[str]] = {}
        for campaign in ("v4", "coupled", "guard"):
            candidate[campaign], launch_reports[campaign], failures = run_campaign(
                binary, evidence.specs[campaign], args.timeout
            )
            accounting_failures.update(failures)
        candidate["v4"].update(
            {key: evidence.baseline_v4[key] for key in V4_CONTROL_KEYS}
        )
        core = reduce_core(
            live_v4=evidence.live_v4,
            live_coupled=evidence.live_coupled,
            live_guard=evidence.live_guard,
            baseline_v4=evidence.baseline_v4,
            baseline_coupled=evidence.baseline_coupled,
            baseline_guard=evidence.baseline_guard,
            candidate_v4=candidate["v4"],
            candidate_coupled=candidate["coupled"],
            candidate_guard=candidate["guard"],
            accounting_failures=accounting_failures,
        )
        core["mechanism_sanity"] = mechanism_sanity(launch_reports["v4"])

        candidate_production, _, production_accounting = run_campaign(
            binary, evidence.specs["production"], args.timeout
        )
        result = attach_production(
            core,
            live=evidence.live_production,
            baseline=evidence.baseline_production,
            candidate=candidate_production,
            accounting_failures=production_accounting,
        )
        write_json(result_path, result)
    except (EvaluationError, OSError, subprocess.TimeoutExpired) as error:
        print(f"LDGSTS front-end transfer: {error}", file=sys.stderr)
        return 1
    print(f"{result_path.resolve()}\n{result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
