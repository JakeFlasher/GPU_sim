"""Independent semantic checks for direct-simulator replay JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from workloads.research_probes.trace_invariants import (
    LaunchTraceInvariants,
    derive_trace_invariants,
)


def _identity(launch: Mapping[str, Any]) -> tuple[int, int, int] | None:
    values = tuple(launch.get(key) for key in ("device_id", "stream_id", "kernel_id"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return None
    return values  # type: ignore[return-value]


def _int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _nonzero_counters(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _int(item)
        for key, item in value.items()
        if _int(item) != 0
    }


def _expected_global_memory(
    expected: LaunchTraceInvariants,
) -> dict[str, int]:
    return {
        "load_instructions": expected.global_load_instructions,
        "store_instructions": expected.global_store_instructions,
        "atomic_instructions": expected.global_atomic_instructions,
        "load_sectors": expected.global_load_sectors,
        "store_sectors": expected.global_store_sectors,
        "atomic_sectors": expected.global_atomic_sectors,
        "async_source_sectors": expected.async_source_sectors,
    }


def _global_memory_error(
    memory: Mapping[str, Any],
    expected: LaunchTraceInvariants,
    label: str,
) -> tuple[str | None, dict[str, Any]]:
    expected_values = _expected_global_memory(expected)
    required_fields = (
        "load_instructions",
        "store_instructions",
        "atomic_instructions",
        "reduction_instructions",
        "load_sectors",
        "store_sectors",
        "atomic_sectors",
        "reduction_sectors",
    )
    malformed = {
        key: memory.get(key)
        for key in required_fields
        if type(memory.get(key)) is not int or memory.get(key) < 0
    }
    if malformed:
        return (
            f"{label}: global_memory_issue has malformed or negative fields {malformed}",
            {
                "status": "malformed",
                "malformed_fields": malformed,
                "expected": expected_values,
            },
        )
    observed = {
        key: _int(memory.get(key))
        for key in (
            "load_instructions",
            "store_instructions",
            "atomic_instructions",
            "reduction_instructions",
            "load_sectors",
            "store_sectors",
            "atomic_sectors",
            "reduction_sectors",
        )
    }
    checks = {
        "global_load_instructions": observed["load_instructions"]
        == expected_values["load_instructions"],
        "global_store_instructions": observed["store_instructions"]
        == expected_values["store_instructions"],
        "global_atomic_instructions": observed["atomic_instructions"]
        + observed["reduction_instructions"]
        == expected_values["atomic_instructions"],
        "global_load_sectors": observed["load_sectors"]
        == expected_values["load_sectors"],
        "global_store_sectors": observed["store_sectors"]
        == expected_values["store_sectors"],
        "global_atomic_sectors": observed["atomic_sectors"]
        + observed["reduction_sectors"]
        == expected_values["atomic_sectors"],
    }
    error = next(
        (
            f"{label}: trace-derived {name} mismatch "
            f"(observed={observed}, expected={expected_values})"
            for name, passed in checks.items()
            if not passed
        ),
        None,
    )
    return error, {"observed": observed, "expected": expected_values, "checks": checks}


def _async_source_error(
    counters: Mapping[str, Any],
    expected: LaunchTraceInvariants,
    label: str,
) -> tuple[str | None, dict[str, Any]]:
    """Check the source-sector conservation lane for cacheable async copies."""

    if expected.async_source_sectors == 0:
        return None, {
            "status": "not_applicable",
            "promotion_eligible": True,
            "expected_async_source_sectors": 0,
        }
    l1 = counters.get("l1")
    if not isinstance(l1, Mapping):
        return f"{label}: async-source invariant has no L1 counters", {
            "status": "unavailable",
            "promotion_eligible": False,
            "expected_async_source_sectors": expected.async_source_sectors,
        }
    observed = _int(l1.get("hits")) + _int(l1.get("misses"))
    # A bypass route contributes to the same trace-visible source total.  A
    # mixed atomic/async trace has no exact attribution invariant in this
    # schema, so it is explicitly unknown and must not be promoted as passed.
    ambiguous = bool(
        expected.global_atomic_sectors != 0
        or getattr(expected, "global_atomic_instructions", 0) != 0
    )
    if not ambiguous:
        observed += _int(l1.get("bypass_sectors"))
    expected_total = expected.global_load_sectors + expected.async_source_sectors
    snapshot = {
        "status": "unknown" if ambiguous else "checked",
        "promotion_eligible": not ambiguous,
        "unknown_reason": (
            "atomic and async-source sectors share the observed L1 lane"
            if ambiguous
            else None
        ),
        "expected_async_source_sectors": expected.async_source_sectors,
        "expected_l1_read_sectors": expected_total,
        "observed_l1_read_sectors": observed,
        "passed": None if ambiguous else observed == expected_total,
    }
    if ambiguous:
        return (
            f"{label}: async-source conservation is unknown for mixed atomic/async traffic",
            snapshot,
        )
    if observed != expected_total:
        return f"{label}: trace-derived async_source_sectors mismatch ({snapshot})", snapshot
    return None, snapshot


def _hierarchy_error(counters: Mapping[str, Any], label: str) -> tuple[str | None, dict[str, Any]]:
    l2 = counters.get("l2")
    by_source = counters.get("l2_by_source")
    if not isinstance(l2, Mapping) or not isinstance(by_source, Mapping):
        return f"{label}: hierarchy counters are missing", {}
    source_sectors = sum(
        _int(by_source.get(source, {}).get("sectors"))
        for source in ("l1_fill", "bypass", "untagged", "store")
        if isinstance(by_source.get(source), Mapping)
    )
    source_hits = sum(
        _int(by_source.get(source, {}).get("hits"))
        for source in ("l1_fill", "bypass", "untagged", "store")
        if isinstance(by_source.get(source), Mapping)
    )
    observed = {
        "source_sectors": source_sectors,
        "source_hits": source_hits,
        "l2_total": _int(l2.get("hits")) + _int(l2.get("misses")),
        "l2_hits": _int(l2.get("hits")),
        "l2_line_misses": _int(l2.get("line_misses")),
        "l2_sector_fill_misses": _int(l2.get("sector_fill_misses")),
        "l2_write_tag_hits": _int(l2.get("write_tag_hits")),
        "dram_read_sectors": _int(counters.get("dram", {}).get("read_sectors"))
        if isinstance(counters.get("dram"), Mapping)
        else 0,
    }
    checks = {
        "source_sector_conservation": observed["source_sectors"] == observed["l2_total"],
        "source_hit_conservation": observed["source_hits"] == observed["l2_hits"],
        "line_misses_bounded": observed["l2_line_misses"] <= _int(l2.get("misses")),
        "sector_fill_misses_bounded": observed["l2_sector_fill_misses"]
        <= _int(l2.get("misses")),
        "write_tag_hits_bounded": observed["l2_write_tag_hits"] <= observed["l2_hits"],
        "dram_misses_bounded": observed["dram_read_sectors"] <= observed["l2_total"],
    }
    for name, passed in checks.items():
        if not passed:
            return f"{label}: hierarchy invariant {name} failed ({observed})", {
                "observed": observed,
                "checks": checks,
            }
    return None, {"observed": observed, "checks": checks}


def _readiness_error(launch: Mapping[str, Any], label: str) -> tuple[str | None, dict[str, Any]]:
    timing = launch.get("timing")
    if not isinstance(timing, Mapping):
        return f"{label}: timing is missing", {}
    issue = _int(timing.get("diagnostic_last_issue_cycles"))
    terminal = timing.get("terminal_modeled_event_time")
    last_event = timing.get("last_processed_modeled_system_event_time")
    terminal_value = _int(terminal) if terminal is not None else None
    last_value = _int(last_event) if last_event is not None else None
    frontier = timing.get("completion_frontier")
    readiness = timing.get("readiness")
    if not isinstance(frontier, Mapping) or not isinstance(readiness, Mapping):
        return f"{label}: completion/readiness diagnostics are missing", {}
    checks: dict[str, bool] = {}
    if terminal_value is not None:
        checks["terminal_ge_issue"] = terminal_value >= issue
    if last_value is not None and terminal_value is not None:
        checks["last_event_le_terminal"] = last_value <= terminal_value
    async_groups = readiness.get("async_groups")
    if isinstance(async_groups, Mapping):
        checks["async_waits_le_commits"] = _int(async_groups.get("groups_waited")) <= _int(
            async_groups.get("groups_committed")
        )
        checks["async_cause_groups_bounded"] = sum(
            _int(async_groups.get(key))
            for key in (
                "source_limited_groups",
                "destination_local_limited_groups",
                "destination_sm_service_limited_groups",
            )
        ) <= _int(async_groups.get("groups_waited"))
    barriers = readiness.get("cta_barriers")
    pipeline = readiness.get("barrier_pipeline")
    if isinstance(barriers, Mapping) and isinstance(pipeline, Mapping):
        checks["barrier_generation_conservation"] = _int(
            barriers.get("generations")
        ) == _int(pipeline.get("generations_completed"))

    tails = launch.get("counters", {}).get("unmodeled_tails")
    frontier_tails = frontier.get("unmodeled_tails")
    if frontier.get("status") == "incomplete_unmodeled_retirement_tails":
        checks["frontier_tail_flags_match"] = tails == frontier_tails
        if terminal_value is not None:
            checks["frontier_lower_bound_matches_terminal"] = _int(
                frontier.get("lower_bound_cycles")
            ) == terminal_value
    elif frontier.get("status") == "complete":
        checks["complete_frontier_has_no_tails"] = not any(tails.values()) if isinstance(tails, Mapping) else False

    for name, passed in checks.items():
        if not passed:
            return f"{label}: readiness invariant {name} failed", {
                "checks": checks,
                "readiness_schema": readiness.get("schema_version"),
                "limiting_cycles": _nonzero_counters(readiness.get("limiting_cycles")),
                "constraint_exposure_cycles": _nonzero_counters(
                    readiness.get("constraint_exposure_cycles")
                ),
                "async_groups": async_groups,
                "cta_barriers": barriers,
                "timing": {
                    "issue": issue,
                    "last_event": last_value,
                    "terminal": terminal_value,
                    "completion_frontier": frontier,
                },
            }
    return None, {
        "checks": checks,
        "readiness_schema": readiness.get("schema_version"),
        "limiting_cycles": _nonzero_counters(readiness.get("limiting_cycles")),
        "constraint_exposure_cycles": _nonzero_counters(
            readiness.get("constraint_exposure_cycles")
        ),
        "async_groups": async_groups,
        "cta_barriers": barriers,
        "timing": {
            "issue": issue,
            "last_event": last_value,
            "terminal": terminal_value,
            "completion_frontier": frontier,
        },
    }


def validate_replay_document(
    document: Mapping[str, Any],
    trace_dir: Path,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate every launch against trace-derived and causal invariants."""

    invariants = derive_trace_invariants(trace_dir)
    expected_by_identity = invariants.by_identity()
    launches = document.get("launches")
    if not isinstance(launches, list):
        return [], [f"{label}: launches is not a list"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, launch in enumerate(launches):
        if not isinstance(launch, Mapping):
            errors.append(f"{label}: launch {index} is not an object")
            continue
        identity = _identity(launch)
        expected = expected_by_identity.get(identity) if identity is not None else None
        if expected is None:
            errors.append(f"{label}: launch {index} identity is absent from trace")
            continue
        prefix = f"{label}: launch {index}"
        counters = launch.get("counters")
        if not isinstance(counters, Mapping):
            errors.append(f"{prefix}: counters are missing")
            continue
        coverage = launch.get("coverage")
        checks: dict[str, bool] = {
            "coverage_full": coverage == {
                "kind": "full",
                "blocks_observed": expected.blocks,
                "population_blocks": expected.blocks,
            },
            "instruction_conservation": _int(counters.get("instructions_issued"))
            == expected.instructions,
            "record_conservation": _int(counters.get("records_seen"))
            == expected.instructions,
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(
                    f"{prefix}: trace-derived {name} failed "
                    f"(expected instructions={expected.instructions}, blocks={expected.blocks})"
                )
        traffic = launch.get("timing", {}).get("readiness", {}).get("global_memory_issue")
        global_traffic_applicable = any(
            value != 0
            for value in (
                expected.global_load_instructions,
                expected.global_store_instructions,
                expected.global_atomic_instructions,
                expected.global_load_sectors,
                expected.global_store_sectors,
                expected.global_atomic_sectors,
            )
        )
        traffic_snapshot: dict[str, Any]
        if isinstance(traffic, Mapping):
            error, traffic_snapshot = _global_memory_error(traffic, expected, prefix)
            checks["trace_global_memory_conservation"] = error is None
            if error is not None:
                errors.append(error)
        elif global_traffic_applicable:
            traffic_snapshot = {
                "status": "missing",
                "applicable": True,
                "expected": _expected_global_memory(expected),
            }
            checks["trace_global_memory_conservation"] = False
            errors.append(
                f"{prefix}: global_memory_issue diagnostic is required for trace-derived global traffic"
            )
        else:
            # Keep the inapplicable case explicit in reports.  Silently
            # omitting this lane made a gutted replay look semantically valid.
            traffic_snapshot = {
                "status": "not_applicable",
                "applicable": False,
                "expected": _expected_global_memory(expected),
            }
            checks["trace_global_memory_conservation"] = True
        async_error, async_snapshot = _async_source_error(counters, expected, prefix)
        traffic_snapshot["async_source"] = async_snapshot
        checks["async_source_conservation"] = async_error is None
        if async_error is not None:
            errors.append(async_error)
        hierarchy_error, hierarchy_snapshot = _hierarchy_error(counters, prefix)
        checks["hierarchy_conservation"] = hierarchy_error is None
        if hierarchy_error is not None:
            errors.append(hierarchy_error)
        readiness_error, readiness_snapshot = _readiness_error(launch, prefix)
        checks["readiness_conservation"] = readiness_error is None
        if readiness_error is not None:
            errors.append(readiness_error)
        rows.append(
            {
                "label": label,
                "launch_index": index,
                "identity": {
                    "device_id": expected.device_id,
                    "stream_id": expected.stream_id,
                    "kernel_id": expected.kernel_id,
                },
                "trace_expected": expected.as_dict(),
                "observed": {
                    "coverage": coverage,
                    "instructions_issued": counters.get("instructions_issued"),
                    "records_seen": counters.get("records_seen"),
                    "l1": {
                        key: _int(counters.get("l1", {}).get(key))
                        for key in ("hits", "misses", "merges", "stall_events")
                    },
                    "l2": {
                        key: _int(counters.get("l2", {}).get(key))
                        for key in ("hits", "misses", "line_misses", "sector_fill_misses")
                    },
                    "dram": {
                        "read_sectors": _int(
                            counters.get("dram", {}).get("read_sectors")
                        )
                    },
                    "unmodeled_tails": counters.get("unmodeled_tails"),
                    "traffic": traffic_snapshot,
                    "hierarchy": hierarchy_snapshot,
                    "readiness": readiness_snapshot,
                },
                "checks": checks,
                "promotion_eligible": all(checks.values())
                and async_snapshot.get("promotion_eligible", True),
                "passed": all(checks.values())
                and async_snapshot.get("promotion_eligible", True),
            }
        )
    return rows, errors
