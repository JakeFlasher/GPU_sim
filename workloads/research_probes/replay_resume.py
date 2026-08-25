"""Shared validation for deterministic direct-simulator campaign replays.

The helpers in this module decide only whether an existing replay result is a
complete result for the current inputs.  They do not confer admission or
publication status.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

_BASE_ROOT_KEYS = {
    "schema_version",
    "simulator_version",
    "architecture",
    "memory_mode",
    "cache_scope",
    "carded_dram_bandwidth",
    "memory_service_candidate_count",
    "memory_admission_candidate_count",
    "launches",
}
_BASE_LAUNCH_KEYS = {
    "device_id",
    "stream_id",
    "kernel_id",
    "function_unique_id",
    "kernel_name",
    "grid",
    "block",
    "cluster",
    "coverage",
    "counters",
    "timing",
}
_LAUNCH_IDENTITY_KEYS = (
    "device_id",
    "stream_id",
    "kernel_id",
    "function_unique_id",
    "kernel_name",
    "grid",
    "block",
    "cluster",
)
_COUNTER_KEYS = {
    "instructions_issued",
    "records_seen",
    "l1",
    "l2",
    "l2_by_source",
    "dram",
    "unmodeled_tails",
}
_L1_COUNTER_KEYS = {
    "hits",
    "misses",
    "merges",
    "stall_events",
    "line_fills",
    "bypass_sectors",
    "store_sectors",
    "reduction_sectors",
}
_L2_COUNTER_KEYS = {
    "hits",
    "misses",
    "line_misses",
    "sector_fill_misses",
    "write_tag_hits",
}
_TAIL_KEYS = {
    "reduction_read_modify_write",
    "dirty_writeback",
    "coherence",
    "async_copy_shared_visibility",
}
_READINESS_REQUIRED_KEYS = {
    "schema_version",
    "limiting_cycles",
    "constraint_exposure_cycles",
    "delayed_ready_issues",
    "tied_limiting_causes",
    "async_groups",
    "cta_barriers",
}
_READINESS_OPTIONAL_KEYS = {
    "barrier_pipeline",
    "control_pipeline",
    "result_writeback",
    "clock_issue",
    "global_memory_issue",
    "memory_service_candidates",
    "memory_admission_candidates",
    "memory_admission_feedback",
}
_CAUSE_REQUIRED_KEYS = {
    "control_stall",
    "dependence_counter_read",
    "dependence_counter_non_memory_write",
    "dependence_counter_shared_write",
    "dependence_counter_l1tex",
    "dependence_counter_unknown_origin",
    "async_source",
    "async_destination_local",
    "async_destination_sm_service",
    "cta_barrier_rendezvous",
    "pipeline_tensor",
    "eligible_not_selected",
}
_CAUSE_OPTIONAL_KEYS = {
    "dependence_counter_global_load_operand_read",
    "dependence_counter_global_store_operand_read",
    "dependence_counter_global_atomic_operand_read",
    "dependence_counter_global_reduction_operand_read",
    "pipeline_barrier",
    "pipeline_branch",
    "memory_admission_queue",
    "memory_write_completion",
}
_ASYNC_KEYS = {
    "groups_committed",
    "groups_waited",
    "source_limited_groups",
    "destination_local_limited_groups",
    "destination_sm_service_limited_groups",
    "source_cycles",
    "destination_local_cycles",
    "destination_sm_service_cycles",
    "completion_hidden_before_wait_cycles",
}
_CTA_KEYS = {
    "generations",
    "total_arrival_spread_cycles",
    "max_arrival_spread_cycles",
    "total_rendezvous_warp_cycles",
    "releases_after_participant_retirement",
}


def _strict_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def load_json_strict(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object_pairs,
        parse_constant=_reject_json_constant,
    )


def candidate_configurations(
    path: Path,
    *,
    require_exactly_one: bool = False,
) -> tuple[Mapping[str, object], ...]:
    document = load_json_strict(path)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "candidates",
    }:
        raise RuntimeError(f"candidate document fields are invalid: {path}")
    if document.get("schema_version") != 1:
        raise RuntimeError(f"candidate document schema_version is not 1: {path}")
    candidates = document.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"candidate document has no candidates: {path}")
    if len(candidates) > 16:
        raise RuntimeError(f"candidate document exceeds 16 candidates: {path}")
    if require_exactly_one and len(candidates) != 1:
        raise RuntimeError(
            f"feedback document must contain exactly one candidate, got {len(candidates)}: {path}"
        )
    if any(not isinstance(candidate, dict) for candidate in candidates):
        raise RuntimeError(f"candidate document contains a non-object candidate: {path}")
    return tuple(candidates)


@cache
def simulator_package_version() -> str:
    with (REPO_ROOT / "sim/Cargo.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    version = manifest.get("workspace", {}).get("package", {}).get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("sim/Cargo.toml has no workspace package version")
    return version


def latest_input_mtime_ns(paths: Sequence[Path]) -> int:
    latest = 0
    for path in paths:
        material = (
            [path]
            if path.is_file()
            else [item for item in path.rglob("*") if item.is_file()]
        )
        if not material:
            raise RuntimeError(f"replay input has no files: {path}")
        latest = max(latest, *(item.stat().st_mtime_ns for item in material))
    return latest


def _numbered_directory(path: Path, prefix: str) -> int:
    matched = re.fullmatch(rf"{re.escape(prefix)}(-?[0-9]+)", path.name)
    if matched is None:
        raise RuntimeError(f"unexpected trace directory name: {path}")
    return int(matched.group(1))


def expected_launches_from_trace(
    trace_dir: Path,
    *,
    grid: Sequence[int],
    block: Sequence[int] | None,
    cluster: Sequence[int] = (1, 1, 1),
) -> tuple[dict[str, object], ...]:
    """Derive exact launch keys for the single-function research captures."""

    exec_info = load_json_strict(
        trace_dir / "extra_info/enhanced_execution_info.json"
    )
    if not isinstance(exec_info, dict) or not isinstance(exec_info.get("kernels"), list):
        raise RuntimeError(f"trace execution metadata is malformed: {trace_dir}")
    kernels = exec_info["kernels"]
    if len(kernels) != 1 or not isinstance(kernels[0], dict):
        raise RuntimeError(
            f"campaign replay identity requires exactly one traced function: {trace_dir}"
        )
    function = kernels[0]
    function_id = function.get("unique_function_id")
    kernel_name = function.get("kernel_name")
    if isinstance(function_id, bool) or not isinstance(function_id, int):
        raise RuntimeError(f"trace function id is malformed: {trace_dir}")
    if not isinstance(kernel_name, str) or not kernel_name:
        raise RuntimeError(f"trace kernel name is malformed: {trace_dir}")

    launch_roots = sorted(
        path
        for path in (trace_dir / "threadblocks").glob(
            "device_*/stream_*/kernel_*"
        )
        if path.is_dir()
    )
    if not launch_roots:
        raise RuntimeError(f"trace has no launch payload directories: {trace_dir}")
    geometry = {
        "grid": list(grid),
        "block": None if block is None else list(block),
        "cluster": list(cluster),
    }
    if any(
        len(value) != 3 or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        for value in (geometry["grid"], geometry["cluster"])
    ) or (
        geometry["block"] is not None
        and (
            len(geometry["block"]) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in geometry["block"]
            )
        )
    ):
        raise RuntimeError("launch geometry must contain three integer dimensions")

    launches = []
    for launch_root in launch_roots:
        stream_root = launch_root.parent
        device_root = stream_root.parent
        launches.append(
            {
                "device_id": _numbered_directory(device_root, "device_"),
                "stream_id": _numbered_directory(stream_root, "stream_"),
                "kernel_id": _numbered_directory(launch_root, "kernel_"),
                "function_unique_id": function_id,
                "kernel_name": kernel_name,
                **geometry,
            }
        )
    launches.sort(
        key=lambda launch: (
            launch["device_id"],
            launch["stream_id"],
            launch["kernel_id"],
        )
    )
    return tuple(launches)


@dataclass(frozen=True, slots=True)
class ReplayExpectation:
    launches: tuple[Mapping[str, object], ...]
    carded_dram: bool
    cache_scope: str = "launch"
    architecture: str = "sm_86"
    memory_mode: str = "l1-l2"
    service_candidates: tuple[Mapping[str, object], ...] = ()
    admission_candidates: tuple[Mapping[str, object], ...] = ()
    feedback_candidate: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.launches:
            raise ValueError("replay expectation must contain at least one launch")
        if self.cache_scope not in {"launch", "stream-l2"}:
            raise ValueError(f"unsupported cache scope {self.cache_scope!r}")
        candidate_modes = sum(
            (
                bool(self.service_candidates),
                bool(self.admission_candidates),
                self.feedback_candidate is not None,
            )
        )
        if candidate_modes > 1:
            raise ValueError("candidate replay expectations are mutually exclusive")
        if candidate_modes and self.cache_scope != "launch":
            raise ValueError("candidate replay expectations require launch cache scope")


def _candidate_configs(readiness: Mapping[str, object], key: str) -> object:
    entries = readiness.get(key)
    if not isinstance(entries, list):
        return None
    configs = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("candidate"), dict):
            return None
        configs.append(entry["candidate"])
    return configs


def _is_u64(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**64


def _u64_object_error(
    value: object,
    required: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> str | None:
    if not isinstance(value, dict):
        return f"{label} is not an object"
    optional = optional or set()
    keys = set(value)
    if not required <= keys or not keys <= required | optional:
        return f"{label} fields differ"
    if any(not _is_u64(item) for item in value.values()):
        return f"{label} contains a malformed counter"
    return None


def _tail_error(value: object, label: str) -> str | None:
    if not isinstance(value, dict) or set(value) != _TAIL_KEYS:
        return f"{label} fields differ"
    if any(type(item) is not bool for item in value.values()):
        return f"{label} contains a non-boolean flag"
    return None


def _l2_error(value: object, label: str) -> str | None:
    return _u64_object_error(value, _L2_COUNTER_KEYS, label)


def _counter_error(value: object, label: str) -> str | None:
    if not isinstance(value, dict) or set(value) != _COUNTER_KEYS:
        return f"{label} fields differ"
    for key in ("instructions_issued", "records_seen"):
        if not _is_u64(value[key]):
            return f"{label}.{key} is malformed"
    error = _u64_object_error(value["l1"], _L1_COUNTER_KEYS, f"{label}.l1")
    if error is not None:
        return error
    error = _l2_error(value["l2"], f"{label}.l2")
    if error is not None:
        return error
    by_source = value["l2_by_source"]
    if not isinstance(by_source, dict) or set(by_source) != {
        "l1_fill",
        "bypass",
        "untagged",
        "store",
    }:
        return f"{label}.l2_by_source fields differ"
    for source, counters in by_source.items():
        error = _u64_object_error(
            counters, {"sectors", "hits"}, f"{label}.l2_by_source.{source}"
        )
        if error is not None:
            return error
    error = _u64_object_error(
        value["dram"],
        {"read_sectors", "bandwidth_delayed_responses"},
        f"{label}.dram",
    )
    if error is not None:
        return error
    return _tail_error(value["unmodeled_tails"], f"{label}.unmodeled_tails")


def _diagnostic_tree_error(value: object, label: str) -> str | None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    if _is_u64(value):
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            error = _diagnostic_tree_error(item, f"{label}[{index}]")
            if error is not None:
                return error
        return None
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not key for key in value):
            return f"{label} contains a malformed key"
        for key, item in value.items():
            error = _diagnostic_tree_error(item, f"{label}.{key}")
            if error is not None:
                return error
        return None
    return f"{label} contains a malformed diagnostic value"


def _readiness_error(value: object, label: str) -> str | None:
    if not isinstance(value, dict):
        return f"{label} is missing"
    keys = set(value)
    if not _READINESS_REQUIRED_KEYS <= keys or not keys <= (
        _READINESS_REQUIRED_KEYS | _READINESS_OPTIONAL_KEYS
    ):
        return f"{label} fields differ"
    schema = value["schema_version"]
    if not _is_u64(schema) or not 1 <= schema <= 13:
        return f"{label} schema is malformed"
    for name in ("delayed_ready_issues", "tied_limiting_causes"):
        if not _is_u64(value[name]):
            return f"{label}.{name} is malformed"
    for name in ("limiting_cycles", "constraint_exposure_cycles"):
        error = _u64_object_error(
            value[name],
            _CAUSE_REQUIRED_KEYS,
            f"{label}.{name}",
            optional=_CAUSE_OPTIONAL_KEYS,
        )
        if error is not None:
            return error
    error = _u64_object_error(value["async_groups"], _ASYNC_KEYS, f"{label}.async_groups")
    if error is not None:
        return error
    error = _u64_object_error(value["cta_barriers"], _CTA_KEYS, f"{label}.cta_barriers")
    if error is not None:
        return error
    for name in _READINESS_OPTIONAL_KEYS & keys:
        error = _diagnostic_tree_error(value[name], f"{label}.{name}")
        if error is not None:
            return error
    return None


def _timing_error(value: object, label: str) -> str | None:
    if not isinstance(value, dict) or set(value) != {
        "diagnostic_last_issue_cycles",
        "last_processed_modeled_system_event_time",
        "terminal_modeled_event_time",
        "completion_frontier",
        "readiness",
    }:
        return f"{label} fields differ from direct schema 1"
    if not _is_u64(value["diagnostic_last_issue_cycles"]):
        return f"{label}.diagnostic_last_issue_cycles is malformed"
    for name in (
        "last_processed_modeled_system_event_time",
        "terminal_modeled_event_time",
    ):
        if value[name] is not None and not _is_u64(value[name]):
            return f"{label}.{name} is malformed"
    frontier = value["completion_frontier"]
    if not isinstance(frontier, dict):
        return f"{label}.completion_frontier is malformed"
    status = frontier.get("status")
    if status == "complete":
        if set(frontier) != {"status", "cycles"} or (
            frontier["cycles"] is not None and not _is_u64(frontier["cycles"])
        ):
            return f"{label}.completion_frontier complete payload is malformed"
    elif status == "incomplete_unmodeled_retirement_tails":
        if set(frontier) != {"status", "lower_bound_cycles", "unmodeled_tails"}:
            return f"{label}.completion_frontier incomplete fields differ"
        if frontier["lower_bound_cycles"] is not None and not _is_u64(
            frontier["lower_bound_cycles"]
        ):
            return f"{label}.completion_frontier lower bound is malformed"
        error = _tail_error(
            frontier["unmodeled_tails"],
            f"{label}.completion_frontier.unmodeled_tails",
        )
        if error is not None:
            return error
    else:
        return f"{label}.completion_frontier status is malformed"
    return _readiness_error(value["readiness"], f"{label}.readiness")


def _report_error(
    document: object,
    expectation: ReplayExpectation,
) -> str | None:
    if not isinstance(document, dict):
        return "result root is not an object"

    root_keys = set(_BASE_ROOT_KEYS)
    if expectation.feedback_candidate is not None:
        root_keys.add("memory_admission_feedback_candidate_id")
    if expectation.cache_scope == "stream-l2":
        root_keys.add("cache_session")
    if set(document) != root_keys:
        return "result root fields differ from direct schema 1"
    if document.get("schema_version") != 1:
        return "result schema_version is not 1"
    if document.get("simulator_version") != simulator_package_version():
        return "result simulator_version differs from the current package"
    if document.get("architecture") != expectation.architecture:
        return "result architecture differs"
    if document.get("memory_mode") != expectation.memory_mode:
        return "result memory mode differs"
    if document.get("cache_scope") != expectation.cache_scope:
        return "result cache scope differs"
    if document.get("carded_dram_bandwidth") is not expectation.carded_dram:
        return "result carded-DRAM state differs"
    if document.get("memory_service_candidate_count") != len(
        expectation.service_candidates
    ):
        return "result memory-service candidate count differs"
    if document.get("memory_admission_candidate_count") != len(
        expectation.admission_candidates
    ):
        return "result memory-admission candidate count differs"
    if expectation.feedback_candidate is not None and document.get(
        "memory_admission_feedback_candidate_id"
    ) != expectation.feedback_candidate.get("candidate_id"):
        return "result feedback candidate id differs"

    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != len(expectation.launches):
        return "result launch count differs"
    prior_kernel_ids: list[int] = []
    for index, (launch, expected) in enumerate(zip(launches, expectation.launches)):
        if not isinstance(launch, dict):
            return f"launch {index} is not an object"
        launch_keys = set(_BASE_LAUNCH_KEYS)
        if expectation.cache_scope == "stream-l2":
            launch_keys.add("cache_handoff")
        if set(launch) != launch_keys:
            return f"launch {index} fields differ from direct schema 1"
        actual_identity = {key: launch.get(key) for key in _LAUNCH_IDENTITY_KEYS}
        expected_identity = {key: expected.get(key) for key in _LAUNCH_IDENTITY_KEYS}
        if actual_identity != expected_identity:
            return f"launch {index} identity differs from the trace"

        coverage = launch.get("coverage")
        grid = expected_identity["grid"]
        expected_blocks = (
            grid[0] * grid[1] * grid[2]
            if isinstance(grid, list) and len(grid) == 3
            else None
        )
        if coverage != {
            "kind": "full",
            "blocks_observed": expected_blocks,
            "population_blocks": expected_blocks,
        }:
            return f"launch {index} does not report exact full coverage"
        error = _counter_error(launch.get("counters"), f"launch {index} counters")
        if error is not None:
            return error
        timing = launch.get("timing")
        error = _timing_error(timing, f"launch {index} timing")
        if error is not None:
            return error
        assert isinstance(timing, dict)
        readiness = timing.get("readiness")
        assert isinstance(readiness, dict)
        readiness_schema = readiness.get("schema_version")
        assert isinstance(readiness_schema, int) and not isinstance(readiness_schema, bool)

        candidate_keys = {
            "memory_service_candidates",
            "memory_admission_candidates",
            "memory_admission_feedback",
        }
        if expectation.service_candidates:
            if readiness_schema not in {7, 8}:
                return f"launch {index} memory-service readiness schema differs"
            actual = _candidate_configs(readiness, "memory_service_candidates")
            if actual != list(expectation.service_candidates):
                return f"launch {index} memory-service candidate configuration differs"
            candidate_keys.remove("memory_service_candidates")
        elif expectation.admission_candidates:
            if readiness_schema != 9:
                return f"launch {index} memory-admission readiness schema differs"
            actual = _candidate_configs(readiness, "memory_admission_candidates")
            if actual != list(expectation.admission_candidates):
                return f"launch {index} memory-admission candidate configuration differs"
            candidate_keys.remove("memory_admission_candidates")
        elif expectation.feedback_candidate is not None:
            expected_feedback_schema = (
                13
                if expectation.feedback_candidate.get(
                    "write_completion_request_cycles", 0
                )
                != 0
                or expectation.feedback_candidate.get(
                    "write_completion_region_cycles", 0
                )
                != 0
                else 12
                if expectation.feedback_candidate.get(
                    "write_completion_latency_cycles", 0
                )
                != 0
                else 11
                if expectation.feedback_candidate.get(
                    "membar_waits_for_service_completion"
                )
                is True
                else 10
            )
            if readiness_schema != expected_feedback_schema:
                return f"launch {index} feedback readiness schema differs"
            feedback = readiness.get("memory_admission_feedback")
            if not isinstance(feedback, dict) or feedback.get("candidate") != dict(
                expectation.feedback_candidate
            ):
                return f"launch {index} feedback candidate configuration differs"
            candidate_keys.remove("memory_admission_feedback")
        elif not 1 <= readiness_schema <= 8:
            return f"launch {index} ordinary readiness schema is outside 1..=8"
        if any(key in readiness for key in candidate_keys):
            return f"launch {index} contains an unexpected candidate mode"

        if expectation.cache_scope == "stream-l2":
            handoff = launch.get("cache_handoff")
            if handoff != {
                "scope": "stream-l2",
                "sequence_index": index,
                "seeded_by_kernel_ids": prior_kernel_ids,
            }:
                return f"launch {index} cache handoff identity differs"
        kernel_id = launch.get("kernel_id")
        if isinstance(kernel_id, int) and not isinstance(kernel_id, bool):
            prior_kernel_ids.append(kernel_id)

    if expectation.cache_scope == "stream-l2":
        first = expectation.launches[0]
        session = document.get("cache_session")
        if not isinstance(session, dict) or set(session) != {
            "scope",
            "device_id",
            "stream_id",
            "completed_launches",
            "cumulative_l2",
        }:
            return "cache-session fields differ from direct schema 1"
        if (
            session.get("scope") != "stream-l2"
            or session.get("device_id") != first.get("device_id")
            or session.get("stream_id") != first.get("stream_id")
            or session.get("completed_launches") != len(expectation.launches)
            or _l2_error(session.get("cumulative_l2"), "cache-session cumulative_l2")
            is not None
        ):
            return "cache-session identity differs"
    return None


def simulator_output_reuse_error(
    path: Path,
    dependencies: Sequence[Path],
    expectation: ReplayExpectation,
) -> str | None:
    if not path.is_file() or path.stat().st_size == 0:
        return "result file is missing or empty"
    try:
        latest_dependency = latest_input_mtime_ns(dependencies)
    except (OSError, RuntimeError) as error:
        return str(error)
    if path.stat().st_mtime_ns <= latest_dependency:
        return "result does not strictly postdate every replay input"
    try:
        document = load_json_strict(path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        return f"result is not strict JSON: {error}"
    return _report_error(document, expectation)


def reusable_simulator_output(
    path: Path,
    dependencies: Sequence[Path],
    expectation: ReplayExpectation,
) -> bool:
    return simulator_output_reuse_error(path, dependencies, expectation) is None


def replay_pair_error(
    first: Path,
    second: Path,
    dependencies: Sequence[Path],
    expectation: ReplayExpectation,
) -> str | None:
    for path in (first, second):
        error = simulator_output_reuse_error(path, dependencies, expectation)
        if error is not None:
            return f"{path}: {error}"
    try:
        if first.read_bytes() != second.read_bytes():
            return "replay pair is not byte-identical"
    except OSError as error:
        return f"could not read replay pair: {error}"
    return None


def load_replay_pair(
    first: Path,
    second: Path,
    dependencies: Sequence[Path],
    expectation: ReplayExpectation,
) -> Mapping[str, object]:
    """Validate a deterministic pair against current inputs, then load it."""

    error = replay_pair_error(first, second, dependencies, expectation)
    if error is not None:
        raise RuntimeError(error)
    document = load_json_strict(first)
    if not isinstance(document, dict):
        raise RuntimeError(f"{first}: validated replay root is not an object")
    return document
