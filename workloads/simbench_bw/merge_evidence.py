#!/usr/bin/env python3
"""Historical CUPTI/NVBit two-run evidence replay join.

This tool joins frozen sidecars from the retired counter-free workflow. That
workflow ran the same kernel twice because CUPTI Activity and NVBit injection
were intentionally kept in separate producer processes. This module performs
offline replay only and never invokes either producer. It joins the archived
runs by the canonical launch key:

    (slug, shape_point, repeat_index, launch_index, kernel_signature_sha256)

It validates both historical sidecars, checks binary identity, rejects
unmatched or duplicated launches, and writes joined records only when
the entire input set is conflict-free.

Usage:
    python3 merge_evidence.py join-cupti-nvbit \
        --cupti-dir <dir> --nvbit-dir <dir> --output-dir <dir> \
        --platform {h100|rtx_pro_6000_blackwell|rtx_5060_laptop|rtx_3090_ti}

Exit codes:
    0  every CUPTI/NVBit launch joined successfully.
    1  conflict requires manual resolution; no joined files were written.
    2  CLI usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any


_MODULE_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_MODULE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_REPO_ROOT))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ImmutableFile,
    PinnedDirectoryReader,
    PinnedPathSetMonitor,
    inspect_immutable_directory_generation,
    open_retained_regular,
    publish_directory,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    DEFAULT_TRACE_ADMISSION_LIMITS,
    AdmittedTraceSources,
    TraceAdmission,
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)
from workloads.simbench_bw.nvbit_trace_summary_adapter import (  # noqa: E402
    compute_kernel_signature_sha256,
)
from workloads.simbench_bw.phase_cycle_ledger_aggregator import (  # noqa: E402
    PhaseLedgerSchemaInvalid,
    serialize_phase_cycle_ledger,
)


SUITE_ROOT = Path(__file__).resolve().parent
PLATFORM_KEYS: frozenset[str] = frozenset({
    "h100",
    "rtx_pro_6000_blackwell",
    "rtx_5060_laptop",
    "rtx_3090_ti",
})


# ---------------------------------------------------------------------------
# join-cupti-nvbit subcommand: NVBit + CUPTI two-run offline join.
# ---------------------------------------------------------------------------

JOINED_EVIDENCE_SCHEMA_VERSION: int = 1

# Canonical 5-tuple join key
# (`(slug, shape_point, repeat_index, launch_index, kernel_signature_sha256)`).
# The first 4 components form the "natural key" (the same kernel launch
# in two identical runs); the 5th component is the load-bearing
# integrity check — if the same natural key appears on both sides with
# different signature hashes, the two runs DID NOT produce bit-
# identical kernel shapes and the join is invalid.
_JOIN_NATURAL_KEY_FIELDS: tuple[str, ...] = (
    "slug", "shape_point", "repeat_index", "launch_index",
)
_JOIN_INTEGRITY_KEY_FIELD: str = "kernel_signature_sha256"

_CUPTI_SIDECAR_SCHEMA_PATH = (
    SUITE_ROOT
    / "schemas"
    / "historical"
    / "cupti_activity_sidecar.v1.schema.json"
)
_NVBIT_TRACE_SUMMARY_SCHEMA_PATH = SUITE_ROOT / "schemas" / "nvbit_trace_summary.schema.json"
_NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE = (
    "nvbit_trace_summary_generations"
)
_RAW_TRACE_GENERATION_NAMESPACE = "raw_trace_generations"
_PHASE_LEDGER_GENERATION_NAMESPACE = "phase_cycle_ledger_generations"
_TRACE_CAPTURE_ATTEMPT_NAMESPACE = "trace_capture_attempts"
_PHASE_LEDGER_FILENAME_RE = re.compile(
    r"^phase_cycle_ledger_([0-9]+)_([0-9]+)\.json$"
)
_NVBIT_TRACE_SUMMARY_FILENAME_RE = re.compile(
    r"^nvbit_trace_summary_([0-9]+)_([0-9]+)\.json$"
)
_KNOWN_NONAUTHORITY_TRACE_FILENAMES = frozenset({
    "trace_volume_exceeded.json",
})
_JOIN_DISCOVERY_MAX_FILES = 250_000
_JOIN_DISCOVERY_MAX_DIRECTORIES = 100_000
_JOIN_DISCOVERY_MAX_TOTAL_FILE_BYTES = 4 * 1024**4
_JOIN_DISCOVERY_MAX_RELATIVE_PATH_BYTES = 64 * 1024 * 1024


def _load_join_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the historical CUPTI and retained NVBit replay schemas."""
    with open(_CUPTI_SIDECAR_SCHEMA_PATH) as f:
        cupti_schema = json.load(f)
    with open(_NVBIT_TRACE_SUMMARY_SCHEMA_PATH) as f:
        nvbit_schema = json.load(f)
    return cupti_schema, nvbit_schema


def _allowed_platforms_for_join() -> frozenset[str]:
    """Canonical platform set for joined output paths."""
    return PLATFORM_KEYS


def _snapshot_join_input_tree(
    root: Path,
    *,
    reader: PinnedDirectoryReader | None = None,
):
    """Inventory one join input root through a bounded no-follow traversal."""
    if reader is not None:
        return reader.snapshot_tree(
            max_files=_JOIN_DISCOVERY_MAX_FILES,
            max_directories=_JOIN_DISCOVERY_MAX_DIRECTORIES,
            max_total_file_bytes=_JOIN_DISCOVERY_MAX_TOTAL_FILE_BYTES,
            max_total_relative_path_bytes=(
                _JOIN_DISCOVERY_MAX_RELATIVE_PATH_BYTES
            ),
        )
    with PinnedDirectoryReader(root) as owned_reader:
        return _snapshot_join_input_tree(root, reader=owned_reader)


def _path_from_snapshot(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    return Path(root).joinpath(*relative.parts)


def _retain_join_input_tree(
    root: Path,
    *,
    source_stack: ExitStack,
    retained_authorities: list[Any],
) -> tuple[PinnedDirectoryReader, Any]:
    """Pin, inventory, and continuously monitor one complete join input tree."""
    reader = source_stack.enter_context(PinnedDirectoryReader(root))
    monitor = source_stack.enter_context(
        PinnedPathSetMonitor.from_reader(
            reader,
            regular_paths=(),
            recursive_directories=(".",),
            max_files=_JOIN_DISCOVERY_MAX_FILES,
            max_directories=_JOIN_DISCOVERY_MAX_DIRECTORIES,
            max_total_file_bytes=_JOIN_DISCOVERY_MAX_TOTAL_FILE_BYTES,
            max_total_relative_path_bytes=(
                _JOIN_DISCOVERY_MAX_RELATIVE_PATH_BYTES
            ),
        )
    )
    retained_authorities.append(monitor)
    snapshot = _snapshot_join_input_tree(root, reader=reader)
    monitor.verify_unchanged()
    return reader, snapshot


def _is_hidden_or_reserved_join_input(root: Path, candidate: Path) -> bool:
    """Exclude staging and evidence namespaces from legacy/activation discovery."""
    try:
        parent_parts = candidate.relative_to(root).parts[:-1]
    except ValueError:
        return True
    reserved = {
        _NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE,
        _RAW_TRACE_GENERATION_NAMESPACE,
        _PHASE_LEDGER_GENERATION_NAMESPACE,
        _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
    }
    return any(
        part.startswith(".") or part in reserved for part in parent_parts
    )


def _discover_json_files(
    root: Path,
    glob_pattern: str,
    *,
    tree_snapshot=None,
) -> list[Path]:
    """Discover matching regular files from one bounded no-follow inventory."""
    snapshot = (
        _snapshot_join_input_tree(root)
        if tree_snapshot is None
        else tree_snapshot
    )
    return sorted(
        _path_from_snapshot(root, relative_path)
        for relative_path, _metadata in snapshot.files
        if PurePosixPath(relative_path).match(glob_pattern)
    )


def _preload_join_input_bytes(
    root: Path,
    reader: PinnedDirectoryReader,
    tree_snapshot: Any,
    paths: list[Path],
) -> dict[Path, bytes]:
    """Read discovered bytes through the same pinned root and inventory."""
    inventory = dict(tree_snapshot.files)
    preloaded: dict[Path, bytes] = {}
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ImmutableArtifactError(
                f"discovered input path escapes pinned root: {path}"
            ) from exc
        metadata = inventory.get(relative)
        if metadata is None:
            raise ImmutableArtifactError(
                f"discovered input is absent from pinned inventory: {relative}"
            )
        preloaded[path] = reader.read_snapshot(
            relative,
            max_bytes=metadata[4],
            expected_metadata=metadata,
        ).payload
    reader.verify_path()
    return preloaded


def _admitted_launch_indices(
    admission: TraceAdmission,
) -> tuple[int, ...] | None:
    """Return the exact contiguous launch authority admitted from raw bytes."""
    if type(admission.launch_count) is not int or admission.launch_count <= 0:
        return None
    indices = tuple(launch.launch_index for launch in admission.launches)
    if any(type(index) is not int for index in indices):
        return None
    if indices != tuple(range(admission.launch_count)):
        return None
    return indices


def _is_active_nvbit_trace_record(
    path: Path, record: object,
) -> bool:
    """Recognize active or legacy typed-unavailable per-shape authority."""
    if type(record) is not dict:
        return False
    shape_point = record.get("shape_point")
    if type(shape_point) is not str or not shape_point:
        return False
    if path.name != f"trace_{shape_point}.json":
        return False
    activation_state = record.get("activation_state")
    if activation_state != "active" and not (
        "activation_state" not in record
        and record.get("result_kind") == "typed_unavailable"
    ):
        return False
    return (
        record.get("shape_set") == "trace_shapes"
        or "trace_pb_path" in record
        or "nvbit_trace_summary_paths" in record
        or "nvbit_trace_summary_emit_status" in record
    )


def _namespace_entries(
    root: Path,
    namespace: str,
    *,
    tree_snapshot,
) -> list[Path]:
    """Find ancestor, direct, and nested namespaces in a bounded inventory."""
    entries: set[Path] = set()
    root_absolute = Path(root).absolute()
    for candidate in (root_absolute, *root_absolute.parents):
        if candidate.name == namespace:
            entries.add(candidate)
            break
    for relative_path, _metadata in tree_snapshot.directories:
        if relative_path == ".":
            continue
        relative = PurePosixPath(relative_path)
        if relative.name == namespace:
            entries.add(root_absolute.joinpath(*relative.parts))
    return sorted(entries)


def _generation_namespace_entries(root: Path, *, tree_snapshot) -> list[Path]:
    return _namespace_entries(
        root,
        _NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE,
        tree_snapshot=tree_snapshot,
    )


def _active_summary_binding_error(
    summary: object,
    *,
    trace_record: dict[str, Any],
    trace_admission: TraceAdmission,
    filename: str,
) -> str | None:
    """Bind one committed summary body to its admitted raw generation."""
    if type(summary) is not dict:
        return "referenced summary JSON is not an object"
    if summary.get("schema_version") != 2:
        return "active generated summary must use admitted schema_version 2"
    match = _NVBIT_TRACE_SUMMARY_FILENAME_RE.fullmatch(filename)
    if match is None:
        return "referenced summary filename is noncanonical"
    repeat_index, launch_index = (int(value) for value in match.groups())
    if (
        repeat_index != 0
        or type(summary.get("repeat_index")) is not int
        or summary.get("repeat_index") != repeat_index
    ):
        return "referenced summary repeat_index disagrees with active capture"
    if (
        type(summary.get("launch_index")) is not int
        or summary.get("launch_index") != launch_index
    ):
        return "referenced summary launch_index disagrees with its filename"
    if not 0 <= launch_index < trace_admission.launch_count:
        return "referenced summary launch_index is outside admitted launches"

    launch = trace_admission.launches[launch_index]
    expected = {
        "slug": trace_record.get("slug"),
        "shape_point": trace_record.get("shape_point"),
        "kernel_name": launch.kernel_name,
        "kernel_signature_sha256": compute_kernel_signature_sha256(
            launch.kernel_signature_inputs()
        ),
        "inst_count": launch.inst_count,
        "binary_sha256": trace_record.get("binary_sha256"),
        "nvbit_trace_pb_path": trace_admission.dynamic_trace_path,
        "nvbit_trace_pb_sha256": trace_admission.dynamic_trace_sha256,
        "nvbit_tracer_build_hash": trace_record.get(
            "nvbit_tracer_build_hash"
        ),
        "nvbit_threadblock_payload_size_bytes": (
            trace_admission.threadblock_payload_size_bytes
        ),
        "nvbit_threadblock_payload_manifest_sha256": (
            trace_admission.threadblock_payload_manifest_sha256
        ),
        "trace_admission_sha256": trace_admission.trace_admission_sha256,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            return (
                f"referenced summary field {field!r} does not match its "
                "admitted raw generation and active trace record"
            )

    raw_generation = trace_record.get("raw_trace_generation")
    if type(raw_generation) is not dict:
        return "active trace record has no raw generation descriptor"
    raw_generation_path = raw_generation.get("path")
    if type(raw_generation_path) is not str:
        return "active trace record raw generation path is invalid"
    record_trace_path = (
        PurePosixPath(raw_generation_path) / "dynamic_trace.pb"
    ).as_posix()
    record_bindings = {
        "trace_pb_path": record_trace_path,
        "dynamic_trace_sha256": trace_admission.dynamic_trace_sha256,
        "threadblock_payload_size_bytes": (
            trace_admission.threadblock_payload_size_bytes
        ),
        "threadblock_payload_manifest_sha256": (
            trace_admission.threadblock_payload_manifest_sha256
        ),
        "trace_admission_sha256": trace_admission.trace_admission_sha256,
    }
    for field, expected_value in record_bindings.items():
        if trace_record.get(field) != expected_value:
            return (
                f"active trace record field {field!r} disagrees with its "
                "admitted raw generation"
            )
    return None


def _active_raw_generation_binding_error(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    raw_namespace_entries: list[Path],
) -> str | None:
    """Authenticate the exact raw generation named by one active record."""
    descriptor = trace_record.get("raw_trace_generation")
    if descriptor is None:
        if raw_namespace_entries:
            return (
                "raw generation authority exists but the active trace record "
                "has no raw_trace_generation reference"
            )
        # Pre-Task236 records remain readable only in trees with no raw
        # generation authority at all.
        return None
    if type(descriptor) is not dict:
        return "raw_trace_generation must be an object"

    shape_point = trace_record.get("shape_point")
    attempt_id = trace_record.get("trace_capture_attempt_id")
    relative_value = descriptor.get("path")
    if type(relative_value) is not str:
        return "raw_trace_generation.path must be a string"
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(relative.parts) != 3
        or relative.parts[0] != _RAW_TRACE_GENERATION_NAMESPACE
        or relative.parts[1] != shape_point
        or relative.parts[2] != attempt_id
        or descriptor.get("attempt_id") != attempt_id
    ):
        return "raw_trace_generation.path is not the exact active attempt path"

    seals = (
        "content_manifest_sha256",
        "filesystem_generation_sha256",
        "trace_admission_sha256",
        "trace_filesystem_generation_sha256",
        "dynamic_trace_sha256",
    )
    for field in seals:
        value = descriptor.get(field)
        if type(value) is not str or not _SHA256_RE.match(value):
            return f"raw_trace_generation.{field} must be lowercase 64-hex"
    admission_limits = DEFAULT_TRACE_ADMISSION_LIMITS
    raw_byte_limit = (
        admission_limits.trace_root_bytes
        + admission_limits.allocations_sidecar_bytes
        + admission_limits.stats_bytes
        + admission_limits.total_threadblock_bytes
    )
    bounds = {
        "file_count": admission_limits.tree_files,
        "directory_count": admission_limits.tree_directories,
        "total_file_bytes": raw_byte_limit,
    }
    for field, maximum in bounds.items():
        value = descriptor.get(field)
        minimum = 1 if field != "total_file_bytes" else 0
        if type(value) is not int or not minimum <= value <= maximum:
            return (
                f"raw_trace_generation.{field} is invalid or exceeds its "
                "hard admission bound"
            )

    if (
        trace_record.get("trace_admission_sha256")
        != descriptor["trace_admission_sha256"]
        or trace_record.get("trace_filesystem_generation_sha256")
        != descriptor["trace_filesystem_generation_sha256"]
        or trace_record.get("dynamic_trace_sha256")
        != descriptor["dynamic_trace_sha256"]
    ):
        return "top-level raw trace seals disagree with raw_trace_generation"
    expected_trace_pb_path = (
        relative / "dynamic_trace.pb"
    ).as_posix()
    if trace_record.get("trace_pb_path") != expected_trace_pb_path:
        return "trace_pb_path does not name the exact raw generation"

    generation_path = trace_record_path.parent.joinpath(*relative.parts)
    try:
        observed = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=True,
            max_files=descriptor["file_count"],
            max_directories=descriptor["directory_count"],
            max_total_file_bytes=descriptor["total_file_bytes"],
            max_total_relative_path_bytes=(
                admission_limits.tree_relative_path_bytes
            ),
        )
    except ImmutableArtifactResourceError as exc:
        return (
            "trace_snapshot_resource_unavailable: referenced raw generation "
            f"verification failed: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return (
            "referenced raw generation is absent or invalid: "
            f"{type(exc).__name__}: {exc}"
        )
    if (
        observed.content_manifest_sha256
        != descriptor["content_manifest_sha256"]
        or observed.file_count != descriptor["file_count"]
        or observed.directory_count != descriptor["directory_count"]
        or observed.total_file_bytes != descriptor["total_file_bytes"]
    ):
        return "referenced raw generation content descriptor is stale"
    if (
        observed.filesystem_generation_sha256
        != descriptor["filesystem_generation_sha256"]
    ):
        return "referenced raw generation filesystem seal is stale"
    try:
        admitted = require_trace_admission(generation_path)
    except TraceAdmissionError as exc:
        if exc.typed_reason == "trace_snapshot_resource_unavailable":
            return (
                "trace_snapshot_resource_unavailable: referenced raw "
                f"generation admission failed: {exc.detail}"
            )
        return (
            "referenced raw generation is not admissible: "
            f"{exc.typed_reason}: {exc.detail}"
        )
    except ImmutableArtifactResourceError as exc:
        return (
            "trace_snapshot_resource_unavailable: referenced raw generation "
            f"admission failed: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return (
            "referenced raw generation admission failed: "
            f"{type(exc).__name__}: {exc}"
        )
    if (
        admitted.trace_admission_sha256
        != descriptor["trace_admission_sha256"]
        or admitted.trace_filesystem_generation_sha256
        != descriptor["trace_filesystem_generation_sha256"]
        or admitted.dynamic_trace_sha256
        != descriptor["dynamic_trace_sha256"]
    ):
        return "referenced raw generation admission seals are stale"
    dynamic = next(
        (
            identity
            for identity in observed.files
            if identity.relative_path == "dynamic_trace.pb"
        ),
        None,
    )
    if dynamic is None or dynamic.sha256 != descriptor["dynamic_trace_sha256"]:
        return "referenced raw generation dynamic_trace.pb seal is stale"
    return None


def _reload_active_trace_admission(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    source_stack: ExitStack,
) -> tuple[AdmittedTraceSources | None, str | None]:
    """Retain the exact admitted raw generation through downstream use."""
    descriptor = trace_record.get("raw_trace_generation")
    if type(descriptor) is not dict or type(descriptor.get("path")) is not str:
        return None, "active summaries require an admitted raw generation"
    relative = PurePosixPath(descriptor["path"])
    generation_path = trace_record_path.parent.joinpath(*relative.parts)
    try:
        expected_admission = require_trace_admission(generation_path)
        sources = source_stack.enter_context(
            open_admitted_trace_sources(
                generation_path,
                expected_admission=expected_admission,
            )
        )
    except TraceAdmissionError as exc:
        if exc.typed_reason == "trace_snapshot_resource_unavailable":
            return None, (
                "trace_snapshot_resource_unavailable: active raw generation "
                f"reload failed: {exc.detail}"
            )
        return None, (
            "active raw generation changed before summary binding: "
            f"{exc.typed_reason}: {exc.detail}"
        )
    except ImmutableArtifactResourceError as exc:
        return None, (
            "trace_snapshot_resource_unavailable: active raw generation "
            f"reload failed: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return None, (
            "active raw generation changed before summary binding: "
            f"{type(exc).__name__}: {exc}"
        )
    admission = sources.admission
    if (
        admission.trace_admission_sha256
        != trace_record.get("trace_admission_sha256")
        or admission.trace_filesystem_generation_sha256
        != trace_record.get("trace_filesystem_generation_sha256")
    ):
        return None, "active raw generation seals changed before summary binding"
    return sources, None


def _active_phase_generation_binding_error(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    phase_namespace_entries: list[Path],
    trace_admission: TraceAdmission,
) -> str | None:
    """Authenticate only the phase generation explicitly named by authority."""
    phase = trace_record.get("phase_inject_ab")
    if phase is None:
        if phase_namespace_entries:
            return (
                "phase generation authority exists but the active trace record "
                "has no phase_inject_ab reference"
            )
        return None
    if type(phase) is not dict or type(phase.get("requested")) is not bool:
        return "phase_inject_ab must carry a boolean requested field"
    requested = phase["requested"]
    repeat_count = phase.get("repeat_count")
    descriptor = phase.get("generation")
    evidence = phase.get("evidence")
    if not requested:
        if repeat_count is not None and (
            type(repeat_count) is not int or repeat_count != 0
        ):
            return "unrequested phase evidence has an invalid repeat_count"
        if descriptor is not None:
            return "unrequested phase evidence must not carry generation authority"
        if evidence is not None and (
            type(evidence) is not list or evidence
        ):
            return "unrequested phase evidence must not carry generation authority"
        return None
    if type(repeat_count) is not int or repeat_count <= 0:
        return "requested phase evidence has no positive repeat_count"
    admitted_launch_indices = _admitted_launch_indices(trace_admission)
    if admitted_launch_indices is None:
        return "admitted raw launch authority is malformed"
    if type(descriptor) is not dict:
        return "requested phase evidence has no generation descriptor"
    if type(evidence) is not list or not evidence:
        return "requested phase evidence has no nonempty evidence list"

    shape_point = trace_record.get("shape_point")
    attempt_id = trace_record.get("trace_capture_attempt_id")
    relative_value = descriptor.get("path")
    if type(relative_value) is not str:
        return "phase generation path must be a string"
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(relative.parts) != 3
        or relative.parts[0] != _PHASE_LEDGER_GENERATION_NAMESPACE
        or relative.parts[1] != shape_point
        or relative.parts[2] != attempt_id
        or descriptor.get("attempt_id") != attempt_id
    ):
        return "phase generation path is not the exact active attempt path"
    for field in ("content_manifest_sha256", "filesystem_generation_sha256"):
        value = descriptor.get(field)
        if type(value) is not str or not _SHA256_RE.match(value):
            return f"phase generation {field} must be lowercase 64-hex"
    counts = {
        "file_count": 4096,
        "directory_count": 1,
        "total_file_bytes": 256 * 1024 * 1024,
    }
    for field, maximum in counts.items():
        value = descriptor.get(field)
        minimum = 1 if field != "total_file_bytes" else 0
        if type(value) is not int or not minimum <= value <= maximum:
            return f"phase generation {field} is invalid"

    generation_path = trace_record_path.parent.joinpath(*relative.parts)
    try:
        observed = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=True,
            max_files=descriptor["file_count"],
            max_directories=descriptor["directory_count"],
            max_total_file_bytes=descriptor["total_file_bytes"],
            max_total_relative_path_bytes=4 * 1024 * 1024,
        )
    except ImmutableArtifactResourceError as exc:
        return (
            "trace_snapshot_resource_unavailable: referenced phase generation "
            f"verification failed: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return (
            "referenced phase generation is absent or invalid: "
            f"{type(exc).__name__}: {exc}"
        )
    if (
        observed.content_manifest_sha256
        != descriptor["content_manifest_sha256"]
        or observed.filesystem_generation_sha256
        != descriptor["filesystem_generation_sha256"]
        or observed.file_count != descriptor["file_count"]
        or observed.directory_count != descriptor["directory_count"]
        or observed.total_file_bytes != descriptor["total_file_bytes"]
    ):
        return "referenced phase generation descriptor is stale"

    expected_file_count = repeat_count * len(admitted_launch_indices)
    if observed.file_count != expected_file_count:
        return (
            "referenced phase generation file count does not equal the "
            "admitted repeat-by-launch authority"
        )

    expected_evidence: list[dict[str, Any]] = []
    observed_pairs: set[tuple[int, int]] = set()
    identities = {identity.relative_path: identity for identity in observed.files}
    try:
        with PinnedDirectoryReader(generation_path) as reader:
            snapshot = reader.snapshot_tree(
                require_readonly_files=True,
                require_readonly_directories=True,
                max_files=descriptor["file_count"],
                max_directories=descriptor["directory_count"],
                max_total_file_bytes=descriptor["total_file_bytes"],
                max_total_relative_path_bytes=4 * 1024 * 1024,
            )
            files = dict(snapshot.files)
            directories = dict(snapshot.directories)
            if set(files) != set(identities) or set(directories) != {"."}:
                return "referenced phase generation inventory changed before read"
            for relative_path, identity in sorted(identities.items()):
                match = _PHASE_LEDGER_FILENAME_RE.fullmatch(relative_path)
                if match is None:
                    return (
                        "referenced phase generation contains a noncanonical file: "
                        f"{relative_path}"
                    )
                repeat_text, launch_text = match.groups()
                repeat_index = int(repeat_text)
                launch_index = int(launch_text)
                if (
                    str(repeat_index) != repeat_text
                    or str(launch_index) != launch_text
                ):
                    return (
                        "referenced phase generation contains a noncanonical "
                        f"numeric filename: {relative_path}"
                    )
                payload = reader.read_snapshot(
                    relative_path,
                    expected_metadata=files[relative_path],
                ).payload
                if (
                    len(payload) != identity.size_bytes
                    or hashlib.sha256(payload).hexdigest() != identity.sha256
                ):
                    return (
                        "referenced phase generation payload seal changed: "
                        f"{relative_path}"
                    )
                try:
                    ledger = json.loads(payload)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    return (
                        "referenced phase ledger is invalid JSON: "
                        f"{relative_path}: {exc}"
                    )
                if type(ledger) is not dict:
                    return f"referenced phase ledger is not an object: {relative_path}"
                try:
                    serialize_phase_cycle_ledger(ledger)
                except (PhaseLedgerSchemaInvalid, TypeError, ValueError) as exc:
                    return (
                        "referenced phase ledger fails schema validation: "
                        f"{relative_path}: {exc}"
                    )
                if launch_index not in admitted_launch_indices:
                    return (
                        "referenced phase ledger names an unadmitted launch: "
                        f"{relative_path}"
                    )
                if (
                    ledger.get("slug") != trace_record.get("slug")
                    or ledger.get("shape_point") != trace_record.get("shape_point")
                    or type(ledger.get("repeat_index")) is not int
                    or ledger.get("repeat_index") != repeat_index
                    or type(ledger.get("launch_index")) is not int
                    or ledger.get("launch_index") != launch_index
                    or ledger.get("kernel_signature_sha256")
                    != trace_record.get("kernel_signature_sha256")
                    or ledger.get("binary_sha256")
                    != trace_record.get("binary_sha256")
                    or ledger.get("nvbit_tracer_build_hash")
                    != trace_record.get("nvbit_tracer_build_hash")
                ):
                    return (
                        "referenced phase ledger body disagrees with admitted "
                        "raw launch, execution authority, or filename: "
                        f"{relative_path}"
                    )
                observed_pairs.add((repeat_index, launch_index))
                expected_evidence.append({
                    "path": (relative / relative_path).as_posix(),
                    "sha256": identity.sha256,
                    "size_bytes": identity.size_bytes,
                })
            reader.verify_path()
    except ImmutableArtifactResourceError as exc:
        return (
            "trace_snapshot_resource_unavailable: referenced phase generation "
            f"read failed: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return (
            "referenced phase generation changed during authenticated read: "
            f"{type(exc).__name__}: {exc}"
        )

    expected_pairs = {
        (repeat_index, launch_index)
        for repeat_index in range(repeat_count)
        for launch_index in admitted_launch_indices
    }
    if observed_pairs != expected_pairs:
        return (
            "referenced phase generation does not equal the admitted "
            "repeat-by-launch authority"
        )
    if evidence != expected_evidence:
        return "referenced phase evidence list is stale or incomplete"
    return None


def _active_summary_generation_descriptor_error(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    trace_admission: TraceAdmission,
) -> str | None:
    descriptor = trace_record.get("nvbit_trace_summary_generation")
    if type(descriptor) is not dict:
        return "active trace record has no summary generation descriptor"
    expected_relative = PurePosixPath(
        _NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE,
        str(trace_record.get("shape_point")),
        str(trace_record.get("trace_admission_sha256")),
        str(trace_record.get("trace_filesystem_generation_sha256")),
    )
    if descriptor.get("path") != expected_relative.as_posix():
        return "summary generation descriptor path is stale"
    for field in ("content_manifest_sha256", "filesystem_generation_sha256"):
        value = descriptor.get(field)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            return f"summary generation descriptor {field} is invalid"
    file_count = descriptor.get("file_count")
    directory_count = descriptor.get("directory_count")
    total_file_bytes = descriptor.get("total_file_bytes")
    if type(file_count) is not int or not 1 <= file_count <= 4096:
        return "summary generation descriptor file_count is invalid"
    if type(directory_count) is not int or directory_count != 1:
        return "summary generation descriptor directory_count is invalid"
    if (
        type(total_file_bytes) is not int
        or not 0 <= total_file_bytes <= 256 * 1024 * 1024
    ):
        return "summary generation descriptor total_file_bytes is invalid"
    generation_path = trace_record_path.parent.joinpath(
        *expected_relative.parts
    )
    try:
        observed = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=True,
            max_files=file_count,
            max_directories=directory_count,
            max_total_file_bytes=total_file_bytes,
            max_total_relative_path_bytes=4 * 1024 * 1024,
        )
    except ImmutableArtifactResourceError as exc:
        return (
            "trace_snapshot_resource_unavailable: summary generation "
            f"verification failed: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return (
            "summary generation is absent or invalid: "
            f"{type(exc).__name__}: {exc}"
        )
    if (
        observed.content_manifest_sha256
        != descriptor["content_manifest_sha256"]
        or observed.filesystem_generation_sha256
        != descriptor["filesystem_generation_sha256"]
        or observed.file_count != file_count
        or observed.directory_count != directory_count
        or observed.total_file_bytes != total_file_bytes
    ):
        return "summary generation content or filesystem descriptor is stale"
    admitted_launch_indices = _admitted_launch_indices(trace_admission)
    if admitted_launch_indices is None:
        return "admitted raw launch authority is malformed"
    observed_pairs: set[tuple[int, int]] = set()
    for identity in observed.files:
        match = _NVBIT_TRACE_SUMMARY_FILENAME_RE.fullmatch(
            identity.relative_path
        )
        if match is None:
            return (
                "summary generation contains a noncanonical file: "
                f"{identity.relative_path}"
            )
        repeat_text, launch_text = match.groups()
        repeat_index = int(repeat_text)
        launch_index = int(launch_text)
        if (
            str(repeat_index) != repeat_text
            or str(launch_index) != launch_text
        ):
            return (
                "summary generation contains a noncanonical numeric filename: "
                f"{identity.relative_path}"
            )
        observed_pairs.add((repeat_index, launch_index))
    expected_pairs = {
        (0, launch_index) for launch_index in admitted_launch_indices
    }
    if observed_pairs != expected_pairs or observed.file_count != len(
        expected_pairs
    ):
        return (
            "summary generation does not equal the admitted raw launch authority"
        )
    expected_files = [
        {
            "path": (expected_relative / identity.relative_path).as_posix(),
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for identity in observed.files
    ]
    if descriptor.get("files") != expected_files:
        return "summary generation file identities are stale"
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            expected_files,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if descriptor.get("manifest_sha256") != manifest_sha256:
        return "summary generation manifest seal is stale"
    if trace_record.get("nvbit_trace_summary_paths") != [
        identity["path"] for identity in expected_files
    ]:
        return "active summary reference list disagrees with its descriptor"
    return None


def _discover_nvbit_summary_inputs_with_records(
    root: Path,
    *,
    source_stack: ExitStack,
    retained_authorities: list[Any],
) -> tuple[list[Path], list[str], dict[Path, dict[str, Any]]]:
    """Select NVBit summaries under active trace-record authority.

    New publications are immutable directory generations.  When any canonical
    trace record is present, only its exact referenced paths are eligible;
    stale, unreferenced, and deterministic staging generations are ignored.
    A generation namespace without an active trace record refuses legacy glob
    fallback.  Flat standalone summaries remain replay-compatible only when
    neither authority surface exists.
    """
    try:
        _root_reader, tree_snapshot = _retain_join_input_tree(
            root,
            source_stack=source_stack,
            retained_authorities=retained_authorities,
        )
    except ImmutableArtifactLimitError as exc:
        return [], [
            "joined_evidence_nvbit_discovery_limit_exceeded: "
            f"{type(exc).__name__}: {exc}"
        ], {}
    except ImmutableArtifactResourceError as exc:
        return [], [
            "joined_evidence_nvbit_discovery_resource_unavailable: "
            f"{type(exc).__name__}: {exc}"
        ], {}
    except (ImmutableArtifactError, OSError) as exc:
        return [], [
            "joined_evidence_nvbit_discovery_invalid: "
            f"{type(exc).__name__}: {exc}"
        ], {}

    namespace_entries = _generation_namespace_entries(
        root, tree_snapshot=tree_snapshot
    )
    raw_namespace_entries = _namespace_entries(
        root,
        _RAW_TRACE_GENERATION_NAMESPACE,
        tree_snapshot=tree_snapshot,
    )
    phase_namespace_entries = _namespace_entries(
        root,
        _PHASE_LEDGER_GENERATION_NAMESPACE,
        tree_snapshot=tree_snapshot,
    )
    attempt_namespace_entries = _namespace_entries(
        root,
        _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
        tree_snapshot=tree_snapshot,
    )
    active_records: list[
        tuple[Path, dict[str, Any], Any, AdmittedTraceSources]
    ] = []
    errors: list[str] = []

    # A join root may contain multiple artifact directories. Locate canonical
    # activation leaves only from the bounded inventory, and never dispatch hidden
    # staging or evidence namespace contents as activation authority.
    for candidate in _discover_json_files(
        root, "trace_*.json", tree_snapshot=tree_snapshot
    ):
        if _is_hidden_or_reserved_join_input(root, candidate):
            continue
        try:
            retained_record = source_stack.enter_context(
                open_retained_regular(candidate)
            )
            record = json.loads(retained_record.read_bytes())
        except (
            ImmutableArtifactError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            if candidate.name not in _KNOWN_NONAUTHORITY_TRACE_FILENAMES:
                errors.append(
                    "joined_evidence_nvbit_trace_authority_invalid_"
                    f"{candidate}: unreadable canonical trace-record candidate: "
                    f"{type(exc).__name__}: {exc}"
                )
            continue
        if not _is_active_nvbit_trace_record(candidate, record):
            if candidate.name not in _KNOWN_NONAUTHORITY_TRACE_FILENAMES:
                errors.append(
                    "joined_evidence_nvbit_trace_authority_invalid_"
                    f"{candidate}: canonical trace-record candidate does not "
                    "form an explicitly active canonical trace record"
                )
            continue
        raw_binding_error = None
        if (
            record.get("activation_state") == "active"
            or record.get("nvbit_trace_summary_emit_status") == "ok"
        ):
            raw_binding_error = _active_raw_generation_binding_error(
                candidate,
                record,
                raw_namespace_entries=raw_namespace_entries,
            )
        if raw_binding_error is not None:
            raw_error_kind = (
                "resource_unavailable"
                if raw_binding_error.startswith(
                    "trace_snapshot_resource_unavailable:"
                )
                else "invalid"
            )
            errors.append(
                "joined_evidence_nvbit_raw_generation_"
                f"{raw_error_kind}_{candidate}: {raw_binding_error}"
            )
            continue
        if record.get("nvbit_trace_summary_emit_status") != "ok":
            errors.append(
                "joined_evidence_nvbit_trace_authority_invalid_"
                f"{candidate}: active trace record does not carry summary "
                "status 'ok'"
            )
            continue
        trace_sources, reload_error = _reload_active_trace_admission(
            candidate,
            record,
            source_stack=source_stack,
        )
        if reload_error is not None or trace_sources is None:
            reload_error = reload_error or "active raw admission is unavailable"
            reload_error_kind = (
                "resource_unavailable"
                if reload_error.startswith(
                    "trace_snapshot_resource_unavailable:"
                )
                else "invalid"
            )
            errors.append(
                "joined_evidence_nvbit_raw_generation_"
                f"{reload_error_kind}_{candidate}: {reload_error}"
            )
            continue
        summary_descriptor_error = None
        if record.get("nvbit_trace_summary_emit_status") == "ok":
            summary_descriptor_error = (
                _active_summary_generation_descriptor_error(
                    candidate,
                    record,
                    trace_admission=trace_sources.admission,
                )
            )
        if summary_descriptor_error is not None:
            summary_error_kind = (
                "resource_unavailable"
                if summary_descriptor_error.startswith(
                    "trace_snapshot_resource_unavailable:"
                )
                else "invalid"
            )
            errors.append(
                "joined_evidence_nvbit_summary_generation_"
                f"{summary_error_kind}_{candidate}: "
                f"{summary_descriptor_error}"
            )
            continue
        phase_binding_error = _active_phase_generation_binding_error(
            candidate,
            record,
            phase_namespace_entries=phase_namespace_entries,
            trace_admission=trace_sources.admission,
        )
        if phase_binding_error is not None:
            phase_error_kind = (
                "resource_unavailable"
                if phase_binding_error.startswith(
                    "trace_snapshot_resource_unavailable:"
                )
                else "invalid"
            )
            errors.append(
                "joined_evidence_nvbit_phase_generation_"
                f"{phase_error_kind}_{candidate}: {phase_binding_error}"
            )
            continue
        retained_authorities.extend((retained_record, trace_sources))
        active_records.append(
            (candidate, record, retained_record, trace_sources)
        )

    if not active_records:
        if errors:
            return [], errors, {}
        if (
            namespace_entries
            or raw_namespace_entries
            or phase_namespace_entries
            or attempt_namespace_entries
        ):
            errors.append(
                "joined_evidence_nvbit_generation_authority_missing: "
                "summary, raw generation namespace, phase generation namespace, "
                "or trace-attempt authority namespace exists without an active "
                "canonical trace-record reference; refusing standalone-summary "
                "fallback"
            )
            return [], errors, {}
        legacy_paths: list[Path] = []
        legacy_preloaded: dict[Path, dict[str, Any]] = {}
        for candidate in _discover_json_files(
            root,
            "nvbit_trace_summary_*.json",
            tree_snapshot=tree_snapshot,
        ):
            if _is_hidden_or_reserved_join_input(root, candidate):
                continue
            try:
                retained = source_stack.enter_context(
                    open_retained_regular(candidate)
                )
                retained_authorities.append(retained)
                candidate_record = json.loads(retained.read_bytes())
            except (
                ImmutableArtifactError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                # Preserve malformed historical candidates for the normal schema
                # validator, which emits the established detailed diagnostic.
                legacy_paths.append(candidate)
                continue
            if (
                type(candidate_record) is dict
                and candidate_record.get("schema_version") == 2
            ):
                errors.append(
                    "joined_evidence_nvbit_generation_authority_missing_"
                    f"{candidate}: admitted schema_version 2 summary has no "
                    "active generation authority"
                )
                continue
            legacy_paths.append(candidate)
            legacy_preloaded[candidate] = candidate_record
        if errors:
            return [], errors, {}
        return legacy_paths, errors, legacy_preloaded

    seen_references: set[Path] = set()
    generation_authority: dict[
        Path, dict[str, tuple[Path, dict[str, Any], AdmittedTraceSources]]
    ] = {}
    for (
        trace_record_path,
        trace_record,
        _retained_record,
        trace_sources,
    ) in active_records:
        shape_point = trace_record["shape_point"]
        status = trace_record.get("nvbit_trace_summary_emit_status")
        references = trace_record.get("nvbit_trace_summary_paths")
        admission_sha256 = trace_record.get("trace_admission_sha256")
        filesystem_sha256 = trace_record.get(
            "trace_filesystem_generation_sha256"
        )
        error_prefix = (
            "joined_evidence_nvbit_trace_authority_invalid_"
            f"{trace_record_path}: "
        )

        if status != "ok":
            errors.append(
                error_prefix
                + "active trace record does not carry summary status 'ok'"
            )
            continue
        if type(references) is not list or not references:
            errors.append(
                error_prefix
                + "active trace record has no nonempty summary-path list"
            )
            continue
        if type(admission_sha256) is not str or not _SHA256_RE.match(
            admission_sha256
        ):
            errors.append(
                error_prefix
                + "trace_admission_sha256 must be lowercase 64-hex"
            )
            continue
        if type(filesystem_sha256) is not str or not _SHA256_RE.match(
            filesystem_sha256
        ):
            errors.append(
                error_prefix
                + "trace_filesystem_generation_sha256 must be lowercase 64-hex"
            )
            continue

        expected_prefix = (
            _NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE,
            shape_point,
            admission_sha256,
            filesystem_sha256,
        )
        for reference in references:
            if type(reference) is not str:
                errors.append(error_prefix + "summary path must be a string")
                continue
            relative = PurePosixPath(reference)
            if (
                relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or relative.as_posix() != reference
                or len(relative.parts) != 5
                or relative.parts[:4] != expected_prefix
                or not _NVBIT_TRACE_SUMMARY_FILENAME_RE.fullmatch(
                    relative.parts[-1]
                )
            ):
                errors.append(
                    error_prefix
                    + f"summary path is outside its exact generation: {reference!r}"
                )
                continue

            full_path = trace_record_path.parent.joinpath(*relative.parts)
            if full_path in seen_references:
                errors.append(
                    error_prefix
                    + f"duplicate active summary reference: {reference!r}"
                )
                continue
            seen_references.add(full_path)
            generation_authority.setdefault(full_path.parent, {})[
                full_path.name
            ] = (full_path, trace_record, trace_sources)

    # Pin each referenced final directory from inventory through byte reads.
    # The parsed records returned below are the exact bytes consumed by the
    # join, so no pathname is reopened after the authority check.
    referenced_paths: list[Path] = []
    preloaded_records: dict[Path, dict[str, Any]] = {}
    for generation_directory, expected in sorted(
        generation_authority.items()
    ):
        generation_records: dict[Path, dict[str, Any]] = {}
        try:
            with PinnedDirectoryReader(generation_directory) as reader:
                snapshot = reader.snapshot_tree(
                    require_readonly_files=True,
                    require_readonly_directories=True,
                    max_files=len(expected),
                    max_directories=1,
                    max_total_file_bytes=max(1, len(expected)) * 16 * 1024 * 1024,
                    max_total_relative_path_bytes=max(1, len(expected)) * 4096,
                )
                files = dict(snapshot.files)
                directories = dict(snapshot.directories)
                observed_names = set(files)
                expected_names = set(expected)
                if observed_names != expected_names or set(directories) != {"."}:
                    errors.append(
                        "joined_evidence_nvbit_generation_inventory_invalid_"
                        f"{generation_directory}: expected={sorted(expected_names)} "
                        f"observed={sorted(observed_names)}"
                    )
                    continue
                root_metadata = directories["."]
                if stat.S_IMODE(root_metadata[2]) != 0o555:
                    errors.append(
                        "joined_evidence_nvbit_generation_inventory_invalid_"
                        f"{generation_directory}: final directory mode is not 0555"
                    )
                    continue

                generation_valid = True
                for filename, (
                    full_path,
                    trace_record,
                    trace_sources,
                ) in sorted(expected.items()):
                    metadata = files[filename]
                    if (
                        stat.S_IMODE(metadata[2]) != 0o444
                        or metadata[3] != 1
                    ):
                        errors.append(
                            "joined_evidence_nvbit_generation_inventory_invalid_"
                            f"{generation_directory}: {filename} is not a "
                            "single-link 0444 committed file"
                        )
                        generation_valid = False
                        continue
                    retained_summary = source_stack.enter_context(
                        open_retained_regular(full_path)
                    )
                    retained_authorities.append(retained_summary)
                    if retained_summary.snapshot != metadata:
                        raise ImmutableArtifactError(
                            "summary generation changed after bounded inventory"
                        )
                    payload = retained_summary.read_bytes()
                    try:
                        summary = json.loads(payload)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        errors.append(
                            "joined_evidence_nvbit_generation_binding_invalid_"
                            f"{full_path}: invalid JSON: {exc}"
                        )
                        generation_valid = False
                        continue
                    binding_error = _active_summary_binding_error(
                        summary,
                        trace_record=trace_record,
                        trace_admission=trace_sources.admission,
                        filename=filename,
                    )
                    if binding_error is not None:
                        errors.append(
                            "joined_evidence_nvbit_generation_binding_invalid_"
                            f"{full_path}: {binding_error}"
                        )
                        generation_valid = False
                        continue
                    generation_records[full_path] = summary
                reader.verify_path()
        except ImmutableArtifactResourceError as exc:
            errors.append(
                "joined_evidence_nvbit_generation_resource_unavailable_"
                f"{generation_directory}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactError, OSError) as exc:
            errors.append(
                "joined_evidence_nvbit_generation_inventory_invalid_"
                f"{generation_directory}: {type(exc).__name__}: {exc}"
            )
            continue

        if not generation_valid or len(generation_records) != len(expected):
            continue
        preloaded_records.update(generation_records)
        referenced_paths.extend(generation_records)

    authority_stable = True
    for (
        trace_record_path,
        _trace_record,
        retained_record,
        trace_sources,
    ) in active_records:
        try:
            retained_record.verify_unchanged()
            trace_sources.verify_unchanged()
        except (ImmutableArtifactError, OSError) as exc:
            errors.append(
                "joined_evidence_nvbit_trace_authority_changed_"
                f"{trace_record_path}: {type(exc).__name__}: {exc}"
            )
            authority_stable = False
    if not authority_stable:
        return [], errors, {}

    return sorted(referenced_paths), errors, preloaded_records


def _discover_nvbit_summary_inputs(
    root: Path,
) -> tuple[list[Path], list[str]]:
    """Compatibility wrapper exposing selected paths and conflicts."""
    try:
        with ExitStack() as source_stack:
            retained_authorities: list[Any] = []
            paths, errors, _records = (
                _discover_nvbit_summary_inputs_with_records(
                    root,
                    source_stack=source_stack,
                    retained_authorities=retained_authorities,
                )
            )
            for authority in retained_authorities:
                authority.verify_unchanged()
            return paths, errors
    except ImmutableArtifactLimitError as exc:
        return [], [
            "joined_evidence_nvbit_authority_limit_exceeded: "
            f"{type(exc).__name__}: {exc}"
        ]
    except ImmutableArtifactResourceError as exc:
        return [], [
            "joined_evidence_nvbit_authority_resource_unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]
    except (ImmutableArtifactError, TraceAdmissionError, OSError) as exc:
        return [], [
            "joined_evidence_nvbit_authority_changed: "
            f"{type(exc).__name__}: {exc}"
        ]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Stdlib fallback validators for the fields the join consumes
# directly. These guarantee schema-invalid conflicts fire even when
# `jsonschema` is not installed in the runtime environment. The
# fallback does NOT replace jsonschema for full-schema enforcement;
# it covers the join-relevant subset and runs FIRST so a deterministic
# token is emitted regardless of jsonschema availability. When
# jsonschema is available it runs as a stricter superset.

_NVBIT_SUMMARY_V1_REQUIRED_FIELDS: dict[str, type] = {
    "schema_version": int,
    "slug": str,
    "shape_point": str,
    "repeat_index": int,
    "launch_index": int,
    "kernel_name": str,
    "kernel_signature_sha256": str,
    "inst_count": int,
    "binary_sha256": str,
    "nvbit_trace_pb_path": str,
    "nvbit_trace_pb_sha256": str,
    "nvbit_tracer_build_hash": str,
}
_NVBIT_SUMMARY_V2_REQUIRED_FIELDS: dict[str, type] = {
    **_NVBIT_SUMMARY_V1_REQUIRED_FIELDS,
    "nvbit_threadblock_payload_size_bytes": int,
    "nvbit_threadblock_payload_manifest_sha256": str,
    "trace_admission_sha256": str,
}
_NVBIT_SUMMARY_SHA256_FIELDS: tuple[str, ...] = (
    "kernel_signature_sha256",
    "binary_sha256",
    "nvbit_trace_pb_sha256",
)
_NVBIT_SUMMARY_V2_SHA256_FIELDS: tuple[str, ...] = (
    "nvbit_tracer_build_hash",
    "nvbit_threadblock_payload_manifest_sha256",
    "trace_admission_sha256",
)
_NVBIT_SUMMARY_NONNEGATIVE_INTS: tuple[str, ...] = (
    "repeat_index", "launch_index", "inst_count",
)
_NVBIT_SUMMARY_MIN_LENGTH_STRINGS: tuple[str, ...] = (
    "slug",
    "shape_point",
    "kernel_name",
    "nvbit_trace_pb_path",
    "nvbit_tracer_build_hash",
)

_CUPTI_JOIN_RELEVANT_REQUIRED_FIELDS: dict[str, type] = {
    "slug": str,
    "shape_point": str,
    "repeat_index": int,
    "launch_index": int,
    "kernel_signature_sha256": str,
    "binary_sha256": str,
}
_CUPTI_JOIN_RELEVANT_SHA256_FIELDS: tuple[str, ...] = (
    "kernel_signature_sha256", "binary_sha256",
)


def _validate_nvbit_summary_stdlib(record: dict[str, Any]) -> str | None:
    """Validate the exact historical-v1 or admitted-v2 summary shape."""
    if type(record) is not dict:
        return f"record has type {type(record).__name__}, expected object"

    schema_version = record.get("schema_version")
    if type(schema_version) is not int:
        return "field 'schema_version' must be an integer"
    if schema_version == 1:
        required_fields = _NVBIT_SUMMARY_V1_REQUIRED_FIELDS
        sha256_fields = _NVBIT_SUMMARY_SHA256_FIELDS
        nonnegative_ints = _NVBIT_SUMMARY_NONNEGATIVE_INTS
    elif schema_version == 2:
        required_fields = _NVBIT_SUMMARY_V2_REQUIRED_FIELDS
        sha256_fields = (
            *_NVBIT_SUMMARY_SHA256_FIELDS,
            *_NVBIT_SUMMARY_V2_SHA256_FIELDS,
        )
        nonnegative_ints = (
            *_NVBIT_SUMMARY_NONNEGATIVE_INTS,
            "nvbit_threadblock_payload_size_bytes",
        )
    else:
        return (
            f"field 'schema_version' is {schema_version!r}; expected exactly 1 or 2"
        )

    for field, expected_type in required_fields.items():
        if field not in record:
            return f"missing required field {field!r}"
        value = record[field]
        if expected_type is int and isinstance(value, bool):
            return f"field {field!r} is bool, expected integer"
        if not isinstance(value, expected_type):
            return (
                f"field {field!r} has type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )
    for field in nonnegative_ints:
        if record[field] < 0:
            return f"field {field!r} is negative ({record[field]}); minimum 0"
    for field in _NVBIT_SUMMARY_MIN_LENGTH_STRINGS:
        if len(record[field]) < 1:
            return f"field {field!r} is empty; minLength 1"
    for field in sha256_fields:
        if not _SHA256_RE.match(record[field]):
            return (
                f"field {field!r} value {record[field]!r} does not match "
                f"^[0-9a-f]{{64}}$"
            )
    extras = set(record) - set(required_fields)
    if extras:
        return f"additionalProperties not allowed: {sorted(extras)}"
    return None


def _validate_cupti_sidecar_stdlib(record: dict[str, Any]) -> str | None:
    """Stdlib fallback validator for the CUPTI sidecar fields the
    join directly consumes: 5-tuple join key + binary_sha256. Other
    CUPTI fields fall through to jsonschema's full-schema enforcement
    when available. Returns None if valid, or a one-line error
    message if invalid.
    """
    for field, expected_type in _CUPTI_JOIN_RELEVANT_REQUIRED_FIELDS.items():
        if field not in record:
            return f"missing required field {field!r}"
        value = record[field]
        if expected_type is int and isinstance(value, bool):
            return f"field {field!r} is bool, expected integer"
        if not isinstance(value, expected_type):
            return (
                f"field {field!r} has type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )
    if record["repeat_index"] < 0 or record["launch_index"] < 0:
        return "field 'repeat_index' or 'launch_index' is negative; minimum 0"
    for field in _CUPTI_JOIN_RELEVANT_SHA256_FIELDS:
        if not _SHA256_RE.match(record[field]):
            return (
                f"field {field!r} value {record[field]!r} does not match "
                f"^[0-9a-f]{{64}}$"
            )
    return None


def _validate_against_schema(
    record: dict[str, Any], schema: dict[str, Any], source_path: Path,
    *, side_label: str,
) -> str | None:
    """Validate a record. Returns None if valid, or a one-line error
    message if invalid. Runs the stdlib fallback FIRST (so the
    join-relevant fields are gated even without jsonschema), then
    runs jsonschema as a stricter superset when importable.
    """
    if side_label == "nvbit":
        stdlib_err = _validate_nvbit_summary_stdlib(record)
    elif side_label == "cupti":
        stdlib_err = _validate_cupti_sidecar_stdlib(record)
    else:
        stdlib_err = None
    if stdlib_err is not None:
        return f"{source_path}: {stdlib_err}"
    try:
        import jsonschema  # local import so the dep stays optional
    except ImportError:
        return None
    try:
        jsonschema.validate(record, schema)
    except jsonschema.ValidationError as e:
        return f"{source_path}: {e.message}"
    return None


def _index_records_by_natural_key_with_counts(
    paths: list[Path],
    schema: dict[str, Any],
    side_label: str,
    *,
    preloaded_records: dict[Path, Any] | None = None,
    preloaded_payloads: dict[Path, bytes] | None = None,
) -> tuple[dict[tuple, list[dict[str, Any]]], list[str]]:
    """Load each path, schema-validate, and group by the natural-key
    4-tuple `(slug, shape_point, repeat_index, launch_index)`. All
    records on the same natural key are preserved (no deduplication)
    so duplicate 5-tuples within one side are observable downstream
    as launch-count mismatches. Active generated summaries arrive through
    ``preloaded_records`` so the join consumes descriptor-pinned parsed values.
    Other retained inputs arrive through ``preloaded_payloads`` so malformed JSON
    is diagnosed without reopening a pathname after admission.
    Returns (natural_key_to_records, schema_invalid_tokens).
    """
    schema_invalid: list[str] = []
    natural_to_records: dict[tuple, list[dict[str, Any]]] = {}
    if preloaded_records is None:
        preloaded_records = {}
    if preloaded_payloads is None:
        preloaded_payloads = {}
    for path in paths:
        try:
            if path in preloaded_records:
                record = preloaded_records[path]
            elif path in preloaded_payloads:
                record = json.loads(preloaded_payloads[path])
            else:
                with open(path) as f:
                    record = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{path}: "
                f"not readable valid JSON: {exc}"
            )
            continue
        validation_err = _validate_against_schema(
            record, schema, path, side_label=side_label,
        )
        if validation_err is not None:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{validation_err}"
            )
            continue
        # Build the natural-key 4-tuple from the validated record.
        try:
            natural = tuple(record[f] for f in _JOIN_NATURAL_KEY_FIELDS)
        except KeyError as e:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{path}: missing required join field {e}"
            )
            continue
        natural_to_records.setdefault(natural, []).append(record)
    return natural_to_records, schema_invalid


def _joined_output_path(
    output_dir: Path, platform: str, record: dict[str, Any],
) -> Path:
    """Output path: <output-dir>/<platform>/<slug>/<shape_point>/joined_<rep>_<launch>.json"""
    return (
        output_dir / platform / record["slug"] / record["shape_point"]
        / f"joined_{record['repeat_index']}_{record['launch_index']}.json"
    )


def _run_join_cupti_nvbit(args: argparse.Namespace) -> int:
    """Retain active NVBit authority through the complete join publication."""
    try:
        with ExitStack() as source_stack:
            retained_authorities: list[Any] = []
            return _run_join_cupti_nvbit_retained(
                args,
                source_stack=source_stack,
                retained_authorities=retained_authorities,
            )
    except ImmutableArtifactLimitError as exc:
        print(
            "conflict: joined_evidence_input_limit_exceeded: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    except ImmutableArtifactResourceError as exc:
        print(
            "conflict: joined_evidence_input_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    except (ImmutableArtifactError, TraceAdmissionError, OSError) as exc:
        print(
            "conflict: joined_evidence_input_authority_changed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


def _run_join_cupti_nvbit_retained(
    args: argparse.Namespace,
    *,
    source_stack: ExitStack,
    retained_authorities: list[Any],
) -> int:
    """NVBit + CUPTI two-run offline join with retained input authority."""
    # Hard prerequisite: jsonschema MUST be importable so the full
    # CUPTI sidecar schema (required fields / dim3 sub-objects /
    # activity_kind enum / memcpy/memset arrays / etc.) is enforced
    # before any record is admitted. The stdlib subset validators
    # `_validate_cupti_sidecar_stdlib` + `_validate_nvbit_summary_stdlib`
    # cover the join-relevant fields (5-tuple + binary_sha256) as
    # defense-in-depth but cannot enforce the full schema. Refusing
    # to run without jsonschema is simpler and safer than maintaining
    # a stdlib mirror of the full CUPTI schema.
    try:
        import jsonschema  # noqa: F401  # pyright: ignore[reportMissingModuleSource, reportUnusedImport]
    except ImportError:
        print(
            "joined_evidence_jsonschema_unavailable: the join-cupti-nvbit "
            "subcommand requires the `jsonschema` Python runtime to "
            "enforce the full CUPTI sidecar + NVBit summary schemas. "
            "Install with `pip install jsonschema` and re-run.",
            file=sys.stderr,
        )
        return 2

    cupti_dir = Path(args.cupti_dir)
    nvbit_dir = Path(args.nvbit_dir)
    output_dir = Path(args.output_dir)

    if not cupti_dir.is_dir():
        print(f"joined_evidence_cupti_dir_missing: {cupti_dir}", file=sys.stderr)
        return 2
    if not nvbit_dir.is_dir():
        print(f"joined_evidence_nvbit_dir_missing: {nvbit_dir}", file=sys.stderr)
        return 2

    allowed_platforms = _allowed_platforms_for_join()
    if args.platform not in allowed_platforms:
        print(
            f"joined_evidence_unknown_platform: {args.platform!r} not in {sorted(allowed_platforms)}",
            file=sys.stderr,
        )
        return 2

    cupti_schema, nvbit_schema = _load_join_schemas()
    cupti_reader, cupti_tree_snapshot = _retain_join_input_tree(
        cupti_dir,
        source_stack=source_stack,
        retained_authorities=retained_authorities,
    )
    cupti_paths = _discover_json_files(
        cupti_dir,
        "cupti_activity_*.json",
        tree_snapshot=cupti_tree_snapshot,
    )
    cupti_preloaded = _preload_join_input_bytes(
        cupti_dir,
        cupti_reader,
        cupti_tree_snapshot,
        cupti_paths,
    )
    nvbit_paths, nvbit_discovery_errs, nvbit_preloaded = (
        _discover_nvbit_summary_inputs_with_records(
            nvbit_dir,
            source_stack=source_stack,
            retained_authorities=retained_authorities,
        )
    )

    def verify_join_input_authority(
        _output_root: Path | None = None,
        _output_root_descriptor: int | None = None,
    ) -> None:
        for authority in retained_authorities:
            authority.verify_unchanged()

    cupti_natural, cupti_schema_errs = _index_records_by_natural_key_with_counts(
        cupti_paths,
        cupti_schema,
        "cupti",
        preloaded_payloads=cupti_preloaded,
    )
    nvbit_natural, nvbit_schema_errs = _index_records_by_natural_key_with_counts(
        nvbit_paths,
        nvbit_schema,
        "nvbit",
        preloaded_records=nvbit_preloaded,
    )

    conflicts: list[str] = []
    conflicts.extend(nvbit_discovery_errs)
    conflicts.extend(cupti_schema_errs)
    conflicts.extend(nvbit_schema_errs)

    all_natural_keys = sorted(set(cupti_natural) | set(nvbit_natural))

    joined_records: list[tuple[Path, dict[str, Any]]] = []
    for natural in all_natural_keys:
        cupti_records_here = cupti_natural.get(natural, [])
        nvbit_records_here = nvbit_natural.get(natural, [])
        if cupti_records_here and not nvbit_records_here:
            token = "/".join(str(c) for c in natural)
            conflicts.append(f"joined_evidence_unmatched_cupti_{token}")
            continue
        if nvbit_records_here and not cupti_records_here:
            token = "/".join(str(c) for c in natural)
            conflicts.append(f"joined_evidence_unmatched_nvbit_{token}")
            continue
        # Both sides present. Per-side full-key uniqueness check:
        # within ONE side, every full 5-tuple
        # `(slug, shape_point, repeat_index, launch_index, sig)` MUST
        # appear at most once. A historical producer that reused launch_index
        # or emitted duplicate summaries on the same side trips this
        # check even when the cross-side count + signature multisets
        # both match.
        cupti_full_key_counts = Counter(
            r[_JOIN_INTEGRITY_KEY_FIELD] for r in cupti_records_here
        )
        nvbit_full_key_counts = Counter(
            r[_JOIN_INTEGRITY_KEY_FIELD] for r in nvbit_records_here
        )
        per_side_dups = {
            "cupti": {sig: c for sig, c in cupti_full_key_counts.items() if c > 1},
            "nvbit": {sig: c for sig, c in nvbit_full_key_counts.items() if c > 1},
        }
        if per_side_dups["cupti"] or per_side_dups["nvbit"]:
            token = "/".join(str(c) for c in natural)
            conflicts.append(
                f"joined_evidence_nvbit_cupti_launch_count_mismatch_{token}: "
                f"per_side_duplicate_5tuple cupti_dups={per_side_dups['cupti']} "
                f"nvbit_dups={per_side_dups['nvbit']}"
            )
            continue
        # Cross-side count check: same natural key MUST appear with
        # equal counts on both sides.
        cupti_count = len(cupti_records_here)
        nvbit_count = len(nvbit_records_here)
        if cupti_count != nvbit_count:
            token = "/".join(str(c) for c in natural)
            conflicts.append(
                f"joined_evidence_nvbit_cupti_launch_count_mismatch_{token}: "
                f"cupti_count={cupti_count} nvbit_count={nvbit_count}"
            )
            continue
        # Same natural key, same count on both sides. Verify
        # signatures match (multiset equality). A signature mismatch
        # here means the two runs DID NOT produce bit-identical
        # kernel shapes; the join is invalid.
        cupti_sigs = sorted(r[_JOIN_INTEGRITY_KEY_FIELD] for r in cupti_records_here)
        nvbit_sigs = sorted(r[_JOIN_INTEGRITY_KEY_FIELD] for r in nvbit_records_here)
        if cupti_sigs != nvbit_sigs:
            token = "/".join(str(c) for c in natural)
            conflicts.append(
                f"joined_evidence_signature_mismatch_{token}: "
                f"cupti={cupti_sigs} nvbit={nvbit_sigs}"
            )
            continue
        # Same natural key, same count, same signatures. Pair each
        # CUPTI record to the NVBit record with the same signature
        # (both lists are sorted by signature for determinism). For
        # each pair, verify binary_sha256 matches; emit joined record
        # on success.
        cupti_records_sorted = sorted(
            cupti_records_here, key=lambda r: r[_JOIN_INTEGRITY_KEY_FIELD],
        )
        nvbit_records_sorted = sorted(
            nvbit_records_here, key=lambda r: r[_JOIN_INTEGRITY_KEY_FIELD],
        )
        for cupti_record, nvbit_record in zip(cupti_records_sorted, nvbit_records_sorted):
            sig = cupti_record[_JOIN_INTEGRITY_KEY_FIELD]
            cupti_binary = cupti_record["binary_sha256"]
            nvbit_binary = nvbit_record["binary_sha256"]
            if cupti_binary != nvbit_binary:
                token = "/".join(str(c) for c in natural)
                conflicts.append(
                    f"joined_evidence_binary_sha256_mismatch_{token}: "
                    f"cupti={cupti_binary} nvbit={nvbit_binary}"
                )
                continue
            joined = {
                "merge_evidence_schema_version": JOINED_EVIDENCE_SCHEMA_VERSION,
                "joined_at_utc": _utc_now_iso8601(),
                "natural_key": {f: cupti_record[f] for f in _JOIN_NATURAL_KEY_FIELDS},
                "kernel_signature_sha256": sig,
                "binary_sha256": cupti_binary,
                "cupti_sidecar": cupti_record,
                "nvbit_trace_summary": nvbit_record,
            }
            out_path = _joined_output_path(output_dir, args.platform, cupti_record)
            joined_records.append((out_path, joined))

    # Defense-in-depth final guard: every joined record's output path
    # MUST be unique across the queued writes. The upstream per-side
    # uniqueness + cross-side count + signature-multiset checks should
    # already have caught any case that produces a path collision, but
    # this guard is the load-bearing safety net so a future regression
    # in those upstream checks cannot silently overwrite a joined file.
    output_path_counts = Counter(p for p, _ in joined_records)
    duplicate_paths = sorted(
        p for p, count in output_path_counts.items() if count > 1
    )
    for dup_path in duplicate_paths:
        conflicts.append(
            f"joined_evidence_output_path_collision_{dup_path}: "
            f"{output_path_counts[dup_path]} joined records target the same "
            f"output path; refusing to write (fail-closed)."
        )

    for c in conflicts:
        print(f"conflict: {c}", file=sys.stderr)

    if conflicts:
        print(
            f"merge_evidence join-cupti-nvbit: {len(conflicts)} conflict(s); "
            f"{len(joined_records)} clean join(s) NOT written (fail-closed).",
            file=sys.stderr,
        )
        return 1

    if args.check:
        verify_join_input_authority()
        for path, _ in joined_records:
            print(f"would_write: {path}")
        print(
            f"merge_evidence join-cupti-nvbit: --check: {len(joined_records)} "
            f"join(s) would be written; no files created."
        )
        return 0

    if joined_records:
        material: list[ImmutableFile] = []
        platform_root = output_dir / args.platform
        for path, record in sorted(joined_records, key=lambda item: item[0]):
            try:
                relative = path.relative_to(platform_root).as_posix()
            except ValueError as exc:
                print(
                    "conflict: joined_evidence_output_path_escape_"
                    f"{path}: {exc}",
                    file=sys.stderr,
                )
                return 1
            payload = (json.dumps(
                record, indent=2, sort_keys=False
            ) + "\n").encode("utf-8")
            material.append(ImmutableFile.from_bytes(relative, payload))
        try:
            publish_directory(
                output_dir,
                args.platform,
                material,
                before_commit=verify_join_input_authority,
                accept_existing=False,
            )
        except ImmutableArtifactResourceError as exc:
            print(
                "conflict: joined_evidence_output_resource_unavailable: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        except (ImmutableArtifactError, OSError) as exc:
            print(
                "conflict: joined_evidence_output_publication_invalid: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
    print(
        f"merge_evidence join-cupti-nvbit: ok: {len(joined_records)} joined "
        f"file(s) written under {output_dir}/{args.platform}/."
    )
    return 0


def _utc_now_iso8601() -> str:
    """Current UTC time as ISO-8601 with `Z` suffix."""
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SimBench-BW CUPTI/NVBit two-run offline join tool.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    jcn = sub.add_parser(
        "join-cupti-nvbit",
        help="Join NVBit and CUPTI two-run evidence by canonical 5-tuple.",
    )
    jcn.add_argument("--cupti-dir", required=True,
                     help="Directory tree containing cupti_activity_*.json sidecars.")
    jcn.add_argument("--nvbit-dir", required=True,
                     help="Directory tree containing nvbit_trace_summary_*.json files.")
    jcn.add_argument("--output-dir", required=True,
                     help="Where joined files are written: "
                          "<output-dir>/<platform>/<slug>/<shape_point>/"
                          "joined_<rep>_<launch>.json")
    jcn.add_argument("--platform", required=True,
                     help="Target platform key (one of the schema's allowlist: "
                          "h100, rtx_pro_6000_blackwell, rtx_3090_ti).")
    jcn.add_argument("--check", action="store_true",
                     help="Report-only: do NOT write joined files.")
    jcn.add_argument("--strict", action="store_true",
                     help="Treat warnings as conflicts.")

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "join-cupti-nvbit":
        return _run_join_cupti_nvbit(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
