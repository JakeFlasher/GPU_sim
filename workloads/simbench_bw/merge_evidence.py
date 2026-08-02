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
import errno
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


_MODULE_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_MODULE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_REPO_ROOT))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    DirectoryGenerationFile,
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ImmutableDirectoryGeneration,
    ImmutableFile,
    PinnedDirectoryReader,
    PinnedPathSetMonitor,
    PinnedTreeSnapshot,
    inspect_immutable_directory_generation,
    publish_directory,
    strict_json_loads,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    DEFAULT_TRACE_ADMISSION_LIMITS,
    AdmittedTraceSources,
    TraceAdmission,
    TraceAdmissionError,
    open_admitted_trace_sources,
    raw_trace_generation_limits,
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
_JOIN_DESCRIPTOR_RESOURCE_ERRNOS: frozenset[int] = frozenset({
    errno.EMFILE,
    errno.ENFILE,
})
_RFC3339_DATETIME_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})$"
)


def _is_rfc3339_datetime(value: object) -> bool:
    if type(value) is not str or _RFC3339_DATETIME_RE.fullmatch(value) is None:
        return False
    normalized = value[:-1] + "+00:00" if value[-1] in "Zz" else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _exact_json_equal(observed: object, expected: object) -> bool:
    """Compare decoded JSON without Python bool/int/float coercion."""
    pending = [(observed, expected)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            if set(left) != set(right):
                return False
            pending.extend((left[key], right[key]) for key in left)
        elif type(left) is list:
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right, strict=True))
        elif left != right:
            return False
    return True


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
_MEASUREMENT_MANIFEST_SCHEMA_PATH = (
    SUITE_ROOT / "schemas" / "measurement_manifest.schema.json"
)
_NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE = (
    "nvbit_trace_summary_generations"
)
_RAW_TRACE_GENERATION_NAMESPACE = "raw_trace_generations"
_TRACE_DERIVED_GENERATION_NAMESPACE = "trace_derived_generations"
_PHASE_LEDGER_GENERATION_NAMESPACE = "phase_cycle_ledger_generations"
_TRACE_CAPTURE_ATTEMPT_NAMESPACE = "trace_capture_attempts"
_JOIN_RESERVED_NAMESPACE_COMPONENTS = frozenset({
    _NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE,
    _RAW_TRACE_GENERATION_NAMESPACE,
    _TRACE_DERIVED_GENERATION_NAMESPACE,
    _PHASE_LEDGER_GENERATION_NAMESPACE,
    _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
})
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
_JOIN_PRELOAD_MAX_FILE_BYTES = 16 * 1024 * 1024
_JOIN_PRELOAD_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_JOIN_WORKING_SET_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_JOIN_MATERIALIZED_MAX_OBJECTS = 4_000_000
_JOIN_OUTPUT_MAX_FILE_BYTES = 64 * 1024 * 1024
_JOIN_OUTPUT_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_JOIN_MATERIALIZATION_TRACKER_ENTRY_BYTES = sys.getsizeof({0: None})
_DIRECTORY_GENERATION_DESCRIPTOR_FIELDS = frozenset({
    "schema_version",
    "attempt_id",
    "path",
    "content_manifest_sha256",
    "filesystem_generation_sha256",
    "file_count",
    "directory_count",
    "total_file_bytes",
})
_RAW_GENERATION_DESCRIPTOR_FIELDS = _DIRECTORY_GENERATION_DESCRIPTOR_FIELDS | {
    "trace_admission_sha256",
    "trace_filesystem_generation_sha256",
    "dynamic_trace_sha256",
}
_SUMMARY_GENERATION_DESCRIPTOR_FIELDS = frozenset({
    "schema_version",
    "path",
    "content_manifest_sha256",
    "filesystem_generation_sha256",
    "file_count",
    "directory_count",
    "total_file_bytes",
    "files",
    "manifest_sha256",
})


@dataclass(slots=True)
class _JoinPreloadBudget:
    """Reserve every retained JSON allocation before any payload is read."""

    total_bytes: int = 0
    reserved: dict[str, int] = field(default_factory=dict)

    def reserve(
        self,
        entries: list[tuple[Path, int]],
        *,
        role: str,
    ) -> None:
        additions: dict[str, int] = {}
        for path, size in entries:
            if type(size) is not int or size < 0:
                raise ImmutableArtifactError(
                    f"{role} has an invalid inventoried size: {path}"
                )
            if size > _JOIN_PRELOAD_MAX_FILE_BYTES:
                raise ImmutableArtifactLimitError(
                    f"{role} file {path} is {size} bytes; maximum "
                    f"is {_JOIN_PRELOAD_MAX_FILE_BYTES}"
                )
            key = str(path.absolute())
            prior = self.reserved.get(key, additions.get(key))
            if prior is not None:
                if prior != size:
                    raise ImmutableArtifactError(
                        f"{role} retained size changed for {path}"
                    )
                continue
            additions[key] = size
        attempted = self.total_bytes + sum(additions.values())
        if attempted > _JOIN_PRELOAD_MAX_TOTAL_BYTES:
            raise ImmutableArtifactLimitError(
                f"{role} aggregate preload is {attempted} bytes; maximum "
                f"is {_JOIN_PRELOAD_MAX_TOTAL_BYTES}"
            )
        self.reserved.update(additions)
        self.total_bytes = attempted


def _retained_path_references(
    path: PurePosixPath,
    *,
    role: str,
) -> list[object]:
    references: list[object] = []
    has_component_storage = False
    for path_type in type(path).__mro__:
        slots = path_type.__dict__.get("__slots__", ())
        if type(slots) is str:
            slots = (slots,)
        for slot in slots:
            if slot == "__weakref__":
                continue
            descriptor = path_type.__dict__.get(slot)
            if descriptor is None:
                raise ImmutableArtifactError(
                    f"{role} cannot inspect pathlib retained storage"
                )
            try:
                retained = descriptor.__get__(path, type(path))
            except AttributeError:
                continue
            if slot in {"_raw_paths", "_parts"}:
                if type(retained) not in {list, tuple}:
                    raise ImmutableArtifactError(
                        f"{role} cannot account pathlib retained storage"
                    )
                has_component_storage = True
            references.append(retained)
    if not has_component_storage:
        raise ImmutableArtifactError(
            f"{role} cannot account pathlib retained storage"
        )
    return references


@dataclass(slots=True)
class _JoinMaterializationBudget:
    """Bound decoded records and queued output retained beside input bytes."""

    total_bytes: int = 0
    object_count: int = 0
    seen_objects: dict[int, object] = field(default_factory=dict)

    def _measure(
        self,
        value: object,
        *,
        retained_input_bytes: int,
        role: str,
    ) -> tuple[dict[int, object], int, int]:
        pending = [value]
        added_objects_by_id: dict[int, object] = {}
        expanded_references: set[int] = set()
        added_bytes = 0
        added_objects = 0
        while pending:
            current = pending.pop()
            identity = id(current)
            if (
                identity not in self.seen_objects
                and identity not in added_objects_by_id
            ):
                added_objects_by_id[identity] = current
                added_objects += 1
                if (
                    self.object_count + added_objects
                    > _JOIN_MATERIALIZED_MAX_OBJECTS
                ):
                    raise ImmutableArtifactLimitError(
                        f"{role} retains more than "
                        f"{_JOIN_MATERIALIZED_MAX_OBJECTS} decoded objects"
                    )
                added_bytes += (
                    sys.getsizeof(current)
                    + _JOIN_MATERIALIZATION_TRACKER_ENTRY_BYTES
                )
                attempted = retained_input_bytes + self.total_bytes + added_bytes
                if attempted > _JOIN_WORKING_SET_MAX_TOTAL_BYTES:
                    raise ImmutableArtifactLimitError(
                        f"{role} retained input plus decoded/output material is "
                        f"{attempted} bytes; maximum is "
                        f"{_JOIN_WORKING_SET_MAX_TOTAL_BYTES}"
                    )
            if identity in expanded_references:
                continue
            expanded_references.add(identity)
            if type(current) is dict:
                for key, item in current.items():
                    pending.append(key)
                    pending.append(item)
            elif type(current) in {list, tuple, set, frozenset}:
                pending.extend(current)
            elif isinstance(current, PurePosixPath):
                pending.extend(
                    _retained_path_references(current, role=role)
                )
            elif type(current) is PinnedTreeSnapshot:
                pending.extend((current.files, current.directories))
            elif type(current) is ImmutableDirectoryGeneration:
                pending.extend((
                    current.path,
                    current.content_manifest_sha256,
                    current.filesystem_generation_sha256,
                    current.file_count,
                    current.directory_count,
                    current.total_file_bytes,
                    current.files,
                ))
            elif type(current) is DirectoryGenerationFile:
                pending.extend((
                    current.relative_path,
                    current.sha256,
                    current.size_bytes,
                ))
            elif type(current) is ImmutableFile:
                pending.extend((
                    current.relative_path,
                    current.sha256,
                    current.size_bytes,
                    current.source_path,
                    current.payload,
                    current.allow_hardlinks,
                ))
            elif type(current) is AdmittedTraceSources:
                # The capability's raw bytes and decoded trace graph are already
                # bounded by trace admission. Charge and retain its queue-owned
                # shell identity here without traversing live descriptors/stacks.
                pass
            elif type(current) not in {
                type(None),
                bool,
                int,
                float,
                str,
                bytes,
            }:
                raise ImmutableArtifactError(
                    f"{role} contains unsupported materialized type "
                    f"{type(current).__name__}"
                )
        return added_objects_by_id, added_bytes, added_objects

    def preflight(
        self,
        value: object,
        *,
        retained_input_bytes: int,
        role: str,
    ) -> None:
        """Bound one decoded graph without retaining rejected authority."""
        self._measure(
            value,
            retained_input_bytes=retained_input_bytes,
            role=role,
        )

    def reserve(
        self,
        value: object,
        *,
        retained_input_bytes: int,
        role: str,
    ) -> None:
        added_objects_by_id, added_bytes, added_objects = self._measure(
            value,
            retained_input_bytes=retained_input_bytes,
            role=role,
        )
        self.seen_objects.update(added_objects_by_id)
        self.object_count += added_objects
        self.total_bytes += added_bytes

    def reserve_bytes(
        self,
        size: int,
        *,
        retained_input_bytes: int,
        role: str,
    ) -> None:
        """Charge non-graph container/object storage retained by the join."""
        if type(size) is not int or size < 0:
            raise ImmutableArtifactError(
                f"{role} has an invalid retained byte charge"
            )
        attempted = retained_input_bytes + self.total_bytes + size
        if attempted > _JOIN_WORKING_SET_MAX_TOTAL_BYTES:
            raise ImmutableArtifactLimitError(
                f"{role} retained input plus decoded/output material is "
                f"{attempted} bytes; maximum is "
                f"{_JOIN_WORKING_SET_MAX_TOTAL_BYTES}"
            )
        self.total_bytes += size


@dataclass(slots=True)
class _JoinOutputBudget:
    """Bound queued serialized output before immutable publication."""

    total_bytes: int = 0

    def reserve(self, path: Path, size: int) -> None:
        if type(size) is not int or size < 0:
            raise ImmutableArtifactError(
                f"joined output has invalid serialized size: {path}"
            )
        if size > _JOIN_OUTPUT_MAX_FILE_BYTES:
            raise ImmutableArtifactLimitError(
                f"joined output {path} is {size} bytes; maximum is "
                f"{_JOIN_OUTPUT_MAX_FILE_BYTES}"
            )
        attempted = self.total_bytes + size
        if attempted > _JOIN_OUTPUT_MAX_TOTAL_BYTES:
            raise ImmutableArtifactLimitError(
                f"joined output aggregate is {attempted} bytes; maximum is "
                f"{_JOIN_OUTPUT_MAX_TOTAL_BYTES}"
            )
        self.total_bytes = attempted


def _load_join_schema(
    path: Path,
    *,
    role: str,
    source_stack: ExitStack | None = None,
    retained_authorities: list[Any] | None = None,
    preload_budget: _JoinPreloadBudget | None = None,
    materialization_budget: _JoinMaterializationBudget | None = None,
) -> dict[str, Any]:
    if (preload_budget is None) != (materialization_budget is None):
        raise TypeError("schema preload and materialization budgets must be paired")
    if preload_budget is None:
        preload_budget = _JoinPreloadBudget()
        materialization_budget = _JoinMaterializationBudget()
    assert materialization_budget is not None
    if source_stack is None:
        with ExitStack() as owned_stack:
            return _load_join_schema(
                path,
                role=role,
                source_stack=owned_stack,
                retained_authorities=[],
                preload_budget=preload_budget,
                materialization_budget=materialization_budget,
            )
    if retained_authorities is None:
        raise TypeError("retained_authorities is required with source_stack")
    reader = source_stack.enter_context(PinnedDirectoryReader(path.parent))
    monitor = source_stack.enter_context(
        PinnedPathSetMonitor.from_reader(
            reader,
            regular_paths=(path.name,),
            recursive_directories=(),
        )
    )
    retained_authorities.append(monitor)
    monitor.verify_unchanged()
    snapshot = reader.read_snapshot(
        path.name,
        max_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
    )
    payload = snapshot.payload
    preload_budget.reserve(
        [(path, snapshot.metadata[4])],
        role=role,
    )
    monitor.verify_unchanged()
    value = strict_json_loads(payload, label=role)
    materialization_budget.reserve(
        value,
        retained_input_bytes=preload_budget.total_bytes,
        role=f"decoded {role}",
    )
    if type(value) is not dict:
        raise ImmutableArtifactError(f"{role} root is not an object")
    return value


def _load_join_schemas(
    *,
    source_stack: ExitStack | None = None,
    retained_authorities: list[Any] | None = None,
    preload_budget: _JoinPreloadBudget | None = None,
    materialization_budget: _JoinMaterializationBudget | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load CUPTI, NVBit, and measurement-manifest replay schemas."""
    if (preload_budget is None) != (materialization_budget is None):
        raise TypeError("schema preload and materialization budgets must be paired")
    if preload_budget is None:
        preload_budget = _JoinPreloadBudget()
        materialization_budget = _JoinMaterializationBudget()
    assert materialization_budget is not None
    return (
        _load_join_schema(
            _CUPTI_SIDECAR_SCHEMA_PATH,
            role="CUPTI schema",
            source_stack=source_stack,
            retained_authorities=retained_authorities,
            preload_budget=preload_budget,
            materialization_budget=materialization_budget,
        ),
        _load_join_schema(
            _NVBIT_TRACE_SUMMARY_SCHEMA_PATH,
            role="NVBit summary schema",
            source_stack=source_stack,
            retained_authorities=retained_authorities,
            preload_budget=preload_budget,
            materialization_budget=materialization_budget,
        ),
        _load_join_schema(
            _MEASUREMENT_MANIFEST_SCHEMA_PATH,
            role="measurement manifest schema",
            source_stack=source_stack,
            retained_authorities=retained_authorities,
            preload_budget=preload_budget,
            materialization_budget=materialization_budget,
        ),
    )


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


def _join_path_has_hidden_or_reserved_ancestor(path: Path) -> bool:
    return any(
        component.startswith(".")
        or component in _JOIN_RESERVED_NAMESPACE_COMPONENTS
        for component in Path(path).absolute().parts
        if component not in {Path(path).anchor, "/"}
    )


def _retain_join_input_tree(
    root: Path,
    *,
    source_stack: ExitStack,
    retained_authorities: list[Any],
) -> tuple[PinnedDirectoryReader, Any]:
    """Pin, inventory, and continuously monitor one complete join input tree."""
    if _join_path_has_hidden_or_reserved_ancestor(root):
        raise ImmutableArtifactError(
            "join input root is inside a hidden or reserved evidence namespace: "
            f"{Path(root).absolute()}"
        )
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
    return _join_path_has_hidden_or_reserved_ancestor(candidate.parent) or any(
        part.startswith(".") or part in _JOIN_RESERVED_NAMESPACE_COMPONENTS
        for part in parent_parts
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
    *,
    preload_budget: _JoinPreloadBudget | None = None,
) -> dict[Path, bytes]:
    """Read discovered bytes only after one aggregate allocation preflight."""
    inventory = dict(tree_snapshot.files)
    relative_paths: list[tuple[Path, str, Any]] = []
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
        relative_paths.append((path, relative, metadata))

    budget = preload_budget or _JoinPreloadBudget()
    budget.reserve(
        [(path, metadata[4]) for path, _relative, metadata in relative_paths],
        role="joined-evidence input",
    )
    preloaded: dict[Path, bytes] = {}
    for path, relative, metadata in relative_paths:
        preloaded[path] = reader.read_snapshot(
            relative,
            max_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
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
    """Recognize only the explicit canonical active-record authority shape."""
    if type(record) is not dict:
        return False
    shape_point = record.get("shape_point")
    schema_version = record.get("schema_version")
    return (
        type(schema_version) is int
        and schema_version == 1
        and type(shape_point) is str
        and bool(shape_point)
        and path.name == f"trace_{shape_point}.json"
        and record.get("activation_state") == "active"
        and record.get("shape_set") == "trace_shapes"
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
            candidate = root_absolute.joinpath(*relative.parts)
            if _is_hidden_or_reserved_join_input(root_absolute, candidate):
                continue
            entries.add(candidate)
    return sorted(entries)


def _generation_namespace_entries(root: Path, *, tree_snapshot) -> list[Path]:
    return _namespace_entries(
        root,
        _NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE,
        tree_snapshot=tree_snapshot,
    )


def _artifact_namespace_entries(
    record_path: Path,
    namespace: str,
    entries: list[Path],
) -> list[Path]:
    """Restrict optional namespace authority to one record's artifact root."""
    expected = (record_path.parent / namespace).absolute()
    return [entry for entry in entries if entry == expected]


def _wrong_join_authority_node_errors(
    root: Path,
    *,
    tree_snapshot: Any,
) -> list[str]:
    """Reject authority-like names whose node type would evade selection."""
    reserved_namespaces = frozenset({
        _NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE,
        _RAW_TRACE_GENERATION_NAMESPACE,
        _TRACE_DERIVED_GENERATION_NAMESPACE,
        _PHASE_LEDGER_GENERATION_NAMESPACE,
        _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
    })
    errors: list[str] = []
    for relative_path, _metadata in tree_snapshot.files:
        relative = PurePosixPath(relative_path)
        candidate = _path_from_snapshot(root, relative_path)
        if _is_hidden_or_reserved_join_input(root, candidate):
            continue
        if relative.name in reserved_namespaces:
            errors.append(
                "joined_evidence_nvbit_authority_node_invalid_"
                f"{candidate}: reserved generation authority namespace is "
                "not a directory"
            )
    for relative_path, _metadata in tree_snapshot.directories:
        if relative_path == ".":
            continue
        relative = PurePosixPath(relative_path)
        candidate = _path_from_snapshot(root, relative_path)
        if _is_hidden_or_reserved_join_input(root, candidate):
            continue
        if (
            relative.name not in _KNOWN_NONAUTHORITY_TRACE_FILENAMES
            and PurePosixPath(relative.name).match("trace_*.json")
        ):
            errors.append(
                "joined_evidence_nvbit_authority_node_invalid_"
                f"{candidate}: canonical trace-record candidate is not a "
                "regular file"
            )
        elif _NVBIT_TRACE_SUMMARY_FILENAME_RE.fullmatch(relative.name):
            errors.append(
                "joined_evidence_nvbit_authority_node_invalid_"
                f"{candidate}: legacy NVBit summary candidate is not a "
                "regular file"
            )
    return errors


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
    if (
        type(summary.get("schema_version")) is not int
        or summary["schema_version"] != 2
    ):
        return "active generated summary must use exact integer schema_version 2"
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
        if not _exact_json_equal(summary.get(field), expected_value):
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
        if not _exact_json_equal(trace_record.get(field), expected_value):
            return (
                f"active trace record field {field!r} disagrees with its "
                "admitted raw generation"
            )
    return None


def _active_measurement_manifest_binding_error(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    authority_root: Path,
    authority_reader: PinnedDirectoryReader,
    authority_file_inventory: dict[str, Any],
    measurement_manifest_schema: dict[str, Any],
    preload_budget: _JoinPreloadBudget,
    materialization_budget: _JoinMaterializationBudget,
    expected_platform: str | None,
) -> str | None:
    """Authenticate and schema-bind one active trace execution manifest."""
    shape_point = trace_record.get("shape_point")
    descriptor = trace_record.get("measurement_manifest")
    if type(descriptor) is not dict or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        return "active trace record has no exact measurement-manifest descriptor"
    expected_name = f"measurement_manifest_{shape_point}.json"
    if descriptor.get("path") != expected_name:
        return "measurement manifest path does not name the exact active shape"
    if (
        type(descriptor.get("sha256")) is not str
        or _SHA256_RE.fullmatch(descriptor["sha256"]) is None
        or type(descriptor.get("size_bytes")) is not int
        or descriptor["size_bytes"] < 0
    ):
        return "measurement manifest descriptor identity is invalid"

    root_absolute = Path(authority_root).absolute()
    try:
        record_relative = trace_record_path.absolute().relative_to(root_absolute)
    except ValueError:
        return "active trace record escapes retained measurement authority"
    manifest_relative = (
        PurePosixPath(record_relative.as_posix()).parent / expected_name
    ).as_posix()
    metadata = authority_file_inventory.get(manifest_relative)
    if metadata is None:
        return "measurement manifest is absent from retained authority inventory"
    manifest_path = root_absolute.joinpath(*PurePosixPath(manifest_relative).parts)
    preload_budget.reserve(
        [(manifest_path, metadata[4])],
        role="measurement-manifest authority",
    )
    if stat.S_IMODE(metadata[2]) != 0o444 or metadata[3] != 1:
        return "measurement manifest is not a single-link 0444 committed file"
    payload = authority_reader.read_snapshot(
        manifest_relative,
        max_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
        expected_metadata=metadata,
    ).payload
    if (
        len(payload) != descriptor["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
    ):
        return "measurement manifest descriptor is stale"
    manifest = strict_json_loads(payload, label="measurement manifest")
    materialization_budget.preflight(
        manifest,
        retained_input_bytes=preload_budget.total_bytes,
        role="decoded measurement-manifest authority",
    )
    if type(manifest) is not dict:
        return "measurement manifest root is not an object"
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 2
    ):
        return (
            "measurement manifest fails closed v2 schema: schema_version "
            "must be the exact integer 2"
        )
    if not _is_rfc3339_datetime(manifest.get("timestamp_utc")):
        return (
            "measurement manifest fails closed v2 schema: timestamp_utc is "
            "not an RFC-3339 date-time"
        )
    try:
        import jsonschema
    except ImportError:
        return "measurement manifest schema runtime is unavailable"
    try:
        jsonschema.validate(
            manifest,
            measurement_manifest_schema,
            format_checker=jsonschema.FormatChecker(),
        )
    except jsonschema.SchemaError as exc:
        return f"measurement manifest schema authority is invalid: {exc.message}"
    except jsonschema.ValidationError as exc:
        return f"measurement manifest fails closed v2 schema: {exc.message}"
    expected = {
        "schema_version": 2,
        "mode": "trace",
        "slug": trace_record.get("slug"),
        "shape_point": shape_point,
        "binary_sha256": trace_record.get("binary_sha256"),
    }
    if expected_platform is not None:
        expected["gpu_sku"] = expected_platform
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            return (
                f"measurement manifest field {field!r} disagrees with active "
                "trace execution authority"
            )
    manifest_signature = manifest.get("kernel_signature_sha256")
    if (
        manifest_signature is not None
        and manifest_signature != trace_record.get("kernel_signature_sha256")
    ):
        return (
            "measurement manifest field 'kernel_signature_sha256' disagrees "
            "with active trace execution authority"
        )
    authority_reader.verify_path()
    return None


def _active_raw_generation_binding_error(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    raw_namespace_entries: list[Path],
    preload_budget: _JoinPreloadBudget,
    materialization_budget: _JoinMaterializationBudget,
) -> str | None:
    """Authenticate the exact raw generation named by one active record."""
    materialization_budget.reserve(
        raw_namespace_entries,
        retained_input_bytes=preload_budget.total_bytes,
        role="retained raw generation namespace paths",
    )
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
    if set(descriptor) != _RAW_GENERATION_DESCRIPTOR_FIELDS:
        return "raw_trace_generation descriptor is not exact and closed"
    schema_version = descriptor.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return "raw_trace_generation schema_version must equal one"

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
    raw_limits = raw_trace_generation_limits(admission_limits)
    bounds = {
        "file_count": raw_limits["max_files"],
        "directory_count": raw_limits["max_directories"],
        "total_file_bytes": raw_limits["max_total_file_bytes"],
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
            max_file_bytes=raw_limits["max_file_bytes"],
            max_file_bytes_by_path=raw_limits["max_file_bytes_by_path"],
        )
        materialization_budget.reserve(
            observed,
            retained_input_bytes=preload_budget.total_bytes,
            role="inspected raw trace generation identity",
        )
    except ImmutableArtifactLimitError as exc:
        return (
            "referenced raw generation exceeds the bounded materialization "
            f"limit: {type(exc).__name__}: {exc}"
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


def _active_derived_generation_binding_error(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    derived_namespace_entries: list[Path],
    preload_budget: _JoinPreloadBudget,
    materialization_budget: _JoinMaterializationBudget,
) -> str | None:
    """Authenticate the exact derived generation named by one active record."""
    materialization_budget.reserve(
        derived_namespace_entries,
        retained_input_bytes=preload_budget.total_bytes,
        role="retained derived generation namespace paths",
    )
    descriptor = trace_record.get("trace_derived_generation")
    if descriptor is None:
        if derived_namespace_entries or trace_record.get(
            "trace_capture_attempt_event"
        ) is not None:
            return (
                "derived generation authority exists but the active trace record "
                "has no trace_derived_generation reference"
            )
        return None
    if type(descriptor) is not dict or set(descriptor) != {
        "schema_version",
        "attempt_id",
        "path",
        "content_manifest_sha256",
        "filesystem_generation_sha256",
        "file_count",
        "directory_count",
        "total_file_bytes",
    }:
        return "trace_derived_generation descriptor is not exact and closed"
    shape_point = trace_record.get("shape_point")
    attempt_id = trace_record.get("trace_capture_attempt_id")
    relative_value = descriptor.get("path")
    if type(relative_value) is not str:
        return "trace_derived_generation.path must be a string"
    relative = PurePosixPath(relative_value)
    schema_version = descriptor.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != 1
        or relative.is_absolute()
        or relative.as_posix() != relative_value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(relative.parts) != 3
        or relative.parts[0] != _TRACE_DERIVED_GENERATION_NAMESPACE
        or relative.parts[1] != shape_point
        or relative.parts[2] != attempt_id
        or descriptor.get("attempt_id") != attempt_id
    ):
        return (
            "trace_derived_generation.path is not the exact active attempt path"
        )
    for field in (
        "content_manifest_sha256",
        "filesystem_generation_sha256",
    ):
        value = descriptor.get(field)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            return f"trace_derived_generation.{field} must be lowercase 64-hex"
    limits = DEFAULT_TRACE_ADMISSION_LIMITS
    derived_byte_limit = (
        limits.trace_root_bytes
        + limits.allocations_sidecar_bytes
        + limits.stats_bytes
        + limits.total_threadblock_bytes
    )
    bounds = {
        "file_count": (0, limits.tree_files),
        "directory_count": (1, limits.tree_directories),
        "total_file_bytes": (0, derived_byte_limit),
    }
    for field, (minimum, maximum) in bounds.items():
        value = descriptor.get(field)
        if type(value) is not int or not minimum <= value <= maximum:
            return (
                f"trace_derived_generation.{field} is invalid or exceeds its "
                "hard admission bound"
            )
    generation_path = trace_record_path.parent.joinpath(*relative.parts)
    try:
        observed = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=False,
            max_files=descriptor["file_count"],
            max_directories=descriptor["directory_count"],
            max_total_file_bytes=descriptor["total_file_bytes"],
            max_total_relative_path_bytes=limits.tree_relative_path_bytes,
        )
        materialization_budget.reserve(
            observed,
            retained_input_bytes=preload_budget.total_bytes,
            role="inspected derived trace generation identity",
        )
    except ImmutableArtifactLimitError as exc:
        return (
            "referenced derived generation exceeds the bounded materialization "
            f"limit: {type(exc).__name__}: {exc}"
        )
    except ImmutableArtifactResourceError as exc:
        return (
            "trace_snapshot_resource_unavailable: referenced derived generation "
            f"verification failed: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return (
            "referenced derived generation is absent or invalid: "
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
        return "referenced derived generation descriptor is stale"
    return None


def _active_attempt_event_binding_error(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    authority_root: Path,
    authority_reader: PinnedDirectoryReader,
    authority_file_inventory: dict[
        str, tuple[int, int, int, int, int, int, int]
    ],
    attempt_namespace_entries: list[Path],
    preload_budget: _JoinPreloadBudget,
    materialization_budget: _JoinMaterializationBudget,
) -> str | None:
    """Authenticate the activation-ready event referenced by an active record."""
    descriptor = trace_record.get("trace_capture_attempt_event")
    if descriptor is None:
        if attempt_namespace_entries or trace_record.get(
            "trace_derived_generation"
        ) is not None:
            return (
                "trace-attempt authority exists but the active trace record has "
                "no trace_capture_attempt_event reference"
            )
        return None
    if type(descriptor) is not dict or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        return "trace_capture_attempt_event descriptor is not exact and closed"
    shape_point = trace_record.get("shape_point")
    attempt_id = trace_record.get("trace_capture_attempt_id")
    expected_relative = PurePosixPath(
        _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
        str(shape_point),
        str(attempt_id),
        "activation_ready.json",
    )
    relative_value = descriptor.get("path")
    sha256 = descriptor.get("sha256")
    size_bytes = descriptor.get("size_bytes")
    if (
        type(relative_value) is not str
        or PurePosixPath(relative_value) != expected_relative
        or expected_relative.as_posix() != relative_value
    ):
        return "trace_capture_attempt_event path is not the active-ready event"
    if (
        type(sha256) is not str
        or _SHA256_RE.fullmatch(sha256) is None
        or type(size_bytes) is not int
        or not 0 <= size_bytes <= _JOIN_PRELOAD_MAX_FILE_BYTES
    ):
        return "trace_capture_attempt_event descriptor identity is invalid"
    record_relative = trace_record_path.relative_to(authority_root)
    event_relative = (
        PurePosixPath(record_relative.as_posix()).parent / expected_relative
    ).as_posix()
    metadata = authority_file_inventory.get(event_relative)
    if metadata is None:
        return "trace_capture_attempt_event is absent from retained authority"
    if stat.S_IMODE(metadata[2]) != 0o444 or metadata[3] != 1:
        return "trace_capture_attempt_event is not a single-link 0444 file"
    preload_budget.reserve(
        [(trace_record_path.parent / expected_relative, metadata[4])],
        role="trace capture attempt event authority",
    )
    payload = authority_reader.read_snapshot(
        event_relative,
        max_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
        expected_metadata=metadata,
    ).payload
    if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != sha256:
        return "trace_capture_attempt_event descriptor is stale"
    event = strict_json_loads(payload, label="trace capture attempt event")
    materialization_budget.preflight(
        event,
        retained_input_bytes=preload_budget.total_bytes,
        role="decoded trace capture attempt-event authority",
    )
    phase = trace_record.get("phase_inject_ab")
    phase_generation = (
        phase.get("generation") if type(phase) is dict else None
    )
    expected_event = {
        "schema_version": 1,
        "slug": trace_record.get("slug"),
        "shape_point": shape_point,
        "attempt_id": attempt_id,
        "status": "activation_ready",
        "typed_reason": None,
        "activation_state": "non_active",
        "raw_trace_generation": trace_record.get("raw_trace_generation"),
        "raw_staging_path": None,
        "raw_unresolved_committed_path": None,
        "trace_derived_generation": trace_record.get(
            "trace_derived_generation"
        ),
        "derived_staging_path": None,
        "phase_cycle_ledger_generation": phase_generation,
        "phase_ledger_staging_path": None,
        "nvbit_trace_summary_generation": trace_record.get(
            "nvbit_trace_summary_generation"
        ),
        "measurement_manifest": trace_record.get("measurement_manifest"),
    }
    if not _exact_json_equal(event, expected_event):
        return (
            "trace_capture_attempt_event body disagrees with active generation "
            "authority"
        )
    authority_reader.verify_path()
    return None


def _load_active_trace_admission(
    trace_record_path: Path,
    trace_record: dict[str, Any],
) -> tuple[TraceAdmission | None, str | None]:
    """Reload one active raw generation without retaining rejected authority."""
    descriptor = trace_record.get("raw_trace_generation")
    if type(descriptor) is not dict or type(descriptor.get("path")) is not str:
        return None, "active summaries require an admitted raw generation"
    relative = PurePosixPath(descriptor["path"])
    generation_path = trace_record_path.parent.joinpath(*relative.parts)
    try:
        admission = require_trace_admission(generation_path)
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
    if (
        admission.trace_admission_sha256
        != trace_record.get("trace_admission_sha256")
        or admission.trace_filesystem_generation_sha256
        != trace_record.get("trace_filesystem_generation_sha256")
    ):
        return None, (
            "active raw generation admission seals are stale before summary binding"
        )
    return admission, None


def _retain_active_trace_admission(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    admission: TraceAdmission,
    *,
    source_stack: ExitStack,
) -> tuple[AdmittedTraceSources | None, str | None]:
    """Retain already-validated raw authority through downstream publication."""
    descriptor = trace_record["raw_trace_generation"]
    relative = PurePosixPath(descriptor["path"])
    generation_path = trace_record_path.parent.joinpath(*relative.parts)
    with ExitStack() as candidate_stack:
        try:
            sources = candidate_stack.enter_context(
                open_admitted_trace_sources(
                    generation_path,
                    expected_admission=admission,
                )
            )
        except TraceAdmissionError as exc:
            if exc.typed_reason == "trace_snapshot_resource_unavailable":
                return None, (
                    "trace_snapshot_resource_unavailable: active raw generation "
                    f"retention failed: {exc.detail}"
                )
            return None, (
                "active raw generation changed before summary retention: "
                f"{exc.typed_reason}: {exc.detail}"
            )
        except ImmutableArtifactResourceError as exc:
            return None, (
                "trace_snapshot_resource_unavailable: active raw generation "
                f"retention failed: {type(exc).__name__}: {exc}"
            )
        except (ImmutableArtifactError, OSError) as exc:
            return None, (
                "active raw generation changed before summary retention: "
                f"{type(exc).__name__}: {exc}"
            )
        if sources.admission != admission:
            return None, "active raw generation changed before summary retention"
        source_stack.enter_context(candidate_stack.pop_all())
        return sources, None


def _preflight_generation_payload_descriptors(
    entries: object,
    *,
    authority_root: Path,
    generation_relative: PurePosixPath,
    filename_pattern: re.Pattern[str],
    expected_pairs: set[tuple[int, int]],
    file_count: int,
    total_file_bytes: int,
    preload_budget: _JoinPreloadBudget,
    role: str,
) -> str | None:
    """Reserve exact declared JSON payloads before generation hashing."""
    if type(entries) is not list or len(entries) != file_count:
        return f"{role} payload descriptor list has the wrong cardinality"
    observed_pairs: set[tuple[int, int]] = set()
    observed_paths: set[str] = set()
    reserve: list[tuple[Path, int]] = []
    declared_total = 0
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            return f"{role} payload descriptor is not closed"
        path_value = entry.get("path")
        sha256 = entry.get("sha256")
        size_bytes = entry.get("size_bytes")
        if type(path_value) is not str:
            return f"{role} payload path is not a string"
        relative = PurePosixPath(path_value)
        if (
            relative.is_absolute()
            or relative.as_posix() != path_value
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.parts[:-1] != generation_relative.parts
        ):
            return f"{role} payload path escapes its exact generation"
        match = filename_pattern.fullmatch(relative.parts[-1])
        if match is None:
            return f"{role} payload filename is noncanonical"
        pair = tuple(int(value) for value in match.groups())
        if tuple(str(value) for value in pair) != match.groups():
            return f"{role} payload filename has noncanonical indices"
        if pair not in expected_pairs:
            return f"{role} payload names an unadmitted repeat or launch"
        if path_value in observed_paths or pair in observed_pairs:
            return f"{role} payload descriptors contain duplicates"
        if type(sha256) is not str or _SHA256_RE.fullmatch(sha256) is None:
            return f"{role} payload SHA-256 is invalid"
        if type(size_bytes) is not int or size_bytes < 0:
            return f"{role} payload size is invalid"
        observed_paths.add(path_value)
        observed_pairs.add(pair)
        declared_total += size_bytes
        reserve.append((authority_root.joinpath(*relative.parts), size_bytes))
    if observed_pairs != expected_pairs:
        return f"{role} payload descriptors are incomplete"
    if declared_total != total_file_bytes:
        return f"{role} payload sizes disagree with generation total"
    preload_budget.reserve(reserve, role=f"{role} payload authority")
    return None


def _active_phase_generation_binding_error(
    trace_record_path: Path,
    trace_record: dict[str, Any],
    *,
    phase_namespace_entries: list[Path],
    trace_admission: TraceAdmission,
    preload_budget: _JoinPreloadBudget,
    materialization_budget: _JoinMaterializationBudget,
) -> str | None:
    """Authenticate only the phase generation explicitly named by authority."""
    materialization_budget.reserve(
        phase_namespace_entries,
        retained_input_bytes=preload_budget.total_bytes,
        role="retained phase generation namespace paths",
    )
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
    if set(descriptor) != _DIRECTORY_GENERATION_DESCRIPTOR_FIELDS:
        return "phase generation descriptor is not exact and closed"
    schema_version = descriptor.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return "phase generation schema_version must equal one"
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

    expected_file_count = repeat_count * len(admitted_launch_indices)
    if descriptor["file_count"] != expected_file_count:
        return (
            "referenced phase generation file count does not equal the "
            "admitted repeat-by-launch authority"
        )
    expected_pairs = {
        (repeat_index, launch_index)
        for repeat_index in range(repeat_count)
        for launch_index in admitted_launch_indices
    }
    generation_path = trace_record_path.parent.joinpath(*relative.parts)
    preflight_error = _preflight_generation_payload_descriptors(
        evidence,
        authority_root=trace_record_path.parent,
        generation_relative=relative,
        filename_pattern=_PHASE_LEDGER_FILENAME_RE,
        expected_pairs=expected_pairs,
        file_count=descriptor["file_count"],
        total_file_bytes=descriptor["total_file_bytes"],
        preload_budget=preload_budget,
        role="phase generation",
    )
    if preflight_error is not None:
        return preflight_error
    try:
        observed = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=True,
            max_files=descriptor["file_count"],
            max_directories=descriptor["directory_count"],
            max_total_file_bytes=descriptor["total_file_bytes"],
            max_total_relative_path_bytes=4 * 1024 * 1024,
            max_file_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
        )
        materialization_budget.reserve(
            observed,
            retained_input_bytes=preload_budget.total_bytes,
            role="inspected phase ledger generation identity",
        )
    except ImmutableArtifactLimitError:
        raise
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

    expected_evidence: list[dict[str, Any]] = []
    observed_pairs: set[tuple[int, int]] = set()
    identities = {identity.relative_path: identity for identity in observed.files}
    materialization_budget.reserve(
        (expected_pairs, identities),
        retained_input_bytes=preload_budget.total_bytes,
        role="retained phase generation identity maps",
    )
    preload_budget.reserve(
        [
            (generation_path / relative_path, identity.size_bytes)
            for relative_path, identity in identities.items()
        ],
        role="phase-ledger authority",
    )
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
            materialization_budget.reserve(
                (snapshot, files, directories),
                retained_input_bytes=preload_budget.total_bytes,
                role="retained phase generation inventory",
            )
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
                    max_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
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
                    ledger = strict_json_loads(
                        payload,
                        label=f"phase ledger {relative_path}",
                    )
                except MemoryError as exc:
                    return (
                        "trace_snapshot_resource_unavailable: referenced phase "
                        f"ledger decode failed: {relative_path}: {exc}"
                    )
                except (ImmutableArtifactError, ValueError) as exc:
                    return (
                        "referenced phase ledger is invalid JSON: "
                        f"{relative_path}: {exc}"
                    )
                if type(ledger) is not dict:
                    return f"referenced phase ledger is not an object: {relative_path}"
                materialization_budget.preflight(
                    ledger,
                    retained_input_bytes=preload_budget.total_bytes,
                    role=f"decoded phase ledger {relative_path}",
                )
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
    except (ImmutableArtifactResourceError, MemoryError) as exc:
        return (
            "trace_snapshot_resource_unavailable: referenced phase generation "
            f"read failed: {type(exc).__name__}: {exc}"
        )
    except (ImmutableArtifactError, OSError) as exc:
        return (
            "referenced phase generation changed during authenticated read: "
            f"{type(exc).__name__}: {exc}"
        )

    materialization_budget.reserve(
        (expected_evidence, observed_pairs),
        retained_input_bytes=preload_budget.total_bytes,
        role="retained phase evidence projection",
    )
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
    preload_budget: _JoinPreloadBudget,
    materialization_budget: _JoinMaterializationBudget,
) -> str | None:
    descriptor = trace_record.get("nvbit_trace_summary_generation")
    if type(descriptor) is not dict:
        return "active trace record has no summary generation descriptor"
    if set(descriptor) != _SUMMARY_GENERATION_DESCRIPTOR_FIELDS:
        return "summary generation descriptor is not exact and closed"
    schema_version = descriptor.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return "summary generation schema_version must equal one"
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
    admitted_launch_indices = _admitted_launch_indices(trace_admission)
    if admitted_launch_indices is None:
        return "admitted raw launch authority is malformed"
    expected_pairs = {
        (0, launch_index) for launch_index in admitted_launch_indices
    }
    if file_count != len(expected_pairs):
        return (
            "summary generation does not equal the admitted raw launch authority"
        )
    generation_path = trace_record_path.parent.joinpath(
        *expected_relative.parts
    )
    preflight_error = _preflight_generation_payload_descriptors(
        descriptor.get("files"),
        authority_root=trace_record_path.parent,
        generation_relative=expected_relative,
        filename_pattern=_NVBIT_TRACE_SUMMARY_FILENAME_RE,
        expected_pairs=expected_pairs,
        file_count=file_count,
        total_file_bytes=total_file_bytes,
        preload_budget=preload_budget,
        role="summary generation",
    )
    if preflight_error is not None:
        return preflight_error
    try:
        observed = inspect_immutable_directory_generation(
            generation_path,
            require_nonempty=True,
            max_files=file_count,
            max_directories=directory_count,
            max_total_file_bytes=total_file_bytes,
            max_total_relative_path_bytes=4 * 1024 * 1024,
            max_file_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
        )
        materialization_budget.reserve(
            (expected_pairs, observed),
            retained_input_bytes=preload_budget.total_bytes,
            role="inspected summary generation identity",
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
    materialization_budget.reserve(
        (observed_pairs, expected_files),
        retained_input_bytes=preload_budget.total_bytes,
        role="retained summary generation projections",
    )
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
    measurement_manifest_schema: dict[str, Any] | None = None,
    preload_budget: _JoinPreloadBudget | None = None,
    materialization_budget: _JoinMaterializationBudget | None = None,
    expected_platform: str | None = None,
) -> tuple[list[Path], list[str], dict[Path, bytes]]:
    """Select NVBit summaries and retain their exact payload bytes.

    New publications are immutable directory generations.  When any canonical
    trace record is present, only its exact referenced paths are eligible;
    stale, unreferenced, and deterministic staging generations are ignored.
    A generation namespace without an active trace record refuses legacy glob
    fallback.  Flat standalone summaries remain replay-compatible only when
    neither authority surface exists.
    """
    budget = preload_budget or _JoinPreloadBudget()
    decoded_budget = materialization_budget or _JoinMaterializationBudget()
    manifest_schema = measurement_manifest_schema
    if manifest_schema is None:
        manifest_schema = _load_join_schema(
            _MEASUREMENT_MANIFEST_SCHEMA_PATH,
            role="measurement manifest schema",
            source_stack=source_stack,
            retained_authorities=retained_authorities,
            preload_budget=budget,
            materialization_budget=decoded_budget,
        )
    try:
        root_reader, tree_snapshot = _retain_join_input_tree(
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
    try:
        decoded_budget.reserve(
            tree_snapshot,
            retained_input_bytes=budget.total_bytes,
            role="retained NVBit discovery snapshot",
        )
    except ImmutableArtifactLimitError as exc:
        return [], [
            "joined_evidence_nvbit_trace_authority_limit_exceeded: "
            f"{type(exc).__name__}: {exc}"
        ], {}
    except MemoryError as exc:
        return [], [
            "joined_evidence_nvbit_trace_authority_resource_unavailable: "
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
    derived_namespace_entries = _namespace_entries(
        root,
        _TRACE_DERIVED_GENERATION_NAMESPACE,
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
        tuple[Path, dict[str, Any], AdmittedTraceSources]
    ] = []
    try:
        decoded_budget.reserve(
            active_records,
            retained_input_bytes=budget.total_bytes,
            role="active canonical trace-record queue",
        )
    except ImmutableArtifactLimitError as exc:
        return [], [
            "joined_evidence_nvbit_trace_authority_limit_exceeded: "
            f"{type(exc).__name__}: {exc}"
        ], {}
    except MemoryError as exc:
        return [], [
            "joined_evidence_nvbit_trace_authority_resource_unavailable: "
            f"{type(exc).__name__}: {exc}"
        ], {}
    errors = _wrong_join_authority_node_errors(
        root,
        tree_snapshot=tree_snapshot,
    )

    # A join root may contain multiple artifact directories. Locate canonical
    # activation leaves only from the bounded inventory, and never dispatch hidden
    # staging or evidence namespace contents as activation authority.
    file_inventory = dict(tree_snapshot.files)
    trace_candidates = [
        candidate
        for candidate in _discover_json_files(
            root, "trace_*.json", tree_snapshot=tree_snapshot
        )
        if not _is_hidden_or_reserved_join_input(root, candidate)
    ]
    try:
        decoded_budget.reserve(
            (
                namespace_entries,
                raw_namespace_entries,
                derived_namespace_entries,
                phase_namespace_entries,
                attempt_namespace_entries,
                active_records,
                errors,
                file_inventory,
                trace_candidates,
            ),
            retained_input_bytes=budget.total_bytes,
            role="retained NVBit discovery inventories",
        )
    except ImmutableArtifactLimitError as exc:
        return [], [
            "joined_evidence_nvbit_trace_authority_limit_exceeded: "
            f"{type(exc).__name__}: {exc}"
        ], {}
    except MemoryError as exc:
        return [], [
            "joined_evidence_nvbit_trace_authority_resource_unavailable: "
            f"{type(exc).__name__}: {exc}"
        ], {}

    error_list_baseline_bytes = sys.getsizeof(errors)

    def finalize_discovery(
        paths: list[Path],
        payloads: dict[Path, bytes],
        *,
        role: str,
    ) -> tuple[list[Path], list[str], dict[Path, bytes]]:
        error_projection = tuple(errors)
        decoded_budget.reserve(
            (active_records, paths, payloads, error_projection),
            retained_input_bytes=budget.total_bytes,
            role=role,
        )
        decoded_budget.reserve_bytes(
            max(0, sys.getsizeof(errors) - error_list_baseline_bytes),
            retained_input_bytes=budget.total_bytes,
            role=f"{role} error queue growth",
        )
        return paths, errors, payloads

    try:
        budget.reserve(
            [
                (
                    candidate,
                    file_inventory[candidate.relative_to(root).as_posix()][4],
                )
                for candidate in trace_candidates
            ],
            role="canonical trace-record authority",
        )
    except ImmutableArtifactLimitError as exc:
        return [], [
            "joined_evidence_nvbit_trace_authority_limit_exceeded: "
            f"{type(exc).__name__}: {exc}"
        ], {}
    except (ImmutableArtifactResourceError, MemoryError) as exc:
        return [], [
            "joined_evidence_nvbit_trace_authority_resource_unavailable: "
            f"{type(exc).__name__}: {exc}"
        ], {}
    for candidate in trace_candidates:
        relative_candidate = candidate.relative_to(root).as_posix()
        candidate_metadata = file_inventory[relative_candidate]
        if (
            not stat.S_ISREG(candidate_metadata[2])
            or stat.S_IMODE(candidate_metadata[2]) != 0o444
            or candidate_metadata[3] != 1
        ):
            errors.append(
                "joined_evidence_nvbit_trace_authority_invalid_"
                f"{candidate}: canonical trace-record candidate is not a "
                "single-link 0444 regular file"
            )
            continue
        try:
            payload = root_reader.read_snapshot(
                relative_candidate,
                max_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
                expected_metadata=file_inventory[relative_candidate],
            ).payload
            record = strict_json_loads(
                payload,
                label=f"canonical trace record {candidate}",
            )
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "joined_evidence_nvbit_trace_authority_resource_unavailable_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except (
            ImmutableArtifactError,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            if candidate.name not in _KNOWN_NONAUTHORITY_TRACE_FILENAMES:
                errors.append(
                    "joined_evidence_nvbit_trace_authority_invalid_"
                    f"{candidate}: unreadable canonical trace-record candidate: "
                    f"{type(exc).__name__}: {exc}"
                )
            continue
        try:
            decoded_budget.reserve(
                record,
                retained_input_bytes=budget.total_bytes,
                role=f"decoded canonical trace record {candidate}",
            )
        except ImmutableArtifactLimitError as exc:
            errors.append(
                "joined_evidence_nvbit_trace_authority_limit_exceeded_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except MemoryError as exc:
            errors.append(
                "joined_evidence_nvbit_trace_authority_resource_unavailable_"
                f"{candidate}: {type(exc).__name__}: {exc}"
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
                raw_namespace_entries=_artifact_namespace_entries(
                    candidate,
                    _RAW_TRACE_GENERATION_NAMESPACE,
                    raw_namespace_entries,
                ),
                preload_budget=budget,
                materialization_budget=decoded_budget,
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
        derived_binding_error = _active_derived_generation_binding_error(
            candidate,
            record,
            derived_namespace_entries=_artifact_namespace_entries(
                candidate,
                _TRACE_DERIVED_GENERATION_NAMESPACE,
                derived_namespace_entries,
            ),
            preload_budget=budget,
            materialization_budget=decoded_budget,
        )
        if derived_binding_error is not None:
            derived_error_kind = (
                "resource_unavailable"
                if derived_binding_error.startswith(
                    "trace_snapshot_resource_unavailable:"
                )
                else "invalid"
            )
            errors.append(
                "joined_evidence_nvbit_derived_generation_"
                f"{derived_error_kind}_{candidate}: {derived_binding_error}"
            )
            continue
        if record.get("nvbit_trace_summary_emit_status") != "ok":
            errors.append(
                "joined_evidence_nvbit_trace_authority_invalid_"
                f"{candidate}: active trace record does not carry summary "
                "status 'ok'"
            )
            continue
        trace_admission, reload_error = _load_active_trace_admission(
            candidate,
            record,
        )
        if reload_error is not None or trace_admission is None:
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
        try:
            manifest_binding_error = _active_measurement_manifest_binding_error(
                candidate,
                record,
                authority_root=root,
                authority_reader=root_reader,
                authority_file_inventory=file_inventory,
                measurement_manifest_schema=manifest_schema,
                preload_budget=budget,
                materialization_budget=decoded_budget,
                expected_platform=expected_platform,
            )
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "joined_evidence_nvbit_measurement_manifest_resource_unavailable_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except ImmutableArtifactLimitError as exc:
            errors.append(
                "joined_evidence_nvbit_measurement_manifest_limit_exceeded_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except OSError as exc:
            error_kind = (
                "resource_unavailable"
                if exc.errno in _JOIN_DESCRIPTOR_RESOURCE_ERRNOS
                else "invalid"
            )
            errors.append(
                "joined_evidence_nvbit_measurement_manifest_"
                f"{error_kind}_{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactError, ValueError) as exc:
            errors.append(
                "joined_evidence_nvbit_measurement_manifest_invalid_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        if manifest_binding_error is not None:
            errors.append(
                "joined_evidence_nvbit_measurement_manifest_invalid_"
                f"{candidate}: {manifest_binding_error}"
            )
            continue
        try:
            summary_descriptor_error = (
                _active_summary_generation_descriptor_error(
                    candidate,
                    record,
                    trace_admission=trace_admission,
                    preload_budget=budget,
                    materialization_budget=decoded_budget,
                )
            )
        except ImmutableArtifactLimitError as exc:
            errors.append(
                "joined_evidence_nvbit_summary_generation_limit_exceeded_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "joined_evidence_nvbit_summary_generation_resource_unavailable_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactError, OSError, ValueError) as exc:
            errors.append(
                "joined_evidence_nvbit_summary_generation_invalid_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
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
        try:
            phase_binding_error = _active_phase_generation_binding_error(
                candidate,
                record,
                phase_namespace_entries=_artifact_namespace_entries(
                    candidate,
                    _PHASE_LEDGER_GENERATION_NAMESPACE,
                    phase_namespace_entries,
                ),
                trace_admission=trace_admission,
                preload_budget=budget,
                materialization_budget=decoded_budget,
            )
        except ImmutableArtifactLimitError as exc:
            errors.append(
                "joined_evidence_nvbit_phase_generation_limit_exceeded_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "joined_evidence_nvbit_phase_generation_resource_unavailable_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactError, OSError, ValueError) as exc:
            errors.append(
                "joined_evidence_nvbit_phase_generation_invalid_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
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
        try:
            attempt_binding_error = _active_attempt_event_binding_error(
                candidate,
                record,
                authority_root=root,
                authority_reader=root_reader,
                authority_file_inventory=file_inventory,
                attempt_namespace_entries=_artifact_namespace_entries(
                    candidate,
                    _TRACE_CAPTURE_ATTEMPT_NAMESPACE,
                    attempt_namespace_entries,
                ),
                preload_budget=budget,
                materialization_budget=decoded_budget,
            )
        except ImmutableArtifactLimitError as exc:
            errors.append(
                "joined_evidence_nvbit_attempt_event_limit_exceeded_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "joined_evidence_nvbit_attempt_event_resource_unavailable_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        except (
            ImmutableArtifactError,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            errors.append(
                "joined_evidence_nvbit_attempt_event_invalid_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
            continue
        if attempt_binding_error is not None:
            errors.append(
                "joined_evidence_nvbit_attempt_event_invalid_"
                f"{candidate}: {attempt_binding_error}"
            )
            continue
        trace_sources, reload_error = _retain_active_trace_admission(
            candidate,
            record,
            trace_admission,
            source_stack=source_stack,
        )
        if reload_error is not None or trace_sources is None:
            reload_error = reload_error or "active raw retention is unavailable"
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
        retained_authorities.append(trace_sources)
        queued_active_record = (candidate, record, trace_sources)
        appended = False
        try:
            decoded_budget.reserve(
                queued_active_record,
                retained_input_bytes=budget.total_bytes,
                role=f"retained active trace-record tuple {candidate}",
            )
            prior_list_bytes = sys.getsizeof(active_records)
            active_records.append(queued_active_record)
            appended = True
            decoded_budget.reserve_bytes(
                sys.getsizeof(active_records) - prior_list_bytes,
                retained_input_bytes=budget.total_bytes,
                role=f"active trace-record queue growth {candidate}",
            )
        except ImmutableArtifactLimitError as exc:
            if appended:
                active_records.pop()
            errors.append(
                "joined_evidence_nvbit_trace_authority_limit_exceeded_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )
        except MemoryError as exc:
            if appended:
                active_records.pop()
            errors.append(
                "joined_evidence_nvbit_trace_authority_resource_unavailable_"
                f"{candidate}: {type(exc).__name__}: {exc}"
            )

    if not active_records:
        if errors:
            return finalize_discovery(
                [], {}, role="failed canonical trace-record discovery"
            )
        if (
            namespace_entries
            or raw_namespace_entries
            or derived_namespace_entries
            or phase_namespace_entries
            or attempt_namespace_entries
        ):
            errors.append(
                "joined_evidence_nvbit_generation_authority_missing: "
                "summary, raw generation namespace, derived generation namespace, "
                "phase generation namespace, or trace-attempt authority namespace "
                "exists without an active canonical trace-record reference; "
                "refusing standalone-summary fallback"
            )
            return finalize_discovery(
                [], {}, role="unbound generation authority discovery"
            )
        legacy_paths: list[Path] = []
        legacy_preloaded_payloads: dict[Path, bytes] = {}
        legacy_candidates = [
            candidate
            for candidate in _discover_json_files(
                root,
                "nvbit_trace_summary_*.json",
                tree_snapshot=tree_snapshot,
            )
            if not _is_hidden_or_reserved_join_input(root, candidate)
        ]
        try:
            decoded_budget.reserve(
                legacy_candidates,
                retained_input_bytes=budget.total_bytes,
                role="retained legacy NVBit summary candidates",
            )
            budget.reserve(
                [
                    (
                        candidate,
                        file_inventory[
                            candidate.relative_to(root).as_posix()
                        ][4],
                    )
                    for candidate in legacy_candidates
                ],
                role="legacy NVBit summary authority",
            )
        except ImmutableArtifactLimitError as exc:
            return [], [
                "joined_evidence_nvbit_legacy_limit_exceeded: "
                f"{type(exc).__name__}: {exc}"
            ], {}
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            return [], [
                "joined_evidence_nvbit_legacy_resource_unavailable: "
                f"{type(exc).__name__}: {exc}"
            ], {}

        for candidate in legacy_candidates:
            relative_candidate = candidate.relative_to(root).as_posix()
            metadata = file_inventory[relative_candidate]
            try:
                payload = root_reader.read_snapshot(
                    relative_candidate,
                    max_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
                    expected_metadata=metadata,
                ).payload
            except (ImmutableArtifactResourceError, MemoryError) as exc:
                errors.append(
                    "joined_evidence_nvbit_legacy_resource_unavailable_"
                    f"{candidate}: {type(exc).__name__}: {exc}"
                )
                continue
            except (ImmutableArtifactError, OSError) as exc:
                errors.append(
                    "joined_evidence_nvbit_legacy_authority_invalid_"
                    f"{candidate}: {type(exc).__name__}: {exc}"
                )
                continue
            try:
                candidate_record = strict_json_loads(
                    payload,
                    label=f"legacy NVBit summary {candidate}",
                )
                decoded_budget.preflight(
                    candidate_record,
                    retained_input_bytes=budget.total_bytes,
                    role=f"decoded legacy NVBit summary {candidate}",
                )
            except ImmutableArtifactLimitError as exc:
                errors.append(
                    "joined_evidence_nvbit_legacy_limit_exceeded_"
                    f"{candidate}: {type(exc).__name__}: {exc}"
                )
                continue
            except MemoryError as exc:
                errors.append(
                    "joined_evidence_nvbit_legacy_resource_unavailable_"
                    f"{candidate}: JSON decode allocation failed: {exc}"
                )
                continue
            except (ImmutableArtifactError, ValueError):
                # Preserve malformed historical bytes for the normal schema
                # validator, which emits the established detailed diagnostic.
                legacy_paths.append(candidate)
                legacy_preloaded_payloads[candidate] = payload
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
            legacy_preloaded_payloads[candidate] = payload
        try:
            root_reader.verify_path()
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "joined_evidence_nvbit_legacy_resource_unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        except (ImmutableArtifactError, OSError) as exc:
            errors.append(
                "joined_evidence_nvbit_legacy_authority_invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        if errors:
            return finalize_discovery(
                [], {}, role="failed legacy NVBit summary discovery"
            )
        return finalize_discovery(
            legacy_paths,
            legacy_preloaded_payloads,
            role="retained legacy NVBit summary inputs",
        )

    seen_references: set[Path] = set()
    generation_authority: dict[
        Path, dict[str, tuple[Path, dict[str, Any], AdmittedTraceSources]]
    ] = {}
    for (
        trace_record_path,
        trace_record,
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

    decoded_budget.reserve(
        (seen_references, generation_authority),
        retained_input_bytes=budget.total_bytes,
        role="retained active summary generation authority map",
    )

    # Pin each referenced final directory from inventory through byte reads.
    # The byte payloads returned below are the exact values consumed by the join,
    # so no pathname is reopened after the authority check.
    referenced_paths: list[Path] = []
    preloaded_payloads: dict[Path, bytes] = {}
    for generation_directory, expected in sorted(
        generation_authority.items()
    ):
        generation_payloads: dict[Path, bytes] = {}
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
                decoded_budget.reserve(
                    (
                        generation_directory,
                        snapshot,
                        files,
                        directories,
                        observed_names,
                        expected_names,
                    ),
                    retained_input_bytes=budget.total_bytes,
                    role="retained active summary generation inventory",
                )
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
                for filename in sorted(expected):
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
                if not generation_valid:
                    continue

                budget.reserve(
                    [
                        (expected[filename][0], files[filename][4])
                        for filename in sorted(expected)
                    ],
                    role="active NVBit summary authority",
                )
                for filename, (
                    full_path,
                    trace_record,
                    trace_sources,
                ) in sorted(expected.items()):
                    payload = reader.read_snapshot(
                        filename,
                        max_bytes=_JOIN_PRELOAD_MAX_FILE_BYTES,
                        expected_metadata=files[filename],
                    ).payload
                    try:
                        summary = strict_json_loads(
                            payload,
                            label=f"active NVBit summary {full_path}",
                        )
                        decoded_budget.preflight(
                            summary,
                            retained_input_bytes=budget.total_bytes,
                            role=f"decoded active NVBit summary {full_path}",
                        )
                    except ImmutableArtifactLimitError as exc:
                        errors.append(
                            "joined_evidence_nvbit_generation_limit_exceeded_"
                            f"{full_path}: {type(exc).__name__}: {exc}"
                        )
                        generation_valid = False
                        continue
                    except MemoryError as exc:
                        errors.append(
                            "joined_evidence_nvbit_generation_resource_unavailable_"
                            f"{full_path}: JSON decode allocation failed: {exc}"
                        )
                        generation_valid = False
                        continue
                    except (ImmutableArtifactError, ValueError) as exc:
                        errors.append(
                            "joined_evidence_nvbit_generation_binding_invalid_"
                            f"{full_path}: invalid strict JSON: {exc}"
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
                    generation_payloads[full_path] = payload
                reader.verify_path()
        except ImmutableArtifactLimitError as exc:
            errors.append(
                "joined_evidence_nvbit_generation_limit_exceeded_"
                f"{generation_directory}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "joined_evidence_nvbit_generation_resource_unavailable_"
                f"{generation_directory}: {type(exc).__name__}: {exc}"
            )
            continue
        except (ImmutableArtifactError, OSError, ValueError) as exc:
            errors.append(
                "joined_evidence_nvbit_generation_inventory_invalid_"
                f"{generation_directory}: {type(exc).__name__}: {exc}"
            )
            continue

        if not generation_valid or len(generation_payloads) != len(expected):
            continue
        decoded_budget.reserve(
            generation_payloads,
            retained_input_bytes=budget.total_bytes,
            role="retained active summary generation payload map",
        )
        preloaded_payloads.update(generation_payloads)
        referenced_paths.extend(generation_payloads)

    authority_stable = True
    for (
        trace_record_path,
        _trace_record,
        trace_sources,
    ) in active_records:
        try:
            trace_sources.verify_unchanged()
        except (ImmutableArtifactResourceError, MemoryError) as exc:
            errors.append(
                "joined_evidence_nvbit_trace_authority_resource_unavailable_"
                f"{trace_record_path}: {type(exc).__name__}: {exc}"
            )
            authority_stable = False
        except (ImmutableArtifactError, OSError) as exc:
            errors.append(
                "joined_evidence_nvbit_trace_authority_changed_"
                f"{trace_record_path}: {type(exc).__name__}: {exc}"
            )
            authority_stable = False
    if not authority_stable:
        return finalize_discovery(
            [], {}, role="changed active summary authority discovery"
        )

    sorted_references = sorted(referenced_paths)
    return finalize_discovery(
        sorted_references,
        preloaded_payloads,
        role="retained active NVBit summary inputs",
    )


def _discover_nvbit_summary_inputs(
    root: Path,
    *,
    expected_platform: str | None = None,
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
                    expected_platform=expected_platform,
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
    except (ImmutableArtifactResourceError, MemoryError) as exc:
        return [], [
            "joined_evidence_nvbit_authority_resource_unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]
    except (
        ImmutableArtifactError,
        TraceAdmissionError,
        OSError,
        ValueError,
    ) as exc:
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
    record: object,
    schema: dict[str, Any],
    source_path: Path,
    *,
    side_label: str,
) -> str | None:
    """Validate one retained record against the applicable closed schema."""
    if type(record) is not dict:
        return (
            f"{source_path}: record has type {type(record).__name__}, "
            "expected object"
        )
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
    except jsonschema.SchemaError as e:
        return f"{source_path}: schema authority is invalid: {e.message}"
    except jsonschema.ValidationError as e:
        return f"{source_path}: {e.message}"
    return None


def _index_records_by_natural_key_with_counts(
    paths: list[Path],
    schema: dict[str, Any],
    side_label: str,
    *,
    preloaded_payloads: dict[Path, bytes] | None = None,
    materialization_budget: _JoinMaterializationBudget | None = None,
    retained_input_bytes: int = 0,
) -> tuple[dict[tuple, list[dict[str, Any]]], list[str]]:
    """Load each path, schema-validate, and group by the natural-key
    4-tuple `(slug, shape_point, repeat_index, launch_index)`. All
    records on the same natural key are preserved (no deduplication)
    so duplicate 5-tuples within one side are observable downstream
    as launch-count mismatches. Every input must arrive as retained bytes so
    strict decoding never reopens a pathname after admission.
    Returns (natural_key_to_records, schema_invalid_tokens).
    """
    schema_invalid: list[str] = []
    natural_to_records: dict[tuple, list[dict[str, Any]]] = {}
    if preloaded_payloads is None:
        preloaded_payloads = {}
    budget = materialization_budget or _JoinMaterializationBudget()
    for path in paths:
        try:
            if path in preloaded_payloads:
                record = strict_json_loads(
                    preloaded_payloads[path],
                    label=f"{side_label} join input {path}",
                )
            else:
                schema_invalid.append(
                    f"joined_evidence_schema_invalid_{side_label}_{path}: "
                    "retained preload is missing; refusing pathname fallback"
                )
                continue
        except MemoryError as exc:
            schema_invalid.append(
                f"joined_evidence_resource_unavailable_{side_label}_{path}: "
                f"JSON decode allocation failed: {exc}"
            )
            continue
        except (ImmutableArtifactError, ValueError) as exc:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{path}: "
                f"not readable valid strict JSON: {exc}"
            )
            continue
        try:
            budget.preflight(
                record,
                retained_input_bytes=retained_input_bytes,
                role=f"{side_label} decoded join record {path}",
            )
        except ImmutableArtifactLimitError as exc:
            schema_invalid.append(
                f"joined_evidence_materialization_limit_{side_label}_{path}: {exc}"
            )
            continue
        except MemoryError as exc:
            schema_invalid.append(
                f"joined_evidence_resource_unavailable_{side_label}_{path}: "
                f"decoded-record preflight failed: {exc}"
            )
            continue
        try:
            validation_err = _validate_against_schema(
                record, schema, path, side_label=side_label,
            )
        except MemoryError as exc:
            schema_invalid.append(
                f"joined_evidence_resource_unavailable_{side_label}_{path}: "
                f"schema validation allocation failed: {exc}"
            )
            continue
        except RecursionError as exc:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{path}: "
                f"schema validation nesting exceeds safe limits: {exc}"
            )
            continue
        if validation_err is not None:
            schema_invalid.append(
                f"joined_evidence_schema_invalid_{side_label}_{validation_err}"
            )
            continue
        try:
            budget.reserve(
                record,
                retained_input_bytes=retained_input_bytes,
                role=f"{side_label} decoded join record {path}",
            )
        except ImmutableArtifactLimitError as exc:
            schema_invalid.append(
                f"joined_evidence_materialization_limit_{side_label}_{path}: {exc}"
            )
            continue
        except MemoryError as exc:
            schema_invalid.append(
                f"joined_evidence_resource_unavailable_{side_label}_{path}: "
                f"decoded-record accounting failed: {exc}"
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
    except MemoryError as exc:
        print(
            "conflict: joined_evidence_processing_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    except (
        ImmutableArtifactError,
        TraceAdmissionError,
        OSError,
        ValueError,
    ) as exc:
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
    platform_root = output_dir / args.platform

    preload_budget = _JoinPreloadBudget()
    materialization_budget = _JoinMaterializationBudget()
    (
        cupti_schema,
        nvbit_schema,
        measurement_manifest_schema,
    ) = _load_join_schemas(
        source_stack=source_stack,
        retained_authorities=retained_authorities,
        preload_budget=preload_budget,
        materialization_budget=materialization_budget,
    )
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
        preload_budget=preload_budget,
    )
    nvbit_paths, nvbit_discovery_errs, nvbit_preloaded = (
        _discover_nvbit_summary_inputs_with_records(
            nvbit_dir,
            source_stack=source_stack,
            retained_authorities=retained_authorities,
            measurement_manifest_schema=measurement_manifest_schema,
            preload_budget=preload_budget,
            materialization_budget=materialization_budget,
            expected_platform=args.platform,
        )
    )

    def verify_join_input_authority(
        _output_root: Path | None = None,
        _output_root_descriptor: int | None = None,
    ) -> None:
        for authority in retained_authorities:
            authority.verify_unchanged()

    def finalize_join_input_authority(
        _output_root: Path,
        _output_root_descriptor: int,
    ) -> None:
        """Close retained inputs while publication rollback remains armed."""
        verify_join_input_authority()
        source_stack.close()

    cupti_natural, cupti_schema_errs = _index_records_by_natural_key_with_counts(
        cupti_paths,
        cupti_schema,
        "cupti",
        preloaded_payloads=cupti_preloaded,
        materialization_budget=materialization_budget,
        retained_input_bytes=preload_budget.total_bytes,
    )
    nvbit_natural, nvbit_schema_errs = _index_records_by_natural_key_with_counts(
        nvbit_paths,
        nvbit_schema,
        "nvbit",
        preloaded_payloads=nvbit_preloaded,
        materialization_budget=materialization_budget,
        retained_input_bytes=preload_budget.total_bytes,
    )
    try:
        materialization_budget.reserve(
            (cupti_natural, nvbit_natural),
            retained_input_bytes=preload_budget.total_bytes,
            role="decoded join indexes",
        )
    except ImmutableArtifactLimitError as exc:
        print(
            "conflict: joined_evidence_materialization_limit_exceeded: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    except MemoryError as exc:
        print(
            "conflict: joined_evidence_processing_resource_unavailable: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    conflicts: list[str] = []
    conflicts.extend(nvbit_discovery_errs)
    conflicts.extend(cupti_schema_errs)
    conflicts.extend(nvbit_schema_errs)

    all_natural_keys = sorted(set(cupti_natural) | set(nvbit_natural))

    joined_records: list[tuple[str, dict[str, Any]]] = []
    try:
        materialization_budget.reserve(
            joined_records,
            retained_input_bytes=preload_budget.total_bytes,
            role="joined-record queue",
        )
    except ImmutableArtifactLimitError as exc:
        print(
            "conflict: joined_evidence_materialization_limit_exceeded: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
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
            try:
                relative_output = out_path.relative_to(platform_root).as_posix()
            except ValueError as exc:
                conflicts.append(
                    "joined_evidence_output_path_escape_"
                    f"{out_path}: {exc}"
                )
                continue
            queued_record = (relative_output, joined)
            appended = False
            try:
                materialization_budget.preflight(
                    queued_record,
                    retained_input_bytes=preload_budget.total_bytes,
                    role=f"queued joined record {out_path}",
                )
                materialization_budget.reserve(
                    queued_record,
                    retained_input_bytes=preload_budget.total_bytes,
                    role=f"queued joined record {out_path}",
                )
                prior_list_bytes = sys.getsizeof(joined_records)
                joined_records.append(queued_record)
                appended = True
                materialization_budget.reserve_bytes(
                    sys.getsizeof(joined_records) - prior_list_bytes,
                    retained_input_bytes=preload_budget.total_bytes,
                    role=f"joined-record queue growth {out_path}",
                )
            except ImmutableArtifactLimitError as exc:
                if appended:
                    joined_records.pop()
                conflicts.append(
                    "joined_evidence_materialization_limit_"
                    f"{out_path}: {type(exc).__name__}: {exc}"
                )
                continue
            except MemoryError as exc:
                if appended:
                    joined_records.pop()
                conflicts.append(
                    "joined_evidence_processing_resource_unavailable_"
                    f"{out_path}: {type(exc).__name__}: {exc}"
                )
                continue

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

    material: list[ImmutableFile] = []
    if joined_records:
        output_budget = _JoinOutputBudget()
        try:
            materialization_budget.reserve(
                material,
                retained_input_bytes=preload_budget.total_bytes,
                role="immutable output material queue",
            )
        except ImmutableArtifactLimitError as exc:
            print(
                "conflict: joined_evidence_materialization_limit_exceeded: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        for relative, record in sorted(joined_records, key=lambda item: item[0]):
            path = platform_root.joinpath(*PurePosixPath(relative).parts)
            try:
                payload = (json.dumps(
                    record, indent=2, sort_keys=False
                ) + "\n").encode("utf-8")
                output_budget.reserve(path, len(payload))
                materialization_budget.reserve(
                    payload,
                    retained_input_bytes=preload_budget.total_bytes,
                    role=f"serialized joined output {path}",
                )
                immutable_file = ImmutableFile.from_bytes(relative, payload)
                materialization_budget.reserve(
                    immutable_file,
                    retained_input_bytes=preload_budget.total_bytes,
                    role=f"immutable joined output object {path}",
                )
                prior_list_bytes = sys.getsizeof(material)
                material.append(immutable_file)
                materialization_budget.reserve_bytes(
                    sys.getsizeof(material) - prior_list_bytes,
                    retained_input_bytes=preload_budget.total_bytes,
                    role=f"immutable output material queue growth {path}",
                )
            except ImmutableArtifactLimitError as exc:
                print(
                    "conflict: joined_evidence_output_limit_exceeded: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
            except MemoryError as exc:
                print(
                    "conflict: joined_evidence_output_resource_unavailable: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1

    if args.check and joined_records and os.path.lexists(platform_root):
        print(
            "conflict: joined_evidence_output_publication_invalid: "
            f"destination already exists: {platform_root}",
            file=sys.stderr,
        )
        return 1

    if args.check:
        verify_join_input_authority()
        for relative, _record in joined_records:
            print(
                "would_write: "
                f"{platform_root.joinpath(*PurePosixPath(relative).parts)}"
            )
        print(
            f"merge_evidence join-cupti-nvbit: --check: {len(joined_records)} "
            f"join(s) would be written; no files created."
        )
        return 0

    if joined_records:
        try:
            publish_directory(
                output_dir,
                args.platform,
                material,
                before_commit=verify_join_input_authority,
                after_commit=finalize_join_input_authority,
                accept_existing=False,
            )
        except (ImmutableArtifactResourceError, MemoryError) as exc:
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
