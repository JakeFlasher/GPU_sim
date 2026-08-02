"""Fail-closed admission for a completed forward NVBit trace capture.

Historical M1 payload authorities are verified by their dedicated manifest
gate and must never be dispatched through this forward-capture consumer.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import os
import sys
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    MetadataSnapshot,
    PinnedDirectoryReader,
    PinnedPathSetMonitor,
    PinnedRegularSnapshot,
    PinnedTreeSnapshot,
)
from tools.nvbit_tracer.import_trace_contract import (
    AllocationEntry,
    TraceContractRuntimePrerequisiteMissing,
    TraceContractTrackedBindingsCorrupt,
    _import_pb_modules,
    parse_allocations_sidecar_bytes,
    validate_threadblock,
)
from tools.nvbit_tracer.proto_binding_loader import (
    DynamicTraceBindings,
    ProtoBindingIntegrityError,
    ProtoBindingRuntimePrerequisiteError,
    load_dynamic_trace_bindings,
)
from tools.nvbit_tracer.protobuf_wire_preflight import (
    ProtobufDecodedResourceLimitExceeded,
    ProtobufWireMalformed,
    preflight_threadblock_protobuf,
    preflight_trace_protobuf,
)


EXPECTED_TRACE_CONTRACT_ID = "sm86_addr_carry_v1"
EXPECTED_MEMORY_PAYLOAD_CLASSES = (1, 2, 3, 6, 8)
TRACE_ADMISSION_SCOPE = "forward_capture_only"
I128_MIN = -(2**127)
I128_MAX = 2**127 - 1
I64_MAX = 2**63 - 1
U64_MAX = 2**64 - 1
_MIB = 1024 * 1024
_GIB = 1024 * _MIB


@dataclass(frozen=True)
class TraceAdmissionLimits:
    """Implementation-owned ceilings for one forward trace snapshot."""

    trace_root_bytes: int = 256 * _MIB
    allocations_sidecar_bytes: int = 16 * _MIB
    allocation_entries: int = 131_072
    allocation_json_nesting: int = 16
    stats_bytes: int = 64 * _MIB
    stats_rows: int = 131_072
    stats_columns: int = 256
    threadblock_bytes: int = 256 * _MIB
    total_threadblock_bytes: int = 16 * _GIB
    threadblock_files: int = 131_072
    tree_files: int = 150_000
    tree_directories: int = 65_536
    tree_relative_path_bytes: int = 32 * _MIB

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"trace admission limit {field} must be a positive integer"
                )
        if self.threadblock_files > self.tree_files:
            raise ValueError(
                "threadblock file limit cannot exceed the complete tree file limit"
            )


DEFAULT_TRACE_ADMISSION_LIMITS = TraceAdmissionLimits()


def raw_trace_generation_limits(
    limits: TraceAdmissionLimits,
) -> dict[str, int | dict[str, int]]:
    """Return the shared immutable raw-generation inspection policy."""
    if not isinstance(limits, TraceAdmissionLimits):
        raise TypeError("limits must be a TraceAdmissionLimits value")
    if limits.trace_root_bytes != limits.threadblock_bytes:
        raise ValueError(
            "raw generation inspection requires equal trace-root and "
            "threadblock per-file limits"
        )
    return {
        "max_files": limits.tree_files,
        "max_directories": limits.tree_directories,
        "max_total_file_bytes": (
            limits.trace_root_bytes
            + limits.allocations_sidecar_bytes
            + limits.stats_bytes
            + limits.total_threadblock_bytes
        ),
        "max_total_relative_path_bytes": limits.tree_relative_path_bytes,
        "max_file_bytes": max(
            limits.trace_root_bytes,
            limits.threadblock_bytes,
        ),
        "max_file_bytes_by_path": {
            "stats.csv": limits.stats_bytes,
            "extra_info/allocations.json": limits.allocations_sidecar_bytes,
        },
    }


def _parse_ascii_signed_decimal(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    if (
        not digits
        or len(digits) > 39
        or not all("0" <= char <= "9" for char in digits)
    ):
        raise ValueError("not a bounded ASCII signed-decimal token")
    return int(token, 10)


class TraceAdmissionError(RuntimeError):
    """A capture exists but is not safe to publish or consume."""

    def __init__(self, typed_reason: str, detail: str) -> None:
        super().__init__(f"{typed_reason}: {detail}")
        self.typed_reason = typed_reason
        self.detail = detail


_TRACE_RESOURCE_ERRNOS = frozenset({
    errno.EMFILE,
    errno.ENFILE,
    errno.ENOMEM,
    errno.ENOSPC,
})


def _is_trace_resource_error(exc: BaseException) -> bool:
    pending: list[BaseException] = [exc]
    observed: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in observed:
            continue
        observed.add(identity)
        if isinstance(current, (MemoryError, ImmutableArtifactResourceError)):
            return True
        if (
            isinstance(current, OSError)
            and current.errno in _TRACE_RESOURCE_ERRNOS
        ):
            return True
        for chained in (current.__cause__, current.__context__):
            if isinstance(chained, BaseException):
                pending.append(chained)
    return False


def _raise_trace_source_open_error(
    exc: ImmutableArtifactError | OSError | MemoryError,
    *,
    detail_prefix: str,
) -> NoReturn:
    """Translate source-capability open/auth failures without collapsing limits."""
    if isinstance(exc, ImmutableArtifactLimitError):
        raise TraceAdmissionError(
            "trace_inventory_limit_exceeded",
            f"{detail_prefix}: {exc}",
        ) from exc
    if _is_trace_resource_error(exc):
        raise TraceAdmissionError(
            "trace_snapshot_resource_unavailable",
            f"{detail_prefix}: {exc}",
        ) from exc
    raise TraceAdmissionError(
        "trace_snapshot_changed",
        f"{detail_prefix}: {exc}",
    ) from exc


def _require_source_monitor_quiet(
    monitor: PinnedPathSetMonitor,
    *,
    detail_prefix: str,
) -> None:
    try:
        monitor.verify_unchanged()
    except (ImmutableArtifactError, OSError, MemoryError) as exc:
        _raise_trace_source_open_error(exc, detail_prefix=detail_prefix)


@dataclass(frozen=True)
class TraceLaunchSummary:
    """Immutable summary inputs derived from one admitted root and payload set."""

    launch_index: int
    device_id: int
    stream_id: int
    kernel_id: int
    kernel_name: str
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    shared_mem_static_bytes: int
    shared_mem_dynamic_bytes: int
    regs_per_thread: int
    inst_count: int

    def kernel_signature_inputs(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "grid_z": self.grid_z,
            "block_x": self.block_x,
            "block_y": self.block_y,
            "block_z": self.block_z,
            "shared_mem_static_bytes": self.shared_mem_static_bytes,
            "shared_mem_dynamic_bytes": self.shared_mem_dynamic_bytes,
            "regs_per_thread": self.regs_per_thread,
        }


@dataclass(frozen=True)
class TraceAdmission:
    launch_count: int
    threadblock_count: int
    instruction_count: int
    stats_row_count: int
    expected_record_count: int
    admitted_record_count: int
    dynamic_trace_path: str
    dynamic_trace_size_bytes: int
    dynamic_trace_sha256: str
    allocations_sidecar_size_bytes: int | None
    allocations_sidecar_sha256: str | None
    stats_size_bytes: int
    stats_sha256: str
    threadblock_payload_size_bytes: int
    threadblock_payload_manifest_sha256: str
    trace_filesystem_generation_sha256: str
    launches: tuple[TraceLaunchSummary, ...]
    trace_admission_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        # Keep the portable byte/content seal independent of its presentation
        # path and local inode/directory generations. Dataclass equality still
        # binds both the path and separate filesystem-generation seal during
        # same-host downstream use.
        model = {
            "schema_version": 1,
            "scope": TRACE_ADMISSION_SCOPE,
            "launch_count": self.launch_count,
            "threadblock_count": self.threadblock_count,
            "instruction_count": self.instruction_count,
            "stats_row_count": self.stats_row_count,
            "expected_record_count": self.expected_record_count,
            "admitted_record_count": self.admitted_record_count,
            "dynamic_trace_size_bytes": self.dynamic_trace_size_bytes,
            "dynamic_trace_sha256": self.dynamic_trace_sha256,
            "allocations_sidecar_size_bytes": self.allocations_sidecar_size_bytes,
            "allocations_sidecar_sha256": self.allocations_sidecar_sha256,
            "stats_size_bytes": self.stats_size_bytes,
            "stats_sha256": self.stats_sha256,
            "threadblock_payload_size_bytes": self.threadblock_payload_size_bytes,
            "threadblock_payload_manifest_sha256": (
                self.threadblock_payload_manifest_sha256
            ),
            "launches": [asdict(launch) for launch in self.launches],
        }
        canonical = json.dumps(
            model,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        payload = b"nvbit-trace-admission-v1\n" + canonical
        object.__setattr__(
            self,
            "trace_admission_sha256",
            hashlib.sha256(payload).hexdigest(),
        )


@dataclass(frozen=True)
class AdmittedAllocation:
    """Immutable allocation row detached from the mutable parser result."""

    allocation_id: int
    context_id: int
    device_id: int
    base_address: int
    size_bytes: int
    alive: bool


@dataclass(frozen=True)
class _AdmittedSourceIdentity:
    relative_path: str
    metadata: MetadataSnapshot
    size_bytes: int
    sha256: str
    max_bytes: int


@dataclass(frozen=True)
class _TraceAdmissionSnapshot:
    admission: TraceAdmission
    sources: tuple[_AdmittedSourceIdentity, ...]
    threadblock_paths: tuple[str, ...]
    allocations: dict[int, AllocationEntry] | None


class AdmittedTraceSources:
    """Context-owned access to the exact byte sources of one admission."""

    def __init__(
        self,
        *,
        admission: TraceAdmission,
        reader: PinnedDirectoryReader,
        monitor: PinnedPathSetMonitor,
        snapshot: _TraceAdmissionSnapshot,
        resources: ExitStack,
    ) -> None:
        self._admission = admission
        self._reader = reader
        self._monitor = monitor
        self._sources = {
            source.relative_path: source for source in snapshot.sources
        }
        self._threadblock_paths = snapshot.threadblock_paths
        self._allocations = (
            None
            if snapshot.allocations is None
            else {
                allocation_id: AdmittedAllocation(
                    allocation_id=entry.allocation_id,
                    context_id=entry.context_id,
                    device_id=entry.device_id,
                    base_address=entry.base_address,
                    size_bytes=entry.size_bytes,
                    alive=entry.alive,
                )
                for allocation_id, entry in snapshot.allocations.items()
            }
        )
        self._resources = resources
        self._closed = False

    @property
    def admission(self) -> TraceAdmission:
        return self._admission

    @property
    def threadblock_paths(self) -> tuple[str, ...]:
        self._require_open()
        return self._threadblock_paths

    @property
    def allocations(self) -> Mapping[int, AdmittedAllocation] | None:
        self._require_open()
        try:
            self._monitor.verify_unchanged()
            result = (
                None
                if self._allocations is None
                else MappingProxyType({
                    allocation_id: AdmittedAllocation(
                        allocation_id=entry.allocation_id,
                        context_id=entry.context_id,
                        device_id=entry.device_id,
                        base_address=entry.base_address,
                        size_bytes=entry.size_bytes,
                        alive=entry.alive,
                    )
                    for allocation_id, entry in self._allocations.items()
                })
            )
            self._monitor.verify_unchanged()
            return result
        except BaseException as exc:
            self._poison(exc)
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise ImmutableArtifactError(
                "admitted trace sources capability is closed"
            )

    def _poison(self, primary_error: BaseException) -> None:
        """Close every retained resource without replacing the source failure."""
        try:
            self.close()
        except BaseException as close_error:
            try:
                primary_error.add_note(
                    "admitted trace source cleanup failed: "
                    f"{close_error}"
                )
            except (AttributeError, TypeError):
                pass

    def _read_source(self, relative_path: str) -> PinnedRegularSnapshot:
        self._require_open()
        source = self._sources.get(relative_path)
        if source is None:
            error = TraceAdmissionError(
                "trace_source_not_admitted",
                f"path is not an admitted trace byte source: {relative_path}",
            )
            self._poison(error)
            raise error
        try:
            self._monitor.verify_unchanged()
            try:
                snapshot = self._reader.read_snapshot(
                    relative_path,
                    max_bytes=source.max_bytes,
                    expected_metadata=source.metadata,
                )
                if snapshot.metadata != source.metadata:
                    raise ImmutableArtifactError(
                        "admitted trace source generation changed: "
                        f"{relative_path}"
                    )
                if len(snapshot.payload) != source.size_bytes:
                    raise ImmutableArtifactError(
                        f"admitted trace source size changed: {relative_path}"
                    )
                if hashlib.sha256(snapshot.payload).hexdigest() != source.sha256:
                    raise ImmutableArtifactError(
                        "admitted trace source SHA-256 changed: "
                        f"{relative_path}"
                    )
            except BaseException as exc:
                try:
                    self._monitor.verify_unchanged()
                except BaseException as verification_error:
                    try:
                        exc.add_note(
                            "admitted trace source postcheck failed: "
                            f"{verification_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
                raise
            self._monitor.verify_unchanged()
            return snapshot
        except BaseException as exc:
            self._poison(exc)
            raise

    def read_dynamic_trace_snapshot(self) -> PinnedRegularSnapshot:
        return self._read_source("dynamic_trace.pb")

    def read_stats_snapshot(self) -> PinnedRegularSnapshot:
        return self._read_source("stats.csv")

    def read_allocations_snapshot(self) -> PinnedRegularSnapshot | None:
        if "extra_info/allocations.json" not in self._sources:
            self.verify_unchanged()
            return None
        return self._read_source("extra_info/allocations.json")

    def read_threadblock_snapshot(
        self, relative_path: str,
    ) -> PinnedRegularSnapshot:
        self.verify_unchanged()
        if relative_path not in self._threadblock_paths:
            error = TraceAdmissionError(
                "trace_source_not_admitted",
                f"path is not an admitted threadblock source: {relative_path}",
            )
            self._poison(error)
            raise error
        return self._read_source(relative_path)

    def verify_unchanged(self) -> None:
        self._require_open()
        try:
            self._monitor.verify_unchanged()
            self._reader.verify_path()
            self._monitor.verify_unchanged()
        except BaseException as exc:
            self._poison(exc)
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._resources.close()

    def __enter__(self) -> AdmittedTraceSources:
        try:
            self.verify_unchanged()
        except BaseException as exc:
            try:
                self.close()
            except BaseException as close_error:
                try:
                    exc.add_note(
                        f"admitted trace source entry cleanup failed: {close_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            try:
                self.verify_unchanged()
            except BaseException as verification_error:
                try:
                    exc.add_note(
                        "admitted trace source revalidation failed: "
                        f"{verification_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            self.close()
            return False
        try:
            self.verify_unchanged()
        finally:
            self.close()
        return False


def _trace_filesystem_generation_sha256(
    tree: PinnedTreeSnapshot,
    source_paths: tuple[str, ...],
    lexical_directory_chain: tuple[tuple[str, int, int, int], ...],
) -> str:
    """Seal local file generations and their descriptor-pinned parent chains."""
    files = dict(tree.files)
    directories = dict(tree.directories)
    parent_paths = {"."}
    file_entries: list[dict[str, object]] = []
    for relative_path in sorted(source_paths):
        metadata = files.get(relative_path)
        if metadata is None:
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                f"admitted byte source disappeared from inventory: {relative_path}",
            )
        path = PurePosixPath(relative_path)
        parent = path.parent
        while parent.as_posix() != ".":
            parent_paths.add(parent.as_posix())
            parent = parent.parent
        file_entries.append({
            "path": relative_path,
            "metadata": [str(value) for value in metadata],
        })

    directory_entries: list[dict[str, object]] = []
    for relative_path in sorted(parent_paths):
        metadata = directories.get(relative_path)
        if metadata is None:
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                "admitted byte-source ancestor disappeared from inventory: "
                f"{relative_path}",
            )
        directory_entries.append({
            "path": relative_path,
            # Child publication changes directory size, link count, and times.
            # The retained-chain contract binds the stable directory node and
            # its mode while deliberately permitting unrelated sibling output.
            "node": [str(metadata[index]) for index in (0, 1, 2)],
        })

    model = {
        "schema_version": 1,
        "scope": "admitted_byte_sources_and_parent_chains",
        "lexical_directory_chain": [
            {
                "component": component,
                "node": [str(device), str(inode), str(mode)],
            }
            for component, device, inode, mode in lexical_directory_chain
        ],
        "directories": directory_entries,
        "files": file_entries,
    }
    canonical = json.dumps(
        model,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(
        b"nvbit-trace-filesystem-generation-v1\n" + canonical
    ).hexdigest()


def monitor_admitted_trace_sources(
    trace_dir: Path,
    admission: TraceAdmission,
    *,
    limits: TraceAdmissionLimits = DEFAULT_TRACE_ADMISSION_LIMITS,
) -> AdmittedTraceSources:
    """Open the legacy monitor surface after exact admission authentication."""
    return open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
        limits=limits,
    )


def _raise_admission_prerequisite(
    exc: TraceContractTrackedBindingsCorrupt | TraceContractRuntimePrerequisiteMissing,
) -> NoReturn:
    if _is_trace_resource_error(exc):
        raise TraceAdmissionError(
            "trace_snapshot_resource_unavailable", str(exc)
        ) from exc
    if isinstance(exc, TraceContractTrackedBindingsCorrupt):
        raise TraceAdmissionError("tracked_protobuf_corrupt", str(exc)) from exc
    raise TraceAdmissionError(
        "protobuf_runtime_prerequisite_missing", str(exc)
    ) from exc


def _load_authenticated_bindings() -> DynamicTraceBindings:
    """Translate central-loader failures into the established hard taxonomy."""
    try:
        return load_dynamic_trace_bindings()
    except ProtoBindingRuntimePrerequisiteError as exc:
        raise TraceContractRuntimePrerequisiteMissing(str(exc)) from exc
    except ProtoBindingIntegrityError as exc:
        raise TraceContractTrackedBindingsCorrupt(str(exc)) from exc


def _import_trace_pb2() -> Any:
    return _load_authenticated_bindings().trace_pb2


def _parse_stats_bytes(
    raw: bytes,
    stats_path: Path,
    *,
    max_rows: int,
    max_columns: int,
) -> tuple[int, int, int]:
    try:
        text = raw.decode("utf-8")
        rows = csv.reader(io.StringIO(text, newline=""))
        header = [cell.strip() for cell in next(rows)]
    except StopIteration as exc:
        raise TraceAdmissionError("stats_empty", str(stats_path)) from exc
    except UnicodeError as exc:
        raise TraceAdmissionError("stats_schema_invalid", str(exc)) from exc
    except csv.Error as exc:
        raise TraceAdmissionError("stats_schema_invalid", str(exc)) from exc
    if len(header) > max_columns:
        raise TraceAdmissionError(
            "stats_column_limit_exceeded",
            f"stats.csv header has {len(header)} columns; limit={max_columns}",
        )
    try:
        expected_index = header.index("total_insts")
        admitted_index = header.index("total_reported_insts")
    except ValueError as exc:
        raise TraceAdmissionError(
            "stats_schema_invalid",
            "stats.csv lacks total_insts/total_reported_insts",
        ) from exc

    expected = admitted = row_count = parsed_row_count = 0
    try:
        for line_number, row in enumerate(rows, start=2):
            parsed_row_count += 1
            if parsed_row_count > max_rows:
                raise TraceAdmissionError(
                    "stats_row_limit_exceeded",
                    f"stats.csv has more than {max_rows} data rows",
                )
            if len(row) > max_columns:
                raise TraceAdmissionError(
                    "stats_column_limit_exceeded",
                    f"row {line_number} has {len(row)} columns; "
                    f"limit={max_columns}",
                )
            cells = [cell.strip() for cell in row]
            if not any(cells):
                continue
            if len(cells) <= max(expected_index, admitted_index):
                raise TraceAdmissionError(
                    "stats_schema_invalid", f"row {line_number} is truncated"
                )
            try:
                row_expected = _parse_ascii_signed_decimal(cells[expected_index])
                row_admitted = _parse_ascii_signed_decimal(cells[admitted_index])
            except ValueError as exc:
                raise TraceAdmissionError(
                    "stats_schema_invalid",
                    f"row {line_number} has non-integer counters",
                ) from exc
            if not (
                I128_MIN <= row_expected <= I128_MAX
                and I128_MIN <= row_admitted <= I128_MAX
            ):
                raise TraceAdmissionError(
                    "stats_schema_invalid",
                    f"row {line_number} has counters outside the signed parse domain",
                )
            if row_expected < 0 or row_admitted < 0 or row_admitted > row_expected:
                raise TraceAdmissionError(
                    "stats_counters_invalid",
                    f"row {line_number}: expected={row_expected} "
                    f"admitted={row_admitted}",
                )
            if row_expected > U64_MAX or row_admitted > U64_MAX:
                raise TraceAdmissionError(
                    "stats_schema_invalid",
                    f"row {line_number} has counters outside the uint64 domain",
                )
            if expected > U64_MAX - row_expected:
                raise TraceAdmissionError(
                    "trace_counter_overflow", "stats expected total exceeds uint64"
                )
            if admitted > U64_MAX - row_admitted:
                raise TraceAdmissionError(
                    "trace_counter_overflow", "stats admitted total exceeds uint64"
                )
            expected += row_expected
            admitted += row_admitted
            row_count += 1
    except csv.Error as exc:
        raise TraceAdmissionError("stats_schema_invalid", str(exc)) from exc
    if row_count == 0 or expected == 0 or admitted == 0:
        raise TraceAdmissionError(
            "trace_payload_empty",
            f"stats rows={row_count} expected={expected} admitted={admitted}",
        )
    return row_count, expected, admitted


def _lexical_absolute_trace_dir(trace_dir: Path) -> Path:
    supplied = Path(trace_dir)
    raw = os.fspath(supplied)
    if "\x00" in raw:
        raise TraceAdmissionError(
            "trace_root_unreadable", "trace directory path contains a NUL byte"
        )
    try:
        raw.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TraceAdmissionError(
            "trace_root_unreadable", "trace directory path is not UTF-8"
        ) from exc
    if ".." in supplied.parts:
        raise TraceAdmissionError(
            "trace_root_unreadable",
            f"trace directory path aliases an ancestor via '..': {supplied}",
        )
    return Path(os.path.abspath(raw))


def _serialized_dynamic_trace_path(trace_dir: Path) -> str:
    path = trace_dir / "dynamic_trace.pb"
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _snapshot_lexical_directory_chain(
    reader: PinnedDirectoryReader,
    *,
    closure: bool,
) -> tuple[tuple[str, int, int, int], ...]:
    try:
        return reader.path_chain_nodes()
    except (ImmutableArtifactError, OSError, MemoryError) as exc:
        if _is_trace_resource_error(exc):
            raise TraceAdmissionError(
                "trace_snapshot_resource_unavailable", str(exc)
            ) from exc
        reason = "trace_snapshot_changed" if closure else "trace_filesystem_invalid"
        raise TraceAdmissionError(reason, str(exc)) from exc


def _snapshot_tree(
    reader: PinnedDirectoryReader,
    limits: TraceAdmissionLimits,
    *,
    closure: bool,
) -> PinnedTreeSnapshot:
    try:
        snapshot = reader.snapshot_tree(
            max_files=limits.tree_files,
            max_directories=limits.tree_directories,
            max_total_relative_path_bytes=limits.tree_relative_path_bytes,
        )
    except ImmutableArtifactLimitError as exc:
        raise TraceAdmissionError(
            "trace_inventory_limit_exceeded", str(exc)
        ) from exc
    except (ImmutableArtifactError, OSError, MemoryError) as exc:
        if _is_trace_resource_error(exc):
            raise TraceAdmissionError(
                "trace_snapshot_resource_unavailable", str(exc)
            ) from exc
        reason = "trace_snapshot_changed" if closure else "trace_filesystem_invalid"
        raise TraceAdmissionError(reason, str(exc)) from exc
    for inventory in (snapshot.files, snapshot.directories):
        for relative_path, _metadata in inventory:
            try:
                relative_path.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                reason = (
                    "trace_snapshot_changed" if closure else "trace_filesystem_invalid"
                )
                raise TraceAdmissionError(
                    reason,
                    f"trace inventory path is not UTF-8: {relative_path!r}",
                ) from exc
    return snapshot


def _require_file_metadata(
    files: dict[str, MetadataSnapshot],
    directories: set[str],
    relative_path: str,
    *,
    typed_reason: str,
) -> MetadataSnapshot:
    metadata = files.get(relative_path)
    if metadata is not None:
        return metadata
    detail = "path is a directory" if relative_path in directories else "file is missing"
    raise TraceAdmissionError(typed_reason, f"{relative_path}: {detail}")


def _read_inventory_member(
    reader: PinnedDirectoryReader,
    relative_path: str,
    expected_metadata: MetadataSnapshot,
    *,
    max_bytes: int,
    unreadable_reason: str,
    size_reason: str,
) -> PinnedRegularSnapshot:
    size = expected_metadata[4]
    if size > max_bytes:
        raise TraceAdmissionError(
            size_reason,
            f"{relative_path}: {size} bytes exceeds implementation limit "
            f"{max_bytes}",
        )
    try:
        snapshot = reader.read_snapshot(
            relative_path,
            max_bytes=max_bytes,
            expected_metadata=expected_metadata,
        )
    except (ImmutableArtifactError, OSError, MemoryError) as exc:
        if _is_trace_resource_error(exc):
            raise TraceAdmissionError(
                "trace_snapshot_resource_unavailable",
                f"{relative_path}: {exc}",
            ) from exc
        try:
            reader.verify_regular(relative_path, expected_metadata)
        except (ImmutableArtifactError, OSError, MemoryError) as verification_exc:
            if _is_trace_resource_error(verification_exc):
                raise TraceAdmissionError(
                    "trace_snapshot_resource_unavailable",
                    f"{relative_path}: {verification_exc}",
                ) from verification_exc
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                f"{relative_path}: file generation changed after inventory",
            ) from exc
        raise TraceAdmissionError(
            unreadable_reason, f"{relative_path}: {exc}"
        ) from exc
    if snapshot.metadata != expected_metadata:
        raise TraceAdmissionError(
            "trace_snapshot_changed",
            f"{relative_path}: file generation changed after inventory",
        )
    return snapshot


def _validate_trace_root(root_bytes: bytes, root_path: Path) -> Any:
    try:
        preflight_trace_protobuf(root_bytes)
    except MemoryError as exc:
        raise TraceAdmissionError(
            "trace_snapshot_resource_unavailable",
            f"{root_path}: protobuf preflight exhausted memory",
        ) from exc
    except ProtobufDecodedResourceLimitExceeded as exc:
        raise TraceAdmissionError(
            "protobuf_decoded_resource_limit_exceeded",
            f"{root_path}: {exc}",
        ) from exc
    except ProtobufWireMalformed as exc:
        raise TraceAdmissionError("trace_root_unparseable", str(exc)) from exc

    try:
        trace_pb2 = _import_trace_pb2()
    except (
        TraceContractTrackedBindingsCorrupt,
        TraceContractRuntimePrerequisiteMissing,
    ) as exc:
        _raise_admission_prerequisite(exc)
    root = trace_pb2.Trace()
    try:
        root.ParseFromString(root_bytes)
    except MemoryError as exc:
        raise TraceAdmissionError(
            "trace_snapshot_resource_unavailable",
            f"{root_path}: protobuf decode exhausted memory",
        ) from exc
    except Exception as exc:  # protobuf raises version-specific decode classes
        raise TraceAdmissionError("trace_root_unparseable", str(exc)) from exc
    if root.trace_contract_id != EXPECTED_TRACE_CONTRACT_ID:
        raise TraceAdmissionError(
            "trace_contract_mismatch",
            f"got {root.trace_contract_id!r}, expected {EXPECTED_TRACE_CONTRACT_ID!r}",
        )
    capabilities = sorted(int(value) for value in root.memory_payload_classes)
    if capabilities != list(EXPECTED_MEMORY_PAYLOAD_CLASSES):
        raise TraceAdmissionError(
            "trace_capabilities_invalid",
            f"got {capabilities}, expected {list(EXPECTED_MEMORY_PAYLOAD_CLASSES)}",
        )
    return root


def _ordered_launches(root: Any) -> list[tuple[int, int, Any]]:
    launches: list[tuple[int, int, Any]] = []
    launch_ids: set[tuple[int, int, int]] = set()
    for device_key in sorted(root.gpu_device.keys()):
        device = root.gpu_device[device_key]
        if int(device_key) != int(device.id):
            raise TraceAdmissionError(
                "trace_launch_identity_mismatch",
                f"device map key {device_key} disagrees with embedded id {device.id}",
            )
        for stream_key in sorted(device.streams.keys()):
            stream = device.streams[stream_key]
            if int(stream_key) != int(stream.id):
                raise TraceAdmissionError(
                    "trace_launch_identity_mismatch",
                    f"stream map key {stream_key} disagrees with embedded id "
                    f"{stream.id}",
                )
            for kernel in sorted(stream.kernels, key=lambda item: int(item.id)):
                launch_id = (int(device_key), int(stream_key), int(kernel.id))
                if launch_id in launch_ids:
                    raise TraceAdmissionError(
                        "trace_launch_identity_mismatch",
                        f"duplicate launch identity {launch_id}",
                    )
                launch_ids.add(launch_id)
                if len(launches) == U64_MAX:
                    raise TraceAdmissionError(
                        "trace_counter_overflow", "launch count exceeds uint64"
                    )
                launches.append((int(device_key), int(stream_key), kernel))
                if int(kernel.function_unique_id) <= 0:
                    raise TraceAdmissionError(
                        "function_unique_id_invalid",
                        f"root launch {launch_id} has function_unique_id "
                        f"{kernel.function_unique_id}",
                    )
                geometry = (
                    int(kernel.grid_dim.x),
                    int(kernel.grid_dim.y),
                    int(kernel.grid_dim.z),
                    int(kernel.block_dim.x),
                    int(kernel.block_dim.y),
                    int(kernel.block_dim.z),
                )
                if any(value <= 0 for value in geometry):
                    raise TraceAdmissionError(
                        "trace_launch_geometry_invalid",
                        f"launch {launch_id} has grid/block geometry {geometry}",
                    )
    if not launches:
        raise TraceAdmissionError("trace_launches_empty", "root Trace has zero launches")
    return launches


def _import_threadblock_pb2() -> Any:
    try:
        return _load_authenticated_bindings().threadblock_pb2
    except (
        TraceContractTrackedBindingsCorrupt,
        TraceContractRuntimePrerequisiteMissing,
    ) as exc:
        _raise_admission_prerequisite(exc)


def _launch_summary(
    launch_index: int,
    device_id: int,
    stream_id: int,
    kernel: Any,
    inst_count: int,
) -> TraceLaunchSummary:
    combined_shared_bytes = int(kernel.size_shared_memory)
    if combined_shared_bytes < 0:
        raise TraceAdmissionError(
            "trace_launch_shared_memory_invalid",
            f"launch {(device_id, stream_id, int(kernel.id))} has negative "
            f"combined shared-memory bytes {combined_shared_bytes}",
        )
    if combined_shared_bytes:
        raise TraceAdmissionError(
            "shared_memory_split_unavailable",
            "NVBit kernel.size_shared_memory carries the producer's combined "
            "static+dynamic byte count; the split cannot be authenticated for "
            f"launch {(device_id, stream_id, int(kernel.id))}",
        )
    # CUDA static and dynamic shared-memory sizes are nonnegative, so a combined
    # producer value of zero authenticates the sole possible split: zero + zero.
    return TraceLaunchSummary(
        launch_index=launch_index,
        device_id=device_id,
        stream_id=stream_id,
        kernel_id=int(kernel.id),
        kernel_name=str(kernel.name),
        grid_x=int(kernel.grid_dim.x),
        grid_y=int(kernel.grid_dim.y),
        grid_z=int(kernel.grid_dim.z),
        block_x=int(kernel.block_dim.x),
        block_y=int(kernel.block_dim.y),
        block_z=int(kernel.block_dim.z),
        shared_mem_static_bytes=0,
        shared_mem_dynamic_bytes=0,
        regs_per_thread=int(kernel.number_of_registers),
        inst_count=inst_count,
    )


def _require_trace_admission_snapshot(
    trace_dir: Path,
    *,
    limits: TraceAdmissionLimits = DEFAULT_TRACE_ADMISSION_LIMITS,
    reader: PinnedDirectoryReader | None = None,
) -> _TraceAdmissionSnapshot:
    """Admit exact bounded bytes and retain their ephemeral source identities."""

    if not isinstance(limits, TraceAdmissionLimits):
        raise TypeError("limits must be a TraceAdmissionLimits value")
    trace_dir = _lexical_absolute_trace_dir(trace_dir)
    if reader is None:
        try:
            reader_context = PinnedDirectoryReader(trace_dir)
        except (ImmutableArtifactError, OSError, MemoryError) as exc:
            if _is_trace_resource_error(exc):
                raise TraceAdmissionError(
                    "trace_snapshot_resource_unavailable", str(exc)
                ) from exc
            raise TraceAdmissionError("trace_root_unreadable", str(exc)) from exc
    else:
        reader_context = nullcontext(reader)

    with reader_context as reader:
        initial_directory_chain = _snapshot_lexical_directory_chain(
            reader,
            closure=False,
        )
        initial_tree = _snapshot_tree(reader, limits, closure=False)
        files = dict(initial_tree.files)
        directories = {path for path, _metadata in initial_tree.directories}

        root_metadata = _require_file_metadata(
            files,
            directories,
            "dynamic_trace.pb",
            typed_reason="trace_root_unreadable",
        )
        root_snapshot = _read_inventory_member(
            reader,
            "dynamic_trace.pb",
            root_metadata,
            max_bytes=limits.trace_root_bytes,
            unreadable_reason="trace_root_unreadable",
            size_reason="trace_root_size_limit_exceeded",
        )
        root_bytes = root_snapshot.payload
        root_path = trace_dir / "dynamic_trace.pb"
        if not root_bytes:
            raise TraceAdmissionError("trace_root_empty", str(root_path))

        if "threadblocks" not in directories:
            raise TraceAdmissionError(
                "trace_threadblocks_empty",
                "trace has no regular threadblocks/ directory",
            )
        threadblock_members = tuple(
            path
            for path in sorted(files)
            if PurePosixPath(path).parts[:1] == ("threadblocks",)
        )
        non_payload_path = next(
            (
                path
                for path in threadblock_members
                if PurePosixPath(path).suffix != ".pb"
            ),
            None,
        )
        if non_payload_path is not None:
            raise TraceAdmissionError(
                "trace_payload_nonprotobuf",
                "every regular file under threadblocks/ must be an admitted "
                f".pb payload: {non_payload_path}",
            )
        payload_paths = threadblock_members
        if not payload_paths:
            raise TraceAdmissionError(
                "trace_threadblocks_empty",
                "threadblocks/ contains zero protobuf files",
            )
        if len(payload_paths) > limits.threadblock_files:
            raise TraceAdmissionError(
                "trace_payload_file_limit_exceeded",
                f"threadblock payload count {len(payload_paths)} exceeds "
                f"implementation limit {limits.threadblock_files}",
            )
        payload_size = 0
        for path in payload_paths:
            size = files[path][4]
            if size > limits.threadblock_bytes:
                raise TraceAdmissionError(
                    "trace_threadblock_size_limit_exceeded",
                    f"{path}: {size} bytes exceeds implementation limit "
                    f"{limits.threadblock_bytes}",
                )
            if payload_size > limits.total_threadblock_bytes - size:
                raise TraceAdmissionError(
                    "trace_payload_size_limit_exceeded",
                    "aggregate threadblock payload bytes exceed implementation "
                    f"limit {limits.total_threadblock_bytes}",
                )
            payload_size += size

        root = _validate_trace_root(root_bytes, root_path)
        launches = _ordered_launches(root)

        allocation_relative = "extra_info/allocations.json"
        allocation_snapshot: PinnedRegularSnapshot | None = None
        if allocation_relative in directories:
            raise TraceAdmissionError(
                "allocations_sidecar_malformed",
                f"{allocation_relative}: path is a directory",
            )
        allocation_metadata = files.get(allocation_relative)
        if allocation_metadata is not None:
            allocation_snapshot = _read_inventory_member(
                reader,
                allocation_relative,
                allocation_metadata,
                max_bytes=limits.allocations_sidecar_bytes,
                unreadable_reason="allocations_sidecar_malformed",
                size_reason="allocations_sidecar_size_limit_exceeded",
            )
            allocations, load_error = parse_allocations_sidecar_bytes(
                allocation_snapshot.payload,
                trace_dir / allocation_relative,
                max_entries=limits.allocation_entries,
                max_nesting=limits.allocation_json_nesting,
            )
            if load_error is not None:
                raise TraceAdmissionError(
                    load_error.typed_reason or "allocations_sidecar_malformed",
                    load_error.detail or "allocation sidecar validation failed",
                )
        else:
            allocations = None

        stats_metadata = _require_file_metadata(
            files,
            directories,
            "stats.csv",
            typed_reason="stats_unreadable",
        )
        stats_snapshot = _read_inventory_member(
            reader,
            "stats.csv",
            stats_metadata,
            max_bytes=limits.stats_bytes,
            unreadable_reason="stats_unreadable",
            size_reason="stats_size_limit_exceeded",
        )

        payloads_by_parent: dict[str, list[str]] = {}
        for path in payload_paths:
            parent = PurePosixPath(path).parent.as_posix()
            payloads_by_parent.setdefault(parent, []).append(path)

        threadblock_pb2 = _import_threadblock_pb2()
        consumed_paths: set[str] = set()
        threadblock_count = 0
        instruction_count = 0
        payload_digests: list[tuple[str, int, str]] = []
        launch_summaries: list[TraceLaunchSummary] = []
        for launch_index, (device_key, stream_key, kernel) in enumerate(launches):
            launch_dir = (
                f"threadblocks/device_{device_key}/stream_{stream_key}/"
                f"kernel_{kernel.id}"
            )
            launch_paths = payloads_by_parent.get(launch_dir, [])
            if not launch_paths:
                raise TraceAdmissionError(
                    "trace_launch_payload_missing",
                    f"root launch {(device_key, stream_key, kernel.id)} has no payload",
                )

            grid = (
                int(kernel.grid_dim.x),
                int(kernel.grid_dim.y),
                int(kernel.grid_dim.z),
            )
            grid_plane = grid[0] * grid[1]
            grid_total = grid_plane * grid[2]
            if grid_plane > I64_MAX or grid_total > I64_MAX:
                raise TraceAdmissionError(
                    "grid_arithmetic_overflow",
                    f"launch {(device_key, stream_key, kernel.id)} grid exceeds int64",
                )
            block_indices: set[int] = set()
            launch_instruction_count = 0
            for relative_path in launch_paths:
                snapshot = _read_inventory_member(
                    reader,
                    relative_path,
                    files[relative_path],
                    max_bytes=limits.threadblock_bytes,
                    unreadable_reason="trace_threadblock_unreadable",
                    size_reason="trace_threadblock_size_limit_exceeded",
                )
                raw = snapshot.payload
                path = trace_dir.joinpath(*PurePosixPath(relative_path).parts)
                if not raw:
                    raise TraceAdmissionError("trace_threadblock_empty", str(path))
                try:
                    preflight_threadblock_protobuf(raw)
                except MemoryError as exc:
                    raise TraceAdmissionError(
                        "trace_snapshot_resource_unavailable",
                        f"{path}: protobuf preflight exhausted memory",
                    ) from exc
                except ProtobufDecodedResourceLimitExceeded as exc:
                    raise TraceAdmissionError(
                        "protobuf_decoded_resource_limit_exceeded",
                        f"{path}: {exc}",
                    ) from exc
                except ProtobufWireMalformed as exc:
                    raise TraceAdmissionError(
                        "trace_threadblock_unparseable", f"{path}: {exc}"
                    ) from exc
                message = threadblock_pb2.threadblock()
                try:
                    message.ParseFromString(raw)
                except MemoryError as exc:
                    raise TraceAdmissionError(
                        "trace_snapshot_resource_unavailable",
                        f"{path}: protobuf decode exhausted memory",
                    ) from exc
                except Exception as exc:
                    raise TraceAdmissionError(
                        "trace_threadblock_unparseable", f"{path}: {exc}"
                    ) from exc
                if not message.HasField("block_id"):
                    raise TraceAdmissionError("block_id_missing", str(path))
                block = message.block_id
                coordinates = (int(block.x), int(block.y), int(block.z))
                block_index = (
                    coordinates[0]
                    + coordinates[1] * grid[0]
                    + coordinates[2] * grid_plane
                )
                if any(
                    coordinate < 0 or coordinate >= extent
                    for coordinate, extent in zip(coordinates, grid, strict=True)
                ):
                    raise TraceAdmissionError(
                        "block_id_out_of_range",
                        f"{path}: block {coordinates} flattens to {block_index}, "
                        f"grid volume {grid_total}",
                    )
                if block_index in block_indices:
                    raise TraceAdmissionError(
                        "block_id_duplicate",
                        f"{path}: duplicate block index {block_index}",
                    )
                block_indices.add(block_index)

                block_records = sum(
                    len(warp.instructions) for warp in message.warps.values()
                )
                if block_records <= 0:
                    raise TraceAdmissionError(
                        "trace_threadblock_empty",
                        f"{path} has zero instruction records",
                    )
                contract = validate_threadblock(
                    message,
                    label=str(path),
                    allocations=allocations,
                    address_carrying_v1=True,
                    expected_function_unique_id=int(kernel.function_unique_id),
                )
                if not contract.ok:
                    raise TraceAdmissionError(
                        contract.typed_reason or "trace_contract_invalid",
                        contract.detail
                        or "threadblock contract validation failed",
                    )
                consumed_paths.add(relative_path)
                if threadblock_count == U64_MAX:
                    raise TraceAdmissionError(
                        "trace_counter_overflow",
                        "threadblock count exceeds uint64",
                    )
                threadblock_count += 1
                if block_records > U64_MAX - instruction_count:
                    raise TraceAdmissionError(
                        "trace_counter_overflow",
                        "instruction count exceeds uint64",
                    )
                if block_records > U64_MAX - launch_instruction_count:
                    raise TraceAdmissionError(
                        "trace_counter_overflow",
                        "launch instruction count exceeds uint64",
                    )
                instruction_count += block_records
                launch_instruction_count += block_records
                payload_digests.append(
                    (
                        relative_path,
                        len(raw),
                        hashlib.sha256(raw).hexdigest(),
                    )
                )

            launch_summaries.append(
                _launch_summary(
                    launch_index,
                    device_key,
                    stream_key,
                    kernel,
                    launch_instruction_count,
                )
            )

        orphan = next(
            (path for path in payload_paths if path not in consumed_paths), None
        )
        if orphan is not None:
            raise TraceAdmissionError(
                "trace_payload_orphan",
                f"{trace_dir.joinpath(*PurePosixPath(orphan).parts)} does not "
                "belong to a root launch",
            )

        stats_row_count, expected, admitted = _parse_stats_bytes(
            stats_snapshot.payload,
            trace_dir / "stats.csv",
            max_rows=limits.stats_rows,
            max_columns=limits.stats_columns,
        )

        final_tree = _snapshot_tree(reader, limits, closure=True)
        final_directory_chain = _snapshot_lexical_directory_chain(
            reader,
            closure=True,
        )
        if (
            final_tree != initial_tree
            or final_directory_chain != initial_directory_chain
        ):
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                "complete trace tree or lexical ancestor identity changed during admission",
            )

        payload_manifest = hashlib.sha256()
        payload_manifest.update(b"nvbit-threadblock-snapshot-manifest-v1\n")
        for path, size, digest in sorted(payload_digests):
            payload_manifest.update(
                f"{path}\t{size}\t{digest}\n".encode("utf-8")
            )
        allocation_payload = (
            allocation_snapshot.payload if allocation_snapshot is not None else None
        )
        generation_source_paths = (
            "dynamic_trace.pb",
            "stats.csv",
            *payload_paths,
            *((allocation_relative,) if allocation_snapshot is not None else ()),
        )
        filesystem_generation_sha256 = _trace_filesystem_generation_sha256(
            initial_tree,
            generation_source_paths,
            initial_directory_chain,
        )
        admission = TraceAdmission(
            launch_count=len(launches),
            threadblock_count=threadblock_count,
            instruction_count=instruction_count,
            stats_row_count=stats_row_count,
            expected_record_count=expected,
            admitted_record_count=admitted,
            dynamic_trace_path=_serialized_dynamic_trace_path(trace_dir),
            dynamic_trace_size_bytes=len(root_bytes),
            dynamic_trace_sha256=hashlib.sha256(root_bytes).hexdigest(),
            allocations_sidecar_size_bytes=(
                len(allocation_payload) if allocation_payload is not None else None
            ),
            allocations_sidecar_sha256=(
                hashlib.sha256(allocation_payload).hexdigest()
                if allocation_payload is not None
                else None
            ),
            stats_size_bytes=len(stats_snapshot.payload),
            stats_sha256=hashlib.sha256(stats_snapshot.payload).hexdigest(),
            threadblock_payload_size_bytes=payload_size,
            threadblock_payload_manifest_sha256=payload_manifest.hexdigest(),
            trace_filesystem_generation_sha256=filesystem_generation_sha256,
            launches=tuple(launch_summaries),
        )
        sources = [
            _AdmittedSourceIdentity(
                relative_path="dynamic_trace.pb",
                metadata=root_snapshot.metadata,
                size_bytes=len(root_bytes),
                sha256=admission.dynamic_trace_sha256,
                max_bytes=limits.trace_root_bytes,
            ),
            _AdmittedSourceIdentity(
                relative_path="stats.csv",
                metadata=stats_snapshot.metadata,
                size_bytes=len(stats_snapshot.payload),
                sha256=admission.stats_sha256,
                max_bytes=limits.stats_bytes,
            ),
        ]
        if allocation_snapshot is not None:
            assert admission.allocations_sidecar_sha256 is not None
            sources.append(
                _AdmittedSourceIdentity(
                    relative_path=allocation_relative,
                    metadata=allocation_snapshot.metadata,
                    size_bytes=len(allocation_snapshot.payload),
                    sha256=admission.allocations_sidecar_sha256,
                    max_bytes=limits.allocations_sidecar_bytes,
                )
            )
        sources.extend(
            _AdmittedSourceIdentity(
                relative_path=relative_path,
                metadata=files[relative_path],
                size_bytes=size_bytes,
                sha256=digest,
                max_bytes=limits.threadblock_bytes,
            )
            for relative_path, size_bytes, digest in sorted(payload_digests)
        )
        return _TraceAdmissionSnapshot(
            admission=admission,
            sources=tuple(sources),
            threadblock_paths=payload_paths,
            allocations=allocations,
        )


def require_trace_admission(
    trace_dir: Path,
    *,
    limits: TraceAdmissionLimits = DEFAULT_TRACE_ADMISSION_LIMITS,
) -> TraceAdmission:
    """Admit exact bounded bytes from one anchored stable tree snapshot."""
    try:
        return _require_trace_admission_snapshot(
            trace_dir,
            limits=limits,
        ).admission
    except (ImmutableArtifactError, OSError, MemoryError) as exc:
        if _is_trace_resource_error(exc):
            raise TraceAdmissionError(
                "trace_snapshot_resource_unavailable",
                f"trace admission exhausted a required host resource: {exc}",
            ) from exc
        raise


def open_admitted_trace_sources(
    trace_dir: Path,
    *,
    expected_admission: TraceAdmission,
    limits: TraceAdmissionLimits = DEFAULT_TRACE_ADMISSION_LIMITS,
) -> AdmittedTraceSources:
    """Open a monitored capability for exactly one admitted trace generation."""
    if not isinstance(expected_admission, TraceAdmission):
        raise TypeError("expected_admission must be a TraceAdmission value")
    if not isinstance(limits, TraceAdmissionLimits):
        raise TypeError("limits must be a TraceAdmissionLimits value")
    trace_dir = _lexical_absolute_trace_dir(trace_dir)

    with ExitStack() as construction:
        try:
            reader = PinnedDirectoryReader(trace_dir)
        except (ImmutableArtifactError, OSError, MemoryError) as exc:
            _raise_trace_source_open_error(
                exc,
                detail_prefix="cannot open expected admitted trace root",
            )
        construction.callback(reader.close)

        allocation_expected = (
            expected_admission.allocations_sidecar_size_bytes is not None
            or expected_admission.allocations_sidecar_sha256 is not None
        )
        regular_paths = ["dynamic_trace.pb", "stats.csv"]
        if allocation_expected:
            regular_paths.append("extra_info/allocations.json")
        try:
            monitor = PinnedPathSetMonitor.from_reader(
                reader,
                regular_paths=regular_paths,
                optional_regular_paths=("extra_info/allocations.json",),
                recursive_directories=("threadblocks",),
                max_files=limits.threadblock_files,
                max_directories=limits.tree_directories,
                max_total_file_bytes=limits.total_threadblock_bytes,
                max_total_relative_path_bytes=limits.tree_relative_path_bytes,
            )
        except (ImmutableArtifactError, OSError, MemoryError) as exc:
            _raise_trace_source_open_error(
                exc,
                detail_prefix="cannot arm expected admitted trace monitor",
            )
        construction.callback(monitor.close)

        _require_source_monitor_quiet(
            monitor,
            detail_prefix="expected admitted trace changed before authentication",
        )
        try:
            snapshot = _require_trace_admission_snapshot(
                trace_dir,
                limits=limits,
                reader=reader,
            )
        except TraceAdmissionError as exc:
            preserved_authentication_reasons = {
                "trace_snapshot_resource_unavailable",
                "shared_memory_split_unavailable",
                "tracked_protobuf_corrupt",
                "protobuf_runtime_prerequisite_missing",
                "protobuf_bindings_mismatch",
            }
            if (
                exc.typed_reason in preserved_authentication_reasons
                or "limit_exceeded" in exc.typed_reason
            ):
                raise
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                f"fresh admitted trace authentication failed: {exc}",
            ) from exc
        except (ImmutableArtifactError, OSError, MemoryError) as exc:
            _raise_trace_source_open_error(
                exc,
                detail_prefix="fresh admitted trace authentication failed",
            )
        _require_source_monitor_quiet(
            monitor,
            detail_prefix="expected admitted trace changed during authentication",
        )
        observed_admission = snapshot.admission
        if (
            observed_admission.trace_admission_sha256
            != expected_admission.trace_admission_sha256
        ):
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                "fresh portable trace admission does not match the expected admission",
            )
        if (
            observed_admission.trace_filesystem_generation_sha256
            != expected_admission.trace_filesystem_generation_sha256
        ):
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                "fresh local trace generation does not match the expected admission",
            )
        if observed_admission != expected_admission:
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                "fresh trace admission disagrees with the expected admission",
            )
        _require_source_monitor_quiet(
            monitor,
            detail_prefix="expected admitted trace changed at authentication closure",
        )

        resources = construction.pop_all()

    try:
        capability = AdmittedTraceSources(
            admission=expected_admission,
            reader=reader,
            monitor=monitor,
            snapshot=snapshot,
            resources=resources,
        )
    except (ImmutableArtifactError, OSError, MemoryError) as exc:
        try:
            resources.close()
        except BaseException as close_error:
            try:
                exc.add_note(
                    f"admitted source construction cleanup failed: {close_error}"
                )
            except (AttributeError, TypeError):
                pass
        _raise_trace_source_open_error(
            exc,
            detail_prefix="cannot construct admitted trace capability",
        )
    except BaseException as exc:
        try:
            resources.close()
        except BaseException as close_error:
            try:
                exc.add_note(
                    f"admitted source construction cleanup failed: {close_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise
    try:
        capability.verify_unchanged()
    except (ImmutableArtifactError, OSError, MemoryError) as exc:
        _raise_trace_source_open_error(
            exc,
            detail_prefix="expected admitted trace changed before capability return",
        )
    except BaseException:
        capability.close()
        raise
    return capability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"authority scope: {TRACE_ADMISSION_SCOPE}",
    )
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = require_trace_admission(args.trace_dir)
    except TraceAdmissionError as exc:
        print(f"trace_admission: FAIL {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
