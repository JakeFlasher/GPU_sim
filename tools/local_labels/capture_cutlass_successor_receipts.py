#!/usr/bin/env python3
"""Capture one diagnostic CUTLASS successor NCU or NVBit phase.

The command is deliberately fail-closed.  It validates an operator-authored
successor preflight against the immutable 237f17c cohort, then checks the live
host before starting the selected process.  A receipt is written only after
label-driver or NVBit capture, artifact hashing, and semantic validation all
pass.  This tool never edits canonical or successor manifests.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import errno
import fcntl
import functools
import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import admit_cutlass_successor_capture as admission  # noqa: E402
from tools.local_labels import campaign_preflight  # noqa: E402
from tools.local_labels import delegated_cgroup  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import metric_registry  # noqa: E402
from tools.local_labels import ncu_action_executor as ncu_executor  # noqa: E402
from tools.local_labels import parse_ncu_metrics  # noqa: E402
from tools.local_labels import protected_paths  # noqa: E402
from tools.local_labels import sidecar_emitter  # noqa: E402
from tools.local_labels.capture_toolchain_contract import (  # noqa: E402
    parse_nvcc_release_banner,
    require_sol_cuda_release,
    ncu_version_matches,
)
from tools.local_labels import build_cutlass_successor_cohort as cohort  # noqa: E402
from tools.nvbit_tracer import capture_process as nvbit_capture_process  # noqa: E402
from tools.nvbit_tracer import external_tracer_build as external_build  # noqa: E402
from tools.nvbit_tracer import runtime_env as nvbit_runtime  # noqa: E402
from tools.nvbit_tracer.external_tracer import (  # noqa: E402
    EXTERNAL_TRACER_MAX_BYTES,
    ExternalTracerAuthority,
    ExternalTracerAuthorityError,
    require_external_tracer_authority,
    verify_retained_tracer_authority,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    DEFAULT_TRACE_ADMISSION_LIMITS,
    TraceAdmissionError,
    _import_trace_pb2,
    _require_trace_admission_snapshot,
    raw_trace_generation_limits,
    require_trace_admission,
)
from tools.nvbit_tracer.protobuf_wire_preflight import (  # noqa: E402
    ProtobufDecodedResourceLimitExceeded,
    ProtobufWireMalformed,
    preflight_trace_protobuf,
)

PINNED_COMMIT = admission.PINNED_BUNDLE_COMMIT
ARCH = "sm_86"
SKU = "rtx_3090_ti"
CUDA_RELEASE = {"major": 13, "minor": 2}
NVBIT_VERSION = admission.NVBIT_VERSION
NCU_VERSION = "2026.1.0"
GPU_NAME = "NVIDIA GeForce RTX 3090 Ti"
DRIVER_VERSION = "610.43.03"
SM_CLOCK_MHZ = 1350
MEM_CLOCK_MHZ = 10501
NVBIT_PB_LIB = ROOT / admission.NVBIT_PB_LIB_REL
NVBIT_VERSION_HEADER = ROOT / "tools/nvbit_tracer/nvbit_release/core/nvbit.h"
LABEL_DRIVER = ROOT / "tools/local_labels/label_driver.sh"
CUDA_NVCC = Path("/usr/local/cuda-13.2/bin/nvcc")
CUDA_CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
CAPTURE_TIMEOUT_S = 3600
CAPTURE_STREAM_LIMIT_BYTES = 8 * 1024 * 1024
CAPTURE_STREAM_READ_BYTES = 64 * 1024
CAPTURE_PROCESS_POLL_S = 0.05
CAPTURE_CLEANUP_TIMEOUT_S = 5.0
CAPTURE_PRODUCER_SELF_SHA256 = (
    "61a3934b4974848c750fdcc4081f79a6588e7e12043ff11ec285ca7b16f72588"
)
EXPECTED_TRACE_CONTRACT_ID = admission.TRACE_CONTRACT_ID
EXPECTED_MEMORY_PAYLOAD_CLASSES = admission.TRACE_MEMORY_PAYLOAD_CLASSES
_LABEL_COLLECTION_PATH = "/usr/local/cuda-13.2/bin:/usr/bin:/bin"
_LABEL_COLLECTION_ENVIRONMENT_NAMES = (
    admission.NCU_COLLECTION_OPTIONAL_ENVIRONMENT_NAMES
)
_NCU_HANDOFF_STDERR = (
    b"label_driver: NCU_LAUNCH_SET_POSTPROCESSING_UNIMPLEMENTED: "
    b"explicit launch-set report retained without a sidecar\n"
)
NVBIT_CAPTURE_STDERR_PROTOCOL = (
    "nvbit_1_8_sm86_empty_or_startup_diagnostics_v1"
)
_NVBIT_CAPTURE_STDOUT_NAME = "capture_process.stdout.bin"
_NVBIT_CAPTURE_STDERR_NAME = "capture_process.stderr.bin"
_NVBIT_CAPTURE_STDERR_PATTERN = re.compile(
    rb"WARNING: Do not call CUDA memory allocation in nvbit_at_ctx_init\(\)\. "
    rb"It will cause deadlocks\. Do them in nvbit_tool_init\(\)\. If you "
    rb"encounter deadlocks, remove CUDA API calls to debug\.\n"
    rb"INFO: tracer binary_version initialised per-context from device compute "
    rb"capability at tool-init: binary_version=86 \(ctx=0x[1-9a-f][0-9a-f]*\)\n"
    rb"INFO: cuFuncGetAttribute\(BINARY_VERSION\) returned attr_bv=0 "
    rb"\(CUresult=400\); falling back to device compute capability 8\.6 -> "
    rb"binary_version=86\n"
)
_NCU_INITIAL_ARTIFACT_NAMES = frozenset({
    "label_run_manifest_0_launch_set.json",
    "run_0_launch_set.ncu-rep",
})
_NCU_COMPLETE_ARTIFACT_NAMES = frozenset({
    *_NCU_INITIAL_ARTIFACT_NAMES,
    "label_0_0.json",
    "run_0_launch_set.launch_observation.json",
})
_NCU_PERCENTAGE_METRICS = (
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
)
_NCU_UNIT_METRICS = (
    "gpc__cycles_elapsed.max",
    "gpu__time_duration.sum",
)
CAPTURE_RESERVATION_PATH = (
    Path("/run/user")
    / str(os.getuid())
    / "gpu-simulator-cutlass-successor-capture.lock"
)
CAPTURE_RESERVATION_OWNER_LIMIT_BYTES = 16 * 1024
_INPUT_RESOURCE_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE, errno.ENOMEM})
_HASH_READ_BYTES = 1024 * 1024
OPERATOR_PREFLIGHT_MAX_BYTES = 64 * 1024
REPOSITORY_INPUT_MAX_BYTES = 1024 * 1024 * 1024
CAPTURE_SOURCE_MAX_BYTES = 4 * 1024 * 1024
TOOL_SOURCE_MAX_BYTES = 16 * 1024 * 1024
EXTERNAL_BUILD_RECEIPT_MAX_BYTES = 16 * 1024 * 1024
EXTERNAL_BUILD_ARTIFACT_IDENTITY_MAX_BYTES = 4 * 1024 * 1024
NVBIT_VERSION_HEADER_MAX_BYTES = 1024 * 1024
BOOT_ID_MAX_BYTES = 128
NCU_MANIFEST_MAX_BYTES = 64 * 1024
NCU_OBSERVATION_MAX_BYTES = 2 * 1024 * 1024
NCU_SIDECAR_MAX_BYTES = 2 * 1024 * 1024
TRACE_SUMMARY_MAX_BYTES = 2 * 1024 * 1024
CAPTURE_FAILURE_NAMESPACE_MAX_ENTRIES = 16
JSON_MAX_NODES = 16_384
JSON_MAX_CONTAINER_ITEMS = 4_096
JSON_MAX_DEPTH = 64
NCU_HANDOFF_MAX_TOTAL_BYTES = (
    parse_ncu_metrics.NCU_REPORT_MAX_BYTES
    + NCU_MANIFEST_MAX_BYTES
    + NCU_OBSERVATION_MAX_BYTES
    + NCU_SIDECAR_MAX_BYTES
)
FAILED_NCU_HANDOFF_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024


class CaptureError(RuntimeError):
    """A preflight or capture contract failed closed."""

    def __init__(
        self,
        detail: str,
        *,
        code: str = "capture_failed",
        stdout: bytes = b"",
        stderr: bytes = b"",
        process_id: int | None = None,
        started_at_utc: str | None = None,
        process_role: str | None = None,
        process_command: list[str] | None = None,
        process_environment: dict[str, str] | None = None,
        process_tool_sha256: str | None = None,
        containment: delegated_cgroup.CgroupContainmentEvidence | None = None,
    ) -> None:
        self.detail = detail
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.process_id = process_id
        self.started_at_utc = started_at_utc
        self.process_role = process_role
        self.process_command = process_command
        self.process_environment = process_environment
        self.process_tool_sha256 = process_tool_sha256
        self.containment = containment
        super().__init__(detail)


@dataclass(frozen=True)
class NvbitExternalBuildAuthority:
    bundle_path: Path
    expected_tracer_sha256: str
    source_commit_sha1: str
    source_tree_sha1: str
    source_inventory_sha256: str
    source_manifest_payload: bytes
    build_receipt_payload: bytes
    artifact_identity_payload: bytes

    def provenance(self) -> dict[str, Any]:
        return {
            "authority_kind": external_build.REVIEWED_AUTHORITY_KIND,
            "expected_tracer_sha256": self.expected_tracer_sha256,
            "source_commit_sha1": self.source_commit_sha1,
            "source_tree_sha1": self.source_tree_sha1,
            "source_inventory_sha256": self.source_inventory_sha256,
            "source_manifest_sha256": sha256_bytes(self.source_manifest_payload),
            "build_receipt_sha256": sha256_bytes(self.build_receipt_payload),
            "artifact_identity_sha256": sha256_bytes(
                self.artifact_identity_payload
            ),
        }

    def documents(self) -> tuple[tuple[str, str, bytes, int], ...]:
        return (
            (
                "external_build_source_manifest",
                "nvbit_external_build_source_manifest.json",
                self.source_manifest_payload,
                external_build.MAX_SOURCE_BYTES,
            ),
            (
                "external_build_receipt",
                "nvbit_external_build_receipt.json",
                self.build_receipt_payload,
                EXTERNAL_BUILD_RECEIPT_MAX_BYTES,
            ),
            (
                "external_build_artifact_identity",
                "nvbit_external_build_artifact_identity.json",
                self.artifact_identity_payload,
                EXTERNAL_BUILD_ARTIFACT_IDENTITY_MAX_BYTES,
            ),
        )


@dataclass(frozen=True)
class RetainedNvbitBuildDocument:
    role: str
    path: Path
    payload: bytes
    retained: immutable.RetainedRegular

    def artifact(self, root: Path) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path.relative_to(root).as_posix(),
            "sha256": self.retained.sha256,
            "size_bytes": self.retained.size_bytes,
        }


def _metadata_snapshot(metadata: os.stat_result) -> immutable.MetadataSnapshot:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _typed_input_error(
    label: str,
    exc: BaseException,
    *,
    action: str,
) -> CaptureError:
    if isinstance(exc, FileNotFoundError) or (
        isinstance(exc, OSError) and exc.errno == errno.ENOENT
    ):
        code = "capture_input_missing"
    elif isinstance(exc, immutable.ImmutableArtifactLimitError):
        code = "capture_input_limit_exceeded"
    elif isinstance(exc, (MemoryError, immutable.ImmutableArtifactResourceError)) or (
        isinstance(exc, OSError) and exc.errno in _INPUT_RESOURCE_ERRNOS
    ):
        code = "capture_input_resource_unavailable"
    elif isinstance(exc, immutable.ImmutableArtifactError):
        code = "capture_generation_changed"
    else:
        code = "capture_input_io_failed"
    return CaptureError(f"{label} {action} failed: {exc}", code=code)


def _open_bounded_regular(
    path: Path,
    label: str,
    *,
    max_bytes: int,
) -> immutable.RetainedRegular:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise CaptureError(
            f"{label} byte limit is invalid",
            code="capture_input_limit_invalid",
        )
    try:
        return immutable.open_retained_regular(path, max_bytes=max_bytes)
    except (
        MemoryError,
        OSError,
        immutable.ImmutableArtifactError,
    ) as exc:
        error = _typed_input_error(label, exc, action="retained open")
        raise error from exc


@contextlib.contextmanager
def _retained_regular_capability(
    retained: immutable.RetainedRegular,
    label: str,
):
    """Enter and close an exact retained file with typed transition errors."""

    try:
        retained.__enter__()
    except (
        MemoryError,
        OSError,
        immutable.ImmutableArtifactError,
    ) as exc:
        try:
            retained.close()
        except BaseException as close_error:
            try:
                exc.add_note(f"{label} retained-open cleanup failed: {close_error}")
            except (AttributeError, TypeError):
                pass
        error = _typed_input_error(label, exc, action="retained entry")
        raise error from exc
    try:
        yield retained
    except BaseException as primary:
        retained.__exit__(type(primary), primary, primary.__traceback__)
        raise
    else:
        try:
            retained.__exit__(None, None, None)
        except (
            MemoryError,
            OSError,
            immutable.ImmutableArtifactError,
        ) as exc:
            error = _typed_input_error(label, exc, action="retained close")
            raise error from exc


@contextlib.contextmanager
def _retained_bounded_regular(
    path: Path,
    label: str,
    *,
    max_bytes: int,
):
    """Enter and close a bounded retained file with typed transition errors."""

    retained = _open_bounded_regular(path, label, max_bytes=max_bytes)
    with _retained_regular_capability(retained, label) as entered:
        yield entered


def _read_retained_bytes(
    retained: immutable.RetainedRegular,
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    try:
        return retained.read_bytes(max_bytes=max_bytes)
    except (
        MemoryError,
        OSError,
        immutable.ImmutableArtifactError,
    ) as exc:
        error = _typed_input_error(label, exc, action="retained read")
        raise error from exc


def _require_retained_executable(
    retained: immutable.RetainedRegular,
    label: str,
) -> None:
    try:
        mode = os.fstat(retained.descriptor).st_mode
    except OSError as exc:
        error = _typed_input_error(label, exc, action="mode inspection")
        raise error from exc
    if mode & 0o111 == 0:
        raise CaptureError(
            f"{label} is not executable",
            code="capture_input_integrity_failed",
        )


def _preflight_json_structure(
    payload: bytes,
    *,
    label: str,
    max_nodes: int = JSON_MAX_NODES,
    max_container_items: int = JSON_MAX_CONTAINER_ITEMS,
    max_depth: int = JSON_MAX_DEPTH,
) -> None:
    """Bound JSON nodes and container cardinality before ``json.loads``."""

    for name, value in (
        ("node", max_nodes),
        ("container-item", max_container_items),
        ("depth", max_depth),
    ):
        if type(value) is not int or value <= 0:
            raise CaptureError(
                f"{label} JSON {name} limit is invalid",
                code="capture_json_limit_invalid",
            )
    length = len(payload)
    nodes = 0

    def invalid(detail: str) -> None:
        raise CaptureError(
            f"{label} is not valid JSON: {detail}",
            code="capture_input_integrity_failed",
        )

    def skip_space(index: int) -> int:
        while index < length and payload[index] in b" \t\r\n":
            index += 1
        return index

    def parse_string(index: int) -> int:
        if index >= length or payload[index] != 0x22:
            invalid("object key or string is malformed")
        index += 1
        while index < length:
            value = payload[index]
            if value == 0x22:
                return index + 1
            if value < 0x20:
                invalid("string contains an unescaped control byte")
            if value != 0x5C:
                index += 1
                continue
            index += 1
            if index >= length:
                invalid("string escape is truncated")
            escape = payload[index]
            if escape == ord("u"):
                if index + 4 >= length or any(
                    digit not in b"0123456789abcdefABCDEF"
                    for digit in payload[index + 1 : index + 5]
                ):
                    invalid("Unicode escape is malformed")
                index += 5
            elif escape in b'"\\/bfnrt':
                index += 1
            else:
                invalid("string escape is malformed")
        invalid("string is unterminated")
        raise AssertionError("unreachable")

    def parse_number(index: int) -> int:
        if payload[index] == ord("-"):
            index += 1
            if index >= length:
                invalid("number is truncated")
        if payload[index] == ord("0"):
            index += 1
            if index < length and ord("0") <= payload[index] <= ord("9"):
                invalid("number has a leading zero")
        elif ord("1") <= payload[index] <= ord("9"):
            while index < length and ord("0") <= payload[index] <= ord("9"):
                index += 1
        else:
            invalid("number is malformed")
        if index < length and payload[index] == ord("."):
            index += 1
            start = index
            while index < length and ord("0") <= payload[index] <= ord("9"):
                index += 1
            if index == start:
                invalid("number fraction is malformed")
        if index < length and payload[index] in b"eE":
            index += 1
            if index < length and payload[index] in b"+-":
                index += 1
            start = index
            while index < length and ord("0") <= payload[index] <= ord("9"):
                index += 1
            if index == start:
                invalid("number exponent is malformed")
        return index

    def parse_value(index: int, depth: int) -> int:
        nonlocal nodes
        if depth > max_depth:
            raise CaptureError(
                f"{label} JSON depth limit exceeded: {depth} > {max_depth}",
                code="capture_json_node_limit_exceeded",
            )
        index = skip_space(index)
        if index >= length:
            invalid("value is absent")
        nodes += 1
        if nodes > max_nodes:
            raise CaptureError(
                f"{label} JSON node limit exceeded: {nodes} > {max_nodes}",
                code="capture_json_node_limit_exceeded",
            )
        token = payload[index]
        if token == ord('"'):
            return parse_string(index)
        if token == ord("{"):
            index = skip_space(index + 1)
            items = 0
            if index < length and payload[index] == ord("}"):
                return index + 1
            while True:
                index = parse_string(index)
                index = skip_space(index)
                if index >= length or payload[index] != ord(":"):
                    invalid("object key lacks a colon")
                items += 1
                if items > max_container_items:
                    raise CaptureError(
                        f"{label} JSON object cardinality limit exceeded: "
                        f"{items} > {max_container_items}",
                        code="capture_json_cardinality_limit_exceeded",
                    )
                index = parse_value(index + 1, depth + 1)
                index = skip_space(index)
                if index < length and payload[index] == ord("}"):
                    return index + 1
                if index >= length or payload[index] != ord(","):
                    invalid("object separator is malformed")
                index = skip_space(index + 1)
        if token == ord("["):
            index = skip_space(index + 1)
            items = 0
            if index < length and payload[index] == ord("]"):
                return index + 1
            while True:
                items += 1
                if items > max_container_items:
                    raise CaptureError(
                        f"{label} JSON array cardinality limit exceeded: "
                        f"{items} > {max_container_items}",
                        code="capture_json_cardinality_limit_exceeded",
                    )
                index = parse_value(index, depth + 1)
                index = skip_space(index)
                if index < length and payload[index] == ord("]"):
                    return index + 1
                if index >= length or payload[index] != ord(","):
                    invalid("array separator is malformed")
                index = skip_space(index + 1)
        for literal in (b"true", b"false", b"null"):
            if payload.startswith(literal, index):
                return index + len(literal)
        if token == ord("-") or ord("0") <= token <= ord("9"):
            return parse_number(index)
        invalid("value token is malformed")
        raise AssertionError("unreachable")

    try:
        end = skip_space(parse_value(0, 1))
    except MemoryError as exc:
        raise CaptureError(
            f"{label} JSON structural preflight exhausted memory",
            code="capture_input_resource_unavailable",
        ) from exc
    if end != length:
        invalid("trailing content is present")


def _strict_json_value(payload: bytes, *, label: str) -> Any:
    _preflight_json_structure(payload, label=label)
    try:
        return immutable.strict_json_loads(payload, label=label)
    except MemoryError as exc:
        raise CaptureError(
            f"{label} JSON materialization exhausted memory",
            code="capture_input_resource_unavailable",
        ) from exc
    except immutable.ImmutableArtifactError as exc:
        raise CaptureError(
            f"{label} is not valid JSON: {exc}",
            code="capture_input_integrity_failed",
        ) from exc


def canonical_sha(value: Any) -> str:
    return admission.canonical_sha(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_retained_snapshot(
    reader: immutable.PinnedDirectoryReader,
    relative_path: str,
    expected_metadata: immutable.MetadataSnapshot,
    *,
    label: str,
    max_bytes: int,
) -> tuple[str, int]:
    """Hash one exact descriptor-relative inode without materializing it."""

    pinned = None
    try:
        immutable.safe_relative_path(relative_path)
        reader.verify_path()
        pinned = immutable._open_relative_regular_at(
            reader.descriptor,
            relative_path,
        )
        before = _metadata_snapshot(os.fstat(pinned.descriptor))
        if before != expected_metadata:
            raise immutable.ImmutableArtifactError(
                f"pinned regular-file generation changed: {relative_path}"
            )
        size_bytes = before[4]
        if size_bytes > max_bytes:
            raise immutable.ImmutableArtifactLimitError(
                f"bounded read exceeds max_bytes={max_bytes}: {size_bytes}"
            )
        digest = hashlib.sha256()
        observed_bytes = 0
        os.lseek(pinned.descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(pinned.descriptor, _HASH_READ_BYTES)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise immutable.ImmutableArtifactLimitError(
                    f"bounded read exceeds max_bytes={max_bytes}: "
                    f"> {max_bytes}"
                )
            digest.update(chunk)
        after = _metadata_snapshot(os.fstat(pinned.descriptor))
        if after != before or observed_bytes != size_bytes:
            raise immutable.ImmutableArtifactError(
                f"pinned regular-file generation changed: {relative_path}"
            )
        pinned.verify_path()
        reader.verify_path()
        return digest.hexdigest(), size_bytes
    except (
        MemoryError,
        OSError,
        immutable.ImmutableArtifactError,
    ) as exc:
        error = _typed_input_error(label, exc, action="descriptor-bound hash")
        raise error from exc
    finally:
        if pinned is not None:
            try:
                pinned.close()
            except (
                MemoryError,
                OSError,
                immutable.ImmutableArtifactError,
            ) as close_error:
                active = sys.exception()
                if active is not None:
                    try:
                        active.add_note(
                            f"{label} descriptor close failed: {close_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
                else:
                    error = _typed_input_error(
                        label,
                        close_error,
                        action="descriptor close",
                    )
                    raise error from close_error


def _bounded_directory_names(
    parent: immutable.PinnedDirectoryReader,
    *,
    max_entries: int,
    label: str,
) -> tuple[str, ...]:
    if type(max_entries) is not int or max_entries < 0:
        raise CaptureError(
            f"{label} namespace cardinality limit is invalid",
            code="capture_namespace_limit_invalid",
        )
    observed: list[str] = []
    try:
        parent.verify_path()
        with immutable._scandir_independent(parent.descriptor) as entries:
            for entry in entries:
                if len(observed) >= max_entries:
                    raise CaptureError(
                        f"{label} namespace differs: cardinality limit exceeded "
                        f"({max_entries})",
                        code="capture_namespace_cardinality_limit_exceeded",
                    )
                observed.append(entry.name)
        parent.verify_path()
    except CaptureError:
        raise
    except MemoryError as exc:
        raise CaptureError(
            f"cannot inspect {label} namespace: memory exhausted",
            code="capture_namespace_resource_unavailable",
        ) from exc
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"cannot inspect {label} namespace: {exc}",
            code=(
                "capture_namespace_resource_unavailable"
                if isinstance(exc, OSError) and exc.errno in _INPUT_RESOURCE_ERRNOS
                else "capture_generation_changed"
            ),
        ) from exc
    return tuple(observed)


def _artifact_byte_limit(role: str, relative_path: str = "") -> int:
    if role == "ncu_report":
        return parse_ncu_metrics.NCU_REPORT_MAX_BYTES
    if role == "label_run_manifest":
        return NCU_MANIFEST_MAX_BYTES
    if role == "ncu_launch_observation":
        return NCU_OBSERVATION_MAX_BYTES
    if role == "label_sidecar":
        return NCU_SIDECAR_MAX_BYTES
    if role == "phase_tool":
        return EXTERNAL_TRACER_MAX_BYTES
    if role == "trace_summary":
        return TRACE_SUMMARY_MAX_BYTES
    if role == "retained_failed_handoff_artifact":
        return REPOSITORY_INPUT_MAX_BYTES
    try:
        policy = _capture_inventory_policy()
        max_file_bytes = int(policy["max_file_bytes"])
        path_limits = policy["max_file_bytes_by_path"]
        if not isinstance(path_limits, dict):
            raise TypeError("trace path byte limits are not an object")
        return min(
            max_file_bytes,
            int(path_limits.get(relative_path, max_file_bytes)),
        )
    except (MemoryError, TypeError, ValueError, KeyError) as exc:
        raise CaptureError(
            f"artifact byte-limit policy is unavailable: {exc}",
            code="capture_input_resource_unavailable",
        ) from exc


def _host_identity_sha256(live: dict[str, Any]) -> str:
    return canonical_sha({
        "gpu": live["gpu_name"],
        "smi": live["smi_stdout_sha256"],
        "gpu_uuid_sha256": live["gpu_uuid_sha256"],
        "driver_version": live["driver_version"],
        "identity_stdout_sha256": live["identity_stdout_sha256"],
    })


def _capture_reservation_owner_detail(descriptor: int) -> str:
    try:
        payload = os.pread(
            descriptor,
            CAPTURE_RESERVATION_OWNER_LIMIT_BYTES + 1,
            0,
        )
        if len(payload) > CAPTURE_RESERVATION_OWNER_LIMIT_BYTES:
            return "owner metadata exceeds its byte limit"
        owner = _strict_json_value(
            payload,
            label="capture reservation owner",
        )
        if type(owner) is not dict:
            return "owner metadata is unavailable"
        fields = {
            key: owner.get(key)
            for key in ("pid", "phase", "matrix_key", "variant_id")
        }
        return json.dumps(fields, sort_keys=True, separators=(",", ":"))
    except (
        OSError,
        CaptureError,
        immutable.ImmutableArtifactError,
    ):
        return "owner metadata is unavailable"


def _write_capture_reservation_owner(
    descriptor: int,
    owner: dict[str, Any],
) -> None:
    payload = immutable.pretty_json_bytes(owner)
    if len(payload) > CAPTURE_RESERVATION_OWNER_LIMIT_BYTES:
        raise CaptureError(
            "capture reservation owner metadata exceeds its byte limit",
            code="capture_phase_reservation_invalid",
        )
    try:
        os.ftruncate(descriptor, 0)
        offset = 0
        while offset < len(payload):
            written = os.pwrite(descriptor, payload[offset:], offset)
            if written <= 0:
                raise OSError(errno.EIO, "reservation owner write made no progress")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise CaptureError(
            f"capture reservation owner metadata could not be written: {exc}",
            code=(
                "capture_phase_reservation_resource_unavailable"
                if exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.ENOSPC}
                else "capture_phase_reservation_failed"
            ),
        ) from exc


@contextlib.contextmanager
def _capture_phase_reservation(
    *,
    matrix_key: str,
    variant_id: str,
    phase: str,
    output_root: Path,
    reservation_path: Path | None = None,
):
    """Serialize separate-process NCU/NVBit phases across invocations.

    The persistent owner text is diagnostic only.  ``flock`` is the authority,
    so an owner killed without cleanup leaves an unlocked, safely reusable file.
    """

    path = CAPTURE_RESERVATION_PATH if reservation_path is None else reservation_path
    if not path.is_absolute() or len(path.parts) < 2:
        raise CaptureError(
            "capture phase reservation path must be absolute",
            code="capture_phase_reservation_invalid",
        )
    parent = path.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise CaptureError(
            f"capture phase reservation directory is unavailable: {parent}: {exc}",
            code="capture_phase_reservation_resource_unavailable",
        ) from exc
    if (
        parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.getuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
    ):
        raise CaptureError(
            f"capture phase reservation directory is not private: {parent}",
            code="capture_phase_reservation_invalid",
        )

    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CaptureError(
            f"capture phase reservation could not be opened: {exc}",
            code=(
                "capture_phase_reservation_resource_unavailable"
                if exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.ENOSPC}
                else "capture_phase_reservation_failed"
            ),
        ) from exc

    locked = False
    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = os.stat(path, follow_symlinks=False)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_mode)
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or descriptor_metadata.st_nlink != 1
            or descriptor_metadata.st_uid != os.getuid()
            or stat.S_IMODE(descriptor_metadata.st_mode) != 0o600
            or identity(descriptor_metadata) != identity(path_metadata)
        ):
            raise CaptureError(
                "capture phase reservation file identity is unsafe",
                code="capture_phase_reservation_invalid",
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:
            owner_detail = _capture_reservation_owner_detail(descriptor)
            raise CaptureError(
                "another CUTLASS capture phase owns the reservation: "
                f"{owner_detail}",
                code="capture_phase_reserved",
            ) from exc
        owner = {
            "schema_version": 1,
            "artifact_kind": "cutlass_successor_capture_phase_reservation",
            "boot_id_sha256": _boot_id_sha256(),
            "pid": os.getpid(),
            "phase": phase,
            "matrix_key": matrix_key,
            "variant_id": variant_id,
            "output_root": str(output_root),
            "acquired_at_utc": (
                dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            ),
        }
        _write_capture_reservation_owner(descriptor, owner)
        yield owner
    except CaptureError:
        raise
    except OSError as exc:
        raise CaptureError(
            f"capture phase reservation failed: {exc}",
            code=(
                "capture_phase_reservation_resource_unavailable"
                if exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.ENOSPC}
                else "capture_phase_reservation_failed"
            ),
        ) from exc
    finally:
        active = sys.exception()
        cleanup_errors: list[OSError] = []
        if locked:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError as exc:
                cleanup_errors.append(exc)
        try:
            os.close(descriptor)
        except OSError as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            detail = "; ".join(str(error) for error in cleanup_errors)
            if active is not None:
                try:
                    active.add_note(
                        f"capture phase reservation cleanup failed: {detail}"
                    )
                except (AttributeError, TypeError):
                    pass
            else:
                raise CaptureError(
                    f"capture phase reservation cleanup failed: {detail}",
                    code="capture_phase_reservation_cleanup_failed",
                )


def _regular(path: Path, label: str) -> None:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise CaptureError(f"{label} unavailable: {path}") from exc
    if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
        raise CaptureError(f"{label} is not a single regular file: {path}")


def _validate_capture_producer_identity(
    source: Path,
    retained: immutable.RetainedRegular,
) -> tuple[Path, str]:
    payload = _read_retained_bytes(
        retained,
        "capture producer",
        max_bytes=CAPTURE_SOURCE_MAX_BYTES,
    )
    try:
        declared = admission._capture_producer_self_digest(payload)
    except admission.AdmissionError as exc:
        raise CaptureError(
            f"capture producer immutable identity failed: {exc}",
            code="capture_input_integrity_failed",
        ) from exc
    if declared != CAPTURE_PRODUCER_SELF_SHA256:
        raise CaptureError(
            "loaded capture producer disagrees with its named source self-seal",
            code="capture_input_integrity_failed",
        )
    if declared != admission.CAPTURE_PRODUCER_SELF_SHA256_AUTHORITY:
        raise CaptureError(
            "capture producer differs from independent self-seal authority",
            code="capture_input_integrity_failed",
        )
    return source, retained.sha256


def _capture_producer_identity() -> tuple[Path, str]:
    source = Path(__file__).resolve(strict=True)
    try:
        with _retained_bounded_regular(
            source,
            "capture producer",
            max_bytes=CAPTURE_SOURCE_MAX_BYTES,
        ) as retained:
            return _validate_capture_producer_identity(source, retained)
    except CaptureError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        error = _typed_input_error(
            "capture producer",
            exc,
            action="immutable identity validation",
        )
        raise error from exc


def _absolute_external(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CaptureError(f"{label} must be an absolute path")
    candidate = Path(path.anchor)
    for component in path.parts[1:]:
        candidate /= component
        if candidate.is_symlink():
            raise CaptureError(f"{label} path contains a symlink: {candidate}")
    if path.is_symlink():
        raise CaptureError(f"{label} is a symlink: {path}")
    resolved = path.resolve(strict=False)
    resolved_parts = resolved.parts
    if any(
        resolved_parts[index : index + 2] == (".claude", "worktrees")
        for index in range(len(resolved_parts) - 1)
    ):
        raise CaptureError(f"{label} must not come from a Claude worktree")
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise CaptureError(f"{label} must be external to the repository: {path}")
    for ancestor in resolved.parents:
        marker = ancestor / ".git"
        try:
            marker.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise CaptureError(
                f"{label} Git-worktree status is unavailable: {ancestor}"
            ) from exc
        raise CaptureError(f"{label} must come from a non-Git build tree")
    repository_status = protected_paths.git_repository_status(resolved)
    if repository_status == protected_paths.GIT_REPOSITORY_STATUS_UNAVAILABLE:
        raise CaptureError(f"{label} Git-repository status is unavailable")
    if repository_status != protected_paths.GIT_REPOSITORY_NOT_FOUND:
        raise CaptureError(f"{label} must come from a non-Git build tree")
    return resolved


def _require_nvbit_tracer_authority(
    path: Path | None,
    sha256: str | None,
) -> ExternalTracerAuthority:
    """Require the explicit CLI pair without an environment fallback."""

    try:
        return require_external_tracer_authority(
            path,
            sha256,
            repo_root=ROOT,
            environment={},
        )
    except ExternalTracerAuthorityError as exc:
        raise CaptureError(
            f"external NVBit tracer authority is invalid: {exc}",
            code="capture_input_integrity_failed",
        ) from exc


def _require_nvbit_runtime_manifest_authority(
    path: Path | None,
    sha256: str | None,
) -> tuple[Path, str]:
    """Require one explicit external runtime manifest path and exact digest."""

    if path is None or sha256 is None:
        raise CaptureError(
            "NVBit capture requires runtime manifest path and SHA-256 authority",
            code="capture_input_integrity_failed",
        )
    if (
        type(sha256) is not str
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise CaptureError(
            "NVBit runtime manifest SHA-256 must be exact lowercase 64-hex",
            code="capture_input_integrity_failed",
        )
    resolved = _absolute_external(Path(path), "NVBit runtime manifest")
    try:
        metadata = resolved.lstat()
    except OSError as exc:
        raise CaptureError(
            f"NVBit runtime manifest is unavailable: {exc}",
            code="capture_input_integrity_failed",
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 1 <= metadata.st_size <= nvbit_runtime.RUNTIME_MANIFEST_MAX_BYTES
    ):
        raise CaptureError(
            "NVBit runtime manifest must be one nonempty, singly linked bounded file",
            code="capture_input_integrity_failed",
        )
    return resolved, sha256


def _retain_nvbit_external_build_authority(
    stack: contextlib.ExitStack,
    *,
    bundle_path: Path | None,
    reviewed_tracer_sha256: str | None,
    tracer_authority: ExternalTracerAuthority,
    tracer_retained: immutable.RetainedRegular,
) -> NvbitExternalBuildAuthority:
    if bundle_path is None or reviewed_tracer_sha256 is None:
        raise CaptureError(
            "NVBit capture requires an admitted external build bundle and "
            "independently reviewed tracer SHA-256",
            code="capture_input_integrity_failed",
        )
    if (
        type(reviewed_tracer_sha256) is not str
        or len(reviewed_tracer_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in reviewed_tracer_sha256
        )
    ):
        raise CaptureError(
            "independently reviewed tracer SHA-256 must be exact lowercase 64-hex",
            code="capture_input_integrity_failed",
        )
    bundle = _absolute_external(Path(bundle_path), "admitted external build bundle")
    try:
        reader = stack.enter_context(immutable.PinnedDirectoryReader(bundle))
        snapshot = stack.enter_context(
            reader.retain_tree_generation(
                require_readonly_files=True,
                require_readonly_directories=True,
                max_files=len(external_build.BUNDLE_FILES),
                max_directories=1,
                max_total_file_bytes=(
                    external_build.MAX_SOURCE_BYTES
                    + EXTERNAL_BUILD_RECEIPT_MAX_BYTES
                    + EXTERNAL_BUILD_ARTIFACT_IDENTITY_MAX_BYTES
                    + EXTERNAL_TRACER_MAX_BYTES
                ),
                max_total_relative_path_bytes=64 * 1024,
            )
        )
        if {relative for relative, _metadata in snapshot.files} != set(
            external_build.BUNDLE_FILES
        ) or {relative for relative, _metadata in snapshot.directories} != {"."}:
            raise CaptureError(
                "admitted external build bundle inventory is not exact",
                code="capture_input_integrity_failed",
            )
        artifact = external_build.verify_bundle(
            bundle,
            repo_root=ROOT,
            expected_tracer_sha256=reviewed_tracer_sha256,
            cuobjdump=CUDA_CUOBJDUMP,
            timeout_seconds=external_build.DEFAULT_TIMEOUT_SECONDS,
        )
        reader.verify_path()
        metadata = dict(snapshot.files)
        source_payload = reader.read_snapshot(
            external_build.SOURCE_MANIFEST_NAME,
            max_bytes=external_build.MAX_SOURCE_BYTES,
            expected_metadata=metadata[external_build.SOURCE_MANIFEST_NAME],
        ).payload
        build_payload = reader.read_snapshot(
            external_build.BUILD_RECEIPT_NAME,
            max_bytes=EXTERNAL_BUILD_RECEIPT_MAX_BYTES,
            expected_metadata=metadata[external_build.BUILD_RECEIPT_NAME],
        ).payload
        artifact_payload = reader.read_snapshot(
            external_build.ARTIFACT_IDENTITY_NAME,
            max_bytes=EXTERNAL_BUILD_ARTIFACT_IDENTITY_MAX_BYTES,
            expected_metadata=metadata[external_build.ARTIFACT_IDENTITY_NAME],
        ).payload
        tracer_metadata = metadata[external_build.TRACER_NAME]
        bundle_tracer_size = tracer_metadata[4]
        tracer_identity = artifact["artifact"]
        source_identity = artifact["source_identity"]
        if (
            tracer_authority.sha256 != reviewed_tracer_sha256
            or tracer_retained.sha256 != reviewed_tracer_sha256
            or tracer_retained.size_bytes != tracer_identity["size_bytes"]
            or tracer_identity
            != {
                "name": external_build.TRACER_NAME,
                "sha256": reviewed_tracer_sha256,
                "size_bytes": bundle_tracer_size,
                "source_kind": external_build.EXTERNAL_TRACER_SOURCE_KIND,
            }
            or sha256_bytes(source_payload)
            != artifact["source_manifest"]["sha256"]
            or len(source_payload) != artifact["source_manifest"]["size_bytes"]
            or sha256_bytes(build_payload)
            != artifact["build_receipt"]["sha256"]
            or len(build_payload) != artifact["build_receipt"]["size_bytes"]
        ):
            raise CaptureError(
                "runtime tracer differs from independently admitted external build",
                code="capture_input_integrity_failed",
            )
        reader.verify_regular(external_build.TRACER_NAME, tracer_metadata)
        reader.verify_path()
    except CaptureError:
        raise
    except (
        OSError,
        MemoryError,
        immutable.ImmutableArtifactError,
        external_build.ExternalTracerBuildAuthorityError,
        external_build.ExternalTracerBuildProcessError,
    ) as exc:
        raise CaptureError(
            f"admitted external build authority is invalid: {exc}",
            code="capture_input_integrity_failed",
        ) from exc
    return NvbitExternalBuildAuthority(
        bundle_path=bundle,
        expected_tracer_sha256=reviewed_tracer_sha256,
        source_commit_sha1=source_identity["head_commit_sha1"],
        source_tree_sha1=source_identity["head_tree_sha1"],
        source_inventory_sha256=source_identity["inventory_sha256"],
        source_manifest_payload=source_payload,
        build_receipt_payload=build_payload,
        artifact_identity_payload=artifact_payload,
    )


def _verify_runtime_manifest_checkpoint(
    retained: immutable.RetainedRegular | None,
    expected_sha256: str | None,
    *,
    stage: str,
    initial: bool = False,
) -> None:
    if retained is None or expected_sha256 is None:
        raise CaptureError(
            f"NVBit runtime manifest {stage} authority is absent",
            code="capture_input_integrity_failed" if initial else "capture_generation_changed",
        )
    try:
        retained.verify_unchanged()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"NVBit runtime manifest {stage} verification failed: {exc}",
            code="capture_input_integrity_failed" if initial else "capture_generation_changed",
        ) from exc
    if retained.sha256 != expected_sha256:
        raise CaptureError(
            f"NVBit runtime manifest {stage} digest differs from authority",
            code="capture_input_integrity_failed" if initial else "capture_generation_changed",
        )


def _verify_external_tracer_checkpoint(
    retained: immutable.RetainedRegular,
    authority: ExternalTracerAuthority,
    *,
    stage: str,
    initial: bool = False,
) -> None:
    """Rehash and rebind the retained external source at a stage boundary."""

    try:
        retained.verify_unchanged()
        verify_retained_tracer_authority(retained, authority)
    except (
        OSError,
        immutable.ImmutableArtifactError,
        ExternalTracerAuthorityError,
    ) as exc:
        raise CaptureError(
            f"external NVBit tracer {stage} verification failed: {exc}",
            code=(
                "capture_input_integrity_failed"
                if initial
                else "capture_generation_changed"
            ),
        ) from exc


def _verify_retained_phase_tool_checkpoint(
    retained: immutable.RetainedRegular,
    expected_sha256: str,
    *,
    stage: str,
) -> None:
    """Rehash the copied launch capability and bind it to the source digest."""

    try:
        retained.verify_unchanged()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"retained NVBit phase tool {stage} verification failed: {exc}",
            code="capture_generation_changed",
        ) from exc
    if retained.sha256 != expected_sha256:
        raise CaptureError(
            f"retained NVBit phase tool {stage} digest differs from authority",
            code="capture_generation_changed",
        )


def _verify_nvbit_tracer_chain(
    authority: ExternalTracerAuthority | None,
    source: immutable.RetainedRegular | None,
    copied: immutable.RetainedRegular | None,
    expected_sha256: str,
    *,
    stage: str,
) -> None:
    if authority is None or source is None or copied is None:
        raise CaptureError(
            f"NVBit tracer {stage} descriptor authority is absent",
            code="capture_generation_changed",
        )
    _verify_external_tracer_checkpoint(source, authority, stage=stage)
    _verify_retained_phase_tool_checkpoint(
        copied,
        expected_sha256,
        stage=stage,
    )


def _validated_capture_output_root(path: Path) -> Path:
    try:
        protected = protected_paths.assert_external_output(
            path,
            repo_root=ROOT,
            operation="write CUTLASS successor capture output",
        )
    except protected_paths.ProtectedPathError as exc:
        raise CaptureError(f"capture output root is not allowed: {exc}") from exc
    return _absolute_external(protected, "capture output root")


def _check_dt(value: str, label: str) -> None:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise CaptureError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CaptureError(f"{label} must include a timezone")


def _load_object(
    path: Path,
    label: str,
    *,
    max_bytes: int = OPERATOR_PREFLIGHT_MAX_BYTES,
) -> tuple[dict[str, Any], bytes]:
    with _retained_bounded_regular(path, label, max_bytes=max_bytes) as retained:
        payload = _read_retained_bytes(
            retained,
            label,
            max_bytes=max_bytes,
        )
        value = _strict_json_value(payload, label=label)
    if type(value) is not dict:
        raise CaptureError(
            f"{label} is not an object",
            code="capture_input_integrity_failed",
        )
    return value, payload


def _hash_repo_file(rel: str, expected: str, label: str) -> None:
    path = ROOT / rel
    with _retained_bounded_regular(
        path,
        label,
        max_bytes=REPOSITORY_INPUT_MAX_BYTES,
    ) as retained:
        observed = retained.sha256
    if observed != expected:
        raise CaptureError(
            f"{label} hash drifted: {rel}",
            code="capture_input_integrity_failed",
        )


def _row_for(matrix_key: str, variant_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, roster, matrix = admission.validate_immutable_bundle()
    rows = {
        (row["matrix_key"], row["variant_id"]): row for row in matrix["rows"]
    }
    key = (matrix_key, variant_id)
    if key not in rows:
        raise CaptureError(f"unknown successor matrix_key+variant_id: {key}")
    roster_rows = {(row["matrix_key"], row["variant_id"]): row for row in roster["rows"]}
    if key not in roster_rows:
        raise CaptureError(f"successor roster row is missing: {key}")
    return rows[key], roster_rows[key]


def _expected_preflight(row: dict[str, Any], roster_row: dict[str, Any]) -> dict[str, Any]:
    shape = row["shape"]
    static = row["static_launch_inventory"]
    witness = roster_row["witness"]
    return {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_preflight",
        "matrix_key": row["matrix_key"],
        "variant_id": row["variant_id"],
        "arch": ARCH,
        "sku": SKU,
        "build_arch": ARCH,
        "source_commit": PINNED_COMMIT,
        "binary_path": static["binary_path"],
        "binary_sha256": static["binary_sha256"],
        "witness_path": witness["path"],
        "witness_sha256": witness["sha256"],
        "shape_path": shape["path"],
        "shape_sha256": shape["sha256"],
        "no_competing_processes": True,
        "competing_processes": [],
        "gpu_name": GPU_NAME,
        "persistence_mode": "Enabled",
        "sm_clock_mhz": SM_CLOCK_MHZ,
        "mem_clock_mhz": MEM_CLOCK_MHZ,
        "cuda_release": CUDA_RELEASE,
        "nvbit_version": NVBIT_VERSION,
        "ncu_version": NCU_VERSION,
    }


def validate_successor_preflight(
    path: Path, row: dict[str, Any], roster_row: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path.is_symlink():
        raise CaptureError("successor preflight must be an absolute regular file")
    document, payload = _load_object(path, "successor preflight")
    expected = _expected_preflight(row, roster_row)
    if document != expected:
        mismatches = [key for key in expected if document.get(key) != expected[key]]
        extras = sorted(set(document) - set(expected))
        raise CaptureError(
            f"successor preflight pins mismatch fields={mismatches!r} extras={extras!r}"
        )
    _hash_repo_file(expected["binary_path"], expected["binary_sha256"], "successor binary")
    _hash_repo_file(expected["witness_path"], expected["witness_sha256"], "successor witness")
    _hash_repo_file(expected["shape_path"], expected["shape_sha256"], "successor shape")
    if expected["source_commit"] != PINNED_COMMIT:
        raise CaptureError("successor preflight source commit is not the immutable pin")
    return expected, payload


def _capture_inventory_policy() -> dict[str, int | dict[str, int]]:
    return raw_trace_generation_limits(DEFAULT_TRACE_ADMISSION_LIMITS)


def _capture_tree_inventory(
    root: Path,
    *,
    byte_budget: int | None = None,
    retained_root_descriptor: int | None = None,
) -> dict[str, Any]:
    """Incrementally inventory a capture tree under admission-owned ceilings."""

    if byte_budget is not None and (
        type(byte_budget) is not int or byte_budget <= 0
    ):
        raise CaptureError(
            "capture byte budget must be positive",
            code="capture_inventory_policy_invalid",
        )
    try:
        policy = _capture_inventory_policy()
        max_files = int(policy["max_files"])
        max_directories = int(policy["max_directories"])
        max_total_bytes = int(policy["max_total_file_bytes"])
        max_total_path_bytes = int(policy["max_total_relative_path_bytes"])
        max_file_bytes = int(policy["max_file_bytes"])
        path_limits = policy["max_file_bytes_by_path"]
        assert isinstance(path_limits, dict)
        stack: list[tuple[Path, str]] = []
    except MemoryError as exc:
        raise CaptureError(
            "capture inventory policy allocation exhausted memory",
            code="capture_inventory_resource_unavailable",
        ) from exc
    file_count = 0
    directory_count = 0
    total_bytes = 0
    total_path_bytes = 0
    first_file: str | None = None
    try:
        if retained_root_descriptor is None:
            try:
                root_metadata = root.lstat()
            except FileNotFoundError:
                return {
                    "file_count": 0,
                    "directory_count": 0,
                    "total_bytes": 0,
                    "total_relative_path_bytes": 0,
                    "first_file": None,
                }
            root_is_symlink = stat.S_ISLNK(root_metadata.st_mode)
        else:
            if (
                type(retained_root_descriptor) is not int
                or retained_root_descriptor < 0
                or root
                != Path(f"/proc/self/fd/{retained_root_descriptor}")
            ):
                raise CaptureError(
                    "capture inventory retained root is not exact",
                    code="capture_inventory_unsafe_entry",
                )
            try:
                root_metadata = os.fstat(retained_root_descriptor)
            except OSError as exc:
                raise CaptureError(
                    f"capture inventory retained root is unavailable: {exc}",
                    code="capture_inventory_unsafe_entry",
                ) from exc
            root_is_symlink = False
        if root_is_symlink or not stat.S_ISDIR(root_metadata.st_mode):
            raise CaptureError(
                f"capture output descendant is unsafe: {root}",
                code="capture_inventory_unsafe_entry",
            )
        stack.append((root, ""))
        directory_count = 1
        while stack:
            directory, relative_directory = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative_path = (
                        f"{relative_directory}/{entry.name}"
                        if relative_directory
                        else entry.name
                    )
                    total_path_bytes += len(os.fsencode(relative_path))
                    if total_path_bytes > max_total_path_bytes:
                        raise CaptureError(
                            "capture inventory relative-path byte limit exceeded: "
                            f"{total_path_bytes} > {max_total_path_bytes}",
                            code="capture_inventory_path_limit_exceeded",
                        )
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(metadata.st_mode):
                        raise CaptureError(
                            f"capture output descendant is unsafe: {entry.path}",
                            code="capture_inventory_unsafe_entry",
                        )
                    if stat.S_ISDIR(metadata.st_mode):
                        directory_count += 1
                        if directory_count > max_directories:
                            raise CaptureError(
                                "capture inventory directory-count limit exceeded: "
                                f"{directory_count} > {max_directories}",
                                code="capture_inventory_directory_limit_exceeded",
                            )
                        stack.append((Path(entry.path), relative_path))
                        continue
                    if not stat.S_ISREG(metadata.st_mode):
                        raise CaptureError(
                            f"capture output descendant is unsafe: {entry.path}",
                            code="capture_inventory_unsafe_entry",
                        )
                    file_count += 1
                    if file_count > max_files:
                        raise CaptureError(
                            "capture inventory file-count limit exceeded: "
                            f"{file_count} > {max_files}",
                            code="capture_inventory_file_limit_exceeded",
                        )
                    file_limit = min(
                        max_file_bytes,
                        int(path_limits.get(relative_path, max_file_bytes)),
                    )
                    if metadata.st_size > file_limit:
                        raise CaptureError(
                            "capture inventory per-file byte limit exceeded: "
                            f"{relative_path}: {metadata.st_size} > {file_limit}",
                            code="capture_inventory_file_bytes_exceeded",
                        )
                    total_bytes += metadata.st_size
                    if total_bytes > max_total_bytes:
                        raise CaptureError(
                            "capture inventory total-byte limit exceeded: "
                            f"{total_bytes} > {max_total_bytes}",
                            code="capture_inventory_total_limit_exceeded",
                        )
                    if byte_budget is not None and total_bytes > byte_budget:
                        raise CaptureError(
                            "capture trace byte budget exceeded: "
                            f"{total_bytes} > {byte_budget}",
                            code="capture_trace_byte_budget_exceeded",
                        )
                    if first_file is None:
                        first_file = relative_path
    except CaptureError:
        raise
    except MemoryError as exc:
        raise CaptureError(
            "capture inventory exhausted memory",
            code="capture_inventory_resource_unavailable",
        ) from exc
    except OSError as exc:
        code = (
            "capture_inventory_resource_unavailable"
            if exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM}
            else "capture_inventory_io_failed"
        )
        raise CaptureError(
            f"capture output inventory failed: {exc}",
            code=code,
        ) from exc
    return {
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "total_relative_path_bytes": total_path_bytes,
        "first_file": first_file,
    }


def _reject_preexisting_output_descendants(
    root: Path, *, output_dir: Path, reserved_files: tuple[Path, ...]
) -> None:
    """Reject stale entries only in this phase's output namespace."""
    if root.is_symlink() or not root.is_dir():
        raise CaptureError(f"capture output root is not a directory: {root}")

    def check_components(path: Path) -> None:
        try:
            parts = path.relative_to(root).parts
        except ValueError as exc:
            raise CaptureError(f"capture output path escapes root: {path}") from exc
        candidate = root
        for component in parts:
            candidate /= component
            if candidate.exists() or candidate.is_symlink():
                if candidate.is_symlink() or not candidate.is_dir():
                    raise CaptureError(
                        f"pre-existing output descendant is unsafe: {candidate}"
                    )

    check_components(output_dir)
    inventory = _capture_tree_inventory(output_dir)
    if inventory["file_count"]:
        raise CaptureError(
            "pre-existing capture output must be empty: "
            f"{output_dir / str(inventory['first_file'])}"
        )

    for path in reserved_files:
        check_components(path.parent)
        if path.exists() or path.is_symlink():
            raise CaptureError(f"pre-existing output artifact must be removed: {path}")


def _capture_tree_bytes(
    root: Path,
    *,
    retained_root_descriptor: int | None = None,
) -> int:
    return int(_capture_tree_inventory(
        root,
        retained_root_descriptor=retained_root_descriptor,
    )["total_bytes"])


def _require_capture_tree_budget(root: Path, byte_budget: int) -> None:
    try:
        observed_bytes = _capture_tree_bytes(root)
    except CaptureError:
        raise
    if observed_bytes > byte_budget:
        raise CaptureError(
            "post-processed capture trace exceeds its byte budget: "
            f"{observed_bytes} > {byte_budget}",
            code="capture_trace_byte_budget_exceeded",
        )


def _kill_capture_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _process_exited_unreaped(process: subprocess.Popen[bytes]) -> bool:
    """Observe child exit without releasing its PID/process-group authority."""
    while True:
        try:
            status = os.waitid(
                os.P_PID,
                process.pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except InterruptedError:
            continue
        except (ChildProcessError, OSError) as exc:
            raise CaptureError(
                f"capture process exit state is unavailable: {exc}"
            ) from exc
        return status is not None


def _read_capture_streams(
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    *,
    timeout_s: float,
) -> tuple[bool, OSError | None, bool]:
    exceeded = False
    progressed = False
    try:
        ready = selector.select(timeout_s)
    except OSError as exc:
        return False, exc, False
    for key, _mask in ready:
        label = key.data
        try:
            chunk = os.read(key.fd, CAPTURE_STREAM_READ_BYTES)
        except BlockingIOError:
            continue
        except OSError as exc:
            return exceeded, exc, progressed
        progressed = True
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        retained = buffers[label]
        remaining = CAPTURE_STREAM_LIMIT_BYTES - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            exceeded = True
    return exceeded, None, progressed


def _copy_capture_stream_prefix(buffer: bytearray, length: int) -> bytes:
    if length == len(buffer):
        return bytes(buffer)
    return bytes(memoryview(buffer)[:length])


def _materialize_capture_stream(
    buffer: bytearray,
) -> tuple[bytes, MemoryError | None]:
    """Copy one stream and salvage an exact prefix after allocation failure."""

    first_error: MemoryError | None = None
    candidate_length = len(buffer)
    retry_full = True
    while candidate_length > 0:
        try:
            return _copy_capture_stream_prefix(buffer, candidate_length), first_error
        except MemoryError as exc:
            if first_error is None:
                first_error = exc
            if retry_full:
                retry_full = False
            else:
                candidate_length //= 2
    return b"", first_error


def _attach_capture_secondary(
    primary: BaseException,
    secondary: BaseException | None,
    label: str,
) -> None:
    if secondary is None:
        return
    try:
        primary.add_note(f"{label}: {secondary}")
    except BaseException:
        pass


def _close_generation_stack_preserving(
    generation_stack: contextlib.ExitStack,
    primary: BaseException,
    *,
    label: str,
) -> None:
    """Close generations while separating integrity drift from cleanup failure."""

    notes_before = len(getattr(primary, "__notes__", ()))
    close_error: BaseException | None = None
    try:
        generation_stack.__exit__(
            type(primary),
            primary,
            primary.__traceback__,
        )
    except BaseException as exc:
        if exc is not primary:
            close_error = exc
    new_notes = getattr(primary, "__notes__", ())[notes_before:]
    revalidation_note = next(
        (
            note
            for note in new_notes
            if "revalidation failed" in note
        ),
        None,
    )
    if revalidation_note is not None:
        if close_error is not None:
            raise close_error
        raise immutable.ImmutableArtifactError(revalidation_note)
    _attach_capture_secondary(primary, close_error, label)


@contextlib.contextmanager
def _revalidate_tree_generation_on_primary(
    reader: immutable.PinnedDirectoryReader,
    initial: immutable.PinnedTreeSnapshot,
    *,
    limits: dict[str, int],
    label: str,
):
    """Revalidate a tree even when its enclosing stack carries a primary."""

    try:
        yield
    except BaseException as primary:
        try:
            final = reader.snapshot_tree(**limits)
            if final != initial:
                raise immutable.ImmutableArtifactError(
                    "retained tree generation changed during use"
                )
            reader.verify_path()
        except BaseException as revalidation_error:
            _attach_capture_secondary(
                primary,
                revalidation_error,
                f"{label} revalidation failed",
            )
            raise
        raise


def _retain_capture_failure_streams(
    failure_root: Path | None,
    prefix: str,
    stdout: bytes,
    stderr: bytes,
    primary: BaseException,
) -> None:
    """Best-effort evidence retention that cannot replace a typed primary."""

    if failure_root is None:
        return
    try:
        _write_new(failure_root / f"{prefix}.stdout.bin", stdout)
    except BaseException as exc:
        try:
            primary.add_note(f"stdout failure-stream retention failed: {exc}")
        except BaseException:
            pass
    try:
        _write_new(failure_root / f"{prefix}.stderr.bin", stderr)
    except BaseException as exc:
        try:
            primary.add_note(f"stderr failure-stream retention failed: {exc}")
        except BaseException:
            pass


def _run_capture_command(
    command: list[str], env: dict[str, str], cwd: Path, *,
    timeout_s: float = CAPTURE_TIMEOUT_S, byte_root: Path | None = None,
    byte_root_descriptor: int | None = None,
    byte_budget: int | None = None, failure_root: Path | None = None,
    inherited_descriptors: tuple[int, ...] = (),
    require_cgroup_containment: bool = False,
) -> tuple[
    int, int, str, bytes, bytes,
    delegated_cgroup.CgroupContainmentEvidence | None,
]:
    if type(require_cgroup_containment) is not bool:
        raise CaptureError("cgroup containment requirement must be boolean")
    if timeout_s <= 0:
        raise CaptureError("capture timeout must be positive")
    if byte_budget is not None and byte_budget <= 0:
        raise CaptureError("capture byte budget must be positive")
    if byte_root_descriptor is not None:
        if (
            type(byte_root_descriptor) is not int
            or byte_root_descriptor < 0
            or byte_root is None
            or byte_budget is None
            or byte_root != Path(f"/proc/self/fd/{byte_root_descriptor}")
            or byte_root_descriptor not in inherited_descriptors
        ):
            raise CaptureError(
                "capture byte-root descriptor authority is not exact",
                code="capture_inventory_unsafe_entry",
            )
        try:
            byte_root_metadata = os.fstat(byte_root_descriptor)
        except OSError as exc:
            raise CaptureError(
                f"capture byte-root descriptor is unavailable: {exc}",
                code="capture_inventory_unsafe_entry",
            ) from exc
        if not stat.S_ISDIR(byte_root_metadata.st_mode):
            raise CaptureError(
                "capture byte-root descriptor is not a directory",
                code="capture_inventory_unsafe_entry",
            )
    if CAPTURE_STREAM_LIMIT_BYTES <= 0 or CAPTURE_STREAM_READ_BYTES <= 0:
        raise CaptureError("capture stream limits must be positive")
    started = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    _check_dt(started, "process start")
    deadline = time.monotonic() + timeout_s
    try:
        budget_exceeded = threading.Event()
        inventory_failed = threading.Event()
        inventory_resource_failed = threading.Event()
        inventory_failures: list[CaptureError] = []
        stop_watchdog = threading.Event()
        watchdog_done = threading.Event()
        watchdog_unexpected_exit = threading.Event()
        inventory_resource_failure = CaptureError(
            "capture inventory watchdog exhausted memory",
            code="capture_inventory_resource_unavailable",
        )
        inventory_failure_without_detail = CaptureError(
            "capture inventory failed without retained detail",
            code="capture_inventory_failed",
        )
        watchdog_terminated_failure = CaptureError(
            "capture inventory watchdog terminated unexpectedly",
            code="capture_inventory_watchdog_terminated",
        )
        output_resource_failure = CaptureError(
            "capture process output resources are unavailable",
            code="capture_output_resource_unavailable",
        )
        budget_failure = CaptureError(
            "capture process exceeded its trace byte budget",
            code="capture_trace_byte_budget_exceeded",
        )
        process_resource_failure = CaptureError(
            "capture process wait resources are unavailable",
            code="capture_process_resource_unavailable",
        )
        terminal_inventory_resource_failure = CaptureError(
            "terminal capture inventory resources are unavailable",
            code="capture_inventory_resource_unavailable",
        )
        terminal_inventory_io_failure = CaptureError(
            "capture process left unsafe output because terminal inventory failed",
            code="capture_inventory_io_failed",
        )
        terminal_inventory_failure_without_detail = CaptureError(
            "terminal capture inventory failed without retained detail",
            code="capture_inventory_failed",
        )
    except MemoryError as exc:
        raise CaptureError(
            "capture inventory watchdog state is unavailable",
            code="capture_inventory_resource_unavailable",
        ) from exc

    def record_inventory_failure(error: CaptureError) -> None:
        if error.code == "capture_inventory_resource_unavailable":
            inventory_resource_failed.set()
        inventory_failed.set()
        try:
            inventory_failures.append(error)
        except MemoryError:
            inventory_resource_failed.set()

    def observed_capture_bytes() -> int:
        assert byte_root is not None
        if byte_root_descriptor is None:
            return _capture_tree_bytes(byte_root)
        return _capture_tree_bytes(
            byte_root,
            retained_root_descriptor=byte_root_descriptor,
        )

    def watch_bytes() -> None:
        while not stop_watchdog.wait(0.1):
            if byte_root is None or byte_budget is None:
                continue
            try:
                observed_bytes = observed_capture_bytes()
                if observed_bytes > byte_budget:
                    raise CaptureError(
                        "capture trace byte budget exceeded: "
                        f"{observed_bytes} > {byte_budget}",
                        code="capture_trace_byte_budget_exceeded",
                    )
            except CaptureError as exc:
                if exc.code == "capture_trace_byte_budget_exceeded":
                    budget_exceeded.set()
                else:
                    record_inventory_failure(exc)
                return
            except MemoryError:
                inventory_resource_failed.set()
                inventory_failed.set()
                try:
                    inventory_failures.append(inventory_resource_failure)
                except MemoryError:
                    pass
                return
            except OSError as exc:
                record_inventory_failure(CaptureError(
                    f"capture inventory watchdog failed: {exc}",
                    code=(
                        "capture_inventory_resource_unavailable"
                        if exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM}
                        else "capture_inventory_io_failed"
                    ),
                ))
                return
            except Exception as exc:
                record_inventory_failure(CaptureError(
                    f"capture inventory watchdog failed: {exc}",
                    code="capture_inventory_failed",
                ))
                return

    def run_watchdog() -> None:
        try:
            watch_bytes()
        except MemoryError as exc:
            inventory_resource_failed.set()
            inventory_failed.set()
            try:
                inventory_failures.append(inventory_resource_failure)
            except MemoryError:
                pass
            try:
                exc.add_note("watchdog MemoryError escaped its typed scan boundary")
            except (AttributeError, TypeError):
                pass
        except OSError as exc:
            record_inventory_failure(CaptureError(
                f"capture inventory watchdog failed: {exc}",
                code=(
                    "capture_inventory_resource_unavailable"
                    if exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM}
                    else "capture_inventory_io_failed"
                ),
            ))
        except Exception as exc:
            record_inventory_failure(CaptureError(
                f"capture inventory watchdog failed unexpectedly: {exc}",
                code="capture_inventory_failed",
            ))
        except BaseException:
            watchdog_unexpected_exit.set()
        finally:
            if (
                not stop_watchdog.is_set()
                and not budget_exceeded.is_set()
                and not inventory_failed.is_set()
            ):
                watchdog_unexpected_exit.set()
            watchdog_done.set()

    try:
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
    except MemoryError as exc:
        raise CaptureError(
            "capture bounded output buffers are unavailable",
            code="capture_output_resource_unavailable",
        ) from exc

    containment: delegated_cgroup.DelegatedCgroupV2 | None = None
    process: subprocess.Popen[bytes] | None = None
    popen_arguments = {
        "cwd": str(cwd),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "start_new_session": True,
        "pass_fds": inherited_descriptors,
    }
    selector: selectors.BaseSelector | None = None
    watchdog: threading.Thread | None = None
    failure_kind: str | None = None
    failure_cause: BaseException | None = None
    output_exceeded = False
    containment_evidence: delegated_cgroup.CgroupContainmentEvidence | None = None
    containment_error: BaseException | None = None
    group_kill_error: BaseException | None = None
    process_cleanup_error: BaseException | None = None
    process_wait_error: BaseException | None = None
    wait_resource_error: BaseException | None = None
    active_base_exception: BaseException | None = None
    watchdog_died_unexpectedly = False
    watchdog_join_timeout = False
    try:
        with nvbit_capture_process.bounded_launch(deadline):
            if require_cgroup_containment:
                try:
                    containment = delegated_cgroup.DelegatedCgroupV2.create()
                except delegated_cgroup.CgroupContainmentError as exc:
                    raise CaptureError(
                        f"capture descendant containment is unavailable: {exc}"
                    ) from exc
            process = (
                subprocess.Popen(command, **popen_arguments)
                if containment is None
                else containment.popen(command, **popen_arguments)
            )
    except BaseException as active_error:
        cleanup_errors: list[tuple[str, BaseException]] = []
        launch_group_kill_error: BaseException | None = None
        launch_containment_evidence: (
            delegated_cgroup.CgroupContainmentEvidence | None
        ) = None
        if process is not None:
            try:
                _kill_capture_process(process)
            except BaseException as cleanup_error:
                launch_group_kill_error = cleanup_error
            if containment is not None:
                try:
                    containment.kill_all()
                except BaseException as cleanup_error:
                    cleanup_errors.append(("descendant termination", cleanup_error))
            try:
                process.wait(timeout=CAPTURE_CLEANUP_TIMEOUT_S)
            except BaseException as cleanup_error:
                cleanup_errors.append(("leader reaping", cleanup_error))
            for stream in (process.stdout, process.stderr):
                if stream is None:
                    continue
                try:
                    stream.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(("pipe closure", cleanup_error))
        if containment is not None:
            try:
                if process is None:
                    containment.close_after_error()
                else:
                    launch_containment_evidence = containment.finish()
            except BaseException as cleanup_error:
                cleanup_errors.append(("containment removal", cleanup_error))
        if launch_group_kill_error is not None:
            if launch_containment_evidence is None:
                cleanup_errors.append((
                    "process-group termination",
                    launch_group_kill_error,
                ))
            elif not isinstance(launch_group_kill_error, Exception):
                _attach_capture_secondary(
                    active_error,
                    launch_group_kill_error,
                    "capture launch process-group cleanup interrupted",
                )
        if cleanup_errors:
            detail = "; ".join(
                f"{label} failed: {error}"
                for label, error in cleanup_errors
            )
            cleanup_code = (
                "capture_containment_cleanup_failed"
                if any(
                    label in {"descendant termination", "containment removal"}
                    for label, _error in cleanup_errors
                )
                else "capture_process_cleanup_failed"
            )
            cleanup_failure = CaptureError(
                f"capture launch cleanup failed: {detail}",
                code=cleanup_code,
                process_id=None if process is None else process.pid,
                started_at_utc=started,
            )
            _attach_capture_secondary(
                cleanup_failure,
                active_error,
                "active capture launch failure",
            )
            _retain_capture_failure_streams(
                failure_root,
                "capture_launch_cleanup",
                b"",
                b"",
                cleanup_failure,
            )
            raise cleanup_failure from cleanup_errors[0][1]
        if isinstance(
            active_error, nvbit_capture_process.CaptureLaunchTimeout
        ):
            timeout_error = CaptureError(
                f"capture process exceeded timeout of {timeout_s:g}s during launch",
                code="capture_timeout",
                started_at_utc=started,
            )
            _retain_capture_failure_streams(
                failure_root,
                "capture_timeout",
                b"",
                b"",
                timeout_error,
            )
            raise timeout_error from active_error
        if not isinstance(active_error, Exception):
            raise
        if isinstance(active_error, CaptureError):
            raise
        raise CaptureError(
            f"capture process could not start: {active_error}",
            code=(
                "capture_output_resource_unavailable"
                if isinstance(active_error, MemoryError)
                or (
                    isinstance(active_error, OSError)
                    and active_error.errno == errno.ENOMEM
                )
                else "capture_process_start_failed"
            ),
        ) from active_error

    assert process is not None

    try:
        if process.stdout is None or process.stderr is None:  # pragma: no cover
            raise CaptureError(
                "capture process streams are unavailable",
                process_id=process.pid,
                started_at_utc=started,
            )
        selector = selectors.DefaultSelector()
        try:
            for label, stream in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
        except (OSError, ValueError) as exc:
            raise CaptureError(
                f"capture process stream setup failed: {exc}",
                process_id=process.pid,
                started_at_utc=started,
            ) from exc

        try:
            watchdog_candidate = threading.Thread(
                target=run_watchdog,
                daemon=True,
            )
            watchdog_candidate.start()
            watchdog = watchdog_candidate
        except (MemoryError, RuntimeError) as exc:
            raise CaptureError(
                f"capture inventory watchdog could not start: {exc}",
                code="capture_inventory_resource_unavailable",
                process_id=process.pid,
                started_at_utc=started,
            ) from exc
        while True:
            now = time.monotonic()
            wait_s = max(0.0, min(CAPTURE_PROCESS_POLL_S, deadline - now))
            stream_exceeded, stream_error, _progressed = _read_capture_streams(
                selector, buffers, timeout_s=wait_s,
            )
            output_exceeded = output_exceeded or stream_exceeded
            if stream_error is not None:
                failure_kind = "stream"
                failure_cause = stream_error
                break
            if output_exceeded:
                failure_kind = "output_limit"
                break
            if inventory_failed.is_set():
                failure_kind = "inventory"
                failure_cause = (
                    inventory_failures[0]
                    if inventory_failures
                    else (
                        inventory_resource_failure
                        if inventory_resource_failed.is_set()
                        else inventory_failure_without_detail
                    )
                )
                break
            if budget_exceeded.is_set():
                failure_kind = "budget"
                break
            if (
                watchdog_unexpected_exit.is_set()
                or watchdog_done.is_set()
                or not watchdog.is_alive()
            ):
                failure_kind = "inventory"
                failure_cause = watchdog_terminated_failure
                break
            if time.monotonic() >= deadline:
                failure_kind = "timeout"
                break
            if _process_exited_unreaped(process):
                break
    except CaptureError as exc:
        failure_kind = "setup"
        failure_cause = exc
    except MemoryError as exc:
        failure_kind = "resource"
        failure_cause = exc
    except Exception as exc:
        failure_kind = "monitor"
        failure_cause = exc
    except BaseException as exc:
        active_base_exception = exc
    finally:
        # Snapshot unexpected watchdog death before asking it to stop.  A fast
        # child exit must not erase a concurrent watchdog failure.
        watchdog_alive_before_stop = True
        if watchdog is not None:
            try:
                watchdog_alive_before_stop = watchdog.is_alive()
            except MemoryError:
                inventory_resource_failed.set()
                inventory_failed.set()
            except Exception as exc:
                if process_cleanup_error is None:
                    process_cleanup_error = exc
            except BaseException as exc:
                if active_base_exception is None:
                    active_base_exception = exc
        if (
            watchdog is not None
            and not budget_exceeded.is_set()
            and not inventory_failed.is_set()
            and (
                watchdog_unexpected_exit.is_set()
                or watchdog_done.is_set()
                or not watchdog_alive_before_stop
            )
        ):
            watchdog_died_unexpectedly = True
        # Bound every teardown phase. Production launches also retain cgroup
        # authority, so process-group escapes are killed before leader reaping.
        stop_watchdog.set()
        try:
            _kill_capture_process(process)
        except BaseException as exc:
            if isinstance(exc, Exception):
                group_kill_error = exc
            elif active_base_exception is None:
                active_base_exception = exc
            else:
                try:
                    active_base_exception.add_note(
                        f"capture process-group kill interrupted: {exc}"
                    )
                except (AttributeError, TypeError):
                    pass
        if containment is not None:
            try:
                containment.kill_all()
            except BaseException as exc:
                containment_error = exc
        if watchdog is not None:
            try:
                watchdog.join(timeout=CAPTURE_CLEANUP_TIMEOUT_S)
            except BaseException as exc:
                if isinstance(exc, Exception):
                    process_cleanup_error = exc
                elif active_base_exception is None:
                    active_base_exception = exc
                else:
                    try:
                        active_base_exception.add_note(
                            f"capture watchdog join interrupted: {exc}"
                        )
                    except (AttributeError, TypeError):
                        pass
            try:
                watchdog_alive_after_join = watchdog.is_alive()
            except MemoryError:
                inventory_resource_failed.set()
                inventory_failed.set()
                watchdog_alive_after_join = True
            except Exception as exc:
                if process_cleanup_error is None:
                    process_cleanup_error = exc
                watchdog_alive_after_join = True
            except BaseException as exc:
                if active_base_exception is None:
                    active_base_exception = exc
                watchdog_alive_after_join = True
            if watchdog_alive_after_join:
                watchdog_join_timeout = True
            if (
                watchdog_unexpected_exit.is_set()
                or (
                    not watchdog_alive_after_join
                    and not watchdog_done.is_set()
                )
            ):
                watchdog_died_unexpectedly = True
        try:
            process.wait(timeout=CAPTURE_CLEANUP_TIMEOUT_S)
        except BaseException as exc:
            process_wait_error = exc
            if (
                isinstance(exc, MemoryError)
                or isinstance(exc, OSError)
                and exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM}
            ):
                wait_resource_error = exc
            if not isinstance(exc, Exception):
                if active_base_exception is None:
                    active_base_exception = exc
                else:
                    try:
                        active_base_exception.add_note(
                            f"capture process wait interrupted: {exc}"
                        )
                    except (AttributeError, TypeError):
                        pass
        if containment is not None:
            try:
                containment_evidence = containment.finish()
            except BaseException as exc:
                if containment_error is not None:
                    try:
                        exc.add_note(
                            f"initial cgroup.kill failed: {containment_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
                containment_error = exc
        if group_kill_error is not None and containment_evidence is None:
            if process_cleanup_error is None:
                process_cleanup_error = group_kill_error
            else:
                try:
                    process_cleanup_error.add_note(
                        f"capture process-group kill failed: {group_kill_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        if process_wait_error is not None:
            try:
                process.wait(timeout=CAPTURE_CLEANUP_TIMEOUT_S)
            except BaseException as retry_error:
                retry_failure = (
                    retry_error
                    if isinstance(retry_error, Exception)
                    else RuntimeError(
                        "capture process retry wait was interrupted before "
                        "leader reaping was proven"
                    )
                )
                if process_cleanup_error is None:
                    process_cleanup_error = retry_failure
                else:
                    try:
                        process_cleanup_error.add_note(
                            f"capture process retry wait failed: {retry_failure}"
                        )
                    except (AttributeError, TypeError):
                        pass
                try:
                    process_cleanup_error.add_note(
                        f"initial capture process wait failed: {process_wait_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            else:
                process_wait_error = None
        if selector is not None:
            try:
                while selector.get_map() and not output_exceeded:
                    stream_exceeded, stream_error, progressed = (
                        _read_capture_streams(
                            selector, buffers, timeout_s=0,
                        )
                    )
                    output_exceeded = output_exceeded or stream_exceeded
                    if stream_error is not None and failure_cause is None:
                        failure_cause = stream_error
                        failure_kind = failure_kind or "stream"
                        break
                    if not progressed:
                        break
                for key in list(selector.get_map().values()):
                    selector.unregister(key.fileobj)
                selector.close()
            except BaseException as exc:
                if process_cleanup_error is None:
                    process_cleanup_error = exc
                else:
                    try:
                        process_cleanup_error.add_note(
                            f"capture stream cleanup failed: {exc}"
                        )
                    except (AttributeError, TypeError):
                        pass
        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as exc:
                if process_cleanup_error is None:
                    process_cleanup_error = exc
                else:
                    try:
                        process_cleanup_error.add_note(
                            f"capture pipe cleanup failed: {exc}"
                        )
                    except (AttributeError, TypeError):
                        pass
        if watchdog_join_timeout:
            timeout_error = RuntimeError(
                "capture byte-budget watchdog did not stop within "
                f"{CAPTURE_CLEANUP_TIMEOUT_S:g}s"
            )
            if process_cleanup_error is None:
                process_cleanup_error = timeout_error
            else:
                try:
                    process_cleanup_error.add_note(str(timeout_error))
                except (AttributeError, TypeError):
                    pass

    try:
        stdout, stdout_materialization_error = _materialize_capture_stream(
            buffers["stdout"]
        )
    except MemoryError as exc:
        stdout, stdout_materialization_error = b"", exc
    try:
        stderr, stderr_materialization_error = _materialize_capture_stream(
            buffers["stderr"]
        )
    except MemoryError as exc:
        stderr, stderr_materialization_error = b"", exc
    materialization_error = (
        stdout_materialization_error or stderr_materialization_error
    )
    if materialization_error is not None:
        failure_kind = "resource"
        failure_cause = materialization_error

    if containment_error is not None:
        error = CaptureError(
            f"capture descendant containment cleanup failed: {containment_error}",
            code="capture_containment_cleanup_failed",
            stdout=stdout,
            stderr=stderr,
            process_id=process.pid,
            started_at_utc=started,
        )
        _retain_capture_failure_streams(
            failure_root,
            "capture_containment_cleanup",
            stdout,
            stderr,
            error,
        )
        if active_base_exception is not None:
            try:
                error.add_note(
                    "active capture interruption was superseded by containment "
                    f"cleanup failure: {active_base_exception}"
                )
            except (AttributeError, TypeError):
                pass
        if process_cleanup_error is not None:
            try:
                error.add_note(
                    f"capture process cleanup also failed: {process_cleanup_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise error from containment_error
    if process_cleanup_error is not None:
        error = CaptureError(
            f"capture process cleanup failed: {process_cleanup_error}",
            code="capture_process_cleanup_failed",
            stdout=stdout,
            stderr=stderr,
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        )
        _retain_capture_failure_streams(
            failure_root,
            "capture_cleanup",
            stdout,
            stderr,
            error,
        )
        if active_base_exception is not None:
            try:
                error.add_note(
                    "active capture interruption was superseded by process "
                    f"cleanup failure: {active_base_exception}"
                )
            except (AttributeError, TypeError):
                pass
        if failure_cause is not None:
            try:
                error.add_note(f"active capture failure: {failure_cause}")
            except (AttributeError, TypeError):
                pass
        raise error from process_cleanup_error
    if watchdog_died_unexpectedly and failure_kind is None:
        failure_kind = "inventory"
        failure_cause = watchdog_terminated_failure
    if failure_kind == "setup":
        assert isinstance(failure_cause, CaptureError)
        _retain_capture_failure_streams(
            failure_root,
            "capture_setup",
            stdout,
            stderr,
            failure_cause,
        )
        failure_cause.stdout = stdout
        failure_cause.stderr = stderr
        failure_cause.containment = containment_evidence
        _attach_capture_secondary(
            failure_cause,
            active_base_exception,
            "active capture interruption superseded by setup failure",
        )
        raise failure_cause
    if failure_kind is None and output_exceeded:
        failure_kind = "output_limit"
    if failure_kind is None and inventory_failed.is_set():
        failure_kind = "inventory"
        failure_cause = (
            inventory_failures[0]
            if inventory_failures
            else (
                inventory_resource_failure
                if inventory_resource_failed.is_set()
                else inventory_failure_without_detail
            )
        )
    if failure_kind is None and budget_exceeded.is_set():
        failure_kind = "budget"
    if failure_kind is None and wait_resource_error is not None:
        failure_kind = "process_resource"
        failure_cause = wait_resource_error
    if byte_root is not None and byte_budget is not None:
        terminal_inventory_failure: CaptureError | None = None
        try:
            observed_bytes = observed_capture_bytes()
            if observed_bytes > byte_budget:
                terminal_inventory_failure = budget_failure
        except CaptureError as exc:
            terminal_inventory_failure = exc
        except MemoryError as exc:
            terminal_inventory_failure = terminal_inventory_resource_failure
            terminal_inventory_failure.__cause__ = exc
        except OSError as exc:
            terminal_inventory_failure = (
                terminal_inventory_resource_failure
                if exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM}
                else terminal_inventory_io_failure
            )
            terminal_inventory_failure.__cause__ = exc
        except Exception as exc:
            terminal_inventory_failure = terminal_inventory_failure_without_detail
            terminal_inventory_failure.__cause__ = exc
        if terminal_inventory_failure is not None and failure_kind is None:
            if terminal_inventory_failure.code == (
                "capture_trace_byte_budget_exceeded"
            ):
                failure_kind = "budget"
                failure_cause = None
            else:
                failure_kind = "inventory"
                failure_cause = terminal_inventory_failure

    if failure_kind is None and active_base_exception is not None:
        raise active_base_exception
    if failure_kind == "timeout":
        error = CaptureError(
            f"capture process exceeded timeout of {timeout_s:g}s",
            code="capture_timeout",
            stdout=stdout,
            stderr=stderr,
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        )
        _attach_capture_secondary(
            error,
            active_base_exception,
            "active capture interruption superseded by timeout",
        )
        _retain_capture_failure_streams(
            failure_root, "capture_timeout", stdout, stderr, error,
        )
        raise error from failure_cause
    if failure_kind == "output_limit":
        error = CaptureError(
            "capture process exceeded its bounded stdout/stderr byte limit",
            code="capture_output_limit",
            stdout=stdout,
            stderr=stderr,
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        )
        _attach_capture_secondary(
            error,
            active_base_exception,
            "active capture interruption superseded by output limit",
        )
        _retain_capture_failure_streams(
            failure_root, "capture_output_limit", stdout, stderr, error,
        )
        raise error from failure_cause
    if failure_kind == "resource":
        output_resource_failure.stdout = stdout
        output_resource_failure.stderr = stderr
        output_resource_failure.process_id = process.pid
        output_resource_failure.started_at_utc = started
        output_resource_failure.containment = containment_evidence
        if failure_cause is not None:
            _attach_capture_secondary(
                output_resource_failure,
                failure_cause,
                "capture output resource cause",
            )
        _attach_capture_secondary(
            output_resource_failure,
            active_base_exception,
            "active capture interruption superseded by output resource failure",
        )
        _retain_capture_failure_streams(
            failure_root,
            "capture_output_resource",
            stdout,
            stderr,
            output_resource_failure,
        )
        raise output_resource_failure from failure_cause
    if failure_kind == "process_resource":
        process_resource_failure.stdout = stdout
        process_resource_failure.stderr = stderr
        process_resource_failure.process_id = process.pid
        process_resource_failure.started_at_utc = started
        process_resource_failure.containment = containment_evidence
        _attach_capture_secondary(
            process_resource_failure,
            active_base_exception,
            "active capture interruption superseded by process resource failure",
        )
        _retain_capture_failure_streams(
            failure_root,
            "capture_process_resource",
            stdout,
            stderr,
            process_resource_failure,
        )
        raise process_resource_failure from failure_cause
    if failure_kind == "stream":
        error = CaptureError(
            f"capture process stream collection failed: {failure_cause}",
            code="capture_stream_collection_failed",
            stdout=stdout,
            stderr=stderr,
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        )
        _attach_capture_secondary(
            error,
            active_base_exception,
            "active capture interruption superseded by stream failure",
        )
        _retain_capture_failure_streams(
            failure_root, "capture_stream_error", stdout, stderr, error,
        )
        raise error from failure_cause
    if failure_kind == "monitor":
        error = CaptureError(
            f"capture process monitor failed: {failure_cause}",
            code="capture_monitor_failed",
            stdout=stdout,
            stderr=stderr,
            process_id=process.pid,
            started_at_utc=started,
            containment=containment_evidence,
        )
        _attach_capture_secondary(
            error,
            active_base_exception,
            "active capture interruption superseded by monitor failure",
        )
        _retain_capture_failure_streams(
            failure_root, "capture_monitor_error", stdout, stderr, error,
        )
        raise error from failure_cause
    if failure_kind == "inventory":
        assert isinstance(failure_cause, CaptureError)
        failure_cause.stdout = stdout
        failure_cause.stderr = stderr
        failure_cause.process_id = process.pid
        failure_cause.started_at_utc = started
        failure_cause.containment = containment_evidence
        _attach_capture_secondary(
            failure_cause,
            active_base_exception,
            "active capture interruption superseded by inventory failure",
        )
        _retain_capture_failure_streams(
            failure_root,
            "capture_inventory_error",
            stdout,
            stderr,
            failure_cause,
        )
        raise failure_cause
    if failure_kind == "budget":
        budget_failure.stdout = stdout
        budget_failure.stderr = stderr
        budget_failure.process_id = process.pid
        budget_failure.started_at_utc = started
        budget_failure.containment = containment_evidence
        _attach_capture_secondary(
            budget_failure,
            active_base_exception,
            "active capture interruption superseded by trace budget failure",
        )
        _retain_capture_failure_streams(
            failure_root, "capture_budget", stdout, stderr, budget_failure,
        )
        raise budget_failure from failure_cause
    return (
        process.returncode,
        process.pid,
        started,
        stdout,
        stderr,
        containment_evidence,
    )


def _boot_id_sha256() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CaptureError(
                "Linux boot identity is not a regular procfs file",
                code="capture_input_integrity_failed",
            )
        payload = os.pread(descriptor, BOOT_ID_MAX_BYTES + 1, 0)
        if len(payload) > BOOT_ID_MAX_BYTES:
            raise CaptureError(
                "Linux boot identity exceeds its byte limit",
                code="capture_input_limit_exceeded",
            )
        payload = payload.strip()
    except CaptureError:
        raise
    except (MemoryError, OSError) as exc:
        error = _typed_input_error(
            "Linux boot identity",
            exc,
            action="descriptor-bound read",
        )
        raise error from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as close_error:
                active = sys.exception()
                if active is not None:
                    try:
                        active.add_note(
                            "Linux boot identity descriptor close failed: "
                            f"{close_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
                else:
                    error = _typed_input_error(
                        "Linux boot identity",
                        close_error,
                        action="descriptor close",
                    )
                    raise error from close_error
    if re.fullmatch(
        rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        payload,
    ) is None:
        raise CaptureError(
            "Linux boot identity is malformed",
            code="capture_input_integrity_failed",
        )
    return sha256_bytes(payload)


def _query(command: list[str], label: str) -> tuple[str, list[dict[str, str]]]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureError(f"{label} query failed: {exc}") from exc
    if result.returncode != 0:
        raise CaptureError(f"{label} query failed with status {result.returncode}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return result.stdout, [{"stdout": line} for line in lines]


def _nvbit_version() -> str:
    with _retained_bounded_regular(
        NVBIT_VERSION_HEADER,
        "NVBit version header",
        max_bytes=NVBIT_VERSION_HEADER_MAX_BYTES,
    ) as retained:
        payload = _read_retained_bytes(
            retained,
            "NVBit version header",
            max_bytes=NVBIT_VERSION_HEADER_MAX_BYTES,
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CaptureError(
            f"NVBit version header is not UTF-8: {exc}",
            code="capture_input_integrity_failed",
        ) from exc
    matches = re.findall(r'#define\s+NVBIT_VERSION\s+"([^"]+)"', text)
    if matches != [NVBIT_VERSION]:
        raise CaptureError(
            f"NVBit version pin mismatch: {matches!r}",
            code="capture_input_integrity_failed",
        )
    return matches[0]


def _ncu_identity_from_authority(
    authority: dict[str, Any],
) -> ncu_executor.ExpectedNcuIdentity:
    required = {
        "status",
        "expected",
        "execution_mode",
        "host_machine",
        "launcher_resolved_path",
        "launcher_sha256",
        "launcher_target_relpath",
        "resolved_path",
        "executable_sha256",
        "version_line",
    }
    if type(authority) is not dict or set(authority) != required:
        raise CaptureError("live NCU authority fields differ")
    if (
        authority["status"] != campaign_preflight.STATUS_OK
        or authority["expected"] != NCU_VERSION
        or authority["execution_mode"] != campaign_preflight.NCU_EXECUTION_MODE
        or authority["host_machine"] != campaign_preflight.NCU_HOST_MACHINE
        or authority["launcher_target_relpath"]
        != campaign_preflight.NCU_LAUNCHER_TARGET_RELPATH
    ):
        raise CaptureError("live NCU authority pins differ")
    launcher_raw = authority["launcher_resolved_path"]
    resolved_raw = authority["resolved_path"]
    if (
        type(launcher_raw) is not str
        or type(resolved_raw) is not str
        or not Path(launcher_raw).is_absolute()
        or not Path(resolved_raw).is_absolute()
        or os.path.normpath(launcher_raw) != launcher_raw
        or os.path.normpath(resolved_raw) != resolved_raw
    ):
        raise CaptureError("live NCU authority paths differ")
    launcher = Path(launcher_raw)
    resolved = Path(resolved_raw)
    if resolved != launcher.parent / authority["launcher_target_relpath"]:
        raise CaptureError("live NCU launcher and inner executable paths differ")
    if not all(
        type(authority[name]) is str
        and re.fullmatch(r"[0-9a-f]{64}", authority[name]) is not None
        for name in ("launcher_sha256", "executable_sha256")
    ):
        raise CaptureError("live NCU authority hashes are invalid")
    try:
        return admission._validate_ncu_authority(authority)
    except admission.AdmissionError as exc:
        raise CaptureError(f"live NCU executable authority is invalid: {exc}") from exc


def _probe_ncu_authority() -> tuple[dict[str, Any], str]:
    observed = campaign_preflight.probe_ncu(
        {"toolchain": {"ncu_version_prefix": NCU_VERSION}},
        require_cgroup_containment=True,
    )
    authority = {
        name: observed.get(name)
        for name in (
            "status",
            "expected",
            "execution_mode",
            "host_machine",
            "launcher_resolved_path",
            "launcher_sha256",
            "launcher_target_relpath",
            "resolved_path",
            "executable_sha256",
            "version_line",
        )
    }
    expected = _ncu_identity_from_authority(authority)
    try:
        result = ncu_executor.execute_ncu_action(
            expected,
            ["--version"],
            timeout_seconds=30,
            require_cgroup_containment=True,
        )
        stdout = result.stdout.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ncu_executor.NcuActionExecutorError) as exc:
        raise CaptureError(f"live NCU version output is invalid: {exc}") from exc
    expected_stdout = (
        "NVIDIA (R) Nsight Compute Command Line Profiler\n"
        "Copyright (c) 2018-2026 NVIDIA Corporation\n"
        f"{expected.version_line}\n"
    )
    if result.returncode != 0 or result.stderr or stdout != expected_stdout:
        raise CaptureError("live NCU version output differs from exact authority")
    return authority, stdout


def _label_driver_identity() -> ncu_executor.ExpectedExecutableIdentity:
    try:
        with _retained_bounded_regular(
            LABEL_DRIVER,
            "label driver",
            max_bytes=TOOL_SOURCE_MAX_BYTES,
        ) as retained:
            _require_retained_executable(retained, "label driver")
            if retained.sha256 != admission.LABEL_DRIVER_SHA256:
                raise CaptureError(
                    "label driver differs from the pinned capture authority",
                    code="capture_input_integrity_failed",
                )
            mode = stat.S_IMODE(os.fstat(retained.descriptor).st_mode)
            if mode != admission.LABEL_DRIVER_MODE:
                raise CaptureError(
                    "label driver mode differs from the pinned capture authority",
                    code="capture_input_integrity_failed",
                )
            expected = ncu_executor.build_expected_executable_identity(
                resolved_path=str(retained.path),
                executable_sha256=retained.sha256,
                executable_mode=admission.LABEL_DRIVER_MODE,
            )
            retained.verify_unchanged()
    except CaptureError:
        raise
    except (
        MemoryError,
        OSError,
        immutable.ImmutableArtifactError,
        ncu_executor.NcuActionExecutorError,
    ) as exc:
        error = _typed_input_error(
            "label driver authority",
            exc,
            action="validation",
        )
        raise error from exc
    return expected


def _label_collection_environment(
    source: dict[str, str], *, cwd: Path,
) -> ncu_executor.NcuCollectionEnvironment:
    resolved_cwd = cwd.resolve(strict=True)
    values = {
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": _LABEL_COLLECTION_PATH,
        "PWD": str(resolved_cwd),
        "PYTHONNOUSERSITE": "1",
        "LABEL_DRIVER_GPU_INDEX": "0",
        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
    }
    for name in _LABEL_COLLECTION_ENVIRONMENT_NAMES:
        value = source.get(name)
        if value:
            values[name] = value
    try:
        return ncu_executor.NcuCollectionEnvironment(values)
    except ncu_executor.NcuActionExecutorError as exc:
        raise CaptureError(f"label collection environment is invalid: {exc}") from exc


def _command_option(command: list[str], name: str) -> str:
    if command.count(name) != 1:
        raise CaptureError(f"NCU command requires exactly one {name}")
    index = command.index(name)
    if index + 1 >= len(command) or command[index + 1] == "--":
        raise CaptureError(f"NCU command has no value for {name}")
    return command[index + 1]


def _run_ncu_capture_command(
    command: list[str],
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    environment: ncu_executor.NcuCollectionEnvironment,
    driver_identity: ncu_executor.ExpectedExecutableIdentity,
    retained_target: immutable.RetainedRegular,
    cwd: Path,
    *,
    timeout_s: float = CAPTURE_TIMEOUT_S,
    failure_root: Path | None = None,
) -> tuple[
    int, int, str, bytes, bytes,
    delegated_cgroup.CgroupContainmentEvidence,
]:
    if not timeout_s > 0:
        raise CaptureError("capture timeout must be positive")
    expected_options = {
        "--ncu": expected_ncu_identity.resolved_path,
        "--ncu-executable-sha256": expected_ncu_identity.executable_sha256,
        "--ncu-version-line": expected_ncu_identity.version_line,
        "--target-processes": "all",
        "--filter-mode": "global",
        "--replay-mode": "kernel",
        "--cache-control": "all",
        "--clock-control": "none",
        "--profile-from-start": "on",
        "--launch-skip": "0",
        "--launch-count": "1",
    }
    if command[0] != str(LABEL_DRIVER) or "--ncu-exec-path" in command:
        raise CaptureError("NCU command does not name the canonical label driver")
    for name, expected in expected_options.items():
        if _command_option(command, name) != expected:
            raise CaptureError(f"NCU command authority differs for {name}")
    if command.count("--") != 1:
        raise CaptureError("NCU command requires one application separator")
    supervised_command: list[object] = list(command)
    binary_index = supervised_command.index("--binary") + 1
    if supervised_command[binary_index] != str(retained_target.path):
        raise CaptureError("NCU target command differs from retained executable")
    supervised_command[binary_index] = retained_target.proc_path
    separator = supervised_command.index("--")
    supervised_command[separator:separator] = [
        "--ncu-exec-path",
        ncu_executor.AUTHORIZED_NCU_PROC_PATH,
    ]
    if environment.as_dict()["PWD"] != str(cwd.resolve(strict=True)):
        raise CaptureError("NCU collection PWD differs from the process directory")
    try:
        result = ncu_executor.execute_ncu_supervised_executable(
            expected_ncu_identity,
            driver_identity,
            supervised_command,
            timeout_seconds=timeout_s,
            environment=environment,
            working_directory=cwd,
            auxiliary_descriptors=(retained_target.descriptor,),
            require_cgroup_containment=True,
        )
    except ncu_executor.NcuActionExecutorError as exc:
        if failure_root is not None:
            _write_new(failure_root / "ncu_executor_failure.stdout.bin", exc.stdout)
            _write_new(failure_root / "ncu_executor_failure.stderr.bin", exc.stderr)
        detail = exc.stderr[-400:] if exc.stderr else str(exc).encode()
        process_command = (
            None
            if exc.process_command is None
            else [
                str(retained_target.path)
                if item == retained_target.proc_path
                else item
                for item in exc.process_command
            ]
        )
        raise CaptureError(
            f"descriptor-bound NCU capture failed: {detail!r}",
            process_id=exc.process_id,
            started_at_utc=exc.started_at_utc,
            process_role=exc.process_role,
            process_command=process_command,
            process_environment=exc.process_environment,
            process_tool_sha256=exc.process_tool_sha256,
            containment=exc.containment,
        ) from exc
    if type(result.process_id) is not int or result.process_id <= 0:
        raise CaptureError("descriptor-bound NCU capture returned no process identity")
    if type(result.started_at_utc) is not str:
        raise CaptureError("descriptor-bound NCU capture returned no start time")
    if result.containment is None:
        raise CaptureError(
            "descriptor-bound NCU capture returned no containment evidence"
        )
    _check_dt(result.started_at_utc, "process start")
    return (
        result.returncode,
        result.process_id,
        result.started_at_utc,
        result.stdout,
        result.stderr,
        result.containment,
    )


def validate_live_capture_preflight(*, phase: str) -> dict[str, Any]:
    """Check phase-local live host state only after static preflight pins pass."""
    if phase not in {"ncu", "nvbit"}:
        raise CaptureError("live preflight phase is invalid")
    smi_command = [
        "nvidia-smi", "--query-gpu=name,compute_cap,persistence_mode,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    smi_stdout, _ = _query(smi_command, "nvidia-smi device")
    smi_lines = [line.strip() for line in smi_stdout.splitlines() if line.strip()]
    if len(smi_lines) != 1:
        raise CaptureError(f"expected one pinned GPU, observed={smi_lines!r}")
    fields = [part.strip() for part in smi_lines[0].split(",")]
    if fields != [GPU_NAME, "8.6", "Enabled", str(SM_CLOCK_MHZ), str(MEM_CLOCK_MHZ)]:
        raise CaptureError(f"live GPU state is not pinned: {fields!r}")
    apps_command = [
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits",
    ]
    apps_stdout, _ = _query(apps_command, "nvidia-smi competing-process")
    if apps_stdout.strip() and "no running processes" not in apps_stdout.lower():
        raise CaptureError(f"competing GPU processes are present: {apps_stdout.strip()!r}")
    identity_command = [
        "nvidia-smi", "--query-gpu=uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    identity_stdout, _ = _query(identity_command, "nvidia-smi device identity")
    identity_lines = [
        line.strip() for line in identity_stdout.splitlines() if line.strip()
    ]
    if len(identity_lines) != 1:
        raise CaptureError(
            f"expected one selected GPU identity, observed={identity_lines!r}"
        )
    identity_fields = [part.strip() for part in identity_lines[0].split(",")]
    if (
        len(identity_fields) != 2
        or re.fullmatch(admission.GPU_UUID_PATTERN, identity_fields[0]) is None
        or identity_fields[1] != DRIVER_VERSION
    ):
        raise CaptureError(
            f"live GPU identity is not pinned: {identity_fields!r}"
        )
    gpu_uuid_sha256 = sha256_bytes(identity_fields[0].encode("utf-8"))
    nvcc_command = [str(CUDA_NVCC), "--version"]
    nvcc_stdout, _ = _query(nvcc_command, "CUDA 13.2 nvcc")
    try:
        cuda = parse_nvcc_release_banner(nvcc_stdout)
        require_sol_cuda_release(cuda)
    except ValueError as exc:
        raise CaptureError(f"CUDA release pin failed: {exc}") from exc
    nvbit = _nvbit_version()
    result = {
        "smi_command": smi_command,
        "smi_stdout": smi_stdout,
        "smi_stdout_sha256": sha256_bytes(smi_stdout.encode()),
        "competing_process_command": apps_command,
        "competing_process_stdout": apps_stdout,
        "competing_process_stdout_sha256": sha256_bytes(apps_stdout.encode()),
        "identity_command": identity_command,
        "identity_stdout": identity_stdout,
        "identity_stdout_sha256": sha256_bytes(identity_stdout.encode()),
        "gpu_uuid_sha256": gpu_uuid_sha256,
        "driver_version": DRIVER_VERSION,
        "nvcc_command": nvcc_command,
        "nvcc_stdout": nvcc_stdout,
        "nvcc_stdout_sha256": sha256_bytes(nvcc_stdout.encode()),
        "cuda_release": cuda,
        "nvbit_version": nvbit,
        "gpu_name": GPU_NAME,
        "persistence_mode": "Enabled",
        "sm_clock_mhz": SM_CLOCK_MHZ,
        "mem_clock_mhz": MEM_CLOCK_MHZ,
    }
    if phase == "ncu":
        ncu_authority, ncu_stdout = _probe_ncu_authority()
        if not ncu_version_matches(ncu_authority["version_line"], NCU_VERSION):
            raise CaptureError("NCU version is not the pinned 2026.1.0 release")
        result.update({
            "ncu_command": [ncu_authority["resolved_path"], "--version"],
            "ncu_stdout": ncu_stdout,
            "ncu_stdout_sha256": sha256_bytes(ncu_stdout.encode()),
            "ncu_version": NCU_VERSION,
            "ncu_authority": ncu_authority,
        })
    return result


def _typed_trace_capture_error(
    stage: str,
    exc: BaseException,
) -> CaptureError:
    if isinstance(exc, CaptureError):
        return exc
    if isinstance(exc, TraceAdmissionError):
        code = exc.typed_reason
    elif isinstance(exc, ProtobufDecodedResourceLimitExceeded):
        code = "protobuf_decoded_resource_limit_exceeded"
    elif isinstance(exc, ProtobufWireMalformed):
        code = "trace_root_unparseable"
    elif isinstance(exc, FileNotFoundError) or (
        isinstance(exc, OSError) and exc.errno == errno.ENOENT
    ):
        code = "trace_root_unparseable"
    elif isinstance(exc, immutable.ImmutableArtifactLimitError):
        code = "trace_snapshot_limit_exceeded"
    elif isinstance(exc, immutable.ImmutableArtifactResourceError):
        code = "trace_snapshot_resource_unavailable"
    elif isinstance(exc, immutable.ImmutableArtifactError):
        code = "capture_generation_changed"
    elif isinstance(exc, MemoryError) or (
        isinstance(exc, OSError)
        and exc.errno in {errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.ENOSPC}
    ):
        code = "trace_snapshot_resource_unavailable"
    else:
        code = "capture_trace_validation_failed"
    return CaptureError(
        f"NVBit trace {stage} validation failed: {exc}",
        code=code,
    )


def _trace_path_max_bytes(relative_path: str) -> int:
    try:
        policy = _capture_inventory_policy()
        maximum = int(policy["max_file_bytes"])
        path_limits = policy["max_file_bytes_by_path"]
        if not isinstance(path_limits, dict):
            raise TypeError("trace path byte limits are not an object")
        return min(maximum, int(path_limits.get(relative_path, maximum)))
    except (MemoryError, TypeError, ValueError, KeyError) as exc:
        raise CaptureError(
            f"NVBit trace byte-limit policy is unavailable: {exc}",
            code="trace_snapshot_resource_unavailable",
        ) from exc


def _validate_trace_producer(
    trace_root: Path,
    *,
    reader: immutable.PinnedDirectoryReader | None = None,
    expected_metadata: immutable.MetadataSnapshot | None = None,
) -> dict[str, Any]:
    """Bind producer metadata to one descriptor-retained protobuf inode."""

    owns_reader = reader is None
    stack = contextlib.ExitStack()
    try:
        if reader is None:
            reader = stack.enter_context(
                immutable.PinnedDirectoryReader(trace_root)
            )
        dynamic_limit = _trace_path_max_bytes("dynamic_trace.pb")
        snapshot = reader.read_snapshot(
            "dynamic_trace.pb",
            max_bytes=dynamic_limit,
            expected_metadata=expected_metadata,
        )
        payload = snapshot.payload
        preflight_trace_protobuf(payload)
        trace_pb2 = _import_trace_pb2()
        trace = trace_pb2.Trace()
        trace.ParseFromString(payload)
        metadata = {
            "trace_contract_id": trace.trace_contract_id,
            "memory_payload_classes": sorted(trace.memory_payload_classes),
            "binary_version": trace.binary_version,
            "nvbit_version": trace.nvbit_version,
        }
    except CaptureError:
        raise
    except (
        TraceAdmissionError,
        ProtobufDecodedResourceLimitExceeded,
        ProtobufWireMalformed,
        MemoryError,
        OSError,
        immutable.ImmutableArtifactError,
    ) as exc:
        error = _typed_trace_capture_error("producer-metadata", exc)
        raise error from exc
    except Exception as exc:
        error = _typed_trace_capture_error("producer-metadata", exc)
        raise error from exc
    finally:
        if owns_reader:
            stack.close()
    if metadata["trace_contract_id"] != EXPECTED_TRACE_CONTRACT_ID:
        raise CaptureError(
            "NVBit trace producer/schema contract mismatch",
            code="trace_contract_mismatch",
        )
    if metadata["memory_payload_classes"] != EXPECTED_MEMORY_PAYLOAD_CLASSES:
        raise CaptureError(
            "NVBit trace producer payload-class mismatch",
            code="trace_capabilities_invalid",
        )
    if (
        metadata["binary_version"] != admission.NVBIT_TRACE_BINARY_VERSION
        or metadata["nvbit_version"] != NVBIT_VERSION
    ):
        raise CaptureError(
            "NVBit trace producer version mismatch",
            code="trace_producer_version_mismatch",
        )
    return metadata


def _environment_document(
    preflight: dict[str, Any],
    live: dict[str, Any],
    env: dict[str, str],
    *,
    capture_kind: str,
) -> dict[str, Any]:
    if capture_kind == "ncu":
        selected = dict(env)
    elif capture_kind == "nvbit":
        selected = dict(env)
    else:
        raise CaptureError("environment document capture kind is invalid")
    output_keys = [
        "smi_stdout", "competing_process_stdout", "identity_stdout",
        "nvcc_stdout",
    ]
    if capture_kind == "ncu":
        output_keys.append("ncu_stdout")
    return {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_capture_environment",
        "source_commit": PINNED_COMMIT,
        "preflight_pins_sha256": canonical_sha(preflight),
        "live_checks": {key: value for key, value in live.items() if key not in output_keys},
        "live_outputs": {key: live[key] for key in output_keys},
        "environment": selected,
    }


def _write_new(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with immutable.PinnedDirectoryReader(path.parent) as parent:
            immutable.write_regular_bytes_at(
                parent.descriptor,
                path.name,
                payload,
                mode=0o400,
            )
            parent.verify_path()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"cannot exclusively publish capture output {path}: {exc}"
        ) from exc


def _capture_metadata_paths(
    root: Path,
    *,
    variant_id: str,
    phase: str,
) -> tuple[Path, Path, Path, Path]:
    return (
        root / f"{variant_id}.preflight_{phase}.json",
        root / f"{variant_id}.environment_{phase}.json",
        root / f"{variant_id}.{phase}{admission.RECEIPT_SUFFIX}",
        root / "failures" / variant_id / phase,
    )


def _verify_unchanged(
    retained: immutable.RetainedRegular,
    expected_sha256: str,
    label: str,
) -> None:
    try:
        retained.verify_unchanged()
    except (
        MemoryError,
        OSError,
        immutable.ImmutableArtifactError,
    ) as exc:
        error = _typed_input_error(label, exc, action="generation verification")
        raise error from exc
    if retained.sha256 != expected_sha256:
        raise CaptureError(
            f"{label} changed during capture",
            code="capture_input_integrity_failed",
        )


def _artifact(root: Path, path: Path, role: str) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    with _retained_bounded_regular(
        path,
        f"{role} artifact",
        max_bytes=_artifact_byte_limit(role, relative),
    ) as retained:
        if retained.size_bytes <= 0:
            raise CaptureError(
                f"{role} artifact is zero-sized: {path}",
                code="capture_input_integrity_failed",
            )
        return {
            "role": role,
            "path": relative,
            "sha256": retained.sha256,
            "size_bytes": retained.size_bytes,
        }


def _nvbit_trace_generation_limits(row: dict[str, Any]) -> dict[str, int]:
    policy = _capture_inventory_policy()
    return {
        "max_files": int(policy["max_files"]),
        "max_directories": int(policy["max_directories"]),
        "max_total_file_bytes": min(
            int(policy["max_total_file_bytes"]),
            int(row["trace_budget"]["max_trace_bytes"]),
        ),
        "max_total_relative_path_bytes": int(
            policy["max_total_relative_path_bytes"]
        ),
    }


def _require_bounded_nvbit_snapshot(
    snapshot: immutable.PinnedTreeSnapshot,
) -> None:
    """Enforce admission per-file limits before any semantic decoder runs."""

    try:
        policy = _capture_inventory_policy()
        max_file_bytes = int(policy["max_file_bytes"])
        path_limits = policy["max_file_bytes_by_path"]
        assert isinstance(path_limits, dict)
        threadblock_count = 0
        for relative_path, metadata in snapshot.files:
            file_limit = min(
                max_file_bytes,
                int(path_limits.get(relative_path, max_file_bytes)),
            )
            if metadata[4] > file_limit:
                raise CaptureError(
                    "NVBit pre-admission inventory per-file limit exceeded: "
                    f"{relative_path}: {metadata[4]} > {file_limit}",
                    code="capture_inventory_file_bytes_exceeded",
                )
            if (
                relative_path.startswith("threadblocks/")
                and relative_path.endswith(".pb")
            ):
                threadblock_count += 1
                if (
                    threadblock_count
                    > DEFAULT_TRACE_ADMISSION_LIMITS.threadblock_files
                ):
                    raise CaptureError(
                        "NVBit pre-admission threadblock file-count limit exceeded",
                        code="capture_inventory_file_limit_exceeded",
                    )
    except CaptureError:
        raise
    except MemoryError as exc:
        raise CaptureError(
            "NVBit pre-admission inventory exhausted memory",
            code="capture_inventory_resource_unavailable",
        ) from exc


def _validate_nvbit_trace_generation(
    trace_root: Path,
    *,
    stage: str,
    reader: immutable.PinnedDirectoryReader | None = None,
    dynamic_trace_metadata: immutable.MetadataSnapshot | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        if reader is None:
            admission_result = require_trace_admission(trace_root)
        else:
            admission_result = _require_trace_admission_snapshot(
                trace_root,
                limits=DEFAULT_TRACE_ADMISSION_LIMITS,
                reader=reader,
                historical_only=False,
            ).admission
        producer = _validate_trace_producer(
            trace_root,
            reader=reader,
            expected_metadata=dynamic_trace_metadata,
        )
    except CaptureError:
        raise
    except (
        TraceAdmissionError,
        ProtobufDecodedResourceLimitExceeded,
        ProtobufWireMalformed,
        MemoryError,
        OSError,
    ) as exc:
        error = _typed_trace_capture_error(stage, exc)
        raise error from exc
    except Exception as exc:
        error = _typed_trace_capture_error(stage, exc)
        raise error from exc
    return admission_result, producer


def _require_nvbit_summary_extension(
    reader: immutable.PinnedDirectoryReader,
    source_snapshot: immutable.PinnedTreeSnapshot,
    retained_snapshot: immutable.PinnedTreeSnapshot,
    *,
    summary_relative: str,
    summary_payload: bytes,
) -> None:
    source_files = dict(source_snapshot.files)
    retained_files = dict(retained_snapshot.files)
    if summary_relative in source_files:
        raise CaptureError("NVBit producer tree already contains a trace summary")
    if set(retained_files) != {*source_files, summary_relative}:
        raise CaptureError(
            "NVBit retained trace inventory is not the admitted generation plus summary"
        )
    for relative, metadata in source_files.items():
        if retained_files[relative] != metadata:
            raise CaptureError(
                f"NVBit retained trace generation changed before summary retention: {relative}"
            )

    source_directories = dict(source_snapshot.directories)
    retained_directories = dict(retained_snapshot.directories)
    if set(retained_directories) != set(source_directories):
        raise CaptureError(
            "NVBit retained trace directory inventory changed before summary retention"
        )
    for relative, metadata in source_directories.items():
        retained_metadata = retained_directories[relative]
        if relative == ".":
            if retained_metadata[:4] != metadata[:4]:
                raise CaptureError(
                    "NVBit retained trace root identity changed before summary retention"
                )
        elif retained_metadata != metadata:
            raise CaptureError(
                "NVBit retained trace directory generation changed before summary "
                f"retention: {relative}"
            )

    summary = reader.read_snapshot(
        summary_relative,
        max_bytes=TRACE_SUMMARY_MAX_BYTES,
        expected_metadata=retained_files[summary_relative],
    )
    if summary.payload != summary_payload:
        raise CaptureError("NVBit retained trace summary bytes changed")


def _snapshot_artifact(
    root: Path,
    trace_root: Path,
    reader: immutable.PinnedDirectoryReader,
    metadata_by_relative: dict[str, immutable.MetadataSnapshot],
    relative: str,
    role: str,
    *,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    try:
        metadata = metadata_by_relative[relative]
    except KeyError as exc:
        raise CaptureError(
            f"{role} artifact is absent from retained NVBit generation: {relative}"
        ) from exc
    digest, size_bytes = _hash_retained_snapshot(
        reader,
        relative,
        metadata,
        label=f"{role} artifact",
        max_bytes=_artifact_byte_limit(role, relative),
    )
    if require_nonempty and size_bytes <= 0:
        raise CaptureError(
            f"{role} artifact is zero-sized: {trace_root / relative}",
            code="capture_input_integrity_failed",
        )
    return {
        "role": role,
        "path": (trace_root / relative).relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _snapshot_json_object(
    reader: immutable.PinnedDirectoryReader,
    metadata_by_relative: dict[str, immutable.MetadataSnapshot],
    relative: str,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = metadata_by_relative[relative]
    except KeyError as exc:
        raise CaptureError(
            f"{label} is absent from the retained generation: {relative}",
            code="capture_input_missing",
        ) from exc
    try:
        snapshot = reader.read_snapshot(
            relative,
            max_bytes=max_bytes,
            expected_metadata=metadata,
        )
    except (
        MemoryError,
        OSError,
        immutable.ImmutableArtifactError,
    ) as exc:
        error = _typed_input_error(
            label,
            exc,
            action="descriptor-bound retained read",
        )
        raise error from exc
    value = _strict_json_value(snapshot.payload, label=label)
    if type(value) is not dict:
        raise CaptureError(
            f"{label} is not an object",
            code="capture_input_integrity_failed",
        )
    return value, snapshot.payload


def _retain_nvbit_trace_generation(
    *,
    generation_stack: contextlib.ExitStack,
    root: Path,
    trace_root: Path,
    row: dict[str, Any],
    capture_stdout: bytes,
    capture_stderr: bytes,
    retained_trace_reader: immutable.PinnedDirectoryReader | None = None,
) -> tuple[
    immutable.PinnedDirectoryReader,
    immutable.PinnedTreeSnapshot,
    Any,
    dict[str, Any],
    Path,
    bytes,
]:
    """Admit, summarize, and retain one exact monitored producer generation."""

    if retained_trace_reader is None:
        reader = generation_stack.enter_context(
            immutable.PinnedDirectoryReader(trace_root)
        )
    else:
        reader = retained_trace_reader
        reader.verify_path()
    limits = _nvbit_trace_generation_limits(row)
    try:
        with reader.retain_tree_generation(**limits) as source_snapshot:
            _require_bounded_nvbit_snapshot(source_snapshot)
            if any(
                relative_path == "trace_summary.json"
                for relative_path, _metadata in source_snapshot.files
            ):
                raise CaptureError(
                    "NVBit producer tree already contains a trace summary"
                )
            initial_admission, initial_producer = (
                _validate_nvbit_trace_generation(
                    trace_root,
                    stage="initial retained-generation",
                    reader=reader,
                    dynamic_trace_metadata=dict(source_snapshot.files).get(
                        "dynamic_trace.pb"
                    ),
                )
            )
    except CaptureError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            "NVBit trace changed during initial retained-generation admission: "
            f"{exc}",
            code=(
                "capture_inventory_resource_unavailable"
                if isinstance(exc, immutable.ImmutableArtifactResourceError)
                else "capture_inventory_limit_exceeded"
                if isinstance(exc, immutable.ImmutableArtifactLimitError)
                else "capture_inventory_io_failed"
                if isinstance(exc, OSError)
                else "capture_generation_changed"
            ),
        ) from exc

    summary = {
        "schema_version": 1,
        "producer": admission.NVBIT_TRACE_SUMMARY_PRODUCER,
        "artifact_kind": "cutlass_successor_nvbit_trace_summary",
        "trace_contract_id": initial_producer["trace_contract_id"],
        "memory_payload_classes": initial_producer["memory_payload_classes"],
        "binary_version": initial_producer["binary_version"],
        "nvbit_version": initial_producer["nvbit_version"],
        "trace_contract_status": "pass",
        "expected_record_count": initial_admission.expected_record_count,
        "admitted_record_count": initial_admission.admitted_record_count,
        "threadblock_count": initial_admission.threadblock_count,
        "dynamic_trace": "dynamic_trace.pb",
        "stats": "stats.csv",
        "capture_stdout": (
            _NVBIT_CAPTURE_STDOUT_NAME if capture_stdout else None
        ),
        "capture_stderr": (
            _NVBIT_CAPTURE_STDERR_NAME if capture_stderr else None
        ),
        "capture_stderr_protocol": NVBIT_CAPTURE_STDERR_PROTOCOL,
    }
    summary_path = trace_root / "trace_summary.json"
    summary_payload = (
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    ).encode()
    _write_new(summary_path, summary_payload)

    try:
        retained_snapshot = generation_stack.enter_context(
            reader.retain_tree_generation(**limits)
        )
        _require_bounded_nvbit_snapshot(retained_snapshot)
        _require_nvbit_summary_extension(
            reader,
            source_snapshot,
            retained_snapshot,
            summary_relative=summary_path.name,
            summary_payload=summary_payload,
        )
        retained_admission, retained_producer = (
            _validate_nvbit_trace_generation(
                trace_root,
                stage="retained-generation revalidation",
                reader=reader,
                dynamic_trace_metadata=dict(retained_snapshot.files).get(
                    "dynamic_trace.pb"
                ),
            )
        )
    except CaptureError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"NVBit retained trace generation validation failed: {exc}",
            code=(
                "capture_inventory_resource_unavailable"
                if isinstance(exc, immutable.ImmutableArtifactResourceError)
                else "capture_inventory_limit_exceeded"
                if isinstance(exc, immutable.ImmutableArtifactLimitError)
                else "capture_inventory_io_failed"
                if isinstance(exc, OSError)
                else "capture_generation_changed"
            ),
        ) from exc
    if (
        retained_admission.trace_admission_sha256
        != initial_admission.trace_admission_sha256
        or retained_producer != initial_producer
    ):
        raise CaptureError(
            "NVBit retained trace semantics differ from the summarized generation"
        )
    return (
        reader,
        retained_snapshot,
        retained_admission,
        retained_producer,
        summary_path,
        summary_payload,
    )


def _require_nvbit_stream_artifact_bindings(
    root: Path,
    trace_root: Path,
    artifacts: list[dict[str, Any]],
    *,
    capture_stdout: bytes,
    capture_stderr: bytes,
) -> None:
    for name, payload in (
        (_NVBIT_CAPTURE_STDOUT_NAME, capture_stdout),
        (_NVBIT_CAPTURE_STDERR_NAME, capture_stderr),
    ):
        relative = (trace_root / name).relative_to(root).as_posix()
        matches = [
            artifact for artifact in artifacts
            if artifact["path"] == relative
        ]
        if not payload:
            if matches:
                raise CaptureError(
                    "NVBit capture stream artifact exists for an empty buffer"
                )
            continue
        expected = {
            "role": "trace_auxiliary",
            "path": relative,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
        if matches != [expected]:
            raise CaptureError(
                "NVBit capture stream artifact differs from captured buffer"
            )


def _require_directory_namespace(
    parent: immutable.PinnedDirectoryReader,
    *,
    expected_names: frozenset[str],
    label: str,
) -> None:
    observed = _bounded_directory_names(
        parent,
        max_entries=len(expected_names),
        label=label,
    )
    observed_names = frozenset(observed)
    if len(observed) != len(observed_names) or observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extras = sorted(observed_names - expected_names)
        raise CaptureError(
            f"{label} namespace differs: "
            f"expected={sorted(expected_names)!r} observed={sorted(observed_names)!r}",
            code=(
                "capture_input_missing"
                if missing and not extras
                else "capture_generation_changed"
            ),
        )


def _require_ncu_cell_namespace(
    parent: immutable.PinnedDirectoryReader,
    *,
    expected_names: frozenset[str],
) -> None:
    _require_directory_namespace(
        parent,
        expected_names=expected_names,
        label="NCU handoff",
    )


def _validate_ncu_handoff_manifest(
    document: Any,
    *,
    row: dict[str, Any],
) -> None:
    kernel_signature = hashlib.sha256(
        (
            f"{row['successor_slug']}|"
            f"{row['static_launch_inventory']['binary_path']}"
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
        "kernel_signature_sha256": kernel_signature,
        "kernel_signature_source": "fallback_legacy",
        "arch": ARCH,
        "slug": row["successor_slug"],
        "shape_point": row["shape"]["shape_point"],
        "repeat_index": 0,
        "launch_index": 0,
    }
    if type(document) is not dict or document != expected:
        raise CaptureError(
            "NCU handoff run manifest differs from successor authority",
            code="capture_input_integrity_failed",
        )


def _ncu_observed_launch(
    row: dict[str, Any], observation: dict[str, Any],
) -> dict[str, Any]:
    try:
        admission._require_declared_only_launch_contract(row)
    except admission.AdmissionError as exc:
        raise CaptureError(f"NCU launch authority validation failed: {exc}") from exc
    identity = observation["identity"]
    try:
        device = admission._ncu_device_name(identity.get("Device"))
        block_dim = admission._ncu_dimension(
            identity.get("Block Size"), "Block Size"
        )
        grid_dim = admission._ncu_dimension(
            identity.get("Grid Size"), "Grid Size"
        )
        allowed_kernel_names = admission._sass_kernel_names(row)
    except admission.AdmissionError as exc:
        raise CaptureError(f"NCU launch identity validation failed: {exc}") from exc
    kernel_name = identity.get("Kernel Name")
    if (
        identity.get("ID") != "0"
        or device != GPU_NAME
        or kernel_name not in allowed_kernel_names
    ):
        raise CaptureError("NCU launch identity differs from successor authority")
    return {
        "kernel_name": kernel_name,
        "grid_dim": grid_dim,
        "block_dim": block_dim,
    }


def _nvbit_observed_launch(
    row: dict[str, Any], admission_result: Any,
) -> dict[str, Any]:
    try:
        admission._require_declared_only_launch_contract(row)
    except admission.AdmissionError as exc:
        raise CaptureError(f"NVBit launch authority validation failed: {exc}") from exc
    if len(admission_result.launches) != 1:
        raise CaptureError("NVBit trace must contain exactly one admitted launch")
    launch = admission_result.launches[0]
    try:
        kernel_name = admission._nvbit_kernel_base(launch.kernel_name)
        allowed_kernel_names = admission._sass_kernel_names(row)
    except admission.AdmissionError as exc:
        raise CaptureError(f"NVBit launch identity validation failed: {exc}") from exc
    dimensions = (
        launch.grid_x,
        launch.grid_y,
        launch.grid_z,
        launch.block_x,
        launch.block_y,
        launch.block_z,
    )
    if (
        launch.launch_index != 0
        or launch.device_id != 0
        or kernel_name not in allowed_kernel_names
        or any(type(value) is not int or value <= 0 for value in dimensions)
    ):
        raise CaptureError("NVBit launch identity differs from successor authority")
    return {
        "kernel_name": kernel_name,
        "grid_dim": [launch.grid_x, launch.grid_y, launch.grid_z],
        "block_dim": [launch.block_x, launch.block_y, launch.block_z],
    }


def _build_ncu_sidecar(
    *,
    row: dict[str, Any],
    live: dict[str, Any],
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    imported: parse_ncu_metrics.ImportedReportCsv,
) -> tuple[dict[str, Any], dict[str, Any]]:
    percentage = parse_ncu_metrics.parse_raw_page_csv(
        imported.text,
        _NCU_PERCENTAGE_METRICS,
    )
    if not percentage.ok:
        raise CaptureError(
            "NCU percentage metric extraction failed: "
            f"{percentage.typed_reason}: {percentage.detail}"
        )
    unit_tagged = parse_ncu_metrics.extract_unit_tagged_metrics(
        imported.text,
        metric_registry.load_registry(),
        _NCU_UNIT_METRICS,
    )
    if not unit_tagged.ok:
        raise CaptureError(
            "NCU unit-tagged metric extraction failed: "
            f"{unit_tagged.typed_reason}: {unit_tagged.detail}"
        )
    observation = parse_ncu_metrics.build_launch_observation_document(
        csv_text=imported.text,
        report_sha256=imported.report_sha256,
        requested_launch_skip=0,
        requested_launch_count=1,
    )
    parse_ncu_metrics.validate_launch_observation_document(
        observation,
        requested_launch_skip=0,
        requested_launch_count=1,
        report_sha256=imported.report_sha256,
    )
    _ncu_observed_launch(row, observation)

    sidecar_environment = {
        "SIDECAR_DEVICE": ARCH,
        "SIDECAR_SLUG": row["successor_slug"],
        "SIDECAR_SHAPE_POINT": row["shape"]["shape_point"],
        "SIDECAR_REPEAT_INDEX": "0",
        "SIDECAR_LAUNCH_INDEX": "0",
        "SIDECAR_BINARY_SHA256": row["static_launch_inventory"]["binary_sha256"],
        "SIDECAR_PROFILER_VERSION": expected_ncu_identity.version_line,
        "SIDECAR_CUDA_VERSION": "13.2",
        "SIDECAR_DRIVER_VERSION": live["driver_version"],
        "SIDECAR_GPU_UUID_SHA256": live["gpu_uuid_sha256"],
        "SIDECAR_AGGREGATION": "kernel",
        "SIDECAR_UNITS": "sector_hit_rate_fraction",
        "SIDECAR_CONFIDENCE_LEVEL": "0.95",
        "SIDECAR_WARMUP_COUNT": "0",
        "SIDECAR_HIT_RATE_L2": str(
            percentage.fraction_values["lts__t_sector_hit_rate.pct"]
        ),
        "SIDECAR_HIT_RATE_L1TEX": str(
            percentage.fraction_values["l1tex__t_sector_hit_rate.pct"]
        ),
        "SIDECAR_EVIDENCE_CLASS": "campaign",
        "SIDECAR_NCU_CLOCK_CONTROL": "none",
        "SIDECAR_NCU_CACHE_CONTROL": "all",
        "SIDECAR_NCU_REPLAY_MODE": "kernel",
        "SIDECAR_NCU_TARGET_PROCESSES": "all",
        "SIDECAR_NCU_FILTER_MODE": "global",
        "SIDECAR_NCU_PROFILE_FROM_START": "on",
        "SIDECAR_NCU_LAUNCH_SKIP": "0",
        "SIDECAR_NCU_LAUNCH_COUNT": "1",
        "SIDECAR_HOST_GPU_INDEX": "0",
        "SIDECAR_HOST_PERSISTENCE_MODE": live["persistence_mode"],
        "SIDECAR_HOST_SM_CLOCK_MHZ": str(live["sm_clock_mhz"]),
        "SIDECAR_HOST_MEM_CLOCK_MHZ": str(live["mem_clock_mhz"]),
    }
    sidecar = sidecar_emitter.build_sidecar(sidecar_environment)
    diagnostics = parse_ncu_metrics.diagnostics_to_json_shape(
        parse_ncu_metrics._merged_parser_diagnostics_result(
            percentage,
            unit_tagged,
        )
    )
    if diagnostics:
        sidecar["parser_diagnostics"] = diagnostics
    sidecar["unit_tagged_metrics"] = [
        {
            "ncu_metric": name,
            "unit": unit_tagged.metrics[name]["unit"],
            "value": unit_tagged.metrics[name]["value"],
        }
        for name in _NCU_UNIT_METRICS
    ]
    try:
        from tools.local_labels import import_label

        result = import_label.validate(sidecar)
    except Exception as exc:
        raise CaptureError(f"NCU label sidecar validation failed: {exc}") from exc
    if not result.ok:
        raise CaptureError(
            "NCU label sidecar semantic validation failed: "
            f"{result.typed_reason}"
        )
    return sidecar, observation


def _postprocess_ncu_handoff(
    *,
    root: Path,
    row: dict[str, Any],
    live: dict[str, Any],
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    generation_stack: contextlib.ExitStack | None = None,
) -> None:
    cell = root / ARCH / row["successor_slug"] / "trace_00"
    report_path = cell / "run_0_launch_set.ncu-rep"
    manifest_path = cell / "label_run_manifest_0_launch_set.json"
    observation_path = cell / "run_0_launch_set.launch_observation.json"
    sidecar_path = cell / "label_0_0.json"
    try:
        with contextlib.ExitStack() as retained_stack:
            parent = retained_stack.enter_context(
                immutable.PinnedDirectoryReader(cell)
            )
            _require_ncu_cell_namespace(
                parent,
                expected_names=_NCU_INITIAL_ARTIFACT_NAMES,
            )
            retained_report = retained_stack.enter_context(
                _retained_bounded_regular(
                    report_path,
                    "NCU handoff report",
                    max_bytes=parse_ncu_metrics.NCU_REPORT_MAX_BYTES,
                )
            )
            retained_manifest = retained_stack.enter_context(
                _retained_bounded_regular(
                    manifest_path,
                    "NCU handoff run manifest",
                    max_bytes=NCU_MANIFEST_MAX_BYTES,
                )
            )
            if retained_report.size_bytes <= 0 or retained_manifest.size_bytes <= 0:
                raise CaptureError("NCU handoff artifact is empty")
            active_stack = retained_stack
            if generation_stack is not None:
                generation_stack.enter_context(retained_stack.pop_all())
                active_stack = generation_stack
            manifest_payload = _read_retained_bytes(
                retained_manifest,
                "NCU handoff run manifest",
                max_bytes=NCU_MANIFEST_MAX_BYTES,
            )
            manifest = _strict_json_value(
                manifest_payload,
                label="NCU handoff run manifest",
            )
            _validate_ncu_handoff_manifest(manifest, row=row)
            imported, import_error = parse_ncu_metrics._import_retained_report_csv(
                expected_ncu_identity,
                retained_report,
                require_cgroup_containment=True,
            )
            if import_error is not None or imported is None:
                detail = getattr(import_error, "detail", repr(import_error))
                raise CaptureError(
                    f"NCU handoff report import failed: {detail}",
                    code=(
                        getattr(import_error, "typed_reason", None)
                        or "ncu_import_failed"
                    ),
                )
            if (
                imported.report_sha256 != retained_report.sha256
                or imported.report_size_bytes != retained_report.size_bytes
            ):
                raise CaptureError(
                    "NCU imported report identity differs from retained handoff",
                    code="capture_input_integrity_failed",
                )
            sidecar, observation = _build_ncu_sidecar(
                row=row,
                live=live,
                expected_ncu_identity=expected_ncu_identity,
                imported=imported,
            )
            sidecar_payload = immutable.pretty_json_bytes(sidecar)
            observation_payload = immutable.pretty_json_bytes(observation)

            def verify_handoff() -> None:
                retained_report.verify_unchanged()
                retained_manifest.verify_unchanged()
                parent.verify_path()

            def authorize_observation() -> None:
                verify_handoff()
                with _retained_bounded_regular(
                    observation_path,
                    "committed NCU launch observation",
                    max_bytes=NCU_OBSERVATION_MAX_BYTES,
                ) as committed_observation:
                    if _read_retained_bytes(
                        committed_observation,
                        "committed NCU launch observation",
                        max_bytes=NCU_OBSERVATION_MAX_BYTES,
                    ) != observation_payload:
                        raise CaptureError(
                            "committed NCU launch observation differs from generated bytes",
                            code="capture_input_integrity_failed",
                        )
                    _require_ncu_cell_namespace(
                        parent,
                        expected_names=frozenset({
                            *_NCU_INITIAL_ARTIFACT_NAMES,
                            observation_path.name,
                        }),
                    )
                    committed_observation.verify_unchanged()

            retained_observation = immutable.publish_retained_regular_file(
                cell,
                observation_path.name,
                observation_payload,
                pinned_parent=parent,
                after_commit=authorize_observation,
            )
            retained_observation = active_stack.enter_context(
                _retained_regular_capability(
                    retained_observation,
                    "committed NCU launch observation",
                )
            )

            def verify_observation() -> None:
                verify_handoff()
                retained_observation.verify_unchanged()

            def authorize_sidecar() -> None:
                verify_observation()
                with _retained_bounded_regular(
                    sidecar_path,
                    "committed NCU sidecar",
                    max_bytes=NCU_SIDECAR_MAX_BYTES,
                ) as committed_sidecar:
                    if _read_retained_bytes(
                        committed_sidecar,
                        "committed NCU sidecar",
                        max_bytes=NCU_SIDECAR_MAX_BYTES,
                    ) != sidecar_payload:
                        raise CaptureError(
                            "committed NCU sidecar differs from generated bytes",
                            code="capture_input_integrity_failed",
                        )
                    _require_ncu_cell_namespace(
                        parent,
                        expected_names=_NCU_COMPLETE_ARTIFACT_NAMES,
                    )
                    committed_sidecar.verify_unchanged()

            retained_sidecar = immutable.publish_retained_regular_file(
                cell,
                sidecar_path.name,
                sidecar_payload,
                pinned_parent=parent,
                after_commit=authorize_sidecar,
            )
            retained_sidecar = active_stack.enter_context(
                _retained_regular_capability(
                    retained_sidecar,
                    "committed NCU sidecar",
                )
            )
            verify_observation()
            retained_sidecar.verify_unchanged()
    except CaptureError:
        raise
    except MemoryError as exc:
        raise CaptureError(
            f"NCU handoff postprocessing failed: {exc}",
            code="capture_input_resource_unavailable",
        ) from exc
    except (OSError, immutable.ImmutableArtifactError) as exc:
        error = _typed_input_error(
            "NCU handoff",
            exc,
            action="postprocessing",
        )
        raise error from exc
    except (
        TypeError,
        ValueError,
        KeyError,
        RecursionError,
    ) as exc:
        raise CaptureError(
            f"NCU handoff postprocessing failed: {exc}",
            code="capture_input_integrity_failed",
        ) from exc


def _retain_failed_ncu_handoff(
    root: Path,
    row: dict[str, Any],
    generation_stack: contextlib.ExitStack,
) -> list[dict[str, Any]]:
    cell = root / ARCH / row["successor_slug"] / "trace_00"
    try:
        metadata = cell.lstat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise CaptureError(f"cannot inspect failed NCU handoff root: {exc}") from exc
    if cell.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise CaptureError("failed NCU handoff root is not a pinned directory")
    reader = generation_stack.enter_context(
        immutable.PinnedDirectoryReader(cell)
    )
    limits = {
        "max_files": 64,
        "max_directories": 16,
        "max_total_file_bytes": FAILED_NCU_HANDOFF_MAX_TOTAL_BYTES,
        "max_total_relative_path_bytes": 64 * 1024,
    }
    snapshot = generation_stack.enter_context(
        reader.retain_tree_generation(**limits)
    )
    generation_stack.enter_context(
        _revalidate_tree_generation_on_primary(
            reader,
            snapshot,
            limits=limits,
            label="failed NCU handoff generation",
        )
    )
    roles = {
        "run_0_launch_set.ncu-rep": "ncu_report",
        "label_run_manifest_0_launch_set.json": "label_run_manifest",
        "run_0_launch_set.launch_observation.json": "ncu_launch_observation",
        "label_0_0.json": "label_sidecar",
    }
    artifacts: list[dict[str, Any]] = []
    metadata_by_relative = dict(snapshot.files)
    for relative, _expected_metadata in snapshot.files:
        artifacts.append(
            _snapshot_artifact(
                root,
                cell,
                reader,
                metadata_by_relative,
                relative,
                roles.get(relative, "retained_failed_handoff_artifact"),
                require_nonempty=False,
            )
        )
    return artifacts


def _write_failure_provenance(
    failure_root: Path,
    *,
    root: Path,
    phase: str,
    row: dict[str, Any],
    command: list[str],
    live: dict[str, Any],
    environment_payload: bytes,
    default_environment: dict[str, str],
    capture_producer_sha256: str,
    containment_supervisor_sha256: str,
    phase_tool_sha256: str,
    runtime_manifest_sha256: str | None,
    action_executor_sha256: str | None,
    error: CaptureError,
    retained_phase_tool_path: Path | None = None,
    retained_phase_tool_payload: bytes | None = None,
    retained_runtime_manifest_path: Path | None = None,
    retained_runtime_manifest_payload: bytes | None = None,
    external_build_authority: NvbitExternalBuildAuthority | None = None,
    retained_build_documents: tuple[RetainedNvbitBuildDocument, ...] = (),
    generation_stack: contextlib.ExitStack | None = None,
) -> None:
    if generation_stack is None:
        generation_stack = contextlib.ExitStack()
    try:
        retained_nvbit_authority_artifacts: list[dict[str, Any]] = []
        if phase == "nvbit":
            expected_phase_tool_path = (
                root
                / "capture_tools"
                / f"{row['variant_id']}.nvbit_tracer_tool.so"
            )
            if retained_phase_tool_path != expected_phase_tool_path:
                raise CaptureError(
                    "failed NVBit capture lacks its canonical retained phase tool"
                )
            expected_runtime_manifest_path = (
                root
                / "capture_tools"
                / f"{row['variant_id']}.nvbit_runtime_manifest.json"
            )
            if (
                retained_runtime_manifest_path
                != expected_runtime_manifest_path
                or runtime_manifest_sha256 is None
            ):
                raise CaptureError(
                    "failed NVBit capture lacks its canonical retained runtime manifest"
                )

            def retain_expected_authority(
                path: Path,
                *,
                label: str,
                expected_sha256: str,
                max_bytes: int,
            ) -> immutable.RetainedRegular:
                retained = generation_stack.enter_context(
                    _retained_bounded_regular(
                        path,
                        label,
                        max_bytes=max_bytes,
                    )
                )
                if retained.sha256 != expected_sha256:
                    raise CaptureError(
                        f"{label} identity changed",
                        code="capture_input_integrity_failed",
                    )
                return retained

            try:
                retained_phase_tool = retain_expected_authority(
                    retained_phase_tool_path,
                    label="failed NVBit capture phase tool",
                    expected_sha256=phase_tool_sha256,
                    max_bytes=EXTERNAL_TRACER_MAX_BYTES,
                )
                phase_tool_artifact_path = retained_phase_tool_path
            except (
                CaptureError,
                OSError,
                immutable.ImmutableArtifactError,
            ) as canonical_error:
                if (
                    retained_phase_tool_payload is None
                    or not retained_phase_tool_payload
                    or sha256_bytes(retained_phase_tool_payload)
                    != phase_tool_sha256
                ):
                    raise CaptureError(
                        "failed NVBit capture phase-tool bytes are unavailable"
                    ) from canonical_error
                phase_tool_artifact_path = (
                    root
                    / "capture_tools"
                    / f"{row['variant_id']}.failed_nvbit_tracer_tool.so"
                )
                _write_new(
                    phase_tool_artifact_path,
                    retained_phase_tool_payload,
                )
                retained_phase_tool = retain_expected_authority(
                    phase_tool_artifact_path,
                    label="failed NVBit capture phase tool",
                    expected_sha256=phase_tool_sha256,
                    max_bytes=EXTERNAL_TRACER_MAX_BYTES,
                )
            retained_nvbit_authority_artifacts.append({
                "role": "phase_tool",
                "path": phase_tool_artifact_path.relative_to(root).as_posix(),
                "sha256": retained_phase_tool.sha256,
                "size_bytes": retained_phase_tool.size_bytes,
            })

            try:
                retained_runtime_manifest = retain_expected_authority(
                    retained_runtime_manifest_path,
                    label="failed NVBit capture runtime manifest",
                    expected_sha256=runtime_manifest_sha256,
                    max_bytes=nvbit_runtime.RUNTIME_MANIFEST_MAX_BYTES,
                )
                runtime_manifest_artifact_path = (
                    retained_runtime_manifest_path
                )
            except (
                CaptureError,
                OSError,
                immutable.ImmutableArtifactError,
            ) as canonical_error:
                if (
                    retained_runtime_manifest_payload is None
                    or not retained_runtime_manifest_payload
                    or sha256_bytes(retained_runtime_manifest_payload)
                    != runtime_manifest_sha256
                ):
                    raise CaptureError(
                        "failed NVBit capture runtime-manifest bytes are unavailable"
                    ) from canonical_error
                runtime_manifest_artifact_path = (
                    root
                    / "capture_tools"
                    / f"{row['variant_id']}.failed_nvbit_runtime_manifest.json"
                )
                _write_new(
                    runtime_manifest_artifact_path,
                    retained_runtime_manifest_payload,
                )
                retained_runtime_manifest = retain_expected_authority(
                    runtime_manifest_artifact_path,
                    label="failed NVBit capture runtime manifest",
                    expected_sha256=runtime_manifest_sha256,
                    max_bytes=nvbit_runtime.RUNTIME_MANIFEST_MAX_BYTES,
                )
            retained_nvbit_authority_artifacts.append({
                "role": "runtime_manifest",
                "path": runtime_manifest_artifact_path.relative_to(root).as_posix(),
                "sha256": retained_runtime_manifest.sha256,
                "size_bytes": retained_runtime_manifest.size_bytes,
            })
            if external_build_authority is None:
                raise CaptureError(
                    "failed NVBit capture lacks external build authority"
                )
            retained_by_role = {
                document.role: document for document in retained_build_documents
            }
            for role, filename, payload, max_bytes in (
                external_build_authority.documents()
            ):
                canonical_path = (
                    root / "capture_tools" / f"{row['variant_id']}.{filename}"
                )
                expected_digest = sha256_bytes(payload)
                retained_document = retained_by_role.get(role)
                try:
                    if (
                        retained_document is None
                        or retained_document.path != canonical_path
                    ):
                        retained = retain_expected_authority(
                            canonical_path,
                            label=f"failed NVBit capture {role}",
                            expected_sha256=expected_digest,
                            max_bytes=max_bytes,
                        )
                    else:
                        retained = retained_document.retained
                        retained.verify_unchanged()
                        if retained.sha256 != expected_digest:
                            raise CaptureError(
                                f"failed NVBit capture {role} identity changed",
                                code="capture_input_integrity_failed",
                            )
                    artifact_path = canonical_path
                except (
                    CaptureError,
                    OSError,
                    immutable.ImmutableArtifactError,
                ) as canonical_error:
                    artifact_path = canonical_path.with_name(
                        f"{row['variant_id']}.failed_{filename}"
                    )
                    _write_new(artifact_path, payload)
                    retained = retain_expected_authority(
                        artifact_path,
                        label=f"failed NVBit capture {role}",
                        expected_sha256=expected_digest,
                        max_bytes=max_bytes,
                    )
                    try:
                        canonical_error.add_note(
                            f"retained fallback {artifact_path.name}"
                        )
                    except (AttributeError, TypeError):
                        pass
                retained_nvbit_authority_artifacts.append({
                    "role": role,
                    "path": artifact_path.relative_to(root).as_posix(),
                    "sha256": retained.sha256,
                    "size_bytes": retained.size_bytes,
                })
        elif any(
            value is not None
            for value in (
                runtime_manifest_sha256,
                retained_phase_tool_path,
                retained_phase_tool_payload,
                retained_runtime_manifest_path,
                retained_runtime_manifest_payload,
                external_build_authority,
                *retained_build_documents,
            )
        ):
            raise CaptureError("NCU failure carries NVBit runtime authority")
        failure_root.mkdir(parents=True, exist_ok=True)
        parent = generation_stack.enter_context(
            immutable.PinnedDirectoryReader(failure_root)
        )
        observed_names = _bounded_directory_names(
            parent,
            max_entries=CAPTURE_FAILURE_NAMESPACE_MAX_ENTRIES,
            label="capture failure",
        )
        stream_names = frozenset(observed_names)
        if (
            len(observed_names) != len(stream_names)
            or any(not name.endswith(".bin") for name in stream_names)
        ):
            raise CaptureError(
                "capture failure namespace contains non-stream evidence"
            )
        _require_directory_namespace(
            parent,
            expected_names=stream_names,
            label="capture failure",
        )
        streams = []
        for name in sorted(stream_names):
            path = failure_root / name
            retained_stream = generation_stack.enter_context(
                _retained_bounded_regular(
                    path,
                    "capture failure stream",
                    max_bytes=CAPTURE_STREAM_LIMIT_BYTES,
                )
            )
            streams.append({
                "path": name,
                "sha256": retained_stream.sha256,
                "size_bytes": retained_stream.size_bytes,
            })
        handoff_artifacts = (
            _retain_failed_ncu_handoff(root, row, generation_stack)
            if phase == "ncu"
            else retained_nvbit_authority_artifacts
        )
        failed_command = error.process_command or command
        failed_environment = error.process_environment or default_environment
        failed_tool_sha256 = error.process_tool_sha256 or phase_tool_sha256
        process = {
            "role": error.process_role or f"{phase}_capture",
            "boot_id_sha256": _boot_id_sha256(),
            "command": failed_command,
            "command_sha256": canonical_sha(failed_command),
            "environment": failed_environment,
            "environment_sha256": canonical_sha(failed_environment),
            "host_identity_sha256": _host_identity_sha256(live),
            "pid": error.process_id,
            "started_at_utc": error.started_at_utc,
        }
        if error.containment is not None:
            process["containment"] = error.containment.as_dict()
        provenance = {
            "capture_producer_sha256": capture_producer_sha256,
            "containment_supervisor_sha256": containment_supervisor_sha256,
            "phase_tool_sha256": failed_tool_sha256,
        }
        if action_executor_sha256 is not None:
            provenance["action_executor_sha256"] = action_executor_sha256
        if runtime_manifest_sha256 is not None:
            provenance["runtime_manifest_sha256"] = runtime_manifest_sha256
        if external_build_authority is not None:
            provenance["external_build_authority"] = (
                external_build_authority.provenance()
            )
        document = {
            "schema_version": 1,
            "artifact_kind": "cutlass_successor_capture_failure",
            "status": "failed_unadmitted",
            "capture_kind": phase,
            "matrix_key": row["matrix_key"],
            "variant_id": row["variant_id"],
            "failure_code": error.code,
            "detail": error.detail,
            "capture_environment_document_sha256": sha256_bytes(
                environment_payload
            ),
            "process": process,
            "provenance": provenance,
            "streams": streams,
            "handoff_artifacts": handoff_artifacts,
        }
        document["failure_seal_sha256"] = canonical_sha(document)
        output = failure_root / "failure_provenance.json"
        payload = immutable.pretty_json_bytes(document)

        def close_failure_generation() -> None:
            expected_names = frozenset({*stream_names, output.name})
            max_total_file_bytes = sum(
                stream["size_bytes"] for stream in streams
            ) + len(payload)
            with immutable.PinnedDirectoryReader(failure_root) as final_parent:
                with final_parent.retain_tree_generation(
                    max_files=len(expected_names),
                    max_directories=1,
                    max_total_file_bytes=max_total_file_bytes,
                    max_total_relative_path_bytes=64 * 1024,
                ):
                    _require_directory_namespace(
                        final_parent,
                        expected_names=expected_names,
                        label="capture failure",
                    )
                    _close_generation_stack_preserving(
                        generation_stack,
                        error,
                        label="failure-provenance generation cleanup failed",
                    )

        immutable.publish_regular_file(
            failure_root,
            output.name,
            payload,
            pinned_parent=parent,
            after_commit=close_failure_generation,
        )
    except CaptureError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise CaptureError(
            f"cannot transactionally publish failure provenance: {exc}"
        ) from exc
    finally:
        _close_generation_stack_preserving(
            generation_stack,
            error,
            label="failure-provenance generation cleanup failed",
        )


def _receipt(
    *, root: Path, phase: str, row: dict[str, Any], preflight_path: Path,
    preflight_payload: bytes, environment_path: Path, environment_payload: bytes,
    command: list[str], process: dict[str, Any], artifacts: list[dict[str, Any]],
    semantic: dict[str, Any], observed_launch: dict[str, Any], captured_at: str,
    capture_producer_sha256: str,
    containment_supervisor_sha256: str,
    phase_tool_sha256: str,
    runtime_manifest_sha256: str | None,
    action_executor_sha256: str | None,
    external_build_authority: NvbitExternalBuildAuthority | None,
) -> dict[str, Any]:
    preflight_relative = preflight_path.relative_to(root).as_posix()
    environment_digest = sha256_bytes(environment_payload)
    provenance = {
        "capture_producer_sha256": capture_producer_sha256,
        "containment_supervisor_sha256": containment_supervisor_sha256,
        "phase_tool_sha256": phase_tool_sha256,
    }
    if phase == "ncu":
        if action_executor_sha256 is None:
            raise CaptureError("NCU receipt lacks action-executor provenance")
        if runtime_manifest_sha256 is not None:
            raise CaptureError("NCU receipt carries NVBit runtime-manifest provenance")
        if external_build_authority is not None:
            raise CaptureError("NCU receipt carries NVBit external build provenance")
        provenance["action_executor_sha256"] = action_executor_sha256
    else:
        if action_executor_sha256 is not None:
            raise CaptureError("NVBit receipt carries NCU executor provenance")
        if runtime_manifest_sha256 is None:
            raise CaptureError("NVBit receipt lacks runtime-manifest provenance")
        if external_build_authority is None:
            raise CaptureError("NVBit receipt lacks external build provenance")
        provenance["runtime_manifest_sha256"] = runtime_manifest_sha256
        provenance["external_build_authority"] = (
            external_build_authority.provenance()
        )
    receipt = {
        "schema_version": admission.ACTIVE_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": "cutlass_successor_capture_receipt",
        "receipt": {
            "capture_kind": phase,
            "status": "pass",
            "matrix_key": row["matrix_key"],
            "variant_id": row["variant_id"],
            "identity": {
                "slug": row["successor_slug"],
                "shape_point": row["shape"]["shape_point"],
                "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
                "source_commit": PINNED_COMMIT,
                "kernel_signature_sha256": hashlib.sha256(
                    f"{row['successor_slug']}|{row['static_launch_inventory']['binary_path']}".encode()
                ).hexdigest(),
                "arch": ARCH,
                "ncu_metric_version": admission.DEFAULT_METRIC_VERSION,
            },
            "launch_identity": {
                "launch_index": 0,
                "kernel_name": observed_launch["kernel_name"],
                "kernel_signature_sha256": hashlib.sha256(
                    f"{row['successor_slug']}|{row['static_launch_inventory']['binary_path']}".encode()
                ).hexdigest(),
                "grid_dim": list(observed_launch["grid_dim"]),
                "block_dim": list(observed_launch["block_dim"]),
            },
            "process": process,
            "provenance": provenance,
            "captured_at_utc": captured_at,
            "preflight_path": preflight_relative,
            "preflight_sha256": sha256_bytes(preflight_payload),
            "environment_sha256": environment_digest,
            "command": command,
            "command_sha256": canonical_sha(command),
            "semantic_validation": semantic,
            "artifacts": artifacts,
        },
    }
    launch = receipt["receipt"]["launch_identity"]
    launch["launch_identity_sha256"] = canonical_sha({k: v for k, v in launch.items() if k != "launch_identity_sha256"})
    receipt["receipt"]["artifacts"].extend([
        {"role": "preflight", "path": preflight_relative, "sha256": sha256_bytes(preflight_payload), "size_bytes": len(preflight_payload)},
        {"role": "environment", "path": environment_path.relative_to(root).as_posix(), "sha256": environment_digest, "size_bytes": len(environment_payload)},
    ])
    receipt["receipt_seal_sha256"] = canonical_sha(receipt)
    return receipt


def _finalize_capture_in_generation(
    *,
    generation_stack: contextlib.ExitStack,
    root: Path,
    phase: str,
    row: dict[str, Any],
    live: dict[str, Any],
    pid: int,
    started: str,
    containment: delegated_cgroup.CgroupContainmentEvidence,
    command: list[str],
    capture_source: Path,
    capture_source_retained: immutable.RetainedRegular,
    capture_producer_sha256: str,
    containment_supervisor_retained: immutable.RetainedRegular,
    containment_supervisor_sha256: str,
    phase_tool_sha256: str,
    retained_phase_tool_path: Path | None,
    retained_runtime_manifest_path: Path | None,
    runtime_manifest_sha256: str | None,
    action_executor_retained: immutable.RetainedRegular | None,
    action_executor_sha256: str | None,
    preflight_out: Path,
    preflight_payload: bytes,
    environment_out: Path,
    environment_payload: bytes,
    receipt_path: Path,
    external_tracer_authority: ExternalTracerAuthority | None = None,
    external_tracer_retained: immutable.RetainedRegular | None = None,
    retained_phase_tool: immutable.RetainedRegular | None = None,
    retained_runtime_manifest: immutable.RetainedRegular | None = None,
    external_build_authority: NvbitExternalBuildAuthority | None = None,
    retained_build_documents: tuple[RetainedNvbitBuildDocument, ...] = (),
    capture_stdout: bytes = b"",
    capture_stderr: bytes = b"",
    retained_trace_reader: immutable.PinnedDirectoryReader | None = None,
) -> tuple[Path, bytes]:
    process = {
        "host_identity_sha256": _host_identity_sha256(live),
        "boot_id_sha256": _boot_id_sha256(),
        "pid": pid,
        "started_at_utc": started,
        "command_sha256": canonical_sha(command),
        "containment": containment.as_dict(),
    }
    process["process_identity_sha256"] = canonical_sha(process)
    artifacts: list[dict[str, Any]] = []
    observed_launch: dict[str, Any]
    if phase == "ncu":
        cell = root / ARCH / row["successor_slug"] / "trace_00"
        cell_reader = generation_stack.enter_context(
            immutable.PinnedDirectoryReader(cell)
        )
        _require_ncu_cell_namespace(
            cell_reader,
            expected_names=_NCU_COMPLETE_ARTIFACT_NAMES,
        )
        retained_cell_snapshot = generation_stack.enter_context(
            cell_reader.retain_tree_generation(
                max_files=len(_NCU_COMPLETE_ARTIFACT_NAMES),
                max_directories=1,
                max_total_file_bytes=NCU_HANDOFF_MAX_TOTAL_BYTES,
                max_total_relative_path_bytes=64 * 1024,
            )
        )
        required = {
            "ncu_report": "run_0_launch_set.ncu-rep",
            "label_sidecar": "label_0_0.json",
            "label_run_manifest": "label_run_manifest_0_launch_set.json",
            "ncu_launch_observation": (
                "run_0_launch_set.launch_observation.json"
            ),
        }
        retained_metadata = dict(retained_cell_snapshot.files)
        for role, relative in required.items():
            artifacts.append(
                _snapshot_artifact(
                    root,
                    cell,
                    cell_reader,
                    retained_metadata,
                    relative,
                    role,
                )
            )
        sidecar, _ = _snapshot_json_object(
            cell_reader,
            retained_metadata,
            required["label_sidecar"],
            label="NCU label sidecar",
            max_bytes=NCU_SIDECAR_MAX_BYTES,
        )
        run_manifest, _ = _snapshot_json_object(
            cell_reader,
            retained_metadata,
            required["label_run_manifest"],
            label="NCU label run manifest",
            max_bytes=NCU_MANIFEST_MAX_BYTES,
        )
        _validate_ncu_handoff_manifest(run_manifest, row=row)
        observation, _ = _snapshot_json_object(
            cell_reader,
            retained_metadata,
            required["ncu_launch_observation"],
            label="NCU launch observation",
            max_bytes=NCU_OBSERVATION_MAX_BYTES,
        )
        report_artifact = next(
            artifact for artifact in artifacts if artifact["role"] == "ncu_report"
        )
        try:
            parse_ncu_metrics.validate_launch_observation_document(
                observation,
                requested_launch_skip=0,
                requested_launch_count=1,
                report_sha256=report_artifact["sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError(
                f"NCU launch observation validation failed: {exc}"
            ) from exc
        observed_launch = _ncu_observed_launch(row, observation)
        metrics = {
            item.get("ncu_metric")
            for item in sidecar.get("unit_tagged_metrics", [])
            if isinstance(item, dict)
        }
        if not {
            "gpc__cycles_elapsed.max",
            "gpu__time_duration.sum",
        }.issubset(metrics):
            raise CaptureError("NCU sidecar lacks both required diagnostic metrics")
        semantic = {
            "semantic_kind": "ncu_metric_validation",
            "primary_metric": "gpc__cycles_elapsed.max",
            "secondary_metric": "gpu__time_duration.sum",
            "primary_metric_present": True,
            "secondary_metric_present": True,
            "target_only": True,
            "simulator_config_present": False,
        }
    else:
        _verify_nvbit_tracer_chain(
            external_tracer_authority,
            external_tracer_retained,
            retained_phase_tool,
            phase_tool_sha256,
            stage="pre-admission",
        )
        trace_root = root / "nvbit" / row["successor_slug"]
        if retained_trace_reader is not None:
            retained_trace_reader.verify_path()
        if not _nvbit_capture_stderr_matches(capture_stderr):
            raise CaptureError("NVBit capture stderr differs from closed protocol")
        capture_stdout_path = trace_root / _NVBIT_CAPTURE_STDOUT_NAME
        capture_stderr_path = trace_root / _NVBIT_CAPTURE_STDERR_NAME
        if capture_stdout:
            _write_new(capture_stdout_path, capture_stdout)
        if capture_stderr:
            _write_new(capture_stderr_path, capture_stderr)
        (
            trace_reader,
            retained_trace_snapshot,
            admission_result,
            _producer,
            summary_path,
            _summary_payload,
        ) = _retain_nvbit_trace_generation(
            generation_stack=generation_stack,
            root=root,
            trace_root=trace_root,
            row=row,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr,
            retained_trace_reader=retained_trace_reader,
        )
        _verify_nvbit_tracer_chain(
            external_tracer_authority,
            external_tracer_retained,
            retained_phase_tool,
            phase_tool_sha256,
            stage="post-admission",
        )
        if retained_phase_tool_path is None or retained_phase_tool is None:
            raise CaptureError("NVBit receipt lacks a retained phase tool")
        expected_phase_tool_path = (
            root
            / "capture_tools"
            / f"{row['variant_id']}.nvbit_tracer_tool.so"
        )
        if retained_phase_tool_path != expected_phase_tool_path:
            raise CaptureError("retained NVBit phase-tool path is not canonical")
        artifacts.append({
            "role": "phase_tool",
            "path": retained_phase_tool_path.relative_to(root).as_posix(),
            "sha256": retained_phase_tool.sha256,
            "size_bytes": retained_phase_tool.size_bytes,
        })
        if (
            retained_runtime_manifest_path is None
            or retained_runtime_manifest is None
            or runtime_manifest_sha256 is None
        ):
            raise CaptureError("NVBit receipt lacks retained runtime manifest")
        _verify_runtime_manifest_checkpoint(
            retained_runtime_manifest,
            runtime_manifest_sha256,
            stage="pre-admission",
        )
        artifacts.append({
            "role": "runtime_manifest",
            "path": retained_runtime_manifest_path.relative_to(root).as_posix(),
            "sha256": retained_runtime_manifest.sha256,
            "size_bytes": retained_runtime_manifest.size_bytes,
        })
        if external_build_authority is None or len(retained_build_documents) != 3:
            raise CaptureError("NVBit receipt lacks retained external build authority")
        for document in retained_build_documents:
            document.retained.verify_unchanged()
            if (
                document.retained.sha256 != sha256_bytes(document.payload)
                or document.retained.size_bytes != len(document.payload)
            ):
                raise CaptureError(
                    f"retained NVBit {document.role} changed before admission",
                    code="capture_generation_changed",
                )
            artifacts.append(document.artifact(root))
        observed_launch = _nvbit_observed_launch(row, admission_result)
        retained_files = dict(retained_trace_snapshot.files)
        for relative, role in (
            ("dynamic_trace.pb", "dynamic_trace"),
            ("stats.csv", "stats"),
        ):
            artifacts.append(
                _snapshot_artifact(
                    root,
                    trace_root,
                    trace_reader,
                    retained_files,
                    relative,
                    role,
                )
            )
        threadblock_relatives: set[str] = set()
        for relative in sorted(retained_files):
            parts = Path(relative).parts
            if (
                len(parts) >= 2
                and parts[0] == "threadblocks"
                and Path(relative).suffix == ".pb"
            ):
                threadblock_relatives.add(relative)
                artifacts.append(
                    _snapshot_artifact(
                        root,
                        trace_root,
                        trace_reader,
                        retained_files,
                        relative,
                        "threadblock",
                    )
                )
        artifacts.append(
            _snapshot_artifact(
                root,
                trace_root,
                trace_reader,
                retained_files,
                summary_path.name,
                "trace_summary",
            )
        )
        if not any(item["role"] == "threadblock" for item in artifacts):
            raise CaptureError("NVBit trace has no threadblock payload artifacts")
        typed_paths = {
            "dynamic_trace.pb",
            "stats.csv",
            summary_path.name,
            *threadblock_relatives,
        }
        for relative in sorted(set(retained_files) - typed_paths):
            artifacts.append(
                _snapshot_artifact(
                    root,
                    trace_root,
                    trace_reader,
                    retained_files,
                    relative,
                    "trace_auxiliary",
                )
            )
        _require_nvbit_stream_artifact_bindings(
            root,
            trace_root,
            artifacts,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr,
        )
        semantic = {
            "semantic_kind": "nvbit_trace_validation",
            "trace_contract_status": "pass",
            "dynamic_trace_role": "dynamic_trace",
            "summary_role": "trace_summary",
            "descriptor_bindings": {
                "command_0": "capture_target",
                "CUDA_INJECTION64_PATH": "phase_tool",
                "TRACES_FOLDER": "trace_root",
            },
            "target_only": True,
            "simulator_config_present": False,
        }
        _require_capture_tree_budget(
            trace_root,
            row["trace_budget"]["max_trace_bytes"],
        )
    _verify_unchanged(
        capture_source_retained,
        capture_producer_sha256,
        "capture producer",
    )
    _verify_unchanged(
        containment_supervisor_retained,
        containment_supervisor_sha256,
        "cgroup containment supervisor",
    )
    if phase == "ncu":
        if action_executor_sha256 is None or action_executor_retained is None:
            raise CaptureError("NCU action executor authority is absent")
        _verify_unchanged(
            action_executor_retained,
            action_executor_sha256,
            "NCU action executor",
        )
    if phase == "nvbit":
        if retained_phase_tool_path is None:
            raise CaptureError("retained NVBit phase tool is absent")
        _verify_nvbit_tracer_chain(
            external_tracer_authority,
            external_tracer_retained,
            retained_phase_tool,
            phase_tool_sha256,
            stage="finalization",
        )
        _verify_runtime_manifest_checkpoint(
            retained_runtime_manifest,
            runtime_manifest_sha256,
            stage="finalization",
        )
    captured = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    receipt = _receipt(
        root=root,
        phase=phase,
        row=row,
        preflight_path=preflight_out,
        preflight_payload=preflight_payload,
        environment_path=environment_out,
        environment_payload=environment_payload,
        command=command,
        process=process,
        artifacts=artifacts,
        semantic=semantic,
        observed_launch=observed_launch,
        captured_at=captured,
        capture_producer_sha256=capture_producer_sha256,
        containment_supervisor_sha256=containment_supervisor_sha256,
        phase_tool_sha256=phase_tool_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        action_executor_sha256=action_executor_sha256,
        external_build_authority=external_build_authority,
    )
    try:
        admission._validate_registered_receipt_schema(
            receipt,
            "capture receipt",
        )
    except Exception as exc:
        raise CaptureError(f"capture receipt schema/producer mismatch: {exc}") from exc
    if phase == "nvbit":
        _verify_nvbit_tracer_chain(
            external_tracer_authority,
            external_tracer_retained,
            retained_phase_tool,
            phase_tool_sha256,
            stage="post-finalization",
        )
        _verify_runtime_manifest_checkpoint(
            retained_runtime_manifest,
            runtime_manifest_sha256,
            stage="post-finalization",
        )
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    return receipt_path, payload


def _close_publication_generation(
    generation_stack: contextlib.ExitStack,
    authority_stack: contextlib.ExitStack | None,
) -> None:
    """Close retained output and source capabilities inside rollback guards."""

    primary: BaseException | None = None
    try:
        generation_stack.close()
    except BaseException as exc:
        primary = exc
    try:
        if authority_stack is None:
            raise CaptureError(
                "capture publication authority stack is absent",
                code="capture_generation_changed",
            )
        authority_stack.close()
    except BaseException as raw_close_error:
        close_error = (
            CaptureError(
                "capture publication authority close failed: "
                f"{raw_close_error}",
                code="capture_generation_changed",
            )
            if isinstance(
                raw_close_error,
                (OSError, immutable.ImmutableArtifactError),
            )
            else raw_close_error
        )
        if primary is None:
            primary = close_error
        else:
            _attach_capture_secondary(
                primary,
                close_error,
                "publication authority close also failed",
            )
    if primary is not None:
        raise primary


def _close_nvbit_publication_generation(
    generation_stack: contextlib.ExitStack,
    tracer_authority_stack: contextlib.ExitStack | None,
    authority: ExternalTracerAuthority | None,
    external_tracer_retained: immutable.RetainedRegular | None,
    retained_phase_tool: immutable.RetainedRegular | None,
    phase_tool_sha256: str,
    *,
    authority_stack: contextlib.ExitStack | None = None,
    external_runtime_manifest_retained: immutable.RetainedRegular | None = None,
    runtime_manifest_sha256: str | None = None,
) -> None:
    """Close all generation and tracer capabilities inside rollback guards."""

    primary: BaseException | None = None
    try:
        generation_stack.close()
    except BaseException as exc:
        primary = exc
    try:
        _verify_nvbit_tracer_chain(
            authority,
            external_tracer_retained,
            retained_phase_tool,
            phase_tool_sha256,
            stage="publication after generation cleanup",
        )
    except BaseException as verification_error:
        if primary is None:
            primary = verification_error
        else:
            _attach_capture_secondary(
                primary,
                verification_error,
                "publication tracer revalidation also failed",
            )
    try:
        if external_runtime_manifest_retained is not None:
            _verify_runtime_manifest_checkpoint(
                external_runtime_manifest_retained,
                runtime_manifest_sha256,
                stage="publication after generation cleanup",
            )
        elif authority_stack is not None:
            raise CaptureError(
                "NVBit source runtime-manifest publication authority is absent",
                code="capture_generation_changed",
            )
    except BaseException as verification_error:
        if primary is None:
            primary = verification_error
        else:
            _attach_capture_secondary(
                primary,
                verification_error,
                "publication runtime-manifest revalidation also failed",
            )
    try:
        publication_authority_stack = (
            authority_stack
            if authority_stack is not None
            else tracer_authority_stack
        )
        if tracer_authority_stack is None:
            raise CaptureError(
                "NVBit tracer publication descriptor stack is absent",
                code="capture_generation_changed",
            )
        if publication_authority_stack is None:
            raise CaptureError(
                "NVBit publication authority stack is absent",
                code="capture_generation_changed",
            )
        publication_authority_stack.close()
    except BaseException as raw_close_error:
        close_error = (
            CaptureError(
                "NVBit publication authority close failed: "
                f"{raw_close_error}",
                code="capture_generation_changed",
            )
            if isinstance(
                raw_close_error,
                (OSError, immutable.ImmutableArtifactError),
            )
            else raw_close_error
        )
        if primary is None:
            primary = close_error
        else:
            _attach_capture_secondary(
                primary,
                close_error,
                "publication tracer descriptor close also failed",
            )
    if primary is not None:
        raise primary


def _finalize_capture(
    *,
    generation_stack: contextlib.ExitStack | None = None,
    authority_stack: contextlib.ExitStack | None = None,
    root: Path,
    phase: str,
    row: dict[str, Any],
    live: dict[str, Any],
    pid: int,
    started: str,
    containment: delegated_cgroup.CgroupContainmentEvidence,
    command: list[str],
    capture_source: Path,
    capture_source_retained: immutable.RetainedRegular,
    capture_producer_sha256: str,
    containment_supervisor_retained: immutable.RetainedRegular,
    containment_supervisor_sha256: str,
    phase_tool_sha256: str,
    action_executor_retained: immutable.RetainedRegular | None,
    action_executor_sha256: str | None,
    preflight_out: Path,
    preflight_payload: bytes,
    environment_out: Path,
    environment_payload: bytes,
    receipt_path: Path,
    retained_phase_tool_path: Path | None = None,
    retained_runtime_manifest_path: Path | None = None,
    runtime_manifest_sha256: str | None = None,
    tracer_authority_stack: contextlib.ExitStack | None = None,
    external_tracer_authority: ExternalTracerAuthority | None = None,
    external_tracer_retained: immutable.RetainedRegular | None = None,
    external_runtime_manifest_retained: immutable.RetainedRegular | None = None,
    retained_phase_tool: immutable.RetainedRegular | None = None,
    retained_runtime_manifest: immutable.RetainedRegular | None = None,
    external_build_authority: NvbitExternalBuildAuthority | None = None,
    retained_build_documents: tuple[RetainedNvbitBuildDocument, ...] = (),
    capture_stdout: bytes = b"",
    capture_stderr: bytes = b"",
    retained_trace_reader: immutable.PinnedDirectoryReader | None = None,
) -> Path:
    owns_generation_stack = generation_stack is None
    if generation_stack is None:
        generation_stack = contextlib.ExitStack()
    try:
        output, payload = _finalize_capture_in_generation(
            generation_stack=generation_stack,
            root=root,
            phase=phase,
            row=row,
            live=live,
            pid=pid,
            started=started,
            containment=containment,
            command=command,
            capture_source=capture_source,
            capture_source_retained=capture_source_retained,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_retained=containment_supervisor_retained,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            retained_phase_tool_path=retained_phase_tool_path,
            retained_runtime_manifest_path=retained_runtime_manifest_path,
            runtime_manifest_sha256=runtime_manifest_sha256,
            action_executor_retained=action_executor_retained,
            action_executor_sha256=action_executor_sha256,
            preflight_out=preflight_out,
            preflight_payload=preflight_payload,
            environment_out=environment_out,
            environment_payload=environment_payload,
            receipt_path=receipt_path,
            external_tracer_authority=external_tracer_authority,
            external_tracer_retained=external_tracer_retained,
            retained_phase_tool=retained_phase_tool,
            retained_runtime_manifest=retained_runtime_manifest,
            external_build_authority=external_build_authority,
            retained_build_documents=retained_build_documents,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr,
            retained_trace_reader=retained_trace_reader,
        )
        if phase == "nvbit":
            _verify_nvbit_tracer_chain(
                external_tracer_authority,
                external_tracer_retained,
                retained_phase_tool,
                phase_tool_sha256,
                stage="receipt publication",
            )
            _verify_runtime_manifest_checkpoint(
                retained_runtime_manifest,
                runtime_manifest_sha256,
                stage="receipt publication",
            )
        parent = generation_stack.enter_context(
            immutable.PinnedDirectoryReader(output.parent)
        )
        after_commit = (
            functools.partial(
                _close_nvbit_publication_generation,
                generation_stack,
                tracer_authority_stack,
                external_tracer_authority,
                external_tracer_retained,
                retained_phase_tool,
                phase_tool_sha256,
                authority_stack=authority_stack,
                external_runtime_manifest_retained=(
                    external_runtime_manifest_retained
                ),
                runtime_manifest_sha256=runtime_manifest_sha256,
            )
            if phase == "nvbit"
            else functools.partial(
                _close_publication_generation,
                generation_stack,
                authority_stack,
            )
        )
        try:
            immutable.publish_regular_file(
                output.parent,
                output.name,
                payload,
                pinned_parent=parent,
                after_commit=after_commit,
            )
        except (OSError, immutable.ImmutableArtifactError) as exc:
            raise CaptureError(
                f"cannot transactionally publish capture receipt {output}: {exc}"
            ) from exc
        return output
    finally:
        if owns_generation_stack:
            generation_stack.close()


def _nvbit_capture_stderr_matches(stderr: bytes) -> bool:
    return stderr == b"" or _NVBIT_CAPTURE_STDERR_PATTERN.fullmatch(stderr) is not None


def _capture_protocol_succeeded(
    phase: str,
    *,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
) -> bool:
    if phase == "ncu":
        return (
            returncode == 3
            and stdout == b""
            and stderr == _NCU_HANDOFF_STDERR
        )
    if phase == "nvbit":
        return returncode == 0 and _nvbit_capture_stderr_matches(stderr)
    raise CaptureError("capture result phase is invalid")


def _close_retained_trace_reader_preserving(
    reader: immutable.PinnedDirectoryReader | None,
    primary: BaseException,
) -> None:
    if reader is None:
        return
    reader.__exit__(type(primary), primary, primary.__traceback__)


def _capture_phase_entry(function):
    @functools.wraps(function)
    def guarded(
        *,
        matrix_key: str,
        variant_id: str,
        phase: str,
        preflight_path: Path,
        output_root: Path,
        nvbit_tracer_path: Path | None = None,
        nvbit_tracer_sha256: str | None = None,
        nvbit_runtime_manifest: Path | None = None,
        nvbit_runtime_manifest_sha256: str | None = None,
        nvbit_external_build_bundle: Path | None = None,
        nvbit_reviewed_tracer_sha256: str | None = None,
    ) -> Path:
        if phase not in {"ncu", "nvbit"}:
            raise CaptureError("phase must be ncu or nvbit")
        if phase == "nvbit" and (
            nvbit_tracer_path is None
            or nvbit_tracer_sha256 is None
            or nvbit_runtime_manifest is None
            or nvbit_runtime_manifest_sha256 is None
            or nvbit_external_build_bundle is None
            or nvbit_reviewed_tracer_sha256 is None
        ):
            raise CaptureError(
                "NVBit capture requires exactly --nvbit-tracer, "
                "--nvbit-tracer-sha256, --nvbit-runtime-manifest, "
                "--nvbit-runtime-manifest-sha256, "
                "--nvbit-external-build-bundle, and "
                "--nvbit-reviewed-tracer-sha256"
            )
        if phase == "ncu" and any(value is not None for value in (
            nvbit_tracer_path,
            nvbit_tracer_sha256,
            nvbit_runtime_manifest,
            nvbit_runtime_manifest_sha256,
            nvbit_external_build_bundle,
            nvbit_reviewed_tracer_sha256,
        )):
            raise CaptureError(
                "NCU capture must not receive NVBit tracer/runtime authority arguments"
            )
        tracer_authority = (
            _require_nvbit_tracer_authority(
                nvbit_tracer_path,
                nvbit_tracer_sha256,
            )
            if phase == "nvbit"
            else None
        )
        runtime_manifest_authority = (
            _require_nvbit_runtime_manifest_authority(
                nvbit_runtime_manifest,
                nvbit_runtime_manifest_sha256,
            )
            if phase == "nvbit"
            else None
        )
        root = _validated_capture_output_root(output_root)
        if tracer_authority is not None:
            try:
                tracer_authority.path.relative_to(root)
            except ValueError:
                pass
            else:
                raise CaptureError(
                    "NVBit tracer must be outside the capture output root"
                )
        if runtime_manifest_authority is not None:
            try:
                runtime_manifest_authority[0].relative_to(root)
            except ValueError:
                pass
            else:
                raise CaptureError(
                    "NVBit runtime manifest must be outside the capture output root"
                )
        if nvbit_external_build_bundle is not None:
            build_bundle = _absolute_external(
                nvbit_external_build_bundle,
                "admitted external build bundle",
            )
            try:
                build_bundle.relative_to(root)
            except ValueError:
                pass
            else:
                raise CaptureError(
                    "NVBit admitted build bundle must be outside the capture output root"
                )
        with _capture_phase_reservation(
            matrix_key=matrix_key,
            variant_id=variant_id,
            phase=phase,
            output_root=root,
        ):
            with contextlib.ExitStack() as authority_stack:
                tracer_authority_stack: contextlib.ExitStack | None = None
                external_tracer_retained = None
                external_runtime_manifest_retained = None
                external_build_authority = None
                if tracer_authority is not None:
                    tracer_authority_stack = authority_stack.enter_context(
                        contextlib.ExitStack()
                    )
                    external_tracer_retained = tracer_authority_stack.enter_context(
                        _retained_bounded_regular(
                            tracer_authority.path,
                            "external NVBit tracer",
                            max_bytes=EXTERNAL_TRACER_MAX_BYTES,
                        )
                    )
                    _verify_external_tracer_checkpoint(
                        external_tracer_retained,
                        tracer_authority,
                        stage="initial authority",
                        initial=True,
                    )
                    external_build_authority = (
                        _retain_nvbit_external_build_authority(
                            authority_stack,
                            bundle_path=nvbit_external_build_bundle,
                            reviewed_tracer_sha256=(
                                nvbit_reviewed_tracer_sha256
                            ),
                            tracer_authority=tracer_authority,
                            tracer_retained=external_tracer_retained,
                        )
                    )
                if runtime_manifest_authority is not None:
                    runtime_manifest_path, runtime_manifest_sha256 = (
                        runtime_manifest_authority
                    )
                    external_runtime_manifest_retained = authority_stack.enter_context(
                        _retained_bounded_regular(
                            runtime_manifest_path,
                            "external NVBit runtime manifest",
                            max_bytes=nvbit_runtime.RUNTIME_MANIFEST_MAX_BYTES,
                        )
                    )
                    _verify_runtime_manifest_checkpoint(
                        external_runtime_manifest_retained,
                        runtime_manifest_sha256,
                        stage="initial authority",
                        initial=True,
                    )
                return function(
                    authority_stack=authority_stack,
                    matrix_key=matrix_key,
                    variant_id=variant_id,
                    phase=phase,
                    preflight_path=preflight_path,
                    output_root=root,
                    nvbit_tracer_authority=tracer_authority,
                    nvbit_tracer_stack=tracer_authority_stack,
                    external_tracer_retained=external_tracer_retained,
                    runtime_manifest_authority=runtime_manifest_authority,
                    external_runtime_manifest_retained=(
                        external_runtime_manifest_retained
                    ),
                    external_build_authority=external_build_authority,
                )

    return guarded


@_capture_phase_entry
def capture_phase(
    *,
    authority_stack: contextlib.ExitStack,
    matrix_key: str,
    variant_id: str,
    phase: str,
    preflight_path: Path,
    output_root: Path,
    nvbit_tracer_authority: ExternalTracerAuthority | None = None,
    nvbit_tracer_stack: contextlib.ExitStack | None = None,
    external_tracer_retained: immutable.RetainedRegular | None = None,
    runtime_manifest_authority: tuple[Path, str] | None = None,
    external_runtime_manifest_retained: immutable.RetainedRegular | None = None,
    external_build_authority: NvbitExternalBuildAuthority | None = None,
) -> Path:
    if phase not in {"ncu", "nvbit"}:
        raise CaptureError("phase must be ncu or nvbit")
    if phase == "nvbit" and (
        nvbit_tracer_authority is None
        or nvbit_tracer_stack is None
        or external_tracer_retained is None
        or runtime_manifest_authority is None
        or external_runtime_manifest_retained is None
        or external_build_authority is None
    ):
        raise CaptureError(
            "NVBit capture tracer/runtime/build descriptor authority is absent"
        )
    if phase == "ncu" and any(value is not None for value in (
        nvbit_tracer_authority,
        nvbit_tracer_stack,
        external_tracer_retained,
        runtime_manifest_authority,
        external_runtime_manifest_retained,
        external_build_authority,
    )):
        raise CaptureError(
            "NCU capture received NVBit tracer/runtime/build descriptor authority"
        )
    root = _absolute_external(output_root, "capture output root")
    root.mkdir(parents=True, exist_ok=True)
    row, roster_row = _row_for(matrix_key, variant_id)
    preflight, preflight_payload = validate_successor_preflight(preflight_path, row, roster_row)
    live = validate_live_capture_preflight(phase=phase)
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity | None = None
    if phase == "ncu":
        expected_ncu_identity = _ncu_identity_from_authority(
            live["ncu_authority"]
        )
    capture_source = Path(__file__).resolve(strict=True)
    capture_source_retained = authority_stack.enter_context(
        _retained_bounded_regular(
            capture_source,
            "capture producer",
            max_bytes=CAPTURE_SOURCE_MAX_BYTES,
        )
    )
    capture_source, capture_producer_sha256 = _validate_capture_producer_identity(
        capture_source,
        capture_source_retained,
    )
    containment_source = Path(delegated_cgroup.__file__).resolve(strict=True)
    containment_supervisor_retained = authority_stack.enter_context(
        _retained_bounded_regular(
            containment_source,
            "cgroup containment supervisor",
            max_bytes=TOOL_SOURCE_MAX_BYTES,
        )
    )
    containment_supervisor_sha256 = containment_supervisor_retained.sha256
    env = dict(os.environ)
    command: list[str]
    collection_environment: ncu_executor.NcuCollectionEnvironment | None = None
    driver_identity: ncu_executor.ExpectedExecutableIdentity | None = None
    action_executor_retained: immutable.RetainedRegular | None = None
    action_executor_sha256: str | None = None
    phase_tool_sha256: str | None = None
    retained_phase_tool_path: Path | None = None
    retained_phase_tool: immutable.RetainedRegular | None = None
    retained_runtime_manifest_path: Path | None = None
    retained_runtime_manifest: immutable.RetainedRegular | None = None
    runtime_manifest_sha256: str | None = None
    retained_build_documents: tuple[RetainedNvbitBuildDocument, ...] = ()
    external_build_document_paths: tuple[Path, ...] = ()
    binary = ROOT / row["static_launch_inventory"]["binary_path"]
    process_cwd = binary.parent
    if phase == "ncu":
        if expected_ncu_identity is None:
            raise CaptureError("NCU executable authority was not initialized")
        executor_source = Path(ncu_executor.__file__).resolve(strict=True)
        action_executor_retained = authority_stack.enter_context(
            _retained_bounded_regular(
                executor_source,
                "NCU action executor",
                max_bytes=TOOL_SOURCE_MAX_BYTES,
            )
        )
        action_executor_sha256 = action_executor_retained.sha256
        if action_executor_sha256 != admission.NCU_ACTION_EXECUTOR_SHA256:
            raise CaptureError(
                "NCU action executor differs from pinned authority",
                code="capture_input_integrity_failed",
            )
        for key in (
            "NVBIT_PB_PREFIX", "CUDA_INJECTION64_PATH", "TRACES_FOLDER",
            "USER_DEFINED_FOLDERS",
        ):
            env.pop(key, None)
        env["LD_SOURCE_COMMIT_OVERRIDE"] = PINNED_COMMIT
        command = admission._expected_ncu_command(
            root, row, expected_ncu_identity
        )
        collection_environment = _label_collection_environment(
            env, cwd=process_cwd
        )
        driver_identity = _label_driver_identity()
        phase_tool_sha256 = driver_identity.executable_sha256
        phase_output_dir = root / ARCH / row["successor_slug"] / "trace_00"
    else:
        env = admission._expected_nvbit_environment(root, row)
        trace_root = root / "nvbit" / row["successor_slug"]
        phase_output_dir = trace_root
        command = admission._expected_nvbit_command(row)
        if nvbit_tracer_authority is None:
            raise CaptureError("NVBit tracer authority was not initialized")
        retained_phase_tool_path = (
            root / "capture_tools" / f"{variant_id}.nvbit_tracer_tool.so"
        )
        retained_runtime_manifest_path = (
            root / "capture_tools" / f"{variant_id}.nvbit_runtime_manifest.json"
        )
        if runtime_manifest_authority is None:
            raise CaptureError("NVBit runtime manifest authority was not initialized")
        runtime_manifest_sha256 = runtime_manifest_authority[1]
        if external_build_authority is None:
            raise CaptureError("NVBit external build authority was not initialized")
        external_build_document_paths = tuple(
            root / "capture_tools" / f"{variant_id}.{filename}"
            for _role, filename, _payload, _max_bytes
            in external_build_authority.documents()
        )
    preflight_out, environment_out, receipt_path, failure_root = (
        _capture_metadata_paths(root, variant_id=variant_id, phase=phase)
    )
    reserved_files = [preflight_out, environment_out, receipt_path]
    if retained_phase_tool_path is not None:
        reserved_files.extend((
            retained_phase_tool_path,
            root
            / "capture_tools"
            / f"{variant_id}.failed_nvbit_tracer_tool.so",
        ))
    if retained_runtime_manifest_path is not None:
        reserved_files.extend((
            retained_runtime_manifest_path,
            root
            / "capture_tools"
            / f"{variant_id}.failed_nvbit_runtime_manifest.json",
        ))
    for path in external_build_document_paths:
        reserved_files.extend((
            path,
            path.with_name(f"{variant_id}.failed_{path.name.removeprefix(f'{variant_id}.')}")
        ))
    _reject_preexisting_output_descendants(
        root,
        output_dir=phase_output_dir,
        reserved_files=tuple(reserved_files),
    )
    _reject_preexisting_output_descendants(
        root,
        output_dir=failure_root,
        reserved_files=(failure_root / "failure_provenance.json",),
    )
    retained_phase_tool_payload: bytes | None = None
    retained_runtime_manifest_payload: bytes | None = None
    recorded_environment = (
        collection_environment.as_dict()
        if collection_environment is not None
        else env
    )
    environment_doc = _environment_document(
        preflight,
        live,
        recorded_environment,
        capture_kind=phase,
    )
    environment_payload = (
        json.dumps(environment_doc, sort_keys=True, indent=2) + "\n"
    ).encode()
    pid: int | None = None
    started: str | None = None
    stdout = b""
    stderr = b""
    containment: delegated_cgroup.CgroupContainmentEvidence | None = None
    retained_trace_reader: immutable.PinnedDirectoryReader | None = None
    try:
        if phase == "nvbit":
            if (
                nvbit_tracer_authority is None
                or external_tracer_retained is None
                or retained_phase_tool_path is None
                or external_runtime_manifest_retained is None
                or retained_runtime_manifest_path is None
                or runtime_manifest_sha256 is None
                or external_build_authority is None
                or len(external_build_document_paths) != 3
            ):
                raise CaptureError(
                    "NVBit tracer/runtime/build authority was not initialized"
                )
            _verify_external_tracer_checkpoint(
                external_tracer_retained,
                nvbit_tracer_authority,
                stage="pre-copy",
            )
            phase_tool_sha256 = nvbit_tracer_authority.sha256
            retained_phase_tool_payload = _read_retained_bytes(
                external_tracer_retained,
                "external NVBit tracer",
                max_bytes=EXTERNAL_TRACER_MAX_BYTES,
            )
            _verify_external_tracer_checkpoint(
                external_tracer_retained,
                nvbit_tracer_authority,
                stage="copy snapshot",
            )
            _verify_runtime_manifest_checkpoint(
                external_runtime_manifest_retained,
                runtime_manifest_sha256,
                stage="pre-copy",
            )
            retained_runtime_manifest_payload = _read_retained_bytes(
                external_runtime_manifest_retained,
                "external NVBit runtime manifest",
                max_bytes=nvbit_runtime.RUNTIME_MANIFEST_MAX_BYTES,
            )
            _verify_runtime_manifest_checkpoint(
                external_runtime_manifest_retained,
                runtime_manifest_sha256,
                stage="copy snapshot",
            )
        if phase_tool_sha256 is None:
            raise CaptureError("capture phase-tool authority was not initialized")
        if phase == "nvbit":
            if (
                retained_phase_tool_path is None
                or retained_phase_tool_payload is None
                or retained_runtime_manifest_path is None
                or retained_runtime_manifest_payload is None
                or runtime_manifest_sha256 is None
            ):
                raise CaptureError("NVBit tracer/runtime snapshot authority is absent")
            _write_new(retained_phase_tool_path, retained_phase_tool_payload)
            _write_new(
                retained_runtime_manifest_path,
                retained_runtime_manifest_payload,
            )
            if nvbit_tracer_stack is None:
                raise CaptureError("NVBit tracer descriptor stack is absent")
            if external_build_authority is None:
                raise CaptureError("NVBit external build snapshot authority is absent")
            if len(external_build_document_paths) != len(
                external_build_authority.documents()
            ):
                raise CaptureError("NVBit external build document paths are incomplete")
            build_documents: list[RetainedNvbitBuildDocument] = []
            for path, (role, _filename, payload, max_bytes) in zip(
                external_build_document_paths,
                external_build_authority.documents(),
                strict=True,
            ):
                _write_new(path, payload)
                retained_document = nvbit_tracer_stack.enter_context(
                    _retained_bounded_regular(
                        path,
                        f"retained NVBit {role}",
                        max_bytes=max_bytes,
                    )
                )
                if (
                    retained_document.sha256 != sha256_bytes(payload)
                    or retained_document.size_bytes != len(payload)
                ):
                    raise CaptureError(
                        f"retained NVBit {role} differs from admitted build",
                        code="capture_generation_changed",
                    )
                build_documents.append(
                    RetainedNvbitBuildDocument(
                        role=role,
                        path=path,
                        payload=payload,
                        retained=retained_document,
                    )
                )
            retained_build_documents = tuple(build_documents)
            retained_phase_tool = nvbit_tracer_stack.enter_context(
                _retained_bounded_regular(
                    retained_phase_tool_path,
                    "retained NVBit tracer",
                    max_bytes=EXTERNAL_TRACER_MAX_BYTES,
                )
            )
            retained_runtime_manifest = nvbit_tracer_stack.enter_context(
                _retained_bounded_regular(
                    retained_runtime_manifest_path,
                    "retained NVBit runtime manifest",
                    max_bytes=nvbit_runtime.RUNTIME_MANIFEST_MAX_BYTES,
                )
            )
            _verify_runtime_manifest_checkpoint(
                retained_runtime_manifest,
                runtime_manifest_sha256,
                stage="post-copy",
            )
            _verify_nvbit_tracer_chain(
                nvbit_tracer_authority,
                external_tracer_retained,
                retained_phase_tool,
                phase_tool_sha256,
                stage="post-copy",
            )
            trace_root.mkdir(parents=True, exist_ok=True)
            try:
                retained_trace_reader = immutable.PinnedDirectoryReader(trace_root)
                retained_trace_reader.__enter__()
            except (OSError, MemoryError, immutable.ImmutableArtifactError) as exc:
                if retained_trace_reader is not None:
                    try:
                        retained_trace_reader.close()
                    except BaseException as cleanup:
                        try:
                            exc.add_note(
                                "NVBit trace-root pin cleanup also failed: "
                                f"{cleanup}"
                            )
                        except (AttributeError, TypeError):
                            pass
                    retained_trace_reader = None
                raise _typed_trace_capture_error(
                    "output-root pin", exc
                ) from exc
        _write_new(preflight_out, preflight_payload)
        _write_new(environment_out, environment_payload)
        with _retained_bounded_regular(
            binary,
            "capture target",
            max_bytes=REPOSITORY_INPUT_MAX_BYTES,
        ) as retained_target:
            _require_retained_executable(retained_target, "capture target")
            if (
                retained_target.sha256
                != row["static_launch_inventory"]["binary_sha256"]
            ):
                raise CaptureError(
                    "capture target differs from pinned binary authority",
                    code="capture_input_integrity_failed",
                )
            if phase == "ncu":
                if (
                    collection_environment is None
                    or driver_identity is None
                    or expected_ncu_identity is None
                ):
                    raise CaptureError("NCU execution authority was not initialized")
                (
                    returncode, pid, started, stdout, stderr, containment,
                ) = _run_ncu_capture_command(
                    command,
                    expected_ncu_identity,
                    collection_environment,
                    driver_identity,
                    retained_target,
                    process_cwd,
                    timeout_s=CAPTURE_TIMEOUT_S,
                    failure_root=failure_root,
                )
            else:
                if (
                    retained_phase_tool_path is None
                    or retained_phase_tool is None
                    or retained_runtime_manifest_path is None
                    or retained_runtime_manifest is None
                    or runtime_manifest_sha256 is None
                ):
                    raise CaptureError("retained NVBit tracer/runtime authority is absent")
                if retained_trace_reader is None:
                    raise CaptureError(
                        "NVBit trace-root descriptor authority is absent",
                        code="capture_generation_changed",
                    )
                _verify_nvbit_tracer_chain(
                    nvbit_tracer_authority,
                    external_tracer_retained,
                    retained_phase_tool,
                    phase_tool_sha256,
                    stage="pre-launch",
                )
                trace_proc_path = Path(
                    f"/proc/self/fd/{retained_trace_reader.descriptor}"
                )
                sanitized_runtime_environment = (
                    nvbit_runtime.sanitized_capture_environment(env)
                )
                with nvbit_runtime.retained_tracer_runtime_env(
                    retained_phase_tool_path,
                    base_env=sanitized_runtime_environment,
                    pb_lib=NVBIT_PB_LIB,
                    runtime_manifest=retained_runtime_manifest_path,
                    retained_descriptor=retained_phase_tool.descriptor,
                    retained_manifest_descriptor=(
                        retained_runtime_manifest.descriptor
                    ),
                ) as retained_runtime:
                    runtime_environment = dict(retained_runtime.environment)
                    retained_runtime_descriptors = retained_runtime.pass_fds
                    expected_runtime_names = set(
                        sanitized_runtime_environment
                    ) | {"LD_LIBRARY_PATH"}
                    expected_runtime_paths = {
                        f"/proc/self/fd/{descriptor}"
                        for descriptor in retained_runtime_descriptors
                    }
                    if (
                        set(runtime_environment) != expected_runtime_names
                        or runtime_environment.get("LD_LIBRARY_PATH")
                        not in expected_runtime_paths
                    ):
                        raise CaptureError(
                            "NVBit runtime environment differs from closed pins"
                        )
                    expected_runtime_environment = dict(
                        sanitized_runtime_environment
                    )
                    expected_runtime_environment["LD_LIBRARY_PATH"] = env[
                        "LD_LIBRARY_PATH"
                    ]
                    contract_environment = dict(runtime_environment)
                    contract_environment["LD_LIBRARY_PATH"] = env[
                        "LD_LIBRARY_PATH"
                    ]
                    if contract_environment != expected_runtime_environment:
                        raise CaptureError(
                            "NVBit runtime environment differs from closed pins"
                        )
                    for name in nvbit_runtime.NVBIT_CAPTURE_ENVIRONMENT_CONTROLS:
                        if name in env and name != "LD_LIBRARY_PATH":
                            runtime_environment[name] = env[name]
                    runtime_environment["CUDA_INJECTION64_PATH"] = (
                        retained_phase_tool.proc_path
                    )
                    runtime_environment["TRACES_FOLDER"] = str(trace_proc_path)
                    expected_child_environment = dict(env)
                    for name in (
                        "CUDA_INJECTION64_PATH",
                        "LD_LIBRARY_PATH",
                        "TRACES_FOLDER",
                    ):
                        expected_child_environment[name] = runtime_environment[
                            name
                        ]
                    if runtime_environment != expected_child_environment:
                        raise CaptureError(
                            "NVBit child environment differs from closed pins"
                        )
                    runtime_command = [retained_target.proc_path, *command[1:]]
                    (
                        returncode, pid, started, stdout, stderr, containment,
                    ) = _run_capture_command(
                        runtime_command,
                        runtime_environment,
                        process_cwd,
                        timeout_s=float(row["trace_budget"]["max_wall_time_s"]),
                        byte_root=trace_proc_path,
                        byte_root_descriptor=retained_trace_reader.descriptor,
                        byte_budget=row["trace_budget"]["max_trace_bytes"],
                        failure_root=failure_root,
                        inherited_descriptors=tuple(dict.fromkeys((
                            retained_target.descriptor,
                            retained_phase_tool.descriptor,
                            retained_trace_reader.descriptor,
                            *retained_runtime_descriptors,
                        ))),
                        require_cgroup_containment=True,
                    )
                    _verify_nvbit_tracer_chain(
                        nvbit_tracer_authority,
                        external_tracer_retained,
                        retained_phase_tool,
                        phase_tool_sha256,
                        stage="post-launch",
                    )
                    _verify_runtime_manifest_checkpoint(
                        retained_runtime_manifest,
                        runtime_manifest_sha256,
                        stage="post-launch",
                    )
                    if retained_trace_reader is None:
                        raise CaptureError(
                            "NVBit trace-root descriptor authority is absent",
                            code="capture_generation_changed",
                        )
                    try:
                        retained_trace_reader.verify_path()
                    except immutable.ImmutableArtifactError as exc:
                        raise CaptureError(
                            f"NVBit trace root changed during capture: {exc}",
                            code="capture_generation_changed",
                        ) from exc
    except (
        OSError,
        immutable.ImmutableArtifactError,
        nvbit_runtime.TracerRuntimeError,
    ) as raw_error:
        if phase_tool_sha256 is None:
            exc = CaptureError(
                f"cannot snapshot capture phase-tool authority: {raw_error}"
            )
            _close_retained_trace_reader_preserving(
                retained_trace_reader,
                exc,
            )
            retained_trace_reader = None
            raise exc from raw_error
        exc = CaptureError(
            f"retained capture authority failed: {raw_error}",
            process_id=pid,
            started_at_utc=started,
            process_role=(None if pid is None else f"{phase}_capture"),
            process_command=(None if pid is None else list(command)),
            process_environment=(
                None if pid is None else dict(recorded_environment)
            ),
            process_tool_sha256=(None if pid is None else phase_tool_sha256),
            containment=containment,
        )
        _close_retained_trace_reader_preserving(retained_trace_reader, exc)
        retained_trace_reader = None
        _write_failure_provenance(
            failure_root,
            root=root,
            phase=phase,
            row=row,
            command=command,
            live=live,
            environment_payload=environment_payload,
            default_environment=recorded_environment,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
            action_executor_sha256=action_executor_sha256,
            retained_phase_tool_path=retained_phase_tool_path,
            retained_phase_tool_payload=retained_phase_tool_payload,
            retained_runtime_manifest_path=retained_runtime_manifest_path,
            retained_runtime_manifest_payload=(
                retained_runtime_manifest_payload
            ),
            external_build_authority=external_build_authority,
            retained_build_documents=retained_build_documents,
            error=exc,
        )
        raise exc from raw_error
    except CaptureError as exc:
        _close_retained_trace_reader_preserving(retained_trace_reader, exc)
        retained_trace_reader = None
        if phase_tool_sha256 is None:
            raise
        if exc.process_id is not None and exc.process_command is None:
            exc.process_role = f"{phase}_capture"
            exc.process_command = list(command)
            exc.process_environment = dict(recorded_environment)
            exc.process_tool_sha256 = phase_tool_sha256
        if exc.containment is None:
            exc.containment = containment
        _write_failure_provenance(
            failure_root,
            root=root,
            phase=phase,
            row=row,
            command=command,
            live=live,
            environment_payload=environment_payload,
            default_environment=recorded_environment,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
            action_executor_sha256=action_executor_sha256,
            retained_phase_tool_path=retained_phase_tool_path,
            retained_phase_tool_payload=retained_phase_tool_payload,
            retained_runtime_manifest_path=retained_runtime_manifest_path,
            retained_runtime_manifest_payload=(
                retained_runtime_manifest_payload
            ),
            external_build_authority=external_build_authority,
            retained_build_documents=retained_build_documents,
            error=exc,
        )
        raise
    except BaseException as exc:
        _close_retained_trace_reader_preserving(retained_trace_reader, exc)
        retained_trace_reader = None
        raise
    capture_protocol_ok = _capture_protocol_succeeded(
        phase,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if not capture_protocol_ok:
        _write_new(failure_root / "capture_failure.stdout.bin", stdout)
        _write_new(failure_root / "capture_failure.stderr.bin", stderr)
        failure = CaptureError(
            f"{phase} capture failed returncode={returncode} "
            f"stdout={stdout[-400:]!r} stderr={stderr[-400:]!r}",
            process_id=pid,
            started_at_utc=started,
            process_role=f"{phase}_capture",
            process_command=list(command),
            process_environment=dict(recorded_environment),
            process_tool_sha256=phase_tool_sha256,
            containment=containment,
        )
        _close_retained_trace_reader_preserving(retained_trace_reader, failure)
        retained_trace_reader = None
        _write_failure_provenance(
            failure_root,
            root=root,
            phase=phase,
            row=row,
            command=command,
            live=live,
            environment_payload=environment_payload,
            default_environment=recorded_environment,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
            action_executor_sha256=action_executor_sha256,
            retained_phase_tool_path=retained_phase_tool_path,
            retained_phase_tool_payload=retained_phase_tool_payload,
            retained_runtime_manifest_path=retained_runtime_manifest_path,
            retained_runtime_manifest_payload=(
                retained_runtime_manifest_payload
            ),
            external_build_authority=external_build_authority,
            retained_build_documents=retained_build_documents,
            error=failure,
        )
        raise failure
    if pid is None or started is None or containment is None:
        error = CaptureError(
            "successful capture has incomplete process or containment provenance"
        )
        _close_retained_trace_reader_preserving(retained_trace_reader, error)
        retained_trace_reader = None
        raise error
    generation_stack = contextlib.ExitStack()
    finalization_trace_reader: immutable.PinnedDirectoryReader | None = None
    try:
        if phase == "ncu":
            if expected_ncu_identity is None:
                raise CaptureError(
                    "successful NCU handoff has no executable authority"
                )
            _postprocess_ncu_handoff(
                root=root,
                row=row,
                live=live,
                expected_ncu_identity=expected_ncu_identity,
                generation_stack=generation_stack,
            )
        else:
            if retained_trace_reader is None:
                raise CaptureError(
                    "successful NVBit capture lost trace-root descriptor authority",
                    code="capture_generation_changed",
                )
            finalization_trace_reader = generation_stack.enter_context(
                retained_trace_reader
            )
            retained_trace_reader = None
        return _finalize_capture(
            generation_stack=generation_stack,
            authority_stack=authority_stack,
            root=root,
            phase=phase,
            row=row,
            live=live,
            pid=pid,
            started=started,
            containment=containment,
            command=command,
            capture_source=capture_source,
            capture_source_retained=capture_source_retained,
            capture_producer_sha256=capture_producer_sha256,
            containment_supervisor_retained=containment_supervisor_retained,
            containment_supervisor_sha256=containment_supervisor_sha256,
            phase_tool_sha256=phase_tool_sha256,
            retained_phase_tool_path=retained_phase_tool_path,
            retained_runtime_manifest_path=retained_runtime_manifest_path,
            runtime_manifest_sha256=runtime_manifest_sha256,
            action_executor_retained=action_executor_retained,
            action_executor_sha256=action_executor_sha256,
            preflight_out=preflight_out,
            preflight_payload=preflight_payload,
            environment_out=environment_out,
            environment_payload=environment_payload,
            receipt_path=receipt_path,
            tracer_authority_stack=nvbit_tracer_stack,
            external_tracer_authority=nvbit_tracer_authority,
            external_tracer_retained=external_tracer_retained,
            external_runtime_manifest_retained=(
                external_runtime_manifest_retained
            ),
            retained_phase_tool=retained_phase_tool,
            retained_runtime_manifest=retained_runtime_manifest,
            external_build_authority=external_build_authority,
            retained_build_documents=retained_build_documents,
            capture_stdout=stdout,
            capture_stderr=stderr,
            retained_trace_reader=finalization_trace_reader,
        )
    except BaseException as raw_error:
        if isinstance(raw_error, CaptureError):
            exc = raw_error
        elif phase == "nvbit" and isinstance(
            raw_error,
            (
                TraceAdmissionError,
                ProtobufDecodedResourceLimitExceeded,
                ProtobufWireMalformed,
                MemoryError,
                OSError,
            ),
        ):
            exc = _typed_trace_capture_error("post-capture", raw_error)
        elif isinstance(raw_error, MemoryError):
            exc = CaptureError(
                f"post-capture validation failed: {raw_error}",
                code="capture_resource_unavailable",
            )
        else:
            exc = CaptureError(
                f"post-capture validation failed: {raw_error}",
                code="post_capture_validation_failed",
            )
        _close_retained_trace_reader_preserving(retained_trace_reader, exc)
        retained_trace_reader = None
        try:
            diagnostic_failures = []
            for path, payload in (
                (failure_root / "post_validation.stdout.bin", stdout),
                (failure_root / "post_validation.stderr.bin", stderr),
            ):
                try:
                    _write_new(path, payload)
                except Exception as diagnostic_error:
                    diagnostic_failures.append(
                        f"{path.name}: {diagnostic_error}"
                    )
            if diagnostic_failures:
                exc.detail += (
                    "; post-validation stream retention failed: "
                    + "; ".join(diagnostic_failures)
                )
            exc.process_id = pid
            exc.started_at_utc = started
            exc.process_role = f"{phase}_postprocess_validation"
            exc.process_command = list(command)
            exc.process_environment = dict(recorded_environment)
            exc.process_tool_sha256 = phase_tool_sha256
            exc.containment = containment
            _write_failure_provenance(
                failure_root,
                root=root,
                phase=phase,
                row=row,
                command=command,
                live=live,
                environment_payload=environment_payload,
                default_environment=recorded_environment,
                capture_producer_sha256=capture_producer_sha256,
                containment_supervisor_sha256=containment_supervisor_sha256,
                phase_tool_sha256=phase_tool_sha256,
                runtime_manifest_sha256=runtime_manifest_sha256,
                action_executor_sha256=action_executor_sha256,
                retained_phase_tool_path=retained_phase_tool_path,
                retained_phase_tool_payload=retained_phase_tool_payload,
                retained_runtime_manifest_path=retained_runtime_manifest_path,
                retained_runtime_manifest_payload=(
                    retained_runtime_manifest_payload
                ),
                external_build_authority=external_build_authority,
                retained_build_documents=retained_build_documents,
                error=exc,
                generation_stack=generation_stack,
            )
        finally:
            _close_generation_stack_preserving(
                generation_stack,
                exc,
                label="post-validation generation cleanup failed",
            )
        if not isinstance(raw_error, Exception):
            raise
        if exc is raw_error:
            raise
        raise exc from raw_error


class _StoreUnique(argparse.Action):
    """Reject repeated authority options instead of silently taking the last."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} must be supplied exactly once")
        setattr(namespace, self.dest, values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix-key", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--phase", choices=("ncu", "nvbit"), required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--nvbit-tracer",
        type=Path,
        action=_StoreUnique,
        help=(
            "absolute symlink-free non-Git tracer_tool.so; required exactly "
            "once with --nvbit-tracer-sha256 for the NVBit phase"
        ),
    )
    parser.add_argument(
        "--nvbit-tracer-sha256",
        action=_StoreUnique,
        help=(
            "exact lowercase SHA-256 for --nvbit-tracer; required exactly "
            "once for the NVBit phase"
        ),
    )
    parser.add_argument(
        "--nvbit-reviewed-tracer-sha256",
        action=_StoreUnique,
        help=(
            "independently reviewed exact tracer SHA-256 recorded by the "
            "admitted build bundle; required exactly once for the NVBit phase"
        ),
    )
    parser.add_argument(
        "--nvbit-external-build-bundle",
        type=Path,
        action=_StoreUnique,
        help=(
            "absolute external admitted build bundle containing source_manifest.json, "
            "build_receipt.json, artifact_identity.json, and tracer_tool.so"
        ),
    )
    parser.add_argument(
        "--nvbit-runtime-manifest",
        type=Path,
        action=_StoreUnique,
        help=(
            "absolute external exact runtime DSO manifest; required exactly "
            "once for the NVBit phase"
        ),
    )
    parser.add_argument(
        "--nvbit-runtime-manifest-sha256",
        action=_StoreUnique,
        help=(
            "exact lowercase SHA-256 for --nvbit-runtime-manifest; required "
            "exactly once for the NVBit phase"
        ),
    )
    args = parser.parse_args(argv)
    if args.phase == "nvbit" and any(value is None for value in (
        args.nvbit_tracer,
        args.nvbit_tracer_sha256,
        args.nvbit_runtime_manifest,
        args.nvbit_runtime_manifest_sha256,
        args.nvbit_external_build_bundle,
        args.nvbit_reviewed_tracer_sha256,
    )):
        parser.error(
            "--phase nvbit requires exactly --nvbit-tracer, "
            "--nvbit-tracer-sha256, --nvbit-runtime-manifest, "
            "--nvbit-runtime-manifest-sha256, --nvbit-external-build-bundle, "
            "and --nvbit-reviewed-tracer-sha256"
        )
    if args.phase != "nvbit" and any(value is not None for value in (
        args.nvbit_tracer,
        args.nvbit_tracer_sha256,
        args.nvbit_runtime_manifest,
        args.nvbit_runtime_manifest_sha256,
        args.nvbit_external_build_bundle,
        args.nvbit_reviewed_tracer_sha256,
    )):
        parser.error(
            "NVBit tracer/runtime authority arguments are valid only with "
            "--phase nvbit"
        )
    try:
        output = capture_phase(
            matrix_key=args.matrix_key,
            variant_id=args.variant_id,
            phase=args.phase,
            preflight_path=args.preflight,
            output_root=args.output_root,
            nvbit_tracer_path=args.nvbit_tracer,
            nvbit_tracer_sha256=args.nvbit_tracer_sha256,
            nvbit_runtime_manifest=args.nvbit_runtime_manifest,
            nvbit_runtime_manifest_sha256=(
                args.nvbit_runtime_manifest_sha256
            ),
            nvbit_external_build_bundle=args.nvbit_external_build_bundle,
            nvbit_reviewed_tracer_sha256=(
                args.nvbit_reviewed_tracer_sha256
            ),
        )
    except (CaptureError, OSError, ValueError) as exc:
        print(f"CUTLASS successor capture refused: {exc}", file=sys.stderr)
        return 1
    print(f"CUTLASS successor {args.phase} receipt: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
