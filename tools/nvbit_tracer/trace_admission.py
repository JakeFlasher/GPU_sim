"""Fail-closed admission for a completed forward NVBit trace capture.

Historical M1 payload authorities are verified by their dedicated manifest
gate and must never be dispatched through this forward-capture consumer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.nvbit_tracer.import_trace_contract import (
    TraceContractRuntimePrerequisiteMissing,
    TraceContractTrackedBindingsCorrupt,
    _import_pb_modules,
    load_allocations_sidecar,
    validate_threadblock,
)
from tools.nvbit_tracer.protobuf_wire_preflight import (
    ProtobufDecodedResourceLimitExceeded,
    ProtobufWireMalformed,
    preflight_threadblock_protobuf,
    preflight_trace_protobuf,
)


PB_GENERATED = REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"
EXPECTED_TRACE_CONTRACT_ID = "sm86_addr_carry_v1"
EXPECTED_MEMORY_PAYLOAD_CLASSES = (1, 2, 3, 6, 8)
TRACE_ADMISSION_SCOPE = "forward_capture_only"
I128_MIN = -(2**127)
I128_MAX = 2**127 - 1
I64_MAX = 2**63 - 1
U64_MAX = 2**64 - 1


def _parse_ascii_signed_decimal(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    if not digits or not all("0" <= char <= "9" for char in digits):
        raise ValueError("not an ASCII signed-decimal token")
    return int(token, 10)


class TraceAdmissionError(RuntimeError):
    """A capture exists but is not safe to publish or consume."""

    def __init__(self, typed_reason: str, detail: str) -> None:
        super().__init__(f"{typed_reason}: {detail}")
        self.typed_reason = typed_reason
        self.detail = detail


@dataclass(frozen=True)
class TraceAdmission:
    launch_count: int
    threadblock_count: int
    instruction_count: int
    stats_row_count: int
    expected_record_count: int
    admitted_record_count: int


def _raise_admission_prerequisite(
    exc: TraceContractTrackedBindingsCorrupt | TraceContractRuntimePrerequisiteMissing,
) -> NoReturn:
    if isinstance(exc, TraceContractTrackedBindingsCorrupt):
        raise TraceAdmissionError("tracked_protobuf_corrupt", str(exc)) from exc
    raise TraceAdmissionError(
        "protobuf_runtime_prerequisite_missing", str(exc)
    ) from exc


def _import_trace_pb2() -> Any:
    # The shared loader verifies the runtime, all nine tracked authorities, and
    # module provenance before the root Trace binding is imported.
    _import_pb_modules()
    if str(PB_GENERATED) not in sys.path:
        sys.path.insert(0, str(PB_GENERATED))
    try:
        import trace_pb2  # type: ignore[import-not-found]
    except Exception as exc:
        if exc.__class__.__module__.startswith("google.protobuf"):
            raise TraceContractRuntimePrerequisiteMissing(
                "protobuf runtime prerequisite is incompatible with the "
                f"tracked trace binding: {type(exc).__name__}: {exc}"
            ) from exc
        raise TraceContractTrackedBindingsCorrupt(
            "tracked protobuf authority trace_pb2.py is unusable; restore "
            "its byte-exact committed content with `git restore "
            "--source=HEAD --worktree -- "
            "workloads/simbench_bw/_pb_generated/trace_pb2.py`; do not "
            f"regenerate bindings in the canonical checkout: {exc}"
        ) from exc
    try:
        Path(trace_pb2.__file__).resolve().relative_to(PB_GENERATED.resolve())
    except (AttributeError, ValueError) as exc:
        raise TraceAdmissionError(
            "protobuf_bindings_mismatch",
            f"trace_pb2 resolved outside {PB_GENERATED}: "
            f"{getattr(trace_pb2, '__file__', None)}",
        ) from exc
    return trace_pb2


def _parse_stats(stats_path: Path) -> tuple[int, int, int]:
    try:
        with stats_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise TraceAdmissionError("stats_unreadable", str(exc)) from exc
    except (UnicodeError, csv.Error) as exc:
        raise TraceAdmissionError("stats_schema_invalid", str(exc)) from exc
    if not rows:
        raise TraceAdmissionError("stats_empty", str(stats_path))
    header = [cell.strip() for cell in rows[0]]
    try:
        expected_index = header.index("total_insts")
        admitted_index = header.index("total_reported_insts")
    except ValueError as exc:
        raise TraceAdmissionError(
            "stats_schema_invalid",
            "stats.csv lacks total_insts/total_reported_insts",
        ) from exc

    expected = admitted = row_count = 0
    for line_number, row in enumerate(rows[1:], start=2):
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
                "stats_schema_invalid", f"row {line_number} has non-integer counters"
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
                f"row {line_number}: expected={row_expected} admitted={row_admitted}",
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
    if row_count == 0 or expected == 0 or admitted == 0:
        raise TraceAdmissionError(
            "trace_payload_empty",
            f"stats rows={row_count} expected={expected} admitted={admitted}",
        )
    return row_count, expected, admitted


def require_trace_admission(trace_dir: Path) -> TraceAdmission:
    """Require a non-vacuous root, payload, contract walk, and producer stats."""

    trace_dir = Path(trace_dir).resolve()
    root_path = trace_dir / "dynamic_trace.pb"
    try:
        root_bytes = root_path.read_bytes()
    except OSError as exc:
        raise TraceAdmissionError("trace_root_unreadable", str(exc)) from exc
    if not root_bytes:
        raise TraceAdmissionError("trace_root_empty", str(root_path))

    # Rust snapshots the payload inventory before interpreting a nonempty root.
    # Match that fail-closed precedence without weakening the dedicated empty-root
    # diagnosis above.
    threadblocks_root = trace_dir / "threadblocks"
    if not threadblocks_root.is_dir():
        raise TraceAdmissionError(
            "trace_threadblocks_empty", "trace has no threadblocks/ directory"
        )
    threadblock_paths = sorted(threadblocks_root.rglob("*.pb"))
    if not threadblock_paths:
        raise TraceAdmissionError(
            "trace_threadblocks_empty", "threadblocks/ contains zero protobuf files"
        )
    payloads_by_parent: dict[Path, list[Path]] = {}
    for path in threadblock_paths:
        payloads_by_parent.setdefault(path.parent, []).append(path)

    try:
        preflight_trace_protobuf(root_bytes)
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

    allocations, load_error = load_allocations_sidecar(trace_dir)
    if load_error is not None:
        raise TraceAdmissionError(
            load_error.typed_reason or "allocations_sidecar_malformed",
            load_error.detail or "allocation sidecar validation failed",
        )

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
                    f"stream map key {stream_key} disagrees with embedded id {stream.id}",
                )
            for kernel in stream.kernels:
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

    try:
        threadblock_pb2, _instruction_pb2, _address_pb2 = _import_pb_modules()
    except (
        TraceContractTrackedBindingsCorrupt,
        TraceContractRuntimePrerequisiteMissing,
    ) as exc:
        _raise_admission_prerequisite(exc)
    try:
        Path(threadblock_pb2.__file__).resolve().relative_to(PB_GENERATED.resolve())
    except (AttributeError, ValueError) as exc:
        raise TraceAdmissionError(
            "protobuf_bindings_mismatch",
            f"threadblock_pb2 resolved outside {PB_GENERATED}: "
            f"{getattr(threadblock_pb2, '__file__', None)}",
        ) from exc

    consumed_paths: set[Path] = set()
    threadblock_count = 0
    instruction_count = 0
    for device_key, stream_key, kernel in launches:
        launch_dir = (
            threadblocks_root
            / f"device_{device_key}"
            / f"stream_{stream_key}"
            / f"kernel_{kernel.id}"
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
        for path in launch_paths:
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise TraceAdmissionError(
                    "trace_threadblock_unreadable", f"{path}: {exc}"
                ) from exc
            if not raw:
                raise TraceAdmissionError("trace_threadblock_empty", str(path))
            try:
                preflight_threadblock_protobuf(raw)
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
                    "block_id_duplicate", f"{path}: duplicate block index {block_index}"
                )
            block_indices.add(block_index)

            block_records = sum(len(warp.instructions) for warp in message.warps.values())
            if block_records <= 0:
                raise TraceAdmissionError(
                    "trace_threadblock_empty", f"{path} has zero instruction records"
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
                    contract.detail or "threadblock contract validation failed",
                )
            consumed_paths.add(path)
            if threadblock_count == U64_MAX:
                raise TraceAdmissionError(
                    "trace_counter_overflow", "threadblock count exceeds uint64"
                )
            threadblock_count += 1
            if block_records > U64_MAX - instruction_count:
                raise TraceAdmissionError(
                    "trace_counter_overflow", "instruction count exceeds uint64"
                )
            instruction_count += block_records

    orphan = next((path for path in threadblock_paths if path not in consumed_paths), None)
    if orphan is not None:
        raise TraceAdmissionError(
            "trace_payload_orphan", f"{orphan} does not belong to a root launch"
        )

    stats_row_count, expected, admitted = _parse_stats(trace_dir / "stats.csv")
    return TraceAdmission(
        launch_count=len(launches),
        threadblock_count=threadblock_count,
        instruction_count=instruction_count,
        stats_row_count=stats_row_count,
        expected_record_count=expected,
        admitted_record_count=admitted,
    )


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
