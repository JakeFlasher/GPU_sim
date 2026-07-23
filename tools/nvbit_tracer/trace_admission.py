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
    validate_trace_dir,
)


PB_GENERATED = REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"
EXPECTED_TRACE_CONTRACT_ID = "sm86_addr_carry_v1"
EXPECTED_MEMORY_PAYLOAD_CLASSES = frozenset({1, 2, 3, 6, 8})
TRACE_ADMISSION_SCOPE = "forward_capture_only"


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


def _parse_stats(stats_path: Path) -> tuple[int, int]:
    try:
        with stats_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise TraceAdmissionError("stats_unreadable", str(exc)) from exc
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
            row_expected = int(cells[expected_index])
            row_admitted = int(cells[admitted_index])
        except ValueError as exc:
            raise TraceAdmissionError(
                "stats_schema_invalid", f"row {line_number} has non-integer counters"
            ) from exc
        if row_expected < 0 or row_admitted < 0 or row_admitted > row_expected:
            raise TraceAdmissionError(
                "stats_counters_invalid",
                f"row {line_number}: expected={row_expected} admitted={row_admitted}",
            )
        expected += row_expected
        admitted += row_admitted
        row_count += 1
    if row_count == 0 or expected <= 0 or admitted <= 0:
        raise TraceAdmissionError(
            "trace_payload_empty",
            f"stats rows={row_count} expected={expected} admitted={admitted}",
        )
    return expected, admitted


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

    try:
        trace_pb2 = _import_trace_pb2()
    except (TraceContractTrackedBindingsCorrupt,
            TraceContractRuntimePrerequisiteMissing) as exc:
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
    missing_classes = sorted(
        EXPECTED_MEMORY_PAYLOAD_CLASSES - set(root.memory_payload_classes)
    )
    if missing_classes:
        raise TraceAdmissionError(
            "trace_capabilities_incomplete", f"missing classes {missing_classes}"
        )

    launches = []
    for device in root.gpu_device.values():
        for stream in device.streams.values():
            launches.extend(stream.kernels)
    if not launches:
        raise TraceAdmissionError("trace_launches_empty", "root Trace has zero launches")
    for index, kernel in enumerate(launches):
        geometry = (
            int(kernel.grid_dim.x), int(kernel.grid_dim.y), int(kernel.grid_dim.z),
            int(kernel.block_dim.x), int(kernel.block_dim.y), int(kernel.block_dim.z),
        )
        if any(value <= 0 for value in geometry):
            raise TraceAdmissionError(
                "trace_launch_geometry_invalid",
                f"launch {index} has grid/block geometry {geometry}",
            )

    try:
        contract = validate_trace_dir(trace_dir)
    except (TraceContractTrackedBindingsCorrupt,
            TraceContractRuntimePrerequisiteMissing) as exc:
        _raise_admission_prerequisite(exc)
    if not contract.ok:
        raise TraceAdmissionError(
            contract.typed_reason or "trace_contract_invalid",
            contract.detail or "threadblock contract validation failed",
        )
    threadblock_paths = sorted((trace_dir / "threadblocks").rglob("*.pb"))
    if not threadblock_paths:
        raise TraceAdmissionError(
            "trace_threadblocks_empty", "threadblocks/ contains zero protobuf files"
        )
    try:
        threadblock_pb2, _instruction_pb2, _address_pb2 = _import_pb_modules()
    except (TraceContractTrackedBindingsCorrupt,
            TraceContractRuntimePrerequisiteMissing) as exc:
        _raise_admission_prerequisite(exc)
    try:
        Path(threadblock_pb2.__file__).resolve().relative_to(PB_GENERATED.resolve())
    except (AttributeError, ValueError) as exc:
        raise TraceAdmissionError(
            "protobuf_bindings_mismatch",
            f"threadblock_pb2 resolved outside {PB_GENERATED}: "
            f"{getattr(threadblock_pb2, '__file__', None)}",
        ) from exc
    instruction_count = 0
    for path in threadblock_paths:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise TraceAdmissionError(
                "trace_threadblock_unreadable", f"{path}: {exc}"
            ) from exc
        if not raw:
            raise TraceAdmissionError("trace_threadblock_empty", str(path))
        message = threadblock_pb2.threadblock()
        try:
            message.ParseFromString(raw)
        except Exception as exc:
            raise TraceAdmissionError(
                "trace_threadblock_unparseable", f"{path}: {exc}"
            ) from exc
        block_records = sum(len(warp.instructions) for warp in message.warps.values())
        if block_records <= 0:
            raise TraceAdmissionError(
                "trace_threadblock_empty", f"{path} has zero instruction records"
            )
        instruction_count += block_records

    expected, admitted = _parse_stats(trace_dir / "stats.csv")
    return TraceAdmission(
        launch_count=len(launches),
        threadblock_count=len(threadblock_paths),
        instruction_count=instruction_count,
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
