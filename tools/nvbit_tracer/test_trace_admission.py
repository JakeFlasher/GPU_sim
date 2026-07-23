from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.nvbit_tracer.trace_admission import (
    PB_GENERATED,
    TRACE_ADMISSION_SCOPE,
    TraceAdmissionError,
    _raise_admission_prerequisite,
    require_trace_admission,
)
from tools.nvbit_tracer.import_trace_contract import (
    TraceContractRuntimePrerequisiteMissing,
    TraceContractTrackedBindingsCorrupt,
)


def _pb_modules():
    required = (
        "address_pb2.py", "cuda_stream_pb2.py", "dim3d_pb2.py",
        "gpu_device_pb2.py", "instruction_pb2.py", "kernel_pb2.py",
        "threadblock_pb2.py", "trace_pb2.py", "warp_pb2.py",
    )
    missing = [
        name
        for name in required
        if not (PB_GENERATED / name).is_file()
        or (PB_GENERATED / name).is_symlink()
    ]
    assert not missing, (
        "tracked protobuf authority is missing; restore byte-exact committed "
        "content with `git restore --source=HEAD --worktree -- "
        + " ".join(
            f"workloads/simbench_bw/_pb_generated/{name}" for name in missing
        )
        + "`"
    )
    if str(PB_GENERATED) not in sys.path:
        sys.path.insert(0, str(PB_GENERATED))
    try:
        import threadblock_pb2  # type: ignore[import-not-found]
        import trace_pb2  # type: ignore[import-not-found]
    except ImportError as exc:
        pytest.fail(f"tracked dynamic-trace authorities are unusable: {exc}")
    return trace_pb2, threadblock_pb2


def _write_trace(root: Path, *, launch: bool = True, instruction: bool = True,
                 expected: int = 1, admitted: int = 1) -> Path:
    trace_pb2, threadblock_pb2 = _pb_modules()
    root.mkdir()

    trace = trace_pb2.Trace()
    trace.trace_contract_id = "sm86_addr_carry_v1"
    trace.memory_payload_classes.extend([1, 2, 3, 6, 8])
    if launch:
        device = trace.gpu_device[0]
        device.id = 0
        stream = device.streams[0]
        stream.id = 0
        kernel = stream.kernels.add()
        kernel.id = 0
        kernel.name = "fixture_kernel"
        kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
        kernel.block_dim.x = 32
        kernel.block_dim.y = kernel.block_dim.z = 1
    (root / "dynamic_trace.pb").write_bytes(trace.SerializeToString())

    tb = threadblock_pb2.threadblock()
    tb.block_id.x = tb.block_id.y = tb.block_id.z = 0
    if instruction:
        warp = tb.warps[0]
        warp.id = 0
        inst = warp.instructions.add()
        inst.pc = inst.vpc = 0x1000
        inst.active_mask = inst.predicate_mask = 1
        inst.function_unique_id = 1
        inst.data_width = 4
        inst.warpid_tb = 0
    tb_path = root / "threadblocks" / "device_0" / "stream_0" / "kernel_0" / "0.pb"
    tb_path.parent.mkdir(parents=True)
    tb_path.write_bytes(tb.SerializeToString())
    (root / "stats.csv").write_text(
        "kernel_id,total_insts,total_reported_insts\n"
        f"0,{expected},{admitted}\n",
        encoding="utf-8",
    )
    return root


def test_require_trace_admission_accepts_nonvacuous_capture(tmp_path: Path) -> None:
    result = require_trace_admission(_write_trace(tmp_path / "trace"))
    assert result.launch_count == 1
    assert result.threadblock_count == 1
    assert result.instruction_count == 1
    assert result.expected_record_count == 1
    assert result.admitted_record_count == 1


def test_require_trace_admission_rejects_zero_launch_root(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace", launch=False)
    with pytest.raises(TraceAdmissionError, match="trace_launches_empty"):
        require_trace_admission(trace_dir)


def test_require_trace_admission_rejects_empty_threadblock(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace", instruction=False)
    with pytest.raises(TraceAdmissionError, match="trace_threadblock_empty"):
        require_trace_admission(trace_dir)


def test_require_trace_admission_rejects_zero_admission_stats(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace", expected=0, admitted=0)
    with pytest.raises(TraceAdmissionError, match="trace_payload_empty"):
        require_trace_admission(trace_dir)


def test_trace_admission_is_forward_capture_only() -> None:
    assert TRACE_ADMISSION_SCOPE == "forward_capture_only"


@pytest.mark.parametrize(
    ("exc", "reason"),
    (
        (TraceContractTrackedBindingsCorrupt("bad binding"), "tracked_protobuf_corrupt"),
        (
            TraceContractRuntimePrerequisiteMissing("missing runtime"),
            "protobuf_runtime_prerequisite_missing",
        ),
    ),
)
def test_contract_prerequisites_keep_distinct_typed_reasons(
    exc: TraceContractTrackedBindingsCorrupt
    | TraceContractRuntimePrerequisiteMissing,
    reason: str,
) -> None:
    with pytest.raises(TraceAdmissionError) as caught:
        _raise_admission_prerequisite(exc)
    assert caught.value.typed_reason == reason
