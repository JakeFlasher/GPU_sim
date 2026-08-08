from __future__ import annotations

import errno
import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import tools.local_labels.immutable_artifact as immutable_artifact
import tools.nvbit_tracer.import_trace_contract as import_trace_contract
import tools.nvbit_tracer.trace_admission as trace_admission
from tools.nvbit_tracer.trace_admission import (
    DEFAULT_TRACE_ADMISSION_LIMITS,
    HISTORICAL_TRACE_ADMISSION_SCOPE,
    TRACE_ADMISSION_SCOPE,
    TraceAdmissionError,
    _raise_admission_prerequisite,
    monitor_admitted_trace_sources,
    open_admitted_trace_sources,
    require_historical_trace_admission,
    require_trace_admission,
)
from tools.nvbit_tracer.import_trace_contract import (
    TraceContractRuntimePrerequisiteMissing,
    TraceContractTrackedBindingsCorrupt,
)
from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings


def _pb_modules():
    bindings = load_dynamic_trace_bindings()
    return bindings.trace_pb2, bindings.threadblock_pb2


def _write_trace(
    root: Path,
    *,
    launch: bool = True,
    instruction: bool = True,
    expected: int = 1,
    admitted: int = 1,
    trace_contract_id: str = trace_admission.EXPECTED_TRACE_CONTRACT_ID,
    include_split_fields: bool = True,
) -> Path:
    trace_pb2, threadblock_pb2 = _pb_modules()
    root.mkdir()

    trace = trace_pb2.Trace()
    trace.trace_contract_id = trace_contract_id
    trace.memory_payload_classes.extend([1, 2, 3, 6, 8])
    if launch:
        device = trace.gpu_device[0]
        device.id = 0
        stream = device.streams[0]
        stream.id = 0
        kernel = stream.kernels.add()
        kernel.id = 0
        kernel.name = "fixture_kernel"
        kernel.function_unique_id = 1
        kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
        kernel.block_dim.x = 32
        kernel.block_dim.y = kernel.block_dim.z = 1
        if include_split_fields:
            kernel.shared_memory_static_bytes = 0
            kernel.shared_memory_dynamic_bytes = 0
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


def _read_root(trace_dir: Path):
    trace_pb2, _threadblock_pb2 = _pb_modules()
    root = trace_pb2.Trace()
    root.ParseFromString((trace_dir / "dynamic_trace.pb").read_bytes())
    return root


def _write_root(trace_dir: Path, root) -> None:
    (trace_dir / "dynamic_trace.pb").write_bytes(root.SerializeToString())


def _threadblock_path(trace_dir: Path, kernel_id: int = 0) -> Path:
    return (
        trace_dir
        / "threadblocks"
        / "device_0"
        / "stream_0"
        / f"kernel_{kernel_id}"
        / "0.pb"
    )


def _open_descriptor_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_require_trace_admission_accepts_nonvacuous_capture(tmp_path: Path) -> None:
    result = require_trace_admission(_write_trace(tmp_path / "trace"))
    assert result.launch_count == 1
    assert result.threadblock_count == 1
    assert result.instruction_count == 1
    assert result.stats_row_count == 1
    assert result.expected_record_count == 1
    assert result.admitted_record_count == 1
    assert result.admission_scope == TRACE_ADMISSION_SCOPE
    assert result.trace_contract_id == trace_admission.EXPECTED_TRACE_CONTRACT_ID
    assert result.memory_payload_classes == (1, 2, 3, 6, 8)
    root_bytes = (tmp_path / "trace" / "dynamic_trace.pb").read_bytes()
    stats_bytes = (tmp_path / "trace" / "stats.csv").read_bytes()
    assert result.dynamic_trace_size_bytes == len(root_bytes)
    assert result.dynamic_trace_sha256 == hashlib.sha256(root_bytes).hexdigest()
    assert result.stats_size_bytes == len(stats_bytes)
    assert result.stats_sha256 == hashlib.sha256(stats_bytes).hexdigest()
    assert result.allocations_sidecar_size_bytes is None
    assert result.allocations_sidecar_sha256 is None
    assert result.threadblock_payload_size_bytes > 0
    assert len(result.threadblock_payload_manifest_sha256) == 64
    assert len(result.launches) == 1
    assert result.launches[0].kernel_name == "fixture_kernel"
    assert result.launches[0].shared_mem_static_bytes == 0
    assert result.launches[0].shared_mem_dynamic_bytes == 0
    assert result.launches[0].inst_count == 1
    assert len(result.trace_filesystem_generation_sha256) == 64
    assert len(result.trace_admission_sha256) == 64
    changed = replace(result, admitted_record_count=2)
    assert changed.trace_admission_sha256 != result.trace_admission_sha256


def test_legacy_v1_requires_explicit_historical_read_only_admission(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(
        tmp_path / "legacy",
        trace_contract_id=trace_admission.LEGACY_TRACE_CONTRACT_ID,
        include_split_fields=False,
    )

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "legacy_trace_historical_only"

    historical = require_historical_trace_admission(trace_dir)
    assert historical.admission_scope == HISTORICAL_TRACE_ADMISSION_SCOPE
    assert historical.trace_contract_id == trace_admission.LEGACY_TRACE_CONTRACT_ID
    with open_admitted_trace_sources(
        trace_dir,
        expected_admission=historical,
    ) as sources:
        sources.verify_unchanged()


def test_legacy_v1_retains_historical_set_capability_semantics(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(
        tmp_path / "legacy-reordered",
        trace_contract_id=trace_admission.LEGACY_TRACE_CONTRACT_ID,
        include_split_fields=False,
    )
    root = _read_root(trace_dir)
    capabilities = list(root.memory_payload_classes)
    del root.memory_payload_classes[:]
    root.memory_payload_classes.extend(reversed(capabilities))
    _write_root(trace_dir, root)

    historical = require_historical_trace_admission(trace_dir)

    assert historical.admission_scope == HISTORICAL_TRACE_ADMISSION_SCOPE
    assert historical.memory_payload_classes == tuple(reversed(capabilities))


def test_trace_admission_does_not_fabricate_shared_memory_split(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(
        tmp_path / "trace",
        trace_contract_id=trace_admission.LEGACY_TRACE_CONTRACT_ID,
        include_split_fields=False,
    )
    root = _read_root(trace_dir)
    root.gpu_device[0].streams[0].kernels[0].size_shared_memory = 4096
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_historical_trace_admission(trace_dir)

    assert caught.value.typed_reason == "shared_memory_split_unavailable"
    assert "combined static+dynamic" in caught.value.detail


def test_trace_admission_accepts_authenticated_shared_memory_split(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    root.trace_contract_id = trace_admission.EXPECTED_TRACE_CONTRACT_ID
    kernel = root.gpu_device[0].streams[0].kernels[0]
    kernel.size_shared_memory = 6144
    kernel.shared_memory_static_bytes = 2048
    kernel.shared_memory_dynamic_bytes = 4096
    _write_root(trace_dir, root)

    admitted = require_trace_admission(trace_dir)

    assert admitted.launches[0].shared_mem_static_bytes == 2048
    assert admitted.launches[0].shared_mem_dynamic_bytes == 4096


def test_trace_admission_requires_explicit_zero_split_components(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    root.trace_contract_id = trace_admission.EXPECTED_TRACE_CONTRACT_ID
    kernel = root.gpu_device[0].streams[0].kernels[0]
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0
    _write_root(trace_dir, root)

    admitted = require_trace_admission(trace_dir)

    assert admitted.launches[0].shared_mem_static_bytes == 0
    assert admitted.launches[0].shared_mem_dynamic_bytes == 0


@pytest.mark.parametrize(
    "present_field",
    ["shared_memory_static_bytes", "shared_memory_dynamic_bytes", None],
)
def test_trace_admission_rejects_incomplete_advertised_split(
    tmp_path: Path,
    present_field: str | None,
) -> None:
    trace_dir = _write_trace(
        tmp_path / "trace", include_split_fields=False
    )
    root = _read_root(trace_dir)
    root.trace_contract_id = trace_admission.EXPECTED_TRACE_CONTRACT_ID
    if present_field is not None:
        setattr(root.gpu_device[0].streams[0].kernels[0], present_field, 0)
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)

    assert caught.value.typed_reason == "shared_memory_split_missing"


def test_trace_admission_rejects_split_sum_mismatch(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    root.trace_contract_id = trace_admission.EXPECTED_TRACE_CONTRACT_ID
    kernel = root.gpu_device[0].streams[0].kernels[0]
    kernel.size_shared_memory = 4096
    kernel.shared_memory_static_bytes = 1024
    kernel.shared_memory_dynamic_bytes = 2048
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)

    assert caught.value.typed_reason == "shared_memory_split_mismatch"


def test_trace_admission_rejects_unadvertised_split_fields(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(
        tmp_path / "trace",
        trace_contract_id=trace_admission.LEGACY_TRACE_CONTRACT_ID,
        include_split_fields=False,
    )
    root = _read_root(trace_dir)
    kernel = root.gpu_device[0].streams[0].kernels[0]
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_historical_trace_admission(trace_dir)

    assert caught.value.typed_reason == "shared_memory_split_unadvertised"


def test_trace_admission_rejects_negative_combined_shared_memory(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    root.gpu_device[0].streams[0].kernels[0].size_shared_memory = -1
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)

    assert caught.value.typed_reason == "trace_launch_shared_memory_invalid"


def test_trace_admission_content_seal_is_relocation_portable(
    tmp_path: Path,
) -> None:
    first = require_trace_admission(_write_trace(tmp_path / "trace-a"))
    second = require_trace_admission(_write_trace(tmp_path / "trace-b"))

    assert first.dynamic_trace_path != second.dynamic_trace_path
    assert first.trace_admission_sha256 == second.trace_admission_sha256
    assert (
        first.trace_filesystem_generation_sha256
        != second.trace_filesystem_generation_sha256
    )


def test_trace_admission_generation_binds_lexical_ancestors(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "capture-parent"
    parent.mkdir()
    trace_dir = _write_trace(parent / "trace")
    first = require_trace_admission(trace_dir)

    moved_parent = tmp_path / "capture-parent-original"
    parent.rename(moved_parent)
    parent.mkdir()
    (moved_parent / "trace").rename(trace_dir)
    moved_parent.rmdir()

    second = require_trace_admission(trace_dir)

    assert second.trace_admission_sha256 == first.trace_admission_sha256
    assert (
        second.trace_filesystem_generation_sha256
        != first.trace_filesystem_generation_sha256
    )
    assert second != first


def test_trace_admission_distinguishes_same_byte_file_generation(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    first = require_trace_admission(trace_dir)
    payload_path = _threadblock_path(trace_dir)
    replacement = payload_path.with_suffix(".replacement")
    replacement.write_bytes(payload_path.read_bytes())
    os.replace(replacement, payload_path)

    second = require_trace_admission(trace_dir)

    assert second.trace_admission_sha256 == first.trace_admission_sha256
    assert (
        second.trace_filesystem_generation_sha256
        != first.trace_filesystem_generation_sha256
    )
    assert second != first


def test_trace_admission_generation_allows_unrelated_sibling_output(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    extra_info = trace_dir / "extra_info"
    extra_info.mkdir()
    first = require_trace_admission(trace_dir)

    (extra_info / "feature.json").write_text("{}\n", encoding="utf-8")
    second = require_trace_admission(trace_dir)

    assert second.trace_admission_sha256 == first.trace_admission_sha256
    assert (
        second.trace_filesystem_generation_sha256
        == first.trace_filesystem_generation_sha256
    )
    assert second == first


def test_trace_admission_consumes_no_trace_bytes_through_pathlib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("admission reopened trace bytes by pathname")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    result = require_trace_admission(trace_dir)
    assert result.launch_count == 1


def test_trace_admission_rejects_non_utf8_inventory_path(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    raw_path = os.fsencode(trace_dir) + b"/non-utf8-\xff"
    descriptor = os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_filesystem_invalid"
    assert "not UTF-8" in caught.value.detail


def test_trace_admission_types_post_inventory_generation_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    stats_path = trace_dir / "stats.csv"
    original_read = trace_admission.PinnedDirectoryReader.read_snapshot

    def replace_then_read(
        reader,
        relative_path: str,
        *,
        max_bytes=None,
        expected_metadata=None,
    ):
        if relative_path == "stats.csv":
            replacement = stats_path.with_suffix(".replacement")
            replacement.write_bytes(stats_path.read_bytes())
            os.replace(replacement, stats_path)
        return original_read(
            reader,
            relative_path,
            max_bytes=max_bytes,
            expected_metadata=expected_metadata,
        )

    monkeypatch.setattr(
        trace_admission.PinnedDirectoryReader,
        "read_snapshot",
        replace_then_read,
    )
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_snapshot_changed"
    assert "generation changed after inventory" in caught.value.detail


@pytest.mark.parametrize(
    "artifact",
    ("dynamic_trace.pb", "stats.csv", "threadblock", "allocations.json"),
)
def test_trace_admission_rejects_symlinked_input_leaf(
    tmp_path: Path,
    artifact: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    if artifact == "threadblock":
        victim = _threadblock_path(trace_dir)
    elif artifact == "allocations.json":
        victim = trace_dir / "extra_info" / "allocations.json"
        victim.parent.mkdir()
        victim.write_bytes(b"[]\n")
    else:
        victim = trace_dir / artifact
    target = tmp_path / f"target-{artifact.replace('/', '-')}"
    victim.rename(target)
    victim.symlink_to(target)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_filesystem_invalid"
    assert "symlink" in caught.value.detail


def test_trace_admission_rejects_symlinked_root_and_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    trace_dir = _write_trace(real_parent / "trace")
    linked_trace = tmp_path / "linked-trace"
    linked_trace.symlink_to(trace_dir, target_is_directory=True)
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(linked_trace)
    assert caught.value.typed_reason == "trace_root_unreadable"

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(linked_parent / "trace")
    assert caught.value.typed_reason == "trace_root_unreadable"


@pytest.mark.parametrize(
    "artifact",
    ("dynamic_trace.pb", "stats.csv", "threadblock", "allocations.json"),
)
def test_trace_admission_rejects_hardlinked_input(
    tmp_path: Path,
    artifact: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    if artifact == "threadblock":
        victim = _threadblock_path(trace_dir)
    elif artifact == "allocations.json":
        victim = trace_dir / "extra_info" / "allocations.json"
        victim.parent.mkdir()
        victim.write_bytes(b"[]\n")
    else:
        victim = trace_dir / artifact
    os.link(victim, tmp_path / f"hardlink-{artifact.replace('/', '-')}")

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_filesystem_invalid"
    assert "single-link regular file" in caught.value.detail


@pytest.mark.parametrize("artifact", ("dynamic_trace.pb", "stats.csv", "threadblock"))
def test_trace_admission_rejects_fifo_without_opening_it(
    tmp_path: Path,
    artifact: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    victim = (
        _threadblock_path(trace_dir)
        if artifact == "threadblock"
        else trace_dir / artifact
    )
    victim.unlink()
    os.mkfifo(victim)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_filesystem_invalid"
    assert "non-regular" in caught.value.detail


def test_trace_admission_rejects_nonpayload_threadblock_members(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    (_threadblock_path(trace_dir).parent / "producer.log").write_text(
        "not a protobuf payload\n",
        encoding="utf-8",
    )

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)

    assert caught.value.typed_reason == "trace_payload_nonprotobuf"
    assert "producer.log" in caught.value.detail


def test_trace_admission_bounds_every_input_class(tmp_path: Path) -> None:
    cases: list[tuple[str, Path, object]] = []

    root_dir = _write_trace(tmp_path / "root")
    root_size = (root_dir / "dynamic_trace.pb").stat().st_size
    cases.append(
        (
            "trace_root_size_limit_exceeded",
            root_dir,
            replace(DEFAULT_TRACE_ADMISSION_LIMITS, trace_root_bytes=root_size - 1),
        )
    )

    stats_dir = _write_trace(tmp_path / "stats")
    cases.append(
        (
            "stats_size_limit_exceeded",
            stats_dir,
            replace(DEFAULT_TRACE_ADMISSION_LIMITS, stats_bytes=1),
        )
    )

    sidecar_dir = _write_trace(tmp_path / "sidecar")
    sidecar = sidecar_dir / "extra_info" / "allocations.json"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"[]\n")
    cases.append(
        (
            "allocations_sidecar_size_limit_exceeded",
            sidecar_dir,
            replace(DEFAULT_TRACE_ADMISSION_LIMITS, allocations_sidecar_bytes=1),
        )
    )

    payload_dir = _write_trace(tmp_path / "payload")
    cases.append(
        (
            "trace_threadblock_size_limit_exceeded",
            payload_dir,
            replace(DEFAULT_TRACE_ADMISSION_LIMITS, threadblock_bytes=1),
        )
    )

    aggregate_dir = _write_trace(tmp_path / "aggregate")
    cases.append(
        (
            "trace_payload_size_limit_exceeded",
            aggregate_dir,
            replace(DEFAULT_TRACE_ADMISSION_LIMITS, total_threadblock_bytes=1),
        )
    )

    inventory_dir = _write_trace(tmp_path / "inventory")
    cases.append(
        (
            "trace_inventory_limit_exceeded",
            inventory_dir,
            replace(
                DEFAULT_TRACE_ADMISSION_LIMITS,
                threadblock_files=2,
                tree_files=2,
            ),
        )
    )

    path_budget_dir = _write_trace(tmp_path / "path-budget")
    cases.append(
        (
            "trace_inventory_limit_exceeded",
            path_budget_dir,
            replace(
                DEFAULT_TRACE_ADMISSION_LIMITS,
                tree_relative_path_bytes=1,
            ),
        )
    )

    allocation_count_dir = _write_trace(tmp_path / "allocation-count")
    allocation_count_path = (
        allocation_count_dir / "extra_info" / "allocations.json"
    )
    allocation_count_path.parent.mkdir()
    allocation_count_path.write_bytes(b"[{}, {}]\n")
    cases.append(
        (
            "allocations_sidecar_resource_limit_exceeded",
            allocation_count_dir,
            replace(DEFAULT_TRACE_ADMISSION_LIMITS, allocation_entries=1),
        )
    )

    allocation_nesting_dir = _write_trace(tmp_path / "allocation-nesting")
    allocation_nesting_path = (
        allocation_nesting_dir / "extra_info" / "allocations.json"
    )
    allocation_nesting_path.parent.mkdir()
    allocation_nesting_path.write_bytes(b"[{}]\n")
    cases.append(
        (
            "allocations_sidecar_resource_limit_exceeded",
            allocation_nesting_dir,
            replace(DEFAULT_TRACE_ADMISSION_LIMITS, allocation_json_nesting=1),
        )
    )

    for reason, trace_dir, limits in cases:
        with pytest.raises(TraceAdmissionError) as caught:
            require_trace_admission(trace_dir, limits=limits)
        assert caught.value.typed_reason == reason


@pytest.mark.parametrize(
    "payload",
    (
        b'[{"a":0,"b":0,"c":0,"d":0,"e":0,"f":0,"g":[]}]',
        b'[{"allocation_id":1,"extra":[0,1,2]}]',
    ),
)
def test_allocation_parser_rejects_decoded_container_amplification(
    tmp_path: Path,
    payload: bytes,
) -> None:
    table, error = import_trace_contract.parse_allocations_sidecar_bytes(
        payload,
        tmp_path / "allocations.json",
    )
    assert table is None
    assert error is not None
    assert error.typed_reason == "allocations_sidecar_resource_limit_exceeded"


@pytest.mark.parametrize(
    "payload",
    (b"[{}{}]", b'[{"a":1}{"b":2}]'),
)
def test_allocation_parser_types_missing_object_comma_as_malformed(
    tmp_path: Path,
    payload: bytes,
) -> None:
    table, error = import_trace_contract.parse_allocations_sidecar_bytes(
        payload,
        tmp_path / "allocations.json",
    )
    assert table is None
    assert error is not None
    assert error.typed_reason == "allocations_sidecar_malformed"
    assert "separated by commas" in (error.detail or "")


def test_allocation_parser_rejects_excessive_flat_object_members(
    tmp_path: Path,
) -> None:
    payload = json.dumps(
        [{f"unexpected_{index}": index for index in range(65)}]
    ).encode("utf-8")
    table, error = import_trace_contract.parse_allocations_sidecar_bytes(
        payload,
        tmp_path / "allocations.json",
    )
    assert table is None
    assert error is not None
    assert error.typed_reason == "allocations_sidecar_resource_limit_exceeded"


def test_allocation_parser_propagates_host_memory_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exhaust_memory(_raw: bytes, _path: Path):
        raise MemoryError("planted")

    monkeypatch.setattr(
        import_trace_contract,
        "_parse_allocations_sidecar_bytes_after_preflight",
        exhaust_memory,
    )
    with pytest.raises(MemoryError, match="planted"):
        import_trace_contract.parse_allocations_sidecar_bytes(
            b"[]",
            tmp_path / "allocations.json",
        )


def test_allocation_parser_memory_exhaustion_is_trace_resource_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    sidecar = trace_dir / "extra_info" / "allocations.json"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"[]\n")

    def exhaust_memory(_raw: bytes, _path: Path):
        raise MemoryError("planted")

    monkeypatch.setattr(
        import_trace_contract,
        "_parse_allocations_sidecar_bytes_after_preflight",
        exhaust_memory,
    )
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)

    assert caught.value.typed_reason == "trace_snapshot_resource_unavailable"
    assert isinstance(caught.value.__cause__, MemoryError)


@pytest.mark.parametrize("closure", (False, True))
def test_trace_snapshot_resource_failure_has_stable_typed_reason(
    closure: bool,
) -> None:
    class ResourceStarvedReader:
        def snapshot_tree(self, **_kwargs):
            raise trace_admission.ImmutableArtifactResourceError("planted resource gap")

    with pytest.raises(TraceAdmissionError) as caught:
        trace_admission._snapshot_tree(
            ResourceStarvedReader(),
            DEFAULT_TRACE_ADMISSION_LIMITS,
            closure=closure,
        )
    assert caught.value.typed_reason == "trace_snapshot_resource_unavailable"


@pytest.mark.parametrize(
    "failure",
    (
        MemoryError("planted memory exhaustion"),
        OSError(errno.EMFILE, "planted process descriptor exhaustion"),
        OSError(errno.ENFILE, "planted system descriptor exhaustion"),
    ),
)
def test_trace_root_open_resource_failures_are_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")

    def fail_reader(_path: Path):
        raise failure

    monkeypatch.setattr(trace_admission, "PinnedDirectoryReader", fail_reader)
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_snapshot_resource_unavailable"
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("resource_errno", (errno.EMFILE, errno.ENFILE))
@pytest.mark.parametrize("operation", ("require", "open"))
def test_wrapped_descriptor_exhaustion_preserves_resource_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_errno: int,
    operation: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    original_open = immutable_artifact.os.open

    def exhaust_anchored_root(path, flags, *args, **kwargs):
        if path == "dynamic_trace.pb" and kwargs.get("dir_fd") is not None:
            raise OSError(resource_errno, "planted descriptor exhaustion")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        immutable_artifact.os, "open", exhaust_anchored_root,
    )
    with pytest.raises(TraceAdmissionError) as caught:
        if operation == "require":
            require_trace_admission(trace_dir)
        else:
            open_admitted_trace_sources(
                trace_dir, expected_admission=admission,
            )

    assert caught.value.typed_reason == "trace_snapshot_resource_unavailable"
    assert trace_admission._is_trace_resource_error(caught.value.__cause__)


@pytest.mark.parametrize("resource_errno", (errno.EMFILE, errno.ENFILE))
@pytest.mark.parametrize("operation", ("require", "open"))
def test_wrapped_binding_loader_exhaustion_preserves_resource_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_errno: int,
    operation: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    resource_failure = OSError(
        resource_errno, "planted binding descriptor exhaustion",
    )

    def fail_binding_load():
        raise trace_admission.ProtoBindingIntegrityError(
            "binding open failed"
        ) from resource_failure

    monkeypatch.setattr(
        trace_admission, "load_dynamic_trace_bindings", fail_binding_load,
    )
    with pytest.raises(TraceAdmissionError) as caught:
        if operation == "require":
            require_trace_admission(trace_dir)
        else:
            open_admitted_trace_sources(
                trace_dir, expected_admission=admission,
            )

    assert caught.value.typed_reason == "trace_snapshot_resource_unavailable"
    assert trace_admission._is_trace_resource_error(caught.value.__cause__)


@pytest.mark.parametrize("mutation", ("root_replace", "stats_write", "payload_write", "inventory"))
def test_trace_admission_rejects_post_parse_snapshot_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root_path = trace_dir / "dynamic_trace.pb"
    stats_path = trace_dir / "stats.csv"
    payload_path = _threadblock_path(trace_dir)
    original_parse = trace_admission._parse_stats_bytes

    def parse_then_mutate(
        raw: bytes, path: Path, **kwargs: int,
    ) -> tuple[int, int, int]:
        result = original_parse(raw, path, **kwargs)
        if mutation == "root_replace":
            replacement = root_path.with_suffix(".replacement")
            replacement.write_bytes(root_path.read_bytes())
            os.replace(replacement, root_path)
        elif mutation == "stats_write":
            current = stats_path.read_bytes()
            changed = current.replace(b"0,1,1\n", b"0,2,1\n")
            assert len(changed) == len(current) and changed != current
            stats_path.write_bytes(changed)
        elif mutation == "payload_write":
            current = payload_path.read_bytes()
            payload_path.write_bytes(bytes((current[0] ^ 1,)) + current[1:])
        else:
            payload_path.with_name("late.pb").write_bytes(payload_path.read_bytes())
        return result

    monkeypatch.setattr(trace_admission, "_parse_stats_bytes", parse_then_mutate)
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_snapshot_changed"


def test_legacy_admitted_source_monitor_handles_absent_optional_parent(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    baseline = _open_descriptor_count()

    with monitor_admitted_trace_sources(trace_dir, admission) as monitor:
        monitor.verify_unchanged()

    assert _open_descriptor_count() == baseline


def test_legacy_admitted_source_monitor_authenticates_supplied_admission(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    forged = replace(admission, dynamic_trace_sha256="0" * 64)
    baseline = _open_descriptor_count()

    with pytest.raises(TraceAdmissionError) as caught:
        monitor_admitted_trace_sources(trace_dir, forged)

    assert caught.value.typed_reason == "trace_snapshot_changed"
    assert _open_descriptor_count() == baseline


def test_legacy_admitted_source_monitor_guards_optional_parent_creation(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    baseline = _open_descriptor_count()
    monitor = monitor_admitted_trace_sources(trace_dir, admission)
    (trace_dir / "extra_info").mkdir()

    try:
        with pytest.raises(immutable_artifact.ImmutableArtifactError):
            monitor.verify_unchanged()
    finally:
        monitor.close()

    assert _open_descriptor_count() == baseline


@pytest.mark.parametrize("parent_preexists", (False, True))
def test_admitted_source_capability_guards_optional_sidecar_absence(
    tmp_path: Path,
    parent_preexists: bool,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    extra_info = trace_dir / "extra_info"
    if parent_preexists:
        extra_info.mkdir()
    admission = require_trace_admission(trace_dir)
    baseline = _open_descriptor_count()
    sources = open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    )
    if not parent_preexists:
        extra_info.mkdir()
    (extra_info / "allocations.json").write_bytes(b"[]\n")

    with pytest.raises(immutable_artifact.ImmutableArtifactError):
        _ = sources.allocations

    assert sources._closed is True
    assert _open_descriptor_count() == baseline


def test_legacy_source_monitor_open_failure_uses_trace_taxonomy(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    trace_dir.rename(tmp_path / "moved-trace")

    with pytest.raises(TraceAdmissionError) as caught:
        monitor_admitted_trace_sources(trace_dir, admission)

    assert caught.value.typed_reason == "trace_snapshot_changed"


def test_admitted_source_monitor_rejects_restored_payload_bytes(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    (trace_dir / "extra_info").mkdir()
    admission = require_trace_admission(trace_dir)
    payload_path = _threadblock_path(trace_dir)
    original = payload_path.read_bytes()

    with pytest.raises(
        trace_admission.ImmutableArtifactError,
        match="protected input tree changed",
    ):
        with monitor_admitted_trace_sources(trace_dir, admission):
            assert require_trace_admission(trace_dir) == admission
            payload_path.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
            payload_path.write_bytes(original)


def test_admitted_source_monitor_allows_unprotected_feature_outputs(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    extra_info = trace_dir / "extra_info"
    extra_info.mkdir()
    admission = require_trace_admission(trace_dir)

    with monitor_admitted_trace_sources(trace_dir, admission) as monitor:
        (extra_info / "feature.json").write_text("{}\n", encoding="utf-8")
        monitor.verify_unchanged()
        assert (
            require_trace_admission(trace_dir).trace_admission_sha256
            == admission.trace_admission_sha256
        )


def test_open_admitted_trace_sources_reads_exact_ephemeral_manifest(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    allocation_path = trace_dir / "extra_info" / "allocations.json"
    allocation_path.parent.mkdir()
    allocation_path.write_bytes(b"[]\n")
    admission = require_trace_admission(trace_dir)
    expected_root = (trace_dir / "dynamic_trace.pb").read_bytes()
    expected_stats = (trace_dir / "stats.csv").read_bytes()
    expected_threadblock = _threadblock_path(trace_dir).read_bytes()
    baseline = _open_descriptor_count()

    with open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    ) as sources:
        assert sources.admission == admission
        assert sources.threadblock_paths == (
            "threadblocks/device_0/stream_0/kernel_0/0.pb",
        )
        assert sources.read_dynamic_trace_snapshot().payload == expected_root
        assert sources.read_stats_snapshot().payload == expected_stats
        allocation_snapshot = sources.read_allocations_snapshot()
        assert allocation_snapshot is not None
        assert allocation_snapshot.payload == b"[]\n"
        assert sources.allocations == {}
        assert (
            sources.read_threadblock_snapshot(sources.threadblock_paths[0]).payload
            == expected_threadblock
        )
        sources.verify_unchanged()

    assert _open_descriptor_count() == baseline


def test_admitted_allocations_are_monitor_bracketed_and_deeply_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    allocation_path = trace_dir / "extra_info" / "allocations.json"
    allocation_path.parent.mkdir()
    allocation_path.write_text(
        json.dumps(
            [
                {
                    "allocation_id": 1,
                    "context_id": 2,
                    "device_id": 0,
                    "base_address": "0x1000",
                    "size_bytes": 64,
                    "alive": True,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    admission = require_trace_admission(trace_dir)

    with open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    ) as sources:
        original_verify = sources._monitor.verify_unchanged
        checks = 0

        def count_verify() -> None:
            nonlocal checks
            checks += 1
            original_verify()

        monkeypatch.setattr(sources._monitor, "verify_unchanged", count_verify)
        allocations = sources.allocations
        assert checks == 2
        assert allocations is not None
        assert allocations[1].base_address == 0x1000
        with pytest.raises(TypeError):
            allocations[2] = allocations[1]  # type: ignore[index]
        with pytest.raises(FrozenInstanceError):
            allocations[1].size_bytes = 1  # type: ignore[misc]
        object.__setattr__(allocations[1], "size_bytes", 7)
        second = sources.allocations
        assert second is not allocations
        assert second is not None
        assert second[1] is not allocations[1]
        assert second[1].size_bytes == 64


def test_open_admitted_trace_sources_uses_one_descriptor_anchored_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    expected_root = (trace_dir / "dynamic_trace.pb").read_bytes()
    expected_threadblock = _threadblock_path(trace_dir).read_bytes()
    original_reader = trace_admission.PinnedDirectoryReader
    reader_count = 0

    class CountingReader(original_reader):
        def __init__(self, root: Path) -> None:
            nonlocal reader_count
            reader_count += 1
            super().__init__(root)

    def forbidden_path_read(*_args, **_kwargs):
        raise AssertionError("admitted source capability reopened bytes by pathname")

    monkeypatch.setattr(trace_admission, "PinnedDirectoryReader", CountingReader)
    monkeypatch.setattr(Path, "read_bytes", forbidden_path_read)
    monkeypatch.setattr(Path, "open", forbidden_path_read)

    with open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    ) as sources:
        assert sources.read_dynamic_trace_snapshot().payload == expected_root
        assert sources.read_allocations_snapshot() is None
        assert sources.allocations is None
        assert (
            sources.read_threadblock_snapshot(sources.threadblock_paths[0]).payload
            == expected_threadblock
        )

    assert reader_count == 1


def test_open_admitted_trace_sources_reuses_caller_directory_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    with immutable_artifact.PinnedDirectoryReader(trace_dir) as reader:
        admission = trace_admission._require_trace_admission_snapshot(
            trace_dir,
            reader=reader,
        ).admission
        monkeypatch.setattr(
            trace_admission,
            "PinnedDirectoryReader",
            lambda _path: pytest.fail("supplied trace-root pin was reopened"),
        )
        with open_admitted_trace_sources(
            trace_dir,
            expected_admission=admission,
            reader=reader,
        ) as sources:
            assert sources.read_dynamic_trace_snapshot().payload
        reader.verify_path()
        assert reader.read("stats.csv")


@pytest.mark.parametrize(
    "substitution",
    ("symlink", "fifo", "same_bytes", "oversize", "metadata"),
)
def test_admitted_trace_source_rejects_substitution_before_first_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    victim = _threadblock_path(trace_dir)
    payload = victim.read_bytes()
    external = tmp_path / "external.pb"
    external.write_bytes(payload)
    baseline = _open_descriptor_count()
    sources = open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    )
    original_read_snapshot = immutable_artifact.PinnedDirectoryReader.read_snapshot
    substituted = False

    def substitute_then_read(
        reader,
        relative_path: str,
        *,
        max_bytes=None,
        expected_metadata=None,
    ):
        nonlocal substituted
        if not substituted and relative_path == sources.threadblock_paths[0]:
            if substitution == "symlink":
                victim.unlink()
                victim.symlink_to(external)
            elif substitution == "fifo":
                victim.unlink()
                os.mkfifo(victim)
            elif substitution in {"same_bytes", "oversize"}:
                replacement = victim.with_suffix(".replacement")
                replacement.write_bytes(
                    payload if substitution == "same_bytes" else payload + b"oversize"
                )
                os.replace(replacement, victim)
            else:
                victim.chmod(0o600)
            substituted = True
        return original_read_snapshot(
            reader,
            relative_path,
            max_bytes=max_bytes,
            expected_metadata=expected_metadata,
        )

    def unexpected_byte_read(*_args, **_kwargs):
        pytest.fail("substituted trace source bytes were consumed")

    monkeypatch.setattr(
        immutable_artifact.PinnedDirectoryReader,
        "read_snapshot",
        substitute_then_read,
    )
    monkeypatch.setattr(
        immutable_artifact,
        "_read_fd_snapshot",
        unexpected_byte_read,
    )
    relative_path = sources.threadblock_paths[0]
    with pytest.raises(immutable_artifact.ImmutableArtifactError):
        sources.read_threadblock_snapshot(relative_path)

    assert substituted is True
    assert sources._closed is True
    with pytest.raises(
        immutable_artifact.ImmutableArtifactError,
        match="capability is closed",
    ):
        _ = sources.threadblock_paths
    assert _open_descriptor_count() == baseline


@pytest.mark.parametrize(
    "mismatch",
    ("portable", "local_generation", "semantic_invalid", "missing_root"),
)
def test_open_admitted_trace_sources_rejects_mismatch_and_cleans_up(
    tmp_path: Path,
    mismatch: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    expected = require_trace_admission(trace_dir)
    if mismatch == "portable":
        expected = replace(
            expected,
            admitted_record_count=expected.admitted_record_count + 1,
        )
    elif mismatch == "local_generation":
        victim = _threadblock_path(trace_dir)
        replacement = victim.with_suffix(".replacement")
        replacement.write_bytes(victim.read_bytes())
        os.replace(replacement, victim)
    elif mismatch == "semantic_invalid":
        (trace_dir / "stats.csv").write_bytes(b"not,capture,stats\n")
    else:
        trace_dir.rename(tmp_path / "moved-trace")
    baseline = _open_descriptor_count()

    with pytest.raises(TraceAdmissionError) as caught:
        open_admitted_trace_sources(
            trace_dir,
            expected_admission=expected,
        )

    assert caught.value.typed_reason == "trace_snapshot_changed"
    assert _open_descriptor_count() == baseline


@pytest.mark.parametrize(
    ("failure", "reason"),
    (
        (
            immutable_artifact.ImmutableArtifactResourceError(
                "planted monitor resource failure"
            ),
            "trace_snapshot_resource_unavailable",
        ),
        (
            MemoryError("planted memory exhaustion"),
            "trace_snapshot_resource_unavailable",
        ),
        (
            OSError(errno.EMFILE, "planted process descriptor exhaustion"),
            "trace_snapshot_resource_unavailable",
        ),
        (
            OSError(errno.ENFILE, "planted system descriptor exhaustion"),
            "trace_snapshot_resource_unavailable",
        ),
        (
            immutable_artifact.ImmutableArtifactError(
                "planted monitor identity failure"
            ),
            "trace_snapshot_changed",
        ),
    ),
)
def test_open_admitted_trace_sources_translates_monitor_open_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    reason: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    baseline = _open_descriptor_count()

    def fail_monitor(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        trace_admission.PinnedPathSetMonitor,
        "from_reader",
        fail_monitor,
    )
    with pytest.raises(TraceAdmissionError) as caught:
        open_admitted_trace_sources(
            trace_dir,
            expected_admission=admission,
        )

    assert caught.value.typed_reason == reason
    assert caught.value.__cause__ is failure
    assert _open_descriptor_count() == baseline


def test_open_admitted_trace_sources_translates_authentication_monitor_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    baseline = _open_descriptor_count()
    original_from_reader = trace_admission.PinnedPathSetMonitor.from_reader
    failure = immutable_artifact.ImmutableArtifactError(
        "planted authentication monitor failure"
    )

    def arm_then_fail(*args, **kwargs):
        monitor = original_from_reader(*args, **kwargs)

        def fail_verify() -> None:
            raise failure

        monitor.verify_unchanged = fail_verify
        return monitor

    monkeypatch.setattr(
        trace_admission.PinnedPathSetMonitor,
        "from_reader",
        arm_then_fail,
    )
    with pytest.raises(TraceAdmissionError) as caught:
        open_admitted_trace_sources(
            trace_dir,
            expected_admission=admission,
        )

    assert caught.value.typed_reason == "trace_snapshot_changed"
    assert caught.value.__cause__ is failure
    assert _open_descriptor_count() == baseline


def test_admitted_trace_sources_entry_failure_closes_resources(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    victim = _threadblock_path(trace_dir)
    payload = victim.read_bytes()
    baseline = _open_descriptor_count()
    sources = open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    )
    victim.write_bytes(payload)

    with pytest.raises(immutable_artifact.ImmutableArtifactError):
        with sources:
            pytest.fail("context body reached after source mutation")

    assert _open_descriptor_count() == baseline


@pytest.mark.parametrize("forgery", ("size", "sha256"))
def test_admitted_trace_source_rechecks_ephemeral_size_and_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    baseline = _open_descriptor_count()
    sources = open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    )
    relative_path = sources.threadblock_paths[0]
    original_read = sources._reader.read_snapshot

    def forged_read(
        requested_path: str,
        *,
        max_bytes=None,
        expected_metadata=None,
    ):
        snapshot = original_read(
            requested_path,
            max_bytes=max_bytes,
            expected_metadata=expected_metadata,
        )
        if forgery == "size":
            payload = snapshot.payload + b"x"
        else:
            payload = bytes((snapshot.payload[0] ^ 1,)) + snapshot.payload[1:]
        return immutable_artifact.PinnedRegularSnapshot(
            payload,
            snapshot.metadata,
        )

    monkeypatch.setattr(sources._reader, "read_snapshot", forged_read)
    with pytest.raises(
        immutable_artifact.ImmutableArtifactError,
        match="size changed" if forgery == "size" else "SHA-256 changed",
    ):
        sources.read_threadblock_snapshot(relative_path)

    assert sources._closed is True
    assert _open_descriptor_count() == baseline


def test_admitted_trace_read_resource_failure_is_primary_and_poisoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    baseline = _open_descriptor_count()
    sources = open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    )
    failure = immutable_artifact.ImmutableArtifactResourceError(
        "planted admitted read resource failure"
    )
    original_close = sources._resources.close

    def fail_read(*_args, **_kwargs):
        raise failure

    def close_then_fail() -> None:
        original_close()
        raise RuntimeError("planted cleanup failure")

    monkeypatch.setattr(sources._reader, "read_snapshot", fail_read)
    monkeypatch.setattr(sources._resources, "close", close_then_fail)
    with pytest.raises(
        immutable_artifact.ImmutableArtifactResourceError,
        match="planted admitted read resource failure",
    ) as caught:
        sources.read_dynamic_trace_snapshot()

    assert caught.value is failure
    assert any(
        "cleanup failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert sources._closed is True
    assert _open_descriptor_count() == baseline


def test_unadmitted_threadblock_read_poison_closes_capability(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    baseline = _open_descriptor_count()
    sources = open_admitted_trace_sources(
        trace_dir,
        expected_admission=admission,
    )

    with pytest.raises(TraceAdmissionError) as caught:
        sources.read_threadblock_snapshot("stats.csv")

    assert caught.value.typed_reason == "trace_source_not_admitted"
    assert sources._closed is True
    assert _open_descriptor_count() == baseline


def test_open_admitted_trace_sources_does_not_weaken_read_limits(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    root_size = (trace_dir / "dynamic_trace.pb").stat().st_size
    limits = replace(
        DEFAULT_TRACE_ADMISSION_LIMITS,
        trace_root_bytes=root_size - 1,
    )
    baseline = _open_descriptor_count()

    with pytest.raises(TraceAdmissionError) as caught:
        open_admitted_trace_sources(
            trace_dir,
            expected_admission=admission,
            limits=limits,
        )

    assert caught.value.typed_reason == "trace_root_size_limit_exceeded"
    assert _open_descriptor_count() == baseline


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


def test_root_capabilities_are_exact_not_subset_only(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    root.memory_payload_classes.append(1)
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_capabilities_invalid"


def test_root_capabilities_require_exact_order(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    capabilities = list(root.memory_payload_classes)
    del root.memory_payload_classes[:]
    root.memory_payload_classes.extend(reversed(capabilities))
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_capabilities_invalid"


@pytest.mark.parametrize("field", ("device", "stream"))
def test_root_map_keys_must_match_embedded_ids(tmp_path: Path, field: str) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    if field == "device":
        root.gpu_device[0].id = 1
    else:
        root.gpu_device[0].streams[0].id = 1
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_launch_identity_mismatch"


def test_root_requires_positive_function_identity(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    root.gpu_device[0].streams[0].kernels[0].function_unique_id = 0
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "function_unique_id_invalid"


def test_duplicate_launch_identity_is_rejected(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    stream = root.gpu_device[0].streams[0]
    duplicate = stream.kernels.add()
    duplicate.CopyFrom(stream.kernels[0])
    _write_root(trace_dir, root)

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_launch_identity_mismatch"


def test_every_root_payload_is_semantically_walked(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    root = _read_root(trace_dir)
    stream = root.gpu_device[0].streams[0]
    second = stream.kernels.add()
    second.CopyFrom(stream.kernels[0])
    second.id = 1
    second.function_unique_id = 2
    _write_root(trace_dir, root)

    _trace_pb2, threadblock_pb2 = _pb_modules()
    message = threadblock_pb2.threadblock()
    message.ParseFromString(_threadblock_path(trace_dir).read_bytes())
    instruction = message.warps[0].instructions[0]
    instruction.function_unique_id = 2
    instruction.ClearField("warpid_tb")
    second_path = _threadblock_path(trace_dir, 1)
    second_path.parent.mkdir(parents=True)
    second_path.write_bytes(message.SerializeToString())

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "warpid_tb_missing"


def test_payloads_are_bijectively_joined_to_root_launches(tmp_path: Path) -> None:
    missing_dir = _write_trace(tmp_path / "missing")
    _threadblock_path(missing_dir).unlink()
    orphan = _threadblock_path(missing_dir, 9)
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(missing_dir)
    assert caught.value.typed_reason == "trace_launch_payload_missing"

    orphan_dir = _write_trace(tmp_path / "orphan")
    orphan = _threadblock_path(orphan_dir, 9)
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(_threadblock_path(orphan_dir).read_bytes())
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(orphan_dir)
    assert caught.value.typed_reason == "trace_payload_orphan"


def test_instruction_function_identity_must_join_root(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    _trace_pb2, threadblock_pb2 = _pb_modules()
    message = threadblock_pb2.threadblock()
    path = _threadblock_path(trace_dir)
    message.ParseFromString(path.read_bytes())
    message.warps[0].instructions[0].function_unique_id = 2
    path.write_bytes(message.SerializeToString())

    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "function_unique_id_mismatch"


def test_block_identity_is_required_and_range_checked(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "missing")
    _trace_pb2, threadblock_pb2 = _pb_modules()
    message = threadblock_pb2.threadblock()
    path = _threadblock_path(trace_dir)
    message.ParseFromString(path.read_bytes())
    message.ClearField("block_id")
    path.write_bytes(message.SerializeToString())
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "block_id_missing"

    trace_dir = _write_trace(tmp_path / "range")
    path = _threadblock_path(trace_dir)
    message = threadblock_pb2.threadblock()
    message.ParseFromString(path.read_bytes())
    message.block_id.x = 1
    path.write_bytes(message.SerializeToString())
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "block_id_out_of_range"


def test_stats_counters_are_uint64_and_totals_are_checked(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "too-wide")
    (trace_dir / "stats.csv").write_text(
        f"total_insts,total_reported_insts\n{2**64},1\n", encoding="utf-8"
    )
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "stats_schema_invalid"

    trace_dir = _write_trace(tmp_path / "overflow")
    (trace_dir / "stats.csv").write_text(
        "total_insts,total_reported_insts\n"
        f"{2**64 - 1},1\n"
        "1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(trace_dir)
    assert caught.value.typed_reason == "trace_counter_overflow"

    trace_dir = _write_trace(tmp_path / "maximum")
    (trace_dir / "stats.csv").write_text(
        "total_insts,total_reported_insts\n"
        f"{2**64 - 1},{2**64 - 1}\n",
        encoding="utf-8",
    )
    admitted = require_trace_admission(trace_dir)
    assert admitted.expected_record_count == 2**64 - 1
    assert admitted.admitted_record_count == 2**64 - 1


def test_stats_rows_and_columns_have_independent_bounds(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "rows")
    (trace_dir / "stats.csv").write_text(
        "total_insts,total_reported_insts\n1,1\n1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(
            trace_dir,
            limits=replace(DEFAULT_TRACE_ADMISSION_LIMITS, stats_rows=1),
        )
    assert caught.value.typed_reason == "stats_row_limit_exceeded"

    trace_dir = _write_trace(tmp_path / "columns")
    (trace_dir / "stats.csv").write_text(
        "total_insts,total_reported_insts,extra\n1,1,x\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceAdmissionError) as caught:
        require_trace_admission(
            trace_dir,
            limits=replace(DEFAULT_TRACE_ADMISSION_LIMITS, stats_columns=2),
        )
    assert caught.value.typed_reason == "stats_column_limit_exceeded"


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
