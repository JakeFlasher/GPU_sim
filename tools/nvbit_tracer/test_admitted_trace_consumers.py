from __future__ import annotations

import builtins
import hashlib
import importlib
import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.divergence_proxies.bundle_from_trace import (
    consume_admitted_trace_sources as consume_divergence_sources,
    consume_trace_dir as consume_divergence_trace_dir,
)
from tools.hrd_profiler.bundle_from_trace import (
    emit_hrd_bundle,
    emit_hrd_bundle_from_admitted_sources,
)
from tools.hrd_profiler.observation_builder import (
    _walk_launch,
    build_launches_from_admitted_sources,
    build_launches_from_trace_dir,
)
from tools.local_labels import (
    regenerate_sidecars_for_cell as regenerate_cell,
    regenerate_v5_trace_sidecars as regenerate_bulk,
)
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
)
from tools.nvbit_tracer import import_trace_contract, trace_admission
from tools.nvbit_tracer.trace_admission import (
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)
from tools.order_projections.projector import (
    compute_order_projections,
    compute_order_projections_from_admitted_sources,
)
from tools.sector_stream.consume_trace import (
    AdmittedSectorStream,
    consume_admitted_trace_sources as consume_sector_sources,
    consume_trace_dir as consume_sector_trace_dir,
)
from tools.sharing_locality.classifier import (
    compute_sharing_locality,
    compute_sharing_locality_from_admitted_sources,
)
from workloads.simbench_bw import cache_predictor, run_suite


_MANIFEST_PAIRING_INDEX = {
    "slug": "fixture",
    "shape_point": "trace_00",
    "binary_sha256": "0" * 64,
    "source_commit": "0" * 64,
    "kernel_signature_sha256": "0" * 64,
}
_DEVICE = {
    "device_name": "fixture GPU",
    "compute_capability_major": 8,
    "compute_capability_minor": 6,
    "sm_count": 1,
    "total_global_mem_bytes": 1024,
    "l2_cache_size_bytes": 6 * 1024 * 1024,
}


def _write_memory_trace(
    root: Path,
    *,
    kernel_id: int = 0,
    kernel_name: str = "fixture_kernel",
    address_value: int = 0xCAFE0000,
) -> Path:
    trace_pb2 = trace_admission._import_trace_pb2()
    threadblock_pb2, instruction_pb2, _address_pb2 = (
        import_trace_contract._import_pb_modules()
    )

    root.mkdir()
    trace = trace_pb2.Trace()
    trace.trace_contract_id = trace_admission.EXPECTED_TRACE_CONTRACT_ID
    trace.memory_payload_classes.extend([1, 2, 3, 6, 8])
    device = trace.gpu_device[0]
    device.id = 0
    stream = device.streams[0]
    stream.id = 0
    kernel = stream.kernels.add()
    kernel.id = kernel_id
    kernel.name = kernel_name
    kernel.function_unique_id = kernel_id + 1
    kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
    kernel.block_dim.x = 32
    kernel.block_dim.y = kernel.block_dim.z = 1
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0
    (root / "dynamic_trace.pb").write_bytes(trace.SerializeToString())

    tb = threadblock_pb2.threadblock()
    tb.block_id.x = tb.block_id.y = tb.block_id.z = 0
    warp = tb.warps[0]
    warp.id = 0
    inst = warp.instructions.add()
    inst.pc = inst.vpc = 0x1000
    inst.active_mask = inst.predicate_mask = 1
    inst.function_unique_id = kernel_id + 1
    inst.data_width = 4
    inst.warpid_tb = 0
    inst.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
    inst.memory_op_access_kind = instruction_pb2.MEMORY_OP_ACCESS_KIND_LOAD
    address = inst.addresses.add()
    address.data_width = 4
    address.allocation_id = 7
    address.addrs.append(address_value)
    address.addrs.extend([-1] * 31)
    tb_path = (
        root
        / "threadblocks"
        / "device_0"
        / "stream_0"
        / f"kernel_{kernel_id}"
        / "tb_0,0,0.pb"
    )
    tb_path.parent.mkdir(parents=True)
    tb_path.write_bytes(tb.SerializeToString())

    allocation_path = root / "extra_info" / "allocations.json"
    allocation_path.parent.mkdir()
    allocation_path.write_text(
        "[{\"allocation_id\":7,\"context_id\":1,\"device_id\":0,"
        "\"base_address\":\"0xcafe0000\",\"size_bytes\":65536,"
        "\"alive\":true}]\n",
        encoding="utf-8",
    )
    (root / "stats.csv").write_text(
        "kernel id,kernel mangled name,total_insts,total_reported_insts\n"
        f"kernel-{kernel_id}.trace,{kernel_name},1,1\n",
        encoding="utf-8",
    )
    return root


def _write_two_launch_trace(root: Path) -> Path:
    trace_pb2 = trace_admission._import_trace_pb2()
    threadblock_pb2, instruction_pb2, _address_pb2 = (
        import_trace_contract._import_pb_modules()
    )
    root.mkdir()
    trace = trace_pb2.Trace()
    trace.trace_contract_id = trace_admission.EXPECTED_TRACE_CONTRACT_ID
    trace.memory_payload_classes.extend([1, 2, 3, 6, 8])
    device = trace.gpu_device[0]
    device.id = 0
    stream = device.streams[0]
    stream.id = 0
    for kernel_id in (0, 1):
        kernel = stream.kernels.add()
        kernel.id = kernel_id
        kernel.name = f"fixture_kernel_{kernel_id}"
        kernel.function_unique_id = kernel_id + 1
        kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
        kernel.block_dim.x = 32
        kernel.block_dim.y = kernel.block_dim.z = 1
        kernel.shared_memory_static_bytes = 0
        kernel.shared_memory_dynamic_bytes = 0

        tb = threadblock_pb2.threadblock()
        tb.block_id.x = tb.block_id.y = tb.block_id.z = 0
        warp = tb.warps[0]
        warp.id = 0
        inst = warp.instructions.add()
        inst.pc = inst.vpc = 0x1000
        inst.active_mask = inst.predicate_mask = 1
        inst.function_unique_id = kernel_id + 1
        inst.data_width = 4
        inst.warpid_tb = 0
        inst.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
        inst.memory_op_access_kind = (
            instruction_pb2.MEMORY_OP_ACCESS_KIND_LOAD
        )
        address = inst.addresses.add()
        address.data_width = 4
        address.allocation_id = 7
        address.addrs.append(0xCAFE0000)
        address.addrs.extend([-1] * 31)
        tb_path = (
            root / "threadblocks" / "device_0" / "stream_0"
            / f"kernel_{kernel_id}" / "tb_0,0,0.pb"
        )
        tb_path.parent.mkdir(parents=True)
        tb_path.write_bytes(tb.SerializeToString())

    (root / "dynamic_trace.pb").write_bytes(trace.SerializeToString())
    allocation_path = root / "extra_info" / "allocations.json"
    allocation_path.parent.mkdir()
    allocation_path.write_text(
        "[{\"allocation_id\":7,\"context_id\":1,\"device_id\":0,"
        "\"base_address\":\"0xcafe0000\",\"size_bytes\":65536,"
        "\"alive\":true}]\n",
        encoding="utf-8",
    )
    (root / "stats.csv").write_text(
        "kernel id,kernel mangled name,total_insts,total_reported_insts\n"
        "kernel-0.trace,fixture_kernel_0,1,1\n"
        "kernel-1.trace,fixture_kernel_1,1,1\n",
        encoding="utf-8",
    )
    return root


def _forbidden_validation(*_args, **_kwargs):
    raise AssertionError("capability core revalidated a raw trace path")


def test_capability_sources_are_mandatory_keyword_only() -> None:
    cores = (
        consume_sector_sources,
        compute_order_projections_from_admitted_sources,
        compute_sharing_locality_from_admitted_sources,
        build_launches_from_admitted_sources,
        consume_divergence_sources,
        emit_hrd_bundle_from_admitted_sources,
        _walk_launch,
    )
    for core in cores:
        parameter = inspect.signature(core).parameters["sources"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


def test_path_compatible_wrappers_admit_open_and_invoke_cores(
    tmp_path: Path,
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")

    launches = build_launches_from_trace_dir(
        trace_dir,
        capacity_bytes=6 * 1024 * 1024,
        num_l2_slices=1,
    )
    assert launches.total_observations == 1

    order = compute_order_projections(
        trace_dir=trace_dir,
        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
        device=_DEVICE,
        capacity_bytes=6 * 1024 * 1024,
    )
    assert order["evidence"][0]["threadblocks_walked"] == 1

    sharing = compute_sharing_locality(
        trace_dir=trace_dir,
        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
        device=_DEVICE,
    )
    assert sharing["evidence"][0]["addresses_observed"] == 1

    hrd = emit_hrd_bundle(
        trace_dir=trace_dir,
        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
        device=_DEVICE,
        cuda_l2_cache_size_bytes=6 * 1024 * 1024,
        cuda_num_l2_slices=1,
    )
    assert hrd["schema_admits"] is True

    divergence = consume_divergence_trace_dir(
        trace_dir,
        "sm_86",
        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
    )
    assert divergence["schema_admits"] is True
    assert divergence["sector_stream_records_consumed"] == 1


def test_existing_sector_jsonl_is_never_an_input_authority(tmp_path: Path) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    sector_jsonl = trace_dir / "sector_stream" / "sector_stream.jsonl"
    sector_jsonl.parent.mkdir()
    forged_payload = b"this is not a sector-stream record\n"
    sector_jsonl.write_bytes(forged_payload)

    divergence = consume_divergence_trace_dir(
        trace_dir,
        "sm_86",
        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
    )
    assert divergence["schema_admits"] is True
    assert divergence["sector_stream_records_consumed"] == 1
    assert sector_jsonl.read_bytes() == forged_payload

    hrd = emit_hrd_bundle(
        trace_dir=trace_dir,
        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
        device=_DEVICE,
        cuda_l2_cache_size_bytes=6 * 1024 * 1024,
        cuda_num_l2_slices=1,
    )
    assert hrd["schema_admits"] is True
    assert hrd["total_observations"] == 1
    assert sector_jsonl.read_bytes() == forged_payload


def test_capability_cores_never_reopen_or_inventory_trace_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        raw_source_names = {"dynamic_trace.pb", "stats.csv"}
        original_rglob = Path.rglob
        original_glob = Path.glob
        original_read_bytes = Path.read_bytes
        original_open = Path.open
        original_resolve = Path.resolve
        original_exists = Path.exists
        original_stat = Path.stat
        trace_root = Path(os.path.abspath(os.fspath(trace_dir)))

        def _guard_raw_trace_path(original):
            def _guarded(path: Path, *args, **kwargs):
                candidate = Path(os.path.abspath(os.fspath(path)))
                try:
                    relative = candidate.relative_to(trace_root)
                except ValueError:
                    return original(path, *args, **kwargs)
                if (
                    not relative.parts
                    or relative.parts[0] in {"threadblocks", "extra_info"}
                    or relative.name in raw_source_names
                    or relative.parts
                    == ("sector_stream", "sector_stream.jsonl")
                ):
                    raise AssertionError(
                        "capability core accessed a raw trace through pathlib"
                    )
                return original(path, *args, **kwargs)

            return _guarded

        with monkeypatch.context() as hostile:
            hostile.setattr(Path, "rglob", _guard_raw_trace_path(original_rglob))
            hostile.setattr(Path, "glob", _guard_raw_trace_path(original_glob))
            hostile.setattr(
                Path,
                "read_bytes",
                _guard_raw_trace_path(original_read_bytes),
            )
            hostile.setattr(Path, "open", _guard_raw_trace_path(original_open))
            hostile.setattr(
                Path,
                "resolve",
                _guard_raw_trace_path(original_resolve),
            )
            hostile.setattr(
                Path,
                "exists",
                _guard_raw_trace_path(original_exists),
            )
            hostile.setattr(Path, "stat", _guard_raw_trace_path(original_stat))
            hostile.setattr(
                import_trace_contract,
                "validate_trace_dir",
                _forbidden_validation,
            )
            hostile.setattr(
                import_trace_contract,
                "load_allocations_sidecar",
                _forbidden_validation,
            )

            sector = consume_sector_sources(
                sources=sources,
                output_root=trace_dir,
                write_jsonl=True,
            )
            assert isinstance(sector, AdmittedSectorStream)
            assert sector.contract_ok is True
            assert sector.total_records == 1
            assert len(sector.records) == 1

            order = compute_order_projections_from_admitted_sources(
                sources=sources,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device=_DEVICE,
                capacity_bytes=6 * 1024 * 1024,
            )
            assert order["evidence"][0]["threadblocks_walked"] == 1

            sharing = compute_sharing_locality_from_admitted_sources(
                sources=sources,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device={
                    key: _DEVICE[key]
                    for key in (
                        "device_name",
                        "compute_capability_major",
                        "compute_capability_minor",
                        "sm_count",
                    )
                },
            )
            assert sharing["evidence"][0]["addresses_observed"] == 1

            launches = build_launches_from_admitted_sources(
                sources=sources,
                capacity_bytes=6 * 1024 * 1024,
                num_l2_slices=1,
            )
            assert launches.total_threadblocks == 1
            assert launches.total_observations == 1

            divergence = consume_divergence_sources(
                sources=sources,
                sector_stream=sector,
                output_root=trace_dir,
                device="sm_86",
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
            )
            assert divergence["schema_admits"] is True
            assert divergence["sector_stream_records_consumed"] == 1

            hrd = emit_hrd_bundle_from_admitted_sources(
                sources=sources,
                sector_stream=sector,
                output_root=trace_dir,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device=_DEVICE,
                cuda_l2_cache_size_bytes=6 * 1024 * 1024,
                cuda_num_l2_slices=1,
            )
            assert hrd["contract_ok"] is True
            assert hrd["schema_admits"] is True
            assert hrd["launches_consumed"] == 1

            cache = (
                cache_predictor.predict_cache_hit_rate_from_admitted_sources(
                    sources=sources,
                    sector_stream=sector,
                    trace_dir=trace_dir,
                    sector_stream_path=sector.jsonl_path,
                    cache_model_capacity_bytes=6 * 1024 * 1024,
                    cache_model_num_l2_slices=1,
                    manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                    device="sm_86",
                    sharing_locality_path=(
                        trace_dir
                        / "sharing_locality"
                        / "sharing_locality.json"
                    ),
                    hrd_profile_path=(
                        trace_dir / "hrd_profile" / "hrd_profile.json"
                    ),
                )
            )
            assert (
                cache["input_sidecar_digests"]["sector_stream"]["sha256"]
                == hashlib.sha256(sector.jsonl_bytes()).hexdigest()
            )


def test_hrd_walker_rejects_arbitrary_threadblock_path_via_capability(
    tmp_path: Path,
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    sources = open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    )
    threadblock_pb2 = import_trace_contract._import_pb_modules()[0]
    try:
        with pytest.raises(TraceAdmissionError) as caught:
            _walk_launch(
                ["threadblocks/device_0/stream_0/kernel_0/not-admitted.pb"],
                sources=sources,
                allocations=sources.allocations,
                capacity_bytes=6 * 1024 * 1024,
                threadblock_pb2_mod=threadblock_pb2,
                num_l2_slices=1,
            )
        assert caught.value.typed_reason == "trace_source_not_admitted"
    finally:
        sources.close()


@pytest.mark.parametrize(
    "failure",
    (
        ImmutableArtifactResourceError("planted resource failure"),
        TraceAdmissionError("trace_source_not_admitted", "planted path"),
    ),
)
def test_sector_core_propagates_capability_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    sources = open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    )

    def fail_read(_relative_path: str):
        raise failure

    monkeypatch.setattr(sources, "read_threadblock_snapshot", fail_read)
    try:
        with pytest.raises(type(failure)) as caught:
            consume_sector_sources(
                sources=sources,
                output_root=None,
                write_jsonl=False,
            )
        assert caught.value is failure
    finally:
        sources.close()


def test_hrd_semantic_skip_does_not_mask_source_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    sources = open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    )
    failure = ImmutableArtifactResourceError("planted closure failure")
    verify_calls = 0

    def fail_at_closure() -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise failure

    monkeypatch.setattr(sources, "verify_unchanged", fail_at_closure)
    try:
        with pytest.raises(ImmutableArtifactResourceError) as caught:
            build_launches_from_admitted_sources(
                sources=sources,
                capacity_bytes=6 * 1024 * 1024,
                num_l2_slices=None,
            )
        assert caught.value is failure
    finally:
        sources.close()


def test_consumer_propagates_source_generation_change(tmp_path: Path) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    tb_path = next((trace_dir / "threadblocks").rglob("*.pb"))
    payload = tb_path.read_bytes()
    sources = open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    )
    try:
        tb_path.write_bytes(payload)
        with pytest.raises(ImmutableArtifactError):
            compute_sharing_locality_from_admitted_sources(
                sources=sources,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device=_DEVICE,
            )
    finally:
        sources.close()


@pytest.mark.parametrize("substitution", ("symlink", "fifo"))
def test_path_wrapper_rejects_special_threadblock_sources(
    tmp_path: Path, substitution: str
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    tb_path = next((trace_dir / "threadblocks").rglob("*.pb"))
    payload = tb_path.read_bytes()
    tb_path.unlink()
    if substitution == "symlink":
        external = tmp_path / "external.pb"
        external.write_bytes(payload)
        tb_path.symlink_to(external)
    else:
        os.mkfifo(tb_path)

    with pytest.raises(TraceAdmissionError):
        consume_sector_trace_dir(trace_dir, write_jsonl=False)


def test_path_wrapper_rejects_same_byte_generation_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    expected = require_trace_admission(trace_dir)
    tb_path = next((trace_dir / "threadblocks").rglob("*.pb"))
    replacement = tb_path.with_suffix(".replacement")
    replacement.write_bytes(tb_path.read_bytes())
    os.replace(replacement, tb_path)
    monkeypatch.setattr(
        "tools.sector_stream.consume_trace.require_trace_admission",
        lambda _trace_dir: expected,
    )

    with pytest.raises(TraceAdmissionError) as caught:
        consume_sector_trace_dir(trace_dir, write_jsonl=False)
    assert caught.value.typed_reason == "trace_snapshot_changed"


def test_admitted_sector_stream_is_deeply_immutable_and_sealed(
    tmp_path: Path,
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector = consume_sector_sources(
            sources=sources,
            output_root=None,
            write_jsonl=False,
        )
        assert isinstance(sector, AdmittedSectorStream)
        assert sector.trace_admission_sha256 == admission.trace_admission_sha256
        assert (
            sector.trace_filesystem_generation_sha256
            == admission.trace_filesystem_generation_sha256
        )
        assert isinstance(sector.records, tuple)
        with pytest.raises(TypeError):
            sector.records[0]["result_kind"] = "forged"  # type: ignore[index]
        with pytest.raises(FrozenInstanceError):
            sector.trace_admission_sha256 = "f" * 64  # type: ignore[misc]
        detached = sector.mutable_records()
        detached[0]["result_kind"] = "forged"
        assert sector.records[0]["result_kind"] == "sector_stream"
        assert b'"result_kind": "sector_stream"' in sector.jsonl_bytes()
        with pytest.raises(TypeError, match="AdmittedSectorStream"):
            consume_divergence_sources(
                sources=sources,
                sector_stream=sector.records,  # type: ignore[arg-type]
                output_root=trace_dir,
                device="sm_86",
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
            )
        with pytest.raises(TypeError, match="AdmittedSectorStream"):
            emit_hrd_bundle_from_admitted_sources(
                sources=sources,
                sector_stream=sector.records,  # type: ignore[arg-type]
                output_root=trace_dir,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device=_DEVICE,
                cuda_l2_cache_size_bytes=6 * 1024 * 1024,
                cuda_num_l2_slices=1,
            )
        with pytest.raises(TypeError, match="AdmittedSectorStream"):
            cache_predictor.predict_cache_hit_rate_from_admitted_sources(
                sources=sources,
                sector_stream=sector.records,  # type: ignore[arg-type]
                trace_dir=trace_dir,
                sector_stream_path=None,
                cache_model_capacity_bytes=6 * 1024 * 1024,
                cache_model_num_l2_slices=1,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device="sm_86",
            )

    with pytest.raises(TypeError):
        AdmittedSectorStream()  # type: ignore[call-arg]


def test_sector_consumers_reject_cross_wired_portable_seal_before_publication(
    tmp_path: Path,
) -> None:
    first = _write_memory_trace(tmp_path / "first", address_value=0xCAFE0000)
    second = _write_memory_trace(tmp_path / "second", address_value=0xCAFE0020)
    first_admission = require_trace_admission(first)
    second_admission = require_trace_admission(second)
    assert (
        first_admission.trace_admission_sha256
        != second_admission.trace_admission_sha256
    )
    with open_admitted_trace_sources(
        first, expected_admission=first_admission
    ) as first_sources, open_admitted_trace_sources(
        second, expected_admission=second_admission
    ) as second_sources:
        sector = consume_sector_sources(
            sources=first_sources,
            output_root=None,
            write_jsonl=False,
        )
        for consumer in ("divergence", "hrd", "cache"):
            with pytest.raises(TraceAdmissionError) as caught:
                if consumer == "divergence":
                    consume_divergence_sources(
                        sources=second_sources,
                        sector_stream=sector,
                        output_root=second,
                        device="sm_86",
                        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                    )
                elif consumer == "hrd":
                    emit_hrd_bundle_from_admitted_sources(
                        sources=second_sources,
                        sector_stream=sector,
                        output_root=second,
                        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                        device=_DEVICE,
                        cuda_l2_cache_size_bytes=6 * 1024 * 1024,
                        cuda_num_l2_slices=1,
                    )
                else:
                    cache_predictor.predict_cache_hit_rate_from_admitted_sources(
                        sources=second_sources,
                        sector_stream=sector,
                        trace_dir=second,
                        sector_stream_path=None,
                        cache_model_capacity_bytes=6 * 1024 * 1024,
                        cache_model_num_l2_slices=1,
                        manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                        device="sm_86",
                    )
            assert caught.value.typed_reason == "trace_snapshot_changed"
            assert "trace_admission_sha256" in str(caught.value)
    assert not (second / "divergence_proxies").exists()
    assert not (second / "hrd_profile").exists()


def test_sector_consumers_reject_same_bytes_different_generation_seal(
    tmp_path: Path,
) -> None:
    first = _write_memory_trace(tmp_path / "first")
    second = _write_memory_trace(tmp_path / "second")
    first_admission = require_trace_admission(first)
    second_admission = require_trace_admission(second)
    assert (
        first_admission.trace_admission_sha256
        == second_admission.trace_admission_sha256
    )
    assert (
        first_admission.trace_filesystem_generation_sha256
        != second_admission.trace_filesystem_generation_sha256
    )
    with open_admitted_trace_sources(
        first, expected_admission=first_admission
    ) as first_sources, open_admitted_trace_sources(
        second, expected_admission=second_admission
    ) as second_sources:
        sector = consume_sector_sources(
            sources=first_sources,
            output_root=None,
            write_jsonl=False,
        )
        with pytest.raises(TraceAdmissionError) as caught:
            consume_divergence_sources(
                sources=second_sources,
                sector_stream=sector,
                output_root=second,
                device="sm_86",
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
            )
        assert caught.value.typed_reason == "trace_snapshot_changed"
        assert "trace_filesystem_generation_sha256" in str(caught.value)
        with pytest.raises(TraceAdmissionError) as hrd_error:
            emit_hrd_bundle_from_admitted_sources(
                sources=second_sources,
                sector_stream=sector,
                output_root=second,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device=_DEVICE,
                cuda_l2_cache_size_bytes=6 * 1024 * 1024,
                cuda_num_l2_slices=1,
            )
        assert hrd_error.value.typed_reason == "trace_snapshot_changed"
        assert "trace_filesystem_generation_sha256" in str(hrd_error.value)
        with pytest.raises(TraceAdmissionError) as cache_error:
            cache_predictor.predict_cache_hit_rate_from_admitted_sources(
                sources=second_sources,
                sector_stream=sector,
                trace_dir=second,
                sector_stream_path=None,
                cache_model_capacity_bytes=6 * 1024 * 1024,
                cache_model_num_l2_slices=1,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device="sm_86",
            )
        assert cache_error.value.typed_reason == "trace_snapshot_changed"
        assert "trace_filesystem_generation_sha256" in str(cache_error.value)
    assert not (second / "divergence_proxies").exists()
    assert not (second / "hrd_profile").exists()


def test_multi_launch_trace_is_refused_at_all_singular_feature_boundaries(
    tmp_path: Path,
) -> None:
    trace_dir = _write_two_launch_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    assert admission.launch_count == 2
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector = consume_sector_sources(
            sources=sources,
            output_root=None,
            write_jsonl=False,
        )
        assert len(sector.records) == 2

        with pytest.raises(TraceAdmissionError) as divergence_error:
            consume_divergence_sources(
                sources=sources,
                sector_stream=sector,
                output_root=trace_dir,
                device="sm_86",
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
            )
        assert (
            divergence_error.value.typed_reason
            == "trace_launch_count_unsupported"
        )

        with pytest.raises(TraceAdmissionError) as order_error:
            compute_order_projections_from_admitted_sources(
                sources=sources,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device=_DEVICE,
                capacity_bytes=6 * 1024 * 1024,
            )
        assert order_error.value.typed_reason == "trace_launch_count_unsupported"

        with pytest.raises(TraceAdmissionError) as hrd_error:
            emit_hrd_bundle_from_admitted_sources(
                sources=sources,
                sector_stream=sector,
                output_root=trace_dir,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device=_DEVICE,
                cuda_l2_cache_size_bytes=6 * 1024 * 1024,
                cuda_num_l2_slices=1,
            )
        assert hrd_error.value.typed_reason == "trace_launch_count_unsupported"

    assert not (trace_dir / "divergence_proxies").exists()
    assert not (trace_dir / "hrd_profile").exists()


@pytest.mark.parametrize(
    ("stats_kernel_id", "stats_kernel_name"),
    ((9, "fixture_kernel"), (0, "wrong_kernel_name")),
)
def test_divergence_binds_stats_identity_to_sole_admitted_launch(
    tmp_path: Path,
    stats_kernel_id: int,
    stats_kernel_name: str,
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    (trace_dir / "stats.csv").write_text(
        "kernel id,kernel mangled name,total_insts,total_reported_insts\n"
        f"kernel-{stats_kernel_id}.trace,{stats_kernel_name},1,1\n",
        encoding="utf-8",
    )
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector = consume_sector_sources(
            sources=sources,
            output_root=None,
            write_jsonl=False,
        )
        with pytest.raises(TraceAdmissionError) as caught:
            consume_divergence_sources(
                sources=sources,
                sector_stream=sector,
                output_root=trace_dir,
                device="sm_86",
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
            )
        assert caught.value.typed_reason == "trace_launch_identity_mismatch"
    assert not (trace_dir / "divergence_proxies").exists()


def test_divergence_requires_exactly_one_stats_row(tmp_path: Path) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    with (trace_dir / "stats.csv").open("a", encoding="utf-8") as stats:
        stats.write("kernel-1.trace,other_kernel,1,1\n")
    admission = require_trace_admission(trace_dir)
    assert admission.launch_count == 1
    assert admission.stats_row_count == 2
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector = consume_sector_sources(
            sources=sources,
            output_root=None,
            write_jsonl=False,
        )
        with pytest.raises(TraceAdmissionError) as caught:
            consume_divergence_sources(
                sources=sources,
                sector_stream=sector,
                output_root=trace_dir,
                device="sm_86",
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
            )
        assert caught.value.typed_reason == "stats_schema_invalid"
    assert not (trace_dir / "divergence_proxies").exists()


def test_sector_capability_cores_reject_bare_records_and_require_mpi() -> None:
    divergence_parameters = inspect.signature(
        consume_divergence_sources
    ).parameters
    hrd_parameters = inspect.signature(
        emit_hrd_bundle_from_admitted_sources
    ).parameters
    assert "sector_stream_records" not in divergence_parameters
    assert "sector_stream_records" not in hrd_parameters
    assert divergence_parameters["sector_stream"].default is inspect.Parameter.empty
    assert hrd_parameters["sector_stream"].default is inspect.Parameter.empty
    assert (
        divergence_parameters["manifest_pairing_index"].default
        is inspect.Parameter.empty
    )

    with pytest.raises(TypeError, match="manifest_pairing_index"):
        consume_divergence_sources(
            sources=None,  # type: ignore[arg-type]
            sector_stream=None,  # type: ignore[arg-type]
            output_root=Path("unused"),
            device="sm_86",
        )


@pytest.mark.parametrize(
    "failure",
    (
        TraceAdmissionError("trace_snapshot_changed", "planted admission failure"),
        ImmutableArtifactError("planted immutable failure"),
        ImmutableArtifactLimitError("planted immutable limit failure"),
        ImmutableArtifactResourceError("planted immutable resource failure"),
    ),
)
def test_cache_predictor_propagates_identical_hard_source_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    trace_dir = _write_memory_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    hrd_path = Path(__file__).resolve().parents[1] / "hrd_profiler"
    monkeypatch.syspath_prepend(str(hrd_path))
    short_observation_builder = importlib.import_module("observation_builder")

    def fail_builder(**_kwargs):
        raise failure

    monkeypatch.setattr(
        short_observation_builder,
        "build_launches_from_admitted_sources",
        fail_builder,
    )
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector = consume_sector_sources(
            sources=sources,
            output_root=None,
            write_jsonl=False,
        )
        with pytest.raises(type(failure)) as caught:
            cache_predictor.predict_cache_hit_rate_from_admitted_sources(
                sources=sources,
                sector_stream=sector,
                trace_dir=trace_dir,
                sector_stream_path=None,
                cache_model_capacity_bytes=6 * 1024 * 1024,
                cache_model_num_l2_slices=1,
                manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
                device="sm_86",
            )
        assert caught.value is failure


def test_run_suite_sector_failure_suppresses_complete_feature_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = SimpleNamespace(launch_count=1)
    verification_calls: list[None] = []
    sources = SimpleNamespace(
        admission=admission,
        verify_unchanged=lambda: verification_calls.append(None),
    )
    monkeypatch.setattr(
        run_suite,
        "_ensure_sector_stream_jsonl",
        lambda *_args, **_kwargs: (False, "planted_sector_failure", None),
    )
    called: list[str] = []
    for name in (
        "_emit_hrd_bundle_for_run_summary",
        "_emit_divergence_proxies_for_run_summary",
        "_emit_sharing_locality_for_run_summary",
        "_emit_order_projections_for_run_summary",
        "_emit_perturbation_run_ledger_for_run_summary",
        "_emit_trace_convergence_confidence_for_run_summary",
    ):
        monkeypatch.setattr(
            run_suite,
            name,
            lambda *_args, _name=name, **_kwargs: called.append(_name),
        )

    artifact_record: dict[str, object] = {}
    run_suite._emit_admitted_trace_features(
        artifact_record=artifact_record,
        artifact_dir=tmp_path,
        traces_folder=tmp_path / "trace",
        trace_admission=admission,
        slug="fixture",
        shape_point={"name": "trace_00"},
        binary_sha256="0" * 64,
        kernel_signature_sha256="0" * 64,
        props=SimpleNamespace(),
        trace_sources=sources,
    )

    assert called == []
    assert len(verification_calls) == 2
    assert artifact_record["sector_stream_path"] is None
    assert artifact_record["sector_stream_skip_reason"] == "planted_sector_failure"
    assert artifact_record["hrd_skip_reason"] == "planted_sector_failure"
    assert (
        artifact_record["divergence_proxies_skip_reason"]
        == "planted_sector_failure"
    )
    assert (
        artifact_record["sharing_locality_skip_reason"]
        == "planted_sector_failure"
    )
    assert (
        artifact_record["order_projections_skip_reason"]
        == "planted_sector_failure"
    )


def test_run_suite_refuses_multi_launch_before_sector_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = SimpleNamespace(launch_count=2)
    sources = SimpleNamespace(
        admission=admission,
        verify_unchanged=lambda: None,
    )

    def forbidden_sector_generation(*_args, **_kwargs):
        raise AssertionError("multi-launch batch reached sector publication")

    monkeypatch.setattr(
        run_suite,
        "_ensure_sector_stream_jsonl",
        forbidden_sector_generation,
    )
    with pytest.raises(TraceAdmissionError) as caught:
        run_suite._emit_admitted_trace_features(
            artifact_record={},
            artifact_dir=tmp_path,
            traces_folder=tmp_path / "trace",
            trace_admission=admission,
            slug="fixture",
            shape_point={"name": "trace_00"},
            binary_sha256="0" * 64,
            kernel_signature_sha256="0" * 64,
            props=SimpleNamespace(),
            trace_sources=sources,
        )
    assert caught.value.typed_reason == "trace_launch_count_unsupported"


@pytest.mark.parametrize(
    "failure",
    (
        TraceAdmissionError("trace_snapshot_changed", "planted schema failure"),
        ImmutableArtifactError("planted schema immutable failure"),
        ImmutableArtifactLimitError("planted schema limit failure"),
        ImmutableArtifactResourceError("planted schema resource failure"),
    ),
)
def test_run_suite_sharing_validation_propagates_identical_hard_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    sharing_path = Path(__file__).resolve().parents[1] / "sharing_locality"
    monkeypatch.syspath_prepend(str(sharing_path))
    short_classifier = importlib.import_module("classifier")
    monkeypatch.setattr(
        short_classifier,
        "compute_sharing_locality_from_admitted_sources",
        lambda **_kwargs: {"schema_version": 1},
    )

    def fail_read(_path: Path, **_kwargs):
        raise failure

    monkeypatch.setattr(run_suite, "read_regular", fail_read)
    props = SimpleNamespace(
        device_name="fixture GPU",
        compute_capability_major=8,
        compute_capability_minor=6,
        sm_count=1,
    )
    with pytest.raises(type(failure)) as caught:
        run_suite._emit_sharing_locality_for_run_summary(
            artifact_record={},
            artifact_dir=tmp_path,
            traces_folder=tmp_path / "trace",
            slug="fixture",
            shape_point={"name": "trace_00"},
            binary_sha256="0" * 64,
            kernel_signature_sha256="0" * 64,
            props=props,
            trace_sources=SimpleNamespace(),
        )
    assert caught.value is failure


@pytest.mark.parametrize(
    ("helper_name", "import_name"),
    (
        ("_emit_sharing_locality_for_run_summary", "classifier"),
        ("_emit_order_projections_for_run_summary", "projector"),
    ),
)
def test_run_suite_feature_import_propagates_hard_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    import_name: str,
) -> None:
    failure = ImmutableArtifactResourceError("planted producer import failure")
    original_import = builtins.__import__

    def fail_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == import_name:
            raise failure
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    props = SimpleNamespace(
        device_name="fixture GPU",
        compute_capability_major=8,
        compute_capability_minor=6,
        sm_count=1,
        l2_cache_size_bytes=6 * 1024 * 1024,
    )
    helper = getattr(run_suite, helper_name)
    artifact_record: dict[str, object] = {}
    with pytest.raises(ImmutableArtifactResourceError) as caught:
        helper(
            artifact_record=artifact_record,
            artifact_dir=tmp_path,
            traces_folder=tmp_path / "trace",
            slug="fixture",
            shape_point={"name": "trace_00"},
            binary_sha256="0" * 64,
            kernel_signature_sha256="0" * 64,
            props=props,
            trace_sources=SimpleNamespace(),
        )
    assert caught.value is failure
    assert not any(key.endswith("skip_reason") for key in artifact_record)


@pytest.mark.parametrize(
    "failure",
    (
        TraceAdmissionError("trace_snapshot_changed", "planted regeneration failure"),
        ImmutableArtifactError("planted regeneration immutable failure"),
        ImmutableArtifactLimitError("planted regeneration limit failure"),
        ImmutableArtifactResourceError("planted regeneration resource failure"),
    ),
)
def test_per_cell_regeneration_hard_failure_stops_later_producers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    trace_dir = tmp_path / "trace"
    admission = SimpleNamespace(launch_count=1)
    source_verify_calls: list[None] = []
    sources = SimpleNamespace(
        admission=admission,
        verify_unchanged=lambda: source_verify_calls.append(None),
    )
    sector_verify_calls: list[object] = []
    sector = SimpleNamespace(
        schema_admits_all=True,
        schema_first_failure=None,
        jsonl_path=trace_dir / "sector_stream" / "sector_stream.jsonl",
        verify_for=lambda observed: sector_verify_calls.append(observed),
    )
    counts = {"admit": 0, "open": 0, "sector": 0, "closed": 0}

    def fake_admit(observed_trace_dir: Path):
        assert observed_trace_dir == trace_dir
        counts["admit"] += 1
        return admission

    @contextmanager
    def fake_open(observed_trace_dir: Path, *, expected_admission):
        assert observed_trace_dir == trace_dir
        assert expected_admission is admission
        counts["open"] += 1
        try:
            yield sources
        finally:
            counts["closed"] += 1

    def fake_sector(**kwargs):
        assert kwargs["sources"] is sources
        assert kwargs["output_root"] == trace_dir
        assert kwargs["write_jsonl"] is True
        counts["sector"] += 1
        return sector

    monkeypatch.setattr(
        regenerate_cell,
        "label_mpi_for_cell",
        lambda **_kwargs: dict(_MANIFEST_PAIRING_INDEX),
    )
    monkeypatch.setattr(
        regenerate_cell,
        "_resolve_trace_dir",
        lambda **_kwargs: trace_dir,
    )
    monkeypatch.setattr(regenerate_cell, "require_trace_admission", fake_admit)
    monkeypatch.setattr(regenerate_cell, "open_admitted_trace_sources", fake_open)
    monkeypatch.setattr(
        regenerate_cell,
        "consume_admitted_trace_sources",
        fake_sector,
    )
    sharing_module = importlib.import_module("tools.sharing_locality.classifier")

    def fail_sharing(**_kwargs):
        raise failure

    monkeypatch.setattr(
        sharing_module,
        "compute_sharing_locality_from_admitted_sources",
        fail_sharing,
    )
    later_calls: list[str] = []
    for name in (
        "_regenerate_hrd_profile_with_label_mpi",
        "_write_sector_stream_mpi_sibling",
        "_regenerate_divergence_proxies_with_label_mpi",
        "_regenerate_order_projections_with_label_mpi",
    ):
        monkeypatch.setattr(
            regenerate_cell,
            name,
            lambda *_args, _name=name, **_kwargs: later_calls.append(_name),
        )

    with pytest.raises(type(failure)) as caught:
        regenerate_cell.regenerate_sidecars_for_cell(
            arch="sm_86",
            slug="fixture",
            shape_point="trace_00",
        )
    assert caught.value is failure
    assert counts == {"admit": 1, "open": 1, "sector": 1, "closed": 1}
    assert sector_verify_calls == [sources]
    assert source_verify_calls == []
    assert later_calls == []


def test_bulk_regeneration_groups_cell_and_runs_each_capability_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = tmp_path / "trace"
    report_path = tmp_path / "report.json"
    actions = (
        "REGENERATE_SHARING_LOCALITY_FROM_TRACE",
        "REGENERATE_HRD_PROFILE_FROM_TRACE",
        "REGENERATE_SECTOR_STREAM_FROM_TRACE",
        "REGENERATE_SECTOR_STREAM_FROM_TRACE",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "rows": [
                {
                    "arch": "sm_86",
                    "slug": "fixture",
                    "shape_point": "trace_00",
                    "suggested_operator_action": action,
                }
                for action in actions
            ]
        }),
        encoding="utf-8",
    )
    admission = SimpleNamespace(launch_count=1)
    source_verify_calls: list[None] = []
    sources = SimpleNamespace(
        admission=admission,
        verify_unchanged=lambda: source_verify_calls.append(None),
    )
    sector_verify_calls: list[object] = []
    sector = SimpleNamespace(
        schema_admits_all=True,
        schema_first_failure=None,
        jsonl_path=trace_dir / "sector_stream" / "sector_stream.jsonl",
        verify_for=lambda observed: sector_verify_calls.append(observed),
    )
    counts = {"admit": 0, "open": 0, "sector": 0, "closed": 0}

    def fake_admit(_trace_dir: Path):
        counts["admit"] += 1
        return admission

    @contextmanager
    def fake_open(_trace_dir: Path, *, expected_admission):
        assert expected_admission is admission
        counts["open"] += 1
        try:
            yield sources
        finally:
            counts["closed"] += 1

    def fake_sector(**kwargs):
        assert kwargs["sources"] is sources
        assert kwargs["output_root"] == trace_dir
        assert kwargs["write_jsonl"] is True
        counts["sector"] += 1
        return sector

    producer_calls: list[str] = []

    def producer_result(label: str):
        def invoke(**_kwargs):
            producer_calls.append(label)
            return {
                "status": "written",
                "typed_failure_reason": None,
                "output_path": str(trace_dir / f"{label}.json"),
                "producer_invoked": label,
            }

        return invoke

    monkeypatch.setattr(
        regenerate_bulk,
        "_resolve_trace_dir",
        lambda **_kwargs: trace_dir,
    )
    monkeypatch.setattr(
        regenerate_bulk,
        "_resolve_real_mpi_from_admitted_bundle",
        lambda **_kwargs: dict(_MANIFEST_PAIRING_INDEX),
    )
    monkeypatch.setattr(regenerate_bulk, "require_trace_admission", fake_admit)
    monkeypatch.setattr(regenerate_bulk, "open_admitted_trace_sources", fake_open)
    monkeypatch.setattr(
        regenerate_bulk,
        "consume_admitted_trace_sources",
        fake_sector,
    )
    monkeypatch.setattr(
        regenerate_bulk,
        "_regenerate_sharing_locality",
        producer_result("sharing"),
    )
    monkeypatch.setattr(
        regenerate_bulk,
        "_regenerate_hrd_profile",
        producer_result("hrd"),
    )
    monkeypatch.setattr(
        regenerate_bulk,
        "_regenerate_sector_stream",
        producer_result("sector"),
    )

    report = regenerate_bulk.regenerate_manifest_regenerate_rows(
        manifest_path=manifest_path,
        report_path=report_path,
        dry_run=False,
    )
    assert counts == {"admit": 1, "open": 1, "sector": 1, "closed": 1}
    assert sector_verify_calls == [sources]
    assert source_verify_calls == [None]
    assert producer_calls == ["sharing", "hrd", "sector"]
    assert report["counts"]["total_regenerate_rows"] == 4
    assert report["counts"]["written"] == 4
    assert len(report["per_cell_reports"]) == 4
    assert report_path.is_file()


def test_bulk_regeneration_hard_failure_stops_later_producers_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = tmp_path / "trace"
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "rows": [
                {
                    "arch": "sm_86",
                    "slug": "fixture",
                    "shape_point": "trace_00",
                    "suggested_operator_action": action,
                }
                for action in (
                    "REGENERATE_SHARING_LOCALITY_FROM_TRACE",
                    "REGENERATE_HRD_PROFILE_FROM_TRACE",
                    "REGENERATE_SECTOR_STREAM_FROM_TRACE",
                )
            ]
        }),
        encoding="utf-8",
    )
    admission = SimpleNamespace(launch_count=1)
    sources = SimpleNamespace(
        admission=admission,
        verify_unchanged=lambda: None,
    )
    sector = SimpleNamespace(
        schema_admits_all=True,
        schema_first_failure=None,
        jsonl_path=trace_dir / "sector_stream" / "sector_stream.jsonl",
        verify_for=lambda _sources: None,
    )
    failure = ImmutableArtifactResourceError("planted bulk resource failure")

    @contextmanager
    def fake_open(_trace_dir: Path, *, expected_admission):
        assert expected_admission is admission
        yield sources

    monkeypatch.setattr(
        regenerate_bulk,
        "_resolve_trace_dir",
        lambda **_kwargs: trace_dir,
    )
    monkeypatch.setattr(
        regenerate_bulk,
        "_resolve_real_mpi_from_admitted_bundle",
        lambda **_kwargs: dict(_MANIFEST_PAIRING_INDEX),
    )
    monkeypatch.setattr(
        regenerate_bulk,
        "require_trace_admission",
        lambda _trace_dir: admission,
    )
    monkeypatch.setattr(regenerate_bulk, "open_admitted_trace_sources", fake_open)
    monkeypatch.setattr(
        regenerate_bulk,
        "consume_admitted_trace_sources",
        lambda **_kwargs: sector,
    )
    sharing_module = importlib.import_module("tools.sharing_locality.classifier")

    def fail_sharing(**_kwargs):
        raise failure

    monkeypatch.setattr(
        sharing_module,
        "compute_sharing_locality_from_admitted_sources",
        fail_sharing,
    )
    later_calls: list[str] = []
    for name in ("_regenerate_hrd_profile", "_regenerate_sector_stream"):
        monkeypatch.setattr(
            regenerate_bulk,
            name,
            lambda _name=name, **_kwargs: later_calls.append(_name),
        )

    with pytest.raises(ImmutableArtifactResourceError) as caught:
        regenerate_bulk.regenerate_manifest_regenerate_rows(
            manifest_path=manifest_path,
            report_path=report_path,
            dry_run=False,
        )
    assert caught.value is failure
    assert later_calls == []
    assert not report_path.exists()


def test_headline_replaces_stale_jsonl_and_cache_uses_bound_generated_bytes(
    tmp_path: Path,
) -> None:
    from tools.local_labels.run_headline_cohort_predict_compare import (
        _ensure_sector_stream,
    )

    trace_dir = _write_memory_trace(tmp_path / "trace")
    sector_path = trace_dir / "sector_stream" / "sector_stream.jsonl"
    sector_path.parent.mkdir()
    stale = b"forged stale sector stream\n"
    sector_path.write_bytes(stale)
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector = _ensure_sector_stream(trace_dir, sources=sources)
        generated = sector.jsonl_bytes()
        assert generated != stale
        assert sector_path.read_bytes() == generated

        second_forgery = b"forged after regeneration\n"
        sector_path.write_bytes(second_forgery)
        bundle = cache_predictor.predict_cache_hit_rate_from_admitted_sources(
            sources=sources,
            sector_stream=sector,
            trace_dir=trace_dir,
            sector_stream_path=sector_path,
            cache_model_capacity_bytes=6 * 1024 * 1024,
            cache_model_num_l2_slices=1,
            manifest_pairing_index=_MANIFEST_PAIRING_INDEX,
            device="sm_86",
        )
    digest = bundle["input_sidecar_digests"]["sector_stream"]["sha256"]
    assert digest == hashlib.sha256(generated).hexdigest()
    assert digest != hashlib.sha256(second_forgery).hexdigest()


def test_live_disabled_sm120_helper_replaces_existing_sector_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.local_labels import sm120_activation_bundle_producer as sm120

    trace_dir = _write_memory_trace(tmp_path / "trace")
    output_path = tmp_path / "derived" / "sector_stream.jsonl"
    output_path.parent.mkdir()
    stale = b"forged stale sm120 sector stream\n"
    output_path.write_bytes(stale)
    monkeypatch.setattr(
        sm120,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector, observed_path = sm120._generate_sector_stream(
            sources=sources,
            output_path=output_path,
        )
        sector.verify_for(sources)
    assert isinstance(sector, AdmittedSectorStream)
    assert observed_path == output_path
    assert output_path.read_bytes() == sector.jsonl_bytes()
    assert output_path.read_bytes() != stale
