from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.cache_predictor import l1_production_helper
from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings
from tools.nvbit_tracer.protobuf_wire_preflight import ProtobufWireMalformed


def _trace_and_kernel(
    *,
    contract_id: str = "sm86_addr_carry_shmem_split_v2",
    capabilities: tuple[int, ...] = (1, 2, 3, 6, 8),
    combined: int = 6144,
    static: int | None = 2048,
    dynamic: int | None = 4096,
):
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    trace = trace_pb2.Trace()
    trace.trace_contract_id = contract_id
    trace.memory_payload_classes.extend(capabilities)
    kernel = trace.gpu_device[0].streams[0].kernels.add()
    kernel.size_shared_memory = combined
    if static is not None:
        kernel.shared_memory_static_bytes = static
    if dynamic is not None:
        kernel.shared_memory_dynamic_bytes = dynamic
    return trace, kernel


def test_authenticated_shared_memory_split_uses_trace_components() -> None:
    trace, kernel = _trace_and_kernel()

    assert l1_production_helper._authenticated_shared_memory_split(
        trace, kernel, None
    ) == (2048, 4096)
    assert l1_production_helper._authenticated_shared_memory_split(
        trace, kernel, 4096
    ) == (2048, 4096)


@pytest.mark.parametrize(
    ("mutation", "caller_dynamic", "detail"),
    [
        (
            {"contract_id": "sm86_addr_carry_v1"},
            None,
            "split is unavailable",
        ),
        ({"static": None}, None, "both shared-memory components"),
        ({"dynamic": None}, None, "both shared-memory components"),
        ({"combined": 6145}, None, "disagree with the combined field"),
        ({}, 1, "caller dynamic shared-memory value disagrees"),
    ],
)
def test_authenticated_shared_memory_split_fails_closed(
    mutation: dict[str, object],
    caller_dynamic: int | None,
    detail: str,
) -> None:
    trace, kernel = _trace_and_kernel(**mutation)

    with pytest.raises(
        l1_production_helper.L1ProductionHelperSharedMemorySplitUnavailable,
        match=detail,
    ):
        l1_production_helper._authenticated_shared_memory_split(
            trace, kernel, caller_dynamic
        )


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        encoded.append(byte)
        if not value:
            return bytes(encoded)


def _message_field(number: int, payload: bytes) -> bytes:
    return bytes((number << 3 | 2,)) + _varint(len(payload)) + payload


def _trace_with_kernel_suffix(trace, suffix: bytes) -> bytes:
    metadata = type(trace)()
    metadata.CopyFrom(trace)
    metadata.ClearField("gpu_device")
    kernel = trace.gpu_device[0].streams[0].kernels[0].SerializeToString() + suffix
    stream = _message_field(2, kernel)
    stream_entry = _message_field(2, stream)
    device = _message_field(2, stream_entry)
    device_entry = _message_field(2, device)
    return metadata.SerializeToString() + _message_field(6, device_entry)


def _public_trace_bytes(mutation: str) -> bytes:
    trace, kernel = _trace_and_kernel()
    kernel.id = 7
    kernel.name = "fixture_kernel___0"
    kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
    kernel.block_dim.x = 32
    kernel.block_dim.y = kernel.block_dim.z = 1
    if mutation == "valid":
        return trace.SerializeToString()
    if mutation == "missing_dynamic":
        kernel.ClearField("shared_memory_dynamic_bytes")
        return trace.SerializeToString()
    if mutation == "sum_mismatch":
        kernel.size_shared_memory = 6145
        return trace.SerializeToString()
    if mutation == "duplicate_static":
        return _trace_with_kernel_suffix(
            trace,
            _varint(11 << 3) + _varint(2048),
        )
    if mutation == "wrong_static_wire_type":
        return _trace_with_kernel_suffix(trace, _message_field(11, b"\x00"))
    raise AssertionError(f"unknown mutation {mutation!r}")


def _predict_public_trace(
    trace_pb_path: Path,
    *,
    classifier_manifest: dict | None = None,
    retained_trace_pb_payload: bytes | None = None,
) -> dict:
    return l1_production_helper.predict_l1_from_trace_dir(
        trace_pb_path=trace_pb_path,
        threadblock_pb_paths=[],
        kernel_id=7,
        classifier_manifest=(
            {} if classifier_manifest is None else classifier_manifest
        ),
        sm_count=1,
        l1tex_capacity_source="fixture",
        l1_shared_carveout_source="fixture",
        arch_token="sm_86",
        manifest_pairing_index={},
        input_sidecar_digests={},
        device="sm_86",
        cache_model_capacity_bytes=128 * 1024,
        retained_trace_pb_payload=retained_trace_pb_payload,
        retained_threadblock_pb_payloads=(
            [] if retained_trace_pb_payload is not None else None
        ),
    )


@pytest.mark.parametrize(
    ("mutation", "exception", "detail"),
    (
        (
            "missing_dynamic",
            l1_production_helper.L1ProductionHelperSharedMemorySplitUnavailable,
            "both shared-memory components",
        ),
        (
            "sum_mismatch",
            l1_production_helper.L1ProductionHelperSharedMemorySplitUnavailable,
            "disagree with the combined field",
        ),
        (
            "duplicate_static",
            ProtobufWireMalformed,
            "duplicate shared-memory split protobuf field",
        ),
        (
            "wrong_static_wire_type",
            ProtobufWireMalformed,
            "known protobuf field has incompatible wire type",
        ),
    ),
)
def test_public_l1_path_rejects_malformed_shared_memory_split_wire(
    tmp_path: Path,
    mutation: str,
    exception: type[Exception],
    detail: str,
) -> None:
    trace_pb_path = tmp_path / "dynamic_trace.pb"
    trace_pb_path.write_bytes(_public_trace_bytes(mutation))

    with pytest.raises(exception, match=detail):
        _predict_public_trace(trace_pb_path)


_NO_LOAD_MANIFEST = {
    "sass_provenance_status": "matched_no_load_class_instructions",
}


@pytest.mark.parametrize(
    ("mutation", "exception", "detail"),
    (
        (
            "duplicate_static",
            ProtobufWireMalformed,
            "duplicate shared-memory split protobuf field",
        ),
        (
            "missing_dynamic",
            l1_production_helper.L1ProductionHelperSharedMemorySplitUnavailable,
            "both shared-memory components",
        ),
    ),
)
def test_no_load_shortcut_still_rejects_malformed_root_and_split(
    tmp_path: Path,
    mutation: str,
    exception: type[Exception],
    detail: str,
) -> None:
    trace_pb_path = tmp_path / "dynamic_trace.pb"
    trace_pb_path.write_bytes(_public_trace_bytes(mutation))

    with pytest.raises(exception, match=detail):
        _predict_public_trace(
            trace_pb_path,
            classifier_manifest=_NO_LOAD_MANIFEST,
        )


def test_no_load_shortcut_preflights_exact_retained_root_bytes(
    tmp_path: Path,
) -> None:
    trace_pb_path = tmp_path / "dynamic_trace.pb"
    trace_pb_path.write_bytes(_public_trace_bytes("valid"))

    with pytest.raises(
        ProtobufWireMalformed,
        match="duplicate shared-memory split protobuf field",
    ):
        _predict_public_trace(
            trace_pb_path,
            classifier_manifest=_NO_LOAD_MANIFEST,
            retained_trace_pb_payload=_public_trace_bytes("duplicate_static"),
        )


def test_valid_no_load_shortcut_returns_after_contract_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_pb_path = tmp_path / "dynamic_trace.pb"
    trace_pb_path.write_bytes(_public_trace_bytes("valid"))
    calls: list[dict] = []
    expected = {
        "result_kind": "typed_unavailable",
        "typed_reason": "predictor_l1_no_modeled_l1_eligible_accesses",
    }

    def predict_no_load(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        l1_production_helper,
        "_load_cache_predictor",
        lambda: SimpleNamespace(
            predict_l1_hit_rate_from_manifest=predict_no_load,
        ),
    )

    assert _predict_public_trace(
        trace_pb_path,
        classifier_manifest=_NO_LOAD_MANIFEST,
    ) is expected
    assert len(calls) == 1
    assert calls[0]["cache_op_manifest"] == _NO_LOAD_MANIFEST


@pytest.mark.parametrize(
    "capabilities",
    (
        (),
        (1, 2, 3, 6),
        (1, 1, 2, 3, 6, 8),
        (1, 2, 3, 6, 8, 9),
    ),
)
def test_no_load_shortcut_rejects_noncanonical_trace_capabilities(
    tmp_path: Path,
    capabilities: tuple[int, ...],
) -> None:
    trace, kernel = _trace_and_kernel(capabilities=capabilities)
    kernel.id = 7
    kernel.name = "fixture_kernel___0"
    kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
    kernel.block_dim.x = 32
    kernel.block_dim.y = kernel.block_dim.z = 1
    trace_pb_path = tmp_path / "dynamic_trace.pb"
    trace_pb_path.write_bytes(trace.SerializeToString())

    with pytest.raises(
        l1_production_helper.L1ProductionHelperTraceCapabilitiesInvalid,
        match="trace capabilities invalid",
    ):
        _predict_public_trace(
            trace_pb_path,
            classifier_manifest=_NO_LOAD_MANIFEST,
        )


@pytest.mark.parametrize(
    "contract_id",
    ("sm86_addr_carry_v1", "unknown_future_contract"),
)
@pytest.mark.parametrize("early_return", ("missing_kernel", "missing_geometry"))
def test_exact_contract_rejected_before_kernel_or_geometry_unavailable(
    tmp_path: Path,
    contract_id: str,
    early_return: str,
) -> None:
    trace, kernel = _trace_and_kernel(contract_id=contract_id)
    kernel.name = "fixture_kernel___0"
    if early_return == "missing_kernel":
        kernel.id = 8
        kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
        kernel.block_dim.x = 32
        kernel.block_dim.y = kernel.block_dim.z = 1
    else:
        kernel.id = 7
    trace_pb_path = tmp_path / "dynamic_trace.pb"
    trace_pb_path.write_bytes(trace.SerializeToString())

    with pytest.raises(
        l1_production_helper.L1ProductionHelperSharedMemorySplitUnavailable,
        match="unsupported trace_contract_id",
    ):
        _predict_public_trace(trace_pb_path)


def test_threadblock_wire_preflight_runs_before_each_decode(
    tmp_path: Path,
) -> None:
    good_path = tmp_path / "good.pb"
    bad_path = tmp_path / "bad.pb"
    good_payload = b"\x0a\x00"
    good_path.write_bytes(good_payload)
    bad_path.write_bytes(b"\x0a\x00\x08\x00")
    decoded: list[bytes] = []

    class Threadblock:
        block_id = SimpleNamespace(x=0, y=0, z=0)
        warps: dict[int, object] = {}

        @staticmethod
        def HasField(name: str) -> bool:
            return name == "block_id"

        @staticmethod
        def ParseFromString(raw: bytes) -> None:
            decoded.append(raw)

    threadblock_pb2 = SimpleNamespace(threadblock=Threadblock)
    with pytest.raises(
        ProtobufWireMalformed,
        match="known protobuf field has incompatible wire type",
    ):
        l1_production_helper._walk_threadblock_protos(
            threadblock_pb2,
            [good_path, bad_path],
            SimpleNamespace(),
            grid_x=1,
            grid_y=1,
            grid_dim=1,
        )

    assert decoded == [good_payload]


def test_public_l1_path_validates_later_split_kernel(
    tmp_path: Path,
) -> None:
    trace, first = _trace_and_kernel()
    first.id = 7
    first.name = "selected_kernel___0"
    first.grid_dim.x = first.grid_dim.y = first.grid_dim.z = 1
    first.block_dim.x = 32
    first.block_dim.y = first.block_dim.z = 1
    later = trace.gpu_device[0].streams[0].kernels.add()
    later.CopyFrom(first)
    later.id = 8
    later.ClearField("shared_memory_dynamic_bytes")
    trace_pb_path = tmp_path / "dynamic_trace.pb"
    trace_pb_path.write_bytes(trace.SerializeToString())

    with pytest.raises(
        l1_production_helper.L1ProductionHelperSharedMemorySplitUnavailable,
        match="both shared-memory components",
    ):
        _predict_public_trace(trace_pb_path)
