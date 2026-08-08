from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.local_labels import extend_launch_parameters as launch_parameters
from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings


def _write_trace(
    path: Path,
    *,
    contract_id: str = "sm86_addr_carry_shmem_split_v2",
    memory_payload_classes: tuple[int, ...] = (
        launch_parameters.EXPECTED_MEMORY_PAYLOAD_CLASSES
    ),
    combined: int = 6144,
    static: int | None = 2048,
    dynamic: int | None = 4096,
) -> Path:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    trace = trace_pb2.Trace()
    trace.trace_contract_id = contract_id
    trace.memory_payload_classes.extend(memory_payload_classes)
    device = trace.gpu_device[0]
    device.id = 0
    stream = device.streams[0]
    stream.id = 0
    kernel = stream.kernels.add()
    kernel.id = 7
    kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
    kernel.block_dim.x = 32
    kernel.block_dim.y = kernel.block_dim.z = 1
    kernel.size_shared_memory = combined
    if static is not None:
        kernel.shared_memory_static_bytes = static
    if dynamic is not None:
        kernel.shared_memory_dynamic_bytes = dynamic
    path.write_bytes(trace.SerializeToString())
    return path


class _DecodeMemoryTrace:
    def ParseFromString(self, _raw: bytes) -> None:
        raise MemoryError("synthetic protobuf decode allocation failure")


class _DecodeMemoryTraceModule:
    Trace = _DecodeMemoryTrace


def _cli_fixture(tmp_path: Path) -> tuple[object, list[str]]:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    label_admitted_path = tmp_path / "label_admitted.json"
    label_admitted_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "arch": "sm_86",
                        "slug": "fixture",
                        "shape_point": "tiny",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    trace_path = (
        tmp_path
        / "workloads/simbench_bw/results"
        / "fixture/rtx_3090_ti/sm_86/trace_tiny/dynamic_trace.pb"
    )
    trace_path.parent.mkdir(parents=True)
    _write_trace(trace_path)
    argv = [
        "--label-admitted",
        str(label_admitted_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--arch-dir",
        "rtx_3090_ti/sm_86",
        "--repo-root",
        str(tmp_path),
    ]
    return trace_pb2, argv


def test_extract_geometry_uses_authenticated_static_component(tmp_path: Path) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb")

    geometry = launch_parameters._extract_first_kernel_geometry(path, trace_pb2)

    assert geometry.kernel_id == 7
    assert geometry.static_shared_bytes == 2048


def test_extract_geometry_preserves_zero_static_for_dynamic_workload(
    tmp_path: Path,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(
        tmp_path / "dynamic_trace.pb",
        combined=4096,
        static=0,
        dynamic=4096,
    )

    geometry = launch_parameters._extract_first_kernel_geometry(path, trace_pb2)

    assert geometry.static_shared_bytes == 0


def test_extract_geometry_rejects_malformed_wire(tmp_path: Path) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = tmp_path / "dynamic_trace.pb"
    path.write_bytes(b"\x08")

    with pytest.raises(launch_parameters.LaunchParametersTraceMalformed):
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)


def test_extract_geometry_types_decoded_resource_ceiling(
    tmp_path: Path, monkeypatch,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb")

    def fail_preflight(_raw: bytes) -> None:
        raise launch_parameters.ProtobufDecodedResourceLimitExceeded(
            "kernels", 1_000_001, 1_000_000
        )

    monkeypatch.setattr(
        launch_parameters, "preflight_trace_protobuf", fail_preflight
    )

    with pytest.raises(
        launch_parameters.LaunchParametersTraceLimitExceeded,
        match="decoded resource kernels",
    ) as captured:
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)

    assert isinstance(
        captured.value.__cause__,
        launch_parameters.ProtobufDecodedResourceLimitExceeded,
    )


@pytest.mark.parametrize(
    ("stage", "detail"),
    [
        ("preflight", "cannot preflight dynamic_trace.pb"),
        ("decode", "cannot decode dynamic_trace.pb"),
    ],
)
def test_extract_geometry_types_stage_memory_failure(
    tmp_path: Path, monkeypatch, stage: str, detail: str,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb")
    trace_module = trace_pb2

    if stage == "preflight":
        def fail_preflight(_raw: bytes) -> None:
            raise MemoryError("synthetic preflight allocation failure")

        monkeypatch.setattr(
            launch_parameters, "preflight_trace_protobuf", fail_preflight
        )
    else:
        trace_module = _DecodeMemoryTraceModule

    with pytest.raises(
        launch_parameters.LaunchParametersTraceResourceUnavailable,
        match=detail,
    ) as captured:
        launch_parameters._extract_first_kernel_geometry(path, trace_module)

    assert isinstance(captured.value.__cause__, MemoryError)


def test_extract_geometry_rejects_trace_above_byte_ceiling(
    tmp_path: Path, monkeypatch,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb")
    monkeypatch.setattr(
        launch_parameters, "_TRACE_ROOT_MAX_BYTES", path.stat().st_size - 1
    )

    with pytest.raises(launch_parameters.LaunchParametersTraceLimitExceeded):
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)


def test_extract_geometry_accepts_trace_at_byte_ceiling(
    tmp_path: Path, monkeypatch,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb")
    monkeypatch.setattr(
        launch_parameters, "_TRACE_ROOT_MAX_BYTES", path.stat().st_size
    )

    geometry = launch_parameters._extract_first_kernel_geometry(path, trace_pb2)

    assert geometry.kernel_id == 7


def test_extract_geometry_rejects_missing_trace_with_specific_type(
    tmp_path: Path, monkeypatch,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2

    def fail_preflight(_raw: bytes) -> None:
        raise MemoryError("preflight must not precede missing-path typing")

    monkeypatch.setattr(
        launch_parameters, "preflight_trace_protobuf", fail_preflight
    )

    with pytest.raises(launch_parameters.LaunchParametersTraceMissing):
        launch_parameters._extract_first_kernel_geometry(
            tmp_path / "missing.pb", trace_pb2
        )


def test_extract_geometry_types_materialization_memory_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb")

    def fail_read(*_args, **_kwargs):
        raise MemoryError("synthetic allocation failure")

    monkeypatch.setattr(
        launch_parameters.immutable.RetainedRegular,
        "read_bytes",
        fail_read,
    )

    with pytest.raises(
        launch_parameters.LaunchParametersTraceResourceUnavailable
    ):
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)


@pytest.mark.parametrize(
    "error_type",
    [
        launch_parameters.LaunchParametersTraceLimitExceeded,
        launch_parameters.LaunchParametersTraceResourceUnavailable,
        launch_parameters.LaunchParametersTraceGenerationChanged,
    ],
)
def test_main_returns_refusal_for_typed_retained_trace_failures(
    monkeypatch, capsys, error_type: type[Exception]
) -> None:
    def fail_build(**_kwargs):
        raise error_type("synthetic retained trace failure")

    monkeypatch.setattr(
        launch_parameters, "build_sidecars_for_cohort", fail_build
    )

    assert launch_parameters.main([]) == 2
    assert "synthetic retained trace failure" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("stage", "detail"),
    [
        ("decoded_ceiling", "decoded resource kernels"),
        ("preflight_memory", "cannot preflight dynamic_trace.pb"),
        ("decode_memory", "cannot decode dynamic_trace.pb"),
    ],
)
def test_main_refuses_preflight_and_decode_resource_failures(
    tmp_path: Path, monkeypatch, capsys, stage: str, detail: str,
) -> None:
    trace_pb2, argv = _cli_fixture(tmp_path)
    trace_module = trace_pb2

    if stage == "decoded_ceiling":
        def fail_preflight(_raw: bytes) -> None:
            raise launch_parameters.ProtobufDecodedResourceLimitExceeded(
                "kernels", 1_000_001, 1_000_000
            )

        monkeypatch.setattr(
            launch_parameters, "preflight_trace_protobuf", fail_preflight
        )
    elif stage == "preflight_memory":
        def fail_preflight(_raw: bytes) -> None:
            raise MemoryError("synthetic preflight allocation failure")

        monkeypatch.setattr(
            launch_parameters, "preflight_trace_protobuf", fail_preflight
        )
    else:
        trace_module = _DecodeMemoryTraceModule

    monkeypatch.setattr(
        launch_parameters, "_load_trace_bindings", lambda: trace_module
    )

    assert launch_parameters.main(argv) == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("extend_launch_parameters: ")
    assert detail in stderr
    assert not (tmp_path / "out").exists()


def test_extract_geometry_rejects_generation_swap_during_read(
    tmp_path: Path, monkeypatch,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb")
    displaced = tmp_path / "displaced.pb"
    real_read = launch_parameters.immutable.RetainedRegular.read_bytes

    def read_then_swap(retained, *args, **kwargs):
        payload = real_read(retained, *args, **kwargs)
        path.rename(displaced)
        path.write_bytes(payload)
        return payload

    monkeypatch.setattr(
        launch_parameters.immutable.RetainedRegular,
        "read_bytes",
        read_then_swap,
    )

    def fail_preflight(_raw: bytes) -> None:
        raise MemoryError("preflight must not mask a retained generation change")

    monkeypatch.setattr(
        launch_parameters, "preflight_trace_protobuf", fail_preflight
    )

    with pytest.raises(
        launch_parameters.LaunchParametersTraceGenerationChanged
    ):
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)


def test_extract_geometry_types_retained_resource_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb")

    def fail_open(*_args, **_kwargs):
        raise launch_parameters.immutable.ImmutableArtifactResourceError(
            "synthetic descriptor exhaustion"
        )

    monkeypatch.setattr(
        launch_parameters.immutable, "open_retained_regular", fail_open
    )
    with pytest.raises(
        launch_parameters.LaunchParametersTraceResourceUnavailable
    ):
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ({"static": None}, "shared_memory_static_bytes"),
        ({"dynamic": None}, "shared_memory_dynamic_bytes"),
        ({"combined": 6145}, "does not equal"),
    ],
)
def test_extract_geometry_rejects_unauthenticated_shared_memory_split(
    tmp_path: Path,
    mutation: dict[str, object],
    detail: str,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(tmp_path / "dynamic_trace.pb", **mutation)

    with pytest.raises(
        launch_parameters.LaunchParametersSharedMemorySplitUnavailable,
        match=detail,
    ):
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)


def test_legacy_geometry_rejects_nonzero_combined_shared_memory(
    tmp_path: Path,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(
        tmp_path / "dynamic_trace.pb",
        contract_id="sm86_addr_carry_v1",
        static=None,
        dynamic=None,
    )

    with pytest.raises(
        launch_parameters.LaunchParametersSharedMemorySplitUnavailable,
        match="legacy.*nonzero combined shared-memory",
    ):
        launch_parameters._extract_first_kernel_geometry(
            path, trace_pb2, historical_only=True
        )


def test_legacy_geometry_authenticates_zero_as_zero_static(tmp_path: Path) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(
        tmp_path / "dynamic_trace.pb",
        contract_id="sm86_addr_carry_v1",
        combined=0,
        static=None,
        dynamic=None,
    )

    geometry = launch_parameters._extract_first_kernel_geometry(
        path, trace_pb2, historical_only=True
    )

    assert geometry.static_shared_bytes == 0


def test_public_geometry_builder_rejects_legacy_trace(
    tmp_path: Path,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(
        tmp_path / "dynamic_trace.pb",
        contract_id="sm86_addr_carry_v1",
        combined=0,
        static=None,
        dynamic=None,
    )

    with pytest.raises(
        launch_parameters.LaunchParametersTraceContractUnsupported,
        match="historical-read-only",
    ):
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)


def test_legacy_geometry_rejects_unadvertised_split_fields(tmp_path: Path) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(
        tmp_path / "dynamic_trace.pb",
        contract_id="sm86_addr_carry_v1",
        combined=0,
        static=0,
        dynamic=0,
    )

    with pytest.raises(
        launch_parameters.LaunchParametersSharedMemorySplitUnavailable,
        match="legacy.*unauthenticated.*split fields",
    ):
        launch_parameters._extract_first_kernel_geometry(
            path, trace_pb2, historical_only=True
        )


@pytest.mark.parametrize(
    "contract_id",
    ["sm86_addr_carry_shmem_split_v20", ""],
    ids=["mistyped", "missing"],
)
def test_extract_geometry_rejects_unknown_trace_contract(
    tmp_path: Path,
    contract_id: str,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(
        tmp_path / "dynamic_trace.pb",
        contract_id=contract_id,
    )

    with pytest.raises(
        launch_parameters.LaunchParametersTraceContractUnsupported,
        match="trace_contract_id",
    ):
        launch_parameters._extract_first_kernel_geometry(path, trace_pb2)


@pytest.mark.parametrize(
    "contract_id",
    ["sm86_addr_carry_shmem_split_v2", "sm86_addr_carry_v1"],
    ids=["split-v2", "legacy-v1"],
)
@pytest.mark.parametrize(
    "memory_payload_classes",
    [
        (),
        (1, 2, 3, 6),
        (1, 2, 3, 6, 8, 4),
        (2, 1, 3, 6, 8),
        (1, 2, 3, 6, 8, 8),
        (1, 2, 3, 6, 4),
        (1, 2, 3, 6, 11),
    ],
    ids=[
        "missing",
        "incomplete",
        "extra",
        "reordered",
        "duplicate",
        "wrong",
        "unknown",
    ],
)
def test_extract_geometry_rejects_noncanonical_capability_vector(
    tmp_path: Path,
    contract_id: str,
    memory_payload_classes: tuple[int, ...],
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    path = _write_trace(
        tmp_path / "dynamic_trace.pb",
        contract_id=contract_id,
        memory_payload_classes=memory_payload_classes,
    )

    with pytest.raises(
        launch_parameters.LaunchParametersTraceCapabilitiesInvalid,
        match="memory_payload_classes",
    ):
        launch_parameters._extract_first_kernel_geometry(
            path,
            trace_pb2,
            historical_only=contract_id == "sm86_addr_carry_v1",
        )


def test_public_builder_rejects_noncanonical_split_v2_capability_vector(
    tmp_path: Path,
) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    cell = {"slug": "fixture", "shape_point": "tiny"}
    trace_path = (
        tmp_path
        / "workloads/simbench_bw/results"
        / "fixture/rtx_3090_ti/sm_86/trace_tiny/dynamic_trace.pb"
    )
    trace_path.parent.mkdir(parents=True)
    _write_trace(trace_path, memory_payload_classes=(1, 2, 3, 6, 8, 1))

    with pytest.raises(
        launch_parameters.LaunchParametersTraceCapabilitiesInvalid,
        match="memory_payload_classes",
    ):
        launch_parameters.build_launch_parameters_sidecar(
            cell=cell,
            repo_root=tmp_path,
            trace_pb2_mod=trace_pb2,
        )


def test_public_builder_validates_later_split_kernel(tmp_path: Path) -> None:
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    cell = {"slug": "fixture", "shape_point": "tiny"}
    trace_path = (
        tmp_path
        / "workloads/simbench_bw/results"
        / "fixture/rtx_3090_ti/sm_86/trace_tiny/dynamic_trace.pb"
    )
    trace_path.parent.mkdir(parents=True)
    _write_trace(trace_path)
    trace = trace_pb2.Trace()
    trace.ParseFromString(trace_path.read_bytes())
    first = trace.gpu_device[0].streams[0].kernels[0]
    later = trace.gpu_device[1].streams[0].kernels.add()
    later.CopyFrom(first)
    later.id = 8
    later.ClearField("shared_memory_dynamic_bytes")
    trace_path.write_bytes(trace.SerializeToString())

    with pytest.raises(
        launch_parameters.LaunchParametersSharedMemorySplitUnavailable,
        match="shared_memory_dynamic_bytes",
    ):
        launch_parameters.build_launch_parameters_sidecar(
            cell=cell,
            repo_root=tmp_path,
            trace_pb2_mod=trace_pb2,
        )
