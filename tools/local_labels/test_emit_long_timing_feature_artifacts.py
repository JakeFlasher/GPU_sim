"""Consumer-path tests for long-timing NVBit launch metadata."""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import emit_long_timing_feature_artifacts as timing  # noqa: E402
from tools.nvbit_tracer import trace_admission  # noqa: E402
from workloads.simbench_bw import nvbit_trace_summary_adapter as adapter  # noqa: E402


def _retained_runtime(environment: dict[str, str], *pass_fds: int):
    return nullcontext(
        SimpleNamespace(environment=environment, pass_fds=pass_fds)
    )


def _cell() -> dict:
    return {
        "slug": "fixture",
        "shape_point": "fixture_shape",
        "kernel_signature_sha256": "d" * 64,
        "regs_per_thread": 48,
        "launch_geometry": {
            "kernel_name": "fixture_kernel",
            "grid_x": 7,
            "grid_y": 2,
            "grid_z": 1,
            "block_x": 128,
            "block_y": 2,
            "block_z": 1,
            "shared_mem_static_bytes": 2048,
            "shared_mem_dynamic_bytes": 4096,
        },
    }


def _source_generation(seed: str = "a") -> dict:
    return timing.immutable.seal_model(
        {
            "schema_version": 1,
            "artifact_kind": "timing_nvbit_retained_source_generation",
            "trace_contract_id": timing._TIMING_TRACE_CONTRACT_ID,
            "trace_admission_sha256": seed * 64,
            "trace_filesystem_generation_sha256": seed * 64,
            "bounded_inventory": {
                "source_file_count": 3,
                "source_total_bytes": 123,
                "max_tree_files": 10,
                "max_tree_directories": 10,
                "max_trace_root_bytes": 1000,
                "max_stats_bytes": 1000,
                "max_threadblock_files": 10,
                "max_threadblock_bytes": 1000,
                "max_total_threadblock_bytes": 1000,
            },
        },
        field="source_generation_sha256",
    )


def _admission() -> SimpleNamespace:
    geometry = {
        "kernel_name": "fixture_kernel___0",
        "grid_x": 7,
        "grid_y": 2,
        "grid_z": 1,
        "block_x": 128,
        "block_y": 2,
        "block_z": 1,
        "shared_mem_static_bytes": 2048,
        "shared_mem_dynamic_bytes": 4096,
        "regs_per_thread": 48,
    }
    launch = SimpleNamespace(
        launch_index=0,
        device_id=0,
        stream_id=0,
        kernel_id=0,
        kernel_signature_inputs=lambda: dict(geometry),
    )
    return SimpleNamespace(
        launch_count=1,
        stats_row_count=1,
        launches=(launch,),
    )


def _stats_payload(**overrides: object) -> bytes:
    row = {
        "device_id": 0,
        "stream_id": 0,
        "kernel id": "kernel-0.trace",
        "kernel mangled name": "fixture_kernel",
        "grid_dimX": 7,
        "grid_dimY": 2,
        "grid_dimZ": 1,
        "block_dimX": 128,
        "block_dimY": 2,
        "block_dimZ": 1,
        **overrides,
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _write_stats(path: Path, **overrides: object) -> None:
    path.write_bytes(_stats_payload(**overrides))


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, SimpleNamespace, list[dict]]:
    trace_root = tmp_path / "traces"
    trace_dir = trace_root / "fixture_shape"
    trace_dir.mkdir(parents=True)
    _write_stats(trace_dir / "stats.csv")
    binary = tmp_path / "fixture.bin"
    binary.write_bytes(b"binary")
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"tracer")
    admission = _admission()
    emitted: list[dict] = []

    monkeypatch.setattr(timing, "_binary_path_for", lambda _cell: binary)
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    monkeypatch.setattr(
        trace_admission,
        "require_trace_admission",
        lambda path: admission,
    )

    def emit(**kwargs):
        emitted.append(kwargs)
        record = {
            "kernel_name": admission.launches[0].kernel_signature_inputs()[
                "kernel_name"
            ],
            "kernel_signature_sha256": "c" * 64,
            "inst_count": 99,
        }
        timing.immutable.write_regular_bytes_at(
            kwargs["output_descriptor"],
            kwargs["output_path"].name,
            (json.dumps(record) + "\n").encode("utf-8"),
        )
        return record

    monkeypatch.setattr(adapter, "emit_admitted_nvbit_trace_summary", emit)
    return trace_root, trace_dir, tracer, admission, emitted


def _write_split_trace(path: Path) -> Path:
    from tools.nvbit_tracer.test_trace_admission import _write_trace

    trace_dir = _write_trace(path)
    bindings = timing.load_dynamic_trace_bindings()
    root = bindings.trace_pb2.Trace()
    root.ParseFromString((trace_dir / "dynamic_trace.pb").read_bytes())
    root.trace_contract_id = timing._TIMING_TRACE_CONTRACT_ID
    kernel = root.gpu_device[0].streams[0].kernels[0]
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0
    (trace_dir / "dynamic_trace.pb").write_bytes(root.SerializeToString())
    return trace_dir


def _identity_cell() -> dict:
    cell = _cell()
    cell.update(
        {
            "arch": "sm_86",
            "binary_sha256": "a" * 64,
            "source_commit": "b" * 40,
        }
    )
    return cell


def _write_identity_raw_trace(trace_dir: Path, cell: dict) -> None:
    bindings = timing.load_dynamic_trace_bindings()
    trace = bindings.trace_pb2.Trace()
    trace.trace_contract_id = timing._TIMING_TRACE_CONTRACT_ID
    trace.memory_payload_classes.extend([1, 2, 3, 6, 8])
    device = trace.gpu_device[0]
    device.id = 0
    stream = device.streams[0]
    stream.id = 0
    kernel = stream.kernels.add()
    kernel.id = 0
    kernel.name = cell["launch_geometry"]["kernel_name"]
    kernel.function_unique_id = 1
    kernel.grid_dim.x = cell["launch_geometry"]["grid_x"]
    kernel.grid_dim.y = cell["launch_geometry"]["grid_y"]
    kernel.grid_dim.z = cell["launch_geometry"]["grid_z"]
    kernel.block_dim.x = cell["launch_geometry"]["block_x"]
    kernel.block_dim.y = cell["launch_geometry"]["block_y"]
    kernel.block_dim.z = cell["launch_geometry"]["block_z"]
    kernel.shared_memory_static_bytes = cell["launch_geometry"][
        "shared_mem_static_bytes"
    ]
    kernel.shared_memory_dynamic_bytes = cell["launch_geometry"][
        "shared_mem_dynamic_bytes"
    ]
    kernel.size_shared_memory = (
        kernel.shared_memory_static_bytes + kernel.shared_memory_dynamic_bytes
    )
    kernel.number_of_registers = cell["regs_per_thread"]
    (trace_dir / "dynamic_trace.pb").write_bytes(trace.SerializeToString())

    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.block_id.x = threadblock.block_id.y = threadblock.block_id.z = 0
    warp = threadblock.warps[0]
    warp.id = 0
    instruction = warp.instructions.add()
    instruction.pc = instruction.vpc = 0x1000
    instruction.active_mask = instruction.predicate_mask = 0b111
    instruction.function_unique_id = 1
    instruction.data_width = 4
    instruction.warpid_tb = 0
    instruction.memory_op_class = timing._MEMORY_OP_CLASS_GLOBAL
    instruction.memory_op_access_kind = timing._ACCESS_KIND_STORE
    operand = instruction.addresses.add()
    operand.data_width = 4
    operand.compression_format = 0
    operand.address_space = 1
    operand.allocation_id = 1
    operand.addrs.extend([-1] * 32)
    operand.addrs[0] = 0x1000
    operand.addrs[1] = 0x1020
    operand.addrs[2] = 0x1040
    threadblock_path = (
        trace_dir
        / "threadblocks"
        / "device_0"
        / "stream_0"
        / "kernel_0"
        / "0.pb"
    )
    threadblock_path.parent.mkdir(parents=True)
    threadblock_path.write_bytes(threadblock.SerializeToString())
    allocations_path = trace_dir / "extra_info" / "allocations.json"
    allocations_path.parent.mkdir()
    allocations_path.write_text(
        json.dumps(
            [
                {
                    "allocation_id": 1,
                    "context_id": 1,
                    "device_id": 0,
                    "base_address": "0x1000",
                    "size_bytes": 4096,
                    "alive": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    (trace_dir / "stats.csv").write_bytes(
        _stats_payload(total_insts=1, total_reported_insts=1)
    )


def _write_identity_capture(
    trace_root: Path,
    cell: dict,
) -> tuple[Path, dict, dict]:
    trace_dir = trace_root / cell["shape_point"]
    trace_dir.mkdir(parents=True)
    _write_identity_raw_trace(trace_dir, cell)
    source_generation = timing._snapshot_timing_trace_source(trace_dir)[
        "source_generation"
    ]
    tracer_payload = b"retained split tracer"
    stdout_payload = b""
    stderr_payload = b""
    (trace_dir / timing._TIMING_RETAINED_TRACER_NAME).write_bytes(tracer_payload)
    (trace_dir / timing._TIMING_STDOUT_NAME).write_bytes(stdout_payload)
    (trace_dir / timing._TIMING_STDERR_NAME).write_bytes(stderr_payload)
    start = timing.immutable.seal_model(
        {
            "schema_version": 1,
            "artifact_kind": "timing_nvbit_launch_start",
            "descriptor_bindings": {
                "command_0": "target_binary",
                "CUDA_INJECTION64_PATH": "tracer",
                "TRACES_FOLDER": "trace_directory",
            },
        },
        field="start_seal_sha256",
    )
    (trace_dir / timing._TIMING_START_NAME).write_text(
        json.dumps(start, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt = timing.immutable.seal_model(
        {
            "schema_version": 1,
            "artifact_kind": "timing_nvbit_launch_attempt",
            "status": "succeeded",
            "shape_point": cell["shape_point"],
            "start_seal_sha256": start["start_seal_sha256"],
            "target_binary": {"sha256": cell["binary_sha256"]},
            "tracer": {"sha256": timing.immutable.sha256_bytes(tracer_payload)},
            "trace_source_generation": source_generation,
            "streams": {
                "stdout": {
                    "path": timing._TIMING_STDOUT_NAME,
                    "sha256": timing.immutable.sha256_bytes(stdout_payload),
                    "size_bytes": len(stdout_payload),
                },
                "stderr": {
                    "path": timing._TIMING_STDERR_NAME,
                    "sha256": timing.immutable.sha256_bytes(stderr_payload),
                    "size_bytes": len(stderr_payload),
                },
            },
        },
        field="attempt_seal_sha256",
    )
    (trace_dir / timing._TIMING_ATTEMPT_NAME).write_text(
        json.dumps(attempt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    capture_ref = timing._require_timing_capture_attempt(
        trace_dir,
        shape_point=cell["shape_point"],
        binary_sha256=cell["binary_sha256"],
        tracer_sha256=timing.immutable.sha256_bytes(tracer_payload),
        expected_attempt_seal_sha256=attempt["attempt_seal_sha256"],
    )
    return trace_dir, capture_ref, source_generation


def _identity_sector_summary(
    cell: dict,
    *,
    trace_dir: Path,
    capture_ref: dict,
    source_generation: dict,
) -> dict:
    launch_geometry = {
        "kernel_name": f"{cell['launch_geometry']['kernel_name']}___0",
        **{
            field: int(cell["launch_geometry"][field])
            for field in timing._LAUNCH_GEOMETRY_FIELDS
        },
    }
    threadblock_payloads = tuple(
        (
            threadblock_path.relative_to(trace_dir).as_posix(),
            threadblock_path.read_bytes(),
        )
        for threadblock_path in sorted((trace_dir / "threadblocks").rglob("*.pb"))
    )
    threadblock_set_sha256 = timing._sha256_of_threadblock_snapshots(
        threadblock_payloads
    )
    summarize_launch = {
        "cold_count": 3,
        "finite_miss_count": 0,
        "unique_sector_count": 3,
        "working_set_sector_count": 0,
        "reuse_distance_finite_p50": None,
        "reuse_distance_finite_p90": None,
        "reuse_distance_finite_p99": None,
        "per_op_class_counts": {"1": 3},
        "per_access_kind_counts": {"2": 3},
        "load_count": 0,
        "store_count": 3,
        "atomic_count": 0,
        "async_copy_count": 0,
        "unspecified_kind_count": 0,
    }
    observation_basis = {
        "trace_contract_ok": True,
        "trace_contract_id": timing._TIMING_TRACE_CONTRACT_ID,
        "trace_source_generation": source_generation,
        "threadblock_proto_file_count": len(threadblock_payloads),
        "threadblock_proto_set_sha256": threadblock_set_sha256,
        "memory_instruction_record_count": 1,
        "observed_access_count": 3,
        "observed_hit_count": 0,
        "observed_miss_count": 3,
        "stats_total_insts": 1,
        "stats_total_reported_insts": 1,
    }
    observed_path = trace_dir / "observed_sector_summary.json"
    observed_summary = {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_observed_sector_summary",
        "arch": cell["arch"],
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "producer": timing.OBSERVED_SECTOR_PRODUCER,
        "trace_window": {
            "instruction_trace_mode": timing.OBSERVED_TRACE_MODE,
            "instr_begin": timing.TRACE_INSTR_BEGIN,
            "instr_end": timing.TRACE_INSTR_END,
            "window_role": "per_cell_observed_global_memory_window",
        },
        "observation_basis": observation_basis,
        "summarize_launch": summarize_launch,
    }
    observed_payload = (
        json.dumps(observed_summary, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    observed_path.write_bytes(observed_payload)
    sector_counts = timing._derive_sector_counts_from_observation(observed_summary)
    return {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_sector_summary",
        "arch": cell["arch"],
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "producer": "nvbit_observed_summarize_launch_sector_summary_v1",
        "deploy_legal": True,
        "source_trace": {
            "trace_dir": str(trace_dir),
            "nvbit_trace_pb_path": str(trace_dir / "dynamic_trace.pb"),
            "nvbit_trace_pb_sha256": timing.sha256_file(
                trace_dir / "dynamic_trace.pb"
            ),
            "capture_mode": (
                "per_cell_same_binary_same_shape_nvbit_bounded_observed_threadblock_trace"
            ),
            "instruction_trace_mode": timing.OBSERVED_TRACE_MODE,
            "instr_begin": timing.TRACE_INSTR_BEGIN,
            "instr_end": timing.TRACE_INSTR_END,
            "trace_contract_id": timing._TIMING_TRACE_CONTRACT_ID,
            "trace_source_generation": source_generation,
            "threadblock_proto_file_count": len(threadblock_payloads),
            "threadblock_proto_set_sha256": threadblock_set_sha256,
            "threadblock_proto_retention": (
                "retained_raw_threadblock_protos_after_digest"
            ),
            "nvbit_launch_attempt": capture_ref,
        },
        "nvbit_launch_summary": {
            "path": str(trace_dir / "nvbit_launch_summary.json"),
            "sha256": "b" * 64,
            "raw_kernel_name": launch_geometry["kernel_name"],
            "raw_kernel_signature_sha256": "c" * 64,
            "harmonized_kernel_name": cell["launch_geometry"]["kernel_name"],
            "harmonized_kernel_signature_sha256": cell[
                "kernel_signature_sha256"
            ],
            "inst_count": 1,
            "launch_index": 0,
            "repeat_index": 0,
            "stats_csv_path": str(trace_dir / "stats.csv"),
            "stats_csv_sha256": timing.sha256_file(trace_dir / "stats.csv"),
            "trace_contract_id": timing._TIMING_TRACE_CONTRACT_ID,
            "trace_source_generation": source_generation,
            "instruction_trace_mode": timing.OBSERVED_TRACE_MODE,
            "launch_geometry": launch_geometry,
        },
        "observed_sector_summary": {
            "path": str(observed_path),
            "sha256": timing.immutable.sha256_bytes(observed_payload),
            "artifact_kind": "timing_nvbit_observed_sector_summary",
        },
        "summarize_launch": summarize_launch,
        "observation_basis": observation_basis,
        "static_inputs": {
            "static_sass_features_path": "static-features.json",
            "static_sass_features_sha256": "f" * 64,
            "static_work_path": "static-work.json",
            "static_work_sha256": "0" * 64,
        },
        "sector_counts": sector_counts,
        "sector_count_derivation": {
            "source": "nvbit_observed_threadblock_proto_sector_stream",
            "observed_access_count": observation_basis["observed_access_count"],
            "observed_sector_summary_path": str(observed_path),
            "instruction_trace_mode": timing.OBSERVED_TRACE_MODE,
        },
        "theoretical_occupancy": {
            "active_blocks_per_sm": 2,
            "active_warps_per_sm": 16,
            "warps_per_block": 8,
            "source": "test_launch_geometry",
        },
    }


def test_launch_summary_uses_authenticated_geometry_and_same_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trace_root, trace_dir, tracer, admission, emitted = _prepare(
        tmp_path,
        monkeypatch,
    )

    summary = timing._emit_nvbit_launch_summary(
        cell=_cell(),
        trace_dir=trace_dir,
        admission=admission,
        stats_payload=(trace_dir / "stats.csv").read_bytes(),
        binary_sha256="a" * 64,
        tracer_build_sha256=timing.sha256_file(tracer),
        source_generation=_source_generation(),
    )

    assert len(emitted) == 1
    assert emitted[0]["admission"] is admission
    assert emitted[0]["binary_sha256"] == "a" * 64
    assert emitted[0]["tracer_build_sha256"] == timing.sha256_file(tracer)
    assert summary["launch_geometry"] == {
        "kernel_name": "fixture_kernel___0",
        "grid_x": 7,
        "grid_y": 2,
        "grid_z": 1,
        "block_x": 128,
        "block_y": 2,
        "block_z": 1,
        "shared_mem_static_bytes": 2048,
        "shared_mem_dynamic_bytes": 4096,
    }
    assert summary["trace_contract_id"] == timing._TIMING_TRACE_CONTRACT_ID
    assert summary["trace_source_generation"] == _source_generation()


def test_launch_summary_refuses_replacement_without_changing_existing_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trace_root, trace_dir, tracer, admission, _emitted = _prepare(
        tmp_path,
        monkeypatch,
    )
    summary_path = trace_dir / "nvbit_launch_summary.json"
    sentinel = b"existing launch summary\n"
    summary_path.write_bytes(sentinel)

    with pytest.raises(timing.immutable.ImmutableArtifactError):
        timing._emit_nvbit_launch_summary(
            cell=_cell(),
            trace_dir=trace_dir,
            admission=admission,
            stats_payload=(trace_dir / "stats.csv").read_bytes(),
            binary_sha256="a" * 64,
            tracer_build_sha256=timing.sha256_file(tracer),
            source_generation=_source_generation(),
        )

    assert summary_path.read_bytes() == sentinel


@pytest.mark.parametrize(
    "field, catalog_value, admitted_value",
    (
        ("shared_mem_static_bytes", 1024, 2048),
        ("shared_mem_dynamic_bytes", 2048, 4096),
    ),
)
def test_launch_summary_rejects_catalog_split_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    catalog_value: int,
    admitted_value: int,
) -> None:
    _trace_root, trace_dir, tracer, admission, emitted = _prepare(
        tmp_path,
        monkeypatch,
    )
    cell = _cell()
    cell["launch_geometry"][field] = catalog_value

    with pytest.raises(RuntimeError, match=field):
        timing._emit_nvbit_launch_summary(
            cell=cell,
            trace_dir=trace_dir,
            admission=admission,
            stats_payload=(trace_dir / "stats.csv").read_bytes(),
            binary_sha256="a" * 64,
            tracer_build_sha256=timing.sha256_file(tracer),
            source_generation=_source_generation(),
        )

    assert admitted_value != catalog_value
    assert emitted == []
    assert not (trace_dir / "nvbit_launch_summary.json").exists()


def test_launch_summary_rejects_stats_geometry_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trace_root, trace_dir, tracer, admission, emitted = _prepare(
        tmp_path,
        monkeypatch,
    )
    _write_stats(trace_dir / "stats.csv", grid_dimX=8)

    with pytest.raises(RuntimeError, match="stats.csv launch geometry"):
        timing._emit_nvbit_launch_summary(
            cell=_cell(),
            trace_dir=trace_dir,
            admission=admission,
            stats_payload=(trace_dir / "stats.csv").read_bytes(),
            binary_sha256="a" * 64,
            tracer_build_sha256=timing.sha256_file(tracer),
            source_generation=_source_generation(),
        )

    assert emitted == []


@pytest.mark.parametrize(
    "stats_override",
    (
        {"device_id": 1},
        {"stream_id": 2},
        {"kernel id": "kernel-9.trace"},
        {"kernel mangled name": "other_kernel"},
    ),
)
def test_launch_summary_rejects_stats_launch_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stats_override: dict[str, object],
) -> None:
    _trace_root, trace_dir, tracer, admission, emitted = _prepare(
        tmp_path,
        monkeypatch,
    )
    _write_stats(trace_dir / "stats.csv", **stats_override)

    with pytest.raises(RuntimeError, match="stats.csv launch identity"):
        timing._emit_nvbit_launch_summary(
            cell=_cell(),
            trace_dir=trace_dir,
            admission=admission,
            stats_payload=(trace_dir / "stats.csv").read_bytes(),
            binary_sha256="a" * 64,
            tracer_build_sha256=timing.sha256_file(tracer),
            source_generation=_source_generation(),
        )

    assert emitted == []


def test_launch_summary_rejects_multiple_authenticated_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trace_root, trace_dir, tracer, admission, emitted = _prepare(
        tmp_path,
        monkeypatch,
    )
    second = SimpleNamespace(
        launch_index=1,
        device_id=0,
        stream_id=0,
        kernel_id=1,
        kernel_signature_inputs=admission.launches[0].kernel_signature_inputs,
    )
    admission.launch_count = 2
    admission.launches = (*admission.launches, second)

    with pytest.raises(RuntimeError, match="exactly one authenticated launch"):
        timing._emit_nvbit_launch_summary(
            cell=_cell(),
            trace_dir=trace_dir,
            admission=admission,
            stats_payload=(trace_dir / "stats.csv").read_bytes(),
            binary_sha256="a" * 64,
            tracer_build_sha256=timing.sha256_file(tracer),
            source_generation=_source_generation(),
        )

    assert emitted == []


def test_launch_summary_rejects_multiple_stats_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trace_root, trace_dir, tracer, admission, emitted = _prepare(
        tmp_path,
        monkeypatch,
    )
    stats_payload = (trace_dir / "stats.csv").read_bytes()
    stats_payload += stats_payload.splitlines(keepends=True)[1]

    with pytest.raises(RuntimeError, match="exactly one data row"):
        timing._emit_nvbit_launch_summary(
            cell=_cell(),
            trace_dir=trace_dir,
            admission=admission,
            stats_payload=stats_payload,
            binary_sha256="a" * 64,
            tracer_build_sha256=timing.sha256_file(tracer),
            source_generation=_source_generation(),
        )

    assert emitted == []


def test_timing_capture_executes_binary_and_tracer_through_retained_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tmp_path / "external-build" / "tracer_tool.so"
    tracer.parent.mkdir()
    tracer.write_bytes(b"split tracer bytes")
    binary = tmp_path / "fixture.bin"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    trace_root = tmp_path / "traces"
    cell = _cell()
    cell.update({
        "binary_sha256": timing.sha256_file(binary),
        "run_args": ["--shape", "fixture"],
    })
    runtime_descriptors: list[int] = []
    hostile_environment = {
        "CUDA_INJECTION64_PATH": "/hostile/injection64.so",
        "CUDA_INJECTION32_PATH": "/hostile/injection32.so",
        "LD_PRELOAD": "/hostile/preload.so",
        "LD_AUDIT": "/hostile/audit.so",
        "NVBIT_PB_PREFIX": "/hostile/pb",
        "NVBIT_TOOL": "/hostile/tool.so",
        "SIMBENCH_NVBIT_PHASE_INJECT": "hostile",
        "TRACE_SAMPLING_SIDECAR": "/hostile/sidecar.json",
        "TOOL_VERBOSE": "99",
        "TRACES_FOLDER": "/hostile/traces",
        "USER_DEFINED_FOLDERS": "hostile",
    }
    for name, value in hostile_environment.items():
        monkeypatch.setenv(name, value)

    def runtime_env(path, *, base_env, pb_lib, runtime_manifest, retained_descriptor):
        assert path == tracer.resolve()
        assert pb_lib == timing.DEFAULT_PB_LIB
        assert all(name not in base_env for name in hostile_environment)
        assert (
            Path(f"/proc/self/fd/{retained_descriptor}").read_bytes()
            == tracer.read_bytes()
        )
        runtime_descriptors.append(retained_descriptor)
        return _retained_runtime(
            {**base_env, **hostile_environment},
            retained_descriptor + 1,
        )

    def run(command, *, env, pass_fds, **kwargs):
        assert kwargs["cwd"] == str(timing.REPO_ROOT)
        assert kwargs["timeout"] == 900
        assert command[0].startswith("/proc/self/fd/")
        assert env["CUDA_INJECTION64_PATH"].startswith("/proc/self/fd/")
        assert "CUDA_INJECTION32_PATH" not in env
        assert "LD_PRELOAD" not in env
        assert "NVBIT_TOOL" not in env
        binary_descriptor = int(command[0].rsplit("/", 1)[1])
        tracer_descriptor = int(
            env["CUDA_INJECTION64_PATH"].rsplit("/", 1)[1]
        )
        trace_descriptor = int(env["TRACES_FOLDER"].rsplit("/", 1)[1])
        assert pass_fds == (
            binary_descriptor,
            tracer_descriptor,
            trace_descriptor,
            tracer_descriptor + 1,
        )
        assert os.pread(binary_descriptor, 4096, 0) == binary.read_bytes()
        assert os.pread(tracer_descriptor, 4096, 0) == tracer.read_bytes()
        trace_dir = Path(env["TRACES_FOLDER"])
        (trace_dir / "threadblocks").mkdir()
        (trace_dir / "dynamic_trace.pb").write_bytes(b"root")
        (trace_dir / "threadblocks" / "0.pb").write_bytes(b"threadblock")
        return SimpleNamespace(returncode=0, stdout=b"success\xff", stderr=b"")

    monkeypatch.setattr(timing, "_binary_path_for", lambda _cell: binary)
    monkeypatch.setattr(
        timing,
        "_require_external_tracer",
        lambda path, *, trace_root: Path(path).resolve(),
    )
    monkeypatch.setattr(timing, "retained_tracer_runtime_env", runtime_env)
    source_generation = _source_generation()
    monkeypatch.setattr(
        timing,
        "_snapshot_timing_trace_source",
        lambda _trace_dir: {
            "source_generation": source_generation,
            "admission": _admission(),
            "stats_payload": _stats_payload(),
        },
    )
    monkeypatch.setattr(timing, "_run_timing_capture", run)
    identity = timing._capture_nvbit_observed_sector_trace(
        cell=cell,
        trace_root=trace_root,
        tracer_so_path=tracer.resolve(),
    )

    assert identity["binary_sha256"] == timing.sha256_file(binary)
    assert identity["tracer_so_sha256"] == timing.sha256_file(tracer)
    assert len(identity["nvbit_launch_attempt_seal_sha256"]) == 64
    assert identity["trace_source_generation_sha256"] == source_generation[
        "source_generation_sha256"
    ]
    assert len(runtime_descriptors) == 1
    trace_dir = trace_root / cell["shape_point"]
    assert (trace_dir / "dynamic_trace.pb").is_file()
    assert (trace_dir / "nvbit_retained_tracer.so").read_bytes() == tracer.read_bytes()
    assert (trace_dir / "nvbit_launch.stdout.bin").read_bytes() == b"success\xff"
    attempt = json.loads((trace_dir / "nvbit_launch_attempt.json").read_text())
    start = json.loads((trace_dir / "nvbit_launch_start.json").read_text())
    timing.immutable.verify_model_seal(attempt, field="attempt_seal_sha256")
    assert attempt["status"] == "succeeded"
    assert attempt["trace_source_generation"] == source_generation
    assert start["descriptor_bindings"] == {
        "command_0": "target_binary",
        "CUDA_INJECTION64_PATH": "tracer",
        "TRACES_FOLDER": "trace_directory",
    }
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    attempt_ref = timing._require_timing_capture_attempt(
        trace_dir,
        shape_point=cell["shape_point"],
        binary_sha256=identity["binary_sha256"],
        tracer_sha256=identity["tracer_so_sha256"],
        expected_attempt_seal_sha256=identity[
            "nvbit_launch_attempt_seal_sha256"
        ],
    )
    assert attempt_ref["attempt_seal_sha256"] == (
        identity["nvbit_launch_attempt_seal_sha256"]
    )
    assert attempt_ref["trace_source_generation"] == source_generation
    assert attempt["invocation"]["command"] == [
        "target_binary_descriptor",
        "--shape",
        "fixture",
    ]
    assert attempt["invocation"]["trace_window"] == {
        "instr_begin": timing.TRACE_INSTR_BEGIN,
        "instr_end": timing.TRACE_INSTR_END,
        "dynamic_kernel_limit_end": 1,
        "terminate_upon_limit": 0,
    }
    assert attempt["invocation"]["environment_scope"] == (
        "effective_process_environment_v1"
    )
    assert len(attempt["invocation"]["environment_sha256"]) == 64

    stdout_path = trace_dir / "nvbit_launch.stdout.bin"
    replacement = trace_dir / "replacement.stdout.bin"
    replacement.write_bytes(b"tampered")
    replacement.chmod(0o444)
    replacement.replace(stdout_path)
    with pytest.raises(RuntimeError, match="stdout identity mismatch"):
        timing._require_timing_capture_attempt(
            trace_dir,
            shape_point=cell["shape_point"],
            binary_sha256=identity["binary_sha256"],
            tracer_sha256=identity["tracer_so_sha256"],
            expected_attempt_seal_sha256=identity[
                "nvbit_launch_attempt_seal_sha256"
            ],
        )


def test_timing_capture_bounded_runtime_retains_exact_noisy_failure_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_TIMING_CAPTURE_MAX_OUTPUT_BYTES", 64)

    class TestContainment:
        def popen(self, command, **kwargs):
            return timing.subprocess.Popen(command, **kwargs)

        def kill_all(self):
            return None

        def finish(self):
            return timing.nvbit_process.delegated_cgroup.CgroupContainmentEvidence(
                kind=(
                    timing.nvbit_process.delegated_cgroup.CGROUP_V2_KILL_V1
                ),
                parent_cgroup_sha256="a" * 64,
                cgroup_inode=123,
                terminal_populated_zero=True,
                leaf_removed=True,
            )

        def close_after_error(self):
            return None

    monkeypatch.setattr(
        timing.nvbit_process.delegated_cgroup.DelegatedCgroupV2,
        "create",
        TestContainment,
    )

    with pytest.raises(
        timing.nvbit_process.NvbitCaptureProcessError
    ) as caught:
        timing._run_timing_capture(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 4096)",
            ],
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=5,
            pass_fds=(),
        )

    assert caught.value.code == timing.nvbit_process.NVBIT_CAPTURE_OUTPUT_LIMIT
    assert caught.value.stdout == b"x" * 64
    assert caught.value.stderr == b""
    assert caught.value.process_id is not None


def test_timing_capture_attempt_preserves_typed_bounded_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tmp_path / "external-build" / "tracer_tool.so"
    tracer.parent.mkdir()
    tracer.write_bytes(b"split tracer")
    binary = tmp_path / "fixture.bin"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    cell = _cell()
    cell.update({
        "binary_sha256": timing.sha256_file(binary),
        "run_args": [],
    })
    exact_stdout = b"bounded stdout \xff"
    exact_stderr = b"bounded stderr \x80"

    monkeypatch.setattr(timing, "_binary_path_for", lambda _cell: binary)
    monkeypatch.setattr(
        timing,
        "_require_external_tracer",
        lambda path, *, trace_root: Path(path).resolve(),
    )
    monkeypatch.setattr(
        timing,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(dict(base_env), retained_descriptor + 1)
        ),
    )

    def fail_bounded(*_args, **_kwargs):
        error = timing.nvbit_process.NvbitCaptureProcessError(
            timing.nvbit_process.NVBIT_CAPTURE_OUTPUT_LIMIT,
            "synthetic bounded output limit",
        )
        error.stdout = exact_stdout
        error.stderr = exact_stderr
        error.process_id = 4321
        error.started_at_utc = "2026-08-04T00:00:00Z"
        raise error

    monkeypatch.setattr(timing, "_run_timing_capture", fail_bounded)
    trace_root = tmp_path / "traces"
    with pytest.raises(
        timing.nvbit_process.NvbitCaptureProcessError
    ) as caught:
        timing._capture_nvbit_observed_sector_trace(
            cell=cell,
            trace_root=trace_root,
            tracer_so_path=tracer,
        )

    assert caught.value.code == timing.nvbit_process.NVBIT_CAPTURE_OUTPUT_LIMIT
    trace_dir = trace_root / cell["shape_point"]
    assert (trace_dir / timing._TIMING_STDOUT_NAME).read_bytes() == exact_stdout
    assert (trace_dir / timing._TIMING_STDERR_NAME).read_bytes() == exact_stderr
    attempt = json.loads((trace_dir / timing._TIMING_ATTEMPT_NAME).read_text())
    assert attempt["process"] == {
        "returncode": None,
        "error_code": timing.nvbit_process.NVBIT_CAPTURE_OUTPUT_LIMIT,
        "error_detail": (
            "nvbit_capture_output_limit: synthetic bounded output limit"
        ),
        "process_id": 4321,
        "started_at_utc": "2026-08-04T00:00:00Z",
    }


def test_timing_capture_retains_exact_failed_attempt_and_allows_fresh_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tmp_path / "external-build" / "tracer_tool.so"
    tracer.parent.mkdir()
    tracer.write_bytes(b"exact split tracer")
    binary = tmp_path / "fixture.bin"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    cell = _cell()
    cell.update({
        "binary_sha256": timing.sha256_file(binary),
        "run_args": [],
    })
    run_count = 0

    def runtime_env(path, *, base_env, pb_lib, runtime_manifest, retained_descriptor):
        assert path == tracer.resolve()
        assert pb_lib == timing.DEFAULT_PB_LIB
        assert retained_descriptor >= 0
        return _retained_runtime(dict(base_env), retained_descriptor + 1)

    def run(_command, *, env, **_kwargs):
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            return SimpleNamespace(
                returncode=9,
                stdout=b"failed stdout \xff",
                stderr=b"failed stderr \x80",
            )
        trace_dir = Path(env["TRACES_FOLDER"])
        (trace_dir / "threadblocks").mkdir()
        (trace_dir / "dynamic_trace.pb").write_bytes(b"root")
        (trace_dir / "threadblocks" / "0.pb").write_bytes(b"threadblock")
        return SimpleNamespace(returncode=0, stdout=b"fresh success", stderr=b"")

    monkeypatch.setattr(timing, "_binary_path_for", lambda _cell: binary)
    monkeypatch.setattr(
        timing,
        "_require_external_tracer",
        lambda path, *, trace_root: Path(path).resolve(),
    )
    monkeypatch.setattr(timing, "retained_tracer_runtime_env", runtime_env)
    source_generation = _source_generation()
    monkeypatch.setattr(
        timing,
        "_snapshot_timing_trace_source",
        lambda _trace_dir: {
            "source_generation": source_generation,
            "admission": _admission(),
            "stats_payload": _stats_payload(),
        },
    )
    monkeypatch.setattr(timing, "_run_timing_capture", run)

    failed_root = tmp_path / "failed-traces"
    with pytest.raises(RuntimeError, match="rc=9"):
        timing._capture_nvbit_observed_sector_trace(
            cell=cell,
            trace_root=failed_root,
            tracer_so_path=tracer.resolve(),
        )

    failed_dir = failed_root / cell["shape_point"]
    assert (failed_dir / "nvbit_retained_tracer.so").read_bytes() == tracer.read_bytes()
    assert (failed_dir / "nvbit_launch.stdout.bin").read_bytes() == b"failed stdout \xff"
    assert (failed_dir / "nvbit_launch.stderr.bin").read_bytes() == b"failed stderr \x80"
    start = json.loads((failed_dir / "nvbit_launch_start.json").read_text())
    attempt = json.loads((failed_dir / "nvbit_launch_attempt.json").read_text())
    timing.immutable.verify_model_seal(start, field="start_seal_sha256")
    timing.immutable.verify_model_seal(attempt, field="attempt_seal_sha256")
    assert attempt["status"] == "failed"
    assert attempt["process"]["returncode"] == 9
    assert attempt["start_seal_sha256"] == start["start_seal_sha256"]

    with pytest.raises(RuntimeError, match="refusing to replace existing"):
        timing._capture_nvbit_observed_sector_trace(
            cell=cell,
            trace_root=failed_root,
            tracer_so_path=tracer.resolve(),
        )
    assert run_count == 1

    fresh_root = tmp_path / "fresh-traces"
    identity = timing._capture_nvbit_observed_sector_trace(
        cell=cell,
        trace_root=fresh_root,
        tracer_so_path=tracer.resolve(),
    )
    assert run_count == 2
    assert identity["tracer_so_sha256"] == timing.sha256_file(tracer)
    fresh_attempt = json.loads(
        (fresh_root / cell["shape_point"] / "nvbit_launch_attempt.json").read_text()
    )
    assert fresh_attempt["status"] == "succeeded"


def test_timing_capture_seals_operator_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tmp_path / "external-build" / "tracer_tool.so"
    tracer.parent.mkdir()
    tracer.write_bytes(b"split tracer")
    binary = tmp_path / "fixture.bin"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    cell = _cell()
    cell.update({
        "binary_sha256": timing.sha256_file(binary),
        "run_args": ["--interrupted"],
    })
    trace_root = tmp_path / "traces"

    monkeypatch.setattr(timing, "_binary_path_for", lambda _cell: binary)
    monkeypatch.setattr(
        timing,
        "_require_external_tracer",
        lambda path, *, trace_root: Path(path).resolve(),
    )
    monkeypatch.setattr(
        timing,
        "retained_tracer_runtime_env",
        lambda _path, *, base_env, retained_descriptor, **_kwargs: (
            _retained_runtime(dict(base_env), retained_descriptor + 1)
        ),
    )
    monkeypatch.setattr(
        timing,
        "_run_timing_capture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        timing._capture_nvbit_observed_sector_trace(
            cell=cell,
            trace_root=trace_root,
            tracer_so_path=tracer.resolve(),
        )

    trace_dir = trace_root / cell["shape_point"]
    attempt = json.loads((trace_dir / "nvbit_launch_attempt.json").read_text())
    timing.immutable.verify_model_seal(attempt, field="attempt_seal_sha256")
    assert attempt["status"] == "failed"
    assert attempt["process"]["error_code"] == "KeyboardInterrupt"
    assert (trace_dir / "nvbit_launch.stdout.bin").read_bytes() == b""
    assert (trace_dir / "nvbit_launch.stderr.bin").read_bytes() == b""


def test_summary_publication_refuses_replacement_and_preserves_existing_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    sentinel = b"existing summary bytes\n"
    path.write_bytes(sentinel)

    with pytest.raises(timing.immutable.ImmutableArtifactError):
        timing._publish_json_summary(path, {"replacement": True})

    assert path.read_bytes() == sentinel


def test_sector_summary_validates_attempt_before_any_derived_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_root = tmp_path / "traces"
    cell = _cell()
    trace_dir = timing._trace_dir_for(trace_root, cell)
    trace_dir.mkdir(parents=True)

    monkeypatch.setattr(
        timing,
        "_require_timing_capture_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("attempt invalid")
        ),
    )
    monkeypatch.setattr(
        timing,
        "_snapshot_timing_trace_source",
        lambda *_args, **_kwargs: pytest.fail("source opened before attempt validation"),
    )

    with pytest.raises(RuntimeError, match="attempt invalid"):
        timing.build_sector_summary(
            cell,
            regs_per_thread=48,
            static_work_path=tmp_path / "static-work.json",
            static_features_path=tmp_path / "static-features.json",
            trace_root=trace_root,
            binary_sha256="a" * 64,
            tracer_build_sha256="b" * 64,
            nvbit_launch_attempt_seal_sha256="c" * 64,
        )

    assert not (trace_dir / "nvbit_launch_summary.json").exists()
    assert not (trace_dir / "observed_sector_summary.json").exists()


def test_sector_summary_rejects_attempt_source_generation_mismatch_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_root = tmp_path / "traces"
    cell = _cell()
    trace_dir = timing._trace_dir_for(trace_root, cell)
    trace_dir.mkdir(parents=True)
    attempt_generation = _source_generation("a")
    observed_generation = _source_generation("b")
    monkeypatch.setattr(
        timing,
        "_require_timing_capture_attempt",
        lambda *_args, **_kwargs: {
            "trace_source_generation": attempt_generation,
        },
    )
    monkeypatch.setattr(
        timing,
        "_snapshot_timing_trace_source",
        lambda *_args, **_kwargs: {
            "source_generation": observed_generation,
        },
    )
    monkeypatch.setattr(
        timing.trace_admission_contract,
        "open_admitted_trace_sources",
        lambda *_args, **_kwargs: pytest.fail(
            "mismatched source generation reached retained consumer"
        ),
    )

    with pytest.raises(RuntimeError, match="attempt and retained trace source"):
        timing.build_sector_summary(
            cell,
            regs_per_thread=48,
            static_work_path=tmp_path / "static-work.json",
            static_features_path=tmp_path / "static-features.json",
            trace_root=trace_root,
            binary_sha256="a" * 64,
            tracer_build_sha256="b" * 64,
            nvbit_launch_attempt_seal_sha256="c" * 64,
        )

    assert not (trace_dir / "nvbit_launch_summary.json").exists()
    assert not (trace_dir / "observed_sector_summary.json").exists()


def test_sector_summary_retains_raw_threadblock_protos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_root = tmp_path / "traces"
    cell = _cell()
    cell.update({
        "arch": "sm_86",
        "binary_sha256": "a" * 64,
        "source_commit": "b" * 40,
    })
    trace_dir = trace_root / cell["shape_point"]
    threadblock = trace_dir / "threadblocks" / "0.pb"
    threadblock.parent.mkdir(parents=True)
    threadblock_payload = b"unique raw threadblock evidence"
    threadblock.write_bytes(threadblock_payload)
    (trace_dir / "dynamic_trace.pb").write_bytes(b"trace")
    observed_path = trace_dir / "observed_sector_summary.json"
    observed_path.write_text("{}\n")
    static_features = tmp_path / "static-features.json"
    static_work = tmp_path / "static-work.json"
    static_features.write_text("{}\n")
    static_work.write_text("{}\n")
    observed_summary = {
        "observation_basis": {
            "threadblock_proto_file_count": 1,
            "threadblock_proto_set_sha256": "c" * 64,
            "observed_access_count": 3,
        },
        "summarize_launch": {
            "finite_miss_count": 0,
            "per_op_class_counts": {"1": 3},
            "per_access_kind_counts": {"2": 3},
        },
    }
    source_generation = _source_generation()
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    monkeypatch.setattr(
        timing,
        "_snapshot_timing_trace_source",
        lambda _trace_dir: {
            "admission": object(),
            "dynamic_trace_payload": b"trace",
            "stats_payload": b"",
            "trace_contract_id": timing._TIMING_TRACE_CONTRACT_ID,
            "source_generation": source_generation,
            "limits": timing._TIMING_TRACE_LIMITS,
        },
    )
    retained_sources = SimpleNamespace(verify_unchanged=lambda: None)
    monkeypatch.setattr(
        timing.trace_admission_contract,
        "open_admitted_trace_sources",
        lambda *_args, **_kwargs: nullcontext(retained_sources),
    )
    monkeypatch.setattr(
        timing, "_verify_timing_trace_source", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        timing,
        "_emit_nvbit_launch_summary",
        lambda **_kwargs: {"launch_geometry": cell["launch_geometry"]},
    )
    monkeypatch.setattr(
        timing,
        "_summarize_observed_sector_trace",
        lambda **_kwargs: observed_summary,
    )

    monkeypatch.setattr(
        timing,
        "_require_timing_capture_attempt",
        lambda *_args, **_kwargs: {
            "path": "attempt.json",
            "trace_source_generation": source_generation,
        },
    )

    summary = timing.build_sector_summary(
        cell,
        regs_per_thread=48,
        static_work_path=static_work,
        static_features_path=static_features,
        trace_root=trace_root,
        binary_sha256="a" * 64,
        tracer_build_sha256="d" * 64,
        nvbit_launch_attempt_seal_sha256="e" * 64,
    )

    assert threadblock.read_bytes() == threadblock_payload
    assert (
        summary["source_trace"]["threadblock_proto_retention"]
        == "retained_raw_threadblock_protos_after_digest"
    )


@pytest.mark.parametrize("kind", ("relative", "repository", "symlink", "capture_root"))
def test_timing_capture_rejects_nonexternal_tracer_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    trace_root = tmp_path / "traces"
    external = tmp_path / "external" / "tracer.so"
    external.parent.mkdir()
    external.write_bytes(b"tracer")
    if kind == "relative":
        tracer = Path("relative-tracer.so")
    elif kind == "repository":
        tracer = Path(timing.__file__).resolve()
    elif kind == "symlink":
        tracer = tmp_path / "tracer-link.so"
        tracer.symlink_to(external)
    else:
        trace_root.mkdir()
        tracer = trace_root / "tracer.so"
        tracer.write_bytes(b"tracer")

    monkeypatch.setattr(
        timing.campaign_preflight, "is_inside_git_worktree", lambda _path: False,
    )
    monkeypatch.setattr(
        timing,
        "_run_timing_capture",
        lambda *_args, **_kwargs: pytest.fail("capture ran with invalid tracer"),
    )
    with pytest.raises((RuntimeError, ValueError)):
        timing._capture_nvbit_observed_sector_trace(
            cell=_cell(), trace_root=trace_root, tracer_so_path=tracer,
        )
    assert not (trace_root / _cell()["shape_point"]).exists()


def test_timing_attach_requires_fresh_capture_for_missing_sector_summary(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    trace_root = tmp_path / "traces"
    with pytest.raises(RuntimeError, match="fresh descriptor-bound capture"):
        timing.attach_feature_artifacts_to_cohort(
            {"cells": [_cell()]},
            feature_dir=feature_dir,
            trace_root=trace_root,
        )
    assert not feature_dir.exists()
    assert not trace_root.exists()


def test_timing_attach_capture_flags_require_explicit_tracer_before_output(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    trace_root = tmp_path / "traces"
    with pytest.raises(RuntimeError, match="requires --tracer-so"):
        timing.attach_feature_artifacts_to_cohort(
            {"cells": [_cell()]},
            feature_dir=feature_dir,
            trace_root=trace_root,
            capture_missing_nvbit=True,
        )
    assert not feature_dir.exists()
    assert not trace_root.exists()


def test_active_sector_schema_accepts_producer_generation_fields_and_stays_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    schema = json.loads(timing._SECTOR_SUMMARY_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(summary)

    generation_paths = (
        ("source_trace", "trace_contract_id"),
        ("source_trace", "trace_source_generation"),
        ("nvbit_launch_summary", "trace_contract_id"),
        ("nvbit_launch_summary", "trace_source_generation"),
        ("observation_basis", "trace_contract_id"),
        ("observation_basis", "trace_source_generation"),
        ("source_trace", "nvbit_launch_attempt", "trace_source_generation"),
    )
    for path in generation_paths:
        candidate = json.loads(json.dumps(summary))
        cursor = candidate
        for component in path[:-1]:
            cursor = cursor[component]
        del cursor[path[-1]]
        assert list(validator.iter_errors(candidate)), path

    for object_name in (
        "source_trace",
        "nvbit_launch_summary",
        "observation_basis",
    ):
        candidate = json.loads(json.dumps(summary))
        candidate[object_name]["schema_drift_field"] = "forbidden"
        assert list(validator.iter_errors(candidate)), object_name


def test_active_sector_schema_rejects_vacuous_zero_count_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    schema = json.loads(timing._SECTOR_SUMMARY_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    positive_count_paths = (
        ("source_trace", "threadblock_proto_file_count"),
        ("observation_basis", "threadblock_proto_file_count"),
        ("observation_basis", "memory_instruction_record_count"),
        ("observation_basis", "observed_access_count"),
        ("observation_basis", "stats_total_insts"),
        ("observation_basis", "stats_total_reported_insts"),
        ("summarize_launch", "unique_sector_count"),
        ("summarize_launch", "store_count"),
        ("sector_counts", "global_write_32b_sectors"),
        ("sector_counts", "total_32b_sectors"),
        ("sector_counts", "total_128b_lines"),
        ("sector_counts", "estimated_bytes"),
        ("sector_count_derivation", "observed_access_count"),
    )
    for path in positive_count_paths:
        candidate = json.loads(json.dumps(summary))
        cursor = candidate
        for component in path[:-1]:
            cursor = cursor[component]
        cursor[path[-1]] = 0
        assert list(validator.iter_errors(candidate)), path


def test_existing_sector_summary_admission_validates_schema_and_observed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    summary_path = tmp_path / "sector-summary.json"
    payload = (json.dumps(summary, sort_keys=True) + "\n").encode("utf-8")
    summary_path.write_bytes(payload)

    admitted, admitted_sha256 = timing._load_current_sector_summary(
        summary_path,
        cell=cell,
        trace_root=trace_root,
    )

    assert admitted == summary
    assert admitted_sha256 == timing.immutable.sha256_bytes(payload)


def test_existing_sector_summary_rejects_authoritative_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    summary["source_trace"]["forged_schema_extension"] = True
    summary_path = tmp_path / "sector-summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="violates authoritative schema"):
        timing._load_current_sector_summary(
            summary_path,
            cell=cell,
            trace_root=trace_root,
        )


def test_existing_sector_summary_rejects_referenced_observed_byte_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    observed_path = trace_dir / "observed_sector_summary.json"
    observed_path.write_bytes(observed_path.read_bytes() + b" ")
    summary_path = tmp_path / "sector-summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"observed_sector_summary\.sha256"):
        timing._load_current_sector_summary(
            summary_path,
            cell=cell,
            trace_root=trace_root,
        )


def test_existing_sector_summary_rejects_rehashed_observed_summary_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    observed_path = trace_dir / "observed_sector_summary.json"
    forged = json.loads(observed_path.read_text(encoding="utf-8"))
    forged["summarize_launch"]["cold_count"] = 2
    forged_payload = (json.dumps(forged, sort_keys=True) + "\n").encode("utf-8")
    observed_path.write_bytes(forged_payload)
    summary["observed_sector_summary"]["sha256"] = (
        timing.immutable.sha256_bytes(forged_payload)
    )
    summary_path = tmp_path / "sector-summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="summarize_launch"):
        timing._load_current_sector_summary(
            summary_path,
            cell=cell,
            trace_root=trace_root,
        )


def test_existing_sector_summary_rejects_coordinated_summary_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    observed_path = trace_dir / "observed_sector_summary.json"
    forged = json.loads(observed_path.read_text(encoding="utf-8"))
    forged["observation_basis"]["observed_access_count"] = 2
    forged["observation_basis"]["observed_miss_count"] = 2
    forged["summarize_launch"]["cold_count"] = 2
    forged["summarize_launch"]["unique_sector_count"] = 2
    forged["summarize_launch"]["per_op_class_counts"] = {"1": 2}
    forged["summarize_launch"]["per_access_kind_counts"] = {"2": 2}
    forged["summarize_launch"]["store_count"] = 2
    forged_payload = (json.dumps(forged, sort_keys=True) + "\n").encode("utf-8")
    observed_path.write_bytes(forged_payload)
    summary["observed_sector_summary"]["sha256"] = (
        timing.immutable.sha256_bytes(forged_payload)
    )
    summary["observation_basis"] = forged["observation_basis"]
    summary["summarize_launch"] = forged["summarize_launch"]
    summary["sector_counts"] = timing._derive_sector_counts_from_observation(forged)
    summary["sector_count_derivation"]["observed_access_count"] = 2
    summary_path = tmp_path / "sector-summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="observed_sector_summary.contents"):
        timing._load_current_sector_summary(
            summary_path,
            cell=cell,
            trace_root=trace_root,
        )


def test_existing_sector_summary_rejects_missing_retained_threadblock_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    for threadblock_path in (trace_dir / "threadblocks").rglob("*.pb"):
        threadblock_path.unlink()
    summary_path = tmp_path / "sector-summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no threadblock protobufs"):
        timing._load_current_sector_summary(
            summary_path,
            cell=cell,
            trace_root=trace_root,
        )


def test_sector_count_derivation_rejects_non_global_observation() -> None:
    observed = {
        "observation_basis": {"observed_access_count": 3},
        "summarize_launch": {
            "per_op_class_counts": {"2": 3},
            "per_access_kind_counts": {"2": 3},
        },
    }

    with pytest.raises(RuntimeError, match="not an all-global sector stream"):
        timing._derive_sector_counts_from_observation(observed)


@pytest.mark.parametrize(
    "field, forged_value",
    (
        ("global_write_32b_sectors", 2),
        ("total_32b_sectors", 4),
        ("total_128b_lines", 2),
        ("estimated_bytes", 128),
    ),
)
def test_existing_sector_summary_rejects_forged_sector_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    forged_value: int,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    summary["sector_counts"][field] = forged_value
    summary_path = tmp_path / "sector-summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sector_counts"):
        timing._load_current_sector_summary(
            summary_path,
            cell=cell,
            trace_root=trace_root,
        )


@pytest.mark.parametrize(
    "field, forged_value",
    (
        ("observed_access_count", 2),
        ("observed_sector_summary_path", "forged-observed-summary.json"),
    ),
)
def test_existing_sector_summary_rejects_forged_sector_derivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    forged_value: object,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    summary["sector_count_derivation"][field] = forged_value
    summary_path = tmp_path / "sector-summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sector_count_derivation"):
        timing._load_current_sector_summary(
            summary_path,
            cell=cell,
            trace_root=trace_root,
        )


@pytest.mark.parametrize(
    "field, foreign_value",
    (
        ("artifact_kind", "foreign_sector_summary"),
        ("slug", "copied_foreign_slug"),
        ("shape_point", "copied_foreign_shape"),
        ("binary_sha256", "e" * 64),
        ("kernel_signature_sha256", "e" * 64),
        ("source_commit", "e" * 40),
    ),
)
def test_timing_attach_rejects_copied_foreign_sector_summary_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    foreign_value: str,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    summary[field] = foreign_value
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    summary_path = (
        feature_dir / f"{cell['shape_point']}_nvbit_sector_summary.json"
    )
    foreign_payload = (json.dumps(summary, sort_keys=True) + "\n").encode()
    summary_path.write_bytes(foreign_payload)
    cache_path = tmp_path / "predictions" / "cache.json"

    with pytest.raises(
        RuntimeError,
        match=rf"(violates authoritative schema.*{field}|identity mismatch: {field})",
    ):
        timing.attach_feature_artifacts_to_cohort(
            {"cells": [cell]},
            feature_dir=feature_dir,
            trace_root=trace_root,
            cache_hit_predictions_path=cache_path,
        )

    assert summary_path.read_bytes() == foreign_payload
    assert not cache_path.exists()


def test_timing_attach_rejects_foreign_kernel_before_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    summary["nvbit_launch_summary"]["harmonized_kernel_name"] = "foreign_kernel"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    summary_path = (
        feature_dir / f"{cell['shape_point']}_nvbit_sector_summary.json"
    )
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=r"nvbit_launch_summary\.harmonized_kernel_name",
    ):
        timing.attach_feature_artifacts_to_cohort(
            {"cells": [cell]},
            feature_dir=feature_dir,
            trace_root=trace_root,
            cache_hit_predictions_path=tmp_path / "cache.json",
        )


def test_timing_attach_rejects_foreign_capture_seal_before_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "_repo_rel", lambda path: str(path))
    cell = _identity_cell()
    trace_root = tmp_path / "traces"
    trace_dir, capture_ref, source_generation = _write_identity_capture(
        trace_root,
        cell,
    )
    summary = _identity_sector_summary(
        cell,
        trace_dir=trace_dir,
        capture_ref=capture_ref,
        source_generation=source_generation,
    )
    summary["source_trace"]["nvbit_launch_attempt"][
        "attempt_seal_sha256"
    ] = "f" * 64
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    summary_path = (
        feature_dir / f"{cell['shape_point']}_nvbit_sector_summary.json"
    )
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="launch attempt identity mismatch"):
        timing.attach_feature_artifacts_to_cohort(
            {"cells": [cell]},
            feature_dir=feature_dir,
            trace_root=trace_root,
            cache_hit_predictions_path=tmp_path / "cache.json",
        )


@pytest.mark.parametrize(
    "shape_point",
    ("../escaped", "nested/escape", r"nested\\escape", ".", ".."),
)
def test_timing_attach_rejects_unsafe_shape_point_before_output(
    tmp_path: Path,
    shape_point: str,
) -> None:
    cell = _cell()
    cell["shape_point"] = shape_point
    feature_dir = tmp_path / "features"
    trace_root = tmp_path / "traces"

    with pytest.raises(ValueError, match="safe bounded path component"):
        timing.attach_feature_artifacts_to_cohort(
            {"cells": [cell]},
            feature_dir=feature_dir,
            trace_root=trace_root,
            cache_hit_predictions_path=tmp_path / "cache.json",
        )

    assert not feature_dir.exists()
    assert not trace_root.exists()
    assert not (tmp_path / "escaped_nvbit_sector_summary.json").exists()


def test_timing_attach_rejects_symlinked_output_root_before_output(
    tmp_path: Path,
) -> None:
    real_feature_dir = tmp_path / "real-features"
    real_feature_dir.mkdir()
    feature_dir = tmp_path / "feature-link"
    feature_dir.symlink_to(real_feature_dir, target_is_directory=True)
    trace_root = tmp_path / "traces"

    with pytest.raises(
        timing.protected_paths.ProtectedPathError,
        match="symlinked output path",
    ):
        timing.attach_feature_artifacts_to_cohort(
            {"cells": []},
            feature_dir=feature_dir,
            trace_root=trace_root,
            cache_hit_predictions_path=tmp_path / "cache.json",
        )

    assert list(real_feature_dir.iterdir()) == []
    assert not trace_root.exists()


def test_timing_attach_refuses_existing_output_target_without_truncation(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    trace_root = tmp_path / "traces"
    cache_path = tmp_path / "predictions" / "cache.json"
    cache_path.parent.mkdir()
    sentinel = b"existing cache prediction bytes\n"
    cache_path.write_bytes(sentinel)

    with pytest.raises(timing.immutable.ImmutableArtifactError, match="already exists"):
        timing.attach_feature_artifacts_to_cohort(
            {"cells": []},
            feature_dir=feature_dir,
            trace_root=trace_root,
            cache_hit_predictions_path=cache_path,
        )

    assert cache_path.read_bytes() == sentinel
    assert not feature_dir.exists()
    assert not trace_root.exists()


def test_timing_trace_source_requires_exact_split_v2_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.nvbit_tracer.test_trace_admission import _write_trace

    trace_dir = _write_trace(
        tmp_path / "legacy-trace",
        trace_contract_id="sm86_addr_carry_v1",
        include_split_fields=False,
    )
    monkeypatch.setattr(
        timing.trace_admission_contract,
        "require_trace_admission",
        lambda *_args, **_kwargs: pytest.fail(
            "generic admission ran before the timing split-v2 guard"
        ),
    )

    with pytest.raises(RuntimeError, match="trace contract mismatch"):
        timing._snapshot_timing_trace_source(trace_dir)


def _fail_if_trace_decoder_runs(*_args: object, **_kwargs: object) -> None:
    pytest.fail("trace decoder ran before bounded source inventory checks")


def test_timing_trace_source_enforces_file_count_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_split_trace(tmp_path / "trace")
    limits = replace(
        timing._TIMING_TRACE_LIMITS,
        tree_files=2,
        threadblock_files=2,
    )
    monkeypatch.setattr(
        timing,
        "_preflight_threadblock_payload",
        _fail_if_trace_decoder_runs,
    )
    monkeypatch.setattr(
        timing,
        "_require_timing_trace_contract_id",
        _fail_if_trace_decoder_runs,
    )

    with pytest.raises(timing.immutable.ImmutableArtifactLimitError):
        timing._snapshot_timing_trace_source(trace_dir, limits=limits)


def test_timing_trace_source_rejects_many_zero_byte_files_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_split_trace(tmp_path / "trace")
    noise = trace_dir / "zero-byte-noise"
    noise.mkdir()
    for index in range(64):
        (noise / f"empty-{index:03d}").touch()
    limits = replace(
        timing._TIMING_TRACE_LIMITS,
        tree_files=32,
        threadblock_files=32,
    )
    monkeypatch.setattr(
        timing,
        "_preflight_threadblock_payload",
        _fail_if_trace_decoder_runs,
    )
    monkeypatch.setattr(
        timing,
        "_require_timing_trace_contract_id",
        _fail_if_trace_decoder_runs,
    )

    with pytest.raises(
        timing.immutable.ImmutableArtifactLimitError,
        match="file limit",
    ):
        timing._snapshot_timing_trace_source(trace_dir, limits=limits)


def test_timing_trace_source_enforces_total_bytes_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_split_trace(tmp_path / "trace")
    (trace_dir / "malformed-extra.bin").write_bytes(b"x" * 512)
    limits = replace(
        timing._TIMING_TRACE_LIMITS,
        trace_root_bytes=100,
        allocations_sidecar_bytes=1,
        stats_bytes=100,
        threadblock_bytes=100,
        total_threadblock_bytes=100,
    )
    monkeypatch.setattr(
        timing,
        "_preflight_threadblock_payload",
        _fail_if_trace_decoder_runs,
    )
    monkeypatch.setattr(
        timing,
        "_require_timing_trace_contract_id",
        _fail_if_trace_decoder_runs,
    )

    with pytest.raises(timing.immutable.ImmutableArtifactLimitError):
        timing._snapshot_timing_trace_source(trace_dir, limits=limits)


def test_timing_trace_source_enforces_per_file_bytes_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_split_trace(tmp_path / "trace")
    (trace_dir / "dynamic_trace.pb").write_bytes(b"malformed")
    limits = replace(
        timing._TIMING_TRACE_LIMITS,
        trace_root_bytes=1,
        threadblock_bytes=1,
    )
    monkeypatch.setattr(
        timing,
        "_preflight_threadblock_payload",
        _fail_if_trace_decoder_runs,
    )
    monkeypatch.setattr(
        timing,
        "_require_timing_trace_contract_id",
        _fail_if_trace_decoder_runs,
    )

    with pytest.raises(timing.immutable.ImmutableArtifactLimitError):
        timing._snapshot_timing_trace_source(trace_dir, limits=limits)


def test_timing_trace_source_snapshot_rejects_later_payload_generation(
    tmp_path: Path,
) -> None:
    trace_dir = _write_split_trace(tmp_path / "trace")
    source = timing._snapshot_timing_trace_source(trace_dir)
    threadblock_path = next((trace_dir / "threadblocks").rglob("*.pb"))
    replacement = trace_dir / "replacement.pb"
    replacement.write_bytes(threadblock_path.read_bytes())
    replacement.replace(threadblock_path)

    with pytest.raises(RuntimeError, match="source generation changed"):
        timing._verify_timing_trace_source(trace_dir, source)


def test_timing_trace_source_rejects_empty_threadblock_set_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_split_trace(tmp_path / "trace")
    for threadblock_path in (trace_dir / "threadblocks").rglob("*.pb"):
        threadblock_path.unlink()
    monkeypatch.setattr(
        timing.trace_admission_contract,
        "require_trace_admission",
        lambda *_args, **_kwargs: pytest.fail(
            "generic admission ran for an empty timing threadblock set"
        ),
    )

    with pytest.raises(RuntimeError, match="no threadblock protobufs"):
        timing._snapshot_timing_trace_source(trace_dir)


def test_observed_sector_consumer_rejects_nonempty_vacuous_threadblock(
    tmp_path: Path,
) -> None:
    from tools.nvbit_tracer import import_trace_contract

    threadblock_pb2_mod = import_trace_contract._import_pb_modules()[0]
    empty_threadblock = threadblock_pb2_mod.threadblock().SerializeToString()
    cell = _identity_cell()
    source_snapshot = {
        "threadblocks": (("threadblocks/empty.pb", empty_threadblock),),
        "stats_payload": _stats_payload(
            total_insts=1,
            total_reported_insts=1,
        ),
        "trace_contract_id": timing._TIMING_TRACE_CONTRACT_ID,
        "source_generation": _source_generation(),
    }

    with pytest.raises(RuntimeError, match="observed sector trace has zero accesses"):
        timing._summarize_observed_sector_trace(
            cell=cell,
            trace_root=tmp_path / "traces",
            source_snapshot=source_snapshot,
        )

    assert not (
        tmp_path
        / "traces"
        / cell["shape_point"]
        / "observed_sector_summary.json"
    ).exists()


def test_observed_sector_summary_rejects_malformed_threadblock_before_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.nvbit_tracer import import_trace_contract

    cell = _cell()
    cell.update({
        "arch": "sm_86",
        "binary_sha256": "a" * 64,
        "source_commit": "b" * 40,
    })
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    trace_dir = _write_split_trace(timing._trace_dir_for(trace_root, cell))
    threadblock_path = next((trace_dir / "threadblocks").rglob("*.pb"))
    threadblock_path.write_bytes(b"\x08\x00")

    def fail_if_decoder_runs(*_args: object, **_kwargs: object) -> None:
        pytest.fail("trace decoder ran before threadblock wire preflight")

    monkeypatch.setattr(
        import_trace_contract,
        "validate_trace_dir",
        fail_if_decoder_runs,
    )
    monkeypatch.setattr(
        import_trace_contract,
        "_import_pb_modules",
        fail_if_decoder_runs,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"NVBit threadblock protobuf is malformed .*"
            r"known protobuf field has incompatible wire type"
        ),
    ):
        timing._summarize_observed_sector_trace(
            cell=cell,
            trace_root=trace_root,
        )

    assert not (trace_dir / "observed_sector_summary.json").exists()
