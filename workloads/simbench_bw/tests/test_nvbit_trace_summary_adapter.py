"""C1 ownership tests for the tracked NVBit Python protobuf authorities."""

from __future__ import annotations

import builtins
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

import merge_evidence  # noqa: E402
import nvbit_trace_summary_adapter as adapter  # noqa: E402
import phase_cycle_ledger_aggregator as phase_ledger  # noqa: E402
import phase_inject_orchestrator as phase_inject  # noqa: E402
import run_suite  # noqa: E402
import verify_artifacts  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
)
from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    ProtoBindingIntegrityError,
    ProtoBindingRuntimePrerequisiteError,
)
from tools.nvbit_tracer.test_trace_admission import _write_trace  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)

# Fixture construction now uses the authenticated central loader, so it creates
# no disposable short-name aliases.  Keep the cleanup roster explicit and empty:
# deleting central-owned entries would poison the nonrollbackable descriptor state.
_TEST_PROTOBUF_SHORT_ALIASES: tuple[str, ...] = ()


def _inotify_queue_overflow_event() -> bytes:
    return b"".join(
        (
            (-1).to_bytes(4, sys.byteorder, signed=True),
            immutable._INOTIFY_Q_OVERFLOW.to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
            (0).to_bytes(4, sys.byteorder),
        )
    )


def _summary_generation_relative_paths(
    admission,
    *,
    shape_point: str,
) -> list[str]:
    generation = (
        Path(run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE)
        / shape_point
        / admission.trace_admission_sha256
        / admission.trace_filesystem_generation_sha256
    )
    return [
        (generation / f"nvbit_trace_summary_0_{launch_index}.json").as_posix()
        for launch_index in range(admission.launch_count)
    ]


def _write_trace_into_existing(root: Path) -> Path:
    """Populate a production-owned stage; cleanup is test-fixture-only."""
    scratch = root.with_name(
        f"{root.name}.fixture-{os.urandom(8).hex()}"
    )
    _write_trace(scratch)
    try:
        for child in tuple(scratch.iterdir()):
            shutil.move(os.fspath(child), os.fspath(root / child.name))
    finally:
        scratch.rmdir()
    return root


def _write_trace_with_launch_count(root: Path, launch_count: int) -> Path:
    if launch_count < 1:
        raise ValueError("launch_count must be positive")
    _write_trace(root)
    if launch_count == 1:
        return root
    trace_pb2, threadblock_pb2 = adapter._import_pb_modules()
    trace = trace_pb2.Trace()
    trace.ParseFromString((root / "dynamic_trace.pb").read_bytes())
    stream = trace.gpu_device[0].streams[0]
    template_kernel = stream.kernels[0]
    template_payload = (
        root
        / "threadblocks"
        / "device_0"
        / "stream_0"
        / "kernel_0"
        / "0.pb"
    ).read_bytes()
    for launch_index in range(1, launch_count):
        kernel = stream.kernels.add()
        kernel.CopyFrom(template_kernel)
        kernel.id = launch_index
        kernel.name = f"fixture_kernel_{launch_index}"
        kernel.function_unique_id = launch_index + 1
        threadblock = threadblock_pb2.threadblock()
        threadblock.ParseFromString(template_payload)
        threadblock.warps[0].instructions[0].function_unique_id = (
            launch_index + 1
        )
        payload_path = (
            root
            / "threadblocks"
            / "device_0"
            / "stream_0"
            / f"kernel_{launch_index}"
            / "0.pb"
        )
        payload_path.parent.mkdir(parents=True)
        payload_path.write_bytes(threadblock.SerializeToString())
    (root / "dynamic_trace.pb").write_bytes(trace.SerializeToString())
    (root / "stats.csv").write_text(
        "kernel_id,total_insts,total_reported_insts\n"
        + "".join(
            f"{launch_index},1,1\n"
            for launch_index in range(launch_count)
        ),
        encoding="utf-8",
    )
    return root


def _valid_phase_ledger_payload(
    *,
    slug: str,
    shape_point: str,
    repeat_index: int,
    launch_index: int,
    binary_sha256: str = "a" * 64,
    kernel_signature_sha256: str = "b" * 64,
    tracer_build_sha256: str = "c" * 64,
) -> bytes:
    records = [
        phase_ledger.PhaseClockRecord(
            launch_index=launch_index,
            cta_id_x=0,
            cta_id_y=0,
            cta_id_z=0,
            sm_id=0,
            warpid_tb=0,
            phase_id=0,
            phase_kind=0,
            clock_role=phase_ledger.ClockRole.phase_open.value,
            clock64=100,
            active_mask=0xFFFFFFFF,
        ),
        phase_ledger.PhaseClockRecord(
            launch_index=launch_index,
            cta_id_x=0,
            cta_id_y=0,
            cta_id_z=0,
            sm_id=0,
            warpid_tb=0,
            phase_id=0,
            phase_kind=0,
            clock_role=phase_ledger.ClockRole.phase_close.value,
            clock64=110,
            active_mask=0xFFFFFFFF,
        ),
    ]
    ledger = phase_ledger.build_phase_cycle_ledger(
        records,
        slug=slug,
        shape_point=shape_point,
        repeat_index=repeat_index,
        launch_index=launch_index,
        kernel_signature_sha256=kernel_signature_sha256,
        binary_sha256=binary_sha256,
        nvbit_tracer_build_hash=tracer_build_sha256,
        claim_scope="measured_local",
        ab_overhead_record={
            "t_wall_no_inject_ns": 100,
            "t_wall_with_inject_ns": 101,
            "overhead_pct": 1.0,
            "admission_threshold_pct": 5.0,
            "admitted": True,
        },
        dual_run_manifest_ref="dual_run_manifest.json",
    )
    return phase_ledger.serialize_phase_cycle_ledger(ledger)


def _publish_raw_trace_fixture(
    *,
    artifact_dir: Path,
    source_root: Path,
    shape_point: str,
) -> tuple[object, dict[str, object], str]:
    attempt_id = f"attempt-fixture-{shape_point}"
    material = [
        immutable.ImmutableFile.from_bytes(
            path.relative_to(source_root).as_posix(),
            path.read_bytes(),
        )
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
    ]
    generation = immutable.publish_directory(
        artifact_dir
        / run_suite.RAW_TRACE_GENERATION_NAMESPACE
        / shape_point,
        attempt_id,
        material,
    )
    admitted = require_trace_admission(generation)
    observed = immutable.inspect_immutable_directory_generation(
        generation,
        require_nonempty=True,
        max_files=4096,
        max_directories=4096,
        max_total_file_bytes=512 * 1024 * 1024,
        max_total_relative_path_bytes=32 * 1024 * 1024,
    )
    descriptor = run_suite._directory_generation_record(
        generation=observed,
        artifact_dir=artifact_dir,
        attempt_id=attempt_id,
    )
    descriptor.update({
        "trace_admission_sha256": admitted.trace_admission_sha256,
        "trace_filesystem_generation_sha256": (
            admitted.trace_filesystem_generation_sha256
        ),
        "dynamic_trace_sha256": admitted.dynamic_trace_sha256,
    })
    return admitted, descriptor, attempt_id


def _publish_active_summary_fixture(
    *,
    artifact_dir: Path,
    source_root: Path,
    shape_point: str,
    summary_slug: str = "fixture",
    record_slug: str = "fixture",
) -> tuple[object, dict[str, object], Path]:
    admitted, raw_descriptor, attempt_id = _publish_raw_trace_fixture(
        artifact_dir=artifact_dir,
        source_root=source_root,
        shape_point=shape_point,
    )
    result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=admitted,
        slug=summary_slug,
        shape_point=shape_point,
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )
    assert result["status"] == "ok"
    record_path = _write_active_trace_record(
        artifact_dir,
        admission=admitted,
        shape_point=shape_point,
        paths=result["paths"],
        slug=record_slug,
        raw_trace_generation=raw_descriptor,
        attempt_id=attempt_id,
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
    )
    active = json.loads(record_path.read_text(encoding="utf-8"))
    _add_minimal_manifest_authority(
        artifact_dir=artifact_dir,
        active=active,
    )
    _replace_active_record_for_hostile_fixture(record_path, active)
    return admitted, result, record_path


def _configure_successful_trace_handle(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_dir: Path,
) -> tuple[dict[str, object], run_suite.DeviceProps]:
    """Patch GPU execution only; keep Task238 publications and gates real."""
    executable = tmp_path / "global_scalar_32b_copy_run"
    executable.write_bytes(b"retained executable fixture")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"retained tracer fixture")
    monkeypatch.setattr(run_suite, "NVBIT_TRACER_SO", tracer)
    monkeypatch.setattr(
        run_suite, "_find_kernel_executable", lambda _slug: executable
    )
    monkeypatch.setattr(
        run_suite,
        "_artifact_dir_for_row",
        lambda _row, _props: artifact_dir,
    )

    def fake_capture(*, traces_folder: Path, **_kwargs) -> dict[str, object]:
        _write_trace_into_existing(traces_folder)
        return {
            "baseline_proc": SimpleNamespace(
                returncode=0,
                stdout='{"wall_time_ms": 1.0}',
                stderr="",
            ),
            "baseline_elapsed_s": 0.001,
            "baseline_result": {"wall_time_ms": 1.0},
            "trace_proc": SimpleNamespace(
                returncode=0,
                stdout='{"wall_time_ms": 2.0}',
                stderr="",
            ),
            "trace_elapsed_s": 0.002,
            "binary_sha256": hashlib.sha256(
                b"retained executable fixture"
            ).hexdigest(),
            "tracer_build_sha256": hashlib.sha256(
                b"retained tracer fixture"
            ).hexdigest(),
        }

    monkeypatch.setattr(run_suite, "_run_retained_nvbit_capture", fake_capture)
    monkeypatch.setattr(
        run_suite,
        "_emit_admitted_trace_features",
        lambda **_kwargs: None,
    )
    row: dict[str, object] = {
        "slug": "global_scalar_32b_copy",
        "trace_budget_mib": 64,
        "seed_policy": {"seed": 42},
        "arch_gate": ["sm_86"],
    }
    props = run_suite.DeviceProps()
    props.device_name = "fixture RTX 3090 Ti"
    props.compute_capability_major = 8
    props.compute_capability_minor = 6
    props.uuid = "GPU-fixture"
    props.sm_count = 84
    props.total_global_mem_bytes = 1
    props.cuda_runtime_version = "fixture-cuda"
    props.driver_version = "fixture-driver"
    return row, props


def _write_active_trace_record(
    artifact_dir: Path,
    *,
    admission,
    shape_point: str,
    paths: list[str],
    slug: str = "fixture",
    raw_trace_generation: dict[str, object] | None = None,
    attempt_id: str | None = None,
    binary_sha256: str = "a" * 64,
    tracer_build_sha256: str = "b" * 64,
    kernel_signature_sha256: str = "b" * 64,
) -> Path:
    generation_relative = Path(paths[0]).parent
    assert all(Path(item).parent == generation_relative for item in paths)
    generation = immutable.inspect_immutable_directory_generation(
        artifact_dir / generation_relative,
        max_files=4096,
        max_directories=1,
        max_total_file_bytes=256 * 1024 * 1024,
        max_total_relative_path_bytes=4 * 1024 * 1024,
    )
    generation_files = [
        {
            "path": (generation_relative / identity.relative_path).as_posix(),
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for identity in generation.files
    ]
    generation_manifest = hashlib.sha256(
        json.dumps(
            generation_files,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    record = {
        "schema_version": 1,
        "slug": slug,
        "shape_set": "trace_shapes",
        "shape_point": shape_point,
        "trace_pb_path": f"trace_{shape_point}/dynamic_trace.pb",
        "trace_admission_sha256": admission.trace_admission_sha256,
        "trace_filesystem_generation_sha256": (
            admission.trace_filesystem_generation_sha256
        ),
        "nvbit_trace_summary_emit_status": "ok",
        "nvbit_trace_summary_paths": list(paths),
        "nvbit_trace_summary_generation": {
            "schema_version": 1,
            "path": generation_relative.as_posix(),
            "content_manifest_sha256": generation.content_manifest_sha256,
            "filesystem_generation_sha256": (
                generation.filesystem_generation_sha256
            ),
            "file_count": generation.file_count,
            "directory_count": generation.directory_count,
            "total_file_bytes": generation.total_file_bytes,
            "files": generation_files,
            "manifest_sha256": generation_manifest,
        },
    }
    if raw_trace_generation is not None:
        assert attempt_id is not None
        record.update({
            "trace_capture_attempt_id": attempt_id,
            "raw_trace_generation": raw_trace_generation,
            "trace_pb_path": (
                f"{raw_trace_generation['path']}/dynamic_trace.pb"
            ),
            "binary_sha256": binary_sha256,
            "nvbit_tracer_build_hash": tracer_build_sha256,
            "kernel_signature_sha256": kernel_signature_sha256,
            "dynamic_trace_sha256": admission.dynamic_trace_sha256,
            "threadblock_payload_size_bytes": (
                admission.threadblock_payload_size_bytes
            ),
            "threadblock_payload_manifest_sha256": (
                admission.threadblock_payload_manifest_sha256
            ),
            "activation_state": "active",
        })
    path = artifact_dir / f"trace_{shape_point}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def _replace_active_record_for_hostile_fixture(
    path: Path,
    record: dict[str, object],
) -> None:
    """Test-only replacement that still preserves the prior authority bytes."""
    immutable.retire_regular_file(path)
    immutable.publish_regular_file(
        path.parent,
        path.name,
        (json.dumps(record, indent=2) + "\n").encode("utf-8"),
    )


def _add_minimal_manifest_authority(
    *,
    artifact_dir: Path,
    active: dict[str, object],
) -> None:
    shape_point = str(active["shape_point"])
    manifest = {
        "schema_version": 2,
        "slug": active["slug"],
        "mode": "trace",
        "warm_up_iters": 0,
        "repeat_iters": 1,
        "cv_pct": 0.0,
        "cuda_version": "fixture-cuda",
        "driver_version": "fixture-driver",
        "gpu_sku": "rtx_3090_ti",
        "gpu_uuid": "GPU-fixture",
        "sm_count": 1,
        "total_global_mem_bytes": 1,
        "device_name": "fixture device",
        "compute_capability_major": 1,
        "compute_capability_minor": 0,
        "binary_sha256": active["binary_sha256"],
        "compiler_flags": [],
        "kb_capability_card": (
            "references/NVIDIA_Manuals/blackwell_compatibility/"
            "01_blackwell-architecture-compatibility.md"
        ),
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "shape_point": shape_point,
    }
    manifest_payload = (
        json.dumps(manifest, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_name = f"measurement_manifest_{shape_point}.json"
    manifest_path = artifact_dir / manifest_name
    if manifest_path.exists():
        manifest_payload = manifest_path.read_bytes()
    else:
        immutable.publish_regular_file(
            artifact_dir,
            manifest_name,
            manifest_payload,
        )
    active["measurement_manifest"] = {
        "path": manifest_name,
        "sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "size_bytes": len(manifest_payload),
    }


def _replace_measurement_manifest_for_hostile_fixture(
    *,
    artifact_dir: Path,
    active_path: Path,
    active: dict[str, object],
    payload: bytes,
) -> None:
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest_path = artifact_dir / str(descriptor["path"])
    immutable.retire_regular_file(manifest_path)
    immutable.publish_regular_file(
        artifact_dir,
        manifest_path.name,
        payload,
    )
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["size_bytes"] = len(payload)
    _replace_active_record_for_hostile_fixture(active_path, active)


def _bind_manifest_and_phase_generation(
    *,
    artifact_dir: Path,
    active_path: Path,
    repeat_count: object,
    payloads: dict[str, bytes],
) -> dict[str, object]:
    active = json.loads(active_path.read_text(encoding="utf-8"))
    shape_point = str(active["shape_point"])
    attempt_id = str(active["trace_capture_attempt_id"])
    _add_minimal_manifest_authority(
        artifact_dir=artifact_dir,
        active=active,
    )
    final = immutable.publish_directory(
        artifact_dir
        / run_suite.PHASE_LEDGER_GENERATION_NAMESPACE
        / shape_point,
        attempt_id,
        [
            immutable.ImmutableFile.from_bytes(filename, payload)
            for filename, payload in sorted(payloads.items())
        ],
        accept_existing=False,
    )
    generation = immutable.inspect_immutable_directory_generation(
        final,
        require_nonempty=True,
        max_files=4096,
        max_directories=1,
        max_total_file_bytes=256 * 1024 * 1024,
        max_total_relative_path_bytes=4 * 1024 * 1024,
    )
    descriptor = run_suite._directory_generation_record(
        generation=generation,
        artifact_dir=artifact_dir,
        attempt_id=attempt_id,
    )
    relative = Path(str(descriptor["path"]))
    active["phase_inject_ab"] = {
        "requested": True,
        "repeat_count": repeat_count,
        "generation": descriptor,
        "evidence": [
            {
                "path": (relative / identity.relative_path).as_posix(),
                "sha256": identity.sha256,
                "size_bytes": identity.size_bytes,
            }
            for identity in generation.files
        ],
    }
    _replace_active_record_for_hostile_fixture(active_path, active)
    return active


def _reseal_summary_generation_authority(
    *,
    artifact_dir: Path,
    active_path: Path,
    active: dict[str, object],
) -> None:
    descriptor = active["nvbit_trace_summary_generation"]
    assert isinstance(descriptor, dict)
    relative = Path(str(descriptor["path"]))
    observed = immutable.inspect_immutable_directory_generation(
        artifact_dir / relative,
        require_nonempty=True,
        max_files=4096,
        max_directories=1,
        max_total_file_bytes=256 * 1024 * 1024,
        max_total_relative_path_bytes=4 * 1024 * 1024,
    )
    files = [
        {
            "path": (relative / identity.relative_path).as_posix(),
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for identity in observed.files
    ]
    descriptor.update({
        "content_manifest_sha256": observed.content_manifest_sha256,
        "filesystem_generation_sha256": observed.filesystem_generation_sha256,
        "file_count": observed.file_count,
        "directory_count": observed.directory_count,
        "total_file_bytes": observed.total_file_bytes,
        "files": files,
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                files,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    })
    active["nvbit_trace_summary_paths"] = [item["path"] for item in files]
    _replace_active_record_for_hostile_fixture(active_path, active)


def _raw_file_hashes(path: Path) -> dict[str, str]:
    observed = immutable.inspect_immutable_directory_generation(
        path,
        max_files=4096,
        max_directories=4096,
        max_total_file_bytes=512 * 1024 * 1024,
        max_total_relative_path_bytes=32 * 1024 * 1024,
    )
    return {
        identity.relative_path: identity.sha256
        for identity in observed.files
    }


def _prepare_legacy_join_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, SimpleNamespace]:
    fixture_root = (
        SUITE_ROOT
        / "golden"
        / "cupti_nvbit_join_fixtures"
        / "medium_rmsnorm_bf16"
    )
    cupti_dir = tmp_path / "cupti"
    nvbit_dir = tmp_path / "nvbit"
    output_dir = tmp_path / "joined"
    cupti_dir.mkdir()
    nvbit_dir.mkdir()
    shutil.copyfile(
        fixture_root / "cupti_sidecar.json",
        cupti_dir / "cupti_activity_0_0.json",
    )
    shutil.copyfile(
        fixture_root / "nvbit_trace_summary.json",
        nvbit_dir / "nvbit_trace_summary_0_0.json",
    )
    args = SimpleNamespace(
        cupti_dir=str(cupti_dir),
        nvbit_dir=str(nvbit_dir),
        output_dir=str(output_dir),
        platform="rtx_3090_ti",
        check=False,
        strict=False,
    )
    return cupti_dir, nvbit_dir, output_dir, args


@pytest.mark.parametrize(
    ("limit_name", "expected_token"),
    (
        (
            "_JOIN_WORKING_SET_MAX_TOTAL_BYTES",
            "joined_evidence_input_limit_exceeded",
        ),
        ("_JOIN_OUTPUT_MAX_TOTAL_BYTES", "joined_evidence_output_limit_exceeded"),
    ),
)
def test_join_bounds_decoded_and_serialized_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    limit_name: str,
    expected_token: str,
) -> None:
    _cupti_dir, _nvbit_dir, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    monkeypatch.setattr(merge_evidence, limit_name, 1)

    rc = merge_evidence._run_join_cupti_nvbit(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert expected_token in captured.err
    assert "joined_evidence_input_resource_unavailable" not in captured.err
    assert not (output_dir / args.platform).exists()


def test_join_materialization_budget_isolates_decoded_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = {"decoded": ["x" * 64]}
    retained_input_bytes = 17
    probe = merge_evidence._JoinMaterializationBudget()
    probe.reserve(
        value,
        retained_input_bytes=retained_input_bytes,
        role="decoded probe",
    )
    monkeypatch.setattr(
        merge_evidence,
        "_JOIN_WORKING_SET_MAX_TOTAL_BYTES",
        retained_input_bytes + probe.total_bytes - 1,
    )

    with pytest.raises(
        ImmutableArtifactLimitError,
        match="retained input plus decoded/output material",
    ):
        merge_evidence._JoinMaterializationBudget().reserve(
            value,
            retained_input_bytes=retained_input_bytes,
            role="decoded fixture",
        )


def test_join_materialization_budget_isolates_object_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        merge_evidence,
        "_JOIN_WORKING_SET_MAX_TOTAL_BYTES",
        1 << 30,
    )
    monkeypatch.setattr(
        merge_evidence,
        "_JOIN_MATERIALIZED_MAX_OBJECTS",
        1,
    )

    with pytest.raises(
        ImmutableArtifactLimitError,
        match="decoded objects",
    ):
        merge_evidence._JoinMaterializationBudget().reserve(
            [None],
            retained_input_bytes=0,
            role="object-count fixture",
        )


def test_join_materialization_budget_retains_identity_owners() -> None:
    budget = merge_evidence._JoinMaterializationBudget()
    first = {"decoded": ["first reservation"]}
    first_identity = id(first)
    budget.reserve(
        first,
        retained_input_bytes=0,
        role="first decoded fixture",
    )
    del first

    retained_first = budget.seen_objects[first_identity]
    assert id(retained_first) == first_identity
    prior_count = budget.object_count
    second = {"decoded": ["second reservation"]}
    budget.reserve(
        second,
        retained_input_bytes=0,
        role="second decoded fixture",
    )

    assert budget.object_count > prior_count
    assert budget.seen_objects[id(second)] is second


@pytest.mark.parametrize(
    "budget_factory",
    (
        merge_evidence._JoinMaterializationBudget,
        verify_artifacts._CanonicalAuthorityMaterializationBudget,
    ),
)
def test_materialization_budget_revisits_lazy_path_storage(
    budget_factory,
) -> None:
    path = Path("authority/raw/trace.json")
    budget = budget_factory()
    budget.reserve(
        path,
        retained_input_bytes=0,
        role="fresh path fixture",
    )
    prior_count = budget.object_count

    _ = path.parts
    _ = str(path)
    _ = hash(path)
    initialized_references: list[object] = []
    for path_type in type(path).__mro__:
        slots = path_type.__dict__.get("__slots__", ())
        if type(slots) is str:
            slots = (slots,)
        for slot in slots:
            if slot == "__weakref__":
                continue
            descriptor = path_type.__dict__.get(slot)
            assert descriptor is not None
            try:
                initialized_references.append(
                    descriptor.__get__(path, type(path))
                )
            except AttributeError:
                pass

    budget.reserve(
        path,
        retained_input_bytes=0,
        role="path with initialized lazy caches",
    )

    assert budget.object_count > prior_count
    assert any(type(value) is list for value in initialized_references)
    for retained in initialized_references:
        assert budget.seen_objects[id(retained)] is retained
        if type(retained) in {list, tuple}:
            for member in retained:
                assert budget.seen_objects[id(member)] is member


@pytest.mark.parametrize(
    "budget_factory",
    (
        merge_evidence._JoinMaterializationBudget,
        verify_artifacts._CanonicalAuthorityMaterializationBudget,
    ),
)
def test_materialization_budget_revisits_seen_queue_references(
    budget_factory,
) -> None:
    path = Path("authority/raw/queued.json")
    queued_path = (path,)
    queue = [queued_path]
    budget = budget_factory()
    budget.reserve(
        queue,
        retained_input_bytes=0,
        role="fresh path queue fixture",
    )
    prior_count = budget.object_count

    _ = path.parts
    _ = str(path)
    _ = hash(path)
    budget.reserve(
        queue,
        retained_input_bytes=0,
        role="terminal path queue fixture",
    )

    assert budget.object_count > prior_count
    for slot in (
        "_tail_cached",
        "_str",
        "_str_normcase_cached",
        "_parts_normcase_cached",
        "_hash",
    ):
        try:
            retained = object.__getattribute__(path, slot)
        except AttributeError:
            continue
        assert budget.seen_objects[id(retained)] is retained


def test_join_materialization_budget_isolates_serialized_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = merge_evidence._JoinMaterializationBudget()
    decoded = {"decoded": "fixture"}
    budget.reserve(
        decoded,
        retained_input_bytes=0,
        role="decoded fixture",
    )
    payload = b"serialized fixture payload"
    monkeypatch.setattr(
        merge_evidence,
        "_JOIN_WORKING_SET_MAX_TOTAL_BYTES",
        budget.total_bytes + sys.getsizeof(payload) - 1,
    )

    with pytest.raises(
        ImmutableArtifactLimitError,
        match="retained input plus decoded/output material",
    ):
        budget.reserve(
            payload,
            retained_input_bytes=0,
            role="serialized joined output fixture",
        )


@pytest.mark.parametrize("check", (False, True), ids=("write", "check"))
@pytest.mark.parametrize(
    ("limit_kind", "expected_detail"),
    (
        ("per_file", "joined output "),
        ("aggregate", "joined output aggregate"),
    ),
)
def test_join_enforces_isolated_output_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    check: bool,
    limit_kind: str,
    expected_detail: str,
) -> None:
    _cupti_dir, _nvbit_dir, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    args.check = check
    if limit_kind == "per_file":
        monkeypatch.setattr(merge_evidence, "_JOIN_OUTPUT_MAX_FILE_BYTES", 1)
        monkeypatch.setattr(
            merge_evidence,
            "_JOIN_OUTPUT_MAX_TOTAL_BYTES",
            1 << 30,
        )
    else:
        monkeypatch.setattr(
            merge_evidence,
            "_JOIN_OUTPUT_MAX_FILE_BYTES",
            1 << 30,
        )
        monkeypatch.setattr(merge_evidence, "_JOIN_OUTPUT_MAX_TOTAL_BYTES", 1)

    rc = merge_evidence._run_join_cupti_nvbit(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert "joined_evidence_output_limit_exceeded" in captured.err
    assert expected_detail in captured.err
    if limit_kind == "per_file":
        assert "joined output aggregate" not in captured.err
    assert not (output_dir / args.platform).exists()


@pytest.mark.parametrize("check", (False, True), ids=("write", "check"))
def test_join_rejects_output_path_escape_before_check_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    check: bool,
) -> None:
    _cupti_dir, _nvbit_dir, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    args.check = check
    escaped = tmp_path / "escaped" / "joined_0_0.json"
    monkeypatch.setattr(
        merge_evidence,
        "_joined_output_path",
        lambda *_args, **_kwargs: escaped,
    )

    rc = merge_evidence._run_join_cupti_nvbit(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert "joined_evidence_output_path_escape" in captured.err
    assert not escaped.exists()
    assert not (output_dir / args.platform).exists()


def test_join_classifies_output_allocation_failure_as_output_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _cupti_dir, _nvbit_dir, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )

    def fail_output_materialization(
        _cls: type[ImmutableFile],
        _relative_path: str,
        _payload: bytes,
    ) -> ImmutableFile:
        raise MemoryError("synthetic output materialization exhaustion")

    monkeypatch.setattr(
        merge_evidence.ImmutableFile,
        "from_bytes",
        classmethod(fail_output_materialization),
    )

    rc = merge_evidence._run_join_cupti_nvbit(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert "joined_evidence_output_resource_unavailable" in captured.err
    assert "synthetic output materialization exhaustion" in captured.err
    assert "joined_evidence_input_resource_unavailable" not in captured.err
    assert not (output_dir / args.platform).exists()


def test_kernel_signature_matches_every_committed_golden_fixture() -> None:
    fixture_root = SUITE_ROOT / "golden" / "kernel_signature_fixtures"
    fixtures = sorted(path for path in fixture_root.iterdir() if path.is_dir())
    assert fixtures
    for fixture in fixtures:
        inputs = json.loads((fixture / "inputs.json").read_text(encoding="utf-8"))
        expected_canonical = (fixture / "expected_canonical.txt").read_text(
            encoding="utf-8"
        ).removesuffix("\n")
        expected_sha = (fixture / "expected_sha256.txt").read_text(
            encoding="utf-8"
        ).strip()
        assert adapter.canonical_serialize(inputs) == expected_canonical, fixture
        assert adapter.compute_kernel_signature_sha256(inputs) == expected_sha, fixture


def test_missing_tracked_binding_is_repository_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_authority():
        raise ProtoBindingIntegrityError(
            "tracked protobuf authority is unusable; restore byte-exact committed "
            "content with `git restore --source=HEAD --worktree -- "
            "workloads/simbench_bw/_pb_generated`; do not regenerate bindings"
        )

    monkeypatch.setattr(adapter, "load_dynamic_trace_bindings", reject_authority)
    with pytest.raises(adapter.NvbitTrackedProtobufCorruption) as caught:
        adapter._import_pb_modules()
    diagnostic = str(caught.value)
    assert "tracked protobuf authority" in diagnostic
    assert "git restore --source=HEAD --worktree --" in diagnostic
    assert "do not regenerate" in diagnostic


def test_missing_protobuf_runtime_is_prerequisite_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_runtime():
        raise ProtoBindingRuntimePrerequisiteError(
            "protobuf runtime prerequisite is missing; install exactly "
            "protobuf==7.34.1 from requirements.txt"
        )

    monkeypatch.setattr(adapter, "load_dynamic_trace_bindings", reject_runtime)
    with pytest.raises(adapter.NvbitProtobufRuntimePrerequisiteMissing) as caught:
        adapter._import_pb_modules()
    diagnostic = str(caught.value)
    assert "runtime prerequisite is missing" in diagnostic
    assert "requirements.txt" in diagnostic


def test_merge_join_consumes_only_active_committed_generation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = (
        SUITE_ROOT
        / "golden"
        / "cupti_nvbit_join_fixtures"
        / "medium_rmsnorm_bf16"
    )
    cupti_dir = tmp_path / "cupti"
    nvbit_dir = tmp_path / "nvbit"
    output_dir = tmp_path / "joined"
    cupti_dir.mkdir()
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=nvbit_dir,
    )
    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_OK
    active = json.loads(
        (nvbit_dir / "trace_trace_00.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (nvbit_dir / active["nvbit_trace_summary_paths"][0]).read_text(
            encoding="utf-8"
        )
    )
    cupti = json.loads(
        (fixture_root / "cupti_sidecar.json").read_text(encoding="utf-8")
    )
    for field in (
        "slug",
        "shape_point",
        "repeat_index",
        "launch_index",
        "kernel_signature_sha256",
        "binary_sha256",
    ):
        cupti[field] = summary[field]
    (cupti_dir / "cupti_activity_0_0.json").write_text(
        json.dumps(cupti, indent=2) + "\n", encoding="utf-8"
    )

    # Both files would make the historical recursive glob fail.  Active
    # reference discovery must ignore them completely.
    (nvbit_dir / "nvbit_trace_summary_0_99.json").write_text(
        "not json\n", encoding="utf-8"
    )
    stale_root = (
        nvbit_dir
        / run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE
        / "stale_shape"
        / ("c" * 64)
    )
    immutable.publish_directory(
        stale_root,
        "d" * 64,
        [
            immutable.ImmutableFile.from_bytes(
                "nvbit_trace_summary_0_7.json", b"not json\n"
            )
        ],
    )

    rc = merge_evidence._run_join_cupti_nvbit(
        SimpleNamespace(
            cupti_dir=str(cupti_dir),
            nvbit_dir=str(nvbit_dir),
            output_dir=str(output_dir),
            platform="rtx_3090_ti",
            check=True,
            strict=False,
        )
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "1 join(s) would be written" in captured.out
    assert "schema_invalid_nvbit" not in captured.err
    assert not output_dir.exists()


def test_merge_retains_raw_trace_authority_through_output_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_root = (
        SUITE_ROOT
        / "golden"
        / "cupti_nvbit_join_fixtures"
        / "medium_rmsnorm_bf16"
    )
    cupti_dir = tmp_path / "cupti"
    nvbit_dir = tmp_path / "nvbit"
    output_dir = tmp_path / "joined"
    cupti_dir.mkdir()
    _admission, result, active_path = _publish_active_summary_fixture(
        artifact_dir=nvbit_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    summary = json.loads(
        (nvbit_dir / result["paths"][0]).read_text(encoding="utf-8")
    )
    cupti = json.loads(
        (fixture_root / "cupti_sidecar.json").read_text(encoding="utf-8")
    )
    for field in (
        "slug",
        "shape_point",
        "repeat_index",
        "launch_index",
        "kernel_signature_sha256",
        "binary_sha256",
    ):
        cupti[field] = summary[field]
    (cupti_dir / "cupti_activity_0_0.json").write_text(
        json.dumps(cupti, indent=2) + "\n", encoding="utf-8"
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    dynamic_trace = (
        nvbit_dir
        / active["raw_trace_generation"]["path"]
        / "dynamic_trace.pb"
    )
    original_payload = dynamic_trace.read_bytes()
    commit_attempted = False

    def mutate_before_commit(
        output_root: Path,
        _final_name: str,
        _files,
        *,
        before_commit,
        **_kwargs,
    ) -> Path:
        nonlocal commit_attempted
        commit_attempted = True
        dynamic_trace.chmod(0o644)
        dynamic_trace.write_bytes(original_payload)
        dynamic_trace.chmod(0o444)
        before_commit(output_root, -1)
        raise AssertionError("mutated authority must refuse publication")

    monkeypatch.setattr(
        merge_evidence, "publish_directory", mutate_before_commit
    )
    rc = merge_evidence._run_join_cupti_nvbit(
        SimpleNamespace(
            cupti_dir=str(cupti_dir),
            nvbit_dir=str(nvbit_dir),
            output_dir=str(output_dir),
            platform="rtx_3090_ti",
            check=False,
            strict=False,
        )
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert commit_attempted is True
    assert "authority" in captured.err or "publication_invalid" in captured.err
    assert not (output_dir / "rtx_3090_ti").exists()


@pytest.mark.parametrize("mutation", ("replace", "add"))
def test_merge_retains_complete_cupti_tree_through_output_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    cupti_dir, _nvbit_dir, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    source = cupti_dir / "cupti_activity_0_0.json"
    original_payload = source.read_bytes()
    commit_attempted = False

    def mutate_before_commit(
        output_root: Path,
        _final_name: str,
        _files,
        *,
        before_commit,
        **_kwargs,
    ) -> Path:
        nonlocal commit_attempted
        commit_attempted = True
        if mutation == "replace":
            replacement = source.with_suffix(".replacement")
            replacement.write_bytes(original_payload)
            os.replace(replacement, source)
        else:
            (cupti_dir / "cupti_activity_0_1.json").write_bytes(
                original_payload
            )
        before_commit(output_root, -1)
        raise AssertionError("mutated CUPTI authority must refuse publication")

    monkeypatch.setattr(
        merge_evidence, "publish_directory", mutate_before_commit
    )
    rc = merge_evidence._run_join_cupti_nvbit(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert commit_attempted is True
    assert "authority" in captured.err or "publication_invalid" in captured.err
    assert not (output_dir / args.platform).exists()


def test_merge_retires_join_when_final_source_authority_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cupti_dir, _nvbit_dir, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    source = cupti_dir / "cupti_activity_0_0.json"
    source_payload = source.read_bytes()
    original_publish = merge_evidence.publish_directory
    authority_calls = 0

    def publish_with_final_mutation(
        output_root: Path,
        final_name: str,
        files,
        *,
        before_commit,
        **kwargs,
    ) -> Path:
        def mutate_then_authorize(root: Path, descriptor: int) -> None:
            nonlocal authority_calls
            authority_calls += 1
            if authority_calls == 3:
                replacement = source.with_suffix(".replacement")
                replacement.write_bytes(source_payload)
                os.replace(replacement, source)
            before_commit(root, descriptor)

        return original_publish(
            output_root,
            final_name,
            files,
            before_commit=mutate_then_authorize,
            **kwargs,
        )

    monkeypatch.setattr(
        merge_evidence, "publish_directory", publish_with_final_mutation,
    )
    rc = merge_evidence._run_join_cupti_nvbit(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert authority_calls == 3
    assert "authority" in captured.err or "publication_invalid" in captured.err
    assert not (output_dir / args.platform).exists()
    retired = [
        path for path in output_dir.iterdir()
        if path.name.startswith(".failed-directory-")
    ]
    assert len(retired) == 1
    assert any(path.is_file() for path in retired[0].rglob("*"))


def test_merge_ignores_legacy_summary_under_hidden_staging_directory(
    tmp_path: Path,
) -> None:
    fixture_root = (
        SUITE_ROOT
        / "golden"
        / "cupti_nvbit_join_fixtures"
        / "medium_rmsnorm_bf16"
    )
    nvbit_dir = tmp_path / "nvbit"
    staging = nvbit_dir / ".attempt.staging"
    staging.mkdir(parents=True)
    shutil.copyfile(
        fixture_root / "nvbit_trace_summary.json",
        staging / "nvbit_trace_summary_0_0.json",
    )

    paths, errors = merge_evidence._discover_nvbit_summary_inputs(nvbit_dir)

    assert paths == []
    assert errors == []


def test_merge_rejects_byte_identical_summary_generation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_root = (
        SUITE_ROOT
        / "golden"
        / "cupti_nvbit_join_fixtures"
        / "medium_rmsnorm_bf16"
    )
    cupti_dir = tmp_path / "cupti"
    nvbit_dir = tmp_path / "nvbit"
    output_dir = tmp_path / "joined"
    cupti_dir.mkdir()
    _admission, result, _active_path = _publish_active_summary_fixture(
        artifact_dir=nvbit_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    summary_path = nvbit_dir / result["paths"][0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cupti = json.loads(
        (fixture_root / "cupti_sidecar.json").read_text(encoding="utf-8")
    )
    for field in (
        "slug",
        "shape_point",
        "repeat_index",
        "launch_index",
        "kernel_signature_sha256",
        "binary_sha256",
    ):
        cupti[field] = summary[field]
    (cupti_dir / "cupti_activity_0_0.json").write_text(
        json.dumps(cupti, indent=2) + "\n", encoding="utf-8"
    )

    original_check = merge_evidence._active_summary_generation_descriptor_error
    replaced = False

    def replace_after_descriptor_check(
        trace_record_path: Path,
        trace_record: dict[str, object],
        *,
        trace_admission,
        preload_budget,
        materialization_budget,
    ) -> str | None:
        nonlocal replaced
        error = original_check(
            trace_record_path,
            trace_record,
            trace_admission=trace_admission,
            preload_budget=preload_budget,
            materialization_budget=materialization_budget,
        )
        if error is None and not replaced:
            replaced = True
            payload = summary_path.read_bytes()
            generation = summary_path.parent
            generation.chmod(0o755)
            replacement = generation / ".replacement"
            replacement.write_bytes(payload)
            replacement.chmod(0o444)
            os.replace(replacement, summary_path)
            generation.chmod(0o555)
        return error

    monkeypatch.setattr(
        merge_evidence,
        "_active_summary_generation_descriptor_error",
        replace_after_descriptor_check,
    )
    rc = merge_evidence._run_join_cupti_nvbit(
        SimpleNamespace(
            cupti_dir=str(cupti_dir),
            nvbit_dir=str(nvbit_dir),
            output_dir=str(output_dir),
            platform="rtx_3090_ti",
            check=True,
            strict=False,
        )
    )

    captured = capsys.readouterr()
    assert replaced is True
    assert rc == 1
    assert "joined_evidence_input_authority_changed" in captured.err
    assert not output_dir.exists()


def test_merge_join_preserves_standalone_legacy_fixture(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_root = (
        SUITE_ROOT
        / "golden"
        / "cupti_nvbit_join_fixtures"
        / "medium_rmsnorm_bf16"
    )
    cupti_dir = tmp_path / "cupti"
    nvbit_dir = tmp_path / "nvbit"
    cupti_dir.mkdir()
    nvbit_dir.mkdir()
    shutil.copyfile(
        fixture_root / "cupti_sidecar.json",
        cupti_dir / "cupti_activity_0_0.json",
    )
    shutil.copyfile(
        fixture_root / "nvbit_trace_summary.json",
        nvbit_dir / "nvbit_trace_summary_0_0.json",
    )

    rc = merge_evidence._run_join_cupti_nvbit(
        SimpleNamespace(
            cupti_dir=str(cupti_dir),
            nvbit_dir=str(nvbit_dir),
            output_dir=str(tmp_path / "joined"),
            platform="rtx_3090_ti",
            check=True,
            strict=False,
        )
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "1 join(s) would be written" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("check", (False, True), ids=("write", "check"))
@pytest.mark.parametrize(
    "destination_kind",
    ("regular", "directory", "symlink", "hardlink", "fifo"),
)
def test_merge_rejects_every_preexisting_platform_destination_class(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    destination_kind: str,
    check: bool,
) -> None:
    _cupti, _nvbit, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    args.check = check
    output_dir.mkdir()
    destination = output_dir / args.platform
    retained = tmp_path / "retained"
    retained.write_bytes(b"retained merge destination bytes")
    if destination_kind == "regular":
        destination.write_bytes(b"planted")
    elif destination_kind == "directory":
        destination.mkdir()
        (destination / "marker").write_bytes(b"planted")
    elif destination_kind == "symlink":
        destination.symlink_to(retained)
    elif destination_kind == "hardlink":
        os.link(retained, destination)
    else:
        os.mkfifo(destination)

    try:
        rc = merge_evidence._run_join_cupti_nvbit(args)
        captured = capsys.readouterr()
        assert rc == 1
        assert "joined_evidence_output_publication_invalid" in captured.err
        assert destination.lstat()
        assert retained.read_bytes() == b"retained merge destination bytes"
        if destination_kind == "directory":
            assert (destination / "marker").read_bytes() == b"planted"
    finally:
        # Explicit test-fixture cleanup prevents hostile FIFO/link nodes from
        # escaping into pytest's recursive cleanup path.
        if destination_kind == "directory":
            (destination / "marker").unlink()
            destination.rmdir()
        else:
            destination.unlink()


def test_merge_platform_generation_has_no_partial_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cupti, _nvbit, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    observed_before_commit = False

    def publish_with_visibility_gate(
        output_root: Path,
        final_name: str,
        files,
        **kwargs,
    ) -> Path:
        material = tuple(files)
        upstream_before_commit = kwargs.pop("before_commit", None)

        def before_commit(root_path: Path, descriptor: int) -> None:
            nonlocal observed_before_commit
            if upstream_before_commit is not None:
                upstream_before_commit(root_path, descriptor)
            stage_path = root_path / f".{final_name}.staging"
            final_path = root_path / final_name
            if not stage_path.exists():
                # Initial authorization has neither namespace; final
                # authorization has only the complete committed namespace.
                if final_path.exists():
                    assert tuple(final_path.rglob("joined_*.json"))
                return
            observed_before_commit = True
            assert not final_path.exists()
            with immutable.PinnedDirectoryReader(stage_path) as reader:
                snapshot = reader.snapshot_tree(
                    require_readonly_files=True,
                    max_files=len(material),
                    max_directories=8,
                    max_total_file_bytes=16 * 1024 * 1024,
                    max_total_relative_path_bytes=4096,
                )
                assert len(snapshot.files) == len(material)

        return immutable.publish_directory(
            output_root,
            final_name,
            material,
            before_commit=before_commit,
            **kwargs,
        )

    monkeypatch.setattr(
        merge_evidence, "publish_directory", publish_with_visibility_gate
    )
    rc = merge_evidence._run_join_cupti_nvbit(args)

    assert rc == 0
    assert observed_before_commit is True
    platform = output_dir / args.platform
    outputs = tuple(platform.rglob("joined_*.json"))
    assert len(outputs) == 1
    assert stat.S_IMODE(platform.stat().st_mode) == 0o555
    assert stat.S_IMODE(outputs[0].stat().st_mode) == 0o444
    # Test-only permission restoration for pytest cleanup.
    for directory in sorted(
        (path for path in platform.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
    ):
        directory.chmod(0o755)
    platform.chmod(0o755)


def test_run_suite_dependency_failures_propagate_hard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    (tmp_path / "fixture_run").write_bytes(b"executable")
    (tmp_path / "tracer_tool.so").write_bytes(b"tracer")
    original_import = builtins.__import__

    def reject_adapter(name: str, *args, **kwargs):
        if name == "nvbit_trace_summary_adapter":
            raise ImportError("planted missing tracked adapter")
        return original_import(name, *args, **kwargs)

    cases: list[tuple[str, object]] = [("adapter_import_failed", reject_adapter)]

    def corrupt_emit(**_kwargs) -> dict[str, object]:
        raise adapter.NvbitTrackedProtobufCorruption("planted corruption")

    fake_adapter = SimpleNamespace(
        emit_admitted_nvbit_trace_summary=corrupt_emit,
        NvbitTrackedProtobufCorruption=adapter.NvbitTrackedProtobufCorruption,
        NvbitProtobufRuntimePrerequisiteMissing=(
            adapter.NvbitProtobufRuntimePrerequisiteMissing
        ),
    )
    cases.append(("tracked_protobuf_corrupt", fake_adapter))

    for expected_status, planted in cases:
        with monkeypatch.context() as context:
            if callable(planted):
                context.setattr(builtins, "__import__", planted)
            else:
                context.setitem(sys.modules, "nvbit_trace_summary_adapter", planted)
            result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
                trace_admission=admission,
                slug="fixture",
                shape_point="fixture_shape",
                binary_sha256=hashlib.sha256(b"executable").hexdigest(),
                tracer_build_sha256=hashlib.sha256(b"tracer").hexdigest(),
                artifact_dir=tmp_path,
            )
        assert result["status"] == expected_status
        assert result["failed_hard"] is True
        assert result["paths"] == []
        assert "skipped" not in result["status"]


def test_admitted_summary_never_reopens_trace_bytes(tmp_path: Path) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    executable = tmp_path / "fixture_run"
    tracer = tmp_path / "tracer_tool.so"
    executable.write_bytes(b"executable")
    tracer.write_bytes(b"tracer")
    binary_sha256 = hashlib.sha256(b"executable").hexdigest()
    tracer_build_sha256 = hashlib.sha256(b"tracer").hexdigest()
    shutil.rmtree(trace_dir)
    executable.unlink()
    tracer.unlink()

    output = tmp_path / "summary.json"
    record = adapter.emit_admitted_nvbit_trace_summary(
        admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        repeat_index=0,
        launch_index=0,
        binary_sha256=binary_sha256,
        tracer_build_sha256=tracer_build_sha256,
        output_path=output,
    )

    assert output.is_file()
    assert record["kernel_name"] == "fixture_kernel"
    assert record["inst_count"] == 1
    assert record["binary_sha256"] == binary_sha256
    assert record["nvbit_tracer_build_hash"] == tracer_build_sha256
    assert record["nvbit_trace_pb_path"] == admission.dynamic_trace_path
    assert record["nvbit_trace_pb_sha256"] == admission.dynamic_trace_sha256
    assert record["schema_version"] == 2
    assert record["nvbit_threadblock_payload_size_bytes"] == (
        admission.threadblock_payload_size_bytes
    )
    assert record["nvbit_threadblock_payload_manifest_sha256"] == (
        admission.threadblock_payload_manifest_sha256
    )
    assert record["trace_admission_sha256"] == admission.trace_admission_sha256
    assert merge_evidence._validate_nvbit_summary_stdlib(record) is None

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (SUITE_ROOT / "schemas" / "nvbit_trace_summary.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(record, schema)


def test_summary_fallback_dispatches_historical_v1_and_admitted_v2() -> None:
    fixture = (
        SUITE_ROOT
        / "golden"
        / "cupti_nvbit_join_fixtures"
        / "medium_rmsnorm_bf16"
        / "nvbit_trace_summary.json"
    )
    v1 = json.loads(fixture.read_text(encoding="utf-8"))
    v1["nvbit_tracer_build_hash"] = "retained-legacy-build-id"
    assert merge_evidence._validate_nvbit_summary_stdlib(v1) is None

    v1_with_admission_field = {
        **v1,
        "trace_admission_sha256": "a" * 64,
    }
    assert "additionalProperties" in str(
        merge_evidence._validate_nvbit_summary_stdlib(v1_with_admission_field)
    )

    v2 = {
        **v1,
        "schema_version": 2,
        "nvbit_tracer_build_hash": "a" * 64,
        "nvbit_threadblock_payload_size_bytes": 42,
        "nvbit_threadblock_payload_manifest_sha256": "b" * 64,
        "trace_admission_sha256": "c" * 64,
    }
    assert merge_evidence._validate_nvbit_summary_stdlib(v2) is None
    del v2["trace_admission_sha256"]
    assert "missing required field" in str(
        merge_evidence._validate_nvbit_summary_stdlib(v2)
    )


def test_admitted_summary_refuses_planted_output_symlink(tmp_path: Path) -> None:
    admission = require_trace_admission(_write_trace(tmp_path / "trace"))
    target = tmp_path / "target.json"
    target.write_bytes(b"do not truncate")
    output = tmp_path / "summary.json"
    output.symlink_to(target)

    with pytest.raises(ImmutableArtifactError, match="single-link regular file"):
        adapter.emit_admitted_nvbit_trace_summary(
            admission=admission,
            slug="fixture",
            shape_point="fixture_shape",
            repeat_index=0,
            launch_index=0,
            binary_sha256=hashlib.sha256(b"executable").hexdigest(),
            tracer_build_sha256=hashlib.sha256(b"tracer").hexdigest(),
            output_path=output,
        )

    assert target.read_bytes() == b"do not truncate"
    assert output.is_symlink()


def test_run_suite_refuses_planted_canonical_trace_record_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"do not truncate")
    output = tmp_path / "trace_fixture.json"
    output.symlink_to(target)

    with pytest.raises(ImmutableArtifactError, match="publication already exists"):
        run_suite._publish_trace_artifact_record(
            output,
            {"schema_version": 1, "slug": "fixture"},
        )

    assert target.read_bytes() == b"do not truncate"
    assert output.is_symlink()


def test_run_suite_emits_from_retained_digests_across_launches(
    tmp_path: Path,
) -> None:
    admission = require_trace_admission(_write_trace(tmp_path / "trace"))
    second_launch = replace(admission.launches[0], launch_index=1)
    admission = replace(
        admission,
        launch_count=2,
        launches=(admission.launches[0], second_launch),
    )
    binary_sha256 = hashlib.sha256(b"executed bytes").hexdigest()
    tracer_build_sha256 = hashlib.sha256(b"injected bytes").hexdigest()
    result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256=binary_sha256,
        tracer_build_sha256=tracer_build_sha256,
        artifact_dir=tmp_path / "summaries",
    )

    assert result["status"] == "ok"
    assert result["paths"] == _summary_generation_relative_paths(
        admission,
        shape_point="fixture_shape",
    )
    for relative_path in result["paths"]:
        record = json.loads(
            (tmp_path / "summaries" / relative_path).read_text(encoding="utf-8")
        )
        assert record["binary_sha256"] == binary_sha256
        assert record["nvbit_tracer_build_hash"] == tracer_build_sha256


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("slug", "other_fixture"),
        ("shape_point", "other_shape"),
        ("kernel_name", "other_kernel"),
        ("kernel_signature_sha256", "c" * 64),
        ("inst_count", 2),
        ("binary_sha256", "c" * 64),
        ("nvbit_trace_pb_path", "/other/dynamic_trace.pb"),
        ("nvbit_trace_pb_sha256", "c" * 64),
        ("nvbit_tracer_build_hash", "c" * 64),
        ("nvbit_threadblock_payload_size_bytes", 27),
        ("nvbit_threadblock_payload_manifest_sha256", "c" * 64),
        ("trace_admission_sha256", "c" * 64),
        (None, True),
    ),
)
def test_producer_activation_rejects_every_contradictory_v2_field(
    tmp_path: Path,
    field: str | None,
    replacement: object,
) -> None:
    admission = require_trace_admission(_write_trace(tmp_path / "trace"))
    artifact_dir = tmp_path / "summaries"
    result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )
    assert result["status"] == "ok"
    summary_path = artifact_dir / result["paths"][0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if field is None:
        summary["unexpected_field"] = replacement
    else:
        summary[field] = replacement
    summary_path.chmod(0o644)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary_path.chmod(0o444)

    with pytest.raises(
        ImmutableArtifactError,
        match="disagrees with exact admitted launch and execution identity",
    ):
        run_suite._require_one_committed_nvbit_summary_generation(
            emit_result=result,
            trace_admission=admission,
            slug="fixture",
            shape_point="fixture_shape",
            binary_sha256="a" * 64,
            tracer_build_sha256="b" * 64,
            artifact_dir=artifact_dir,
        )


def test_run_suite_two_shapes_coexist_and_active_references_find_both(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "summaries"
    results: dict[str, dict] = {}
    for shape_point in ("trace_small", "trace_large"):
        _admission, result, _record_path = _publish_active_summary_fixture(
            artifact_dir=artifact_dir,
            source_root=_write_trace(tmp_path / f"source_{shape_point}"),
            shape_point=shape_point,
        )
        results[shape_point] = result

    first_path = results["trace_small"]["paths"][0]
    second_path = results["trace_large"]["paths"][0]
    assert first_path != second_path
    assert (artifact_dir / first_path).is_file()
    assert (artifact_dir / second_path).is_file()

    # A legacy flat file and an unrelated partial generation are not active
    # once canonical trace-record references exist.
    stale_flat = artifact_dir / "nvbit_trace_summary_0_99.json"
    stale_flat.write_text('{"stale": true}\n', encoding="utf-8")
    partial = (
        artifact_dir
        / run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE
        / "unreferenced_shape"
        / ("c" * 64)
        / ".dddd.staging"
    )
    partial.mkdir(parents=True)
    (partial / "nvbit_trace_summary_0_0.json").write_text(
        '{"partial": true}\n', encoding="utf-8"
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert errors == []
    assert discovered == sorted(
        [artifact_dir / first_path, artifact_dir / second_path]
    )


def test_generation_identity_includes_local_filesystem_generation(
    tmp_path: Path,
) -> None:
    first_admission = require_trace_admission(_write_trace(tmp_path / "trace"))
    replacement_generation = (
        "0" * 64
        if first_admission.trace_filesystem_generation_sha256 != "0" * 64
        else "1" * 64
    )
    second_admission = replace(
        first_admission,
        trace_filesystem_generation_sha256=replacement_generation,
    )
    assert (
        second_admission.trace_admission_sha256
        == first_admission.trace_admission_sha256
    )

    artifact_dir = tmp_path / "summaries"
    first = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=first_admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )
    second = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=second_admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )

    assert first["status"] == second["status"] == "ok"
    assert first["paths"] != second["paths"]
    assert (artifact_dir / first["paths"][0]).is_file()
    assert (artifact_dir / second["paths"][0]).is_file()


def test_merge_refuses_legacy_fallback_beneath_summary_generation_namespace(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "nvbit_trace_summary_0_0.json"
    legacy.write_text('{"legacy": true}\n', encoding="utf-8")
    generation_root = (
        tmp_path
        / run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE
        / "trace_shape"
        / ("a" * 64)
    )
    published = immutable.publish_directory(
        generation_root,
        "b" * 64,
        [
            immutable.ImmutableFile.from_bytes(
                "nvbit_trace_summary_0_0.json",
                b'{"unreferenced": true}\n',
            )
        ],
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        tmp_path
    )
    assert discovered == []
    assert len(errors) == 1
    assert "generation_authority_missing" in errors[0]

    descendant_paths, descendant_errors = (
        merge_evidence._discover_nvbit_summary_inputs(published)
    )
    assert descendant_paths == []
    assert len(descendant_errors) == 1
    assert "hidden or reserved evidence namespace" in descendant_errors[0]


@pytest.mark.parametrize(
    "namespace_name",
    (
        run_suite.RAW_TRACE_GENERATION_NAMESPACE,
        run_suite.TRACE_DERIVED_GENERATION_NAMESPACE,
        run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE,
    ),
)
def test_merge_refuses_legacy_fallback_when_generation_namespace_exists(
    tmp_path: Path,
    namespace_name: str,
) -> None:
    legacy = tmp_path / "nvbit_trace_summary_0_0.json"
    legacy.write_text('{"legacy": true}\n', encoding="utf-8")
    immutable.ensure_directory(
        tmp_path
        / namespace_name
        / "trace_shape"
        / "attempt-preserved-failure"
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        tmp_path
    )

    assert discovered == []
    assert len(errors) == 1
    assert "generation_authority_missing" in errors[0]
    assert "generation namespace" in errors[0]


def test_merge_preserves_flat_standalone_legacy_discovery(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "nested" / "nvbit_trace_summary_0_0.json"
    legacy.parent.mkdir()
    legacy.write_text('{"legacy": true}\n', encoding="utf-8")

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        tmp_path
    )
    assert errors == []
    assert discovered == [legacy]


@pytest.mark.parametrize(
    ("node_name", "node_kind"),
    (
        ("trace_fixture.json", "directory"),
        ("nvbit_trace_summary_1_0.json", "directory"),
        (run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE, "file"),
        (run_suite.RAW_TRACE_GENERATION_NAMESPACE, "file"),
        (run_suite.TRACE_DERIVED_GENERATION_NAMESPACE, "file"),
        (run_suite.PHASE_LEDGER_GENERATION_NAMESPACE, "file"),
        (run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE, "file"),
    ),
)
def test_merge_wrong_authority_node_type_blocks_legacy_fallback(
    tmp_path: Path,
    node_name: str,
    node_kind: str,
) -> None:
    legacy = tmp_path / "nvbit_trace_summary_0_0.json"
    legacy.write_text('{"schema_version": 1}\n', encoding="utf-8")
    candidate = tmp_path / node_name
    if node_kind == "directory":
        candidate.mkdir()
    else:
        candidate.write_bytes(b"wrong authority node type\n")

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        tmp_path
    )

    assert discovered == []
    assert any(
        "authority_node_invalid" in error and str(candidate) in error
        for error in errors
    )


@pytest.mark.parametrize("limit_kind", ("per_file", "aggregate"))
def test_join_preload_limits_fire_before_the_first_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    paths = [
        tmp_path / "cupti_activity_0_0.json",
        tmp_path / "cupti_activity_0_1.json",
    ]
    for path in paths:
        path.write_bytes(b"{}\n")

    with merge_evidence.PinnedDirectoryReader(tmp_path) as reader:
        snapshot = merge_evidence._snapshot_join_input_tree(
            tmp_path,
            reader=reader,
        )
        if limit_kind == "per_file":
            monkeypatch.setattr(
                merge_evidence,
                "_JOIN_PRELOAD_MAX_FILE_BYTES",
                2,
            )
            monkeypatch.setattr(
                merge_evidence,
                "_JOIN_PRELOAD_MAX_TOTAL_BYTES",
                100,
            )
        else:
            monkeypatch.setattr(
                merge_evidence,
                "_JOIN_PRELOAD_MAX_FILE_BYTES",
                10,
            )
            monkeypatch.setattr(
                merge_evidence,
                "_JOIN_PRELOAD_MAX_TOTAL_BYTES",
                5,
            )
        read_count = 0
        original_read_snapshot = (
            merge_evidence.PinnedDirectoryReader.read_snapshot
        )

        def track_read(*args: object, **kwargs: object):
            nonlocal read_count
            read_count += 1
            return original_read_snapshot(*args, **kwargs)

        monkeypatch.setattr(
            merge_evidence.PinnedDirectoryReader,
            "read_snapshot",
            track_read,
        )

        with pytest.raises(ImmutableArtifactLimitError):
            merge_evidence._preload_join_input_bytes(
                tmp_path,
                reader,
                snapshot,
                paths,
                preload_budget=merge_evidence._JoinPreloadBudget(),
            )

        assert read_count == 0


def test_join_schema_read_is_bounded_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_bytes(b"{}\n")
    monkeypatch.setattr(
        merge_evidence,
        "_JOIN_PRELOAD_MAX_FILE_BYTES",
        2,
    )
    decoded = False

    def reject_decode(*_args: object, **_kwargs: object) -> object:
        nonlocal decoded
        decoded = True
        raise AssertionError("oversized schema reached JSON decode")

    monkeypatch.setattr(merge_evidence, "strict_json_loads", reject_decode)
    with pytest.raises(ImmutableArtifactLimitError):
        merge_evidence._load_join_schema(schema_path, role="fixture schema")
    assert decoded is False


def test_canonical_candidate_replacement_is_rejected_before_unbounded_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / "nvbit_trace_summary_0_0.json"
    legacy.write_text('{"schema_version": 1}\n', encoding="utf-8")
    candidate = tmp_path / "trace_fixture.json"
    candidate.write_text("[]\n", encoding="utf-8")
    original_discover = merge_evidence._discover_json_files
    original_hash_fd = immutable._hash_fd
    replaced = False

    def replace_after_inventory(*args: object, **kwargs: object) -> list[Path]:
        nonlocal replaced
        paths = original_discover(*args, **kwargs)
        if kwargs.get("tree_snapshot") is not None and "trace_*.json" in args:
            replacement = tmp_path / ".large-replacement"
            with replacement.open("wb") as stream:
                stream.truncate(
                    merge_evidence._JOIN_PRELOAD_MAX_FILE_BYTES + 1
                )
            os.replace(replacement, candidate)
            replaced = True
        return paths

    def reject_large_hash(descriptor: int):
        if os.fstat(descriptor).st_size > (
            merge_evidence._JOIN_PRELOAD_MAX_FILE_BYTES
        ):
            raise AssertionError("replacement was hashed before bounded admission")
        return original_hash_fd(descriptor)

    monkeypatch.setattr(
        merge_evidence,
        "_discover_json_files",
        replace_after_inventory,
    )
    monkeypatch.setattr(immutable, "_hash_fd", reject_large_hash)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        tmp_path
    )

    assert replaced is True
    assert discovered == []
    assert errors


def test_rejected_manifest_does_not_retain_raw_source_watches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    active["measurement_manifest"]["sha256"] = "0" * 64
    _replace_active_record_for_hostile_fixture(active_path, active)
    retained_open_count = 0

    def reject_retained_open(*_args: object, **_kwargs: object):
        nonlocal retained_open_count
        retained_open_count += 1
        raise AssertionError("rejected candidate retained raw source authority")

    monkeypatch.setattr(
        merge_evidence,
        "open_admitted_trace_sources",
        reject_retained_open,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )

    assert discovered == []
    assert any("measurement_manifest_invalid" in error for error in errors)
    assert retained_open_count == 0


@pytest.mark.parametrize("generation_kind", ("summary", "phase"))
def test_generation_budget_preflight_precedes_content_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_kind: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    budget = merge_evidence._JoinPreloadBudget(
        total_bytes=merge_evidence._JOIN_PRELOAD_MAX_TOTAL_BYTES,
    )

    def reject_generation_hash(*_args: object, **_kwargs: object):
        raise AssertionError("generation hashing preceded preload admission")

    monkeypatch.setattr(
        merge_evidence,
        "inspect_immutable_directory_generation",
        reject_generation_hash,
    )
    if generation_kind == "summary":
        with pytest.raises(ImmutableArtifactLimitError):
            merge_evidence._active_summary_generation_descriptor_error(
                active_path,
                active,
                trace_admission=admission,
                preload_budget=budget,
                materialization_budget=merge_evidence._JoinMaterializationBudget(),
            )
        return

    attempt_id = active["trace_capture_attempt_id"]
    relative = Path(
        run_suite.PHASE_LEDGER_GENERATION_NAMESPACE,
        "fixture_shape",
        attempt_id,
    )
    active["phase_inject_ab"] = {
        "requested": True,
        "repeat_count": 1,
        "generation": {
            "schema_version": 1,
            "path": relative.as_posix(),
            "attempt_id": attempt_id,
            "content_manifest_sha256": "0" * 64,
            "filesystem_generation_sha256": "0" * 64,
            "file_count": 1,
            "directory_count": 1,
            "total_file_bytes": 1,
        },
        "evidence": [
            {
                "path": (
                    relative / "phase_cycle_ledger_0_0.json"
                ).as_posix(),
                "sha256": "0" * 64,
                "size_bytes": 1,
            }
        ],
    }
    with pytest.raises(ImmutableArtifactLimitError):
        merge_evidence._active_phase_generation_binding_error(
            active_path,
            active,
            phase_namespace_entries=[],
            trace_admission=admission,
            preload_budget=budget,
            materialization_budget=merge_evidence._JoinMaterializationBudget(),
        )


@pytest.mark.parametrize("consumer", ("merge", "verifier"))
def test_generation_actual_per_file_limit_precedes_content_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace_with_launch_count(tmp_path / "trace", 2),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    descriptor = active["nvbit_trace_summary_generation"]
    assert isinstance(descriptor, dict)
    files = descriptor["files"]
    assert isinstance(files, list) and len(files) == 2
    padded_path = artifact_dir / str(files[0]["path"])
    generation_path = padded_path.parent
    generation_path.chmod(0o755)
    padded_path.chmod(0o644)
    try:
        padded_path.write_bytes(padded_path.read_bytes() + b" " * 4096)
        padded_path.chmod(0o444)
    finally:
        generation_path.chmod(0o555)
    _reseal_summary_generation_authority(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
    )
    descriptor = active["nvbit_trace_summary_generation"]
    assert isinstance(descriptor, dict)
    files = descriptor["files"]
    assert isinstance(files, list) and len(files) == 2
    actual_sizes = [int(item["size_bytes"]) for item in files]
    per_file_limit = max(actual_sizes) - 1
    total_file_bytes = int(descriptor["total_file_bytes"])
    assert total_file_bytes <= per_file_limit * len(files)
    files[0]["size_bytes"] = per_file_limit
    files[1]["size_bytes"] = total_file_bytes - per_file_limit
    assert all(int(item["size_bytes"]) <= per_file_limit for item in files)

    original_hash_fd = immutable._hash_fd

    def reject_summary_hash(descriptor_fd: int):
        target = os.readlink(f"/proc/self/fd/{descriptor_fd}")
        if run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE in target:
            raise AssertionError("summary content hashing preceded actual-size limit")
        return original_hash_fd(descriptor_fd)

    monkeypatch.setattr(immutable, "_hash_fd", reject_summary_hash)
    if consumer == "merge":
        monkeypatch.setattr(
            merge_evidence,
            "_JOIN_PRELOAD_MAX_FILE_BYTES",
            per_file_limit,
        )
        error = merge_evidence._active_summary_generation_descriptor_error(
            active_path,
            active,
            trace_admission=admission,
            preload_budget=merge_evidence._JoinPreloadBudget(),
            materialization_budget=merge_evidence._JoinMaterializationBudget(),
        )
        assert error is not None
        errors = [error]
    else:
        monkeypatch.setattr(
            verify_artifacts,
            "_CANONICAL_AUTHORITY_JSON_MAX_BYTES",
            per_file_limit,
        )
        errors = verify_artifacts._verify_active_summary_generation(
            active_path,
            active,
            trace_admission=admission,
        )

    assert any("bounded per-file byte limit" in error for error in errors)



def test_merge_preserves_malformed_retained_legacy_bytes(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "nvbit_trace_summary_0_0.json"
    legacy.write_text(
        '{"schema_version":1,"schema_version":1}\n',
        encoding="utf-8",
    )
    with ExitStack() as source_stack:
        retained_authorities: list[object] = []
        paths, errors, payloads = (
            merge_evidence._discover_nvbit_summary_inputs_with_records(
                tmp_path,
                source_stack=source_stack,
                retained_authorities=retained_authorities,
            )
        )
        assert errors == []
        assert paths == [legacy]
        for authority in retained_authorities:
            authority.verify_unchanged()

    legacy.write_text("{}\n", encoding="utf-8")
    _cupti_schema, nvbit_schema, _manifest_schema = (
        merge_evidence._load_join_schemas()
    )
    records, schema_errors = (
        merge_evidence._index_records_by_natural_key_with_counts(
            paths,
            nvbit_schema,
            "nvbit",
            preloaded_payloads=payloads,
        )
    )

    assert records == {}
    assert len(schema_errors) == 1
    assert "duplicate JSON key" in schema_errors[0]


def test_merge_discovery_fails_closed_at_bounded_inventory_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "nvbit_trace_summary_0_0.json").write_text(
        '{"schema_version": 1}\n', encoding="utf-8"
    )
    (tmp_path / "unrelated.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(merge_evidence, "_JOIN_DISCOVERY_MAX_FILES", 1)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(tmp_path)

    assert discovered == []
    assert len(errors) == 1
    assert "discovery_limit_exceeded" in errors[0]


def test_merge_rejects_parent_root_nested_unreferenced_staging_v2(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "join_root"
    staging = (
        parent
        / "cell"
        / run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE
        / "fixture_shape"
        / ("a" * 64)
        / ("b" * 64)
        / ".interrupted.staging"
    )
    staging.mkdir(parents=True)
    (staging / "nvbit_trace_summary_0_0.json").write_text(
        json.dumps({"schema_version": 2}) + "\n", encoding="utf-8"
    )
    legacy = parent / "nvbit_trace_summary_0_0.json"
    legacy.write_text('{"schema_version": 1}\n', encoding="utf-8")

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(parent)

    assert discovered == []
    assert len(errors) == 1
    assert "generation_authority_missing" in errors[0]
    assert "standalone-summary fallback" in errors[0]


def test_merge_rejects_standalone_admitted_v2_summary(tmp_path: Path) -> None:
    admission = require_trace_admission(_write_trace(tmp_path / "trace"))
    summary_path = tmp_path / "nvbit_trace_summary_0_0.json"
    adapter.emit_admitted_nvbit_trace_summary(
        admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        repeat_index=0,
        launch_index=0,
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        output_path=summary_path,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(tmp_path)

    assert discovered == []
    assert len(errors) == 1
    assert "generation_authority_missing" in errors[0]
    assert "schema_version 2" in errors[0]


def test_failed_rerun_attempt_namespace_blocks_stale_flat_v1_reactivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    fixture = (
        SUITE_ROOT
        / "golden"
        / "cupti_nvbit_join_fixtures"
        / "medium_rmsnorm_bf16"
        / "nvbit_trace_summary.json"
    )
    stale = artifact_dir / "nvbit_trace_summary_0_0.json"
    shutil.copyfile(fixture, stale)
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    row["trace_budget_mib"] = 0
    (artifact_dir / "trace_trace_00.json").write_text(
        '{"previous": "activation"}\n', encoding="utf-8"
    )

    result = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    assert result == run_suite.EXIT_SAMPLING_REQUIRED
    assert not (artifact_dir / "trace_trace_00.json").exists()
    assert (artifact_dir / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE).is_dir()
    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "generation_authority_missing" in errors[0]
    assert stale.is_file()


def test_merge_binds_committed_summary_body_to_active_trace_record(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "summaries"
    _admission, result, _record_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
        record_slug="wrong_slug",
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "generation_binding_invalid" in errors[0]
    assert "slug" in errors[0]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("duplicate", "duplicate JSON key"),
        ("nonfinite", "non-finite JSON number"),
        ("deep", "invalid active NVBit summary"),
    ),
)
def test_merge_strictly_decodes_active_summary_payload(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    summary_path = artifact_dir / str(result["paths"][0])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        body = json.dumps(summary, separators=(",", ":"), sort_keys=True)
        payload = ("{\"schema_version\":2," + body[1:] + "\n").encode(
            "utf-8"
        )
    elif mutation == "nonfinite":
        summary["inst_count"] = float("nan")
        payload = (json.dumps(summary, sort_keys=True) + "\n").encode("utf-8")
    else:
        payload = ("[" * 10_000 + "0" + "]" * 10_000).encode("utf-8")
    generation_path = summary_path.parent
    generation_path.chmod(0o755)
    try:
        summary_path.chmod(0o644)
        summary_path.write_bytes(payload)
        summary_path.chmod(0o444)
    finally:
        generation_path.chmod(0o555)
    _reseal_summary_generation_authority(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )

    assert discovered == []
    assert len(errors) == 1
    assert "generation_binding_invalid" in errors[0]
    assert expected in errors[0]


def test_merge_rejects_legacy_typed_unavailable_as_active_authority(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "nvbit_trace_summary_0_0.json"
    stale.write_text('{"stale": true}\n', encoding="utf-8")
    active = {
        "schema_version": 1,
        "slug": "fixture",
        "shape_set": "trace_shapes",
        "shape_point": "fixture_shape",
        "result_kind": "typed_unavailable",
        "typed_reason": "arch_feature_unavailable",
        "nvbit_trace_summary_emit_status": "typed_unavailable",
        "nvbit_trace_summary_paths": [],
    }
    immutable.publish_regular_file(
        tmp_path,
        "trace_fixture_shape.json",
        (json.dumps(active, indent=2) + "\n").encode("utf-8"),
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        tmp_path
    )
    assert discovered == []
    assert len(errors) == 1
    assert "explicitly active canonical trace record" in errors[0]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("missing_file", "absent from retained authority inventory"),
        ("path_substitution", "does not name the exact active shape"),
        ("sha256_drift", "descriptor is stale"),
        ("size_drift", "descriptor is stale"),
    ),
)
def test_merge_requires_exact_measurement_manifest_identity(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest_path = artifact_dir / str(descriptor["path"])
    if mutation == "missing_file":
        immutable.retire_regular_file(manifest_path)
    elif mutation == "path_substitution":
        substitute = "measurement_manifest_other_shape.json"
        immutable.publish_regular_file(
            artifact_dir,
            substitute,
            manifest_path.read_bytes(),
        )
        descriptor["path"] = substitute
    elif mutation == "sha256_drift":
        descriptor["sha256"] = "c" * 64
    else:
        descriptor["size_bytes"] = int(descriptor["size_bytes"]) + 1
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )

    assert discovered == []
    assert len(errors) == 1
    assert "measurement_manifest_invalid" in errors[0]
    assert expected in errors[0]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("schema_version", "fails closed v2 schema"),
        ("integral_float_schema_version", "fails closed v2 schema"),
        ("extra_property", "fails closed v2 schema"),
        ("mode", "field 'mode' disagrees"),
        ("slug", "field 'slug' disagrees"),
        ("shape_point", "field 'shape_point' disagrees"),
        ("binary_sha256", "field 'binary_sha256' disagrees"),
        (
            "kernel_signature_sha256",
            "field 'kernel_signature_sha256' disagrees",
        ),
        ("timestamp_utc", "fails closed v2 schema"),
    ),
)
def test_merge_schema_and_identity_bind_measurement_manifest(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest = json.loads(
        (artifact_dir / str(descriptor["path"])).read_text(encoding="utf-8")
    )
    if mutation == "schema_version":
        manifest[mutation] = 1
    elif mutation == "integral_float_schema_version":
        manifest["schema_version"] = 2.0
    elif mutation == "extra_property":
        manifest[mutation] = "forbidden"
    elif mutation == "mode":
        manifest[mutation] = "timing"
    elif mutation in {"slug", "shape_point"}:
        manifest[mutation] = "other"
    else:
        manifest[mutation] = "c" * 64
    payload = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    _replace_measurement_manifest_for_hostile_fixture(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
        payload=payload,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )

    assert discovered == []
    assert len(errors) == 1
    assert "measurement_manifest_invalid" in errors[0]
    assert expected in errors[0]
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any(
        "measurement manifest" in error and expected in error
        for error in verifier_errors
    )


def test_merge_binds_measurement_manifest_gpu_sku_to_requested_platform(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest = json.loads(
        (artifact_dir / str(descriptor["path"])).read_text(encoding="utf-8")
    )
    manifest["gpu_sku"] = "rtx_pro_6000_blackwell"
    payload = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    _replace_measurement_manifest_for_hostile_fixture(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
        payload=payload,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )

    assert discovered == []
    assert len(errors) == 1
    assert "measurement_manifest_invalid" in errors[0]
    assert "field 'gpu_sku' disagrees" in errors[0]


def test_verifier_binds_measurement_manifest_gpu_sku_to_canonical_path(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "rtx_3090_ti" / "sm_86" / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest = json.loads(
        (artifact_dir / str(descriptor["path"])).read_text(encoding="utf-8")
    )
    manifest["gpu_sku"] = "rtx_pro_6000_blackwell"
    payload = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    _replace_measurement_manifest_for_hostile_fixture(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
        payload=payload,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any(
        "measurement manifest field 'gpu_sku' disagrees" in error
        for error in errors
    )


def test_verifier_rejects_multiple_canonical_platform_path_authorities(
    tmp_path: Path,
) -> None:
    artifact_dir = (
        tmp_path
        / "rtx_3090_ti"
        / "rtx_pro_6000_blackwell"
        / "sm_86"
        / "artifact"
    )
    _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any(
        "multiple canonical platform authorities" in error for error in errors
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("duplicate", "duplicate JSON key"),
        ("nonfinite", "non-finite JSON number"),
        ("deep", "invalid measurement manifest"),
    ),
)
def test_merge_strictly_decodes_measurement_manifest(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest = json.loads(
        (artifact_dir / str(descriptor["path"])).read_text(encoding="utf-8")
    )
    if mutation == "duplicate":
        body = json.dumps(manifest, separators=(",", ":"), sort_keys=True)
        payload = ("{\"schema_version\":2," + body[1:] + "\n").encode(
            "utf-8"
        )
    elif mutation == "nonfinite":
        manifest["cv_pct"] = float("nan")
        payload = (json.dumps(manifest, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    else:
        payload = ("[" * 10_000 + "0" + "]" * 10_000).encode("utf-8")
    _replace_measurement_manifest_for_hostile_fixture(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
        payload=payload,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )

    assert discovered == []
    assert len(errors) == 1
    assert "measurement_manifest_invalid" in errors[0]
    assert expected in errors[0]
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any("measurement manifest" in error for error in verifier_errors)
    if mutation != "deep":
        assert any(expected in error for error in verifier_errors)


def test_merge_types_measurement_manifest_memory_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    original = merge_evidence.strict_json_loads

    def fail_manifest_decode(payload: bytes, *, label: str = "JSON") -> object:
        if label == "measurement manifest":
            raise MemoryError("synthetic manifest allocation exhaustion")
        return original(payload, label=label)

    monkeypatch.setattr(
        merge_evidence,
        "strict_json_loads",
        fail_manifest_decode,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )

    assert discovered == []
    assert len(errors) == 1
    assert "measurement_manifest_resource_unavailable" in errors[0]
    assert "synthetic manifest allocation exhaustion" in errors[0]


@pytest.mark.parametrize("error_number", (errno.EMFILE, errno.ENFILE))
def test_merge_types_measurement_manifest_descriptor_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    original = merge_evidence.PinnedDirectoryReader.read_snapshot

    def fail_manifest_read(
        reader,
        relative_path: str,
        **kwargs,
    ):
        if PurePosixPath(relative_path).name.startswith("measurement_manifest_"):
            raise OSError(error_number, "synthetic descriptor exhaustion")
        return original(reader, relative_path, **kwargs)

    monkeypatch.setattr(
        merge_evidence.PinnedDirectoryReader,
        "read_snapshot",
        fail_manifest_read,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )

    assert discovered == []
    assert len(errors) == 1
    assert "measurement_manifest_resource_unavailable" in errors[0]
    assert "synthetic descriptor exhaustion" in errors[0]


@pytest.mark.parametrize("consumer", ("merge", "verifier"))
@pytest.mark.parametrize("error_number", (errno.EMFILE, errno.ENFILE))
def test_consumers_type_wrapped_manifest_open_exhaustion_as_resource_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    error_number: int,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest_name = str(descriptor["path"])
    real_open = immutable.os.open

    def fail_manifest_open(path, flags, *args, **kwargs):
        if PurePosixPath(os.fspath(path)).name == manifest_name:
            raise OSError(error_number, "synthetic wrapped descriptor exhaustion")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(immutable.os, "open", fail_manifest_open)

    if consumer == "merge":
        discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
            artifact_dir
        )
        assert discovered == []
    else:
        errors = verify_artifacts._verify_canonical_trace_pb_admission(
            artifact_dir
        )

    assert any(
        "ResourceUnavailable" in error or "resource_unavailable" in error
        for error in errors
    ), errors
    assert any(
        "synthetic wrapped descriptor exhaustion" in error for error in errors
    )


def test_handle_trace_typed_unavailable_record_is_non_active_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "global_scalar_32b_copy_run"
    executable.write_bytes(b"fixture executable")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"fixture tracer")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    stale = artifact_dir / "nvbit_trace_summary_0_0.json"
    stale.write_text('{"stale": true}\n', encoding="utf-8")

    monkeypatch.setattr(run_suite, "NVBIT_TRACER_SO", tracer)
    monkeypatch.setattr(
        run_suite,
        "_find_kernel_executable",
        lambda _slug: executable,
    )
    monkeypatch.setattr(
        run_suite,
        "_artifact_dir_for_row",
        lambda _row, _props: artifact_dir,
    )
    monkeypatch.setattr(
        run_suite,
        "_run_retained_nvbit_capture",
        lambda **_kwargs: {
            "baseline_proc": SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    "result_kind": "typed_unavailable",
                    "typed_reason": "arch_feature_unavailable",
                }),
                stderr="",
            ),
            "baseline_elapsed_s": 0.001,
            "baseline_result": {
                "result_kind": "typed_unavailable",
                "typed_reason": "arch_feature_unavailable",
            },
            "trace_proc": None,
            "trace_elapsed_s": None,
            "binary_sha256": "a" * 64,
            "tracer_build_sha256": "b" * 64,
        },
    )
    props = run_suite.DeviceProps()
    row = {
        "slug": "global_scalar_32b_copy",
        "trace_budget_mib": 64,
        "seed_policy": {"seed": 42},
        "arch_gate": ["sm_86"],
    }

    rc = run_suite.handle_trace(
        row,
        props,
        tmp_path,
        shape_point_name="trace_00",
    )

    assert rc == run_suite.EXIT_OK
    record = json.loads(
        (artifact_dir / "trace_trace_00.json").read_text(encoding="utf-8")
    )
    assert record["shape_set"] == "trace_shapes"
    assert record["shape_point"] == "trace_00"
    assert record["activation_state"] == "non_active"
    assert record["nvbit_trace_summary_emit_status"] == "typed_unavailable"
    assert record["nvbit_trace_summary_paths"] == []
    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "does not form an explicitly active canonical trace record" in errors[0]
    assert stale.exists()


def test_run_suite_summary_batch_failure_leaves_no_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = require_trace_admission(_write_trace(tmp_path / "trace"))
    second_launch = replace(admission.launches[0], launch_index=1)
    admission = replace(
        admission,
        launch_count=2,
        launches=(admission.launches[0], second_launch),
    )
    artifact_dir = tmp_path / "summaries"
    artifact_dir.mkdir()
    legacy = artifact_dir / "nvbit_trace_summary_0_9.json"
    legacy_payload = b'{"legacy": true}\n'
    legacy.write_bytes(legacy_payload)
    original_emit = adapter.emit_admitted_nvbit_trace_summary

    def fail_second_launch(**kwargs):
        if kwargs["launch_index"] == 1:
            raise RuntimeError("planted second-launch failure")
        return original_emit(**kwargs)

    monkeypatch.setattr(
        adapter,
        "emit_admitted_nvbit_trace_summary",
        fail_second_launch,
    )
    result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )

    assert result["status"] == "adapter_error"
    assert result["paths"] == []
    assert legacy.read_bytes() == legacy_payload
    assert not (
        artifact_dir
        / run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE
    ).exists()
    assert not tuple(artifact_dir.glob(".*.retired"))


@pytest.mark.parametrize(
    ("resource_error", "expected_detail"),
    (
        (
            ImmutableArtifactResourceError(
                "VERIFY_RESOURCE_BOUND: inotify event queue overflowed"
            ),
            "inotify event queue overflowed",
        ),
        (
            MemoryError("synthetic summary allocation exhaustion"),
            "synthetic summary allocation exhaustion",
        ),
    ),
    ids=("immutable-resource", "memory-error"),
)
def test_run_suite_summary_resource_failure_is_typed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_error: BaseException,
    expected_detail: str,
) -> None:
    admission = require_trace_admission(_write_trace(tmp_path / "trace"))

    def fail_with_resource_error(**_kwargs) -> None:
        raise resource_error

    monkeypatch.setattr(
        adapter,
        "emit_admitted_nvbit_trace_summary",
        fail_with_resource_error,
    )
    result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=tmp_path / "summaries",
    )

    assert result["status"] == "resource_unavailable"
    assert result["failed_hard"] is True
    assert result["paths"] == []
    assert "trace_summary_resource_unavailable" in result["failure_reason"]
    assert expected_detail in result["failure_reason"]


def test_run_suite_summary_generation_refuses_exact_destination_reuse(
    tmp_path: Path,
) -> None:
    admission = require_trace_admission(_write_trace(tmp_path / "trace"))
    artifact_dir = tmp_path / "summaries"
    kwargs = {
        "trace_admission": admission,
        "slug": "fixture",
        "shape_point": "fixture_shape",
        "binary_sha256": "a" * 64,
        "tracer_build_sha256": "b" * 64,
        "artifact_dir": artifact_dir,
    }
    first = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(**kwargs)
    assert first["status"] == "ok"
    generation_path = artifact_dir / Path(first["paths"][0]).parent
    before = immutable.inspect_immutable_directory_generation(
        generation_path,
        max_files=1,
        max_directories=1,
        max_total_file_bytes=256 * 1024 * 1024,
        max_total_relative_path_bytes=4 * 1024 * 1024,
    )

    second = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(**kwargs)

    assert second["status"] == "adapter_error"
    assert second["paths"] == []
    assert "already exists or raced" in second["failure_reason"]
    assert immutable.inspect_immutable_directory_generation(
        generation_path,
        max_files=before.file_count,
        max_directories=before.directory_count,
        max_total_file_bytes=before.total_file_bytes,
        max_total_relative_path_bytes=4 * 1024 * 1024,
    ) == before


def test_run_suite_smaller_rerun_retains_but_does_not_select_old_launches(
    tmp_path: Path,
) -> None:
    one_launch = require_trace_admission(_write_trace(tmp_path / "trace"))
    launches = tuple(
        replace(one_launch.launches[0], launch_index=launch_index)
        for launch_index in range(3)
    )
    three_launches = replace(
        one_launch,
        launch_count=3,
        launches=launches,
    )
    artifact_dir = tmp_path / "summaries"

    first = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=three_launches,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )
    second = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=one_launch,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert len(first["paths"]) == 3
    assert len(second["paths"]) == 1
    assert set(first["paths"]).isdisjoint(second["paths"])
    for relative_path in [*first["paths"], *second["paths"]]:
        assert (artifact_dir / relative_path).is_file()

    _active_admission, active_result, _record_path = (
        _publish_active_summary_fixture(
            artifact_dir=artifact_dir,
            source_root=_write_trace(tmp_path / "active_trace"),
            shape_point="fixture_shape",
        )
    )
    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert errors == []
    assert discovered == [artifact_dir / active_result["paths"][0]]


def test_run_suite_feature_batch_rejects_restored_trace_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    (trace_dir / "extra_info").mkdir()
    admission = require_trace_admission(trace_dir)
    payload_path = next((trace_dir / "threadblocks").rglob("*.pb"))
    original_payload = payload_path.read_bytes()

    def mutate_and_restore(**_kwargs):
        payload_path.write_bytes(
            bytes((original_payload[0] ^ 1,)) + original_payload[1:]
        )
        payload_path.write_bytes(original_payload)

    monkeypatch.setattr(
        run_suite,
        "_emit_hrd_bundle_for_run_summary",
        mutate_and_restore,
    )
    for name in (
        "_emit_divergence_proxies_for_run_summary",
        "_emit_sharing_locality_for_run_summary",
        "_emit_order_projections_for_run_summary",
        "_emit_perturbation_run_ledger_for_run_summary",
        "_emit_trace_convergence_confidence_for_run_summary",
    ):
        monkeypatch.setattr(run_suite, name, lambda **_kwargs: None)

    with pytest.raises(
        ImmutableArtifactError,
        match="protected input tree changed",
    ):
        with open_admitted_trace_sources(
            trace_dir, expected_admission=admission
        ) as trace_sources:
            run_suite._emit_admitted_trace_features(
                artifact_record={},
                artifact_dir=tmp_path,
                traces_folder=trace_dir,
                trace_admission=admission,
                slug="fixture",
                shape_point={"name": "fixture_shape"},
                binary_sha256="a" * 64,
                kernel_signature_sha256="b" * 64,
                props=SimpleNamespace(),
                trace_sources=trace_sources,
            )


def test_run_suite_feature_batch_types_inotify_queue_overflow_as_resource_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    (trace_dir / "extra_info").mkdir()
    admission = require_trace_admission(trace_dir)
    overflow_event = _inotify_queue_overflow_event()

    def arm_queue_overflow(**_kwargs) -> None:
        monkeypatch.setattr(
            immutable.os,
            "read",
            lambda _descriptor, _size: overflow_event,
        )

    monkeypatch.setattr(
        run_suite,
        "_emit_hrd_bundle_for_run_summary",
        arm_queue_overflow,
    )
    for name in (
        "_emit_divergence_proxies_for_run_summary",
        "_emit_sharing_locality_for_run_summary",
        "_emit_order_projections_for_run_summary",
        "_emit_perturbation_run_ledger_for_run_summary",
        "_emit_trace_convergence_confidence_for_run_summary",
    ):
        monkeypatch.setattr(run_suite, name, lambda **_kwargs: None)

    with pytest.raises(
        ImmutableArtifactResourceError,
        match="inotify event queue overflowed",
    ):
        with open_admitted_trace_sources(
            trace_dir, expected_admission=admission
        ) as trace_sources:
            run_suite._emit_admitted_trace_features(
                artifact_record={},
                artifact_dir=tmp_path,
                traces_folder=trace_dir,
                trace_admission=admission,
                slug="fixture",
                shape_point={"name": "fixture_shape"},
                binary_sha256="a" * 64,
                kernel_signature_sha256="b" * 64,
                props=SimpleNamespace(),
                trace_sources=trace_sources,
            )


def test_merge_refuses_active_reference_that_omits_generation_launch(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "summaries"
    admission, raw_descriptor, attempt_id = _publish_raw_trace_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace_with_launch_count(
            tmp_path / "trace", 2
        ),
        shape_point="fixture_shape",
    )
    result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )
    assert result["status"] == "ok"
    active_path = _write_active_trace_record(
        artifact_dir,
        admission=admission,
        shape_point="fixture_shape",
        paths=[result["paths"][0]],
        raw_trace_generation=raw_descriptor,
        attempt_id=attempt_id,
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    _add_minimal_manifest_authority(
        artifact_dir=artifact_dir,
        active=active,
    )
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "summary_generation_invalid" in errors[0]
    assert "reference list disagrees" in errors[0]


@pytest.mark.parametrize(
    "failure_kind",
    ("inotify_queue_overflow", "typed_admission_resource"),
)
def test_handle_trace_reports_feature_resource_failure_without_snapshot_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_kind: str,
) -> None:
    executable = tmp_path / "global_scalar_32b_copy_run"
    executable.write_bytes(b"retained executable fixture")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"retained tracer fixture")
    artifact_dir = tmp_path / "artifact"
    original_os_read = immutable.os.read

    monkeypatch.setattr(run_suite, "NVBIT_TRACER_SO", tracer)
    monkeypatch.setattr(
        run_suite,
        "_find_kernel_executable",
        lambda _slug: executable,
    )
    monkeypatch.setattr(
        run_suite,
        "_artifact_dir_for_row",
        lambda _row, _props: artifact_dir,
    )

    def fake_capture(*, traces_folder: Path, **_kwargs) -> dict[str, object]:
        _write_trace_into_existing(traces_folder)
        return {
            "baseline_proc": SimpleNamespace(
                returncode=0,
                stdout='{"wall_time_ms": 1.0}',
                stderr="",
            ),
            "baseline_elapsed_s": 0.001,
            "baseline_result": {"wall_time_ms": 1.0},
            "trace_proc": SimpleNamespace(
                returncode=0,
                stdout='{"wall_time_ms": 2.0}',
                stderr="",
            ),
            "trace_elapsed_s": 0.002,
            "binary_sha256": "a" * 64,
            "tracer_build_sha256": "b" * 64,
        }

    monkeypatch.setattr(run_suite, "_run_retained_nvbit_capture", fake_capture)

    if failure_kind == "inotify_queue_overflow":
        overflow_event = _inotify_queue_overflow_event()

        def arm_queue_overflow(**_kwargs) -> None:
            monkeypatch.setattr(
                immutable.os,
                "read",
                lambda _descriptor, _size: overflow_event,
            )

        monkeypatch.setattr(
            run_suite,
            "_emit_hrd_bundle_for_run_summary",
            arm_queue_overflow,
        )
        for name in (
            "_emit_divergence_proxies_for_run_summary",
            "_emit_sharing_locality_for_run_summary",
            "_emit_order_projections_for_run_summary",
            "_emit_perturbation_run_ledger_for_run_summary",
            "_emit_trace_convergence_confidence_for_run_summary",
        ):
            monkeypatch.setattr(run_suite, name, lambda **_kwargs: None)
    else:
        def fail_feature_generation(**_kwargs) -> None:
            raise TraceAdmissionError(
                "trace_snapshot_resource_unavailable",
                "VERIFY_RESOURCE_BOUND: inotify event queue overflowed",
            )

        monkeypatch.setattr(
            run_suite,
            "_emit_admitted_trace_features",
            fail_feature_generation,
        )
    props = run_suite.DeviceProps()
    row = {
        "slug": "global_scalar_32b_copy",
        "trace_budget_mib": 64,
        "seed_policy": {"seed": 42},
        "arch_gate": ["sm_86"],
    }

    result = run_suite.handle_trace(
        row,
        props,
        tmp_path,
        shape_point_name="trace_00",
    )
    if failure_kind == "inotify_queue_overflow":
        monkeypatch.setattr(immutable.os, "read", original_os_read)

    captured = capsys.readouterr()
    assert result == run_suite.EXIT_BUILD_FAILED
    assert (
        "trace_feature_source_resource_unavailable: "
        "trace_snapshot_resource_unavailable:"
    ) in captured.err
    assert "inotify event queue overflowed" in captured.err
    assert "trace_snapshot_changed" not in captured.err
    assert not (artifact_dir / "trace_trace_00.json").exists()


def test_handle_trace_final_gate_rejects_same_byte_unadmitted_raw_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-fixed-extra-replacement"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)
    fake_capture = run_suite._run_retained_nvbit_capture

    def capture_with_unadmitted_extra(
        *, traces_folder: Path, **kwargs
    ) -> dict[str, object]:
        result = fake_capture(traces_folder=traces_folder, **kwargs)
        (traces_folder / "capture_diagnostic.log").write_bytes(
            b"complete raw diagnostic bytes"
        )
        return result

    monkeypatch.setattr(
        run_suite, "_run_retained_nvbit_capture", capture_with_unadmitted_extra
    )
    original_budget_sidecars = run_suite._emit_nvbit_budget_sidecars

    def replace_extra_after_raw_commit(**kwargs):
        result = original_budget_sidecars(**kwargs)
        raw_root = (
            artifact_dir
            / run_suite.RAW_TRACE_GENERATION_NAMESPACE
            / "trace_00"
            / attempt_id
        )
        target = raw_root / "capture_diagnostic.log"
        payload = target.read_bytes()
        original_inode = target.stat().st_ino
        raw_root.chmod(0o755)
        try:
            target.unlink()
            target.write_bytes(payload)
            target.chmod(0o444)
        finally:
            raw_root.chmod(0o555)
        assert target.stat().st_ino != original_inode
        return result

    monkeypatch.setattr(
        run_suite,
        "_emit_nvbit_budget_sidecars",
        replace_extra_after_raw_commit,
    )
    rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    assert rc == run_suite.EXIT_BUILD_FAILED
    assert not (artifact_dir / "trace_trace_00.json").exists()
    event_path = (
        artifact_dir
        / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
        / "trace_00"
        / attempt_id
        / "preactivation_failed.json"
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["activation_state"] == "non_active"
    assert event["status"] == "preactivation_final_evidence_generation_gate_failed"
    assert event["typed_reason"] == "trace_filesystem_invalid"


def test_handle_trace_real_feature_producers_commit_inside_derived_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    real_feature_emitter = run_suite._emit_admitted_trace_features
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    props.compute_capability_major = 8
    props.compute_capability_minor = 6
    props.l2_cache_size_bytes = 6 * 1024 * 1024
    props.num_l2_slices = 6
    fixture_capture = run_suite._run_retained_nvbit_capture

    def capture_with_feature_compatible_stats(
        *, traces_folder: Path, **kwargs
    ) -> dict[str, object]:
        result = fixture_capture(traces_folder=traces_folder, **kwargs)
        (traces_folder / "stats.csv").write_text(
            "kernel id,kernel mangled name,total_insts,total_reported_insts\n"
            "kernel-0.trace,fixture_kernel,1,1\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        run_suite,
        "_run_retained_nvbit_capture",
        capture_with_feature_compatible_stats,
    )
    monkeypatch.setattr(
        run_suite,
        "_emit_admitted_trace_features",
        real_feature_emitter,
    )

    rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    assert rc == run_suite.EXIT_OK
    active = json.loads(
        (artifact_dir / "trace_trace_00.json").read_text(encoding="utf-8")
    )
    derived = active["trace_derived_generation"]
    derived_prefix = f"{derived['path']}/"
    evidence = active["trace_feature_evidence"]
    assert evidence
    assert all(item["path"].startswith(derived_prefix) for item in evidence)
    assert all((artifact_dir / item["path"]).is_file() for item in evidence)
    observed = immutable.inspect_immutable_directory_generation(
        artifact_dir / derived["path"],
        require_nonempty=False,
        max_files=derived["file_count"],
        max_directories=derived["directory_count"],
        max_total_file_bytes=derived["total_file_bytes"],
        max_total_relative_path_bytes=32 * 1024 * 1024,
    )
    assert observed.content_manifest_sha256 == derived[
        "content_manifest_sha256"
    ]


def test_handle_trace_success_activates_only_after_all_required_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    order: list[str] = []
    original_summary = run_suite._emit_nvbit_trace_summaries_for_trace_artifact
    original_manifest = run_suite._emit_manifest
    original_activation = run_suite._publish_trace_artifact_record

    monkeypatch.setattr(
        run_suite,
        "_emit_admitted_trace_features",
        lambda **_kwargs: order.append("features"),
    )

    def emit_summary(**kwargs):
        order.append("summary")
        return original_summary(**kwargs)

    def emit_manifest(*args, **kwargs):
        order.append("manifest")
        return original_manifest(*args, **kwargs)

    def activate(path: Path, record: dict[str, object]) -> None:
        order.append("activation")
        assert not path.exists()
        for descriptor_field in (
            "raw_trace_generation",
            "trace_derived_generation",
            "measurement_manifest",
            "trace_capture_attempt_event",
        ):
            descriptor = record[descriptor_field]
            assert isinstance(descriptor, dict)
            referenced = artifact_dir / str(descriptor["path"])
            assert referenced.exists()
        summaries = record["nvbit_trace_summary_paths"]
        assert isinstance(summaries, list) and summaries
        assert all((artifact_dir / path).is_file() for path in summaries)
        original_activation(path, record)

    monkeypatch.setattr(
        run_suite,
        "_emit_nvbit_trace_summaries_for_trace_artifact",
        emit_summary,
    )
    monkeypatch.setattr(run_suite, "_emit_manifest", emit_manifest)
    monkeypatch.setattr(
        run_suite, "_publish_trace_artifact_record", activate
    )

    rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    assert rc == run_suite.EXIT_OK
    assert order == ["features", "summary", "manifest", "activation"]
    active_path = artifact_dir / "trace_trace_00.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    assert active["activation_state"] == "active"
    assert active["typed_tombstone"] is False
    assert active["nvbit_trace_summary_emit_status"] == "ok"
    generation = active["nvbit_trace_summary_generation"]
    assert generation["file_count"] == 1
    assert generation["files"][0]["path"] == active[
        "nvbit_trace_summary_paths"
    ][0]
    assert (
        artifact_dir / generation["files"][0]["path"]
    ).read_bytes()
    raw = active["raw_trace_generation"]
    assert raw["attempt_id"] == active["trace_capture_attempt_id"]
    assert raw["path"] == (
        f"{run_suite.RAW_TRACE_GENERATION_NAMESPACE}/trace_00/"
        f"{active['trace_capture_attempt_id']}"
    )
    assert active["trace_pb_path"] == f"{raw['path']}/dynamic_trace.pb"
    observed_raw = immutable.inspect_immutable_directory_generation(
        artifact_dir / raw["path"],
        max_files=raw["file_count"],
        max_directories=raw["directory_count"],
        max_total_file_bytes=raw["total_file_bytes"],
        max_total_relative_path_bytes=32 * 1024 * 1024,
    )
    assert observed_raw.content_manifest_sha256 == raw[
        "content_manifest_sha256"
    ]
    assert observed_raw.filesystem_generation_sha256 == raw[
        "filesystem_generation_sha256"
    ]
    assert raw["trace_admission_sha256"] == active[
        "trace_admission_sha256"
    ]
    assert raw["trace_filesystem_generation_sha256"] == active[
        "trace_filesystem_generation_sha256"
    ]
    manifest = active["measurement_manifest"]
    manifest_path = artifact_dir / manifest["path"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest[
        "sha256"
    ]
    assert stat.S_IMODE(active_path.stat().st_mode) == 0o444


def _prepare_generation_authority_fixture(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_OK
    active_path = artifact_dir / "trace_trace_00.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    return artifact_dir, active_path, active


def test_verify_artifacts_accepts_valid_attempt_scoped_generation_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, _active = (
        _prepare_generation_authority_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    assert verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    ) == []


@pytest.mark.parametrize(
    "ancestor",
    (".hidden-stage", run_suite.RAW_TRACE_GENERATION_NAMESPACE),
    ids=("hidden", "reserved"),
)
def test_consumers_reject_scan_root_below_nonauthority_namespace(
    tmp_path: Path,
    ancestor: str,
) -> None:
    artifact_dir = tmp_path / ancestor / "artifact"
    artifact_dir.mkdir(parents=True)

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert discovered == []
    assert any("hidden or reserved evidence namespace" in error for error in merge_errors)
    assert any("inside hidden or reserved namespace" in error for error in verifier_errors)


def test_merge_accounts_decoded_active_record_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    class RejectDecodedActiveRecord:
        def reserve(
            self,
            _value: object,
            *,
            retained_input_bytes: int,
            role: str,
        ) -> None:
            assert retained_input_bytes >= 0
            if role.startswith("decoded canonical trace record "):
                raise ImmutableArtifactLimitError(
                    "synthetic decoded active-record graph limit"
                )

        def reserve_bytes(
            self,
            _size: int,
            *,
            retained_input_bytes: int,
            role: str,
        ) -> None:
            assert retained_input_bytes >= 0
            assert role

    monkeypatch.setattr(
        merge_evidence,
        "_JoinMaterializationBudget",
        RejectDecodedActiveRecord,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )

    assert discovered == []
    assert any(
        "joined_evidence_nvbit_trace_authority_limit_exceeded" in error
        and "synthetic decoded active-record graph limit" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_verifier"),
    (
        ("shape_set", None, "shape_set must equal 'trace_shapes'"),
        ("shape_set", "other_shapes", "shape_set must equal 'trace_shapes'"),
        (
            "nvbit_trace_summary_emit_status",
            None,
            "summary status must equal 'ok'",
        ),
        (
            "nvbit_trace_summary_emit_status",
            "typed_unavailable",
            "summary status must equal 'ok'",
        ),
    ),
    ids=("missing-shape-set", "wrong-shape-set", "missing-status", "wrong-status"),
)
def test_verifier_rejects_missing_or_wrong_active_discriminators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    expected_verifier: str,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    if value is None:
        active.pop(field)
    else:
        active[field] = value
    _replace_active_record_for_hostile_fixture(active_path, active)

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any(expected_verifier in error for error in errors)


@pytest.mark.parametrize(
    "authority",
    (
        "active_record",
        "raw_generation",
        "derived_generation",
        "summary_generation",
    ),
)
@pytest.mark.parametrize(
    "schema_version",
    (2, True, 1.0),
    ids=("unknown-int", "boolean", "integral-float"),
)
def test_consumers_reject_nonexact_generation_authority_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
    schema_version: object,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    if authority == "active_record":
        active["schema_version"] = schema_version
        expected_merge = "explicitly active canonical trace record"
    elif authority == "raw_generation":
        descriptor = active["raw_trace_generation"]
        assert isinstance(descriptor, dict)
        descriptor["schema_version"] = schema_version
        expected_merge = "raw_trace_generation schema_version must equal one"
    elif authority == "derived_generation":
        descriptor = active["trace_derived_generation"]
        assert isinstance(descriptor, dict)
        descriptor["schema_version"] = schema_version
        expected_merge = (
            "trace_derived_generation.path is not the exact active attempt path"
        )
    else:
        descriptor = active["nvbit_trace_summary_generation"]
        assert isinstance(descriptor, dict)
        descriptor["schema_version"] = schema_version
        expected_merge = "summary generation schema_version must equal one"
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert discovered == []
    assert any(expected_merge in error for error in merge_errors)
    assert any("schema_version must equal one" in error for error in verifier_errors)


def test_consumers_reject_writable_active_trace_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    active_path.chmod(0o644)

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert discovered == []
    assert any("single-link 0444 regular file" in error for error in merge_errors)
    assert any(
        "single-link 0444 regular file" in error for error in verifier_errors
    )


@pytest.mark.parametrize("consumer", ("merge", "verifier"))
@pytest.mark.parametrize("error_number", (errno.EMFILE, errno.ENFILE))
def test_consumers_type_tree_snapshot_exhaustion_as_resource_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    error_number: int,
) -> None:
    artifact_dir, _active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    original_snapshot_tree = immutable.PinnedDirectoryReader.snapshot_tree

    def exhaust_during_snapshot(reader, *args, **kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(
                immutable.os,
                "dup",
                lambda _descriptor: (_ for _ in ()).throw(
                    OSError(error_number, "synthetic tree snapshot exhaustion")
                ),
            )
            return original_snapshot_tree(reader, *args, **kwargs)

    monkeypatch.setattr(
        immutable.PinnedDirectoryReader,
        "snapshot_tree",
        exhaust_during_snapshot,
    )

    if consumer == "merge":
        discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
            artifact_dir,
            expected_platform="rtx_3090_ti",
        )
        assert discovered == []
    else:
        errors = verify_artifacts._verify_canonical_trace_pb_admission(
            artifact_dir
        )

    assert any(
        "ResourceUnavailable" in error or "resource_unavailable" in error
        for error in errors
    ), errors
    assert any("synthetic tree snapshot exhaustion" in error for error in errors)


def test_verifier_loads_summary_schema_through_bounded_retained_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    original = immutable.PinnedDirectoryReader.read_snapshot
    observed_limits: list[int | None] = []
    summary_limits: list[int | None] = []

    def track_schema_read(reader, relative_path: str, **kwargs):
        name = PurePosixPath(relative_path).name
        if name == "nvbit_trace_summary.schema.json":
            observed_limits.append(kwargs.get("max_bytes"))
        elif name.startswith("nvbit_trace_summary_"):
            summary_limits.append(kwargs.get("max_bytes"))
        return original(reader, relative_path, **kwargs)

    monkeypatch.setattr(
        immutable.PinnedDirectoryReader,
        "read_snapshot",
        track_schema_read,
    )

    assert verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    ) == []
    assert observed_limits == [
        verify_artifacts._CANONICAL_AUTHORITY_JSON_MAX_BYTES
    ]
    assert summary_limits
    assert set(summary_limits) == {
        verify_artifacts._CANONICAL_AUTHORITY_JSON_MAX_BYTES
    }


def test_verifier_types_summary_schema_resource_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    original = immutable.PinnedDirectoryReader.read_snapshot

    def fail_schema_read(reader, relative_path: str, **kwargs):
        if PurePosixPath(relative_path).name == "nvbit_trace_summary.schema.json":
            raise ImmutableArtifactResourceError(
                "synthetic summary schema exhaustion"
            )
        return original(reader, relative_path, **kwargs)

    monkeypatch.setattr(
        immutable.PinnedDirectoryReader,
        "read_snapshot",
        fail_schema_read,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any("ResourceUnavailable" in error for error in errors), errors
    assert any("synthetic summary schema exhaustion" in error for error in errors)


def test_verifier_rejects_same_byte_schema_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_path = tmp_path / "schema.json"
    payload = b'{"type":"object"}\n'
    schema_path.write_bytes(payload)
    original = immutable.PinnedDirectoryReader.read_snapshot
    replaced = False

    def replace_after_schema_read(reader, relative_path: str, **kwargs):
        nonlocal replaced
        result = original(reader, relative_path, **kwargs)
        if relative_path == schema_path.name and not replaced:
            replacement = tmp_path / ".schema-replacement.json"
            replacement.write_bytes(payload)
            os.replace(replacement, schema_path)
            replaced = True
        return result

    monkeypatch.setattr(
        immutable.PinnedDirectoryReader,
        "read_snapshot",
        replace_after_schema_read,
    )

    with pytest.raises(ImmutableArtifactError):
        with verify_artifacts._load_canonical_schema_validator(
            schema_path,
            label="fixture schema",
        ):
            pass
    assert replaced is True


def test_verifier_retains_schema_monitor_through_validation(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.json"
    payload = b'{"type":"object"}\n'
    schema_path.write_bytes(payload)

    with pytest.raises(ImmutableArtifactError):
        with verify_artifacts._load_canonical_schema_validator(
            schema_path,
            label="fixture schema",
        ) as validator:
            replacement = tmp_path / ".schema-replacement.json"
            replacement.write_bytes(payload)
            os.replace(replacement, schema_path)
            assert list(validator.iter_errors({})) == []


def test_verifier_preflights_aggregate_json_before_summary_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        verify_artifacts,
        "_CANONICAL_AUTHORITY_JSON_MAX_TOTAL_BYTES",
        active_path.stat().st_size,
    )
    original_inspect = verify_artifacts.inspect_immutable_directory_generation

    def reject_summary_hash(path: Path, *args, **kwargs):
        if verify_artifacts._SUMMARY_GENERATION_NAMESPACE in Path(path).parts:
            raise AssertionError("summary hashing preceded aggregate preflight")
        return original_inspect(path, *args, **kwargs)

    monkeypatch.setattr(
        verify_artifacts,
        "inspect_immutable_directory_generation",
        reject_summary_hash,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any("aggregate retained JSON authority" in error for error in errors)


def test_verifier_preserves_raw_admission_resource_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    def fail_raw_admission(_path: Path):
        raise TraceAdmissionError(
            "trace_snapshot_resource_unavailable",
            "synthetic raw admission exhaustion",
        )

    monkeypatch.setattr(
        verify_artifacts,
        "require_trace_admission",
        fail_raw_admission,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any("ResourceUnavailable" in error for error in errors), errors
    assert any("synthetic raw admission exhaustion" in error for error in errors)


def test_merge_accepts_valid_attempt_scoped_generation_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, active = (
        _prepare_generation_authority_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )

    assert errors == []
    assert discovered == [
        artifact_dir / path for path in active["nvbit_trace_summary_paths"]
    ]


def test_merge_relative_root_keeps_artifact_local_namespace_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    active.pop("trace_derived_generation")
    active.pop("trace_capture_attempt_event")
    _replace_active_record_for_hostile_fixture(active_path, active)
    monkeypatch.chdir(tmp_path)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        Path(artifact_dir.name),
        expected_platform="rtx_3090_ti",
    )

    assert discovered == []
    assert any(
        "derived generation authority exists" in error
        or "trace-attempt authority exists" in error
        for error in errors
    )


def test_merge_relative_root_checks_attempt_namespace_after_valid_derived_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    active.pop("trace_capture_attempt_event")
    _replace_active_record_for_hostile_fixture(active_path, active)
    monkeypatch.chdir(tmp_path)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        Path(artifact_dir.name),
        expected_platform="rtx_3090_ti",
    )

    assert discovered == []
    assert any("trace-attempt authority exists" in error for error in errors)
    assert not any(
        "derived generation authority exists" in error for error in errors
    )


def test_merge_scopes_optional_namespaces_to_each_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modern_tmp = tmp_path / "modern"
    modern_tmp.mkdir()
    modern_dir, _active_path, modern = _prepare_generation_authority_fixture(
        tmp_path=modern_tmp,
        monkeypatch=monkeypatch,
    )
    legacy_dir = tmp_path / "legacy" / "artifact"
    _admission, _result, legacy_path = _publish_active_summary_fixture(
        artifact_dir=legacy_dir,
        source_root=_write_trace(tmp_path / "legacy-trace"),
        shape_point="fixture_shape",
    )
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(tmp_path)

    assert errors == []
    assert set(discovered) == {
        modern_dir / path for path in modern["nvbit_trace_summary_paths"]
    } | {
        legacy_dir / path for path in legacy["nvbit_trace_summary_paths"]
    }


@pytest.mark.parametrize("consumer", ("merge", "verifier"))
@pytest.mark.parametrize(
    ("mutation", "expected_terms"),
    (
        ("stale_derived_seal", ("derived", "stale")),
        ("missing_derived_generation", ("derived", "absent")),
        ("stale_attempt_event", ("attempt", "stale")),
        ("mismatched_attempt_event", ("body", "disagrees")),
    ),
)
def test_consumers_reject_stale_derived_and_attempt_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    mutation: str,
    expected_terms: tuple[str, ...],
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    if mutation == "stale_derived_seal":
        derived = active["trace_derived_generation"]
        assert isinstance(derived, dict)
        derived["content_manifest_sha256"] = "0" * 64
    elif mutation == "missing_derived_generation":
        derived = active["trace_derived_generation"]
        assert isinstance(derived, dict)
        generation_path = artifact_dir / str(derived["path"])
        generation_path.rename(
            generation_path.with_name(f".{generation_path.name}.missing")
        )
    elif mutation == "stale_attempt_event":
        event_descriptor = active["trace_capture_attempt_event"]
        assert isinstance(event_descriptor, dict)
        event_descriptor["sha256"] = "0" * 64
    else:
        event_descriptor = active["trace_capture_attempt_event"]
        assert isinstance(event_descriptor, dict)
        event_path = artifact_dir / str(event_descriptor["path"])
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["slug"] = "mismatched-fixture"
        payload = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        immutable.retire_regular_file(event_path)
        immutable.publish_regular_file(
            event_path.parent,
            event_path.name,
            payload,
        )
        event_descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
        event_descriptor["size_bytes"] = len(payload)
    _replace_active_record_for_hostile_fixture(active_path, active)

    if consumer == "merge":
        discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
            artifact_dir,
            expected_platform="rtx_3090_ti",
        )
        assert discovered == []
    else:
        errors = verify_artifacts._verify_canonical_trace_pb_admission(
            artifact_dir
        )

    assert any(
        all(term in error.lower() for term in expected_terms)
        for error in errors
    )


def test_verify_artifacts_discovers_valid_nested_generation_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert verify_artifacts._verify_canonical_trace_pb_admission(tmp_path) == []


@pytest.mark.parametrize(
    "mutation",
    ("extra_descriptor_key", "writable_manifest"),
)
def test_manifest_descriptor_and_file_integrity_match_across_consumers(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest_path = artifact_dir / str(descriptor["path"])
    if mutation == "extra_descriptor_key":
        descriptor["extra"] = "forbidden"
        _replace_active_record_for_hostile_fixture(active_path, active)
    else:
        manifest_path.chmod(0o644)

    discovered, merge_errors = (
        merge_evidence._discover_nvbit_summary_inputs(artifact_dir)
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert discovered == []
    assert any("measurement_manifest_invalid" in error for error in merge_errors)
    assert any("measurement manifest" in error for error in verifier_errors)



def test_verify_artifacts_bounds_active_measurement_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest_path = artifact_dir / str(descriptor["path"])
    immutable.retire_regular_file(manifest_path)
    payload = (
        json.dumps(
            {
                "padding": "x"
                * verify_artifacts._CANONICAL_AUTHORITY_JSON_MAX_BYTES,
                "shape_point": active["shape_point"],
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert len(payload) > verify_artifacts._CANONICAL_AUTHORITY_JSON_MAX_BYTES
    immutable.publish_regular_file(
        artifact_dir,
        manifest_path.name,
        payload,
    )
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["size_bytes"] = len(payload)
    _replace_active_record_for_hostile_fixture(active_path, active)

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any(
        "measurement manifest exceeds the bounded authority JSON limit" in error
        for error in errors
    )


def test_verify_artifacts_rejects_measurement_manifest_descriptor_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    payload = (
        json.dumps(
            {
                "shape_point": active["shape_point"],
                "substitute": True,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    substitute_name = "measurement_manifest_substitute.json"
    immutable.publish_regular_file(artifact_dir, substitute_name, payload)
    active["measurement_manifest"] = {
        "path": substitute_name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    _replace_active_record_for_hostile_fixture(active_path, active)

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any(
        "measurement manifest path does not name the exact active shape authority"
        in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("decode_error", "expected_prefix", "expected_detail"),
    (
        (
            ValueError("synthetic large-number decode refusal"),
            "CanonicalGenerationAuthorityInvalid:",
            "measurement manifest is absent or invalid: ValueError",
        ),
        (
            MemoryError("synthetic manifest decode exhaustion"),
            "CanonicalGenerationAuthorityResourceUnavailable:",
            "measurement manifest resource unavailable: MemoryError",
        ),
    ),
    ids=("value-error", "memory-error"),
)
def test_verify_artifacts_classifies_measurement_manifest_decode_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decode_error: BaseException,
    expected_prefix: str,
    expected_detail: str,
) -> None:
    artifact_dir, _active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    descriptor = active["measurement_manifest"]
    assert isinstance(descriptor, dict)
    manifest_payload = (artifact_dir / str(descriptor["path"])).read_bytes()
    real_strict_json_loads = verify_artifacts.strict_json_loads

    def decode_with_manifest_failure(
        payload: bytes,
        *,
        label: str = "JSON",
    ) -> object:
        if payload == manifest_payload and label == "measurement manifest":
            raise decode_error
        return real_strict_json_loads(payload, label=label)

    monkeypatch.setattr(
        verify_artifacts,
        "strict_json_loads",
        decode_with_manifest_failure,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert any(
        error.startswith(expected_prefix) and expected_detail in error
        for error in errors
    )


def test_verify_artifacts_classifies_large_integer_json_as_invalid(
    tmp_path: Path,
) -> None:
    digit_limit = sys.get_int_max_str_digits()
    if digit_limit == 0:
        pytest.skip("Python integer string conversion limit is disabled")
    candidate = immutable.publish_regular_file(
        tmp_path,
        "trace_fixture.json",
        (
            '{"activation_state":"active","shape_point":"fixture","value":'
            + "9" * (digit_limit + 1)
            + "}\n"
        ).encode("utf-8"),
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert any(
        error.startswith("CanonicalGenerationAuthorityInvalid:")
        and str(candidate) in error
        and "ValueError" in error
        for error in errors
    )


def test_verify_artifacts_classifies_deeply_nested_json_as_invalid(
    tmp_path: Path,
) -> None:
    candidate = immutable.publish_regular_file(
        tmp_path,
        "trace_fixture.json",
        ("[" * 10_000 + "0" + "]" * 10_000).encode("utf-8"),
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert any(
        error.startswith("CanonicalGenerationAuthorityInvalid:")
        and str(candidate) in error
        and "maximum recursion depth" in error
        for error in errors
    )


def test_verify_artifacts_classifies_json_decode_memory_error_as_resource_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = immutable.publish_regular_file(
        tmp_path,
        "trace_fixture.json",
        b"{}\n",
    )

    def fail_decode(*_args: object, **_kwargs: object) -> object:
        raise MemoryError("synthetic decode exhaustion")

    monkeypatch.setattr(verify_artifacts, "_read_snapshot_json", fail_decode)

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert any(
        error.startswith("CanonicalGenerationAuthorityResourceUnavailable:")
        and str(candidate) in error
        and "MemoryError" in error
        for error in errors
    )


def test_verify_artifacts_preserves_per_artifact_legacy_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    legacy_shape = tmp_path / "legacy-sibling" / "trace_legacy"
    legacy_shape.mkdir(parents=True)
    trace_pb = legacy_shape / "dynamic_trace.pb"
    trace_pb.write_bytes(b"legacy raw bytes")

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert any(
        error.startswith("RawOnlyTraceTree:") and str(trace_pb) in error
        for error in errors
    )
    assert not any(
        error.startswith("LegacyTraceWithGenerationAuthority:")
        and str(trace_pb) in error
        for error in errors
    )


def test_verify_artifacts_nested_tombstone_blocks_local_legacy_fallback(
    tmp_path: Path,
) -> None:
    cell = tmp_path / "nested" / "cell"
    shape_dir = cell / "trace_fixture"
    shape_dir.mkdir(parents=True)
    trace_pb = shape_dir / "dynamic_trace.pb"
    trace_pb.write_bytes(b"legacy raw bytes")
    (cell / "trace_fixture.json").write_text(
        json.dumps(
            {
                "activation_state": "non_active",
                "shape_point": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert any(
        error.startswith("CanonicalGenerationAuthorityMissing:")
        and str(cell) in error
        for error in errors
    )
    assert any(
        error.startswith("LegacyTraceWithGenerationAuthority:")
        and str(trace_pb) in error
        for error in errors
    )


def test_verify_artifacts_discovers_nested_active_leaf_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, active = (
        _prepare_generation_authority_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )
    nested = artifact_dir / "nested"
    nested.mkdir()
    nested_active = json.loads(json.dumps(active))
    nested_active["shape_point"] = "spoof"
    nested_active["trace_capture_attempt_id"] = "attempt-spoof"
    nested_active["raw_trace_generation"]["path"] = "outside-authority"
    nested_record = immutable.publish_regular_file(
        nested,
        "trace_spoof.json",
        (json.dumps(nested_active) + "\n").encode("utf-8"),
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert any(str(nested_record) in error for error in errors)
    assert any(
        "raw trace generation path is not the exact active attempt path" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("stale_raw_seal", "raw trace generation content"),
        ("missing_raw_generation", "raw trace generation is absent"),
        ("tampered_raw_reference", "exact active attempt path"),
        ("stale_summary_seal", "summary generation content"),
        ("missing_summary_generation", "summary generation is absent"),
    ),
)
def test_verify_artifacts_rejects_stale_or_tampered_generation_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    if mutation == "stale_raw_seal":
        descriptor = active["raw_trace_generation"]
        descriptor["content_manifest_sha256"] = "0" * 64
        _replace_active_record_for_hostile_fixture(active_path, active)
    elif mutation == "missing_raw_generation":
        raw_path = artifact_dir / active["raw_trace_generation"]["path"]
        raw_path.rename(raw_path.with_name(f".{raw_path.name}.missing"))
    elif mutation == "tampered_raw_reference":
        descriptor = active["raw_trace_generation"]
        descriptor["path"] = (
            f"{run_suite.RAW_TRACE_GENERATION_NAMESPACE}/trace_00/"
            "attempt-tampered"
        )
        _replace_active_record_for_hostile_fixture(active_path, active)
    elif mutation == "stale_summary_seal":
        descriptor = active["nvbit_trace_summary_generation"]
        descriptor["content_manifest_sha256"] = "0" * 64
        _replace_active_record_for_hostile_fixture(active_path, active)
    else:
        summary_path = (
            artifact_dir / active["nvbit_trace_summary_generation"]["path"]
        )
        summary_path.rename(
            summary_path.with_name(f".{summary_path.name}.missing")
        )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert errors
    assert any("CanonicalGenerationAuthorityInvalid" in error for error in errors)
    assert any(expected in error for error in errors)


def test_verify_artifacts_disables_legacy_fallback_beside_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    legacy = artifact_dir / "trace_trace_00"
    legacy.mkdir()
    raw_path = artifact_dir / active["raw_trace_generation"]["path"]
    (legacy / "dynamic_trace.pb").write_bytes(
        (raw_path / "dynamic_trace.pb").read_bytes()
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert any("LegacyTraceWithGenerationAuthority" in error for error in errors)


def test_verify_artifacts_rejects_generation_namespace_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, _active = (
        _prepare_generation_authority_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )
    external = tmp_path / "external-phase-namespace"
    external.mkdir()
    namespace = artifact_dir / run_suite.PHASE_LEDGER_GENERATION_NAMESPACE
    namespace.symlink_to(external, target_is_directory=True)

    errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert any("without symlink indirection" in error for error in errors)


@pytest.mark.parametrize(
    "namespace_name",
    (
        run_suite.RAW_TRACE_GENERATION_NAMESPACE,
        run_suite.TRACE_DERIVED_GENERATION_NAMESPACE,
        run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE,
    ),
)
def test_verify_artifacts_rejects_regular_file_generation_namespace(
    tmp_path: Path,
    namespace_name: str,
) -> None:
    cell = tmp_path / "nested" / "cell"
    cell.mkdir(parents=True)
    namespace = cell / namespace_name
    namespace.write_bytes(b"not a directory")
    shape_dir = cell / "trace_fixture"
    shape_dir.mkdir()
    trace_pb = shape_dir / "dynamic_trace.pb"
    trace_pb.write_bytes(b"legacy raw bytes")

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert any(
        error.startswith("CanonicalGenerationAuthorityInvalid:")
        and str(namespace) in error
        and "not a no-follow directory" in error
        for error in errors
    )
    assert any(
        error.startswith("LegacyTraceWithGenerationAuthority:")
        and str(trace_pb) in error
        for error in errors
    )


@pytest.mark.parametrize(
    "namespace_name",
    (
        run_suite.TRACE_DERIVED_GENERATION_NAMESPACE,
        run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE,
    ),
)
def test_verify_artifacts_reserved_directory_namespace_blocks_legacy_fallback(
    tmp_path: Path,
    namespace_name: str,
) -> None:
    cell = tmp_path / "nested" / "cell"
    namespace = cell / namespace_name
    namespace.mkdir(parents=True)
    shape_dir = cell / "trace_fixture"
    shape_dir.mkdir()
    trace_pb = shape_dir / "dynamic_trace.pb"
    trace_pb.write_bytes(b"legacy raw bytes")

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert any(
        error.startswith("CanonicalGenerationAuthorityMissing:")
        and str(cell) in error
        for error in errors
    )
    assert any(
        error.startswith("LegacyTraceWithGenerationAuthority:")
        and str(trace_pb) in error
        for error in errors
    )
    assert not any(
        error.startswith("RawOnlyTraceTree:") and str(trace_pb) in error
        for error in errors
    )


def test_verify_artifacts_directory_trace_candidate_blocks_legacy_fallback(
    tmp_path: Path,
) -> None:
    cell = tmp_path / "nested" / "cell"
    cell.mkdir(parents=True)
    candidate = cell / "trace_fixture.json"
    candidate.mkdir()
    (candidate / "marker").write_bytes(b"directory-shaped candidate")
    shape_dir = cell / "trace_fixture"
    shape_dir.mkdir()
    trace_pb = shape_dir / "dynamic_trace.pb"
    trace_pb.write_bytes(b"legacy raw bytes")

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert any(
        error.startswith("CanonicalGenerationAuthorityInvalid:")
        and str(candidate) in error
        and "canonical trace-record candidate" in error
        for error in errors
    )
    assert any(
        error.startswith("LegacyTraceWithGenerationAuthority:")
        and str(trace_pb) in error
        for error in errors
    )
    assert not any(
        error.startswith("RawOnlyTraceTree:") and str(trace_pb) in error
        for error in errors
    )


@pytest.mark.parametrize("node_kind", ("symlink", "fifo"))
def test_verify_artifacts_special_trace_candidate_fails_before_fallback(
    tmp_path: Path,
    node_kind: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    candidate = root / "trace_fixture.json"
    if node_kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text("{}\n", encoding="utf-8")
        candidate.symlink_to(target)
    else:
        os.mkfifo(candidate)
    shape_dir = root / "trace_fixture"
    shape_dir.mkdir()
    trace_pb = shape_dir / "dynamic_trace.pb"
    trace_pb.write_bytes(b"legacy raw bytes")

    errors = verify_artifacts._verify_canonical_trace_pb_admission(root)

    assert any(
        error.startswith("CanonicalGenerationAuthorityInvalid:")
        for error in errors
    )
    assert not any(
        error.startswith("RawOnlyTraceTree:")
        or error.startswith("LegacyTraceWithGenerationAuthority:")
        for error in errors
    )


def test_verify_artifacts_rejects_symlinked_intermediate_discovery_node(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "trace_spoof.json").write_text("{}\n", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").symlink_to(external, target_is_directory=True)

    errors = verify_artifacts._verify_canonical_trace_pb_admission(root)

    assert any("without symlink indirection" in error for error in errors)
    assert not any(
        error.startswith("CanonicalGenerationAuthorityMissing:")
        for error in errors
    )


@pytest.mark.parametrize(
    "bound",
    ("files", "directories", "total_bytes", "path_bytes"),
)
def test_verify_artifacts_discovery_limits_fail_before_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
) -> None:
    if bound == "files":
        (tmp_path / "one").write_bytes(b"")
        (tmp_path / "two").write_bytes(b"")
        monkeypatch.setattr(
            verify_artifacts,
            "_CANONICAL_DISCOVERY_MAX_FILES",
            1,
        )
    elif bound == "directories":
        (tmp_path / "nested").mkdir()
        monkeypatch.setattr(
            verify_artifacts,
            "_CANONICAL_DISCOVERY_MAX_DIRECTORIES",
            1,
        )
    elif bound == "total_bytes":
        (tmp_path / "payload").write_bytes(b"xx")
        monkeypatch.setattr(
            verify_artifacts,
            "_CANONICAL_DISCOVERY_MAX_TOTAL_FILE_BYTES",
            1,
        )
    else:
        (tmp_path / "long-relative-name").write_bytes(b"")
        monkeypatch.setattr(
            verify_artifacts,
            "_CANONICAL_DISCOVERY_MAX_RELATIVE_PATH_BYTES",
            1,
        )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert len(errors) == 1
    assert "bounded authority discovery limit exceeded" in errors[0]
    assert "RawOnlyTraceTree" not in errors[0]


def test_verify_artifacts_rejects_candidate_inserted_after_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = verify_artifacts._verify_legacy_canonical_trace_pb_admission
    inserted = False

    def insert_candidate_then_verify(*args, **kwargs) -> list[str]:
        nonlocal inserted
        inserted = True
        (tmp_path / "trace_race.json").write_text("{}\n", encoding="utf-8")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        verify_artifacts,
        "_verify_legacy_canonical_trace_pb_admission",
        insert_candidate_then_verify,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)

    assert inserted is True
    assert any("trace authority changed" in error for error in errors)


def _prepare_phase_generation_authority_fixture(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )

    def publish_phase_generation(**kwargs) -> dict[str, object]:
        generation_root = (
            artifact_dir
            / run_suite.PHASE_LEDGER_GENERATION_NAMESPACE
            / kwargs["shape_point_name"]
        )
        payload = _valid_phase_ledger_payload(
            slug=kwargs["slug"],
            shape_point=kwargs["shape_point_name"],
            repeat_index=0,
            launch_index=0,
            binary_sha256=kwargs["binary_sha256"],
            kernel_signature_sha256=kwargs["kernel_signature_sha256"],
            tracer_build_sha256=kwargs["tracer_build_sha256"],
        )
        final = immutable.publish_directory(
            generation_root,
            kwargs["attempt_id"],
            [
                immutable.ImmutableFile.from_bytes(
                    "phase_cycle_ledger_0_0.json",
                    payload,
                )
            ],
            accept_existing=False,
        )
        generation = immutable.inspect_immutable_directory_generation(
            final,
            require_nonempty=True,
            max_files=1,
            max_directories=1,
            max_total_file_bytes=len(payload),
            max_total_relative_path_bytes=4096,
        )
        return {
            "return_code": run_suite.EXIT_OK,
            "typed_reason": None,
            "generation": generation,
            "staging_path": None,
            "summary": {"n_admitted": 1, "n_rejected": 0},
        }

    monkeypatch.setattr(
        run_suite,
        "_run_phase_inject_ab_for_handle_trace",
        publish_phase_generation,
    )
    assert run_suite.handle_trace(
        row,
        props,
        tmp_path,
        shape_point_name="trace_00",
        phase_inject_ab=True,
        ab_repeat_count=1,
    ) == run_suite.EXIT_OK
    active_path = artifact_dir / "trace_trace_00.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    return artifact_dir, active_path, active


def _publish_phase_identity_generation(
    tmp_path: Path,
    records: dict[str, dict[str, object]],
):
    material = [
        immutable.ImmutableFile.from_bytes(
            filename,
            (json.dumps(record, sort_keys=True) + "\n").encode(),
        )
        for filename, record in sorted(records.items())
    ]
    final = immutable.publish_directory(
        tmp_path / "phase_identity_generations",
        "attempt-phase-identity",
        material,
        accept_existing=False,
    )
    return immutable.inspect_immutable_directory_generation(
        final,
        require_nonempty=True,
        **run_suite._trace_generation_commit_limits(),
    )


def test_phase_identity_rejects_noncanonical_normalized_key(
    tmp_path: Path,
) -> None:
    generation = _publish_phase_identity_generation(
        tmp_path,
        {
            "phase_cycle_ledger_00_0.json": {
                "shape_point": "trace_00",
                "repeat_index": 0,
                "launch_index": 0,
            },
        },
    )

    with pytest.raises(
        ImmutableArtifactError,
        match="filename indices are not canonical",
    ):
        run_suite._phase_evidence_identities(
            artifact_dir=tmp_path,
            shape_point_name="trace_00",
            expected_repeat_count=1,
            expected_launch_indices=(0,),
            phase_requested=True,
            generation=generation,
        )


def test_phase_identity_rejects_consistent_partial_launch_vectors(
    tmp_path: Path,
) -> None:
    generation = _publish_phase_identity_generation(
        tmp_path,
        {
            f"phase_cycle_ledger_{repeat_index}_0.json": {
                "shape_point": "trace_00",
                "repeat_index": repeat_index,
                "launch_index": 0,
            }
            for repeat_index in range(2)
        },
    )

    with pytest.raises(
        ImmutableArtifactError,
        match="does not equal the admitted repeat-by-launch authority",
    ):
        run_suite._phase_evidence_identities(
            artifact_dir=tmp_path,
            shape_point_name="trace_00",
            expected_repeat_count=2,
            expected_launch_indices=(0, 1),
            phase_requested=True,
            generation=generation,
        )


def test_handle_trace_binds_exact_phase_generation_before_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, active = (
        _prepare_phase_generation_authority_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )
    attempt_id = active["trace_capture_attempt_id"]
    phase = active["phase_inject_ab"]
    assert phase["requested"] is True
    assert phase["repeat_count"] == 1
    assert phase["generation"]["path"] == (
        f"{run_suite.PHASE_LEDGER_GENERATION_NAMESPACE}/trace_00/{attempt_id}"
    )
    assert len(phase["evidence"]) == 1
    assert verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    ) == []

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert errors == []
    assert [path.relative_to(artifact_dir).as_posix() for path in discovered] == (
        active["nvbit_trace_summary_paths"]
    )


@pytest.mark.parametrize(
    "failure_mode",
    ("source_mutation", "source_reader_cleanup"),
)
def test_phase_postcommit_failure_event_binds_preserved_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-phase-postcommit-failure"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)
    monkeypatch.setattr(
        run_suite,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        phase_inject._consumer,
        "load_admission_registry",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        phase_inject._consumer,
        "admission_threshold_for_slug",
        lambda _admission, _slug: 5.0,
    )
    monkeypatch.setattr(
        phase_inject._consumer,
        "admit_phase_cycle_ledger",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        phase_inject._agg,
        "decode_phase_clock_records",
        lambda _payload: [SimpleNamespace(launch_index=0)],
    )
    monkeypatch.setattr(
        phase_inject._agg,
        "build_phase_cycle_ledger",
        lambda _records, **kwargs: {
            "shape_point": kwargs["shape_point"],
            "repeat_index": kwargs["repeat_index"],
            "launch_index": kwargs["launch_index"],
        },
    )
    monkeypatch.setattr(
        phase_inject._agg,
        "serialize_phase_cycle_ledger",
        lambda ledger: (json.dumps(ledger, sort_keys=True) + "\n").encode(),
    )
    phase_source: Path | None = None
    phase_source_reader: immutable.PinnedDirectoryReader | None = None

    def emit_phase_record(inputs, **_kwargs) -> dict[str, object]:
        nonlocal phase_source, phase_source_reader
        phase_source = inputs.traces_folder / "phase_records_0_0.bin"
        phase_source.write_bytes(b"phase record before commit")
        assert inputs.traces_reader is not None
        phase_source_reader = inputs.traces_reader
        assert inputs.expected_launch_indices == (0,)
        inputs.executable_authority.verify_unchanged()
        source_snapshot = inputs.traces_reader.snapshot_tree(
            max_files=phase_inject._PHASE_SOURCE_MAX_FILES,
            max_directories=phase_inject._PHASE_SOURCE_MAX_DIRECTORIES,
            max_total_file_bytes=phase_inject._PHASE_SOURCE_MAX_BYTES,
            max_total_relative_path_bytes=(
                phase_inject._PHASE_SOURCE_MAX_RELATIVE_PATH_BYTES
            ),
        )
        return {
            "t_wall_no_inject_ns": 100,
            "t_wall_with_inject_ns": 101,
            "overhead_pct": 1.0,
            "admission_threshold_pct": 5.0,
            "admitted": True,
            "_phase_source_snapshot": source_snapshot,
        }

    monkeypatch.setattr(
        phase_inject,
        "run_ab_overhead_pair",
        emit_phase_record,
    )
    original_commit = phase_inject.ExclusiveDirectoryStage.commit

    def commit_then_mutate_phase_source(stage, **commit_kwargs):
        generation = original_commit(stage, **commit_kwargs)
        if (
            failure_mode == "source_mutation"
            and phase_inject.PHASE_LEDGER_GENERATION_NAMESPACE
            in stage.output_root.parts
        ):
            assert phase_source is not None
            phase_source.write_bytes(b"phase source changed after commit")
        return generation

    monkeypatch.setattr(
        phase_inject.ExclusiveDirectoryStage,
        "commit",
        commit_then_mutate_phase_source,
    )
    original_reader_close = immutable.PinnedDirectoryReader.close

    def close_phase_source_reader_then_fail(
        reader: immutable.PinnedDirectoryReader,
    ) -> None:
        original_reader_close(reader)
        if (
            failure_mode == "source_reader_cleanup"
            and reader is phase_source_reader
        ):
            raise OSError("planted phase source reader cleanup failure")

    monkeypatch.setattr(
        immutable.PinnedDirectoryReader,
        "close",
        close_phase_source_reader_then_fail,
    )
    assert run_suite.handle_trace(
        row,
        props,
        tmp_path,
        shape_point_name="trace_00",
        phase_inject_ab=True,
        ab_repeat_count=1,
    ) == run_suite.EXIT_BUILD_FAILED
    assert not (artifact_dir / "trace_trace_00.json").exists()

    event_path = (
        artifact_dir
        / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
        / "trace_00"
        / attempt_id
        / "preactivation_failed.json"
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))
    descriptor = event["phase_cycle_ledger_generation"]
    assert event["activation_state"] == "non_active"
    assert event["status"] == "optional_phase_failed"
    assert event["typed_reason"] == "trace_filesystem_invalid"
    assert descriptor["path"] == (
        f"{run_suite.PHASE_LEDGER_GENERATION_NAMESPACE}/trace_00/{attempt_id}"
    )
    assert event["phase_ledger_staging_path"] is None
    assert (artifact_dir / descriptor["path"]).is_dir()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("stale_descriptor", "descriptor is stale"),
        ("missing_generation", "absent or invalid"),
    ),
)
def test_merge_rejects_stale_or_missing_active_phase_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    artifact_dir, active_path, active = (
        _prepare_phase_generation_authority_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )
    descriptor = active["phase_inject_ab"]["generation"]
    if mutation == "stale_descriptor":
        descriptor["content_manifest_sha256"] = "0" * 64
        _replace_active_record_for_hostile_fixture(active_path, active)
    else:
        generation_path = artifact_dir / descriptor["path"]
        generation_path.rename(
            generation_path.with_name(f".{generation_path.name}.missing")
        )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "phase_generation_invalid" in errors[0]
    assert expected in errors[0]

    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any("phase ledger generation" in error for error in verifier_errors)


def test_unrequested_phase_record_cannot_authorize_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = (
        _prepare_phase_generation_authority_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )
    active["phase_inject_ab"]["requested"] = False
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "unrequested phase evidence" in errors[0]
    assert any(
        "unrequested phase evidence carries active authority" in error
        for error in verify_artifacts._verify_canonical_trace_pb_admission(
            artifact_dir
        )
    )


def test_merge_ignores_unreferenced_prior_phase_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    stale_root = (
        artifact_dir
        / run_suite.PHASE_LEDGER_GENERATION_NAMESPACE
        / "trace_00"
    )
    immutable.publish_directory(
        stale_root,
        "attempt-prior",
        [
            immutable.ImmutableFile.from_bytes(
                "phase_cycle_ledger_9_9.json",
                b'{"stale":true}\n',
            )
        ],
        accept_existing=False,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert errors == []
    assert [path.relative_to(artifact_dir).as_posix() for path in discovered] == (
        active["nvbit_trace_summary_paths"]
    )

    active.pop("phase_inject_ab")
    _replace_active_record_for_hostile_fixture(active_path, active)
    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "has no phase_inject_ab reference" in errors[0]
    assert any(
        "phase generation authority exists" in error
        for error in verify_artifacts._verify_canonical_trace_pb_admission(
            artifact_dir
        )
    )


def test_merge_allows_unrelated_sibling_events_during_terminal_monitor_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cupti_dir, _nvbit_dir, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    original_retain = merge_evidence._retain_join_input_tree
    sibling = cupti_dir.with_name("unrelated-terminal-sibling")
    mutation_armed = False
    mutated = False

    def retain_with_terminal_sibling_event(
        root: Path,
        *,
        source_stack: ExitStack,
        retained_authorities: list[object],
    ):
        nonlocal mutation_armed
        result = original_retain(
            root,
            source_stack=source_stack,
            retained_authorities=retained_authorities,
        )
        if Path(root).absolute() != cupti_dir.absolute():
            return result
        monitor = retained_authorities[-1]
        assert isinstance(monitor, immutable.PinnedPathSetMonitor)
        low_level_monitor = monitor._monitor
        assert low_level_monitor is not None
        original_close = low_level_monitor.close

        def mutate_then_close() -> None:
            nonlocal mutated
            sibling.write_bytes(b"unrelated\n")
            sibling.unlink()
            mutated = True
            original_close()

        monkeypatch.setattr(low_level_monitor, "close", mutate_then_close)
        mutation_armed = True
        return result

    monkeypatch.setattr(
        merge_evidence,
        "_retain_join_input_tree",
        retain_with_terminal_sibling_event,
    )

    rc = merge_evidence._run_join_cupti_nvbit(args)

    captured = capsys.readouterr()
    assert mutation_armed is True
    assert mutated is True
    assert rc == 0
    assert captured.err == ""
    assert (output_dir / args.platform).is_dir()
    assert not sibling.exists()


def test_merge_retires_join_when_source_mutates_during_terminal_monitor_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cupti_dir, _nvbit_dir, output_dir, args = _prepare_legacy_join_fixture(
        tmp_path
    )
    source = cupti_dir / "cupti_activity_0_0.json"
    source_payload = source.read_bytes()
    original_retain = merge_evidence._retain_join_input_tree
    mutation_armed = False
    mutated = False

    def retain_with_terminal_mutation(
        root: Path,
        *,
        source_stack: ExitStack,
        retained_authorities: list[object],
    ):
        nonlocal mutation_armed
        result = original_retain(
            root,
            source_stack=source_stack,
            retained_authorities=retained_authorities,
        )
        if Path(root).absolute() != cupti_dir.absolute():
            return result
        monitor = retained_authorities[-1]
        assert isinstance(monitor, immutable.PinnedPathSetMonitor)
        low_level_monitor = monitor._monitor
        assert low_level_monitor is not None
        original_close = low_level_monitor.close

        def mutate_then_close() -> None:
            nonlocal mutated
            replacement = source.with_name(".terminal-replacement.json")
            replacement.write_bytes(source_payload)
            os.replace(replacement, source)
            mutated = True
            original_close()

        monkeypatch.setattr(low_level_monitor, "close", mutate_then_close)
        mutation_armed = True
        return result

    monkeypatch.setattr(
        merge_evidence,
        "_retain_join_input_tree",
        retain_with_terminal_mutation,
    )

    rc = merge_evidence._run_join_cupti_nvbit(args)

    captured = capsys.readouterr()
    assert mutation_armed is True
    assert mutated is True
    assert rc == 1
    assert "publication_invalid" in captured.err
    assert not (output_dir / args.platform).exists()
    retired = sorted(
        path
        for path in output_dir.iterdir()
        if path.name.startswith(".failed-directory-")
    )
    assert len(retired) == 1
    assert any(path.is_file() for path in retired[0].rglob("joined_*.json"))


def _plant_oversized_raw_member(
    *,
    artifact_dir: Path,
    active_path: Path,
    active: dict[str, object],
    member: str,
) -> tuple[str, int, dict[str, int | dict[str, int]]]:
    raw_descriptor = active["raw_trace_generation"]
    assert isinstance(raw_descriptor, dict)
    raw_root = artifact_dir / str(raw_descriptor["path"])
    existing_sizes = [
        path.stat().st_size for path in raw_root.rglob("*") if path.is_file()
    ]
    assert existing_sizes

    if member == "unknown":
        relative = "oversized-unknown.bin"
        payload = b"x" * (max(existing_sizes) + 1)
    elif member == "stats":
        relative = "stats.csv"
        payload = (raw_root / relative).read_bytes() + b"x"
    else:
        relative = "extra_info/allocations.json"
        payload = b"[{}]\n"

    target = raw_root.joinpath(*PurePosixPath(relative).parts)
    old_size = target.stat().st_size if target.exists() else 0
    old_directory_count = sum(1 for path in raw_root.rglob("*") if path.is_dir())
    raw_root.chmod(0o755)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        for parent in target.parents:
            if parent == raw_root:
                break
            parent.chmod(0o755)
        if target.exists():
            target.chmod(0o644)
        target.write_bytes(payload)
        target.chmod(0o444)
        for parent in target.parents:
            if parent == raw_root:
                break
            parent.chmod(0o555)
    finally:
        raw_root.chmod(0o555)

    if old_size == 0:
        raw_descriptor["file_count"] = int(raw_descriptor["file_count"]) + 1
    new_directory_count = sum(
        1 for path in raw_root.rglob("*") if path.is_dir()
    )
    raw_descriptor["directory_count"] = (
        int(raw_descriptor["directory_count"])
        + new_directory_count
        - old_directory_count
    )
    raw_descriptor["total_file_bytes"] = (
        int(raw_descriptor["total_file_bytes"])
        + len(payload)
        - old_size
    )
    _replace_active_record_for_hostile_fixture(active_path, active)

    policy = dict(
        merge_evidence.raw_trace_generation_limits(
            merge_evidence.DEFAULT_TRACE_ADMISSION_LIMITS
        )
    )
    path_limits = dict(policy["max_file_bytes_by_path"])
    if member == "unknown":
        policy["max_file_bytes"] = len(payload) - 1
    else:
        policy["max_file_bytes"] = max(
            int(policy["max_file_bytes"]), len(payload)
        )
        path_limits[relative] = len(payload) - 1
    policy["max_file_bytes_by_path"] = path_limits
    return relative, len(payload), policy


@pytest.mark.parametrize("consumer", ("merge", "verifier"))
@pytest.mark.parametrize("member", ("unknown", "stats", "allocations"))
def test_consumers_reject_oversized_raw_members_before_hash_or_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    member: str,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    relative, _size, policy = _plant_oversized_raw_member(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
        member=member,
    )
    module = merge_evidence if consumer == "merge" else verify_artifacts
    monkeypatch.setattr(
        module,
        "raw_trace_generation_limits",
        lambda _limits: policy,
    )

    def reject_hash(_descriptor: int):
        raise AssertionError("oversized raw member reached content hashing")

    def reject_admission(_path: Path):
        raise AssertionError("oversized raw member reached trace admission")

    monkeypatch.setattr(immutable, "_hash_fd", reject_hash)
    monkeypatch.setattr(module, "require_trace_admission", reject_admission)

    if consumer == "merge":
        discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
            artifact_dir,
            expected_platform="rtx_3090_ti",
        )
        assert discovered == []
    else:
        errors = verify_artifacts._verify_canonical_trace_pb_admission(
            artifact_dir
        )

    assert any(
        "bounded per-file byte limit" in error and relative in error
        for error in errors
    ), errors


def test_verifier_budgets_decoded_active_graph_before_recursive_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    validator_calls = 0

    class RejectDecodedActiveGraph:
        def reserve_bytes(
            self,
            _size: int,
            *,
            retained_input_bytes: int,
            role: str,
        ) -> None:
            assert retained_input_bytes >= 0
            assert role

        def reserve(
            self,
            _value: object,
            *,
            retained_input_bytes: int,
            role: str,
        ) -> None:
            assert retained_input_bytes >= 0
            if role == "decoded canonical trace-record JSON authority":
                raise ImmutableArtifactLimitError(
                    "synthetic decoded active graph limit"
                )

    def reject_recursive_validation(*_args: object, **_kwargs: object) -> list[str]:
        nonlocal validator_calls
        validator_calls += 1
        raise AssertionError("unbounded active graph reached recursive validation")

    monkeypatch.setattr(
        verify_artifacts,
        "_CanonicalAuthorityMaterializationBudget",
        RejectDecodedActiveGraph,
    )
    monkeypatch.setattr(
        verify_artifacts,
        "_verify_active_generation_record",
        reject_recursive_validation,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert validator_calls == 0
    assert any(
        "bounded authority JSON limit" in error
        and "synthetic decoded active graph limit" in error
        for error in errors
    ), errors


def test_consumers_reject_integral_float_generated_summary_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    descriptor = active["nvbit_trace_summary_generation"]
    assert isinstance(descriptor, dict)
    files = descriptor["files"]
    assert isinstance(files, list) and files
    summary_path = artifact_dir / str(files[0]["path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = 2.0
    payload = (json.dumps(summary, sort_keys=True) + "\n").encode("utf-8")
    generation_path = summary_path.parent
    generation_path.chmod(0o755)
    summary_path.chmod(0o644)
    try:
        summary_path.write_bytes(payload)
        summary_path.chmod(0o444)
    finally:
        generation_path.chmod(0o555)
    _reseal_summary_generation_authority(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
    )
    event_descriptor = active["trace_capture_attempt_event"]
    assert isinstance(event_descriptor, dict)
    event_path = artifact_dir / str(event_descriptor["path"])
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["nvbit_trace_summary_generation"] = active[
        "nvbit_trace_summary_generation"
    ]
    event_payload = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    immutable.retire_regular_file(event_path)
    immutable.publish_regular_file(
        event_path.parent,
        event_path.name,
        event_payload,
    )
    event_descriptor["sha256"] = hashlib.sha256(event_payload).hexdigest()
    event_descriptor["size_bytes"] = len(event_payload)
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert discovered == []
    assert any(
        "exact integer schema_version 2" in error for error in merge_errors
    ), merge_errors
    assert any(
        "summary generation file" in error
        and "disagrees with admitted raw launch" in error
        for error in verifier_errors
    ), verifier_errors


def test_merge_materialization_preflight_precedes_schema_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "retained.json"
    payload = b'{"schema_version":1}\n'
    validator_calls = 0
    reserve_calls = 0

    class RejectPreflight:
        def preflight(self, _value: object, **_kwargs: object) -> None:
            raise ImmutableArtifactLimitError("synthetic pre-schema graph limit")

        def reserve(self, _value: object, **_kwargs: object) -> None:
            nonlocal reserve_calls
            reserve_calls += 1

    def reject_schema(*_args: object, **_kwargs: object) -> str | None:
        nonlocal validator_calls
        validator_calls += 1
        raise AssertionError("preflight-rejected graph reached schema validation")

    monkeypatch.setattr(merge_evidence, "_validate_against_schema", reject_schema)
    indexed, errors = merge_evidence._index_records_by_natural_key_with_counts(
        [path],
        {},
        "fixture",
        preloaded_payloads={path: payload},
        materialization_budget=RejectPreflight(),
        retained_input_bytes=len(payload),
    )

    assert indexed == {}
    assert validator_calls == 0
    assert reserve_calls == 0
    assert len(errors) == 1
    assert "joined_evidence_materialization_limit" in errors[0]
    assert "synthetic pre-schema graph limit" in errors[0]


def test_merge_manifest_preflight_precedes_jsonschema_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jsonschema

    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    validator_calls = 0

    class RejectManifestGraph:
        def preflight(
            self,
            _value: object,
            *,
            retained_input_bytes: int,
            role: str,
        ) -> None:
            assert retained_input_bytes > 0
            assert role == "decoded measurement-manifest authority"
            raise ImmutableArtifactLimitError("synthetic manifest graph limit")

    def reject_validation(*_args: object, **_kwargs: object) -> None:
        nonlocal validator_calls
        validator_calls += 1
        raise AssertionError("preflight-rejected manifest reached jsonschema")

    monkeypatch.setattr(jsonschema, "validate", reject_validation)
    preload_budget = merge_evidence._JoinPreloadBudget()
    with immutable.PinnedDirectoryReader(artifact_dir) as reader:
        inventory = dict(reader.snapshot_tree().files)
        with pytest.raises(
            ImmutableArtifactLimitError,
            match="synthetic manifest graph limit",
        ):
            merge_evidence._active_measurement_manifest_binding_error(
                active_path,
                active,
                authority_root=artifact_dir,
                authority_reader=reader,
                authority_file_inventory=inventory,
                measurement_manifest_schema={},
                preload_budget=preload_budget,
                materialization_budget=RejectManifestGraph(),
                expected_platform="rtx_3090_ti",
            )

    assert validator_calls == 0


def test_merge_attempt_event_uses_materialization_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    class RejectAttemptGraph:
        def preflight(
            self,
            _value: object,
            *,
            retained_input_bytes: int,
            role: str,
        ) -> None:
            assert retained_input_bytes > 0
            assert role == "decoded trace capture attempt-event authority"
            raise ImmutableArtifactLimitError("synthetic attempt graph limit")

    preload_budget = merge_evidence._JoinPreloadBudget()
    with immutable.PinnedDirectoryReader(artifact_dir) as reader:
        inventory = dict(reader.snapshot_tree().files)
        with pytest.raises(
            ImmutableArtifactLimitError,
            match="synthetic attempt graph limit",
        ):
            merge_evidence._active_attempt_event_binding_error(
                active_path,
                active,
                authority_root=artifact_dir,
                authority_reader=reader,
                authority_file_inventory=inventory,
                attempt_namespace_entries=[],
                preload_budget=preload_budget,
                materialization_budget=RejectAttemptGraph(),
            )


@pytest.mark.parametrize("consumer", ("merge", "verifier"))
@pytest.mark.parametrize("schema_version", (True, 1.0), ids=("boolean", "float"))
@pytest.mark.parametrize("location", ("top", "nested-raw"))
def test_consumers_reject_nonexact_attempt_event_version_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    schema_version: object,
    location: str,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    descriptor = active["trace_capture_attempt_event"]
    assert isinstance(descriptor, dict)
    event_path = artifact_dir / str(descriptor["path"])
    event = json.loads(event_path.read_text(encoding="utf-8"))
    if location == "top":
        event["schema_version"] = schema_version
    else:
        nested = event["raw_trace_generation"]
        assert isinstance(nested, dict)
        nested["schema_version"] = schema_version
    payload = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    immutable.retire_regular_file(event_path)
    immutable.publish_regular_file(
        event_path.parent,
        event_path.name,
        payload,
    )
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["size_bytes"] = len(payload)
    _replace_active_record_for_hostile_fixture(active_path, active)

    if consumer == "merge":
        discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
            artifact_dir,
            expected_platform="rtx_3090_ti",
        )
        assert discovered == []
    else:
        errors = verify_artifacts._verify_canonical_trace_pb_admission(
            artifact_dir
        )

    assert any(
        "attempt" in error.lower()
        and "body disagrees" in error.lower()
        for error in errors
    ), errors


@pytest.mark.parametrize("consumer", ("merge", "verifier"))
@pytest.mark.parametrize("node_kind", ("directory", "file"))
@pytest.mark.parametrize(
    "namespace",
    (
        run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE,
        run_suite.RAW_TRACE_GENERATION_NAMESPACE,
        run_suite.TRACE_DERIVED_GENERATION_NAMESPACE,
        run_suite.PHASE_LEDGER_GENERATION_NAMESPACE,
        run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE,
    ),
)
def test_hidden_staged_reserved_nodes_do_not_suppress_legacy_fallback(
    tmp_path: Path,
    consumer: str,
    node_kind: str,
    namespace: str,
) -> None:
    if consumer == "merge":
        _cupti, authority_root, _output, _args = _prepare_legacy_join_fixture(
            tmp_path
        )
        expected = authority_root / "nvbit_trace_summary_0_0.json"
    else:
        authority_root = tmp_path / "verifier"
        trace_shape = authority_root / "trace_fixture"
        trace_shape.mkdir(parents=True)
        expected = trace_shape / "dynamic_trace.pb"
        expected.write_bytes(b"legacy raw trace bytes")

    staged_node = authority_root / ".attempt.staging" / namespace
    staged_node.parent.mkdir(parents=True, exist_ok=True)
    if node_kind == "directory":
        staged_node.mkdir()
    else:
        staged_node.write_bytes(b"staged nonauthority")

    if consumer == "merge":
        discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
            authority_root
        )
        assert discovered == [expected]
        assert errors == []
    else:
        errors = verify_artifacts._verify_canonical_trace_pb_admission(
            authority_root
        )
        assert any(
            error.startswith("RawOnlyTraceTree:") and str(expected) in error
            for error in errors
        ), errors
        assert not any(str(staged_node) in error for error in errors)


def test_verifier_ignores_legacy_looking_trace_inside_hidden_stage(
    tmp_path: Path,
) -> None:
    authority_root = tmp_path / "artifact"
    legitimate_shape = authority_root / "trace_fixture"
    legitimate_shape.mkdir(parents=True)
    legitimate_trace = legitimate_shape / "dynamic_trace.pb"
    legitimate_trace.write_bytes(b"legitimate legacy bytes")
    hidden_shape = authority_root / ".attempt.staging" / "trace_hidden"
    hidden_shape.mkdir(parents=True)
    hidden_trace = hidden_shape / "dynamic_trace.pb"
    hidden_trace.write_bytes(b"hidden staged bytes")

    errors = verify_artifacts._verify_canonical_trace_pb_admission(authority_root)

    assert any(
        error.startswith("RawOnlyTraceTree:")
        and str(legitimate_trace) in error
        for error in errors
    ), errors
    assert not any(str(hidden_trace) in error for error in errors)


def test_run_suite_types_adapter_import_memory_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    artifact_dir = tmp_path / "artifact"
    original_import = builtins.__import__

    def exhaust_adapter_import(name: str, *args: object, **kwargs: object):
        if name == "nvbit_trace_summary_adapter":
            raise MemoryError("synthetic adapter import exhaustion")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", exhaust_adapter_import)
    result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        artifact_dir=artifact_dir,
    )

    assert result["status"] == "resource_unavailable"
    assert result["failed_hard"] is True
    assert result["paths"] == []
    assert "trace_summary_resource_unavailable" in result["failure_reason"]
    assert "synthetic adapter import exhaustion" in result["failure_reason"]
    assert not (
        artifact_dir / run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE
    ).exists()


def test_join_materialization_budget_accounts_actual_retained_roots() -> None:
    active_record = {"schema_version": 1, "shape_point": "fixture"}
    active_path = Path("artifact/trace_fixture.json")
    active_tuple = (active_path, active_record)
    active_queue = [active_tuple]
    payload = b"joined payload\n"
    immutable_file = immutable.ImmutableFile.from_bytes(
        "fixture/joined_0_0.json",
        payload,
    )
    material = [immutable_file]
    budget = merge_evidence._JoinMaterializationBudget()

    budget.reserve(
        active_queue,
        retained_input_bytes=0,
        role="actual active queue fixture",
    )
    budget.reserve(
        material,
        retained_input_bytes=0,
        role="actual immutable material fixture",
    )

    for retained in (
        active_queue,
        active_tuple,
        active_path,
        active_record,
        material,
        immutable_file,
        immutable_file.sha256,
        immutable_file.size_bytes,
        payload,
    ):
        assert budget.seen_objects[id(retained)] is retained


def test_join_accounts_production_immutable_output_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cupti, _nvbit, output_dir, args = _prepare_legacy_join_fixture(tmp_path)
    instances: list[merge_evidence._JoinMaterializationBudget] = []

    class TrackingBudget(merge_evidence._JoinMaterializationBudget):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

    monkeypatch.setattr(merge_evidence, "_JoinMaterializationBudget", TrackingBudget)

    assert merge_evidence._run_join_cupti_nvbit(args) == 0
    assert (output_dir / args.platform).is_dir()
    assert len(instances) == 1
    retained = list(instances[0].seen_objects.values())
    immutable_files = [
        value for value in retained if type(value) is immutable.ImmutableFile
    ]
    assert len(immutable_files) == 1
    immutable_file = immutable_files[0]
    assert instances[0].seen_objects[id(immutable_file.sha256)] is (
        immutable_file.sha256
    )
    assert instances[0].seen_objects[id(immutable_file.size_bytes)] is (
        immutable_file.size_bytes
    )
    assert any(
        type(value) is list
        and value
        and value[0] is immutable_file
        for value in retained
    )


def test_merge_accounts_returned_legacy_input_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cupti, nvbit_dir, _output_dir, _args = _prepare_legacy_join_fixture(tmp_path)
    instances: list[merge_evidence._JoinMaterializationBudget] = []

    class TrackingBudget(merge_evidence._JoinMaterializationBudget):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

    monkeypatch.setattr(merge_evidence, "_JoinMaterializationBudget", TrackingBudget)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(nvbit_dir)

    assert errors == []
    assert len(discovered) == 1
    assert len(instances) == 1
    retained = instances[0].seen_objects
    assert retained[id(discovered)] is discovered
    assert retained[id(discovered[0])] is discovered[0]
    assert any(
        type(value) is immutable.PinnedTreeSnapshot
        for value in retained.values()
    )


def test_merge_uses_one_explicit_raw_admission_and_rejects_pre_retention_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    raw_descriptor = active["raw_trace_generation"]
    assert isinstance(raw_descriptor, dict)
    raw_root = artifact_dir / str(raw_descriptor["path"])
    stats_path = raw_root / "stats.csv"
    stats_payload = stats_path.read_bytes()
    real_require = merge_evidence.require_trace_admission
    real_open = merge_evidence.open_admitted_trace_sources
    require_calls = 0
    retained_open_calls = 0

    def count_require(path: Path):
        nonlocal require_calls
        require_calls += 1
        return real_require(path)

    def mutate_before_retained_open(*args: object, **kwargs: object):
        nonlocal retained_open_calls
        retained_open_calls += 1
        raw_root.chmod(0o755)
        try:
            replacement = raw_root / ".stats-replacement"
            replacement.write_bytes(stats_payload)
            replacement.chmod(0o444)
            os.replace(replacement, stats_path)
        finally:
            raw_root.chmod(0o555)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(merge_evidence, "require_trace_admission", count_require)
    monkeypatch.setattr(
        merge_evidence,
        "open_admitted_trace_sources",
        mutate_before_retained_open,
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )

    assert discovered == []
    assert require_calls == 1
    assert retained_open_calls == 1
    assert any(
        "changed before summary retention" in error
        or "authority_changed" in error
        for error in errors
    ), errors


def test_merge_accounts_production_active_path_record_tuples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    instances: list[merge_evidence._JoinMaterializationBudget] = []
    reserve_roles: list[str] = []

    class TrackingBudget(merge_evidence._JoinMaterializationBudget):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

        def reserve(
            self,
            value: object,
            *,
            retained_input_bytes: int,
            role: str,
        ) -> None:
            reserve_roles.append(role)
            super().reserve(
                value,
                retained_input_bytes=retained_input_bytes,
                role=role,
            )

    monkeypatch.setattr(merge_evidence, "_JoinMaterializationBudget", TrackingBudget)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir,
        expected_platform="rtx_3090_ti",
    )

    assert errors == []
    assert discovered == [
        artifact_dir / path for path in active["nvbit_trace_summary_paths"]
    ]
    assert len(instances) == 1
    retained = list(instances[0].seen_objects.values())
    active_tuples = [
        value
        for value in retained
        if type(value) is tuple
        and len(value) == 3
        and isinstance(value[0], Path)
        and type(value[1]) is dict
        and value[1].get("activation_state") == "active"
        and type(value[2]) is merge_evidence.AdmittedTraceSources
    ]
    assert len(active_tuples) == 2
    for retained_tuple in active_tuples:
        initialized_path_references: list[object] = []
        for path_type in type(retained_tuple[0]).__mro__:
            slots = path_type.__dict__.get("__slots__", ())
            if type(slots) is str:
                slots = (slots,)
            for slot in slots:
                if slot == "__weakref__":
                    continue
                descriptor = path_type.__dict__.get(slot)
                assert descriptor is not None
                try:
                    initialized_path_references.append(
                        descriptor.__get__(
                            retained_tuple[0], type(retained_tuple[0])
                        )
                    )
                except AttributeError:
                    pass
        for retained_path_reference in initialized_path_references:
            assert (
                instances[0].seen_objects[id(retained_path_reference)]
                is retained_path_reference
            )
    matching_active_tuples = [
        value for value in active_tuples if value[0] == active_path
    ]
    assert len(matching_active_tuples) == 1
    active_tuple = matching_active_tuples[0]
    assert instances[0].seen_objects[id(active_tuple[0])] is active_tuple[0]
    assert instances[0].seen_objects[id(active_tuple[1])] is active_tuple[1]
    assert any(
        type(value) is list and active_tuple in value for value in retained
    )
    for path in discovered:
        assert instances[0].seen_objects[id(path)] is path
    assert any(type(value) is immutable.PinnedTreeSnapshot for value in retained)
    assert any(
        type(value) is immutable.ImmutableDirectoryGeneration
        for value in retained
    )
    assert any(
        type(value) is immutable.DirectoryGenerationFile for value in retained
    )
    assert "decoded measurement manifest schema" in reserve_roles


def test_verifier_accounts_production_discovery_and_generation_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, active_path, _active = _prepare_generation_authority_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    instances: list[
        verify_artifacts._CanonicalAuthorityMaterializationBudget
    ] = []
    reserve_roles: list[str] = []

    class TrackingBudget(
        verify_artifacts._CanonicalAuthorityMaterializationBudget
    ):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

        def reserve(
            self,
            value: object,
            *,
            retained_input_bytes: int,
            role: str,
        ) -> None:
            reserve_roles.append(role)
            super().reserve(
                value,
                retained_input_bytes=retained_input_bytes,
                role=role,
            )

    monkeypatch.setattr(
        verify_artifacts,
        "_CanonicalAuthorityMaterializationBudget",
        TrackingBudget,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(artifact_dir)

    assert errors == []
    assert len(instances) == 1
    retained = list(instances[0].seen_objects.values())
    assert any(type(value) is immutable.PinnedTreeSnapshot for value in retained)
    assert any(
        type(value) is immutable.ImmutableDirectoryGeneration
        for value in retained
    )
    assert any(
        type(value) is immutable.DirectoryGenerationFile for value in retained
    )
    queued_records = [
        value
        for value in retained
        if type(value) is tuple
        and len(value) == 2
        and isinstance(value[0], Path)
        and type(value[1]) is dict
        and value[1].get("activation_state") == "active"
    ]
    assert len(queued_records) == 1
    queued_path = queued_records[0][0]
    initialized_path_references: list[object] = []
    for path_type in type(queued_path).__mro__:
        slots = path_type.__dict__.get("__slots__", ())
        if type(slots) is str:
            slots = (slots,)
        for slot in slots:
            if slot == "__weakref__":
                continue
            descriptor = path_type.__dict__.get(slot)
            assert descriptor is not None
            try:
                initialized_path_references.append(
                    descriptor.__get__(queued_path, type(queued_path))
                )
            except AttributeError:
                pass
    for retained_path_reference in initialized_path_references:
        assert (
            instances[0].seen_objects[id(retained_path_reference)]
            is retained_path_reference
        )
    assert queued_path == active_path
    assert instances[0].seen_objects[id(queued_records[0])] is queued_records[0]
    assert instances[0].seen_objects[id(queued_path)] is queued_path
    assert "retained canonical authority discovery inventories" in reserve_roles
    assert "decoded NVBit trace summary schema" in reserve_roles
    assert "decoded measurement manifest schema" in reserve_roles
    assert "retained summary generation inventory" in reserve_roles
    assert "retained phase generation inventory" not in reserve_roles


@pytest.mark.parametrize(
    "failed_stage",
    ("features", "summary_publish", "manifest", "optional_phase"),
)
def test_handle_trace_pre_activation_failure_retires_stale_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    active_path = artifact_dir / "trace_trace_00.json"
    previous_payload = b'{"previous_success":true}\n'
    active_path.write_bytes(previous_payload)
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )

    if failed_stage == "features":
        def fail_features(**_kwargs) -> None:
            raise ImmutableArtifactError("planted feature failure")

        monkeypatch.setattr(
            run_suite, "_emit_admitted_trace_features", fail_features
        )
    elif failed_stage == "summary_publish":
        monkeypatch.setattr(
            run_suite,
            "_emit_nvbit_trace_summaries_for_trace_artifact",
            lambda **_kwargs: {
                "status": "adapter_error",
                "paths": [],
                "failure_reason": "planted summary failure",
                "failed_hard": True,
            },
        )
    elif failed_stage == "manifest":
        def fail_manifest(*_args, **_kwargs):
            raise ImmutableArtifactError("planted manifest failure")

        monkeypatch.setattr(run_suite, "_emit_manifest", fail_manifest)
    else:
        monkeypatch.setattr(
            run_suite,
            "_run_phase_inject_ab_for_handle_trace",
            lambda **_kwargs: {
                "return_code": run_suite.EXIT_BUILD_FAILED,
                "typed_reason": "planted_optional_phase_failure",
                "generation": None,
                "staging_path": None,
                "summary": None,
            },
        )

    rc = run_suite.handle_trace(
        row,
        props,
        tmp_path,
        shape_point_name="trace_00",
        phase_inject_ab=failed_stage == "optional_phase",
    )

    assert rc == run_suite.EXIT_BUILD_FAILED
    assert not active_path.exists()
    retired = tuple(
        artifact_dir.glob(".trace_trace_00.json.*.retired")
    )
    assert len(retired) == 1
    assert retired[0].read_bytes() == previous_payload


def test_handle_trace_previous_success_then_failed_rerun_preserves_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    first_rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )
    assert first_rc == run_suite.EXIT_OK
    active_path = artifact_dir / "trace_trace_00.json"
    previous_payload = active_path.read_bytes()
    previous = json.loads(previous_payload)
    previous_summary = artifact_dir / previous[
        "nvbit_trace_summary_paths"
    ][0]
    previous_summary_payload = previous_summary.read_bytes()
    previous_raw = artifact_dir / previous["raw_trace_generation"]["path"]
    previous_raw_descriptor = immutable.inspect_immutable_directory_generation(
        previous_raw,
        max_files=previous["raw_trace_generation"]["file_count"],
        max_directories=previous["raw_trace_generation"]["directory_count"],
        max_total_file_bytes=previous["raw_trace_generation"][
            "total_file_bytes"
        ],
        max_total_relative_path_bytes=32 * 1024 * 1024,
    )
    previous_raw_hashes = _raw_file_hashes(previous_raw)

    def fail_features(**_kwargs) -> None:
        raise ImmutableArtifactError("planted rerun feature failure")

    monkeypatch.setattr(
        run_suite, "_emit_admitted_trace_features", fail_features
    )
    second_rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    assert second_rc == run_suite.EXIT_BUILD_FAILED
    assert not active_path.exists()
    retired = tuple(
        artifact_dir.glob(".trace_trace_00.json.*.retired")
    )
    assert len(retired) == 1
    assert retired[0].read_bytes() == previous_payload
    assert previous_summary.read_bytes() == previous_summary_payload
    observed_previous_raw = immutable.inspect_immutable_directory_generation(
        previous_raw,
        max_files=previous_raw_descriptor.file_count,
        max_directories=previous_raw_descriptor.directory_count,
        max_total_file_bytes=previous_raw_descriptor.total_file_bytes,
        max_total_relative_path_bytes=32 * 1024 * 1024,
    )
    assert observed_previous_raw == previous_raw_descriptor
    assert _raw_file_hashes(previous_raw) == previous_raw_hashes
    raw_generations = tuple(
        path for path in (
            artifact_dir
            / run_suite.RAW_TRACE_GENERATION_NAMESPACE
            / "trace_00"
        ).iterdir()
        if not path.name.startswith(".")
    )
    assert len(raw_generations) == 2
    failed_raw = next(path for path in raw_generations if path != previous_raw)
    assert _raw_file_hashes(failed_raw)
    failed_attempt = failed_raw.name
    failure_event = (
        artifact_dir
        / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
        / "trace_00"
        / failed_attempt
        / "preactivation_failed.json"
    )
    failure = json.loads(failure_event.read_text(encoding="utf-8"))
    assert failure["activation_state"] == "non_active"
    assert failure["attempt_id"] == failed_attempt
    assert failure["raw_trace_generation"]["path"] == str(
        failed_raw.relative_to(artifact_dir)
    )


def test_handle_trace_two_attempt_generations_coexist_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_OK
    active_path = artifact_dir / "trace_trace_00.json"
    first_payload = active_path.read_bytes()
    first = json.loads(first_payload)
    first_raw = artifact_dir / first["raw_trace_generation"]["path"]
    first_descriptor = immutable.inspect_immutable_directory_generation(
        first_raw,
        max_files=first["raw_trace_generation"]["file_count"],
        max_directories=first["raw_trace_generation"]["directory_count"],
        max_total_file_bytes=first["raw_trace_generation"][
            "total_file_bytes"
        ],
        max_total_relative_path_bytes=32 * 1024 * 1024,
    )
    first_hashes = _raw_file_hashes(first_raw)
    first_summary = artifact_dir / first["nvbit_trace_summary_paths"][0]
    first_summary_payload = first_summary.read_bytes()

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_OK
    second = json.loads(active_path.read_text(encoding="utf-8"))
    second_raw = artifact_dir / second["raw_trace_generation"]["path"]

    assert first["trace_capture_attempt_id"] != second[
        "trace_capture_attempt_id"
    ]
    assert first_raw != second_raw
    assert first_raw.is_dir() and second_raw.is_dir()
    assert immutable.inspect_immutable_directory_generation(
        first_raw,
        max_files=first_descriptor.file_count,
        max_directories=first_descriptor.directory_count,
        max_total_file_bytes=first_descriptor.total_file_bytes,
        max_total_relative_path_bytes=32 * 1024 * 1024,
    ) == first_descriptor
    assert _raw_file_hashes(first_raw) == first_hashes
    assert first_summary.read_bytes() == first_summary_payload
    assert first["nvbit_trace_summary_paths"] != second[
        "nvbit_trace_summary_paths"
    ]
    assert (
        artifact_dir / second["nvbit_trace_summary_paths"][0]
    ).is_file()
    raw_root = (
        artifact_dir
        / run_suite.RAW_TRACE_GENERATION_NAMESPACE
        / "trace_00"
    )
    assert {
        child.name for child in raw_root.iterdir()
        if not child.name.startswith(".")
    } == {
        first["trace_capture_attempt_id"],
        second["trace_capture_attempt_id"],
    }
    derived_root = (
        artifact_dir
        / run_suite.TRACE_DERIVED_GENERATION_NAMESPACE
        / "trace_00"
    )
    assert {
        child.name for child in derived_root.iterdir()
        if not child.name.startswith(".")
    } == {
        first["trace_capture_attempt_id"],
        second["trace_capture_attempt_id"],
    }
    retired = tuple(
        artifact_dir.glob(".trace_trace_00.json.*.retired")
    )
    assert len(retired) == 1
    assert retired[0].read_bytes() == first_payload
    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert errors == []
    assert discovered == [
        artifact_dir / second["nvbit_trace_summary_paths"][0]
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("wrong_path", "not the exact active attempt path"),
        ("content_manifest", "content descriptor is stale"),
        ("portable_seal", "admission seals are stale"),
        ("filesystem_seal", "admission seals are stale"),
        ("directory_filesystem_seal", "filesystem seal is stale"),
        ("file_count_bound", "hard admission bound"),
    ),
)
def test_merge_rejects_wrong_or_stale_active_raw_generation_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_OK
    active_path = artifact_dir / "trace_trace_00.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    raw = active["raw_trace_generation"]

    if mutation == "wrong_path":
        raw["path"] = (
            f"{run_suite.RAW_TRACE_GENERATION_NAMESPACE}/trace_00/"
            "attempt-stale-reference"
        )
    elif mutation == "content_manifest":
        raw["content_manifest_sha256"] = (
            "0" * 64
            if raw["content_manifest_sha256"] != "0" * 64
            else "1" * 64
        )
    elif mutation == "portable_seal":
        replacement = (
            "0" * 64
            if active["trace_admission_sha256"] != "0" * 64
            else "1" * 64
        )
        active["trace_admission_sha256"] = replacement
        raw["trace_admission_sha256"] = replacement
    elif mutation == "filesystem_seal":
        replacement = (
            "0" * 64
            if active["trace_filesystem_generation_sha256"] != "0" * 64
            else "1" * 64
        )
        active["trace_filesystem_generation_sha256"] = replacement
        raw["trace_filesystem_generation_sha256"] = replacement
    elif mutation == "directory_filesystem_seal":
        raw["filesystem_generation_sha256"] = (
            "0" * 64
            if raw["filesystem_generation_sha256"] != "0" * 64
            else "1" * 64
        )
    else:
        raw["file_count"] = (
            merge_evidence.DEFAULT_TRACE_ADMISSION_LIMITS.tree_files + 1
        )
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "joined_evidence_nvbit_raw_generation_invalid" in errors[0]
    assert expected_error in errors[0]


def test_merge_preserves_raw_generation_resource_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_OK

    def fail_resource(*_args, **_kwargs):
        raise ImmutableArtifactResourceError(
            "VERIFY_RESOURCE_BOUND: planted raw verifier exhaustion"
        )

    monkeypatch.setattr(
        merge_evidence,
        "inspect_immutable_directory_generation",
        fail_resource,
    )
    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert len(errors) == 1
    assert "raw_generation_resource_unavailable" in errors[0]
    assert "planted raw verifier exhaustion" in errors[0]


def test_merge_rejects_symlink_indirection_for_active_raw_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_OK
    active = json.loads(
        (artifact_dir / "trace_trace_00.json").read_text(encoding="utf-8")
    )
    raw = artifact_dir / active["raw_trace_generation"]["path"]
    retained = raw.with_name(f".{raw.name}.test-retained")
    raw.rename(retained)
    raw.symlink_to(retained.name)
    try:
        discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
            artifact_dir
        )
        assert discovered == []
        assert len(errors) == 1
        assert "joined_evidence_nvbit_discovery_invalid" in errors[0]
        assert "symlink" in errors[0]
        assert run_suite.RAW_TRACE_GENERATION_NAMESPACE in errors[0]
    finally:
        raw.unlink()
        retained.rename(raw)


def test_handle_trace_preserves_preexisting_attempt_stage_as_typed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-fixed-stage"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)
    hidden_stage = (
        artifact_dir
        / run_suite.RAW_TRACE_GENERATION_NAMESPACE
        / "trace_00"
        / f".{attempt_id}.staging"
    )
    hidden_stage.mkdir(parents=True)
    marker = hidden_stage / "marker"
    marker.write_bytes(b"preserved interrupted staging")
    capture_called = False

    def reject_capture(**_kwargs):
        nonlocal capture_called
        capture_called = True
        raise AssertionError("preexisting stage must prevent capture")

    monkeypatch.setattr(
        run_suite, "_run_retained_nvbit_capture", reject_capture
    )
    rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    assert rc == run_suite.EXIT_BUILD_FAILED
    assert capture_called is False
    assert marker.read_bytes() == b"preserved interrupted staging"
    assert not (artifact_dir / "trace_trace_00.json").exists()
    event_path = (
        artifact_dir
        / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
        / "trace_00"
        / attempt_id
        / "stage_creation_failed.json"
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["activation_state"] == "non_active"
    assert event["status"] == "stage_creation_failed"
    assert event["typed_reason"] == "trace_filesystem_invalid"
    assert event["raw_staging_path"] == str(
        hidden_stage.relative_to(artifact_dir)
    )


def test_handle_trace_interrupted_raw_commit_preserves_frozen_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-fixed-commit"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)
    original_rename = immutable._rename_noreplace_at

    def interrupt_raw_commit(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        if (
            source_name == f".{attempt_id}.staging"
            and destination_name == attempt_id
        ):
            raise OSError(errno.EIO, "planted raw commit interruption")
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        immutable, "_rename_noreplace_at", interrupt_raw_commit
    )
    rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    hidden_stage = (
        artifact_dir
        / run_suite.RAW_TRACE_GENERATION_NAMESPACE
        / "trace_00"
        / f".{attempt_id}.staging"
    )
    final = hidden_stage.with_name(attempt_id)
    assert rc == run_suite.EXIT_BUILD_FAILED
    assert hidden_stage.is_dir()
    assert not final.exists()
    assert (hidden_stage / "dynamic_trace.pb").is_file()
    assert stat.S_IMODE(hidden_stage.stat().st_mode) == 0o555
    assert stat.S_IMODE(
        (hidden_stage / "dynamic_trace.pb").stat().st_mode
    ) == 0o444
    assert not (artifact_dir / "trace_trace_00.json").exists()
    event_path = (
        artifact_dir
        / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
        / "trace_00"
        / attempt_id
        / "raw_commit_failed.json"
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["activation_state"] == "non_active"
    assert event["status"] == "raw_commit_failed"
    assert event["typed_reason"] == "trace_filesystem_invalid"
    assert event["raw_staging_path"] == str(
        hidden_stage.relative_to(artifact_dir)
    )
    # Test-only permission restoration for pytest cleanup.
    for directory in sorted(
        (path for path in hidden_stage.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
    ):
        directory.chmod(0o755)
    hidden_stage.chmod(0o755)


def test_handle_trace_postcommit_input_failure_binds_raw_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-postcommit-input-failure"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)
    original_verify = immutable.RetainedExecutable.verify_unchanged
    verify_count = 0

    def fail_after_raw_commit(retained) -> None:
        nonlocal verify_count
        verify_count += 1
        original_verify(retained)
        if verify_count == 4:
            raise ImmutableArtifactError(
                "planted executable failure after raw commit"
            )

    monkeypatch.setattr(
        immutable.RetainedExecutable,
        "verify_unchanged",
        fail_after_raw_commit,
    )

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_BUILD_FAILED
    assert not (artifact_dir / "trace_trace_00.json").exists()

    event_path = (
        artifact_dir
        / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
        / "trace_00"
        / attempt_id
        / "raw_commit_failed.json"
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))
    descriptor = event["raw_trace_generation"]
    assert event["status"] == "raw_commit_failed"
    assert event["typed_reason"] == "trace_filesystem_invalid"
    assert event["raw_staging_path"] is None
    assert descriptor["path"] == (
        f"{run_suite.RAW_TRACE_GENERATION_NAMESPACE}/trace_00/{attempt_id}"
    )
    assert (artifact_dir / descriptor["path"]).is_dir()


def _install_raw_stage_close_failure(
    monkeypatch: pytest.MonkeyPatch,
    error_factory,
) -> None:
    original_close = immutable.ExclusiveDirectoryStage.close

    def close_then_fail(stage) -> None:
        original_close(stage)
        if run_suite.RAW_TRACE_GENERATION_NAMESPACE in stage.output_root.parts:
            raise error_factory()

    monkeypatch.setattr(
        immutable.ExclusiveDirectoryStage,
        "close",
        close_then_fail,
    )


@pytest.mark.parametrize(
    ("failure_kind", "expected_reason"),
    (
        ("filesystem", "trace_filesystem_invalid"),
        ("resource", "trace_snapshot_resource_unavailable"),
    ),
)
def test_handle_trace_raw_stage_close_after_commit_is_typed_and_preserves_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_reason: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = f"attempt-raw-close-{failure_kind}"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)

    def cleanup_error():
        if failure_kind == "resource":
            return ImmutableArtifactResourceError(
                "planted raw stage resource close failure"
            )
        return OSError(errno.EIO, "planted raw stage close failure")

    _install_raw_stage_close_failure(monkeypatch, cleanup_error)

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_BUILD_FAILED

    final = (
        artifact_dir
        / run_suite.RAW_TRACE_GENERATION_NAMESPACE
        / "trace_00"
        / attempt_id
    )
    assert final.is_dir()
    assert (final / "dynamic_trace.pb").is_file()
    assert not (artifact_dir / "trace_trace_00.json").exists()
    event = json.loads(
        (
            artifact_dir
            / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
            / "trace_00"
            / attempt_id
            / "raw_commit_failed.json"
        ).read_text(encoding="utf-8")
    )
    assert event["status"] == "raw_stage_closure_failed"
    assert event["typed_reason"] == expected_reason
    assert event["raw_staging_path"] is None
    assert event["raw_trace_generation"]["path"] == str(
        final.relative_to(artifact_dir)
    )


def test_handle_trace_raw_stage_close_preserves_selected_capture_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-capture-failure-close"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)

    def fail_capture(**_kwargs):
        raise run_suite.TracerRuntimeError("planted tracer dependency failure")

    monkeypatch.setattr(run_suite, "_run_retained_nvbit_capture", fail_capture)
    _install_raw_stage_close_failure(
        monkeypatch,
        lambda: OSError(errno.EIO, "planted raw close failure"),
    )

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_BUILD_FAILED

    event = json.loads(
        (
            artifact_dir
            / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
            / "trace_00"
            / attempt_id
            / "capture_failed.json"
        ).read_text(encoding="utf-8")
    )
    assert event["status"] == "capture_failed"
    assert event["typed_reason"] == "tracer_so_deps_unresolved"


def test_handle_trace_raw_stage_close_preserves_typed_unavailable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-typed-unavailable-close"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)

    def typed_capture(**_kwargs):
        return {
            "baseline_proc": SimpleNamespace(
                returncode=0,
                stdout='{"result_kind":"typed_unavailable"}',
                stderr="",
            ),
            "baseline_elapsed_s": 0.001,
            "baseline_result": {
                "result_kind": "typed_unavailable",
                "typed_reason": "planted_unsupported_device",
            },
            "trace_proc": None,
            "trace_elapsed_s": 0.0,
            "binary_sha256": hashlib.sha256(
                b"retained executable fixture"
            ).hexdigest(),
            "tracer_build_sha256": hashlib.sha256(
                b"retained tracer fixture"
            ).hexdigest(),
        }

    monkeypatch.setattr(run_suite, "_run_retained_nvbit_capture", typed_capture)
    _install_raw_stage_close_failure(
        monkeypatch,
        lambda: OSError(errno.EIO, "planted raw close failure"),
    )

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_OK

    tombstone = json.loads(
        (artifact_dir / "trace_trace_00.json").read_text(encoding="utf-8")
    )
    assert tombstone["activation_state"] == "non_active"
    assert tombstone["typed_reason"] == "planted_unsupported_device"
    event = json.loads(
        (
            artifact_dir
            / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
            / "trace_00"
            / attempt_id
            / "typed_unavailable.json"
        ).read_text(encoding="utf-8")
    )
    assert event["status"] == "typed_unavailable"
    assert event["typed_reason"] == "planted_unsupported_device"


def test_handle_trace_unexpected_raw_error_remains_primary_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )

    def fail_capture(**_kwargs):
        raise RuntimeError("planted unexpected capture error")

    monkeypatch.setattr(run_suite, "_run_retained_nvbit_capture", fail_capture)
    _install_raw_stage_close_failure(
        monkeypatch,
        lambda: OSError(errno.EIO, "planted raw close failure"),
    )

    with pytest.raises(
        RuntimeError, match="planted unexpected capture error"
    ) as exc_info:
        run_suite.handle_trace(
            row, props, tmp_path, shape_point_name="trace_00"
        )

    assert any(
        "raw trace stage cleanup also failed" in note
        and "planted raw close failure" in note
        for note in getattr(exc_info.value, "__notes__", ())
    )


def test_handle_trace_postcommit_recovery_failure_binds_final_raw_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-postcommit-recovery-failure"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)
    original_commit = immutable.ExclusiveDirectoryStage.commit

    def commit_then_raise(stage, **kwargs):
        generation = original_commit(stage, **kwargs)
        if run_suite.RAW_TRACE_GENERATION_NAMESPACE in stage.output_root.parts:
            raise ImmutableArtifactError(
                "planted failure after committed raw rename"
            )
        return generation

    final = (
        artifact_dir
        / run_suite.RAW_TRACE_GENERATION_NAMESPACE
        / "trace_00"
        / attempt_id
    )
    original_inspect = run_suite.inspect_immutable_directory_generation

    def fail_raw_recovery(path: Path, **kwargs):
        if Path(path) == final:
            raise ImmutableArtifactError(
                "planted committed raw recovery inspection failure"
            )
        return original_inspect(path, **kwargs)

    monkeypatch.setattr(
        immutable.ExclusiveDirectoryStage,
        "commit",
        commit_then_raise,
    )
    monkeypatch.setattr(
        run_suite,
        "inspect_immutable_directory_generation",
        fail_raw_recovery,
    )

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_BUILD_FAILED
    assert final.is_dir()
    assert not (artifact_dir / "trace_trace_00.json").exists()

    event_path = (
        artifact_dir
        / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
        / "trace_00"
        / attempt_id
        / "raw_commit_failed.json"
    )
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["raw_trace_generation"] is None
    assert event["raw_staging_path"] is None
    assert event["raw_unresolved_committed_path"] == str(
        final.relative_to(artifact_dir)
    )


def test_handle_trace_unexpected_failure_preserves_primary_on_source_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_with_cleanup(*_args, source_stack, **_kwargs) -> int:
        def fail_cleanup() -> None:
            raise ImmutableArtifactError("planted outer source close failure")

        source_stack.callback(fail_cleanup)
        raise RuntimeError("planted outer primary failure")

    monkeypatch.setattr(
        run_suite,
        "_handle_trace_with_sources",
        fail_with_cleanup,
    )

    with pytest.raises(
        RuntimeError, match="planted outer primary failure"
    ) as exc_info:
        run_suite.handle_trace({}, run_suite.DeviceProps(), tmp_path)

    assert any(
        "trace source cleanup also failed" in note
        and "planted outer source close failure" in note
        for note in getattr(exc_info.value, "__notes__", ())
    )


def test_handle_trace_cleanup_failure_preserves_selected_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def select_typed_failure(*_args, source_stack, **_kwargs) -> int:
        def fail_cleanup() -> None:
            raise ImmutableArtifactError("planted retained-source close failure")

        source_stack.callback(fail_cleanup)
        return run_suite.EXIT_SAMPLING_REQUIRED

    monkeypatch.setattr(
        run_suite,
        "_handle_trace_with_sources",
        select_typed_failure,
    )

    assert run_suite.handle_trace(
        {}, run_suite.DeviceProps(), tmp_path
    ) == run_suite.EXIT_SAMPLING_REQUIRED
    assert "trace_source_cleanup_failed_after_typed_result" in capsys.readouterr().err


def test_handle_trace_real_derived_cleanup_preserves_selected_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    monkeypatch.setattr(
        run_suite,
        "_run_phase_inject_ab_for_handle_trace",
        lambda **_kwargs: {
            "return_code": run_suite.EXIT_SAMPLING_REQUIRED,
            "typed_reason": "planted_phase_refusal",
            "generation": None,
            "staging_path": None,
            "summary": None,
        },
    )
    original_close = immutable.ExclusiveDirectoryStage.close

    def fail_derived_close(stage) -> None:
        original_close(stage)
        if run_suite.TRACE_DERIVED_GENERATION_NAMESPACE in stage.output_root.parts:
            raise OSError(errno.EIO, "planted derived close failure")

    monkeypatch.setattr(
        immutable.ExclusiveDirectoryStage,
        "close",
        fail_derived_close,
    )

    assert run_suite.handle_trace(
        row,
        props,
        tmp_path,
        shape_point_name="trace_00",
        phase_inject_ab=True,
        ab_repeat_count=1,
    ) == run_suite.EXIT_SAMPLING_REQUIRED
    assert not (artifact_dir / "trace_trace_00.json").exists()
    assert "trace_derived_stage_cleanup_failed" in capsys.readouterr().err


def test_handle_trace_retires_activation_on_derived_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    attempt_id = "attempt-derived-close-failure"
    monkeypatch.setattr(run_suite, "_new_trace_attempt_id", lambda: attempt_id)
    original_close = immutable.ExclusiveDirectoryStage.close

    def fail_derived_close(stage) -> None:
        original_close(stage)
        if run_suite.TRACE_DERIVED_GENERATION_NAMESPACE in stage.output_root.parts:
            raise OSError(errno.EIO, "planted derived close failure")

    monkeypatch.setattr(
        immutable.ExclusiveDirectoryStage,
        "close",
        fail_derived_close,
    )

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_BUILD_FAILED
    active_path = artifact_dir / "trace_trace_00.json"
    assert not active_path.exists()
    retired = tuple(artifact_dir.glob(".trace_trace_00.json.*.retired"))
    assert len(retired) == 1
    event = json.loads(
        (
            artifact_dir
            / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
            / "trace_00"
            / attempt_id
            / "activation_failed.json"
        ).read_text(encoding="utf-8")
    )
    assert event["status"] == "activation_derived_stage_closure_failed"
    assert event["activation_state"] == "non_active"


def test_handle_trace_activation_rollback_preserves_concurrent_locator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    active_path = artifact_dir / "trace_trace_00.json"
    concurrent_payload = b'{"concurrent":"active-replacement"}\n'
    published_payload = []
    expected_retirements = []
    original_retire = run_suite.retire_regular_file

    def record_retirement(path, *args, **kwargs):
        if Path(path) == active_path and kwargs.get("expected_payload") is not None:
            expected_retirements.append(kwargs["expected_payload"])
        return original_retire(path, *args, **kwargs)

    monkeypatch.setattr(run_suite, "retire_regular_file", record_retirement)
    original_close = immutable.ExclusiveDirectoryStage.close
    raced = False

    def replace_locator_then_fail(stage) -> None:
        nonlocal raced
        original_close(stage)
        if (
            not raced
            and run_suite.TRACE_DERIVED_GENERATION_NAMESPACE
            in stage.output_root.parts
        ):
            raced = True
            published_payload.append(active_path.read_bytes())
            replacement = active_path.with_name(".concurrent-active")
            replacement.write_bytes(concurrent_payload)
            replacement.chmod(0o444)
            os.replace(replacement, active_path)
            raise OSError(errno.EIO, "planted derived close failure after race")

    monkeypatch.setattr(
        immutable.ExclusiveDirectoryStage,
        "close",
        replace_locator_then_fail,
    )

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_BUILD_FAILED

    assert raced is True
    assert active_path.read_bytes() == concurrent_payload
    assert len(published_payload) == 1
    assert expected_retirements == published_payload
    assert all(
        path.read_bytes() != concurrent_payload
        for path in artifact_dir.glob(".trace_trace_00.json.*.retired")
    )
    assert (
        "trace_failed_activation_retirement_failed"
        in capsys.readouterr().err
    )


def test_handle_trace_tombstone_rollback_preserves_concurrent_locator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    active_path = artifact_dir / "trace_trace_00.json"
    concurrent_payload = b'{"concurrent":"tombstone-replacement"}\n'
    published_payload = []
    expected_retirements = []
    original_retire = run_suite.retire_regular_file

    def record_retirement(path, *args, **kwargs):
        if Path(path) == active_path and kwargs.get("expected_payload") is not None:
            expected_retirements.append(kwargs["expected_payload"])
        return original_retire(path, *args, **kwargs)

    monkeypatch.setattr(run_suite, "retire_regular_file", record_retirement)

    def typed_capture(**_kwargs):
        return {
            "baseline_proc": SimpleNamespace(
                returncode=0,
                stdout='{"result_kind":"typed_unavailable"}',
                stderr="",
            ),
            "baseline_elapsed_s": 0.001,
            "baseline_result": {
                "result_kind": "typed_unavailable",
                "typed_reason": "planted_unsupported_device",
            },
            "trace_proc": None,
            "trace_elapsed_s": 0.0,
            "binary_sha256": hashlib.sha256(
                b"retained executable fixture"
            ).hexdigest(),
            "tracer_build_sha256": hashlib.sha256(
                b"retained tracer fixture"
            ).hexdigest(),
        }

    monkeypatch.setattr(run_suite, "_run_retained_nvbit_capture", typed_capture)
    original_close = immutable.RetainedRegular.close
    raced = False

    def close_tracer_after_replacement(retained) -> None:
        nonlocal raced
        original_close(retained)
        if not raced and active_path.exists():
            raced = True
            published_payload.append(active_path.read_bytes())
            replacement = active_path.with_name(".concurrent-tombstone")
            replacement.write_bytes(concurrent_payload)
            replacement.chmod(0o444)
            os.replace(replacement, active_path)
            raise ImmutableArtifactError(
                "planted retained tracer close failure after race"
            )

    monkeypatch.setattr(
        immutable.RetainedRegular,
        "close",
        close_tracer_after_replacement,
    )

    assert run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    ) == run_suite.EXIT_BUILD_FAILED

    assert raced is True
    assert active_path.read_bytes() == concurrent_payload
    assert len(published_payload) == 1
    assert expected_retirements == published_payload
    assert all(
        path.read_bytes() != concurrent_payload
        for path in artifact_dir.glob(".trace_trace_00.json.*.retired")
    )
    stderr = capsys.readouterr().err
    assert "trace_failed_tombstone_retirement_failed" in stderr
    assert "trace_tombstone_publication_failed" in stderr


@pytest.mark.parametrize("repeat_count", (0, -1))
def test_handle_trace_rejects_nonpositive_phase_repeat_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repeat_count: int,
) -> None:
    def unexpected_dispatch(*_args, **_kwargs):
        raise AssertionError("invalid repeat count reached trace dispatch")

    monkeypatch.setattr(
        run_suite,
        "_handle_trace_with_sources",
        unexpected_dispatch,
    )
    marker = tmp_path / "existing-activation"
    marker.write_bytes(b"must remain unchanged")

    assert run_suite.handle_trace(
        {},
        run_suite.DeviceProps(),
        tmp_path,
        phase_inject_ab=True,
        ab_repeat_count=repeat_count,
    ) == run_suite.EXIT_USAGE
    assert marker.read_bytes() == b"must remain unchanged"


def test_run_suite_cli_rejects_nonpositive_phase_repeat_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_preflight(*_args, **_kwargs):
        raise AssertionError("invalid repeat count reached repository or GPU preflight")

    monkeypatch.setattr(run_suite, "load_roster", unexpected_preflight)
    monkeypatch.setattr(run_suite, "startup_identity_check", unexpected_preflight)

    assert run_suite.main([
        "--mode",
        "trace",
        "--phase-inject-ab",
        "--ab-repeat-count",
        "0",
    ]) == run_suite.EXIT_USAGE


def test_handle_trace_crash_after_retirement_leaves_no_stale_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    active_path = artifact_dir / "trace_trace_00.json"
    previous_payload = b'{"previous_success":true}\n'
    active_path.write_bytes(previous_payload)
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    original_retire = run_suite.retire_regular_file

    def retire_then_crash(path: Path):
        retired = original_retire(path)
        raise RuntimeError(f"planted crash after retirement: {retired}")

    monkeypatch.setattr(run_suite, "retire_regular_file", retire_then_crash)
    with pytest.raises(RuntimeError, match="planted crash after retirement"):
        run_suite.handle_trace(
            row, props, tmp_path, shape_point_name="trace_00"
        )

    assert not active_path.exists()
    retired = tuple(
        artifact_dir.glob(".trace_trace_00.json.*.retired")
    )
    assert len(retired) == 1
    assert retired[0].read_bytes() == previous_payload


@pytest.mark.parametrize("crash_after_commit", (False, True))
def test_handle_trace_activation_exception_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after_commit: bool,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    original_activation = run_suite._publish_trace_artifact_record

    def crash(path: Path, record: dict[str, object]) -> None:
        if crash_after_commit:
            original_activation(path, record)
        raise RuntimeError("planted activation crash")

    monkeypatch.setattr(run_suite, "_publish_trace_artifact_record", crash)
    with pytest.raises(RuntimeError, match="planted activation crash"):
        run_suite.handle_trace(
            row, props, tmp_path, shape_point_name="trace_00"
        )

    active_path = artifact_dir / "trace_trace_00.json"
    assert active_path.exists() is crash_after_commit
    attempt_events = tuple(
        (
            artifact_dir
            / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
            / "trace_00"
        ).glob("*/activation_ready.json")
    )
    assert len(attempt_events) == 1
    ready = json.loads(attempt_events[0].read_text(encoding="utf-8"))
    assert ready["activation_state"] == "non_active"
    assert ready["status"] == "activation_ready"
    assert (
        artifact_dir / ready["raw_trace_generation"]["path"]
    ).is_dir()
    assert (
        artifact_dir
        / ready["nvbit_trace_summary_generation"]["files"][0]["path"]
    ).is_file()
    if crash_after_commit:
        active = json.loads(active_path.read_text(encoding="utf-8"))
        assert active["activation_state"] == "active"
        assert active["nvbit_trace_summary_generation"]["file_count"] == 1


def test_handle_trace_activation_destination_race_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    original_activation = run_suite._publish_trace_artifact_record
    planted = b'{"concurrent":true}\n'

    def plant_race(path: Path, record: dict[str, object]) -> None:
        path.write_bytes(planted)
        original_activation(path, record)

    monkeypatch.setattr(
        run_suite, "_publish_trace_artifact_record", plant_race
    )
    rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    active_path = artifact_dir / "trace_trace_00.json"
    assert rc == run_suite.EXIT_BUILD_FAILED
    assert active_path.read_bytes() == planted
    activation_failures = tuple(
        (
            artifact_dir
            / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
            / "trace_00"
        ).glob("*/activation_failed.json")
    )
    assert len(activation_failures) == 1
    failure = json.loads(
        activation_failures[0].read_text(encoding="utf-8")
    )
    assert failure["activation_state"] == "non_active"
    assert failure["status"] == "activation_failed"
    assert failure["typed_reason"] == "trace_activation_failed"
    assert (
        artifact_dir / failure["raw_trace_generation"]["path"]
    ).is_dir()
    assert (
        artifact_dir
        / failure["nvbit_trace_summary_generation"]["files"][0]["path"]
    ).is_file()


@pytest.mark.parametrize(
    "mutated_source", ("executable", "tracer", "raw_trace")
)
def test_handle_trace_retires_activation_when_retained_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_source: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    executable = tmp_path / "global_scalar_32b_copy_run"
    tracer = tmp_path / "tracer_tool.so"
    original_activation = run_suite._publish_trace_artifact_record

    def mutate_then_publish(path: Path, record: dict[str, object]) -> None:
        if mutated_source == "raw_trace":
            raw_record = record["raw_trace_generation"]
            assert isinstance(raw_record, dict)
            raw_root = artifact_dir / str(raw_record["path"])
            target = raw_root / "dynamic_trace.pb"
            original = target.read_bytes()
            target.chmod(0o644)
            target.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
            target.write_bytes(original)
            target.chmod(0o444)
        else:
            target = executable if mutated_source == "executable" else tracer
            replacement = target.with_name(f"{target.name}.replacement")
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o755 if target == executable else 0o644)
            os.replace(replacement, target)
        original_activation(path, record)

    monkeypatch.setattr(
        run_suite,
        "_publish_trace_artifact_record",
        mutate_then_publish,
    )
    rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    active_path = artifact_dir / "trace_trace_00.json"
    assert rc == run_suite.EXIT_BUILD_FAILED
    assert not active_path.exists()
    failed_activations = tuple(
        artifact_dir.glob(".trace_trace_00.json.*.retired")
    )
    assert len(failed_activations) == 1
    failed_record = json.loads(
        failed_activations[0].read_text(encoding="utf-8")
    )
    assert failed_record["activation_state"] == "active"
    activation_failures = tuple(
        (
            artifact_dir
            / run_suite.TRACE_CAPTURE_ATTEMPT_NAMESPACE
            / "trace_00"
        ).glob("*/activation_failed.json")
    )
    assert len(activation_failures) == 1
    failure = json.loads(
        activation_failures[0].read_text(encoding="utf-8")
    )
    assert failure["status"] == "activation_failed"


@pytest.mark.parametrize(
    "node_kind", ("directory", "symlink", "hardlink", "fifo")
)
def test_handle_trace_retirement_refuses_special_or_linked_active_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node_kind: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    active_path = artifact_dir / "trace_trace_00.json"
    target = artifact_dir / "retained-target.json"
    target_payload = b"retained target bytes\n"
    target.write_bytes(target_payload)
    if node_kind == "directory":
        active_path.mkdir()
        (active_path / "marker").write_bytes(b"planted directory")
    elif node_kind == "symlink":
        active_path.symlink_to(target)
    elif node_kind == "hardlink":
        os.link(target, active_path)
    else:
        os.mkfifo(active_path)

    row, props = _configure_successful_trace_handle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        artifact_dir=artifact_dir,
    )
    capture_called = False

    def reject_capture(**_kwargs):
        nonlocal capture_called
        capture_called = True
        raise AssertionError("capture must not run after retirement refusal")

    monkeypatch.setattr(
        run_suite, "_run_retained_nvbit_capture", reject_capture
    )
    rc = run_suite.handle_trace(
        row, props, tmp_path, shape_point_name="trace_00"
    )

    assert rc == run_suite.EXIT_BUILD_FAILED
    assert capture_called is False
    assert target.read_bytes() == target_payload
    assert active_path.lstat()
    if node_kind == "directory":
        assert (active_path / "marker").read_bytes() == b"planted directory"
    assert not tuple(
        artifact_dir.glob(".trace_trace_00.json.*.retired")
    )


def test_raw_generation_policy_rejects_heterogeneous_generic_limits() -> None:
    limits = replace(
        run_suite.DEFAULT_TRACE_ADMISSION_LIMITS,
        threadblock_bytes=(
            run_suite.DEFAULT_TRACE_ADMISSION_LIMITS.threadblock_bytes - 1
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires equal trace-root and threadblock per-file limits",
    ):
        merge_evidence.raw_trace_generation_limits(limits)


def test_trace_generation_commit_limits_bind_actual_per_file_ceiling() -> None:
    limits = run_suite.DEFAULT_TRACE_ADMISSION_LIMITS
    commit_limits = run_suite._trace_generation_commit_limits()

    assert commit_limits["max_file_bytes"] == max(
        limits.trace_root_bytes,
        limits.threadblock_bytes,
    )
    assert commit_limits["max_file_bytes_by_path"] == {
        "stats.csv": limits.stats_bytes,
        "extra_info/allocations.json": limits.allocations_sidecar_bytes,
    }
    assert commit_limits["max_file_bytes"] < commit_limits[
        "max_total_file_bytes"
    ]

    derived_limits = run_suite._derived_generation_commit_limits()
    assert "max_file_bytes" not in derived_limits
    assert "max_file_bytes_by_path" not in derived_limits
    assert (
        derived_limits["max_total_file_bytes"]
        == commit_limits["max_total_file_bytes"]
    )


def test_run_suite_executes_and_injects_retained_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "fixture_run"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"tracer bytes")
    traces = tmp_path / "trace"
    traces.mkdir()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_runtime_env(path: Path, *, retained_descriptor: int):
        assert path == tracer
        assert retained_descriptor >= 0
        return {}

    def fake_run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(run_suite, "tracer_runtime_env", fake_runtime_env)
    monkeypatch.setattr("subprocess.run", fake_run)
    traces_descriptor = os.open(traces, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with immutable.open_retained_executable(executable) as retained_executable:
            with immutable.open_retained_regular(tracer) as retained_tracer:
                result = run_suite._run_retained_nvbit_capture(
                    retained_executable=retained_executable,
                    retained_tracer=retained_tracer,
                    args=["--fixture"],
                    traces_folder=traces,
                    traces_descriptor=traces_descriptor,
                )
    finally:
        os.close(traces_descriptor)

    assert len(calls) == 2
    baseline_command, baseline_kwargs = calls[0]
    trace_command, trace_kwargs = calls[1]
    assert baseline_command[0].startswith("/proc/self/fd/")
    assert trace_command[0] == baseline_command[0]
    executable_fd = int(trace_command[0].rsplit("/", 1)[1])
    tracer_path = str(trace_kwargs["env"]["CUDA_INJECTION64_PATH"])
    assert tracer_path.startswith("/proc/self/fd/")
    tracer_fd = int(tracer_path.rsplit("/", 1)[1])
    expected_capture_path = f"/proc/self/fd/{traces_descriptor}"
    assert baseline_kwargs["env"]["TRACES_FOLDER"] == expected_capture_path
    assert trace_kwargs["env"]["TRACES_FOLDER"] == expected_capture_path
    assert baseline_kwargs["pass_fds"] == (
        executable_fd,
        traces_descriptor,
    )
    assert trace_kwargs["pass_fds"] == (
        executable_fd,
        tracer_fd,
        traces_descriptor,
    )
    assert result["binary_sha256"] == hashlib.sha256(
        executable.read_bytes()
    ).hexdigest()
    assert result["tracer_build_sha256"] == hashlib.sha256(
        tracer.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("mutated_input", ("executable", "tracer"))
def test_run_suite_rejects_retained_input_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_input: str,
) -> None:
    executable = tmp_path / "fixture_run"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"tracer bytes")
    traces = tmp_path / "trace"
    traces.mkdir()
    call_count = 0

    monkeypatch.setattr(
        run_suite,
        "tracer_runtime_env",
        lambda _path, *, retained_descriptor: {},
    )

    def fake_run(_command: list[str], **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            target = executable if mutated_input == "executable" else tracer
            replacement = target.with_suffix(target.suffix + ".replacement")
            replacement.write_bytes(target.read_bytes())
            if target == executable:
                replacement.chmod(0o755)
            os.replace(replacement, target)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(
        ImmutableArtifactError,
        match="named regular-file path changed",
    ):
        with immutable.open_retained_executable(
            executable
        ) as retained_executable:
            with immutable.open_retained_regular(tracer) as retained_tracer:
                run_suite._run_retained_nvbit_capture(
                    retained_executable=retained_executable,
                    retained_tracer=retained_tracer,
                    args=[],
                    traces_folder=traces,
                )


def test_nvbit_budget_sidecars_are_attempt_scoped_and_preserve_legacy_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    legacy_payloads = {
        "nvbit_slowdown_budget_exceeded.json": b"legacy slowdown bytes\n",
        "nvbit_drop_rate_exceeded.json": b"legacy drop bytes\n",
    }
    for name, payload in legacy_payloads.items():
        (artifact_dir / name).write_bytes(payload)
    monkeypatch.setattr(
        run_suite._trace_budgets,
        "NVBIT_SLOWDOWN_BUDGET_MIN_BASELINE_TIME_S",
        0.0,
    )
    monkeypatch.setattr(
        run_suite._trace_budgets,
        "NVBIT_SLOWDOWN_BUDGET_MAX_RATIO",
        1.0,
    )
    monkeypatch.setattr(
        run_suite._trace_budgets,
        "NVBIT_DROP_RATE_BUDGET_MAX",
        0.1,
    )

    first_root = artifact_dir / ".attempt-one.staging"
    second_root = artifact_dir / ".attempt-two.staging"
    first_root.mkdir()
    second_root.mkdir()
    first = run_suite._emit_nvbit_budget_sidecars(
        output_root=first_root,
        artifact_dir=artifact_dir,
        slug="fixture",
        shape_point="trace_00",
        actual_slowdown_ratio=2.0,
        baseline_wall_time_s=1.0,
        actual_wall_time_s=2.0,
        actual_drop_rate=0.5,
        expected_record_count=100,
        admitted_record_count=50,
    )
    first_payloads = {
        kind: path.read_bytes()
        for kind, path in first.items()
        if path is not None
    }
    assert set(first_payloads) == {"slowdown", "drop_rate"}
    for name, payload in legacy_payloads.items():
        assert not (artifact_dir / name).exists()
        retired = tuple(artifact_dir.glob(f".{name}.*.retired"))
        assert len(retired) == 1
        assert retired[0].read_bytes() == payload

    second = run_suite._emit_nvbit_budget_sidecars(
        output_root=second_root,
        artifact_dir=artifact_dir,
        slug="fixture",
        shape_point="trace_00",
        actual_slowdown_ratio=3.0,
        baseline_wall_time_s=1.0,
        actual_wall_time_s=3.0,
        actual_drop_rate=0.75,
        expected_record_count=100,
        admitted_record_count=25,
    )
    assert all(path is not None for path in second.values())
    assert all(
        path.read_bytes() == first_payloads[kind]
        for kind, path in first.items()
        if path is not None
    )
    assert first["slowdown"] != second["slowdown"]
    assert first["drop_rate"] != second["drop_rate"]
    assert stat.S_IMODE(first["slowdown"].stat().st_mode) == 0o444
    assert stat.S_IMODE(first["drop_rate"].stat().st_mode) == 0o444


def test_run_suite_rechecks_admitted_root_size_without_deleting_source(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    admitted_size = admission.dynamic_trace_size_bytes
    sidecar = tmp_path / "artifact" / "trace_volume_exceeded.json"

    exceeded = run_suite._admitted_trace_exceeds_budget(
        trace_admission=admission,
        budget_bytes=admitted_size - 1,
        sidecar=sidecar,
        slug="fixture",
        shape_point="fixture_shape",
        actual_wall_time_s=1.25,
    )

    assert exceeded is True
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["actual_trace_bytes"] == admitted_size
    assert record["max_trace_bytes"] == admitted_size - 1
    assert (trace_dir / "dynamic_trace.pb").is_file()
    first_payload = sidecar.read_bytes()

    exceeded = run_suite._admitted_trace_exceeds_budget(
        trace_admission=admission,
        budget_bytes=admitted_size - 1,
        sidecar=sidecar,
        slug="fixture",
        shape_point="fixture_shape",
        actual_wall_time_s=2.5,
    )
    assert exceeded is True
    second_record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert second_record["actual_wall_time_s"] == 2.5
    retired = tuple(
        sidecar.parent.glob(".trace_volume_exceeded.json.*.retired")
    )
    assert len(retired) == 1
    assert retired[0].read_bytes() == first_payload

    exceeded = run_suite._admitted_trace_exceeds_budget(
        trace_admission=admission,
        budget_bytes=admitted_size,
        sidecar=sidecar,
        slug="fixture",
        shape_point="fixture_shape",
        actual_wall_time_s=1.25,
    )
    assert exceeded is False
    assert not sidecar.exists()
    retired = tuple(sidecar.parent.glob(".trace_volume_exceeded.json.*.retired"))
    assert len(retired) == 2
    retired_records = {
        entry["actual_wall_time_s"]
        for entry in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in retired
        )
    }
    assert retired_records == {record["actual_wall_time_s"], 2.5}


def test_run_suite_emits_all_summaries_after_trace_tree_disappears(
    tmp_path: Path,
) -> None:
    trace_dir = _write_trace(tmp_path / "trace")
    admission = require_trace_admission(trace_dir)
    executable = tmp_path / "fixture_run"
    tracer = tmp_path / "tracer_tool.so"
    executable.write_bytes(b"executable")
    tracer.write_bytes(b"tracer")
    shutil.rmtree(trace_dir)

    result = run_suite._emit_nvbit_trace_summaries_for_trace_artifact(
        trace_admission=admission,
        slug="fixture",
        shape_point="fixture_shape",
        binary_sha256=hashlib.sha256(b"executable").hexdigest(),
        tracer_build_sha256=hashlib.sha256(b"tracer").hexdigest(),
        artifact_dir=tmp_path,
    )

    assert result == {
        "status": "ok",
        "paths": _summary_generation_relative_paths(
            admission,
            shape_point="fixture_shape",
        ),
        "failure_reason": None,
        "failed_hard": False,
    }


def test_binding_consumers_offer_no_generation_or_graceful_fallback() -> None:
    forbidden = (
        "nvbit-pb-python",
        "skipped_adapter_import_failed",
        "skipped_missing_protobuf_bindings",
        'typed_reason="protobuf_bindings_missing"',
        "degrade gracefully",
    )
    candidates = sorted(
        {
            *REPO_ROOT.joinpath("tools").rglob("*.py"),
            *SUITE_ROOT.rglob("*.py"),
        }
    )
    inspected = 0
    for path in candidates:
        if path.name.startswith("test_") or "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if (
            "proto_binding_loader" not in source
            and "_import_pb_modules" not in source
            and "TrackedBindingsCorrupt" not in source
        ):
            continue
        inspected += 1
        for token in forbidden:
            assert token not in source, (path, token)
    assert inspected >= 10


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("kernel_name", "other_kernel"),
        ("kernel_signature_sha256", "c" * 64),
        ("inst_count", 2),
        ("binary_sha256", "c" * 64),
        ("nvbit_trace_pb_path", "other/dynamic_trace.pb"),
        ("nvbit_trace_pb_sha256", "c" * 64),
        ("nvbit_tracer_build_hash", "c" * 64),
        ("nvbit_threadblock_payload_size_bytes", 2),
        ("nvbit_threadblock_payload_manifest_sha256", "c" * 64),
    ),
)
def test_verifier_binds_complete_summary_body_to_raw_launch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    _add_minimal_manifest_authority(
        artifact_dir=artifact_dir,
        active=active,
    )
    summary_path = artifact_dir / result["paths"][0]
    generation_path = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = replacement
    generation_path.chmod(0o755)
    try:
        summary_path.chmod(0o644)
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_path.chmod(0o444)
    finally:
        generation_path.chmod(0o555)
    _reseal_summary_generation_authority(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any(
        "disagrees with admitted raw launch and execution authority" in error
        for error in errors
    )


def test_verifier_extra_summary_launch_returns_typed_error(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    admission, result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    _add_minimal_manifest_authority(
        artifact_dir=artifact_dir,
        active=active,
    )
    summary_path = artifact_dir / result["paths"][0]
    generation_path = summary_path.parent
    extra = json.loads(summary_path.read_text(encoding="utf-8"))
    extra["launch_index"] = admission.launch_count
    extra_path = generation_path / (
        f"nvbit_trace_summary_0_{admission.launch_count}.json"
    )
    generation_path.chmod(0o755)
    try:
        extra_path.write_text(
            json.dumps(extra, indent=2) + "\n",
            encoding="utf-8",
        )
        extra_path.chmod(0o444)
    finally:
        generation_path.chmod(0o555)
    _reseal_summary_generation_authority(
        artifact_dir=artifact_dir,
        active_path=active_path,
        active=active,
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any("names an unadmitted launch" in error for error in errors)
    assert any(
        "does not equal the admitted raw launch authority" in error
        for error in errors
    )


@pytest.mark.parametrize(
    "mutation",
    ("kernel_signature_sha256", "binary_sha256", "nvbit_tracer_build_hash"),
)
def test_consumers_bind_phase_body_to_raw_launch_and_execution(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _summaries, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    values = {
        "kernel_signature_sha256": active["kernel_signature_sha256"],
        "binary_sha256": "a" * 64,
        "nvbit_tracer_build_hash": "b" * 64,
    }
    values[mutation] = "c" * 64
    payload = _valid_phase_ledger_payload(
        slug="fixture",
        shape_point="fixture_shape",
        repeat_index=0,
        launch_index=0,
        binary_sha256=values["binary_sha256"],
        kernel_signature_sha256=values["kernel_signature_sha256"],
        tracer_build_sha256=values["nvbit_tracer_build_hash"],
    )
    _bind_manifest_and_phase_generation(
        artifact_dir=artifact_dir,
        active_path=active_path,
        repeat_count=1,
        payloads={"phase_cycle_ledger_0_0.json": payload},
    )

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert any(
        "phase ledger body disagrees with admitted raw launch" in error
        for error in merge_errors
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any(
        "phase ledger generation file" in error
        and "disagrees with admitted raw launch" in error
        for error in verifier_errors
    )


def test_merge_markerless_canonical_candidate_blocks_legacy_fallback(
    tmp_path: Path,
) -> None:
    (tmp_path / "nvbit_trace_summary_0_0.json").write_text(
        '{"schema_version": 1}\n',
        encoding="utf-8",
    )
    (tmp_path / "trace_fixture.json").write_text("[]\n", encoding="utf-8")

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(tmp_path)
    assert discovered == []
    assert any("canonical trace-record candidate" in error for error in errors)


@pytest.mark.parametrize(
    "record",
    (
        [],
        {"activation_state": "non_active", "shape_point": "fixture"},
        {
            "shape_point": "fixture",
            "phase_inject_ab": {
                "requested": False,
                "repeat_count": 0,
                "generation": None,
                "evidence": [],
            },
        },
    ),
    ids=("markerless", "non_active", "phase_only"),
)
def test_verifier_canonical_candidate_disables_legacy_fallback(
    tmp_path: Path,
    record: object,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "trace_fixture.json").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any("CanonicalGenerationAuthorityMissing" in error for error in errors)


def test_verifier_retains_active_record_through_generation_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, _active_path, _active = (
        _prepare_generation_authority_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )
    original = verify_artifacts._verify_active_generation_record
    replaced = False

    def replace_active_then_verify(
        record_path: Path,
        record: dict[str, object],
        **kwargs: object,
    ) -> list[str]:
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement = record_path.with_name(".active-replacement.json")
            replacement.write_bytes(record_path.read_bytes())
            replacement.chmod(0o444)
            os.replace(replacement, record_path)
        return original(record_path, record, **kwargs)

    monkeypatch.setattr(
        verify_artifacts,
        "_verify_active_generation_record",
        replace_active_then_verify,
    )
    errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert replaced is True
    assert any("trace authority changed" in error for error in errors)


@pytest.mark.parametrize(
    "schema_version",
    (2, True, 1.0),
    ids=("unknown-int", "boolean", "integral-float"),
)
def test_consumers_reject_nonexact_phase_generation_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _summaries, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    base_active = json.loads(active_path.read_text(encoding="utf-8"))
    active = _bind_manifest_and_phase_generation(
        artifact_dir=artifact_dir,
        active_path=active_path,
        repeat_count=1,
        payloads={
            "phase_cycle_ledger_0_0.json": _valid_phase_ledger_payload(
                slug=str(base_active["slug"]),
                shape_point="fixture_shape",
                repeat_index=0,
                launch_index=0,
                binary_sha256=str(base_active["binary_sha256"]),
                kernel_signature_sha256=str(
                    base_active["kernel_signature_sha256"]
                ),
                tracer_build_sha256=str(
                    base_active["nvbit_tracer_build_hash"]
                ),
            )
        },
    )
    phase = active["phase_inject_ab"]
    assert isinstance(phase, dict)
    descriptor = phase["generation"]
    assert isinstance(descriptor, dict)
    descriptor["schema_version"] = schema_version
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )

    assert discovered == []
    assert any(
        "phase generation schema_version must equal one" in error
        for error in merge_errors
    )
    assert any(
        "phase generation schema_version must equal one" in error
        for error in verifier_errors
    )


def test_verifier_bounds_phase_payload_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    admission, _summaries, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    base_active = json.loads(active_path.read_text(encoding="utf-8"))
    active = _bind_manifest_and_phase_generation(
        artifact_dir=artifact_dir,
        active_path=active_path,
        repeat_count=1,
        payloads={
            "phase_cycle_ledger_0_0.json": _valid_phase_ledger_payload(
                slug=str(base_active["slug"]),
                shape_point="fixture_shape",
                repeat_index=0,
                launch_index=0,
                binary_sha256=str(base_active["binary_sha256"]),
                kernel_signature_sha256=str(
                    base_active["kernel_signature_sha256"]
                ),
                tracer_build_sha256=str(
                    base_active["nvbit_tracer_build_hash"]
                ),
            )
        },
    )
    original = immutable.PinnedDirectoryReader.read_snapshot
    observed_limits: list[int | None] = []

    def track_phase_read(reader, relative_path: str, **kwargs):
        if PurePosixPath(relative_path).name.startswith("phase_cycle_ledger_"):
            observed_limits.append(kwargs.get("max_bytes"))
        return original(reader, relative_path, **kwargs)

    monkeypatch.setattr(
        immutable.PinnedDirectoryReader,
        "read_snapshot",
        track_phase_read,
    )

    errors = verify_artifacts._verify_active_phase_generation(
        active_path,
        active,
        trace_admission=admission,
    )

    assert errors == []
    assert observed_limits
    assert set(observed_limits) == {
        verify_artifacts._CANONICAL_AUTHORITY_JSON_MAX_BYTES
    }


def test_verifier_rejects_unbounded_phase_repeat_without_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    admission, _summaries, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = _bind_manifest_and_phase_generation(
        artifact_dir=artifact_dir,
        active_path=active_path,
        repeat_count=10**12,
        payloads={
            "phase_cycle_ledger_0_0.json": _valid_phase_ledger_payload(
                slug="fixture",
                shape_point="fixture_shape",
                repeat_index=0,
                launch_index=0,
                binary_sha256="a" * 64,
                kernel_signature_sha256=(
                    adapter.compute_kernel_signature_sha256(
                        admission.launches[0].kernel_signature_inputs()
                    )
                ),
                tracer_build_sha256="b" * 64,
            )
        },
    )

    original_range = range

    def reject_unbounded_range(value: int):
        if value > 4096:
            raise AssertionError(
                "unbounded phase authority reached range expansion"
            )
        return original_range(value)

    monkeypatch.setattr(
        verify_artifacts,
        "range",
        reject_unbounded_range,
        raising=False,
    )
    errors = verify_artifacts._verify_active_phase_generation(
        active_path,
        active,
        trace_admission=admission,
    )
    assert errors == [
        "requested phase repeat-by-launch authority exceeds the hard "
        "phase generation file bound"
    ]


def test_consumers_reject_self_consistent_partial_summary_generation(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    shape_point = "fixture_shape"
    admission, raw_descriptor, attempt_id = _publish_raw_trace_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace_with_launch_count(tmp_path / "trace", 2),
        shape_point=shape_point,
    )
    scratch = tmp_path / "summary.json"
    adapter.emit_admitted_nvbit_trace_summary(
        admission=admission,
        slug="fixture",
        shape_point=shape_point,
        repeat_index=0,
        launch_index=0,
        binary_sha256="a" * 64,
        tracer_build_sha256="b" * 64,
        output_path=scratch,
    )
    filename = "nvbit_trace_summary_0_0.json"
    final = immutable.publish_directory(
        artifact_dir
        / run_suite.NVBIT_TRACE_SUMMARY_GENERATION_NAMESPACE
        / shape_point
        / admission.trace_admission_sha256,
        admission.trace_filesystem_generation_sha256,
        [immutable.ImmutableFile.from_bytes(filename, scratch.read_bytes())],
        accept_existing=False,
    )
    relative_path = (final.relative_to(artifact_dir) / filename).as_posix()
    active_path = _write_active_trace_record(
        artifact_dir,
        admission=admission,
        shape_point=shape_point,
        paths=[relative_path],
        raw_trace_generation=raw_descriptor,
        attempt_id=attempt_id,
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    _add_minimal_manifest_authority(
        artifact_dir=artifact_dir,
        active=active,
    )
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert any(
        "does not equal the admitted raw launch authority" in error
        for error in merge_errors
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any(
        "does not equal the admitted raw launch authority" in error
        for error in verifier_errors
    )


def test_consumers_reject_phase_generation_missing_admitted_launches(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    admission, summaries, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace_with_launch_count(tmp_path / "trace", 2),
        shape_point="fixture_shape",
    )
    assert len(summaries["paths"]) == admission.launch_count == 2
    _bind_manifest_and_phase_generation(
        artifact_dir=artifact_dir,
        active_path=active_path,
        repeat_count=2,
        payloads={
            f"phase_cycle_ledger_{repeat_index}_0.json": (
                _valid_phase_ledger_payload(
                    slug="fixture",
                    shape_point="fixture_shape",
                    repeat_index=repeat_index,
                    launch_index=0,
                )
            )
            for repeat_index in range(2)
        },
    )

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert any(
        "repeat-by-launch authority" in error for error in merge_errors
    )
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any(
        "repeat-by-launch authority" in error for error in verifier_errors
    )


@pytest.mark.parametrize("mutation", ("invalid_json", "boolean_index"))
def test_consumers_authenticate_phase_ledger_bodies(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _summaries, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    if mutation == "invalid_json":
        payload = b"{invalid-json\n"
    else:
        ledger = json.loads(_valid_phase_ledger_payload(
            slug="fixture",
            shape_point="fixture_shape",
            repeat_index=0,
            launch_index=0,
        ))
        ledger["repeat_index"] = False
        payload = (json.dumps(ledger, sort_keys=True) + "\n").encode("utf-8")
    _bind_manifest_and_phase_generation(
        artifact_dir=artifact_dir,
        active_path=active_path,
        repeat_count=1,
        payloads={"phase_cycle_ledger_0_0.json": payload},
    )

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert any("phase ledger" in error for error in merge_errors)
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any("phase ledger" in error for error in verifier_errors)


def test_merge_rejects_explicit_non_active_generation_authority(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    active["activation_state"] = "non_active"
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert any("explicitly active canonical trace record" in error for error in errors)


def test_merge_malformed_authority_blocks_legacy_fallback(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "nvbit_trace_summary_0_0.json"
    legacy.write_text('{"schema_version": 1}\n', encoding="utf-8")
    immutable.publish_regular_file(
        tmp_path,
        "trace_fixture.json",
        (
            json.dumps({
                "activation_state": "active",
                "nvbit_trace_summary_emit_status": "ok",
            })
            + "\n"
        ).encode("utf-8"),
    )

    discovered, errors = merge_evidence._discover_nvbit_summary_inputs(tmp_path)
    assert discovered == []
    assert any("explicitly active canonical trace record" in error for error in errors)


def test_verifier_returns_error_for_unhashable_active_shape(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    (artifact_dir / run_suite.RAW_TRACE_GENERATION_NAMESPACE).mkdir(
        parents=True
    )
    immutable.publish_regular_file(
        artifact_dir,
        "trace_bad.json",
        (
            json.dumps({
                "schema_version": 1,
                "activation_state": "active",
                "shape_set": "trace_shapes",
                "nvbit_trace_summary_emit_status": "ok",
                "shape_point": [],
            })
            + "\n"
        ).encode("utf-8"),
    )

    errors = verify_artifacts._verify_canonical_trace_pb_admission(tmp_path)
    assert any("malformed active shape authority" in error for error in errors)


def test_consumers_reject_boolean_unrequested_repeat_count(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    _admission, _result, active_path = _publish_active_summary_fixture(
        artifact_dir=artifact_dir,
        source_root=_write_trace(tmp_path / "trace"),
        shape_point="fixture_shape",
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    _add_minimal_manifest_authority(
        artifact_dir=artifact_dir,
        active=active,
    )
    active["phase_inject_ab"] = {
        "requested": False,
        "repeat_count": False,
        "generation": None,
        "evidence": [],
    }
    _replace_active_record_for_hostile_fixture(active_path, active)

    discovered, merge_errors = merge_evidence._discover_nvbit_summary_inputs(
        artifact_dir
    )
    assert discovered == []
    assert any("invalid repeat_count" in error for error in merge_errors)
    verifier_errors = verify_artifacts._verify_canonical_trace_pb_admission(
        artifact_dir
    )
    assert any("invalid repeat_count" in error for error in verifier_errors)
