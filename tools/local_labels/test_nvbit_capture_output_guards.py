"""Two output-guard tests per inventory row, plus trace admission regressions."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

from tools.local_labels import activation_trace_orchestrator as activation
from tools.local_labels import campaign_preflight as campaign
from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import roster_path_resolver as resolver
from tools.local_labels import roster_trace_capture as roster_capture
from tools.local_labels import sm120_activation_bundle_producer as sm120
from tools.local_labels import stage_measurement_manifests_from_predictor_trace as staging
from tools.local_labels import trace_register_cap_variant_cells as variants
from tools.local_labels.canary import emit_canary_manifest as canary
from tools.local_labels.immutable_artifact import (
    VERIFY_RESOURCE_BOUND,
    ImmutableArtifactError,
    ImmutableArtifactResourceError,
)
from tools.local_labels.protected_paths import ProtectedPathError, is_protected_path
from tools.nvbit_tracer.trace_admission import TraceAdmission, TraceLaunchSummary


REPO = Path(__file__).resolve().parents[2]
STAGING = REPO / "results" / "nvbit_capture_staging"
PROTECTED = REPO / "data" / "local_labels" / "campaign"
GOLDEN_SCRIPT = REPO / "tools" / "trace_sampling" / "golden_cells" / "capture_golden_cells.sh"


def _admitted_trace(
    *,
    launch_count: int = 1,
    with_allocations: bool = False,
) -> TraceAdmission:
    launch = TraceLaunchSummary(
        launch_index=0,
        device_id=0,
        stream_id=0,
        kernel_id=7,
        kernel_name="fixture_kernel",
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=32,
        block_y=1,
        block_z=1,
        shared_mem_static_bytes=0,
        shared_mem_dynamic_bytes=0,
        regs_per_thread=16,
        inst_count=7,
    )
    launches = (launch,)
    if launch_count == 2:
        launches = (
            launch,
            replace(launch, launch_index=1, kernel_id=8),
        )
    return TraceAdmission(
        launch_count=launch_count,
        threadblock_count=launch_count,
        instruction_count=7 * launch_count,
        stats_row_count=1,
        expected_record_count=7 * launch_count,
        admitted_record_count=7 * launch_count,
        dynamic_trace_path="fixture/dynamic_trace.pb",
        dynamic_trace_size_bytes=101,
        dynamic_trace_sha256="1" * 64,
        allocations_sidecar_size_bytes=11 if with_allocations else None,
        allocations_sidecar_sha256="5" * 64 if with_allocations else None,
        stats_size_bytes=23,
        stats_sha256="2" * 64,
        threadblock_payload_size_bytes=37 * launch_count,
        threadblock_payload_manifest_sha256="3" * 64,
        trace_filesystem_generation_sha256="4" * 64,
        launches=launches,
    )


def _assert_sidecar_admission_identities(
    record: dict[str, object],
    admission: TraceAdmission,
) -> None:
    assert record["trace_bytes"] == admission.dynamic_trace_size_bytes
    assert record["dynamic_trace_size_bytes"] == admission.dynamic_trace_size_bytes
    assert record["dynamic_trace_sha256"] == admission.dynamic_trace_sha256
    assert record["stats_size_bytes"] == admission.stats_size_bytes
    assert record["stats_sha256"] == admission.stats_sha256
    assert (
        record["threadblock_payload_size_bytes"]
        == admission.threadblock_payload_size_bytes
    )
    assert (
        record["threadblock_payload_manifest_sha256"]
        == admission.threadblock_payload_manifest_sha256
    )
    assert (
        record["allocations_sidecar_size_bytes"]
        == admission.allocations_sidecar_size_bytes
    )
    assert (
        record["allocations_sidecar_sha256"]
        == admission.allocations_sidecar_sha256
    )
    assert record["trace_admission_sha256"] == admission.trace_admission_sha256
    assert (
        record["trace_filesystem_generation_sha256"]
        == admission.trace_filesystem_generation_sha256
    )
    assert (
        record["trace_filesystem_generation_sha256"]
        != record["trace_admission_sha256"]
    )


def _write_executable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)


class _RetainedBytesFixture:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.verify_count = 0

    def read_bytes(self) -> bytes:
        return self.payload

    def verify_unchanged(self) -> None:
        self.verify_count += 1


def _activation_capture_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admission: TraceAdmission,
) -> tuple[dict[str, Path], bytes]:
    repo = tmp_path / "activation_repo"
    slug = "fixture_activation"
    binary = repo / "workloads" / "portable_lcd" / slug / f"{slug}_run"
    source = binary.with_name(f"{slug}.cu")
    tracer = repo / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
    tracer_bytes = b"retained activation tracer"
    _write_executable(binary, b"retained activation executable")
    source.write_bytes(b"__global__ void fixture_kernel() {}\n")
    tracer.parent.mkdir(parents=True, exist_ok=True)
    tracer.write_bytes(tracer_bytes)

    cell = repo / "results" / "nvbit_capture_staging" / "activation" / "cell"
    plan = {
        "output_root": cell.parent,
        "cell_dir": cell,
        "traces_folder": cell / "trace_timing_shape",
        "sidecar_path": cell / "trace_timing_shape.json",
        "manifest_path": cell / "measurement_manifest_timing_shape.json",
        "workload_output_root": cell / "workload_output",
        "portable_manifest": cell / "workload_output" / slug / "manifest.json",
    }
    monkeypatch.setattr(activation, "_REPO_ROOT", repo)
    monkeypatch.setattr(activation, "_TRACER_SO", tracer)
    monkeypatch.setattr(
        activation,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(activation, "_git_head", lambda: "f" * 40)
    monkeypatch.setattr(
        activation,
        "_resolve_clean_source_commit",
        lambda _source, _retained: "f" * 40,
    )
    monkeypatch.setattr(activation, "_tracer_env", lambda **_kwargs: {})
    monkeypatch.setattr(
        activation,
        "require_trace_admission",
        lambda _traces_folder: admission,
    )
    monkeypatch.setattr(
        activation,
        "open_admitted_trace_sources",
        lambda _traces_folder, *, expected_admission: nullcontext(
            type(
                "FixtureTraceSources",
                (),
                {"verify_unchanged": lambda self: None},
            )()
        ),
    )

    def fake_run(cmd, **kwargs):
        trace_root = Path(kwargs["env"]["TRACES_FOLDER"])
        (trace_root / "dynamic_trace.pb").write_bytes(b"fixture trace")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(activation.subprocess, "run", fake_run)
    return plan, tracer_bytes


def _variant_capture_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admission: TraceAdmission,
) -> tuple[dict[str, Path], bytes]:
    repo = tmp_path / "variant_repo"
    binary = repo / "bin" / "register_spill_local_mem_run__cap64"
    tracer = repo / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
    shapes = repo / "workloads" / "simbench_bw" / "shapes" / "register_spill_local_mem.toml"
    tracer_bytes = b"retained variant tracer"
    _write_executable(binary, b"retained variant executable")
    tracer.parent.mkdir(parents=True, exist_ok=True)
    tracer.write_bytes(tracer_bytes)
    shapes.parent.mkdir(parents=True, exist_ok=True)
    shapes.write_text("trace_shapes = []\n", encoding="utf-8")
    for relative_path in variants._SOURCE_COMMIT_AUTHORITY_PATHS:
        authority_path = repo / relative_path
        if authority_path == shapes:
            continue
        authority_path.parent.mkdir(parents=True, exist_ok=True)
        authority_path.write_bytes(
            f"fixture authority: {relative_path}\n".encode("utf-8")
        )

    variant_dir = repo / "results" / "nvbit_capture_staging" / "variants" / "cap64"
    sidecar = variant_dir / "trace_trace_l2_accuracy_xl.json"
    plan = {
        "output_root": variant_dir.parent,
        "variant_dir": variant_dir,
        "traces_folder": variant_dir / "trace_trace_l2_accuracy_xl",
        "summary_generation_root": variant_dir / "summary_generations",
        "attempt_record_root": variant_dir / "capture_attempts",
        "workload_output_root": variant_dir / "workload_output",
        "sidecar_path": sidecar,
        "summary_path": variant_dir / "nvbit_trace_summary_0_0.json",
        "tmp_path": sidecar.with_name(sidecar.name + ".tmp"),
    }
    monkeypatch.setattr(variants, "_REPO_ROOT", repo)
    monkeypatch.setattr(variants, "_TRACER_SO", tracer)
    monkeypatch.setattr(variants, "_SHAPES_TOML", shapes)
    monkeypatch.setattr(
        variants,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(variants, "_variant_binary_path", lambda _slug, _cap: binary)
    monkeypatch.setattr(variants, "_git_head", lambda: "e" * 40)
    monkeypatch.setattr(
        variants,
        "_resolve_source_commit",
        lambda *_args, **_kwargs: ("e" * 40, False),
    )
    monkeypatch.setattr(
        variants,
        "_kernel_args_for_shape",
        lambda _shape_point, _seed, **_kwargs: ["--fixture"],
    )
    monkeypatch.setattr(variants, "tracer_runtime_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        variants,
        "require_trace_admission",
        lambda _traces_folder: admission,
    )
    monkeypatch.setattr(
        variants,
        "open_admitted_trace_sources",
        lambda _traces_folder, *, expected_admission: nullcontext(
            type(
                "FixtureTraceSources",
                (),
                {"verify_unchanged": lambda self: None},
            )()
        ),
    )

    def fake_variant_run(_cmd, env, _cwd, **_kwargs):
        trace_root = Path(env["TRACES_FOLDER"])
        (trace_root / "dynamic_trace.pb").write_bytes(b"fixture trace")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(
        variants,
        "_run_workload",
        fake_variant_run,
    )
    return plan, tracer_bytes


def _arm_zero_mutation(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Instrument every mutation/launch family a refusal must precede."""

    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"mutation occurred before clearance: {name}")
        return fail

    monkeypatch.setattr(Path, "mkdir", forbidden("mkdir"))
    monkeypatch.setattr(Path, "write_text", forbidden("write_text"))
    monkeypatch.setattr(Path, "unlink", forbidden("unlink"))
    monkeypatch.setattr(shutil, "rmtree", forbidden("rmtree"))
    monkeypatch.setattr(os, "replace", forbidden("replace"))
    monkeypatch.setattr(tempfile, "mkdtemp", forbidden("mkdtemp"))
    monkeypatch.setattr(subprocess, "run", forbidden("subprocess"))
    return calls


def test_activation_orchestrator_safe_output_plan() -> None:
    plan = activation._activation_output_plan(
        slug="layernorm_fwd_bf16_sm80",
        shape_point="timing_shape",
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        fallback_source="portable_lcd",
        output_root=STAGING / "test_activation",
    )
    assert all(path.is_relative_to(STAGING) for path in plan.values())
    assert plan["portable_manifest"].name == "manifest.json"


def test_activation_orchestrator_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        activation._activation_output_plan(
            slug="layernorm_fwd_bf16_sm80",
            shape_point="timing_shape",
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            fallback_source="portable_lcd",
            output_root=PROTECTED / "activation",
        )
    assert calls == []


def test_activation_source_commit_binds_clean_authority_and_retained_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "workloads" / "portable_lcd" / "fixture" / "fixture.cu"
    payload = b"__global__ void fixture() {}\n"
    head = "a" * 40
    retained = _RetainedBytesFixture(payload)
    commands: list[list[str]] = []
    monkeypatch.setattr(activation, "_REPO_ROOT", repo)
    monkeypatch.setattr(activation, "_git_head", lambda: head)

    def git_run(command: list[str], **_kwargs):
        commands.append(command)
        if command[3] == "status":
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[3] == "show":
            return subprocess.CompletedProcess(command, 0, payload, b"")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(activation.subprocess, "run", git_run)

    assert activation._resolve_clean_source_commit(source, retained) == head
    assert retained.verify_count == 1
    source_relative = source.relative_to(repo).as_posix()
    assert commands == [
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--",
            *activation._SOURCE_COMMIT_AUTHORITY_PATHS,
            source_relative,
        ],
        [
            "git",
            "-C",
            str(repo),
            "show",
            f"{head}:{source_relative}",
        ],
    ]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("dirty", "authority is dirty or untracked"),
        ("source_mismatch", "retained activation source bytes disagree"),
        ("head_drift", "source_commit changed during authority verification"),
    ],
)
def test_activation_source_commit_rejects_contaminated_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "workloads" / "portable_lcd" / "fixture" / "fixture.cu"
    payload = b"__global__ void fixture() {}\n"
    head = "a" * 40
    retained = _RetainedBytesFixture(payload)
    heads = iter((head, "b" * 40)) if failure == "head_drift" else None
    monkeypatch.setattr(activation, "_REPO_ROOT", repo)
    monkeypatch.setattr(
        activation,
        "_git_head",
        (lambda: next(heads)) if heads is not None else (lambda: head),
    )

    def git_run(command: list[str], **_kwargs):
        if command[3] == "status":
            status = " M tools/nvbit_tracer/runtime_env.py\n" if failure == "dirty" else ""
            return subprocess.CompletedProcess(command, 0, status, "")
        if command[3] == "show":
            committed = b"different source\n" if failure == "source_mismatch" else payload
            return subprocess.CompletedProcess(command, 0, committed, b"")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(activation.subprocess, "run", git_run)

    with pytest.raises(RuntimeError, match=message):
        activation._resolve_clean_source_commit(source, retained)
    assert retained.verify_count == 0


@pytest.mark.parametrize("invalid_identity", ("attempt", "source_commit"))
def test_activation_rejects_boolean_publication_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_identity: str,
) -> None:
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, _admitted_trace()
    )
    if invalid_identity == "attempt":
        monkeypatch.setattr(activation, "_new_capture_attempt_id", lambda: True)
    else:
        monkeypatch.setattr(
            activation,
            "_resolve_clean_source_commit",
            lambda _source, _retained: True,
        )
    with pytest.raises((RuntimeError, ValueError), match="(exact string|40-hex)"):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )
    assert not plan["cell_dir"].exists()


def test_activation_orchestrator_propagates_retained_trace_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, tracer_bytes = _activation_capture_fixture(
        tmp_path,
        monkeypatch,
        admission,
    )

    sidecar = activation.trace_activation_cell(
        slug="fixture_activation",
        shape_point="timing_shape",
        _precleared_plan=plan,
    )

    assert sidecar["kernel_name"] == "fixture_kernel"
    assert sidecar["nvbit_tracer_build_hash"] == hashlib.sha256(
        tracer_bytes
    ).hexdigest()
    _assert_sidecar_admission_identities(sidecar, admission)
    roster_record = roster_capture._activation_capture_record(
        slug="fixture_activation",
        cell_dir=plan["cell_dir"],
        sidecar=sidecar,
    )
    assert roster_record["binary_sha256"] == sidecar["binary_sha256"]
    assert roster_record["nvbit_tracer_build_hash"] == sidecar[
        "nvbit_tracer_build_hash"
    ]
    _assert_sidecar_admission_identities(roster_record, admission)
    assert sidecar["allocations_sidecar_size_bytes"] is None
    assert sidecar["allocations_sidecar_sha256"] is None
    assert json.loads(plan["sidecar_path"].read_text(encoding="utf-8")) == sidecar


def test_activation_orchestrator_rejects_multi_launch_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace(launch_count=2)
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path,
        monkeypatch,
        admission,
    )

    with pytest.raises(RuntimeError, match=r"launch_count == 1; got 2"):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    assert not plan["sidecar_path"].exists()
    assert not plan["manifest_path"].exists()


def test_activation_capture_passes_all_output_capabilities_by_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    observed: dict[str, object] = {}

    def capture_descriptors(command, **kwargs):
        observed["command"] = command
        observed["env"] = dict(kwargs["env"])
        observed["pass_fds"] = tuple(kwargs["pass_fds"])
        trace_root = Path(kwargs["env"]["TRACES_FOLDER"])
        (trace_root / "dynamic_trace.pb").write_bytes(b"fixture trace")
        for descriptor in kwargs["pass_fds"]:
            os.fstat(descriptor)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(activation.subprocess, "run", capture_descriptors)
    activation.trace_activation_cell(
        slug="fixture_activation",
        shape_point="timing_shape",
        _precleared_plan=plan,
    )

    command = observed["command"]
    env = observed["env"]
    pass_fds = observed["pass_fds"]
    assert isinstance(command, list)
    assert isinstance(env, dict)
    assert isinstance(pass_fds, tuple)
    assert command[0].startswith("/proc/self/fd/")
    output_index = command.index("--output_dir") + 1
    assert command[output_index].startswith("/proc/self/fd/")
    assert str(env["CUDA_INJECTION64_PATH"]).startswith("/proc/self/fd/")
    assert str(env["TRACES_FOLDER"]).startswith("/proc/self/fd/")
    inherited = {
        int(command[0].rsplit("/", 1)[1]),
        int(command[output_index].rsplit("/", 1)[1]),
        int(str(env["CUDA_INJECTION64_PATH"]).rsplit("/", 1)[1]),
        int(str(env["TRACES_FOLDER"]).rsplit("/", 1)[1]),
    }
    assert inherited == set(pass_fds)


def test_activation_failed_rerun_preserves_prior_attempt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    attempts = iter(("attempt-one", "attempt-two"))
    monkeypatch.setattr(
        activation, "_new_capture_attempt_id", lambda: next(attempts)
    )
    first = activation.trace_activation_cell(
        slug="fixture_activation",
        shape_point="timing_shape",
        _precleared_plan=plan,
    )
    first_sidecar = plan["sidecar_path"].read_bytes()
    first_manifest = plan["manifest_path"].read_bytes()
    first_raw = plan["cell_dir"] / first["raw_trace_generation"]["path"]
    first_trace = (first_raw / "dynamic_trace.pb").read_bytes()

    def reject_admission(_path: Path):
        raise activation.TraceAdmissionError(
            "trace_snapshot_changed", "planted second-attempt failure"
        )

    monkeypatch.setattr(activation, "require_trace_admission", reject_admission)
    with pytest.raises(RuntimeError, match="not admissible"):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    assert not plan["sidecar_path"].exists()
    assert not plan["manifest_path"].exists()
    retired_sidecars = tuple(
        plan["cell_dir"].glob(".trace_timing_shape.json.*.retired")
    )
    retired_manifests = tuple(
        plan["cell_dir"].glob(
            ".measurement_manifest_timing_shape.json.*.retired"
        )
    )
    assert len(retired_sidecars) == 1
    assert retired_sidecars[0].read_bytes() == first_sidecar
    assert len(retired_manifests) == 1
    assert retired_manifests[0].read_bytes() == first_manifest
    assert (first_raw / "dynamic_trace.pb").read_bytes() == first_trace
    generations = {
        path.name
        for path in plan["traces_folder"].iterdir()
        if not path.name.startswith(".")
    }
    assert generations == {"attempt-one", "attempt-two"}


def test_activation_two_successful_attempts_coexist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    attempts = iter(("attempt-one", "attempt-two"))
    monkeypatch.setattr(
        activation, "_new_capture_attempt_id", lambda: next(attempts)
    )
    first = activation.trace_activation_cell(
        slug="fixture_activation",
        shape_point="timing_shape",
        _precleared_plan=plan,
    )
    first_raw = plan["cell_dir"] / first["raw_trace_generation"]["path"]
    first_bytes = (first_raw / "dynamic_trace.pb").read_bytes()
    second = activation.trace_activation_cell(
        slug="fixture_activation",
        shape_point="timing_shape",
        _precleared_plan=plan,
    )
    second_raw = plan["cell_dir"] / second["raw_trace_generation"]["path"]

    assert first_raw != second_raw
    assert (first_raw / "dynamic_trace.pb").read_bytes() == first_bytes
    assert (second_raw / "dynamic_trace.pb").read_bytes() == b"fixture trace"
    assert json.loads(plan["sidecar_path"].read_text(encoding="utf-8"))[
        "trace_capture_attempt_id"
    ] == "attempt-two"


def test_activation_head_drift_during_final_source_close_retires_locators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    source_commit = "f" * 40
    current_head = {"value": source_commit}
    monkeypatch.setattr(
        activation, "_git_head", lambda: current_head["value"]
    )
    original_open = activation.open_retained_executable

    class HeadDriftingExecutable:
        def __init__(self, retained) -> None:
            self._retained = retained

        def __getattr__(self, name):
            return getattr(self._retained, name)

        def __enter__(self):
            self._retained.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            result = self._retained.__exit__(exc_type, exc, traceback)
            current_head["value"] = "a" * 40
            return result

    monkeypatch.setattr(
        activation,
        "open_retained_executable",
        lambda path: HeadDriftingExecutable(original_open(path)),
    )
    with pytest.raises(
        RuntimeError,
        match="source_commit changed during capture publication",
    ):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    assert not plan["sidecar_path"].exists()
    assert not plan["manifest_path"].exists()
    assert len(tuple(plan["cell_dir"].glob(
        ".trace_timing_shape.json.*.retired"
    ))) == 1
    assert len(tuple(plan["cell_dir"].glob(
        ".measurement_manifest_timing_shape.json.*.retired"
    ))) == 1


def test_activation_source_change_retires_new_locators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    binary = (
        activation._REPO_ROOT
        / "workloads"
        / "portable_lcd"
        / "fixture_activation"
        / "fixture_activation_run"
    )
    original_publish = activation.publish_retained_regular_file

    def mutate_before_sidecar(parent, name, payload, **kwargs):
        if name == plan["sidecar_path"].name:
            replacement = binary.with_name(f"{binary.name}.replacement")
            replacement.write_bytes(binary.read_bytes())
            replacement.chmod(0o755)
            os.replace(replacement, binary)
        return original_publish(parent, name, payload, **kwargs)

    monkeypatch.setattr(
        activation, "publish_retained_regular_file", mutate_before_sidecar
    )
    with pytest.raises(
        ImmutableArtifactError,
        match="named regular-file path changed",
    ):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    assert not plan["sidecar_path"].exists()
    assert not plan["manifest_path"].exists()
    assert len(tuple(plan["cell_dir"].glob(".trace_timing_shape.json.*.retired"))) == 1
    assert len(tuple(plan["cell_dir"].glob(
        ".measurement_manifest_timing_shape.json.*.retired"
    ))) == 1


def test_activation_retained_roster_source_change_retires_locators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    binary = (
        activation._REPO_ROOT
        / "workloads"
        / "portable_lcd"
        / "fixture_activation"
        / "fixture_activation_run"
    )
    retained = ExitStack()
    activation.trace_activation_cell(
        slug="fixture_activation",
        shape_point="timing_shape",
        _precleared_plan=plan,
        _retention_stack=retained,
    )
    assert plan["sidecar_path"].is_file()
    assert plan["manifest_path"].is_file()

    replacement = binary.with_name(f"{binary.name}.replacement")
    replacement.write_bytes(binary.read_bytes())
    replacement.chmod(0o755)
    os.replace(replacement, binary)
    with pytest.raises(
        ImmutableArtifactError,
        match="named regular-file path changed",
    ):
        retained.close()

    assert not plan["sidecar_path"].exists()
    assert not plan["manifest_path"].exists()
    raw = plan["cell_dir"] / "trace_timing_shape"
    assert any(path.is_dir() for path in raw.iterdir())


@pytest.mark.parametrize("target_key", ("sidecar_path", "manifest_path"))
@pytest.mark.parametrize("mutation", ("replacement", "same_inode"))
def test_activation_final_locator_authentication_preserves_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_key: str,
    mutation: str,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    original_open = activation.open_retained_executable
    winner_payload = f"concurrent {target_key} winner\n".encode("utf-8")
    mutated_payload = f"mutated {target_key} generation\n".encode("utf-8")
    winner_identity: tuple[int, int] | None = None
    mutated = False

    class LocatorMutatingExecutable:
        def __init__(self, retained) -> None:
            self._retained = retained

        def __getattr__(self, name):
            return getattr(self._retained, name)

        def __enter__(self):
            self._retained.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal mutated, winner_identity
            result = self._retained.__exit__(exc_type, exc, traceback)
            if not mutated:
                mutated = True
                target = plan[target_key]
                if mutation == "replacement":
                    replacement = target.with_name(f".concurrent-{target.name}")
                    replacement.write_bytes(winner_payload)
                    replacement.chmod(0o444)
                    os.replace(replacement, target)
                    winner_identity = (
                        target.stat().st_dev,
                        target.stat().st_ino,
                    )
                else:
                    target.chmod(0o644)
                    target.write_bytes(mutated_payload)
            return result

    monkeypatch.setattr(
        activation,
        "open_retained_executable",
        lambda path: LocatorMutatingExecutable(original_open(path)),
    )
    with pytest.raises(ImmutableArtifactError):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    assert mutated is True
    target = plan[target_key]
    other_key = (
        "manifest_path" if target_key == "sidecar_path" else "sidecar_path"
    )
    other = plan[other_key]
    if mutation == "replacement":
        assert winner_identity is not None
        assert (target.stat().st_dev, target.stat().st_ino) == winner_identity
        assert target.read_bytes() == winner_payload
        salvaged = tuple(
            path for path in plan["cell_dir"].glob(".failed-regular-*")
            if not path.name.endswith(".salvaging")
        )
        assert len(salvaged) == 1
        assert json.loads(salvaged[0].read_text(encoding="utf-8"))[
            "slug"
        ] == "fixture_activation"
    else:
        assert not target.exists()
        retired_targets = tuple(
            plan["cell_dir"].glob(f".{target.name}.*.retired")
        )
        assert len(retired_targets) == 1
        assert retired_targets[0].read_bytes() == mutated_payload
    assert not other.exists()
    assert len(tuple(
        plan["cell_dir"].glob(f".{other.name}.*.retired")
    )) == 1


@pytest.mark.parametrize("target_key", ("sidecar_path", "manifest_path"))
@pytest.mark.parametrize("mutation", ("replacement", "same_inode"))
def test_activation_authenticated_close_rejects_monitor_close_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_key: str,
    mutation: str,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    original_publish = activation.publish_retained_regular_file
    target = plan[target_key]
    winner_payload = f"close-race {target_key} winner\n".encode()
    mutated_payload = f"close-race {target_key} mutation\n".encode()
    winner_identity: tuple[int, int] | None = None

    def instrument_close(parent, name, payload, **kwargs):
        retained = original_publish(parent, name, payload, **kwargs)
        if name != target.name:
            return retained
        state = retained._monitor_state
        assert state is not None
        original_close = state.monitor.close

        def mutate_during_close() -> None:
            nonlocal winner_identity
            if mutation == "replacement":
                replacement = target.with_name(f".close-race-{target.name}")
                replacement.write_bytes(winner_payload)
                replacement.chmod(0o444)
                os.replace(replacement, target)
                winner_identity = (
                    target.stat().st_dev,
                    target.stat().st_ino,
                )
            else:
                target.chmod(0o644)
                target.write_bytes(mutated_payload)
            original_close()

        monkeypatch.setattr(state.monitor, "close", mutate_during_close)
        return retained

    monkeypatch.setattr(
        activation, "publish_retained_regular_file", instrument_close,
    )
    with pytest.raises(ImmutableArtifactError):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    other_key = (
        "manifest_path" if target_key == "sidecar_path" else "sidecar_path"
    )
    assert not plan[other_key].exists()
    if mutation == "replacement":
        assert winner_identity is not None
        assert (target.stat().st_dev, target.stat().st_ino) == winner_identity
        assert target.read_bytes() == winner_payload
        salvaged = tuple(
            path for path in plan["cell_dir"].glob(".failed-regular-*")
            if not path.name.endswith(".salvaging")
        )
        assert len(salvaged) == 1
    else:
        assert not target.exists()
        retired = tuple(plan["cell_dir"].glob(f".{target.name}.*.retired"))
        assert len(retired) == 1
        assert retired[0].read_bytes() == mutated_payload


@pytest.mark.parametrize("target_key", ("sidecar_path", "manifest_path"))
def test_activation_authenticated_close_rejects_final_pinned_close_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_key: str,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    target = plan[target_key]
    winner_payload = f"final pinned {target_key} winner\n".encode()
    original_publish = activation.publish_retained_regular_file
    winner_identity: tuple[int, int] | None = None

    def instrument_pinned_close(parent, name, payload, **kwargs):
        retained = original_publish(parent, name, payload, **kwargs)
        if name != target.name:
            return retained
        original_close = retained._pinned.close

        def replace_during_pinned_close() -> None:
            nonlocal winner_identity
            replacement = target.with_name(f".final-pinned-{target.name}")
            replacement.write_bytes(winner_payload)
            replacement.chmod(0o444)
            os.replace(replacement, target)
            winner_identity = (
                target.stat().st_dev,
                target.stat().st_ino,
            )
            original_close()

        monkeypatch.setattr(
            retained._pinned, "close", replace_during_pinned_close,
        )
        return retained

    monkeypatch.setattr(
        activation,
        "publish_retained_regular_file",
        instrument_pinned_close,
    )
    with pytest.raises(ImmutableArtifactError):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    assert winner_identity is not None
    assert (target.stat().st_dev, target.stat().st_ino) == winner_identity
    assert target.read_bytes() == winner_payload
    other_key = (
        "manifest_path" if target_key == "sidecar_path" else "sidecar_path"
    )
    assert not plan[other_key].exists()
    salvaged = tuple(
        path for path in plan["cell_dir"].glob(".failed-regular-*")
        if not path.name.endswith(".salvaging")
    )
    assert len(salvaged) == 1
    assert json.loads(salvaged[0].read_text(encoding="utf-8"))["slug"] == (
        "fixture_activation"
    )


@pytest.mark.parametrize("target_key", ("sidecar_path", "manifest_path"))
def test_activation_authenticated_close_rejects_guard_release_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_key: str,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    target = plan[target_key]
    winner_payload = f"guard release {target_key} winner\n".encode()
    original_close_descriptors = immutable._close_descriptors
    original_preserve = activation.preserve_retained_regular_file
    winner_identity: tuple[int, int] | None = None
    guard_release_calls = 0
    replaced = False
    transferred_descriptors: list[int] = []

    def replace_during_guard_release(
        descriptors,
        *,
        active_error=None,
        label="descriptor close",
    ) -> None:
        nonlocal guard_release_calls, replaced, winner_identity
        materialized = tuple(descriptors)
        if label == "authenticated retained close guard release":
            guard_release_calls += 1
            if not replaced:
                replaced = True
                replacement = target.with_name(f".guard-release-{target.name}")
                replacement.write_bytes(winner_payload)
                replacement.chmod(0o444)
                os.replace(replacement, target)
                winner_identity = (
                    target.stat().st_dev,
                    target.stat().st_ino,
                )
        original_close_descriptors(
            materialized,
            active_error=active_error,
            label=label,
        )

    def observe_transferred_capability(retained, *, expected_payload):
        transferred_descriptors.extend(
            (
                retained._pinned.descriptor,
                *retained._pinned.directory_descriptors,
            )
        )
        return original_preserve(retained, expected_payload=expected_payload)

    monkeypatch.setattr(
        immutable, "_close_descriptors", replace_during_guard_release,
    )
    monkeypatch.setattr(
        activation,
        "preserve_retained_regular_file",
        observe_transferred_capability,
    )
    with pytest.raises(ImmutableArtifactError):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    assert replaced is True
    assert guard_release_calls == 2
    assert winner_identity is not None
    assert (target.stat().st_dev, target.stat().st_ino) == winner_identity
    assert target.read_bytes() == winner_payload
    other_key = (
        "manifest_path" if target_key == "sidecar_path" else "sidecar_path"
    )
    other = plan[other_key]
    assert not other.exists()
    assert len(tuple(
        plan["cell_dir"].glob(f".{other.name}.*.retired")
    )) == 1
    salvaged = tuple(
        path for path in plan["cell_dir"].glob(".failed-regular-*")
        if not path.name.endswith(".salvaging")
    )
    assert len(salvaged) == 1
    assert json.loads(salvaged[0].read_text(encoding="utf-8"))["slug"] == (
        "fixture_activation"
    )
    assert transferred_descriptors
    for descriptor in transferred_descriptors:
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("target_key", ("sidecar_path", "manifest_path"))
def test_activation_authenticated_close_covers_lock_reader_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_key: str,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    target = plan[target_key]
    winner_payload = f"activation lock close {target_key} winner\n".encode()
    original_lock = activation._lock_activation_cell
    original_reader_close = immutable.PinnedDirectoryReader.close
    original_preserve = activation.preserve_retained_regular_file
    locked_reader = None
    winner_identity: tuple[int, int] | None = None
    lock_reader_close_calls = 0
    replaced = False
    transferred_descriptors: list[int] = []

    def retain_locked_reader(reader) -> None:
        nonlocal locked_reader
        locked_reader = reader
        original_lock(reader)

    def replace_during_lock_reader_close(self) -> None:
        nonlocal lock_reader_close_calls, replaced, winner_identity
        if self is locked_reader:
            lock_reader_close_calls += 1
            if not replaced:
                replaced = True
                replacement = target.with_name(f".lock-close-{target.name}")
                replacement.write_bytes(winner_payload)
                replacement.chmod(0o444)
                os.replace(replacement, target)
                winner_identity = (
                    target.stat().st_dev,
                    target.stat().st_ino,
                )
        original_reader_close(self)

    def observe_transferred_capability(retained, *, expected_payload):
        transferred_descriptors.extend(
            (
                retained._pinned.descriptor,
                *retained._pinned.directory_descriptors,
            )
        )
        return original_preserve(retained, expected_payload=expected_payload)

    monkeypatch.setattr(
        activation, "_lock_activation_cell", retain_locked_reader,
    )
    monkeypatch.setattr(
        immutable.PinnedDirectoryReader,
        "close",
        replace_during_lock_reader_close,
    )
    monkeypatch.setattr(
        activation,
        "preserve_retained_regular_file",
        observe_transferred_capability,
    )
    with pytest.raises(ImmutableArtifactError):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )

    assert replaced is True
    assert lock_reader_close_calls == 1
    assert winner_identity is not None
    assert (target.stat().st_dev, target.stat().st_ino) == winner_identity
    assert target.read_bytes() == winner_payload
    other_key = (
        "manifest_path" if target_key == "sidecar_path" else "sidecar_path"
    )
    other = plan[other_key]
    assert not other.exists()
    assert len(tuple(
        plan["cell_dir"].glob(f".{other.name}.*.retired")
    )) == 1
    salvaged = tuple(
        path for path in plan["cell_dir"].glob(".failed-regular-*")
        if not path.name.endswith(".salvaging")
    )
    assert len(salvaged) == 1
    assert json.loads(salvaged[0].read_text(encoding="utf-8"))["slug"] == (
        "fixture_activation"
    )
    assert transferred_descriptors
    for descriptor in transferred_descriptors:
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize(
    "arguments",
    (
        tuple(f"arg-{index}" for index in range(65)),
        tuple("x" * 4096 for _index in range(17)),
    ),
)
def test_activation_argument_bounds_precede_subprocess_and_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        activation.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )
    output_root = tmp_path / "activation"
    with pytest.raises(
        ImmutableArtifactResourceError, match=VERIFY_RESOURCE_BOUND,
    ):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            output_root=output_root,
            extra_app_args=arguments,
        )
    assert not output_root.exists()


@pytest.mark.parametrize("boundary", ("head", "status", "show"))
@pytest.mark.parametrize("resource_errno", (errno.EMFILE, errno.ENFILE))
def test_activation_git_pipe_exhaustion_keeps_resource_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    resource_errno: int,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "workloads" / "fixture.cu"
    source.parent.mkdir(parents=True)
    source_payload = b"__global__ void fixture() {}\n"
    source.write_bytes(source_payload)
    retained_source = _RetainedBytesFixture(source_payload)
    head = "f" * 40
    exhausted_calls = 0

    def run_git(command, **_kwargs):
        nonlocal exhausted_calls
        if boundary == "head":
            assert command == ["git", "rev-parse", "HEAD"]
        elif boundary == "status":
            assert "status" in command
        elif "status" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        else:
            assert "show" in command
        exhausted_calls += 1
        raise OSError(
            resource_errno,
            f"planted Git {boundary} pipe exhaustion",
        )

    monkeypatch.setattr(activation, "_REPO_ROOT", repo)
    monkeypatch.setattr(activation.subprocess, "run", run_git)
    if boundary != "head":
        monkeypatch.setattr(activation, "_git_head", lambda: head)

    with pytest.raises(
        ImmutableArtifactResourceError, match=VERIFY_RESOURCE_BOUND,
    ) as caught:
        if boundary == "head":
            activation._git_head()
        else:
            activation._resolve_clean_source_commit(source, retained_source)

    assert exhausted_calls == 1
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == resource_errno


def test_roster_maps_activation_git_pipe_exhaustion_to_resource_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    manifest_path = output_root / "capture_manifest.json"
    cell_dir = output_root / "portable" / "fixture_activation"
    roster = [{
        "slug": "fixture_activation",
        "shape_point": "timing_shape",
        "binary": "bin/fixture_activation",
        "app_args": [],
    }]
    plans: list[dict[str, Path | bool]] = [{
        "portable": True,
        "cell_dir": cell_dir,
        "traces_folder": cell_dir / "trace_timing_shape",
        "workload_output_root": cell_dir / "workload_output",
    }]

    def exhaust_git_head(**_kwargs):
        return activation._git_head()

    def exhaust_pipe(*_args, **_kwargs):
        raise OSError(errno.EMFILE, "planted roster Git pipe exhaustion")

    monkeypatch.setattr(
        roster_capture, "_new_capture_attempt_id", lambda: "git-resource-attempt",
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    monkeypatch.setattr(
        roster_capture, "trace_activation_cell", exhaust_git_head,
    )
    monkeypatch.setattr(activation.subprocess, "run", exhaust_pipe)

    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 1

    attempt = json.loads((
        output_root
        / "capture_manifest_attempts"
        / "git-resource-attempt.json"
    ).read_text(encoding="utf-8"))
    assert attempt["cells"][0]["typed_reason"] == (
        "trace_snapshot_resource_unavailable"
    )
    assert VERIFY_RESOURCE_BOUND in attempt["cells"][0]["detail"]


@pytest.mark.parametrize(
    ("boundary", "resource_errno"),
    (("lock_dup", errno.EMFILE), ("subprocess_pipe", errno.ENFILE)),
)
def test_activation_late_descriptor_exhaustion_keeps_resource_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    resource_errno: int,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    if boundary == "lock_dup":
        monkeypatch.setattr(
            activation.subprocess,
            "run",
            lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
        )

        def exhaust_lock_dup(_descriptor: int) -> int:
            raise OSError(resource_errno, "planted activation dup exhaustion")

        monkeypatch.setattr(activation.os, "dup", exhaust_lock_dup)
    else:
        def exhaust_subprocess(*_args, **_kwargs):
            raise OSError(resource_errno, "planted activation pipe exhaustion")

        monkeypatch.setattr(activation.subprocess, "run", exhaust_subprocess)

    with pytest.raises(
        ImmutableArtifactResourceError, match=VERIFY_RESOURCE_BOUND,
    ):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
        )
    assert not plan["sidecar_path"].exists()
    assert not plan["manifest_path"].exists()


class _FailingRegistrationStack(ExitStack):
    def enter_context(self, context_manager):
        context_manager.__enter__()
        raise RuntimeError("planted retention registration failure")


def test_activation_registration_failure_closes_transferred_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _activation_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    original_open = activation.open_retained_executable
    executable_closed = False

    class TrackingExecutable:
        def __init__(self, retained) -> None:
            self._retained = retained

        def __getattr__(self, name):
            return getattr(self._retained, name)

        def __enter__(self):
            self._retained.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal executable_closed
            try:
                return self._retained.__exit__(exc_type, exc, traceback)
            finally:
                executable_closed = True

    monkeypatch.setattr(
        activation,
        "open_retained_executable",
        lambda path: TrackingExecutable(original_open(path)),
    )
    retention = _FailingRegistrationStack()
    with pytest.raises(RuntimeError, match="registration failure"):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            _precleared_plan=plan,
            _retention_stack=retention,
        )

    assert executable_closed is True
    assert not plan["sidecar_path"].exists()
    assert not plan["manifest_path"].exists()
    assert len(tuple(plan["cell_dir"].glob(".*.retired"))) >= 2
    with immutable.PinnedDirectoryReader(plan["traces_folder"]) as lock_reader:
        activation._lock_activation_cell(lock_reader)
    retention.close()


def test_activation_rejects_invalid_retention_stack_before_mutation(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "activation"
    with pytest.raises(TypeError, match="ExitStack"):
        activation.trace_activation_cell(
            slug="fixture_activation",
            shape_point="timing_shape",
            output_root=output_root,
            _retention_stack=object(),  # type: ignore[arg-type]
        )
    assert not output_root.exists()


def test_roster_capture_safe_batch_plan() -> None:
    plans, manifest = roster_capture._roster_capture_plan(
        roster=[{
            "slug": "layernorm_fwd_bf16_sm80",
            "shape_point": "timing_shape",
            "app_args": ["--timing_shape"],
        }],
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=STAGING / "test_roster",
        manifest_out=None,
    )
    assert manifest.is_relative_to(STAGING)
    assert plans[0]["portable"] is True
    assert Path(plans[0]["traces_folder"]).is_relative_to(STAGING)


def test_roster_capture_rejects_duplicate_output_identity() -> None:
    duplicate = {
        "slug": "layernorm_fwd_bf16_sm80",
        "shape_point": "timing_shape",
        "app_args": ["--timing_shape"],
    }
    with pytest.raises(
        ValueError,
        match="duplicate roster trace output identity",
    ):
        roster_capture._roster_capture_plan(
            roster=[duplicate, dict(duplicate)],
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            output_root=STAGING / "test_roster_duplicate",
            manifest_out=None,
        )


def test_roster_empty_refuses_before_path_clearance_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ValueError, match="at least one cell"):
        roster_capture._roster_capture_plan(
            roster=[], sku="rtx_3090_ti", arch_dir="sm_86",
            output_root=PROTECTED / "roster", manifest_out=None,
        )
    assert calls == []


def test_roster_direct_capture_propagates_every_trace_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace(launch_count=2, with_allocations=True)
    binary = tmp_path / "direct" / "fixture_run"
    tracer = tmp_path / "tracer_tool.so"
    binary_bytes = b"retained direct executable"
    tracer_bytes = b"retained direct tracer"
    _write_executable(binary, binary_bytes)
    tracer.write_bytes(tracer_bytes)
    monkeypatch.setattr(roster_capture, "_TRACER_SO", tracer)
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_tracer_env", lambda **_kwargs: {})
    monkeypatch.setattr(
        roster_capture,
        "require_trace_admission",
        lambda _trace_dir: admission,
    )
    monkeypatch.setattr(
        roster_capture,
        "open_admitted_trace_sources",
        lambda _traces_folder, *, expected_admission: nullcontext(
            type(
                "FixtureTraceSources",
                (),
                {"verify_unchanged": lambda self: None},
            )()
        ),
    )

    observed: dict[str, object] = {}

    def fake_direct_run(cmd, **kwargs):
        observed["command"] = list(cmd)
        observed["env"] = dict(kwargs["env"])
        observed["pass_fds"] = tuple(kwargs["pass_fds"])
        trace_root = Path(kwargs["env"]["TRACES_FOLDER"])
        (trace_root / "dynamic_trace.pb").write_bytes(b"fixture trace")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        roster_capture.subprocess,
        "run",
        fake_direct_run,
    )

    record = roster_capture._capture_direct(
        "fixture",
        binary,
        [],
        tmp_path / "trace",
        tmp_path / "workload-output",
    )

    assert record["ok"] is True
    command = observed["command"]
    env = observed["env"]
    pass_fds = observed["pass_fds"]
    assert isinstance(command, list)
    assert isinstance(env, dict)
    assert isinstance(pass_fds, tuple)
    output_index = command.index("--output_dir") + 1
    proc_paths = {
        command[0],
        command[output_index],
        str(env["CUDA_INJECTION64_PATH"]),
        str(env["TRACES_FOLDER"]),
    }
    assert all(path.startswith("/proc/self/fd/") for path in proc_paths)
    assert {
        int(path.rsplit("/", 1)[1]) for path in proc_paths
    } == set(pass_fds)
    assert record["binary_sha256"] == hashlib.sha256(binary_bytes).hexdigest()
    assert record["nvbit_tracer_build_hash"] == hashlib.sha256(
        tracer_bytes
    ).hexdigest()
    assert record["trace_admission_sha256"] == admission.trace_admission_sha256
    assert record["trace_filesystem_generation_sha256"] == (
        admission.trace_filesystem_generation_sha256
    )
    assert record["dynamic_trace_size_bytes"] == admission.dynamic_trace_size_bytes
    assert record["dynamic_trace_sha256"] == admission.dynamic_trace_sha256
    assert record["stats_size_bytes"] == admission.stats_size_bytes
    assert record["stats_sha256"] == admission.stats_sha256
    assert record["threadblock_payload_size_bytes"] == (
        admission.threadblock_payload_size_bytes
    )
    assert record["threadblock_payload_manifest_sha256"] == (
        admission.threadblock_payload_manifest_sha256
    )
    assert record["allocations_sidecar_size_bytes"] == (
        admission.allocations_sidecar_size_bytes
    )
    assert record["allocations_sidecar_sha256"] == (
        admission.allocations_sidecar_sha256
    )
    assert record["launch_count"] == 2
    assert [launch["launch_index"] for launch in record["launches"]] == [0, 1]


def test_roster_failed_rerun_preserves_attempt_and_retires_stale_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    manifest_path = output_root / "capture_manifest.json"
    roster = [{
        "slug": "fixture_direct",
        "shape_point": "timing_shape",
        "binary": "bin/fixture_direct",
        "app_args": [],
    }]
    plans: list[dict[str, Path | bool]] = [{
        "portable": False,
        "cell_dir": output_root / "direct" / "fixture_direct",
        "traces_folder": (
            output_root / "direct" / "fixture_direct" / "trace_timing_shape"
        ),
        "workload_output_root": (
            output_root / "direct" / "fixture_direct" / "workload_output"
        ),
        "portable_manifest": (
            output_root
            / "direct"
            / "fixture_direct"
            / "workload_output"
            / "fixture_direct"
            / "manifest.json"
        ),
    }]
    attempts = iter(("roster-attempt-one", "roster-attempt-two"))
    monkeypatch.setattr(
        roster_capture, "_new_capture_attempt_id", lambda: next(attempts)
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    outcomes = iter((True, False))
    observed_active_during_close: list[bool] = []

    class RetainedFixture:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            observed_active_during_close.append(manifest_path.is_file())
            return False

    def fake_direct(*_args, retention_stack, **_kwargs):
        retention_stack.enter_context(RetainedFixture())
        ok = next(outcomes)
        return (
            _valid_direct_aggregate_record()
            if ok
            else _valid_direct_exception_record()
        )

    monkeypatch.setattr(roster_capture, "_capture_direct", fake_direct)

    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 0
    first_active = manifest_path.read_bytes()
    assert observed_active_during_close == [True]

    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 1

    assert not manifest_path.exists()
    retired = tuple(output_root.glob(".capture_manifest.json.*.retired"))
    assert len(retired) == 1
    assert retired[0].read_bytes() == first_active
    attempts_root = output_root / "capture_manifest_attempts"
    assert {path.name for path in attempts_root.iterdir()} == {
        "roster-attempt-one.json",
        "roster-attempt-two.json",
    }


def test_roster_launch_vector_rejects_inconsistent_admission() -> None:
    admission = _admitted_trace(launch_count=2)
    missing_launch = replace(admission, launches=admission.launches[:1])
    with pytest.raises(RuntimeError, match="launch_count 2 disagrees with 1"):
        roster_capture._all_admitted_launch_records(missing_launch)

    reversed_launches = replace(admission, launches=tuple(reversed(admission.launches)))
    with pytest.raises(RuntimeError, match="indexes must be contiguous and ordered"):
        roster_capture._all_admitted_launch_records(reversed_launches)


def _valid_direct_aggregate_record(
    *,
    slug: str = "fixture_direct",
) -> dict:
    digest = "a" * 64
    return {
        "slug": slug,
        "shape_point": "timing_shape",
        "ok": True,
        "contract_ok": True,
        "trace_capture_attempt_id": "cell-attempt",
        "raw_trace_generation": {
            "attempt_id": "cell-attempt",
            "path": "trace_timing_shape/cell-attempt",
            "content_manifest_sha256": "b" * 64,
            "filesystem_generation_sha256": "c" * 64,
            "file_count": 1,
            "directory_count": 1,
            "total_file_bytes": 1,
        },
        "traces_folder": "/bounded/trace_timing_shape/cell-attempt",
        "trace_bytes": 1,
        "trace_pb_bytes": 1,
        "capture_seconds": 0.0,
        "records_expected": 1,
        "records_admitted": 1,
        "drop_count": 0,
        "binary_sha256": digest,
        "nvbit_tracer_build_hash": digest,
        "trace_admission_sha256": digest,
        "trace_filesystem_generation_sha256": "c" * 64,
        "dynamic_trace_size_bytes": 1,
        "dynamic_trace_sha256": digest,
        "allocations_sidecar_size_bytes": None,
        "allocations_sidecar_sha256": None,
        "stats_size_bytes": 0,
        "stats_sha256": digest,
        "threadblock_payload_size_bytes": 0,
        "threadblock_payload_manifest_sha256": digest,
        "launch_count": 1,
        "launches": [{
            "launch_index": 0,
            "device_id": 0,
            "stream_id": 0,
            "kernel_id": 0,
            "kernel_name": "fixture_kernel",
            "grid_x": 1,
            "grid_y": 1,
            "grid_z": 1,
            "block_x": 1,
            "block_y": 1,
            "block_z": 1,
            "shared_mem_static_bytes": 0,
            "shared_mem_dynamic_bytes": 0,
            "regs_per_thread": 0,
            "inst_count": 1,
        }],
        "via": "direct_injection",
    }


def _valid_direct_exception_record(
    *,
    slug: str = "fixture_direct",
) -> dict:
    return {
        "slug": slug,
        "shape_point": "timing_shape",
        "ok": False,
        "typed_reason": "capture_exception",
        "detail": "planted capture failure",
        "trace_capture_attempt_id": "cell-attempt",
        "via": "direct_injection",
    }


def _single_direct_roster_fixture(
    output_root: Path,
) -> tuple[list[dict], list[dict[str, Path | bool]], Path]:
    manifest_path = output_root / "capture_manifest.json"
    roster = [{
        "slug": "fixture_direct",
        "shape_point": "timing_shape",
        "binary": "bin/fixture_direct",
        "app_args": [],
    }]
    cell = output_root / "direct" / "fixture_direct"
    plans: list[dict[str, Path | bool]] = [{
        "portable": False,
        "cell_dir": cell,
        "traces_folder": cell / "trace_timing_shape",
        "workload_output_root": cell / "workload_output",
        "portable_manifest": (
            cell / "workload_output" / "fixture_direct" / "manifest.json"
        ),
    }]
    return roster, plans, manifest_path


@pytest.mark.parametrize(
    "hostile_case",
    (
        "top_extra_field",
        "cell_extra_field",
        "boolean_trace_bytes",
        "negative_trace_bytes",
        "nonfinite_capture_seconds",
        "boolean_device_id",
        "negative_device_id",
        "missing_launch_field",
        "launch_count_mismatch",
        "trace_bytes_mismatch",
        "boolean_raw_count",
        "raw_attempt_mismatch",
        "raw_path_attempt_mismatch",
        "raw_path_generation_root_mismatch",
        "raw_path_extra_depth",
        "traces_folder_attempt_mismatch",
        "zero_raw_file_count",
        "zero_raw_directory_count",
        "incomplete_raw_total_bytes",
        "drop_count_mismatch",
        "coercive_host_clock",
    ),
)
def test_aggregate_manifest_strictly_rejects_hostile_models(
    hostile_case: str,
) -> None:
    roster = [{"slug": "fixture_direct", "shape_point": "timing_shape"}]
    manifest = {
        "generated_by": "roster_trace_capture",
        "capture_attempt_id": "aggregate-attempt",
        "host_clock_state": {},
        "cells": [_valid_direct_aggregate_record()],
    }
    cell = manifest["cells"][0]
    launch = cell["launches"][0]
    if hostile_case == "top_extra_field":
        manifest["unexpected"] = "authority"
    elif hostile_case == "cell_extra_field":
        cell["unexpected"] = "authority"
    elif hostile_case == "boolean_trace_bytes":
        cell["trace_bytes"] = True
    elif hostile_case == "negative_trace_bytes":
        cell["trace_bytes"] = -1
    elif hostile_case == "nonfinite_capture_seconds":
        cell["capture_seconds"] = float("nan")
    elif hostile_case == "boolean_device_id":
        launch["device_id"] = False
    elif hostile_case == "negative_device_id":
        launch["device_id"] = -1
    elif hostile_case == "missing_launch_field":
        del launch["stream_id"]
    elif hostile_case == "launch_count_mismatch":
        cell["launch_count"] = 2
    elif hostile_case == "trace_bytes_mismatch":
        cell["trace_pb_bytes"] = 2
    elif hostile_case == "boolean_raw_count":
        cell["raw_trace_generation"]["file_count"] = True
    elif hostile_case == "raw_attempt_mismatch":
        cell["raw_trace_generation"]["attempt_id"] = "other-attempt"
    elif hostile_case == "raw_path_attempt_mismatch":
        cell["raw_trace_generation"]["path"] = (
            "trace_timing_shape/other-attempt"
        )
    elif hostile_case == "raw_path_generation_root_mismatch":
        cell["raw_trace_generation"]["path"] = (
            "other_generation_root/cell-attempt"
        )
    elif hostile_case == "raw_path_extra_depth":
        cell["raw_trace_generation"]["path"] = (
            "nested/trace_timing_shape/cell-attempt"
        )
        cell["traces_folder"] = (
            "/bounded/nested/trace_timing_shape/cell-attempt"
        )
    elif hostile_case == "traces_folder_attempt_mismatch":
        cell["traces_folder"] = (
            "/bounded/trace_timing_shape/other-attempt"
        )
    elif hostile_case == "zero_raw_file_count":
        cell["raw_trace_generation"]["file_count"] = 0
    elif hostile_case == "zero_raw_directory_count":
        cell["raw_trace_generation"]["directory_count"] = 0
    elif hostile_case == "incomplete_raw_total_bytes":
        cell["stats_size_bytes"] = 1
    elif hostile_case == "drop_count_mismatch":
        cell["drop_count"] = 1
    elif hostile_case == "coercive_host_clock":
        manifest["host_clock_state"] = {"clock": 7}
    else:  # pragma: no cover - exhaustive parameter invariant
        raise AssertionError(hostile_case)

    with pytest.raises(ValueError):
        roster_capture._validate_aggregate_manifest(manifest, roster)


def test_aggregate_manifest_accepts_complete_exact_model() -> None:
    roster = [{"slug": "fixture_direct", "shape_point": "timing_shape"}]
    manifest = {
        "generated_by": "roster_trace_capture",
        "capture_attempt_id": "aggregate-attempt",
        "host_clock_state": {},
        "cells": [_valid_direct_aggregate_record()],
    }
    assert roster_capture._validate_aggregate_manifest(manifest, roster) is manifest
    assert roster_capture._bounded_manifest_payload(manifest, roster).endswith(b"\n")


def test_roster_compositional_manifest_budget_precedes_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    required = roster_capture._require_roster_manifest_budget(len(roster))
    monkeypatch.setattr(
        roster_capture, "_MAX_ROSTER_MANIFEST_BYTES", required - 1,
    )
    monkeypatch.setattr(
        roster_capture,
        "require_file_descriptor_headroom",
        lambda _required, *, label: None,
    )
    monkeypatch.setattr(
        roster_capture,
        "_capture_direct",
        lambda *_args, **_kwargs: pytest.fail("capture must not start"),
    )

    with pytest.raises(
        ImmutableArtifactResourceError, match=VERIFY_RESOURCE_BOUND,
    ):
        roster_capture._run_roster_capture(
            roster=roster,
            plans=plans,
            manifest_path=manifest_path,
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    "row",
    (
        True,
        {"slug": True, "shape_point": "timing_shape", "app_args": []},
        {"slug": "fixture", "shape_point": 7, "app_args": []},
        {"slug": "fixture", "shape_point": "timing_shape", "app_args": [False]},
        {
            "slug": "fixture",
            "shape_point": "timing_shape",
            "binary": False,
            "app_args": [],
        },
    ),
)
def test_roster_plan_rejects_non_string_cell_identities(row: object) -> None:
    with pytest.raises((ValueError, ImmutableArtifactError)):
        roster_capture._roster_capture_plan(
            roster=[row],
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            output_root=STAGING / "invalid-roster-identity",
            manifest_out=None,
        )


def test_roster_retained_capability_maximum_fails_resource_typed() -> None:
    roster = [
        {
            "slug": f"fixture_{index}",
            "shape_point": "timing_shape",
            "app_args": [],
        }
        for index in range(roster_capture.MAX_RETAINED_ROSTER_CELLS + 1)
    ]
    with pytest.raises(
        ImmutableArtifactResourceError, match=VERIFY_RESOURCE_BOUND,
    ):
        roster_capture._roster_capture_plan(
            roster=roster,
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            output_root=STAGING / "roster-over-budget",
            manifest_out=None,
        )


def test_roster_descriptor_headroom_exhaustion_precedes_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)

    def reject_headroom(_required: int, *, label: str) -> None:
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: planted {label} exhaustion"
        )

    monkeypatch.setattr(
        roster_capture, "require_file_descriptor_headroom", reject_headroom,
    )
    monkeypatch.setattr(
        roster_capture,
        "_capture_direct",
        lambda *_args, **_kwargs: pytest.fail("capture must not start"),
    )
    with pytest.raises(
        ImmutableArtifactResourceError, match=VERIFY_RESOURCE_BOUND,
    ):
        roster_capture._run_roster_capture(
            roster=roster,
            plans=plans,
            manifest_path=manifest_path,
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize("mutation", ("replacement", "same_inode"))
def test_roster_final_aggregate_authentication_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    monkeypatch.setattr(
        roster_capture, "_new_capture_attempt_id", lambda: "aggregate-attempt"
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    winner_payload = b"concurrent aggregate winner\n"
    mutated_payload = b"mutated aggregate generation\n"
    winner_identity: tuple[int, int] | None = None

    class AggregateMutator:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal winner_identity
            assert manifest_path.is_file()
            if mutation == "replacement":
                replacement = manifest_path.with_name(".aggregate-winner")
                replacement.write_bytes(winner_payload)
                replacement.chmod(0o444)
                os.replace(replacement, manifest_path)
                winner_identity = (
                    manifest_path.stat().st_dev,
                    manifest_path.stat().st_ino,
                )
            else:
                manifest_path.chmod(0o644)
                manifest_path.write_bytes(mutated_payload)
            return False

    def fake_direct(*_args, retention_stack, **_kwargs):
        retention_stack.enter_context(AggregateMutator())
        return _valid_direct_aggregate_record()

    monkeypatch.setattr(roster_capture, "_capture_direct", fake_direct)
    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 2

    attempt = (
        output_root / "capture_manifest_attempts" / "aggregate-attempt.json"
    )
    assert attempt.is_file()
    if mutation == "replacement":
        assert winner_identity is not None
        assert (
            manifest_path.stat().st_dev,
            manifest_path.stat().st_ino,
        ) == winner_identity
        assert manifest_path.read_bytes() == winner_payload
        salvaged = tuple(
            path for path in output_root.glob(".failed-regular-*")
            if not path.name.endswith(".salvaging")
        )
        assert len(salvaged) == 1
        assert json.loads(salvaged[0].read_text(encoding="utf-8"))[
            "capture_attempt_id"
        ] == "aggregate-attempt"
    else:
        assert not manifest_path.exists()
        retired = tuple(
            output_root.glob(".capture_manifest.json.*.retired")
        )
        assert len(retired) == 1
        assert retired[0].read_bytes() == mutated_payload


@pytest.mark.parametrize("mutation", ("replacement", "same_inode"))
def test_roster_authenticated_close_rejects_monitor_close_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    monkeypatch.setattr(
        roster_capture, "_new_capture_attempt_id", lambda: "close-race-attempt",
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    monkeypatch.setattr(
        roster_capture,
        "_capture_direct",
        lambda *_args, **_kwargs: _valid_direct_aggregate_record(),
    )
    original_publish = roster_capture.publish_retained_regular_file
    winner_payload = b"aggregate close-race winner\n"
    mutated_payload = b"aggregate close-race mutation\n"
    winner_identity: tuple[int, int] | None = None

    def instrument_close(parent, name, payload, **kwargs):
        retained = original_publish(parent, name, payload, **kwargs)
        state = retained._monitor_state
        assert state is not None
        original_close = state.monitor.close

        def mutate_during_close() -> None:
            nonlocal winner_identity
            if mutation == "replacement":
                replacement = manifest_path.with_name(".close-race-winner")
                replacement.write_bytes(winner_payload)
                replacement.chmod(0o444)
                os.replace(replacement, manifest_path)
                winner_identity = (
                    manifest_path.stat().st_dev,
                    manifest_path.stat().st_ino,
                )
            else:
                manifest_path.chmod(0o644)
                manifest_path.write_bytes(mutated_payload)
            original_close()

        monkeypatch.setattr(state.monitor, "close", mutate_during_close)
        return retained

    monkeypatch.setattr(
        roster_capture, "publish_retained_regular_file", instrument_close,
    )
    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 2

    attempt = (
        output_root
        / "capture_manifest_attempts"
        / "close-race-attempt.json"
    )
    assert attempt.is_file()
    if mutation == "replacement":
        assert winner_identity is not None
        assert (
            manifest_path.stat().st_dev,
            manifest_path.stat().st_ino,
        ) == winner_identity
        assert manifest_path.read_bytes() == winner_payload
        salvaged = tuple(
            path for path in output_root.glob(".failed-regular-*")
            if not path.name.endswith(".salvaging")
        )
        assert len(salvaged) == 1
    else:
        assert not manifest_path.exists()
        retired = tuple(output_root.glob(".capture_manifest.json.*.retired"))
        assert len(retired) == 1
        assert retired[0].read_bytes() == mutated_payload


def test_roster_authenticated_close_rejects_final_pinned_close_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    monkeypatch.setattr(
        roster_capture, "_new_capture_attempt_id", lambda: "pinned-close-attempt",
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    monkeypatch.setattr(
        roster_capture,
        "_capture_direct",
        lambda *_args, **_kwargs: _valid_direct_aggregate_record(),
    )
    original_publish = roster_capture.publish_retained_regular_file
    winner_payload = b"aggregate final-pinned winner\n"
    winner_identity: tuple[int, int] | None = None

    def instrument_pinned_close(parent, name, payload, **kwargs):
        retained = original_publish(parent, name, payload, **kwargs)
        original_close = retained._pinned.close

        def replace_during_pinned_close() -> None:
            nonlocal winner_identity
            replacement = manifest_path.with_name(".final-pinned-winner")
            replacement.write_bytes(winner_payload)
            replacement.chmod(0o444)
            os.replace(replacement, manifest_path)
            winner_identity = (
                manifest_path.stat().st_dev,
                manifest_path.stat().st_ino,
            )
            original_close()

        monkeypatch.setattr(
            retained._pinned, "close", replace_during_pinned_close,
        )
        return retained

    monkeypatch.setattr(
        roster_capture,
        "publish_retained_regular_file",
        instrument_pinned_close,
    )
    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 2

    attempt = (
        output_root
        / "capture_manifest_attempts"
        / "pinned-close-attempt.json"
    )
    assert attempt.is_file()
    assert winner_identity is not None
    assert (
        manifest_path.stat().st_dev,
        manifest_path.stat().st_ino,
    ) == winner_identity
    assert manifest_path.read_bytes() == winner_payload
    salvaged = tuple(
        path for path in output_root.glob(".failed-regular-*")
        if not path.name.endswith(".salvaging")
    )
    assert len(salvaged) == 1


def test_roster_authenticated_close_rejects_guard_release_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    monkeypatch.setattr(
        roster_capture,
        "_new_capture_attempt_id",
        lambda: "guard-release-attempt",
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    monkeypatch.setattr(
        roster_capture,
        "_capture_direct",
        lambda *_args, **_kwargs: _valid_direct_aggregate_record(),
    )
    original_close_descriptors = immutable._close_descriptors
    original_preserve = roster_capture.preserve_retained_regular_file
    winner_payload = b"aggregate guard-release winner\n"
    winner_identity: tuple[int, int] | None = None
    guard_release_calls = 0
    replaced = False
    transferred_descriptors: list[int] = []

    def replace_during_guard_release(
        descriptors,
        *,
        active_error=None,
        label="descriptor close",
    ) -> None:
        nonlocal guard_release_calls, replaced, winner_identity
        materialized = tuple(descriptors)
        if label == "authenticated retained close guard release":
            guard_release_calls += 1
            if not replaced:
                replaced = True
                replacement = manifest_path.with_name(".guard-release-winner")
                replacement.write_bytes(winner_payload)
                replacement.chmod(0o444)
                os.replace(replacement, manifest_path)
                winner_identity = (
                    manifest_path.stat().st_dev,
                    manifest_path.stat().st_ino,
                )
        original_close_descriptors(
            materialized,
            active_error=active_error,
            label=label,
        )

    def observe_transferred_capability(retained, *, expected_payload):
        transferred_descriptors.extend(
            (
                retained._pinned.descriptor,
                *retained._pinned.directory_descriptors,
            )
        )
        return original_preserve(retained, expected_payload=expected_payload)

    monkeypatch.setattr(
        immutable, "_close_descriptors", replace_during_guard_release,
    )
    monkeypatch.setattr(
        roster_capture,
        "preserve_retained_regular_file",
        observe_transferred_capability,
    )
    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 2

    attempt = (
        output_root
        / "capture_manifest_attempts"
        / "guard-release-attempt.json"
    )
    assert attempt.is_file()
    assert replaced is True
    assert guard_release_calls == 1
    assert winner_identity is not None
    assert (
        manifest_path.stat().st_dev,
        manifest_path.stat().st_ino,
    ) == winner_identity
    assert manifest_path.read_bytes() == winner_payload
    salvaged = tuple(
        path for path in output_root.glob(".failed-regular-*")
        if not path.name.endswith(".salvaging")
    )
    assert len(salvaged) == 1
    assert json.loads(salvaged[0].read_text(encoding="utf-8"))[
        "capture_attempt_id"
    ] == "guard-release-attempt"
    assert transferred_descriptors
    for descriptor in transferred_descriptors:
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_roster_authenticated_close_owns_aggregate_lock_reader_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    monkeypatch.setattr(
        roster_capture,
        "_new_capture_attempt_id",
        lambda: "aggregate-lock-close-attempt",
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    monkeypatch.setattr(
        roster_capture,
        "_capture_direct",
        lambda *_args, **_kwargs: _valid_direct_aggregate_record(),
    )
    original_lock = roster_capture._lock_activation_cell
    original_reader_close = immutable.PinnedDirectoryReader.close
    original_authenticated_close = (
        roster_capture.close_retained_regular_files_authenticated
    )
    original_preserve = roster_capture.preserve_retained_regular_file
    locked_reader = None
    winner_payload = b"aggregate lock-reader-close winner\n"
    winner_identity: tuple[int, int] | None = None
    lock_reader_close_calls = 0
    replaced = False
    observed_close_errors: list[BaseException] = []
    transferred_descriptors: list[int] = []

    def retain_locked_reader(reader) -> None:
        nonlocal locked_reader
        locked_reader = reader
        original_lock(reader)

    def replace_during_lock_reader_close(self) -> None:
        nonlocal lock_reader_close_calls, replaced, winner_identity
        if self is locked_reader:
            lock_reader_close_calls += 1
            if not replaced:
                replaced = True
                replacement = manifest_path.with_name(
                    ".aggregate-lock-close-winner"
                )
                replacement.write_bytes(winner_payload)
                replacement.chmod(0o444)
                os.replace(replacement, manifest_path)
                winner_identity = (
                    manifest_path.stat().st_dev,
                    manifest_path.stat().st_ino,
                )
        original_reader_close(self)

    def observe_authenticated_close(*args, **kwargs) -> None:
        try:
            original_authenticated_close(*args, **kwargs)
        except BaseException as error:
            observed_close_errors.append(error)
            raise

    def observe_transferred_capability(retained, *, expected_payload):
        transferred_descriptors.extend(
            (
                retained._pinned.descriptor,
                *retained._pinned.directory_descriptors,
            )
        )
        return original_preserve(retained, expected_payload=expected_payload)

    monkeypatch.setattr(
        roster_capture, "_lock_activation_cell", retain_locked_reader,
    )
    monkeypatch.setattr(
        immutable.PinnedDirectoryReader,
        "close",
        replace_during_lock_reader_close,
    )
    monkeypatch.setattr(
        roster_capture,
        "close_retained_regular_files_authenticated",
        observe_authenticated_close,
    )
    monkeypatch.setattr(
        roster_capture,
        "preserve_retained_regular_file",
        observe_transferred_capability,
    )
    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 2

    attempt = (
        output_root
        / "capture_manifest_attempts"
        / "aggregate-lock-close-attempt.json"
    )
    assert attempt.is_file()
    assert replaced is True
    assert lock_reader_close_calls == 1
    assert len(observed_close_errors) == 1
    assert type(observed_close_errors[0]) is ImmutableArtifactError
    assert winner_identity is not None
    assert (
        manifest_path.stat().st_dev,
        manifest_path.stat().st_ino,
    ) == winner_identity
    assert manifest_path.read_bytes() == winner_payload
    salvaged = tuple(
        path for path in output_root.glob(".failed-regular-*")
        if not path.name.endswith(".salvaging")
    )
    assert len(salvaged) == 1
    assert json.loads(salvaged[0].read_text(encoding="utf-8"))[
        "capture_attempt_id"
    ] == "aggregate-lock-close-attempt"
    assert transferred_descriptors
    for descriptor in transferred_descriptors:
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_failed_direct_capture_exposes_retained_close_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "direct" / "fixture_run"
    tracer = tmp_path / "tracer_tool.so"
    _write_executable(binary, b"direct failure executable")
    tracer.write_bytes(b"direct failure tracer")
    monkeypatch.setattr(roster_capture, "_TRACER_SO", tracer)
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_tracer_env", lambda **_kwargs: {})
    original_open = roster_capture.open_retained_executable

    class CloseFailingExecutable:
        def __init__(self, retained) -> None:
            self._retained = retained

        def __getattr__(self, name):
            return getattr(self._retained, name)

        def __enter__(self):
            self._retained.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._retained.__exit__(exc_type, exc, traceback)
            raise ImmutableArtifactError("planted direct close identity failure")

    monkeypatch.setattr(
        roster_capture,
        "open_retained_executable",
        lambda path: CloseFailingExecutable(original_open(path)),
    )
    monkeypatch.setattr(
        roster_capture.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 23, "", "planted capture failure"
        ),
    )

    record = roster_capture._capture_direct(
        "fixture_direct",
        binary,
        [],
        tmp_path / "trace",
        tmp_path / "workload-output",
        attempt_id="direct-failure-attempt",
    )

    assert record["ok"] is False
    assert record["typed_reason"] == "capture_failed"
    assert record["returncode"] == 23
    assert record["close_diagnostics"] == [{
        "typed_reason": "retained_source_close_failed",
        "detail": "planted direct close identity failure",
    }]


def test_signed_subprocess_returncode_preserves_capture_failed_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "direct" / "fixture_run"
    tracer = tmp_path / "tracer_tool.so"
    binary_payload = b"signaled direct executable"
    tracer_payload = b"signaled direct tracer"
    _write_executable(binary, binary_payload)
    tracer.write_bytes(tracer_payload)
    monkeypatch.setattr(roster_capture, "_TRACER_SO", tracer)
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_tracer_env", lambda **_kwargs: {})
    monkeypatch.setattr(
        roster_capture.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, -9, "", "terminated by signal 9",
        ),
    )

    record = roster_capture._capture_direct(
        "fixture_direct",
        binary,
        [],
        tmp_path / "trace",
        tmp_path / "workload-output",
        attempt_id="signed-returncode-attempt",
    )
    assert record["typed_reason"] == "capture_failed"
    assert record["returncode"] == -9
    assert record["stderr_tail"] == "terminated by signal 9"
    assert record["binary_sha256"] == hashlib.sha256(binary_payload).hexdigest()
    assert record["nvbit_tracer_build_hash"] == hashlib.sha256(
        tracer_payload
    ).hexdigest()
    record["shape_point"] = "timing_shape"
    record["via"] = "direct_injection"
    assert roster_capture._validate_capture_record(
        record,
        expected_slug="fixture_direct",
        expected_shape_point="timing_shape",
        index=0,
    ) is record


def test_direct_exception_close_diagnostic_enters_attempt_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    binary = tmp_path / "bin" / "fixture_direct"
    tracer = tmp_path / "tracer_tool.so"
    _write_executable(binary, b"direct exception executable")
    tracer.write_bytes(b"direct exception tracer")
    monkeypatch.setattr(roster_capture, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(roster_capture, "_TRACER_SO", tracer)
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    monkeypatch.setattr(roster_capture, "_tracer_env", lambda **_kwargs: {})
    attempts = iter(("aggregate-exception", "direct-exception"))
    monkeypatch.setattr(
        roster_capture, "_new_capture_attempt_id", lambda: next(attempts),
    )
    original_open = roster_capture.open_retained_executable

    class CloseFailingExecutable:
        def __init__(self, retained) -> None:
            self._retained = retained

        def __getattr__(self, name):
            return getattr(self._retained, name)

        def __enter__(self):
            self._retained.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._retained.__exit__(exc_type, exc, traceback)
            raise ImmutableArtifactError("planted exception close failure")

    monkeypatch.setattr(
        roster_capture,
        "open_retained_executable",
        lambda path: CloseFailingExecutable(original_open(path)),
    )
    monkeypatch.setattr(
        roster_capture.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("planted direct subprocess exception")
        ),
    )

    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 1
    attempt = json.loads((
        output_root
        / "capture_manifest_attempts"
        / "aggregate-exception.json"
    ).read_text(encoding="utf-8"))
    record = attempt["cells"][0]
    assert record["typed_reason"] == "capture_exception"
    assert record["trace_capture_attempt_id"] == "direct-exception"
    assert record["close_diagnostics"] == [{
        "typed_reason": "retained_source_close_failed",
        "detail": "planted exception close failure",
    }]


@pytest.mark.parametrize(
    ("boundary", "resource_errno"),
    (
        ("retained_open", errno.EMFILE),
        ("stage_open", errno.ENFILE),
        ("subprocess_run", errno.EMFILE),
    ),
)
def test_direct_late_descriptor_exhaustion_keeps_resource_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    resource_errno: int,
) -> None:
    binary = tmp_path / "direct" / "fixture_run"
    tracer = tmp_path / "tracer_tool.so"
    _write_executable(binary, b"resource executable")
    tracer.write_bytes(b"resource tracer")
    monkeypatch.setattr(roster_capture, "_TRACER_SO", tracer)
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_tracer_env", lambda **_kwargs: {})
    monkeypatch.setattr(
        roster_capture.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )

    if boundary == "retained_open":
        def exhaust_retained_open(*_args, **_kwargs):
            raise OSError(resource_errno, "planted retained-open exhaustion")

        monkeypatch.setattr(
            immutable,
            "_open_regular_with_early_path_monitor",
            exhaust_retained_open,
        )
    elif boundary == "stage_open":
        def exhaust_stage_open(*_args, **_kwargs):
            raise OSError(resource_errno, "planted stage-open exhaustion")

        monkeypatch.setattr(
            immutable, "_open_pinned_relative_directory", exhaust_stage_open,
        )
    else:
        def exhaust_subprocess(*_args, **_kwargs):
            raise OSError(resource_errno, "planted direct subprocess exhaustion")

        monkeypatch.setattr(
            roster_capture.subprocess, "run", exhaust_subprocess,
        )

    record = roster_capture._capture_direct(
        "fixture_direct",
        binary,
        [],
        tmp_path / "trace",
        tmp_path / "workload-output",
        attempt_id="resource-attempt",
    )
    assert record["ok"] is False
    assert record["typed_reason"] == "trace_snapshot_resource_unavailable"
    assert VERIFY_RESOURCE_BOUND in record["detail"]
    assert record["trace_capture_attempt_id"] == "resource-attempt"
    if boundary == "stage_open":
        assert (tmp_path / "trace" / ".resource-attempt.staging").is_dir()


def test_direct_capture_rejects_boolean_attempt_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    with pytest.raises(ValueError, match="exact string"):
        roster_capture._capture_direct(
            "fixture_direct",
            tmp_path / "missing-binary",
            [],
            tmp_path / "trace",
            tmp_path / "workload-output",
            attempt_id=True,
        )
    assert not (tmp_path / "trace").exists()
    assert not (tmp_path / "workload-output").exists()


def test_roster_rejects_boolean_launch_authority() -> None:
    admission = _admitted_trace()
    with pytest.raises(RuntimeError, match="exact integer"):
        roster_capture._all_admitted_launch_records(
            replace(admission, launch_count=True)
        )
    boolean_launch = replace(admission.launches[0], launch_index=False)
    with pytest.raises(RuntimeError, match="exact integer"):
        roster_capture._all_admitted_launch_records(
            replace(admission, launches=(boolean_launch,))
        )


def test_roster_invalid_aggregate_identity_never_becomes_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    monkeypatch.setattr(
        roster_capture, "_new_capture_attempt_id", lambda: "invalid-aggregate"
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})
    monkeypatch.setattr(
        roster_capture,
        "_capture_direct",
        lambda *_args, **_kwargs: {
            "slug": "fixture_direct",
            "ok": True,
            "contract_ok": True,
            "trace_capture_attempt_id": True,
        },
    )

    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 1
    assert not manifest_path.exists()
    attempt = json.loads((
        output_root
        / "capture_manifest_attempts"
        / "invalid-aggregate.json"
    ).read_text(encoding="utf-8"))
    assert attempt["cells"][0]["typed_reason"] == "capture_exception"


def test_roster_cell_resource_failure_keeps_existing_typed_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "roster"
    roster, plans, manifest_path = _single_direct_roster_fixture(output_root)
    monkeypatch.setattr(
        roster_capture, "_new_capture_attempt_id", lambda: "resource-attempt"
    )
    monkeypatch.setattr(
        roster_capture,
        "_assert_staging_output",
        lambda path, *, operation: Path(path),
    )
    monkeypatch.setattr(roster_capture, "_host_clock_state", lambda: {})

    def reject_capture(*_args, **_kwargs):
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: planted retained descriptor exhaustion"
        )

    monkeypatch.setattr(roster_capture, "_capture_direct", reject_capture)
    assert roster_capture._run_roster_capture(
        roster=roster,
        plans=plans,
        manifest_path=manifest_path,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=output_root,
    ) == 1
    attempt = json.loads((
        output_root
        / "capture_manifest_attempts"
        / "resource-attempt.json"
    ).read_text(encoding="utf-8"))
    assert attempt["cells"][0]["typed_reason"] == (
        "trace_snapshot_resource_unavailable"
    )
    assert VERIFY_RESOURCE_BOUND in attempt["cells"][0]["detail"]
    assert not manifest_path.exists()


def test_register_cap_safe_output_plan() -> None:
    plan = variants._variant_output_plan(
        slug="register_spill_local_mem",
        shape_point="trace_l2_accuracy_xl",
        register_cap=64,
        sku="rtx_3090_ti",
        arch_dir="sm_86",
        output_root=STAGING / "test_variants",
    )
    assert all(path.is_relative_to(STAGING) for path in plan.values())
    assert plan["tmp_path"].name.endswith(".tmp")


def test_register_cap_main_rejects_duplicate_caps_before_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        variants,
        "_variant_output_plan",
        lambda **_kwargs: pytest.fail("duplicate caps must fail before planning"),
    )
    assert variants.main([
        "--shape-point",
        "trace_l2_accuracy_xl",
        "--caps",
        "64,64",
    ]) == 2


def test_register_cap_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        variants._variant_output_plan(
            slug="register_spill_local_mem",
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            output_root=PROTECTED / "variants",
        )
    assert calls == []


def test_register_cap_source_commit_binds_complete_retained_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    head = "a" * 40
    payloads = {
        relative_path: f"authority {relative_path}\n".encode("utf-8")
        for relative_path in variants._SOURCE_COMMIT_AUTHORITY_PATHS
    }
    retained = {
        relative_path: _RetainedBytesFixture(payload)
        for relative_path, payload in payloads.items()
    }
    heads = iter((head, head))
    commands: list[list[str]] = []
    monkeypatch.setattr(variants, "_REPO_ROOT", repo)
    monkeypatch.setattr(variants, "_git_head", lambda: next(heads))

    def git_run(command: list[str], **_kwargs):
        commands.append(command)
        if command[3] == "status":
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[3] == "show":
            relative_path = command[4].split(":", 1)[1]
            return subprocess.CompletedProcess(
                command, 0, payloads[relative_path], b""
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(variants.subprocess, "run", git_run)

    assert variants._resolve_source_commit(retained) == (head, False)
    assert all(item.verify_count == 1 for item in retained.values())
    assert (
        "workloads/simbench_bw/shapes/register_spill_local_mem.toml"
        in variants._SOURCE_COMMIT_AUTHORITY_PATHS
    )
    assert commands[0] == [
        "git",
        "-C",
        str(repo),
        "status",
        "--porcelain",
        "--",
        *variants._SOURCE_COMMIT_AUTHORITY_PATHS,
    ]
    assert [command[4] for command in commands[1:]] == [
        f"{head}:{relative_path}"
        for relative_path in variants._SOURCE_COMMIT_AUTHORITY_PATHS
    ]


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("dirty", "authority is dirty or untracked"),
        ("byte_mismatch", "authority bytes disagree"),
        ("head_drift", "changed during authority verification"),
    ),
)
def test_register_cap_source_commit_rejects_contamination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    repo = tmp_path / "repo"
    head = "a" * 40
    payloads = {
        relative_path: f"authority {relative_path}\n".encode("utf-8")
        for relative_path in variants._SOURCE_COMMIT_AUTHORITY_PATHS
    }
    retained = {
        relative_path: _RetainedBytesFixture(payload)
        for relative_path, payload in payloads.items()
    }
    heads = (
        iter((head, "b" * 40))
        if failure == "head_drift"
        else iter((head, head))
    )
    monkeypatch.setattr(variants, "_REPO_ROOT", repo)
    monkeypatch.setattr(variants, "_git_head", lambda: next(heads))

    def git_run(command: list[str], **_kwargs):
        if command[3] == "status":
            detail = (
                " M workloads/simbench_bw/shapes/"
                "register_spill_local_mem.toml\n"
                if failure == "dirty"
                else ""
            )
            return subprocess.CompletedProcess(command, 0, detail, "")
        if command[3] == "show":
            relative_path = command[4].split(":", 1)[1]
            payload = payloads[relative_path]
            if (
                failure == "byte_mismatch"
                and relative_path.endswith("register_spill_local_mem.toml")
            ):
                payload = b"different committed shape authority\n"
            return subprocess.CompletedProcess(command, 0, payload, b"")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(variants.subprocess, "run", git_run)

    with pytest.raises(RuntimeError, match=message):
        variants._resolve_source_commit(retained)


def test_register_cap_dirty_override_retains_every_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    head = "a" * 40
    retained = {
        relative_path: _RetainedBytesFixture(
            f"dirty authority {relative_path}\n".encode("utf-8")
        )
        for relative_path in variants._SOURCE_COMMIT_AUTHORITY_PATHS
    }
    heads = iter((head, head))
    monkeypatch.setattr(variants, "_REPO_ROOT", repo)
    monkeypatch.setattr(variants, "_git_head", lambda: next(heads))

    def git_run(command: list[str], **_kwargs):
        if command[3] == "status":
            return subprocess.CompletedProcess(
                command,
                0,
                " M tools/local_labels/trace_register_cap_variant_cells.py\n",
                "",
            )
        if command[3] == "show":
            return subprocess.CompletedProcess(
                command, 0, b"committed authority differs\n", b""
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(variants.subprocess, "run", git_run)

    assert variants._resolve_source_commit(
        retained, allow_dirty=True
    ) == (head, True)
    assert all(item.verify_count == 1 for item in retained.values())


def test_register_cap_lock_precedes_attempt_root_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    plan["variant_dir"].mkdir(parents=True)

    def reject_lock(_reader) -> None:
        raise ImmutableArtifactError("planted live register-cap publisher")

    monkeypatch.setattr(variants, "_lock_activation_cell", reject_lock)
    with pytest.raises(
        ImmutableArtifactError,
        match="planted live register-cap publisher",
    ):
        variants.trace_variant_cell(
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            _precleared_plan=plan,
        )

    assert tuple(plan["variant_dir"].iterdir()) == ()
    for key in (
        "traces_folder",
        "summary_generation_root",
        "attempt_record_root",
        "workload_output_root",
    ):
        assert not plan[key].exists()


def test_register_cap_summary_emission_stays_on_stage_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    attempt_id = "variant-summary-race"
    monkeypatch.setattr(
        variants, "_new_capture_attempt_id", lambda: attempt_id
    )
    original_emit = variants.emit_admitted_nvbit_trace_summary
    displaced = plan["summary_generation_root"] / ".displaced-summary-stage"
    planted_payload = b"foreign summary winner\n"
    observed_descriptor: int | None = None

    def replace_stage_path(**kwargs):
        nonlocal observed_descriptor
        observed_descriptor = kwargs.get("output_descriptor")
        assert isinstance(observed_descriptor, int)
        assert kwargs["output_path"] == Path(variants._SUMMARY_NAME)
        staging = (
            plan["summary_generation_root"] / f".{attempt_id}.staging"
        )
        staging.rename(displaced)
        staging.mkdir()
        (staging / variants._SUMMARY_NAME).write_bytes(planted_payload)
        return original_emit(**kwargs)

    monkeypatch.setattr(
        variants,
        "emit_admitted_nvbit_trace_summary",
        replace_stage_path,
    )
    with pytest.raises(ImmutableArtifactError):
        variants.trace_variant_cell(
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            _precleared_plan=plan,
        )

    assert observed_descriptor is not None
    assert (displaced / variants._SUMMARY_NAME).is_file()
    assert (
        plan["summary_generation_root"]
        / f".{attempt_id}.staging"
        / variants._SUMMARY_NAME
    ).read_bytes() == planted_payload
    assert not plan["summary_path"].exists()
    assert not plan["sidecar_path"].exists()


def test_register_cap_summary_comparison_is_byte_and_type_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    attempt_id = "variant-summary-type-substitution"
    monkeypatch.setattr(
        variants, "_new_capture_attempt_id", lambda: attempt_id
    )
    original_create = variants.create_exclusive_directory_stage

    def create_mutating_stage(output_root: Path, final_name: str):
        stage = original_create(output_root, final_name)
        if Path(output_root) == plan["summary_generation_root"]:
            original_commit = stage.commit

            def mutate_then_commit(**kwargs):
                summary_path = stage.staging_path / variants._SUMMARY_NAME
                record = json.loads(summary_path.read_text(encoding="utf-8"))
                assert record["repeat_index"] == 0
                record["repeat_index"] = False
                summary_path.chmod(0o644)
                summary_path.write_text(
                    json.dumps(record, indent=2) + "\n",
                    encoding="utf-8",
                )
                summary_path.chmod(0o444)
                return original_commit(**kwargs)

            stage.commit = mutate_then_commit
        return stage

    monkeypatch.setattr(
        variants,
        "create_exclusive_directory_stage",
        create_mutating_stage,
    )
    with pytest.raises(
        ImmutableArtifactError,
        match="summary bytes disagree with emitter output",
    ):
        variants.trace_variant_cell(
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            _precleared_plan=plan,
        )

    assert (
        plan["summary_generation_root"]
        / attempt_id
        / variants._SUMMARY_NAME
    ).is_file()
    assert not plan["summary_path"].exists()
    assert not plan["sidecar_path"].exists()


def test_register_cap_head_drift_after_attempt_commit_stays_nonactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    attempt_id = "variant-head-drift"
    source_commit = "e" * 40
    current_head = {"value": source_commit}
    monkeypatch.setattr(
        variants, "_new_capture_attempt_id", lambda: attempt_id
    )
    monkeypatch.setattr(
        variants, "_git_head", lambda: current_head["value"]
    )
    original_publish = variants.publish_regular_file

    def drift_after_attempt_record(parent, name, payload, **kwargs):
        result = original_publish(parent, name, payload, **kwargs)
        if Path(parent) == plan["attempt_record_root"]:
            current_head["value"] = "f" * 40
        return result

    monkeypatch.setattr(
        variants, "publish_regular_file", drift_after_attempt_record
    )
    with pytest.raises(
        RuntimeError,
        match="source_commit changed during capture publication",
    ):
        variants.trace_variant_cell(
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            _precleared_plan=plan,
        )

    assert (
        plan["attempt_record_root"] / f"{attempt_id}.json"
    ).is_file()
    assert (plan["traces_folder"] / attempt_id).is_dir()
    assert (plan["summary_generation_root"] / attempt_id).is_dir()
    assert not plan["summary_path"].exists()
    assert not plan["sidecar_path"].exists()


def test_register_cap_late_source_close_mutation_retires_locators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    attempt_id = "variant-late-summary-mutation"
    monkeypatch.setattr(
        variants, "_new_capture_attempt_id", lambda: attempt_id
    )
    original_open = variants.open_retained_executable
    mutated = False

    class MutatingExecutable:
        def __init__(self, retained) -> None:
            self._retained = retained

        def __getattr__(self, name):
            return getattr(self._retained, name)

        def __enter__(self):
            self._retained.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal mutated
            result = self._retained.__exit__(exc_type, exc, traceback)
            summary_generation = (
                plan["summary_generation_root"] / attempt_id
            )
            summary = summary_generation / variants._SUMMARY_NAME
            summary_generation.chmod(0o755)
            summary.chmod(0o644)
            summary.write_bytes(b"late source-close mutation\n")
            mutated = True
            return result

    monkeypatch.setattr(
        variants,
        "open_retained_executable",
        lambda path: MutatingExecutable(original_open(path)),
    )
    with pytest.raises(ImmutableArtifactError):
        variants.trace_variant_cell(
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            _precleared_plan=plan,
        )

    assert mutated is True
    assert not plan["sidecar_path"].exists()
    assert not plan["summary_path"].exists()
    assert len(tuple(plan["variant_dir"].glob(
        ".trace_trace_l2_accuracy_xl.json.*.retired"
    ))) == 1
    assert len(tuple(plan["variant_dir"].glob(
        ".nvbit_trace_summary_0_0.json.*.retired"
    ))) == 1


def test_register_cap_producer_propagates_retained_trace_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace(with_allocations=True)
    plan, tracer_bytes = _variant_capture_fixture(
        tmp_path,
        monkeypatch,
        admission,
    )

    sidecar = variants.trace_variant_cell(
        shape_point="trace_l2_accuracy_xl",
        register_cap=64,
        _precleared_plan=plan,
    )

    tracer_sha256 = hashlib.sha256(tracer_bytes).hexdigest()
    assert sidecar["nvbit_tracer_build_hash"] == tracer_sha256
    _assert_sidecar_admission_identities(sidecar, admission)
    assert sidecar["allocations_sidecar_size_bytes"] == 11
    assert sidecar["allocations_sidecar_sha256"] == "5" * 64
    assert json.loads(plan["sidecar_path"].read_text(encoding="utf-8")) == sidecar
    summary = json.loads(plan["summary_path"].read_text(encoding="utf-8"))
    assert summary["nvbit_tracer_build_hash"] == tracer_sha256
    assert summary["nvbit_trace_pb_sha256"] == admission.dynamic_trace_sha256
    assert (
        summary["nvbit_threadblock_payload_size_bytes"]
        == admission.threadblock_payload_size_bytes
    )
    assert (
        summary["nvbit_threadblock_payload_manifest_sha256"]
        == admission.threadblock_payload_manifest_sha256
    )
    assert summary["trace_admission_sha256"] == admission.trace_admission_sha256


def test_register_cap_descriptor_bound_launch_and_attempt_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    observed: dict[str, object] = {}

    def capture_launch(command, env, _cwd, *, pass_fds):
        observed["command"] = list(command)
        observed["env"] = dict(env)
        observed["pass_fds"] = tuple(pass_fds)
        (Path(env["TRACES_FOLDER"]) / "dynamic_trace.pb").write_bytes(
            b"fixture trace"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(variants, "_run_workload", capture_launch)
    monkeypatch.setattr(
        variants, "_new_capture_attempt_id", lambda: "variant-attempt"
    )
    sidecar = variants.trace_variant_cell(
        shape_point="trace_l2_accuracy_xl",
        register_cap=64,
        _precleared_plan=plan,
    )

    command = observed["command"]
    env = observed["env"]
    pass_fds = observed["pass_fds"]
    assert isinstance(command, list)
    assert isinstance(env, dict)
    assert isinstance(pass_fds, tuple)
    proc_paths = {
        command[0],
        str(env["CUDA_INJECTION64_PATH"]),
        str(env["TRACES_FOLDER"]),
    }
    assert all(path.startswith("/proc/self/fd/") for path in proc_paths)
    assert {
        int(path.rsplit("/", 1)[1]) for path in proc_paths
    }.issubset(set(pass_fds))
    assert len(pass_fds) == 5
    assert sidecar["trace_capture_attempt_id"] == "variant-attempt"
    assert (
        plan["variant_dir"] / sidecar["raw_trace_generation"]["path"]
    ).is_dir()
    assert (
        plan["variant_dir"] / sidecar["summary_generation"]["path"]
    ).is_dir()


def test_register_cap_dirty_rerun_stays_nonactive_and_preserves_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    attempts = iter(("variant-clean", "variant-dirty"))
    dirty_states = iter((False, True))
    monkeypatch.setattr(
        variants, "_new_capture_attempt_id", lambda: next(attempts)
    )
    monkeypatch.setattr(
        variants,
        "_resolve_source_commit",
        lambda *_args, **_kwargs: ("e" * 40, next(dirty_states)),
    )
    variants.trace_variant_cell(
        shape_point="trace_l2_accuracy_xl",
        register_cap=64,
        _precleared_plan=plan,
    )
    active_sidecar = plan["sidecar_path"].read_bytes()
    active_summary = plan["summary_path"].read_bytes()

    with pytest.raises(RuntimeError, match="provenance_dirty_override"):
        variants.trace_variant_cell(
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            allow_dirty=True,
            _precleared_plan=plan,
        )

    assert plan["sidecar_path"].read_bytes() == active_sidecar
    assert plan["summary_path"].read_bytes() == active_summary
    assert {
        path.name for path in plan["traces_folder"].iterdir()
    } == {"variant-clean", "variant-dirty"}
    assert {
        path.name for path in plan["summary_generation_root"].iterdir()
    } == {"variant-clean", "variant-dirty"}
    dirty_record = json.loads(
        (plan["attempt_record_root"] / "variant-dirty.json").read_text(
            encoding="utf-8"
        )
    )
    assert dirty_record["activation_state"] == "non_admissible"
    assert dirty_record["typed_reason"] == "provenance_dirty_override"


def test_register_cap_two_successful_generations_coexist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace()
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path, monkeypatch, admission
    )
    attempts = iter(("variant-one", "variant-two"))
    monkeypatch.setattr(
        variants, "_new_capture_attempt_id", lambda: next(attempts)
    )
    first = variants.trace_variant_cell(
        shape_point="trace_l2_accuracy_xl",
        register_cap=64,
        _precleared_plan=plan,
    )
    first_raw = plan["variant_dir"] / first["raw_trace_generation"]["path"]
    first_bytes = (first_raw / "dynamic_trace.pb").read_bytes()
    second = variants.trace_variant_cell(
        shape_point="trace_l2_accuracy_xl",
        register_cap=64,
        _precleared_plan=plan,
    )

    assert (first_raw / "dynamic_trace.pb").read_bytes() == first_bytes
    assert (
        plan["variant_dir"] / second["raw_trace_generation"]["path"]
    ).is_dir()
    assert len(tuple(plan["traces_folder"].iterdir())) == 2
    assert len(tuple(plan["summary_generation_root"].iterdir())) == 2
    assert json.loads(plan["sidecar_path"].read_text(encoding="utf-8"))[
        "trace_capture_attempt_id"
    ] == "variant-two"


def test_register_cap_producer_rejects_multi_launch_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admitted_trace(launch_count=2)
    plan, _tracer_bytes = _variant_capture_fixture(
        tmp_path,
        monkeypatch,
        admission,
    )
    monkeypatch.setattr(
        variants,
        "emit_admitted_nvbit_trace_summary",
        lambda **_kwargs: pytest.fail("summary emission must not select launch zero"),
    )

    with pytest.raises(RuntimeError, match=r"launch_count == 1; got 2"):
        variants.trace_variant_cell(
            shape_point="trace_l2_accuracy_xl",
            register_cap=64,
            _precleared_plan=plan,
        )

    assert not plan["summary_path"].exists()
    assert not plan["sidecar_path"].exists()


def _golden_plan_environment(**additions: str) -> dict[str, str]:
    environment = {
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": "/bin:/usr/bin",
    }
    environment.update(additions)
    return environment


def test_golden_capture_safe_plan_is_nonmutating() -> None:
    output = STAGING / "golden_cells"
    existed = output.exists()
    proc = subprocess.run(
        [str(GOLDEN_SCRIPT), "--plan-only", "gs_stream_copy"],
        cwd=REPO,
        env=_golden_plan_environment(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"root={output}" in proc.stdout
    assert output.exists() is existed


def test_golden_capture_retired_output_override_cannot_redirect() -> None:
    forbidden = PROTECTED / "guard_negative_must_not_exist"
    assert not forbidden.exists()
    proc = subprocess.run(
        [str(GOLDEN_SCRIPT), "--plan-only", "gs_stream_copy"],
        cwd=REPO,
        env=_golden_plan_environment(
            NVBIT_CAPTURE_OUTPUT_ROOT=str(forbidden),
        ),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"root={STAGING / 'golden_cells'}" in proc.stdout
    assert not forbidden.exists()


def test_canary_safe_output_plan() -> None:
    plan = canary._canary_output_plan(
        out=STAGING / "test_canary" / "manifest.json",
        work=STAGING / "test_canary" / "work",
    )
    assert len([key for key in plan if key.endswith("_binary")]) == 3
    assert len([key for key in plan if key.endswith("_trace")]) == 3
    assert all(path.is_relative_to(STAGING) for path in plan.values())


def test_canary_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        canary._canary_output_plan(
            out=STAGING / "test_canary" / "manifest.json",
            work=PROTECTED / "canary",
        )
    assert calls == []


def test_campaign_preflight_safe_output_plan(tmp_path: Path) -> None:
    output, smoke = campaign._campaign_output_plan(
        output=STAGING / "test_campaign" / "preflight.json",
        smoke_temp_root=tmp_path / "smoke",
    )
    assert output.is_relative_to(STAGING)
    assert all(not is_protected_path(path, repo_root=REPO) for path in smoke.values())
    assert not smoke["root"].exists()


def test_campaign_preflight_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        campaign._campaign_output_plan(output=PROTECTED / "preflight.json")
    assert calls == []


def test_measurement_stager_safe_mirrored_output_plan() -> None:
    source = (
        REPO / "workloads" / "portable_lcd" / "results" / "slug"
        / "rtx_3090_ti" / "sm_86"
    )
    target = staging._staging_artifact_dir(
        source, output_root=STAGING / "test_manifests",
    )
    assert target.is_relative_to(STAGING)
    assert target.parts[-6:] == source.parts[-6:]


def test_measurement_stager_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        staging._staging_artifact_dir(
            REPO / "workloads" / "portable_lcd" / "results" / "slug",
            output_root=PROTECTED / "manifests",
        )
    assert calls == []


def test_sm120_bundle_safe_output_plan(tmp_path: Path) -> None:
    plan = sm120._bundle_output_plan(
        candidates=[("register_spill_local_mem", "trace_l2_accuracy_xl")],
        output_root=STAGING / "test_sm120",
        helper_output=tmp_path / "sm120_helper",
    )
    cell = plan["cells"][("register_spill_local_mem", "trace_l2_accuracy_xl")]
    assert plan["output_root"].is_relative_to(STAGING)
    assert all(path.is_relative_to(STAGING) for path in cell.values())
    assert not plan["helper"].is_relative_to(REPO)


def test_sm120_bundle_protected_output_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ProtectedPathError):
        sm120._bundle_output_plan(
            candidates=[], output_root=PROTECTED / "sm120",
        )
    assert calls == []


def test_roster_resolver_returns_read_only_authority_path() -> None:
    path = resolver.resolve_cell_root(
        slug="layernorm_fwd_bf16_sm80",
        shape_point="timing_shape",
        sku="rtx_3090_ti",
        arch_dir="sm_86",
    )
    assert is_protected_path(path, repo_root=REPO)
    assert "write" not in resolver.__dict__


def test_roster_resolver_invalid_fallback_has_zero_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _arm_zero_mutation(monkeypatch)
    with pytest.raises(ValueError):
        resolver.resolve_cell_root(
            slug="not_in_roster",
            shape_point="not_a_shape",
            sku="rtx_3090_ti",
            arch_dir="sm_86",
            fallback_source="not_a_source",
        )
    assert calls == []
