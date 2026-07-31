from __future__ import annotations

import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

import phase_inject_orchestrator as phase  # noqa: E402
import run_suite as suite  # noqa: E402
from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ImmutableArtifactResourceError,
    PinnedDirectoryReader,
    inspect_immutable_directory_generation,
    open_retained_executable,
)


@contextmanager
def _ab_inputs(
    tmp_path: Path,
    *,
    expected_launch_indices: tuple[int, ...] = (0,),
    repeat_count: int = 1,
    args: list[str] | None = None,
):
    executable = tmp_path / "target"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer.so"
    tracer.write_bytes(b"tracer fixture")
    traces = tmp_path / "traces"
    traces.mkdir()
    tracer_descriptor = os.open(tracer, os.O_RDONLY)
    traces_descriptor = os.open(traces, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with open_retained_executable(executable) as authority:
            yield (
                phase.AbOverheadInputs(
                    executable_authority=authority,
                    args=[] if args is None else args,
                    tracer_so_path=tracer,
                    tracer_descriptor=tracer_descriptor,
                    traces_folder=traces,
                    expected_launch_indices=expected_launch_indices,
                    traces_descriptor=traces_descriptor,
                    repeat_count=repeat_count,
                ),
                authority,
                executable,
                traces,
            )
    finally:
        os.close(traces_descriptor)
        os.close(tracer_descriptor)


def test_phase_ledger_emission_rejects_zero_evidence(tmp_path: Path) -> None:
    with pytest.raises(
        phase.PhaseLedgerEvidenceMissing,
        match="phase_inject_ledger_evidence_missing",
    ):
        phase.emit_per_launch_phase_ledgers(
            traces_folder=tmp_path / "traces",
            ledger_output_dir=tmp_path / "ledgers",
            attempt_id="attempt-empty",
            expected_repeat_count=1,
            expected_launch_indices=(0,),
            slug="fixture",
            shape_point="timing_shape",
            kernel_signature_sha256="1" * 64,
            binary_sha256="2" * 64,
            nvbit_tracer_build_hash="3" * 64,
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


def test_phase_ledger_emission_retains_but_rejects_partial_evidence(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    partial = traces / ".partial-phase-records_0_0.bin"
    partial.write_bytes(b"retained incomplete flush bytes")

    with pytest.raises(
        phase.PhaseRecordEvidenceInvalid,
        match="phase_inject_record_evidence_invalid",
    ):
        phase.emit_per_launch_phase_ledgers(
            traces_folder=traces,
            ledger_output_dir=tmp_path / "ledgers",
            attempt_id="attempt-partial-only",
            expected_repeat_count=1,
            expected_launch_indices=(0,),
            slug="fixture",
            shape_point="timing_shape",
            kernel_signature_sha256="1" * 64,
            binary_sha256="2" * 64,
            nvbit_tracer_build_hash="3" * 64,
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

    assert partial.read_bytes() == b"retained incomplete flush bytes"
    assert not (tmp_path / "ledgers" / "attempt-partial-only").exists()


def test_phase_ab_rejects_missing_retained_authority_before_subprocess(
    tmp_path: Path,
) -> None:
    inputs = phase.AbOverheadInputs(
        executable_authority=None,  # type: ignore[arg-type]
        args=[],
        tracer_so_path=tmp_path / "missing-tracer.so",
        tracer_descriptor=-1,
        traces_folder=tmp_path / "traces",
        expected_launch_indices=(0,),
        repeat_count=1,
    )
    with pytest.raises(
        ImmutableArtifactError,
        match="phase executable authority must be a RetainedExecutable",
    ):
        phase.run_ab_overhead_pair(inputs, admission_threshold_pct=5.0)


def test_phase_ab_executes_and_injects_retained_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "target"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer.so"
    tracer.write_bytes(b"tracer fixture")
    traces = tmp_path / "traces"
    traces.mkdir()
    tracer_descriptor = os.open(tracer, os.O_RDONLY)
    traces_descriptor = os.open(traces, os.O_RDONLY | os.O_DIRECTORY)
    calls: list[tuple[list[str], dict[str, object]]] = []
    timer_values = iter((0, 100, 100, 205))

    def fake_runtime_env(
        path: Path,
        *,
        base_env: dict[str, str] | None,
        retained_descriptor: int,
    ) -> dict[str, str]:
        assert path == tracer
        assert base_env == {"FIXTURE": "1"}
        assert retained_descriptor == tracer_descriptor
        return {"FIXTURE": "1"}

    def fake_runner(command: list[str], **kwargs):
        calls.append((command, kwargs))
        invocation_descriptor = int(command[0].rsplit("/", 1)[-1])
        invoked = os.fstat(invocation_descriptor)
        admitted = os.fstat(executable_authority.descriptor)
        assert (invoked.st_dev, invoked.st_ino) == (
            admitted.st_dev,
            admitted.st_ino,
        )
        env = kwargs["env"]
        if env.get("SIMBENCH_NVBIT_PHASE_INJECT") == "1":
            repeat_index = env[phase.PHASE_REPEAT_INDEX_ENV]
            (
                Path(env["TRACES_FOLDER"])
                / f"phase_records_{repeat_index}_0.bin"
            ).write_bytes(b"phase record fixture")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(phase, "tracer_runtime_env", fake_runtime_env)
    try:
        with open_retained_executable(executable) as executable_authority:
            result = phase.run_ab_overhead_pair(
                phase.AbOverheadInputs(
                    executable_authority=executable_authority,
                    args=["--fixture"],
                    tracer_so_path=tracer,
                    tracer_descriptor=tracer_descriptor,
                    traces_folder=traces,
                    expected_launch_indices=(0,),
                    traces_descriptor=traces_descriptor,
                    repeat_count=1,
                    base_env={"FIXTURE": "1"},
                ),
                admission_threshold_pct=5.0,
                subprocess_runner=fake_runner,
                timer_ns=lambda: next(timer_values),
            )
    finally:
        os.close(traces_descriptor)
        os.close(tracer_descriptor)

    assert result["overhead_pct"] == 5.0
    assert len(calls) == 2
    expected_tracer = f"/proc/self/fd/{tracer_descriptor}"
    expected_traces = f"/proc/self/fd/{traces_descriptor}"
    for command, kwargs in calls:
        invocation_descriptor = int(command[0].rsplit("/", 1)[-1])
        assert command == [
            f"/proc/self/fd/{invocation_descriptor}",
            "--fixture",
        ]
        assert kwargs["pass_fds"] == (
            invocation_descriptor,
            tracer_descriptor,
            traces_descriptor,
        )
        assert kwargs["text"] is False
        assert kwargs["env"]["CUDA_INJECTION64_PATH"] == expected_tracer
        assert kwargs["env"]["TRACES_FOLDER"] == expected_traces
    assert "SIMBENCH_NVBIT_PHASE_INJECT" not in calls[0][1]["env"]
    assert calls[0][1]["env"][phase.PHASE_REPEAT_INDEX_ENV] == "0"
    assert calls[1][1]["env"]["SIMBENCH_NVBIT_PHASE_INJECT"] == "1"
    assert calls[1][1]["env"][phase.PHASE_REPEAT_INDEX_ENV] == "0"


@pytest.mark.parametrize("repeat_count", (0, -1))
def test_phase_ab_rejects_nonpositive_repeat_before_capture(
    tmp_path: Path,
    repeat_count: int,
) -> None:
    inputs = phase.AbOverheadInputs(
        executable_authority=None,  # type: ignore[arg-type]
        args=[],
        tracer_so_path=tmp_path / "missing-tracer",
        tracer_descriptor=-1,
        traces_folder=tmp_path / "missing-traces",
        expected_launch_indices=(0,),
        repeat_count=repeat_count,
    )

    with pytest.raises(
        phase.PhaseInjectAbRepeatCountInvalid,
        match="phase_inject_ab_repeat_count_invalid",
    ):
        phase.run_ab_overhead_pair(inputs, admission_threshold_pct=5.0)


def test_phase_ab_assigns_distinct_repeat_identity_to_every_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "target"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer.so"
    tracer.write_bytes(b"tracer fixture")
    traces = tmp_path / "traces"
    traces.mkdir()
    tracer_descriptor = os.open(tracer, os.O_RDONLY)
    traces_descriptor = os.open(traces, os.O_RDONLY | os.O_DIRECTORY)
    injected_repeats: list[str] = []
    timer_values = iter(range(12))

    def fake_runner(command: list[str], **kwargs):
        invocation_descriptor = int(command[0].rsplit("/", 1)[-1])
        invoked = os.fstat(invocation_descriptor)
        admitted = os.fstat(executable_authority.descriptor)
        assert (invoked.st_dev, invoked.st_ino) == (
            admitted.st_dev,
            admitted.st_ino,
        )
        env = kwargs["env"]
        if env.get("SIMBENCH_NVBIT_PHASE_INJECT") == "1":
            repeat_index = env[phase.PHASE_REPEAT_INDEX_ENV]
            injected_repeats.append(repeat_index)
            trace_root = Path(env["TRACES_FOLDER"])
            (trace_root / f"phase_records_{repeat_index}_0.bin").write_bytes(
                f"repeat {repeat_index}".encode()
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )
    try:
        with open_retained_executable(executable) as executable_authority:
            result = phase.run_ab_overhead_pair(
                phase.AbOverheadInputs(
                    executable_authority=executable_authority,
                    args=[],
                    tracer_so_path=tracer,
                    tracer_descriptor=tracer_descriptor,
                    traces_folder=traces,
                    expected_launch_indices=(0,),
                    traces_descriptor=traces_descriptor,
                    repeat_count=3,
                ),
                admission_threshold_pct=5.0,
                subprocess_runner=fake_runner,
                timer_ns=lambda: next(timer_values),
            )
    finally:
        os.close(traces_descriptor)
        os.close(tracer_descriptor)

    assert result["overhead_pct"] == 0.0
    assert injected_repeats == ["0", "1", "2"]
    assert {
        path.name: path.read_bytes() for path in sorted(traces.iterdir())
    } == {
        "phase_records_0_0.bin": b"repeat 0",
        "phase_records_1_0.bin": b"repeat 1",
        "phase_records_2_0.bin": b"repeat 2",
    }


def test_phase_ab_rejects_stale_repeat_file_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "target"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    tracer = tmp_path / "tracer.so"
    tracer.write_bytes(b"tracer fixture")
    traces = tmp_path / "traces"
    traces.mkdir()
    stale = traces / "phase_records_1_0.bin"
    stale.write_bytes(b"stale repeat one")
    tracer_descriptor = os.open(tracer, os.O_RDONLY)
    traces_descriptor = os.open(traces, os.O_RDONLY | os.O_DIRECTORY)
    called = False

    def unexpected_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("stale phase evidence must refuse before subprocess")

    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )
    try:
        with open_retained_executable(executable) as executable_authority:
            with pytest.raises(
                phase.PhaseRecordSourceNotFresh,
                match="source is not empty before the A/B attempt",
            ):
                phase.run_ab_overhead_pair(
                    phase.AbOverheadInputs(
                        executable_authority=executable_authority,
                        args=[],
                        tracer_so_path=tracer,
                        tracer_descriptor=tracer_descriptor,
                        traces_folder=traces,
                        expected_launch_indices=(0,),
                        traces_descriptor=traces_descriptor,
                        repeat_count=2,
                    ),
                    admission_threshold_pct=5.0,
                    subprocess_runner=unexpected_runner,
                )
    finally:
        os.close(traces_descriptor)
        os.close(tracer_descriptor)

    assert called is False
    assert stale.read_bytes() == b"stale repeat one"


@pytest.mark.parametrize(
    "mutation", ("mutate", "replace", "close_and_reuse")
)
def test_phase_ab_executable_authority_sabotage_fails_after_one_exact_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )
    calls = 0

    def hostile_runner(command: list[str], **_kwargs):
        nonlocal calls
        calls += 1
        invocation_descriptor = int(command[0].rsplit("/", 1)[-1])
        invoked = os.fstat(invocation_descriptor)
        admitted = os.fstat(authority.descriptor)
        assert (invoked.st_dev, invoked.st_ino) == (
            admitted.st_dev,
            admitted.st_ino,
        )
        if mutation == "mutate":
            executable.write_bytes(b"#!/bin/sh\nexit 7\n")
        elif mutation == "replace":
            executable.rename(tmp_path / "retained-original")
            executable.write_bytes(b"#!/bin/sh\nexit 8\n")
            executable.chmod(0o755)
        else:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"#!/bin/sh\nexit 9\n")
            replacement.chmod(0o755)
            admitted_descriptor = authority.descriptor
            os.close(admitted_descriptor)
            reused = os.open(replacement, os.O_RDONLY)
            assert reused == admitted_descriptor
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(
        phase.PhaseInjectExecutableAuthorityInvalid,
        match="phase_inject_executable_authority_invalid",
    ):
        with _ab_inputs(tmp_path, repeat_count=2) as fixture:
            inputs, authority, executable, _traces = fixture
            phase.run_ab_overhead_pair(
                inputs,
                admission_threshold_pct=5.0,
                subprocess_runner=hostile_runner,
                timer_ns=iter(range(20)).__next__,
            )

    assert calls == 1


def test_phase_ab_timeout_invalid_utf8_stderr_is_bounded_replacement_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )

    def timeout_runner(command: list[str], **_kwargs):
        raise subprocess.TimeoutExpired(
            command,
            7,
            stderr=b"prefix-\xff-\x80-suffix",
        )

    with _ab_inputs(tmp_path) as fixture:
        inputs, _authority, _executable, _traces = fixture
        with pytest.raises(phase.PhaseInjectAbRunTimeout) as failure:
            phase.run_ab_overhead_pair(
                inputs,
                admission_threshold_pct=5.0,
                subprocess_runner=timeout_runner,
                timeout_per_run_s=7,
            )

    assert failure.value.stderr == "prefix-�-�-suffix"
    assert failure.value.timeout_s == 7


def test_phase_ab_none_timeout_is_only_explicit_unbounded_debug_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )
    observed_kwargs: list[dict[str, object]] = []

    def runner(_command: list[str], **kwargs):
        observed_kwargs.append(kwargs)
        env = kwargs["env"]
        if env.get("SIMBENCH_NVBIT_PHASE_INJECT") == "1":
            (Path(env["TRACES_FOLDER"]) / "phase_records_0_0.bin").write_bytes(
                b"complete source fixture"
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with _ab_inputs(tmp_path) as fixture:
        inputs, _authority, _executable, _traces = fixture
        result = phase.run_ab_overhead_pair(
            inputs,
            admission_threshold_pct=5.0,
            subprocess_runner=runner,
            timer_ns=iter((0, 100, 100, 201)).__next__,
            timeout_per_run_s=None,
        )

    assert result["overhead_pct"] == 1.0
    assert len(observed_kwargs) == 2
    assert all("timeout" not in kwargs for kwargs in observed_kwargs)


def test_phase_ab_rejects_missing_admitted_launch_in_each_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )

    def runner(_command: list[str], **kwargs):
        env = kwargs["env"]
        if env.get("SIMBENCH_NVBIT_PHASE_INJECT") == "1":
            (Path(env["TRACES_FOLDER"]) / "phase_records_0_0.bin").write_bytes(
                b"launch zero only"
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with _ab_inputs(tmp_path, expected_launch_indices=(0, 1)) as fixture:
        inputs, _authority, _executable, _traces = fixture
        with pytest.raises(
            phase.PhaseRecordSourceNotFresh,
            match="exact admitted launch vector",
        ):
            phase.run_ab_overhead_pair(
                inputs,
                admission_threshold_pct=5.0,
                subprocess_runner=runner,
                timer_ns=iter((0, 1, 1, 2)).__next__,
            )


@pytest.mark.parametrize(
    "threshold",
    (
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
        "5",
        -1,
        phase.PHASE_ADMISSION_THRESHOLD_MAX_PCT + 1,
        10**1000,
    ),
)
def test_phase_ab_rejects_coercive_or_unbounded_threshold_before_authority(
    tmp_path: Path,
    threshold: object,
) -> None:
    inputs = phase.AbOverheadInputs(
        executable_authority=None,  # type: ignore[arg-type]
        args=[],
        tracer_so_path=tmp_path / "unused-tracer",
        tracer_descriptor=-1,
        traces_folder=tmp_path / "unused-traces",
        expected_launch_indices=(0,),
        repeat_count=1,
    )
    with pytest.raises(phase.PhaseInjectAdmissionThresholdInvalid):
        phase.run_ab_overhead_pair(
            inputs,
            admission_threshold_pct=threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "timeout",
    (
        True,
        0,
        -1,
        1.0,
        float("nan"),
        float("inf"),
        "5",
        phase.PHASE_TIMEOUT_MAX_S + 1,
    ),
)
def test_phase_ab_rejects_coercive_or_unbounded_timeout_before_authority(
    tmp_path: Path,
    timeout: object,
) -> None:
    inputs = phase.AbOverheadInputs(
        executable_authority=None,  # type: ignore[arg-type]
        args=[],
        tracer_so_path=tmp_path / "unused-tracer",
        tracer_descriptor=-1,
        traces_folder=tmp_path / "unused-traces",
        expected_launch_indices=(0,),
        repeat_count=1,
    )
    with pytest.raises(phase.PhaseInjectTimeoutInvalid):
        phase.run_ab_overhead_pair(
            inputs,
            admission_threshold_pct=5.0,
            timeout_per_run_s=timeout,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("repeat_count", "launch_indices", "args", "resource"),
    (
        (
            phase.PHASE_AB_MAX_REPEAT_COUNT + 1,
            (0,),
            [],
            "A/B repeat count",
        ),
        (
            1,
            tuple(range(phase.PHASE_AB_MAX_LAUNCH_COUNT + 1)),
            [],
            "admitted phase launch count",
        ),
        (
            3,
            tuple(range(phase.PHASE_AB_MAX_MATRIX_CELLS // 3 + 1)),
            [],
            "phase repeat-by-launch matrix cells",
        ),
        (
            1,
            (0,),
            ["x"] * (phase.PHASE_AB_MAX_ARGUMENT_COUNT + 1),
            "A/B argument count",
        ),
        (
            1,
            (0,),
            ["x" * phase.PHASE_AB_MAX_ARGUMENT_BYTES],
            "A/B aggregate argument bytes",
        ),
    ),
)
def test_phase_ab_work_caps_fail_before_executable_or_subprocess(
    tmp_path: Path,
    repeat_count: int,
    launch_indices: tuple[int, ...],
    args: list[str],
    resource: str,
) -> None:
    inputs = phase.AbOverheadInputs(
        executable_authority=None,  # type: ignore[arg-type]
        args=args,
        tracer_so_path=tmp_path / "unused-tracer",
        tracer_descriptor=-1,
        traces_folder=tmp_path / "unused-traces",
        expected_launch_indices=launch_indices,
        repeat_count=repeat_count,
    )
    with pytest.raises(phase.PhaseInjectResourceLimitExceeded) as failure:
        phase.run_ab_overhead_pair(inputs, admission_threshold_pct=5.0)

    assert failure.value.resource == resource
    assert not (tmp_path / "unused-traces").exists()


def test_phase_ab_rejects_noncontiguous_launch_authority_before_side_effects(
    tmp_path: Path,
) -> None:
    inputs = phase.AbOverheadInputs(
        executable_authority=None,  # type: ignore[arg-type]
        args=[],
        tracer_so_path=tmp_path / "unused-tracer",
        tracer_descriptor=-1,
        traces_folder=tmp_path / "unused-traces",
        expected_launch_indices=(0, 2),
        repeat_count=1,
    )
    with pytest.raises(phase.PhaseInjectLaunchAuthorityInvalid):
        phase.run_ab_overhead_pair(inputs, admission_threshold_pct=5.0)


def test_phase_ab_custom_runner_output_is_bounded_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )
    calls = 0

    def noisy_runner(_command: list[str], **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=0,
            stdout=b"x" * (phase.PHASE_AB_MAX_CAPTURE_BYTES + 1),
            stderr=b"invalid-\xff",
        )

    with _ab_inputs(tmp_path) as fixture:
        inputs, _authority, _executable, _traces = fixture
        with pytest.raises(phase.PhaseInjectAbOutputLimitExceeded):
            phase.run_ab_overhead_pair(
                inputs,
                admission_threshold_pct=5.0,
                subprocess_runner=noisy_runner,
                timer_ns=iter((0, 1)).__next__,
            )

    assert calls == 1


def test_phase_ab_timeout_stderr_replacement_output_has_exact_byte_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {},
    )

    def timeout_runner(command: list[str], **_kwargs):
        raise subprocess.TimeoutExpired(
            command,
            2,
            stderr=b"\xff" * (phase.PHASE_AB_MAX_DIAGNOSTIC_BYTES + 100),
        )

    with _ab_inputs(tmp_path) as fixture:
        inputs, _authority, _executable, _traces = fixture
        with pytest.raises(phase.PhaseInjectAbRunTimeout) as failure:
            phase.run_ab_overhead_pair(
                inputs,
                admission_threshold_pct=5.0,
                subprocess_runner=timeout_runner,
                timeout_per_run_s=2,
            )

    encoded = failure.value.stderr.encode("utf-8")
    assert len(encoded) <= phase.PHASE_AB_MAX_DIAGNOSTIC_BYTES
    assert failure.value.stderr.endswith("\n...[truncated]")
    assert "�" in failure.value.stderr


def test_phase_ab_environment_bytes_fail_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: {
            "OVERSIZED": "x" * phase.PHASE_AB_MAX_ENVIRONMENT_BYTES
        },
    )
    called = False

    def unexpected_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("environment limit must precede subprocess")

    with _ab_inputs(tmp_path) as fixture:
        inputs, _authority, _executable, _traces = fixture
        with pytest.raises(phase.PhaseInjectResourceLimitExceeded) as failure:
            phase.run_ab_overhead_pair(
                inputs,
                admission_threshold_pct=5.0,
                subprocess_runner=unexpected_runner,
            )

    assert failure.value.resource == "A/B aggregate environment bytes"
    assert called is False


def test_default_phase_subprocess_runner_bounds_output_and_reaps() -> None:
    with pytest.raises(phase.PhaseInjectAbOutputLimitExceeded):
        phase._bounded_subprocess_runner(
            [
                sys.executable,
                "-c",
                (
                    "import os; os.write(1, b'x' * "
                    f"{phase.PHASE_AB_MAX_CAPTURE_BYTES + 1})"
                ),
            ],
            capture_output=True,
            text=False,
            env=dict(os.environ),
            pass_fds=(),
            timeout=10,
        )


def test_tracer_phase_record_filename_uses_repeat_environment(
    tmp_path: Path,
) -> None:
    tracer_source = (
        REPO_ROOT
        / "tools"
        / "nvbit_tracer"
        / "tracer_tool"
        / "tracer_tool.cu"
    ).read_text(encoding="utf-8")
    executable_source = re.sub(
        r"//[^\n]*|/\*.*?\*/",
        "",
        tracer_source,
        flags=re.DOTALL,
    )
    assert 'std::getenv("SIMBENCH_NVBIT_PHASE_REPEAT_INDEX")' in (
        executable_source
    )
    assert "phase_record_writer.WriteRecord(" not in executable_source
    assert "simbench_nvbit::PublishCanonicalRecordOrFail(" in executable_source
    assert re.search(
        r"phase_record_writer\.AdmitDevice\(\s*device_id,",
        executable_source,
    )
    assert executable_source.count("phase_record_writer.OpenDirectory(") == 1
    assert executable_source.index("phase_record_writer.OpenDirectory(") < (
        executable_source.index("void nvbit_at_cuda_event(")
    )

    begin_launch = executable_source.index("phase_record_writer.BeginLaunch(")
    managed_identity_write = executable_source.index(
        "simbench_phase_launch_index_for_inject = this_launch_id;"
    )
    assert begin_launch < managed_identity_write
    assert re.search(
        r"const\s+uint32_t\s+this_launch_id\s*=\s*"
        r"pending_accepted_launch\.trace_launch_id\s*;",
        executable_source,
    )
    assert re.search(
        r"const\s+uint32_t\s+phase_this_launch_id\s*=\s*"
        r"pending_accepted_launch\.trace_launch_id\s*;",
        executable_source,
    )
    assert "phase_record_writer.EndLaunch(" in executable_source

    timeout_branch = executable_source.index("if (phase_flush_timed_out)")
    partial_write = executable_source.index(
        "phase_record_writer.WritePartialRecord(", timeout_branch
    )
    timeout_fatal = executable_source.index(
        '"phase-channel flush timeout"', partial_write
    )
    canonical_write = executable_source.index(
        "simbench_nvbit::PublishCanonicalRecordOrFail(", timeout_fatal
    )
    assert timeout_branch < partial_write < timeout_fatal < canonical_write
    assert ".partial-phase-records_<repeat>_<launch>.bin" in executable_source

    assert re.search(
        r"struct\s+CTXstate\s*\{.*?int\s+device_id\s*=\s*-1\s*;",
        executable_source,
        flags=re.DOTALL,
    )
    assert executable_source.count("cuCtxGetDevice(") == 1
    assert re.search(
        r"const\s+CUresult\s+status\s*=\s*cuCtxGetDevice\("
        r"&observed_device\)\s*;.*?status\s*!=\s*CUDA_SUCCESS",
        executable_source,
        flags=re.DOTALL,
    )
    assert "checked_stored_context_device_id(" in executable_source
    assert re.search(
        r"const\s+cudaError_t\s+count_status\s*=\s*"
        r"cudaGetDeviceCount\(&num_devices\)\s*;",
        executable_source,
    )
    makefile = (
        REPO_ROOT
        / "tools"
        / "nvbit_tracer"
        / "tracer_tool"
        / "Makefile"
    ).read_text(encoding="utf-8")
    assert re.search(
        r"(?m)^tracer_tool\.o:.*\bphase_record_writer\.h\b.*"
        r"\bphase_record_path\.h\b",
        makefile,
    )
    assert re.search(
        r"(?m)^HOST_OBJECTS=.*\bphase_record_writer\.o\b",
        makefile,
    )

    compiler = shutil.which("c++")
    assert compiler is not None, "host C++ compiler is required for tracer helper test"
    source = tmp_path / "phase_record_path_test.cc"
    binary = tmp_path / "phase_record_path_test"
    source.write_text(
        """
#include <iostream>
#include "phase_record_path.h"

int main() {
  char output[256];
  if (!simbench_nvbit::format_phase_record_path(
          output, sizeof(output), "/tmp/trace root", 12, 34)) {
    return 1;
  }
  std::cout << output << "\\n";
  if (!simbench_nvbit::format_partial_phase_record_leaf(
          output, sizeof(output), 12, 34)) {
    return 2;
  }
  std::cout << output << "\\n";
  char bounded[8];
  if (simbench_nvbit::format_phase_record_path(
          bounded, sizeof(bounded), "/tmp/trace root", 12, 34)) {
    return 3;
  }
  return 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(
                REPO_ROOT
                / "tools"
                / "nvbit_tracer"
                / "tracer_tool"
            ),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "/tmp/trace root/phase_records_12_34.bin\n"
        ".partial-phase-records_12_34.bin\n"
    )


def test_tracer_callback_current_device_query_is_branch_local() -> None:
    tracer_source = (
        REPO_ROOT
        / "tools"
        / "nvbit_tracer"
        / "tracer_tool"
        / "tracer_tool.cu"
    ).read_text(encoding="utf-8")
    executable_source = re.sub(
        r"//[^\n]*|/\*.*?\*/",
        "",
        tracer_source,
        flags=re.DOTALL,
    )

    callback_start = executable_source.index("void nvbit_at_cuda_event(")
    memcpy_dispatch = executable_source.index(
        "if (cbid == API_CUDA_cuMemcpyHtoD_v2)", callback_start
    )
    callback_preamble = executable_source[callback_start:memcpy_dispatch]
    assert "checked_current_context_device_id(" not in callback_preamble
    assert "cuCtxGetDevice(" not in callback_preamble

    launch_dispatch = executable_source.index(
        "cbid == API_CUDA_cuLaunchKernel_ptsz", memcpy_dispatch
    )
    memcpy_branch = executable_source[memcpy_dispatch:launch_dispatch]
    assert "checked_current_context_device_id(" in memcpy_branch

    allocation_dispatch = executable_source.index(
        "cbid == API_CUDA_cuMemAlloc_v2", launch_dispatch
    )
    launch_branch = executable_source[launch_dispatch:allocation_dispatch]
    assert "checked_current_context_device_id(" in launch_branch
    assert "const int launch_device_id" in launch_branch

    explicit_sync = executable_source.index(
        "cbid == API_CUDA_cuCtxSynchronize_v2", allocation_dispatch
    )
    address_reserve = executable_source.index(
        "cbid == API_CUDA_cuMemAddressReserve", explicit_sync
    )
    assert "checked_current_context_device_id(" not in (
        executable_source[explicit_sync:address_reserve]
    )
    assert "checked_stored_context_device_id(" in (
        executable_source[allocation_dispatch:explicit_sync]
    )
    assert "const int context_device_id" not in executable_source[
        callback_start:
    ]


def _valid_phase_record_payload(*, launch_index: int = 0) -> bytes:
    def record(clock_role: int, clock64: int) -> bytes:
        return struct.pack(
            phase._agg.PHASE_CLOCK_RECORD_STRUCT_FORMAT,
            launch_index,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            clock_role,
            clock64,
            0xFFFFFFFF,
        )

    return record(phase._agg.ClockRole.phase_open.value, 100) + record(
        phase._agg.ClockRole.phase_close.value, 110
    )


def _valid_phase_ledger() -> dict[str, object]:
    records = [
        phase._agg.PhaseClockRecord(
            launch_index=0,
            cta_id_x=0,
            cta_id_y=0,
            cta_id_z=0,
            sm_id=0,
            warpid_tb=0,
            phase_id=0,
            phase_kind=0,
            clock_role=phase._agg.ClockRole.phase_open.value,
            clock64=100,
            active_mask=0xFFFFFFFF,
        ),
        phase._agg.PhaseClockRecord(
            launch_index=0,
            cta_id_x=0,
            cta_id_y=0,
            cta_id_z=0,
            sm_id=0,
            warpid_tb=0,
            phase_id=0,
            phase_kind=0,
            clock_role=phase._agg.ClockRole.phase_close.value,
            clock64=110,
            active_mask=0xFFFFFFFF,
        ),
    ]
    return phase._agg.build_phase_cycle_ledger(
        records,
        slug="fixture",
        shape_point="timing_shape",
        repeat_index=0,
        launch_index=0,
        kernel_signature_sha256="1" * 64,
        binary_sha256="2" * 64,
        nvbit_tracer_build_hash="3" * 64,
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


def test_phase_ledger_streaming_serializer_preserves_canonical_bytes() -> None:
    ledger = _valid_phase_ledger()
    definitions = ledger["phase_definitions"]
    assert isinstance(definitions, list)
    assert isinstance(definitions[0], dict)
    definitions[0]["source_location_hint"] = 'line\n"\\é😀\x00'
    expected = (
        json.dumps(ledger, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")

    assert phase._agg.serialize_phase_cycle_ledger(ledger) == expected


def test_phase_ledger_string_size_preflight_matches_canonical_encoder() -> None:
    for value in (
        "",
        'ascii\n"\\',
        "\x00\x1f\x7f",
        "éΩ",
        "𐏿",
        "😀",
    ):
        assert phase._agg._ascii_json_string_size(value) == len(
            json.dumps(value, ensure_ascii=True).encode("utf-8")
        )


def test_phase_ledger_streaming_cap_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _valid_phase_ledger()
    expected = (
        json.dumps(ledger, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    monkeypatch.setattr(
        phase._agg,
        "PHASE_LEDGER_MAX_BYTES",
        len(expected),
    )
    assert phase._agg.serialize_phase_cycle_ledger(ledger) == expected

    monkeypatch.setattr(
        phase._agg,
        "PHASE_LEDGER_MAX_BYTES",
        len(expected) - 1,
    )
    with pytest.raises(
        phase._agg.PhaseLedgerSerializationLimitExceeded,
        match="phase_ledger_serialization_limit_exceeded",
    ) as failure:
        phase._agg.serialize_phase_cycle_ledger(ledger)
    assert failure.value.max_bytes == len(expected) - 1
    assert failure.value.attempted_size_bytes_at_least == len(expected)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("overhead_pct", float("nan")),
        ("overhead_pct", float("inf")),
        ("overhead_pct", float("-inf")),
        ("admission_threshold_pct", float("nan")),
        ("admission_threshold_pct", float("inf")),
        ("admission_threshold_pct", float("-inf")),
    ),
)
def test_phase_ledger_streaming_serializer_rejects_every_nonfinite_number(
    field: str,
    value: float,
) -> None:
    ledger = _valid_phase_ledger()
    overhead = ledger["ab_overhead_record"]
    assert isinstance(overhead, dict)
    overhead[field] = value

    with pytest.raises(
        phase._agg.PhaseLedgerCanonicalSerializationInvalid,
        match="phase_ledger_canonical_serialization_invalid",
    ) as failure:
        phase._agg.serialize_phase_cycle_ledger(ledger)

    assert isinstance(failure.value, phase._agg.PhaseLedgerSchemaInvalid)
    assert failure.value.json_path == f"$.ab_overhead_record.{field}"


def test_phase_ledger_serialization_limit_precedes_compat_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _valid_phase_ledger()
    definitions = ledger["phase_definitions"]
    assert isinstance(definitions, list)
    assert isinstance(definitions[0], dict)
    definitions[0]["source_location_hint"] = "x" * 4096
    output = tmp_path / "phase_cycle_ledger_0_0.json"
    monkeypatch.setattr(phase._agg, "PHASE_LEDGER_MAX_BYTES", 2048)

    with pytest.raises(
        phase._agg.PhaseLedgerSerializationLimitExceeded,
        match="phase_ledger_serialization_limit_exceeded",
    ) as failure:
        phase._agg.emit_phase_cycle_ledger(ledger, output)

    assert isinstance(failure.value, phase._agg.PhaseLedgerSchemaInvalid)
    assert failure.value.max_bytes == 2048
    assert failure.value.attempted_size_bytes_at_least > 2048
    assert not output.exists()
    assert not (tmp_path / f".{output.name}.partial").exists()


def test_phase_ledger_compat_emitter_publishes_canonical_bytes(
    tmp_path: Path,
) -> None:
    ledger = _valid_phase_ledger()
    output = tmp_path / "phase_cycle_ledger_0_0.json"
    expected = phase._agg.serialize_phase_cycle_ledger(ledger)

    assert phase._agg.emit_phase_cycle_ledger(ledger, output) is None
    assert output.read_bytes() == expected
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


@pytest.mark.parametrize(
    "node_kind", ("regular", "directory", "symlink", "hardlink", "fifo")
)
def test_phase_ledger_compat_emitter_preserves_preexisting_destination(
    tmp_path: Path,
    node_kind: str,
) -> None:
    ledger = _valid_phase_ledger()
    output = tmp_path / "phase_cycle_ledger_0_0.json"
    target = tmp_path / "retained-target"
    target.write_bytes(b"retained-target-bytes")
    if node_kind == "regular":
        output.write_bytes(b"unique-prior-ledger-evidence")
    elif node_kind == "directory":
        output.mkdir()
        (output / "marker").write_bytes(b"directory-evidence")
    elif node_kind == "symlink":
        output.symlink_to(target)
    elif node_kind == "hardlink":
        os.link(target, output)
    else:
        os.mkfifo(output)

    try:
        with pytest.raises(ImmutableArtifactError, match="already exists"):
            phase._agg.emit_phase_cycle_ledger(ledger, output)

        if node_kind == "regular":
            assert output.read_bytes() == b"unique-prior-ledger-evidence"
        elif node_kind == "directory":
            assert (output / "marker").read_bytes() == b"directory-evidence"
        elif node_kind == "symlink":
            assert output.is_symlink()
            assert target.read_bytes() == b"retained-target-bytes"
        elif node_kind == "hardlink":
            assert output.stat().st_ino == target.stat().st_ino
            assert target.read_bytes() == b"retained-target-bytes"
        else:
            assert stat.S_ISFIFO(output.lstat().st_mode)
    finally:
        if node_kind == "fifo":
            output.unlink()


def test_phase_ledger_invalid_body_precedes_existing_destination_failure(
    tmp_path: Path,
) -> None:
    ledger = _valid_phase_ledger()
    overhead = ledger["ab_overhead_record"]
    assert isinstance(overhead, dict)
    overhead["overhead_pct"] = float("nan")
    output = tmp_path / "phase_cycle_ledger_0_0.json"
    output.write_bytes(b"unique-prior-ledger-evidence")

    with pytest.raises(phase._agg.PhaseLedgerCanonicalSerializationInvalid):
        phase._agg.emit_phase_cycle_ledger(ledger, output)

    assert output.read_bytes() == b"unique-prior-ledger-evidence"


def _phase_emission_contract_kwargs(
    tmp_path: Path,
    traces: Path,
    *,
    attempt_id: str,
) -> dict[str, object]:
    return {
        "traces_folder": traces,
        "ledger_output_dir": tmp_path / "ledger-generations",
        "attempt_id": attempt_id,
        "expected_repeat_count": 1,
        "expected_launch_indices": (0,),
        "slug": "fixture",
        "shape_point": "timing_shape",
        "kernel_signature_sha256": "1" * 64,
        "binary_sha256": "2" * 64,
        "nvbit_tracer_build_hash": "3" * 64,
        "claim_scope": "measured_local",
        "ab_overhead_record": {
            "t_wall_no_inject_ns": 100,
            "t_wall_with_inject_ns": 101,
            "overhead_pct": 1.0,
            "admission_threshold_pct": 5.0,
            "admitted": True,
        },
        "dual_run_manifest_ref": "dual_run_manifest.json",
    }


def _phase_emission_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    attempt_id: str,
) -> dict[str, object]:
    traces = tmp_path / "traces"
    traces.mkdir(exist_ok=True)
    (traces / "phase_records_0_0.bin").write_bytes(b"phase-record-fixture")
    monkeypatch.setattr(
        phase._agg,
        "decode_phase_clock_records",
        lambda _payload: [SimpleNamespace(launch_index=0)],
    )

    def build(_records, **kwargs):
        return {
            "schema_version": 1,
            "slug": kwargs["slug"],
            "shape_point": kwargs["shape_point"],
            "repeat_index": kwargs["repeat_index"],
            "launch_index": kwargs["launch_index"],
        }

    monkeypatch.setattr(phase._agg, "build_phase_cycle_ledger", build)
    monkeypatch.setattr(
        phase._agg,
        "serialize_phase_cycle_ledger",
        lambda ledger: (json.dumps(ledger, sort_keys=True) + "\n").encode(),
    )
    monkeypatch.setattr(
        phase._consumer,
        "admit_phase_cycle_ledger",
        lambda *_args, **_kwargs: None,
    )
    return _phase_emission_contract_kwargs(
        tmp_path,
        traces,
        attempt_id=attempt_id,
    )


@pytest.mark.parametrize(
    "relative_path",
    (
        "phase_records_00_0.bin",
        "phase_records_0_00.bin",
        "nested/phase_records_0_0.bin",
        "planted-phase_records_0_0.bin",
        ".partial-phase-records_00_0.bin",
    ),
)
def test_phase_ledger_rejects_every_malformed_phase_record_name(
    tmp_path: Path,
    relative_path: str,
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    planted = traces / relative_path
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(b"retained malformed evidence")
    kwargs = _phase_emission_contract_kwargs(
        tmp_path,
        traces,
        attempt_id="attempt-malformed-name",
    )

    with pytest.raises(
        phase.PhaseRecordEvidenceInvalid,
        match="phase_inject_record_evidence_invalid",
    ):
        phase.emit_per_launch_phase_ledgers(**kwargs)

    assert planted.read_bytes() == b"retained malformed evidence"
    assert not (
        Path(kwargs["ledger_output_dir"]) / str(kwargs["attempt_id"])
    ).exists()


def test_phase_ledger_rejects_empty_canonical_record_file(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    source = traces / "phase_records_0_0.bin"
    source.write_bytes(b"")
    kwargs = _phase_emission_contract_kwargs(
        tmp_path,
        traces,
        attempt_id="attempt-empty-canonical",
    )

    with pytest.raises(
        phase.PhaseRecordPayloadInvalid,
        match="canonical payload is empty",
    ):
        phase.emit_per_launch_phase_ledgers(**kwargs)

    assert source.exists() and source.stat().st_size == 0
    assert not (
        Path(kwargs["ledger_output_dir"]) / str(kwargs["attempt_id"])
    ).exists()


@pytest.mark.parametrize("node_kind", ("directory", "symlink", "hardlink", "fifo"))
def test_phase_ledger_rejects_special_or_planted_link_authority(
    tmp_path: Path,
    node_kind: str,
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    source = traces / "phase_records_0_0.bin"
    target = tmp_path / "retained-target"
    target.write_bytes(b"retained target bytes")
    if node_kind == "directory":
        source.mkdir()
    elif node_kind == "symlink":
        source.symlink_to(target)
    elif node_kind == "hardlink":
        os.link(target, source)
    else:
        os.mkfifo(source)
    kwargs = _phase_emission_contract_kwargs(
        tmp_path,
        traces,
        attempt_id=f"attempt-{node_kind}-source",
    )

    try:
        with pytest.raises(ImmutableArtifactError):
            phase.emit_per_launch_phase_ledgers(**kwargs)
        assert not (
            Path(kwargs["ledger_output_dir"]) / str(kwargs["attempt_id"])
        ).exists()
        if node_kind in {"symlink", "hardlink"}:
            assert target.read_bytes() == b"retained target bytes"
    finally:
        if node_kind == "fifo":
            source.unlink()


def test_phase_ledger_rejects_decode_trailing_byte_before_stage(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    source = traces / "phase_records_0_0.bin"
    source.write_bytes(_valid_phase_record_payload() + b"\xff")
    kwargs = _phase_emission_contract_kwargs(
        tmp_path,
        traces,
        attempt_id="attempt-trailing-byte",
    )

    with pytest.raises(
        phase.PhaseRecordPayloadInvalid,
        match="complete record sequence",
    ):
        phase.emit_per_launch_phase_ledgers(**kwargs)

    assert source.read_bytes().endswith(b"\xff")
    assert not (
        Path(kwargs["ledger_output_dir"])
        / f".{kwargs['attempt_id']}.staging"
    ).exists()


def test_phase_ledger_rejects_record_payload_size_before_decode_allocation(
    tmp_path: Path,
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    source = traces / "phase_records_0_0.bin"
    with source.open("wb") as stream:
        stream.truncate(phase.PHASE_RECORD_MAX_BYTES + 1)
    kwargs = _phase_emission_contract_kwargs(
        tmp_path,
        traces,
        attempt_id="attempt-record-size-limit",
    )

    with pytest.raises(phase.PhaseInjectResourceLimitExceeded) as failure:
        phase.emit_per_launch_phase_ledgers(**kwargs)

    assert failure.value.resource.startswith("phase-record payload bytes")
    assert not (
        Path(kwargs["ledger_output_dir"])
        / f".{kwargs['attempt_id']}.staging"
    ).exists()


@pytest.mark.parametrize(
    ("expected_launch_indices", "observed_launch_indices"),
    (((0, 1), (0,)), ((0,), (0, 1))),
)
def test_phase_ledger_requires_exact_admitted_launch_vector_before_decode(
    tmp_path: Path,
    expected_launch_indices: tuple[int, ...],
    observed_launch_indices: tuple[int, ...],
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    for launch_index in observed_launch_indices:
        (traces / f"phase_records_0_{launch_index}.bin").write_bytes(
            b"matrix identity fixture"
        )
    kwargs = _phase_emission_contract_kwargs(
        tmp_path,
        traces,
        attempt_id="attempt-exact-launch-vector",
    )
    kwargs["expected_launch_indices"] = expected_launch_indices

    with pytest.raises(phase.PhaseLedgerRepeatCoverageInvalid):
        phase.emit_per_launch_phase_ledgers(**kwargs)

    assert not (
        Path(kwargs["ledger_output_dir"])
        / f".{kwargs['attempt_id']}.staging"
    ).exists()


def test_phase_ledger_rejects_noncontiguous_expected_launch_authority(
    tmp_path: Path,
) -> None:
    kwargs = _phase_emission_contract_kwargs(
        tmp_path,
        tmp_path / "unopened-traces",
        attempt_id="attempt-noncontiguous-authority",
    )
    kwargs["expected_launch_indices"] = (0, 2)

    with pytest.raises(phase.PhaseInjectLaunchAuthorityInvalid):
        phase.emit_per_launch_phase_ledgers(**kwargs)

    assert not Path(kwargs["ledger_output_dir"]).exists()


def test_phase_ledger_rejects_foreign_launch_before_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "phase_records_0_0.bin").write_bytes(
        b"mixed-launch-phase-record-fixture"
    )
    monkeypatch.setattr(
        phase._agg,
        "decode_phase_clock_records",
        lambda _payload: [
            SimpleNamespace(launch_index=0),
            SimpleNamespace(launch_index=2),
            SimpleNamespace(launch_index=1),
        ],
    )

    def unexpected_serializer(_ledger):
        raise AssertionError("mismatched launch records must not serialize")

    def unexpected_admission(*_args, **_kwargs):
        raise AssertionError("mismatched launch records must not be admitted")

    monkeypatch.setattr(
        phase._agg,
        "serialize_phase_cycle_ledger",
        unexpected_serializer,
    )
    monkeypatch.setattr(
        phase._consumer,
        "admit_phase_cycle_ledger",
        unexpected_admission,
    )

    attempt_id = "attempt-mixed-launch"
    ledger_root = tmp_path / "ledger-generations"
    with pytest.raises(
        phase._agg.PhaseRecordLaunchIndexMismatch,
        match="phase_record_launch_index_mismatch",
    ) as failure:
        phase.emit_per_launch_phase_ledgers(
            traces_folder=traces,
            ledger_output_dir=ledger_root,
            attempt_id=attempt_id,
            expected_repeat_count=1,
            expected_launch_indices=(0,),
            slug="fixture",
            shape_point="timing_shape",
            kernel_signature_sha256="1" * 64,
            binary_sha256="2" * 64,
            nvbit_tracer_build_hash="3" * 64,
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

    assert failure.value.ledger_launch_index == 0
    assert failure.value.offending_launch_indices == (1, 2)
    assert not (ledger_root / attempt_id).exists()
    assert not (ledger_root / f".{attempt_id}.staging").exists()


@pytest.mark.parametrize("mutation", ("mutate", "delete", "substitute"))
def test_phase_ledger_commit_seal_rejects_hidden_stage_sabotage_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path,
        monkeypatch,
        attempt_id=f"attempt-hidden-{mutation}",
    )
    original_commit = phase.ExclusiveDirectoryStage.commit

    def sabotage_then_commit(stage, **commit_kwargs):
        ledger = stage.staging_path / "phase_cycle_ledger_0_0.json"
        if mutation == "mutate":
            ledger.chmod(0o644)
            ledger.write_bytes(b"mutated after admission")
        elif mutation == "delete":
            ledger.unlink()
        else:
            ledger.unlink()
            ledger.write_bytes(b"substituted after admission")
        return original_commit(stage, **commit_kwargs)

    monkeypatch.setattr(
        phase.ExclusiveDirectoryStage,
        "commit",
        sabotage_then_commit,
    )

    with pytest.raises(ImmutableArtifactError):
        phase.emit_per_launch_phase_ledgers(**kwargs)

    root = Path(kwargs["ledger_output_dir"])
    assert not (root / str(kwargs["attempt_id"])).exists()
    assert (root / f".{kwargs['attempt_id']}.staging").is_dir()


def test_phase_ledger_monitor_rejects_transient_mutate_restore_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path,
        monkeypatch,
        attempt_id="attempt-hidden-transient",
    )

    def mutate_restore(stage) -> None:
        ledger = stage.staging_path / "phase_cycle_ledger_0_0.json"
        original = ledger.read_bytes()
        ledger.chmod(0o644)
        ledger.write_bytes(b"transient replacement")
        ledger.write_bytes(original)
        ledger.chmod(0o444)

    with pytest.raises(
        ImmutableArtifactError,
        match="protected input tree changed",
    ):
        phase.emit_per_launch_phase_ledgers(
            **kwargs,
            before_commit=mutate_restore,
        )

    root = Path(kwargs["ledger_output_dir"])
    assert not (root / str(kwargs["attempt_id"])).exists()
    assert (root / f".{kwargs['attempt_id']}.staging").is_dir()


def test_phase_ledger_generation_has_no_partial_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-complete"
    )
    final = Path(kwargs["ledger_output_dir"]) / "attempt-complete"
    observed_before_commit = False

    def before_commit(stage) -> None:
        nonlocal observed_before_commit
        assert not final.exists()
        assert stage.staging_path.is_dir()
        assert (
            stage.staging_path / "phase_cycle_ledger_0_0.json"
        ).is_file()
        observed_before_commit = True

    result = phase.emit_per_launch_phase_ledgers(
        **kwargs,
        before_commit=before_commit,
    )

    assert observed_before_commit
    generation = result["generation"]
    assert generation.path == final
    assert not (final.parent / ".attempt-complete.staging").exists()
    assert stat.S_IMODE(final.stat().st_mode) == 0o555
    assert stat.S_IMODE(
        (final / "phase_cycle_ledger_0_0.json").stat().st_mode
    ) == 0o444
    assert inspect_immutable_directory_generation(
        final,
        max_files=1,
        max_directories=1,
        max_total_file_bytes=generation.total_file_bytes,
        max_total_relative_path_bytes=4096,
    ) == generation


def test_phase_generation_cannot_acquire_credit_when_request_is_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path,
        monkeypatch,
        attempt_id="attempt-request-false",
    )
    generation = phase.emit_per_launch_phase_ledgers(**kwargs)["generation"]

    with pytest.raises(
        ImmutableArtifactError,
        match="requested is not exact true",
    ):
        suite._phase_evidence_identities(
            artifact_dir=tmp_path,
            shape_point_name="timing_shape",
            expected_repeat_count=1,
            expected_launch_indices=(0,),
            phase_requested=False,
            generation=generation,
        )


def test_phase_ledger_generation_binds_every_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-three-repeats"
    )
    traces = Path(kwargs["traces_folder"])
    (traces / "phase_records_1_0.bin").write_bytes(b"phase repeat one")
    (traces / "phase_records_2_0.bin").write_bytes(b"phase repeat two")
    kwargs["expected_repeat_count"] = 3

    result = phase.emit_per_launch_phase_ledgers(**kwargs)

    assert result["n_admitted"] == 3
    assert {
        identity.relative_path for identity in result["generation"].files
    } == {
        "phase_cycle_ledger_0_0.json",
        "phase_cycle_ledger_1_0.json",
        "phase_cycle_ledger_2_0.json",
    }


def test_phase_ledger_generation_enforces_shared_cap_before_next_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-cumulative-limit"
    )
    traces = Path(kwargs["traces_folder"])
    (traces / "phase_records_0_1.bin").write_bytes(b"phase launch one")
    kwargs["expected_launch_indices"] = (0, 1)
    first_ledger = {
        "schema_version": 1,
        "slug": "fixture",
        "shape_point": "timing_shape",
        "repeat_index": 0,
        "launch_index": 0,
    }
    second_ledger = dict(first_ledger, launch_index=1)
    first_payload = phase._agg.serialize_phase_cycle_ledger(first_ledger)
    second_payload = phase._agg.serialize_phase_cycle_ledger(second_ledger)
    shared_cap = len(first_payload) + len(second_payload) - 1
    assert len(first_payload) < shared_cap
    assert len(second_payload) < shared_cap
    monkeypatch.setattr(
        phase._agg,
        "PHASE_LEDGER_MAX_BYTES",
        shared_cap,
    )

    with pytest.raises(
        phase._agg.PhaseLedgerSerializationLimitExceeded,
        match="phase_ledger_serialization_limit_exceeded",
    ) as failure:
        phase.emit_per_launch_phase_ledgers(**kwargs)

    assert failure.value.max_bytes == shared_cap
    assert failure.value.attempted_size_bytes_at_least == (
        len(first_payload) + len(second_payload)
    )
    hidden = (
        Path(kwargs["ledger_output_dir"])
        / ".attempt-cumulative-limit.staging"
    )
    assert not hidden.exists()


def test_phase_ledger_rejects_missing_repeat_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-missing-repeat"
    )
    kwargs["expected_repeat_count"] = 2

    with pytest.raises(
        phase.PhaseLedgerRepeatCoverageInvalid,
        match="phase_inject_repeat_coverage_invalid",
    ):
        phase.emit_per_launch_phase_ledgers(**kwargs)

    root = Path(kwargs["ledger_output_dir"])
    assert not (root / "attempt-missing-repeat").exists()
    assert not (root / ".attempt-missing-repeat.staging").exists()


def test_phase_ledger_retains_caller_source_reader_through_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-retained-source"
    )
    traces = Path(kwargs["traces_folder"])
    observed_during_commit = False
    original_commit = phase.ExclusiveDirectoryStage.commit

    with PinnedDirectoryReader(traces) as source_reader:
        def checked_commit(stage, **commit_kwargs):
            nonlocal observed_during_commit
            source_reader.verify_path()
            observed_during_commit = True
            return original_commit(stage, **commit_kwargs)

        monkeypatch.setattr(
            phase.ExclusiveDirectoryStage,
            "commit",
            checked_commit,
        )
        result = phase.emit_per_launch_phase_ledgers(
            **kwargs,
            source_reader=source_reader,
        )
        source_reader.verify_path()

    assert observed_during_commit
    assert result["generation"].path.name == "attempt-retained-source"


def test_phase_ledger_postcommit_failure_carries_final_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-postcommit-source-change"
    )
    source = Path(kwargs["traces_folder"]) / "phase_records_0_0.bin"
    original_commit = phase.ExclusiveDirectoryStage.commit
    original_close = phase.ExclusiveDirectoryStage.close

    def commit_then_mutate(stage, **commit_kwargs):
        generation = original_commit(stage, **commit_kwargs)
        source.write_bytes(b"changed after ledger commit")
        return generation

    def fail_close_after_commit(stage) -> None:
        original_close(stage)
        if stage.committed:
            raise OSError("planted phase stage close failure")

    monkeypatch.setattr(
        phase.ExclusiveDirectoryStage,
        "commit",
        commit_then_mutate,
    )
    monkeypatch.setattr(
        phase.ExclusiveDirectoryStage,
        "close",
        fail_close_after_commit,
    )

    with pytest.raises(phase.PhaseLedgerPostCommitFailure) as failure:
        phase.emit_per_launch_phase_ledgers(**kwargs)

    final = Path(kwargs["ledger_output_dir"]) / str(kwargs["attempt_id"])
    assert failure.value.typed_reason == "trace_filesystem_invalid"
    assert failure.value.generation.path == final
    assert final.is_dir()
    assert not (final.parent / f".{final.name}.staging").exists()
    assert any(
        "phase transaction cleanup also failed" in note
        for note in getattr(failure.value, "__notes__", ())
    )


@pytest.mark.parametrize("cleanup_target", ("stage", "monitor", "reader"))
@pytest.mark.parametrize(
    ("resource_failure", "typed_reason"),
    (
        (False, "trace_filesystem_invalid"),
        (True, "trace_snapshot_resource_unavailable"),
    ),
)
def test_phase_ledger_success_cleanup_failure_preserves_generation_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_target: str,
    resource_failure: bool,
    typed_reason: str,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path,
        monkeypatch,
        attempt_id=f"attempt-cleanup-{cleanup_target}-{resource_failure}",
    )
    committed = False
    original_commit = phase.ExclusiveDirectoryStage.commit

    def tracked_commit(stage, **commit_kwargs):
        nonlocal committed
        generation = original_commit(stage, **commit_kwargs)
        committed = True
        return generation

    monkeypatch.setattr(
        phase.ExclusiveDirectoryStage,
        "commit",
        tracked_commit,
    )
    if cleanup_target == "stage":
        owner = phase.ExclusiveDirectoryStage
    elif cleanup_target == "monitor":
        owner = phase.PinnedPathSetMonitor
    else:
        owner = phase.PinnedDirectoryReader
    original_close = owner.close

    def fail_after_close(instance) -> None:
        original_close(instance)
        if committed:
            if resource_failure:
                raise ImmutableArtifactResourceError(
                    "planted postcommit cleanup resource failure"
                )
            raise OSError("planted postcommit cleanup filesystem failure")

    monkeypatch.setattr(owner, "close", fail_after_close)

    with pytest.raises(phase.PhaseLedgerPostCommitFailure) as failure:
        phase.emit_per_launch_phase_ledgers(**kwargs)

    final = Path(kwargs["ledger_output_dir"]) / str(kwargs["attempt_id"])
    assert failure.value.typed_reason == typed_reason
    assert failure.value.generation.path == final
    observed = inspect_immutable_directory_generation(
        final,
        max_files=failure.value.generation.file_count,
        max_directories=failure.value.generation.directory_count,
        max_total_file_bytes=failure.value.generation.total_file_bytes,
        max_total_relative_path_bytes=4096,
    )
    assert observed == failure.value.generation


def test_phase_ledger_recovers_exact_generation_when_commit_hook_raises_late(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path,
        monkeypatch,
        attempt_id="attempt-commit-hook-late-failure",
    )
    original_commit = phase.ExclusiveDirectoryStage.commit

    def commit_then_raise(stage, **commit_kwargs):
        original_commit(stage, **commit_kwargs)
        raise OSError("planted failure after successful generation rename")

    monkeypatch.setattr(
        phase.ExclusiveDirectoryStage,
        "commit",
        commit_then_raise,
    )

    with pytest.raises(phase.PhaseLedgerPostCommitFailure) as failure:
        phase.emit_per_launch_phase_ledgers(**kwargs)

    final = Path(kwargs["ledger_output_dir"]) / str(kwargs["attempt_id"])
    assert failure.value.typed_reason == "trace_filesystem_invalid"
    assert failure.value.generation.path == final
    assert inspect_immutable_directory_generation(
        final,
        max_files=failure.value.generation.file_count,
        max_directories=failure.value.generation.directory_count,
        max_total_file_bytes=failure.value.generation.total_file_bytes,
        max_total_relative_path_bytes=4096,
    ) == failure.value.generation


def test_phase_ledger_rejects_restored_source_change_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-mutated-source"
    )
    source = Path(kwargs["traces_folder"]) / "phase_records_0_0.bin"
    original = source.read_bytes()

    def mutate_and_restore(_stage) -> None:
        source.write_bytes(b"changed-phase-record")
        source.write_bytes(original)

    with pytest.raises(
        ImmutableArtifactError,
        match="protected input tree changed",
    ):
        phase.emit_per_launch_phase_ledgers(
            **kwargs,
            before_commit=mutate_and_restore,
        )

    root = Path(kwargs["ledger_output_dir"])
    assert not (root / "attempt-mutated-source").exists()
    assert (
        root
        / ".attempt-mutated-source.staging"
        / "phase_cycle_ledger_0_0.json"
    ).is_file()


@pytest.mark.parametrize(
    "node_kind", ("regular", "directory", "symlink", "hardlink", "fifo")
)
def test_phase_ledger_generation_rejects_preexisting_destination_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node_kind: str,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-preexisting"
    )
    root = Path(kwargs["ledger_output_dir"])
    root.mkdir()
    final = root / "attempt-preexisting"
    target = tmp_path / "retained-target"
    target.write_bytes(b"retained-target-bytes")
    if node_kind == "regular":
        final.write_bytes(b"regular-winner")
    elif node_kind == "directory":
        final.mkdir()
        (final / "marker").write_bytes(b"directory-winner")
    elif node_kind == "symlink":
        final.symlink_to(target)
    elif node_kind == "hardlink":
        os.link(target, final)
    else:
        os.mkfifo(final)

    try:
        with pytest.raises(ImmutableArtifactError, match="already exists"):
            phase.emit_per_launch_phase_ledgers(**kwargs)
        if node_kind == "regular":
            assert final.read_bytes() == b"regular-winner"
        elif node_kind == "directory":
            assert (final / "marker").read_bytes() == b"directory-winner"
        elif node_kind == "symlink":
            assert final.is_symlink()
            assert target.read_bytes() == b"retained-target-bytes"
        elif node_kind == "hardlink":
            assert final.stat().st_ino == target.stat().st_ino
            assert target.read_bytes() == b"retained-target-bytes"
        else:
            assert stat.S_ISFIFO(final.lstat().st_mode)
    finally:
        if node_kind == "fifo":
            final.unlink()


def test_phase_ledger_generation_destination_race_preserves_both_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-race"
    )
    root = Path(kwargs["ledger_output_dir"])
    final = root / "attempt-race"

    def plant_winner(_stage) -> None:
        final.mkdir()
        (final / "winner").write_bytes(b"concurrent-winner")

    with pytest.raises(ImmutableArtifactError, match="destination appeared"):
        phase.emit_per_launch_phase_ledgers(
            **kwargs,
            before_commit=plant_winner,
        )

    assert (final / "winner").read_bytes() == b"concurrent-winner"
    hidden = root / ".attempt-race.staging"
    assert (
        hidden / "phase_cycle_ledger_0_0.json"
    ).is_file()


def test_phase_ledger_generation_interruption_preserves_hidden_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-interrupted"
    )
    root = Path(kwargs["ledger_output_dir"])

    def interrupt(_stage) -> None:
        raise RuntimeError("planted phase publication interruption")

    with pytest.raises(RuntimeError, match="planted phase publication"):
        phase.emit_per_launch_phase_ledgers(
            **kwargs,
            before_commit=interrupt,
        )

    assert not (root / "attempt-interrupted").exists()
    assert (
        root
        / ".attempt-interrupted.staging"
        / "phase_cycle_ledger_0_0.json"
    ).is_file()


def test_phase_ledger_failed_new_attempt_preserves_prior_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-prior-success"
    )
    first = phase.emit_per_launch_phase_ledgers(**first_kwargs)["generation"]
    first_payload = (
        first.path / "phase_cycle_ledger_0_0.json"
    ).read_bytes()

    second_kwargs = dict(first_kwargs)
    second_kwargs["attempt_id"] = "attempt-new-failure"

    def interrupt(_stage) -> None:
        raise RuntimeError("planted failure after prior success")

    with pytest.raises(RuntimeError, match="failure after prior success"):
        phase.emit_per_launch_phase_ledgers(
            **second_kwargs,
            before_commit=interrupt,
        )

    assert (
        first.path / "phase_cycle_ledger_0_0.json"
    ).read_bytes() == first_payload
    assert inspect_immutable_directory_generation(
        first.path,
        max_files=first.file_count,
        max_directories=first.directory_count,
        max_total_file_bytes=first.total_file_bytes,
        max_total_relative_path_bytes=4096,
    ) == first
    second_root = Path(first_kwargs["ledger_output_dir"])
    assert not (second_root / "attempt-new-failure").exists()
    assert (
        second_root
        / ".attempt-new-failure.staging"
        / "phase_cycle_ledger_0_0.json"
    ).is_file()


def test_phase_ledger_attempt_generations_coexist_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_kwargs = _phase_emission_kwargs(
        tmp_path, monkeypatch, attempt_id="attempt-one"
    )
    first = phase.emit_per_launch_phase_ledgers(**first_kwargs)["generation"]
    first_payload = (
        first.path / "phase_cycle_ledger_0_0.json"
    ).read_bytes()
    second_kwargs = dict(first_kwargs)
    second_kwargs["attempt_id"] = "attempt-two"
    second = phase.emit_per_launch_phase_ledgers(**second_kwargs)["generation"]

    assert first.path != second.path
    assert first.path.is_dir() and second.path.is_dir()
    assert (
        first.path / "phase_cycle_ledger_0_0.json"
    ).read_bytes() == first_payload
    assert first.content_manifest_sha256 == second.content_manifest_sha256
    assert (
        first.filesystem_generation_sha256
        != second.filesystem_generation_sha256
    )
