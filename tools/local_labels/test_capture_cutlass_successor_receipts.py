from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.local_labels import capture_cutlass_successor_receipts as capture


MATRIX_KEY = "cutlass_successor:vendor_gemm_cublas:vendor_gemm_cutlass"
VARIANT_ID = "vendor_gemm_cutlass_bf16_simt_trace00_v1"
GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"


def _containment() -> capture.delegated_cgroup.CgroupContainmentEvidence:
    return capture.delegated_cgroup.CgroupContainmentEvidence(
        kind=capture.delegated_cgroup.CGROUP_V2_KILL_V1,
        parent_cgroup_sha256="c" * 64,
        cgroup_inode=1234,
        terminal_populated_zero=True,
        leaf_removed=True,
    )


def test_capture_producer_identity_authenticates_loaded_self_seal() -> None:
    path, digest = capture._capture_producer_identity()

    assert path == Path(capture.__file__).resolve()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_capture_producer_identity_rejects_resealed_named_replacement(
    monkeypatch,
) -> None:
    payload = Path(capture.__file__).read_bytes().replace(
        b"Capture one diagnostic",
        b"Capture old diagnostic",
        1,
    )
    match = capture.admission._CAPTURE_PRODUCER_SELF_SEAL.search(payload)
    assert match is not None
    start, end = match.span(1)
    normalized = payload[:start] + (b"0" * 64) + payload[end:]
    replacement_seal = hashlib.sha256(normalized).hexdigest().encode("ascii")
    replacement = payload[:start] + replacement_seal + payload[end:]

    class RetainedReplacement:
        sha256 = hashlib.sha256(replacement).hexdigest()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read_bytes(self):
            return replacement

    monkeypatch.setattr(
        capture.immutable,
        "open_retained_regular",
        lambda _path: RetainedReplacement(),
    )

    with pytest.raises(
        capture.CaptureError,
        match="loaded capture producer disagrees",
    ):
        capture._capture_producer_identity()


def _pinned_preflight(tmp_path: Path) -> Path:
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    document = capture._expected_preflight(row, roster_row)
    path = tmp_path / "successor_preflight.json"
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    return path


def test_preflight_clock_refusal_happens_before_live_gpu_queries(tmp_path: Path, monkeypatch):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["sm_clock_mhz"] = 1410
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda **_kwargs: pytest.fail(
            "live GPU query must not run after static preflight refusal"
        ),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *args, **kwargs: pytest.fail("capture process must not run"),
    )
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="ncu",
            preflight_path=path,
            output_root=tmp_path / "capture-output",
        )
    assert not list((tmp_path / "capture-output").glob("*.capture_receipt.json"))


def test_preflight_requires_exact_successor_identity(tmp_path: Path):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["binary_sha256"] = "0" * 64
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.validate_successor_preflight(path, row, roster_row)


def test_preflight_rejects_extras_and_wrong_toolchain(tmp_path: Path):
    path = _pinned_preflight(tmp_path)
    document = json.loads(path.read_text())
    document["cuda_release"] = {"major": 13, "minor": 3}
    document["unexpected"] = True
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="preflight pins mismatch"):
        capture.validate_successor_preflight(path, row, roster_row)


def test_preflight_rejects_non_absolute_input(tmp_path: Path):
    row, roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    with pytest.raises(capture.CaptureError, match="absolute"):
        capture.validate_successor_preflight(Path("relative-preflight.json"), row, roster_row)


def test_nvbit_live_preflight_never_discovers_or_invokes_ncu(monkeypatch):
    observed_queries = []

    def query(_command, label):
        observed_queries.append(label)
        if label == "nvidia-smi device":
            return (
                f"{capture.GPU_NAME}, 8.6, Enabled, "
                f"{capture.SM_CLOCK_MHZ}, {capture.MEM_CLOCK_MHZ}\n",
                [],
            )
        if label == "nvidia-smi competing-process":
            return "", []
        if label == "nvidia-smi device identity":
            return f"{GPU_UUID}, {capture.DRIVER_VERSION}\n", []
        if label == "CUDA 13.2 nvcc":
            return "synthetic nvcc", []
        pytest.fail(f"unexpected live query: {label}")

    monkeypatch.setattr(capture, "_query", query)
    monkeypatch.setattr(capture, "_nvbit_version", lambda: capture.NVBIT_VERSION)
    monkeypatch.setattr(
        capture,
        "parse_nvcc_release_banner",
        lambda _stdout: {"major": 13, "minor": 2},
    )
    monkeypatch.setattr(capture, "require_sol_cuda_release", lambda _cuda: None)
    monkeypatch.setattr(
        capture,
        "_probe_ncu_authority",
        lambda: pytest.fail("NVBit preflight must not launch or discover NCU"),
    )

    live = capture.validate_live_capture_preflight(phase="nvbit")

    assert observed_queries == [
        "nvidia-smi device",
        "nvidia-smi competing-process",
        "nvidia-smi device identity",
        "CUDA 13.2 nvcc",
    ]
    assert not any(key.startswith("ncu_") for key in live)


@pytest.mark.parametrize(
    "malformed_uuid",
    (
        "GPU-11111111111111111111111111111111",
        "GPU-11111111-1111-1111-1111-11111111111",
        "GPU-11111111-1111-1111-1111-111111111111-",
    ),
)
def test_live_preflight_rejects_malformed_gpu_uuid(
    monkeypatch,
    malformed_uuid: str,
):
    def query(_command, label):
        if label == "nvidia-smi device":
            return (
                f"{capture.GPU_NAME}, 8.6, Enabled, "
                f"{capture.SM_CLOCK_MHZ}, {capture.MEM_CLOCK_MHZ}\n",
                [],
            )
        if label == "nvidia-smi competing-process":
            return "", []
        if label == "nvidia-smi device identity":
            return f"{malformed_uuid}, {capture.DRIVER_VERSION}\n", []
        pytest.fail(f"unexpected live query: {label}")

    monkeypatch.setattr(capture, "_query", query)

    with pytest.raises(capture.CaptureError, match="live GPU identity is not pinned"):
        capture.validate_live_capture_preflight(phase="nvbit")


def test_live_preflight_rejects_retired_driver_authority(monkeypatch):
    def query(_command, label):
        if label == "nvidia-smi device":
            return (
                f"{capture.GPU_NAME}, 8.6, Enabled, "
                f"{capture.SM_CLOCK_MHZ}, {capture.MEM_CLOCK_MHZ}\n",
                [],
            )
        if label == "nvidia-smi competing-process":
            return "", []
        if label == "nvidia-smi device identity":
            return f"{GPU_UUID}, 610.43.02\n", []
        pytest.fail(f"unexpected live query: {label}")

    monkeypatch.setattr(capture, "_query", query)

    with pytest.raises(capture.CaptureError, match="live GPU identity is not pinned"):
        capture.validate_live_capture_preflight(phase="nvbit")


def test_capture_metadata_is_namespaced_by_variant_and_phase(tmp_path: Path):
    first = set(
        capture._capture_metadata_paths(
            tmp_path,
            variant_id="vendor_gemm_cutlass_bf16_simt_trace00_v1",
            phase="ncu",
        )
    )
    second = set(
        capture._capture_metadata_paths(
            tmp_path,
            variant_id="vendor_sgemm_cutlass_fp32_simt_trace00_v1",
            phase="ncu",
        )
    )
    first_nvbit = set(
        capture._capture_metadata_paths(
            tmp_path,
            variant_id="vendor_gemm_cutlass_bf16_simt_trace00_v1",
            phase="nvbit",
        )
    )

    assert first.isdisjoint(second)
    assert first.isdisjoint(first_nvbit)
    assert second.isdisjoint(first_nvbit)


def test_post_capture_tool_identity_rejects_substitution(tmp_path: Path):
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"pinned tracer")
    expected = capture.sha256_bytes(tracer.read_bytes())
    tracer.write_bytes(b"substituted tracer")

    with pytest.raises(capture.CaptureError, match="changed during capture"):
        capture._verify_unchanged(tracer, expected, "NVBit tracer")


def test_failure_provenance_seals_process_tool_and_stream_identity(
    tmp_path: Path, monkeypatch,
):
    failure_root = tmp_path / "failure"
    failure_root.mkdir()
    (failure_root / "capture_timeout.stdout.bin").write_bytes(b"partial stdout")
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    command = ["/fixture/target", "--shape", "128"]
    environment_payload = b'{"environment":{"LANG":"C"}}\n'
    error = capture.CaptureError(
        "synthetic timeout",
        process_id=4321,
        started_at_utc="2026-08-02T00:00:00Z",
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    capture._write_failure_provenance(
        failure_root,
        root=tmp_path,
        phase="ncu",
        row=row,
        command=command,
        live={
            "gpu_name": capture.GPU_NAME,
            "smi_stdout_sha256": "c" * 64,
            "gpu_uuid_sha256": "u" * 64,
            "driver_version": capture.DRIVER_VERSION,
            "identity_stdout_sha256": "i" * 64,
        },
        environment_payload=environment_payload,
        default_environment={"LANG": "C"},
        capture_producer_sha256="d" * 64,
        containment_supervisor_sha256="a" * 64,
        phase_tool_sha256="e" * 64,
        action_executor_sha256="f" * 64,
        error=error,
    )

    document = json.loads((failure_root / "failure_provenance.json").read_text())
    assert document["process"] == {
        "role": "ncu_capture",
        "boot_id_sha256": "b" * 64,
        "command": command,
        "command_sha256": capture.canonical_sha(command),
        "environment": {"LANG": "C"},
        "environment_sha256": capture.canonical_sha({"LANG": "C"}),
        "host_identity_sha256": capture.canonical_sha({
            "gpu": capture.GPU_NAME,
            "smi": "c" * 64,
            "gpu_uuid_sha256": "u" * 64,
            "driver_version": capture.DRIVER_VERSION,
            "identity_stdout_sha256": "i" * 64,
        }),
        "pid": 4321,
        "started_at_utc": "2026-08-02T00:00:00Z",
    }
    assert document["capture_environment_document_sha256"] == (
        capture.sha256_bytes(environment_payload)
    )
    assert document["handoff_artifacts"] == []
    assert document["provenance"] == {
        "action_executor_sha256": "f" * 64,
        "capture_producer_sha256": "d" * 64,
        "containment_supervisor_sha256": "a" * 64,
        "phase_tool_sha256": "e" * 64,
    }
    assert document["streams"] == [
        {
            "path": "capture_timeout.stdout.bin",
            "sha256": capture.sha256_bytes(b"partial stdout"),
            "size_bytes": len(b"partial stdout"),
        }
    ]
    seal = document.pop("failure_seal_sha256")
    assert seal == capture.canonical_sha(document)


def _nvbit_live() -> dict:
    smi_stdout = (
        f"{capture.GPU_NAME}, 8.6, Enabled, "
        f"{capture.SM_CLOCK_MHZ}, {capture.MEM_CLOCK_MHZ}\n"
    )
    return {
        "smi_command": ["nvidia-smi"],
        "smi_stdout": smi_stdout,
        "smi_stdout_sha256": capture.sha256_bytes(smi_stdout.encode()),
        "competing_process_command": ["nvidia-smi", "--compute-apps"],
        "competing_process_stdout": "",
        "competing_process_stdout_sha256": capture.sha256_bytes(b""),
        "identity_command": ["nvidia-smi", "--identity"],
        "identity_stdout": f"{GPU_UUID}, {capture.DRIVER_VERSION}\n",
        "identity_stdout_sha256": capture.sha256_bytes(
            f"{GPU_UUID}, {capture.DRIVER_VERSION}\n".encode()
        ),
        "gpu_uuid_sha256": capture.sha256_bytes(GPU_UUID.encode()),
        "driver_version": capture.DRIVER_VERSION,
        "nvcc_command": [str(capture.CUDA_NVCC), "--version"],
        "nvcc_stdout": "synthetic nvcc",
        "nvcc_stdout_sha256": capture.sha256_bytes(b"synthetic nvcc"),
        "cuda_release": dict(capture.CUDA_RELEASE),
        "nvbit_version": capture.NVBIT_VERSION,
        "gpu_name": capture.GPU_NAME,
        "persistence_mode": "Enabled",
        "sm_clock_mhz": capture.SM_CLOCK_MHZ,
        "mem_clock_mhz": capture.MEM_CLOCK_MHZ,
    }


@pytest.mark.parametrize(
    ("failure_type", "stream_write_failure"),
    (
        (OSError, False),
        (TypeError, False),
        (KeyboardInterrupt, False),
        (TypeError, True),
    ),
)
def test_nvbit_descriptor_execution_and_post_validation_failure_provenance(
    tmp_path: Path,
    monkeypatch,
    failure_type: type[BaseException],
    stream_write_failure: bool,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    live = _nvbit_live()
    observed = {}

    def run(command, environment, _cwd, **kwargs):
        observed.update({
            "command": command,
            "environment": environment,
            "descriptors": kwargs["inherited_descriptors"],
        })
        assert command[0].startswith("/proc/self/fd/")
        assert environment["CUDA_INJECTION64_PATH"].startswith(
            "/proc/self/fd/"
        )
        for descriptor in kwargs["inherited_descriptors"]:
            capture.os.fstat(descriptor)
        return (
            0,
            4321,
            "2026-08-02T00:00:00Z",
            b"validated stdout",
            b"",
            _containment(),
        )

    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: live if phase == "nvbit" else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "tracer_runtime_env",
        lambda _path, *, base_env, **_kwargs: dict(base_env),
    )
    monkeypatch.setattr(capture, "_run_capture_command", run)

    class GenerationGuard:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            observed["generation_closed"] = True
            return False

    def finalize(**kwargs):
        kwargs["generation_stack"].enter_context(GenerationGuard())
        raise failure_type("synthetic post-validation failure")

    monkeypatch.setattr(capture, "_finalize_capture", finalize)
    if stream_write_failure:
        real_write_new = capture._write_new

        def write_with_diagnostic_failure(path: Path, payload: bytes) -> None:
            if path.name == "post_validation.stdout.bin":
                raise capture.CaptureError("synthetic diagnostic write failure")
            real_write_new(path, payload)

        monkeypatch.setattr(capture, "_write_new", write_with_diagnostic_failure)
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    expected_error = (
        KeyboardInterrupt
        if failure_type is KeyboardInterrupt
        else capture.CaptureError
    )
    expected_message = (
        "synthetic post-validation failure"
        if failure_type is KeyboardInterrupt
        else "post-capture validation"
    )
    with pytest.raises(expected_error, match=expected_message):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
        )

    assert len(observed["descriptors"]) == 2
    assert observed["generation_closed"] is True
    failure_root = output_root / "failures" / VARIANT_ID / "nvbit"
    stdout_path = failure_root / "post_validation.stdout.bin"
    if stream_write_failure:
        assert not stdout_path.exists()
    else:
        assert stdout_path.read_bytes() == b"validated stdout"
    document = json.loads((failure_root / "failure_provenance.json").read_text())
    if stream_write_failure:
        assert "post-validation stream retention failed" in document["detail"]
    assert document["process"]["pid"] == 4321
    assert document["process"]["role"] == "nvbit_postprocess_validation"
    assert document["process"]["command"][0] != observed["command"][0]
    assert document["process"]["environment"]["CUDA_INJECTION64_PATH"] == (
        str(capture.TRACER_SO)
    )
    assert document["provenance"] == {
        "capture_producer_sha256": capture.sha256_bytes(
            Path(capture.__file__).read_bytes()
        ),
        "containment_supervisor_sha256": capture.sha256_bytes(
            Path(capture.delegated_cgroup.__file__).read_bytes()
        ),
        "phase_tool_sha256": capture.sha256_bytes(capture.TRACER_SO.read_bytes()),
    }


def test_nvbit_runtime_preflight_failure_is_retained(
    tmp_path: Path, monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "tracer_runtime_env",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            capture.nvbit_runtime.TracerRuntimeError(
                "synthetic runtime dependency refusal"
            )
        ),
    )
    monkeypatch.setattr(
        capture,
        "_run_capture_command",
        lambda *_args, **_kwargs: pytest.fail("runtime refusal must not launch"),
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(capture.CaptureError, match="runtime dependency refusal"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
        )

    failure = json.loads(
        (
            output_root
            / "failures"
            / VARIANT_ID
            / "nvbit"
            / "failure_provenance.json"
        ).read_text()
    )
    assert failure["process"]["pid"] is None
    assert failure["process"]["role"] == "nvbit_capture"
    assert failure["process"]["command"][0].startswith(str(capture.ROOT))
    assert failure["process"]["environment"]["CUDA_INJECTION64_PATH"] == str(
        capture.TRACER_SO
    )


def test_nvbit_process_failure_records_canonical_descriptor_authority(
    tmp_path: Path, monkeypatch,
):
    preflight = _pinned_preflight(tmp_path)
    output_root = tmp_path / "capture-output"
    observed = {}
    monkeypatch.setattr(
        capture,
        "validate_live_capture_preflight",
        lambda *, phase: _nvbit_live()
        if phase == "nvbit"
        else pytest.fail("wrong phase"),
    )
    monkeypatch.setattr(
        capture.nvbit_runtime,
        "tracer_runtime_env",
        lambda _path, *, base_env, **_kwargs: dict(base_env),
    )

    def fail(command, environment, _cwd, **_kwargs):
        observed["command"] = command
        observed["environment"] = environment
        raise capture.CaptureError(
            "synthetic descriptor-bound timeout",
            process_id=4321,
            started_at_utc="2026-08-02T00:00:00Z",
            containment=_containment(),
        )

    monkeypatch.setattr(capture, "_run_capture_command", fail)
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(capture.CaptureError, match="descriptor-bound timeout"):
        capture.capture_phase(
            matrix_key=MATRIX_KEY,
            variant_id=VARIANT_ID,
            phase="nvbit",
            preflight_path=preflight,
            output_root=output_root,
        )

    failure = json.loads(
        (
            output_root
            / "failures"
            / VARIANT_ID
            / "nvbit"
            / "failure_provenance.json"
        ).read_text()
    )
    assert observed["command"][0].startswith("/proc/self/fd/")
    assert observed["environment"]["CUDA_INJECTION64_PATH"].startswith(
        "/proc/self/fd/"
    )
    assert failure["process"]["pid"] == 4321
    assert failure["process"]["command"][0] != observed["command"][0]
    assert failure["process"]["environment"]["CUDA_INJECTION64_PATH"] == str(
        capture.TRACER_SO
    )


def test_exclusive_writer_never_follows_preexisting_symlink(tmp_path: Path):
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"retained")
    destination = tmp_path / "receipt.json"
    destination.symlink_to(victim)

    with pytest.raises(capture.CaptureError, match="exclusively publish"):
        capture._write_new(destination, b"forged")

    assert victim.read_bytes() == b"retained"
    assert destination.is_symlink()


def test_preexisting_sibling_files_are_allowed_but_current_output_is_safe(tmp_path: Path):
    root = tmp_path / "output"
    root.mkdir()
    (root / "prior-phase").mkdir()
    (root / "prior-phase" / "prior.ncu-rep").write_bytes(b"prior")
    current = root / "nvbit" / "successor"
    capture._reject_preexisting_output_descendants(
        root, output_dir=current, reserved_files=(root / "preflight_nvbit.json",)
    )

    current.parent.mkdir(parents=True)
    current.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(capture.CaptureError, match="unsafe"):
        capture._reject_preexisting_output_descendants(
            root, output_dir=current, reserved_files=()
        )


def test_preexisting_output_files_are_rejected(tmp_path: Path):
    root = tmp_path / "output-file"
    root.mkdir()
    output_dir = root / "ncu"
    output_dir.write_bytes(b"stale")
    with pytest.raises(capture.CaptureError, match="unsafe"):
        capture._reject_preexisting_output_descendants(
            root, output_dir=output_dir, reserved_files=()
        )


def test_receipt_publication_rejects_artifact_generation_mutation(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "capture"
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    cell = root / capture.ARCH / row["successor_slug"] / "trace_00"
    cell.mkdir(parents=True)
    sidecar = cell / "label_0_0.json"
    sidecar.write_text(json.dumps({
        "unit_tagged_metrics": [
            {"ncu_metric": "gpc__cycles_elapsed.max"},
            {"ncu_metric": "gpu__time_duration.sum"},
        ],
    }))
    (cell / "run_0_launch_set.ncu-rep").write_bytes(b"report")
    run_manifest = {
        "schema_version": 1,
        "source_commit": capture.PINNED_COMMIT,
        "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
        "kernel_signature_sha256": hashlib.sha256(
            (
                f"{row['successor_slug']}|"
                f"{row['static_launch_inventory']['binary_path']}"
            ).encode()
        ).hexdigest(),
        "kernel_signature_source": "fallback_legacy",
        "arch": capture.ARCH,
        "slug": row["successor_slug"],
        "shape_point": row["shape"]["shape_point"],
        "repeat_index": 0,
        "launch_index": 0,
    }
    (cell / "label_run_manifest_0_launch_set.json").write_text(
        json.dumps(run_manifest, sort_keys=True, indent=2) + "\n"
    )
    observation = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": 0,
        "requested_launch_count": 1,
        "report_sha256": capture.sha256_bytes(b"report"),
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": "0",
            "Device": capture.GPU_NAME,
            "Kernel Name": "fixture",
            "Block Size": "(1, 1, 1)",
            "Grid Size": "(1, 1, 1)",
        },
    }
    observation["observation_sha256"] = capture.sha256_bytes(
        json.dumps(
            observation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    )
    (cell / "run_0_launch_set.launch_observation.json").write_text(
        json.dumps(observation, sort_keys=True, indent=2) + "\n"
    )
    preflight_payload = b"{\"preflight\":true}\n"
    environment_payload = b"{\"environment\":true}\n"
    preflight = root / f"{VARIANT_ID}.preflight_ncu.json"
    environment = root / f"{VARIANT_ID}.environment_ncu.json"
    preflight.write_bytes(preflight_payload)
    environment.write_bytes(environment_payload)
    receipt = root / f"{VARIANT_ID}.ncu.capture_receipt.json"
    original_validate = capture.admission._validate_schema

    def mutate_after_validation(*args, **kwargs):
        original_validate(*args, **kwargs)
        sidecar.write_text(json.dumps({
            "unit_tagged_metrics": [
                {"ncu_metric": "gpc__cycles_elapsed.max"},
            ],
        }))

    monkeypatch.setattr(capture.admission, "_validate_schema", mutate_after_validation)

    with pytest.raises(
        capture.CaptureError,
        match="transactionally publish capture receipt",
    ):
        capture._finalize_capture(
            root=root,
            phase="ncu",
            row=row,
            live={
                "gpu_name": capture.GPU_NAME,
                "smi_stdout_sha256": "a" * 64,
                "gpu_uuid_sha256": "b" * 64,
                "driver_version": capture.DRIVER_VERSION,
                "identity_stdout_sha256": "c" * 64,
            },
            pid=1234,
            started="2026-08-02T00:00:00Z",
            containment=_containment(),
            command=[str(capture.LABEL_DRIVER), "synthetic"],
            capture_source=Path(capture.__file__),
            capture_producer_sha256=capture.sha256_bytes(
                Path(capture.__file__).read_bytes()
            ),
            containment_supervisor_sha256=capture.sha256_bytes(
                Path(capture.delegated_cgroup.__file__).read_bytes()
            ),
            phase_tool_sha256="b" * 64,
            action_executor_sha256=capture.sha256_bytes(
                Path(capture.ncu_executor.__file__).read_bytes()
            ),
            preflight_out=preflight,
            preflight_payload=preflight_payload,
            environment_out=environment,
            environment_payload=environment_payload,
            receipt_path=receipt,
        )

    assert not receipt.exists()
    retired = [
        path
        for path in receipt.parent.iterdir()
        if path.name.startswith(".failed-regular-")
        and not path.name.endswith(".salvaging")
    ]
    assert len(retired) == 1
    assert json.loads(retired[0].read_bytes())["receipt"]["status"] == "pass"


def test_zero_size_artifacts_are_rejected(tmp_path: Path):
    root = tmp_path / "output"
    root.mkdir()
    empty = root / "empty.pb"
    empty.touch()
    with pytest.raises(capture.CaptureError, match="zero-sized"):
        capture._artifact(root, empty, "threadblock")


def test_trace_producer_version_mismatch_is_rejected(tmp_path: Path):
    trace_pb2 = capture._import_trace_pb2()
    trace = trace_pb2.Trace()
    trace.trace_contract_id = capture.EXPECTED_TRACE_CONTRACT_ID
    trace.memory_payload_classes.extend(capture.EXPECTED_MEMORY_PAYLOAD_CLASSES)
    trace.binary_version = 120
    trace.nvbit_version = capture.NVBIT_VERSION
    (tmp_path / "dynamic_trace.pb").write_bytes(trace.SerializeToString())
    with pytest.raises(capture.CaptureError, match="producer version mismatch"):
        capture._validate_trace_producer(tmp_path)


def test_stream_reader_bounds_each_ready_descriptor_to_one_read(monkeypatch):
    key = SimpleNamespace(data="stdout", fd=73, fileobj=object())

    class AlwaysReadySelector:
        def select(self, _timeout):
            return [(key, 1)]

        def unregister(self, _fileobj):
            pytest.fail("nonempty synthetic stream was unregistered")

    reads = 0

    def always_readable(_descriptor, size):
        nonlocal reads
        reads += 1
        if reads > 1:
            pytest.fail("one readiness event drained without a bounded return")
        return b"x" * size

    monkeypatch.setattr(capture.os, "read", always_readable)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}

    exceeded, error, progressed = capture._read_capture_streams(
        AlwaysReadySelector(),
        buffers,
        timeout_s=0,
    )

    assert error is None
    assert progressed is True
    assert exceeded is False
    assert reads == 1
    assert buffers["stdout"] == b"x" * capture.CAPTURE_STREAM_READ_BYTES


def test_capture_timeout_is_bounded(tmp_path: Path):
    with pytest.raises(capture.CaptureError, match="timeout"):
        capture._run_capture_command(
            ["python", "-c", "import time; time.sleep(2)"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=0.05,
        )


def test_capture_timeout_retains_process_streams(tmp_path: Path):
    failure_root = tmp_path / "failure"
    command = [
        capture.sys.executable,
        "-c",
        (
            "import sys,time; "
            "sys.stdout.write('timeout stdout\\n'); sys.stdout.flush(); "
            "sys.stderr.write('timeout stderr\\n'); sys.stderr.flush(); "
            "time.sleep(2)"
        ),
    ]
    with pytest.raises(capture.CaptureError, match="timeout") as caught:
        capture._run_capture_command(
            command,
            dict(capture.os.environ),
            tmp_path,
            timeout_s=0.05,
            failure_root=failure_root,
        )
    assert caught.value.process_id is not None
    assert caught.value.process_id > 0
    assert caught.value.started_at_utc is not None
    assert caught.value.started_at_utc.endswith("Z")
    assert (failure_root / "capture_timeout.stdout.bin").read_bytes() == (
        b"timeout stdout\n"
    )
    assert (failure_root / "capture_timeout.stderr.bin").read_bytes() == (
        b"timeout stderr\n"
    )


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_capture_streams_are_bounded_and_retained(
    tmp_path: Path, monkeypatch, stream: str,
):
    failure_root = tmp_path / "failure"
    monkeypatch.setattr(capture, "CAPTURE_STREAM_LIMIT_BYTES", 32)
    expression = (
        "sys.stdout.buffer.write(b'x' * 128)"
        if stream == "stdout"
        else "sys.stderr.buffer.write(b'x' * 128)"
    )
    with pytest.raises(capture.CaptureError, match="stdout/stderr byte limit"):
        capture._run_capture_command(
            [capture.sys.executable, "-c", f"import sys; {expression}"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            failure_root=failure_root,
        )

    assert (
        failure_root / f"capture_output_limit.{stream}.bin"
    ).read_bytes() == b"x" * 32
    other = "stderr" if stream == "stdout" else "stdout"
    assert (
        failure_root / f"capture_output_limit.{other}.bin"
    ).read_bytes() == b""


def test_trace_byte_budget_stops_capture(tmp_path: Path):
    command = [
        capture.sys.executable,
        "-c",
        f"from pathlib import Path; import time; Path({str(tmp_path / 'payload.bin')!r}).write_bytes(b'x' * 64); time.sleep(2)",
    ]
    with pytest.raises(capture.CaptureError, match="byte budget"):
        capture._run_capture_command(
            command,
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=8,
        )


def test_capture_setup_cancellation_reaps_started_process(
    tmp_path: Path, monkeypatch,
):
    started = []
    original_popen = capture.subprocess.Popen

    def record_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        started.append(process)
        return process

    interruption = KeyboardInterrupt("synthetic selector setup interruption")
    monkeypatch.setattr(capture.subprocess, "Popen", record_popen)
    monkeypatch.setattr(
        capture.selectors,
        "DefaultSelector",
        lambda: (_ for _ in ()).throw(interruption),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "import time; time.sleep(2)"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=3,
        )

    assert failure.value is interruption
    assert len(started) == 1
    assert started[0].poll() is not None


def test_process_group_kill_cancellation_does_not_skip_cgroup_finish(
    tmp_path: Path, monkeypatch,
):
    calls = []
    interruption = KeyboardInterrupt("synthetic process-group kill interruption")

    class Containment:
        def popen(self, command, **kwargs):
            return capture.subprocess.Popen(command, **kwargs)

        def kill_all(self):
            calls.append("kill_all")

        def finish(self):
            calls.append("finish")
            return _containment()

    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        Containment,
    )
    monkeypatch.setattr(
        capture,
        "_kill_capture_process",
        lambda _process: (_ for _ in ()).throw(interruption),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "pass"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            require_cgroup_containment=True,
        )

    assert failure.value is interruption
    assert "kill_all" in calls
    assert calls[-1] == "finish"


def test_process_wait_failure_supersedes_setup_cancellation(
    tmp_path: Path, monkeypatch,
):
    original_popen = capture.subprocess.Popen
    spawned = []

    class FailingWaitProcess:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, timeout=None):
            raise capture.subprocess.TimeoutExpired(self.args, timeout)

    def wrap_popen(*args, **kwargs):
        wrapped = FailingWaitProcess(original_popen(*args, **kwargs))
        spawned.append(wrapped)
        return wrapped

    monkeypatch.setattr(capture.subprocess, "Popen", wrap_popen)
    monkeypatch.setattr(
        capture.selectors,
        "DefaultSelector",
        lambda: (_ for _ in ()).throw(
            KeyboardInterrupt("synthetic selector setup interruption")
        ),
    )

    try:
        with pytest.raises(
            capture.CaptureError,
            match="capture process cleanup failed",
        ) as failure:
            capture._run_capture_command(
                [capture.sys.executable, "-c", "import time; time.sleep(2)"],
                dict(capture.os.environ),
                tmp_path,
                timeout_s=3,
            )
        assert any(
            "selector setup interruption" in note
            for note in failure.value.__notes__
        )
    finally:
        if spawned:
            spawned[0].process.wait(timeout=2)


def test_watchdog_join_is_bounded_and_typed(tmp_path: Path, monkeypatch):
    joins = []

    class StuckWatchdog:
        def __init__(self, *, target, daemon):
            assert callable(target)
            assert daemon is True

        def start(self):
            return None

        def join(self, timeout=None):
            joins.append(timeout)

        def is_alive(self):
            return True

    monkeypatch.setattr(capture.threading, "Thread", StuckWatchdog)
    monkeypatch.setattr(capture, "CAPTURE_CLEANUP_TIMEOUT_S", 0.01)

    with pytest.raises(
        capture.CaptureError,
        match="byte-budget watchdog did not stop",
    ):
        capture._run_capture_command(
            [capture.sys.executable, "-c", "pass"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
        )

    assert joins == [0.01]


def test_watchdog_is_joined_before_process_authority_release(
    tmp_path: Path, monkeypatch,
):
    entered = capture.threading.Event()
    release = capture.threading.Event()
    kill_threads: list[str] = []
    result: list[tuple[int, int, str, bytes, bytes]] = []
    errors: list[BaseException] = []
    original_kill = capture._kill_capture_process

    def blocked_scan(_root: Path) -> int:
        entered.set()
        if not release.wait(2):
            raise OSError("synthetic blocked watchdog")
        return 0

    def record_kill(process) -> None:
        kill_threads.append(capture.threading.current_thread().name)
        original_kill(process)

    def run_capture() -> None:
        try:
            result.append(capture._run_capture_command(
                [
                    capture.sys.executable,
                    "-c",
                    "import time; time.sleep(0.15)",
                ],
                dict(capture.os.environ),
                tmp_path,
                timeout_s=2,
                byte_root=tmp_path,
                byte_budget=1024,
            ))
        except BaseException as exc:  # surfaced in the owner test thread
            errors.append(exc)

    monkeypatch.setattr(capture, "_capture_tree_bytes", blocked_scan)
    monkeypatch.setattr(capture, "_kill_capture_process", record_kill)
    owner = capture.threading.Thread(
        target=run_capture, name="capture-owner-test",
    )
    owner.start()
    try:
        assert entered.wait(1)
        time.sleep(0.1)
        assert owner.is_alive()
    finally:
        release.set()
        owner.join(2)

    assert not owner.is_alive()
    assert errors == []
    assert result and result[0][0] == 0
    assert kill_threads
    assert set(kill_threads) == {"capture-owner-test"}


def test_terminal_trace_byte_budget_rejects_fast_final_burst(tmp_path: Path):
    command = [
        capture.sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"Path({str(tmp_path / 'payload.bin')!r}).write_bytes(b'x' * 64)"
        ),
    ]
    with pytest.raises(capture.CaptureError, match="byte budget") as caught:
        capture._run_capture_command(
            command,
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=8,
        )
    assert caught.value.process_id is not None
    assert caught.value.started_at_utc is not None


def test_post_processed_trace_tree_must_remain_within_budget(tmp_path: Path):
    (tmp_path / "raw-trace.bin").write_bytes(b"12345678")
    capture._require_capture_tree_budget(tmp_path, 8)
    (tmp_path / "trace_summary.json").write_bytes(b"x")

    with pytest.raises(capture.CaptureError, match="post-processed"):
        capture._require_capture_tree_budget(tmp_path, 8)


def test_terminal_trace_scan_error_retains_process_and_streams(
    tmp_path: Path, monkeypatch,
):
    failure_root = tmp_path / "failure"
    monkeypatch.setattr(
        capture,
        "_capture_tree_bytes",
        lambda _root: (_ for _ in ()).throw(OSError("synthetic stat race")),
    )
    with pytest.raises(capture.CaptureError, match="unsafe output") as caught:
        capture._run_capture_command(
            [
                capture.sys.executable,
                "-c",
                "import sys; sys.stdout.write('completed output')",
            ],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=2,
            byte_root=tmp_path,
            byte_budget=1024,
            failure_root=failure_root,
        )

    assert caught.value.process_id is not None
    assert caught.value.process_id > 0
    assert caught.value.started_at_utc is not None
    assert (failure_root / "capture_budget.stdout.bin").read_bytes() == (
        b"completed output"
    )


def test_containment_cleanup_failure_supersedes_base_exception(
    tmp_path: Path, monkeypatch,
):
    class FailingContainment:
        def popen(self, command, **kwargs):
            return capture.subprocess.Popen(command, **kwargs)

        def kill_all(self):
            raise capture.delegated_cgroup.CgroupContainmentError(
                "synthetic cgroup.kill failure"
            )

        def finish(self):
            raise capture.delegated_cgroup.CgroupContainmentError(
                "synthetic terminal cleanup failure"
            )

    containment = FailingContainment()
    monkeypatch.setattr(
        capture.delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: containment,
    )
    calls = 0

    def interrupt_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("synthetic interruption")
        return False, None, False

    monkeypatch.setattr(capture, "_read_capture_streams", interrupt_once)

    with pytest.raises(
        capture.CaptureError,
        match="containment cleanup failed.*synthetic terminal cleanup failure",
    ) as caught:
        capture._run_capture_command(
            [capture.sys.executable, "-c", "import time; time.sleep(2)"],
            dict(capture.os.environ),
            tmp_path,
            timeout_s=3,
            require_cgroup_containment=True,
        )

    assert any(
        "synthetic interruption" in note
        for note in getattr(caught.value, "__notes__", ())
    )


@pytest.mark.parametrize("escape_call", ["setsid", "setpgid"])
def test_cgroup_containment_kills_group_escape_descendant(
    tmp_path: Path, escape_call: str,
):
    if capture.os.environ.get("GPU_SIM_DELEGATED_CGROUP_TEST") != "1":
        pytest.skip("requires a systemd-delegated cgroup-v2 test service")
    ready = tmp_path / f"{escape_call}-ready.txt"
    marker = tmp_path / "late-write.bin"
    escape_source = (
        "os.setsid()" if escape_call == "setsid" else "os.setpgid(0, 0)"
    )
    child_source = (
        "import os,time\n"
        "from pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        f"    {escape_source}\n"
        f"    Path({str(ready)!r}).write_text(f'{{os.getpid()}},{{os.getsid(0)}},{{os.getpgrp()}},{{Path(\"/proc/self/cgroup\").read_text().strip()}}')\n"
        "    os.close(1)\n"
        "    os.close(2)\n"
        "    time.sleep(0.4)\n"
        f"    Path({str(marker)!r}).write_bytes(b'late')\n"
        "    os._exit(0)\n"
        "deadline = time.monotonic() + 2\n"
        f"while not Path({str(ready)!r}).is_file() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        f"os._exit(0 if Path({str(ready)!r}).is_file() else 9)\n"
    )

    result = capture._run_capture_command(
        [capture.sys.executable, "-c", child_source],
        dict(capture.os.environ),
        tmp_path,
        timeout_s=3,
        require_cgroup_containment=True,
    )

    assert result[0] == 0
    ready_fields = ready.read_text().split(",", maxsplit=3)
    descendant_pid, descendant_sid, descendant_pgid = [
        int(value) for value in ready_fields[:3]
    ]
    descendant_cgroup = ready_fields[3]
    assert descendant_cgroup.startswith("0::/")
    assert descendant_cgroup.rsplit("/", maxsplit=1)[-1].startswith("gpu-sim-")
    assert descendant_pgid == descendant_pid
    if escape_call == "setsid":
        assert descendant_sid == descendant_pid
    else:
        assert descendant_sid != descendant_pid
    assert result[5] is not None
    assert result[5].as_dict()["terminal_populated_zero"] is True
    time.sleep(0.5)
    assert not marker.exists()


def test_successful_capture_kills_pipe_detached_descendants(tmp_path: Path):
    marker = tmp_path / "late-write.bin"
    command = [
        capture.sys.executable,
        "-c",
        (
            "import os,time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.close(1)\n"
            "    os.close(2)\n"
            "    time.sleep(0.25)\n"
            f"    open({str(marker)!r}, 'wb').write(b'late')\n"
            "    os._exit(0)\n"
            "os._exit(0)\n"
        ),
    ]

    result = capture._run_capture_command(
        command,
        dict(capture.os.environ),
        tmp_path,
        timeout_s=2,
    )

    assert result[0] == 0
    time.sleep(0.4)
    assert not marker.exists()


def _ncu_authority() -> dict:
    return {
        "status": "ok",
        "expected": capture.NCU_VERSION,
        "execution_mode": capture.campaign_preflight.NCU_EXECUTION_MODE,
        "host_machine": capture.campaign_preflight.NCU_HOST_MACHINE,
        "launcher_resolved_path": capture.admission.NCU_LAUNCHER_PATH,
        "launcher_sha256": capture.admission.NCU_LAUNCHER_SHA256,
        "launcher_target_relpath": (
            capture.campaign_preflight.NCU_LAUNCHER_TARGET_RELPATH
        ),
        "resolved_path": capture.admission.NCU_EXECUTABLE_PATH,
        "executable_sha256": capture.admission.NCU_EXECUTABLE_SHA256,
        "version_line": capture.admission.NCU_VERSION_LINE,
    }


def _ncu_command(identity: capture.ncu_executor.ExpectedNcuIdentity) -> list[str]:
    return [
        str(capture.LABEL_DRIVER),
        "--binary", "/fixture/target",
        "--ncu", identity.resolved_path,
        "--ncu-executable-sha256", identity.executable_sha256,
        "--ncu-version-line", identity.version_line,
        "--target-processes", "all",
        "--filter-mode", "global",
        "--replay-mode", "kernel",
        "--cache-control", "all",
        "--clock-control", "none",
        "--profile-from-start", "on",
        "--launch-skip", "0",
        "--launch-count", "1",
        "--",
        "128",
    ]


def _retained_target():
    return SimpleNamespace(
        path=Path("/fixture/target"),
        proc_path="/proc/self/fd/91",
        descriptor=91,
    )


def _ncu_handoff_manifest(row: dict) -> dict:
    return {
        "schema_version": 1,
        "source_commit": capture.PINNED_COMMIT,
        "binary_sha256": row["static_launch_inventory"]["binary_sha256"],
        "kernel_signature_sha256": hashlib.sha256(
            (
                f"{row['successor_slug']}|"
                f"{row['static_launch_inventory']['binary_path']}"
            ).encode()
        ).hexdigest(),
        "kernel_signature_source": "fallback_legacy",
        "arch": capture.ARCH,
        "slug": row["successor_slug"],
        "shape_point": row["shape"]["shape_point"],
        "repeat_index": 0,
        "launch_index": 0,
    }


def _ncu_details_csv(row: dict, kernel_name: str = "_Zfixture_kernel") -> str:
    launch = row["shape"]["trace_shape"]["launch_config"]
    block = ",".join(str(value) for value in launch["block"])
    grid = ",".join(str(value) for value in launch["grid"])
    prefix = (
        f'0,{capture.GPU_NAME},{kernel_name},"{block}","{grid}"'
    )
    return (
        "ID,Device,Kernel Name,Block Size,Grid Size,Metric Name,Metric Value\n"
        f"{prefix},lts__t_sector_hit_rate.pct,50\n"
        f"{prefix},l1tex__t_sector_hit_rate.pct,25\n"
        f"{prefix},gpc__cycles_elapsed.max,123\n"
        f"{prefix},gpu__time_duration.sum,456\n"
    )


def _write_ncu_handoff(root: Path, row: dict) -> tuple[Path, bytes]:
    cell = root / capture.ARCH / row["successor_slug"] / "trace_00"
    cell.mkdir(parents=True)
    report_payload = b"synthetic retained NCU report"
    (cell / "run_0_launch_set.ncu-rep").write_bytes(report_payload)
    (cell / "label_run_manifest_0_launch_set.json").write_text(
        json.dumps(_ncu_handoff_manifest(row), sort_keys=True, indent=2) + "\n"
    )
    return cell, report_payload


def test_ncu_handoff_rejects_duplicate_manifest_keys(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    manifest = cell / "label_run_manifest_0_launch_set.json"
    payload = manifest.read_text()
    manifest.write_text(
        payload.replace(
            '"source_commit":',
            f'"source_commit": "{capture.PINNED_COMMIT}",\n  "source_commit":',
            1,
        )
    )
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_report_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "duplicate manifest must fail before report import"
        ),
    )

    with pytest.raises(capture.CaptureError, match="duplicate JSON key"):
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )


def test_failed_ncu_handoff_inventory_binds_retained_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    failure_root = root / "failure"
    failure_root.mkdir()
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    capture._write_failure_provenance(
        failure_root,
        root=root,
        phase="ncu",
        row=row,
        command=["/fixture/target"],
        live=_nvbit_live(),
        environment_payload=b"{}\n",
        default_environment={"LANG": "C"},
        capture_producer_sha256="c" * 64,
        containment_supervisor_sha256="d" * 64,
        phase_tool_sha256="e" * 64,
        action_executor_sha256="f" * 64,
        error=capture.CaptureError("synthetic postprocess failure"),
    )

    document = json.loads((failure_root / "failure_provenance.json").read_text())
    artifacts = {
        artifact["role"]: artifact
        for artifact in document["handoff_artifacts"]
    }
    assert artifacts == {
        "label_run_manifest": {
            "role": "label_run_manifest",
            "path": (
                cell / "label_run_manifest_0_launch_set.json"
            ).relative_to(root).as_posix(),
            "sha256": capture.sha256_bytes(
                (cell / "label_run_manifest_0_launch_set.json").read_bytes()
            ),
            "size_bytes": (
                cell / "label_run_manifest_0_launch_set.json"
            ).stat().st_size,
        },
        "ncu_report": {
            "role": "ncu_report",
            "path": (
                cell / "run_0_launch_set.ncu-rep"
            ).relative_to(root).as_posix(),
            "sha256": capture.sha256_bytes(report_payload),
            "size_bytes": len(report_payload),
        },
    }


def test_failed_ncu_handoff_mutation_retires_failure_provenance(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    failure_root = root / "failure"
    failure_root.mkdir()
    report = cell / "run_0_launch_set.ncu-rep"
    real_publish = capture.immutable.publish_regular_file

    def publish_with_handoff_mutation(*args, **kwargs):
        after_commit = kwargs["after_commit"]

        def mutate_then_close():
            replacement = report.with_name("replacement.ncu-rep")
            replacement.write_bytes(report.read_bytes())
            replacement.replace(report)
            after_commit()

        kwargs["after_commit"] = mutate_then_close
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        capture.immutable,
        "publish_regular_file",
        publish_with_handoff_mutation,
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(
        capture.CaptureError,
        match="transactionally publish failure provenance",
    ):
        capture._write_failure_provenance(
            failure_root,
            root=root,
            phase="ncu",
            row=row,
            command=["/fixture/target"],
            live=_nvbit_live(),
            environment_payload=b"{}\n",
            default_environment={"LANG": "C"},
            capture_producer_sha256="c" * 64,
            containment_supervisor_sha256="d" * 64,
            phase_tool_sha256="e" * 64,
            action_executor_sha256="f" * 64,
            error=capture.CaptureError("synthetic postprocess failure"),
        )

    assert not (failure_root / "failure_provenance.json").exists()
    retired = [
        path for path in failure_root.iterdir()
        if path.name.startswith(".failed-regular-")
    ]
    assert len(retired) == 1


def test_late_failure_stream_retires_failure_provenance(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    failure_root = root / "failure"
    failure_root.mkdir(parents=True)
    initial_stream = failure_root / "initial.stdout.bin"
    initial_stream.write_bytes(b"initial")
    provenance = failure_root / "failure_provenance.json"
    real_verify = capture.immutable.RetainedRegular.verify_unchanged

    def verify_with_late_stream(retained):
        if retained.path == initial_stream and provenance.exists():
            late_stream = failure_root / "late.stderr.bin"
            if not late_stream.exists():
                late_stream.write_bytes(b"late")
        real_verify(retained)

    monkeypatch.setattr(
        capture.immutable.RetainedRegular,
        "verify_unchanged",
        verify_with_late_stream,
    )
    monkeypatch.setattr(capture, "_boot_id_sha256", lambda: "b" * 64)

    with pytest.raises(
        capture.CaptureError,
        match="transactionally publish failure provenance",
    ):
        capture._write_failure_provenance(
            failure_root,
            root=root,
            phase="ncu",
            row=row,
            command=["/fixture/target"],
            live=_nvbit_live(),
            environment_payload=b"{}\n",
            default_environment={"LANG": "C"},
            capture_producer_sha256="c" * 64,
            containment_supervisor_sha256="d" * 64,
            phase_tool_sha256="e" * 64,
            action_executor_sha256="f" * 64,
            error=capture.CaptureError("synthetic capture failure"),
        )

    assert not (failure_root / "failure_provenance.json").exists()
    retired = [
        path for path in failure_root.iterdir()
        if path.name.startswith(".failed-regular-")
    ]
    assert len(retired) == 1


def test_ncu_handoff_is_imported_and_published_as_singleton(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    csv_text = _ncu_details_csv(row)
    observed = {}

    def imported(expected, report_path, **kwargs):
        observed["expected"] = expected
        observed["report_path"] = report_path
        observed["containment"] = kwargs["require_cgroup_containment"]
        return (
            capture.parse_ncu_metrics.ImportedReportCsv(
                text=csv_text,
                report_sha256=capture.sha256_bytes(report_payload),
                report_size_bytes=len(report_payload),
            ),
            None,
        )

    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_report_csv",
        imported,
    )
    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"_Zfixture_kernel"},
    )

    capture._postprocess_ncu_handoff(
        root=root,
        row=row,
        live=_nvbit_live(),
        expected_ncu_identity=identity,
    )

    assert observed == {
        "expected": identity,
        "report_path": cell / "run_0_launch_set.ncu-rep",
        "containment": True,
    }
    assert {path.name for path in cell.iterdir()} == set(
        capture._NCU_COMPLETE_ARTIFACT_NAMES
    )
    sidecar = json.loads((cell / "label_0_0.json").read_text())
    assert sidecar["cache_hit_rate_l2_sector"] == 0.5
    assert sidecar["cache_hit_rate_l1tex_sector"] == 0.25
    assert sidecar["measurement_protocol"] == {
        "ncu_clock_control": "none",
        "ncu_cache_control": "all",
        "ncu_replay_mode": "kernel",
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_profile_from_start": True,
        "ncu_launch_skip": 0,
        "ncu_launch_count": 1,
        "host_gpu_index": 0,
        "host_persistence_mode": "Enabled",
        "host_sm_clock_mhz": capture.SM_CLOCK_MHZ,
        "host_mem_clock_mhz": capture.MEM_CLOCK_MHZ,
    }
    assert sidecar["unit_tagged_metrics"] == [
        {
            "ncu_metric": "gpc__cycles_elapsed.max",
            "unit": "cycles",
            "value": 123.0,
        },
        {
            "ncu_metric": "gpu__time_duration.sum",
            "unit": "ns",
            "value": 456.0,
        },
    ]
    observation = json.loads(
        (cell / "run_0_launch_set.launch_observation.json").read_text()
    )
    assert observation["report_sha256"] == capture.sha256_bytes(report_payload)
    assert observation["identity"]["Kernel Name"] == "_Zfixture_kernel"


def test_ncu_handoff_generation_stays_retained_until_receipt_commit(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_report_csv",
        lambda *_args, **_kwargs: (
            capture.parse_ncu_metrics.ImportedReportCsv(
                text=_ncu_details_csv(row),
                report_sha256=capture.sha256_bytes(report_payload),
                report_size_bytes=len(report_payload),
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"_Zfixture_kernel"},
    )
    report = cell / "run_0_launch_set.ncu-rep"

    with pytest.raises(
        capture.immutable.ImmutableArtifactError,
        match="changed|replaced|identity|generation",
    ):
        with contextlib.ExitStack() as generation_stack:
            capture._postprocess_ncu_handoff(
                root=root,
                row=row,
                live=_nvbit_live(),
                expected_ncu_identity=identity,
                generation_stack=generation_stack,
            )
            replacement = report.with_name("replacement.ncu-rep")
            replacement.write_bytes(report_payload)
            replacement.replace(report)


@pytest.mark.parametrize(
    "artifact_name",
    (
        "run_0_launch_set.launch_observation.json",
        "label_0_0.json",
    ),
)
def test_ncu_generated_artifact_stays_retained_until_receipt_commit(
    tmp_path: Path,
    monkeypatch,
    artifact_name: str,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, report_payload = _write_ncu_handoff(root, row)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_report_csv",
        lambda *_args, **_kwargs: (
            capture.parse_ncu_metrics.ImportedReportCsv(
                text=_ncu_details_csv(row),
                report_sha256=capture.sha256_bytes(report_payload),
                report_size_bytes=len(report_payload),
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        capture.admission,
        "_sass_kernel_names",
        lambda _row: {"_Zfixture_kernel"},
    )

    with pytest.raises(
        capture.immutable.ImmutableArtifactError,
        match="changed|replaced|identity|generation",
    ):
        with contextlib.ExitStack() as generation_stack:
            capture._postprocess_ncu_handoff(
                root=root,
                row=row,
                live=_nvbit_live(),
                expected_ncu_identity=identity,
                generation_stack=generation_stack,
            )
            artifact = cell / artifact_name
            replacement = artifact.with_name(f"replacement-{artifact.name}")
            replacement.write_bytes(artifact.read_bytes())
            replacement.replace(artifact)


def test_ncu_handoff_rejects_unexpected_namespace_before_import(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    cell, _report_payload = _write_ncu_handoff(root, row)
    (cell / "unexpected.sidecar.json").write_text("{}\n")
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_report_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "unexpected namespace must fail before report import"
        ),
    )

    with pytest.raises(capture.CaptureError, match="namespace differs"):
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )


def test_ncu_handoff_rejects_imported_report_identity_drift(
    tmp_path: Path,
    monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    root = tmp_path / "capture"
    _cell, report_payload = _write_ncu_handoff(root, row)
    imported = capture.parse_ncu_metrics.ImportedReportCsv(
        text=_ncu_details_csv(row),
        report_sha256="0" * 64,
        report_size_bytes=len(report_payload),
    )
    monkeypatch.setattr(
        capture.parse_ncu_metrics,
        "_import_report_csv",
        lambda *_args, **_kwargs: (imported, None),
    )

    with pytest.raises(capture.CaptureError, match="identity differs"):
        capture._postprocess_ncu_handoff(
            root=root,
            row=row,
            live=_nvbit_live(),
            expected_ncu_identity=capture._ncu_identity_from_authority(
                _ncu_authority()
            ),
        )


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (0, b"", capture._NCU_HANDOFF_STDERR),
        (3, b"unexpected", capture._NCU_HANDOFF_STDERR),
        (3, b"", capture._NCU_HANDOFF_STDERR.rstrip(b"\n")),
        (3, b"", capture._NCU_HANDOFF_STDERR + b"extra\n"),
        (4, b"", capture._NCU_HANDOFF_STDERR),
    ],
)
def test_ncu_handoff_accepts_only_exact_driver_result(
    returncode: int,
    stdout: bytes,
    stderr: bytes,
):
    assert not capture._capture_protocol_succeeded(
        "ncu",
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    assert capture._capture_protocol_succeeded(
        "ncu",
        returncode=3,
        stdout=b"",
        stderr=capture._NCU_HANDOFF_STDERR,
    )


def test_ncu_authority_requires_launcher_inner_path_binding():
    authority = _ncu_authority()
    authority["resolved_path"] = str(
        Path(authority["launcher_resolved_path"]).parent / "target/other/ncu"
    )
    with pytest.raises(capture.CaptureError, match="paths differ"):
        capture._ncu_identity_from_authority(authority)


def test_ncu_probe_records_exact_descriptor_output(monkeypatch):
    authority = _ncu_authority()
    monkeypatch.setattr(
        capture.campaign_preflight,
        "probe_ncu",
        lambda _pins, **_kwargs: dict(authority),
    )
    version_stdout = (
        "NVIDIA (R) Nsight Compute Command Line Profiler\n"
        "Copyright (c) 2018-2026 NVIDIA Corporation\n"
        + authority["version_line"]
        + "\n"
    )
    observed_calls = []

    def execute(*args, **kwargs):
        observed_calls.append((args, kwargs))
        return capture.ncu_executor.NcuActionResult(
            0,
            version_stdout.encode(),
            b"",
        )

    monkeypatch.setattr(
        capture.ncu_executor,
        "execute_ncu_action",
        execute,
    )
    observed_authority, stdout = capture._probe_ncu_authority()
    assert observed_authority == authority
    assert stdout == version_stdout
    assert observed_calls[0][1]["require_cgroup_containment"] is True


def test_ncu_probe_rejects_unrecorded_banner_output(monkeypatch):
    authority = _ncu_authority()
    monkeypatch.setattr(
        capture.campaign_preflight,
        "probe_ncu",
        lambda _pins, **_kwargs: dict(authority),
    )
    version_stdout = (
        "NVIDIA (R) Nsight Compute Command Line Profiler\n"
        "Copyright (c) 2018-2026 NVIDIA Corporation\n"
        + authority["version_line"]
        + "\n"
    )
    monkeypatch.setattr(
        capture.ncu_executor,
        "execute_ncu_action",
        lambda *_args, **_kwargs: capture.ncu_executor.NcuActionResult(
            0,
            (version_stdout + "unrecorded banner\n").encode(),
            b"",
        ),
    )
    with pytest.raises(capture.CaptureError, match="exact authority"):
        capture._probe_ncu_authority()


def test_ncu_capture_uses_descriptor_bound_supervision(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555,
    )
    observed = {}

    def execute(expected, expected_driver, command, **kwargs):
        observed.update({
            "expected": expected,
            "driver": expected_driver,
            "command": command,
            "environment": kwargs["environment"].as_dict(),
            "working_directory": kwargs["working_directory"],
            "auxiliary_descriptors": kwargs["auxiliary_descriptors"],
            "require_cgroup_containment": kwargs["require_cgroup_containment"],
        })
        return capture.ncu_executor.NcuActionResult(
            0,
            b"",
            b"",
            process_id=4321,
            started_at_utc="2026-08-02T00:00:00Z",
            containment=_containment(),
        )

    monkeypatch.setattr(
        capture.ncu_executor, "execute_ncu_supervised_executable", execute,
    )
    env = {
        "CUDA_INJECTION64_PATH": "/hostile/tracer.so",
        "LD_PRELOAD": "/hostile/preload.so",
        "LD_SOURCE_COMMIT_OVERRIDE": capture.PINNED_COMMIT,
        "TMPDIR": str(tmp_path),
    }
    collection_environment = capture._label_collection_environment(
        env, cwd=tmp_path
    )
    result = capture._run_ncu_capture_command(
        _ncu_command(identity),
        identity,
        collection_environment,
        driver,
        _retained_target(),
        tmp_path,
    )
    assert result[0:2] == (0, 4321)
    assert observed["expected"] is identity
    assert observed["driver"] is driver
    command = observed["command"]
    separator = command.index("--")
    assert command[separator - 2:separator] == [
        "--ncu-exec-path",
        capture.ncu_executor.AUTHORIZED_NCU_PROC_PATH,
    ]
    assert command[command.index("--binary") + 1] == _retained_target().proc_path
    assert command[separator + 1] == "128"
    assert observed["auxiliary_descriptors"] == (_retained_target().descriptor,)
    assert observed["working_directory"] == tmp_path
    assert observed["require_cgroup_containment"] is True
    collection_env = observed["environment"]
    assert collection_env["LD_SOURCE_COMMIT_OVERRIDE"] == capture.PINNED_COMMIT
    assert collection_env["TMPDIR"] == str(tmp_path)
    assert "CUDA_INJECTION64_PATH" not in collection_env
    assert "LD_PRELOAD" not in collection_env


def test_generated_ncu_command_substitutes_binary_option(
    tmp_path: Path, monkeypatch,
):
    row, _roster = capture._row_for(MATRIX_KEY, VARIANT_ID)
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    binary = capture.ROOT / row["static_launch_inventory"]["binary_path"]
    retained_target = SimpleNamespace(
        path=binary,
        proc_path="/proc/self/fd/91",
        descriptor=91,
    )
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555,
    )
    observed = {}

    def execute(_expected, _expected_driver, command, **_kwargs):
        observed["command"] = command
        return capture.ncu_executor.NcuActionResult(
            0,
            b"",
            b"",
            process_id=4321,
            started_at_utc="2026-08-02T00:00:00Z",
            containment=_containment(),
        )

    monkeypatch.setattr(
        capture.ncu_executor, "execute_ncu_supervised_executable", execute,
    )
    environment = capture._label_collection_environment({}, cwd=binary.parent)
    capture._run_ncu_capture_command(
        capture.admission._expected_ncu_command(tmp_path, row, identity),
        identity,
        environment,
        driver,
        retained_target,
        binary.parent,
    )

    command = observed["command"]
    assert command[command.index("--binary") + 1] == retained_target.proc_path
    assert command[command.index("--") + 1:] == [
        str(value) for value in row["shape"]["actual_args"]
    ]


def test_ncu_capture_failure_canonicalizes_target_descriptor(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555,
    )
    environment = capture._label_collection_environment({}, cwd=tmp_path)

    def fail(_expected, _expected_driver, command, **kwargs):
        error = capture.ncu_executor.NcuActionExecutorError(
            capture.ncu_executor.NCU_ACTION_TIMEOUT,
            "synthetic supervised timeout",
        )
        error.process_id = 4321
        error.started_at_utc = "2026-08-02T00:00:00Z"
        error.process_role = "ncu_supervised_executable"
        error.process_command = tuple(
            identity.resolved_path
            if item is capture.ncu_executor.AUTHORIZED_NCU_PROC_PATH
            else str(item)
            for item in command
        )
        error.process_environment = kwargs["environment"].as_dict()
        error.process_tool_sha256 = driver.executable_sha256
        raise error

    monkeypatch.setattr(
        capture.ncu_executor, "execute_ncu_supervised_executable", fail,
    )
    with pytest.raises(capture.CaptureError, match="descriptor-bound NCU") as caught:
        capture._run_ncu_capture_command(
            _ncu_command(identity),
            identity,
            environment,
            driver,
            _retained_target(),
            tmp_path,
        )

    assert _retained_target().proc_path not in caught.value.process_command
    assert str(_retained_target().path) in caught.value.process_command
    assert caught.value.process_environment == environment.as_dict()


def test_ncu_capture_rejects_command_authority_drift(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    command = _ncu_command(identity)
    command[command.index("--launch-count") + 1] = "2"
    monkeypatch.setattr(
        capture.ncu_executor,
        "execute_ncu_supervised_executable",
        lambda *_args, **_kwargs: pytest.fail("drifted command must not run"),
    )
    environment = capture._label_collection_environment({}, cwd=tmp_path)
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555
    )
    with pytest.raises(capture.CaptureError, match="--launch-count"):
        capture._run_ncu_capture_command(
            command, identity, environment, driver, _retained_target(), tmp_path
        )


def test_ncu_capture_rejects_profile_from_start_off(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    command = _ncu_command(identity)
    command[command.index("--profile-from-start") + 1] = "off"
    monkeypatch.setattr(
        capture.ncu_executor,
        "execute_ncu_supervised_executable",
        lambda *_args, **_kwargs: pytest.fail("inactive profiler command must not run"),
    )
    environment = capture._label_collection_environment({}, cwd=tmp_path)
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555
    )
    with pytest.raises(capture.CaptureError, match="--profile-from-start"):
        capture._run_ncu_capture_command(
            command, identity, environment, driver, _retained_target(), tmp_path
        )


def test_label_driver_identity_requires_independent_digest(monkeypatch):
    monkeypatch.setattr(capture.admission, "LABEL_DRIVER_SHA256", "0" * 64)
    with pytest.raises(capture.CaptureError, match="pinned capture authority"):
        capture._label_driver_identity()


def test_descriptor_executed_label_driver_resolves_repository_siblings(
    tmp_path: Path,
):
    ncu = tmp_path / "ncu"
    ncu.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        "'NVIDIA (R) Nsight Compute Command Line Profiler' "
        "'Copyright (c) 2018-2026 NVIDIA Corporation' "
        "'Version 2026.1.0.0 (build synthetic)'\n"
    )
    ncu.chmod(0o555)
    ncu_identity = capture.ncu_executor.ExpectedNcuIdentity(
        str(ncu),
        capture.sha256_bytes(ncu.read_bytes()),
        "Version 2026.1.0.0 (build synthetic)",
    )
    driver_identity = capture._label_driver_identity()
    environment = capture._label_collection_environment(
        {}, cwd=capture.ROOT
    )
    result = capture.ncu_executor.execute_ncu_supervised_executable(
        ncu_identity,
        driver_identity,
        [
            str(capture.LABEL_DRIVER),
            "--ncu-exec-path",
            capture.ncu_executor.AUTHORIZED_NCU_PROC_PATH,
        ],
        timeout_seconds=2,
        environment=environment,
        working_directory=capture.ROOT,
    )
    assert result.returncode == 2
    assert b"SOURCE_PATH_INVALID" not in result.stderr
    assert b"local-label driver" in result.stderr


def test_descriptor_failure_streams_are_retained(
    tmp_path: Path, monkeypatch,
):
    identity = capture._ncu_identity_from_authority(_ncu_authority())
    driver = capture.ncu_executor.ExpectedExecutableIdentity(
        str(capture.LABEL_DRIVER), "d" * 64, 0o555
    )
    environment = capture._label_collection_environment({}, cwd=tmp_path)
    failure = capture.ncu_executor.NcuActionExecutorError(
        capture.ncu_executor.NCU_ACTION_TIMEOUT,
        "synthetic timeout",
    )
    failure.stdout = b"bounded stdout"
    failure.stderr = b"bounded stderr"
    failure.process_id = 4321
    failure.started_at_utc = "2026-08-02T00:00:00Z"

    def execute(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        capture.ncu_executor, "execute_ncu_supervised_executable", execute
    )
    failure_root = tmp_path / "retained-failure"
    with pytest.raises(capture.CaptureError, match="descriptor-bound") as caught:
        capture._run_ncu_capture_command(
            _ncu_command(identity),
            identity,
            environment,
            driver,
            _retained_target(),
            tmp_path,
            failure_root=failure_root,
        )
    assert caught.value.process_id == 4321
    assert caught.value.started_at_utc == "2026-08-02T00:00:00Z"
    assert (failure_root / "ncu_executor_failure.stdout.bin").read_bytes() == (
        b"bounded stdout"
    )
    assert (failure_root / "ncu_executor_failure.stderr.bin").read_bytes() == (
        b"bounded stderr"
    )


def test_nvbit_runtime_environment_excludes_hostile_ambient_controls(
    tmp_path: Path, monkeypatch,
):
    row, _roster_row = capture._row_for(MATRIX_KEY, VARIANT_ID)
    expected = capture.admission._expected_nvbit_environment(tmp_path, row)
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"synthetic tracer")
    pb_lib = capture.NVBIT_PB_LIB
    monkeypatch.setenv("LD_PRELOAD", "/tmp/hostile.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/hostile-python")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(
        capture.nvbit_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: capture.subprocess.CompletedProcess(
            [],
            0,
            stdout=f"libprotobuf.so.1 => {pb_lib / 'libprotobuf.so.1'} (0x0)\n",
            stderr="",
        ),
    )

    observed = capture.nvbit_runtime.tracer_runtime_env(
        tracer,
        base_env=expected,
        pb_lib=pb_lib,
    )

    assert observed == expected
    assert set(observed) == {
        "CUDA_INJECTION64_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "PWD",
        "TRACES_FOLDER",
        "USER_DEFINED_FOLDERS",
    }
    assert observed["CUDA_VISIBLE_DEVICES"] == "0"
    assert "LD_PRELOAD" not in observed
    assert "PYTHONPATH" not in observed


def test_environment_provenance_names_trace_folder_controls():
    document = capture._environment_document(
        {"pin": "static"}, {
            "check": "live",
            "smi_stdout": "smi",
            "competing_process_stdout": "",
            "identity_stdout": "identity",
            "nvcc_stdout": "nvcc",
            "ncu_stdout": "ncu",
        },
        {"TRACES_FOLDER": "/capture/traces", "USER_DEFINED_FOLDERS": "1"},
        capture_kind="nvbit",
    )
    assert document["environment"]["TRACES_FOLDER"] == "/capture/traces"
    assert document["environment"]["USER_DEFINED_FOLDERS"] == "1"
