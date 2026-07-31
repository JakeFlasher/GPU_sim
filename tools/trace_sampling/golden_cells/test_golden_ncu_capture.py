"""Focused authority tests for the golden-cell NCU adapter."""
from __future__ import annotations

from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor
from tools.trace_sampling.golden_cells import golden_capture_lock
from tools.trace_sampling.golden_cells import golden_ncu_receipt
from tools.trace_sampling.golden_cells import retained_nvbit_launch
from tools.trace_sampling.golden_cells import retained_nvcc_compile
from tools.trace_sampling.golden_cells import golden_ncu_capture as golden


CSV = (
    "ID,Kernel Name,Block Size,Grid Size,Device,Metric Name,Metric Unit,Metric Value\n"
    "1,_Z15syntheticKernelv,\"(1024, 1, 1)\",\"(36864, 1, 1)\",0,lts__t_sector_hit_rate.pct,%,50.0\n"
).encode()


def _paths(tmp_path: Path) -> dict[str, Path]:
    cell = tmp_path / "results" / "nvbit_capture_staging" / "golden_cells" / "gs_stream_copy"
    cell.mkdir(parents=True)
    binary = cell / "gs_stream_copy.bin"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o555)
    trace = cell / "trace.staged.1"
    trace.mkdir()
    (trace / "fixture").write_bytes(b"trace fixture")
    return {
        "cell": cell,
        "binary": binary,
        "trace": trace,
        "csv": cell / "ncu_l2.csv.staged.1",
        "stderr": cell / "ncu_l2.stderr.staged.1",
        "observation": cell / "ncu_launch_observation.json.staged.1",
        "run": cell / "ncu_run_manifest.json.staged.1",
        "capture": cell / "ncu_capture_manifest.json.staged.1",
        "receipt": cell / "ncu_collection_receipt.json.staged.1",
        "published_trace": cell / "trace",
        "published_csv": cell / "ncu_l2.csv",
        "published_stderr": cell / "ncu_l2.stderr",
        "published_observation": cell / "ncu_launch_observation.json",
        "published_run": cell / "ncu_run_manifest.json",
        "published_capture": cell / "ncu_capture_manifest.json",
        "published_receipt": cell / "ncu_collection_receipt.json",
    }


def test_shell_routes_ncu_through_preflight_bound_adapter() -> None:
    script = (Path(__file__).with_name("capture_golden_cells.sh")).read_text()
    assert script.startswith("#!/bin/bash -p\n")
    assert "NVBIT_CAPTURE_PYTHON" not in script
    assert "NVBIT_CAPTURE_OUTPUT_ROOT" not in script
    assert "GOLDEN_NCU_PREFLIGHT:-" not in script
    assert '"$PYTHON" -I -S' in script
    assert "refusing ambient startup variable" in script
    assert "golden_ncu_capture.py" in script
    assert "--preflight" in script
    assert "--preflight-sha256" in script
    assert "--binary-sha256" in script
    assert "--validate-only" in script
    assert "publish_target_check" in script
    assert "verify_published_receipt" in script
    assert "publish_target_check \"$bin\" binary" in script
    assert script.index('publish_target_check "$bin" binary') < script.index('retained_nvcc_compile.py')
    assert "reject_ambient_cuda_selectors" in script
    assert "resolve_admitted_toolchain" in script
    assert "retained_nvcc_compile.py" in script
    assert "retained_nvbit_launch.py" in script
    assert "publish_directory" in script
    assert "publish_regular_file" in script
    assert "golden_capture_lock.py" in script
    assert "receipt_metadata.st_mode & 0o222" in script
    assert "retained_receipt.read_bytes()" in script
    assert "immutable.read_regular(receipt_path)" not in script
    assert 'int(target["mode"], 8)' in script
    assert "fsync_publication_path" in script
    assert "ncu --metrics" not in script
    assert "ncu --version" not in script


def test_privileged_shell_refuses_bash_startup_injection(tmp_path: Path) -> None:
    marker = tmp_path / "injected"
    startup = tmp_path / "bash-env"
    startup.write_text(f"touch {marker}\n", encoding="utf-8")
    script = Path(__file__).with_name("capture_golden_cells.sh")
    result = subprocess.run(
        (str(script), "--plan-only", "gs_stream_copy"),
        capture_output=True,
        text=True,
        env={"BASH_ENV": str(startup), "PATH": "/bin:/usr/bin"},
    )
    assert result.returncode == 2
    assert "refusing ambient startup variable: BASH_ENV" in result.stderr
    assert not marker.exists()


def test_capture_lock_uses_nonreplaceable_abstract_namespace(tmp_path: Path) -> None:
    name = golden_capture_lock._lock_name(tmp_path)
    owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    contender = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        owner.bind(name)
        golden_capture_lock._validate_socket_descriptor(owner.fileno(), name)
        with pytest.raises(OSError):
            contender.bind(name)
    finally:
        contender.close()
        owner.close()


def test_capture_lock_refuses_startup_injection_and_builds_closed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prohibited = (
        "BASH_ENV", "CUDA_INJECTION64_PATH", "GIT_EXEC_PATH", "LD_PRELOAD",
        "PYTHONPATH",
    )
    monkeypatch.setenv("PATH", "/tmp/fake-bin")
    for name in prohibited:
        monkeypatch.setenv(name, f"/tmp/{name.lower()}")
    with pytest.raises(
        golden_capture_lock.GoldenCaptureLockError,
        match="prohibited startup environment",
    ):
        golden_capture_lock._execution_environment(9)
    for name in golden_capture_lock._PROHIBITED_STARTUP_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    environment = golden_capture_lock._execution_environment(9)
    assert environment == {
        "GOLDEN_CAPTURE_LOCK_FD": "9",
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": "/bin:/usr/bin",
        "PYTHONNOUSERSITE": "1",
    }


def test_retained_helpers_reject_final_publication_destinations() -> None:
    cell = golden.STAGING_ROOT / "golden_cells" / "gs_stream_copy"
    with pytest.raises(retained_nvcc_compile.RetainedNvccError, match="staged"):
        retained_nvcc_compile._require_staging_output(
            cell / "gs_stream_copy.bin",
        )
    with pytest.raises(retained_nvbit_launch.RetainedNvbitError, match="staged"):
        retained_nvbit_launch._require_staging_trace(cell / "trace")


def test_preflight_hash_mismatch_refuses_before_schema_or_capture(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(golden.GoldenNcuCaptureError, match="SHA-256 mismatch"):
        golden._load_preflight(path, expected_sha256="0" * 64)


def test_checkout_validation_ignores_ambient_fake_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    pins = repo / "tools" / "local_labels" / "campaign_protocol_pins.json"
    pins.parent.mkdir(parents=True)
    pins.write_text("{}\n", encoding="utf-8")
    for arguments in (
        ("init", "-q"),
        ("config", "user.email", "fixture@example.invalid"),
        ("config", "user.name", "Fixture"),
        ("add", "."),
        ("commit", "-qm", "fixture"),
    ):
        subprocess.run(
            ("/usr/bin/git", "-C", str(repo), *arguments),
            check=True,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/bin:/usr/bin"},
        )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\nprintf '%s\\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    monkeypatch.setattr(golden, "REPO_ROOT", repo)
    with pytest.raises(golden.GoldenNcuCaptureError, match="git HEAD differs"):
        golden._validate_current_checkout({
            "git_head": "a" * 40,
            "pins_sha256": golden._sha256_file(pins),
        })


def test_writable_target_mode_is_refused_before_ncu(tmp_path: Path) -> None:
    binary = tmp_path / "target.bin"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    with immutable.open_retained_executable(binary) as retained:
        with pytest.raises(golden.GoldenNcuCaptureError, match="write-protected"):
            golden._retained_target_mode(retained)


def test_missing_preflight_refuses_before_ncu_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(golden, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(golden, "STAGING_ROOT", tmp_path / "results" / "nvbit_capture_staging")
    called: list[object] = []
    monkeypatch.setattr(
        golden.ncu_executor,
        "execute_ncu_action",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )
    with pytest.raises(golden.GoldenNcuCaptureError, match="preflight manifest"):
        golden.collect_golden_ncu(
            preflight_path=tmp_path / "missing-preflight.json",
            preflight_sha256="a" * 64,
            slug="gs_stream_copy",
            binary=paths["binary"],
            binary_sha256=golden._sha256_file(paths["binary"]),
            trace_dir=paths["trace"],
            csv_out=paths["csv"],
            stderr_out=paths["stderr"],
            observation_out=paths["observation"],
            run_manifest_out=paths["run"],
            capture_manifest_out=paths["capture"],
            receipt_out=paths["receipt"],
            published_trace_dir=paths["published_trace"],
            published_csv_out=paths["published_csv"],
            published_stderr_out=paths["published_stderr"],
            published_observation_out=paths["published_observation"],
            published_run_manifest_out=paths["published_run"],
            published_capture_manifest_out=paths["published_capture"],
            published_receipt_out=paths["published_receipt"],
        )
    assert called == []
    assert not paths["csv"].exists()


def test_collection_binds_identity_controls_and_published_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(golden, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(golden, "STAGING_ROOT", tmp_path / "results" / "nvbit_capture_staging")
    ncu_path = tmp_path / "ncu"
    ncu_path.write_bytes(b"ncu fixture")
    ncu_path.chmod(0o755)
    identity = ncu_action_executor.build_expected_ncu_identity(
        resolved_path=str(ncu_path),
        executable_sha256="a" * 64,
        version_line="Version 2026.1.0 (fixture)",
    )
    monkeypatch.setattr(
        golden,
        "_load_preflight",
        lambda path, expected_sha256: ("b" * 64, identity, {"schema_version": 3}),
    )
    monkeypatch.setattr(
        golden.trace_admission,
        "require_trace_admission",
        lambda _trace: SimpleNamespace(
            launch_count=1,
            threadblock_count=1,
            instruction_count=1,
            expected_record_count=1,
            admitted_record_count=1,
        ),
    )
    calls: list[tuple[object, ...]] = []

    def fake_execute(expected, arguments, *, timeout_seconds, auxiliary_descriptors):
        calls.append((expected, tuple(arguments), timeout_seconds, tuple(auxiliary_descriptors)))
        assert expected == identity
        assert tuple(arguments[:12]) == (
            "--metrics", golden.GOLDEN_METRIC,
            "--clock-control", "none",
            "--cache-control", "all",
            "--replay-mode", "kernel",
            "--launch-count", "1",
            "--csv", "--",
        )
        assert arguments[-1].startswith("/proc/self/fd/")
        assert auxiliary_descriptors
        return ncu_action_executor.NcuActionResult(0, CSV, b"")

    monkeypatch.setattr(golden.ncu_executor, "execute_ncu_action", fake_execute)
    receipt = golden.collect_golden_ncu(
        preflight_path=tmp_path / "preflight.json",
        preflight_sha256="b" * 64,
        slug="gs_stream_copy",
        binary=paths["binary"],
        binary_sha256=golden._sha256_file(paths["binary"]),
        trace_dir=paths["trace"],
        csv_out=paths["csv"],
        stderr_out=paths["stderr"],
        observation_out=paths["observation"],
        run_manifest_out=paths["run"],
        capture_manifest_out=paths["capture"],
        receipt_out=paths["receipt"],
        published_trace_dir=paths["published_trace"],
        published_csv_out=paths["published_csv"],
        published_stderr_out=paths["published_stderr"],
        published_observation_out=paths["published_observation"],
        published_run_manifest_out=paths["published_run"],
        published_capture_manifest_out=paths["published_capture"],
        published_receipt_out=paths["published_receipt"],
    )
    assert len(calls) == 1
    assert receipt["preflight_sha256"] == "b" * 64
    assert receipt["ncu_identity"] == {
        "resolved_path": str(ncu_path),
        "executable_sha256": "a" * 64,
        "version_line": "Version 2026.1.0 (fixture)",
    }
    assert receipt["command"][receipt["command"].index("--") + 1].startswith(
        "/proc/self/fd/"
    )
    assert receipt["supervision"]["execution_mode"] == "bounded_direct_ncu_v1"
    assert receipt["environment"] == ncu_action_executor.ncu_action_environment()
    assert receipt["target_binary"]["sha256"] == golden._sha256_file(paths["binary"])
    assert receipt["target_binary"]["mode"] == "0555"
    assert receipt["target_argument_binding"]["target_mode"] == "0555"
    assert receipt["trace_admission"]["admitted_record_count"] == 1
    assert receipt["trace_tree_bytes"] == len(b"trace fixture")
    assert receipt["launch_controls"]["launch_count"] == 1
    broken = dict(receipt)
    broken["status"] = "fail"
    broken.pop("receipt_seal_sha256")
    with pytest.raises(golden_ncu_receipt.GoldenReceiptError, match="fixed identity"):
        golden_ncu_receipt.validate_intrinsic_receipt(
            immutable.seal_model(broken, "receipt_seal_sha256")
        )
    with pytest.raises(golden_ncu_receipt.GoldenReceiptError, match="descriptor"):
        golden_ncu_receipt._command(
            (*receipt["command"][:-1], "/tmp/other-target.bin")
        )
    writable_target = dict(receipt)
    writable_target["target_binary"] = {
        **receipt["target_binary"], "mode": "0755",
    }
    writable_target.pop("receipt_seal_sha256")
    with pytest.raises(golden_ncu_receipt.GoldenReceiptError, match="binding"):
        golden_ncu_receipt.validate_intrinsic_receipt(
            immutable.seal_model(writable_target, "receipt_seal_sha256")
        )
    assert paths["csv"].read_bytes() == CSV
    assert paths["run"].exists() and paths["capture"].exists()


def test_no_replace_publication_preserves_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "trace.staged.1"
    destination = tmp_path / "trace"
    source.write_bytes(b"new")
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        immutable.rename_noreplace(source, destination)
    assert source.read_bytes() == b"new"
    assert destination.read_bytes() == b"existing"


def test_details_csv_payload_strips_recognized_preamble() -> None:
    payload = golden._details_csv_payload(
        b"==WARNING== cache control is disabled\n"
        b"==PROF== Connected to process\n" + CSV
    )
    assert payload == CSV.decode()


def test_details_csv_payload_rejects_unrecognized_preamble() -> None:
    with pytest.raises(golden.GoldenNcuCaptureError, match="unexpected preamble"):
        golden._details_csv_payload(b"unexpected\n" + CSV)
