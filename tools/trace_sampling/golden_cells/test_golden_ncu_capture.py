"""Focused authority tests for the golden-cell NCU adapter."""
from __future__ import annotations

from pathlib import Path
import json
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor
from tools.trace_sampling.golden_cells import golden_capture_lock
from tools.trace_sampling.golden_cells import golden_ncu_receipt
from tools.trace_sampling.golden_cells import retained_golden_publication
from tools.trace_sampling.golden_cells import retained_nvbit_launch
from tools.trace_sampling.golden_cells import retained_nvcc_compile
from tools.trace_sampling.golden_cells import golden_ncu_capture as golden


CSV = (
    "ID,Kernel Name,Block Size,Grid Size,Device,Metric Name,Metric Unit,Metric Value\n"
    "1,_Z15syntheticKernelv,\"(1024, 1, 1)\",\"(36864, 1, 1)\",0,lts__t_sector_hit_rate.pct,%,50.0\n"
).encode()


def _trace_launch(
    *,
    kernel_name: str = "_Z15syntheticKernelv___0",
    grid_dim: tuple[int, int, int] = (36864, 1, 1),
    block_dim: tuple[int, int, int] = (1024, 1, 1),
) -> SimpleNamespace:
    return SimpleNamespace(
        launch_index=0,
        device_id=0,
        stream_id=0,
        kernel_id=1,
        kernel_name=kernel_name,
        grid_x=grid_dim[0],
        grid_y=grid_dim[1],
        grid_z=grid_dim[2],
        block_x=block_dim[0],
        block_y=block_dim[1],
        block_z=block_dim[2],
    )


def _trace_admission_fixture(
    *, launches: tuple[SimpleNamespace, ...] | None = None,
) -> SimpleNamespace:
    selected = (_trace_launch(),) if launches is None else launches
    return SimpleNamespace(
        launch_count=1,
        threadblock_count=1,
        instruction_count=1,
        expected_record_count=1,
        admitted_record_count=1,
        launches=selected,
    )


def _launch_observation(csv_payload: bytes = CSV) -> dict[str, object]:
    report_sha256 = golden._sha256_bytes(csv_payload)
    document = golden.parse_ncu_metrics.build_launch_observation_document(
        csv_text=csv_payload.decode(),
        report_sha256=report_sha256,
        requested_launch_skip=0,
        requested_launch_count=1,
    )
    golden.parse_ncu_metrics.validate_launch_observation_document(
        document,
        requested_launch_skip=0,
        requested_launch_count=1,
        report_sha256=report_sha256,
    )
    return document


def _paths(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "results" / "nvbit_capture_staging" / "golden_cells"
    cell = root / f"gs_stream_copy.staged.1.{('a' * 32)}"
    published_cell = root / "gs_stream_copy"
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
        "published_binary": published_cell / "gs_stream_copy.bin",
        "published_trace": published_cell / "trace",
        "published_csv": published_cell / "ncu_l2.csv",
        "published_stderr": published_cell / "ncu_l2.stderr",
        "published_observation": published_cell / "ncu_launch_observation.json",
        "published_run": published_cell / "ncu_run_manifest.json",
        "published_capture": published_cell / "ncu_capture_manifest.json",
        "published_receipt": published_cell / "ncu_collection_receipt.json",
    }


def _publication_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Path:
    root = tmp_path / "results" / "nvbit_capture_staging" / "golden_cells"
    root.mkdir(parents=True)
    monkeypatch.setattr(retained_golden_publication, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(retained_golden_publication, "_GOLDEN_ROOT", root)
    return root


def _write_nvbit_generation_handoff(trace_dir: Path) -> bytes:
    metadata = trace_dir.stat()
    document = immutable.seal_model(
        {
            "schema_version": 1,
            "artifact_kind": "golden_nvbit_launch_attempt",
            "status": "succeeded",
            "trace_generation": {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
            },
        },
        field="attempt_seal_sha256",
    )
    payload = immutable.canonical_json_bytes(document)
    (trace_dir / "nvbit_launch_attempt.json").write_bytes(payload)
    return payload


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
    assert "golden_ncu_capture._trace_admission_model" in script
    assert "parse_ncu_metrics.validate_launch_observation_document" in script
    assert "golden_ncu_capture._bind_ncu_observation_to_trace" in script
    assert "published golden NCU/trace launch binding differs" in script
    assert "publish_target_check \"$bin\" binary" in script
    assert script.index('publish_target_check "$bin" binary') < script.index(
        "retained_nvcc_compile.py"
    )
    assert "reject_ambient_cuda_selectors" in script
    assert "resolve_admitted_toolchain" in script
    assert "retained_nvcc_compile.py" in script
    assert '--source "$cu" --preflight "$PREFLIGHT"' in script
    assert '--preflight-sha256 "$PREFLIGHT_SHA256" --output' in script
    assert "retained_nvbit_launch.py" in script
    assert "retained_golden_publication.py" in script
    assert "secrets.token_hex(16)" in script
    assert 'cell="$GOLDEN/${slug}.staged.${ATTEMPT_ID}"' in script
    assert 'ncu_csv=""; ncu_stderr=""; ncu_observation=""' in script
    assert "process-cell" in script
    assert "commit-cell --stage" not in script
    assert 'trace_admission.py" "$staged"' not in script
    assert "tools/nvbit_tracer/tracer_tool/tracer_tool.so" not in script
    assert 'injection["tracer_source_path"]' in script
    assert "external_non_git_retained_descriptor_v1" in script
    assert "--tracer-source-kind" in script
    assert 'if (( ! PLAN_ONLY )) && [[ -z "$PREFLIGHT"' in script
    assert "runtime_env.py" not in script
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
    with pytest.raises(retained_nvcc_compile.RetainedNvccError, match="attempt cell"):
        retained_nvcc_compile._require_staging_output(
            cell / "gs_stream_copy.bin",
        )
    with pytest.raises(retained_nvbit_launch.RetainedNvbitError, match="staged"):
        retained_nvbit_launch._require_staging_trace(cell / "trace")


def test_retained_compiler_accepts_only_nonce_attempt_cell_output() -> None:
    root = retained_nvcc_compile._GOLDEN_STAGING_ROOT / "golden_cells"
    attempt = root / f"gs_stream_copy.staged.17.{('a' * 32)}"
    output = attempt / "gs_stream_copy.bin.staged.17"
    assert retained_nvcc_compile._require_staging_output(output) == output
    source = Path(retained_nvcc_compile.__file__).read_text(encoding="utf-8")
    assert "ulimit -f" in source
    assert retained_nvcc_compile._MAX_BINARY_BYTES == 1024 * 1024 * 1024


def test_retained_compiler_file_size_launcher_enforces_limit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bounded-output"
    program = (
        "import os; "
        f"fd=os.open({str(output)!r}, os.O_WRONLY|os.O_CREAT, 0o600); "
        "written=os.write(fd, b'x' * 4096); "
        "raise SystemExit(0 if written == 4096 else 3)"
    )
    command = retained_nvcc_compile._file_size_limited_command(
        bash_executable=str(retained_nvcc_compile._RETAINED_BASH),
        target_executable=sys.executable,
        target_arguments=("-I", "-S", "-c", program),
        max_file_blocks=1,
    )
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/bin:/usr/bin"},
        cwd="/",
    )
    assert result.returncode != 0
    assert output.stat().st_size <= 1024


def test_retained_compiler_rejects_legacy_preflight_schema() -> None:
    class LegacyPreflight:
        def read_bytes(self):
            return json.dumps({
                "schema_version": 3,
                "git_head": "a" * 40,
                "working_tree_clean": True,
                "capture_admitted": True,
            }).encode()

        def verify_unchanged(self):
            return None

    class UnusedSource:
        def verify_unchanged(self):
            return None

    with pytest.raises(
        retained_nvcc_compile.RetainedNvccError,
        match="requires preflight schema_version 4",
    ):
        retained_nvcc_compile._require_preflight_source_authority(
            preflight=LegacyPreflight(),
            source=UnusedSource(),
            retained_git=object(),
        )


def test_retained_compiler_binds_source_to_preflight_clean_head_bytes() -> None:
    source_path = (
        retained_nvcc_compile.REPO_ROOT
        / "tools"
        / "trace_sampling"
        / "golden_cells"
        / "gs_stream_copy.cu"
    )
    head = subprocess.run(
        ("/usr/bin/git", "-C", str(retained_nvcc_compile.REPO_ROOT),
         "rev-parse", "--verify", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    preflight_payload = json.dumps({
        "schema_version": 4,
        "git_head": head,
        "working_tree_clean": True,
        "capture_admitted": True,
    }).encode()

    class FakePreflight:
        def read_bytes(self):
            return preflight_payload

        def verify_unchanged(self):
            return None

    class AlteredSource:
        path = source_path

        def read_bytes(self):
            return b"altered source bytes\n"

        def verify_unchanged(self):
            return None

    with immutable.open_retained_executable(
        Path("/usr/bin/git")
    ) as retained_git:
        with immutable.open_retained_regular(source_path) as retained_source:
            assert retained_nvcc_compile._require_preflight_source_authority(
                preflight=FakePreflight(),
                source=retained_source,
                retained_git=retained_git,
            ) == head
        with pytest.raises(
            retained_nvcc_compile.RetainedNvccError,
            match="differs from preflight clean-HEAD bytes",
        ):
            retained_nvcc_compile._require_preflight_source_authority(
                preflight=FakePreflight(),
                source=AlteredSource(),
                retained_git=retained_git,
            )


def test_golden_ncu_rejects_visible_cell_as_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "results" / "nvbit_capture_staging" / "golden_cells"
    cell = root / "gs_stream_copy"
    cell.mkdir(parents=True)
    monkeypatch.setattr(
        golden, "STAGING_ROOT", tmp_path / "results" / "nvbit_capture_staging",
    )
    with pytest.raises(golden.GoldenNcuCaptureError, match="selected cell"):
        golden._validate_cell_paths(
            slug="gs_stream_copy",
            binary=cell / "gs_stream_copy.bin",
            published_binary=cell / "gs_stream_copy.bin",
            trace_dir=cell / "trace.staged.1",
            published_trace_dir=cell / "trace",
            staged_artifacts=(),
            published_artifacts=(),
        )


def test_publication_helper_rejects_paths_outside_golden_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = _publication_root(monkeypatch, tmp_path)
    final = root / "gs_stream_copy"
    outside = tmp_path / f"gs_stream_copy.staged.19.{('b' * 32)}"
    with pytest.raises(
        retained_golden_publication.GoldenPublicationError,
        match="escapes guarded staging",
    ):
        retained_golden_publication.prepare_cell_stage(outside, final)


def test_process_trace_cli_does_not_require_ncu_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    staged = tmp_path / "trace.staged.1"
    published = tmp_path / "trace"
    calls: list[tuple[Path, Path, object, object]] = []
    monkeypatch.setattr(
        retained_golden_publication,
        "process_trace_generation",
        lambda source, destination, *, derive_ncu, prepare_handoff: calls.append(
            (source, destination, derive_ncu, prepare_handoff)
        ),
    )
    cell_stage = tmp_path / "gs_stream_copy.staged.1" / "cell"
    cell_final = tmp_path / "gs_stream_copy"
    assert retained_golden_publication.main([
        "process-cell", "--staged-trace", str(staged),
        "--trace-out", str(published),
        "--cell-stage", str(cell_stage),
        "--cell-final", str(cell_final),
    ]) == 0
    assert len(calls) == 1
    assert calls[0][:3] == (staged, published, None)
    assert callable(calls[0][3])


def test_preflight_hash_mismatch_refuses_before_schema_or_capture(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(golden.GoldenNcuCaptureError, match="SHA-256 mismatch"):
        golden._load_preflight(path, expected_sha256="0" * 64)


def test_live_golden_collection_pins_active_preflight_v4_schema() -> None:
    payload = golden.PREFLIGHT_SCHEMA.read_bytes()
    schema = immutable.strict_json_loads(payload, label="preflight schema")
    assert golden._sha256_bytes(payload) == golden.PREFLIGHT_SCHEMA_SHA256
    assert schema["properties"]["schema_version"]["const"] == 4


def test_live_golden_collection_rejects_historical_v3_preflight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preflight.json"
    path.write_text('{"schema_version":3}\n', encoding="utf-8")
    with pytest.raises(
        golden.GoldenNcuCaptureError,
        match="live golden collection requires preflight schema_version 4",
    ):
        golden._load_preflight(
            path, expected_sha256=golden._sha256_file(path),
        )


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


def test_trace_admission_rejects_vacuous_count_only_launch_inventory() -> None:
    with pytest.raises(
        golden.GoldenNcuCaptureError,
        match="exactly one admitted launch",
    ):
        golden._trace_admission_model(
            _trace_admission_fixture(launches=()),
        )


def test_exact_ncu_and_admitted_trace_launch_binding_passes() -> None:
    trace_model = golden._trace_admission_model(_trace_admission_fixture())

    binding = golden._bind_ncu_observation_to_trace(
        _launch_observation(),
        trace_model,
        slug="gs_stream_copy",
    )

    assert binding["ncu_observation"]["kernel_name"] == (
        golden._nvbit_kernel_base(
            binding["admitted_trace_launch"]["kernel_name"]
        )
    )
    assert binding["ncu_observation"]["grid_dim"] == [36864, 1, 1]
    assert binding["ncu_observation"]["block_dim"] == [1024, 1, 1]


@pytest.mark.parametrize(
    ("trace_launch", "match"),
    (
        (_trace_launch(kernel_name="_Z17differentKernelv___0"), "kernel_name"),
        (
            _trace_launch(grid_dim=(18432, 2, 1)),
            "grid_dim",
        ),
        (
            _trace_launch(block_dim=(512, 2, 1)),
            "block_dim",
        ),
    ),
    ids=("kernel", "same-product-grid", "same-product-block"),
)
def test_ncu_binding_rejects_wrong_kernel_or_nonexact_geometry(
    trace_launch: SimpleNamespace,
    match: str,
) -> None:
    trace_model = golden._trace_admission_model(
        _trace_admission_fixture(launches=(trace_launch,)),
    )

    with pytest.raises(
        golden.GoldenNcuCaptureError,
        match=r"differs from exact admitted trace launch",
    ) as failure:
        golden._bind_ncu_observation_to_trace(
            _launch_observation(),
            trace_model,
            slug="gs_stream_copy",
        )
    assert match in str(failure.value)


@pytest.mark.parametrize(
    "kernel_name",
    ("_Z15syntheticKernelv", "_Z15syntheticKernelv___1"),
    ids=("missing-suffix", "ambiguous-variant"),
)
def test_ncu_binding_requires_unique_nvbit_variant_zero(
    kernel_name: str,
) -> None:
    trace_model = golden._trace_admission_model(
        _trace_admission_fixture(
            launches=(_trace_launch(kernel_name=kernel_name),)
        )
    )

    with pytest.raises(golden.GoldenNcuCaptureError, match="NVBit kernel"):
        golden._bind_ncu_observation_to_trace(
            _launch_observation(),
            trace_model,
            slug="gs_stream_copy",
        )


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
            published_binary=paths["published_binary"],
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
        lambda path, expected_sha256: ("b" * 64, identity, {"schema_version": 4}),
    )
    monkeypatch.setattr(
        golden.trace_admission,
        "require_trace_admission",
        lambda _trace: _trace_admission_fixture(),
    )
    calls: list[tuple[object, ...]] = []

    def fake_execute(
        expected,
        arguments,
        *,
        timeout_seconds,
        auxiliary_descriptors,
        require_cgroup_containment,
    ):
        calls.append((
            expected,
            tuple(arguments),
            timeout_seconds,
            tuple(auxiliary_descriptors),
            require_cgroup_containment,
        ))
        assert require_cgroup_containment is True
        assert expected == identity
        assert tuple(arguments[:14]) == (
            "--metrics", golden.GOLDEN_METRIC,
            "--clock-control", "none",
            "--cache-control", "all",
            "--replay-mode", "kernel",
            "--kernel-name-base", "mangled",
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
        published_binary=paths["published_binary"],
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
    assert receipt["trace_admission"]["launches"] == [{
        "launch_index": 0,
        "device_id": 0,
        "stream_id": 0,
        "kernel_id": 1,
        "kernel_name": "_Z15syntheticKernelv___0",
        "grid_dim": [36864, 1, 1],
        "block_dim": [1024, 1, 1],
    }]
    assert receipt["launch_binding"]["ncu_observation"] == {
        "report_launch_id": "1",
        "device_id": 0,
        "kernel_name": "_Z15syntheticKernelv",
        "grid_dim": [36864, 1, 1],
        "block_dim": [1024, 1, 1],
    }
    assert receipt["launch_binding"]["admitted_trace_launch"] == (
        receipt["trace_admission"]["launches"][0]
    )
    assert receipt["schema_version"] == 2
    assert receipt["kind"] == "golden_ncu_collection_receipt_v2"
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
    mismatched_launch = json.loads(json.dumps(receipt))
    mismatched_launch["launch_binding"]["ncu_observation"][
        "kernel_name"
    ] = "_Z17differentKernelv"
    mismatched_launch.pop("receipt_seal_sha256")
    with pytest.raises(
        golden_ncu_receipt.GoldenReceiptError,
        match="observation differs from admitted trace launch",
    ):
        golden_ncu_receipt.validate_intrinsic_receipt(
            immutable.seal_model(mismatched_launch, "receipt_seal_sha256")
        )
    counts_only = json.loads(json.dumps(receipt))
    counts_only["trace_admission"].pop("launches")
    counts_only.pop("receipt_seal_sha256")
    with pytest.raises(
        golden_ncu_receipt.GoldenReceiptError,
        match="trace admission fields differ",
    ):
        golden_ncu_receipt.validate_intrinsic_receipt(
            immutable.seal_model(counts_only, "receipt_seal_sha256")
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


def test_retained_trace_generation_rejects_staged_tree_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = _publication_root(monkeypatch, tmp_path)
    cell = root / f"gs_stream_copy.staged.11.{('b' * 32)}"
    cell.mkdir()
    staged = cell / "trace.staged.1"
    staged.mkdir()
    (staged / "payload").write_bytes(b"admitted")
    _write_nvbit_generation_handoff(staged)
    published = cell / "trace"
    displaced = cell / "displaced"
    admission = SimpleNamespace(
        trace_admission_sha256="a" * 64,
        launch_count=1,
        threadblock_count=1,
        instruction_count=1,
        expected_record_count=1,
        admitted_record_count=1,
    )
    monkeypatch.setattr(
        retained_golden_publication.trace_admission,
        "require_trace_admission",
        lambda _path: admission,
    )

    def replace_generation(_trace: Path) -> None:
        staged.rename(displaced)
        staged.mkdir()
        (staged / "payload").write_bytes(b"admitted")

    with pytest.raises(immutable.ImmutableArtifactError):
        retained_golden_publication.process_trace_generation(
            staged,
            published,
            derive_ncu=replace_generation,
        )
    assert not published.exists()
    assert (displaced / "payload").read_bytes() == b"admitted"
    assert (staged / "payload").read_bytes() == b"admitted"


def test_launch_handoff_rejects_replacement_before_publication_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = _publication_root(monkeypatch, tmp_path)
    cell = root / f"gs_stream_copy.staged.15.{('f' * 32)}"
    cell.mkdir()
    staged = cell / "trace.staged.1"
    staged.mkdir()
    (staged / "payload").write_bytes(b"captured")
    handoff_payload = _write_nvbit_generation_handoff(staged)
    displaced = cell / "captured-generation"
    staged.rename(displaced)
    staged.mkdir()
    (staged / "payload").write_bytes(b"replacement")
    (staged / "nvbit_launch_attempt.json").write_bytes(handoff_payload)

    with pytest.raises(
        retained_golden_publication.GoldenPublicationError,
        match="differs from the NVBit launch handoff",
    ):
        retained_golden_publication.process_trace_generation(
            staged,
            cell / "trace",
        )
    assert not (cell / "trace").exists()
    assert (displaced / "payload").read_bytes() == b"captured"
    assert (staged / "payload").read_bytes() == b"replacement"


def test_retained_trace_generation_relocates_exact_admitted_inode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = _publication_root(monkeypatch, tmp_path)
    cell = root / f"gs_stream_copy.staged.12.{('c' * 32)}"
    cell.mkdir()
    staged = cell / "trace.staged.1"
    staged.mkdir()
    payload = staged / "payload"
    payload.write_bytes(b"admitted")
    _write_nvbit_generation_handoff(staged)
    source_inode = staged.stat().st_ino
    admission = SimpleNamespace(
        trace_admission_sha256="b" * 64,
        launch_count=1,
        threadblock_count=1,
        instruction_count=1,
        expected_record_count=1,
        admitted_record_count=1,
    )
    monkeypatch.setattr(
        retained_golden_publication.trace_admission,
        "require_trace_admission",
        lambda _path: admission,
    )
    published = cell / "trace"
    retained_golden_publication.process_trace_generation(staged, published)
    assert not staged.exists()
    assert published.stat().st_ino == source_inode
    assert (published / "payload").read_bytes() == b"admitted"


def test_cell_handoff_rejects_trace_replacement_before_final_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = _publication_root(monkeypatch, tmp_path)
    cell = root / f"gs_stream_copy.staged.13.{('d' * 32)}"
    final = root / "gs_stream_copy"
    retained_golden_publication.prepare_cell_stage(cell, final)
    staged = cell / "trace.staged.1"
    staged.mkdir()
    (staged / "payload").write_bytes(b"admitted")
    _write_nvbit_generation_handoff(staged)
    published = cell / "trace"
    admission = SimpleNamespace(
        trace_admission_sha256="c" * 64,
        launch_count=1,
        threadblock_count=1,
        instruction_count=1,
        expected_record_count=1,
        admitted_record_count=1,
    )
    monkeypatch.setattr(
        retained_golden_publication.trace_admission,
        "require_trace_admission",
        lambda _path: admission,
    )
    displaced = cell / "displaced-trace"

    def prepare(trace_descriptor: int):
        handoff = retained_golden_publication._CellCommitHandoff.open(
            cell,
            final,
            expected_trace_descriptor=trace_descriptor,
        )
        complete = handoff.complete

        def replace_then_complete() -> None:
            published.rename(displaced)
            published.mkdir()
            (published / "payload").write_bytes(b"admitted")
            complete()

        handoff.complete = replace_then_complete
        return handoff

    with pytest.raises(
        retained_golden_publication.GoldenPublicationError,
        match="trace identity changed",
    ):
        retained_golden_publication.process_trace_generation(
            staged,
            published,
            prepare_handoff=prepare,
        )
    assert not final.exists()
    assert (displaced / "payload").read_bytes() == b"admitted"
    assert (published / "payload").read_bytes() == b"admitted"


def test_cell_handoff_publishes_derived_artifacts_before_atomic_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = _publication_root(monkeypatch, tmp_path)
    cell = root / f"gs_stream_copy.staged.14.{('e' * 32)}"
    final = root / "gs_stream_copy"
    retained_golden_publication.prepare_cell_stage(cell, final)
    trace = cell / "trace"
    trace.mkdir()
    (trace / "payload").write_bytes(b"admitted")
    staged_report = cell / "ncu_l2.csv.staged.1"
    staged_report.write_bytes(b"report")
    report = cell / "ncu_l2.csv"
    with immutable.PinnedDirectoryReader(trace) as trace_reader:
        handoff = retained_golden_publication._CellCommitHandoff.open(
            cell,
            final,
            expected_trace_descriptor=trace_reader.descriptor,
            artifacts=((staged_report, report),),
        )
    try:
        handoff.complete()
    finally:
        handoff.close()
    assert (final / "trace" / "payload").read_bytes() == b"admitted"
    assert (final / "ncu_l2.csv").read_bytes() == b"report"
    assert (final / "ncu_l2.csv.staged.1").read_bytes() == b"report"


def test_full_cell_commit_crash_retains_attempt_and_retry_can_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = _publication_root(monkeypatch, tmp_path)
    final = root / "gs_stream_copy"
    first = root / f"gs_stream_copy.staged.101.{('d' * 32)}"
    second = root / f"gs_stream_copy.staged.102.{('e' * 32)}"
    retained_golden_publication.prepare_cell_stage(first, final)
    (first / "trace").mkdir()
    (first / "trace" / "payload").write_bytes(b"first-trace")
    (first / "receipt").write_bytes(b"first-attempt")
    real_rename = immutable._rename_noreplace_at

    def crash_before_commit(*_args) -> None:
        raise KeyboardInterrupt("injected pre-commit crash")

    monkeypatch.setattr(immutable, "_rename_noreplace_at", crash_before_commit)
    with pytest.raises(KeyboardInterrupt, match="injected pre-commit crash"):
        retained_golden_publication.commit_cell_stage(first, final)
    assert first.is_dir()
    assert not final.exists()

    monkeypatch.setattr(immutable, "_rename_noreplace_at", real_rename)
    retained_golden_publication.prepare_cell_stage(second, final)
    (second / "trace").mkdir()
    (second / "trace" / "payload").write_bytes(b"retry-trace")
    (second / "receipt").write_bytes(b"retry-attempt")
    retained_golden_publication.commit_cell_stage(second, final)
    assert (final / "receipt").read_bytes() == b"retry-attempt"
    assert first.is_dir()
    assert not second.exists()


def test_full_cell_commit_never_replaces_existing_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = _publication_root(monkeypatch, tmp_path)
    final = root / "gs_stream_copy"
    stage = root / f"gs_stream_copy.staged.103.{('f' * 32)}"
    retained_golden_publication.prepare_cell_stage(stage, final)
    (stage / "trace").mkdir()
    (stage / "trace" / "payload").write_bytes(b"candidate-trace")
    (stage / "receipt").write_bytes(b"candidate")
    final.mkdir()
    (final / "receipt").write_bytes(b"winner")
    with pytest.raises(FileExistsError):
        retained_golden_publication.commit_cell_stage(stage, final)
    assert (final / "receipt").read_bytes() == b"winner"
    assert (stage / "receipt").read_bytes() == b"candidate"
