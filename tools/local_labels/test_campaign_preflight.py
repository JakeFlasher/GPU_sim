"""Unit + host-integration tests for the capture preflight gate."""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import campaign_preflight as cp  # noqa: E402
from tools.local_labels import sol_capture_environment as se  # noqa: E402

_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "campaign_preflight_manifest.schema.json").read_text()
)

_SOL_COMPONENTS = {
    "python_version": "3.12.12",
    "torch_version": "2.10.0+cu130",
    "nvcc_version": "13.2",
    "cuda_runtime_version": "13.0",
    "cutlass_tree_hash": "c" * 64,
    "cudnn_version": "9.15.1",
    "sol_execbench_package_version": "1.0.2",
    "nvcc_ccbin": "g++-14",
    "cxx_compiler": "",
    "c_compiler": "",
}
_SOL_NATIVE_ENV_HASH = se.canonical_sha256(_SOL_COMPONENTS)
_SOL_ENVIRONMENT_CONTRACT = {
    "CUTLASS_DIR": "third_party/cutlass",
    "PATH_prepend": "tools/sol_execbench/.venv/bin",
    **se.LEGACY_PINNED_COMPILER_ENVIRONMENT,
}
_SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT = {
    "CUTLASS_DIR": "third_party/cutlass",
    "PATH_prepend": "tools/sol_execbench/.venv/bin",
    **se.PINNED_COMPILER_ENVIRONMENT,
}


def _ok_sol_environment_probe() -> dict:
    return {
        "status": "ok",
        "typed_error": None,
        "detail": "certified non-degraded SOL native environment restored",
        "ambient": {},
        "environment_contract": dict(_SOL_ENVIRONMENT_CONTRACT),
        "environment_contract_sha256": "1" * 64,
        "constructed_environment_delta": dict(
            se.LEGACY_PINNED_COMPILER_ENVIRONMENT
        ),
        "matrix_pin": {
            "path": (
                "data/local_labels/campaign/"
                "sol_native_env_fingerprint_sm86.json"
            ),
            "sha256": "2" * 64,
            "native_env_hash": _SOL_NATIVE_ENV_HASH,
        },
        "matrix_sha256": "3" * 64,
        "observed_fingerprint": {
            "components": _SOL_COMPONENTS,
            "hash": _SOL_NATIVE_ENV_HASH,
            "missing": [],
            "degraded": False,
        },
        "compiler_identity": {
            "requested": "g++-14",
            "resolved_realpath": "/usr/bin/g++-14",
            "executable_sha256": "4" * 64,
            "version_line": "g++-14 (GCC) 14.3.1",
        },
        "nvcc_release": {"major": 13, "minor": 2},
        "nvcc_version_output": "Cuda compilation tools, release 13.2",
    }


def _ok_device_probe() -> dict:
    return {
        "status": "ok",
        "selection_policy": cp.DEVICE_SELECTION_POLICY,
        "selected_host_gpu_index": 0,
        "nvidia_smi_device_count": 1,
        "gpu_uuid_sha256": "6" * 64,
    }


def _ok_ncu_probe() -> dict:
    version = "Version 2026.1.0.0 (public-release)"
    launcher = "/opt/nvidia/nsight-compute/2026.1.0/ncu"
    return {
        "status": "ok",
        "expected": "2026.1.0",
        "observed": version,
        "execution_mode": cp.NCU_EXECUTION_MODE,
        "host_machine": cp.NCU_HOST_MACHINE,
        "launcher_resolved_path": launcher,
        "launcher_sha256": "4" * 64,
        "launcher_target_relpath": cp.NCU_LAUNCHER_TARGET_RELPATH,
        "resolved_path": str(
            Path(launcher).parent / cp.NCU_LAUNCHER_TARGET_RELPATH
        ),
        "executable_sha256": "5" * 64,
        "version_line": version,
    }


def _synthetic_ncu_install(tmp_path: Path) -> tuple[Path, Path]:
    launcher = tmp_path / "ncu"
    launcher.write_text("#!/bin/sh\nexit 99\n")
    launcher.chmod(0o555)
    inner = launcher.parent / cp.NCU_LAUNCHER_TARGET_RELPATH
    inner.parent.mkdir(parents=True)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[16:18] = (2).to_bytes(2, "little")
    header[18:20] = (62).to_bytes(2, "little")
    inner.write_bytes(bytes(header) + b"synthetic inner ncu")
    inner.chmod(0o555)
    return launcher, inner


# ---- pure logic (no GPU) --------------------------------------------------- #
def test_version_matches_modes():
    assert cp.version_matches("2026.1.0.0", "2026.1.0", mode="prefix")
    assert not cp.version_matches("2026.2.0.0", "2026.1.0", mode="prefix")
    assert cp.version_matches("13.3", "13.3", mode="exact")
    assert not cp.version_matches("13.2", "13.3", mode="exact")
    assert not cp.version_matches(None, "13.3", mode="exact")


def test_device_probe_binds_single_index_uuid_without_disclosure(monkeypatch):
    raw_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    observed = []

    def run(command):
        observed.append(command)
        return 0, f"0, {raw_uuid}\n", ""

    monkeypatch.setattr(cp, "_run", run)
    result = cp.probe_device()
    assert observed == [list(cp._DEVICE_QUERY)]
    assert result == {
        "status": cp.STATUS_OK,
        "selection_policy": cp.DEVICE_SELECTION_POLICY,
        "selected_host_gpu_index": 0,
        "nvidia_smi_device_count": 1,
        "gpu_uuid_sha256": hashlib.sha256(raw_uuid.encode()).hexdigest(),
    }
    assert raw_uuid not in json.dumps(result, sort_keys=True)


@pytest.mark.parametrize(
    "returncode,stdout,expected_status,expected_count",
    [
        (1, "", cp.STATUS_UNAVAILABLE, None),
        (0, "", cp.STATUS_MISMATCH, 0),
        (
            0,
            "0, GPU-01234567-89ab-cdef-0123-456789abcdef\n"
            "1, GPU-fedcba98-7654-3210-fedc-ba9876543210\n",
            cp.STATUS_MISMATCH,
            2,
        ),
        (
            0,
            "1, GPU-01234567-89ab-cdef-0123-456789abcdef\n",
            cp.STATUS_MISMATCH,
            1,
        ),
        (0, "0, not-a-gpu-uuid\n", cp.STATUS_MISMATCH, 1),
        (0, "0, GPU-deadbeef\n", cp.STATUS_MISMATCH, 1),
        (
            0,
            '0, "GPU-01234567-89ab-cdef-0123-\n456789abcdef"\n',
            cp.STATUS_MISMATCH,
            1,
        ),
        (
            0,
            "0, GPU-01234567-89ab-cdef-0123-456789abcdef\t\n",
            cp.STATUS_MISMATCH,
            1,
        ),
        (
            0,
            '0, "GPU-01234567-89ab-cdef-0123-456789abcdef\n',
            cp.STATUS_FAILED,
            None,
        ),
        (
            0,
            "0, GPU-01234567-89ab-cdef-0123-456789abcdef\n\n"
            "\n",
            cp.STATUS_FAILED,
            None,
        ),
        (0, "0\n", cp.STATUS_FAILED, None),
    ],
)
def test_device_probe_rejects_unbound_inventory_without_uuid_disclosure(
    monkeypatch, returncode, stdout, expected_status, expected_count,
):
    raw_stderr = "GPU-deadbeef-dead-beef-dead-beefdeadbeef"
    monkeypatch.setattr(
        cp, "_run", lambda _command: (returncode, stdout, raw_stderr),
    )
    result = cp.probe_device()
    assert result["status"] == expected_status
    assert result["selected_host_gpu_index"] is None
    assert result["nvidia_smi_device_count"] == expected_count
    assert result["gpu_uuid_sha256"] is None
    serialized = json.dumps(result, sort_keys=True)
    assert raw_stderr not in serialized
    assert "GPU-01234567-89ab-cdef-0123-456789abcdef" not in serialized


def test_toolchain_probes_use_structured_release_and_component_boundaries(
    monkeypatch, tmp_path,
):
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return 0, "Cuda compilation tools, release 13.2, V13.2.0\n", ""

    monkeypatch.setattr(cp, "_run", run)
    pins = {
        "toolchain": {
            "nvcc_release": "13.2",
            "ncu_version_prefix": "2026.1.0",
        }
    }
    nvcc = cp.probe_nvcc(
        pins, executable="/usr/local/cuda-13.2/bin/nvcc"
    )
    assert nvcc == {
        "status": cp.STATUS_OK,
        "observed": {"major": 13, "minor": 2},
        "expected": {"major": 13, "minor": 2},
        "executable": "/usr/local/cuda-13.2/bin/nvcc",
    }
    assert commands == [["/usr/local/cuda-13.2/bin/nvcc", "--version"]]

    launcher, inner = _synthetic_ncu_install(tmp_path)
    result = cp.probe_ncu(
        pins,
        resolve_command=lambda _name: str(launcher),
        run_version=lambda _descriptor: (
            0, "Version 2026.1.0.0 (public-release)\n", "",
        ),
    )
    assert result == {
        "status": cp.STATUS_OK,
        "expected": "2026.1.0",
        "observed": "Version 2026.1.0.0 (public-release)",
        "execution_mode": cp.NCU_EXECUTION_MODE,
        "host_machine": cp.NCU_HOST_MACHINE,
        "launcher_resolved_path": str(launcher),
        "launcher_sha256": cp.immutable.sha256_file(launcher)[0],
        "launcher_target_relpath": cp.NCU_LAUNCHER_TARGET_RELPATH,
        "resolved_path": str(inner),
        "executable_sha256": cp.immutable.sha256_file(inner)[0],
        "version_line": "Version 2026.1.0.0 (public-release)",
    }

    mismatch = cp.probe_ncu(
        pins,
        resolve_command=lambda _name: str(launcher),
        run_version=lambda _descriptor: (
            0, "Version 2026.10.0.0 (public-release)\n", "",
        ),
    )
    assert mismatch["status"] == cp.STATUS_MISMATCH
    assert mismatch["observed"] is None
    assert mismatch["resolved_path"] is None
    assert mismatch["executable_sha256"] is None


def test_ncu_probe_runs_retained_inner_and_never_launcher(tmp_path):
    launcher, inner = _synthetic_ncu_install(tmp_path)
    observed = {}

    def run(descriptor):
        metadata = os.fstat(descriptor)
        observed["identity"] = (metadata.st_dev, metadata.st_ino)
        observed["header"] = os.pread(descriptor, 4, 0)
        return 0, "Version 2026.1.0.0 (public-release)\n", ""

    result = cp.probe_ncu(
        {"toolchain": {"ncu_version_prefix": "2026.1.0"}},
        resolve_command=lambda _name: str(launcher),
        run_version=run,
    )
    inner_stat = inner.stat()
    assert result["status"] == cp.STATUS_OK
    assert result["resolved_path"] == str(inner)
    assert observed == {
        "identity": (inner_stat.st_dev, inner_stat.st_ino),
        "header": b"\x7fELF",
    }


def test_ncu_probe_refuses_unsupported_machine_before_resolution():
    def resolve(_name):
        pytest.fail("PATH resolution reached for unsupported host")

    result = cp.probe_ncu(
        {"toolchain": {"ncu_version_prefix": "2026.1.0"}},
        resolve_command=resolve,
        host_machine="aarch64",
    )
    assert result["status"] == cp.STATUS_FAILED
    assert result["resolved_path"] is None
    assert "unsupported NCU host machine" in result["detail"]


@pytest.mark.parametrize(
    ("offset", "payload", "detail"),
    [
        (0, b"NOPE", "not an ELF"),
        (4, b"\x01", "not little-endian ELF64"),
        (5, b"\x02", "not little-endian ELF64"),
        (16, (3).to_bytes(2, "little"), "not ET_EXEC"),
        (18, (183).to_bytes(2, "little"), "not EM_X86_64"),
    ],
)
def test_ncu_probe_rejects_wrong_inner_elf_identity(
    tmp_path, offset, payload, detail,
):
    launcher, inner = _synthetic_ncu_install(tmp_path)
    inner.chmod(0o755)
    with inner.open("r+b") as stream:
        stream.seek(offset)
        stream.write(payload)
    inner.chmod(0o555)
    result = cp.probe_ncu(
        {"toolchain": {"ncu_version_prefix": "2026.1.0"}},
        resolve_command=lambda _name: str(launcher),
        run_version=lambda _descriptor: pytest.fail(
            "version subprocess reached for invalid inner ELF"
        ),
    )
    assert result["status"] == cp.STATUS_FAILED
    assert result["resolved_path"] is None
    assert detail in result["detail"]


def test_ncu_probe_rejects_duplicate_version_lines_with_null_identity(tmp_path):
    launcher, _inner = _synthetic_ncu_install(tmp_path)
    version = "Version 2026.1.0.0 (public-release)\n"
    result = cp.probe_ncu(
        {"toolchain": {"ncu_version_prefix": "2026.1.0"}},
        resolve_command=lambda _name: str(launcher),
        run_version=lambda _descriptor: (0, version + version, ""),
    )
    assert result["status"] == cp.STATUS_MISMATCH
    assert result["resolved_path"] is None
    assert result["launcher_resolved_path"] is None


def test_retained_ncu_version_runner_uses_fd_and_c_locale(monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "Version 2026.1.0\n", "")

    monkeypatch.setattr(cp.subprocess, "run", run)
    assert cp._run_retained_ncu_version(17) == (
        0, "Version 2026.1.0\n", "",
    )
    assert observed["command"] == ["/proc/self/fd/17", "--version"]
    assert observed["kwargs"]["pass_fds"] == (17,)
    assert observed["kwargs"]["capture_output"] is True
    assert observed["kwargs"]["text"] is True
    assert observed["kwargs"]["check"] is False
    assert observed["kwargs"]["timeout"] == 30
    assert observed["kwargs"]["env"] == cp.ncu_executor.ncu_action_environment()


@pytest.mark.parametrize(
    ("pins", "resolved", "version", "status"),
    [
        ({"toolchain": {"ncu_version_prefix": True}}, None, None, "failed"),
        ({"toolchain": {"ncu_version_prefix": "2026.1.0"}}, None, None,
         "unavailable"),
        ({"toolchain": {"ncu_version_prefix": "2026.1.0"}}, "relative/ncu",
         None, "failed"),
        ({"toolchain": {"ncu_version_prefix": "2026.1.0"}}, "fixture",
         "no version banner\n", "mismatch"),
    ],
)
def test_ncu_probe_failures_never_retain_partial_identity(
    tmp_path, pins, resolved, version, status,
):
    launcher, _inner = _synthetic_ncu_install(tmp_path)
    candidate = str(launcher) if resolved == "fixture" else resolved
    result = cp.probe_ncu(
        pins,
        resolve_command=lambda _name: candidate,
        run_version=lambda _descriptor: (0, version or "", ""),
    )
    assert result["status"] == status
    assert result["observed"] is None
    assert result["execution_mode"] is None
    assert result["host_machine"] is None
    assert result["launcher_resolved_path"] is None
    assert result["launcher_sha256"] is None
    assert result["launcher_target_relpath"] is None
    assert result["resolved_path"] is None
    assert result["executable_sha256"] is None
    assert result["version_line"] is None


def test_ncu_probe_rejects_executable_mutation_during_version_observation(
    tmp_path,
):
    launcher, inner = _synthetic_ncu_install(tmp_path)

    def mutate(_descriptor):
        inner.chmod(0o755)
        inner.write_bytes(b"mutated ncu")
        inner.chmod(0o555)
        return 0, "Version 2026.1.0.0 (public-release)\n", ""

    result = cp.probe_ncu(
        {"toolchain": {"ncu_version_prefix": "2026.1.0"}},
        resolve_command=lambda _name: str(launcher),
        run_version=mutate,
    )
    assert result["status"] == cp.STATUS_FAILED
    assert result["resolved_path"] is None
    assert result["executable_sha256"] is None
    assert "metadata changed" in result["detail"]


def test_ncu_probe_rejects_launcher_mutation_during_inner_version_observation(
    tmp_path,
):
    launcher, _inner = _synthetic_ncu_install(tmp_path)

    def mutate(_descriptor):
        launcher.chmod(0o755)
        launcher.write_text("#!/bin/sh\nexit 0\n")
        launcher.chmod(0o555)
        return 0, "Version 2026.1.0.0 (public-release)\n", ""

    result = cp.probe_ncu(
        {"toolchain": {"ncu_version_prefix": "2026.1.0"}},
        resolve_command=lambda _name: str(launcher),
        run_version=mutate,
    )
    assert result["status"] == cp.STATUS_FAILED
    assert result["resolved_path"] is None
    assert result["launcher_resolved_path"] is None
    assert "metadata changed" in result["detail"]


def test_ncu_probe_converts_version_timeout_to_failed_null_identity(tmp_path):
    launcher, _inner = _synthetic_ncu_install(tmp_path)

    def timeout(_descriptor):
        raise subprocess.TimeoutExpired(["retained-ncu", "--version"], 30)

    result = cp.probe_ncu(
        {"toolchain": {"ncu_version_prefix": "2026.1.0"}},
        resolve_command=lambda _name: str(launcher),
        run_version=timeout,
    )
    assert result["status"] == cp.STATUS_FAILED
    assert result["observed"] is None
    assert result["resolved_path"] is None
    assert result["executable_sha256"] is None
    assert result["version_line"] is None
    assert "timed out after 30 seconds" in result["detail"]


def test_version_matches_bad_mode():
    with pytest.raises(ValueError):
        cp.version_matches("x", "x", mode="substring")


def test_remaining_dirty_lines_ignore_and_rename():
    lines = [
        "?? data/local_labels/campaign/preflight_manifest.json",
        " M tools/local_labels/campaign_preflight.py",
        "R  old/path.py -> new/path.py",
        "",
    ]
    # Ignoring the manifest dir drops the first line; rename keeps the new path.
    kept = cp.remaining_dirty_lines(lines, ["data/local_labels/campaign"])
    assert not any("preflight_manifest.json" in ln for ln in kept)
    assert any("campaign_preflight.py" in ln for ln in kept)
    assert any("new/path.py" in ln for ln in kept)
    assert "" not in kept


def test_evaluate_budget():
    budget = {
        "retention_budget_gib": 200,
        "metadata_margin_gib": 8,
        "raw_staging_margin_gib": 16,
    }
    ok, _ = cp.evaluate_budget(budget, 500.0)
    assert ok == cp.STATUS_OK
    low, _ = cp.evaluate_budget(budget, 216.0)
    assert low == cp.STATUS_MISMATCH
    bad, _ = cp.evaluate_budget({"retention_budget_gib": 200}, 500.0)
    assert bad == cp.STATUS_FAILED


def test_trace_budget_missing_file_fails_closed(tmp_path):
    probe = cp.probe_trace_budget(tmp_path / "nope.json", tmp_path)
    assert probe["status"] == cp.STATUS_FAILED


# ---- injection smoke verdict (pure, fail-closed; no GPU) ------------------- #
def _smoke_kwargs(**over):
    """A fully-passing smoke, overridable per-case."""
    base = dict(ldd_missing=[], returncode=0, pb_present=True, stats_present=True,
                expected=4800, admitted=4800, validator_ok=True,
                validator_reason=None, threadblocks_pb_count=4,
                root_trace_ok=True, root_trace_reason=None,
                root_trace_launch_count=1, banner_seen=True,
                full_coverage_ok=True, full_coverage_reason=None)
    base.update(over)
    return base


def test_injection_smoke_pass():
    status, _ = cp.injection_smoke_verdict(**_smoke_kwargs())
    assert status == cp.STATUS_OK


def test_injection_smoke_pass_without_banner_is_still_ok():
    # A contract-valid nonzero trace proves injection even if the banner was
    # buffered away; the banner is diagnostic, never a gate.
    status, detail = cp.injection_smoke_verdict(**_smoke_kwargs(banner_seen=False))
    assert status == cp.STATUS_OK
    assert "banner" in detail.lower()


def test_injection_smoke_ldd_missing_blocks():
    status, detail = cp.injection_smoke_verdict(
        **_smoke_kwargs(ldd_missing=["libprotobuf.so.3 => not found"]))
    assert status == cp.STATUS_FAILED
    assert "SILENTLY" in detail or "unresolved" in detail


def test_injection_smoke_ldd_ok_but_no_trace_blocks():
    # The exact silent-dlopen trap: deps resolve, app runs, but no trace lands.
    status, _ = cp.injection_smoke_verdict(
        **_smoke_kwargs(pb_present=False, stats_present=False,
                        expected=None, admitted=None, validator_ok=None))
    assert status == cp.STATUS_FAILED


def test_injection_smoke_zero_admitted_blocks():
    status, _ = cp.injection_smoke_verdict(**_smoke_kwargs(expected=100, admitted=0))
    assert status == cp.STATUS_FAILED


def test_injection_smoke_contract_invalid_blocks():
    # Trace present with records, but the tracer's own validator rejects it.
    status, detail = cp.injection_smoke_verdict(
        **_smoke_kwargs(validator_ok=False,
                        validator_reason="unsupported_memory_path"))
    assert status == cp.STATUS_FAILED
    assert "unsupported_memory_path" in detail


def test_injection_smoke_empty_threadblocks_blocks():
    # validate_trace_dir returns ok=True on a present-but-empty threadblocks/;
    # a nonzero stats.csv must NOT rescue that vacuous pass — the per-threadblock
    # records are the actual simulator input.
    status, detail = cp.injection_smoke_verdict(**_smoke_kwargs(threadblocks_pb_count=0))
    assert status == cp.STATUS_FAILED
    assert "vacuous" in detail.lower()


def test_injection_smoke_nonzero_exit_blocks():
    status, _ = cp.injection_smoke_verdict(**_smoke_kwargs(returncode=139))
    assert status == cp.STATUS_FAILED


def test_injection_smoke_incomplete_full_grid_blocks():
    status, detail = cp.injection_smoke_verdict(
        **_smoke_kwargs(
            full_coverage_ok=False,
            full_coverage_reason="decoded_threadblocks=3 != grid=4",
        )
    )
    assert status == cp.STATUS_FAILED
    assert "decoded_threadblocks=3" in detail


def test_full_trace_coverage_requires_exact_grid_records_and_no_sampling():
    passing = dict(
        root_launch_count=1, grid_threadblocks=4, decoded_threadblocks=4,
        threadblock_pb_count=4, decoded_records=4800,
        expected_records=4800, admitted_records=4800, stats_row_count=1,
        sampling_sidecar_present=False, zero_record_threadblocks=0,
    )
    assert cp.full_trace_coverage_verdict(**passing)[0] == cp.STATUS_OK
    for changed in (
        {"root_launch_count": 2},
        {"stats_row_count": 2},
        {"decoded_threadblocks": 3},
        {"threadblock_pb_count": 3},
        {"zero_record_threadblocks": 1},
        {"decoded_records": 0},
        {"admitted_records": 4799},
        {"sampling_sidecar_present": True},
    ):
        assert cp.full_trace_coverage_verdict(
            **(passing | changed)
        )[0] == cp.STATUS_FAILED


def test_full_trace_coverage_accepts_exact_three_launch_aggregate_only():
    passing = dict(
        root_launch_count=3, expected_launch_count=3,
        grid_threadblocks=96, decoded_threadblocks=96,
        threadblock_pb_count=96, decoded_records=1384288,
        expected_records=1384288, admitted_records=1384288,
        stats_row_count=3, sampling_sidecar_present=False,
        zero_record_threadblocks=0,
    )
    assert cp.full_trace_coverage_verdict(**passing)[0] == cp.STATUS_OK
    for changed in (
        {"root_launch_count": 2}, {"stats_row_count": 2},
        {"decoded_threadblocks": 95}, {"threadblock_pb_count": 95},
        {"zero_record_threadblocks": 1}, {"admitted_records": 1384287},
    ):
        assert cp.full_trace_coverage_verdict(
            **(passing | changed)
        )[0] == cp.STATUS_FAILED


def test_stats_parser_fails_closed_on_one_malformed_row(tmp_path):
    stats = tmp_path / "stats.csv"
    stats.write_text(
        "kernel,total_insts,total_reported_insts\n"
        "k0,10,10\n"
        "k1,not-an-int,7\n"
    )
    details = cp._parse_stats_admission_details(stats)
    assert details["expected"] is None
    assert "non-integer" in details["error"]


def test_stats_parser_records_exact_row_count(tmp_path):
    stats = tmp_path / "stats.csv"
    stats.write_text(
        "kernel,total_insts,total_reported_insts\n"
        "k0,10,10\n"
    )
    assert cp._parse_stats_admission_details(stats) == {
        "expected": 10, "admitted": 10, "row_count": 1, "error": None,
    }


def test_smoke_workload_output_is_forced_under_temp_root(tmp_path):
    argv = cp._smoke_workload_argv(["--correctness_shape"], tmp_path)
    assert argv == [
        "--correctness_shape", "--output_dir",
        str(tmp_path / "workload_output"),
    ]
    with pytest.raises(ValueError, match="must not pre-bind"):
        cp._smoke_workload_argv(
            ["--correctness_shape", "--output_dir", "/tmp/escape"], tmp_path,
        )


def test_preflight_bootstraps_trace_consumer_import_outside_checkout(tmp_path):
    """The executable script must not rely on the caller's cwd/PYTHONPATH."""
    script = _REPO / "tools" / "local_labels" / "campaign_preflight.py"
    code = f"""
import importlib.util
import pathlib
import sys
import tempfile
spec = importlib.util.spec_from_file_location("isolated_preflight", {str(script)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from tools.trace_sampling.counting_pass import analyze_trace_dir
assert callable(analyze_trace_dir)
assert {str(_REPO)!r} in sys.path
trace_dir = pathlib.Path(tempfile.mkdtemp())
analysis = module._analyze_full_trace_coverage(
    trace_dir, {{"launch_count": 1}}, 1, 1, 1,
)
assert analysis["status"] == module.STATUS_FAILED
assert "dynamic_trace.pb" in analysis["detail"]
assert "No module named" not in analysis["detail"]
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_injection_smoke_default_root_trace_blocks():
    status, detail = cp.injection_smoke_verdict(
        **_smoke_kwargs(root_trace_ok=False,
                        root_trace_reason="Trace contains zero kernel launches",
                        root_trace_launch_count=0))
    assert status == cp.STATUS_FAILED
    assert "zero kernel launches" in detail


def test_root_trace_verdict_requires_contract_capabilities_and_geometry():
    common = dict(
        pb_bytes=64, parse_error=None,
        trace_contract_id="sm86_addr_carry_v1",
        memory_payload_classes=[1, 2, 3, 6, 8],
        launch_geometries=[{"grid_dim": [1, 1, 1], "block_dim": [32, 1, 1]}],
    )
    assert cp.root_trace_verdict(**common)[0] == cp.STATUS_OK
    assert cp.root_trace_verdict(**(common | {"pb_bytes": 0}))[0] == cp.STATUS_FAILED
    assert cp.root_trace_verdict(
        **(common | {"trace_contract_id": "legacy"}))[0] == cp.STATUS_FAILED
    assert cp.root_trace_verdict(
        **(common | {"memory_payload_classes": [1, 2, 3, 6]}))[0] == cp.STATUS_FAILED
    assert cp.root_trace_verdict(
        **(common | {"launch_geometries": []}))[0] == cp.STATUS_FAILED
    assert cp.root_trace_verdict(**(
        common | {"launch_geometries": [
            {"grid_dim": [0, 1, 1], "block_dim": [32, 1, 1]}]}))[0] == cp.STATUS_FAILED


def test_process_hygiene_is_closed_and_non_vacuous():
    status, _, conflicts = cp.process_hygiene_verdict([
        {"pid": 1, "argv": ["/usr/bin/python", "harmless.py"],
         "executable": "/usr/bin/python3.12"},
        {"pid": 2, "argv": ["/venv/bin/sol-execbench", "evaluate"],
         "executable": "/venv/bin/sol-execbench"},
        {"pid": 3, "argv": ["profiler", "--clock-control", "none", "app"],
         "executable": "/opt/nvidia/ncu"},
    ])
    assert status == cp.STATUS_FAILED
    assert {c["marker"] for c in conflicts} == {"sol-execbench", "ncu"}
    ok, _, none = cp.process_hygiene_verdict([
        {"pid": 4, "argv": ["/usr/bin/python", "harmless.py"],
         "executable": "/usr/bin/python3.12"}])
    assert ok == cp.STATUS_OK
    assert none == []


def _process_row(pid, ppid, started, argv, executable):
    return {
        "pid": pid,
        "ppid": ppid,
        "start_time_ticks": started,
        "argv": argv,
        "executable": executable,
    }


def test_launcher_shell_marker_prose_is_not_a_capture_process():
    # Retained event 1 used this shape: an ancestor bash -lc program carried
    # SOL paths in variable values, and the old raw-substring scanner called it
    # an evaluator even though bash was only the launcher/monitor.
    launcher = _process_row(
        1366622, 1, 20,
        [
            "/usr/bin/bash", "-lc",
            "set -euo pipefail; "
            "REPO=/home/jakeshea/GPU_simulator_rebase_live_e38788ca; "
            "RUN=/var/tmp/evidence/sol_runnability.external.v2.json; "
            "PY=$REPO/tools/sol_execbench/.venv/bin/python; exec $PY campaign.py",
        ],
        "/usr/bin/bash",
    )
    status, _, conflicts = cp.process_hygiene_verdict([launcher])
    assert status == cp.STATUS_OK
    assert conflicts == []


def test_python_inline_sol_adapter_import_is_a_real_conflict():
    # Retained event 3 used a real independent Python -c adapter/prob-packager
    # probe.  Parse Python imports structurally so this remains distinguishable
    # from the launcher's shell command text.
    adapter = _process_row(
        1434903, 1, 30,
        [
            "/home/jakeshea/GPU_simulator_rebase/tools/sol_execbench/.venv/bin/python",
            "-c",
            "from sol_execbench.driver import capture_adapter as adapter\n"
            "from sol_execbench.driver import problem_packager\n"
            "print(adapter, problem_packager)\n",
        ],
        "/usr/bin/python3.12",
    )
    status, _, conflicts = cp.process_hygiene_verdict([adapter])
    assert status == cp.STATUS_FAILED
    assert [(row["pid"], row["marker"]) for row in conflicts] == [
        (1434903, "sol_execbench")
    ]


def test_independent_pytest_sol_target_is_a_real_conflict():
    pytest_process = _process_row(
        70, 1, 30,
        [
            "/usr/bin/python3", "-m", "pytest", "-q",
            "tools/sol_execbench/tests/test_capture_adapter.py",
        ],
        "/usr/bin/python3.12",
    )
    status, _, conflicts = cp.process_hygiene_verdict([pytest_process])
    assert status == cp.STATUS_FAILED
    assert conflicts[0]["marker"] == "sol_execbench"


def test_python_shebang_console_script_is_a_real_conflict():
    process = _process_row(
        71, 1, 30,
        [
            "/usr/bin/python3",
            "/home/user/.local/bin/sol-execbench",
            "evaluate",
        ],
        "/usr/bin/python3.12",
    )
    status, _, conflicts = cp.process_hygiene_verdict([process])
    assert status == cp.STATUS_FAILED
    assert conflicts[0]["marker"] == "sol-execbench"


def test_only_verified_ancestor_chain_is_ignored_not_descendants_or_peers():
    rows = [
        _process_row(1, 0, 1, ["/sbin/init"], "/sbin/init"),
        _process_row(
            10, 1, 10,
            ["/usr/bin/bash", "-lc", "PY=/repo/tools/sol_execbench/.venv/bin/python"],
            "/usr/bin/bash",
        ),
        _process_row(
            20, 10, 20,
            ["/usr/bin/python", "tools/local_labels/sol_workload_capture_campaign.py"],
            "/usr/bin/python3.12",
        ),
        _process_row(
            30, 20, 30,
            ["/usr/bin/python", "tools/local_labels/campaign_preflight.py"],
            "/usr/bin/python3.12",
        ),
        # A leaked descendant is never covered by the ancestor exemption.
        _process_row(
            40, 30, 40,
            ["/venv/bin/sol-execbench", "evaluate"],
            "/venv/bin/sol-execbench",
        ),
        # The same real probe on an independent branch remains a conflict.
        _process_row(
            50, 1, 40,
            ["/usr/bin/python", "-c", "import sol_execbench.driver.capture_adapter"],
            "/usr/bin/python3.12",
        ),
        # Reparenting to init cannot turn an orphan capture into an ancestor.
        _process_row(
            60, 1, 40,
            ["/venv/bin/sol-execbench", "evaluate"],
            "/venv/bin/sol-execbench",
        ),
    ]
    ignored = cp._verified_ancestor_pids(rows, 30)
    assert ignored == {1, 10, 20, 30}
    status, _, conflicts = cp.process_hygiene_verdict(
        rows, ignored_pids=ignored,
    )
    assert status == cp.STATUS_FAILED
    assert {row["pid"] for row in conflicts} == {40, 50, 60}


def test_probe_applies_verified_ancestor_exclusion(monkeypatch):
    rows = [
        _process_row(1, 0, 1, ["/sbin/init"], "/sbin/init"),
        _process_row(
            20, 1, 20,
            ["/venv/bin/sol-execbench", "parent-controller"],
            "/venv/bin/sol-execbench",
        ),
        _process_row(
            30, 20, 30,
            ["/usr/bin/python", "tools/local_labels/campaign_preflight.py"],
            "/usr/bin/python3.12",
        ),
        _process_row(
            40, 1, 40,
            ["/venv/bin/sol-execbench", "independent"],
            "/venv/bin/sol-execbench",
        ),
    ]
    monkeypatch.setattr(cp, "_snapshot_processes", lambda: rows)
    monkeypatch.setattr(cp.os, "getpid", lambda: 30)
    result = cp.probe_capture_processes()
    assert result["ignored_self_pid"] == 30
    assert result["ignored_verified_ancestor_pids"] == [1, 20]
    assert result["enumerated_processes"] == 1
    assert [row["pid"] for row in result["conflicts"]] == [40]


def test_parent_pid_reuse_never_grants_an_ancestor_exemption():
    rows = [
        _process_row(
            20, 1, 101,
            ["/venv/bin/sol-execbench", "evaluate"],
            "/venv/bin/sol-execbench",
        ),
        _process_row(
            30, 20, 100,
            ["/usr/bin/python", "tools/local_labels/campaign_preflight.py"],
            "/usr/bin/python3.12",
        ),
    ]
    ignored = cp._verified_ancestor_pids(rows, 30)
    assert ignored == {30}
    status, _, conflicts = cp.process_hygiene_verdict(
        rows, ignored_pids=ignored,
    )
    assert status == cp.STATUS_FAILED
    assert [row["pid"] for row in conflicts] == [20]


def test_executable_identity_beats_spoofed_argv_zero():
    rows = [
        _process_row(80, 1, 20, ["ncu", "-lc", "echo harmless"], "/usr/bin/bash"),
        _process_row(81, 1, 20, ["renamed-profiler", "app"], "/opt/nvidia/ncu"),
    ]
    status, _, conflicts = cp.process_hygiene_verdict(rows)
    assert status == cp.STATUS_FAILED
    assert [(row["pid"], row["marker"]) for row in conflicts] == [(81, "ncu")]


def test_clock_state_verdict_requires_repeated_exact_p0_samples():
    ref = {"persistence_mode": "Enabled", "sm_clock_mhz": 1350,
           "mem_clock_mhz": 10501}
    sample = {"persistence_mode": "Enabled", "graphics_clock_mhz": 1350,
              "sm_clock_mhz": 1350, "mem_clock_mhz": 10501, "pstate": "P0"}
    assert cp.clock_state_verdict([sample, sample, sample], ref)[0] == cp.STATUS_OK
    assert cp.clock_state_verdict([sample], ref)[0] == cp.STATUS_FAILED
    drift = sample | {"mem_clock_mhz": 10251}
    assert cp.clock_state_verdict([sample, drift, sample], ref)[0] == cp.STATUS_MISMATCH


def test_wait_for_clock_state_repeats_complete_gate_until_exact_p0(monkeypatch):
    attempts = iter([
        {"status": cp.STATUS_MISMATCH, "observed": {"pstate": "P2"},
         "detail": "P2"},
        {"status": cp.STATUS_OK, "observed": {"pstate": "P0"},
         "detail": "three exact samples"},
    ])
    sleeps = []
    monkeypatch.setattr(cp, "probe_clock_state", lambda _pins: next(attempts))
    monkeypatch.setattr(cp.time, "sleep", sleeps.append)
    result = cp.wait_for_clock_state(
        {}, max_attempts=3, poll_interval_s=0.25,
    )
    assert result["status"] == cp.STATUS_OK
    assert result["settle_attempt_count"] == 2
    assert [row["status"] for row in result["settle_attempts"]] == [
        cp.STATUS_MISMATCH, cp.STATUS_OK,
    ]
    assert sleeps == [0.25]


def test_wait_for_clock_state_exhaustion_stays_fail_closed(monkeypatch):
    monkeypatch.setattr(cp, "probe_clock_state", lambda _pins: {
        "status": cp.STATUS_MISMATCH, "observed": {"pstate": "P2"},
        "detail": "still P2",
    })
    monkeypatch.setattr(cp.time, "sleep", lambda _seconds: None)
    result = cp.wait_for_clock_state({}, max_attempts=2, poll_interval_s=1)
    assert result["status"] == cp.STATUS_MISMATCH
    assert result["settle_attempt_count"] == 2
    assert "did not settle" in result["detail"]


# ---- certified SOL environment (pure mocks; no compiler/GPU execution) ---- #
def _write_sol_environment_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    (tmp_path / "third_party" / "cutlass").mkdir(parents=True)
    (tmp_path / "tools" / "sol_execbench" / ".venv" / "bin").mkdir(
        parents=True
    )
    fingerprint_rel = Path(
        "data/local_labels/campaign/sol_native_env_fingerprint_sm86.json"
    )
    fingerprint_path = tmp_path / fingerprint_rel
    fingerprint_path.parent.mkdir(parents=True)
    fingerprint = {
        "schema_version": 1,
        "fingerprint_id": "sm86_sol_native_env_fingerprint_v1",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": _SOL_NATIVE_ENV_HASH,
        "components": _SOL_COMPONENTS,
        "missing_components": [],
        "degraded": False,
    }
    fingerprint_path.write_text(json.dumps(fingerprint, sort_keys=True))
    environment = dict(_SOL_ENVIRONMENT_CONTRACT)
    matrix = {
        "input_pins": {
            "sol_native_env_fingerprint": {
                "path": fingerprint_rel.as_posix(),
                "sha256": se.sha256_file(fingerprint_path),
                "native_env_hash": _SOL_NATIVE_ENV_HASH,
            },
        },
        "cells": [{
            "identity": {
                "roster_source": "sol_execbench",
                "slug": "problem_070_bootstrap",
            },
            "disposition": "runnable",
            "invocation": {
                "adapter_contract": {"environment": environment},
            },
        }],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    return matrix, matrix_path, fingerprint_path


def _supplemental_environment_matrix(matrix: dict) -> dict:
    adapter = {
        "environment": dict(_SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT),
        "native_env_hash": _SOL_NATIVE_ENV_HASH,
    }
    return {
        "schema_version": 2,
        "artifact_id": se.SUPPLEMENTAL_MATRIX_ARTIFACT_ID,
        "input_pins": copy.deepcopy(matrix["input_pins"]),
        "variants": [{
            "slug": "problem_070_bootstrap",
            "variant_id": "native",
            "native_env_hash": _SOL_NATIVE_ENV_HASH,
            "adapter_contract": adapter,
            "invocation": {"adapter_contract": copy.deepcopy(adapter)},
        }],
    }


def _mock_certified_runtime(monkeypatch, tmp_path: Path) -> None:
    def build_full_certified_env(contract, *, ambient=None):
        result = dict(ambient or {})
        for name in (*se.PINNED_COMPILER_ENVIRONMENT, *se.HAZARD_VARIABLES):
            result.pop(name, None)
        result.update(contract)
        return result

    def build_legacy_certified_env(contract, *, ambient=None):
        result = dict(ambient or {})
        for name in (*se.LEGACY_PINNED_COMPILER_ENVIRONMENT, *se.HAZARD_VARIABLES):
            result.pop(name, None)
        result.update(contract)
        return result

    monkeypatch.setattr(se, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        se,
        "_runtime_api",
        lambda: (build_full_certified_env, se.canonical_sha256),
    )
    monkeypatch.setattr(
        se,
        "_legacy_runtime_api",
        lambda: (build_legacy_certified_env, se.canonical_sha256),
    )
    compiler_result = lambda environment, requested: ({
            "requested": requested,
            "resolved_realpath": "/usr/bin/g++-14",
            "executable_sha256": "4" * 64,
            "version_line": "g++-14 (GCC) 14.3.1",
        }, "Cuda compilation tools, release 13.2")
    monkeypatch.setattr(se, "compiler_identity", compiler_result)
    monkeypatch.setattr(se, "legacy_compiler_identity", compiler_result)


def _matching_fingerprint(environment) -> dict:
    assert environment.get("NVCC_CCBIN") == "g++-14"
    if "CUDA_HOME" in environment:
        assert {
            key: environment.get(key) for key in se.PINNED_COMPILER_ENVIRONMENT
        } == se.PINNED_COMPILER_ENVIRONMENT
    else:
        assert not (
            set(se.PINNED_COMPILER_ENVIRONMENT) - {"NVCC_CCBIN"}
        ).intersection(environment)
    assert "NVCC_PREPEND_FLAGS" not in environment
    assert "NVCC_APPEND_FLAGS" not in environment
    return {
        "components": dict(_SOL_COMPONENTS),
        "hash": _SOL_NATIVE_ENV_HASH,
        "missing": [],
        "degraded": False,
    }


@pytest.mark.parametrize(
    "ambient",
    [
        {"PATH": "/usr/bin"},
        {"PATH": "/usr/bin", "NVCC_CCBIN": "g++-14"},
    ],
)
def test_sol_environment_probe_accepts_unset_or_exact_ambient_compiler(
        monkeypatch, tmp_path, ambient):
    matrix, matrix_path, _fingerprint_path = _write_sol_environment_fixture(
        tmp_path
    )
    _mock_certified_runtime(monkeypatch, tmp_path)
    probe = se.probe_sol_environment(
        matrix,
        matrix_path=matrix_path,
        ambient=ambient,
        fingerprint_probe=_matching_fingerprint,
        repo_root=tmp_path,
    )
    assert probe["status"] == "ok", probe
    assert probe["typed_error"] is None
    assert probe["environment_contract"] == _SOL_ENVIRONMENT_CONTRACT
    assert probe["matrix_pin"]["native_env_hash"] == _SOL_NATIVE_ENV_HASH
    assert probe["observed_fingerprint"]["components"] == _SOL_COMPONENTS
    assert probe["compiler_identity"]["requested"] == "g++-14"
    assert probe["matrix_sha256"] == se.sha256_file(matrix_path)


def test_sol_environment_default_probe_uses_package_import(
        monkeypatch, tmp_path):
    from tools.local_labels import sol_reconstitution

    matrix, matrix_path, _fingerprint_path = _write_sol_environment_fixture(
        tmp_path
    )
    _mock_certified_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sol_reconstitution,
        "native_env_fingerprint",
        lambda *, environ: _matching_fingerprint(environ),
    )
    probe = se.probe_sol_environment(
        matrix,
        matrix_path=matrix_path,
        ambient={"PATH": "/usr/bin"},
        repo_root=tmp_path,
    )
    assert probe["status"] == "ok", probe
    assert probe["observed_fingerprint"]["hash"] == _SOL_NATIVE_ENV_HASH


def test_sol_certification_returns_the_exact_environment_given_to_probe(
        monkeypatch, tmp_path):
    matrix, matrix_path, _fingerprint_path = _write_sol_environment_fixture(
        tmp_path
    )
    _mock_certified_runtime(monkeypatch, tmp_path)
    observed_environment = None

    def fingerprint(environment):
        nonlocal observed_environment
        observed_environment = environment
        return _matching_fingerprint(environment)

    probe, certified_environment = se.certify_sol_environment(
        matrix,
        matrix_path=matrix_path,
        ambient={
            "PATH": "/usr/bin",
            "SOL_EXECBENCH_CAPTURE_REQUEST": "/tmp/stale-request.json",
        },
        fingerprint_probe=fingerprint,
        repo_root=tmp_path,
    )
    assert probe["status"] == "ok"
    assert observed_environment is certified_environment
    assert certified_environment["PYTORCH_ALLOC_CONF"] == (
        "expandable_segments:True"
    )
    assert "SOL_EXECBENCH_CAPTURE_REQUEST" not in certified_environment


@pytest.mark.parametrize(
    ("ambient", "detail_fragment"),
    [
        ({"NVCC_CCBIN": "/usr/bin/g++-15"}, "NVCC_CCBIN"),
        ({"NVCC_CCBIN": "/usr/bin/g++-14"}, "NVCC_CCBIN"),
        ({"NVCC_PREPEND_FLAGS": "-Xcompiler foo"}, "NVCC_PREPEND_FLAGS"),
        ({"NVCC_APPEND_FLAGS": ""}, "NVCC_APPEND_FLAGS"),
        ({"CUDA_HOME": "/opt/cuda"}, "CUDA_HOME"),
        ({"CC": "/usr/bin/gcc-15"}, "CC"),
        ({"CXX": "/usr/bin/g++-15"}, "CXX"),
        ({"CUDAHOSTCXX": "/usr/bin/g++-15"}, "CUDAHOSTCXX"),
    ],
)
def test_sol_environment_ambient_conflicts_refuse_before_construction(
        monkeypatch, ambient, detail_fragment):
    full_only = detail_fragment in {"CUDA_HOME", "CC", "CXX", "CUDAHOSTCXX"}
    contract = dict(
        _SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT
        if full_only else _SOL_ENVIRONMENT_CONTRACT
    )
    admit = (
        se.require_ambient_admission
        if full_only else se.require_legacy_ambient_admission
    )

    def forbidden_runtime():
        raise AssertionError("environment construction ran after ambient refusal")

    monkeypatch.setattr(se, "_runtime_api", forbidden_runtime)
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        admit(contract, ambient)
    assert caught.value.code == se.AMBIENT_CONFLICT
    assert detail_fragment in caught.value.detail


def test_sol_environment_exact_gcc14_compiler_snapshots_are_admitted():
    contract = dict(_SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT)
    ambient = {
        "CUDA_HOME": "/usr/local/cuda-13.2",
        "NVCC_CCBIN": "g++-14",
        "CC": "/usr/bin/gcc-14",
        "CXX": "/usr/bin/g++-14",
        "CUDAHOSTCXX": "g++-14",
    }
    snapshot = se.require_ambient_admission(contract, ambient)
    assert snapshot["NVCC_CCBIN"] == {"present": True, "value": "g++-14"}
    assert snapshot["CXX"]["value"] == "/usr/bin/g++-14"


def test_clean_environment_wrapper_removes_every_producer_admission_input():
    wrapper = _REPO / "tools/local_labels/run_sm86_capture_clean_env.sh"
    ambient_names = (
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "NVCC_CCBIN",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "CUDA_HOME",
        "CC",
        "CXX",
        "CUDAHOSTCXX",
        "CUTLASS_DIR",
        "PYTHONPATH",
        "PYTHONHOME",
        "FLASHINFER_TRACE_DIR",
    )
    environment = dict(os.environ)
    environment.update({name: f"planted-{name}" for name in ambient_names})
    probe = (
        "import json, os; "
        f"names={ambient_names!r}; "
        "print(json.dumps({"
        "'present': {name: name in os.environ for name in names}, "
        "'path': os.environ.get('PATH'), "
        "'python_no_user_site': os.environ.get('PYTHONNOUSERSITE')}, "
        "sort_keys=True))"
    )

    result = subprocess.run(
        [str(wrapper), sys.executable, "-c", probe],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "present": {name: False for name in ambient_names},
        "path": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
        "python_no_user_site": "1",
    }


def test_sol_capture_environment_construction_is_deterministic(
        monkeypatch, tmp_path):
    matrix, _matrix_path, _fingerprint_path = _write_sol_environment_fixture(
        tmp_path
    )
    _mock_certified_runtime(monkeypatch, tmp_path)
    ambient = {"PATH": "/usr/bin", "HOME": "/tmp/home"}
    first = se.build_capture_environment(matrix, ambient)
    second = se.build_capture_environment(matrix, ambient)
    assert first == second
    assert ambient == {"PATH": "/usr/bin", "HOME": "/tmp/home"}
    assert first["NVCC_CCBIN"] == "g++-14"
    assert not (
        set(se.PINNED_COMPILER_ENVIRONMENT) - {"NVCC_CCBIN"}
    ).intersection(first)
    assert first["PATH"].split(os.pathsep)[:2] == [
        str(tmp_path / "tools" / "sol_execbench" / ".venv" / "bin"),
        "/usr/bin",
    ]


def test_sol_capture_environment_missing_contract_path_refuses(
        monkeypatch, tmp_path):
    matrix, _matrix_path, _fingerprint_path = _write_sol_environment_fixture(
        tmp_path
    )
    (tmp_path / "third_party" / "cutlass").rmdir()
    _mock_certified_runtime(monkeypatch, tmp_path)
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.build_capture_environment(matrix, {"PATH": "/usr/bin"})
    assert caught.value.code == "SOL_ENV_PATH_MISSING"


def test_sol_compiler_identity_refuses_unresolved_compiler(monkeypatch):
    monkeypatch.setattr(se.shutil, "which", lambda *_args, **_kwargs: None)
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.compiler_identity({"PATH": "/empty"}, requested="g++-14")
    assert caught.value.code == "SOL_ENV_COMPILER_UNRESOLVED"


def test_sol_compiler_identity_refuses_nvcc_outside_pinned_cuda_home(
        monkeypatch):
    def which(name, *, path=None):
        del path
        if name == "g++-14":
            return "/usr/bin/g++-14"
        if name == "nvcc":
            return "/opt/cuda/bin/nvcc"
        return None

    monkeypatch.setattr(se.shutil, "which", which)
    environment = {
        **se.PINNED_COMPILER_ENVIRONMENT,
        "PATH": "/opt/cuda/bin:/usr/bin",
    }
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.compiler_identity(environment, requested="g++-14")
    assert caught.value.code == "SOL_ENV_NVCC_MISMATCH"


@pytest.mark.parametrize(
    "observed_change",
    [
        {"hash": "f" * 64},
        {"degraded": True},
        {"missing": ["cudnn_version"]},
        {"components": _SOL_COMPONENTS | {"cudnn_version": "drifted"}},
    ],
)
def test_sol_environment_probe_rejects_fingerprint_drift(
        monkeypatch, tmp_path, observed_change):
    matrix, matrix_path, _fingerprint_path = _write_sol_environment_fixture(
        tmp_path
    )
    _mock_certified_runtime(monkeypatch, tmp_path)
    observed = _matching_fingerprint(dict(se.PINNED_COMPILER_ENVIRONMENT))
    observed.update(observed_change)
    probe = se.probe_sol_environment(
        matrix,
        matrix_path=matrix_path,
        ambient={"PATH": "/usr/bin"},
        fingerprint_probe=lambda _environment: observed,
        repo_root=tmp_path,
    )
    assert probe["status"] == "failed"
    assert probe["typed_error"] == "SOL_ENV_FINGERPRINT_MISMATCH"


def test_sol_fingerprint_pin_rejects_file_hash_drift(tmp_path):
    matrix, _matrix_path, fingerprint_path = _write_sol_environment_fixture(
        tmp_path
    )
    fingerprint_path.write_text(fingerprint_path.read_text() + "\n")
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.fingerprint_pin(matrix, repo_root=tmp_path)
    assert caught.value.code == "SOL_ENV_MATRIX_PIN_MISMATCH"


def test_sol_matrix_environment_rejects_mixed_or_unpinned_contracts(tmp_path):
    matrix, _matrix_path, _fingerprint_path = _write_sol_environment_fixture(
        tmp_path
    )
    mixed = copy.deepcopy(matrix)
    second = copy.deepcopy(mixed["cells"][0])
    second["identity"]["slug"] = "problem_071_second"
    second["invocation"]["adapter_contract"]["environment"][
        "NVCC_CCBIN"
    ] = "g++-15"
    mixed["cells"].append(second)
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.matrix_environment_contract(mixed)
    assert caught.value.code == "SOL_ENV_MATRIX_CONTRACT_MIXED"

    missing = copy.deepcopy(matrix)
    del missing["cells"][0]["invocation"]["adapter_contract"]["environment"][
        "NVCC_CCBIN"
    ]
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.matrix_environment_contract(missing)
    assert caught.value.code == "SOL_ENV_MATRIX_CONTRACT_INVALID"


def test_actual_tracked_canonical_matrix_reaches_expected_fingerprint_drift(
    monkeypatch,
):
    matrix_path = (
        _REPO / "data" / "local_labels" / "campaign"
        / "sm86_reference_capture_matrix.json"
    )
    matrix = json.loads(matrix_path.read_text())
    assert se.matrix_environment_contract(matrix) == _SOL_ENVIRONMENT_CONTRACT

    def build_legacy(contract, *, ambient=None):
        result = dict(ambient or {})
        result.update(contract)
        return result

    monkeypatch.setattr(
        se, "_legacy_runtime_api",
        lambda: (build_legacy, se.canonical_sha256),
    )
    monkeypatch.setattr(
        se, "legacy_compiler_identity",
        lambda environment, requested: ({
            "requested": requested,
            "resolved_realpath": "/usr/bin/g++-14",
            "executable_sha256": "4" * 64,
            "version_line": "g++-14 synthetic tracked-matrix probe",
        }, "Cuda compilation tools, release 13.2"),
    )
    current_host = {
        "components": {**_SOL_COMPONENTS, "nvcc_version": "13.2"},
        "hash": "f" * 64,
        "missing": [],
        "degraded": False,
    }
    probe = se.probe_sol_environment(
        matrix,
        matrix_path=matrix_path,
        ambient={"PATH": "/usr/local/cuda-13.2/bin:/usr/bin"},
        fingerprint_probe=lambda _environment: current_host,
        repo_root=_REPO,
    )
    assert probe["status"] == "failed"
    assert probe["typed_error"] == "SOL_ENV_FINGERPRINT_MISMATCH"
    assert probe["typed_error"] != "SOL_ENV_MATRIX_CONTRACT_INVALID"

    malformed = copy.deepcopy(matrix)
    for cell in malformed["cells"]:
        identity = cell.get("identity") or {}
        if (
            cell.get("disposition") == "runnable"
            and identity.get("roster_source") == "sol_execbench"
        ):
            cell["invocation"]["adapter_contract"]["environment"][
                "CUDA_HOME"
            ] = "/usr/local/cuda-13.2"
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.matrix_environment_contract(malformed)
    assert caught.value.code == "SOL_ENV_MATRIX_CONTRACT_INVALID"


def test_legacy_and_full_contract_apis_are_individually_strict():
    assert se.legacy_certifying_environment_contract(
        _SOL_ENVIRONMENT_CONTRACT
    ) == se.LEGACY_PINNED_COMPILER_ENVIRONMENT
    assert se.certifying_environment_contract(
        _SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT
    ) == se.PINNED_COMPILER_ENVIRONMENT
    with pytest.raises(se.SolEnvironmentRefusal):
        se.certifying_environment_contract(_SOL_ENVIRONMENT_CONTRACT)
    with pytest.raises(se.SolEnvironmentRefusal):
        se.legacy_certifying_environment_contract(
            _SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT
        )


def test_supplemental_sol_environment_is_separate_closed_and_live_pinned(
    monkeypatch, tmp_path,
):
    canonical, _canonical_path, _fingerprint_path = (
        _write_sol_environment_fixture(tmp_path)
    )
    matrix = _supplemental_environment_matrix(canonical)
    matrix_path = tmp_path / "supplemental.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    _mock_certified_runtime(monkeypatch, tmp_path)
    probe = se.probe_supplemental_sol_environment(
        matrix, matrix_path=matrix_path, ambient={"PATH": "/usr/bin"},
        fingerprint_probe=_matching_fingerprint, repo_root=tmp_path,
    )
    assert probe["status"] == "ok", probe
    assert probe["environment_contract"] == (
        _SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT
    )
    assert probe["matrix_pin"] == matrix["input_pins"][
        "sol_native_env_fingerprint"
    ]

    mixed = copy.deepcopy(matrix)
    second = copy.deepcopy(mixed["variants"][0])
    second["slug"] = "problem_071_second"
    second["adapter_contract"]["environment"]["NVCC_CCBIN"] = "g++-15"
    second["invocation"]["adapter_contract"] = copy.deepcopy(
        second["adapter_contract"]
    )
    mixed["variants"].append(second)
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.supplemental_matrix_environment_contract(mixed)
    assert caught.value.code == "SOL_ENV_SUPPLEMENTAL_MATRIX_MIXED"

    stale = copy.deepcopy(matrix)
    stale["variants"][0]["native_env_hash"] = "f" * 64
    failed = se.probe_supplemental_sol_environment(
        stale, matrix_path=matrix_path, ambient={"PATH": "/usr/bin"},
        fingerprint_probe=_matching_fingerprint, repo_root=tmp_path,
    )
    assert failed["typed_error"] == "SOL_ENV_SUPPLEMENTAL_MATRIX_INVALID"


def test_sol_cell_attestation_is_self_sealed_and_rejects_failed_probe():
    probe = _ok_sol_environment_probe()
    identity = {"roster_source": "sol_execbench", "slug": "problem_070"}
    attestation = se.build_cell_attestation(identity=identity, probe=probe)
    unsealed = {
        key: value for key, value in attestation.items()
        if key != "attestation_sha256"
    }
    assert attestation["certified_hash_equal"] is True
    assert attestation["degraded"] is False
    assert attestation["missing_components"] == []
    assert attestation["components_sha256"] == se.canonical_sha256(
        _SOL_COMPONENTS
    )
    assert attestation["attestation_sha256"] == se.canonical_sha256(unsealed)
    with pytest.raises(se.SolEnvironmentRefusal) as caught:
        se.build_cell_attestation(
            identity=identity,
            probe={"status": "failed", "typed_error": se.AMBIENT_CONFLICT},
        )
    assert caught.value.code == "SOL_ENV_ATTESTATION_INVALID"


def test_preflight_sol_probe_wrapper_hashes_matrix_and_fails_closed(
        monkeypatch, tmp_path):
    missing = cp.probe_sol_native_environment(tmp_path / "missing.json")
    assert missing["status"] == cp.STATUS_FAILED
    assert missing["typed_error"] == "SOL_ENV_MATRIX_MISSING"

    matrix_path = tmp_path / "matrix.json"
    matrix = {
        "schema_version": 1,
        "matrix_id": "sm86_reference_capture_matrix_v1",
        "cells": [],
    }
    matrix_path.write_text(json.dumps(matrix))
    expected = _ok_sol_environment_probe()
    seen = {}

    def fake_probe(matrix, *, matrix_path, ambient):
        seen["matrix"] = matrix
        seen["matrix_path"] = matrix_path
        seen["ambient"] = ambient
        return expected

    monkeypatch.setattr(cp.sol_environment, "probe_sol_environment", fake_probe)
    result = cp.probe_sol_native_environment(matrix_path)
    assert seen["matrix"] == matrix
    assert seen["matrix_path"] == matrix_path
    assert seen["ambient"] is cp.os.environ
    assert result["status"] == cp.STATUS_OK
    assert result["matrix_path"] == str(matrix_path)
    assert result["matrix_sha256"] == se.sha256_file(matrix_path)


def test_preflight_dispatches_supplemental_matrix_without_canonical_fallback(
    monkeypatch, tmp_path,
):
    canonical, _canonical_path, _fingerprint_path = (
        _write_sol_environment_fixture(tmp_path)
    )
    matrix = _supplemental_environment_matrix(canonical)
    matrix_path = tmp_path / "supplemental.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    seen = {}

    def supplemental(candidate, *, matrix_path, ambient):
        seen["candidate"] = candidate
        seen["path"] = matrix_path
        return _ok_sol_environment_probe()

    monkeypatch.setattr(
        cp.sol_environment, "probe_supplemental_sol_environment", supplemental,
    )
    monkeypatch.setattr(
        cp.sol_environment, "probe_sol_environment",
        lambda *_args, **_kwargs: pytest.fail("canonical fallback used"),
    )
    result = cp.probe_sol_native_environment(matrix_path)
    assert seen == {"candidate": matrix, "path": matrix_path}
    assert result["matrix_sha256"] == se.sha256_file(matrix_path)


# ---- schema enforces smoke evidence on an admitted injection probe --------- #
def _minimal_manifest(injection_probe):
    return {
        "schema_version": 3,
        "arch": "sm_86",
        "captured_at_utc": "2026-07-10T12:00:00+00:00",
        "generated_at_utc": "2026-07-10T12:00:00+00:00",
        "git_head": "0" * 40,
        "working_tree_clean": True,
        "pins_sha256": "0" * 64,
        "capture_matrix_sha256": "3" * 64,
        "checks": {
            "worktree": {"status": "ok"},
            "ncu": _ok_ncu_probe(),
            "driver": {"status": "ok"},
            "nvcc": {"status": "ok"},
            "nvbit": {"status": "ok"},
            "capture_processes": {"status": "ok"},
            "clock_state": {"status": "ok"},
            "injection_runtime": injection_probe,
            "trace_budget": {"status": "ok"},
            "sol_environment": _ok_sol_environment_probe(),
        },
        "blockers": [],
        "capture_admitted": True,
    }


def test_retained_preflight_v2_schema_is_byte_exact_and_separate():
    path = (
        _REPO / "workloads" / "simbench_bw" / "schemas"
        / "campaign_preflight_manifest.v2.schema.json"
    )
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "04e117bf85a52ed936942ce2f171e11bdd02581421803301884e31ad0144110e"
    )
    historical = json.loads(payload)
    assert historical["properties"]["schema_version"]["const"] == 2
    assert _SCHEMA["properties"]["schema_version"]["const"] == 3


def test_active_schema_keeps_device_additive_but_closes_present_authority():
    historical = _minimal_manifest({"status": "failed", "detail": "not run"})
    jsonschema.validate(historical, _SCHEMA)

    current = copy.deepcopy(historical)
    current["checks"]["device"] = _ok_device_probe()
    jsonschema.validate(current, _SCHEMA)


@pytest.mark.parametrize(
    "mutation",
    ["status", "policy", "index", "count", "hash", "detail", "extra"],
)
def test_active_schema_rejects_malformed_device_authority(mutation):
    manifest = _minimal_manifest({"status": "failed", "detail": "not run"})
    device = _ok_device_probe()
    if mutation == "status":
        device["status"] = "failed"
    elif mutation == "policy":
        device["selection_policy"] = "first_row_v0"
    elif mutation == "index":
        device["selected_host_gpu_index"] = 1
    elif mutation == "count":
        device["nvidia_smi_device_count"] = 2
    elif mutation == "hash":
        device["gpu_uuid_sha256"] = "not-a-hash"
    elif mutation == "detail":
        device["detail"] = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    else:
        device["raw_uuid"] = "GPU-forbidden"
    manifest["checks"]["device"] = device
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, _SCHEMA)


def test_schema_rejects_wrapper_only_success_identity():
    manifest = _minimal_manifest({"status": "failed", "detail": "not run"})
    ncu = manifest["checks"]["ncu"]
    for field in (
        "execution_mode", "host_machine", "launcher_resolved_path",
        "launcher_sha256", "launcher_target_relpath",
    ):
        del ncu[field]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, _SCHEMA)


def test_schema_rejects_ok_injection_without_smoke_evidence():
    # status=ok but no admitted_insts / validator_ok / hashes -> must fail.
    bad = _minimal_manifest({"status": "ok"})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, _SCHEMA)


def test_schema_rejects_ok_injection_with_zero_admitted():
    bad = _minimal_manifest({
        "status": "ok", "admitted_insts": 0, "validator_ok": True,
        "threadblocks_pb_count": 4,
        "dynamic_trace_pb_sha256": "a" * 64, "stats_csv_sha256": "b" * 64,
    })
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, _SCHEMA)


def test_schema_rejects_ok_injection_with_zero_threadblocks():
    # An admitted probe with a vacuously-green validator (0 threadblock records)
    # must be rejected by the schema too, not only the runtime verdict.
    bad = _minimal_manifest({
        "status": "ok", "admitted_insts": 4800, "validator_ok": True,
        "threadblocks_pb_count": 0,
        "dynamic_trace_pb_sha256": "a" * 64, "stats_csv_sha256": "b" * 64,
    })
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, _SCHEMA)


def test_schema_accepts_ok_injection_with_full_smoke_evidence():
    good = _minimal_manifest({
        "status": "ok", "admitted_insts": 4800, "expected_insts": 4800,
        "validator_ok": True, "threadblocks_pb_count": 4,
        "root_trace_ok": True, "root_trace_pb_bytes": 100,
        "root_trace_contract_id": "sm86_addr_carry_v1",
        "root_trace_memory_payload_classes": [1, 2, 3, 6, 8],
        "root_trace_launch_count": 1,
        "full_coverage_ok": True, "grid_threadblock_count": 4,
        "decoded_threadblock_count": 4, "decoded_record_count": 4800,
        "zero_record_threadblock_count": 0,
        "sampling_sidecar_present": False, "stats_row_count": 1,
        "tracer_so_sha256": "c" * 64, "smoke_binary_sha256": "d" * 64,
        "dynamic_trace_pb_sha256": "a" * 64, "stats_csv_sha256": "b" * 64,
    })
    jsonschema.validate(good, _SCHEMA)  # must not raise

    legacy = copy.deepcopy(good)
    legacy["schema_version"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(legacy, _SCHEMA)

    ncu_mutations = []
    missing_hash = copy.deepcopy(good)
    missing_hash["checks"]["ncu"].pop("executable_sha256")
    ncu_mutations.append(missing_hash)
    null_path = copy.deepcopy(good)
    null_path["checks"]["ncu"]["resolved_path"] = None
    ncu_mutations.append(null_path)
    relative_path = copy.deepcopy(good)
    relative_path["checks"]["ncu"]["resolved_path"] = "relative/ncu"
    ncu_mutations.append(relative_path)
    malformed_hash = copy.deepcopy(good)
    malformed_hash["checks"]["ncu"]["executable_sha256"] = "not-a-hash"
    ncu_mutations.append(malformed_hash)
    wrong_version = copy.deepcopy(good)
    wrong_version["checks"]["ncu"]["version_line"] = "Version 2026.10.0"
    ncu_mutations.append(wrong_version)
    unexpected = copy.deepcopy(good)
    unexpected["checks"]["ncu"]["self_attested"] = True
    ncu_mutations.append(unexpected)
    failed_with_identity = copy.deepcopy(good)
    failed_with_identity["checks"]["ncu"]["status"] = "failed"
    ncu_mutations.append(failed_with_identity)
    for invalid in ncu_mutations:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, _SCHEMA)

    missing_compiler_evidence = copy.deepcopy(good)
    del missing_compiler_evidence["checks"]["sol_environment"][
        "compiler_identity"
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_compiler_evidence, _SCHEMA)

    wrong_literal_compiler = copy.deepcopy(good)
    wrong_literal_compiler["checks"]["sol_environment"][
        "environment_contract"
    ]["NVCC_CCBIN"] = "/usr/bin/g++-14"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(wrong_literal_compiler, _SCHEMA)


def test_schema_admits_only_exact_canonical_or_supplemental_sol_contract():
    manifest = _minimal_manifest({
        "status": "ok", "admitted_insts": 4800, "expected_insts": 4800,
        "validator_ok": True, "threadblocks_pb_count": 4,
        "root_trace_ok": True, "root_trace_pb_bytes": 100,
        "root_trace_contract_id": "sm86_addr_carry_v1",
        "root_trace_memory_payload_classes": [1, 2, 3, 6, 8],
        "root_trace_launch_count": 1,
        "full_coverage_ok": True, "grid_threadblock_count": 4,
        "decoded_threadblock_count": 4, "decoded_record_count": 4800,
        "zero_record_threadblock_count": 0,
        "sampling_sidecar_present": False, "stats_row_count": 1,
        "tracer_so_sha256": "c" * 64, "smoke_binary_sha256": "d" * 64,
        "dynamic_trace_pb_sha256": "a" * 64,
        "stats_csv_sha256": "b" * 64,
    })
    canonical_probe = manifest["checks"]["sol_environment"]
    assert canonical_probe["environment_contract"] == _SOL_ENVIRONMENT_CONTRACT
    jsonschema.validate(manifest, _SCHEMA)

    supplemental = copy.deepcopy(manifest)
    supplemental["checks"]["sol_environment"]["environment_contract"] = (
        dict(_SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT)
    )
    jsonschema.validate(supplemental, _SCHEMA)

    malformed_contracts = [
        {**_SOL_ENVIRONMENT_CONTRACT, "CUDA_HOME": "/usr/local/cuda-13.2"},
        {
            key: value
            for key, value in _SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT.items()
            if key != "CUDAHOSTCXX"
        },
        {**_SOL_ENVIRONMENT_CONTRACT, "EXTRA": "forbidden"},
        {**_SUPPLEMENTAL_SOL_ENVIRONMENT_CONTRACT, "EXTRA": "forbidden"},
    ]
    for contract in malformed_contracts:
        invalid = copy.deepcopy(manifest)
        invalid["checks"]["sol_environment"]["environment_contract"] = contract
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, _SCHEMA)


def test_schema_requires_closed_structured_nvcc_release():
    manifest = _minimal_manifest({"status": "failed", "detail": "not run"})
    for invalid_release in (
        None,
        "13.2",
        {"major": 13},
        {"major": 13, "minor": 2, "patch": 0},
        {"major": True, "minor": 2},
        {"major": 13, "minor": -1},
    ):
        invalid = copy.deepcopy(manifest)
        probe = invalid["checks"]["sol_environment"]
        if invalid_release is None:
            probe.pop("nvcc_release")
        else:
            probe["nvcc_release"] = invalid_release
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, _SCHEMA)


def test_schema_rejects_ok_injection_without_exact_full_coverage():
    probe = {
        "status": "ok", "admitted_insts": 4800, "expected_insts": 4800,
        "validator_ok": True, "threadblocks_pb_count": 4,
        "root_trace_ok": True, "root_trace_pb_bytes": 100,
        "root_trace_contract_id": "sm86_addr_carry_v1",
        "root_trace_memory_payload_classes": [1, 2, 3, 6, 8],
        "root_trace_launch_count": 1,
        "full_coverage_ok": False, "grid_threadblock_count": 4,
        "decoded_threadblock_count": 3, "decoded_record_count": 3600,
        "zero_record_threadblock_count": 0,
        "sampling_sidecar_present": False, "stats_row_count": 1,
        "tracer_so_sha256": "c" * 64, "smoke_binary_sha256": "d" * 64,
        "dynamic_trace_pb_sha256": "a" * 64, "stats_csv_sha256": "b" * 64,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_minimal_manifest(probe), _SCHEMA)


def test_schema_allows_failed_injection_without_smoke_evidence():
    # A blocked smoke need not carry hashes/counts.
    blocked = _minimal_manifest({"status": "failed", "detail": "no trace"})
    blocked["blockers"] = ["injection_runtime"]
    blocked["capture_admitted"] = False
    jsonschema.validate(blocked, _SCHEMA)  # must not raise


def test_build_manifest_never_runs_kernel_when_process_hygiene_fails(
        monkeypatch, tmp_path):
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps({
        "arch": "sm_86", "measurement_protocol": {"version": "v2.1"}}))
    budget_path = tmp_path / "budget.json"
    budget_path.write_text("{}")
    ok = {"status": cp.STATUS_OK}
    monkeypatch.setattr(cp, "probe_worktree", lambda *_: ok | {
        "git_head": "0" * 40, "working_tree_clean": True})
    monkeypatch.setattr(cp, "probe_ncu", lambda *_: _ok_ncu_probe())
    monkeypatch.setattr(cp, "probe_driver", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_device", lambda: _ok_device_probe())
    monkeypatch.setattr(cp, "probe_nvcc", lambda *_, **__: ok)
    monkeypatch.setattr(cp, "probe_nvbit", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_clock_state", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_trace_budget", lambda *_: ok)
    monkeypatch.setattr(
        cp, "probe_sol_native_environment", lambda *_: _ok_sol_environment_probe()
    )
    monkeypatch.setattr(cp, "probe_covariates", lambda: ok)
    monkeypatch.setattr(cp, "probe_capture_processes", lambda: {
        "status": cp.STATUS_FAILED, "conflicts": [{"pid": 7}]})

    def forbidden_smoke(*_args, **_kwargs):
        raise AssertionError("injection smoke ran despite a conflicting process")

    monkeypatch.setattr(cp, "probe_injection_runtime", forbidden_smoke)
    manifest = cp.build_manifest(
        pins_path=pins_path, budget_path=budget_path, output_root=tmp_path,
        ignore_relpaths=[], repo_root=_REPO)
    assert manifest["capture_admitted"] is False
    assert "capture_processes" in manifest["blockers"]
    assert manifest["checks"]["injection_runtime"]["not_run"] is True


def test_build_manifest_device_refusal_blocks_before_smoke(
    monkeypatch, tmp_path,
):
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps({
        "arch": "sm_86", "measurement_protocol": {"version": "v2.1"},
    }))
    budget_path = tmp_path / "budget.json"
    budget_path.write_text("{}")
    ok = {"status": cp.STATUS_OK}
    monkeypatch.setattr(cp, "probe_worktree", lambda *_: ok | {
        "git_head": "0" * 40, "working_tree_clean": True,
    })
    monkeypatch.setattr(cp, "probe_ncu", lambda *_: _ok_ncu_probe())
    monkeypatch.setattr(cp, "probe_driver", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_device", lambda: {
        "status": cp.STATUS_MISMATCH,
        "selection_policy": cp.DEVICE_SELECTION_POLICY,
        "selected_host_gpu_index": None,
        "nvidia_smi_device_count": 2,
        "gpu_uuid_sha256": None,
    })
    monkeypatch.setattr(cp, "probe_nvcc", lambda *_, **__: ok)
    monkeypatch.setattr(cp, "probe_nvbit", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_capture_processes", lambda: ok)
    monkeypatch.setattr(cp, "probe_clock_state", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_trace_budget", lambda *_: ok)
    monkeypatch.setattr(
        cp, "probe_sol_native_environment", lambda *_: _ok_sol_environment_probe(),
    )
    monkeypatch.setattr(cp, "probe_covariates", lambda: ok)

    def forbidden_smoke(*_args, **_kwargs):
        raise AssertionError("injection smoke ran without device authority")

    monkeypatch.setattr(cp, "probe_injection_runtime", forbidden_smoke)
    manifest = cp.build_manifest(
        pins_path=pins_path,
        budget_path=budget_path,
        output_root=tmp_path,
        ignore_relpaths=[],
        repo_root=_REPO,
    )
    assert manifest["capture_admitted"] is False
    assert manifest["blockers"] == ["device", "injection_runtime"]
    assert manifest["checks"]["injection_runtime"]["not_run"] is True


def test_build_manifest_sol_environment_refusal_blocks_before_smoke(
        monkeypatch, tmp_path):
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps({
        "arch": "sm_86", "measurement_protocol": {"version": "v2.1"},
    }))
    budget_path = tmp_path / "budget.json"
    budget_path.write_text("{}")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text("{}")
    ok = {"status": cp.STATUS_OK}
    monkeypatch.setattr(cp, "probe_worktree", lambda *_: ok | {
        "git_head": "0" * 40, "working_tree_clean": True,
    })
    monkeypatch.setattr(cp, "probe_ncu", lambda *_: _ok_ncu_probe())
    monkeypatch.setattr(cp, "probe_driver", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_device", lambda: _ok_device_probe())
    monkeypatch.setattr(cp, "probe_nvcc", lambda *_, **__: ok)
    monkeypatch.setattr(cp, "probe_nvbit", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_capture_processes", lambda: ok)
    monkeypatch.setattr(cp, "probe_clock_state", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_trace_budget", lambda *_: ok)
    monkeypatch.setattr(cp, "probe_covariates", lambda: ok)
    monkeypatch.setattr(cp, "probe_sol_native_environment", lambda path: {
        "status": cp.STATUS_FAILED,
        "typed_error": se.AMBIENT_CONFLICT,
        "detail": "ambient NVCC_CCBIN='/usr/bin/g++-15'",
        "matrix_sha256": se.sha256_file(path),
    })

    def forbidden_smoke(*_args, **_kwargs):
        raise AssertionError("GPU smoke ran after SOL environment refusal")

    monkeypatch.setattr(cp, "probe_injection_runtime", forbidden_smoke)
    manifest = cp.build_manifest(
        pins_path=pins_path,
        budget_path=budget_path,
        output_root=tmp_path,
        ignore_relpaths=[],
        repo_root=_REPO,
        matrix_path=matrix_path,
    )
    assert manifest["capture_admitted"] is False
    assert manifest["blockers"] == ["injection_runtime", "sol_environment"]
    assert manifest["checks"]["injection_runtime"]["not_run"] is True
    assert manifest["checks"]["sol_environment"]["typed_error"] == (
        se.AMBIENT_CONFLICT
    )
    assert manifest["capture_matrix_sha256"] == se.sha256_file(matrix_path)


@pytest.mark.parametrize(
    ("expected_kind", "expected_errno"),
    [
        ("capacity", errno.ENOSPC),
        ("capacity", errno.EDQUOT),
        ("integrity", errno.EIO),
        ("integrity", errno.EROFS),
    ],
)
def test_injection_smoke_storage_stop_preserves_tree_and_is_typed(
    tmp_path, monkeypatch, expected_kind, expected_errno,
):
    repo = tmp_path / "repo"
    pb_lib = repo / "pb"
    pb_lib.mkdir(parents=True)
    tracer = repo / "tracer.so"
    tracer.write_bytes(b"tracer")
    smoke_binary = repo / "smoke"
    smoke_binary.write_text("fixture")
    smoke_binary.chmod(0o755)
    smoke_root = tmp_path / "campaign" / "smoke-attempt"
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[0] == "ldd":
            return 0, "", ""
        return 1, "", "read failed: Input/output error"

    monkeypatch.setattr(cp, "_run", fake_run)
    monkeypatch.setattr(
        cp.os, "write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError(expected_errno, "injected output probe failure")
        ),
    )
    pins = {
        "injection_runtime": {
            "pb_lib_relpath": "pb",
            "tracer_so_relpath": "tracer.so",
            "smoke_binary_relpath": "smoke",
            "smoke_shape_argv": [],
        }
    }
    with pytest.raises(cp.PreflightStorageFailure) as caught:
        cp.probe_injection_runtime(
            pins, repo, smoke_temp_root=smoke_root,
        )
    assert caught.value.kind == expected_kind
    assert caught.value.error_number == expected_errno
    assert smoke_root.is_dir()
    assert (smoke_root / "smoke_out").is_dir()
    assert (smoke_root / "workload_output").is_dir()
    assert len(calls) == 2


def test_injection_smoke_input_diagnostic_is_not_output_storage(
    tmp_path, monkeypatch,
):
    repo = tmp_path / "repo"
    (repo / "pb").mkdir(parents=True)
    (repo / "tracer.so").write_bytes(b"tracer")
    smoke_binary = repo / "smoke"
    smoke_binary.write_text("fixture")
    smoke_binary.chmod(0o755)
    smoke_root = tmp_path / "campaign" / "smoke-attempt"

    def fake_run(command, **_kwargs):
        if command[0] == "ldd":
            return 0, "", ""
        return 1, "", "read failed: Input/output error"

    monkeypatch.setattr(cp, "_run", fake_run)
    result = cp.probe_injection_runtime({
        "injection_runtime": {
            "pb_lib_relpath": "pb",
            "tracer_so_relpath": "tracer.so",
            "smoke_binary_relpath": "smoke",
            "smoke_shape_argv": [],
        }
    }, repo, smoke_temp_root=smoke_root)
    assert result["status"] == cp.STATUS_FAILED
    assert "Input/output error" in result["smoke_stderr_tail"]
    assert smoke_root.is_dir()


@pytest.mark.parametrize("fault", ["short_write", "close", "unlink"])
def test_failed_smoke_output_probe_types_transaction_faults(
    tmp_path, monkeypatch, fault,
):
    traces = tmp_path / "traces"
    traces.mkdir()
    if fault == "short_write":
        monkeypatch.setattr(cp.os, "write", lambda *_args: 0)
    elif fault == "close":
        real_close = cp.os.close
        state = {"failed": False}

        def close_once(descriptor):
            if not state["failed"]:
                state["failed"] = True
                raise OSError(errno.EIO, "injected probe close failure")
            return real_close(descriptor)

        monkeypatch.setattr(cp.os, "close", close_once)
    else:
        real_unlink = Path.unlink

        def fail_probe_unlink(path, *args, **kwargs):
            if path.name.startswith(".preflight-storage-probe-"):
                raise OSError(
                    getattr(errno, "ESTALE", errno.EIO),
                    "injected probe unlink failure",
                )
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_probe_unlink)
    with pytest.raises(cp.PreflightStorageFailure) as caught:
        cp._probe_failed_smoke_output_storage(traces)
    assert caught.value.kind == cp.PREFLIGHT_STORAGE_INTEGRITY
    assert list(traces.iterdir())


def test_smoke_cleanup_storage_failure_is_typed_and_preserved(
    tmp_path, monkeypatch,
):
    tree = tmp_path / "smoke"
    tree.mkdir()
    (tree / "evidence").write_bytes(b"partial")
    monkeypatch.setattr(
        cp.shutil, "rmtree",
        lambda _path: (_ for _ in ()).throw(
            OSError(errno.EIO, "injected cleanup failure")
        ),
    )
    with pytest.raises(cp.PreflightStorageFailure) as caught:
        cp._cleanup_smoke_output(tree)
    assert caught.value.operation == "injection_smoke_cleanup"
    assert (tree / "evidence").read_bytes() == b"partial"


@pytest.mark.parametrize(
    ("boundary", "operation"),
    [
        ("root", "injection_smoke_root_read"),
        ("stats", "injection_smoke_stats_read"),
        ("validator", "injection_smoke_contract_read"),
        ("payload", "injection_smoke_payload_read"),
    ],
)
def test_generated_smoke_read_faults_are_typed(boundary, operation, tmp_path, monkeypatch):
    error_number = getattr(errno, "ESTALE", errno.EIO)
    if boundary == "root":
        path = tmp_path / "dynamic_trace.pb"
        path.write_bytes(b"trace")
        real_read = Path.read_bytes

        def fail_read(candidate):
            if candidate == path:
                raise OSError(error_number, "injected root read failure")
            return real_read(candidate)

        monkeypatch.setattr(Path, "read_bytes", fail_read)
        call = lambda: cp._validate_root_trace(path, tmp_path)
    elif boundary == "stats":
        path = tmp_path / "stats.csv"
        path.write_text("total_insts,total_reported_insts\n1,1\n")
        real_open = Path.open

        def fail_open(candidate, *args, **kwargs):
            if candidate == path:
                raise OSError(error_number, "injected stats read failure")
            return real_open(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_open)
        call = lambda: cp._parse_stats_admission_details(path)
    elif boundary == "validator":
        import import_trace_contract
        monkeypatch.setattr(
            import_trace_contract, "validate_trace_dir",
            lambda _path: (_ for _ in ()).throw(
                OSError(error_number, "injected validator read failure")
            ),
        )
        call = lambda: cp._validate_smoke_trace(tmp_path, cp.REPO_ROOT)
    else:
        from tools.trace_sampling import counting_pass
        monkeypatch.setattr(
            counting_pass, "analyze_trace_dir",
            lambda *_args: (_ for _ in ()).throw(
                OSError(error_number, "injected payload read failure")
            ),
        )
        call = lambda: cp._analyze_full_trace_coverage(
            tmp_path, {"launch_count": 1}, 1, 1, 1,
        )
    with pytest.raises(cp.PreflightStorageFailure) as caught:
        call()
    assert caught.value.operation == operation


def test_preflight_main_emits_one_machine_parseable_storage_marker(
    tmp_path, monkeypatch, capsys,
):
    pins = tmp_path / "pins.json"
    pins.write_text("{}\n")
    output = tmp_path / "manifest.json"
    smoke_root = tmp_path / "smoke"
    monkeypatch.setattr(
        cp, "_campaign_output_plan",
        lambda **_kwargs: (
            output,
            {"root": smoke_root, "traces": smoke_root / "smoke_out",
             "workload_output": smoke_root / "workload_output"},
        ),
    )

    def fail_build(**_kwargs):
        raise cp.PreflightStorageFailure(
            cp.PREFLIGHT_STORAGE_CAPACITY,
            error_number=errno.ENOSPC,
            operation="injection_smoke_tree",
            detail="injected full filesystem",
        )

    monkeypatch.setattr(cp, "build_manifest", fail_build)
    assert cp.main([
        "--pins", str(pins), "--output", str(output),
        "--smoke-temp-root", str(smoke_root),
    ]) == 3
    document = cp.parse_preflight_storage_marker(capsys.readouterr().err)
    assert document is not None
    assert document["kind"] == cp.PREFLIGHT_STORAGE_CAPACITY
    assert document["errno"] == errno.ENOSPC
    assert document["smoke_tree_preserved"] is True


# ---- host integration (needs the live toolchain) --------------------------- #
_HAVE_GPU = shutil.which("nvidia-smi") and shutil.which("ncu")


@pytest.mark.hardware_evidence
@pytest.mark.skipif(not _HAVE_GPU, reason="no live NVIDIA toolchain on this host")
def test_environment_probes_ok_on_reference_host(tmp_path):
    manifest = cp.build_manifest(
        pins_path=cp.DEFAULT_PINS,
        budget_path=cp.DEFAULT_BUDGET,
        output_root=_REPO,
        ignore_relpaths=[],
    )
    jsonschema.validate(manifest, _SCHEMA)
    # Every cheap check EXCEPT the worktree (which may be dirty while these files
    # are uncommitted) must be green on the provisioned reference host.
    for name in ("ncu", "driver", "device", "nvcc", "nvbit", "capture_processes",
                 "clock_state", "trace_budget"):
        assert manifest["checks"][name]["status"] == cp.STATUS_OK, \
            (name, manifest["checks"][name])
    # The gate is exactly "no blockers".
    assert manifest["capture_admitted"] == (not manifest["blockers"])
    assert manifest["checks"]["clock_state"]["observed"]["sm_clock_mhz"] == 1350
    assert manifest["checks"]["clock_state"]["observed"]["mem_clock_mhz"] == 10501
    # Exercise the real smoke directly even when build_manifest correctly refuses
    # to run a kernel from a dirty worktree.
    inj = cp.probe_injection_runtime(
        json.loads(cp.DEFAULT_PINS.read_text()), _REPO)
    assert inj["status"] == cp.STATUS_OK, inj
    assert inj["admitted_insts"] >= 1, inj
    assert inj["validator_ok"] is True, inj
    assert inj["threadblocks_pb_count"] >= 1, inj
    assert inj["root_trace_ok"] is True, inj
    assert inj["root_trace_contract_id"] == "sm86_addr_carry_v1", inj
    assert inj["root_trace_launch_count"] >= 1, inj
    assert {1, 2, 3, 6, 8}.issubset(inj["root_trace_memory_payload_classes"])
    assert len(inj["dynamic_trace_pb_sha256"]) == 64
    assert len(inj["stats_csv_sha256"]) == 64
    assert inj["ldd_missing_deps"] == []
