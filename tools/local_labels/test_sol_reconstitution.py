"""Tests for the SOL-ExecBench host-runnability determination."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import sol_reconstitution as sr  # noqa: E402
from tools.local_labels import campaign_exclusion_ledger as el  # noqa: E402

_RUN_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "sol_runnability.external.v2.schema.json").read_text()
)
_FP_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "sol_native_env_fingerprint.schema.json").read_text()
)


def _traces(*statuses):
    return json.dumps([{"status": s} for s in statuses])


def _probe_contract(*, corpus_verified=True, outer_timeout=2000.0):
    target = sr.probe_target_identity(sr.in_scope_cells())
    contract = {
        "probe_protocol_id": sr.PROBE_PROTOCOL_ID,
        "probe_protocol_sha256": "1" * 64,
        "probe_config_path": sr.PROBE_CONFIG.relative_to(sr.REPO_ROOT).as_posix(),
        "probe_config_sha256": sr.PROBE_CONFIG_SHA256,
        "cli_timeout_seconds": sr.CLI_TIMEOUT_SECONDS,
        "outer_timeout_seconds": outer_timeout,
        "sol_execbench_package_sha256": "2" * 64,
        "sol_execbench_package_file_count": 100,
        "sol_execbench_module_path": (
            "tools/sol_execbench/src/sol_execbench/__init__.py"
        ),
        "sol_execbench_module_sha256": sr._sha256_file(
            sr.SOL_PACKAGE_SOURCE_ROOT / "sol_execbench" / "__init__.py"
        ),
        "installed_distribution_manifest_sha256": "3" * 64,
        "installed_distribution_count": 86,
        "installed_distribution_file_count": 1000,
        "installed_distribution_record_verified": True,
        "uv_lock_sha256": "4" * 64,
        "uv_binary_sha256": "5" * 64,
        "uv_version": "uv 0.test",
        "data_annex_manifest_sha256": sr.DATA_ANNEX_MANIFEST_SHA256,
        "flashinfer_corpus_manifest_sha256": sr.FLASHINFER_CORPUS_MANIFEST_SHA256,
        "flashinfer_corpus_file_count": sr.FLASHINFER_CORPUS_FILE_COUNT,
        "flashinfer_corpus_blob_count": sr.FLASHINFER_CORPUS_BLOB_COUNT,
        "flashinfer_corpus_verified": corpus_verified,
        "flashinfer_corpus_error": (
            None if corpus_verified else "planted corpus verification failure"
        ),
        "native_env_hash": sr.CERTIFIED_NATIVE_ENV_HASH,
        **target,
        "_environment": {"PATH": os.environ.get("PATH", "")},
        "_flashinfer_corpus_root": (
            sr.FLASHINFER_CORPUS_ROOT if corpus_verified else None
        ),
        "_validated": True,
    }
    contract["resume_identity_sha256"] = sr.resume_identity_sha256(contract)
    return contract


def _full_trace(identity, status="PASSED", *, log="complete diagnostic log"):
    evaluation = {
        "status": status,
        "environment": {"hardware": "RTX_3090_Ti", "libs": {"torch": "2.12.0"}},
        "timestamp": "2026-07-16T00:00:00Z",
        "log": log,
        "correctness": None,
        "performance": None,
    }
    if status == "PASSED":
        evaluation["correctness"] = {
            "max_relative_error": 0.0,
            "max_absolute_error": 0.0,
            "has_nan": False,
            "has_inf": False,
        }
        evaluation["performance"] = {
            "latency_ms": 1.0,
            "reference_latency_ms": 0.0,
            "speedup_factor": 0.0,
        }
    return {
        "definition": "test_definition",
        "workload": {
            "uuid": identity["uuid"],
            "axes": identity["axes"],
            "inputs": {},
        },
        "solution": "baseline",
        "evaluation": evaluation,
    }


def test_sol_exec_environment_pins_cuda_gcc_and_local_source_root():
    environment = sr.sol_exec_env({
        "PATH": os.environ.get("PATH", ""),
        "FLASHINFER_TRACE_DIR": "/untrusted/ambient/corpus",
    })
    assert {
        key: environment.get(key) for key in sr.SOL_COMPILER_ENVIRONMENT
    } == sr.SOL_COMPILER_ENVIRONMENT
    assert environment["PATH"].split(os.pathsep)[:2] == [
        "/usr/local/cuda-13.2/bin",
        str(sr.SOL_VENV_PYTHON.parent),
    ]
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(
        sr.SOL_PACKAGE_SOURCE_ROOT.resolve()
    )
    assert "FLASHINFER_TRACE_DIR" not in environment


def test_flashinfer_trace_dir_is_injected_only_for_fib():
    base = {"PATH": os.environ.get("PATH", ""), "FLASHINFER_TRACE_DIR": "/ambient"}
    l1 = sr.sol_exec_env(base, category="L1")
    fib = sr.sol_exec_env(
        base, category="FIB", flashinfer_corpus_root=sr.FLASHINFER_CORPUS_ROOT
    )
    assert "FLASHINFER_TRACE_DIR" not in l1
    assert fib["FLASHINFER_TRACE_DIR"] == str(sr.FLASHINFER_CORPUS_ROOT.resolve())
    with pytest.raises(sr.PrerequisiteContractError):
        sr.sol_exec_env(base, category="FIB")


@pytest.mark.parametrize(
    "ambient",
    [
        {"PATH": "/usr/bin", "CUDA_HOME": "/opt/cuda"},
        {"PATH": "/usr/bin", "CC": "gcc-13"},
        {"PATH": "/usr/bin", "NVCC_CCBIN": "g++-15"},
        {"PATH": "/usr/bin", "NVCC_APPEND_FLAGS": ""},
    ],
)
def test_sol_exec_environment_rejects_toolchain_drift(ambient):
    with pytest.raises(RuntimeError):
        sr.sol_exec_env(ambient)


def test_real_sol_execution_identity_is_repo_local_and_cuda_132():
    environment = sr.sol_exec_env({"PATH": os.environ.get("PATH", "")})
    source = sr.require_sol_execution_identity(environment)
    source.relative_to(
        (sr.SOL_PACKAGE_SOURCE_ROOT / "sol_execbench").resolve()
    )


def test_real_native_environment_matches_certified_torch_212_stack():
    fingerprint = sr.native_env_fingerprint({"PATH": os.environ.get("PATH", "")})
    assert fingerprint["degraded"] is False
    assert fingerprint["hash"] == sr.CERTIFIED_NATIVE_ENV_HASH
    assert fingerprint["components"]["torch_version"] == "2.12.0+cu130"


def test_sol_execution_identity_rejects_external_package_source(monkeypatch):
    environment = sr.sol_exec_env({"PATH": os.environ.get("PATH", "")})
    real_which = sr.shutil.which

    def which(name, *, path=None):
        if name == "nvcc":
            return "/usr/local/cuda-13.2/bin/nvcc"
        return real_which(name, path=path)

    monkeypatch.setattr(sr.shutil, "which", which)
    monkeypatch.setattr(
        sr.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/"
                "tools/sol_execbench/src/sol_execbench/__init__.py\n"
            ),
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="module identity mismatch"):
        sr.require_sol_execution_identity(environment)


def test_sol_execution_identity_rejects_repo_local_sibling_module(monkeypatch):
    environment = sr.sol_exec_env({"PATH": os.environ.get("PATH", "")})
    real_which = sr.shutil.which

    def which(name, *, path=None):
        if name == "nvcc":
            return "/usr/local/cuda-13.2/bin/nvcc"
        return real_which(name, path=path)

    monkeypatch.setattr(sr.shutil, "which", which)
    monkeypatch.setattr(
        sr.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=str(sr.SOL_PACKAGE_SOURCE_ROOT / "sol_execbench" / "cli" / "main.py") + "\n",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="module identity mismatch"):
        sr.require_sol_execution_identity(environment)


def test_historical_probe_config_is_byte_exact_and_has_no_trailing_newline():
    payload = sr.PROBE_CONFIG.read_bytes()
    assert payload == (
        b'{"warmup_runs": 1, "iterations": 3, '
        b'"benchmark_reference": false, "lock_clocks": false}'
    )
    assert sr._sha256_bytes(payload) == sr.PROBE_CONFIG_SHA256
    assert sr.verify_probe_config() == sr.PROBE_CONFIG_DOCUMENT


def test_flashinfer_corpus_matches_filtered_data_annex_authority():
    proof = sr.verify_flashinfer_corpus()
    assert proof == {
        "root": str(sr.FLASHINFER_CORPUS_ROOT),
        "data_annex_manifest_sha256": sr.DATA_ANNEX_MANIFEST_SHA256,
        "filtered_manifest_sha256": sr.FLASHINFER_CORPUS_MANIFEST_SHA256,
        "file_count": 5244,
        "safetensors_count": 2611,
    }


def test_flashinfer_corpus_verifier_rejects_tampered_payload(tmp_path):
    annex = tmp_path / "data" / "annex"
    corpus = annex / "flashinfer_trace"
    corpus.mkdir(parents=True)
    (corpus / "payload.safetensors").write_bytes(b"original")
    row = (
        f"{sr._sha256_bytes(b'original')}  "
        "./flashinfer_trace/payload.safetensors\n"
    ).encode()
    manifest = tmp_path / "data_annex.sha256"
    manifest.write_bytes(row)
    kwargs = {
        "data_annex_root": annex,
        "manifest_path": manifest,
        "expected_manifest_sha256": sr._sha256_bytes(row),
        "expected_filtered_sha256": sr._sha256_bytes(row),
        "expected_file_count": 1,
        "expected_blob_count": 1,
    }
    assert sr.verify_flashinfer_corpus(**kwargs)["file_count"] == 1
    (corpus / "payload.safetensors").write_bytes(b"tampered")
    with pytest.raises(sr.PrerequisiteContractError, match="content mismatch"):
        sr.verify_flashinfer_corpus(**kwargs)


def test_flashinfer_capture_contract_is_category_closed_and_byte_bound(
    monkeypatch,
):
    probe = _probe_contract()
    fib = sr.flashinfer_capture_contract(probe, category="FIB")
    l1 = sr.flashinfer_capture_contract(probe, category="L1")
    assert set(fib) == set(sr.FLASHINFER_CAPTURE_CONTRACT_KEYS)
    assert fib == {
        "policy": "verified_fib_only",
        "category": "FIB",
        "environment_variable": "FLASHINFER_TRACE_DIR",
        "injection_policy": "verified_absolute_root",
        "root": "data/annex/flashinfer_trace",
        "data_annex_manifest_path": "evidence/manifests/data_annex.sha256",
        "data_annex_manifest_sha256": sr.DATA_ANNEX_MANIFEST_SHA256,
        "filtered_manifest_sha256": sr.FLASHINFER_CORPUS_MANIFEST_SHA256,
        "file_count": sr.FLASHINFER_CORPUS_FILE_COUNT,
        "safetensors_count": sr.FLASHINFER_CORPUS_BLOB_COUNT,
    }
    assert l1["injection_policy"] == "forbidden_remove_ambient"
    assert l1["root"] is None

    observed = {}

    def verify(**kwargs):
        observed.update(kwargs)
        return {"root": str(sr.FLASHINFER_CORPUS_ROOT.resolve())}

    monkeypatch.setattr(sr, "verify_flashinfer_corpus", verify)
    assert sr.verify_flashinfer_capture_contract(fib) == (
        sr.FLASHINFER_CORPUS_ROOT.resolve()
    )
    assert observed == {
        "data_annex_root": sr.REPO_ROOT / "data/annex",
        "manifest_path": sr.REPO_ROOT / "evidence/manifests/data_annex.sha256",
        "expected_manifest_sha256": sr.DATA_ANNEX_MANIFEST_SHA256,
        "expected_filtered_sha256": sr.FLASHINFER_CORPUS_MANIFEST_SHA256,
        "expected_file_count": sr.FLASHINFER_CORPUS_FILE_COUNT,
        "expected_blob_count": sr.FLASHINFER_CORPUS_BLOB_COUNT,
    }
    assert sr.verify_flashinfer_capture_contract(l1) is None
    tampered = dict(fib, root="data/annex/other")
    with pytest.raises(sr.PrerequisiteContractError, match="invalid"):
        sr.verify_flashinfer_capture_contract(tampered)


def test_sol_package_identity_changes_with_package_source(tmp_path):
    package = tmp_path / "sol_execbench"
    source = package / "src" / "sol_execbench"
    source.mkdir(parents=True)
    for name in (".python-version", "pyproject.toml", "uv.lock"):
        (package / name).write_text(name)
    module = source / "__init__.py"
    module.write_text("VERSION = 1\n")
    first = sr.sol_package_identity(package)
    module.write_text("VERSION = 2\n")
    second = sr.sol_package_identity(package)
    assert first["file_count"] == 4
    assert first["sha256"] != second["sha256"]


def test_installed_distribution_manifest_rejects_unlocked_drift():
    manifest = [{
        "name": "unexpected-package",
        "version": "1.0",
        "files": [{"path": "unexpected.py", "sha256": "a" * 64}],
    }]
    with pytest.raises(sr.EnvironmentContractError, match="absent from frozen uv.lock"):
        sr.validate_installed_distribution_manifest(
            manifest, {("expected-package", "1.0")}
        )


def test_installed_distribution_manifest_binds_every_declared_file():
    base = [{
        "name": "expected-package",
        "version": "1.0",
        "files": [
            {"path": "expected.py", "sha256": "a" * 64},
            {"path": "expected.dist-info/METADATA", "sha256": "b" * 64},
        ],
    }]
    assert sr.validate_installed_distribution_manifest(
        base, {("expected-package", "1.0")}
    ) == 2
    drift = json.loads(json.dumps(base))
    drift[0]["files"][1]["sha256"] = "not-a-digest"
    with pytest.raises(sr.EnvironmentContractError, match="file manifest"):
        sr.validate_installed_distribution_manifest(
            drift, {("expected-package", "1.0")}
        )


def test_installed_distribution_identity_rejects_record_hash_drift(
    tmp_path, monkeypatch
):
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "expected-package"\nversion = "1.0"\n')
    calls = 0

    def fake_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        if "sync" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=1, stdout="",
            stderr="installed distribution RECORD hash mismatch: planted",
        )

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    with pytest.raises(sr.EnvironmentContractError, match="manifest probe failed"):
        sr.installed_distribution_identity(
            {"PATH": os.environ.get("PATH", "")}, uv_binary=uv, lock_path=lock
        )
    assert calls == 2


def test_installed_distribution_identity_rejects_unpinned_production_uv(
    tmp_path, monkeypatch
):
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    monkeypatch.setattr(sr, "UV_BINARY", uv)
    monkeypatch.setattr(sr, "UV_BINARY_SHA256", "0" * 64)

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("an unpinned uv executable was invoked")

    monkeypatch.setattr(sr.subprocess, "run", forbidden_run)
    with pytest.raises(
        sr.EnvironmentContractError, match="pinned uv binary SHA-256 mismatch"
    ):
        sr.installed_distribution_identity(
            {"PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin"}
        )


def test_resume_identity_binds_protocol_package_corpus_and_toolchain():
    base = {
        key: value for key, value in _probe_contract().items()
        if not key.startswith("_") and key != "resume_identity_sha256"
    }
    original = sr.resume_identity_sha256(base)
    for field in (
        "probe_protocol_sha256",
        "sol_execbench_package_sha256",
        "installed_distribution_manifest_sha256",
        "flashinfer_corpus_manifest_sha256",
        "native_env_hash",
        "probe_target_set_sha256",
    ):
        changed = dict(base)
        changed[field] = "f" * 64
        assert sr.resume_identity_sha256(changed) != original, field


# ---- pure classifier (no GPU) ---------------------------------------------- #
def test_runnable_all_passed():
    st, _ = sr.classify_cli_outcome(returncode=0, stdout=_traces("PASSED", "PASSED", "PASSED"),
                                    stderr="", expected_workloads=3)
    assert st == sr.RUNNABLE


def test_compile_failed():
    st, _ = sr.classify_cli_outcome(
        returncode=1, stdout="",
        stderr="Compiling C++/CUDA solution...\nCompilation failed\nninja: build stopped: subcommand failed.",
        expected_workloads=7)
    assert st == sr.COMPILE_FAILED


def test_correctness_failed():
    # CLI exits 1 when not all PASSED, but traces are present.
    st, _ = sr.classify_cli_outcome(
        returncode=1, stdout=_traces("PASSED", "INVALID_REFERENCE", "PASSED"),
        stderr="", expected_workloads=3)
    assert st == sr.CORRECTNESS_FAILED


def test_host_capacity_oom_wins_over_compile():
    st, _ = sr.classify_cli_outcome(
        returncode=1, stdout="",
        stderr="RuntimeError: CUDA out of memory. Tried to allocate 20 GiB",
        expected_workloads=4)
    assert st == sr.HOST_CAPACITY_OOM


def test_eval_incomplete_short_trace_set():
    st, _ = sr.classify_cli_outcome(returncode=0, stdout=_traces("PASSED", "PASSED"),
                                    stderr="", expected_workloads=3)
    assert st == sr.EVAL_INCOMPLETE


def test_eval_error_nonzero_no_traces():
    st, _ = sr.classify_cli_outcome(returncode=139, stdout="",
                                    stderr="Segmentation fault", expected_workloads=5)
    assert st == sr.EVAL_ERROR


def test_inner_cli_timeout_is_typed_before_correctness():
    st, _ = sr.classify_cli_outcome(
        returncode=1,
        stdout="",
        stderr="subprocess.TimeoutExpired: evaluation timed out after 1500 seconds",
        expected_workloads=5,
    )
    assert st == sr.EVAL_TIMEOUT


def test_workload_timeout_trace_is_not_mislabeled_correctness():
    st, _ = sr.classify_cli_outcome(
        returncode=1,
        stdout=_traces("PASSED", "TIMEOUT"),
        stderr="",
        expected_workloads=2,
    )
    assert st == sr.EVAL_TIMEOUT


def test_all_passed_but_nonzero_exit_fails_closed():
    st, _ = sr.classify_cli_outcome(returncode=1, stdout=_traces("PASSED", "PASSED"),
                                    stderr="", expected_workloads=2)
    assert st == sr.EVAL_ERROR


def test_zero_expected_workloads_never_runnable():
    # A benchmark that declares 0 workloads cannot be confirmed runnable even if
    # the CLI returns a passing trace ('N/0 passed' is not a pass).
    st, _ = sr.classify_cli_outcome(returncode=0, stdout=_traces("PASSED"),
                                    stderr="", expected_workloads=0)
    assert st == sr.EVAL_INCOMPLETE


def test_nonstring_trace_status_does_not_raise():
    # A status that arrives as a non-string (dict/number) must not crash the
    # classifier — it is simply not "passed".
    stdout = json.dumps([{"status": {"unexpected": "object"}},
                         {"status": 7}, {"status": "PASSED"}])
    st, _ = sr.classify_cli_outcome(returncode=1, stdout=stdout, stderr="",
                                    expected_workloads=3)
    assert st == sr.CORRECTNESS_FAILED  # not all passed, but no exception


def test_parse_traces_nested_evaluation_status():
    traces = sr._parse_traces(json.dumps(
        [{"evaluation": {"status": {"value": "PASSED"}}}]))
    assert traces and traces[0]["status"] == "PASSED"


def test_directly_nested_status_dict_is_unwrapped():
    # {"status": {"value": "PASSED"}} must unwrap so a passing cell is not
    # mislabeled correctness_failed (and must never crash the classifier).
    stdout = json.dumps([{"status": {"value": "PASSED"}},
                         {"status": {"value": "PASSED"}}])
    st, _ = sr.classify_cli_outcome(returncode=0, stdout=stdout, stderr="",
                                    expected_workloads=2)
    assert st == sr.RUNNABLE


def test_compile_oom_is_compile_not_host_capacity():
    # A HOST-compiler out-of-memory during build is a compile failure, not a
    # CUDA host-capacity deferral (which would wrongly route to a shrink).
    st, _ = sr.classify_cli_outcome(
        returncode=1, stdout="",
        stderr="cc1plus: out of memory allocating 8 GiB\nninja: build stopped: subcommand failed.",
        expected_workloads=7)
    assert st == sr.COMPILE_FAILED


def test_parse_traces_jsonl_and_wrapped_object():
    assert len(sr._parse_traces('{"status":"PASSED"}\n{"status":"PASSED"}\n')) == 2
    assert len(sr._parse_traces('{"traces":[{"status":"PASSED"}]}')) == 1
    assert sr._parse_traces("") == []
    assert sr._parse_traces("not json at all") == []


# ---- typed-reason lockstep with the exclusion vocabulary ------------------- #
def test_status_to_typed_reason_is_lockstep():
    for status, reason in sr.STATUS_TO_TYPED_REASON.items():
        assert status in sr.STATUSES
        assert reason in el.REASON_VOCABULARY, (status, reason)
    # Every non-runnable status has a typed mapping (no silent gap).
    for status in sr.STATUSES:
        if status != sr.RUNNABLE:
            assert status in sr.STATUS_TO_TYPED_REASON


# ---- roster iteration + resolution (reads real repo state) ----------------- #
def test_in_scope_cells_is_201():
    cells = sr.in_scope_cells()
    assert len(cells) == 201
    assert {c["category"] for c in cells} <= {"L1", "L2", "FIB"}
    assert all(c["language"] for c in cells)


def test_resolve_paths_resolves_a_real_cell():
    cell = {"category": "FIB", "slug": "problem_001_fused_add_rmsnorm_h2048",
            "language": "cuda_cpp"}
    baseline_dir, solution_json, benchmark_dir = sr.resolve_paths(cell)
    assert solution_json.is_file()
    assert benchmark_dir.is_dir()
    assert (benchmark_dir / "workload.jsonl").is_file()


def test_resolve_paths_missing_solution_raises():
    with pytest.raises(FileNotFoundError):
        sr.resolve_paths({"category": "FIB", "slug": "problem_999_nope",
                          "language": "cuda_cpp"})


def test_workload_identity_hash_excludes_lf_only(tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    raw = b'  {"uuid":"u","axes":{"n":1}}  \r'
    workload = benchmark / "proxy.jsonl"
    workload.write_bytes(raw + b"\n")
    identity = sr._load_workload_identities(benchmark, workload)[0]
    assert identity["workload_sha256"] == sr._sha256_bytes(raw)
    assert identity["workload_sha256"] != sr._sha256_bytes(raw.strip())
    assert identity["workload_sha256_scope"] == (
        "exact_json_line_bytes_excluding_lf_only"
    )


def test_atomic_write_preserves_old_file_if_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "artifact.json"
    target.write_bytes(b"old\n")
    monkeypatch.setattr(
        sr.os, "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("planted replace failure")),
    )
    with pytest.raises(OSError, match="planted replace failure"):
        sr.atomic_write_bytes(target, b"new\n")
    assert target.read_bytes() == b"old\n"
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


def test_atomic_write_refuses_final_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"held")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(sr.PrerequisiteContractError, match="symlink"):
        sr.atomic_write_bytes(link, b"replacement")
    assert target.read_bytes() == b"held"


def test_external_publication_is_exclusive_exact_and_read_only(tmp_path):
    target = tmp_path / "matrix.json"
    payload = b'{"sealed":true}\n'
    sr.publish_external_bytes_exclusive(target, payload)
    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o222 == 0
    for existing in (payload, b'{"sealed":false}\n'):
        target.chmod(0o600)
        target.write_bytes(existing)
        with pytest.raises(
            sr.PrerequisiteContractError, match="refusing to overwrite"
        ):
            sr.publish_external_bytes_exclusive(target, payload)


def test_external_publication_refuses_symlink_target_and_ancestor(tmp_path):
    held = tmp_path / "held"
    held.write_bytes(b"held")
    target_link = tmp_path / "matrix.json"
    target_link.symlink_to(held)
    with pytest.raises(sr.PrerequisiteContractError, match="overwrite"):
        sr.publish_external_bytes_exclusive(target_link, b"replacement")
    assert held.read_bytes() == b"held"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(sr.PrerequisiteContractError, match="canonical real"):
        sr.publish_external_bytes_exclusive(
            linked_parent / "matrix.json", b"replacement"
        )
    assert not (real_parent / "matrix.json").exists()


def test_external_publication_refuses_relative_repo_local_and_missing_parent(
    tmp_path,
):
    with pytest.raises(sr.PrerequisiteContractError, match="absolute"):
        sr.publish_external_bytes_exclusive(Path("relative.json"), b"payload")
    with pytest.raises(sr.PrerequisiteContractError, match="outside"):
        sr.publish_external_bytes_exclusive(
            sr.REPO_ROOT / ".forbidden-campaign-output", b"payload"
        )
    with pytest.raises(sr.PrerequisiteContractError, match="unavailable"):
        sr.publish_external_bytes_exclusive(
            tmp_path / "missing" / "matrix.json", b"payload"
        )


def test_external_publication_removes_partial_file_on_interrupted_write(
    tmp_path, monkeypatch,
):
    target = tmp_path / "matrix.json"

    def interrupted(descriptor, payload):
        sr.os.write(descriptor, payload[:3])
        raise OSError("planted interrupted write")

    monkeypatch.setattr(sr, "_write_all", interrupted)
    with pytest.raises(OSError, match="planted interrupted write"):
        sr.publish_external_bytes_exclusive(target, b"complete payload")
    assert not target.exists()
    assert not target.is_symlink()


def test_bound_external_root_publication_and_nofollow_reads(tmp_path):
    root = tmp_path / "campaign-root"
    root.mkdir(mode=0o700)
    identity = sr.validate_external_campaign_root(root)
    published = sr.publish_external_child(identity, "authority.json", b"{}\n")
    assert published == root / "authority.json"
    assert published.stat().st_mode & 0o777 == 0o444
    assert sr.read_external_child(identity, "authority.json") == b"{}\n"
    with pytest.raises(sr.PrerequisiteContractError, match="overwrite"):
        sr.publish_external_child(identity, "authority.json", b"{}\n")

    held = tmp_path / "held.json"
    held.write_bytes(b"held\n")
    (root / "link.json").symlink_to(held)
    with pytest.raises(sr.PrerequisiteContractError, match="unreadable"):
        sr.read_external_child(identity, "link.json")
    with pytest.raises(sr.PrerequisiteContractError, match="overwrite"):
        sr.publish_external_child(identity, "link.json", b"replacement\n")
    assert held.read_bytes() == b"held\n"


def test_source_worktree_state_refuses_any_dirty_byte(monkeypatch):
    planted = SimpleNamespace(
        returncode=0, stdout=b" M tracked.py\0?? untracked.txt\0", stderr=b"",
    )
    monkeypatch.setattr(sr.subprocess, "run", lambda *_a, **_k: planted)
    with pytest.raises(sr.PrerequisiteContractError, match="completely clean"):
        sr.source_worktree_state()


def test_run_one_uses_historical_timeouts_and_keeps_complete_uuid_diagnostics(
    monkeypatch,
):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "L1")
    _, _, benchmark = sr.resolve_paths(cell)
    identities = sr._load_workload_identities(benchmark)
    stdout = "\n".join(json.dumps(_full_trace(identity)) for identity in identities) + "\n"
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="full stderr retained")

    monkeypatch.setattr(
        sr, "require_sol_execution_identity",
        lambda _env: sr.SOL_PACKAGE_SOURCE_ROOT / "sol_execbench" / "__init__.py",
    )
    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    record = sr.run_one(
        cell, timeout=2000.0, probe_contract=_probe_contract()
    )
    assert record["status"] == sr.RUNNABLE
    assert record["diagnostics_complete"] is True
    assert record["returned_workloads"] == len(identities)
    assert record["passed_workloads"] == len(identities)
    assert record["stderr"] == "full stderr retained"
    assert record["stderr_tail"] == "full stderr retained"
    assert all(row["trace"]["evaluation"]["log"] == "complete diagnostic log"
               for row in record["workload_diagnostics"])
    command = observed["command"]
    assert command[command.index("--config") + 1] == str(sr.PROBE_CONFIG)
    assert command[command.index("--timeout") + 1] == "1500"
    assert observed["kwargs"]["timeout"] == 2000.0
    assert "FLASHINFER_TRACE_DIR" not in observed["kwargs"]["env"]
    portable = record["command"]
    assert portable[0] == "tools/sol_execbench/.venv/bin/python"
    assert portable[portable.index("--config") + 1] == (
        "tools/local_labels/sol_full_correctness_config.v1.json"
    )
    assert all(
        not Path(portable[index]).is_absolute() for index in (0, 4, 6, 8, 11)
    )
    sr._require_certified_record_command(
        record,
        resume_identity=record["execution_contract_sha256"],
        key=(record["category"], record["slug"], record["language"]),
    )


def test_native_and_proxy_share_the_exact_certified_command_contract(monkeypatch):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "L1")
    _, _, benchmark = sr.resolve_paths(cell)
    identities = sr._load_workload_identities(benchmark)
    stdout = "\n".join(json.dumps(_full_trace(row)) for row in identities) + "\n"
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        sr, "require_sol_execution_identity",
        lambda _env: sr.SOL_PACKAGE_SOURCE_ROOT / "sol_execbench" / "__init__.py",
    )
    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    contract = _probe_contract()
    native = sr.run_one(cell, timeout=2000, probe_contract=contract)
    proxy = sr.run_certified(
        cell, timeout=2000, probe_contract=contract,
        workload_path=benchmark / "workload.jsonl",
    )
    assert native["status"] == proxy["status"] == sr.RUNNABLE
    assert native["execution_kind"] == "native"
    assert proxy["execution_kind"] == "proxy"
    assert commands[0] == commands[1]
    assert native["execution_contract_sha256"] == proxy["execution_contract_sha256"]


def test_proxy_definition_override_binds_and_launches_exact_selected_bytes(
    tmp_path, monkeypatch,
):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "L1")
    _, _, benchmark = sr.resolve_paths(cell)
    workload = benchmark / "workload.jsonl"
    identities = sr._load_workload_identities(benchmark, workload)
    stdout = "\n".join(json.dumps(_full_trace(row)) for row in identities) + "\n"
    selected_definition = tmp_path / "reduced-definition.json"
    selected_definition.write_bytes(b'{"name":"reduced-proxy-definition"}\n')
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        sr, "require_sol_execution_identity",
        lambda _env: sr.SOL_PACKAGE_SOURCE_ROOT / "sol_execbench" / "__init__.py",
    )
    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    record = sr.run_certified(
        cell, timeout=2000, probe_contract=_probe_contract(),
        workload_path=workload, definition_path=selected_definition,
    )

    definition_identity = {
        "path": selected_definition.resolve().as_posix(),
        "sha256": sr._sha256_file(selected_definition),
    }
    assert record["status"] == sr.RUNNABLE
    assert record["execution_kind"] == "proxy"
    assert record["definition_input"] == definition_identity
    assert record["execution_input_state"]["definition"] == {
        **definition_identity, "state": "present", "detail": None,
    }
    assert record["execution_input_state_sha256"] == sr._canonical_sha256(
        record["execution_input_state"]
    )
    assert record["command"][4] == definition_identity["path"]
    assert observed["command"][4] == str(selected_definition.resolve())
    assert observed["command"][observed["command"].index("--config") + 1] == str(
        sr.PROBE_CONFIG
    )


def test_definition_override_without_workload_override_is_typed_refusal(
    tmp_path, monkeypatch,
):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "L1")
    selected_definition = tmp_path / "definition.json"
    selected_definition.write_text('{"name":"invalid-definition-only-override"}\n')
    monkeypatch.setattr(
        sr.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("definition-only override launched"),
    )

    record = sr.run_certified(
        cell, timeout=2000, probe_contract=_probe_contract(),
        definition_path=selected_definition,
    )

    assert record["status"] == sr.PREREQUISITE_MISSING
    assert record["detail"] == "definition override requires a workload override"
    assert record["execution_kind"] == "native"
    assert record["command"] is None
    assert record["execution_input_state"]["definition"] == {
        "path": selected_definition.resolve().as_posix(),
        "state": "present",
        "sha256": sr._sha256_file(selected_definition),
        "detail": None,
    }


def test_proxy_definition_override_byte_drift_refuses_before_launch(
    tmp_path, monkeypatch,
):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "L1")
    _, _, benchmark = sr.resolve_paths(cell)
    selected_definition = tmp_path / "definition.json"
    selected_definition.write_text('{"name":"before"}\n')
    original_input_state = sr.execution_input_state
    snapshots = 0

    def snapshot_then_mutate(*args, **kwargs):
        nonlocal snapshots
        state = original_input_state(*args, **kwargs)
        snapshots += 1
        if snapshots == 1:
            selected_definition.write_text('{"name":"after"}\n')
        return state

    monkeypatch.setattr(sr, "execution_input_state", snapshot_then_mutate)
    monkeypatch.setattr(
        sr.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("drifted definition launched"),
    )
    record = sr.run_certified(
        cell, timeout=2000, probe_contract=_probe_contract(),
        workload_path=benchmark / "workload.jsonl",
        definition_path=selected_definition,
    )

    assert snapshots == 2
    assert record["status"] == sr.PREREQUISITE_MISSING
    assert record["execution_kind"] == "proxy"
    assert record["detail"] == (
        "execution input presence/bytes changed during preparation"
    )
    assert record["command"] is None


def test_run_one_timeout_is_distinct_and_preserves_partial_output(monkeypatch):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "L1")
    _, _, benchmark = sr.resolve_paths(cell)
    identity = sr._load_workload_identities(benchmark)[0]
    partial = json.dumps(_full_trace(identity)) + "\n"

    def timeout(*_args, **_kwargs):
        raise sr.subprocess.TimeoutExpired(
            cmd="probe", timeout=2000, output=partial, stderr="timeout diagnostic"
        )

    monkeypatch.setattr(
        sr, "require_sol_execution_identity",
        lambda _env: sr.SOL_PACKAGE_SOURCE_ROOT / "sol_execbench" / "__init__.py",
    )
    monkeypatch.setattr(sr.subprocess, "run", timeout)
    record = sr.run_one(cell, timeout=2000.0, probe_contract=_probe_contract())
    assert record["status"] == sr.EVAL_TIMEOUT
    assert record["stdout"] == partial
    assert record["stderr"] == "timeout diagnostic"
    assert record["returned_workloads"] == 1
    assert record["workload_diagnostics"][0]["trace"] == _full_trace(identity)


def test_fib_corpus_failure_is_prerequisite_not_correctness_and_never_launches(
    monkeypatch,
):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "FIB")
    monkeypatch.setattr(
        sr.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not launch"),
    )
    record = sr.run_one(
        cell, timeout=2000.0, probe_contract=_probe_contract(corpus_verified=False)
    )
    assert record["status"] == sr.PREREQUISITE_MISSING
    assert "planted corpus verification failure" in record["detail"]
    assert record["command"] is None
    assert record["workload_diagnostics"]
    assert all(not row["diagnostic_present"] for row in record["workload_diagnostics"])


def test_environment_refusal_is_distinct_and_precedes_correctness(
    monkeypatch,
):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "L1")
    monkeypatch.setattr(
        sr, "sol_exec_env",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("wrong compiler")),
    )
    monkeypatch.setattr(
        sr.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not launch"),
    )
    record = sr.run_one(cell, timeout=2000.0, probe_contract=_probe_contract())
    assert record["status"] == sr.ENVIRONMENT_INVALID
    assert "wrong compiler" in record["detail"]
    assert record["returned_workloads"] == 0


def test_outer_timeout_below_2000_is_prerequisite_and_never_launches(monkeypatch):
    cell = next(cell for cell in sr.in_scope_cells() if cell["category"] == "L1")
    monkeypatch.setattr(
        sr.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not launch"),
    )
    record = sr.run_one(
        cell, timeout=1999.0, probe_contract=_probe_contract(outer_timeout=1999.0)
    )
    assert record["status"] == sr.PREREQUISITE_MISSING
    assert "below the required 2000s" in record["detail"]


@pytest.mark.parametrize(
    "limit,slugs", [(1, None), (None, {"problem_001_x"})]
)
def test_probe_rejects_partial_canonical_modes_before_any_probe(
    tmp_path, monkeypatch, limit, slugs
):
    monkeypatch.setattr(
        sr, "native_env_fingerprint",
        lambda *_args, **_kwargs: pytest.fail("partial mode reached environment probe"),
    )
    with pytest.raises(sr.PrerequisiteContractError, match="partial canonical"):
        sr._probe(
            tmp_path / "runnability.json", limit=limit, timeout=2000, slugs=slugs
        )


@pytest.mark.parametrize("flag,value", [("--limit", "1"), ("--slugs", "problem_x")])
def test_probe_cli_has_no_partial_mode(tmp_path, flag, value):
    with pytest.raises(SystemExit) as exc:
        sr.main(["probe", "--out", str(tmp_path / "out.json"), flag, value])
    assert exc.value.code == 2
    assert not (tmp_path / "out.json").exists()


def _negative_checkpoint_record(cell, input_state):
    def identity(name):
        item = input_state[name]
        return (
            {"path": item["path"], "sha256": item["sha256"]}
            if item["state"] == "present" else None
        )

    return sr._finish_record(
        {
            **cell,
            "prior_done": False,
            "prior_deferred_correctness": False,
            "execution_kind": "native",
            "definition_input": identity("definition"),
            "workload_input": identity("workload"),
            "solution_input": identity("solution"),
            "execution_contract_sha256": None,
            "execution_input_state": input_state,
            "execution_input_state_sha256": sr._canonical_sha256(input_state),
        },
        [], status=sr.RESOLVE_ERROR, detail="planted missing input",
        outer_timeout_seconds=2000,
    )


@pytest.mark.parametrize("appearing_input", ["solution", "definition", "workload"])
def test_negative_checkpoint_is_not_reused_after_input_appears(
    tmp_path, monkeypatch, appearing_input
):
    baselines = tmp_path / "baselines"
    benchmarks = tmp_path / "benchmarks"
    monkeypatch.setattr(sr, "BASELINES_ROOT", baselines)
    monkeypatch.setattr(sr, "BENCHMARK_ROOT", benchmarks)
    cell = {"category": "L1", "slug": "problem_x", "language": "cuda_cpp"}
    solution = baselines / "L1/problem_x/cuda_cpp/solution.json"
    benchmark = benchmarks / "L1/definition_x"
    solution.parent.mkdir(parents=True)
    benchmark.mkdir(parents=True)
    definition = benchmark / "definition.json"
    workload = benchmark / "workload.jsonl"
    if appearing_input != "solution":
        solution.write_text('{"definition":"definition_x"}\n')
    if appearing_input != "definition":
        definition.write_text('{"name":"definition_x"}\n')
    if appearing_input != "workload":
        workload.write_text('{"uuid":"u","axes":{"n":1}}\n')

    before = sr.execution_input_state(cell)
    record = _negative_checkpoint_record(cell, before)
    contract = _probe_contract()
    assert sr._checkpoint_record_is_resumable(record, cell, contract)

    if appearing_input == "solution":
        solution.write_text('{"definition":"definition_x"}\n')
    elif appearing_input == "definition":
        definition.write_text('{"name":"definition_x"}\n')
    else:
        workload.write_text('{"uuid":"u","axes":{"n":1}}\n')
    assert not sr._checkpoint_record_is_resumable(record, cell, contract)


def test_summarize_counts():
    recs = [{"status": sr.RUNNABLE}, {"status": sr.COMPILE_FAILED},
            {"status": sr.COMPILE_FAILED}, {"status": sr.HOST_CAPACITY_OOM}]
    s = sr.summarize(recs)
    assert s["total"] == 4 and s["runnable"] == 1
    assert s["by_status"]["compile_failed"] == 2


# ---- artifact schema validation -------------------------------------------- #
def _source_closure_fixture(records):
    return {
        "fixed_files": [{"path": "tools/local_labels/producer.py", "sha256": "6" * 64}],
        "targets": [
            {
                "category": row["category"], "slug": row["slug"],
                "language": row["language"],
                "execution_input_state": row["execution_input_state"],
            }
            for row in records
        ],
    }


def _valid_runnability_artifact(status=sr.COMPILE_FAILED):
    records = []
    for index in range(201):
        input_state = {
            "definition": {
                "path": None, "state": "unresolved", "sha256": None,
                "detail": "solution_not_present",
            },
            "workload": {
                "path": None, "state": "unresolved", "sha256": None,
                "detail": "solution_not_present",
            },
            "solution": {
                "path": f"missing/solution-{index}.json", "state": "missing",
                "sha256": None, "detail": None,
            },
        }
        identity = {
            "workload_index": 0,
            "source_line": 1,
            "uuid": f"00000000-0000-0000-0000-{index:012d}",
            "axes": {"batch_size": 1},
            "workload_sha256": "9" * 64,
            "workload_sha256_scope": (
                "exact_json_line_bytes_excluding_lf_only"
            ),
        }
        records.append(sr._finish_record(
            {
                "slug": f"problem_{index:03d}_x",
                "category": "FIB",
                "language": "cuda_cpp",
                "prior_done": False,
                "prior_deferred_correctness": False,
                "execution_kind": "native",
                "definition_input": None,
                "workload_input": None,
                "solution_input": None,
                "execution_contract_sha256": None,
                "execution_input_state": input_state,
                "execution_input_state_sha256": sr._canonical_sha256(input_state),
            },
            [identity],
            status=status,
            detail="test outcome",
            outer_timeout_seconds=2000.0,
        ))
    contract = {
        key: value for key, value in _probe_contract().items()
        if not key.startswith("_")
    }
    return {
        "schema_version": sr.ARTIFACT_SCHEMA_VERSION,
        "artifact_id": sr.ARTIFACT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": sr.CERTIFIED_NATIVE_ENV_HASH,
        "native_env_degraded": False,
        "publication_state": "complete",
        **contract,
        "source_commit": "a" * 40,
        "source_worktree_state": [],
        "source_worktree_state_sha256": sr._canonical_sha256([]),
        "probe_source_closure": _source_closure_fixture(records),
        "probe_source_closure_file_count": 6,
        "probe_source_closure_target_count": 201,
        "probe_source_closure_sha256": "6" * 64,
        "external_campaign_owner": {
            "name": sr.RUNNABILITY_OWNER_MARKER_NAME,
            "sha256": "7" * 64,
            "owner_seal_sha256": "8" * 64,
        },
        "external_output_name": "runnability.json",
        "external_checkpoint_name": "runnability.json.partial",
        "in_scope_total": 201,
        "probed": 201,
        "pending": 0,
        "summary": sr.summarize(records),
        "status_to_typed_reason": sr.STATUS_TO_TYPED_REASON,
        "cells": records,
    }


def test_runnability_artifact_schema_valid():
    jsonschema.validate(_valid_runnability_artifact(), _RUN_SCHEMA)


def test_runnability_schema_rejects_bad_status():
    bad = _valid_runnability_artifact()
    bad["cells"][0]["status"] = "gpu_sad"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, _RUN_SCHEMA)


def test_external_runnability_schema_rejects_unknown_cell_field():
    bad = _valid_runnability_artifact()
    bad["cells"][0]["unsealed_surprise"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, _RUN_SCHEMA)


def _strict_runnability_artifact():
    cells = sr.in_scope_cells()
    contract = _probe_contract()
    records = []
    for cell in cells:
        input_state = sr.execution_input_state(cell)

        def identity(name):
            item = input_state[name]
            return (
                {"path": item["path"], "sha256": item["sha256"]}
                if item["state"] == "present" else None
            )

        records.append(sr._finish_record(
            {
                **cell,
                "execution_kind": "native",
                "definition_input": identity("definition"),
                "workload_input": identity("workload"),
                "solution_input": identity("solution"),
                "execution_contract_sha256": None,
                "execution_input_state": input_state,
                "execution_input_state_sha256": sr._canonical_sha256(input_state),
            },
            [], status=sr.COMPILE_FAILED, detail="test compile failure",
            outer_timeout_seconds=2000,
        ))
    return {
        "schema_version": sr.ARTIFACT_SCHEMA_VERSION,
        "artifact_id": sr.ARTIFACT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": sr.CERTIFIED_NATIVE_ENV_HASH,
        "native_env_degraded": False,
        "publication_state": "complete",
        **sr.public_probe_contract(contract),
        "source_commit": "a" * 40,
        "source_worktree_state": [],
        "source_worktree_state_sha256": sr._canonical_sha256([]),
        "probe_source_closure": _source_closure_fixture(records),
        "probe_source_closure_file_count": 6,
        "probe_source_closure_target_count": 201,
        "probe_source_closure_sha256": "6" * 64,
        "external_campaign_owner": {
            "name": sr.RUNNABILITY_OWNER_MARKER_NAME,
            "sha256": "7" * 64,
            "owner_seal_sha256": "8" * 64,
        },
        "external_output_name": "runnability.json",
        "external_checkpoint_name": "runnability.json.partial",
        "in_scope_total": 201,
        "probed": 201,
        "pending": 0,
        "summary": sr.summarize(records),
        "status_to_typed_reason": sr.STATUS_TO_TYPED_REASON,
        "cells": records,
    }, contract


def test_completed_runnability_validator_accepts_exact_201_scope():
    artifact, contract = _strict_runnability_artifact()
    sr.validate_completed_runnability(artifact, expected_contract=contract)


def test_checkpoint_rejects_unknown_cell_field_without_salvage():
    artifact, contract = _strict_runnability_artifact()
    artifact["publication_state"] = "checkpoint"
    artifact["cells"][0]["unsealed_surprise"] = True
    expected_cells = sr.in_scope_cells()
    expected_by_key = {
        (row["category"], row["slug"], row["language"]): row
        for row in expected_cells
    }
    external_contract = {
        key: artifact[key] for key in (
            "source_commit", "source_worktree_state",
            "source_worktree_state_sha256", "probe_source_closure_file_count",
            "probe_source_closure",
            "probe_source_closure_target_count", "probe_source_closure_sha256",
            "external_campaign_owner", "external_output_name",
            "external_checkpoint_name",
        )
    }
    with pytest.raises(
        sr.PrerequisiteContractError, match="violates external-v2 schema"
    ):
        sr._validated_checkpoint_records(
            artifact, contract=contract, external_contract=external_contract,
            expected_by_key=expected_by_key,
        )


def test_completed_runnability_rejects_faked_missing_input_state():
    artifact, contract = _strict_runnability_artifact()
    record = artifact["cells"][0]
    record["execution_input_state"]["solution"] = {
        "path": record["execution_input_state"]["solution"]["path"],
        "state": "missing", "sha256": None, "detail": None,
    }
    record["solution_input"] = None
    record["execution_input_state_sha256"] = sr._canonical_sha256(
        record["execution_input_state"]
    )
    with pytest.raises(sr.PrerequisiteContractError, match="input state drift"):
        sr.validate_completed_runnability(artifact, expected_contract=contract)


@pytest.mark.parametrize("mutation", ["partial", "checkpoint", "contract_drift"])
def test_completed_runnability_validator_rejects_partial_or_stale(mutation):
    artifact, contract = _strict_runnability_artifact()
    if mutation == "partial":
        artifact["cells"].pop()
        artifact["probed"] = 200
        artifact["pending"] = 1
    elif mutation == "checkpoint":
        artifact["publication_state"] = "checkpoint"
    else:
        artifact["sol_execbench_package_sha256"] = "f" * 64
        artifact["resume_identity_sha256"] = sr.resume_identity_sha256(artifact)
    with pytest.raises(sr.PrerequisiteContractError):
        sr.validate_completed_runnability(artifact, expected_contract=contract)
