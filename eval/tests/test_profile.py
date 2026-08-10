from __future__ import annotations

import hashlib
import inspect
import json
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from eval import profile
from eval.contracts.validation import canonical_sha256, load_json_file_strict, validate_result

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SOURCE = ROOT / "sim/run/tests/fixtures/addr_carry_canary_bundle/run_manifest.v2.json"
RESULT_TEMPLATE = Path(__file__).parent / "fixtures/runtime_record/sim_result_template.v2.json"
SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "b" * 40
SOURCE_ANCHOR = "c" * 64
SOURCE_FILESYSTEM = "d" * 64
SOURCE_RECEIPT = "e" * 64
SOURCE_TRACKED = "f" * 64
SOURCE_WORKTREE = "1" * 64


@dataclass(frozen=True)
class _Inputs:
    repo_root: Path
    profiler_path: Path
    manifest_path: Path
    source_authority_path: Path
    result: dict[str, Any]
    result_bytes: bytes


def _canonical_line(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def _inputs(tmp_path: Path) -> _Inputs:
    repo_root = tmp_path / "repo"
    binary_path = repo_root / profile.RELEASE_BINARY_RELATIVE
    binary_path.parent.mkdir(parents=True)
    binary_path.write_bytes(b"synthetic release sim-run image\n")
    binary_path.chmod(0o700)
    producer_path = repo_root / "eval/profile.py"
    producer_path.parent.mkdir(parents=True)
    producer_path.write_bytes(Path(profile.__file__).read_bytes())
    producer_schema = repo_root / "eval/schemas/sim_profile_record.v1.schema.json"
    producer_schema.parent.mkdir(parents=True)
    producer_schema.write_bytes(profile.PROFILE_SCHEMA_PATH.read_bytes())

    tools = tmp_path / "tools"
    tools.mkdir()
    profiler_path = tools / "perf"
    profiler_path.write_bytes(b"synthetic user-space perf image\n")
    profiler_path.chmod(0o700)

    input_root = tmp_path / "input"
    input_root.mkdir()
    manifest = load_json_file_strict(MANIFEST_SOURCE)
    assert isinstance(manifest, dict)
    manifest["scope"] = "profile-unit/non-authoritative:v1"
    launch_key = manifest["launches"][0]["launch_key"]
    launch_key["workload_id"] = "profile-unit.workload"
    launch_key["matched_execution_id"] = "profile-unit.execution"
    manifest_path = input_root / "run_manifest.v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_sha256 = profile._sha256_canonical(manifest)

    result = load_json_file_strict(RESULT_TEMPLATE)
    assert isinstance(result, dict)
    result = deepcopy(result)
    result["build_identity"]["binary_sha256"] = hashlib.sha256(binary_path.read_bytes()).hexdigest()
    result["build_identity"]["sim_git_commit"] = SOURCE_COMMIT
    result["build_identity"]["sim_git_dirty"] = False
    result["build_identity"]["claimability"] = {
        "status": "claimable",
        "basis": "clean_build",
    }
    result["configuration_identity"]["run_manifest_sha256"] = manifest_sha256
    result["configuration_identity"]["arch_card_sha256"] = manifest["configuration_evidence"]["arch_card_sha256"]
    result["configuration_identity"]["pchase_card_sha256"] = manifest["configuration_evidence"]["pchase_card_sha256"]
    result["arch_card_identity"]["card_sha256"] = manifest["configuration_evidence"]["arch_card_sha256"]
    result["arch_card_identity"]["sm_arch"] = manifest["architecture"]["arch_id"]
    result["arch_card_identity"]["sku"] = manifest["architecture"]["sku"]
    result["launches"][0]["launch_key"] = deepcopy(launch_key)
    result["diagnostics"]["per_launch"][0]["launch_key"] = deepcopy(launch_key)
    result["result_sha256"] = canonical_sha256(result, "result_sha256")
    validate_result(result)
    source_authority_path = tmp_path / "source-authority" / "gate-001"
    return _Inputs(
        repo_root,
        profiler_path,
        manifest_path,
        source_authority_path,
        result,
        _canonical_line(result),
    )


class _FakeSourceAuthority:
    def __init__(self, *, drift_on_call: int | None = None) -> None:
        self.calls = 0
        self.drift_on_call = drift_on_call

    def __call__(
        self,
        generation_path: Path,
        content_manifest_sha256: str,
        repo_root: Path,
    ) -> dict[str, Any]:
        self.calls += 1
        assert content_manifest_sha256 == SOURCE_ANCHOR
        receipt = SOURCE_RECEIPT
        if self.drift_on_call == self.calls:
            receipt = "2" * 64
        return {
            "generation_path": str(Path(generation_path).absolute()),
            "content_manifest_sha256": SOURCE_ANCHOR,
            "filesystem_generation_sha256": SOURCE_FILESYSTEM,
            "receipt_sha256": receipt,
            "source_git_commit": SOURCE_COMMIT,
            "source_git_tree": SOURCE_TREE,
            "source_git_branch": "profile-test",
            "repository_root": str(repo_root),
            "tracked_inputs_manifest_sha256": SOURCE_TRACKED,
            "checked_out_worktree_manifest_sha256": SOURCE_WORKTREE,
            "producer_commit_files": {
                "eval/profile.py": {
                    "size_bytes": (repo_root / "eval/profile.py").stat().st_size,
                    "sha256": hashlib.sha256(
                        (repo_root / "eval/profile.py").read_bytes()
                    ).hexdigest(),
                },
                "eval/schemas/sim_profile_record.v1.schema.json": {
                    "size_bytes": (
                        repo_root
                        / "eval/schemas/sim_profile_record.v1.schema.json"
                    ).stat().st_size,
                    "sha256": hashlib.sha256(
                        (
                            repo_root
                            / "eval/schemas/sim_profile_record.v1.schema.json"
                        ).read_bytes()
                    ).hexdigest(),
                },
            },
            "current_source": "matched",
            "source_clean": True,
        }


class _FakePerf:
    def __init__(
        self,
        inputs: _Inputs,
        *,
        profiled_result: bytes | None = None,
        config_return_code: int = 0,
        config_stdout: bytes = b"",
        config_stderr: bytes = b"",
        record_return_code: int = 0,
        record_stderr: bytes = b"[ perf record: captured fake samples ]\n",
        report_return_code: int = 0,
        report_stdout: bytes = (
            b"# Overhead  Command  Shared Object  Symbol\n"
            b"  100.00%  sim-run  sim-run  [.] sim_run::runner_v2::run_manifest_v2\n"
        ),
        report_stderr: bytes = b"",
        timed_out_phase: str | None = None,
        transient_replace_perf_data: bool = False,
    ) -> None:
        self.inputs = inputs
        self.profiled_result = profiled_result or inputs.result_bytes
        self.config_return_code = config_return_code
        self.config_stdout = config_stdout
        self.config_stderr = config_stderr
        self.record_return_code = record_return_code
        self.record_stderr = record_stderr
        self.report_return_code = report_return_code
        self.report_stdout = report_stdout
        self.report_stderr = report_stderr
        self.timed_out_phase = timed_out_phase
        self.transient_replace_perf_data = transient_replace_perf_data
        self.data_path: Path | None = None
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, retained: Any, *, argv: list[str], cwd: Path, environment: dict[str, str], timeout_ns: int) -> profile._Observation:
        assert cwd == self.inputs.repo_root
        assert timeout_ns > 0
        self.calls.append((list(argv), dict(environment)))
        index = len(self.calls)
        phase = (
            "unsupported_global"
            if argv[1:2] == ["--no-config"]
            else "config"
            if argv[1:] == ["config", "--list"]
            else "version"
            if argv[1:] == ["--version"]
            else "record"
            if argv[1:2] == ["record"]
            else "report"
            if argv[1:2] == ["report"]
            else "reference"
        )
        stdout = b""
        stderr = b""
        return_code = 0
        if phase == "unsupported_global":
            stderr = b"Unknown option: --no-config\n"
            return_code = 129
        elif phase == "config":
            stdout = self.config_stdout
            stderr = self.config_stderr
            return_code = self.config_return_code
        elif phase == "version":
            stdout = b"perf version 7.1.5-1\n"
        elif phase == "reference":
            stdout = self.inputs.result_bytes
        elif phase == "record":
            output_index = argv.index("--output") + 1
            self.data_path = Path(argv[output_index])
            self.data_path.write_bytes(b"PERFILE2\x00synthetic-samples")
            subject_path = Path(argv[argv.index("--") + 1])
            assert subject_path.read_bytes() == (
                self.inputs.repo_root / profile.RELEASE_BINARY_RELATIVE
            ).read_bytes()
            stdout = self.profiled_result
            stderr = self.record_stderr
            return_code = self.record_return_code
        else:
            retained_input = Path(argv[argv.index("--input") + 1])
            assert retained_input.read_bytes() == b"PERFILE2\x00synthetic-samples"
            if self.transient_replace_perf_data:
                assert self.data_path is not None
                backup = self.data_path.with_name("perf.data.transient-backup")
                self.data_path.rename(backup)
                self.data_path.write_bytes(b"PERFILE2\x00synthetic-samples")
                assert retained_input.read_bytes() == backup.read_bytes()
                self.data_path.unlink()
                backup.rename(self.data_path)
            stdout = self.report_stdout
            stderr = self.report_stderr
            return_code = self.report_return_code
        start = index * 1_000_000
        return profile._Observation(
            monotonic_start_ns=start,
            monotonic_stop_ns=start + 100_000,
            user_cpu_ns=10_000,
            system_cpu_ns=5_000,
            peak_rss_kib=1234,
            return_code=return_code,
            timed_out=self.timed_out_phase == phase,
            stdout=stdout,
            stderr=stderr,
        )


def _record(
    inputs: _Inputs,
    output: Path,
    fake: _FakePerf,
    *,
    source: _FakeSourceAuthority | None = None,
) -> dict[str, Any]:
    source = source or _FakeSourceAuthority()
    return profile._record_profile(
        inputs.manifest_path,
        output,
        timeout_seconds=2.0,
        repo_root=inputs.repo_root,
        profiler_path=inputs.profiler_path,
        spawn=fake,
        source_authority_generation=inputs.source_authority_path,
        source_authority_content_manifest_sha256=SOURCE_ANCHOR,
        source_authority_verifier=source,
    )


def test_fake_perf_publishes_closed_immutable_profile_and_reloads(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output_root = tmp_path / "profiles"
    output_root.mkdir()
    output = output_root / "profile-001"
    fake = _FakePerf(inputs)
    source = _FakeSourceAuthority()

    record = _record(inputs, output, fake, source=source)

    assert record["outcome"] == {"status": "PROFILE_MEASURED"}
    assert record["claimability"] == {
        "status": "claimable",
        "basis": "externally_anchored_clean_source_and_matching_clean_release.v1",
    }
    assert record["protocol"]["event"] == "cycles:u"
    assert record["protocol"]["call_graph"] == "dwarf"
    assert record["profiler_identity"]["version"] == "perf version 7.1.5-1"
    assert record["configuration_isolation"] == {
        "mechanism": (
            "documented_perf_config_dev_null_plus_empty_effective_config_probe.v1"
        ),
        "environment_variable": "PERF_CONFIG",
        "environment_value": "/dev/null",
        "effective_config_probe": "perf_config_list_empty.v1",
        "probe_stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "probe_stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }
    assert record["inputs"]["binary"]["sha256"] == hashlib.sha256(
        (inputs.repo_root / profile.RELEASE_BINARY_RELATIVE).read_bytes()
    ).hexdigest()
    assert record["sim_result_binding"]["file_sha256"] == hashlib.sha256(inputs.result_bytes).hexdigest()
    assert record["phase_diagnostics"]["role"] == "variable_host_diagnostics_outside_sim_result_v2"
    assert [item["path"] for item in record["artifacts"]] == sorted(profile._ARTIFACT_NAMES)
    assert sorted(path.name for path in output.iterdir()) == sorted((*profile._ARTIFACT_NAMES, profile._RECORD_NAME))
    assert stat_mode(output) == 0o555
    assert all(stat_mode(path) == 0o444 for path in output.iterdir())
    assert profile._load_profile_record(
        output,
        repo_root=inputs.repo_root,
        expected_source_authority_content_manifest_sha256=SOURCE_ANCHOR,
        source_authority_verifier=source,
    ) == record
    assert len(fake.calls) == 5
    assert source.calls == 5
    assert all(environment == dict(profile._ENVIRONMENT_ITEMS) for _argv, environment in fake.calls)
    perf_argv = [argv for argv, _environment in fake.calls if argv[0] == str(inputs.profiler_path)]
    assert perf_argv[0] == [str(inputs.profiler_path), "config", "--list"]
    assert perf_argv[1] == [str(inputs.profiler_path), "--version"]
    assert perf_argv[2][1] == "record"
    assert perf_argv[3][1] == "report"
    assert all("--no-config" not in argv for argv in perf_argv)
    subject = record["supervision"]["profile_subject"]
    assert subject["pid_authority"] == "unavailable_not_claimed"
    assert subject["hierarchy"] == (
        "perf_spawned_descendant_without_independent_pid_authority.v1"
    )
    assert subject["descendant_cleanup"] == (
        "perf_session_process_group_terminated_before_wrapper_wait4.v1"
    )
    assert record["supervision"]["report_input"]["transport"] == (
        "proc_parent_retained_perf_data_inode.v1"
    )
    assert record["source_authority"]["source_git_commit"] == SOURCE_COMMIT
    assert record["source_authority"]["producer_profile"]["sha256"] == hashlib.sha256(
        (inputs.repo_root / "eval/profile.py").read_bytes()
    ).hexdigest()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_profile_publication_is_exclusive_no_replace(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    output = root / "same-generation"
    _record(inputs, output, _FakePerf(inputs))

    with pytest.raises(profile.ProfileRecorderError) as captured:
        _record(inputs, output, _FakePerf(inputs))

    assert captured.value.code == "PROFILE_FAILED"
    assert profile._load_profile_record(
        output,
        repo_root=inputs.repo_root,
        expected_source_authority_content_manifest_sha256=SOURCE_ANCHOR,
        source_authority_verifier=_FakeSourceAuthority(),
    )["outcome"]["status"] == "PROFILE_MEASURED"


def test_profiled_result_must_be_byte_identical(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    alternate = deepcopy(inputs.result)
    alternate["diagnostics"]["per_launch"][0]["diagnostic_last_issue_cycles"] += 1
    alternate["result_sha256"] = canonical_sha256(alternate, "result_sha256")
    validate_result(alternate)
    root = tmp_path / "profiles"
    root.mkdir()

    with pytest.raises(profile.ProfileRecorderError) as captured:
        _record(inputs, root / "nondeterministic", _FakePerf(inputs, profiled_result=_canonical_line(alternate)))

    assert captured.value.code == "PROFILE_NONDETERMINISM"


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (
            profile._Observation(1, 2, 0, 0, 0, 1, False, b"", b"Permission denied: perf_event_paranoid\n"),
            "PROFILE_PERMISSION_DENIED",
        ),
        (
            profile._Observation(1, 2, 0, 0, 0, 1, False, b"", b"event is not supported\n"),
            "PROFILE_UNSUPPORTED",
        ),
        (
            profile._Observation(1, 2, 0, 0, 0, 1, False, b"", b"unexpected perf failure\n"),
            "PROFILE_FAILED",
        ),
        (
            profile._Observation(1, 2, 0, 0, 0, -9, True, b"", b""),
            "PROFILE_TIMEOUT",
        ),
    ],
)
def test_perf_failures_are_typed(observation: profile._Observation, expected: str) -> None:
    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile._classify_perf_failure(observation, phase="record")
    assert captured.value.code == expected


def test_missing_profiler_is_typed_unavailable(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile._record_profile(
            inputs.manifest_path,
            root / "missing",
            timeout_seconds=1.0,
            repo_root=inputs.repo_root,
            profiler_path=tmp_path / "does-not-exist-perf",
            spawn=_FakePerf(inputs),
            source_authority_generation=inputs.source_authority_path,
            source_authority_content_manifest_sha256=SOURCE_ANCHOR,
            source_authority_verifier=_FakeSourceAuthority(),
        )
    assert captured.value.code == "PROFILE_UNAVAILABLE"


def test_input_identity_mismatch_is_typed_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    manifest = load_json_file_strict(inputs.manifest_path)
    assert isinstance(manifest, dict)
    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile._validate_result_identity(
            inputs.result,
            binary_sha256="0" * 64,
            manifest=manifest,
            manifest_canonical_sha256=profile._sha256_canonical(manifest),
            source_git_commit=SOURCE_COMMIT,
        )
    assert captured.value.code == "PROFILE_INPUT_DRIFT"


def test_stream_overflow_is_typed() -> None:
    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile._require_stream_bound(b"xx", 1, role="fake stream")
    assert captured.value.code == "PROFILE_OVERFLOW"


def test_secret_bearing_environment_names_are_rejected() -> None:
    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile._environment({"SERVICE_API_TOKEN": "not-retained"})
    assert captured.value.code == "PROFILE_ENVIRONMENT_SECRET_FORBIDDEN"


def test_validator_rejects_secret_environment_name_even_after_reseal(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    record = _record(inputs, root / "valid", _FakePerf(inputs))
    forged = deepcopy(record)
    forged["environment"][0]["name"] = "API_TOKEN"
    forged["record_sha256"] = canonical_sha256(forged, "record_sha256")

    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile.validate_profile_record(forged)
    assert captured.value.code == "PROFILE_ENVIRONMENT_SECRET_FORBIDDEN"


def test_unsymbolized_report_is_rejected_without_fallback(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    fake = _FakePerf(inputs, report_stdout=b"# report\n  100.00% sim-run sim-run [unknown]\n")

    with pytest.raises(profile.ProfileRecorderError) as captured:
        _record(inputs, root / "unknown", fake)
    assert captured.value.code == "PROFILE_UNSUPPORTED"


def test_perf_data_transient_replace_restore_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()

    with pytest.raises(profile.ProfileRecorderError) as captured:
        _record(
            inputs,
            root / "replace-restore",
            _FakePerf(inputs, transient_replace_perf_data=True),
        )

    assert captured.value.code == "PROFILE_INPUT_DRIFT"


@pytest.mark.parametrize("dirty", [False, True])
def test_stale_or_dirty_release_subject_is_rejected(
    tmp_path: Path, dirty: bool
) -> None:
    inputs = _inputs(tmp_path)
    altered = deepcopy(inputs.result)
    if dirty:
        altered["build_identity"]["sim_git_dirty"] = True
        altered["build_identity"]["claimability"] = {
            "status": "unclaimable",
            "reason": "dirty_build",
        }
    else:
        altered["build_identity"]["sim_git_commit"] = "9" * 40
    altered["result_sha256"] = canonical_sha256(altered, "result_sha256")
    validate_result(altered)
    changed_inputs = replace(
        inputs,
        result=altered,
        result_bytes=_canonical_line(altered),
    )
    root = tmp_path / "profiles"
    root.mkdir()

    with pytest.raises(profile.ProfileRecorderError) as captured:
        _record(changed_inputs, root / "rejected-source", _FakePerf(changed_inputs))

    assert captured.value.code == "PROFILE_SOURCE_AUTHORITY_REJECTED"


def test_source_authority_drift_during_profile_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    source = _FakeSourceAuthority(drift_on_call=2)

    with pytest.raises(profile.ProfileRecorderError) as captured:
        _record(inputs, root / "source-drift", _FakePerf(inputs), source=source)

    assert captured.value.code == "PROFILE_SOURCE_AUTHORITY_REJECTED"


def test_validator_rejects_exact_subject_pid_overclaim(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    record = _record(inputs, root / "valid-hierarchy", _FakePerf(inputs))
    forged = deepcopy(record)
    forged["supervision"]["profile_subject"]["pid_authority"] = "exact_child"
    forged["record_sha256"] = canonical_sha256(forged, "record_sha256")

    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile.validate_profile_record(forged)

    assert captured.value.code == "PROFILE_RECORD_INVALID"


@pytest.mark.parametrize(
    ("gate_code", "expected"),
    [
        ("GATE_STREAM_OVERSIZED", "PROFILE_OVERFLOW"),
        ("GATE_STREAM_READ_FAILED", "PROFILE_FAILED"),
        ("GATE_STREAM_DRAIN_TIMEOUT", "PROFILE_FAILED"),
        ("GATE_STREAM_CAPTURE_UNAVAILABLE", "PROFILE_FAILED"),
    ],
)
def test_only_actual_gate_stream_oversize_maps_to_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_code: str,
    expected: str,
) -> None:
    def fail_spawn(**_kwargs: Any) -> Any:
        raise profile._gate_supervision.GateReceiptError(gate_code, "synthetic")

    monkeypatch.setattr(
        profile._gate_supervision,
        "_spawn_exact_child",
        fail_spawn,
    )
    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile._spawn_exact(
            type("SyntheticRetained", (), {"path": Path("/synthetic")})(),
            argv=["/synthetic"],
            cwd=tmp_path,
            environment={},
            timeout_ns=1,
        )
    assert captured.value.code == expected


def test_loader_requires_independent_source_anchor(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    record_path = root / "anchored"
    _record(inputs, record_path, _FakePerf(inputs))

    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile._load_profile_record(
            record_path,
            repo_root=inputs.repo_root,
            expected_source_authority_content_manifest_sha256="0" * 64,
            source_authority_verifier=_FakeSourceAuthority(),
        )

    assert captured.value.code == "PROFILE_SOURCE_AUTHORITY_REJECTED"


@pytest.mark.parametrize(
    "relative_path",
    ["eval/profile.py", "eval/schemas/sim_profile_record.v1.schema.json"],
)
def test_loader_rejects_producer_or_schema_byte_drift(
    tmp_path: Path, relative_path: str
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    record_path = root / "producer-bound"
    _record(inputs, record_path, _FakePerf(inputs))
    changed = inputs.repo_root / relative_path
    changed.write_bytes(changed.read_bytes() + b"\n")

    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile._load_profile_record(
            record_path,
            repo_root=inputs.repo_root,
            expected_source_authority_content_manifest_sha256=SOURCE_ANCHOR,
            source_authority_verifier=_FakeSourceAuthority(),
        )

    assert captured.value.code == "PROFILE_INPUT_DRIFT"


def test_validator_rejects_unsupported_global_no_config_argv(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    record = _record(inputs, root / "forced-perf-config", _FakePerf(inputs))
    forged = deepcopy(record)
    forged["invocations"]["record"]["argv"].insert(1, "--no-config")
    forged["record_sha256"] = canonical_sha256(forged, "record_sha256")

    with pytest.raises(profile.ProfileRecorderError) as captured:
        profile.validate_profile_record(forged)

    assert captured.value.code == "PROFILE_RECORD_INVALID"


def test_strict_perf_715_grammar_and_empty_config_probe_are_used(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    fake = _FakePerf(inputs)

    record = _record(inputs, root / "supported-perf-grammar", fake)

    perf_calls = [
        (argv, environment)
        for argv, environment in fake.calls
        if argv[0] == str(inputs.profiler_path)
    ]
    assert [argv[1] for argv, _environment in perf_calls] == [
        "config",
        "--version",
        "record",
        "report",
    ]
    assert all("--no-config" not in argv for argv, _environment in perf_calls)
    assert all(
        environment["PERF_CONFIG"] == "/dev/null"
        for _argv, environment in perf_calls
    )
    assert record["invocations"]["config"]["argv"] == [
        str(inputs.profiler_path),
        "config",
        "--list",
    ]


def test_ambient_user_perf_configuration_cannot_enter_child_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    ambient_home = tmp_path / "ambient-home"
    ambient_home.mkdir()
    ambient_config = ambient_home / ".perfconfig"
    ambient_config.write_text("[ui]\nshow-headers = false\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(ambient_home))
    monkeypatch.setenv("PERF_CONFIG", str(ambient_config))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "ambient-xdg"))
    root = tmp_path / "profiles"
    root.mkdir()
    fake = _FakePerf(inputs)

    record = _record(inputs, root / "ambient-config-isolated", fake)

    assert record["environment"] == [
        {"name": name, "value": value}
        for name, value in sorted(profile._ENVIRONMENT_ITEMS)
    ]
    for _argv, environment in fake.calls:
        assert environment["PERF_CONFIG"] == "/dev/null"
        assert "HOME" not in environment
        assert "XDG_CONFIG_HOME" not in environment
        assert str(ambient_config) not in environment.values()


def test_perf_config_override_and_nonempty_effective_config_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(profile.ProfileRecorderError) as override_error:
        profile._environment({"PERF_CONFIG": str(tmp_path / "ambient.perfconfig")})
    assert override_error.value.code == "PROFILE_ENVIRONMENT_NOT_AUDITED"

    inputs = _inputs(tmp_path)
    root = tmp_path / "profiles"
    root.mkdir()
    fake = _FakePerf(inputs, config_stdout=b"ui.show-headers=false\n")
    with pytest.raises(profile.ProfileRecorderError) as probe_error:
        _record(inputs, root / "config-leak", fake)
    assert probe_error.value.code == "PROFILE_ENVIRONMENT_NOT_AUDITED"


def test_schema_is_closed_draft_2020_12() -> None:
    schema = load_json_file_strict(profile.PROFILE_SCHEMA_PATH)
    assert isinstance(schema, dict)
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)


def test_profile_surface_is_structurally_independent_and_uses_exact_child() -> None:
    source = inspect.getsource(profile)
    for forbidden_import in (
        "from eval import bench",
        "from eval import labels",
        "from eval import score",
        "eval.production_evaluator",
    ):
        assert forbidden_import not in source
    assert "_gate_supervision._spawn_exact_child" in source
    assert tuple(inspect.signature(profile.record_profile).parameters) == (
        "manifest_path",
        "output_generation",
        "source_authority_generation",
        "source_authority_content_manifest_sha256",
        "timeout_seconds",
    )
