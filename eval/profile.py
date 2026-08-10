"""Target-free Linux ``perf`` profiling for the release ``sim-run`` binary.

The scientific input lane is one strict ``sim_run_manifest.v2`` and the fixed
repository release executable; an externally anchored clean-source receipt is a
producer-authority prerequisite, not a simulator feature. The module accepts no
evaluator targets or comparison evidence. Producer-owned direct children use the
repository exact-child implementation. The sim-run process created by ``perf``
is instead recorded honestly as a retained-inode descendant without independent
PID authority; no exact-child claim is made for it. A successful profile is
published as one no-replace immutable directory generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import ValidationError
from jsonschema.validators import validator_for

from eval.contracts.validation import (
    ContractValidationError,
    canonical_sha256,
    load_json_file_strict,
    loads_json_strict,
    validate_manifest,
    validate_result,
)
from tools.dev import gate_receipt as _gate_supervision
from tools.local_labels.immutable_artifact import (
    DirectoryGenerationFile,
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    PinnedDirectoryReader,
    canonical_json_bytes,
    create_exclusive_directory_stage,
    inspect_immutable_directory_generation,
    open_retained_executable,
    open_retained_regular,
    strict_json_loads,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA_PATH = ROOT / "eval/schemas/sim_profile_record.v1.schema.json"
RELEASE_BINARY_RELATIVE = Path("sim/target/release/sim-run")
DEFAULT_PERF_EXECUTABLE = Path("/usr/bin/perf")
DEFAULT_TIMEOUT_SECONDS = 1800.0
MAX_BINARY_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_PROFILE_RECORD_BYTES = 4 * 1024 * 1024
MAX_PERF_DATA_BYTES = 8 * 1024 * 1024 * 1024
MAX_TEXT_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_GENERATION_BYTES = MAX_PERF_DATA_BYTES + 3 * MAX_TEXT_ARTIFACT_BYTES + MAX_RESULT_BYTES

_RECORD_NAME = "sim_profile_record.v1.json"
_PERF_DATA_NAME = "perf.data"
_REPORT_NAME = "perf.report.txt"
_RESULT_NAME = "sim_result.v2.json"
_RECORD_STDERR_NAME = "perf-record.stderr.txt"
_REPORT_STDERR_NAME = "perf-report.stderr.txt"
_ARTIFACT_NAMES = (
    _PERF_DATA_NAME,
    _REPORT_NAME,
    _RECORD_STDERR_NAME,
    _REPORT_STDERR_NAME,
    _RESULT_NAME,
)
_PERF_CONFIG_ENVIRONMENT_NAME = "PERF_CONFIG"
_PERF_CONFIG_DISABLED_VALUE = "/dev/null"
_PERF_CONFIG_ISOLATION_MECHANISM = (
    "documented_perf_config_dev_null_plus_empty_effective_config_probe.v1"
)
_PERF_CONFIG_PROBE = "perf_config_list_empty.v1"
_ENVIRONMENT_ITEMS = (
    ("LANG", "C"),
    ("LC_ALL", "C"),
    ("NO_COLOR", "1"),
    ("PAGER", "cat"),
    (_PERF_CONFIG_ENVIRONMENT_NAME, _PERF_CONFIG_DISABLED_VALUE),
    ("PERF_PAGER", "cat"),
    ("RUST_BACKTRACE", "0"),
    ("TERM", "dumb"),
    ("TZ", "UTC"),
)
_ALLOWED_ENVIRONMENT_NAMES = frozenset(name for name, _value in _ENVIRONMENT_ITEMS)
_SECRET_FRAGMENTS = (
    "AUTH",
    "BEARER",
    "COOKIE",
    "CREDENTIAL",
    "PASSWD",
    "PASSWORD",
    "PRIVATE",
    "SECRET",
    "SESSION",
    "TOKEN",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_PERF_VERSION_RE = re.compile(r"^perf version [^\x00\r\n]{1,256}$")
_MAX_U64 = (1 << 64) - 1


class ProfileRecorderError(RuntimeError):
    """One typed fail-closed profiling outcome."""

    def __init__(
        self, code: str, detail: str, *, context: Mapping[str, Any] | None = None
    ) -> None:
        self.code = code
        self.detail = detail
        self.context = dict(context or {})
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class _Observation:
    monotonic_start_ns: int
    monotonic_stop_ns: int
    user_cpu_ns: int
    system_cpu_ns: int
    peak_rss_kib: int
    return_code: int
    timed_out: bool
    stdout: bytes
    stderr: bytes


Spawn = Callable[..., _Observation]
SourceAuthorityVerifier = Callable[[Path, str, Path], dict[str, Any]]

_PRODUCER_RELATIVE = Path("eval/profile.py")
_PRODUCER_SCHEMA_RELATIVE = Path("eval/schemas/sim_profile_record.v1.schema.json")


def _fail(code: str, detail: str, **context: Any) -> None:
    raise ProfileRecorderError(code, detail, context=context)


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _created_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _timeout_ns(timeout_seconds: float) -> int:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        _fail("PROFILE_TIMEOUT_INVALID", "profile timeout must be finite and positive")
    value = int(round(float(timeout_seconds) * 1_000_000_000))
    if value <= 0 or value > _MAX_U64:
        _fail("PROFILE_TIMEOUT_INVALID", "profile timeout is outside the v1 bound")
    return value


def _validate_environment_name(name: str) -> None:
    if type(name) is not str or _ENVIRONMENT_NAME_RE.fullmatch(name) is None:
        _fail("PROFILE_ENVIRONMENT_INVALID", "environment variable name is invalid")
    upper = name.upper()
    if (
        any(fragment in upper for fragment in _SECRET_FRAGMENTS)
        or upper.endswith("_KEY")
        or "_KEY_" in upper
    ):
        _fail(
            "PROFILE_ENVIRONMENT_SECRET_FORBIDDEN",
            "secret-bearing environment variable names are forbidden",
            name=name,
        )


def _environment(
    assignments: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, str], str]:
    values = dict(_ENVIRONMENT_ITEMS)
    for name, value in (assignments or {}).items():
        _validate_environment_name(name)
        if name not in _ALLOWED_ENVIRONMENT_NAMES:
            _fail(
                "PROFILE_ENVIRONMENT_NOT_AUDITED",
                "environment variable is outside the closed profiler allowlist",
                name=name,
            )
        if type(value) is not str or "\x00" in value or len(value.encode("utf-8")) > 4096:
            _fail("PROFILE_ENVIRONMENT_INVALID", "environment value is invalid", name=name)
        if value != values[name]:
            _fail(
                "PROFILE_ENVIRONMENT_NOT_AUDITED",
                "fixed profiler environment value cannot be overridden",
                name=name,
            )
    for name in values:
        _validate_environment_name(name)
    ordered = sorted(values.items())
    entries = [{"name": name, "value": value} for name, value in ordered]
    return entries, dict(ordered), _sha256_canonical({"environment": entries})


def _capture_affinity() -> list[int]:
    if platform.system() != "Linux" or not hasattr(os, "sched_getaffinity"):
        _fail(
            "PROFILE_UNSUPPORTED",
            "target-free profiling requires Linux process affinity support",
        )
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except OSError as exc:
        _fail("PROFILE_UNAVAILABLE", "process affinity is unavailable", errno=exc.errno)
    if not affinity:
        _fail("PROFILE_UNAVAILABLE", "process affinity set is empty")
    return affinity


def _read_cpu_model() -> str:
    try:
        payload = Path("/proc/cpuinfo").read_bytes()
    except OSError as exc:
        _fail("PROFILE_UNAVAILABLE", "CPU identity is unavailable", errno=exc.errno)
    if len(payload) > 4 * 1024 * 1024:
        _fail("PROFILE_OVERFLOW", "CPU identity exceeds its byte bound")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail("PROFILE_UNAVAILABLE", "CPU identity is not UTF-8")
    for line in text.splitlines():
        if line.startswith("model name") and ":" in line:
            model = line.split(":", 1)[1].strip()
            if model:
                return model
    _fail("PROFILE_UNAVAILABLE", "CPU model is absent")


def _host_identity(affinity: list[int]) -> dict[str, Any]:
    host = {
        "os_name": platform.system(),
        "kernel_release": platform.release(),
        "kernel_version": platform.version(),
        "machine": platform.machine(),
        "cpu_model": _read_cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpus": affinity,
    }
    if (
        host["os_name"] != "Linux"
        or not all(isinstance(host[key], str) and host[key] for key in (
            "kernel_release",
            "kernel_version",
            "machine",
            "cpu_model",
        ))
        or type(host["logical_cpu_count"]) is not int
        or host["logical_cpu_count"] <= 0
    ):
        _fail("PROFILE_UNSUPPORTED", "Linux kernel or CPU identity is incomplete")
    host["host_sha256"] = _sha256_canonical(host)
    return host


def _verify_source_authority(
    generation_path: Path,
    expected_content_manifest_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify one externally hash-anchored clean-source gate generation."""

    if _SHA256_RE.fullmatch(expected_content_manifest_sha256) is None:
        _fail(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "source-authority content-manifest anchor is malformed",
        )
    absolute = Path(os.path.abspath(os.fspath(generation_path)))
    generation_limits = {
        "max_files": 3,
        "max_directories": 1,
        "max_total_file_bytes": (
            2 * _gate_supervision._MAX_STREAM_BYTES
            + _gate_supervision._MAX_RECEIPT_BYTES
        ),
        "max_total_relative_path_bytes": 256,
    }
    try:
        with PinnedDirectoryReader(absolute) as reader:
            with reader.retain_tree_generation(
                require_readonly_files=True,
                require_readonly_directories=True,
                **generation_limits,
            ):
                verified = _gate_supervision.verify_generation(
                    absolute,
                    require_current_source=True,
                    expected_content_manifest_sha256=(
                        expected_content_manifest_sha256
                    ),
                )
                receipt_payload = reader.read(
                    _gate_supervision.RECEIPT_NAME,
                    max_bytes=_gate_supervision._MAX_RECEIPT_BYTES,
                )
                receipt = strict_json_loads(
                    receipt_payload,
                    label="profile source-authority gate receipt",
                )
                root = _gate_supervision._validate_document(receipt)
    except (_gate_supervision.GateReceiptError, ImmutableArtifactError) as exc:
        raise ProfileRecorderError(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "externally anchored current-source authority did not verify",
        ) from exc
    source = root["source"]
    git = source["git"]
    if (
        verified["status"] != "verified_anchored"
        or verified["content_manifest_anchor"] != "matched"
        or verified["current_source"] != "matched"
        or verified["outcome"] != {"status": "passed", "reason": None}
        or verified["shell_return_code"] != 0
        or git["repository_root"] != str(repo_root)
        or git["clean"] is not True
        or git["status_porcelain_v1"] != ""
    ):
        _fail(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "source authority is not anchored, current, clean, successful, and repository-local",
        )
    try:
        _return_code, tracked_payload = _gate_supervision._run_git(
            repo_root,
            [
                "ls-files",
                "--error-unmatch",
                "--",
                _PRODUCER_RELATIVE.as_posix(),
                _PRODUCER_SCHEMA_RELATIVE.as_posix(),
            ],
        )
        tracked_paths = tracked_payload.decode("utf-8", errors="strict").splitlines()
        expected_paths = sorted(
            (
                _PRODUCER_RELATIVE.as_posix(),
                _PRODUCER_SCHEMA_RELATIVE.as_posix(),
            )
        )
        if tracked_paths != expected_paths:
            _fail(
                "PROFILE_SOURCE_AUTHORITY_REJECTED",
                "profile producer and schema are not exact tracked source members",
            )
        producer_commit_files: dict[str, Any] = {}
        for relative_path in expected_paths:
            _return_code, committed_payload = _gate_supervision._run_git(
                repo_root,
                ["show", f"{git['commit']}:{relative_path}"],
            )
            producer_commit_files[relative_path] = {
                "size_bytes": len(committed_payload),
                "sha256": hashlib.sha256(committed_payload).hexdigest(),
            }
    except UnicodeDecodeError as exc:
        raise ProfileRecorderError(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "tracked producer inventory is not strict UTF-8",
        ) from exc
    except _gate_supervision.GateReceiptError as exc:
        raise ProfileRecorderError(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "tracked producer membership could not be authenticated",
        ) from exc
    return {
        "generation_path": verified["generation_path"],
        "content_manifest_sha256": verified["content_manifest_sha256"],
        "filesystem_generation_sha256": verified[
            "filesystem_generation_sha256"
        ],
        "receipt_sha256": verified["receipt_sha256"],
        "source_git_commit": git["commit"],
        "source_git_tree": git["tree"],
        "source_git_branch": git["branch"],
        "repository_root": git["repository_root"],
        "tracked_inputs_manifest_sha256": source["tracked_inputs"][
            "manifest_sha256"
        ],
        "checked_out_worktree_manifest_sha256": source[
            "checked_out_worktree"
        ]["manifest_sha256"],
        "producer_commit_files": producer_commit_files,
        "current_source": "matched",
        "source_clean": True,
    }


def _seconds_to_ns(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        _fail("PROFILE_FAILED", "child CPU usage is invalid")
    return int(round(value * 1_000_000_000))


def _spawn_exact(
    retained: Any,
    *,
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_ns: int,
) -> _Observation:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    try:
        cwd_fd = os.open(cwd, flags)
    except OSError as exc:
        _fail("PROFILE_UNAVAILABLE", "profile working directory is unavailable", errno=exc.errno)
    try:
        try:
            observed = _gate_supervision._spawn_exact_child(
                retained=retained,
                process_argv0=str(retained.path),
                arguments=list(argv[1:]),
                cwd_descriptor=cwd_fd,
                environment=environment,
                timeout_ns=timeout_ns,
            )
        except _gate_supervision.GateReceiptError as exc:
            code = (
                "PROFILE_OVERFLOW"
                if exc.code == "GATE_STREAM_OVERSIZED"
                else "PROFILE_UNSUPPORTED"
                if exc.code
                in {
                    "GATE_PLATFORM_UNSUPPORTED",
                    "GATE_REAPER_AUTHORITY_UNAVAILABLE",
                }
                else "PROFILE_FAILED"
            )
            raise ProfileRecorderError(code, "exact-child supervision failed") from exc
    finally:
        os.close(cwd_fd)
    return _Observation(
        monotonic_start_ns=observed.monotonic_start_ns,
        monotonic_stop_ns=observed.monotonic_stop_ns,
        user_cpu_ns=_seconds_to_ns(float(observed.usage.ru_utime)),
        system_cpu_ns=_seconds_to_ns(float(observed.usage.ru_stime)),
        peak_rss_kib=int(observed.usage.ru_maxrss),
        return_code=observed.return_code,
        timed_out=observed.timed_out,
        stdout=observed.stdout,
        stderr=observed.stderr,
    )


def _phase(observation: _Observation) -> dict[str, Any]:
    if observation.monotonic_stop_ns <= observation.monotonic_start_ns:
        _fail("PROFILE_FAILED", "profile phase monotonic interval is invalid")
    return {
        "monotonic_start_ns": observation.monotonic_start_ns,
        "monotonic_stop_ns": observation.monotonic_stop_ns,
        "wall_time_ns": observation.monotonic_stop_ns - observation.monotonic_start_ns,
        "user_cpu_ns": observation.user_cpu_ns,
        "system_cpu_ns": observation.system_cpu_ns,
        "peak_rss_kib": observation.peak_rss_kib,
        "return_code": observation.return_code,
        "timed_out": observation.timed_out,
    }


def _require_stream_bound(payload: bytes, maximum: int, *, role: str) -> None:
    if len(payload) > maximum:
        _fail("PROFILE_OVERFLOW", f"{role} exceeds its byte bound")


def _canonical_text(payload: bytes, *, role: str, allow_empty: bool) -> str:
    _require_stream_bound(payload, MAX_TEXT_ARTIFACT_BYTES, role=role)
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail("PROFILE_FAILED", f"{role} is not strict UTF-8")
    if "\x00" in text or "\r" in text:
        _fail("PROFILE_FAILED", f"{role} is not canonical text")
    if not text:
        if allow_empty:
            return text
        _fail("PROFILE_FAILED", f"{role} is empty")
    if not text.endswith("\n"):
        _fail("PROFILE_FAILED", f"{role} lacks its canonical final newline")
    return text


def _decode_result(payload: bytes, *, source: str) -> dict[str, Any]:
    _require_stream_bound(payload, MAX_RESULT_BYTES, role="simulator result")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = loads_json_strict(text, source=source)
    except (UnicodeDecodeError, ContractValidationError, RecursionError) as exc:
        raise ProfileRecorderError("PROFILE_FAILED", "simulator result is not strict JSON") from exc
    if not isinstance(value, dict):
        _fail("PROFILE_FAILED", "simulator result root is not an object")
    try:
        validate_result(value)
    except ContractValidationError as exc:
        raise ProfileRecorderError("PROFILE_FAILED", "simulator result contract is invalid") from exc
    if payload != _canonical_json_line(value):
        _fail("PROFILE_FAILED", "simulator result is not canonical JSON")
    return value


def _manifest_document(retained: Any) -> tuple[dict[str, Any], bytes, str]:
    try:
        payload = retained.read_bytes(max_bytes=MAX_MANIFEST_BYTES)
    except ImmutableArtifactLimitError as exc:
        raise ProfileRecorderError("PROFILE_OVERFLOW", "manifest exceeds its byte bound") from exc
    try:
        text = payload.decode("utf-8", errors="strict")
        value = loads_json_strict(text, source=str(retained.path))
        validate_manifest(value)
    except (UnicodeDecodeError, ContractValidationError, RecursionError) as exc:
        raise ProfileRecorderError("PROFILE_FAILED", "run manifest v2 admission failed") from exc
    assert isinstance(value, dict)
    return value, payload, _sha256_canonical(value)


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    binary_sha256: str,
    manifest: Mapping[str, Any],
    manifest_canonical_sha256: str,
    source_git_commit: str,
) -> None:
    build = result["build_identity"]
    if build["cargo_profile"] != "release":
        _fail("PROFILE_UNSUPPORTED", "profile subject is not release-codegen sim-run")
    if (
        build["sim_git_dirty"] is not False
        or build["claimability"]
        != {"status": "claimable", "basis": "clean_build"}
        or build["sim_git_commit"] != source_git_commit
    ):
        _fail(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "release subject is dirty or stale relative to clean source authority",
        )
    if build["binary_sha256"] != binary_sha256:
        _fail("PROFILE_INPUT_DRIFT", "simulator result names a different binary")
    if "internal_phase_timings_ns" in result["diagnostics"]:
        _fail(
            "PROFILE_UNSUPPORTED",
            "variable phase diagnostics must remain outside sim_result.v2",
        )
    configuration = result["configuration_identity"]
    if configuration["run_manifest_sha256"] != manifest_canonical_sha256:
        _fail("PROFILE_INPUT_DRIFT", "simulator result names a different manifest")
    if configuration["resolved_configuration_sha256"] != _sha256_canonical(
        result["resolved_configuration"]
    ):
        _fail("PROFILE_INPUT_DRIFT", "resolved configuration digest disagrees")
    expected = manifest["configuration_evidence"]
    if (
        configuration["arch_card_sha256"] != expected["arch_card_sha256"]
        or configuration["pchase_card_sha256"] != expected["pchase_card_sha256"]
        or result["arch_card_identity"]["card_sha256"]
        != expected["arch_card_sha256"]
    ):
        _fail("PROFILE_INPUT_DRIFT", "card or configuration identity disagrees")
    architecture = manifest["architecture"]
    if (
        result["arch_card_identity"]["sm_arch"] != architecture["arch_id"]
        or result["arch_card_identity"]["sku"] != architecture["sku"]
    ):
        _fail("PROFILE_INPUT_DRIFT", "architecture identity disagrees")


def _classify_perf_failure(observation: _Observation, *, phase: str) -> None:
    if observation.timed_out:
        _fail("PROFILE_TIMEOUT", f"perf {phase} exceeded the fixed timeout")
    _require_stream_bound(observation.stdout, MAX_RESULT_BYTES, role=f"perf {phase} stdout")
    _require_stream_bound(observation.stderr, MAX_TEXT_ARTIFACT_BYTES, role=f"perf {phase} stderr")
    if observation.return_code == 0:
        return
    diagnostic = observation.stderr.decode("utf-8", errors="replace").lower()
    if any(token in diagnostic for token in (
        "permission denied",
        "operation not permitted",
        "perf_event_paranoid",
        "no permission to enable",
        "access to performance monitoring",
    )):
        _fail("PROFILE_PERMISSION_DENIED", f"perf {phase} lacked user-space permission")
    if any(token in diagnostic for token in (
        "not supported",
        "unsupported",
        "unknown event",
        "no such event",
        "failed to open event",
    )):
        _fail("PROFILE_UNSUPPORTED", f"perf {phase} is unsupported on this host")
    _fail("PROFILE_FAILED", f"perf {phase} failed", return_code=observation.return_code)


def _config_isolation_identity(
    observation: _Observation,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Fail closed unless documented PERF_CONFIG isolation is effective."""

    _classify_perf_failure(observation, phase="config isolation probe")
    if environment.get(_PERF_CONFIG_ENVIRONMENT_NAME) != _PERF_CONFIG_DISABLED_VALUE:
        _fail(
            "PROFILE_ENVIRONMENT_NOT_AUDITED",
            "PERF_CONFIG is not forced to the documented disabled value",
        )
    if observation.stdout or observation.stderr:
        _fail(
            "PROFILE_ENVIRONMENT_NOT_AUDITED",
            "perf reported non-empty effective configuration under PERF_CONFIG=/dev/null",
        )
    return {
        "mechanism": _PERF_CONFIG_ISOLATION_MECHANISM,
        "environment_variable": _PERF_CONFIG_ENVIRONMENT_NAME,
        "environment_value": _PERF_CONFIG_DISABLED_VALUE,
        "effective_config_probe": _PERF_CONFIG_PROBE,
        "probe_stdout_sha256": hashlib.sha256(observation.stdout).hexdigest(),
        "probe_stderr_sha256": hashlib.sha256(observation.stderr).hexdigest(),
    }


def _require_symbolized_report(report: str, subject_name: str) -> None:
    subject = subject_name.lower()
    rows = [line for line in report.splitlines() if line and not line.startswith("#")]
    matching = [line for line in rows if subject in line.lower()]
    if not matching or all("[unknown]" in line.lower() for line in matching):
        _fail(
            "PROFILE_UNSUPPORTED",
            "perf report does not contain a symbolized sim-run subject row",
        )


def _write_stage_file(stage_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, 0o600, dir_fd=stage_fd)
    except OSError as exc:
        _fail("PROFILE_FAILED", "profile artifact could not be created exclusively", path=name)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                _fail("PROFILE_FAILED", "profile artifact write made no progress", path=name)
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"path": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _file_identity(retained: Any) -> dict[str, Any]:
    size = retained.snapshot[4] if hasattr(retained, "snapshot") else retained.size_bytes
    return {"path": str(retained.path), "size_bytes": size, "sha256": retained.sha256}


def _parent_retained_proc_path(descriptor: int) -> str:
    if type(descriptor) is not int or descriptor < 3:
        _fail(
            "PROFILE_FAILED",
            "retained descriptor cannot be exposed to the perf descendant",
        )
    return f"/proc/{os.getpid()}/fd/{descriptor}"


def _config_probe_argv(profiler_path: Path) -> list[str]:
    return [str(profiler_path), "config", "--list"]


def _profile_argv(
    profiler_path: Path,
    *,
    data_path: Path,
    binary_subject_path: str,
    manifest_path: Path,
) -> list[str]:
    return [
        str(profiler_path),
        "record",
        "--event=cycles:u",
        "--call-graph=dwarf",
        "--freq=999",
        "--output",
        str(data_path),
        "--",
        binary_subject_path,
        str(manifest_path),
    ]


def _report_argv(profiler_path: Path, *, retained_data_path: str) -> list[str]:
    return [
        str(profiler_path),
        "report",
        "--stdio",
        "--no-children",
        "--sort=dso,symbol",
        "--input",
        retained_data_path,
    ]


def _invocation(argv: Sequence[str], cwd: Path, environment_sha256: str) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "working_directory": str(cwd),
        "environment_sha256": environment_sha256,
    }


def _source_authority_binding(
    authority: Mapping[str, Any],
    *,
    producer: Any,
    producer_schema: Any,
) -> dict[str, Any]:
    bound = deepcopy(dict(authority))
    committed = bound.get("producer_commit_files")
    expected_committed = {
        _PRODUCER_RELATIVE.as_posix(): {
            "size_bytes": producer.size_bytes,
            "sha256": producer.sha256,
        },
        _PRODUCER_SCHEMA_RELATIVE.as_posix(): {
            "size_bytes": producer_schema.size_bytes,
            "sha256": producer_schema.sha256,
        },
    }
    if committed != expected_committed:
        _fail(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "retained profile producer or schema differs from the clean source commit",
        )
    bound["producer_profile"] = _file_identity(producer)
    bound["producer_schema"] = _file_identity(producer_schema)
    return bound


def _record_profile_impl(
    manifest_path: str | Path,
    output_generation: str | Path,
    *,
    timeout_seconds: float,
    repo_root: Path,
    profiler_path: Path,
    spawn: Spawn,
    source_authority_generation: Path,
    source_authority_content_manifest_sha256: str,
    source_authority_verifier: SourceAuthorityVerifier,
    environment_assignments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    repo_root = Path(os.path.abspath(os.fspath(repo_root)))
    output = Path(os.path.abspath(os.fspath(output_generation)))
    try:
        output.relative_to(repo_root)
    except ValueError:
        pass
    else:
        _fail(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "profile output must remain outside the clean source checkout",
        )
    producer_context = None
    try:
        producer_context = open_retained_regular(
            repo_root / _PRODUCER_RELATIVE,
            max_bytes=MAX_PROFILE_RECORD_BYTES,
        )
        schema_context = open_retained_regular(
            repo_root / _PRODUCER_SCHEMA_RELATIVE,
            max_bytes=MAX_PROFILE_RECORD_BYTES,
        )
    except ImmutableArtifactError as exc:
        if producer_context is not None:
            producer_context.close()
        raise ProfileRecorderError(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "profile producer or schema bytes are unavailable",
        ) from exc
    with producer_context as producer, schema_context as producer_schema:
        authority = source_authority_verifier(
            Path(source_authority_generation),
            source_authority_content_manifest_sha256,
            repo_root,
        )
        binding = _source_authority_binding(
            authority,
            producer=producer,
            producer_schema=producer_schema,
        )
        return _record_profile_with_source(
            manifest_path,
            output_generation,
            timeout_seconds=timeout_seconds,
            repo_root=repo_root,
            profiler_path=profiler_path,
            spawn=spawn,
            source_authority=binding,
            source_authority_generation=Path(source_authority_generation),
            source_authority_content_manifest_sha256=(
                source_authority_content_manifest_sha256
            ),
            source_authority_verifier=source_authority_verifier,
            retained_producer=producer,
            retained_producer_schema=producer_schema,
            environment_assignments=environment_assignments,
        )


def _record_profile_with_source(
    manifest_path: str | Path,
    output_generation: str | Path,
    *,
    timeout_seconds: float,
    repo_root: Path,
    profiler_path: Path,
    spawn: Spawn,
    source_authority: Mapping[str, Any],
    source_authority_generation: Path,
    source_authority_content_manifest_sha256: str,
    source_authority_verifier: SourceAuthorityVerifier,
    retained_producer: Any,
    retained_producer_schema: Any,
    environment_assignments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    timeout_ns = _timeout_ns(timeout_seconds)
    repo_root = Path(os.path.abspath(os.fspath(repo_root)))
    profiler_path = Path(os.path.abspath(os.fspath(profiler_path)))
    output = Path(os.path.abspath(os.fspath(output_generation)))
    if output.name in {"", ".", ".."}:
        _fail("PROFILE_OUTPUT_INVALID", "profile generation name is invalid")
    affinity = _capture_affinity()
    host = _host_identity(affinity)
    environment_entries, environment, environment_sha256 = _environment(environment_assignments)
    binary_path = repo_root / RELEASE_BINARY_RELATIVE

    try:
        profiler_context = open_retained_executable(profiler_path)
    except ImmutableArtifactError as exc:
        raise ProfileRecorderError("PROFILE_UNAVAILABLE", "perf executable is unavailable") from exc
    try:
        binary_context = open_retained_executable(binary_path, allow_hardlinks=True)
    except ImmutableArtifactError as exc:
        profiler_context.close()
        raise ProfileRecorderError("PROFILE_UNAVAILABLE", "release sim-run is unavailable") from exc

    with profiler_context as profiler, binary_context as binary:
        try:
            manifest_context = open_retained_regular(Path(manifest_path), max_bytes=MAX_MANIFEST_BYTES)
        except ImmutableArtifactLimitError as exc:
            raise ProfileRecorderError("PROFILE_OVERFLOW", "manifest exceeds its byte bound") from exc
        except ImmutableArtifactError as exc:
            raise ProfileRecorderError("PROFILE_UNAVAILABLE", "run manifest is unavailable") from exc
        with manifest_context as manifest:
            manifest_document, _manifest_bytes, manifest_canonical_sha256 = _manifest_document(manifest)

            config_argv = _config_probe_argv(profiler.path)
            config_observation = spawn(
                profiler,
                argv=config_argv,
                cwd=repo_root,
                environment=environment,
                timeout_ns=timeout_ns,
            )
            config_isolation = _config_isolation_identity(
                config_observation,
                environment,
            )

            version_argv = [str(profiler.path), "--version"]
            version_observation = spawn(
                profiler,
                argv=version_argv,
                cwd=repo_root,
                environment=environment,
                timeout_ns=timeout_ns,
            )
            _classify_perf_failure(version_observation, phase="version probe")
            if version_observation.stderr:
                _fail("PROFILE_UNAVAILABLE", "perf version probe emitted stderr")
            version_text = _canonical_text(
                version_observation.stdout,
                role="perf version output",
                allow_empty=False,
            ).removesuffix("\n")
            if _PERF_VERSION_RE.fullmatch(version_text) is None:
                _fail("PROFILE_UNAVAILABLE", "perf version output is not canonical")

            reference_argv = [str(binary.path), str(manifest.path)]
            reference = spawn(
                binary,
                argv=reference_argv,
                cwd=repo_root,
                environment=environment,
                timeout_ns=timeout_ns,
            )
            if reference.timed_out:
                _fail("PROFILE_TIMEOUT", "reference sim-run exceeded the fixed timeout")
            if reference.return_code != 0 or reference.stderr:
                _fail("PROFILE_FAILED", "reference sim-run did not exit cleanly")
            reference_result = _decode_result(reference.stdout, source="reference sim-run stdout")
            _validate_result_identity(
                reference_result,
                binary_sha256=binary.sha256,
                manifest=manifest_document,
                manifest_canonical_sha256=manifest_canonical_sha256,
                source_git_commit=source_authority["source_git_commit"],
            )
            profiler.verify_unchanged()
            binary.verify_unchanged()
            manifest.verify_unchanged()
            if _capture_affinity() != affinity:
                _fail("PROFILE_INPUT_DRIFT", "process affinity changed before perf record")

            try:
                stage_context = create_exclusive_directory_stage(output.parent, output.name)
            except ImmutableArtifactError as exc:
                raise ProfileRecorderError("PROFILE_FAILED", "profile generation could not be reserved") from exc
            with stage_context as stage:
                data_path = stage.staging_path / _PERF_DATA_NAME
                producer_pid = os.getpid()
                subject_proc_path = _parent_retained_proc_path(binary.descriptor)
                record_argv = _profile_argv(
                    profiler.path,
                    data_path=data_path,
                    binary_subject_path=subject_proc_path,
                    manifest_path=manifest.path,
                )
                recorded = spawn(
                    profiler,
                    argv=record_argv,
                    cwd=repo_root,
                    environment=environment,
                    timeout_ns=timeout_ns,
                )
                _classify_perf_failure(recorded, phase="record")
                profiled_result = _decode_result(recorded.stdout, source="profiled sim-run stdout")
                _validate_result_identity(
                    profiled_result,
                    binary_sha256=binary.sha256,
                    manifest=manifest_document,
                    manifest_canonical_sha256=manifest_canonical_sha256,
                    source_git_commit=source_authority["source_git_commit"],
                )
                if recorded.stdout != reference.stdout:
                    _fail(
                        "PROFILE_NONDETERMINISM",
                        "profiled and reference simulator results are not byte-identical",
                    )
                record_stderr_text = _canonical_text(
                    recorded.stderr,
                    role="perf record stderr",
                    allow_empty=True,
                )
                try:
                    perf_data_context = open_retained_regular(
                        data_path,
                        max_bytes=MAX_PERF_DATA_BYTES,
                    )
                except ImmutableArtifactLimitError as exc:
                    raise ProfileRecorderError(
                        "PROFILE_OVERFLOW", "perf.data exceeds its byte bound"
                    ) from exc
                except ImmutableArtifactError as exc:
                    raise ProfileRecorderError(
                        "PROFILE_FAILED",
                        "perf record did not produce a stable regular perf.data",
                    ) from exc
                with perf_data_context as perf_data:
                    if perf_data.size_bytes == 0:
                        _fail("PROFILE_FAILED", "perf.data is empty")
                    data_identity = {
                        "path": _PERF_DATA_NAME,
                        "size_bytes": perf_data.size_bytes,
                        "sha256": perf_data.sha256,
                    }
                    magic = os.pread(perf_data.descriptor, 8, 0)
                    perf_data.verify_unchanged()
                    if magic != b"PERFILE2":
                        _fail("PROFILE_FAILED", "raw perf.data magic is invalid")
                    perf_data_descriptor = perf_data.descriptor
                    retained_data_proc_path = _parent_retained_proc_path(
                        perf_data_descriptor
                    )
                    report_argv = _report_argv(
                        profiler.path,
                        retained_data_path=retained_data_proc_path,
                    )
                    reported = spawn(
                        profiler,
                        argv=report_argv,
                        cwd=repo_root,
                        environment=environment,
                        timeout_ns=timeout_ns,
                    )
                    _classify_perf_failure(reported, phase="report")
                    perf_data.verify_unchanged()
                    report_text = _canonical_text(
                        reported.stdout,
                        role="perf text report",
                        allow_empty=False,
                    )
                    report_stderr_text = _canonical_text(
                        reported.stderr,
                        role="perf report stderr",
                        allow_empty=True,
                    )
                    _require_symbolized_report(report_text, binary.path.name)

                profiler.verify_unchanged()
                binary.verify_unchanged()
                manifest.verify_unchanged()
                retained_producer.verify_unchanged()
                retained_producer_schema.verify_unchanged()
                if _capture_affinity() != affinity:
                    _fail("PROFILE_INPUT_DRIFT", "process affinity changed during profiling")
                refreshed_authority = _source_authority_binding(
                    source_authority_verifier(
                        source_authority_generation,
                        source_authority_content_manifest_sha256,
                        repo_root,
                    ),
                    producer=retained_producer,
                    producer_schema=retained_producer_schema,
                )
                if refreshed_authority != source_authority:
                    _fail(
                        "PROFILE_SOURCE_AUTHORITY_REJECTED",
                        "source authority changed during profiling",
                    )

                artifacts = [data_identity]
                artifacts.append(_write_stage_file(stage.descriptor, _REPORT_NAME, reported.stdout))
                artifacts.append(
                    _write_stage_file(
                        stage.descriptor,
                        _RECORD_STDERR_NAME,
                        record_stderr_text.encode("utf-8"),
                    )
                )
                artifacts.append(
                    _write_stage_file(
                        stage.descriptor,
                        _REPORT_STDERR_NAME,
                        report_stderr_text.encode("utf-8"),
                    )
                )
                artifacts.append(_write_stage_file(stage.descriptor, _RESULT_NAME, recorded.stdout))
                artifacts.sort(key=lambda item: item["path"])

                trace = profiled_result["trace_identity"]
                result_binding = {
                    "file_sha256": hashlib.sha256(recorded.stdout).hexdigest(),
                    "result_sha256": profiled_result["result_sha256"],
                    "build_identity": deepcopy(profiled_result["build_identity"]),
                    "trace_identity": {
                        "bundle_id": trace["bundle_id"],
                        "bundle_path": trace["bundle_path"],
                        "bundle_sha256": trace["bundle_sha256"],
                        "wire_version": trace["wire_version"],
                        "total_size_bytes": trace["total_size_bytes"],
                        "record_count": trace["record_count"],
                        "instruction_count": trace["instruction_count"],
                        "identity_sha256": _sha256_canonical(trace),
                    },
                    "arch_card_identity": deepcopy(profiled_result["arch_card_identity"]),
                    "configuration_identity": deepcopy(profiled_result["configuration_identity"]),
                    "resolved_configuration_sha256": _sha256_canonical(
                        profiled_result["resolved_configuration"]
                    ),
                }
                document: dict[str, Any] = {
                    "schema_version": "sim_profile_record.v1",
                    "record_sha256": "0" * 64,
                    "created_at_utc": _created_at_utc(),
                    "authority_role": "target_free_cpu_profile_diagnostic_only",
                    "claimability": {
                        "status": "claimable",
                        "basis": "externally_anchored_clean_source_and_matching_clean_release.v1",
                    },
                    "protocol": {
                        "platform": "linux_perf_user_space_only.v1",
                        "event": "cycles:u",
                        "call_graph": "dwarf",
                        "frequency_hz": 999,
                        "symbolization": "perf_report_stdio_dso_symbol.v1",
                        "determinism_gate": "direct_then_profiled_sim_result_byte_identity.v1",
                        "timeout_ns": timeout_ns,
                    },
                    "profiler_identity": {
                        **_file_identity(profiler),
                        "version": version_text,
                        "version_stdout_sha256": hashlib.sha256(version_observation.stdout).hexdigest(),
                    },
                    "configuration_isolation": config_isolation,
                    "source_authority": deepcopy(dict(source_authority)),
                    "supervision": {
                        "reference_subject": {
                            "hierarchy": "profile_producer_exact_child.v1",
                            "executable_transport": "inherited_retained_release_inode.v1",
                            "termination": "waitid_wnowait_then_wait4_exact_child.v1",
                        },
                        "profile_wrapper": {
                            "hierarchy": "profile_producer_exact_child.v1",
                            "executable_transport": "inherited_retained_perf_inode.v1",
                            "termination": "waitid_wnowait_then_wait4_exact_child.v1",
                        },
                        "profile_subject": {
                            "hierarchy": "perf_spawned_descendant_without_independent_pid_authority.v1",
                            "pid_authority": "unavailable_not_claimed",
                            "executable_transport": "proc_parent_retained_release_inode.v1",
                            "producer_pid": producer_pid,
                            "retained_descriptor": binary.descriptor,
                            "argv_path": subject_proc_path,
                            "descendant_cleanup": "perf_session_process_group_terminated_before_wrapper_wait4.v1",
                            "termination": "perf_wrapper_zero_exit_plus_valid_byte_identical_result_and_perf_data.v1",
                        },
                        "report_input": {
                            "transport": "proc_parent_retained_perf_data_inode.v1",
                            "producer_pid": producer_pid,
                            "retained_descriptor": perf_data_descriptor,
                            "argv_path": retained_data_proc_path,
                        },
                    },
                    "host_identity": host,
                    "environment": environment_entries,
                    "environment_sha256": environment_sha256,
                    "invocations": {
                        "config": _invocation(config_argv, repo_root, environment_sha256),
                        "version": _invocation(version_argv, repo_root, environment_sha256),
                        "reference": _invocation(reference_argv, repo_root, environment_sha256),
                        "record": _invocation(record_argv, repo_root, environment_sha256),
                        "report": _invocation(report_argv, repo_root, environment_sha256),
                    },
                    "inputs": {
                        "binary": _file_identity(binary),
                        "manifest": {
                            **_file_identity(manifest),
                            "canonical_sha256": manifest_canonical_sha256,
                            "schema_version": 2,
                            "scope": manifest_document["scope"],
                        },
                    },
                    "sim_result_binding": result_binding,
                    "phase_diagnostics": {
                        "role": "variable_host_diagnostics_outside_sim_result_v2",
                        "config": _phase(config_observation),
                        "version": _phase(version_observation),
                        "reference": _phase(reference),
                        "record": _phase(recorded),
                        "report": _phase(reported),
                    },
                    "artifacts": artifacts,
                    "publication": {
                        "policy": "exclusive_immutable_directory_generation.v1",
                        "generation_name": output.name,
                        "record_path": _RECORD_NAME,
                        "closed_paths": sorted((*_ARTIFACT_NAMES, _RECORD_NAME)),
                    },
                    "outcome": {"status": "PROFILE_MEASURED"},
                }
                document["record_sha256"] = canonical_sha256(document, "record_sha256")
                validate_profile_record(document)
                final_authority = _source_authority_binding(
                    source_authority_verifier(
                        source_authority_generation,
                        source_authority_content_manifest_sha256,
                        repo_root,
                    ),
                    producer=retained_producer,
                    producer_schema=retained_producer_schema,
                )
                if final_authority != source_authority:
                    _fail(
                        "PROFILE_SOURCE_AUTHORITY_REJECTED",
                        "source authority changed before profile publication",
                    )
                record_payload = _canonical_json_line(document)
                record_identity = _write_stage_file(stage.descriptor, _RECORD_NAME, record_payload)
                expected_files = tuple(
                    DirectoryGenerationFile(item["path"], item["sha256"], item["size_bytes"])
                    for item in sorted((*artifacts, record_identity), key=lambda item: item["path"])
                )
                try:
                    generation = stage.commit(
                        max_files=len(expected_files),
                        max_directories=1,
                        max_total_file_bytes=MAX_GENERATION_BYTES,
                        max_total_relative_path_bytes=4096,
                        max_file_bytes_by_path={
                            _PERF_DATA_NAME: MAX_PERF_DATA_BYTES,
                            _REPORT_NAME: MAX_TEXT_ARTIFACT_BYTES,
                            _RECORD_STDERR_NAME: MAX_TEXT_ARTIFACT_BYTES,
                            _REPORT_STDERR_NAME: MAX_TEXT_ARTIFACT_BYTES,
                            _RESULT_NAME: MAX_RESULT_BYTES,
                            _RECORD_NAME: MAX_PROFILE_RECORD_BYTES,
                        },
                        expected_files=expected_files,
                    )
                except ImmutableArtifactLimitError as exc:
                    raise ProfileRecorderError("PROFILE_OVERFLOW", "profile generation exceeds its bound") from exc
                except ImmutableArtifactError as exc:
                    raise ProfileRecorderError("PROFILE_FAILED", "profile generation publication failed") from exc
                if generation.path != output:
                    _fail("PROFILE_FAILED", "profile generation committed at an unexpected path")

    loaded = _load_profile_record(
        output,
        repo_root=repo_root,
        expected_source_authority_content_manifest_sha256=(
            source_authority_content_manifest_sha256
        ),
        source_authority_verifier=source_authority_verifier,
    )
    if loaded != document:
        _fail("PROFILE_FAILED", "published profile record did not replay byte-exactly")
    return document


def _record_profile(
    manifest_path: str | Path,
    output_generation: str | Path,
    *,
    timeout_seconds: float,
    repo_root: Path,
    profiler_path: Path,
    spawn: Spawn,
    source_authority_generation: Path,
    source_authority_content_manifest_sha256: str,
    source_authority_verifier: SourceAuthorityVerifier = _verify_source_authority,
    environment_assignments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Map every producer boundary into the closed profile outcome taxonomy."""

    try:
        return _record_profile_impl(
            manifest_path,
            output_generation,
            timeout_seconds=timeout_seconds,
            repo_root=repo_root,
            profiler_path=profiler_path,
            spawn=spawn,
            source_authority_generation=source_authority_generation,
            source_authority_content_manifest_sha256=(
                source_authority_content_manifest_sha256
            ),
            source_authority_verifier=source_authority_verifier,
            environment_assignments=environment_assignments,
        )
    except ProfileRecorderError:
        raise
    except ImmutableArtifactLimitError as exc:
        raise ProfileRecorderError(
            "PROFILE_OVERFLOW", "profile evidence exceeded an immutable byte bound"
        ) from exc
    except ImmutableArtifactError as exc:
        raise ProfileRecorderError(
            "PROFILE_INPUT_DRIFT", "retained profile identity changed"
        ) from exc
    except MemoryError as exc:
        raise ProfileRecorderError(
            "PROFILE_FAILED", "profile process exhausted a required bounded resource"
        ) from exc
    except OSError as exc:
        raise ProfileRecorderError(
            "PROFILE_FAILED",
            "profile operating-system operation failed",
            context={"errno": exc.errno},
        ) from exc


def record_profile(
    manifest_path: str | Path,
    output_generation: str | Path,
    source_authority_generation: str | Path,
    source_authority_content_manifest_sha256: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Publish one profile bound to an externally anchored clean-source gate."""

    return _record_profile(
        manifest_path,
        output_generation,
        timeout_seconds=timeout_seconds,
        repo_root=ROOT,
        profiler_path=DEFAULT_PERF_EXECUTABLE,
        spawn=_spawn_exact,
        source_authority_generation=Path(source_authority_generation),
        source_authority_content_manifest_sha256=(
            source_authority_content_manifest_sha256
        ),
    )


def _read_schema() -> dict[str, Any]:
    value = load_json_file_strict(PROFILE_SCHEMA_PATH)
    if not isinstance(value, dict):
        raise RuntimeError("profile schema root is not an object")
    return value


_PROFILE_SCHEMA = _read_schema()
_PROFILE_VALIDATOR_CLASS = validator_for(_PROFILE_SCHEMA)
_PROFILE_VALIDATOR_CLASS.check_schema(_PROFILE_SCHEMA)
_PROFILE_VALIDATOR = _PROFILE_VALIDATOR_CLASS(_PROFILE_SCHEMA)


def _schema_path(error: ValidationError) -> str:
    path = "$"
    for component in error.absolute_path:
        path += f"[{component}]" if isinstance(component, int) else f".{component}"
    return path


def validate_profile_record(document: Any) -> dict[str, Any]:
    """Validate schema and all target-free semantic bindings without opening files."""

    if not isinstance(document, dict):
        _fail("PROFILE_RECORD_INVALID", "profile record root must be an object")
    try:
        _PROFILE_VALIDATOR.validate(document)
    except ValidationError as exc:
        raise ProfileRecorderError(
            "PROFILE_RECORD_INVALID",
            exc.message,
            context={"json_path": _schema_path(exc)},
        ) from exc
    if document["record_sha256"] != canonical_sha256(document, "record_sha256"):
        _fail("PROFILE_RECORD_INVALID", "profile record canonical seal does not match")
    try:
        created = datetime.fromisoformat(document["created_at_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProfileRecorderError("PROFILE_RECORD_INVALID", "created_at_utc is invalid") from exc
    if created.utcoffset() != timezone.utc.utcoffset(created):
        _fail("PROFILE_RECORD_INVALID", "created_at_utc is not UTC")

    if document["claimability"] != {
        "status": "claimable",
        "basis": "externally_anchored_clean_source_and_matching_clean_release.v1",
    }:
        _fail("PROFILE_RECORD_INVALID", "profile claimability basis is invalid")

    environment = document["environment"]
    if environment != sorted(environment, key=lambda item: item["name"]):
        _fail("PROFILE_RECORD_INVALID", "profile environment is not canonically ordered")
    for entry in environment:
        _validate_environment_name(entry["name"])
    expected_entries, _values, expected_environment_sha256 = _environment()
    if environment != expected_entries or document["environment_sha256"] != expected_environment_sha256:
        _fail("PROFILE_RECORD_INVALID", "profile environment is not the exact audited closure")
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    if document["configuration_isolation"] != {
        "mechanism": _PERF_CONFIG_ISOLATION_MECHANISM,
        "environment_variable": _PERF_CONFIG_ENVIRONMENT_NAME,
        "environment_value": _PERF_CONFIG_DISABLED_VALUE,
        "effective_config_probe": _PERF_CONFIG_PROBE,
        "probe_stdout_sha256": empty_sha256,
        "probe_stderr_sha256": empty_sha256,
    }:
        _fail(
            "PROFILE_RECORD_INVALID",
            "perf configuration isolation is not the exact audited contract",
        )

    host = deepcopy(document["host_identity"])
    host_sha256 = host.pop("host_sha256")
    if host_sha256 != _sha256_canonical(host):
        _fail("PROFILE_RECORD_INVALID", "host identity hash does not match")
    if host["os_name"] != "Linux":
        _fail("PROFILE_RECORD_INVALID", "profile host is not Linux")

    profiler = document["profiler_identity"]
    if _PERF_VERSION_RE.fullmatch(profiler["version"]) is None:
        _fail("PROFILE_RECORD_INVALID", "perf version is not canonical")
    if profiler["path"] != os.path.abspath(profiler["path"]):
        _fail("PROFILE_RECORD_INVALID", "perf path is not canonical absolute")
    binary = document["inputs"]["binary"]
    manifest = document["inputs"]["manifest"]
    for identity in (binary, manifest):
        if identity["path"] != os.path.abspath(identity["path"]):
            _fail("PROFILE_RECORD_INVALID", "input path is not canonical absolute")
    binding = document["sim_result_binding"]
    if binding["build_identity"]["binary_sha256"] != binary["sha256"]:
        _fail("PROFILE_RECORD_INVALID", "result build does not bind the binary")
    source = document["source_authority"]
    build = binding["build_identity"]
    if (
        source["source_clean"] is not True
        or source["current_source"] != "matched"
        or build["cargo_profile"] != "release"
        or build["sim_git_dirty"] is not False
        or build["claimability"]
        != {"status": "claimable", "basis": "clean_build"}
        or build["sim_git_commit"] != source["source_git_commit"]
    ):
        _fail(
            "PROFILE_RECORD_INVALID",
            "release build is not bound to clean current source authority",
        )
    if binding["configuration_identity"]["run_manifest_sha256"] != manifest["canonical_sha256"]:
        _fail("PROFILE_RECORD_INVALID", "result configuration does not bind the manifest")
    if binding["configuration_identity"]["resolved_configuration_sha256"] != binding[
        "resolved_configuration_sha256"
    ]:
        _fail("PROFILE_RECORD_INVALID", "resolved configuration hash does not match")
    if binding["configuration_identity"]["arch_card_sha256"] != binding["arch_card_identity"]["card_sha256"]:
        _fail("PROFILE_RECORD_INVALID", "card and configuration identities disagree")

    working_directory = document["invocations"]["version"]["working_directory"]
    expected_binary_path = str(Path(working_directory) / RELEASE_BINARY_RELATIVE)
    if (
        binary["path"] != expected_binary_path
        or source["repository_root"] != working_directory
        or source["producer_profile"]["path"]
        != str(Path(working_directory) / _PRODUCER_RELATIVE)
        or source["producer_schema"]["path"]
        != str(Path(working_directory) / _PRODUCER_SCHEMA_RELATIVE)
        or source["producer_commit_files"]
        != {
            _PRODUCER_RELATIVE.as_posix(): {
                "size_bytes": source["producer_profile"]["size_bytes"],
                "sha256": source["producer_profile"]["sha256"],
            },
            _PRODUCER_SCHEMA_RELATIVE.as_posix(): {
                "size_bytes": source["producer_schema"]["size_bytes"],
                "sha256": source["producer_schema"]["sha256"],
            },
        }
    ):
        _fail(
            "PROFILE_RECORD_INVALID",
            "profile binary or producer bytes are outside the source authority",
        )
    expected_argv = {
        "config": [profiler["path"], "config", "--list"],
        "version": [profiler["path"], "--version"],
        "reference": [binary["path"], manifest["path"]],
    }
    for role, argv in expected_argv.items():
        if document["invocations"][role]["argv"] != argv:
            _fail("PROFILE_RECORD_INVALID", f"{role} argv is not exact")
    generation_name = document["publication"]["generation_name"]
    if generation_name in {".", ".."}:
        _fail("PROFILE_RECORD_INVALID", "profile generation name is not canonical")
    record_argv = document["invocations"]["record"]["argv"]
    report_argv = document["invocations"]["report"]["argv"]
    expected_data_path = record_argv[6] if len(record_argv) == 10 else ""
    fixed_record_prefix = [
        profiler["path"],
        "record",
        "--event=cycles:u",
        "--call-graph=dwarf",
        "--freq=999",
        "--output",
    ]
    expected_staging_leaf = f".{generation_name}.staging"
    supervision = document["supervision"]
    subject = supervision["profile_subject"]
    report_input = supervision["report_input"]
    expected_subject_path = (
        f"/proc/{subject['producer_pid']}/fd/{subject['retained_descriptor']}"
    )
    expected_report_input_path = (
        f"/proc/{report_input['producer_pid']}/fd/"
        f"{report_input['retained_descriptor']}"
    )
    if (
        supervision["reference_subject"]
        != {
            "hierarchy": "profile_producer_exact_child.v1",
            "executable_transport": "inherited_retained_release_inode.v1",
            "termination": "waitid_wnowait_then_wait4_exact_child.v1",
        }
        or supervision["profile_wrapper"]
        != {
            "hierarchy": "profile_producer_exact_child.v1",
            "executable_transport": "inherited_retained_perf_inode.v1",
            "termination": "waitid_wnowait_then_wait4_exact_child.v1",
        }
        or subject["hierarchy"]
        != "perf_spawned_descendant_without_independent_pid_authority.v1"
        or subject["pid_authority"] != "unavailable_not_claimed"
        or subject["executable_transport"]
        != "proc_parent_retained_release_inode.v1"
        or subject["descendant_cleanup"]
        != "perf_session_process_group_terminated_before_wrapper_wait4.v1"
        or subject["termination"]
        != "perf_wrapper_zero_exit_plus_valid_byte_identical_result_and_perf_data.v1"
        or subject["argv_path"] != expected_subject_path
        or report_input["transport"]
        != "proc_parent_retained_perf_data_inode.v1"
        or report_input["producer_pid"] != subject["producer_pid"]
        or report_input["argv_path"] != expected_report_input_path
        or len(record_argv) != 10
        or record_argv[:6] != fixed_record_prefix
        or expected_data_path != os.path.abspath(expected_data_path)
        or Path(expected_data_path).name != _PERF_DATA_NAME
        or Path(expected_data_path).parent.name != expected_staging_leaf
        or record_argv[7:]
        != ["--", expected_subject_path, manifest["path"]]
        or report_argv
        != [
            profiler["path"],
            "report",
            "--stdio",
            "--no-children",
            "--sort=dso,symbol",
            "--input",
            expected_report_input_path,
        ]
    ):
        _fail(
            "PROFILE_RECORD_INVALID",
            "perf process hierarchy, transport, or argv is not the exact audited protocol",
        )
    expected_version_hash = hashlib.sha256(
        (profiler["version"] + "\n").encode("utf-8")
    ).hexdigest()
    if profiler["version_stdout_sha256"] != expected_version_hash:
        _fail("PROFILE_RECORD_INVALID", "perf version output hash does not match")
    for invocation in document["invocations"].values():
        if invocation["environment_sha256"] != document["environment_sha256"]:
            _fail("PROFILE_RECORD_INVALID", "invocation environment hash disagrees")
        if invocation["working_directory"] != working_directory:
            _fail("PROFILE_RECORD_INVALID", "invocation working directories disagree")
        if invocation["working_directory"] != os.path.abspath(invocation["working_directory"]):
            _fail("PROFILE_RECORD_INVALID", "invocation working directory is not canonical")

    phases = document["phase_diagnostics"]
    if phases["role"] != "variable_host_diagnostics_outside_sim_result_v2":
        _fail("PROFILE_RECORD_INVALID", "phase diagnostics role is invalid")
    ordered_phases = [
        phases["config"],
        phases["version"],
        phases["reference"],
        phases["record"],
        phases["report"],
    ]
    for index, phase in enumerate(ordered_phases):
        if phase["monotonic_stop_ns"] - phase["monotonic_start_ns"] != phase["wall_time_ns"]:
            _fail("PROFILE_RECORD_INVALID", "phase wall time is not stop minus start")
        if phase["timed_out"] or phase["return_code"] != 0:
            _fail("PROFILE_RECORD_INVALID", "measured profile contains a failed phase")
        if index and phase["monotonic_start_ns"] < ordered_phases[index - 1]["monotonic_stop_ns"]:
            _fail("PROFILE_RECORD_INVALID", "profile phases overlap or are out of order")

    artifacts = document["artifacts"]
    if [item["path"] for item in artifacts] != sorted(_ARTIFACT_NAMES):
        _fail("PROFILE_RECORD_INVALID", "profile artifact closure is not exact")
    publication = document["publication"]
    if publication["closed_paths"] != sorted((*_ARTIFACT_NAMES, _RECORD_NAME)):
        _fail("PROFILE_RECORD_INVALID", "publication path closure is not exact")
    result_artifact = next(item for item in artifacts if item["path"] == _RESULT_NAME)
    if result_artifact["sha256"] != binding["file_sha256"]:
        _fail("PROFILE_RECORD_INVALID", "result artifact hash disagrees")
    return document


def _load_profile_record(
    path: str | Path,
    *,
    repo_root: Path,
    expected_source_authority_content_manifest_sha256: str,
    source_authority_verifier: SourceAuthorityVerifier = _verify_source_authority,
) -> dict[str, Any]:
    generation_path = Path(os.path.abspath(os.fspath(path)))
    try:
        generation = inspect_immutable_directory_generation(
            generation_path,
            max_files=len(_ARTIFACT_NAMES) + 1,
            max_directories=1,
            max_total_file_bytes=MAX_GENERATION_BYTES,
            max_total_relative_path_bytes=4096,
            max_file_bytes_by_path={
                _PERF_DATA_NAME: MAX_PERF_DATA_BYTES,
                _REPORT_NAME: MAX_TEXT_ARTIFACT_BYTES,
                _RECORD_STDERR_NAME: MAX_TEXT_ARTIFACT_BYTES,
                _REPORT_STDERR_NAME: MAX_TEXT_ARTIFACT_BYTES,
                _RESULT_NAME: MAX_RESULT_BYTES,
                _RECORD_NAME: MAX_PROFILE_RECORD_BYTES,
            },
        )
    except ImmutableArtifactLimitError as exc:
        raise ProfileRecorderError("PROFILE_OVERFLOW", "profile generation exceeds its bound") from exc
    except ImmutableArtifactError as exc:
        raise ProfileRecorderError("PROFILE_RECORD_INVALID", "profile generation is not immutable") from exc
    observed = {item.relative_path: item for item in generation.files}
    if set(observed) != set((*_ARTIFACT_NAMES, _RECORD_NAME)):
        _fail("PROFILE_RECORD_INVALID", "profile generation file closure is invalid")
    try:
        with open_retained_regular(
            generation_path / _RECORD_NAME,
            max_bytes=MAX_PROFILE_RECORD_BYTES,
        ) as retained_record:
            payload = retained_record.read_bytes(max_bytes=MAX_PROFILE_RECORD_BYTES)
    except ImmutableArtifactError as exc:
        raise ProfileRecorderError("PROFILE_RECORD_INVALID", "profile record is unreadable") from exc
    try:
        value = loads_json_strict(payload.decode("utf-8", errors="strict"), source=str(generation_path / _RECORD_NAME))
    except (UnicodeDecodeError, ContractValidationError) as exc:
        raise ProfileRecorderError("PROFILE_RECORD_INVALID", "profile record is not strict JSON") from exc
    document = validate_profile_record(value)
    if payload != _canonical_json_line(document):
        _fail("PROFILE_RECORD_INVALID", "profile record is not canonical JSON")
    if document["publication"]["generation_name"] != generation_path.name:
        _fail("PROFILE_RECORD_INVALID", "profile generation name disagrees")
    for identity in document["artifacts"]:
        observed_identity = observed[identity["path"]]
        if (
            observed_identity.sha256 != identity["sha256"]
            or observed_identity.size_bytes != identity["size_bytes"]
        ):
            _fail("PROFILE_RECORD_INVALID", "profile artifact identity disagrees")

    try:
        with open_retained_regular(
            generation_path / _PERF_DATA_NAME,
            max_bytes=MAX_PERF_DATA_BYTES,
        ) as perf_data:
            if perf_data.size_bytes < 8:
                _fail("PROFILE_RECORD_INVALID", "raw perf.data is truncated")
            magic = os.pread(perf_data.descriptor, 8, 0)
            perf_data.verify_unchanged()
            if magic != b"PERFILE2":
                _fail("PROFILE_RECORD_INVALID", "raw perf.data magic is invalid")
        with open_retained_regular(
            generation_path / _REPORT_NAME,
            max_bytes=MAX_TEXT_ARTIFACT_BYTES,
        ) as report_file:
            report_payload = report_file.read_bytes(max_bytes=MAX_TEXT_ARTIFACT_BYTES)
        report_text = _canonical_text(
            report_payload,
            role="retained perf text report",
            allow_empty=False,
        )
        _require_symbolized_report(report_text, Path(document["inputs"]["binary"]["path"]).name)
        for diagnostic_name in (_RECORD_STDERR_NAME, _REPORT_STDERR_NAME):
            with open_retained_regular(
                generation_path / diagnostic_name,
                max_bytes=MAX_TEXT_ARTIFACT_BYTES,
            ) as diagnostic_file:
                diagnostic_payload = diagnostic_file.read_bytes(
                    max_bytes=MAX_TEXT_ARTIFACT_BYTES
                )
            _canonical_text(
                diagnostic_payload,
                role=f"retained {diagnostic_name}",
                allow_empty=True,
            )
    except ProfileRecorderError:
        raise
    except ImmutableArtifactError as exc:
        raise ProfileRecorderError(
            "PROFILE_RECORD_INVALID", "profile artifacts are unreadable"
        ) from exc

    repo_root = Path(os.path.abspath(os.fspath(repo_root)))
    source_identity = document["source_authority"]
    if (
        _SHA256_RE.fullmatch(
            expected_source_authority_content_manifest_sha256
        )
        is None
        or source_identity["content_manifest_sha256"]
        != expected_source_authority_content_manifest_sha256
    ):
        _fail(
            "PROFILE_SOURCE_AUTHORITY_REJECTED",
            "profile record disagrees with the external source-authority anchor",
        )
    try:
        with open_retained_regular(
            repo_root / _PRODUCER_RELATIVE,
            max_bytes=MAX_PROFILE_RECORD_BYTES,
        ) as producer, open_retained_regular(
            repo_root / _PRODUCER_SCHEMA_RELATIVE,
            max_bytes=MAX_PROFILE_RECORD_BYTES,
        ) as producer_schema:
            current_source = _source_authority_binding(
                source_authority_verifier(
                    Path(source_identity["generation_path"]),
                    source_identity["content_manifest_sha256"],
                    repo_root,
                ),
                producer=producer,
                producer_schema=producer_schema,
            )
    except ProfileRecorderError:
        raise
    except ImmutableArtifactError as exc:
        raise ProfileRecorderError(
            "PROFILE_INPUT_DRIFT", "profile producer or schema identity changed"
        ) from exc
    if current_source != source_identity:
        _fail(
            "PROFILE_INPUT_DRIFT",
            "current source authority differs from the profile record",
        )
    if _host_identity(_capture_affinity()) != document["host_identity"]:
        _fail("PROFILE_INPUT_DRIFT", "kernel, CPU, or affinity identity changed")
    binary_identity = document["inputs"]["binary"]
    if binary_identity["path"] != str(repo_root / RELEASE_BINARY_RELATIVE):
        _fail("PROFILE_RECORD_INVALID", "profile binary is not this checkout's release sim-run")
    try:
        with open_retained_executable(Path(binary_identity["path"]), allow_hardlinks=True) as binary:
            if _file_identity(binary) != binary_identity:
                _fail("PROFILE_INPUT_DRIFT", "release sim-run differs from the profile record")
        profiler_identity = document["profiler_identity"]
        with open_retained_executable(Path(profiler_identity["path"])) as profiler:
            current_profiler = _file_identity(profiler)
            if any(current_profiler[key] != profiler_identity[key] for key in ("path", "size_bytes", "sha256")):
                _fail("PROFILE_INPUT_DRIFT", "perf executable differs from the profile record")
        manifest_identity = document["inputs"]["manifest"]
        with open_retained_regular(Path(manifest_identity["path"]), max_bytes=MAX_MANIFEST_BYTES) as manifest:
            manifest_document, _payload, canonical_digest = _manifest_document(manifest)
            current_manifest = _file_identity(manifest)
            if (
                any(current_manifest[key] != manifest_identity[key] for key in ("path", "size_bytes", "sha256"))
                or canonical_digest != manifest_identity["canonical_sha256"]
                or manifest_document["scope"] != manifest_identity["scope"]
            ):
                _fail("PROFILE_INPUT_DRIFT", "manifest differs from the profile record")
    except ProfileRecorderError:
        raise
    except ImmutableArtifactError as exc:
        raise ProfileRecorderError("PROFILE_INPUT_DRIFT", "retained profile input is unavailable") from exc

    result_identity = next(item for item in document["artifacts"] if item["path"] == _RESULT_NAME)
    try:
        with open_retained_regular(generation_path / _RESULT_NAME, max_bytes=MAX_RESULT_BYTES) as result_file:
            result_payload = result_file.read_bytes(max_bytes=MAX_RESULT_BYTES)
    except ImmutableArtifactError as exc:
        raise ProfileRecorderError("PROFILE_RECORD_INVALID", "simulator result artifact is unreadable") from exc
    if hashlib.sha256(result_payload).hexdigest() != result_identity["sha256"]:
        _fail("PROFILE_RECORD_INVALID", "simulator result artifact hash changed")
    result = _decode_result(result_payload, source=str(generation_path / _RESULT_NAME))
    _validate_result_identity(
        result,
        binary_sha256=binary_identity["sha256"],
        manifest=manifest_document,
        manifest_canonical_sha256=document["inputs"]["manifest"]["canonical_sha256"],
        source_git_commit=source_identity["source_git_commit"],
    )
    trace = result["trace_identity"]
    expected_binding = {
        "file_sha256": result_identity["sha256"],
        "result_sha256": result["result_sha256"],
        "build_identity": result["build_identity"],
        "trace_identity": {
            "bundle_id": trace["bundle_id"],
            "bundle_path": trace["bundle_path"],
            "bundle_sha256": trace["bundle_sha256"],
            "wire_version": trace["wire_version"],
            "total_size_bytes": trace["total_size_bytes"],
            "record_count": trace["record_count"],
            "instruction_count": trace["instruction_count"],
            "identity_sha256": _sha256_canonical(trace),
        },
        "arch_card_identity": result["arch_card_identity"],
        "configuration_identity": result["configuration_identity"],
        "resolved_configuration_sha256": _sha256_canonical(
            result["resolved_configuration"]
        ),
    }
    if document["sim_result_binding"] != expected_binding:
        _fail("PROFILE_RECORD_INVALID", "simulator result projection disagrees")
    return document


def load_profile_record(
    path: str | Path,
    source_authority_content_manifest_sha256: str,
) -> dict[str, Any]:
    """Load a profile against an independently supplied source anchor."""

    return _load_profile_record(
        path,
        repo_root=ROOT,
        expected_source_authority_content_manifest_sha256=(
            source_authority_content_manifest_sha256
        ),
    )


def _error_line(error: ProfileRecorderError) -> str:
    return json.dumps(
        {"code": error.code, "detail": error.detail},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 5 or arguments[0] != "record":
        sys.stderr.write(
            _error_line(
                ProfileRecorderError(
                    "CLI_USAGE_FAILED",
                    "usage: python -m eval.profile record MANIFEST_V2 OUTPUT_GENERATION SOURCE_GATE_GENERATION SOURCE_CONTENT_MANIFEST_SHA256",
                )
            )
        )
        return 2
    try:
        document = record_profile(
            arguments[1], arguments[2], arguments[3], arguments[4]
        )
    except ProfileRecorderError as exc:
        sys.stderr.write(_error_line(exc))
        return 1
    sys.stdout.write(
        json.dumps(
            {
                "generation_path": str(Path(arguments[2]).absolute()),
                "record_sha256": document["record_sha256"],
                "status": document["outcome"]["status"],
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
