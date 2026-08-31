#!/usr/bin/env python3
"""Capture static, direct, NVBit, and simulator evidence for the multiline assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ensure_directory,
    read_regular,
    replace_regular_bytes,
    sha256_file,
)

DEFAULT_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_multiline_response_20260831_v1"
)
BINARY = HERE / "async_multiline_response"
SOURCE = HERE / "async_multiline_response.cu"
MAKEFILE = HERE / "Makefile"
CONTROL_TOOL = REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"
CONTROL_TOOL_ROOT = CONTROL_TOOL.parent
NVBIT_CAPTURE = (
    REPO_ROOT
    / "workloads/research_probes/ldgsts_burst_release/capture_nvbit.py"
)
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
CUDA_BIN = Path(os.environ.get("CUDA_BIN", "/usr/local/cuda-13.2/bin"))
NVCC = CUDA_BIN / "nvcc"
CUOBJDUMP = CUDA_BIN / "cuobjdump"

DEPTHS = (1, 2)
LEADS = (0, 4, 8, 16)
VARIANTS = ("span_expensive", "span_cheap")
REPETITIONS = tuple(range(1, 11))
TRACE_GROUPS = 64
DIRECT_GROUPS = 8192
# Optional immutable generation metadata supplied by a thin campaign wrapper.
# `None` preserves every predecessor static-binding byte and default.
STATIC_CAMPAIGN_CONTRACT: Mapping[str, Any] | None = None


class CaptureError(RuntimeError):
    """A required command or retained-artifact check failed."""


@dataclass(frozen=True)
class Cell:
    depth: int
    lead: int

    @property
    def direct_name(self) -> str:
        return (
            f"async_multiline_response_d{self.depth}_l{self.lead}_"
            f"g{DIRECT_GROUPS}"
        )

    def endpoint_name(self, variant: str) -> str:
        return (
            f"async_multiline_response_{variant}_d{self.depth}_"
            f"l{self.lead}_g{TRACE_GROUPS}"
        )


def sha256(path: Path) -> str:
    if _coordinate_artifact_path(path):
        try:
            return sha256_file(_lexical_absolute(path))[0]
        except ImmutableArtifactError as error:
            raise CaptureError(f"cannot hash coordinate artifact {path}: {error}") from error
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def strict_json_equal(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            strict_json_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(observed) == len(expected) and all(
            strict_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def atomic_write_text(path: Path, value: str) -> None:
    if _coordinate_artifact_path(path):
        try:
            checked = _lexical_absolute(path)
            ensure_directory(checked.parent)
            replace_regular_bytes(checked, value.encode("utf-8"))
        except ImmutableArtifactError as error:
            raise CaptureError(
                f"cannot publish coordinate artifact {path}: {error}"
            ) from error
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json_text(value))


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = read_bytes(path)
        if _coordinate_artifact_path(path):
            # Imported lazily so the predecessor multiline/v1 path keeps its
            # original dependency surface and decoder behavior.
            from line_ownership_campaign_contract import (
                ContractError as CampaignContractError,
                strict_json_loads,
            )

            try:
                value = strict_json_loads(payload, context=path)
            except CampaignContractError as error:
                raise CaptureError(f"cannot read {path}: {error}") from error
        else:
            value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise CaptureError(f"{path}: expected one JSON object")
    return value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _coordinate_artifact_path(path: Path) -> bool:
    contract = STATIC_CAMPAIGN_CONTRACT
    if contract is None:
        return False
    candidate = _lexical_absolute(path)
    root = _lexical_absolute(Path(str(contract["evidence_root"])))
    aggregate = _lexical_absolute(Path(str(contract["aggregate_report"])))
    return candidate == aggregate or candidate == root or root in candidate.parents


def read_bytes(path: Path) -> bytes:
    if _coordinate_artifact_path(path):
        try:
            return read_regular(_lexical_absolute(path))
        except ImmutableArtifactError as error:
            raise CaptureError(f"cannot read coordinate artifact {path}: {error}") from error
    return path.read_bytes()


def read_text(path: Path) -> str:
    try:
        return read_bytes(path).decode("utf-8")
    except UnicodeDecodeError as error:
        raise CaptureError(f"artifact is not UTF-8: {path}") from error


def ensure_output_directory(path: Path) -> None:
    if _coordinate_artifact_path(path):
        try:
            ensure_directory(_lexical_absolute(path))
        except ImmutableArtifactError as error:
            raise CaptureError(f"cannot create coordinate directory {path}: {error}") from error
    else:
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CommandExecution:
    stdout: str
    stderr: str
    return_code: int | None
    timed_out: bool
    started_at: str
    finished_at: str
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_identity(path: Path) -> dict[str, Any]:
    if _coordinate_artifact_path(path):
        checked = _lexical_absolute(path)
        payload = read_bytes(checked)
        return {
            "path": str(checked),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    resolved = path.resolve()
    if not resolved.is_file():
        raise CaptureError(f"identity input is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def identities(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: file_identity(path) for name, path in sorted(paths.items())}


def control_tool_identities() -> dict[str, dict[str, Any]]:
    return identities(
        {
            f"sass_control/{path.name}": path
            for path in sorted(CONTROL_TOOL_ROOT.glob("*.py"))
            if path.is_file()
        }
    )


def command_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def execute(
    command: Sequence[str],
    *,
    timeout: int,
    environment: Mapping[str, str] | None = None,
) -> CommandExecution:
    started_at = utc_now()
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment) if environment is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return CommandExecution(
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
            timed_out=False,
            started_at=started_at,
            finished_at=utc_now(),
        )
    except subprocess.TimeoutExpired as error:
        return CommandExecution(
            stdout=command_output_text(error.stdout),
            stderr=command_output_text(error.stderr),
            return_code=None,
            timed_out=True,
            started_at=started_at,
            finished_at=utc_now(),
            error=str(error),
        )
    except OSError as error:
        return CommandExecution(
            stdout="",
            stderr=str(error) + "\n",
            return_code=None,
            timed_out=False,
            started_at=started_at,
            finished_at=utc_now(),
            error=str(error),
        )


def resolve_executable(
    command: Sequence[str], environment_value: Mapping[str, str] | None
) -> Path:
    token = command[0]
    if "/" in token:
        result = Path(token)
    else:
        found = shutil.which(
            token,
            path=None if environment_value is None else environment_value.get("PATH"),
        )
        if found is None:
            raise CaptureError(f"command executable is unavailable: {token}")
        result = Path(found)
    if not result.is_file():
        raise CaptureError(f"command executable is unavailable: {result}")
    return result


def environment_overrides() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def recorded_run(
    command: Sequence[str],
    *,
    timeout: int,
    manifest_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    environment_value: Mapping[str, str] | None = None,
    combined_output_path: Path | None = None,
) -> CommandExecution:
    execution = execute(command, timeout=timeout, environment=environment_value)
    atomic_write_text(stdout_path, execution.stdout)
    atomic_write_text(stderr_path, execution.stderr)
    if combined_output_path is not None:
        atomic_write_text(
            combined_output_path, execution.stdout + execution.stderr
        )
    outputs = {
        name: file_identity(path)
        for name, path in sorted(output_paths.items())
        if path.is_file()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "command": list(command),
        "configured_timeout_seconds": timeout,
        "started_at": execution.started_at,
        "finished_at": execution.finished_at,
        "return_code": execution.return_code,
        "timed_out": execution.timed_out,
        "executable": file_identity(
            resolve_executable(command, environment_value)
        ),
        "inputs": identities(input_paths),
        "outputs": outputs,
        "stdout": file_identity(stdout_path),
        "stderr": file_identity(stderr_path),
        "environment_overrides": environment_overrides(),
    }
    if execution.error is not None:
        manifest["error"] = execution.error
    atomic_write_json(manifest_path, manifest)
    if execution.timed_out:
        raise CaptureError(
            f"command timed out after {timeout}s: {list(command)}; "
            f"manifest={manifest_path}"
        )
    if execution.return_code != 0:
        raise CaptureError(
            f"command exited {execution.return_code}: {list(command)}\n"
            f"{execution.stderr[-4000:]}\nmanifest={manifest_path}"
        )
    return execution


def run_unrecorded(
    command: Sequence[str],
    *,
    timeout: int,
    environment_value: Mapping[str, str] | None = None,
) -> CommandExecution:
    execution = execute(command, timeout=timeout, environment=environment_value)
    if execution.timed_out:
        raise CaptureError(f"command timed out after {timeout}s: {list(command)}")
    if execution.return_code != 0:
        raise CaptureError(
            f"command exited {execution.return_code}: {list(command)}\n"
            f"{execution.stderr[-4000:]}"
        )
    return execution


def validate_iso_interval(started_at: Any, finished_at: Any, context: Path) -> None:
    if not isinstance(started_at, str) or not isinstance(finished_at, str):
        raise CaptureError(f"{context}: missing ISO start/finish timestamps")
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
    except ValueError as error:
        raise CaptureError(f"{context}: invalid ISO timestamp: {error}") from error
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise CaptureError(f"{context}: invalid command timestamp interval")


def validate_run_manifest(
    manifest_path: Path,
    *,
    command: Sequence[str],
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    environment_value: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    document = read_json(manifest_path)
    validate_iso_interval(
        document.get("started_at"), document.get("finished_at"), manifest_path
    )
    expected = {
        "schema_version": 1,
        "command": list(command),
        "configured_timeout_seconds": timeout,
        "return_code": 0,
        "timed_out": False,
        "executable": file_identity(
            resolve_executable(command, environment_value)
        ),
        "inputs": identities(input_paths),
        "outputs": identities(output_paths),
        "stdout": file_identity(stdout_path),
        "stderr": file_identity(stderr_path),
        "environment_overrides": environment_overrides(),
    }
    mismatches = {
        field: {"observed": document.get(field), "expected": value}
        for field, value in expected.items()
        if not strict_json_equal(document.get(field), value)
    }
    if mismatches or document.get("error") is not None:
        raise CaptureError(
            f"{manifest_path}: stale or invalid run manifest: {mismatches}"
        )
    return document


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update(environment_overrides())
    return value


def require_binary(expected_sha256: str | None = None) -> dict[str, Any]:
    if not BINARY.is_file() or not os.access(BINARY, os.X_OK):
        raise CaptureError(f"probe binary is missing or not executable: {BINARY}")
    observed = sha256(BINARY)
    if expected_sha256 is not None and observed != expected_sha256:
        raise CaptureError(
            f"probe binary identity mismatch: {observed} != {expected_sha256}"
        )
    return {
        "path": str(BINARY.resolve()),
        "sha256": observed,
        "size_bytes": BINARY.stat().st_size,
    }


def tool_version(path: Path, *arguments: str) -> dict[str, Any]:
    command = [str(path), *arguments]
    execution = run_unrecorded(
        command, timeout=30, environment_value=environment()
    )
    return {
        "command": command,
        "stdout": execution.stdout,
        "stderr": execution.stderr,
    }


def current_static_binding() -> dict[str, Any]:
    for required in (SOURCE, MAKEFILE, CONTROL_TOOL, NVCC, CUOBJDUMP):
        if not required.is_file():
            raise CaptureError(f"static binding input is missing: {required}")
    make_path = resolve_executable(["make"], environment())
    python_path = Path(sys.executable)
    binding = {
        "schema_version": 2,
        "source": file_identity(SOURCE),
        "makefile": file_identity(MAKEFILE),
        "control_tools": control_tool_identities(),
        "toolchain": {
            "make": file_identity(make_path),
            "python": {
                **file_identity(python_path),
                "version": sys.version,
            },
            "nvcc": {
                **file_identity(NVCC),
                "version": tool_version(NVCC, "--version"),
            },
            "cuobjdump": {
                **file_identity(CUOBJDUMP),
                "version": tool_version(CUOBJDUMP, "--version"),
            },
        },
        "environment_overrides": environment_overrides(),
    }
    if STATIC_CAMPAIGN_CONTRACT is not None:
        binding["campaign_contract"] = dict(STATIC_CAMPAIGN_CONTRACT)
    return binding


def static_campaign_inputs(root: Path) -> dict[str, Path]:
    if STATIC_CAMPAIGN_CONTRACT is None:
        return {}
    record = root.resolve() / "campaign_contract.json"
    if not record.is_file() or read_json(record) != dict(STATIC_CAMPAIGN_CONTRACT):
        raise CaptureError(f"{record}: static campaign contract mismatch")
    return {"campaign_contract": record}


def static_build_command() -> list[str]:
    command = [
        "make",
        "-B",
        "-C",
        str(HERE),
    ]
    if MAKEFILE != HERE / "Makefile":
        command.extend(("-f", str(MAKEFILE)))
    command.extend((f"NVCC={NVCC}", BINARY.name))
    return command


def static_control_command(static: Path) -> list[str]:
    return [
        sys.executable,
        str(CONTROL_TOOL),
        "--binary",
        str(BINARY),
        "--arch",
        "sm_86",
        "--toolkit-bindir",
        str(CUDA_BIN),
        "--out",
        str(static / "control_bits.json"),
    ]


def complete_stage_or_absent(paths: Sequence[Path], context: str) -> bool:
    # Empty stderr/stdout is a valid successful command result; completeness
    # comes from the manifest's exact hashes, not a nonzero-size heuristic.
    present = [path.is_file() for path in paths]
    if all(present):
        return True
    if any(path.exists() for path in paths):
        raise CaptureError(
            f"{context}: incomplete retained stage must be moved aside before recapture"
        )
    return False


def static_command_document(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    static = root / "static"
    binary = require_binary()
    artifacts = identities(
        {
            "build_log": static / "build.log",
            "control_bits": static / "control_bits.json",
            "control_log": static / "control.log",
            "resources": static / "resources.txt",
            "sass": static / "sass.txt",
        }
    )
    runs = identities(
        {
            "build": static / "build.run.json",
            "control": static / "control.run.json",
            "resources": static / "resources.run.json",
            "sass": static / "sass.run.json",
        }
    )
    return {
        "schema_version": 2,
        "binding": dict(binding),
        "binary": binary,
        "commands": {
            "build": static_build_command(),
            "sass": [str(CUOBJDUMP), "--dump-sass", str(BINARY)],
            "resources": [
                str(CUOBJDUMP),
                "--dump-resource-usage",
                str(BINARY),
            ],
            "control": static_control_command(static),
        },
        "artifacts": artifacts,
        "run_manifests": runs,
    }


def validate_static(root: Path) -> dict[str, Any]:
    static = root / "static"
    binding_path = static / "binding.json"
    command_path = static / "command.json"
    if not binding_path.is_file() or not command_path.is_file():
        raise CaptureError(
            f"{static}: retained static lacks schema-2 binding/command metadata; "
            "move the legacy or incomplete directory aside and recapture"
        )
    binding = current_static_binding()
    if not strict_json_equal(read_json(binding_path), binding):
        raise CaptureError(
            f"{binding_path}: retained static is stale relative to source, Makefile, "
            "control tooling, or CUDA toolchain"
        )
    binary = require_binary()
    common_inputs = {
        **static_campaign_inputs(root),
        "source": SOURCE,
        "makefile": MAKEFILE,
        "nvcc": NVCC,
    }
    build_manifest = static / "build.run.json"
    build_timeout = read_json(build_manifest).get("configured_timeout_seconds")
    if not isinstance(build_timeout, int) or build_timeout < 1:
        raise CaptureError(f"{build_manifest}: invalid configured timeout")
    validate_run_manifest(
        build_manifest,
        command=static_build_command(),
        timeout=build_timeout,
        stdout_path=static / "build.stdout.txt",
        stderr_path=static / "build.stderr.txt",
        input_paths=common_inputs,
        output_paths={"binary": BINARY, "build_log": static / "build.log"},
        environment_value=environment(),
    )
    for label, option in (
        ("sass", "--dump-sass"),
        ("resources", "--dump-resource-usage"),
    ):
        manifest = static / f"{label}.run.json"
        configured = read_json(manifest).get("configured_timeout_seconds")
        if not isinstance(configured, int) or configured < 1:
            raise CaptureError(f"{manifest}: invalid configured timeout")
        validate_run_manifest(
            manifest,
            command=[str(CUOBJDUMP), option, str(BINARY)],
            timeout=configured,
            stdout_path=static / f"{label}.txt",
            stderr_path=static / f"{label}.stderr.txt",
            input_paths={
                **static_campaign_inputs(root),
                "binary": BINARY,
                "cuobjdump": CUOBJDUMP,
            },
            output_paths={label: static / f"{label}.txt"},
            environment_value=environment(),
        )
    control_manifest = static / "control.run.json"
    control_timeout = read_json(control_manifest).get("configured_timeout_seconds")
    if not isinstance(control_timeout, int) or control_timeout < 1:
        raise CaptureError(f"{control_manifest}: invalid configured timeout")
    control_inputs = {
        **static_campaign_inputs(root),
        "binary": BINARY,
        "python": Path(sys.executable),
    }
    control_inputs.update(
        {
            name: Path(identity["path"])
            for name, identity in binding["control_tools"].items()
        }
    )
    validate_run_manifest(
        control_manifest,
        command=static_control_command(static),
        timeout=control_timeout,
        stdout_path=static / "control.stdout.txt",
        stderr_path=static / "control.stderr.txt",
        input_paths=control_inputs,
        output_paths={
            "control_bits": static / "control_bits.json",
            "control_log": static / "control.log",
        },
        environment_value=environment(),
    )
    expected = static_command_document(root, binding)
    observed = read_json(command_path)
    if not strict_json_equal(observed, expected):
        raise CaptureError(
            f"{command_path}: stale static command/artifact binding"
        )
    if expected["binary"] != binary:
        raise CaptureError(f"{static}: retained static binary identity drifted")
    return binary


def retained_binary(root: Path) -> dict[str, Any]:
    return validate_static(root)


def selected_cells(smoke: bool, selector: str | None) -> list[Cell]:
    if smoke:
        if selector is not None:
            raise CaptureError("--smoke and --cells are mutually exclusive")
        return [Cell(1, 16)]
    configured = [Cell(depth, lead) for depth in DEPTHS for lead in LEADS]
    if selector is None:
        return configured
    by_name = {f"d{cell.depth}l{cell.lead}": cell for cell in configured}
    values: list[Cell] = []
    for raw in selector.split(","):
        key = raw.strip()
        if key not in by_name:
            raise CaptureError(f"unknown cell {key!r}; expected e.g. d1l16")
        if by_name[key] not in values:
            values.append(by_name[key])
    if not values:
        raise CaptureError("--cells selected no cells")
    return values


def selected_repetitions(smoke: bool, selector: str | None) -> list[int]:
    if smoke:
        if selector is not None:
            raise CaptureError("--smoke and --repeats are mutually exclusive")
        return [1]
    if selector is None:
        return list(REPETITIONS)
    values: list[int] = []
    for raw in selector.split(","):
        try:
            repetition = int(raw.strip())
        except ValueError as error:
            raise CaptureError(f"invalid repetition {raw!r}") from error
        if repetition not in REPETITIONS:
            raise CaptureError("repetitions must be in 1..10")
        if repetition not in values:
            values.append(repetition)
    if not values:
        raise CaptureError("--repeats selected no repetitions")
    return values


def capture_static(root: Path, timeout: int) -> None:
    static = root / "static"
    command_path = static / "command.json"
    if command_path.is_file():
        validate_static(root)
        print(f"static existing: {static}", flush=True)
        return
    ensure_output_directory(static)
    binding = current_static_binding()
    binding_path = static / "binding.json"
    if binding_path.is_file():
        if not strict_json_equal(read_json(binding_path), binding):
            raise CaptureError(
                f"{binding_path}: stale retained static must be moved aside before recapture"
            )
    elif any(static.iterdir()):
        raise CaptureError(
            f"{static}: unbound legacy/incomplete static must be moved aside before recapture"
        )
    else:
        atomic_write_json(binding_path, binding)

    build_files = (
        static / "build.run.json",
        static / "build.stdout.txt",
        static / "build.stderr.txt",
        static / "build.log",
    )
    if complete_stage_or_absent(build_files, "static build"):
        manifest = read_json(static / "build.run.json")
        configured = manifest.get("configured_timeout_seconds")
        if not isinstance(configured, int) or configured < 1:
            raise CaptureError(f"{static / 'build.run.json'}: invalid configured timeout")
        validate_run_manifest(
            static / "build.run.json",
            command=static_build_command(),
            timeout=configured,
            stdout_path=static / "build.stdout.txt",
            stderr_path=static / "build.stderr.txt",
            input_paths={
                **static_campaign_inputs(root),
                "source": SOURCE,
                "makefile": MAKEFILE,
                "nvcc": NVCC,
            },
            output_paths={"binary": BINARY, "build_log": static / "build.log"},
            environment_value=environment(),
        )
    else:
        recorded_run(
            static_build_command(),
            timeout=timeout,
            manifest_path=static / "build.run.json",
            stdout_path=static / "build.stdout.txt",
            stderr_path=static / "build.stderr.txt",
            combined_output_path=static / "build.log",
            input_paths={
                **static_campaign_inputs(root),
                "source": SOURCE,
                "makefile": MAKEFILE,
                "nvcc": NVCC,
            },
            output_paths={"binary": BINARY, "build_log": static / "build.log"},
            environment_value=environment(),
        )
    binary = require_binary()

    for option, output in (
        ("--dump-sass", "sass.txt"),
        ("--dump-resource-usage", "resources.txt"),
    ):
        label = output.removesuffix(".txt")
        command = [str(CUOBJDUMP), option, str(BINARY)]
        stage_files = (
            static / f"{label}.run.json",
            static / output,
            static / f"{label}.stderr.txt",
        )
        if complete_stage_or_absent(stage_files, f"static {label}"):
            configured = read_json(static / f"{label}.run.json").get(
                "configured_timeout_seconds"
            )
            if not isinstance(configured, int) or configured < 1:
                raise CaptureError(
                    f"{static / f'{label}.run.json'}: invalid configured timeout"
                )
            validate_run_manifest(
                static / f"{label}.run.json",
                command=command,
                timeout=configured,
                stdout_path=static / output,
                stderr_path=static / f"{label}.stderr.txt",
                input_paths={
                    **static_campaign_inputs(root),
                    "binary": BINARY,
                    "cuobjdump": CUOBJDUMP,
                },
                output_paths={label: static / output},
                environment_value=environment(),
            )
        else:
            recorded_run(
                command,
                timeout=timeout,
                manifest_path=static / f"{label}.run.json",
                stdout_path=static / output,
                stderr_path=static / f"{label}.stderr.txt",
                input_paths={
                    **static_campaign_inputs(root),
                    "binary": BINARY,
                    "cuobjdump": CUOBJDUMP,
                },
                output_paths={label: static / output},
                environment_value=environment(),
            )

    control_files = (
        static / "control.run.json",
        static / "control_bits.json",
        static / "control.stdout.txt",
        static / "control.stderr.txt",
        static / "control.log",
    )
    control_inputs = {
        **static_campaign_inputs(root),
        "binary": BINARY,
        "python": Path(sys.executable),
    }
    control_inputs.update(
        {
            name: Path(identity["path"])
            for name, identity in binding["control_tools"].items()
        }
    )
    control_command = static_control_command(static)
    if complete_stage_or_absent(control_files, "static control"):
        configured = read_json(static / "control.run.json").get(
            "configured_timeout_seconds"
        )
        if not isinstance(configured, int) or configured < 1:
            raise CaptureError(f"{static / 'control.run.json'}: invalid configured timeout")
        validate_run_manifest(
            static / "control.run.json",
            command=control_command,
            timeout=configured,
            stdout_path=static / "control.stdout.txt",
            stderr_path=static / "control.stderr.txt",
            input_paths=control_inputs,
            output_paths={
                "control_bits": static / "control_bits.json",
                "control_log": static / "control.log",
            },
            environment_value=environment(),
        )
    else:
        recorded_run(
            control_command,
            timeout=timeout,
            manifest_path=static / "control.run.json",
            stdout_path=static / "control.stdout.txt",
            stderr_path=static / "control.stderr.txt",
            combined_output_path=static / "control.log",
            input_paths=control_inputs,
            output_paths={
                "control_bits": static / "control_bits.json",
                "control_log": static / "control.log",
            },
            environment_value=environment(),
        )
    atomic_write_json(command_path, static_command_document(root, binding))
    if validate_static(root) != binary:
        raise CaptureError(f"{static}: static binary identity changed during capture")
    print(f"static: {static}", flush=True)


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def direct_command(cell: Cell, repetition: int) -> list[str]:
    return [
        str(BINARY),
        "--groups",
        str(DIRECT_GROUPS),
        "--depth",
        str(cell.depth),
        "--lead-samples",
        str(cell.lead),
        "--order",
        expected_order(repetition),
        "--markers",
        "off",
    ]


def validate_direct(
    destination: Path,
    cell: Cell,
    repetition: int,
    binary: Mapping[str, Any],
) -> None:
    path = destination / "result.json"
    payload = read_json(path)
    expected = {
        "workload": "async_multiline_response",
        "mode": "pair",
        "order": expected_order(repetition),
        "depth": cell.depth,
        "lead_clock_samples": cell.lead,
        "groups": DIRECT_GROUPS,
        "correct": True,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise CaptureError(f"{path}: direct binding mismatch: {mismatches}")
    for variant in VARIANTS:
        arm = payload.get(variant)
        if not isinstance(arm, dict) or arm.get("correct") is not True:
            raise CaptureError(f"{path}: invalid {variant} result")
    command_path = destination / "command.json"
    command = read_json(command_path)
    expected_command = direct_command(cell, repetition)
    configured = command.get("configured_timeout_seconds")
    if not isinstance(configured, int) or configured < 1:
        raise CaptureError(f"{command_path}: invalid configured timeout")
    run_manifest = destination / "direct.run.json"
    validate_run_manifest(
        run_manifest,
        command=expected_command,
        timeout=configured,
        stdout_path=path,
        stderr_path=destination / "stderr.txt",
        input_paths={"probe_binary": BINARY},
        output_paths={"result": path},
        environment_value=environment(),
    )
    expected_document = {
        "schema_version": 2,
        "command": expected_command,
        "configured_timeout_seconds": configured,
        "binary": dict(binary),
        "environment_overrides": environment_overrides(),
        "run_manifest": file_identity(run_manifest),
    }
    if command != expected_document:
        raise CaptureError(f"{destination}: direct command/binary binding mismatch")


def capture_direct(
    root: Path, cells: Sequence[Cell], repetitions: Sequence[int], timeout: int
) -> None:
    binary = retained_binary(root)
    for cell in cells:
        for repetition in repetitions:
            destination = (
                root
                / "direct"
                / cell.direct_name
                / f"repeat-{repetition}"
            )
            stage_files = (
                destination / "result.json",
                destination / "stderr.txt",
                destination / "direct.run.json",
                destination / "command.json",
            )
            if complete_stage_or_absent(stage_files, f"direct {destination}"):
                validate_direct(destination, cell, repetition, binary)
                print(f"direct existing: {cell.direct_name}/repeat-{repetition}")
                continue
            command = direct_command(cell, repetition)
            destination.mkdir(parents=True, exist_ok=True)
            result_path = destination / "result.json"
            recorded_run(
                command,
                timeout=timeout,
                manifest_path=destination / "direct.run.json",
                stdout_path=result_path,
                stderr_path=destination / "stderr.txt",
                input_paths={"probe_binary": BINARY},
                output_paths={"result": result_path},
                environment_value=environment(),
            )
            atomic_write_json(
                destination / "command.json",
                {
                    "schema_version": 2,
                    "command": command,
                    "configured_timeout_seconds": timeout,
                    "binary": binary,
                    "environment_overrides": environment_overrides(),
                    "run_manifest": file_identity(destination / "direct.run.json"),
                },
            )
            validate_direct(destination, cell, repetition, binary)
            print(f"direct: {cell.direct_name}/repeat-{repetition}", flush=True)


def nvbit_commands(
    destination: Path, cell: Cell, variant: str, timeout: int
) -> tuple[list[str], list[str]]:
    name = cell.endpoint_name(variant)
    capture_command = [
        sys.executable,
        str(NVBIT_CAPTURE),
        "--output",
        str(destination),
        "--cell",
        name,
        "--binary",
        str(BINARY),
        "--mode",
        "ncu",
        "--gpu",
        "0",
        "--timeout",
        str(timeout),
        "--target-args",
        "--groups",
        str(TRACE_GROUPS),
        "--depth",
        str(cell.depth),
        "--lead-samples",
        str(cell.lead),
        "--variant",
        variant,
        "--markers",
        "on",
    ]
    return capture_command, [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        "Delegate=yes",
        *capture_command,
    ]


def validate_nvbit(
    root: Path,
    destination: Path,
    cell: Cell,
    variant: str,
    binary: Mapping[str, Any],
) -> None:
    complete = read_json(destination / "complete.json")
    binding = read_json(destination / "campaign_binding.json")
    expected_name = cell.endpoint_name(variant)
    if complete.get("complete") is not True or complete.get("key") != expected_name:
        raise CaptureError(f"{destination}: invalid NVBit completion record")
    if complete.get("binary_sha256") != binary["sha256"]:
        raise CaptureError(f"{destination}: wrong NVBit binary")
    if binding.get("cell") != expected_name:
        raise CaptureError(f"{destination}: wrong NVBit binding")
    if binding.get("output_contract", {}).get("passed") is not True:
        raise CaptureError(f"{destination}: failed NVBit output contract")
    run_root = root / "runs/nvbit" / expected_name
    command_path = run_root / "command.json"
    command_document = read_json(command_path)
    target_timeout = command_document.get("configured_target_timeout_seconds")
    outer_timeout = command_document.get("configured_timeout_seconds")
    if not isinstance(target_timeout, int) or target_timeout < 1:
        raise CaptureError(f"{command_path}: invalid target timeout")
    if not isinstance(outer_timeout, int) or outer_timeout < target_timeout:
        raise CaptureError(f"{command_path}: invalid wrapper timeout")
    capture_command, command = nvbit_commands(
        destination, cell, variant, target_timeout
    )
    run_manifest = run_root / "wrapper.run.json"
    output_paths = {
        "campaign_binding": destination / "campaign_binding.json",
        "complete": destination / "complete.json",
        "dynamic_trace": destination / "dynamic_trace.pb",
    }
    validate_run_manifest(
        run_manifest,
        command=command,
        timeout=outer_timeout,
        stdout_path=run_root / "wrapper.stdout.txt",
        stderr_path=run_root / "wrapper.stderr.txt",
        input_paths={"nvbit_wrapper": NVBIT_CAPTURE, "probe_binary": BINARY},
        output_paths=output_paths,
        environment_value=environment(),
    )
    expected_document = {
        "schema_version": 2,
        "capture_command": capture_command,
        "command": command,
        "configured_target_timeout_seconds": target_timeout,
        "configured_timeout_seconds": outer_timeout,
        "binary": dict(binary),
        "nvbit_wrapper": file_identity(NVBIT_CAPTURE),
        "environment_overrides": environment_overrides(),
        "outputs": identities(output_paths),
        "run_manifest": file_identity(run_manifest),
    }
    if command_document != expected_document:
        raise CaptureError(f"{command_path}: stale NVBit wrapper binding")


def capture_nvbit(root: Path, cells: Sequence[Cell], timeout: int) -> None:
    binary = retained_binary(root)
    for cell in cells:
        for variant in VARIANTS:
            name = cell.endpoint_name(variant)
            destination = root / "nvbit" / name
            if (destination / "complete.json").is_file():
                validate_nvbit(root, destination, cell, variant, binary)
                print(f"nvbit existing: {name}")
                continue
            run_root = root / "runs/nvbit" / name
            if destination.exists() and any(destination.iterdir()):
                raise CaptureError(
                    f"{destination}: incomplete NVBit output must be moved aside before recapture"
                )
            if run_root.exists() and any(run_root.iterdir()):
                raise CaptureError(
                    f"{run_root}: incomplete NVBit run record must be moved aside before recapture"
                )
            run_root.mkdir(parents=True, exist_ok=True)
            capture_command, command = nvbit_commands(
                destination, cell, variant, timeout
            )
            outer_timeout = timeout + 120
            output_paths = {
                "campaign_binding": destination / "campaign_binding.json",
                "complete": destination / "complete.json",
                "dynamic_trace": destination / "dynamic_trace.pb",
            }
            recorded_run(
                command,
                timeout=outer_timeout,
                manifest_path=run_root / "wrapper.run.json",
                stdout_path=run_root / "wrapper.stdout.txt",
                stderr_path=run_root / "wrapper.stderr.txt",
                input_paths={
                    "nvbit_wrapper": NVBIT_CAPTURE,
                    "probe_binary": BINARY,
                },
                output_paths=output_paths,
                environment_value=environment(),
            )
            atomic_write_json(
                run_root / "command.json",
                {
                    "schema_version": 2,
                    "capture_command": capture_command,
                    "command": command,
                    "configured_target_timeout_seconds": timeout,
                    "configured_timeout_seconds": outer_timeout,
                    "binary": binary,
                    "nvbit_wrapper": file_identity(NVBIT_CAPTURE),
                    "environment_overrides": environment_overrides(),
                    "outputs": identities(output_paths),
                    "run_manifest": file_identity(run_root / "wrapper.run.json"),
                },
            )
            validate_nvbit(root, destination, cell, variant, binary)
            print(f"nvbit: {name}", flush=True)


def build_simulator(timeout: int) -> None:
    command = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(REPO_ROOT / "sim/Cargo.toml"),
        "-p",
        "sim-run",
    ]
    run_unrecorded(
        command, timeout=timeout, environment_value=environment()
    )
    if not SIM_RUN.is_file() or not os.access(SIM_RUN, os.X_OK):
        raise CaptureError(f"sim-run build did not produce {SIM_RUN}")


def replay_command(root: Path, cell: Cell, variant: str) -> list[str]:
    return [
        str(SIM_RUN),
        "--trace",
        str(root / "nvbit" / cell.endpoint_name(variant)),
        "--control",
        str(root / "static/control_bits.json"),
        "--arch-card",
        str(ARCH_CARD),
        "--pchase-card",
        str(PCHASE_CARD),
        "--mode",
        "l1-l2",
    ]


def validate_replay(
    root: Path, destination: Path, cell: Cell, variant: str
) -> None:
    first = destination / "replay-1.json"
    second = destination / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise CaptureError(f"{destination}: incomplete replay twins")
    if first.read_bytes() != second.read_bytes():
        raise CaptureError(f"{destination}: replay twins differ")
    read_json(first)
    trace = root / "nvbit" / cell.endpoint_name(variant) / "dynamic_trace.pb"
    control = root / "static/control_bits.json"
    command_path = destination / "command.json"
    command_document = read_json(command_path)
    configured = command_document.get("configured_timeout_seconds")
    if not isinstance(configured, int) or configured < 1:
        raise CaptureError(f"{command_path}: invalid configured timeout")
    command = replay_command(root, cell, variant)
    input_paths = {
        "arch_card": ARCH_CARD,
        "control": control,
        "pchase_card": PCHASE_CARD,
        "sim_run": SIM_RUN,
        "trace": trace,
    }
    for repeat in (1, 2):
        validate_run_manifest(
            destination / f"replay-{repeat}.run.json",
            command=command,
            timeout=configured,
            stdout_path=destination / f"replay-{repeat}.json",
            stderr_path=destination / f"stderr-{repeat}.txt",
            input_paths=input_paths,
            output_paths={
                "replay": destination / f"replay-{repeat}.json"
            },
            environment_value=environment(),
        )
    expected = {
        "schema_version": 2,
        "command": command,
        "configured_timeout_seconds": configured,
        "inputs": identities(input_paths),
        "outputs": identities({"replay_1": first, "replay_2": second}),
        "run_manifests": identities(
            {
                "replay_1": destination / "replay-1.run.json",
                "replay_2": destination / "replay-2.run.json",
            }
        ),
        "environment_overrides": environment_overrides(),
    }
    if command_document != expected:
        raise CaptureError(f"{destination}: replay input/binary binding mismatch")


def capture_replay(root: Path, cells: Sequence[Cell], timeout: int) -> None:
    build_simulator(timeout)
    for cell in cells:
        for variant in VARIANTS:
            name = cell.endpoint_name(variant)
            trace = root / "nvbit" / name / "dynamic_trace.pb"
            if not trace.is_file() or trace.stat().st_size == 0:
                raise CaptureError(f"missing trace for {name}: {trace}")
            destination = root / "simulator" / name
            if (destination / "command.json").is_file():
                validate_replay(root, destination, cell, variant)
                print(f"replay existing: {name}")
                continue
            destination.mkdir(parents=True, exist_ok=True)
            command = replay_command(root, cell, variant)
            input_paths = {
                "arch_card": ARCH_CARD,
                "control": root / "static/control_bits.json",
                "pchase_card": PCHASE_CARD,
                "sim_run": SIM_RUN,
                "trace": trace,
            }
            for repeat in (1, 2):
                stage_files = (
                    destination / f"replay-{repeat}.json",
                    destination / f"stderr-{repeat}.txt",
                    destination / f"replay-{repeat}.run.json",
                )
                if complete_stage_or_absent(
                    stage_files, f"replay {name} repeat {repeat}"
                ):
                    manifest = read_json(
                        destination / f"replay-{repeat}.run.json"
                    )
                    configured = manifest.get("configured_timeout_seconds")
                    if not isinstance(configured, int) or configured < 1:
                        raise CaptureError(
                            f"{destination / f'replay-{repeat}.run.json'}: "
                            "invalid configured timeout"
                        )
                    validate_run_manifest(
                        destination / f"replay-{repeat}.run.json",
                        command=command,
                        timeout=configured,
                        stdout_path=destination / f"replay-{repeat}.json",
                        stderr_path=destination / f"stderr-{repeat}.txt",
                        input_paths=input_paths,
                        output_paths={
                            "replay": destination / f"replay-{repeat}.json"
                        },
                        environment_value=environment(),
                    )
                    if configured != timeout:
                        raise CaptureError(
                            f"{destination}: replay twins use different configured timeouts"
                        )
                    continue
                recorded_run(
                    command,
                    timeout=timeout,
                    manifest_path=destination / f"replay-{repeat}.run.json",
                    stdout_path=destination / f"replay-{repeat}.json",
                    stderr_path=destination / f"stderr-{repeat}.txt",
                    input_paths=input_paths,
                    output_paths={
                        "replay": destination / f"replay-{repeat}.json"
                    },
                    environment_value=environment(),
                )
            atomic_write_json(
                destination / "command.json",
                {
                    "schema_version": 2,
                    "command": command,
                    "configured_timeout_seconds": timeout,
                    "inputs": identities(input_paths),
                    "outputs": identities(
                        {
                            "replay_1": destination / "replay-1.json",
                            "replay_2": destination / "replay-2.json",
                        }
                    ),
                    "run_manifests": identities(
                        {
                            "replay_1": destination / "replay-1.run.json",
                            "replay_2": destination / "replay-2.run.json",
                        }
                    ),
                    "environment_overrides": environment_overrides(),
                },
            )
            validate_replay(root, destination, cell, variant)
            print(f"replay: {name}", flush=True)


def validate_selected(root: Path, cells: Sequence[Cell], repetitions: Sequence[int]) -> None:
    binary = retained_binary(root)
    for relative in (
        "static/control_bits.json",
        "static/resources.txt",
        "static/sass.txt",
        "static/build.log",
    ):
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise CaptureError(f"missing static artifact: {path}")
    for cell in cells:
        for repetition in repetitions:
            validate_direct(
                root / "direct" / cell.direct_name / f"repeat-{repetition}",
                cell,
                repetition,
                binary,
            )
        for variant in VARIANTS:
            validate_nvbit(
                root,
                root / "nvbit" / cell.endpoint_name(variant),
                cell,
                variant,
                binary,
            )
            validate_replay(
                root,
                root / "simulator" / cell.endpoint_name(variant),
                cell,
                variant,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--phase",
        choices=("all", "static", "direct", "nvbit", "replay"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cells", help="comma-separated subset such as d1l16,d2l4")
    parser.add_argument("--repeats", help="comma-separated subset in 1..10")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        if args.timeout < 1:
            raise CaptureError("--timeout must be positive")
        cells = selected_cells(args.smoke, args.cells)
        repetitions = selected_repetitions(args.smoke, args.repeats)
        root = args.root.resolve()
        if args.validate_only:
            validate_selected(root, cells, repetitions)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "root": str(root),
                        "cells": [f"d{cell.depth}l{cell.lead}" for cell in cells],
                        "repetitions": repetitions,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.phase in {"all", "static"}:
            capture_static(root, args.timeout)
        if args.phase in {"all", "direct"}:
            capture_direct(root, cells, repetitions, args.timeout)
        if args.phase in {"all", "nvbit"}:
            capture_nvbit(root, cells, args.timeout)
        if args.phase in {"all", "replay"}:
            capture_replay(root, cells, args.timeout)
        return 0
    except (CaptureError, OSError, ValueError) as error:
        print(f"multiline execution capture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
