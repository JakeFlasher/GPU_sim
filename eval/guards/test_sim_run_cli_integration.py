from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from eval.contracts.validation import (
    load_json_file_strict,
    loads_json_strict,
    validate_manifest,
    validate_result,
    validate_sim_error,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path("sim/run/tests/fixtures/addr_carry_canary_bundle")
MANIFEST_V2 = FIXTURE_ROOT / "run_manifest.v2.json"
MANIFEST_V1 = FIXTURE_ROOT / "run_manifest.json"
MISSING_MANIFEST = FIXTURE_ROOT / "task345-missing-manifest.json"
SIM_RUN_ARGV = (
    "cargo",
    "run",
    "--quiet",
    "--manifest-path",
    "sim/Cargo.toml",
    "-p",
    "sim-run",
    "--",
)
SUBPROCESS_TIMEOUT_SECONDS = 300.0
MAX_OUTPUT_BYTES_PER_STREAM = 1024 * 1024


def _sanitized_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["CARGO_TERM_COLOR"] = "never"
    environment["NO_COLOR"] = "1"
    return environment


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_sim_run(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    argv = [*SIM_RUN_ARGV, *arguments]
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=_sanitized_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    streams = {
        process.stdout: bytearray(),
        process.stderr: bytearray(),
    }
    deadline = time.monotonic() + SUBPROCESS_TIMEOUT_SECONDS
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                selector.register(stream, selectors.EVENT_READ)

            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _kill_process_group(process)
                    pytest.fail(
                        f"sim-run exceeded {SUBPROCESS_TIMEOUT_SECONDS:.0f}s timeout"
                    )
                for key, _ in selector.select(timeout=min(0.1, remaining)):
                    stream = key.fileobj
                    assert hasattr(stream, "read")
                    buffer = streams[stream]
                    allowance = MAX_OUTPUT_BYTES_PER_STREAM - len(buffer) + 1
                    chunk = os.read(key.fd, min(64 * 1024, allowance))
                    if not chunk:
                        selector.unregister(stream)
                        continue
                    buffer.extend(chunk)
                    if len(buffer) > MAX_OUTPUT_BYTES_PER_STREAM:
                        _kill_process_group(process)
                        pytest.fail(
                            "sim-run exceeded the per-stream output limit of "
                            f"{MAX_OUTPUT_BYTES_PER_STREAM} bytes"
                        )

        return_code = process.wait(timeout=5)
    finally:
        if process.poll() is None:
            _kill_process_group(process)
        process.stdout.close()
        process.stderr.close()

    return subprocess.CompletedProcess(
        argv,
        return_code,
        stdout=bytes(streams[process.stdout]),
        stderr=bytes(streams[process.stderr]),
    )


def _load_strict_output(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        pytest.fail(f"{source} is not strict UTF-8: {exc}")
    document = loads_json_strict(text, source=source)
    assert isinstance(document, dict), f"{source} must contain one JSON object"
    return document


def _canonical_json_line(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _assert_typed_error(
    completed: subprocess.CompletedProcess[bytes],
    *,
    failure_type: str,
    stage: str,
) -> dict[str, Any]:
    assert completed.returncode != 0
    assert completed.stdout == b""
    assert completed.stderr != b""
    document = _load_strict_output(completed.stderr, source="sim-run stderr")
    validate_sim_error(document)
    assert completed.stderr == _canonical_json_line(document)
    assert completed.returncode == document["exit_status"]["exit_code"]
    assert document["error"]["failure_type"] == failure_type
    assert document["error"]["stage"] == stage
    return document


def test_sim_run_v2_cli_success_is_strict_validated_canonical_json() -> None:
    # Admit the checked-in input through the real production API before execution;
    # no launch-manifest or baseline producer participates in this test.
    manifest = load_json_file_strict(ROOT / MANIFEST_V2)
    validate_manifest(manifest)

    completed = _run_sim_run(MANIFEST_V2.as_posix())

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stderr == b""
    assert completed.stdout != b""
    document = _load_strict_output(completed.stdout, source="sim-run stdout")
    validate_result(document)
    assert completed.stdout == _canonical_json_line(document)
    # Build identity and binary SHA are intentionally build-specific. Fixed-context
    # Rust tests, not this public-CLI integration test, own the full byte golden.


@pytest.mark.parametrize(
    ("arguments", "provided_count"),
    [
        ((), 0),
        ((MANIFEST_V2.as_posix(), MANIFEST_V2.as_posix()), 2),
    ],
)
def test_sim_run_usage_errors_are_typed_and_exclusive(
    arguments: tuple[str, ...], provided_count: int
) -> None:
    completed = _run_sim_run(*arguments)

    document = _assert_typed_error(
        completed,
        failure_type="CLI_USAGE_FAILED",
        stage="cli_usage",
    )
    assert document["error"]["evidence"] == {
        "provided_positional_argument_count": provided_count,
        "expected_positional_argument_count": 1,
    }


def test_sim_run_missing_manifest_is_a_typed_manifest_error() -> None:
    assert not (ROOT / MISSING_MANIFEST).exists()

    completed = _run_sim_run(MISSING_MANIFEST.as_posix())

    _assert_typed_error(
        completed,
        failure_type="MANIFEST_ADMISSION_FAILED",
        stage="manifest_admission",
    )


def test_sim_run_malformed_manifest_is_a_typed_manifest_error(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed-run-manifest.json"
    malformed.write_bytes(b'{"schema_version":2,"scope":')

    completed = _run_sim_run(str(malformed))

    _assert_typed_error(
        completed,
        failure_type="MANIFEST_ADMISSION_FAILED",
        stage="manifest_admission",
    )


def test_sim_run_rejects_v1_manifest_with_a_typed_manifest_error() -> None:
    legacy = load_json_file_strict(ROOT / MANIFEST_V1)
    assert legacy["schema_version"] == 1

    completed = _run_sim_run(MANIFEST_V1.as_posix())

    _assert_typed_error(
        completed,
        failure_type="MANIFEST_ADMISSION_FAILED",
        stage="manifest_admission",
    )
