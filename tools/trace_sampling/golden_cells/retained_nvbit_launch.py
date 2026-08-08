"""Run one golden cell with retained binary and tracer capabilities."""
from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import campaign_preflight  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import protected_paths  # noqa: E402
from tools.nvbit_tracer import capture_process  # noqa: E402
from tools.nvbit_tracer import runtime_env  # noqa: E402
from tools.nvbit_tracer.external_tracer import (  # noqa: E402
    EXTERNAL_TRACER_MAX_BYTES,
)


_ACTION_TIMEOUT_SECONDS = 15 * 60
_MAX_ACTION_OUTPUT_BYTES = 64 * 1024 * 1024
_GOLDEN_STAGING_ROOT = (REPO_ROOT / "results" / "nvbit_capture_staging").resolve()
_TRACER_SOURCE_KIND = "external_non_git_retained_descriptor_v1"
_RETAINED_TRACER_NAME = "nvbit_retained_tracer.so"
_STDOUT_NAME = "nvbit_launch.stdout.bin"
_STDERR_NAME = "nvbit_launch.stderr.bin"
_ATTEMPT_NAME = "nvbit_launch_attempt.json"


class RetainedNvbitError(RuntimeError):
    """The admitted binary or tracer could not be retained."""


def _require_lower_hex_sha256(value: str, *, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RetainedNvbitError(f"{label} must be lowercase 64-hex")
    return value


def _require_staging_trace(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(candidate)
    if re.fullmatch(r"trace\.staged\.[1-9][0-9]*", candidate.name) is None:
        raise RetainedNvbitError(
            f"trace output does not use the closed staged trace name: {candidate}"
        )
    try:
        metadata = candidate.lstat()
    except FileNotFoundError:
        pass
    else:
        kind = "directory" if stat.S_ISDIR(metadata.st_mode) else "entry"
        raise RetainedNvbitError(
            f"trace output {kind} already exists and cannot be reused: {candidate}"
        )
    if candidate.resolve(strict=False) != candidate:
        raise RetainedNvbitError(f"trace output contains a symlink: {candidate}")
    try:
        candidate.relative_to(_GOLDEN_STAGING_ROOT)
        protected_paths.assert_write_allowed(
            candidate,
            repo_root=REPO_ROOT,
            operation="retained golden trace output",
        )
    except (ValueError, protected_paths.ProtectedPathError) as exc:
        raise RetainedNvbitError(
            f"trace output is outside the guarded golden staging root: {candidate}"
        ) from exc
    return candidate


def _write_attempt_artifacts(
    trace_descriptor: int,
    *,
    tracer_payload: bytes,
    stdout: bytes,
    stderr: bytes,
) -> None:
    immutable.write_regular_bytes_at(
        trace_descriptor,
        _RETAINED_TRACER_NAME,
        tracer_payload,
        mode=0o444,
    )
    immutable.write_regular_bytes_at(
        trace_descriptor,
        _STDOUT_NAME,
        stdout,
        mode=0o444,
    )
    immutable.write_regular_bytes_at(
        trace_descriptor,
        _STDERR_NAME,
        stderr,
        mode=0o444,
    )


def _write_attempt(
    trace_descriptor: int,
    *,
    args: argparse.Namespace,
    binary_path: Path,
    binary_sha256: str,
    binary_size_bytes: int,
    tracer_path: Path,
    tracer_sha256: str,
    tracer_size_bytes: int,
    status: str,
    stdout: bytes,
    stderr: bytes,
    returncode: int | None,
    error_code: str | None,
    error_detail: str | None,
    process_id: int | None,
    started_at_utc: str | None,
    trace_generation_identity: tuple[int, int],
    teardown_failures: list[BaseException],
) -> None:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "golden_nvbit_launch_attempt",
        "status": status,
        "preflight_sha256": args.preflight_sha256,
        "target_binary": {
            "path": str(binary_path),
            "sha256": binary_sha256,
            "size_bytes": binary_size_bytes,
        },
        "tracer": {
            "source_kind": args.tracer_source_kind,
            "source_path": str(tracer_path),
            "expected_sha256": args.tracer_sha256,
            "sha256": tracer_sha256,
            "size_bytes": tracer_size_bytes,
            "snapshot_path": _RETAINED_TRACER_NAME,
            "execution_mode": "retained_proc_fd_v1",
        },
        "descriptor_bindings": {
            "command_0": "target_binary",
            "CUDA_INJECTION64_PATH": "tracer",
            "TRACES_FOLDER": "trace_directory",
        },
        "trace_generation": {
            "device": trace_generation_identity[0],
            "inode": trace_generation_identity[1],
        },
        "process": {
            "returncode": returncode,
            "error_code": error_code,
            "error_detail": error_detail,
            "process_id": process_id,
            "started_at_utc": started_at_utc,
        },
        "teardown_failures": [
            {
                "error_code": getattr(
                    teardown_failure,
                    "code",
                    type(teardown_failure).__name__,
                ),
                "error_detail": str(teardown_failure),
            }
            for teardown_failure in teardown_failures
        ],
        "streams": {
            "stdout": {
                "path": _STDOUT_NAME,
                "sha256": immutable.sha256_bytes(stdout),
                "size_bytes": len(stdout),
            },
            "stderr": {
                "path": _STDERR_NAME,
                "sha256": immutable.sha256_bytes(stderr),
                "size_bytes": len(stderr),
            },
        },
    }
    sealed = immutable.seal_model(receipt, field="attempt_seal_sha256")
    immutable.write_regular_bytes_at(
        trace_descriptor,
        _ATTEMPT_NAME,
        immutable.canonical_json_bytes(sealed),
        mode=0o444,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--tracer-sha256", required=True)
    parser.add_argument("--tracer-source-kind", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--pb-lib", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    attempt_descriptor: int | None = None
    sealing_root: immutable.PinnedDirectoryReader | None = None
    try:
        _require_lower_hex_sha256(args.binary_sha256, label="binary SHA-256")
        _require_lower_hex_sha256(args.tracer_sha256, label="tracer SHA-256")
        _require_lower_hex_sha256(
            args.preflight_sha256,
            label="preflight SHA-256",
        )
        if args.tracer_source_kind != _TRACER_SOURCE_KIND:
            raise RetainedNvbitError("tracer source kind is not admitted for live use")
        try:
            args.tracer = campaign_preflight._absolute_external_tracer(
                args.tracer,
                repo_root=REPO_ROOT,
            )
        except ValueError as exc:
            raise RetainedNvbitError(
                f"external tracer validation failed: {exc}"
            ) from exc
        trace_dir = _require_staging_trace(args.trace_dir)
        immutable.ensure_directory(trace_dir.parent)
        with immutable.PinnedDirectoryReader(trace_dir.parent) as trace_parent:
            trace_parent.verify_path()
            try:
                os.mkdir(trace_dir.name, mode=0o755, dir_fd=trace_parent.descriptor)
            except FileExistsError as exc:
                raise RetainedNvbitError(
                    f"trace output appeared before exclusive creation: {trace_dir}"
                ) from exc
            os.fsync(trace_parent.descriptor)
            trace_parent.verify_path()

        result = None
        failure: BaseException | None = None
        teardown_failures: list[BaseException] = []
        binary_path: Path | None = None
        binary_sha256: str | None = None
        binary_size_bytes: int | None = None
        tracer_path: Path | None = None
        tracer_sha256: str | None = None
        tracer_size_bytes: int | None = None
        tracer_payload: bytes | None = None
        try:
            with (
                immutable.open_retained_executable(args.binary) as binary,
                immutable.open_retained_regular(
                    args.tracer, max_bytes=EXTERNAL_TRACER_MAX_BYTES
                ) as tracer,
                immutable.PinnedDirectoryReader(trace_dir) as trace_root,
            ):
                binary_path = binary.path
                binary_sha256 = binary.sha256
                binary_size_bytes = os.fstat(binary.descriptor).st_size
                tracer_path = tracer.path
                tracer_sha256 = tracer.sha256
                tracer_size_bytes = tracer.size_bytes
                tracer_payload = tracer.read_bytes()
                attempt_descriptor = os.dup(trace_root.descriptor)
                try:
                    if binary.sha256 != args.binary_sha256:
                        raise RetainedNvbitError(
                            "target binary SHA-256 changed before NVBit"
                        )
                    if tracer.sha256 != args.tracer_sha256:
                        raise RetainedNvbitError(
                            "tracer SHA-256 changed before NVBit"
                        )
                    with runtime_env.retained_tracer_runtime_env(
                        args.tracer,
                        base_env={
                            "LANG": "C",
                            "LANGUAGE": "C",
                            "LC_ALL": "C",
                            "PATH": "/bin:/usr/bin",
                            "PYTHONNOUSERSITE": "1",
                        },
                        pb_lib=args.pb_lib,
                        runtime_manifest=runtime_env.default_runtime_manifest_path(
                            args.tracer
                        ),
                        retained_descriptor=tracer.descriptor,
                    ) as retained_runtime:
                        env = dict(retained_runtime.environment)
                        env.update({
                            "CUDA_INJECTION64_PATH": tracer.proc_path,
                            "USER_DEFINED_FOLDERS": "1",
                            "TRACES_FOLDER": f"/proc/self/fd/{trace_root.descriptor}",
                        })
                        try:
                            result = capture_process.execute_capture_process(
                                (binary.proc_path,),
                                environment=env,
                                working_directory="/",
                                inherited_descriptors=(
                                    binary.descriptor,
                                    tracer.descriptor,
                                    trace_root.descriptor,
                                    *retained_runtime.pass_fds,
                                ),
                                timeout_seconds=_ACTION_TIMEOUT_SECONDS,
                                max_output_bytes=_MAX_ACTION_OUTPUT_BYTES,
                                timeout_code="golden_nvbit_launch_timeout",
                                launch_code="golden_nvbit_launch_failed",
                            )
                        except (
                            OSError,
                            RetainedNvbitError,
                            immutable.ImmutableArtifactError,
                            capture_process.NvbitCaptureProcessError,
                            runtime_env.TracerRuntimeError,
                        ) as exc:
                            failure = exc
                    if failure is None:
                        binary.verify_unchanged()
                        tracer.verify_unchanged()
                except (
                    OSError,
                    RetainedNvbitError,
                    immutable.ImmutableArtifactError,
                    capture_process.NvbitCaptureProcessError,
                    runtime_env.TracerRuntimeError,
                ) as exc:
                    if failure is None:
                        failure = exc
                    else:
                        teardown_failures.append(exc)
        except (
            OSError,
            RetainedNvbitError,
            immutable.ImmutableArtifactError,
            capture_process.NvbitCaptureProcessError,
            runtime_env.TracerRuntimeError,
        ) as exc:
            if failure is None:
                failure = exc
            else:
                teardown_failures.append(exc)

        if (
            attempt_descriptor is None
            or binary_path is None
            or binary_sha256 is None
            or binary_size_bytes is None
            or tracer_path is None
            or tracer_sha256 is None
            or tracer_size_bytes is None
            or tracer_payload is None
        ):
            if failure is not None:
                raise failure
            raise RetainedNvbitError(
                "retained launch metadata is incomplete before attempt sealing"
            )

        trace_metadata = os.fstat(attempt_descriptor)
        trace_generation_identity = (
            trace_metadata.st_dev,
            trace_metadata.st_ino,
        )
        try:
            sealing_root = immutable.PinnedDirectoryReader(trace_dir)
            sealing_root.__enter__()
            sealing_metadata = os.fstat(sealing_root.descriptor)
            sealing_identity = (
                sealing_metadata.st_dev,
                sealing_metadata.st_ino,
            )
            if sealing_identity != trace_generation_identity:
                raise immutable.ImmutableArtifactError(
                    "staged trace generation changed before attempt sealing"
                )
        except (OSError, immutable.ImmutableArtifactError) as exc:
            if sealing_root is not None:
                try:
                    sealing_root.close()
                except (OSError, immutable.ImmutableArtifactError) as close_exc:
                    teardown_failures.append(close_exc)
                sealing_root = None
            if failure is None:
                failure = exc
            else:
                teardown_failures.append(exc)

        stdout = (
            result.stdout
            if result is not None
            else getattr(failure, "stdout", b"")
        )
        stderr = (
            result.stderr
            if result is not None
            else getattr(failure, "stderr", b"")
        )
        returncode = result.returncode if result is not None else None
        if failure is not None:
            error_code = getattr(
                failure,
                "code",
                type(failure).__name__,
            )
            error_detail = str(failure)
            status = "failed"
            if result is not None:
                process_id = result.process_id
                started_at_utc = result.started_at_utc
            else:
                process_id = getattr(failure, "process_id", None)
                started_at_utc = getattr(failure, "started_at_utc", None)
        elif returncode != 0:
            error_code = "target_nonzero_exit"
            error_detail = f"target exited with status {returncode}"
            status = "failed"
            process_id = result.process_id
            started_at_utc = result.started_at_utc
        else:
            error_code = None
            error_detail = None
            status = "succeeded"
            process_id = result.process_id
            started_at_utc = result.started_at_utc

        _write_attempt_artifacts(
            attempt_descriptor,
            tracer_payload=tracer_payload,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            receipt_descriptor = os.dup(attempt_descriptor)
        except OSError as exc:
            teardown_failures.append(exc)
        else:
            descriptor_to_close = attempt_descriptor
            attempt_descriptor = receipt_descriptor
            try:
                os.close(descriptor_to_close)
            except OSError as exc:
                teardown_failures.append(exc)
        if teardown_failures:
            status = "failed"

        _write_attempt(
            attempt_descriptor,
            args=args,
            binary_path=binary_path,
            binary_sha256=binary_sha256,
            binary_size_bytes=binary_size_bytes,
            tracer_path=tracer_path,
            tracer_sha256=tracer_sha256,
            tracer_size_bytes=tracer_size_bytes,
            status=status,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            error_code=error_code,
            error_detail=error_detail,
            process_id=process_id,
            started_at_utc=started_at_utc,
            trace_generation_identity=trace_generation_identity,
            teardown_failures=teardown_failures,
        )
        if stdout:
            sys.stdout.buffer.write(stdout)
        if stderr:
            sys.stderr.buffer.write(stderr)
        if status == "succeeded":
            descriptor_to_close = attempt_descriptor
            attempt_descriptor = None
            immutable._close_descriptors(
                (descriptor_to_close,),
                label="final retained trace descriptor close",
            )
            if sealing_root is not None:
                sealing_to_close = sealing_root
                sealing_root = None
                sealing_to_close.__exit__(None, None, None)
            return 0
        if sealing_root is not None:
            sealing_to_close = sealing_root
            sealing_root = None
            sealing_to_close.__exit__(None, None, None)
        if failure is not None:
            print(f"retained_nvbit_launch_refused: {failure}", file=sys.stderr)
        for teardown_failure in teardown_failures:
            print(
                "retained_nvbit_launch_refused: teardown also failed: "
                f"{teardown_failure}",
                file=sys.stderr,
            )
        if failure is not None or teardown_failures:
            return 2
        return returncode if returncode is not None else 2
    except (
        OSError,
        immutable.ImmutableArtifactError,
        RetainedNvbitError,
    ) as exc:
        print(f"retained_nvbit_launch_refused: {exc}", file=sys.stderr)
        return 2
    finally:
        if sealing_root is not None:
            try:
                sealing_root.close()
            except (OSError, immutable.ImmutableArtifactError) as exc:
                print(
                    "retained_nvbit_launch_cleanup_warning: attempt sealing "
                    f"monitor close failed: {exc}",
                    file=sys.stderr,
                )
        if attempt_descriptor is not None:
            try:
                os.close(attempt_descriptor)
            except OSError as exc:
                print(
                    "retained_nvbit_launch_cleanup_warning: final retained trace "
                    f"descriptor close failed: {exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
