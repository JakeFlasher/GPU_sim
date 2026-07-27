#!/usr/bin/env python3
"""Hold the golden capture process lock while executing the shell collector."""
from __future__ import annotations

import argparse
import errno
import hashlib
import os
import socket
import sys
from pathlib import Path

_LOCK_FD_ENVIRONMENT = "GOLDEN_CAPTURE_LOCK_FD"
_LOCK_PREFIX = b"\0gpu-simulator-golden-capture-v1-"
_CLOSED_PATH = "/bin:/usr/bin"
_PROHIBITED_STARTUP_ENVIRONMENT = frozenset({
    "BASH_ENV",
    "CUDA_DEVICE_ORDER",
    "CUDA_INJECTION32_PATH",
    "CUDA_INJECTION64_PATH",
    "CUDA_VISIBLE_DEVICES",
    "ENV",
    "GPU_DEVICE_ORDINAL",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "NVBIT_TOOL",
    "NVIDIA_VISIBLE_DEVICES",
    "NV_GPU",
    "PYTHONHOME",
    "PYTHONPATH",
    "TRACES_FOLDER",
    "USER_DEFINED_FOLDERS",
})


class GoldenCaptureLockError(RuntimeError):
    """The process-global golden capture lock is absent or invalid."""


def _lock_name(repo_root: Path) -> bytes:
    canonical = repo_root.resolve(strict=True)
    digest = hashlib.sha256(os.fsencode(canonical)).hexdigest().encode("ascii")
    return _LOCK_PREFIX + digest


def _validate_socket_descriptor(descriptor: int, expected_name: bytes) -> None:
    try:
        retained = socket.socket(fileno=descriptor)
    except (OSError, ValueError) as exc:
        raise GoldenCaptureLockError("capture lock descriptor is unavailable") from exc
    try:
        if (
            retained.family != socket.AF_UNIX
            or retained.type & socket.SOCK_STREAM != socket.SOCK_STREAM
            or retained.getsockname() != expected_name
        ):
            raise GoldenCaptureLockError(
                "capture lock descriptor does not own the expected abstract socket"
            )
    except OSError as exc:
        raise GoldenCaptureLockError("capture lock descriptor is invalid") from exc
    finally:
        retained.detach()


def _parse_descriptor(value: str) -> int:
    try:
        descriptor = int(value, 10)
    except ValueError as exc:
        raise GoldenCaptureLockError("capture lock descriptor is not an integer") from exc
    if descriptor < 3:
        raise GoldenCaptureLockError("capture lock descriptor is reserved")
    return descriptor


def _validate_startup_environment() -> None:
    prohibited = sorted(
        name for name in _PROHIBITED_STARTUP_ENVIRONMENT if name in os.environ
    )
    if prohibited:
        raise GoldenCaptureLockError(
            "prohibited startup environment is present: " + ", ".join(prohibited)
        )


def _execution_environment(lock_descriptor: int) -> dict[str, str]:
    _validate_startup_environment()
    return {
        _LOCK_FD_ENVIRONMENT: str(lock_descriptor),
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": _CLOSED_PATH,
        "PYTHONNOUSERSITE": "1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--verify-fd")
    parser.add_argument("--exec-script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        expected_name = _lock_name(args.repo_root)
        _validate_startup_environment()
        if args.verify_fd is not None:
            if args.exec_script is not None or args.arguments:
                raise GoldenCaptureLockError(
                    "lock verification does not accept an execution command"
                )
            _validate_socket_descriptor(
                _parse_descriptor(args.verify_fd), expected_name,
            )
            return 0
        if args.exec_script is None:
            raise GoldenCaptureLockError("lock execution requires --exec-script")
        if not args.exec_script.is_absolute():
            script = (Path.cwd() / args.exec_script).resolve(strict=True)
        else:
            script = args.exec_script.resolve(strict=True)
        expected_script = (
            args.repo_root.resolve(strict=True)
            / "tools" / "trace_sampling" / "golden_cells"
            / "capture_golden_cells.sh"
        )
        if script != expected_script:
            raise GoldenCaptureLockError(
                "capture lock execution script is not the golden collector"
            )
        lock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            lock.bind(expected_name)
        except OSError as exc:
            lock.close()
            if exc.errno == errno.EADDRINUSE:
                raise GoldenCaptureLockError(
                    "another golden capture owns the process-global lock"
                ) from exc
            raise GoldenCaptureLockError(f"cannot bind golden capture lock: {exc}") from exc
        lock.set_inheritable(True)
        environment = _execution_environment(lock.fileno())
        arguments = list(args.arguments)
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        os.execve(script, [os.fspath(script), *arguments], environment)
        raise AssertionError("unreachable")
    except (OSError, GoldenCaptureLockError) as exc:
        print(f"golden_capture_lock_refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
