"""Streaming JSONL I/O with mandatory zstd output for replay artifacts.

Workload-pack JSONL files are outside this helper's scope. Replay and
post-replay artifacts use ``*.jsonl.zst`` by default while readers accept both
the compressed form and legacy uncompressed files.
"""

from __future__ import annotations

from contextlib import contextmanager
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from typing import Iterator, TextIO


def _require_zstd() -> str:
    executable = shutil.which("zstd")
    if executable is None:
        raise RuntimeError("zstd is required for replay JSONL artifacts")
    return executable


def _diagnostic_bytes(data: bytes) -> str:
    """Render compressor diagnostics without changing artifact text."""
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return repr(data)


def _terminate_process_group(process: subprocess.Popen[bytes], *, grace: float = 2.0) -> None:
    """Terminate a child session, including descendants after the leader exits.

    ``Popen.poll()`` only describes the session leader.  A child can exit while
    a descendant keeps the process group alive, so checking ``poll`` before
    ``killpg`` is an unsafe shortcut.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def resolve_jsonl_path(path: Path) -> Path:
    """Resolve a logical JSONL path to compressed output or legacy input."""
    path = Path(path)
    if path.name.endswith(".jsonl.zst"):
        # Callers commonly persist the published name.  If an older archive
        # only has the pre-compression sibling, recover it transparently.
        if path.is_file():
            return path
        legacy = Path(str(path)[: -len(".zst")])
        if legacy.is_file():
            return legacy
        return path
    compressed = Path(str(path) + ".zst")
    if compressed.is_file():
        return compressed
    return path


@contextmanager
def open_jsonl_reader(path: Path) -> Iterator[TextIO]:
    """Read either a zstd-compressed JSONL artifact or a legacy JSONL file."""
    resolved = resolve_jsonl_path(path)
    if resolved.name.endswith(".jsonl.zst"):
        process = subprocess.Popen(
            [_require_zstd(), "-q", "-dc", str(resolved)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        stderr_pipe = process.stderr
        # JSONL is a UTF-8 contract.  Replacement decoding would silently
        # turn a corrupt artifact into different JSON/text.
        stream = io.TextIOWrapper(process.stdout, encoding="utf-8")
        try:
            yield stream
            # A caller may intentionally consume only a prefix. Probe for EOF
            # so early termination does not turn zstd's SIGPIPE into a false
            # decompression failure; full readers reach EOF and are checked.
            exhausted = stream.read(1) == ""
            stream.close()
            if not exhausted:
                # Prefix consumers intentionally stop before EOF.  Terminate
                # the compressor's process group and close both pipe ends;
                # do not turn the expected SIGPIPE into a decompression error.
                _terminate_process_group(process)
                if stderr_pipe is not None:
                    stderr_pipe.close()
                return
            stderr = stderr_pipe.read() if stderr_pipe is not None else b""
            if stderr_pipe is not None:
                stderr_pipe.close()
            returncode = process.wait()
            if returncode:
                raise RuntimeError(
                    f"zstd decompression failed for {resolved}: "
                    f"{_diagnostic_bytes(stderr)}"
                )
        except BaseException:
            stream.close()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if stderr_pipe is not None:
                stderr_pipe.close()
            raise
    else:
        with resolved.open("r", encoding="utf-8") as stream:
            yield stream


@contextmanager
def open_jsonl_writer(path: Path) -> Iterator[TextIO]:
    """Write JSONL through zstd and atomically publish ``path.jsonl.zst``.

    Pass a logical ``*.jsonl`` path. The uncompressed name is never created;
    the published artifact is ``*.jsonl.zst``. A failed compressor leaves no
    partial published artifact.
    """
    logical = Path(path)
    if not logical.name.endswith(".jsonl"):
        raise ValueError(f"logical JSONL path must end with .jsonl: {logical}")
    final = Path(str(logical) + ".zst")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=final.name + ".tmp.", dir=str(final.parent))
    temporary = Path(temporary_name)
    output_file = os.fdopen(temporary_fd, "wb")
    threads = os.environ.get("SM120_JSONL_ZSTD_THREADS", "1")
    if not threads.isdigit() or int(threads) < 1:
        output_file.close()
        raise ValueError("SM120_JSONL_ZSTD_THREADS must be a positive integer")
    try:
        process = subprocess.Popen(
            [_require_zstd(), "-q", f"-T{threads}", "-1", "-c"],
            stdin=subprocess.PIPE,
            stdout=output_file,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except BaseException:
        output_file.close()
        temporary.unlink(missing_ok=True)
        raise
    assert process.stdin is not None
    stderr_pipe = process.stderr
    stream = io.TextIOWrapper(process.stdin, encoding="utf-8")
    try:
        yield stream
        stream.flush()
        stream.close()
        returncode = process.wait()
        stderr = stderr_pipe.read() if stderr_pipe is not None else b""
        if stderr_pipe is not None:
            stderr_pipe.close()
        output_file.close()
        if returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"zstd compression failed for {logical}: "
                f"{_diagnostic_bytes(stderr)}"
            )
        os.replace(temporary, final)
        # A stale pre-compression sibling is ambiguous and must not survive a
        # successful publication.  Removal occurs after compression so a
        # failed run still leaves the legacy artifact recoverable.
        logical.unlink(missing_ok=True)
    except BaseException:
        try:
            stream.close()
        finally:
            if stderr_pipe is not None:
                stderr_pipe.close()
            output_file.close()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            temporary.unlink(missing_ok=True)
        raise


def jsonl_compression_description() -> str:
    """Return the effective zstd command label used for artifact metadata."""
    threads = os.environ.get("SM120_JSONL_ZSTD_THREADS", "1")
    return f"zstd -T{threads} -1"
