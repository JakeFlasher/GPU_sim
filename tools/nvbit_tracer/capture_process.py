"""Bounded, delegated-cgroup supervision for NVBit capture children."""

from __future__ import annotations

import datetime as dt
import math
import os
import selectors
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

from tools.local_labels import delegated_cgroup

NVBIT_CAPTURE_OUTPUT_LIMIT = "nvbit_capture_output_limit"
_TERMINATION_GRACE_SECONDS = 5.0
_IO_CHUNK_BYTES = 64 * 1024


class CaptureLaunchTimeout(Exception):
    pass


@contextmanager
def _bounded_launch(deadline: float) -> Iterator[None]:
    """Interrupt a stalled POSIX launch without consuming ambient alarm state."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CaptureLaunchTimeout("NVBit capture launch deadline expired")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("NVBit capture launch cannot replace an active alarm timer")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expire(_signal_number: int, _frame: object) -> None:
        raise CaptureLaunchTimeout("NVBit capture launch deadline expired")

    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def bounded_launch(deadline: float) -> Iterator[None]:
    """Bound cgroup creation and Popen by one absolute monotonic deadline."""

    with _bounded_launch(deadline):
        yield


@dataclass(frozen=True)
class NvbitCaptureProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    process_id: int | None = None
    started_at_utc: str | None = None
    containment: delegated_cgroup.CgroupContainmentEvidence | None = None


class NvbitCaptureProcessError(RuntimeError):
    """A typed NVBit child-execution or descendant-cleanup failure."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        process_cleanup_incomplete: bool = False,
        observed_output_bytes_lower_bound: int | None = None,
    ) -> None:
        if type(process_cleanup_incomplete) is not bool:
            raise TypeError("process_cleanup_incomplete must be an exact boolean")
        if (
            observed_output_bytes_lower_bound is not None
            and (
                type(observed_output_bytes_lower_bound) is not int
                or observed_output_bytes_lower_bound < 0
            )
        ):
            raise TypeError(
                "observed_output_bytes_lower_bound must be a nonnegative exact "
                "integer or None"
            )
        self.code = code
        self.detail = detail
        self.process_cleanup_incomplete = process_cleanup_incomplete
        self.observed_output_bytes_lower_bound = observed_output_bytes_lower_bound
        self.stdout = b""
        self.stderr = b""
        self.process_id: int | None = None
        self.started_at_utc: str | None = None
        self.containment: delegated_cgroup.CgroupContainmentEvidence | None = None
        super().__init__(f"{code}: {detail}")


def _validate_request(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    working_directory: str,
    inherited_descriptors: Sequence[int],
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> tuple[tuple[str, ...], dict[str, str], tuple[int, ...]]:
    if not isinstance(command, (tuple, list)) or not command:
        raise ValueError("NVBit capture command must be a nonempty tuple or list")
    if len(command) > 4096 or any(type(value) is not str or not value for value in command):
        raise ValueError("NVBit capture command contains an invalid argument")
    if type(environment) is not dict:
        raise ValueError("NVBit capture environment must be an exact dictionary")
    if len(environment) > 4096 or any(
        type(key) is not str or type(value) is not str or not key
        for key, value in environment.items()
    ):
        raise ValueError("NVBit capture environment contains an invalid entry")
    if type(working_directory) is not str or not working_directory:
        raise ValueError("NVBit capture working directory must be a nonempty string")
    if not isinstance(inherited_descriptors, tuple):
        raise ValueError("NVBit capture inherited descriptors must be a tuple")
    if len(inherited_descriptors) > 4096 or any(
        type(descriptor) is not int or descriptor < 0
        for descriptor in inherited_descriptors
    ):
        raise ValueError("NVBit capture inherited descriptor is invalid")
    if len(set(inherited_descriptors)) != len(inherited_descriptors):
        raise ValueError("NVBit capture inherited descriptors must be unique")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("NVBit capture timeout must be finite and positive")
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("NVBit capture output limit must be a positive integer")
    for value, label in ((timeout_code, "timeout"), (launch_code, "launch")):
        if type(value) is not str or not value or len(value) > 128:
            raise ValueError(f"NVBit capture {label} code is invalid")
    return tuple(command), dict(environment), tuple(inherited_descriptors)


def _copy_failure(
    target: NvbitCaptureProcessError,
    source: NvbitCaptureProcessError,
) -> None:
    target.stdout = source.stdout
    target.stderr = source.stderr
    target.process_id = source.process_id
    target.started_at_utc = source.started_at_utc
    target.containment = source.containment
    target.observed_output_bytes_lower_bound = (
        source.observed_output_bytes_lower_bound
    )
    target.process_cleanup_incomplete = (
        target.process_cleanup_incomplete or source.process_cleanup_incomplete
    )


def _signal_group(process: subprocess.Popen[bytes], signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return


def _process_exited_unreaped(process: subprocess.Popen[bytes]) -> bool:
    try:
        status = os.waitid(
            os.P_PID,
            process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError as exc:
        raise NvbitCaptureProcessError(
            "nvbit_capture_process_lost",
            "NVBit capture child authority was released before cleanup",
        ) from exc
    except OSError as exc:
        raise NvbitCaptureProcessError(
            "nvbit_capture_process_observation_failed",
            f"cannot observe NVBit capture child without reaping it: {exc}",
        ) from exc
    return status is not None


def _wait_for_exit_unreaped(
    process: subprocess.Popen[bytes], timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if _process_exited_unreaped(process):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))


def _stop_and_reap(
    process: subprocess.Popen[bytes],
    containment: delegated_cgroup.DelegatedCgroupV2,
) -> None:
    if process.returncode is not None:
        try:
            containment.kill_all()
        except BaseException as exc:
            raise NvbitCaptureProcessError(
                "nvbit_capture_cleanup_failed",
                f"NVBit capture descendant termination failed: {exc}",
                process_cleanup_incomplete=True,
            ) from exc
        return

    termination_error: BaseException | None = None
    try:
        _signal_group(process, signal.SIGTERM)
        if not _wait_for_exit_unreaped(process, _TERMINATION_GRACE_SECONDS):
            try:
                containment.kill_all()
            except BaseException as exc:
                termination_error = exc
            else:
                if not _wait_for_exit_unreaped(
                    process, _TERMINATION_GRACE_SECONDS
                ):
                    termination_error = NvbitCaptureProcessError(
                        "nvbit_capture_cleanup_failed",
                        "NVBit capture child did not exit after cgroup termination",
                        process_cleanup_incomplete=True,
                    )
        else:
            # The leader remains unreaped PID/PGID authority while cgroup.kill
            # removes descendants that escaped its session or process group.
            try:
                containment.kill_all()
            except BaseException as exc:
                termination_error = exc
    except BaseException as exc:
        termination_error = exc

    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except BaseException as reap_error:
        failure = NvbitCaptureProcessError(
            "nvbit_capture_cleanup_failed",
            f"cannot reap NVBit capture child: {reap_error}",
            process_cleanup_incomplete=True,
        )
        if termination_error is not None:
            try:
                failure.add_note(
                    f"active NVBit capture termination error: {termination_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise failure from reap_error

    if termination_error is not None:
        if isinstance(termination_error, NvbitCaptureProcessError):
            raise termination_error
        raise NvbitCaptureProcessError(
            "nvbit_capture_cleanup_failed",
            f"NVBit capture descendant termination failed: {termination_error}",
            process_cleanup_incomplete=True,
        ) from termination_error


def _close_monitor_resources(
    selector: selectors.BaseSelector | None,
    process: subprocess.Popen[bytes],
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    if selector is not None:
        try:
            selector.close()
        except BaseException as exc:
            errors.append(exc)
    for pipe in (process.stdout, process.stderr):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except BaseException as exc:
                errors.append(exc)
    return tuple(errors)


def _new_output_buffers() -> dict[str, bytearray]:
    return {"stdout": bytearray(), "stderr": bytearray()}


def _copy_stream_prefix(buffer: bytearray, length: int) -> bytes:
    if length == len(buffer):
        return bytes(buffer)
    return bytes(memoryview(buffer)[:length])


def _materialize_stream(
    buffer: bytearray,
) -> tuple[bytes, MemoryError | None]:
    first_error: MemoryError | None = None
    candidate_length = len(buffer)
    retry_full = True
    while candidate_length > 0:
        try:
            return _copy_stream_prefix(buffer, candidate_length), first_error
        except MemoryError as exc:
            if first_error is None:
                first_error = exc
            if retry_full:
                retry_full = False
            else:
                candidate_length //= 2
    return b"", first_error


def _collect_bounded(
    process: subprocess.Popen[bytes],
    *,
    containment: delegated_cgroup.DelegatedCgroupV2,
    started_at_utc: str,
    timeout_seconds: float,
    deadline_monotonic: float | None = None,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> NvbitCaptureProcessResult:
    selector: selectors.BaseSelector | None = None
    buffers: dict[str, bytearray] | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    close_error: BaseException | None = None
    returncode: int | None = None
    process_reaped = False
    try:
        buffers = _new_output_buffers()
        total = 0
        deadline = (
            time.monotonic() + timeout_seconds
            if deadline_monotonic is None
            else deadline_monotonic
        )
        if process.stdout is None or process.stderr is None:
            raise NvbitCaptureProcessError(
                launch_code, "NVBit capture child pipes are unavailable",
            )
        try:
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        except (OSError, ValueError, KeyError) as exc:
            raise NvbitCaptureProcessError(
                launch_code,
                f"cannot initialize NVBit capture output monitor: {exc}",
            ) from exc
        leader_exit_observed = False
        while selector.get_map():
            if not leader_exit_observed and _process_exited_unreaped(process):
                containment.kill_all()
                leader_exit_observed = True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NvbitCaptureProcessError(
                    timeout_code,
                    f"NVBit capture child exceeded {timeout_seconds} seconds",
                )
            try:
                events = selector.select(timeout=min(0.1, remaining))
            except OSError as exc:
                raise NvbitCaptureProcessError(
                    launch_code,
                    f"cannot monitor NVBit capture output: {exc}",
                ) from exc
            for key, _mask in events:
                try:
                    chunk = os.read(key.fd, _IO_CHUNK_BYTES)
                except OSError as exc:
                    raise NvbitCaptureProcessError(
                        launch_code,
                        f"cannot read NVBit capture output: {exc}",
                    ) from exc
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                remaining_capacity = max_output_bytes - total
                if remaining_capacity > 0:
                    retained = (
                        chunk
                        if len(chunk) <= remaining_capacity
                        else chunk[:remaining_capacity]
                    )
                    buffers[key.data].extend(retained)
                total += len(chunk)
                if total > max_output_bytes:
                    raise NvbitCaptureProcessError(
                        NVBIT_CAPTURE_OUTPUT_LIMIT,
                        "NVBit capture output exceeds "
                        f"{max_output_bytes} bytes; observed at least {total} bytes",
                        observed_output_bytes_lower_bound=total,
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NvbitCaptureProcessError(
                timeout_code,
                f"NVBit capture child exceeded {timeout_seconds} seconds",
            )
        if not leader_exit_observed:
            if not _wait_for_exit_unreaped(process, remaining):
                raise NvbitCaptureProcessError(
                    timeout_code,
                    f"NVBit capture child exceeded {timeout_seconds} seconds",
                )
            containment.kill_all()
        returncode = process.wait()
        process_reaped = True
    except BaseException as exc:
        primary_error = exc
    finally:
        if not process_reaped:
            try:
                _stop_and_reap(process, containment)
                process_reaped = True
            except BaseException as exc:
                cleanup_error = exc
                try:
                    _stop_and_reap(process, containment)
                    process_reaped = True
                except BaseException as retry_error:
                    try:
                        cleanup_error.add_note(
                            "NVBit capture process cleanup retry failed: "
                            f"{retry_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
                else:
                    cleanup_error = None
        try:
            close_errors = _close_monitor_resources(selector, process)
        except BaseException as exc:
            close_error = exc
        else:
            if close_errors:
                try:
                    detail = "; ".join(str(error) for error in close_errors)
                    close_error = NvbitCaptureProcessError(
                        launch_code,
                        "cannot close NVBit capture process resources: " + detail,
                    )
                    close_error.__cause__ = close_errors[0]
                except BaseException as exc:
                    close_error = exc

    stdout = b""
    stderr = b""
    materialization_error: MemoryError | None = None
    if buffers is not None:
        stdout, stdout_error = _materialize_stream(buffers["stdout"])
        stderr, stderr_error = _materialize_stream(buffers["stderr"])
        materialization_error = stdout_error or stderr_error

    if cleanup_error is not None:
        failure = NvbitCaptureProcessError(
            launch_code,
            f"NVBit capture process cleanup failed: {cleanup_error}",
            process_cleanup_incomplete=True,
        )
        if isinstance(primary_error, NvbitCaptureProcessError):
            _copy_failure(failure, primary_error)
        failure.stdout = stdout
        failure.stderr = stderr
        failure.process_id = process.pid
        failure.started_at_utc = started_at_utc
        if primary_error is not None:
            try:
                failure.add_note(f"active NVBit capture error: {primary_error}")
            except (AttributeError, TypeError):
                pass
        raise failure from cleanup_error

    if materialization_error is not None:
        failure = NvbitCaptureProcessError(
            launch_code,
            "NVBit capture output materialization exhausted memory",
        )
        if isinstance(primary_error, NvbitCaptureProcessError):
            _copy_failure(failure, primary_error)
        failure.stdout = stdout
        failure.stderr = stderr
        failure.process_id = process.pid
        failure.started_at_utc = started_at_utc
        if primary_error is not None:
            try:
                failure.add_note(f"active NVBit capture error: {primary_error}")
            except (AttributeError, TypeError):
                pass
        if close_error is not None:
            try:
                failure.add_note(f"process resource closure failed: {close_error}")
            except (AttributeError, TypeError):
                pass
        raise failure from materialization_error

    if close_error is not None:
        if primary_error is not None:
            try:
                primary_error.add_note(
                    f"NVBit process resource closure failed: {close_error}"
                )
            except (AttributeError, TypeError):
                pass
        else:
            failure = (
                close_error
                if isinstance(close_error, NvbitCaptureProcessError)
                else NvbitCaptureProcessError(
                    launch_code,
                    f"cannot close NVBit capture process resources: {close_error}",
                )
            )
            failure.stdout = stdout
            failure.stderr = stderr
            failure.process_id = process.pid
            failure.started_at_utc = started_at_utc
            raise failure from close_error

    if primary_error is not None:
        if isinstance(primary_error, NvbitCaptureProcessError):
            primary_error.stdout = stdout
            primary_error.stderr = stderr
            primary_error.process_id = process.pid
            primary_error.started_at_utc = started_at_utc
            raise primary_error
        if isinstance(primary_error, MemoryError):
            failure = NvbitCaptureProcessError(
                launch_code,
                "NVBit capture output collection exhausted memory",
            )
            failure.stdout = stdout
            failure.stderr = stderr
            failure.process_id = process.pid
            failure.started_at_utc = started_at_utc
            raise failure from primary_error
        raise primary_error

    assert returncode is not None
    return NvbitCaptureProcessResult(
        returncode,
        stdout,
        stderr,
        process_id=process.pid,
        started_at_utc=started_at_utc,
    )


def execute_capture_process(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    working_directory: str,
    inherited_descriptors: tuple[int, ...],
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> NvbitCaptureProcessResult:
    """Execute one NVBit capture child with mandatory cgroup-v2 containment."""

    command_tuple, environment_dict, descriptor_tuple = _validate_request(
        command,
        environment=environment,
        working_directory=working_directory,
        inherited_descriptors=inherited_descriptors,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        timeout_code=timeout_code,
        launch_code=launch_code,
    )
    timeout_seconds_float = float(timeout_seconds)
    deadline_monotonic = time.monotonic() + timeout_seconds_float
    started_at_utc = (
        dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    containment: delegated_cgroup.DelegatedCgroupV2 | None = None
    process: subprocess.Popen[bytes] | None = None
    popen_arguments = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": environment_dict,
        "cwd": working_directory,
        "pass_fds": descriptor_tuple,
        "start_new_session": True,
    }
    try:
        with bounded_launch(deadline_monotonic):
            containment = delegated_cgroup.DelegatedCgroupV2.create()
            process = containment.popen(command_tuple, **popen_arguments)
    except BaseException as exc:
        cleanup_incomplete = False
        cleanup_error: BaseException | None = None
        if containment is not None:
            try:
                containment.close_after_error()
            except BaseException as active_cleanup_error:
                cleanup_incomplete = True
                cleanup_error = active_cleanup_error
                try:
                    exc.add_note(
                        "NVBit delegated cgroup launch cleanup failed: "
                        f"{active_cleanup_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        if not isinstance(exc, Exception):
            if cleanup_error is not None:
                failure = NvbitCaptureProcessError(
                    launch_code,
                    f"NVBit delegated cgroup launch cleanup failed: {cleanup_error}",
                    process_cleanup_incomplete=True,
                )
                try:
                    failure.add_note(f"active NVBit capture launch error: {exc}")
                except (AttributeError, TypeError):
                    pass
                raise failure from cleanup_error
            raise
        if isinstance(exc, CaptureLaunchTimeout):
            failure = NvbitCaptureProcessError(
                timeout_code,
                f"NVBit capture launch exceeded {timeout_seconds_float} seconds",
                process_cleanup_incomplete=cleanup_incomplete,
            )
        elif (
            containment is None
            and isinstance(exc, delegated_cgroup.CgroupContainmentError)
        ):
            failure = NvbitCaptureProcessError(
                launch_code,
                f"cannot establish NVBit delegated cgroup containment: {exc}",
                process_cleanup_incomplete=cleanup_incomplete,
            )
        else:
            failure = NvbitCaptureProcessError(
                launch_code,
                f"cannot launch contained NVBit capture process: {exc}",
                process_cleanup_incomplete=cleanup_incomplete,
            )
        if cleanup_error is not None:
            raise failure from cleanup_error
        raise failure from exc

    assert containment is not None
    assert process is not None

    try:
        result = _collect_bounded(
            process,
            containment=containment,
            started_at_utc=started_at_utc,
            timeout_seconds=timeout_seconds_float,
            deadline_monotonic=deadline_monotonic,
            max_output_bytes=max_output_bytes,
            timeout_code=timeout_code,
            launch_code=launch_code,
        )
    except BaseException as active_error:
        if isinstance(active_error, NvbitCaptureProcessError):
            process_error = active_error
        else:
            process_error = NvbitCaptureProcessError(
                launch_code,
                f"contained NVBit capture execution failed: {active_error}",
            )
            process_error.process_id = process.pid
            process_error.started_at_utc = started_at_utc
        try:
            process_error.containment = containment.finish()
        except BaseException as cleanup_error:
            failure = NvbitCaptureProcessError(
                launch_code,
                f"NVBit delegated cgroup cleanup failed: {cleanup_error}",
                process_cleanup_incomplete=True,
            )
            _copy_failure(failure, process_error)
            try:
                failure.add_note(f"active NVBit capture error: {active_error}")
            except (AttributeError, TypeError):
                pass
            raise failure from cleanup_error
        if not isinstance(active_error, Exception):
            raise
        if not isinstance(active_error, NvbitCaptureProcessError):
            raise process_error from active_error
        if active_error.code in {timeout_code, NVBIT_CAPTURE_OUTPUT_LIMIT}:
            raise
        if active_error.code == launch_code:
            raise
        failure = NvbitCaptureProcessError(launch_code, active_error.detail)
        _copy_failure(failure, active_error)
        raise failure from active_error

    try:
        containment_evidence = containment.finish()
    except BaseException as cleanup_error:
        failure = NvbitCaptureProcessError(
            launch_code,
            f"NVBit delegated cgroup cleanup failed: {cleanup_error}",
            process_cleanup_incomplete=True,
        )
        failure.stdout = result.stdout
        failure.stderr = result.stderr
        failure.process_id = result.process_id
        failure.started_at_utc = result.started_at_utc
        raise failure from cleanup_error
    return NvbitCaptureProcessResult(
        result.returncode,
        result.stdout,
        result.stderr,
        process_id=result.process_id,
        started_at_utc=result.started_at_utc,
        containment=containment_evidence,
    )
