"""A/B overhead orchestration + per-launch phase-ledger emission for
the non-root measurement NVBit phase-injection contract.

This module wires the GPU-side phase-injection tracer extension
(consolidated under `tools/nvbit_tracer/tracer_tool/`) to the
host-side aggregator + emitter (`phase_cycle_ledger_aggregator.py`)
and the consumer's admission rules (`phase_cycle_ledger_consumer.py`).
The producer-side test surface uses synthetic subprocess outputs;
the production-side surface drives `run_suite.py::handle_trace` with
the `--ab-overhead` flag.

Public API:

  * `run_ab_overhead_pair(...)` — invokes the retained kernel executable on
    every no-injection / with-injection side of the bounded repeat matrix;
    returns the schema-bound `ab_overhead_record` fields plus a private retained
    source snapshot that `run_suite.py` strips before ledger construction.
  * `emit_per_launch_phase_ledgers(...)` — scans a traces folder
    for `phase_records_<repeat>_<launch>.bin` files (emitted by
    the tracer extension's per-launch flush path), decodes each
    via the aggregator's struct format, builds a schema-valid
    `phase_cycle_ledger_<repeat>_<launch>.json` sidecar, runs the
    consumer's `admit_phase_cycle_ledger`, and reports the per-
    ledger admission verdict.

Both helpers are host-independent in the sense that the tracer
binary + GPU launch behavior is encapsulated behind subprocess
calls; tests can monkeypatch the subprocess to feed synthetic
phase-record bytes without a real Blackwell run.
"""

from __future__ import annotations

import errno
import hashlib
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import phase_cycle_ledger_aggregator as _agg  # type: ignore[import-not-found]
import phase_cycle_ledger_consumer as _consumer  # type: ignore[import-not-found]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.local_labels.immutable_artifact import (  # noqa: E402
    DirectoryGenerationFile,
    ExclusiveDirectoryStage,
    ImmutableArtifactError,
    ImmutableArtifactResourceError,
    ImmutableDirectoryGeneration,
    PinnedDirectoryReader,
    PinnedPathSetMonitor,
    PinnedTreeSnapshot,
    RetainedExecutable,
    create_exclusive_directory_stage,
    inspect_immutable_directory_generation,
    publish_regular_file,
)
from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    TracerRuntimeError,
    tracer_runtime_env,
)


PHASE_LEDGER_GENERATION_NAMESPACE = "phase_cycle_ledger_generations"
PHASE_REPEAT_INDEX_ENV = "SIMBENCH_NVBIT_PHASE_REPEAT_INDEX"

# Task 277 implementation-owned work domains.  These caps are deliberately
# below the surrounding immutable-tree ceilings so the exact repeat x launch
# matrix, arguments, decoded records, and child diagnostics are bounded before
# their corresponding allocations or subprocess side effects.
PHASE_AB_MAX_REPEAT_COUNT = 64
PHASE_AB_MAX_LAUNCH_COUNT = 1024
PHASE_AB_MAX_MATRIX_CELLS = 2048
PHASE_AB_MAX_ARGUMENT_COUNT = 64
PHASE_AB_MAX_ARGUMENT_BYTES = 64 * 1024
PHASE_AB_MAX_ENVIRONMENT_ENTRIES = 4096
PHASE_AB_MAX_ENVIRONMENT_BYTES = 4 * 1024 * 1024
PHASE_AB_MAX_CAPTURE_BYTES = 1024 * 1024
PHASE_AB_MAX_DIAGNOSTIC_BYTES = 64 * 1024
PHASE_RECORD_MAX_RECORDS = 262_144
PHASE_RECORD_MAX_BYTES = PHASE_RECORD_MAX_RECORDS * _agg.PHASE_CLOCK_RECORD_SIZE
PHASE_TIMEOUT_MAX_S = 3600
PHASE_ADMISSION_THRESHOLD_MAX_PCT = 1000.0

_PHASE_AB_IO_CHUNK_BYTES = 64 * 1024
_PHASE_SOURCE_MAX_FILES = 4096
_PHASE_SOURCE_MAX_DIRECTORIES = 4096
_PHASE_SOURCE_MAX_BYTES = 512 * 1024 * 1024
_PHASE_SOURCE_MAX_RELATIVE_PATH_BYTES = 32 * 1024 * 1024
_PHASE_LEDGER_MAX_FILES = PHASE_AB_MAX_MATRIX_CELLS
_PHASE_LEDGER_MAX_RELATIVE_PATH_BYTES = 4 * 1024 * 1024


PHASE_RECORDS_FILENAME_RE = re.compile(
    r"^phase_records_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)\.bin$"
)
_PARTIAL_PHASE_RECORDS_FILENAME_RE = re.compile(
    r"^\.partial-phase-records_(0|[1-9][0-9]*)_"
    r"(0|[1-9][0-9]*)\.bin$"
)


def _bounded_value_label(value: object) -> str:
    """Describe rejected scalar input without coercion or unbounded repr work."""
    if value is None:
        return "None"
    if type(value) is bool:
        return "True" if value else "False"
    if type(value) is int:
        if value.bit_length() > 256:
            return f"<int bit_length={value.bit_length()}>"
        return repr(value)
    if type(value) is float:
        return repr(value)
    if type(value) is str:
        prefix = value[:128]
        suffix = "..." if len(value) > len(prefix) else ""
        return repr(prefix) + suffix
    return f"<{type(value).__name__}>"


@dataclass(frozen=True)
class AbOverheadInputs:
    """Inputs needed for a paired no-injection / with-injection
    overhead run."""

    executable_authority: RetainedExecutable
    args: list[str]
    tracer_so_path: Path
    tracer_descriptor: int
    traces_folder: Path
    expected_launch_indices: tuple[int, ...]
    traces_descriptor: int | None = None
    traces_reader: PinnedDirectoryReader | None = None
    repeat_count: int = 3
    base_env: dict[str, str] | None = None


class PhaseRecordSourceNotFresh(Exception):
    """The A/B attempt did not exclusively create its phase-record set."""

    TOKEN = "phase_inject_record_source_not_fresh"

    def __init__(self, detail: str):
        super().__init__(f"{self.TOKEN}: {detail}")
        self.detail = detail


class PhaseRecordEvidenceInvalid(ImmutableArtifactError):
    """A phase-record pathname or file generation is not canonical evidence."""

    TOKEN = "phase_inject_record_evidence_invalid"

    def __init__(self, detail: str):
        super().__init__(f"{self.TOKEN}: {detail}")
        self.detail = detail


class PhaseRecordPayloadInvalid(Exception):
    """A canonical phase-record file is empty or not completely decodable."""

    TOKEN = "phase_inject_record_payload_invalid"

    def __init__(self, relative_path: str, detail: str):
        super().__init__(f"{self.TOKEN}: {relative_path}: {detail}")
        self.relative_path = relative_path
        self.detail = detail


class PhaseInjectLaunchAuthorityInvalid(Exception):
    """The requested launch vector is not exact contiguous admitted authority."""

    TOKEN = "phase_inject_launch_authority_invalid"

    def __init__(self, detail: str):
        super().__init__(f"{self.TOKEN}: {detail}")
        self.detail = detail


class PhaseInjectAdmissionThresholdInvalid(Exception):
    """The A/B admission threshold is outside its exact numeric domain."""

    TOKEN = "phase_inject_admission_threshold_invalid"

    def __init__(self, value: object):
        super().__init__(
            f"{self.TOKEN}: admission_threshold_pct must be an exact int or "
            f"float in [0, {PHASE_ADMISSION_THRESHOLD_MAX_PCT}] and finite; "
            f"got {_bounded_value_label(value)}"
        )
        self.value = value


class PhaseInjectTimeoutInvalid(Exception):
    """The per-run timeout is outside its exact bounded domain."""

    TOKEN = "phase_inject_timeout_invalid"

    def __init__(self, value: object):
        super().__init__(
            f"{self.TOKEN}: timeout_per_run_s must be None or an exact integer "
            f"in [1, {PHASE_TIMEOUT_MAX_S}]; "
            f"got {_bounded_value_label(value)}"
        )
        self.value = value


class PhaseInjectResourceLimitExceeded(Exception):
    """Finite phase orchestration work or retained output exceeded its cap."""

    TOKEN = "phase_inject_resource_limit_exceeded"

    def __init__(self, *, resource: str, limit: int, observed: int):
        super().__init__(
            f"{self.TOKEN}: {resource} observed={observed} exceeds limit={limit}"
        )
        self.resource = resource
        self.limit = limit
        self.observed = observed


class PhaseInjectExecutableAuthorityInvalid(Exception):
    """The admitted executable capability changed or became unavailable."""

    TOKEN = "phase_inject_executable_authority_invalid"

    def __init__(self, detail: str):
        super().__init__(f"{self.TOKEN}: {detail}")
        self.detail = detail


class PhaseInjectAbOutputLimitExceeded(PhaseInjectResourceLimitExceeded):
    """The supervised child emitted more output than the retained byte cap."""

    TOKEN = "phase_inject_ab_output_limit_exceeded"

    def __init__(self, *, observed: int, stderr: str):
        PhaseInjectResourceLimitExceeded.__init__(
            self,
            resource="A/B subprocess combined stdout/stderr bytes",
            limit=PHASE_AB_MAX_CAPTURE_BYTES,
            observed=observed,
        )
        self.args = (
            f"{self.TOKEN}: combined stdout/stderr observed={observed} exceeds "
            f"limit={PHASE_AB_MAX_CAPTURE_BYTES}; stderr={stderr!r}",
        )
        self.stderr = stderr


@contextmanager
def _preserve_primary_cleanup(context):
    """Never let context cleanup replace an already-selected typed failure."""
    resource = context.__enter__()
    try:
        yield resource
    except BaseException as primary:
        try:
            context.__exit__(type(primary), primary, primary.__traceback__)
        except BaseException as cleanup:
            try:
                primary.add_note(
                    "phase transaction cleanup also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            except (AttributeError, TypeError):
                pass
        raise
    else:
        context.__exit__(None, None, None)


def _phase_record_files(
    snapshot: PinnedTreeSnapshot,
) -> dict[str, tuple[int, int, int, int, int, int, int]]:
    files: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    for relative_path, _metadata in snapshot.directories:
        if relative_path == ".":
            continue
        leaf = relative_path.rsplit("/", 1)[-1]
        if (
            "phase_records_" in leaf
            or "partial-phase-records_" in leaf
        ):
            raise PhaseRecordEvidenceInvalid(
                f"phase-record authority is a directory: {relative_path!r}"
            )
    for relative_path, metadata in snapshot.files:
        leaf = relative_path.rsplit("/", 1)[-1]
        match = PHASE_RECORDS_FILENAME_RE.fullmatch(relative_path)
        if match is not None:
            if not stat.S_ISREG(metadata[2]) or metadata[3] != 1:
                raise PhaseRecordEvidenceInvalid(
                    "canonical phase-record authority is not a single-link "
                    f"regular file: {relative_path!r}"
                )
            if metadata[4] <= 0:
                raise PhaseRecordPayloadInvalid(
                    relative_path, "canonical payload is empty"
                )
            if metadata[4] > PHASE_RECORD_MAX_BYTES:
                raise PhaseInjectResourceLimitExceeded(
                    resource=f"phase-record payload bytes for {relative_path}",
                    limit=PHASE_RECORD_MAX_BYTES,
                    observed=metadata[4],
                )
            files[relative_path] = metadata
            continue
        if _PARTIAL_PHASE_RECORDS_FILENAME_RE.fullmatch(relative_path):
            raise PhaseRecordEvidenceInvalid(
                "retained timeout-partial phase evidence is noncanonical and "
                f"cannot be admitted: {relative_path!r}"
            )
        if (
            "phase_records_" in leaf
            or "partial-phase-records_" in leaf
        ):
            raise PhaseRecordEvidenceInvalid(
                f"noncanonical phase-record filename {relative_path!r}"
            )
    return files


def _validated_repeat_and_launch_domain(
    repeat_count: object,
    expected_launch_indices: object,
) -> tuple[int, tuple[int, ...]]:
    if type(repeat_count) is not int or repeat_count <= 0:
        raise PhaseInjectAbRepeatCountInvalid(repeat_count)
    if repeat_count > PHASE_AB_MAX_REPEAT_COUNT:
        raise PhaseInjectResourceLimitExceeded(
            resource="A/B repeat count",
            limit=PHASE_AB_MAX_REPEAT_COUNT,
            observed=repeat_count,
        )
    if type(expected_launch_indices) is not tuple:
        raise PhaseInjectLaunchAuthorityInvalid(
            "expected_launch_indices must be an exact tuple"
        )
    launch_count = len(expected_launch_indices)
    if launch_count == 0:
        raise PhaseInjectLaunchAuthorityInvalid(
            "expected_launch_indices must be nonempty"
        )
    if launch_count > PHASE_AB_MAX_LAUNCH_COUNT:
        raise PhaseInjectResourceLimitExceeded(
            resource="admitted phase launch count",
            limit=PHASE_AB_MAX_LAUNCH_COUNT,
            observed=launch_count,
        )
    if any(type(index) is not int for index in expected_launch_indices):
        raise PhaseInjectLaunchAuthorityInvalid(
            "every launch index must be an exact integer"
        )
    canonical = tuple(range(launch_count))
    if expected_launch_indices != canonical:
        raise PhaseInjectLaunchAuthorityInvalid(
            "launch authority must be canonical contiguous indices beginning "
            f"at zero; got {expected_launch_indices!r}"
        )
    matrix_cells = repeat_count * launch_count
    if matrix_cells > PHASE_AB_MAX_MATRIX_CELLS:
        raise PhaseInjectResourceLimitExceeded(
            resource="phase repeat-by-launch matrix cells",
            limit=PHASE_AB_MAX_MATRIX_CELLS,
            observed=matrix_cells,
        )
    return repeat_count, expected_launch_indices


def _validated_arguments(args: object) -> tuple[str, ...]:
    if type(args) is not list:
        raise ImmutableArtifactError(
            "phase A/B arguments must be an exact list"
        )
    if len(args) > PHASE_AB_MAX_ARGUMENT_COUNT:
        raise PhaseInjectResourceLimitExceeded(
            resource="A/B argument count",
            limit=PHASE_AB_MAX_ARGUMENT_COUNT,
            observed=len(args),
        )
    retained: list[str] = []
    total_bytes = 0
    for index, argument in enumerate(args):
        if type(argument) is not str or "\x00" in argument:
            raise ImmutableArtifactError(
                f"phase A/B argument {index} must be an exact NUL-free string"
            )
        minimum_total_bytes = total_bytes + len(argument) + 1
        if minimum_total_bytes > PHASE_AB_MAX_ARGUMENT_BYTES:
            raise PhaseInjectResourceLimitExceeded(
                resource="A/B aggregate argument bytes",
                limit=PHASE_AB_MAX_ARGUMENT_BYTES,
                observed=minimum_total_bytes,
            )
        try:
            encoded = argument.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ImmutableArtifactError(
                f"phase A/B argument {index} is not strict UTF-8"
            ) from exc
        total_bytes += len(encoded) + 1
        if total_bytes > PHASE_AB_MAX_ARGUMENT_BYTES:
            raise PhaseInjectResourceLimitExceeded(
                resource="A/B aggregate argument bytes",
                limit=PHASE_AB_MAX_ARGUMENT_BYTES,
                observed=total_bytes,
            )
        retained.append(argument)
    return tuple(retained)


def _validated_admission_threshold(value: object) -> float:
    if type(value) is int:
        if value < 0 or value > PHASE_ADMISSION_THRESHOLD_MAX_PCT:
            raise PhaseInjectAdmissionThresholdInvalid(value)
        return float(value)
    if type(value) is float:
        if (
            not math.isfinite(value)
            or value < 0
            or value > PHASE_ADMISSION_THRESHOLD_MAX_PCT
        ):
            raise PhaseInjectAdmissionThresholdInvalid(value)
        return value
    raise PhaseInjectAdmissionThresholdInvalid(value)


def _validated_timeout(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not (1 <= value <= PHASE_TIMEOUT_MAX_S):
        raise PhaseInjectTimeoutInvalid(value)
    return value


def _validate_environment(environment: object) -> None:
    if type(environment) is not dict:
        raise ImmutableArtifactError(
            "phase A/B environment must be an exact string dictionary"
        )
    if len(environment) > PHASE_AB_MAX_ENVIRONMENT_ENTRIES:
        raise PhaseInjectResourceLimitExceeded(
            resource="A/B environment entry count",
            limit=PHASE_AB_MAX_ENVIRONMENT_ENTRIES,
            observed=len(environment),
        )
    total_bytes = 0
    for key, value in environment.items():
        if (
            type(key) is not str
            or type(value) is not str
            or not key
            or "=" in key
            or "\x00" in key
            or "\x00" in value
        ):
            raise ImmutableArtifactError(
                "phase A/B environment keys and values must be exact "
                "NUL-free strings with nonempty '='-free keys"
            )
        minimum_total_bytes = total_bytes + len(key) + len(value) + 2
        if minimum_total_bytes > PHASE_AB_MAX_ENVIRONMENT_BYTES:
            raise PhaseInjectResourceLimitExceeded(
                resource="A/B aggregate environment bytes",
                limit=PHASE_AB_MAX_ENVIRONMENT_BYTES,
                observed=minimum_total_bytes,
            )
        try:
            total_bytes += len(key.encode("utf-8", errors="strict"))
            total_bytes += len(value.encode("utf-8", errors="strict")) + 2
        except UnicodeEncodeError as exc:
            raise ImmutableArtifactError(
                "phase A/B environment is not strict UTF-8"
            ) from exc
        if total_bytes > PHASE_AB_MAX_ENVIRONMENT_BYTES:
            raise PhaseInjectResourceLimitExceeded(
                resource="A/B aggregate environment bytes",
                limit=PHASE_AB_MAX_ENVIRONMENT_BYTES,
                observed=total_bytes,
            )


def _bounded_diagnostic_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        raw = value[: PHASE_AB_MAX_DIAGNOSTIC_BYTES + 1]
    elif type(value) is str:
        if len(value) > PHASE_AB_MAX_DIAGNOSTIC_BYTES:
            value = value[: PHASE_AB_MAX_DIAGNOSTIC_BYTES + 1]
        raw = value.encode("utf-8", errors="replace")
    else:
        raw = (
            f"<unsupported diagnostic payload type {type(value).__name__}>"
        ).encode("ascii", errors="replace")
    decoded = raw.decode("utf-8", errors="replace")
    encoded = decoded.encode("utf-8")
    if len(encoded) <= PHASE_AB_MAX_DIAGNOSTIC_BYTES:
        return decoded
    suffix = "\n...[truncated]"
    prefix_limit = PHASE_AB_MAX_DIAGNOSTIC_BYTES - len(suffix.encode("ascii"))
    prefix = encoded[:prefix_limit].decode("utf-8", errors="ignore")
    return prefix + suffix


def _captured_output_bytes(value: object, *, label: str) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if type(value) is str:
        if len(value) > PHASE_AB_MAX_CAPTURE_BYTES:
            raise PhaseInjectAbOutputLimitExceeded(
                observed=len(value),
                stderr=_bounded_diagnostic_text(value if label == "stderr" else ""),
            )
        return value.encode("utf-8", errors="replace")
    raise ImmutableArtifactError(
        f"phase A/B subprocess {label} must be bytes, string, or None"
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise
    process.wait()


def _bounded_subprocess_runner(
    command: list[str],
    *,
    capture_output: bool,
    text: bool,
    env: dict[str, str],
    pass_fds: tuple[int, ...],
    timeout: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run one child with bounded combined output and process-group cleanup."""
    if capture_output is not True or text is not False:
        raise ImmutableArtifactError(
            "bounded phase subprocess runner requires binary captured output"
        )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        pass_fds=pass_fds,
        start_new_session=True,
    )
    selector: selectors.BaseSelector | None = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total_bytes = 0
    deadline = None if timeout is None else time.monotonic() + timeout
    primary: BaseException | None = None
    try:
        if process.stdout is None or process.stderr is None:
            raise ImmutableArtifactError(
                "bounded phase subprocess pipes are unavailable"
            )
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            if deadline is None:
                wait_s = 0.1
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        command,
                        timeout,
                        output=bytes(buffers["stdout"]),
                        stderr=bytes(buffers["stderr"]),
                    )
                wait_s = min(0.1, remaining)
            for key, _mask in selector.select(timeout=wait_s):
                chunk = os.read(key.fd, _PHASE_AB_IO_CHUNK_BYTES)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                capacity = PHASE_AB_MAX_CAPTURE_BYTES - total_bytes
                buffers[key.data].extend(chunk[: max(0, capacity)])
                total_bytes += len(chunk)
                if total_bytes > PHASE_AB_MAX_CAPTURE_BYTES:
                    raise PhaseInjectAbOutputLimitExceeded(
                        observed=total_bytes,
                        stderr=_bounded_diagnostic_text(buffers["stderr"]),
                    )
        if deadline is None:
            returncode = process.wait()
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=bytes(buffers["stdout"]),
                    stderr=bytes(buffers["stderr"]),
                )
            returncode = process.wait(timeout=remaining)
        return subprocess.CompletedProcess(
            command,
            returncode,
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
        )
    except BaseException as exc:
        primary = exc
        try:
            _kill_process_group(process)
        except BaseException as cleanup:
            try:
                exc.add_note(
                    "phase subprocess cleanup also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            except (AttributeError, TypeError):
                pass
        raise
    finally:
        close_errors: list[BaseException] = []
        if primary is None:
            try:
                _kill_process_group(process)
            except BaseException as exc:
                close_errors.append(exc)
        if selector is not None:
            try:
                selector.close()
            except BaseException as exc:
                close_errors.append(exc)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None and not pipe.closed:
                try:
                    pipe.close()
                except BaseException as exc:
                    close_errors.append(exc)
        if close_errors:
            detail = "; ".join(
                f"{type(exc).__name__}: {exc}" for exc in close_errors
            )
            if primary is not None:
                try:
                    primary.add_note(
                        f"phase subprocess resource cleanup failed: {detail}"
                    )
                except (AttributeError, TypeError):
                    pass
            else:
                raise ImmutableArtifactError(
                    f"phase subprocess resource cleanup failed: {detail}"
                ) from close_errors[0]


def _median_ns(samples: list[int]) -> int:
    s = sorted(samples)
    return int(s[len(s) // 2])


_DEFAULT_TIMEOUT_PER_RUN_S = 600


def run_ab_overhead_pair(
    inputs: AbOverheadInputs,
    *,
    admission_threshold_pct: float,
    subprocess_runner: Callable[..., Any] | None = None,
    timer_ns=time.perf_counter_ns,
    timeout_per_run_s: int | None = _DEFAULT_TIMEOUT_PER_RUN_S,
) -> dict[str, Any]:
    """Run the executable twice (no-injection NVBit + with-injection
    NVBit) `repeat_count` times each. Compute the median wall-clock
    times for each side and the overhead percentage. Returns the schema-bound
    overhead fields plus `_phase_source_snapshot`, the exact retained source tree
    after the final corresponding subprocess.

    The `subprocess_runner` and `timer_ns` parameters allow tests to
    monkeypatch behavior without a real GPU run; production callers
    use the defaults.

    Each subprocess invocation is bounded by `timeout_per_run_s` (default
    600 s) so the A/B orchestration cannot hang indefinitely. A timeout
    on either side surfaces as `PhaseInjectAbRunTimeout` with the typed
    reason `phase_inject_ab_run_timeout` (distinct from the return-code-
    driven `PhaseInjectAbRunFailed` which carries `phase_inject_ab_run_failed`).
    Both typed exceptions are caught by the `run_suite.handle_trace`
    boundary and translated to `EXIT_BUILD_FAILED` with a stderr
    diagnostic. Pass `timeout_per_run_s=None` to opt out (production
    runs should keep the bound; the opt-out path exists for tests and
    for explicit-debug operator runs)."""
    repeat_count, expected_launch_indices = _validated_repeat_and_launch_domain(
        inputs.repeat_count,
        inputs.expected_launch_indices,
    )
    arguments = _validated_arguments(inputs.args)
    threshold_pct = _validated_admission_threshold(admission_threshold_pct)
    timeout_s = _validated_timeout(timeout_per_run_s)
    if not isinstance(inputs.executable_authority, RetainedExecutable):
        raise ImmutableArtifactError(
            "phase executable authority must be a RetainedExecutable"
        )
    executable_authority = inputs.executable_authority
    admitted_executable_descriptor = executable_authority.descriptor
    tracer_descriptor = inputs.tracer_descriptor
    for label, descriptor in (
        ("phase executable", admitted_executable_descriptor),
        ("phase tracer", tracer_descriptor),
    ):
        if type(descriptor) is not int or descriptor < 0:
            raise ImmutableArtifactError(
                f"{label} descriptor must be a nonnegative integer"
            )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ImmutableArtifactError(
                f"{label} descriptor is not a single-link regular file"
            )
    admitted_executable_metadata = os.fstat(admitted_executable_descriptor)
    if not (admitted_executable_metadata.st_mode & 0o111):
        raise ImmutableArtifactError(
            "phase executable descriptor is not executable"
        )

    def verify_executable_authority() -> None:
        try:
            if executable_authority.descriptor != admitted_executable_descriptor:
                raise ImmutableArtifactError(
                    "retained executable descriptor identity changed"
                )
            executable_authority.verify_unchanged()
            observed = os.fstat(admitted_executable_descriptor)
            if (
                observed.st_dev != admitted_executable_metadata.st_dev
                or observed.st_ino != admitted_executable_metadata.st_ino
                or observed.st_mode != admitted_executable_metadata.st_mode
                or observed.st_nlink != admitted_executable_metadata.st_nlink
                or observed.st_size != admitted_executable_metadata.st_size
                or observed.st_mtime_ns != admitted_executable_metadata.st_mtime_ns
                or observed.st_ctime_ns != admitted_executable_metadata.st_ctime_ns
            ):
                raise ImmutableArtifactError(
                    "retained executable descriptor generation changed"
                )
        except (ImmutableArtifactError, OSError) as exc:
            raise PhaseInjectExecutableAuthorityInvalid(str(exc)) from exc

    verify_executable_authority()
    try:
        base_env = tracer_runtime_env(
            inputs.tracer_so_path,
            base_env=inputs.base_env,
            retained_descriptor=tracer_descriptor,
        )
    except TracerRuntimeError as exc:
        raise PhaseInjectRuntimeUnavailable(str(exc)) from exc
    tracer_proc_path = f"/proc/self/fd/{tracer_descriptor}"
    effective_runner = (
        _bounded_subprocess_runner
        if subprocess_runner is None
        else subprocess_runner
    )
    env_no_inject = dict(base_env)
    env_no_inject["CUDA_INJECTION64_PATH"] = tracer_proc_path
    capture_fds: tuple[int, ...] = ()
    capture_folder = str(inputs.traces_folder)
    if inputs.traces_descriptor is not None:
        descriptor = inputs.traces_descriptor
        if type(descriptor) is not int or descriptor < 0:
            raise ImmutableArtifactError(
                "phase trace staging descriptor must be a nonnegative integer"
            )
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ImmutableArtifactError(
                "phase trace staging descriptor is not a directory"
            )
        capture_folder = f"/proc/self/fd/{descriptor}"
        capture_fds = (descriptor,)
    env_no_inject["TRACES_FOLDER"] = capture_folder
    # The tracer ignores TRACES_FOLDER unless USER_DEFINED_FOLDERS=1
    # is also set (see tracer_tool.cu's R51 redirect at the
    # `user_defined_folders == 1` gate). Without this, the tracer
    # writes phase_records_<repeat>_<launch>.bin to its hardcoded
    # default `<cwd>/traces/` path, where emit_per_launch_phase_ledgers
    # cannot find them and emits zero ledger sidecars even though
    # phase records ARE produced. handle_trace's regular-trace env
    # builder sets this for the non-phase-inject path; the A/B pair
    # must set it on both sides to keep the producer output
    # co-located with traces_folder.
    env_no_inject["USER_DEFINED_FOLDERS"] = "1"
    # The tracer's env-gated activation switch is OFF for the
    # no-inject side (so the existing instruction-trace pipeline
    # runs unchanged) and ON for the with-inject side.
    env_no_inject.pop("SIMBENCH_NVBIT_PHASE_INJECT", None)

    env_with_inject = dict(env_no_inject)
    env_with_inject["SIMBENCH_NVBIT_PHASE_INJECT"] = "1"
    _validate_environment(env_no_inject)
    _validate_environment(env_with_inject)
    verify_executable_authority()

    def _run_side(side: str, env: dict[str, str]) -> tuple[Any, int]:
        verify_executable_authority()
        try:
            invocation_descriptor = os.dup(admitted_executable_descriptor)
        except OSError as exc:
            raise PhaseInjectExecutableAuthorityInvalid(
                f"cannot duplicate admitted executable descriptor: {exc}"
            ) from exc
        primary: BaseException | None = None
        proc: Any | None = None
        t_start = timer_ns()
        t_end: int | None = None
        try:
            invocation_metadata = os.fstat(invocation_descriptor)
            if (
                invocation_metadata.st_dev != admitted_executable_metadata.st_dev
                or invocation_metadata.st_ino
                != admitted_executable_metadata.st_ino
                or invocation_metadata.st_mode
                != admitted_executable_metadata.st_mode
                or invocation_metadata.st_nlink
                != admitted_executable_metadata.st_nlink
                or invocation_metadata.st_size
                != admitted_executable_metadata.st_size
            ):
                raise PhaseInjectExecutableAuthorityInvalid(
                    "duplicated child executable is not the admitted inode"
                )
            runner_kwargs: dict[str, Any] = {
                "capture_output": True,
                "text": False,
                "env": env,
                "pass_fds": (
                    invocation_descriptor,
                    tracer_descriptor,
                    *capture_fds,
                ),
            }
            if timeout_s is not None:
                runner_kwargs["timeout"] = timeout_s
            try:
                proc = effective_runner(
                    [f"/proc/self/fd/{invocation_descriptor}", *arguments],
                    **runner_kwargs,
                )
                t_end = timer_ns()
            except subprocess.TimeoutExpired as exc:
                primary = PhaseInjectAbRunTimeout(
                    side=side,
                    timeout_s=timeout_s or 0,
                    stderr=_bounded_diagnostic_text(exc.stderr),
                )
            except BaseException as exc:
                primary = exc
        finally:
            try:
                os.close(invocation_descriptor)
            except OSError as exc:
                close_failure = PhaseInjectExecutableAuthorityInvalid(
                    f"child executable descriptor cleanup failed: {exc}"
                )
                if primary is None:
                    primary = close_failure
                else:
                    try:
                        primary.add_note(str(close_failure))
                    except (AttributeError, TypeError):
                        pass
            try:
                verify_executable_authority()
            except PhaseInjectExecutableAuthorityInvalid as exc:
                if primary is None:
                    primary = exc
                else:
                    try:
                        primary.add_note(
                            "executable authority also failed: " + str(exc)
                        )
                    except (AttributeError, TypeError):
                        pass
        if primary is not None:
            raise primary
        assert proc is not None and t_end is not None
        stdout_bytes = _captured_output_bytes(proc.stdout, label="stdout")
        stderr_bytes = _captured_output_bytes(proc.stderr, label="stderr")
        output_bytes = len(stdout_bytes) + len(stderr_bytes)
        if output_bytes > PHASE_AB_MAX_CAPTURE_BYTES:
            raise PhaseInjectAbOutputLimitExceeded(
                observed=output_bytes,
                stderr=_bounded_diagnostic_text(stderr_bytes),
            )
        if type(proc.returncode) is not int or not (
            -(2**31) <= proc.returncode <= 2**31 - 1
        ):
            raise ImmutableArtifactError(
                "phase A/B subprocess returncode must be an exact signed "
                "32-bit integer"
            )
        if proc.returncode != 0:
            raise PhaseInjectAbRunFailed(
                side=side,
                returncode=proc.returncode,
                stderr=_bounded_diagnostic_text(stderr_bytes),
            )
        return proc, t_end - t_start

    if inputs.traces_reader is None:
        trace_reader_context = PinnedDirectoryReader(inputs.traces_folder)
    else:
        trace_reader_context = nullcontext(inputs.traces_reader)
    with _preserve_primary_cleanup(trace_reader_context) as trace_reader:
        if inputs.traces_descriptor is not None:
            retained_directory = os.fstat(inputs.traces_descriptor)
            reader_directory = os.fstat(trace_reader.descriptor)
            if (
                retained_directory.st_dev != reader_directory.st_dev
                or retained_directory.st_ino != reader_directory.st_ino
            ):
                raise ImmutableArtifactError(
                    "phase trace reader disagrees with subprocess directory "
                    "descriptor"
                )

        def snapshot_source() -> PinnedTreeSnapshot:
            snapshot = trace_reader.snapshot_tree(
                max_files=_PHASE_SOURCE_MAX_FILES,
                max_directories=_PHASE_SOURCE_MAX_DIRECTORIES,
                max_total_file_bytes=_PHASE_SOURCE_MAX_BYTES,
                max_total_relative_path_bytes=(
                    _PHASE_SOURCE_MAX_RELATIVE_PATH_BYTES
                ),
            )
            trace_reader.verify_path()
            return snapshot

        initial_snapshot = snapshot_source()
        initial_phase_files = _phase_record_files(initial_snapshot)
        if initial_phase_files:
            raise PhaseRecordSourceNotFresh(
                "phase-record source is not empty before the A/B attempt: "
                f"{sorted(initial_phase_files)}"
            )

        no_inject_samples_ns: list[int] = []
        with_inject_samples_ns: list[int] = []
        final_source_snapshot = initial_snapshot
        for repeat_index in range(repeat_count):
            repeat_no_inject_env = dict(env_no_inject)
            repeat_no_inject_env[PHASE_REPEAT_INDEX_ENV] = str(repeat_index)
            phase_before_no_inject = _phase_record_files(snapshot_source())
            _, dt = _run_side("no_inject", repeat_no_inject_env)
            no_inject_samples_ns.append(dt)
            phase_after_no_inject = _phase_record_files(snapshot_source())
            if phase_after_no_inject != phase_before_no_inject:
                raise PhaseRecordSourceNotFresh(
                    "no-inject subprocess changed phase-record evidence for "
                    f"repeat {repeat_index}"
                )

            repeat_with_inject_env = dict(env_with_inject)
            repeat_with_inject_env[PHASE_REPEAT_INDEX_ENV] = str(repeat_index)
            before_with_inject = snapshot_source()
            phase_before_with_inject = _phase_record_files(before_with_inject)
            _, dt = _run_side("with_inject", repeat_with_inject_env)
            with_inject_samples_ns.append(dt)
            final_source_snapshot = snapshot_source()
            phase_after_with_inject = _phase_record_files(
                final_source_snapshot
            )
            if any(
                phase_after_with_inject.get(path) != metadata
                for path, metadata in phase_before_with_inject.items()
            ):
                raise PhaseRecordSourceNotFresh(
                    "with-inject subprocess changed prior phase-record "
                    f"evidence during repeat {repeat_index}"
                )
            new_paths = sorted(
                set(phase_after_with_inject) - set(phase_before_with_inject)
            )
            expected_new_paths = [
                f"phase_records_{repeat_index}_{launch_index}.bin"
                for launch_index in expected_launch_indices
            ]
            if new_paths != expected_new_paths:
                raise PhaseRecordSourceNotFresh(
                    "with-inject subprocess did not emit the exact admitted "
                    f"launch vector for repeat {repeat_index}: "
                    f"expected={expected_new_paths}, observed={new_paths}"
                )

        expected_final_paths = {
            f"phase_records_{repeat_index}_{launch_index}.bin"
            for repeat_index in range(repeat_count)
            for launch_index in expected_launch_indices
        }
        final_phase_paths = set(_phase_record_files(final_source_snapshot))
        if final_phase_paths != expected_final_paths:
            raise PhaseRecordSourceNotFresh(
                "final A/B source does not equal the admitted repeat-by-launch "
                f"matrix: expected={sorted(expected_final_paths)}, "
                f"observed={sorted(final_phase_paths)}"
            )
        verify_executable_authority()

        median_no = _median_ns(no_inject_samples_ns)
        median_with = _median_ns(with_inject_samples_ns)
        if median_no <= 0:
            raise PhaseInjectAbRunFailed(
                side="no_inject",
                returncode=0,
                stderr=(
                    f"median_no_inject_wall_ns={median_no}: zero/negative "
                    "median wall time, cannot compute overhead percentage."
                ),
            )
        overhead_pct = max(
            0.0, ((median_with - median_no) / median_no) * 100.0
        )
        return {
            "t_wall_no_inject_ns": int(median_no),
            "t_wall_with_inject_ns": int(median_with),
            "overhead_pct": float(overhead_pct),
            "admission_threshold_pct": threshold_pct,
            "admitted": overhead_pct <= threshold_pct,
            "_phase_source_snapshot": final_source_snapshot,
        }


class PhaseInjectAbRepeatCountInvalid(Exception):
    """The A/B pair cannot form a median or per-repeat evidence set."""

    TOKEN = "phase_inject_ab_repeat_count_invalid"

    def __init__(self, repeat_count: object):
        super().__init__(
            f"{self.TOKEN}: repeat_count must be an exact strictly positive "
            f"integer no greater than {PHASE_AB_MAX_REPEAT_COUNT}; "
            f"got {_bounded_value_label(repeat_count)}"
        )
        self.repeat_count = repeat_count


class PhaseInjectAbRunFailed(Exception):
    """Raised when one side of the A/B overhead pair returned
    non-zero. The `side` payload names which side failed so the
    caller can surface the correct typed diagnostic."""

    TOKEN = "phase_inject_ab_run_failed"

    def __init__(self, *, side: str, returncode: int, stderr: str):
        super().__init__(
            f"{self.TOKEN}: side={side!r} rc={returncode} "
            f"stderr={stderr!r}; the A/B overhead pair must complete "
            "cleanly on both sides."
        )
        self.side = side
        self.returncode = returncode
        self.stderr = stderr

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "side": self.side,
            "returncode": self.returncode,
            "stderr": self.stderr,
        }


class PhaseInjectRuntimeUnavailable(Exception):
    """The tracer cannot be dependency-checked before the A/B pair."""

    TOKEN = "phase_inject_runtime_unavailable"

    def __init__(self, detail: str):
        super().__init__(f"{self.TOKEN}: {detail}")
        self.detail = detail


class PhaseInjectAbRunTimeout(Exception):
    """Raised when one side of the A/B overhead pair exceeded the per-run
    timeout. Distinguishable from `PhaseInjectAbRunFailed` because the
    failure mode is wall-clock-bounded, not return-code-driven. Closes
    the R1-review-recorded blocker that an unbounded subprocess.run let
    a transient pathological GPU state burn 14 h+ of CPU."""

    TOKEN = "phase_inject_ab_run_timeout"

    def __init__(self, *, side: str, timeout_s: int, stderr: str):
        super().__init__(
            f"{self.TOKEN}: side={side!r} timeout_s={timeout_s} "
            f"stderr={stderr!r}; the A/B overhead pair's per-run "
            "subprocess exceeded its bounded timeout. Investigate "
            "tracer phase-injection orchestration before re-running."
        )
        self.side = side
        self.timeout_s = timeout_s
        self.stderr = stderr

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "side": self.side,
            "timeout_s": self.timeout_s,
            "stderr": self.stderr,
        }


class PhaseLedgerRepeatCoverageInvalid(Exception):
    """The committed A/B repeat-by-launch vector is incomplete or inconsistent."""

    TOKEN = "phase_inject_repeat_coverage_invalid"

    def __init__(
        self,
        *,
        expected_repeat_count: int,
        expected_launch_indices: tuple[int, ...],
        observed_launches: dict[int, set[int]],
    ) -> None:
        observed = {
            repeat_index: sorted(launches)
            for repeat_index, launches in sorted(observed_launches.items())
        }
        super().__init__(
            f"{self.TOKEN}: expected repeats "
            f"{list(range(expected_repeat_count))}, expected launches "
            f"{list(expected_launch_indices)}, observed launches {observed}"
        )
        self.expected_repeat_count = expected_repeat_count
        self.expected_launch_indices = expected_launch_indices
        self.observed_launches = observed


class PhaseLedgerPostCommitFailure(Exception):
    """A source capability failed after its immutable ledger commit."""

    TOKEN = "phase_inject_postcommit_source_invalid"

    def __init__(
        self,
        *,
        generation: ImmutableDirectoryGeneration,
        typed_reason: str,
        cause: BaseException,
    ) -> None:
        super().__init__(
            f"{self.TOKEN}: {typed_reason}: {type(cause).__name__}: {cause}"
        )
        self.generation = generation
        self.typed_reason = typed_reason
        self.cause = cause
        for note in getattr(cause, "__notes__", ()):
            try:
                self.add_note(note)
            except (AttributeError, TypeError):
                break


class PhaseLedgerEvidenceMissing(Exception):
    """The phase-enabled run emitted no decodable per-launch evidence."""

    TOKEN = "phase_inject_ledger_evidence_missing"

    def __init__(self, traces_folder: Path):
        super().__init__(
            f"{self.TOKEN}: no phase_records_<repeat>_<launch>.bin file under "
            f"{traces_folder} produced a matching nonempty launch ledger"
        )
        self.traces_folder = traces_folder


def _decode_phase_record_payload(
    relative_path: str,
    payload: bytes,
) -> list[Any]:
    if not payload:
        raise PhaseRecordPayloadInvalid(
            relative_path, "canonical payload is empty"
        )
    if len(payload) > PHASE_RECORD_MAX_BYTES:
        raise PhaseInjectResourceLimitExceeded(
            resource=f"phase-record payload bytes for {relative_path}",
            limit=PHASE_RECORD_MAX_BYTES,
            observed=len(payload),
        )
    record_count = len(payload) // _agg.PHASE_CLOCK_RECORD_SIZE
    if record_count > PHASE_RECORD_MAX_RECORDS:
        raise PhaseInjectResourceLimitExceeded(
            resource=f"decoded phase-record count for {relative_path}",
            limit=PHASE_RECORD_MAX_RECORDS,
            observed=record_count,
        )
    try:
        records = _agg.decode_phase_clock_records(payload)
    except (ValueError, TypeError) as exc:
        raise PhaseRecordPayloadInvalid(
            relative_path,
            f"payload is not a complete record sequence: {exc}",
        ) from exc
    if type(records) is not list or not records:
        raise PhaseRecordPayloadInvalid(
            relative_path, "decoder produced no complete records"
        )
    if len(records) > PHASE_RECORD_MAX_RECORDS:
        raise PhaseInjectResourceLimitExceeded(
            resource=f"decoded phase-record count for {relative_path}",
            limit=PHASE_RECORD_MAX_RECORDS,
            observed=len(records),
        )
    return records


def _expected_phase_paths(
    repeat_count: int,
    launch_indices: tuple[int, ...],
) -> set[str]:
    return {
        f"phase_records_{repeat_index}_{launch_index}.bin"
        for repeat_index in range(repeat_count)
        for launch_index in launch_indices
    }


def _observed_launch_matrix(paths: set[str]) -> dict[int, set[int]]:
    observed: dict[int, set[int]] = {}
    for relative_path in sorted(paths):
        match = PHASE_RECORDS_FILENAME_RE.fullmatch(relative_path)
        assert match is not None
        observed.setdefault(int(match.group(1)), set()).add(
            int(match.group(2))
        )
    return observed


def _ledger_file_identities(
    payloads: dict[str, bytes],
) -> tuple[DirectoryGenerationFile, ...]:
    return tuple(
        DirectoryGenerationFile(
            relative_path=relative_path,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        for relative_path, payload in sorted(payloads.items())
    )


def _authenticate_staged_ledgers(
    reader: PinnedDirectoryReader,
    expected_payloads: dict[str, bytes],
    *,
    total_file_bytes: int,
) -> PinnedTreeSnapshot:
    snapshot = reader.snapshot_tree(
        max_files=_PHASE_LEDGER_MAX_FILES,
        max_directories=1,
        max_total_file_bytes=total_file_bytes,
        max_total_relative_path_bytes=_PHASE_LEDGER_MAX_RELATIVE_PATH_BYTES,
    )
    files = dict(snapshot.files)
    directories = dict(snapshot.directories)
    if set(directories) != {"."}:
        raise ImmutableArtifactError(
            "phase ledger stage contains an unexpected nested directory"
        )
    if set(files) != set(expected_payloads):
        raise ImmutableArtifactError(
            "phase ledger stage inventory disagrees with admitted payloads"
        )
    for relative_path, expected_payload in sorted(expected_payloads.items()):
        metadata = files[relative_path]
        if (
            not stat.S_ISREG(metadata[2])
            or metadata[3] != 1
            or metadata[4] != len(expected_payload)
        ):
            raise ImmutableArtifactError(
                "phase ledger stage file generation is invalid: "
                f"{relative_path}"
            )
        observed = reader.read_snapshot(
            relative_path,
            max_bytes=len(expected_payload),
            expected_metadata=metadata,
        ).payload
        if observed != expected_payload:
            raise ImmutableArtifactError(
                f"phase ledger stage byte seal mismatch: {relative_path}"
            )
    reader.verify_path()
    return snapshot


def _is_resource_failure(exc: BaseException) -> bool:
    pending = [exc]
    seen: set[int] = set()
    resource_errnos = {errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.ENOSPC}
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (MemoryError, ImmutableArtifactResourceError)):
            return True
        if isinstance(current, OSError) and current.errno in resource_errnos:
            return True
        for chained in (current.__cause__, current.__context__):
            if isinstance(chained, BaseException):
                pending.append(chained)
    return False


def _postcommit_failure(
    generation: ImmutableDirectoryGeneration,
    cause: BaseException,
) -> PhaseLedgerPostCommitFailure:
    if isinstance(cause, PhaseLedgerPostCommitFailure):
        return cause
    typed_reason = (
        "trace_snapshot_resource_unavailable"
        if _is_resource_failure(cause)
        else "trace_filesystem_invalid"
    )
    return PhaseLedgerPostCommitFailure(
        generation=generation,
        typed_reason=typed_reason,
        cause=cause,
    )


def _inspect_committed_stage(
    stage: ExclusiveDirectoryStage,
) -> ImmutableDirectoryGeneration:
    retained = stage.committed_generation
    if retained is not None:
        return retained
    return inspect_immutable_directory_generation(
        stage.final_path,
        require_nonempty=True,
        max_files=_PHASE_LEDGER_MAX_FILES,
        max_directories=1,
        max_total_file_bytes=_agg.PHASE_LEDGER_MAX_BYTES,
        max_total_relative_path_bytes=_PHASE_LEDGER_MAX_RELATIVE_PATH_BYTES,
    )


def emit_per_launch_phase_ledgers(
    traces_folder: Path,
    ledger_output_dir: Path,
    *,
    attempt_id: str,
    expected_repeat_count: int,
    expected_launch_indices: tuple[int, ...],
    slug: str,
    shape_point: str,
    kernel_signature_sha256: str,
    binary_sha256: str,
    nvbit_tracer_build_hash: str,
    claim_scope: str,
    ab_overhead_record: dict[str, Any],
    dual_run_manifest_ref: str,
    admission: _consumer.PhaseInjectionAdmission | None = None,
    admission_threshold_pct: float | None = None,
    before_commit: Callable[[ExclusiveDirectoryStage], None] | None = None,
    source_reader: PinnedDirectoryReader | None = None,
    expected_source_snapshot: PinnedTreeSnapshot | None = None,
) -> dict[str, Any]:
    """Publish one complete attempt-scoped phase-ledger generation.

    ``ledger_output_dir`` is the generation namespace root; its immutable final
    child is ``attempt_id``.  The externally produced phase-record source is
    inventoried and read through one pinned no-follow capability.  Individual
    ledgers remain hidden in an exclusive stage until every source has decoded,
    schema validation and admission have completed, and one no-replace directory
    rename commits the complete read-only set.  A precommit failure preserves the
    hidden stage; a postcommit failure carries the exact retained final generation.
    Prior and racing final generations are never removed or reused.
    """
    if type(attempt_id) is not str or not attempt_id:
        raise ImmutableArtifactError("phase ledger attempt_id must be nonempty")
    expected_repeat_count, expected_launch_indices = (
        _validated_repeat_and_launch_domain(
            expected_repeat_count,
            expected_launch_indices,
        )
    )
    validated_admission_threshold = (
        None
        if admission_threshold_pct is None
        else _validated_admission_threshold(admission_threshold_pct)
    )

    emitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    staged_ledger_bytes = 0
    generation: ImmutableDirectoryGeneration | None = None
    stage: ExclusiveDirectoryStage | None = None
    owns_source_reader = source_reader is None
    if source_reader is None:
        try:
            source_reader = PinnedDirectoryReader(traces_folder)
        except FileNotFoundError as exc:
            raise PhaseLedgerEvidenceMissing(traces_folder) from exc
    source_context = (
        source_reader if owns_source_reader else nullcontext(source_reader)
    )
    try:
        with (
            _preserve_primary_cleanup(source_context) as reader,
            _preserve_primary_cleanup(ExitStack()) as source_monitors,
        ):
            source_monitor = source_monitors.enter_context(
                PinnedPathSetMonitor.from_reader(
                    reader,
                    regular_paths=(),
                    recursive_directories=(".",),
                    max_files=_PHASE_SOURCE_MAX_FILES,
                    max_directories=_PHASE_SOURCE_MAX_DIRECTORIES,
                    max_total_file_bytes=_PHASE_SOURCE_MAX_BYTES,
                    max_total_relative_path_bytes=(
                        _PHASE_SOURCE_MAX_RELATIVE_PATH_BYTES
                    ),
                )
            )
            source_snapshot = reader.snapshot_tree(
                max_files=_PHASE_SOURCE_MAX_FILES,
                max_directories=_PHASE_SOURCE_MAX_DIRECTORIES,
                max_total_file_bytes=_PHASE_SOURCE_MAX_BYTES,
                max_total_relative_path_bytes=(
                    _PHASE_SOURCE_MAX_RELATIVE_PATH_BYTES
                ),
            )
            if (
                expected_source_snapshot is not None
                and source_snapshot != expected_source_snapshot
            ):
                raise ImmutableArtifactError(
                    "phase record source changed after the A/B subprocess set"
                )
            phase_files = _phase_record_files(source_snapshot)
            if not phase_files:
                raise PhaseLedgerEvidenceMissing(traces_folder)
            expected_paths = _expected_phase_paths(
                expected_repeat_count,
                expected_launch_indices,
            )
            observed_paths = set(phase_files)
            observed_launches = _observed_launch_matrix(observed_paths)
            if observed_paths != expected_paths:
                raise PhaseLedgerRepeatCoverageInvalid(
                    expected_repeat_count=expected_repeat_count,
                    expected_launch_indices=expected_launch_indices,
                    observed_launches=observed_launches,
                )

            staged_payloads: dict[str, bytes] = {}
            for relative_path in sorted(expected_paths):
                metadata = phase_files[relative_path]
                payload = reader.read_snapshot(
                    relative_path,
                    max_bytes=PHASE_RECORD_MAX_BYTES,
                    expected_metadata=metadata,
                ).payload
                records = _decode_phase_record_payload(relative_path, payload)
                match = PHASE_RECORDS_FILENAME_RE.fullmatch(relative_path)
                assert match is not None
                repeat_index = int(match.group(1))
                launch_index = int(match.group(2))
                ledger = _agg.build_phase_cycle_ledger(
                    records,
                    slug=slug,
                    shape_point=shape_point,
                    repeat_index=repeat_index,
                    launch_index=launch_index,
                    kernel_signature_sha256=kernel_signature_sha256,
                    binary_sha256=binary_sha256,
                    nvbit_tracer_build_hash=nvbit_tracer_build_hash,
                    claim_scope=claim_scope,
                    ab_overhead_record=ab_overhead_record,
                    dual_run_manifest_ref=dual_run_manifest_ref,
                )
                filename = (
                    f"phase_cycle_ledger_{repeat_index}_{launch_index}.json"
                )
                ledger_payload = _agg.serialize_phase_cycle_ledger(ledger)
                attempted_staged_bytes = staged_ledger_bytes + len(
                    ledger_payload
                )
                if attempted_staged_bytes > _agg.PHASE_LEDGER_MAX_BYTES:
                    raise _agg.PhaseLedgerSerializationLimitExceeded(
                        max_bytes=_agg.PHASE_LEDGER_MAX_BYTES,
                        attempted_size_bytes_at_least=attempted_staged_bytes,
                    )
                staged_ledger_bytes = attempted_staged_bytes
                staged_payloads[filename] = ledger_payload
                verdict: dict[str, Any] = {
                    "repeat_index": repeat_index,
                    "launch_index": launch_index,
                    "filename": filename,
                    "admitted": True,
                }
                try:
                    _consumer.admit_phase_cycle_ledger(
                        ledger,
                        admission=admission,
                        admission_threshold_pct=(
                            validated_admission_threshold
                        ),
                    )
                    emitted.append(verdict)
                except _consumer.NvbitPhaseInjectionOverheadExceeded as exc:
                    verdict["admitted"] = False
                    verdict["reason"] = exc.to_json()
                    rejected.append(verdict)
                except _consumer.PhaseBoundaryMappingUnresolved as exc:
                    verdict["admitted"] = False
                    verdict["reason"] = exc.to_json()
                    rejected.append(verdict)

            closure = reader.snapshot_tree(
                max_files=_PHASE_SOURCE_MAX_FILES,
                max_directories=_PHASE_SOURCE_MAX_DIRECTORIES,
                max_total_file_bytes=_PHASE_SOURCE_MAX_BYTES,
                max_total_relative_path_bytes=(
                    _PHASE_SOURCE_MAX_RELATIVE_PATH_BYTES
                ),
            )
            if closure != source_snapshot:
                raise ImmutableArtifactError(
                    "phase record source generation changed before ledger commit"
                )
            reader.verify_path()
            source_monitor.verify_unchanged()

            stage = create_exclusive_directory_stage(
                ledger_output_dir, attempt_id
            )
            with _preserve_primary_cleanup(stage):
                for filename, ledger_payload in sorted(staged_payloads.items()):
                    publish_regular_file(
                        stage.staging_path,
                        filename,
                        ledger_payload,
                    )

                stage_reader_context = PinnedDirectoryReader(stage.staging_path)
                with _preserve_primary_cleanup(
                    stage_reader_context
                ) as stage_reader:
                    stage_monitor_context = PinnedPathSetMonitor.from_reader(
                        stage_reader,
                        regular_paths=(),
                        recursive_directories=(".",),
                        max_files=_PHASE_LEDGER_MAX_FILES,
                        max_directories=1,
                        max_total_file_bytes=staged_ledger_bytes,
                        max_total_relative_path_bytes=(
                            _PHASE_LEDGER_MAX_RELATIVE_PATH_BYTES
                        ),
                    )
                    with _preserve_primary_cleanup(
                        stage_monitor_context
                    ) as stage_monitor:
                        admitted_stage_snapshot = _authenticate_staged_ledgers(
                            stage_reader,
                            staged_payloads,
                            total_file_bytes=staged_ledger_bytes,
                        )
                        if before_commit is not None:
                            before_commit(stage)
                        stage_monitor.verify_unchanged()
                        closure_stage_snapshot = _authenticate_staged_ledgers(
                            stage_reader,
                            staged_payloads,
                            total_file_bytes=staged_ledger_bytes,
                        )
                        if closure_stage_snapshot != admitted_stage_snapshot:
                            raise ImmutableArtifactError(
                                "phase ledger stage generation changed after "
                                "admission"
                            )
                        stage_monitor.verify_unchanged()

                source_monitor.verify_unchanged()
                reader.verify_path()
                expected_ledger_files = _ledger_file_identities(
                    staged_payloads
                )
                try:
                    generation = stage.commit(
                        require_nonempty=True,
                        max_files=_PHASE_LEDGER_MAX_FILES,
                        max_directories=1,
                        max_total_file_bytes=_agg.PHASE_LEDGER_MAX_BYTES,
                        max_total_relative_path_bytes=(
                            _PHASE_LEDGER_MAX_RELATIVE_PATH_BYTES
                        ),
                        expected_files=expected_ledger_files,
                    )
                except BaseException as exc:
                    if stage.committed:
                        recovered = _inspect_committed_stage(stage)
                        raise _postcommit_failure(recovered, exc) from exc
                    raise
                try:
                    source_monitor.verify_unchanged()
                    reader.verify_path()
                    stage.verify_path()
                    observed_generation = inspect_immutable_directory_generation(
                        stage.final_path,
                        require_nonempty=True,
                        max_files=_PHASE_LEDGER_MAX_FILES,
                        max_directories=1,
                        max_total_file_bytes=_agg.PHASE_LEDGER_MAX_BYTES,
                        max_total_relative_path_bytes=(
                            _PHASE_LEDGER_MAX_RELATIVE_PATH_BYTES
                        ),
                    )
                    if observed_generation != generation:
                        raise ImmutableArtifactError(
                            "phase ledger generation changed during postcommit "
                            "authentication"
                        )
                except BaseException as exc:
                    raise _postcommit_failure(generation, exc) from exc
    except PhaseLedgerPostCommitFailure:
        raise
    except BaseException as exc:
        if generation is not None:
            raise _postcommit_failure(generation, exc) from exc
        if stage is not None and stage.committed:
            recovered = _inspect_committed_stage(stage)
            raise _postcommit_failure(recovered, exc) from exc
        raise

    assert generation is not None
    for verdict in (*emitted, *rejected):
        verdict["output_path"] = str(
            generation.path / str(verdict.pop("filename"))
        )
    return {
        "emitted_admitted": emitted,
        "emitted_rejected": rejected,
        "n_admitted": len(emitted),
        "n_rejected": len(rejected),
        "generation": generation,
    }
