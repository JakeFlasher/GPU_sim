"""Activation NVBit-trace orchestrator (tools/local_labels scope).

`workloads/simbench_bw/run_suite.py --mode trace` is microbench-shaped:
it walks `workloads/simbench_bw/{l1,l2,l3,l4}/<slug>/<slug>_run`
and consumes per-slug shape TOMLs under `workloads/simbench_bw/shapes/`.
The 5 activation cells in the headline cohort live under
`workloads/portable_lcd/<slug>/<slug>_run` with a different CLI
contract (binary takes `--timing_shape` / `--correctness_shape`
flags, not `--shape-point` paired with TOML). Rather than widen
run_suite.py, this orchestrator stays inside the local-label trace
surface (`tools/local_labels/`) and produces the same
on-disk sidecar shape the existing prereq gate
(``test_head_coherent_retrace_prereq.py``) expects:

    results/nvbit_capture_staging/activation_trace/workloads/<source>/results/
        <slug>/rtx_3090_ti/sm_86/trace_<shape_point>.json
    results/nvbit_capture_staging/activation_trace/workloads/<source>/results/
        <slug>/rtx_3090_ti/sm_86/trace_<shape_point>/dynamic_trace.pb

The sidecar carries the four admission-counter fields
(``expected_record_count``, ``admitted_record_count``,
``actual_drop_rate``, ``decoder_gap_dropped``), the retained tracer identity,
the portable admitted-content identities, and the separately named local
filesystem-generation identity, plus ``binary_sha256`` and ``source_commit`` so
the prereq gate admits the cell as HEAD-coherent.

The tracer is authorized for this trace surface, while NCU label collection
is handled separately by the descriptor-bound label driver. This orchestrator
writes only to its declared trace-evidence paths and does not invoke NCU.
It remains co-located with the label-driver code for the shared admission
workflow.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACER_SO = _REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.nvbit_tracer.runtime_env import tracer_runtime_env  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    DEFAULT_TRACE_ADMISSION_LIMITS,
    TraceAdmission,
    TraceAdmissionError,
    TraceLaunchSummary,
    open_admitted_trace_sources,
    require_trace_admission,
)
from tools.local_labels.immutable_artifact import (  # noqa: E402
    VERIFY_RESOURCE_BOUND,
    ImmutableArtifactError,
    ImmutableArtifactResourceError,
    ImmutableDirectoryGeneration,
    MetadataSnapshot,
    PinnedDirectoryReader,
    RetainedRegular,
    close_retained_regular_files_authenticated,
    create_exclusive_directory_stage,
    inspect_immutable_directory_generation,
    open_retained_executable,
    open_retained_regular,
    preserve_retained_regular_file,
    publish_retained_regular_file,
    retire_regular_file,
)
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_write_allowed,
)


def _tracer_env(*, retained_descriptor: int | None = None) -> dict:
    """Return the shared, pinned and dependency-checked tracer environment."""
    return tracer_runtime_env(
        _TRACER_SO,
        retained_descriptor=retained_descriptor,
    )

# Shapes the orchestrator can produce. The Round-7 widening adds
# `trace_l2_accuracy_xl` / `trace_l2_accuracy_lg` for portable_lcd
# rows. The portable_lcd runner at
# `workloads/portable_lcd/runner_common.h:137-156` already maps
# both generic shape names to Timing mode, so we invoke the binary
# with `--shape-point <name>` for those (the runner does NOT accept
# a `--trace_l2_accuracy_xl` flag form). `timing_shape` and
# `correctness_shape` keep the legacy `--<name>` invocation.
_SUPPORTED_SHAPES: tuple[str, ...] = (
    "timing_shape",
    "correctness_shape",
    "trace_l2_accuracy_xl",
    "trace_l2_accuracy_lg",
)
_GENERIC_SHAPE_POINT_SHAPES: frozenset[str] = frozenset({
    "trace_l2_accuracy_xl",
    "trace_l2_accuracy_lg",
})

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.local_labels.roster_path_resolver import resolve_cell_root  # noqa: E402


_STAGING_ROOT = _REPO_ROOT / "results" / "nvbit_capture_staging"
_DEFAULT_OUTPUT_ROOT = _STAGING_ROOT / "activation_trace"
_IDENTITY_MAX_BYTES = 512
_ARGUMENT_MAX_BYTES = 4096
_MAX_ACTIVATION_APP_ARGS = 64
_MAX_ACTIVATION_APP_ARG_BYTES = 64 * 1024
_DESCRIPTOR_RESOURCE_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE})


def _raise_if_descriptor_resource_unavailable(
    error: BaseException,
    *,
    label: str,
) -> None:
    if (
        isinstance(error, OSError)
        and error.errno in _DESCRIPTOR_RESOURCE_ERRNOS
    ):
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: {label}: {error}"
        ) from error


def _require_identity_string(
    value: object,
    *,
    label: str,
    component: bool = False,
    max_bytes: int = _IDENTITY_MAX_BYTES,
) -> str:
    """Return one exact, bounded identity string; never coerce input."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8") from exc
    if (
        len(encoded) > max_bytes
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        or (component and (value in {".", ".."} or "/" in value or "\\" in value))
    ):
        raise ValueError(f"{label} is not a valid bounded identity")
    return value


def _require_nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{label} must be a nonnegative exact integer")
    return value


def _require_hex_identity(value: object, *, label: str, length: int) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} must be lowercase {length}-hex")
    return value


def _require_app_arguments(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{label} must be an exact tuple")
    if len(value) > _MAX_ACTIVATION_APP_ARGS:
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: {label} count {len(value)} exceeds "
            f"{_MAX_ACTIVATION_APP_ARGS}"
        )
    checked: list[str] = []
    total_bytes = 0
    for index, item in enumerate(value):
        argument = _require_identity_string(
            item,
            label=f"{label}[{index}]",
            max_bytes=_ARGUMENT_MAX_BYTES,
        )
        total_bytes += len(argument.encode("utf-8"))
        if total_bytes > _MAX_ACTIVATION_APP_ARG_BYTES:
            raise ImmutableArtifactResourceError(
                f"{VERIFY_RESOURCE_BOUND}: {label} aggregate bytes "
                f"{total_bytes} exceed {_MAX_ACTIVATION_APP_ARG_BYTES}"
            )
        checked.append(argument)
    return tuple(checked)


def _validate_activation_cell_identities(
    *,
    slug: object,
    shape_point: object,
    sku: object,
    arch_dir: object,
    fallback_source: object,
    extra_app_args: object,
) -> tuple[str, str, str, str, str, tuple[str, ...]]:
    return (
        _require_identity_string(slug, label="slug", component=True),
        _require_identity_string(
            shape_point, label="shape_point", component=True,
        ),
        _require_identity_string(sku, label="sku", component=True),
        _require_identity_string(arch_dir, label="arch_dir", component=True),
        _require_identity_string(
            fallback_source, label="fallback_source", component=True,
        ),
        _require_app_arguments(extra_app_args, label="extra_app_args"),
    )


def _assert_staging_output(path: Path | str, *, operation: str) -> Path:
    """Clear one active capture output and keep it inside staging."""

    candidate = assert_write_allowed(
        path, repo_root=_REPO_ROOT, operation=operation,
    )
    staging = _STAGING_ROOT.resolve(strict=False)
    try:
        candidate.relative_to(staging)
    except ValueError as exc:
        raise ProtectedPathError(
            f"refusing to {operation} outside {staging}: {candidate}"
        ) from exc
    return candidate


def _reject_output_dir_extras(extra_app_args: tuple[str, ...]) -> None:
    """The orchestrator, not caller extras, owns portable output routing."""

    if any(
        arg == "--output_dir" or arg.startswith("--output_dir=")
        for arg in extra_app_args
    ):
        raise ValueError(
            "extra_app_args must not contain --output_dir or "
            "--output_dir=...; use output_root so every child output is "
            "cleared before launch"
        )


def _activation_output_plan(
    *,
    slug: str,
    shape_point: str,
    sku: str,
    arch_dir: str,
    fallback_source: str,
    output_root: Path | str = _DEFAULT_OUTPUT_ROOT,
    extra_app_args: tuple[str, ...] = (),
) -> dict[str, Path]:
    """Resolve and clear every output for one activation capture.

    ``resolve_cell_root`` is a read-only layout authority.  Its protected
    canonical result is used only as a relative layout template; every write
    is rebased below ``results/nvbit_capture_staging``.
    """

    slug, shape_point, sku, arch_dir, fallback_source, extra_app_args = (
        _validate_activation_cell_identities(
            slug=slug,
            shape_point=shape_point,
            sku=sku,
            arch_dir=arch_dir,
            fallback_source=fallback_source,
            extra_app_args=extra_app_args,
        )
    )
    _reject_output_dir_extras(extra_app_args)
    root = _assert_staging_output(output_root, operation="select output root")
    source_cell = resolve_cell_root(
        slug=slug,
        shape_point=shape_point,
        sku=sku,
        arch_dir=arch_dir,
        fallback_source=fallback_source,
    )
    try:
        relative_cell = source_cell.resolve(strict=False).relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ProtectedPathError(
            f"roster cell root escapes repository: {source_cell}"
        ) from exc
    cell_dir = root / relative_cell
    candidates = {
        "output_root": root,
        "cell_dir": cell_dir,
        "traces_folder": cell_dir / f"trace_{shape_point}",
        "sidecar_path": cell_dir / f"trace_{shape_point}.json",
        "manifest_path": cell_dir / f"measurement_manifest_{shape_point}.json",
        "workload_output_root": cell_dir / "workload_output",
        "portable_manifest": cell_dir / "workload_output" / slug / "manifest.json",
    }
    return {
        name: _assert_staging_output(path, operation=f"prepare {name}")
        for name, path in candidates.items()
    }


def _run_git_subprocess(
    command: list[str],
    *,
    resource_label: str,
    **kwargs,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, **kwargs)
    except OSError as descriptor_error:
        _raise_if_descriptor_resource_unavailable(
            descriptor_error,
            label=resource_label,
        )
        raise


def _git_head() -> str:
    out = _run_git_subprocess(
        ["git", "rev-parse", "HEAD"],
        resource_label="Git HEAD subprocess pipe allocation",
        capture_output=True, text=True, cwd=str(_REPO_ROOT), check=True,
    )
    return out.stdout.strip()


_SOURCE_COMMIT_AUTHORITY_PATHS = (
    "tools/local_labels/activation_trace_orchestrator.py",
    "tools/local_labels/immutable_artifact.py",
    "tools/nvbit_tracer/runtime_env.py",
    "tools/nvbit_tracer/trace_admission.py",
)


def _resolve_clean_source_commit(
    source_path: Path,
    retained_source,
) -> str:
    try:
        source_relative = source_path.relative_to(_REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"activation source is outside repository: {source_path}"
        ) from exc
    head = _git_head()
    relevant_paths = (*_SOURCE_COMMIT_AUTHORITY_PATHS, source_relative)
    status = _run_git_subprocess(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "status",
            "--porcelain",
            "--",
            *relevant_paths,
        ],
        resource_label="Git status subprocess pipe allocation",
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(
            "cannot verify activation source-commit cleanliness"
        )
    if status.stdout.strip():
        raise RuntimeError(
            "activation source-commit authority is dirty or untracked: "
            f"{status.stdout.strip()!r}"
        )
    committed_source = _run_git_subprocess(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "show",
            f"{head}:{source_relative}",
        ],
        resource_label="Git show subprocess pipe allocation",
        capture_output=True,
    )
    if committed_source.returncode != 0:
        raise RuntimeError(
            f"activation source is absent from source_commit {head}: "
            f"{source_relative}"
        )
    if committed_source.stdout != retained_source.read_bytes():
        raise RuntimeError(
            "retained activation source bytes disagree with source_commit"
        )
    if _git_head() != head:
        raise RuntimeError(
            "activation source_commit changed during authority verification"
        )
    retained_source.verify_unchanged()
    return head


def _verify_source_commit_head(source_commit: str) -> None:
    if _git_head() != source_commit:
        raise RuntimeError(
            "activation source_commit changed during capture publication"
        )


def _new_capture_attempt_id() -> str:
    return f"{time.time_ns():020d}-{os.urandom(8).hex()}"


def _trace_generation_commit_limits() -> dict[str, int]:
    limits = DEFAULT_TRACE_ADMISSION_LIMITS
    return {
        "max_files": limits.tree_files,
        "max_directories": limits.tree_directories,
        "max_total_file_bytes": (
            limits.trace_root_bytes
            + limits.stats_bytes
            + limits.total_threadblock_bytes
            + limits.allocations_sidecar_bytes
        ),
        "max_total_relative_path_bytes": limits.tree_relative_path_bytes,
    }


def _verify_committed_trace_generation(
    generation: ImmutableDirectoryGeneration,
) -> None:
    observed = inspect_immutable_directory_generation(
        generation.path,
        **_trace_generation_commit_limits(),
    )
    if observed != generation:
        raise ImmutableArtifactError(
            "committed raw trace generation changed after publication"
        )


def _raw_generation_record(
    generation: ImmutableDirectoryGeneration,
    *,
    cell_dir: Path,
    attempt_id: str,
) -> dict[str, object]:
    attempt_id = _require_identity_string(
        attempt_id, label="raw generation attempt_id", component=True,
    )
    return {
        "attempt_id": attempt_id,
        "path": generation.path.relative_to(cell_dir).as_posix(),
        "content_manifest_sha256": generation.content_manifest_sha256,
        "filesystem_generation_sha256": (
            generation.filesystem_generation_sha256
        ),
        "file_count": generation.file_count,
        "directory_count": generation.directory_count,
        "total_file_bytes": generation.total_file_bytes,
    }


def _validate_trace_admission_identities(
    admission: TraceAdmission,
    *,
    producer: str,
) -> None:
    """Reject coercible or malformed admission identities before publication."""
    if not isinstance(admission, TraceAdmission):
        raise RuntimeError(f"{producer}: admission must be a TraceAdmission")
    for field in (
        "launch_count",
        "threadblock_count",
        "instruction_count",
        "stats_row_count",
        "expected_record_count",
        "admitted_record_count",
        "dynamic_trace_size_bytes",
        "stats_size_bytes",
        "threadblock_payload_size_bytes",
    ):
        _require_nonnegative_integer(
            getattr(admission, field), label=f"{producer} {field}",
        )
    if admission.admitted_record_count > admission.expected_record_count:
        raise RuntimeError(
            f"{producer}: admitted_record_count exceeds expected_record_count"
        )
    if admission.allocations_sidecar_size_bytes is not None:
        _require_nonnegative_integer(
            admission.allocations_sidecar_size_bytes,
            label=f"{producer} allocations_sidecar_size_bytes",
        )
    _require_identity_string(
        admission.dynamic_trace_path,
        label=f"{producer} dynamic_trace_path",
        max_bytes=4096,
    )
    for field in (
        "dynamic_trace_sha256",
        "stats_sha256",
        "threadblock_payload_manifest_sha256",
        "trace_filesystem_generation_sha256",
        "trace_admission_sha256",
    ):
        _require_hex_identity(
            getattr(admission, field),
            label=f"{producer} {field}",
            length=64,
        )
    if admission.allocations_sidecar_sha256 is not None:
        _require_hex_identity(
            admission.allocations_sidecar_sha256,
            label=f"{producer} allocations_sidecar_sha256",
            length=64,
        )
    if (admission.allocations_sidecar_size_bytes is None) != (
        admission.allocations_sidecar_sha256 is None
    ):
        raise RuntimeError(
            f"{producer}: allocations sidecar identity is inconsistent"
        )
    if type(admission.launches) is not tuple:
        raise RuntimeError(f"{producer}: launches must be an exact tuple")
    integer_launch_fields = (
        "launch_index",
        "device_id",
        "stream_id",
        "kernel_id",
        "grid_x",
        "grid_y",
        "grid_z",
        "block_x",
        "block_y",
        "block_z",
        "shared_mem_static_bytes",
        "shared_mem_dynamic_bytes",
        "regs_per_thread",
        "inst_count",
    )
    for position, launch in enumerate(admission.launches):
        if not isinstance(launch, TraceLaunchSummary):
            raise RuntimeError(
                f"{producer}: launch {position} must be a TraceLaunchSummary"
            )
        for field in integer_launch_fields:
            _require_nonnegative_integer(
                getattr(launch, field),
                label=f"{producer} launch {position} {field}",
            )
        _require_identity_string(
            launch.kernel_name,
            label=f"{producer} launch {position} kernel_name",
            max_bytes=4096,
        )


def _require_single_admitted_launch(
    admission: TraceAdmission,
    *,
    producer: str,
) -> TraceLaunchSummary:
    """Return launch zero only for an internally consistent singleton trace."""

    _validate_trace_admission_identities(admission, producer=producer)
    if admission.launch_count != len(admission.launches):
        raise RuntimeError(
            f"{producer}: admitted launch_count {admission.launch_count} "
            f"disagrees with {len(admission.launches)} retained launch record(s)"
        )
    if admission.launch_count != 1:
        raise RuntimeError(
            f"{producer}: singleton producer contract requires "
            f"launch_count == 1; got {admission.launch_count}"
        )
    launch = admission.launches[0]
    if launch.launch_index != 0:
        raise RuntimeError(
            f"{producer}: singleton admitted launch has launch_index "
            f"{launch.launch_index}, expected 0"
        )
    return launch


def _trace_admission_identity_fields(admission: TraceAdmission) -> dict[str, object]:
    """Serialize portable content seals separately from the local generation."""

    _validate_trace_admission_identities(
        admission, producer="trace admission identity",
    )
    if (admission.allocations_sidecar_size_bytes is None) != (
        admission.allocations_sidecar_sha256 is None
    ):
        raise RuntimeError(
            "trace admission allocations identity is internally inconsistent"
        )
    return {
        # Portable content identities and their corresponding retained sizes.
        "dynamic_trace_size_bytes": admission.dynamic_trace_size_bytes,
        "dynamic_trace_sha256": admission.dynamic_trace_sha256,
        "stats_size_bytes": admission.stats_size_bytes,
        "stats_sha256": admission.stats_sha256,
        "threadblock_payload_size_bytes": (
            admission.threadblock_payload_size_bytes
        ),
        "threadblock_payload_manifest_sha256": (
            admission.threadblock_payload_manifest_sha256
        ),
        "allocations_sidecar_size_bytes": (
            admission.allocations_sidecar_size_bytes
        ),
        "allocations_sidecar_sha256": admission.allocations_sidecar_sha256,
        "trace_admission_sha256": admission.trace_admission_sha256,
        # Same-host file-generation identity. This is deliberately distinct from
        # the portable content/admission seals above.
        "trace_filesystem_generation_sha256": (
            admission.trace_filesystem_generation_sha256
        ),
    }


def _retire_matching_locator(
    path: Path,
    expected_payload: bytes,
    expected_snapshot: MetadataSnapshot | None = None,
) -> Path | None:
    return retire_regular_file(
        path,
        expected_payload=expected_payload,
        expected_snapshot=expected_snapshot,
    )


def _lock_activation_cell(reader: PinnedDirectoryReader) -> None:
    try:
        fcntl.flock(reader.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ImmutableArtifactError(
            "activation trace cell is owned by another live publisher"
        ) from exc
    except OSError as exc:
        raise ImmutableArtifactError(
            f"cannot lock activation trace cell: {exc}"
        ) from exc
    reader.verify_path()


def _release_activation_lock(descriptor: int) -> None:
    release_error: BaseException | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except BaseException as exc:
        release_error = exc
    try:
        os.close(descriptor)
    except BaseException as exc:
        if release_error is None:
            release_error = exc
        else:
            try:
                release_error.add_note(
                    f"activation lock descriptor close also failed: {exc}"
                )
            except (AttributeError, TypeError):
                pass
    if release_error is not None:
        raise ImmutableArtifactError(
            f"activation publication lock release failed: {release_error}"
        ) from release_error


class _RetainedActivationPublication:
    """Keep one published activation record bound to its retained sources."""

    def __init__(
        self,
        *,
        capture_sources: ExitStack,
        admitted_sources: ExitStack,
        lock_sources: ExitStack,
        lock_hold_sources: ExitStack,
        lock_reader: PinnedDirectoryReader,
        raw_generation: ImmutableDirectoryGeneration,
        sidecar_payload: bytes,
        manifest_payload: bytes,
        sidecar_locator: RetainedRegular | None = None,
        manifest_locator: RetainedRegular | None = None,
        sidecar_path: Path | None = None,
        manifest_path: Path | None = None,
        publication_checks: tuple[Callable[[], None], ...] = (),
    ) -> None:
        for label, stack in (
            ("capture_sources", capture_sources),
            ("admitted_sources", admitted_sources),
            ("lock_sources", lock_sources),
            ("lock_hold_sources", lock_hold_sources),
        ):
            if not isinstance(stack, ExitStack):
                raise TypeError(f"{label} must be an ExitStack")
        if type(lock_reader) is not PinnedDirectoryReader:
            raise TypeError("lock_reader must be an exact PinnedDirectoryReader")
        self._capture_sources = capture_sources
        self._admitted_sources = admitted_sources
        self._lock_sources = lock_sources
        self._lock_hold_sources = lock_hold_sources
        self._lock_reader = lock_reader
        self._raw_generation = raw_generation
        if type(publication_checks) is not tuple or any(
            not callable(check) for check in publication_checks
        ):
            raise TypeError("publication_checks must be a tuple of callables")
        if type(sidecar_payload) is not bytes or type(manifest_payload) is not bytes:
            raise TypeError("activation locator payloads must be bytes")

        opened_locators: list[RetainedRegular] = []

        def resolve_locator(
            locator: RetainedRegular | None,
            path: Path | None,
            payload: bytes,
            *,
            label: str,
        ) -> RetainedRegular:
            if (locator is None) == (path is None):
                raise TypeError(
                    f"{label} requires exactly one retained locator or path"
                )
            if locator is None:
                assert path is not None
                if not isinstance(path, Path):
                    raise TypeError(f"{label} path must be a Path")
                locator = open_retained_regular(path)
                opened_locators.append(locator)
            elif not isinstance(locator, RetainedRegular):
                raise TypeError(f"{label} locator must be a retained regular file")
            if locator.read_bytes() != payload:
                raise ImmutableArtifactError(
                    f"{label} retained locator payload mismatch"
                )
            return locator

        try:
            resolved_sidecar = resolve_locator(
                sidecar_locator,
                sidecar_path,
                sidecar_payload,
                label="activation sidecar",
            )
            resolved_manifest = resolve_locator(
                manifest_locator,
                manifest_path,
                manifest_payload,
                label="activation manifest",
            )
        except BaseException as resolution_error:
            for opened in reversed(opened_locators):
                try:
                    opened.close()
                except BaseException as close_error:
                    try:
                        resolution_error.add_note(
                            "legacy locator capability close failed: "
                            f"{close_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
            raise
        self._publication_checks = publication_checks
        self._locators = (
            (resolved_sidecar, sidecar_payload),
            (resolved_manifest, manifest_payload),
        )

    @property
    def locator_identities(
        self,
    ) -> tuple[tuple[Path, bytes, MetadataSnapshot], ...]:
        return tuple(
            (locator.path, payload, locator.snapshot)
            for locator, payload in self._locators
        )

    def __enter__(self) -> "_RetainedActivationPublication":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        closure_errors: list[BaseException] = []

        def add_note(error: BaseException, detail: str) -> None:
            try:
                error.add_note(detail)
            except (AttributeError, TypeError):
                pass

        def preserve_locators(error: BaseException) -> None:
            for locator, expected_payload in self._locators:
                try:
                    preserve_retained_regular_file(
                        locator, expected_payload=expected_payload,
                    )
                except BaseException as preservation_error:
                    add_note(
                        error,
                        "failed activation locator preservation for "
                        f"{locator.path}: {preservation_error}",
                    )

        def close_locators(error: BaseException | None) -> None:
            for locator, _expected_payload in reversed(self._locators):
                try:
                    locator.close()
                except BaseException as close_error:
                    closure_errors.append(close_error)
                    if error is not None:
                        add_note(
                            error,
                            "activation locator capability close failed: "
                            f"{close_error}",
                        )

        def close_stack(stack: ExitStack) -> None:
            try:
                stack.close()
            except BaseException as closure_error:
                closure_errors.append(closure_error)

        # Raw/source capabilities close first while both exact locator inodes,
        # mutation monitors, and the publication lock remain retained.
        close_stack(self._capture_sources)
        close_stack(self._admitted_sources)
        try:
            _verify_committed_trace_generation(self._raw_generation)
        except BaseException as closure_error:
            closure_errors.append(closure_error)
        for check in self._publication_checks:
            try:
                check()
            except BaseException as closure_error:
                closure_errors.append(closure_error)
        try:
            self._lock_reader.verify_path()
        except BaseException as closure_error:
            closure_errors.append(closure_error)

        if exc is not None or closure_errors:
            # Preserve body/earlier closure precedence while still draining every
            # lock stack and authenticating both locators before preservation.
            close_stack(self._lock_sources)
            close_stack(self._lock_hold_sources)
            for locator, _expected_payload in self._locators:
                try:
                    locator.verify_unchanged()
                except BaseException as closure_error:
                    closure_errors.append(closure_error)

            primary = exc if exc is not None else closure_errors[0]
            for closure_error in closure_errors:
                if closure_error is not primary:
                    add_note(
                        primary,
                        f"additional retained-source failure: {closure_error}",
                    )
            preserve_locators(primary)
            close_locators(primary)
            if exc is not None:
                if closure_errors:
                    add_note(
                        exc,
                        "retained activation source closure failed: "
                        f"{closure_errors[0]}",
                    )
                return False
            raise primary

        # Drain and close both locator monitors as one authenticated set while
        # the cell lock is still held. The lock-stack callbacks run with both
        # inode descriptors retained; any callback or post-release identity
        # failure therefore leaves both exact generations salvageable.
        try:
            close_retained_regular_files_authenticated(
                tuple(locator for locator, _payload in self._locators),
                lock_reader=self._lock_reader,
                closure_callbacks=(
                    self._lock_sources.close,
                    self._lock_hold_sources.close,
                ),
            )
        except BaseException as close_error:
            preserve_locators(close_error)
            close_locators(close_error)
            raise
        return False


def trace_activation_cell(
    *,
    slug: str,
    shape_point: str,
    arch_dir: str = "sm_86",
    sku: str = "rtx_3090_ti",
    extra_app_args: tuple[str, ...] = (),
    fallback_source: str = "portable_lcd",
    output_root: Path | str = _DEFAULT_OUTPUT_ROOT,
    _precleared_plan: dict[str, Path] | None = None,
    _retention_stack: ExitStack | None = None,
    _published_locators: list[
        tuple[Path, bytes, MetadataSnapshot]
    ] | None = None,
) -> dict:
    """Run the NVBit tracer against the activation binary for one
    cell and emit the schema-shaped sidecar at the prereq-gate's
    expected location.

    ``extra_app_args`` are additional CLI arguments forwarded to the
    activation binary AFTER the shape-mode flag. They are recorded
    in the sidecar's ``extra_app_args`` field so the trace identity
    is auditable. The manifest's ``shape_point`` is unchanged by
    these extras; the activation orchestrator preserves the
    manifest-level shape identity while letting the operator grow
    the per-launch measurement axes via the binary's CLI surface.

    Returns the sidecar dict (also written to disk). Raises on any
    capture failure (binary missing, tracer missing, stats.csv
    missing, dynamic_trace.pb missing)."""
    slug, shape_point, sku, arch_dir, fallback_source, extra_app_args = (
        _validate_activation_cell_identities(
            slug=slug,
            shape_point=shape_point,
            sku=sku,
            arch_dir=arch_dir,
            fallback_source=fallback_source,
            extra_app_args=extra_app_args,
        )
    )
    if _retention_stack is not None and not isinstance(_retention_stack, ExitStack):
        raise TypeError("_retention_stack must be an ExitStack when provided")
    if _published_locators is not None and type(_published_locators) is not list:
        raise TypeError("_published_locators must be a list when provided")
    if shape_point not in _SUPPORTED_SHAPES:
        raise ValueError(
            f"shape_point must be one of {_SUPPORTED_SHAPES}; "
            f"got {shape_point!r}"
        )
    if sku not in _DEVICE_PROPS_BY_SKU:
        raise ValueError(f"unsupported activation sku: {sku!r}")
    plan = _precleared_plan or _activation_output_plan(
        slug=slug,
        shape_point=shape_point,
        sku=sku,
        arch_dir=arch_dir,
        fallback_source=fallback_source,
        output_root=output_root,
        extra_app_args=extra_app_args,
    )
    # Defensive per-call clearance remains even when main() preflighted the
    # complete batch.  This occurs before Git, tracer-environment, launch, or
    # mutation.
    plan = {
        name: _assert_staging_output(path, operation=f"recheck {name}")
        for name, path in plan.items()
    }
    if not _TRACER_SO.is_file():
        raise FileNotFoundError(
            f"NVBit tracer .so missing at {_TRACER_SO}; restore the "
            "byte-exact preserved artifact before capture."
        )

    binary = (
        _REPO_ROOT / "workloads" / "portable_lcd" / slug / f"{slug}_run"
    )
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"activation binary missing or not executable: {binary}"
        )

    source = (
        _REPO_ROOT / "workloads" / "portable_lcd" / slug / f"{slug}.cu"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"activation source missing: {source} "
            f"(needed for kernel_signature_sha256)"
        )

    attempt_id = _require_identity_string(
        _new_capture_attempt_id(),
        label="trace_capture_attempt_id",
        component=True,
    )
    cell_dir = plan["cell_dir"]
    generation_root = plan["traces_folder"]
    sidecar_path = plan["sidecar_path"]
    manifest_path = plan["manifest_path"]
    workload_output_root = plan["workload_output_root"] / attempt_id

    # The portable_lcd runner accepts `--timing_shape` /
    # `--correctness_shape` for the two canonical modes, AND
    # `--shape-point <name>` for the generic shapes.
    if shape_point in _GENERIC_SHAPE_POINT_SHAPES:
        shape_argv: list[str] = ["--shape-point", shape_point]
    else:
        shape_argv = [f"--{shape_point}"]

    with (
        ExitStack() as lock_hold_sources,
        ExitStack() as lock_sources,
        ExitStack() as capture_sources,
    ):
        retained_binary = capture_sources.enter_context(
            open_retained_executable(binary)
        )
        retained_tracer = capture_sources.enter_context(
            open_retained_regular(_TRACER_SO)
        )
        retained_source = capture_sources.enter_context(
            open_retained_regular(source)
        )
        head = _require_hex_identity(
            _resolve_clean_source_commit(source, retained_source),
            label="source_commit",
            length=40,
        )
        binary_sha = _require_hex_identity(
            retained_binary.sha256, label="binary_sha256", length=64,
        )
        tracer_build_sha = _require_hex_identity(
            retained_tracer.sha256,
            label="nvbit_tracer_build_hash",
            length=64,
        )
        kernel_sig = _require_hex_identity(
            retained_source.sha256,
            label="kernel_signature_sha256",
            length=64,
        )

        cell_dir.mkdir(parents=True, exist_ok=True)
        generation_root.mkdir(parents=True, exist_ok=True)
        locked_generation_root = lock_sources.enter_context(
            PinnedDirectoryReader(generation_root)
        )
        _lock_activation_cell(locked_generation_root)
        try:
            lock_hold_descriptor = os.dup(locked_generation_root.descriptor)
        except OSError as descriptor_error:
            _raise_if_descriptor_resource_unavailable(
                descriptor_error,
                label="activation publication lock duplication",
            )
            raise
        lock_hold_sources.callback(
            _release_activation_lock, lock_hold_descriptor
        )

        # Previous active locators are retired byte-exact while this cell's
        # publication lock and every launch-defining source remain retained.
        # Raw generations are never deleted.
        retire_regular_file(sidecar_path)
        retire_regular_file(manifest_path)

        stage = create_exclusive_directory_stage(generation_root, attempt_id)
        with stage:
            workload_output_root.mkdir(parents=True, exist_ok=False)
            retained_workload_output = capture_sources.enter_context(
                PinnedDirectoryReader(workload_output_root)
            )
            env = _tracer_env(
                retained_descriptor=retained_tracer.descriptor,
            )
            env["CUDA_INJECTION64_PATH"] = retained_tracer.proc_path
            env["USER_DEFINED_FOLDERS"] = "1"
            env["TRACES_FOLDER"] = f"/proc/self/fd/{stage.descriptor}"
            invocation = [
                retained_binary.proc_path,
                *shape_argv,
                *extra_app_args,
                "--output_dir",
                f"/proc/self/fd/{retained_workload_output.descriptor}",
            ]
            retained_binary.verify_unchanged()
            retained_tracer.verify_unchanged()
            retained_source.verify_unchanged()
            _verify_source_commit_head(head)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    invocation,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=str(binary.parent),
                    pass_fds=(
                        retained_binary.descriptor,
                        retained_tracer.descriptor,
                        stage.descriptor,
                        retained_workload_output.descriptor,
                    ),
                )
            except OSError as descriptor_error:
                _raise_if_descriptor_resource_unavailable(
                    descriptor_error,
                    label="activation subprocess pipe allocation",
                )
                raise
            t1 = time.perf_counter()
            retained_binary.verify_unchanged()
            retained_tracer.verify_unchanged()
            retained_source.verify_unchanged()
            _verify_source_commit_head(head)
            retained_workload_output.verify_path()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"activation binary {binary} returned {proc.returncode}; "
                    f"stderr (tail): {proc.stderr[-500:]!r}"
                )
            raw_generation = stage.commit(
                require_nonempty=True,
                **_trace_generation_commit_limits(),
            )
            _verify_committed_trace_generation(raw_generation)
            retained_binary.verify_unchanged()
            retained_tracer.verify_unchanged()
            retained_source.verify_unchanged()
            _verify_source_commit_head(head)

        traces_folder = raw_generation.path
        pb_path = traces_folder / "dynamic_trace.pb"
        _verify_committed_trace_generation(raw_generation)
        try:
            admission = require_trace_admission(traces_folder)
        except TraceAdmissionError as exc:
            raise RuntimeError(
                f"NVBit capture for {slug}/{shape_point} is not admissible: {exc}"
            ) from exc
        _verify_committed_trace_generation(raw_generation)
        launch = _require_single_admitted_launch(
            admission,
            producer=f"activation trace {slug}/{shape_point}",
        )
        expected = admission.expected_record_count
        admitted = admission.admitted_record_count
        drop_count = max(expected - admitted, 0)
        drop_rate = (drop_count / expected) if expected else 0.0
        raw_generation_record = _raw_generation_record(
            raw_generation,
            cell_dir=cell_dir,
            attempt_id=attempt_id,
        )

        sidecar = {
            "schema_version": 1,
            "slug": slug,
            "shape_set": "activation_shapes",
            "shape_point": shape_point,
            "trace_capture_attempt_id": attempt_id,
            "raw_trace_generation": raw_generation_record,
            "trace_pb_path": str(pb_path.relative_to(cell_dir)),
            "trace_bytes": admission.dynamic_trace_size_bytes,
            "wall_time_s": t1 - t0,
            "tracer_so_path": str(
                retained_tracer.path.relative_to(_REPO_ROOT),
            ),
            "nvbit_tracer_build_hash": tracer_build_sha,
            **_trace_admission_identity_fields(admission),
            "expected_record_count": expected,
            "admitted_record_count": admitted,
            "actual_drop_rate": drop_rate,
            "channel_overflow_dropped": 0,
            "ring_buffer_loss_dropped": 0,
            "decoder_gap_dropped": 0,
            "expected_record_mismatch_dropped": drop_count,
            "binary_sha256": binary_sha,
            "source_commit": head,
            "kernel_signature_sha256": kernel_sig,
            "kernel_name": launch.kernel_name,
            "device_name": _DEVICE_PROPS_BY_SKU[sku]["device_name"],
            "arch": arch_dir,
            "extra_app_args": list(extra_app_args),
            "hrd_skip_reason": "activation_orchestrator_emit_only",
        }
        manifest_payload = _build_activation_manifest_payload(
            slug=slug,
            shape_point=shape_point,
            binary_sha256=binary_sha,
            source_commit=head,
            kernel_signature_sha256=kernel_sig,
            sku=sku,
        )
        sidecar_payload = (
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        manifest_bytes = (
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

        admitted_sources = ExitStack()
        published_locators: list[tuple[RetainedRegular, bytes]] = []
        publication_transferred = False
        try:
            with admitted_sources:
                trace_sources = admitted_sources.enter_context(
                    open_admitted_trace_sources(
                        traces_folder,
                        expected_admission=admission,
                    )
                )

                def verify_publication_sources() -> None:
                    _verify_committed_trace_generation(raw_generation)
                    trace_sources.verify_unchanged()
                    retained_binary.verify_unchanged()
                    retained_tracer.verify_unchanged()
                    retained_source.verify_unchanged()
                    retained_workload_output.verify_path()
                    _verify_source_commit_head(head)

                verify_publication_sources()
                manifest_locator = publish_retained_regular_file(
                    manifest_path.parent,
                    manifest_path.name,
                    manifest_bytes,
                )
                published_locators.append((manifest_locator, manifest_bytes))
                manifest_locator.verify_unchanged()
                verify_publication_sources()
                sidecar_locator = publish_retained_regular_file(
                    sidecar_path.parent,
                    sidecar_path.name,
                    sidecar_payload,
                )
                published_locators.append((sidecar_locator, sidecar_payload))
                manifest_locator.verify_unchanged()
                sidecar_locator.verify_unchanged()
                verify_publication_sources()
                retained_capture_sources = capture_sources.pop_all()
                retained_admitted_sources = admitted_sources.pop_all()
                retained_lock_sources = lock_sources.pop_all()
                retained_lock_hold_sources = lock_hold_sources.pop_all()
                try:
                    retained_publication = _RetainedActivationPublication(
                        capture_sources=retained_capture_sources,
                        admitted_sources=retained_admitted_sources,
                        lock_sources=retained_lock_sources,
                        lock_hold_sources=retained_lock_hold_sources,
                        lock_reader=locked_generation_root,
                        raw_generation=raw_generation,
                        sidecar_locator=sidecar_locator,
                        sidecar_payload=sidecar_payload,
                        manifest_locator=manifest_locator,
                        manifest_payload=manifest_bytes,
                        publication_checks=(
                            lambda: _verify_source_commit_head(head),
                        ),
                    )
                except BaseException as construction_error:
                    for retained_stack in (
                        retained_capture_sources,
                        retained_admitted_sources,
                        retained_lock_sources,
                        retained_lock_hold_sources,
                    ):
                        try:
                            retained_stack.close()
                        except BaseException as close_error:
                            try:
                                construction_error.add_note(
                                    "activation publication construction cleanup "
                                    f"failed: {close_error}"
                                )
                            except (AttributeError, TypeError):
                                pass
                    raise
                if _retention_stack is None:
                    retained_publication.__enter__()
                    publication_transferred = True
                    retained_publication.__exit__(None, None, None)
                    if _published_locators is not None:
                        _published_locators.extend(
                            retained_publication.locator_identities
                        )
                else:
                    try:
                        _retention_stack.enter_context(retained_publication)
                    except BaseException as registration_error:
                        try:
                            retained_publication.__exit__(
                                type(registration_error),
                                registration_error,
                                registration_error.__traceback__,
                            )
                        except BaseException as cleanup_error:
                            try:
                                registration_error.add_note(
                                    "activation retention-stack registration "
                                    f"cleanup failed: {cleanup_error}"
                                )
                            except (AttributeError, TypeError):
                                pass
                            # __exit__ is fail-closed, but if an injected failure
                            # escapes it, explicitly drain every transferred stack;
                            # the outer publication handler retains both locators.
                            for retained_stack in (
                                retained_capture_sources,
                                retained_admitted_sources,
                                retained_lock_sources,
                                retained_lock_hold_sources,
                            ):
                                try:
                                    retained_stack.close()
                                except BaseException as stack_error:
                                    try:
                                        registration_error.add_note(
                                            "activation registration fallback "
                                            f"stack close failed: {stack_error}"
                                        )
                                    except (AttributeError, TypeError):
                                        pass
                        else:
                            publication_transferred = True
                        raise
                    publication_transferred = True
                    if _published_locators is not None:
                        _published_locators.extend(
                            retained_publication.locator_identities
                        )
        except BaseException as publication_error:
            if not publication_transferred:
                for locator, expected_payload in reversed(published_locators):
                    try:
                        preserve_retained_regular_file(
                            locator, expected_payload=expected_payload,
                        )
                    except BaseException as preservation_error:
                        try:
                            publication_error.add_note(
                                "failed activation locator preservation for "
                                f"{locator.path}: {preservation_error}"
                            )
                        except (AttributeError, TypeError):
                            pass
                    try:
                        locator.close()
                    except BaseException as close_error:
                        try:
                            publication_error.add_note(
                                "failed activation locator capability close for "
                                f"{locator.path}: {close_error}"
                            )
                        except (AttributeError, TypeError):
                            pass
            raise

        return sidecar


# Device-prop constants keyed by GPU SKU. Originally pinned for the
# 3090 Ti only; round-5 (sm120_l2_observability_30_cell_cohort loop)
# extended the lookup to include the RTX 5060 laptop sm_120 entry so
# the orchestrator emits the correct sku + device_name + arch on the
# 5060 laptop host. The values mirror the microbench manifests under
# `workloads/simbench_bw/results/<slug>/<sku>/<arch>/`.
_RTX_3090_TI_PROPS = {
    "cuda_version": "13.2",
    # NVIDIA driver version observed on the 3090 Ti host. The CUDA
    # toolkit string `"13.2"` belongs in `cuda_version`, NOT here; the
    # earlier `driver_version = "13.2"` was that toolkit value misrouted
    # into the driver slot. The committed sm_86 label sidecars
    # (`data/local_labels/sm_86/.../label_*.json`) and the
    # device-properties manifest's 3090 Ti entry carry `595.71.05`, so
    # the consumer-facing `driver_version` must match — mirrors the
    # 5060 entry's correction below.
    "driver_version": "595.71.05",
    "gpu_uuid": "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c",
    "sm_count": 84,
    "total_global_mem_bytes": 25282478080,
    "compute_capability_major": 8,
    "compute_capability_minor": 6,
    "device_name": "NVIDIA GeForce RTX 3090 Ti",
    "gpu_sku": "rtx_3090_ti",
}

_RTX_5060_LAPTOP_PROPS = {
    "cuda_version": "13.2",
    # NVIDIA driver version observed on the 5060 laptop host. The
    # earlier value `"13.2"` was the CUDA toolkit string copied into
    # the wrong slot; the live provenance block in
    # `.humanize/rlcr/2026-05-26_15-40-37/round-5-summary.md` and the
    # 4 sm_120 measurement manifests committed in Rounds 4 + 5 all
    # carry `driver_version = "595.71.05"`. Round 6 reconciles the
    # orchestrator's manifest emission with that observed value so
    # the consumer-facing `driver_version` field cannot drift from
    # the device the binaries actually ran on.
    "driver_version": "595.71.05",
    "gpu_uuid": "GPU-7106d751-e907-1dd4-6c9a-ba17bb7a606c",
    "sm_count": 26,
    "total_global_mem_bytes": 8082096128,
    "compute_capability_major": 12,
    "compute_capability_minor": 0,
    "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "gpu_sku": "rtx_5060_laptop",
}

_DEVICE_PROPS_BY_SKU: dict[str, dict] = {
    "rtx_3090_ti": _RTX_3090_TI_PROPS,
    "rtx_5060_laptop": _RTX_5060_LAPTOP_PROPS,
}


def _build_activation_manifest_payload(
    *,
    slug: str,
    shape_point: str,
    binary_sha256: str,
    source_commit: str,
    kernel_signature_sha256: str,
    sku: str = "rtx_3090_ti",
) -> dict:
    """Build the schema-v2 measurement_manifest.json payload
    for an activation cell. The schema-required field set is
    pinned in workloads/simbench_bw/schemas/measurement_manifest.schema.json.

    The orchestrator does not perform a CUPTI Activity
    collection or NVML covariate sampling — those fields stay
    omitted (the schema marks them optional). The optional
    `emitter` field attributes provenance to the activation
    orchestrator so a downstream consumer can distinguish
    these manifests from the run_suite.py-emitted ones.
    """
    slug = _require_identity_string(slug, label="manifest slug", component=True)
    shape_point = _require_identity_string(
        shape_point, label="manifest shape_point", component=True,
    )
    sku = _require_identity_string(sku, label="manifest sku", component=True)
    binary_sha256 = _require_hex_identity(
        binary_sha256, label="manifest binary_sha256", length=64,
    )
    source_commit = _require_hex_identity(
        source_commit, label="manifest source_commit", length=40,
    )
    kernel_signature_sha256 = _require_hex_identity(
        kernel_signature_sha256,
        label="manifest kernel_signature_sha256",
        length=64,
    )
    if sku not in _DEVICE_PROPS_BY_SKU:
        raise ValueError(f"unsupported manifest sku: {sku!r}")
    return {
        "schema_version": 2,
        "slug": slug,
        "mode": "trace",
        # Trace-mode runs use a single-shot capture with no
        # NCU repeats; the warm/repeat/cv fields satisfy the
        # schema's required-field contract while documenting
        # that the activation orchestrator does not loop.
        "warm_up_iters": 0,
        "repeat_iters": 1,
        "cv_pct": 0.0,
        # Device-prop values keyed by sku; see _DEVICE_PROPS_BY_SKU.
        "cuda_version": _DEVICE_PROPS_BY_SKU[sku]["cuda_version"],
        "driver_version": _DEVICE_PROPS_BY_SKU[sku]["driver_version"],
        "gpu_sku": _DEVICE_PROPS_BY_SKU[sku]["gpu_sku"],
        "gpu_uuid": _DEVICE_PROPS_BY_SKU[sku]["gpu_uuid"],
        "sm_count": _DEVICE_PROPS_BY_SKU[sku]["sm_count"],
        "total_global_mem_bytes": (
            _DEVICE_PROPS_BY_SKU[sku]["total_global_mem_bytes"]
        ),
        "device_name": _DEVICE_PROPS_BY_SKU[sku]["device_name"],
        "compute_capability_major": (
            _DEVICE_PROPS_BY_SKU[sku]["compute_capability_major"]
        ),
        "compute_capability_minor": (
            _DEVICE_PROPS_BY_SKU[sku]["compute_capability_minor"]
        ),
        "binary_sha256": binary_sha256,
        # Activation binaries are built with the canonical
        # portable_lcd LCD flag set declared in
        # workloads/portable_lcd/common.mk's ARCH_LCD_FULL
        # (ARCH_SM86 + ARCH_SM90 + ARCH_SM120 + ARCH_PTX_FALLBACK):
        # virtual arch compute_80 only, real arches sm_86/sm_90/sm_120,
        # plus a compute_80 PTX fallback for forward JIT compatibility.
        # The LCD discipline forbids architecture-conditional 'a' / 'f'
        # tokens (sm_90a, sm_120a, sm_120f); see common.mk lines
        # 14, 19, 28, 96-103. Mirroring the canonical flag set here
        # satisfies the schema's required-field contract without
        # re-shelling-out to the toolchain AND matches what the
        # binaries were actually built with.
        "compiler_flags": [
            "-gencode", "arch=compute_80,code=sm_86",
            "-gencode", "arch=compute_80,code=sm_90",
            "-gencode", "arch=compute_80,code=sm_120",
            "-gencode", "arch=compute_80,code=compute_80",
        ],
        "kb_capability_card": (
            "references/NVIDIA_Manuals/blackwell_compatibility/"
            "01_blackwell-architecture-compatibility.md"
        ),
        "timestamp_utc": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ),
        "shape_point": shape_point,
        # Round-8 (loop sm120_l2_observability_30_cell_cohort): the
        # canonical `measurement_manifest.schema.json` was extended
        # to declare `kernel_signature_sha256` as an OPTIONAL
        # property so the sweep harness's `--measurement-manifests`
        # contract has a single source of truth for the join-key
        # field. Pre-Round-8 this field lived only in the
        # `trace_<shape>.json` sidecar and the staging tool
        # silently re-derived it; emitting it here too keeps the
        # producer-side rich manifest aligned with what the
        # staging tool preserves.
        "kernel_signature_sha256": kernel_signature_sha256,
    }
    # The activation-cell `source_commit` and `tracer_so_path` are
    # NOT in the canonical manifest schema (they live in the
    # per-shape `trace_<shape>.json` sidecar alongside other
    # capture metadata). `_ = source_commit` is a deliberate no-op
    # to keep the function signature stable for callers that
    # provide it.
    _ = source_commit


_ACTIVATION_CELLS: tuple[tuple[str, str], ...] = (
    ("layernorm_fwd_bf16_sm80", "timing_shape"),
    ("rmsnorm_fwd_bf16_sm80", "timing_shape"),
    ("silu_fwd_bf16_sm80", "correctness_shape"),
    ("silu_fwd_bf16_sm80", "timing_shape"),
    ("softmax_safe_bf16_sm80", "timing_shape"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the NVBit tracer against the 5 activation cohort "
            "cells and emit prereq-gate-shaped sidecars."
        ),
    )
    parser.add_argument(
        "--slug",
        default=None,
        help=(
            "Single slug to trace (paired with --shape-point). "
            "If omitted, all 5 activation cells are traced."
        ),
    )
    parser.add_argument(
        "--shape-point",
        default=None,
        help="Single shape_point to trace (timing_shape | correctness_shape).",
    )
    parser.add_argument(
        "--app-arg",
        action="append",
        default=[],
        help=(
            "Forward an extra CLI argument to the activation "
            "binary AFTER the shape-mode flag. May be passed "
            "multiple times (e.g., --app-arg --BH --app-arg 16). "
            "Recorded in the sidecar's extra_app_args field."
        ),
    )
    parser.add_argument(
        "--sku",
        default="rtx_3090_ti",
        choices=sorted(_DEVICE_PROPS_BY_SKU.keys()),
        help=(
            "Target GPU SKU directory under workloads/simbench_bw/"
            "results/<slug>/ in the mirrored staging layout. Default "
            "rtx_3090_ti; use rtx_5060_laptop "
            "when running on the 5060 laptop sm_120 host."
        ),
    )
    parser.add_argument(
        "--arch-dir",
        default="sm_86",
        help=(
            "Architecture-bound subpath under "
            "the mirrored staging cell layout. Default sm_86; "
            "use sm_120 on the 5060 laptop."
        ),
    )
    parser.add_argument(
        "--fallback-source",
        default="portable_lcd",
        choices=("portable_lcd", "simbench_bw"),
        help=(
            "Result-tree source for off-sm120-roster activation cells. "
            "Default portable_lcd mirrors the canonical relative layout "
            "beneath the guarded output root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=(
            "Active capture root. It must remain below "
            "results/nvbit_capture_staging; roster-resolved canonical paths "
            "are read-only layout templates."
        ),
    )
    args = parser.parse_args(argv)

    if (args.slug is None) != (args.shape_point is None):
        parser.error("--slug and --shape-point must be provided together")

    targets = (
        ((args.slug, args.shape_point),)
        if args.slug is not None
        else _ACTIVATION_CELLS
    )
    extras = tuple(args.app_arg)
    try:
        plans = [
            _activation_output_plan(
                slug=slug,
                shape_point=shape_point,
                sku=args.sku,
                arch_dir=args.arch_dir,
                fallback_source=args.fallback_source,
                output_root=args.output_root,
                extra_app_args=extras,
            )
            for slug, shape_point in targets
        ]
    except (ProtectedPathError, ValueError) as exc:
        print(f"activation_trace_orchestrator: output refusal: {exc}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for (slug, shape_point), plan in zip(targets, plans, strict=True):
        print(f"=== {slug} / {shape_point} ===", file=sys.stderr)
        try:
            sidecar = trace_activation_cell(
                slug=slug, shape_point=shape_point,
                arch_dir=args.arch_dir,
                sku=args.sku,
                extra_app_args=extras,
                fallback_source=args.fallback_source,
                output_root=args.output_root,
                _precleared_plan=plan,
            )
            print(
                f"OK: {slug}/{shape_point}: trace_bytes="
                f"{sidecar['trace_bytes']} expected="
                f"{sidecar['expected_record_count']} "
                f"admitted={sidecar['admitted_record_count']} "
                f"binary_sha256={sidecar['binary_sha256'][:16]}...",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"FAIL: {slug}/{shape_point}: {exc}", file=sys.stderr)
            failures.append(f"{slug}/{shape_point}: {exc}")

    if failures:
        print(
            f"\nactivation_trace_orchestrator: {len(failures)} cell(s) "
            f"failed:",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
