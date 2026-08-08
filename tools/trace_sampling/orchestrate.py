"""Two-pass trace-sampling orchestration entry point.

Drives one budget-breaching cell through the ladder using the tracer's capture
modes: pass 1 (counting-only) -> admissibility gate -> segment selection -> pass 2
(selective-dump of the chosen whole blocks) -> compress + validate -> the
closed-schema `trace_sampling.json`. Gate-forbidden cells are routed to a typed
`unavailable` (a reduced-shape needs a separate shape-adjust wrapper). The
per-segment HT estimate itself is finalized later at scoring time (the sim yields
only per-SM aggregates, so per-segment measurement is a sub-trace-replay seam);
this entry point records the sampling plan + the exact captured record count.

The roster capture driver calls `orchestrate_cell`; the tracer modes + this local
orchestration are the trace-sampling machinery.
"""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from admissibility_gate import MIN_SEGMENTS, M_SUBSET, classify
from compression import compress_trace_dir
from counting_pass import cell_stats_from_capture
from selection import SelectionPlanError, build_segment_population, plan_from_stats
from sidecar import SidecarPublicationState, build_sidecar, write_sidecar

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ImmutableArtifactResourceError,
    PinnedDirectoryReader,
    RetainedExecutable,
    RetainedRegular,
    open_retained_executable,
    open_retained_regular,
    publish_regular_file,
    retire_regular_file,
    salvage_retained_regular_descriptor,
    write_regular_bytes_at,
)
from tools.nvbit_tracer import capture_process as nvbit_process  # noqa: E402
from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)
from tools.nvbit_tracer.external_tracer import (  # noqa: E402
    EXTERNAL_TRACER_MAX_BYTES,
)
from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    TracerRuntimeError,
    default_runtime_manifest_path,
    retained_tracer_runtime_env,
)

DEFAULT_TRACER_SO = _REPO / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
DEFAULT_PB_LIB = _REPO / ".toolchain" / "pb" / "lib"
CAPTURE_SIDECAR = Path("extra_info") / "trace_sampling_capture.json"
_PASS_TIMEOUT_SECONDS = 60 * 60
_PASS_MAX_STDOUT_STDERR_BYTES = 8 * 1024 * 1024
_PASS_TIMEOUT_CODE = "sampling_target_timeout"
_PASS_LAUNCH_CODE = "sampling_target_launch_failed"


class SamplingRefusal(RuntimeError):
    """Typed refusal: no sampled sidecar or estimator point may be emitted."""

    def __init__(self, reason: str, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        self.evidence_state: str | None = None
        self.stdout = b""
        self.stderr = b""
        self.process_id: int | None = None
        self.failure_evidence: tuple[dict[str, object], ...] = ()
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _create_fresh_directory(path: Path, reason: str) -> None:
    """Atomically reserve a new output directory without deleting user data."""
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SamplingRefusal(reason, f"output path already exists: {path}") from exc
    except OSError as exc:
        raise SamplingRefusal(reason, f"cannot create {path}: {exc}") from exc


def _refuse_existing_output(path: Path, reason: str) -> None:
    if path.exists() or path.is_symlink():
        raise SamplingRefusal(reason, f"output path already exists: {path}")


def _require_sampling_inputs_unchanged(
    retained_binary: RetainedExecutable,
    retained_tracer: RetainedRegular,
) -> None:
    """Revalidate both execution capabilities with stable typed refusal."""
    try:
        retained_binary.verify_unchanged()
        retained_tracer.verify_unchanged()
    except ImmutableArtifactResourceError as exc:
        raise SamplingRefusal(
            "sampling_input_resource_unavailable",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    except (ImmutableArtifactError, OSError) as exc:
        raise SamplingRefusal(
            "sampling_input_invalid",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _require_sampling_outputs_unchanged(
    retained_outputs: list[PinnedDirectoryReader],
) -> None:
    """Revalidate every pass output root with stable typed refusal."""

    try:
        for retained_output in retained_outputs:
            retained_output.verify_path()
    except ImmutableArtifactResourceError as exc:
        raise SamplingRefusal(
            "sampling_output_resource_unavailable",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    except (ImmutableArtifactError, OSError) as exc:
        raise SamplingRefusal(
            "sampling_output_generation_changed",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _require_sampling_sidecar_unchanged(
    retained_sidecar: RetainedRegular,
    expected_payload: bytes,
) -> None:
    """Bind the active locator to the exact returned sidecar bytes."""
    try:
        payload = retained_sidecar.read_bytes()
        retained_sidecar.verify_unchanged()
    except ImmutableArtifactResourceError as exc:
        raise SamplingRefusal(
            "sidecar_publication_resource_unavailable",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    except (ImmutableArtifactError, OSError) as exc:
        raise SamplingRefusal(
            "sidecar_publication_changed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if payload != expected_payload:
        raise SamplingRefusal(
            "sidecar_publication_payload_mismatch",
            "published trace_sampling.json bytes disagree with returned sidecar",
        )


def _bind_sidecar_evidence_state(
    refusal: SamplingRefusal,
    state: str,
) -> None:
    refusal.evidence_state = state
    suffix = f"sidecar_evidence_state={state}"
    refusal.detail = (
        f"{refusal.detail}; {suffix}" if refusal.detail else suffix
    )
    refusal.args = (f"{refusal.reason}: {refusal.detail}",)


@contextmanager
def _retained_sidecar_return_boundary(
    retained_sidecar: RetainedRegular,
    publication_state: SidecarPublicationState,
):
    """Revalidate on return without letting cleanup replace a primary failure."""
    entered = retained_sidecar.__enter__()
    try:
        yield entered
    except BaseException as primary:
        try:
            retained_sidecar.__exit__(
                type(primary), primary, primary.__traceback__
            )
        except BaseException as cleanup:
            try:
                primary.add_note(
                    "retained sidecar cleanup also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            except (AttributeError, TypeError):
                pass
        raise
    else:
        retained_sidecar.__exit__(None, None, None)
        publication_state.close()


def _retain_sampling_process_failure(
    refusal: SamplingRefusal,
    *,
    out_dir: Path,
    output_reader: PinnedDirectoryReader,
    stdout: bytes,
    stderr: bytes,
) -> SamplingRefusal:
    """Keep bounded partial streams in the failed pass directory."""

    refusal.stdout = stdout
    refusal.stderr = stderr
    evidence: list[dict[str, object]] = []
    try:
        for stream, payload in (("stdout", stdout), ("stderr", stderr)):
            name = f"capture_process.{stream}.bin"
            write_regular_bytes_at(
                output_reader.descriptor,
                name,
                payload,
                mode=0o400,
            )
            evidence.append({
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            })
    except BaseException as preservation_error:
        try:
            refusal.add_note(
                "partial sampling process evidence preservation failed: "
                f"{type(preservation_error).__name__}: {preservation_error}"
            )
        except (AttributeError, TypeError):
            pass
    refusal.failure_evidence = tuple(evidence)
    return refusal


def _run_retained_pass(
    retained_binary: RetainedExecutable,
    app_args: list[str],
    out_dir: Path,
    output_reader: PinnedDirectoryReader,
    mode: int,
    *,
    selected_blocks: list[int] | None,
    retained_tracer: RetainedRegular,
    pb_lib: Path,
) -> None:
    """Execute one pass only through already-retained capabilities."""
    _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
    try:
        output_reader.verify_path()
    except ImmutableArtifactResourceError as exc:
        raise SamplingRefusal(
            "sampling_output_resource_unavailable",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    except (ImmutableArtifactError, OSError) as exc:
        raise SamplingRefusal(
            "sampling_output_generation_changed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    output_proc_path = Path(f"/proc/self/fd/{output_reader.descriptor}")
    try:
        runtime_context = retained_tracer_runtime_env(
            retained_tracer.path,
            pb_lib=pb_lib,
            runtime_manifest=default_runtime_manifest_path(
                retained_tracer.path
            ),
            retained_descriptor=retained_tracer.descriptor,
        )
    except TracerRuntimeError as exc:
        raise SamplingRefusal("tracer_runtime_unavailable", str(exc)) from exc
    result = None
    try:
        with runtime_context as retained_runtime:
            _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
            env = dict(retained_runtime.environment)
            env["CUDA_INJECTION64_PATH"] = retained_tracer.proc_path
            env["USER_DEFINED_FOLDERS"] = "1"
            env["TRACES_FOLDER"] = str(output_proc_path)
            env["TRACE_SAMPLING_MODE"] = str(mode)
            if mode == 2:
                env["TRACE_SAMPLING_SELECTED_BLOCKS"] = ",".join(
                    str(block_id) for block_id in (selected_blocks or [])
                )
            try:
                result = nvbit_process.execute_capture_process(
                    (retained_binary.proc_path, *app_args),
                    environment=env,
                    working_directory=str(retained_binary.path.parent),
                    inherited_descriptors=tuple(dict.fromkeys((
                        retained_binary.descriptor,
                        retained_tracer.descriptor,
                        output_reader.descriptor,
                        *retained_runtime.pass_fds,
                    ))),
                    timeout_seconds=_PASS_TIMEOUT_SECONDS,
                    max_output_bytes=_PASS_MAX_STDOUT_STDERR_BYTES,
                    timeout_code=_PASS_TIMEOUT_CODE,
                    launch_code=_PASS_LAUNCH_CODE,
                )
            except nvbit_process.NvbitCaptureProcessError as exc:
                if exc.process_cleanup_incomplete:
                    reason = "sampling_target_cleanup_failed"
                elif exc.code == _PASS_TIMEOUT_CODE:
                    reason = _PASS_TIMEOUT_CODE
                elif exc.code == nvbit_process.NVBIT_CAPTURE_OUTPUT_LIMIT:
                    reason = "sampling_target_output_limit"
                elif exc.code == _PASS_LAUNCH_CODE:
                    reason = _PASS_LAUNCH_CODE
                else:
                    reason = "sampling_target_execution_failed"
                refusal = SamplingRefusal(reason, exc.detail)
                refusal.process_id = exc.process_id
                raise _retain_sampling_process_failure(
                    refusal,
                    out_dir=out_dir,
                    output_reader=output_reader,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                ) from exc
            _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
    except TracerRuntimeError as exc:
        refusal = SamplingRefusal("tracer_runtime_unavailable", str(exc))
        if result is not None:
            refusal.process_id = result.process_id
            _retain_sampling_process_failure(
                refusal,
                out_dir=out_dir,
                output_reader=output_reader,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        raise refusal from exc
    assert result is not None
    # Fail closed on a nonzero target exit: a benchmark that fails a correctness
    # check while still writing a partial trace must NOT be routed as valid input.
    if result.returncode != 0:
        refusal = SamplingRefusal(
            "target_process_failed",
            f"pass mode={mode}: target exited {result.returncode}\n"
            f"stdout: {result.stdout.decode(errors='replace')[-2000:]}\n"
            f"stderr: {result.stderr.decode(errors='replace')[-2000:]}",
        )
        refusal.process_id = result.process_id
        raise _retain_sampling_process_failure(
            refusal,
            out_dir=out_dir,
            output_reader=output_reader,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    if not (output_proc_path / "dynamic_trace.pb").is_file():
        refusal = SamplingRefusal(
            "capture_trace_missing",
            f"pass mode={mode}: no dynamic_trace.pb (injection failed?)",
        )
        refusal.process_id = result.process_id
        raise _retain_sampling_process_failure(
            refusal,
            out_dir=out_dir,
            output_reader=output_reader,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
    _require_sampling_outputs_unchanged([output_reader])


def run_pass(
    binary: Path,
    app_args: list[str],
    out_dir: Path,
    mode: int,
    *,
    selected_blocks: list[int] | None = None,
    tracer_so: Path = DEFAULT_TRACER_SO,
    pb_lib: Path = DEFAULT_PB_LIB,
    retained_binary: RetainedExecutable | None = None,
    retained_tracer: RetainedRegular | None = None,
    retained_output_stack: ExitStack | None = None,
    retained_output_readers: list[PinnedDirectoryReader] | None = None,
) -> Path:
    """Run one instrumented pass from exact retained binary/tracer inodes.

    Standalone callers retain both inputs for this pass. ``orchestrate_cell``
    supplies one pair that remains open across pass 1, every pass-2 attempt, and
    the final sidecar decision.
    """
    _create_fresh_directory(out_dir, "capture_output_exists")
    if (retained_binary is None) != (retained_tracer is None):
        raise SamplingRefusal(
            "sampling_input_invalid",
            "retained binary and tracer must be supplied together",
        )

    try:
        output_reader = PinnedDirectoryReader(out_dir)
    except ImmutableArtifactResourceError as exc:
        raise SamplingRefusal(
            "sampling_output_resource_unavailable",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    except (ImmutableArtifactError, OSError) as exc:
        raise SamplingRefusal(
            "sampling_output_generation_changed",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    def execute(reader: PinnedDirectoryReader) -> None:
        if retained_binary is not None and retained_tracer is not None:
            _run_retained_pass(
                retained_binary,
                app_args,
                out_dir,
                reader,
                mode,
                selected_blocks=selected_blocks,
                retained_tracer=retained_tracer,
                pb_lib=pb_lib,
            )
            return
        try:
            with (
                open_retained_executable(binary) as owned_binary,
                open_retained_regular(
                    tracer_so, max_bytes=EXTERNAL_TRACER_MAX_BYTES
                ) as owned_tracer,
            ):
                _run_retained_pass(
                    owned_binary,
                    app_args,
                    out_dir,
                    reader,
                    mode,
                    selected_blocks=selected_blocks,
                    retained_tracer=owned_tracer,
                    pb_lib=pb_lib,
                )
        except SamplingRefusal:
            raise
        except ImmutableArtifactResourceError as exc:
            raise SamplingRefusal(
                "sampling_input_resource_unavailable",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        except (ImmutableArtifactError, OSError) as exc:
            raise SamplingRefusal(
                "sampling_input_invalid",
                f"{type(exc).__name__}: {exc}",
            ) from exc

    if retained_output_stack is not None:
        try:
            retained_reader = retained_output_stack.enter_context(output_reader)
        except ImmutableArtifactResourceError as exc:
            raise SamplingRefusal(
                "sampling_output_resource_unavailable",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        except (ImmutableArtifactError, OSError) as exc:
            raise SamplingRefusal(
                "sampling_output_generation_changed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if retained_output_readers is not None:
            retained_output_readers.append(retained_reader)
        execute(retained_reader)
        return Path(f"/proc/self/fd/{retained_reader.descriptor}")

    try:
        with output_reader as retained_reader:
            execute(retained_reader)
    except SamplingRefusal:
        raise
    except ImmutableArtifactResourceError as exc:
        raise SamplingRefusal(
            "sampling_output_resource_unavailable",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    except (ImmutableArtifactError, OSError) as exc:
        raise SamplingRefusal(
            "sampling_output_generation_changed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    return out_dir


def segments_from_stats(stats) -> dict[int, list[int]]:
    """Return the complete first-launch population, including its folded tail."""
    population = build_segment_population(stats)
    return {segment.id: list(segment.block_ids) for segment in population.segments}


def select_segments(stats, *, seed: int = 0,
                    sample_size: int = MIN_SEGMENTS) -> list[dict]:
    """Return certainty units followed by one deterministic seeded SRS draw."""
    return plan_from_stats(
        stats, seed=seed, sample_size=sample_size).sidecar_segments()


def verify_pass2_payload(
        pass2_dir: Path, grid: tuple[int, int, int], selected: list[int],
        expected_serialized_bytes: dict[int, int],
        ) -> tuple[dict[int, int], dict[int, int]]:
    """Verify the first-launch pass-2 roster and each raw protobuf byte size.

    Returns ``(records_by_block, raw_bytes_by_block)``.  The per-file byte check
    binds the producer's mode-2 ``serialized_bytes`` evidence to the actual raw
    payload, so certainty bytes cannot be redistributed between selected blocks.
    """
    threadblock_pb2 = load_dynamic_trace_bindings().threadblock_pb2
    from replay_count_producer import first_launch_dir  # type: ignore

    if (not isinstance(expected_serialized_bytes, dict)
            or any(type(block_id) is not int or block_id < 0
                   or type(size) is not int or size <= 0
                   for block_id, size in expected_serialized_bytes.items())):
        raise SamplingRefusal("capture_serialized_bytes_malformed")
    if (len(selected) != len(set(selected))
            or sorted(expected_serialized_bytes) != sorted(selected)):
        raise SamplingRefusal(
            "capture_serialized_roster_mismatch",
            f"sizes={sorted(expected_serialized_bytes)}, selected={sorted(selected)}")
    try:
        first_dir, _ = first_launch_dir(pass2_dir)
    except Exception as exc:
        raise SamplingRefusal("pass2_payload_unreadable", str(exc)) from exc
    gx, gy, gz = grid
    grid_total = gx * gy * gz
    records_by_block: dict[int, int] = {}
    raw_bytes_by_block: dict[int, int] = {}
    for p in sorted(first_dir.rglob("*.pb")):
        tb = threadblock_pb2.threadblock()
        try:
            raw = p.read_bytes()
            tb.ParseFromString(raw)
        except Exception as exc:
            raise SamplingRefusal(
                "pass2_payload_unreadable", f"{p}: {exc}") from exc
        if not tb.HasField("block_id"):
            raise SamplingRefusal(
                "pass2_block_id_missing", f"payload={p}")
        cx, cy, cz = int(tb.block_id.x), int(tb.block_id.y), int(tb.block_id.z)
        if not (0 <= cx < gx and 0 <= cy < gy and 0 <= cz < gz):
            raise SamplingRefusal(
                "pass2_block_out_of_grid",
                f"CTA ({cx},{cy},{cz}) outside grid ({gx},{gy},{gz})")
        fx = cx + cy * gx + cz * gx * gy
        if not (0 <= fx < grid_total):
            raise SamplingRefusal(
                "pass2_block_out_of_grid",
                f"block_id {fx} outside [0,{grid_total})")
        if fx in records_by_block:
            raise SamplingRefusal("pass2_duplicate_block", str(fx))
        expected_size = expected_serialized_bytes.get(fx)
        if expected_size is None:
            raise SamplingRefusal(
                "pass2_payload_membership_mismatch",
                f"dumped unexpected block_id {fx}")
        if len(raw) != expected_size:
            raise SamplingRefusal(
                "pass2_serialized_bytes_mismatch",
                f"block={fx}, payload={len(raw)}, capture={expected_size}")
        records_by_block[fx] = sum(
            len(tb.warps[wi].instructions) for wi in tb.warps)
        raw_bytes_by_block[fx] = len(raw)
    if sorted(records_by_block) != sorted(selected):
        raise SamplingRefusal(
            "pass2_payload_membership_mismatch",
            f"dumped={sorted(records_by_block)}, selected={sorted(selected)}")
    return records_by_block, raw_bytes_by_block


def verify_capture_selection(pass2_dir: Path, selected: list[int]) -> str:
    """Bind the producer's selected-ID declaration and return its raw SHA-256."""
    path = pass2_dir / CAPTURE_SIDECAR
    try:
        raw = path.read_bytes()
        capture = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SamplingRefusal("capture_sidecar_unreadable", str(exc)) from exc
    if type(capture.get("mode")) is not int or capture["mode"] != 2:
        raise SamplingRefusal(
            "capture_mode_mismatch", f"expected mode 2, got {capture.get('mode')!r}")
    declared = capture.get("selected_block_ids")
    if not isinstance(declared, list) or any(type(value) is not int or value < 0
                                             for value in declared):
        raise SamplingRefusal("capture_selected_ids_malformed")
    if len(declared) != len(set(declared)):
        raise SamplingRefusal("capture_selected_ids_duplicate")
    if sorted(declared) != sorted(selected):
        raise SamplingRefusal(
            "capture_selected_ids_mismatch",
            f"producer={sorted(declared)}, candidate={sorted(selected)}")
    return hashlib.sha256(raw).hexdigest()


def _selected_block_ids(plan) -> list[int]:
    return sorted({block_id for segment in plan.selected_segments
                   for block_id in segment.block_ids})


def _same_launch_identity(bootstrap, observed) -> None:
    fields = ("kernel_name", "grid", "block", "n_blocks")
    mismatch = {field: (getattr(bootstrap, field), getattr(observed, field))
                for field in fields
                if getattr(bootstrap, field) != getattr(observed, field)}
    bootstrap_ids = sorted(block.block_id for block in bootstrap.blocks)
    observed_ids = sorted(block.block_id for block in observed.blocks)
    if bootstrap_ids != observed_ids:
        mismatch["block_ids"] = (bootstrap_ids, observed_ids)
    if mismatch:
        raise SamplingRefusal("pass2_launch_identity_drift", repr(mismatch))


def _certainty_serialized_bytes(stats, plan) -> int:
    by_id = {block.block_id: block for block in stats.blocks}
    total = 0
    for segment in plan.certainty_segments:
        for block_id in segment.block_ids:
            block = by_id[block_id]
            value = getattr(block, "serialized_bytes", None)
            if type(value) is not int or value <= 0:
                raise SamplingRefusal(
                    "missing_exact_certainty_bytes",
                    f"block={block_id}, serialized_bytes={value!r}")
            total += value
    return total


def _selection_auxiliary(stats, selected_blocks: list[int]) -> dict:
    selected = set(selected_blocks)
    expected_full = 0
    selected_total = 0
    for block in stats.blocks:
        value = getattr(block, "sector_accesses", None)
        if type(value) is not int or value < 0:
            raise SamplingRefusal(
                "malformed_sector_auxiliary",
                f"block={block.block_id}, sector_accesses={value!r}")
        expected_full += value
        if block.block_id in selected:
            selected_total += value
    return {
        "name": "l2_sector_transactions_32b",
        "sector_bytes": 32,
        "scope": "pass2_all_blocks",
        "expected_full": expected_full,
        "selected": selected_total,
    }


def _orchestrate_cell_retained(
    binary: Path,
    app_args: list[str],
    work_dir: Path,
    *,
    retained_binary: RetainedExecutable,
    retained_tracer: RetainedRegular,
    retained_output_stack: ExitStack,
    retained_output_readers: list[PinnedDirectoryReader],
    args_hash: str = "0" * 8,
    cooperative: bool = False,
    seed: int = 0,
    sample_size: int = MIN_SEGMENTS,
    max_trace_bytes: int | None = None,
    max_reconcile_attempts: int = 3,
    tracer_so: Path = DEFAULT_TRACER_SO,
    pb_lib: Path = DEFAULT_PB_LIB,
) -> dict:
    """Compute one sampled sidecar while retained inputs remain live."""
    cell_id = binary.stem
    out = work_dir / "trace_sampling.json"
    compressed_out = work_dir / "pass2_zst"
    _refuse_existing_output(out, "sidecar_output_exists")
    _refuse_existing_output(compressed_out, "compression_output_exists")
    _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
    pass1 = work_dir / "pass1"
    pass1_access = run_pass(
        binary,
        app_args,
        pass1,
        mode=1,
        tracer_so=tracer_so,
        pb_lib=pb_lib,
        retained_binary=retained_binary,
        retained_tracer=retained_tracer,
        retained_output_stack=retained_output_stack,
        retained_output_readers=retained_output_readers,
    ) or pass1
    _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
    _require_sampling_outputs_unchanged(retained_output_readers)
    stats = cell_stats_from_capture(
        pass1_access, cell_id, cooperative=cooperative
    )
    _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
    gate = classify(stats, sample_size=sample_size)

    if gate.method == M_SUBSET:
        if type(max_trace_bytes) is not int or max_trace_bytes <= 0:
            raise SamplingRefusal(
                "missing_cell_trace_budget",
                "subset capture requires an explicit positive max_trace_bytes")
        if type(max_reconcile_attempts) is not int or max_reconcile_attempts <= 0:
            raise SamplingRefusal(
                "invalid_reconcile_attempts", repr(max_reconcile_attempts))
        bootstrap_force = tuple(gate.force_include_blocks)
        try:
            candidate_plan = plan_from_stats(
                stats, seed=seed, sample_size=sample_size,
                force_include_block_ids=bootstrap_force)
        except SelectionPlanError as exc:
            raise SamplingRefusal("bootstrap_selection_refused", str(exc)) from exc
        candidate = _selected_block_ids(candidate_plan)

        accepted = None
        for attempt in range(1, max_reconcile_attempts + 1):
            pass2 = work_dir / f"pass2_attempt_{attempt}"
            pass2_access = run_pass(
                binary,
                app_args,
                pass2,
                mode=2,
                selected_blocks=candidate,
                tracer_so=tracer_so,
                pb_lib=pb_lib,
                retained_binary=retained_binary,
                retained_tracer=retained_tracer,
                retained_output_stack=retained_output_stack,
                retained_output_readers=retained_output_readers,
            ) or pass2
            _require_sampling_inputs_unchanged(
                retained_binary, retained_tracer
            )
            _require_sampling_outputs_unchanged(retained_output_readers)
            pass2_stats = cell_stats_from_capture(
                pass2_access, cell_id, cooperative=cooperative)
            _require_sampling_inputs_unchanged(
                retained_binary, retained_tracer
            )
            _same_launch_identity(stats, pass2_stats)
            capture_stats_sha256 = verify_capture_selection(
                pass2_access, candidate
            )
            candidate_set = set(candidate)
            candidate_serialized_bytes = {
                block.block_id: block.serialized_bytes
                for block in pass2_stats.blocks
                if block.block_id in candidate_set
            }
            actual_records, actual_raw_bytes = verify_pass2_payload(
                pass2_access,
                pass2_stats.grid,
                candidate,
                candidate_serialized_bytes,
            )
            _require_sampling_inputs_unchanged(
                retained_binary, retained_tracer
            )

            provisional_gate = classify(
                pass2_stats, sample_size=sample_size,
                force_include_block_ids=bootstrap_force)
            if provisional_gate.method != M_SUBSET:
                raise SamplingRefusal(
                    "pass2_gate_forced",
                    f"attempt={attempt}, verdict={provisional_gate.verdict}")
            final_force = tuple(provisional_gate.force_include_blocks)
            try:
                desired_plan = plan_from_stats(
                    pass2_stats, seed=seed, sample_size=sample_size,
                    force_include_block_ids=final_force)
            except SelectionPlanError as exc:
                raise SamplingRefusal(
                    "pass2_selection_refused",
                    f"attempt={attempt}, {exc}") from exc
            desired = _selected_block_ids(desired_plan)
            certainty_bytes = _certainty_serialized_bytes(
                pass2_stats, desired_plan)
            final_gate = classify(
                pass2_stats, sample_size=sample_size,
                force_include_block_ids=final_force,
                certainty_bytes=certainty_bytes,
                max_trace_bytes=max_trace_bytes)
            if final_gate.method != M_SUBSET:
                # The final, exact raw-PB certainty budget is a real gate.  A
                # reduced-shape label does not exist here, so emit no sampled
                # point and return the established typed unavailable method.
                sidecar = build_sidecar(
                    cell_id=cell_id,
                    kernel_name=pass2_stats.kernel_name,
                    args_hash=args_hash,
                    stats=pass2_stats,
                    gate=final_gate,
                    compression=None,
                    records_captured=0,
                    tracer_version=5,
                    stats_pass_version=pass2_stats.stats_pass_version,
                    method="unavailable",
                    reason="gate_forced",
                )
                _require_sampling_inputs_unchanged(
                    retained_binary, retained_tracer
                )
                return sidecar
            if desired != candidate:
                candidate = desired
                bootstrap_force = final_force
                continue
            accepted = (attempt, pass2_access, pass2_stats, desired_plan, final_gate,
                        certainty_bytes, capture_stats_sha256, actual_records,
                        actual_raw_bytes)
            break

        if accepted is None:
            raise SamplingRefusal(
                "placement_reconciliation_exhausted",
                f"max_attempts={max_reconcile_attempts}")

        (attempts, pass2_access, final_stats, plan, gate, certainty_bytes,
         capture_stats_sha256, actual_records, actual_raw_bytes) = accepted
        segments = plan.sidecar_segments()
        selected = _selected_block_ids(plan)
        selected_set = set(selected)
        selected_payload_bytes = sum(actual_raw_bytes.values())
        if selected_payload_bytes > max_trace_bytes:
            raise SamplingRefusal(
                "selected_payload_over_budget",
                f"raw={selected_payload_bytes}, max={max_trace_bytes}")
        _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
        _create_fresh_directory(compressed_out, "compression_output_exists")
        comp = compress_trace_dir(pass2_access, compressed_out)
        _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
        selected_serialized_bytes = sum(
            block.serialized_bytes for block in final_stats.blocks
            if block.block_id in selected_set)
        if (comp.raw_bytes != selected_payload_bytes
                or selected_payload_bytes != selected_serialized_bytes):
            raise SamplingRefusal(
                "serialized_payload_size_mismatch",
                f"compressed_input={comp.raw_bytes}, payload={selected_payload_bytes}, "
                f"counting={selected_serialized_bytes}")
        captured = sum(actual_records.values())
        strata = plan.sidecar_strata()
        estimator = {
            "mode": "weighted_full_shape_estimate",
            "targets": ["sectors", "l2_sector_hit_rate"],
            "not_recoverable": ["cycles", "sol_pct_timing"],
            # Finalized at scoring time (per-segment sub-trace replay seam).
            "convergence": {"status": "budget_limited_wide_ci"},
        }
        sidecar = build_sidecar(
            cell_id=cell_id,
            kernel_name=final_stats.kernel_name,
            args_hash=args_hash,
            stats=final_stats,
            gate=gate,
            compression=comp,
            records_captured=captured,
            tracer_version=5,
            stats_pass_version=final_stats.stats_pass_version,
            method=M_SUBSET,
            reason="byte_budget",
            segments=segments,
            strata=strata,
            estimator=estimator,
            decompressed_payload_sha256=comp.decompressed_sha256,
            seed=seed,
            rng_algo="pcg64",
            selected_segment_ids=[
                segment.id for segment in plan.selected_segments
            ],
            certainty_segments=[
                segment.id for segment in plan.certainty_segments
            ],
            force_include_blocks=list(plan.population.force_include_block_ids),
            selection_reconciliation={
                "status": "fixed_point",
                "attempts": attempts,
                "max_attempts": max_reconcile_attempts,
                "capture_stats_sha256": capture_stats_sha256,
            },
            selection_budget={
                "max_trace_bytes": max_trace_bytes,
                "basis": "raw_threadblock_pb",
                "certainty_fraction_numerator": 2,
                "certainty_fraction_denominator": 5,
                "certainty_bytes": certainty_bytes,
            },
            selection_auxiliary=_selection_auxiliary(final_stats, selected),
        )
    else:
        # Gate-forbidden -> typed unavailable (a reduced shape needs a separate
        # shape-adjust wrapper; never silently subset a forbidden cell).
        sidecar = build_sidecar(
            cell_id=cell_id,
            kernel_name=stats.kernel_name,
            args_hash=args_hash,
            stats=stats,
            gate=gate,
            compression=None,
            records_captured=0,
            tracer_version=5,
            stats_pass_version=stats.stats_pass_version,
            method="unavailable",
            reason="gate_forced",
        )

    _require_sampling_inputs_unchanged(retained_binary, retained_tracer)
    _require_sampling_outputs_unchanged(retained_output_readers)
    return sidecar


def orchestrate_cell(
    binary: Path,
    app_args: list[str],
    work_dir: Path,
    *,
    args_hash: str = "0" * 8,
    cooperative: bool = False,
    seed: int = 0,
    sample_size: int = MIN_SEGMENTS,
    max_trace_bytes: int | None = None,
    max_reconcile_attempts: int = 3,
    tracer_so: Path = DEFAULT_TRACER_SO,
    pb_lib: Path = DEFAULT_PB_LIB,
) -> dict:
    """Run both passes under one exact retained binary/tracer generation.

    Publication occurs while both input capabilities remain live. The exact
    published sidecar generation is then retained while those inputs close and
    is revalidated as part of the return boundary. Any post-dispatch failure
    retires only the exact payload and, once known, exact generation owned by
    this attempt.
    """
    out = work_dir / "trace_sampling.json"
    compressed_out = work_dir / "pass2_zst"
    _refuse_existing_output(out, "sidecar_output_exists")
    _refuse_existing_output(compressed_out, "compression_output_exists")

    expected_payload: bytes | None = None
    published_snapshot = None
    publication_state: SidecarPublicationState | None = None
    publication_attempted = False
    input_stack = ExitStack()
    retained_output_readers: list[PinnedDirectoryReader] = []

    def close_inputs_preserving(primary: BaseException) -> None:
        try:
            input_stack.__exit__(
                type(primary), primary, primary.__traceback__
            )
        except BaseException as cleanup:
            try:
                primary.add_note(
                    "trace-sampling input cleanup also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            except (AttributeError, TypeError):
                pass

    def bind_evidence_state(primary: BaseException, state: str) -> None:
        if isinstance(primary, SamplingRefusal):
            _bind_sidecar_evidence_state(primary, state)
            return
        try:
            primary.add_note(f"sidecar_evidence_state={state}")
        except (AttributeError, TypeError):
            pass

    def close_publication_state_preserving(
        primary: BaseException,
    ) -> None:
        if publication_state is None:
            return
        try:
            publication_state.close()
        except BaseException as cleanup:
            try:
                primary.add_note(
                    "retained sidecar publication close also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            except (AttributeError, TypeError):
                pass

    def retire_failed_publication(primary: BaseException) -> None:
        if not publication_attempted or expected_payload is None:
            close_publication_state_preserving(primary)
            return

        snapshot = (
            publication_state.snapshot
            if publication_state is not None
            and publication_state.complete
            else published_snapshot
        )
        if snapshot is None:
            bind_evidence_state(
                primary,
                "publication_identity_incomplete_active_locator_untouched",
            )
            close_publication_state_preserving(primary)
            return

        retired = None
        retirement_error: BaseException | None = None
        try:
            retired = retire_regular_file(
                out,
                expected_payload=expected_payload,
                expected_snapshot=snapshot,
            )
        except BaseException as cleanup:
            retirement_error = cleanup
            try:
                primary.add_note(
                    "trace-sampling sidecar retirement also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            except (AttributeError, TypeError):
                pass

        if retired is not None:
            bind_evidence_state(primary, "expected_payload_retired")
            close_publication_state_preserving(primary)
            return

        salvaged = None
        salvage_error: BaseException | None = None
        if publication_state is not None and publication_state.complete:
            try:
                publication_state.verify_retained_payload()
                salvaged = salvage_retained_regular_descriptor(
                    out,
                    publication_state.descriptor,
                    expected_payload=expected_payload,
                    expected_snapshot=snapshot,
                )
            except BaseException as cleanup:
                salvage_error = cleanup
                try:
                    primary.add_note(
                        "retained sidecar publication salvage also failed: "
                        f"{type(cleanup).__name__}: {cleanup}"
                    )
                except (AttributeError, TypeError):
                    pass

        if salvaged is not None:
            bind_evidence_state(
                primary,
                "retained_payload_salvaged_active_locator_untouched",
            )
            try:
                primary.add_note(
                    "retained sidecar publication evidence preserved as "
                    f"{salvaged}"
                )
            except (AttributeError, TypeError):
                pass
        elif retirement_error is None and salvage_error is None:
            bind_evidence_state(primary, "active_locator_absent")
        else:
            bind_evidence_state(primary, "active_locator_not_retired")
        close_publication_state_preserving(primary)

    try:
        retained_binary = input_stack.enter_context(
            open_retained_executable(binary)
        )
        retained_tracer = input_stack.enter_context(
            open_retained_regular(
                tracer_so, max_bytes=EXTERNAL_TRACER_MAX_BYTES
            )
        )
        sidecar = _orchestrate_cell_retained(
            binary,
            app_args,
            work_dir,
            retained_binary=retained_binary,
            retained_tracer=retained_tracer,
            retained_output_stack=input_stack,
            retained_output_readers=retained_output_readers,
            args_hash=args_hash,
            cooperative=cooperative,
            seed=seed,
            sample_size=sample_size,
            max_trace_bytes=max_trace_bytes,
            max_reconcile_attempts=max_reconcile_attempts,
            tracer_so=tracer_so,
            pb_lib=pb_lib,
        )
        _require_sampling_inputs_unchanged(
            retained_binary, retained_tracer
        )
        _require_sampling_outputs_unchanged(retained_output_readers)
        expected_payload = (
            json.dumps(sidecar, indent=2) + "\n"
        ).encode("utf-8")
        publication_state = SidecarPublicationState(expected_payload)
        publication_attempted = True
        try:
            write_sidecar(
                sidecar,
                out,
                publication_state=publication_state,
            )
        except SamplingRefusal:
            raise
        except Exception as exc:
            raise SamplingRefusal(
                "sidecar_publication_failed",
                f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            if (
                not publication_state.complete
                or publication_state.snapshot is None
            ):
                raise RuntimeError(
                    "sidecar writer returned without sealed publication identity"
                )
            publication_state.verify_retained_payload()
        except Exception as exc:
            raise SamplingRefusal(
                "sidecar_publication_changed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        published_snapshot = publication_state.snapshot
        _require_sampling_outputs_unchanged(retained_output_readers)

        try:
            retained_sidecar = open_retained_regular(out)
        except ImmutableArtifactResourceError as exc:
            raise SamplingRefusal(
                "sidecar_publication_resource_unavailable",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        except (ImmutableArtifactError, OSError) as exc:
            raise SamplingRefusal(
                "sidecar_publication_changed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if retained_sidecar.snapshot != published_snapshot:
            refusal = SamplingRefusal(
                "sidecar_publication_changed",
                "active sidecar locator differs from writer publication identity",
            )
            try:
                retained_sidecar.close()
            except BaseException as cleanup:
                try:
                    refusal.add_note(
                        "mismatched retained sidecar close also failed: "
                        f"{type(cleanup).__name__}: {cleanup}"
                    )
                except (AttributeError, TypeError):
                    pass
            raise refusal

        try:
            with _retained_sidecar_return_boundary(
                retained_sidecar,
                publication_state,
            ) as retained_publication:
                _require_sampling_sidecar_unchanged(
                    retained_publication, expected_payload
                )
                _require_sampling_inputs_unchanged(
                    retained_binary, retained_tracer
                )
                _require_sampling_outputs_unchanged(
                    retained_output_readers
                )
                try:
                    input_stack.close()
                except ImmutableArtifactResourceError as exc:
                    raise SamplingRefusal(
                        "sampling_input_resource_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    ) from exc
                except (ImmutableArtifactError, OSError) as exc:
                    raise SamplingRefusal(
                        "sampling_input_invalid",
                        f"{type(exc).__name__}: {exc}",
                    ) from exc
                _require_sampling_sidecar_unchanged(
                    retained_publication, expected_payload
                )
                try:
                    returned_payload = (
                        json.dumps(sidecar, indent=2) + "\n"
                    ).encode("utf-8")
                except Exception as exc:
                    raise SamplingRefusal(
                        "sidecar_return_payload_changed",
                        f"{type(exc).__name__}: {exc}",
                    ) from exc
                if returned_payload != expected_payload:
                    raise SamplingRefusal(
                        "sidecar_return_payload_changed",
                        "returned sidecar serialization changed after publication",
                    )
                return sidecar
        except SamplingRefusal:
            raise
        except ImmutableArtifactResourceError as exc:
            raise SamplingRefusal(
                "sidecar_publication_resource_unavailable",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        except (ImmutableArtifactError, OSError) as exc:
            raise SamplingRefusal(
                "sidecar_publication_changed",
                f"{type(exc).__name__}: {exc}",
            ) from exc
    except SamplingRefusal as primary:
        close_inputs_preserving(primary)
        retire_failed_publication(primary)
        raise
    except ImmutableArtifactResourceError as exc:
        refusal = SamplingRefusal(
            "sampling_input_resource_unavailable",
            f"{type(exc).__name__}: {exc}",
        )
        close_inputs_preserving(refusal)
        retire_failed_publication(refusal)
        raise refusal from exc
    except (ImmutableArtifactError, OSError) as exc:
        refusal = SamplingRefusal(
            "sampling_input_invalid",
            f"{type(exc).__name__}: {exc}",
        )
        close_inputs_preserving(refusal)
        retire_failed_publication(refusal)
        raise refusal from exc
    except BaseException as primary:
        close_inputs_preserving(primary)
        retire_failed_publication(primary)
        raise
