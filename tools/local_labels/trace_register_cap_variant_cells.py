"""Variant-aware NVBit trace producer for the register-spill build-variant family.

Mirrors ``activation_trace_orchestrator.py`` (a ``tools/local_labels`` NVBit
trace producer) but for the ``register_spill_local_mem`` register-cap build
variants:

- selects the PRE-BUILT per-cap binary
  ``spill_variants/<slug>_run__cap<cap>`` (built via ``make spill-variants``),
  NOT the default binary;
- writes under a NON-COLLIDING per-variant subdir
  ``<cell_root>/variants/register_cap_<cap>/`` so two caps never share a path
  (the exact layout the variant-aware staging tool consumes);
- derives ``kernel_signature_sha256`` from the LIVE ``dynamic_trace.pb`` via the
  shared ``nvbit_trace_summary_adapter`` (the real launch geometry the kernel
  executed, NOT the stale shape-TOML ``launch_config``);
- emits ``trace_<shape>.json`` carrying the canonical ``register_cap`` /
  ``variant_id`` + the real per-cap ``binary_sha256`` + the live signature +
  the retained tracer hash, portable admitted-content identities, and separate
  local filesystem-generation identity.

NVBit tracing only -- this producer invokes the NVBit tracer, NEVER NCU, and
writes no ``.ncu-rep``. The NVBit tracer is an authorized trace input; the
producer lives under ``tools/local_labels/`` to stay co-located with the
rest of the build-variant machinery.

Identity note: the NVBit tracer reports ``regs_per_thread = 0`` for these
kernels (``cuFuncGetAttribute(BINARY_VERSION)`` returns 0 -> fallback), so the
LIVE ``kernel_signature_sha256`` is IDENTICAL across caps (same kernel_name,
real grid/block, shared=0, regs=0). Register-cap variants are therefore
distinguished by ``binary_sha256`` (distinct per-cap binaries) +
``register_cap`` / ``variant_id``, NOT by the signature; the evidence-manifest
join key includes ``binary_sha256``, so distinct caps stay distinct cells.

Verified live on the RTX 3090 Ti (sm_86). A sm_120 variant build is a cross-host
follow-on (``make spill-variants`` targets sm_86); the producer is parameterized
by ``--sku`` / ``--arch-dir`` but has only been exercised on sm_86.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from contextlib import ExitStack
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from variant_identity import variant_identity  # noqa: E402
from roster_path_resolver import resolve_cell_root  # noqa: E402
from verify_spill_sweep import LADDER_CAPS  # noqa: E402
from workloads.simbench_bw.nvbit_trace_summary_adapter import (  # noqa: E402
    NvbitProtobufBindingsMissing,
    emit_admitted_nvbit_trace_summary,
)
from tools.nvbit_tracer.runtime_env import tracer_runtime_env  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    TraceAdmission,
    TraceAdmissionError,
    TraceLaunchSummary,
    open_admitted_trace_sources,
    require_trace_admission,
)
from tools.local_labels.activation_trace_orchestrator import (  # noqa: E402
    _RetainedActivationPublication,
    _lock_activation_cell,
    _new_capture_attempt_id,
    _raw_generation_record,
    _release_activation_lock,
    _retire_matching_locator,
    _trace_generation_commit_limits,
    _verify_committed_trace_generation,
)
from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ImmutableDirectoryGeneration,
    PinnedDirectoryReader,
    create_exclusive_directory_stage,
    inspect_immutable_directory_generation,
    open_retained_executable,
    open_retained_regular,
    publish_regular_file,
    retire_regular_file,
    strict_json_loads,
)
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_write_allowed,
)

_TRACER_SO = (
    _REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
)
_SLUG = "register_spill_local_mem"
_KERNEL_DIR = _REPO_ROOT / "workloads" / "simbench_bw" / "l1" / _SLUG
_SHAPES_TOML = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "shapes" / f"{_SLUG}.toml"
)
_DEFAULT_SEED = 42
_DEVICE_NAME_BY_SKU = {
    "rtx_3090_ti": "NVIDIA GeForce RTX 3090 Ti",
    "rtx_5060_laptop": "NVIDIA GeForce RTX 5060 Laptop GPU",
}
_STAGING_ROOT = _REPO_ROOT / "results" / "nvbit_capture_staging"
_DEFAULT_OUTPUT_ROOT = _STAGING_ROOT / "register_cap_variants"


def _assert_staging_output(path: Path | str, *, operation: str) -> Path:
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


def _variant_output_plan(
    *,
    slug: str,
    shape_point: str,
    register_cap: int,
    sku: str,
    arch_dir: str,
    output_root: Path | str = _DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    """Resolve and clear every output before Git or capture subprocesses."""

    variant_id, _cap, err = variant_identity({"register_cap": register_cap})
    if err is not None:
        raise ValueError(
            f"trace_register_cap_variant_cells: invalid register cap: {err}"
        )
    root = _assert_staging_output(output_root, operation="select variant output root")
    source_cell = resolve_cell_root(
        slug=slug, shape_point=shape_point, sku=sku, arch_dir=arch_dir,
    )
    try:
        relative_cell = source_cell.resolve(strict=False).relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ProtectedPathError(
            f"roster cell root escapes repository: {source_cell}"
        ) from exc
    variant_dir = root / relative_cell / "variants" / str(variant_id)
    sidecar = variant_dir / f"trace_{shape_point}.json"
    candidates = {
        "output_root": root,
        "variant_dir": variant_dir,
        "traces_folder": variant_dir / f"trace_{shape_point}",
        "summary_generation_root": variant_dir / "summary_generations",
        "attempt_record_root": variant_dir / "capture_attempts",
        "workload_output_root": variant_dir / "workload_output",
        "sidecar_path": sidecar,
        "summary_path": variant_dir / "nvbit_trace_summary_0_0.json",
        "tmp_path": sidecar.with_name(sidecar.name + ".tmp"),
    }
    return {
        name: _assert_staging_output(path, operation=f"prepare {name}")
        for name, path in candidates.items()
    }


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown_source_commit"


# Every tracked Python or data input that defines capture arguments, admission,
# summary bytes, variant identity, or publication semantics. A clean activation
# requires retained bytes for this exact set to match one stable Git tree.
_PRODUCER_RELPATH = "tools/local_labels/trace_register_cap_variant_cells.py"
_SOURCE_COMMIT_AUTHORITY_PATHS = (
    _PRODUCER_RELPATH,
    "tools/local_labels/activation_trace_orchestrator.py",
    "tools/local_labels/immutable_artifact.py",
    "tools/local_labels/protected_paths.py",
    "tools/local_labels/roster_path_resolver.py",
    "tools/local_labels/variant_identity.py",
    "tools/local_labels/verify_spill_sweep.py",
    "tools/nvbit_tracer/import_trace_contract.py",
    "tools/nvbit_tracer/proto_binding_loader.py",
    "tools/nvbit_tracer/protobuf_wire_preflight.py",
    "tools/nvbit_tracer/runtime_env.py",
    "tools/nvbit_tracer/trace_admission.py",
    "workloads/simbench_bw/nvbit_trace_summary_adapter.py",
    "workloads/simbench_bw/shapes/register_spill_local_mem.toml",
)


def _resolve_source_commit(
    retained_authority: dict[str, object],
    *,
    allow_dirty: bool = False,
) -> tuple[str, bool]:
    """Bind retained capture authority bytes to one stable source commit."""
    expected = set(_SOURCE_COMMIT_AUTHORITY_PATHS)
    if set(retained_authority) != expected or len(retained_authority) != len(expected):
        missing = sorted(expected - set(retained_authority))
        extra = sorted(set(retained_authority) - expected)
        raise RuntimeError(
            "register-cap source-commit authority set is incomplete: "
            f"missing={missing}, extra={extra}"
        )
    head = _git_head()
    if not head or head == "unknown_source_commit":
        raise RuntimeError(
            "cannot resolve git HEAD; the trace's source_commit would be "
            "unverifiable -- run inside a git work tree"
        )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "status",
            "--porcelain",
            "--",
            *_SOURCE_COMMIT_AUTHORITY_PATHS,
        ],
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(
            "cannot verify register-cap source-commit cleanliness"
        )
    dirty_detail = status.stdout.strip()
    dirty_override = bool(dirty_detail)
    if dirty_override and not allow_dirty:
        raise RuntimeError(
            "register-cap source-commit authority is dirty or untracked: "
            f"{dirty_detail!r}"
        )

    for relative_path in _SOURCE_COMMIT_AUTHORITY_PATHS:
        retained = retained_authority[relative_path]
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "show",
                f"{head}:{relative_path}",
            ],
            capture_output=True,
        )
        if committed.returncode != 0:
            raise RuntimeError(
                f"register-cap authority is absent from source_commit {head}: "
                f"{relative_path}"
            )
        retained_payload = retained.read_bytes()
        if not dirty_override and committed.stdout != retained_payload:
            raise RuntimeError(
                "retained register-cap authority bytes disagree with "
                f"source_commit for {relative_path}"
            )
        retained.verify_unchanged()
    if _git_head() != head:
        raise RuntimeError(
            "register-cap source_commit changed during authority verification"
        )
    return head, dirty_override


def _verify_source_commit_head(source_commit: str) -> None:
    if _git_head() != source_commit:
        raise RuntimeError(
            "register-cap source_commit changed during capture publication"
        )


def _run_workload(
    cmd: list[str],
    env: dict[str, str],
    cwd: str,
    *,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run the retained NVBit workload; split out for offline interception."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        pass_fds=pass_fds,
    )


def _require_single_admitted_launch(
    admission: TraceAdmission,
    *,
    producer: str,
) -> TraceLaunchSummary:
    """Return launch zero only for an internally consistent singleton trace."""

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


_SUMMARY_NAME = "nvbit_trace_summary_0_0.json"
_SUMMARY_GENERATION_LIMITS = {
    "max_files": 1,
    "max_directories": 1,
    "max_total_file_bytes": 4 * 1024 * 1024,
    "max_total_relative_path_bytes": 4096,
}


def _verify_committed_summary_generation(
    generation: ImmutableDirectoryGeneration,
) -> None:
    observed = inspect_immutable_directory_generation(
        generation.path,
        require_nonempty=True,
        **_SUMMARY_GENERATION_LIMITS,
    )
    if observed != generation or tuple(
        identity.relative_path for identity in observed.files
    ) != (_SUMMARY_NAME,):
        raise ImmutableArtifactError(
            "committed register-cap summary generation changed after publication"
        )


def _variant_binary_path(slug: str, register_cap: int) -> Path:
    """The pre-built per-cap binary path (built by ``make spill-variants``)."""
    return _KERNEL_DIR / "spill_variants" / f"{slug}_run__cap{register_cap}"


def _kernel_args_for_shape(
    shape_point: str,
    seed: int,
    *,
    retained_shapes_payload: bytes | None = None,
) -> list[str]:
    """Build workload arguments from one retained shape-authority snapshot."""
    if retained_shapes_payload is None:
        with _SHAPES_TOML.open("rb") as fh:
            data = tomllib.load(fh)
    else:
        try:
            data = tomllib.loads(retained_shapes_payload.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(
                f"invalid retained shape TOML {_SHAPES_TOML}: {exc}"
            ) from exc
    for section in ("trace_shapes", "timing_shapes", "correctness_shapes"):
        for shape in data.get(section, []):
            if shape.get("name") == shape_point:
                axes = shape.get("axes", {})
                return [
                    "--warps", str(int(axes["warps"])),
                    "--live-regs", str(int(axes["live_regs"])),
                    "--iterations", str(int(axes["iterations"])),
                    "--seed", str(int(seed)),
                    "--shape-point", shape_point,
                ]
    raise ValueError(
        f"trace_register_cap_variant_cells: shape_point {shape_point!r} not "
        f"found in {_SHAPES_TOML}"
    )


def _build_variant_sidecar(
    *,
    slug: str,
    shape_point: str,
    register_cap: int,
    variant_id: str,
    trace_capture_attempt_id: str,
    raw_trace_generation: dict[str, object],
    summary_generation: dict[str, object],
    arch_dir: str,
    sku: str,
    binary_sha256: str,
    kernel_signature_sha256: str,
    kernel_name: str,
    source_commit: str,
    trace_pb_relpath: str,
    trace_bytes: int,
    nvbit_tracer_build_hash: str,
    admission: TraceAdmission,
    wall_time_s: float,
    expected_record_count: int,
    admitted_record_count: int,
    dirty_override: bool = False,
) -> dict:
    """Assemble the per-variant ``trace_<shape>.json`` payload. Pure (no I/O) so
    the field contract is unit-testable. Carries the canonical build-variant
    identity (``register_cap`` / ``variant_id``) the variant-aware staging tool
    requires, the four join-key fields (slug, shape_point, binary_sha256,
    kernel_signature_sha256), the retained tracer hash, the portable admitted
    content identities, and the separate local filesystem-generation identity.
    When ``dirty_override`` is set (an allow_dirty
    run over uncommitted provenance-relevant files), a
    ``provenance_dirty_override: true`` marker is added so downstream treats the
    trace as NON-ADMISSIBLE; a clean run omits the field (byte-minimal)."""
    drop_count = max(expected_record_count - admitted_record_count, 0)
    drop_rate = (
        drop_count / expected_record_count if expected_record_count else 0.0
    )
    sidecar = {
        "schema_version": 1,
        "slug": slug,
        "shape_set": "trace_shapes",
        "shape_point": shape_point,
        "register_cap": int(register_cap),
        "variant_id": variant_id,
        "trace_capture_attempt_id": trace_capture_attempt_id,
        "raw_trace_generation": raw_trace_generation,
        "summary_generation": summary_generation,
        "trace_pb_path": trace_pb_relpath,
        "trace_bytes": int(trace_bytes),
        "wall_time_s": wall_time_s,
        "tracer_so_path": str(_TRACER_SO.relative_to(_REPO_ROOT)),
        "nvbit_tracer_build_hash": nvbit_tracer_build_hash,
        **_trace_admission_identity_fields(admission),
        "expected_record_count": int(expected_record_count),
        "admitted_record_count": int(admitted_record_count),
        "actual_drop_rate": drop_rate,
        "channel_overflow_dropped": 0,
        "ring_buffer_loss_dropped": 0,
        "decoder_gap_dropped": 0,
        "expected_record_mismatch_dropped": drop_count,
        "binary_sha256": binary_sha256,
        "source_commit": source_commit,
        "kernel_signature_sha256": kernel_signature_sha256,
        "kernel_name": kernel_name,
        "device_name": _DEVICE_NAME_BY_SKU.get(sku, sku),
        "arch": arch_dir,
        # The producer emits the raw .pb + per-launch summary but NOT the 4
        # predictor-side feature derivations (those belong to a downstream
        # pipeline that consumes dynamic_trace.pb directly).
        "hrd_skip_reason": "register_cap_variant_trace_emit_only",
    }
    if dirty_override:
        # Mark a dirty-override trace NON-ADMISSIBLE: it was emitted over
        # uncommitted provenance-relevant files, so its source_commit cannot
        # reproduce the emitting logic.
        sidecar["provenance_dirty_override"] = True
    return sidecar


def trace_variant_cell(
    *,
    shape_point: str,
    register_cap: int,
    slug: str = _SLUG,
    sku: str = "rtx_3090_ti",
    arch_dir: str = "sm_86",
    seed: int = _DEFAULT_SEED,
    allow_dirty: bool = False,
    output_root: Path | str = _DEFAULT_OUTPUT_ROOT,
    _precleared_plan: dict[str, Path] | None = None,
) -> dict:
    """Run the NVBit tracer against the pre-built per-cap binary for one
    ``(slug, shape_point, register_cap)`` cell and emit the per-variant trace
    JSON under the mirrored staging cell layout at
    ``<output_root>/<cell-relative>/variants/register_cap_<cap>/``.

    Fails closed (typed exception, no JSON written) on: a noncanonical /
    negative cap; a non-reproducible provenance state (HEAD does not contain the
    producer, or provenance-relevant files are uncommitted -- unless
    ``allow_dirty``); a missing pre-built per-cap binary; a missing NVBit tracer
    .so; a nonzero workload exit; a missing or inadmissible trace; a trace with
    anything other than one deterministically indexed launch; or absent protobuf
    bindings (the live signature cannot be derived). The trace sidecar is
    atomically replaced only after every check passes, and the prior sidecar is
    removed before launch, so no consumable stale trace_<shape>.json can survive
    a failed rerun."""
    variant_id, cap, err = variant_identity({"register_cap": register_cap})
    if err is not None:
        raise ValueError(
            f"trace_register_cap_variant_cells: register_cap "
            f"{register_cap!r} is not a valid build-variant cap: {err}"
        )
    plan = _precleared_plan or _variant_output_plan(
        slug=slug,
        shape_point=shape_point,
        register_cap=cap,
        sku=sku,
        arch_dir=arch_dir,
        output_root=output_root,
    )
    plan = {
        name: _assert_staging_output(path, operation=f"recheck {name}")
        for name, path in plan.items()
    }
    if not _TRACER_SO.is_file():
        raise FileNotFoundError(
            f"NVBit tracer .so missing at {_TRACER_SO}; restore the "
            "byte-exact preserved artifact before capture."
        )
    binary = _variant_binary_path(slug, cap)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"per-cap binary missing or not executable: {binary}; run "
            f"`make -C {_KERNEL_DIR} spill-variants` to build the cap ladder "
            f"before tracing."
        )

    variant_dir = plan["variant_dir"]
    generation_root = plan["traces_folder"]
    sidecar_path = plan["sidecar_path"]
    summary_path = plan["summary_path"]
    summary_generation_root = plan["summary_generation_root"]
    attempt_record_root = plan["attempt_record_root"]
    workload_output_root = plan["workload_output_root"]
    attempt_id = _new_capture_attempt_id()

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
        retained_shapes = capture_sources.enter_context(
            open_retained_regular(_SHAPES_TOML)
        )
        retained_authority: dict[str, object] = {
            "workloads/simbench_bw/shapes/register_spill_local_mem.toml": (
                retained_shapes
            ),
        }
        for relative_path in _SOURCE_COMMIT_AUTHORITY_PATHS:
            if relative_path in retained_authority:
                continue
            retained_authority[relative_path] = capture_sources.enter_context(
                open_retained_regular(_REPO_ROOT / relative_path)
            )
        source_commit, dirty_override = _resolve_source_commit(
            retained_authority,
            allow_dirty=allow_dirty,
        )
        args = _kernel_args_for_shape(
            shape_point,
            seed,
            retained_shapes_payload=retained_shapes.read_bytes(),
        )
        binary_sha = retained_binary.sha256
        tracer_build_sha = retained_tracer.sha256

        variant_dir.mkdir(parents=True, exist_ok=True)
        locked_variant = lock_sources.enter_context(
            PinnedDirectoryReader(variant_dir)
        )
        _lock_activation_cell(locked_variant)
        lock_hold_descriptor = os.dup(locked_variant.descriptor)
        lock_hold_sources.callback(
            _release_activation_lock, lock_hold_descriptor
        )
        generation_root.mkdir(parents=True, exist_ok=True)
        summary_generation_root.mkdir(exist_ok=True)
        attempt_record_root.mkdir(exist_ok=True)
        locked_variant.verify_path()

        raw_stage = create_exclusive_directory_stage(
            generation_root, attempt_id
        )
        with raw_stage:
            workload_attempt = workload_output_root / attempt_id
            workload_attempt.mkdir(parents=True, exist_ok=False)
            retained_workload_output = capture_sources.enter_context(
                PinnedDirectoryReader(workload_attempt)
            )
            env = tracer_runtime_env(
                _TRACER_SO,
                retained_descriptor=retained_tracer.descriptor,
            )
            env["CUDA_INJECTION64_PATH"] = retained_tracer.proc_path
            env["USER_DEFINED_FOLDERS"] = "1"
            env["TRACES_FOLDER"] = f"/proc/self/fd/{raw_stage.descriptor}"
            retained_binary.verify_unchanged()
            retained_tracer.verify_unchanged()
            for retained in retained_authority.values():
                retained.verify_unchanged()
            _verify_source_commit_head(source_commit)
            t0 = time.perf_counter()
            proc = _run_workload(
                [retained_binary.proc_path, *args],
                env,
                str(binary.parent),
                pass_fds=(
                    retained_binary.descriptor,
                    retained_tracer.descriptor,
                    retained_shapes.descriptor,
                    raw_stage.descriptor,
                    retained_workload_output.descriptor,
                ),
            )
            t1 = time.perf_counter()
            retained_binary.verify_unchanged()
            retained_tracer.verify_unchanged()
            for retained in retained_authority.values():
                retained.verify_unchanged()
            _verify_source_commit_head(source_commit)
            retained_workload_output.verify_path()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"workload {binary} returned {proc.returncode}; "
                    f"stderr (tail): {proc.stderr[-500:]!r}"
                )
            raw_generation = raw_stage.commit(
                require_nonempty=True,
                **_trace_generation_commit_limits(),
            )
            _verify_committed_trace_generation(raw_generation)

        traces_folder = raw_generation.path
        pb_path = traces_folder / "dynamic_trace.pb"
        _verify_committed_trace_generation(raw_generation)
        try:
            admission = require_trace_admission(traces_folder)
        except TraceAdmissionError as exc:
            raise RuntimeError(
                f"NVBit capture for {slug}/{shape_point}/register_cap_{cap} "
                f"is not admissible: {exc}"
            ) from exc
        launch = _require_single_admitted_launch(
            admission,
            producer=(
                f"register-cap trace {slug}/{shape_point}/register_cap_{cap}"
            ),
        )
        _verify_committed_trace_generation(raw_generation)

        admitted_sources = ExitStack()
        summary_published = False
        sidecar_published = False
        summary_payload = b""
        sidecar_payload = b""
        try:
            with admitted_sources:
                trace_sources = admitted_sources.enter_context(
                    open_admitted_trace_sources(
                        traces_folder,
                        expected_admission=admission,
                    )
                )
                summary_stage = create_exclusive_directory_stage(
                    summary_generation_root, attempt_id
                )
                with summary_stage:
                    try:
                        record = emit_admitted_nvbit_trace_summary(
                            admission=admission,
                            slug=slug,
                            shape_point=shape_point,
                            repeat_index=0,
                            launch_index=0,
                            binary_sha256=binary_sha,
                            tracer_build_sha256=tracer_build_sha,
                            output_path=Path(_SUMMARY_NAME),
                            output_descriptor=summary_stage.descriptor,
                        )
                        expected_summary_payload = (
                            json.dumps(record, indent=2, sort_keys=False) + "\n"
                        ).encode("utf-8")
                    except NvbitProtobufBindingsMissing as exc:
                        raise RuntimeError(
                            "protobuf bindings unavailable; cannot derive the "
                            f"live kernel_signature for {slug}/{shape_point}/"
                            f"register_cap_{cap}: {exc}"
                        ) from exc
                    summary_generation = summary_stage.commit(
                        require_nonempty=True,
                        **_SUMMARY_GENERATION_LIMITS,
                    )
                _verify_committed_summary_generation(summary_generation)
                retained_summary = admitted_sources.enter_context(
                    open_retained_regular(
                        summary_generation.path / _SUMMARY_NAME
                    )
                )
                summary_payload = retained_summary.read_bytes()
                if summary_payload != expected_summary_payload:
                    raise ImmutableArtifactError(
                        "committed register-cap trace summary bytes disagree "
                        "with emitter output"
                    )
                summary_record = strict_json_loads(
                    summary_payload,
                    label="committed register-cap trace summary",
                )
                if summary_record != record:
                    raise ImmutableArtifactError(
                        "committed register-cap trace summary disagrees with "
                        "emitter result"
                    )
                raw_generation_record = _raw_generation_record(
                    raw_generation,
                    cell_dir=variant_dir,
                    attempt_id=attempt_id,
                )
                summary_generation_record = _raw_generation_record(
                    summary_generation,
                    cell_dir=variant_dir,
                    attempt_id=attempt_id,
                )
                sidecar = _build_variant_sidecar(
                    slug=slug,
                    shape_point=shape_point,
                    register_cap=cap,
                    variant_id=variant_id,
                    trace_capture_attempt_id=attempt_id,
                    raw_trace_generation=raw_generation_record,
                    summary_generation=summary_generation_record,
                    arch_dir=arch_dir,
                    sku=sku,
                    binary_sha256=binary_sha,
                    kernel_signature_sha256=str(
                        record["kernel_signature_sha256"]
                    ),
                    kernel_name=str(record["kernel_name"]),
                    source_commit=source_commit,
                    trace_pb_relpath=str(pb_path.relative_to(variant_dir)),
                    trace_bytes=admission.dynamic_trace_size_bytes,
                    nvbit_tracer_build_hash=tracer_build_sha,
                    admission=admission,
                    wall_time_s=t1 - t0,
                    expected_record_count=admission.expected_record_count,
                    admitted_record_count=admission.admitted_record_count,
                    dirty_override=dirty_override,
                )
                sidecar_payload = (
                    json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                attempt_payload = (
                    json.dumps(
                        {
                            "activation_state": (
                                "non_admissible"
                                if dirty_override
                                else "candidate"
                            ),
                            "typed_reason": (
                                "provenance_dirty_override"
                                if dirty_override
                                else None
                            ),
                            "sidecar": sidecar,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")

                def verify_sources() -> None:
                    _verify_committed_trace_generation(raw_generation)
                    _verify_committed_summary_generation(summary_generation)
                    trace_sources.verify_unchanged()
                    retained_binary.verify_unchanged()
                    retained_tracer.verify_unchanged()
                    for retained in retained_authority.values():
                        retained.verify_unchanged()
                    retained_summary.verify_unchanged()
                    retained_workload_output.verify_path()
                    locked_variant.verify_path()
                    _verify_source_commit_head(source_commit)

                verify_sources()
                publish_regular_file(
                    attempt_record_root,
                    f"{attempt_id}.json",
                    attempt_payload,
                )
                verify_sources()
                if dirty_override:
                    retained_capture_sources = capture_sources.pop_all()
                    retained_capture_sources.close()
                else:
                    retire_regular_file(sidecar_path)
                    retire_regular_file(summary_path)
                    publish_regular_file(
                        summary_path.parent,
                        summary_path.name,
                        summary_payload,
                    )
                    summary_published = True
                    verify_sources()
                    publish_regular_file(
                        sidecar_path.parent,
                        sidecar_path.name,
                        sidecar_payload,
                    )
                    sidecar_published = True
                    verify_sources()
                    retained_publication = _RetainedActivationPublication(
                        capture_sources=capture_sources.pop_all(),
                        admitted_sources=admitted_sources.pop_all(),
                        lock_sources=lock_sources.pop_all(),
                        lock_hold_sources=lock_hold_sources.pop_all(),
                        lock_reader=locked_variant,
                        raw_generation=raw_generation,
                        sidecar_path=sidecar_path,
                        sidecar_payload=sidecar_payload,
                        manifest_path=summary_path,
                        manifest_payload=summary_payload,
                        publication_checks=(
                            lambda: _verify_committed_summary_generation(
                                summary_generation
                            ),
                            lambda: _verify_source_commit_head(source_commit),
                        ),
                    )
                    retained_publication.__enter__()
                    retained_publication.__exit__(None, None, None)
            if dirty_override:
                _verify_committed_trace_generation(raw_generation)
                _verify_committed_summary_generation(summary_generation)
                _verify_source_commit_head(source_commit)
                locked_variant.verify_path()
        except BaseException as publication_error:
            for published, path, expected_payload in (
                (sidecar_published, sidecar_path, sidecar_payload),
                (summary_published, summary_path, summary_payload),
            ):
                if not published:
                    continue
                try:
                    _retire_matching_locator(path, expected_payload)
                except BaseException as retirement_error:
                    try:
                        publication_error.add_note(
                            f"failed register-cap locator retirement for {path}: "
                            f"{retirement_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
            raise

        if dirty_override:
            raise RuntimeError(
                "provenance_dirty_override: immutable attempt evidence was "
                "preserved but no active sidecar or summary was published"
            )
        return sidecar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trace the register_spill_local_mem register-cap build variants "
            "with NVBit and emit per-variant trace JSONs under "
            "variants/register_cap_<cap>/."
        ),
    )
    parser.add_argument(
        "--shape-point", required=True,
        help="Shape point to trace (e.g. trace_l2_accuracy_xl).",
    )
    parser.add_argument(
        "--caps", default=None,
        help=(
            "Comma-separated register caps to trace; default the canonical "
            f"ladder {','.join(str(c) for c in LADDER_CAPS)}."
        ),
    )
    parser.add_argument("--sku", default="rtx_3090_ti")
    parser.add_argument("--arch-dir", default="sm_86")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help=(
            "Proceed even when provenance-relevant files are uncommitted; the "
            "emitted trace is marked NON-ADMISSIBLE (provenance_dirty_override). "
            "Default fails closed so a trace's source_commit reproduces the "
            "producer."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT,
        help="Active output root below results/nvbit_capture_staging.",
    )
    args = parser.parse_args(argv)

    if args.caps:
        try:
            caps = tuple(int(c.strip()) for c in args.caps.split(",") if c.strip())
        except ValueError:
            sys.stderr.write(
                f"trace_register_cap_variant_cells: --caps {args.caps!r} must "
                f"be comma-separated integers\n"
            )
            return 2
    else:
        caps = tuple(LADDER_CAPS)

    if len(set(caps)) != len(caps):
        sys.stderr.write(
            "trace_register_cap_variant_cells: --caps contains duplicate "
            "register-cap output identities\n"
        )
        return 2

    try:
        plans = [
            _variant_output_plan(
                slug=_SLUG,
                shape_point=args.shape_point,
                register_cap=cap,
                sku=args.sku,
                arch_dir=args.arch_dir,
                output_root=args.output_root,
            )
            for cap in caps
        ]
    except (ProtectedPathError, ValueError) as exc:
        sys.stderr.write(
            f"trace_register_cap_variant_cells: output refusal: {exc}\n"
        )
        return 2

    failures: list[str] = []
    for cap, plan in zip(caps, plans, strict=True):
        sys.stderr.write(f"=== register_cap_{cap} / {args.shape_point} ===\n")
        try:
            sidecar = trace_variant_cell(
                shape_point=args.shape_point, register_cap=cap,
                sku=args.sku, arch_dir=args.arch_dir, seed=args.seed,
                allow_dirty=args.allow_dirty,
                output_root=args.output_root,
                _precleared_plan=plan,
            )
            sys.stderr.write(
                f"OK: register_cap_{cap}: binary_sha256="
                f"{sidecar['binary_sha256'][:16]}... "
                f"kernel_signature_sha256="
                f"{sidecar['kernel_signature_sha256'][:16]}... "
                f"admitted={sidecar['admitted_record_count']}\n"
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            sys.stderr.write(f"FAIL: register_cap_{cap}: {exc}\n")
            failures.append(f"register_cap_{cap}: {exc}")

    if failures:
        sys.stderr.write(
            f"\ntrace_register_cap_variant_cells: {len(failures)} cap(s) "
            f"failed:\n"
        )
        for f in failures:
            sys.stderr.write(f"  {f}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
