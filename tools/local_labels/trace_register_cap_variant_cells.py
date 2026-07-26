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
  ``variant_id`` + the real per-cap ``binary_sha256`` + the live signature.

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
import hashlib
import os
import shutil
import subprocess
import sys
import time
import tomllib
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
    emit_nvbit_trace_summary,
)
from tools.nvbit_tracer.runtime_env import tracer_runtime_env  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    TraceAdmissionError,
    require_trace_admission,
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
        "sidecar_path": sidecar,
        "summary_path": variant_dir / "nvbit_trace_summary_0_0.json",
        "tmp_path": sidecar.with_name(sidecar.name + ".tmp"),
    }
    return {
        name: _assert_staging_output(path, operation=f"prepare {name}")
        for name, path in candidates.items()
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown_source_commit"


# Repo-relative path of this producer; its source_commit MUST resolve to a
# commit that contains this file so the emitting code is reproducible.
_PRODUCER_RELPATH = "tools/local_labels/trace_register_cap_variant_cells.py"
# The files whose source determines the emitted trace's identity fields: this
# producer + the two reused modules that compute the join keys (the live
# signature adapter + the canonical variant normalizer). A dirty/uncommitted
# state in any of these means the recorded source_commit cannot reproduce the
# emitting logic, so the producer fails closed.
_PROVENANCE_RELEVANT_PATHS = (
    _PRODUCER_RELPATH,
    "workloads/simbench_bw/nvbit_trace_summary_adapter.py",
    "tools/local_labels/variant_identity.py",
)


def _commit_contains_path(commit: str, rel_path: str) -> bool:
    """True iff ``rel_path`` exists in ``commit``'s tree (``git cat-file -e``)."""
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "cat-file", "-e", f"{commit}:{rel_path}"],
        capture_output=True, text=True,
    ).returncode == 0


def _path_is_dirty(rel_path: str) -> bool:
    """True iff ``rel_path`` has uncommitted changes OR is untracked
    (``git status --porcelain`` prints a line for it)."""
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "status", "--porcelain", "--", rel_path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return True  # fail closed when status cannot be determined
    return bool(out.stdout.strip())


def _resolve_source_commit(*, allow_dirty: bool = False) -> tuple[str, bool]:
    """Resolve the trace's ``source_commit`` and fail closed on a
    non-reproducible provenance state. Returns ``(commit, dirty_override)``.

    Fails closed (RuntimeError) when:
    - git HEAD is unresolvable;
    - HEAD does NOT contain the producer file (the recorded commit could not
      reproduce the emitting code -- the exact Round-20 defect);
    - any provenance-relevant file is dirty/untracked, UNLESS ``allow_dirty``,
      in which case it proceeds and returns ``dirty_override=True`` so the
      caller records a non-admissible marker."""
    head = _git_head()
    if not head or head == "unknown_source_commit":
        raise RuntimeError(
            "cannot resolve git HEAD; the trace's source_commit would be "
            "unverifiable -- run inside a git work tree"
        )
    if not _commit_contains_path(head, _PRODUCER_RELPATH):
        raise RuntimeError(
            f"HEAD {head} does not contain the producer {_PRODUCER_RELPATH}; "
            f"commit the producer BEFORE tracing so the trace's source_commit "
            f"reproduces the emitting code (Codex Round-20 provenance defect)."
        )
    dirty = [p for p in _PROVENANCE_RELEVANT_PATHS if _path_is_dirty(p)]
    if dirty:
        if not allow_dirty:
            raise RuntimeError(
                f"provenance-relevant files are uncommitted/untracked: "
                f"{dirty}; commit them before tracing (or pass allow_dirty to "
                f"emit a NON-ADMISSIBLE dirty-marked trace). The trace's "
                f"source_commit must reproduce the emitting logic."
            )
        return head, True
    return head, False


def _run_workload(
    cmd: list[str], env: dict[str, str], cwd: str,
) -> subprocess.CompletedProcess[str]:
    """Run the (NVBit-injected) workload. A thin wrapper so tests can intercept
    the launch without mutating the global ``subprocess`` module."""
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)


def _variant_binary_path(slug: str, register_cap: int) -> Path:
    """The pre-built per-cap binary path (built by ``make spill-variants``)."""
    return _KERNEL_DIR / "spill_variants" / f"{slug}_run__cap{register_cap}"


def _kernel_args_for_shape(shape_point: str, seed: int) -> list[str]:
    """Build the workload CLI args for ``shape_point`` from the shape TOML's
    ``axes`` (warps / live_regs / iterations). The work-envelope axes drive the
    real launch geometry (grid=(warps,1,1)/block=(32,1,1)); the spill phenomenon
    is the BUILD-TIME register cap, not these axes. Raises ``ValueError`` for an
    unknown shape_point."""
    with _SHAPES_TOML.open("rb") as fh:
        data = tomllib.load(fh)
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
    arch_dir: str,
    sku: str,
    binary_sha256: str,
    kernel_signature_sha256: str,
    kernel_name: str,
    source_commit: str,
    trace_pb_relpath: str,
    trace_bytes: int,
    wall_time_s: float,
    expected_record_count: int,
    admitted_record_count: int,
    dirty_override: bool = False,
) -> dict:
    """Assemble the per-variant ``trace_<shape>.json`` payload. Pure (no I/O) so
    the field contract is unit-testable. Carries the canonical build-variant
    identity (``register_cap`` / ``variant_id``) the variant-aware staging tool
    requires, plus the four join-key fields (slug, shape_point, binary_sha256,
    kernel_signature_sha256). When ``dirty_override`` is set (an allow_dirty
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
        "trace_pb_path": trace_pb_relpath,
        "trace_bytes": int(trace_bytes),
        "wall_time_s": wall_time_s,
        "tracer_so_path": str(_TRACER_SO.relative_to(_REPO_ROOT)),
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
    .so; a nonzero workload exit; a missing dynamic_trace.pb or stats.csv; or
    absent protobuf bindings (the live signature cannot be derived). The trace
    sidecar is written via a temp file + atomic rename ONLY after every check
    passes, and the prior sidecar is removed BEFORE launch, so no consumable
    stale trace_<shape>.json can survive a failed rerun."""
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
    # Provenance guard FIRST (before launch / before touching the prior
    # artifacts): the recorded source_commit must reproduce this producer, and
    # the provenance-relevant files must be committed -- else fail closed. The
    # Round-20 defect was tracing BEFORE committing the producer, so the trace
    # recorded a commit that could not reproduce the emitting code.
    source_commit, dirty_override = _resolve_source_commit(
        allow_dirty=allow_dirty
    )
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

    binary_sha = _sha256_file(binary)
    variant_dir = plan["variant_dir"]
    traces_folder = plan["traces_folder"]
    sidecar_path = plan["sidecar_path"]
    summary_path = plan["summary_path"]
    # Stale-safety: once provenance + pre-launch checks pass, wipe the prior
    # trace folder AND the prior consumable sidecars BEFORE launch, so a
    # post-launch failure cannot leave a stale trace_<shape>.json /
    # nvbit_trace_summary that the staging tool would consume as fresh.
    _assert_staging_output(variant_dir, operation="create variant directory")
    variant_dir.mkdir(parents=True, exist_ok=True)
    if traces_folder.exists():
        _assert_staging_output(traces_folder, operation="remove prior variant trace")
        shutil.rmtree(traces_folder)
    for stale in (sidecar_path, summary_path):
        if stale.exists():
            _assert_staging_output(stale, operation="remove stale variant sidecar")
            stale.unlink()
    traces_folder.mkdir(parents=True, exist_ok=True)

    env = tracer_runtime_env(_TRACER_SO)
    env["CUDA_INJECTION64_PATH"] = str(_TRACER_SO)
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(traces_folder)
    args = _kernel_args_for_shape(shape_point, seed)

    t0 = time.perf_counter()
    proc = _run_workload([str(binary), *args], env, str(binary.parent))
    t1 = time.perf_counter()

    pb_path = traces_folder / "dynamic_trace.pb"
    if proc.returncode != 0:
        raise RuntimeError(
            f"workload {binary} returned {proc.returncode}; stderr (tail): "
            f"{proc.stderr[-500:]!r}"
        )
    if not pb_path.is_file():
        raise RuntimeError(
            f"NVBit capture for {slug}/{shape_point}/register_cap_{cap} did "
            f"not produce dynamic_trace.pb at {pb_path}; stderr (tail): "
            f"{proc.stderr[-500:]!r}"
        )

    try:
        admission = require_trace_admission(traces_folder)
    except TraceAdmissionError as exc:
        raise RuntimeError(
            f"NVBit capture for {slug}/{shape_point}/register_cap_{cap} "
            f"is not admissible: {exc}"
        ) from exc
    expected = admission.expected_record_count
    admitted = admission.admitted_record_count

    # LIVE kernel signature from the captured .pb (real geometry), via the
    # shared adapter. The per-launch summary is written alongside the trace
    # (its prior copy was already removed above).
    try:
        _assert_staging_output(summary_path, operation="write trace summary")
        record = emit_nvbit_trace_summary(
            traces_dir=traces_folder,
            slug=slug,
            shape_point=shape_point,
            repeat_index=0,
            launch_index=0,
            executable_path=binary,
            tracer_so_path=_TRACER_SO,
            output_path=summary_path,
        )
    except NvbitProtobufBindingsMissing as exc:
        raise RuntimeError(
            f"protobuf bindings unavailable; cannot derive the live "
            f"kernel_signature for {slug}/{shape_point}/register_cap_{cap}: "
            f"{exc}"
        ) from exc

    sidecar = _build_variant_sidecar(
        slug=slug,
        shape_point=shape_point,
        register_cap=cap,
        variant_id=variant_id,
        arch_dir=arch_dir,
        sku=sku,
        binary_sha256=binary_sha,
        kernel_signature_sha256=str(record["kernel_signature_sha256"]),
        kernel_name=str(record["kernel_name"]),
        source_commit=source_commit,
        trace_pb_relpath=str(pb_path.relative_to(variant_dir)),
        trace_bytes=pb_path.stat().st_size,
        wall_time_s=t1 - t0,
        expected_record_count=expected,
        admitted_record_count=admitted,
        dirty_override=dirty_override,
    )
    # Atomic publish: write to a temp file in the same dir and os.replace into
    # place ONLY now that the .pb, stats.csv, summary, binary hash, and
    # signature have all succeeded. A crash mid-write leaves only the temp file,
    # never a partial/half-written trace_<shape>.json.
    import json
    tmp_path = plan["tmp_path"]
    _assert_staging_output(tmp_path, operation="write temporary variant sidecar")
    tmp_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _assert_staging_output(tmp_path, operation="replace temporary variant sidecar")
    _assert_staging_output(sidecar_path, operation="publish variant sidecar")
    os.replace(tmp_path, sidecar_path)
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
