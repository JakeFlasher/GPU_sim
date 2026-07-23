"""Activation NVBit-trace orchestrator (tools/local_labels scope).

`workloads/simbench_bw/run_suite.py --mode trace` is microbench-shaped:
it walks `workloads/simbench_bw/{l1,l2,l3,l4}/<slug>/<slug>_run`
and consumes per-slug shape TOMLs under `workloads/simbench_bw/shapes/`.
The 5 activation cells in the headline cohort live under
`workloads/portable_lcd/<slug>/<slug>_run` with a different CLI
contract (binary takes `--timing_shape` / `--correctness_shape`
flags, not `--shape-point` paired with TOML). Rather than widen
run_suite.py, this orchestrator stays inside the local-label
exception scope (`tools/local_labels/`) and produces the same
on-disk sidecar shape the existing prereq gate
(``test_head_coherent_retrace_prereq.py``) expects:

    results/nvbit_capture_staging/activation_trace/workloads/<source>/results/
        <slug>/rtx_3090_ti/sm_86/trace_<shape_point>.json
    results/nvbit_capture_staging/activation_trace/workloads/<source>/results/
        <slug>/rtx_3090_ti/sm_86/trace_<shape_point>/dynamic_trace.pb

The sidecar carries the four admission-counter fields
(``expected_record_count``, ``admitted_record_count``,
``actual_drop_rate``, ``decoder_gap_dropped``) plus
``binary_sha256`` and ``source_commit`` so the prereq gate
admits the cell as HEAD-coherent.

Per CLAUDE.md Rule 1 local-label exception: NCU is invoked only
under `tools/local_labels/` and writes only into the quarantine
path. This orchestrator does NOT invoke NCU; it invokes the NVBit
tracer (which is permitted globally — see Rule 1's exception
scope). Living under `tools/local_labels/` keeps the activation
orchestrator co-located with the other label-driver code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACER_SO = _REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.nvbit_tracer.runtime_env import tracer_runtime_env  # noqa: E402
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    TraceAdmissionError,
    require_trace_admission,
)
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_write_allowed,
)


def _tracer_env() -> dict:
    """Return the shared, pinned and dependency-checked tracer environment."""
    return tracer_runtime_env(_TRACER_SO)

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


def _git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), check=True,
    )
    return out.stdout.strip()


def _binary_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _kernel_signature_sha256_from_source(source_path: Path) -> str:
    """The microbench convention is to hash the .cu source as a
    proxy for the per-slug kernel signature. The activation
    orchestrator mirrors that: hash the per-slug .cu file at HEAD
    so the sidecar's ``kernel_signature_sha256`` is deterministic
    and aligns with the predictor-side join field."""
    h = hashlib.sha256()
    with open(source_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _stats_csv_kernel_name(traces_folder: Path) -> str | None:
    stats_csv = traces_folder / "stats.csv"
    if not stats_csv.is_file():
        return None
    try:
        with stats_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header = [h.strip() for h in rows[0]]
    try:
        idx = header.index("kernel_name")
    except ValueError:
        return None
    return rows[1][idx].strip() if len(rows[1]) > idx else None


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
    if shape_point not in _SUPPORTED_SHAPES:
        raise ValueError(
            f"shape_point must be one of {_SUPPORTED_SHAPES}; "
            f"got {shape_point!r}"
        )
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

    head = _git_head()
    binary_sha = _binary_sha256(binary)
    kernel_sig = _kernel_signature_sha256_from_source(source)

    cell_dir = plan["cell_dir"]
    traces_folder = plan["traces_folder"]
    sidecar_path = plan["sidecar_path"]
    workload_output_root = plan["workload_output_root"]

    if traces_folder.exists():
        _assert_staging_output(traces_folder, operation="remove prior trace")
        shutil.rmtree(traces_folder)
    traces_folder.mkdir(parents=True, exist_ok=True)
    workload_output_root.mkdir(parents=True, exist_ok=True)

    env = _tracer_env()
    env["CUDA_INJECTION64_PATH"] = str(_TRACER_SO)
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(traces_folder)

    # The portable_lcd runner accepts `--timing_shape` /
    # `--correctness_shape` for the two canonical modes, AND
    # `--shape-point <name>` for the generic `trace_l2_accuracy_xl`
    # / `trace_l2_accuracy_lg` shapes (per
    # `workloads/portable_lcd/runner_common.h:137-156`). The
    # generic form must be passed as TWO arguments, not as a single
    # `--shape-point=trace_l2_accuracy_xl` token, because the
    # runner uses argv-index parsing.
    if shape_point in _GENERIC_SHAPE_POINT_SHAPES:
        shape_argv: list[str] = ["--shape-point", shape_point]
    else:
        shape_argv = [f"--{shape_point}"]
    invocation = [
        str(binary), *shape_argv, *extra_app_args,
        "--output_dir", str(workload_output_root),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        invocation,
        capture_output=True, text=True, env=env, cwd=str(binary.parent),
    )
    t1 = time.perf_counter()

    pb_path = traces_folder / "dynamic_trace.pb"
    if proc.returncode != 0:
        raise RuntimeError(
            f"activation binary {binary} returned {proc.returncode}; "
            f"stderr (tail): {proc.stderr[-500:]!r}"
        )
    if not pb_path.is_file():
        raise RuntimeError(
            f"NVBit capture for {slug}/{shape_point} did not produce "
            f"dynamic_trace.pb at {pb_path}; stderr (tail): "
            f"{proc.stderr[-500:]!r}"
        )

    try:
        admission = require_trace_admission(traces_folder)
    except TraceAdmissionError as exc:
        raise RuntimeError(
            f"NVBit capture for {slug}/{shape_point} is not admissible: {exc}"
        ) from exc
    expected = admission.expected_record_count
    admitted = admission.admitted_record_count
    drop_count = max(expected - admitted, 0)
    drop_rate = (drop_count / expected) if expected else 0.0

    sidecar = {
        "schema_version": 1,
        "slug": slug,
        "shape_set": "activation_shapes",
        "shape_point": shape_point,
        "trace_pb_path": str(pb_path.relative_to(cell_dir)),
        "trace_bytes": pb_path.stat().st_size,
        "wall_time_s": t1 - t0,
        "tracer_so_path": str(
            _TRACER_SO.relative_to(_REPO_ROOT),
        ),
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
        "kernel_name": _stats_csv_kernel_name(traces_folder),
        "device_name": _DEVICE_PROPS_BY_SKU[sku]["device_name"],
        "arch": arch_dir,
        # Record the per-launch CLI extras (if any) so the trace
        # identity is auditable. Empty tuple = default shape.
        "extra_app_args": list(extra_app_args),
        # The activation orchestrator emits the raw protobuf +
        # sector_stream sidecar but NOT the 4 predictor-side
        # derivations (hrd_profile, divergence_proxies,
        # sharing_locality, order_projections). Recording
        # `hrd_skip_reason` lets the repository verifier's
        # upstream-skip carve-out
        # (verify_artifacts.py:305) admit the cell tree without
        # demanding feature sidecars the orchestrator never
        # emits — those derivations belong to a downstream
        # pipeline that consumes dynamic_trace.pb directly.
        "hrd_skip_reason": "activation_orchestrator_emit_only",
    }
    _assert_staging_output(sidecar_path, operation="write trace sidecar")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    # Emit the per-shape measurement_manifest_<shape_point>.json
    # next to the summary. The repository verifier
    # (verify_artifacts._verify_canonical_trace_pb_admission)
    # rejects any canonical-tree dynamic_trace.pb whose
    # artifact_dir lacks this manifest with the typed
    # `RawOnlyTraceTree` diagnostic. The microbench path emits
    # a single cell-level `measurement_manifest.json` via
    # run_suite._emit_manifest because each microbench cell has
    # exactly one shape; the activation path mirrors the same
    # emission point but with the SHAPE-KEYED filename so
    # multi-shape activation slugs (e.g., silu_fwd_bf16_sm80
    # traced for both correctness_shape AND timing_shape under
    # the same cell_dir) preserve shape-level provenance. The
    # second trace's manifest used to overwrite the first
    # trace's manifest at the cell-level filename; the
    # per-shape filename keeps both manifests independent. The
    # verifier looks for the per-shape filename first and
    # falls back to the cell-level filename for backwards
    # compatibility with the microbench layout.
    manifest_path = plan["manifest_path"]
    manifest_payload = _build_activation_manifest_payload(
        slug=slug,
        shape_point=shape_point,
        binary_sha256=binary_sha,
        source_commit=head,
        kernel_signature_sha256=kernel_sig,
        sku=sku,
    )
    _assert_staging_output(manifest_path, operation="write measurement manifest")
    with open(manifest_path, "w") as f:
        json.dump(manifest_payload, f, indent=2, sort_keys=True)
        f.write("\n")
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
