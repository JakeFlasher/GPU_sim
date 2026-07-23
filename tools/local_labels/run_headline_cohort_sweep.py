"""Orchestrator for the 13-cell headline-cohort live NCU sweep.

Stages 13 measurement manifests under
``data/local_labels/sweeps/<sweep_id>/manifests/``, runs the
provenance audit per cell (must PASS for all 13 before the live
sweep starts), then drives ``sweep_repeatability_matrix.py --live``
over the full matrix
(``warmup_count={0,1,5,10} × cache_control={all,none} × repeats=5``
= 40 invocations × 13 cells = 520 invocations, ~4.3 hr operator
wall-clock at ~30 s/invocation).

Outputs:
  - data/local_labels/sweeps/<sweep_id>/manifests/<cell>.json
  - sweep_repeatability_matrix's standard output tree under
    data/local_labels/sweeps/<sweep_id>/
  - aggregated label_admitted.json + label_excluded.json +
    label_repeatability.json (via repeatability_aggregator.py)

The orchestrator preserves the existing
``data/local_labels/label_admitted.json`` historical 5 activation
labels at ``source_commit=0c16e335...`` per the
supersession-bookkeeping discipline.

Per CLAUDE.md Rule 1: NCU is invoked only via this orchestrator's
delegation to ``sweep_repeatability_matrix.py``; both live under
``tools/local_labels/`` (the local-label exception scope).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

# The cohort manifest validator lives under
# tools/sm_86_predictor_accuracy_recovery_v2/. Add that dir to
# sys.path so the in-process import succeeds without requiring
# the orchestrator to be invoked from the repo root.
_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
_COHORT_MANIFEST_TOOLS_DIR = (
    _REPO_ROOT_FOR_IMPORTS
    / "tools"
    / "sm_86_predictor_accuracy_recovery_v2"
)
if str(_COHORT_MANIFEST_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_COHORT_MANIFEST_TOOLS_DIR))

import validate_cohort_manifest as _vcm  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEPS_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"
_RESULTS_ROOT = _REPO_ROOT / "workloads" / "simbench_bw" / "results"


_HEADLINE_COHORT_13: tuple[tuple[str, str, str], ...] = (
    # 8 microbench cells (slug, shape_point, binary_path)
    ("working_set_capacity_sweep", "trace_l2_accuracy_xl",
     "workloads/simbench_bw/l1/working_set_capacity_sweep/working_set_capacity_sweep_run"),
    ("l2_thrash_pointer_chase", "trace_l2_accuracy_xl",
     "workloads/simbench_bw/l1/l2_thrash_pointer_chase/l2_thrash_pointer_chase_run"),
    ("dram_stream_stride_mix", "trace_l2_accuracy_xl",
     "workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix_run"),
    ("carveout_sensitivity_probe", "trace_l2_accuracy_xl",
     "workloads/simbench_bw/l1/carveout_sensitivity_probe/carveout_sensitivity_probe_run"),
    ("warm_cold_pair_relaunch", "trace_l2_accuracy_xl",
     "workloads/simbench_bw/l1/warm_cold_pair_relaunch/warm_cold_pair_relaunch_run"),
    ("l2_persisting_window_sweep", "trace_l2_accuracy_xl",
     "workloads/simbench_bw/l1/l2_persisting_window_sweep/l2_persisting_window_sweep_run"),
    ("l2_prefetch_hint_sweep", "trace_l2_accuracy_xl",
     "workloads/simbench_bw/l1/l2_prefetch_hint_sweep/l2_prefetch_hint_sweep_run"),
    ("register_spill_local_mem", "trace_l2_accuracy_xl",
     "workloads/simbench_bw/l1/register_spill_local_mem/register_spill_local_mem_run"),
    # 5 activation cells (slug, shape_point, binary_path)
    ("layernorm_fwd_bf16_sm80", "timing_shape",
     "workloads/portable_lcd/layernorm_fwd_bf16_sm80/layernorm_fwd_bf16_sm80_run"),
    ("rmsnorm_fwd_bf16_sm80", "timing_shape",
     "workloads/portable_lcd/rmsnorm_fwd_bf16_sm80/rmsnorm_fwd_bf16_sm80_run"),
    ("silu_fwd_bf16_sm80", "correctness_shape",
     "workloads/portable_lcd/silu_fwd_bf16_sm80/silu_fwd_bf16_sm80_run"),
    ("silu_fwd_bf16_sm80", "timing_shape",
     "workloads/portable_lcd/silu_fwd_bf16_sm80/silu_fwd_bf16_sm80_run"),
    ("softmax_safe_bf16_sm80", "timing_shape",
     "workloads/portable_lcd/softmax_safe_bf16_sm80/softmax_safe_bf16_sm80_run"),
)


def _resolve_cohort(
    cells_filter: str | None,
    *,
    cohort_manifest_path: Path | None = None,
    candidate_catalog_path: Path | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Return the cohort subset for the sweep.

    With ``cells_filter`` unset, returns the full
    ``_HEADLINE_COHORT_13`` tuple (back-compat).

    With ``cells_filter`` set + ``cohort_manifest_path`` and
    ``candidate_catalog_path`` both supplied, resolves each
    requested ``<arch>/<slug>/<shape_point>`` against:
      - the legacy ``_HEADLINE_COHORT_13`` map (fastest path);
      - the cohort manifest's cells list (the cell must appear
        there in either ``admitted`` or ``attempted_excluded``
        admission status);
      - the candidate catalog's binary_path entries (for cells
        not in the legacy tuple).
    Any cell that fails all three lookups raises ValueError.

    With ``cells_filter`` set + no manifest/catalog supplied,
    falls back to the legacy ``_HEADLINE_COHORT_13`` resolution
    (back-compat for invocations that predate the catalog).
    """
    if cells_filter is None or not cells_filter.strip():
        return _HEADLINE_COHORT_13
    requested = [s.strip() for s in cells_filter.split(",") if s.strip()]
    legacy_by_id = {
        f"sm_86/{slug}/{shape_point}": (slug, shape_point, binary)
        for slug, shape_point, binary in _HEADLINE_COHORT_13
    }
    if cohort_manifest_path is None and candidate_catalog_path is None:
        unknown = [r for r in requested if r not in legacy_by_id]
        if unknown:
            raise ValueError(
                "--cells contains cell IDs not in _HEADLINE_COHORT_13: "
                f"{unknown}. Cohort drift refused; supply "
                "--cohort-manifest + --candidate-catalog to admit "
                "cells from the expanded manifest."
            )
        return tuple(legacy_by_id[r] for r in requested)
    manifest_cell_ids: set[str] = set()
    if cohort_manifest_path is not None:
        # Load the manifest through the full validator so the
        # schema's closed enums, required-field set, accounting
        # invariants, identity-tuple lockstep, replacement-history
        # shape, AND identity-bound label-presence are all
        # enforced before any catalog lookup happens. The
        # validator's typed exceptions
        # (CohortManifestSchemaError,
        # CohortManifestDuplicateCellError,
        # CohortManifestAccountingDriftError,
        # CohortManifestIdentityLockstepError,
        # CohortManifestLabelPresenceError) propagate so the
        # operator sees the manifest defect, not a "Cohort drift
        # refused" surface error. The label-presence check refuses
        # any admitted cell whose identity tuple is absent from
        # data/local_labels/label_admitted.json before the
        # resolver iterates the cohort.
        manifest = _vcm.validate_manifest_with_path_reconciliation(
            cohort_manifest_path,
            enforce_label_presence=True,
        )
        # Manifest IDs come from each cell's actual arch field on
        # the validated dataclass list. A manifested arch="sm_120"
        # cell does NOT resolve to an sm_86 request even when the
        # candidate catalog has a matching slug.
        for cell in manifest.cells:
            manifest_cell_ids.add(
                f"{cell.arch}/{cell.slug}/{cell.shape_point}"
            )
    catalog_by_slug: dict[str, str] = {}
    if candidate_catalog_path is not None:
        with open(candidate_catalog_path) as fh:
            catalog = json.load(fh)
        for entry in catalog.get("candidates", []):
            catalog_by_slug[entry["slug"]] = entry["binary_path"]
    resolved: list[tuple[str, str, str]] = []
    unknown: list[str] = []
    for r in requested:
        if r in legacy_by_id:
            resolved.append(legacy_by_id[r])
            continue
        if r not in manifest_cell_ids:
            unknown.append(r)
            continue
        parts = r.split("/")
        if len(parts) != 3 or parts[0] != "sm_86":
            unknown.append(r)
            continue
        _arch, slug, shape_point = parts
        binary = catalog_by_slug.get(slug)
        if binary is None:
            unknown.append(r)
            continue
        resolved.append((slug, shape_point, binary))
    if unknown:
        raise ValueError(
            "--cells contains unresolvable cell IDs: "
            f"{unknown}. Each cell must be in either the legacy "
            "_HEADLINE_COHORT_13 tuple, or in the cohort_manifest's "
            "cells list with a candidate-catalog binary_path entry "
            "for its slug."
        )
    return tuple(resolved)


def _binary_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _trace_sidecar_for(slug: str, shape_point: str) -> Path:
    return (
        _RESULTS_ROOT / slug / "rtx_3090_ti" / "sm_86"
        / f"trace_{shape_point}.json"
    )


def _read_kernel_signature_sha256(slug: str, shape_point: str) -> str:
    """Read kernel_signature_sha256 from the trace sidecar so the
    measurement manifest declares the same kernel-signature value
    the predictor side sees (the audit cross-checks both sides on
    this field)."""
    sidecar = _trace_sidecar_for(slug, shape_point)
    with open(sidecar) as f:
        payload = json.load(f)
    return str(payload["kernel_signature_sha256"])


def stage_manifests(
    sweep_dir: Path,
    cohort: tuple[tuple[str, str, str], ...] = _HEADLINE_COHORT_13,
) -> dict[str, Path]:
    """Write measurement manifests under
    ``<sweep_dir>/manifests/<cell_id>.json``. Returns a map of
    cell_id -> manifest_path keyed by ``arch/slug/shape_point``.
    The default ``cohort`` writes all 13; passing a subset (via
    ``_resolve_cohort``) writes only the requested cells.
    """
    manifests_dir = sweep_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for slug, shape_point, binary_relpath in cohort:
        binary = _REPO_ROOT / binary_relpath
        binary_sha = _binary_sha256(binary)
        kernel_sig = _read_kernel_signature_sha256(slug, shape_point)
        manifest = {
            "schema_version": 2,
            "slug": slug,
            "shape_point": shape_point,
            "binary_sha256": binary_sha,
            "kernel_signature_sha256": kernel_sig,
        }
        cell_id = f"sm_86/{slug}/{shape_point}"
        # The on-disk filename uses the same ``cell=`` convention the
        # sweep harness's per-cell output dirs use.
        safe_name = (
            cell_id.replace("/", "__").replace("=", "_")
        )
        manifest_path = manifests_dir / f"{safe_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        out[cell_id] = manifest_path
    return out


def run_provenance_audit(
    cell_id: str, manifest_path: Path,
) -> tuple[int, str, str]:
    """Invoke ``provenance_audit.py`` for one cell. Returns
    ``(returncode, stdout, stderr)``. Exit code 0 == PASS.

    Always invokes the audit in ``--live-sweep`` mode so the
    orchestrator's ``--audit-only`` precheck enforces the same
    source_commit narrowing as the live sweep preflight in
    ``sweep_repeatability_matrix._run_provenance_audit_preflight``
    (``live_sweep=True``). Without this flag, the default
    offline audit admits HEAD^ predictor sidecars; the live
    sweep's preflight rejects them; ``--audit-only`` would
    return success for a cohort the live sweep will then
    immediately fail. Codex R52-review P2 fix.
    """
    arch, slug, shape_point = cell_id.split("/", 2)
    sidecar = _trace_sidecar_for(slug, shape_point)
    proc = subprocess.run(
        [
            "python3",
            str(_REPO_ROOT / "tools" / "local_labels" / "provenance_audit.py"),
            "--predictor-trace-json", str(sidecar),
            "--manifest", str(manifest_path),
            "--arch", arch,
            "--live-sweep",
        ],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    return proc.returncode, proc.stdout, proc.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate the 13-cell headline-cohort live NCU sweep."
        ),
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Override the sweep-id (default: "
            "sweep_<UTC_timestamp>_<short-hash>)."
        ),
    )
    parser.add_argument(
        "--warmup-counts", default="0,1,5,10",
        help="Comma-separated warmup counts (default: 0,1,5,10).",
    )
    parser.add_argument(
        "--cache-controls", default="none,all",
        help="Comma-separated cache-control modes (default: none,all).",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Repeats per (cell, config) tuple (default: 5).",
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help=(
            "Stage manifests + run audit; do NOT launch the live "
            "sweep (smoke-test the prep stages)."
        ),
    )
    parser.add_argument(
        "--cells", default=None,
        help=(
            "Optional comma-separated list of "
            "<arch>/<slug>/<shape_point> cell IDs to filter the "
            "sweep cohort. With neither --cohort-manifest nor "
            "--candidate-catalog supplied, each ID MUST be in the "
            "legacy _HEADLINE_COHORT_13 tuple (back-compat). With "
            "both flags supplied, cells may also come from the "
            "expanded cohort_manifest as long as the candidate "
            "catalog supplies a binary_path for the slug. Default: "
            "run the full 13-cell legacy cohort."
        ),
    )
    parser.add_argument(
        "--cohort-manifest", type=Path, default=None,
        help=(
            "Optional path to cohort_manifest.json. When supplied "
            "together with --candidate-catalog, the orchestrator "
            "resolves --cells against the manifest's cells list "
            "(not just the legacy _HEADLINE_COHORT_13 tuple). The "
            "default cohort_manifest path is "
            "data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json"
            "; pass the flag explicitly to opt in to expanded-cohort "
            "resolution."
        ),
    )
    parser.add_argument(
        "--candidate-catalog", type=Path, default=None,
        help=(
            "Optional path to expansion_candidates.json. When "
            "supplied together with --cohort-manifest, the "
            "orchestrator looks up binary_path entries via this "
            "catalog for cells outside the legacy "
            "_HEADLINE_COHORT_13 tuple. Default catalog path is "
            "data/sm_86_predictor_accuracy_recovery_v2/expansion_candidates.json."
        ),
    )
    args = parser.parse_args(argv)

    try:
        cohort = _resolve_cohort(
            args.cells,
            cohort_manifest_path=args.cohort_manifest,
            candidate_catalog_path=args.candidate_catalog,
        )
    except ValueError as exc:
        print(f"--cells error: {exc}", file=sys.stderr)
        return 2

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), check=True,
        ).stdout.strip()
        sweep_id = f"sweep_{ts}_{head}"

    sweep_dir = _SWEEPS_ROOT / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"sweep_id: {sweep_id}", file=sys.stderr)
    print(f"sweep_dir: {sweep_dir}", file=sys.stderr)

    print(
        f"\n=== staging {len(cohort)} measurement manifests ===",
        file=sys.stderr,
    )
    manifests = stage_manifests(sweep_dir, cohort)
    for cell_id, path in sorted(manifests.items()):
        print(f"  {cell_id} -> {path.relative_to(_REPO_ROOT)}", file=sys.stderr)

    print(
        f"\n=== running provenance audit (must PASS for all "
        f"{len(manifests)}) ===",
        file=sys.stderr,
    )
    audit_failures: list[str] = []
    for cell_id in sorted(manifests.keys()):
        rc, stdout, stderr = run_provenance_audit(cell_id, manifests[cell_id])
        first_line = stdout.strip().split("\n", 1)[0] if stdout else "(no stdout)"
        print(f"  {cell_id}: rc={rc} {first_line}", file=sys.stderr)
        if rc != 0:
            audit_failures.append(
                f"{cell_id}: rc={rc} stderr={stderr.strip()!r}"
            )

    if audit_failures:
        print(
            f"\nprovenance audit FAILED for {len(audit_failures)} cell(s):",
            file=sys.stderr,
        )
        for f in audit_failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(
        f"\nprovenance audit: {len(manifests)}/{len(manifests)} PASS",
        file=sys.stderr,
    )

    if args.audit_only:
        print("\n--audit-only: skipping live sweep launch", file=sys.stderr)
        return 0

    # Build the candidate string + binary map + manifest map + predictor-trace map
    candidates: list[str] = []
    binary_pairs: list[str] = []
    manifest_pairs: list[str] = []
    trace_pairs: list[str] = []
    binary_paths_by_cell: dict[str, str] = {
        f"sm_86/{slug}/{shape_point}": str(_REPO_ROOT / binary_relpath)
        for slug, shape_point, binary_relpath in cohort
    }
    for cell_id, manifest_path in sorted(manifests.items()):
        candidates.append(cell_id)
        binary_pairs.append(f"{cell_id}={binary_paths_by_cell[cell_id]}")
        manifest_pairs.append(f"{cell_id}={manifest_path}")
        arch, slug, shape_point = cell_id.split("/", 2)
        trace_path = _trace_sidecar_for(slug, shape_point)
        trace_pairs.append(f"{cell_id}={trace_path}")

    cmd = [
        "python3",
        str(_REPO_ROOT / "tools" / "local_labels" / "sweep_repeatability_matrix.py"),
        "--live",
        "--candidates", ",".join(candidates),
        "--binaries", ",".join(binary_pairs),
        "--measurement-manifests", ",".join(manifest_pairs),
        "--predictor-trace-jsons", ",".join(trace_pairs),
        "--warmup-counts", args.warmup_counts,
        "--cache-controls", args.cache_controls,
        "--repeats", str(args.repeats),
        "--out-root", str(sweep_dir),
    ]
    print(
        f"\n=== launching sweep matrix ({len(candidates)} cells × "
        f"{len(args.warmup_counts.split(','))} warmups × "
        f"{len(args.cache_controls.split(','))} cache-controls × "
        f"{args.repeats} repeats) ===",
        file=sys.stderr,
    )
    print(f"  cmd: {' '.join(cmd[:6])} ... [truncated]", file=sys.stderr)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    t1 = time.perf_counter()
    print(
        f"\nsweep returncode={proc.returncode} wall_time={t1 - t0:.1f}s",
        file=sys.stderr,
    )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
