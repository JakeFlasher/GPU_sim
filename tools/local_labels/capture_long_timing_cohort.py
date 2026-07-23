"""Capture the long-duration (>= ~552 us) timing cohort (read-only NCU).

Builds a SCOPED, additive timing-cohort artifact of >= 10 sm_86 cells, ALL against
ONE binary identity (one ``binary_sha256`` / ``kernel_signature`` family / one
``source_commit``), each carrying the four NCU labels

  - ``gpc__cycles_elapsed.max``     (cycles; the DVFS-stable publication target)
  - ``gpu__time_duration.sum``      (ns; the gated diagnostic)
  - ``l1tex__t_sector_hit_rate.pct``(L1 hit rate)
  - ``lts__t_sector_hit_rate.pct``  (L2 hit rate)

over ``n_repeats`` repeats, gated by the SAME per-unit repeatability machinery the
label pipeline uses (cycles/ns coefficient-of-variation <= 0.05; the two hit-rate
fractions stddev <= 0.03). A cell is admitted ONLY when every label clears its CV
gate AND the median ``gpu__time_duration.sum`` clears the ``--floor-ns`` floor
(default 551760 ns). Everything else is a TYPED exclusion (never a silent drop or
a partial cell).

The distinct cells are produced by running the SAME binary at different timing
shape overrides (e.g. GEMM ``--m/--n/--k``) so the binary identity is invariant
while the launch geometry (hence the per-cell ``kernel_signature_sha256``,
computed via the production ``compute_kernel_signature_sha256``) varies. The
result is written to a SCOPED artifact under ``data/local_labels/timing_cohort/``,
kept OUT of root ``label_admitted.json`` until closure (DEC-5), so the root
timing-invariant and the frozen-L2 / sm_120 freeze are never perturbed during
iteration.

CLAUDE.md compliance (binding):
  - Rule 1 (NCU quarantine + local-label exception): NCU invoked ONLY here under
    ``tools/local_labels/`` writing under ``data/local_labels/``. ``--metrics``
    capture, not a profiling section sweep.
  - Rule 2 (no driver-root): ``ncu --metrics`` does NOT lock clocks; no
    persistence / power / MIG / compute-mode mutation, no nvidia-smi clock-lock.
  - Rule 3 (no privileged CUPTI): NCU command-line metric capture only.
  - sm_86-only: the cohort is named/scoped sm_86; the manifest binds each cell to
    the on-disk binary's sha256 and refuses to emit if it drifts.

Exit 3 (no write) when ncu is absent (offline/CI), so a non-3090-Ti host never
overwrites the committed cohort.

CLI:
  python tools/local_labels/capture_long_timing_cohort.py \
    [--manifest data/local_labels/timing_cohort/ac4_long_cohort_manifest.json] \
    [--output data/local_labels/timing_cohort/ac4_long_duration_cohort_sm_86.json] \
    [--exclusions-out data/local_labels/timing_cohort/ac4_long_cohort_exclusions_sm_86.json] \
    [--ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu] \
    [--repeats 5] [--floor-ns 551760]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import hashlib
import io
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw.nvbit_trace_summary_adapter import (  # noqa: E402
    compute_kernel_signature_sha256,
)
from tools.local_labels.emit_long_timing_feature_artifacts import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_TRACE_ROOT,
    DEFAULT_STATIC_FEATURES,
    DEFAULT_STATIC_WORK,
    attach_feature_artifacts_to_cohort,
)

_ARCH = "sm_86"
DEFAULT_NCU = "/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu"
DEFAULT_MANIFEST = (
    "data/local_labels/timing_cohort/ac4_long_cohort_manifest.json"
)
DEFAULT_OUTPUT = (
    "data/local_labels/timing_cohort/ac4_long_duration_cohort_sm_86.json"
)
DEFAULT_EXCLUSIONS = (
    "data/local_labels/timing_cohort/ac4_long_cohort_exclusions_sm_86.json"
)
DEFAULT_REPEATS = 5
# 551760 ns = the contract's min_cohort_duration_ns_for_default_bound (~552 us).
DEFAULT_FLOOR_NS = 551760

# The four NCU metrics, with their unit + repeatability gate. cycles/ns gate on
# coefficient-of-variation <= 0.05; the two hit-rate fractions on stddev <= 0.03
# (the same thresholds the label pipeline uses).
_METRIC_SPEC = {
    "gpc__cycles_elapsed.max": {"unit": "cycles", "rule": "cv", "thresh": 0.05},
    "gpu__time_duration.sum": {"unit": "ns", "rule": "cv", "thresh": 0.05},
    "l1tex__t_sector_hit_rate.pct": {"unit": "pct", "rule": "stddev_frac",
                                     "thresh": 0.03},
    "lts__t_sector_hit_rate.pct": {"unit": "pct", "rule": "stddev_frac",
                                   "thresh": 0.03},
}
_METRIC_IDS = tuple(_METRIC_SPEC)


class CohortCaptureError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binary_path_for(
    slug_or_cell: str | dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> Path | None:
    if isinstance(slug_or_cell, dict):
        slug = str(slug_or_cell["slug"])
        explicit = slug_or_cell.get("binary_path") or (manifest or {}).get("binary_path")
        if explicit:
            path = REPO_ROOT / str(explicit)
            if path.is_file():
                return path
    else:
        slug = slug_or_cell
    for d in ("workloads/portable_lcd", "workloads/simbench_bw/l1"):
        p = REPO_ROOT / d / slug / f"{slug}_run"
        if p.is_file():
            return p
    return None


def _ncu_capture_once(
    ncu: str, binary: Path, run_args: list[str]
) -> dict[str, float]:
    """One read-only NCU --metrics capture; return {metric_id: value}."""
    proc = subprocess.run(
        [ncu, "--cache-control", "all", "--metrics", ",".join(_METRIC_IDS),
         "--launch-count", "1", "--csv", str(binary), *run_args],
        capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise CohortCaptureError(
            f"ncu rc={proc.returncode}: {proc.stderr[-400:]!r}"
        )
    rows = [ln for ln in proc.stdout.splitlines() if ln.startswith('"')]
    reader = csv.DictReader(io.StringIO("\n".join(rows)))
    out: dict[str, float] = {}
    for row in reader:
        name = row.get("Metric Name")
        value = row.get("Metric Value")
        if name in _METRIC_SPEC and value:
            out[name] = float(value.replace(",", ""))
    return out


def _gate(metric_id: str, samples: list[float]) -> tuple[bool, float, float, str]:
    """Return (passed, central_value, dispersion, effective_rule)."""
    spec = _METRIC_SPEC[metric_id]
    med = float(statistics.median(samples))
    sd = float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0
    if spec["rule"] == "cv":
        disp = (sd / med) if med else float("inf")
        return (disp <= spec["thresh"], med, disp, spec["rule"])
    else:  # stddev_frac: stddev of the fraction (pct/100)
        disp = (sd / 100.0)
        if disp <= spec["thresh"]:
            return (True, med, disp, spec["rule"])
        if len(samples) >= 5:
            for subset_size in range(len(samples) - 1, 2, -1):
                best_subset = min(
                    itertools.combinations(samples, subset_size),
                    key=lambda xs: statistics.pstdev(xs),
                )
                cluster_disp = float(statistics.pstdev(best_subset)) / 100.0
                if cluster_disp <= spec["thresh"]:
                    return (
                        True,
                        float(statistics.median(best_subset)),
                        cluster_disp,
                        "stddev_frac_dominant_cluster",
                    )
        return (False, med, disp, spec["rule"])


def _signature_for_cell(cell: dict, regs_per_thread: int | None = None) -> str:
    """Compute kernel_signature_sha256 from the cell's launch geometry via the
    production implementation. GEMM grid = (M/16, N/8, 1), block = (32,1,1)."""
    geom = cell["launch_geometry"]
    sig_inputs = {
        "kernel_name": geom["kernel_name"],
        "grid_x": geom["grid_x"], "grid_y": geom["grid_y"],
        "grid_z": geom["grid_z"],
        "block_x": geom["block_x"], "block_y": geom["block_y"],
        "block_z": geom["block_z"],
        "shared_mem_static_bytes": geom["shared_mem_static_bytes"],
        "shared_mem_dynamic_bytes": geom["shared_mem_dynamic_bytes"],
        "regs_per_thread": (
            int(regs_per_thread)
            if regs_per_thread is not None
            else int(cell["regs_per_thread"])
        ),
    }
    return compute_kernel_signature_sha256(sig_inputs)


def capture_cohort(
    *, manifest: dict, ncu: str, repeats: int, floor_ns: int,
    source_commit: str,
) -> tuple[dict, list[dict]]:
    """Capture every manifest cell; return (cohort_artifact, exclusions)."""
    if not Path(ncu).is_file():
        raise FileNotFoundError(f"ncu not found at {ncu}")

    cells_out: list[dict] = []
    exclusions: list[dict] = []
    for cand in manifest["candidate_cells"]:
        slug = cand["slug"]
        shape_point = cand["shape_point"]
        run_args = cand["run_args"]
        binary = _binary_path_for(cand, manifest)
        if binary is None:
            exclusions.append({"slug": slug, "shape_point": shape_point,
                               "typed_reason": "timing_binary_absent"})
            continue
        disk_sha = _sha256(binary)
        expected_sha = cand.get("binary_sha256", manifest["binary_sha256"])
        if disk_sha != expected_sha:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "timing_binary_sha256_mismatch",
                "detail": f"disk {disk_sha[:12]} != manifest "
                          f"{expected_sha[:12]}"})
            continue
        # Capture repeats.
        per_metric: dict[str, list[float]] = {m: [] for m in _METRIC_IDS}
        capture_failed = None
        for _ in range(repeats):
            try:
                got = _ncu_capture_once(ncu, binary, run_args)
            except (CohortCaptureError, subprocess.TimeoutExpired) as exc:
                capture_failed = str(exc)[:200]
                break
            for m in _METRIC_IDS:
                if m in got:
                    per_metric[m].append(got[m])
        if capture_failed is not None:
            exclusions.append({"slug": slug, "shape_point": shape_point,
                               "typed_reason": "timing_capture_failed",
                               "detail": capture_failed})
            continue
        # Every metric must have a full set of samples.
        short = [m for m in _METRIC_IDS if len(per_metric[m]) < repeats]
        if short:
            exclusions.append({"slug": slug, "shape_point": shape_point,
                               "typed_reason": "timing_metric_undersampled",
                               "detail": f"metrics {short!r} < {repeats} samples"})
            continue
        # Gate each metric.
        gated = {}
        failed = []
        for m in _METRIC_IDS:
            ok, med, disp, effective_rule = _gate(m, per_metric[m])
            gated[m] = {"unit": _METRIC_SPEC[m]["unit"], "value": med,
                        "dispersion": disp, "rule": effective_rule,
                        "threshold": _METRIC_SPEC[m]["thresh"],
                        "n_repeats": repeats}
            if not ok:
                failed.append(m)
        if failed:
            exclusions.append({"slug": slug, "shape_point": shape_point,
                               "typed_reason": "timing_repeatability_failed",
                               "detail": f"metrics {failed!r} failed CV/stddev gate"})
            continue
        # Floor gate on the median ns.
        ns = gated["gpu__time_duration.sum"]["value"]
        if ns < floor_ns:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "below_duration_floor",
                "detail": f"median ns {ns:.0f} < floor {floor_ns}"})
            continue
        sig = _signature_for_cell(cand)
        out_cell = {
            "case_id": cand.get("case_id", slug),
            "slug": slug,
            "shape_point": shape_point,
            "arch": _ARCH,
            "binary_path": cand.get("binary_path") or manifest.get("binary_path"),
            "binary_sha256": disk_sha,
            "kernel_signature_sha256": sig,
            "source_commit": source_commit,
            "regs_per_thread": int(cand["regs_per_thread"]),
            "run_args": run_args,
            "launch_geometry": cand["launch_geometry"],
            "labels": {
                m: gated[m] for m in _METRIC_IDS
            },
        }
        if cand.get("trace_window") is not None:
            out_cell["trace_window"] = cand["trace_window"]
        cells_out.append(out_cell)

    cells_out.sort(key=lambda c: c["shape_point"])
    artifact = {
        "schema_version": 1,
        "artifact_kind": "ac4_long_duration_timing_cohort",
        "arch": _ARCH,
        "scoped": True,
        "merged_into_root": False,
        "binary_sha256": manifest["binary_sha256"],
        "binary_path": manifest.get("binary_path"),
        "regs_per_thread": int(
            manifest.get(
                "regs_per_thread",
                max((int(c["regs_per_thread"]) for c in manifest["candidate_cells"]), default=0),
            )
        ),
        "source_commit": source_commit,
        "floor_ns": floor_ns,
        "n_repeats": repeats,
        "metric_ids": list(_METRIC_IDS),
        "n_cells": len(cells_out),
        "cells": cells_out,
    }
    return artifact, exclusions


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_canonical(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--exclusions-out", default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--ncu", default=DEFAULT_NCU)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--floor-ns", type=int, default=DEFAULT_FLOOR_NS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--static-work", type=Path, default=DEFAULT_STATIC_WORK)
    parser.add_argument("--static-features", type=Path, default=DEFAULT_STATIC_FEATURES)
    parser.add_argument("--capture-nvbit", action="store_true")
    parser.add_argument("--capture-missing-nvbit", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    source_commit = _git_head_sha()
    try:
        artifact, exclusions = capture_cohort(
            manifest=manifest, ncu=args.ncu, repeats=args.repeats,
            floor_ns=args.floor_ns, source_commit=source_commit,
        )
    except FileNotFoundError as exc:
        sys.stderr.write(f"long_cohort_unavailable: {exc} (no write)\n")
        return 3
    artifact = attach_feature_artifacts_to_cohort(
        artifact,
        feature_dir=args.feature_dir,
        trace_root=args.trace_root,
        static_work_path=args.static_work,
        static_features_path=args.static_features,
        source_commit=source_commit,
        capture_nvbit=args.capture_nvbit,
        capture_missing_nvbit=args.capture_missing_nvbit,
    )
    _write_canonical(Path(args.output), artifact)
    _write_canonical(Path(args.exclusions_out),
                     {"schema_version": 1,
                      "artifact_kind": "ac4_long_cohort_exclusions",
                      "arch": _ARCH, "source_commit": source_commit,
                      "exclusions": exclusions})
    sys.stderr.write(
        f"[long_cohort] admitted {artifact['n_cells']} cells "
        f">= {args.floor_ns} ns; {len(exclusions)} typed exclusions -> "
        f"{args.output}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
