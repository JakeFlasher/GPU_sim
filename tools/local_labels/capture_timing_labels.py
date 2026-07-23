"""Capture execution-time labels (cycles + ns) and land them on admitted cells.

Captures the unit-tagged execution-time metrics ``gpc__cycles_elapsed.max``
(unit=cycles, DVFS-stable) and ``gpu__time_duration.sum`` (unit=ns) for the
admitted sm_86 cells via the local NCU quarantine, gates them through the SAME
per-unit repeatability machinery the label pipeline uses
(``repeatability_aggregator._aggregate_unit_tagged_metrics`` →
``_dispersion_for_rule``, coefficient-of-variation ≤ registry threshold over
``n_repeats`` ≥ floor), and SPLICES the surviving ``unit_tagged_metrics`` block
into each matching committed admitted cell IN PLACE — preserving every other
field (L2/L1 hit rates, join keys, ``source_commit``, repeatability metadata)
byte-for-byte so the frozen L2 v5 head / its byte-identity regression guard is
not perturbed.

Why in-place splice (not re-aggregate): the committed ``label_admitted.json`` was
produced by a multi-pass supersession/repair/restamp pipeline; re-running the bare
aggregator on the current per-repeat sidecars admits ZERO of the 20 committed sm_86
cells (provenance duplicates / repeatability state differ), so re-aggregation would
destroy the committed store. The unit-tagged timing block is purely ADDITIVE (no
admitted cell carries it yet), so splicing it onto the existing cells is the only
path that adds the timing labels while keeping the L2/L1/provenance fields exactly
as committed.

CLAUDE.md compliance (binding):
  - Rule 1 (NCU quarantine + local-label exception): NCU invoked ONLY here under
    ``tools/local_labels/``; the optional intermediate ``.ncu-rep`` and the
    committed label store live under ``data/local_labels/``. This is metric
    capture (``--metrics``), not a profiling section sweep.
  - Rule 2 (no driver-root): ``ncu --metrics`` does NOT lock clocks; no
    persistence / power / MIG / compute-mode mutation, no nvidia-smi clock-lock.
  - Rule 3 (no privileged CUPTI): NCU command-line metric capture, not our own
    privileged CUPTI Events/Metrics/Range/PM/PerfWorks.
  - sm_86-only: captures only arch==sm_86 admitted cells, and binds a timing
    label ONLY when the on-disk binary's sha256 MATCHES the cell's committed
    ``binary_sha256`` (so the timing label provenance agrees with the L2/L1 label
    provenance); a mismatch is a typed exclusion, never a silent bind.

CLI:
  python tools/local_labels/capture_timing_labels.py \
    [--label-admitted data/local_labels/label_admitted.json] \
    [--ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu] \
    [--repeats 5] [--arch sm_86] \
    [--exclusions-out data/local_labels/timing_label_exclusions_sm_86.json] \
    [--dry-run]

  Exit 3 (no write) when ncu is not on the host (offline/CI), so a non-3090-Ti
  host never overwrites the committed timing labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_LOCAL_LABELS = REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS))

from corpus_cell_identity import cell_variant_id  # noqa: E402

DEFAULT_LABEL_ADMITTED = "data/local_labels/label_admitted.json"
DEFAULT_EXCLUSIONS_OUT = "data/local_labels/timing_label_exclusions_sm_86.json"
DEFAULT_NCU = "/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu"
DEFAULT_REPEATS = 5
_ARCH = "sm_86"

# The two execution-time unit-tagged metrics. Both are registry entries
# with carrier=unit_tagged_metric, unit cycles/ns, coefficient_of_variation rule.
_TIMING_NCU_METRICS = ("gpc__cycles_elapsed.max", "gpu__time_duration.sum")


def _variant_fields(cell: dict) -> dict:
    """The optional build-variant identity fields (``register_cap`` /
    ``variant_id``) to stamp onto a timing-exclusion record so a register-cap
    variant cell's exclusion is DISTINCT from its siblings'. Empty for a base
    cell (byte-identical exclusion records for the base corpus)."""
    out: dict[str, Any] = {}
    if cell.get("register_cap") is not None:
        out["register_cap"] = cell["register_cap"]
    if cell.get("variant_id") is not None:
        out["variant_id"] = cell["variant_id"]
    return out

# Candidate on-disk binary locations for an admitted cell's slug.
# The dual-workload l2 tree (e.g. rmsnorm_fwd_bf16) is a sibling binary home
# alongside the l1 tree; both are simbench_bw-owned per-slug binary roots.
_BINARY_DIRS = (
    "workloads/simbench_bw/l1",
    "workloads/simbench_bw/l2",
    "workloads/portable_lcd",
)


from orchestrate_sm86_label_recapture import _app_args_for_cell


def _binary_path_for(slug: str) -> Path | None:
    for d in _BINARY_DIRS:
        p = REPO_ROOT / d / slug / f"{slug}_run"
        if p.is_file():
            return p
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ncu_capture_once(
    ncu: str, binary: Path, app_args: list[str],
) -> dict[str, float]:
    """Run one read-only NCU metric capture and return {ncu_metric: value}.

    ``--metrics`` capture only — no clock-lock, no section sweep. Parses the CSV
    table NCU writes to stdout (thousands separators stripped)."""
    proc = subprocess.run(
        [
            ncu, "--metrics", ",".join(_TIMING_NCU_METRICS),
            "--launch-count", "1", "--csv", str(binary), *app_args,
        ],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ncu rc={proc.returncode}: {proc.stderr[-400:]!r}"
        )
    rows = [ln for ln in proc.stdout.splitlines() if ln.startswith('"')]
    reader = csv.DictReader(io.StringIO("\n".join(rows)))
    out: dict[str, float] = {}
    for row in reader:
        name = row.get("Metric Name")
        value = row.get("Metric Value")
        if name in _TIMING_NCU_METRICS and value:
            out[name] = float(value.replace(",", ""))
    return out


def _capture_cell_sidecars(
    ncu: str, binary: Path, repeats: int, app_args: list[str],
) -> dict[int, list[dict]]:
    """Capture ``repeats`` repeats and return a by_repeat dict shaped exactly as
    ``repeatability_aggregator._aggregate_unit_tagged_metrics`` consumes: each
    repeat maps to a one-element list of a synthetic per-launch sidecar carrying
    a ``unit_tagged_metrics`` list."""
    from metric_registry import load_registry

    registry = load_registry()
    by_repeat: dict[int, list[dict]] = {}
    for r in range(repeats):
        captured = _ncu_capture_once(ncu, binary, app_args)
        utm: list[dict] = []
        for ncu_metric in _TIMING_NCU_METRICS:
            if ncu_metric not in captured:
                continue
            entry = registry.get_by_ncu_metric(ncu_metric)
            if entry is None:
                continue
            utm.append({
                "ncu_metric": ncu_metric,
                "unit": entry.unit,
                "value": float(captured[ncu_metric]),
            })
        by_repeat[r] = [{"unit_tagged_metrics": utm}]
    return by_repeat


def capture_and_land(
    *,
    label_admitted_path: Path,
    ncu: str,
    repeats: int,
    arch: str = _ARCH,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[dict]]:
    """Capture timing for each admitted ``arch`` cell, gate via the real
    aggregator CV machinery, and splice the surviving block in place.

    Returns ``(updated_admitted_dict, exclusions)``. Raises FileNotFoundError if
    ncu is absent (caller maps to exit 3)."""
    from repeatability_aggregator import (
        DEFAULT_REPEATS_REQUIRED,
        _aggregate_unit_tagged_metrics,
    )

    if not Path(ncu).is_file():
        raise FileNotFoundError(f"ncu not found at {ncu}")

    admitted = json.loads(label_admitted_path.read_text(encoding="utf-8"))
    exclusions: list[dict] = []
    n_landed = 0
    for cell in admitted.get("cells", []):
        if cell.get("arch") != arch:
            continue
        slug = cell["slug"]
        shape_point = cell["shape_point"]
        # Mixed-arity key: a register-cap variant cell is distinct from its
        # siblings (the variant suffix), via the shared normalizer. Base cells
        # reduce to slug@shape_point (byte-identical log + accounting).
        variant_id = cell_variant_id(cell)
        key = f"{slug}@{shape_point}@{variant_id}" if variant_id else (
            f"{slug}@{shape_point}"
        )
        vfields = _variant_fields(cell)
        binary = _binary_path_for(slug)
        if binary is None:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "timing_binary_absent",
                "detail": f"no runnable binary on disk for {slug}",
                **vfields,
            })
            continue
        disk_sha = _sha256(binary)
        if disk_sha != cell.get("binary_sha256"):
            # Provenance integrity: a timing label must bind to the SAME binary
            # the L2/L1 labels were captured against. A rebuilt binary is a typed
            # exclusion, never a silent cross-binary bind.
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "timing_binary_sha256_mismatch",
                "detail": (
                    f"on-disk binary sha256 {disk_sha[:12]} != admitted "
                    f"binary_sha256 {str(cell.get('binary_sha256'))[:12]}"
                ),
                **vfields,
            })
            continue
        try:
            app_args = _app_args_for_cell(slug=slug, shape_point=shape_point)
        except ValueError as exc:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "timing_app_arg_resolution_failed",
                "detail": str(exc)[:200],
                **vfields,
            })
            continue
        try:
            by_repeat = _capture_cell_sidecars(ncu, binary, repeats, app_args)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "timing_capture_failed",
                "detail": str(exc)[:200],
                **vfields,
            })
            continue
        gated = _aggregate_unit_tagged_metrics(by_repeat, DEFAULT_REPEATS_REQUIRED)
        gated_metrics = {m["ncu_metric"] for m in gated}
        missing = [m for m in _TIMING_NCU_METRICS if m not in gated_metrics]
        if missing:
            # A timing metric failed its CV gate or under-sampled -> the cell is
            # NOT admitted as a timing cell (typed exclusion); do not partially
            # land one timing metric without the other.
            exclusions.append({
                "slug": slug, "shape_point": shape_point,
                "typed_reason": "timing_repeatability_failed",
                "detail": (
                    f"timing metrics {missing!r} failed the CV<=threshold gate "
                    f"over {repeats} repeats"
                ),
                **vfields,
            })
            continue
        # Splice ONLY the timing block in place, preserving every other field.
        # The committed admitted cells carry their keys in sorted order and the
        # file is written with indent=2 + NO sort_keys; to keep unchanged cells
        # byte-identical and the updated cell key-sorted, rebuild this cell's
        # dict with the new key inserted in sorted position.
        cell["unit_tagged_metrics"] = sorted(
            gated, key=lambda m: m["ncu_metric"]
        )
        reordered = {k: cell[k] for k in sorted(cell)}
        cell.clear()
        cell.update(reordered)
        n_landed += 1
        print(
            f"  [{key}] landed timing: "
            + ", ".join(
                f"{m['ncu_metric']}={m['value']} (CV={m['dispersion']})"
                for m in gated
            ),
            file=sys.stderr,
        )
    print(
        f"[capture_timing] landed timing on {n_landed} {arch} cells; "
        f"{len(exclusions)} typed exclusions",
        file=sys.stderr,
    )
    return admitted, exclusions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-admitted", default=DEFAULT_LABEL_ADMITTED)
    parser.add_argument("--ncu", default=DEFAULT_NCU)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--arch", default=_ARCH)
    parser.add_argument("--exclusions-out", default=DEFAULT_EXCLUSIONS_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    label_path = Path(args.label_admitted)
    try:
        admitted, exclusions = capture_and_land(
            label_admitted_path=label_path,
            ncu=args.ncu,
            repeats=args.repeats,
            arch=args.arch,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        sys.stderr.write(
            f"timing_capture_unavailable: {exc} (no write performed)\n"
        )
        return 3

    if args.dry_run:
        sys.stderr.write("[capture_timing] --dry-run: no files written\n")
        return 0

    # Write with indent=2 + NO sort_keys to match the committed admitted store's
    # exact formatting, so cells we did not touch stay byte-identical (the frozen
    # L2 inputs are unchanged) and only the spliced timing cells diff.
    label_path.write_text(
        json.dumps(admitted, indent=2) + "\n", encoding="utf-8",
    )
    excl_artifact = {
        "schema_version": 1,
        "artifact_kind": "timing_label_exclusions",
        "arch": args.arch,
        "n_exclusions": len(exclusions),
        "exclusions": sorted(
            exclusions,
            key=lambda e: (
                e["slug"], e["shape_point"], e.get("variant_id") or "",
            ),
        ),
    }
    Path(args.exclusions_out).write_text(
        json.dumps(excl_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
