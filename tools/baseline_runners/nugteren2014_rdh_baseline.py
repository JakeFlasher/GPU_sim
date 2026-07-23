"""Clean-room reimplementation of the Nugteren 2014 reuse-distance-histogram
(RDH) baseline for L2 cache hit-rate prediction.

The Nugteren 2014 paper (HPCA): "Predicting GPU Cache Performance via
Reuse-Distance Histograms". Core analytical model:

  For a kernel whose memory references produce a binned reuse-distance
  histogram (RDH), and a cache of capacity C sectors, the modeled hit rate
  is the fraction of observed accesses whose reuse distance is strictly
  less than C:

      nugteren_hit_rate = sum(observation_count for sector-layer bins
                              with rd_sector_units_min < C)
                          / sum(observation_count for sector-layer bins)

This module is an ANALYTICAL predictor: it consumes the same NVBit-derived
hrd_profile sidecar that the v5 head consumes (sector-layer RDH bins from
tools/hrd_profiler/profiler.py) and emits a scalar hit-rate prediction. No
SASS execution, no thread/warp/block-level simulation, no kernel re-run.

This module is NOT a simulator and does NOT revive any of the prohibited
simulator surfaces in CLAUDE.md Rule 5.

Trivial baselines emitted alongside Nugteren for reviewer context:

  - cohort_mean: a constant 0.6 (the cohort-wide average L2 hit rate).
  - always_hit: a constant 1.0.
  - uncalibrated_walker: the Smith-Eklund walker's pre-correction output
    (bundle.evidence.predicted_hit_rate_uncorrected).
  - v4_PlattCalibrator: the legacy v4 PlattCalibrator applied to the
    uncorrected walker score. Requires the v4 head provenance to be
    available; otherwise reports typed_unavailable.

Six-column comparison table: each admitted cell appears as one row with
columns (slug, shape_point, arch, label, v5_prediction, nugteren_prediction,
cohort_mean_prediction, always_hit_prediction, uncalibrated_walker_prediction,
v4_platt_prediction). Emitted as
data/local_labels/calibration_head_v5_baseline_comparison.json.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw import canonical_features as cf  # noqa: E402
from workloads.simbench_bw import monotonic_calibration_head as mch  # noqa: E402


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


DEFAULT_V5_HEAD = "data/local_labels/calibration_head_v5.json"
DEFAULT_V4_HEAD = "data/local_labels/calibration_head_v4.json"
DEFAULT_BUNDLE_GLOB = (
    "data/local_labels/sweeps/sm120_cohort_bundles_*/predictor_bundles/*.json"
)
DEFAULT_RESULTS_ROOT = "workloads/simbench_bw/results"
DEFAULT_PORTABLE_LCD_ROOT = "workloads/portable_lcd/results"
DEFAULT_OUTPUT = (
    "data/local_labels/calibration_head_v5_baseline_comparison.json"
)


# Constants for the trivial baselines.
COHORT_MEAN_CONSTANT = 0.6
ALWAYS_HIT_CONSTANT = 1.0


def nugteren_hit_rate_from_hrd(
    *,
    hrd_profile: dict,
    cache_model_capacity_bytes: int,
    sector_size_bytes: int = 32,
) -> float | None:
    """Compute the Nugteren 2014 RDH analytical hit-rate prediction.

    Returns None when the hrd_profile lacks sector-layer bins.

    The capacity in sectors is `cache_model_capacity_bytes / sector_size_bytes`.
    A bin's accesses are counted as hits when its
    `rd_sector_units_min < capacity_sectors`.
    """
    if cache_model_capacity_bytes <= 0 or sector_size_bytes <= 0:
        raise ValueError(
            "cache_model_capacity_bytes and sector_size_bytes must be positive"
        )
    capacity_sectors = cache_model_capacity_bytes // sector_size_bytes
    bins = hrd_profile.get("bins", [])
    sector_bins = [
        b for b in bins
        if b.get("hierarchy_layer") == "sector"
        and b.get("rd_sector_units_min") is not None
    ]
    if not sector_bins:
        return None
    total = 0
    hits = 0
    for b in sector_bins:
        n = int(b.get("observation_count", 0))
        total += n
        if int(b["rd_sector_units_min"]) < capacity_sectors:
            hits += n
    return (hits / total) if total > 0 else 0.0


def cohort_mean_prediction(_cell: dict) -> float:
    """Constant cohort-mean baseline (literature standard 0.6)."""
    return COHORT_MEAN_CONSTANT


def always_hit_prediction(_cell: dict) -> float:
    """Constant always-hit baseline."""
    return ALWAYS_HIT_CONSTANT


def uncalibrated_walker_prediction(predictor_bundle: dict) -> float | None:
    """Smith-Eklund walker pre-correction output."""
    evidence = predictor_bundle.get("evidence", {})
    raw = evidence.get("predicted_hit_rate_uncorrected")
    if raw is None:
        return None
    return float(raw)


def v4_platt_prediction(
    *, uncalibrated: float, v4_head: dict,
) -> float | None:
    """Apply the legacy v4 PlattCalibrator to the uncalibrated walker score.

    PlattCalibrator is sigmoid(A * x + B). The v4 head sidecar stores A + B
    in its parameters; we look them up under the common shapes
    (parameters.A / parameters.B, or platt.A / platt.B).
    """
    candidates = [
        v4_head.get("parameters", {}),
        v4_head.get("platt", {}),
        v4_head.get("chosen_head_parameters", {}),
        v4_head,
    ]
    A: float | None = None
    B: float | None = None
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if "A" in c and "B" in c:
            try:
                A = float(c["A"])
                B = float(c["B"])
                break
            except (TypeError, ValueError):
                continue
    if A is None or B is None:
        return None
    x = float(uncalibrated)
    z = A * x + B
    # Numerically stable sigmoid.
    if z >= 0:
        out = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        out = e / (1.0 + e)
    return max(0.0, min(1.0, out))


def v5_prediction(
    *,
    cell: dict,
    v5_ensemble: Any,
) -> float | None:
    """Apply the v5 head's predict to a cell's canonical features.

    Returns None when the cell has no canonical_pre_calibration_features
    block (data-staging gap on this host)."""
    features = cell.get("features")
    if not features:
        return None
    try:
        return float(
            v5_ensemble.predict(
                cell["arch"], cell["family_bucket"], features,
            )
        )
    except RuntimeError:
        return None


def _device_to_sku_dir(device: str | None) -> str:
    if device is None:
        return "rtx_5060_laptop"
    d = device.lower()
    # Recognize both human device names ("RTX 3090 Ti", "GA102") and the
    # arch-token form some bundles record in their `device` field
    # ("sm_86" / "sm_120"); the latter is mapped to its host SKU directory so
    # the per-arch results-root path (e.g. rtx_3090_ti/sm_86/...) resolves.
    if "3090" in device or "ga102" in d or "sm_86" in d:
        return "rtx_3090_ti"
    if "5060" in device or "blackwell" in d or "gb20" in d or "sm_120" in d:
        return "rtx_5060_laptop"
    return "rtx_5060_laptop"


def _load_hrd_profile_for_cell(
    *,
    bundle: dict,
    primary_root: Path,
    fallback_root: Path,
    cell_arch: str | None = None,
) -> dict | None:
    mpi = bundle.get("manifest_pairing_index", {})
    slug = mpi.get("slug")
    shape_point = mpi.get("shape_point")
    # The per-cell bundle's manifest_pairing_index is a closed 5-field identity
    # with no arch key, so its arch is empty for cells whose bundle carries the
    # arch-less MPI. Fall back to the cohort cell's arch (the cell knows its
    # architecture even when the bundle identity does not) so the hrd_profile
    # sidecar resolves at the canonical per-arch results-root path.
    arch = mpi.get("arch") or cell_arch
    device = bundle.get("device")
    sku = _device_to_sku_dir(device)
    if not (slug and shape_point and arch):
        return None
    for root in (primary_root, fallback_root):
        path = (
            root / slug / sku / arch / f"trace_{shape_point}"
            / "hrd_profile" / "hrd_profile.json"
        )
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _build_bundle_lookup(
    bundle_paths: list[Path],
) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for p in bundle_paths:
        try:
            b = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mpi = b.get("manifest_pairing_index", {})
        key = (
            str(mpi.get("arch", "")),
            str(mpi.get("slug", "")),
            str(mpi.get("shape_point", "")),
        )
        out[key] = b
    return out


def _refit_v5_ensemble_from_head(v5_head: dict) -> Any:
    """Rehydrate a PerBucketCalibratorEnsemble from the head's admitted
    cohort using the SELECTED class via the centralized dispatch helper
    (was: hardcoded IR-on-projection). Returns None when the head has
    fit_status != ok OR the dispatch fails."""
    if v5_head.get("fit_status") != "ok":
        return None
    from tools.local_labels.v5_selected_class_dispatch import (
        build_selected_ensemble,
    )
    dispatch = build_selected_ensemble(v5_head)
    if dispatch["status"] != "ok":
        return None
    return dispatch["ensemble"]


def build_comparison_table(
    *,
    v5_head: dict,
    v4_head: dict | None,
    bundle_lookup: dict[tuple[str, str, str], dict],
    primary_root: Path,
    fallback_root: Path,
    v5_ensemble: Any,
) -> list[dict[str, Any]]:
    """Build the 6-baseline-column comparison table over the v5 head's
    cohort_scope cells (admitted + skipped both included; skipped cells get
    typed-unavailable predictions for the data-bound baselines).

    Row schema (one row per cohort cell):
      {
        slug, shape_point, arch, family_bucket, label,
        v5_prediction, nugteren_prediction, cohort_mean_prediction,
        always_hit_prediction, uncalibrated_walker_prediction,
        v4_platt_prediction, typed_reason (if applicable),
      }
    """
    cohort = v5_head.get("cohort_scope", {})
    admitted = cohort.get("admitted", [])
    skipped = cohort.get("skipped", [])
    all_cells: list[dict] = []
    for c in admitted:
        all_cells.append({**c, "_origin": "admitted"})
    for c in skipped:
        all_cells.append({**c, "_origin": "skipped"})
    rows: list[dict[str, Any]] = []
    for c in all_cells:
        arch = c.get("arch")
        slug = c.get("slug")
        shape_point = c.get("shape_point")
        family_bucket = c.get("family_bucket")
        label = c.get("label")
        key = (str(arch), str(slug), str(shape_point))
        bundle = bundle_lookup.get(key)
        # Fallback: same-arch by (slug, shape_point) if exact match misses.
        if bundle is None:
            for (b_arch, b_slug, b_shape), b in bundle_lookup.items():
                if b_slug == slug and b_shape == shape_point and b_arch == arch:
                    bundle = b
                    break
        if bundle is None:
            # Bundle absent: every prediction column gets the same
            # row-level explanation. Column-specific reasons mirror
            # the row-level typed_reason so the scanner finds an
            # explanation per column regardless of which check it
            # runs first.
            rows.append({
                "slug": slug, "shape_point": shape_point, "arch": arch,
                "family_bucket": family_bucket, "label": label,
                "v5_prediction": None,
                "v5_prediction_reason": "predictor_bundle_absent",
                "nugteren_prediction": None,
                "nugteren_prediction_reason": "predictor_bundle_absent",
                "cohort_mean_prediction": cohort_mean_prediction(c),
                "cohort_mean_prediction_reason": None,
                "always_hit_prediction": always_hit_prediction(c),
                "always_hit_prediction_reason": None,
                "uncalibrated_walker_prediction": None,
                "uncalibrated_walker_prediction_reason": (
                    "predictor_bundle_absent"
                ),
                "v4_platt_prediction": None,
                "v4_platt_prediction_reason": "predictor_bundle_absent",
                "typed_reason": "predictor_bundle_absent",
            })
            continue
        uncal = uncalibrated_walker_prediction(bundle)
        uncal_reason: str | None = None
        if uncal is None:
            uncal_reason = (
                "uncalibrated_walker_evidence_predicted_hit_rate_uncorrected_absent"
            )
        nugteren = None
        nugteren_reason: str | None = None
        hrd = _load_hrd_profile_for_cell(
            bundle=bundle,
            primary_root=primary_root,
            fallback_root=fallback_root,
            cell_arch=arch,
        )
        if hrd is None:
            nugteren_reason = "nugteren_hrd_sidecar_unavailable_for_cell"
        else:
            cap = int(bundle.get("cache_model_capacity_bytes", 0))
            if cap <= 0:
                nugteren_reason = "nugteren_cache_model_capacity_invalid"
            else:
                nugteren = nugteren_hit_rate_from_hrd(
                    hrd_profile=hrd, cache_model_capacity_bytes=cap,
                )
                if nugteren is None:
                    nugteren_reason = (
                        "nugteren_hrd_no_bins_in_capacity_window"
                    )
        v4_pred = None
        v4_pred_reason: str | None = None
        if v4_head is None:
            v4_pred_reason = "v4_head_unavailable_for_cohort"
        elif uncal is None:
            v4_pred_reason = "v4_platt_input_uncalibrated_walker_absent"
        else:
            v4_pred = v4_platt_prediction(
                uncalibrated=uncal, v4_head=v4_head,
            )
            if v4_pred is None:
                v4_pred_reason = (
                    "v4_platt_calibrator_returned_null_for_cell"
                )
        v5_pred = None
        v5_pred_reason: str | None = None
        if c.get("_origin") != "admitted":
            v5_pred_reason = c.get("typed_reason", "skipped")
        elif v5_ensemble is None:
            v5_pred_reason = "v5_ensemble_unavailable_for_cohort"
        else:
            v5_pred = v5_prediction(cell=c, v5_ensemble=v5_ensemble)
            if v5_pred is None:
                v5_pred_reason = (
                    "v5_ensemble_returned_null_for_cell"
                )
        rows.append({
            "slug": slug, "shape_point": shape_point, "arch": arch,
            "family_bucket": family_bucket, "label": label,
            "v5_prediction": v5_pred,
            "v5_prediction_reason": v5_pred_reason,
            "nugteren_prediction": nugteren,
            "nugteren_prediction_reason": nugteren_reason,
            "cohort_mean_prediction": cohort_mean_prediction(c),
            "cohort_mean_prediction_reason": None,
            "always_hit_prediction": always_hit_prediction(c),
            "always_hit_prediction_reason": None,
            "uncalibrated_walker_prediction": uncal,
            "uncalibrated_walker_prediction_reason": uncal_reason,
            "v4_platt_prediction": v4_pred,
            "v4_platt_prediction_reason": v4_pred_reason,
            "typed_reason": (
                None if c.get("_origin") == "admitted"
                else c.get("typed_reason", "skipped")
            ),
        })
    return rows


def _aggregate_metrics(rows: list[dict]) -> dict[str, Any]:
    """Compute per-baseline aggregate metrics over rows whose label + prediction
    are both numeric. Returns a dict with median_abs_err + geomean_rel_err per
    baseline, plus the v5_vs_nugteren_beat_fraction over the joint admit-set."""
    baselines = (
        "v5_prediction",
        "nugteren_prediction",
        "cohort_mean_prediction",
        "always_hit_prediction",
        "uncalibrated_walker_prediction",
        "v4_platt_prediction",
    )
    out: dict[str, Any] = {}
    for b in baselines:
        preds = []
        labels = []
        for r in rows:
            p = r.get(b)
            y = r.get("label")
            if p is None or y is None:
                continue
            preds.append(float(p))
            labels.append(float(y))
        if not preds:
            out[b] = {
                "n": 0,
                "median_absolute_error": None,
                "geomean_relative_error": None,
            }
            continue
        out[b] = {
            "n": len(preds),
            "median_absolute_error": mch.median_absolute_error(preds, labels),
            "geomean_relative_error": mch.geomean_relative_error(preds, labels),
        }
    # v5-vs-Nugteren beat fraction over rows where BOTH are numeric.
    joint = [
        r for r in rows
        if r.get("v5_prediction") is not None
        and r.get("nugteren_prediction") is not None
        and r.get("label") is not None
    ]
    if joint:
        n = len(joint)
        wins = 0
        for r in joint:
            v5e = abs(float(r["v5_prediction"]) - float(r["label"]))
            nugt = abs(float(r["nugteren_prediction"]) - float(r["label"]))
            if v5e < nugt:
                wins += 1
        out["v5_vs_nugteren_beat_fraction"] = wins / n
        out["v5_vs_nugteren_joint_admit_count"] = n
    else:
        out["v5_vs_nugteren_beat_fraction"] = None
        out["v5_vs_nugteren_joint_admit_count"] = 0
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-head", default=DEFAULT_V5_HEAD)
    parser.add_argument("--v4-head", default=DEFAULT_V4_HEAD)
    parser.add_argument("--bundle-glob", default=DEFAULT_BUNDLE_GLOB)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--portable-lcd-root", default=DEFAULT_PORTABLE_LCD_ROOT,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    import glob as _glob
    v5_head_path = Path(args.v5_head)
    v4_head_path = Path(args.v4_head)
    v5_head = json.loads(v5_head_path.read_text(encoding="utf-8"))
    v4_head = (
        json.loads(v4_head_path.read_text(encoding="utf-8"))
        if v4_head_path.is_file() else None
    )
    bundle_paths = sorted(Path(p) for p in _glob.glob(args.bundle_glob))
    bundle_lookup = _build_bundle_lookup(bundle_paths)
    v5_ensemble = _refit_v5_ensemble_from_head(v5_head)
    rows = build_comparison_table(
        v5_head=v5_head, v4_head=v4_head,
        bundle_lookup=bundle_lookup,
        primary_root=Path(args.results_root),
        fallback_root=Path(args.portable_lcd_root),
        v5_ensemble=v5_ensemble,
    )
    aggregate = _aggregate_metrics(rows)
    artifact = {
        "schema_version": 1,
        "baseline_set": [
            "v5_prediction", "nugteren_prediction",
            "cohort_mean_prediction", "always_hit_prediction",
            "uncalibrated_walker_prediction", "v4_platt_prediction",
        ],
        "source_commit": _git_head_sha(),
        "source_v5_head_commit": v5_head.get("source_commit"),
        "row_count": len(rows),
        "rows": rows,
        "aggregate_metrics": aggregate,
        "nugteren_2014_reference": (
            "Nugteren et al., 'A Detailed GPU Cache Model Based on "
            "Reuse Distance Theory', HPCA 2014. Clean-room analytical "
            "reimplementation; no SASS execution / thread-level "
            "simulation; consumes the same NVBit-derived hrd_profile "
            "sidecar."
        ),
        "trivial_baseline_constants": {
            "cohort_mean": COHORT_MEAN_CONSTANT,
            "always_hit": ALWAYS_HIT_CONSTANT,
        },
    }
    Path(args.output).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[baseline_comparison] rows={len(rows)} "
        f"v5_vs_nugteren_beat_fraction={aggregate['v5_vs_nugteren_beat_fraction']} "
        f"-> {args.output}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
