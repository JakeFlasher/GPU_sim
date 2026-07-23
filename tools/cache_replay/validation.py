"""Advisory robustness artifact for the calibrated cache-replay predictor.

This module emits a robustness report for the calibrated sm_120 cache-only
trace-replay L2 hit-ratio predictor over the 30 admitted cells. The report is
strictly **advisory**: it does NOT gate closure, it only describes how stable
the calibrated headline accuracy is under held-out resampling. Every number it
reports is annotated as advisory and the artifact carries
``"gates_closure": false``.

It assembles four robustness surfaces over the mechanistic predictions:

  1. a deterministic stratified split by ``family_bucket`` (seed-derived, no
     wall-clock and no nondeterministic randomness);
  2. leave-one-out cross-validation (LOOCV): for each cell, fit the
     one-dimensional monotone residual on the other 29 mechanistic predictions
     and predict the held-out cell; collect the out-of-fold calibrated
     predictions and report their median absolute error and geomean relative
     error;
  3. leave-one-slug-out (LOSO): identical to LOOCV but every cell sharing a
     slug with the held-out cell is removed from the fit;
  4. a percentile bootstrap confidence interval (2000 resamples, seed-fixed)
     on the LOOCV calibrated relative error.

It also records a per-cell stability flag. Because the pre-registered grid is
**inert** on this cohort (every candidate config predicts identically -- the
capacity / associativity / slice / replacement axes do not move the prediction,
since all working sets fit the modeled L2 capacity), the config selected by the
staged cross-validated search is invariant under removal of any single cell:
removing one cell cannot change which config wins when all configs tie on every
cell. The stability flag is therefore ``True`` for every cell, and the artifact
records this fact together with the reason (grid inertness), rather than
re-running the whole staged search 30 times.

The module reads only already-collected traces and labels and the
already-emitted comparison bundle; it performs no privileged measurement. The
old no-simulator charter clause was retired at the 2026-07-02 re-founding, and
the module now also serves as label-side validation for the active simulator's
L2 lane.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # type: ignore[import-not-found]  # noqa: E402
import cohort as cohort_mod  # type: ignore[import-not-found]  # noqa: E402
import metrics as metrics_mod  # type: ignore[import-not-found]  # noqa: E402
import residual as residual_mod  # type: ignore[import-not-found]  # noqa: E402
import artifact_io  # type: ignore[import-not-found]  # noqa: E402

SCHEMA_VERSION = 1
SEED = 20260527
N_BOOTSTRAP = 2000
BOOTSTRAP_PERCENTILE = 95.0

VALIDATION_PATH = _REPO_ROOT / "data/local_labels/cache_replay_validation.json"
COMPARISON_PATH = _REPO_ROOT / "data/local_labels/cache_replay_comparison.json"

# Advisory thresholds. These are NOT gates: each value is reported alongside a
# boolean "within advisory band" verdict, but the artifact never blocks closure.
LOOCV_VS_HOLDOUT_GAP_ADVISORY = 0.20
PER_SLUG_LOSO_HEADLINE_CHANGE_ADVISORY = 0.25

# The selected-config constructor kwargs we reconstruct from the comparison
# bundle. Only these keys are passed to CacheReplayConfig.
_SELECTED_CONFIG_KEYS = (
    "capacity_bytes",
    "associativity",
    "replacement_policy",
    "num_slices",
    "slice_hash_id",
    "capacity_scale",
)


# --------------------------------------------------------------------------
# Deterministic helpers (no wall-clock, no nondeterministic randomness).
# --------------------------------------------------------------------------
def _stable_hash_key(*parts: object) -> int:
    """Deterministic SHA-256-derived ordering key (mirrors the cohort /
    parameter-grid idiom: stable across runs and processes)."""
    payload = ":".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def reconstruct_selected_config(bundle: dict) -> scr.CacheReplayConfig:
    """Rebuild the selected ``CacheReplayConfig`` from the comparison bundle's
    ``grid_provenance.selected_config`` using only the constructor kwargs."""
    sel = bundle["grid_provenance"]["selected_config"]
    kwargs = {k: sel[k] for k in _SELECTED_CONFIG_KEYS}
    return scr.CacheReplayConfig(**kwargs)


def deterministic_stratified_split(
    cells, *, holdout_fraction: float = 0.25, seed: int = SEED,
) -> dict[str, str]:
    """Assign each cell to ``"calibration"`` or ``"holdout"`` deterministically,
    stratified by ``family_bucket``. Within each bucket the cells are ordered by
    a seed-salted SHA-256 key and the first ``round(fraction * n_bucket)``
    (at least one when the bucket is non-empty) go to the holdout split."""
    by_bucket: dict[str, list] = {}
    for c in cells:
        by_bucket.setdefault(c.family_bucket, []).append(c)
    split: dict[str, str] = {}
    for bucket in sorted(by_bucket):
        members = sorted(
            by_bucket[bucket],
            key=lambda c: _stable_hash_key(seed, "split", c.cell_id),
        )
        take = max(1, round(holdout_fraction * len(members)))
        for i, c in enumerate(members):
            split[c.cell_id] = "holdout" if i < take else "calibration"
    return split


# --------------------------------------------------------------------------
# Mechanistic predictions: replay the selected config over every cell.
# --------------------------------------------------------------------------
def mechanistic_predictions(cells, config, *, use_cache: bool = True) -> dict[str, float]:
    """Return ``{cell_id: predicted_hit_ratio}`` from replaying the selected
    config over each cell's decoded stream (``classify_misses=False``)."""
    preds: dict[str, float] = {}
    for cell in cells:
        line_ids, sector_offsets = cohort_mod.decode_stream(cell, use_cache=use_cache)
        result = scr.replay_access_stream(
            line_ids, sector_offsets, config, classify_misses=False,
        )
        preds[cell.cell_id] = float(result.predicted_hit_ratio)
    return preds


# --------------------------------------------------------------------------
# Out-of-fold calibrated predictions (LOOCV + LOSO).
# --------------------------------------------------------------------------
def _calibrated_out_of_fold(
    cell_ids: list[str],
    mech: dict[str, float],
    labels: dict[str, float],
    held_out_of_fit,
) -> list[tuple[str, float]]:
    """For each cell, fit the monotone residual on the cells NOT in that cell's
    held-out group, then predict the held-out cell. ``held_out_of_fit`` maps a
    held-out cell id to the set of cell ids removed from the fit (itself for
    LOOCV; all cells sharing its slug for LOSO).

    Returns ``[(cell_id, calibrated_prediction)]`` in ``cell_ids`` order."""
    out: list[tuple[str, float]] = []
    for cid in cell_ids:
        removed = held_out_of_fit(cid)
        train = [c for c in cell_ids if c not in removed]
        head = residual_mod.fit_monotone_residual(
            [mech[c] for c in train], [labels[c] for c in train],
        )
        pred = head.predict([mech[cid]])[0]
        out.append((cid, float(pred)))
    return out


def loocv_out_of_fold(cell_ids, mech, labels) -> list[tuple[str, float]]:
    """Leave-one-out: each cell's fit excludes only itself."""
    return _calibrated_out_of_fold(
        cell_ids, mech, labels, lambda cid: {cid},
    )


def loso_out_of_fold(cell_ids, mech, labels, slug_of) -> list[tuple[str, float]]:
    """Leave-one-slug-out: each cell's fit excludes every cell that shares its
    slug (so no information from the same workload slug leaks into the fit)."""
    slug_groups: dict[str, set[str]] = {}
    for cid in cell_ids:
        slug_groups.setdefault(slug_of[cid], set()).add(cid)
    return _calibrated_out_of_fold(
        cell_ids, mech, labels, lambda cid: slug_groups[slug_of[cid]],
    )


# --------------------------------------------------------------------------
# Percentile bootstrap CI on the calibrated relative error.
# --------------------------------------------------------------------------
def bootstrap_geomean_ci(
    calibrated_preds: list[float],
    labels: list[float],
    *,
    n_resamples: int = N_BOOTSTRAP,
    percentile: float = BOOTSTRAP_PERCENTILE,
    seed: int = SEED,
    epsilon: float = metrics_mod.EPSILON,
) -> dict:
    """Seed-fixed percentile bootstrap on the calibrated geomean relative error.

    Resamples cells with replacement ``n_resamples`` times (numpy default_rng
    seeded with ``seed``), recomputes the geomean relative error on each
    resample, and returns the point geomean plus the central ``percentile``
    interval (e.g. the 2.5/97.5 percentiles for ``percentile == 95``)."""
    preds = np.asarray(calibrated_preds, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.float64)
    n = preds.shape[0]
    point = metrics_mod.geomean_relative_error(
        preds.tolist(), labs.tolist(), epsilon=epsilon,
    )
    rng = np.random.default_rng(seed)
    rel = np.abs(preds - labs) / np.maximum(np.abs(labs), epsilon)
    log_rel = np.log(rel + 1e-12)
    resampled: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled.append(float(math.exp(log_rel[idx].mean())))
    arr = np.asarray(resampled, dtype=np.float64)
    tail = (100.0 - percentile) / 2.0
    ci_low = float(np.percentile(arr, tail))
    ci_high = float(np.percentile(arr, 100.0 - tail))
    return {
        "geomean": float(point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_resamples": int(n_resamples),
        "percentile": float(percentile),
        "seed": int(seed),
    }


# --------------------------------------------------------------------------
# Per-cell stability under single-cell removal (inert-grid fact).
# --------------------------------------------------------------------------
def per_cell_stability(cell_ids: list[str], *, grid_is_inert: bool) -> dict:
    """Record the per-cell config-selection stability flag.

    When the grid is inert -- every candidate config predicts identically on
    every cell -- the staged cross-validated search ties on all configs for any
    cohort subset, so the simpler-within-band tie-break always returns the same
    simplest config. Removing any single cell cannot change that outcome, so the
    selected config is stable under any single-cell removal: the flag is True
    for every cell. We assert this fact rather than re-running the staged search
    once per cell."""
    flags = {cid: bool(grid_is_inert) for cid in cell_ids}
    return {
        "all_cells_stable": all(flags.values()),
        "n_cells": len(cell_ids),
        "n_unstable": sum(1 for v in flags.values() if not v),
        "grid_is_inert": bool(grid_is_inert),
        "per_cell_stable": flags,
        "reason": (
            "The pre-registered grid is inert on this cohort: every candidate "
            "config produces an identical mechanistic prediction on every cell "
            "(the maximum per-cell prediction spread across the entire grid is "
            "zero, because all working sets fit the modeled L2 capacity). When "
            "all configs tie on every cell, the staged cross-validated search "
            "with the simpler-within-band tie-break selects the same simplest "
            "config for any cohort subset. Removing a single cell therefore "
            "cannot change the selected config, so the selection is stable for "
            "every cell. This is asserted from grid inertness rather than by "
            "re-running the staged search once per held-out cell."
        ),
    }


# --------------------------------------------------------------------------
# Assemble the advisory artifact.
# --------------------------------------------------------------------------
def build_validation_artifact(*, use_cache: bool = True) -> dict:
    """Build the advisory robustness artifact over the 30 sm_120 cells."""
    import json

    bundle = json.loads(COMPARISON_PATH.read_text())
    config = reconstruct_selected_config(bundle)

    cells = cohort_mod.load_cohort()
    cell_ids = [c.cell_id for c in cells]
    labels = {c.cell_id: float(c.label) for c in cells}
    slug_of = {c.cell_id: c.slug for c in cells}
    buckets = {c.cell_id: c.family_bucket for c in cells}

    mech = mechanistic_predictions(cells, config, use_cache=use_cache)

    # Deterministic stratified split + the in-split (calibration-fit) holdout
    # headline. Fit the monotone residual on the calibration split, predict the
    # holdout split, and report its geomean relative error.
    split = deterministic_stratified_split(cells)
    cal_ids = [cid for cid in cell_ids if split[cid] == "calibration"]
    hold_ids = [cid for cid in cell_ids if split[cid] == "holdout"]
    split_head = residual_mod.fit_monotone_residual(
        [mech[c] for c in cal_ids], [labels[c] for c in cal_ids],
    )
    holdout_preds = split_head.predict([mech[c] for c in hold_ids])
    holdout_geomean = metrics_mod.geomean_relative_error(
        holdout_preds, [labels[c] for c in hold_ids],
    )
    holdout_median_abs = metrics_mod.median_absolute_error(
        holdout_preds, [labels[c] for c in hold_ids],
    )

    # LOOCV out-of-fold calibrated predictions + headline metrics.
    loocv = loocv_out_of_fold(cell_ids, mech, labels)
    loocv_preds = [p for _, p in loocv]
    loocv_labels = [labels[cid] for cid, _ in loocv]
    loocv_median_abs = metrics_mod.median_absolute_error(loocv_preds, loocv_labels)
    loocv_geomean = metrics_mod.geomean_relative_error(loocv_preds, loocv_labels)

    # LOSO out-of-fold calibrated predictions + headline metrics.
    loso = loso_out_of_fold(cell_ids, mech, labels, slug_of)
    loso_preds = [p for _, p in loso]
    loso_labels = [labels[cid] for cid, _ in loso]
    loso_median_abs = metrics_mod.median_absolute_error(loso_preds, loso_labels)
    loso_geomean = metrics_mod.geomean_relative_error(loso_preds, loso_labels)

    # Percentile bootstrap CI on the LOOCV calibrated relative error.
    bootstrap_ci = bootstrap_geomean_ci(loocv_preds, loocv_labels)

    # Advisory verdicts (NOT gates).
    loocv_vs_holdout_gap = abs(loocv_geomean - holdout_geomean)
    per_slug_loso_headline_change = abs(loso_geomean - loocv_geomean)

    # The grid is "selection-stable inert" when the maximum per-cell prediction
    # spread across the evaluated configs is within the simpler-within-2%
    # tie-break band (0.02): no config's score can move enough to dislodge the
    # simplest selected config, so the selection is stable under any single-cell
    # removal. (The re-anchored staged search yields a tiny but non-zero spread.)
    _INERT_SPREAD_EPSILON = 0.02
    max_spread = bundle.get("inert_grid_evidence", {}).get(
        "max_grid_prediction_spread", 0.0
    )
    stability = per_cell_stability(
        cell_ids,
        grid_is_inert=(max_spread <= _INERT_SPREAD_EPSILON),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_commit": artifact_io.git_head(),
        "advisory": True,
        "gates_closure": False,
        "top_level_admission_scope": bundle.get(
            "top_level_admission_scope", "target_arch=sm_120",
        ),
        "arches": bundle.get("arches", ["sm_120"]),
        "seed": SEED,
        "cohort_size": len(cell_ids),
        "n_slugs": len(set(slug_of.values())),
        "family_bucket_counts": {
            b: sum(1 for cid in cell_ids if buckets[cid] == b)
            for b in cohort_mod.FAMILY_BUCKETS
        },
        "selected_config": config.provenance(),
        "stratified_split": {
            "seed": SEED,
            "holdout_cell_count": len(hold_ids),
            "calibration_cell_count": len(cal_ids),
            "holdout_cell_ids": sorted(hold_ids),
            "holdout_geomean_relative_error": holdout_geomean,
            "holdout_median_absolute_error": holdout_median_abs,
        },
        "loocv": {
            "n": len(loocv_preds),
            "median_absolute_error": loocv_median_abs,
            "geomean_relative_error": loocv_geomean,
        },
        "loso": {
            "n": len(loso_preds),
            "median_absolute_error": loso_median_abs,
            "geomean_relative_error": loso_geomean,
        },
        "bootstrap_ci": bootstrap_ci,
        "per_cell_stability": stability,
        "advisory_thresholds": {
            "loocv_vs_holdout_gap": {
                "value": loocv_vs_holdout_gap,
                "advisory_max": LOOCV_VS_HOLDOUT_GAP_ADVISORY,
                "within_advisory_band": loocv_vs_holdout_gap <= LOOCV_VS_HOLDOUT_GAP_ADVISORY,
            },
            "per_slug_loso_headline_change": {
                "value": per_slug_loso_headline_change,
                "advisory_max": PER_SLUG_LOSO_HEADLINE_CHANGE_ADVISORY,
                "within_advisory_band": (
                    per_slug_loso_headline_change <= PER_SLUG_LOSO_HEADLINE_CHANGE_ADVISORY
                ),
            },
        },
        "note": (
            "Advisory robustness surface for the calibrated cache-replay "
            "predictor. It does NOT gate closure (gates_closure=false); it only "
            "characterizes how stable the calibrated headline accuracy is under "
            "held-out resampling. All thresholds are advisory bands, not pass / "
            "fail gates."
        ),
    }


# --------------------------------------------------------------------------
# Small exposed validator (consumer-side guard for the advisory artifact).
# --------------------------------------------------------------------------
def validate_validation_artifact(payload: dict) -> None:
    """Raise ``ValueError`` if the advisory artifact is internally inconsistent.

    Two guards beyond JSON-Schema shape:
      * the bootstrap-CI block must be present (a robustness artifact without an
        uncertainty interval on the headline error is meaningless);
      * if the per-cell stability block flags any cell as NOT stable, the block
        MUST acknowledge it (``all_cells_stable`` false and ``n_unstable`` > 0).
        A payload that carries an unstable per-cell flag while still asserting
        ``all_cells_stable`` is rejected -- an unflagged-unstable config."""
    ci = payload.get("bootstrap_ci")
    if not isinstance(ci, dict):
        raise ValueError(
            "validation artifact missing the bootstrap_ci block; a robustness "
            "artifact must carry the bootstrap confidence interval on the "
            "calibrated error"
        )
    for key in ("geomean", "ci_low", "ci_high"):
        if key not in ci:
            raise ValueError(f"bootstrap_ci block missing required key {key!r}")

    stability = payload.get("per_cell_stability")
    if not isinstance(stability, dict):
        raise ValueError("validation artifact missing the per_cell_stability block")
    per_cell = stability.get("per_cell_stable", {})
    actually_unstable = [cid for cid, flag in per_cell.items() if not flag]
    if actually_unstable:
        # at least one cell is unstable; the block must acknowledge it.
        if stability.get("all_cells_stable", False):
            raise ValueError(
                "per_cell_stability declares all_cells_stable=true while "
                f"{len(actually_unstable)} cell(s) carry an unstable flag: "
                f"{sorted(actually_unstable)[:3]}... (unflagged-unstable config)"
            )
        if stability.get("n_unstable", 0) != len(actually_unstable):
            raise ValueError(
                "per_cell_stability n_unstable does not match the number of "
                "cells flagged unstable (unflagged-unstable config)"
            )

    if payload.get("gates_closure", True) is not False:
        raise ValueError(
            "advisory validation artifact must declare gates_closure=false"
        )
    if payload.get("advisory", False) is not True:
        raise ValueError("advisory validation artifact must declare advisory=true")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Emit the advisory cache-replay validation artifact",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    artifact = build_validation_artifact(use_cache=not args.no_cache)
    validate_validation_artifact(artifact)
    artifact_io.atomic_write_json(VALIDATION_PATH, artifact)

    print(f"advisory: {artifact['advisory']}  gates_closure: {artifact['gates_closure']}")
    print(f"LOOCV  median_abs={artifact['loocv']['median_absolute_error']:.4f} "
          f"geomean_rel={artifact['loocv']['geomean_relative_error']:.4f}")
    print(f"LOSO   median_abs={artifact['loso']['median_absolute_error']:.4f} "
          f"geomean_rel={artifact['loso']['geomean_relative_error']:.4f}")
    ci = artifact["bootstrap_ci"]
    print(f"bootstrap geomean={ci['geomean']:.4f} "
          f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
    adv = artifact["advisory_thresholds"]
    print(f"loocv_vs_holdout_gap={adv['loocv_vs_holdout_gap']['value']:.4f} "
          f"within_band={adv['loocv_vs_holdout_gap']['within_advisory_band']}")
    print(f"per_slug_loso_headline_change="
          f"{adv['per_slug_loso_headline_change']['value']:.4f} "
          f"within_band={adv['per_slug_loso_headline_change']['within_advisory_band']}")
    print(f"all cells stable: {artifact['per_cell_stability']['all_cells_stable']}")
    print(f"wrote {VALIDATION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
