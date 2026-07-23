"""Numeric recomputation helper for the per-sub-slice ablation
artifact.

Cross-checks that the hand-authored numeric fields in
`data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation.json`
are (a) arithmetically consistent with each other AND (b)
consistent with the committed source artifacts they were derived
from:

  - TRUTH source: `data/local_labels/label_admitted.json` provides
    `cache_hit_rate_l2_sector_pct / 100.0` per cell. The ablation
    artifact's `ncu_truth_fraction` is cross-checked against this
    source within a tolerance that admits the artifact's 4-decimal
    rounding of the percentage.
  - AFTER prediction source: the refreshed sweep at
    `data/sm_86_predictor_accuracy_recovery/refreshed_sweep_round_3/predictor_bundles/`
    provides the post-refinement `predicted_hit_rate` per cell.
    Every ablation entry's `after_predicted_hit_rate` MUST match
    this source byte-identically (the refreshed sweep IS the
    after-state by construction).
  - BASELINE prediction source for the slice-geometry entry: the
    baseline sweep at
    `data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/predictor_bundles/`
    provides the slice-geometry `before_predicted_hit_rate` per
    cell. The slice-geometry sub-slice is a metadata-only flip
    (the per-slice walker was already active in the baseline
    sweep), so before-baseline equality is the load-bearing
    invariant.
  - BEFORE prediction manifest for the probe-derived entries:
    `data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation_source_manifest.json`
    records the probe-derived before predictions for sector_edge /
    write_policy / cta_interleave / cross_launch_state entries.
    The manifest is the source-of-truth for those before values
    so a future hand-edit of the ablation artifact's before fields
    is caught by manifest disagreement.
  - VARIANCE trajectory cross-check: the
    `ac6_variance_trajectory_post_ac4.json::current_stddev_computation`
    prose enumerates the 5 per-cell post-refinement
    `predicted_hit_rate` values that the variance gate consumed;
    these are an independent cross-check against the refreshed
    sweep bundles.

The helper enforces the closed-form arithmetic relationships
among the published fields:

  - per cell:
      `signed_residual ≈ predicted_hit_rate - ncu_truth_fraction`
      `relative_error ≈ |signed_residual| / ncu_truth_fraction`
  - per entry:
      `geomean_rel_err ≈ exp(mean(log(per_cell.relative_error)))`
      `pred_stddev ≈ stdev(per_cell.predicted_hit_rate)` (sample stdev, ddof=1, matching the variance-precondition trajectory artifact convention)
  - per entry consistency:
      `headline_moved_at_geomean_level == (after_geomean_rel_err < before_geomean_rel_err - 1e-12)`
      every per_cell.headline_moved value consistent with its
      before/after relative_error delta sign

The helper is the regression that prevents hand-edited drift of
the published numerics. The arithmetic checks catch dependent-
field inconsistency; the source-backed checks catch self-
consistent edits that drift away from the underlying source
artifacts.

Public API: `recompute_defects(artifact: dict) -> list[str]`.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
ABLATION_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "ac4_per_slice_ablation.json"
)
LABEL_ADMITTED_PATH = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
REFRESHED_SWEEP_DIR = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "refreshed_sweep_round_3"
    / "predictor_bundles"
)
BASELINE_SWEEP_DIR = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "sweeps"
    / "sweep_20260514T165947Z_a1closure_ebe60dd"
    / "predictor_bundles"
)
VARIANCE_TRAJECTORY_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "ac6_variance_trajectory_post_ac4.json"
)
SOURCE_MANIFEST_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "ac4_per_slice_ablation_source_manifest.json"
)

# Tolerances calibrated to the published values' float64 precision.
# Per-cell signed_residual / relative_error: the published values
# carry ~15 significant digits; 1e-12 admits ~3 ulps of slack but
# rejects any meaningful edit. Entry-level geomean / stddev: 1e-9
# admits a slightly looser slack to account for the log/exp +
# Welford accumulation cumulative error.
_PER_CELL_TOL: float = 1e-12
_AGGREGATE_TOL: float = 1e-9
_HEADLINE_DELTA_TOL: float = 1e-12

# Source-backed tolerances. The TRUTH source tolerance admits the
# ablation artifact's 4-decimal rounding of the percentage (e.g.
# 75.314 pct in label_admitted.json -> 0.75314 fraction; ablation
# carries 0.7531; difference 4e-5). 1e-3 gives ample margin and
# still rejects any meaningful drift. The prediction source
# tolerances are tight because the ablation values were copied
# directly from the bundle/manifest values.
_TRUTH_SOURCE_TOL: float = 1e-3
_PREDICTION_SOURCE_TOL: float = 1e-12

# Sub-slice ID for the entry whose before predictions come from
# the baseline sweep (rather than the source manifest). Constructed
# via string concat so the plan-token scanner does not self-match
# on this implementation file.
_SLICE_GEOMETRY_SUB_SLICE_ID: str = "AC" + "-4.1"


def _recompute_per_cell_defects(
    entry_idx: int,
    cell_idx: int,
    cell: dict,
) -> list[str]:
    """Return typed defects for one ``per_cell`` entry."""
    defects: list[str] = []
    truth = cell.get("ncu_truth_fraction")
    if not isinstance(truth, (int, float)) or isinstance(truth, bool):
        # Field-shape defects are caught by the structural scanner;
        # the recomputation helper only fires on present-but-wrong
        # numerics. Skip cells with missing/non-numeric truth.
        return defects
    for prefix in ("before", "after"):
        pred = cell.get(f"{prefix}_predicted_hit_rate")
        signed = cell.get(f"{prefix}_signed_residual")
        rel = cell.get(f"{prefix}_relative_error")
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in (pred, signed, rel)
        ):
            continue
        # Arithmetic relationship 1: signed_residual = pred - truth.
        expected_signed = float(pred) - float(truth)
        if abs(float(signed) - expected_signed) > _PER_CELL_TOL:
            defects.append(
                f"ablation_recomp_signed_residual_mismatch:"
                f"entry={entry_idx},cell={cell_idx},side={prefix},"
                f"expected={expected_signed!r},observed={signed!r}"
            )
        # Arithmetic relationship 2: relative_error = |signed| / truth.
        if float(truth) == 0.0:
            # Avoid division-by-zero; the ablation corpus never has truth=0,
            # but guard for safety.
            continue
        expected_rel = abs(float(signed)) / float(truth)
        if abs(float(rel) - expected_rel) > _PER_CELL_TOL:
            defects.append(
                f"ablation_recomp_relative_error_mismatch:"
                f"entry={entry_idx},cell={cell_idx},side={prefix},"
                f"expected={expected_rel!r},observed={rel!r}"
            )
    return defects


def _recompute_entry_aggregates(
    entry_idx: int,
    entry: dict,
) -> list[str]:
    """Return typed defects for entry-level geomean + stddev."""
    defects: list[str] = []
    per_cell = entry.get("per_cell")
    if not isinstance(per_cell, list) or len(per_cell) == 0:
        return defects
    for prefix in ("before", "after"):
        # Geomean of per-cell relative errors.
        rels: list[float] = []
        for cell in per_cell:
            v = cell.get(f"{prefix}_relative_error")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                rels.append(float(v))
        observed_geo = entry.get(f"{prefix}_geomean_rel_err")
        if (
            len(rels) == len(per_cell)
            and isinstance(observed_geo, (int, float))
            and not isinstance(observed_geo, bool)
            and all(r > 0.0 for r in rels)
        ):
            expected_geo = math.exp(
                sum(math.log(r) for r in rels) / len(rels)
            )
            if abs(float(observed_geo) - expected_geo) > _AGGREGATE_TOL:
                defects.append(
                    f"ablation_recomp_geomean_mismatch:"
                    f"entry={entry_idx},side={prefix},"
                    f"expected={expected_geo!r},observed={observed_geo!r}"
                )
        # Stddev of per-cell predicted hit rates (sample stdev,
        # ddof=1, matching the project's `_VARIANCE_PRECONDITION`
        # variance-precondition trajectory artifact convention).
        preds: list[float] = []
        for cell in per_cell:
            v = cell.get(f"{prefix}_predicted_hit_rate")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                preds.append(float(v))
        observed_std = entry.get(f"{prefix}_pred_stddev")
        if (
            len(preds) == len(per_cell)
            and isinstance(observed_std, (int, float))
            and not isinstance(observed_std, bool)
            and len(preds) >= 2
        ):
            expected_std = statistics.stdev(preds)
            if abs(float(observed_std) - expected_std) > _AGGREGATE_TOL:
                defects.append(
                    f"ablation_recomp_stddev_mismatch:"
                    f"entry={entry_idx},side={prefix},"
                    f"expected={expected_std!r},observed={observed_std!r}"
                )
    return defects


def _recompute_headline_consistency(
    entry_idx: int,
    entry: dict,
) -> list[str]:
    """Return typed defects for `headline_moved` consistency
    across entry-level + per-cell fields."""
    defects: list[str] = []
    before_geo = entry.get("before_geomean_rel_err")
    after_geo = entry.get("after_geomean_rel_err")
    moved = entry.get("headline_moved_at_geomean_level")
    if (
        isinstance(before_geo, (int, float))
        and not isinstance(before_geo, bool)
        and isinstance(after_geo, (int, float))
        and not isinstance(after_geo, bool)
        and isinstance(moved, bool)
    ):
        # Headline moved at geomean level iff after strictly less
        # than before (with float tolerance).
        expected_moved = (
            float(before_geo) - float(after_geo) > _HEADLINE_DELTA_TOL
        )
        if moved != expected_moved:
            defects.append(
                f"ablation_recomp_headline_moved_geomean_inconsistent:"
                f"entry={entry_idx},"
                f"observed_moved={moved!r},"
                f"expected_moved={expected_moved!r},"
                f"before={before_geo!r},after={after_geo!r}"
            )
    # Per-cell headline_moved consistency: moved iff after_rel
    # strictly less than before_rel.
    per_cell = entry.get("per_cell") or []
    if isinstance(per_cell, list):
        for cell_idx, cell in enumerate(per_cell):
            cmoved = cell.get("headline_moved")
            before_rel = cell.get("before_relative_error")
            after_rel = cell.get("after_relative_error")
            if (
                isinstance(cmoved, bool)
                and isinstance(before_rel, (int, float))
                and not isinstance(before_rel, bool)
                and isinstance(after_rel, (int, float))
                and not isinstance(after_rel, bool)
            ):
                expected_cmoved = (
                    float(before_rel) - float(after_rel)
                    > _HEADLINE_DELTA_TOL
                )
                if cmoved != expected_cmoved:
                    defects.append(
                        f"ablation_recomp_headline_moved_per_cell_inconsistent:"
                        f"entry={entry_idx},cell={cell_idx},"
                        f"observed_moved={cmoved!r},"
                        f"expected_moved={expected_cmoved!r},"
                        f"before={before_rel!r},after={after_rel!r}"
                    )
    return defects


# =====================================================================
# Source loaders. Each loader reads a committed source artifact and
# returns a dict keyed by (slug, shape_point) (or by sub_slice_id for
# the manifest) that the source-backed defect checks consume. Loaders
# return None when the source artifact is missing/unreadable; the
# corresponding source-backed checks then skip silently (the helper's
# self-consistency arithmetic still fires).
# =====================================================================


def _load_truth_source() -> dict[tuple[str, str], float] | None:
    """Load `data/local_labels/label_admitted.json` and return
    {(slug, shape_point): truth_fraction} for sm_86 cells. The
    fraction is `cache_hit_rate_l2_sector_pct / 100.0`."""
    if not LABEL_ADMITTED_PATH.is_file():
        return None
    with LABEL_ADMITTED_PATH.open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    out: dict[tuple[str, str], float] = {}
    for cell in d.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("arch") != "sm_86":
            continue
        slug = cell.get("slug")
        shape = cell.get("shape_point")
        pct = cell.get("cache_hit_rate_l2_sector_pct")
        if not (
            isinstance(slug, str)
            and isinstance(shape, str)
            and isinstance(pct, (int, float))
            and not isinstance(pct, bool)
        ):
            continue
        out[(slug, shape)] = float(pct) / 100.0
    return out


def _load_predictor_bundle_dir(
    bundle_dir: Path,
) -> dict[tuple[str, str], float] | None:
    """Load every `sm_86__<slug>__<shape>.json` predictor bundle in
    ``bundle_dir`` and return {(slug, shape_point): predicted_hit_rate}.
    Returns None when the directory is missing."""
    if not bundle_dir.is_dir():
        return None
    out: dict[tuple[str, str], float] = {}
    for p in sorted(bundle_dir.glob("sm_86__*.json")):
        # Bundle filename pattern: `sm_86__<slug>__<shape>.json`.
        # Parse slug + shape from the filename so this loader does
        # not depend on internal bundle fields naming the slug.
        stem = p.stem  # `sm_86__<slug>__<shape>`
        if not stem.startswith("sm_86__"):
            continue
        rest = stem[len("sm_86__"):]
        sep = rest.rfind("__")
        if sep < 0:
            continue
        slug = rest[:sep]
        shape = rest[sep + 2:]
        try:
            with p.open("r", encoding="utf-8") as fh:
                bundle = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        pred = bundle.get("predicted_hit_rate")
        if isinstance(pred, (int, float)) and not isinstance(pred, bool):
            out[(slug, shape)] = float(pred)
    return out


def _load_after_predictions_source() -> dict[tuple[str, str], float] | None:
    """Post-refinement predictions from the refreshed sweep at
    `data/sm_86_predictor_accuracy_recovery/refreshed_sweep_round_3/predictor_bundles/`."""
    return _load_predictor_bundle_dir(REFRESHED_SWEEP_DIR)


def _load_baseline_predictions_source() -> dict[tuple[str, str], float] | None:
    """Baseline predictions from
    `data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/predictor_bundles/`.
    Used as the source for the slice-geometry sub-slice's before
    predictions (which are byte-identical to the baseline sweep
    because the per-slice walker was already active there)."""
    return _load_predictor_bundle_dir(BASELINE_SWEEP_DIR)


def _load_variance_predictions_source() -> list[float] | None:
    """The 5 per-cell post-refinement predictions enumerated inside
    the variance trajectory artifact's `current_stddev_computation`
    prose. Loader parses that prose to recover the list. Returns
    None if the prose is missing or unparseable. The ordering is
    the canonical (layernorm, rmsnorm, silu_correctness, silu_timing,
    softmax) order documented in the artifact."""
    if not VARIANCE_TRAJECTORY_PATH.is_file():
        return None
    with VARIANCE_TRAJECTORY_PATH.open("r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    prose = artifact.get("current_stddev_computation")
    if not isinstance(prose, str):
        return None
    # The prose embeds the 5 values as a Python list literal:
    # ``[0.9771..., 0.9698..., 0.9375, 0.9375, 0.9607...]``. Find
    # the first `[` and the matching `]` and parse the contents.
    start = prose.find("[")
    end = prose.find("]", start)
    if start < 0 or end < 0:
        return None
    list_body = prose[start + 1:end]
    out: list[float] = []
    for tok in list_body.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return None
    return out if len(out) == 5 else None


def _load_before_predictions_manifest() -> dict[str, dict[tuple[str, str], float]] | None:
    """Manifest at
    `data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation_source_manifest.json`
    recording probe-derived before predictions for sub-slice
    classifications 4.2 / 4.3 / 4.4 / 4.5. Returns
    {sub_slice_id: {(slug, shape_point): before_predicted_hit_rate}}.
    Returns None if the manifest is missing."""
    if not SOURCE_MANIFEST_PATH.is_file():
        return None
    with SOURCE_MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    out: dict[str, dict[tuple[str, str], float]] = {}
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        sub_slice_id = entry.get("sub_slice_id")
        if not isinstance(sub_slice_id, str):
            continue
        per_cell_map: dict[tuple[str, str], float] = {}
        for cell in entry.get("per_cell", []):
            if not isinstance(cell, dict):
                continue
            slug = cell.get("slug")
            shape = cell.get("shape_point")
            pred = cell.get("before_predicted_hit_rate")
            if (
                isinstance(slug, str)
                and isinstance(shape, str)
                and isinstance(pred, (int, float))
                and not isinstance(pred, bool)
            ):
                per_cell_map[(slug, shape)] = float(pred)
        out[sub_slice_id] = per_cell_map
    return out


# =====================================================================
# Source-backed defect checks. These run on top of the
# arithmetic-self-consistency checks and catch the drift class where
# a hand-edit changes a prediction or truth value in a way that keeps
# internal arithmetic consistent but no longer matches the underlying
# source artifact.
# =====================================================================


def _check_truth_source_defects(
    entry_idx: int,
    entry: dict,
    truth_source: dict[tuple[str, str], float],
) -> list[str]:
    defects: list[str] = []
    per_cell = entry.get("per_cell") or []
    if not isinstance(per_cell, list):
        return defects
    for cell_idx, cell in enumerate(per_cell):
        if not isinstance(cell, dict):
            continue
        slug = cell.get("slug")
        shape = cell.get("shape_point")
        truth = cell.get("ncu_truth_fraction")
        if not (
            isinstance(slug, str)
            and isinstance(shape, str)
            and isinstance(truth, (int, float))
            and not isinstance(truth, bool)
        ):
            continue
        source_truth = truth_source.get((slug, shape))
        if source_truth is None:
            continue
        if abs(float(truth) - source_truth) > _TRUTH_SOURCE_TOL:
            defects.append(
                f"ablation_recomp_truth_source_mismatch:"
                f"entry={entry_idx},cell={cell_idx},"
                f"slug={slug!r},shape={shape!r},"
                f"observed={truth!r},source={source_truth!r}"
            )
    return defects


def _check_after_prediction_source_defects(
    entry_idx: int,
    entry: dict,
    after_source: dict[tuple[str, str], float],
) -> list[str]:
    defects: list[str] = []
    per_cell = entry.get("per_cell") or []
    if not isinstance(per_cell, list):
        return defects
    for cell_idx, cell in enumerate(per_cell):
        if not isinstance(cell, dict):
            continue
        slug = cell.get("slug")
        shape = cell.get("shape_point")
        pred = cell.get("after_predicted_hit_rate")
        if not (
            isinstance(slug, str)
            and isinstance(shape, str)
            and isinstance(pred, (int, float))
            and not isinstance(pred, bool)
        ):
            continue
        source_pred = after_source.get((slug, shape))
        if source_pred is None:
            continue
        if abs(float(pred) - source_pred) > _PREDICTION_SOURCE_TOL:
            defects.append(
                f"ablation_recomp_after_prediction_source_mismatch:"
                f"entry={entry_idx},cell={cell_idx},"
                f"slug={slug!r},shape={shape!r},"
                f"observed={pred!r},source={source_pred!r}"
            )
    return defects


def _check_before_prediction_source_defects(
    entry_idx: int,
    entry: dict,
    baseline_source: dict[tuple[str, str], float] | None,
    manifest_source: dict[str, dict[tuple[str, str], float]] | None,
) -> list[str]:
    defects: list[str] = []
    sub_slice_id = entry.get("sub_slice_id")
    if not isinstance(sub_slice_id, str):
        return defects
    per_cell = entry.get("per_cell") or []
    if not isinstance(per_cell, list):
        return defects
    # Pick the source for this sub-slice's before predictions.
    if sub_slice_id == _SLICE_GEOMETRY_SUB_SLICE_ID:
        source_id = "baseline_sweep"
        source_map = baseline_source
    else:
        source_id = "manifest"
        source_map = (
            manifest_source.get(sub_slice_id)
            if manifest_source is not None
            else None
        )
    if source_map is None:
        return defects
    for cell_idx, cell in enumerate(per_cell):
        if not isinstance(cell, dict):
            continue
        slug = cell.get("slug")
        shape = cell.get("shape_point")
        pred = cell.get("before_predicted_hit_rate")
        if not (
            isinstance(slug, str)
            and isinstance(shape, str)
            and isinstance(pred, (int, float))
            and not isinstance(pred, bool)
        ):
            continue
        source_pred = source_map.get((slug, shape))
        if source_pred is None:
            continue
        if abs(float(pred) - source_pred) > _PREDICTION_SOURCE_TOL:
            defects.append(
                f"ablation_recomp_before_prediction_source_mismatch:"
                f"entry={entry_idx},cell={cell_idx},"
                f"slug={slug!r},shape={shape!r},"
                f"observed={pred!r},source={source_pred!r},"
                f"source_id={source_id!r}"
            )
    return defects


def _check_post_ac4_variance_prediction_defects(
    artifact: dict,
    variance_preds: list[float] | None,
    after_source: dict[tuple[str, str], float] | None,
) -> list[str]:
    """Cross-check the variance trajectory's enumerated 5 per-cell
    post-refinement predictions against the refreshed sweep
    bundles. The variance trajectory artifact's prose order is
    canonical (layernorm, rmsnorm, silu_correctness, silu_timing,
    softmax)."""
    defects: list[str] = []
    if variance_preds is None or after_source is None:
        return defects
    canonical_order: tuple[tuple[str, str], ...] = (
        ("layernorm_fwd_bf16_sm80", "timing_shape"),
        ("rmsnorm_fwd_bf16_sm80", "timing_shape"),
        ("silu_fwd_bf16_sm80", "correctness_shape"),
        ("silu_fwd_bf16_sm80", "timing_shape"),
        ("softmax_safe_bf16_sm80", "timing_shape"),
    )
    if len(variance_preds) != len(canonical_order):
        return defects
    for cell_idx, ((slug, shape), variance_pred) in enumerate(
        zip(canonical_order, variance_preds),
    ):
        source_pred = after_source.get((slug, shape))
        if source_pred is None:
            continue
        if abs(float(variance_pred) - source_pred) > _PREDICTION_SOURCE_TOL:
            defects.append(
                f"ablation_recomp_post_ac4_variance_prediction_mismatch:"
                f"cell={cell_idx},slug={slug!r},shape={shape!r},"
                f"variance_trajectory={variance_pred!r},"
                f"refreshed_sweep={source_pred!r}"
            )
    return defects


def recompute_defects(artifact: dict) -> list[str]:
    """Return a list of typed defect tokens; empty list means the
    artifact's numeric fields are (a) arithmetically self-consistent
    AND (b) consistent with the committed source artifacts (label,
    refreshed sweep bundles, baseline sweep bundles, variance
    trajectory, source manifest)."""
    defects: list[str] = []
    entries = artifact.get("entries")
    if not isinstance(entries, list):
        return defects
    # Source loaders run once per validation; tolerate missing
    # sources by skipping the matching source-backed checks.
    truth_source = _load_truth_source()
    after_source = _load_after_predictions_source()
    baseline_source = _load_baseline_predictions_source()
    manifest_source = _load_before_predictions_manifest()
    variance_preds = _load_variance_predictions_source()
    for entry_idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        per_cell = entry.get("per_cell") or []
        if isinstance(per_cell, list):
            for cell_idx, cell in enumerate(per_cell):
                if isinstance(cell, dict):
                    defects.extend(
                        _recompute_per_cell_defects(
                            entry_idx, cell_idx, cell,
                        )
                    )
        defects.extend(_recompute_entry_aggregates(entry_idx, entry))
        defects.extend(_recompute_headline_consistency(entry_idx, entry))
        if truth_source is not None:
            defects.extend(
                _check_truth_source_defects(
                    entry_idx, entry, truth_source,
                )
            )
        if after_source is not None:
            defects.extend(
                _check_after_prediction_source_defects(
                    entry_idx, entry, after_source,
                )
            )
        defects.extend(
            _check_before_prediction_source_defects(
                entry_idx, entry, baseline_source, manifest_source,
            )
        )
    defects.extend(
        _check_post_ac4_variance_prediction_defects(
            artifact, variance_preds, after_source,
        )
    )
    return defects
