"""Fit and emit the SEPARATE bounded-[0,1] L1/TEX sector-hit-rate calibration head.

This is a STANDALONE single-arch head INSTANCE that reuses the exact
``PerBucketCalibratorEnsemble`` pattern + deterministic fold seed (20260527) +
model-class hierarchy + composite scoring of the frozen L2 v5 head, but fits an
INDEPENDENT calibrator on the L1 sector-hit-rate label
(``cache_hit_rate_l1tex_sector_pct``) instead of the L2 label. It writes a NEW
artifact (``data/local_labels/sm_86_l1_head.json``) and NEVER reads or writes the
frozen L2 head ``data/local_labels/sm_86_v5_head.json`` — the L2 head stays
byte-identical (the frozen-L2 regression guard pins its sha256).

Why a single-feature reduction
------------------------------
The frozen L2 head's six canonical pre-calibration features
(``footprint_ratio``, ``long_rd_tail_mass``, ``intra_lane_reuse``,
``global_share``, ``gridDim_density`` beyond ``raw_prediction``) are produced by
the L2 reuse-distance walker / hrd_profiler observation builder and are
physically argued for the L2 cache-model. The per-cell L1 prediction artifact
``predicted_l1_per_admitted_cell.json`` carries only the L1 walker's
``predicted_hit_rate`` (no canonical-feature block). So the L1 head is a
single-feature isotonic calibrator: the five L2-coupled features are zero-filled
so the frozen 3-feature projection
``+1*raw_prediction -1*footprint_ratio -1*long_rd_tail_mass`` reduces exactly to
``raw_prediction``. The shared ``canonical_features`` monotonic-sign table is
left UNTOUCHED (the L2 head depends on it byte-for-byte); the L1 head simply
exercises the ``raw_prediction`` axis (monotonic +1) of that table.

Coverage honesty
----------------
On the current sm_86 cohort only the cells whose L1 walker emitted a SCALAR
prediction enter the fit; cells whose L1 prediction is ``typed_unavailable``
(e.g. ``predictor_l1_sass_provenance_missing``) are recorded in
``cohort_scope.skipped`` with the producer's root-cause typed reason and never
enter the fit. The emitted artifact's ``cohort_scope.admitted_count`` is the
HONEST L1 training-set size; the comparison runner's coverage gate (smoke 5 /
publication 10) is applied and reported by the headline driver, not papered over
here.

CLI:
  python tools/local_labels/fit_l1_calibration_head.py \
    [--label-admitted data/local_labels/label_admitted.json] \
    [--l1-predictions data/sm_86_predictor_accuracy_recovery_v2/predicted_l1_per_admitted_cell.json] \
    [--family-buckets data/local_labels/sm120_cohort_family_buckets.toml] \
    [--arch sm_86] \
    [--output data/local_labels/sm_86_l1_head.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw import canonical_features as cf  # noqa: E402
from workloads.simbench_bw import monotonic_calibration_head as mch  # noqa: E402
from tools.local_labels.corpus_cell_identity import (  # noqa: E402
    cell_variant_id,
    is_base_predictor_fit_cell,
)

# Reuse the frozen-L2 fitter's PURE helpers. Importing them (rather than
# re-implementing) keeps the L1 fit lockstep with the L2 fold/scoring/selection
# protocol while leaving the L2 fitter source and the L2 head output untouched.
from tools.local_labels.fit_monotonic_calibration_head_v5 import (  # noqa: E402
    _git_head_sha,
    _load_family_buckets,
    _load_label_cells,
    _load_split_by_key,
    _model_class_factory_for,
    _resolve_family_bucket,
    _score_candidate,
)

DEFAULT_LABEL_ADMITTED = "data/local_labels/label_admitted.json"
DEFAULT_L1_PREDICTIONS = (
    "data/sm_86_predictor_accuracy_recovery_v2/predicted_l1_per_admitted_cell.json"
)
DEFAULT_FAMILY_BUCKETS_TOML = "data/local_labels/sm120_cohort_family_buckets.toml"
DEFAULT_OUTPUT = "data/local_labels/sm_86_l1_head.json"
DEFAULT_ARCH = "sm_86"

# The L1 label field (percent), parallel to the L2 head's
# cache_hit_rate_l2_sector_pct.
L1_LABEL_FIELD = "cache_hit_rate_l1tex_sector_pct"
# l1tex_label_status values that DO carry a usable mean L1 label.
_L1_LABEL_STATUS_USABLE = frozenset({"mean_present", "all_zero"})


def _load_l1_predictions(path: Path) -> dict[tuple[str, str, str], dict]:
    """Return per-cell L1 prediction rows keyed by (arch, slug, shape_point)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], dict] = {}
    for cell in raw.get("cells", []):
        key = (
            str(cell.get("arch", "")),
            str(cell.get("slug", "")),
            str(cell.get("shape_point", "")),
        )
        out[key] = cell
    return out


def _l1_features(raw_prediction: float) -> dict[str, float]:
    """Build the single-feature L1 feature vector.

    Zero-fills every canonical feature, then sets ``raw_prediction`` so the
    frozen 3-feature projection reduces to ``raw_prediction`` (the five
    L2-coupled features contribute 0 to the projection). The shared
    ``canonical_features`` sign table is read, never mutated.
    """
    features = {name: 0.0 for name in cf.CANONICAL_FEATURE_NAMES}
    features["raw_prediction"] = float(raw_prediction)
    return features


def _build_l1_cell_records(
    *,
    label_cells: list[dict],
    l1_predictions: dict[tuple[str, str, str], dict],
    family_to_bucket: dict[str, str],
    split_by_key: dict[tuple, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Join label cells against the per-cell L1 walker predictions.

    Returns ``(admitted_cells, skipped_cells)`` mirroring the v5 fitter's record
    shape so the same ``PerBucketCalibratorEnsemble`` machinery consumes them.
    A cell is admitted only when its L1 prediction is a non-null SCALAR AND it
    carries a usable L1 label; every other case is a typed skip.
    """
    admitted: list[dict] = []
    skipped: list[dict] = []
    for cell in label_cells:
        arch = str(cell.get("arch", ""))
        slug = str(cell.get("slug", ""))
        shape_point = str(cell.get("shape_point", ""))
        variant_id = cell_variant_id(cell)
        # Mixed-arity identity: sibling register-cap variants share their
        # (slug, shape_point) name; the canonical variant_id ("" for a base
        # cell) keeps them distinct. The per-cell L1 prediction artifact
        # covers BASE cells (its producer keys (slug, shape_point) without a
        # variant axis), so a register-cap variant falls to the typed
        # l1_prediction_absent skip below — an honest typed exclusion, not a
        # silent gap.
        identity = (arch, slug, shape_point, variant_id)
        pred = l1_predictions.get((arch, slug, shape_point)) if not variant_id else None
        if pred is None:
            skipped.append({
                "arch": arch, "slug": slug, "shape_point": shape_point,
                "typed_reason": "l1_prediction_absent",
                "identity": identity,
            })
            continue
        result_kind = pred.get("result_kind")
        if result_kind != "scalar":
            # Propagate the producer's root-cause typed reason (e.g.
            # predictor_l1_sass_provenance_missing) so the L1 head's skip rows
            # carry the same machine-readable falsification status as the
            # per-cell prediction artifact.
            typed_reason = pred.get("typed_reason") or "predictor_l1_unavailable"
            skipped.append({
                "arch": arch, "slug": slug, "shape_point": shape_point,
                "typed_reason": str(typed_reason),
                "identity": identity,
            })
            continue
        raw_prediction = pred.get("predicted_hit_rate")
        if raw_prediction is None:
            skipped.append({
                "arch": arch, "slug": slug, "shape_point": shape_point,
                "typed_reason": "l1_predicted_hit_rate_absent",
                "identity": identity,
            })
            continue
        status = cell.get("l1tex_label_status")
        if status is not None and status not in _L1_LABEL_STATUS_USABLE:
            skipped.append({
                "arch": arch, "slug": slug, "shape_point": shape_point,
                "typed_reason": f"l1tex_label_status_{status}",
                "identity": identity,
            })
            continue
        label_pct = cell.get(L1_LABEL_FIELD)
        if label_pct is None:
            skipped.append({
                "arch": arch, "slug": slug, "shape_point": shape_point,
                "typed_reason": f"label_{L1_LABEL_FIELD}_absent",
                "identity": identity,
            })
            continue
        family_bucket = _resolve_family_bucket(slug, family_to_bucket)
        record = {
            "identity": identity,
            "arch": arch,
            "family_bucket": family_bucket,
            "slug": slug,
            "shape_point": shape_point,
            "features": _l1_features(float(raw_prediction)),
            "label": float(label_pct) / 100.0,
        }
        if variant_id:
            record["variant_id"] = variant_id
        if split_by_key is not None:
            split = split_by_key.get((slug, shape_point, variant_id))
            if split is None and arch == "sm_86":
                raise ValueError(
                    "fit_l1_calibration_head: admitted sm_86 cell "
                    f"({slug!r}, {shape_point!r}, variant={variant_id!r}) "
                    "has no cohort-split assignment; a split-aware fit "
                    "cannot silently include an unassigned cell"
                )
            if split is not None:
                record["split_assignment"] = split
        admitted.append(record)
    return admitted, skipped


def _empty_or_indeterminate_artifact(
    *,
    fit_status: str,
    status: str,
    dep_probe: dict,
    dependency_versions_flat: dict,
    skipped: list[dict],
    extra_skipped: list[dict] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skipped_rows = [
        {k: v for k, v in s.items() if k != "identity"} for s in skipped
    ]
    if extra_skipped:
        skipped_rows += extra_skipped
    art: dict[str, Any] = {
        "schema_version": 1,
        "calibration_head_version": 5,
        "head_kind": "l1_sector_hit_rate",
        "cache_layer": "L1TEX",
        "metric_target": "l1_hit_rate",
        "label_field": L1_LABEL_FIELD,
        "source_commit": _git_head_sha(),
        "dependency_availability": dep_probe,
        "dependency_versions": dependency_versions_flat,
        "composite_score_formula": dict(mch.COMPOSITE_SCORE_FORMULA),
        "feature_schema": {
            "canonical_feature_names": list(cf.CANONICAL_FEATURE_NAMES),
            "monotonic_constraints_by_feature": dict(
                cf.MONOTONIC_CONSTRAINT_SIGNS,
            ),
            "l1_active_feature": "raw_prediction",
            "l1_feature_reduction": (
                "five L2-coupled canonical features are zero-filled so the "
                "frozen projection reduces to raw_prediction; the shared "
                "canonical_features sign table is read, not mutated"
            ),
        },
        "fold_assignments": {},
        "model_class_per_bucket": {},
        "monotonic_constraints_by_feature": {},
        "selected_hyperparameters": {},
        "training_cell_ids": {},
        "per_bucket_provenance": {
            "by_bucket": {},
            "pooled_fallback_present": False,
            "pooled_used_features": [],
        },
        "cohort_scope": {
            "admitted": [],
            "skipped": skipped_rows,
            "admitted_count": 0,
            "skipped_count": len(skipped_rows),
        },
        "fit_status": fit_status,
        "status": status,
    }
    if extra:
        art.update(extra)
    return art


def fit_and_emit_l1(
    *,
    label_admitted_path: Path,
    l1_predictions_path: Path,
    family_buckets_path: Path,
    output_path: Path,
    arch: str = DEFAULT_ARCH,
    cohort_split_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full L1 head fit + emission. Returns the artifact dict."""
    label_cells = _load_label_cells(label_admitted_path)
    if arch:
        label_cells = [c for c in label_cells if str(c.get("arch", "")) == arch]
    # Predictor-fit cohort (same scope as the v5 fitter + the L1 prediction
    # producer). Since the controlled refit the pending sets are empty and
    # the predicate admits register-cap variants; a variant still lands as a
    # typed l1_prediction_absent skip (the per-cell L1 prediction artifact
    # covers base cells). A future corpus addition re-populates the pending
    # sets and is excluded here until the next refit.
    if arch == "sm_86":
        label_cells = [c for c in label_cells if is_base_predictor_fit_cell(c)]
    l1_predictions = _load_l1_predictions(l1_predictions_path)
    family_to_bucket = _load_family_buckets(family_buckets_path)
    split_by_key = (
        _load_split_by_key(cohort_split_path)
        if cohort_split_path is not None else None
    )
    admitted, skipped = _build_l1_cell_records(
        label_cells=label_cells,
        l1_predictions=l1_predictions,
        family_to_bucket=family_to_bucket,
        split_by_key=split_by_key,
    )
    # Split-aware fit scope (mirrors the v5 fitter): CV + the final fit run
    # on the calibration+validation subset; the sealed holdout cells are
    # recorded in cohort_scope.admitted for the gate emitters' exactly-once
    # evaluation but never enter the fit.
    if split_by_key is not None:
        fit_cells = [
            c for c in admitted if c.get("split_assignment") != "holdout"
        ]
        if admitted and not fit_cells:
            raise ValueError(
                "fit_l1_calibration_head: every admitted cell is holdout-"
                "assigned; a split-aware fit needs a non-empty "
                "calibration+validation subset"
            )
    else:
        fit_cells = admitted
    dep_probe = mch.probe_dependency_availability()
    dependency_versions_flat = {
        e["package"]: e.get("version") for e in dep_probe.values()
    }

    def _scope(art: dict) -> dict:
        art["arch"] = arch
        art["head_scope"] = "standalone_single_arch_l1"
        return art

    if not admitted:
        artifact = _empty_or_indeterminate_artifact(
            fit_status="empty_cohort_no_admitted_cells",
            status="indeterminate_insufficient_data",
            dep_probe=dep_probe,
            dependency_versions_flat=dependency_versions_flat,
            skipped=skipped,
        )
        output_path.write_text(
            json.dumps(_scope(artifact), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact

    try:
        mch.select_active_model_class(dependency_availability=dep_probe)
    except (RuntimeError, NotImplementedError) as exc:
        extra_skipped = [
            {
                k: v
                for k, v in {**c, "typed_reason": "deferred_dependencies_absent"}.items()
                if k not in ("identity", "features")
            }
            for c in admitted
        ]
        artifact = _empty_or_indeterminate_artifact(
            fit_status="indeterminate_dependencies_absent",
            status="indeterminate_insufficient_data",
            dep_probe=dep_probe,
            dependency_versions_flat=dependency_versions_flat,
            skipped=skipped,
            extra_skipped=extra_skipped,
            extra={"dependency_absence_reason": str(exc)},
        )
        output_path.write_text(
            json.dumps(_scope(artifact), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact

    identities = [c["identity"] for c in fit_cells]
    strata = [(c["arch"], c["family_bucket"]) for c in fit_cells]
    fold_assignments, fold_decision = mch.deterministic_fold_assignments(
        identities, strata,
    )
    candidates: list[dict[str, Any]] = []
    skipped_candidates: list[dict[str, Any]] = []
    for entry in mch.MODEL_CLASS_HIERARCHY:
        cname = entry["name"]
        if not dep_probe.get(cname, {}).get("available"):
            skipped_candidates.append({
                "name": cname,
                "skip_reason": "dependency_unavailable",
                "dependency_package": entry["dependency_package"],
            })
            continue
        try:
            candidate_factory = _model_class_factory_for(cname)
        except NotImplementedError as exc:
            skipped_candidates.append({
                "name": cname, "skip_reason": "factory_unavailable",
                "detail": str(exc),
            })
            continue
        try:
            cv_scores = _score_candidate(
                cells=fit_cells,
                model_class_factory=candidate_factory,
                fold_assignments=fold_assignments,
            )
        except Exception as exc:  # noqa: BLE001 - structured skip record
            skipped_candidates.append({
                "name": cname, "skip_reason": "scoring_failure",
                "detail": str(exc),
            })
            continue
        candidates.append({
            "name": cname,
            "parameter_count": entry["parameter_count"],
            "primary": cv_scores["primary"],
            "secondary": cv_scores["secondary"],
            "tertiary": cv_scores["tertiary"],
            "n_cv_predictions": cv_scores["n_cv_predictions"],
        })
    if not candidates:
        raise RuntimeError(
            "no model class from the registered hierarchy completed CV scoring "
            f"for the L1 head; skip_records={skipped_candidates!r}"
        )
    selected = mch.composite_score_select_simpler(candidates=candidates)
    factory = _model_class_factory_for(selected["name"])
    ensemble = mch.PerBucketCalibratorEnsemble(model_class_factory=factory)
    ensemble.fit(fit_cells)
    provenance = ensemble.provenance_summary()
    head_proto = factory()
    used_features = getattr(head_proto, "used_features", cf.CANONICAL_FEATURE_NAMES)
    constraints_for_used = {
        n: cf.MONOTONIC_CONSTRAINT_SIGNS[n] for n in used_features
    }
    model_class_per_bucket: dict[str, str] = {}
    monotonic_by_bucket: dict[str, dict[str, int]] = {}
    for bucket_key, entry in provenance["by_bucket"].items():
        model_class_per_bucket[bucket_key] = (
            selected["name"] if entry["kind"] == "per_bucket"
            else f"{selected['name']}__pooled_fallback_with_one_hot"
        )
        monotonic_by_bucket[bucket_key] = dict(constraints_for_used)
        if entry["kind"] == "pooled_fallback":
            monotonic_by_bucket[bucket_key]["family_bucket_one_hot"] = 0
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "calibration_head_version": 5,
        "head_kind": "l1_sector_hit_rate",
        "cache_layer": "L1TEX",
        "metric_target": "l1_hit_rate",
        "label_field": L1_LABEL_FIELD,
        "source_commit": _git_head_sha(),
        "dependency_availability": dep_probe,
        "dependency_versions": dependency_versions_flat,
        "composite_score_formula": dict(mch.COMPOSITE_SCORE_FORMULA),
        "feature_schema": {
            "canonical_feature_names": list(cf.CANONICAL_FEATURE_NAMES),
            "monotonic_constraints_by_feature": dict(
                cf.MONOTONIC_CONSTRAINT_SIGNS,
            ),
            "l1_active_feature": "raw_prediction",
            "l1_feature_reduction": (
                "five L2-coupled canonical features are zero-filled so the "
                "frozen projection reduces to raw_prediction; the shared "
                "canonical_features sign table is read, not mutated"
            ),
        },
        "fold_assignments": {
            repr(list(ident)): fold for ident, fold in fold_assignments.items()
        },
        "fold_construction": fold_decision,
        "model_class_per_bucket": model_class_per_bucket,
        "monotonic_constraints_by_feature": monotonic_by_bucket,
        "candidate_scores": candidates,
        "skipped_candidates": skipped_candidates,
        "selected_hyperparameters": {
            selected["name"]: {
                "increasing": True,
                "out_of_bounds": "clip",
                "projection_weights": list(mch.FROZEN_PROJECTION_WEIGHTS),
                "projection_features": list(mch.FROZEN_PROJECTION_FEATURES),
            },
        },
        "selected_class": {
            "name": selected["name"],
            "parameter_count": selected["parameter_count"],
            "primary": selected["primary"],
            "secondary": selected["secondary"],
            "tertiary": selected["tertiary"],
            "selection_reason": selected["selection_reason"],
        },
        "training_cell_ids": {
            bucket_key: entry["training_cell_ids"]
            for bucket_key, entry in provenance["by_bucket"].items()
        },
        "per_bucket_provenance": provenance,
        "status": "ok",
        "cohort_scope": {
            "admitted": [
                {
                    "arch": c["arch"],
                    "slug": c["slug"],
                    "shape_point": c["shape_point"],
                    "family_bucket": c["family_bucket"],
                    "label": c["label"],
                    "features": dict(c["features"]),
                    **(
                        {"variant_id": c["variant_id"]}
                        if c.get("variant_id") else {}
                    ),
                    **(
                        {"split_assignment": c["split_assignment"]}
                        if c.get("split_assignment") else {}
                    ),
                }
                for c in admitted
            ],
            "skipped": [
                {k: v for k, v in s.items() if k != "identity"} for s in skipped
            ],
            "admitted_count": len(admitted),
            "skipped_count": len(skipped),
        },
        "fit_scope": {
            "split_aware": cohort_split_path is not None,
            "cohort_split_path": (
                str(cohort_split_path) if cohort_split_path is not None
                else None
            ),
            "fit_splits": (
                ["calibration", "validation"]
                if cohort_split_path is not None else None
            ),
            "n_fit": len(fit_cells),
            "n_holdout_excluded_from_fit": len(admitted) - len(fit_cells),
        },
        "fit_status": "ok",
    }
    output_path.write_text(
        json.dumps(_scope(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-admitted", default=DEFAULT_LABEL_ADMITTED)
    parser.add_argument("--l1-predictions", default=DEFAULT_L1_PREDICTIONS)
    parser.add_argument("--family-buckets", default=DEFAULT_FAMILY_BUCKETS_TOML)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument(
        "--cohort-split", default=None,
        help=(
            "Path to cohort_split.json. When set, the L1 fit becomes "
            "split-aware (fit on calibration+validation; sealed holdout "
            "recorded for the gates' exactly-once evaluation)."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    artifact = fit_and_emit_l1(
        label_admitted_path=Path(args.label_admitted),
        l1_predictions_path=Path(args.l1_predictions),
        family_buckets_path=Path(args.family_buckets),
        output_path=Path(args.output),
        arch=args.arch,
        cohort_split_path=(
            Path(args.cohort_split) if args.cohort_split else None
        ),
    )
    print(
        f"[fit_l1] status={artifact['fit_status']} "
        f"admitted={artifact['cohort_scope']['admitted_count']} "
        f"skipped={artifact['cohort_scope']['skipped_count']} -> {args.output}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
