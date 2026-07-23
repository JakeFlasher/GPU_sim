"""Build the interim diagnostic report over the
HEADLINE_COHORT_13. Emits a markdown report at
``data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report.md``
plus a strict-schema-validated companion sidecar at
``data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report_sidecar.json``.

The report consumes:

  - The variance-trajectory artifact (per-admitted-cell
    predicted_l2_hit_rate / ncu_truth_l2_fraction /
    signed_residual / predictor_design_coverage_bucket).
  - The candidate-feature catalog (top-level
    recommendations; the report quotes a small set of
    them).
  - The cohort accounting artifact (13-cell coverage with
    each cell's admission_status +
    predictor_design_coverage_bucket).
  - The admitted-label index (per-cell L1 truth fraction
    via cache_hit_rate_l1tex_sector_pct).

The 9 admitted cells in the cohort are mostly L2-stressors
without SASS provenance. The L1 predictor's scalar bundle
is gated on SASS provenance + cta-to-sm scheduling
information; in practice every admitted cell's L1 path
fail-closes with typed_unavailable. The report tags these
cells with l1_status=typed_unavailable_persistent and
omits them from the friendly_l1 / hostile_l1 geomeans.

The plan_2_candidate enum is set per the rule book:

  - typed_unavailable_persistent when BOTH L1 and L2
    paths emit typed_unavailable.
  - already_meets_target_no_action_needed when L2
    rel_err < 0.10 AND (L1 rel_err < 0.10 OR L1
    typed_unavailable).
  - predictor_friendly_plan_2_accuracy_target for
    predictor_friendly cells that are in-domain for the
    Smith-Eklund baseline; the downstream
    calibration-fitting plan is expected to close the
    gap.
  - predictor_hostile_physics_challenge_plan_2 for
    predictor_hostile cells whose residual is
    out-of-domain for the current Smith-Eklund baseline;
    the downstream plan's physics work is required.

The plan_1_closure_status enum is set per the schema's
conditional rule:

  - closed_all_targets_hit only when all three hard
    targets pass.
  - aborted_unrecoverable_per_user_decision when any
    hard target is missed AND the operator override is
    accepted (the current state under the operator
    partial-closure decision).
  - pending_targets_not_yet_hit_plan_1_5_required when
    any hard target is missed AND the operator override
    is not accepted (would route to a corpus-or-
    baseline-strengthening round).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]

# Local sibling import. The label_loader module lives in the
# same directory; its load_labels_for_mode helper is the
# canonical entrypoint for operational blinding (calibration/
# validation runners cannot accept the sealed holdout labels
# path; the helper fails closed before any disk read).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import label_loader  # noqa: E402

# Shared mixed-arity identity helper: cell_variant_id yields "" for a base cell
# and the canonical register_cap_<n> for a build variant. The predictor /
# headline cohort this report covers is the BASE admitted set; register-cap
# build variants are a separate factorial corpus folded in by a later predictor
# refit, so they are excluded from the 1:1 prediction-to-admitted coverage here.
_LOCAL_LABELS_DIR = Path(__file__).resolve().parents[1] / "local_labels"
if str(_LOCAL_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS_DIR))
from corpus_cell_identity import (  # noqa: E402
    is_base_predictor_fit_cell,
)


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SCHEMA_PATH = (
    _HERE / "interim_diagnostic_report_sidecar.schema.json"
)
_DEFAULT_AC5_ARTIFACT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "ac5_variance_trajectory_post_corpus_expansion.json"
)
_DEFAULT_AC6_ARTIFACT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "ac6_missing_feature_candidates.json"
)
_DEFAULT_COHORT_ACCOUNTING = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_accounting_post_corpus_expansion.json"
)
_DEFAULT_LABEL_ADMITTED = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
_DEFAULT_PREDICTED_L1_ARTIFACT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "predicted_l1_per_admitted_cell.json"
)
_DEFAULT_OUT_MD = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "interim_diagnostic_report.md"
)
_DEFAULT_OUT_SIDECAR = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "interim_diagnostic_report_sidecar.json"
)

_DEFAULT_PREDICTED_L1_SCHEMA = (
    _HERE / "predicted_l1_per_admitted_cell.schema.json"
)
_DOCUMENTED_FLOOR_STDDEV = 0.10
_HARD_1_FRIENDLY_L2_THRESHOLD = 0.15
_HARD_2_FRIENDLY_L1_THRESHOLD = 0.20
_REL_ERR_EPSILON = 1e-9
_ALREADY_MEETS_TARGET_REL_ERR = 0.10
_AC6_QUOTED_RECOMMENDATIONS = 3

# The 5 identity fields the predicted-L1 artifact MUST mirror
# from each admitted-cell row. Used by the per-cell identity
# check below. A mismatch on any field is a load-bearing
# failure that fails-closed: the report cannot consume a
# prediction made against a different binary / source_commit
# / kernel signature than the admitted truth label.
_PREDICTED_L1_IDENTITY_FIELDS: tuple[str, ...] = (
    "arch",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
    "ncu_metric_version",
)


def _rel_err(predicted: float, truth: float) -> float:
    """Compute |predicted - truth| / max(truth, epsilon)."""
    denom = max(abs(truth), _REL_ERR_EPSILON)
    return abs(predicted - truth) / denom


def _geomean_or_none(values: list[float]) -> float | None:
    """statistics.geometric_mean requires all-positive
    inputs. When the list is empty OR carries any
    non-positive value, return None (the bucket's geomean
    is undefined)."""
    if not values:
        return None
    if any(v <= 0.0 for v in values):
        return None
    return statistics.geometric_mean(values)


def _classify_plan_2_candidate(
    *,
    l1_status: str,
    l2_status: str,
    rel_err_l1: float | None,
    rel_err_l2: float | None,
    bucket: str,
) -> str:
    """Classifier per the rule book. The 4 enum values:
    already_meets_target_no_action_needed,
    predictor_friendly_plan_2_accuracy_target,
    predictor_hostile_physics_challenge_plan_2,
    typed_unavailable_persistent."""
    if (
        l1_status == "typed_unavailable_persistent"
        and l2_status == "typed_unavailable_persistent"
    ):
        return "typed_unavailable_persistent"
    l2_under = (
        rel_err_l2 is not None
        and rel_err_l2 < _ALREADY_MEETS_TARGET_REL_ERR
    )
    l1_ok = (
        l1_status == "typed_unavailable_persistent"
        or (rel_err_l1 is not None
            and rel_err_l1 < _ALREADY_MEETS_TARGET_REL_ERR)
    )
    if l2_under and l1_ok:
        return "already_meets_target_no_action_needed"
    if bucket == "predictor_hostile":
        return "predictor_hostile_physics_challenge_plan_2"
    return "predictor_friendly_plan_2_accuracy_target"


def _classify_plan_1_closure_status(
    *,
    hard_1: bool,
    hard_2: bool,
    hard_3: bool,
    override_accepted: bool,
) -> str:
    if hard_1 and hard_2 and hard_3:
        return "closed_all_targets_hit"
    if override_accepted:
        return "aborted_unrecoverable_per_user_decision"
    return "pending_targets_not_yet_hit_plan_1_5_required"


def _bucket_block(
    *,
    rel_errs_scalar: list[float],
    n_typed_unavailable: int,
) -> dict:
    """Return a per-bucket headline block. n_bounded is 0
    in the current model (the L1/L2 predictors emit either
    scalar or typed_unavailable; no bounded bundles in
    this round's surface)."""
    return {
        "n_cells": len(rel_errs_scalar) + n_typed_unavailable,
        "n_scalar": len(rel_errs_scalar),
        "n_bounded": 0,
        "n_typed_unavailable": n_typed_unavailable,
        "geomean_rel_err": _geomean_or_none(rel_errs_scalar),
    }


def _validate_predicted_l1_artifact(
    artifact: dict, *, schema_path: Path,
) -> None:
    """Validate the predicted-L1 artifact against its
    schema. Raises ``jsonschema.ValidationError`` on
    failure."""
    with schema_path.open(encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(artifact, schema)


def _index_predicted_l1_cells(
    artifact: dict,
) -> dict[tuple, dict]:
    """Build the mixed-arity ``(slug, shape_point[, variant_id]) -> row``
    map, raising ``ValueError`` on duplicate keys. A register-cap build
    variant (folded into the corpus coverage by the controlled refit)
    widens its key with the canonical ``variant_id`` so sibling caps never
    collapse; base cells keep the historical 2-tuple arity."""
    out: dict[tuple, dict] = {}
    for r in artifact.get("cells", []):
        key = (str(r["slug"]), str(r["shape_point"]))
        if r.get("variant_id"):
            key = key + (str(r["variant_id"]),)
        if key in out:
            raise ValueError(
                f"predicted_l1_per_admitted_cell artifact "
                f"contains duplicate cell "
                f"key {key!r}; the report's per-cell join "
                f"requires unique keys"
            )
        out[key] = r
    return out


def _check_predicted_l1_identity(
    *,
    predicted_l1_row: dict,
    admitted_row: dict,
    slug: str,
    shape_point: str,
) -> None:
    """Verify the predicted-L1 row's identity tuple matches
    the admitted-label row. Raises ``ValueError`` on
    mismatch; the report cannot consume a prediction made
    against a different binary / source_commit / kernel
    signature than the admitted truth label."""
    mismatches: list[str] = []
    for field in _PREDICTED_L1_IDENTITY_FIELDS:
        pred = predicted_l1_row.get(field)
        adm = admitted_row.get(field)
        if pred is None or adm is None:
            continue  # the schema enforces presence; coverage
            # check fires separately below.
        if str(pred) != str(adm):
            mismatches.append(
                f"{field}: predicted={pred!r} != admitted={adm!r}"
            )
    if mismatches:
        raise ValueError(
            f"predicted_l1 artifact row {slug}/{shape_point} "
            f"identity tuple disagrees with label_admitted "
            f"row: {'; '.join(mismatches)}; the report's "
            f"per-cell rel_err computation cannot proceed "
            f"with provenance-divergent inputs"
        )


def _check_predicted_l1_coverage(
    *,
    predicted_l1_by_key: dict[tuple[str, str], dict],
    admitted_by_key: dict[tuple[str, str], dict],
) -> None:
    """Verify every admitted sm_86 cell has exactly one
    matching prediction row. Raises ``ValueError`` on any
    coverage gap. The check is one-way: extra predicted-L1
    rows (predictions for non-admitted cells) are tolerated
    -- the artifact may include rows for cells that fall out
    of admission between generations."""
    missing = [
        key for key in admitted_by_key
        if key not in predicted_l1_by_key
    ]
    if missing:
        raise ValueError(
            f"predicted_l1 artifact does not cover every "
            f"admitted sm_86 cell; missing rows for: "
            f"{sorted(missing)}. The report requires a 1:1 "
            f"prediction-to-admitted mapping for the HARD-2 "
            f"metric"
        )


def build_report(
    *,
    ac5_artifact_path: Path = _DEFAULT_AC5_ARTIFACT,
    ac6_artifact_path: Path = _DEFAULT_AC6_ARTIFACT,
    cohort_accounting_path: Path = _DEFAULT_COHORT_ACCOUNTING,
    label_admitted_path: Path = _DEFAULT_LABEL_ADMITTED,
    predicted_l1_artifact_path: Path | None = _DEFAULT_PREDICTED_L1_ARTIFACT,
    predicted_l1_schema_path: Path = _DEFAULT_PREDICTED_L1_SCHEMA,
) -> dict:
    """Build the sidecar dict over the 13-cell cohort.
    Pure function: takes the inputs explicitly so tests
    can inject synthetic surfaces.

    ``predicted_l1_artifact_path`` points to the artifact
    emitted by ``compute_predicted_l1_per_admitted_cell.py``.
    When ``None`` OR the file does not exist, the report
    falls back to the legacy ``predicted_l1=None`` behavior
    so the historical contract holds for callers that have
    not yet generated the artifact; this is the legacy
    fail-soft path.

    When the artifact IS present, the report enforces 4
    load-bearing guards before consuming any row:
      1. The artifact validates against
         ``predicted_l1_per_admitted_cell.schema.json``.
      2. No duplicate ``(slug, shape_point)`` keys.
      3. Every admitted sm_86 cell has exactly one matching
         prediction row (coverage check).
      4. Each prediction row's identity tuple (arch,
         binary_sha256, source_commit,
         kernel_signature_sha256, ncu_metric_version)
         matches the admitted row's identity tuple.
    Any guard violation raises ``ValueError`` /
    ``jsonschema.ValidationError`` -- the report does NOT
    silently fall back to legacy when the artifact is
    present-but-broken.
    """
    with ac5_artifact_path.open(encoding="utf-8") as fh:
        ac5 = json.load(fh)
    with ac6_artifact_path.open(encoding="utf-8") as fh:
        ac6 = json.load(fh)
    with cohort_accounting_path.open(encoding="utf-8") as fh:
        cohort = json.load(fh)
    admitted = label_loader.load_labels_for_mode(
        label_admitted_path, "calibration",
    )
    predicted_l1_by_key: dict[tuple[str, str], dict] = {}
    # Track whether an artifact was loaded SEPARATELY from
    # whether the indexed map ended up non-empty. A
    # present-but-empty artifact (cohort_size=0, cells=[])
    # would otherwise bypass the coverage guard because
    # ``predicted_l1_by_key`` evaluates to False on the empty
    # map. The schema's ``cells.minItems=1`` is the first
    # line of defense; this flag is the second, in case a
    # caller uses a tolerant validator.
    artifact_loaded = False
    if (
        predicted_l1_artifact_path is not None
        and predicted_l1_artifact_path.is_file()
    ):
        with predicted_l1_artifact_path.open(encoding="utf-8") as fh:
            predicted_l1_artifact = json.load(fh)
        _validate_predicted_l1_artifact(
            predicted_l1_artifact,
            schema_path=predicted_l1_schema_path,
        )
        predicted_l1_by_key = _index_predicted_l1_cells(
            predicted_l1_artifact,
        )
        artifact_loaded = True

    ac5_by_key: dict[tuple[str, str], dict] = {
        (r["slug"], r["shape_point"]): r
        for r in ac5["cells"]
    }
    # Mixed-arity admitted index: a register-cap build variant (folded by
    # the controlled refit) widens its key with the canonical variant_id so
    # sibling caps never collapse; base cells keep the 2-tuple arity.
    from corpus_cell_identity import cell_variant_id as _cvid
    admitted_by_key: dict[tuple, dict] = {}
    for r in admitted.get("cells", []):
        if not (isinstance(r, dict) and is_base_predictor_fit_cell(r)):
            continue
        key = (str(r["slug"]), str(r["shape_point"]))
        vid = _cvid(r)
        if vid:
            key = key + (vid,)
        admitted_by_key[key] = r

    # When the predicted-L1 artifact was loaded, enforce
    # the 1:1 coverage guard before the per-cell loop runs.
    # An admitted cell with no matching prediction is a
    # contract violation -- the loop would silently mark it
    # typed_unavailable, masking the missing-row defect.
    # The check runs on artifact_loaded, not on
    # ``predicted_l1_by_key`` truthiness, so a present-but-
    # empty artifact (which has an empty index) does NOT
    # silently bypass the coverage check.
    if artifact_loaded:
        _check_predicted_l1_coverage(
            predicted_l1_by_key=predicted_l1_by_key,
            admitted_by_key=admitted_by_key,
        )

    cohort_cells = sorted(
        cohort.get("cells", []),
        key=lambda c: (str(c.get("slug")), str(c.get("shape_point"))),
    )
    if len(cohort_cells) != 13:
        raise ValueError(
            f"build_interim_diagnostic_report: "
            f"cohort_accounting_post_corpus_expansion.json "
            f"must carry exactly 13 cells; got {len(cohort_cells)}"
        )

    rows: list[dict] = []
    friendly_l2_scalar: list[float] = []
    friendly_l2_typed_unavailable = 0
    friendly_l1_scalar: list[float] = []
    friendly_l1_typed_unavailable = 0
    hostile_l2_scalar: list[float] = []
    hostile_l2_typed_unavailable = 0
    hostile_l1_scalar: list[float] = []
    hostile_l1_typed_unavailable = 0

    for cohort_cell in cohort_cells:
        slug = str(cohort_cell["slug"])
        sp = str(cohort_cell["shape_point"])
        key = (slug, sp)
        admission_status = str(cohort_cell["admission_status"])
        bucket = str(
            cohort_cell["predictor_design_coverage_bucket"]
        )
        ac5_row = ac5_by_key.get(key)
        admitted_row = admitted_by_key.get(key)

        # L2 status + values: admitted cells with a
        # variance-trajectory row produce scalar L2;
        # excluded cells emit typed_unavailable_persistent.
        if (
            admission_status == "admitted"
            and ac5_row is not None
        ):
            ncu_truth_l2 = float(ac5_row["ncu_truth_l2_fraction"])
            predicted_l2 = float(ac5_row["predicted_l2_hit_rate"])
            rel_err_l2 = _rel_err(predicted_l2, ncu_truth_l2)
            signed_residual_l2 = float(ac5_row["signed_residual"])
            l2_status = "scalar_present"
        else:
            ncu_truth_l2 = None
            predicted_l2 = None
            rel_err_l2 = None
            signed_residual_l2 = None
            l2_status = "typed_unavailable_persistent"

        # L1 status + values: consume the predicted-L1
        # artifact when available. The admitted-label index
        # provides the L1 truth fraction; the predicted-L1
        # artifact provides the L1 head's per-cell scalar
        # prediction. ``rel_err_l1`` is computed only when
        # BOTH truth and a scalar prediction exist;
        # otherwise the row's L1 status remains
        # ``typed_unavailable_persistent`` and the bucket's
        # typed-unavailable counter increments.
        if admitted_row is not None:
            l1_pct = admitted_row.get(
                "cache_hit_rate_l1tex_sector_pct"
            )
            ncu_truth_l1 = (
                float(l1_pct) / 100.0
                if l1_pct is not None else None
            )
        else:
            ncu_truth_l1 = None
        predicted_l1_row = predicted_l1_by_key.get(key)
        if (
            predicted_l1_row is not None
            and admitted_row is not None
        ):
            # Identity guard: the predicted-L1 row's identity
            # tuple MUST match the admitted-label row's
            # identity tuple. The report cannot consume a
            # prediction made against a different binary /
            # source_commit / kernel signature than the
            # admitted truth label.
            _check_predicted_l1_identity(
                predicted_l1_row=predicted_l1_row,
                admitted_row=admitted_row,
                slug=slug,
                shape_point=sp,
            )
        predicted_l1 = None
        rel_err_l1 = None
        l1_status = "typed_unavailable_persistent"
        if (
            admission_status == "admitted"
            and predicted_l1_row is not None
            and predicted_l1_row.get("result_kind") in (
                "scalar", "bounded",
            )
        ):
            predicted_l1_value = predicted_l1_row.get(
                "predicted_hit_rate"
            )
            if predicted_l1_value is not None:
                predicted_l1 = float(predicted_l1_value)
                if ncu_truth_l1 is not None and ncu_truth_l1 > 0.0:
                    rel_err_l1 = _rel_err(predicted_l1, ncu_truth_l1)
                    l1_status = "scalar_present"
                elif ncu_truth_l1 is not None:
                    # Zero-truth cell (l1tex_label_status=all_zero, e.g.
                    # working_set_capacity_sweep): a relative error against a
                    # zero denominator is not meaningful — scoring it via the
                    # epsilon denominator would let one cell dominate the
                    # bucket geomean. Per the low-support denominator-floor
                    # doctrine, the cell keeps its scalar prediction in the
                    # report column but stays OUT of the rel_err geomean
                    # (counted as non-scored, same as the task12 preflight's
                    # truth<=0 exclusion — the two consumers must agree).
                    l1_status = "typed_unavailable_persistent"
                else:
                    # We have a scalar prediction but no
                    # truth label (e.g., the 2 microbench
                    # cells that need Path B operator NCU).
                    # The L1 path is admitted as a scalar
                    # prediction, but rel_err cannot be
                    # computed; the cell still contributes
                    # a scalar to the report's prediction
                    # column while staying out of the rel_err
                    # geomean. Keep l1_status as typed_unavailable_persistent
                    # so the cell does not enter the
                    # friendly_l1 scalar accounting.
                    l1_status = "typed_unavailable_persistent"

        # Residual decomposition: in the absence of a
        # quantitative bucket-attribution model, the
        # report records attributed_to_named_bucket=0.0
        # and unattributable_per_cell=signed_residual_l2
        # so the arithmetic check (attributed +
        # unattributable == signed_residual) passes.
        if signed_residual_l2 is not None:
            decomposition = {
                "attributed_to_named_bucket": 0.0,
                "unattributable_per_cell": signed_residual_l2,
            }
        else:
            decomposition = {
                "attributed_to_named_bucket": None,
                "unattributable_per_cell": None,
            }

        plan_2_candidate = _classify_plan_2_candidate(
            l1_status=l1_status,
            l2_status=l2_status,
            rel_err_l1=rel_err_l1,
            rel_err_l2=rel_err_l2,
            bucket=bucket,
        )

        rows.append({
            "arch": "sm_86",
            "slug": slug,
            "shape_point": sp,
            "admission_status": admission_status,
            "predictor_design_coverage_bucket": bucket,
            "ncu_truth_l1_fraction": ncu_truth_l1,
            "predicted_l1_hit_rate": predicted_l1,
            "rel_err_l1": rel_err_l1,
            "l1_status": l1_status,
            "ncu_truth_l2_fraction": ncu_truth_l2,
            "predicted_l2_hit_rate": predicted_l2,
            "rel_err_l2": rel_err_l2,
            "l2_status": l2_status,
            "signed_residual_l2": signed_residual_l2,
            "signed_residual_decomposition": decomposition,
            "plan_2_candidate": plan_2_candidate,
        })

        # Per-bucket accounting (admitted cells only;
        # excluded cells do not contribute to the
        # headline geomeans).
        if admission_status == "admitted":
            if bucket == "predictor_friendly":
                if rel_err_l2 is not None:
                    friendly_l2_scalar.append(rel_err_l2)
                else:
                    friendly_l2_typed_unavailable += 1
                if rel_err_l1 is not None:
                    friendly_l1_scalar.append(rel_err_l1)
                else:
                    friendly_l1_typed_unavailable += 1
            else:
                if rel_err_l2 is not None:
                    hostile_l2_scalar.append(rel_err_l2)
                else:
                    hostile_l2_typed_unavailable += 1
                if rel_err_l1 is not None:
                    hostile_l1_scalar.append(rel_err_l1)
                else:
                    hostile_l1_typed_unavailable += 1

    per_bucket_headline = {
        "friendly_l1": _bucket_block(
            rel_errs_scalar=friendly_l1_scalar,
            n_typed_unavailable=friendly_l1_typed_unavailable,
        ),
        "friendly_l2": _bucket_block(
            rel_errs_scalar=friendly_l2_scalar,
            n_typed_unavailable=friendly_l2_typed_unavailable,
        ),
        "hostile_l1": _bucket_block(
            rel_errs_scalar=hostile_l1_scalar,
            n_typed_unavailable=hostile_l1_typed_unavailable,
        ),
        "hostile_l2": _bucket_block(
            rel_errs_scalar=hostile_l2_scalar,
            n_typed_unavailable=hostile_l2_typed_unavailable,
        ),
    }

    variance_verdict = str(ac5["precondition_status"])
    cross_cell_stddev = float(ac5["cross_cell_stddev"])
    override = dict(ac5.get("partial_closure_override", {}))

    hard_1 = (
        per_bucket_headline["friendly_l2"]["geomean_rel_err"]
        is not None
        and per_bucket_headline["friendly_l2"]["geomean_rel_err"]
        < _HARD_1_FRIENDLY_L2_THRESHOLD
    )
    hard_2 = (
        per_bucket_headline["friendly_l1"]["geomean_rel_err"]
        is not None
        and per_bucket_headline["friendly_l1"]["geomean_rel_err"]
        < _HARD_2_FRIENDLY_L1_THRESHOLD
    )
    hard_3 = cross_cell_stddev >= _DOCUMENTED_FLOOR_STDDEV

    closure_status = _classify_plan_1_closure_status(
        hard_1=hard_1, hard_2=hard_2, hard_3=hard_3,
        override_accepted=bool(override.get("accepted", False)),
    )

    ac6_recommendations = [
        {
            "feature_name": str(r["feature_name"]),
            "recommendation_for_plan_2": str(
                r["recommendation_for_plan_2"]
            ),
        }
        for r in ac6.get("recommendations", [])[
            :_AC6_QUOTED_RECOMMENDATIONS
        ]
    ]

    sidecar: dict = {
        "schema_version": 1,
        "generator": (
            "tools/sm_86_predictor_accuracy_recovery_v2/"
            "build_interim_diagnostic_report.py"
        ),
        "cohort_size": len(rows),
        "cells": rows,
        "per_bucket_headline": per_bucket_headline,
        "variance_precondition_verdict": variance_verdict,
        "cross_cell_stddev": cross_cell_stddev,
        "documented_floor_stddev": _DOCUMENTED_FLOOR_STDDEV,
        "ac6_top_recommendations": ac6_recommendations,
        "hard_targets": {
            "hard_1_friendly_l2_lt_0_15": hard_1,
            "hard_2_friendly_l1_lt_0_20": hard_2,
            "hard_3_stddev_ge_0_10": hard_3,
        },
        "plan_1_closure_status": closure_status,
        "partial_closure_override": override,
    }
    return sidecar


def validate_sidecar(sidecar: dict) -> None:
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(instance=sidecar, schema=schema)


def render_markdown(sidecar: dict) -> str:
    """Render a human-readable markdown report from the
    sidecar dict. The markdown is the operator-facing
    surface; the sidecar is the machine-readable contract.
    """
    cross_stddev = sidecar["cross_cell_stddev"]
    floor = sidecar["documented_floor_stddev"]
    closure = sidecar["plan_1_closure_status"]
    override = sidecar["partial_closure_override"]
    bucket = sidecar["per_bucket_headline"]

    def _fmt(x: float | None, n: int = 4) -> str:
        return "n/a" if x is None else f"{x:.{n}f}"

    lines: list[str] = []
    lines.append("# Interim Diagnostic Report")
    lines.append("")
    lines.append(
        "**Cohort**: 13-cell HEADLINE_COHORT_13 "
        "(9 admitted + 4 excluded under the operator "
        "partial-closure override)."
    )
    lines.append(
        f"**Closure status**: `{closure}`."
    )
    lines.append(
        f"**Variance precondition verdict**: "
        f"`{sidecar['variance_precondition_verdict']}` "
        f"(cross_cell_stddev={_fmt(cross_stddev)} vs "
        f"documented_floor_stddev={_fmt(floor)})."
    )
    lines.append("")
    lines.append("## Hard Targets")
    lines.append("")
    hard = sidecar["hard_targets"]
    lines.append(
        f"- HARD-1 (friendly_l2 geomean < 0.15): "
        f"{'PASS' if hard['hard_1_friendly_l2_lt_0_15'] else 'MISS'} "
        f"(observed geomean={_fmt(bucket['friendly_l2']['geomean_rel_err'])})."
    )
    lines.append(
        f"- HARD-2 (friendly_l1 geomean < 0.20): "
        f"{'PASS' if hard['hard_2_friendly_l1_lt_0_20'] else 'MISS'} "
        f"(observed geomean={_fmt(bucket['friendly_l1']['geomean_rel_err'])})."
    )
    lines.append(
        f"- HARD-3 (cross_cell_stddev >= 0.10): "
        f"{'PASS' if hard['hard_3_stddev_ge_0_10'] else 'MISS'} "
        f"(observed stddev={_fmt(cross_stddev)})."
    )
    lines.append("")
    lines.append("## Per-Bucket Headline (independent L1 / L2 geomeans)")
    lines.append("")
    lines.append(
        "| Bucket-Layer | n_cells | n_scalar | "
        "n_typed_unavailable | geomean_rel_err |"
    )
    lines.append(
        "|--------------|---------|----------|"
        "---------------------|-----------------|"
    )
    for key in ("friendly_l1", "friendly_l2", "hostile_l1", "hostile_l2"):
        b = bucket[key]
        lines.append(
            f"| {key} | {b['n_cells']} | {b['n_scalar']} | "
            f"{b['n_typed_unavailable']} | "
            f"{_fmt(b['geomean_rel_err'])} |"
        )
    lines.append("")
    lines.append("## Candidate-Feature Catalog Top Recommendations (quoted)")
    lines.append("")
    for r in sidecar["ac6_top_recommendations"]:
        lines.append(
            f"- `{r['feature_name']}` → "
            f"`{r['recommendation_for_plan_2']}`"
        )
    lines.append("")
    lines.append("## Per-Cell Residual Decomposition")
    lines.append("")
    lines.append(
        "| slug | shape_point | admission_status | bucket "
        "| ncu_truth_l1 | predicted_l1 | rel_err_l1 | l1_status "
        "| ncu_truth_l2 | predicted_l2 | rel_err_l2 | l2_status "
        "| signed_residual_l2 | plan_2_candidate |"
    )
    lines.append(
        "|------|-------------|-----------------|--------|"
        "-------------|--------------|------------|-----------|"
        "-------------|--------------|------------|-----------|"
        "--------------------|------------------|"
    )
    for c in sidecar["cells"]:
        lines.append(
            f"| {c['slug']} | {c['shape_point']} "
            f"| {c['admission_status']} "
            f"| {c['predictor_design_coverage_bucket']} "
            f"| {_fmt(c['ncu_truth_l1_fraction'])} "
            f"| {_fmt(c['predicted_l1_hit_rate'])} "
            f"| {_fmt(c['rel_err_l1'])} | {c['l1_status']} "
            f"| {_fmt(c['ncu_truth_l2_fraction'])} "
            f"| {_fmt(c['predicted_l2_hit_rate'])} "
            f"| {_fmt(c['rel_err_l2'])} | {c['l2_status']} "
            f"| {_fmt(c['signed_residual_l2'])} "
            f"| {c['plan_2_candidate']} |"
        )
    lines.append("")
    lines.append("## Operator Partial-Closure Override")
    lines.append("")
    lines.append(f"- Accepted: {override.get('accepted', False)}")
    lines.append(
        f"- Date: {override.get('date_iso', 'n/a')}"
    )
    lines.append(
        f"- Decided admitted count: "
        f"{override.get('decided_admitted_count', 'n/a')}"
    )
    lines.append("")
    lines.append("**Reason (verbatim):**")
    lines.append("")
    lines.append(f"> {override.get('reason', 'n/a')}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the interim diagnostic report markdown "
            "+ sidecar over the HEADLINE_COHORT_13."
        ),
    )
    parser.add_argument(
        "--ac5-artifact", type=Path,
        default=_DEFAULT_AC5_ARTIFACT,
    )
    parser.add_argument(
        "--ac6-artifact", type=Path,
        default=_DEFAULT_AC6_ARTIFACT,
    )
    parser.add_argument(
        "--cohort-accounting", type=Path,
        default=_DEFAULT_COHORT_ACCOUNTING,
    )
    parser.add_argument(
        "--label-admitted", type=Path,
        default=_DEFAULT_LABEL_ADMITTED,
    )
    parser.add_argument(
        "--predicted-l1-artifact", type=Path,
        default=_DEFAULT_PREDICTED_L1_ARTIFACT,
        help=(
            "Path to the per-admitted-cell predicted-L1 "
            "artifact emitted by "
            "compute_predicted_l1_per_admitted_cell.py. "
            "When the file does not exist OR is set to '' "
            "(empty), the report falls back to the legacy "
            "predicted_l1=None behavior."
        ),
    )
    parser.add_argument(
        "--out-md", type=Path, default=_DEFAULT_OUT_MD,
    )
    parser.add_argument(
        "--out-sidecar", type=Path,
        default=_DEFAULT_OUT_SIDECAR,
    )
    args = parser.parse_args(argv)

    predicted_l1_path: Path | None = args.predicted_l1_artifact
    if predicted_l1_path is not None and str(predicted_l1_path) == "":
        predicted_l1_path = None

    try:
        sidecar = build_report(
            ac5_artifact_path=args.ac5_artifact,
            ac6_artifact_path=args.ac6_artifact,
            cohort_accounting_path=args.cohort_accounting,
            label_admitted_path=args.label_admitted,
            predicted_l1_artifact_path=predicted_l1_path,
        )
        validate_sidecar(sidecar)
    except (ValueError, jsonschema.ValidationError) as exc:
        sys.stderr.write(
            f"build_interim_diagnostic_report: {exc}\n"
        )
        return 2

    args.out_sidecar.parent.mkdir(parents=True, exist_ok=True)
    with args.out_sidecar.open("w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)
        fh.write("\n")

    md = render_markdown(sidecar)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_md.open("w", encoding="utf-8") as fh:
        fh.write(md)
        if not md.endswith("\n"):
            fh.write("\n")

    sys.stderr.write(
        f"build_interim_diagnostic_report: wrote {args.out_md} "
        f"(cohort_size={sidecar['cohort_size']}, "
        f"closure={sidecar['plan_1_closure_status']})\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
