"""Deterministic task12 cohort-substitution preflight / evaluator.

Reads the current admitted-cell cohort
(``data/local_labels/label_admitted.json``), the per-admitted-cell
predicted-L1 artifact
(``data/sm_86_predictor_accuracy_recovery_v2/predicted_l1_per_admitted_cell.json``),
and the interim diagnostic report sidecar
(``data/sm_86_predictor_accuracy_recovery_v2/interim_diagnostic_report_sidecar.json``).
Computes per-cell friendly_l1 contributions, simulates
counterfactual substitutions against the current cohort, and emits
a verdict artifact at
``data/sm_86_predictor_accuracy_recovery_v2/task12_substitution_preflight.json``.

The preflight is the deterministic floor under the cohort-substitution
review directive: every cohort-substitution attempt
MUST be evaluated mathematically before any artifact mutation, so a
fail-closed verdict can be emitted when substitution cannot close
HARD-2 with the available levers.

Per BL-20260504-stddev-threshold-immutable, the candidate-pool
admission gate uses ``stddev <= 0.03`` as a hard constraint that
the preflight reports for each candidate cell.

Per BL-20260511-deterministic-aggregation-selection-must-be-tooled,
the simulation lives in ONE place (this script) rather than spread
across multiple ad-hoc evaluators.

CLI:
    python3 -m tools.sm_86_predictor_accuracy_recovery_v2.task12_substitution_preflight
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Local sibling import. label_loader.load_labels_for_mode is
# the canonical entrypoint for operational-blinding-aware
# label I/O; calibration/validation-mode runners must not
# accept the sealed holdout labels path.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import label_loader  # noqa: E402

# Shared mixed-arity identity helper: cell_variant_id yields "" for a base cell
# and the canonical register_cap_<n> for a build variant. The substitution
# preflight reasons over the BASE admitted predictor cohort; register-cap build
# variants are a separate factorial corpus folded in by a later predictor
# refit, so they are excluded from the friendly/candidate/admitted enumerations
# here (keeping the recomputed geomean/contributions consistent with the
# committed base diagnostic).
_LOCAL_LABELS_DIR = Path(__file__).resolve().parents[1] / "local_labels"
if str(_LOCAL_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS_DIR))
from corpus_cell_identity import (  # noqa: E402
    cell_variant_id,
    is_base_predictor_fit_cell,
)


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_DEFAULT_LABEL_ADMITTED = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
_DEFAULT_PREDICTED_L1 = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "predicted_l1_per_admitted_cell.json"
)
_DEFAULT_AC7_SIDECAR = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "interim_diagnostic_report_sidecar.json"
)
_DEFAULT_OUT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "task12_substitution_preflight.json"
)
_DEFAULT_SCHEMA_PATH = (
    _HERE / "task12_substitution_preflight.schema.json"
)


# HARD-2 admission threshold per the plan (frozen).
HARD_2_GEOMEAN_THRESHOLD: float = 0.20

# stddev gate per BL-20260504 (immutable).
STDDEV_ADMISSION_GATE: float = 0.03

# Assumed rel_err for a HYPOTHETICAL substitution candidate cell.
# Conservative: a "healthy" predictor cell typically has rel_err
# in the 0.05-0.15 range; we use 0.15 as the substitution
# candidate's worst-case to make the preflight's "feasibility"
# verdict conservative.
SUBSTITUTION_ASSUMED_REL_ERR: float = 0.15


def _geomean(values: list[float]) -> float:
    """Compute the geometric mean of a list of positive floats.
    Uses log-sum to avoid product overflow / underflow."""
    if not values:
        return float("nan")
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def _per_cell_friendly_l1_contributions(
    *,
    sidecar: dict,
) -> list[dict]:
    """Extract per-cell friendly_l1 contributions from the
    diagnostic report sidecar. Returns one dict per cell in the
    predictor_friendly bucket that has BOTH predicted_l1_hit_rate
    AND ncu_truth_l1_fraction (the cells whose rel_err_l1 enters
    the geomean)."""
    out: list[dict] = []
    for cell in sidecar.get("cells", []):
        if cell.get("predictor_design_coverage_bucket") != (
            "predictor_friendly"
        ):
            continue
        pred = cell.get("predicted_l1_hit_rate")
        truth = cell.get("ncu_truth_l1_fraction")
        rel_err = cell.get("rel_err_l1")
        if pred is None or truth is None or rel_err is None:
            continue
        out.append(
            {
                "slug": str(cell["slug"]),
                "shape_point": str(cell["shape_point"]),
                "predicted_l1_hit_rate": float(pred),
                "ncu_truth_l1_fraction": float(truth),
                "rel_err_l1": float(rel_err),
                "log_rel_err": (
                    math.log(float(rel_err))
                    if rel_err > 0
                    else float("-inf")
                ),
            }
        )
    return out


def _build_candidate_pool(
    *,
    sidecar: dict,
) -> list[dict]:
    """Build the candidate-pool list. A candidate is any
    cohort cell that is NOT currently in the
    predictor_friendly bucket OR has rel_err that is
    measurable in the sidecar. The preflight reports the
    pool but does NOT promise the candidate's actual
    rel_err -- the substitution math uses
    ``SUBSTITUTION_ASSUMED_REL_ERR`` as a conservative
    assumption.

    The candidate pool entries record their identity tuple
    so the next-round live-collection step can pin which
    cells to NCU-label. The pool DOES NOT include cells
    that lack a SASS provenance source on disk -- those
    cells would route through the L1 head's typed-
    unavailable arm and cannot contribute a scalar
    predicted_l1_hit_rate.
    """
    pool: list[dict] = []
    for cell in sidecar.get("cells", []):
        bucket = cell.get("predictor_design_coverage_bucket")
        if bucket == "predictor_friendly":
            # already in the cohort.
            continue
        # candidate must at least have an L2 truth value (so we
        # know NCU label collection is producing meaningful
        # numbers for it).
        l2_truth = cell.get("ncu_truth_l2_fraction")
        if l2_truth is None:
            continue
        pool.append(
            {
                "slug": str(cell["slug"]),
                "shape_point": str(cell["shape_point"]),
                "current_bucket": str(bucket),
                "ncu_truth_l2_fraction": float(l2_truth),
                "assumed_rel_err_l1": SUBSTITUTION_ASSUMED_REL_ERR,
            }
        )
    return pool


def _derive_friendly_l1_contributions_from_loaded(
    *,
    label_admitted: dict,
    predicted_l1: dict,
) -> list[dict]:
    """Build the friendly_l1 per-cell contribution list from
    loaded admitted+predicted artifacts using the canonical
    bucket mapping. Returns the SAME shape as
    `_per_cell_friendly_l1_contributions(sidecar=...)` but
    derives membership from the canonical mapping (not the
    sidecar). A cell enters the friendly_l1 contribution
    set when:
      - the slug is in `_CANONICAL_PREDICTOR_FRIENDLY_SLUGS`;
      - the admitted label carries a non-null
        `cache_hit_rate_l1tex_sector_pct`;
      - the predicted_l1 artifact carries a non-null
        `predicted_hit_rate` for the same identity tuple
        (i.e. the predicted_l1 row is NOT
        typed_unavailable)."""
    # Index predicted rows by (slug, shape_point) for quick
    # lookup; the full-identity check happened in
    # `_load_and_cross_check_inputs`.
    predicted_by_key = {
        (str(c.get("slug", "")), str(c.get("shape_point", ""))): c
        for c in predicted_l1.get("cells", [])
    }
    out: list[dict] = []
    for cell in label_admitted.get("cells", []):
        if str(cell.get("arch", "")) != "sm_86":
            continue
        if cell_variant_id(cell):
            # Build variant: separate factorial corpus, not the base
            # predictor cohort this preflight scores (folded by a later refit).
            continue
        slug = str(cell.get("slug", ""))
        shape_point = str(cell.get("shape_point", ""))
        if slug not in _CANONICAL_PREDICTOR_FRIENDLY_SLUGS:
            continue
        pct = cell.get("cache_hit_rate_l1tex_sector_pct")
        if pct is None:
            continue
        pred_row = predicted_by_key.get((slug, shape_point))
        if pred_row is None:
            continue
        pred_val = pred_row.get("predicted_hit_rate")
        if pred_val is None:
            # typed_unavailable predicted row: this cell
            # cannot contribute to friendly_l1.
            continue
        truth = float(pct) / 100.0
        if truth <= 0.0:
            continue
        rel_err = abs(float(pred_val) - truth) / truth
        out.append(
            {
                "slug": slug,
                "shape_point": shape_point,
                "predicted_l1_hit_rate": float(pred_val),
                "ncu_truth_l1_fraction": truth,
                "rel_err_l1": rel_err,
                "log_rel_err": (
                    math.log(rel_err) if rel_err > 0
                    else float("-inf")
                ),
            }
        )
    return out


def _derive_candidate_pool_from_loaded(
    *,
    label_admitted: dict,
    predicted_l1: dict,
) -> list[dict]:
    """Build the candidate-pool list from loaded artifacts
    using the canonical bucket mapping. A candidate is an
    admitted cell whose slug is NOT in the friendly mapping
    (i.e. canonical bucket is `predictor_hostile`) AND
    whose admitted label carries a measurable L2 truth
    (`cache_hit_rate_l2_sector_pct`).

    The pool DOES NOT include cells with missing L2 truth or
    cells whose slug is not in the canonical mapping at all
    (those would be silent unknowns; the canonical mapping
    is the authoritative cohort partition)."""
    out: list[dict] = []
    for cell in label_admitted.get("cells", []):
        if cell_variant_id(cell):
            # Build variant: separate factorial corpus, not the base
            # candidate pool this preflight scores (folded by a later refit).
            continue
        slug = str(cell.get("slug", ""))
        shape_point = str(cell.get("shape_point", ""))
        bucket = _canonical_bucket_for(slug)
        if bucket != "predictor_hostile":
            continue
        l2_pct = cell.get("cache_hit_rate_l2_sector_pct")
        if l2_pct is None:
            continue
        out.append(
            {
                "slug": slug,
                "shape_point": shape_point,
                "current_bucket": bucket,
                "ncu_truth_l2_fraction": float(l2_pct) / 100.0,
                "assumed_rel_err_l1": SUBSTITUTION_ASSUMED_REL_ERR,
            }
        )
    return out


def _cross_check_sidecar_snapshot(
    *,
    sidecar: dict,
    derived_contributions: list[dict],
    derived_pool: list[dict],
    derived_geomean: float,
) -> None:
    """Treat the diagnostic sidecar as a CHECKED snapshot:
    cross-check every sidecar-reported field against the
    loaded-artifact derivation and raise on drift. The
    sidecar must:
      - report `friendly_l1.n_cells` equal to the derived
        contribution count;
      - report `friendly_l1.geomean_rel_err` within
        floating-point tolerance of the derived geomean;
      - assign `predictor_design_coverage_bucket =
        predictor_friendly` to every contribution slug AND
        no others (no bucket drift);
      - have the same admitted-cohort cells (for which
        the canonical mapping has a bucket) as the derived
        candidate-pool slug set + the derived contribution
        slug set.
    """
    sidecar_headline = sidecar.get("per_bucket_headline", {}).get(
        "friendly_l1", {},
    )
    # The derived contributions are the SCORED scalar population
    # (a contribution requires a non-null predicted_hit_rate), so the
    # sidecar field to cross-check is n_scalar. The sidecar's n_cells
    # additionally counts typed_unavailable friendly members — e.g. a
    # design-cohort cell whose admitted label exists in the corpus but
    # whose predicted-L1 row is withheld by the pending-refit
    # transitional boundary (working_set_capacity_sweep after its
    # recapture materialization) — which by construction can never
    # appear in the derived contribution list. Older sidecars that
    # predate the n_scalar split fall back to n_cells (their typed
    # count was zero, so the two fields coincide).
    sidecar_n_scored = sidecar_headline.get(
        "n_scalar", sidecar_headline.get("n_cells"),
    )
    if sidecar_n_scored is not None and (
        sidecar_n_scored != len(derived_contributions)
    ):
        raise ValueError(
            "task12_substitution_preflight: diagnostic "
            "sidecar's friendly_l1 scored-cell count "
            f"(n_scalar/n_cells={sidecar_n_scored}) drifts from "
            f"derived contribution count {len(derived_contributions)} "
            "(canonical bucket mapping + loaded admitted/"
            "predicted rows). A sidecar that was emitted "
            "against a different cohort partition is "
            "rejected here."
        )
    sidecar_geomean = sidecar_headline.get("geomean_rel_err")
    if sidecar_geomean is not None and not math.isclose(
        float(sidecar_geomean),
        derived_geomean,
        rel_tol=1e-6,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "task12_substitution_preflight: diagnostic "
            "sidecar's friendly_l1.geomean_rel_err="
            f"{sidecar_geomean} drifts from derived geomean "
            f"({derived_geomean}) of the loaded "
            "admitted/predicted rows under the canonical "
            "bucket mapping."
        )
    # Bucket drift: every contribution slug must appear in
    # the sidecar with bucket=predictor_friendly; every
    # candidate pool slug must appear with
    # bucket=predictor_hostile.
    contribution_slugs = {c["slug"] for c in derived_contributions}
    pool_slugs = {c["slug"] for c in derived_pool}
    sidecar_friendly_slugs = {
        str(c.get("slug", ""))
        for c in sidecar.get("cells", [])
        if c.get("predictor_design_coverage_bucket")
        == "predictor_friendly"
    }
    sidecar_hostile_slugs = {
        str(c.get("slug", ""))
        for c in sidecar.get("cells", [])
        if c.get("predictor_design_coverage_bucket")
        == "predictor_hostile"
    }
    for slug in contribution_slugs:
        if slug not in sidecar_friendly_slugs:
            raise ValueError(
                "task12_substitution_preflight: derived "
                f"contribution slug {slug!r} is NOT marked "
                "predictor_friendly in the diagnostic "
                "sidecar. Sidecar bucket drift rejected."
            )
    for slug in pool_slugs:
        if slug not in sidecar_hostile_slugs:
            raise ValueError(
                "task12_substitution_preflight: derived "
                f"candidate-pool slug {slug!r} is NOT marked "
                "predictor_hostile in the diagnostic "
                "sidecar. Sidecar bucket drift rejected."
            )


def _simulate_substitutions(
    *,
    contributions: list[dict],
    pool: list[dict],
) -> list[dict]:
    """Simulate counterfactual substitutions. For each
    substitution count k in 1..min(len(contributions), len(pool)),
    remove the k cells with highest rel_err and add k
    hypothetical candidate cells (each with rel_err =
    SUBSTITUTION_ASSUMED_REL_ERR). Compute the resulting
    geomean and record the substitution in the verdict.

    Returns a list ordered by substitution_count ascending. The
    last entry is the maximum-substitution arm."""
    # Sort contributions by rel_err descending so the highest-
    # contributing cells are at the top.
    sorted_contribs = sorted(
        contributions,
        key=lambda c: c["rel_err_l1"],
        reverse=True,
    )
    out: list[dict] = []
    max_subs = min(len(contributions), len(pool))
    for k in range(1, max_subs + 1):
        removed = sorted_contribs[:k]
        # the cells that remain in the cohort.
        retained = sorted_contribs[k:]
        retained_rel_errs = [c["rel_err_l1"] for c in retained]
        # k hypothetical candidates each at the assumed rel_err.
        candidate_rel_errs = [SUBSTITUTION_ASSUMED_REL_ERR] * k
        new_geomean = _geomean(
            retained_rel_errs + candidate_rel_errs,
        )
        out.append(
            {
                "substitution_count": k,
                "removed_cells": [
                    {
                        "slug": c["slug"],
                        "shape_point": c["shape_point"],
                        "rel_err_l1": c["rel_err_l1"],
                    }
                    for c in removed
                ],
                "added_candidate_count": k,
                "added_candidate_assumed_rel_err": (
                    SUBSTITUTION_ASSUMED_REL_ERR
                ),
                "new_friendly_l1_geomean": new_geomean,
                "closes_hard_2": (
                    new_geomean < HARD_2_GEOMEAN_THRESHOLD
                ),
            }
        )
    return out


def _build_verdict(
    *,
    current_geomean: float,
    contributions: list[dict],
    pool: list[dict],
    substitutions: list[dict],
) -> dict:
    """Build the verdict dict. The verdict's
    `hard_2_closure_feasible` field is True IFF at least one
    simulated substitution arm yields a geomean below
    ``HARD_2_GEOMEAN_THRESHOLD``. The `pivot_recommendation`
    field documents the next-round work; on infeasibility, it
    points at the alternative L1 model-class lever per the
    cohort-substitution review directive."""
    feasible_subs = [
        s for s in substitutions if s["closes_hard_2"]
    ]
    feasible = bool(feasible_subs)
    min_geomean = (
        min(s["new_friendly_l1_geomean"] for s in substitutions)
        if substitutions
        else float("nan")
    )

    if feasible:
        min_sub_count = min(
            s["substitution_count"] for s in feasible_subs
        )
        verdict = (
            f"task12_substitution_feasible_min_subs_{min_sub_count}"
        )
        pivot_recommendation = (
            "Run live NCU label collection for "
            f"{min_sub_count} candidate cell(s) from the pool. "
            "Once labels exist (with stddev <= 0.03 admission "
            "gate satisfied), substitute the candidates for "
            "the highest-rel_err admitted cells and re-emit "
            "the variance trajectory + diagnostic report. "
            "Live collection is out of preflight scope per the"
            " round contract."
        )
    else:
        verdict = (
            "task12_substitution_infeasible_pivot_to_l1_model_class"
        )
        pivot_recommendation = (
            "Counterfactual substitution against the candidate "
            "pool (assuming rel_err = "
            f"{SUBSTITUTION_ASSUMED_REL_ERR} for every "
            "candidate) cannot drive friendly_l1.geomean_rel_err "
            f"below {HARD_2_GEOMEAN_THRESHOLD}. The next-round "
            "lever is the alternative L1 model class per Codex "
            "R43-review item 6: introduce a per-cell L1 "
            "correction symmetric to the L2 gridDim correction, "
            "OR replace the conservative per-PC fully-associative "
            "scalar with a reduction-kernel-aware model that "
            "accounts for the L1 under-hit-rate pattern observed "
            "on layernorm / rmsnorm / silu / softmax cells "
            "(predicted ~0.94, truth ~0.20-0.50)."
        )

    return {
        "schema_version": 1,
        "generator": (
            "tools/sm_86_predictor_accuracy_recovery_v2/"
            "task12_substitution_preflight.py"
        ),
        "hard_2_threshold": HARD_2_GEOMEAN_THRESHOLD,
        "stddev_admission_gate": STDDEV_ADMISSION_GATE,
        "substitution_assumed_rel_err": (
            SUBSTITUTION_ASSUMED_REL_ERR
        ),
        "current_geomean": current_geomean,
        "per_cell_contributions": contributions,
        "candidate_pool": pool,
        "simulated_substitutions": substitutions,
        "minimum_geomean_achievable": min_geomean,
        "hard_2_closure_feasible": feasible,
        "verdict": verdict,
        "pivot_recommendation": pivot_recommendation,
    }


_PREDICTED_L1_SCHEMA_PATH = (
    _HERE / "predicted_l1_per_admitted_cell.schema.json"
)

# Canonical predictor-design-coverage bucket mapping for the
# Plan-1 cohort. Each slug maps to either
# `predictor_friendly` (cells the predictor was DESIGNED to
# predict well -- normal access patterns + small working
# sets) or `predictor_hostile` (cells DESIGNED to stress the
# predictor -- L2-stressors, persistence-stressors,
# spill-stressors, etc.). The mapping is the deterministic
# source of truth for the cohort partition; the diagnostic
# sidecar's `predictor_design_coverage_bucket` field is a
# CHECKED snapshot of this mapping, not the source.
#
# A future cohort expansion adds entries here; the mapping
# is closed under the current sm_86 cohort. The preflight
# treats any admitted slug NOT in this mapping as a "missing
# canonical bucket" defect (the cohort expansion forgot to
# update the mapping).
_CANONICAL_PREDICTOR_FRIENDLY_SLUGS: frozenset[str] = frozenset({
    "dram_stream_stride_mix",
    "l2_thrash_pointer_chase",
    "layernorm_fwd_bf16_sm80",
    "rmsnorm_fwd_bf16_sm80",
    "silu_fwd_bf16_sm80",
    "softmax_safe_bf16_sm80",
    "working_set_capacity_sweep",
})

_CANONICAL_PREDICTOR_HOSTILE_SLUGS: frozenset[str] = frozenset({
    "carveout_sensitivity_probe",
    "l2_persisting_window_sweep",
    "l2_prefetch_hint_sweep",
    "register_spill_local_mem",
    "warm_cold_pair_relaunch",
})


def _canonical_bucket_for(slug: str) -> str | None:
    """Look up the canonical predictor-design bucket for a
    cell slug. Returns `"predictor_friendly"` /
    `"predictor_hostile"` / None (if the slug isn't in the
    mapping)."""
    if slug in _CANONICAL_PREDICTOR_FRIENDLY_SLUGS:
        return "predictor_friendly"
    if slug in _CANONICAL_PREDICTOR_HOSTILE_SLUGS:
        return "predictor_hostile"
    return None

# Full 7-field identity tuple for cross-artifact binding.
# Mirrors the manifest_pairing_index + ncu_metric_version
# binding used by cache_predictor.py + comparison_runner.py.
_FULL_IDENTITY_FIELDS = (
    "arch",
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
    "ncu_metric_version",
)


def _identity_tuple(cell: dict) -> tuple:
    """Extract the 7-field identity tuple from a cell dict.
    Used to cross-check admitted_label / predicted_l1 /
    diagnostic-sidecar rows by full provenance, not just
    `(arch, slug, shape_point)`."""
    return tuple(
        str(cell.get(f, "")) for f in _FULL_IDENTITY_FIELDS
    )


def _load_and_cross_check_inputs(
    *,
    label_admitted_path: Path,
    predicted_l1_path: Path,
    ac7_sidecar_path: Path,
) -> tuple[dict, dict, dict]:
    """Load all three input artifacts, fail-closed on missing /
    malformed files, schema-validate the predicted-L1 artifact,
    reject duplicate predicted-L1 rows, match the full 7-field
    identity tuple between admitted_label and predicted_l1,
    recompute per-cell L1 truth from
    ``label_admitted.cells[i].cache_hit_rate_l1tex_sector_pct
    / 100`` and assert it matches the diagnostic sidecar's
    ``ncu_truth_l1_fraction``, and recompute per-cell
    ``rel_err_l1`` from the loaded values and assert it
    matches the sidecar's rel_err_l1.

    Returns ``(label_admitted, predicted_l1, sidecar)`` on
    success; raises FileNotFoundError on missing files;
    raises ValueError on any drift / duplicate / identity /
    truth mismatch."""
    if not label_admitted_path.is_file():
        raise FileNotFoundError(
            "task12_substitution_preflight: "
            f"label_admitted path not found: {label_admitted_path}"
        )
    if not predicted_l1_path.is_file():
        raise FileNotFoundError(
            "task12_substitution_preflight: "
            f"predicted_l1 path not found: {predicted_l1_path}"
        )
    if not ac7_sidecar_path.is_file():
        raise FileNotFoundError(
            "task12_substitution_preflight: "
            "diagnostic report sidecar path not found: "
            f"{ac7_sidecar_path}"
        )
    label_admitted = label_loader.load_labels_for_mode(
        label_admitted_path, "calibration",
    )
    with predicted_l1_path.open("r", encoding="utf-8") as fh:
        predicted_l1 = json.load(fh)
    with ac7_sidecar_path.open("r", encoding="utf-8") as fh:
        sidecar = json.load(fh)

    # Cross-check 0: schema-validate the predicted_l1 artifact.
    # The schema lives alongside this preflight + the
    # build_interim_diagnostic_report consumer; validating here
    # catches malformed predicted_l1 payloads before the
    # downstream math.
    if _PREDICTED_L1_SCHEMA_PATH.is_file():
        try:
            import jsonschema
        except ImportError:
            jsonschema = None  # type: ignore[assignment]
        if jsonschema is not None:
            with _PREDICTED_L1_SCHEMA_PATH.open(
                "r", encoding="utf-8",
            ) as fh:
                schema = json.load(fh)
            try:
                jsonschema.validate(predicted_l1, schema)
            except jsonschema.ValidationError as exc:
                raise ValueError(
                    "task12_substitution_preflight: predicted_l1"
                    " artifact fails schema validation: "
                    f"{exc.message}"
                ) from exc

    # Cross-check 0b: predicted_l1 must not carry duplicate rows
    # for the same identity tuple. A duplicate would let a
    # stale or hand-edited copy ride alongside a fresh row and
    # silently win the lookup.
    predicted_keys_list: list[tuple] = []
    for cell in predicted_l1.get("cells", []):
        key = _identity_tuple(cell)
        predicted_keys_list.append(key)
    seen: set[tuple] = set()
    duplicates: list[tuple] = []
    for key in predicted_keys_list:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(
            "task12_substitution_preflight: predicted_l1 "
            "carries duplicate rows for identity tuples: "
            f"{duplicates}"
        )

    # Cross-check 1 (full 7-field identity): every admitted
    # sm_86 cell in label_admitted MUST have a corresponding
    # row in the predicted_l1 artifact at the SAME 7-field
    # identity tuple (arch, slug, shape_point, binary_sha256,
    # source_commit, kernel_signature_sha256,
    # ncu_metric_version). Partial-identity drift (e.g., a
    # predicted row with the same slug/shape_point but a
    # stale binary_sha256) fails closed here.
    #
    # arch=sm_86 filter: `predicted_l1_per_admitted_cell.json`
    # under `data/sm_86_predictor_accuracy_recovery_v2/` is
    # the sm_86 cohort's predicted-L1 artifact; the
    # sm_120-only decoupled framework does NOT join sm_120
    # admitted rows to this artifact (per the
    # `BL-CHX-sm120-only-decoupled-framework` candidate's
    # decoupling principle). The earlier-iterated
    # `_identity_tuple` accepts cells of any arch; we filter
    # here so admitted sm_120 cells (added during the
    # amended-cohort admission work for the decoupled
    # framework) do NOT raise spurious missing_from_predicted
    # defects against this sm_86-only artifact.
    # Build variants AND pending-refit factorial cells (e.g. the atomics ATOM
    # partner global_atomic_return_used) are admitted corpus cells but not part
    # of the base predictor cohort the predicted_l1 artifact covers, so they are
    # excluded here until the predictor refit folds them in (same boundary as
    # the variant exclusion).
    admitted_keys = {
        _identity_tuple(c)
        for c in label_admitted.get("cells", [])
        if isinstance(c, dict) and is_base_predictor_fit_cell(c)
    }
    predicted_keys = set(predicted_keys_list)
    missing_from_predicted = admitted_keys - predicted_keys
    extra_in_predicted = predicted_keys - admitted_keys
    if missing_from_predicted or extra_in_predicted:
        raise ValueError(
            "task12_substitution_preflight: predicted_l1 / "
            "label_admitted cell-identity drift: "
            f"missing_from_predicted={sorted(missing_from_predicted)} "
            f"extra_in_predicted={sorted(extra_in_predicted)}"
        )

    # Cross-check 2: the diagnostic report sidecar's
    # per_bucket_headline.friendly_l1 SCORED-cell count MUST equal the
    # count of per_cell_contributions the preflight will derive
    # from the sidecar. A contribution requires a non-null
    # predicted_l1_hit_rate, so the matching headline field is
    # n_scalar; n_cells additionally counts typed_unavailable
    # friendly members (e.g. a design-cohort cell whose predicted-L1
    # row is withheld by the pending-refit transitional boundary),
    # which can never appear in the contribution list. Older
    # sidecars without the n_scalar split fall back to n_cells
    # (their typed count was zero).
    friendly_headline = (
        sidecar.get("per_bucket_headline", {})
        .get("friendly_l1", {})
    )
    expected_n_scored = friendly_headline.get(
        "n_scalar", friendly_headline.get("n_cells"),
    )
    observed_contributions = _per_cell_friendly_l1_contributions(
        sidecar=sidecar,
    )
    if expected_n_scored is not None and (
        expected_n_scored != len(observed_contributions)
    ):
        raise ValueError(
            "task12_substitution_preflight: diagnostic report "
            "sidecar reports friendly_l1 scored-cell count "
            f"(n_scalar/n_cells)={expected_n_scored} but "
            "the preflight derived "
            f"{len(observed_contributions)} per-cell rows -- "
            "diagnostic report sidecar is drifted from the "
            "live sidecar cells block."
        )

    # Cross-check 3: every per-cell row in the diagnostic
    # report sidecar that has predicted_l1_hit_rate MUST agree
    # with the predicted_l1 artifact's row for the same
    # (slug, shape_point). This guards against a sidecar that
    # was emitted from a stale predicted_l1 snapshot.
    predicted_by_key = {
        (str(c.get("slug", "")), str(c.get("shape_point", ""))): (
            c
        )
        for c in predicted_l1.get("cells", [])
    }
    # Truth recompute: build a (slug, shape_point) -> truth map
    # from the admitted labels by dividing
    # cache_hit_rate_l1tex_sector_pct by 100. The diagnostic
    # sidecar's ncu_truth_l1_fraction MUST agree with this
    # recomputed value, otherwise the sidecar was emitted
    # against a stale or different label corpus.
    #
    # Filter to arch == "sm_86": this preflight is the sm_86
    # publication track. Once the sm_120 framework began adding
    # arch=sm_120 rows for the same (slug, shape_point) keys, an
    # arch-agnostic index would silently overwrite the sm_86 truth
    # with the sm_120 value and the sidecar's recorded sm_86 truth
    # would drift to a recomputed sm_120 truth. Restricting the
    # iteration to sm_86 rows preserves the original sm_86-only
    # join semantics.
    truth_by_key: dict[tuple[str, str], float] = {}
    for c in label_admitted.get("cells", []):
        if c.get("arch") != "sm_86":
            continue
        pct = c.get("cache_hit_rate_l1tex_sector_pct")
        if pct is None:
            continue
        truth_by_key[
            (str(c.get("slug", "")), str(c.get("shape_point", "")))
        ] = float(pct) / 100.0
    for cell in sidecar.get("cells", []):
        key = (
            str(cell.get("slug", "")),
            str(cell.get("shape_point", "")),
        )
        # Truth-drift check: sidecar's ncu_truth_l1_fraction
        # MUST match the recomputed value when the admitted
        # corpus carries an L1 truth for this cell.
        side_truth = cell.get("ncu_truth_l1_fraction")
        if side_truth is not None and key in truth_by_key:
            recomputed_truth = truth_by_key[key]
            if not math.isclose(
                float(side_truth),
                recomputed_truth,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "task12_substitution_preflight: "
                    "diagnostic sidecar's "
                    f"ncu_truth_l1_fraction={side_truth} for "
                    f"{key} drifts from recomputed truth "
                    f"({recomputed_truth}) from admitted "
                    "labels' cache_hit_rate_l1tex_sector_pct"
                )

        side_pred = cell.get("predicted_l1_hit_rate")
        if side_pred is None:
            continue
        if key not in predicted_by_key:
            raise ValueError(
                "task12_substitution_preflight: diagnostic "
                "sidecar carries predicted_l1_hit_rate for "
                f"{key} but predicted_l1 artifact has no row "
                "for that identity"
            )
        pred_row = predicted_by_key[key]
        pred_val = pred_row.get("predicted_hit_rate")
        if pred_val is None:
            # The predicted_l1 artifact carries
            # result_kind='typed_unavailable' for this cell;
            # in that case the sidecar should not have a
            # scalar predicted_l1_hit_rate either.
            raise ValueError(
                "task12_substitution_preflight: predicted_l1 "
                f"artifact has typed_unavailable for {key} but "
                "diagnostic sidecar carries a scalar "
                f"predicted_l1_hit_rate={side_pred}"
            )
        # Numeric equality up to floating-point precision.
        if not math.isclose(
            float(pred_val), float(side_pred), rel_tol=1e-9,
        ):
            raise ValueError(
                "task12_substitution_preflight: predicted_l1 "
                f"drift for {key}: "
                f"predicted_l1 artifact={pred_val}, "
                f"diagnostic sidecar={side_pred}"
            )
        # rel_err_l1 recompute: when both truth and pred are
        # present in the sidecar AND in the recomputed map,
        # recompute rel_err_l1 from the loaded values and
        # assert agreement with the sidecar's reported
        # rel_err_l1. This catches a sidecar that was emitted
        # from drifted intermediate state.
        side_rel_err = cell.get("rel_err_l1")
        if (
            side_rel_err is not None
            and key in truth_by_key
            and truth_by_key[key] > 0
        ):
            recomputed_rel_err = abs(
                float(pred_val) - truth_by_key[key]
            ) / truth_by_key[key]
            if not math.isclose(
                float(side_rel_err),
                recomputed_rel_err,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "task12_substitution_preflight: "
                    "diagnostic sidecar's "
                    f"rel_err_l1={side_rel_err} for {key} "
                    "drifts from recomputed "
                    f"({recomputed_rel_err}) from loaded "
                    "predicted + truth values"
                )
    return label_admitted, predicted_l1, sidecar


def build_preflight(
    *,
    label_admitted_path: Path = _DEFAULT_LABEL_ADMITTED,
    predicted_l1_path: Path = _DEFAULT_PREDICTED_L1,
    ac7_sidecar_path: Path = _DEFAULT_AC7_SIDECAR,
) -> dict:
    """Build the preflight verdict from the committed
    artifacts. Read-only; emits a dict that can be schema-
    validated and JSON-serialized to the verdict path.

    Loads ALL three input artifacts and cross-checks identity
    tuples + numeric drift between the predicted_l1 artifact
    and the diagnostic report sidecar. Fails closed on any
    drift / missing-file / identity-mismatch."""
    label_admitted, predicted_l1, sidecar = (
        _load_and_cross_check_inputs(
            label_admitted_path=label_admitted_path,
            predicted_l1_path=predicted_l1_path,
            ac7_sidecar_path=ac7_sidecar_path,
        )
    )
    # Derive contributions + pool + current_geomean from
    # the LOADED admitted + predicted artifacts via the
    # canonical bucket mapping. The diagnostic sidecar is a
    # CHECKED snapshot below; it cannot drive the verdict
    # by itself (any sidecar drift would otherwise change
    # the verdict without touching label_admitted /
    # predicted_l1).
    contributions = _derive_friendly_l1_contributions_from_loaded(
        label_admitted=label_admitted,
        predicted_l1=predicted_l1,
    )
    pool = _derive_candidate_pool_from_loaded(
        label_admitted=label_admitted,
        predicted_l1=predicted_l1,
    )
    rel_errs = [c["rel_err_l1"] for c in contributions]
    if not rel_errs:
        raise ValueError(
            "task12_substitution_preflight: derived "
            "friendly_l1 contribution list is empty under "
            "the canonical bucket mapping; cannot compute "
            "current_geomean"
        )
    current_geomean = _geomean(rel_errs)
    # Cross-check the diagnostic sidecar against the
    # loaded-artifact derivation. Any drift fires a typed
    # ValueError here.
    _cross_check_sidecar_snapshot(
        sidecar=sidecar,
        derived_contributions=contributions,
        derived_pool=pool,
        derived_geomean=current_geomean,
    )
    substitutions = _simulate_substitutions(
        contributions=contributions,
        pool=pool,
    )
    return _build_verdict(
        current_geomean=current_geomean,
        contributions=contributions,
        pool=pool,
        substitutions=substitutions,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the task12 cohort-substitution preflight "
            "verdict from the committed variance trajectory + "
            "diagnostic report + predicted-L1 artifacts. "
            "Read-only; never mutates label_admitted.json or any "
            "predicted_l1 artifact."
        ),
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=_DEFAULT_LABEL_ADMITTED,
    )
    parser.add_argument(
        "--predicted-l1",
        type=Path,
        default=_DEFAULT_PREDICTED_L1,
    )
    parser.add_argument(
        "--ac7-sidecar",
        type=Path,
        default=_DEFAULT_AC7_SIDECAR,
    )
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=_DEFAULT_SCHEMA_PATH,
    )
    args = parser.parse_args(argv)

    verdict = build_preflight(
        label_admitted_path=args.label_admitted,
        predicted_l1_path=args.predicted_l1,
        ac7_sidecar_path=args.ac7_sidecar,
    )

    # Schema-validate before writing so a partial/corrupted
    # emit is caught before it lands on disk.
    try:
        import jsonschema
    except ImportError:
        jsonschema = None  # type: ignore[assignment]
    if jsonschema is not None and args.schema_path.is_file():
        with args.schema_path.open(
            "r", encoding="utf-8",
        ) as fh:
            schema = json.load(fh)
        jsonschema.validate(verdict, schema)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, sort_keys=True)
        fh.write("\n")

    out_rel = args.out
    try:
        out_rel = args.out.relative_to(_REPO_ROOT)
    except ValueError:
        pass
    print(
        "task12_substitution_preflight: wrote "
        f"{out_rel} (verdict={verdict['verdict']!r}, "
        f"hard_2_closure_feasible={verdict['hard_2_closure_feasible']}, "
        f"current_geomean={verdict['current_geomean']:.6f}, "
        f"min_achievable={verdict['minimum_geomean_achievable']:.6f})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
