#!/usr/bin/env python3
"""Slug-disjoint cycle scoring + the DVFS-decomposed ns layer for
the GCoM-lite head, on top of the committed evaluation artifact.

Scoring: LOSO and LOOCV are RECOMPUTED here — every fold actually
refits the evaluator's ridge (same machinery, same lambda) on the
remaining fit-pool cells and predicts the held-out cell; nothing is
copied from any prior timing lane. With exactly one fit cell per
slug in the current pool the two procedures COINCIDE (recorded
structurally). The headline is the geomean log-relative error over
the held-out fold predictions vs the directional 0.20 target (0.15
stretch) — an honest MISS is an acceptable close.

The ns layer publishes every modeled cell as an INTERVAL via
``t0 + max(alpha/f_mem, beta/f_core)``: alpha = the post-ridge
memory-attributed cycles, beta = the post-ridge compute+stall
cycles, t0 = 0 (launch overhead is not separable from the committed
evidence — documented), and the CORE clock band sourced from the
committed card properties' accepted-NVML covariate medians
(read-only evidence; the band must sit inside the plausibility
window or the interval is rejected). f_mem is bound EQUAL to f_core:
no committed evidence decouples a memory-clock axis (the cycle
counter is core-domain and the card latencies are measured in core
cycles), so the form collapses to ``predicted_cycles`` swept over
the band and regenerates the published interval exactly. A POINT-ns
claim requires BOTH gates — the interval floor above the committed
noise floor AND in-window NVML samples; the committed run manifests
carry NO NVML samples, so every committed row lands interval-only
with the typed reason, and the point path is pinned by synthetic
test arms.

The deploy crosswalk is honest: per fit cell it carries the
full-fit prediction, the LOSO held-out prediction, the cycles
label, the ns interval with its decomposition parameters + clock
evidence, the measured duration label, and whether the interval
contains it; the sealed-holdout-member cell stays prediction-only.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
for p in (str(REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gcom_lite_cycles_evaluator import (  # noqa: E402
    DEPLOYABLE_CHANNEL,
    FEATURE_NAMES,
    RIDGE_LAMBDA,
    _ridge_fit,
    assert_features_deploy_legal,
)

DEFAULT_GCOM = (
    REPO_ROOT / "data" / "local_labels" / "timing_predictions"
    / "gcom_lite_cycles_sm_86.json"
)
DEFAULT_COHORT = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort"
    / "sm86_timing_cohort_expanded.json"
)
DEFAULT_SPLIT = (
    REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_split.json"
)
DEFAULT_CARD = (
    REPO_ROOT / "data" / "pchase" / "sm_86_static_card.json"
)
DEFAULT_LEGACY_COHORT = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort"
    / "ac4_long_duration_cohort_sm_86.json"
)
DEFAULT_OUT = (
    REPO_ROOT / "data" / "local_labels" / "timing_predictions"
    / "gcom_lite_scoring_sm_86.json"
)
DEFAULT_SCHEMA = _HERE / "gcom_lite_scoring.schema.json"

_CYCLES_LABEL = "gpc__cycles_elapsed.max"
_DURATION_LABEL = "gpu__time_duration.sum"

GEOMEAN_TARGET = 0.20
GEOMEAN_STRETCH = 0.15
# Core-clock plausibility window: the committed card NVML policy's
# active floor up to a documented generous boost ceiling for this
# device family.
PLAUSIBILITY_CEILING_MHZ = 2100.0


class ScoringInputError(Exception):
    """A committed input is missing or fails its value audit."""


class NsClaimRejected(Exception):
    """A point-ns claim or interval violates its gates."""


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def core_clock_band_mhz(card: dict) -> tuple[float, float, list]:
    """The observed core-clock band: min/max of the accepted-NVML
    sm-clock medians across the card's value properties (read-only
    covariate evidence)."""
    medians = []
    for name, prop in card.get("properties", {}).items():
        cov = prop.get("covariates") or {}
        clock = cov.get("sm_clock_mhz_median")
        if isinstance(clock, (int, float)) and clock > 0:
            medians.append((name, float(clock)))
    if not medians:
        raise ScoringInputError(
            "no accepted-NVML sm-clock covariates on the card")
    values = [m[1] for m in medians]
    return min(values), max(values), medians


def plausibility_floor_mhz(card: dict) -> float:
    policy = card.get("nvml_policy") or {}
    floor = policy.get("active_clock_floor_mhz")
    if not isinstance(floor, (int, float)) or floor <= 0:
        raise ScoringInputError(
            "card nvml_policy.active_clock_floor_mhz unavailable")
    return float(floor)


def assert_clock_band_plausible(
    band: tuple[float, float], *, floor_mhz: float,
) -> None:
    lo, hi = band
    if lo < floor_mhz or hi > PLAUSIBILITY_CEILING_MHZ or lo > hi:
        raise NsClaimRejected(
            f"core clock band [{lo}, {hi}] MHz lies outside the "
            f"plausibility window [{floor_mhz}, "
            f"{PLAUSIBILITY_CEILING_MHZ}]"
        )


def ns_row(
    *,
    predicted_cycles: float,
    alpha_memory_cycles: float,
    beta_compute_stall_cycles: float,
    clock_band_mhz: tuple[float, float],
    plausibility_floor: float,
    floor_ns: float,
    in_window_nvml_samples: list | None,
) -> dict:
    """One cell's ns publication.

    The DVFS form is ``t0 + max(alpha/f_mem, beta/f_core)`` with
    alpha = the POST-RIDGE memory-attributed cycles and beta = the
    post-ridge compute+stall-attributed cycles. The committed
    evidence does NOT decouple a memory-clock axis from the core
    clock — ``gpc__cycles_elapsed.max`` is a core-clock counter and
    the card's L2/DRAM latencies are themselves measured in core
    cycles — so f_mem is bound EQUAL to f_core (both ride the
    core-clock band) and that gap is recorded on the row. Under the
    bottleneck (max) model the dominant attributed term equals the
    prediction exactly, so the form regenerates the published
    interval ``predicted_cycles`` swept over the band — asserted
    here as the bottleneck-equals-prediction identity (it fails
    closed if a future imbalance term breaks it).

    Interval is published whenever the band is plausible; a POINT-ns
    claim requires BOTH the committed noise floor AND in-window
    NVML samples."""
    assert_clock_band_plausible(
        clock_band_mhz, floor_mhz=plausibility_floor)
    bottleneck_cycles = max(alpha_memory_cycles,
                            beta_compute_stall_cycles)
    if predicted_cycles <= 0 or abs(
            bottleneck_cycles - predicted_cycles
    ) / predicted_cycles > 1e-9:
        raise NsClaimRejected(
            f"bottleneck-equals-prediction identity violated: "
            f"max(alpha={alpha_memory_cycles}, "
            f"beta={beta_compute_stall_cycles}) = "
            f"{bottleneck_cycles} != predicted "
            f"{predicted_cycles} (an imbalance term would break "
            f"the published form; the ns layer must be revised "
            f"before publishing)"
        )
    f_lo, f_hi = clock_band_mhz
    interval_lo_ns = predicted_cycles / (f_hi / 1000.0)
    interval_hi_ns = predicted_cycles / (f_lo / 1000.0)
    bottleneck = (
        "memory"
        if alpha_memory_cycles >= beta_compute_stall_cycles
        else "compute_stall"
    )
    row = {
        "interval_ns": [interval_lo_ns, interval_hi_ns],
        "decomposition": {
            "t0_ns": 0.0,
            "t0_note": (
                "launch overhead not separable from the committed "
                "evidence; absorbed into the interval"
            ),
            "alpha_memory_cycles": alpha_memory_cycles,
            "beta_compute_stall_cycles":
                beta_compute_stall_cycles,
            "bottleneck": bottleneck,
            "bottleneck_cycles": bottleneck_cycles,
            "form": "t0 + max(alpha/f_mem, beta/f_core)",
            "f_core_band_mhz": list(clock_band_mhz),
            "f_core_evidence": (
                "card accepted-NVML sm-clock covariate medians"
            ),
            "f_mem_binding": "f_mem == f_core",
            "f_mem_note": (
                "no committed evidence decouples a memory-clock "
                "axis: gpc__cycles_elapsed.max is a core-clock "
                "counter and the card L2/DRAM latencies are "
                "measured in core cycles, so f_mem is held equal "
                "to f_core over the band; an independent memory-"
                "DVFS arm would require committed per-cell "
                "memory-clock evidence that does not exist"
            ),
        },
    }
    if in_window_nvml_samples:
        if interval_lo_ns <= floor_ns:
            raise NsClaimRejected(
                f"point-ns claim below the committed noise floor "
                f"({interval_lo_ns} <= {floor_ns})"
            )
        ordered = sorted(in_window_nvml_samples)
        mid = len(ordered) // 2
        f_median = (
            ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0)
        assert_clock_band_plausible(
            (f_median, f_median), floor_mhz=plausibility_floor)
        row["point_ns"] = predicted_cycles / (f_median / 1000.0)
        row["point_ns_basis"] = {
            "n_in_window_nvml_samples":
                len(in_window_nvml_samples),
            "f_core_median_mhz": f_median,
        }
        row["ns_claim"] = "point_and_interval"
    else:
        row["point_ns"] = None
        row["point_ns_basis"] = None
        row["ns_claim"] = "interval_only"
        row["interval_only_reason"] = (
            "no_in_window_nvml_samples_committed"
        )
    return row


def _cell_key(row: dict) -> tuple:
    return (row["slug"], row["shape_point"], row["binary_sha256"])


def build_label_index(cohort: dict) -> dict:
    """Index expanded-cohort admitted timing labels by FULL identity
    (slug, shape_point, binary_sha256). Reject duplicate full-
    identity keys AND ambiguous (slug, shape_point) aliases that map
    to more than one binary_sha256 — either makes the timing-label
    join unsafe (a weak (slug, shape_point) join could silently pick
    a stale binary's labels)."""
    index: dict = {}
    alias: dict = {}
    for c in cohort["admitted"]:
        fk = (c["slug"], c["shape_point"], c["binary_sha256"])
        if fk in index:
            raise ScoringInputError(
                f"duplicate full-identity cohort label row: {fk}")
        index[fk] = c["timing_labels"]
        alias.setdefault(
            (c["slug"], c["shape_point"]), set()).add(
                c["binary_sha256"])
    ambiguous = sorted(
        k for k, hashes in alias.items() if len(hashes) > 1)
    if ambiguous:
        raise ScoringInputError(
            f"ambiguous (slug, shape_point) cohort aliases mapping "
            f"to multiple binary_sha256: {ambiguous}")
    return index


def assert_modeled_rows_have_labels(
    *, fit_rows: list, holdout_rows: list, label_index: dict,
) -> None:
    """Every modeled row must resolve to EXACTLY ONE full-identity
    cohort label. A stale/mismatched binary_sha256 or an absent
    label row fails closed here, before any fold fit or crosswalk
    label is consumed."""
    missing = sorted(
        _cell_key(r) for r in (fit_rows + holdout_rows)
        if _cell_key(r) not in label_index)
    if missing:
        raise ScoringInputError(
            f"modeled row(s) lack an exact full-identity cohort "
            f"label match (stale/mismatched binary_sha256 or absent "
            f"label row): {missing}")


def canonical_holdout_keys(split: dict) -> set:
    """The sealed-holdout identity keys read DIRECTLY from the
    frozen cohort split — the independent source of truth, not the
    upstream gcom artifact's own `status` field."""
    keys = set()
    for c in split["splits"]["holdout"]["cells"]:
        it = c["identity_tuple"]
        keys.add((it["slug"], it["shape_point"],
                  it["binary_sha256"]))
    return keys


def assert_holdout_containment(
    *, fit_rows: list, holdout_rows: list, canonical_holdout: set,
) -> None:
    """Cross-check the gcom partition against the canonical split.
    A cell whose label is consumed (every fit-pool row is both a
    fold train row and a crosswalk-label row) MUST NOT be a
    canonical holdout member; a row claiming holdout-reserved MUST
    be one. A mislabeled upstream `status` fails closed here rather
    than silently leaking a sealed label."""
    leaked = sorted(
        (r["slug"], r["shape_point"]) for r in fit_rows
        if _cell_key(r) in canonical_holdout)
    if leaked:
        raise ScoringInputError(
            f"canonical-holdout cell(s) entered the fit/scoring "
            f"pool (sealed label would be consumed): {leaked}")
    misreserved = sorted(
        (r["slug"], r["shape_point"]) for r in holdout_rows
        if _cell_key(r) not in canonical_holdout)
    if misreserved:
        raise ScoringInputError(
            f"row(s) marked holdout-reserved are not canonical "
            f"holdout members: {misreserved}")


def build_scoring(
    *,
    gcom_path: Path,
    cohort_path: Path,
    card_path: Path,
    legacy_cohort_path: Path,
    split_path: Path,
) -> dict:
    # Independent re-assertion that the scoring feature set carries
    # no forbidden timing / label-only field (the scorer reads
    # features straight from each cell's committed vector).
    assert_features_deploy_legal(FEATURE_NAMES)

    gcom = json.loads(gcom_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    card = json.loads(card_path.read_text(encoding="utf-8"))
    legacy = json.loads(
        legacy_cohort_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    canonical_holdout = canonical_holdout_keys(split)

    floor_ns = legacy.get("floor_ns")
    if not isinstance(floor_ns, (int, float)) or floor_ns <= 0:
        raise ScoringInputError(
            "legacy cohort floor_ns unavailable; the committed "
            "noise floor must be present evidence")
    band_lo, band_hi, band_evidence = core_clock_band_mhz(card)
    plaus_floor = plausibility_floor_mhz(card)
    assert_clock_band_plausible(
        (band_lo, band_hi), floor_mhz=plaus_floor)

    labels = build_label_index(cohort)
    fit_rows = [
        r for r in gcom["cells"]
        if r["status"] == "modeled_fit_pool"
    ]
    holdout_rows = [
        r for r in gcom["cells"]
        if r["status"] == "modeled_holdout_reserved"
    ]
    assert_holdout_containment(
        fit_rows=fit_rows, holdout_rows=holdout_rows,
        canonical_holdout=canonical_holdout)
    # Closed-identity gate: every modeled row must resolve to exactly
    # one full-identity cohort label BEFORE any fold fit or crosswalk
    # label is consumed; a stale/mismatched binary_sha256 or absent
    # label row fails closed here.
    assert_modeled_rows_have_labels(
        fit_rows=fit_rows, holdout_rows=holdout_rows,
        label_index=labels)
    if not fit_rows:
        raise ScoringInputError("no fit-pool rows to score")

    slugs = [r["slug"] for r in fit_rows]
    loso_equals_loocv = len(set(slugs)) == len(fit_rows)

    def _cycles_label(row: dict) -> float:
        return labels[_cell_key(row)][_CYCLES_LABEL]["value"]

    def _refit_predict(train: list, held: dict) -> dict:
        """Refit the evaluator's ridge on the TRAIN rows and scale
        the held cell's prior by the residual."""
        rows = [
            ([r["feature_vector"][n] for n in FEATURE_NAMES],
             math.log(_cycles_label(r) / r["analytical_prior_cycles"]))
            for r in train
        ]
        coeffs = _ridge_fit(rows, RIDGE_LAMBDA)
        x = [held["feature_vector"][n] for n in FEATURE_NAMES]
        residual = sum(c * v for c, v in zip(coeffs, x))
        predicted = (
            held["analytical_prior_cycles"] * math.exp(residual))
        label = _cycles_label(held)
        return {
            "held_out_slug": held["slug"],
            "shape_point": held["shape_point"],
            "n_train": len(train),
            "held_out_predicted_cycles": predicted,
            "label_cycles": label,
            "log_ratio": math.log(predicted / label),
            "abs_log_ratio": abs(math.log(predicted / label)),
        }

    # LOOCV: leave one CELL out.
    folds = [
        _refit_predict(
            [r for j, r in enumerate(fit_rows) if j != i],
            fit_rows[i])
        for i in range(len(fit_rows))
    ]
    geomean_rel = math.exp(
        sum(f["abs_log_ratio"] for f in folds) / len(folds)) - 1.0

    # LOSO: leave one SLUG out (all of that slug's cells); predict
    # each held cell from the slug-disjoint train pool. On the
    # one-cell-per-slug surface this partitions identically to LOOCV
    # and yields the same per-fold predictions, but it is computed
    # independently so a future multi-cell-per-slug surface gets a
    # genuinely distinct LOSO number.
    loso_folds = []
    for held_slug in sorted(set(slugs)):
        train = [r for r in fit_rows if r["slug"] != held_slug]
        for held in [r for r in fit_rows if r["slug"] == held_slug]:
            loso_folds.append(_refit_predict(train, held))
    loso_geomean_rel = math.exp(
        sum(f["abs_log_ratio"] for f in loso_folds)
        / len(loso_folds)) - 1.0

    achieved_vs_target = [
        {
            "gate": "loso_geomean_log_relative_error",
            "target": GEOMEAN_TARGET,
            "achieved": loso_geomean_rel,
            "met": loso_geomean_rel <= GEOMEAN_TARGET,
        },
        {
            "gate": "loso_geomean_log_relative_error_stretch",
            "target": GEOMEAN_STRETCH,
            "achieved": loso_geomean_rel,
            "met": loso_geomean_rel <= GEOMEAN_STRETCH,
        },
        {
            "gate": "loocv_geomean_log_relative_error",
            "target": GEOMEAN_TARGET,
            "achieved": geomean_rel,
            "met": geomean_rel <= GEOMEAN_TARGET,
        },
    ]

    def _robustness_surface(
        procedure: str, surface_folds: list, geomean: float,
    ) -> dict:
        return {
            "procedure": procedure,
            "fold_count": len(surface_folds),
            "distinct_held_out_slug_count": len(
                {f["held_out_slug"] for f in surface_folds}),
            "train_sizes": sorted(
                {f["n_train"] for f in surface_folds}),
            "geomean_log_relative_error": geomean,
            "target": GEOMEAN_TARGET,
            "stretch_target": GEOMEAN_STRETCH,
            "met": geomean <= GEOMEAN_TARGET,
            "met_stretch": geomean <= GEOMEAN_STRETCH,
        }

    robustness_tables = {
        "loso_equals_loocv": loso_equals_loocv,
        "coincidence_note": (
            "one fit cell per slug, so leave-one-slug-out and "
            "leave-one-cell-out partition identically and yield the "
            "same per-fold predictions and geomean"
            if loso_equals_loocv else
            "the fit pool has multiple cells per slug, so "
            "leave-one-slug-out and leave-one-cell-out differ and "
            "are reported separately"
        ),
        "surfaces": [
            _robustness_surface(
                "leave_one_slug_out", loso_folds, loso_geomean_rel),
            _robustness_surface(
                "leave_one_cell_out", folds, geomean_rel),
        ],
    }

    # ns layer + crosswalk
    crosswalk = []
    by_fold = {
        (f["held_out_slug"], f["shape_point"]): f for f in folds
    }
    for row in fit_rows + holdout_rows:
        terms = row["component_terms"]
        # Post-ridge attribution: scale each prior component by the
        # ridge factor g = predicted / analytical_prior so the
        # bottleneck (max) attributed term equals the prediction.
        ridge_factor = (
            row["predicted_cycles"] / row["analytical_prior_cycles"])
        alpha = terms["memory"]["cycles"] * ridge_factor
        beta = (
            terms["compute"]["cycles"] + terms["stall"]["cycles"]
        ) * ridge_factor
        ns = ns_row(
            predicted_cycles=row["predicted_cycles"],
            alpha_memory_cycles=alpha,
            beta_compute_stall_cycles=beta,
            clock_band_mhz=(band_lo, band_hi),
            plausibility_floor=plaus_floor,
            floor_ns=float(floor_ns),
            in_window_nvml_samples=None,  # none committed
        )
        is_holdout = row["status"] == "modeled_holdout_reserved"
        entry = {
            "slug": row["slug"],
            "shape_point": row["shape_point"],
            "status": row["status"],
            "full_fit_predicted_cycles": row["predicted_cycles"],
            "prediction_channel": row["prediction_channel"],
            "ns": ns,
        }
        if is_holdout:
            entry.update({
                "loso_held_out_predicted_cycles": None,
                "label_cycles": None,
                "duration_label_ns": None,
                "interval_contains_duration_label": None,
                "note": (
                    "sealed-holdout member: labels reserved for "
                    "the one-touch evidence-harness lane"
                ),
            })
        else:
            fold = by_fold[(row["slug"], row["shape_point"])]
            tl = labels[_cell_key(row)]
            duration = tl[_DURATION_LABEL]["value"]
            lo, hi = ns["interval_ns"]
            entry.update({
                "loso_held_out_predicted_cycles":
                    fold["held_out_predicted_cycles"],
                "label_cycles": fold["label_cycles"],
                "duration_label_ns": duration,
                "interval_contains_duration_label": (
                    lo <= duration <= hi),
            })
        crosswalk.append(entry)

    n_point_ns = sum(
        1 for r in crosswalk
        if r["ns"]["ns_claim"] == "point_and_interval")

    # DERIVED (not hardcoded): every fit-pool row had its label
    # consumed; the containment assertion above already raised if
    # any were a canonical holdout member, so this is provably
    # False — computed here so the emitted flag is evidence, not a
    # literal.
    holdout_labels_consumed = any(
        _cell_key(r) in canonical_holdout for r in fit_rows)

    # The committed noise floor is the legacy long-duration cohort's
    # floor; every published interval lower edge sits below it, so a
    # point-ns claim is impossible by construction on this short-
    # kernel cohort regardless of NVML availability. The floor is
    # retained as the committed noise evidence and to exercise the
    # synthetic gate.
    max_interval_lo = max(
        r["ns"]["interval_ns"][0] for r in crosswalk)
    floor_exceeds_all_intervals = float(floor_ns) > max_interval_lo

    return {
        "schema_version": 1,
        "artifact_kind": "gcom_lite_scoring",
        "arch": "sm_86",
        "generated_at_commit": _git_head(),
        "deployable_channel": DEPLOYABLE_CHANNEL,
        "inputs": {
            "gcom_artifact": str(gcom_path.relative_to(REPO_ROOT)),
            "cohort": str(cohort_path.relative_to(REPO_ROOT)),
            "card": str(card_path.relative_to(REPO_ROOT)),
            "cohort_split": str(split_path.relative_to(REPO_ROOT)),
            "legacy_cohort_floor_source": str(
                legacy_cohort_path.relative_to(REPO_ROOT)),
        },
        "metric_definitions": {
            "geomean_log_relative_error": (
                "exp(mean(|ln(pred/label)|)) - 1 over held-out "
                "fold predictions"
            ),
            "fold_procedure": (
                "per-fold ridge refit on the remaining fit-pool "
                "cells with the evaluator's lambda and features; "
                "the held-out cell's prior is scaled by the "
                "refit residual"
            ),
        },
        "holdout_containment": {
            "canonical_split_source": str(
                split_path.relative_to(REPO_ROOT)),
            "n_canonical_holdout_cells": len(canonical_holdout),
            "n_fit_pool_cells_cross_checked": len(fit_rows),
            "n_holdout_reserved_cells_cross_checked":
                len(holdout_rows),
            "note": (
                "fit/holdout membership cross-checked against the "
                "frozen split identity tuples, not the upstream "
                "status field; a mislabeled status fails closed"
            ),
        },
        "slug_disjointness": {
            "n_folds": len(folds),
            "n_distinct_slugs": len(set(slugs)),
            "loso_equals_loocv": loso_equals_loocv,
            "note": (
                "one fit cell per slug in the current pool, so "
                "leave-one-slug-out and leave-one-cell-out "
                "coincide" if loso_equals_loocv else None
            ),
        },
        "folds": folds,
        "achieved_vs_target": achieved_vs_target,
        "robustness_tables": robustness_tables,
        "ns_layer": {
            "noise_floor_ns": float(floor_ns),
            "noise_floor_provenance": (
                "the legacy long-duration cohort's committed "
                "floor_ns; a bare scalar carried forward as the "
                "committed noise evidence"
            ),
            "point_ns_floor_exceeds_all_intervals":
                floor_exceeds_all_intervals,
            "point_ns_impossible_note": (
                "every published interval lower edge sits below "
                "this floor, so no cell in this short-kernel cohort "
                "could make a credible point-ns claim even if NVML "
                "samples were committed; the floor is retained to "
                "exercise the synthetic point-ns gate"
            ),
            "core_clock_band_mhz": [band_lo, band_hi],
            "core_clock_band_evidence": [
                {"property": name, "sm_clock_mhz_median": v}
                for name, v in band_evidence
            ],
            "plausibility_window_mhz": [
                plaus_floor, PLAUSIBILITY_CEILING_MHZ],
            "n_point_ns_claims": n_point_ns,
            "point_ns_note": (
                "no committed run manifest carries in-window NVML "
                "samples, so every row is interval-only (the "
                "honest outcome); the point-ns path is pinned by "
                "synthetic test arms"
            ),
        },
        "deploy_crosswalk": sorted(
            crosswalk,
            key=lambda r: (r["slug"], r["shape_point"])),
        "holdout_labels_consumed": holdout_labels_consumed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcom", type=Path, default=DEFAULT_GCOM)
    parser.add_argument(
        "--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument(
        "--legacy-cohort", type=Path, default=DEFAULT_LEGACY_COHORT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--schema-path", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    artifact = build_scoring(
        gcom_path=args.gcom,
        cohort_path=args.cohort,
        card_path=args.card,
        legacy_cohort_path=args.legacy_cohort,
        split_path=args.split,
    )
    import jsonschema
    schema = json.loads(
        args.schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    headline = artifact["achieved_vs_target"][0]
    print(
        f"gcom_lite_slug_disjoint_scoring: wrote {args.out} "
        f"(geomean log-rel {headline['achieved']:.4f} vs "
        f"{headline['target']} -> met={headline['met']}; "
        f"{artifact['slug_disjointness']['n_folds']} folds)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
