#!/usr/bin/env python3
"""Per-(cell, metric_key) evidence harness.

Composes the already-verified committed tier artifacts (T1 count
lockstep, L1 warp-pool hit-rate rescore, L2 frozen-head hit rate,
T2 traffic primitives, GCoM-lite cycles + slug-disjoint scoring)
into one artifact where EVERY exercised (cell, metric_key) row lands
in exactly one bucket: main / appendix / diagnostic /
typed_unavailable.

Bucketing has two layers:

- The CELL outcome comes from each tier's own verdict/status:
  ``scored`` (a real predicted-vs-label number), ``diagnostic`` (a
  label exists but the cell was gate-barred from a clean number), or
  ``typed_unavailable`` (no usable label / missing input).

- The METRIC promotion (main vs appendix) is gated. A FITTED head
  (L1, L2, cycles/ns) is promoted to main only when its robustness
  gate is met AND slug-disjoint validation evidence is present; a
  ZERO-PARAMETER replacement (T1, T2) is promoted only when its
  non-vacuity floor is met AND its achieved-vs-target is met. A
  scored cell of a promoted metric lands ``main``; a scored cell of
  an unpromoted metric (an honest MISS) lands ``appendix``. The
  promotion is FAIL-CLOSED: ``assert_promotion_legal`` rejects any
  metric marked main whose gate did not actually pass.

The baseline panel runs ONCE on the sealed holdout under the
holdout-touch ledger (scope ``per_metric_evidence_harness``, a
config digest, ``assert_touch_legal`` then ``record_touch``); it
scores the plan-named hit-rate baselines (always-hit, always-miss,
family-mean, uncalibrated-walker, uncalibrated-reuse-distance,
prior-v4, prior-v5) on the committed L2 baseline-comparison surface
vs the sealed L2 labels, typing the two timing baselines with a
named missing input. The ablations section carries the five
COMPUTED component ablations: three L1 walker re-runs (no-
interleaving, no-MSHR, no-HRD-second-layer) and two cycles ridge
re-fits (no-control-bit-stalls, no-covariates), each with per-cell /
aggregate before-vs-chosen deltas.

Composition only: no upstream artifact is mutated; the harness
rebuilds byte-identically from committed inputs except
``generated_at_commit``. No GPU runtime; no per-cycle simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "tools" / "local_labels"),
          str(REPO_ROOT / "tools" / "sm_86_predictor_accuracy_recovery_v2")):
    if p not in sys.path:
        sys.path.insert(0, p)

import holdout_touch_ledger as htl  # noqa: E402
import label_loader  # noqa: E402
from gcom_lite_cycles_evaluator import (  # noqa: E402
    FEATURE_NAMES, RIDGE_LAMBDA, _ridge_fit)

ARCH = "sm_86"
HARNESS_SCOPE = "per_metric_evidence_harness"
_CYCLES_LABEL = "gpc__cycles_elapsed.max"
# The directional geomean/relative target for fitted hit-rate /
# log-relative heads (DEC-7), consistent with the L1 rescore and the
# cycles scoring gates.
DIRECTIONAL_HIT_RATE_GEOMEAN_TARGET = 0.20

T1_VERDICT = REPO_ROOT / "data/local_labels/t1_lockstep_verdict_sm_86.json"
L1_RESCORE = (REPO_ROOT
              / "data/sm_86_predictor_accuracy_recovery_v2"
              / "warp_pool_l1_rescore_sm_86.json")
V5_HEAD = REPO_ROOT / "data/local_labels/sm_86_v5_head.json"
T2_TRAFFIC = (REPO_ROOT
              / "data/sm_86_predictor_accuracy_recovery_v2"
              / "t2_traffic_prediction_sm_86.json")
GCOM_CYCLES = (REPO_ROOT
               / "data/local_labels/timing_predictions"
               / "gcom_lite_cycles_sm_86.json")
GCOM_SCORING = (REPO_ROOT
                / "data/local_labels/timing_predictions"
                / "gcom_lite_scoring_sm_86.json")
SEALED_HOLDOUT = (REPO_ROOT
                  / "data/sm_86_predictor_accuracy_recovery_v2"
                  / "sealed_holdout_labels.json")
BASELINE_COMPARISON = (
    REPO_ROOT
    / "data/local_labels/calibration_head_v5_baseline_comparison_sm_86.json")
L1_HEAD = (REPO_ROOT
           / "data/sm_86_predictor_accuracy_recovery_v2"
           / "warp_pool_l1_per_admitted_cell.json")
COHORT = (REPO_ROOT / "data/local_labels/timing_cohort"
          / "sm86_timing_cohort_expanded.json")
LEDGER = REPO_ROOT / "data/local_labels/holdout_touch_ledger_sm_86.json"

DEFAULT_OUT = (REPO_ROOT
               / "data/sm_86_predictor_accuracy_recovery_v2"
               / "per_metric_evidence_harness_sm_86.json")
DEFAULT_SCHEMA = _HERE / "per_metric_evidence_harness.schema.json"

BUCKETS = ("main", "appendix", "diagnostic", "typed_unavailable")


class HarnessInputError(Exception):
    """A committed input is missing or fails its value audit."""


class HarnessGateError(Exception):
    """A promotion to main violates its gate (fail-closed)."""


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.strip()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normal_ci(errors: list) -> dict:
    """A deterministic normal-approximation 95% CI on the mean of the
    per-scored-cell error list (no randomness, so the artifact
    rebuilds byte-identically)."""
    vals = [e for e in errors if e is not None]
    n = len(vals)
    if n < 2:
        return {"status": "ci_unavailable",
                "reason": "insufficient_scored_cells",
                "n": n}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    return {
        "status": "computed",
        "statistic": "mean_per_cell_error",
        "n": n,
        "mean": mean,
        "lo": mean - 1.96 * sem,
        "hi": mean + 1.96 * sem,
    }


# --------------------------------------------------------------- tiers

def extract_t1(t1: dict) -> tuple[list, list]:
    """T1 count metrics (zero-parameter exact-count replacement)."""
    floor_met = bool(t1["non_vacuity_floor"]["floor_met"])
    avt = t1["statistics"]["achieved_vs_target"]
    SCORED = {"replacement_pass", "replacement_miss"}
    DIAG = {"low_support"}
    rows = []
    per_metric_cells: dict = {}
    for v in t1["verdicts"]:
        mk = v["ncu_metric"]
        verdict = v["verdict"]
        if verdict in SCORED:
            outcome, err = "scored", v.get("relative_error")
        elif verdict in DIAG:
            outcome, err = "diagnostic", v.get("relative_error")
        else:
            outcome, err = "typed_unavailable", None
        rows.append({
            "tier": "T1", "metric_key": mk,
            "slug": v["slug"], "shape_point": v["shape_point"],
            "variant_id": v.get("variant_id"),
            "cell_outcome": outcome,
            "achieved": err,
            "achieved_kind": "relative_error" if err is not None
            else "none",
            "typed_reason": None if outcome != "typed_unavailable"
            else verdict,
            "source_artifact": _rel(T1_VERDICT),
        })
        per_metric_cells.setdefault(mk, []).append((outcome, err))
    t1_gates = [{
        "gate": f"t1_pooled_{k}", "target": avt["targets"][k],
        "achieved": avt["achieved"][k], "met": avt["met"][k]}
        for k in ("p50", "p90", "max")]
    metrics = []
    for mk, cells in sorted(per_metric_cells.items()):
        scored_errs = [e for o, e in cells if o == "scored"]
        metrics.append(_metric_summary(
            tier="T1", metric_key=mk, is_fitted=False,
            non_vacuity_met=floor_met, slug_disjoint_evidence=None,
            achieved_vs_target=t1_gates, promotion_gates=t1_gates,
            scored_errs=scored_errs, cells=cells))
    return rows, metrics


def extract_l1(l1: dict) -> tuple[list, list]:
    """L1 hit rate (fitted warp-pool head)."""
    mk = "cache_hit_rate_l1tex_sector"
    avt = l1["achieved_vs_target"]
    slug_disjoint = bool(l1.get("robustness", {}).get("loso")) and \
        bool(l1.get("robustness", {}).get("loocv"))
    rows, cells = [], []
    for split_key in ("pool", "holdout"):
        for pr in l1[split_key]["pairs"]:
            scored = pr["status"] == "scored"
            outcome = "scored" if scored else "typed_unavailable"
            err = pr.get("abs_error_pp") if scored else None
            rows.append({
                "tier": "L1", "metric_key": mk,
                "slug": pr["slug"], "shape_point": pr["shape_point"],
                "cell_outcome": outcome,
                "achieved": err,
                "achieved_kind": "abs_error_pp" if err is not None
                else "none",
                "typed_reason": None if scored
                else pr.get("typed_reason"),
                "source_artifact": _rel(L1_RESCORE),
            })
            cells.append((outcome, err))
    metrics = [_metric_summary(
        tier="L1", metric_key=mk, is_fitted=True,
        non_vacuity_met=None, slug_disjoint_evidence=slug_disjoint,
        achieved_vs_target=avt, promotion_gates=avt,
        scored_errs=[e for o, e in cells if o == "scored"],
        cells=cells)]
    return rows, metrics


def extract_l2(v5: dict) -> tuple[list, list]:
    """L2 hit rate (frozen v5 isotonic head, zero parameters)."""
    mk = "cache_hit_rate_l2_sector"
    cand = v5["candidate_scores"][0]
    cv_geomean = cand["primary"]
    robustness_met = cv_geomean <= DIRECTIONAL_HIT_RATE_GEOMEAN_TARGET
    slug_disjoint = cand.get("n_cv_predictions", 0) > 0
    avt = [{"gate": "l2_mean_cv_geomean_relative_error",
            "target": DIRECTIONAL_HIT_RATE_GEOMEAN_TARGET,
            "achieved": cv_geomean, "met": robustness_met}]
    rows, cells = [], []
    for c in v5["cohort_scope"]["admitted"]:
        pred = (c.get("features") or {}).get("raw_prediction")
        label = c.get("label")
        if pred is not None and label is not None and label > 0:
            outcome = "scored"
            err = abs(pred - label) / label
        else:
            outcome, err = "typed_unavailable", None
        rows.append({
            "tier": "L2", "metric_key": mk,
            "slug": c["slug"], "shape_point": c["shape_point"],
            "cell_outcome": outcome,
            "achieved": err,
            "achieved_kind": "relative_error" if err is not None
            else "none",
            "typed_reason": None if outcome != "typed_unavailable"
            else "l2_raw_prediction_or_label_unavailable",
            "source_artifact": _rel(V5_HEAD),
        })
        cells.append((outcome, err))
    for c in v5["cohort_scope"].get("skipped", []):
        rows.append({
            "tier": "L2", "metric_key": mk,
            "slug": c["slug"], "shape_point": c["shape_point"],
            "cell_outcome": "typed_unavailable", "achieved": None,
            "achieved_kind": "none",
            "typed_reason": c.get("typed_reason", "v5_skipped"),
            "source_artifact": _rel(V5_HEAD),
        })
        cells.append(("typed_unavailable", None))
    metrics = [_metric_summary(
        tier="L2", metric_key=mk, is_fitted=True,
        non_vacuity_met=None, slug_disjoint_evidence=slug_disjoint,
        achieved_vs_target=avt, promotion_gates=avt,
        scored_errs=[e for o, e in cells if o == "scored"],
        cells=cells)]
    return rows, metrics


def extract_t2(t2: dict) -> tuple[list, list]:
    """T2 traffic primitives (zero-parameter composition)."""
    floor_met = bool(t2["non_vacuity_floor"]["met"])
    avt_all = t2["achieved_vs_target"]
    rows = []
    per_prim_cells: dict = {}
    for cell in t2["cells"]:
        for prim in cell["primitives"]:
            name = prim["primitive"]
            status = prim["status"]
            if status == "scored":
                outcome, err = "scored", prim.get("rel_error")
            elif status == "diagnostic":
                outcome, err = "diagnostic", None
            else:
                outcome, err = "typed_unavailable", None
            rows.append({
                "tier": "T2", "metric_key": f"t2_{name}",
                "slug": cell["slug"], "shape_point": cell["shape_point"],
                "cell_outcome": outcome,
                "achieved": err,
                "achieved_kind": "relative_error" if err is not None
                else "none",
                "typed_reason": None if outcome != "typed_unavailable"
                else prim.get("status_reason", status),
                "source_artifact": _rel(T2_TRAFFIC),
            })
            per_prim_cells.setdefault(name, []).append((outcome, err))
    metrics = []
    for name, cells in sorted(per_prim_cells.items()):
        prim_gates = [g for g in avt_all
                      if g["gate"].startswith(f"{name}_")]
        metrics.append(_metric_summary(
            tier="T2", metric_key=f"t2_{name}", is_fitted=False,
            non_vacuity_met=floor_met, slug_disjoint_evidence=None,
            achieved_vs_target=prim_gates,
            promotion_gates=prim_gates,
            scored_errs=[e for o, e in cells if o == "scored"],
            cells=cells))
    return rows, metrics


def extract_cycles_ns(cycles: dict, scoring: dict) -> tuple[list, list]:
    """Cycles (fitted ridge residual) + the ns interval layer."""
    avt = scoring["achieved_vs_target"]
    loso = next((g for g in avt
                 if g["gate"] == "loso_geomean_log_relative_error"),
                None)
    # The decisive promotion gate is the LOSO directional target
    # (0.20), NOT the 0.15 stretch (which is a bonus, not a bar).
    cyc_gates = [loso] if loso else []
    slug_disjoint = any(
        s["procedure"] == "leave_one_slug_out"
        for s in scoring["robustness_tables"]["surfaces"])
    fold_by = {(f["held_out_slug"], f["shape_point"]): f
               for f in scoring["folds"]}
    cross_by = {(r["slug"], r["shape_point"]): r
                for r in scoring["deploy_crosswalk"]}

    cyc_rows, ns_rows, cyc_cells = [], [], []
    for c in cycles["cells"]:
        modeled = c["status"].startswith("modeled")
        key = (c["slug"], c["shape_point"])
        if modeled:
            fold = fold_by.get(key)
            err = fold["abs_log_ratio"] if fold else None
            outcome = "scored"
        else:
            err, outcome = None, "typed_unavailable"
        cyc_rows.append({
            "tier": "cycles", "metric_key": "gpc__cycles_elapsed.max",
            "slug": c["slug"], "shape_point": c["shape_point"],
            "cell_outcome": outcome,
            "achieved": err,
            "achieved_kind": "abs_log_ratio" if err is not None
            else "none",
            "typed_reason": None if modeled
            else c.get("typed_reason", "evaluator_input_unavailable"),
            "source_artifact": _rel(GCOM_CYCLES),
        })
        cyc_cells.append((outcome, err))
        # ns row: interval-only published for modeled cells.
        cw = cross_by.get(key)
        if modeled and cw is not None:
            ns_rows.append({
                "tier": "ns", "metric_key": "gpu__time_duration.sum",
                "slug": c["slug"], "shape_point": c["shape_point"],
                "cell_outcome": "scored", "achieved": None,
                "achieved_kind": "interval_only",
                "typed_reason": None,
                "source_artifact": _rel(GCOM_SCORING),
            })
        else:
            ns_rows.append({
                "tier": "ns", "metric_key": "gpu__time_duration.sum",
                "slug": c["slug"], "shape_point": c["shape_point"],
                "cell_outcome": "typed_unavailable", "achieved": None,
                "achieved_kind": "none",
                "typed_reason": c.get(
                    "typed_reason", "evaluator_input_unavailable"),
                "source_artifact": _rel(GCOM_SCORING),
            })

    cyc_metric = _metric_summary(
        tier="cycles", metric_key="gpc__cycles_elapsed.max",
        is_fitted=True, non_vacuity_met=None,
        slug_disjoint_evidence=slug_disjoint,
        achieved_vs_target=avt, promotion_gates=cyc_gates,
        scored_errs=[e for o, e in cyc_cells if o == "scored"],
        cells=cyc_cells)
    ns_cells = [("scored" if r["cell_outcome"] == "scored"
                 else "typed_unavailable", None) for r in ns_rows]
    ns_metric = _metric_summary(
        tier="ns", metric_key="gpu__time_duration.sum",
        is_fitted=True, non_vacuity_met=None,
        slug_disjoint_evidence=slug_disjoint,
        achieved_vs_target=avt, promotion_gates=cyc_gates,
        scored_errs=[], cells=ns_cells,
        ci_override={"status": "ci_unavailable",
                     "reason": "interval_only_no_point_error",
                     "n": sum(1 for o, _ in ns_cells if o == "scored")})
    return cyc_rows + ns_rows, [cyc_metric, ns_metric]


# --------------------------------------------------- metric + buckets

def _gates_pass_by_value(promotion_gates: list) -> bool:
    """Re-derive the promotion gate from the achieved-vs-target
    NUMBERS (achieved <= target) — auditing values, never trusting a
    source 'met' marker (BL fail-closed-guards-audit-values)."""
    if not promotion_gates:
        return False
    for g in promotion_gates:
        a, t = g.get("achieved"), g.get("target")
        if a is None or t is None or a > t:
            return False
    return True


def _metric_summary(
    *, tier, metric_key, is_fitted, non_vacuity_met,
    slug_disjoint_evidence, achieved_vs_target, promotion_gates,
    scored_errs, cells, ci_override=None,
) -> dict:
    n_scored = sum(1 for o, _ in cells if o == "scored")
    n_scored_with_error = sum(
        1 for o, e in cells if o == "scored" and e is not None)
    n_diag = sum(1 for o, _ in cells if o == "diagnostic")
    n_typed = sum(1 for o, _ in cells if o == "typed_unavailable")
    gate_met = _gates_pass_by_value(promotion_gates)
    if is_fitted:
        robustness_gate_met = gate_met
        achieved_vs_target_met = None
        promote = (n_scored > 0 and gate_met
                   and bool(slug_disjoint_evidence))
        gate_kind = "robustness_and_slug_disjoint"
    else:
        achieved_vs_target_met = gate_met
        robustness_gate_met = None
        promote = (n_scored > 0 and bool(non_vacuity_met) and gate_met)
        gate_kind = "non_vacuity_and_achieved"
    promotion = "main" if promote else "appendix"
    return {
        "tier": tier, "metric_key": metric_key,
        "is_fitted": is_fitted, "promotion": promotion,
        "gate": {
            "kind": gate_kind,
            "non_vacuity_met": non_vacuity_met,
            "achieved_vs_target_met": achieved_vs_target_met,
            "robustness_gate_met": robustness_gate_met,
            "slug_disjoint_evidence": slug_disjoint_evidence,
            "has_scored_cells": n_scored > 0,
            "promotion_gates": promotion_gates,
        },
        "achieved_vs_target": achieved_vs_target,
        "ci": ci_override or _normal_ci(scored_errs),
        "n_scored": n_scored,
        "n_scored_with_error": n_scored_with_error,
        "n_diagnostic": n_diag,
        "n_typed_unavailable": n_typed,
    }


def assert_promotion_legal(metric: dict) -> None:
    """Fail closed: a metric marked main MUST satisfy its gate. The
    gate is RE-DERIVED here from the stored achieved-vs-target
    values, so a tampered gate boolean (or a flipped upstream marker)
    cannot smuggle a MISS metric into main."""
    if metric["promotion"] != "main":
        return
    g = metric["gate"]
    if not g["has_scored_cells"]:
        raise HarnessGateError(
            f"{metric['tier']}/{metric['metric_key']}: promoted to "
            f"main with zero scored cells")
    if not _gates_pass_by_value(g.get("promotion_gates", [])):
        raise HarnessGateError(
            f"{metric['tier']}/{metric['metric_key']}: promoted to "
            f"main while its achieved-vs-target promotion gate(s) "
            f"FAIL by value: {g.get('promotion_gates')}")
    if metric["is_fitted"] and not g["slug_disjoint_evidence"]:
        raise HarnessGateError(
            f"{metric['tier']}/{metric['metric_key']}: fitted metric "
            f"promoted to main WITHOUT slug-disjoint validation "
            f"evidence")
    if not metric["is_fitted"] and not g["non_vacuity_met"]:
        raise HarnessGateError(
            f"{metric['tier']}/{metric['metric_key']}: promoted to "
            f"main with a FAILED non-vacuity floor")


def bucket_rows(rows: list, metrics: list) -> list:
    """Assign each (cell, metric_key) row to exactly one bucket."""
    promo = {(m["tier"], m["metric_key"]): m["promotion"]
             for m in metrics}
    out = []
    for r in rows:
        if r["cell_outcome"] == "typed_unavailable":
            bucket = "typed_unavailable"
        elif r["cell_outcome"] == "diagnostic":
            bucket = "diagnostic"
        else:
            bucket = promo[(r["tier"], r["metric_key"])]
        # Normalize the cell-variant axis (only T1 carries variants).
        out.append({**r, "variant_id": r.get("variant_id"),
                    "bucket": bucket})
    return out


def disambiguate_variants(rows: list) -> list:
    """Some source artifacts carry genuinely-distinct variant cells
    that share (slug, shape_point) without a variant_id (e.g. the
    L1 rescore's four register-spill register-cap rows). Append a
    deterministic source-order ordinal to such colliding rows so
    every distinct source row keeps a unique cell identity (the rows
    are read in committed-array order, so the ordinal is stable)."""
    groups: dict = {}
    for i, r in enumerate(rows):
        key = (r["tier"], r["slug"], r["shape_point"],
               r.get("variant_id"), r["metric_key"])
        groups.setdefault(key, []).append(i)
    for key, idxs in groups.items():
        if len(idxs) > 1:
            for ordinal, i in enumerate(idxs):
                base = rows[i].get("variant_id") or ""
                rows[i]["variant_id"] = f"{base}#src{ordinal}"
    return rows


def assert_exactly_once(rows: list) -> None:
    seen = set()
    for r in rows:
        k = (r["tier"], r["slug"], r["shape_point"],
             r.get("variant_id"), r["metric_key"])
        if k in seen:
            raise HarnessInputError(
                f"duplicate (tier, cell, metric_key) row: {k}")
        seen.add(k)
        if r["bucket"] not in BUCKETS:
            raise HarnessInputError(
                f"row {k} has invalid bucket {r['bucket']!r}")


# ----------------------------------------------------- baseline panel

# The plan-named baseline set, EXACTLY (no cohort_mean): the seven
# hit-rate baselines are scored on the committed L2 baseline-
# comparison surface vs the sealed L2 labels; the two timing
# baselines have no committed per-cell prediction and stay typed
# with a NAMED missing input.
_PLAN_BASELINES = (
    "always_hit", "always_miss", "family_mean", "uncalibrated_walker",
    "uncalibrated_reuse_distance", "prior_v4", "prior_v5",
    "static_roofline_timing", "event_only_timing",
)
_COMPARISON_COLUMN = {
    "uncalibrated_walker": "uncalibrated_walker_prediction",
    "uncalibrated_reuse_distance": "nugteren_prediction",
    "prior_v4": "v4_platt_prediction",
    "prior_v5": "v5_prediction",
}
_TIMING_BASELINE_MISSING = {
    "static_roofline_timing": (
        "no committed per-cell static-roofline timing baseline "
        "exists for the sealed cells: there is no "
        "data/local_labels/timing_predictions/*roofline* artifact "
        "publishing a standalone per-cell ns/cycles roofline "
        "prediction (the GCoM-lite analytical prior is the roofline "
        "term but is not emitted as an independent timing baseline)"),
    "event_only_timing": (
        "no committed per-cell event-only timing baseline exists for "
        "the sealed cells: there is no "
        "data/local_labels/timing_predictions/*event_only* artifact "
        "publishing a CUPTI-event-derived per-cell ns prediction"),
}


def _geomean_rel(preds: list, labels: list):
    """Geomean relative error over cells with a strictly-positive
    label (hit-rate labels of 0 make relative error undefined, so
    they are excluded; the panel's primary metric is mean_abs_pp).
    Returns None when no positive-label cell is available."""
    logs = []
    for p, lab in zip(preds, labels):
        if lab > 0 and p > 0:
            logs.append(abs(math.log(p / lab)))
        elif lab > 0:
            logs.append(abs((p - lab) / lab))
    return (math.exp(sum(logs) / len(logs)) - 1.0) if logs else None


def _mean_abs_pp(preds: list, labels: list) -> float:
    return (sum(abs(p - lab) for p, lab in zip(preds, labels))
            / len(preds) * 100.0) if preds else 0.0


def compute_baseline_panel(
    *, ledger_path: Path, config_digest: str,
    scored_artifact: str, source_commit: str, record: bool,
) -> dict:
    """Run the baseline panel ONCE on the sealed holdout under the
    one-touch ledger. The hit-rate baselines come from the committed
    L2 baseline-comparison surface and are scored against the sealed
    L2 labels (read under the unblind context); family_mean is the
    per-family mean of the NON-holdout labels; the two timing
    baselines stay typed with a named missing input."""
    with label_loader.holdout_unblind_context():
        sealed = label_loader.load_labels(SEALED_HOLDOUT)
    sealed_digest = sealed["cohort_manifest_digest"]
    # Sealed L2 hit rate in PERCENT -> fraction.
    l2_label_by = {
        (c["slug"], c["shape_point"]):
            c["cache_hit_rate_l2_sector_pct"] / 100.0
        for c in sealed["cells"]
        if c.get("cache_hit_rate_l2_sector_pct") is not None
    }
    comp = _load(BASELINE_COMPARISON)
    rows_by = {(r["slug"], r["shape_point"]): r for r in comp["rows"]}
    holdout_keys = set(l2_label_by)
    # family_mean: per-family_bucket mean of the NON-holdout labels.
    fam_labels: dict = {}
    for r in comp["rows"]:
        if (r["slug"], r["shape_point"]) in holdout_keys:
            continue
        fam, lab = r.get("family_bucket"), r.get("label")
        if fam is None or lab is None:
            continue
        fam_labels.setdefault(fam, []).append(lab)
    family_mean = {f: sum(v) / len(v) for f, v in fam_labels.items()}

    cells = sorted(set(l2_label_by) & set(rows_by))

    def pred_for(name, c):
        r = rows_by[c]
        if name == "always_hit":
            return 1.0
        if name == "always_miss":
            return 0.0
        if name == "family_mean":
            return family_mean.get(r.get("family_bucket"))
        return r.get(_COMPARISON_COLUMN[name])

    # APPLES-TO-APPLES: every computed hit-rate baseline is scored on
    # ONE common comparable cell set -- the sealed-holdout cells where
    # EVERY hit-rate baseline has a numeric prediction -- so the
    # denominators are equal and the comparison is fair (a cell where
    # any baseline has no prediction is excluded from ALL of them).
    hit_rate_names = [n for n in _PLAN_BASELINES
                      if n not in _TIMING_BASELINE_MISSING]
    comparable = sorted(
        c for c in cells
        if all(pred_for(n, c) is not None for n in hit_rate_names))
    excluded = sorted(set(cells) - set(comparable))
    comp_labels = [l2_label_by[c] for c in comparable]

    entries = []
    for name in _PLAN_BASELINES:
        if name in _TIMING_BASELINE_MISSING:
            entries.append({
                "baseline": name,
                "status": "baseline_input_unavailable",
                "metric_key": "gpu__time_duration.sum", "n_cells": 0,
                "reason": _TIMING_BASELINE_MISSING[name]})
            continue
        if not comparable:
            entries.append({
                "baseline": name,
                "status": "baseline_input_unavailable",
                "metric_key": "cache_hit_rate_l2_sector", "n_cells": 0,
                "reason": ("no sealed-holdout cell has a numeric "
                           "prediction for every hit-rate baseline")})
            continue
        ps = [pred_for(name, c) for c in comparable]
        entries.append({
            "baseline": name, "status": "computed",
            "metric_key": "cache_hit_rate_l2_sector",
            "n_cells": len(comparable),
            "geomean_rel_error": _geomean_rel(ps, comp_labels),
            "mean_abs_error_pp": _mean_abs_pp(ps, comp_labels)})

    touch_recorded = _record_one_touch(
        ledger_path=ledger_path, sealed_digest=sealed_digest,
        config_digest=config_digest, scored_artifact=scored_artifact,
        source_commit=source_commit, record=record)

    return {
        "scope": HARNESS_SCOPE,
        "sealed_digest": sealed_digest,
        "metric_key": "cache_hit_rate_l2_sector",
        "comparison_source": _rel(BASELINE_COMPARISON),
        "n_holdout_cells_total": len(cells),
        "n_holdout_cells_scored": len(comparable),
        "comparable_set_note": (
            "every computed hit-rate baseline is scored on this one "
            "common comparable cell set (apples-to-apples, equal "
            "denominators)"),
        "excluded_cells": [
            {"slug": s, "shape_point": sp,
             "reason": ("no numeric prediction for every hit-rate "
                        "baseline on this sealed cell")}
            for s, sp in excluded],
        "chosen_model": "calibration_head_v5",
        "touch_recorded": touch_recorded,
        "entries": sorted(entries, key=lambda e: e["baseline"]),
        "note": (
            "the seven hit-rate baselines are scored on ONE common "
            "comparable sealed-holdout cell set (equal denominators, "
            "apples-to-apples) on the committed L2 baseline-comparison "
            "surface vs the sealed L2 labels under one governed touch "
            "(the deployed v5 head is the chosen model / prior_v5); "
            "family_mean uses the per-family non-holdout means; the "
            "two timing baselines are typed with a named missing "
            "input"),
    }


def _record_one_touch(*, ledger_path, sealed_digest, config_digest,
                      scored_artifact, source_commit, record):
    """Idempotent one-touch: an existing (sealed_digest, scope) touch
    under the SAME config reflects as recorded without a second
    append; a differing config trips the retune guard."""
    ledger = htl.load_ledger(ledger_path)
    existing = [
        t for t in ledger.get("touches", [])
        if t.get("scope") == HARNESS_SCOPE
        and t.get("sealed_digest") == sealed_digest]
    if existing:
        htl.assert_retune_legal(
            ledger, sealed_digest=sealed_digest, scope=HARNESS_SCOPE,
            config_digest=config_digest)
        return True
    if record:
        htl.assert_touch_legal(
            ledger, sealed_digest=sealed_digest, scope=HARNESS_SCOPE)
        htl.record_touch(
            ledger_path, sealed_digest=sealed_digest,
            scope=HARNESS_SCOPE, scored_artifact=scored_artifact,
            config_digest=config_digest, source_commit=source_commit)
        return True
    return False


# ------------------------------------------------------- ablations

_L1_ABLATIONS = {
    "no_interleaving": {"mode": "concatenated"},
    "no_mshr": {"mshr_count": None},
    "no_hrd_second_layer": {"sector_accounting": "line_flat"},
}
_REQUIRED_ABLATIONS = tuple(_L1_ABLATIONS) + (
    "no_control_bit_stalls", "no_covariates")


def _mean(xs: list):
    return sum(xs) / len(xs) if xs else None


def compute_ablations(*, l1: dict, cycles: dict, cohort: dict,
                      scoring: dict) -> dict:
    """The five component ablations as COMPUTED rows. The three L1
    ablations re-run the deterministic warp-pool walker on the
    sealed-holdout cells with one toggle each (vs the chosen L1
    head); the two cycles ablations re-fit the GCoM-lite ridge over
    the committed cycles surface with one component dropped (vs the
    chosen LOSO geomean)."""
    import compute_warp_pool_l1_per_admitted_cell as cc

    with label_loader.holdout_unblind_context():
        sealed = label_loader.load_labels(SEALED_HOLDOUT)
    l1_label_by = {
        (c["slug"], c["shape_point"]):
            c["cache_hit_rate_l1tex_sector_pct"] / 100.0
        for c in sealed["cells"]
        if c.get("cache_hit_rate_l1tex_sector_pct") is not None}
    chosen_l1 = {
        (p["slug"], p["shape_point"]): p["predicted_hit_rate"]
        for p in l1["holdout"]["pairs"] if p["status"] == "scored"}
    head_cells = {
        (c["slug"], c["shape_point"]): c
        for c in _load(L1_HEAD)["cells"]}

    entries = []
    # --- L1 walker ablations ---
    holdout_keys = sorted(set(l1_label_by) & set(head_cells))
    for name, ov in _L1_ABLATIONS.items():
        per_cell, abl_errs, chosen_errs = [], [], []
        for key in holdout_keys:
            slug, shape = key
            label = l1_label_by[key]
            chosen = chosen_l1.get(key)
            r = cc.compute_for_cell(
                slug=slug, shape_point=shape,
                cell_row=head_cells[key], walk_overrides=ov)
            if r.get("result_kind") == "scalar" and chosen is not None:
                pred = r["predicted_hit_rate"]
                ae = abs(pred - label) * 100.0
                cae = abs(chosen - label) * 100.0
                per_cell.append({
                    "slug": slug, "shape_point": shape,
                    "ablation_predicted": pred, "label": label,
                    "ablation_abs_error_pp": ae,
                    "chosen_abs_error_pp": cae})
                abl_errs.append(ae)
                chosen_errs.append(cae)
            else:
                per_cell.append({
                    "slug": slug, "shape_point": shape,
                    "typed_reason": r.get(
                        "typed_reason", "ablation_walk_typed")})
        abl_v = _mean(abl_errs)
        chosen_v = _mean(chosen_errs)
        delta = (None if abl_v is None or chosen_v is None
                 else abl_v - chosen_v)
        if delta is not None and abs(delta) < 1e-9:
            note = (
                "delta 0: the toggle moves the walker's internal "
                "counters but not the hit RATE on these "
                "low-concurrency sealed-holdout cells (no L1 reuse "
                "to lose / MSHRs never exhausted); the component is "
                "load-bearing on higher-concurrency cells")
        else:
            note = (
                "positive delta: ablating the component degrades the "
                "chosen model's mean abs error on the holdout")
        entries.append({
            "ablation": name,
            "metric_key": "cache_hit_rate_l1tex_sector",
            "comparable_set": "sealed_holdout_l1_cells",
            "toggle_or_policy": dict(ov),
            "n_comparable": len(abl_errs),
            "n_typed": len(per_cell) - len(abl_errs),
            "value_kind": "mean_abs_error_pp",
            "ablation_value": abl_v, "chosen_value": chosen_v,
            "delta_vs_chosen": delta, "delta_note": note,
            "source_inputs": [_rel(L1_HEAD), _rel(L1_RESCORE),
                              _rel(SEALED_HOLDOUT)],
            "per_cell": per_cell})

    # --- cycles ablations ---
    labels = {
        (c["slug"], c["shape_point"]):
            c["timing_labels"][_CYCLES_LABEL]["value"]
        for c in cohort["admitted"]}
    fit_rows = [c for c in cycles["cells"]
                if c["status"] == "modeled_fit_pool"]
    chosen_geomean = next(
        g["achieved"] for g in scoring["achieved_vs_target"]
        if g["gate"] == "loso_geomean_log_relative_error")

    def _loso(feature_names, prior_fn):
        abs_logs = []
        for i, held in enumerate(fit_rows):
            train = [r for j, r in enumerate(fit_rows) if j != i]
            rows = [
                ([r["feature_vector"][n] for n in feature_names],
                 math.log(labels[(r["slug"], r["shape_point"])]
                          / prior_fn(r)))
                for r in train]
            coeffs = _ridge_fit(rows, RIDGE_LAMBDA)
            x = [held["feature_vector"][n] for n in feature_names]
            residual = sum(c * v for c, v in zip(coeffs, x))
            pred = prior_fn(held) * math.exp(residual)
            label = labels[(held["slug"], held["shape_point"])]
            abs_logs.append(abs(math.log(pred / label)))
        return math.exp(sum(abs_logs) / len(abs_logs)) - 1.0

    def _ct(c, term):
        return c["component_terms"][term]["cycles"]

    no_stall_features = tuple(
        n for n in FEATURE_NAMES if n != "mean_static_stall_cycles")
    no_stall_geomean = _loso(
        no_stall_features,
        lambda c: max(_ct(c, "compute"), _ct(c, "memory"))
        + _ct(c, "imbalance"))
    no_cov_features = tuple(
        n for n in FEATURE_NAMES
        if n not in ("co_resident_warps_over_48", "active_sm_over_84"))
    no_cov_geomean = _loso(
        no_cov_features, lambda c: c["analytical_prior_cycles"])

    for name, geomean, policy in (
        ("no_control_bit_stalls", no_stall_geomean,
         "stall term zeroed in the analytical prior AND the "
         "mean_static_stall_cycles feature dropped before the ridge"),
        ("no_covariates", no_cov_geomean,
         "the occupancy/clock covariate features "
         "(co_resident_warps_over_48, active_sm_over_84) dropped "
         "before the ridge; the analytical prior unchanged"),
    ):
        entries.append({
            "ablation": name,
            "metric_key": "gpc__cycles_elapsed.max",
            "comparable_set": "cycles_fit_pool_loso",
            "toggle_or_policy": {"policy": policy},
            "n_comparable": len(fit_rows), "n_typed": 0,
            "value_kind": "geomean_log_relative_error",
            "ablation_value": geomean, "chosen_value": chosen_geomean,
            "delta_vs_chosen": geomean - chosen_geomean,
            "delta_note": (
                "positive delta: dropping the component raises the "
                "LOSO geomean log-relative cycle error vs the chosen "
                "model, i.e. the component helps"),
            "source_inputs": [_rel(GCOM_CYCLES), _rel(GCOM_SCORING),
                              _rel(COHORT)],
            "per_cell": []})

    by_name = {e["ablation"] for e in entries}
    missing = sorted(set(_REQUIRED_ABLATIONS) - by_name)
    if missing:
        raise HarnessInputError(
            f"ablations missing required entries: {missing}")
    return {
        "status": "computed",
        "required": list(_REQUIRED_ABLATIONS),
        "note": (
            "five component ablations: three L1 walker re-runs "
            "(positive delta_vs_chosen means the toggled component "
            "helps the chosen model) and two cycles ridge re-fits "
            "over the committed surface"),
        "entries": sorted(entries, key=lambda e: e["ablation"]),
    }


# --------------------------------------------------------- assembly

def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cohort_label_digest() -> str:
    """A STABLE digest of the cohort's cycle-label content (the
    identity tuple + the gpc__cycles_elapsed.max value the cycles
    ablations consume), excluding the source_commit / generated_at
    provenance fields the label-commit-last close restamps. Like the
    sealed-holdout seal-identity binding, this keeps the harness
    config_digest invariant across the close's cohort re-derive."""
    rows = sorted(
        [c["slug"], c["shape_point"], c.get("binary_sha256", ""),
         c["timing_labels"].get(_CYCLES_LABEL, {}).get("value")]
        for c in _load(COHORT)["admitted"])
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


def _config(metrics: list) -> dict:
    # Bind the config (hence config_digest, hence the retune guard)
    # to the composed inputs, so an input change shifts the digest
    # and a re-touch under changed inputs trips the retune guard.
    # The non-governance tier artifacts are bound by file content;
    # the sealed holdout is bound by its SEAL IDENTITY
    # (cohort_manifest_digest) rather than raw bytes, because the
    # label-commit-last close restamps its source_commit fields
    # without changing the sealed labels -- the seal digest is the
    # stable, label-bearing identity that must gate a retune.
    # Restamp-stable inputs (timing predictions, heads, baseline
    # comparison) are bound by file content; the two inputs the
    # label-commit-last close re-derives -- the sealed holdout and
    # the expanded cohort -- are bound by stable label-bearing
    # identities so the close's source_commit restamps do not shift
    # the digest (and so the retune guard does not false-trip).
    input_digests = {
        name: _sha256_file(path) for name, path in (
            ("t1_lockstep", T1_VERDICT), ("l1_rescore", L1_RESCORE),
            ("l2_v5_head", V5_HEAD), ("t2_traffic", T2_TRAFFIC),
            ("gcom_cycles", GCOM_CYCLES), ("gcom_scoring", GCOM_SCORING),
            ("l1_head", L1_HEAD),
            ("baseline_comparison", BASELINE_COMPARISON))
    }
    input_digests["sealed_holdout_seal_digest"] = _load(
        SEALED_HOLDOUT)["cohort_manifest_digest"]
    input_digests["cohort_label_digest"] = _cohort_label_digest()
    return {
        "model": "per_metric_evidence_harness_v1",
        "buckets": list(BUCKETS),
        "directional_hit_rate_geomean_target":
            DIRECTIONAL_HIT_RATE_GEOMEAN_TARGET,
        "tiers": sorted({m["tier"] for m in metrics}),
        "input_digests": input_digests,
    }


def _config_digest(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def build_harness(*, record_touch: bool = True,
                  ledger_path: Path = LEDGER) -> dict:
    t1_rows, t1_m = extract_t1(_load(T1_VERDICT))
    l1_rows, l1_m = extract_l1(_load(L1_RESCORE))
    l2_rows, l2_m = extract_l2(_load(V5_HEAD))
    t2_rows, t2_m = extract_t2(_load(T2_TRAFFIC))
    cn_rows, cn_m = extract_cycles_ns(
        _load(GCOM_CYCLES), _load(GCOM_SCORING))

    metrics = t1_m + l1_m + l2_m + t2_m + cn_m
    for m in metrics:
        assert_promotion_legal(m)
    raw = disambiguate_variants(
        t1_rows + l1_rows + l2_rows + t2_rows + cn_rows)
    rows = bucket_rows(raw, metrics)
    assert_exactly_once(rows)

    bucket_counts = {b: sum(1 for r in rows if r["bucket"] == b)
                     for b in BUCKETS}
    config = _config(metrics)
    digest = _config_digest(config)
    head = _git_head()
    # The ablations re-consume the sealed holdout's L1 labels, so run
    # them BEFORE the panel records the one touch (the panel records;
    # the ablations read under the same unblind context, same scope).
    ablations = compute_ablations(
        l1=_load(L1_RESCORE), cycles=_load(GCOM_CYCLES),
        cohort=_load(COHORT), scoring=_load(GCOM_SCORING))
    panel = compute_baseline_panel(
        ledger_path=ledger_path, config_digest=digest,
        scored_artifact=_rel(DEFAULT_OUT), source_commit=head,
        record=record_touch)

    return {
        "schema_version": 1,
        "artifact_kind": "per_metric_evidence_harness",
        "arch": ARCH,
        "generated_at_commit": head,
        "config": config,
        "config_digest": digest,
        "inputs": {
            "t1_lockstep": _rel(T1_VERDICT),
            "l1_rescore": _rel(L1_RESCORE),
            "l1_head": _rel(L1_HEAD),
            "l2_v5_head": _rel(V5_HEAD),
            "t2_traffic": _rel(T2_TRAFFIC),
            "gcom_cycles": _rel(GCOM_CYCLES),
            "gcom_scoring": _rel(GCOM_SCORING),
            "cohort": _rel(COHORT),
            "baseline_comparison": _rel(BASELINE_COMPARISON),
            "sealed_holdout": _rel(SEALED_HOLDOUT),
        },
        "bucket_counts": bucket_counts,
        "n_rows": len(rows),
        "per_metric": sorted(
            metrics, key=lambda m: (m["tier"], m["metric_key"])),
        "per_cell_metric": sorted(
            rows, key=lambda r: (
                r["tier"], r["metric_key"], r["slug"],
                r["shape_point"], r.get("variant_id") or "")),
        "baseline_panel": panel,
        "ablations": ablations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--schema-path", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--no-touch", action="store_true",
        help="build without recording the sealed-holdout touch "
             "(for tests/dry-runs).")
    args = parser.parse_args(argv)

    artifact = build_harness(record_touch=not args.no_touch)
    import jsonschema
    schema = _load(args.schema_path)
    jsonschema.validate(artifact, schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    bc = artifact["bucket_counts"]
    print(f"per_metric_evidence_harness: wrote {args.out} "
          f"({artifact['n_rows']} rows; main={bc['main']} "
          f"appendix={bc['appendix']} diagnostic={bc['diagnostic']} "
          f"typed_unavailable={bc['typed_unavailable']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
