"""Accuracy gate for the calibrated cache-replay predictor on the 30-cell cohort.

The gate enumerates four unweighted criteria against the sm_120 NCU labels:
  * full-cohort median absolute hit-rate error <= 0.055;
  * full-cohort geomean relative error          <= 0.12;
  * holdout geomean relative error              <= 0.25 (the 7 holdout cells);
  * per family-bucket p90 absolute error        <= 0.15 for every bucket N >= 7.

The gate is a pure scanner over already-computed metrics, so it carries positive
and negative fixtures: a predictor at median absolute error 0.10 fails the
median criterion, and a bucket (N=8, p90=0.20) fails the per-bucket criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

FULL_COHORT_MEDIAN_ABS_MAX = 0.055
FULL_COHORT_GEOMEAN_REL_MAX = 0.12
HOLDOUT_GEOMEAN_REL_MAX = 0.25
PER_BUCKET_P90_ABS_MAX = 0.15
PER_BUCKET_MIN_N = 7


@dataclass
class GateCriterion:
    name: str
    value: float
    threshold: float
    passed: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
        }


def evaluate_accuracy_gate(
    *,
    full_cohort_median_abs: float,
    full_cohort_geomean_rel: float,
    holdout_geomean_rel: float,
    per_bucket_p90: dict[str, dict],
) -> dict:
    """Evaluate the four-criterion gate. ``per_bucket_p90`` maps bucket -> a dict
    carrying ``n`` and ``p90_absolute_error``. Returns a verdict dict with each
    criterion and an overall pass flag (every criterion must pass)."""
    criteria: list[GateCriterion] = [
        GateCriterion(
            "full_cohort_median_absolute_error",
            full_cohort_median_abs, FULL_COHORT_MEDIAN_ABS_MAX,
            full_cohort_median_abs <= FULL_COHORT_MEDIAN_ABS_MAX,
        ),
        GateCriterion(
            "full_cohort_geomean_relative_error",
            full_cohort_geomean_rel, FULL_COHORT_GEOMEAN_REL_MAX,
            full_cohort_geomean_rel <= FULL_COHORT_GEOMEAN_REL_MAX,
        ),
        GateCriterion(
            "holdout_geomean_relative_error",
            holdout_geomean_rel, HOLDOUT_GEOMEAN_REL_MAX,
            holdout_geomean_rel <= HOLDOUT_GEOMEAN_REL_MAX,
        ),
    ]
    per_bucket_results = []
    per_bucket_pass = True
    for bucket, stats in sorted(per_bucket_p90.items()):
        n = stats.get("n", 0)
        if n < PER_BUCKET_MIN_N:
            per_bucket_results.append({
                "bucket": bucket, "n": n, "p90_absolute_error": stats.get("p90_absolute_error"),
                "evaluated": False, "passed": True,
            })
            continue
        p90 = stats["p90_absolute_error"]
        ok = p90 <= PER_BUCKET_P90_ABS_MAX
        per_bucket_pass = per_bucket_pass and ok
        per_bucket_results.append({
            "bucket": bucket, "n": n, "p90_absolute_error": p90,
            "threshold": PER_BUCKET_P90_ABS_MAX, "evaluated": True, "passed": ok,
        })

    scalar_pass = all(c.passed for c in criteria)
    overall = scalar_pass and per_bucket_pass
    return {
        "criteria": [c.as_dict() for c in criteria],
        "per_bucket_p90": per_bucket_results,
        "per_bucket_passed": per_bucket_pass,
        "passed": overall,
    }


def evaluate_bundle(bundle: dict, *, predictor_column: str = "replay_calibrated_prediction") -> dict:
    """Evaluate the gate against a comparison bundle for the given predictor
    column. Uses the bundle's aggregate metrics, holdout geomean, and per-bucket
    p90 block."""
    agg = bundle["aggregate_metrics"][predictor_column]
    per_bucket = bundle.get("per_family_bucket_p90_calibrated", {})
    return evaluate_accuracy_gate(
        full_cohort_median_abs=agg["median_absolute_error"],
        full_cohort_geomean_rel=agg["geomean_relative_error"],
        holdout_geomean_rel=bundle.get("holdout_geomean", 1.0),
        per_bucket_p90=per_bucket,
    )
