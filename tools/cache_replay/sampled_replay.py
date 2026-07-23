"""Sampled replay mode + sampled-vs-full fidelity report.

Set sampling exploits the per-set independence of a set-associative cache: a
sampled subset of sets (1/modulus, modulus in {32, 64, 128}) is replayed EXACTLY,
and the hit ratio over the sampled accesses is an unbiased estimator of the
full-cache hit ratio, with a Wilson score interval and a bootstrap interval for
the extrapolation uncertainty. The fidelity report checks that a sampled mode
processing no more than ten percent of sector records differs from full replay by
at most 0.02 absolute hit ratio on at least 27 of the 30 cells AND preserves the
full-cohort accuracy verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # noqa: E402
import cohort as cohort_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import artifact_io  # noqa: E402
import accuracy_gate  # noqa: E402
import residual as residual_mod  # noqa: E402
import calibrate as calibrate_mod  # noqa: E402

_AC4_CRITERIA = (
    "full_cohort_median_absolute_error",
    "full_cohort_geomean_relative_error",
    "holdout_geomean_relative_error",
)


def _ac4_verdict(cell_ids, calibrated, labels, buckets, holdout_ids) -> dict:
    """Full four-criterion accuracy-gate verdict for a calibrated prediction map."""
    preds = [calibrated[c] for c in cell_ids]
    labs = [labels[c] for c in cell_ids]
    holdout = [c for c in cell_ids if c in holdout_ids]
    holdout_geomean = (
        metrics_mod.geomean_relative_error(
            [calibrated[c] for c in holdout], [labels[c] for c in holdout])
        if holdout else 0.0
    )
    per_bucket = {}
    for b in cohort_mod.FAMILY_BUCKETS:
        bc = [c for c in cell_ids if buckets[c] == b]
        if bc:
            per_bucket[b] = {
                "n": len(bc),
                "p90_absolute_error": metrics_mod.p90_absolute_error(
                    [calibrated[c] for c in bc], [labels[c] for c in bc]),
            }
    return accuracy_gate.evaluate_accuracy_gate(
        full_cohort_median_abs=metrics_mod.median_absolute_error(preds, labs),
        full_cohort_geomean_rel=metrics_mod.geomean_relative_error(preds, labs),
        holdout_geomean_rel=holdout_geomean,
        per_bucket_p90=per_bucket,
    )


def _criterion_passed(verdict: dict, name: str) -> bool:
    for c in verdict["criteria"]:
        if c["name"] == name:
            return bool(c["passed"])
    raise KeyError(name)


def verdict_preserved(full_verdict: dict, sampled_verdict: dict) -> dict:
    """Per-criterion accuracy-gate verdict preservation between full and sampled
    replay. Every one of the four accuracy subcriteria (median, geomean, holdout
    geomean, per-bucket p90) must stay on the same pass/fail side."""
    preserved = {
        name.replace("full_cohort_", "").replace("_relative_error", "")
            .replace("_absolute_error", ""):
        _criterion_passed(full_verdict, name) == _criterion_passed(sampled_verdict, name)
        for name in _AC4_CRITERIA
    }
    preserved["per_bucket_p90"] = (
        full_verdict["per_bucket_passed"] == sampled_verdict["per_bucket_passed"]
    )
    return preserved

FIDELITY_PATH = _REPO_ROOT / "data/local_labels/cache_replay_sampled_fidelity.json"
_COMPARISON = _REPO_ROOT / "data/local_labels/cache_replay_comparison.json"

DEFAULT_SET_MODULUS = 32          # 1/32 of sets -> ~3% of records (<= 10%)
ABS_DELTA_BAR = 0.02
MIN_CELLS_WITHIN_BAR = 27
MAX_RECORD_FRACTION = 0.10
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260527
_CONFIG_KWARGS = (
    "capacity_bytes", "associativity", "replacement_policy",
    "num_slices", "slice_hash_id", "capacity_scale",
)


def reconstruct_selected_config(bundle: dict) -> scr.CacheReplayConfig:
    sel = bundle["grid_provenance"]["selected_config"]
    return scr.CacheReplayConfig(**{k: sel[k] for k in _CONFIG_KWARGS})


def _bootstrap_hit_ratio_ci(hit_flags: np.ndarray) -> tuple[float, float]:
    """Percentile-95 bootstrap CI of the hit ratio over the sampled accesses."""
    n = len(hit_flags)
    if n == 0:
        return (0.0, 1.0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for b in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        means[b] = hit_flags[idx].mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def evaluate_sampling_fidelity(
    per_cell: list[dict], *, ac4_verdict_preserved: bool,
) -> dict:
    """Pure fidelity gate over the per-cell sampled-vs-full deltas."""
    n = len(per_cell)
    within = sum(1 for r in per_cell if r["abs_delta"] <= ABS_DELTA_BAR)
    over = n - within
    max_delta = max((r["abs_delta"] for r in per_cell), default=0.0)
    max_fraction = max((r["sampled_record_fraction"] for r in per_cell), default=0.0)
    passed = (
        within >= MIN_CELLS_WITHIN_BAR
        and max_fraction <= MAX_RECORD_FRACTION
        and ac4_verdict_preserved
    )
    return {
        "n_cells": n,
        "n_within_abs_delta_bar": within,
        "n_over_abs_delta_bar": over,
        "abs_delta_bar": ABS_DELTA_BAR,
        "min_cells_within_bar": MIN_CELLS_WITHIN_BAR,
        "max_abs_delta": max_delta,
        "max_sampled_record_fraction": max_fraction,
        "max_record_fraction_bar": MAX_RECORD_FRACTION,
        "ac4_verdict_preserved": ac4_verdict_preserved,
        "passed": passed,
    }


def validate_sampled_fidelity_report(report: dict) -> None:
    """Production validate-before-emit guard for the sampled-fidelity artifact.

    Validates the closed draft-2020-12 schema AND enforces the semantic
    invariant the schema cannot express: the fidelity gate's preservation
    summary must equal the conjunction of the per-criterion preservation flags,
    and if ANY accuracy subcriterion verdict flipped (a False flag) the fidelity
    gate must NOT report passed. Raises ``ValueError`` on violation."""
    artifact_io.validate_against_schema(
        report, "cache_replay_sampled_fidelity.schema.json")
    preserved = report["ac4_verdict_preserved"]
    gate = report["fidelity_gate"]
    expected_all = all(preserved.values())
    if gate.get("ac4_verdict_preserved") != expected_all:
        raise ValueError(
            "fidelity_gate.ac4_verdict_preserved must equal the conjunction of "
            "the per-criterion verdict-preservation flags"
        )
    flipped = sorted(name for name, kept in preserved.items() if not kept)
    if flipped and gate.get("passed"):
        raise ValueError(
            "the fidelity gate must not pass when an accuracy subcriterion "
            f"verdict flipped between full and sampled replay: {flipped}"
        )


def run_sampled_fidelity(*, set_modulus: int = DEFAULT_SET_MODULUS) -> dict:
    bundle = json.loads(_COMPARISON.read_text())
    config = reconstruct_selected_config(bundle)
    full_by_cell = {
        f"{r['slug']}/{r['shape_point']}": r["replay_prediction"]
        for r in bundle["rows"]
    }

    per_cell = []
    full_preds, sampled_preds, labels = [], [], []
    cids, buckets_list = [], []
    for cell, line_ids, sector_offsets in cohort_mod.iter_cohort_streams():
        lids = line_ids.tolist()
        sof = sector_offsets.tolist()
        full_hit = full_by_cell.get(cell.cell_id)
        if full_hit is None:
            full_hit = scr.replay_access_stream(
                lids, sof, config, classify_misses=False
            ).predicted_hit_ratio
        samp = scr.replay_access_stream(
            lids, sof, config, set_sample_modulus=set_modulus
        )
        frac = samp.total_accesses / len(lids) if lids else 0.0
        boot_lo, boot_hi = _bootstrap_hit_ratio_ci(
            np.concatenate([
                np.ones(samp.hits, dtype=np.int8),
                np.zeros(samp.total_accesses - samp.hits, dtype=np.int8),
            ]) if samp.total_accesses else np.zeros(0, dtype=np.int8)
        )
        per_cell.append({
            "cell_id": cell.cell_id,
            "full_hit_ratio": full_hit,
            "sampled_hit_ratio": samp.predicted_hit_ratio,
            "abs_delta": abs(full_hit - samp.predicted_hit_ratio),
            "sampled_record_fraction": frac,
            "sampled_records": samp.total_accesses,
            "wilson_ci": {"ci_low": samp.ci_low, "ci_high": samp.ci_high},
            "bootstrap_ci": {"ci_low": boot_lo, "ci_high": boot_hi},
        })
        full_preds.append(full_hit)
        sampled_preds.append(samp.predicted_hit_ratio)
        labels.append(cell.label)
        cids.append(cell.cell_id)
        buckets_list.append(cell.family_bucket)

    # Accuracy-gate verdict preservation: evaluate ALL FOUR accuracy subcriteria
    # (median, geomean, holdout geomean, per-bucket p90) for full vs sampled,
    # each with the same one-dimensional monotone residual the headline uses.
    holdout_ids = calibrate_mod.holdout_cell_ids(cohort_mod.load_cohort())
    label_by = dict(zip(cids, labels))
    bucket_by = dict(zip(cids, buckets_list))
    cal_full = residual_mod.fit_monotone_residual(full_preds, labels).predict(full_preds)
    cal_samp = residual_mod.fit_monotone_residual(sampled_preds, labels).predict(sampled_preds)
    full_verdict = _ac4_verdict(cids, dict(zip(cids, cal_full)), label_by, bucket_by, holdout_ids)
    sampled_verdict = _ac4_verdict(cids, dict(zip(cids, cal_samp)), label_by, bucket_by, holdout_ids)
    preserved = verdict_preserved(full_verdict, sampled_verdict)
    ac4_preserved = all(preserved.values())

    gate = evaluate_sampling_fidelity(per_cell, ac4_verdict_preserved=ac4_preserved)

    return {
        "schema_version": 1,
        "source_commit": artifact_io.git_head(),
        "top_level_admission_scope": "target_arch=sm_120",
        "arches": ["sm_120"],
        "set_sample_modulus": set_modulus,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "full_cohort_median_absolute_error": metrics_mod.median_absolute_error(full_preds, labels),
        "sampled_cohort_median_absolute_error": metrics_mod.median_absolute_error(sampled_preds, labels),
        "full_ac4_verdict": full_verdict,
        "sampled_ac4_verdict": sampled_verdict,
        "ac4_verdict_preserved": preserved,
        "fidelity_gate": gate,
        "per_cell": per_cell,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Sampled-vs-full fidelity report")
    parser.add_argument("--set-modulus", type=int, default=DEFAULT_SET_MODULUS)
    args = parser.parse_args(argv)
    report = run_sampled_fidelity(set_modulus=args.set_modulus)
    validate_sampled_fidelity_report(report)
    artifact_io.atomic_write_json(FIDELITY_PATH, report)
    g = report["fidelity_gate"]
    print(f"set sampling 1/{report['set_sample_modulus']}: "
          f"{g['n_within_abs_delta_bar']}/{g['n_cells']} cells within "
          f"{g['abs_delta_bar']} (max delta {g['max_abs_delta']:.4f}); "
          f"max record fraction {g['max_sampled_record_fraction']:.4f}; "
          f"ac4 verdict preserved={g['ac4_verdict_preserved']}; passed={g['passed']}")
    print(f"wrote {FIDELITY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
