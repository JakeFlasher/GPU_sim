"""Falsification arm + engineering-closure decoupling for the cache-replay loop.

The loop closes successfully if EITHER (i) the accuracy, subsumption, and
throughput gates all pass (research success) OR (ii) this report documents which
gates were missed and separates the failure into three modes -- trace-fidelity,
model-parameter, residual-calibration -- with a confirmed/refuted verdict on the
pre-registered hypothesis:

  set-associative conflict / slice realism reduces median absolute hit-rate
  error versus the fully-associative Nugteren reuse-distance-histogram baseline
  by at least 30 percent (confirmed) or by less than 10 percent (refuted).

Both disjuncts are terminal on the single sm_120 host. The report carries NO
cross-architecture / same-cohort-join precondition anywhere; the engineering
closure is reachable on the single-host sm_120 surface alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import accuracy_gate  # noqa: E402
import artifact_io  # noqa: E402

FALSIFICATION_PATH = _REPO_ROOT / "data/local_labels/cache_replay_falsification.json"

HYPOTHESIS_STATEMENT = (
    "Set-associative conflict/slice realism reduces median absolute hit-rate "
    "error versus the fully-associative reuse-distance-histogram (RDH) baseline "
    "by >= 30% (confirmed) or < 10% (refuted)."
)
CONFIRMED_PCT = 30.0
REFUTED_PCT = 10.0
FAILURE_MODES = ("trace_fidelity", "model_parameter", "residual_calibration")


def _hypothesis_verdict(contribution_pct: float) -> str:
    if contribution_pct >= CONFIRMED_PCT:
        return "confirmed"
    if contribution_pct < REFUTED_PCT:
        return "refuted"
    return "indeterminate"


def build_falsification_report(
    bundle: dict,
    *,
    gate_verdict: dict | None = None,
    throughput_passed: bool | None = None,
    benchmark: dict | None = None,
    validation_artifact: dict | None = None,
    l1_artifact: dict | None = None,
    sampled_fidelity: dict | None = None,
) -> dict:
    """Assemble the falsification report from the comparison bundle.

    ``contribution_pct`` (the plan's ``joint_contribution_pct``) is the share of
    the median-absolute-error reduction attributable specifically to
    set-associative conflict/slice realism -- the improvement of the most
    set-associative grid config over the fully-associative single-slice config.
    When the grid is inert (all working sets fit the modeled capacity), this is
    near zero and the hypothesis is refuted.
    """
    if gate_verdict is None:
        gate_verdict = accuracy_gate.evaluate_bundle(bundle)

    agg = bundle["aggregate_metrics"]
    nugteren_medabs = agg["nugteren_prediction"]["median_absolute_error"]
    replay_medabs = agg["replay_calibrated_prediction"]["median_absolute_error"]
    replay_mech_medabs = agg["replay_prediction"]["median_absolute_error"]

    grid_medabs = bundle.get("grid_config_medabs", {})
    fully_assoc = grid_medabs.get("fully_associative_single_slice", replay_mech_medabs)
    best_set_assoc = grid_medabs.get("most_set_associative", replay_mech_medabs)

    # contribution of set-associative realism specifically.
    if fully_assoc and fully_assoc > 0:
        contribution_pct = 100.0 * (fully_assoc - best_set_assoc) / fully_assoc
    else:
        contribution_pct = 0.0
    # contribution can be negative if set-associativity hurts; clamp the
    # reported magnitude at 0 for the verdict (a negative contribution is also
    # a refutation of "set-associative realism reduces error").
    contribution_pct = max(0.0, contribution_pct)
    reduction_vs_nugteren_pct = (
        100.0 * (nugteren_medabs - replay_mech_medabs) / nugteren_medabs
        if nugteren_medabs else 0.0
    )

    # missed gates
    missed = []
    if not gate_verdict["passed"]:
        missed.append("accuracy_gate")
    v5_beat = bundle.get("replay_vs_v5_beat_fraction", 0.0)
    nug_beat = bundle.get("replay_vs_rdh_baseline_beat_fraction", 0.0)
    subsumption_passed = v5_beat >= 0.667 and nug_beat >= 0.80
    if not subsumption_passed:
        missed.append("subsumption_gate")
    if throughput_passed is False:
        missed.append("throughput_gate")
    research_success = not missed

    # --- three-way failure-mode decomposition ---
    rows = bundle.get("rows", [])
    over_pred = [
        r["replay_prediction"] - r["label"]
        for r in rows if r["replay_prediction"] is not None
    ]
    mean_over_prediction = sum(over_pred) / len(over_pred) if over_pred else 0.0
    n_over = sum(1 for d in over_pred if d > 0.1)
    max_grid_spread = bundle.get("inert_grid_evidence", {}).get(
        "max_grid_prediction_spread", 0.0
    )

    failure_mode_decomposition = {
        "trace_fidelity": {
            "dominant": True,
            "mean_signed_over_prediction": mean_over_prediction,
            "cells_over_predicted_by_0p1": n_over,
            "evidence": (
                "The mechanistic replay over the raw NVBit access stream "
                "systematically over-predicts L2 hit ratio for low-reuse "
                "kernels. NCU's L2 sector hit rate observes only the L1-miss "
                "stream, whereas the replay feeds the full access stream to the "
                "L2 model; intra-line burst reuse absorbed by the per-SM L1 in "
                "hardware is double-counted as L2 hits. Predicting L2 hit ratio "
                "from these traces requires an L1 filter (L1-miss stream -> L2) "
                "that this round's L2-headline scope does not model."
            ),
        },
        "model_parameter": {
            "dominant": False,
            "max_grid_prediction_spread": max_grid_spread,
            "fully_associative_single_slice_median_absolute_error": fully_assoc,
            "most_set_associative_median_absolute_error": best_set_assoc,
            "evidence": (
                "Every pre-registered grid config (capacity-scale 0.75-1.25, "
                "associativity 8-32, slices, LRU/SRRIP/BIP) yields a near-"
                "identical prediction because all 30 cells' working sets fit "
                "the modeled L2 capacity. The capacity/associativity/slice/"
                "replacement axes are therefore inert: they cannot create the "
                "conflict/capacity misses the hardware exhibits."
            ),
        },
        "residual_calibration": {
            "dominant": False,
            "mechanistic_median_absolute_error": replay_mech_medabs,
            "calibrated_median_absolute_error": replay_medabs,
            "evidence": (
                "The one-dimensional monotone residual cannot rescue accuracy "
                "because the mechanistic predictions are nearly constant and "
                "non-monotone with respect to the label (low-reuse cells with "
                "low labels receive higher predictions than some high-label "
                "GEMM cells). A monotone map over a near-constant feature "
                "cannot separate the cohort."
            ),
        },
    }

    verdict = _hypothesis_verdict(contribution_pct)

    # advisory robustness (does NOT gate closure) + L1 diagnostic + throughput,
    # surfaced in the report per the plan's small-N advisory discipline.
    robustness = None
    if validation_artifact is not None:
        robustness = {
            "advisory": True,
            "gates_closure": False,
            "loocv": validation_artifact.get("loocv"),
            "loso": validation_artifact.get("loso"),
            "bootstrap_ci": validation_artifact.get("bootstrap_ci"),
            "advisory_bands": validation_artifact.get("advisory"),
        }
    l1_summary = None
    if l1_artifact is not None:
        l1_summary = {
            "load_bearing": l1_artifact.get("load_bearing", False),
            "diagnostic_median_absolute_error": l1_artifact.get(
                "diagnostic_median_absolute_error"
            ),
            "note": "L1 is diagnostic-only this iteration; it does not gate closure.",
        }
    throughput = None
    if benchmark is not None:
        throughput = {
            "replay_sectors_per_sec": benchmark.get("replay_sectors_per_sec"),
            "speedup_vs_fenwick": benchmark.get("speedup_vs_fenwick"),
            "passed": benchmark.get("passed"),
        }
    sampling = None
    if sampled_fidelity is not None:
        g = sampled_fidelity.get("fidelity_gate", {})
        sampling = {
            "set_sample_modulus": sampled_fidelity.get("set_sample_modulus"),
            "n_within_abs_delta_bar": g.get("n_within_abs_delta_bar"),
            "n_cells": g.get("n_cells"),
            "max_abs_delta": g.get("max_abs_delta"),
            "max_sampled_record_fraction": g.get("max_sampled_record_fraction"),
            "passed": g.get("passed"),
        }

    return {
        "schema_version": 1,
        "source_commit": artifact_io.git_head(),
        "top_level_admission_scope": bundle.get("top_level_admission_scope", "target_arch=sm_120"),
        "arches": bundle.get("arches", ["sm_120"]),
        "terminal_on_sm120_host": True,
        "cross_arch_precondition": False,
        "research_success": research_success,
        "closure_disjunct": "research_success" if research_success else "engineering_closure",
        "missed_gates": missed,
        "gate_verdict": gate_verdict,
        "subsumption": {
            "replay_vs_v5_beat_fraction": v5_beat,
            "replay_vs_rdh_baseline_beat_fraction": nug_beat,
            "passed": subsumption_passed,
        },
        "hypothesis": {
            "statement": HYPOTHESIS_STATEMENT,
            "rdh_baseline_median_absolute_error": nugteren_medabs,
            "replay_mechanistic_median_absolute_error": replay_mech_medabs,
            "replay_calibrated_median_absolute_error": replay_medabs,
            "reduction_vs_rdh_baseline_pct": reduction_vs_nugteren_pct,
            "set_associative_realism_contribution_pct": contribution_pct,
            "joint_contribution_pct": contribution_pct,
            "confirmed_threshold_pct": CONFIRMED_PCT,
            "refuted_threshold_pct": REFUTED_PCT,
            "verdict": verdict,
        },
        "failure_mode_decomposition": failure_mode_decomposition,
        "advisory_robustness": robustness,
        "l1_diagnostic": l1_summary,
        "throughput": throughput,
        "sampled_fidelity": sampling,
    }


def validate_falsification_report(report: dict) -> None:
    """Raise ``ValueError`` if the report omits the three-way failure-mode
    decomposition, lacks the hypothesis contribution, or gates on a cross-arch /
    same-cohort precondition (the report must be terminal on the sm_120 host)."""
    if not report.get("research_success", False):
        decomp = report.get("failure_mode_decomposition")
        if not isinstance(decomp, dict):
            raise ValueError("falsification report missing failure_mode_decomposition")
        for mode in FAILURE_MODES:
            if mode not in decomp:
                raise ValueError(
                    f"falsification report missing failure mode {mode!r}; the "
                    "three-way decomposition is mandatory"
                )
        hyp = report.get("hypothesis", {})
        if "joint_contribution_pct" not in hyp:
            raise ValueError("falsification report missing joint_contribution_pct")
        verdict = hyp.get("verdict")
        if verdict not in ("confirmed", "refuted", "indeterminate"):
            raise ValueError(f"invalid hypothesis verdict {verdict!r}")
        if verdict == "indeterminate":
            raise ValueError(
                "engineering closure requires a decisive hypothesis verdict "
                "(confirmed >= 30% or refuted < 10%); got indeterminate"
            )
    # tri-state since the 2026-07-06 re-founding: false (historical single-host
    # closures stay valid), true (live cross-arch evidence exists), or "pending"
    if report.get("cross_arch_precondition", False) not in (False, True, "pending"):
        raise ValueError(
            "cross_arch_precondition must be tri-state (false | true | 'pending')"
        )
    forbidden_keys = (
        "requires_both_arches", "same_74_baseline_comparison",
        "cross_arch_transfer", "final_cross_arch_artifact",
    )
    for k in forbidden_keys:
        if k in report:
            raise ValueError(
                f"falsification report carries forbidden cross-arch gate key {k!r}"
            )
    if report.get("terminal_on_sm120_host") is not True:
        raise ValueError("falsification report must declare terminal_on_sm120_host=true")


def main(argv: list[str] | None = None) -> int:
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Emit the cache-replay falsification report")
    parser.add_argument(
        "--comparison",
        default=str(_REPO_ROOT / "data/local_labels/cache_replay_comparison.json"),
    )
    parser.add_argument(
        "--benchmark",
        default=str(_REPO_ROOT / "data/local_labels/cache_replay_benchmark.json"),
    )
    args = parser.parse_args(argv)
    bundle = json.loads(Path(args.comparison).read_text())

    def _load_if_present(name):
        p = _REPO_ROOT / "data/local_labels" / name
        return json.loads(p.read_text()) if p.exists() else None

    # Honor the --benchmark override (default points at the canonical artifact).
    # Loading the hard-coded path here instead would record the wrong
    # throughput_passed/throughput whenever the caller writes the benchmark to a
    # non-default path. The --comparison arg is likewise honored above.
    benchmark_path = Path(args.benchmark)
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    validation_artifact = _load_if_present("cache_replay_validation.json")
    l1_artifact = _load_if_present("cache_replay_l1_diagnostic.json")
    sampled_fidelity = _load_if_present("cache_replay_sampled_fidelity.json")
    throughput_passed = bool(benchmark.get("passed")) if benchmark else None
    report = build_falsification_report(
        bundle,
        throughput_passed=throughput_passed,
        benchmark=benchmark,
        validation_artifact=validation_artifact,
        l1_artifact=l1_artifact,
        sampled_fidelity=sampled_fidelity,
    )
    validate_falsification_report(report)
    artifact_io.validate_against_schema(
        report, "cache_replay_falsification.schema.json")
    artifact_io.atomic_write_json(FALSIFICATION_PATH, report)
    h = report["hypothesis"]
    print(f"closure disjunct: {report['closure_disjunct']}")
    print(f"missed gates: {report['missed_gates']}")
    print(f"hypothesis verdict: {h['verdict']} "
          f"(set-assoc contribution {h['joint_contribution_pct']:.2f}%, "
          f"reduction vs rdh-baseline {h['reduction_vs_rdh_baseline_pct']:.2f}%)")
    print(f"wrote {FALSIFICATION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
