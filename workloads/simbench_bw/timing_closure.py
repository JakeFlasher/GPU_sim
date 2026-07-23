#!/usr/bin/env python3
"""Closure artifacts for the sm_86 timing cycle headline and deploy-ns leg."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import timing_prediction_contract as contract  # noqa: E402
from workloads.simbench_bw import nvml_covariate_join as nvml  # noqa: E402
from workloads.simbench_bw.timing_accuracy_runner import (  # noqa: E402
    TimingAccuracyError,
    assert_prediction_bundle_deployable_channel,
)
from workloads.simbench_bw.timing_cycle_predictor import (  # noqa: E402
    DEPLOYABLE_CHANNEL,
)


DEFAULT_COHORT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_cohort"
    / "ac4_long_duration_cohort_sm_86.json"
)
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "timing_cycle_predictions_sm_86.json"
)
DEFAULT_ACCURACY = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "timing_accuracy_report_sm_86.json"
)
DEFAULT_CROSSWALK = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "deploy_ns_crosswalk_sm_86.json"
)
DEFAULT_CANDIDATE_COVERAGE_VERDICT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_cohort"
    / "ac4_candidate_coverage_verdict_sm_86.json"
)
DEFAULT_DEPLOY_SIDECAR_DIR = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "deploy_ns_sidecars"
)
DEFAULT_CLOSURE = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "timing_closure_sm_86.json"
)
CLOCK_UNCERTAINTY_FRACTION = 0.06
DUAL_SOURCE_PASS_RATE_THRESHOLD = 0.75


class TimingClosureError(ValueError):
    """Raised when a closure state is internally inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _prediction_lookup(predictions: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (cell["slug"], cell["shape_point"]): cell
        for cell in predictions.get("cells", [])
    }


def _sidecar_repeat_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _sidecar_ref(path: Path) -> dict[str, str]:
    return {"path": _repo_rel(path), "sha256": _sha256_file(path)}


def _load_matching_sidecars(
    *,
    cell: dict[str, Any],
    sidecar_dir: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(
        (sidecar_dir / cell["shape_point"]).glob("cupti_activity_*.json"),
        key=_sidecar_repeat_index,
    ):
        sidecar = _load_json(path)
        for field in (
            "slug",
            "shape_point",
            "binary_sha256",
            "kernel_signature_sha256",
        ):
            if sidecar[field] != cell[field]:
                raise TimingClosureError(
                    f"deploy sidecar {path} has mismatched {field}"
                )
        out.append((path, sidecar))
    return out


def _dual_source_summary(
    sidecars: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    repeat_count = len(sidecars)
    failures = []
    passing = []
    for path, sidecar in sidecars:
        repeat_index = int(sidecar.get("repeat_index", _sidecar_repeat_index(path)))
        if bool(sidecar.get("dual_source_reconciliation_passed", False)):
            passing.append((path, sidecar))
        else:
            event_ns = float(sidecar.get("cuda_event_wall_time_ms", 0.0)) * 1.0e6
            gap_ns = sidecar.get("dual_source_reconciliation_gap_ns")
            failures.append(
                {
                    "repeat_index": repeat_index,
                    "gap_ns": gap_ns if isinstance(gap_ns, int) else None,
                    "relative_pct": (
                        abs(float(gap_ns)) / event_ns * 100.0
                        if isinstance(gap_ns, int) and event_ns > 0.0
                        else None
                    ),
                    "reason": "reconciliation_failed",
                    "cupti_activity_sidecar": _sidecar_ref(path),
                }
            )
    pass_count = len(passing)
    pass_rate = pass_count / repeat_count if repeat_count else 0.0
    passed_gate = repeat_count > 0 and pass_rate >= DUAL_SOURCE_PASS_RATE_THRESHOLD
    return {
        "repeat_count": repeat_count,
        "pass_count": pass_count,
        "failure_count": repeat_count - pass_count,
        "pass_rate": pass_rate,
        "pass_rate_threshold": DUAL_SOURCE_PASS_RATE_THRESHOLD,
        "passed_gate": passed_gate,
        "passing_sidecars": passing,
        "failures": failures,
    }


def build_crosswalk_artifact(
    *,
    cohort: dict[str, Any],
    contract_doc: dict[str, Any],
    cohort_path: Path = DEFAULT_COHORT,
    sidecar_dir: Path = DEFAULT_DEPLOY_SIDECAR_DIR,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Emit the same-binary/same-shape deploy-observable-vs-NCU crosswalk."""
    leg = contract.ns_leg(contract_doc)
    checked_cells = []
    for cell in cohort.get("cells", []):
        ncu_ns = float(cell["labels"]["gpu__time_duration.sum"]["value"])
        sidecars = _load_matching_sidecars(cell=cell, sidecar_dir=sidecar_dir)
        if not sidecars:
            checked_cells.append(
                {
                    "slug": cell["slug"],
                    "shape_point": cell["shape_point"],
                    "arch": cell["arch"],
                    "binary_sha256": cell["binary_sha256"],
                    "kernel_signature_sha256": cell["kernel_signature_sha256"],
                    "ncu_duration_ns": ncu_ns,
                    "cupti_activity_duration_ns": None,
                    "cuda_event_duration_ns": None,
                    "cupti_activity_sidecar": None,
                    "cupti_vs_ncu_delta_ns": None,
                    "cuda_event_vs_ncu_delta_ns": None,
                    "agreement_bound_ns": None,
                    "relative_delta_cupti_vs_ncu": None,
                    "relative_delta_cuda_event_vs_ncu": None,
                    "dual_source_reconciliation_passed": None,
                    "dual_source_repeat_count": 0,
                    "dual_source_pass_count": 0,
                    "dual_source_failure_count": 0,
                    "dual_source_pass_rate": 0.0,
                    "dual_source_pass_rate_threshold": DUAL_SOURCE_PASS_RATE_THRESHOLD,
                    "dual_source_reconciliation_failures": [],
                    "cupti_activity_sidecars": [],
                    "typed_reason": "no_committed_same_binary_same_shape_cupti_activity_sidecar",
                    "status": "missing_deploy_observable",
                }
            )
            continue
        dual_source = _dual_source_summary(sidecars)
        selected_path, sidecar = (
            dual_source["passing_sidecars"][0]
            if dual_source["passing_sidecars"]
            else sidecars[0]
        )
        cupti_ns = float(sidecar["duration_ns"])
        event_ns = float(sidecar["cuda_event_wall_time_ms"]) * 1.0e6
        cupti_delta = abs(cupti_ns - ncu_ns)
        event_delta = abs(event_ns - ncu_ns)
        bound = max(
            float(leg["dual_source_noise_floor_ns"]),
            float(leg["floor_relative_bound_default"]) * ncu_ns,
        )
        measured_agreement = cupti_delta <= bound and event_delta <= bound
        if not dual_source["passed_gate"]:
            status = "dual_source_pass_rate_failed"
            cell_typed_reason = "deploy_ns_dual_source_pass_rate_failed"
        elif measured_agreement:
            status = "agreement_pass"
            cell_typed_reason = None
        else:
            status = "agreement_fail"
            cell_typed_reason = "deploy_ns_crosswalk_measured_agreement_failed"
        checked_cells.append(
            {
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "arch": cell["arch"],
                "binary_sha256": cell["binary_sha256"],
                "kernel_signature_sha256": cell["kernel_signature_sha256"],
                "ncu_duration_ns": ncu_ns,
                "cupti_activity_duration_ns": cupti_ns,
                "cuda_event_duration_ns": event_ns,
                "cupti_activity_sidecar": {
                    "path": _repo_rel(selected_path),
                    "sha256": _sha256_file(selected_path),
                },
                "cupti_activity_sidecars": [_sidecar_ref(path) for path, _ in sidecars],
                "cupti_vs_ncu_delta_ns": cupti_delta,
                "cuda_event_vs_ncu_delta_ns": event_delta,
                "agreement_bound_ns": bound,
                "relative_delta_cupti_vs_ncu": cupti_delta / ncu_ns,
                "relative_delta_cuda_event_vs_ncu": event_delta / ncu_ns,
                "dual_source_reconciliation_passed": bool(
                    dual_source["passed_gate"]
                ),
                "dual_source_repeat_count": dual_source["repeat_count"],
                "dual_source_pass_count": dual_source["pass_count"],
                "dual_source_failure_count": dual_source["failure_count"],
                "dual_source_pass_rate": dual_source["pass_rate"],
                "dual_source_pass_rate_threshold": dual_source["pass_rate_threshold"],
                "dual_source_reconciliation_failures": dual_source["failures"],
                "typed_reason": cell_typed_reason,
                "status": status,
            }
        )
    statuses = {cell["status"] for cell in checked_cells}
    if statuses == {"agreement_pass"}:
        crosswalk_status = "agreement_pass"
        claim_allowed = True
        typed_reason = None
    elif "dual_source_pass_rate_failed" in statuses:
        crosswalk_status = "dual_source_pass_rate_failed"
        claim_allowed = False
        typed_reason = "deploy_ns_dual_source_pass_rate_failed"
    elif "agreement_fail" in statuses:
        crosswalk_status = "agreement_fail"
        claim_allowed = False
        typed_reason = "deploy_ns_crosswalk_measured_agreement_failed"
    elif "indeterminate" in statuses:
        crosswalk_status = "indeterminate"
        claim_allowed = False
        typed_reason = "deploy_ns_crosswalk_indeterminate"
    else:
        crosswalk_status = "missing_deploy_observable"
        claim_allowed = False
        typed_reason = "no_committed_same_binary_same_shape_cupti_activity_sidecar"
    return {
        "schema_version": 1,
        "artifact_kind": "timing_deploy_ns_crosswalk",
        "arch": cohort["arch"],
        "source_commit": source_commit or _source_commit(),
        "cohort_artifact": _repo_rel(cohort_path),
        "reference_metric_id": leg["crosswalk_reference_metric_id"],
        "deploy_observable": leg["crosswalk_deploy_observable"],
        "same_binary_same_shape_required": True,
        "dual_source_noise_floor_ns": leg["dual_source_noise_floor_ns"],
        "floor_relative_bound": leg["floor_relative_bound_default"],
        "crosswalk_status": crosswalk_status,
        "claim_allowed": claim_allowed,
        "typed_reason": typed_reason,
        "checked_cells": checked_cells,
    }


def _ns_interval(predicted_cycles: float, reference: dict[str, Any]) -> dict[str, float]:
    center = nvml.cycles_to_ns(predicted_cycles, reference=reference, regime="short_burst")
    return {
        "lower_ns": center * (1.0 - CLOCK_UNCERTAINTY_FRACTION),
        "center_ns": center,
        "upper_ns": center * (1.0 + CLOCK_UNCERTAINTY_FRACTION),
    }


def assert_honest_closure(closure: dict[str, Any], crosswalk: dict[str, Any]) -> None:
    if closure["cycle_terminal_state"] == "research_success":
        if not closure["cycle_accuracy"]["publication_gate"]["passes"]:
            raise TimingClosureError("cycle success requires a passing cycle gate")
        if closure["cycle_accuracy"].get("scored_channel") != DEPLOYABLE_CHANNEL:
            raise TimingClosureError(
                "cycle success requires the deployable scored channel "
                f"{DEPLOYABLE_CHANNEL!r}; a verdict derived from the "
                "demoted same-slug diagnostic arm is forbidden"
            )
    if closure["deploy_ns_terminal_state"] == "research_success":
        if not crosswalk.get("claim_allowed"):
            raise TimingClosureError(
                "deploy-ns success requires a passing crosswalk artifact"
            )
        if crosswalk.get("crosswalk_status") != "agreement_pass":
            raise TimingClosureError("deploy-ns success requires measured agreement")
        for cell in crosswalk.get("checked_cells", []):
            if cell.get("status") != "agreement_pass":
                raise TimingClosureError(
                    "deploy-ns success requires every checked cell to pass"
                )
            if (
                cell.get("cupti_activity_duration_ns") is None
                or cell.get("cuda_event_duration_ns") is None
            ):
                raise TimingClosureError(
                    "deploy-ns success requires CUPTI Activity and cudaEvent values"
                )
            if cell.get("dual_source_reconciliation_passed") is not True:
                raise TimingClosureError(
                    "deploy-ns success requires passing dual-source reconciliation"
                )
            if (
                float(cell.get("dual_source_pass_rate", 0.0))
                < float(cell.get("dual_source_pass_rate_threshold", 1.0))
            ):
                raise TimingClosureError(
                    "deploy-ns success requires a passing dual-source pass rate"
                )
            if int(cell.get("dual_source_pass_count", 0)) <= 0:
                raise TimingClosureError(
                    "deploy-ns success requires at least one passing sidecar"
                )
            if not cell.get("cupti_activity_sidecars"):
                raise TimingClosureError(
                    "deploy-ns success requires repeated sidecar evidence"
                )
    if closure.get("deploy_ns_point_claimed") is True and not crosswalk.get("claim_allowed"):
        raise TimingClosureError("deploy-ns point claim is forbidden without crosswalk pass")
    candidate = closure.get("candidate_coverage")
    if isinstance(candidate, dict):
        if (
            candidate.get("terminal_state") == "verdict_falsification"
            and candidate.get("prevents_overall_research_success") is True
            and closure.get("overall_terminal_state") == "research_success"
        ):
            raise TimingClosureError(
                "overall success is forbidden when candidate coverage is falsified"
            )


def build_closure_artifact(
    *,
    accuracy: dict[str, Any],
    predictions: dict[str, Any],
    cohort: dict[str, Any],
    crosswalk: dict[str, Any],
    candidate_coverage_verdict: dict[str, Any],
    accuracy_path: Path = DEFAULT_ACCURACY,
    prediction_path: Path = DEFAULT_PREDICTIONS,
    crosswalk_path: Path = DEFAULT_CROSSWALK,
    candidate_coverage_verdict_path: Path = DEFAULT_CANDIDATE_COVERAGE_VERDICT,
    source_commit: str | None = None,
) -> dict[str, Any]:
    # Fail closed BEFORE deriving any terminal state: the closure may
    # only be driven by the deployable scored channel. An accuracy
    # report missing the channel declaration (pre-demotion shape) or
    # scoring a diagnostic channel cannot close anything.
    scored_channel = accuracy.get("scored_channel")
    if scored_channel != DEPLOYABLE_CHANNEL:
        raise TimingClosureError(
            "closure_scored_channel_not_deployable: the accuracy report "
            f"must score the deployable channel {DEPLOYABLE_CHANNEL!r}; "
            f"got {scored_channel!r}. Terminal states cannot derive from "
            "the demoted same-slug diagnostic arm."
        )
    # Defense in depth: re-run the full bundle channel/value audit on
    # the predictions input. A value-substituted bundle (the retired
    # same-slug numbers wearing the deployable channel string) cannot
    # reach a terminal state even alongside a forged accuracy report.
    try:
        assert_prediction_bundle_deployable_channel(predictions)
    except TimingAccuracyError as exc:
        raise TimingClosureError(
            f"closure_prediction_bundle_rejected: {exc}"
        ) from exc
    reference = nvml.load_reference_operating_point(regime="short_burst")
    pred_lookup = _prediction_lookup(predictions)
    intervals = []
    for cell in cohort.get("cells", []):
        pred_cell = pred_lookup[(cell["slug"], cell["shape_point"])]
        predicted_cycles = float(pred_cell["prediction"]["predicted_cycles"])
        interval = _ns_interval(predicted_cycles, reference)
        intervals.append(
            {
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "regime": "long",
                "predicted_cycles": predicted_cycles,
                "observed_ncu_ns_diagnostic": float(
                    cell["labels"]["gpu__time_duration.sum"]["value"]
                ),
                **interval,
            }
        )

    cycle_state = (
        "research_success"
        if accuracy["publication_gate"]["passes"]
        else "verdict_falsification"
    )
    deploy_ns_state = (
        "research_success" if crosswalk["claim_allowed"] else "verdict_falsification"
    )
    candidate_prevents_success = bool(
        candidate_coverage_verdict.get("prevents_overall_research_success")
    )
    overall_state = (
        "verdict_falsification" if candidate_prevents_success else cycle_state
    )
    closure = {
        "schema_version": 1,
        "artifact_kind": "timing_closure",
        "arch": accuracy["arch"],
        "source_commit": source_commit or _source_commit(),
        "cycle_terminal_state": cycle_state,
        "deploy_ns_terminal_state": deploy_ns_state,
        "overall_terminal_state": overall_state,
        "accuracy_report": _repo_rel(accuracy_path),
        "prediction_bundle": _repo_rel(prediction_path),
        "crosswalk_artifact": _repo_rel(crosswalk_path),
        "candidate_coverage_artifact": _repo_rel(candidate_coverage_verdict_path),
        "cycle_accuracy": {
            "scored_channel": scored_channel,
            "scores": accuracy["scores"],
            "publication_gate": accuracy["publication_gate"],
            "result_kind": accuracy["result_kind"],
        },
        "deploy_ns": {
            "claim_kind": (
                "deploy_ns_pass" if crosswalk["claim_allowed"] else "verdict_falsification"
            ),
            "typed_reason": crosswalk["typed_reason"],
            "crosswalk_claim_allowed": crosswalk["claim_allowed"],
            "point_result_claimed": False,
            "clock_uncertainty_fraction": CLOCK_UNCERTAINTY_FRACTION,
            "anchor_regime": "short_burst",
            "anchor_source": reference["source"],
            "sm_clock_ref_mhz": reference["sm_clock_ref_mhz"],
        },
        "deploy_ns_point_claimed": False,
        "candidate_coverage": {
            "terminal_state": candidate_coverage_verdict["terminal_state"],
            "prevents_overall_research_success": candidate_prevents_success,
            "typed_reason": candidate_coverage_verdict["typed_reason"],
            "uncovered_required_families": candidate_coverage_verdict[
                "uncovered_required_families"
            ],
        },
        "ns_intervals": intervals,
        "evidence_status": "Offline-verified",
    }
    assert_honest_closure(closure, crosswalk)
    return closure


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--accuracy", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--contract", type=Path, default=contract.DEFAULT_CONTRACT_PATH)
    parser.add_argument("--sidecar-dir", type=Path, default=DEFAULT_DEPLOY_SIDECAR_DIR)
    parser.add_argument("--crosswalk-out", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument(
        "--candidate-coverage-verdict",
        type=Path,
        default=DEFAULT_CANDIDATE_COVERAGE_VERDICT,
    )
    parser.add_argument("--closure-out", type=Path, default=DEFAULT_CLOSURE)
    args = parser.parse_args(argv)

    cohort = _load_json(args.cohort)
    predictions = _load_json(args.predictions)
    accuracy = _load_json(args.accuracy)
    candidate_coverage_verdict = _load_json(args.candidate_coverage_verdict)
    contract_doc = contract.load_contract(args.contract)
    crosswalk = build_crosswalk_artifact(
        cohort=cohort,
        contract_doc=contract_doc,
        cohort_path=args.cohort,
        sidecar_dir=args.sidecar_dir,
    )
    write_json(crosswalk, args.crosswalk_out)
    closure = build_closure_artifact(
        accuracy=accuracy,
        predictions=predictions,
        cohort=cohort,
        crosswalk=crosswalk,
        candidate_coverage_verdict=candidate_coverage_verdict,
        accuracy_path=args.accuracy,
        prediction_path=args.predictions,
        crosswalk_path=args.crosswalk_out,
        candidate_coverage_verdict_path=args.candidate_coverage_verdict,
    )
    write_json(closure, args.closure_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
