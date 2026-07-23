"""Emit the execution-time prediction contract artifact for the local
RTX 3090 Ti (sm_86).

The contract fixes the execution-time prediction publication discipline: the
DVFS-stable cycle count ``gpc__cycles_elapsed.max`` is the sole publication
headline target; the derived wall-clock nanosecond leg is a clock-uncertainty
interval that may not close the success terminal state on its own and is gated
behind a CUPTI-Activity-vs-NCU agreement (crosswalk) artifact plus a cohort
long enough to clear the measured dual-source noise floor.

This producer builds the contract dict, validates it against the canonical
schema (``workloads/simbench_bw/schemas/timing_prediction_contract.schema.json``)
BEFORE writing, then emits canonical JSON
(``sort_keys=True, indent=2`` + trailing newline) to
``data/local_labels/timing_prediction_contract_sm_86.json``. Validating before
the write is deliberate: a non-conformant contract must never reach disk, so
the producer shares the consumer's single source of truth (the schema) instead
of trusting hand-built field shapes.

The numeric thresholds are pinned in the schema as ``const`` values; this
producer reproduces the same literals so the emitted artifact validates. The
nanosecond-floor and cohort-duration constants are derived from the measured
provenanced dual-source p99 noise floor recorded in
``workloads/simbench_bw/golden/cupti_dual_source_tolerance.json``.

CLI:
  python tools/local_labels/emit_timing_prediction_contract.py \
    [--out-path data/local_labels/timing_prediction_contract_sm_86.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import timing_prediction_contract as _contract  # noqa: E402

_DEFAULT_OUT = (
    REPO_ROOT / "data" / "local_labels" / "timing_prediction_contract_sm_86.json"
)

# Provenanced p99 dual-source reconciliation noise floor (nanoseconds) for this
# host, sourced from workloads/simbench_bw/golden/cupti_dual_source_tolerance.json
# (calibration_gap_distribution_ns.p99). Any deploy-nanosecond claim must clear
# it floor-relative.
_DUAL_SOURCE_NOISE_FLOOR_NS = 110352
_DEFAULT_FLOOR_RELATIVE_BOUND = 0.2
_TIGHTER_FLOOR_RELATIVE_BOUND = 0.1
# Minimum cohort kernel duration for a floor-relative claim = floor / bound.
_MIN_COHORT_NS_DEFAULT_BOUND = round(
    _DUAL_SOURCE_NOISE_FLOOR_NS / _DEFAULT_FLOOR_RELATIVE_BOUND
)
_MIN_COHORT_NS_TIGHTER_BOUND = round(
    _DUAL_SOURCE_NOISE_FLOOR_NS / _TIGHTER_FLOOR_RELATIVE_BOUND
)

# Pinned bootstrap configuration, matching the cache runner's pinned floor.
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_CI_CONFIDENCE = 0.95

# Pinned cycle accuracy gate thresholds (multiplicative log-target).
_GEOMEAN_RELATIVE_ERROR_MAX = 0.15
_MEDIAN_ABS_LOG_RELATIVE_ERROR_MAX = 0.1
_BOOTSTRAP_GEOMEAN_UPPER_BOUND_MAX = 0.25

# Coverage tiers.
_SMOKE_MIN_CELLS = 5
_PUBLICATION_MIN_CELLS = 10

# Plausible SM-clock band (MHz) for the cycles-to-nanoseconds conversion; the
# conversion fails closed outside this band.
_CLOCK_BAND_MIN_MHZ = 300
_CLOCK_BAND_MAX_MHZ = 3000


def _resolve_current_source_commit() -> str:
    """Return the current ``HEAD`` commit hash, or ``"unknown"`` when git is
    unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def build_contract(*, source_commit: str | None = None) -> dict:
    """Build the contract dict. Pure (no I/O beyond an optional git lookup);
    the result is what gets schema-validated and written."""
    commit = source_commit if source_commit is not None else _resolve_current_source_commit()
    generated_at = (
        _dt.datetime.now(tz=_dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    return {
        "schema_version": 1,
        "artifact_kind": "timing_prediction_contract",
        "arch": "sm_86",
        "source_commit": commit,
        "generated_at_utc": generated_at,
        "headline": {
            "publication_target_metric_id": "gpc__cycles_elapsed.max",
            "publication_target_registry_key": "exec_cycles_gpc_max",
            "target_unit": "cycles",
            "is_sole_publication_headline": True,
            "rationale": (
                "The elapsed cycle count is DVFS-stable, so it is the only "
                "load-bearing publication target. The nanosecond value is the "
                "cycle count multiplied by a clock conversion that "
                "double-counts the cycle error and carries clock-regime "
                "uncertainty, so it is a derived diagnostic, not a co-equal "
                "headline."
            ),
        },
        "cycle_error_definition": {
            "per_cell_residual_formula": "ln(p_i / t_i)",
            "geomean_relative_error_formula": "exp(mean_i(abs(r_i))) - 1",
            "median_abs_log_relative_error_formula": "median_i(abs(r_i))",
            "bootstrap_statistic": "geomean_relative_error",
            "epsilon_floor_applied": False,
            "requires_strictly_positive_inputs": True,
            "forbidden_bounded_fraction_formula": (
                "abs(pred - true) / max(true, epsilon)"
            ),
        },
        "publication_gate": {
            "min_scored_cells": _PUBLICATION_MIN_CELLS,
            "geomean_relative_error_max": _GEOMEAN_RELATIVE_ERROR_MAX,
            "median_abs_log_relative_error_max": _MEDIAN_ABS_LOG_RELATIVE_ERROR_MAX,
            "bootstrap_geomean_upper_bound_max": _BOOTSTRAP_GEOMEAN_UPPER_BOUND_MAX,
            "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
            "bootstrap_ci_confidence": _BOOTSTRAP_CI_CONFIDENCE,
            "closes_research_success": True,
        },
        "smoke_tier": {
            "min_scored_cells": _SMOKE_MIN_CELLS,
            "reports_only": True,
            "closes_research_success": False,
        },
        "deploy_ns_leg": {
            "role": "derived_clock_uncertainty_interval",
            "closes_research_success_alone": False,
            "requires_crosswalk_artifact": True,
            "crosswalk_reference_metric_id": "gpu__time_duration.sum",
            "crosswalk_deploy_observable": "cupti_activity_plus_cuda_event",
            "dual_source_noise_floor_ns": _DUAL_SOURCE_NOISE_FLOOR_NS,
            "floor_relative_bound_default": _DEFAULT_FLOOR_RELATIVE_BOUND,
            "min_cohort_duration_ns_for_default_bound": _MIN_COHORT_NS_DEFAULT_BOUND,
            "tighter_bound": _TIGHTER_FLOOR_RELATIVE_BOUND,
            "min_cohort_duration_ns_for_tighter_bound": _MIN_COHORT_NS_TIGHTER_BOUND,
            "clock_plausibility_band_mhz": {
                "min": _CLOCK_BAND_MIN_MHZ,
                "max": _CLOCK_BAND_MAX_MHZ,
            },
        },
        "terminal_states": {
            "research_success": (
                "Recorded when the cycle publication gate passes on the long "
                "cohort; the cycle headline is the only leg permitted to close "
                "this state."
            ),
            "verdict_falsification": (
                "Recorded when a leg is honestly refuted below its floor; a "
                "first-class terminal outcome, never a silent or reported-only "
                "result."
            ),
            "deferred_cross_arch_requires_sm120_session": (
                "Sentinel recorded for any cross-architecture leg this "
                "single-architecture session cannot witness."
            ),
        },
        "evidence_statuses": ["Offline-verified", "Verified-live-sm_86"],
    }


def emit_contract(*, out_path: Path, source_commit: str | None = None) -> dict:
    """Build, schema-validate, and write the contract artifact. Returns the
    written dict."""
    doc = build_contract(source_commit=source_commit)
    # Fail closed before touching disk: a non-conformant contract is a defect,
    # not an artifact.
    _contract.validate_contract(doc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return doc


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output JSON (default: {_DEFAULT_OUT.relative_to(REPO_ROOT)})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    doc = emit_contract(out_path=args.out_path)
    try:
        shown = args.out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.out_path
    print(f"wrote {shown} (source_commit={doc['source_commit']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
