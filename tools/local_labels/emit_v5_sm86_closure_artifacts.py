"""Emit sm_86-scoped v5 closure artifacts (accuracy gates, baseline comparison,
robustness, falsification) to ``*_sm_86.json`` paths.

This orchestrator REUSES the existing single-head closure emitters
(``emit_v5_closure_artifacts``, ``emit_v5_robustness_artifact``, the Nugteren
2014 reuse-distance-histogram baseline runner) pointed at the STANDALONE sm_86
head, then layers two pieces those emitters do not produce on their own:

  - a ``per_slice`` block in the accuracy artifact with an ``sm_86_only`` row
    (the evaluated sm_86 slice) and a ``same_30_frozen_b1646ef7`` row -- a
    regression guard that re-derives the frozen sm_120 cohort's accuracy from
    the untouched joint head and confirms it replays within 1e-9 of the values
    committed at the frozen reference, proving the standalone sm_86 fit did not
    disturb the sm_120 fit;

  - explicit ``deferred_cross_arch_requires_sm120_session`` records (each with a
    backing regeneration-ledger citation) for the cross-architecture transfer
    legs and the joint same-74 falsification verdict, which a single-architecture
    session cannot witness. No finite cross-architecture number is asserted.

The same-74 / joint closure artifacts (the non-suffixed paths) are left
UNTOUCHED -- they belong to the deferred sm_120-session deliverable and are
validated as-is by the cohort-coverage scanner.

CLI:
  python tools/local_labels/emit_v5_sm86_closure_artifacts.py \
    [--sm86-head data/local_labels/sm_86_v5_head.json] \
    [--ledger data/local_labels/sm86_cohort_regen_ledger.json] \
    [--joint-head data/local_labels/calibration_head_v5.json] \
    [--v4-head data/local_labels/calibration_head_v4.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels.corpus_cell_identity import (  # noqa: E402
    is_base_predictor_fit_cell,
)
from tools.local_labels import emit_v5_closure_artifacts as _closure  # noqa: E402
from tools.local_labels import emit_v5_robustness_artifact as _robustness  # noqa: E402
from tools.baseline_runners import nugteren2014_rdh_baseline as _nugteren  # noqa: E402

# The frozen sm_120 cohort reference: the 30 sm_120 admitted artifacts are
# pinned at this commit and treated as untouched inputs this session.
FROZEN_SM120_REF = "b1646ef7"
FROZEN_REPLAY_TOLERANCE = 1e-9
# Sentinel recorded on every cross-architecture / joint same-74 leg this
# single-architecture session cannot witness.
DEFERRED_STATUS = "deferred_cross_arch_requires_sm120_session"

DEFAULT_SM86_HEAD = "data/local_labels/sm_86_v5_head.json"
DEFAULT_LEDGER = "data/local_labels/sm86_cohort_regen_ledger.json"
DEFAULT_JOINT_HEAD = "data/local_labels/calibration_head_v5.json"
DEFAULT_V4_HEAD = "data/local_labels/calibration_head_v4.json"

DEFAULT_ACCURACY_OUT = (
    "data/local_labels/calibration_head_v5_accuracy_gates_sm_86.json"
)
DEFAULT_BASELINE_OUT = (
    "data/local_labels/calibration_head_v5_baseline_comparison_sm_86.json"
)
DEFAULT_ROBUSTNESS_OUT = (
    "data/local_labels/calibration_head_v5_robustness_sm_86.json"
)
DEFAULT_FALSIFICATION_OUT = (
    "data/local_labels/calibration_head_v5_falsification_sm_86.json"
)

# Frozen-cohort accuracy gates committed at the frozen reference, used as the
# regression-guard baseline for the same_30_frozen_b1646ef7 row.
_FROZEN_ACCURACY_REL = (
    "data/local_labels/calibration_head_v5_accuracy_gates.json"
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_show_json(ref: str, rel_path: str) -> dict | None:
    try:
        out = subprocess.check_output(
            ["git", "show", f"{ref}:{rel_path}"], cwd=str(REPO_ROOT),
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _flatten_gate_metrics(gates_artifact: dict | None) -> dict[str, float]:
    """Flatten the headline numeric accuracy-gate metrics of a gates artifact
    into a flat {name: value} map for a tolerance comparison. Only numeric
    values are included; None-valued (not-computed) gates are skipped."""
    out: dict[str, float] = {}
    gates = (gates_artifact or {}).get("gates") or {}
    for key in (
        "full_cohort_median_absolute_error",
        "full_cohort_geomean_relative_error",
        "holdout_geomean_relative_error",
    ):
        value = gates.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
    per_bucket = gates.get("per_bucket_p90_absolute_error") or {}
    for bucket, entry in per_bucket.items():
        value = (entry or {}).get("p90_absolute_error")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"per_bucket_p90/{bucket}"] = float(value)
    return out


def _ledger_admitted_bundle_lookup(
    ledger_path: Path,
) -> dict[tuple[str, str, str], dict]:
    """Build a ``{(arch, slug, shape_point): bundle}`` lookup from the ledger's
    admitted cells, keyed by the ledger cell's ARCHITECTURE (sm_86) so it
    matches the head's cohort cells during the baseline join.

    The per-cell predictor bundles' ``manifest_pairing_index`` is a closed
    5-field identity with NO arch key, so keying by the bundle's own MPI would
    produce ``arch=''`` keys that miss the sm_86 cohort cells (whose arch is
    sm_86) -- the downstream baseline join does an exact-then-same-arch lookup
    and never matches an empty-arch bundle to a sm_86 cell. Each bundle is
    sha256-verified against the ledger's recorded digest, so the lookup is
    collision-proof against a superseded same-(slug,shape) bundle left on disk.
    """
    led = json.loads(ledger_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], dict] = {}
    for cell in led.get("cells", []):
        if cell.get("status") != "admitted":
            continue
        # Base predictor-fit cohort only: a register-cap variant or pending-refit
        # factorial cell shares its base (arch, slug, shape_point) join key and
        # is not yet folded into the head, so it must not supply a baseline row
        # here (it would collide on the bare 3-tuple key and report a different
        # admitted count than the fitted head). Same scope as the v5 fitter.
        if not is_base_predictor_fit_cell(cell):
            continue
        rel = cell.get("predictor_bundle_path")
        if not rel:
            raise ValueError(
                "admitted ledger cell missing predictor_bundle_path: "
                f"{cell.get('arch')}/{cell.get('slug')}/{cell.get('shape_point')}"
            )
        candidate = Path(rel)
        resolved = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        data = resolved.read_bytes()
        expected = (cell.get("artifact_sha256") or {}).get("predictor_bundle")
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(
                f"admitted predictor bundle sha256 mismatch for {rel}"
            )
        key = (
            str(cell.get("arch", "")),
            str(cell.get("slug", "")),
            str(cell.get("shape_point", "")),
        )
        out[key] = json.loads(data)
    return out


def _ledger_citation(ledger_path: Path) -> dict[str, Any]:
    """Backing citation for every deferred leg: which ledger establishes that
    the sm_86 lane carries live sm_86 evidence, and why cross-architecture
    closure still needs the sm_120 session."""
    led = json.loads(ledger_path.read_text(encoding="utf-8"))
    # Scope to the base predictor-fit cohort (same as the fitted head + baseline
    # lookup) so the citation's admitted count + live-evidence summary describe
    # the cohort the closure is actually over, not the wider corpus.
    admitted = [
        c for c in led.get("cells", [])
        if c.get("status") == "admitted" and is_base_predictor_fit_cell(c)
    ]
    live_statuses = sorted({
        str(c.get("live_evidence_status")) for c in admitted
    })
    return {
        "ledger_path": _rel(ledger_path),
        "sm_86_admitted_count": len(admitted),
        "sm_86_live_evidence_status": live_statuses,
        "note": (
            "the sm_86 lane carries live sm_86 evidence (see ledger); the "
            "cross-architecture transfer legs and the joint same-74 "
            "falsification verdict require the matching sm_120-session live "
            "witness and are deferred -- no finite cross-architecture number "
            "is asserted here"
        ),
    }


def _same_30_frozen_regression_guard(joint_head_path: Path) -> dict[str, Any]:
    """Re-derive the frozen sm_120 cohort's accuracy from the untouched joint
    head and confirm it replays within tolerance of the values committed at the
    frozen reference. A drift beyond tolerance means the sm_120 fit was
    disturbed (it must not have been: this session writes only the standalone
    sm_86 head)."""
    committed = _git_show_json(FROZEN_SM120_REF, _FROZEN_ACCURACY_REL)
    with tempfile.TemporaryDirectory() as td:
        replay = _closure._emit_ac3_gates(
            v5_head_path=joint_head_path,
            output_path=Path(td) / "frozen_sm120_replay.json",
        )
    replay_metrics = _flatten_gate_metrics(replay)
    committed_metrics = _flatten_gate_metrics(committed)
    shared = sorted(set(replay_metrics) & set(committed_metrics))
    per_metric_abs_diff = {
        k: abs(replay_metrics[k] - committed_metrics[k]) for k in shared
    }
    max_abs_diff = max(per_metric_abs_diff.values()) if per_metric_abs_diff else 0.0
    only_replay = sorted(set(replay_metrics) - set(committed_metrics))
    only_committed = sorted(set(committed_metrics) - set(replay_metrics))
    within = (
        committed is not None
        and bool(shared)
        and max_abs_diff <= FROZEN_REPLAY_TOLERANCE
        and not only_replay
        and not only_committed
    )
    return {
        "frozen_ref": FROZEN_SM120_REF,
        "tolerance": FROZEN_REPLAY_TOLERANCE,
        "committed_artifact_available": committed is not None,
        "replay_status": replay.get("status"),
        "replay_metrics": replay_metrics,
        "committed_metrics": committed_metrics,
        "compared_metric_keys": shared,
        "per_metric_abs_diff": per_metric_abs_diff,
        "max_abs_diff": max_abs_diff,
        "metrics_only_in_replay": only_replay,
        "metrics_only_in_committed": only_committed,
        "within_tolerance": within,
        "frozen_b1646ef7_drift": not within,
    }


def _deferred_leg(closure_target: str, citation: dict, **extra: Any) -> dict:
    leg = {
        "closure_target": closure_target,
        "status": DEFERRED_STATUS,
        "finite_cross_arch_number_asserted": False,
        "ledger_citation": citation,
    }
    leg.update(extra)
    return leg


def emit(
    *,
    sm86_head_path: Path,
    ledger_path: Path,
    joint_head_path: Path,
    v4_head_path: Path,
    accuracy_out: Path,
    baseline_out: Path,
    robustness_out: Path,
    falsification_out: Path,
) -> dict[str, Any]:
    sm86_head = json.loads(sm86_head_path.read_text(encoding="utf-8"))
    sm86_head_commit = sm86_head.get("source_commit")
    citation = _ledger_citation(ledger_path)
    deferred_legs = [
        _deferred_leg(
            "cross_arch_transfer", citation, direction="sm_86_to_sm_120",
        ),
        _deferred_leg(
            "cross_arch_transfer", citation, direction="sm_120_to_sm_86",
        ),
        _deferred_leg("same_74_baseline_comparison", citation),
        _deferred_leg("same_74_falsification_verdict", citation),
    ]

    # 1. Accuracy gates over the sm_86 slice + the frozen-30 regression guard.
    accuracy = _closure._emit_ac3_gates(
        v5_head_path=sm86_head_path, output_path=accuracy_out,
    )
    frozen_guard = _same_30_frozen_regression_guard(joint_head_path)
    accuracy["scope"] = "sm_86_only"
    accuracy["per_slice"] = {
        "sm_86_only": {
            "status": accuracy.get("status"),
            "admitted_count": accuracy.get("admitted_count"),
            "gates": accuracy.get("gates"),
            "verdict": accuracy.get("verdict"),
        },
        "same_30_frozen_b1646ef7": frozen_guard,
    }
    accuracy["deferred_cross_arch"] = {
        "status": DEFERRED_STATUS,
        "slices": [
            "same_74", "same_30_under_same74_head", "sm_120_only",
            "holdout_same_74",
        ],
        "ledger_citation": citation,
    }
    accuracy_out.write_text(
        json.dumps(accuracy, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    # 2. Baseline comparison over the sm_86 slice. Bundles are selected from
    # the ledger's admitted set (sha256-verified, collision-proof) rather than
    # a filesystem glob, so the per-cell baseline columns bind to the audited
    # bundle for each admitted cell.
    bundle_lookup = _ledger_admitted_bundle_lookup(ledger_path)
    v4_head = (
        json.loads(v4_head_path.read_text(encoding="utf-8"))
        if v4_head_path.is_file() else None
    )
    v5_ensemble = _nugteren._refit_v5_ensemble_from_head(sm86_head)
    rows = _nugteren.build_comparison_table(
        v5_head=sm86_head,
        v4_head=v4_head,
        bundle_lookup=bundle_lookup,
        primary_root=REPO_ROOT / "workloads" / "simbench_bw" / "results",
        fallback_root=REPO_ROOT / "workloads" / "portable_lcd" / "results",
        v5_ensemble=v5_ensemble,
    )
    aggregate = _nugteren._aggregate_metrics(rows)
    baseline = {
        "schema_version": 1,
        "scope": "sm_86_only",
        "baseline_set": [
            "v5_prediction", "nugteren_prediction", "cohort_mean_prediction",
            "always_hit_prediction", "uncalibrated_walker_prediction",
            "v4_platt_prediction",
        ],
        "source_commit": _closure._git_head_sha(),
        "source_v5_head_commit": sm86_head_commit,
        "row_count": len(rows),
        "rows": rows,
        "aggregate_metrics": aggregate,
        "nugteren_2014_reference": (
            "Nugteren et al., 'A Detailed GPU Cache Model Based on Reuse "
            "Distance Theory', HPCA 2014. Clean-room analytical "
            "reimplementation over the sm_86 slice."
        ),
        "deferred_cross_arch": {
            "status": DEFERRED_STATUS,
            "note": (
                "the joint same-74 baseline comparison (30 sm_120 + the sm_86 "
                "cells) is the sm_120 session's deliverable"
            ),
            "ledger_citation": citation,
        },
    }
    baseline_out.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    # 3. Robustness (LOOCV vs holdout, LOSO) over the sm_86 slice.
    robustness = _robustness.emit_robustness(
        v5_head_path=sm86_head_path, output_path=robustness_out,
    )
    robustness["scope"] = "sm_86_only"
    robustness_out.write_text(
        json.dumps(robustness, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    # 4. Falsification verdict over the sm_86 slice (informational). The
    # authoritative joint same-74 verdict + the cross-architecture transfer
    # legs are recorded deferred with backing citations; no finite
    # cross-architecture number is asserted.
    falsification = _closure._emit_falsification(
        v5_head_path=sm86_head_path,
        output_path=falsification_out,
        v4_head_path=v4_head_path if v4_head_path.is_file() else None,
        bundle_lookup=bundle_lookup or None,
    )
    falsification["scope"] = "sm_86_only"
    falsification["verdict_slice"] = "sm_86_only"
    falsification["verdict_is_informational"] = True
    # The authoritative same-74 falsification verdict is the sm_120 session's
    # to make; this loop records it as deferred rather than research_success /
    # verdict_falsification over any same-74 slice.
    falsification["closure_path"] = DEFERRED_STATUS
    falsification["missed_closure_targets"] = deferred_legs
    falsification["cross_arch_transfer"] = {
        "status": DEFERRED_STATUS,
        "directions": ["sm_86_to_sm_120", "sm_120_to_sm_86"],
        "finite_cross_arch_number_asserted": False,
        "ledger_citation": citation,
    }
    falsification_out.write_text(
        json.dumps(falsification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "robustness": robustness,
        "falsification": falsification,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sm86-head", default=DEFAULT_SM86_HEAD)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--joint-head", default=DEFAULT_JOINT_HEAD)
    parser.add_argument("--v4-head", default=DEFAULT_V4_HEAD)
    parser.add_argument("--accuracy-output", default=DEFAULT_ACCURACY_OUT)
    parser.add_argument("--baseline-output", default=DEFAULT_BASELINE_OUT)
    parser.add_argument("--robustness-output", default=DEFAULT_ROBUSTNESS_OUT)
    parser.add_argument(
        "--falsification-output", default=DEFAULT_FALSIFICATION_OUT,
    )
    args = parser.parse_args(argv)
    result = emit(
        sm86_head_path=Path(args.sm86_head),
        ledger_path=Path(args.ledger),
        joint_head_path=Path(args.joint_head),
        v4_head_path=Path(args.v4_head),
        accuracy_out=Path(args.accuracy_output),
        baseline_out=Path(args.baseline_output),
        robustness_out=Path(args.robustness_output),
        falsification_out=Path(args.falsification_output),
    )
    acc = result["accuracy"]
    fals = result["falsification"]
    rob = result["robustness"]
    guard = acc["per_slice"]["same_30_frozen_b1646ef7"]
    print(
        f"[v5_sm86_closure] accuracy.status={acc.get('status')} "
        f"admitted={acc.get('admitted_count')} "
        f"frozen_b1646ef7_drift={guard.get('frozen_b1646ef7_drift')} "
        f"max_abs_diff={guard.get('max_abs_diff')} "
        f"baseline.rows={result['baseline'].get('row_count')} "
        f"robustness.status={rob.get('status')} "
        f"falsification.verdict={fals.get('verdict')} "
        f"closure_path={fals.get('closure_path')}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
