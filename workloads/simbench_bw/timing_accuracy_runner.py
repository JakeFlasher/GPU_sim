#!/usr/bin/env python3
"""Timing-specific accuracy runner for the cycle prediction bundle.

This scorer joins prediction rows to the scoped long-duration timing cohort on
a timing-only key: slug, shape_point, arch, binary identity, kernel signature,
source commit, and metric target. The cache-layer dimension is intentionally
absent because this target is elapsed cycles, not a per-cache-layer fraction.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import timing_prediction_contract as contract  # noqa: E402
from workloads.simbench_bw.timing_cycle_predictor import (  # noqa: E402
    DEMOTED_MODEL_VERSIONS,
    DEPLOYABLE_CHANNEL,
    DIAGNOSTIC_SAME_SLUG_CHANNEL,
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
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "timing_accuracy_report_sm_86.json"
)

JOIN_KEY_FIELDS = (
    "slug",
    "shape_point",
    "arch",
    "binary_sha256",
    "kernel_signature_sha256",
    "source_commit",
    "metric_target",
)
FORBIDDEN_JOIN_FIELDS = ("cache_layer",)
METRIC_TARGET = "cycles"
RESAMPLE_SEED = 20260531


class TimingAccuracyError(ValueError):
    """Raised when predictions cannot be joined or scored."""


# Per-cell prediction-object keys that mark a same-slug / neighbor
# derivation. A deployable prediction carrying any of these is a
# diagnostic value masquerading as the headline and is rejected.
_SAME_SLUG_MARKER_FIELDS = (
    "channel",
    "neighbor_slug",
    "neighbor_shape_point",
    "neighbor_work_ratio",
)

# Value-level substitution tolerances. Marker fields are trivially
# strippable, so the guard rejects a deployable prediction whose
# NUMBERS sit in a forbidden proximity band around the demoted
# same-slug value. The band is SUBSTANTIVE, not exact-copy-sized: an
# epsilon-perturbed copy (x1.0001, x1.001, x1.01) is still a
# same-slug-derived headline and must be rejected. Calibration: the
# committed honest bundle's MINIMUM deployable-vs-diagnostic relative
# gap is ~2.417e-2 (pinned with headroom by
# test_committed_bundle_passes_value_substitution_audit); the band is
# a fixed 1e-2 — no larger than half the measured minimum gap, and
# wide enough that any disguise tighter than 1% relative is caught.
# This arm is the LOAD-BEARING rejection and is independent of the
# stored per-cell error field. The abs-log-error arm additionally
# catches a forged stored error equal to the diagnostic's.
_SAME_SLUG_VALUE_REL_TOLERANCE = 1e-2
_SAME_SLUG_ERROR_ABS_TOLERANCE = 1e-12


def assert_prediction_bundle_deployable_channel(
    predictions: dict[str, Any],
) -> None:
    """Fail-closed pre-scoring guard: the scored bundle must be the
    demotion-aware deployable ridge-on-roofline channel.

    Rejects (typed reasons in the message):
      - a bundle or cell whose model_version is a demoted
        same-slug-substitution version;
      - a bundle missing the deployable-channel declaration, the
        demoted-channels declaration, or the typed cohort-diversity
        limitation block;
      - a cell whose prediction_channel is not the deployable channel
        (e.g. the demoted diagnostic channel promoted to headline);
      - a cell missing the demoted same-slug diagnostics block this
        guard audits, or whose non-null diagnostic does not carry the
        diagnostic channel marker;
      - a deployable prediction object carrying same-slug/neighbor
        marker fields;
      - VALUE-level substitution: a deployable predicted_cycles equal
        (within tolerance) to the demoted diagnostic value, or a
        cell-level validation_abs_log_error equal to the diagnostic's
        (the rounded-value disguise) — markers are strippable, the
        numbers are the contraband.
    """
    model_version = predictions.get("model_version")
    if model_version in DEMOTED_MODEL_VERSIONS:
        raise TimingAccuracyError(
            "prediction_bundle_demoted_model_version: refusing to score "
            f"bundle model_version={model_version!r}; its deployable "
            "headline substitutes same-slug truth labels."
        )
    if predictions.get("deployable_channel") != DEPLOYABLE_CHANNEL:
        raise TimingAccuracyError(
            "prediction_bundle_channel_declaration_missing: the bundle "
            f"must declare deployable_channel={DEPLOYABLE_CHANNEL!r}; "
            f"got {predictions.get('deployable_channel')!r}."
        )
    demoted_channels = predictions.get("demoted_channels") or []
    if DIAGNOSTIC_SAME_SLUG_CHANNEL not in demoted_channels:
        raise TimingAccuracyError(
            "prediction_bundle_demoted_channels_missing: the bundle "
            f"must declare {DIAGNOSTIC_SAME_SLUG_CHANNEL!r} in "
            "demoted_channels so the value-substitution audit below "
            "has the diagnostic channel to compare against."
        )
    limitation = predictions.get("cohort_limitation")
    if (
        not isinstance(limitation, dict)
        or not limitation.get("typed_reason")
        or limitation.get("same_slug_substitution_authorized") is not False
    ):
        raise TimingAccuracyError(
            "prediction_bundle_demotion_limitation_missing: the bundle "
            "must carry the typed cohort_limitation block with "
            "same_slug_substitution_authorized=false."
        )
    for row in predictions.get("cells", []):
        cell_id = f"{row.get('slug')}/{row.get('shape_point')}"
        if row.get("model_version") in DEMOTED_MODEL_VERSIONS:
            raise TimingAccuracyError(
                "prediction_cell_demoted_model_version: cell "
                f"{cell_id} carries a demoted model_version."
            )
        channel = row.get("prediction_channel")
        if channel != DEPLOYABLE_CHANNEL:
            raise TimingAccuracyError(
                "prediction_cell_channel_not_deployable: cell "
                f"{cell_id} declares prediction_channel={channel!r} "
                f"(diagnostic channels such as "
                f"{DIAGNOSTIC_SAME_SLUG_CHANNEL!r} cannot be scored)."
            )
        markers = sorted(
            set(_SAME_SLUG_MARKER_FIELDS).intersection(
                row.get("prediction", {})
            )
        )
        if markers:
            raise TimingAccuracyError(
                "prediction_cell_same_slug_sourced: cell "
                f"{cell_id} deployable prediction carries same-slug "
                f"marker fields {markers}."
            )
        diagnostics = row.get("diagnostics")
        if (
            not isinstance(diagnostics, dict)
            or "same_slug_scaled_neighbor" not in diagnostics
        ):
            raise TimingAccuracyError(
                "prediction_cell_diagnostics_block_missing: cell "
                f"{cell_id} lacks the demoted same-slug diagnostics "
                "block the value-substitution audit requires."
            )
        diag = diagnostics["same_slug_scaled_neighbor"]
        if diag is None:
            continue
        if diag.get("channel") != DIAGNOSTIC_SAME_SLUG_CHANNEL:
            raise TimingAccuracyError(
                "prediction_cell_diagnostic_channel_mismatch: cell "
                f"{cell_id} diagnostic carries channel="
                f"{diag.get('channel')!r}; expected "
                f"{DIAGNOSTIC_SAME_SLUG_CHANNEL!r}."
            )
        # VALUE-level substitution audit: marker fields are trivially
        # strippable, so compare the numbers themselves. A deployable
        # prediction equal to the demoted same-slug value (within
        # tolerance) IS the retired v3 headline wearing v4 clothes.
        deploy_predicted = float(row["prediction"]["predicted_cycles"])
        diag_predicted = float(diag["predicted_cycles"])
        rel_gap = abs(deploy_predicted - diag_predicted) / max(
            abs(diag_predicted), 1.0
        )
        if rel_gap <= _SAME_SLUG_VALUE_REL_TOLERANCE:
            raise TimingAccuracyError(
                "prediction_cell_same_slug_value_substitution: cell "
                f"{cell_id} deployable predicted_cycles "
                f"{deploy_predicted!r} equals the demoted same-slug "
                f"diagnostic value {diag_predicted!r} within relative "
                f"tolerance {_SAME_SLUG_VALUE_REL_TOLERANCE}; the "
                "deployable headline may not be computed by same-slug "
                "substitution."
            )
        cell_err = row.get("validation_abs_log_error")
        diag_err = diag.get("validation_abs_log_error")
        if (
            cell_err is not None
            and diag_err is not None
            and abs(float(cell_err) - float(diag_err))
            <= _SAME_SLUG_ERROR_ABS_TOLERANCE
        ):
            raise TimingAccuracyError(
                "prediction_cell_same_slug_value_substitution: cell "
                f"{cell_id} cell-level validation_abs_log_error "
                f"{cell_err!r} equals the demoted diagnostic's "
                f"{diag_err!r} (rounded-value disguise); the "
                "deployable headline may not be computed by same-slug "
                "substitution."
            )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


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


def join_key_for_label(cell: dict[str, Any]) -> dict[str, str]:
    return {
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "arch": cell["arch"],
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "metric_target": METRIC_TARGET,
    }


def _assert_timing_join_key(join_key: dict[str, Any]) -> None:
    unexpected = set(join_key) - set(JOIN_KEY_FIELDS)
    missing = set(JOIN_KEY_FIELDS) - set(join_key)
    forbidden = set(FORBIDDEN_JOIN_FIELDS).intersection(join_key)
    if unexpected or missing or forbidden:
        raise TimingAccuracyError(
            "invalid timing join key: "
            f"unexpected={sorted(unexpected)}, missing={sorted(missing)}, "
            f"forbidden={sorted(forbidden)}"
        )


def _prediction_lookup(predictions: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in predictions.get("cells", []):
        join_key = row.get("join_key", {})
        _assert_timing_join_key(join_key)
        key = tuple(join_key[field] for field in JOIN_KEY_FIELDS)
        if key in lookup:
            raise TimingAccuracyError(f"duplicate prediction join key: {key!r}")
        lookup[key] = row
    return lookup


def joined_scored_cells(
    predictions: dict[str, Any], cohort: dict[str, Any]
) -> list[dict[str, Any]]:
    lookup = _prediction_lookup(predictions)
    scored: list[dict[str, Any]] = []
    for label_cell in cohort.get("cells", []):
        join_key = join_key_for_label(label_cell)
        key = tuple(join_key[field] for field in JOIN_KEY_FIELDS)
        pred_cell = lookup.get(key)
        if pred_cell is None:
            raise TimingAccuracyError(f"missing prediction for timing key: {key!r}")
        predicted = float(pred_cell["prediction"]["predicted_cycles"])
        observed = float(label_cell["labels"]["gpc__cycles_elapsed.max"]["value"])
        if predicted <= 0.0 or observed <= 0.0:
            raise TimingAccuracyError("cycle predictions and labels must be positive")
        scored.append(
            {
                "slug": label_cell["slug"],
                "shape_point": label_cell["shape_point"],
                "arch": label_cell["arch"],
                "regime": pred_cell.get("regime", "long"),
                "join_key": join_key,
                "predicted_cycles": predicted,
                "observed_cycles": observed,
                "signed_log_ratio": math.log(predicted / observed),
            }
        )
    return scored


def _coverage_block(doc: dict[str, Any], n_scored: int) -> dict[str, Any]:
    gate = contract.publication_gate(doc)
    smoke = contract.smoke_tier(doc)
    return {
        "n_scored": n_scored,
        "publication_min_scored_cells": gate["min_scored_cells"],
        "smoke_min_scored_cells": smoke["min_scored_cells"],
        "publication_floor_met": n_scored >= gate["min_scored_cells"],
        "smoke_floor_met": n_scored >= smoke["min_scored_cells"],
    }


def build_accuracy_report_from_objects(
    *,
    predictions: dict[str, Any],
    cohort: dict[str, Any],
    contract_doc: dict[str, Any],
    prediction_path: Path = DEFAULT_PREDICTIONS,
    cohort_path: Path = DEFAULT_COHORT,
    source_commit: str | None = None,
) -> dict[str, Any]:
    assert_prediction_bundle_deployable_channel(predictions)
    scored_cells = joined_scored_cells(predictions, cohort)
    pairs = [
        (cell["predicted_cycles"], cell["observed_cycles"]) for cell in scored_cells
    ]
    gate_cfg = contract.publication_gate(contract_doc)
    scores = contract.score_cycle_metrics(
        pairs,
        resamples=gate_cfg["bootstrap_resamples"],
        confidence=gate_cfg["bootstrap_ci_confidence"],
        seed=RESAMPLE_SEED,
    )
    gate = contract.evaluate_publication_gate(contract_doc, scores)
    coverage = _coverage_block(contract_doc, len(scored_cells))
    result_kind = "publication_pass" if gate["passes"] else "typed_unavailable"
    typed_reason = None if gate["passes"] else "cycle_publication_gate_not_met"
    return {
        "schema_version": 1,
        "artifact_kind": "timing_accuracy_report",
        "arch": cohort["arch"],
        "source_commit": source_commit or _source_commit(),
        "metric_target": METRIC_TARGET,
        "prediction_mode": "leave_one_out_validation",
        "scored_channel": DEPLOYABLE_CHANNEL,
        "demoted_channels_present": list(
            predictions.get("demoted_channels", [])
        ),
        "prediction_model_version": predictions.get("model_version"),
        "prediction_bundle": _repo_rel(prediction_path),
        "cohort_artifact": _repo_rel(cohort_path),
        "scoring_method": contract.PINNED_CYCLE_ERROR_METHOD,
        "join_key_fields": list(JOIN_KEY_FIELDS),
        "forbidden_join_fields": list(FORBIDDEN_JOIN_FIELDS),
        "bootstrap": {
            "resamples": gate_cfg["bootstrap_resamples"],
            "ci_confidence": gate_cfg["bootstrap_ci_confidence"],
            "seed": RESAMPLE_SEED,
        },
        "coverage": coverage,
        "scores": scores,
        "publication_gate": gate,
        "result_kind": result_kind,
        "typed_unavailable_reason": typed_reason,
        "evidence_status": "Offline-verified",
        "cells": scored_cells,
    }


def build_accuracy_report(
    *,
    prediction_path: Path = DEFAULT_PREDICTIONS,
    cohort_path: Path = DEFAULT_COHORT,
    contract_path: Path = contract.DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    predictions = _load_json(prediction_path)
    cohort = _load_json(cohort_path)
    doc = contract.load_contract(contract_path)
    return build_accuracy_report_from_objects(
        predictions=predictions,
        cohort=cohort,
        contract_doc=doc,
        prediction_path=prediction_path,
        cohort_path=cohort_path,
    )


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--contract", type=Path, default=contract.DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_accuracy_report(
        prediction_path=args.predictions,
        cohort_path=args.cohort,
        contract_path=args.contract,
    )
    write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
