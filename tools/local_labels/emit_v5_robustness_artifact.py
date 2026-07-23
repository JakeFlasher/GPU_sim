"""Emit the v5 robustness artifact.

Implements the robustness gates row from the goal tracker:

  - LOOCV-vs-holdout primary-metric gap: refit the v5 ensemble in
    leave-one-out CV on the admitted cells, compute the LOOCV primary
    score (geomean relative error), compare to the holdout primary
    score (geomean over cells whose roster split tag is 'holdout').
    Pass when |LOOCV - holdout| / max(LOOCV, eps) <= 0.20.

  - Leave-one-slug-out (LOSO): for each distinct slug present in the
    admitted cohort, refit the v5 ensemble with that slug's cells
    removed, evaluate the headline geomean on the held-out cells, and
    compute the relative change to the full-cohort headline geomean.
    Pass when max_relative_change <= 0.25.

Emits ``data/local_labels/calibration_head_v5_robustness.json`` with
status=ok when admitted_count is sufficient OR
status=indeterminate_insufficient_data with a structured placeholder
when the cohort is too small for the scanner to be meaningful.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw import monotonic_calibration_head as mch  # noqa: E402


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


DEFAULT_V5_HEAD = "data/local_labels/calibration_head_v5.json"
DEFAULT_OUTPUT = "data/local_labels/calibration_head_v5_robustness.json"


# Thresholds per the goal tracker's robustness gates row.
LOOCV_HOLDOUT_RELATIVE_GAP_MAX = 0.20
LOSO_MAX_RELATIVE_CHANGE_MAX = 0.25

# Minimum admitted-count below which the scanner emits indeterminate.
MIN_ADMITTED_FOR_LOOCV = 3
MIN_ADMITTED_FOR_LOSO = 4


def _skipped_cell_reasons_from_head(head: dict) -> dict[str, str]:
    """Build a structured map of cohort-skipped cells -> typed_reason
    from the v5 head's cohort_scope. Indeterminate gate-artifact arms
    embed this map so the cohort-coverage scanner can verify the
    indeterminate state has per-cell linkage to the v5 head's
    skipped list per the audit-visibility contract pinned by the
    cohort-coverage scanner."""
    skipped = head.get("cohort_scope", {}).get("skipped", []) or []
    out: dict[str, str] = {}
    for c in skipped:
        key = (
            f"{c.get('arch', '')}/"
            f"{c.get('slug', '')}/"
            f"{c.get('shape_point', '')}"
        )
        out[key] = c.get("typed_reason", "unknown")
    return out


def _rehydrate_cells(v5_head: dict) -> list[dict]:
    admitted = v5_head.get("cohort_scope", {}).get("admitted", [])
    cells: list[dict] = []
    for c in admitted:
        if not c.get("features"):
            continue
        variant_id = str(c.get("variant_id", "") or "")
        cells.append({
            # Mixed-arity identity: sibling register-cap variants share their
            # (slug, shape_point) name, so the identity widens with the
            # canonical variant_id ("" for a base cell).
            "identity": (c["arch"], c["slug"], c["shape_point"], variant_id),
            "arch": c["arch"],
            "family_bucket": c["family_bucket"],
            "slug": c["slug"],
            "shape_point": c["shape_point"],
            "variant_id": variant_id,
            "split_assignment": c.get("split_assignment"),
            "features": c["features"],
            "label": float(c["label"]),
        })
    return cells


def _split_fit_and_holdout(
    head: dict, cells: list[dict],
) -> tuple[list[dict], list[dict], bool]:
    """Partition the rehydrated cells into (fit_pool, holdout, split_aware).

    A split-aware head (``fit_scope.split_aware``) records every admitted
    cell's ``split_assignment``; the fit pool is the calibration+validation
    subset (everything the ensemble was fit on) and the holdout list is the
    sealed cells the gates evaluate exactly once. A legacy head without
    fit_scope yields (cells, [], False) and the caller falls back to the
    roster-split holdout path.
    """
    fit_scope = head.get("fit_scope") or {}
    if not fit_scope.get("split_aware"):
        return cells, [], False
    fit_pool = [c for c in cells if c.get("split_assignment") != "holdout"]
    holdout = [c for c in cells if c.get("split_assignment") == "holdout"]
    return fit_pool, holdout, True


def _loocv_primary_score(
    cells: list[dict],
    factory: Any | None = None,
) -> float | None:
    """Refit the ensemble n times (leave-one-out), aggregate the
    out-of-fold predictions, compute geomean_relative_error.

    ``factory`` is the selected-class factory (from
    ``v5_selected_class_dispatch.resolve_selected_factory``). When omitted,
    falls back to IR-on-projection only for backwards compatibility with
    callers that have no v5 head; official artifact callers MUST pass the
    factory so non-IR selected heads are evaluated correctly.
    """
    if factory is None:
        factory = mch.IsotonicRegressionOnFrozenProjection
    preds: list[float] = []
    labels: list[float] = []
    for i in range(len(cells)):
        train = cells[:i] + cells[i + 1:]
        test = cells[i]
        if not train:
            continue
        ens = mch.PerBucketCalibratorEnsemble(
            model_class_factory=factory,
        )
        try:
            ens.fit(train)
            p = ens.predict(
                test["arch"], test["family_bucket"], test["features"],
            )
        except (RuntimeError, ImportError):
            continue
        preds.append(float(p))
        labels.append(float(test["label"]))
    if not preds:
        return None
    return mch.geomean_relative_error(preds, labels)


def _holdout_primary_score(
    cells: list[dict],
    factory: Any | None = None,
    holdout: list[dict] | None = None,
) -> tuple[float | None, str]:
    """Compute geomean over the holdout subset.

    Split-aware path (``holdout`` supplied by the caller from the head's
    fit_scope partition): the ensemble is fit on ``cells`` — the
    calibration+validation fit pool, which by construction excludes the
    holdout — and predicts the sealed holdout cells exactly once. The
    legacy path (``holdout=None``) derives holdout membership from the
    roster split index and fits on the full cell list (the pre-refit
    behavior, kept for heads without fit_scope).

    Returns (geomean, status_str). status is 'ok' when a holdout subset
    exists, 'no_holdout_cells' when none, 'roster_unavailable' on import
    error.

    ``factory`` is the selected-class factory. When omitted falls back to
    IR-on-projection (backwards compatibility); official artifact callers
    pass the factory so non-IR selected heads are evaluated correctly.
    """
    if factory is None:
        factory = mch.IsotonicRegressionOnFrozenProjection
    if holdout is None:
        try:
            from workloads.simbench_bw.comparison_runner import (
                _load_roster_split_index,
            )
        except ImportError:
            return None, "roster_unavailable"
        split_index = _load_roster_split_index()
        # Key on the cell's own arch: the roster split tags are sm_120-cohort
        # assignments (see comparison_runner._ROSTER_ARCH), so a non-sm_120
        # cell that reuses a roster (slug, shape_point) is correctly excluded
        # from the holdout subset rather than inheriting the sm_120 split.
        holdout = [
            c for c in cells
            if split_index.get((c["arch"], c["slug"], c["shape_point"]))
            == "holdout"
        ]
    if not holdout:
        return None, "no_holdout_cells"
    ensemble = mch.PerBucketCalibratorEnsemble(
        model_class_factory=factory,
    )
    try:
        ensemble.fit(cells)
    except (RuntimeError, ImportError):
        return None, "ensemble_fit_failed"
    preds = []
    labels = []
    skipped_unseen_bucket = 0
    for c in holdout:
        try:
            p = ensemble.predict(
                c["arch"], c["family_bucket"], c["features"],
            )
        except RuntimeError:
            # Holdout cell whose (arch, family_bucket) never appeared in the
            # fit pool: the ensemble has no head to route through. Counted
            # (not silently dropped) via the status string below.
            skipped_unseen_bucket += 1
            continue
        preds.append(float(p))
        labels.append(float(c["label"]))
    if not preds:
        return None, "no_scoreable_holdout_cells"
    status = "ok" if skipped_unseen_bucket == 0 else (
        f"ok_with_{skipped_unseen_bucket}_unseen_bucket_holdout_cells"
    )
    return mch.geomean_relative_error(preds, labels), status


def _loso_max_relative_change(
    cells: list[dict],
    full_cohort_primary: float,
    factory: Any | None = None,
) -> dict[str, Any]:
    """For each distinct slug, refit without that slug + evaluate
    headline geomean over the FULL ensemble on the held-out cells.

    Returns {max_relative_change: float, per_slug: {slug: rel_change}}.

    ``factory`` is the selected-class factory; falls back to IR-on-projection
    only for backwards-compatible callers.
    """
    if factory is None:
        factory = mch.IsotonicRegressionOnFrozenProjection
    by_slug: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_slug[c["slug"]].append(c)
    per_slug: dict[str, float] = {}
    for slug, members in by_slug.items():
        train = [c for c in cells if c["slug"] != slug]
        if not train:
            continue
        ens = mch.PerBucketCalibratorEnsemble(
            model_class_factory=factory,
        )
        try:
            ens.fit(train)
        except (RuntimeError, ImportError):
            continue
        preds = []
        labels = []
        for c in train:
            try:
                p = ens.predict(
                    c["arch"], c["family_bucket"], c["features"],
                )
            except RuntimeError:
                continue
            preds.append(float(p))
            labels.append(float(c["label"]))
        if not preds:
            continue
        new_primary = mch.geomean_relative_error(preds, labels)
        denom = max(abs(full_cohort_primary), 1e-12)
        rel_change = abs(new_primary - full_cohort_primary) / denom
        per_slug[slug] = rel_change
    if not per_slug:
        return {"max_relative_change": None, "per_slug": {}}
    return {
        "max_relative_change": max(per_slug.values()),
        "per_slug": per_slug,
    }


def emit_robustness(
    *, v5_head_path: Path, output_path: Path,
) -> dict[str, Any]:
    head = json.loads(v5_head_path.read_text(encoding="utf-8"))
    all_cells = _rehydrate_cells(head)
    # Split-aware partition: every robustness statistic (LOOCV, LOSO, the
    # full-cohort baseline) is computed over the FIT POOL (cal+val — the
    # cells the head was fit on); the sealed holdout cells are evaluated
    # exactly once by _holdout_primary_score against a fit-pool-fitted
    # ensemble. Legacy heads without fit_scope keep the prior behavior.
    cells, holdout_cells, split_aware = _split_fit_and_holdout(
        head, all_cells,
    )
    n = len(cells)
    if n < MIN_ADMITTED_FOR_LOOCV:
        artifact = {
            "schema_version": 1,
            "status": "indeterminate_insufficient_data",
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
            "minimum_admitted_for_loocv": MIN_ADMITTED_FOR_LOOCV,
            "minimum_admitted_for_loso": MIN_ADMITTED_FOR_LOSO,
            "reason": (
                "Robustness gates need >= 3 admitted cells for a stable "
                "LOOCV primary score and >= 4 for the leave-one-slug-out "
                "max-relative-change scan. The v5 head's cohort_scope "
                "records the admitted cells; this artifact lands as a "
                "placeholder so downstream readers see the robustness gate "
                "status without crashing."
            ),
            "gates": {
                "loocv_vs_holdout_gap_pass": None,
                "loso_max_relative_change_pass": None,
            },
            "skipped_cell_reasons": _skipped_cell_reasons_from_head(head),
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    # Full-cohort primary score for the LOSO baseline. Use the SELECTED
    # class via the centralized dispatch (was: hardcoded IR-on-projection).
    from tools.local_labels.v5_selected_class_dispatch import (
        build_selected_ensemble, resolve_selected_factory,
    )
    try:
        selected_factory = resolve_selected_factory(head)
    except (ImportError, NotImplementedError) as exc:
        # No-fallback contract: when the selected factory cannot be
        # resolved, emit an indeterminate artifact rather than silently
        # refitting IR-on-projection.
        artifact = {
            "schema_version": 1,
            "status": "indeterminate_dependencies_absent",
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
            "selected_class_name": None,
            "reason": (
                f"resolve_selected_factory raised "
                f"{type(exc).__name__}: {exc!s}"
            ),
            "gates": {
                "loocv_vs_holdout_gap_pass": None,
                "loso_max_relative_change_pass": None,
            },
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    dispatch = build_selected_ensemble(head, cells=cells)
    if dispatch["status"] != "ok":
        artifact = {
            "schema_version": 1,
            "status": dispatch["status"],
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
            "selected_class_name": dispatch["selected_class_name"],
            "reason": dispatch["reason"],
            "gates": {
                "loocv_vs_holdout_gap_pass": None,
                "loso_max_relative_change_pass": None,
            },
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    full_ensemble = dispatch["ensemble"]
    full_preds = [
        full_ensemble.predict(c["arch"], c["family_bucket"], c["features"])
        for c in cells
    ]
    full_labels = [c["label"] for c in cells]
    full_cohort_primary = mch.geomean_relative_error(full_preds, full_labels)
    # LOOCV-vs-holdout primary-score gap (the goal tracker's robustness
    # gates row). Pass the SELECTED factory
    # so non-IR heads are evaluated correctly when chosen by composite.
    loocv = _loocv_primary_score(cells, factory=selected_factory)
    holdout, holdout_status = _holdout_primary_score(
        cells,
        factory=selected_factory,
        holdout=holdout_cells if split_aware else None,
    )
    if loocv is not None and holdout is not None:
        denom = max(abs(loocv), 1e-12)
        loocv_holdout_gap = abs(loocv - holdout) / denom
        loocv_pass = (
            "pass" if loocv_holdout_gap <= LOOCV_HOLDOUT_RELATIVE_GAP_MAX
            else "fail"
        )
    else:
        loocv_holdout_gap = None
        loocv_pass = "indeterminate_insufficient_data"
    # LOSO max-relative-change (the second robustness gate).
    if n >= MIN_ADMITTED_FOR_LOSO:
        loso = _loso_max_relative_change(
            cells, full_cohort_primary, factory=selected_factory,
        )
        loso_max = loso["max_relative_change"]
        if loso_max is not None:
            loso_pass = (
                "pass" if loso_max <= LOSO_MAX_RELATIVE_CHANGE_MAX
                else "fail"
            )
        else:
            loso_pass = "indeterminate_insufficient_data"
    else:
        loso = {"max_relative_change": None, "per_slug": {}}
        loso_max = None
        loso_pass = "indeterminate_insufficient_data"
    artifact = {
        "schema_version": 1,
        "status": "ok",
        "admitted_count": n,
        "split_aware": split_aware,
        "n_holdout_cells": len(holdout_cells),
        "source_commit": _git_head_sha(),
        "source_v5_head_commit": head.get("source_commit"),
        "gates": {
            "loocv_primary_score": loocv,
            "holdout_primary_score": holdout,
            "holdout_status": holdout_status,
            "loocv_holdout_relative_gap": loocv_holdout_gap,
            "loocv_holdout_relative_gap_max": LOOCV_HOLDOUT_RELATIVE_GAP_MAX,
            "loocv_vs_holdout_gap_pass": loocv_pass,
            "loso_max_relative_change": loso_max,
            "loso_max_relative_change_max": LOSO_MAX_RELATIVE_CHANGE_MAX,
            "loso_max_relative_change_pass": loso_pass,
            "loso_per_slug": loso["per_slug"],
        },
        "full_cohort_primary_baseline": full_cohort_primary,
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-head", default=DEFAULT_V5_HEAD)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    artifact = emit_robustness(
        v5_head_path=Path(args.v5_head),
        output_path=Path(args.output),
    )
    print(
        f"[v5_robustness] status={artifact.get('status')} "
        f"admitted={artifact.get('admitted_count')} -> {args.output}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
