"""Centralized v5 selected-class dispatch.

Reads the selected model class from a v5 head's provenance and returns
the matching factory + ensemble + selected-class diagnostics. Every
official v5 artifact (closure scanner / robustness scanner / active-path
comparison driver / Nugteren baseline / comparison_runner v5 mode) MUST
go through this helper rather than hardcoding
``IsotonicRegressionOnFrozenProjection`` so that when pygam / LightGBM /
XGBoost wins the composite score, the official artifacts evaluate the
ACTUALLY-SELECTED head and not a stale IR fallback.

Public API:

  - ``resolve_selected_class_name(v5_head)`` -> str: the canonical class
    name pulled from ``selected_class.name`` with fallback to the first
    available entry in ``MODEL_CLASS_HIERARCHY`` for legacy heads that
    omit the selected_class block.

  - ``resolve_selected_factory(v5_head)`` -> callable: maps the selected
    class name through ``_model_class_factory_for`` (lazily imports the
    fitter to avoid circular-import problems).

  - ``rehydrate_cells_from_v5_head(v5_head)`` -> list[dict]: shared cell
    rehydration logic (deduplicates the implementations spread across
    the 4 emitter modules).

  - ``build_selected_ensemble(v5_head, cells=None)`` -> dict: returns
    {ensemble, factory, selected_class_name, cells, status, reason}
    with status in {"ok", "indeterminate_dependencies_absent",
    "indeterminate_no_cells", "indeterminate_setup_failure"} so callers
    can branch on a single sentinel instead of catching exceptions
    everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw import monotonic_calibration_head as mch  # noqa: E402


def resolve_selected_class_name(v5_head: dict) -> str:
    """Pull the selected class name from the v5 head's provenance.

    Order of preference:
      1. ``v5_head["selected_class"]["name"]`` (the composite-scored
         winner; the canonical source).
      2. ``v5_head["candidate_scores"][0]["name"]`` (the first scored
         candidate; legacy heads emitted before selected_class landed).
      3. First available entry in ``MODEL_CLASS_HIERARCHY`` (fallback).
    """
    sc = v5_head.get("selected_class")
    if isinstance(sc, dict) and sc.get("name"):
        return str(sc["name"])
    candidates = v5_head.get("candidate_scores")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict) and first.get("name"):
            return str(first["name"])
    return mch.MODEL_CLASS_HIERARCHY[0]["name"]


def resolve_selected_factory(v5_head: dict) -> Any:
    """Map the selected class name to its constructor via the fitter's
    canonical factory lookup. Lazy import keeps this module free of the
    fitter's transitive imports for callers that only need name resolution."""
    name = resolve_selected_class_name(v5_head)
    from tools.local_labels.fit_monotonic_calibration_head_v5 import (
        _model_class_factory_for,
    )
    return _model_class_factory_for(name)


def rehydrate_cells_from_v5_head(v5_head: dict) -> list[dict]:
    """Build per-cell records (with features) from
    ``v5_head.cohort_scope.admitted``. Cells lacking features are dropped
    (the v5 head may have been emitted with features omitted to keep the
    JSON small; this helper requires features to be present for the
    consumer to refit).

    Split-aware heads (``fit_scope.split_aware``): only the FIT-POOL cells
    (calibration+validation) are rehydrated — every dispatch consumer
    refits the ensemble from this list, and a split-aware head's sealed
    holdout cells must never enter a refit (they are evaluated exactly
    once by the gate emitters' dedicated holdout blocks). Identities are
    mixed-arity (variant_id-widened) so sibling register-cap variants
    never collapse."""
    split_aware = bool((v5_head.get("fit_scope") or {}).get("split_aware"))
    admitted = v5_head.get("cohort_scope", {}).get("admitted", [])
    cells: list[dict] = []
    for c in admitted:
        if not c.get("features"):
            continue
        if split_aware and c.get("split_assignment") == "holdout":
            continue
        variant_id = str(c.get("variant_id", "") or "")
        cells.append({
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


def build_selected_ensemble(
    v5_head: dict,
    cells: list[dict] | None = None,
) -> dict[str, Any]:
    """Build the v5 ensemble using the selected class's factory.

    Returns a dict with ``status`` field in:
      - ``"ok"``: ensemble + factory + selected_class_name + cells populated
      - ``"indeterminate_dependencies_absent"``: factory or fit raises ImportError
      - ``"indeterminate_no_cells"``: zero cells with features after rehydration
      - ``"indeterminate_setup_failure"``: any other RuntimeError from fit

    Callers use the ``status`` sentinel to branch instead of catching
    exceptions. The ensemble is None when status != "ok".
    """
    out: dict[str, Any] = {
        "status": "indeterminate_setup_failure",
        "ensemble": None,
        "factory": None,
        "selected_class_name": None,
        "cells": None,
        "reason": "",
    }
    selected_name = resolve_selected_class_name(v5_head)
    out["selected_class_name"] = selected_name
    try:
        factory = resolve_selected_factory(v5_head)
    except (ImportError, NotImplementedError) as exc:
        out["status"] = "indeterminate_dependencies_absent"
        out["reason"] = (
            f"factory lookup for {selected_name!r} failed: {exc!s}"
        )
        return out
    out["factory"] = factory
    if cells is None:
        cells = rehydrate_cells_from_v5_head(v5_head)
    out["cells"] = cells
    if not cells:
        out["status"] = "indeterminate_no_cells"
        out["reason"] = (
            "v5_head.cohort_scope.admitted has zero cells with features"
        )
        return out
    ensemble = mch.PerBucketCalibratorEnsemble(model_class_factory=factory)
    try:
        ensemble.fit(cells)
    except ImportError as exc:
        out["status"] = "indeterminate_dependencies_absent"
        out["reason"] = (
            f"ensemble fit raised ImportError ({selected_name}): {exc!s}"
        )
        return out
    except RuntimeError as exc:
        out["status"] = "indeterminate_setup_failure"
        out["reason"] = (
            f"ensemble fit raised RuntimeError ({selected_name}): {exc!s}"
        )
        return out
    out["status"] = "ok"
    out["ensemble"] = ensemble
    return out


__all__ = [
    "build_selected_ensemble",
    "rehydrate_cells_from_v5_head",
    "resolve_selected_class_name",
    "resolve_selected_factory",
]
