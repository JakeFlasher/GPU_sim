"""V5 active-path consumer for the validated sm_86 lane.

This module retains the historical per-architecture and transfer-evaluation
implementation for source archaeology, but production execution is now
sm_86-only. Any explicit sm_120 target or v5 head containing an sm_120 row
returns a typed-unavailable JSON report before output planning, calibration,
transfer fitting, or scoring. Historical sm_120 artifacts are never rewritten.

For a pure sm_86 v5 head, the existing consumer path remains unchanged: it
loads the v5 head, applies the selected calibration class, and emits the
scratch comparison artifacts under the existing output-path safety contract.

Historical behavior (not an active claim) wired the v5 monotonic calibration
head into the canonical comparison path and evaluated both transfer directions
(sm_86 -> sm_120 and sm_120 -> sm_86). The retained implementation below is
reachable only for an sm_86-only input after the safety gate.

Output-path safety contract: the per-arch head emissions are DERIVED
VIEWS rehydrated from whatever joint head is supplied — they are NOT the
canonical ledger-driven fitted heads. A historical producer collision
silently overwrote the canonical fitted
``data/local_labels/sm_86_v5_head.json`` with an ``admitted_count: 0``
rehydration stub. All default output paths therefore resolve to the
gitignored scratch directory ``data/local_labels/uncommitted/``, and
writing to ANY git-tracked path (the canonical committed artifacts in
particular) fails closed with ``CommittedOutputPathError`` unless the
explicit ``--allow-committed-paths`` opt-in flag is passed.

The retained transfer-eval contract was:
  - Fit head_A on arch A's calibration split
  - Predict on arch B's holdout cells using head_A
  - transfer_geomean_relative_error = geomean rel err of head_A on arch B
  - transfer_median_absolute_error = median abs err of head_A on arch B
  - epsilon-floored 2x gate against the native head_B result
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw import canonical_features as cf  # noqa: E402
from workloads.simbench_bw import monotonic_calibration_head as mch  # noqa: E402
from tools.local_labels.sm120_template_safety import (  # noqa: E402
    SM120_TEMPLATE_UNAVAILABLE_REASON,
    canonical_arch,
    is_sm120_template_arch,
    unavailable_result,
)


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


DEFAULT_V5_HEAD = "data/local_labels/calibration_head_v5.json"

# Committed canonical artifact paths this module's defaults previously
# clobbered. Writing to any of these (or any other git-tracked path)
# requires the explicit --allow-committed-paths opt-in.
COMMITTED_SM_120_HEAD = "data/local_labels/sm_120_v5_head.json"
COMMITTED_SM_86_HEAD = "data/local_labels/sm_86_v5_head.json"
COMMITTED_COMPARISON_OUT = (
    "data/local_labels/v5_active_path_comparison_bundle.json"
)

# Non-committed scratch defaults. data/local_labels/uncommitted/ is
# gitignored, so a default invocation can never rewrite a committed
# artifact; the .from_joint_rehydration suffix names the derivation so
# the scratch heads cannot be mistaken for the canonical fitted heads.
DEFAULT_SM_120_HEAD_OUT = (
    "data/local_labels/uncommitted/"
    "sm_120_v5_head.from_joint_rehydration.json"
)
DEFAULT_SM_86_HEAD_OUT = (
    "data/local_labels/uncommitted/"
    "sm_86_v5_head.from_joint_rehydration.json"
)
DEFAULT_COMPARISON_OUT = (
    "data/local_labels/uncommitted/v5_active_path_comparison_bundle.json"
)


class CommittedOutputPathError(ValueError):
    """An output path resolves to a committed (git-tracked) artifact and
    ``--allow-committed-paths`` was not passed. Fail-closed guard for the
    historical producer collision that overwrote the canonical fitted
    sm_86 head with an admitted_count=0 rehydration stub."""

    typed_reason = "committed_output_path_requires_opt_in"


def _is_git_tracked(path: Path) -> bool:
    """Best-effort: True iff ``path`` is tracked by the git repository
    that contains it. Returns False when the parent directory does not
    exist, the path is outside any git work tree, or git is unavailable
    (the deterministic canonical-path check in
    ``assert_output_paths_not_committed`` does not depend on git)."""
    parent = path.resolve().parent
    if not parent.is_dir():
        return False
    try:
        subprocess.check_call(
            ["git", "-C", str(parent), "ls-files", "--error-unmatch",
             path.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    return True


def assert_output_paths_not_committed(
    output_paths: list[str], *, allow_committed_paths: bool,
) -> None:
    """Refuse to write committed artifacts without the explicit opt-in.

    Two detection layers: (a) deterministic membership in the known
    canonical committed paths (repo-root anchored), (b) best-effort
    git-tracked probe of each resolved output path. Raises
    ``CommittedOutputPathError`` listing every offending path."""
    if allow_committed_paths:
        return
    canonical = {
        (REPO_ROOT / COMMITTED_SM_120_HEAD).resolve(),
        (REPO_ROOT / COMMITTED_SM_86_HEAD).resolve(),
        (REPO_ROOT / COMMITTED_COMPARISON_OUT).resolve(),
    }
    offenders = [
        raw for raw in output_paths
        if Path(raw).resolve() in canonical or _is_git_tracked(Path(raw))
    ]
    if offenders:
        raise CommittedOutputPathError(
            f"{CommittedOutputPathError.typed_reason}: refusing to write "
            f"committed artifact path(s) {offenders}; pass "
            "--allow-committed-paths to opt in explicitly."
        )


TRANSFER_EPSILON_FLOOR = 0.03
TRANSFER_REL_GATE_MULTIPLIER = 2.0
TRANSFER_MEDIAN_ABS_GATE_MAX = 0.12


def _v5_head_contains_sm120(v5_head: dict[str, Any]) -> bool:
    """Return whether a v5 head requests or contains the blocked lane.

    The v5 artifact may carry the architecture at the top level and/or on
    each admitted cohort row. Treat either marker as authoritative so a
    malformed mixed artifact cannot silently transfer an sm_120 row.
    """
    if is_sm120_template_arch(v5_head.get("target_arch")):
        return True
    cohort_scope = v5_head.get("cohort_scope")
    if not isinstance(cohort_scope, dict):
        return False
    admitted = cohort_scope.get("admitted", [])
    if not isinstance(admitted, list):
        return False
    return any(
        isinstance(cell, dict)
        and is_sm120_template_arch(cell.get("arch"))
        for cell in admitted
    )


def _sm120_template_unavailable_result(
    *,
    target_arch: str | None,
    requested_output_paths: list[str],
) -> dict[str, Any]:
    """Build the non-scoring result for a blocked transfer evaluation."""
    return unavailable_result(
        operation="run_v5_active_path_comparison",
        arch="sm_120",
        target_arch=target_arch,
        requested_output_paths=requested_output_paths,
        output_paths_written=[],
        comparison_output_path=None,
        comparison_typed_reason="target_arch_unlabeled_extrapolation",
    )


def _emit_sm120_template_unavailable(
    result: dict[str, Any],
) -> int:
    """Print a typed JSON report without creating any output artifact."""
    print(
        f"run_v5_active_path_comparison: {result['typed_diagnostic']}: "
        f"{SM120_TEMPLATE_UNAVAILABLE_REASON}",
        file=sys.stderr,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2


def _rehydrate_cells_from_v5_head(v5_head: dict) -> list[dict]:
    """Pull the admitted cells (with features) from the v5 head's cohort_scope."""
    admitted = v5_head.get("cohort_scope", {}).get("admitted", [])
    cells = []
    for c in admitted:
        if not c.get("features"):
            continue
        cells.append({
            "identity": (c["arch"], c["slug"], c["shape_point"]),
            "arch": c["arch"],
            "family_bucket": c["family_bucket"],
            "features": c["features"],
            "label": float(c["label"]),
            "slug": c["slug"],
            "shape_point": c["shape_point"],
        })
    return cells


def _fit_arch_head(
    cells_for_arch: list[dict],
    v5_head: dict | None = None,
) -> Any | None:
    """Fit a PerBucketCalibratorEnsemble on cells for a single arch.
    Returns the ensemble OR None when no cells admit OR when the host
    lacks the calibration-head dependencies (graceful degradation: the
    caller emits per_arch_transfer_diagnostics.status=indeterminate_dependencies_absent
    instead of crashing).

    When v5_head is provided, the SELECTED class drives the ensemble via
    the centralized dispatch helper (was: hardcoded IR-on-projection).
    """
    if not cells_for_arch:
        return None
    if v5_head is not None:
        from tools.local_labels.v5_selected_class_dispatch import (
            resolve_selected_factory,
        )
        try:
            factory = resolve_selected_factory(v5_head)
        except (ImportError, NotImplementedError):
            return None
    else:
        factory = mch.IsotonicRegressionOnFrozenProjection
    ensemble = mch.PerBucketCalibratorEnsemble(model_class_factory=factory)
    try:
        ensemble.fit(cells_for_arch)
    except ImportError:
        return None
    return ensemble


def _emit_arch_head_artifact(
    *,
    arch: str,
    ensemble: Any | None,
    cells: list[dict],
    source_v5_head: dict,
    output_path: Path,
) -> dict[str, Any]:
    """Emit data/local_labels/sm_{arch}_v5_head.json with provenance."""
    if ensemble is None:
        artifact = {
            "schema_version": 1,
            "calibration_head_version": 5,
            "arch": arch,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": source_v5_head.get("source_commit"),
            "status": "indeterminate_insufficient_data",
            "admitted_count": 0,
            "reason": (
                f"no admitted cells for arch={arch!r} on this host's "
                "cohort_scope; cannot fit a per-arch head."
            ),
            "composite_score_formula": dict(mch.COMPOSITE_SCORE_FORMULA),
            "feature_schema": dict(
                source_v5_head.get("feature_schema", {})
            ),
        }
    else:
        provenance = ensemble.provenance_summary()
        artifact = {
            "schema_version": 1,
            "calibration_head_version": 5,
            "arch": arch,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": source_v5_head.get("source_commit"),
            "status": "ok",
            "admitted_count": len(cells),
            "training_cell_ids": [
                list(c["identity"]) for c in cells
            ],
            "per_bucket_provenance": provenance,
            "composite_score_formula": dict(mch.COMPOSITE_SCORE_FORMULA),
            "feature_schema": dict(
                source_v5_head.get("feature_schema", {})
            ),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _predict_with_ensemble(
    ensemble: Any, cells: list[dict],
) -> list[tuple[float, float]]:
    """Return (predicted, label) pairs for each cell."""
    out: list[tuple[float, float]] = []
    for c in cells:
        try:
            p = ensemble.predict(c["arch"], c["family_bucket"], c["features"])
        except RuntimeError:
            continue
        out.append((float(p), float(c["label"])))
    return out


def _aggregate_metrics(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    preds = [p for p, _ in pairs]
    labels = [y for _, y in pairs]
    if not preds:
        return {
            "n": 0,
            "median_absolute_error": None,
            "geomean_relative_error": None,
        }
    return {
        "n": len(preds),
        "median_absolute_error": mch.median_absolute_error(preds, labels),
        "geomean_relative_error": mch.geomean_relative_error(preds, labels),
    }


def _compute_transfer_diagnostics(
    *,
    sm_86_ensemble: Any | None,
    sm_120_ensemble: Any | None,
    sm_86_cells: list[dict],
    sm_120_cells: list[dict],
) -> dict[str, Any]:
    """Compute per_arch_transfer_diagnostics with both transfer directions
    + the epsilon-floored 2x rule + the median-abs-err gate."""
    native_sm_86 = (
        _aggregate_metrics(_predict_with_ensemble(sm_86_ensemble, sm_86_cells))
        if sm_86_ensemble is not None else {"n": 0}
    )
    native_sm_120 = (
        _aggregate_metrics(_predict_with_ensemble(sm_120_ensemble, sm_120_cells))
        if sm_120_ensemble is not None else {"n": 0}
    )
    transfer_86_to_120 = (
        _aggregate_metrics(_predict_with_ensemble(sm_86_ensemble, sm_120_cells))
        if sm_86_ensemble is not None and sm_120_cells else {"n": 0}
    )
    transfer_120_to_86 = (
        _aggregate_metrics(_predict_with_ensemble(sm_120_ensemble, sm_86_cells))
        if sm_120_ensemble is not None and sm_86_cells else {"n": 0}
    )

    def _gate(
        transfer: dict, native: dict,
    ) -> dict[str, Any]:
        t_geo = transfer.get("geomean_relative_error")
        n_geo = native.get("geomean_relative_error")
        t_med = transfer.get("median_absolute_error")
        if t_geo is None or n_geo is None:
            return {
                "status": "indeterminate_insufficient_data",
                "transfer_n": transfer.get("n", 0),
                "native_n": native.get("n", 0),
            }
        threshold = TRANSFER_REL_GATE_MULTIPLIER * max(
            n_geo, TRANSFER_EPSILON_FLOOR,
        )
        rel_gate = "pass" if t_geo <= threshold else "fail"
        med_gate = (
            "pass" if (t_med is not None and t_med <= TRANSFER_MEDIAN_ABS_GATE_MAX)
            else "fail"
        )
        return {
            "status": "ok",
            "transfer_geomean_relative_error": t_geo,
            "native_geomean_relative_error": n_geo,
            "epsilon_floored_relative_gate_threshold": threshold,
            "epsilon_floored_relative_gate_status": rel_gate,
            "transfer_median_absolute_error": t_med,
            "median_absolute_gate_threshold": TRANSFER_MEDIAN_ABS_GATE_MAX,
            "median_absolute_gate_status": med_gate,
            "transfer_n": transfer.get("n", 0),
            "native_n": native.get("n", 0),
        }

    return {
        "transfer_epsilon_floor": TRANSFER_EPSILON_FLOOR,
        "transfer_relative_gate_multiplier": TRANSFER_REL_GATE_MULTIPLIER,
        "transfer_median_absolute_gate_max": TRANSFER_MEDIAN_ABS_GATE_MAX,
        "sm_86_native": native_sm_86,
        "sm_120_native": native_sm_120,
        "sm_86_to_sm_120": _gate(transfer_86_to_120, native_sm_120),
        "sm_120_to_sm_86": _gate(transfer_120_to_86, native_sm_86),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-head", default=DEFAULT_V5_HEAD)
    parser.add_argument("--target-arch", default=None)
    parser.add_argument("--sm-120-head-out", default=DEFAULT_SM_120_HEAD_OUT)
    parser.add_argument("--sm-86-head-out", default=DEFAULT_SM_86_HEAD_OUT)
    parser.add_argument("--comparison-out", default=DEFAULT_COMPARISON_OUT)
    parser.add_argument(
        "--allow-committed-paths", action="store_true",
        help=(
            "Explicit opt-in required before any output path may target "
            "a committed (git-tracked) artifact. Without this flag the "
            "run fails closed with CommittedOutputPathError instead of "
            "rewriting a committed file."
        ),
    )
    args = parser.parse_args(argv)
    if args.target_arch is not None:
        args.target_arch = canonical_arch(args.target_arch)
    requested_output_paths = [
        args.sm_120_head_out,
        args.sm_86_head_out,
        args.comparison_out,
    ]

    # An explicit sm_120 target is blocked before reading any calibration
    # artifact, checking output paths, fitting, or scoring. This keeps the
    # result a diagnostic rather than a partially-emitted transfer bundle.
    if is_sm120_template_arch(args.target_arch):
        return _emit_sm120_template_unavailable(
            _sm120_template_unavailable_result(
                target_arch=args.target_arch,
                requested_output_paths=requested_output_paths,
            )
        )

    # For an unqualified invocation, inspect only the v5 head's architecture
    # metadata. If it contains sm_120 (including a mixed cohort), stop before
    # output planning or any per-arch fit. Pure sm_86 heads retain the existing
    # active behavior below.
    v5_head = json.loads(Path(args.v5_head).read_text(encoding="utf-8"))
    if _v5_head_contains_sm120(v5_head):
        return _emit_sm120_template_unavailable(
            _sm120_template_unavailable_result(
                target_arch=args.target_arch,
                requested_output_paths=requested_output_paths,
            )
        )

    assert_output_paths_not_committed(
        requested_output_paths,
        allow_committed_paths=args.allow_committed_paths,
    )
    cells = _rehydrate_cells_from_v5_head(v5_head)
    sm_120_cells = [c for c in cells if c["arch"] == "sm_120"]
    sm_86_cells = [c for c in cells if c["arch"] == "sm_86"]
    sm_120_ensemble = _fit_arch_head(sm_120_cells, v5_head=v5_head)
    sm_86_ensemble = _fit_arch_head(sm_86_cells, v5_head=v5_head)
    _emit_arch_head_artifact(
        arch="sm_120", ensemble=sm_120_ensemble, cells=sm_120_cells,
        source_v5_head=v5_head, output_path=Path(args.sm_120_head_out),
    )
    _emit_arch_head_artifact(
        arch="sm_86", ensemble=sm_86_ensemble, cells=sm_86_cells,
        source_v5_head=v5_head, output_path=Path(args.sm_86_head_out),
    )
    # Per-cell predictions + per-arch metrics + per_arch_transfer_diagnostics.
    # Use the SELECTED class via the centralized dispatch helper. When the
    # dispatch is NOT ok, the per-cell loop emits v5_prediction=None on
    # every row + the artifact's top-level status records the dispatch
    # failure (NO implicit IR fallback per the no-fallback contract).
    from tools.local_labels.v5_selected_class_dispatch import (
        build_selected_ensemble,
    )
    full_dispatch = build_selected_ensemble(v5_head, cells=cells)
    full_ensemble = (
        full_dispatch["ensemble"] if full_dispatch["status"] == "ok"
        else None
    )
    per_cell_rows = []
    for c in cells:
        p_val: float | None
        if full_ensemble is None:
            # No-fallback contract: dispatch failed at the top level;
            # every row emits v5_prediction=None (the artifact's
            # selected_dispatch_status records the reason).
            p_val = None
        else:
            try:
                p = full_ensemble.predict(
                    c["arch"], c["family_bucket"], c["features"],
                )
                p_val = float(p)
            except (RuntimeError, ImportError):
                # Per-row graceful degradation: emit the row with
                # v5_prediction=None so consumers see the same row count
                # regardless of dependency availability.
                p_val = None
        per_cell_rows.append({
            "arch": c["arch"], "slug": c["slug"],
            "shape_point": c["shape_point"],
            "family_bucket": c["family_bucket"],
            "label": c["label"],
            "v5_prediction": p_val,
            "absolute_error": (
                abs(p_val - c["label"]) if p_val is not None else None
            ),
        })
    transfer = _compute_transfer_diagnostics(
        sm_86_ensemble=sm_86_ensemble,
        sm_120_ensemble=sm_120_ensemble,
        sm_86_cells=sm_86_cells,
        sm_120_cells=sm_120_cells,
    )
    # Guard the aggregate-metrics block on full_ensemble availability:
    # the no-fallback contract sets full_ensemble=None when
    # build_selected_ensemble returns a non-ok dispatch, so an
    # unconditional ._fitted dereference would AttributeError on the
    # exact failure path the bundle is supposed to advertise via
    # selected_dispatch_status.
    if full_ensemble is not None and full_ensemble._fitted:
        sm_120_aggregate = _aggregate_metrics(
            _predict_with_ensemble(full_ensemble, sm_120_cells),
        )
        sm_86_aggregate = _aggregate_metrics(
            _predict_with_ensemble(full_ensemble, sm_86_cells),
        )
    else:
        sm_120_aggregate = _aggregate_metrics([])
        sm_86_aggregate = _aggregate_metrics([])
    comparison = {
        "schema_version": 1,
        "source_commit": _git_head_sha(),
        "source_v5_head_commit": v5_head.get("source_commit"),
        "selected_dispatch_status": full_dispatch["status"],
        "selected_dispatch_reason": full_dispatch.get("reason", ""),
        "selected_class_name": full_dispatch.get("selected_class_name"),
        "consumer_module": "tools/local_labels/run_v5_active_path_comparison.py",
        "consumer_invokes": (
            "workloads/simbench_bw/canonical_features.predict_active_path "
            "via workloads/simbench_bw/monotonic_calibration_head."
            "PerBucketCalibratorEnsemble"
        ),
        "row_count": len(per_cell_rows),
        "rows": per_cell_rows,
        "per_arch_aggregate_metrics": {
            "sm_120": sm_120_aggregate,
            "sm_86": sm_86_aggregate,
        },
        "per_arch_transfer_diagnostics": transfer,
        "feature_schema": dict(v5_head.get("feature_schema", {})),
    }
    comparison_out_path = Path(args.comparison_out)
    comparison_out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_out_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v5_active_path_comparison] rows={len(per_cell_rows)} "
        f"sm_120_n={len(sm_120_cells)} sm_86_n={len(sm_86_cells)} "
        f"-> {args.comparison_out}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
