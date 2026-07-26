"""Fit the three calibration heads (Isotonic + Platt + Beta) on
an admitted-cell corpus filtered by the ``--target-arch`` flag,
emit ``data/local_labels/calibration_head_v1.json`` (the trained
chosen-head sidecar) AND
``data/local_labels/calibration_evaluation_v1.json`` (the per-
head LOOCV residual + uncalibrated-baseline + chosen-head
rationale sidecar).

The targets-never-features boundary applies here: NCU is NOT invoked; this
script only reads existing label_admitted.json + predictor bundle JSONs and
writes the two sidecars. Living under ``tools/local_labels/`` keeps it
co-located with the other label-driver scripts; it does NOT cross into
hardware collection (no NCU, no nvidia-smi mutation, no privileged CUPTI).

``--target-arch`` admits ``sm_86`` (the legacy default) and
``sm_120`` (the active sm_120-only decoupled framework target).
The chosen arch is recorded verbatim in the emitted eval
sidecar's ``target_arch`` field; the legacy ``sm_86_only`` pin
is preserved only for callers that explicitly pass the default
without overriding it. Leave-arch-out evaluation defers to a
follow-up cross-arch RLCR loop.

DEC-T9 ``simplicity_one_se_rule`` (pinned by the active round
contract): simplicity ordering is PlattCalibrator (2 params) <
IsotonicCalibrator (PAV) < BetaCalibrator (3 params); pick the
simplest head whose LOOCV p90 residual is within
``1 * loocv_p90_se_best`` of the best p90 across all three.
The tie-breaker is the simplicity order itself.

DEC-T5 threshold (pinned): the downstream consumer reads the
chosen head's LOOCV p90 from the evaluation sidecar and fires
``calibration_loocv_p90_residual_exceeded`` typed reason when
that p90 > 0.30. The fit script does NOT decide that — it just
records the value so the consumer can act on it.

Recommended invocation against the committed dependency pin
(``requirements.txt`` at repo root, per DEC-T8):

    uv run --with-requirements requirements.txt \\
      python -m tools.local_labels.fit_calibration_head

Equivalent ad hoc invocation:

    uv run --with numpy --with scipy --with scikit-learn \\
      python -m tools.local_labels.fit_calibration_head
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUITE_ROOT = _REPO_ROOT / "workloads" / "simbench_bw"
_DATA_ROOT = _REPO_ROOT / "data" / "local_labels"

sys.path.insert(0, str(_SUITE_ROOT))


_TARGET_ARCH_PIN = "sm_86_only"
_SELECTION_RULE_PIN = "simplicity_one_se_rule"

# Allowed values for the ``--target-arch`` CLI flag. ``sm_120`` is
# the active target for the sm_120-only decoupled framework (RTX
# 5060 laptop Blackwell consumer). ``sm_86`` is retained only for
# legacy v1 back-compat — it is NOT an active development target
# in the current framework; existing v1 readers that explicitly
# pass the default must continue to admit. Other tokens are
# rejected at parse time.
_ALLOWED_TARGET_ARCHES = frozenset({"sm_86", "sm_120"})


def _target_arch_pin_for(target_arch: str) -> str:
    """Map ``--target-arch`` to the eval sidecar's ``target_arch``
    field value. ``sm_86`` preserves the legacy ``sm_86_only`` pin
    for back-compat with existing v1 readers; ``sm_120`` records
    ``sm_120`` verbatim so the sm_120-only decoupled framework
    surfaces the arch unambiguously."""
    if target_arch == "sm_86":
        return _TARGET_ARCH_PIN
    return target_arch

# Simplicity ordering for DEC-T9 simplicity_one_se_rule. Index 0
# is simplest.
_SIMPLICITY_ORDER = (
    "PlattCalibrator",
    "IsotonicCalibrator",
    "BetaCalibrator",
)

_DEFAULT_OUTPUT_HEAD_PATH = _DATA_ROOT / "calibration_head_v1.json"
_DEFAULT_OUTPUT_EVAL_PATH = _DATA_ROOT / "calibration_evaluation_v1.json"
_DEFAULT_ADMITTED_PATH = _DATA_ROOT / "label_admitted.json"
_DEFAULT_COMPARISON_CONTRACT_PATH = (
    _SUITE_ROOT / "comparison_contract.toml"
)
_DEFAULT_BUNDLES_ROOT = _DATA_ROOT / "sweeps"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "fit_calibration_head requires numpy. Install numpy + scipy "
            "+ scikit-learn (e.g., via `uv run --with numpy --with scipy "
            "--with scikit-learn python -m tools.local_labels.fit_calibration_head`)."
        ) from exc
    return np


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha() -> str:
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _bundle_5tuple(bundle: dict) -> tuple[str, ...]:
    mpi = bundle.get("manifest_pairing_index", {})
    return (
        str(mpi.get("slug", "")),
        str(mpi.get("shape_point", "")),
        str(mpi.get("binary_sha256", "")),
        str(mpi.get("source_commit", "")),
        str(mpi.get("kernel_signature_sha256", "")),
    )


def _admitted_5tuple(cell: dict) -> tuple[str, ...]:
    return (
        str(cell.get("slug", "")),
        str(cell.get("shape_point", "")),
        str(cell.get("binary_sha256", "")),
        str(cell.get("source_commit", "")),
        str(cell.get("kernel_signature_sha256", "")),
    )


def _scan_predictor_bundles(bundles_root: Path) -> dict[tuple[str, ...], dict]:
    """Walk every ``sweeps/<sweep>/predictor_bundles/*.json``
    file and return the 5-tuple → bundle index. When multiple
    sweeps emit identical bundles, the most recently modified
    sweep wins (mtime-based dedupe)."""
    by_key: dict[tuple[str, ...], tuple[float, dict]] = {}
    for path in bundles_root.glob("*/predictor_bundles/*.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                bundle = json.load(fh)
        except (OSError, ValueError):
            continue
        if bundle.get("result_kind") != "scalar":
            continue
        if not isinstance(bundle.get("predicted_hit_rate"), (int, float)):
            continue
        if isinstance(bundle["predicted_hit_rate"], bool):
            continue
        key = _bundle_5tuple(bundle)
        if not all(part for part in key):
            continue
        mtime = path.stat().st_mtime
        prev = by_key.get(key)
        if prev is None or mtime > prev[0]:
            by_key[key] = (mtime, bundle)
    return {k: v[1] for k, v in by_key.items()}


def _build_training_rows(
    admitted_cells: list[dict],
    bundle_index: dict[tuple[str, ...], dict],
    *,
    target_arch: str = "sm_86",
) -> list[dict]:
    """Join admitted-label cells on the 5-tuple key to predictor
    bundles. Returns one row per admitted cell with a matching
    bundle. Each row carries the 7-tuple manifest_pairing_index
    fields (the 5-tuple plus arch + ncu_metric_version) plus
    ``predicted`` (from the bundle) and ``label`` (from the
    admitted cell, normalized from the 0-100% percentage to a
    [0, 1] rate).
    """
    rows: list[dict] = []
    for cell in admitted_cells:
        if cell.get("arch") != target_arch:
            continue
        cell_pct = cell.get("cache_hit_rate_l2_sector_pct")
        if not isinstance(cell_pct, (int, float)) or isinstance(cell_pct, bool):
            continue
        key = _admitted_5tuple(cell)
        bundle = bundle_index.get(key)
        if bundle is None:
            continue
        # Arch verification (decoupling-discipline guard): a bundle
        # whose resolved arch disagrees with the admitted cell's arch
        # MUST NOT join, even when the 5-tuple matches. Without this
        # guard a sm_86 bundle could be silently consumed by a
        # ``--target-arch sm_120`` invocation if the 5-tuple happens
        # to overlap, re-introducing the cross-arch coupling the
        # sm_120-only decoupled framework explicitly forbids. Bundles
        # produced by the sm_120-only activation bundle producer
        # carry an explicit ``manifest_pairing_index.arch`` field;
        # legacy microbench bundles that predate this field fall back
        # to parsing the ``device`` string (an ``RTX 3090 Ti`` device
        # marker maps to sm_86, an ``RTX 5060 Laptop`` device marker
        # maps to sm_120).
        #
        # Strictness ratchet for ``target_arch == "sm_120"``: a
        # bundle whose arch cannot be resolved (None) is treated as
        # AMBIGUOUS and REJECTED — the sm_120-only path will NOT
        # admit an unconstrained legacy bundle even when the 5-tuple
        # matches by coincidence. For ``target_arch == "sm_86"`` the
        # legacy tolerance is preserved: None means unconstrained
        # admission so existing v1 fits continue to reproduce.
        bundle_arch = _bundle_arch(bundle)
        if bundle_arch is None:
            if target_arch == "sm_120":
                continue
        elif bundle_arch != cell["arch"]:
            continue
        predicted = bundle["predicted_hit_rate"]
        rows.append({
            "arch": cell["arch"],
            "slug": cell["slug"],
            "shape_point": cell["shape_point"],
            "binary_sha256": cell["binary_sha256"],
            "kernel_signature_sha256": cell["kernel_signature_sha256"],
            "source_commit": cell["source_commit"],
            "ncu_metric_version": cell["ncu_metric_version"],
            "predicted": float(predicted),
            "label": float(cell_pct) / 100.0,
        })
    return rows


def _bundle_arch(bundle: dict) -> str | None:
    """Resolve a predictor bundle's arch token. Returns ``None``
    when the bundle carries neither an explicit
    ``manifest_pairing_index.arch`` field nor a recognized
    ``device`` marker; the caller treats ``None`` as
    "unconstrained" (the legacy v1 single-arch corpus pre-dates
    this verification and is admitted unchanged).

    Recognized devices: ``NVIDIA GeForce RTX 3090 Ti`` -> ``sm_86``;
    ``NVIDIA GeForce RTX 5060 Laptop GPU`` (substring match)
    -> ``sm_120``. Any other device string returns ``None``.
    """
    mpi = bundle.get("manifest_pairing_index", {})
    arch = mpi.get("arch")
    if isinstance(arch, str) and arch:
        return arch
    device = bundle.get("device")
    if isinstance(device, str):
        if "RTX 3090 Ti" in device:
            return "sm_86"
        if "RTX 5060 Laptop" in device:
            return "sm_120"
    return None


def _loocv_residuals(
    calibrator_factory: Any,
    rows: list[dict],
) -> list[float]:
    """Compute leave-one-out cross-validation absolute residuals.

    For each held-out row i, fit a fresh calibrator on the
    remaining (n-1) rows' (predicted, label) pairs, then
    transform the held-out predicted and compare to the
    held-out label.
    """
    np = _require_numpy()
    residuals: list[float] = []
    n = len(rows)
    for hold_out in range(n):
        train_p = np.array([
            r["predicted"] for i, r in enumerate(rows) if i != hold_out
        ])
        train_y = np.array([
            r["label"] for i, r in enumerate(rows) if i != hold_out
        ])
        head = calibrator_factory()
        head.fit(train_p, train_y)
        held_pred = np.array([rows[hold_out]["predicted"]])
        calibrated, _envelope = head.transform(held_pred)
        residual = float(abs(float(calibrated[0]) - rows[hold_out]["label"]))
        residuals.append(residual)
    return residuals


def _bootstrap_p90_se(
    residuals: list[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> float:
    """Bootstrap the standard error of the p90 estimator over
    ``residuals``. Returns 0.0 if the residual list has fewer
    than 2 entries (the SE is undefined with a single fold)."""
    if len(residuals) < 2:
        return 0.0
    np = _require_numpy()
    rng = np.random.default_rng(seed)
    arr = np.asarray(residuals, dtype=float)
    n = arr.shape[0]
    p90s = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        p90s[i] = float(np.percentile(arr[idx], 90))
    return float(np.std(p90s, ddof=1))


def _per_head_residual_block(residuals: list[float]) -> dict[str, Any]:
    np = _require_numpy()
    arr = np.asarray(residuals, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "se_p90": _bootstrap_p90_se(residuals),
        "per_fold_residuals": [float(r) for r in residuals],
    }


def _uncalibrated_baseline(rows: list[dict]) -> dict[str, Any]:
    np = _require_numpy()
    arr = np.asarray(
        [abs(r["predicted"] - r["label"]) for r in rows], dtype=float,
    )
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "per_cell_residuals": [float(x) for x in arr],
    }


def _choose_head_simplicity_one_se(
    loocv_table: dict[str, dict],
) -> tuple[str, dict]:
    """Apply DEC-T9 simplicity_one_se_rule.

    Find the best p90 across the 3 heads. The se_threshold is
    ``1.0 * se_p90`` of the best head. Among heads whose p90
    is within ``best_p90 + se_threshold``, pick the SIMPLEST per
    ``_SIMPLICITY_ORDER`` (Platt < Isotonic < Beta).

    Returns ``(chosen_head_name, rationale_dict)``.
    """
    p90s = {h: float(b["p90"]) for h, b in loocv_table.items()}
    best_p90_head = min(p90s, key=lambda h: p90s[h])
    best_p90 = p90s[best_p90_head]
    best_se = float(loocv_table[best_p90_head]["se_p90"])
    se_threshold = best_se  # 1 * SE
    eligible = [
        h for h in _SIMPLICITY_ORDER
        if h in p90s and p90s[h] <= best_p90 + se_threshold
    ]
    chosen = eligible[0] if eligible else best_p90_head
    return chosen, {
        "selection_rule": _SELECTION_RULE_PIN,
        "chosen_head": chosen,
        "chosen_head_p90": float(p90s[chosen]),
        "best_p90": float(best_p90),
        "best_p90_head": str(best_p90_head),
        "chosen_head_p90_distance_from_best": float(p90s[chosen] - best_p90),
        "se_threshold": float(se_threshold),
    }


def _sklearn_available() -> bool:
    """Return True iff scikit-learn imports successfully on this
    host. The Isotonic calibrator depends on
    ``sklearn.isotonic.IsotonicRegression``; when sklearn is
    absent, the optional-head policy omits Isotonic from the
    candidate set rather than failing the entire fit."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def _calibrator_factories() -> dict[str, Any]:
    """Lazy-import the calibrator classes from
    ``workloads/simbench_bw/calibration_head.py`` and return a
    dict of `{class_name: factory_callable}`. The factories are
    no-arg constructors; the fit happens at LOOCV time.

    Optional-head policy: when scikit-learn is not installed on
    this host, ``IsotonicCalibrator`` is omitted from the returned
    dict (its fit imports sklearn lazily and would otherwise raise
    at LOOCV time). Platt + Beta are stdlib-only and always
    available. The caller records the omission in
    ``_skipped_heads_for_provenance()`` so the eval sidecar surfaces
    the optional-dependency state explicitly."""
    import calibration_head as ch  # type: ignore[import-not-found]
    factories: dict[str, Any] = {
        "PlattCalibrator": ch.PlattCalibrator,
        "BetaCalibrator": ch.BetaCalibrator,
    }
    if _sklearn_available():
        factories["IsotonicCalibrator"] = ch.IsotonicCalibrator
    return factories


def _skipped_heads_for_provenance() -> dict[str, str]:
    """Return a `{head_name: reason}` dict recording any heads
    that the current host omits from the candidate set. Recorded
    verbatim in the eval sidecar so downstream readers (and
    future reviewers) can verify the simplicity-one-SE rule was
    applied over the available heads, not over the full canonical
    set."""
    skipped: dict[str, str] = {}
    if not _sklearn_available():
        skipped["IsotonicCalibrator"] = "sklearn_unavailable"
    return skipped


def _calibrator_library_provenance() -> dict[str, dict[str, Any]]:
    """Build the calibrator_libraries object for the trained-head
    sidecar. Isotonic carries sklearn version; Platt + Beta
    carry the stdlib_only flag."""
    try:
        import sklearn
        sklearn_version = str(sklearn.__version__)
    except ImportError:
        sklearn_version = "unknown"
    return {
        "isotonic": {
            "library_id": "sklearn",
            "version": sklearn_version,
        },
        "platt": {
            "library_id": "stdlib_handrolled",
            "stdlib_only": True,
        },
        "beta": {
            "library_id": "stdlib_handrolled",
            "stdlib_only": True,
        },
    }


def _fitted_head_serialized(head_name: str, head: Any) -> dict[str, Any]:
    """Return a JSON-serializable dict of the head's fitted
    parameters. Used to compute trained_head_sha256."""
    if head_name == "PlattCalibrator":
        return {"A": float(head.A), "B": float(head.B)}
    if head_name == "BetaCalibrator":
        return {"a": float(head.a), "b": float(head.b), "c": float(head.c)}
    # IsotonicCalibrator: sklearn IsotonicRegression's piecewise
    # state is captured by (X_thresholds_, y_thresholds_).
    model = head._model
    x_thr = getattr(model, "X_thresholds_", None)
    y_thr = getattr(model, "y_thresholds_", None)
    if x_thr is None or y_thr is None:
        # Fallback: stash the input/output min/max for at-least
        # something tamper-detectable.
        return {
            "training_predicted_min": float(head.training_predicted_min),
            "training_predicted_max": float(head.training_predicted_max),
        }
    return {
        "x_thresholds": [float(x) for x in x_thr],
        "y_thresholds": [float(y) for y in y_thr],
    }


def fit_and_emit(
    *,
    admitted_path: Path = _DEFAULT_ADMITTED_PATH,
    bundles_root: Path = _DEFAULT_BUNDLES_ROOT,
    comparison_contract_path: Path = _DEFAULT_COMPARISON_CONTRACT_PATH,
    head_output_path: Path = _DEFAULT_OUTPUT_HEAD_PATH,
    eval_output_path: Path = _DEFAULT_OUTPUT_EVAL_PATH,
    target_arch: str = "sm_86",
) -> tuple[Path, Path, dict[str, Any]]:
    """Run the full fit pipeline end to end. Returns
    ``(head_path, eval_path, summary_dict)``.

    The summary dict contains the chosen head, its LOOCV p90,
    the simplicity rationale, the number of training pairs, and
    whether the chosen head exceeds the DEC-T5 0.30 threshold.
    The caller can use the threshold flag to know whether the
    downstream comparison_runner should fire
    ``calibration_loocv_p90_residual_exceeded``.
    """
    np = _require_numpy()

    with admitted_path.open("r", encoding="utf-8") as fh:
        admitted = json.load(fh)
    admitted_cells = admitted.get("cells", [])
    bundle_index = _scan_predictor_bundles(bundles_root)
    rows = _build_training_rows(
        admitted_cells, bundle_index, target_arch=target_arch,
    )
    if len(rows) < 2:
        raise RuntimeError(
            f"fit_calibration_head: only {len(rows)} joinable "
            f"(predicted, label) training pairs for arch={target_arch!r}. "
            f"Need at least 2 cells to run LOOCV."
        )

    factories = _calibrator_factories()
    loocv_table: dict[str, Any] = {}
    fitted_heads: dict[str, Any] = {}
    train_p = np.array([r["predicted"] for r in rows])
    train_y = np.array([r["label"] for r in rows])
    for head_name, factory in factories.items():
        residuals = _loocv_residuals(factory, rows)
        loocv_table[head_name] = _per_head_residual_block(residuals)
        # Also fit the full-data head for the trained-head sidecar.
        full = factory()
        full.fit(train_p, train_y)
        fitted_heads[head_name] = full

    chosen, rationale = _choose_head_simplicity_one_se(loocv_table)
    chosen_head = fitted_heads[chosen]
    chosen_params = _fitted_head_serialized(chosen, chosen_head)

    contract_sha256 = _sha256_of_file(comparison_contract_path)
    source_commit = _git_head_sha()
    training_cells = sorted(
        {(r["slug"], r["shape_point"], r["arch"]) for r in rows}
    )

    # --- trained-head sidecar ----------------------------------
    head_payload = {
        "schema_version": 1,
        "chosen_head": chosen,
        "calibrator_libraries": _calibrator_library_provenance(),
        "training_cells": [
            {"slug": s, "shape_point": sp, "arch": a}
            for (s, sp, a) in training_cells
        ],
        "training_pair_count": len(rows),
        "training_timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "trained_head_sha256": hashlib.sha256(
            json.dumps(chosen_params, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "comparison_contract_sha256": contract_sha256,
        "training_rows": rows,
        "source_commit": source_commit,
        # The trained-head sidecar records ``target_arch`` (mirrors
        # the eval sidecar) so downstream consumers + the head
        # provenance scanner can verify the active arch without
        # cross-referencing the eval file. The sm_120-only
        # decoupled framework requires this field present on v3
        # heads; the legacy ``sm_86`` path emits ``sm_86_only`` for
        # back-compat with v1 readers.
        "target_arch": _target_arch_pin_for(target_arch),
    }
    head_output_path.parent.mkdir(parents=True, exist_ok=True)
    with head_output_path.open("w", encoding="utf-8") as fh:
        json.dump(head_payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # --- evaluation sidecar ------------------------------------
    uncal_baseline = _uncalibrated_baseline(rows)
    eval_payload = {
        "schema_version": 1,
        "chosen_head": chosen,
        "selection_rule": _SELECTION_RULE_PIN,
        "loocv_table": loocv_table,
        "uncalibrated_baseline_table": uncal_baseline,
        "target_arch": _target_arch_pin_for(target_arch),
        "comparison_contract_sha256": contract_sha256,
        "source_commit": source_commit,
        "training_cells": [
            {"slug": s, "shape_point": sp, "arch": a}
            for (s, sp, a) in training_cells
        ],
        "training_pair_count": len(rows),
        "rationale": rationale,
        "skipped_heads": _skipped_heads_for_provenance(),
    }
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_output_path.open("w", encoding="utf-8") as fh:
        json.dump(eval_payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    chosen_p90 = float(loocv_table[chosen]["p90"])
    summary = {
        "chosen_head": chosen,
        "chosen_head_p90": chosen_p90,
        "rationale": rationale,
        "training_pair_count": len(rows),
        "loocv_p90_residual_exceeded": chosen_p90 > 0.30,
        "head_output_path": str(head_output_path),
        "eval_output_path": str(eval_output_path),
    }
    return head_output_path, eval_output_path, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Isotonic + Platt + Beta calibration heads on the "
            "sm_86 admitted-cell corpus and emit calibration_head_v1.json "
            "+ calibration_evaluation_v1.json."
        ),
    )
    parser.add_argument(
        "--admitted-path", type=Path, default=_DEFAULT_ADMITTED_PATH,
    )
    parser.add_argument(
        "--bundles-root", type=Path, default=_DEFAULT_BUNDLES_ROOT,
    )
    parser.add_argument(
        "--comparison-contract-path",
        type=Path,
        default=_DEFAULT_COMPARISON_CONTRACT_PATH,
    )
    parser.add_argument(
        "--head-output-path", type=Path, default=_DEFAULT_OUTPUT_HEAD_PATH,
    )
    parser.add_argument(
        "--eval-output-path", type=Path, default=_DEFAULT_OUTPUT_EVAL_PATH,
    )
    parser.add_argument(
        "--target-arch",
        default="sm_86",
        choices=sorted(_ALLOWED_TARGET_ARCHES),
        help=(
            "Arch to filter admitted cells by (default: sm_86 for "
            "back-compat with v1; sm_120 for the sm_120-only "
            "decoupled framework path)."
        ),
    )
    args = parser.parse_args(argv)

    head_path, eval_path, summary = fit_and_emit(
        admitted_path=args.admitted_path,
        bundles_root=args.bundles_root,
        comparison_contract_path=args.comparison_contract_path,
        head_output_path=args.head_output_path,
        eval_output_path=args.eval_output_path,
        target_arch=args.target_arch,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"head sidecar: {head_path}", file=sys.stderr)
    print(f"evaluation sidecar: {eval_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
