"""Coverage-preserving frontier fallback shared by alpha committee models.

The fallback is deliberately independent of score-row NCU labels.  It fits a
log-linear ridge projection on a separate campaign's sparse replay frontiers
and trace-shape features, then returns a wide residual interval whose width
grows with feature distance.  It is a coverage path, not a claim that an
unmatched workload shares an archetype or that the hidden mechanisms are
identified.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")


def _surrogate():
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import sm86_ncu_projection_surrogate as surrogate  # type: ignore

    return surrogate


def fit_frontier_fallback(
    train_campaign: Path,
    score_campaign: Path,
    train_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
    ridge: float = 1e-3,
) -> dict:
    """Fit independent-cohort frontier fallback and return per-score-row data."""
    surrogate = _surrogate()
    if not train_rows:
        raise RuntimeError("fallback calibration campaign has no rows")
    feature_maps = [surrogate.frontier_features(row, train_campaign) for row in train_rows]
    score_maps = [surrogate.frontier_features(row, score_campaign) for row in score_rows]
    names = sorted(feature_maps[0])
    if any(sorted(item) != names for item in feature_maps + score_maps):
        raise RuntimeError("fallback frontier feature schema differs")
    x_train = np.asarray([[1.0] + [item[name] for name in names] for item in feature_maps], dtype=float)
    x_score = np.asarray([[1.0] + [item[name] for name in names] for item in score_maps], dtype=float)
    predictions: dict[str, np.ndarray] = {}
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    residual_q90: dict[str, float] = {}
    coefficients: dict[str, list[float]] = {}
    for target in TARGETS:
        y_train = np.asarray([float(row[target]) for row in train_rows], dtype=float)
        # Match the repository's frontier lane: sparse replay frontiers are
        # already in the endpoint domain, so a direct ridge fit is less
        # distortive than the log-target trace model.  NCU labels remain
        # calibration inputs from the independent fallback cohort only.
        beta, scale = surrogate.fit_ridge(x_train, y_train, ridge)
        train_pred = (x_train / scale) @ beta
        score_pred = np.maximum((x_score / scale) @ beta, 0.0)
        residual = np.abs(train_pred - y_train)
        q90 = float(np.quantile(residual, 0.90))
        predictions[target] = score_pred
        residual_q90[target] = q90
        coefficients[target] = beta.tolist()
        intervals[target] = (
            np.maximum(score_pred - q90, 0.0),
            score_pred + q90,
        )
    feature_scale = np.sqrt(np.mean(x_train[:, 1:] * x_train[:, 1:], axis=0))
    feature_scale[feature_scale == 0.0] = 1.0
    normalized_train = x_train[:, 1:] / feature_scale
    normalized_score = x_score[:, 1:] / feature_scale
    distances = np.asarray([
        float(np.min(np.linalg.norm(normalized_train - row, axis=1)))
        for row in normalized_score
    ])
    # The base residual is calibrated in-family; widen it for a distant
    # fallback row without converting distance into a binary refusal.
    distance_factor = 1.0 + np.minimum(distances / 3.0, 8.0)
    for target in TARGETS:
        low, high = intervals[target]
        center = predictions[target]
        widened = residual_q90[target] * distance_factor
        intervals[target] = (
            np.maximum(center - widened, 0.0),
            center + widened,
        )
    return {
        "predictions": predictions,
        "intervals": intervals,
        "feature_names": names,
        "feature_distances": distances,
        "distance_factor": distance_factor,
        "residual_q90": residual_q90,
        "coefficients": coefficients,
        "contract": "independent frontier/trace fallback; no NCU or pipe features",
    }


def apply_independent_phase_transfer(
    fallback: dict,
    cold_rows: list[dict[str, str]],
    warm_campaign: Path,
    holdout_family: str = "",
) -> dict:
    """Apply a disjoint-family warm/cold ratio to fallback predictions.

    Ratios are computed only on the fallback calibration rows, after removing
    the scored holdout family.  The score workload's warm NCU value is never
    read by this function.
    """
    surrogate = _surrogate()
    warm_rows = surrogate.load_rows(warm_campaign)
    if holdout_family:
        cold_rows = [
            row for row in cold_rows
            if row.get("workload", "").split("__", 1)[0] != holdout_family
        ]
        warm_rows = [
            row for row in warm_rows
            if row.get("workload", "").split("__", 1)[0] != holdout_family
        ]
    warm_by_workload = {row["workload"]: row for row in warm_rows}
    ratios: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for target in TARGETS:
        values = [
            float(warm_by_workload[row["workload"]][target]) / max(float(row[target]), 1.0)
            for row in cold_rows
            if row["workload"] in warm_by_workload
            and float(row.get(target, 0.0) or 0.0) > 0
            and float(warm_by_workload[row["workload"]].get(target, 0.0) or 0.0) > 0
        ]
        if not values:
            raise RuntimeError("phase-transfer calibration has no disjoint overlap")
        ratios[target] = float(np.median(values))
        ranges[target] = (float(np.quantile(values, 0.10)), float(np.quantile(values, 0.90)))
        center = fallback["predictions"][target] * ratios[target]
        low, high = fallback["intervals"][target]
        # Preserve the original residual and add the independent ratio band.
        ratio_low, ratio_high = ranges[target]
        fallback["predictions"][target] = center
        fallback["intervals"][target] = (
            np.maximum(np.minimum(low * ratio_low, center), 0.0),
            np.maximum(high * ratio_high, center),
        )
    fallback["phase_transfer_ratios"] = ratios
    fallback["phase_transfer_ratio_q10_q90"] = ranges
    fallback["phase_transfer_campaign"] = str(warm_campaign)
    return fallback
