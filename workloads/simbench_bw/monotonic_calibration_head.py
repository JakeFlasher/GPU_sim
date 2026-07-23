"""Per-bucket monotonic calibration head v5.

Pre-registered model-class hierarchy (per the goal tracker's the model-class hierarchy row):

  A. ``IsotonicRegressionOnFrozenProjection`` -- sklearn.IsotonicRegression
     fit on a FROZEN linear projection of (raw_prediction, footprint_ratio,
     long_rd_tail_mass) with weights (+1, -1, -1). The simplest of the three
     candidate classes; one-dimensional non-parametric PAV. Lower-bound
     scope per the plan's Path Boundaries: this is the ONLY model class
     evaluated when pygam and LightGBM are unavailable.

  B. ``MonotoneGAMOnSixFeatures`` -- pygam.LinearGAM with sign-constrained
     smooths on all 6 features. Admitted only when pygam is importable.

  C. ``MonotoneGBMOnSixFeatures`` -- lightgbm.LGBMRegressor with
     ``monotone_constraints`` matching the the monotonic-constraint-sign table sign table. Admitted
     only when lightgbm is importable.

The active-path active path entry is ``workloads/simbench_bw/canonical_features.py``
which consumes ONLY the 6-feature block + raw_prediction. This module owns
the head's fit/predict surface; the canonical_features module owns the
wire contract + the active-path domain-mismatch enforcement.

Per-bucket discipline (the per-bucket minimum-N rule): each (arch, family_bucket) with N >= 8
admitted cells gets its own per-bucket head; smaller buckets fall back to a
pooled head with ``family_bucket`` as a one-hot covariate. The per-bucket
kind is recorded in provenance.

Deterministic stratified round-robin fold construction (the model-class hierarchy): within each
(arch, family_bucket) stratum, sort cells by ``hash(seed=20260527,
cell_identity_tuple)``, then assign folds round-robin. Fallback to
non-stratified k-fold if any stratum has fewer cells than the
minimum-per-fold criterion; the fallback decision is recorded.

Composite scoring (the model-class hierarchy): primary = mean CV geomean relative error;
secondary tiebreaker = median absolute hit-rate error; tertiary = log-loss
with eps=1e-6 and fractional-Bernoulli labels. Simpler-model tiebreaker on
PRIMARY only within <=2% difference.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

# Late import: canonical_features owns CANONICAL_FEATURE_NAMES, which the
# multi-feature model classes consume to enumerate input columns in the
# canonical order.
from workloads.simbench_bw.canonical_features import (
    CANONICAL_FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------

# Frozen 3-feature projection per the model-class hierarchy's IsotonicRegressionOnFrozenProjection
# specification. The weights are NOT learnable; pinning them at registration
# time avoids calibration leakage that a learnable projection would introduce.
FROZEN_PROJECTION_FEATURES: tuple[str, ...] = (
    "raw_prediction",
    "footprint_ratio",
    "long_rd_tail_mass",
)
FROZEN_PROJECTION_WEIGHTS: tuple[float, ...] = (+1.0, -1.0, -1.0)


# Per-bucket fitting minimum per the per-bucket minimum-N rule.
PER_BUCKET_MIN_N: int = 8


# Deterministic stratified round-robin fold construction seed per the model-class hierarchy.
FOLD_SEED: int = 20260527
N_FOLDS_DEFAULT: int = 3


# Composite score formula text per the model-class hierarchy (recorded verbatim in provenance).
COMPOSITE_SCORE_FORMULA: dict[str, str] = {
    "primary": "mean_cv_geomean_relative_error",
    "secondary_tiebreaker": "median_absolute_hit_rate_error",
    "tertiary": "log_loss_eps_1e_minus_6_fractional_bernoulli",
    "simpler_model_tiebreaker_scope": "primary_metric_only_within_2pct_diff",
    "simpler_model_tiebreaker_threshold_fraction": "0.02",
}


# Log-loss epsilon clip per the model-class hierarchy.
LOG_LOSS_EPS: float = 1e-6


# Pre-registered model-class hierarchy per the model-class hierarchy (registration order =
# preference order; the simpler-model tiebreaker resolves ties on PRIMARY).
MODEL_CLASS_HIERARCHY: tuple[dict[str, Any], ...] = (
    {
        "name": "isotonic_regression_on_frozen_projection",
        "dependency_package": "sklearn",
        "import_path": "sklearn.isotonic",
        "parameter_count": 0,
        "tier": "preferred",
    },
    {
        "name": "monotone_gam_on_six_features",
        "dependency_package": "pygam",
        "import_path": "pygam",
        "parameter_count": 18,  # 6 features * 3 spline coefficients each (approx)
        "tier": "secondary",
    },
    {
        "name": "monotone_gbm_on_six_features",
        "dependency_package": "lightgbm",
        "import_path": "lightgbm",
        "parameter_count": 1024,  # depends on n_estimators; upper-bound budget
        "tier": "fallback",
    },
    {
        "name": "monotone_xgboost_last_resort",
        "dependency_package": "xgboost",
        "import_path": "xgboost",
        "parameter_count": 2048,
        "tier": "last_resort",
    },
)


def probe_dependency_availability() -> dict[str, dict[str, Any]]:
    """Probe each candidate model class's dependency import status.

    Returns a dict
      ``{class_name: {available: bool, version: str | None, import_path: str}}``
    where ``version`` is the installed package's ``__version__`` attribute when
    accessible, ``None`` otherwise.

    Pure runtime probe; safe to call on any host. Records dependency status
    for the v5 head's ``dependency_availability`` provenance field.
    """
    import importlib
    out: dict[str, dict[str, Any]] = {}
    for entry in MODEL_CLASS_HIERARCHY:
        name = entry["name"]
        import_path = entry["import_path"]
        pkg = entry["dependency_package"]
        try:
            mod = importlib.import_module(import_path)
            version = getattr(mod, "__version__", None)
            if version is None:
                try:
                    base = importlib.import_module(pkg)
                    version = getattr(base, "__version__", None)
                except ImportError:
                    version = None
            out[name] = {
                "available": True,
                "version": str(version) if version is not None else None,
                "import_path": import_path,
                "package": pkg,
            }
        except ImportError:
            out[name] = {
                "available": False,
                "version": None,
                "import_path": import_path,
                "package": pkg,
            }
    return out


def select_active_model_class(
    *,
    dependency_availability: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Walk the hierarchy in preference order and return the first available
    entry's metadata. When ``dependency_availability`` is None the function
    probes live; pass a recorded dict to make the selection reproducible from
    a frozen provenance file.

    Returns the hierarchy entry augmented with ``available=True`` and
    ``selected=True``. Raises ``RuntimeError`` when no class is available.
    """
    if dependency_availability is None:
        dependency_availability = probe_dependency_availability()
    for entry in MODEL_CLASS_HIERARCHY:
        name = entry["name"]
        status = dependency_availability.get(name, {})
        if status.get("available"):
            result = dict(entry)
            result["selected"] = True
            result["version"] = status.get("version")
            return result
    raise RuntimeError(
        "No model class from the the model-class hierarchy hierarchy is importable on this "
        "host. Install one of: sklearn (preferred), pygam, lightgbm, or "
        "xgboost (last resort)."
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def projected_scalar(features: dict) -> float:
    """Frozen 3-feature linear projection.

    Returns the scalar ``raw_prediction - footprint_ratio - long_rd_tail_mass``.
    The IR-on-projection model class consumes this scalar as its 1-D input.
    """
    s = 0.0
    for w, n in zip(FROZEN_PROJECTION_WEIGHTS, FROZEN_PROJECTION_FEATURES):
        s += w * float(features[n])
    return s


def _validate_score_inputs(
    predictions: Sequence[float], labels: Sequence[float],
) -> tuple[list[float], list[float]]:
    p = [float(x) for x in predictions]
    y = [float(x) for x in labels]
    if len(p) != len(y):
        raise ValueError(
            f"predictions and labels must have equal length; got "
            f"{len(p)} vs {len(y)}"
        )
    return p, y


def geomean_relative_error(
    predictions: Sequence[float],
    labels: Sequence[float],
    *,
    epsilon: float = 0.01,
) -> float:
    """Geometric mean of |p - y| / max(epsilon, |y|).

    Matches the comparison_runner contract's epsilon-floored relative-error
    formula. Returns 0.0 on empty input.
    """
    p, y = _validate_score_inputs(predictions, labels)
    if not p:
        return 0.0
    log_sum = 0.0
    for pi, yi in zip(p, y):
        denom = max(epsilon, abs(yi))
        rel = abs(pi - yi) / denom
        # +1e-12 to avoid log(0). 1e-12 is well below the the accuracy-gate thresholds gates' floors.
        log_sum += math.log(rel + 1e-12)
    return math.exp(log_sum / len(p))


def median_absolute_error(
    predictions: Sequence[float], labels: Sequence[float],
) -> float:
    """Median of |p - y|. Returns 0.0 on empty input."""
    p, y = _validate_score_inputs(predictions, labels)
    if not p:
        return 0.0
    abs_errs = sorted(abs(pi - yi) for pi, yi in zip(p, y))
    n = len(abs_errs)
    mid = n // 2
    if n % 2 == 1:
        return abs_errs[mid]
    return (abs_errs[mid - 1] + abs_errs[mid]) / 2.0


def log_loss_fractional_bernoulli(
    predictions: Sequence[float],
    labels: Sequence[float],
    *,
    eps: float = LOG_LOSS_EPS,
) -> float:
    """Cross-entropy on continuous [0, 1]-valued labels.

    ``- mean( y * log(p) + (1 - y) * log(1 - p) )`` with predictions clipped
    to ``[eps, 1 - eps]`` to keep the logarithms finite when the head emits
    a saturated 0 or 1.
    """
    p, y = _validate_score_inputs(predictions, labels)
    if not p:
        return 0.0
    total = 0.0
    for pi, yi in zip(p, y):
        p_clip = max(eps, min(1.0 - eps, pi))
        total -= (yi * math.log(p_clip) + (1.0 - yi) * math.log(1.0 - p_clip))
    return total / len(p)


def composite_score_select_simpler(
    *,
    candidates: list[dict[str, Any]],
    primary_tiebreaker_relative_threshold: float = 0.02,
) -> dict[str, Any]:
    """Select the winning candidate per the composite-score protocol.

    ``candidates`` is a list of dicts each carrying:
      - ``name``: model-class name
      - ``parameter_count``: integer (simpler = lower)
      - ``primary``: mean CV geomean rel err (lower = better)
      - ``secondary``: median absolute hit-rate err (lower = better, used as
                       tiebreaker)
      - ``tertiary``: log-loss (lower = better, used as second tiebreaker)

    Selection rule (per the goal tracker's the model-class hierarchy row):
      1. Find the candidate with the LOWEST primary score.
      2. If multiple candidates have primary scores within
         ``primary_tiebreaker_relative_threshold`` of the leader's primary
         score (RELATIVE: ``|candidate.primary - leader.primary| /
         max(|leader.primary|, eps) <= threshold``), pick the one with the
         FEWEST parameters (simpler-model tiebreaker on PRIMARY only).
      3. If a tie REMAINS (same parameter count), break on secondary then
         tertiary.

    Returns the winning candidate dict augmented with a ``selection_reason``
    string.
    """
    if not candidates:
        raise ValueError("composite_score_select_simpler requires >=1 candidate")
    sorted_by_primary = sorted(candidates, key=lambda c: c["primary"])
    leader = sorted_by_primary[0]
    leader_primary = leader["primary"]
    # Relative threshold: |delta| / max(|leader.primary|, eps) <= threshold.
    # The eps floor protects against divide-by-zero on a degenerate leader
    # primary = 0 case (which a future tightened predictor could produce).
    denom = max(abs(leader_primary), 1e-12)
    within_threshold = [
        c for c in sorted_by_primary
        if abs(c["primary"] - leader_primary) / denom
        <= primary_tiebreaker_relative_threshold
    ]
    if len(within_threshold) == 1:
        result = dict(within_threshold[0])
        result["selection_reason"] = (
            f"primary_unique_leader (primary={leader_primary:.6f})"
        )
        return result
    # Simpler-model tiebreaker on PRIMARY only (per the model-class hierarchy).
    sorted_by_simpler = sorted(
        within_threshold,
        key=lambda c: (
            c["parameter_count"],
            c["secondary"],
            c["tertiary"],
        ),
    )
    winner = sorted_by_simpler[0]
    result = dict(winner)
    result["selection_reason"] = (
        "simpler_model_tiebreaker (primary_within_"
        f"{primary_tiebreaker_relative_threshold * 100:.1f}pct_of_leader, "
        f"fewest_parameter_count={winner['parameter_count']})"
    )
    return result


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------


def _hash_cell(seed: int, identity: tuple) -> int:
    """Deterministic 64-bit hash of (seed, identity) via SHA-256 first 8 bytes.

    Pure-Python; reproducible across Python builds (unlike the stdlib ``hash``
    which randomizes per-process when PYTHONHASHSEED is unset).
    """
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    h.update(b":")
    h.update(repr(identity).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big")


def deterministic_fold_assignments(
    cell_identities: Sequence[tuple],
    stratum_keys: Sequence[tuple],
    *,
    seed: int = FOLD_SEED,
    n_folds: int = N_FOLDS_DEFAULT,
) -> tuple[dict[tuple, int], dict[str, Any]]:
    """Return ``(assignments, decision_record)``.

    Deterministic stratified round-robin construction:
      1. Group cells by stratum_key.
      2. Within each stratum, sort cells by ``hash(seed, identity)``.
      3. Assign folds round-robin in sorted order.
      4. If any stratum has fewer cells than n_folds, fall back to a
         non-stratified n-fold construction on the full hash-sorted list.
         The decision is recorded in ``decision_record["stratification"]``.
    """
    if len(cell_identities) != len(stratum_keys):
        raise ValueError(
            f"cell_identities and stratum_keys must have equal length; "
            f"got {len(cell_identities)} vs {len(stratum_keys)}"
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    by_stratum: dict[tuple, list[tuple]] = defaultdict(list)
    for ident, sk in zip(cell_identities, stratum_keys):
        by_stratum[sk].append(ident)

    # Minimum-per-fold criterion: each stratum must have >= n_folds cells
    # so the round-robin can place >= 1 cell from each stratum in each fold.
    smallest_stratum_n = (
        min(len(v) for v in by_stratum.values()) if by_stratum else 0
    )
    if smallest_stratum_n < n_folds:
        # Fall back to non-stratified n-fold.
        all_sorted = sorted(
            cell_identities,
            key=lambda ident: _hash_cell(seed, ident),
        )
        assignments: dict[tuple, int] = {}
        for i, ident in enumerate(all_sorted):
            assignments[ident] = i % n_folds
        decision_record = {
            "stratification": "non_stratified_fallback",
            "reason": (
                f"smallest_stratum_n={smallest_stratum_n} < n_folds={n_folds}"
            ),
            "n_folds": n_folds,
            "seed": seed,
            "stratum_sizes": {
                repr(k): len(v) for k, v in by_stratum.items()
            },
        }
        return assignments, decision_record

    # Stratified round-robin: within each stratum, sort by hash, assign folds.
    assignments = {}
    for sk in sorted(by_stratum.keys(), key=lambda x: repr(x)):
        members = sorted(
            by_stratum[sk], key=lambda ident: _hash_cell(seed, ident),
        )
        for i, ident in enumerate(members):
            assignments[ident] = i % n_folds
    decision_record = {
        "stratification": "stratified_round_robin",
        "n_folds": n_folds,
        "seed": seed,
        "stratum_sizes": {
            repr(k): len(v) for k, v in by_stratum.items()
        },
    }
    return assignments, decision_record


# ---------------------------------------------------------------------------
# Model class A: sklearn IsotonicRegression on frozen 3-feature projection
# ---------------------------------------------------------------------------


class PooledIsotonicWithBucketOneHot:
    """Pooled-fallback model class: IR on the frozen 3-feature projection
    PLUS a learned per-bucket one-hot offset, with the bucket index as a
    LOAD-BEARING MODEL INPUT (not an external bias correction).

    The fit + predict API takes ``family_bucket`` as part of the model's
    input row, so the training surface is the multivariate feature vector
    ``{raw_prediction, footprint_ratio, long_rd_tail_mass,
    family_bucket_one_hot}``. The internal training algorithm is a
    closed-form backfit:

      1. For each bucket, initialize ``offset[bucket] = 0``.
      2. Backfit (3 iterations is sufficient for convergence at this N):
         a. ``adjusted_label[i] = label[i] - offset[bucket_i]``
         b. Fit sklearn IsotonicRegression on
            ``(projection_scalar[i], adjusted_label[i])``.
         c. For each bucket, recompute
            ``offset[bucket] = mean(label[i] - ir.predict(projection_scalar[i]))``
            over rows in that bucket.

    The bucket offset is recorded under ``bucket_offsets`` as part of the
    model's parameters; ``used_features`` reports
    ``raw_prediction + footprint_ratio + long_rd_tail_mass +
    family_bucket_one_hot`` so consumers see the model's actual training
    surface. The pooled fallback's predict path requires
    ``family_bucket`` as an input alongside the feature dict.
    """

    name: str = "pooled_isotonic_with_bucket_one_hot"
    parameter_count: int = 1  # +1 over IR-on-projection for the bucket offset surface

    @property
    def used_features(self) -> tuple[str, ...]:
        return FROZEN_PROJECTION_FEATURES + ("family_bucket_one_hot",)

    def __init__(self) -> None:
        self._model: Any = None
        self._bucket_offsets: dict[str, float] = {}
        self._fitted: bool = False

    def fit(
        self,
        cells: Sequence[dict],
        *,
        backfit_iterations: int = 3,
    ) -> "PooledIsotonicWithBucketOneHot":
        if not cells:
            raise ValueError(
                "PooledIsotonicWithBucketOneHot.fit requires >= 1 cell"
            )
        if any("family_bucket" not in c for c in cells):
            raise ValueError(
                "PooledIsotonicWithBucketOneHot.fit requires every cell to "
                "carry family_bucket as a load-bearing model input"
            )
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise ImportError(
                "PooledIsotonicWithBucketOneHot requires scikit-learn."
            ) from exc
        buckets = sorted({c["family_bucket"] for c in cells})
        self._bucket_offsets = {b: 0.0 for b in buckets}
        scalars = [projected_scalar(c["features"]) for c in cells]
        labels = [float(c["label"]) for c in cells]
        cell_buckets = [c["family_bucket"] for c in cells]
        for _ in range(backfit_iterations):
            adjusted = [
                labels[i] - self._bucket_offsets[cell_buckets[i]]
                for i in range(len(cells))
            ]
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            ir.fit(scalars, adjusted)
            self._model = ir
            for b in buckets:
                residuals = [
                    labels[i] - float(ir.predict([scalars[i]])[0])
                    for i in range(len(cells))
                    if cell_buckets[i] == b
                ]
                if residuals:
                    self._bucket_offsets[b] = sum(residuals) / len(residuals)
        self._fitted = True
        return self

    def predict(self, features: dict, *, family_bucket: str) -> float:
        if not self._fitted or self._model is None:
            raise RuntimeError(
                "PooledIsotonicWithBucketOneHot.predict before fit"
            )
        s = projected_scalar(features)
        base = float(self._model.predict([s])[0])
        offset = self._bucket_offsets.get(family_bucket, 0.0)
        return max(0.0, min(1.0, base + offset))

    def monotonic_constraints(self) -> dict[str, int]:
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        out = {n: MONOTONIC_CONSTRAINT_SIGNS[n] for n in FROZEN_PROJECTION_FEATURES}
        # family_bucket_one_hot enters as an unconstrained categorical
        # covariate (sign=0); the IR + offset structure preserves monotonicity
        # in the projection dimension while the bucket offset is free.
        out["family_bucket_one_hot"] = 0
        return out


class PooledMonotoneWithBucketOneHot:
    """Generic pooled-fallback head over an ARBITRARY selected base class.

    Realizes ``family_bucket`` as a one-hot covariate (the per-bucket
    minimum-N rule) as a per-bucket additive offset -- a categorical
    main-effect term -- on top of whatever monotone base class the hierarchy
    selected::

        prediction = clip(base(features) + bucket_offset[family_bucket], 0, 1)

    The base is the SAME class chosen for the per-bucket heads, so a
    non-sklearn tier (``MonotoneGAMOnSixFeatures`` / ``MonotoneGBMOnSixFeatures``
    / the XGBoost last resort) does NOT drag sklearn into the pooled path, and
    the per-bucket provenance class name (``<selected>__pooled_fallback_with_one_hot``)
    names the class actually fit. Offsets are fit by the same 3-iteration
    backfit ``PooledIsotonicWithBucketOneHot`` uses; for an
    ``IsotonicRegressionOnFrozenProjection`` base the two are numerically
    identical, which is why ``PerBucketCalibratorEnsemble`` keeps the dedicated
    isotonic specialization for the IR base and uses THIS wrapper only for
    non-IR bases (so a frozen IR-selected head stays byte-identical on re-fit).

    The base must expose ``fit(features_per_cell, labels_per_cell)`` and
    ``predict(features)`` (the contract every registered model class honors).
    """

    def __init__(self, base_factory: Any) -> None:
        self._base_factory = base_factory
        self._base: Any = None
        self._bucket_offsets: dict[str, float] = {}
        self._fitted: bool = False
        proto = base_factory()
        self._base_used_features: tuple[str, ...] = tuple(
            getattr(proto, "used_features", CANONICAL_FEATURE_NAMES)
        )
        self._base_name: str = getattr(
            proto, "name", getattr(base_factory, "__name__", "unknown_base"),
        )

    @property
    def used_features(self) -> tuple[str, ...]:
        return tuple(self._base_used_features) + ("family_bucket_one_hot",)

    def fit(
        self,
        cells: Sequence[dict],
        *,
        backfit_iterations: int = 3,
    ) -> "PooledMonotoneWithBucketOneHot":
        if not cells:
            raise ValueError(
                "PooledMonotoneWithBucketOneHot.fit requires >= 1 cell"
            )
        if any("family_bucket" not in c for c in cells):
            raise ValueError(
                "PooledMonotoneWithBucketOneHot.fit requires every cell to "
                "carry family_bucket as a load-bearing model input"
            )
        buckets = sorted({c["family_bucket"] for c in cells})
        self._bucket_offsets = {b: 0.0 for b in buckets}
        features = [c["features"] for c in cells]
        labels = [float(c["label"]) for c in cells]
        cell_buckets = [c["family_bucket"] for c in cells]
        for _ in range(backfit_iterations):
            adjusted = [
                labels[i] - self._bucket_offsets[cell_buckets[i]]
                for i in range(len(cells))
            ]
            base = self._base_factory()
            base.fit(features, adjusted)
            self._base = base
            for b in buckets:
                residuals = [
                    labels[i] - float(base.predict(features[i]))
                    for i in range(len(cells))
                    if cell_buckets[i] == b
                ]
                if residuals:
                    self._bucket_offsets[b] = sum(residuals) / len(residuals)
        self._fitted = True
        return self

    def predict(self, features: dict, *, family_bucket: str) -> float:
        if not self._fitted or self._base is None:
            raise RuntimeError(
                "PooledMonotoneWithBucketOneHot.predict before fit"
            )
        base = float(self._base.predict(features))
        offset = self._bucket_offsets.get(family_bucket, 0.0)
        return max(0.0, min(1.0, base + offset))

    def monotonic_constraints(self) -> dict[str, int]:
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        out = {
            n: MONOTONIC_CONSTRAINT_SIGNS[n] for n in self._base_used_features
        }
        # family_bucket_one_hot enters as an unconstrained categorical
        # covariate (sign=0); the additive per-bucket offset preserves the base
        # model's monotonic surface in the continuous features.
        out["family_bucket_one_hot"] = 0
        return out


class IsotonicRegressionOnFrozenProjection:
    """Model class A from the model-class hierarchy.

    Fits a sklearn IsotonicRegression on the scalar projection
    ``+1 * raw_prediction - 1 * footprint_ratio - 1 * long_rd_tail_mass``.
    Parameter count is reported as 0 (non-parametric PAV) so the
    simpler-model tiebreaker prefers it on ties.
    """

    name: str = "isotonic_regression_on_frozen_projection"
    parameter_count: int = 0
    used_features: tuple[str, ...] = FROZEN_PROJECTION_FEATURES

    def __init__(self) -> None:
        self._model: Any = None
        self._fitted: bool = False

    def fit(
        self,
        features_per_cell: Sequence[dict],
        labels_per_cell: Sequence[float],
    ) -> "IsotonicRegressionOnFrozenProjection":
        if len(features_per_cell) != len(labels_per_cell):
            raise ValueError(
                f"features_per_cell and labels_per_cell must have equal "
                f"length; got {len(features_per_cell)} vs {len(labels_per_cell)}"
            )
        if not features_per_cell:
            raise ValueError(
                "IsotonicRegressionOnFrozenProjection.fit requires >= 1 cell"
            )
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise ImportError(
                "IsotonicRegressionOnFrozenProjection requires scikit-learn. "
                "Install sklearn or fall back to the next model-class tier "
                "(pygam GAM / LightGBM monotone GBM)."
            ) from exc
        scalars = [projected_scalar(f) for f in features_per_cell]
        labels = [float(y) for y in labels_per_cell]
        self._model = IsotonicRegression(
            increasing=True, out_of_bounds="clip",
        )
        self._model.fit(scalars, labels)
        self._fitted = True
        return self

    def predict(self, features: dict) -> float:
        if not self._fitted or self._model is None:
            raise RuntimeError(
                "IsotonicRegressionOnFrozenProjection.predict before fit"
            )
        s = projected_scalar(features)
        out = self._model.predict([s])
        return float(out[0])

    def monotonic_constraints(self) -> dict[str, int]:
        """Return the constraint signs for the features this class USES.

        Only the frozen projection's features appear -- the monotonic-constraint-sign table says the
        per-bucket provenance records constraints "only on used_features".
        """
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        return {
            n: MONOTONIC_CONSTRAINT_SIGNS[n]
            for n in self.used_features
        }


# ---------------------------------------------------------------------------
# PerBucketCalibratorEnsemble
# ---------------------------------------------------------------------------


class PerBucketCalibratorEnsemble:
    """Per-(arch, family_bucket) ensemble of monotonic heads.

    Per-bucket head when N >= ``PER_BUCKET_MIN_N``; otherwise the bucket's
    cells contribute to a pooled fallback head whose ``family_bucket`` enters
    as a one-hot covariate.

    The one-hot covariate is realized as a per-bucket bias offset on top of
    the pooled monotone fit. Concretely, the pooled head fits a single IR on
    the frozen 3-feature projection over ALL pooled cells, then per-bucket
    mean residuals (label - pooled_head.predict(features)) are stored as the
    bucket-specific bias. At predict time the pooled fallback returns
    ``clip(pooled_head.predict(features) + bucket_bias[bucket], 0.0, 1.0)``.
    This is the structural one-hot extension within the pooled head's
    monotonic surface; provenance records the expanded used_features
    accordingly.

    The ensemble exposes:

      - ``fit(cells)`` where each cell is a dict
        ``{identity: ..., arch: ..., family_bucket: ..., features: {...}, label: ...}``.
      - ``predict(arch, family_bucket, features)`` returning the calibrated
        scalar.
      - ``provenance_summary()`` returning the per-bucket kind + used_features
        table for ``calibration_head_v5.json``.
    """

    def __init__(self, model_class_factory: Any) -> None:
        self._model_class_factory = model_class_factory
        self._per_bucket: dict[tuple[str, str], Any] = {}
        self._pooled: Any | None = None
        self._pooled_bias: dict[tuple[str, str], float] = {}
        self._pooled_used_features: tuple[str, ...] = ()
        self._kinds: dict[tuple[str, str], str] = {}
        self._n_train: dict[tuple[str, str], int] = {}
        self._training_cell_ids: dict[tuple[str, str], list[tuple]] = (
            defaultdict(list)
        )
        self._fitted: bool = False

    def fit(
        self,
        cells: Iterable[dict],
        *,
        min_n: int = PER_BUCKET_MIN_N,
    ) -> "PerBucketCalibratorEnsemble":
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for cell in cells:
            key = (cell["arch"], cell["family_bucket"])
            groups[key].append(cell)
        pooled_cells: list[dict] = []
        pooled_keys: list[tuple[str, str]] = []
        for key, members in groups.items():
            self._n_train[key] = len(members)
            self._training_cell_ids[key] = [c["identity"] for c in members]
            if len(members) >= min_n:
                head = self._model_class_factory()
                head.fit(
                    [c["features"] for c in members],
                    [c["label"] for c in members],
                )
                self._per_bucket[key] = head
                self._kinds[key] = "per_bucket"
            else:
                self._kinds[key] = "pooled_fallback"
                pooled_cells.extend(members)
                pooled_keys.append(key)
        if pooled_cells:
            # Pooled fallback uses the SELECTED model class (the same class the
            # hierarchy chose for the per-bucket heads), with family_bucket as a
            # LOAD-BEARING one-hot covariate realized as a per-bucket additive
            # offset (not an external bias correction). Using the selected class
            # -- rather than always isotonic -- keeps a non-sklearn tier from
            # dragging in sklearn and keeps the per-bucket provenance class name
            # truthful. The bucket offset is part of the model's parameter set
            # and lands in provenance via the head's bucket_offsets attribute.
            pooled_head = self._build_pooled_head()
            pooled_head.fit(pooled_cells)
            self._pooled = pooled_head
            self._pooled_used_features = pooled_head.used_features
            for key in pooled_keys:
                bucket = key[1]
                self._pooled_bias[key] = pooled_head._bucket_offsets.get(
                    bucket, 0.0,
                )
        self._fitted = True
        return self

    def _build_pooled_head(self) -> Any:
        """Build the pooled-fallback head from the SELECTED model-class factory.

        The pooled fallback uses the SAME class the hierarchy selected, so a
        non-sklearn tier does not import sklearn and the downstream provenance
        class name (``<selected>__pooled_fallback_with_one_hot``) is truthful.
        The sklearn ``IsotonicRegressionOnFrozenProjection`` base keeps its
        dedicated ``PooledIsotonicWithBucketOneHot`` specialization so a frozen
        IR-selected head stays byte-identical on re-fit; every other base uses
        the generic additive-offset wrapper (numerically identical to the
        specialization for the IR base, which is why the dispatch is safe).
        """
        proto = self._model_class_factory()
        if (
            getattr(proto, "name", None)
            == IsotonicRegressionOnFrozenProjection.name
        ):
            return PooledIsotonicWithBucketOneHot()
        return PooledMonotoneWithBucketOneHot(self._model_class_factory)

    def predict(
        self,
        arch: str,
        family_bucket: str,
        features: dict,
    ) -> float:
        if not self._fitted:
            raise RuntimeError(
                "PerBucketCalibratorEnsemble.predict before fit"
            )
        key = (arch, family_bucket)
        head = self._per_bucket.get(key)
        if head is not None:
            return head.predict(features)
        if self._pooled is None:
            raise RuntimeError(
                f"no per-bucket head for {(arch, family_bucket)!r} and no "
                f"pooled fallback (the bucket admitted no cells at fit time)"
            )
        # Pooled fallback receives family_bucket as a load-bearing model input
        # via PooledIsotonicWithBucketOneHot's predict API (the bucket offset
        # is internal to the model's parameters, NOT an external bias
        # correction).
        return float(self._pooled.predict(
            features, family_bucket=family_bucket,
        ))

    def provenance_summary(self) -> dict[str, Any]:
        """Return the per-bucket provenance block for calibration_head_v5.json.

        Shape:
          {
            "by_bucket": {
              "<arch>/<family_bucket>": {
                "kind": "per_bucket" or "pooled_fallback",
                "n_train": int,
                "training_cell_ids": [tuple identities, ...],
                "pooled_bias": float | null  # only when kind == pooled_fallback
              },
              ...
            },
            "pooled_fallback_present": bool,
            "pooled_used_features": [feature_name, ...]  # includes
              "family_bucket_one_hot" when pooled fallback is present.
          }
        """
        by_bucket: dict[str, Any] = {}
        for key, kind in sorted(self._kinds.items(), key=lambda kv: kv[0]):
            arch, bucket = key
            entry: dict[str, Any] = {
                "kind": kind,
                "n_train": self._n_train[key],
                "training_cell_ids": [
                    list(t) for t in self._training_cell_ids[key]
                ],
            }
            if kind == "pooled_fallback":
                entry["pooled_bias"] = self._pooled_bias.get(key, 0.0)
            by_bucket[f"{arch}/{bucket}"] = entry
        return {
            "by_bucket": by_bucket,
            "pooled_fallback_present": self._pooled is not None,
            "pooled_used_features": list(self._pooled_used_features),
        }


class MonotoneGAMOnSixFeatures:
    """pygam.LinearGAM with sign-constrained smooths on all 6 canonical features.

    Per the goal tracker's model-class hierarchy row this is the secondary
    candidate, admitted when pygam is importable. Constraint signs follow
    MONOTONIC_CONSTRAINT_SIGNS verbatim:
      raw_prediction=+1 (monotonic_inc), footprint_ratio=-1 (monotonic_dec),
      long_rd_tail_mass=-1 (monotonic_dec), intra_lane_reuse=+1 (monotonic_inc),
      global_share=0 (unconstrained), gridDim_density=-1 (monotonic_dec).
    """

    name: str = "monotone_gam_on_six_features"
    parameter_count: int = 18

    @property
    def used_features(self) -> tuple[str, ...]:
        return CANONICAL_FEATURE_NAMES

    def __init__(self) -> None:
        self._model: Any = None
        self._fitted: bool = False

    def fit(
        self,
        features_per_cell: Sequence[dict],
        labels_per_cell: Sequence[float],
    ) -> "MonotoneGAMOnSixFeatures":
        try:
            from pygam import LinearGAM, s
        except ImportError as exc:
            raise RuntimeError(
                "MonotoneGAMOnSixFeatures requires pygam (>= 0.9.0). "
                "Install pygam or fall back to the next tier (LightGBM "
                "monotone GBM)."
            ) from exc
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        # Build per-feature constraints: monotonic_inc / monotonic_dec / None.
        terms = None
        for i, name in enumerate(CANONICAL_FEATURE_NAMES):
            sign = MONOTONIC_CONSTRAINT_SIGNS[name]
            if sign > 0:
                term = s(i, constraints="monotonic_inc")
            elif sign < 0:
                term = s(i, constraints="monotonic_dec")
            else:
                term = s(i)
            terms = term if terms is None else terms + term
        X = [
            [float(f[n]) for n in CANONICAL_FEATURE_NAMES]
            for f in features_per_cell
        ]
        y = [float(v) for v in labels_per_cell]
        self._model = LinearGAM(terms).fit(X, y)
        self._fitted = True
        return self

    def predict(self, features: dict) -> float:
        if not self._fitted or self._model is None:
            raise RuntimeError("MonotoneGAMOnSixFeatures.predict before fit")
        X = [[float(features[n]) for n in CANONICAL_FEATURE_NAMES]]
        return max(0.0, min(1.0, float(self._model.predict(X)[0])))

    def monotonic_constraints(self) -> dict[str, int]:
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        return {n: MONOTONIC_CONSTRAINT_SIGNS[n] for n in CANONICAL_FEATURE_NAMES}


class MonotoneGBMOnSixFeatures:
    """lightgbm.LGBMRegressor with monotone_constraints on the 6-feature
    vector. Per the goal tracker's model-class hierarchy row this is the
    fallback when pygam is absent."""

    name: str = "monotone_gbm_on_six_features"
    parameter_count: int = 1024

    @property
    def used_features(self) -> tuple[str, ...]:
        return CANONICAL_FEATURE_NAMES

    def __init__(self) -> None:
        self._model: Any = None
        self._fitted: bool = False

    def fit(
        self,
        features_per_cell: Sequence[dict],
        labels_per_cell: Sequence[float],
    ) -> "MonotoneGBMOnSixFeatures":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError(
                "MonotoneGBMOnSixFeatures requires lightgbm (>= 4.0). "
                "Install lightgbm or fall back to the last-resort XGBoost tier."
            ) from exc
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        constraints = [
            MONOTONIC_CONSTRAINT_SIGNS[n] for n in CANONICAL_FEATURE_NAMES
        ]
        X = [
            [float(f[n]) for n in CANONICAL_FEATURE_NAMES]
            for f in features_per_cell
        ]
        y = [float(v) for v in labels_per_cell]
        self._model = lgb.LGBMRegressor(
            monotone_constraints=constraints,
            n_estimators=100,
            verbose=-1,
        )
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, features: dict) -> float:
        if not self._fitted or self._model is None:
            raise RuntimeError("MonotoneGBMOnSixFeatures.predict before fit")
        X = [[float(features[n]) for n in CANONICAL_FEATURE_NAMES]]
        return max(0.0, min(1.0, float(self._model.predict(X)[0])))

    def monotonic_constraints(self) -> dict[str, int]:
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        return {n: MONOTONIC_CONSTRAINT_SIGNS[n] for n in CANONICAL_FEATURE_NAMES}


class MonotoneXGBoostLastResort:
    """xgboost.XGBRegressor with monotone_constraints, admitted only as a
    last-resort fallback when sklearn IR, pygam, and lightgbm are all
    unavailable."""

    name: str = "monotone_xgboost_last_resort"
    parameter_count: int = 2048

    @property
    def used_features(self) -> tuple[str, ...]:
        return CANONICAL_FEATURE_NAMES

    def __init__(self) -> None:
        self._model: Any = None
        self._fitted: bool = False

    def fit(
        self,
        features_per_cell: Sequence[dict],
        labels_per_cell: Sequence[float],
    ) -> "MonotoneXGBoostLastResort":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "MonotoneXGBoostLastResort requires xgboost. The host has "
                "no other model class from the registered hierarchy."
            ) from exc
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        constraints = tuple(
            MONOTONIC_CONSTRAINT_SIGNS[n] for n in CANONICAL_FEATURE_NAMES
        )
        X = [
            [float(f[n]) for n in CANONICAL_FEATURE_NAMES]
            for f in features_per_cell
        ]
        y = [float(v) for v in labels_per_cell]
        self._model = xgb.XGBRegressor(
            monotone_constraints=str(constraints).replace(" ", ""),
            n_estimators=100,
        )
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, features: dict) -> float:
        if not self._fitted or self._model is None:
            raise RuntimeError("MonotoneXGBoostLastResort.predict before fit")
        X = [[float(features[n]) for n in CANONICAL_FEATURE_NAMES]]
        return max(0.0, min(1.0, float(self._model.predict(X)[0])))

    def monotonic_constraints(self) -> dict[str, int]:
        from workloads.simbench_bw.canonical_features import (
            MONOTONIC_CONSTRAINT_SIGNS,
        )
        return {n: MONOTONIC_CONSTRAINT_SIGNS[n] for n in CANONICAL_FEATURE_NAMES}


__all__ = [
    "COMPOSITE_SCORE_FORMULA",
    "FOLD_SEED",
    "FROZEN_PROJECTION_FEATURES",
    "FROZEN_PROJECTION_WEIGHTS",
    "IsotonicRegressionOnFrozenProjection",
    "LOG_LOSS_EPS",
    "MODEL_CLASS_HIERARCHY",
    "MonotoneGAMOnSixFeatures",
    "MonotoneGBMOnSixFeatures",
    "MonotoneXGBoostLastResort",
    "N_FOLDS_DEFAULT",
    "PER_BUCKET_MIN_N",
    "PerBucketCalibratorEnsemble",
    "PooledIsotonicWithBucketOneHot",
    "PooledMonotoneWithBucketOneHot",
    "composite_score_select_simpler",
    "deterministic_fold_assignments",
    "geomean_relative_error",
    "log_loss_fractional_bernoulli",
    "median_absolute_error",
    "probe_dependency_availability",
    "projected_scalar",
    "select_active_model_class",
]
