"""Calibration heads for the cache hit-rate predictor.

Three calibrator classes that fit a 1-D monotonic transform from
predicted-hit-rate to label-hit-rate on small ground-truth
cohorts:

  - ``IsotonicCalibrator``: scikit-learn's
    ``IsotonicRegression(increasing=True, out_of_bounds="clip")``.
    Non-parametric piecewise-constant fit; best for irregular
    miscalibration but easy to overfit on small N.
  - ``PlattCalibrator``: hand-rolled 2-parameter logistic
    ``sigmoid(A * x + B)``. Pure numpy; Newton's-method fit.
    Smooth, low-DoF; matches the literature norm for small-N
    binary calibration extended to a continuous rate.
  - ``BetaCalibrator``: hand-rolled 3-parameter beta calibration
    per Kull-Filho-Flach 2017
    ``f(x) = sigmoid(a*log(x/(1-x)) + b*log(1-x) + c)``.
    Pure numpy; closed form for inference once ``(a, b, c)`` are
    fit via Newton's method.

All three calibrators share the same surface:

  - ``fit(predicted, label, weights=None)``: fit the parameters
    to the training pairs. Stores ``training_predicted_min``
    and ``training_predicted_max`` for the envelope check.
  - ``transform(predicted) -> (calibrated, out_of_envelope_mask)``:
    return the calibrated array AND a boolean mask of indices
    whose input value falls outside the training envelope. The
    consumer routes envelope-violating cells to a typed
    fail-closed reason (e.g.,
    ``calibration_reuse_distance_out_of_envelope`` in
    ``comparison_runner.py``) rather than trusting an
    extrapolated calibration value.

The module imports cleanly without ``numpy`` / ``scipy`` /
``sklearn`` installed; the optional dependencies are imported
lazily inside ``fit`` / ``transform`` so callers that only
need the type names (e.g., type annotations, isinstance checks)
do not pay the import cost or get an ImportError.

Pooled global head policy: the calibrator sees only
``(predicted, label)`` pairs. ``arch`` is NOT a fitted feature
and is NOT an argument here; the consumer records ``arch`` in
the training-row provenance sidecar separately so per-arch
residuals can be reported in the evaluation block without
implicit per-arch fits. This pinned policy matches the
small-N regime (the active cohort has at most 26 total cells
across sm_86 and sm_120, well below what a per-arch fit would
need to avoid noise).
"""

from __future__ import annotations

from typing import Any, Optional


def _require_numpy() -> Any:
    """Lazily import numpy and surface a clear ImportError if
    missing. The module itself imports cleanly without numpy;
    only ``fit`` and ``transform`` need it."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "calibration_head requires numpy at fit / transform "
            "time. Install numpy (and scipy + scikit-learn for "
            "IsotonicCalibrator) in the active environment."
        ) from exc
    return np


def _to_1d_array(arr_like: Any) -> Any:
    """Convert a Python list / tuple / numpy-array-like into a
    1-D numpy float array. Caller-side input shape is intentionally
    permissive."""
    np = _require_numpy()
    return np.asarray(arr_like, dtype=float).reshape(-1)


def _validate_fit_inputs(
    predicted: Any, label: Any, weights: Optional[Any],
) -> tuple[Any, Any, Optional[Any]]:
    """Coerce + shape-check + range-check the fit inputs.

    Raises ``ValueError`` if predicted / label / weights have
    mismatched lengths or if any value falls outside the
    closed unit interval ``[0, 1]`` (the calibrator's domain).
    """
    np = _require_numpy()
    p = _to_1d_array(predicted)
    y = _to_1d_array(label)
    if p.size != y.size:
        raise ValueError(
            f"predicted and label must have equal length; got "
            f"{p.size} vs {y.size}"
        )
    if p.size == 0:
        raise ValueError("calibrator fit requires at least one (predicted, label) pair")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(
            "predicted values must lie in [0, 1] (hit-rate domain)"
        )
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError(
            "label values must lie in [0, 1] (hit-rate domain)"
        )
    w = None
    if weights is not None:
        w = _to_1d_array(weights)
        if w.size != p.size:
            raise ValueError(
                f"weights must match predicted/label length; got "
                f"{w.size} vs {p.size}"
            )
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
    return p, y, w


class _BaseCalibrator:
    """Shared envelope-tracking + transform-shape behavior."""

    def __init__(self) -> None:
        self.training_predicted_min: float | None = None
        self.training_predicted_max: float | None = None
        self._fitted: bool = False

    def _record_envelope(self, predicted: Any) -> None:
        np = _require_numpy()
        self.training_predicted_min = float(np.min(predicted))
        self.training_predicted_max = float(np.max(predicted))

    def _envelope_mask(self, predicted: Any) -> Any:
        """Return a bool array True at indices outside the
        training envelope."""
        np = _require_numpy()
        p = _to_1d_array(predicted)
        if (
            self.training_predicted_min is None
            or self.training_predicted_max is None
        ):
            raise RuntimeError(
                "calibrator transform called before fit; "
                "envelope is undefined"
            )
        below = p < self.training_predicted_min
        above = p > self.training_predicted_max
        return np.logical_or(below, above)


class IsotonicCalibrator(_BaseCalibrator):
    """Non-parametric monotonic calibration via the pool-adjacent-
    violators algorithm (sklearn's ``IsotonicRegression``).

    Per DEC-T8: scikit-learn permitted for this class only;
    PlattCalibrator + BetaCalibrator are hand-rolled to keep the
    dependency surface narrow.
    """

    library_id: str = "sklearn"

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    def fit(
        self,
        predicted: Any,
        label: Any,
        weights: Optional[Any] = None,
    ) -> "IsotonicCalibrator":
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise ImportError(
                "IsotonicCalibrator requires scikit-learn. "
                "Install sklearn or use PlattCalibrator / "
                "BetaCalibrator instead."
            ) from exc
        p, y, w = _validate_fit_inputs(predicted, label, weights)
        self._model = IsotonicRegression(
            increasing=True, out_of_bounds="clip",
        )
        if w is None:
            self._model.fit(p, y)
        else:
            self._model.fit(p, y, sample_weight=w)
        self._record_envelope(p)
        self._fitted = True
        return self

    def transform(self, predicted: Any) -> tuple[Any, Any]:
        if not self._fitted or self._model is None:
            raise RuntimeError("IsotonicCalibrator.transform called before fit")
        np = _require_numpy()
        p = _to_1d_array(predicted)
        calibrated = self._model.transform(p)
        return np.asarray(calibrated, dtype=float), self._envelope_mask(p)


def _sigmoid(x: Any) -> Any:
    """Numerically-stable sigmoid using pure numpy. Avoids scipy
    so PlattCalibrator + BetaCalibrator do not require scipy
    for inference."""
    np = _require_numpy()
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


def _newton_fit_two_param(
    predicted: Any, label: Any, weights: Optional[Any],
    feature_fn: Any,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, ...]:
    """Generic Newton's-method MLE for a 2-parameter logistic
    family ``p(y=1 | x) = sigmoid(theta . f(x))`` where ``f(x)``
    is the ``feature_fn``-built feature matrix (shape (n, k)).

    Returns the fit ``theta`` tuple. The likelihood is the
    standard cross-entropy on the [0, 1]-valued `label`
    (continuous-label extension of Platt scaling — Platt 1999
    originally targeted binary labels; the continuous extension
    is mathematically identical with `label` interpreted as a
    success-rate target).
    """
    np = _require_numpy()
    X = feature_fn(predicted)  # shape (n, k)
    y = _to_1d_array(label)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = _to_1d_array(weights)
    n, k = X.shape
    theta = np.zeros(k, dtype=float)
    for _ in range(max_iter):
        eta = X @ theta
        p = _sigmoid(eta)
        # Score: X^T W (y - p)
        score = X.T @ (w * (y - p))
        # Hessian: -X^T W diag(p*(1-p)) X
        wpp = w * p * (1.0 - p)
        H = -(X.T * wpp) @ X
        # Ridge for numerical stability if the Hessian is
        # near-singular on small N.
        H = H - 1e-8 * np.eye(k)
        try:
            step = np.linalg.solve(H, score)
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        theta = theta - step
        if np.max(np.abs(step)) < tol:
            break
    return tuple(float(t) for t in theta)


class PlattCalibrator(_BaseCalibrator):
    """2-parameter logistic calibration:
    ``calibrated = sigmoid(A * predicted + B)``.

    Pure numpy; fit via Newton's-method MLE on the continuous
    cross-entropy. Lowest-DoF of the three calibrators (smooth
    and very hard to overfit).
    """

    library_id: str = "stdlib_handrolled"

    def __init__(self) -> None:
        super().__init__()
        self.A: float | None = None
        self.B: float | None = None

    def fit(
        self,
        predicted: Any,
        label: Any,
        weights: Optional[Any] = None,
    ) -> "PlattCalibrator":
        np = _require_numpy()
        p, y, w = _validate_fit_inputs(predicted, label, weights)

        def _platt_features(x: Any) -> Any:
            n = x.shape[0]
            return np.column_stack([x, np.ones(n)])

        theta = _newton_fit_two_param(p, y, w, _platt_features)
        A_hat, B_hat = theta
        # Monotonicity projection: sigmoid(A*x + B) is
        # non-decreasing iff A >= 0. If the unconstrained Newton
        # fit produced A < 0 (the anti-correlated-label regime),
        # fall back to A = 0 and re-fit B as the logit of the
        # weighted mean label, clipped away from 0 / 1 to avoid
        # +/-infinity. Under A = 0 the constant transform
        # sigmoid(B) is the maximum-likelihood estimate, so this
        # is the best constant monotone fallback under the same
        # cross-entropy objective.
        if A_hat < 0.0:
            if w is None:
                mean_label = float(np.mean(y))
            else:
                w_sum = float(np.sum(w))
                if w_sum <= 0.0:
                    mean_label = float(np.mean(y))
                else:
                    mean_label = float(np.sum(w * y) / w_sum)
            eps = 1e-6
            mean_clipped = min(max(mean_label, eps), 1.0 - eps)
            A_hat = 0.0
            B_hat = float(np.log(mean_clipped / (1.0 - mean_clipped)))
        self.A, self.B = float(A_hat), float(B_hat)
        self._record_envelope(p)
        self._fitted = True
        return self

    def transform(self, predicted: Any) -> tuple[Any, Any]:
        if not self._fitted or self.A is None or self.B is None:
            raise RuntimeError("PlattCalibrator.transform called before fit")
        np = _require_numpy()
        p = _to_1d_array(predicted)
        calibrated = _sigmoid(self.A * p + self.B)
        return np.asarray(calibrated, dtype=float), self._envelope_mask(p)


class BetaCalibrator(_BaseCalibrator):
    """3-parameter beta calibration per Kull-Filho-Flach 2017:
    ``calibrated = sigmoid(a * log(x/(1-x)) + b * log(1-x) + c)``.

    Pure numpy; fit via Newton's-method MLE. Strictly more
    expressive than Platt (3 parameters vs. 2) and strictly less
    expressive than Isotonic (parametric vs. non-parametric
    piecewise constant). Per DEC-T9's simplicity ordering, Beta
    is the most complex of the three.
    """

    library_id: str = "stdlib_handrolled"

    def __init__(self) -> None:
        super().__init__()
        self.a: float | None = None
        self.b: float | None = None
        self.c: float | None = None

    @staticmethod
    def _clip_for_logit(np_module: Any, x: Any, eps: float = 1e-6) -> Any:
        """Clamp x away from 0 and 1 so log(x) / log(1-x) stay
        finite. Beta calibration's feature transform is undefined
        at the open-interval boundaries."""
        return np_module.clip(x, eps, 1.0 - eps)

    def fit(
        self,
        predicted: Any,
        label: Any,
        weights: Optional[Any] = None,
    ) -> "BetaCalibrator":
        np = _require_numpy()
        p, y, w = _validate_fit_inputs(predicted, label, weights)
        # Feature transform: [log(x/(1-x)), log(1-x), 1].
        # The combination of log(x/(1-x)) and log(1-x) admits
        # log(x), log(1-x), and the intercept as linear
        # combinations, recovering the full beta family.
        p_clip = self._clip_for_logit(np, p)

        def _beta_features(x: Any) -> Any:
            x_clip = self._clip_for_logit(np, x)
            return np.column_stack([
                np.log(x_clip / (1.0 - x_clip)),
                np.log(1.0 - x_clip),
                np.ones(x_clip.shape[0]),
            ])

        theta = _newton_fit_two_param(p_clip, y, w, _beta_features)
        a_hat, b_hat, c_hat = theta
        # Monotonicity projection: for sigmoid(a*log(x/(1-x))
        # + b*log(1-x) + c) the derivative wrt x in (0, 1)
        # reduces to (a - b*x) / (x*(1-x)). The denominator is
        # strictly positive on the open interval, so the
        # transform is non-decreasing iff (a - b*x) >= 0 for all
        # x in (0, 1), which is equivalent to a >= 0 AND b <= a.
        # Project after the Newton fit: a := max(0, a); then
        # b := min(b, a). Per the prior Codex review's required
        # repair. The intercept c stays unchanged so the central
        # location of the transform is preserved (the projection
        # only constrains the shape).
        a_hat = max(0.0, a_hat)
        b_hat = min(b_hat, a_hat)
        self.a, self.b, self.c = float(a_hat), float(b_hat), float(c_hat)
        self._record_envelope(p)
        self._fitted = True
        return self

    def transform(self, predicted: Any) -> tuple[Any, Any]:
        if (
            not self._fitted
            or self.a is None
            or self.b is None
            or self.c is None
        ):
            raise RuntimeError("BetaCalibrator.transform called before fit")
        np = _require_numpy()
        p = _to_1d_array(predicted)
        p_clip = self._clip_for_logit(np, p)
        logit = (
            self.a * np.log(p_clip / (1.0 - p_clip))
            + self.b * np.log(1.0 - p_clip)
            + self.c
        )
        calibrated = _sigmoid(logit)
        return np.asarray(calibrated, dtype=float), self._envelope_mask(p)


__all__ = [
    "BetaCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
]
