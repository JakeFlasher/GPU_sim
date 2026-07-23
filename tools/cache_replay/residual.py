"""At-most-one-dimensional monotone residual correction.

On top of the mechanistic set-associative replay prediction the calibrator
applies a single monotone correction: an isotonic (non-parametric, monotone)
map from the replay prediction to the calibrated hit ratio. This is the
``label - replay_prediction`` residual head restricted to one interpretable
feature and a monotone shape; no free-form machine-learning model is fit over
the cohort. A provenance guard rejects any residual head that declares more
than one free dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# The single feature the monotone residual is allowed to depend on.
RESIDUAL_FEATURE = "replay_predicted_hit_ratio"
RESIDUAL_MAX_FREE_DIMENSIONS = 1


@dataclass
class MonotoneResidual:
    """A fitted one-dimensional monotone (isotonic) residual head."""

    feature: str
    increasing: bool
    n_train: int
    _model: object = None  # sklearn IsotonicRegression

    def predict(self, replay_predictions: Sequence[float]) -> list[float]:
        if self._model is None:
            # identity fallback (unfit): pass the mechanistic prediction through
            return [min(1.0, max(0.0, float(p))) for p in replay_predictions]
        out = self._model.predict([float(p) for p in replay_predictions])
        return [min(1.0, max(0.0, float(v))) for v in out]

    def provenance(self) -> dict:
        return {
            "residual_kind": "isotonic_monotone_1d",
            "feature": self.feature,
            "free_dimensions": 1,
            "increasing": self.increasing,
            "n_train": self.n_train,
        }


def fit_monotone_residual(
    replay_predictions: Sequence[float],
    labels: Sequence[float],
    *,
    increasing: bool = True,
) -> MonotoneResidual:
    """Fit a one-dimensional isotonic residual: a monotone map from the replay
    prediction to the calibrated hit ratio."""
    from sklearn.isotonic import IsotonicRegression

    x = [float(p) for p in replay_predictions]
    y = [float(v) for v in labels]
    model = IsotonicRegression(
        increasing=increasing, out_of_bounds="clip", y_min=0.0, y_max=1.0,
    )
    model.fit(x, y)
    return MonotoneResidual(
        feature=RESIDUAL_FEATURE,
        increasing=increasing,
        n_train=len(x),
        _model=model,
    )


def validate_residual_provenance(prov: dict) -> None:
    """Raise ``ValueError`` if a residual-head provenance block declares more
    than one free dimension (the no-overfitting guard)."""
    free = prov.get("free_dimensions")
    if free is None:
        raise ValueError("residual provenance missing 'free_dimensions'")
    if not isinstance(free, int) or free > RESIDUAL_MAX_FREE_DIMENSIONS:
        raise ValueError(
            f"residual head declares {free!r} free dimensions; the monotone "
            f"residual admits at most {RESIDUAL_MAX_FREE_DIMENSIONS}"
        )
