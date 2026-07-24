"""Streaming-vs-reuse segmented L2 calibrator for the sm_86 expanded fit.

The single-trace SetAssociativeSimulator under-resolves dense L2 reuse: at low
`raw_l2` it cannot separate true-streaming dense kernels (true L2 read-hit ~ 0)
from small-reuse dense kernels (true L2 read-hit 0.1-0.8). This module adds the
deterministic dense-reuse feature from the Tier-2 frame —

    reuse_factor = total_sector_accesses / distinct_sectors

(the average number of times each cache line is touched; computed from the trace
ONLY, never the NCU label) — and uses it to SEGMENT the dense cohort before the
isotonic fit (the approach Codex R32 directed):

  * STREAMING segment (reuse_factor < threshold): each line is touched only ~1-N
    times -> first-touch-dominated -> calibrate L2 against `reuse_factor`.
  * REUSE segment (reuse_factor >= threshold): lines are re-read enough to be
    L2-resident -> the simulator's `raw_l2` LRU replay is informative ->
    calibrate L2 against `raw_l2`.

`threshold` is a FIXED PHYSICAL hyperparameter (4.0 = "a cache line read >= 4x on
average is L2-resident reuse, not streaming"), NOT tuned to the labels; the
held-out accuracy is reported via leave-one-out CV. This stays LOCAL to the sm_86
calibration path; the shared `IsotonicCalibrator` (L1 head) is untouched.

NVBit/NCU are calibration-only (CLAUDE.md Rule 1); nothing here ships to B200.
"""
from __future__ import annotations

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

from dataclasses import dataclass
from typing import Final

from sklearn.isotonic import IsotonicRegression

# Re-export the harvest schema version so the params emitter can cross-stamp it.
from sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep.harvest_sweep_traces import (  # noqa: E402,F401
    SIDECAR_SCHEMA_VERSION,
)

# Fixed physical segmentation threshold (avg cache-line touches). Pre-declared,
# label-independent; 4-5 is a stable plateau in the held-out MAPE.
SEGMENT_THRESHOLD: Final[float] = 4.0


@dataclass(frozen=True)
class L2Cell:
    slug: str
    bucket: str
    reuse_factor: float
    raw_l2: float
    ncu_l2_label: float


def _fit_iso(xs, ys) -> IsotonicRegression | None:
    refuse_legacy_ncu_calibration('sm86_expanded.dense_reuse_l2_calibrator._fit_iso')
    if len({round(x, 12) for x in xs}) < 2 or not xs:
        return None
    return IsotonicRegression(out_of_bounds="clip").fit(list(xs), list(ys))


@dataclass
class SegmentedL2Model:
    threshold: float
    streaming_head: IsotonicRegression | None  # reuse_factor -> label
    reuse_head: IsotonicRegression | None      # raw_l2 -> label
    streaming_mean: float                       # fallback when a head is undefined
    reuse_mean: float

    def predict(self, reuse_factor: float, raw_l2: float) -> float:
        if reuse_factor < self.threshold:
            if self.streaming_head is not None:
                return float(self.streaming_head.predict([reuse_factor])[0])
            return self.streaming_mean
        if self.reuse_head is not None:
            return float(self.reuse_head.predict([raw_l2])[0])
        return self.reuse_mean


def fit_segmented(cells: list[L2Cell], threshold: float = SEGMENT_THRESHOLD) -> SegmentedL2Model:
    refuse_legacy_ncu_calibration('sm86_expanded.dense_reuse_l2_calibrator.fit_segmented')
    stream = [c for c in cells if c.reuse_factor < threshold]
    reuse = [c for c in cells if c.reuse_factor >= threshold]
    sh = _fit_iso([c.reuse_factor for c in stream], [c.ncu_l2_label for c in stream])
    rh = _fit_iso([c.raw_l2 for c in reuse], [c.ncu_l2_label for c in reuse])
    smean = (sum(c.ncu_l2_label for c in stream) / len(stream)) if stream else 0.0
    rmean = (sum(c.ncu_l2_label for c in reuse) / len(reuse)) if reuse else 0.0
    return SegmentedL2Model(threshold, sh, rh, smean, rmean)


def loocv_l2_abs_errors(cells: list[L2Cell], threshold: float = SEGMENT_THRESHOLD) -> dict[str, float]:
    """Per-cell leave-one-out absolute L2 error (fraction domain). The held-out
    estimator of the segmented model's accuracy."""
    refuse_legacy_ncu_calibration('sm86_expanded.dense_reuse_l2_calibrator.loocv_l2_abs_errors')
    out: dict[str, float] = {}
    for i, c in enumerate(cells):
        train = [cells[j] for j in range(len(cells)) if j != i]
        model = fit_segmented(train, threshold)
        out[c.slug] = abs(model.predict(c.reuse_factor, c.raw_l2) - c.ncu_l2_label)
    return out


def is_identity(cells: list[L2Cell], eps: float = 1e-9) -> bool:
    """No-label-leak / non-trivial guard: the fitted L2 prediction must not equal
    the label for every cell (that would mean the feature IS the label)."""
    refuse_legacy_ncu_calibration('sm86_expanded.dense_reuse_l2_calibrator.is_identity')
    model = fit_segmented(cells)
    gaps = [abs(model.predict(c.reuse_factor, c.raw_l2) - c.ncu_l2_label) for c in cells]
    return max(gaps, default=0.0) <= eps
