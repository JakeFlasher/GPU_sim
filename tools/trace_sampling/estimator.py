"""Wave-segment stratified HT calibrated-ratio estimator.

The primary sampling unit is a wave SEGMENT (a full concurrency slice of whole
blocks). Per sampled segment the caller supplies the measured target totals `t_s`
(from a per-segment sub-trace replay with a ghost warm-up — the simulator has no
per-segment attribution, so this measurement is injected, not read from a single
full replay) and the EXACT same-run record count `x_s` (from the counting pass).

Estimates:
  - additive counts: Horvitz-Thompson  T̂ = Σ_C t_s + Σ_h (N_h/n_h) Σ_{S_h} t_s,
    then combined-ratio CALIBRATED by the exact record count  T̂_cal = T̂ · X/X̂.
  - rates: ratio of CALIBRATED HT totals (never a mean of per-segment rates).
The record-count estimator `X̂` is an exact self-check: if the known-exact `X`
falls OUTSIDE the bootstrap CI of `X̂`, the sampling design is biased for this
cell and the estimate is ABORTED (not calibrated away). Cycles are never
estimated here (`not_recoverable`). CIs are stratified cluster bootstrap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

# Stopping-rule thresholds.
RATE_HALFWIDTH_ABS = 0.02       # 95% CI half-width for rate targets
COUNT_HALFWIDTH_REL = 0.05      # 95% CI half-width, relative, for count targets
MIN_SEGMENTS = 8
MIN_PER_STRATUM = 2
BOOTSTRAP_B = 1000
BOOTSTRAP_ALPHA = 0.05


class EstimatorInputError(ValueError):
    """Typed refusal for a malformed or incomplete sampling design."""

    def __init__(self, reason: str, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class Segment:
    id: int
    stratum: int                      # certainty segments use stratum = -1
    measured: dict[str, float]        # target name -> measured total for this segment
    record_count: float               # x_s, exact same-run


@dataclass
class Stratum:
    id: int
    N_h: int                          # total segments in the stratum (population)
    sampled_ids: list[int]            # ids of sampled segments in this stratum


@dataclass
class RateTarget:
    name: str
    num: str                          # additive target used as numerator
    den: str                          # additive target used as denominator


@dataclass
class EstimateResult:
    point: dict[str, float]
    ci: dict[str, tuple[float, float]]
    converged: dict[str, bool]
    status: str                       # converged | budget_limited_wide_ci | unavailable
    calibration_ok: bool
    x_true: float
    x_hat: float
    not_recoverable: list[str] = field(default_factory=list)


def _ht_total(segments: dict[int, Segment], strata: list[Stratum],
              certainty_ids: list[int], target: str) -> float:
    total = 0.0
    for cid in certainty_ids:
        total += segments[cid].measured[target]
    for h in strata:
        n_h = len(h.sampled_ids)
        s = sum(segments[i].measured[target] for i in h.sampled_ids)
        total += (h.N_h / n_h) * s
    return total


def _ht_records(segments, strata, certainty_ids) -> float:
    total = 0.0
    for cid in certainty_ids:
        total += segments[cid].record_count
    for h in strata:
        n_h = len(h.sampled_ids)
        s = sum(segments[i].record_count for i in h.sampled_ids)
        total += (h.N_h / n_h) * s
    return total


def _resample(strata: list[Stratum], rng: np.random.Generator) -> list[Stratum]:
    """Stratified cluster bootstrap: resample sampled segment ids WITH
    replacement within each stratum (certainty segments are fixed)."""
    out = []
    for h in strata:
        ids = h.sampled_ids
        if len(ids) == h.N_h:
            # A finite-population census has no sampling uncertainty.
            out.append(h)
            continue
        drawn = [int(value) for value in rng.choice(
            ids, size=len(ids), replace=True)]
        out.append(Stratum(id=h.id, N_h=h.N_h, sampled_ids=drawn))
    return out


def _is_finite_nonnegative(value: object) -> bool:
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0.0)


def _validate_inputs(
        segments: dict[int, Segment], strata: list[Stratum],
        certainty_ids: list[int], x_true: float,
        count_targets: list[str], rate_targets: list[RateTarget],
        seed: int) -> list[str]:
    """Validate every item used by HT weighting before doing any arithmetic."""
    if type(seed) is not int or seed < 0:
        raise EstimatorInputError("invalid_seed", repr(seed))
    if not _is_finite_nonnegative(x_true):
        raise EstimatorInputError("invalid_exact_record_total", repr(x_true))
    if not isinstance(segments, dict) or not segments:
        raise EstimatorInputError("empty_segment_sample")
    if not isinstance(strata, list) or not strata:
        raise EstimatorInputError("empty_strata")
    if not isinstance(certainty_ids, list):
        raise EstimatorInputError("invalid_certainty_collection")
    if not isinstance(count_targets, list) or not isinstance(rate_targets, list):
        raise EstimatorInputError("invalid_target_collection")

    def require_target_name(name: object, context: str) -> str:
        if not isinstance(name, str) or not name:
            raise EstimatorInputError("invalid_target_name", context)
        return name

    count_names = [require_target_name(name, "count target")
                   for name in count_targets]
    if len(count_names) != len(set(count_names)):
        raise EstimatorInputError("duplicate_count_target")

    rate_names: list[str] = []
    required_targets = list(count_names)
    for index, target in enumerate(rate_targets):
        if not isinstance(target, RateTarget):
            raise EstimatorInputError(
                "invalid_rate_target", f"index={index}")
        rate_names.append(require_target_name(target.name, "rate target"))
        for role, name in (("numerator", target.num),
                           ("denominator", target.den)):
            required = require_target_name(name, f"rate {role}")
            if required not in required_targets:
                required_targets.append(required)
    if len(rate_names) != len(set(rate_names)):
        raise EstimatorInputError("duplicate_rate_target")
    if set(rate_names).intersection(count_names):
        raise EstimatorInputError("target_output_name_collision")
    if not required_targets:
        raise EstimatorInputError("empty_target_set")

    for key, segment in segments.items():
        if type(key) is not int or key < 0:
            raise EstimatorInputError("invalid_segment_id", repr(key))
        if not isinstance(segment, Segment) or segment.id != key:
            raise EstimatorInputError(
                "segment_key_id_mismatch", f"key={key}")
        if type(segment.stratum) is not int or segment.stratum < -1:
            raise EstimatorInputError(
                "invalid_segment_stratum", f"segment={key}")
        if not _is_finite_nonnegative(segment.record_count):
            raise EstimatorInputError(
                "invalid_segment_record_count", f"segment={key}")
        if not isinstance(segment.measured, dict):
            raise EstimatorInputError(
                "invalid_segment_measurements", f"segment={key}")
        for target in required_targets:
            if target not in segment.measured:
                raise EstimatorInputError(
                    "missing_segment_target",
                    f"segment={key}, target={target}")
            if not _is_finite_nonnegative(segment.measured[target]):
                raise EstimatorInputError(
                    "invalid_segment_target",
                    f"segment={key}, target={target}")

    certainty_seen: set[int] = set()
    for segment_id in certainty_ids:
        if type(segment_id) is not int or segment_id < 0:
            raise EstimatorInputError(
                "invalid_certainty_segment_id", repr(segment_id))
        if segment_id in certainty_seen:
            raise EstimatorInputError(
                "duplicate_certainty_segment", str(segment_id))
        segment = segments.get(segment_id)
        if segment is None:
            raise EstimatorInputError(
                "unknown_certainty_segment", str(segment_id))
        if segment.stratum != -1:
            raise EstimatorInputError(
                "certainty_stratum_mismatch", f"segment={segment_id}")
        certainty_seen.add(segment_id)

    stratum_seen: set[int] = set()
    ordinary_seen: set[int] = set()
    for stratum in strata:
        if not isinstance(stratum, Stratum):
            raise EstimatorInputError("invalid_stratum")
        if type(stratum.id) is not int or stratum.id < 0:
            raise EstimatorInputError("invalid_stratum_id", repr(stratum.id))
        if stratum.id in stratum_seen:
            raise EstimatorInputError("duplicate_stratum_id", str(stratum.id))
        stratum_seen.add(stratum.id)
        if type(stratum.N_h) is not int or stratum.N_h <= 0:
            raise EstimatorInputError(
                "invalid_stratum_population", f"stratum={stratum.id}")
        if not isinstance(stratum.sampled_ids, list):
            raise EstimatorInputError(
                "invalid_stratum_sample", f"stratum={stratum.id}")
        n_h = len(stratum.sampled_ids)
        if n_h < MIN_PER_STRATUM:
            raise EstimatorInputError(
                "stratum_sample_below_floor",
                f"stratum={stratum.id}, n_h={n_h}")
        if n_h > stratum.N_h:
            raise EstimatorInputError(
                "stratum_sample_exceeds_population",
                f"stratum={stratum.id}, n_h={n_h}, N_h={stratum.N_h}")
        local_seen: set[int] = set()
        for segment_id in stratum.sampled_ids:
            if type(segment_id) is not int or segment_id < 0:
                raise EstimatorInputError(
                    "invalid_sampled_segment_id", repr(segment_id))
            if segment_id in local_seen:
                raise EstimatorInputError(
                    "duplicate_segment_within_stratum", str(segment_id))
            if segment_id in ordinary_seen:
                raise EstimatorInputError(
                    "segment_in_multiple_strata", str(segment_id))
            if segment_id in certainty_seen:
                raise EstimatorInputError(
                    "certainty_ordinary_overlap", str(segment_id))
            segment = segments.get(segment_id)
            if segment is None:
                raise EstimatorInputError(
                    "unknown_sampled_segment", str(segment_id))
            if segment.stratum != stratum.id:
                raise EstimatorInputError(
                    "sampled_segment_stratum_mismatch",
                    f"segment={segment_id}, expected={stratum.id}, "
                    f"observed={segment.stratum}")
            local_seen.add(segment_id)
            ordinary_seen.add(segment_id)

    accounted = certainty_seen.union(ordinary_seen)
    if accounted != set(segments):
        raise EstimatorInputError(
            "unaccounted_sample_segments",
            repr(sorted(set(segments).difference(accounted))))
    return required_targets


def estimate(segments: dict[int, Segment], strata: list[Stratum],
             certainty_ids: list[int], x_true: float,
             count_targets: list[str], rate_targets: list[RateTarget],
             *, seed: int = 0) -> EstimateResult:
    """Full estimate + stratified-cluster-bootstrap CIs + calibration abort."""
    all_targets = _validate_inputs(
        segments, strata, certainty_ids, x_true, count_targets, rate_targets,
        seed)
    n_sampled = len(certainty_ids) + sum(len(h.sampled_ids) for h in strata)

    # Point HT totals + calibration factor from the exact record count.
    x_hat = _ht_records(segments, strata, certainty_ids)
    cal = (x_true / x_hat) if x_hat > 0 else 1.0
    raw = {t: _ht_total(segments, strata, certainty_ids, t) for t in all_targets}
    point_counts = {t: raw[t] * cal for t in count_targets}
    point_rates = {}
    for rt in rate_targets:
        den = raw[rt.den] * cal
        point_rates[rt.name] = (raw[rt.num] * cal / den) if den > 0 else 0.0

    # Bootstrap.
    rng = np.random.default_rng(seed)
    boot_counts = {t: [] for t in count_targets}
    boot_rates = {rt.name: [] for rt in rate_targets}
    boot_xhat = []
    for _ in range(BOOTSTRAP_B):
        rs = _resample(strata, rng)
        bx = _ht_records(segments, rs, certainty_ids)
        boot_xhat.append(bx)
        bcal = (x_true / bx) if bx > 0 else 1.0
        braw = {t: _ht_total(segments, rs, certainty_ids, t) for t in all_targets}
        for t in count_targets:
            boot_counts[t].append(braw[t] * bcal)
        for rt in rate_targets:
            bden = braw[rt.den] * bcal
            boot_rates[rt.name].append((braw[rt.num] * bcal / bden) if bden > 0 else 0.0)

    lo_q, hi_q = 100 * BOOTSTRAP_ALPHA / 2, 100 * (1 - BOOTSTRAP_ALPHA / 2)

    def ci(vals):
        return (float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q)))

    # Calibration self-check: x_true must lie inside the bootstrap CI of x_hat.
    x_lo, x_hi = ci(boot_xhat)
    # Bootstrap percentiles and the analytically exact auxiliary total can differ
    # by a few ulps after HT weighting.  Apply one scale-aware numeric tolerance
    # here, at the estimator contract, so callers never need a local override.
    x_eps = 1e-12 * max(1.0, abs(x_true))
    calibration_ok = x_lo - x_eps <= x_true <= x_hi + x_eps

    point: dict[str, float] = {}
    ci_out: dict[str, tuple[float, float]] = {}
    converged: dict[str, bool] = {}
    for t in count_targets:
        point[t] = point_counts[t]
        lo, hi = ci(boot_counts[t])
        ci_out[t] = (lo, hi)
        hw = (hi - lo) / 2.0
        converged[t] = point[t] > 0 and (hw / point[t]) <= COUNT_HALFWIDTH_REL
    for rt in rate_targets:
        point[rt.name] = point_rates[rt.name]
        lo, hi = ci(boot_rates[rt.name])
        ci_out[rt.name] = (lo, hi)
        converged[rt.name] = ((hi - lo) / 2.0) <= RATE_HALFWIDTH_ABS

    floors_ok = n_sampled >= MIN_SEGMENTS
    if not calibration_ok:
        status = "unavailable"
    elif floors_ok and all(converged.values()):
        status = "converged"
    else:
        status = "budget_limited_wide_ci"

    return EstimateResult(point=point, ci=ci_out, converged=converged,
                          status=status, calibration_ok=calibration_ok,
                          x_true=x_true, x_hat=x_hat,
                          not_recoverable=["cycles", "sol_pct_timing"])
