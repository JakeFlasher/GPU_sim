#!/usr/bin/env python3
"""Shared, uncensored dispersion and classification primitives for M6."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

D2_EXACT = {5: 2.326, 10: 3.078, 20: 3.735, 30: 4.086}
CLASS_BOUNDARIES = (0.005, 0.02, 0.10)
DETERMINISM_CLASSES = (
    "vendor_defect",
    "insufficient_evidence",
    "multimodal",
    "deterministic",
    "quasi_stable",
    "clock_noise",
    "noisy",
    "unstable",
)


class DispersionError(ValueError):
    """A dispersion input violates the finite numeric contract."""


def _values(values: Sequence[float | int]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DispersionError("values must be JSON numbers, not booleans")
        value = float(value)
        if not math.isfinite(value):
            raise DispersionError("values must be finite")
        out.append(value)
    return out


def _resolution(value: float, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DispersionError("absolute resolution must be a JSON number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0 or (not allow_zero and resolved == 0.0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise DispersionError(f"absolute resolution must be finite and {qualifier}")
    return resolved


def dispersion_gate(rule: str, values: Sequence[float | int]) -> float | None:
    """Return the historical admission statistic without classification transforms."""

    vals = _values(values)
    if len(vals) < 2:
        return None
    mean = statistics.fmean(vals)
    if rule == "stddev_abs":
        return statistics.stdev(vals)
    if rule == "coefficient_of_variation":
        spread = statistics.stdev(vals)
        if mean == 0:
            return 0.0 if spread == 0 else None
        return spread / abs(mean)
    if rule == "relative_tolerance":
        spread = max(vals) - min(vals)
        if spread == 0:
            return 0.0
        if mean == 0:
            return None
        return spread / abs(mean)
    raise DispersionError(f"unknown repeatability rule: {rule}")


def d2_for_n(n: int) -> float:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise DispersionError("d2 requires integer N >= 2")
    if n in D2_EXACT:
        return D2_EXACT[n]
    # Blom order-statistic approximation. The four campaign tier values above
    # remain exact constants; this branch handles OOR-reduced sample counts.
    p = (n - 0.375) / (n + 0.25)
    return 2.0 * NormalDist().inv_cdf(p)


def sigma_rel(
    rule: str,
    values: Sequence[float | int],
    *,
    absolute_resolution: float = 0.0,
) -> float | None:
    """Map all registered repeatability rules to a common sigma-relative scale."""

    absolute_resolution = _resolution(absolute_resolution)
    vals = _values(values)
    if len(vals) < 2:
        return None
    if max(vals) == min(vals):
        return 0.0
    mean = statistics.fmean(vals)
    denominator = abs(mean)
    if denominator < absolute_resolution:
        denominator = absolute_resolution
    if denominator == 0:
        return None
    if rule in {"stddev_abs", "coefficient_of_variation"}:
        return statistics.stdev(vals) / denominator
    if rule == "relative_tolerance":
        return (max(vals) - min(vals)) / (d2_for_n(len(vals)) * denominator)
    raise DispersionError(f"unknown repeatability rule: {rule}")


def c4(n: int) -> float:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise DispersionError("c4 requires integer N >= 2")
    return math.exp(
        0.5 * math.log(2.0 / (n - 1))
        + math.lgamma(n / 2.0)
        - math.lgamma((n - 1) / 2.0)
    )


def sigma_upper_factor(n: int, confidence: float = 0.95) -> float:
    """One-sided sigma upper factor using the Wilson-Hilferty chi-square quantile."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise DispersionError("sigma upper factor requires integer N >= 2")
    if not 0.5 < confidence < 1.0:
        raise DispersionError("confidence must be in (0.5,1)")
    degrees = n - 1
    z = NormalDist().inv_cdf(1.0 - confidence)
    base = 1.0 - 2.0 / (9.0 * degrees) + z * math.sqrt(2.0 / (9.0 * degrees))
    if base <= 0:
        return float("inf")
    chi_lower = degrees * base**3
    return math.sqrt(degrees / chi_lower)


def classify_sigma(value: float | None, values: Sequence[float | int]) -> str:
    vals = _values(values)
    if len(vals) < 5 or value is None:
        return "insufficient_evidence"
    if max(vals) == min(vals):
        return "deterministic"
    if value <= CLASS_BOUNDARIES[0]:
        return "quasi_stable"
    if value <= CLASS_BOUNDARIES[1]:
        return "clock_noise"
    if value <= CLASS_BOUNDARIES[2]:
        return "noisy"
    return "unstable"


@dataclass(frozen=True)
class ModalityResult:
    flagged: bool
    outlier_flag: bool
    certified: bool
    suspected: bool
    split_index: int | None
    gap: float | None
    reason: str | None


def _continuous_delta_bic(values: Sequence[float], split_index: int, variance_floor: float) -> float:
    """Return BIC(single Gaussian) - BIC(two-component Gaussian mixture)."""

    vals = [float(value) for value in values]
    n = len(vals)
    floor = max(float(variance_floor) ** 2, 1e-30)

    def gaussian_logpdf(value: float, mean: float, variance: float) -> float:
        variance = max(variance, floor)
        return -0.5 * (math.log(2.0 * math.pi * variance) + (value - mean) ** 2 / variance)

    mean_one = statistics.fmean(vals)
    variance_one = max(statistics.fmean((value - mean_one) ** 2 for value in vals), floor)
    loglike_one = sum(gaussian_logpdf(value, mean_one, variance_one) for value in vals)

    left = vals[:split_index]
    right = vals[split_index:]
    weights = [len(left) / n, len(right) / n]
    means = [statistics.fmean(left), statistics.fmean(right)]
    variances = [
        max(statistics.fmean((value - means[index]) ** 2 for value in group), floor)
        for index, group in enumerate((left, right))
    ]
    loglike_two = float("-inf")
    for _ in range(100):
        responsibilities: list[tuple[float, float]] = []
        next_loglike = 0.0
        for value in vals:
            logs = [
                math.log(max(weights[index], 1e-12))
                + gaussian_logpdf(value, means[index], variances[index])
                for index in (0, 1)
            ]
            pivot = max(logs)
            total = math.exp(logs[0] - pivot) + math.exp(logs[1] - pivot)
            next_loglike += pivot + math.log(total)
            responsibilities.append(
                (math.exp(logs[0] - pivot) / total, math.exp(logs[1] - pivot) / total)
            )
        masses = [sum(row[index] for row in responsibilities) for index in (0, 1)]
        if min(masses) < 1e-6:
            return float("-inf")
        new_weights = [mass / n for mass in masses]
        new_means = [
            sum(row[index] * value for row, value in zip(responsibilities, vals)) / masses[index]
            for index in (0, 1)
        ]
        new_variances = [
            max(
                sum(
                    row[index] * (value - new_means[index]) ** 2
                    for row, value in zip(responsibilities, vals)
                )
                / masses[index],
                floor,
            )
            for index in (0, 1)
        ]
        weights, means, variances = new_weights, new_means, new_variances
        if abs(next_loglike - loglike_two) < 1e-10:
            loglike_two = next_loglike
            break
        loglike_two = next_loglike
    bic_one = 2 * math.log(n) - 2.0 * loglike_one
    bic_two = 5 * math.log(n) - 2.0 * loglike_two
    return bic_one - bic_two


def modality_result(
    values: Sequence[float | int],
    *,
    discrete: bool,
    absolute_resolution: float,
    tier2: bool,
) -> ModalityResult:
    absolute_resolution = _resolution(absolute_resolution)
    vals = sorted(_values(values))
    if len(vals) < 4:
        return ModalityResult(False, False, False, False, None, None, None)
    if discrete:
        counts: dict[float, int] = {}
        for value in vals:
            counts[value] = counts.get(value, 0) + 1
        distinct = sorted(counts)
        if len(distinct) < 2:
            return ModalityResult(False, False, False, False, None, None, None)
        recurrent = [value for value in distinct if counts[value] >= 2]
        flagged = len(recurrent) >= 2
        outlier = not flagged
        certified = tier2 and len([v for v in distinct if counts[v] >= 3]) >= 2
        return ModalityResult(
            flagged,
            outlier,
            certified,
            flagged and not certified,
            None,
            max(distinct) - min(distinct),
            "discrete_support" if flagged else "single_support_value",
        )

    gaps = [right - left for left, right in zip(vals, vals[1:])]
    largest_index = max(range(len(gaps)), key=gaps.__getitem__)
    largest = gaps[largest_index]
    left = vals[: largest_index + 1]
    right = vals[largest_index + 1 :]
    if len(left) == 1 or len(right) == 1:
        return ModalityResult(False, True, False, False, largest_index + 1, largest, "single_support_split")
    left_sd = statistics.stdev(left) if len(left) > 1 else 0.0
    right_sd = statistics.stdev(right) if len(right) > 1 else 0.0
    second = sorted(gaps, reverse=True)[1] if len(gaps) > 1 else 0.0
    mean = abs(statistics.fmean(vals))
    flagged = (
        largest >= 4.0 * max(left_sd, right_sd, absolute_resolution)
        and (mean == 0 or largest / mean >= 0.005)
        and (second == 0 or largest >= 2.5 * second)
    )
    support3 = len(left) >= 3 and len(right) >= 3
    delta_bic = (
        _continuous_delta_bic(vals, largest_index + 1, absolute_resolution)
        if flagged and tier2 and support3
        else float("-inf")
    )
    certified = flagged and tier2 and support3 and delta_bic > 6.0
    reason = None
    if flagged:
        reason = (
            "continuous_largest_gap_delta_bic"
            if certified
            else "continuous_largest_gap"
        )
    return ModalityResult(
        flagged,
        False,
        certified,
        flagged and not certified,
        largest_index + 1,
        largest,
        reason,
    )


def trigger_codes(
    values: Sequence[float | int],
    *,
    rule: str,
    absolute_resolution: float,
    discrete: bool,
    any_out_of_range: bool,
) -> tuple[str, ...]:
    vals = _values(values)
    codes: set[str] = set()
    modality = modality_result(
        vals,
        discrete=discrete,
        absolute_resolution=absolute_resolution,
        tier2=False,
    )
    if modality.flagged:
        codes.add("modality_flag")
    if modality.outlier_flag:
        codes.add("single_point_outlier")
    rel = sigma_rel(rule, vals, absolute_resolution=absolute_resolution)
    if rel is not None and rel > 0:
        if any(boundary / 2.0 <= rel <= boundary * 2.0 for boundary in CLASS_BOUNDARIES):
            codes.add("boundary_proximity")
    if any_out_of_range:
        codes.add("out_of_range")
    if discrete and len(set(vals)) >= 2:
        codes.add("discrete_multiple_values")
    return tuple(sorted(codes))


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(_values(values))
    if not vals:
        raise DispersionError("percentile requires values")
    if not 0.0 <= q <= 1.0:
        raise DispersionError("percentile q outside [0,1]")
    position = (len(vals) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return vals[lower]
    weight = position - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def pooled_p90_upper(
    values: Sequence[float], *, seed: int, replicates: int = 1000
) -> tuple[float, float]:
    vals = _values(values)
    if not vals:
        raise DispersionError("pooled floor requires values")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DispersionError("bootstrap seed must be an integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise DispersionError("bootstrap replicates must be positive")
    p90 = percentile(vals, 0.90)
    rng = random.Random(seed)
    boot = [
        percentile([vals[rng.randrange(len(vals))] for _ in vals], 0.90)
        for _ in range(replicates)
    ]
    return p90, percentile(boot, 0.90)


def median_bootstrap_ci(
    values: Sequence[float], *, seed: int, replicates: int = 1000
) -> tuple[float, float, float]:
    vals = _values(values)
    if not vals:
        raise DispersionError("median CI requires values")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DispersionError("bootstrap seed must be an integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise DispersionError("bootstrap replicates must be positive")
    rng = random.Random(seed)
    boot = [
        statistics.median(vals[rng.randrange(len(vals))] for _ in vals)
        for _ in range(replicates)
    ]
    return statistics.median(vals), percentile(boot, 0.025), percentile(boot, 0.975)


def benjamini_hochberg(
    p_values: Sequence[float | int], *, q: float = 0.05
) -> tuple[bool, ...]:
    """Return BH-FDR discoveries in the caller's original p-value order.

    The helper is deliberately pure and deterministic: ties are resolved by
    original position, inputs are never modified, and malformed probabilities
    fail closed instead of being silently clipped.
    """
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise DispersionError("FDR q must be a number")
    q_value = float(q)
    if not math.isfinite(q_value) or not 0.0 < q_value <= 1.0:
        raise DispersionError("FDR q must be finite and in (0,1]")
    probabilities: list[float] = []
    for value in p_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DispersionError("p-values must be JSON numbers, not booleans")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise DispersionError("p-values must be finite and in [0,1]")
        probabilities.append(probability)
    if not probabilities:
        return ()
    ordered = sorted(enumerate(probabilities), key=lambda pair: (pair[1], pair[0]))
    cutoff_rank = 0
    count = len(ordered)
    for rank, (_index, probability) in enumerate(ordered, start=1):
        if probability <= q_value * rank / count:
            cutoff_rank = rank
    discoveries = [False] * count
    if cutoff_rank:
        for index, _probability in ordered[:cutoff_rank]:
            discoveries[index] = True
    return tuple(discoveries)


# Short compatibility alias used by analysis callers.
bh_fdr = benjamini_hochberg
