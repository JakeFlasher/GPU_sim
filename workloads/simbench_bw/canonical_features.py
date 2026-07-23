"""Canonical pre-calibration feature assembler for the monotonic calibration head.

The monotonic calibration head consumes a 6-feature vector per (slug, shape_point,
arch) cell:

  - raw_prediction      : Smith-Eklund walker's per-launch arithmetic-mean hit
                          rate BEFORE any multiplicative correction. Pulled
                          from predictor_bundle.evidence.predicted_hit_rate_uncorrected.
  - footprint_ratio     : (unique_sector_count * sector_size_bytes) / cache_model_capacity_bytes.
                          A ratio in [0, ~100] expressing how many times the
                          working set overflows the modeled cache capacity.
  - long_rd_tail_mass   : fraction of sector-layer reuse-distance observations whose
                          rd_sector_units_min is at or above the long-RD threshold
                          (4096 sectors by default; matches the L2 high-footprint
                          signature predicate's threshold).
  - intra_lane_reuse    : sharing_locality.grounded[class_name=intra_lane].fraction.
                          Lane-local reuse fraction in [0, 1].
  - global_share        : fraction of sector_stream records whose memory_op_class
                          is "GLOBAL". The remaining records are SHARED, LOCAL,
                          PARAM, etc.
  - gridDim_density     : grid_volume / sm_count_for_device. A density in
                          [0, large] expressing per-SM CTA oversubscription.

Sign discipline (from Codex's monotone-constraint resolution; see the goal
tracker's the monotonic-constraint-sign table row): raw_prediction = +1, footprint_ratio = -1,
long_rd_tail_mass = -1, intra_lane_reuse = +1, global_share = 0
(unconstrained), gridDim_density = -1.

The module also owns the active-path domain-mismatch enforcement: the
``predict_v5_active_path`` entry-point rejects any per-slug correction kwarg
inherited from the legacy 3-stage multiplicative correction chain (gridDim
penalty + per-slug L1/L2 factors + Platt 2-param head). The retired path's
constants live in cache_predictor.py and are NOT imported here on purpose;
the import-discipline guard is enforced by the the canonical-features contract source-text scan.

Pure-Python; no numpy / scipy / sklearn imports. The downstream calibration
head module (calibration_head.py / monotonic_calibration_head.py) owns the
fit/predict surface and brings in sklearn lazily.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from arch_profiles import is_research_gap, load_profile


class ActivePathDomainMismatchError(Exception):
    """Raised when the active path receives a retired correction argument."""

    def __init__(
        self,
        *,
        offending_kwarg: str,
        value: Any,
        detail: str = "",
    ) -> None:
        self.offending_kwarg = offending_kwarg
        self.value = value
        self.detail = detail
        msg = (
            f"active path rejects legacy correction kwarg "
            f"{offending_kwarg!r}={value!r}"
        )
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)

    def to_json(self) -> dict[str, Any]:
        return {
            "exception": type(self).__name__,
            "offending_kwarg": self.offending_kwarg,
            "value": None if self.value is None else str(self.value),
            "detail": self.detail,
        }


MONOTONIC_CONSTRAINT_SIGNS: dict[str, int] = {
    "raw_prediction": +1,
    "footprint_ratio": -1,
    "long_rd_tail_mass": -1,
    "intra_lane_reuse": +1,
    "global_share": 0,
    "gridDim_density": -1,
}


CANONICAL_FEATURE_NAMES: tuple[str, ...] = tuple(
    MONOTONIC_CONSTRAINT_SIGNS.keys()
)


LONG_RD_TAIL_THRESHOLD_SECTORS: int = 4096


_FORBIDDEN_FACTOR_KWARGS: frozenset[str] = frozenset({
    "factor",
    "factor_override",
    "per_slug_factor",
    "l2_reduction_factor",
    "l2_high_footprint_factor",
    "gridDim_penalty_alpha",
})


_SM_COUNT_BY_ARCH: dict[str, int] = {
    "sm_86": 84,
    "sm_120": 26,  # measured on the live 5060 laptop (arch card); was a guessed 50
}


_DEFAULT_SM_COUNT: int = 84


class ArchitectureIdentityUnavailable(ValueError):
    """Typed fail-closed result for an unknown or ambiguous device."""

    TOKEN = "architecture_identity_missing"


    """Raised when the monotonic calibration head's active path is invoked
    with a per-slug correction factor or any kwarg inherited from the retired
    3-stage multiplicative correction chain (gridDim penalty + per-slug L1/L2
    factors + Platt 2-param head)."""

    def __init__(
        self,
        *,
        offending_kwarg: str,
        value: Any,
        detail: str = "",
    ) -> None:
        self.offending_kwarg = offending_kwarg
        self.value = value
        self.detail = detail
        msg = (
            f"active path rejects legacy correction kwarg "
            f"{offending_kwarg!r}={value!r}"
        )
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)

    def to_json(self) -> dict[str, Any]:
        return {
            "exception": type(self).__name__,
            "offending_kwarg": self.offending_kwarg,
            "value": None if self.value is None else str(self.value),
            "detail": self.detail,
        }


def _validate_feature_value(name: str, value: float) -> float:
    """Range-check a single feature value. Raises ValueError on out-of-range.

    Range table (per the canonical-features contract schema + negative-test contract):
      - raw_prediction, long_rd_tail_mass, intra_lane_reuse, global_share: [0, 1]
      - footprint_ratio: [0, 100]
      - gridDim_density: [0, +inf)
    """
    v = float(value)
    if math.isnan(v):
        raise ValueError(f"feature {name!r} is NaN")
    if math.isinf(v):
        raise ValueError(f"feature {name!r} is infinite")
    if name == "footprint_ratio":
        if v < 0.0 or v > 100.0:
            raise ValueError(
                f"feature {name!r}={v} outside permitted range [0, 100]"
            )
    elif name in {
        "raw_prediction",
        "long_rd_tail_mass",
        "intra_lane_reuse",
        "global_share",
    }:
        if v < 0.0 or v > 1.0:
            raise ValueError(
                f"feature {name!r}={v} outside permitted range [0, 1]"
            )
    elif name == "gridDim_density":
        if v < 0.0:
            raise ValueError(
                f"feature {name!r}={v} must be >= 0"
            )
    return v


def compute_footprint_ratio(
    *,
    unique_sector_count: int,
    cache_model_capacity_bytes: int,
    sector_size_bytes: int = 32,
) -> float:
    """(unique_sector_count * sector_size_bytes) / cache_model_capacity_bytes.

    A ratio in [0, ~100]. Values >= 1 mean the unique working set exceeds the
    modeled cache capacity; values >> 1 are the high-footprint signature.
    """
    if cache_model_capacity_bytes <= 0:
        raise ValueError(
            "cache_model_capacity_bytes must be positive; "
            f"got {cache_model_capacity_bytes}"
        )
    if sector_size_bytes <= 0:
        raise ValueError(
            f"sector_size_bytes must be positive; got {sector_size_bytes}"
        )
    if unique_sector_count < 0:
        raise ValueError(
            f"unique_sector_count must be >= 0; got {unique_sector_count}"
        )
    return (
        float(unique_sector_count * sector_size_bytes)
        / float(cache_model_capacity_bytes)
    )


def compute_long_rd_tail_mass(
    *,
    hrd_profile: dict,
    threshold_sectors: int = LONG_RD_TAIL_THRESHOLD_SECTORS,
) -> float:
    """Fraction of sector-layer observations with rd_sector_units_min >= threshold.

    Matches the formula in cache_predictor._l2_high_footprint_signature_check
    + test_l2_high_footprint_correction._ac5_read_per_cell_evidence so the
    feature value is byte-equivalent to the legacy signature predicate's input.
    """
    bins = hrd_profile.get("bins", [])
    sector_bins = [
        b for b in bins
        if b.get("hierarchy_layer") == "sector"
        and b.get("rd_sector_units_min") is not None
    ]
    total_sector_obs = sum(
        int(b.get("observation_count", 0)) for b in sector_bins
    )
    if total_sector_obs == 0:
        return 0.0
    long_rd_obs = sum(
        int(b.get("observation_count", 0))
        for b in sector_bins
        if int(b["rd_sector_units_min"]) >= threshold_sectors
    )
    return long_rd_obs / total_sector_obs


def compute_intra_lane_reuse(*, sharing_locality: dict) -> float:
    """sharing_locality.grounded[class_name=intra_lane].fraction.

    Returns 0.0 when the intra_lane entry is absent (e.g., the sharing_locality
    sidecar emitted a typed_unavailable for the grounded layer).
    """
    grounded = sharing_locality.get("grounded", [])
    for entry in grounded:
        if entry.get("class_name") == "intra_lane":
            return float(entry.get("fraction", 0.0))
    return 0.0


def compute_global_share(
    *,
    sector_stream_records: Iterable[dict] | None = None,
    global_count: int | None = None,
    total_count: int | None = None,
) -> float:
    """Either supply the precomputed (global_count, total_count) pair OR an
    iterable of sector_stream records to scan.

    A record is "global" when its memory_op_class field is the literal string
    "GLOBAL" (matches the legacy test fixture's scan logic).
    """
    if global_count is not None and total_count is not None:
        if total_count < 0 or global_count < 0:
            raise ValueError(
                "global_count and total_count must be non-negative"
            )
        if total_count == 0:
            return 0.0
        if global_count > total_count:
            raise ValueError(
                f"global_count={global_count} exceeds total_count={total_count}"
            )
        return float(global_count) / float(total_count)
    if sector_stream_records is None:
        raise ValueError(
            "compute_global_share: pass (global_count, total_count) OR "
            "sector_stream_records"
        )
    total = 0
    g = 0
    for rec in sector_stream_records:
        total += 1
        if rec.get("memory_op_class") == "GLOBAL":
            g += 1
    return (g / total) if total > 0 else 0.0


def _resolve_sm_count_for_device(device: str | None) -> int:
    """Resolve SM count from the identified architecture card.

    Unknown/ambiguous identities must not inherit the sm_86 denominator.
    """
    if device is None:
        raise ArchitectureIdentityUnavailable(
            "architecture_identity_missing: device identity is required"
        )
    try:
        profile = load_profile(device)
    except Exception as exc:
        raise ArchitectureIdentityUnavailable(
            f"architecture_identity_missing: cannot resolve {device!r}: {exc}"
        ) from exc
    sm_count = profile.sm_count
    if is_research_gap(sm_count) or not isinstance(sm_count, int) or sm_count <= 0:
        raise ArchitectureIdentityUnavailable(
            f"architecture_identity_missing: SM count unavailable for "
            f"{profile.arch_id!r}"
        )
    return int(sm_count)


def compute_gridDim_density(
    *,
    grid_volume: int | None,
    device: str | None,
) -> float:
    """grid_volume / sm_count_for_device.

    Returns 0.0 when grid_volume is None (legacy bundles without launch-parameter
    sidecar context). The active path treats absent grid information as a zero
    density, which the head's monotonic constraint admits without penalty.
    """
    if grid_volume is None:
        return 0.0
    if grid_volume < 0:
        raise ValueError(f"grid_volume must be >= 0; got {grid_volume}")
    sm_count = _resolve_sm_count_for_device(device)
    return float(grid_volume) / float(sm_count)


def assemble_canonical_features(
    *,
    predictor_bundle: dict,
    sharing_locality: dict,
    hrd_profile: dict,
    unique_sector_count: int,
    global_share_value: float | None = None,
    sector_stream_records: Iterable[dict] | None = None,
    grid_volume: int | None = None,
) -> dict[str, float]:
    """Assemble the 6-feature canonical pre-calibration vector.

    The bundle MUST carry ``evidence.predicted_hit_rate_uncorrected`` (the
    pre-correction Smith-Eklund walker output). The cache_model_capacity_bytes
    and (optionally) cache_model_grid_volume + device fields are sourced from
    the bundle's top-level provenance.

    ``unique_sector_count`` is passed explicitly by the caller because the
    legacy bundle schema does not expose it; the producer (e.g.,
    sm120_activation_bundle_producer.py) knows the value from the
    observation_builder pass that emitted the bundle.

    ``global_share_value`` may be supplied as a precomputed scalar to avoid
    re-scanning sector_stream; otherwise ``sector_stream_records`` is scanned.

    Returns a dict with the 6 feature names as keys, each value range-checked
    via _validate_feature_value (footprint_ratio in [0, 100]; the four-frequency
    features in [0, 1]; gridDim_density >= 0).
    """
    evidence = predictor_bundle.get("evidence", {})
    raw_pred = evidence.get("predicted_hit_rate_uncorrected")
    if raw_pred is None:
        raise ValueError(
            "predictor_bundle.evidence.predicted_hit_rate_uncorrected is "
            "required for the active path's raw_prediction feature"
        )
    raw_prediction = _validate_feature_value(
        "raw_prediction", float(raw_pred),
    )

    capacity_bytes = int(
        predictor_bundle.get("cache_model_capacity_bytes", 0)
    )
    sector_size_bytes = int(
        predictor_bundle.get("sector_size_bytes", 32)
    )
    footprint_ratio = _validate_feature_value(
        "footprint_ratio",
        compute_footprint_ratio(
            unique_sector_count=unique_sector_count,
            cache_model_capacity_bytes=capacity_bytes,
            sector_size_bytes=sector_size_bytes,
        ),
    )

    long_rd_tail_mass = _validate_feature_value(
        "long_rd_tail_mass",
        compute_long_rd_tail_mass(hrd_profile=hrd_profile),
    )

    intra_lane_reuse = _validate_feature_value(
        "intra_lane_reuse",
        compute_intra_lane_reuse(sharing_locality=sharing_locality),
    )

    if global_share_value is None:
        gs_value = compute_global_share(
            sector_stream_records=sector_stream_records,
        )
    else:
        gs_value = global_share_value
    global_share = _validate_feature_value("global_share", gs_value)

    bundle_grid_volume = predictor_bundle.get("cache_model_grid_volume")
    if bundle_grid_volume is not None:
        grid_volume_resolved: int | None = int(bundle_grid_volume)
    else:
        grid_volume_resolved = grid_volume
    device = predictor_bundle.get("device")
    gridDim_density = _validate_feature_value(
        "gridDim_density",
        compute_gridDim_density(
            grid_volume=grid_volume_resolved, device=device,
        ),
    )

    return {
        "raw_prediction": raw_prediction,
        "footprint_ratio": footprint_ratio,
        "long_rd_tail_mass": long_rd_tail_mass,
        "intra_lane_reuse": intra_lane_reuse,
        "global_share": global_share,
        "gridDim_density": gridDim_density,
    }


def _reject_factor_kwargs(kwargs: dict[str, Any]) -> None:
    """Raise ActivePathDomainMismatchError on any forbidden per-slug factor kwarg."""
    for name in _FORBIDDEN_FACTOR_KWARGS:
        if name in kwargs:
            raise ActivePathDomainMismatchError(
                offending_kwarg=name,
                value=kwargs[name],
                detail=(
                    "active path consumes only the canonical 6-feature vector "
                    "+ raw_prediction; per-slug factor injection is retired"
                ),
            )


def predict_active_path(
    *,
    bundle_with_features: dict,
    head_predict_fn: Any,
    **forbidden_kwargs: Any,
) -> dict[str, Any]:
    """Active path entry-point for the monotonic calibration head.

    Reads the 6-feature canonical block + raw_prediction from the bundle and
    invokes the head's predict function on the feature vector. Returns
    ``{calibrated_hit_rate: float, features_consumed: dict}``.

    Raises ActivePathDomainMismatchError on any per-slug factor kwarg
    (factor, factor_override, per_slug_factor, l2_reduction_factor,
    l2_high_footprint_factor, gridDim_penalty_alpha) -- these belong to the
    retired v4-and-earlier multiplicative correction chain and have no place
    in the active path.
    """
    _reject_factor_kwargs(forbidden_kwargs)
    features = bundle_with_features.get("canonical_pre_calibration_features")
    if features is None:
        raise ValueError(
            "active path requires bundle.canonical_pre_calibration_features "
            "block; backfill via assemble_canonical_features"
        )
    missing = [n for n in CANONICAL_FEATURE_NAMES if n not in features]
    if missing:
        raise ValueError(
            f"canonical_pre_calibration_features missing fields: {missing!r}"
        )
    extra = [n for n in features.keys() if n not in CANONICAL_FEATURE_NAMES]
    if extra:
        raise ValueError(
            f"canonical_pre_calibration_features has unexpected fields: "
            f"{extra!r}"
        )
    for name in CANONICAL_FEATURE_NAMES:
        _validate_feature_value(name, features[name])
    calibrated = head_predict_fn(features)
    return {
        "calibrated_hit_rate": float(calibrated),
        "features_consumed": dict(features),
    }


__all__ = [
    "ActivePathDomainMismatchError",
    "CANONICAL_FEATURE_NAMES",
    "LONG_RD_TAIL_THRESHOLD_SECTORS",
    "MONOTONIC_CONSTRAINT_SIGNS",
    "assemble_canonical_features",
    "compute_footprint_ratio",
    "compute_gridDim_density",
    "compute_global_share",
    "compute_intra_lane_reuse",
    "compute_long_rd_tail_mass",
    "predict_active_path",
]
