#!/usr/bin/env python3
"""Per-launch alignment join + covariate regression layer for the
NVML covariate path (non-root measurement task6 + task7).

Reads a paired (CUPTI Activity sidecar, NVML covariate sidecar) and:

  1. Verifies the in-process natural-key tuple (slug, shape_point,
     repeat_index, binary_sha256, preflight_manifest_sha256) matches
     between the two producers; if not, emits a typed
     `nvml_join_key_mismatch` diagnostic. The per-launch
     `kernel_signature_sha256` is intentionally absent from the NVML
     side — the non-root measurement contract does not name it for the per-repeat
     NVML sidecar, and the cross-process producer-pair join
     (dual_run_manifest schema, CUPTI run vs NVBit run) carries
     per-launch signatures in its own schema.

  2. For each CUPTI Activity per-launch (start_ns, end_ns) window,
     counts how many NVML samples fall within the window. Launch
     windows with zero samples are recorded as alignment gaps.

  3. When at least one launch window has zero NVML samples, returns
     a typed `nvml_alignment_gap` diagnostic listing every uncovered
     window so downstream consumers can surface the gap in their
     per-host provenance.

  4. Fits a covariate regression of the form
         T_wall_corrected = T_wall
           - beta_temp  * (temp_c     - temp_ref_c)
           - beta_power * (power_mw   - power_ref_mw)
           - beta_clock * (sm_clock_mhz - sm_clock_ref_mhz)
     using least-squares on the per-launch covariate aggregates, and
     reports the percent reduction in |gap_ns| residual variance vs
     uncorrected `T_wall`. Below the threshold registered in
     `nvml_covariate_acceptance.toml`, the result includes a typed
     `nvml_covariate_regression_ineffective` diagnostic with the
     actual reduction percentage.

The module exposes a Python API (`join_cupti_nvml`,
`join_cupti_nvml_paths`) and a `python3 nvml_covariate_join.py`
script entry point. The non-root measurement `simbench_bw_validation
--metric=...` execution surface is task23 territory and is not yet
plumbed through this module.

Exit codes:
  0  join + regression succeeded; the typed JSON envelope reports the
     verdict.
  1  join failed (key mismatch, schema mismatch, or required input
     missing).
  2  usage error.
"""

from __future__ import annotations

import json
import math
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUITE_ROOT.parent.parent
ACCEPTANCE_TOML = SUITE_ROOT / "nvml_covariate_acceptance.toml"
# The committed calibration artifact emitted by
# tools/local_labels/calibrate_nvml_reference_operating_point.py (a real
# read-only NVML sweep on the local RTX 3090 Ti). It is the source of truth for
# the reference operating point; the [reference] block in the acceptance TOML is
# a fallback that is RETRACTED (nan/-1) on this branch until calibrated.
REFERENCE_ARTIFACT = (
    REPO_ROOT / "data" / "local_labels"
    / "nvml_reference_operating_point_sm_86.json"
)
# The SHORT-BURST anchor (cool, un-throttled full-boost ~1.87 GHz operating
# point) produced by tools/local_labels/calibrate_nvml_short_burst_anchor.py.
# A SHORT kernel runs in this window, not at the sustained 90 C / ~1350 MHz
# equilibrium the REFERENCE_ARTIFACT records; converting a short kernel's cycle
# count at the sustained clock over-predicts ns by ~38 %.
SHORT_BURST_ARTIFACT = (
    REPO_ROOT / "data" / "local_labels"
    / "nvml_short_burst_anchor_sm_86.json"
)
# The committed anchor artifact for each regime. The short regime selects the
# short_burst anchor; consuming the sustained anchor for a short-regime
# conversion is rejected (see load_reference_operating_point).
_ANCHOR_ARTIFACT_BY_REGIME = {
    "sustained": REFERENCE_ARTIFACT,
    "short_burst": SHORT_BURST_ARTIFACT,
}
_VALID_ANCHOR_REGIMES = frozenset(_ANCHOR_ARTIFACT_BY_REGIME)

# Plausibility band for a calibrated SM-clock reference (GA102 / RTX 3090 Ti).
# A reference outside this band is treated as unset/implausible and fails closed.
_MIN_PLAUSIBLE_SM_CLOCK_REF_MHZ = 300.0
_MAX_PLAUSIBLE_SM_CLOCK_REF_MHZ = 3000.0


# --- Typed diagnostics ------------------------------------------------------


class NvmlJoinKeyMismatch(Exception):
    TOKEN = "nvml_join_key_mismatch"

    def __init__(self, mismatched_field: str, cupti_value: Any, nvml_value: Any):
        super().__init__(
            f"{self.TOKEN}: field {mismatched_field!r} differs — "
            f"cupti={cupti_value!r} nvml={nvml_value!r}"
        )
        self.mismatched_field = mismatched_field
        self.cupti_value = cupti_value
        self.nvml_value = nvml_value

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "mismatched_field": self.mismatched_field,
            "cupti_value": self.cupti_value,
            "nvml_value": self.nvml_value,
        }


# --- Acceptance registry loader --------------------------------------------


@dataclass(frozen=True)
class NvmlAcceptance:
    min_residual_variance_reduction_pct: float
    temp_ref_c: float
    power_ref_mw: float
    sm_clock_ref_mhz: float
    mem_clock_ref_mhz: float


class NvmlReferenceUnset(Exception):
    """The NVML reference operating point is unset/retracted (the calibration
    sweep has not populated it) — the regression / ns-conversion MUST fail closed
    rather than centering deviations on a placeholder."""

    TOKEN = "nvml_covariate_reference_unset"

    def __init__(self, detail: str):
        super().__init__(f"{self.TOKEN}: {detail}")
        self.detail = detail

    def to_json(self) -> dict[str, Any]:
        return {"diagnostic": self.TOKEN, "detail": self.detail}


class NvmlClockImplausible(Exception):
    """An NVML clock value (reference or per-sample) is outside the physically
    plausible band — fail closed rather than convert cycles->ns on a bad clock."""

    TOKEN = "nvml_clock_implausible"

    def __init__(self, detail: str, value: Any):
        super().__init__(f"{self.TOKEN}: {detail} (value={value!r})")
        self.detail = detail
        self.value = value

    def to_json(self) -> dict[str, Any]:
        return {"diagnostic": self.TOKEN, "detail": self.detail, "value": self.value}


class NvmlAnchorRegimeMismatch(Exception):
    """The loaded NVML anchor artifact does not carry the requested regime
    discriminator — e.g. a short-regime conversion resolved the sustained
    (anchor_regime=sustained, ~1350 MHz) artifact, or an artifact lacks the
    anchor_regime field entirely. Fail closed rather than convert a short
    kernel's cycles at the wrong (sustained-throttle) clock."""

    TOKEN = "nvml_anchor_regime_mismatch"

    def __init__(self, detail: str, requested_regime: str, found_regime: Any):
        super().__init__(
            f"{self.TOKEN}: {detail} (requested={requested_regime!r}, "
            f"found={found_regime!r})"
        )
        self.detail = detail
        self.requested_regime = requested_regime
        self.found_regime = found_regime

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "detail": self.detail,
            "requested_regime": self.requested_regime,
            "found_regime": self.found_regime,
        }


def _is_unset_ref_value(value: Any) -> bool:
    """A reference field is 'unset' when it is None, NaN, or a negative sentinel
    (the retracted block uses nan / -1)."""
    if value is None:
        return True
    try:
        f = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or f < 0


def load_reference_operating_point(
    *,
    regime: str = "sustained",
    artifact_path: Path | None = None,
    toml_path: Path = ACCEPTANCE_TOML,
) -> dict[str, float]:
    """Return the calibrated reference operating point for ``regime``, preferring
    the committed calibration artifact and (for the sustained regime only)
    falling back to the acceptance TOML's [reference] block.

    ``regime`` selects which anchor to load: ``sustained`` (the 90 C SW-thermal
    boost-equilibrium ~1350 MHz point, the default for backward compatibility) or
    ``short_burst`` (the cool full-boost ~1.87 GHz point a short kernel sees).
    When the loaded artifact carries an ``anchor_regime`` field it MUST match the
    requested regime, so a short-regime conversion CANNOT silently consume the
    sustained anchor; for ``short_burst`` the discriminator is REQUIRED (an
    artifact lacking it raises :class:`NvmlAnchorRegimeMismatch`).

    Fail-closed: raises :class:`NvmlReferenceUnset` when neither source carries a
    populated reference (the retracted nan/-1 placeholders count as unset), and
    :class:`NvmlClockImplausible` when the SM-clock reference is outside the
    plausible band. NEVER substitutes a default. The returned dict carries
    ``temp_ref_c`` / ``power_ref_mw`` / ``sm_clock_ref_mhz`` / ``mem_clock_ref_mhz``
    plus a ``source`` provenance key."""
    if regime not in _VALID_ANCHOR_REGIMES:
        raise ValueError(
            f"unknown anchor regime {regime!r}; expected one of "
            f"{sorted(_VALID_ANCHOR_REGIMES)}"
        )
    if artifact_path is None:
        artifact_path = _ANCHOR_ARTIFACT_BY_REGIME[regime]
    ref: dict[str, Any] | None = None
    source = ""
    if artifact_path.is_file():
        art = json.loads(artifact_path.read_text())
        found_regime = art.get("anchor_regime")
        # Regime discriminator: a loaded artifact carrying anchor_regime MUST
        # match the requested regime (so a short-regime conversion cannot consume
        # the sustained anchor). For the short_burst regime the discriminator is
        # REQUIRED — an artifact lacking it fails closed. The sustained regime
        # tolerates a missing discriminator for backward compatibility with
        # legacy / synthetic-fixture artifacts that predate the field.
        if found_regime is not None and found_regime != regime:
            raise NvmlAnchorRegimeMismatch(
                f"artifact {artifact_path.name} carries anchor_regime "
                f"{found_regime!r} but a {regime!r}-regime conversion was "
                f"requested; refusing to convert at the wrong clock",
                regime, found_regime,
            )
        if found_regime is None and regime == "short_burst":
            raise NvmlAnchorRegimeMismatch(
                f"artifact {artifact_path.name} lacks the anchor_regime "
                f"discriminator; a short_burst conversion requires an artifact "
                f"explicitly tagged anchor_regime=short_burst",
                regime, found_regime,
            )
        rop = art.get("reference_operating_point")
        if isinstance(rop, dict):
            ref = {
                "temp_ref_c": rop.get("temp_ref_c"),
                "power_ref_mw": rop.get("power_ref_mw"),
                "sm_clock_ref_mhz": rop.get("sm_clock_ref_mhz"),
                "mem_clock_ref_mhz": rop.get("mem_clock_ref_mhz"),
            }
            source = f"calibration_artifact:{artifact_path.name}"
    if (ref is None or any(_is_unset_ref_value(ref[k]) for k in ref)) and (
        regime == "sustained"
    ):
        # Fall back to the TOML [reference] block. The TOML carries only the
        # legacy sustained reference, so the fallback is sustained-regime only —
        # a missing short_burst artifact must NOT silently resolve to the
        # sustained TOML clock (it raises NvmlReferenceUnset below instead).
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        tref = data.get("reference", {})
        ref = {
            "temp_ref_c": tref.get("temp_ref_c"),
            "power_ref_mw": tref.get("power_ref_mw"),
            "sm_clock_ref_mhz": tref.get("sm_clock_ref_mhz"),
            "mem_clock_ref_mhz": tref.get("mem_clock_ref_mhz"),
        }
        source = f"acceptance_toml:{toml_path.name}"
    if ref is None:
        # No artifact for this regime and no admissible fallback.
        raise NvmlReferenceUnset(
            f"no {regime!r} NVML anchor artifact found at {artifact_path} and no "
            f"admissible fallback; run "
            f"tools/local_labels/calibrate_nvml_short_burst_anchor.py "
            f"(short_burst) or calibrate_nvml_reference_operating_point.py "
            f"(sustained) on the local RTX 3090 Ti"
        )
    unset = [k for k in ref if _is_unset_ref_value(ref[k])]
    if unset:
        raise NvmlReferenceUnset(
            f"reference fields {unset!r} are unset/retracted (nan/-1/None); run "
            f"tools/local_labels/calibrate_nvml_reference_operating_point.py on "
            f"the local RTX 3090 Ti to populate them"
        )
    sm = float(ref["sm_clock_ref_mhz"])
    if not (_MIN_PLAUSIBLE_SM_CLOCK_REF_MHZ <= sm <= _MAX_PLAUSIBLE_SM_CLOCK_REF_MHZ):
        raise NvmlClockImplausible(
            f"sm_clock_ref_mhz outside plausible band "
            f"[{_MIN_PLAUSIBLE_SM_CLOCK_REF_MHZ}, "
            f"{_MAX_PLAUSIBLE_SM_CLOCK_REF_MHZ}] MHz",
            sm,
        )
    return {
        "temp_ref_c": float(ref["temp_ref_c"]),
        "power_ref_mw": float(ref["power_ref_mw"]),
        "sm_clock_ref_mhz": sm,
        "mem_clock_ref_mhz": float(ref["mem_clock_ref_mhz"]),
        "source": source,
    }


def cycles_to_ns(
    cycles: float,
    *,
    sm_clock_mhz: float | None = None,
    reference: dict[str, float] | None = None,
    regime: str = "sustained",
) -> float:
    """Convert a ``gpc__cycles_elapsed.max`` cycle count to nanoseconds using the
    regime-appropriate calibrated SM-clock reference.

    ``ns = cycles / (sm_clock_mhz * 1e6) * 1e9 = cycles * 1000 / sm_clock_mhz``.

    ``regime`` selects the anchor when neither ``sm_clock_mhz`` nor ``reference``
    is supplied: a SHORT kernel should pass ``regime="short_burst"`` so its cycle
    count is converted at the cool full-boost ~1.87 GHz clock it actually runs
    at, NOT the sustained 90 C ~1350 MHz equilibrium (which would over-predict ns
    by ~38 %). A short-regime conversion that would resolve the sustained anchor
    is rejected with :class:`NvmlAnchorRegimeMismatch`.

    Fail-closed: when no ``sm_clock_mhz`` is passed, the calibrated reference is
    loaded via :func:`load_reference_operating_point` (which raises
    :class:`NvmlReferenceUnset` / :class:`NvmlClockImplausible` /
    :class:`NvmlAnchorRegimeMismatch`). An explicitly-passed ``sm_clock_mhz`` is
    itself plausibility-checked; an implausible value raises
    :class:`NvmlClockImplausible`. NEVER substitutes a default clock."""
    if cycles < 0:
        raise ValueError(f"cycles must be non-negative; got {cycles!r}")
    if sm_clock_mhz is None:
        if reference is None:
            reference = load_reference_operating_point(regime=regime)
        sm_clock_mhz = float(reference["sm_clock_ref_mhz"])
    sm_clock_mhz = float(sm_clock_mhz)
    if _is_unset_ref_value(sm_clock_mhz) or not (
        _MIN_PLAUSIBLE_SM_CLOCK_REF_MHZ <= sm_clock_mhz
        <= _MAX_PLAUSIBLE_SM_CLOCK_REF_MHZ
    ):
        raise NvmlClockImplausible(
            "sm_clock_mhz for cycles->ns conversion is unset/implausible",
            sm_clock_mhz,
        )
    return cycles * 1000.0 / sm_clock_mhz


def load_acceptance(path: Path = ACCEPTANCE_TOML) -> NvmlAcceptance:
    """Load `nvml_covariate_acceptance.toml` + the calibrated reference operating
    point. The acceptance THRESHOLD comes from the TOML; the REFERENCE values
    come from :func:`load_reference_operating_point` (calibration artifact first,
    TOML [reference] fallback), so a retracted/unset reference fails closed
    (raises :class:`NvmlReferenceUnset`) instead of feeding the regression a
    nan/-1 anchor."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    accept = data["acceptance"]
    ref = load_reference_operating_point(toml_path=path)
    return NvmlAcceptance(
        min_residual_variance_reduction_pct=float(accept["min_residual_variance_reduction_pct"]),
        temp_ref_c=ref["temp_ref_c"],
        power_ref_mw=ref["power_ref_mw"],
        sm_clock_ref_mhz=ref["sm_clock_ref_mhz"],
        mem_clock_ref_mhz=ref["mem_clock_ref_mhz"],
    )


# --- Natural-key validation -------------------------------------------------


_JOIN_KEY_FIELDS = (
    "slug",
    "shape_point",
    "repeat_index",
    "binary_sha256",
    "preflight_manifest_sha256",
)
# `kernel_signature_sha256` is intentionally absent from the in-process
# CUPTI/NVML natural-key check: the CUPTI sidecar's value is computed
# per-LAUNCH from the runtime kernel parameters, while the NVML sidecar
# is per-REPEAT (aggregating many launches) and the non-root measurement contract does
# not name a per-repeat NVML kernel signature. The cross-process
# producer-pair join (the dual_run_manifest schema, CUPTI run vs
# NVBit run) does carry per-launch kernel signatures and is a different
# surface entirely.


def _verify_natural_key(cupti: dict[str, Any], nvml: dict[str, Any]) -> None:
    """Raise NvmlJoinKeyMismatch on the first natural-key field that
    differs between the CUPTI and NVML sidecars."""
    for field in _JOIN_KEY_FIELDS:
        if cupti.get(field) != nvml.get(field):
            raise NvmlJoinKeyMismatch(field, cupti.get(field), nvml.get(field))


# --- Per-launch alignment ---------------------------------------------------


def _alignment_records(
    launches: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """For each CUPTI launch (start_ns, end_ns) window, count and aggregate
    the NVML samples falling within. Returns
        (per_launch_records, alignment_gap_windows)
    where alignment_gap_windows lists every launch with zero in-window samples.
    """
    per_launch: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for li, launch in enumerate(launches):
        start = int(launch["start_ns"])
        end = int(launch["end_ns"])
        in_window = [s for s in samples if start <= int(s["timestamp_ns"]) <= end]
        if not in_window:
            gaps.append(
                {"launch_index": int(launch.get("launch_index", li)), "start_ns": start, "end_ns": end}
            )
            per_launch.append(
                {
                    "launch_index": int(launch.get("launch_index", li)),
                    "n_samples_in_window": 0,
                    "covariate_aggregate": None,
                    "duration_ns": int(launch.get("duration_ns", end - start)),
                    "cuda_event_wall_time_ms": float(launch.get("cuda_event_wall_time_ms", 0.0)),
                    "dual_source_reconciliation_gap_ns": int(
                        launch.get("dual_source_reconciliation_gap_ns", 0)
                    ),
                }
            )
            continue
        n = len(in_window)
        covariate = {
            "temp_c": sum(float(s["temp_c"]) for s in in_window) / n,
            "power_mw": sum(float(s["power_mw"]) for s in in_window) / n,
            "sm_clock_mhz": sum(float(s["sm_clock_mhz"]) for s in in_window) / n,
            "mem_clock_mhz": sum(float(s["mem_clock_mhz"]) for s in in_window) / n,
        }
        per_launch.append(
            {
                "launch_index": int(launch.get("launch_index", li)),
                "n_samples_in_window": n,
                "covariate_aggregate": covariate,
                "duration_ns": int(launch.get("duration_ns", end - start)),
                "cuda_event_wall_time_ms": float(launch.get("cuda_event_wall_time_ms", 0.0)),
                "dual_source_reconciliation_gap_ns": int(
                    launch.get("dual_source_reconciliation_gap_ns", 0)
                ),
            }
        )
    return per_launch, gaps


# --- Covariate regression ---------------------------------------------------


def _solve_3x3(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Solve a 3x3 linear system via Gauss-Jordan elimination. Returns
    None when the system is singular or close-to-singular."""
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    n = 3
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot_row][col]) < 1e-12:
            return None
        a[col], a[pivot_row] = a[pivot_row], a[col]
        pivot = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= pivot
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[r][n] for r in range(n)]


def fit_covariate_regression(
    per_launch: Sequence[dict[str, Any]],
    acceptance: NvmlAcceptance,
) -> dict[str, Any]:
    """Fit the non-root measurement covariate regression on the per-launch covariate
    aggregates, return the percent reduction in |gap_ns| residual
    variance, and emit the typed `nvml_covariate_regression_ineffective`
    diagnostic when the reduction falls below the registered threshold.

    The regression solves the 3x3 normal equations for
        gap_ns = beta_temp  * (temp_c     - temp_ref_c)
               + beta_power * (power_mw   - power_ref_mw)
               + beta_clock * (sm_clock_mhz - sm_clock_ref_mhz)
               + epsilon
    by least squares over launches that carry covariate aggregates.
    """
    rows = [r for r in per_launch if r["covariate_aggregate"] is not None]
    if len(rows) < 4:
        return {
            "diagnostic": "nvml_regression_undersampled",
            "n_launches_with_covariates": len(rows),
            "min_launches_required": 4,
            "regression_admitted": False,
        }
    xs = [
        (
            float(r["covariate_aggregate"]["temp_c"]) - acceptance.temp_ref_c,
            float(r["covariate_aggregate"]["power_mw"]) - acceptance.power_ref_mw,
            float(r["covariate_aggregate"]["sm_clock_mhz"]) - acceptance.sm_clock_ref_mhz,
        )
        for r in rows
    ]
    ys = [float(r["dual_source_reconciliation_gap_ns"]) for r in rows]

    matrix = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for x, y in zip(xs, ys):
        for i in range(3):
            for j in range(3):
                matrix[i][j] += x[i] * x[j]
            rhs[i] += x[i] * y
    coeffs = _solve_3x3(matrix, rhs)
    if coeffs is None:
        return {
            "diagnostic": "nvml_regression_singular",
            "n_launches_with_covariates": len(rows),
            "regression_admitted": False,
        }
    beta_temp, beta_power, beta_clock = coeffs

    mean_y = sum(ys) / len(ys)
    var_uncorrected = sum((y - mean_y) ** 2 for y in ys) / len(ys)
    residuals = [
        y
        - (beta_temp * x[0] + beta_power * x[1] + beta_clock * x[2])
        for x, y in zip(xs, ys)
    ]
    mean_r = sum(residuals) / len(residuals)
    var_corrected = sum((r - mean_r) ** 2 for r in residuals) / len(residuals)

    if var_uncorrected == 0:
        reduction_pct = 0.0
    else:
        reduction_pct = max(0.0, (var_uncorrected - var_corrected) / var_uncorrected * 100.0)

    threshold = acceptance.min_residual_variance_reduction_pct
    admitted = reduction_pct >= threshold
    out: dict[str, Any] = {
        "regression_admitted": admitted,
        "regression_residual_variance_reduction_pct": reduction_pct,
        "regression_threshold_pct": threshold,
        "n_launches_with_covariates": len(rows),
        "beta_temp": beta_temp,
        "beta_power": beta_power,
        "beta_clock": beta_clock,
        "var_uncorrected_ns2": var_uncorrected,
        "var_corrected_ns2": var_corrected,
    }
    if not admitted:
        out["diagnostic"] = "nvml_covariate_regression_ineffective"
    return out


# --- Top-level join driver --------------------------------------------------


def join_cupti_nvml(
    cupti_payload: dict[str, Any],
    nvml_payload: dict[str, Any],
    *,
    acceptance: NvmlAcceptance | None = None,
) -> dict[str, Any]:
    """Join a CUPTI Activity sidecar's per-launch records with an NVML
    covariate sidecar's samples, run the covariate regression, and
    return a typed-JSON record describing the result.

    Inputs are dicts (already parsed from JSON); use `join_cupti_nvml_paths`
    to read paths from disk and validate them through the schema-check
    surface first.
    """
    if acceptance is None:
        acceptance = load_acceptance()
    _verify_natural_key(cupti_payload, nvml_payload)

    launches = cupti_payload.get("per_launch_records") or [
        {
            "launch_index": cupti_payload["launch_index"],
            "start_ns": cupti_payload["start_ns"],
            "end_ns": cupti_payload["end_ns"],
            "duration_ns": cupti_payload.get("duration_ns", cupti_payload["end_ns"] - cupti_payload["start_ns"]),
            "cuda_event_wall_time_ms": cupti_payload.get("cuda_event_wall_time_ms", 0.0),
            "dual_source_reconciliation_gap_ns": cupti_payload.get("dual_source_reconciliation_gap_ns", 0),
        }
    ]
    samples = nvml_payload.get("samples", [])

    per_launch, gaps = _alignment_records(launches, samples)
    out: dict[str, Any] = {
        "diagnostic": "nvml_join_ok",
        "slug": cupti_payload["slug"],
        "shape_point": cupti_payload["shape_point"],
        "repeat_index": cupti_payload["repeat_index"],
        "n_launches": len(launches),
        "n_samples": len(samples),
        "per_launch": per_launch,
    }
    if gaps:
        out["diagnostic"] = "nvml_alignment_gap"
        out["alignment_gap_windows_ns"] = gaps
    regression = fit_covariate_regression(per_launch, acceptance)
    out["regression"] = regression
    if regression.get("diagnostic") == "nvml_covariate_regression_ineffective":
        out.setdefault("notes", []).append("nvml_covariate_regression_ineffective")
    return out


def join_cupti_nvml_paths(
    cupti_path: Path,
    nvml_path: Path,
    *,
    acceptance: NvmlAcceptance | None = None,
) -> dict[str, Any]:
    """File-system wrapper for `join_cupti_nvml`. Reads, parses, and
    delegates."""
    cupti_payload = json.loads(cupti_path.read_text())
    nvml_payload = json.loads(nvml_path.read_text())
    return join_cupti_nvml(cupti_payload, nvml_payload, acceptance=acceptance)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2:
        sys.stderr.write(
            "usage: nvml_covariate_join CUPTI_SIDECAR_PATH NVML_SIDECAR_PATH\n"
        )
        return 2
    cupti_path = Path(argv[0])
    nvml_path = Path(argv[1])
    if not cupti_path.is_file() or not nvml_path.is_file():
        sys.stderr.write(f"usage_error: input path missing — {cupti_path}, {nvml_path}\n")
        return 2
    try:
        result = join_cupti_nvml_paths(cupti_path, nvml_path)
    except NvmlJoinKeyMismatch as e:
        sys.stderr.write(json.dumps(e.to_json(), sort_keys=True, indent=2) + "\n")
        return 1
    sys.stdout.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
