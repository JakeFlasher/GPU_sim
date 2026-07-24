"""`cache_prediction` surface — emission helper.

Registered manifest surface that emits a per-baseline
`cache_prediction` entry. The entry carries the
standard 5-status surface enum plus the surface-
specific provenance fields (`prediction_mode`,
`calibration_source`, `confidence_band`,
`b200_validated`) per the CLAUDE.md cache-prediction
acceptance contract.

This module is the THIN emission helper. The surface
entry is built from a validated `PredictorBundle`
(via `validate_predictor_bundle`); the helper routes
the input through the schema validator at the
emission boundary so an invalid bundle cannot be
emitted as a surface.

Bundle factory wiring (per-launch sidecar artifact
emission, predictor-body replacement) is out of scope
for this module — the helper takes a pre-built
`PredictorBundle` and produces the manifest entry.

Imports only stdlib + `framework_contract` +
`predictor_bundle_schema`. NO calibration-subtree
imports; the NCU quarantine boundary is preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from sol_baselines_harness.metrics.framework_contract import (
    CellJoinKey,
)
from sol_baselines_harness.metrics.pooled_cross_arch_head import (
    PooledHeadFitResult,
)
from sol_baselines_harness.metrics.predictor_bundle_schema import (
    ManifestPairingIndex,
    PredictorBundle,
    PredictorBundleSchemaError,
    validate_predictor_bundle,
)
from sol_baselines_harness.surfaces.base import (
    HostInfo,
    Surface,
    SurfaceParseResult,
    SurfaceProbeResult,
    SurfaceRunResult,
    SURFACE_STATUS_OK,
    SURFACE_STATUS_UNAVAILABLE,
    SURFACE_STATUSES,
    VALID_SURFACES,
)


CACHE_PREDICTION_SURFACE_NAME: str = "cache_prediction"


# Closed enum of prediction-mode strings the surface
# entry's `prediction_mode` field may carry:
#   * `in_distribution` — sm_86 or sm_120 host where the
#     predictor's calibration corpus was collected.
#   * `zero_shot` — B200 host where the predictor is
#     deployed without an arch-specific calibration
#     sweep.
CACHE_PREDICTION_PREDICTION_MODE_IN_DISTRIBUTION: str = "in_distribution"
CACHE_PREDICTION_PREDICTION_MODE_ZERO_SHOT: str = "zero_shot"
CACHE_PREDICTION_PREDICTION_MODES: frozenset[str] = frozenset({
    CACHE_PREDICTION_PREDICTION_MODE_IN_DISTRIBUTION,
    CACHE_PREDICTION_PREDICTION_MODE_ZERO_SHOT,
})


CACHE_PREDICTION_SURFACE_TYPED_REASONS: frozenset[str] = frozenset({
    "cache_prediction_surface_bundle_invalid",
    "cache_prediction_surface_prediction_mode_invalid",
    "cache_prediction_surface_provenance_field_invalid",
    "cache_prediction_surface_entry_invalid",
})


# Canonical stringification template for the
# (lower, upper) float confidence-band tuple carried
# on the training side
# (`PooledHeadFitResult.confidence_band`). The training
# side carries the numeric LOOCV residual pair; the
# surface side carries a non-empty descriptor string
# (`CachePredictionProvenance.confidence_band`). The
# emission path (eventual B200 deployment hook) MUST
# stringify the tuple via this exact template so the
# emitted descriptor parses round-trip and any future
# downstream consumer of the manifest entry can recover
# the numeric band by regex / .format() reversal.
#
# Format reads as:
#   `loocv_residual_band_<lower:.4f>_to_<upper:.4f>`
#
# Four decimal places are sufficient for the deployed
# accuracy gate (±5 pp friendly cohort, ±15 pp hostile)
# while remaining human-readable in manifest JSON
# inspection.
CACHE_PREDICTION_CONFIDENCE_BAND_FORMAT: str = (
    "loocv_residual_band_{lower:.4f}_to_{upper:.4f}"
)


# Canonical regex source string for the inverse direction of
# the confidence-band stringification. The pattern matches
# EXACTLY the set of strings the canonical formatter
# ``{:.4f}`` can produce for any ``(lower, upper)`` pair
# already validated as ``0.0 <= lower <= upper <= 1.0``
# rounded to four decimal places. That set is:
#
#   * ``0\.[0-9]{4}`` — any value strictly less than 1.0
#     (the integer part is always the single digit ``0``).
#   * ``1\.0000`` — the closed-interval upper endpoint
#     (the only string with integer part ``1`` the formatter
#     can produce inside the validated domain).
#
# An earlier wider pattern (``[0-9]+\.[0-9]{4}``) accepted
# non-canonical leading-zero spellings the formatter can never
# produce, e.g., ``00.0500`` / ``01.0000`` /
# ``000.0000_to_001.0000``. Those spellings parse back to the
# same numeric tuple as the canonical form, but
# ``format.format(*parse(s)) != s`` byte-for-byte because the
# formatter canonicalises to ``0.0500`` / ``1.0000`` /
# ``0.0000_to_1.0000``. The narrower pattern below closes that
# round-trip identity gap.
#
# Domain validation (``[0.0, 1.0]``-clipping + ordering)
# still lives in the parser function body as defense in depth.
# Under the narrower pattern the domain check is unreachable
# on the happy path (the regex's ``0\.[0-9]{4}|1\.0000``
# alternation already enforces ``[0.0, 1.0]``); a future regex
# loosening would re-engage it.
#
# The regex is consumed via ``fullmatch`` (not ``match``), so
# no explicit ``^`` / ``$`` anchors are needed; leading /
# trailing whitespace is rejected automatically.
CACHE_PREDICTION_CONFIDENCE_BAND_REGEX_PATTERN: str = (
    r"loocv_residual_band_"
    r"(?P<lower>0\.[0-9]{4}|1\.0000)_to_"
    r"(?P<upper>0\.[0-9]{4}|1\.0000)"
)
CACHE_PREDICTION_CONFIDENCE_BAND_REGEX: re.Pattern[str] = re.compile(
    CACHE_PREDICTION_CONFIDENCE_BAND_REGEX_PATTERN
)


class CachePredictionSurfaceError(Exception):
    """Raised on `cache_prediction` surface emission /
    validation failures.

    Carries the typed-reason string so downstream
    callers can compose typed-unavailable bundles or
    report-table errors without parsing message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class CachePredictionProvenance:
    """Provenance block for the cache-prediction surface
    entry.

    Fields:
      * `prediction_mode` ∈
        `CACHE_PREDICTION_PREDICTION_MODES`.
      * `calibration_source` — non-empty string
        describing the calibration corpus(es) the
        bundle's calibration head was fit on (e.g.,
        `"sm_86"`, `"sm_86+sm_120"`).
      * `confidence_band` — non-empty string carrying
        the LOOCV residual descriptor (e.g.,
        `"loocv_p95_3.2pp"`).
      * `b200_validated` — bool. `False` until the B200
        deployment hook lands; flips to `True` only
        after the deployed-side hook has run and
        emitted a confirmed B200 artifact.
    """

    prediction_mode: str
    calibration_source: str
    confidence_band: str
    b200_validated: bool


@dataclass(frozen=True)
class CachePredictionSurfaceEntry:
    """Per-baseline `cache_prediction` surface entry.

    Carries the standard 5-status surface enum + the
    surface-specific provenance block + the
    bundle's `manifest_pairing_index` so a downstream
    consumer can join the surface entry to the
    corresponding bundle without re-reading the bundle.
    """

    status: str
    provenance: CachePredictionProvenance
    manifest_pairing_index: ManifestPairingIndex
    tool_version: Optional[str] = None
    command: Optional[str] = None
    exit_code: Optional[int] = None
    artifact_path: Optional[str] = None
    waiver_reason: Optional[str] = None


def validate_cache_prediction_provenance(
    provenance: CachePredictionProvenance,
) -> None:
    """Validate the four provenance fields.

    Runs per-field checks first, then the cross-field
    zero-shot invariant. The cross-field invariant
    fires only when every per-field check has already
    accepted the inputs.

    Raises typed reasons:
      * `cache_prediction_surface_prediction_mode_invalid`
        when `prediction_mode` is not in the closed
        enum.
      * `cache_prediction_surface_provenance_field_invalid`
        for empty `calibration_source` /
        `confidence_band`, non-bool `b200_validated`, or
        for `prediction_mode == "zero_shot"` combined
        with a provenance shape that contradicts the
        deployed B200 zero-shot contract (cross-arch
        pooled source, populated confidence band,
        pre-online-judge B200 flag). The cross-field
        violation reuses the existing
        `cache_prediction_surface_provenance_field_invalid`
        typed reason to minimise vocabulary churn; the
        message string names the offending field so
        downstream callers can disambiguate without a
        new vocabulary entry.
    """
    if not isinstance(provenance, CachePredictionProvenance):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_provenance_field_invalid",
            f"provenance must be a "
            f"CachePredictionProvenance; got "
            f"{type(provenance).__name__}",
        )
    if provenance.prediction_mode not in CACHE_PREDICTION_PREDICTION_MODES:
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_prediction_mode_invalid",
            f"prediction_mode {provenance.prediction_mode!r} "
            f"is not in the closed enum "
            f"{sorted(CACHE_PREDICTION_PREDICTION_MODES)}",
        )
    for name in ("calibration_source", "confidence_band"):
        value = getattr(provenance, name)
        if not isinstance(value, str) or not value.strip():
            raise CachePredictionSurfaceError(
                "cache_prediction_surface_provenance_field_invalid",
                f"provenance.{name} must be a non-empty "
                f"string; got {value!r}",
            )
    # `b200_validated` must be a real bool — not a
    # truthy int — so the JSON-encoded manifest entry
    # never carries `0` / `1` where a `false` / `true`
    # is contracted.
    if not isinstance(provenance.b200_validated, bool):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_provenance_field_invalid",
            f"provenance.b200_validated must be a bool; "
            f"got {type(provenance.b200_validated).__name__}",
        )
    # Cross-field zero-shot invariant. The deployed B200
    # zero-shot cache-prediction surface emission must
    # carry: calibration_source="sm_86+sm_120" (cross-arch
    # pooled source), a populated non-empty confidence_band,
    # and b200_validated=False (the flag flips to True only
    # after the deployed-side hook has run). Per-field
    # checks above already enforce the non-empty
    # confidence_band string for any mode, so the cross-
    # field invariant only re-checks calibration_source and
    # b200_validated. This invariant mirrors the
    # `PooledHeadFitResult._validate_zero_shot_cross_field_invariant`
    # contract in
    # `baselines_harness.metrics.pooled_cross_arch_head`;
    # the prediction-mode vocabulary equality drift-guard
    # test in `test_pooled_cross_arch_head` pins the two
    # surfaces' prediction-mode frozensets to be equal.
    if provenance.prediction_mode == "zero_shot":
        if provenance.calibration_source != "sm_86+sm_120":
            raise CachePredictionSurfaceError(
                "cache_prediction_surface_provenance_field_invalid",
                (
                    f"prediction_mode='zero_shot' requires "
                    f"calibration_source='sm_86+sm_120' (the "
                    f"cross-arch pooled source); got "
                    f"calibration_source={provenance.calibration_source!r}"
                ),
            )
        if provenance.b200_validated is not False:
            raise CachePredictionSurfaceError(
                "cache_prediction_surface_provenance_field_invalid",
                (
                    f"prediction_mode='zero_shot' requires "
                    f"b200_validated=False until the deployed-side "
                    f"hook has run and emitted a confirmed B200 "
                    f"artifact; got "
                    f"b200_validated={provenance.b200_validated!r}"
                ),
            )


def _canonicalize_zero_bound(value: float) -> float:
    """Coerce IEEE-754 signed zero (``-0.0``) into the canonical
    unsigned zero (``+0.0``) for confidence-band stringification.

    The training-side validator
    (``_validate_confidence_band`` in
    ``baselines_harness.metrics.pooled_cross_arch_head``)
    accepts ``-0.0`` because ``-0.0 < 0.0`` evaluates to
    ``False`` per IEEE-754 signed-zero semantics: ``-0.0``
    compares equal to ``+0.0`` for ordering. The Python
    ``{:.4f}`` formatter, however, preserves the sign bit and
    emits ``"-0.0000"`` for ``-0.0`` — a string the canonical
    confidence-band regex (:data:`CACHE_PREDICTION_CONFIDENCE_BAND_REGEX`)
    rejects because the per-bound alternation
    ``0\\.[0-9]{4}|1\\.0000`` admits no minus sign.

    A residual band ``(-0.0, X)`` therefore round-trips
    asymmetrically: the training-side fit result is valid, the
    converter emits an unparseable descriptor, and the
    downstream consumer fails closed at parse time. This helper
    closes that asymmetry at the emission boundary: signed zero
    is canonicalized to plain zero BEFORE the formatter sees
    it, so the emitted descriptor always lies inside the
    canonical regex's accepted language.

    The check ``value == 0.0`` is the simplest formulation —
    ``-0.0 == 0.0`` is ``True`` and ``+0.0 == 0.0`` is also
    ``True``, so both flavours of zero map to the plain-zero
    canonical form. Any non-zero ``value`` is returned
    unchanged.

    This helper is intentionally private; downstream callers
    must not need to know about signed-zero canonicalization —
    they consume only canonical descriptor strings.
    """
    if value == 0.0:
        return 0.0
    return value


def cache_prediction_provenance_from_pooled_head_fit_result(
    fit_result: PooledHeadFitResult,
) -> CachePredictionProvenance:
    """Convert a training-side :class:`PooledHeadFitResult` into a
    deployed surface-side :class:`CachePredictionProvenance`.

    The two dataclasses share three field names exactly
    (``prediction_mode``, ``calibration_source``, ``b200_validated``)
    but disagree on ``confidence_band``:

    * Training side: ``tuple[float, float] | None`` carrying the
      numeric LOOCV residual pair (``(lower, upper)`` in
      ``[0.0, 1.0]`` with ``lower <= upper``) or ``None`` for the
      "not yet computed" sentinel.
    * Surface side: non-empty descriptor string used as the
      ``cache_prediction`` manifest entry's confidence_band field.

    This converter is the canonical bridge between the two
    surfaces. The eventual B200 deployment-hook emission path
    MUST flow through this function so any future downstream
    consumer can reverse-parse the band via
    :data:`CACHE_PREDICTION_CONFIDENCE_BAND_FORMAT` without
    discovering an ad-hoc stringification on a per-emission-site
    basis.

    Failure modes (typed reasons):

    * ``cache_prediction_surface_provenance_field_invalid`` — the
      input is not a :class:`PooledHeadFitResult` instance, or the
      input's ``confidence_band`` is ``None`` (the training-side
      "not yet computed" sentinel is incompatible with deployed
      emission which requires a populated descriptor string).

    The converter applies defense in depth: after constructing the
    :class:`CachePredictionProvenance`, it re-runs
    :func:`validate_cache_prediction_provenance` on the result. If
    the source ``fit_result`` was a fully-validated
    :class:`PooledHeadFitResult` (constructor-time cross-field
    invariants held) then the produced provenance MUST also pass
    surface-side validation, including the surface-side cross-
    field zero-shot invariant. The re-validation is a regression
    guard against future contract-shape drift.

    Note: the converter intentionally does NOT re-run the
    training-side ``__post_init__`` validators on the input. The
    input is a frozen dataclass; if it was constructed without
    raising, its per-field + cross-field invariants already hold.
    The type-check at function entry is the only input gate
    needed.
    """
    if not isinstance(fit_result, PooledHeadFitResult):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_provenance_field_invalid",
            (
                f"cache_prediction_provenance_from_pooled_head_fit_result "
                f"requires a PooledHeadFitResult; got "
                f"{type(fit_result).__name__}"
            ),
        )
    if fit_result.confidence_band is None:
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_provenance_field_invalid",
            (
                "fit_result.confidence_band is None (training-side "
                "'not yet computed' sentinel); deployed surface "
                "emission requires a populated confidence_band "
                "descriptor string"
            ),
        )
    lower, upper = fit_result.confidence_band
    # Canonicalize signed zero before stringification: IEEE-754
    # `-0.0` passes the training-side validator (because
    # `-0.0 < 0.0` is False) but the `{:.4f}` formatter
    # preserves the sign bit and would emit `-0.0000`, which the
    # canonical confidence-band regex rejects. The
    # `_canonicalize_zero_bound` helper maps both flavours of
    # zero to plain `+0.0` so the descriptor always lies inside
    # the canonical regex's accepted language.
    canonical_lower = _canonicalize_zero_bound(float(lower))
    canonical_upper = _canonicalize_zero_bound(float(upper))
    band_descriptor = CACHE_PREDICTION_CONFIDENCE_BAND_FORMAT.format(
        lower=canonical_lower,
        upper=canonical_upper,
    )
    provenance = CachePredictionProvenance(
        prediction_mode=fit_result.prediction_mode,
        calibration_source=fit_result.calibration_source,
        confidence_band=band_descriptor,
        b200_validated=fit_result.b200_validated,
    )
    # Defense in depth: re-run surface-side per-field +
    # cross-field validation on the produced provenance.
    # If the source fit_result was valid, this is a no-op;
    # if a future refactor breaks the contract-shape
    # alignment between PooledHeadFitResult and
    # CachePredictionProvenance, this re-validation
    # surfaces it at converter call time.
    validate_cache_prediction_provenance(provenance)
    return provenance


def parse_cache_prediction_confidence_band(
    band_descriptor: str,
) -> tuple[float, float]:
    """Reverse-parse a canonical ``confidence_band`` descriptor
    string back into the numeric ``(lower, upper)`` LOOCV residual
    pair.

    This is the structural inverse of the forward stringification
    performed by
    :func:`cache_prediction_provenance_from_pooled_head_fit_result`
    via :data:`CACHE_PREDICTION_CONFIDENCE_BAND_FORMAT`. The
    round-trip identity holds:

    * ``parse(format.format(lower=L, upper=U)) == (L, U)`` for any
      ``(L, U)`` already rounded to four decimal places in
      ``[0.0, 1.0]`` with ``L <= U``.
    * ``format.format(lower=parse(s)[0], upper=parse(s)[1]) == s``
      for any canonical descriptor string ``s``.

    Downstream consumers (manifest comparators, sidecar audit
    tools, comparison artifact regenerators) MUST flow through
    this function so the reverse direction has exactly one
    sanctioned implementation — ad-hoc regex / float-parser logic
    is forbidden by convention so two consumers cannot disagree
    on precision strictness, whitespace tolerance, domain
    validation, or inverted-bounds rejection.

    Failure modes (all raise
    :class:`CachePredictionSurfaceError` with typed reason
    ``cache_prediction_surface_provenance_field_invalid``):

    * Input is not ``str``.
    * Input does not ``fullmatch`` the canonical regex
      :data:`CACHE_PREDICTION_CONFIDENCE_BAND_REGEX` — per-bound
      alternation ``0\\.[0-9]{4}|1\\.0000``, exactly the
      canonical formatter's output language for ``[0.0, 1.0]``
      bounds. The regex rejects wrong prefix, wrong precision,
      missing bound, leading / trailing whitespace, leading-
      zero spellings (``00.0500`` / ``01.0000`` /
      ``000.0000_to_001.0000``), strings with a leading minus
      (``-0.0000``), and out-of-domain spellings like
      ``"loocv_residual_band_10.5000_to_20.0000"`` (the
      ``10.5000`` upper bound matches neither alternative).
    * Either bound lies outside ``[0.0, 1.0]``. Under the
      narrow regex this case is unreachable on the happy path
      (the regex already enforces ``[0.0, 1.0]``), but the
      domain check is preserved as defense in depth so a future
      regex loosening cannot silently admit out-of-domain
      bounds.
    * ``lower > upper``.
    """
    if not isinstance(band_descriptor, str):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_provenance_field_invalid",
            (
                f"parse_cache_prediction_confidence_band requires "
                f"a str; got {type(band_descriptor).__name__}"
            ),
        )
    match = CACHE_PREDICTION_CONFIDENCE_BAND_REGEX.fullmatch(
        band_descriptor,
    )
    if match is None:
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_provenance_field_invalid",
            (
                f"band_descriptor={band_descriptor!r} does not "
                f"match canonical format "
                f"{CACHE_PREDICTION_CONFIDENCE_BAND_FORMAT!r}"
            ),
        )
    lower = float(match.group("lower"))
    upper = float(match.group("upper"))
    for name, value in (("lower", lower), ("upper", upper)):
        if value < 0.0 or value > 1.0:
            raise CachePredictionSurfaceError(
                "cache_prediction_surface_provenance_field_invalid",
                (
                    f"parsed {name}={value!r} lies outside "
                    f"[0.0, 1.0]; canonical band-descriptor "
                    f"bounds must fall within the LOOCV "
                    f"residual domain"
                ),
            )
    if lower > upper:
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_provenance_field_invalid",
            (
                f"parsed lower={lower!r} > upper={upper!r}; "
                f"canonical band-descriptor lower bound must not "
                f"exceed upper bound"
            ),
        )
    return (lower, upper)


# Canonical sidecar artifact relative-path that an ok
# `CachePredictionSurfaceEntry` must carry. Kept here
# (the surface module) so the public final-entry
# validator owns its acceptance shape; the sidecar
# module imports this constant rather than
# hand-rolling the same string.
CACHE_PREDICTION_SURFACE_ARTIFACT_RELPATH: str = (
    "cache_prediction/predictions.jsonl"
)


# Canonical `command` shape on an ok entry: the
# `sidecar_sha256:` prefix + a 64-character lowercase
# hex digest of the on-disk sidecar bytes. The
# attacher threads this onto the entry; the public
# final-entry validator enforces it at serialization
# time.
CACHE_PREDICTION_SURFACE_COMMAND_RE: re.Pattern[str] = re.compile(
    r"^sidecar_sha256:[0-9a-f]{64}$"
)


def _validate_cache_prediction_surface_entry_pre_attach(
    entry: CachePredictionSurfaceEntry,
) -> None:
    """Validate the four base fields of an unattached
    `CachePredictionSurfaceEntry`.

    Runs the dataclass-type check + status enum check +
    provenance validator + pairing-index type check. The
    helper exists for INTERNAL pre-attach use by the
    sidecar attacher and the emit helper; an unattached
    entry has no `artifact_path` / `command` yet, so
    the public final-entry validator's sidecar-shape
    check does not apply at this point.

    Downstream callers that need to validate a serialized
    entry MUST use `validate_cache_prediction_surface_entry`
    (the final-entry validator) instead — it routes
    through this base validator + enforces the on-disk
    sidecar shape when `status == "ok"`.
    """
    if not isinstance(entry, CachePredictionSurfaceEntry):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_entry_invalid",
            f"entry must be a CachePredictionSurfaceEntry; "
            f"got {type(entry).__name__}",
        )
    if entry.status not in SURFACE_STATUSES:
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_entry_invalid",
            f"status {entry.status!r} is not in the closed "
            f"surface-status enum {sorted(SURFACE_STATUSES)}",
        )
    validate_cache_prediction_provenance(entry.provenance)
    if not isinstance(entry.manifest_pairing_index, ManifestPairingIndex):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_entry_invalid",
            f"manifest_pairing_index must be a "
            f"ManifestPairingIndex; got "
            f"{type(entry.manifest_pairing_index).__name__}",
        )


def validate_cache_prediction_surface_entry(
    entry: CachePredictionSurfaceEntry,
) -> None:
    """Validate a final-shape `CachePredictionSurfaceEntry`
    before serialization.

    This is the public serialization-acceptance boundary.
    Downstream manifest writers MUST call this before
    serializing an entry — it routes through
    `_validate_cache_prediction_surface_entry_pre_attach`
    for the base-field checks AND, when `status` is
    `"ok"`, requires the entry carry the canonical
    sidecar identity:

      * `artifact_path == "cache_prediction/predictions.jsonl"`
      * `command` matches
        `^sidecar_sha256:[0-9a-f]{64}$`.

    Any other shape on an ok entry — `artifact_path =
    None`, a different relative path, `command = None`,
    a malformed digest — raises
    `cache_prediction_surface_entry_invalid`. This
    closes the fail-open path where a downstream caller
    could construct an ok entry directly (bypassing the
    emit helper) and claim a manifest `status: ok`
    without an on-disk sidecar.

    For non-ok statuses (unavailable / blocked / failed /
    skipped) the sidecar shape is not enforced; those
    surface statuses are explicitly the "no payload"
    states.
    """
    _validate_cache_prediction_surface_entry_pre_attach(entry)
    if entry.status != SURFACE_STATUS_OK:
        return
    if entry.artifact_path != CACHE_PREDICTION_SURFACE_ARTIFACT_RELPATH:
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_entry_invalid",
            f"ok entry artifact_path must equal "
            f"{CACHE_PREDICTION_SURFACE_ARTIFACT_RELPATH!r}; got "
            f"{entry.artifact_path!r}",
        )
    if not isinstance(entry.command, str):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_entry_invalid",
            f"ok entry command must be a string matching "
            f"{CACHE_PREDICTION_SURFACE_COMMAND_RE.pattern!r}; "
            f"got {type(entry.command).__name__}",
        )
    if not CACHE_PREDICTION_SURFACE_COMMAND_RE.match(entry.command):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_entry_invalid",
            f"ok entry command does not match the "
            f"canonical sidecar-digest pattern "
            f"{CACHE_PREDICTION_SURFACE_COMMAND_RE.pattern!r}; "
            f"got {entry.command!r}",
        )


def emit_cache_prediction_surface_entry(
    bundles: Sequence[PredictorBundle],
    bundle_dir: Path,
    *,
    prediction_mode: str,
    calibration_source: str,
    confidence_band: str,
    b200_validated: bool,
) -> CachePredictionSurfaceEntry:
    """Emit an ok `cache_prediction` surface entry by
    writing the per-launch sidecar artifact and
    attaching its path + SHA-256 digest to the entry.

    The acceptance boundary is fail-closed end to end:
      * `bundles` is a non-empty sequence of
        `PredictorBundle` objects. A non-sequence
        argument, an empty sequence, or any
        non-bundle element raises
        `cache_prediction_surface_bundle_invalid`.
      * Every bundle is routed through
        `validate_predictor_bundle` at the writer
        boundary (via
        `write_cache_prediction_sidecar`); the
        attacher then re-reads the sidecar and
        re-validates each bundle at the reader
        boundary, so the surface entry cannot
        promote to `status="ok"` against an
        invalid bundle, a tampered on-disk
        artifact, or a sidecar belonging to a
        different baseline.
      * Every bundle's `manifest_pairing_index`
        must match the primary bundle's; the
        attacher cross-checks this against the
        surface entry's pairing index.
      * The provenance block is validated per
        field (off-enum mode + empty
        calibration_source / confidence_band +
        non-bool b200_validated all rejected).

    On success the returned entry carries:
      * `status = "ok"`.
      * `artifact_path =
        "cache_prediction/predictions.jsonl"`.
      * `command = f"sidecar_sha256:<64-hex>"`
        for the SHA-256 digest of the on-disk
        sidecar bytes.
      * The provided provenance block.
      * The primary bundle's `manifest_pairing_index`.

    Sidecar-layer typed reasons raised internally
    (`cache_prediction_sidecar_*`) are translated
    into surface-layer typed reasons
    (`cache_prediction_surface_bundle_invalid` for
    bundle / write / read failures;
    `cache_prediction_surface_entry_invalid` for
    pairing-index mismatch / missing artifact) so
    the public boundary's typed-reason vocabulary
    stays internal to the surface module.

    Sidecar imports are deferred to inside this
    function so the surface module's static import
    set stays unchanged at module load time — the
    NCU quarantine static-guard inspects the
    top-level imports, not function-scope ones.
    """
    if not isinstance(bundle_dir, Path):
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_entry_invalid",
            f"bundle_dir must be a pathlib.Path; got "
            f"{type(bundle_dir).__name__}",
        )
    if not isinstance(bundles, (list, tuple)) or len(bundles) == 0:
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_bundle_invalid",
            "bundles must be a non-empty sequence of "
            "PredictorBundle; an empty sequence cannot "
            "produce an ok surface entry.",
        )
    for i, bundle in enumerate(bundles):
        if not isinstance(bundle, PredictorBundle):
            raise CachePredictionSurfaceError(
                "cache_prediction_surface_bundle_invalid",
                f"bundles[{i}] must be a PredictorBundle; "
                f"got {type(bundle).__name__}",
            )
    provenance = CachePredictionProvenance(
        prediction_mode=prediction_mode,
        calibration_source=calibration_source,
        confidence_band=confidence_band,
        b200_validated=b200_validated,
    )
    validate_cache_prediction_provenance(provenance)
    # Deferred import: keeps the surface module's
    # static import set unchanged. The sidecar
    # module imports back from `cache_prediction`
    # for the entry dataclass, so a top-level
    # import here would create a circular import
    # at module load time.
    from sol_baselines_harness.surfaces.cache_prediction_sidecar import (
        CachePredictionSidecarError,
        attach_sidecar_to_surface_entry,
        write_cache_prediction_sidecar,
    )
    try:
        write_cache_prediction_sidecar(bundle_dir, bundles)
    except CachePredictionSidecarError as exc:
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_bundle_invalid",
            f"sidecar write failed ({exc.typed_reason}: "
            f"{exc.message})",
        ) from exc
    primary = bundles[0]
    entry = CachePredictionSurfaceEntry(
        status=SURFACE_STATUS_OK,
        provenance=provenance,
        manifest_pairing_index=primary.manifest_pairing_index,
    )
    # Use the pre-attach base validator here: the entry
    # has no artifact_path / command yet — those are
    # what the attacher adds.
    _validate_cache_prediction_surface_entry_pre_attach(entry)
    try:
        attached = attach_sidecar_to_surface_entry(entry, bundle_dir)
    except CachePredictionSidecarError as exc:
        # The attacher re-routes every bundle through
        # `validate_predictor_bundle` and cross-checks
        # the pairing index; a failure here means an
        # invariant beyond the writer's contract is
        # broken (pairing-index mismatch, on-disk
        # tampering between write and attach, etc.).
        if exc.typed_reason == "cache_prediction_sidecar_entry_invalid":
            raise CachePredictionSurfaceError(
                "cache_prediction_surface_entry_invalid",
                f"sidecar attach failed "
                f"({exc.typed_reason}: {exc.message})",
            ) from exc
        raise CachePredictionSurfaceError(
            "cache_prediction_surface_bundle_invalid",
            f"sidecar attach failed "
            f"({exc.typed_reason}: {exc.message})",
        ) from exc
    # The attached entry is now final-shape — route it
    # through the public final-entry validator so a
    # broken `attach_sidecar_to_surface_entry`
    # implementation cannot silently produce an
    # off-canonical artifact_path / command.
    validate_cache_prediction_surface_entry(attached)
    return attached


# Sanity check at import time: the surface name MUST
# be in `VALID_SURFACES`. A future drift on either
# side fails loudly at import rather than at
# runtime-construction.
assert CACHE_PREDICTION_SURFACE_NAME in VALID_SURFACES, (
    f"surface name {CACHE_PREDICTION_SURFACE_NAME!r} not "
    f"in VALID_SURFACES {sorted(VALID_SURFACES)}"
)


_CACHE_PREDICTION_PASSIVE_WAIVER: str = (
    "no per-launch cache-prediction sidecar producer "
    "is wired into the bundle; the cache_prediction "
    "surface is registered to lock the framework "
    "vocabulary and reports `unavailable` while no "
    "predictor payload is emitted."
)


class CachePredictionSurface(Surface):
    """Passive `cache_prediction` surface plugin.

    Registers the surface against the orchestrator so
    the framework-wide invariant
    `VALID_SURFACES ⊆ default_surface_plugins().keys()`
    holds, without emitting a per-launch prediction
    payload.

    All three abstract methods return
    `SURFACE_STATUS_UNAVAILABLE` with the same waiver
    naming the missing producer + predictor body. The
    plugin is NOT eval-driving — it never spawns the
    upstream evaluator, never invokes NCU, never
    mutates clocks, and imports zero calibration-
    subtree symbols (NCU quarantine preserved).

    `schema_version` / `impl_version` enter the
    orchestrator's cache-identity tuple so a
    predictor-body change invalidates only this
    surface's cache slot, not the whole bundle.
    """

    name: str = CACHE_PREDICTION_SURFACE_NAME
    schema_version: str = "1"
    impl_version: str = "1"

    def probe(self, host_info: HostInfo) -> SurfaceProbeResult:
        return SurfaceProbeResult(
            status=SURFACE_STATUS_UNAVAILABLE,
            tool_version=None,
            waiver_reason=_CACHE_PREDICTION_PASSIVE_WAIVER,
        )

    def run(
        self,
        baseline_dir: Path,
        problem_dir: Path,
        bundle_dir: Path,
        build_mode: str,
        workload_subset: List[int],
        dispatch_command: Optional[List[str]] = None,
        plan: Optional[Any] = None,
    ) -> SurfaceRunResult:
        return SurfaceRunResult(
            status=SURFACE_STATUS_UNAVAILABLE,
            waiver_reason=_CACHE_PREDICTION_PASSIVE_WAIVER,
        )

    def parse(self, bundle_dir: Path) -> SurfaceParseResult:
        return SurfaceParseResult(
            status=SURFACE_STATUS_UNAVAILABLE,
            artifact_path=None,
            waiver_reason=_CACHE_PREDICTION_PASSIVE_WAIVER,
            size_bytes=0,
            compressed_size_bytes=0,
        )


__all__ = [
    "CACHE_PREDICTION_CONFIDENCE_BAND_FORMAT",
    "CACHE_PREDICTION_CONFIDENCE_BAND_REGEX",
    "CACHE_PREDICTION_CONFIDENCE_BAND_REGEX_PATTERN",
    "CACHE_PREDICTION_PREDICTION_MODES",
    "CACHE_PREDICTION_PREDICTION_MODE_IN_DISTRIBUTION",
    "CACHE_PREDICTION_PREDICTION_MODE_ZERO_SHOT",
    "CACHE_PREDICTION_SURFACE_ARTIFACT_RELPATH",
    "CACHE_PREDICTION_SURFACE_COMMAND_RE",
    "CACHE_PREDICTION_SURFACE_NAME",
    "CACHE_PREDICTION_SURFACE_TYPED_REASONS",
    "CachePredictionProvenance",
    "CachePredictionSurface",
    "CachePredictionSurfaceEntry",
    "CachePredictionSurfaceError",
    "cache_prediction_provenance_from_pooled_head_fit_result",
    "emit_cache_prediction_surface_entry",
    "parse_cache_prediction_confidence_band",
    "validate_cache_prediction_provenance",
    "validate_cache_prediction_surface_entry",
]
