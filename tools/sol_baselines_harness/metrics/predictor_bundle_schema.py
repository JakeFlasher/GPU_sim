"""Predictor bundle schema validator (deployed scope).

Validates the shape + invariants of a predictor bundle
emitted by the future cache_prediction surface. Mirrors
the sibling project's
`/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/schemas/cache_predictor.schema.json`
contract: one bundle per `(slug, shape_point,
binary_sha256, source_commit,
kernel_signature_sha256)` cell records a predictor
result with one of three result arms (scalar / bounded
/ typed_unavailable).

## Surfaces

  * `PREDICTOR_BUNDLE_RESULT_KIND_SCALAR` /
    `_BOUNDED` / `_TYPED_UNAVAILABLE` — string
    constants for the three result arms.
  * `PREDICTOR_BUNDLE_RESULT_KINDS` — closed frozenset
    of the three arms.
  * `ManifestPairingIndex` — 5-field provenance
    binding (`slug`, `shape_point`, `binary_sha256`,
    `source_commit`, `kernel_signature_sha256`).
    Pinned to the sibling
    `manifest_pairing_index` object shape verbatim.
  * `PredictorBundleArmPayload` — opaque arm-specific
    mapping. The composite validator routes through
    arm-specific required/forbidden field contracts.
  * `PredictorBundle` — composite bundle dataclass.
  * `validate_predictor_bundle_result_kind(value)` —
    closed-enum check on the result-kind string.
  * `validate_predictor_bundle_design_coverage(value)`
    — closed-enum check on the predictor-design-
    coverage bucket (re-uses `PREDICTOR_BUCKETS`).
  * `validate_manifest_pairing_index(index)` — shape
    check + SHA-256 routing on the two digest fields.
  * `validate_predictor_bundle(bundle)` — composite
    validator that runs every check + the arm-specific
    contract.
  * `validate_pairing_index_against_cell_join_key(index,
    key)` — cross-check that the 5 pairing-index
    fields match the 5 corresponding `CellJoinKey`
    fields.

Imports stdlib + the deployed `framework_contract` +
`predictor_design_coverage` modules only. NO
calibration-subtree imports; the NCU quarantine
boundary is preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from sol_baselines_harness.metrics.framework_contract import (
    CI_CONFIDENCE,
    CellJoinKey,
    FrameworkContractError,
    INPUT_SIDECAR_REQUIRED_NAMES,
    InputSidecarDigest,
    validate_input_sidecar_digests,
    validate_sha256_digest,
)
from sol_baselines_harness.metrics.predictor_design_coverage import (
    PREDICTOR_BUCKETS,
)


# Closed result-kind enum. The sibling schema's
# `result_kind` property pins exactly these three
# strings; the validator rejects any other value.
PREDICTOR_BUNDLE_RESULT_KIND_SCALAR: str = "scalar"
PREDICTOR_BUNDLE_RESULT_KIND_BOUNDED: str = "bounded"
PREDICTOR_BUNDLE_RESULT_KIND_TYPED_UNAVAILABLE: str = "typed_unavailable"

PREDICTOR_BUNDLE_RESULT_KINDS: frozenset[str] = frozenset({
    PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
    PREDICTOR_BUNDLE_RESULT_KIND_BOUNDED,
    PREDICTOR_BUNDLE_RESULT_KIND_TYPED_UNAVAILABLE,
})


# Closed typed-reason enum for the schema validator. A
# downstream caller can branch on the typed_reason to
# compose a typed-unavailable bundle or a report-table
# error without parsing message strings.
PREDICTOR_BUNDLE_SCHEMA_TYPED_REASONS: frozenset[str] = frozenset({
    "predictor_bundle_schema_result_kind_invalid",
    "predictor_bundle_schema_design_coverage_invalid",
    "predictor_bundle_schema_manifest_pairing_index_invalid",
    "predictor_bundle_schema_pairing_index_drift",
    "predictor_bundle_schema_arm_required_field_missing",
    "predictor_bundle_schema_arm_forbidden_field_present",
    "predictor_bundle_schema_arm_invariant_violated",
    "predictor_bundle_schema_sidecar_digest_invalid",
    "predictor_bundle_schema_sidecar_circular_input",
})


# Arm-specific required-field contracts. The sibling
# schema pins these via per-arm `if/then` blocks with
# `required` arrays; we encode them as frozensets here.
_SCALAR_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "predicted_hit_rate",
    "ci_low",
    "ci_high",
    "ci_half_width",
    "ci_confidence",
})
_SCALAR_FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "typed_reason",
    "hit_rate_lower",
    "hit_rate_upper",
})

_BOUNDED_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "hit_rate_lower",
    "hit_rate_upper",
    "underconfigured_bin_count",
    "admitted_bin_count",
})
_BOUNDED_FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "typed_reason",
    "predicted_hit_rate",
    "ci_low",
    "ci_high",
    "ci_half_width",
    "ci_confidence",
})

_TYPED_UNAVAILABLE_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "typed_reason",
    "underconfigured_bin_count",
    "admitted_bin_count",
})
_TYPED_UNAVAILABLE_FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "predicted_hit_rate",
    "ci_low",
    "ci_high",
    "ci_half_width",
    "ci_confidence",
    "hit_rate_lower",
    "hit_rate_upper",
})


# IEEE-754 tolerance for the scalar arm's
# `ci_half_width == (ci_high - ci_low) / 2.0` invariant.
# The sibling project's bootstrap CI is computed with
# `numpy.quantile` which can introduce a few ULPs of
# round-trip noise; the invariant test uses a
# conservative absolute tolerance.
_CI_HALF_WIDTH_CONSISTENCY_TOL: float = 1e-9


class PredictorBundleSchemaError(Exception):
    """Raised on predictor-bundle schema violations.

    Carries the typed-reason string so downstream
    callers can compose typed-unavailable bundles or
    report-table errors without parsing message
    strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class ManifestPairingIndex:
    """5-field provenance binding mirroring the sibling
    `manifest_pairing_index` object exactly. The
    verifier cross-checks every field against the
    enclosing run summary's `CellJoinKey` to reject a
    bundle copied from a different binary / kernel /
    source_commit.

    Note: this is a strict subset of the 7-tuple
    `CellJoinKey` — the pairing index omits `arch` and
    `ncu_metric_version` because the enclosing run
    summary already carries those once at the
    per-arch level.
    """

    slug: str
    shape_point: str
    binary_sha256: str
    source_commit: str
    kernel_signature_sha256: str


@dataclass(frozen=True)
class PredictorBundle:
    """Composite predictor bundle.

    `result_kind` selects the arm; `arm_payload` carries
    the arm-specific fields (`predicted_hit_rate` /
    `ci_*` for scalar, `hit_rate_lower` /
    `hit_rate_upper` / `*_bin_count` for bounded,
    `typed_reason` / `*_bin_count` for
    typed_unavailable). The composite validator routes
    the arm payload through the arm-specific contract;
    the dataclass itself uses a plain mapping so the
    validator can audit per-arm presence/absence.

    `algorithm_version` is the predictor model
    identifier (e.g., `smith_eklund_lru_v1`) — the
    sibling schema uses `minLength: 1`, so an empty
    string is invalid.

    `input_sidecar_digests` carries the
    `InputSidecarDigest` records the predictor consumed.
    The composite validator routes this list through
    `framework_contract.validate_input_sidecar_digests`
    with the arm-conditional required-name contract:
    scalar + bounded arms require `sector_stream`
    (per the sibling cache_predictor.schema.json
    non-circular witness rule); typed_unavailable arms
    may legitimately have no witness when the predictor
    abstains on missing/empty trace, but every entry
    that IS present must still pass the unknown-name +
    circular-output + safe-path checks.
    """

    result_kind: str
    predictor_design_coverage: str
    manifest_pairing_index: ManifestPairingIndex
    algorithm_version: str
    arm_payload: Mapping[str, object] = field(default_factory=dict)
    input_sidecar_digests: tuple[InputSidecarDigest, ...] = ()


def validate_predictor_bundle_result_kind(value: object) -> None:
    """Closed-enum check on the result-kind string.

    Raises
    `predictor_bundle_schema_result_kind_invalid` on
    non-string, empty string, or string outside
    `PREDICTOR_BUNDLE_RESULT_KINDS`.
    """
    if not isinstance(value, str) or not value.strip():
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_result_kind_invalid",
            f"result_kind must be a non-empty string; "
            f"got {value!r}",
        )
    if value not in PREDICTOR_BUNDLE_RESULT_KINDS:
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_result_kind_invalid",
            f"result_kind {value!r} is not in the closed "
            f"enum {sorted(PREDICTOR_BUNDLE_RESULT_KINDS)}",
        )


def validate_predictor_bundle_design_coverage(value: object) -> None:
    """Closed-enum check on the predictor-design-coverage
    bucket. Re-uses `PREDICTOR_BUCKETS` from the
    deployed `predictor_design_coverage` module so
    `predictor_bundle_schema` and the partition stay
    in lockstep.
    """
    if not isinstance(value, str) or not value.strip():
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_design_coverage_invalid",
            f"predictor_design_coverage must be a non-empty "
            f"string; got {value!r}",
        )
    if value not in PREDICTOR_BUCKETS:
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_design_coverage_invalid",
            f"predictor_design_coverage {value!r} is not "
            f"in the closed enum "
            f"{sorted(PREDICTOR_BUCKETS)}",
        )


def validate_manifest_pairing_index(
    index: ManifestPairingIndex,
) -> None:
    """Shape check on the 5-field
    `ManifestPairingIndex`. Free-form fields (`slug`,
    `shape_point`, `source_commit`) must be non-empty
    strings; SHA-256 fields (`binary_sha256`,
    `kernel_signature_sha256`) route through
    `framework_contract.validate_sha256_digest`.
    """
    if not isinstance(index, ManifestPairingIndex):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_manifest_pairing_index_invalid",
            f"manifest_pairing_index must be a "
            f"ManifestPairingIndex; got "
            f"{type(index).__name__}",
        )
    for name in ("slug", "shape_point", "source_commit"):
        value = getattr(index, name)
        if not isinstance(value, str) or not value.strip():
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_manifest_pairing_index_invalid",
                f"manifest_pairing_index.{name} must be a "
                f"non-empty string; got {value!r}",
            )
    for name in ("binary_sha256", "kernel_signature_sha256"):
        value = getattr(index, name)
        try:
            validate_sha256_digest(
                value, field_name=f"manifest_pairing_index.{name}",
            )
        except FrameworkContractError as exc:
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_manifest_pairing_index_invalid",
                f"manifest_pairing_index.{name}: "
                f"{exc.message}",
            ) from exc


def validate_pairing_index_against_cell_join_key(
    index: ManifestPairingIndex,
    key: CellJoinKey,
) -> None:
    """Cross-check that the 5 pairing-index fields
    match the 5 corresponding `CellJoinKey` fields
    bit-for-bit. A mismatch on ANY field raises
    `predictor_bundle_schema_pairing_index_drift`
    naming the disagreeing field so the caller can
    fix the wiring without parsing strings.

    The shared fields are `slug` / `shape_point` /
    `binary_sha256` / `source_commit` /
    `kernel_signature_sha256`; the pairing index does
    NOT include `arch` or `ncu_metric_version` because
    those live at the enclosing run-summary level.
    """
    if not isinstance(index, ManifestPairingIndex):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_manifest_pairing_index_invalid",
            f"manifest_pairing_index must be a "
            f"ManifestPairingIndex; got "
            f"{type(index).__name__}",
        )
    if not isinstance(key, CellJoinKey):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_pairing_index_drift",
            f"cell_join_key must be a CellJoinKey; got "
            f"{type(key).__name__}",
        )
    for field_name in (
        "slug", "shape_point", "binary_sha256",
        "source_commit", "kernel_signature_sha256",
    ):
        index_value = getattr(index, field_name)
        key_value = getattr(key, field_name)
        if index_value != key_value:
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_pairing_index_drift",
                f"manifest_pairing_index.{field_name} "
                f"disagrees with cell_join_key.{field_name}: "
                f"{index_value!r} vs {key_value!r}",
            )


def _validate_arm_payload(
    result_kind: str,
    payload: Mapping[str, object],
) -> None:
    """Per-arm required / forbidden field contract.

    Required-field absence raises
    `predictor_bundle_schema_arm_required_field_missing`;
    forbidden-field presence raises
    `predictor_bundle_schema_arm_forbidden_field_present`.
    Each diagnostic names the arm + the offending
    field.
    """
    if not isinstance(payload, Mapping):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_required_field_missing",
            f"arm_payload must be a Mapping; got "
            f"{type(payload).__name__}",
        )
    if result_kind == PREDICTOR_BUNDLE_RESULT_KIND_SCALAR:
        required = _SCALAR_REQUIRED_FIELDS
        forbidden = _SCALAR_FORBIDDEN_FIELDS
    elif result_kind == PREDICTOR_BUNDLE_RESULT_KIND_BOUNDED:
        required = _BOUNDED_REQUIRED_FIELDS
        forbidden = _BOUNDED_FORBIDDEN_FIELDS
    elif result_kind == PREDICTOR_BUNDLE_RESULT_KIND_TYPED_UNAVAILABLE:
        required = _TYPED_UNAVAILABLE_REQUIRED_FIELDS
        forbidden = _TYPED_UNAVAILABLE_FORBIDDEN_FIELDS
    else:
        # Should not reach here — caller validates
        # result_kind upstream — but the fallthrough
        # check keeps the function defensive against
        # internal-call ordering errors.
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_result_kind_invalid",
            f"result_kind {result_kind!r} is not in the "
            f"closed enum",
        )
    missing = sorted(required - set(payload))
    if missing:
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_required_field_missing",
            f"arm {result_kind!r} requires fields "
            f"{sorted(required)}; missing: {missing}",
        )
    present_forbidden = sorted(forbidden & set(payload))
    if present_forbidden:
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_forbidden_field_present",
            f"arm {result_kind!r} forbids fields "
            f"{sorted(forbidden)}; present: "
            f"{present_forbidden}",
        )


def _validate_arm_invariants(
    result_kind: str,
    payload: Mapping[str, object],
) -> None:
    """Per-arm value invariants beyond presence /
    absence:

      * scalar arm:
        `0 <= ci_low <= predicted_hit_rate <= ci_high <= 1`
        and `ci_half_width == (ci_high - ci_low) / 2`
        within IEEE-754 tolerance, `0 < ci_confidence
        < 1`.
      * bounded arm: `0 <= hit_rate_lower <=
        hit_rate_upper <= 1`.
      * typed_unavailable arm: `typed_reason` is a
        non-empty string.

    All invariant failures raise
    `predictor_bundle_schema_arm_invariant_violated`
    naming the violated invariant.
    """
    if result_kind == PREDICTOR_BUNDLE_RESULT_KIND_SCALAR:
        predicted = _as_float(payload, "predicted_hit_rate")
        ci_low = _as_float(payload, "ci_low")
        ci_high = _as_float(payload, "ci_high")
        ci_half_width = _as_float(payload, "ci_half_width")
        ci_confidence = _as_float(payload, "ci_confidence")
        _require_unit_interval("predicted_hit_rate", predicted)
        _require_unit_interval("ci_low", ci_low)
        _require_unit_interval("ci_high", ci_high)
        if ci_low > predicted or predicted > ci_high:
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"scalar arm requires "
                f"ci_low <= predicted_hit_rate <= ci_high; "
                f"got ci_low={ci_low!r}, "
                f"predicted_hit_rate={predicted!r}, "
                f"ci_high={ci_high!r}",
            )
        expected_half = (ci_high - ci_low) / 2.0
        if abs(ci_half_width - expected_half) > _CI_HALF_WIDTH_CONSISTENCY_TOL:
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"scalar arm requires "
                f"ci_half_width == (ci_high - ci_low) / 2 "
                f"within tolerance "
                f"{_CI_HALF_WIDTH_CONSISTENCY_TOL!r}; "
                f"got ci_half_width={ci_half_width!r}, "
                f"expected ~{expected_half!r}",
            )
        # ci_confidence is pinned to the canonical
        # cross-study bootstrap-CI confidence constant
        # (`framework_contract.CI_CONFIDENCE = 0.95`).
        # The sibling cache_predictor.schema.json uses
        # `const: 0.95`; an off-canonical value here is
        # schema drift, not an allowed model choice.
        if ci_confidence != CI_CONFIDENCE:
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"scalar arm requires "
                f"ci_confidence == {CI_CONFIDENCE!r} "
                f"(canonical framework-contract value); "
                f"got {ci_confidence!r}",
            )
    elif result_kind == PREDICTOR_BUNDLE_RESULT_KIND_BOUNDED:
        lower = _as_float(payload, "hit_rate_lower")
        upper = _as_float(payload, "hit_rate_upper")
        _require_unit_interval("hit_rate_lower", lower)
        _require_unit_interval("hit_rate_upper", upper)
        if lower > upper:
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"bounded arm requires "
                f"hit_rate_lower <= hit_rate_upper; got "
                f"{lower!r} > {upper!r}",
            )
        # Bin-count fields routed through the
        # non-negative-int helper so a negative / float /
        # string / non-finite value fails at the
        # invariant-violation boundary.
        _as_nonnegative_int(payload, "underconfigured_bin_count")
        _as_nonnegative_int(payload, "admitted_bin_count")
    elif result_kind == PREDICTOR_BUNDLE_RESULT_KIND_TYPED_UNAVAILABLE:
        reason = payload.get("typed_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"typed_unavailable arm requires "
                f"typed_reason to be a non-empty string; "
                f"got {reason!r}",
            )
        # Same bin-count validation applies for the
        # typed-unavailable arm (both bin counts are in
        # its required-field list).
        _as_nonnegative_int(payload, "underconfigured_bin_count")
        _as_nonnegative_int(payload, "admitted_bin_count")


def _as_float(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    # JSON Schema `type: number` does not include
    # booleans, but Python's `isinstance(True, int)` is
    # True so a `(int, float)` check would silently
    # admit `True`/`False` as `1.0`/`0.0`. Explicit
    # bool rejection mirrors the same guard in
    # `_as_nonnegative_int()` and keeps the bundle
    # validator aligned with the sibling schema's
    # type contract.
    if isinstance(value, bool):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_invariant_violated",
            f"arm payload field {name!r} must be a "
            f"number; got bool {value!r}",
        )
    if not isinstance(value, (int, float)):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_invariant_violated",
            f"arm payload field {name!r} must be a number; "
            f"got {type(value).__name__}",
        )
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_invariant_violated",
            f"arm payload field {name!r} is non-finite "
            f"({f!r})",
        )
    return f


def _require_unit_interval(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_invariant_violated",
            f"arm payload field {name!r} must be in "
            f"[0, 1]; got {value!r}",
        )


def _as_nonnegative_int(
    payload: Mapping[str, object], name: str,
) -> int:
    """Coerce a payload field to a non-negative integer.

    Rejects every non-integer shape with the
    invariant-violated typed reason: bool (sneaks in
    via `isinstance(True, int)` so we exclude it
    explicitly), non-integer floats (`1.5`), strings,
    `None`, non-finite floats. An integer-valued float
    like `1.0` is accepted (Python's `int(1.0)` is
    lossless) so the contract doesn't reject otherwise-
    valid TOML/JSON-roundtripped data.

    Pinned for the sibling schema's `minimum: 0`
    integer contract on `underconfigured_bin_count`
    and `admitted_bin_count`.
    """
    value = payload.get(name)
    # `bool` is a subclass of `int`; reject it
    # explicitly because the count contract is integer
    # semantics, not boolean.
    if isinstance(value, bool):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_invariant_violated",
            f"arm payload field {name!r} must be a "
            f"non-negative integer; got bool {value!r}",
        )
    if isinstance(value, int):
        if value < 0:
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"arm payload field {name!r} must be a "
                f"non-negative integer; got {value!r}",
            )
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"arm payload field {name!r} is "
                f"non-finite ({value!r})",
            )
        # Reject non-integer-valued floats; accept
        # integer-valued floats (e.g. `5.0`).
        if value != int(value):
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"arm payload field {name!r} must be a "
                f"non-negative integer; got non-integer "
                f"float {value!r}",
            )
        ival = int(value)
        if ival < 0:
            raise PredictorBundleSchemaError(
                "predictor_bundle_schema_arm_invariant_violated",
                f"arm payload field {name!r} must be a "
                f"non-negative integer; got {value!r}",
            )
        return ival
    raise PredictorBundleSchemaError(
        "predictor_bundle_schema_arm_invariant_violated",
        f"arm payload field {name!r} must be a "
        f"non-negative integer; got "
        f"{type(value).__name__} ({value!r})",
    )


def validate_predictor_bundle(bundle: PredictorBundle) -> None:
    """Composite predictor-bundle validator.

    Runs every check in turn so a downstream consumer
    fails closed on the first violation:
      1. `result_kind` enum check.
      2. `predictor_design_coverage` enum check.
      3. `manifest_pairing_index` shape + SHA check.
      4. `algorithm_version` non-empty string.
      5. `input_sidecar_digests` contract: scalar +
         bounded arms require `sector_stream` (per the
         sibling cache_predictor.schema.json
         non-circular witness rule);
         typed_unavailable arms may have empty
         sidecar list but all entries (if any) still
         pass unknown-name / circular-output / safe-
         path checks.
      6. Arm-specific required + forbidden field
         contract.
      7. Arm-specific value invariants.
    """
    if not isinstance(bundle, PredictorBundle):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_result_kind_invalid",
            f"argument must be a PredictorBundle; got "
            f"{type(bundle).__name__}",
        )
    validate_predictor_bundle_result_kind(bundle.result_kind)
    validate_predictor_bundle_design_coverage(
        bundle.predictor_design_coverage,
    )
    validate_manifest_pairing_index(bundle.manifest_pairing_index)
    if (
        not isinstance(bundle.algorithm_version, str)
        or not bundle.algorithm_version.strip()
    ):
        raise PredictorBundleSchemaError(
            "predictor_bundle_schema_arm_required_field_missing",
            f"algorithm_version must be a non-empty "
            f"string; got {bundle.algorithm_version!r}",
        )
    _validate_input_sidecar_digests_for_bundle(bundle)
    _validate_arm_payload(bundle.result_kind, bundle.arm_payload)
    _validate_arm_invariants(bundle.result_kind, bundle.arm_payload)


# Map framework-contract typed reasons to bundle-schema
# typed reasons so the bundle's typed-reason vocabulary
# stays internal to the bundle layer. Downstream
# consumers branch on `PREDICTOR_BUNDLE_SCHEMA_TYPED_REASONS`
# only; they should never see a raw framework-contract
# reason leak through.
_FRAMEWORK_TO_BUNDLE_TYPED_REASON: Mapping[str, str] = {
    "framework_contract_sidecar_digest_invalid":
        "predictor_bundle_schema_sidecar_digest_invalid",
    "framework_contract_sidecar_digest_required_missing":
        "predictor_bundle_schema_sidecar_digest_invalid",
    "framework_contract_sidecar_digest_duplicate":
        "predictor_bundle_schema_sidecar_digest_invalid",
    "framework_contract_sidecar_circular_input":
        "predictor_bundle_schema_sidecar_circular_input",
    "framework_contract_sha256_malformed":
        "predictor_bundle_schema_sidecar_digest_invalid",
}


def _validate_input_sidecar_digests_for_bundle(
    bundle: PredictorBundle,
) -> None:
    """Route `bundle.input_sidecar_digests` through the
    framework-contract validator with arm-conditional
    required names. Translate any
    `FrameworkContractError` into the matching bundle-
    schema typed reason.
    """
    if bundle.result_kind == PREDICTOR_BUNDLE_RESULT_KIND_TYPED_UNAVAILABLE:
        # Typed-unavailable bundles may legitimately
        # have no sidecar witness (the predictor
        # abstained on missing/empty trace). Still run
        # validation on whatever IS present so unknown
        # names, circular names, and unsafe paths are
        # rejected.
        required_names: tuple[str, ...] = ()
    else:
        # Scalar + bounded require the canonical
        # `sector_stream` witness per the sibling
        # cache_predictor.schema.json non-circular rule.
        required_names = INPUT_SIDECAR_REQUIRED_NAMES
    try:
        validate_input_sidecar_digests(
            bundle.input_sidecar_digests,
            required_names=required_names,
        )
    except FrameworkContractError as exc:
        bundle_reason = _FRAMEWORK_TO_BUNDLE_TYPED_REASON.get(
            exc.typed_reason,
            "predictor_bundle_schema_sidecar_digest_invalid",
        )
        raise PredictorBundleSchemaError(
            bundle_reason,
            f"input_sidecar_digests: {exc.message}",
        ) from exc


__all__ = [
    "PREDICTOR_BUNDLE_RESULT_KIND_BOUNDED",
    "PREDICTOR_BUNDLE_RESULT_KIND_SCALAR",
    "PREDICTOR_BUNDLE_RESULT_KIND_TYPED_UNAVAILABLE",
    "PREDICTOR_BUNDLE_RESULT_KINDS",
    "PREDICTOR_BUNDLE_SCHEMA_TYPED_REASONS",
    "ManifestPairingIndex",
    "PredictorBundle",
    "PredictorBundleSchemaError",
    "validate_manifest_pairing_index",
    "validate_pairing_index_against_cell_join_key",
    "validate_predictor_bundle",
    "validate_predictor_bundle_design_coverage",
    "validate_predictor_bundle_result_kind",
]
