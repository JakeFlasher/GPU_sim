"""Variant-aware calibrator-dispatch tables for the sibling
coverage-expansion artifact family.

The deployed predictor's default-arch dispatch tables in
``baselines_harness.metrics.cache_predictor``
(``_CALIBRATOR_PARAMS_BY_ARCH``, ``_LEDGER_PATH_BY_ARCH``,
``_VALIDATION_SPLIT_PATH_BY_ARCH``) remain the load-bearing
path the predictor walks unchanged. The variant tables here
add a typed indirection a variant-aware caller uses when it
needs the ``coverage_expansion_v1`` sibling family (or any
future named variant) without rewriting the default-arch
contract.

The ``variant=None`` resolver path delegates to the default-arch
table; the historical lookup signature is preserved for every
caller that does not opt in to a variant. This module is
deliberately positioned in the coverage-expansion subpackage
(NOT in ``metrics/cache_predictor.py``) so the baseline-immutability
gate on the deployed predictor file is not perturbed by
variant-table additions.

Closed vocabulary for variant tokens: any value outside
``CALIBRATOR_VARIANT_VOCABULARY`` raises
``CalibratorVariantDispatchError`` from
``resolve_calibrator_params_path`` / its sibling resolvers.
Adding a new variant requires extending the vocabulary AND the
per-arch dispatch entries together.

Important: the coverage-expansion variant resolves to a sidecar
path that does NOT necessarily exist on disk yet. Earlier
stages of the coverage-expansion work emit the path-pinned
artifacts; until then the upstream loader fails closed via
``calibrator_live_artifact_missing`` when the variant is
selected. The dispatch tables themselves are pure
string-resolution — they do NOT touch disk.
"""

from __future__ import annotations

from typing import Final

from sol_baselines_harness.metrics.cache_predictor import (
    _CALIBRATOR_PARAMS_BY_ARCH,
    _LEDGER_PATH_BY_ARCH,
    _VALIDATION_SPLIT_PATH_BY_ARCH,
    PredictorCalibratorLoaderError,
    _resolve_sidecar_path,
    load_calibrator_for_arch,
)


# Sibling variant-keyed dispatch tables. Each maps
# ``(arch, variant)`` → resolved relative path. A ``None``
# variant falls back to the corresponding default-arch table in
# ``cache_predictor.py`` (the existing deployed predictor path).
_CALIBRATOR_PARAMS_BY_ARCH_VARIANT: Final[dict[tuple[str, str], str]] = {
    ("sm_120", "coverage_expansion_v1"): (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "isotonic_params_sm120_coverage_expansion_v1.toml"
    ),
    ("sm_86", "pilot_v1"): (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "isotonic_params_sm86.toml"
    ),
}


_LEDGER_PATH_BY_ARCH_VARIANT: Final[dict[tuple[str, str], str]] = {
    ("sm_120", "coverage_expansion_v1"): (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "label_ledger_v1_sm120_coverage_expansion_v1.toml"
    ),
    ("sm_86", "pilot_v1"): (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "label_ledger_v1_sm86_pilot_v1.toml"
    ),
}


_VALIDATION_SPLIT_PATH_BY_ARCH_VARIANT: Final[
    dict[tuple[str, str], str]
] = {
    ("sm_120", "coverage_expansion_v1"): (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "historical_label_split_v1_sm120_coverage_expansion_v1.toml"
    ),
    ("sm_86", "pilot_v1"): (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "historical_label_split_v1_sm86_pilot_v1.toml"
    ),
}


CALIBRATOR_VARIANT_VOCABULARY: Final[frozenset[str]] = frozenset({
    "coverage_expansion_v1",
    "pilot_v1",
})


_DISPATCH_ERROR_VOCAB: Final[frozenset[str]] = frozenset({
    "calibrator_variant_unknown_token",
    "calibrator_variant_missing_arch_entry",
})


class CalibratorVariantDispatchError(Exception):
    """Raised when the variant-aware resolvers reject the
    ``(arch, variant)`` request.

    Carries a closed-vocabulary ``typed_reason`` so report-layer
    callers can dispatch without parsing message strings."""

    def __init__(self, typed_reason: str, message: str) -> None:
        if typed_reason not in _DISPATCH_ERROR_VOCAB:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of {sorted(_DISPATCH_ERROR_VOCAB)}"
            )
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


def _resolve_variant_aware(
    arch: str,
    variant: str | None,
    *,
    default_table: dict[str, str],
    variant_table: dict[tuple[str, str], str],
) -> str:
    """Common variant-aware resolution.

    - ``variant is None`` → look up in ``default_table``. Raises
      ``PredictorCalibratorLoaderError`` on unknown arch.
    - non-None variant → validate against
      ``CALIBRATOR_VARIANT_VOCABULARY``, then look up
      ``(arch, variant)`` in ``variant_table``. Raises
      ``CalibratorVariantDispatchError`` on unknown variant or
      missing arch entry.
    """
    if variant is None:
        if arch not in _CALIBRATOR_PARAMS_BY_ARCH:
            raise PredictorCalibratorLoaderError(
                "calibrator_unknown_arch_tag",
                f"arch_tag {arch!r} not in default dispatch table; "
                f"known arches: {sorted(_CALIBRATOR_PARAMS_BY_ARCH)}",
            )
        return default_table.get(arch, "")
    if variant not in CALIBRATOR_VARIANT_VOCABULARY:
        raise CalibratorVariantDispatchError(
            "calibrator_variant_unknown_token",
            f"variant {variant!r} not in closed vocabulary "
            f"{sorted(CALIBRATOR_VARIANT_VOCABULARY)}",
        )
    key = (arch, variant)
    if key not in variant_table:
        raise CalibratorVariantDispatchError(
            "calibrator_variant_missing_arch_entry",
            f"variant table has no entry for {key!r}; known keys: "
            f"{sorted(variant_table)}",
        )
    return variant_table[key]


def resolve_calibrator_params_path(
    arch: str, variant: str | None = None
) -> str:
    """Variant-aware resolver for the calibrator-params path.

    ``variant=None`` returns the same path the deployed
    predictor's ``_CALIBRATOR_PARAMS_BY_ARCH[arch]`` lookup
    resolves to (preserves the historical lookup signature;
    unchanged for all existing callers). Other variants resolve
    via ``_CALIBRATOR_PARAMS_BY_ARCH_VARIANT[(arch, variant)]``.
    """
    return _resolve_variant_aware(
        arch,
        variant,
        default_table=_CALIBRATOR_PARAMS_BY_ARCH,
        variant_table=_CALIBRATOR_PARAMS_BY_ARCH_VARIANT,
    )


def resolve_ledger_path(
    arch: str, variant: str | None = None
) -> str:
    """Variant-aware resolver for the per-arch ledger path."""
    return _resolve_variant_aware(
        arch,
        variant,
        default_table=_LEDGER_PATH_BY_ARCH,
        variant_table=_LEDGER_PATH_BY_ARCH_VARIANT,
    )


def resolve_validation_split_path(
    arch: str, variant: str | None = None
) -> str:
    """Variant-aware resolver for the per-arch validation-split
    path."""
    return _resolve_variant_aware(
        arch,
        variant,
        default_table=_VALIDATION_SPLIT_PATH_BY_ARCH,
        variant_table=_VALIDATION_SPLIT_PATH_BY_ARCH_VARIANT,
    )


def load_calibrator_for_arch_variant(
    arch_tag: str,
    variant: str | None = None,
    *,
    expected_ledger_sha256: str | None = None,
    expected_split_sha256: str | None = None,
):
    """Variant-aware wrapper around the deployed predictor's
    ``load_calibrator_for_arch``.

    Two paths:

    1. ``variant is None`` — delegates to
       ``load_calibrator_for_arch(arch_tag, ...)`` unchanged.
       This is the historical lookup signature; every existing
       caller that does not opt in to a variant sees byte-stable
       behavior.

    2. ``variant`` is a member of
       ``CALIBRATOR_VARIANT_VOCABULARY`` — resolves the variant
       sidecar / ledger / split paths via the same resolvers
       above, then checks that the resolved sidecar exists on
       disk. The variant sidecar emission step has not yet
       completed (it's part of a future round's mainline work),
       so the file is currently missing for every
       ``(arch, variant)`` pair — the wrapper fails closed via
       ``PredictorCalibratorLoaderError("calibrator_live_artifact_missing")``.
       When the variant sidecar lands, this branch's loading
       semantics will mirror ``load_calibrator_for_arch`` but
       against the variant-resolved paths.

    3. Unknown variant token — raises
       ``CalibratorVariantDispatchError`` from
       ``_resolve_variant_aware`` (closed-vocabulary typed
       reason).

    Contract:
      - ``cache_predictor.py`` remains byte-stable (no edits
        required to wire the variant; this wrapper does the
        wiring entirely from the coverage-expansion subpackage).
      - The wrapper never silently promotes a missing-file row
        to a usable calibrator; the failure mode is typed and
        load-bearing for the artifact-consistency QA that
        arrives in a later round's mainline work.
    """
    if variant is None:
        return load_calibrator_for_arch(
            arch_tag,
            expected_ledger_sha256=expected_ledger_sha256,
            expected_split_sha256=expected_split_sha256,
        )

    # Validate variant + (arch, variant) entry. Raises
    # CalibratorVariantDispatchError on unknown variant token
    # or missing arch entry.
    sidecar_relpath = resolve_calibrator_params_path(arch_tag, variant)
    ledger_relpath = resolve_ledger_path(arch_tag, variant)
    split_relpath = resolve_validation_split_path(arch_tag, variant)

    sidecar_path = _resolve_sidecar_path(sidecar_relpath)
    if not sidecar_path.exists():
        raise PredictorCalibratorLoaderError(
            "calibrator_live_artifact_missing",
            f"variant {variant!r} for arch_tag {arch_tag!r} "
            f"resolves to {sidecar_relpath} which does not "
            "exist on disk; the variant sidecar emission step "
            "has not yet completed",
        )

    # File exists path: delegate to the deployed loader with
    # path overrides. The loader runs the same schema / arch /
    # metric-family / ledger-SHA / split-SHA / anchor-shape
    # gates against the variant-resolved paths. Returns a
    # fitted IsotonicCalibrator (or TypedUnavailableSidecar
    # when the sidecar's calibrator_status is typed_unavailable).
    return load_calibrator_for_arch(
        arch_tag,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_split_sha256=expected_split_sha256,
        sidecar_path_override=sidecar_relpath,
        ledger_path_override=ledger_relpath,
        split_path_override=split_relpath,
    )
