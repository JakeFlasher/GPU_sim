"""Corpus-expansion batch manifest schema, loader, and
typed-error vocabulary.

The batch manifest (`batch-NNN.toml` under the operator's
corpus-expansion directory) declares a stratified slug batch
that downstream hand-port + pack + label-harvest tooling
consumes. Every batch is bounded by two ceilings:

  * `max_slugs` (integer) is capped at
    `CORPUS_EXPANSION_BATCH_MAX_SLUGS` (12).
  * `max_estimated_runtime_minutes` (integer) is capped at
    `CORPUS_EXPANSION_BATCH_MAX_RUNTIME_MINUTES` (120).

The summed per-slug `estimated_runtime_minutes` must not exceed
`max_estimated_runtime_minutes`. The per-family stratification
counts must sum to `len(slugs)` so the batch is provably balanced
across the four canonical SOL-ExecBench categories
(`L1`, `L2`, `Quant`, `FIB`).

The operator-facing schema reference (every required field,
closed vocabulary, ceiling, typed reason, and an inline
well-formed example) lives next to this module at
`corpus_expansion_batch_schema.toml`. The runtime contract is
this module's typed-reason vocabulary; the schema TOML is the
human reference and is not loaded at runtime.

This module is calibration-only and never imported by the
deployed harness path. The NCU quarantine scanner enforces that
contract.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


CORPUS_EXPANSION_BATCH_SCHEMA_VERSION: str = "1.1"
CORPUS_EXPANSION_BATCH_LEGACY_SCHEMA_VERSION: str = "1.0"

CORPUS_EXPANSION_BATCH_SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({
    CORPUS_EXPANSION_BATCH_LEGACY_SCHEMA_VERSION,
    CORPUS_EXPANSION_BATCH_SCHEMA_VERSION,
})

CORPUS_EXPANSION_BATCH_MAX_SLUGS: int = 12
CORPUS_EXPANSION_BATCH_MAX_RUNTIME_MINUTES: int = 120

FRIENDLY_MULTI_SLUG_MIN_SLUGS: int = 3
FRIENDLY_MULTI_SLUG_MAX_SLUGS: int = 6


TARGET_COHORT_PREDICTOR_FRIENDLY: str = "predictor_friendly"
TARGET_COHORT_PREDICTOR_HOSTILE: str = "predictor_hostile"

CORPUS_EXPANSION_BATCH_TARGET_COHORTS: frozenset[str] = frozenset({
    TARGET_COHORT_PREDICTOR_FRIENDLY,
    TARGET_COHORT_PREDICTOR_HOSTILE,
})


TARGET_LANGUAGE_CUDA_CPP: str = "cuda_cpp"
TARGET_LANGUAGE_CUTLASS: str = "cutlass"

CORPUS_EXPANSION_BATCH_TARGET_LANGUAGES: frozenset[str] = frozenset({
    TARGET_LANGUAGE_CUDA_CPP,
    TARGET_LANGUAGE_CUTLASS,
})


FAMILY_L1: str = "L1"
FAMILY_L2: str = "L2"
FAMILY_QUANT: str = "Quant"
FAMILY_FIB: str = "FIB"

CORPUS_EXPANSION_BATCH_FAMILIES: frozenset[str] = frozenset({
    FAMILY_L1, FAMILY_L2, FAMILY_QUANT, FAMILY_FIB,
})


CADENCE_CLASS_FRIENDLY_MULTI_SLUG: str = "friendly_multi_slug"
CADENCE_CLASS_SINGLE_SLUG_ISOLATED: str = "single_slug_isolated"

CORPUS_EXPANSION_BATCH_CADENCE_CLASSES: frozenset[str] = frozenset({
    CADENCE_CLASS_FRIENDLY_MULTI_SLUG,
    CADENCE_CLASS_SINGLE_SLUG_ISOLATED,
})


ISOLATION_REASON_NOVEL_DISPATCH_FAMILY: str = "novel_dispatch_family"
ISOLATION_REASON_HOSTILE_COHORT: str = "hostile_cohort"
ISOLATION_REASON_UNVERIFIED_DISPATCH_FAMILY: str = (
    "unverified_dispatch_family"
)
ISOLATION_REASON_OPERATOR_OVERRIDE_DEBUG_ISOLATION: str = (
    "operator_override_debug_isolation"
)
ISOLATION_REASON_WARM_COLD_PINNING_CANDIDATE: str = (
    "warm_cold_pinning_candidate"
)

CORPUS_EXPANSION_BATCH_ISOLATION_REASONS: frozenset[str] = frozenset({
    ISOLATION_REASON_NOVEL_DISPATCH_FAMILY,
    ISOLATION_REASON_HOSTILE_COHORT,
    ISOLATION_REASON_UNVERIFIED_DISPATCH_FAMILY,
    ISOLATION_REASON_OPERATOR_OVERRIDE_DEBUG_ISOLATION,
    ISOLATION_REASON_WARM_COLD_PINNING_CANDIDATE,
})


CORPUS_EXPANSION_BATCH_TYPED_REASONS: frozenset[str] = frozenset({
    "corpus_expansion_batch_io_error",
    "corpus_expansion_batch_invalid_toml",
    "corpus_expansion_batch_invalid_schema_version",
    "corpus_expansion_batch_missing_field",
    "corpus_expansion_batch_max_slugs_exceeded",
    "corpus_expansion_batch_max_runtime_exceeded",
    "corpus_expansion_batch_runtime_sum_exceeded",
    "corpus_expansion_batch_slug_count_mismatch",
    "corpus_expansion_batch_duplicate_slug",
    "corpus_expansion_batch_invalid_target_cohort",
    "corpus_expansion_batch_invalid_target_language",
    "corpus_expansion_batch_invalid_runtime_estimate",
    "corpus_expansion_batch_family_stratification_mismatch",
    "corpus_expansion_batch_invalid_family",
    "corpus_expansion_batch_non_canonical_slug",
    "corpus_expansion_batch_host_arch_unsupported_slug",
    "corpus_expansion_batch_missing_promotion_array",
    "corpus_expansion_batch_duplicate_promotion",
    "corpus_expansion_batch_promotion_slug_mismatch",
    "corpus_expansion_batch_promotion_cohort_mismatch",
    "corpus_expansion_batch_landed_slug_mismatch",
    "corpus_expansion_batch_invalid_cadence_class",
    "corpus_expansion_batch_invalid_dispatch_family",
    "corpus_expansion_batch_cadence_uplift_unverified_dispatch_family",
    "corpus_expansion_batch_cadence_uplift_pending_verification_dispatch_family",
    "corpus_expansion_batch_cadence_uplift_hostile_in_multi_slug_batch",
    "corpus_expansion_batch_cadence_class_slug_count_mismatch",
    "corpus_expansion_batch_single_slug_isolated_missing_isolation_reason",
    "corpus_expansion_batch_isolation_reason_outside_closed_vocabulary",
    "corpus_expansion_batch_friendly_multi_slug_slug_count_out_of_range",
})


class CorpusExpansionBatchError(Exception):
    """Typed error raised by the batch manifest loader.

    Mirrors the typed-error pattern used throughout the
    calibration package. `typed_reason` is drawn from the closed
    set `CORPUS_EXPANSION_BATCH_TYPED_REASONS`.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in CORPUS_EXPANSION_BATCH_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of {sorted(CORPUS_EXPANSION_BATCH_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class CorpusExpansionBatchSlugEntry:
    """One per-slug entry inside a batch manifest.

    Fields:
      * `slug` — canonical problem slug (e.g.
        `problem_004_gemm_n128_k2048`).
      * `target_cohort` ∈ `CORPUS_EXPANSION_BATCH_TARGET_COHORTS`.
      * `target_language` ∈
        `CORPUS_EXPANSION_BATCH_TARGET_LANGUAGES`.
      * `estimated_runtime_minutes` — strictly positive integer
        documenting the per-slug harvest cost (NCU + build +
        correctness).
    """

    slug: str
    target_cohort: str
    target_language: str
    estimated_runtime_minutes: int


@dataclass(frozen=True)
class CorpusExpansionBatchPromotionEntry:
    """One slug-promotion entry inside a batch manifest.

    The promotion array predeclares the cohort bucket every
    in-batch slug will get in `predictor_design_coverage.py`
    BEFORE the batch lands. The validator requires the
    promotion's slug set to exactly equal the batch's
    `[[slugs]]` slug set, and requires each promotion's
    `target_cohort` to match the same slug's per-`[[slugs]]`
    `target_cohort`.

    Fields:
      * `slug` — canonical problem slug.
      * `target_cohort` ∈ `CORPUS_EXPANSION_BATCH_TARGET_COHORTS`.
    """

    slug: str
    target_cohort: str


@dataclass(frozen=True)
class CorpusExpansionBatchManifest:
    """Parsed batch manifest.

    `family_stratification` maps each family name (from
    `CORPUS_EXPANSION_BATCH_FAMILIES`) to a non-negative count;
    the count sum equals `len(slugs)`. Families with zero
    contribution from this batch are omitted from the mapping;
    a present family with count `0` is also accepted and treated
    identically.

    `predictor_design_coverage_promotions` is the slug-cohort
    predeclaration array described above. Its slug set MUST
    equal the `slugs` slug set; per-slug `target_cohort` values
    MUST agree between the two arrays.

    The cadence-uplift fields below are introduced in
    `schema_version = "1.1"`. Legacy `"1.0"` manifests parse with
    these fields set to `None`; new manifests at `"1.1"` require
    `dispatch_family_signature` and `cadence_class`, and require
    `isolation_reason` when `cadence_class =
    "single_slug_isolated"`.

      * `dispatch_family_signature` — closed-vocabulary string
        cross-referenced against
        `verified_dispatch_families.toml` (when supplied to the
        loader); identifies the dispatch surface the batch's
        slugs route through.
      * `cadence_class` ∈
        `CORPUS_EXPANSION_BATCH_CADENCE_CLASSES`. A
        `friendly_multi_slug` batch must carry 3..6 slugs all
        with `target_cohort = predictor_friendly` AND a
        `dispatch_family_signature` whose
        `verification_status = "verified"`. A
        `single_slug_isolated` batch must carry exactly one slug
        AND declare `isolation_reason`.
      * `isolation_reason` ∈
        `CORPUS_EXPANSION_BATCH_ISOLATION_REASONS` when
        `cadence_class = single_slug_isolated`; `None`
        otherwise.
    """

    schema_version: str
    batch_id: str
    max_slugs: int
    max_estimated_runtime_minutes: int
    rationale: str
    slugs: tuple[CorpusExpansionBatchSlugEntry, ...]
    family_stratification: Mapping[str, int]
    predictor_design_coverage_promotions: tuple[
        CorpusExpansionBatchPromotionEntry, ...
    ]
    dispatch_family_signature: str | None
    cadence_class: str | None
    isolation_reason: str | None


def _require_field(
    table: Mapping[str, object], field_name: str, context: str,
) -> object:
    if field_name not in table:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_field",
            f"{context} is missing required field {field_name!r}",
        )
    return table[field_name]


def _require_int(
    value: object, field_name: str, context: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_field",
            f"{context}.{field_name} must be an integer, got "
            f"{type(value).__name__}",
        )
    return value


def _require_str(
    value: object, field_name: str, context: str,
) -> str:
    if not isinstance(value, str):
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_field",
            f"{context}.{field_name} must be a string, got "
            f"{type(value).__name__}",
        )
    return value


def _parse_slug_entry(
    raw: object, index: int,
) -> CorpusExpansionBatchSlugEntry:
    if not isinstance(raw, Mapping):
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_field",
            f"[[slugs]] index {index} is not a table",
        )
    context = f"[[slugs]] index {index}"
    slug = _require_str(
        _require_field(raw, "slug", context), "slug", context,
    )
    target_cohort = _require_str(
        _require_field(raw, "target_cohort", context),
        "target_cohort", context,
    )
    if target_cohort not in CORPUS_EXPANSION_BATCH_TARGET_COHORTS:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_invalid_target_cohort",
            f"{context}.target_cohort = {target_cohort!r} is not "
            f"one of "
            f"{sorted(CORPUS_EXPANSION_BATCH_TARGET_COHORTS)}",
        )
    target_language = _require_str(
        _require_field(raw, "target_language", context),
        "target_language", context,
    )
    if target_language not in CORPUS_EXPANSION_BATCH_TARGET_LANGUAGES:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_invalid_target_language",
            f"{context}.target_language = {target_language!r} "
            f"is not one of "
            f"{sorted(CORPUS_EXPANSION_BATCH_TARGET_LANGUAGES)}",
        )
    runtime_raw = _require_field(
        raw, "estimated_runtime_minutes", context,
    )
    if isinstance(runtime_raw, bool) or not isinstance(
        runtime_raw, int,
    ):
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_invalid_runtime_estimate",
            f"{context}.estimated_runtime_minutes must be a "
            f"positive integer, got "
            f"{type(runtime_raw).__name__}",
        )
    if runtime_raw <= 0:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_invalid_runtime_estimate",
            f"{context}.estimated_runtime_minutes = "
            f"{runtime_raw} must be strictly positive",
        )
    return CorpusExpansionBatchSlugEntry(
        slug=slug,
        target_cohort=target_cohort,
        target_language=target_language,
        estimated_runtime_minutes=runtime_raw,
    )


def _parse_promotion_entry(
    raw: object, index: int,
) -> CorpusExpansionBatchPromotionEntry:
    if not isinstance(raw, Mapping):
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_field",
            f"[[predictor_design_coverage_promotions]] index "
            f"{index} is not a table",
        )
    context = (
        f"[[predictor_design_coverage_promotions]] index {index}"
    )
    slug = _require_str(
        _require_field(raw, "slug", context), "slug", context,
    )
    target_cohort = _require_str(
        _require_field(raw, "target_cohort", context),
        "target_cohort", context,
    )
    if target_cohort not in CORPUS_EXPANSION_BATCH_TARGET_COHORTS:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_invalid_target_cohort",
            f"{context}.target_cohort = {target_cohort!r} is not "
            f"one of "
            f"{sorted(CORPUS_EXPANSION_BATCH_TARGET_COHORTS)}",
        )
    return CorpusExpansionBatchPromotionEntry(
        slug=slug, target_cohort=target_cohort,
    )


def _validate_family_stratification(
    raw: object, slug_count: int,
) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_family_stratification_mismatch",
            f"family_stratification must be a mapping of "
            f"{{family: count}}, got {type(raw).__name__}",
        )
    cleaned: dict[str, int] = {}
    for family, count in raw.items():
        if family not in CORPUS_EXPANSION_BATCH_FAMILIES:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_invalid_family",
                f"family_stratification key {family!r} is not "
                f"one of {sorted(CORPUS_EXPANSION_BATCH_FAMILIES)}",
            )
        if isinstance(count, bool) or not isinstance(count, int):
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_family_stratification_mismatch",
                f"family_stratification.{family} must be an "
                f"integer, got {type(count).__name__}",
            )
        if count < 0:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_family_stratification_mismatch",
                f"family_stratification.{family} = {count} "
                f"must be non-negative",
            )
        cleaned[family] = count
    total = sum(cleaned.values())
    if total != slug_count:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_family_stratification_mismatch",
            f"family_stratification counts sum to {total} but "
            f"len(slugs) = {slug_count}",
        )
    return cleaned


def load_corpus_expansion_batch_manifest(
    path: Path,
    *,
    canonical_slug_set: Iterable[str] | None = None,
    host_arch_unsupported_slug_set: Iterable[str] | None = None,
    landed_slug_set: Iterable[str] | None = None,
    verified_dispatch_families: Mapping[str, str] | None = None,
) -> CorpusExpansionBatchManifest:
    """Load and validate a corpus-expansion batch manifest.

    The TOML at `path` must declare:
      * `schema_version` ∈
        `CORPUS_EXPANSION_BATCH_SUPPORTED_SCHEMA_VERSIONS`
        (`"1.0"` is accepted as a legacy path; `"1.1"` is the
        current schema and adds the cadence-uplift fields below).
      * `batch_id` (string, non-empty).
      * `max_slugs` (int, `0 < x <= 12`).
      * `max_estimated_runtime_minutes` (int, `0 < x <= 120`).
      * `rationale` (string).
      * `[[slugs]]` array of tables, each with `slug`,
        `target_cohort`, `target_language`,
        `estimated_runtime_minutes`.
      * `[family_stratification]` mapping
        `{L1, L2, Quant, FIB}` → non-negative int counts that
        sum to `len(slugs)`.
      * `[[predictor_design_coverage_promotions]]` array of
        tables, each with `slug` and `target_cohort`. The slug
        set MUST exactly equal the `[[slugs]]` slug set, and
        each promotion's `target_cohort` MUST equal the
        same slug's per-`[[slugs]]` `target_cohort`.

    `schema_version = "1.1"` ALSO requires:
      * `dispatch_family_signature` (string, non-empty;
        cross-referenced against
        `verified_dispatch_families` when supplied).
      * `cadence_class` ∈
        `CORPUS_EXPANSION_BATCH_CADENCE_CLASSES`.
      * `isolation_reason` ∈
        `CORPUS_EXPANSION_BATCH_ISOLATION_REASONS` when
        `cadence_class = single_slug_isolated`.

    `friendly_multi_slug` additionally enforces:
      * `len(slugs)` ∈ `[3, 6]`.
      * Every slug carries
        `target_cohort = predictor_friendly`.
      * `dispatch_family_signature` exists in
        `verified_dispatch_families` with `verification_status
        = "verified"` (when `verified_dispatch_families` is
        supplied).

    `single_slug_isolated` additionally enforces:
      * `len(slugs) == 1`.

    Optional `canonical_slug_set` rejects any slug absent from
    it with `corpus_expansion_batch_non_canonical_slug`.

    Optional `host_arch_unsupported_slug_set` rejects any slug
    in it with `corpus_expansion_batch_host_arch_unsupported_slug`
    (so batches cannot schedule sm_86-incompatible hand-ports).

    Optional `landed_slug_set` reconciles the actual on-disk
    landed slugs against the manifest's `[[slugs]]` slug set
    after a batch has executed. The set MUST exactly equal the
    manifest slug-set; missing or extra landed slugs raise
    `corpus_expansion_batch_landed_slug_mismatch`.

    Optional `verified_dispatch_families` maps
    `family_signature -> verification_status` (typically the
    `family_status_map` of a parsed
    `VerifiedDispatchFamiliesManifest`). When supplied, it
    gates the cadence-uplift checks on v1.1 manifests:
      * A manifest's `dispatch_family_signature` MUST appear in
        the mapping, else
        `corpus_expansion_batch_invalid_dispatch_family`.
      * A `friendly_multi_slug` cadence_class that cites a
        family with `verification_status = "pending_verification"`
        is rejected with
        `corpus_expansion_batch_cadence_uplift_pending_verification_dispatch_family`.

    Raises `CorpusExpansionBatchError` with a typed reason from
    `CORPUS_EXPANSION_BATCH_TYPED_REASONS` on the first
    violation encountered.
    """

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_io_error",
            f"cannot read batch manifest at {path!s}: {exc}",
        ) from exc
    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_invalid_toml",
            f"batch manifest at {path!s} is not valid TOML: "
            f"{exc}",
        ) from exc

    schema_version = _require_str(
        _require_field(data, "schema_version", "batch manifest"),
        "schema_version", "batch manifest",
    )
    if (
        schema_version
        not in CORPUS_EXPANSION_BATCH_SUPPORTED_SCHEMA_VERSIONS
    ):
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_invalid_schema_version",
            f"batch manifest schema_version = "
            f"{schema_version!r} is not one of "
            f"{sorted(CORPUS_EXPANSION_BATCH_SUPPORTED_SCHEMA_VERSIONS)}",
        )

    batch_id = _require_str(
        _require_field(data, "batch_id", "batch manifest"),
        "batch_id", "batch manifest",
    )
    if not batch_id:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_field",
            "batch manifest batch_id must be a non-empty string",
        )

    max_slugs = _require_int(
        _require_field(data, "max_slugs", "batch manifest"),
        "max_slugs", "batch manifest",
    )
    if max_slugs <= 0 or max_slugs > CORPUS_EXPANSION_BATCH_MAX_SLUGS:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_max_slugs_exceeded",
            f"batch manifest max_slugs = {max_slugs} must "
            f"satisfy 0 < max_slugs <= "
            f"{CORPUS_EXPANSION_BATCH_MAX_SLUGS}",
        )

    max_runtime = _require_int(
        _require_field(
            data, "max_estimated_runtime_minutes",
            "batch manifest",
        ),
        "max_estimated_runtime_minutes", "batch manifest",
    )
    if (
        max_runtime <= 0
        or max_runtime > CORPUS_EXPANSION_BATCH_MAX_RUNTIME_MINUTES
    ):
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_max_runtime_exceeded",
            f"batch manifest max_estimated_runtime_minutes = "
            f"{max_runtime} must satisfy 0 < x <= "
            f"{CORPUS_EXPANSION_BATCH_MAX_RUNTIME_MINUTES}",
        )

    rationale = _require_str(
        _require_field(data, "rationale", "batch manifest"),
        "rationale", "batch manifest",
    )

    slugs_raw = _require_field(data, "slugs", "batch manifest")
    if not isinstance(slugs_raw, list) or not slugs_raw:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_field",
            "batch manifest [[slugs]] must be a non-empty array "
            "of tables",
        )
    slug_entries = tuple(
        _parse_slug_entry(entry, index)
        for index, entry in enumerate(slugs_raw)
    )

    if len(slug_entries) > max_slugs:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_slug_count_mismatch",
            f"len(slugs) = {len(slug_entries)} exceeds "
            f"max_slugs = {max_slugs}",
        )

    seen_slugs: set[str] = set()
    for entry in slug_entries:
        if entry.slug in seen_slugs:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_duplicate_slug",
                f"slug {entry.slug!r} appears more than once in "
                f"the manifest's [[slugs]] array",
            )
        seen_slugs.add(entry.slug)

    runtime_sum = sum(
        entry.estimated_runtime_minutes for entry in slug_entries
    )
    if runtime_sum > max_runtime:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_runtime_sum_exceeded",
            f"sum of per-slug estimated_runtime_minutes = "
            f"{runtime_sum} exceeds max_estimated_runtime_minutes "
            f"= {max_runtime}",
        )

    stratification_raw = _require_field(
        data, "family_stratification", "batch manifest",
    )
    family_stratification = _validate_family_stratification(
        stratification_raw, len(slug_entries),
    )

    if canonical_slug_set is not None:
        canonical_set = frozenset(canonical_slug_set)
        for entry in slug_entries:
            if entry.slug not in canonical_set:
                raise CorpusExpansionBatchError(
                    "corpus_expansion_batch_non_canonical_slug",
                    f"slug {entry.slug!r} is not in the canonical "
                    f"SOL-ExecBench slug set",
                )

    if host_arch_unsupported_slug_set is not None:
        unsupported_set = frozenset(host_arch_unsupported_slug_set)
        for entry in slug_entries:
            if entry.slug in unsupported_set:
                raise CorpusExpansionBatchError(
                    "corpus_expansion_batch_host_arch_unsupported_slug",
                    f"slug {entry.slug!r} is marked "
                    f"host_arch_unsupported and cannot be "
                    f"scheduled in a corpus-expansion batch",
                )

    if "predictor_design_coverage_promotions" not in data:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_promotion_array",
            "batch manifest is missing the required "
            "[[predictor_design_coverage_promotions]] array; "
            "every in-batch slug must be predeclared with its "
            "cohort bucket before the batch lands",
        )
    promotions_raw = data["predictor_design_coverage_promotions"]
    if not isinstance(promotions_raw, list) or not promotions_raw:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_promotion_array",
            "[[predictor_design_coverage_promotions]] must be a "
            "non-empty array of tables",
        )
    promotion_entries = tuple(
        _parse_promotion_entry(entry, index)
        for index, entry in enumerate(promotions_raw)
    )
    promotion_slug_to_cohort: dict[str, str] = {}
    for entry in promotion_entries:
        if entry.slug in promotion_slug_to_cohort:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_duplicate_promotion",
                f"slug {entry.slug!r} appears more than once in "
                f"[[predictor_design_coverage_promotions]]",
            )
        promotion_slug_to_cohort[entry.slug] = entry.target_cohort

    manifest_slug_set = {entry.slug for entry in slug_entries}
    promotion_slug_set = set(promotion_slug_to_cohort.keys())
    if manifest_slug_set != promotion_slug_set:
        missing = sorted(manifest_slug_set - promotion_slug_set)
        extra = sorted(promotion_slug_set - manifest_slug_set)
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_promotion_slug_mismatch",
            f"[[predictor_design_coverage_promotions]] slug set "
            f"does not match [[slugs]] slug set: "
            f"missing_from_promotions={missing!r}, "
            f"extra_in_promotions={extra!r}",
        )
    for entry in slug_entries:
        promoted_cohort = promotion_slug_to_cohort[entry.slug]
        if promoted_cohort != entry.target_cohort:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_promotion_cohort_mismatch",
                f"slug {entry.slug!r}: "
                f"[[slugs]].target_cohort = "
                f"{entry.target_cohort!r} but "
                f"[[predictor_design_coverage_promotions]]."
                f"target_cohort = {promoted_cohort!r}",
            )

    if landed_slug_set is not None:
        landed = frozenset(landed_slug_set)
        if landed != manifest_slug_set:
            missing = sorted(manifest_slug_set - landed)
            extra = sorted(landed - manifest_slug_set)
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_landed_slug_mismatch",
                f"landed slug set does not match the manifest "
                f"[[slugs]] slug set: "
                f"missing_from_landed={missing!r}, "
                f"extra_in_landed={extra!r}",
            )

    dispatch_family_signature, cadence_class, isolation_reason = (
        _parse_cadence_uplift_fields(
            data=data,
            schema_version=schema_version,
            slug_entries=slug_entries,
            verified_dispatch_families=verified_dispatch_families,
        )
    )

    return CorpusExpansionBatchManifest(
        schema_version=schema_version,
        batch_id=batch_id,
        max_slugs=max_slugs,
        max_estimated_runtime_minutes=max_runtime,
        rationale=rationale,
        slugs=slug_entries,
        family_stratification=family_stratification,
        predictor_design_coverage_promotions=promotion_entries,
        dispatch_family_signature=dispatch_family_signature,
        cadence_class=cadence_class,
        isolation_reason=isolation_reason,
    )


def _parse_cadence_uplift_fields(
    *,
    data: Mapping[str, object],
    schema_version: str,
    slug_entries: tuple[CorpusExpansionBatchSlugEntry, ...],
    verified_dispatch_families: Mapping[str, str] | None,
) -> tuple[str | None, str | None, str | None]:
    """Parse + validate cadence-uplift fields per schema version.

    Returns `(dispatch_family_signature, cadence_class,
    isolation_reason)`. For legacy v1.0 manifests, the fields are
    optional; if absent the loader returns `(None, None, None)`.
    For v1.1 manifests, `dispatch_family_signature` and
    `cadence_class` are required; `isolation_reason` is required
    when `cadence_class = single_slug_isolated` and forbidden
    otherwise.
    """

    is_legacy = (
        schema_version == CORPUS_EXPANSION_BATCH_LEGACY_SCHEMA_VERSION
    )
    has_any_cadence_field = (
        "dispatch_family_signature" in data
        or "cadence_class" in data
        or "isolation_reason" in data
    )
    if is_legacy and not has_any_cadence_field:
        return (None, None, None)

    dispatch_family_signature = _require_str(
        _require_field(
            data, "dispatch_family_signature", "batch manifest",
        ),
        "dispatch_family_signature", "batch manifest",
    )
    if not dispatch_family_signature:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_missing_field",
            "batch manifest dispatch_family_signature must be a "
            "non-empty string",
        )

    cadence_class = _require_str(
        _require_field(data, "cadence_class", "batch manifest"),
        "cadence_class", "batch manifest",
    )
    if cadence_class not in CORPUS_EXPANSION_BATCH_CADENCE_CLASSES:
        raise CorpusExpansionBatchError(
            "corpus_expansion_batch_invalid_cadence_class",
            f"batch manifest cadence_class = {cadence_class!r} "
            f"is not one of "
            f"{sorted(CORPUS_EXPANSION_BATCH_CADENCE_CLASSES)}",
        )

    isolation_reason: str | None
    if cadence_class == CADENCE_CLASS_SINGLE_SLUG_ISOLATED:
        if "isolation_reason" not in data:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_single_slug_isolated_missing_isolation_reason",
                "batch manifest cadence_class = "
                "'single_slug_isolated' requires isolation_reason "
                "from "
                f"{sorted(CORPUS_EXPANSION_BATCH_ISOLATION_REASONS)}",
            )
        isolation_reason = _require_str(
            data["isolation_reason"],
            "isolation_reason", "batch manifest",
        )
        if (
            isolation_reason
            not in CORPUS_EXPANSION_BATCH_ISOLATION_REASONS
        ):
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_isolation_reason_outside_closed_vocabulary",
                f"batch manifest isolation_reason = "
                f"{isolation_reason!r} is not one of "
                f"{sorted(CORPUS_EXPANSION_BATCH_ISOLATION_REASONS)}",
            )
        if len(slug_entries) != 1:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_cadence_class_slug_count_mismatch",
                f"batch manifest cadence_class = "
                f"'single_slug_isolated' requires exactly one "
                f"[[slugs]] entry, got {len(slug_entries)}",
            )
    else:
        isolation_reason = None
        if "isolation_reason" in data:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_isolation_reason_outside_closed_vocabulary",
                f"batch manifest cadence_class = {cadence_class!r}"
                f" must not declare isolation_reason; that field is "
                f"only valid under cadence_class = "
                f"'single_slug_isolated'",
            )

    if cadence_class == CADENCE_CLASS_FRIENDLY_MULTI_SLUG:
        if (
            len(slug_entries) < FRIENDLY_MULTI_SLUG_MIN_SLUGS
            or len(slug_entries) > FRIENDLY_MULTI_SLUG_MAX_SLUGS
        ):
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_friendly_multi_slug_slug_count_out_of_range",
                f"batch manifest cadence_class = "
                f"'friendly_multi_slug' requires "
                f"{FRIENDLY_MULTI_SLUG_MIN_SLUGS}..{FRIENDLY_MULTI_SLUG_MAX_SLUGS} "
                f"[[slugs]] entries, got {len(slug_entries)}",
            )
        for entry in slug_entries:
            if (
                entry.target_cohort
                != TARGET_COHORT_PREDICTOR_FRIENDLY
            ):
                raise CorpusExpansionBatchError(
                    "corpus_expansion_batch_cadence_uplift_hostile_in_multi_slug_batch",
                    f"batch manifest cadence_class = "
                    f"'friendly_multi_slug' rejects slug "
                    f"{entry.slug!r} with target_cohort = "
                    f"{entry.target_cohort!r}; every slug must be "
                    f"{TARGET_COHORT_PREDICTOR_FRIENDLY!r}",
                )

    if verified_dispatch_families is not None:
        if dispatch_family_signature not in verified_dispatch_families:
            raise CorpusExpansionBatchError(
                "corpus_expansion_batch_invalid_dispatch_family",
                f"batch manifest dispatch_family_signature = "
                f"{dispatch_family_signature!r} is not present in "
                f"the verified-dispatch-families manifest",
            )
        family_status = verified_dispatch_families[
            dispatch_family_signature
        ]
        if cadence_class == CADENCE_CLASS_FRIENDLY_MULTI_SLUG:
            if family_status == "pending_verification":
                raise CorpusExpansionBatchError(
                    "corpus_expansion_batch_cadence_uplift_pending_verification_dispatch_family",
                    f"batch manifest cadence_class = "
                    f"'friendly_multi_slug' cites dispatch family "
                    f"{dispatch_family_signature!r} whose "
                    f"verification_status = 'pending_verification'"
                    f"; only verified families are eligible for "
                    f"'friendly_multi_slug' cadence",
                )
            if family_status != "verified":
                raise CorpusExpansionBatchError(
                    "corpus_expansion_batch_cadence_uplift_unverified_dispatch_family",
                    f"batch manifest cadence_class = "
                    f"'friendly_multi_slug' cites dispatch family "
                    f"{dispatch_family_signature!r} whose "
                    f"verification_status = {family_status!r}; "
                    f"only 'verified' is eligible for "
                    f"'friendly_multi_slug' cadence",
                )

    return (dispatch_family_signature, cadence_class, isolation_reason)


__all__ = [
    "CADENCE_CLASS_FRIENDLY_MULTI_SLUG",
    "CADENCE_CLASS_SINGLE_SLUG_ISOLATED",
    "CORPUS_EXPANSION_BATCH_CADENCE_CLASSES",
    "CORPUS_EXPANSION_BATCH_FAMILIES",
    "CORPUS_EXPANSION_BATCH_ISOLATION_REASONS",
    "CORPUS_EXPANSION_BATCH_LEGACY_SCHEMA_VERSION",
    "CORPUS_EXPANSION_BATCH_MAX_RUNTIME_MINUTES",
    "CORPUS_EXPANSION_BATCH_MAX_SLUGS",
    "CORPUS_EXPANSION_BATCH_SCHEMA_VERSION",
    "CORPUS_EXPANSION_BATCH_SUPPORTED_SCHEMA_VERSIONS",
    "CORPUS_EXPANSION_BATCH_TARGET_COHORTS",
    "CORPUS_EXPANSION_BATCH_TARGET_LANGUAGES",
    "CORPUS_EXPANSION_BATCH_TYPED_REASONS",
    "CorpusExpansionBatchError",
    "CorpusExpansionBatchManifest",
    "CorpusExpansionBatchPromotionEntry",
    "CorpusExpansionBatchSlugEntry",
    "FAMILY_FIB",
    "FAMILY_L1",
    "FAMILY_L2",
    "FAMILY_QUANT",
    "FRIENDLY_MULTI_SLUG_MAX_SLUGS",
    "FRIENDLY_MULTI_SLUG_MIN_SLUGS",
    "ISOLATION_REASON_HOSTILE_COHORT",
    "ISOLATION_REASON_NOVEL_DISPATCH_FAMILY",
    "ISOLATION_REASON_OPERATOR_OVERRIDE_DEBUG_ISOLATION",
    "ISOLATION_REASON_UNVERIFIED_DISPATCH_FAMILY",
    "ISOLATION_REASON_WARM_COLD_PINNING_CANDIDATE",
    "TARGET_COHORT_PREDICTOR_FRIENDLY",
    "TARGET_COHORT_PREDICTOR_HOSTILE",
    "TARGET_LANGUAGE_CUDA_CPP",
    "TARGET_LANGUAGE_CUTLASS",
    "load_corpus_expansion_batch_manifest",
]
