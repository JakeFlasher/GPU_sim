"""Calibration / validation split manifest schema, loader, and
typed-error vocabulary.

The split manifest (`calibration_validation_split_v1.toml`)
separates the calibrator's training input from the validation
holdout. Per-slug `split` ∈ {`calibration_only`,
`validation_only`, `quarantine`}; new corpus additions default
to `validation_only` until explicitly promoted by a
`[[promotion_log]]` entry carrying `reason`, `author`, and
`date_utc`.

This module is calibration-only. Downstream consumers:
  * `IsotonicCalibrator.fit()` consumes the split as a filter
    on training input (validation_only and quarantine slugs
    are rejected with `calibration_input_contains_validation_only_slug`).
  * The accuracy diagnostic reports train and validation
    rel-err independently using the split as the partition.

The deployed harness path does NOT import this module; the
NCU quarantine scanner enforces that contract.
"""

from __future__ import annotations

import datetime as _dt
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping


SPLIT_CALIBRATION_ONLY: str = "calibration_only"
SPLIT_VALIDATION_ONLY: str = "validation_only"
SPLIT_QUARANTINE: str = "quarantine"

CALIBRATION_VALIDATION_SPLIT_VALUES: frozenset[str] = frozenset({
    SPLIT_CALIBRATION_ONLY,
    SPLIT_VALIDATION_ONLY,
    SPLIT_QUARANTINE,
})

CALIBRATION_VALIDATION_SPLIT_SCHEMA_VERSION: int = 1


CALIBRATION_VALIDATION_SPLIT_TYPED_REASONS: frozenset[str] = frozenset({
    "calibration_validation_split_io_error",
    "calibration_validation_split_invalid_toml",
    "calibration_validation_split_invalid_schema_version",
    "calibration_validation_split_missing_field",
    "calibration_validation_split_invalid_split_value",
    "calibration_validation_split_duplicate_slug",
    "calibration_validation_split_promotion_log_missing_field",
    "calibration_validation_split_promotion_log_invalid_value",
    "calibration_validation_split_future_dated_audit_timestamp",
    "calibration_validation_split_calibration_only_missing_promotion",
    "calibration_validation_split_promotion_log_orphan_slug",
    "calibration_validation_split_canonical_slug_missing",
    "calibration_validation_split_canonical_slug_extra",
    "calibration_validation_split_bootstrap_slug_invalid",
})


_BOOTSTRAP_SLUG_PREFIX: str = "_bootstrap_"
# Comfortable +1 day margin on future-date checks to accommodate
# clock skew between the manifest authoring host and the loader
# host. Anything more than one day past the loader's clock is
# rejected as future-dated.
_FUTURE_DATE_SKEW_MARGIN_SECONDS: int = 24 * 60 * 60


class CalibrationValidationSplitError(Exception):
    """Typed error raised by the split manifest loader.

    Mirrors the typed-error pattern used throughout the
    calibration package. `typed_reason` is drawn from the
    closed set `CALIBRATION_VALIDATION_SPLIT_TYPED_REASONS`.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in CALIBRATION_VALIDATION_SPLIT_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; "
                f"expected one of "
                f"{sorted(CALIBRATION_VALIDATION_SPLIT_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class CalibrationValidationSplitPromotion:
    """A single `[[promotion_log]]` row recording a split
    transition for one slug (or the bootstrap initial-state
    entry, whose slug is a sentinel like
    `_bootstrap_calibration_corpus_v1`).
    """

    slug: str
    from_split: str
    to_split: str
    reason: str
    author: str
    date_utc: str


@dataclass(frozen=True)
class CalibrationValidationSplit:
    """Parsed split manifest. `by_slug` maps every slug to its
    split assignment; `promotion_log` carries the audit trail;
    `bootstrap_calibration_slugs` is the operator-declared
    allow-list of calibration_only slugs that pre-date the
    per-slug promotion-log discipline (typically the initial
    calibration corpus seed).
    """

    schema_version: int
    created_at_utc: str
    updated_at_utc: str
    description: str
    by_slug: Mapping[str, str] = field(default_factory=dict)
    promotion_log: tuple[CalibrationValidationSplitPromotion, ...] = ()
    bootstrap_calibration_slugs: frozenset[str] = field(
        default_factory=frozenset,
    )

    def split_for(self, slug: str) -> str | None:
        """Return the split assignment for `slug`, or None
        when the slug is absent from the manifest. Callers
        that want strict behavior (e.g., the IsotonicCalibrator
        fit guard) should treat None as `validation_only` (the
        documented default for new corpus additions) OR fail
        closed depending on the caller's contract.
        """
        return self.by_slug.get(slug)


def _parse_utc_timestamp(value: str, context: str) -> _dt.datetime:
    """Parse an ISO-8601 UTC timestamp string. Accepts the
    canonical `YYYY-MM-DDTHH:MM:SS+00:00` shape used by the
    manifest. Raises a typed error on parse failure.
    """
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_future_dated_audit_timestamp",
            f"{context}: timestamp {value!r} is not a valid "
            f"ISO-8601 UTC string ({exc})",
        ) from exc
    if parsed.tzinfo is None:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_future_dated_audit_timestamp",
            f"{context}: timestamp {value!r} carries no "
            f"timezone; must be an explicit UTC offset like "
            f"'+00:00'",
        )
    return parsed


def _reject_future_timestamp(
    value: str,
    context: str,
    *,
    now_provider: Callable[[], _dt.datetime] | None = None,
) -> None:
    """Reject `value` when it is more than +1 day ahead of the
    loader's clock. The +1 day margin tolerates clock skew
    between the authoring host and the loader host. Future-
    dated audit timestamps would defeat the manifest's audit-trail
    contract (the manifest is supposed to be a historical
    record, not a forward-looking schedule).
    """
    parsed = _parse_utc_timestamp(value, context)
    if now_provider is None:
        now = _dt.datetime.now(tz=_dt.timezone.utc)
    else:
        now = now_provider()
        if now.tzinfo is None:
            now = now.replace(tzinfo=_dt.timezone.utc)
    if (parsed - now).total_seconds() > _FUTURE_DATE_SKEW_MARGIN_SECONDS:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_future_dated_audit_timestamp",
            f"{context}: timestamp {value!r} is more than +1 "
            f"day past the loader's clock ({now.isoformat()}); "
            f"the manifest is an audit trail, not a forward-"
            f"looking schedule",
        )


def _require_str(
    row: Mapping, key: str, context: str,
) -> str:
    if key not in row:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_missing_field",
            f"{context} is missing required field {key!r}",
        )
    value = row[key]
    if not isinstance(value, str):
        raise CalibrationValidationSplitError(
            "calibration_validation_split_promotion_log_invalid_value",
            f"{context}: field {key!r} must be a string; got "
            f"{type(value).__name__}",
        )
    if not value:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_missing_field",
            f"{context}: field {key!r} is empty",
        )
    return value


def load_calibration_validation_split_manifest(
    manifest_path: Path,
    *,
    expected_slugs: frozenset[str] | None = None,
    now_provider: Callable[[], _dt.datetime] | None = None,
) -> CalibrationValidationSplit:
    """Read the TOML manifest at `manifest_path` and return
    the parsed `CalibrationValidationSplit` dataclass.

    Strict validation:
      * I/O errors raise `calibration_validation_split_io_error`.
      * Malformed TOML raises
        `calibration_validation_split_invalid_toml`.
      * `[manifest_meta].schema_version != 1` raises
        `calibration_validation_split_invalid_schema_version`.
      * Audit timestamps (`created_at_utc`, `updated_at_utc`,
        every `[[promotion_log]].date_utc`) more than +1 day
        ahead of the loader's clock raise
        `calibration_validation_split_future_dated_audit_timestamp`.
        Non-ISO-8601-with-explicit-timezone strings also fail
        under this typed reason.
      * Any `[[splits]]` row missing `slug` or `split` raises
        `calibration_validation_split_missing_field`.
      * Any `[[splits]]` row whose `split` is not in
        `CALIBRATION_VALIDATION_SPLIT_VALUES` raises
        `calibration_validation_split_invalid_split_value`.
      * A slug appearing in more than one `[[splits]]` row
        raises `calibration_validation_split_duplicate_slug`.
      * Any `[[promotion_log]]` row missing `slug`,
        `from_split`, `to_split`, `reason`, `author`, or
        `date_utc` raises
        `calibration_validation_split_promotion_log_missing_field`.
      * Any `[[promotion_log]]` row whose `from_split` or
        `to_split` is not in the closed split vocabulary
        raises
        `calibration_validation_split_promotion_log_invalid_value`.
      * A `[[promotion_log]]` row whose `slug` (after stripping
        the optional `_bootstrap_` prefix that marks
        non-per-slug audit rows) is not in `[[splits]]` is an
        orphan and raises
        `calibration_validation_split_promotion_log_orphan_slug`.
      * Any `calibration_only` slug that is NOT in
        `[manifest_meta].bootstrap_calibration_slugs` AND has
        no matching `[[promotion_log]]` entry with
        `to_split = "calibration_only"` raises
        `calibration_validation_split_calibration_only_missing_promotion`.
      * Any slug in `bootstrap_calibration_slugs` that is not
        marked `calibration_only` in `[[splits]]` raises
        `calibration_validation_split_bootstrap_slug_invalid`.

    `expected_slugs` (optional) enforces canonical completeness:
      * Any slug in `expected_slugs` missing from `[[splits]]`
        raises `calibration_validation_split_canonical_slug_missing`.
      * Any `[[splits]]` slug not in `expected_slugs` raises
        `calibration_validation_split_canonical_slug_extra`.

    `now_provider` injects a clock for tests; production calls
    use UTC now.
    """
    try:
        raw = manifest_path.read_bytes()
    except (OSError, FileNotFoundError) as exc:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_io_error",
            f"cannot read manifest at {str(manifest_path)!r}: "
            f"{exc}",
        ) from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_invalid_toml",
            f"manifest at {str(manifest_path)!r} is not valid "
            f"TOML: {exc}",
        ) from exc

    meta = data.get("manifest_meta", {})
    schema_version = meta.get("schema_version")
    if schema_version != CALIBRATION_VALIDATION_SPLIT_SCHEMA_VERSION:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_invalid_schema_version",
            f"manifest at {str(manifest_path)!r} must declare "
            f"[manifest_meta].schema_version = "
            f"{CALIBRATION_VALIDATION_SPLIT_SCHEMA_VERSION}; got "
            f"{schema_version!r}",
        )
    created_at_utc = _require_str(
        meta, "created_at_utc", "[manifest_meta]",
    )
    updated_at_utc = _require_str(
        meta, "updated_at_utc", "[manifest_meta]",
    )
    description = _require_str(
        meta, "description", "[manifest_meta]",
    )
    _reject_future_timestamp(
        created_at_utc,
        "[manifest_meta].created_at_utc",
        now_provider=now_provider,
    )
    _reject_future_timestamp(
        updated_at_utc,
        "[manifest_meta].updated_at_utc",
        now_provider=now_provider,
    )

    bootstrap_raw = meta.get("bootstrap_calibration_slugs", ())
    if not isinstance(bootstrap_raw, (list, tuple)):
        raise CalibrationValidationSplitError(
            "calibration_validation_split_missing_field",
            f"[manifest_meta].bootstrap_calibration_slugs "
            f"must be a TOML array of strings; got "
            f"{type(bootstrap_raw).__name__}",
        )
    bootstrap_set: set[str] = set()
    for index, entry in enumerate(bootstrap_raw):
        if not isinstance(entry, str) or not entry:
            raise CalibrationValidationSplitError(
                "calibration_validation_split_bootstrap_slug_invalid",
                f"[manifest_meta].bootstrap_calibration_slugs "
                f"index {index}: entry must be a non-empty "
                f"string; got {entry!r}",
            )
        bootstrap_set.add(entry)

    splits_raw = data.get("splits", ())
    by_slug: dict[str, str] = {}
    for index, row in enumerate(splits_raw):
        context = f"[[splits]] index {index}"
        if not isinstance(row, dict):
            raise CalibrationValidationSplitError(
                "calibration_validation_split_missing_field",
                f"{context}: row is not a table",
            )
        slug = _require_str(row, "slug", context)
        split = _require_str(row, "split", context)
        if split not in CALIBRATION_VALIDATION_SPLIT_VALUES:
            raise CalibrationValidationSplitError(
                "calibration_validation_split_invalid_split_value",
                f"{context}: split {split!r} is not in the "
                f"closed vocabulary "
                f"{sorted(CALIBRATION_VALIDATION_SPLIT_VALUES)}",
            )
        if slug in by_slug:
            raise CalibrationValidationSplitError(
                "calibration_validation_split_duplicate_slug",
                f"{context}: slug {slug!r} already appeared "
                f"in an earlier [[splits]] row; each slug "
                f"must appear exactly once",
            )
        by_slug[slug] = split

    promotion_raw = data.get("promotion_log", ())
    promotion_log: list[CalibrationValidationSplitPromotion] = []
    for index, row in enumerate(promotion_raw):
        context = f"[[promotion_log]] index {index}"
        if not isinstance(row, dict):
            raise CalibrationValidationSplitError(
                "calibration_validation_split_promotion_log_missing_field",
                f"{context}: row is not a table",
            )
        try:
            slug = _require_str(row, "slug", context)
            from_split = _require_str(row, "from_split", context)
            to_split = _require_str(row, "to_split", context)
            reason = _require_str(row, "reason", context)
            author = _require_str(row, "author", context)
            date_utc = _require_str(row, "date_utc", context)
        except CalibrationValidationSplitError as exc:
            if exc.typed_reason == "calibration_validation_split_missing_field":
                raise CalibrationValidationSplitError(
                    "calibration_validation_split_promotion_log_missing_field",
                    exc.message,
                ) from exc
            raise
        for split_field, value in (
            ("from_split", from_split),
            ("to_split", to_split),
        ):
            if value not in CALIBRATION_VALIDATION_SPLIT_VALUES:
                raise CalibrationValidationSplitError(
                    "calibration_validation_split_promotion_log_invalid_value",
                    f"{context}: {split_field} {value!r} is "
                    f"not in the closed vocabulary "
                    f"{sorted(CALIBRATION_VALIDATION_SPLIT_VALUES)}",
                )
        _reject_future_timestamp(
            date_utc,
            f"{context}.date_utc",
            now_provider=now_provider,
        )
        promotion_log.append(CalibrationValidationSplitPromotion(
            slug=slug,
            from_split=from_split,
            to_split=to_split,
            reason=reason,
            author=author,
            date_utc=date_utc,
        ))

    # Validate promotion-log orphan rejection: every per-slug
    # promotion row (those whose slug does NOT start with
    # `_bootstrap_`) must reference a slug present in
    # [[splits]]. Bootstrap rows are documentation-only and
    # exempt from the orphan check.
    orphan_promotions: list[str] = []
    for entry in promotion_log:
        if entry.slug.startswith(_BOOTSTRAP_SLUG_PREFIX):
            continue
        if entry.slug not in by_slug:
            orphan_promotions.append(entry.slug)
    if orphan_promotions:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_promotion_log_orphan_slug",
            f"[[promotion_log]] contains {len(orphan_promotions)} "
            f"per-slug entry(ies) whose slug is absent from "
            f"[[splits]]: {sorted(set(orphan_promotions))!r}. "
            f"Add a [[splits]] row for each promoted slug or "
            f"remove the orphan promotion entries.",
        )

    # Validate bootstrap allow-list: every slug in
    # bootstrap_calibration_slugs MUST be marked
    # calibration_only in [[splits]].
    bootstrap_misclassified: list[str] = []
    for slug in sorted(bootstrap_set):
        if by_slug.get(slug) != SPLIT_CALIBRATION_ONLY:
            bootstrap_misclassified.append(slug)
    if bootstrap_misclassified:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_bootstrap_slug_invalid",
            f"[manifest_meta].bootstrap_calibration_slugs lists "
            f"{len(bootstrap_misclassified)} slug(s) that are "
            f"NOT marked `calibration_only` in [[splits]]: "
            f"{bootstrap_misclassified!r}. Either remove the "
            f"slug from bootstrap_calibration_slugs or correct "
            f"its [[splits]].split value.",
        )

    # Validate every calibration_only slug is either in the
    # bootstrap allow-list OR has a matching [[promotion_log]]
    # entry with to_split = calibration_only.
    promotion_targets: dict[str, list[CalibrationValidationSplitPromotion]] = {}
    for entry in promotion_log:
        if entry.slug.startswith(_BOOTSTRAP_SLUG_PREFIX):
            continue
        promotion_targets.setdefault(entry.slug, []).append(entry)
    missing_promotion: list[str] = []
    for slug, split in by_slug.items():
        if split != SPLIT_CALIBRATION_ONLY:
            continue
        if slug in bootstrap_set:
            continue
        promotions = promotion_targets.get(slug, ())
        if not any(
            p.to_split == SPLIT_CALIBRATION_ONLY
            for p in promotions
        ):
            missing_promotion.append(slug)
    if missing_promotion:
        raise CalibrationValidationSplitError(
            "calibration_validation_split_calibration_only_missing_promotion",
            f"{len(missing_promotion)} calibration_only "
            f"slug(s) lack a matching [[promotion_log]] entry "
            f"with `to_split = \"calibration_only\"` AND are "
            f"not enumerated in "
            f"[manifest_meta].bootstrap_calibration_slugs: "
            f"{sorted(missing_promotion)[:5]!r}...",
        )

    # Optional canonical-completeness gate.
    if expected_slugs is not None:
        manifest_set = set(by_slug.keys())
        missing_canonical = sorted(expected_slugs - manifest_set)
        extra_canonical = sorted(manifest_set - expected_slugs)
        if missing_canonical:
            raise CalibrationValidationSplitError(
                "calibration_validation_split_canonical_slug_missing",
                f"{len(missing_canonical)} expected canonical "
                f"slug(s) missing from [[splits]]: "
                f"{missing_canonical[:5]!r}...",
            )
        if extra_canonical:
            raise CalibrationValidationSplitError(
                "calibration_validation_split_canonical_slug_extra",
                f"{len(extra_canonical)} unexpected slug(s) "
                f"present in [[splits]] but not in the "
                f"canonical expected set: "
                f"{extra_canonical[:5]!r}...",
            )

    return CalibrationValidationSplit(
        schema_version=schema_version,
        created_at_utc=created_at_utc,
        updated_at_utc=updated_at_utc,
        description=description,
        by_slug=dict(by_slug),
        promotion_log=tuple(promotion_log),
        bootstrap_calibration_slugs=frozenset(bootstrap_set),
    )


__all__ = [
    "CALIBRATION_VALIDATION_SPLIT_SCHEMA_VERSION",
    "CALIBRATION_VALIDATION_SPLIT_TYPED_REASONS",
    "CALIBRATION_VALIDATION_SPLIT_VALUES",
    "CalibrationValidationSplit",
    "CalibrationValidationSplitError",
    "CalibrationValidationSplitPromotion",
    "SPLIT_CALIBRATION_ONLY",
    "SPLIT_QUARANTINE",
    "SPLIT_VALIDATION_ONLY",
    "load_calibration_validation_split_manifest",
]
