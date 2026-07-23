"""Verified-dispatch-families manifest loader.

A corpus-expansion batch manifest declaring
`cadence_class = "friendly_multi_slug"` MUST cite a
`dispatch_family_signature` that appears in
`.humanize/corpus-expansion/verified_dispatch_families.toml` with
`verification_status = "verified"`. The loader exposes a
`load_verified_dispatch_families(path)` API returning a
`VerifiedDispatchFamiliesManifest` dataclass plus the canonical
mapping `{family_signature: verification_status}` that the
`corpus_expansion_batch` loader consumes for cadence-uplift gating.

This module is calibration-only and never imported by the deployed
harness path; the NCU quarantine scanner enforces that contract.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


VERIFIED_DISPATCH_FAMILIES_SCHEMA_VERSION: str = "1.0"

VERIFICATION_STATUS_VERIFIED: str = "verified"
VERIFICATION_STATUS_PENDING: str = "pending_verification"

VERIFIED_DISPATCH_FAMILIES_VERIFICATION_STATUSES: frozenset[str] = frozenset({
    VERIFICATION_STATUS_VERIFIED,
    VERIFICATION_STATUS_PENDING,
})


VERIFIED_DISPATCH_FAMILIES_TYPED_REASONS: frozenset[str] = frozenset({
    "verified_dispatch_families_io_error",
    "verified_dispatch_families_invalid_toml",
    "verified_dispatch_families_invalid_schema_version",
    "verified_dispatch_families_missing_field",
    "verified_dispatch_families_duplicate_family_signature",
    "verified_dispatch_families_invalid_verification_status",
    "verified_dispatch_families_empty_field",
})


class VerifiedDispatchFamiliesError(Exception):
    """Typed error raised by the verified-dispatch-families loader.

    `typed_reason` is drawn from the closed set
    `VERIFIED_DISPATCH_FAMILIES_TYPED_REASONS`.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in VERIFIED_DISPATCH_FAMILIES_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of "
                f"{sorted(VERIFIED_DISPATCH_FAMILIES_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class VerifiedDispatchFamilyEntry:
    """One verified-dispatch-family entry.

    Fields:
      * `family_signature` — closed-vocabulary string used by
        corpus-expansion batch manifests to claim a dispatch
        surface (e.g. `cublasSgemm_basic_fp32`).
      * `first_verified_slug` — canonical SOL-ExecBench slug that
        first surfaced the dispatch via cuBLAS-logger or cohort-B
        probe evidence.
      * `verification_evidence` — free-text citation explaining how
        the dispatch was verified.
      * `verification_status` ∈
        `VERIFIED_DISPATCH_FAMILIES_VERIFICATION_STATUSES`. Only
        `verified` families are eligible for `friendly_multi_slug`
        cadence; `pending_verification` families may appear in
        `single_slug_isolated` batches only.
    """

    family_signature: str
    first_verified_slug: str
    verification_evidence: str
    verification_status: str


@dataclass(frozen=True)
class VerifiedDispatchFamiliesManifest:
    """Parsed verified-dispatch-families manifest.

    `family_status_map` is a `{family_signature:
    verification_status}` dict the corpus-expansion batch loader
    consumes directly for cadence-uplift gating; the dataclass also
    retains the full `families` tuple for evidence inspection.
    """

    schema_version: str
    families: tuple[VerifiedDispatchFamilyEntry, ...]
    family_status_map: Mapping[str, str]


def _require_field(
    table: Mapping[str, object], field_name: str, context: str,
) -> object:
    if field_name not in table:
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_missing_field",
            f"{context} is missing required field {field_name!r}",
        )
    return table[field_name]


def _require_nonempty_str(
    value: object, field_name: str, context: str,
) -> str:
    if not isinstance(value, str):
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_missing_field",
            f"{context}.{field_name} must be a string, got "
            f"{type(value).__name__}",
        )
    if not value:
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_empty_field",
            f"{context}.{field_name} must be a non-empty string",
        )
    return value


def _parse_family_entry(
    raw: object, index: int,
) -> VerifiedDispatchFamilyEntry:
    if not isinstance(raw, Mapping):
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_missing_field",
            f"[[families]] index {index} is not a table",
        )
    context = f"[[families]] index {index}"
    family_signature = _require_nonempty_str(
        _require_field(raw, "family_signature", context),
        "family_signature", context,
    )
    first_verified_slug = _require_nonempty_str(
        _require_field(raw, "first_verified_slug", context),
        "first_verified_slug", context,
    )
    verification_evidence = _require_nonempty_str(
        _require_field(raw, "verification_evidence", context),
        "verification_evidence", context,
    )
    verification_status = _require_nonempty_str(
        _require_field(raw, "verification_status", context),
        "verification_status", context,
    )
    if (
        verification_status
        not in VERIFIED_DISPATCH_FAMILIES_VERIFICATION_STATUSES
    ):
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_invalid_verification_status",
            f"{context}.verification_status = "
            f"{verification_status!r} is not one of "
            f"{sorted(VERIFIED_DISPATCH_FAMILIES_VERIFICATION_STATUSES)}",
        )
    return VerifiedDispatchFamilyEntry(
        family_signature=family_signature,
        first_verified_slug=first_verified_slug,
        verification_evidence=verification_evidence,
        verification_status=verification_status,
    )


def load_verified_dispatch_families(
    path: Path,
) -> VerifiedDispatchFamiliesManifest:
    """Load and validate the verified-dispatch-families manifest.

    The TOML at `path` must declare:
      * `schema_version = "1.0"`.
      * `[[families]]` array of tables, each with
        `family_signature`, `first_verified_slug`,
        `verification_evidence`, and `verification_status` (closed
        vocabulary: `verified` or `pending_verification`).

    Raises `VerifiedDispatchFamiliesError` with a typed reason
    drawn from `VERIFIED_DISPATCH_FAMILIES_TYPED_REASONS`.
    """

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_io_error",
            f"cannot read manifest at {path!s}: {exc}",
        ) from exc
    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_invalid_toml",
            f"manifest at {path!s} is not valid TOML: {exc}",
        ) from exc

    schema_version_raw = _require_field(
        data, "schema_version", "manifest",
    )
    schema_version = _require_nonempty_str(
        schema_version_raw, "schema_version", "manifest",
    )
    if schema_version != VERIFIED_DISPATCH_FAMILIES_SCHEMA_VERSION:
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_invalid_schema_version",
            f"manifest schema_version = {schema_version!r} is not "
            f"{VERIFIED_DISPATCH_FAMILIES_SCHEMA_VERSION!r}",
        )

    families_raw = _require_field(data, "families", "manifest")
    if not isinstance(families_raw, list) or not families_raw:
        raise VerifiedDispatchFamiliesError(
            "verified_dispatch_families_missing_field",
            "manifest [[families]] must be a non-empty array of "
            "tables",
        )
    entries = tuple(
        _parse_family_entry(entry, index)
        for index, entry in enumerate(families_raw)
    )

    family_status_map: dict[str, str] = {}
    for entry in entries:
        if entry.family_signature in family_status_map:
            raise VerifiedDispatchFamiliesError(
                "verified_dispatch_families_duplicate_family_signature",
                f"family_signature {entry.family_signature!r} "
                f"appears more than once in the manifest",
            )
        family_status_map[entry.family_signature] = (
            entry.verification_status
        )

    return VerifiedDispatchFamiliesManifest(
        schema_version=schema_version,
        families=entries,
        family_status_map=family_status_map,
    )


__all__ = [
    "VERIFICATION_STATUS_PENDING",
    "VERIFICATION_STATUS_VERIFIED",
    "VERIFIED_DISPATCH_FAMILIES_SCHEMA_VERSION",
    "VERIFIED_DISPATCH_FAMILIES_TYPED_REASONS",
    "VERIFIED_DISPATCH_FAMILIES_VERIFICATION_STATUSES",
    "VerifiedDispatchFamiliesError",
    "VerifiedDispatchFamiliesManifest",
    "VerifiedDispatchFamilyEntry",
    "load_verified_dispatch_families",
]
