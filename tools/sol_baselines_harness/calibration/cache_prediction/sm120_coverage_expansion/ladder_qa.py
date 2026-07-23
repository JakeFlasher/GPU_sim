"""QA validator for ``sm120_coverage_readiness_ladder_v1.toml``.

Returns / raises typed-failure tokens from
``LADDER_QA_TYPED_FAILURES`` so planted-mutation negatives can
assert against the exact token rather than parsing free-text
messages.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_schema import (
    LADDER_HOST_ARCH,
    LADDER_SCHEMA_VERSION,
    LADDER_TOTAL_PROBLEMS,
    REQUIRED_META_FIELDS,
)
from baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_states import (
    LadderQaError,
    _LADDER_STATES,
)


def compute_corpus_sha256(slugs: list[str]) -> str:
    """Stable corpus SHA-256: ASCII-sort + newline-join + UTF-8.

    Used both by the emission script (stamped into the manifest's
    meta block) and by the QA validator (recomputed from the
    manifest's own entries to detect tampering).
    """
    sorted_slugs = sorted(slugs)
    payload = "\n".join(sorted_slugs).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_ladder_manifest(
    toml_path: Path,
    *,
    expected_corpus_slugs: list[str] | None = None,
) -> None:
    """Validate the on-disk ladder TOML; raise ``LadderQaError``
    with a typed token on the first violation.

    Args:
        toml_path: path to the ladder manifest.
        expected_corpus_slugs: when provided, the validator
            recomputes the corpus SHA over these slugs and asserts
            it matches the manifest's stamped ``corpus_sha256``.
            When ``None``, only structural checks run (count,
            order, vocabulary, duplicates).

    Raises:
        LadderQaError(token=...): on the first detected violation.
            Token vocabulary lives in ``LADDER_QA_TYPED_FAILURES``.
    """
    data = tomllib.loads(toml_path.read_text("utf-8"))
    meta = data.get("meta", {})
    _validate_meta_block(meta)
    entries = data.get("entries", [])
    _validate_entries(entries)
    if expected_corpus_slugs is not None:
        _validate_corpus_sha(meta, expected_corpus_slugs)


def _validate_meta_block(meta: dict[str, Any]) -> None:
    for field_name in REQUIRED_META_FIELDS:
        if field_name not in meta:
            raise LadderQaError(
                "ladder_meta_field_missing",
                f"meta block is missing required field {field_name!r}",
            )
    if meta["schema_version"] != LADDER_SCHEMA_VERSION:
        raise LadderQaError(
            "ladder_meta_field_invalid",
            f"meta.schema_version={meta['schema_version']!r}; "
            f"expected {LADDER_SCHEMA_VERSION!r}",
        )
    if meta["host_arch"] != LADDER_HOST_ARCH:
        raise LadderQaError(
            "ladder_meta_field_invalid",
            f"meta.host_arch={meta['host_arch']!r}; "
            f"expected {LADDER_HOST_ARCH!r}",
        )
    if meta["total_problems"] != LADDER_TOTAL_PROBLEMS:
        raise LadderQaError(
            "ladder_meta_field_invalid",
            f"meta.total_problems={meta['total_problems']!r}; "
            f"expected {LADDER_TOTAL_PROBLEMS}",
        )


def _validate_entries(entries: list[dict[str, Any]]) -> None:
    if len(entries) != LADDER_TOTAL_PROBLEMS:
        raise LadderQaError(
            "ladder_corpus_size_mismatch",
            f"ladder has {len(entries)} entries; "
            f"expected {LADDER_TOTAL_PROBLEMS}",
        )
    seen: set[str] = set()
    prev_slug: str | None = None
    for entry in entries:
        slug = entry.get("slug")
        state = entry.get("state")
        if not isinstance(slug, str):
            raise LadderQaError(
                "ladder_meta_field_invalid",
                f"entry has non-string slug: {entry!r}",
            )
        if not isinstance(state, str):
            raise LadderQaError(
                "ladder_meta_field_invalid",
                f"entry {slug!r} has non-string state: {entry!r}",
            )
        if slug in seen:
            raise LadderQaError(
                "ladder_duplicate_problem",
                f"slug {slug!r} appears more than once",
            )
        seen.add(slug)
        if state not in _LADDER_STATES:
            raise LadderQaError(
                "ladder_state_out_of_vocabulary",
                f"entry {slug!r} state={state!r} is not in _LADDER_STATES; "
                f"valid members are {sorted(_LADDER_STATES)}",
            )
        if prev_slug is not None and slug < prev_slug:
            raise LadderQaError(
                "ladder_ordering_violation",
                f"slug {slug!r} follows {prev_slug!r} but is "
                "ASCII-less than the predecessor; entries must be "
                "sorted ASCII by slug",
            )
        prev_slug = slug


def _validate_corpus_sha(
    meta: dict[str, Any], expected_corpus_slugs: list[str]
) -> None:
    expected = compute_corpus_sha256(expected_corpus_slugs)
    stamped = meta.get("corpus_sha256")
    if stamped != expected:
        raise LadderQaError(
            "ladder_corpus_sha256_mismatch",
            f"meta.corpus_sha256={stamped!r}; "
            f"recomputed over the expected {len(expected_corpus_slugs)}"
            f" slugs = {expected!r}",
        )
