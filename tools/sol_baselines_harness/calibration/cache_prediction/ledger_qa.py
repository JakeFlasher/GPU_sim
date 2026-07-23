"""Production QA validator for the sm_120 NCU label ledger.

Provides ``validate_sm120_ledger(ledger_path, *, sm86_required_path)``
that fails closed with closed-vocabulary typed reasons on:

* Schema malformation (missing required field, duplicate cell name).
* Missing required cell from the sm_86-required cohort (every cell
  in the sm_86 ledger must have a matching sm_120 entry — measured
  or typed-unavailable).
* Store-conflated metric family in any per-entry metric field.

The validator is invoked by taskF1's planted-negative-control tests
with synthetic mutations.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


LEDGER_QA_TYPED_REASONS: frozenset = frozenset({
    "sm120_required_cell_missing",
    "sm120_ledger_duplicate_cell_name",
    "sm120_ledger_missing_required_meta",
    "ledger_metric_family_invalid",
    "sm120_ledger_typed_unavailable_reason_invalid",
    "sm120_ledger_toml_malformed",
})


class LedgerQaError(Exception):
    """Raised by ``validate_sm120_ledger`` on any failure."""

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in LEDGER_QA_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of {sorted(LEDGER_QA_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


_FORBIDDEN_METRIC_SHAPES: tuple = (
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "syslts__t_sector_hit_rate.pct",
)


_REQUIRED_META_FIELDS: tuple = (
    "schema_version",
    "host_arch",
    "total_entries",
    "sm86_required_cell_count",
)


# Closed vocabulary of valid `unavailable_reason` values for
# sm_120 ledger rows. Anchored to the partition + plan
# vocabulary.
#
# an earlier iteration / Codex review blocking #2: the placeholder
# `sm120_binary_compilation_pending` was removed — it is NOT a
# terminal typed-unavailable reason. It described an in-progress
# build state that should be resolved before the row lands in the
# ledger. Any committed ledger entry must carry one of the
# structural reasons below; the QA validator rejects the placeholder
# fail-closed via the planted-mutation negative.
_VALID_TYPED_UNAVAILABLE_REASONS: frozenset = frozenset({
    "hostile_subclass_uncalibrated",
    "shared_memory_no_read_only_sector_traffic",
    "local_memory_private_cache_policy",
    "warm_cold_protocol_unpinned",
    "sm120_repeatability_threshold_exceeded",
    "sm120_repeatability_relative_cv_exceeded",
    "sm120_repeatability_absolute_stddev_exceeded",
    "sm120_l2_geometry_unverified",
    "sm120_fp8_block_scale_metric_unverified",
    "sm120_nvfp4_unsupported_consumer_arch",
    "sm120_tma_metric_family_unverified",
})


def _scan_for_forbidden_metric_family(entry: dict) -> str | None:
    """Walk the entry's string-typed fields (top-level + nested
    historical blocks) and return the offending forbidden metric
    family token when one appears, else None.
    """
    def walk(value: object) -> str | None:
        if isinstance(value, str):
            for forbidden in _FORBIDDEN_METRIC_SHAPES:
                if forbidden in value:
                    return forbidden
            return None
        if isinstance(value, dict):
            for v in value.values():
                hit = walk(v)
                if hit is not None:
                    return hit
            return None
        if isinstance(value, list):
            for item in value:
                hit = walk(item)
                if hit is not None:
                    return hit
            return None
        return None

    return walk(entry)


def validate_sm120_ledger(
    ledger_path,
    *,
    sm86_required_path,
) -> None:
    """Fail closed with a typed reason on any QA violation.

    Returns None on success. Raises ``LedgerQaError`` with one
    of:
      * ``sm120_required_cell_missing``: at least one sm_86
        required cell is absent from the sm_120 ledger.
      * ``sm120_ledger_duplicate_cell_name``: two entries share
        the same `cell_name`.
      * ``sm120_ledger_missing_required_meta``: a top-level
        ``[meta]`` required field is missing.
      * ``ledger_metric_family_invalid``: any per-entry string
        field carries a forbidden store-conflated metric family
        token.
      * ``sm120_ledger_typed_unavailable_reason_invalid``:
        ``unavailable_reason`` not in the closed vocabulary.
      * ``sm120_ledger_toml_malformed``: the ledger file does
        not parse as TOML.
    """
    p = Path(ledger_path) if not isinstance(ledger_path, Path) else ledger_path
    try:
        ledger = tomllib.loads(p.read_text("utf-8"))
    except Exception as exc:
        raise LedgerQaError(
            "sm120_ledger_toml_malformed",
            f"sm_120 ledger at {p} does not parse as TOML: {exc!r}",
        ) from exc
    meta = ledger.get("meta") or {}
    for field in _REQUIRED_META_FIELDS:
        if field not in meta:
            raise LedgerQaError(
                "sm120_ledger_missing_required_meta",
                f"sm_120 ledger [meta] missing required field "
                f"{field!r}",
            )

    entries = ledger.get("entries") or []
    seen_names: dict[str, int] = {}
    sm120_present: set[str] = set()
    for idx, entry in enumerate(entries):
        name = entry.get("cell_name")
        if not name or not isinstance(name, str):
            raise LedgerQaError(
                "sm120_ledger_missing_required_meta",
                f"entry #{idx} missing or invalid cell_name",
            )
        if name in seen_names:
            raise LedgerQaError(
                "sm120_ledger_duplicate_cell_name",
                f"cell_name {name!r} appears in entries #{seen_names[name]} "
                f"and #{idx}",
            )
        seen_names[name] = idx
        sm120_present.add(name)
        # Metric-family scan
        offender = _scan_for_forbidden_metric_family(entry)
        if offender is not None:
            raise LedgerQaError(
                "ledger_metric_family_invalid",
                f"entry {name!r} carries forbidden store-conflated "
                f"metric family {offender!r}",
            )
        # Typed-unavailable reason vocabulary check
        if "unavailable_reason" in entry:
            reason = entry["unavailable_reason"]
            if reason not in _VALID_TYPED_UNAVAILABLE_REASONS:
                raise LedgerQaError(
                    "sm120_ledger_typed_unavailable_reason_invalid",
                    f"entry {name!r} carries unavailable_reason "
                    f"{reason!r} which is not in the closed "
                    f"vocabulary",
                )

    # Required-cell coverage check
    sm86_path = Path(sm86_required_path) if not isinstance(
        sm86_required_path, Path
    ) else sm86_required_path
    sm86 = tomllib.loads(sm86_path.read_text("utf-8"))
    sm86_required: set = set(
        e["cell_name"] for e in sm86.get("entries", [])
    )
    missing = sorted(sm86_required - sm120_present)
    if missing:
        raise LedgerQaError(
            "sm120_required_cell_missing",
            f"the following sm_86-required cells are absent from "
            f"the sm_120 ledger: {missing[:5]!r}"
            + (f" (+{len(missing)-5} more)" if len(missing) > 5 else ""),
        )


__all__ = [
    "LEDGER_QA_TYPED_REASONS",
    "LedgerQaError",
    "validate_sm120_ledger",
]
