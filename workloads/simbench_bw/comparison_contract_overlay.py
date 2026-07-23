"""Loader and required-field validator for sm_86 submission-track
overlays over ``workloads/simbench_bw/comparison_contract.toml``.

The base contract pins the project's broader cross-arch / two-device
program. An overlay narrows the view for a specific submission track
(currently: sm_86 on the RTX 3090 Ti, L2-only prediction with L1 as
appendix-only diagnostics). The overlay's required-field list is the
contract this loader enforces; consumers (the headline runner, the
holdout-evaluation report writer, etc.) MUST query the loaded overlay
through this module rather than parsing the TOML themselves so the
required-field set stays single-source-of-truth.

Required fields
---------------
Every field below MUST be present in the overlay TOML. A missing field
raises :class:`OverlayRequiredFieldMissing` with the dotted path
identifying the gap. The required set is intentionally minimal: it
covers the eight positive-test fields named in the plan plus
schema_version + overlay_kind so the loader can detect a non-overlay
file pointed at the flag.

Typed diagnostic tokens
-----------------------
``cell_arch_outside_overlay_scope`` is the typed diagnostic the
headline runner emits when invoked with an overlay AND a cohort cell
whose arch is outside the overlay's ``scope.arch_set``. The token
literal is exported here as
:data:`CELL_ARCH_OUTSIDE_OVERLAY_SCOPE_DIAGNOSTIC` so the runner, the
overlay-aware tests, and the docstring on this surface stay in
multi-surface lockstep.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - test host runs Python 3.11+
    import tomli as tomllib  # type: ignore[import-not-found]


# ---------------------------------------------------------------------
# Public typed-diagnostic tokens (multi-surface lockstep anchors)
# ---------------------------------------------------------------------

CELL_ARCH_OUTSIDE_OVERLAY_SCOPE_DIAGNOSTIC = "cell_arch_outside_overlay_scope"
"""Typed diagnostic emitted by the headline runner when an overlay is
active AND a cohort cell's arch is not in the overlay's
``scope.arch_set``. Listed here as the canonical literal; the runner
and tests both import this constant."""

OVERLAY_REQUIRED_FIELDS: tuple[tuple[str, ...], ...] = (
    ("schema_version",),
    ("overlay_kind",),
    ("scope", "arch_set"),
    ("scope", "cache_layers"),
    ("scope", "device_set"),
    ("scope", "prior_art_handling"),
    ("coverage_gate", "publication_grade_floor_per_arch"),
    ("coverage_gate", "cohort_paper_target_unique_slugs"),
    ("coverage_gate", "cohort_paper_target_cells"),
    ("coverage_gate", "aspirational_target_unique_slugs"),
)
"""Closed set of required dotted-path fields. The required-field
validator iterates this tuple and raises :class:`OverlayRequiredFieldMissing`
for any path absent from the loaded TOML payload."""


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------


class OverlayValidationError(Exception):
    """Base class for overlay validation failures."""


class OverlayRequiredFieldMissing(OverlayValidationError):
    """Raised when an overlay TOML omits a required dotted-path field."""

    def __init__(self, missing_path: tuple[str, ...]) -> None:
        self.missing_path = missing_path
        super().__init__(
            f"overlay required field missing: {'.'.join(missing_path)}"
        )


class OverlayFieldTypeError(OverlayValidationError):
    """Raised when an overlay required field is present but the wrong
    type (e.g. ``arch_set`` not a list of strings)."""

    def __init__(
        self,
        path: tuple[str, ...],
        expected: str,
        observed: object,
    ) -> None:
        self.path = path
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"overlay field {'.'.join(path)!s} has wrong type: "
            f"expected {expected}, observed {type(observed).__name__}"
        )


# ---------------------------------------------------------------------
# Loaded overlay representation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SubmissionOverlay:
    """In-memory representation of a validated submission-track overlay.

    Construct via :func:`load_overlay`. Consumers query the dataclass
    directly rather than peeking at the raw TOML so the field set stays
    enforced by static typing.
    """

    schema_version: int
    overlay_kind: str
    arch_set: tuple[str, ...]
    cache_layers: tuple[str, ...]
    device_set: tuple[str, ...]
    prior_art_handling: str
    publication_grade_floor_per_arch: int
    cohort_paper_target_unique_slugs: int
    cohort_paper_target_cells: int
    aspirational_target_unique_slugs: int
    raw: Mapping[str, Any]


# ---------------------------------------------------------------------
# Required-field probe
# ---------------------------------------------------------------------


def _walk(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Walk a nested mapping along ``path``. Returns the value at the
    end of the walk, or :class:`OverlayRequiredFieldMissing` raised
    from the missing key. Sentinel ``object()`` for "not found" would
    be confusable with a legitimately-``None`` leaf value, so raise
    instead.
    """
    cursor: Any = payload
    for i, key in enumerate(path):
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise OverlayRequiredFieldMissing(path[: i + 1])
        cursor = cursor[key]
    return cursor


def _require_list_of_str(
    payload: Mapping[str, Any], path: tuple[str, ...],
) -> tuple[str, ...]:
    value = _walk(payload, path)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise OverlayFieldTypeError(
            path=path, expected="list[str]", observed=value,
        )
    return tuple(value)


def _require_int(
    payload: Mapping[str, Any], path: tuple[str, ...],
) -> int:
    value = _walk(payload, path)
    if isinstance(value, bool) or not isinstance(value, int):
        # bool is a subclass of int in Python; reject it explicitly.
        raise OverlayFieldTypeError(
            path=path, expected="int", observed=value,
        )
    return value


def _require_str(
    payload: Mapping[str, Any], path: tuple[str, ...],
) -> str:
    value = _walk(payload, path)
    if not isinstance(value, str):
        raise OverlayFieldTypeError(
            path=path, expected="str", observed=value,
        )
    return value


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------


def load_overlay(path: Path) -> SubmissionOverlay:
    """Read ``path`` as a submission-track overlay and validate every
    required field. Returns the populated :class:`SubmissionOverlay`.

    Raises :class:`OverlayRequiredFieldMissing` for any field listed in
    :data:`OVERLAY_REQUIRED_FIELDS` that the file omits, and
    :class:`OverlayFieldTypeError` when a required field is present but
    has the wrong type.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Walk the required-field tuple once to surface the FIRST missing
    # field with a clean error (consumers running this in a test loop
    # can iterate the closed enumeration).
    for required_path in OVERLAY_REQUIRED_FIELDS:
        _walk(raw, required_path)

    return SubmissionOverlay(
        schema_version=_require_int(raw, ("schema_version",)),
        overlay_kind=_require_str(raw, ("overlay_kind",)),
        arch_set=_require_list_of_str(raw, ("scope", "arch_set")),
        cache_layers=_require_list_of_str(raw, ("scope", "cache_layers")),
        device_set=_require_list_of_str(raw, ("scope", "device_set")),
        prior_art_handling=_require_str(
            raw, ("scope", "prior_art_handling"),
        ),
        publication_grade_floor_per_arch=_require_int(
            raw, ("coverage_gate", "publication_grade_floor_per_arch"),
        ),
        cohort_paper_target_unique_slugs=_require_int(
            raw, ("coverage_gate", "cohort_paper_target_unique_slugs"),
        ),
        cohort_paper_target_cells=_require_int(
            raw, ("coverage_gate", "cohort_paper_target_cells"),
        ),
        aspirational_target_unique_slugs=_require_int(
            raw, ("coverage_gate", "aspirational_target_unique_slugs"),
        ),
        raw=raw,
    )


__all__ = [
    "CELL_ARCH_OUTSIDE_OVERLAY_SCOPE_DIAGNOSTIC",
    "OVERLAY_REQUIRED_FIELDS",
    "OverlayValidationError",
    "OverlayRequiredFieldMissing",
    "OverlayFieldTypeError",
    "SubmissionOverlay",
    "load_overlay",
]
