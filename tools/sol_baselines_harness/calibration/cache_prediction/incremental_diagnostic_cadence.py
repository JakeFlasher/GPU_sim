"""Incremental-diagnostic cadence marker (M2/M4 trigger gate).

After every newly-measured row lands in the calibration ledger,
the operator records the transition in
`.humanize/predictor-regressions/incremental_diagnostic_cadence_marker.toml`.
Each marker entry pins the per-cohort row counts before/after
the ledger update, the per-cohort deltas, whether the cadence
threshold was crossed (`triggered`), and — when triggered — the
path of the emitted intermediate-baseline snapshot.

Cadence thresholds (operator-set per DEC-B):

  * `CADENCE_THRESHOLD_FRIENDLY_NEW_ROWS = 20` — emit a snapshot
    when 20+ newly-measured `predictor_friendly` rows have
    landed since the last triggered snapshot.
  * `CADENCE_THRESHOLD_HOSTILE_NEW_ROWS = 10` — emit a snapshot
    when 10+ newly-measured `predictor_hostile` rows have landed
    since the last triggered snapshot.

The thresholds are cumulative: a marker entry records the
per-cohort delta since the previous triggered entry, NOT since
the previous entry of any kind. A non-triggered entry leaves the
threshold-counter accumulating until the next entry crosses it.

This module is calibration-only and never imported by the
deployed harness path. The NCU quarantine scanner enforces that
contract.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


INCREMENTAL_CADENCE_SCHEMA_VERSION: str = "1.0"

CADENCE_THRESHOLD_FRIENDLY_NEW_ROWS: int = 20
CADENCE_THRESHOLD_HOSTILE_NEW_ROWS: int = 10


INCREMENTAL_CADENCE_TYPED_REASONS: frozenset[str] = frozenset({
    "incremental_cadence_io_error",
    "incremental_cadence_invalid_toml",
    "incremental_cadence_invalid_schema_version",
    "incremental_cadence_missing_field",
    "incremental_cadence_negative_count",
    "incremental_cadence_delta_inconsistent",
    "incremental_cadence_triggered_requires_snapshot_path",
    "cadence_marker_threshold_inconsistent",
    "incremental_cadence_pre_post_discontinuity",
    "incremental_cadence_cumulative_mismatch",
    "incremental_triage_dossier_missing",
})


class IncrementalCadenceMarkerError(Exception):
    """Typed error raised by the cadence-marker loader / writer.

    `typed_reason` is drawn from the closed set
    `INCREMENTAL_CADENCE_TYPED_REASONS`.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in INCREMENTAL_CADENCE_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of "
                f"{sorted(INCREMENTAL_CADENCE_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class IncrementalCadenceMarkerEntry:
    """One ledger-transition record in the cadence marker.

    Fields:
      * `pre_count_friendly` / `pre_count_hostile` — per-cohort
        row counts in the ledger BEFORE this transition.
      * `post_count_friendly` / `post_count_hostile` — per-cohort
        row counts AFTER this transition.
      * `delta_friendly` / `delta_hostile` — per-cohort row-count
        deltas (must equal `post - pre`).
      * `cumulative_delta_friendly_since_trigger` /
        `cumulative_delta_hostile_since_trigger` — cumulative
        per-cohort deltas since the previous triggered entry
        (or since the marker's first entry if none has triggered
        yet). The threshold-cross logic operates on the
        cumulative values, not the per-transition delta.
      * `triggered` — bool; `True` iff at least one cumulative
        delta crossed its cohort threshold and the operator
        emitted a fresh intermediate-baseline snapshot.
      * `emitted_snapshot_path` — required when `triggered`;
        `None` otherwise.
      * `recorded_at_utc` — ISO-8601 UTC timestamp captured when
        the entry was authored.
      * `commit_sha` — git commit SHA at the time the entry was
        authored (so reviewers can replay the snapshot from a
        reproducible point).
    """

    pre_count_friendly: int
    pre_count_hostile: int
    post_count_friendly: int
    post_count_hostile: int
    delta_friendly: int
    delta_hostile: int
    cumulative_delta_friendly_since_trigger: int
    cumulative_delta_hostile_since_trigger: int
    triggered: bool
    emitted_snapshot_path: str | None
    recorded_at_utc: str
    commit_sha: str


@dataclass(frozen=True)
class IncrementalCadenceMarker:
    """Parsed cadence marker.

    `entries` is ordered chronologically: index 0 is the oldest
    transition, index -1 is the most recent. Empty entries are
    valid for an initial author (no transitions recorded yet).
    """

    schema_version: str
    entries: tuple[IncrementalCadenceMarkerEntry, ...]


def should_trigger_snapshot(
    cumulative_delta_friendly_since_trigger: int,
    cumulative_delta_hostile_since_trigger: int,
) -> bool:
    """Return `True` iff either cumulative per-cohort delta has
    crossed its threshold under DEC-B (20 friendly / 10 hostile).

    The predicate is the source-of-truth used by both the
    operator writer (to decide whether to emit a snapshot) and
    the invariant test (to verify the `triggered` field matches
    the cumulative-delta arithmetic).
    """
    return (
        cumulative_delta_friendly_since_trigger
        >= CADENCE_THRESHOLD_FRIENDLY_NEW_ROWS
        or cumulative_delta_hostile_since_trigger
        >= CADENCE_THRESHOLD_HOSTILE_NEW_ROWS
    )


def _require_field(
    table: Mapping[str, object], field_name: str, context: str,
) -> object:
    if field_name not in table:
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_missing_field",
            f"{context} is missing required field {field_name!r}",
        )
    return table[field_name]


def _require_int(
    value: object, field_name: str, context: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_missing_field",
            f"{context}.{field_name} must be an integer, got "
            f"{type(value).__name__}",
        )
    return value


def _require_str(
    value: object, field_name: str, context: str,
) -> str:
    if not isinstance(value, str):
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_missing_field",
            f"{context}.{field_name} must be a string, got "
            f"{type(value).__name__}",
        )
    return value


def _require_non_negative_int(
    value: int, field_name: str, context: str,
) -> int:
    if value < 0:
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_negative_count",
            f"{context}.{field_name} = {value} must be "
            f"non-negative",
        )
    return value


def _parse_entry(
    raw: object, index: int,
) -> IncrementalCadenceMarkerEntry:
    if not isinstance(raw, Mapping):
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_missing_field",
            f"[[entries]] index {index} is not a table",
        )
    context = f"[[entries]] index {index}"
    pre_friendly = _require_non_negative_int(
        _require_int(
            _require_field(raw, "pre_count_friendly", context),
            "pre_count_friendly", context,
        ),
        "pre_count_friendly", context,
    )
    pre_hostile = _require_non_negative_int(
        _require_int(
            _require_field(raw, "pre_count_hostile", context),
            "pre_count_hostile", context,
        ),
        "pre_count_hostile", context,
    )
    post_friendly = _require_non_negative_int(
        _require_int(
            _require_field(raw, "post_count_friendly", context),
            "post_count_friendly", context,
        ),
        "post_count_friendly", context,
    )
    post_hostile = _require_non_negative_int(
        _require_int(
            _require_field(raw, "post_count_hostile", context),
            "post_count_hostile", context,
        ),
        "post_count_hostile", context,
    )
    delta_friendly = _require_int(
        _require_field(raw, "delta_friendly", context),
        "delta_friendly", context,
    )
    delta_hostile = _require_int(
        _require_field(raw, "delta_hostile", context),
        "delta_hostile", context,
    )
    cumulative_friendly = _require_non_negative_int(
        _require_int(
            _require_field(
                raw,
                "cumulative_delta_friendly_since_trigger",
                context,
            ),
            "cumulative_delta_friendly_since_trigger", context,
        ),
        "cumulative_delta_friendly_since_trigger", context,
    )
    cumulative_hostile = _require_non_negative_int(
        _require_int(
            _require_field(
                raw,
                "cumulative_delta_hostile_since_trigger",
                context,
            ),
            "cumulative_delta_hostile_since_trigger", context,
        ),
        "cumulative_delta_hostile_since_trigger", context,
    )
    if delta_friendly != post_friendly - pre_friendly:
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_delta_inconsistent",
            f"{context}.delta_friendly = {delta_friendly} does "
            f"not equal post_count_friendly - pre_count_friendly "
            f"= {post_friendly - pre_friendly}",
        )
    if delta_hostile != post_hostile - pre_hostile:
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_delta_inconsistent",
            f"{context}.delta_hostile = {delta_hostile} does "
            f"not equal post_count_hostile - pre_count_hostile "
            f"= {post_hostile - pre_hostile}",
        )
    triggered_raw = _require_field(raw, "triggered", context)
    if not isinstance(triggered_raw, bool):
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_missing_field",
            f"{context}.triggered must be a boolean, got "
            f"{type(triggered_raw).__name__}",
        )
    triggered = triggered_raw
    snapshot_path_raw = raw.get("emitted_snapshot_path", None)
    if triggered:
        if not isinstance(snapshot_path_raw, str) or not snapshot_path_raw:
            raise IncrementalCadenceMarkerError(
                "incremental_cadence_triggered_requires_snapshot_path",
                f"{context}.emitted_snapshot_path must be a "
                f"non-empty string when triggered = true",
            )
        emitted_snapshot_path: str | None = snapshot_path_raw
    else:
        if snapshot_path_raw is not None:
            if not isinstance(snapshot_path_raw, str):
                raise IncrementalCadenceMarkerError(
                    "incremental_cadence_missing_field",
                    f"{context}.emitted_snapshot_path must be a "
                    f"string when present",
                )
            emitted_snapshot_path = (
                snapshot_path_raw if snapshot_path_raw else None
            )
        else:
            emitted_snapshot_path = None
    recorded_at_utc = _require_str(
        _require_field(raw, "recorded_at_utc", context),
        "recorded_at_utc", context,
    )
    commit_sha = _require_str(
        _require_field(raw, "commit_sha", context),
        "commit_sha", context,
    )
    return IncrementalCadenceMarkerEntry(
        pre_count_friendly=pre_friendly,
        pre_count_hostile=pre_hostile,
        post_count_friendly=post_friendly,
        post_count_hostile=post_hostile,
        delta_friendly=delta_friendly,
        delta_hostile=delta_hostile,
        cumulative_delta_friendly_since_trigger=cumulative_friendly,
        cumulative_delta_hostile_since_trigger=cumulative_hostile,
        triggered=triggered,
        emitted_snapshot_path=emitted_snapshot_path,
        recorded_at_utc=recorded_at_utc,
        commit_sha=commit_sha,
    )


def _validate_marker_history(
    entries: tuple[IncrementalCadenceMarkerEntry, ...],
) -> None:
    """Walk the parsed entries chronologically and enforce the
    history-level invariants the per-entry parser cannot see:

      1. Chronological continuity: for entry N>0, the pre counts
         must equal entry N-1's post counts. A mismatch raises
         `incremental_cadence_pre_post_discontinuity`.
      2. Recomputed cumulative-since-trigger: the trigger-base
         counts start at the FIRST entry's pre counts (so an
         operator can begin tracking cadence against a ledger
         that already carries rows from earlier work) and reset
         to entry K's post counts after every triggered entry
         K. The per-cohort recomputed cumulative is
         `post_count - trigger_base_count`. Any entry whose
         claimed `cumulative_delta_*_since_trigger` diverges
         from the recomputed value raises
         `incremental_cadence_cumulative_mismatch`.
      3. Threshold consistency: every entry's `triggered` field
         must equal `should_trigger_snapshot(recomputed_friendly,
         recomputed_hostile)`. A divergence raises
         `cadence_marker_threshold_inconsistent` and the
         message reports the RECOMPUTED cumulative values, not
         the (possibly-wrong) claimed values, so the diagnostic
         is honest about what the history actually says.

    The history validator is the source-of-truth check for the
    cadence trigger rule. An earlier per-entry-only threshold
    check was insufficient because it accepted any history
    whose per-entry claimed cumulative values were locally
    consistent with each entry's `triggered` flag, even when
    those claimed values disagreed with the actual entry-to-
    entry deltas.
    """
    if not entries:
        return
    trigger_base_friendly = entries[0].pre_count_friendly
    trigger_base_hostile = entries[0].pre_count_hostile
    for index, entry in enumerate(entries):
        context = f"[[entries]] index {index}"
        if index > 0:
            prev = entries[index - 1]
            if (
                entry.pre_count_friendly != prev.post_count_friendly
                or entry.pre_count_hostile != prev.post_count_hostile
            ):
                raise IncrementalCadenceMarkerError(
                    "incremental_cadence_pre_post_discontinuity",
                    f"{context}.pre_count_* "
                    f"({entry.pre_count_friendly} friendly, "
                    f"{entry.pre_count_hostile} hostile) does not "
                    f"equal prior entry's post_count_* "
                    f"({prev.post_count_friendly} friendly, "
                    f"{prev.post_count_hostile} hostile)",
                )
        recomputed_cumulative_friendly = (
            entry.post_count_friendly - trigger_base_friendly
        )
        recomputed_cumulative_hostile = (
            entry.post_count_hostile - trigger_base_hostile
        )
        if (
            entry.cumulative_delta_friendly_since_trigger
            != recomputed_cumulative_friendly
            or entry.cumulative_delta_hostile_since_trigger
            != recomputed_cumulative_hostile
        ):
            raise IncrementalCadenceMarkerError(
                "incremental_cadence_cumulative_mismatch",
                f"{context} claimed cumulative_delta_*_since_trigger "
                f"({entry.cumulative_delta_friendly_since_trigger} "
                f"friendly, "
                f"{entry.cumulative_delta_hostile_since_trigger} "
                f"hostile) does not equal the value recomputed "
                f"from history "
                f"({recomputed_cumulative_friendly} friendly, "
                f"{recomputed_cumulative_hostile} hostile) given "
                f"the trigger-base counts "
                f"({trigger_base_friendly} friendly, "
                f"{trigger_base_hostile} hostile)",
            )
        threshold_says_trigger = should_trigger_snapshot(
            recomputed_cumulative_friendly,
            recomputed_cumulative_hostile,
        )
        if entry.triggered != threshold_says_trigger:
            raise IncrementalCadenceMarkerError(
                "cadence_marker_threshold_inconsistent",
                f"{context}.triggered = {entry.triggered} but "
                f"the cadence thresholds "
                f"({CADENCE_THRESHOLD_FRIENDLY_NEW_ROWS} friendly "
                f"/ {CADENCE_THRESHOLD_HOSTILE_NEW_ROWS} hostile) "
                f"require triggered = {threshold_says_trigger} "
                f"for recomputed cumulative deltas "
                f"({recomputed_cumulative_friendly} friendly, "
                f"{recomputed_cumulative_hostile} hostile)",
            )
        if entry.triggered:
            trigger_base_friendly = entry.post_count_friendly
            trigger_base_hostile = entry.post_count_hostile


def load_cadence_marker(path: Path) -> IncrementalCadenceMarker:
    """Load and validate the cadence-marker TOML at `path`.

    Raises `IncrementalCadenceMarkerError` with a typed reason
    drawn from `INCREMENTAL_CADENCE_TYPED_REASONS` on any
    schema, arithmetic, or threshold-consistency violation.
    """
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_io_error",
            f"cannot read cadence marker at {path!s}: {exc}",
        ) from exc
    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_invalid_toml",
            f"cadence marker at {path!s} is not valid TOML: "
            f"{exc}",
        ) from exc
    schema_version = _require_str(
        _require_field(data, "schema_version", "cadence marker"),
        "schema_version", "cadence marker",
    )
    if schema_version != INCREMENTAL_CADENCE_SCHEMA_VERSION:
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_invalid_schema_version",
            f"cadence marker schema_version = "
            f"{schema_version!r} is not "
            f"{INCREMENTAL_CADENCE_SCHEMA_VERSION!r}",
        )
    entries_raw = data.get("entries", [])
    if not isinstance(entries_raw, list):
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_missing_field",
            f"cadence marker [[entries]] must be an array of "
            f"tables, got {type(entries_raw).__name__}",
        )
    entries = tuple(
        _parse_entry(entry, index)
        for index, entry in enumerate(entries_raw)
    )
    _validate_marker_history(entries)
    return IncrementalCadenceMarker(
        schema_version=schema_version, entries=entries,
    )


def serialize_cadence_marker(
    marker: IncrementalCadenceMarker,
) -> str:
    """Serialize a `IncrementalCadenceMarker` to TOML.

    The serialization is stable: array order is preserved;
    None-valued `emitted_snapshot_path` is omitted from the
    output (it is restored to `None` on round-trip via the
    parser's `triggered = false` path).
    """
    lines: list[str] = []
    lines.append(
        f'schema_version = "{marker.schema_version}"',
    )
    for entry in marker.entries:
        lines.append("")
        lines.append("[[entries]]")
        lines.append(
            f"pre_count_friendly = {entry.pre_count_friendly}",
        )
        lines.append(
            f"pre_count_hostile = {entry.pre_count_hostile}",
        )
        lines.append(
            f"post_count_friendly = {entry.post_count_friendly}",
        )
        lines.append(
            f"post_count_hostile = {entry.post_count_hostile}",
        )
        lines.append(f"delta_friendly = {entry.delta_friendly}")
        lines.append(f"delta_hostile = {entry.delta_hostile}")
        lines.append(
            f"cumulative_delta_friendly_since_trigger = "
            f"{entry.cumulative_delta_friendly_since_trigger}",
        )
        lines.append(
            f"cumulative_delta_hostile_since_trigger = "
            f"{entry.cumulative_delta_hostile_since_trigger}",
        )
        lines.append(
            f"triggered = {str(entry.triggered).lower()}",
        )
        if entry.emitted_snapshot_path is not None:
            lines.append(
                f'emitted_snapshot_path = '
                f'"{entry.emitted_snapshot_path}"',
            )
        lines.append(
            f'recorded_at_utc = "{entry.recorded_at_utc}"',
        )
        lines.append(f'commit_sha = "{entry.commit_sha}"')
    return "\n".join(lines) + "\n"


def append_cadence_marker_entry(
    path: Path, entry: IncrementalCadenceMarkerEntry,
) -> None:
    """Append `entry` to the cadence marker at `path`.

    If `path` does not exist, a fresh marker with one entry is
    written. If `path` exists, the file is loaded + validated,
    the new entry is appended, the cumulative-delta consistency
    is re-checked end-to-end, and the marker is re-serialized.

    Raises `IncrementalCadenceMarkerError` if the resulting
    marker would violate any invariant.
    """
    if path.exists():
        marker = load_cadence_marker(path)
        entries = marker.entries + (entry,)
    else:
        entries = (entry,)
    new_marker = IncrementalCadenceMarker(
        schema_version=INCREMENTAL_CADENCE_SCHEMA_VERSION,
        entries=entries,
    )
    text = serialize_cadence_marker(new_marker)
    tmp = path.parent / f".{path.name}.tmp"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        load_cadence_marker(tmp)
        tmp.replace(path)
    except IncrementalCadenceMarkerError:
        if tmp.exists():
            tmp.unlink()
        raise
    except OSError as exc:
        if tmp.exists():
            tmp.unlink()
        raise IncrementalCadenceMarkerError(
            "incremental_cadence_io_error",
            f"cannot write cadence marker to {path!s}: {exc}",
        ) from exc


__all__ = [
    "CADENCE_THRESHOLD_FRIENDLY_NEW_ROWS",
    "CADENCE_THRESHOLD_HOSTILE_NEW_ROWS",
    "INCREMENTAL_CADENCE_SCHEMA_VERSION",
    "INCREMENTAL_CADENCE_TYPED_REASONS",
    "IncrementalCadenceMarker",
    "IncrementalCadenceMarkerEntry",
    "IncrementalCadenceMarkerError",
    "append_cadence_marker_entry",
    "load_cadence_marker",
    "serialize_cadence_marker",
    "should_trigger_snapshot",
]
