"""Production walker linking emitted diagnostic snapshots to
their required regression dossiers.

Operator contract: after every triggered intermediate-baseline
snapshot lands under `.humanize/diagnostic-baselines/`, every
measured per-cell row with `l1_calibrated_rel_err >= 0.10` OR
`l2_calibrated_rel_err >= 0.10` MUST have a corresponding
`.humanize/predictor-regressions/<problem_slug>.md` dossier on
disk. The walker is the source-of-truth check for the
`incremental_triage_dossier_missing` invariant; it parses the
emitted snapshot's TOML directly so the lint runs against the
SAME bytes the operator saw, not a synthetic in-test fixture.

The walker's `>=` boundary mirrors `accuracy_diagnostic.py`'s
strict-less-than threshold: a cell with rel-err exactly equal to
the threshold (default `0.10`) IS treated as requiring a dossier;
only cells STRICTLY below pass without a dossier. Reviewers can
verify this boundary by feeding the walker a synthetic snapshot
with a row at `0.10`.

This module is calibration-only and never imported by the
deployed harness path. The NCU quarantine scanner enforces that
contract.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


SNAPSHOT_DOSSIER_WALKER_REL_ERR_THRESHOLD: float = 0.10

# The standalone diagnostic snapshots emitted by
# `emit_intermediate_baseline()` write the per-cell rows under
# the unprefixed `[[per_cell_accuracy]]` table (see
# `accuracy_diagnostic.serialize_diagnostic_report()` and its
# `_PER_CELL_TABLE_KEY = "per_cell_accuracy"`). The smoke-appended
# diagnostic section uses the prefixed
# `[[accuracy_diagnostic_per_cell]]` alias (see the
# `(per_cell_accuracy, accuracy_diagnostic_per_cell)` namespace
# tuple in `accuracy_diagnostic.py`). The walker tries the
# standalone name first and falls back to the prefixed alias so
# that BOTH on-disk shapes are honored.
DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME_STANDALONE: str = (
    "per_cell_accuracy"
)
DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME: str = (
    "accuracy_diagnostic_per_cell"
)
DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME_CANDIDATES: tuple[str, ...] = (
    DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME_STANDALONE,
    DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME,
)

STATUS_MEASURED: str = "measured"

INTERMEDIATE_BASELINE_GLOB: str = "intermediate-baseline-*.toml"
FINAL_BASELINE_GLOB: str = "baseline-*-final.toml"

SNAPSHOT_DOSSIER_WALKER_TYPED_REASONS: frozenset[str] = frozenset({
    "snapshot_dossier_walker_io_error",
    "snapshot_dossier_walker_invalid_toml",
    "snapshot_dossier_walker_missing_required_field",
    "incremental_triage_dossier_missing",
})


class SnapshotDossierWalkerError(Exception):
    """Typed error raised by the snapshot dossier walker.

    `typed_reason` is drawn from the closed set
    `SNAPSHOT_DOSSIER_WALKER_TYPED_REASONS`.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in SNAPSHOT_DOSSIER_WALKER_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of "
                f"{sorted(SNAPSHOT_DOSSIER_WALKER_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class MissingDossier:
    """One missing-dossier finding from a snapshot walk.

    Fields:
      * `slug` — the measured cell's `problem_slug` whose
        dossier is absent.
      * `snapshot_path` — the snapshot TOML that surfaced the
        finding (so reviewers can correlate the missing dossier
        with the exact emission that flagged it).
      * `l1_calibrated_rel_err` / `l2_calibrated_rel_err` — the
        cell's per-cohort rel-err at snapshot time, useful for
        prioritizing dossier authoring.
    """

    slug: str
    snapshot_path: Path
    l1_calibrated_rel_err: float
    l2_calibrated_rel_err: float


def select_per_cell_rows_from_snapshot_data(
    data: Mapping[str, object], snapshot_path: Path,
) -> tuple[list[Mapping[str, object]], str]:
    """Resolve the per-cell-rows array from a parsed snapshot
    `data` dict using the standalone-then-alias precedence.

    Returns `(rows, selected_table_name)` on success. The
    `selected_table_name` is one of
    `DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME_CANDIDATES` and
    documents which on-disk shape the rows came from — callers
    that care about logging which table was inspected (the live
    repository lint, for example) read this second tuple
    element.

    Raises `SnapshotDossierWalkerError(
    snapshot_dossier_walker_missing_required_field)` when:
      * NEITHER candidate table key is present in `data`.
      * The selected candidate is present but is not a list of
        tables (mappings).
      * Any individual row in the array is not a mapping.

    Standalone shape (preferred) — emitted by
    `serialize_diagnostic_report()`:
        [[per_cell_accuracy]]
        problem_slug = ...
        ...

    Smoke-appended alias shape — emitted when the diagnostic is
    concatenated after the smoke records:
        [[accuracy_diagnostic_per_cell]]
        problem_slug = ...
        ...
    """
    for candidate in DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME_CANDIDATES:
        if candidate not in data:
            continue
        rows_raw = data[candidate]
        if not isinstance(rows_raw, list):
            raise SnapshotDossierWalkerError(
                "snapshot_dossier_walker_missing_required_field",
                f"snapshot at {snapshot_path!s} "
                f"[[{candidate}]] must be an array of tables, "
                f"got {type(rows_raw).__name__}",
            )
        rows: list[Mapping[str, object]] = []
        for index, row in enumerate(rows_raw):
            if not isinstance(row, Mapping):
                raise SnapshotDossierWalkerError(
                    "snapshot_dossier_walker_missing_required_field",
                    f"snapshot at {snapshot_path!s} "
                    f"[[{candidate}]] index {index} is not a "
                    f"table",
                )
            rows.append(row)
        return rows, candidate
    raise SnapshotDossierWalkerError(
        "snapshot_dossier_walker_missing_required_field",
        f"snapshot at {snapshot_path!s} carries neither "
        f"[[{DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME_STANDALONE}]] "
        f"(the standalone serializer's table name) nor "
        f"[[{DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME}]] (the "
        f"smoke-appended alias); a valid snapshot MUST emit one "
        f"of these per-cell-rows tables",
    )


def _coerce_rel_err(
    value: object, field_name: str, slug: str, snapshot_path: Path,
) -> float | None:
    """Return the rel-err as a `float` if the cell carries a
    finite numeric value; return `None` for the `"nan"` sentinel
    or any non-numeric type (such rows are skipped by the
    walker rather than triggering a dossier — a non-numeric
    rel-err on a measured row is a schema oddity, not a
    dossier-trigger).

    Raises `SnapshotDossierWalkerError(
    snapshot_dossier_walker_missing_required_field)` only when
    `field_name` is absent entirely (caller passes the lookup
    result; a `KeyError` upstream surfaces here).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return None
    raise SnapshotDossierWalkerError(
        "snapshot_dossier_walker_missing_required_field",
        f"snapshot at {snapshot_path!s} cell slug={slug!r} "
        f"field {field_name!r} has unexpected type "
        f"{type(value).__name__}",
    )


def walk_emitted_snapshot_for_missing_dossiers(
    snapshot_path: Path,
    dossier_dir: Path,
    *,
    rel_err_threshold: float = SNAPSHOT_DOSSIER_WALKER_REL_ERR_THRESHOLD,
) -> list[MissingDossier]:
    """Parse `snapshot_path` and return per-cell rows whose
    rel-err is at or above the threshold but whose dossier is
    absent under `dossier_dir`.

    Only rows with status `"measured"` are eligible; rows
    carrying any other status (typed_unavailable, corpus_pending,
    host_arch_unsupported, diagnostic_failed) are skipped per
    the dossier-trigger contract (only measured cells can
    trigger a triage dossier).

    `rel_err_threshold` defaults to `0.10` (the strict-less-than
    user-target threshold). The walker uses `>=` so a cell at
    exactly the threshold IS triggered — mirroring the
    dossier-trigger boundary the cadence-marker invariant
    documents.
    """
    try:
        raw_bytes = snapshot_path.read_bytes()
    except OSError as exc:
        raise SnapshotDossierWalkerError(
            "snapshot_dossier_walker_io_error",
            f"cannot read snapshot at {snapshot_path!s}: {exc}",
        ) from exc
    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SnapshotDossierWalkerError(
            "snapshot_dossier_walker_invalid_toml",
            f"snapshot at {snapshot_path!s} is not valid TOML: "
            f"{exc}",
        ) from exc
    rows, _selected_table = select_per_cell_rows_from_snapshot_data(
        data, snapshot_path,
    )
    missing: list[MissingDossier] = []
    for index, row in enumerate(rows):
        for required in ("problem_slug", "status"):
            if required not in row:
                raise SnapshotDossierWalkerError(
                    "snapshot_dossier_walker_missing_required_field",
                    f"snapshot at {snapshot_path!s} cell index "
                    f"{index} is missing required field "
                    f"{required!r}",
                )
        slug = row["problem_slug"]
        status = row["status"]
        if status != STATUS_MEASURED:
            continue
        for required in (
            "l1_calibrated_rel_err",
            "l2_calibrated_rel_err",
        ):
            if required not in row:
                raise SnapshotDossierWalkerError(
                    "snapshot_dossier_walker_missing_required_field",
                    f"snapshot at {snapshot_path!s} cell "
                    f"slug={slug!r} is missing required field "
                    f"{required!r}",
                )
        l1 = _coerce_rel_err(
            row["l1_calibrated_rel_err"],
            "l1_calibrated_rel_err", slug, snapshot_path,
        )
        l2 = _coerce_rel_err(
            row["l2_calibrated_rel_err"],
            "l2_calibrated_rel_err", slug, snapshot_path,
        )
        if l1 is None and l2 is None:
            continue
        l1_for_trigger = l1 if l1 is not None else 0.0
        l2_for_trigger = l2 if l2 is not None else 0.0
        if (
            l1_for_trigger < rel_err_threshold
            and l2_for_trigger < rel_err_threshold
        ):
            continue
        dossier_file = dossier_dir / f"{slug}.md"
        if dossier_file.is_file():
            continue
        missing.append(
            MissingDossier(
                slug=slug,
                snapshot_path=snapshot_path,
                l1_calibrated_rel_err=l1_for_trigger,
                l2_calibrated_rel_err=l2_for_trigger,
            ),
        )
    return missing


def _discover_snapshot_paths(
    baselines_dir: Path,
) -> Iterable[Path]:
    """Yield every snapshot TOML under `baselines_dir` matching
    the operator's two naming conventions: incremental snapshots
    `intermediate-baseline-<sha>.toml` and final dashboards
    `baseline-<sha>-final.toml`.
    """
    seen: set[Path] = set()
    for pattern in (
        INTERMEDIATE_BASELINE_GLOB, FINAL_BASELINE_GLOB,
    ):
        for path in sorted(baselines_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            yield path


def walk_all_diagnostic_baselines(
    baselines_dir: Path,
    dossier_dir: Path,
    *,
    rel_err_threshold: float = SNAPSHOT_DOSSIER_WALKER_REL_ERR_THRESHOLD,
) -> dict[Path, list[MissingDossier]]:
    """Walk every snapshot under `baselines_dir` matching the
    operator's naming conventions and return a mapping
    `snapshot_path -> list[MissingDossier]` covering missing
    dossiers across all snapshots.

    Snapshots with no missing dossiers are included with an empty
    list — reviewers can confirm the walker actually inspected
    them rather than silently skipping.

    Raises `SnapshotDossierWalkerError` on the first malformed
    snapshot encountered (IO error or invalid TOML).
    """
    if not baselines_dir.is_dir():
        raise SnapshotDossierWalkerError(
            "snapshot_dossier_walker_io_error",
            f"baselines_dir {baselines_dir!s} is not a directory",
        )
    findings: dict[Path, list[MissingDossier]] = {}
    for snapshot_path in _discover_snapshot_paths(baselines_dir):
        findings[snapshot_path] = (
            walk_emitted_snapshot_for_missing_dossiers(
                snapshot_path,
                dossier_dir,
                rel_err_threshold=rel_err_threshold,
            )
        )
    return findings


__all__ = [
    "DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME",
    "DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME_CANDIDATES",
    "DIAGNOSTIC_BASELINE_PER_CELL_TABLE_NAME_STANDALONE",
    "FINAL_BASELINE_GLOB",
    "INTERMEDIATE_BASELINE_GLOB",
    "SNAPSHOT_DOSSIER_WALKER_REL_ERR_THRESHOLD",
    "SNAPSHOT_DOSSIER_WALKER_TYPED_REASONS",
    "STATUS_MEASURED",
    "MissingDossier",
    "SnapshotDossierWalkerError",
    "select_per_cell_rows_from_snapshot_data",
    "walk_all_diagnostic_baselines",
    "walk_emitted_snapshot_for_missing_dossiers",
]
