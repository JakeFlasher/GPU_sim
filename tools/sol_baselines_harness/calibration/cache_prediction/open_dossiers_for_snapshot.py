"""Operator CLI helper for opening regression-triage dossiers.

After every triggered intermediate-baseline snapshot lands under
`.humanize/diagnostic-baselines/`, the operator runs this helper
to open a `.humanize/predictor-regressions/<problem_slug>.md`
dossier for every measured cell whose
`l1_calibrated_rel_err >= 0.10` or
`l2_calibrated_rel_err >= 0.10`. The dossier carries the full
regression-triage schema (slug, measured / raw rel-err values,
denominator-floor fields, `miss_class`,
`resolution_path_attempted`, `attempts_remaining`,
`terminal_classification`); the bounded triage loop in a later
operator round mutates the dossier as each resolution path
runs.

Usage (from the repo root):

    .venv/bin/python tools/sol_baselines_harness/calibration/cache_prediction/open_dossiers_for_snapshot.py \\
        --snapshot-path .humanize/diagnostic-baselines/intermediate-baseline-<sha>.toml

The dossier directory defaults to
`.humanize/predictor-regressions/` relative to the repo root.
Existing dossier files are NEVER overwritten by default;
`--overwrite` re-authors them (use only when intentionally
replanning the triage initial state).

Exit codes:

  * 0 — every required dossier was opened (or already exists).
    Existing dossiers are NOT overwritten and are recorded in
    the `already_existed` list; pass `--overwrite` to re-author
    them.
  * 4 — snapshot parse failure (re-raised
    `SnapshotDossierWalkerError`).

This module is calibration-only and never imported by the
deployed harness path. The NCU quarantine scanner enforces that
contract.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
_TOOLS_ROOT = _REPO_ROOT / "tools"
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction.snapshot_dossier_walker import (  # noqa: E402
    SNAPSHOT_DOSSIER_WALKER_REL_ERR_THRESHOLD,
    STATUS_MEASURED,
    SnapshotDossierWalkerError,
    select_per_cell_rows_from_snapshot_data,
)


MISS_CLASSES: frozenset[str] = frozenset({
    "bias_like",
    "variance_like",
    "zero_traffic",
    "routing_mismatch",
    "warm_cold_mismatch",
    "geometry_mismatch",
    "out_of_distribution",
    "missing_observability_signal",
})

RESOLUTION_PATHS: frozenset[str] = frozenset({
    "calibrator_refit",
    "geometry_fix",
    "routing_fix",
    "observability_extension",
    "typed_unavailable_escalation",
})

TERMINAL_CLASSIFICATIONS: frozenset[str] = frozenset({
    "resolved_below_target",
    "requires_new_observability_landed",
    "typed_unavailable_candidate",
    "known_failing_measured_cell",
})


DEFAULT_MISS_CLASS: str = "out_of_distribution"
DEFAULT_RESOLUTION_PATH: str = "calibrator_refit"
DEFAULT_TERMINAL_CLASSIFICATION: str = "known_failing_measured_cell"
DEFAULT_ATTEMPTS_REMAINING: int = 3
ATTEMPTS_REMAINING_MIN: int = 0
ATTEMPTS_REMAINING_MAX: int = 3
DEFAULT_DOSSIER_DIR_RELATIVE: str = (
    ".humanize/predictor-regressions"
)


@dataclass(frozen=True)
class DossierOpenResult:
    """Result of an `open_dossiers_for_snapshot()` invocation.

    Fields:
      * `snapshot_path` — the parsed snapshot.
      * `dossier_dir` — the directory dossiers were authored in.
      * `opened` — list of dossier paths newly written.
      * `already_existed` — list of dossier paths that already
        existed and were NOT overwritten.
    """

    snapshot_path: Path
    dossier_dir: Path
    opened: tuple[Path, ...]
    already_existed: tuple[Path, ...]


def _format_field(value: Any) -> str:
    """Render a per-cell field for the dossier body. Numeric
    values are rendered with full precision; the `"nan"`
    sentinel and other non-numeric values render verbatim.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(float(value))
    return str(value)


def _dossier_text(
    row: Mapping[str, Any],
    *,
    miss_class: str,
    resolution_path: str,
    attempts_remaining: int,
    terminal_classification: str,
    snapshot_path: Path,
) -> str:
    """Render the dossier markdown body. The frontmatter uses
    YAML-style key:value lines so a later round's triage loop
    can parse the dossier with a simple line-based reader (no
    YAML dependency needed).
    """
    slug = row["problem_slug"]
    measured_l1 = _format_field(row.get("l1_calibrated_rel_err"))
    measured_l2 = _format_field(row.get("l2_calibrated_rel_err"))
    raw_predicted_l1 = _format_field(row.get("predicted_l1_raw"))
    raw_predicted_l2 = _format_field(row.get("predicted_l2_raw"))
    raw_measured_l1 = _format_field(row.get("ncu_l1_label"))
    raw_measured_l2 = _format_field(row.get("ncu_l2_label"))
    denominator_floor = _format_field(
        row.get("denominator_floor", 0.01),
    )
    try:
        denom_floor_pct = (
            float(row.get("denominator_floor", 0.01)) * 100.0
        )
        denominator_floor_pct = repr(denom_floor_pct)
    except (TypeError, ValueError):
        denominator_floor_pct = "nan"
    predictor_bucket = row.get("predictor_bucket", "unknown")
    return (
        f"---\n"
        f"slug: {slug}\n"
        f"predictor_bucket: {predictor_bucket}\n"
        f"measured_l1_rel_err: {measured_l1}\n"
        f"measured_l2_rel_err: {measured_l2}\n"
        f"raw_predicted_l1_hit_rate: {raw_predicted_l1}\n"
        f"raw_predicted_l2_hit_rate: {raw_predicted_l2}\n"
        f"raw_measured_l1_hit_rate: {raw_measured_l1}\n"
        f"raw_measured_l2_hit_rate: {raw_measured_l2}\n"
        f"denominator_floor_applied: {denominator_floor}\n"
        f"denominator_floor_pct: {denominator_floor_pct}\n"
        f"miss_class: {miss_class}\n"
        f"resolution_path_attempted: {resolution_path}\n"
        f"attempts_remaining: {attempts_remaining}\n"
        f"terminal_classification: {terminal_classification}\n"
        f"source_snapshot: {snapshot_path.name}\n"
        f"---\n"
        f"\n"
        f"# Regression Dossier — `{slug}`\n"
        f"\n"
        f"## Snapshot source\n"
        f"\n"
        f"Emitted-snapshot path: `{snapshot_path}`\n"
        f"\n"
        f"## Initial state\n"
        f"\n"
        f"The cell was opened with the conservative initial "
        f"terminal classification `known_failing_measured_cell` "
        f"and `attempts_remaining = {attempts_remaining}` (one "
        f"of each resolution path: calibrator_refit, "
        f"geometry_fix, routing_fix, observability_extension, "
        f"typed_unavailable_escalation). The "
        f"`resolution_path_attempted` field above is the "
        f"scheduled FIRST path the bounded triage loop will "
        f"run; the field carries a closed-vocabulary value at "
        f"open time and the triage loop updates it as each "
        f"path runs.\n"
        f"\n"
        f"## Resolution path attempts\n"
        f"\n"
        f"No attempt has been executed yet. The next scheduled "
        f"path is `{resolution_path}`.\n"
        f"\n"
        f"## Observability extensions landed\n"
        f"\n"
        f"None.\n"
    )


def _iter_high_rel_err_measured_rows(
    snapshot_path: Path,
    *,
    rel_err_threshold: float,
) -> Iterable[Mapping[str, Any]]:
    """Parse the snapshot TOML and yield per-cell rows whose
    `status == "measured"` AND `(l1_rel_err >= threshold OR
    l2_rel_err >= threshold)`.

    Re-uses the walker's
    `select_per_cell_rows_from_snapshot_data()` resolver so the
    standalone `[[per_cell_accuracy]]` shape and the smoke-
    appended `[[accuracy_diagnostic_per_cell]]` alias are both
    honored under the same precedence (standalone first).

    Raises `SnapshotDossierWalkerError` on snapshot
    parse/IO failure or when neither per-cell table key is
    present.
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
    rows, _selected_table = (
        select_per_cell_rows_from_snapshot_data(data, snapshot_path)
    )
    for row in rows:
        if row.get("status") != STATUS_MEASURED:
            continue
        l1 = row.get("l1_calibrated_rel_err")
        l2 = row.get("l2_calibrated_rel_err")
        l1_for_trigger = l1 if isinstance(l1, (int, float)) else 0.0
        l2_for_trigger = l2 if isinstance(l2, (int, float)) else 0.0
        if (
            l1_for_trigger < rel_err_threshold
            and l2_for_trigger < rel_err_threshold
        ):
            continue
        yield row


def open_dossiers_for_snapshot(
    snapshot_path: Path,
    dossier_dir: Path,
    *,
    rel_err_threshold: float = SNAPSHOT_DOSSIER_WALKER_REL_ERR_THRESHOLD,
    default_miss_class: str = DEFAULT_MISS_CLASS,
    default_resolution_path: str = DEFAULT_RESOLUTION_PATH,
    default_attempts_remaining: int = DEFAULT_ATTEMPTS_REMAINING,
    default_terminal_classification: str = (
        DEFAULT_TERMINAL_CLASSIFICATION
    ),
    overwrite: bool = False,
) -> DossierOpenResult:
    """Walk `snapshot_path` for every measured high-rel-err cell
    and author a `.humanize/predictor-regressions/<slug>.md`
    dossier for each.

    Returns a `DossierOpenResult` carrying the snapshot, dossier
    dir, list of newly-opened dossier paths, and list of dossier
    paths that already existed (and were NOT overwritten).
    Existing dossiers are treated as success — they are recorded
    in `already_existed` and the caller exits 0; the bounded
    triage loop owns dossier mutation, not this helper. Pass
    `overwrite=True` to force re-authoring.

    Raises:
      * `SnapshotDossierWalkerError` on snapshot parse / IO
        failure (re-raised by the CLI as exit 4).
      * `ValueError` when any caller-supplied default is outside
        its closed-vocabulary set or attempts_remaining is
        outside `[0, 3]`.

    Behavior:
      * `default_miss_class` defaults to `out_of_distribution`
        (a conservative initial classification operators can
        re-classify during triage).
      * `default_resolution_path` defaults to `calibrator_refit`
        — the deterministic first scheduled path. The
        regression-triage schema requires the field to be a
        closed-vocabulary value at open time; the bounded
        triage loop updates the field as each path runs.
      * `default_terminal_classification` defaults to
        `known_failing_measured_cell` (the conservative
        pre-triage state, updated when triage actually
        resolves the cell).
      * `default_attempts_remaining` defaults to 3 and is
        validated to be an integer in `[0, 3]`.
    """

    if default_miss_class not in MISS_CLASSES:
        raise ValueError(
            f"default_miss_class {default_miss_class!r} is not "
            f"in the closed set {sorted(MISS_CLASSES)}",
        )
    if default_resolution_path not in RESOLUTION_PATHS:
        raise ValueError(
            f"default_resolution_path "
            f"{default_resolution_path!r} is not in the closed "
            f"set {sorted(RESOLUTION_PATHS)}; the regression-"
            f"triage schema requires every dossier to declare "
            f"its scheduled resolution path from this set",
        )
    if (
        default_terminal_classification
        not in TERMINAL_CLASSIFICATIONS
    ):
        raise ValueError(
            f"default_terminal_classification "
            f"{default_terminal_classification!r} is not in the "
            f"closed set {sorted(TERMINAL_CLASSIFICATIONS)}",
        )
    if (
        isinstance(default_attempts_remaining, bool)
        or not isinstance(default_attempts_remaining, int)
    ):
        raise ValueError(
            f"default_attempts_remaining must be an integer in "
            f"[{ATTEMPTS_REMAINING_MIN}, "
            f"{ATTEMPTS_REMAINING_MAX}], got "
            f"{type(default_attempts_remaining).__name__} "
            f"{default_attempts_remaining!r}",
        )
    if not (
        ATTEMPTS_REMAINING_MIN
        <= default_attempts_remaining
        <= ATTEMPTS_REMAINING_MAX
    ):
        raise ValueError(
            f"default_attempts_remaining = "
            f"{default_attempts_remaining} is outside the "
            f"closed interval [{ATTEMPTS_REMAINING_MIN}, "
            f"{ATTEMPTS_REMAINING_MAX}]; the regression-triage "
            f"schema bounds attempts at one of each resolution "
            f"path (initial = 3)",
        )

    dossier_dir.mkdir(parents=True, exist_ok=True)
    opened: list[Path] = []
    already_existed: list[Path] = []
    for row in _iter_high_rel_err_measured_rows(
        snapshot_path, rel_err_threshold=rel_err_threshold,
    ):
        slug = row["problem_slug"]
        dossier_path = dossier_dir / f"{slug}.md"
        if dossier_path.is_file():
            if not overwrite:
                already_existed.append(dossier_path)
                continue
        text = _dossier_text(
            row,
            miss_class=default_miss_class,
            resolution_path=default_resolution_path,
            attempts_remaining=default_attempts_remaining,
            terminal_classification=default_terminal_classification,
            snapshot_path=snapshot_path,
        )
        dossier_path.write_text(text, encoding="utf-8")
        opened.append(dossier_path)
    return DossierOpenResult(
        snapshot_path=snapshot_path,
        dossier_dir=dossier_dir,
        opened=tuple(opened),
        already_existed=tuple(already_existed),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open regression-triage dossiers for every measured "
            "cell in an emitted diagnostic snapshot whose "
            "rel-err is at or above the trigger threshold."
        ),
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        required=True,
        help="path to the emitted intermediate-baseline TOML",
    )
    parser.add_argument(
        "--dossier-dir",
        type=Path,
        default=None,
        help=(
            "dossier output directory (default: in-tree "
            f"{DEFAULT_DOSSIER_DIR_RELATIVE}/)"
        ),
    )
    parser.add_argument(
        "--rel-err-threshold",
        type=float,
        default=SNAPSHOT_DOSSIER_WALKER_REL_ERR_THRESHOLD,
        help=(
            f"rel-err trigger threshold (default: "
            f"{SNAPSHOT_DOSSIER_WALKER_REL_ERR_THRESHOLD}; "
            f"cells AT OR ABOVE this value get a dossier)"
        ),
    )
    parser.add_argument(
        "--default-miss-class",
        type=str,
        default=DEFAULT_MISS_CLASS,
        help=(
            f"initial miss_class for newly-opened dossiers "
            f"(default: {DEFAULT_MISS_CLASS}; closed set: "
            f"{sorted(MISS_CLASSES)})"
        ),
    )
    parser.add_argument(
        "--default-resolution-path",
        type=str,
        default=DEFAULT_RESOLUTION_PATH,
        help=(
            f"initial resolution_path_attempted for newly-opened "
            f"dossiers (default: {DEFAULT_RESOLUTION_PATH}, the "
            f"deterministic first scheduled path; closed set: "
            f"{sorted(RESOLUTION_PATHS)})"
        ),
    )
    parser.add_argument(
        "--default-terminal-classification",
        type=str,
        default=DEFAULT_TERMINAL_CLASSIFICATION,
        help=(
            f"initial terminal_classification for newly-opened "
            f"dossiers (default: "
            f"{DEFAULT_TERMINAL_CLASSIFICATION}; closed set: "
            f"{sorted(TERMINAL_CLASSIFICATIONS)})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help=(
            "re-author existing dossiers (default: existing "
            "dossiers are preserved and recorded in the "
            "`already_existed` list; the helper exits 0)"
        ),
    )
    return parser


def _resolve_dossier_dir(dossier_dir: Path | None) -> Path:
    if dossier_dir is not None:
        return dossier_dir
    return _REPO_ROOT / DEFAULT_DOSSIER_DIR_RELATIVE


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(
        argv if argv is not None else sys.argv[1:],
    )
    dossier_dir = _resolve_dossier_dir(args.dossier_dir)
    try:
        result = open_dossiers_for_snapshot(
            args.snapshot_path,
            dossier_dir,
            rel_err_threshold=args.rel_err_threshold,
            default_miss_class=args.default_miss_class,
            default_resolution_path=args.default_resolution_path,
            default_terminal_classification=(
                args.default_terminal_classification
            ),
            overwrite=args.overwrite,
        )
    except SnapshotDossierWalkerError as exc:
        print(
            f"error: snapshot parse / IO failure: "
            f"{exc.typed_reason}: {exc.message}",
            file=sys.stderr,
        )
        return 4
    high_rel_err_total = len(result.opened) + len(
        result.already_existed,
    )
    print(
        f"open_dossiers_for_snapshot: snapshot="
        f"{result.snapshot_path}",
    )
    print(f"  dossier_dir = {result.dossier_dir}")
    print(
        f"  high_rel_err_measured_cells = {high_rel_err_total}",
    )
    print(f"  opened = {len(result.opened)}")
    print(
        f"  already_existed = {len(result.already_existed)}",
    )
    if not args.overwrite and result.already_existed:
        print(
            f"  (refusing to overwrite existing dossiers; pass "
            f"--overwrite to re-author)",
        )
    for path in result.opened:
        print(f"    opened: {path}")
    for path in result.already_existed:
        print(f"    already-existed: {path}")
    return 0


__all__ = [
    "ATTEMPTS_REMAINING_MAX",
    "ATTEMPTS_REMAINING_MIN",
    "DEFAULT_ATTEMPTS_REMAINING",
    "DEFAULT_DOSSIER_DIR_RELATIVE",
    "DEFAULT_MISS_CLASS",
    "DEFAULT_RESOLUTION_PATH",
    "DEFAULT_TERMINAL_CLASSIFICATION",
    "MISS_CLASSES",
    "RESOLUTION_PATHS",
    "TERMINAL_CLASSIFICATIONS",
    "DossierOpenResult",
    "main",
    "open_dossiers_for_snapshot",
]


if __name__ == "__main__":
    raise SystemExit(main())
