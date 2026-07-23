"""Build the cohort-accounting artifact for the HEADLINE_COHORT_13
cells on a host architecture.

Inputs:

- data/local_labels/label_admitted.json (the post-Round-7 v4
  admitted-cell file). Cells in this file with arch matching the
  host architecture are recorded as ``admitted`` in the cohort
  accounting.

- An optional per-cell exclusion-reasons JSON file describing why
  particular cohort cells are excluded for this round. The shape
  is ``{"<slug>/<shape_point>": {"admission_status": "<enum>",
  "exclusion_reason": "<text>"}, ...}``. Cohort cells absent from
  this file AND absent from the admitted set default to
  ``excluded_operator_skipped`` with a generic operator-skipped
  exclusion_reason.

Output: a strict accounting artifact validating against
``cohort_accounting.schema.json``. The artifact covers every cell
in the constant HEADLINE_COHORT_13 set (silent omissions are a
contract break enforced via the schema's
``minItems=maxItems=13``).

The HEADLINE_COHORT_13 constant set lives in
``workloads/simbench_bw/comparison_runner.py::_PREDICTOR_DESIGN_COVERAGE_BY_CELL``;
the tool consumes that dict at runtime so the cohort definition
stays in lockstep with the comparison runner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SCHEMA_PATH = _HERE / "cohort_accounting.schema.json"

# Shared mixed-arity identity helper: cell_variant_id yields "" for a base cell
# and the canonical register_cap_<n> for a build variant. The headline cohort
# accounting is over the BASE admitted set; register-cap build variants are a
# separate factorial corpus folded in by a later predictor refit, so a
# (slug, shape_point) whose only admitted rows are build variants stays
# classified excluded here (consistent with the committed base accounting).
_LOCAL_LABELS_DIR = _REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS_DIR))
from corpus_cell_identity import cell_variant_id  # noqa: E402


# Closed-enum admission_status values that downstream code can
# reference symbolically. Mirrors the schema's enum.
ADMISSION_STATUS_VALUES: tuple[str, ...] = (
    "admitted",
    "excluded_label_repeatability",
    "excluded_label_provenance_duplicate",
    "excluded_join_key_unmatched",
    "excluded_operator_skipped",
)
_EXCLUDED_STATUSES: frozenset[str] = frozenset(
    s for s in ADMISSION_STATUS_VALUES if s.startswith("excluded_")
)


def _load_headline_cohort_13() -> dict[tuple[str, str], str]:
    """Load the HEADLINE_COHORT_13 constant set from the comparison
    runner. The runner pins the (slug, shape_point) -> bucket map;
    we mirror it as the source of truth so cohort drift surfaces
    immediately via this tool rather than silently emitting fewer
    cells."""
    suite_root = _REPO_ROOT / "workloads" / "simbench_bw"
    if str(suite_root) not in sys.path:
        sys.path.insert(0, str(suite_root))
    import comparison_runner  # type: ignore[import-not-found]

    cohort = dict(
        comparison_runner._PREDICTOR_DESIGN_COVERAGE_BY_CELL,
    )
    if len(cohort) != 13:
        raise ValueError(
            f"build_cohort_accounting: expected 13 cells in "
            f"_PREDICTOR_DESIGN_COVERAGE_BY_CELL; observed "
            f"{len(cohort)} (the constant drifted from the cohort)"
        )
    return cohort


def _load_admitted_cells(
    path: Path, host_arch: str,
) -> tuple[dict[tuple[str, str], dict], int]:
    """Return a dict mapping (slug, shape_point) -> admitted record
    for cells in the file matching ``host_arch``. Also returns the
    file-level schema_version."""
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    schema_version = int(doc.get("schema_version", 0))
    out: dict[tuple[str, str], dict] = {}
    for cell in doc.get("cells", []):
        if str(cell.get("arch", "")) != host_arch:
            continue
        # Build variants are a separate factorial corpus, not part of the
        # base headline cohort accounting (folded in by a later predictor
        # refit); skip them so a base cell that failed admission is not
        # silently re-admitted via a sibling build variant.
        if cell_variant_id(cell):
            continue
        key = (str(cell.get("slug", "")), str(cell.get("shape_point", "")))
        out[key] = cell
    return out, schema_version


def _load_per_cell_exclusions(
    path: Path | None,
) -> dict[tuple[str, str], dict]:
    """Load an optional per-cell exclusions JSON, keyed by
    ``<slug>/<shape_point>``. Returns an empty dict when the path
    is None or the file does not exist."""
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[tuple[str, str], dict] = {}
    for cell_id, entry in (raw or {}).items():
        # Underscore-prefixed keys are documentation-only (e.g.,
        # `_comment` carrying a freeform note about why a sweep was
        # skipped). The loader ignores them so operators can leave
        # provenance notes alongside the structured entries without
        # tripping the format check.
        if str(cell_id).startswith("_"):
            continue
        slug, _sep, shape_point = str(cell_id).partition("/")
        if not slug or not shape_point:
            raise ValueError(
                f"build_cohort_accounting: invalid cell_id "
                f"{cell_id!r} in --per-cell-exclusions; expected "
                f"format <slug>/<shape_point>"
            )
        out[(slug, shape_point)] = dict(entry)
    return out


def _classify_cell(
    slug: str,
    shape_point: str,
    bucket: str,
    *,
    admitted_record: dict | None,
    exclusion_entry: dict | None,
) -> dict:
    """Return one cohort-cell record per the schema's
    ``$defs.cohort_cell`` shape. Classification rule:

    - admitted: the cell appears in the admitted set; status is
      ``admitted``; ``exclusion_reason`` MUST NOT appear.
    - excluded explicit: the cell appears in the exclusions input
      with a status from the excluded_* set; the status + reason
      are carried verbatim.
    - default: the cell is in neither set; status defaults to
      ``excluded_operator_skipped`` with a generic reason naming
      the gap.
    """
    if admitted_record is not None:
        cell: dict = {
            "slug": slug,
            "shape_point": shape_point,
            "predictor_design_coverage_bucket": bucket,
            "admission_status": "admitted",
            "n_repeats": int(admitted_record.get("n_repeats", 0)),
        }
        return cell

    if exclusion_entry is not None:
        status = str(exclusion_entry.get("admission_status", ""))
        if status not in _EXCLUDED_STATUSES:
            raise ValueError(
                f"build_cohort_accounting: per-cell exclusion for "
                f"{slug}/{shape_point} has admission_status="
                f"{status!r}; must be one of "
                f"{sorted(_EXCLUDED_STATUSES)!r}"
            )
        reason = str(exclusion_entry.get("exclusion_reason", "")).strip()
        if not reason:
            raise ValueError(
                f"build_cohort_accounting: per-cell exclusion for "
                f"{slug}/{shape_point} is missing a non-empty "
                f"exclusion_reason"
            )
        return {
            "slug": slug,
            "shape_point": shape_point,
            "predictor_design_coverage_bucket": bucket,
            "admission_status": status,
            "exclusion_reason": reason,
        }

    # Default: operator did not run the cell this round.
    return {
        "slug": slug,
        "shape_point": shape_point,
        "predictor_design_coverage_bucket": bucket,
        "admission_status": "excluded_operator_skipped",
        "exclusion_reason": (
            "operator did not run the SWEEP_RUNBOOK live sweep on "
            "this cell during the current round; the cell is "
            "honestly reported as not-yet-evaluated and will be "
            "re-evaluated on a follow-up operator-budget round"
        ),
    }


def build_cohort_accounting(
    *,
    label_admitted_path: Path,
    per_cell_exclusions_path: Path | None,
    host_arch: str,
) -> dict:
    """Return the cohort-accounting artifact dict. Caller validates
    against the schema and writes to disk."""
    cohort = _load_headline_cohort_13()
    admitted, admitted_schema_version = _load_admitted_cells(
        label_admitted_path, host_arch,
    )
    exclusions = _load_per_cell_exclusions(per_cell_exclusions_path)

    cells: list[dict] = []
    admitted_count = 0
    admitted_count_per_bucket = {
        "predictor_friendly": 0,
        "predictor_hostile": 0,
    }
    # Determinism: emit cells in sorted (slug, shape_point) order so
    # the artifact is byte-deterministic across re-runs with the
    # same inputs.
    for key in sorted(cohort):
        slug, shape_point = key
        bucket = cohort[key]
        record = _classify_cell(
            slug, shape_point, bucket,
            admitted_record=admitted.get(key),
            exclusion_entry=exclusions.get(key),
        )
        cells.append(record)
        if record["admission_status"] == "admitted":
            admitted_count += 1
            admitted_count_per_bucket[bucket] += 1

    # Record the input provenance as a repo-relative path when the
    # input lives under _REPO_ROOT; otherwise carry the absolute
    # path verbatim so test fixtures under /tmp do not crash on the
    # relative-to lookup.
    try:
        recorded_path = str(
            label_admitted_path.relative_to(_REPO_ROOT)
        )
    except ValueError:
        recorded_path = str(label_admitted_path)
    artifact = {
        "schema_version": 1,
        "cohort_size": 13,
        "host_arch": host_arch,
        "label_admitted_path": recorded_path,
        "label_admitted_schema_version": admitted_schema_version,
        "cells": cells,
        "admitted_count": admitted_count,
        "admitted_count_per_bucket": admitted_count_per_bucket,
    }
    return artifact


def _validate(artifact: dict) -> None:
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(instance=artifact, schema=schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the cohort-accounting artifact for the "
            "HEADLINE_COHORT_13 cells on a host architecture. "
            "Reads label_admitted.json + an optional per-cell "
            "exclusion-reasons JSON and emits a strict accounting "
            "artifact covering every cell in the cohort."
        )
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "label_admitted.json",
        help=(
            "Path to label_admitted.json. Default: "
            "data/local_labels/label_admitted.json."
        ),
    )
    parser.add_argument(
        "--per-cell-exclusions",
        type=Path,
        default=None,
        help=(
            "Optional path to a per-cell exclusion-reasons JSON. "
            "When omitted, cohort cells not in the admitted set "
            "default to excluded_operator_skipped."
        ),
    )
    parser.add_argument(
        "--host-arch",
        type=str,
        default="sm_86",
        choices=["sm_86", "sm_120"],
        help=(
            "Host architecture for the accounting. Plan 1's scope "
            "is sm_86 only; sm_120 is reserved for the Tier-2 "
            "cross-arch follow-up."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            _REPO_ROOT
            / "data"
            / "sm_86_predictor_accuracy_recovery_v2"
            / "cohort_accounting_post_corpus_expansion.json"
        ),
        help="Path to write the accounting artifact.",
    )
    args = parser.parse_args(argv)

    artifact = build_cohort_accounting(
        label_admitted_path=args.label_admitted,
        per_cell_exclusions_path=args.per_cell_exclusions,
        host_arch=args.host_arch,
    )
    _validate(artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    sys.stderr.write(
        f"build_cohort_accounting: wrote {args.out} "
        f"(admitted_count={artifact['admitted_count']}/13)\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
