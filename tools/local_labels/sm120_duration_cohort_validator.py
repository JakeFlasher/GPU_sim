"""Integrity validator for the sm_120 CUPTI dual-source duration cohort.

This cohort is SEPARATE from the tri-label roster: it lists the large
single-kernel portable workloads (GEMM/attention) whose kernels scale to
multi-ms and reconcile the dual-source duration bound, with a slug-held-out
split for the duration predictor. The frozen tri-label roster's microbenchmark
cells cannot carry a dual-source duration label (see
``data/local_labels/duration_candidate_diagnostic.json``), so the duration
predictor trains on this cohort instead.

Two layers:

1. ``validate_duration_cohort_structure(cohort)`` — structural + pin gates on
   the cohort itself (slug-held-out split, per-slug cap, sm_120 arch, a 40-hex
   ``source_commit`` pin, unique ``(slug, shape_point)``).
2. ``cohort_admission_coverage(cohort, admitted_path)`` — reports which cohort
   cells are actually admitted in ``duration_admitted.json`` (so a cell that
   fails the repeatability gate is surfaced, not silently assumed present).

These are empirical cohort-integrity gates for a wall-clock duration predictor's
evidence base, not a model of GPU behavior.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

COHORT_SCHEMA_VERSION = 1
COHORT_ID = "sm120_duration_v1"
ARCH = "sm_120"
PER_SLUG_CAP = 4
ADMISSIBLE_SPLITS = ("calibration", "holdout")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class DurationCohortValidationError(ValueError):
    """Raised when the duration cohort violates a structural gate."""


def validate_duration_cohort_structure(cohort: dict) -> list[str]:
    """Return a list of structural/pin violations; empty means valid."""
    problems: list[str] = []
    if cohort.get("schema_version") != COHORT_SCHEMA_VERSION:
        problems.append(f"schema_version must be {COHORT_SCHEMA_VERSION}, got {cohort.get('schema_version')!r}")
    if cohort.get("cohort_id") != COHORT_ID:
        problems.append(f"cohort_id must be {COHORT_ID!r}, got {cohort.get('cohort_id')!r}")
    if cohort.get("arch") != ARCH:
        problems.append(f"arch must be {ARCH!r}, got {cohort.get('arch')!r}")
    if cohort.get("per_slug_cap") != PER_SLUG_CAP:
        problems.append(f"per_slug_cap pin must be {PER_SLUG_CAP}, got {cohort.get('per_slug_cap')!r}")
    source_commit = cohort.get("source_commit")
    if not isinstance(source_commit, str) or not _HEX40.match(source_commit):
        problems.append(f"source_commit must be a 40-char hex git pin, got {source_commit!r}")

    cells = cohort.get("cells")
    if not isinstance(cells, list):
        problems.append("cohort.cells must be a list")
        return problems
    if "total_cells" in cohort and cohort["total_cells"] != len(cells):
        problems.append(f"total_cells {cohort['total_cells']} disagrees with len(cells) {len(cells)}")

    seen: set[tuple[str, str]] = set()
    per_slug: dict[str, int] = {}
    split_by_slug: dict[str, set[str]] = {}
    splits_present: set[str] = set()
    for i, cell in enumerate(cells):
        for field in ("slug", "shape_point", "split", "binary_path"):
            if not cell.get(field):
                problems.append(f"cell[{i}] missing required field {field!r}")
        slug, shape, split = cell.get("slug"), cell.get("shape_point"), cell.get("split")
        if slug is None or shape is None:
            continue
        key = (str(slug), str(shape))
        if key in seen:
            problems.append(f"duplicate (slug, shape_point) cell {key!r}")
        seen.add(key)
        per_slug[str(slug)] = per_slug.get(str(slug), 0) + 1
        if split is not None:
            if split not in ADMISSIBLE_SPLITS:
                problems.append(f"cell {key!r} split {split!r} not in {list(ADMISSIBLE_SPLITS)}")
            else:
                splits_present.add(str(split))
                split_by_slug.setdefault(str(slug), set()).add(str(split))
    for slug, count in sorted(per_slug.items()):
        if count > PER_SLUG_CAP:
            problems.append(f"slug {slug!r} has {count} shape_points; per-slug cap is {PER_SLUG_CAP}")
    for slug, splits in sorted(split_by_slug.items()):
        if len(splits) > 1:
            problems.append(
                f"slug {slug!r} appears in multiple splits {sorted(splits)}; "
                "the slug-held-out split requires each slug entirely in one split"
            )
    for required in ADMISSIBLE_SPLITS:
        if required not in splits_present:
            problems.append(f"split {required!r} has no cells; both splits must be populated")
    return problems


def validate_duration_cohort_structure_or_raise(cohort: dict) -> None:
    problems = validate_duration_cohort_structure(cohort)
    if problems:
        raise DurationCohortValidationError("; ".join(problems))


def cohort_admission_coverage(cohort: dict, admitted_path: Path) -> dict:
    """Report which cohort cells are admitted in ``duration_admitted.json``.

    Returns ``{"admitted": [...], "missing": [...]}`` where each entry is
    ``(slug, shape_point)``. A missing cell is one the repeatability aggregator
    did not admit (e.g. it failed the dual-source or robust-CV gate)."""
    arch = str(cohort.get("arch", ARCH))
    admitted_keys: set[tuple[str, str, str]] = set()
    if admitted_path.is_file():
        payload = json.loads(admitted_path.read_text(encoding="utf-8"))
        for cell in payload.get("cells", []):
            admitted_keys.add((str(cell.get("arch")), str(cell.get("slug")), str(cell.get("shape_point"))))
    admitted: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for cell in cohort.get("cells", []):
        key3 = (arch, str(cell.get("slug")), str(cell.get("shape_point")))
        (admitted if key3 in admitted_keys else missing).append((key3[1], key3[2]))
    return {"admitted": admitted, "missing": missing}
