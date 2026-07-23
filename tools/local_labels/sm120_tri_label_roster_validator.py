"""Integrity validator for the frozen sm_120 tri-label cohort roster.

Two layers:

1. ``validate_roster_structure(roster)`` — pre-collection structural gates on
   the roster itself: >=50 cells, the required family-bucket minima, the
   per-bucket ceiling, the per-slug cap, the slug-held-out split (each slug
   entirely one split; both splits non-empty), `(slug, shape_point)`
   uniqueness, the sm_120 arch pin, and the schema pins. These run on the
   roster alone and must pass before any live collection.

2. ``tri_label_coverage(roster, presence)`` — the source-artifact preflight:
   every cell must carry trace + L2 + L1 + duration evidence. It is fed a
   presence map; ``presence_from_artifacts(...)`` builds that map from the real
   trace / `label_admitted.json` / `duration_admitted.json` artifacts, so the
   live collection satisfies it once those labels exist. A cell missing any one
   of its four sources fails closed.

This is cohort-integrity gating for an empirical predictor's evidence base, not
a model of GPU behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROSTER_SCHEMA_VERSION = 1
ROSTER_ID = "sm120_tri_label_v1"
ARCH = "sm_120"
REQUIRED_BUCKETS: tuple[str, ...] = (
    "memory_stress",
    "tensor_compute",
    "activation_norm",
)
MIN_CELLS = 50
PER_SLUG_CAP = 4
BUCKET_MIN = 6
BUCKET_CEILING_FRACTION = 0.35
ADMISSIBLE_SPLITS = ("calibration", "holdout")
TRI_LABEL_SOURCES = ("trace", "l2", "l1", "duration")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class RosterValidationError(ValueError):
    """Raised when the roster violates a structural integrity gate."""


def validate_roster_structure(
    roster: dict, *, inventory_path: Path | None = None
) -> list[str]:
    """Return a list of structural-gate + pin violations; empty means valid.

    When ``inventory_path`` is provided, the roster's ``inventory_sha256``
    pin is recomputed against that file and a stale pin fails closed."""
    problems: list[str] = []
    if roster.get("schema_version") != ROSTER_SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {ROSTER_SCHEMA_VERSION}, got {roster.get('schema_version')!r}"
        )
    if roster.get("arch") != ARCH:
        problems.append(f"arch must be {ARCH!r}, got {roster.get('arch')!r}")

    # Frozen / commit-pin enforcement: the gate-constant pins must match the
    # validator constants (so a hand-edited roster cannot loosen a gate), and
    # the provenance pins must be present and well-formed.
    if roster.get("roster_id") != ROSTER_ID:
        problems.append(f"roster_id must be {ROSTER_ID!r}, got {roster.get('roster_id')!r}")
    if roster.get("per_slug_cap") != PER_SLUG_CAP:
        problems.append(f"per_slug_cap pin must be {PER_SLUG_CAP}, got {roster.get('per_slug_cap')!r}")
    if roster.get("required_buckets") != list(REQUIRED_BUCKETS):
        problems.append(
            f"required_buckets pin must be {list(REQUIRED_BUCKETS)}, got {roster.get('required_buckets')!r}"
        )
    if roster.get("bucket_min") != BUCKET_MIN:
        problems.append(f"bucket_min pin must be {BUCKET_MIN}, got {roster.get('bucket_min')!r}")
    bcf = roster.get("bucket_ceiling_fraction")
    if not isinstance(bcf, (int, float)) or abs(float(bcf) - BUCKET_CEILING_FRACTION) > 1e-9:
        problems.append(
            f"bucket_ceiling_fraction pin must be {BUCKET_CEILING_FRACTION}, got {bcf!r}"
        )
    inventory_sha256 = roster.get("inventory_sha256")
    if not isinstance(inventory_sha256, str) or not _HEX64.match(inventory_sha256):
        problems.append(f"inventory_sha256 must be a 64-char hex pin, got {inventory_sha256!r}")
    source_commit = roster.get("source_commit")
    if not isinstance(source_commit, str) or not _HEX40.match(source_commit):
        problems.append(f"source_commit must be a 40-char hex git pin, got {source_commit!r}")
    if inventory_path is not None and isinstance(inventory_sha256, str):
        try:
            actual = hashlib.sha256(Path(inventory_path).read_bytes()).hexdigest()
        except OSError as exc:
            problems.append(f"inventory_path {inventory_path!s} unreadable for staleness check: {exc}")
        else:
            if actual != inventory_sha256:
                problems.append(
                    f"inventory_sha256 pin {inventory_sha256} is stale; the current "
                    f"inventory at {inventory_path!s} hashes to {actual}"
                )

    cells = roster.get("cells")
    if not isinstance(cells, list):
        problems.append("roster.cells must be a list")
        return problems
    if "total_cells" in roster and roster["total_cells"] != len(cells):
        problems.append(
            f"total_cells {roster['total_cells']} disagrees with len(cells) {len(cells)}"
        )
    if len(cells) < MIN_CELLS:
        problems.append(f"roster has {len(cells)} cells; contract requires >= {MIN_CELLS}")

    seen_keys: set[tuple[str, str]] = set()
    per_slug: dict[str, int] = {}
    per_bucket: dict[str, int] = {}
    split_by_slug: dict[str, set[str]] = {}
    splits_present: set[str] = set()
    for index, cell in enumerate(cells):
        for field in ("slug", "shape_point", "family_bucket", "split"):
            if not cell.get(field):
                problems.append(f"cell[{index}] missing required field {field!r}")
        slug = cell.get("slug")
        shape_point = cell.get("shape_point")
        bucket = cell.get("family_bucket")
        split = cell.get("split")
        if slug is None or shape_point is None:
            continue
        key = (str(slug), str(shape_point))
        if key in seen_keys:
            problems.append(f"duplicate (slug, shape_point) cell {key!r}")
        seen_keys.add(key)
        per_slug[str(slug)] = per_slug.get(str(slug), 0) + 1
        if bucket is not None:
            per_bucket[str(bucket)] = per_bucket.get(str(bucket), 0) + 1
            if bucket not in REQUIRED_BUCKETS:
                problems.append(
                    f"cell {key!r} family_bucket {bucket!r} is not one of {list(REQUIRED_BUCKETS)}"
                )
        if split is not None:
            if split not in ADMISSIBLE_SPLITS:
                problems.append(f"cell {key!r} split {split!r} is not one of {list(ADMISSIBLE_SPLITS)}")
            else:
                splits_present.add(str(split))
                split_by_slug.setdefault(str(slug), set()).add(str(split))

    for slug, count in sorted(per_slug.items()):
        if count > PER_SLUG_CAP:
            problems.append(
                f"slug {slug!r} has {count} shape_points; per-slug cap is {PER_SLUG_CAP}"
            )
    for bucket in REQUIRED_BUCKETS:
        count = per_bucket.get(bucket, 0)
        if count < BUCKET_MIN:
            problems.append(
                f"family bucket {bucket!r} has {count} cells; minimum is {BUCKET_MIN}"
            )
    total = len(cells)
    if total > 0:
        for bucket, count in sorted(per_bucket.items()):
            if count > BUCKET_CEILING_FRACTION * total:
                problems.append(
                    f"family bucket {bucket!r} has {count}/{total} cells "
                    f"({count / total:.3f}); ceiling is {BUCKET_CEILING_FRACTION}"
                )
    for slug, splits in sorted(split_by_slug.items()):
        if len(splits) > 1:
            problems.append(
                f"slug {slug!r} appears in multiple splits {sorted(splits)}; "
                "the slug-held-out split requires each slug entirely in one split"
            )
    for required_split in ADMISSIBLE_SPLITS:
        if required_split not in splits_present:
            problems.append(f"split {required_split!r} has no cells; both splits must be populated")
    return problems


def validate_roster_structure_or_raise(
    roster: dict, *, inventory_path: Path | None = None
) -> None:
    problems = validate_roster_structure(roster, inventory_path=inventory_path)
    if problems:
        raise RosterValidationError("; ".join(problems))


def tri_label_coverage(roster: dict, presence: dict) -> list[dict]:
    """Return per-cell tri-label coverage gaps. ``presence`` maps
    ``(slug, shape_point)`` to an iterable of the sources present for that
    cell (subset of ``TRI_LABEL_SOURCES``). A cell missing any source is a
    gap; the live collection must close every gap before the cell is admitted.
    """
    gaps: list[dict] = []
    for cell in roster.get("cells", []):
        key = (str(cell.get("slug")), str(cell.get("shape_point")))
        present = set(presence.get(key, ()) or ())
        missing = [src for src in TRI_LABEL_SOURCES if src not in present]
        if missing:
            gaps.append({"slug": key[0], "shape_point": key[1], "missing": missing})
    return gaps


def _load_admitted_cells(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], dict] = {}
    for cell in payload.get("cells", []):
        out[(str(cell.get("arch")), str(cell.get("slug")), str(cell.get("shape_point")))] = cell
    return out


def presence_from_artifacts(
    roster: dict,
    repo_root: Path,
    *,
    label_admitted_path: Path | None = None,
    duration_admitted_path: Path | None = None,
) -> dict[tuple[str, str], set[str]]:
    """Build the tri-label presence map from the real on-disk artifacts:
    the trace JSON (per the roster cell's path), the L2 + L1 labels in
    ``label_admitted.json``, and the duration label in
    ``duration_admitted.json``. The live collection populates these; until
    then a cell simply shows fewer sources present."""
    data_local = repo_root / "data" / "local_labels"
    label_admitted = _load_admitted_cells(
        label_admitted_path or (data_local / "label_admitted.json")
    )
    duration_admitted = _load_admitted_cells(
        duration_admitted_path or (data_local / "duration_admitted.json")
    )
    presence: dict[tuple[str, str], set[str]] = {}
    arch = str(roster.get("arch", ARCH))
    for cell in roster.get("cells", []):
        slug = str(cell.get("slug"))
        shape_point = str(cell.get("shape_point"))
        sources: set[str] = set()
        trace_rel = cell.get("trace_json_path")
        if trace_rel and (repo_root / trace_rel).is_file():
            sources.add("trace")
        label = label_admitted.get((arch, slug, shape_point))
        if label is not None:
            if label.get("cache_hit_rate_l2_sector_pct") is not None:
                sources.add("l2")
            if (
                label.get("cache_hit_rate_l1tex_sector_pct") is not None
                and label.get("l1tex_label_status") in ("mean_present", "all_zero")
            ):
                sources.add("l1")
        if (arch, slug, shape_point) in duration_admitted:
            sources.add("duration")
        presence[(slug, shape_point)] = sources
    return presence
