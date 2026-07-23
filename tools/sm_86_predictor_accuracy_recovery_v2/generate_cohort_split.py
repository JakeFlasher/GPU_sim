"""Deterministic generator for ``cohort_split.json`` over the admitted
cohort recorded in ``cohort_manifest.json``.

The split is DERIVED from the manifest: the manifest is the single
authority for which cells are admitted and which split each admitted
cell belongs to (its ``split_assignment`` field). This generator reads
the admitted set, groups by ``split_assignment`` into the three blocks,
stamps the per-split floors, computes the ``cohort_manifest_digest``
lockstep over the paired manifest, and validates the result against
``validate_cohort_split`` before writing — so the generator fails
closed rather than emitting a split that violates any invariant.

The split-assignment POLICY (``assign_splits``) lives here and is the
ONE source of truth the manifest builder imports so the manifest's
``split_assignment`` fields and this split agree by construction. The
policy is deterministic (no randomness, sorted inputs) so the same
admitted cohort always produces the same split:

  * Multi-cell slugs (a slug with more than one shape_point / build
    variant) go to ``calibration``. This keeps validation/holdout
    single-cell-per-slug (so their >=N-unique-slug floors are met with
    N cells) and concentrates the multi-cell mass in calibration where
    the 1/3 balancing has the most head-room.
  * Single-cell slugs are ordered by ``(family_bucket, slug)`` and
    dealt round-robin into ``holdout`` -> ``validation`` -> ``calibration``
    with holdout/validation capped at their targets and calibration
    taking the overflow. The family-stratified ordering spreads every
    family across the three splits (the family-balance-across-splits
    invariant), and the caps keep calibration the largest block.

Run as a module so the ``tools.local_labels`` import resolves:

    python -m tools.sm_86_predictor_accuracy_recovery_v2.generate_cohort_split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_LOCAL_LABELS_DIR = _REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS_DIR))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from corpus_cell_identity import corpus_cell_key  # noqa: E402
import validate_cohort_split as vcs  # noqa: E402

_DEFAULT_MANIFEST_PATH = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_manifest.json"
)
_DEFAULT_SPLIT_PATH = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_split.json"
)

# Per-split floors. The submission-track split contract requires
# calibration >= 15 cells, validation >= 8 cells, holdout >= 8 cells
# with >= 8 unique holdout slugs. The calibration unique-slug floor is
# 10 (not the pre-relabel
# 14): the live-reconciled admitted union spans 28 unique slugs, so
# with validation and holdout each consuming 8 slugs only 12 remain for
# calibration — a 14-slug calibration floor is arithmetically
# unsatisfiable and was a pre-relabel artifact, not a plan requirement.
_SPLIT_FLOORS = {
    "calibration": {"min_cell_floor": 15, "min_unique_slug_floor": 10},
    "validation": {"min_cell_floor": 8, "min_unique_slug_floor": 8},
    "holdout": {"min_cell_floor": 8, "min_unique_slug_floor": 8},
}

# Targets that cap the single-cell round-robin so calibration stays the
# largest block. Holdout/validation each take this many single-cell
# slugs; the remainder (plus every multi-cell slug) lands in calibration.
_HOLDOUT_TARGET = 8
_VALIDATION_TARGET = 8

_SPLIT_NAMES = ("calibration", "validation", "holdout")


def assign_splits(admitted_cells: list[Mapping[str, Any]]) -> dict[str, str]:
    """Return ``{slug: split_name}`` for every admitted slug via the
    deterministic family-stratified policy documented in the module
    docstring.

    Every cell of a slug shares the slug's split (the split-by-slug
    invariant); the caller maps the slug assignment onto each cell.
    Raises ``ValueError`` if a slug's cells disagree on family_bucket.
    """
    by_slug: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in admitted_cells:
        by_slug[cell["slug"]].append(cell)

    slug_family: dict[str, str] = {}
    for slug, cells in by_slug.items():
        families = {c["family_bucket"] for c in cells}
        if len(families) != 1:
            raise ValueError(
                f"generate_cohort_split: slug {slug!r} has inconsistent "
                f"family_bucket across its cells: {sorted(families)!r}"
            )
        slug_family[slug] = next(iter(families))

    multi_slugs = sorted(s for s, cs in by_slug.items() if len(cs) > 1)
    single_slugs = sorted(
        (s for s, cs in by_slug.items() if len(cs) == 1),
        key=lambda s: (slug_family[s], s),
    )

    assignment: dict[str, str] = {s: "calibration" for s in multi_slugs}

    counts = {name: 0 for name in _SPLIT_NAMES}
    caps = {"holdout": _HOLDOUT_TARGET, "validation": _VALIDATION_TARGET}
    cycle = ["holdout", "validation", "calibration"]
    cursor = 0
    for slug in single_slugs:
        placed = False
        for _ in range(len(cycle)):
            bucket = cycle[cursor % len(cycle)]
            cursor += 1
            if bucket in caps and counts[bucket] >= caps[bucket]:
                continue
            assignment[slug] = bucket
            counts[bucket] += 1
            placed = True
            break
        if not placed:
            assignment[slug] = "calibration"
            counts["calibration"] += 1
    return assignment


def _canonical_manifest_digest(manifest_payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        manifest_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _split_cell_from_manifest(cell: Mapping[str, Any]) -> dict:
    """Project a manifest admitted cell onto the split's
    cell_membership_entry shape (additive register_cap/variant_id for a
    build-variant cell)."""
    out = {
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "arch": cell["arch"],
        "device": cell["device"],
        "split_assignment": cell["split_assignment"],
        "identity_tuple": cell["identity_tuple"],
    }
    for key in ("register_cap", "variant_id"):
        if cell.get(key) is not None:
            out[key] = cell[key]
    return out


def build_split_payload(manifest_payload: Mapping[str, Any]) -> dict:
    """Build the cohort_split payload from the paired manifest's
    admitted cells, grouping by each admitted cell's ``split_assignment``
    field. The manifest is the authority; this is a pure projection +
    floor/digest stamping. Does NOT validate (the caller validates)."""
    admitted = [
        c for c in manifest_payload["cells"]
        if c["admission_status"] == "admitted"
    ]
    blocks: dict[str, list[dict]] = {name: [] for name in _SPLIT_NAMES}
    for cell in admitted:
        sname = cell["split_assignment"]
        if sname not in blocks:
            raise ValueError(
                f"generate_cohort_split: admitted cell "
                f"{corpus_cell_key(cell, with_arch=False)!r} has invalid "
                f"split_assignment={sname!r}"
            )
        blocks[sname].append(_split_cell_from_manifest(cell))

    # Deterministic ordering within each block: (slug, shape_point, variant).
    for cells in blocks.values():
        cells.sort(key=lambda c: corpus_cell_key(c, with_arch=False))

    splits = {
        name: {
            "cells": blocks[name],
            "min_cell_floor": _SPLIT_FLOORS[name]["min_cell_floor"],
            "min_unique_slug_floor": _SPLIT_FLOORS[name]["min_unique_slug_floor"],
        }
        for name in _SPLIT_NAMES
    }
    return {
        "schema_version": 1,
        "frozen_at_commit": manifest_payload["frozen_at_commit"],
        "cohort_manifest_digest": _canonical_manifest_digest(manifest_payload),
        "splits": splits,
    }


def generate(
    manifest_path: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    """Load the manifest, build the split, VALIDATE it (fail-closed),
    and write it to ``out_path``. Returns the split payload."""
    manifest_path = manifest_path or _DEFAULT_MANIFEST_PATH
    out_path = out_path or _DEFAULT_SPLIT_PATH
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = build_split_payload(manifest_payload)
    # Fail closed: a generated split that violates any cross-cell
    # invariant (disjointness, membership, split-by-slug, floors, 1/3
    # balancing, family-balance, identity-binding, digest) must not be
    # written.
    vcs.validate_cohort_split(split_payload, manifest_payload)
    out_path.write_text(
        json.dumps(split_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return split_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cohort_split.json deterministically from the "
            "admitted cohort in cohort_manifest.json (validates "
            "fail-closed before writing)."
        )
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=_DEFAULT_SPLIT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    split = generate(manifest_path=args.manifest, out_path=args.out)
    counts = {
        name: len(split["splits"][name]["cells"]) for name in _SPLIT_NAMES
    }
    sys.stdout.write(
        f"generate_cohort_split: wrote {args.out} "
        f"(calibration={counts['calibration']} "
        f"validation={counts['validation']} holdout={counts['holdout']}; "
        f"schema+invariants ok)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
