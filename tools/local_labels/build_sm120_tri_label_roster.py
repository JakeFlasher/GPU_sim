"""Deterministic generator for the frozen sm_120 tri-label cohort roster.

Selects a >=50-cell `(arch=sm_120, slug, shape_point)` roster from the
candidate inventory (`data/local_labels/sm120_cohort_inventory.json`)
honoring the integrity gates the roster validator enforces:

- per-slug cap: at most ``PER_SLUG_CAP`` shape_points per slug,
- family-bucket minima: each required bucket carries at least the minimum,
- bucket ceiling: no bucket exceeds ``BUCKET_CEILING_FRACTION`` of the roster,
- slug-held-out split: each slug is entirely calibration OR holdout.

The roster is the PRE-collection plan of which cells the cohort targets; it
pins the inventory SHA-256 so the selection is reproducible and
commit-pinned. The actual per-cell tri-label admission (trace + L2 + L1 +
duration evidence present) is gated separately by the validator's
source-artifact preflight once the live collection produces the labels.

Run once and commit the emitted roster:
``python tools/local_labels/build_sm120_tri_label_roster.py \
    --inventory data/local_labels/sm120_cohort_inventory.json \
    --out data/local_labels/sm120_tri_label_roster.json``
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

ROSTER_SCHEMA_VERSION = 1
ROSTER_ID = "sm120_tri_label_v1"
ARCH = "sm_120"

# The three family buckets that carry coverage minima.
REQUIRED_BUCKETS: tuple[str, ...] = (
    "memory_stress",
    "tensor_compute",
    "activation_norm",
)
# At most this many shape_points per slug, so depth does not collapse the
# cohort onto a few slugs.
PER_SLUG_CAP = 4
# Per-bucket selection target. Three buckets * 18 = 54 cells (>= the 50
# floor), each bucket at 33.3% (< the 35% ceiling) and well above the 6
# minimum.
TARGET_PER_BUCKET = 18
# Fraction of each bucket's slugs assigned to the held-out split.
HOLDOUT_FRACTION = 0.3

# Cell fields copied from the inventory candidate into the roster so the
# validator's source-artifact preflight can resolve each cell's evidence.
_CELL_FIELDS = (
    "slug",
    "shape_point",
    "family_bucket",
    "binary_path",
    "trace_json_path",
    "manifest_path",
    "roster_source",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_source_commit() -> str:
    """Return the git commit the roster was generated against (the parent
    of the commit that lands the roster), or a well-formed all-zero
    sentinel when git is unavailable. This is the provenance pin the
    validator requires so a hand-edited or stale roster cannot pass."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "0" * 40
    sha = proc.stdout.strip()
    if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
        return sha
    return "0" * 40


def select_cells(candidates: list[dict]) -> list[dict]:
    """Deterministically select cells honoring the per-slug cap and the
    per-bucket target, restricted to the required buckets on sm_120."""
    pool = [
        c
        for c in candidates
        if c.get("family_bucket") in REQUIRED_BUCKETS
        and ARCH in c.get("arch_gate", [])
    ]
    pool.sort(key=lambda c: (c["family_bucket"], c["slug"], c["shape_point"]))
    selected: list[dict] = []
    per_slug: collections.Counter = collections.Counter()
    for bucket in REQUIRED_BUCKETS:
        taken = 0
        for cand in pool:
            if cand["family_bucket"] != bucket:
                continue
            if taken >= TARGET_PER_BUCKET:
                break
            if per_slug[cand["slug"]] >= PER_SLUG_CAP:
                continue
            selected.append(cand)
            per_slug[cand["slug"]] += 1
            taken += 1
    return selected


def assign_holdout_slugs(selected: list[dict]) -> set[str]:
    """Return the set of slugs assigned to the held-out split. Within each
    bucket the first ``HOLDOUT_FRACTION`` of the (sorted) slugs go to
    holdout, guaranteeing both splits are represented in every bucket
    (each slug belongs to exactly one bucket)."""
    slugs_by_bucket: dict[str, list[str]] = collections.defaultdict(list)
    seen: set[str] = set()
    for cell in selected:
        if cell["slug"] in seen:
            continue
        seen.add(cell["slug"])
        slugs_by_bucket[cell["family_bucket"]].append(cell["slug"])
    holdout: set[str] = set()
    for bucket in REQUIRED_BUCKETS:
        slugs = sorted(slugs_by_bucket.get(bucket, []))
        n_holdout = max(1, round(len(slugs) * HOLDOUT_FRACTION))
        # Keep at least one calibration slug per bucket.
        n_holdout = min(n_holdout, max(0, len(slugs) - 1))
        holdout.update(slugs[:n_holdout])
    return holdout


def build_roster(inventory_path: Path) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    selected = select_cells(candidates)
    holdout_slugs = assign_holdout_slugs(selected)

    cells: list[dict] = []
    for cand in selected:
        cell = {field: cand[field] for field in _CELL_FIELDS if field in cand}
        cell["split"] = "holdout" if cand["slug"] in holdout_slugs else "calibration"
        cells.append(cell)
    cells.sort(key=lambda c: (c["family_bucket"], c["slug"], c["shape_point"]))

    per_bucket = collections.Counter(c["family_bucket"] for c in cells)
    per_split = collections.Counter(c["split"] for c in cells)
    return {
        "schema_version": ROSTER_SCHEMA_VERSION,
        "roster_id": ROSTER_ID,
        "arch": ARCH,
        "per_slug_cap": PER_SLUG_CAP,
        "required_buckets": list(REQUIRED_BUCKETS),
        "bucket_min": 6,
        "bucket_ceiling_fraction": 0.35,
        "inventory_sha256": file_sha256(inventory_path),
        "source_commit": resolve_source_commit(),
        "total_cells": len(cells),
        "cells_per_bucket": {b: per_bucket.get(b, 0) for b in REQUIRED_BUCKETS},
        "cells_per_split": {s: per_split.get(s, 0) for s in ("calibration", "holdout")},
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/local_labels/sm120_cohort_inventory.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/local_labels/sm120_tri_label_roster.json"),
    )
    args = parser.parse_args(argv)
    roster = build_roster(args.inventory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(roster, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"wrote {args.out} with {roster['total_cells']} cells "
        f"(per bucket: {roster['cells_per_bucket']}; "
        f"per split: {roster['cells_per_split']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
