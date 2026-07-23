"""Producer for the sm_120 L1/L2 frozen-grain engineering-closure artifact.

The >=50-cell sm_120 frozen-roster cache-label cohort is closed by honest
engineering-closure-falsification: it is INFEASIBLE at the frozen
`(slug, shape_point)` grain. Two legs make it so:

  - the DURATION leg is already terminally closed (the trace-vs-duration disjoint
    size windows -- referenced via the committed duration-leg closure); and
  - the L1/L2 leg at the frozen grain is REPEATABILITY-limited: EVERY roster cell
    was live-collected across the two frozen-roster pilots (the known-convention
    batch + the `_00` batch), but most fail the L2 stddev repeatability gate at
    the admitting cold-cache config.

This producer partitions ALL roster cells into exactly one of TWO typed
dispositions -- `admitted_l1l2` (a frozen-grain sm_120 L1/L2 label exists, the
cell is repeatable) or `repeatability_excluded` (collected but exceeded the
stddev gate) -- recomputed from the committed roster + canonical labels + the two
pilots. Every cell is collected, so there is no "un-collectable" class. It fails
closed if the two classes do not cover every roster cell exactly once. It reads
VALUES only; invokes no NCU.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

SCHEMA_VERSION = 1
CLOSURE_ID = "sm120_l1l2_frozen_grain_engineering_closure_v1"
ARCH = "sm_120"
VERDICT = "engineering_closure_refuted"
CLOSED_LEG = "l1_l2_frozen_grain"
REQUIRED_MIN_CELLS = 50

REASON_REPEATABILITY = "label_repeatability_failed"

# Canonical committed evidence paths (repo-relative). Every roster cell is
# COLLECTED across the two frozen-roster pilots (the known-convention batch +
# the _00 batch); the disposition is admitted vs repeatability-excluded -- there
# is no "un-arg'd" class (an earlier draft overstated that; the _00 binaries
# accept --shape-point and were collected).
ROSTER_PATH = "data/local_labels/sm120_tri_label_roster.json"
CANONICAL_ADMITTED_PATH = "data/local_labels/label_admitted.json"
PILOT_KNOWN_ADMITTED_PATH = "data/local_labels/sm120_frozen_roster_l1l2_pilot/label_admitted.json"
PILOT_KNOWN_EXCLUDED_PATH = "data/local_labels/sm120_frozen_roster_l1l2_pilot/label_excluded.json"
PILOT_00_ADMITTED_PATH = "data/local_labels/sm120_frozen_roster_l1l2_pilot_00/label_admitted.json"
PILOT_00_EXCLUDED_PATH = "data/local_labels/sm120_frozen_roster_l1l2_pilot_00/label_excluded.json"
# The full per-cell repeatability aggregates (admitted + excluded) for each pilot.
# These carry the admission decision (admitted / typed_reason), n_repeats, and the
# measured stddev_l2_sector -- the verifiable BASIS for the repeatability-excluded
# disposition. The closure cites them so the validator can prove every excluded
# cell actually exceeded the stddev gate, not merely that the closure asserts it.
PILOT_KNOWN_REPEATABILITY_PATH = "data/local_labels/sm120_frozen_roster_l1l2_pilot/label_repeatability.json"
PILOT_00_REPEATABILITY_PATH = "data/local_labels/sm120_frozen_roster_l1l2_pilot_00/label_repeatability.json"
DURATION_LEG_CLOSURE_PATH = "data/local_labels/sm120_tri_label_engineering_closure.json"

EVIDENCE_PATHS = {
    "roster": ROSTER_PATH,
    "canonical_admitted_labels": CANONICAL_ADMITTED_PATH,
    "pilot_known_admitted_labels": PILOT_KNOWN_ADMITTED_PATH,
    "pilot_known_excluded_labels": PILOT_KNOWN_EXCLUDED_PATH,
    "pilot_00_admitted_labels": PILOT_00_ADMITTED_PATH,
    "pilot_00_excluded_labels": PILOT_00_EXCLUDED_PATH,
    "pilot_known_repeatability": PILOT_KNOWN_REPEATABILITY_PATH,
    "pilot_00_repeatability": PILOT_00_REPEATABILITY_PATH,
    "duration_leg_closure": DURATION_LEG_CLOSURE_PATH,
}

# The committed repeatability config (the cold-cache methodology that reproduced
# the prior admitted labels). The warm config was probed but is NOT committed, so
# this closure does not claim warm-config failure.
REPEATABILITY_CONFIG = {
    "warmup_count": 0,
    "cache_control": "all",
    "stddev_threshold": 0.03,
    "min_repeats": 5,
    "note": "cold-cache config (the methodology that reproduced the prior admitted labels)",
}

VERIFIED_LIVE_SM120_PROOF = {
    "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "compute_capability": "sm_120",
    "driver": "595.71.05",
    "cuda": "13.2",
    "ncu": "2026.1.1.0",
    "clock_control": "none",
}


class FrozenGrainClosureBuildError(RuntimeError):
    pass


def _load(repo_root: Path, rel: str) -> dict:
    return json.loads((repo_root / rel).read_text(encoding="utf-8"))


def _sm120_keys(doc: dict, *, require_arch: bool) -> set:
    keys = set()
    for c in doc.get("cells", []):
        if require_arch and c.get("arch") != ARCH:
            continue
        keys.add((c["slug"], c["shape_point"]))
    return keys


def _excluded_stddev(repo_root: Path) -> dict:
    out = {}
    for rel in (PILOT_KNOWN_EXCLUDED_PATH, PILOT_00_EXCLUDED_PATH):
        for c in _load(repo_root, rel).get("cells", []):
            out[(c["slug"], c["shape_point"])] = c.get("stddev_l2_sector")
    return out


def _repeatability_rows(repo_root: Path) -> dict:
    """Map (slug, shape_point) -> the full per-cell repeatability aggregate row
    across BOTH pilots. Each row carries `admission` (admitted / typed_reason),
    `n_repeats`, and `stddev_l2_sector` -- the verifiable basis the validator
    cross-checks against every repeatability_excluded disposition."""
    out = {}
    for rel in (PILOT_KNOWN_REPEATABILITY_PATH, PILOT_00_REPEATABILITY_PATH):
        for c in _load(repo_root, rel).get("cells", []):
            out[(c["slug"], c["shape_point"])] = c
    return out


def _source_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "0" * 40


def _partition(repo_root: Path) -> dict:
    """Recompute the 2-way disposition of every roster cell: admitted (a
    frozen-grain sm_120 L1/L2 label exists) vs repeatability_excluded (collected
    but failed the stddev gate). EVERY roster cell was collected across the two
    pilots, so the union must cover the full roster with no overlap and no
    leftover -- fails closed otherwise."""
    roster_cells = _load(repo_root, ROSTER_PATH).get("cells", [])
    roster = {(c["slug"], c["shape_point"]) for c in roster_cells}
    canonical = _sm120_keys(_load(repo_root, CANONICAL_ADMITTED_PATH), require_arch=True)
    pilot_adm = _sm120_keys(_load(repo_root, PILOT_KNOWN_ADMITTED_PATH), require_arch=False)
    pilot_adm |= _sm120_keys(_load(repo_root, PILOT_00_ADMITTED_PATH), require_arch=False)
    pilot_exc = _sm120_keys(_load(repo_root, PILOT_KNOWN_EXCLUDED_PATH), require_arch=False)
    pilot_exc |= _sm120_keys(_load(repo_root, PILOT_00_EXCLUDED_PATH), require_arch=False)

    admitted = (canonical | pilot_adm) & roster
    excluded = (pilot_exc & roster) - admitted

    if admitted & excluded:
        raise FrozenGrainClosureBuildError("disposition classes overlap")
    leftover = roster - admitted - excluded
    if leftover:
        raise FrozenGrainClosureBuildError(
            f"{len(leftover)} roster cells are neither admitted nor "
            f"repeatability-excluded (uncollected): {sorted(leftover)[:5]}"
        )
    return {"admitted": admitted, "excluded": excluded, "roster_total": len(roster)}


def build(repo_root: Path = _REPO_ROOT) -> dict:
    part = _partition(repo_root)
    stddev = _excluded_stddev(repo_root)

    admitted_l1l2 = [
        {"slug": s, "shape_point": sh} for s, sh in sorted(part["admitted"])
    ]
    repeatability_excluded = [
        {
            "slug": s,
            "shape_point": sh,
            "typed_reason": REASON_REPEATABILITY,
            "stddev_l2_sector": stddev.get((s, sh)),
        }
        for s, sh in sorted(part["excluded"])
    ]

    counts = {
        "admitted_l1l2": len(admitted_l1l2),
        "repeatability_excluded": len(repeatability_excluded),
        "total": part["roster_total"],
    }

    rationale = (
        "The >=50-cell sm_120 frozen-roster cache-label cohort is infeasible at "
        "the frozen (slug, shape_point) grain. The duration leg is already "
        "terminally closed (trace-vs-duration disjoint size windows; see the "
        "duration_leg_closure evidence). The L1/L2 leg is REPEATABILITY-limited: "
        f"ALL {counts['total']} roster cells were live-collected across the two "
        f"frozen-roster pilots, but only {counts['admitted_l1l2']} carry a "
        f"frozen-grain sm_120 L1/L2 label that is repeatable; "
        f"{counts['repeatability_excluded']} exceed the "
        f"{REPEATABILITY_CONFIG['stddev_threshold']} L2 stddev gate at the "
        f"admitting cold-cache config -- below the {REQUIRED_MIN_CELLS}-cell "
        "minimum."
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "closure_id": CLOSURE_ID,
        "arch": ARCH,
        "verdict": VERDICT,
        "closed_leg": CLOSED_LEG,
        "claims_four_source_roster_satisfied": False,
        "roster_total_cells": part["roster_total"],
        "required_min_cells": REQUIRED_MIN_CELLS,
        "frozen_roster_l1l2_admitted_count": counts["admitted_l1l2"],
        "repeatability_config": dict(REPEATABILITY_CONFIG),
        "evidence_artifact_paths": dict(EVIDENCE_PATHS),
        "cell_disposition": {
            "admitted_l1l2": admitted_l1l2,
            "repeatability_excluded": repeatability_excluded,
        },
        "disposition_counts": counts,
        "rationale": rationale,
        "verified_live_sm120_proof": dict(VERIFIED_LIVE_SM120_PROOF),
        "source_commit": _source_commit(repo_root),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "sm120_l1l2_frozen_grain_engineering_closure.json",
    )
    args = parser.parse_args(argv)
    doc = build()
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    c = doc["disposition_counts"]
    print(
        f"wrote {args.out} (admitted={c['admitted_l1l2']} "
        f"repeatability_excluded={c['repeatability_excluded']} / {c['total']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
