"""sm_120 L2 cohort roster validator.

Validates `data/local_labels/sm120_cohort_expansion_roster.toml` against
the diversity, schema, and split constraints declared in the cohort
plan. The validator pairs with
`data/local_labels/sm120_cohort_family_buckets.toml` (the raw-family ->
derived-bucket mapping).

Checks performed:

- Schema. Each row carries the 12 required fields with the correct
  types. `arch_gate` includes "sm_120". `level == "L2"` (the cohort
  IS L2; the source row's `level` is preserved via the inventory's
  source_row_level provenance field but not echoed in the cohort
  roster). `family` is present in the family-bucket mapping.
  `family_bucket` is in the closed enum
  {memory_stress, tensor_compute, activation_norm, other}. `split` is in
  the closed enum {calibration, validation, holdout}.

- Family-bucket mapping consistency. Every row's
  (family, family_bucket) pair is consistent with the
  sm120_cohort_family_buckets.toml mapping.

- Diversity. Among the 30 primary rows:
  - No `family_bucket` exceeds 35% (i.e. 10 of 30).
  - `memory_stress`, `tensor_compute`, `activation_norm` each have
    >=6 rows.
  - Rows tagged "other" count toward the 30 but not toward any
    minimum.

- Split counts. Among primary rows: exactly 15 `calibration`,
  8 `validation`, 7 `holdout`. Each split contains >=1 row from each
  required bucket (memory_stress, tensor_compute, activation_norm) so
  no slice is bucket-degenerate.

- Replacement pool. Exactly 10 replacement rows. Replacement rows have
  `split = "replacement"` (a reserved value distinct from the
  calibration/validation/holdout primary splits).

- Roster-preservation pinning. When the caller supplies an expected
  SHA256 of the roster file, the validator recomputes the SHA256 and
  rejects when they differ.

Usage:

    python -m tools.local_labels.sm120_cohort_roster_validator \\
        --roster data/local_labels/sm120_cohort_expansion_roster.toml \\
        --family-buckets data/local_labels/sm120_cohort_family_buckets.toml \\
        [--pinned-sha256 <hex>]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from pathlib import Path
from typing import Any

__version__ = "1.0.0"

REQUIRED_FIELDS = (
    "slug",
    "shape_point",
    "family",
    "family_bucket",
    "level",
    "arch_gate",
    "split",
    "binary_path",
    "manifest_path",
    "trace_json_path",
    "roster_source",
    "candidate_order",
)

ALLOWED_FAMILY_BUCKETS = frozenset(
    {"memory_stress", "tensor_compute", "activation_norm", "other"}
)

REQUIRED_BUCKET_MIN = 6
BUCKET_CAP_PCT = 35
PRIMARY_COUNT = 30
REPLACEMENT_COUNT = 10
CALIBRATION_COUNT = 15
VALIDATION_COUNT = 8
HOLDOUT_COUNT = 7
PRIMARY_SPLITS = frozenset({"calibration", "validation", "holdout"})
REPLACEMENT_SPLIT = "replacement"
REQUIRED_BUCKETS = frozenset({"memory_stress", "tensor_compute", "activation_norm"})


class CohortValidationError(Exception):
    """Raised when the cohort roster fails one or more checks. Carries
    a list of human-readable defects; the CLI surface prints them in
    order."""

    def __init__(self, defects: list[str]) -> None:
        super().__init__(f"{len(defects)} cohort validation defects")
        self.defects = list(defects)


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_string_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _validate_schema_per_row(
    row: dict,
    idx: int,
    family_bucket_map: dict[str, str],
    defects: list[str],
) -> None:
    label = f"row[{idx}]"
    for key in REQUIRED_FIELDS:
        if key not in row:
            defects.append(f"{label}: missing required field {key!r}")
            return  # downstream checks below would all fail; emit one defect

    slug = row["slug"]
    label = f"row[{idx}] slug={slug!r}"

    if not isinstance(row["shape_point"], str) or not row["shape_point"]:
        defects.append(f"{label}: shape_point must be a non-empty string")
    if not isinstance(row["family"], str) or not row["family"]:
        defects.append(f"{label}: family must be a non-empty string")
    if row.get("family_bucket") not in ALLOWED_FAMILY_BUCKETS:
        defects.append(
            f"{label}: family_bucket {row['family_bucket']!r} is not in "
            f"the closed enum {sorted(ALLOWED_FAMILY_BUCKETS)}"
        )
    if row.get("level") != "L2":
        defects.append(
            f"{label}: level must be 'L2' for the L2 cohort; got "
            f"{row['level']!r}"
        )
    if not _is_string_list(row.get("arch_gate")):
        defects.append(f"{label}: arch_gate must be a list of strings")
    elif "sm_120" not in row["arch_gate"]:
        defects.append(
            f"{label}: arch_gate {row['arch_gate']} does not include 'sm_120'"
        )

    sp = row.get("split")
    if sp not in PRIMARY_SPLITS and sp != REPLACEMENT_SPLIT:
        defects.append(
            f"{label}: split {sp!r} is not one of "
            f"{sorted(PRIMARY_SPLITS) + [REPLACEMENT_SPLIT]}"
        )

    if row.get("roster_source") not in ("simbench_bw", "portable_lcd"):
        defects.append(
            f"{label}: roster_source {row.get('roster_source')!r} is not "
            f"'simbench_bw' or 'portable_lcd'"
        )

    if not isinstance(row.get("candidate_order"), int):
        defects.append(f"{label}: candidate_order must be an integer")

    # AC-2.4 mapping consistency: family -> family_bucket lockstep.
    fam = row.get("family")
    actual_bucket = row.get("family_bucket")
    if isinstance(fam, str):
        expected_bucket = family_bucket_map.get(fam)
        if expected_bucket is None:
            defects.append(
                f"{label}: family {fam!r} not in family-bucket mapping; add a "
                f"row to sm120_cohort_family_buckets.toml [family_to_bucket]"
            )
        elif actual_bucket is not None and expected_bucket != actual_bucket:
            defects.append(
                f"{label}: family {fam!r} maps to bucket {expected_bucket!r} "
                f"per sm120_cohort_family_buckets.toml; row has family_bucket "
                f"{actual_bucket!r} (mismatch)"
            )


def _validate_diversity_and_splits(
    rows: list[dict],
    defects: list[str],
) -> None:
    primary = [r for r in rows if r.get("split") in PRIMARY_SPLITS]
    replacement = [r for r in rows if r.get("split") == REPLACEMENT_SPLIT]

    if len(primary) != PRIMARY_COUNT:
        defects.append(
            f"primary cohort count: expected {PRIMARY_COUNT}, got {len(primary)}"
        )
    if len(replacement) != REPLACEMENT_COUNT:
        defects.append(
            f"replacement pool count: expected {REPLACEMENT_COUNT}, got "
            f"{len(replacement)}"
        )

    if len(primary) != PRIMARY_COUNT:
        return

    # AC-2.2: family-bucket cap and minimums (primary only)
    bucket_counts: dict[str, int] = {}
    for r in primary:
        b = r.get("family_bucket")
        if isinstance(b, str):
            bucket_counts[b] = bucket_counts.get(b, 0) + 1

    cap_count = (BUCKET_CAP_PCT * PRIMARY_COUNT) // 100  # 10 for 35% of 30
    for bucket, count in bucket_counts.items():
        if count > cap_count:
            defects.append(
                f"family_bucket {bucket!r} exceeds {BUCKET_CAP_PCT}% cap "
                f"({count} > {cap_count}) in primary cohort"
            )
    for required in REQUIRED_BUCKETS:
        if bucket_counts.get(required, 0) < REQUIRED_BUCKET_MIN:
            defects.append(
                f"family_bucket {required!r} minimum {REQUIRED_BUCKET_MIN} "
                f"not met in primary cohort (got "
                f"{bucket_counts.get(required, 0)})"
            )

    # AC-2.3: split counts and per-split per-bucket minimums (primary only)
    by_split: dict[str, list[dict]] = {"calibration": [], "validation": [], "holdout": []}
    for r in primary:
        sp = r.get("split")
        if sp in by_split:
            by_split[sp].append(r)

    expected_split_counts = {
        "calibration": CALIBRATION_COUNT,
        "validation": VALIDATION_COUNT,
        "holdout": HOLDOUT_COUNT,
    }
    for split_name, expected in expected_split_counts.items():
        actual = len(by_split[split_name])
        if actual != expected:
            defects.append(
                f"split {split_name!r}: expected {expected} rows, got {actual}"
            )

    for split_name, split_rows in by_split.items():
        per_split_buckets = {r.get("family_bucket") for r in split_rows}
        for required in REQUIRED_BUCKETS:
            if required not in per_split_buckets:
                defects.append(
                    f"split {split_name!r}: required family_bucket "
                    f"{required!r} has 0 rows (each split must have >=1 row "
                    f"in each of {sorted(REQUIRED_BUCKETS)})"
                )

    # Uniqueness within primary: (slug, shape_point) must not repeat.
    seen: set[tuple[str, str]] = set()
    for r in primary:
        key = (r.get("slug", ""), r.get("shape_point", ""))
        if key in seen:
            defects.append(
                f"primary cohort contains duplicate (slug, shape_point) "
                f"{key!r}; the cohort identity unit is (slug, shape_point) "
                f"and duplicates are rejected"
            )
        seen.add(key)


def validate_roster(
    roster_path: Path,
    family_buckets_path: Path,
    pinned_sha256: str | None = None,
) -> None:
    """Validate the cohort roster. Raise CohortValidationError if any
    check fails. The exception's .defects attribute lists all failures.
    """
    defects: list[str] = []

    if pinned_sha256 is not None:
        actual_sha = _sha256_file(roster_path)
        if actual_sha.lower() != pinned_sha256.lower():
            defects.append(
                f"roster SHA256 drift: expected {pinned_sha256.lower()}, "
                f"got {actual_sha}"
            )

    roster = _load_toml(roster_path)
    fbm = _load_toml(family_buckets_path)
    mapping = dict(fbm.get("family_to_bucket", {}))
    bucket_vocab = list(fbm.get("family_buckets", []))
    if set(bucket_vocab) != ALLOWED_FAMILY_BUCKETS:
        defects.append(
            f"family-buckets file declares vocabulary {sorted(bucket_vocab)}; "
            f"validator expects {sorted(ALLOWED_FAMILY_BUCKETS)}"
        )

    rows = roster.get("rows")
    if not isinstance(rows, list) or not rows:
        defects.append("roster has no [[rows]] entries")
        raise CohortValidationError(defects)

    for idx, row in enumerate(rows):
        _validate_schema_per_row(row, idx, mapping, defects)

    _validate_diversity_and_splits(rows, defects)

    if defects:
        raise CohortValidationError(defects)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roster",
        required=True,
        type=Path,
        help="Path to data/local_labels/sm120_cohort_expansion_roster.toml",
    )
    p.add_argument(
        "--family-buckets",
        required=True,
        type=Path,
        help="Path to data/local_labels/sm120_cohort_family_buckets.toml",
    )
    p.add_argument(
        "--pinned-sha256",
        type=str,
        default=None,
        help=(
            "Optional expected SHA256 of the roster file. When supplied, the "
            "validator recomputes the SHA256 and rejects on drift "
            "(AC-2.5 roster-preservation pinning)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        validate_roster(args.roster, args.family_buckets, args.pinned_sha256)
    except CohortValidationError as e:
        for d in e.defects:
            print(f"DEFECT: {d}", file=sys.stderr)
        print(
            f"FAIL: {len(e.defects)} validation defects in cohort roster",
            file=sys.stderr,
        )
        return 2
    print("OK: cohort roster validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
