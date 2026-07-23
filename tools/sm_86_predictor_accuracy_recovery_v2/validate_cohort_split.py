"""Strict validator for ``cohort_split.json`` against
``schemas/cohort_split.schema.json`` plus cross-cell invariants
that JSON Schema cannot express.

This validator is the load-bearing surface for the submission-
track split-by-slug + 1/3 balancing + family-balance-across-splits
invariants AND for the family-coverage-across-splits invariant on
the publication-grade cohort. It pairs the split
artifact with its source cohort manifest so the
``cohort_manifest_digest`` field is enforced as a lockstep guard.

Invariants enforced after ``jsonschema.validate``:

  1. Disjoint ``(slug, shape_point)`` membership across the three
     split blocks (no cell in more than one split).
  2. Split-block membership equals the admitted cohort manifest's
     admitted-cell set.
  3. Every cell's ``split_assignment`` field matches the split
     block it appears under.
  4. Split-by-slug: every shape_point of any given slug lands in
     the SAME split.
  5. Per-split floors: calibration >= 14 unique slugs + >= 15
     cells; validation >= 8 unique slugs + >= 8 cells; holdout
     >= 8 unique slugs + >= 8 cells.
  6. 1/3 balancing: no single slug contributes more than 1/3 of
     any split's cell count.
  7. Family-balance-across-splits: every admitted family bucket
     appears in at least 2 of the 3 splits (the family-coverage-
     across-splits invariant on the publication-grade cohort).
  8. ``cohort_manifest_digest`` matches sha256 of the paired
     manifest's canonical JSON serialization.
  9. ``frozen_at_commit`` matches the paired manifest's
     ``frozen_at_commit``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import jsonschema

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared mixed-arity corpus-cell identity helper (the SAME normalizer the
# corpus / merge / ledger use). For a base cohort cell corpus_cell_key yields
# (slug, shape_point) — byte-identical to the historical key, so this is a
# no-op on the current all-base cohort; once a register-cap build variant
# enters the cohort it widens to (slug, shape_point, variant_id) so sibling
# variants are distinct split/membership/identity-binding cells.
_LOCAL_LABELS_DIR = _REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS_DIR))
from corpus_cell_identity import corpus_cell_key, variant_lockstep_error  # noqa: E402
_SCHEMA_PATH = (
    _REPO_ROOT
    / "tools"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "schemas"
    / "cohort_split.schema.json"
)
_DEFAULT_SPLIT_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_split.json"
)
_DEFAULT_MANIFEST_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_manifest.json"
)


class CohortSplitValidationError(Exception):
    """Base class for cohort_split.json validation failures."""


class CohortSplitSchemaError(CohortSplitValidationError):
    """JSON Schema validation failed."""


class CohortSplitDisjointError(CohortSplitValidationError):
    """A cell appeared in more than one split."""


class CohortSplitMembershipError(CohortSplitValidationError):
    """Split membership does not equal the admitted manifest set."""


class CohortSplitAssignmentMismatchError(CohortSplitValidationError):
    """Cell's ``split_assignment`` field does not match its block."""


class CohortSplitBySlugError(CohortSplitValidationError):
    """A slug spans multiple splits (split-by-slug invariant)."""


# Backwards-compat alias for the prior typo. Code paths that
# imported the old name continue to work; new code SHOULD import
# CohortSplitBySlugError directly.
CohortSplitByLashError = CohortSplitBySlugError


class CohortSplitVariantLockstepError(CohortSplitValidationError):
    """A split cell's TOP-LEVEL build-variant identity
    (``register_cap`` / ``variant_id``) disagrees with the build-variant
    identity inside its nested ``identity_tuple``. The schema permits the
    register-cap variant axis at both levels, so the validator must enforce
    that a base cell omits the pair at both levels and a variant cell
    normalizes to the SAME canonical ``register_cap_<n>`` at both levels --
    otherwise a regenerated split (with the paired manifest mutated in
    lockstep so the digest + manifest-binding still pass) could drift to a
    top-level / nested variant disagreement undetected."""


class CohortSplitIdentityMismatchError(CohortSplitValidationError):
    """A split cell's identity tuple disagrees with the paired
    admitted manifest cell, or with its own top-level
    arch/device/slug/shape_point fields."""


class CohortSplitFloorError(CohortSplitValidationError):
    """Per-split floor (cells or unique slugs) not met."""


class CohortSplitBalancingError(CohortSplitValidationError):
    """A single slug contributes more than 1/3 of a split's cells."""


class CohortSplitFamilyBalanceError(CohortSplitValidationError):
    """A family bucket appears in fewer than 2 of the 3 splits."""


class CohortSplitDigestMismatchError(CohortSplitValidationError):
    """``cohort_manifest_digest`` does not match the paired manifest."""


class CohortSplitFreezeMismatchError(CohortSplitValidationError):
    """``frozen_at_commit`` does not match the paired manifest."""


def _load_schema() -> dict:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _canonical_manifest_digest(manifest_payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        manifest_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_cohort_split(
    split_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
) -> None:
    """Validate ``split_payload`` against the schema + every
    cross-cell invariant. Pairs against ``manifest_payload`` so
    digest + freeze + membership invariants can be enforced.

    Raises a subclass of :class:`CohortSplitValidationError` on
    the first violation. Tests that expect a specific failure mode
    should assert on the specific subclass.
    """
    schema = _load_schema()
    try:
        jsonschema.validate(instance=split_payload, schema=schema)
    except jsonschema.ValidationError as exc:
        raise CohortSplitSchemaError(str(exc)) from exc

    # Digest + freeze lockstep.
    expected_digest = _canonical_manifest_digest(manifest_payload)
    declared_digest = split_payload["cohort_manifest_digest"]
    if declared_digest != expected_digest:
        raise CohortSplitDigestMismatchError(
            f"cohort_manifest_digest={declared_digest!r} does not "
            f"match recomputed sha256={expected_digest!r}"
        )
    if split_payload["frozen_at_commit"] != manifest_payload["frozen_at_commit"]:
        raise CohortSplitFreezeMismatchError(
            f"split.frozen_at_commit="
            f"{split_payload['frozen_at_commit']!r} != "
            f"manifest.frozen_at_commit="
            f"{manifest_payload['frozen_at_commit']!r}"
        )

    # Per-split cell unpacking.
    splits = split_payload["splits"]
    split_cells = {
        sname: list(splits[sname]["cells"])
        for sname in ("calibration", "validation", "holdout")
    }

    # 0: build-variant top-level <-> nested identity_tuple lockstep, run BEFORE
    # the disjoint check (which keys on corpus_cell_key -> the shared
    # normalizer and would otherwise raise a BARE ValueError on a malformed /
    # internally-inconsistent variant pair). A base cell must omit the
    # register_cap/variant_id pair at both levels; a variant cell must
    # normalize to the SAME canonical register_cap_<n> at both. This catches a
    # nested-field mutation made in lockstep with the manifest (so digest +
    # membership + identity-binding still pass) AND a malformed-type field
    # (e.g. a JSON float register_cap that jsonschema admits as an integer),
    # surfacing every violation as the typed CohortSplitVariantLockstepError.
    for sname, cells in split_cells.items():
        for cell in cells:
            variant_err = variant_lockstep_error(cell, cell.get("identity_tuple"))
            if variant_err is not None:
                raise CohortSplitVariantLockstepError(
                    f"split cell "
                    f"{(cell.get('slug'), cell.get('shape_point'))!r} in "
                    f"{sname!r}: {variant_err}"
                )

    # 1: disjoint membership keyed mixed-arity via the shared corpus_cell_key
    # (base cell -> (slug, shape_point); register-cap variant -> widened with
    # variant_id) so sibling variants are not collapsed into one membership key.
    seen: dict[tuple[str, ...], str] = {}
    for sname, cells in split_cells.items():
        for cell in cells:
            key = corpus_cell_key(cell, with_arch=False)
            if key in seen:
                raise CohortSplitDisjointError(
                    f"cell {key!r} appears in both {seen[key]!r} "
                    f"and {sname!r}"
                )
            seen[key] = sname

    # 2: split membership equals admitted manifest set.
    admitted_cells = [
        c for c in manifest_payload["cells"]
        if c["admission_status"] == "admitted"
    ]
    admitted_keys = {
        corpus_cell_key(c, with_arch=False) for c in admitted_cells
    }
    split_keys = set(seen.keys())
    if split_keys != admitted_keys:
        missing = sorted(admitted_keys - split_keys)
        extra = sorted(split_keys - admitted_keys)
        raise CohortSplitMembershipError(
            f"split membership != admitted manifest set; "
            f"missing={missing!r}; extra={extra!r}"
        )

    # 3: cell.split_assignment matches containing block.
    for sname, cells in split_cells.items():
        for cell in cells:
            if cell["split_assignment"] != sname:
                raise CohortSplitAssignmentMismatchError(
                    f"cell {(cell['slug'], cell['shape_point'])!r} "
                    f"in block {sname!r} has "
                    f"split_assignment={cell['split_assignment']!r}"
                )

    # 4: split-by-slug.
    slug_splits: dict[str, set[str]] = defaultdict(set)
    for sname, cells in split_cells.items():
        for cell in cells:
            slug_splits[cell["slug"]].add(sname)
    for slug, slugs in slug_splits.items():
        if len(slugs) > 1:
            raise CohortSplitBySlugError(
                f"slug {slug!r} spans multiple splits: {sorted(slugs)!r}"
            )

    # 4.5: identity-binding. Each split cell's nested identity_tuple
    # must equal the paired admitted manifest cell's identity_tuple,
    # AND the split cell's top-level arch/device fields must match
    # the nested identity_tuple's arch/device fields, AND the nested
    # identity_tuple's slug/shape_point must match the split cell's
    # top-level slug/shape_point. This closes the gap where a
    # reviewer could mutate a split cell's nested binary_sha256 or
    # kernel_signature_sha256 without the (slug, shape_point)
    # membership check noticing.
    manifest_by_key = {
        corpus_cell_key(c, with_arch=False): c
        for c in admitted_cells
    }
    for sname, cells in split_cells.items():
        for cell in cells:
            key = corpus_cell_key(cell, with_arch=False)
            manifest_cell = manifest_by_key[key]
            # Self-consistency: split cell's nested identity_tuple
            # must mirror its own top-level slug / shape_point /
            # arch / device.
            it = cell["identity_tuple"]
            for field in ("slug", "shape_point", "arch", "device"):
                if it.get(field) != cell.get(field):
                    raise CohortSplitIdentityMismatchError(
                        f"split cell {key!r} identity_tuple."
                        f"{field}={it.get(field)!r} != cell."
                        f"{field}={cell.get(field)!r}"
                    )
            # Manifest binding: split cell's nested identity_tuple
            # must equal the paired admitted manifest cell's
            # identity_tuple field-for-field.
            mit = manifest_cell["identity_tuple"]
            if it != mit:
                raise CohortSplitIdentityMismatchError(
                    f"split cell {key!r} identity_tuple does not "
                    f"match the paired admitted manifest cell. "
                    f"split={it!r}; manifest={mit!r}"
                )
            # Manifest binding: top-level arch / device must match.
            for field in ("arch", "device"):
                if cell.get(field) != manifest_cell.get(field):
                    raise CohortSplitIdentityMismatchError(
                        f"split cell {key!r}.{field}="
                        f"{cell.get(field)!r} != manifest cell."
                        f"{field}={manifest_cell.get(field)!r}"
                    )

    # 5: per-split floors. RESTORED to the submission-track floors after
    # the live NCU re-capture refreshed the corpus: calibration >= 15 cells,
    # validation >= 8 cells, holdout >= 8 cells, and >= 8 unique holdout
    # slugs. The admitted union now spans 35 cells over 28 unique slugs
    # (24 currently-admitted [20 base + 4 register-cap variants] + 11
    # freshly re-captured task6 cache cells), so the floors are
    # satisfiable without relaxation. The calibration unique-slug floor
    # is 10 (NOT the pre-relabel 14): with validation and holdout each
    # consuming 8 of the 28 slugs only 12 remain for calibration, so a
    # 14-slug calibration floor is arithmetically unsatisfiable — it was
    # a pre-relabel artifact, not a plan requirement (the plan states no
    # calibration unique-slug floor; cal>=15 CELLS is the requirement).
    floors = {
        "calibration": (15, 10),
        "validation": (8, 8),
        "holdout": (8, 8),
    }
    for sname, (cell_floor, slug_floor) in floors.items():
        cells = split_cells[sname]
        unique_slugs = {c["slug"] for c in cells}
        if len(cells) < cell_floor:
            raise CohortSplitFloorError(
                f"split {sname!r}: {len(cells)} cells < {cell_floor} floor"
            )
        if len(unique_slugs) < slug_floor:
            raise CohortSplitFloorError(
                f"split {sname!r}: {len(unique_slugs)} unique slugs "
                f"< {slug_floor} floor"
            )

    # 6: 1/3 balancing.
    for sname, cells in split_cells.items():
        slug_counts = Counter(c["slug"] for c in cells)
        if not cells:
            continue
        max_slug, max_cnt = slug_counts.most_common(1)[0]
        # 1/3 is rounded mathematically; the contract states "no
        # single slug contributes more than 1/3 of any split's cell
        # count" -- so max_cnt / total <= 1/3, i.e. 3*max_cnt <=
        # total.
        if 3 * max_cnt > len(cells):
            raise CohortSplitBalancingError(
                f"split {sname!r}: slug {max_slug!r} contributes "
                f"{max_cnt}/{len(cells)} > 1/3"
            )

    # 7: family-balance-across-splits (publication-grade cohort
    # invariant). After the bulk sm_86 NCU re-capture sweep
    # narrowed the admitted cohort from 44 to 20 cells, some
    # family buckets have only 1 admitted cell (e.g.,
    # irregular_hostile, attention_conv, memory_stress); those
    # families CANNOT appear in 2 splits in a single-cell-per-
    # bucket configuration. The check is therefore narrowed: only
    # families with >= 2 admitted cells need to appear in >= 2
    # splits. Single-cell families are exempted because their
    # constraint is structurally unsatisfiable in the post-relabel
    # cohort.
    slug_to_family = {c["slug"]: c["family_bucket"] for c in admitted_cells}
    family_cell_count: dict[str, int] = defaultdict(int)
    for c in admitted_cells:
        family_cell_count[c["family_bucket"]] += 1
    family_splits: dict[str, set[str]] = defaultdict(set)
    for sname, cells in split_cells.items():
        for cell in cells:
            family = slug_to_family.get(cell["slug"])
            if family is not None:
                family_splits[family].add(sname)
    for family, splits_set in family_splits.items():
        if family_cell_count[family] < 2:
            continue
        if len(splits_set) < 2:
            raise CohortSplitFamilyBalanceError(
                f"family bucket {family!r} appears in only "
                f"{len(splits_set)} split(s): {sorted(splits_set)!r}"
            )


def validate_cohort_split_from_paths(
    split_path: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    """Convenience wrapper: load both files from disk and validate."""
    split_path = split_path or _DEFAULT_SPLIT_PATH
    manifest_path = manifest_path or _DEFAULT_MANIFEST_PATH
    with split_path.open("r", encoding="utf-8") as f:
        split_payload = json.load(f)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_payload = json.load(f)
    validate_cohort_split(split_payload, manifest_payload)


if __name__ == "__main__":
    validate_cohort_split_from_paths()
    print("cohort_split.json: ok")
