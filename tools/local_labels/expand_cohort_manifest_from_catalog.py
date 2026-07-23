"""Expand ``cohort_manifest.json`` with attempted_excluded entries
for the new candidate slugs catalogued by
``inventory_cohort_candidates.py``.

Reads ``expansion_candidates.json`` (the catalog), reads the
current ``cohort_manifest.json`` (the cohort lock), skips any
candidate whose ``slug`` already appears in the manifest, and
appends a new ``attempted_excluded`` cell per remaining candidate.

Each new cell carries:

  - ``shape_point="timing_shape"`` (canonical default for the
    portable_lcd activation + gemm + attention candidates; the
    operator can override per-cell when running the live sweep
    that admits the cell).
  - ``family_bucket`` from the catalog.
  - ``split_assignment="calibration"`` (the canonical default for
    attempted_excluded cells per the seed manifest).
  - ``expected_trace_path`` pointing at the path the upcoming
    live sweep will populate.
  - ``expected_label_path="data/local_labels/label_admitted.json"``
    (the shared label corpus; per-cell admission tags into this
    corpus when the live sweep completes).
  - ``admission_status="attempted_excluded"`` and
    ``exclusion_typed_reason`` from the catalog
    (``trace_missing`` when no trace exists yet;
    ``label_missing`` when a trace exists but no NCU label has
    been generated).
  - ``identity_tuple`` with the REAL ``binary_sha256`` from the
    catalog (extracted from the on-disk binary), a deterministic
    PLACEHOLDER ``kernel_signature_sha256`` derived as
    ``sha256("pending_kernel_signature:"+slug+":"+shape_point)``
    (the live sweep replaces this with the actual kernel
    signature when the cell is admitted), and ``source_commit``
    matching the manifest's current ``frozen_at_commit``.

Per CLAUDE.md Rule 1: this script reads only on-disk JSON; it
never invokes NCU or any GPU runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG_PATH = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "expansion_candidates.json"
)
_DEFAULT_MANIFEST_PATH = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "cohort_manifest.json"
)

_DEFAULT_SHAPE_POINT = "timing_shape"
_DEFAULT_SPLIT_ASSIGNMENT = "calibration"
_DEFAULT_DEVICE = "RTX 3090 Ti"
_DEFAULT_ARCH = "sm_86"
_DEFAULT_LABEL_PATH = "data/local_labels/label_admitted.json"


def _placeholder_kernel_signature(slug: str, shape_point: str) -> str:
    """Deterministic placeholder for ``kernel_signature_sha256``
    used until the live sweep records the real signature. The
    placeholder is derived as
    ``sha256("pending_kernel_signature:"+slug+":"+shape_point)``
    so it (a) matches the schema's 64-char-hex pattern; (b) is
    stable across runs of this script; (c) is documentably
    distinct from any real kernel signature."""
    seed = f"pending_kernel_signature:{slug}:{shape_point}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _build_attempted_cell(
    *,
    slug: str,
    shape_point: str,
    family_bucket: str,
    split_assignment: str,
    binary_sha256: str,
    source_commit: str,
    typed_reason: str,
) -> dict[str, Any]:
    """Build a single attempted_excluded cell entry for the
    manifest. The identity_tuple's kernel_signature_sha256 is the
    deterministic placeholder; the live sweep replaces it on
    admission."""
    expected_trace = (
        f"workloads/simbench_bw/results/{slug}/rtx_3090_ti/sm_86/"
        f"trace_{shape_point}.json"
    )
    return {
        "slug": slug,
        "shape_point": shape_point,
        "family_bucket": family_bucket,
        "split_assignment": split_assignment,
        "expected_trace_path": expected_trace,
        "expected_label_path": _DEFAULT_LABEL_PATH,
        "admission_status": "attempted_excluded",
        "exclusion_typed_reason": typed_reason,
        "identity_tuple": {
            "slug": slug,
            "shape_point": shape_point,
            "binary_sha256": binary_sha256,
            "kernel_signature_sha256": (
                _placeholder_kernel_signature(slug, shape_point)
            ),
            "source_commit": source_commit,
            "arch": _DEFAULT_ARCH,
            "device": _DEFAULT_DEVICE,
        },
        "arch": _DEFAULT_ARCH,
        "device": _DEFAULT_DEVICE,
    }


def _recompute_counts(
    manifest: Mapping[str, Any],
    expanded_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    expanded_attempted = len(expanded_cells)
    expanded_admitted = sum(
        1 for cell in expanded_cells
        if cell["admission_status"] == "admitted"
    )
    return {
        **manifest,
        "attempted_cell_count": expanded_attempted,
        "admitted_cell_count": expanded_admitted,
        "cells": expanded_cells,
    }


def expand_manifest(
    catalog: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    shape_point: str = _DEFAULT_SHAPE_POINT,
    split_assignment: str = _DEFAULT_SPLIT_ASSIGNMENT,
) -> dict[str, Any]:
    """Return a new manifest payload that retains every existing
    cell + appends one attempted_excluded entry per NEW candidate
    in the catalog. The ``attempted_cell_count`` is recomputed
    against the expanded cells list; the ``admitted_cell_count``
    is unchanged because no entry has been admitted yet."""
    existing_slugs = {cell["slug"] for cell in manifest["cells"]}
    source_commit = manifest["frozen_at_commit"]
    new_entries: list[dict[str, Any]] = []
    for candidate in catalog["candidates"]:
        slug = candidate["slug"]
        if slug in existing_slugs:
            continue
        new_entries.append(_build_attempted_cell(
            slug=slug,
            shape_point=shape_point,
            family_bucket=candidate["family_bucket"],
            split_assignment=split_assignment,
            binary_sha256=candidate["binary_sha256"],
            source_commit=source_commit,
            typed_reason=candidate["proposed_typed_reason"],
        ))
    expanded_cells = list(manifest["cells"]) + new_entries
    return _recompute_counts(manifest, expanded_cells)


def add_depth_first_shapes(
    catalog: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    target_slugs: tuple[str, ...],
    extra_shape_points: tuple[str, ...],
    split_assignment: str = _DEFAULT_SPLIT_ASSIGNMENT,
) -> dict[str, Any]:
    """Return a new manifest payload that adds extra shape_point
    entries for each slug in ``target_slugs``, sourced from the
    catalog's family_bucket + binary_sha256 + proposed typed
    reason. The cohort's depth-first dimension wants >= 6 slugs
    to carry >= 3 shape_points each; this helper appends
    additional attempted_excluded rows under the existing slug's
    identity to reach that floor. ``(slug, shape_point)`` pairs
    already in the manifest are skipped."""
    existing_pairs = {
        (cell["slug"], cell["shape_point"])
        for cell in manifest["cells"]
    }
    catalog_by_slug = {c["slug"]: c for c in catalog["candidates"]}
    source_commit = manifest["frozen_at_commit"]
    new_entries: list[dict[str, Any]] = []
    for slug in target_slugs:
        candidate = catalog_by_slug.get(slug)
        if candidate is None:
            raise ValueError(
                f"slug {slug!r} not in catalog; cannot expand. The "
                f"depth-first expansion requires every target slug "
                f"to have a catalog entry binding its binary_sha256."
            )
        for shape_point in extra_shape_points:
            if (slug, shape_point) in existing_pairs:
                continue
            new_entries.append(_build_attempted_cell(
                slug=slug,
                shape_point=shape_point,
                family_bucket=candidate["family_bucket"],
                split_assignment=split_assignment,
                binary_sha256=candidate["binary_sha256"],
                source_commit=source_commit,
                typed_reason=candidate["proposed_typed_reason"],
            ))
    expanded_cells = list(manifest["cells"]) + new_entries
    return _recompute_counts(manifest, expanded_cells)


def add_planned_vendor_baseline(
    manifest: Mapping[str, Any],
    *,
    slug: str,
    shape_point: str = _DEFAULT_SHAPE_POINT,
    split_assignment: str = _DEFAULT_SPLIT_ASSIGNMENT,
) -> dict[str, Any]:
    """Return a new manifest payload with an
    ``attempted_excluded(build_fail)`` vendor_baselines entry
    appended. The cohort's family-bucket dimension requires at
    least one vendor_baselines cell. When the operator has not
    yet built a sm_86-compatible vendor binary, a planned-
    candidate row keeps the family-bucket count above zero AND
    documents the buildability gap via the closed-enum
    ``exclusion_typed_reason="build_fail"``. The placeholder
    ``binary_sha256`` is derived as
    ``sha256("planned_vendor_baseline:"+slug+":"+shape_point)`` so
    the schema's 64-char-hex pattern is satisfied while the
    placeholder is unmistakable for any real binary hash."""
    import hashlib
    seed = f"planned_vendor_baseline:{slug}:{shape_point}"
    placeholder_binary_sha = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    source_commit = manifest["frozen_at_commit"]
    new_entry = _build_attempted_cell(
        slug=slug,
        shape_point=shape_point,
        family_bucket="vendor_baselines",
        split_assignment=split_assignment,
        binary_sha256=placeholder_binary_sha,
        source_commit=source_commit,
        typed_reason="build_fail",
    )
    expanded_cells = list(manifest["cells"]) + [new_entry]
    return _recompute_counts(manifest, expanded_cells)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    tmp_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Expand cohort_manifest.json with attempted_excluded "
            "entries for candidate slugs catalogued by "
            "inventory_cohort_candidates.py."
        ),
    )
    parser.add_argument(
        "--catalog", type=Path, default=_DEFAULT_CATALOG_PATH,
        help="Path to the expansion_candidates.json catalog.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=_DEFAULT_MANIFEST_PATH,
        help="Path to cohort_manifest.json to expand IN PLACE.",
    )
    parser.add_argument(
        "--shape-point", type=str, default=_DEFAULT_SHAPE_POINT,
        help=(
            "shape_point assigned to every new attempted entry "
            "(default: timing_shape)."
        ),
    )
    parser.add_argument(
        "--depth-first-slugs", type=str, default="",
        help=(
            "Comma-separated list of slugs to receive additional "
            "shape_point entries for the cohort's depth-first "
            "coverage dimension (>= 6 slugs with >= 3 shape_points "
            "each). Each listed slug must have an entry in the "
            "catalog."
        ),
    )
    parser.add_argument(
        "--depth-first-shape-points",
        type=str,
        default="correctness_shape,trace_l2_accuracy_xl",
        help=(
            "Comma-separated list of additional shape_points to "
            "append for each --depth-first-slugs entry. Default: "
            "correctness_shape,trace_l2_accuracy_xl."
        ),
    )
    parser.add_argument(
        "--planned-vendor-baseline-slug", type=str, default="",
        help=(
            "Optional vendor_baselines slug to append as a planned "
            "(build_fail) attempted_excluded entry. Use when the "
            "operator has identified a vendor binary that needs "
            "build work before the live admission sweep can run "
            "it. The placeholder binary_sha256 is deterministic; "
            "the live build replaces it on admission."
        ),
    )
    args = parser.parse_args(argv)
    catalog = _load_json(args.catalog)
    manifest = _load_json(args.manifest)
    original_attempted = manifest["attempted_cell_count"]
    expanded = expand_manifest(
        catalog, manifest, shape_point=args.shape_point,
    )
    if args.depth_first_slugs.strip():
        target_slugs = tuple(
            s.strip() for s in args.depth_first_slugs.split(",")
            if s.strip()
        )
        extra_shape_points = tuple(
            s.strip() for s in args.depth_first_shape_points.split(",")
            if s.strip()
        )
        expanded = add_depth_first_shapes(
            catalog, expanded,
            target_slugs=target_slugs,
            extra_shape_points=extra_shape_points,
        )
    if args.planned_vendor_baseline_slug.strip():
        expanded = add_planned_vendor_baseline(
            expanded, slug=args.planned_vendor_baseline_slug.strip(),
        )
    added = expanded["attempted_cell_count"] - original_attempted
    _write_json(expanded, args.manifest)
    print(
        f"expanded {args.manifest.relative_to(_REPO_ROOT)}: "
        f"+{added} attempted_excluded cells "
        f"(attempted {original_attempted} -> "
        f"{expanded['attempted_cell_count']}, admitted unchanged at "
        f"{expanded['admitted_cell_count']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
