"""Regenerate ``cohort_manifest.json`` over the live-reconciled admitted
UNION: the existing repeatability-admitted base cohort, the four
register-cap build variants, and the freshly re-captured task6 cache
cells (live NCU re-capture, robust repeatability, fresh provenance).

This is the cohort-corpus regeneration step that supersedes the
post-relabel 20-cell manifest. It takes the committed manifest as the
base, flips the
re-captured cells from ``attempted_excluded`` to ``admitted`` with their
FRESH identity tuples, adds the never-before-manifested fresh cells and
the four build variants, and stamps each admitted cell's
``split_assignment`` via the SAME deterministic policy the split
generator uses (``generate_cohort_split.assign_splits``) so the manifest
and the split agree by construction.

Provenance discipline (no stale splice):
  * A re-captured cell's identity tuple + ``source_commit`` come ONLY
    from the committed fresh-label artifact
    ``task6_recapture_selected_label_admitted.json`` (live source_commit,
    stddev_l2 <= 0.03). The cell's ``expected_label_path`` points at that
    artifact so the per-cell label-presence guard verifies the label is
    really there.
  * The four build-variant cells take their identity from the canonical
    ``data/local_labels/label_admitted.json`` rows (where they were
    merged after their own live capture) and point ``expected_label_path``
    there.
  * Existing admitted base cells are carried through byte-identically.

The result is validated against ``validate_manifest`` (schema + identity
lockstep + mixed-arity uniqueness + accounting) before writing — fail
closed.

    python -m tools.sm_86_predictor_accuracy_recovery_v2.build_union_cohort_manifest
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from generate_cohort_split import assign_splits  # noqa: E402
import validate_cohort_manifest as vcm  # noqa: E402

_DATA_DIR = _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
_DEFAULT_MANIFEST_PATH = _DATA_DIR / "cohort_manifest.json"
# The fresh re-captured admitted labels live in the canonical labels dir
# (data/local_labels/) as a sibling of label_admitted.json while a cell is
# label-only (no feature pipeline yet). The predictor-head refit fold MOVES a
# re-captured cell's row into label_admitted.json once its feature pipeline
# (sidecars + predictor bundle + admitted regen-ledger entry) lands; the cell
# is then listed in _MATERIALIZED_RECAPTURE_CELL_FAMILY and re-pointed by
# fold step 2c below. A cell lives in exactly ONE of the two stores; the
# cohort manifest's per-cell expected_label_path points at whichever store
# currently holds the row so the per-cell label-presence guard verifies the
# label is really present.
_RECAPTURE_PATH = _REPO_ROOT / "data" / "local_labels" / "task6_recapture_admitted_labels.json"
_RECAPTURE_REL = "data/local_labels/task6_recapture_admitted_labels.json"
_LABEL_ADMITTED_PATH = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_LABEL_ADMITTED_REL = "data/local_labels/label_admitted.json"

_DEVICE = "RTX 3090 Ti"

# Family buckets for the fresh cells never previously in the manifest.
# The eight re-captured cells already in the manifest keep their recorded
# family_bucket; only the genuinely-new (slug, shape_point) cells need a
# bucket assigned here (existing-sibling family for each).
_NEW_CELL_FAMILY = {
    ("cp_async_shared_tile_copy", "trace_l2_accuracy_xl"): "memory_stress",
    ("rmsnorm_fwd_bf16", "trace_00"): "dl_primitives",
    ("vendor_sgemm_strided_batched_cublas", "trace_00"): "vendor_baselines",
}
_VARIANT_FAMILY = "irregular_hostile"

# Newly-admitted BASE factorial cells that were materialized into the
# canonical label_admitted.json after their own live capture + feature
# pipeline (sidecars + predictor bundle + admitted regen-ledger entry) —
# distinct from the re-captured task6 cells (which stay label-only in the
# recapture sibling) and the build variants. The atomics ATOM partner
# global_atomic_return_used (the return-value-used counterpart of the
# return-ignored global_atomic_histogram) is the first such cell, so it
# shares the atomics family bucket. Each cell's expected_label_path points
# at label_admitted.json (where its admitted row lives).
_NEW_FACTORIAL_CELL_FAMILY = {
    ("global_atomic_return_used", "trace_l2_accuracy_xl"): "irregular_hostile",
    # The paired RED histogram (return-ignored atomicAdd -> REDG) shares the
    # atomics irregular_hostile bucket with its ATOM partner.
    ("global_atomic_histogram", "trace_l2_accuracy_xl"): "irregular_hostile",
    # The opcode-distinct atomic-MAX pair (ATOMG.E.MAX / REDG.E.MAX) shares the
    # same atomics irregular_hostile bucket.
    ("global_atomic_max_return_used", "trace_l2_accuracy_xl"): "irregular_hostile",
    ("global_atomic_max_histogram", "trace_l2_accuracy_xl"): "irregular_hostile",
    # The bitwise atomic-AND pair (ATOMG.E.AND / REDG.E.AND) shares the atomics bucket.
    ("global_atomic_and_return_used", "trace_l2_accuracy_xl"): "irregular_hostile",
    ("global_atomic_and_histogram", "trace_l2_accuracy_xl"): "irregular_hostile",
    # The vector-width request/sector-formation family members are memory-family
    # global copies (roster family="memory", FP32), so they share the
    # memory_stress bucket with the other bandwidth/streaming cells.
    ("global_vectorized_128b_copy", "trace_l2_accuracy_xl"): "memory_stress",
    ("global_vectorized_64b_copy", "trace_l2_accuracy_xl"): "memory_stress",
    ("global_scalar_32b_copy", "trace_l2_accuracy_xl"): "memory_stress",
    # The local-memory family members (dynamic-index + volatile) force .local
    # (LDL/STL) traffic; they are memory-system stress cells, so they share the
    # memory_stress bucket with the other bandwidth/streaming/local cells.
    ("local_dynamic_index_array", "trace_l2_accuracy_xl"): "memory_stress",
    ("local_volatile_array", "trace_l2_accuracy_xl"): "memory_stress",
}

# Re-captured task6 cells MATERIALIZED into the canonical
# ``label_admitted.json`` by the predictor-head refit fold (their rows moved
# out of the recapture sibling once each cell gained its feature pipeline:
# sidecars + predictor bundle + admitted regen-ledger entry). Like the 2b
# factorial cells, the materialized form is AUTHORITATIVE: the manifest cell
# of the same (slug, shape_point) is re-pointed in place from the recapture
# sibling to label_admitted.json. Family buckets are carried over verbatim
# from the cells' prior manifest rows (no re-bucketing on materialization).
_MATERIALIZED_RECAPTURE_CELL_FAMILY = {
    ("working_set_capacity_sweep", "trace_l2_accuracy_xl"): "irregular_hostile",
    ("warm_cold_pair_relaunch", "trace_l2_accuracy_xl"): "memory_stress",
    ("l2_persisting_window_sweep", "trace_l2_accuracy_xl"): "memory_stress",
    ("l2_prefetch_hint_sweep", "trace_l2_accuracy_xl"): "memory_stress",
    ("bf16_hmma_gemm_traceable", "trace_l2_accuracy_xl"): "gemm_family",
    ("fp16_hmma_gemm_traceable", "trace_l2_accuracy_xl"): "gemm_family",
    ("fp32_fma_dep_chain", "trace_l2_accuracy_xl"): "irregular_hostile",
    ("cp_async_shared_tile_copy", "trace_l2_accuracy_xl"): "memory_stress",
    ("rmsnorm_fwd_bf16", "trace_00"): "dl_primitives",
    ("vendor_sgemm_strided_batched_cublas", "trace_00"): "vendor_baselines",
}


def _simbench_trace_rel(slug: str, shape: str) -> str:
    return (
        f"workloads/simbench_bw/results/{slug}/rtx_3090_ti/sm_86/"
        f"trace_{shape}.json"
    )


def _variant_trace_rel(variant_id: str, shape: str) -> str:
    return (
        f"workloads/simbench_bw/results/register_spill_local_mem/"
        f"rtx_3090_ti/sm_86/variants/{variant_id}/trace_{shape}.json"
    )


def _identity_tuple_from_label(row: Mapping[str, Any]) -> dict:
    it = {
        "arch": row["arch"],
        "binary_sha256": row["binary_sha256"],
        "device": _DEVICE,
        "kernel_signature_sha256": row["kernel_signature_sha256"],
        "shape_point": row["shape_point"],
        "slug": row["slug"],
        "source_commit": row["source_commit"],
    }
    for key in ("register_cap", "variant_id"):
        if row.get(key) is not None:
            it[key] = row[key]
    return it


def _admitted_cell_from_label(
    row: Mapping[str, Any],
    *,
    family_bucket: str,
    expected_label_rel: str,
    expected_trace_rel: str,
) -> dict:
    cell = {
        "slug": row["slug"],
        "shape_point": row["shape_point"],
        "arch": row["arch"],
        "device": _DEVICE,
        "family_bucket": family_bucket,
        "split_assignment": "calibration",  # set authoritatively below
        "expected_trace_path": expected_trace_rel,
        "expected_label_path": expected_label_rel,
        "admission_status": "admitted",
        "exclusion_typed_reason": None,
        "identity_tuple": _identity_tuple_from_label(row),
    }
    for key in ("register_cap", "variant_id"):
        if row.get(key) is not None:
            cell[key] = row[key]
    return cell


def build_manifest(
    base_manifest: Mapping[str, Any],
    recapture_cells: list[Mapping[str, Any]],
    variant_rows: list[Mapping[str, Any]],
    frozen_at_commit: str,
    new_factorial_rows: list[Mapping[str, Any]] | None = None,
    materialized_recapture_rows: list[Mapping[str, Any]] | None = None,
    label_admitted_rows: list[Mapping[str, Any]] | None = None,
) -> dict:
    new_factorial_rows = new_factorial_rows or []
    materialized_recapture_rows = materialized_recapture_rows or []
    cells = [copy.deepcopy(c) for c in base_manifest["cells"]]
    by_key = {(c["slug"], c["shape_point"]): c for c in cells}

    # 1. The re-captured cells: flip an existing manifest cell, or add a
    #    new one. Each carries its FRESH identity tuple + recapture
    #    expected_label_path.
    for row in recapture_cells:
        key = (row["slug"], row["shape_point"])
        if key in by_key:
            cell = by_key[key]
            cell["admission_status"] = "admitted"
            cell["exclusion_typed_reason"] = None
            cell["identity_tuple"] = _identity_tuple_from_label(row)
            cell["expected_label_path"] = _RECAPTURE_REL
            cell["expected_trace_path"] = _simbench_trace_rel(*key)
            cell["arch"] = row["arch"]
            cell["device"] = _DEVICE
        else:
            family = _NEW_CELL_FAMILY.get(key)
            if family is None:
                raise ValueError(
                    f"build_union_cohort_manifest: no family bucket mapping "
                    f"for new re-captured cell {key!r}"
                )
            new_cell = _admitted_cell_from_label(
                row,
                family_bucket=family,
                expected_label_rel=_RECAPTURE_REL,
                expected_trace_rel=_simbench_trace_rel(*key),
            )
            cells.append(new_cell)
            by_key[key] = new_cell

    # 2. The four register-cap build variants (canonical label store).
    #    Idempotent: a variant cell keys on (slug, shape_point, variant_id)
    #    so a re-run updates the existing variant cell in place rather
    #    than appending a duplicate.
    variant_index = {
        (c["slug"], c["shape_point"], c.get("variant_id")): c
        for c in cells
        if c.get("variant_id")
    }
    for row in variant_rows:
        new_cell = _admitted_cell_from_label(
            row,
            family_bucket=_VARIANT_FAMILY,
            expected_label_rel=_LABEL_ADMITTED_REL,
            expected_trace_rel=_variant_trace_rel(
                row["variant_id"], row["shape_point"]
            ),
        )
        vkey = (row["slug"], row["shape_point"], row["variant_id"])
        existing = variant_index.get(vkey)
        if existing is not None:
            existing.update(new_cell)
        else:
            cells.append(new_cell)
            variant_index[vkey] = new_cell

    # 2b. Newly-admitted base factorial cells (materialized into
    #     label_admitted.json after their own live capture + feature
    #     pipeline). A materialized base factorial cell is AUTHORITATIVE:
    #     when a cell of the same (slug, shape_point) is already carried
    #     through the base manifest -- including as a stale recapture- or
    #     exclusion-pointed entry from a prior round (e.g. a RED histogram
    #     that was a label-only recapture cell before it earned its own
    #     live capture + feature pipeline) -- re-point it in place to its
    #     canonical label_admitted base-factorial form. Idempotent when the
    #     cell is already in that form (clear + update writes the same
    #     values; split_assignment is re-stamped below).
    factorial_index = {(c["slug"], c["shape_point"]): c for c in cells}
    for row in new_factorial_rows:
        key = (row["slug"], row["shape_point"])
        family = _NEW_FACTORIAL_CELL_FAMILY.get(key)
        if family is None:
            raise ValueError(
                f"build_union_cohort_manifest: no family bucket mapping "
                f"for new factorial cell {key!r}"
            )
        new_cell = _admitted_cell_from_label(
            row,
            family_bucket=family,
            expected_label_rel=_LABEL_ADMITTED_REL,
            expected_trace_rel=_simbench_trace_rel(*key),
        )
        existing = factorial_index.get(key)
        if existing is not None:
            existing.clear()
            existing.update(new_cell)
        else:
            cells.append(new_cell)
            factorial_index[key] = new_cell

    # 2c. Re-captured task6 cells MATERIALIZED into label_admitted.json by
    #     the predictor-head refit fold. Same authoritative-update-in-place
    #     semantics as 2b: every such cell is already in the manifest as a
    #     recapture-pointed entry (expected_label_path at the recapture
    #     sibling), so re-point it to its canonical label_admitted form.
    #     Family buckets come from _MATERIALIZED_RECAPTURE_CELL_FAMILY
    #     (carried over verbatim from the prior manifest rows). Idempotent on
    #     re-run (clear + update writes the same values).
    materialized_index = {(c["slug"], c["shape_point"]): c for c in cells}
    for row in materialized_recapture_rows:
        key = (row["slug"], row["shape_point"])
        family = _MATERIALIZED_RECAPTURE_CELL_FAMILY.get(key)
        if family is None:
            raise ValueError(
                f"build_union_cohort_manifest: no family bucket mapping "
                f"for materialized re-captured cell {key!r}"
            )
        new_cell = _admitted_cell_from_label(
            row,
            family_bucket=family,
            expected_label_rel=_LABEL_ADMITTED_REL,
            expected_trace_rel=_simbench_trace_rel(*key),
        )
        existing = materialized_index.get(key)
        if existing is not None:
            existing.clear()
            existing.update(new_cell)
        else:
            cells.append(new_cell)
            materialized_index[key] = new_cell

    # 2d. Identity refresh from the canonical label store. Every admitted
    #     cell whose expected_label_path points at label_admitted.json must
    #     mirror the CURRENT label row's identity (the validator's
    #     label-presence check requires the joint binary_sha256 +
    #     kernel_signature_sha256 to exist in the label corpus). A targeted
    #     live re-capture that replaces a label row's binary/source_commit
    #     (e.g. a base cell whose binary was rebuilt after its original
    #     label) would otherwise leave the manifest carrying the orphaned
    #     prior identity. Fail closed when an admitted label-admitted-pointed
    #     cell has no matching label row at all.
    label_rows_by_key: dict[tuple, Mapping[str, Any]] = {}
    for row in label_admitted_rows or []:
        if row.get("arch") != "sm_86":
            continue
        key = (row["slug"], row["shape_point"], row.get("variant_id") or "")
        label_rows_by_key[key] = row
    if label_rows_by_key:
        for cell in cells:
            if cell.get("admission_status") != "admitted":
                continue
            if cell.get("expected_label_path") != _LABEL_ADMITTED_REL:
                continue
            key = (
                cell["slug"], cell["shape_point"],
                cell.get("variant_id") or "",
            )
            row = label_rows_by_key.get(key)
            if row is None:
                raise ValueError(
                    "build_union_cohort_manifest: admitted cell "
                    f"{key!r} points at {_LABEL_ADMITTED_REL} but no "
                    "matching label row exists"
                )
            cell["identity_tuple"] = _identity_tuple_from_label(row)

    # 3. Stamp split_assignment on every admitted cell via the shared
    #    deterministic policy (single source of truth with the split
    #    generator).
    admitted = [c for c in cells if c["admission_status"] == "admitted"]
    slug_to_split = assign_splits(admitted)
    for cell in admitted:
        cell["split_assignment"] = slug_to_split[cell["slug"]]

    manifest = {
        "schema_version": int(base_manifest["schema_version"]),
        "frozen_at_commit": frozen_at_commit,
        "attempted_cell_count": len(cells),
        "admitted_cell_count": len(admitted),
        "replacement_history": base_manifest.get("replacement_history", []),
        "cells": cells,
    }
    return manifest


def generate(
    manifest_path: Path | None = None,
    frozen_at_commit: str | None = None,
) -> dict:
    manifest_path = manifest_path or _DEFAULT_MANIFEST_PATH
    base = json.loads(manifest_path.read_text(encoding="utf-8"))
    recapture = json.loads(_RECAPTURE_PATH.read_text(encoding="utf-8"))["cells"]
    label_admitted = json.loads(
        _LABEL_ADMITTED_PATH.read_text(encoding="utf-8")
    )["cells"]
    variant_rows = [
        c for c in label_admitted
        if c.get("arch") == "sm_86" and c.get("variant_id")
    ]
    new_factorial_rows = [
        c for c in label_admitted
        if c.get("arch") == "sm_86"
        and not c.get("variant_id")
        and (c["slug"], c["shape_point"]) in _NEW_FACTORIAL_CELL_FAMILY
    ]
    materialized_recapture_rows = [
        c for c in label_admitted
        if c.get("arch") == "sm_86"
        and not c.get("variant_id")
        and (c["slug"], c["shape_point"]) in _MATERIALIZED_RECAPTURE_CELL_FAMILY
    ]
    if frozen_at_commit is None:
        frozen_at_commit = base["frozen_at_commit"]
    manifest = build_manifest(
        base,
        recapture,
        variant_rows,
        frozen_at_commit,
        new_factorial_rows=new_factorial_rows,
        materialized_recapture_rows=materialized_recapture_rows,
        label_admitted_rows=label_admitted,
    )

    # Validate-before-replace: write to a temp sibling, validate via the
    # strict validator (fail closed), and only os.replace onto the target
    # on success so a failed run never corrupts the committed manifest.
    serialized = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(serialized, encoding="utf-8")
    try:
        vcm.validate_manifest(tmp_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    import os
    os.replace(tmp_path, manifest_path)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate cohort_manifest.json over the live-reconciled "
            "admitted union (validates fail-closed before keeping)."
        )
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--frozen-at-commit",
        type=str,
        default=None,
        help="40-hex commit to stamp (default: keep the base manifest's).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    manifest = generate(
        manifest_path=args.manifest,
        frozen_at_commit=args.frozen_at_commit,
    )
    sys.stdout.write(
        f"build_union_cohort_manifest: wrote {args.manifest} "
        f"(admitted={manifest['admitted_cell_count']} "
        f"attempted={manifest['attempted_cell_count']}; schema+invariants ok)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
