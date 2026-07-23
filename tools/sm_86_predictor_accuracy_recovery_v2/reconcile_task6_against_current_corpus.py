"""Reconcile the imported task6 cohort against the CURRENT canonical corpus.

The submission-track cohort split is specified over the union of the
imported task6 cohort (``data/sm_86_predictor_accuracy_recovery_v2/
task6_selected_label_admitted.json``, 49 sm_86 cells) and the admitted
factorial cells. Since the task6 cohort was imported, the bulk sm_86 relabel
re-capture sweep narrowed the canonical admitted set
(``data/local_labels/label_admitted.json``) and typed-excluded the cells that
failed the repeatability gate (``data/local_labels/label_excluded.json``).

This tool builds the provenance-safe reconciliation Codex's Round-25 review
required BEFORE any split regeneration: it classifies every task6 identity, by
the SHARED corpus-cell identity, into exactly one of:

* ``current_admitted`` — the task6 (slug, shape_point) is admitted in the
  current canonical corpus (a base admitted cell).
* ``current_excluded_repeatability_failed`` — present in
  ``label_excluded.json`` with ``typed_reason == label_repeatability_failed``
  (the relabel evicted it; its task6 label is STALE and must NOT be spliced
  back).
* ``current_excluded_other`` — excluded for any other typed reason.
* ``missing_from_root`` — neither admitted nor excluded in the current corpus
  (never captured under the current relabel sweep).

It also records the register-cap build-variant cells admitted in the current
corpus (a separate factorial axis NOT present in the task6 cohort) and whether
each task6 (slug, shape_point) is a CACHE shape (produces L1/L2 hit-rate
labels: ``trace_l2_accuracy_xl`` / ``trace_l2_accuracy_lg`` / ``trace_00``) vs a
correctness/timing-only shape (no cache label, so not re-capturable as a cache
cell).

The emitted artifact is the load-bearing input to the cohort-split
regeneration: the
``current_admitted`` set + freshly re-captured ``missing_from_root`` /
``*_repeatability_failed`` cells (re-captured live, only freshly admitted
imported) + the register-cap variants form the admitted union the slug-disjoint
split is built over. NO stale repeatability-failed task6 label is ever copied
into the corpus by this tool — it is read-only over the inputs.

CLI:

  python tools/sm_86_predictor_accuracy_recovery_v2/reconcile_task6_against_current_corpus.py \
      [--task6 <path>] [--admitted <path>] [--excluded <path>] [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
# Shared mixed-arity corpus-cell identity helper (the SAME normalizer the
# corpus / merge / ledger use): a base cell yields "" and a register-cap
# variant yields its canonical register_cap_<n>.
_LOCAL_LABELS_DIR = _REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS_DIR))
from corpus_cell_identity import cell_variant_id  # noqa: E402

_TASK6_DEFAULT = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "task6_selected_label_admitted.json"
)
_ADMITTED_DEFAULT = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_EXCLUDED_DEFAULT = _REPO_ROOT / "data" / "local_labels" / "label_excluded.json"
_OUT_DEFAULT = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "task6_current_corpus_reconciliation.json"
)

# Shapes that produce L1/L2 cache hit-rate labels (re-capturable as cache
# cells). Other shapes (correctness_shape / timing_shape) do not carry a cache
# label, so a task6 cell at such a shape cannot be re-admitted as a cache cell.
_CACHE_SHAPES = frozenset({
    "trace_l2_accuracy_xl", "trace_l2_accuracy_lg", "trace_00",
})


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reconcile(
    *, task6_path: Path, admitted_path: Path, excluded_path: Path,
) -> dict[str, Any]:
    task6 = _load(task6_path).get("cells", [])
    admitted = _load(admitted_path).get("cells", [])
    excluded = _load(excluded_path).get("cells", [])

    # Current canonical corpus indices, keyed by (slug, shape_point). Base
    # admitted cells only (a register-cap variant is a distinct factorial axis,
    # tracked separately); the excluded artifact carries device, not arch.
    admitted_base_keys = {
        (c.get("slug"), c.get("shape_point"))
        for c in admitted
        if c.get("arch") == "sm_86" and not cell_variant_id(c)
    }
    variant_cells = sorted(
        {
            (c.get("slug"), c.get("shape_point"), cell_variant_id(c))
            for c in admitted
            if c.get("arch") == "sm_86" and cell_variant_id(c)
        }
    )
    excluded_reason_by_key: dict[tuple[Any, Any], str] = {}
    for c in excluded:
        key = (c.get("slug"), c.get("shape_point"))
        # First typed reason wins; the relabel writes one row per evicted cell.
        excluded_reason_by_key.setdefault(key, str(c.get("typed_reason") or ""))

    classified: list[dict[str, Any]] = []
    for cell in task6:
        slug = cell.get("slug")
        shape_point = cell.get("shape_point")
        key = (slug, shape_point)
        if key in admitted_base_keys:
            klass = "current_admitted"
        elif key in excluded_reason_by_key:
            reason = excluded_reason_by_key[key]
            klass = (
                "current_excluded_repeatability_failed"
                if reason == "label_repeatability_failed"
                else "current_excluded_other"
            )
        else:
            klass = "missing_from_root"
        classified.append({
            "slug": slug,
            "shape_point": shape_point,
            "classification": klass,
            "is_cache_shape": shape_point in _CACHE_SHAPES,
            "excluded_typed_reason": excluded_reason_by_key.get(key),
        })

    counts = Counter(c["classification"] for c in classified)
    # Re-capturable cache cells = non-current-admitted task6 cells at a cache
    # shape (the only ones a live NCU L1/L2 re-capture could re-admit).
    recapturable_cache = [
        c for c in classified
        if c["classification"] != "current_admitted" and c["is_cache_shape"]
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "task6_current_corpus_reconciliation",
        "task6_cell_count": len(task6),
        "current_admitted_base_count": len(admitted_base_keys),
        "register_cap_variant_count": len(variant_cells),
        "register_cap_variants": [
            {"slug": s, "shape_point": sp, "variant_id": v}
            for (s, sp, v) in variant_cells
        ],
        "classification_counts": dict(sorted(counts.items())),
        "recapturable_cache_cell_count": len(recapturable_cache),
        "recapturable_cache_cells": sorted(
            ({"slug": c["slug"], "shape_point": c["shape_point"],
              "classification": c["classification"]}
             for c in recapturable_cache),
            key=lambda c: (c["slug"], c["shape_point"]),
        ),
        "cells": sorted(
            classified, key=lambda c: (c["slug"] or "", c["shape_point"] or "")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task6", type=Path, default=_TASK6_DEFAULT)
    parser.add_argument("--admitted", type=Path, default=_ADMITTED_DEFAULT)
    parser.add_argument("--excluded", type=Path, default=_EXCLUDED_DEFAULT)
    parser.add_argument("--out", type=Path, default=_OUT_DEFAULT)
    args = parser.parse_args(argv)
    artifact = reconcile(
        task6_path=args.task6,
        admitted_path=args.admitted,
        excluded_path=args.excluded,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"reconcile_task6: wrote {args.out.relative_to(_REPO_ROOT)} "
        f"(task6={artifact['task6_cell_count']}, "
        f"counts={artifact['classification_counts']}, "
        f"recapturable_cache={artifact['recapturable_cache_cell_count']}, "
        f"variants={artifact['register_cap_variant_count']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
