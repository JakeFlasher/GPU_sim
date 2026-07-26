"""Helper for the label-supersession workflow.

Adds a closed-enum gate on the typed reasons that
``data/local_labels/label_excluded.json`` admits and a
``supersede_cells`` helper that moves named cells from
``label_admitted.json`` into ``label_excluded.json`` with one of
the admitted typed reasons.

The pre-existing typed reasons (``label_repeatability_failed``,
``label_repeatability_missing``, ``label_schema_admission_failed``)
are emitted by the repeatability aggregator's admission flow
during a normal sweep. The new ``superseded_by_launch_skip_fix``
reason is for the post-collection supersession case: a previously
admitted cell whose NCU label was collected with a producer-side
bug (the ``--launch-skip`` warmup-offset) and must now be
demoted to make room for the re-collected label with the
corrected offset.

The helper preserves the 7-tuple join-key fields on the
excluded record (``binary_sha256``, ``source_commit``,
``kernel_signature_sha256``, ``arch``, ``ncu_metric_version``)
so the comparison runner's join-key discipline still resolves
the historical record's identity at audit time. The
``comparison_bundle.json`` produced from the
post-supersession corpus joins on the NEW label rows and
correctly surfaces ``comparison_runner_join_key_unmatched`` for
the demoted rows if any consumer still references them.

Offline evidence-bookkeeping boundary: this helper is host-only Python; it does
not invoke NCU, does not access the GPU, and does not write outside
``data/local_labels/``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable


_ALLOWED_EXCLUSION_REASONS: frozenset[str] = frozenset({
    "label_repeatability_failed",
    "label_repeatability_missing",
    "label_schema_admission_failed",
    "superseded_by_launch_skip_fix",
    "superseded_by_full_matrix_recollection",
})


def supersede_cells(
    *,
    admitted: dict,
    excluded: dict,
    cell_keys_to_demote: Iterable[tuple[str, str, str, str]],
    typed_reason: str,
    detail: str,
) -> tuple[dict, dict]:
    """Return (new_admitted, new_excluded) with the named cells
    moved from admitted to excluded with ``typed_reason`` +
    ``detail``.

    Each entry in ``cell_keys_to_demote`` is a source-aware
    4-tuple ``(arch, slug, shape_point, source_commit)``. The
    4-tuple uniquely identifies a row within
    ``admitted["cells"]``, disambiguating duplicate-identity
    rows that differ only in ``source_commit``. The helper
    raises:

      - ``ValueError`` when ``typed_reason`` is not in
        ``_ALLOWED_EXCLUSION_REASONS``.
      - ``KeyError`` when any 4-tuple in
        ``cell_keys_to_demote`` does not match a row in
        ``admitted["cells"]``.

    The input corpora are NOT mutated; the returned objects are
    fresh deep copies.

    The excluded record carries every field of the admitted
    source cell unchanged, plus the new ``typed_reason`` and
    ``detail`` fields. This preserves the 7-tuple join key
    (slug, shape_point, binary_sha256, source_commit,
    kernel_signature_sha256, arch, ncu_metric_version) on the
    historical record so the supersession is auditable.
    """
    if typed_reason not in _ALLOWED_EXCLUSION_REASONS:
        raise ValueError(
            f"typed_reason={typed_reason!r} is not in the allowed set "
            f"{sorted(_ALLOWED_EXCLUSION_REASONS)}"
        )

    demote_keys = list(cell_keys_to_demote)
    new_admitted = copy.deepcopy(admitted)
    new_excluded = copy.deepcopy(excluded)
    admitted_cells: list[dict] = new_admitted.setdefault("cells", [])
    excluded_cells: list[dict] = new_excluded.setdefault("cells", [])

    by_key: dict[tuple[str, str, str, str], int] = {}
    for idx, cell in enumerate(admitted_cells):
        key = (
            str(cell.get("arch") or cell.get("device") or ""),
            str(cell.get("slug") or ""),
            str(cell.get("shape_point") or ""),
            str(cell.get("source_commit") or ""),
        )
        by_key[key] = idx

    indices_to_remove: list[int] = []
    for key in demote_keys:
        if key not in by_key:
            raise KeyError(
                f"cell key {key!r} not found in admitted corpus; "
                f"available keys: {sorted(by_key.keys())}"
            )
        source = admitted_cells[by_key[key]]
        excluded_record = copy.deepcopy(source)
        excluded_record["typed_reason"] = typed_reason
        excluded_record["detail"] = detail
        excluded_cells.append(excluded_record)
        indices_to_remove.append(by_key[key])

    new_admitted["cells"] = [
        cell for idx, cell in enumerate(admitted_cells)
        if idx not in set(indices_to_remove)
    ]
    new_excluded["cells"] = excluded_cells
    return new_admitted, new_excluded


def supersede_in_place(
    *,
    admitted_path: Path,
    excluded_path: Path,
    cell_keys_to_demote: Iterable[tuple[str, str, str]],
    typed_reason: str,
    detail: str,
) -> tuple[int, int]:
    """Read the corpora from disk, apply ``supersede_cells``, and
    write the results back. Returns (admitted_count, excluded_count)
    after the move.

    The caller is responsible for ensuring the disk paths are the
    ones the consumer chain expects (typically
    ``data/local_labels/label_admitted.json`` and
    ``data/local_labels/label_excluded.json``).
    """
    with admitted_path.open("r", encoding="utf-8") as fh:
        admitted = json.load(fh)
    with excluded_path.open("r", encoding="utf-8") as fh:
        excluded = json.load(fh)
    new_admitted, new_excluded = supersede_cells(
        admitted=admitted,
        excluded=excluded,
        cell_keys_to_demote=cell_keys_to_demote,
        typed_reason=typed_reason,
        detail=detail,
    )
    admitted_path.write_text(
        json.dumps(new_admitted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    excluded_path.write_text(
        json.dumps(new_excluded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(new_admitted["cells"]), len(new_excluded["cells"])
