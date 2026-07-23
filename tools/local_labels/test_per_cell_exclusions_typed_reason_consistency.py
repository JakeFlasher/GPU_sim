"""Source-to-companion typed-reason consistency scanner for the
per-cell exclusions live-sweep artifact.

The bulk sm_86 NCU re-capture sweep emits two committed artifacts that
must agree on per-cell typed reasons:

* ``data/local_labels/label_excluded.json`` — the source aggregator
  output. Each cell carries one of two typed reasons:
  ``label_repeatability_failed`` (per-cell stddev exceeded the
  ``BL-20260526-typed-exclusion-stddev-threshold`` 0.03 floor) or
  ``label_provenance_duplicate`` (multiple sidecars produced for the
  same cell whose source_commit manifests disagree, so no canonical
  label can be admitted).
* ``data/sm_86_predictor_accuracy_recovery_v2/per_cell_exclusions_live_sweep.json``
  — the companion artifact consumed by ``build_cohort_accounting.py``.
  Each entry must carry the closed-enum ``admission_status`` value
  matching the source ``typed_reason``.

The mapping contract is:

* ``label_repeatability_failed`` →
  ``admission_status=excluded_label_repeatability`` with the per-cell
  ``stddev_l2_sector`` exceeding the 0.03 threshold.
* ``label_provenance_duplicate`` →
  ``admission_status=excluded_label_provenance_duplicate`` with the
  conflicting source_commit manifests preserved in the source
  ``detail`` field.

A violation (companion entry mapping a provenance-duplicate cell to
``excluded_label_repeatability``, OR an exclusion_reason claiming a
stddev threshold breach when the source stddev is at or below 0.03)
is a contract break. This scanner enforces both arms.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABEL_EXCLUDED = _REPO_ROOT / "data" / "local_labels" / "label_excluded.json"
_COMPANION = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "per_cell_exclusions_live_sweep.json"
)

_STDDEV_FLOOR = 0.03

_REASON_TO_STATUS = {
    "label_repeatability_failed": "excluded_label_repeatability",
    "label_provenance_duplicate": "excluded_label_provenance_duplicate",
}


def _load_source_by_key() -> dict[str, dict]:
    payload = json.loads(_LABEL_EXCLUDED.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for cell in payload.get("cells", []):
        key = f"{cell['slug']}/{cell['shape_point']}"
        out[key] = cell
    return out


def _load_companion_by_key() -> dict[str, dict]:
    payload = json.loads(_COMPANION.read_text(encoding="utf-8"))
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def test_companion_keys_match_source_keys() -> None:
    """The companion artifact must carry exactly one entry per source
    exclusion cell (1:1 join). A missing key or an extra key surfaces
    as an audit drift between the source aggregator output and the
    downstream consumer's input.
    """
    src = _load_source_by_key()
    comp = _load_companion_by_key()
    missing = set(src) - set(comp)
    extra = set(comp) - set(src)
    assert not missing, (
        f"source cells absent from companion artifact: {sorted(missing)}"
    )
    assert not extra, (
        f"companion artifact carries cells outside the source: "
        f"{sorted(extra)}"
    )


def test_every_companion_entry_maps_typed_reason_to_correct_status() -> None:
    """For every source exclusion cell, the companion's
    ``admission_status`` must match the closed-enum mapping:
    ``label_repeatability_failed`` →
    ``excluded_label_repeatability``;
    ``label_provenance_duplicate`` →
    ``excluded_label_provenance_duplicate``. A drift here would
    misclassify provenance duplicates as repeatability failures (or
    vice versa) and break the 74→50 cohort-shrink audit story.
    """
    src = _load_source_by_key()
    comp = _load_companion_by_key()
    violations = []
    for key, src_cell in src.items():
        typed_reason = src_cell.get("typed_reason")
        expected_status = _REASON_TO_STATUS.get(typed_reason)
        if expected_status is None:
            violations.append(
                f"{key}: source typed_reason {typed_reason!r} not in "
                f"the closed-enum mapping"
            )
            continue
        actual_status = comp[key].get("admission_status")
        if actual_status != expected_status:
            violations.append(
                f"{key}: source typed_reason={typed_reason!r} expects "
                f"admission_status={expected_status!r} but companion "
                f"has {actual_status!r}"
            )
    assert not violations, "typed_reason → admission_status drift: " + (
        "; ".join(violations)
    )


def test_repeatability_entries_carry_stddev_above_threshold() -> None:
    """Every companion entry whose source typed_reason is
    ``label_repeatability_failed`` must carry a source
    ``stddev_l2_sector`` strictly greater than the 0.03 threshold.
    A repeatability-failure entry whose stddev is at or below the
    floor is a mathematical contradiction (the gate would have
    admitted the cell).
    """
    src = _load_source_by_key()
    violations = []
    for key, src_cell in src.items():
        if src_cell.get("typed_reason") != "label_repeatability_failed":
            continue
        stddev = src_cell.get("stddev_l2_sector")
        if stddev is None or stddev <= _STDDEV_FLOOR:
            violations.append(
                f"{key}: source typed_reason=label_repeatability_failed "
                f"but stddev_l2_sector={stddev!r} is at or below the "
                f"{_STDDEV_FLOOR} floor"
            )
    assert not violations, "stddev contradiction: " + "; ".join(violations)


def test_provenance_duplicate_entries_carry_conflicting_source_commits() -> None:
    """Every companion entry whose source typed_reason is
    ``label_provenance_duplicate`` must carry source ``detail`` text
    that names the conflicting source_commit manifests. A
    provenance-duplicate entry whose detail does not name a commit
    conflict means the typed_reason was applied incorrectly upstream.
    """
    src = _load_source_by_key()
    violations = []
    for key, src_cell in src.items():
        if src_cell.get("typed_reason") != "label_provenance_duplicate":
            continue
        detail = src_cell.get("detail", "")
        if "source_commit" not in detail:
            violations.append(
                f"{key}: source typed_reason="
                f"label_provenance_duplicate but detail does not name "
                f"source_commit conflict: {detail!r}"
            )
    assert not violations, "provenance-duplicate detail drift: " + (
        "; ".join(violations)
    )


def test_companion_exclusion_reason_does_not_claim_stddev_breach_when_in_threshold() -> None:
    """The companion ``exclusion_reason`` prose must not claim a
    stddev threshold breach for a cell whose source stddev is at or
    below 0.03. An earlier pre-regen companion artifact carried the
    mistake: 6 provenance-duplicate cells (whose source typed_reason
    is ``label_provenance_duplicate``, NOT ``label_repeatability_failed``)
    were emitted with prose like
    ``"stddev_l2_sector=0.0028 exceeds 0.03 threshold"`` even though
    0.0028 < 0.03 — a mathematical contradiction. This test pins
    the no-false-breach invariant going forward.
    """
    src = _load_source_by_key()
    comp = _load_companion_by_key()
    violations = []
    for key, src_cell in src.items():
        stddev = src_cell.get("stddev_l2_sector")
        if stddev is None or stddev > _STDDEV_FLOOR:
            continue
        prose = comp[key].get("exclusion_reason", "")
        if "exceeds 0.03 threshold" in prose:
            violations.append(
                f"{key}: source stddev_l2_sector={stddev!r} is at or "
                f"below 0.03 but companion exclusion_reason claims "
                f"the threshold was exceeded: {prose!r}"
            )
    assert not violations, "false stddev-breach claims: " + (
        "; ".join(violations)
    )


def test_companion_carries_source_typed_reason_field() -> None:
    """Every companion entry must carry a ``source_typed_reason``
    field equal to the source ``typed_reason``. This is the
    structured-field counterpart to the closed-enum admission_status
    mapping: it lets downstream consumers route on the exact source
    reason without having to invert the mapping table.
    """
    src = _load_source_by_key()
    comp = _load_companion_by_key()
    violations = []
    for key, src_cell in src.items():
        expected = src_cell.get("typed_reason")
        actual = comp[key].get("source_typed_reason")
        if actual != expected:
            violations.append(
                f"{key}: source_typed_reason={actual!r} != source "
                f"typed_reason={expected!r}"
            )
    assert not violations, "source_typed_reason drift: " + "; ".join(
        violations
    )
