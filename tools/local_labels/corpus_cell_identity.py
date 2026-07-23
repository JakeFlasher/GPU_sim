"""Shared mixed-arity corpus-cell identity for the canonical local-label corpus
and its cohort / ledger / split / timing consumers.

This is the ONE helper every consumer of `data/local_labels/label_admitted.json`
(and the cohort-manifest / split / regen-ledger / timing surfaces) must use to
key a cell, so a base cell keeps its historical arity (byte-identical behavior)
while a register-cap build variant is a DISTINCT cell instead of collapsing onto
its siblings. It imports only the dependency-free `variant_identity` leaf — no
heavy deps — so it can be reused under the bare-python label-driver path too.

- ``cell_variant_id(cell)`` -> canonical ``register_cap_<cap>`` (``""`` for a
  base cell), via the shared normalizer; raises ``ValueError`` on a malformed or
  cross-field-inconsistent ``register_cap`` / ``variant_id`` pair (the per-field
  schema cannot catch disagreement).
- ``corpus_cell_key(cell, with_arch=True)`` -> the mixed-arity identity tuple.
  ``with_arch=True`` -> ``(arch, slug, shape_point[, variant_id])`` (the
  canonical local-label corpus); ``with_arch=False`` ->
  ``(slug, shape_point[, variant_id])`` for split / ledger / timing surfaces
  that key without arch.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from variant_identity import variant_identity  # noqa: E402


# Slugs admitted into the canonical corpus (label_admitted.json + the cohort
# manifest/split + the regen ledger) but NOT yet folded into the BASE predictor
# head / eval / accounting cohort. These were a separate factorial sub-corpus
# whose fold into the fitted v5 head is a downstream predictor refit, exactly
# like the register-cap build variants (which are excluded via
# ``cell_variant_id`` instead, since they share their base slug).
# Removing a slug here is the signal that the refit folded it in.
#
# EMPTY since the controlled predictor-head refit: the 11 factorial cells
# (the atomics ATOM/RED families, the vector-width request/sector-formation
# family, and the local-memory family) are folded into the fitted heads. The
# frozenset (and the ``is_pending_refit_factorial`` predicate) remain the
# transitional-boundary mechanism for FUTURE corpus additions: a newly
# materialized base cell goes back in here until the next refit folds it.
PENDING_REFIT_FACTORIAL_SLUGS = frozenset()


# Re-captured task6 cells MATERIALIZED into the canonical corpus
# (label_admitted.json + cohort manifest re-point + regen-ledger entries +
# timing label/exclusion accounting) but NOT yet folded into the BASE
# predictor head / eval / accounting cohort. Same transitional boundary as
# PENDING_REFIT_FACTORIAL_SLUGS, for the formerly label-only re-captured
# cells (their rows moved out of task6_recapture_admitted_labels.json once
# each cell gained its feature pipeline). Removing a slug here is the signal
# that the predictor-head refit folded it in.
#
# EMPTY since the controlled predictor-head refit: the 10 materialized
# re-captured cells are folded into the fitted heads.
PENDING_REFIT_RECAPTURE_SLUGS = frozenset()


# The union the base-predictor-cohort scoping surfaces exclude until the
# refit lands. Consumers that need "is this cell pending the refit fold?"
# should use is_pending_refit_factorial(cell), which checks this union.
PENDING_REFIT_SLUGS = (
    PENDING_REFIT_FACTORIAL_SLUGS | PENDING_REFIT_RECAPTURE_SLUGS
)


def is_pending_refit_factorial(cell: dict) -> bool:
    """True iff the cell's slug is admitted into the canonical corpus but
    pending the predictor-head refit (the factorial sub-corpus in
    ``PENDING_REFIT_FACTORIAL_SLUGS`` or a materialized re-captured cell in
    ``PENDING_REFIT_RECAPTURE_SLUGS``). The base-predictor-cohort scoping
    surfaces (v5 head fit cohort, closure/accuracy slices, per-admitted-cell L1
    prediction coverage, task12 substitution preflight) exclude such cells in
    addition to the register-cap build variants."""
    return cell.get("slug") in PENDING_REFIT_SLUGS


def is_base_predictor_fit_cell(cell: dict) -> bool:
    """The SINGLE predictor-fit cohort definition shared by the artifact
    PRODUCERS (the v5 / L1 calibration-head fitters, their closure / baseline /
    citation emitters, and the per-admitted-cell L1 predictor) AND the test /
    report-side scoping helpers, so producer output and consumer expectation
    cannot drift.

    True iff the cell is an sm_86 cell whose build-variant identity is
    well-formed and whose slug is not pending a refit fold
    (``is_pending_refit_factorial``; the pending sets are EMPTY since the
    controlled refit — they re-populate only when a future corpus addition
    re-opens the transitional boundary).

    The controlled predictor-head refit FOLDED the register-cap build
    variants into the fit cohort: a variant is a distinct admitted cell
    (mixed-arity key via ``corpus_cell_key``) and participates in the fit
    alongside its base siblings — the fitters key cells and bundles by the
    full closed identity (binary_sha256-bearing), so sibling caps never
    collapse. ``cell_variant_id`` is still invoked for its fail-closed
    validation: a malformed / cross-field-inconsistent register_cap /
    variant_id pair raises ``ValueError`` so a producer fails closed rather
    than silently mis-scoping the fit."""
    cell_variant_id(cell)  # fail-closed validation of the variant identity
    return (
        cell.get("arch") == "sm_86"
        and not is_pending_refit_factorial(cell)
    )


def cell_variant_id(cell: dict) -> str:
    """Canonical build-variant id for a cell/row carrying optional
    ``register_cap`` (int) / ``variant_id`` (str), or ``""`` for a base cell.

    Raises ``ValueError`` on a malformed / noncanonical / cross-field-
    inconsistent pair so a caller keying the corpus fails closed rather than
    silently mis-grouping."""
    record = {
        k: cell[k] for k in ("register_cap", "variant_id")
        if cell.get(k) is not None
    }
    variant_id, _cap, err = variant_identity(record)
    if err is not None:
        raise ValueError(
            f"corpus_cell_identity: cell (slug={cell.get('slug')!r}, "
            f"shape_point={cell.get('shape_point')!r}) has an inconsistent "
            f"build-variant identity: {err}"
        )
    return variant_id


def variant_lockstep_error(
    top_record: dict, nested_record: dict | None,
) -> str | None:
    """Lockstep guard for a cell that carries the build-variant axis at TWO
    levels (a top-level ``register_cap`` / ``variant_id`` pair AND a nested
    ``identity_tuple`` that ALSO carries the pair). Returns ``None`` when both
    records normalize to the SAME canonical build-variant identity (a base cell
    -> both ``""``; a register-cap variant -> the same ``register_cap_<n>``),
    else a typed error string.

    Both records are normalized through the SINGLE canonical normalizer
    ``cell_variant_id`` (no parsing rules are re-inlined here): an internally
    inconsistent pair WITHIN either record (its own ``register_cap`` vs
    ``variant_id`` disagree, or a noncanonical alias) surfaces as that
    normalizer's ``ValueError`` text; a top-level-vs-nested disagreement (incl.
    one level present while the other is absent, since canonical-string
    equality compares ``""`` vs ``register_cap_<n>``) surfaces as a mismatch.
    Comparing the canonical ``variant_id`` strings (which encode the cap as
    ``register_cap_<cap>``) enforces both "same canonical id" and "same numeric
    register cap" in one check.
    """
    try:
        top_vid = cell_variant_id(top_record)
    except ValueError as exc:
        return f"top-level build-variant identity invalid: {exc}"
    try:
        nested_vid = cell_variant_id(nested_record or {})
    except ValueError as exc:
        return f"nested identity_tuple build-variant identity invalid: {exc}"
    if top_vid != nested_vid:
        return (
            f"top-level build-variant identity {top_vid!r} != nested "
            f"identity_tuple build-variant identity {nested_vid!r}"
        )
    return None


def corpus_cell_key(cell: dict, *, with_arch: bool = True) -> tuple:
    """Mixed-arity identity key. Base cells keep the historical arity
    (``(arch, slug, shape_point)`` or ``(slug, shape_point)``), byte-identical to
    the pre-variant behavior; a register-cap build variant widens with its
    canonical ``variant_id`` so sibling caps never collapse."""
    variant_id = cell_variant_id(cell)
    if with_arch:
        base = (
            cell.get("arch", ""),
            cell.get("slug", ""),
            cell.get("shape_point", ""),
        )
    else:
        base = (cell.get("slug", ""), cell.get("shape_point", ""))
    return base if not variant_id else base + (variant_id,)
