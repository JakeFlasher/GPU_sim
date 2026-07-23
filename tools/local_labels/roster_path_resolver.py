"""Roster-aware cell-root resolver for the sm_120 L2 cohort.

The frozen primary + replacement roster at
``data/local_labels/sm120_cohort_expansion_roster.toml`` carries a
``trace_json_path`` field per row that pins the on-disk cell-root for
each ``(slug, shape_point)`` candidate. portable_lcd rows land under
``workloads/portable_lcd/results/<slug>/<sku>/<arch_dir>/``;
simbench_bw rows land under
``workloads/simbench_bw/results/<slug>/<sku>/<arch_dir>/``.

The resolver is a strictly read-only layout authority.  A returned path grants
no write permission: writers must preserve it as an input/layout template,
rebase every active output beneath ``results/nvbit_capture_staging``, and run
the rebased destination through ``protected_paths.assert_write_allowed`` before
any mutation.

Two producer-side tools consume the frozen path:

- ``tools/local_labels/activation_trace_orchestrator.py`` uses it only to
  mirror the predictor trace layout beneath guarded staging.
- ``tools/local_labels/stage_measurement_manifests_from_predictor_trace.py``
  reads canonical trace JSON/manifests and writes a mirrored staging copy.

Hardcoding ``workloads/simbench_bw/results`` in either tool (the
pre-Round-7 state) routes portable_lcd cells to the wrong tree.
This helper centralizes the resolution so both tools share one
source of truth.
"""
from __future__ import annotations

import tomllib
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROSTER_PATH = (
    _REPO_ROOT / "data" / "local_labels" / "sm120_cohort_expansion_roster.toml"
)
_KNOWN_ROSTER_SOURCES: tuple[str, ...] = ("portable_lcd", "simbench_bw")


def _load_roster_rows(roster_path: Path = _ROSTER_PATH) -> list[dict]:
    with roster_path.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data.get("rows", []))


def _row_for(slug: str, shape_point: str) -> dict | None:
    """Return the roster row matching ``(slug, shape_point)`` or
    ``None`` when no row matches."""
    for row in _load_roster_rows():
        if row.get("slug") == slug and row.get("shape_point") == shape_point:
            return row
    return None


def _roster_source_for(slug: str, shape_point: str) -> str | None:
    """Return the ``roster_source`` ("portable_lcd" or "simbench_bw")
    for the matching row, or ``None`` if no row matches.

    The roster keys cells by ``(slug, shape_point)``. When the row
    is not found, callers fall back to the explicit default.
    """
    row = _row_for(slug, shape_point)
    if row is None:
        return None
    src = row.get("roster_source")
    return src if src in _KNOWN_ROSTER_SOURCES else None


def _parse_cell_root_from_trace_json_path(
    trace_json_path: str, *, expected_sku: str, expected_arch_dir: str,
) -> Path | None:
    """Parse the roster's ``trace_json_path`` to extract the on-disk
    cell-root, substituting ``expected_sku`` / ``expected_arch_dir``
    when the frozen path uses different values.

    Round-13 tightening (Codex Round 12 Queued #1 fold-in): the
    pre-Round-13 resolver dispatched on ``roster_source`` alone,
    which is correct when the conventional source-derived path is
    used everywhere but weaker than the AC-2 frozen
    ``trace_json_path`` contract. This parser:

    1. Splits ``trace_json_path`` into its conventional 8-part
       layout
       (``workloads / <source> / results / <slug> / <sku> /
        <arch_dir> / trace_<shape> / dynamic_trace.pb``).
    2. Substitutes the requested ``expected_sku`` /
       ``expected_arch_dir`` into the 5th/6th positions.
    3. Returns the resulting cell-root path
       (``workloads / <source> / results / <slug> / <expected_sku> /
        <expected_arch_dir>``).

    Returns ``None`` when the path layout does not match the
    conventional shape (e.g., for malformed or empty
    ``trace_json_path`` values); callers fall back to
    ``_results_root_for_source``."""
    if not trace_json_path:
        return None
    parts = Path(trace_json_path).parts
    # Expected: ['workloads', '<source>', 'results', '<slug>',
    #            '<sku>', '<arch>', 'trace_<shape>',
    #            'dynamic_trace.pb']
    if len(parts) < 8:
        return None
    workloads, source, results, frozen_slug = parts[0:4]
    _frozen_sku, _frozen_arch, trace_dir, pb_filename = parts[4:8]
    if (
        workloads != "workloads"
        or results != "results"
        or source not in _KNOWN_ROSTER_SOURCES
        or pb_filename != "dynamic_trace.pb"
        or not trace_dir.startswith("trace_")
    ):
        return None
    return (
        _REPO_ROOT / workloads / source / results
        / frozen_slug / expected_sku / expected_arch_dir
    )


def _results_root_for_source(roster_source: str) -> Path:
    """Map ``roster_source`` to its on-disk results-root prefix.

    Both prefixes are repo-root-anchored: portable_lcd rows live
    under ``workloads/portable_lcd/results/``; simbench_bw rows live
    under ``workloads/simbench_bw/results/``. The two prefixes are
    NOT interchangeable: the canonical-tree verifier and the trace
    discovery glob in ``stage_measurement_manifests`` both anchor on
    the prefix that matches the row's ``roster_source``.
    """
    if roster_source == "portable_lcd":
        return _REPO_ROOT / "workloads" / "portable_lcd" / "results"
    if roster_source == "simbench_bw":
        return _REPO_ROOT / "workloads" / "simbench_bw" / "results"
    raise ValueError(
        f"unknown roster_source {roster_source!r}; expected one of "
        f"{_KNOWN_ROSTER_SOURCES}"
    )


def resolve_cell_root(
    *,
    slug: str,
    shape_point: str,
    sku: str,
    arch_dir: str,
    fallback_source: str = "simbench_bw",
) -> Path:
    """Return the read-only canonical cell-root for one roster identity.

    This function performs no mutation and does not authorize callers to
    mutate its result.  Active writers must rebase the returned layout beneath
    guarded staging and clear their concrete destinations independently.

    Round-13 derivation: parse the matching roster row's
    ``trace_json_path`` (the AC-2 frozen contract field) and
    substitute the requested SKU/arch directories. Compare the
    parsed root against the ``roster_source``-derived root; fail
    closed when they disagree (the frozen path and the
    source-classifier are out of sync — a roster-side defect
    that should not be silently papered over).

    Off-roster fallback: when no row matches ``(slug,
    shape_point)``, use ``fallback_source`` so the helper still
    serves the legacy sm_86 / 3090 Ti emission path under
    simbench_bw (the activation-orchestrator default).
    """
    row = _row_for(slug, shape_point)
    if row is None:
        return _results_root_for_source(fallback_source) / slug / sku / arch_dir

    roster_source = row.get("roster_source")
    if roster_source not in _KNOWN_ROSTER_SOURCES:
        raise ValueError(
            f"roster row for ({slug!r}, {shape_point!r}) has unknown "
            f"roster_source={roster_source!r}; expected one of "
            f"{_KNOWN_ROSTER_SOURCES}"
        )

    source_derived_root = (
        _results_root_for_source(roster_source) / slug / sku / arch_dir
    )

    # Parse the frozen trace_json_path and substitute SKU/arch.
    # The two derivations MUST agree; mismatch indicates a
    # roster-side defect.
    parsed_root = _parse_cell_root_from_trace_json_path(
        row.get("trace_json_path", ""),
        expected_sku=sku, expected_arch_dir=arch_dir,
    )
    if parsed_root is None:
        # The roster row is missing or malformed trace_json_path;
        # accept the source-derived root rather than failing
        # closed (the row's roster_source is still authoritative
        # for the prefix).
        return source_derived_root
    if parsed_root != source_derived_root:
        raise ValueError(
            f"roster path mismatch for ({slug!r}, {shape_point!r}, "
            f"sku={sku!r}, arch_dir={arch_dir!r}): "
            f"trace_json_path-derived root {parsed_root} != "
            f"roster_source-derived root {source_derived_root}; "
            f"fix the frozen roster row before downstream "
            f"production"
        )
    return parsed_root


def known_roster_sources() -> tuple[str, ...]:
    """Expose the closed set of recognized roster sources for
    callers that want to validate input ahead of calling
    :func:`resolve_cell_root`."""
    return _KNOWN_ROSTER_SOURCES
