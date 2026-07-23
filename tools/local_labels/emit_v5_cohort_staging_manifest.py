"""Emit a machine-readable operator manifest of the v5 cohort's staging state.

Walks every cell in data/local_labels/label_admitted.json and records, for
each cell, the on-disk presence/absence of:

  - The predictor bundle (under data/local_labels/sweeps/*/predictor_bundles/).
  - The sharing_locality.json sidecar (under
    workloads/simbench_bw/results OR workloads/portable_lcd/results).
  - The hrd_profile.json sidecar (same dual root).
  - The sector_stream.jsonl sidecar (same dual root).
  - The dynamic_trace.pb (the upstream NVBit trace; needed when sidecars
    must be regenerated).

For each missing artifact the manifest records a suggested operator
action: REGENERATE_FROM_TRACE (when dynamic_trace.pb is present + the
producer code can rebuild the missing sidecar) OR CROSS_HOST_SCP (when
even the trace is absent and the data must come from another host).

The manifest enables future round(s) to triage the data-staging blocker
without re-running the full audit by hand.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head_sha() -> str:
    """Return the current git HEAD SHA (40-char lowercase hex), or
    ``"unknown"`` if git is unavailable. Mirrors the pattern used by
    other emitters in this directory (e.g.,
    `emit_v5_robustness_artifact.py`, `run_v5_active_path_comparison.py`)
    so the staging manifest's top-level `source_commit` provenance
    field has the same shape contract as the 9 v5 artifacts'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


DEFAULT_LABEL_ADMITTED = "data/local_labels/label_admitted.json"
DEFAULT_LABEL_EXCLUDED = "data/local_labels/label_excluded.json"
DEFAULT_BUNDLE_GLOB = (
    "data/local_labels/sweeps/**/predictor_bundles/*.json"
)
DEFAULT_SIMBENCH_ROOT = "workloads/simbench_bw/results"
DEFAULT_PORTABLE_LCD_ROOT = "workloads/portable_lcd/results"
DEFAULT_OUTPUT = "data/local_labels/v5_cohort_staging_manifest.json"


def _device_for_arch(arch: str) -> str:
    return {"sm_86": "rtx_3090_ti", "sm_120": "rtx_5060_laptop"}.get(
        arch, "unknown",
    )


def _bundle_lookup_set(
    bundle_glob: str,
) -> dict[tuple[str, str, str], list[Path]]:
    out: dict[tuple[str, str, str], list[Path]] = {}
    for p_str in glob.glob(bundle_glob, recursive=True):
        p = Path(p_str)
        try:
            b = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mpi = b.get("manifest_pairing_index", {})
        key = (
            str(mpi.get("arch", "")),
            str(mpi.get("slug", "")),
            str(mpi.get("shape_point", "")),
        )
        out.setdefault(key, []).append(p)
    return out


def _check_artifact_paths(
    *, arch: str, slug: str, shape_point: str,
    simbench_root: Path, portable_lcd_root: Path,
) -> dict[str, Any]:
    device_dir = _device_for_arch(arch)
    out: dict[str, Any] = {}
    artifact_kinds = {
        "sharing_locality": "sharing_locality/sharing_locality.json",
        "hrd_profile": "hrd_profile/hrd_profile.json",
        "sector_stream": "sector_stream/sector_stream.jsonl",
        "dynamic_trace_pb": "dynamic_trace.pb",
    }
    for kind, suffix in artifact_kinds.items():
        found_at: str | None = None
        for root in (simbench_root, portable_lcd_root):
            path = (
                root / slug / device_dir / arch / f"trace_{shape_point}"
                / suffix
            )
            if path.is_file():
                found_at = str(path)
                break
        out[kind] = {
            "present": found_at is not None,
            "path": found_at,
        }
    return out


def _prefer_label_matched_bundle(
    *,
    candidates: list[Path],
    cell: dict[str, Any],
) -> list[Path]:
    """When multiple legacy bundles share the same (slug,
    shape_point), pick the best candidate via this priority:

      1. MPI 5-tuple matches the cell's label identity AND
         ``result_kind == "scalar"`` (a real walker prediction
         the v5 fitter can consume).
      2. MPI 5-tuple matches the label AND ``result_kind`` is
         anything else (e.g., ``typed_unavailable``).
      3. Fallback to ``candidates[0]`` (alphabetical glob order).

    Returns a single-element list so the caller's existing
    ``bundles[0]`` access still works.

    Rationale: when the cascade re-runs and emits a fresh
    ``scalar`` bundle alongside an older ``typed_unavailable``
    bundle (both with identical MPI), the staging manifest must
    surface the scalar bundle as READY. Without the result_kind
    tiebreaker, the alphabetically-first bundle wins — which is
    often the older typed_unavailable one because timestamps
    sort lexicographically.
    """
    if not candidates:
        return []
    if not all(
        isinstance(cell.get(k), str)
        for k in ("binary_sha256", "source_commit",
                  "kernel_signature_sha256")
    ):
        return [candidates[0]]
    target = {
        "slug": str(cell.get("slug", "")),
        "shape_point": str(cell.get("shape_point", "")),
        "binary_sha256": cell["binary_sha256"],
        "source_commit": cell["source_commit"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
    }
    scalar_match: Path | None = None
    any_match: Path | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mpi = payload.get("manifest_pairing_index", {})
        if not isinstance(mpi, dict):
            continue
        if not all(mpi.get(k) == v for k, v in target.items()):
            continue
        if any_match is None:
            any_match = candidate
        if (
            payload.get("result_kind") == "scalar"
            and scalar_match is None
        ):
            scalar_match = candidate
    if scalar_match is not None:
        return [scalar_match]
    if any_match is not None:
        return [any_match]
    return [candidates[0]]


def _bundle_result_kind(bundle_path: Path) -> tuple[str | None, str | None]:
    """Return (result_kind, typed_reason) from a predictor bundle on
    disk. Returns (None, None) when the bundle is unreadable or
    missing the result_kind field; the typed_reason is None unless
    result_kind == "typed_unavailable"."""
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    rk = payload.get("result_kind")
    if not isinstance(rk, str):
        return None, None
    return rk, payload.get("typed_reason")


# Closed enum for `suggested_operator_action`. The schema at
# `workloads/simbench_bw/schemas/v5_cohort_staging_manifest.schema.json`
# pins this exact 10-value set; the legacy single-field colon-form
# (e.g. "BUNDLE_TYPED_UNAVAILABLE:cache_model_underconfigured") is
# now rejected at schema validation time. Root causes live in the
# separate `action_typed_reason` field.
_OPERATOR_ACTION_ENUM: tuple[str, ...] = (
    "READY",
    "REBUILD_BINARY_LOCAL",
    "REBUILD_TRACE_LOCAL",
    "REGENERATE_SIDECARS_LOCAL",
    "REBUILD_BUNDLE_LOCAL",
    "BUNDLE_TYPED_UNAVAILABLE",
    "BUNDLE_NON_SCALAR",
    "CROSS_HOST_SCP",
    "CROSS_HOST_SCP_SIDECARS",
    "TERMINAL_TYPED_FAILURE",
)

# Actions that MUST carry a non-null `action_typed_reason` root
# cause. For every other action, `action_typed_reason` MUST be
# absent or null per the schema's conditional rule.
_ACTIONS_WITH_TYPED_REASON: frozenset[str] = frozenset({
    "BUNDLE_TYPED_UNAVAILABLE",
    "BUNDLE_NON_SCALAR",
    "TERMINAL_TYPED_FAILURE",
})

# Walker-emitted typed_reasons that indicate a TERMINAL model-
# side failure: regenerating sidecars or re-running the cascade
# will NOT promote the bundle to scalar. Cells with these
# typed_reasons are classified as TERMINAL_TYPED_FAILURE at the
# manifest layer so the closure invariant
# ``ready_count + terminal_typed_failure_count == cohort_size``
# admits them as definite-non-READY rather than leaving them in
# the partial-state BUNDLE_TYPED_UNAVAILABLE bucket. The ledger
# still records the underlying walker rejection
# (`all_bundle_candidates_rejected:<n>:canonical_pre_calibration_features_absent`);
# this set names the typed_reasons that the manifest treats as
# closure-terminal.
#
# Currently includes:
# - ``cache_model_underconfigured``: the walker's HRD bin
#   coverage cannot be configured to admit the cell within the
#   share-override threshold; the model gap is the underlying
#   bottleneck (small-N cell with cold-start observation share
#   > _UNDERCONFIGURED_SHARE_THRESHOLD).
#
# Future additions should be documented with the analogous
# "walker / model cannot resolve this cell within the current
# producer surface" semantics, NOT used as a catch-all to
# admit any bundle failure.
_TERMINAL_WALKER_TYPED_REASONS: frozenset[str] = frozenset({
    "cache_model_underconfigured",
})


def _suggested_operator_action(
    *,
    artifact_state: dict[str, Any],
    predictor_bundle_present: bool,
    bundle_result_kind: str | None = None,
    bundle_typed_reason: str | None = None,
) -> tuple[str, str | None]:
    """Return ``(action, action_typed_reason)`` per the closed enum
    + split typed-reason contract. The colon-form replacement:
    where the legacy producer returned
    ``BUNDLE_TYPED_UNAVAILABLE:cache_model_underconfigured`` as a
    single string, this returns
    ``("BUNDLE_TYPED_UNAVAILABLE", "cache_model_underconfigured")``.
    Per-sidecar regen actions collapse into a single
    ``REGENERATE_SIDECARS_LOCAL`` because the per-cell producer
    (``regenerate_sidecars_for_cell.py``) always invokes the full
    sidecar set in one pass."""
    has_trace = artifact_state.get("dynamic_trace_pb", {}).get("present", False)
    has_sl = artifact_state.get("sharing_locality", {}).get("present", False)
    has_hrd = artifact_state.get("hrd_profile", {}).get("present", False)
    has_ss = artifact_state.get("sector_stream", {}).get("present", False)
    if not predictor_bundle_present and not has_trace:
        return "CROSS_HOST_SCP", None
    if not predictor_bundle_present and has_trace:
        return "REBUILD_BUNDLE_LOCAL", None
    if (not has_sl or not has_hrd or not has_ss) and has_trace:
        return "REGENERATE_SIDECARS_LOCAL", None
    if not has_sl or not has_hrd or not has_ss:
        return "CROSS_HOST_SCP_SIDECARS", None
    if bundle_result_kind == "typed_unavailable":
        # Promote walker-terminal typed_reasons to
        # TERMINAL_TYPED_FAILURE so the closure invariant
        # `ready_count + terminal_typed_failure_count == cohort_size`
        # admits these cells in the terminal bucket. Cells whose
        # typed_reason is not in the terminal allowlist stay at
        # BUNDLE_TYPED_UNAVAILABLE (operator-actionable: a future
        # walker improvement could promote them to scalar).
        if bundle_typed_reason in _TERMINAL_WALKER_TYPED_REASONS:
            return "TERMINAL_TYPED_FAILURE", bundle_typed_reason
        return (
            "BUNDLE_TYPED_UNAVAILABLE",
            bundle_typed_reason or "unknown",
        )
    if bundle_result_kind is not None and bundle_result_kind != "scalar":
        return "BUNDLE_NON_SCALAR", bundle_result_kind
    return "READY", None


def _terminal_typed_failure_row(
    *,
    excluded_cell: dict[str, Any],
) -> dict[str, Any]:
    """Build a manifest row for a cell that was excluded at the
    LABEL layer (e.g., relabel sweep stddev floor failure or
    provenance-duplicate eviction). These cells never had a chance
    to admit downstream; they carry TERMINAL_TYPED_FAILURE with
    the relabel typed_reason as the audit cause.

    Reads from ``data/local_labels/label_excluded.json`` rows
    whose schema is ``{device, slug, shape_point, typed_reason,
    ...}`` (NOTE: `device` not `arch` per the relabel sweep
    convention)."""
    slug = str(excluded_cell.get("slug", ""))
    shape_point = str(excluded_cell.get("shape_point", ""))
    # The relabel sweep records the device under `device`; map it
    # back to `arch` for the manifest's `arch` field so admitted
    # + excluded rows share a uniform shape.
    arch = str(excluded_cell.get("device", ""))
    typed_reason = excluded_cell.get("typed_reason") or "unknown"
    return {
        "arch": arch,
        "slug": slug,
        "shape_point": shape_point,
        "cohort_membership": "excluded",
        "predictor_bundle": {"present": False, "paths": []},
        "artifacts": {
            "sharing_locality": {"present": False, "path": None},
            "hrd_profile": {"present": False, "path": None},
            "sector_stream": {"present": False, "path": None},
            "dynamic_trace_pb": {"present": False, "path": None},
        },
        "suggested_operator_action": "TERMINAL_TYPED_FAILURE",
        "action_typed_reason": typed_reason,
        "legacy_bundle_quarantined_count": 0,
    }


def _legacy_bundle_quarantine_count_for_cell(
    *,
    slug: str,
    shape_point: str,
    label_5tuple: dict[str, str] | None,
    bundle_lookup: dict[tuple[str, str, str], list[Path]],
) -> int:
    """Count legacy bundles whose ``manifest_pairing_index`` has
    same ``(slug, shape_point)`` but mismatched
    ``(binary_sha256, source_commit, kernel_signature_sha256)``
    against the cell's label-derived 5-tuple. Mirrors the
    ``legacy_match_scan`` quarantine bucket logic in
    ``tools/local_labels/build_predictor_bundle_for_cell.py`` but
    returns only the count (not the per-candidate diagnostics).

    Returns 0 when ``label_5tuple`` is None (e.g., excluded cell
    with no admitted-label identity)."""
    if label_5tuple is None:
        return 0
    count = 0
    for (_, b_slug, b_shape), bundle_paths in bundle_lookup.items():
        if b_slug != slug or b_shape != shape_point:
            continue
        for bundle_path in bundle_paths:
            try:
                payload = json.loads(
                    bundle_path.read_text(encoding="utf-8"),
                )
            except (OSError, json.JSONDecodeError):
                continue
            mpi = payload.get("manifest_pairing_index", {})
            if not isinstance(mpi, dict):
                continue
            # Same slug+shape are guaranteed by the outer filter.
            # Compare the three identity fields that drift across
            # relabel sweeps.
            drifted = (
                mpi.get("binary_sha256")
                != label_5tuple["binary_sha256"]
                or mpi.get("source_commit")
                != label_5tuple["source_commit"]
                or mpi.get("kernel_signature_sha256")
                != label_5tuple["kernel_signature_sha256"]
            )
            if drifted:
                count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-admitted", default=DEFAULT_LABEL_ADMITTED)
    parser.add_argument("--label-excluded", default=DEFAULT_LABEL_EXCLUDED)
    parser.add_argument("--bundle-glob", default=DEFAULT_BUNDLE_GLOB)
    parser.add_argument("--simbench-root", default=DEFAULT_SIMBENCH_ROOT)
    parser.add_argument(
        "--portable-lcd-root", default=DEFAULT_PORTABLE_LCD_ROOT,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    label_path = Path(args.label_admitted)
    label_raw = json.loads(label_path.read_text(encoding="utf-8"))
    admitted_cells = label_raw.get("cells", [])
    excluded_path = Path(args.label_excluded)
    if excluded_path.is_file():
        excluded_raw = json.loads(excluded_path.read_text(encoding="utf-8"))
        excluded_cells = excluded_raw.get("cells", [])
    else:
        excluded_cells = []
    bundle_lookup = _bundle_lookup_set(args.bundle_glob)
    simbench_root = Path(args.simbench_root)
    portable_lcd_root = Path(args.portable_lcd_root)
    rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    legacy_bundle_quarantined_bundles = 0
    for cell in admitted_cells:
        arch = str(cell.get("arch", ""))
        slug = str(cell.get("slug", ""))
        shape_point = str(cell.get("shape_point", ""))
        bundle_key = (arch, slug, shape_point)
        bundles = bundle_lookup.get(bundle_key, [])
        if not bundles:
            # No bundle with the cell's arch in MPI — admit
            # empty-arch bundles (the closed-5-tuple MPI does not
            # include arch). Collect every empty-arch bundle that
            # matches slug+shape, then prefer the one whose MPI
            # 5-tuple matches the cell's label identity. Without
            # this filter, a stale legacy bundle for the same
            # (slug, shape_point) but a drifted MPI source_commit
            # would be picked first by raw glob ordering, and the
            # manifest would report BUNDLE_TYPED_UNAVAILABLE even
            # when a fresh-regen scalar bundle exists.
            empty_arch_bundles: list[Path] = []
            for (b_arch, b_slug, b_shape), b_list in bundle_lookup.items():
                if b_slug == slug and b_shape == shape_point and b_arch == "":
                    empty_arch_bundles.extend(b_list)
            if empty_arch_bundles:
                bundles = _prefer_label_matched_bundle(
                    candidates=empty_arch_bundles,
                    cell=cell,
                )
        artifact_state = _check_artifact_paths(
            arch=arch, slug=slug, shape_point=shape_point,
            simbench_root=simbench_root,
            portable_lcd_root=portable_lcd_root,
        )
        bundle_result_kind: str | None = None
        bundle_typed_reason: str | None = None
        if bundles:
            bundle_result_kind, bundle_typed_reason = _bundle_result_kind(
                bundles[0],
            )
        action, action_typed_reason = _suggested_operator_action(
            artifact_state=artifact_state,
            predictor_bundle_present=bool(bundles),
            bundle_result_kind=bundle_result_kind,
            bundle_typed_reason=bundle_typed_reason,
        )
        action_counts[action] += 1
        # Per-row legacy quarantine count: how many legacy bundles
        # on disk match this cell's slug+shape but have stale MPI
        # identity fields (stale-bundle quarantine bucket from
        # build_predictor_bundle_for_cell::legacy_match_scan).
        label_5tuple: dict[str, str] | None = None
        if (
            isinstance(cell.get("binary_sha256"), str)
            and isinstance(cell.get("source_commit"), str)
            and isinstance(cell.get("kernel_signature_sha256"), str)
        ):
            label_5tuple = {
                "slug": slug,
                "shape_point": shape_point,
                "binary_sha256": cell["binary_sha256"],
                "source_commit": cell["source_commit"],
                "kernel_signature_sha256": cell["kernel_signature_sha256"],
            }
        quarantine_count = _legacy_bundle_quarantine_count_for_cell(
            slug=slug,
            shape_point=shape_point,
            label_5tuple=label_5tuple,
            bundle_lookup=bundle_lookup,
        )
        # Aggregate as a BUNDLE COUNT (sum of per-row counts),
        # not a cell count. The field's semantic is "how many
        # legacy bundle files are quarantined across the
        # cohort"; counting cells would conflate the
        # operator-facing audit surface (a cell with 38 stale
        # bundles contributes the same as a cell with 1).
        legacy_bundle_quarantined_bundles += quarantine_count
        row: dict[str, Any] = {
            "arch": arch, "slug": slug, "shape_point": shape_point,
            # Mixed-arity identity: register-cap (and future build)
            # variants share the bare (arch, slug, shape_point)
            # 3-tuple; the explicit variant_id keeps sibling rows
            # distinct for scanners that partition the cohort.
            "variant_id": cell.get("variant_id"),
            # Explicit membership marker: scanners that pin the
            # same-74 closure cohort scope to admitted rows instead
            # of inferring membership from typed-reason strings.
            "cohort_membership": "admitted",
            "predictor_bundle": {
                "present": bool(bundles),
                "paths": [str(b) for b in bundles[:3]],
            },
            "artifacts": artifact_state,
            "suggested_operator_action": action,
            "action_typed_reason": action_typed_reason,
            "legacy_bundle_quarantined_count": quarantine_count,
        }
        rows.append(row)
    # Excluded cells (typed-failure at the label layer): they never
    # had a chance to admit downstream; record them as
    # TERMINAL_TYPED_FAILURE with the relabel typed_reason.
    for excluded in excluded_cells:
        if str(excluded.get("device", "")) not in ("sm_86", "sm_120"):
            continue
        terminal_row = _terminal_typed_failure_row(excluded_cell=excluded)
        action_counts[terminal_row["suggested_operator_action"]] += 1
        rows.append(terminal_row)
    cohort_size = len(rows)
    summary = {
        "by_action": dict(action_counts),
        "ready_count": action_counts.get("READY", 0),
        "terminal_typed_failure_count": action_counts.get(
            "TERMINAL_TYPED_FAILURE", 0,
        ),
        "admitted_count": sum(
            1 for r in rows if r.get("cohort_membership") == "admitted"
        ),
        "excluded_count": sum(
            1 for r in rows if r.get("cohort_membership") == "excluded"
        ),
        "legacy_bundle_quarantined": legacy_bundle_quarantined_bundles,
    }
    artifact = {
        "schema_version": 1,
        "source_commit": _git_head_sha(),
        "cohort_size": cohort_size,
        "rows": rows,
        "summary": summary,
        "operator_notes": (
            "READY rows have a scalar predictor bundle on disk AND "
            "all three required sidecars on disk; the v5 fitter + "
            "closure + baseline runs can consume them directly. "
            "REGENERATE_SIDECARS_LOCAL invokes "
            "regenerate_sidecars_for_cell.py on the in-place "
            "dynamic_trace.pb. REBUILD_BUNDLE_LOCAL invokes the "
            "task5 cascade for cells with sidecars but no bundle. "
            "REBUILD_BINARY_LOCAL / REBUILD_TRACE_LOCAL are for "
            "cells that fail upstream of sidecar regeneration. "
            "CROSS_HOST_SCP / CROSS_HOST_SCP_SIDECARS fetch "
            "artifacts from another host when local regeneration "
            "is not possible. BUNDLE_TYPED_UNAVAILABLE rows have "
            "a typed-unavailable bundle on disk (e.g. "
            "cache_model_underconfigured) — these cells cannot be "
            "admitted by regenerating sidecars; the producer's "
            "underlying input is the bottleneck. BUNDLE_NON_SCALAR "
            "captures any other non-scalar result_kind. "
            "TERMINAL_TYPED_FAILURE rows are cells that cannot "
            "be admitted within the current producer surface: "
            "either excluded at the LABEL layer (relabel sweep "
            "stddev failure or provenance-duplicate eviction) "
            "OR walker-terminal (the predictor bundle's "
            "typed_reason is on the terminal allowlist, e.g. "
            "cache_model_underconfigured, where regenerating "
            "sidecars or re-running the cascade will not "
            "promote the bundle to scalar). Each terminal row "
            "carries its root cause in action_typed_reason. "
            "Root causes for {BUNDLE_TYPED_UNAVAILABLE, "
            "BUNDLE_NON_SCALAR, TERMINAL_TYPED_FAILURE} live in "
            "the separate `action_typed_reason` field (the "
            "legacy single-field colon-form is now "
            "schema-rejected). The summary "
            "`legacy_bundle_quarantined` count is the TOTAL "
            "number of stale-MPI legacy bundles on disk across "
            "admitted cells (sum of per-row "
            "`legacy_bundle_quarantined_count` values); a cell "
            "with 38 stale bundles contributes 38 to the "
            "summary, not 1. "
            "Per CLAUDE.md, sm_86 cells captured on a 3090 Ti "
            "host's sweeps survive raw scp but NOT git fetch; "
            "this manifest lets a future operator session see "
            "exactly which cells need which staging step."
        ),
    }
    Path(args.output).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v5_cohort_staging_manifest] {cohort_size} cells "
        f"({len(admitted_cells)} admitted + "
        f"{cohort_size - len(admitted_cells)} excluded); "
        f"action_counts={dict(action_counts)}; "
        f"legacy_bundle_quarantined={legacy_bundle_quarantined_bundles} "
        f"-> {args.output}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
