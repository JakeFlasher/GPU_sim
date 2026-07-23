"""Compute the per-source ablation matrix for a regenerated sweep
directory's activation cells and emit ``regeneration_delta.json``.

The ablation matrix has one row per admitted activation cell (5 rows
on the current 13-cell cohort) and one column per staleness source
(R16, R17, R24, R28). The per-source semantics are documented inline
on each column; the high-level summary is:

  - R16 ("cold-as-miss denominator"): the predictor's per-launch
    hit_rate now divides by ``hit + miss + cold`` (every sector
    access) instead of ``hit + miss``. Reverting this single line in
    ``cache_predictor._LaunchAggregate.hit_rate`` is mathematically
    equivalent to recomputing the mean with the old formula on the
    same per-launch aggregates; no git worktree is needed.

  - R17 ("all-cold-launch admission"): the predictor now admits any
    launch with ``access_count > 0``; the prior code filtered to
    launches with ``hit + miss > 0`` and dropped all-cold launches.
    Reverting R17 is the inverse filter on the same aggregates.

  - R24 ("comparable_cells_per_arch widening"): the comparison
    runner now counts excluded join-key-known cells into the
    per-arch comparable total. R24 changes a BUNDLE-LEVEL COUNTER,
    not any per-cell ``predicted_hit_rate``, so the per-cell
    ablation delta is mathematically zero by construction. The
    column is recorded with an explicit annotation rather than
    silently dropped, so a future round can re-frame the matrix at
    the bundle level if it wants the R24 delta computed there.

  - R28 ("NCU --launch-skip warmup offset"): the prior NCU sweep
    collected the five activation labels under the buggy
    launch-skip. R28 is a LABEL-COLLECTION-SIDE change; the
    predictor's per-cell output is unchanged by reverting it. The
    meaningful R28 effect lands in Round A1 when fresh labels are
    re-collected under the corrected launch-skip; the Round A0
    column is zero with an ``evidence_pending_a1_label_recollection``
    annotation so the structural surface stays auditable until A1
    can fill it in.

Microbench cells (8 slugs) are explicitly EXCLUDED from the per-
source ablation per DEC-3-RESOLVED: Round C replaces their binaries
with new larger-shape variants (e.g. ``trace_l2_accuracy_xl``), so
their pre-Round-C bundles have no stable identity to ablate against.
The artifact records the 8 deferred slugs in a
``microbench_cells_deferred_to_round_c`` block.

CLI:

  python3 tools/local_labels/regenerate_delta.py \\
      --sweep-dir data/local_labels/sweeps/<new_sweep> \\
      --old-bundle data/local_labels/sweeps/<prior_sweep>/comparison/comparison_bundle.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SUITE_ROOT = _REPO_ROOT / "workloads" / "simbench_bw"
_RESULTS_ROOT = _SUITE_ROOT / "results"

sys.path.insert(0, str(_SUITE_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "hrd_profiler"))

import cache_predictor as cp  # noqa: E402
import observation_builder as ob  # noqa: E402


_DEFAULT_CACHE_MODEL_CAPACITY_BYTES = 6291456  # sm_86 L2 size
_LABEL_ADMITTED_PATH = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_LABEL_EXCLUDED_PATH = _REPO_ROOT / "data" / "local_labels" / "label_excluded.json"
_R28_SUPERSEDED_REASON = "superseded_by_launch_skip_fix"


def _read_admitted_label_row(
    slug: str, shape_point: str,
) -> dict[str, Any] | None:
    """Return the post-R28 admitted row dict for one activation cell.

    Reads ``data/local_labels/label_admitted.json`` and returns the
    first matching ``(slug, shape_point)`` row. The on-disk label
    records ``cache_hit_rate_l2_sector_pct`` (percent); the caller
    converts to the predictor's fraction representation when
    computing relative_error. Returns ``None`` when no admitted
    label exists for the cell — the caller falls back to the
    pending-evidence placeholder.
    """
    if not _LABEL_ADMITTED_PATH.is_file():
        return None
    with _LABEL_ADMITTED_PATH.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for cell in payload.get("cells", []):
        if cell.get("slug") != slug or cell.get("shape_point") != shape_point:
            continue
        if cell.get("cache_hit_rate_l2_sector_pct") is None:
            continue
        return cell
    return None


def _read_admitted_label_fraction(slug: str, shape_point: str) -> float | None:
    """Back-compat wrapper that returns the admitted row's L2 hit
    rate as a fraction in ``[0, 1]``. Returns ``None`` when no
    admitted label exists for the cell.

    The deterministic-selection refresh promoted the underlying
    helper from "return-fraction" to "return-full-row" for the R28
    attribution identity contract; this wrapper preserves the
    legacy interface for any callers that still want just the
    fraction.
    """
    row = _read_admitted_label_row(slug, shape_point)
    if row is None:
        return None
    return float(row["cache_hit_rate_l2_sector_pct"]) / 100.0


def _resolve_commit_timestamps(commits: list[str]) -> dict[str, int]:
    """Return ``{source_commit: unix_timestamp}`` for each input SHA.

    Calls ``git log -1 --format=%ct <sha>`` for each commit. When the
    commit is not reachable from HEAD (e.g., it was rebased away or
    belongs to a deleted branch), the value is recorded as ``-1`` so
    the selection logic sorts the unknown candidate to the bottom.
    The caller's selection helper accepts the ``-1`` sentinel and
    never raises.
    """
    timestamps: dict[str, int] = {}
    for sha in commits:
        try:
            result = subprocess.run(
                ["git", "-C", str(_REPO_ROOT),
                 "log", "-1", "--format=%ct", sha],
                check=True, capture_output=True, text=True,
            )
            timestamps[sha] = int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            timestamps[sha] = -1
    return timestamps


def _select_pre_r28_row(
    candidates: list[dict[str, Any]],
    commit_timestamps: dict[str, int],
) -> dict[str, Any] | None:
    """Pick the pre-R28 row whose source_commit was committed most
    recently among the candidates.

    Selection rule:
      1. Order candidates by their source_commit's git commit
         timestamp (descending; latest-first).
      2. Tie-break by ``source_commit`` string descending, so the
         choice is stable under JSON file reorderings.

    Returns ``None`` when ``candidates`` is empty. Candidates whose
    ``source_commit`` is missing from ``commit_timestamps`` are
    sorted to the bottom (treated as timestamp ``-1``).

    The selection is pure (no side effects); the caller supplies
    the commit-timestamp mapping. Tests can pass synthetic
    timestamps to verify the ordering without invoking ``git``.
    """
    if not candidates:
        return None

    def _sort_key(row: dict[str, Any]) -> tuple[int, str]:
        sc = str(row.get("source_commit", ""))
        ts = commit_timestamps.get(sc, -1)
        return (ts, sc)

    return max(candidates, key=_sort_key)


def _read_pre_r28_label_row(
    slug: str, shape_point: str,
) -> dict[str, Any] | None:
    """Return the selected pre-R28 row dict for one activation cell.

    Reads ``data/local_labels/label_excluded.json``, filters by the
    ``(slug, shape_point)`` + ``typed_reason == 'superseded_by_launch_skip_fix'``
    predicate, resolves each candidate's commit timestamp via
    ``_resolve_commit_timestamps``, then picks the deterministic
    winner via ``_select_pre_r28_row``. Returns ``None`` when no
    candidate exists or the file is missing — the caller falls back
    to the structural ``evidence_pending_a1_label_recollection``
    placeholder.

    Replaces the legacy ``_read_pre_r28_label_fraction`` helper
    which returned ``candidates[-1]`` from JSON file order. See
    ``BL-20260514-nested-provenance-systemic-staleness-not-just-top-level``
    + ``BL-20260509-strict-admission-contract-first``: the audit
    contract requires the pre-R28 source identity to be explicit
    on the artifact, not implicit via JSON order.
    """
    if not _LABEL_EXCLUDED_PATH.is_file():
        return None
    with _LABEL_EXCLUDED_PATH.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    candidates: list[dict[str, Any]] = []
    for cell in payload.get("cells", []):
        if cell.get("slug") != slug or cell.get("shape_point") != shape_point:
            continue
        if cell.get("typed_reason") != _R28_SUPERSEDED_REASON:
            continue
        if cell.get("cache_hit_rate_l2_sector_pct") is None:
            continue
        candidates.append(cell)
    if not candidates:
        return None
    commit_timestamps = _resolve_commit_timestamps(
        [str(c.get("source_commit", "")) for c in candidates]
    )
    return _select_pre_r28_row(candidates, commit_timestamps)


def _relative_error(predicted: float, label: float) -> float:
    """Return ``|predicted - label| / label``, the per-cell relative
    error the comparison_runner reports. ``label`` MUST be a
    positive fraction; the caller filters zero/None labels."""
    return abs(predicted - label) / label
# Slice-geometry-aware effective capacity. The sm_86 / GA102 L2
# is split across 6 slices of 1 MiB each. Matches the
# `_DEFAULT_CACHE_MODEL_NUM_L2_SLICES` constant in the headline
# orchestrator. The `_ablate_one_cell` helper threads this into
# `build_launches_from_trace_dir` so the recomputed
# `predicted_hit_rate_at_head` reflects the slice-aware path the
# orchestrator's bundles take, AND computes a second aggregate
# with `num_l2_slices=1` for the
# `r54_slice_geometry_pass_through_delta` ablation column.
_DEFAULT_CACHE_MODEL_NUM_L2_SLICES = 6


_ACTIVATION_CELLS_5: tuple[tuple[str, str], ...] = (
    ("layernorm_fwd_bf16_sm80", "timing_shape"),
    ("rmsnorm_fwd_bf16_sm80", "timing_shape"),
    ("silu_fwd_bf16_sm80", "correctness_shape"),
    ("silu_fwd_bf16_sm80", "timing_shape"),
    ("softmax_safe_bf16_sm80", "timing_shape"),
)


# Mirrors the durable 8-slug microbench identity set in
# ``tools/local_labels/shadow_headline_diagnostics.py``. Kept in
# lockstep with the ``_PREDICTOR_DESIGN_COVERAGE_BY_CELL`` microbench
# rows in ``workloads/simbench_bw/comparison_runner.py``.
_MICROBENCH_SLUGS_8: tuple[str, ...] = (
    "working_set_capacity_sweep",
    "l2_thrash_pointer_chase",
    "dram_stream_stride_mix",
    "carveout_sensitivity_probe",
    "l2_persisting_window_sweep",
    "l2_prefetch_hint_sweep",
    "warm_cold_pair_relaunch",
    "register_spill_local_mem",
)


def _git_head_commit(repo_root: Path) -> str:
    """Return the full SHA of repo_root's HEAD."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _trace_dir(slug: str, shape_point: str) -> Path:
    return (
        _RESULTS_ROOT
        / slug
        / "rtx_3090_ti"
        / "sm_86"
        / f"trace_{shape_point}"
    )


def _scalar_with_current_rules(aggregates: list[Any]) -> float:
    """Per-launch hit_rate = hit / (hit + miss + cold); admit any
    launch with access_count > 0; mean across admitted launches.
    Matches ``cache_predictor.predict_cache_hit_rate``'s scalar
    rule at lines 645-720."""
    populated = [a for a in aggregates if a.access_count > 0]
    if not populated:
        return 0.0
    per_launch = [a.hit_count / a.access_count for a in populated]
    return sum(per_launch) / len(per_launch)


def _scalar_with_r16_reverted(aggregates: list[Any]) -> float:
    """Per-launch hit_rate = hit / (hit + miss); drop all-cold
    launches (zero divisor); admit launches with access_count > 0;
    mean across admitted launches with finite divisor.
    Matches the pre-R16 cache_predictor formula."""
    populated = [a for a in aggregates if a.access_count > 0]
    finite = [
        a for a in populated
        if (a.hit_count + a.miss_count) > 0
    ]
    if not finite:
        return 0.0
    per_launch = [
        a.hit_count / (a.hit_count + a.miss_count)
        for a in finite
    ]
    return sum(per_launch) / len(per_launch)


def _scalar_with_r17_reverted(aggregates: list[Any]) -> float:
    """Per-launch hit_rate = hit / (hit + miss + cold); admit ONLY
    launches with (hit + miss) > 0; mean. Drops all-cold launches
    from the per-trace mean per the pre-R17 filter (the bug R17
    fixed)."""
    finite = [
        a for a in aggregates
        if (a.hit_count + a.miss_count) > 0
    ]
    if not finite:
        return 0.0
    per_launch = [a.hit_count / a.access_count for a in finite]
    return sum(per_launch) / len(per_launch)


def _build_aggregates(
    trace_dir: Path,
    *,
    num_l2_slices: int,
) -> list[Any]:
    """Walk the trace with a specific slice count and return the
    per-launch aggregate list. Factored out of `_ablate_one_cell`
    so the `r54_slice_geometry_pass_through_delta` row can
    invoke this twice (current slice-aware vs reverted
    fully-associative) without duplicating the contract-ok
    handling."""
    build_result = ob.build_launches_from_trace_dir(
        trace_dir,
        capacity_bytes=_DEFAULT_CACHE_MODEL_CAPACITY_BYTES,
        num_l2_slices=num_l2_slices,
    )
    if not getattr(build_result, "contract_ok", True):
        raise RuntimeError(
            f"trace contract failed at {trace_dir}: "
            f"{getattr(build_result, 'contract_typed_reason', None)!r}"
        )
    launches = getattr(build_result, "launches", build_result)
    return cp._aggregate_launches_from_observations(launches)


_R28_PENDING_ANNOTATION = (
    "NCU --launch-skip warmup offset: label-collection-side change. "
    "Per-cell predicted_hit_rate is unaffected by this label-side "
    "change. value_with_change and value_with_change_reverted are "
    "recorded as the same predicted value so the structural value-"
    "pair contract holds; the post-A1 refresh will overwrite the "
    "pair with the actual pre/post launch-skip relative-error impact."
)


_R28_LIVE_ANNOTATION = (
    "NCU --launch-skip warmup offset: label-collection-side change. "
    "Pre-R28 labels were collected with --launch-skip=0 (profiling "
    "the first warmup launch); post-R28 labels use the corrected "
    "--launch-skip value (admitted by the post-supersession activation "
    "sweep). Per-cell predicted_hit_rate is unchanged (the predictor "
    "reads traces, not labels). value_with_change records the "
    "predictor-vs-label relative_error against the POST-R28 admitted "
    "label; value_with_change_reverted records the relative_error "
    "against the PRE-R28 superseded label. delta = value_with_change "
    "- value_with_change_reverted captures the actual R28 impact on "
    "the bundle's per-cell relative_error."
)


def _build_r28_delta(slug: str, shape_point: str, current: float) -> dict[str, Any]:
    """Build the r28_delta block for one activation cell.

    When both pre-R28 (``superseded_by_launch_skip_fix`` in
    ``label_excluded.json``) and post-R28 (admitted in
    ``label_admitted.json``) labels exist for the cell, the block
    records the predictor-vs-label relative_error on both sides;
    delta is the change in relative_error attributable to R28.
    When either is missing, the block falls back to the structural
    placeholder ``(predicted, predicted, delta=0,
    evidence_pending_a1_label_recollection=True)`` introduced by
    the initial structural pass; future refresh runs replace the
    placeholder once the missing label admits.

    Identity-stamped contract: when both rows are present, the
    block also records the selected pre-R28 source identity
    (``pre_r28_source_commit`` + ``pre_r28_label_hit_rate_pct``)
    and the post-R28 hit-rate (``post_r28_label_hit_rate_pct``)
    so the value pair is reproducible from
    ``label_admitted.json`` + ``label_excluded.json`` without
    re-running the producer's selection logic. The pre-R28 row is
    chosen deterministically by latest git commit timestamp via
    ``_read_pre_r28_label_row``.
    """
    pre_row = _read_pre_r28_label_row(slug, shape_point)
    post_row = _read_admitted_label_row(slug, shape_point)
    if pre_row is None or post_row is None:
        return {
            "delta": 0.0,
            "value_with_change": round(current, 6),
            "value_with_change_reverted": round(current, 6),
            "annotation": _R28_PENDING_ANNOTATION,
            "evidence_pending_a1_label_recollection": True,
        }
    pre_pct = float(pre_row["cache_hit_rate_l2_sector_pct"])
    post_pct = float(post_row["cache_hit_rate_l2_sector_pct"])
    pre = pre_pct / 100.0
    post = post_pct / 100.0
    if pre <= 0 or post <= 0:
        return {
            "delta": 0.0,
            "value_with_change": round(current, 6),
            "value_with_change_reverted": round(current, 6),
            "annotation": _R28_PENDING_ANNOTATION,
            "evidence_pending_a1_label_recollection": True,
        }
    rel_err_post = _relative_error(current, post)
    rel_err_pre = _relative_error(current, pre)
    return {
        "delta": round(rel_err_post - rel_err_pre, 6),
        "value_with_change": round(rel_err_post, 6),
        "value_with_change_reverted": round(rel_err_pre, 6),
        "annotation": _R28_LIVE_ANNOTATION,
        "evidence_pending_a1_label_recollection": False,
        "pre_r28_source_commit": str(pre_row.get("source_commit", "")),
        "pre_r28_label_hit_rate_pct": pre_pct,
        "post_r28_label_hit_rate_pct": post_pct,
    }


def _ablate_one_cell(slug: str, shape_point: str) -> dict[str, Any]:
    """Walk the trace, aggregate under both slice-aware and
    fully-associative paths, and compute the 5-column ablation
    row for one activation cell.

    The current/`predicted_hit_rate_at_head` aggregate is
    computed with `num_l2_slices=6` so the recomputed value
    matches the headline orchestrator's slice-aware bundles.
    A second aggregate with `num_l2_slices=1` powers the
    `r54_slice_geometry_pass_through_delta` column.
    """
    trace_dir = _trace_dir(slug, shape_point)
    if not (trace_dir / "dynamic_trace.pb").is_file():
        raise FileNotFoundError(
            f"missing dynamic_trace.pb for {slug}/{shape_point} at "
            f"{trace_dir}"
        )

    aggregates_current = _build_aggregates(
        trace_dir,
        num_l2_slices=_DEFAULT_CACHE_MODEL_NUM_L2_SLICES,
    )
    aggregates_default1 = _build_aggregates(
        trace_dir,
        num_l2_slices=1,
    )

    current = _scalar_with_current_rules(aggregates_current)
    r16_reverted = _scalar_with_r16_reverted(aggregates_current)
    r17_reverted = _scalar_with_r17_reverted(aggregates_current)
    default1 = _scalar_with_current_rules(aggregates_default1)

    return {
        "slug": slug,
        "shape_point": shape_point,
        "predicted_hit_rate_at_head": round(current, 6),
        "r16_delta": {
            "delta": round(current - r16_reverted, 6),
            "value_with_change": round(current, 6),
            "value_with_change_reverted": round(r16_reverted, 6),
            "annotation": (
                "predictor _LaunchAggregate.hit_rate denominator: "
                "current = hit / (hit + miss + cold); reverted = "
                "hit / (hit + miss), drop all-cold launches."
            ),
        },
        "r17_delta": {
            "delta": round(current - r17_reverted, 6),
            "value_with_change": round(current, 6),
            "value_with_change_reverted": round(r17_reverted, 6),
            "annotation": (
                "predictor launch filter: current admits any launch "
                "with access_count > 0; reverted admits only "
                "launches with hit + miss > 0 (drops all-cold)."
            ),
        },
        "r24_delta": {
            "delta": 0.0,
            "value_with_change": round(current, 6),
            "value_with_change_reverted": round(current, 6),
            "annotation": (
                "comparison-runner comparable_cells_per_arch tally "
                "widening: changes a BUNDLE-LEVEL counter only, not "
                "per-cell predicted_hit_rate; per-cell delta is "
                "mathematically zero by construction. "
                "value_with_change and value_with_change_reverted "
                "are recorded as the same predicted value so the "
                "structural value-pair contract holds even when "
                "the per-cell delta is zero."
            ),
        },
        "r28_delta": _build_r28_delta(slug, shape_point, current),
        "r54_slice_geometry_pass_through_delta": {
            "delta": round(current - default1, 6),
            "value_with_change": round(current, 6),
            "value_with_change_reverted": round(default1, 6),
            "annotation": (
                "slice-aware effective capacity: current = "
                "capacity_bytes // num_l2_slices // 32 (sm_86 "
                "GA102 num_l2_slices=6 -> per-slice capacity "
                "32768 sectors = 1 MiB); reverted = "
                "capacity_bytes // 32 (fully-associative pool, "
                "196608 sectors = 6 MiB). Cells whose per-launch "
                "working set fits within a single slice's 1 MiB "
                "report zero delta (the same sectors are hits "
                "under both predicates); cells that overflow "
                "per-slice capacity report a negative delta "
                "(slice-aware predicts fewer hits)."
            ),
        },
        "per_launch_aggregate_summary": {
            "launch_count": len(aggregates_current),
            "populated_launch_count": sum(
                1 for a in aggregates_current if a.access_count > 0
            ),
            "finite_launch_count": sum(
                1 for a in aggregates_current
                if (a.hit_count + a.miss_count) > 0
            ),
        },
    }


def compute_delta(
    sweep_dir: Path,
    old_bundle: Path | None = None,
) -> dict[str, Any]:
    """Compute the full regeneration_delta payload for one sweep."""
    sweep_dir = (
        sweep_dir if sweep_dir.is_absolute()
        else (_REPO_ROOT / sweep_dir).resolve()
    )
    new_bundle_path = sweep_dir / "comparison_bundle.json"
    if not new_bundle_path.is_file():
        raise FileNotFoundError(
            f"new comparison_bundle.json missing at {new_bundle_path}"
        )
    with new_bundle_path.open("r", encoding="utf-8") as fh:
        new_bundle = json.load(fh)

    activation_rows: list[dict[str, Any]] = []
    for slug, shape_point in _ACTIVATION_CELLS_5:
        activation_rows.append(_ablate_one_cell(slug, shape_point))

    trace_source_commit = None
    for cell in (new_bundle.get("included_cells") or []):
        sc = (cell.get("join_key") or {}).get("source_commit")
        if sc:
            trace_source_commit = sc
            break

    old_bundle_summary: dict[str, Any] = {}
    if old_bundle is not None:
        old_bundle_abs = (
            old_bundle if old_bundle.is_absolute()
            else (_REPO_ROOT / old_bundle).resolve()
        )
        if old_bundle_abs.is_file():
            with old_bundle_abs.open("r", encoding="utf-8") as fh:
                old_payload = json.load(fh)
            try:
                rel = str(old_bundle_abs.relative_to(_REPO_ROOT))
            except ValueError:
                rel = str(old_bundle_abs)
            old_bundle_summary = {
                "path": rel,
                "geomean": old_payload.get("geomean"),
                "geomean_ci_low": old_payload.get("geomean_ci_low"),
                "geomean_ci_high": old_payload.get("geomean_ci_high"),
            }

    return {
        "schema_version": 1,
        "regeneration_build_commit": _git_head_commit(_REPO_ROOT),
        "trace_source_commit": trace_source_commit,
        "new_comparison_bundle_relpath": str(
            new_bundle_path.relative_to(_REPO_ROOT)
        ),
        "old_comparison_bundle": old_bundle_summary or None,
        "headline_summary": {
            "geomean": new_bundle.get("geomean"),
            "geomean_ci_low": new_bundle.get("geomean_ci_low"),
            "geomean_ci_high": new_bundle.get("geomean_ci_high"),
            "included_cell_count": len(
                new_bundle.get("included_cells") or []
            ),
            "excluded_cell_count": len(
                new_bundle.get("excluded_cells") or []
            ),
        },
        "activation_cell_ablations": activation_rows,
        "microbench_cells_deferred_to_round_c": {
            "slugs": list(_MICROBENCH_SLUGS_8),
            "reason": (
                "Microbench cells are rehabilitated in Round C via "
                "new larger-shape variants (DEC-3-RESOLVED); the "
                "pre-Round-C binaries have no stable identity to "
                "ablate against. Per-source ablation re-runs for "
                "any microbench slug MUST be flagged as out-of-"
                "scope by the regeneration_delta scanner."
            ),
        },
    }


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="regenerate_delta.py",
        description=(
            "Compute the per-source ablation matrix for a "
            "regenerated sweep directory's activation cells and "
            "emit regeneration_delta.json beside its "
            "comparison_bundle.json."
        ),
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help=(
            "Path to the new sweep directory containing "
            "comparison_bundle.json + predictor_bundles/."
        ),
    )
    parser.add_argument(
        "--old-bundle",
        type=Path,
        default=None,
        help=(
            "Optional path to a prior comparison_bundle.json for "
            "headline-summary delta context."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for regeneration_delta.json (default: "
            "<sweep-dir>/regeneration_delta.json)."
        ),
    )
    args = parser.parse_args(argv)
    payload = compute_delta(args.sweep_dir, args.old_bundle)
    output = args.output or (args.sweep_dir / "regeneration_delta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
