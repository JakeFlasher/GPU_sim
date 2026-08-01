"""REAL golden-set fidelity validation over captured new-contract trace dirs.

Unlike the controlled unit smoke in `golden_set.py`, this loads REAL trace dirs
listed in a manifest (fail-closed on any missing artifact — no synthetic
fallback), computes full-shape L2 metrics from the REAL decoder
(`cache_replay.sector_cache_replay`), builds REAL wave-segment subsets, and scores
the REAL `estimator.estimate` with BOTH count AND rate targets against every
declared gate plus the sampling-induced error budget against the REAL NCU label.
`pass` is the AND of every required gate.

Per-segment measurements use ghost differencing: replay each
segment's ghost wave from cold with counters muted, replay ghost plus measured
waves from cold, and subtract.  The all-segment ghost-differenced census is the fidelity
estimand.  A separate selection ablation compares those ghost-differenced wave
trials with independently drawn block subsets replayed cold-once against the
production full-trace reference; the flat cache model cannot support a
mechanistic co-residency claim.

The NCU L2 hit rate is a validation target for the error budget only, never a
model, card, or trace input.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parents[1]
for _p in (str(_PKG), str(_REPO), str(_REPO / "tools" / "cache_replay"),
           str(_REPO / "tools" / "local_labels")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)

from admissibility_gate import M_SUBSET, MIN_SEGMENTS, classify
from counting_pass import analyze_trace_dir
from estimator import RateTarget, Segment, Stratum, estimate
from golden_provenance import build_golden_provenance
from replay_count_producer import first_launch_dir
from selection import SelectionPlanError, build_segment_population, build_selection_plan

# Gates (the golden-set acceptance thresholds).
COUNT_TOL = 0.05           # p95 relative error on additive counts
RATE_TOL = 0.02            # p95 absolute error on rates
CI_COVERAGE_MIN = 0.90
RATE_CI_HALFWIDTH = 0.02
COUNT_CI_HALFWIDTH_REL = 0.05
GHOST_DEPTH_RATE_MOVEMENT_TOL = 0.01
ERROR_BUDGET_FACTOR = 0.3  # sampling error <= 0.3 * full-shape model MAE (vs NCU)
FRACTIONS = (1 / 16, 1 / 8, 1 / 4)
N_SEEDS = 5
# Only a floating-point comparison tolerance is allowed in the strict
# wave-segment-not-worse ablation gate; this is not an accuracy margin.
ABLATION_COMPARE_EPS = 1e-12
ABLATION_POLICY_VERSION = "fixed-design-composite-v1"
GOLDEN_MANIFEST_POLICY_VERSION = "real-golden-manifest-v1"
GOLDEN_MANIFEST_SHA256 = (
    "ecc40493d77f7b400f512a89b1b6c31d87798281ea27cab5af9b6299e429962b")
GOLDEN_ABLATION_EXPECTATIONS = {
    "gs_stream_copy": "neutral",
    "gs_stream_reuse": "co_residency_sensitive",
    "gs_var_stride": "neutral",
}
TRACER_VERSION = "nvbit-enhanced-tracer-v5"
STATS_VERSION = "counting-pass-sector-aux-v2"
ESTIMATOR_VERSION = "phase-tercile-neyman-srs-ht-ghost-v2"

# sm_86 L2 geometry (arch_profiles/cards/sm_86.json) + Accel-Sim Ampere defaults.
_L2 = dict(capacity_bytes=6291456, num_slices=6, associativity=16,
           replacement_policy="LRU", slice_hash_id="line_mod")


class GoldenSetError(RuntimeError):
    """Typed fail-closed error (missing real input, absent denominator, ...)."""


def _cache_cfg():
    import sector_cache_replay as scr  # type: ignore
    return scr.CacheReplayConfig(**_L2)


def _pb_files(trace_dir: Path) -> list[Path]:
    launch_dir, _kernel_id = first_launch_dir(trace_dir)
    return sorted(launch_dir.rglob("*.pb"))


def _block_ids_by_path(
        pb_paths: list[Path], gx: int, gy: int) -> dict[Path, int]:
    """Decode each block identity once under one authenticated capability batch."""
    bindings = load_dynamic_trace_bindings()
    threadblock_cls = bindings.threadblock_pb2.threadblock
    block_ids: dict[Path, int] = {}
    for path in pb_paths:
        tb = threadblock_cls()
        tb.ParseFromString(path.read_bytes())
        block_ids[path] = (
            int(tb.block_id.x)
            + int(tb.block_id.y) * gx
            + int(tb.block_id.z) * gx * gy
        )
    # Detect retained binding mutation before the decoded identity map escapes.
    load_dynamic_trace_bindings()
    return block_ids


def _segment_membership(stats) -> dict[int, int]:
    """Return the production population's exhaustive block-to-segment map."""
    population = build_segment_population(stats)
    return {
        block_id: segment.id
        for segment in population.segments for block_id in segment.block_ids
    }


@contextmanager
def _subset_trace_dir(trace_dir: Path, pb_subset: list[Path], scratch: Path):
    """Expose selected block payloads as a temporary trace directory."""
    work = Path(tempfile.mkdtemp(prefix="gs_sub_", dir=scratch))
    try:
        (work / "dynamic_trace.pb").symlink_to((trace_dir / "dynamic_trace.pb").resolve())
        ei = trace_dir / "extra_info"
        if ei.is_dir():
            (work / "extra_info").symlink_to(ei.resolve())
        launch_dir, _kernel_id = first_launch_dir(trace_dir)
        kd = work / launch_dir.relative_to(trace_dir)
        kd.mkdir(parents=True)
        for p in pb_subset:
            (kd / p.name).symlink_to(p.resolve())
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _decode_access_subset(trace_dir: Path, pb_subset: list[Path],
                          scratch: Path) -> tuple[list[int], list[int]]:
    """Decode selected whole blocks without resetting cache state between groups."""
    import sector_cache_replay as scr  # type: ignore

    with _subset_trace_dir(trace_dir, pb_subset, scratch) as work:
        stream = scr.decode_cell_access_stream(work)
        return stream.line_ids, stream.sector_offsets


def _decode_subset(trace_dir: Path, pb_subset: list[Path], scratch: Path) -> tuple[float, float]:
    """Decode + L2-replay a subset of whole threadblock payloads."""
    import sector_cache_replay as scr  # type: ignore

    with _subset_trace_dir(trace_dir, pb_subset, scratch) as work:
        stream = scr.decode_cell_access_stream(work)
        res = scr.replay_access_stream(stream.line_ids, stream.sector_offsets,
                                       _cache_cfg(), classify_misses=False)
        return float(res.total_accesses), float(res.hits)


def _replay_accesses(line_ids: list[int], sector_offsets: list[int]) -> tuple[float, float]:
    """Replay an already ordered access stream through the production L2 model."""
    import sector_cache_replay as scr  # type: ignore

    res = scr.replay_access_stream(
        line_ids, sector_offsets, _cache_cfg(), classify_misses=False)
    return float(res.total_accesses), float(res.hits)


def _full_shape_rate_interval(raw_lo: float, raw_hi: float) -> tuple[float, float, float]:
    """Add the predeclared warm-state allowance without consulting full truth.

    The third value is the conservative half-width bound used by the stopping
    gate: raw bootstrap half-width plus the one-ghost approximation allowance.
    """
    if not 0.0 <= raw_lo <= raw_hi <= 1.0:
        raise GoldenSetError(f"invalid_raw_rate_interval: {(raw_lo, raw_hi)}")
    return (
        max(0.0, raw_lo - GHOST_DEPTH_RATE_MOVEMENT_TOL),
        min(1.0, raw_hi + GHOST_DEPTH_RATE_MOVEMENT_TOL),
        (raw_hi - raw_lo) / 2 + GHOST_DEPTH_RATE_MOVEMENT_TOL,
    )


def _segment_layout(
        stats, block_ids_by_path: dict[Path, int]) -> list[dict]:
    """Freeze each segment's whole-block ghost/measured membership.

    Complete segments contain one ghost wave plus two measured waves.  Modulo-3
    residual complete waves and scheduler-underfilled waves form one
    probability-1 tail; its first wave is ghost and every later wave is measured.
    """
    block_of = {b.block_id: b for b in stats.blocks}
    population = build_segment_population(stats)
    seg_of = {
        block_id: segment.id
        for segment in population.segments for block_id in segment.block_ids
    }
    ghost_wave_of = {
        segment.id: segment.wave_ids[0] for segment in population.segments
    }
    layouts: dict[int, dict] = {}
    for path in sorted(block_ids_by_path, key=block_ids_by_path.__getitem__):
        idx = block_ids_by_path[path]
        block = block_of[idx]
        segment_id = seg_of[idx]
        layout = layouts.setdefault(segment_id, {
            "id": segment_id,
            "ghost_paths": [],
            "measured_paths": [],
            "ghost_records": 0,
            "measured_records": 0,
            "waves": set(),
            "paths_by_wave": {},
            "records_by_wave": {},
        })
        role = "ghost" if block.wave == ghost_wave_of[segment_id] else "measured"
        layout[f"{role}_paths"].append(path)
        layout[f"{role}_records"] += block.records
        layout["waves"].add(block.wave)
        layout["paths_by_wave"].setdefault(block.wave, []).append(path)
        layout["records_by_wave"][block.wave] = (
            layout["records_by_wave"].get(block.wave, 0) + block.records)

    out = []
    for segment_id in sorted(layouts):
        layout = layouts[segment_id]
        if not layout["ghost_paths"] or not layout["measured_paths"]:
            raise GoldenSetError(
                "invalid_ghost_measured_segment: "
                f"segment={segment_id}, ghost_blocks={len(layout['ghost_paths'])}, "
                f"measured_blocks={len(layout['measured_paths'])}")
        layout["waves"] = sorted(layout["waves"])
        out.append(layout)
    return out


def _segment_measurements(
        trace_dir: Path, stats, scratch: Path,
        block_ids_by_path: dict[Path, int] | None = None) -> tuple[
        list[dict], list[dict], dict[str, float], dict[str, float | int],
        dict[Path, int]]:
    """Measure every segment with per-segment ghost differencing.

    Both replays use the production decoder/cache path and start cold.  The
    ghost-only result is subtracted from ghost-plus-measured, so only the latter
    waves contribute metrics while retaining one wave of warm-up state.
    """
    if block_ids_by_path is None:
        gx, gy, _gz = stats.grid
        block_ids_by_path = _block_ids_by_path(_pb_files(trace_dir), gx, gy)
    layouts = _segment_layout(stats, block_ids_by_path)
    depth1 = []
    oracle = {"sectors": 0.0, "hits": 0.0, "records": 0.0}
    sensitivity_segment_id = layouts[len(layouts) // 2]["id"]
    paths_by_wave = {
        wave: paths
        for layout in layouts
        for wave, paths in layout["paths_by_wave"].items()
    }
    sensitivity: dict[str, float | int] | None = None
    for layout in layouts:
        wave_streams = [
            _decode_access_subset(
                trace_dir, layout["paths_by_wave"][wave], scratch)
            for wave in layout["waves"]
        ]
        ghost_lines, ghost_offsets = wave_streams[0]
        measured_lines: list[int] = []
        measured_offsets: list[int] = []
        for wave_lines, wave_offsets in wave_streams[1:]:
            measured_lines.extend(wave_lines)
            measured_offsets.extend(wave_offsets)
        ghost_sectors, ghost_hits = _replay_accesses(ghost_lines, ghost_offsets)
        # Do not decode the union: the production decoder sorts CTAs and could
        # interleave roles.  Concatenating independently decoded streams freezes
        # the binding cold -> ghost -> measured order.
        combined_sectors, combined_hits = _replay_accesses(
            ghost_lines + measured_lines, ghost_offsets + measured_offsets)
        depth1_measurement = {
            "id": layout["id"],
            "sectors": combined_sectors - ghost_sectors,
            "hits": combined_hits - ghost_hits,
            "records": float(layout["measured_records"]),
        }
        if (depth1_measurement["sectors"] < 0 or depth1_measurement["hits"] < 0
                or depth1_measurement["hits"] > depth1_measurement["sectors"]
                or depth1_measurement["records"] <= 0):
            raise GoldenSetError(
                "ghost_differencing_invariant_failed: "
                f"segment={layout['id']}, ghost_depth=1, "
                f"measurement={depth1_measurement}")
        oracle["sectors"] += depth1_measurement["sectors"]
        oracle["hits"] += depth1_measurement["hits"]
        oracle["records"] += depth1_measurement["records"]
        depth1.append(depth1_measurement)

        # The warm-state check varies only the prefix before one FIXED measured
        # wave; it must not compare different measured populations.
        if layout["id"] == sensitivity_segment_id:
            previous_wave = layout["waves"][0] - 1
            if previous_wave not in paths_by_wave:
                raise GoldenSetError("ghost_depth_sensitivity_prefix_missing")
            previous_lines, previous_offsets = _decode_access_subset(
                trace_dir, paths_by_wave[previous_wave], scratch)
            d0_sectors, d0_hits = _replay_accesses(
                measured_lines, measured_offsets)
            d2_ghost_lines = previous_lines + ghost_lines
            d2_ghost_offsets = previous_offsets + ghost_offsets
            d2_ghost_sectors, d2_ghost_hits = _replay_accesses(
                d2_ghost_lines, d2_ghost_offsets)
            d2_combined_sectors, d2_combined_hits = _replay_accesses(
                d2_ghost_lines + measured_lines,
                d2_ghost_offsets + measured_offsets)
            depth_rates = {
                "depth0_rate": d0_hits / d0_sectors if d0_sectors else 0.0,
                "depth1_rate": (
                    depth1_measurement["hits"] / depth1_measurement["sectors"]),
                "depth2_rate": ((d2_combined_hits - d2_ghost_hits)
                                / (d2_combined_sectors - d2_ghost_sectors)),
            }
            sensitivity = {
                "segment_id": sensitivity_segment_id,
                **depth_rates,
                "movement_0_to_1": abs(
                    depth_rates["depth0_rate"] - depth_rates["depth1_rate"]),
                "movement_1_to_2": abs(
                    depth_rates["depth1_rate"] - depth_rates["depth2_rate"]),
            }
    if sensitivity is None:
        raise GoldenSetError("ghost_depth_sensitivity_segment_missing")
    return depth1, layouts, oracle, sensitivity, block_ids_by_path


def _selection_plan(population, fraction: float, seed: int):
    """Return the exact production stratification/allocation plan for a trial."""
    try:
        return build_selection_plan(population, fraction=fraction, seed=seed)
    except SelectionPlanError as exc:
        raise GoldenSetError(
            "insufficient_segment_population_for_fraction_ladder: "
            f"fraction={fraction}, population={len(population.segments)}: {exc}") from exc


def _sampled_segment_ids(population, fraction: float, seed: int) -> list[int]:
    """Use the production phase×tercile Neyman plan without substitutions."""
    plan = _selection_plan(population, fraction, seed)
    return [segment.id for segment in plan.selected_segments]


def _fraction_design(population) -> list[dict]:
    """Validate that every nominal fraction is exercised without floor clamping."""
    n_total = len(population.segments)
    if n_total <= 0:
        raise GoldenSetError("insufficient_segment_population_for_fraction_ladder: empty")
    design = []
    for fraction in FRACTIONS:
        ids = _sampled_segment_ids(population, fraction, 0)
        selected = len(ids)
        realized = selected / n_total
        rounding_tolerance = 0.5 / n_total
        if abs(realized - fraction) > rounding_tolerance + 1e-12:
            raise GoldenSetError(
                "fraction_ladder_rounding_mismatch: "
                f"fraction={fraction}, population={n_total}, realized={realized}")
        design.append({
            "requested_fraction": fraction,
            "population_segments": n_total,
            "ordinary_population_segments": len(population.ordinary_segments),
            "certainty_segment_ids": [
                segment.id for segment in population.certainty_segments],
            "selected_segments": selected,
            "realized_fraction": realized,
        })
    selected_counts = [row["selected_segments"] for row in design]
    if any(b <= a for a, b in zip(selected_counts, selected_counts[1:])):
        raise GoldenSetError(
            "fraction_ladder_collapsed: "
            f"population={n_total}, selected_segments={selected_counts}")
    return design


def _p95(values: list[float]) -> float | None:
    """Return the fixed-design p95, or no statistic for an empty ledger."""
    return float(np.percentile(values, 95)) if values else None


def _manifest_policy_status(manifest_bytes: bytes) -> tuple[str, bool]:
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    return digest, digest == GOLDEN_MANIFEST_SHA256


def _evaluate_ablation_composite(
        ablations: list[dict], expected_expectations: dict[str, str]) -> dict:
    """Recompute the binding ablation verdict from paired trial ledgers.

    Cell summaries are intentionally ignored.  Every statistic and gate is
    rebuilt from estimates, the common production reference, payload budgets,
    and the exact fraction-by-seed grid supplied in ``ablations``.
    """
    failures: list[str] = []

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    expected_policy = dict(expected_expectations)
    if not expected_policy:
        fail("empty_expected_cell_set")
    for slug, expectation in expected_policy.items():
        if (not isinstance(slug, str) or not slug
                or expectation not in {"neutral", "co_residency_sensitive"}):
            fail(f"invalid_expected_cell_policy:{slug!r}")
    expected_set = set(expected_policy)

    rows_by_slug: dict[str, dict] = {}
    actual: list[str] = []
    for index, row in enumerate(ablations):
        if not isinstance(row, dict) or not isinstance(row.get("slug"), str):
            fail(f"invalid_cell_ledger:{index}")
            continue
        slug = row["slug"]
        actual.append(slug)
        if slug in rows_by_slug:
            fail(f"duplicate_cell_coverage:{slug}")
            continue
        rows_by_slug[slug] = row
    if set(actual) != expected_set:
        fail(
            "incomplete_cell_coverage:"
            f"expected={sorted(expected_set)},actual={sorted(set(actual))}")

    expected_grid = {
        (float(fraction), seed)
        for fraction in FRACTIONS for seed in range(N_SEEDS)
    }
    error_keys = (
        "wave_segment_count_error_rel",
        "independent_block_count_error_rel",
        "wave_segment_rate_error_abs",
        "independent_block_rate_error_abs",
    )
    pooled = {key: [] for key in error_keys}
    cell_metrics: dict[str, dict] = {}
    sensitive_slugs: list[str] = []
    evidence_valid = not failures

    def finite_number(value) -> bool:
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value))
        )

    def valid_estimate(estimate) -> bool:
        return (
            isinstance(estimate, dict)
            and finite_number(estimate.get("sectors"))
            and float(estimate["sectors"]) > 0.0
            and finite_number(estimate.get("hits"))
            and 0.0 <= float(estimate["hits"]) <= float(estimate["sectors"])
            and finite_number(estimate.get("l2_hit_rate"))
            and 0.0 <= float(estimate["l2_hit_rate"]) <= 1.0
            and abs(
                float(estimate["l2_hit_rate"])
                - float(estimate["hits"]) / float(estimate["sectors"])
            ) <= ABLATION_COMPARE_EPS
        )

    for slug, row in rows_by_slug.items():
        cell_valid = True
        expectation = row.get("expectation")
        if expectation not in {"neutral", "co_residency_sensitive"}:
            fail(f"{slug}:invalid_ablation_expectation")
            cell_valid = False
        elif expectation != expected_policy.get(slug):
            fail(
                f"{slug}:ablation_expectation_policy_mismatch:"
                f"expected={expected_policy.get(slug)!r},actual={expectation!r}")
            cell_valid = False
        elif expectation == "co_residency_sensitive":
            sensitive_slugs.append(slug)

        reference = row.get("common_production_reference")
        reference_valid = (
            isinstance(reference, dict)
            and finite_number(reference.get("sectors"))
            and float(reference["sectors"]) > 0.0
            and finite_number(reference.get("hits"))
            and 0.0 <= float(reference["hits"]) <= float(reference["sectors"])
            and finite_number(reference.get("rate"))
            and 0.0 <= float(reference["rate"]) <= 1.0
            and abs(
                float(reference["rate"])
                - float(reference["hits"]) / float(reference["sectors"])
            ) <= ABLATION_COMPARE_EPS
        )
        if not reference_valid:
            fail(f"{slug}:invalid_common_production_reference")
            cell_valid = False

        trials = row.get("trials")
        if not isinstance(trials, list):
            fail(f"{slug}:invalid_trial_ledger")
            trials = []
            cell_valid = False
        trial_grid: list[tuple[float, int]] = []
        cell_errors = {key: [] for key in error_keys}
        for trial_index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                fail(f"{slug}:invalid_trial:{trial_index}")
                cell_valid = False
                continue
            fraction = trial.get("requested_fraction")
            seed = trial.get("seed")
            if (not finite_number(fraction)
                    or not isinstance(seed, int) or isinstance(seed, bool)):
                fail(f"{slug}:invalid_trial_grid_point:{trial_index}")
                cell_valid = False
            else:
                trial_grid.append((float(fraction), seed))

            wave_payload = trial.get("wave_segment_payload")
            block_payload = trial.get("independent_block_payload")
            wave_blocks = (wave_payload.get("blocks")
                           if isinstance(wave_payload, dict) else None)
            block_blocks = (block_payload.get("blocks")
                            if isinstance(block_payload, dict) else None)
            valid_budgets = (
                isinstance(wave_blocks, int) and not isinstance(wave_blocks, bool)
                and isinstance(block_blocks, int) and not isinstance(block_blocks, bool)
                and wave_blocks > 0 and block_blocks > 0
            )
            if not valid_budgets or wave_blocks != block_blocks:
                fail(f"{slug}:ablation_block_budget_mismatch")
                cell_valid = False
            witness = trial.get("independent_partial_segment_witness")
            witness_population = (witness.get("segment_block_ids")
                                  if isinstance(witness, dict) else None)
            witness_selected = (witness.get("selected_block_ids")
                                if isinstance(witness, dict) else None)
            witness_segment_id = (witness.get("segment_id")
                                  if isinstance(witness, dict) else None)
            witness_lists_valid = (
                isinstance(witness_population, list)
                and isinstance(witness_selected, list)
                and all(isinstance(block_id, int) and not isinstance(block_id, bool)
                        and block_id >= 0 for block_id in witness_population)
                and all(isinstance(block_id, int) and not isinstance(block_id, bool)
                        and block_id >= 0 for block_id in witness_selected)
                and len(witness_population) == len(set(witness_population))
                and len(witness_selected) == len(set(witness_selected))
                and witness_population == sorted(witness_population)
                and witness_selected == sorted(witness_selected)
                and isinstance(witness_segment_id, int)
                and not isinstance(witness_segment_id, bool)
                and witness_segment_id >= 0
            )
            witness_demonstrates_partial = (
                witness_lists_valid
                and 0 < len(witness_selected) < len(witness_population)
                and set(witness_selected).issubset(witness_population)
            )
            if (trial.get("independent_draw_partially_retained_segment") is not True
                    or not witness_demonstrates_partial):
                fail(f"{slug}:independent_block_draw_preserved_whole_segments")
                cell_valid = False

            wave_estimate = trial.get("wave_segment_estimate")
            block_estimate = trial.get("independent_block_estimate")
            if (not reference_valid or not valid_estimate(wave_estimate)
                    or not valid_estimate(block_estimate)):
                fail(f"{slug}:invalid_trial_estimate:{trial_index}")
                cell_valid = False
                continue
            recomputed = {
                "wave_segment_count_error_rel": abs(
                    float(wave_estimate["sectors"]) - float(reference["sectors"]))
                    / float(reference["sectors"]),
                "independent_block_count_error_rel": abs(
                    float(block_estimate["sectors"]) - float(reference["sectors"]))
                    / float(reference["sectors"]),
                "wave_segment_rate_error_abs": abs(
                    float(wave_estimate["l2_hit_rate"])
                    - float(reference["rate"])),
                "independent_block_rate_error_abs": abs(
                    float(block_estimate["l2_hit_rate"])
                    - float(reference["rate"])),
            }
            for key, value in recomputed.items():
                supplied = trial.get(key)
                if (not finite_number(supplied) or float(supplied) < 0.0
                        or abs(float(supplied) - value) > ABLATION_COMPARE_EPS):
                    fail(f"{slug}:inconsistent_trial_error:{trial_index}:{key}")
                    cell_valid = False
                cell_errors[key].append(value)

        if len(trials) != len(expected_grid) or set(trial_grid) != expected_grid:
            fail(f"{slug}:incomplete_trial_grid")
            cell_valid = False
        if len(trial_grid) != len(set(trial_grid)):
            fail(f"{slug}:duplicate_trial_grid_point")
            cell_valid = False
        if any(len(cell_errors[key]) != len(expected_grid) for key in error_keys):
            cell_valid = False

        if not cell_valid:
            evidence_valid = False
            continue

        metrics = {
            "expectation": expectation,
            "n_trials": len(trials),
            "wave_segment_count_p95": _p95(
                cell_errors["wave_segment_count_error_rel"]),
            "independent_block_count_p95": _p95(
                cell_errors["independent_block_count_error_rel"]),
            "wave_segment_rate_p95": _p95(
                cell_errors["wave_segment_rate_error_abs"]),
            "independent_block_rate_p95": _p95(
                cell_errors["independent_block_rate_error_abs"]),
        }
        metrics.update({
            "wave_segment_count_error_le_independent": (
                metrics["wave_segment_count_p95"]
                <= metrics["independent_block_count_p95"] + ABLATION_COMPARE_EPS),
            "wave_segment_rate_error_le_independent": (
                metrics["wave_segment_rate_p95"]
                <= metrics["independent_block_rate_p95"] + ABLATION_COMPARE_EPS),
            "both_arms_accuracy_ok": (
                metrics["wave_segment_count_p95"] <= COUNT_TOL
                and metrics["independent_block_count_p95"] <= COUNT_TOL
                and metrics["wave_segment_rate_p95"] <= RATE_TOL
                and metrics["independent_block_rate_p95"] <= RATE_TOL),
        })
        cell_metrics[slug] = metrics
        for key in error_keys:
            pooled[key].extend(cell_errors[key])

    if not sensitive_slugs:
        fail("co_residency_sensitive_witness_set_empty")

    complete_fair_evidence_ok = evidence_valid and bool(sensitive_slugs)

    per_cell_absolute_accuracy_ok = (
        complete_fair_evidence_ok and bool(cell_metrics)
        and len(cell_metrics) == len(expected_set)
        and all(metrics["both_arms_accuracy_ok"]
                for metrics in cell_metrics.values())
    )
    if complete_fair_evidence_ok:
        for slug, metrics in cell_metrics.items():
            if metrics["wave_segment_count_p95"] > COUNT_TOL:
                fail(f"{slug}:wave_segment_count_accuracy_failed")
            if metrics["independent_block_count_p95"] > COUNT_TOL:
                fail(f"{slug}:independent_block_count_accuracy_failed")
            if metrics["wave_segment_rate_p95"] > RATE_TOL:
                fail(f"{slug}:wave_segment_rate_accuracy_failed")
            if metrics["independent_block_rate_p95"] > RATE_TOL:
                fail(f"{slug}:independent_block_rate_accuracy_failed")

    aggregate = {
        "wave_segment_count_p95": _p95(
            pooled["wave_segment_count_error_rel"]),
        "independent_block_count_p95": _p95(
            pooled["independent_block_count_error_rel"]),
        "wave_segment_rate_p95": _p95(
            pooled["wave_segment_rate_error_abs"]),
        "independent_block_rate_p95": _p95(
            pooled["independent_block_rate_error_abs"]),
    }
    aggregate_count_not_worse = (
        complete_fair_evidence_ok
        and aggregate["wave_segment_count_p95"]
        <= aggregate["independent_block_count_p95"] + ABLATION_COMPARE_EPS)
    aggregate_rate_not_worse = (
        complete_fair_evidence_ok
        and aggregate["wave_segment_rate_p95"]
        <= aggregate["independent_block_rate_p95"] + ABLATION_COMPARE_EPS)
    pooled_selection_unit_noninferiority_ok = (
        aggregate_count_not_worse and aggregate_rate_not_worse)
    if complete_fair_evidence_ok and not aggregate_count_not_worse:
        fail("pooled_wave_segment_count_regression")
    if complete_fair_evidence_ok and not aggregate_rate_not_worse:
        fail("pooled_wave_segment_rate_regression")

    sensitive_rate_not_worse = bool(sensitive_slugs)
    strict_advantage_slugs: list[str] = []
    if complete_fair_evidence_ok:
        for slug in sensitive_slugs:
            metrics = cell_metrics[slug]
            if not metrics["wave_segment_rate_error_le_independent"]:
                sensitive_rate_not_worse = False
                fail(f"{slug}:co_residency_sensitive_rate_regression")
            if (metrics["wave_segment_rate_p95"] + ABLATION_COMPARE_EPS
                    < metrics["independent_block_rate_p95"]):
                strict_advantage_slugs.append(slug)
    else:
        sensitive_rate_not_worse = False
    if sensitive_slugs and complete_fair_evidence_ok and not strict_advantage_slugs:
        fail("co_residency_sensitive_strict_advantage_missing")
    co_residency_sensitive_witness_ok = (
        complete_fair_evidence_ok and sensitive_rate_not_worse
        and bool(strict_advantage_slugs))

    universal_per_cell_noninferiority = (
        complete_fair_evidence_ok and bool(cell_metrics)
        and all(
            metrics["wave_segment_count_error_le_independent"]
            and metrics["wave_segment_rate_error_le_independent"]
            for metrics in cell_metrics.values())
    )
    composite_acceptance_ok = (
        complete_fair_evidence_ok
        and per_cell_absolute_accuracy_ok
        and pooled_selection_unit_noninferiority_ok
        and co_residency_sensitive_witness_ok
    )
    aggregate.update({
        "wave_segment_count_error_le_independent": aggregate_count_not_worse,
        "wave_segment_rate_error_le_independent": aggregate_rate_not_worse,
        "both_arms_accuracy_ok": per_cell_absolute_accuracy_ok,
    })
    return {
        "policy_version": ABLATION_POLICY_VERSION,
        "expected_expectations": dict(sorted(expected_policy.items())),
        "expected_cell_slugs": sorted(expected_set),
        "evaluated_cell_slugs": sorted(rows_by_slug),
        "sensitive_cell_slugs": sorted(sensitive_slugs),
        "strict_advantage_cell_slugs": sorted(strict_advantage_slugs),
        "complete_fair_evidence_ok": complete_fair_evidence_ok,
        "per_cell_absolute_accuracy_ok": per_cell_absolute_accuracy_ok,
        "pooled_selection_unit_noninferiority_ok": (
            pooled_selection_unit_noninferiority_ok),
        "co_residency_sensitive_witness_ok": (
            co_residency_sensitive_witness_ok),
        "universal_per_cell_noninferiority": (
            universal_per_cell_noninferiority),
        "composite_acceptance_ok": composite_acceptance_ok,
        "per_cell_metrics": cell_metrics,
        "aggregate": aggregate,
        "typed_failure_reasons": failures,
    }


def _selection_ablation(trace_dir: Path, layouts: list[dict], population, stats,
                        r_full_sectors: float, r_full_hits: float,
                        scratch: Path, wave_trials: list[dict],
                        block_ids_by_path: dict[Path, int]) -> dict:
    """Production ghost-differenced estimator versus an independent-block baseline.

    The wave arm reuses the exact estimator trials from the fidelity run.  The
    baseline independently selects individual real block payloads at the same
    block budget and replays that selected subtrace through the same decoder/cache
    path. Sector counts use ordinary-block inverse-probability weighting plus
    exact-record calibration. Hits use the analogous certainty-adjusted
    difference contrast; it is deliberately not called an HT total because cache
    warming can change certainty-block hits in the combined replay. Their ratio
    exposes the cache-order information lost when concurrency segments are
    thinned into individual blocks.
    """
    certainty_ids = {segment.id for segment in population.certainty_segments}
    paths_by_segment = {
        layout["id"]: layout["ghost_paths"] + layout["measured_paths"]
        for layout in layouts
    }
    certainty_paths = [
        path for segment_id in sorted(certainty_ids)
        for path in paths_by_segment[segment_id]
    ]
    ordinary_layouts = [
        layout for layout in layouts if layout["id"] not in certainty_ids
    ]
    ordinary_paths = [
        path for layout in ordinary_layouts for path in paths_by_segment[layout["id"]]
    ]
    records_by_block = {block.block_id: block.records for block in stats.blocks}

    def payload(paths: list[Path]) -> dict[str, int | float]:
        records = sum(records_by_block[block_ids_by_path[path]] for path in paths)
        return {
            "blocks": len(paths),
            "records": records,
            "bytes": sum(path.stat().st_size for path in paths),
            "realized_block_fraction": len(paths) / stats.n_blocks,
        }

    certainty_sectors, certainty_hits = _decode_subset(
        trace_dir, certainty_paths, scratch)
    certainty_payload = payload(certainty_paths)
    full_rate = r_full_hits / r_full_sectors if r_full_sectors else 0.0
    wave_count_errors: list[float] = []
    wave_rate_errors: list[float] = []
    block_count_errors: list[float] = []
    block_rate_errors: list[float] = []
    trials: list[dict] = []
    wave_by_trial = {
        (trial["requested_fraction"], trial["seed"]): trial
        for trial in wave_trials
    }
    for fraction_index, fraction in enumerate(FRACTIONS):
        for seed in range(N_SEEDS):
            wave_trial = wave_by_trial[(fraction, seed)]
            selected_ids = wave_trial["selected_segment_ids"]
            ordinary_selected_ids = [
                segment_id for segment_id in selected_ids
                if segment_id not in certainty_ids
            ]
            selected_wave_ordinary = [
                path for segment_id in ordinary_selected_ids
                for path in paths_by_segment[segment_id]
            ]
            wave_paths = certainty_paths + selected_wave_ordinary

            rng = np.random.default_rng(seed + 90001 + 1000 * fraction_index)
            drawn = rng.choice(
                len(ordinary_paths), size=len(selected_wave_ordinary), replace=False)
            selected_block_ordinary = [ordinary_paths[int(i)] for i in drawn]
            block_paths = certainty_paths + selected_block_ordinary
            if len(block_paths) != len(wave_paths):
                raise GoldenSetError("ablation_block_budget_mismatch")

            selected_block_set = set(selected_block_ordinary)
            partial_layout = next((
                layout for layout in ordinary_layouts
                if 0 < len(selected_block_set.intersection(
                    paths_by_segment[layout["id"]]))
                < len(paths_by_segment[layout["id"]])
            ), None)
            if partial_layout is None:
                raise GoldenSetError(
                    "independent_block_draw_preserved_whole_segments: "
                    f"fraction={fraction}, seed={seed}")
            partial_segment_paths = paths_by_segment[partial_layout["id"]]
            partial_segment_witness = {
                "segment_id": partial_layout["id"],
                "segment_block_ids": sorted(
                    block_ids_by_path[path] for path in partial_segment_paths),
                "selected_block_ids": sorted(
                    block_ids_by_path[path]
                    for path in selected_block_set.intersection(
                        partial_segment_paths)),
            }

            block_sectors, block_hits = _decode_subset(trace_dir, block_paths, scratch)
            wave_payload = payload(wave_paths)
            block_payload = payload(block_paths)
            block_ordinary_payload = payload(selected_block_ordinary)
            block_weight = len(ordinary_paths) / len(selected_block_ordinary)
            block_ht_sectors = (
                certainty_sectors
                + block_weight * (block_sectors - certainty_sectors))
            block_adjusted_hits = (
                certainty_hits
                + block_weight * (block_hits - certainty_hits))
            block_x_hat = (
                certainty_payload["records"]
                + block_weight * block_ordinary_payload["records"])
            block_calibration = stats.total_records / block_x_hat
            block_estimated_sectors = block_ht_sectors * block_calibration
            block_estimated_hits = block_adjusted_hits * block_calibration
            block_rate = (block_estimated_hits / block_estimated_sectors
                          if block_estimated_sectors else 0.0)
            wave_count_error = wave_trial["count_error_rel"]
            block_count_error = (
                abs(block_estimated_sectors - r_full_sectors) / r_full_sectors
                if r_full_sectors else 0.0)
            wave_rate_error = wave_trial["rate_error_abs"]
            block_rate_error = abs(block_rate - full_rate)
            wave_count_errors.append(wave_count_error)
            block_count_errors.append(block_count_error)
            wave_rate_errors.append(wave_rate_error)
            block_rate_errors.append(block_rate_error)
            trials.append({
                "requested_fraction": fraction,
                "seed": seed,
                "selected_segments": len(selected_ids),
                "selected_ordinary_segments": len(ordinary_selected_ids),
                "independent_block_inverse_probability_weight": block_weight,
                "wave_segment_payload": wave_payload,
                "independent_block_payload": block_payload,
                "wave_segment_estimate": wave_trial["point"],
                "independent_block_estimate": {
                    "sectors": block_estimated_sectors,
                    "hits": block_estimated_hits,
                    "l2_hit_rate": block_rate,
                },
                "wave_segment_count_error_rel": wave_count_error,
                "independent_block_count_error_rel": block_count_error,
                "wave_segment_rate_error_abs": wave_rate_error,
                "independent_block_rate_error_abs": block_rate_error,
                "independent_draw_partially_retained_segment": True,
                "independent_partial_segment_witness": partial_segment_witness,
            })

    wave_count_p95 = float(np.percentile(wave_count_errors, 95))
    block_count_p95 = float(np.percentile(block_count_errors, 95))
    wave_rate_p95 = float(np.percentile(wave_rate_errors, 95))
    block_rate_p95 = float(np.percentile(block_rate_errors, 95))
    count_not_worse = wave_count_p95 <= block_count_p95 + ABLATION_COMPARE_EPS
    rate_not_worse = wave_rate_p95 <= block_rate_p95 + ABLATION_COMPARE_EPS
    accuracy_failures = []
    if wave_count_p95 > COUNT_TOL:
        accuracy_failures.append("wave_segment_count_accuracy_failed")
    if block_count_p95 > COUNT_TOL:
        accuracy_failures.append("independent_block_count_accuracy_failed")
    if wave_rate_p95 > RATE_TOL:
        accuracy_failures.append("wave_segment_rate_accuracy_failed")
    if block_rate_p95 > RATE_TOL:
        accuracy_failures.append("independent_block_rate_accuracy_failed")
    comparison_diagnostics = []
    if not count_not_worse:
        comparison_diagnostics.append("wave_segment_count_regression")
    if not rate_not_worse:
        comparison_diagnostics.append("wave_segment_rate_regression")
    summary = {
        "protocol": (
            "production_ghost_differenced_estimator_vs_"
            "independent_block_cold_subtrace"),
        "interpretation": "whole_segment_vs_independent_block_selection_unit",
        "causal_co_residency_claim": False,
        "co_residency_thinning_proxy": (
            "The flat L2 replay detects selection-unit/cache-order bias; it does "
            "not claim a causal concurrency mechanism."),
        "certainty_segment_ids": sorted(certainty_ids),
        "ordinary_draw": "SRSWOR individual .pb payloads; fixed RNG seed offset",
        "count_adjustment": (
            "wave arm uses the production calibrated HT estimator; independent "
            "sector count uses certainty + ordinary-block HT then exact-record "
            "calibration"),
        "rate_adjustment": (
            "certainty-adjusted inverse-probability hit/sector difference ratio; "
            "the hit contrast is not claimed additive under cache interference"),
        "n_trials": len(trials),
        "wave_segment_count_p95": wave_count_p95,
        "independent_block_count_p95": block_count_p95,
        "wave_segment_rate_p95": wave_rate_p95,
        "independent_block_rate_p95": block_rate_p95,
        "wave_segment_count_error_le_independent": count_not_worse,
        "wave_segment_rate_error_le_independent": rate_not_worse,
        "universal_per_cell_noninferiority": count_not_worse and rate_not_worse,
        "both_arms_accuracy_ok": not accuracy_failures,
        "typed_failure_reasons": accuracy_failures,
        "typed_diagnostic_reasons": comparison_diagnostics,
        "trials": trials,
    }
    return summary


def _score_estimator(seg_meas: list[dict], population,
                     r_full_sectors: float, r_full_hits: float,
                     ghost_census_rate: float,
                     x_true: float) -> tuple[dict, list[dict]]:
    """Full-vs-sampled scoring plus an auditable per-trial estimator ledger."""
    n_total = len(population.segments)
    if len(seg_meas) != n_total:
        raise GoldenSetError(
            f"segment_measurement_population_mismatch: {len(seg_meas)} != {n_total}")
    measurements = {segment["id"]: segment for segment in seg_meas}
    certainty_ids = [segment.id for segment in population.certainty_segments]
    r_full_rate = r_full_hits / r_full_sectors if r_full_sectors else 0.0
    count_rels: list[float] = []
    rate_errs: list[float] = []
    count_cov: list[bool] = []
    rate_cov: list[bool] = []
    raw_ghost_rate_cov: list[bool] = []
    rate_hw_ok: list[bool] = []
    count_hw_ok: list[bool] = []
    calib_ok: list[bool] = []
    status_ok: list[bool] = []
    trials: list[dict] = []
    n_seg_ok = True
    for frac in FRACTIONS:
        ids_for_floor = _sampled_segment_ids(population, frac, 0)
        if len(ids_for_floor) < MIN_SEGMENTS:
            n_seg_ok = False
        for seed in range(N_SEEDS):
            plan = _selection_plan(population, frac, seed)
            ids = [segment.id for segment in plan.selected_segments]
            sampled_ids = [i for i in ids if i not in certainty_ids]
            stratum_by_segment = {
                segment.id: selected_stratum.id
                for selected_stratum in plan.strata
                for segment in selected_stratum.population.segments
            }
            segments = {i: Segment(
                                   id=i,
                                   stratum=(-1 if i in certainty_ids
                                            else stratum_by_segment[i]),
                                   measured={"sectors": measurements[i]["sectors"],
                                             "hits": measurements[i]["hits"]},
                                   record_count=measurements[i]["records"]) for i in ids}
            strata = [Stratum(
                id=selected_stratum.id,
                N_h=selected_stratum.N_h,
                sampled_ids=[segment.id
                             for segment in selected_stratum.sampled_segments],
            ) for selected_stratum in plan.strata]
            r = estimate(segments, strata, certainty_ids=certainty_ids, x_true=x_true,
                         count_targets=["sectors", "hits"],
                         rate_targets=[RateTarget("l2_hit_rate", "hits", "sectors")],
                         seed=seed)
            count_error = (abs(r.point["sectors"] - r_full_sectors)
                           / r_full_sectors if r_full_sectors else 0.0)
            rate_error = abs(r.point["l2_hit_rate"] - r_full_rate)
            count_rels.append(count_error)
            rate_errs.append(rate_error)
            clo, chi = r.ci["sectors"]
            count_eps = 1e-12 * max(1.0, abs(r_full_sectors))
            count_covers = clo - count_eps <= r_full_sectors <= chi + count_eps
            count_hw_rel = ((chi - clo) / 2 / r_full_sectors
                            if r_full_sectors else float("inf"))
            count_cov.append(count_covers)
            count_hw_ok.append(count_hw_rel <= COUNT_CI_HALFWIDTH_REL)
            rlo, rhi = r.ci["l2_hit_rate"]
            raw_ghost_covers = rlo - 1e-12 <= ghost_census_rate <= rhi + 1e-12
            full_shape_rlo, full_shape_rhi, rate_hw = _full_shape_rate_interval(
                rlo, rhi)
            rate_covers = (
                full_shape_rlo - 1e-12 <= r_full_rate <= full_shape_rhi + 1e-12)
            raw_rate_hw = (rhi - rlo) / 2
            rate_cov.append(rate_covers)
            raw_ghost_rate_cov.append(raw_ghost_covers)
            rate_hw_ok.append(rate_hw <= RATE_CI_HALFWIDTH)
            calib_ok.append(r.calibration_ok)
            status_ok.append(r.status == "converged")
            trials.append({
                "requested_fraction": frac,
                "realized_fraction": len(ids) / n_total,
                "seed": seed,
                "selected_segment_ids": ids,
                "certainty_segment_ids": certainty_ids,
                "ordinary_population_segments": n_total - len(certainty_ids),
                "sampled_ordinary_segments": len(sampled_ids),
                "strata": plan.sidecar_strata(),
                "point": {"sectors": r.point["sectors"],
                          "hits": r.point["hits"],
                          "l2_hit_rate": r.point["l2_hit_rate"]},
                "ci": {"sectors": [clo, chi],
                       "raw_ghost_l2_hit_rate": [rlo, rhi],
                       "full_shape_l2_hit_rate": [full_shape_rlo, full_shape_rhi]},
                "count_error_rel": count_error,
                "rate_error_abs": rate_error,
                "count_ci_covers_production_full": count_covers,
                "rate_ci_covers_production_full": rate_covers,
                "raw_rate_ci_covers_ghost_census": raw_ghost_covers,
                "count_ci_halfwidth_rel": count_hw_rel,
                "raw_rate_ci_halfwidth_abs": raw_rate_hw,
                "full_shape_rate_ci_halfwidth_bound_abs": rate_hw,
                "status": r.status,
                "calibration_ok": r.calibration_ok,
                "x_true": r.x_true,
                "x_hat": r.x_hat,
            })
    p95_count = float(np.percentile(count_rels, 95))
    p95_rate = float(np.percentile(rate_errs, 95))
    count_coverage = float(np.mean(count_cov))
    rate_coverage = float(np.mean(rate_cov))
    raw_ghost_rate_coverage = float(np.mean(raw_ghost_rate_cov))
    sector_record_ratios = [
        segment["sectors"] / segment["records"]
        for segment in seg_meas if segment["records"] > 0
    ]
    ratio_min = min(sector_record_ratios)
    ratio_max = max(sector_record_ratios)
    score = {
        "production_full_rate": r_full_rate, "n_segments": n_total,
        "p95_count_rel": p95_count, "p95_rate_abs": p95_rate,
        "count_ci_coverage": count_coverage,
        "rate_ci_coverage": rate_coverage,
        "raw_ghost_rate_ci_coverage": raw_ghost_rate_coverage,
        "count_accuracy_ok": p95_count <= COUNT_TOL,
        "rate_accuracy_ok": p95_rate <= RATE_TOL,
        "count_ci_coverage_ok": count_coverage >= CI_COVERAGE_MIN,
        "rate_ci_coverage_ok": rate_coverage >= CI_COVERAGE_MIN,
        "raw_ghost_rate_ci_coverage_ok": raw_ghost_rate_coverage >= CI_COVERAGE_MIN,
        "ci_coverage_ok": (
            count_coverage >= CI_COVERAGE_MIN
            and rate_coverage >= CI_COVERAGE_MIN),
        "rate_ci_halfwidth_ok": all(rate_hw_ok),
        "count_ci_halfwidth_ok": all(count_hw_ok),
        "min_segments_ok": n_seg_ok and n_total >= MIN_SEGMENTS,
        "calibration_ok": all(calib_ok),
        "estimator_status_ok": all(status_ok),
        "sector_per_record_min": ratio_min,
        "sector_per_record_max": ratio_max,
        "count_target_heterogeneous": not np.isclose(
            ratio_min, ratio_max, rtol=1e-12, atol=1e-12),
        "trials": trials,
    }
    return score, trials


def run_golden_set(manifest_path: Path, *, scratch: Path | None = None) -> dict:
    """Real golden-set run over the manifest's cells. Raises GoldenSetError on any
    missing real input."""
    import sector_cache_replay as scr  # type: ignore

    manifest_bytes = Path(manifest_path).read_bytes()
    manifest_sha256, manifest_policy_ok = _manifest_policy_status(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    scratch = scratch or Path(tempfile.mkdtemp(prefix="gs_scratch_"))
    scratch.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    if not manifest_policy_ok:
        failures.append(
            "manifest::policy_sha256_mismatch "
            f"(expected={GOLDEN_MANIFEST_SHA256}, actual={manifest_sha256})")
    positives, negatives, ablations = [], [], []
    fidelity_trials: list[dict] = []
    model_abs_errors: list[float] = []
    sampling_induced_changes_all: list[float] = []
    for cell in manifest["cells"]:
        slug = cell["slug"]
        td = _REPO / cell["trace_dir"]
        if not (td / "dynamic_trace.pb").is_file():
            raise GoldenSetError(f"{slug}: trace dir missing {td}")
        role = cell.get("role")
        if role not in {"positive", "negative"}:
            raise GoldenSetError(f"{slug}: invalid golden-cell role {role!r}")
        ncu = ncu_csv_rel = ncu_kernel_name = None
        if role == "positive":
            ncu = cell.get("ncu_l2_hit_pct")
            if ncu is None:
                raise GoldenSetError(
                    f"{slug}: positive cell missing ncu_l2_hit_pct "
                    "(the error-budget denominator source)")
            ncu_csv_rel = cell.get("ncu_csv")
            if not ncu_csv_rel:
                raise GoldenSetError(
                    f"{slug}: positive cell missing ncu_csv provenance")
            ncu_kernel_name = cell.get("ncu_kernel_name")
            if not ncu_kernel_name:
                raise GoldenSetError(
                    f"{slug}: positive cell missing ncu_kernel_name provenance")
        stats = analyze_trace_dir(td, slug)
        gate = classify(stats)
        if role == "negative":
            forbidden = gate.method != M_SUBSET
            negatives.append({"slug": slug, "verdict": gate.verdict,
                              "forbidding_gates": gate.forbidding_gates,
                              "forbidden": forbidden})
            if not forbidden:
                failures.append(f"NEGATIVE {slug} admitted (verdict={gate.verdict})")
            continue
        # positive: must be subset-admissible on this host.
        if gate.method != M_SUBSET:
            failures.append(f"POSITIVE {slug} not subset-admissible (verdict={gate.verdict})")
            positives.append({"slug": slug, "admissible": False, "verdict": gate.verdict})
            continue
        population = build_segment_population(
            stats, force_include_block_ids=gate.force_include_blocks)
        fraction_design = _fraction_design(population)
        launch_dir, _kernel_id = first_launch_dir(td)
        provenance = build_golden_provenance(
            ncu_csv_path=_REPO / ncu_csv_rel,
            trace_dir=td,
            expected_grid=stats.grid,
            tracer_version=TRACER_VERSION,
            stats_version=STATS_VERSION,
            estimator_version=ESTIMATOR_VERSION,
            expected_kernel_name=ncu_kernel_name,
            first_launch_dir=launch_dir,
        )
        csv_ncu = float(provenance["ncu"]["value_pct"])
        if csv_ncu != float(ncu):
            raise GoldenSetError(
                f"{slug}: manifest ncu_l2_hit_pct={ncu} does not match "
                f"persisted CSV value {csv_ncu}")
        full = scr.replay_cell(td, _cache_cfg(), classify_misses=False)
        r_full_sectors = float(full["total_accesses"])
        r_full_hits = float(full["hits"])
        (depth1_meas, layouts, ghost_oracle, ghost_sensitivity,
         block_ids_by_path) = _segment_measurements(td, stats, scratch)
        full_shape_count_scale = stats.total_records / ghost_oracle["records"]
        full_shape_seg_meas = [
            {
                **segment,
                "sectors": segment["sectors"] * full_shape_count_scale,
                "hits": segment["hits"] * full_shape_count_scale,
            }
            for segment in depth1_meas
        ]
        ghost_census_rate = (ghost_oracle["hits"] / ghost_oracle["sectors"]
                             if ghost_oracle["sectors"] else 0.0)
        score, cell_trials = _score_estimator(
            full_shape_seg_meas, population, r_full_sectors, r_full_hits,
            ghost_census_rate, ghost_oracle["records"])
        ghost_depth_movement_max = float(ghost_sensitivity["movement_1_to_2"])
        ghost_depth_sensitivity_ok = (
            ghost_depth_movement_max <= GHOST_DEPTH_RATE_MOVEMENT_TOL)

        # The flat cache replay cannot demonstrate a causal placement mechanism.
        # It does provide the required selection-unit/thinning comparison against
        # the exact production ghost-differenced estimator trials.
        expectation = cell.get("ablation_expectation")
        if expectation not in {"neutral", "co_residency_sensitive"}:
            raise GoldenSetError(
                f"{slug}: invalid or missing ablation_expectation {expectation!r}")
        ablation = _selection_ablation(
            td, layouts, population, stats, r_full_sectors, r_full_hits,
            scratch, cell_trials, block_ids_by_path)
        effect_gate_ok = (
            expectation != "co_residency_sensitive"
            or ablation["wave_segment_rate_error_le_independent"])
        tail = layouts[-1]
        ablation.update({
            "slug": slug,
            "expectation": expectation,
            "co_residency_effect_gate_ok": effect_gate_ok,
            "fraction_design": fraction_design,
            "common_production_reference": {
                "sectors": r_full_sectors,
                "hits": r_full_hits,
                "rate": r_full_hits / r_full_sectors if r_full_sectors else 0.0,
            },
            "certainty_tail": {
                "id": tail["id"],
                "kind": "probability_one_residual_wave_tail",
                "waves": tail["waves"],
                "ghost_blocks": len(tail["ghost_paths"]),
                "measured_blocks": len(tail["measured_paths"]),
                "total_blocks": len(tail["ghost_paths"]) + len(tail["measured_paths"]),
                "ghost_records": tail["ghost_records"],
                "measured_records": tail["measured_records"],
                "depth2_ghost_blocks": sum(
                    len(tail["paths_by_wave"][wave]) for wave in tail["waves"][:2]),
                "depth2_measured_blocks": sum(
                    len(tail["paths_by_wave"][wave]) for wave in tail["waves"][2:]),
            },
        })
        ablations.append(ablation)
        # sampling-induced error budget vs the REAL NCU label.
        ncu_rate = float(ncu) / 100.0
        model_full_rate = float(full["predicted_hit_ratio"])
        model_abs_error = abs(model_full_rate - ncu_rate)
        sampling_induced_changes = [
            abs(abs(trial["point"]["l2_hit_rate"] - ncu_rate) - model_abs_error)
            for trial in cell_trials
        ]
        sampling_err = float(np.percentile(sampling_induced_changes, 95))
        budget = ERROR_BUDGET_FACTOR * model_abs_error
        fidelity_trials.extend({"slug": slug, **trial} for trial in cell_trials)
        model_abs_errors.append(model_abs_error)
        sampling_induced_changes_all.extend(sampling_induced_changes)
        score.update({"slug": slug, "admissible": True,
                      "measurement_protocol": "per_segment_ghost_differencing",
                      "ghost_census_sectors": ghost_oracle["sectors"],
                      "ghost_census_hits": ghost_oracle["hits"],
                      "ghost_census_measured_records": ghost_oracle["records"],
                      "ghost_census_rate": ghost_census_rate,
                      "ghost_depth_sensitivity": ghost_sensitivity,
                      "ghost_depth_rate_movement_max_abs":
                          ghost_depth_movement_max,
                      "ghost_depth_rate_movement_tol":
                          GHOST_DEPTH_RATE_MOVEMENT_TOL,
                      "ghost_depth_sensitivity_ok": ghost_depth_sensitivity_ok,
                      "full_shape_count_scale": full_shape_count_scale,
                      "production_full_sectors": r_full_sectors,
                      "production_full_hits": r_full_hits,
                      "ncu_l2_hit": ncu_rate,
                      "model_full_l2_hit": model_full_rate,
                      "model_abs_error": model_abs_error,
                      "per_cell_error_budget": budget,
                      "sampling_induced_error_p95": sampling_err,
                      "error_budget_ok": sampling_err <= budget,
                      "provenance": provenance})
        positives.append(score)
        for gate_key in ("count_accuracy_ok", "rate_accuracy_ok",
                         "count_ci_coverage_ok", "rate_ci_coverage_ok",
                         "rate_ci_halfwidth_ok", "count_ci_halfwidth_ok",
                         "min_segments_ok", "calibration_ok", "error_budget_ok",
                         "ghost_depth_sensitivity_ok", "estimator_status_ok"):
            if not score[gate_key]:
                failures.append(f"{slug}: {gate_key} failed")

    manifest_size_ok = 8 <= len(manifest.get("cells", [])) <= 12
    if not manifest_size_ok:
        failures.append(
            f"golden_cell_count_out_of_range: {len(manifest.get('cells', []))} not in [8,12]")

    # The design-level acceptance is pooled over cells x fractions x seeds, in
    # addition to the stricter per-cell gates above.
    fidelity_count_errors = [trial["count_error_rel"] for trial in fidelity_trials]
    fidelity_rate_errors = [trial["rate_error_abs"] for trial in fidelity_trials]
    count_coverage = float(np.mean([
        trial["count_ci_covers_production_full"] for trial in fidelity_trials
    ])) if fidelity_trials else 0.0
    rate_coverage = float(np.mean([
        trial["rate_ci_covers_production_full"] for trial in fidelity_trials
    ])) if fidelity_trials else 0.0
    raw_ghost_rate_coverage = float(np.mean([
        trial["raw_rate_ci_covers_ghost_census"] for trial in fidelity_trials
    ])) if fidelity_trials else 0.0
    fidelity_aggregate = {
        "n_trials": len(fidelity_trials),
        "p95_count_rel": _p95(fidelity_count_errors),
        "p95_rate_abs": _p95(fidelity_rate_errors),
        "count_ci_coverage": count_coverage,
        "rate_ci_coverage": rate_coverage,
        "raw_ghost_rate_ci_coverage": raw_ghost_rate_coverage,
        "count_accuracy_ok": bool(fidelity_trials) and (
            _p95(fidelity_count_errors) <= COUNT_TOL),
        "rate_accuracy_ok": bool(fidelity_trials) and (
            _p95(fidelity_rate_errors) <= RATE_TOL),
        "count_ci_coverage_ok": count_coverage >= CI_COVERAGE_MIN,
        "rate_ci_coverage_ok": rate_coverage >= CI_COVERAGE_MIN,
        "raw_ghost_rate_ci_coverage_ok": raw_ghost_rate_coverage >= CI_COVERAGE_MIN,
    }
    for key in ("count_accuracy_ok", "rate_accuracy_ok",
                "count_ci_coverage_ok", "rate_ci_coverage_ok",
                "raw_ghost_rate_ci_coverage_ok"):
        if not fidelity_aggregate[key]:
            failures.append(f"fidelity_aggregate::{key}")

    model_mae = float(np.mean(model_abs_errors)) if model_abs_errors else None
    sampling_change_p95 = _p95(sampling_induced_changes_all)
    aggregate_budget = (ERROR_BUDGET_FACTOR * model_mae
                        if model_mae is not None else None)
    aggregate_error_budget_ok = (
        sampling_change_p95 is not None and aggregate_budget is not None
        and sampling_change_p95 <= aggregate_budget)
    error_budget_aggregate = {
        "model_mae": model_mae,
        "sampling_induced_sim_vs_ncu_error_change_p95": sampling_change_p95,
        "allowance": aggregate_budget,
        "pass": aggregate_error_budget_ok,
    }
    if not aggregate_error_budget_ok:
        failures.append("error_budget_aggregate::exceeded")

    segment_heterogeneous_cells = [
        p["slug"] for p in positives if p.get("count_target_heterogeneous")
    ]
    independent_block_variation_cells = [
        ablation["slug"] for ablation in ablations
        if ablation["independent_block_count_p95"] > ABLATION_COMPARE_EPS
    ]
    count_validation_nonvacuous = bool(independent_block_variation_cells)
    count_validation = {
        "segment_heterogeneous_cells": segment_heterogeneous_cells,
        "independent_block_variation_cells": independent_block_variation_cells,
        "basis": (
            "Individual block payloads vary in sector/record contribution while "
            "complete wave segments preserve their balanced cluster totals."),
        "pass": count_validation_nonvacuous,
    }
    if not count_validation_nonvacuous:
        failures.append("count_validation::vacuous_homogeneous_targets")

    # The comparative verdict is rebuilt from the paired trial ledgers.  The
    # cell summaries above remain report diagnostics and are not trusted gates.
    admissible_slugs = [
        p["slug"] for p in positives if p.get("admissible")]
    ablation_evaluation = _evaluate_ablation_composite(
        ablations, GOLDEN_ABLATION_EXPECTATIONS)
    for reason in ablation_evaluation["typed_failure_reasons"]:
        failures.append(f"ablation::{reason}")
    for ablation in ablations:
        slug = ablation["slug"]
        metrics = ablation_evaluation["per_cell_metrics"].get(slug)
        if metrics is not None:
            ablation.update({
                key: metrics[key] for key in (
                    "n_trials",
                    "wave_segment_count_p95",
                    "independent_block_count_p95",
                    "wave_segment_rate_p95",
                    "independent_block_rate_p95",
                    "wave_segment_count_error_le_independent",
                    "wave_segment_rate_error_le_independent",
                    "both_arms_accuracy_ok",
                )
            })
            ablation["universal_per_cell_noninferiority"] = (
                metrics["wave_segment_count_error_le_independent"]
                and metrics["wave_segment_rate_error_le_independent"])
            ablation["per_cell_absolute_accuracy_ok"] = (
                metrics["both_arms_accuracy_ok"])
            ablation["co_residency_effect_gate_ok"] = (
                ablation["expectation"] != "co_residency_sensitive"
                or metrics["wave_segment_rate_error_le_independent"])
        prefix = f"{slug}:"
        ablation["typed_failure_reasons"] = [
            reason[len(prefix):]
            for reason in ablation_evaluation["typed_failure_reasons"]
            if reason.startswith(prefix)
        ]

    ablation_slugs = [a["slug"] for a in ablations]
    ablation_present = (
        bool(admissible_slugs)
        and len(ablation_slugs) == len(admissible_slugs)
        and set(ablation_slugs) == set(admissible_slugs))
    per_cell_not_worse = ablation_evaluation[
        "universal_per_cell_noninferiority"]
    co_residency_effect_ok = ablation_evaluation[
        "co_residency_sensitive_witness_ok"]
    wave_segment_not_worse = ablation_evaluation["composite_acceptance_ok"]

    report = {
        "schema_version": 2,
        "gates": {"count_tol": COUNT_TOL, "rate_tol": RATE_TOL,
                  "ci_coverage_min": CI_COVERAGE_MIN, "rate_ci_halfwidth": RATE_CI_HALFWIDTH,
                  "count_ci_halfwidth_rel": COUNT_CI_HALFWIDTH_REL,
                  "ghost_depth_rate_movement_tol": GHOST_DEPTH_RATE_MOVEMENT_TOL,
                  "full_shape_rate_ci_warm_state_allowance":
                      GHOST_DEPTH_RATE_MOVEMENT_TOL,
                  "error_budget_factor": ERROR_BUDGET_FACTOR,
                  "ablation_comparison_epsilon": ABLATION_COMPARE_EPS,
                  "fractions": list(FRACTIONS), "n_seeds": N_SEEDS},
        "versions": {"tracer": TRACER_VERSION, "stats": STATS_VERSION,
                     "estimator": ESTIMATOR_VERSION,
                     "ablation_policy": ABLATION_POLICY_VERSION,
                     "golden_manifest_policy": GOLDEN_MANIFEST_POLICY_VERSION},
        "manifest_sha256": manifest_sha256,
        "manifest_policy_sha256": GOLDEN_MANIFEST_SHA256,
        "manifest_policy_ok": manifest_policy_ok,
        "manifest_cell_count_ok": manifest_size_ok,
        "positive_cells": positives,
        "negative_controls": negatives,
        "fidelity_aggregate": fidelity_aggregate,
        "error_budget_aggregate": error_budget_aggregate,
        "count_validation": count_validation,
        "ablation": {
            "ablation_present": ablation_present,
            "interpretation": "whole_segment_vs_independent_block_selection_unit",
            "causal_co_residency_claim": False,
            "wave_segment_not_worse": wave_segment_not_worse,
            "policy_version": ablation_evaluation["policy_version"],
            "gate_semantics": (
                "Both arms must meet per-cell count/rate accuracy; pooled "
                "wave-segment count/rate p95 must be non-inferior; every "
                "predeclared sensitive witness must preserve rate non-inferiority "
                "and at least one must have a strict rate advantage. Neutral "
                "per-cell ordering is diagnostic."),
            "complete_fair_evidence_ok": ablation_evaluation[
                "complete_fair_evidence_ok"],
            "per_cell_absolute_accuracy_ok": ablation_evaluation[
                "per_cell_absolute_accuracy_ok"],
            "pooled_selection_unit_noninferiority_ok": ablation_evaluation[
                "pooled_selection_unit_noninferiority_ok"],
            "co_residency_sensitive_witness_ok": ablation_evaluation[
                "co_residency_sensitive_witness_ok"],
            "universal_per_cell_noninferiority": ablation_evaluation[
                "universal_per_cell_noninferiority"],
            "composite_acceptance_ok": ablation_evaluation[
                "composite_acceptance_ok"],
            "sensitive_cell_slugs": ablation_evaluation[
                "sensitive_cell_slugs"],
            "strict_advantage_cell_slugs": ablation_evaluation[
                "strict_advantage_cell_slugs"],
            "expected_expectations": ablation_evaluation[
                "expected_expectations"],
            "typed_failure_reasons": ablation_evaluation[
                "typed_failure_reasons"],
            "co_residency_effect_ok": co_residency_effect_ok,
            "per_cell_not_worse": per_cell_not_worse,
            "per_cell": ablations,
            "aggregate": ablation_evaluation["aggregate"],
        },
        "typed_out_of_scope_gaps": {
            "general_evaluation_reporter": (
                "the general evaluation join and per-metric reporter remain "
                "outside this narrow golden-cell NCU validation harness"),
        },
        "failures": failures,
        "pass": not failures,
    }
    return report


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = run_golden_set(Path(args.manifest))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    verdict = "PASS" if report["pass"] else "FAIL"
    print(f"real golden-set [{verdict}]: {len(report['positive_cells'])} positive, "
          f"{len(report['negative_controls'])} negative, {len(report['failures'])} failures")
    for f in report["failures"]:
        print(f"  FAIL: {f}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
