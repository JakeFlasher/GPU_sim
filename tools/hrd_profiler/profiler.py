"""Conditional HRD Profiler (task11 / the HRD-profiler clause).

Implements the R16 task10 / the HRD-profiler clause analysis artifact at
``.humanize/rlcr/2026-05-04_23-45-25/task10-ac6-analysis.md``.

The HRD profiler computes a per-bin reuse-distance distribution
under a four-layer hierarchy (sector / line / allocation / region)
with conditional descent (consult a coarser layer only when the
finer layer reuse distance is unavailable because it is cold or
infinity). Per-bin hit-rate confidence intervals are computed by
percentile bootstrap at the kernel-launch resampling unit. Bins
with insufficient sample count or excess CI half-width are
annotated `cache_model_underconfigured` and trigger the
bundle-level refusal annotation; downstream consumers that
request a single hit-rate scalar receive a typed refusal, not a
default.

The output bundle admits via
``workloads/simbench_bw/schemas/hrd_profile.schema.json`` under
strict ``jsonschema.Draft202012Validator`` validation (the R7
exact-artifact discipline).

Pinned thresholds (from R16 task10 artifact):
- ``BOOTSTRAP_RESAMPLES = 2000``
- ``CI_CONFIDENCE = 0.95``
- ``CI_HALF_WIDTH_TARGET = 0.05``
- ``MINIMUM_BIN_SAMPLE_COUNT = 50``
- ``RD_INFINITY_SENTINEL = 9223372036854775807``
- ``REUSE_DISTANCE_BIN_BOUNDS`` (34 buckets in 32B-sector units).
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Pinned constants from R16 task10 artifact.
# ---------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES: int = 2000
CI_CONFIDENCE: float = 0.95
CI_HALF_WIDTH_TARGET: float = 0.05
MINIMUM_BIN_SAMPLE_COUNT: int = 50
RD_INFINITY_SENTINEL: int = 9223372036854775807


# 34-bucket reuse-distance bin set in 32B-sector units (R16
# Section 2). Each tuple is ``(rd_min, rd_max)`` inclusive.
REUSE_DISTANCE_BIN_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (2, 2), (3, 3),
    (4, 7), (8, 15), (16, 31), (32, 63),
    (64, 127), (128, 255), (256, 511), (512, 1023),
    (1024, 2047), (2048, 4095), (4096, 8191),
    (8192, 16383), (16384, 32767), (32768, 65535),
    (65536, 131071), (131072, 262143), (262144, 524287),
    (524288, 1048575), (1048576, 2097151),
    (2097152, 4194303), (4194304, 8388607),
    (8388608, 16777215), (16777216, 33554431),
    (33554432, 67108863), (67108864, 134217727),
    (134217728, 268435455), (268435456, 536870911),
    (536870912, 1073741823), (1073741824, 2147483647),
    (2147483648, 9223372036854775806),
)


HIERARCHY_LAYERS: tuple[str, ...] = (
    "sector", "line", "allocation", "region",
)


# Producer-side typed-reason set. Three reasons
# (cache_model_capacity_missing, cache_model_capacity_inconsistent,
# cache_model_underconfigured) are pinned in the original-plan
# exclusion-taxonomy enumeration; the other 14 are round-added per the
# C-AC-C lockstep discipline. See
# ``tools/nvbit_tracer/test_exclusion_taxonomy_parity.py`` for
# the parity surfaces.
HRD_PROFILE_TYPED_REASONS: frozenset[str] = frozenset({
    # Reuse-state reasons (9; round-added).
    "sector_reuse_cold",
    "sector_reuse_infinite",
    "line_reuse_cold",
    "line_reuse_infinite",
    "allocation_reuse_cold",
    "allocation_reuse_infinite",
    "region_reuse_cold",
    "region_reuse_infinite",
    "reuse_distance_unobservable",
    # Cache-model spec reasons (6; 2 plan-enumerated + 4 round-added).
    "cache_model_capacity_missing",
    "cache_model_capacity_inconsistent",
    "cache_model_capacity_static_card_disagreement",
    "cache_model_geometry_missing",
    "cache_model_geometry_inconsistent",
    "cache_model_geometry_static_card_disagreement",
    # Bin-quality reason (1; plan-enumerated).
    "cache_model_underconfigured",
    # Compute-budget reason (1; round-added in R54). Fired by
    # the HRD producer when the input sector_stream.jsonl
    # exceeds `HRD_MAX_SECTOR_STREAM_RECORDS` — defense-in-depth
    # above the R53 Fenwick-tree complexity fix, which itself
    # made real-world canonical captures fast enough that the
    # budget cap is never expected to trigger on observed
    # inputs. The cap protects against future pathological
    # inputs (adversarial >1 GB streams, etc.).
    "hrd_compute_budget_exceeded",
})


# R54: Defense-in-depth budget cap on the HRD producer's input
# size. The Fenwick-tree-backed reuse_distance_tracker.py
# returns each access in O(log N), so per-launch compute is
# O(N log N); for sector_stream inputs around 33k records (the
# warm_cold_pair_relaunch trace_00 size) the producer completes
# in well under a second. The cap is set comfortably above
# 33k * 4 layers + headroom so canonical sm_120 captures never
# trigger it on currently-observed inputs; it fires only when
# an adversarial / pathological trace would otherwise consume
# unbounded compute. When fired, the producer skips the
# reuse-distance computation entirely and emits a typed-
# unavailable bundle with `typed_reason ==
# "hrd_compute_budget_exceeded"`.
HRD_MAX_SECTOR_STREAM_RECORDS: int = 10_000_000


# ---------------------------------------------------------------------------
# Bootstrap percentile CI helper.
# ---------------------------------------------------------------------------


def _percentile_bootstrap_ci_launch_unit(
    per_launch_hits: Sequence[Sequence[bool]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CI_CONFIDENCE,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Compute (ci_low, ci_high, ci_half_width) for the per-bin
    hit-rate fraction via launch-unit percentile bootstrap.

    `per_launch_hits` is a sequence of per-launch hit-lists; each
    inner list is a sequence of booleans (True == hit) recorded
    in one kernel launch. The bootstrap resamples WHOLE LAUNCHES
    with replacement (the R16 task10 artifact's pinned resampling
    unit), pools the resampled launches' booleans, and computes
    the hit-rate statistic over the pool.

    Bootstrap resamples whole kernel launches with replacement,
    not individual lane accesses. Lane accesses within a single
    launch are highly correlated, so flat per-access resampling
    overstates independence and produces an overconfident CI;
    launch-unit resampling preserves the launch-level
    correlation structure.
    """
    if not per_launch_hits:
        return (0.0, 0.0, 0.0)
    n_launches = len(per_launch_hits)
    if all(not lh for lh in per_launch_hits):
        return (0.0, 0.0, 0.0)
    rng = rng if rng is not None else random.Random(0)
    rates: list[float] = []
    for _ in range(resamples):
        pool_hits = 0
        pool_n = 0
        for _ in range(n_launches):
            launch = per_launch_hits[rng.randrange(n_launches)]
            for h in launch:
                if h:
                    pool_hits += 1
                pool_n += 1
        rate = (pool_hits / pool_n) if pool_n > 0 else 0.0
        rates.append(rate)
    rates.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(math.floor(alpha / 2.0 * resamples)))
    hi_idx = min(resamples - 1, int(math.ceil((1 - alpha / 2.0) * resamples)) - 1)
    ci_low = rates[lo_idx]
    ci_high = rates[hi_idx]
    half_width = (ci_high - ci_low) / 2.0
    return ci_low, ci_high, half_width


# ---------------------------------------------------------------------------
# Bundle-construction helpers.
# ---------------------------------------------------------------------------


def _bin_index_for_rd(rd: int) -> int | None:
    """Return the index into ``REUSE_DISTANCE_BIN_BOUNDS`` for a
    finite reuse distance. Returns ``None`` for the
    ``RD_INFINITY_SENTINEL`` marker (typed-unavailable arm)."""
    if rd == RD_INFINITY_SENTINEL:
        return None
    for i, (lo, hi) in enumerate(REUSE_DISTANCE_BIN_BOUNDS):
        if lo <= rd <= hi:
            return i
    return None


def _admitted_bin(
    *,
    layer: str,
    rd_min: int,
    rd_max: int,
    per_launch_hits: Sequence[Sequence[bool]],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Build an `ok` per-bin record from launch-grouped hit lists.

    The bootstrap resamples whole launch units (R16 task10
    Section 3); the per-bin point estimate is the pooled
    hit_count / observation_count.
    """
    pooled_hits = [h for launch in per_launch_hits for h in launch]
    n = len(pooled_hits)
    hit_count = sum(1 for h in pooled_hits if h)
    hit_rate = hit_count / n if n > 0 else 0.0
    ci_low, ci_high, half_width = _percentile_bootstrap_ci_launch_unit(
        per_launch_hits, rng=rng,
    )
    return {
        "result_kind": "ok",
        "hierarchy_layer": layer,
        "rd_sector_units_min": rd_min,
        "rd_sector_units_max": rd_max,
        "observation_count": n,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": half_width,
        "low_confidence": False,
    }


def _underconfigured_bin(
    *,
    layer: str,
    rd_min: int,
    rd_max: int,
    observation_count: int,
) -> dict[str, Any]:
    """Build a typed-unavailable per-bin record with
    `cache_model_underconfigured`. Used for both trigger arms
    (``observation_count < 50`` AND
    ``observation_count >= 50 AND ci_half_width > 0.05``)."""
    return {
        "result_kind": "typed_unavailable",
        "typed_reason": "cache_model_underconfigured",
        "hierarchy_layer": layer,
        "rd_sector_units_min": rd_min,
        "rd_sector_units_max": rd_max,
        "observation_count": observation_count,
        "hit_count": None,
        "hit_rate": None,
        "ci_low": None,
        "ci_high": None,
        "ci_half_width": None,
        "low_confidence": None,
    }


def _typed_unavailable_bin(
    *,
    layer: str,
    typed_reason: str,
    observation_count: int,
) -> dict[str, Any]:
    """Build a typed-unavailable per-bin record for cold/infinity
    or trace-drop reasons. The `rd_sector_units_*` fields are
    `null` because the reuse distance is undefined."""
    if typed_reason not in HRD_PROFILE_TYPED_REASONS:
        raise ValueError(
            f"Unknown typed_reason {typed_reason!r}; not in "
            f"HRD_PROFILE_TYPED_REASONS"
        )
    return {
        "result_kind": "typed_unavailable",
        "typed_reason": typed_reason,
        "hierarchy_layer": layer,
        "rd_sector_units_min": None,
        "rd_sector_units_max": None,
        "observation_count": observation_count,
        "hit_count": None,
        "hit_rate": None,
        "ci_low": None,
        "ci_high": None,
        "ci_half_width": None,
        "low_confidence": None,
    }


# ---------------------------------------------------------------------------
# Conditional descent and observation classification.
# ---------------------------------------------------------------------------


def _classify_observation(
    *,
    sector_rd: int | None,
    sector_state: str,
    line_rd: int | None,
    line_state: str,
    allocation_rd: int | None,
    allocation_state: str,
    region_rd: int | None,
    region_state: str,
) -> tuple[str, int | None, str | None]:
    """Implement the conditional-descent rule from Section 1.

    Returns ``(layer, rd, typed_reason_if_unobservable)`` — when
    ``typed_reason_if_unobservable`` is non-None, the
    observation produces a typed-unavailable bin record at
    ``layer`` instead of a numeric reuse distance.

    Strict descent: a finer layer is consulted FIRST; the
    coarser layer is only consulted when the finer is cold or
    infinity. A finite reuse distance at a finer layer is
    interpreted at that layer (no descent on finite values).
    """
    layers = (
        ("sector", sector_rd, sector_state, "sector_reuse_cold",
         "sector_reuse_infinite"),
        ("line", line_rd, line_state, "line_reuse_cold",
         "line_reuse_infinite"),
        ("allocation", allocation_rd, allocation_state,
         "allocation_reuse_cold",
         "allocation_reuse_infinite"),
        ("region", region_rd, region_state, "region_reuse_cold",
         "region_reuse_infinite"),
    )
    last_typed_reason: str | None = None
    last_layer: str = "sector"
    for layer, rd, state, cold_reason, infinite_reason in layers:
        if state == "cold":
            last_typed_reason = cold_reason
            last_layer = layer
            continue
        if state == "infinite" or rd == RD_INFINITY_SENTINEL:
            last_typed_reason = infinite_reason
            last_layer = layer
            continue
        if rd is None:
            last_typed_reason = "reuse_distance_unobservable"
            last_layer = layer
            continue
        return layer, int(rd), None
    # All four layers were cold / infinity / unobservable; emit a
    # typed-unavailable record at the coarsest layer reached.
    return last_layer, None, last_typed_reason


# ---------------------------------------------------------------------------
# Main API: compute_hrd_profile.
# ---------------------------------------------------------------------------


def _normalize_launch_to_observations(launch: Any) -> Sequence[Any]:
    """Resolve a launch entry to its per-access observation
    iterable. The HRD producer's `_BuildResult.launches` may be
    either a bare ``list[dict]`` (legacy synthetic test fixtures
    that predate the per-launch-record migration) or a per-launch
    record dict carrying an ``observations`` key (the current
    delivery surface).

    Mirrors the defensive pattern in
    `cache_predictor.py::_aggregate_launches_from_observations` so
    every consumer of `_BuildResult.launches` sees the same
    boundary. Returns an empty tuple for shapes that do not match
    either form rather than raising — the per-bin walk that
    follows iterates harmlessly over the empty iterable."""
    if isinstance(launch, list):
        return launch
    obs_attr = getattr(launch, "observations", None)
    if obs_attr is not None:
        return obs_attr
    if isinstance(launch, dict):
        return launch.get("observations", [])
    return ()


def compute_hrd_profile(
    *,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    cache_model_spec: dict[str, Any],
    launches: Sequence[Any],
    cache_model_evidence: Sequence[dict[str, Any]] = (),
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Compute the schema-conformant HRD profile bundle.

    Inputs
    ------
    ``manifest_pairing_index``
        simbench manifest-pairing index: ``slug``, ``shape_point``,
        ``binary_sha256``, ``source_commit``,
        ``kernel_signature_sha256``.
    ``device``
        Live device attributes (Section 9 device sub-block).
    ``cache_model_spec``
        Resolved cache-model spec (Section 9 sub-block); should
        be the output of
        :func:`cache_model_spec.resolve_cache_model_spec`.
    ``launches``
        Sequence of per-launch entries. Each entry is either a
        bare ``list[dict]`` of per-access observation dicts
        (legacy / synthetic-fixture shape) or a per-launch record
        dict with an ``observations`` key carrying the same list
        plus the aggregated diagnostic fields (cold_count,
        finite_miss_count, etc.). Both shapes are accepted; the
        function normalizes via
        :func:`_normalize_launch_to_observations` and consumes
        only the observation iterable. The bootstrap resamples
        whole launches (R16 Section 3); a single-launch trace gets
        one launch entry with N observations, a multi-launch trace
        gets multiple entries. Each observation dict has the same
        per-layer keys as before (``hit``, ``sector_rd``,
        ``sector_state``, ``line_rd``, ``line_state``,
        ``allocation_rd``, ``allocation_state``, ``region_rd``,
        ``region_state``).
    ``cache_model_evidence``
        Optional list of info-record dicts to attach to the
        bundle's evidence list.

    Returns
    -------
    A dict admitting the schema at
    ``workloads/simbench_bw/schemas/hrd_profile.schema.json``.
    """
    rng = rng if rng is not None else random.Random(0)
    # Group per-launch hit-lists by (layer, bin_idx). Each entry
    # in `bin_per_launch_hits[(layer, bin_idx)]` is a list of
    # booleans for that launch; the outer list iterates launches.
    bin_per_launch_hits: dict[tuple[str, int], list[list[bool]]] = {}
    typed_unavailable_counts: dict[tuple[str, str], int] = {}
    for launch_idx, launch_entry in enumerate(launches):
        launch_obs = _normalize_launch_to_observations(launch_entry)
        # Per-launch (layer, bin_idx) hit accumulation.
        per_launch: dict[tuple[str, int], list[bool]] = {}
        for obs in launch_obs:
            layer, rd, typed_reason = _classify_observation(
                sector_rd=obs.get("sector_rd"),
                sector_state=obs.get("sector_state", "finite"),
                line_rd=obs.get("line_rd"),
                line_state=obs.get("line_state", "finite"),
                allocation_rd=obs.get("allocation_rd"),
                allocation_state=obs.get("allocation_state", "finite"),
                region_rd=obs.get("region_rd"),
                region_state=obs.get("region_state", "finite"),
            )
            if typed_reason is not None:
                typed_unavailable_counts[(layer, typed_reason)] = (
                    typed_unavailable_counts.get((layer, typed_reason), 0) + 1
                )
                continue
            assert rd is not None
            bin_idx = _bin_index_for_rd(rd)
            if bin_idx is None:
                typed_unavailable_counts[(layer, "reuse_distance_unobservable")] = (
                    typed_unavailable_counts.get(
                        (layer, "reuse_distance_unobservable"), 0,
                    ) + 1
                )
                continue
            per_launch.setdefault((layer, bin_idx), []).append(
                bool(obs.get("hit"))
            )
        # Append this launch's per-bin lists into the global
        # bin -> [launch1_hits, launch2_hits, ...] structure.
        for key, hits in per_launch.items():
            bin_per_launch_hits.setdefault(key, []).append(hits)
    bins: list[dict[str, Any]] = []
    underconfigured_bin_count = 0
    for (layer, bin_idx), per_launch_hits in sorted(
        bin_per_launch_hits.items()
    ):
        rd_min, rd_max = REUSE_DISTANCE_BIN_BOUNDS[bin_idx]
        pooled_n = sum(len(lh) for lh in per_launch_hits)
        if pooled_n < MINIMUM_BIN_SAMPLE_COUNT:
            bins.append(_underconfigured_bin(
                layer=layer, rd_min=rd_min, rd_max=rd_max,
                observation_count=pooled_n,
            ))
            underconfigured_bin_count += 1
            continue
        admitted = _admitted_bin(
            layer=layer, rd_min=rd_min, rd_max=rd_max,
            per_launch_hits=per_launch_hits, rng=rng,
        )
        if admitted["ci_half_width"] > CI_HALF_WIDTH_TARGET:
            bins.append(_underconfigured_bin(
                layer=layer, rd_min=rd_min, rd_max=rd_max,
                observation_count=pooled_n,
            ))
            underconfigured_bin_count += 1
            continue
        bins.append(admitted)
    for (layer, typed_reason), n in sorted(typed_unavailable_counts.items()):
        bins.append(_typed_unavailable_bin(
            layer=layer, typed_reason=typed_reason,
            observation_count=n,
        ))
    evidence: list[dict[str, Any]] = list(cache_model_evidence)
    bundle_underconfigured = underconfigured_bin_count > 0
    if bundle_underconfigured:
        evidence.append({
            "result_kind": "typed_unavailable",
            "typed_reason": "cache_model_underconfigured",
            "scope": "bundle",
            "underconfigured_bin_count": underconfigured_bin_count,
            "scalar_hit_rate_refusal": True,
        })
    return {
        "schema_version": 1,
        "result_kind": "ok",
        "manifest_pairing_index": dict(manifest_pairing_index),
        "device": dict(device),
        "cache_model_spec": dict(cache_model_spec),
        "binning": {
            "unit": "sector_32b",
            "rd_infinity_sentinel": RD_INFINITY_SENTINEL,
            "ci_method": "percentile_bootstrap",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "ci_confidence": CI_CONFIDENCE,
            "ci_half_width_target": CI_HALF_WIDTH_TARGET,
            "minimum_bin_sample_count": MINIMUM_BIN_SAMPLE_COUNT,
        },
        "resampling": {"unit": "kernel_launch"},
        "bins": bins,
        "bundle_underconfigured": bundle_underconfigured,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Downstream scalar-consumer refusal contract.
# ---------------------------------------------------------------------------


class ScalarHitRateRefusedError(Exception):
    """Raised when a downstream consumer requests a single
    hit-rate scalar from an underconfigured bundle. The
    original-plan HRD-profiler-clause negative test
    (``plans/nonroot_gpu_cache_observability_plan.md:203``)
    requires this refusal: a bundle annotated
    `cache_model_underconfigured` MUST NOT yield a default
    scalar value."""

    def __init__(self, message: str = "") -> None:
        super().__init__(
            message or "scalar hit rate refused: bundle is underconfigured"
        )


def request_scalar_hit_rate(bundle: dict[str, Any]) -> float:
    """Return a single scalar hit-rate from an HRD bundle.

    Refuses with :class:`ScalarHitRateRefusedError` when the
    bundle's ``bundle_underconfigured`` flag is True. Otherwise
    returns the simple weighted mean of admitted-bin hit rates
    (weighted by ``observation_count``).
    """
    if bundle.get("bundle_underconfigured"):
        raise ScalarHitRateRefusedError()
    total_count = 0
    total_hits = 0
    for record in bundle.get("bins", []):
        if record.get("result_kind") == "ok":
            total_count += int(record["observation_count"])
            total_hits += int(record["hit_count"])
    if total_count == 0:
        raise ScalarHitRateRefusedError(
            "scalar hit rate undefined: no admitted bins"
        )
    return total_hits / total_count


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CI_CONFIDENCE",
    "CI_HALF_WIDTH_TARGET",
    "HIERARCHY_LAYERS",
    "HRD_PROFILE_TYPED_REASONS",
    "MINIMUM_BIN_SAMPLE_COUNT",
    "RD_INFINITY_SENTINEL",
    "REUSE_DISTANCE_BIN_BOUNDS",
    "ScalarHitRateRefusedError",
    "compute_hrd_profile",
    "request_scalar_hit_rate",
]
