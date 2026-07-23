"""Cumulative-window MRC sampler for the adaptive-sampler clause.

Implements the original-plan Section 7 algorithm: starting from small
trace windows, accumulate observations, build MRC estimates at sector
and line layers, and stop when the max change across selected MRC
quantiles is below `DISTANCE_THRESHOLD = 0.02` for `K = 3` consecutive
windows. Per the task16-ac11-analysis Section 3 contract, each window
k builds an HRD over the cumulative observations from windows `0..k`
(cumulative mode) so the MRC stabilises monotonically.

The sampler consumes the upstream `hrd_profile.json` produced by
`tools/hrd_profiler/bundle_from_trace.py`. Per Section 6 of the
artifact, the sampler is gated by HRD-layer availability: if the
upstream bundle reports `bundle_underconfigured` or the requested
layer (`sector` or `line`) lacks usable bins, the sampler emits
`hrd_layer_unavailable`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from trace_convergence_ledger import (  # type: ignore[import-not-found]  # noqa: E402
    DISTANCE_THRESHOLD,
    HIERARCHY_LAYERS,
    K_CONSECUTIVE_WINDOWS,
    MRC_QUANTILES,
    TRACE_WINDOW_SIZE_BYTES,
    is_usable_admitted_bin,
    usable_admitted_bins,
)


def compute_mrc_at_quantile(
    bins: list[dict[str, Any]],
    *,
    layer: str,
    quantiles: tuple[float, ...] = MRC_QUANTILES,
) -> list[dict[str, Any]]:
    """Reduce an admitted HRD bin table at one hierarchy layer to per-
    quantile MRC fractions. Consumes the schema declared in
    `workloads/simbench_bw/schemas/hrd_profile.schema.json` admitted
    branch: `result_kind == "ok"`, `hierarchy_layer in {sector,line,...}`,
    integer `rd_sector_units_min/max`, integer `observation_count`,
    integer `hit_count`, numeric `hit_rate`. Typed-unavailable bins
    (`result_kind == "typed_unavailable"`) are skipped — the controller
    routes those through the `hrd_layer_unavailable` admission arm
    when no admitted bins remain at the requested layer.

    The MRC at quantile `q` is `cumulative_hit_count /
    total_observations` at the bin where the cumulative observation
    count first crosses `q * total_observations`, with bins sorted by
    `(rd_sector_units_min, rd_sector_units_max)` ascending. The
    selected bin's cached `hit_rate` is NOT used as the MRC value —
    the cumulative formula is the only correct expression.
    """
    layer_bins = usable_admitted_bins(bins, layer=layer)
    total_observations = sum(b["observation_count"] for b in layer_bins)
    out: list[dict[str, Any]] = []
    for q in quantiles:
        if total_observations == 0:
            out.append({
                "hierarchy_layer": layer,
                "quantile": q,
                "mrc_hit_rate_fraction": 0.0,
                "quantile_observation_count": 0,
            })
            continue
        target = q * total_observations
        cum_obs = 0
        cum_hit = 0
        chosen_count = 0
        chosen_hit = 0
        for b in layer_bins:
            cum_obs += b["observation_count"]
            cum_hit += b["hit_count"]
            if cum_obs >= target:
                chosen_count = cum_obs
                chosen_hit = cum_hit
                break
        mrc_q = chosen_hit / total_observations if total_observations > 0 else 0.0
        out.append({
            "hierarchy_layer": layer,
            "quantile": q,
            "mrc_hit_rate_fraction": float(mrc_q),
            "quantile_observation_count": int(chosen_count),
        })
    return out


def compute_distance(
    mrc_prev: list[dict[str, Any]],
    mrc_curr: list[dict[str, Any]],
) -> float:
    """`max over (layer, quantile) abs(MRC_curr - MRC_prev)` per
    Section 5 of the artifact. Returns `0.0` if either side is empty
    (no work to compare; window 0 has no previous)."""
    if not mrc_prev or not mrc_curr:
        return 0.0
    prev_map = {
        (r["hierarchy_layer"], float(r["quantile"])): float(r["mrc_hit_rate_fraction"])
        for r in mrc_prev
    }
    max_delta = 0.0
    for r in mrc_curr:
        key = (r["hierarchy_layer"], float(r["quantile"]))
        if key in prev_map:
            delta = abs(float(r["mrc_hit_rate_fraction"]) - prev_map[key])
            if delta > max_delta:
                max_delta = delta
    return max_delta


def iterate_windows(
    window_bins: list[list[dict[str, Any]]],
    *,
    quantiles: tuple[float, ...] = MRC_QUANTILES,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Walk a list of cumulative HRD bin tables (one per window;
    window `k` already accumulates observations from windows `0..k`)
    and emit per-window records the schema requires. The caller
    supplies cumulative HRDs because cumulative-mode is the
    Section-7-step-2 contract.
    """
    out: list[dict[str, Any]] = []
    prev_mrc: list[dict[str, Any]] = []
    trailing_converged = 0
    for k, bins in enumerate(window_bins):
        mrc_records: list[dict[str, Any]] = []
        for layer in HIERARCHY_LAYERS:
            mrc_records.extend(
                compute_mrc_at_quantile(bins, layer=layer, quantiles=quantiles),
            )
        if k == 0:
            distance: float | None = None
            converged = False
            trailing_converged = 0
        else:
            distance = compute_distance(prev_mrc, mrc_records)
            converged = distance <= distance_threshold
            trailing_converged = trailing_converged + 1 if converged else 0
        observation_count = sum(
            int(b["observation_count"])
            for b in bins
            if is_usable_admitted_bin(b)
        )
        trace_bytes = (k + 1) * TRACE_WINDOW_SIZE_BYTES
        out.append({
            "window_index": k,
            "observation_count_cumulative": observation_count,
            "trace_bytes_cumulative": trace_bytes,
            "mrc_at_quantile": mrc_records,
            "distance_to_previous": distance,
            "converged": converged,
        })
        prev_mrc = mrc_records
    return out


def trailing_converged_count(windows: list[dict[str, Any]]) -> int:
    """Trailing consecutive count of windows whose `converged` is true.
    Used by both the producer (admission decision) and the verifier
    (cached-vs-recomputed `windows_converged` check)."""
    n = 0
    for w in reversed(windows):
        if w.get("converged"):
            n += 1
        else:
            break
    return n


def convergence_admission_for(
    windows: list[dict[str, Any]],
    *,
    k: int = K_CONSECUTIVE_WINDOWS,
    threshold: float = DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    """Return the `admission` block for an `ok` outcome. The caller is
    responsible for routing typed-unavailable cases (budget cap, HRD
    layer unavailable, quantile underconfigured, controller disabled)
    to their respective admission shapes; this helper is the success
    path only.
    """
    if len(windows) < k + 1:
        raise ValueError(
            f"need at least {k + 1} windows to admit; got {len(windows)}",
        )
    converged = trailing_converged_count(windows)
    if converged < k:
        raise ValueError(
            f"trailing_converged_count={converged} < K={k}; "
            "controller must route to typed_unavailable",
        )
    final_residual = windows[-1].get("distance_to_previous")
    if final_residual is None or final_residual > threshold:
        raise ValueError(
            f"final_residual={final_residual} > threshold={threshold}",
        )
    return {
        "result_kind": "ok",
        "windows_converged": converged,
        "final_residual": float(final_residual),
    }


def budget_cap_admission(
    *,
    windows: list[dict[str, Any]],
    max_trace_bytes: int,
    actual_trace_bytes: int,
    trace_volume_exceeded_sidecar_path: str,
) -> dict[str, Any]:
    """`trace_convergence_not_reached_budget_cap` admission shape.
    Composes with `trace_volume_exceeded.json`: the sidecar path field
    binds the convergence sidecar to the existing trace-budget
    sidecar so the verifier can prove dual-sidecar emission."""
    last_distance = windows[-1].get("distance_to_previous") if windows else None
    return {
        "result_kind": "typed_unavailable",
        "typed_reason": "trace_convergence_not_reached_budget_cap",
        "max_trace_bytes": int(max_trace_bytes),
        "actual_trace_bytes": int(actual_trace_bytes),
        "trace_volume_exceeded_sidecar_path": trace_volume_exceeded_sidecar_path,
        "windows_observed": len(windows),
        "trailing_converged_windows": trailing_converged_count(windows),
        "final_residual": last_distance,
    }


def underconfigured_admission(
    *,
    hierarchy_layer: str,
    quantile: float,
    minimum_bin_sample_count: int,
    observed_bin_sample_count: int,
) -> dict[str, Any]:
    """`mrc_quantile_underconfigured` admission shape."""
    return {
        "result_kind": "typed_unavailable",
        "typed_reason": "mrc_quantile_underconfigured",
        "hierarchy_layer": hierarchy_layer,
        "quantile": float(quantile),
        "minimum_bin_sample_count": int(minimum_bin_sample_count),
        "observed_bin_sample_count": int(observed_bin_sample_count),
    }


def hrd_unavailable_admission(
    *,
    hrd_profile_path: str,
    bundle_underconfigured: bool,
    requested_layers: tuple[str, ...],
    unavailable_layer: str,
    usable_bin_count: int,
) -> dict[str, Any]:
    """`hrd_layer_unavailable` admission shape."""
    return {
        "result_kind": "typed_unavailable",
        "typed_reason": "hrd_layer_unavailable",
        "hrd_profile_path": hrd_profile_path,
        "bundle_underconfigured": bool(bundle_underconfigured),
        "requested_layers": list(requested_layers),
        "unavailable_layer": unavailable_layer,
        "usable_bin_count": int(usable_bin_count),
    }


def disabled_admission() -> dict[str, Any]:
    """`trace_convergence_disabled` admission shape — emitted when a
    configuration override disables the stopping rule. The negative
    test contract requires this typed reason rather than silent
    truncation."""
    return {
        "result_kind": "typed_unavailable",
        "typed_reason": "trace_convergence_disabled",
        "config_key": "trace_convergence_enabled",
        "config_value": False,
    }


def window_count_underconfigured_admission(
    *,
    observed_window_count: int,
    actual_trace_bytes: int,
) -> dict[str, Any]:
    """`trace_convergence_window_count_underconfigured` admission
    shape — emitted when the trace is admittedly sized (no budget cap
    breach) but does not have enough cumulative bytes for K consecutive
    windows. The K=3 stationary stopping rule requires at least K+1
    cumulative windows (one baseline + K trailing) at
    `TRACE_WINDOW_SIZE_BYTES` granularity; a trace shorter than
    `(K+1) * TRACE_WINDOW_SIZE_BYTES` cannot populate K trailing
    converged windows. The producer emits this typed-unavailable arm
    rather than fabricating convergence by duplicating a single HRD
    bundle into multiple windows (the synthetic-duplication trap
    exposed by the R37 review).
    """
    required_window_count = K_CONSECUTIVE_WINDOWS + 1
    required_trace_bytes = required_window_count * TRACE_WINDOW_SIZE_BYTES
    return {
        "result_kind": "typed_unavailable",
        "typed_reason": "trace_convergence_window_count_underconfigured",
        "observed_window_count": int(observed_window_count),
        "required_window_count": int(required_window_count),
        "actual_trace_bytes": int(actual_trace_bytes),
        "required_trace_bytes": int(required_trace_bytes),
    }
