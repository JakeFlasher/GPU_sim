"""trace_convergence_confidence producer surface for the adaptive-sampler clause.

Authoritative producer-side constants and bundle assembly. Mirrors the
producer-module pattern from `tools/perturbation_harness/ledger.py`,
`tools/order_projections/projector.py`, and
`tools/sharing_locality/classifier.py`: strict typed-reason set as a
frozenset (the parity inventory's `producer_python_constant` surface),
canonical-JSON `windows_digest` helper, and `build_bundle(...)` that
strict-validates output against the on-disk
`workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`
before returning.

Implements the original-plan Section 7 Condens-style adaptive NVBit
sampler: cumulative-window MRC at sector + line layers, K=3 stationary
stopping rule with `distance_threshold = 0.02` hit-rate-fraction. Module
basename `trace_convergence_ledger.py` (not `ledger.py`) chosen per
`BL-20260505-tools-subdir-module-name-collision` to avoid colliding
with `tools/perturbation_harness/ledger.py`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import jsonschema

# Producer-side typed-reason set. The parity inventory's
# `producer_python_constant` surface is this constant; the schema's
# admission `oneOf` typed-unavailable arms mirror it surface-for-surface.
TRACE_CONVERGENCE_TYPED_REASONS: frozenset[str] = frozenset({
    "trace_convergence_not_reached_budget_cap",
    "mrc_quantile_underconfigured",
    "hrd_layer_unavailable",
    "trace_convergence_disabled",
    "trace_convergence_window_count_underconfigured",
})

# Pinned constants per task16-ac11-analysis.md Sections 4 / 5 / 3.
MRC_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
DISTANCE_THRESHOLD: float = 0.02
K_CONSECUTIVE_WINDOWS: int = 3
TRACE_WINDOW_SIZE_BYTES: int = 16 * 1024 * 1024
DISTANCE_METRIC: str = "max_abs_mrc_quantile_delta"
HIERARCHY_LAYERS: tuple[str, ...] = ("sector", "line")
SCHEMA_VERSION: int = 1

TARGET_ARCH = "sm_120-laptop"


def is_usable_admitted_bin(
    record: Any,
    *,
    layer: str | None = None,
) -> bool:
    """Return True iff `record` is a usable admitted HRD bin record per
    `workloads/simbench_bw/schemas/hrd_profile.schema.json`'s admitted
    branch. Required fields and types:
    - `result_kind == "ok"`
    - `hierarchy_layer in HIERARCHY_LAYERS` (and equals `layer` when
      `layer` is provided);
    - `rd_sector_units_min` and `rd_sector_units_max` are ints;
    - `observation_count` is int and > 0;
    - `hit_count` is int and >= 0;
    - `hit_rate` is numeric (int or float).

    The R37 producer used non-schema field names (`bin_sample_count`,
    `hit_rate_fraction`); records with those fields and not the schema
    fields are rejected. Codex's R37 + R38 reviews flagged the
    duplicated filter contract across `compute_mrc_at_quantile`,
    `_emit_trace_convergence_confidence_for_run_summary`, and
    `run_sampler.py`; this single helper is the shared source of truth
    so all three call sites filter identically.
    """
    if not isinstance(record, dict):
        return False
    if record.get("result_kind") != "ok":
        return False
    record_layer = record.get("hierarchy_layer")
    if record_layer not in HIERARCHY_LAYERS:
        return False
    if layer is not None and record_layer != layer:
        return False
    rd_min = record.get("rd_sector_units_min")
    rd_max = record.get("rd_sector_units_max")
    if not isinstance(rd_min, int) or not isinstance(rd_max, int):
        return False
    if isinstance(rd_min, bool) or isinstance(rd_max, bool):
        return False
    obs = record.get("observation_count")
    if not isinstance(obs, int) or isinstance(obs, bool) or obs <= 0:
        return False
    hit = record.get("hit_count")
    if not isinstance(hit, int) or isinstance(hit, bool) or hit < 0:
        return False
    hit_rate = record.get("hit_rate")
    if isinstance(hit_rate, bool) or not isinstance(hit_rate, (int, float)):
        return False
    return True


def usable_admitted_bins(
    bins: Sequence[Any],
    *,
    layer: str,
) -> list[dict[str, Any]]:
    """Return the admitted bins for `layer` sorted by
    `(rd_sector_units_min, rd_sector_units_max)`. Filters with
    `is_usable_admitted_bin` so callers cannot accept R37-shaped
    records (`bin_sample_count`/`hit_rate_fraction`) by accident."""
    if layer not in HIERARCHY_LAYERS:
        raise ValueError(f"unknown hierarchy_layer: {layer!r}")
    filtered = [b for b in bins if is_usable_admitted_bin(b, layer=layer)]
    filtered.sort(
        key=lambda b: (b["rd_sector_units_min"], b["rd_sector_units_max"]),
    )
    return filtered

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "workloads" / "simbench_bw" / "schemas"
    / "trace_convergence_confidence.schema.json"
)


def canonicalize_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project `windows[]` to the canonical shape over which the
    `windows_digest` is computed. Per Section 8 of the artifact: the
    projection includes only the six per-window fields that affect
    convergence (`window_index`, `observation_count_cumulative`,
    `trace_bytes_cumulative`, `mrc_at_quantile`, `distance_to_previous`,
    `converged`). The `mrc_at_quantile` value is itself canonicalized:
    a list sorted by `(hierarchy_layer, quantile)` whose per-quantile
    dict keys are exactly `{hierarchy_layer, quantile,
    mrc_hit_rate_fraction, quantile_observation_count}`. Future
    schema-additive widenings land a new `_v2` digest field rather
    than mutating v1.
    """
    records: list[dict[str, Any]] = []
    for w in windows:
        mrc = sorted(
            (
                {
                    "hierarchy_layer": q["hierarchy_layer"],
                    "quantile": q["quantile"],
                    "mrc_hit_rate_fraction": q["mrc_hit_rate_fraction"],
                    "quantile_observation_count": q["quantile_observation_count"],
                }
                for q in w["mrc_at_quantile"]
            ),
            key=lambda q: (q["hierarchy_layer"], q["quantile"]),
        )
        records.append({
            "window_index": w["window_index"],
            "observation_count_cumulative": w["observation_count_cumulative"],
            "trace_bytes_cumulative": w["trace_bytes_cumulative"],
            "mrc_at_quantile": list(mrc),
            "distance_to_previous": w["distance_to_previous"],
            "converged": w["converged"],
        })
    return records


def windows_digest(windows: list[dict[str, Any]]) -> str:
    """Hex sha256 over the canonical JSON projection of `windows[]`.
    Sorted keys + compact separators; per-quantile records sorted by
    `(hierarchy_layer, quantile)`. The verifier MUST recompute via
    this exact recipe; cached-vs-recomputed disagreement is
    `BudgetViolationUntracked`-equivalent.
    """
    records = canonicalize_windows(windows)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_pairing_index_dict(
    *, slug: str, shape_point: str, binary_sha256: str,
    source_commit: str, kernel_signature_sha256: str,
) -> dict[str, str]:
    """Construct the manifest_pairing_index dict the schema requires."""
    return {
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": binary_sha256,
        "source_commit": source_commit,
        "kernel_signature_sha256": kernel_signature_sha256,
    }


def build_bundle(
    *,
    manifest_pairing_index: dict[str, str],
    device: dict[str, Any],
    hrd_profile_path: str,
    windows: list[dict[str, Any]],
    admission: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Assemble + strict-validate the trace_convergence_confidence
    bundle. `manifest_pairing_index`, `device`, and `windows[]` shapes
    must already match the schema; this helper wraps them in the
    top-level required-field envelope and validates via
    `jsonschema.Draft202012Validator`. Returns the validated bundle.
    Raises `jsonschema.ValidationError` on failure.
    """
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_pairing_index": manifest_pairing_index,
        "device": device,
        "hrd_profile_path": hrd_profile_path,
        "mrc_quantiles": list(MRC_QUANTILES),
        "distance_metric": DISTANCE_METRIC,
        "k_consecutive_windows": K_CONSECUTIVE_WINDOWS,
        "distance_threshold": DISTANCE_THRESHOLD,
        "trace_window_size_bytes": TRACE_WINDOW_SIZE_BYTES,
        "windows": windows,
        "admission": admission,
        "source_commit": source_commit,
    }
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(bundle)
    return bundle


def per_typed_reason_count(admission: dict[str, Any]) -> dict[str, int]:
    """Per-typed-reason breakdown for the run-summary cache. Returns a
    dict keyed by every reason in `TRACE_CONVERGENCE_TYPED_REASONS`,
    with value 1 for the matching typed_unavailable admission and 0
    otherwise. The admitted `result_kind=ok` admission produces all
    zeros."""
    counts = {r: 0 for r in TRACE_CONVERGENCE_TYPED_REASONS}
    if admission.get("result_kind") == "typed_unavailable":
        reason = admission.get("typed_reason")
        if reason in counts:
            counts[reason] = 1
    return counts
