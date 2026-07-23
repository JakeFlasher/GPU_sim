"""L1 sector-hit-ratio DIAGNOSTIC surface for the sm_120 cache-only replay predictor.

This module emits a coarse L1 hit-ratio diagnostic by replaying each cell's
already-decoded sector-access stream through a SMALL single-L1 capacity model.
It reuses the same sectored set-associative replay engine as the headline L2
predictor, but at the L1 capacity scale.

**This L1 surface is diagnostic only and is NOT load-bearing in this iteration.**
A faithful L1 model would require per-SM / per-CTA scheduling (the 128 KiB
combined L1/shared is private per SM and the access stream would have to be
partitioned across the SMs that actually co-resided), plus .ca / .cg cache-op
eligibility and explicit bypass semantics (global loads that bypass L1, the
streaming vs. caching load distinction). None of that is modeled here: the
diagnostic replays the whole cell stream through one diagnostic L1 and reports
the resulting hit ratio purely as an observability surface.

Honest expectation: because the single-L1, no-per-SM-partition, no-bypass
assumptions are coarse, the diagnostic median absolute error against the L1TEX
ground-truth labels will almost certainly miss the promotion bar. The surface
is therefore expected to be ``load_bearing == false``, and the
:func:`validate_l1_diagnostic` guard enforces that ``load_bearing`` may only be
``true`` when the diagnostic independently reaches the documented accuracy bar
AND explicit L1 eligibility / bypass provenance is present.

It is explicitly NOT a GPU simulator and reads only already-collected traces and
labels; it never invokes privileged measurement, driver-root mutation, or any
privileged profiling surface.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # noqa: E402
import cohort as cohort_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import artifact_io  # noqa: E402

# --------------------------------------------------------------------------
# Emission targets + fixed contract constants.
# --------------------------------------------------------------------------
L1_DIAGNOSTIC_PATH = _REPO_ROOT / "data/local_labels/cache_replay_l1_diagnostic.json"
_SCHEMA_PATH = _HERE / "schemas" / "cache_replay_l1_diagnostic.schema.json"

SCHEMA_VERSION = 1
ALGORITHM_VERSION = "cache_replay_l1_diagnostic_v1"
ARCH = "sm_120"
SKU = "rtx_5060_laptop"
CACHE_LAYER = "L1TEX"

# Documented diagnostic L1 capacity for the RTX 5060 Laptop (sm_120, Blackwell):
# the combined L1 / shared-memory carveout is 128 KiB per SM. The diagnostic
# models a SINGLE L1 of that capacity (no per-SM / per-CTA partition), 4-way
# set-associative, LRU, one slice. This is a deliberately coarse assumption,
# recorded verbatim in the emitted provenance.
L1_DIAGNOSTIC_CAPACITY_BYTES = 128 * 1024  # 131072
L1_DIAGNOSTIC_ASSOCIATIVITY = 4
L1_DIAGNOSTIC_REPLACEMENT = "LRU"
L1_DIAGNOSTIC_NUM_SLICES = 1

# The promotion bar: the diagnostic may only be declared load-bearing if it
# independently reaches this median-absolute-error AND carries explicit L1
# eligibility / bypass provenance.
L1_LOAD_BEARING_MEDABS_BAR = 0.10

# The coarse-assumption note recorded in provenance so the diagnostic scope is
# self-documenting in the emitted artifact.
_L1_ASSUMPTION_NOTE = (
    "single diagnostic L1 of 128 KiB (combined L1/shared per SM on the RTX 5060 "
    "laptop), 4-way LRU, one slice; the whole cell access stream is replayed "
    "through one L1 with NO per-SM / per-CTA scheduling partition and NO "
    ".ca/.cg cache-op eligibility or L1-bypass semantics. This is a coarse "
    "diagnostic observability surface, not a load-bearing L1 prediction."
)


def _l1_diagnostic_config() -> scr.CacheReplayConfig:
    """The fixed diagnostic L1 capacity model (small single L1)."""
    return scr.CacheReplayConfig(
        capacity_bytes=L1_DIAGNOSTIC_CAPACITY_BYTES,
        associativity=L1_DIAGNOSTIC_ASSOCIATIVITY,
        replacement_policy=L1_DIAGNOSTIC_REPLACEMENT,
        num_slices=L1_DIAGNOSTIC_NUM_SLICES,
    )


def build_l1_diagnostic(*, use_cache: bool = True) -> dict[str, Any]:
    """Replay each l1-labeled cell through the diagnostic L1 model and assemble
    the diagnostic surface payload.

    For each cell that HAS an L1TEX label, the cell's already-decoded sector
    stream is replayed through the small diagnostic L1 to produce a predicted L1
    hit ratio. The diagnostic median absolute error is computed against the
    L1TEX ground-truth labels. The surface is marked ``load_bearing`` only when
    the promotion bar is met (it is expected NOT to be met).
    """
    config = _l1_diagnostic_config()
    cohort = cohort_mod.load_cohort()

    rows: list[dict[str, Any]] = []
    preds: list[float] = []
    labels: list[float] = []
    for cell, line_ids, sector_offsets in cohort_mod.iter_cohort_streams(
        cohort, use_cache=use_cache,
    ):
        if cell.l1_label is None:
            continue
        result = scr.replay_access_stream(
            line_ids.tolist(), sector_offsets.tolist(), config,
            classify_misses=True,
        )
        rd = result.as_dict()
        row = {
            "join_key": {
                "slug": cell.slug,
                "shape_point": cell.shape_point,
                "arch": ARCH,
                "family_bucket": cell.family_bucket,
                "cache_layer": CACHE_LAYER,
            },
            "l1_diagnostic_predicted_hit_ratio": rd["predicted_hit_ratio"],
            "l1_ground_truth_hit_ratio": cell.l1_label,
            "absolute_error": abs(rd["predicted_hit_ratio"] - cell.l1_label),
            "miss_breakdown": rd["miss_breakdown"],
            "uncertainty_interval": rd["uncertainty_interval"],
        }
        rows.append(row)
        preds.append(rd["predicted_hit_ratio"])
        labels.append(cell.l1_label)

    median_abs = (
        metrics_mod.median_absolute_error(preds, labels) if preds else None
    )

    # Explicit L1 eligibility / bypass provenance is NOT modeled this iteration;
    # the diagnostic replays the full stream with no .ca/.cg or bypass handling.
    bypass_provenance_present = False

    # The promotion bar: load-bearing only if the diagnostic independently
    # reaches the median-absolute-error bar AND bypass provenance is present.
    meets_accuracy_bar = (
        median_abs is not None and median_abs <= L1_LOAD_BEARING_MEDABS_BAR
    )
    load_bearing = bool(meets_accuracy_bar and bypass_provenance_present)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source_commit": artifact_io.git_head(),
        "arch": ARCH,
        "sku": SKU,
        "cache_layer": CACHE_LAYER,
        "load_bearing": load_bearing,
        "diagnostic_median_absolute_error": median_abs,
        "load_bearing_provenance": {
            "load_bearing_medabs_bar": L1_LOAD_BEARING_MEDABS_BAR,
            "meets_accuracy_bar": meets_accuracy_bar,
            "bypass_provenance_present": bypass_provenance_present,
            "l1_capacity_bytes": L1_DIAGNOSTIC_CAPACITY_BYTES,
            "associativity": L1_DIAGNOSTIC_ASSOCIATIVITY,
            "replacement": L1_DIAGNOSTIC_REPLACEMENT,
            "num_slices": L1_DIAGNOSTIC_NUM_SLICES,
            "assumption_note": _L1_ASSUMPTION_NOTE,
        },
        "cache_parameters": config.provenance(),
        "n_cells": len(rows),
        "rows": rows,
    }
    return payload


def validate_l1_diagnostic(payload: dict[str, Any]) -> None:
    """Raise ``ValueError`` if the payload claims ``load_bearing == true`` but the
    promotion bar is not met.

    The promotion bar (the load-bearing-scope guard): a diagnostic L1 surface may
    only be declared load-bearing when BOTH

      * its diagnostic median absolute error is at most
        :data:`L1_LOAD_BEARING_MEDABS_BAR`, AND
      * explicit L1 eligibility / bypass provenance is present
        (``bypass_provenance_present == true``).

    A ``load_bearing == false`` surface is always admitted; this guard never
    forces the diagnostic to be load-bearing.
    """
    if not isinstance(payload, dict):
        raise ValueError("l1 diagnostic payload must be a dict")
    prov = payload.get("load_bearing_provenance")
    if not isinstance(prov, dict):
        raise ValueError(
            "l1 diagnostic payload missing 'load_bearing_provenance' block"
        )
    load_bearing = payload.get("load_bearing")
    if not isinstance(load_bearing, bool):
        raise ValueError("l1 diagnostic 'load_bearing' must be a bool")
    if not load_bearing:
        return  # a non-load-bearing diagnostic is always admitted.

    medabs = payload.get("diagnostic_median_absolute_error")
    bar = prov.get("load_bearing_medabs_bar", L1_LOAD_BEARING_MEDABS_BAR)
    bypass_present = prov.get("bypass_provenance_present")

    if not isinstance(bypass_present, bool) or not bypass_present:
        raise ValueError(
            "l1 diagnostic declares load_bearing=true but "
            "'bypass_provenance_present' is not true; explicit L1 eligibility / "
            "bypass provenance is required to promote the surface"
        )
    if medabs is None or not isinstance(medabs, (int, float)):
        raise ValueError(
            "l1 diagnostic declares load_bearing=true but has no numeric "
            "'diagnostic_median_absolute_error'"
        )
    if float(medabs) > float(bar):
        raise ValueError(
            f"l1 diagnostic declares load_bearing=true but its median absolute "
            f"error {float(medabs):.4f} exceeds the promotion bar {float(bar):.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Emit the L1 sector-hit-ratio diagnostic surface (sm_120)"
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    payload = build_l1_diagnostic(use_cache=not args.no_cache)
    validate_l1_diagnostic(payload)  # fail-closed before emitting
    artifact_io.atomic_write_json(L1_DIAGNOSTIC_PATH, payload)
    print(f"n_cells: {payload['n_cells']}")
    print(f"diagnostic median absolute error: {payload['diagnostic_median_absolute_error']}")
    print(f"load_bearing: {payload['load_bearing']}")
    print(f"wrote {L1_DIAGNOSTIC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
