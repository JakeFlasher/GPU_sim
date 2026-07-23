"""One-shot regeneration script for the refreshed sm_86 predictor
sweep.

Reads the 5 admitted-cell baseline bundles from
`data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/predictor_bundles/`
to extract the per-cell trace paths + manifest_pairing_index +
device tag, then invokes `cache_predictor.predict_cache_hit_rate(...)`
with the post-refinement parameters:

  cache_model_num_l2_slices=6
  cache_model_num_l2_slices_source="device_properties_manifest"
  cache_model_cta_interleave_enabled=True
  cache_model_hrd_second_layer_enabled=True
  cache_model_geometry_provenance={arch, sku, driver_version,
                                   cuda_toolkit_version,
                                   nvbit_build, manifest_sha256}

The refreshed bundles land at
`data/sm_86_predictor_accuracy_recovery/refreshed_sweep_round_3/predictor_bundles/`.
Also emits
`data/sm_86_predictor_accuracy_recovery/refreshed_sweep_round_3/bundle_provenance_diagnostic.json`
listing the field-presence audit for both the baseline + refreshed
sweeps (so a reviewer can grep the field-by-field state without
re-running the scanner).

Dependency: protobuf Python runtime + the generated `*_pb2.py`
modules under `workloads/simbench_bw/_pb_generated/`. Run via:

  uv run --with protobuf python3 \
    tools/sm_86_predictor_accuracy_recovery/regenerate_refreshed_sweep.py

This is a one-shot regeneration script; not part of the
production sweep harness. The production harness lives at
`tools/local_labels/run_headline_cohort_predict_compare.py`. This
script's narrow purpose: produce the refreshed sweep needed for
the bundle-provenance scanner's real-sweep positive gate + the
ablation artifact for the first sub-slice empirical closure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

sys.path.insert(0, str(_REPO_ROOT / "tools" / "hrd_profiler"))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "nvbit_tracer"))
sys.path.insert(0, str(_REPO_ROOT / "workloads" / "simbench_bw"))

from cache_predictor import predict_cache_hit_rate  # noqa: E402


_BASELINE_SWEEP_DIR = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "sweeps"
    / "sweep_20260514T165947Z_a1closure_ebe60dd"
    / "predictor_bundles"
)
_REFRESHED_SWEEP_DIR = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "refreshed_sweep_round_3"
    / "predictor_bundles"
)
_DIAGNOSTIC_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "refreshed_sweep_round_3"
    / "bundle_provenance_diagnostic.json"
)
_MANIFEST_PATH = (
    _REPO_ROOT / "tools" / "hrd_profiler" / "device_properties.json"
)

# Five admitted activation cells per the sweep's comparison bundle.
_ADMITTED_CELL_BUNDLES = [
    "sm_86__layernorm_fwd_bf16_sm80__timing_shape.json",
    "sm_86__rmsnorm_fwd_bf16_sm80__timing_shape.json",
    "sm_86__silu_fwd_bf16_sm80__correctness_shape.json",
    "sm_86__silu_fwd_bf16_sm80__timing_shape.json",
    "sm_86__softmax_safe_bf16_sm80__timing_shape.json",
]

# Post-refinement geometry provenance block. Sourced from
# tools/hrd_profiler/device_properties.json sm_86 entry; the
# manifest_sha256 is computed from the file at runtime.
_GEOMETRY_PROVENANCE_TEMPLATE = {
    "arch": "sm_86",
    "sku": "rtx_3090_ti",
    "driver_version": "595.71.05",
    "cuda_toolkit_version": "13.2",
    "nvbit_build": "1.8",
}


def _compute_manifest_sha256() -> str:
    with _MANIFEST_PATH.open("rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _resolve_trace_dir(baseline_bundle: dict) -> Path:
    """Extract the trace_dir from the baseline bundle's
    input_sidecar_digests.sector_stream.path field."""
    isd = baseline_bundle.get("input_sidecar_digests", {})
    sector_stream_rel = isd.get("sector_stream", {}).get("path")
    if not sector_stream_rel:
        raise ValueError("baseline bundle has no sector_stream path")
    sector_stream_abs = _REPO_ROOT / sector_stream_rel
    # trace_dir is the parent's parent (sector_stream/sector_stream.jsonl)
    return sector_stream_abs.parent.parent


def _regenerate_bundle(
    baseline_bundle: dict,
    geometry_provenance: dict,
) -> dict:
    """Invoke predict_cache_hit_rate with post-refinement parameters
    using the same trace_dir + manifest_pairing_index as the
    baseline bundle."""
    trace_dir = _resolve_trace_dir(baseline_bundle)
    sector_stream_path = (
        trace_dir / "sector_stream" / "sector_stream.jsonl"
    )
    mpi = baseline_bundle.get("manifest_pairing_index", {})
    device = baseline_bundle.get("device", "NVIDIA GeForce RTX 3090 Ti")
    capacity_bytes = baseline_bundle.get(
        "cache_model_capacity_bytes", 6291456,
    )
    return predict_cache_hit_rate(
        trace_dir=trace_dir,
        sector_stream_path=sector_stream_path,
        cache_model_capacity_bytes=capacity_bytes,
        manifest_pairing_index=mpi,
        device=device,
        cache_model_num_l2_slices=6,
        cache_model_num_l2_slices_source="device_properties_manifest",
        cache_model_cta_interleave_enabled=True,
        cache_model_hrd_second_layer_enabled=True,
        cache_model_geometry_provenance=geometry_provenance,
    )


def _field_audit(bundle: dict) -> dict:
    """Return a compact field-presence audit for diagnostic
    listing."""
    geom = bundle.get("cache_model_geometry_provenance")
    return {
        "cache_model_num_l2_slices": bundle.get(
            "cache_model_num_l2_slices",
        ),
        "cache_model_num_l2_slices_source": bundle.get(
            "cache_model_num_l2_slices_source",
        ),
        "cache_model_cta_interleave_enabled": bundle.get(
            "cache_model_cta_interleave_enabled",
        ),
        "cache_model_hrd_second_layer_enabled": bundle.get(
            "cache_model_hrd_second_layer_enabled",
        ),
        "cache_model_geometry_provenance_present": isinstance(geom, dict),
        "manifest_sha256": (
            geom.get("manifest_sha256") if isinstance(geom, dict) else None
        ),
        "result_kind": bundle.get("result_kind"),
        "predicted_hit_rate": bundle.get("predicted_hit_rate"),
    }


def main() -> int:
    _REFRESHED_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    manifest_sha = _compute_manifest_sha256()
    geometry_provenance = {
        **_GEOMETRY_PROVENANCE_TEMPLATE,
        "manifest_sha256": manifest_sha,
    }

    diagnostic = {
        "generator": "regenerate_refreshed_sweep.py",
        "loop_session": "2026-05-17_11-26-33",
        "round": "round-3",
        "manifest_sha256": manifest_sha,
        "geometry_provenance_template": _GEOMETRY_PROVENANCE_TEMPLATE,
        "per_cell_field_audit": {},
    }

    for bundle_name in _ADMITTED_CELL_BUNDLES:
        baseline_path = _BASELINE_SWEEP_DIR / bundle_name
        if not baseline_path.is_file():
            print(
                f"WARNING: baseline bundle missing: {baseline_path}",
                file=sys.stderr,
            )
            continue
        with baseline_path.open("r", encoding="utf-8") as fh:
            baseline = json.load(fh)

        refreshed = _regenerate_bundle(baseline, geometry_provenance)
        refreshed_path = _REFRESHED_SWEEP_DIR / bundle_name
        with refreshed_path.open("w", encoding="utf-8") as fh:
            json.dump(refreshed, fh, indent=2, sort_keys=True)

        diagnostic["per_cell_field_audit"][bundle_name] = {
            "baseline": _field_audit(baseline),
            "refreshed": _field_audit(refreshed),
        }
        print(
            f"  {bundle_name}: "
            f"baseline_predicted={baseline.get('predicted_hit_rate')} "
            f"refreshed_predicted={refreshed.get('predicted_hit_rate')}"
        )

    with _DIAGNOSTIC_PATH.open("w", encoding="utf-8") as fh:
        json.dump(diagnostic, fh, indent=2, sort_keys=True)
    print(f"Diagnostic written to {_DIAGNOSTIC_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
