"""One-shot regeneration script for the variance-precondition
fail-closed sm_86 predictor sweep.

Reads the 5 admitted-cell baseline bundles from
`data/local_labels/sweeps/sweep_20260514T165947Z_a1closure_ebe60dd/predictor_bundles/`
to extract the per-cell trace paths + manifest_pairing_index +
device tag, then invokes `cache_predictor.predict_cache_hit_rate(...)`
with the post-refinement parameters from the refreshed sweep PLUS
the new `cache_model_variance_precondition_stddev` kwarg drawn from
the committed variance trajectory artifact at
`data/sm_86_predictor_accuracy_recovery/ac6_variance_trajectory_post_ac4.json`.

The predictor's variance gate fires when the input stddev
(0.01833 on the post-refinement cohort) falls below the documented
0.10 floor; every emitted bundle has:

  result_kind="typed_unavailable"
  typed_reason="calibration_variance_precondition_failed"
  headline_validity_failed_reasons=[
      "calibration_variance_precondition_failed",
  ]
  evidence.input_cross_cell_stddev=0.01832926526212725
  evidence.documented_floor_stddev=0.10

Output bundles land at
`data/sm_86_predictor_accuracy_recovery/ac6_fail_closed_sweep/predictor_bundles/`.

This is the producer-side companion to the existing refreshed
sweep at `refreshed_sweep_round_3/`: the refreshed sweep keeps the
historical scalar predictions used by the per-sub-slice ablation
entries verbatim; this fail-closed sweep is the load-bearing
evidence that the variance-precondition gate actually fires on
real production bundles when the artifact reports
precondition_status=not_satisfied.

Dependency: protobuf Python runtime + the generated `*_pb2.py`
modules under `workloads/simbench_bw/_pb_generated/`. Run via:

  uv run --with protobuf python3 \\
    tools/sm_86_predictor_accuracy_recovery/regenerate_ac6_fail_closed_sweep.py
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
sys.path.insert(0, str(_HERE))

from cache_predictor import predict_cache_hit_rate  # noqa: E402
from variance_precondition_loader import (  # noqa: E402
    load_for_fail_closed_emission,
)


_BASELINE_SWEEP_DIR = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "sweeps"
    / "sweep_20260514T165947Z_a1closure_ebe60dd"
    / "predictor_bundles"
)
_FAIL_CLOSED_SWEEP_DIR = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "ac6_fail_closed_sweep"
    / "predictor_bundles"
)
_DIAGNOSTIC_PATH = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery"
    / "ac6_fail_closed_sweep"
    / "bundle_fail_closed_diagnostic.json"
)
_MANIFEST_PATH = (
    _REPO_ROOT / "tools" / "hrd_profiler" / "device_properties.json"
)

# Canonical admitted activation cells (same set as the refreshed
# sweep + the per-sub-slice ablation entries).
_ADMITTED_CELL_BUNDLES = [
    "sm_86__layernorm_fwd_bf16_sm80__timing_shape.json",
    "sm_86__rmsnorm_fwd_bf16_sm80__timing_shape.json",
    "sm_86__silu_fwd_bf16_sm80__correctness_shape.json",
    "sm_86__silu_fwd_bf16_sm80__timing_shape.json",
    "sm_86__softmax_safe_bf16_sm80__timing_shape.json",
]

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
    isd = baseline_bundle.get("input_sidecar_digests", {})
    sector_stream_rel = isd.get("sector_stream", {}).get("path")
    if not sector_stream_rel:
        raise ValueError("baseline bundle has no sector_stream path")
    sector_stream_abs = _REPO_ROOT / sector_stream_rel
    return sector_stream_abs.parent.parent


def _regenerate_fail_closed_bundle(
    baseline_bundle: dict,
    geometry_provenance: dict,
    current_stddev: float,
) -> dict:
    """Invoke predict_cache_hit_rate with the variance gate
    enabled. The gate fires (because the input stddev is below the
    documented floor) and the predictor short-circuits to
    typed_unavailable with the calibration_variance_precondition_failed
    typed reason."""
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
        cache_model_variance_precondition_stddev=current_stddev,
    )


def main() -> int:
    precondition = load_for_fail_closed_emission()
    _FAIL_CLOSED_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    manifest_sha = _compute_manifest_sha256()
    geometry_provenance = {
        **_GEOMETRY_PROVENANCE_TEMPLATE,
        "manifest_sha256": manifest_sha,
    }

    diagnostic = {
        "generator": "regenerate_ac6_fail_closed_sweep.py",
        "loop_session": "2026-05-17_11-26-33",
        "round": "round-11",
        "manifest_sha256": manifest_sha,
        "variance_precondition_input": {
            "current_stddev": precondition.current_stddev,
            "documented_floor_stddev": (
                precondition.documented_floor_stddev
            ),
            "precondition_status": precondition.precondition_status,
            "source_artifact": str(
                precondition.artifact_path.relative_to(_REPO_ROOT)
            ),
        },
        "per_cell_fail_closed_audit": {},
    }

    emitted = 0
    for bundle_name in _ADMITTED_CELL_BUNDLES:
        baseline_path = _BASELINE_SWEEP_DIR / bundle_name
        if not baseline_path.is_file():
            print(
                f"WARNING: baseline bundle missing: {baseline_path}",
                file=sys.stderr,
            )
            continue
        with baseline_path.open("r", encoding="utf-8") as fh:
            baseline_bundle = json.load(fh)
        bundle = _regenerate_fail_closed_bundle(
            baseline_bundle,
            geometry_provenance,
            precondition.current_stddev,
        )
        out_path = _FAIL_CLOSED_SWEEP_DIR / bundle_name
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, sort_keys=True)
            fh.write("\n")
        diagnostic["per_cell_fail_closed_audit"][bundle_name] = {
            "result_kind": bundle.get("result_kind"),
            "typed_reason": bundle.get("typed_reason"),
            "headline_validity_failed_reasons": bundle.get(
                "headline_validity_failed_reasons",
            ),
            "evidence_input_cross_cell_stddev": (
                bundle.get("evidence", {}).get("input_cross_cell_stddev")
            ),
            "evidence_documented_floor_stddev": (
                bundle.get("evidence", {}).get("documented_floor_stddev")
            ),
        }
        emitted += 1
        print(
            f"emitted fail-closed bundle: {bundle_name} "
            f"(typed_reason={bundle.get('typed_reason')!r})",
        )

    with _DIAGNOSTIC_PATH.open("w", encoding="utf-8") as fh:
        json.dump(diagnostic, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"\nemitted {emitted} fail-closed bundle(s); "
        f"diagnostic at {_DIAGNOSTIC_PATH.relative_to(_REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
