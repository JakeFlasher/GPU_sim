"""Build the candidate-feature catalog over the admitted
cohort and emit the artifact at
``data/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.json``.

The catalog is the load-bearing input for the downstream
calibration-fitting plan's missing-feature work: it
enumerates the 5 named candidate covariates the plan should
evaluate, binds per-cell observations against the
variance-trajectory artifact's signed_residual values, and
emits a recommendation per feature.

The 5 named covariates (closed feature_name enum on the
schema) are:

  - per_warp_occupancy:  derived from block volume + warpSize
                         (kernel.proto block_dim).
  - blockDim:            from kernel.proto block_dim
                         (carried in the
                         launch_parameters_sidecar).
  - gridDim:             from kernel.proto grid_dim.
  - dvfs_state:          from nvml_covariate
                         (label-provenance-only; NEVER a
                         predictor input under the
                         no-DVFS-as-predictor-input
                         directive).
  - mshr_pressure_proxy: from the historical CUPTI
                         Activity-derived heuristic; not
                         collected on the 9 admitted cells;
                         tagged `not_collected_for_cell`.

Inputs:
  - ``data/local_labels/label_admitted.json`` (the 9-cell
    admitted index).
  - ``data/sm_86_predictor_accuracy_recovery_v2/ac5_variance_trajectory_post_corpus_expansion.json``
    (per-cell signed_residual values).
  - The per-cell launch_parameters_sidecars produced by
    ``tools/local_labels/extend_launch_parameters.py``
    (carrying grid_dim / block_dim / derived geometry).

Outputs:
  - The catalog at
    ``data/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.json``
    validating against
    ``tools/sm_86_predictor_accuracy_recovery_v2/ac6_missing_feature_candidates.schema.json``.

Per-feature recommendations follow this rule book:

  - The silu_fwd_bf16_sm80 correctness vs timing canonical
    pair carries the gridDim discriminator: both shape
    points predict identically under the current predictor
    (~0.9375) but truth differs by ~0.35 absolute; gridDim
    differs by ~256x between them. A feature whose paired-
    delta over the silu pair is materially nonzero is a
    strong candidate for ``predictor_input``. A feature
    whose paired-delta is zero is either an irrelevant
    covariate or a constant within the pair; recommendation
    falls to ``inconclusive`` under the small-N escape value.

  - dvfs_state is ALWAYS recommended ``discard`` (rationale:
    the no-DVFS-as-predictor-input directive). The
    downstream plan may re-evaluate only via an explicit
    directive reopen with portability evidence.

  - mshr_pressure_proxy is ALWAYS recommended
    ``inconclusive`` on the 9-cell cohort: the auxiliary
    CUPTI-derived sidecar is not collected for these cells.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SCHEMA_PATH = _HERE / "ac6_missing_feature_candidates.schema.json"
_DEFAULT_LABEL_ADMITTED = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
_DEFAULT_AC5_ARTIFACT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "ac5_variance_trajectory_post_corpus_expansion.json"
)
_DEFAULT_SIDECAR_DIR = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "launch_parameters_sidecars"
    / "sm_86"
)
_DEFAULT_OUT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "ac6_missing_feature_candidates.json"
)
_DEFAULT_ARCH_DIR = "rtx_3090_ti/sm_86"

_FEATURE_NAMES: tuple[str, ...] = (
    "per_warp_occupancy",
    "blockDim",
    "gridDim",
    "dvfs_state",
    "mshr_pressure_proxy",
)

_FEATURE_METADATA: dict[str, dict] = {
    "per_warp_occupancy": {
        "sidecar_source_file": (
            "tools/nvbit_tracer/traces_enhanced/dynamic_trace/"
            "kernel.proto"
        ),
        "schema_owner": "kernel_proto",
        "expected_physical_sign": "inconclusive",
    },
    "blockDim": {
        "sidecar_source_file": (
            "tools/nvbit_tracer/traces_enhanced/dynamic_trace/"
            "kernel.proto"
        ),
        "schema_owner": "kernel_proto",
        "expected_physical_sign": "inconclusive",
    },
    "gridDim": {
        "sidecar_source_file": (
            "tools/nvbit_tracer/traces_enhanced/dynamic_trace/"
            "kernel.proto"
        ),
        "schema_owner": "kernel_proto",
        "expected_physical_sign": "positive",
    },
    "dvfs_state": {
        "sidecar_source_file": (
            "workloads/simbench_bw/schemas/nvml_covariate.schema.json"
        ),
        "schema_owner": "nvml",
        "expected_physical_sign": "null_no_directional_hypothesis",
    },
    "mshr_pressure_proxy": {
        "sidecar_source_file": (
            "workloads/simbench_bw/schemas/historical/"
            "cupti_activity_sidecar.v1.schema.json"
        ),
        "schema_owner": "cupti_activity",
        "expected_physical_sign": "positive",
    },
}

_SILU_PAIR_KEYS: tuple[tuple[str, str], tuple[str, str]] = (
    ("silu_fwd_bf16_sm80", "correctness_shape"),
    ("silu_fwd_bf16_sm80", "timing_shape"),
)
_LAYERNORM_CONTRAST_KEY: tuple[str, str] = (
    "layernorm_fwd_bf16_sm80", "timing_shape",
)


def _load_sidecar(
    sidecar_dir: Path, slug: str, shape_point: str,
) -> dict:
    path = sidecar_dir / (
        f"launch_parameters_sidecar_{slug}__{shape_point}.json"
    )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _per_cell_feature_value(
    feature_name: str, sidecar: dict,
) -> tuple[float | None, str]:
    """Return (observed_value, availability_at_prediction_time)
    for a single feature on a single cell's sidecar."""
    if feature_name == "per_warp_occupancy":
        return (
            float(sidecar["derived"]["warps_per_cta"]),
            "prediction_time_available",
        )
    if feature_name == "blockDim":
        # Use threads-per-CTA (block volume) as the scalar
        # blockDim summary; the catalog records the scalar
        # for paired-delta arithmetic. Component fields stay
        # available on the sidecar for the downstream plan.
        return (
            float(sidecar["derived"]["threads_per_cta"]),
            "prediction_time_available",
        )
    if feature_name == "gridDim":
        return (
            float(sidecar["derived"]["grid_volume"]),
            "prediction_time_available",
        )
    if feature_name == "dvfs_state":
        # Per the no-DVFS-as-predictor-input directive: DVFS
        # is always recorded as label-provenance-only; the
        # observed_value is null (the catalog tracks the
        # availability tag, not a numeric mean).
        return (None, "label-provenance-only")
    if feature_name == "mshr_pressure_proxy":
        # The historical CUPTI Activity-derived MSHR-pressure proxy
        # availability follows the launch_parameters_sidecar's
        # checked-absence path: prediction_time_available when
        # the archived sidecar's CUPTI join landed cleanly,
        # not_collected_for_cell after the checked-absence
        # path ran with zero matching candidates.
        avail = str(sidecar.get(
            "mshr_pressure_proxy_availability",
            "not_collected_for_cell",
        ))
        return (None, avail)
    raise ValueError(
        f"build_missing_feature_candidates: unknown feature "
        f"{feature_name}"
    )


def _paired_delta(
    feature_name: str, sidecars_by_key: dict,
) -> tuple[float | None, int]:
    """Compute the silu correctness/timing paired-delta for
    a feature. Returns (delta, sample_size).

    Sample size is the number of cells in the canonical
    pair for which the feature has a non-null observed
    value. For dvfs / mshr the pair has sample_size=0 so
    the catalog records computed_effect=null."""
    corr_key, time_key = _SILU_PAIR_KEYS
    corr_sidecar = sidecars_by_key.get(corr_key)
    time_sidecar = sidecars_by_key.get(time_key)
    if corr_sidecar is None or time_sidecar is None:
        return (None, 0)
    corr_value, corr_avail = _per_cell_feature_value(
        feature_name, corr_sidecar,
    )
    time_value, time_avail = _per_cell_feature_value(
        feature_name, time_sidecar,
    )
    if corr_value is None or time_value is None:
        return (None, 0)
    return (time_value - corr_value, 2)


def _layernorm_contrast(
    feature_name: str, sidecars_by_key: dict,
) -> tuple[float | None, int]:
    """The layernorm contrast: per-feature scalar reading on
    the layernorm timing_shape cell, contrasted against the
    silu_fwd_bf16_sm80 timing_shape cell. Returns
    (layernorm_value - silu_timing_value, sample_size)."""
    layer_key = _LAYERNORM_CONTRAST_KEY
    silu_timing_key = _SILU_PAIR_KEYS[1]
    layer_sidecar = sidecars_by_key.get(layer_key)
    silu_sidecar = sidecars_by_key.get(silu_timing_key)
    if layer_sidecar is None or silu_sidecar is None:
        return (None, 0)
    layer_value, layer_avail = _per_cell_feature_value(
        feature_name, layer_sidecar,
    )
    silu_value, silu_avail = _per_cell_feature_value(
        feature_name, silu_sidecar,
    )
    if layer_value is None or silu_value is None:
        return (None, 0)
    return (layer_value - silu_value, 2)


def _recommendation_for(
    feature_name: str, silu_pair_effect: float | None,
) -> tuple[str, str]:
    """Map (feature_name, silu_pair_effect) to a closed-enum
    recommendation + justification per the catalog's
    rule book."""
    if feature_name == "dvfs_state":
        return (
            "discard",
            "The no-DVFS-as-predictor-input directive "
            "forbids dvfs_state as a predictor input in "
            "both this plan and the downstream "
            "calibration-fitting plan. Lifting the "
            "prohibition requires an explicit directive "
            "reopen with portability evidence (cloud vs "
            "local; sm_86 vs sm_120).",
        )
    if feature_name == "mshr_pressure_proxy":
        return (
            "inconclusive",
            "The historical CUPTI Activity-derived MSHR-pressure proxy "
            "has no per-cell scalar observation under the "
            "retained offline JOIN: the checked-absence path returned "
            "not_collected_for_cell on every admitted cell "
            "(zero archived auxiliary CUPTI Activity sidecars under "
            "the cells' trace directories); sample size on "
            "the cohort is 0 so the small-N escape value "
            "applies.",
        )
    if feature_name == "gridDim":
        if silu_pair_effect is None or silu_pair_effect == 0.0:
            return (
                "inconclusive",
                "The silu correctness/timing paired-delta is "
                "zero or unavailable; no missing-feature "
                "signal on the canonical pair.",
            )
        return (
            "predictor_input",
            "The silu correctness/timing paired-delta on "
            f"gridDim is {silu_pair_effect:+.0f} (grid "
            "volume changes between the two shape points). "
            "Both shape points predict identically under "
            "the current predictor but truth differs by "
            "~0.35 absolute, so gridDim is a strong "
            "candidate for inclusion as a predictor input "
            "in the downstream calibration-fitting plan.",
        )
    # per_warp_occupancy / blockDim: small-N delta zero on
    # the silu pair (same block shape between correctness
    # and timing); recommendation = inconclusive.
    if silu_pair_effect is None or silu_pair_effect == 0.0:
        return (
            "inconclusive",
            f"The silu correctness/timing paired-delta on "
            f"{feature_name} is "
            f"{0.0 if silu_pair_effect is None else silu_pair_effect:+.2f}; "
            "the canonical pair holds this feature "
            "constant by design so the small-N escape "
            "value applies.",
        )
    return (
        "covariate",
        f"The silu correctness/timing paired-delta on "
        f"{feature_name} is {silu_pair_effect:+.2f}. "
        "Recommend as a covariate (downstream plan should "
        "carry the feature but not necessarily as a model "
        "input).",
    )


def build_catalog(
    *,
    label_admitted_path: Path,
    ac5_artifact_path: Path,
    sidecar_dir: Path,
    repo_root: Path = _REPO_ROOT,
    partial_closure_override: dict | None = None,
) -> dict:
    """Build the catalog dict. Pure function: takes the
    inputs explicitly so tests can inject synthetic
    surfaces.

    Reads the variance-trajectory artifact for per-cell signed_residual
    binding (BL-20260511 post-hoc verifier must semantic-bind
    not just hash); raises ValueError if any admitted cell
    is missing from the variance-trajectory artifact (the catalog cannot
    bind to a missing residual)."""
    with label_admitted_path.open(encoding="utf-8") as fh:
        admitted = json.load(fh)
    with ac5_artifact_path.open(encoding="utf-8") as fh:
        ac5 = json.load(fh)

    sm_86_cells = sorted(
        (
            c for c in admitted.get("cells", [])
            if isinstance(c, dict) and c.get("arch") == "sm_86"
        ),
        key=lambda c: (str(c.get("slug")), str(c.get("shape_point"))),
    )
    if not sm_86_cells:
        raise ValueError(
            "build_missing_feature_candidates: zero sm_86 "
            f"admitted cells in {label_admitted_path}"
        )

    ac5_residuals: dict[tuple[str, str], float] = {
        (str(r["slug"]), str(r["shape_point"])): float(
            r["signed_residual"]
        )
        for r in ac5["cells"]
    }
    if partial_closure_override is None:
        partial_closure_override = dict(
            ac5.get("partial_closure_override", {})
        )

    sidecars_by_key: dict[tuple[str, str], dict] = {}
    cells_rows: list[dict] = []
    for cell in sm_86_cells:
        slug = str(cell["slug"])
        shape_point = str(cell["shape_point"])
        key = (slug, shape_point)
        if key not in ac5_residuals:
            raise ValueError(
                f"build_missing_feature_candidates: cell "
                f"{slug}/{shape_point} not present in the "
                f"variance-trajectory artifact at "
                f"{ac5_artifact_path}; cannot bind signed_"
                f"residual"
            )
        sidecar = _load_sidecar(sidecar_dir, slug, shape_point)
        sidecars_by_key[key] = sidecar

    # Build per-cell observations after all sidecars are
    # loaded so the paired-delta sample-size accounting is
    # consistent for every cell row.
    for cell in sm_86_cells:
        slug = str(cell["slug"])
        shape_point = str(cell["shape_point"])
        key = (slug, shape_point)
        sidecar = sidecars_by_key[key]
        observations: list[dict] = []
        for feature_name in _FEATURE_NAMES:
            observed_value, availability = _per_cell_feature_value(
                feature_name, sidecar,
            )
            silu_effect, silu_n = _paired_delta(
                feature_name, sidecars_by_key,
            )
            observation: dict = {
                "feature_name": feature_name,
                "observed_value": observed_value,
                "availability_at_prediction_time": availability,
                "correlation_method": (
                    "paired_delta" if silu_n >= 2 else "inconclusive_small_n"
                ),
                "sample_size": silu_n,
                "computed_effect": silu_effect,
            }
            if availability == "label-provenance-only":
                observation["plan_2_prohibition_status"] = (
                    "forbidden_requires_explicit_reopen"
                )
            observations.append(observation)
        cells_rows.append({
            "arch": "sm_86",
            "slug": slug,
            "shape_point": shape_point,
            "signed_residual": ac5_residuals[key],
            "per_feature_observations": observations,
        })

    features_rows: list[dict] = []
    recommendations_rows: list[dict] = []
    for feature_name in _FEATURE_NAMES:
        meta = _FEATURE_METADATA[feature_name]
        features_rows.append({
            "feature_name": feature_name,
            "sidecar_source_file": meta["sidecar_source_file"],
            "schema_owner": meta["schema_owner"],
            "expected_physical_sign": (
                meta["expected_physical_sign"]
            ),
        })
        silu_effect, _silu_n = _paired_delta(
            feature_name, sidecars_by_key,
        )
        recommendation, justification = _recommendation_for(
            feature_name, silu_effect,
        )
        recommendations_rows.append({
            "feature_name": feature_name,
            "recommendation_for_plan_2": recommendation,
            "justification": justification,
        })

    catalog: dict = {
        "schema_version": 1,
        "generator": (
            "tools/sm_86_predictor_accuracy_recovery_v2/"
            "build_missing_feature_candidates.py"
        ),
        "cohort_admitted_count": len(cells_rows),
        "features": features_rows,
        "cells": cells_rows,
        "recommendations": recommendations_rows,
        "partial_closure_override": partial_closure_override,
    }
    return catalog


def validate_catalog(catalog: dict) -> None:
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(instance=catalog, schema=schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the candidate-feature catalog over the "
            "admitted cohort and emit the artifact."
        ),
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=_DEFAULT_LABEL_ADMITTED,
    )
    parser.add_argument(
        "--ac5-artifact",
        type=Path,
        default=_DEFAULT_AC5_ARTIFACT,
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=_DEFAULT_SIDECAR_DIR,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
    )
    args = parser.parse_args(argv)

    try:
        catalog = build_catalog(
            label_admitted_path=args.label_admitted,
            ac5_artifact_path=args.ac5_artifact,
            sidecar_dir=args.sidecar_dir,
        )
        validate_catalog(catalog)
    except (ValueError, jsonschema.ValidationError) as exc:
        sys.stderr.write(
            f"build_missing_feature_candidates: {exc}\n"
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, sort_keys=True)
        fh.write("\n")
    sys.stderr.write(
        f"build_missing_feature_candidates: wrote {args.out} "
        f"(cohort_admitted_count={catalog['cohort_admitted_count']}, "
        f"features={len(catalog['features'])}, "
        f"recommendations={len(catalog['recommendations'])})\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
