#!/usr/bin/env python3
"""Cycle prediction bundle for the scoped long-duration timing cohort.

The predictor is deliberately small and auditable: it builds a static roofline
prior from launch geometry, static SASS work, and sourced GA102 peak constants,
then fits a ridge-regularized residual on top of that prior. The emitted
per-cell inference feature vector contains only deploy-legal fields. Timing
labels are consumed only while fitting the committed model and by the separate
accuracy runner.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_cohort"
    / "ac4_long_duration_cohort_sm_86.json"
)
DEFAULT_STATIC_WORK = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "static_work"
    / "gemm_hmma_bf16_sm80_sm_86.json"
)
DEFAULT_STATIC_FEATURES = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "static_sass_features"
    / "gemm_hmma_bf16_sm80_sm_86.json"
)
DEFAULT_PEAKS = REPO_ROOT / "tools" / "hrd_profiler" / "ga102_peak_constants.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "timing_cycle_predictions_sm_86.json"
)

# v4 demotion: the v3 same-slug scaled-neighbor leave-one-out arm is
# DEMOTED to a diagnostic-only channel (mirroring the cache-side
# correction-factor demotion). The deployable validation score is the
# ridge-on-roofline prediction with NO same-slug label substitution;
# the same-slug values persist only in the clearly-labeled per-cell
# diagnostics block.
MODEL_VERSION = "timing_cycle_ridge_v4_same_slug_demoted_to_diagnostic"
DEMOTED_MODEL_VERSIONS = frozenset(
    {"timing_cycle_ridge_v3_same_slug_local_residual"}
)
DEPLOYABLE_CHANNEL = (
    "deployable_ridge_on_roofline_no_same_slug_substitution"
)
DIAGNOSTIC_SAME_SLUG_CHANNEL = (
    "diagnostic_same_slug_scaled_neighbor_demoted"
)
# Honest naming: leave-one-out at the current slug diversity is NOT
# slug-disjoint scoring (every fold's residual fit still sees same-slug
# cells). The limitation is typed on the bundle; lifting it requires
# the timing-cohort slug expansion, not a substitution channel.
COHORT_SLUG_DIVERSITY_TYPED_REASON = (
    "timing_cohort_slug_diversity_insufficient_for_slug_disjoint_scoring"
)
RIDGE_LAMBDA = 10.0
TENSOR_ROOFLINE_EFFICIENCY_PRIOR = 0.066
CACHE_HEAD_PREDICTION_PRODUCER = "timing_per_cell_cache_head_prediction_v2"
CACHE_HEAD_PREDICTION_METHOD = "per_cell_l1_l2_monotonic_calibration_head_v5"
CACHE_HEAD_SCORER_NAME = "PerBucketCalibratorEnsemble.predict"
COMPUTE_PRIOR_SOURCE = (
    "static_work_dynamic_flop_count_x_ga102_tensor_peak_x_explicit_efficiency_prior"
)
FEATURE_NAMES = (
    "bias",
    "dynamic_flops_giga",
    "mn_tile_meg",
    "mk_tile_meg",
    "nk_tile_meg",
    "cta_count_10k",
    "m_norm",
    "n_norm",
    "k_norm",
    "structural_bytes_giga",
    "nvbit_total_32b_sectors_giga",
    "predicted_l1_hit_rate",
    "predicted_l2_hit_rate",
    "active_warps_per_sm_norm",
)
FORBIDDEN_FEATURE_FIELDS = frozenset(
    {
        "gpc__cycles_elapsed.max",
        "gpu__time_duration.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__inst_executed.avg.per_cycle_elapsed",
        "occupancy_achieved_warps_pct",
        "ipc_inst_per_cycle",
        "clock64",
    }
)


class TimingPredictionError(ValueError):
    """Raised when the predictor input or model fit is not admissible."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _source_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def assert_feature_vector_deploy_legal(feature_vector: dict[str, float]) -> None:
    """Reject per-cell inference features that name label-only counters."""
    leaked = sorted(FORBIDDEN_FEATURE_FIELDS.intersection(feature_vector))
    if leaked:
        raise TimingPredictionError(
            f"deploy feature vector contains forbidden label-only fields: {leaked}"
        )


def dynamic_flops(cell: dict[str, Any]) -> float:
    geom = cell["launch_geometry"]
    if "dynamic_flops" in geom:
        return float(geom["dynamic_flops"])
    return float(2 * int(geom["m"]) * int(geom["n"]) * int(geom["k"]))


def structural_bytes(cell: dict[str, Any]) -> float:
    geom = cell["launch_geometry"]
    if "structural_bytes" in geom:
        return float(geom["structural_bytes"])
    m = int(geom["m"])
    n = int(geom["n"])
    k = int(geom["k"])
    # BF16 GEMM order-of-magnitude tensor traffic: A + B reads and C write.
    return float(2 * (m * k + k * n + m * n))


def _shape_axis(geom: dict[str, Any], name: str, fallback: float = 1.0) -> float:
    value = geom.get(name, fallback)
    return max(0.0, float(value))


def _artifact_path(cell: dict[str, Any], name: str) -> Path:
    ref = cell["feature_artifacts"][name]
    return REPO_ROOT / ref["path"]


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sector_summary(cell: dict[str, Any]) -> dict[str, Any]:
    return _load_json(_artifact_path(cell, "nvbit_sector_summary"))


def load_cache_hit_prediction(cell: dict[str, Any]) -> dict[str, Any]:
    ref = cell["feature_artifacts"]["cache_hit_predictions"]
    path = REPO_ROOT / ref["path"]
    if _sha256_file(path) != ref["sha256"]:
        raise TimingPredictionError(
            f"cache-hit prediction artifact digest mismatch for {cell['shape_point']}"
        )
    bundle = _load_json(path)
    if bundle.get("producer") != CACHE_HEAD_PREDICTION_PRODUCER:
        raise TimingPredictionError(
            "cache-hit prediction bundle has unsupported producer "
            f"{bundle.get('producer')!r}"
        )
    for row in bundle.get("cells", []):
        if (
            row.get("slug") == cell["slug"]
            and row.get("shape_point") == cell["shape_point"]
            and row.get("arch") == cell["arch"]
            and row.get("binary_sha256") == cell["binary_sha256"]
            and row.get("kernel_signature_sha256")
            == cell["kernel_signature_sha256"]
            and row.get("source_commit") == cell["source_commit"]
        ):
            if row.get("result_kind") != "scalar":
                raise TimingPredictionError(
                    f"cache-hit prediction unavailable for {cell['shape_point']}: "
                    f"{row.get('typed_unavailable_reason')}"
                )
            _assert_cache_prediction_head_scored(row)
            return row
    raise TimingPredictionError(
        f"missing cache-hit prediction row for {cell['shape_point']}"
    )


def _assert_cache_prediction_head_scored(row: dict[str, Any]) -> None:
    provenance = row.get("head_provenance", {})
    if provenance.get("producer") != CACHE_HEAD_PREDICTION_PRODUCER:
        raise TimingPredictionError(
            "cache-hit prediction row has unsupported producer "
            f"{provenance.get('producer')!r}"
        )
    if provenance.get("prediction_method") != CACHE_HEAD_PREDICTION_METHOD:
        raise TimingPredictionError(
            "cache-hit prediction row has unsupported method "
            f"{provenance.get('prediction_method')!r}"
        )
    if (
        provenance.get("prediction_method")
        == "per_cell_observed_reuse_distance_hit_fraction"
    ):
        raise TimingPredictionError("cache-hit prediction used observed reuse shortcut")
    for prefix in ("l1", "l2"):
        layer = provenance.get(f"{prefix}_prediction")
        if not isinstance(layer, dict):
            raise TimingPredictionError(f"missing {prefix} head-scoring provenance")
        if layer.get("result_kind") != "scalar":
            raise TimingPredictionError(
                f"{prefix} cache head prediction unavailable: "
                f"{layer.get('typed_unavailable_reason')}"
            )
        if layer.get("scorer_name") != CACHE_HEAD_SCORER_NAME:
            raise TimingPredictionError(
                f"{prefix} cache head used unsupported scorer "
                f"{layer.get('scorer_name')!r}"
            )
        predicted = layer.get("predicted_sector_hit_rate")
        row_value = row.get(f"{prefix}_sector_hit_rate")
        if predicted is None or row_value is None:
            raise TimingPredictionError(f"{prefix} cache head scalar is null")
        if not math.isclose(
            float(predicted),
            float(row_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise TimingPredictionError(
                f"{prefix} cache head scalar does not match row value"
            )
        if not isinstance(layer.get("input_features"), dict):
            raise TimingPredictionError(f"{prefix} cache head features are missing")


def feature_vector(
    cell: dict[str, Any],
    sector_summary: dict[str, Any],
    cache_prediction: dict[str, Any],
) -> dict[str, float]:
    geom = cell["launch_geometry"]
    m = _shape_axis(geom, "m")
    n = _shape_axis(geom, "n")
    k = _shape_axis(geom, "k")
    cta_count = int(geom["grid_x"]) * int(geom["grid_y"]) * int(geom["grid_z"])
    sectors = sector_summary["sector_counts"]
    occupancy = sector_summary["theoretical_occupancy"]
    features = {
        "bias": 1.0,
        "dynamic_flops_giga": dynamic_flops(cell) / 1.0e9,
        "mn_tile_meg": (m * n) / 1.0e6,
        "mk_tile_meg": (m * k) / 1.0e6,
        "nk_tile_meg": (n * k) / 1.0e6,
        "cta_count_10k": cta_count / 1.0e4,
        "m_norm": m / 4096.0,
        "n_norm": n / 4096.0,
        "k_norm": k / 4096.0,
        "structural_bytes_giga": structural_bytes(cell) / 1.0e9,
        "nvbit_total_32b_sectors_giga": sectors["total_32b_sectors"] / 1.0e9,
        "predicted_l1_hit_rate": float(cache_prediction["l1_sector_hit_rate"]),
        "predicted_l2_hit_rate": float(cache_prediction["l2_sector_hit_rate"]),
        "active_warps_per_sm_norm": float(occupancy["active_warps_per_sm"]) / 64.0,
    }
    assert_feature_vector_deploy_legal(features)
    return features


def _feature_row(features: dict[str, float]) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    if any(len(row) != n for row in matrix):
        raise TimingPredictionError("linear system matrix must be square")
    aug = [list(matrix[i]) + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-12:
            raise TimingPredictionError("residual model fit is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def fit_ridge_residual(
    rows: Sequence[Sequence[float]],
    residual_targets: Sequence[float],
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> list[float]:
    if len(rows) != len(residual_targets):
        raise TimingPredictionError("feature rows and targets have different lengths")
    if not rows:
        raise TimingPredictionError("cannot fit residual model with no rows")
    width = len(rows[0])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for row, target in zip(rows, residual_targets):
        if len(row) != width:
            raise TimingPredictionError("inconsistent feature width")
        for i, left in enumerate(row):
            xty[i] += left * float(target)
            for j, right in enumerate(row):
                xtx[i][j] += left * right
    for i in range(width):
        xtx[i][i] += ridge_lambda
    return _solve_linear_system(xtx, xty)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _peak_value(peaks: dict[str, Any], name: str) -> float:
    return float(peaks["peaks"][name]["value"])


def _true_cycles(cell: dict[str, Any]) -> float:
    return float(cell["labels"]["gpc__cycles_elapsed.max"]["value"])


def _memory_cycles(
    sector_summary: dict[str, Any],
    cache_prediction: dict[str, Any],
    *,
    dram_bytes_per_cycle: float,
) -> float:
    sectors = sector_summary["sector_counts"]
    occupancy = sector_summary["theoretical_occupancy"]
    total_sectors = float(sectors["total_32b_sectors"])
    total_bytes = float(sectors["estimated_bytes"])
    l1_miss = 1.0 - float(cache_prediction["l1_sector_hit_rate"])
    l2_miss = 1.0 - float(cache_prediction["l2_sector_hit_rate"])
    active_blocks = max(1.0, float(occupancy["active_blocks_per_sm"]))
    active_warps = max(1.0, float(occupancy["active_warps_per_sm"]))
    bandwidth_cycles = (total_bytes * l2_miss) / (
        dram_bytes_per_cycle * active_blocks
    )
    latency_cycles = (
        total_sectors * (20.0 * l1_miss + 220.0 * l2_miss)
    ) / (active_warps * active_blocks)
    return max(bandwidth_cycles, latency_cycles)


def _join_key(cell: dict[str, Any]) -> dict[str, str]:
    return {
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "arch": cell["arch"],
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "metric_target": "cycles",
    }


def _fit_residual_model(
    *,
    cells: list[dict[str, Any]],
    sector_summaries: list[dict[str, Any]],
    cache_predictions: list[dict[str, Any]],
    dram_bytes_per_cycle: float,
    tensor_peak_flops_per_cycle: float,
) -> dict[str, Any]:
    feature_rows: list[list[float]] = []
    analytical_priors: list[float] = []
    true_cycles: list[float] = []
    effective_flops_per_cycle = (
        tensor_peak_flops_per_cycle * TENSOR_ROOFLINE_EFFICIENCY_PRIOR
    )
    for cell, sector, cache_pred in zip(cells, sector_summaries, cache_predictions):
        feats = feature_vector(cell, sector, cache_pred)
        feature_rows.append(_feature_row(feats))
        compute_cycles = dynamic_flops(cell) / effective_flops_per_cycle
        memory_cycles = _memory_cycles(
            sector,
            cache_pred,
            dram_bytes_per_cycle=dram_bytes_per_cycle,
        )
        analytical_priors.append(max(compute_cycles, memory_cycles))
        true_cycles.append(_true_cycles(cell))
    residual_targets = [
        truth - prior for truth, prior in zip(true_cycles, analytical_priors)
    ]
    return {
        "effective_flops_per_cycle": effective_flops_per_cycle,
        "coefficients": fit_ridge_residual(feature_rows, residual_targets),
    }


def _predict_with_model(
    *,
    cell: dict[str, Any],
    sector_summary: dict[str, Any],
    cache_prediction: dict[str, Any],
    model: dict[str, Any],
    dram_bytes_per_cycle: float,
) -> dict[str, float]:
    feats = feature_vector(cell, sector_summary, cache_prediction)
    row = _feature_row(feats)
    compute_cycles = dynamic_flops(cell) / model["effective_flops_per_cycle"]
    memory_cycles = _memory_cycles(
        sector_summary,
        cache_prediction,
        dram_bytes_per_cycle=dram_bytes_per_cycle,
    )
    analytical_prior = max(compute_cycles, memory_cycles)
    residual_cycles = _dot(row, model["coefficients"])
    predicted_cycles = max(1.0, analytical_prior + residual_cycles)
    return {
        "pedestal_cycles": 0.0,
        "compute_cycles": compute_cycles,
        "memory_cycles": memory_cycles,
        "analytical_prior_cycles": analytical_prior,
        "residual_cycles": residual_cycles,
        "predicted_cycles": predicted_cycles,
    }


def _same_slug_scaled_neighbor_prediction(
    *,
    holdout_cell: dict[str, Any],
    train_cells: Sequence[dict[str, Any]],
    analytical_prior: float,
) -> dict[str, Any] | None:
    """DIAGNOSTIC-ONLY leave-one-out local residual using labels from
    same-slug training cells.

    DEMOTED channel: the returned values must never replace the
    deployable ridge-on-roofline prediction — they substitute the
    nearest same-slug cell's TRUE label (work-scaled), which is the
    truth-fitted leakage pattern the project demotes. The caller
    records the result only under the per-cell diagnostics block."""
    same_slug = [
        cell for cell in train_cells if cell.get("slug") == holdout_cell.get("slug")
    ]
    if not same_slug:
        return None
    holdout_work = max(1.0, dynamic_flops(holdout_cell))
    nearest = min(
        same_slug,
        key=lambda cell: (
            abs(math.log(holdout_work / max(1.0, dynamic_flops(cell)))),
            cell["shape_point"],
        ),
    )
    nearest_work = max(1.0, dynamic_flops(nearest))
    predicted_cycles = max(1.0, _true_cycles(nearest) * holdout_work / nearest_work)
    return {
        "channel": DIAGNOSTIC_SAME_SLUG_CHANNEL,
        "residual_cycles": predicted_cycles - analytical_prior,
        "predicted_cycles": predicted_cycles,
        "neighbor_slug": nearest["slug"],
        "neighbor_shape_point": nearest["shape_point"],
        "neighbor_work_ratio": holdout_work / nearest_work,
    }


def build_prediction_bundle(
    *,
    cohort_path: Path = DEFAULT_COHORT,
    static_work_path: Path = DEFAULT_STATIC_WORK,
    static_features_path: Path = DEFAULT_STATIC_FEATURES,
    peak_constants_path: Path = DEFAULT_PEAKS,
    source_commit: str | None = None,
) -> dict[str, Any]:
    cohort = _load_json(cohort_path)
    peaks = _load_json(peak_constants_path)
    cells = list(cohort.get("cells", []))
    if not cells:
        raise TimingPredictionError("long-duration cohort has no cells")

    boost_clock_hz = _peak_value(peaks, "boost_sm_clock_mhz") * 1.0e6
    tensor_peak_flops_per_cycle = (
        _peak_value(peaks, "fp16_tensor_dense_flops_per_s") / boost_clock_hz
    )
    dram_bytes_per_cycle = _peak_value(peaks, "dram_bytes_per_s") / boost_clock_hz
    sector_summaries = [load_sector_summary(cell) for cell in cells]
    cache_predictions = [load_cache_hit_prediction(cell) for cell in cells]
    full_fit_model = _fit_residual_model(
        cells=cells,
        sector_summaries=sector_summaries,
        cache_predictions=cache_predictions,
        dram_bytes_per_cycle=dram_bytes_per_cycle,
        tensor_peak_flops_per_cycle=tensor_peak_flops_per_cycle,
    )
    effective_flops_per_cycle = full_fit_model["effective_flops_per_cycle"]

    out_cells: list[dict[str, Any]] = []
    for holdout_idx, (cell, sector, cache_pred) in enumerate(
        zip(cells, sector_summaries, cache_predictions)
    ):
        train_cells = [c for idx, c in enumerate(cells) if idx != holdout_idx]
        train_sectors = [s for idx, s in enumerate(sector_summaries) if idx != holdout_idx]
        train_cache_predictions = [
            p for idx, p in enumerate(cache_predictions) if idx != holdout_idx
        ]
        validation_model = _fit_residual_model(
            cells=train_cells,
            sector_summaries=train_sectors,
            cache_predictions=train_cache_predictions,
            dram_bytes_per_cycle=dram_bytes_per_cycle,
            tensor_peak_flops_per_cycle=tensor_peak_flops_per_cycle,
        )
        feats = feature_vector(cell, sector, cache_pred)
        pred = _predict_with_model(
            cell=cell,
            sector_summary=sector,
            cache_prediction=cache_pred,
            model=validation_model,
            dram_bytes_per_cycle=dram_bytes_per_cycle,
        )
        # DEMOTED diagnostic channel: the same-slug scaled-neighbor
        # value is recorded for diagnosis only and never replaces the
        # deployable ridge-on-roofline prediction (the v3 behavior
        # overwrote pred[...] here — that substitution leaked truth
        # labels into the deployable validation headline).
        truth = _true_cycles(cell)
        local_pred = _same_slug_scaled_neighbor_prediction(
            holdout_cell=cell,
            train_cells=train_cells,
            analytical_prior=pred["analytical_prior_cycles"],
        )
        same_slug_diagnostic: dict[str, Any] | None = None
        if local_pred is not None:
            same_slug_diagnostic = dict(local_pred)
            same_slug_diagnostic["validation_abs_log_error"] = abs(
                math.log(local_pred["predicted_cycles"] / truth)
            )
        predicted_cycles = pred["predicted_cycles"]
        static_work = _load_json(_artifact_path(cell, "static_work"))
        static_features = _load_json(_artifact_path(cell, "static_sass_features"))
        if static_work["binary_sha256"] != cell["binary_sha256"]:
            raise TimingPredictionError(
                f"static work binary mismatch for {cell['shape_point']}"
            )
        if static_features["binary_sha256"] != cell["binary_sha256"]:
            raise TimingPredictionError(
                f"static SASS binary mismatch for {cell['shape_point']}"
            )
        if static_work["kernel_signature_sha256"] != cell["kernel_signature_sha256"]:
            raise TimingPredictionError(
                f"static work signature mismatch for {cell['shape_point']}"
            )
        if (
            static_features["kernel_signature_sha256"]
            != cell["kernel_signature_sha256"]
        ):
            raise TimingPredictionError(
                f"static SASS signature mismatch for {cell['shape_point']}"
            )
        out_cells.append(
            {
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "arch": cell["arch"],
                "regime": "long",
                "metric_target": "cycles",
                "binary_sha256": cell["binary_sha256"],
                "kernel_signature_sha256": cell["kernel_signature_sha256"],
                "source_commit": cell["source_commit"],
                "join_key": _join_key(cell),
                "feature_vector": feats,
                "feature_provenance": {
                    "launch_geometry": "embedded_in_long_duration_cohort",
                    "nvbit_sector_summary": cell["feature_artifacts"][
                        "nvbit_sector_summary"
                    ]["path"],
                    "static_sass_features": cell["feature_artifacts"][
                        "static_sass_features"
                    ]["path"],
                    "static_work": cell["feature_artifacts"]["static_work"]["path"],
                    "device_peak_constants": _repo_rel(peak_constants_path),
                    "cache_hit_predictions": cell["feature_artifacts"][
                        "cache_hit_predictions"
                    ]["path"],
                    "static_work_kernel_signature_sha256": static_work[
                        "kernel_signature_sha256"
                    ],
                    "static_sass_kernel_signature_sha256": static_features[
                        "kernel_signature_sha256"
                    ],
                    "sector_summary_source_trace": sector["source_trace"][
                        "nvbit_trace_pb_path"
                    ],
                    "predicted_cache_hit_rate_source": cache_pred[
                        "head_provenance"
                    ]["producer"],
                    "predicted_l1_artifact": cache_pred["head_provenance"][
                        "l1_prediction"
                    ]["head_artifact"]["path"],
                    "predicted_l2_artifact": cache_pred["head_provenance"][
                        "l2_prediction"
                    ]["head_artifact"]["path"],
                    "compute_prior_source": COMPUTE_PRIOR_SOURCE,
                    "theoretical_occupancy_source": sector[
                        "theoretical_occupancy"
                    ]["source"],
                    "inference_inputs_deploy_legal": True,
                },
                "prediction": pred,
                "prediction_channel": DEPLOYABLE_CHANNEL,
                "diagnostics": {
                    "same_slug_scaled_neighbor": same_slug_diagnostic,
                },
                "prediction_mode": "leave_one_out_validation",
                "model_version": MODEL_VERSION,
                "typed_unavailable_reason": None,
                "validation_abs_log_error": abs(math.log(predicted_cycles / truth)),
            }
        )

    distinct_slugs = sorted({cell["slug"] for cell in cells})
    return {
        "schema_version": 1,
        "artifact_kind": "timing_cycle_prediction_bundle",
        "arch": cohort["arch"],
        "source_commit": source_commit or _source_commit(),
        "model_version": MODEL_VERSION,
        "deployable_channel": DEPLOYABLE_CHANNEL,
        "demoted_channels": [DIAGNOSTIC_SAME_SLUG_CHANNEL],
        "cohort_limitation": {
            "distinct_slug_count": len(distinct_slugs),
            "distinct_slugs": distinct_slugs,
            "typed_reason": COHORT_SLUG_DIVERSITY_TYPED_REASON,
            "same_slug_substitution_authorized": False,
            "note": (
                "Leave-one-out validation at this slug diversity is not "
                "slug-disjoint: every fold's residual fit still sees "
                "same-slug cells. This limitation does NOT authorize "
                "same-slug label substitution in the deployable channel; "
                "lifting it requires expanding the timing cohort's "
                "distinct-slug coverage."
            ),
        },
        "cohort_artifact": _repo_rel(cohort_path),
        "static_work_artifact": "per_cell_feature_artifacts.static_work",
        "static_sass_features_artifact": (
            "per_cell_feature_artifacts.static_sass_features"
        ),
        "device_peak_constants_artifact": _repo_rel(peak_constants_path),
        "training": {
            "target_metric_id": "gpc__cycles_elapsed.max",
            "n_cells": len(cells),
            "labels_used_only_for_residual_fit": True,
            "per_cell_labels_required_at_inference": False,
            "validation_method": "leave_one_out",
        },
        "model": {
            "kind": "pedestal_roofline_ridge_residual",
            "ridge_lambda": RIDGE_LAMBDA,
            "feature_names": list(FEATURE_NAMES),
            "residual_coefficients": full_fit_model["coefficients"],
            "pedestal_cycles": 0.0,
            "tensor_peak_flops_per_cycle": tensor_peak_flops_per_cycle,
            "tensor_roofline_efficiency_prior": TENSOR_ROOFLINE_EFFICIENCY_PRIOR,
            "effective_flops_per_cycle": effective_flops_per_cycle,
            "compute_prior_source": COMPUTE_PRIOR_SOURCE,
            "dram_bytes_per_cycle": dram_bytes_per_cycle,
        },
        "n_cells": len(out_cells),
        "cells": out_cells,
    }


def write_bundle(bundle: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--static-work", type=Path, default=DEFAULT_STATIC_WORK)
    parser.add_argument("--static-features", type=Path, default=DEFAULT_STATIC_FEATURES)
    parser.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    bundle = build_prediction_bundle(
        cohort_path=args.cohort,
        static_work_path=args.static_work,
        static_features_path=args.static_features,
        peak_constants_path=args.peaks,
    )
    write_bundle(bundle, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
