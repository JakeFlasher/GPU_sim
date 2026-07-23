"""Compute the cross-cell L2 prediction stddev over the admitted
cohort and emit the variance-trajectory artifact.

The artifact is the load-bearing input for the calibration-
application decision: this plan ALWAYS defers calibration (per
the no-calibration directive); the precondition_status field
records whether the plan-pinned floor stddev of 0.10 is met,
which the downstream calibration-fitting plan will use to decide
whether calibration is applicable at all.

Inputs:
  - ``data/local_labels/label_admitted.json`` (the operator-
    facing admitted-cell index; filtered to ``arch=sm_86``
    for this plan's scope).
  - For each admitted cell, the per-cell trace directory at
    ``workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/trace_<shape_point>/``
    and its ``sector_stream/sector_stream.jsonl`` sidecar.

Outputs:
  - ``data/sm_86_predictor_accuracy_recovery_v2/ac5_variance_trajectory_post_corpus_expansion.json``
    validating against
    ``tools/sm_86_predictor_accuracy_recovery_v2/ac5_variance_trajectory.schema.json``.

The tool is byte-deterministic for the same input set: the L2
predictor runs with a fixed ``bootstrap_seed`` and the cells
list is sorted by ``(slug, shape_point)``.

CLI:

    python3 tools/sm_86_predictor_accuracy_recovery_v2/compute_variance_trajectory.py \\
        --label-admitted data/local_labels/label_admitted.json \\
        --out data/sm_86_predictor_accuracy_recovery_v2/ac5_variance_trajectory_post_corpus_expansion.json

This artifact also records the operator's
``partial_closure_override`` decision: the explicit acceptance
of the 9-cell admitted cohort as this plan's de-facto admitted
state when the cohort cannot reach the plan-pinned ``>=11``
floor under the contract-pinned ``stddev_threshold=0.03``
repeatability gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]

# Local sibling import. label_loader.load_labels_for_mode is
# the canonical entrypoint for operational-blinding-aware
# label I/O; calibration/validation-mode runners must not
# accept the sealed holdout labels path.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import label_loader  # noqa: E402


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SCHEMA_PATH = _HERE / "ac5_variance_trajectory.schema.json"
_DEFAULT_LABEL_ADMITTED = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
_DEFAULT_OUT = (
    _REPO_ROOT
    / "data"
    / "sm_86_predictor_accuracy_recovery_v2"
    / "ac5_variance_trajectory_post_corpus_expansion.json"
)
_DEFAULT_ARCH_DIR = "rtx_3090_ti/sm_86"
_DOCUMENTED_FLOOR_STDDEV = 0.10

# Operator partial-closure-override metadata (recorded in the
# host-local goal-tracker.md operator-decision log on the
# session date below). The bottleneck is label-side NCU-replay
# variance on predictor_hostile-by-design L2-stressor cells,
# structurally above the contract-pinned 0.03 stddev gate.
_PARTIAL_CLOSURE_OVERRIDE_DEFAULT: dict = {
    "accepted": True,
    "date_iso": "2026-05-18",
    "decided_admitted_count": 9,
    "reason": (
        "Operator accepted the 9-cell admitted cohort as this "
        "plan's de-facto admitted state. The plan-pinned >=11 "
        "admission floor cannot be reached under the contract-"
        "pinned 0.03 stddev repeatability gate because 4 of the 6 "
        "candidate cells in the strengthening matrix are "
        "predictor_hostile-by-design L2-stressors "
        "(dram_stream_stride_mix / l2_persisting_window_sweep / "
        "l2_prefetch_hint_sweep / warm_cold_pair_relaunch) whose "
        "inherent NCU-replay variance is structurally above the "
        "gate. The bottleneck is label-side measurement-"
        "repeatability, not a predictor-accuracy gap; the gate is "
        "not loosened. Set-associative promotion was rejected as "
        "L1-only and not an L2 admission path."
    ),
}

# The 13-cell HEADLINE_COHORT partitioning into the friendly
# vs hostile buckets the plan pins. Sourced from the same
# constant as `cohort_accounting.schema.json`.
_PREDICTOR_FRIENDLY: frozenset[tuple[str, str]] = frozenset({
    ("dram_stream_stride_mix", "trace_l2_accuracy_xl"),
    ("working_set_capacity_sweep", "trace_l2_accuracy_xl"),
    ("l2_thrash_pointer_chase", "trace_l2_accuracy_xl"),
    ("layernorm_fwd_bf16_sm80", "timing_shape"),
    ("rmsnorm_fwd_bf16_sm80", "timing_shape"),
    ("silu_fwd_bf16_sm80", "correctness_shape"),
    ("silu_fwd_bf16_sm80", "timing_shape"),
    ("softmax_safe_bf16_sm80", "timing_shape"),
})
_PREDICTOR_HOSTILE: frozenset[tuple[str, str]] = frozenset({
    ("carveout_sensitivity_probe", "trace_l2_accuracy_xl"),
    ("warm_cold_pair_relaunch", "trace_l2_accuracy_xl"),
    ("l2_persisting_window_sweep", "trace_l2_accuracy_xl"),
    ("l2_prefetch_hint_sweep", "trace_l2_accuracy_xl"),
    ("register_spill_local_mem", "trace_l2_accuracy_xl"),
})


def _load_cache_predictor():
    """Load workloads/simbench_bw/cache_predictor.py via
    importlib so the tool runs without pip-install."""
    cp_path = (
        _REPO_ROOT / "workloads" / "simbench_bw" / "cache_predictor.py"
    )
    module_name = "_ac5_cache_predictor"
    spec = importlib.util.spec_from_file_location(module_name, cp_path)
    module = importlib.util.module_from_spec(spec)
    # cache_predictor imports siblings under workloads/simbench_bw/
    # so push that dir onto sys.path. Register in sys.modules BEFORE
    # exec_module so Python's @dataclass machinery can resolve
    # cls.__module__ during the module-body execution.
    simbench_dir = cp_path.parent
    if str(simbench_dir) not in sys.path:
        sys.path.insert(0, str(simbench_dir))
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _predictor_bucket(slug: str, shape_point: str) -> str:
    key = (slug, shape_point)
    if key in _PREDICTOR_FRIENDLY:
        return "predictor_friendly"
    if key in _PREDICTOR_HOSTILE:
        return "predictor_hostile"
    # Default conservative: treat unknown cells as friendly so
    # the plan-pinned headline geomean doesn't get polluted.
    # The cohort schema enforces membership in the 13-cell set
    # upstream, so unknown keys here are a bug surface.
    raise ValueError(
        f"compute_variance_trajectory: cell "
        f"{slug}/{shape_point} is not in the documented "
        f"HEADLINE_COHORT_13 friendly/hostile partition; "
        f"this is a tool bug — either the cohort widened or "
        f"the partition needs updating"
    )


def _load_launch_parameters_sidecar_for(
    *,
    slug: str,
    shape_point: str,
    repo_root: Path,
    expected_identity: dict[str, str],
    cp_mod,
) -> tuple[dict | None, Path | None, str | None]:
    """Load + semantically validate the per-cell launch-parameter
    sidecar via the shared ``cache_predictor.load_launch_parameters_sidecar``
    loader. Returns ``(grid_dim, sidecar_path, sha256_hex)`` when
    the sidecar exists at the canonical filename AND its embedded
    identity tuple matches ``expected_identity``; returns
    ``(None, None, None)`` when the sidecar file is missing
    (legacy fall-through to the no-correction path). Raises
    ``cache_predictor.LaunchParametersSidecarIdentityMismatch``
    on identity mismatch OR malformed grid_dim payload.

    The producer-side semantic-binding contract requires every
    predictor input that influences the bundle output to be
    content-bound to the caller identity; this helper enforces
    that contract symmetrically with the headline-orchestrator
    caller in ``run_headline_cohort_predict_compare.py``."""
    sidecar_path = (
        repo_root / "data" / "local_labels"
        / "launch_parameters_sidecars" / "sm_86"
        / f"launch_parameters_sidecar_{slug}__{shape_point}.json"
    )
    if not sidecar_path.is_file():
        return None, None, None
    grid_dim, sha256_hex = cp_mod.load_launch_parameters_sidecar(
        sidecar_path=sidecar_path,
        expected_identity=expected_identity,
    )
    return grid_dim, sidecar_path, sha256_hex


def _per_cell_predicted_hit_rate(
    cell: dict,
    *,
    cp_mod,
    arch_dir: str,
    repo_root: Path,
) -> float | None:
    """Drive the L2 predictor on a single admitted cell's
    trace directory. Returns the scalar predicted_hit_rate
    fraction in [0, 1] or None when the predictor emits a
    typed-unavailable bundle (the variance-trajectory contract
    requires every admitted cell to yield a scalar; absence
    flags a regression that the caller must reject)."""
    slug = str(cell["slug"])
    shape_point = str(cell["shape_point"])
    trace_dir = (
        repo_root / "workloads" / "simbench_bw" / "results"
        / slug / arch_dir / f"trace_{shape_point}"
    )
    sector_stream = (
        trace_dir / "sector_stream" / "sector_stream.jsonl"
    )
    if not trace_dir.is_dir():
        return None
    if not sector_stream.is_file():
        return None
    mpi = {
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": str(cell.get("binary_sha256", "")),
        "source_commit": str(cell.get("source_commit", "")),
        "kernel_signature_sha256": str(
            cell.get("kernel_signature_sha256", "")
        ),
    }
    # DEC-2-companion: source gridDim from the committed
    # launch-parameter sidecar (predictor-side, never from
    # labels/truth). Per the R44 semantic-binding contract, the
    # shared loader validates the sidecar's identity tuple
    # against the caller's admitted-row tuple and fails closed
    # on mismatch; a stale or swapped sidecar cannot silently
    # apply the wrong gridDim correction. Missing sidecar ->
    # grid_dim=None -> legacy behavior preserved.
    expected_identity = {
        "arch": str(cell.get("arch", "sm_86")),
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": mpi["binary_sha256"],
        "kernel_signature_sha256": mpi["kernel_signature_sha256"],
    }
    grid_dim, sidecar_path, sidecar_sha256 = (
        _load_launch_parameters_sidecar_for(
            slug=slug,
            shape_point=shape_point,
            repo_root=repo_root,
            expected_identity=expected_identity,
            cp_mod=cp_mod,
        )
    )
    bundle = cp_mod.predict_cache_hit_rate(
        trace_dir=trace_dir,
        sector_stream_path=sector_stream,
        cache_model_capacity_bytes=6_291_456,
        manifest_pairing_index=mpi,
        device="NVIDIA GeForce RTX 3090 Ti",
        bootstrap_seed=42,
        grid_dim=grid_dim,
        launch_parameters_sidecar_path=sidecar_path,
        launch_parameters_sidecar_sha256=sidecar_sha256,
        # R45: forward the validated identity tuple symmetrically
        # with the headline-orchestrator caller so the variance
        # trajectory's per-cell predict bundles carry the same
        # `evidence.launch_parameters_sidecar_identity` field.
        launch_parameters_sidecar_identity=(
            expected_identity if sidecar_path is not None else None
        ),
    )
    if bundle.get("result_kind") != "scalar":
        return None
    return float(bundle.get("predicted_hit_rate"))


def build_variance_trajectory(
    *,
    label_admitted_path: Path,
    arch_dir: str = _DEFAULT_ARCH_DIR,
    repo_root: Path = _REPO_ROOT,
    cp_mod=None,
    partial_closure_override: dict | None = None,
) -> dict:
    """Build the variance-trajectory artifact dict.

    Pure function: takes the inputs explicitly so tests can
    inject synthetic label/predictor surfaces. The caller is
    responsible for writing the result to disk and running
    schema validation; see `main(...)` for the CLI shape.
    """
    if cp_mod is None:
        cp_mod = _load_cache_predictor()
    if partial_closure_override is None:
        partial_closure_override = dict(
            _PARTIAL_CLOSURE_OVERRIDE_DEFAULT
        )

    admitted = label_loader.load_labels_for_mode(
        label_admitted_path, "calibration",
    )
    sm_86_cells = [
        c for c in admitted.get("cells", [])
        if isinstance(c, dict) and c.get("arch") == "sm_86"
    ]
    if not sm_86_cells:
        raise ValueError(
            "compute_variance_trajectory: zero sm_86 admitted "
            f"cells in {label_admitted_path}"
        )

    cells_rows: list[dict] = []
    loocv_rows: list[dict] = []
    predicted_values: list[float] = []
    for cell in sorted(
        sm_86_cells,
        key=lambda c: (str(c.get("slug")), str(c.get("shape_point"))),
    ):
        slug = str(cell["slug"])
        shape_point = str(cell["shape_point"])
        predicted = _per_cell_predicted_hit_rate(
            cell,
            cp_mod=cp_mod,
            arch_dir=arch_dir,
            repo_root=repo_root,
        )
        if predicted is None:
            raise ValueError(
                f"compute_variance_trajectory: cell "
                f"{slug}/{shape_point} did not produce a "
                f"scalar L2 prediction; the variance-"
                f"trajectory contract requires every admitted "
                f"cell to yield a scalar predicted_hit_rate"
            )
        ncu_truth_pct = float(cell.get("cache_hit_rate_l2_sector_pct", 0.0))
        ncu_truth_frac = ncu_truth_pct / 100.0
        signed_residual = predicted - ncu_truth_frac
        bucket = _predictor_bucket(slug, shape_point)
        cells_rows.append({
            "arch": "sm_86",
            "slug": slug,
            "shape_point": shape_point,
            "predicted_l2_hit_rate": predicted,
            "ncu_truth_l2_fraction": ncu_truth_frac,
            "signed_residual": signed_residual,
            "predictor_design_coverage_bucket": bucket,
        })
        loocv_rows.append({
            "arch": "sm_86",
            "slug": slug,
            "shape_point": shape_point,
            "predicted": predicted,
            "truth": ncu_truth_frac,
        })
        predicted_values.append(predicted)

    cross_cell_stddev = statistics.stdev(predicted_values)
    delta_pred = max(predicted_values) - min(predicted_values)
    heuristic_proj = delta_pred / math.sqrt(len(predicted_values) + 1)
    precondition_status = (
        "satisfied"
        if cross_cell_stddev >= _DOCUMENTED_FLOOR_STDDEV
        else "not_satisfied"
    )

    artifact: dict = {
        "schema_version": 1,
        "generator": (
            "tools/sm_86_predictor_accuracy_recovery_v2/"
            "compute_variance_trajectory.py"
        ),
        "cohort_admitted_count": len(cells_rows),
        "cells": cells_rows,
        "cross_cell_stddev": cross_cell_stddev,
        "documented_floor_stddev": _DOCUMENTED_FLOOR_STDDEV,
        "precondition_status": precondition_status,
        "heuristic_stddev_projection": heuristic_proj,
        "loocv_inputs": loocv_rows,
        "calibration_application_verdict": (
            "deferred_to_plan_2_per_dec_1"
        ),
        "partial_closure_override": partial_closure_override,
    }
    return artifact


def validate_artifact(artifact: dict) -> None:
    """Strict-schema-validate the artifact. Caller catches
    `jsonschema.ValidationError` to map to a fail-close
    diagnostic."""
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(instance=artifact, schema=schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the cross-cell L2 prediction stddev over "
            "the admitted cohort and emit the variance-"
            "trajectory artifact."
        ),
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=_DEFAULT_LABEL_ADMITTED,
    )
    parser.add_argument(
        "--arch-dir",
        type=str,
        default=_DEFAULT_ARCH_DIR,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
    )
    args = parser.parse_args(argv)

    try:
        artifact = build_variance_trajectory(
            label_admitted_path=args.label_admitted,
            arch_dir=args.arch_dir,
            repo_root=args.repo_root,
        )
        validate_artifact(artifact)
    except (ValueError, jsonschema.ValidationError) as exc:
        sys.stderr.write(
            f"compute_variance_trajectory: {exc}\n"
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    sys.stderr.write(
        f"compute_variance_trajectory: wrote {args.out} "
        f"(cohort_admitted_count={artifact['cohort_admitted_count']}, "
        f"cross_cell_stddev={artifact['cross_cell_stddev']:.6f}, "
        f"precondition_status={artifact['precondition_status']})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
