"""Emit the L1/TEX sector-hit-rate closure artifact (L1-head-scored).

Promotes the L1 hit-rate prediction to a load-bearing, comparison-runner-scored
metric. The SEPARATE bounded-[0,1] L1 calibration head instance
(``data/local_labels/sm_86_l1_head.json``) is LOAD-BEARING here: this module
rehydrates the head's ``PerBucketCalibratorEnsemble`` (via the shared
``v5_selected_class_dispatch`` no-fallback dispatch), applies it to each scalar
L1 walker row (the SAME zero-filled feature vector the L1 fitter used, with
``raw_prediction`` = the walker's predicted hit rate), and SCORES the CALIBRATED
prediction — never the raw walker value. It writes a single closure bundle whose
keys ARE the admitted manifest (``included_cells``), excluded manifest with typed
reasons (``excluded_cells``), per-layer comparison (``relative_error_stats_l1tex``),
coverage report (``coverage_gate_verdict``), and bootstrap CI
(``relative_error_geomean_ci``), plus an ``l1_head_provenance`` block recording
the head path / source_commit / fit_status / selected class and the per-cell
raw-vs-calibrated values (proving the scored values came from the HEAD, not from
``predicted_l1_per_admitted_cell.json`` directly).

Fail-closed discipline: if the L1 head fails validation
(``calibration_head_version != 5`` / ``head_kind != "l1_sector_hit_rate"`` /
``cache_layer != "L1TEX"`` / ``fit_status != "ok"``), or the ensemble cannot be
rehydrated, or it cannot score a scalar row, this module RAISES — it never
silently falls back to the raw walker prediction.

Coverage honesty
----------------
Only cells whose L1 walker emitted a SCALAR prediction enter the headline L1
geomean (calibrated by the head); cells whose L1 prediction is
``typed_unavailable`` (e.g. ``predictor_l1_sass_provenance_missing``) score as
``predictor_typed_unavailable`` exclusions, NOT join_key_unmatched. The coverage
gate's smoke (5) / publication (10) floors are APPLIED and reported verbatim.

The L1TEX-bundle transform (``build_l1tex_bundles_from_predictions``) and the
head-application (``calibrate_l1_predictions``) are exported so the headline
cohort driver emits IDENTICALLY-shaped, head-calibrated L1TEX bundles in lockstep
with the widened ``cache_layers = ["L1", "L2"]`` contract.

CLI:
  python tools/local_labels/emit_l1_closure_artifacts.py \
    [--l1-head data/local_labels/sm_86_l1_head.json] \
    [--l1-predictions data/sm_86_predictor_accuracy_recovery_v2/predicted_l1_per_admitted_cell.json] \
    [--label-admitted data/local_labels/label_admitted.json] \
    [--contract workloads/simbench_bw/comparison_contract.toml] \
    [--family-buckets data/local_labels/sm120_cohort_family_buckets.toml] \
    [--arch sm_86] \
    [--output data/local_labels/l1_closure_comparison_sm_86.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw import canonical_features as cf  # noqa: E402
from workloads.simbench_bw import comparison_runner as cr  # noqa: E402
from tools.local_labels.corpus_cell_identity import (  # noqa: E402
    is_base_predictor_fit_cell,
)
from tools.local_labels.fit_l1_calibration_head import (  # noqa: E402
    L1_LABEL_FIELD,
    _l1_features,
    _load_family_buckets,
    _resolve_family_bucket,
)
from tools.local_labels.v5_selected_class_dispatch import (  # noqa: E402
    build_selected_ensemble,
)

DEFAULT_L1_HEAD = "data/local_labels/sm_86_l1_head.json"
DEFAULT_L1_PREDICTIONS = (
    "data/sm_86_predictor_accuracy_recovery_v2/predicted_l1_per_admitted_cell.json"
)
DEFAULT_LABEL_ADMITTED = "data/local_labels/label_admitted.json"
DEFAULT_CONTRACT = "workloads/simbench_bw/comparison_contract.toml"
DEFAULT_FAMILY_BUCKETS_TOML = "data/local_labels/sm120_cohort_family_buckets.toml"
DEFAULT_OUTPUT = "data/local_labels/l1_closure_comparison_sm_86.json"
DEFAULT_ARCH = "sm_86"

# sm_86 (GA102) device whose name resolves to arch sm_86 via
# comparison_runner._arch_for_device. The 3090 Ti is the only sm_86 device.
_SM86_DEVICE = "NVIDIA GeForce RTX 3090 Ti"
# L1TEX unified-cache capacity for GA102 (128 KiB). Carried for bundle-shape
# completeness ONLY; the comparison runner scores predicted_hit_rate vs the L1
# label and does NOT consume this value.
_L1TEX_CAPACITY_BYTES = 128 * 1024

# The L1 head contract these emitters require before applying the head.
_REQUIRED_L1_HEAD_FIELDS = {
    "calibration_head_version": 5,
    "head_kind": "l1_sector_hit_rate",
    "cache_layer": "L1TEX",
    "fit_status": "ok",
}


class L1HeadValidationError(RuntimeError):
    """The committed L1 head artifact fails the load-bearing contract."""


class L1HeadNotApplicableError(RuntimeError):
    """The L1 head could not be rehydrated or could not score a scalar row."""


def load_and_validate_l1_head(path: Path) -> dict:
    """Load the L1 head artifact and enforce the load-bearing contract.

    Raises :class:`L1HeadValidationError` (fail closed) when any required field
    is absent or wrong — never returns a head that should not drive scoring.
    """
    if not path.is_file():
        raise L1HeadValidationError(
            f"L1 head artifact not found at {path}; the L1 closure cannot be "
            f"scored without the calibration head (no raw-walker fallback)"
        )
    head = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for field, expected in _REQUIRED_L1_HEAD_FIELDS.items():
        actual = head.get(field)
        if actual != expected:
            errors.append(f"{field}={actual!r} (expected {expected!r})")
    if errors:
        raise L1HeadValidationError(
            f"L1 head at {path} fails the load-bearing contract: "
            + "; ".join(errors)
        )
    return head


def calibrate_l1_predictions(
    predictions: dict,
    l1_head: dict,
    *,
    family_buckets_path: Path,
) -> tuple[list[dict], list[dict], str]:
    """Apply the L1 head to the walker predictions.

    Rehydrates the head's ensemble (no-fallback dispatch) and, for every scalar
    walker row, computes the CALIBRATED prediction from the same zero-filled
    feature vector the fitter used. Returns
    ``(calibrated_cells, head_provenance_rows, selected_class_name)``.

    Fail-closed: raises :class:`L1HeadNotApplicableError` if the ensemble cannot
    be built or a scalar row cannot be scored. typed_unavailable rows pass
    through unchanged.
    """
    dispatch = build_selected_ensemble(l1_head)
    if dispatch["status"] != "ok":
        raise L1HeadNotApplicableError(
            f"cannot rehydrate the L1 head ensemble (status="
            f"{dispatch['status']!r}): {dispatch.get('reason')!r}; refusing to "
            f"score raw walker predictions without the head"
        )
    ensemble = dispatch["ensemble"]
    selected_class_name = dispatch["selected_class_name"]
    family_to_bucket = _load_family_buckets(family_buckets_path)
    # The head's own cohort_scope.admitted is the authoritative (slug,shape)->
    # family_bucket map, guaranteeing the bucket each scalar row is scored under
    # matches the bucket it was trained under.
    bucket_by_key = {
        (c["slug"], c["shape_point"]): c["family_bucket"]
        for c in l1_head.get("cohort_scope", {}).get("admitted", [])
    }
    calibrated_cells: list[dict] = []
    provenance: list[dict] = []
    for cell in predictions.get("cells", []):
        out_cell = dict(cell)
        if cell.get("result_kind") == "scalar":
            raw = float(cell["predicted_hit_rate"])
            key = (cell["slug"], cell["shape_point"])
            bucket = bucket_by_key.get(key) or _resolve_family_bucket(
                cell["slug"], family_to_bucket,
            )
            features = _l1_features(raw)
            try:
                calibrated = float(
                    ensemble.predict(str(cell.get("arch", "")), bucket, features)
                )
            except (RuntimeError, ImportError) as exc:
                raise L1HeadNotApplicableError(
                    f"L1 head could not score scalar cell {key}: {exc}; "
                    f"refusing to fall back to the raw walker value"
                ) from exc
            out_cell["predicted_hit_rate"] = calibrated
            provenance.append({
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "arch": cell.get("arch"),
                "family_bucket": bucket,
                "raw_walker_prediction": raw,
                "calibrated_prediction": calibrated,
            })
        calibrated_cells.append(out_cell)
    return calibrated_cells, provenance, selected_class_name


def build_l1tex_bundles_from_predictions(
    predictions: dict,
    *,
    device: str = _SM86_DEVICE,
    cache_model_capacity_bytes: int = _L1TEX_CAPACITY_BYTES,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    epsilon: float = 0.01,
) -> list[dict[str, Any]]:
    """Transform per-cell L1 predictions into runner L1TEX bundles.

    ``predictions`` is a dict with a ``cells`` list; each scalar cell's
    ``predicted_hit_rate`` is taken AS-IS (callers pass head-CALIBRATED cells
    from :func:`calibrate_l1_predictions`). Each cell becomes one
    ``cache_layer == "L1TEX"`` bundle with the 5-field ``manifest_pairing_index``
    the 8-tuple join consumes; scalar cells carry a degenerate (point) CI;
    non-scalar cells preserve their typed_reason as a ``typed_unavailable``
    bundle so the runner classifies them as ``predictor_typed_unavailable``.
    """
    bundles: list[dict[str, Any]] = []
    for cell in predictions.get("cells", []):
        mpi = {
            "slug": cell["slug"],
            "shape_point": cell["shape_point"],
            "binary_sha256": cell["binary_sha256"],
            "source_commit": cell["source_commit"],
            "kernel_signature_sha256": cell["kernel_signature_sha256"],
        }
        bundle: dict[str, Any] = {
            "schema_version": 1,
            "algorithm_version": "smith_eklund_lru_v1",
            "manifest_pairing_index": mpi,
            "input_sidecar_digests": {},
            "device": device,
            "cache_model_capacity_bytes": cache_model_capacity_bytes,
            "sector_size_bytes": 32,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "epsilon": epsilon,
            "cache_layer": "L1TEX",
            "result_kind": cell["result_kind"],
        }
        if cell["result_kind"] == "scalar":
            point = float(cell["predicted_hit_rate"])
            bundle.update({
                "predicted_hit_rate": point,
                "ci_low": point,
                "ci_high": point,
                "ci_half_width": 0.0,
                "ci_confidence": 0.95,
            })
        else:
            bundle.update({
                "typed_reason": cell.get("typed_reason")
                or "predictor_l1_unavailable",
                "underconfigured_bin_count": 0,
                "admitted_bin_count": 0,
            })
        bundles.append(bundle)
    return bundles


def build_head_calibrated_l1tex_bundles(
    predictions: dict,
    l1_head: dict,
    *,
    family_buckets_path: Path,
    bootstrap_seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict], str]:
    """Convenience: calibrate predictions through the head, then build bundles.

    Returns ``(bundles, head_provenance_rows, selected_class_name)``. This is the
    single shared path both the closure emitter and the headline driver use so
    the scored L1TEX bundles always carry HEAD-calibrated values.
    """
    calibrated_cells, provenance, selected = calibrate_l1_predictions(
        predictions, l1_head, family_buckets_path=family_buckets_path,
    )
    bundles = build_l1tex_bundles_from_predictions(
        {"cells": calibrated_cells}, bootstrap_seed=bootstrap_seed,
    )
    return bundles, provenance, selected


def emit_l1_closure(
    *,
    l1_head_path: Path,
    l1_predictions_path: Path,
    label_admitted_path: Path,
    contract_path: Path,
    output_path: Path,
    family_buckets_path: Path,
    arch: str = DEFAULT_ARCH,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Load+validate the L1 head, calibrate the walker predictions through it,
    score the calibrated L1TEX bundles through the comparison runner, and write
    the L1 closure bundle (with head provenance). Returns the comparison dict."""
    l1_head = load_and_validate_l1_head(l1_head_path)
    predictions = json.loads(l1_predictions_path.read_text(encoding="utf-8"))
    bundles, head_provenance, selected_class_name = (
        build_head_calibrated_l1tex_bundles(
            predictions, l1_head,
            family_buckets_path=family_buckets_path,
            bootstrap_seed=bootstrap_seed,
        )
    )
    base_contract = cr.load_comparison_contract(str(contract_path))
    # Score L1 in isolation: narrow to the L1 layer + this arch so the closure
    # artifact is the L1 head's own coverage/comparison/bootstrap, uncontaminated
    # by the L2 layer (whose bundles are not supplied here).
    contract = dataclasses.replace(
        base_contract, arch_set=(arch,), cache_layers=("L1",),
    )
    label_cells = cr._load_admitted_label_cells(str(label_admitted_path))
    # Scope the closure's label universe to the base predictor-fit cohort so it
    # matches the head-calibrated bundles (built from the scoped predicted_l1)
    # and the scoped l1_head: a register-cap variant / pending-refit factorial
    # cell is admitted in the corpus but not yet folded into the head, so it
    # must not appear in this closure's included/excluded manifests until the
    # refit. The boundary is an sm_86 concept, so other arches use the arch
    # scope alone.
    if arch == "sm_86":
        label_cells = [c for c in label_cells if is_base_predictor_fit_cell(c)]
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=label_cells,
        bootstrap_seed=bootstrap_seed,
        target_arch=arch,
    )
    out["closure_artifact_kind"] = "l1_sector_hit_rate_closure"
    out["closure_cache_layer"] = "L1TEX"
    out["closure_l1_predictions_source"] = str(l1_predictions_path)
    # L1-head provenance: proves the scored L1 values came from the HEAD, not
    # directly from predicted_l1_per_admitted_cell.json.
    out["l1_head_provenance"] = {
        "l1_head_path": str(l1_head_path),
        "l1_head_source_commit": l1_head.get("source_commit"),
        "l1_head_fit_status": l1_head.get("fit_status"),
        "l1_head_label_field": l1_head.get("label_field", L1_LABEL_FIELD),
        "selected_class_name": selected_class_name,
        "head_admitted_count": l1_head.get("cohort_scope", {}).get(
            "admitted_count"
        ),
        "calibrated_cells": head_provenance,
    }
    output_path.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-head", default=DEFAULT_L1_HEAD)
    parser.add_argument("--l1-predictions", default=DEFAULT_L1_PREDICTIONS)
    parser.add_argument("--label-admitted", default=DEFAULT_LABEL_ADMITTED)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--family-buckets", default=DEFAULT_FAMILY_BUCKETS_TOML)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    out = emit_l1_closure(
        l1_head_path=Path(args.l1_head),
        l1_predictions_path=Path(args.l1_predictions),
        label_admitted_path=Path(args.label_admitted),
        contract_path=Path(args.contract),
        output_path=Path(args.output),
        family_buckets_path=Path(args.family_buckets),
        arch=args.arch,
        bootstrap_seed=args.bootstrap_seed,
    )
    pa = (
        out.get("per_arch_diagnostics", {})
        .get("per_arch", {})
        .get(args.arch, {})
    )
    l1 = pa.get("relative_error_stats_l1tex", {})
    cov = pa.get("coverage_gate_verdict", {})
    print(
        f"[emit_l1_closure] result_kind={out.get('result_kind')} "
        f"l1_scalar={l1.get('n_scalar')} "
        f"l1_typed_unavailable={l1.get('n_typed_unavailable')} "
        f"geomean={l1.get('geomean')} "
        f"headline_live={cov.get('headline_count_live')} "
        f"floor={cov.get('floor')} meets_floor={cov.get('meets_floor')} "
        f"-> {args.output}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
