"""Tests for the per_arch_diagnostics wrapper schema in
``workloads/simbench_bw/schemas/comparison_bundle.schema.json``.

The wrapper is an OPTIONAL top-level field added as an additive
extension; existing bundles without the field continue to validate.
Every per-arch entry has fixed-shape required keys with
``additionalProperties: false`` at every nested level. The
``relative_error_stats`` block uses a oneOf with three branches that
bind ``statistics_status`` via ``const``, so quartile fields can ONLY
appear when ``statistics_status == "quartile_emitted"``.

The schema's ``per_arch`` object's ``propertyNames`` match the regex
``^sm_[0-9]+$`` to keep arch keys pinned to the project's conventional
form. The ``unattributed_excluded_reason_counts`` block carries the
three-token vocabulary from the runner's ``_EXCLUDED_DIAGNOSTIC_VALUES``.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUITE_ROOT = _REPO_ROOT / "workloads" / "simbench_bw"
sys.path.insert(0, str(_SUITE_ROOT))


try:
    import jsonschema  # type: ignore[import-not-found]
    _HAS_JSONSCHEMA = True
except ImportError:  # pragma: no cover
    _HAS_JSONSCHEMA = False


_SCHEMA_PATH = _SUITE_ROOT / "schemas" / "comparison_bundle.schema.json"


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


# A baseline minimal-typed-unavailable bundle that validates against
# the existing schema. Tests build per_arch_diagnostics on top of this
# baseline so the additive-extension shape is exercised in isolation.
def _baseline_typed_unavailable_bundle() -> dict:
    return {
        "schema_version": 1,
        "algorithm_version": "comparison_runner_v1",
        "comparison_run_pairing_index": {
            "ncu_metric_name": "lts__t_sector_hit_rate.pct",
            "ncu_metric_version": "ncu_2026.1.x",
            "epsilon": 0.01,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 0,
            "ci_confidence": 0.95,
            "split_label": "all",
        },
        "result_kind": "typed_unavailable",
        "typed_reason": "headline_geomean_admitted_cell_count_below_threshold",
        "top_level_admission_scope": "cross_arch_any_arch",
        "comparable_cells_per_arch": {"sm_86": 0, "sm_120": 0},
        "headline_cells_per_arch": {"sm_86": 0, "sm_120": 0},
        "headline_floor_per_arch": 5,
        "included_cells": [],
        "excluded_cells": [],
    }


def _zero_excluded_reason_counts() -> dict:
    return {
        "comparison_contract_drift": 0,
        "comparison_runner_join_key_unmatched": 0,
        "predictor_typed_unavailable": 0,
    }


def _empty_per_layer_relative_error_stats() -> dict:
    """Zero-count per-layer relative-error stats block. Mirrors the
    runtime shape emitted by _per_arch_diagnostics for a per-arch
    slice with no included cells: every count is zero and geomean
    degenerates to 0.0."""
    return {
        "n_cells": 0,
        "n_scalar": 0,
        "n_bounded": 0,
        "n_typed_unavailable": 0,
        "geomean": 0.0,
    }


def _zero_per_bucket_relative_error_stats() -> dict:
    """Build a synthetic per-bucket block whose two buckets both
    carry ``no_cells`` so the additive shape validates without
    requiring per-bucket sample data."""
    no_cells_entry = {"statistics_status": "no_cells", "stats": None}
    return {
        "predictor_friendly": dict(no_cells_entry),
        "predictor_hostile": dict(no_cells_entry),
    }


def _zero_bounded_arm_relative_error_stats() -> dict:
    """Default bounded-arm block: zero bounded cells, both arms in
    no_cells branch with stats=None."""
    return {
        "n_bounded_cells": 0,
        "lower": {"statistics_status": "no_cells", "stats": None},
        "upper": {"statistics_status": "no_cells", "stats": None},
    }


def _default_relative_error_geomean_ci() -> dict:
    """Default per-arch CI block for fixtures with non-zero headline
    cells. Synthetic but schema-valid; tests exercising the runner
    use the runner-emitted block instead."""
    return {
        "geomean": 0.30,
        "ci_low": 0.20,
        "ci_high": 0.40,
        "ci_half_width": 0.10,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 0,
    }


def _empty_relative_error_geomean_ci() -> dict:
    """Default per-arch CI block for the no_cells branch: a
    deterministic empty-sample block carrying zeros for the four
    numeric fields plus the contract's bootstrap_resamples and
    seed. Mirrors the runner's no-cells emit."""
    return {
        "geomean": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "ci_half_width": 0.0,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 0,
    }


def _make_per_arch_quartile_emitted_record(
    *,
    arch: str = "sm_86",
    n_comparable_cells: int = 5,
    n_headline_cells: int = 5,
    relative_error_stats: dict | None = None,
    signed_residual_stats: dict | None = None,
    bounded_arm_relative_error_stats: dict | None = None,
    relative_error_geomean_ci: dict | None = None,
    excluded_reason_counts: dict | None = None,
    coverage_gate_verdict: dict | None = None,
) -> dict:
    """Build a synthetic quartile_emitted per-arch record. The
    ``relative_error_stats`` branch carries the 7-field stats payload."""
    if relative_error_stats is None:
        relative_error_stats = {
            "mean": 0.30,
            "stdev": 0.158,
            "min": 0.10,
            "max": 0.50,
            "q25": 0.20,
            "q50": 0.30,
            "q75": 0.40,
        }
    if signed_residual_stats is None:
        signed_residual_stats = {
            "mean": 0.0,
            "stdev": 0.158,
            "min": -0.10,
            "max": 0.10,
            "q25": -0.05,
            "q50": 0.0,
            "q75": 0.05,
        }
    if bounded_arm_relative_error_stats is None:
        bounded_arm_relative_error_stats = (
            _zero_bounded_arm_relative_error_stats()
        )
    if relative_error_geomean_ci is None:
        relative_error_geomean_ci = _default_relative_error_geomean_ci()
    if excluded_reason_counts is None:
        excluded_reason_counts = _zero_excluded_reason_counts()
    if coverage_gate_verdict is None:
        coverage_gate_verdict = {
            "arch": arch,
            "headline_count_live": n_headline_cells,
            "headline_count_synthetic": 0,
            "floor": 5,
            "meets_floor": n_headline_cells >= 5,
        }
    return {
        "arch": arch,
        "n_comparable_cells": n_comparable_cells,
        "n_headline_cells": n_headline_cells,
        "statistics_status": "quartile_emitted",
        "relative_error_stats": relative_error_stats,
        "relative_error_stats_l1tex": _empty_per_layer_relative_error_stats(),
        "relative_error_stats_l2": _empty_per_layer_relative_error_stats(),
        "signed_residual_stats": signed_residual_stats,
        "bounded_arm_relative_error_stats": bounded_arm_relative_error_stats,
        "relative_error_geomean_ci": relative_error_geomean_ci,
        "excluded_reason_counts": excluded_reason_counts,
        "coverage_gate_verdict": coverage_gate_verdict,
        "relative_error_stats_per_predictor_design_coverage": (
            _zero_per_bucket_relative_error_stats()
        ),
    }


def _make_per_arch_insufficient_cells_record(
    *,
    arch: str = "sm_86",
    n_headline_cells: int = 3,
) -> dict:
    """Build a synthetic insufficient_cells per-arch record. The
    ``relative_error_stats`` branch carries 4 fields; quartile fields
    are forbidden."""
    return {
        "arch": arch,
        "n_comparable_cells": n_headline_cells,
        "n_headline_cells": n_headline_cells,
        "statistics_status": "insufficient_cells_for_quartile_statistics",
        "relative_error_stats": {
            "mean": 0.20,
            "stdev": 0.05,
            "min": 0.15,
            "max": 0.25,
        },
        "relative_error_stats_l1tex": _empty_per_layer_relative_error_stats(),
        "relative_error_stats_l2": _empty_per_layer_relative_error_stats(),
        "signed_residual_stats": {
            "mean": -0.05,
            "stdev": 0.05,
            "min": -0.10,
            "max": 0.0,
        },
        "bounded_arm_relative_error_stats": (
            _zero_bounded_arm_relative_error_stats()
        ),
        "relative_error_geomean_ci": _default_relative_error_geomean_ci(),
        "excluded_reason_counts": _zero_excluded_reason_counts(),
        "coverage_gate_verdict": {
            "arch": arch,
            "headline_count_live": n_headline_cells,
            "headline_count_synthetic": 0,
            "floor": 5,
            "meets_floor": False,
        },
        "relative_error_stats_per_predictor_design_coverage": (
            _zero_per_bucket_relative_error_stats()
        ),
    }


def _make_per_arch_no_cells_record(*, arch: str = "sm_120") -> dict:
    """Build a synthetic no_cells per-arch record. The
    ``relative_error_stats`` branch is JSON null; the
    coverage_gate_verdict reports zero counts."""
    return {
        "arch": arch,
        "n_comparable_cells": 0,
        "n_headline_cells": 0,
        "statistics_status": "no_cells",
        "relative_error_stats": None,
        "relative_error_stats_l1tex": _empty_per_layer_relative_error_stats(),
        "relative_error_stats_l2": _empty_per_layer_relative_error_stats(),
        "signed_residual_stats": None,
        "bounded_arm_relative_error_stats": (
            _zero_bounded_arm_relative_error_stats()
        ),
        "relative_error_geomean_ci": _empty_relative_error_geomean_ci(),
        "excluded_reason_counts": _zero_excluded_reason_counts(),
        "coverage_gate_verdict": {
            "arch": arch,
            "headline_count_live": 0,
            "headline_count_synthetic": 0,
            "floor": 5,
            "meets_floor": False,
        },
        "relative_error_stats_per_predictor_design_coverage": (
            _zero_per_bucket_relative_error_stats()
        ),
    }


def _make_per_arch_diagnostics_wrapper(
    *,
    per_arch: dict | None = None,
    unattributed: dict | None = None,
) -> dict:
    """Build a synthetic per_arch_diagnostics wrapper carrying the
    requested per-arch records and unattributed exclusion counts."""
    if per_arch is None:
        per_arch = {
            "sm_86": _make_per_arch_quartile_emitted_record(arch="sm_86"),
            "sm_120": _make_per_arch_no_cells_record(arch="sm_120"),
        }
    if unattributed is None:
        unattributed = _zero_excluded_reason_counts()
    return {
        "per_arch": per_arch,
        "unattributed_excluded_reason_counts": unattributed,
    }


# =====================================================================
# Backward compatibility — bundles WITHOUT per_arch_diagnostics still
# validate. The wrapper is an optional additive extension.
# =====================================================================


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_bundle_without_per_arch_diagnostics_validates() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    jsonschema.validate(bundle, schema)


# =====================================================================
# Wrapper presence — bundles with per_arch_diagnostics validate when
# every per-arch record carries the required shape.
# =====================================================================


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_bundle_with_per_arch_diagnostics_full_quartile_emitted_validates() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    bundle["per_arch_diagnostics"] = _make_per_arch_diagnostics_wrapper(
        per_arch={
            "sm_86": _make_per_arch_quartile_emitted_record(arch="sm_86"),
            "sm_120": _make_per_arch_quartile_emitted_record(arch="sm_120"),
        },
    )
    jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_bundle_with_per_arch_diagnostics_insufficient_cells_validates() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    bundle["per_arch_diagnostics"] = _make_per_arch_diagnostics_wrapper(
        per_arch={
            "sm_86": _make_per_arch_insufficient_cells_record(arch="sm_86"),
            "sm_120": _make_per_arch_no_cells_record(arch="sm_120"),
        },
    )
    jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_bundle_with_per_arch_diagnostics_no_cells_validates() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    bundle["per_arch_diagnostics"] = _make_per_arch_diagnostics_wrapper(
        per_arch={
            "sm_86": _make_per_arch_no_cells_record(arch="sm_86"),
            "sm_120": _make_per_arch_no_cells_record(arch="sm_120"),
        },
    )
    jsonschema.validate(bundle, schema)


# =====================================================================
# Negative — additionalProperties:false enforced at every nested level.
# =====================================================================


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_per_arch_record_with_unknown_key_is_rejected() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    record = _make_per_arch_quartile_emitted_record(arch="sm_86")
    record["rogue_extra_field"] = "boom"
    bundle["per_arch_diagnostics"] = {
        "per_arch": {"sm_86": record},
        "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_per_arch_record_missing_required_key_is_rejected() -> None:
    """Every required field on a per-arch record must be enforced by the
    schema. The iteration covers every entry in the per_arch_record
    `required` clause to catch a regression where the schema's required
    list and the negative-test loop fall out of lockstep
    (BL-20260506-schema-validator-probe-driven-tightening: enumerate
    every clause of the schema source text, not just the most obvious
    ones)."""
    schema = _load_schema()
    required_fields = (
        "arch",
        "n_comparable_cells",
        "n_headline_cells",
        "statistics_status",
        "relative_error_stats",
        "signed_residual_stats",
        "bounded_arm_relative_error_stats",
        "relative_error_geomean_ci",
        "excluded_reason_counts",
        "coverage_gate_verdict",
        "relative_error_stats_per_predictor_design_coverage",
    )
    for missing in required_fields:
        bundle = _baseline_typed_unavailable_bundle()
        record = _make_per_arch_quartile_emitted_record(arch="sm_86")
        record.pop(missing)
        bundle["per_arch_diagnostics"] = {
            "per_arch": {"sm_86": record},
            "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_missing_required_key_loop_matches_schema_required_list() -> None:
    """Structural lockstep guard: the iterated set in the
    missing-required-key negative loop above MUST equal the schema's
    `per_arch_record.required` list. If the schema gains a new required
    field but the test loop is not updated, this guard fails before the
    primary negative test is silently weakened."""
    schema = _load_schema()
    schema_required = set(
        schema["$defs"]["per_arch_record"]["required"]
    )
    test_iterated = {
        "arch",
        "n_comparable_cells",
        "n_headline_cells",
        "statistics_status",
        "relative_error_stats",
        "relative_error_stats_l1tex",
        "relative_error_stats_l2",
        "signed_residual_stats",
        "bounded_arm_relative_error_stats",
        "relative_error_geomean_ci",
        "excluded_reason_counts",
        "coverage_gate_verdict",
        "relative_error_stats_per_predictor_design_coverage",
    }
    assert schema_required == test_iterated, (
        f"per-arch required-field list drifted: schema={schema_required!r}, "
        f"test loop={test_iterated!r}"
    )


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_per_arch_record_with_invalid_statistics_status_is_rejected() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    record = _make_per_arch_quartile_emitted_record(arch="sm_86")
    record["statistics_status"] = "invalid_status_token"
    bundle["per_arch_diagnostics"] = {
        "per_arch": {"sm_86": record},
        "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_quartile_field_on_insufficient_cells_record_is_rejected() -> None:
    """The oneOf branch binds statistics_status via const so quartile
    fields (q25, q50, q75) can ONLY appear on records where
    statistics_status == 'quartile_emitted'."""
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    record = _make_per_arch_insufficient_cells_record(arch="sm_86")
    record["relative_error_stats"]["q50"] = 0.20
    bundle["per_arch_diagnostics"] = {
        "per_arch": {"sm_86": record},
        "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_quartile_field_on_no_cells_record_is_rejected() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    record = _make_per_arch_no_cells_record(arch="sm_120")
    record["relative_error_stats"] = {"q50": 0.20}
    bundle["per_arch_diagnostics"] = {
        "per_arch": {"sm_120": record},
        "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_excluded_reason_counts_with_unknown_key_is_rejected() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    record = _make_per_arch_quartile_emitted_record(arch="sm_86")
    record["excluded_reason_counts"]["unknown_token"] = 1
    bundle["per_arch_diagnostics"] = {
        "per_arch": {"sm_86": record},
        "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_per_arch_key_outside_arch_pattern_is_rejected() -> None:
    """The schema's per_arch object pins propertyNames to the
    ``^sm_[0-9]+$`` regex; non-conforming keys are rejected."""
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    bundle["per_arch_diagnostics"] = {
        "per_arch": {
            "rocm_950": _make_per_arch_quartile_emitted_record(arch="rocm_950"),
        },
        "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_unattributed_excluded_reason_counts_with_missing_key_is_rejected() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    incomplete_unattributed = _zero_excluded_reason_counts()
    incomplete_unattributed.pop("comparison_contract_drift")
    bundle["per_arch_diagnostics"] = {
        "per_arch": {
            "sm_86": _make_per_arch_quartile_emitted_record(arch="sm_86"),
        },
        "unattributed_excluded_reason_counts": incomplete_unattributed,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_per_arch_diagnostics_wrapper_with_unknown_key_is_rejected() -> None:
    schema = _load_schema()
    bundle = _baseline_typed_unavailable_bundle()
    bundle["per_arch_diagnostics"] = {
        "per_arch": {
            "sm_86": _make_per_arch_quartile_emitted_record(arch="sm_86"),
        },
        "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
        "rogue_wrapper_field": "boom",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_coverage_gate_verdict_missing_field_is_rejected() -> None:
    """The per-arch coverage_gate_verdict block requires the four
    fields {arch, headline_count_live, headline_count_synthetic,
    floor, meets_floor}. Omitting any one fails validation."""
    schema = _load_schema()
    required = (
        "arch",
        "headline_count_live",
        "headline_count_synthetic",
        "floor",
        "meets_floor",
    )
    for missing in required:
        bundle = _baseline_typed_unavailable_bundle()
        record = _make_per_arch_quartile_emitted_record(arch="sm_86")
        record["coverage_gate_verdict"].pop(missing)
        bundle["per_arch_diagnostics"] = {
            "per_arch": {"sm_86": record},
            "unattributed_excluded_reason_counts": _zero_excluded_reason_counts(),
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bundle, schema)
