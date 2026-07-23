"""Cross-arch comparison-harness integration tests.

Exercises the per_arch_diagnostics + live-only floor wiring end-to-end
on both arches (sm_86 + sm_120) using:

- The 5 real sm_86 admitted cells in
  ``data/local_labels/label_admitted.json`` (migrated to v3 with
  ``evidence_kind: "live"``) joined to synthesized matching predictor
  bundles built from the on-disk join keys.
- A synthetic 5-cell sm_120 fixture declared
  ``evidence_kind: "synthetic"`` that exercises the wire format
  end-to-end without sm_120 hardware. Its per-arch
  ``coverage_gate_verdict.meets_floor`` is False because the live
  count is zero.

The harness is the offline cross-arch readiness check the cross-host
pivot relies on: when sm_120 evidence eventually lands on the RTX
5060 laptop, the operator runs the same harness without code changes
and the per-arch verdict block reflects the live cross-arch
admission state.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUITE_ROOT = _REPO_ROOT / "workloads" / "simbench_bw"
_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SUITE_ROOT))
sys.path.insert(0, str(_TESTS_DIR))

import comparison_runner as cr  # noqa: E402
from conftest_cache_predictor_comparison import (  # noqa: E402
    _make_synthetic_admitted_predictor_bundle,
    _make_synthetic_manifest_pairing_index,
    _make_synthetic_ncu_label_cell,
)


try:
    import jsonschema  # type: ignore[import-not-found]
    _HAS_JSONSCHEMA = True
except ImportError:  # pragma: no cover
    _HAS_JSONSCHEMA = False


_LIVE_LABEL_ADMITTED_PATH = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
_V5_COHORT_STAGING_MANIFEST_PATH = (
    _REPO_ROOT / "data" / "local_labels" / "v5_cohort_staging_manifest.json"
)
_LOCAL_LABEL_ADMITTED_SCHEMA_PATH = (
    _SUITE_ROOT / "schemas" / "local_label_admitted.schema.json"
)
_COMPARISON_BUNDLE_SCHEMA_PATH = (
    _SUITE_ROOT / "schemas" / "comparison_bundle.schema.json"
)


# Mapping from arch token to the device string the predictor bundle
# carries. The runner's _DEVICE_TO_ARCH map inverts this; the
# fixture-builder uses the forward mapping to construct bundles whose
# device-derived arch matches the label cell's arch.
_ARCH_TO_DEVICE: dict[str, str] = {
    "sm_86": "NVIDIA GeForce RTX 3090 Ti",
    "sm_120": "NVIDIA GeForce RTX 5060 Laptop GPU",
}


def _build_predictor_bundle_from_label_cell(
    tmp_path: Path,
    cell: dict,
    *,
    predicted_hit_rate: float | None = None,
    bundle_subdir: str = "bundle",
) -> dict:
    """Synthesize a predictor bundle whose 7-field join key matches
    the given label cell. The bundle's manifest_pairing_index carries
    the cell's slug, shape_point, binary_sha256, source_commit, and
    kernel_signature_sha256; the device-string is derived from the
    cell's arch via ``_ARCH_TO_DEVICE``.

    When ``predicted_hit_rate`` is None, the predicted value is set
    to the cell's ground truth (``cache_hit_rate_l2_sector_pct / 100``)
    so the resulting relative_error is 0.0 — the comparison
    succeeds and the cell is counted toward the headline floor.
    Tests that exercise non-zero error pass an explicit
    ``predicted_hit_rate`` override."""
    if predicted_hit_rate is None:
        predicted_hit_rate = float(
            cell.get("cache_hit_rate_l2_sector_pct", 0.0)
        ) / 100.0
    arch = cell.get("arch", "sm_86")
    device = _ARCH_TO_DEVICE.get(arch, "")
    mpi = _make_synthetic_manifest_pairing_index(
        slug=cell["slug"],
        shape_point=cell["shape_point"],
        binary_sha256=cell["binary_sha256"],
        source_commit=cell["source_commit"],
        kernel_signature_sha256=cell["kernel_signature_sha256"],
    )
    return _make_synthetic_admitted_predictor_bundle(
        tmp_path / bundle_subdir,
        result_kind="scalar",
        predicted_hit_rate=predicted_hit_rate,
        manifest_pairing_index=mpi,
        device=device,
    )


def _baseline_contract() -> cr.ComparisonContract:
    return cr.ComparisonContract(
        ncu_metric_name="lts__t_sector_hit_rate.pct",
        ncu_metric_version="ncu_2026.1.x",
        percent_to_fraction=True,
        epsilon=0.01,
        bootstrap_resamples=2000,
        ci_confidence=0.95,
        bootstrap_unit="cell",
        bounded_arm_handling="report_three",
        typed_unavailable_handling="exclude",
        join_key=(
            "slug",
            "shape_point",
            "binary_sha256",
            "source_commit",
            "kernel_signature_sha256",
            "arch",
            "ncu_metric_version",
        ),
        coverage_gate=cr.CoverageGate(
            smoke_test_floor_per_arch=5,
            publication_grade_floor_per_arch=10,
            aspirational_target_total=20,
            require_both_arches_for_cross_arch=False,
        ),
        log_space_geomean=True,
        per_arch_diagnostics_config=cr.PerArchDiagnosticsConfig(
            diagnostic_vocabulary=tuple(
                sorted(cr._EXCLUDED_DIAGNOSTIC_VALUES),
            ),
            quartile_min_n=4,
        ),
        arch_set=("sm_86", "sm_120"),
    )


def _load_live_admitted_cells() -> list[dict]:
    """Read the on-disk admitted-cell file and return its cells. The
    aggregator-emitted cells carry the runner-joinable 7-field key
    plus profiler_version + ncu_metric_name (added at the runner's
    consumption boundary)."""
    payload = json.loads(_LIVE_LABEL_ADMITTED_PATH.read_text(encoding="utf-8"))
    cells = list(payload.get("cells", []))
    # The runner reads ncu_metric_name from each cell's record at
    # the per-cell join boundary; admitted records emitted by the
    # aggregator do NOT carry it because the field is contract-side
    # only. Fill it in here so the synthesized predictor bundles
    # join cleanly.
    for cell in cells:
        cell.setdefault("ncu_metric_name", "lts__t_sector_hit_rate.pct")
    return cells


def _load_sm86_staging_typed_failures() -> dict[tuple[str, str], str]:
    """Read v5_cohort_staging_manifest.json and return a map from
    each sm_86 (slug, shape_point) that the cohort pivot demoted to
    ``TERMINAL_TYPED_FAILURE`` onto its ``action_typed_reason``.

    This is the live-state source of truth for *why* a former
    activation cell is absent from ``label_admitted.json``: the pivot
    evicts cells via a typed terminal failure (e.g.
    ``label_repeatability_failed``) and the eviction is recorded here,
    not in the admitted corpus. Tests derive the expected
    eviction-vs-survival split from this manifest rather than pinning
    a stale same-74-era membership snapshot."""
    payload = json.loads(
        _V5_COHORT_STAGING_MANIFEST_PATH.read_text(encoding="utf-8"),
    )
    typed_failures: dict[tuple[str, str], str] = {}
    for row in payload.get("rows", []):
        if row.get("arch") != "sm_86":
            continue
        if row.get("suggested_operator_action") != "TERMINAL_TYPED_FAILURE":
            continue
        key = (row.get("slug"), row.get("shape_point"))
        typed_failures[key] = row.get("action_typed_reason")
    return typed_failures


# =====================================================================
# Positive Test 1 — real sm_86 round-trip from on-disk admitted file.
# =====================================================================


# Codex R58 code-review P1: the 5 activation cells are the
# canonical "real sm_86 5-cell" cohort this test exercises.
# Pre-R58 the filter used a hardcoded historical
# source_commit (0c16e335...) which drifted past the
# recapture chain (current label_admitted is at the R57+
# recapture HEAD). Slug-based filtering is recapture-
# resilient: the 5 activation cells survive every recapture
# cycle and are not dependent on a specific source_commit.
_ACTIVATION_CELL_SLUGS_SHAPES: frozenset[tuple[str, str]] = frozenset({
    ("layernorm_fwd_bf16_sm80", "timing_shape"),
    ("rmsnorm_fwd_bf16_sm80", "timing_shape"),
    ("silu_fwd_bf16_sm80", "correctness_shape"),
    ("silu_fwd_bf16_sm80", "timing_shape"),
    ("softmax_safe_bf16_sm80", "timing_shape"),
})


def test_real_sm86_round_trip_admits_with_quartile_emitted_stats(
    tmp_path: Path,
) -> None:
    # The canonical activation cohort
    # (layernorm/rmsnorm/silu correctness+timing/softmax) was the
    # pre-pivot 5-cell baseline. The sm_86-develop pivot demotes some
    # cells via typed terminal failures (e.g.
    # label_repeatability_failed), so the *live* activation set is a
    # subset. Derive the expected split from live state instead of
    # pinning the stale same-74-era membership: the surviving
    # activation cells must be present in label_admitted.json, and
    # every absent former activation cell must carry a typed terminal
    # failure in the v5 cohort staging manifest. The round-trip
    # assertion itself (quartile_emitted + meets_floor) is preserved
    # unchanged by running it over the full live sm_86 admitted
    # corpus, which is the cohort the harness actually admits.
    cells = _load_live_admitted_cells()
    sm86_cells = [c for c in cells if c.get("arch") == "sm_86"]
    live_keys = {(c.get("slug"), c.get("shape_point")) for c in sm86_cells}

    live_activation = _ACTIVATION_CELL_SLUGS_SHAPES & live_keys
    absent_activation = _ACTIVATION_CELL_SLUGS_SHAPES - live_keys

    # Surviving activation cells are genuinely admitted (sanity floor:
    # the pivot did not evict the entire activation cohort).
    assert live_activation, (
        f"data/local_labels/label_admitted.json carries no live sm_86 "
        f"activation cells from {sorted(_ACTIVATION_CELL_SLUGS_SHAPES)}; "
        f"observed live sm_86 keys {sorted(live_keys)}"
    )

    # Every absent former activation cell must be typed-explained as a
    # terminal failure in the staging manifest — eviction is a typed
    # decision recorded in live state, not silent membership drift.
    staging_typed_failures = _load_sm86_staging_typed_failures()
    for key in sorted(absent_activation):
        assert key in staging_typed_failures, (
            f"former activation cell {key} is absent from "
            f"label_admitted.json but has no TERMINAL_TYPED_FAILURE row "
            f"in v5_cohort_staging_manifest.json; an evicted activation "
            f"cell must carry a typed eviction reason, got staging "
            f"typed-failure keys {sorted(staging_typed_failures)}"
        )
        assert staging_typed_failures[key], (
            f"former activation cell {key} has a TERMINAL_TYPED_FAILURE "
            f"row with an empty action_typed_reason; the eviction must "
            f"name a typed reason"
        )

    # Round-trip over the full live sm_86 admitted corpus. This is the
    # cohort the cross-arch harness admits; it satisfies the per-arch
    # smoke floor (>= 5 live cells) and the quartile threshold
    # (>= 4 headline cells), so the quartile_emitted + meets_floor
    # assertions below are NOT weakened — they assert the same
    # round-trip contract against the live corpus rather than a stale
    # fixed-5 snapshot.
    bundles = [
        _build_predictor_bundle_from_label_cell(
            tmp_path, cell, bundle_subdir=f"b_{i}",
        )
        for i, cell in enumerate(sm86_cells)
    ]
    contract = _baseline_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=sm86_cells,
        bootstrap_seed=0,
    )
    sm86 = out["per_arch_diagnostics"]["per_arch"]["sm_86"]
    n_live = len(sm86_cells)
    assert sm86["n_comparable_cells"] == n_live
    assert sm86["n_headline_cells"] == n_live
    assert sm86["statistics_status"] == "quartile_emitted"
    verdict = sm86["coverage_gate_verdict"]
    assert verdict["headline_count_live"] == n_live
    assert verdict["headline_count_synthetic"] == 0
    assert verdict["meets_floor"] is True
    assert out["result_kind"] == "admitted"


# =====================================================================
# Positive Test 2 — synthetic sm_120 wire-format guard.
# =====================================================================


def test_synthetic_sm120_routes_to_typed_unavailable_with_meets_floor_false(
    tmp_path: Path,
) -> None:
    cells: list[dict] = []
    bundles: list[dict] = []
    for i in range(5):
        slug = f"crossarch_synth_sm120_{i}"
        cell = _make_synthetic_ncu_label_cell(
            slug=slug,
            arch="sm_120",
            ground_truth_pct=50.0,
            evidence_kind="synthetic",
            binary_sha256=("a" * 63) + str(i),
            kernel_signature_sha256=("b" * 63) + str(i),
        )
        cells.append(cell)
        bundles.append(
            _build_predictor_bundle_from_label_cell(
                tmp_path, cell, bundle_subdir=f"b120_{i}",
            )
        )
    contract = _baseline_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    sm120 = out["per_arch_diagnostics"]["per_arch"]["sm_120"]
    assert sm120["n_comparable_cells"] == 5
    assert sm120["n_headline_cells"] == 5
    assert sm120["statistics_status"] == "quartile_emitted"
    verdict = sm120["coverage_gate_verdict"]
    assert verdict["headline_count_live"] == 0
    assert verdict["headline_count_synthetic"] == 5
    assert verdict["meets_floor"] is False
    # No live cells anywhere → typed_unavailable.
    assert out["result_kind"] == "typed_unavailable"


# =====================================================================
# Negative Test 1 — small-n via mismatch.
# =====================================================================


def test_small_n_via_join_key_mismatch_routes_to_insufficient_cells(
    tmp_path: Path,
) -> None:
    """4 sm_86 cells + 4 predictor bundles where ONE bundle has
    binary_sha256 differing by 1 hex character produces:
      - 3 admitted headline cells (the matched pairs).
      - statistics_status == insufficient_cells_for_quartile_statistics.
      - excluded_reason_counts.comparison_runner_join_key_unmatched
        == 2: the bundle-side record (the unmatched bundle whose
        binary_sha256 ends in 'e' instead of '0') AND the cell-side
        record (the unmatched cell whose binary_sha256 ends in '0').
        Both records carry arch=sm_86 in their join_key, so both
        route into per_arch.sm_86 (not the unattributed bucket).
        The cell-side record is the supersession-bookkeeping pass
        that prevents admissible-but-unjoined cells from silently
        disappearing from per_arch_diagnostics."""
    cells: list[dict] = []
    bundles: list[dict] = []
    for i in range(4):
        slug = f"crossarch_smalln_{i}"
        binary_sha = ("c" * 63) + str(i)
        kernel_sig = ("d" * 63) + str(i)
        cell = _make_synthetic_ncu_label_cell(
            slug=slug,
            arch="sm_86",
            ground_truth_pct=50.0,
            evidence_kind="live",
            binary_sha256=binary_sha,
            kernel_signature_sha256=kernel_sig,
        )
        cells.append(cell)
        bundle_binary_sha = (
            ("c" * 63) + ("e" if i == 0 else str(i))
        )
        mpi = _make_synthetic_manifest_pairing_index(
            slug=slug,
            shape_point="trace_00",
            binary_sha256=bundle_binary_sha,
            source_commit="deadbeef" * 5,
            kernel_signature_sha256=kernel_sig,
        )
        bundles.append(
            _make_synthetic_admitted_predictor_bundle(
                tmp_path / f"sm_b_{i}",
                result_kind="scalar",
                predicted_hit_rate=0.5,
                manifest_pairing_index=mpi,
                device="NVIDIA GeForce RTX 3090 Ti",
            )
        )
    contract = _baseline_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    sm86 = out["per_arch_diagnostics"]["per_arch"]["sm_86"]
    assert (
        sm86["excluded_reason_counts"][
            "comparison_runner_join_key_unmatched"
        ]
        == 2
    ), (
        f"expected 2 join_key_unmatched records: 1 bundle-side "
        f"(unmatched bundle) + 1 cell-side (unmatched cell, the "
        f"supersession-bookkeeping pass); got "
        f"{sm86['excluded_reason_counts']!r}"
    )
    assert sm86["n_headline_cells"] == 3
    assert sm86["n_comparable_cells"] == sm86["n_headline_cells"] + 2
    assert (
        sm86["statistics_status"]
        == "insufficient_cells_for_quartile_statistics"
    )
    assert sm86["coverage_gate_verdict"]["meets_floor"] is False


# =====================================================================
# Negative Test 2 — no cells.
# =====================================================================


def test_no_cells_for_arch_produces_no_cells_status(
    tmp_path: Path,
) -> None:
    """A fixture with 0 sm_120 cells produces n_comparable_cells == 0,
    n_headline_cells == 0, statistics_status == 'no_cells', and
    relative_error_stats == None."""
    contract = _baseline_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=[],
        label_cells=[],
        bootstrap_seed=0,
    )
    sm120 = out["per_arch_diagnostics"]["per_arch"]["sm_120"]
    assert sm120["n_comparable_cells"] == 0
    assert sm120["n_headline_cells"] == 0
    assert sm120["statistics_status"] == "no_cells"
    assert sm120["relative_error_stats"] is None


# =====================================================================
# Negative Test 3 — unattributed bucket via missing manifest_pairing_index.
#
# Two variants:
# - Primary: a bundle that lacks ``manifest_pairing_index`` is
#   excluded as ``comparison_contract_drift`` BEFORE join-key
#   resolution; the exclusion record has no ``join_key`` field and
#   tallies into the unattributed bucket. Per-arch records are
#   unaffected.
# - Regression guard: an unmapped-device bundle still produces a
#   ``comparison_runner_join_key_unmatched`` record routed to the
#   unattributed bucket (the join_key.arch resolves to 'unknown',
#   outside ``arch_set``).
# =====================================================================


def test_predictor_bundle_lacking_manifest_pairing_index_tallies_unattributed_contract_drift(
    tmp_path: Path,
) -> None:
    """A predictor bundle that lacks ``manifest_pairing_index``
    entirely cannot have its 7-field join key resolved; the runner
    must emit ``comparison_contract_drift`` with ``unmatched_field
    = "manifest_pairing_index"`` and no synthetic ``join_key`` field
    on the exclusion record. The record tallies into
    ``per_arch_diagnostics.unattributed_excluded_reason_counts.comparison_contract_drift``.

    The single sm_86 admissible cell is unjoined by any bundle (the
    only bundle is malformed), so the cell-side post-join pass
    emits one ``comparison_runner_join_key_unmatched`` record with
    arch=sm_86 in its join_key. That record tallies into
    ``per_arch.sm_86.excluded_reason_counts.comparison_runner_join_key_unmatched``;
    the per-arch contract_drift count remains 0 (the malformed
    bundle's drift is unattributed)."""
    bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path / "missing_mpi",
        result_kind="scalar",
        predicted_hit_rate=0.5,
        device="NVIDIA GeForce RTX 3090 Ti",
    )
    bundle.pop("manifest_pairing_index", None)
    cell = _make_synthetic_ncu_label_cell(
        slug="real_kernel",
        arch="sm_86",
        ground_truth_pct=50.0,
        evidence_kind="live",
    )
    contract = _baseline_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=[bundle],
        label_cells=[cell],
        bootstrap_seed=0,
    )
    excluded = out["excluded_cells"]
    contract_drift_records = [
        e for e in excluded
        if e.get("diagnostic") == "comparison_contract_drift"
    ]
    assert len(contract_drift_records) == 1, (
        f"missing-manifest_pairing_index bundle must produce exactly "
        f"one comparison_contract_drift exclusion; got "
        f"{len(contract_drift_records)}: {contract_drift_records!r}"
    )
    rec = contract_drift_records[0]
    assert rec["reason"] == "comparison_contract_drift"
    assert rec["unmatched_field"] == "manifest_pairing_index"
    assert "join_key" not in rec, (
        f"missing-MPI exclusion record must NOT carry a synthetic "
        f"join_key (the runner has no key to resolve); got {rec!r}"
    )
    unattributed = out["per_arch_diagnostics"][
        "unattributed_excluded_reason_counts"
    ]
    assert unattributed["comparison_contract_drift"] == 1, (
        f"unattributed bucket must record the contract-drift "
        f"exclusion produced by the missing-MPI bundle; got "
        f"{unattributed!r}"
    )
    per_arch = out["per_arch_diagnostics"]["per_arch"]
    for arch, record in per_arch.items():
        assert (
            record["excluded_reason_counts"]["comparison_contract_drift"]
            == 0
        ), (
            f"per_arch.{arch}.excluded_reason_counts.comparison_contract_drift "
            f"must be 0 when the malformed bundle routes to the "
            f"unattributed bucket; got {record['excluded_reason_counts']!r}"
        )
    sm86 = per_arch["sm_86"]
    assert (
        sm86["excluded_reason_counts"][
            "comparison_runner_join_key_unmatched"
        ]
        == 1
    ), (
        f"per_arch.sm_86 must record one cell-side "
        f"comparison_runner_join_key_unmatched for the unjoined "
        f"sm_86 cell (the only bundle was malformed and routed to "
        f"unattributed); got {sm86['excluded_reason_counts']!r}"
    )
    assert sm86["n_headline_cells"] == 0, (
        f"sm_86 admits no headline cells (no bundle joined any "
        f"sm_86 cell); got {sm86['n_headline_cells']}"
    )
    assert sm86["n_comparable_cells"] == 1, (
        f"sm_86's n_comparable_cells equals the cell-side "
        f"join_key_unmatched record count (the unjoined cell is "
        f"comparable but not headline); got "
        f"{sm86['n_comparable_cells']}"
    )


def test_predictor_bundle_with_unmapped_device_tallies_unattributed(
    tmp_path: Path,
) -> None:
    """A predictor bundle whose ``device`` string is not a member of
    the runner's ``_DEVICE_TO_ARCH`` map produces a bundle_key with
    arch == 'unknown'. When that bundle fails to join, the
    resulting bundle-side exclusion record carries arch='unknown'
    — outside arch_set — and routes to the unattributed bucket.

    The single sm_86 admissible cell is unjoined (the bundle's
    arch='unknown' does not match the cell's arch='sm_86'); the
    cell-side post-join pass emits one
    ``comparison_runner_join_key_unmatched`` record with
    arch=sm_86 in its join_key. That record routes into
    ``per_arch.sm_86.excluded_reason_counts`` (NOT the
    unattributed bucket, which only carries the bundle-side
    arch='unknown' record)."""
    bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path / "unmapped_device",
        result_kind="scalar",
        predicted_hit_rate=0.5,
        device="Unmapped Hypothetical GPU",
    )
    cell = _make_synthetic_ncu_label_cell(
        slug="real_kernel",
        arch="sm_86",
        ground_truth_pct=50.0,
        evidence_kind="live",
    )
    contract = _baseline_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=[bundle],
        label_cells=[cell],
        bootstrap_seed=0,
    )
    unattributed = out["per_arch_diagnostics"][
        "unattributed_excluded_reason_counts"
    ]
    assert (
        unattributed["comparison_runner_join_key_unmatched"] >= 1
    ), (
        f"unattributed bucket must record the join-key-unmatched "
        f"exclusion when the bundle's join_key.arch is outside "
        f"arch_set; got {unattributed!r}"
    )
    # The bundle-side record routes to unattributed (arch='unknown'
    # is outside arch_set). The per-arch sm_86 record carries the
    # cell-side post-join record (the admissible sm_86 cell that no
    # bundle joined); the unattributed and per-arch sides are
    # disjoint by arch attribution.
    sm86 = out["per_arch_diagnostics"]["per_arch"]["sm_86"]
    assert (
        sm86["excluded_reason_counts"][
            "comparison_runner_join_key_unmatched"
        ]
        == 1
    ), (
        f"per_arch.sm_86 must record exactly one cell-side "
        f"join_key_unmatched record (the cell's arch=sm_86 routes "
        f"into per_arch.sm_86; the bundle-side arch='unknown' "
        f"routes to unattributed); got {sm86['excluded_reason_counts']!r}"
    )


# =====================================================================
# Negative Test 4 — schema-level rejection BEFORE the runner.
# =====================================================================


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_label_record_missing_source_commit_fails_strict_v3_schema(
    tmp_path: Path,
) -> None:
    """A label record missing source_commit fails strict
    label-admitted v5 schema validation BEFORE reaching the runner.
    Mirrors the existing
    test_synthetic_v2_strict_admitted_profiler_version_drift_routes_to_contract_drift
    pattern."""
    cells = _load_live_admitted_cells()
    broken = copy.deepcopy(cells[0])
    del broken["source_commit"]
    label_admitted_doc = {
        "schema_version": 5,
        "cells": [broken],
    }
    schema = json.loads(
        _LOCAL_LABEL_ADMITTED_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        jsonschema.validate(label_admitted_doc, schema)
    assert "source_commit" in str(excinfo.value)


# =====================================================================
# Cross-arch end-to-end — both arches together in one run.
# =====================================================================


def test_cross_arch_run_carries_both_arches_in_per_arch_diagnostics(
    tmp_path: Path,
) -> None:
    """Mixed run: 5 real sm_86 live cells + 5 synthetic sm_120 cells.
    per_arch_diagnostics.per_arch carries records for BOTH arches;
    the live sm_86 leg admits, the synthetic sm_120 leg does not
    meet its per-arch floor; the top-level result is admitted
    because at least one arch (sm_86) has live cells satisfying the
    floor."""
    sm86_cells = [
        c for c in _load_live_admitted_cells() if c.get("arch") == "sm_86"
    ]
    sm86_bundles = [
        _build_predictor_bundle_from_label_cell(
            tmp_path, cell, bundle_subdir=f"a_{i}",
        )
        for i, cell in enumerate(sm86_cells)
    ]
    sm120_cells: list[dict] = []
    sm120_bundles: list[dict] = []
    for i in range(5):
        slug = f"mixed_synth_sm120_{i}"
        cell = _make_synthetic_ncu_label_cell(
            slug=slug,
            arch="sm_120",
            ground_truth_pct=50.0,
            evidence_kind="synthetic",
            binary_sha256=("e" * 63) + str(i),
            kernel_signature_sha256=("f" * 63) + str(i),
        )
        sm120_cells.append(cell)
        sm120_bundles.append(
            _build_predictor_bundle_from_label_cell(
                tmp_path, cell, bundle_subdir=f"c_{i}",
            )
        )
    contract = _baseline_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=sm86_bundles + sm120_bundles,
        label_cells=sm86_cells + sm120_cells,
        bootstrap_seed=0,
    )
    per_arch = out["per_arch_diagnostics"]["per_arch"]
    assert set(per_arch.keys()) == {"sm_86", "sm_120"}
    assert per_arch["sm_86"]["coverage_gate_verdict"]["meets_floor"] is True
    assert per_arch["sm_120"]["coverage_gate_verdict"]["meets_floor"] is False
    # Top-level admitted because sm_86 leg has 5 live cells.
    assert out["result_kind"] == "admitted"


# =====================================================================
# Schema validation — the on-disk bundle validates against the v1
# comparison_bundle schema after the per_arch_diagnostics extension.
# =====================================================================


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_cross_arch_output_bundle_validates_against_comparison_bundle_schema(
    tmp_path: Path,
) -> None:
    sm86_cells = [
        c for c in _load_live_admitted_cells() if c.get("arch") == "sm_86"
    ]
    bundles = [
        _build_predictor_bundle_from_label_cell(
            tmp_path, cell, bundle_subdir=f"v_{i}",
        )
        for i, cell in enumerate(sm86_cells)
    ]
    contract = _baseline_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=sm86_cells,
        bootstrap_seed=0,
    )
    schema = json.loads(
        _COMPARISON_BUNDLE_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    jsonschema.validate(out, schema)
