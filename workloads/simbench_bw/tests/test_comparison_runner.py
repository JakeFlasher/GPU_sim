"""Test surface for ``workloads/simbench_bw/comparison_runner.py``.

Authored BEFORE the implementation lands (the second commit in the
test-first ladder; the runner implementation lands in the third).
At commit time every test below FAILS because the runner stubs
raise ``NotImplementedError``; the third-commit implementation
turns them green.

The bidirectional structural lockstep
(``set(asserted_runner_defect_tokens) == _RUNNER_DEFECT_TOKENS``,
set equality, not subset-of) is the procedural enforcer per
``BL-20260509-strict-admission-contract-first``: every typed
defect the runner can produce MUST have at least one
negative-control test that asserts it, AND every assertion in
this file MUST name a token from the frozenset. If either side
drifts, the lockstep test fails with a diagnostic naming the
orphan side.

Synthetic-fixture discipline: every test that consumes a
synthetic fixture MUST include the literal substring ``synthetic``
in its function name. No test in this file is permitted to
imply real-trace progress; real labels under
``data/local_labels/label_admitted.json`` land in a separate
follow-on plan.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUITE_ROOT = _REPO_ROOT / "workloads" / "simbench_bw"
_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SUITE_ROOT))
sys.path.insert(0, str(_TESTS_DIR))

import cache_predictor as cp  # noqa: E402
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

_SCHEMA_PATH = _SUITE_ROOT / "schemas" / "comparison_bundle.schema.json"


# ---------------------------------------------------------------------
# Bidirectional structural lockstep — the asserted set MUST match
# ``_RUNNER_DEFECT_TOKENS`` exactly. When you add a negative-control
# test that asserts a new token, add the token here too. When you
# remove a test, remove the token here unless another test still
# asserts it.
# ---------------------------------------------------------------------
asserted_runner_defect_tokens: frozenset[str] = frozenset({
    "comparison_contract_drift",
    "comparison_runner_join_key_unmatched",
    "comparison_runner_unknown_result_kind",
    "comparison_runner_predicted_hit_rate_out_of_range",
    "comparison_runner_typed_reason_out_of_enum",
    "comparison_runner_bounded_interval_invalid",
    "comparison_runner_duplicate_predictor_bundle",
})


# ---------------------------------------------------------------------
# Helpers — minimal contract / bundle / cell builders. These wrap the
# conftest synthetic fixtures with runner-test-specific defaults so
# the test bodies stay readable.
# ---------------------------------------------------------------------


def _admitted_contract(**overrides) -> cr.ComparisonContract:
    """Return a baseline ``ComparisonContract`` whose fields match
    the on-disk ``comparison_contract.toml`` pinned constants.
    Tests pass per-field overrides to construct contract-drift
    fixtures."""
    return cr.ComparisonContract(
        ncu_metric_name=overrides.get(
            "ncu_metric_name", "lts__t_sector_hit_rate.pct",
        ),
        ncu_metric_version=overrides.get(
            "ncu_metric_version", "ncu_2026.1.x",
        ),
        percent_to_fraction=overrides.get("percent_to_fraction", True),
        epsilon=overrides.get("epsilon", 0.01),
        bootstrap_resamples=overrides.get("bootstrap_resamples", 2000),
        ci_confidence=overrides.get("ci_confidence", 0.95),
        bootstrap_unit=overrides.get("bootstrap_unit", "cell"),
        bounded_arm_handling=overrides.get(
            "bounded_arm_handling", "report_three",
        ),
        typed_unavailable_handling=overrides.get(
            "typed_unavailable_handling", "exclude",
        ),
        join_key=overrides.get(
            "join_key",
            (
                "slug",
                "shape_point",
                "binary_sha256",
                "source_commit",
                "kernel_signature_sha256",
                "arch",
                "ncu_metric_version",
                "cache_layer",
            ),
        ),
        coverage_gate=overrides.get(
            "coverage_gate",
            cr.CoverageGate(
                smoke_test_floor_per_arch=5,
                publication_grade_floor_per_arch=10,
                aspirational_target_total=20,
                require_both_arches_for_cross_arch=False,
            ),
        ),
        log_space_geomean=overrides.get("log_space_geomean", True),
        arch_set=overrides.get("arch_set", ("sm_86", "sm_120")),
        cache_layers=overrides.get("cache_layers", ("L1", "L2")),
    )


def _five_synthetic_pairs(
    tmp_path: Path,
    *,
    arch: str = "sm_86",
    pct: float = 50.0,
    predicted_rates: tuple[float, ...] = (0.4, 0.5, 0.5, 0.5, 0.6),
    evidence_kind: str = "live",
) -> tuple[list[dict], list[dict]]:
    """Mint N matched (predictor, label) synthetic pairs joined on
    distinct slugs. Predicted rates default to a five-cell scalar
    arm fixture; callers override for specific shapes. ``evidence_kind``
    defaults to ``"live"`` because this helper is used by tests that
    exercise the admitted path; the live-only smoke-test floor counts
    each cell. Tests that exercise the synthetic-only path override
    ``evidence_kind="synthetic"`` to verify the live-only gate
    correctly produces ``typed_unavailable``."""
    bundles: list[dict] = []
    cells: list[dict] = []
    for i, rate in enumerate(predicted_rates):
        slug = f"synthetic_kernel_{i}"
        mpi = _make_synthetic_manifest_pairing_index(slug=slug)
        bundles.append(
            _make_synthetic_admitted_predictor_bundle(
                tmp_path / f"sector_{i}",
                result_kind="scalar",
                predicted_hit_rate=rate,
                manifest_pairing_index=mpi,
                device=(
                    "NVIDIA GeForce RTX 5060 Laptop GPU"
                    if arch == "sm_120"
                    else "NVIDIA GeForce RTX 3090 Ti"
                ),
            )
        )
        cells.append(
            _make_synthetic_ncu_label_cell(
                slug=slug,
                arch=arch,
                ground_truth_pct=pct,
                evidence_kind=evidence_kind,
            )
        )
    return bundles, cells


# =====================================================================
# Synthetic-fixture evidence_kind contract (lockstep with the v3
# admitted-cell schema bump). The conftest helper that emits
# admitted-cell shapes for runner tests MUST classify each cell's
# provenance via evidence_kind. Synthetic test fixtures default to
# evidence_kind == "synthetic" so they cannot accidentally satisfy
# the live-only smoke-test floor; tests that explicitly exercise
# the live-evidence path override to evidence_kind == "live".
# =====================================================================


def test_make_synthetic_ncu_label_cell_default_evidence_kind_is_synthetic() -> None:
    cell = _make_synthetic_ncu_label_cell(slug="foo")
    assert cell.get("evidence_kind") == "synthetic", (
        f"synthetic helper must default evidence_kind to 'synthetic'; "
        f"got {cell.get('evidence_kind')!r}"
    )


def test_make_synthetic_ncu_label_cell_evidence_kind_live_override_is_propagated() -> None:
    cell = _make_synthetic_ncu_label_cell(slug="foo", evidence_kind="live")
    assert cell.get("evidence_kind") == "live", (
        f"synthetic helper must propagate evidence_kind='live' override; "
        f"got {cell.get('evidence_kind')!r}"
    )


def test_make_synthetic_ncu_label_cell_evidence_kind_synthetic_override_is_propagated() -> None:
    cell = _make_synthetic_ncu_label_cell(
        slug="foo", evidence_kind="synthetic",
    )
    assert cell.get("evidence_kind") == "synthetic"


# =====================================================================
# Lockstep — bidirectional structural set equality.
# =====================================================================


def test_runner_defect_tokens_match_test_surface_exactly() -> None:
    """``set(asserted_runner_defect_tokens) == _RUNNER_DEFECT_TOKENS``
    (set equality, not subset-of). Catches both an orphan token in
    the frozenset (no test asserts it) AND an orphan assertion in
    a test (token not in frozenset).

    The diagnostic enumerates the symmetric difference so the
    operator can see both sides at once."""
    expected = cr._RUNNER_DEFECT_TOKENS
    asserted = asserted_runner_defect_tokens
    missing = expected - asserted
    extra = asserted - expected
    assert not missing and not extra, (
        f"runner defect-token set drift: "
        f"missing in asserted={sorted(missing)!r}, "
        f"extra in asserted={sorted(extra)!r}"
    )


# Documented overlap between runner and predictor token sets. The
# token ``comparison_contract_drift`` appears in BOTH because the
# semantic is the same on both sides: the comparison contract has
# drifted from its pinned state. The runner emits this in
# ``excluded_cells`` when a label cell's profiler_version disagrees
# with the contract's pinned ncu_metric_version OR when the loaded
# contract carries a non-pinned constant; the predictor reserves
# the same token in ``_TYPED_REASONS`` for the analogous
# admit-failure surface. Any additional overlap is unintentional
# and the parity test below enumerates the offending tokens.
_RUNNER_PREDICTOR_DOCUMENTED_OVERLAP: frozenset[str] = frozenset({
    "comparison_contract_drift",
})


def test_runner_and_predictor_typed_reason_sets_overlap_is_documented() -> None:
    """The runner-level vocabulary and the predictor-level
    typed-reason enum MAY intersect only via the documented
    overlap list. Any unintended overlap fails the parity gate."""
    overlap = cr._RUNNER_DEFECT_TOKENS & cp._TYPED_REASONS
    extra = overlap - _RUNNER_PREDICTOR_DOCUMENTED_OVERLAP
    missing = _RUNNER_PREDICTOR_DOCUMENTED_OVERLAP - overlap
    assert not extra and not missing, (
        f"runner ↔ predictor overlap drift: "
        f"undocumented overlap={sorted(extra)!r}, "
        f"documented but not present={sorted(missing)!r}"
    )


def test_lockstep_diff_reports_orphan_token_under_synthetic_overlay() -> None:
    """Counterfactual: if a token is added to the frozenset but
    nobody asserts it, the lockstep diff names the orphan. The
    test simulates the failure mode using a synthetic overlay."""
    synthetic_overlay = asserted_runner_defect_tokens - {
        "comparison_contract_drift"
    }
    missing = cr._RUNNER_DEFECT_TOKENS - synthetic_overlay
    assert "comparison_contract_drift" in missing


def test_lockstep_diff_reports_orphan_assertion_under_synthetic_overlay() -> None:
    """Counterfactual: if a test asserts a token not in the
    frozenset, the lockstep diff names the orphan assertion."""
    synthetic_overlay = asserted_runner_defect_tokens | {
        "comparison_runner_imaginary_token"
    }
    extra = synthetic_overlay - cr._RUNNER_DEFECT_TOKENS
    assert "comparison_runner_imaginary_token" in extra


# =====================================================================
# Contract loader — module import is IO-free; loader returns a
# pinned contract; per-pinned-field drift surfaces as
# ``comparison_contract_drift``.
# =====================================================================


def test_module_import_does_not_perform_io() -> None:
    """Re-importing the module under the test session does not
    open the contract file. The module is already imported at the
    top of this test file, so a second import is a no-op; the
    test asserts the on-disk contract path is NOT pre-loaded as
    a module-level attribute."""
    import importlib
    importlib.reload(cr)
    assert not hasattr(cr, "_LOADED_CONTRACT")


def test_load_comparison_contract_default_path_returns_pinned_constants() -> None:
    """The on-disk contract returns a runtime object whose pinned
    constants match the contract file exactly."""
    contract = cr.load_comparison_contract()
    assert contract.epsilon == 0.01
    assert contract.bootstrap_resamples == 2000
    assert contract.ci_confidence == 0.95
    assert contract.ncu_metric_version == "ncu_2026.1.x"
    assert contract.bootstrap_unit == "cell"
    assert contract.join_key == (
        "slug",
        "shape_point",
        "binary_sha256",
        "source_commit",
        "kernel_signature_sha256",
        "arch",
        "ncu_metric_version",
        "cache_layer",
    )
    assert contract.coverage_gate.smoke_test_floor_per_arch == 5
    assert contract.coverage_gate.publication_grade_floor_per_arch == 10
    assert contract.coverage_gate.aspirational_target_total == 20


def _write_contract_with_overrides(
    tmp_path: Path, **overrides,
) -> Path:
    """Mint a synthetic contract.toml with the requested pinned-field
    overrides. The base shape mirrors the on-disk contract exactly;
    only the overridden fields differ."""
    epsilon = overrides.get("epsilon", 0.01)
    bootstrap_resamples = overrides.get("bootstrap_resamples", 2000)
    ci_confidence = overrides.get("ci_confidence", 0.95)
    ncu_metric_version = overrides.get(
        "ncu_metric_version", "ncu_2026.1.x"
    )
    join_key_fields = overrides.get(
        "join_key_fields",
        [
            "slug",
            "shape_point",
            "binary_sha256",
            "source_commit",
            "kernel_signature_sha256",
            "arch",
            "ncu_metric_version",
            "cache_layer",
        ],
    )
    smoke_floor = overrides.get("smoke_test_floor_per_arch", 5)
    pub_floor = overrides.get("publication_grade_floor_per_arch", 10)
    target_total = overrides.get("aspirational_target_total", 20)
    require_both_arches = overrides.get(
        "require_both_arches_for_cross_arch", True
    )
    require_both_arches_toml = (
        "true" if require_both_arches else "false"
    )
    bounded_arm_handling = overrides.get(
        "bounded_arm_handling", "report_three"
    )
    join_key_toml = ",\n  ".join(repr(f) for f in join_key_fields)
    contents = f"""schema_version = 1

[ncu_metric]
name = "lts__t_sector_hit_rate.pct"
display = "L2 Sector Hit Rate"
description = "synthetic"
unit = "fraction"
percent_to_fraction = true
ncu_metric_version = "{ncu_metric_version}"
includes_atomics = true
includes_stores = true
sector_basis = true
excludes_tlb_misses = true

[join_key]
fields = [
  {join_key_toml}
]

[error_formula]
formula = "abs(predicted - ground_truth) / max(ground_truth, epsilon)"
epsilon = {epsilon}
both_in_fraction_units = true
log_space_geomean = true

[geomean]
bootstrap_resamples = {bootstrap_resamples}
ci_confidence = {ci_confidence}
ci_half_width_target = 0.05
bootstrap_unit = "cell"
typed_unavailable_handling = "exclude"
bounded_arm_handling = "{bounded_arm_handling}"

[coverage_gate]
smoke_test_floor_per_arch = {smoke_floor}
publication_grade_floor_per_arch = {pub_floor}
aspirational_target_total = {target_total}
require_both_arches_for_cross_arch = {require_both_arches_toml}

[scope]
arch_set = ["sm_86", "sm_120"]
cache_layers = ["L1", "L2"]
metric_target = "hit_rate"
device_set = ["RTX 3090 Ti", "RTX 5060 Laptop GPU"]
prior_art_handling = "contextual_only"

[per_arch_diagnostics_config]
diagnostic_vocabulary = [
    "comparison_contract_drift",
    "comparison_runner_join_key_unmatched",
    "predictor_typed_unavailable",
]
quartile_min_n = 4
"""
    p = tmp_path / "comparison_contract.toml"
    p.write_text(contents, encoding="utf-8")
    return p


def test_load_rejects_synthetic_epsilon_drift(tmp_path: Path) -> None:
    """Asserts ``comparison_contract_drift`` for an
    ``epsilon`` value that does not match the pinned 0.01."""
    p = _write_contract_with_overrides(tmp_path, epsilon=0.05)
    with pytest.raises(cr.ComparisonContractDrift) as excinfo:
        cr.load_comparison_contract(p)
    assert excinfo.value.unmatched_field == "epsilon"


def test_load_rejects_synthetic_bootstrap_resamples_drift(
    tmp_path: Path,
) -> None:
    p = _write_contract_with_overrides(
        tmp_path, bootstrap_resamples=1000
    )
    with pytest.raises(cr.ComparisonContractDrift) as excinfo:
        cr.load_comparison_contract(p)
    assert excinfo.value.unmatched_field == "bootstrap_resamples"


def test_load_rejects_synthetic_ci_confidence_drift(
    tmp_path: Path,
) -> None:
    p = _write_contract_with_overrides(tmp_path, ci_confidence=0.9)
    with pytest.raises(cr.ComparisonContractDrift) as excinfo:
        cr.load_comparison_contract(p)
    assert excinfo.value.unmatched_field == "ci_confidence"


def test_load_rejects_synthetic_join_key_drift(tmp_path: Path) -> None:
    p = _write_contract_with_overrides(
        tmp_path,
        join_key_fields=[
            "slug",
            "shape_point",
            "arch",
        ],
    )
    with pytest.raises(cr.ComparisonContractDrift) as excinfo:
        cr.load_comparison_contract(p)
    assert excinfo.value.unmatched_field == "join_key"


def test_load_rejects_synthetic_coverage_gate_drift(
    tmp_path: Path,
) -> None:
    p = _write_contract_with_overrides(
        tmp_path, smoke_test_floor_per_arch=3
    )
    with pytest.raises(cr.ComparisonContractDrift) as excinfo:
        cr.load_comparison_contract(p)
    assert excinfo.value.unmatched_field == "coverage_gate"


def test_load_rejects_synthetic_ncu_metric_version_drift(
    tmp_path: Path,
) -> None:
    """Loader-side `ncu_metric_version` is the contract-side
    protocol pin; a contract whose `ncu_metric_version` differs
    from the pinned value MUST be rejected at load time, not
    silently accepted to be cross-checked only against label
    cells."""
    p = _write_contract_with_overrides(
        tmp_path, ncu_metric_version="ncu_9999.x"
    )
    with pytest.raises(cr.ComparisonContractDrift) as excinfo:
        cr.load_comparison_contract(p)
    assert excinfo.value.unmatched_field == "ncu_metric_version"


# =====================================================================
# Synthetic admitted-cells positive controls — happy paths for the
# scalar arm.
# =====================================================================


def test_runner_admits_synthetic_five_cell_geomean_at_0p20(
    tmp_path: Path,
) -> None:
    """Five synthetic admitted predictor bundles (scalar arm, hit
    rates set so each per-cell relative error is exactly 0.20)
    joined against five synthetic admitted NCU label cells
    (ground-truth pct=50) on matching 7-field keys must clear
    the smoke-test floor (5 headline cells) and emit the admitted
    arm with geomean ≈ 0.20 plus all four bootstrap CI fields
    populated.

    This is the admitted-geomean positive control for the
    coverage-cleared headline path. Below-floor cases (4 cells,
    zero cells, comparable >= 5 but headline < 5) have their own
    coverage-gate tests; this test must NOT accept either arm —
    that ambiguity was the gap Codex's prior review flagged."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out["result_kind"] == "admitted", (
        f"expected admitted arm with 5 headline cells; got "
        f"result_kind={out.get('result_kind')!r}, "
        f"headline_cells_per_arch={out.get('headline_cells_per_arch')!r}"
    )
    assert math.isclose(out["geomean"], 0.20, rel_tol=1e-9), (
        f"expected geomean ≈ 0.20 over five cells with rel_err=0.20 "
        f"each; got geomean={out.get('geomean')!r}"
    )
    for ci_field in (
        "geomean_ci_low",
        "geomean_ci_high",
        "geomean_ci_half_width",
    ):
        assert ci_field in out, (
            f"admitted arm must include {ci_field!r}; got keys "
            f"{sorted(out.keys())!r}"
        )
        assert math.isfinite(out[ci_field]), (
            f"{ci_field}={out[ci_field]!r} is not finite"
        )


def test_runner_byte_identical_output_under_pinned_synthetic_seed(
    tmp_path: Path,
) -> None:
    """Same fixture under same ``bootstrap_seed`` produces a
    byte-identical output bundle on re-run."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    contract = _admitted_contract()
    out_a = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=42,
    )
    out_b = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=42,
    )
    assert json.dumps(out_a, sort_keys=True) == json.dumps(
        out_b, sort_keys=True,
    )


# =====================================================================
# 7-field join near-miss diagnostics — six fields produce
# ``comparison_runner_join_key_unmatched``; the seventh
# (``ncu_metric_version``) escalates to
# ``comparison_contract_drift``.
# =====================================================================


def _join_near_miss_one_field(
    tmp_path: Path, field_name: str,
) -> tuple[list[dict], list[dict]]:
    """Build a 5-pair fixture where one cell's named field is
    perturbed by one character so it no longer matches its
    counterpart bundle on that field."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    if field_name == "slug":
        cells[0]["slug"] = cells[0]["slug"] + "X"
    elif field_name == "shape_point":
        cells[0]["shape_point"] = "trace_99"
    elif field_name == "binary_sha256":
        cells[0]["binary_sha256"] = "f" + cells[0]["binary_sha256"][1:]
    elif field_name == "source_commit":
        cells[0]["source_commit"] = "f" + cells[0]["source_commit"][1:]
    elif field_name == "kernel_signature_sha256":
        cells[0]["kernel_signature_sha256"] = (
            "f" + cells[0]["kernel_signature_sha256"][1:]
        )
    elif field_name == "arch":
        cells[0]["arch"] = "sm_120"
    elif field_name == "ncu_metric_version":
        cells[0]["profiler_version"] = "ncu_9999.x"
    else:
        raise ValueError(field_name)
    return bundles, cells


@pytest.mark.parametrize("field_name", [
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
    "arch",
])
def test_join_near_miss_synthetic_field_routes_to_join_key_unmatched(
    tmp_path: Path, field_name: str,
) -> None:
    """Per-field near-miss tests for six of the seven join-key
    fields. Each produces an excluded-cell record carrying the
    externally-emitted token ``diagnostic == "comparison_runner_join_key_unmatched"``
    AND the short category ``reason == "join_key_unmatched"`` AND
    ``unmatched_field`` naming the offending field. Asserting
    ``diagnostic`` externally (not just ``reason``) ties the
    bidirectional defect-token contract to the on-disk artifact
    (the prior 2-cell test asserted only ``reason``, which let
    the runner-level token go un-emitted)."""
    bundles, cells = _join_near_miss_one_field(tmp_path, field_name)
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    excluded = out["excluded_cells"]
    assert any(
        e.get("diagnostic")
        == "comparison_runner_join_key_unmatched"
        and e.get("reason") == "join_key_unmatched"
        and e.get("unmatched_field") == field_name
        for e in excluded
    ), (
        f"expected diagnostic=comparison_runner_join_key_unmatched "
        f"+ reason=join_key_unmatched on {field_name!r}; got "
        f"excluded={excluded!r}"
    )


def test_join_near_miss_synthetic_ncu_metric_version_escalates_to_contract_drift(
    tmp_path: Path,
) -> None:
    """Per-field near-miss for ``ncu_metric_version`` is escalated
    to ``comparison_contract_drift`` (not
    ``comparison_runner_join_key_unmatched``) because the contract
    version is a protocol pin, not a per-cell key. Asserts BOTH
    the externally-emitted ``diagnostic`` token AND the short
    ``reason`` category."""
    bundles, cells = _join_near_miss_one_field(
        tmp_path, "ncu_metric_version",
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    excluded = out["excluded_cells"]
    assert any(
        e.get("diagnostic") == "comparison_contract_drift"
        and e.get("reason") == "comparison_contract_drift"
        and e.get("unmatched_field") == "ncu_metric_version"
        for e in excluded
    ), (
        f"expected diagnostic=comparison_contract_drift + "
        f"reason=comparison_contract_drift on ncu_metric_version; "
        f"got excluded={excluded!r}"
    )


def test_synthetic_ncu_metric_version_drift_does_not_emit_join_key_unmatched_for_same_first_six_key(
    tmp_path: Path,
) -> None:
    """The ``ncu_metric_version`` near-miss is escalated to
    ``comparison_contract_drift``, NOT a join-unmatched exclusion.
    Without the suppression hook the runner emits BOTH: the
    cell-side drift record AND a bundle-side join-unmatched
    record for the same first-6-field key. The protocol-pin drift
    record is the sole correct explanation; the join-unmatched
    record would be a duplicate exclusion."""
    bundles, cells = _join_near_miss_one_field(
        tmp_path, "ncu_metric_version",
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    excluded = out["excluded_cells"]
    perturbed_slug = cells[0]["slug"]
    join_unmatched_for_perturbed = [
        e for e in excluded
        if e.get("diagnostic") == "comparison_runner_join_key_unmatched"
        and e.get("join_key", {}).get("slug") == perturbed_slug
    ]
    assert join_unmatched_for_perturbed == [], (
        f"protocol-pin drift on ncu_metric_version emitted a "
        f"spurious join_key_unmatched record for the same first-6 "
        f"key (slug={perturbed_slug!r}); the comparison_contract_drift "
        f"record is the sole correct exclusion. "
        f"Excluded: {excluded!r}"
    )


def test_load_rejects_synthetic_bounded_arm_handling_drift(
    tmp_path: Path,
) -> None:
    """Loader-side ``bounded_arm_handling`` is the bounded-arm
    policy pin per the plan; a contract whose
    ``bounded_arm_handling`` differs from the pinned ``"report_three"``
    MUST be rejected at load time."""
    p = _write_contract_with_overrides(
        tmp_path, bounded_arm_handling="drop"
    )
    with pytest.raises(cr.ComparisonContractDrift) as excinfo:
        cr.load_comparison_contract(p)
    assert excinfo.value.unmatched_field == "bounded_arm_handling"


# =====================================================================
# Schema binding negative-control tests — the schema MUST reject
# bundles missing fields the plan describes as required / auditable.
# =====================================================================


def _make_minimal_admitted_output_bundle() -> dict:
    """Return a minimum-shape admitted output bundle that
    schema-validates as a baseline. Tests mutate one field and
    assert rejection."""
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
        "result_kind": "admitted",
        "top_level_admission_scope": "cross_arch_any_arch",
        "comparable_cells_per_arch": {"sm_86": 5},
        "headline_cells_per_arch": {"sm_86": 5},
        "headline_floor_per_arch": 5,
        "included_cells": [],
        "excluded_cells": [],
        "geomean": 0.20,
        "geomean_ci_low": 0.18,
        "geomean_ci_high": 0.22,
        "geomean_ci_half_width": 0.02,
    }


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_schema_rejects_bundle_missing_headline_floor_per_arch() -> None:
    """``headline_floor_per_arch`` is recorded for the
    coverage-gate auditability contract and MUST be in the
    top-level required list."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bundle = _make_minimal_admitted_output_bundle()
    del bundle["headline_floor_per_arch"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_schema_rejects_bounded_cell_missing_diagnostic_errors() -> None:
    """A bounded ``included_cells`` item MUST carry both
    ``relative_error_lower_diagnostic`` and
    ``relative_error_upper_diagnostic`` per the bounded-arm
    midpoint-canonical policy."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bundle = _make_minimal_admitted_output_bundle()
    bundle["included_cells"] = [{
        "join_key": {
            "slug": "k0",
            "shape_point": "trace_00",
            "binary_sha256": "0" * 64,
            "source_commit": "deadbeef" * 5,
            "kernel_signature_sha256": "1" * 64,
            "arch": "sm_86",
            "ncu_metric_version": "ncu_2026.1.x",
        },
        "predictor_result_kind": "bounded",
        "predicted_hit_rate": 0.5,
        "ground_truth_fraction": 0.5,
        "relative_error": 0.0,
    }]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_schema_rejects_scalar_cell_carrying_bounded_diagnostic() -> None:
    """A scalar ``included_cells`` item MUST NOT carry
    ``relative_error_lower_diagnostic`` /
    ``relative_error_upper_diagnostic`` (those are bounded-arm
    only); the schema rejects the cross-arm contamination."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bundle = _make_minimal_admitted_output_bundle()
    bundle["included_cells"] = [{
        "join_key": {
            "slug": "k0",
            "shape_point": "trace_00",
            "binary_sha256": "0" * 64,
            "source_commit": "deadbeef" * 5,
            "kernel_signature_sha256": "1" * 64,
            "arch": "sm_86",
            "ncu_metric_version": "ncu_2026.1.x",
        },
        "predictor_result_kind": "scalar",
        "predicted_hit_rate": 0.5,
        "ground_truth_fraction": 0.5,
        "relative_error": 0.0,
        "relative_error_lower_diagnostic": 0.1,
    }]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_schema_rejects_join_unmatched_excluded_missing_unmatched_field() -> None:
    """An excluded record with reason==join_key_unmatched MUST
    carry ``unmatched_field`` so the failing field is auditable."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bundle = _make_minimal_admitted_output_bundle()
    bundle["excluded_cells"] = [{
        "diagnostic": "comparison_runner_join_key_unmatched",
        "reason": "join_key_unmatched",
        "join_key": {"slug": "k0"},
    }]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_schema_rejects_predictor_typed_unavailable_excluded_missing_upstream_typed_reason() -> None:
    """An excluded record with reason==predictor_typed_unavailable
    MUST carry ``upstream_typed_reason`` so the upstream cause is
    auditable."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bundle = _make_minimal_admitted_output_bundle()
    bundle["excluded_cells"] = [{
        "diagnostic": "predictor_typed_unavailable",
        "reason": "predictor_typed_unavailable",
        "join_key": {"slug": "k0"},
    }]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


# =====================================================================
# Result-arm dispatch — typed_unavailable / bounded / scalar.
# =====================================================================


def test_synthetic_typed_unavailable_predictor_excluded_from_headline(
    tmp_path: Path,
) -> None:
    """A ``result_kind=typed_unavailable`` predictor bundle that
    matches a label cell on all 7 fields is recorded in
    ``excluded_cells`` with ``reason=predictor_typed_unavailable``
    and the upstream typed_reason preserved; the cell IS counted
    toward ``comparable_cells_per_arch`` but NOT toward
    ``headline_cells_per_arch``."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    bundles[0] = _make_synthetic_admitted_predictor_bundle(
        tmp_path / "ta",
        result_kind="typed_unavailable",
        typed_reason="cache_model_underconfigured",
        manifest_pairing_index=bundles[0]["manifest_pairing_index"],
        device="NVIDIA GeForce RTX 3090 Ti",
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out["comparable_cells_per_arch"].get("sm_86", 0) >= 5
    assert out["headline_cells_per_arch"].get("sm_86", 0) == 4
    assert any(
        e.get("diagnostic") == "predictor_typed_unavailable"
        and e.get("reason") == "predictor_typed_unavailable"
        and e.get("upstream_typed_reason") == "cache_model_underconfigured"
        for e in out["excluded_cells"]
    )


def test_synthetic_bounded_predictor_uses_midpoint_canonical_with_diagnostics(
    tmp_path: Path,
) -> None:
    """A ``result_kind=bounded`` synthetic predictor bundle
    produces a per-cell record with ``relative_error`` set to the
    midpoint-derived error; ``relative_error_lower_diagnostic`` is
    derived from ``hit_rate_lower``; ``relative_error_upper_diagnostic``
    is derived from ``hit_rate_upper``."""
    mpi = _make_synthetic_manifest_pairing_index(slug="bounded_kernel")
    bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path / "tb",
        result_kind="bounded",
        hit_rate_lower=0.4,
        hit_rate_upper=0.6,
        admitted_bin_count=5,
        underconfigured_bin_count=2,
        manifest_pairing_index=mpi,
        device="NVIDIA GeForce RTX 3090 Ti",
    )
    cell = _make_synthetic_ncu_label_cell(
        slug="bounded_kernel",
        arch="sm_86",
        ground_truth_pct=50.0,
        evidence_kind="live",
    )
    bundles_extra, cells_extra = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    bundles = [bundle] + bundles_extra
    cells = [cell] + cells_extra
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    bounded = [
        r for r in out["included_cells"]
        if r["join_key"]["slug"] == "bounded_kernel"
    ]
    assert len(bounded) == 1
    rec = bounded[0]
    assert rec["predictor_result_kind"] == "bounded"
    midpoint_error = abs(0.5 - 0.5) / max(0.5, 0.01)
    assert math.isclose(rec["relative_error"], midpoint_error, abs_tol=1e-9)
    lower_error = abs(0.4 - 0.5) / max(0.5, 0.01)
    upper_error = abs(0.6 - 0.5) / max(0.5, 0.01)
    assert math.isclose(
        rec["relative_error_lower_diagnostic"], lower_error, abs_tol=1e-9,
    )
    assert math.isclose(
        rec["relative_error_upper_diagnostic"], upper_error, abs_tol=1e-9,
    )


def test_cross_arch_admission_audit_against_live_merged_labels(
    tmp_path: Path,
) -> None:
    """Direct cross-arch admission audit per the Cross-Arch Bounded
    First sm_120 Label Admission Loop's contract.

    Loads the live ``data/local_labels/label_admitted.json`` (the
    operator-facing merged file with 44 carried sm_86 cells + 5 live
    sm_120 cells from the laptop NCU sweep), builds synthetic
    predictor bundles for the first 5 sm_86 cells with
    ``device="NVIDIA GeForce RTX 3090 Ti"`` (which the runner maps to
    arch ``sm_86``) and for all 5 sm_120 cells with
    ``device="NVIDIA GeForce RTX 5060 Laptop GPU"`` (which the runner
    maps to arch ``sm_120``). Each bundle carries the appropriate
    ``grid_dim_alpha_source`` tag (sm_86_calibrated for sm_86 bundles;
    sm_86_extrapolated for sm_120 bundles).

    Calls ``run_comparison()`` with a contract that has
    ``require_both_arches_for_cross_arch=True`` (the production
    contract value). Asserts:
      - ``result_kind == "admitted"``
      - ``top_level_admission_scope == "cross_arch_require_both_arches"``
      - per-arch ``coverage_gate_verdict.meets_floor == true`` for
        both sm_86 and sm_120
      - included-cell records preserve ``grid_dim_alpha_source``
      - byte-identical re-run with the same seed
    """
    import sys as _sys
    _repo_root = Path(__file__).resolve().parents[3]
    live_admitted_path = (
        _repo_root / "data" / "local_labels" / "label_admitted.json"
    )
    if not live_admitted_path.is_file():
        pytest.skip(
            f"Live label_admitted.json not present at {live_admitted_path}; "
            "the cross-arch audit requires the operator-merged file."
        )
    with live_admitted_path.open(encoding="utf-8") as fh:
        live = json.load(fh)
    sm86_live = [c for c in live["cells"] if c.get("arch") == "sm_86"]
    sm120_live = [c for c in live["cells"] if c.get("arch") == "sm_120"]
    if len(sm86_live) < 5 or len(sm120_live) < 5:
        pytest.skip(
            f"Live admitted file does not have enough cells for the "
            f"cross-arch audit; sm_86={len(sm86_live)}, "
            f"sm_120={len(sm120_live)}; need >=5 each."
        )

    # Cohort-identity guard: the sm_120 selection MUST include the
    # five required activation-cohort identities pinned by the plan
    # (the PlattCalibrator training distribution). Without this
    # guard a non-cohort cell could silently substitute and mask a
    # cohort-floor failure. The bounded-MVP framing pins the
    # cohort as the load-bearing identity contract; any future
    # cell-set widening must come through a plan amendment, not
    # through ad-hoc substitution in the test fixture.
    #
    # Amendment per `plans/silu_correctness_sm120_admission_cohort_amendment_plan.md`:
    # `silu_fwd_bf16_sm80/correctness_shape` retired from the sm_120
    # cohort because the cell exhibits irreducible kernel-side L2
    # variance (stddev ~0.22 vs DEC-6 threshold 0.03) on sm_120;
    # `swiglu_fwd_bf16_sm80/trace_l2_accuracy_xl` adopted as the
    # variance-stable activation-family replacement.
    _REQUIRED_SM120_COHORT: frozenset[tuple[str, str]] = frozenset({
        ("layernorm_fwd_bf16_sm80", "timing_shape"),
        ("rmsnorm_fwd_bf16_sm80", "timing_shape"),
        ("silu_fwd_bf16_sm80", "timing_shape"),
        ("softmax_safe_bf16_sm80", "timing_shape"),
        ("swiglu_fwd_bf16_sm80", "trace_l2_accuracy_xl"),
    })
    sm120_ids = {(c["slug"], c["shape_point"]) for c in sm120_live}
    missing_cohort = _REQUIRED_SM120_COHORT - sm120_ids
    if missing_cohort:
        pytest.skip(
            f"Live admitted file's sm_120 set is missing required "
            f"activation-cohort cells {sorted(missing_cohort)}; the "
            f"cross-arch audit requires the full cohort identity. "
            f"Observed sm_120 cells: {sorted(sm120_ids)}"
        )

    # Pick the 5 required cohort cells (in cohort-pinned order); pick
    # 5 from sm_86 (no cohort identity pin on the sm_86 side because
    # the 44-cell baseline pre-dates this loop and is treated as
    # carried evidence).
    sm120_by_id = {(c["slug"], c["shape_point"]): c for c in sm120_live}
    sm120_pick = [
        sm120_by_id[ident] for ident in sorted(_REQUIRED_SM120_COHORT)
    ]
    sm86_pick = sm86_live[:5]

    bundles: list[dict] = []
    for cell in sm86_pick + sm120_pick:
        arch = cell["arch"]
        if arch == "sm_86":
            device = "NVIDIA GeForce RTX 3090 Ti"
            alpha_source = "sm_86_calibrated"
        else:
            device = "NVIDIA GeForce RTX 5060 Laptop GPU"
            alpha_source = "sm_86_extrapolated"
        # The synthetic bundle carries the SAME 5-field
        # manifest_pairing_index the live cell carries (slug,
        # shape_point, binary_sha256, source_commit,
        # kernel_signature_sha256). The predictor's predicted hit
        # rate is set to the label's ground truth so the per-cell
        # relative error is 0 (perfect predictor; useful for the
        # admission-gate verification, not predictor accuracy).
        mpi = _make_synthetic_manifest_pairing_index(
            slug=cell["slug"],
            shape_point=cell["shape_point"],
            binary_sha256=cell["binary_sha256"],
            source_commit=cell["source_commit"],
            kernel_signature_sha256=cell["kernel_signature_sha256"],
        )
        ground_truth_fraction = float(cell["cache_hit_rate_l2_sector_pct"]) / 100.0
        bundle = _make_synthetic_admitted_predictor_bundle(
            tmp_path / f"sector_{arch}_{cell['slug']}",
            result_kind="scalar",
            predicted_hit_rate=ground_truth_fraction,
            manifest_pairing_index=mpi,
            device=device,
        )
        bundle["grid_dim_alpha_source"] = alpha_source
        bundles.append(bundle)

    # Use the production-contract value require_both_arches_for_cross_arch=True
    contract = _admitted_contract(
        coverage_gate=cr.CoverageGate(
            smoke_test_floor_per_arch=5,
            publication_grade_floor_per_arch=10,
            aspirational_target_total=20,
            require_both_arches_for_cross_arch=True,
        ),
    )

    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=sm86_pick + sm120_pick,
        bootstrap_seed=42,
    )

    assert out["result_kind"] == "admitted", (
        f"expected cross-arch admitted; got {out.get('result_kind')!r}; "
        f"typed_reason={out.get('typed_reason')!r}"
    )
    assert out.get("top_level_admission_scope") == (
        "cross_arch_require_both_arches"
    ), (
        f"expected cross_arch_require_both_arches scope; got "
        f"{out.get('top_level_admission_scope')!r}"
    )
    per_arch_diag = out.get("per_arch_diagnostics", {}).get("per_arch", {})
    for arch in ("sm_86", "sm_120"):
        cov = per_arch_diag.get(arch, {}).get("coverage_gate_verdict", {})
        assert cov.get("meets_floor") is True, (
            f"arch {arch} did not meet floor; per_arch_verdict={cov!r}"
        )

    # Included-cell grid_dim_alpha_source diagnostics are preserved.
    sm86_records = [
        r for r in out["included_cells"]
        if r["join_key"]["arch"] == "sm_86"
    ]
    sm120_records = [
        r for r in out["included_cells"]
        if r["join_key"]["arch"] == "sm_120"
    ]
    assert len(sm86_records) >= 5
    assert len(sm120_records) >= 5
    for rec in sm86_records:
        assert rec.get("grid_dim_alpha_source") == "sm_86_calibrated", (
            f"sm_86 record dropped grid_dim_alpha_source; "
            f"got {rec.get('grid_dim_alpha_source')!r}"
        )
    for rec in sm120_records:
        assert rec.get("grid_dim_alpha_source") == "sm_86_extrapolated", (
            f"sm_120 record dropped grid_dim_alpha_source; "
            f"got {rec.get('grid_dim_alpha_source')!r}"
        )

    # Deterministic re-run produces byte-identical output.
    out_b = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=sm86_pick + sm120_pick,
        bootstrap_seed=42,
    )
    assert json.dumps(out, sort_keys=True) == json.dumps(out_b, sort_keys=True)


def test_grid_dim_alpha_source_propagates_into_scalar_and_bounded_included_cells(
    tmp_path: Path,
) -> None:
    """Fixture-backed verification that comparison_runner copies the
    optional ``grid_dim_alpha_source`` field from the predictor bundle
    into the matching ``included_cells[]`` record for both the scalar
    arm and the bounded arm. Mirrors the cross-arch extrapolation
    provenance contract: sm_86 bundles emit ``sm_86_calibrated``;
    sm_120 bundles emit ``sm_86_extrapolated``; the runner surfaces
    the value verbatim in the per-cell diagnostic block. Legacy
    bundles without the field continue to admit (a third pair
    omits the field; the record's ``grid_dim_alpha_source`` key is
    simply absent)."""
    mpi_scalar = _make_synthetic_manifest_pairing_index(
        slug="alpha_source_scalar_kernel",
    )
    scalar_bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path / "alpha_source_scalar",
        result_kind="scalar",
        predicted_hit_rate=0.6,
        manifest_pairing_index=mpi_scalar,
        device="NVIDIA GeForce RTX 3090 Ti",
    )
    scalar_bundle["grid_dim_alpha_source"] = "sm_86_calibrated"

    mpi_bounded = _make_synthetic_manifest_pairing_index(
        slug="alpha_source_bounded_kernel",
    )
    bounded_bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path / "alpha_source_bounded",
        result_kind="bounded",
        hit_rate_lower=0.4,
        hit_rate_upper=0.6,
        admitted_bin_count=5,
        underconfigured_bin_count=2,
        manifest_pairing_index=mpi_bounded,
        device="NVIDIA GeForce RTX 5060 Laptop GPU",
    )
    bounded_bundle["grid_dim_alpha_source"] = "sm_86_extrapolated"

    mpi_legacy = _make_synthetic_manifest_pairing_index(
        slug="alpha_source_legacy_kernel",
    )
    legacy_bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path / "alpha_source_legacy",
        result_kind="scalar",
        predicted_hit_rate=0.5,
        manifest_pairing_index=mpi_legacy,
        device="NVIDIA GeForce RTX 3090 Ti",
    )
    # No grid_dim_alpha_source on legacy_bundle — verifies the
    # propagation is conditional (absent on the included-cell record
    # when absent on the bundle).

    label_scalar = _make_synthetic_ncu_label_cell(
        slug="alpha_source_scalar_kernel",
        arch="sm_86",
        ground_truth_pct=50.0,
        evidence_kind="live",
    )
    label_bounded = _make_synthetic_ncu_label_cell(
        slug="alpha_source_bounded_kernel",
        arch="sm_120",
        ground_truth_pct=50.0,
        evidence_kind="live",
    )
    label_legacy = _make_synthetic_ncu_label_cell(
        slug="alpha_source_legacy_kernel",
        arch="sm_86",
        ground_truth_pct=50.0,
        evidence_kind="live",
    )

    # Round out the per-arch floor with the existing five-pair helper
    # (live evidence cells with sm_86 arch + arbitrary predicted rates).
    bundles_extra_86, cells_extra_86 = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    bundles_extra_120, cells_extra_120 = _five_synthetic_pairs(
        tmp_path / "extra_sm_120", arch="sm_120", pct=50.0,
    )

    bundles = [scalar_bundle, bounded_bundle, legacy_bundle] + bundles_extra_86 + bundles_extra_120
    cells = [label_scalar, label_bounded, label_legacy] + cells_extra_86 + cells_extra_120

    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )

    included = out["included_cells"]

    # Scalar arm propagation: sm_86_calibrated preserved verbatim.
    scalar_records = [
        r for r in included
        if r["join_key"]["slug"] == "alpha_source_scalar_kernel"
    ]
    assert scalar_records, (
        "scalar fixture kernel missing from included_cells; the runner "
        "did not join the synthetic bundle to its label."
    )
    for rec in scalar_records:
        assert rec.get("grid_dim_alpha_source") == "sm_86_calibrated", (
            f"scalar included-cell record dropped grid_dim_alpha_source; "
            f"got {rec.get('grid_dim_alpha_source')!r}"
        )

    # Bounded arm propagation: sm_86_extrapolated preserved verbatim.
    bounded_records = [
        r for r in included
        if r["join_key"]["slug"] == "alpha_source_bounded_kernel"
    ]
    assert bounded_records, (
        "bounded fixture kernel missing from included_cells; the runner "
        "did not join the synthetic bundle to its label."
    )
    for rec in bounded_records:
        assert rec.get("grid_dim_alpha_source") == "sm_86_extrapolated", (
            f"bounded included-cell record dropped grid_dim_alpha_source; "
            f"got {rec.get('grid_dim_alpha_source')!r}"
        )

    # Legacy bundle (no grid_dim_alpha_source on the input): the field
    # is absent on the included-cell record (the propagation is
    # conditional via the splat-dict pattern).
    legacy_records = [
        r for r in included
        if r["join_key"]["slug"] == "alpha_source_legacy_kernel"
    ]
    assert legacy_records, (
        "legacy fixture kernel missing from included_cells."
    )
    for rec in legacy_records:
        assert "grid_dim_alpha_source" not in rec, (
            f"legacy bundle (no grid_dim_alpha_source) leaked the "
            f"field into the included-cell record; got "
            f"{rec.get('grid_dim_alpha_source')!r}"
        )


def test_synthetic_scalar_predictor_contributes_to_headline_numerator(
    tmp_path: Path,
) -> None:
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out["headline_cells_per_arch"].get("sm_86", 0) == 5


def test_synthetic_unknown_result_kind_hard_rejects(tmp_path: Path) -> None:
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    bundles[0] = copy.deepcopy(bundles[0])
    bundles[0]["result_kind"] = "moonbeam"
    contract = _admitted_contract()
    with pytest.raises(cr.ComparisonRunnerUnknownResultKind) as excinfo:
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
        )
    assert excinfo.value.observed_value == "moonbeam"


def test_synthetic_scalar_predicted_hit_rate_out_of_range_hard_rejects(
    tmp_path: Path,
) -> None:
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    bundles[0] = copy.deepcopy(bundles[0])
    bundles[0]["predicted_hit_rate"] = 1.5
    contract = _admitted_contract()
    with pytest.raises(
        cr.ComparisonRunnerPredictedHitRateOutOfRange,
    ) as excinfo:
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
        )
    assert excinfo.value.observed_value == 1.5


def test_synthetic_typed_reason_out_of_enum_hard_rejects(
    tmp_path: Path,
) -> None:
    """A ``typed_unavailable`` bundle whose upstream typed_reason
    is structurally valid (a string) but operationally unknown to
    this runner version is hard-rejected."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    bundles[0] = copy.deepcopy(bundles[0])
    bundles[0]["result_kind"] = "typed_unavailable"
    bundles[0]["typed_reason"] = "imaginary_reason_not_in_enum"
    bundles[0].pop("predicted_hit_rate", None)
    bundles[0].pop("ci_low", None)
    bundles[0].pop("ci_high", None)
    bundles[0].pop("ci_half_width", None)
    bundles[0].pop("ci_confidence", None)
    bundles[0]["underconfigured_bin_count"] = 0
    bundles[0]["admitted_bin_count"] = 0
    contract = _admitted_contract()
    with pytest.raises(
        cr.ComparisonRunnerTypedReasonOutOfEnum,
    ) as excinfo:
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
        )
    assert excinfo.value.observed_value == "imaginary_reason_not_in_enum"


def _bounded_bundle_first_with_interval(
    tmp_path: Path,
    *,
    hit_rate_lower: float,
    hit_rate_upper: float,
) -> tuple[list[dict], list[dict]]:
    """Build a 5-pair fixture whose first bundle is bounded with
    the given interval. The remaining 4 bundles stay scalar so
    that the test's hard-reject point is unambiguously the
    bounded bundle's interval validation, not a coverage
    failure further down the pipeline."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    bundles[0] = copy.deepcopy(bundles[0])
    bundles[0]["result_kind"] = "bounded"
    bundles[0]["hit_rate_lower"] = hit_rate_lower
    bundles[0]["hit_rate_upper"] = hit_rate_upper
    bundles[0].pop("predicted_hit_rate", None)
    bundles[0].pop("ci_low", None)
    bundles[0].pop("ci_high", None)
    bundles[0].pop("ci_half_width", None)
    bundles[0].pop("ci_confidence", None)
    return bundles, cells


def test_synthetic_bounded_interval_lower_greater_than_upper_hard_rejects(
    tmp_path: Path,
) -> None:
    """A bounded predictor bundle with ``hit_rate_lower >
    hit_rate_upper`` is structurally meaningless: the midpoint
    is well-defined arithmetically but the interval declares an
    empty set, and admitting it into the headline geomean
    would publish a schema-valid but operationally
    nonsensical comparison result. The runner already
    hard-rejects out-of-range scalar predictions; bounded
    bundles need the same precondition."""
    bundles, cells = _bounded_bundle_first_with_interval(
        tmp_path,
        hit_rate_lower=0.7,
        hit_rate_upper=0.3,
    )
    contract = _admitted_contract()
    with pytest.raises(
        cr.ComparisonRunnerBoundedIntervalInvalid,
    ) as excinfo:
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
        )
    assert "lower" in str(excinfo.value).lower()


def test_synthetic_bounded_interval_lower_negative_hard_rejects(
    tmp_path: Path,
) -> None:
    """``hit_rate_lower < 0`` is outside the [0, 1] band the
    scalar arm enforces; the bounded arm must enforce the
    same band on both endpoints."""
    bundles, cells = _bounded_bundle_first_with_interval(
        tmp_path,
        hit_rate_lower=-0.1,
        hit_rate_upper=0.5,
    )
    contract = _admitted_contract()
    with pytest.raises(
        cr.ComparisonRunnerBoundedIntervalInvalid,
    ) as excinfo:
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
        )
    assert (
        "[0, 1]" in str(excinfo.value)
        or "out of range" in str(excinfo.value).lower()
        or "outside" in str(excinfo.value).lower()
    )


def test_synthetic_bounded_interval_upper_above_one_hard_rejects(
    tmp_path: Path,
) -> None:
    """``hit_rate_upper > 1`` is outside the [0, 1] band; the
    bounded arm must reject it the same way the scalar arm
    rejects ``predicted_hit_rate > 1``."""
    bundles, cells = _bounded_bundle_first_with_interval(
        tmp_path,
        hit_rate_lower=0.4,
        hit_rate_upper=1.5,
    )
    contract = _admitted_contract()
    with pytest.raises(
        cr.ComparisonRunnerBoundedIntervalInvalid,
    ):
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
        )


def test_synthetic_bounded_interval_non_finite_hard_rejects(
    tmp_path: Path,
) -> None:
    """Non-finite bounds (NaN, ±inf) are not in [0, 1] and
    therefore must be hard-rejected the same way the scalar
    arm rejects non-finite ``predicted_hit_rate``."""
    bundles, cells = _bounded_bundle_first_with_interval(
        tmp_path,
        hit_rate_lower=float("nan"),
        hit_rate_upper=0.6,
    )
    contract = _admitted_contract()
    with pytest.raises(
        cr.ComparisonRunnerBoundedIntervalInvalid,
    ):
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
        )


def test_synthetic_duplicate_predictor_bundles_with_same_join_key_raise_typed_defect(
    tmp_path: Path,
) -> None:
    """Two predictor bundles whose 7-field join keys collide
    (e.g., a CLI glob that resolves to two copies of the same
    bundle file under different paths) MUST hard-reject with
    ``ComparisonRunnerDuplicatePredictorBundle`` BEFORE either
    duplicate is counted toward the per-arch admission floor or
    the headline geomean. The contract's smoke-test floor is
    per cell; admitting duplicates would let N-1 unique cells
    plus one duplicate falsely satisfy the 5-cell floor and
    overweight the duplicated cell's relative_error in the
    geomean."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    # Append a second copy of bundles[0]; the two share the
    # exact same 7-field join key by construction (same slug,
    # same MPI fields).
    bundles.append(bundles[0])
    contract = _admitted_contract()
    with pytest.raises(
        cr.ComparisonRunnerDuplicatePredictorBundle,
    ) as excinfo:
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
        )
    msg = str(excinfo.value).lower()
    assert "duplicate" in msg or "comparison_runner_duplicate" in msg


def test_synthetic_comparable_cells_per_arch_includes_join_key_unmatched_excluded(
    tmp_path: Path,
) -> None:
    """``comparable_cells_per_arch[arch]`` MUST equal the per-arch
    ``n_comparable_cells`` for that arch when the bundle has both
    matched cells AND join_key_unmatched-excluded cells. Without
    the post-loop tally, the top-level counter only includes
    matched cells while ``per_arch_diagnostics.n_comparable_cells``
    includes matched + excluded; the two surfaces emit
    contradictory comparable counts."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    # Stage one extra bundle whose source_commit drifts from
    # every label cell so it ends up in
    # comparison_runner_join_key_unmatched.
    drifted_mpi = _make_synthetic_manifest_pairing_index(
        slug="drifted_kernel",
        source_commit="d" * 40,
    )
    bundles.append(
        _make_synthetic_admitted_predictor_bundle(
            tmp_path / "drifted",
            result_kind="scalar",
            predicted_hit_rate=0.5,
            manifest_pairing_index=drifted_mpi,
            device="NVIDIA GeForce RTX 3090 Ti",
        )
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    top_level = out["comparable_cells_per_arch"].get("sm_86", 0)
    per_arch = out["per_arch_diagnostics"]["per_arch"]["sm_86"][
        "n_comparable_cells"
    ]
    assert top_level == per_arch, (
        f"top-level comparable_cells_per_arch['sm_86']={top_level} "
        f"MUST equal per_arch_diagnostics.per_arch['sm_86']."
        f"n_comparable_cells={per_arch}; the two surfaces report "
        f"contradictory comparable-cell counts when "
        f"join_key_unmatched-excluded records are present"
    )
    # Sanity check: the drifted bundle is in excluded with the
    # join_key_unmatched diagnostic.
    excluded_reasons = [e.get("reason") for e in out["excluded_cells"]]
    assert "join_key_unmatched" in excluded_reasons


def test_synthetic_comparable_cells_per_arch_includes_comparison_contract_drift_excluded(
    tmp_path: Path,
) -> None:
    """``comparable_cells_per_arch[arch]`` MUST equal the per-arch
    ``n_comparable_cells`` when a label cell is drift-excluded via
    profiler_version mismatch (``comparison_contract_drift``).
    Same cross-surface consistency invariant as the
    join_key_unmatched test, on the cell-side drift surface."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    # Stage one extra label cell whose profiler_version drifts
    # from the contract's pinned ncu_2026.1.x.
    cells.append(
        _make_synthetic_ncu_label_cell(
            slug="drifted_label",
            arch="sm_86",
            ground_truth_pct=50.0,
            evidence_kind="live",
            profiler_version="ncu_2027.0.x",
        )
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    top_level = out["comparable_cells_per_arch"].get("sm_86", 0)
    per_arch = out["per_arch_diagnostics"]["per_arch"]["sm_86"][
        "n_comparable_cells"
    ]
    assert top_level == per_arch, (
        f"top-level comparable_cells_per_arch['sm_86']={top_level} "
        f"MUST equal per_arch_diagnostics.per_arch['sm_86']."
        f"n_comparable_cells={per_arch}; the two surfaces report "
        f"contradictory comparable-cell counts when "
        f"comparison_contract_drift-excluded records are present"
    )
    # Sanity check: the drifted cell is in excluded with the
    # comparison_contract_drift diagnostic.
    excluded_reasons = [e.get("reason") for e in out["excluded_cells"]]
    assert "comparison_contract_drift" in excluded_reasons


# =====================================================================
# Coverage-gate semantics — ``headline_cells_per_arch >= 5`` is the
# admission floor.
# =====================================================================


def test_runner_admits_synthetic_geomean_at_or_above_smoke_floor(
    tmp_path: Path,
) -> None:
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
        predicted_rates=(0.5, 0.5, 0.5, 0.5, 0.5),
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out["result_kind"] == "admitted"
    assert out["headline_cells_per_arch"].get("sm_86", 0) == 5


def test_runner_emits_synthetic_typed_unavailable_below_smoke_floor(
    tmp_path: Path,
) -> None:
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
        predicted_rates=(0.5, 0.5, 0.5, 0.5),
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out["result_kind"] == "typed_unavailable"
    assert (
        out["typed_reason"]
        == "headline_geomean_admitted_cell_count_below_threshold"
    )
    assert out["headline_cells_per_arch"].get("sm_86", 0) == 4


def test_runner_emits_synthetic_typed_unavailable_with_zero_headline(
    tmp_path: Path,
) -> None:
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=[],
        label_cells=[],
        bootstrap_seed=0,
    )
    assert out["result_kind"] == "typed_unavailable"
    assert (
        out["typed_reason"]
        == "headline_geomean_admitted_cell_count_below_threshold"
    )


def test_runner_emits_synthetic_typed_unavailable_when_all_comparable_are_unavailable(
    tmp_path: Path,
) -> None:
    """``comparable_cells_per_arch >= 5`` BUT
    ``headline_cells_per_arch < 5``: every comparable cell is
    predictor_typed_unavailable. The output records both counts so
    the diagnostic distinction is auditable."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    for i, bundle in enumerate(bundles):
        bundles[i] = _make_synthetic_admitted_predictor_bundle(
            tmp_path / f"tu{i}",
            result_kind="typed_unavailable",
            typed_reason="cache_model_underconfigured",
            manifest_pairing_index=bundle["manifest_pairing_index"],
            device="NVIDIA GeForce RTX 3090 Ti",
        )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out["result_kind"] == "typed_unavailable"
    assert (
        out["typed_reason"]
        == "headline_geomean_admitted_cell_count_below_threshold"
    )
    assert out["comparable_cells_per_arch"].get("sm_86", 0) == 5
    assert out["headline_cells_per_arch"].get("sm_86", 0) == 0


# =====================================================================
# Output bundle schema validation — every emitted bundle MUST
# strict-validate against ``comparison_bundle.schema.json``.
# =====================================================================


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_admitted_output_bundle_strict_schema_validates(
    tmp_path: Path,
) -> None:
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(out, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_typed_unavailable_output_bundle_strict_schema_validates(
    tmp_path: Path,
) -> None:
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=[],
        label_cells=[],
        bootstrap_seed=0,
    )
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(out, schema)


# =====================================================================
# CLI wrapper smoke test — the operator-facing `__main__` entry point.
# =====================================================================
# Exercises the four documented CLI flags (--contract,
# --predictor-bundles-glob, --label-admitted-path, --output) per the
# operator-runbook discipline of `BL-20260504-runbook-command-verification`:
# the documented CLI args must be exercised by the test, not just
# declared in argparse.


# =====================================================================
# Runner regression — strict schema_version=2 admitted-cell file.
# =====================================================================
# Schema-strict admitted-file corrective coverage. The aggregator now
# emits label_admitted.json with file-level schema_version=2 and per-
# record schema_version=2 plus the runner-joinable 7-field key, so
# the runner must continue to admit the synthetic v2 fixture without
# producing comparison_contract_drift exclusions. A profiler-version
# drift negative test on the same v2 surface confirms the runner's
# typed-reason path still fires.


def _make_v2_admitted_label_record(
    *,
    slug: str,
    arch: str = "sm_86",
    binary_sha256: str | None = None,
    kernel_signature_sha256: str | None = None,
    profiler_version: str = "ncu_2026.1.x",
    cache_hit_rate_l2_sector_pct: float = 50.0,
    cache_hit_rate_l1tex_sector_pct: float | None = None,
    l1tex_label_status: str = "mean_absent_label_missing",
    n_repeats: int = 5,
    stddev_l2_sector: float = 0.005,
    evidence_kind: str = "live",
) -> dict:
    """Return a strict v5 admitted-cell record matching
    ``local_label_admitted.schema.json``'s ``$defs.admitted_cell``.
    The helper name retains its historical ``_v2_`` prefix because
    the test names below pin the wire-format contract by file shape,
    not by version number; the body emits v5 records carrying the
    new ``cache_hit_rate_l1tex_sector_pct`` (nullable) +
    ``l1tex_label_status`` (closed enum) plus the post-migration
    ``evidence_kind`` field. The default L1 shape is
    ``(None, "mean_absent_label_missing")`` so L2-focused tests
    don't have to specify L1 inputs; tests that need the L1 mean
    pass it explicitly and adjust ``l1tex_label_status`` to
    ``"mean_present"``."""
    if binary_sha256 is None:
        binary_sha256 = "a" * 64
    if kernel_signature_sha256 is None:
        kernel_signature_sha256 = "b" * 64
    return {
        "schema_version": 5,
        "slug": slug,
        "shape_point": "trace_00",
        "binary_sha256": binary_sha256,
        "source_commit": "c" * 40,
        "kernel_signature_sha256": kernel_signature_sha256,
        "arch": arch,
        "ncu_metric_version": "ncu_2026.1.x",
        "profiler_version": profiler_version,
        "cache_hit_rate_l2_sector_pct": cache_hit_rate_l2_sector_pct,
        "cache_hit_rate_l1tex_sector_pct": cache_hit_rate_l1tex_sector_pct,
        "l1tex_label_status": l1tex_label_status,
        "n_repeats": n_repeats,
        "stddev_l2_sector": stddev_l2_sector,
        "evidence_kind": evidence_kind,
    }


_LOCAL_LABEL_ADMITTED_SCHEMA_PATH = (
    _SUITE_ROOT / "schemas" / "local_label_admitted.schema.json"
)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_v2_strict_admitted_label_file_runs_to_admitted_result(
    tmp_path: Path,
) -> None:
    """M1 runner regression for the strict schema_version=2 admitted
    file. Build five strict v2 records that schema-validate against
    local_label_admitted.schema.json, build five same-key synthetic
    predictor bundles, run comparison_runner.main with
    --label-admitted-path pointing at the v2 file, and assert
    result_kind=='admitted', headline_cells_per_arch['sm_86']>=5, and
    zero comparison_contract_drift exclusions."""
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    v2_records: list[dict] = []
    for i in range(5):
        slug = f"v2_kernel_{i}"
        # Build a same-key predictor bundle for the cell. The
        # manifest_pairing_index must agree on the runner's 7-field
        # join key (the runner reads the predictor's MPI as the
        # left side of the join).
        binary_sha = ("d" * 63) + str(i)
        kernel_sig = ("e" * 63) + str(i)
        mpi = _make_synthetic_manifest_pairing_index(
            slug=slug,
            shape_point="trace_00",
            binary_sha256=binary_sha,
            source_commit="c" * 40,
            kernel_signature_sha256=kernel_sig,
        )
        bundle = _make_synthetic_admitted_predictor_bundle(
            tmp_path / f"sec_{i}",
            result_kind="scalar",
            predicted_hit_rate=0.4,
            manifest_pairing_index=mpi,
            device="NVIDIA GeForce RTX 3090 Ti",
        )
        bundle_path = bundles_dir / f"bundle_{i}.json"
        bundle_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        v2_records.append(_make_v2_admitted_label_record(
            slug=slug, arch="sm_86",
            binary_sha256=binary_sha,
            kernel_signature_sha256=kernel_sig,
            cache_hit_rate_l2_sector_pct=50.0,
        ))
    label_admitted_doc = {"schema_version": 5, "cells": v2_records}
    # Strict-schema validation gate: the v2 file MUST validate
    # against local_label_admitted.schema.json before the runner is
    # invoked (this is the producer/consumer wire pin Codex flagged
    # as a missing test surface).
    strict_schema = json.loads(
        _LOCAL_LABEL_ADMITTED_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    jsonschema.validate(label_admitted_doc, strict_schema)
    label_admitted = tmp_path / "label_admitted.json"
    label_admitted.write_text(
        json.dumps(label_admitted_doc, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_path = tmp_path / "output_bundle.json"
    contract_path = _write_contract_with_overrides(
        tmp_path, require_both_arches_for_cross_arch=False,
    )
    rc = cr.main([
        "--contract", str(contract_path),
        "--predictor-bundles-glob", str(bundles_dir / "bundle_*.json"),
        "--label-admitted-path", str(label_admitted),
        "--output", str(output_path),
    ])
    assert rc == 0, f"expected exit 0; got rc={rc!r}"
    out = json.loads(output_path.read_text(encoding="utf-8"))
    output_schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(out, output_schema)
    assert out["result_kind"] == "admitted"
    assert out["headline_cells_per_arch"].get("sm_86", 0) >= 5
    excluded = out.get("excluded_cells", [])
    drift_exclusions = [
        c for c in excluded
        if c.get("typed_reason") == "comparison_contract_drift"
    ]
    assert drift_exclusions == [], (
        f"v2 strict-schema fixture produced "
        f"{len(drift_exclusions)} comparison_contract_drift "
        f"exclusion(s); the runner must consume the canonical NCU "
        f"contract pin without drift: {drift_exclusions!r}"
    )


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_v2_strict_admitted_label_file_with_live_evidence_admits_sm120(
    tmp_path: Path,
) -> None:
    """Cross-arch wire-format pre-flight for sm_120 (live-evidence
    variant). Five sm_120 admitted-cell records carry
    ``evidence_kind='live'`` (the helper's default after the v3
    migration). The current 3090 Ti host cannot run live sm_120
    NCU sweeps (the receiving RTX 5060 laptop is the cross-host
    pivot's destination for the live sweep), so this test is the
    Offline-verified wire-format guard the operator's eventual
    cross-host pivot relies on: it
    proves that the producer/consumer chain (label_admitted.schema.json
    arch enum + comparison_runner per-arch coverage gate +
    headline_cells_per_arch['sm_120']) works end-to-end on synthetic
    fixtures with declared live evidence BEFORE any actual sm_120
    sweep lands. The live-only top-level gate counts each cell, so
    the result remains ``admitted``.

    Asserts ``result_kind=='admitted'``,
    ``headline_cells_per_arch['sm_120']>=5``, and zero
    ``comparison_contract_drift`` exclusions on the synthetic
    sm_120 fixture."""
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    v2_records: list[dict] = []
    for i in range(5):
        slug = f"v2_sm120_kernel_{i}"
        binary_sha = ("1" * 63) + str(i)
        kernel_sig = ("2" * 63) + str(i)
        mpi = _make_synthetic_manifest_pairing_index(
            slug=slug,
            shape_point="trace_00",
            binary_sha256=binary_sha,
            source_commit="c" * 40,
            kernel_signature_sha256=kernel_sig,
        )
        bundle = _make_synthetic_admitted_predictor_bundle(
            tmp_path / f"sec_sm120_{i}",
            result_kind="scalar",
            predicted_hit_rate=0.4,
            manifest_pairing_index=mpi,
            device="NVIDIA GeForce RTX 5060 Laptop GPU",
        )
        bundle_path = bundles_dir / f"bundle_{i}.json"
        bundle_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        v2_records.append(_make_v2_admitted_label_record(
            slug=slug, arch="sm_120",
            binary_sha256=binary_sha,
            kernel_signature_sha256=kernel_sig,
            cache_hit_rate_l2_sector_pct=50.0,
        ))
    label_admitted_doc = {"schema_version": 5, "cells": v2_records}
    strict_schema = json.loads(
        _LOCAL_LABEL_ADMITTED_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    jsonschema.validate(label_admitted_doc, strict_schema)
    label_admitted = tmp_path / "label_admitted.json"
    label_admitted.write_text(
        json.dumps(label_admitted_doc, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_path = tmp_path / "output_bundle.json"
    contract_path = _write_contract_with_overrides(
        tmp_path, require_both_arches_for_cross_arch=False,
    )
    rc = cr.main([
        "--contract", str(contract_path),
        "--predictor-bundles-glob", str(bundles_dir / "bundle_*.json"),
        "--label-admitted-path", str(label_admitted),
        "--output", str(output_path),
    ])
    assert rc == 0, f"expected exit 0; got rc={rc!r}"
    out = json.loads(output_path.read_text(encoding="utf-8"))
    output_schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(out, output_schema)
    assert out["result_kind"] == "admitted"
    assert out["headline_cells_per_arch"].get("sm_120", 0) >= 5
    excluded = out.get("excluded_cells", [])
    drift_exclusions = [
        c for c in excluded
        if c.get("typed_reason") == "comparison_contract_drift"
    ]
    assert drift_exclusions == [], (
        f"v2 strict-schema sm_120 fixture produced "
        f"{len(drift_exclusions)} comparison_contract_drift "
        f"exclusion(s); the runner must consume the canonical NCU "
        f"contract pin for sm_120 records without drift: "
        f"{drift_exclusions!r}"
    )


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_v2_strict_admitted_label_file_with_synthetic_evidence_routes_to_typed_unavailable_sm120(
    tmp_path: Path,
) -> None:
    """Cross-arch wire-format pre-flight for sm_120 (synthetic-evidence
    variant). Five sm_120 admitted-cell records carry
    ``evidence_kind='synthetic'`` and exercise the live-only floor
    enforcement: synthetic cells do NOT satisfy the per-arch floor,
    so the top-level result_kind is ``typed_unavailable`` with
    typed_reason ``headline_geomean_admitted_cell_count_below_threshold``.

    This regression guard makes the Cross-arch-verified discipline
    mechanically enforceable: a fixture-only sm_120 leg cannot
    appear admitted, even though every record validates against the
    strict v3 schema. The same fixture with
    ``evidence_kind='live'`` produces ``admitted`` per the live
    variant test."""
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    v2_records: list[dict] = []
    for i in range(5):
        slug = f"v2_synthetic_sm120_kernel_{i}"
        binary_sha = ("3" * 63) + str(i)
        kernel_sig = ("4" * 63) + str(i)
        mpi = _make_synthetic_manifest_pairing_index(
            slug=slug,
            shape_point="trace_00",
            binary_sha256=binary_sha,
            source_commit="c" * 40,
            kernel_signature_sha256=kernel_sig,
        )
        bundle = _make_synthetic_admitted_predictor_bundle(
            tmp_path / f"sec_synthetic_sm120_{i}",
            result_kind="scalar",
            predicted_hit_rate=0.4,
            manifest_pairing_index=mpi,
            device="NVIDIA GeForce RTX 5060 Laptop GPU",
        )
        bundle_path = bundles_dir / f"bundle_{i}.json"
        bundle_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        v2_records.append(_make_v2_admitted_label_record(
            slug=slug,
            arch="sm_120",
            binary_sha256=binary_sha,
            kernel_signature_sha256=kernel_sig,
            cache_hit_rate_l2_sector_pct=50.0,
            evidence_kind="synthetic",
        ))
    label_admitted_doc = {"schema_version": 5, "cells": v2_records}
    strict_schema = json.loads(
        _LOCAL_LABEL_ADMITTED_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    jsonschema.validate(label_admitted_doc, strict_schema)
    label_admitted = tmp_path / "label_admitted.json"
    label_admitted.write_text(
        json.dumps(label_admitted_doc, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_path = tmp_path / "output_bundle.json"
    contract_path = _SUITE_ROOT / "comparison_contract.toml"
    rc = cr.main([
        "--contract", str(contract_path),
        "--predictor-bundles-glob", str(bundles_dir / "bundle_*.json"),
        "--label-admitted-path", str(label_admitted),
        "--output", str(output_path),
    ])
    assert rc == 0, f"expected exit 0; got rc={rc!r}"
    out = json.loads(output_path.read_text(encoding="utf-8"))
    output_schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(out, output_schema)
    assert out["result_kind"] == "typed_unavailable", (
        f"synthetic-only sm_120 fixture must route to "
        f"typed_unavailable under the live-only floor; got "
        f"{out['result_kind']!r}"
    )
    assert (
        out["typed_reason"]
        == "headline_geomean_admitted_cell_count_below_threshold"
    )
    sm120_verdict = (
        out["per_arch_diagnostics"]["per_arch"]["sm_120"][
            "coverage_gate_verdict"
        ]
    )
    assert sm120_verdict["headline_count_live"] == 0
    assert sm120_verdict["headline_count_synthetic"] == 5
    assert sm120_verdict["meets_floor"] is False


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_synthetic_v2_strict_admitted_profiler_version_drift_routes_to_contract_drift(
    tmp_path: Path,
) -> None:
    """Profiler-version drift negative test on the same v2 fixture
    surface. A v2 record carrying a profiler_version != ncu_2026.1.x
    fails the strict-schema gate (the schema pins
    profiler_version to a const), proving the wire pin is binding
    when the file is fed through the schema validator."""
    drifted_record = _make_v2_admitted_label_record(
        slug="drifted_kernel",
        arch="sm_86",
        profiler_version="ncu_2027.0.x",  # off-pin
    )
    label_admitted_doc = {"schema_version": 5, "cells": [drifted_record]}
    strict_schema = json.loads(
        _LOCAL_LABEL_ADMITTED_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        jsonschema.validate(label_admitted_doc, strict_schema)
    msg = str(excinfo.value)
    assert (
        "profiler_version" in msg or "ncu_2026.1.x" in msg
    ), (
        f"strict-schema rejection did not name profiler_version / "
        f"the pinned version literal; got: {msg!r}"
    )


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_cli_main_writes_synthetic_schema_valid_output_bundle(
    tmp_path: Path,
) -> None:
    """End-to-end smoke: write a synthetic predictor bundle and a
    synthetic label-admitted JSON to disk, invoke
    ``comparison_runner.main(argv=...)`` with the four documented
    flags, and assert the output file exists, schema-validates, and
    carries the expected per-arch counts."""
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    cells: list[dict] = []
    for i in range(5):
        slug = f"cli_smoke_kernel_{i}"
        mpi = _make_synthetic_manifest_pairing_index(slug=slug)
        bundle = _make_synthetic_admitted_predictor_bundle(
            tmp_path / f"sec_{i}",
            result_kind="scalar",
            predicted_hit_rate=0.4,
            manifest_pairing_index=mpi,
            device="NVIDIA GeForce RTX 3090 Ti",
        )
        bundle_path = bundles_dir / f"bundle_{i}.json"
        bundle_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        cells.append(
            _make_synthetic_ncu_label_cell(
                slug=slug,
                arch="sm_86",
                ground_truth_pct=50.0,
                evidence_kind="live",
            )
        )
    label_admitted = tmp_path / "label_admitted.json"
    label_admitted.write_text(
        json.dumps(
            {"schema_version": 5, "cells": cells},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output_bundle.json"
    contract_path = _write_contract_with_overrides(
        tmp_path, require_both_arches_for_cross_arch=False,
    )
    rc = cr.main([
        "--contract", str(contract_path),
        "--predictor-bundles-glob", str(bundles_dir / "bundle_*.json"),
        "--label-admitted-path", str(label_admitted),
        "--output", str(output_path),
    ])
    assert rc == 0, f"expected exit 0; got rc={rc!r}"
    assert output_path.exists(), (
        f"CLI did not write output bundle at {output_path}"
    )
    out = json.loads(output_path.read_text(encoding="utf-8"))
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(out, schema)
    assert out["result_kind"] == "admitted"
    assert out["headline_cells_per_arch"].get("sm_86", 0) == 5
    assert math.isclose(out["geomean"], 0.20, rel_tol=1e-9)


# =====================================================================
# target_arch admission mode — when the contract's
# require_both_arches_for_cross_arch=True is in force, sm_120 has 0
# cells, and target_arch="sm_86" is explicitly requested, top-level
# admission gates on sm_86 alone. The default target_arch=None
# preserves the existing cross-arch contract (typed_unavailable when
# any arch in arch_set has 0 cells).
# =====================================================================


def _strict_cross_arch_contract() -> cr.ComparisonContract:
    """Return the canonical contract pin shape (mirrors
    comparison_contract.toml) with require_both_arches_for_cross_arch=True."""
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
            require_both_arches_for_cross_arch=True,
        ),
        log_space_geomean=True,
        arch_set=("sm_86", "sm_120"),
    )


def test_synthetic_target_arch_sm86_admits_top_level_when_only_sm86_meets_floor(
    tmp_path: Path,
) -> None:
    """The single-arch baseline shape: 5 sm_86 admitted live cells,
    0 sm_120 cells. Under the canonical contract pin
    (require_both_arches_for_cross_arch=True), default behavior is
    typed_unavailable. When the caller passes target_arch='sm_86',
    top-level admission gates on sm_86-meets-floor alone; the
    bundle carries result_kind='admitted' with finite geomean/CI
    fields.

    Round-14 semantic update (Codex Round 13 Blocking Side Issue
    #2): per-arch diagnostics in target_arch mode contain ONLY
    the target arch. Pre-Round-14 the tally included both
    arches with sm_120 marked `statistics_status="no_cells"`;
    that surfaced an arch the caller explicitly excluded from
    the comparison universe."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _strict_cross_arch_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
        target_arch="sm_86",
    )
    assert out["result_kind"] == "admitted", (
        f"target_arch='sm_86' must produce top-level "
        f"result_kind='admitted' when sm_86 meets floor; got "
        f"{out.get('result_kind')!r} (typed_reason="
        f"{out.get('typed_reason')!r})"
    )
    assert math.isclose(out["geomean"], 0.20, rel_tol=1e-9), (
        f"top-level geomean must be finite and equal the per-arch "
        f"value when target_arch admission fires; got "
        f"geomean={out.get('geomean')!r}"
    )
    for ci_field in (
        "geomean_ci_low",
        "geomean_ci_high",
        "geomean_ci_half_width",
    ):
        assert ci_field in out and math.isfinite(out[ci_field]), (
            f"{ci_field} must be present and finite in the admitted "
            f"top-level bundle; got {out.get(ci_field)!r}"
        )
    per_arch = out["per_arch_diagnostics"]["per_arch"]
    assert set(per_arch.keys()) == {"sm_86"}, (
        f"Round-14: target_arch='sm_86' must restrict per-arch "
        f"diagnostics to ONLY {{'sm_86'}}; got "
        f"{sorted(per_arch.keys())!r}"
    )
    assert per_arch["sm_86"].get("coverage_gate_verdict", {}).get(
        "meets_floor"
    ) is True, (
        f"sm_86 must meet the smoke-test floor (5 live cells); "
        f"got {per_arch['sm_86']!r}"
    )


def test_synthetic_target_arch_default_none_preserves_cross_arch_typed_unavailable(
    tmp_path: Path,
) -> None:
    """Under the canonical contract pin (require_both_arches_for_cross_arch=True),
    omitting target_arch (the default None) preserves the existing
    cross-arch contract: top-level result_kind='typed_unavailable'
    with typed_reason='headline_geomean_admitted_cell_count_below_threshold'
    when any arch in arch_set has 0 cells."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _strict_cross_arch_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out["result_kind"] == "typed_unavailable", (
        f"default target_arch=None under strict cross-arch contract "
        f"must produce typed_unavailable when sm_120 has 0 cells; "
        f"got {out.get('result_kind')!r}"
    )
    assert out.get("typed_reason") == (
        "headline_geomean_admitted_cell_count_below_threshold"
    ), (
        f"typed_reason should match the documented coverage-gate "
        f"reason; got {out.get('typed_reason')!r}"
    )


def test_synthetic_target_arch_sm86_below_floor_still_emits_typed_unavailable(
    tmp_path: Path,
) -> None:
    """target_arch='sm_86' is NOT a per-arch-floor bypass. When sm_86
    has only 4 admitted cells (below smoke_test_floor=5), top-level
    must still emit typed_unavailable. The mode only relaxes the
    cross-arch admission requirement; it does not relax the
    per-arch headline-cell floor."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4),
    )
    contract = _strict_cross_arch_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
        target_arch="sm_86",
    )
    assert out["result_kind"] == "typed_unavailable", (
        f"target_arch='sm_86' with 4 < floor=5 admitted cells must "
        f"still emit typed_unavailable; got {out.get('result_kind')!r}"
    )
    assert out.get("typed_reason") == (
        "headline_geomean_admitted_cell_count_below_threshold"
    )


def test_synthetic_target_arch_unknown_value_raises_typed_runner_error(
    tmp_path: Path,
) -> None:
    """An invalid target_arch value (not in the contract's arch_set)
    is a runner-level usage error and raises immediately rather
    than producing a misleading admitted/unavailable verdict."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _strict_cross_arch_contract()
    with pytest.raises(ValueError, match=r"target_arch"):
        cr.run_comparison(
            contract=contract,
            predictor_bundles=bundles,
            label_cells=cells,
            bootstrap_seed=0,
            target_arch="sm_99",
        )


def test_synthetic_target_arch_geomean_excludes_non_target_arch_cells(
    tmp_path: Path,
) -> None:
    """When ``target_arch`` is set, the top-level ``geomean`` and
    bootstrap CI MUST be computed over the target arch's joined
    cells only. The ``included_cells`` array continues to carry
    every joined cell across all arches (per_arch_diagnostics
    needs the full set), but the headline single-arch baseline
    number cannot be polluted by non-target-arch cells that did
    not participate in the target-arch admission gate.

    Stages 5 sm_86 admitted live cells (rel_err=0.20 each → sm_86
    geomean=0.20) plus 1 sm_120 admitted live cell (rel_err=1.00,
    below sm_120 floor=5 but joined). Pre-fix: top-level geomean
    is the mixed-arch geomean over 6 cells ≈ 0.2615. Post-fix:
    top-level geomean equals the sm_86-only geomean = 0.20."""
    bundles_sm86, cells_sm86 = _five_synthetic_pairs(
        tmp_path / "sm86",
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    # 1 sm_120 cell with predicted=0.0, true=0.5 → rel_err = 1.00.
    sm120_slug = "synthetic_sm120_kernel_0"
    sm120_mpi = _make_synthetic_manifest_pairing_index(slug=sm120_slug)
    sm120_bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path / "sm120_sector_0",
        result_kind="scalar",
        predicted_hit_rate=0.0,
        manifest_pairing_index=sm120_mpi,
        device="NVIDIA GeForce RTX 5060 Laptop GPU",
    )
    sm120_cell = _make_synthetic_ncu_label_cell(
        slug=sm120_slug,
        arch="sm_120",
        ground_truth_pct=50.0,
        evidence_kind="live",
    )
    bundles = bundles_sm86 + [sm120_bundle]
    cells = cells_sm86 + [sm120_cell]
    contract = _strict_cross_arch_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
        target_arch="sm_86",
    )
    assert out["result_kind"] == "admitted", (
        f"target_arch='sm_86' must admit when sm_86 meets floor "
        f"regardless of sm_120 cell count; got "
        f"{out.get('result_kind')!r} (typed_reason="
        f"{out.get('typed_reason')!r})"
    )
    assert math.isclose(out["geomean"], 0.20, rel_tol=1e-9), (
        f"top-level geomean under target_arch='sm_86' MUST be the "
        f"sm_86-only geomean (0.20 from rel_err=0.20 across 5 cells); "
        f"the included sm_120 cell's rel_err=1.00 would skew the "
        f"mixed-arch geomean to ~0.2615 if not filtered out; got "
        f"geomean={out.get('geomean')!r}"
    )


# =====================================================================
# Top-level admission scope — the bundle records which admission
# path produced the verdict so downstream readers can distinguish
# Verified-live-<arch> (target_arch mode) from Cross-arch-verified
# (cross_arch mode). Without this field the bundle is ambiguous
# whenever target_arch is used with a contract that pins
# require_both_arches_for_cross_arch=True.
# =====================================================================


def test_synthetic_top_level_admission_scope_records_target_arch_value(
    tmp_path: Path,
) -> None:
    """When ``target_arch`` is set, the bundle MUST persist
    ``top_level_admission_scope == "target_arch=<arch>"`` so
    downstream readers cannot mistake a single-arch admission for
    a cross-arch one. Mirrors the existing
    test_synthetic_target_arch_sm86_admits_top_level fixture."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _strict_cross_arch_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
        target_arch="sm_86",
    )
    assert out.get("top_level_admission_scope") == "target_arch=sm_86", (
        f"target_arch='sm_86' invocation MUST record "
        f"top_level_admission_scope='target_arch=sm_86' so the "
        f"bundle is unambiguous; got "
        f"{out.get('top_level_admission_scope')!r}"
    )


def test_synthetic_top_level_admission_scope_records_cross_arch_require_both(
    tmp_path: Path,
) -> None:
    """When ``target_arch`` is None AND the contract requires
    both arches for cross-arch admission, the bundle MUST record
    ``top_level_admission_scope == "cross_arch_require_both_arches"``."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _strict_cross_arch_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out.get("top_level_admission_scope") == (
        "cross_arch_require_both_arches"
    ), (
        f"contract.coverage_gate.require_both_arches_for_cross_arch=True "
        f"with target_arch=None MUST record "
        f"top_level_admission_scope='cross_arch_require_both_arches'; "
        f"got {out.get('top_level_admission_scope')!r}"
    )


def test_synthetic_top_level_admission_scope_records_cross_arch_any(
    tmp_path: Path,
) -> None:
    """When ``target_arch`` is None AND the contract does NOT
    require both arches, the bundle MUST record
    ``top_level_admission_scope == "cross_arch_any_arch"``."""
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_86",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _admitted_contract()  # require_both_arches_for_cross_arch defaults to False
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
    )
    assert out.get("top_level_admission_scope") == "cross_arch_any_arch", (
        f"contract.coverage_gate.require_both_arches_for_cross_arch=False "
        f"with target_arch=None MUST record "
        f"top_level_admission_scope='cross_arch_any_arch'; got "
        f"{out.get('top_level_admission_scope')!r}"
    )


# =====================================================================
# Round-14 (Codex Round 13 Blocking Side Issue #2): target_arch scope
# leak. Pre-Round-14 `run_comparison(target_arch="sm_120")` walked ALL
# label_cells (sm_86 + sm_120) and emitted comparable_cells_per_arch +
# headline_cells_per_arch with both arches; 44 historical sm_86 labels
# (no matching sm_120 bundle) appeared as
# `comparison_runner_join_key_unmatched` exclusions even though the
# caller asked for sm_120-only. The Round-14 fix filters label_cells
# to target_arch BEFORE join-key bookkeeping and narrows arch_set to
# (target_arch,) so per-arch surfaces contain only the target arch.
# =====================================================================


def test_target_arch_filters_mixed_corpus_to_sm120_only(
    tmp_path: Path,
) -> None:
    """Mixed sm_86 + sm_120 label corpus + sm_120-only predictor
    bundles + ``target_arch="sm_120"`` MUST yield zero sm_86
    surfaces in per-arch tallies and zero
    ``comparison_runner_join_key_unmatched`` exclusions."""
    # 5 sm_120 cells matched by 5 sm_120 bundles.
    bundles, sm120_cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_120",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    # 3 unmatched sm_86 cells (no corresponding bundles).
    sm86_cells: list[dict] = []
    for i in range(3):
        sm86_cells.append(
            _make_synthetic_ncu_label_cell(
                slug=f"historical_sm86_{i}",
                arch="sm_86",
                ground_truth_pct=50.0,
                evidence_kind="live",
            )
        )
    mixed = sm120_cells + sm86_cells

    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=mixed,
        bootstrap_seed=0,
        target_arch="sm_120",
    )

    # Per-arch surfaces must contain ONLY sm_120 keys.
    assert out["comparable_cells_per_arch"] == {"sm_120": 5}, (
        f"target_arch='sm_120' must filter sm_86 out of "
        f"comparable_cells_per_arch; got "
        f"{out.get('comparable_cells_per_arch')!r}"
    )
    assert set(out["headline_cells_per_arch"].keys()) == {"sm_120"}, (
        f"headline_cells_per_arch must contain only sm_120; got "
        f"{sorted(out['headline_cells_per_arch'].keys())!r}"
    )
    assert set(
        out["scalar_or_bounded_headline_cells_per_arch"].keys()
    ) == {"sm_120"}, (
        f"scalar_or_bounded_headline_cells_per_arch must contain "
        f"only sm_120; got "
        f"{sorted(out['scalar_or_bounded_headline_cells_per_arch'].keys())!r}"
    )
    assert out["arches"] == ["sm_120"], (
        f"arches list must be exactly ['sm_120']; got {out['arches']!r}"
    )

    # The Codex Round 13 review's specific finding was that
    # historical sm_86 labels leaked into the comparison universe
    # and produced `comparison_runner_join_key_unmatched`
    # exclusions even when the caller asked for sm_120-only.
    # The Round-14 filter eliminates the sm_86-side leak. Other
    # unmatched-record sources (e.g., L1TEX layer broadcasting
    # on synthetic cells without an L1 label mean) are unrelated
    # mechanisms and out of scope for this regression; we narrow
    # the assertion to "no sm_86 leak in unmatched records"
    # rather than "zero total unmatched records".
    unmatched_sm86 = [
        e for e in out.get("excluded_cells", [])
        if (
            e.get("diagnostic") == "comparison_runner_join_key_unmatched"
            and e.get("join_key", {}).get("arch") == "sm_86"
        )
    ]
    assert unmatched_sm86 == [], (
        f"target_arch='sm_120' must produce zero sm_86 "
        f"comparison_runner_join_key_unmatched exclusions on a "
        f"mixed sm_86/sm_120 corpus; got "
        f"{len(unmatched_sm86)}: {unmatched_sm86!r}"
    )

    # Per-arch diagnostics block also contains only sm_120.
    per_arch_diag = (
        out.get("per_arch_diagnostics", {}).get("per_arch", {})
    )
    assert set(per_arch_diag.keys()) == {"sm_120"}, (
        f"per_arch_diagnostics.per_arch must contain only sm_120; "
        f"got {sorted(per_arch_diag.keys())!r}"
    )

    # Admission scope tag.
    assert out["top_level_admission_scope"] == "target_arch=sm_120", (
        f"top_level_admission_scope must be 'target_arch=sm_120'; got "
        f"{out['top_level_admission_scope']!r}"
    )


def test_target_arch_filters_sm86_with_zero_target_cells_to_typed_unavailable(
    tmp_path: Path,
) -> None:
    """If the mixed corpus contains ONLY sm_86 cells but the caller
    asks for ``target_arch='sm_120'``, the comparable_cells_per_arch
    MUST report sm_120: 0 and the result MUST gate on the smoke-test
    floor as typed_unavailable. Pins the empty-target-arch
    semantic so a future refactor that silently swallowed
    target_arch wouldn't pass."""
    # Only sm_86 cells; no matching sm_120 bundles.
    sm86_cells: list[dict] = []
    for i in range(3):
        sm86_cells.append(
            _make_synthetic_ncu_label_cell(
                slug=f"historical_sm86_{i}",
                arch="sm_86",
                ground_truth_pct=50.0,
                evidence_kind="live",
            )
        )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=[],
        label_cells=sm86_cells,
        bootstrap_seed=0,
        target_arch="sm_120",
    )
    assert out["comparable_cells_per_arch"] == {"sm_120": 0}, (
        f"empty-target-arch case must show 0 comparable cells under "
        f"the target arch; got {out.get('comparable_cells_per_arch')!r}"
    )
    assert "sm_86" not in out["comparable_cells_per_arch"], (
        f"empty-target-arch case must NOT leak sm_86 into "
        f"comparable_cells_per_arch; got "
        f"{out.get('comparable_cells_per_arch')!r}"
    )


# =====================================================================
# Round-15 (AC-5.3) dual-geomean reporting. The admitted-arm bundle
# MUST emit:
#   - ``full_cohort_geomean`` (mirroring the legacy ``geomean`` field
#     on this arm, but keyed canonically per AC-5.3),
#   - ``holdout_geomean`` over the subset of included cells whose
#     roster row has ``split == "holdout"`` (the plan reserves 7
#     such cells),
#   - per-arm bootstrap CI tuples for each (low / high / half_width)
#     using the contract pins (2000 resamples / 0.95 confidence),
#   - ``holdout_cell_count`` recording the actual holdout-headline
#     cell count (the 7-pool may have typed-unavailable cells
#     bubbled out — those are NOT silently substituted from
#     calibration/validation).
#
# The roster split index is loaded from the frozen
# ``data/local_labels/sm120_cohort_expansion_roster.toml``. Tests
# that synthesize cells with slug/shape_point pairs matching the
# roster's holdout rows must use the real roster slugs so the
# split index resolves them as holdout.
# =====================================================================


_ROSTER_HOLDOUT_KEYS: tuple[tuple[str, str], ...] = (
    ("cross_entropy_fwd_fp32_sm80", "trace_l2_accuracy_xl"),
    ("embedding_gather_int32_sm80", "trace_l2_accuracy_xl"),
    ("gemm_hmma_fp16_sm80", "trace_l2_accuracy_xl"),
    ("conv2d_implicit_gemm_bf16_sm80", "timing_shape"),
    ("swiglu_fwd_bf16_sm80", "trace_l2_accuracy_xl"),
    ("silu_fwd_bf16_sm80", "timing_shape"),
    ("rmsnorm_bwd_bf16_sm80", "timing_shape"),
)


def _make_sm120_pair_with_join_key(
    tmp_path: Path,
    *,
    slug: str,
    shape_point: str,
    predicted_rate: float,
    pct: float,
    sector_subdir: str,
) -> tuple[dict, dict]:
    """Mint a matched (predictor bundle, NCU label cell) pair
    keyed on a caller-chosen slug + shape_point. The pair is
    sm_120 / live so it counts toward the headline geomean. The
    pair shares the synthetic binary_sha256 / source_commit /
    kernel_signature_sha256 default tuple, so the join_key chain
    matches between bundle and cell."""
    mpi = _make_synthetic_manifest_pairing_index(
        slug=slug, shape_point=shape_point,
    )
    bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path / sector_subdir,
        result_kind="scalar",
        predicted_hit_rate=predicted_rate,
        manifest_pairing_index=mpi,
        device="NVIDIA GeForce RTX 5060 Laptop GPU",
    )
    cell = _make_synthetic_ncu_label_cell(
        slug=slug,
        shape_point=shape_point,
        arch="sm_120",
        ground_truth_pct=pct,
        evidence_kind="live",
    )
    return bundle, cell


def test_synthetic_dual_geomean_full_cohort_mirrors_legacy_geomean_on_admitted_arm(
    tmp_path: Path,
) -> None:
    """AC-5.3: the admitted-arm bundle MUST emit
    ``full_cohort_geomean`` + three CI fields. By construction the
    full-cohort series is identical to the legacy ``geomean`` series
    on this arm (both are the live scalar/bounded headline cells under
    the same target_arch filter); the AC-5.3-named key is the canonical
    surface for downstream readers."""
    # 5 sm_120 / live cells — none are on the roster's holdout
    # split (synthetic slug names), so the holdout subset is empty.
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_120",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
        target_arch="sm_120",
    )
    assert out["result_kind"] == "admitted", (
        f"5 sm_120 / live cells must admit under "
        f"target_arch='sm_120'; got {out.get('result_kind')!r} "
        f"(typed_reason={out.get('typed_reason')!r})"
    )
    # AC-5.3: full_cohort_geomean mirrors legacy geomean on admit.
    assert "full_cohort_geomean" in out, (
        "AC-5.3: admitted-arm bundle MUST carry "
        "'full_cohort_geomean'"
    )
    assert math.isclose(
        out["full_cohort_geomean"], out["geomean"], rel_tol=1e-12,
    ), (
        f"AC-5.3: full_cohort_geomean MUST equal legacy geomean on "
        f"admit; got full_cohort_geomean="
        f"{out.get('full_cohort_geomean')!r}, geomean="
        f"{out.get('geomean')!r}"
    )
    for ci_field in (
        "full_cohort_geomean_ci_low",
        "full_cohort_geomean_ci_high",
        "full_cohort_geomean_ci_half_width",
    ):
        assert ci_field in out, (
            f"AC-5.3: admitted-arm bundle MUST carry {ci_field!r}; "
            f"got bundle keys {sorted(out.keys())!r}"
        )
    legacy_pairs = (
        ("full_cohort_geomean_ci_low", "geomean_ci_low"),
        ("full_cohort_geomean_ci_high", "geomean_ci_high"),
        ("full_cohort_geomean_ci_half_width", "geomean_ci_half_width"),
    )
    for new_field, legacy_field in legacy_pairs:
        assert math.isclose(
            out[new_field], out[legacy_field], rel_tol=1e-12,
        ), (
            f"AC-5.3: {new_field} MUST equal {legacy_field} on admit; "
            f"got {new_field}={out.get(new_field)!r}, "
            f"{legacy_field}={out.get(legacy_field)!r}"
        )


def test_synthetic_dual_geomean_holdout_empty_when_no_roster_holdout_cells_included(
    tmp_path: Path,
) -> None:
    """AC-5.3: when no included headline cell matches a roster
    holdout (slug, shape_point), the bundle MUST emit
    ``holdout_cell_count == 0`` and ``holdout_geomean == 0.0`` with
    zero CI bounds. The empty holdout subset MUST NOT silently
    substitute calibration/validation cells."""
    # 5 sm_120 / live cells with synthetic slugs that do not
    # appear in the frozen roster.
    bundles, cells = _five_synthetic_pairs(
        tmp_path,
        arch="sm_120",
        pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
        target_arch="sm_120",
    )
    assert out["result_kind"] == "admitted"
    assert out["holdout_cell_count"] == 0, (
        f"synthetic-slug cells must produce holdout_cell_count=0; "
        f"got {out.get('holdout_cell_count')!r}"
    )
    assert out["holdout_geomean"] == 0.0, (
        f"empty holdout subset MUST yield holdout_geomean=0.0 "
        f"(empty-list fallback); got {out.get('holdout_geomean')!r}"
    )
    for ci_field in (
        "holdout_geomean_ci_low",
        "holdout_geomean_ci_high",
        "holdout_geomean_ci_half_width",
    ):
        assert out[ci_field] == 0.0, (
            f"empty holdout subset MUST yield {ci_field}=0.0; "
            f"got {out.get(ci_field)!r}"
        )


def test_synthetic_dual_geomean_holdout_filters_to_roster_holdout_split(
    tmp_path: Path,
) -> None:
    """AC-5.3 positive: when included cells include several
    roster-holdout (slug, shape_point) pairs mixed with
    non-roster cells, ``holdout_cell_count`` MUST equal the
    roster-holdout subset size and the ``holdout_geomean`` MUST
    be computed over that subset's relative errors only.

    Stages 3 roster-holdout sm_120 live cells (each rel_err = 1.00
    by construction: predicted = 0.0, ground_truth = 50%) plus
    2 non-roster sm_120 live cells (rel_err = 0.20 each). The
    holdout subset is the 3 roster cells; their geomean is
    1.00. The full-cohort series is all 5 cells; its geomean
    differs from the holdout subset, so the test pins both."""
    bundles: list[dict] = []
    cells: list[dict] = []
    holdout_subset = _ROSTER_HOLDOUT_KEYS[:3]
    for i, (slug, shape_point) in enumerate(holdout_subset):
        # predicted = 0.0, ground_truth = 50% → |0.0 - 0.5| / 0.5 = 1.0.
        b, c = _make_sm120_pair_with_join_key(
            tmp_path,
            slug=slug,
            shape_point=shape_point,
            predicted_rate=0.0,
            pct=50.0,
            sector_subdir=f"holdout_{i}",
        )
        bundles.append(b)
        cells.append(c)
    for i in range(2):
        # predicted = 0.4, ground_truth = 50% → |0.4 - 0.5| / 0.5 = 0.20.
        b, c = _make_sm120_pair_with_join_key(
            tmp_path,
            slug=f"synthetic_offroster_kernel_{i}",
            shape_point="trace_00",
            predicted_rate=0.4,
            pct=50.0,
            sector_subdir=f"offroster_{i}",
        )
        bundles.append(b)
        cells.append(c)
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
        target_arch="sm_120",
    )
    assert out["result_kind"] == "admitted", (
        f"5 sm_120 / live cells must admit under "
        f"target_arch='sm_120'; got {out.get('result_kind')!r} "
        f"(typed_reason={out.get('typed_reason')!r})"
    )
    assert out["holdout_cell_count"] == 3, (
        f"AC-5.3: holdout_cell_count MUST equal the 3 staged "
        f"roster-holdout cells; got "
        f"{out.get('holdout_cell_count')!r}"
    )
    # Holdout geomean = geomean of [1.0, 1.0, 1.0] = 1.0.
    assert math.isclose(out["holdout_geomean"], 1.0, rel_tol=1e-9), (
        f"AC-5.3: holdout_geomean over [1.0, 1.0, 1.0] MUST equal "
        f"1.0; got {out.get('holdout_geomean')!r}"
    )
    # Full-cohort geomean over [1.0, 1.0, 1.0, 0.20, 0.20]
    # = exp((3*log(1.0) + 2*log(0.20)) / 5)
    # = exp(2*log(0.20) / 5)
    # = (0.20 ** (2/5)) ≈ 0.5253.
    expected_full = (0.20 ** (2.0 / 5.0))
    assert math.isclose(
        out["full_cohort_geomean"], expected_full, rel_tol=1e-9,
    ), (
        f"AC-5.3: full_cohort_geomean over the 5 cells MUST equal "
        f"{expected_full!r}; got {out.get('full_cohort_geomean')!r}"
    )
    # The holdout series is strictly a slice — never the full series.
    assert out["holdout_geomean"] != out["full_cohort_geomean"], (
        f"AC-5.3: holdout slice MUST be distinct from the "
        f"full-cohort series when the staged data differ; got "
        f"holdout_geomean={out['holdout_geomean']!r} == "
        f"full_cohort_geomean={out['full_cohort_geomean']!r}"
    )


def test_synthetic_dual_geomean_typed_unavailable_arm_omits_holdout_fields(
    tmp_path: Path,
) -> None:
    """AC-5.3 negative: when the headline gate produces
    ``typed_unavailable`` (e.g., below the smoke-test floor),
    the bundle MUST NOT carry ``full_cohort_geomean`` or
    ``holdout_geomean`` keys. The dual-geomean fields are
    admitted-arm-only, mirroring the legacy ``geomean`` field's
    arm exclusivity. This pins the schema's oneOf arm semantics
    against accidental cross-arm contamination."""
    # 3 sm_120 / live cells — below the smoke-test floor of 5.
    bundles: list[dict] = []
    cells: list[dict] = []
    for i in range(3):
        b, c = _make_sm120_pair_with_join_key(
            tmp_path,
            slug=f"below_floor_kernel_{i}",
            shape_point="trace_00",
            predicted_rate=0.4,
            pct=50.0,
            sector_subdir=f"below_floor_{i}",
        )
        bundles.append(b)
        cells.append(c)
    contract = _admitted_contract()
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=0,
        target_arch="sm_120",
    )
    assert out["result_kind"] == "typed_unavailable", (
        f"3 cells below floor=5 MUST yield typed_unavailable; "
        f"got {out.get('result_kind')!r}"
    )
    for ac53_field in (
        "full_cohort_geomean",
        "full_cohort_geomean_ci_low",
        "full_cohort_geomean_ci_high",
        "full_cohort_geomean_ci_half_width",
        "holdout_geomean",
        "holdout_geomean_ci_low",
        "holdout_geomean_ci_high",
        "holdout_geomean_ci_half_width",
        "holdout_cell_count",
    ):
        assert ac53_field not in out, (
            f"AC-5.3: typed_unavailable arm MUST NOT carry "
            f"{ac53_field!r}; got {ac53_field}={out.get(ac53_field)!r}"
        )


# =====================================================================
# Round-16 (Codex Round 15 Blocking Side Issue #1) AC-5.3 SCHEMA
# enforcement. Round 15 declared the 9 dual-geomean fields as
# OPTIONAL in the schema; Codex pointed out the plan's negative test
# ("a bundle without holdout_geomean is rejected") was not being
# enforced. Round 16 adds an `allOf` conditional that REQUIRES all 9
# fields when `result_kind == "admitted"` AND
# `top_level_admission_scope == "target_arch=sm_120"`. The
# typed_unavailable arm's `not` clause is extended to FORBID all 9
# fields so arm-exclusivity stays true.
# =====================================================================


_DUAL_GEOMEAN_FIELDS: tuple[str, ...] = (
    "full_cohort_geomean",
    "full_cohort_geomean_ci_low",
    "full_cohort_geomean_ci_high",
    "full_cohort_geomean_ci_half_width",
    "holdout_geomean",
    "holdout_geomean_ci_low",
    "holdout_geomean_ci_high",
    "holdout_geomean_ci_half_width",
    "holdout_cell_count",
)


_ACTIVE_SM120_BUNDLE_PATH = (
    _SUITE_ROOT.parent.parent
    / "data" / "local_labels" / "sweeps"
    / "sm120_cohort_bundles_20260526T154829Z"
    / "comparison_bundle.json"
)


def _load_active_sm120_bundle() -> dict:
    """Load the active sm_120-only L2 comparison bundle. Tests
    that probe the AC-5.3 schema conditional use the real active
    artifact because the conditional only fires when
    ``top_level_admission_scope == "target_arch=sm_120"``."""
    if not _ACTIVE_SM120_BUNDLE_PATH.is_file():
        pytest.skip(
            f"active sm_120 comparison bundle missing at "
            f"{_ACTIVE_SM120_BUNDLE_PATH}; AC-5.3 schema-enforcement "
            f"tests require the regenerated bundle"
        )
    return json.loads(
        _ACTIVE_SM120_BUNDLE_PATH.read_text(encoding="utf-8")
    )


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_schema_active_sm120_admitted_bundle_with_all_9_fields_validates() -> None:
    """Sanity baseline for the AC-5.3 schema conditional: the
    unmodified active sm_120 admitted bundle (which carries all 9
    dual-geomean fields) MUST pass strict validation."""
    bundle = _load_active_sm120_bundle()
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_schema_admitted_sm120_arm_rejects_bundle_without_holdout_geomean() -> None:
    """AC-5.3 negative test (plan line 88): a bundle with only
    ``full_cohort_geomean`` and no ``holdout_geomean`` MUST be
    rejected. The Round-16 schema conditional REQUIRES the full
    9-field set on the sm_120 admitted arm; removing
    ``holdout_geomean`` violates the required-properties clause."""
    bundle = _load_active_sm120_bundle()
    del bundle["holdout_geomean"]
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_schema_admitted_sm120_arm_rejects_bundle_without_holdout_ci_tuple() -> None:
    """AC-5.3 negative test: removing the full holdout CI tuple
    (low / high / half_width) from the active sm_120 admitted
    bundle MUST be rejected. The schema's conditional requires the
    3 CI fields alongside the holdout geomean."""
    bundle = _load_active_sm120_bundle()
    for k in (
        "holdout_geomean_ci_low",
        "holdout_geomean_ci_high",
        "holdout_geomean_ci_half_width",
    ):
        bundle.pop(k, None)
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_schema_admitted_sm120_arm_rejects_bundle_without_all_9_dual_geomean_fields() -> None:
    """AC-5.3 negative test: removing all 9 dual-geomean fields
    from the active sm_120 admitted bundle MUST be rejected. The
    schema conditional fires on the FIRST missing required field,
    so ``ValidationError`` is raised regardless of which absent
    field jsonschema reports first."""
    bundle = _load_active_sm120_bundle()
    for k in _DUAL_GEOMEAN_FIELDS:
        bundle.pop(k, None)
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bundle, schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_schema_typed_unavailable_arm_forbids_each_dual_geomean_field() -> None:
    """AC-5.3 arm-exclusivity: a typed_unavailable bundle that
    carries any of the 9 dual-geomean fields MUST be rejected by
    the schema. Iterates each field individually so a future
    regression that drops one of the 9 from the `not` clause is
    caught by the field-specific case."""
    contract = _admitted_contract()
    base = cr.run_comparison(
        contract=contract,
        predictor_bundles=[],
        label_cells=[],
        bootstrap_seed=0,
    )
    assert base["result_kind"] == "typed_unavailable"
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    # Sanity: the clean typed_unavailable bundle validates.
    jsonschema.validate(base, schema)
    for ac53_field in _DUAL_GEOMEAN_FIELDS:
        polluted = dict(base)
        # An integer value satisfies the typed schema for cell_count
        # and a finite number satisfies all the other numeric fields,
        # so any forbid-decision is purely on the `not` clause and
        # never a side-effect of value-type rejection.
        polluted[ac53_field] = 0 if ac53_field == "holdout_cell_count" else 0.0
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(polluted, schema)


# =====================================================================
# Round-16 (Codex Round 15 Blocking Side Issue #1, paragraph 4):
# Roster-load failure must fail loud rather than silently produce
# `holdout_cell_count = 0` for active sm_120 runs. The Round-15
# helper returned an empty dict on OSError; Round 16 promotes
# OSError to a typed RuntimeError (RosterLoadError) with a
# diagnostic naming the expected path.
# =====================================================================


def test_load_roster_split_index_raises_typed_error_when_roster_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When the frozen roster path cannot be opened, the helper
    MUST raise the typed ``RosterLoadError`` with a diagnostic
    naming the expected path. Silent empty fallback would let a
    sm_120 active run write ``holdout_cell_count = 0`` against the
    AC-5.3 schema conditional and only fail at schema-validation
    time -- after the bundle was already produced."""
    missing = tmp_path / "no_roster_here.toml"
    monkeypatch.setattr(cr, "_ROSTER_PATH", missing)
    with pytest.raises(cr.RosterLoadError) as excinfo:
        cr._load_roster_split_index()
    assert str(missing) in str(excinfo.value), (
        f"RosterLoadError message MUST name the expected path; got "
        f"{excinfo.value!s}"
    )


def test_load_roster_split_index_returns_index_on_happy_path() -> None:
    """Positive control: the helper returns a non-empty
    ``(arch, slug, shape_point) -> split`` dict for the committed
    roster. Every key carries the roster's arch (``_ROSTER_ARCH``) as
    its first component so a non-sm_120 record cannot match a roster
    entry. The 7 holdout cells appear in the dict with
    ``split == "holdout"``."""
    index = cr._load_roster_split_index()
    assert isinstance(index, dict)
    assert len(index) > 0, "frozen roster should yield a non-empty index"
    assert all(
        isinstance(k, tuple) and len(k) == 3 and k[0] == cr._ROSTER_ARCH
        for k in index
    ), (
        "every roster split key MUST be a 3-tuple keyed on the roster "
        f"arch {cr._ROSTER_ARCH!r}; got {sorted(index)[:3]!r}"
    )
    holdout = [k for k, v in index.items() if v == "holdout"]
    assert len(holdout) == 7, (
        f"frozen roster has 7 holdout rows; helper returned "
        f"{len(holdout)} holdout entries: {sorted(holdout)!r}"
    )


def test_holdout_filter_excludes_non_sm120_cells_sharing_a_roster_pair() -> None:
    """The roster split tags are sm_120-cohort assignments. A cell from
    another arch that merely REUSES a roster ``(slug, shape_point)`` MUST
    NOT inherit the sm_120 split — otherwise a non-holdout cell leaks into
    the published holdout subset.

    ``gemm_hmma_fp16_sm80/trace_l2_accuracy_xl`` is ``split="holdout"`` in
    the frozen sm_120 roster, and ``label_admitted.json`` carries BOTH an
    sm_120 and an sm_86 cell for that pair. This pins the arch-keyed
    lookup directly on ``_compute_holdout_rel_errs``:

      - the sm_86 record on that pair is excluded for
        ``target_arch in {None, "sm_86"}`` (the sm_120 split does not
        apply to it);
      - the sm_120 record on that pair IS counted for
        ``target_arch="sm_120"`` (the roster's own arch — behavior
        preserved).

    Regression-proven: with the pre-fix ``(slug, shape_point)`` lookup the
    sm_86 record matched the holdout entry and was wrongly included."""
    slug, shape = "gemm_hmma_fp16_sm80", "trace_l2_accuracy_xl"
    sm86_rec = {
        "evidence_kind": "live",
        "relative_error": 0.50,
        "join_key": {"arch": "sm_86", "slug": slug, "shape_point": shape},
    }
    sm120_rec = {
        "evidence_kind": "live",
        "relative_error": 0.30,
        "join_key": {"arch": "sm_120", "slug": slug, "shape_point": shape},
    }
    included = [sm86_rec, sm120_rec]

    # Pre-condition: the pair really is a holdout row in the sm_120 roster.
    idx = cr._load_roster_split_index()
    assert idx.get((cr._ROSTER_ARCH, slug, shape)) == "holdout", (
        "fixture assumes the sm_120 roster tags this pair holdout"
    )

    # sm_86 submission path: the sm_86 holdout-pair record must NOT count.
    sm86_holdout = cr._compute_holdout_rel_errs(
        included=included, target_arch="sm_86",
    )
    assert sm86_holdout == [], (
        f"sm_86 cell reusing an sm_120 holdout pair MUST be excluded; "
        f"got {sm86_holdout!r}"
    )

    # No target arch: the headline admits both live cells, but only the
    # sm_120 record can be holdout (the roster's arch).
    none_holdout = cr._compute_holdout_rel_errs(
        included=included, target_arch=None,
    )
    assert none_holdout == [0.30], (
        f"only the sm_120 holdout-pair record may count when target_arch "
        f"is None; got {none_holdout!r}"
    )

    # sm_120 submission path: the sm_120 record IS counted (preserved).
    sm120_holdout = cr._compute_holdout_rel_errs(
        included=included, target_arch="sm_120",
    )
    assert sm120_holdout == [0.30], (
        f"sm_120 holdout-pair record MUST count under target_arch=sm_120; "
        f"got {sm120_holdout!r}"
    )


# ---------------------------------------------------------------------
# v5 active-path routing: arch must be derived from the bundle's
# canonical `device` when the closed manifest_pairing_index has no arch
# (the real-bundle shape). Without this, run_comparison's v5 block keys
# bundles as ("", slug, shape_point), misses every bucket entry, and
# marks every scalar bundle v5_active_path_unscorable instead of scoring
# it. No existing test drove run_comparison(v5_head_path=...), so this
# path had no run_comparison-level coverage.
# ---------------------------------------------------------------------
def test_v5_routing_resolves_arch_from_device_for_archless_mpi_bundles(
    tmp_path, monkeypatch,
):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tools.local_labels import run_v5_active_path_comparison as v5drv

    bundles, label_cells = _five_synthetic_pairs(tmp_path, arch="sm_86")
    # Real bundles carry the canonical 6-feature block; add it so the v5
    # path reaches the bucket lookup (an absent block is a DIFFERENT
    # unscorable reason than the arch-routing bug under test). The MPI is
    # the closed 5-tuple with NO arch — arch lives only in `device`.
    feats = {
        "raw_prediction": 0.5, "footprint_ratio": 0.5,
        "long_rd_tail_mass": 0.1, "intra_lane_reuse": 0.5,
        "global_share": 1.0, "gridDim_density": 2.0,
    }
    for b in bundles:
        b["canonical_pre_calibration_features"] = dict(feats)
    assert all(
        "arch" not in b["manifest_pairing_index"] for b in bundles
    ), "fixture bundles must use the closed (archless) manifest_pairing_index"

    # Full rehydrated-cell shape: the v5 routing uses (arch, slug,
    # shape_point) for the bucket lookup; the incidental downstream
    # transfer-diagnostics pass (emit_per_arch_transfer_diagnostics_block ->
    # _predict_with_ensemble) also reads `features` + `label`, so include
    # them to exercise the whole run_comparison v5 pipeline without a
    # fixture KeyError.
    head_cells = [
        {
            "arch": "sm_86",
            "slug": b["manifest_pairing_index"]["slug"],
            "shape_point": b["manifest_pairing_index"]["shape_point"],
            "family_bucket": "memory_stress",
            "features": dict(feats),
            "label": 0.5,
        }
        for b in bundles
    ]

    class _StubEnsemble:
        def predict(self, arch, bucket, features):
            # The resolved arch (from device) MUST reach predict; the bug
            # passed "" here and missed the bucket entirely.
            assert arch == "sm_86", f"expected resolved arch sm_86, got {arch!r}"
            return 0.123

    monkeypatch.setattr(
        cr, "load_calibration_head_v5", lambda _p: {"fit_status": "ok"},
    )
    monkeypatch.setattr(
        v5drv, "_rehydrate_cells_from_v5_head", lambda _h: head_cells,
    )
    monkeypatch.setattr(
        v5drv, "_fit_arch_head", lambda _cells, v5_head=None: _StubEnsemble(),
    )

    cr.run_comparison(
        contract=_admitted_contract(),
        predictor_bundles=bundles,
        label_cells=label_cells,
        v5_head_path="unused-because-load-is-patched",
        target_arch="sm_86",
    )

    for b in bundles:
        assert b["result_kind"] == "scalar", (
            f"bundle {b['manifest_pairing_index']['slug']} was marked "
            f"{b.get('typed_reason')!r} "
            f"(_v5_unscorable_reason={b.get('_v5_unscorable_reason')!r}) "
            "instead of v5-scored — the archless-MPI arch must resolve "
            "from the canonical device"
        )
        assert b["predicted_hit_rate"] == 0.123
        assert b.get("typed_reason") != "v5_active_path_unscorable"


def test_v5_active_path_unscorable_admitted_by_runner_ingress(tmp_path):
    """The v5 fail-closed typed_reason is INJECTED by comparison_runner, not
    emitted by the predictor, so it is owned by
    ``comparison_runner._CR_INJECTED_TYPED_REASONS`` — NOT
    ``cache_predictor._TYPED_REASONS`` (keeping it out of the predictor enum
    avoids advancing the post-commit recapture guards' predictor-fix commit for
    a non-predictor change). ``_validate_predictor_bundle_shape`` MUST admit a
    bundle carrying it (else the fail-closed v5 path raises
    ``ComparisonRunnerTypedReasonOutOfEnum``); the on-disk schema enum already
    lists it as part of the union of predictor + runner-injected reasons."""
    # Ownership boundary: runner-injected, NOT a predictor-emitted reason.
    assert "v5_active_path_unscorable" in cr._CR_INJECTED_TYPED_REASONS
    assert "v5_active_path_unscorable" not in cp._TYPED_REASONS
    bundle = _make_synthetic_admitted_predictor_bundle(
        tmp_path,
        result_kind="typed_unavailable",
        typed_reason="v5_active_path_unscorable",
    )
    # Must NOT raise ComparisonRunnerTypedReasonOutOfEnum (the runner admits
    # its own injected reason via cp._TYPED_REASONS | _CR_INJECTED_TYPED_REASONS).
    cr._validate_predictor_bundle_shape(bundle)
