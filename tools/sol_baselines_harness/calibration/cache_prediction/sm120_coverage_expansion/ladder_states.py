"""Closed-vocabulary state + typed-reason frozensets for the
sm_120 coverage-readiness ladder.

Three orthogonal closed sets live here:

1. ``_LADDER_STATES`` — exactly 11 members; the per-problem
   highest-state tag the ladder manifest carries.

2. ``_DIAGNOSTIC_TYPED_REASONS`` — typed reasons that record
   *why* a cell did not advance further. Lives in the sibling
   diagnostics sidecar (``sm120_coverage_expansion_diagnostics_v1.toml``);
   distinct from ladder states.

3. ``_VALID_SIMULATOR_OOD_TYPED_REASONS`` — typed reasons that
   appear in the previously-fit calibrator's
   ``sm120_simulator_unavailable_cells_v1.toml`` manifest's
   ``[[excluded]].typed_reason`` field. Regression-pinned here
   so any change to the baseline vocabulary surfaces in the
   regression sweep, not silently.

A fourth set, ``LADDER_QA_TYPED_FAILURES``, enumerates the typed
failure tokens the ladder QA validator raises. It is the public
contract subsequent test modules assert against.
"""

from __future__ import annotations

from typing import Final

# Closed vocabulary of per-problem ladder states. Order in this
# tuple is documentation only (semantic-progression order); the
# frozenset is unordered. Members:
#
#   - ``spec_only``                                — markdown spec exists; no ``*.cu`` on disk for the slug.
#   - ``cu_compile_failed``                        — ``*.cu`` exists but did not compile under nvcc for the local arch.
#   - ``cu_compiles``                              — ``*.cu`` compiles under nvcc for the local arch.
#   - ``sol_execbench_correct``                    — passes ``sol-execbench`` correctness with ``target_hardware=LOCAL``.
#   - ``ncu_label_available``                      — at least one NCU read-only sector hit-rate label-run completed.
#   - ``tier2_trace_available``                    — Tier-2 NVBit address-trace sidecar harvested.
#   - ``predictor_prediction_available``           — set-associative simulator produced raw predictions; calibrator emitted calibrated predictions.
#   - ``mape_scored``                              — cell is in the validation split and has a per-cell MAPE.
#   - ``typed_unavailable_arch_nvfp4_tcgen05``     — kernel cannot execute on sm_120 (NVFP4 / TCGen05 / wgmma).
#   - ``typed_unavailable_simulator_gap_extension_a2``  — kernel awaits ConstantCacheSimulator submodel.
#   - ``typed_unavailable_simulator_gap_extension_b``   — kernel awaits per-warp / reuse-distance modeling.
_LADDER_STATES: Final[frozenset[str]] = frozenset({
    "spec_only",
    "cu_compile_failed",
    "cu_compiles",
    "sol_execbench_correct",
    "ncu_label_available",
    "tier2_trace_available",
    "predictor_prediction_available",
    "mape_scored",
    "typed_unavailable_arch_nvfp4_tcgen05",
    "typed_unavailable_simulator_gap_extension_a2",
    "typed_unavailable_simulator_gap_extension_b",
})

# Closed vocabulary of diagnostic typed reasons recorded in
# ``sm120_coverage_expansion_diagnostics_v1.toml`` per-cell rows.
# Distinct from ladder states: a single cell may carry one or
# more typed reasons in the diagnostics sidecar while the ladder
# only records the single highest-state token.
_DIAGNOSTIC_TYPED_REASONS: Final[frozenset[str]] = frozenset({
    "ncu_repeatability_failed",
    "tier2_budget_exceeded",
    "tier2_no_launches_captured",
    "predictor_design_coverage_missing",
    "sol_execbench_correctness_failed",
})

# Closed vocabulary of simulator-OOD typed reasons that appear in
# the previously-fit calibrator's
# ``sm120_simulator_unavailable_cells_v1.toml`` manifest.
# Regression-pinned: any change to this set MUST be paired with a
# baseline-immutability re-baseline (which the current plan
# explicitly forbids). The two existing reasons map to:
#
#   - ``simulator_l2_metric_family_undercount`` — pending the
#     ConstantCacheSimulator submodel (Extension A2; 3 cells in
#     the previously-fit baseline:
#     ``problem_011_rotary_position_embedding``,
#     ``problem_034_flux_multi_axis_rope_frequency_computation``,
#     ``problem_038_flux_multi_head_rmsnorm_qk``).
#   - ``hostile_subclass_uncalibrated`` — pending per-warp /
#     reuse-distance modeling (Extension B; 2 cells in the
#     previously-fit baseline:
#     ``problem_012_gqa_paged_decode_h32_kv4_d128_ps1``,
#     ``problem_014_gqa_paged_prefill_causal_h32_kv4_d128_ps1``).
_VALID_SIMULATOR_OOD_TYPED_REASONS: Final[frozenset[str]] = frozenset({
    "simulator_l2_metric_family_undercount",
    "hostile_subclass_uncalibrated",
})

# Ladder QA typed-failure tokens raised by
# ``ladder_qa.validate_ladder_manifest``. These are the public
# contract the ladder-test module's planted-mutation negatives
# assert against.
LADDER_QA_TYPED_FAILURES: Final[frozenset[str]] = frozenset({
    "ladder_corpus_size_mismatch",
    "ladder_duplicate_problem",
    "ladder_state_out_of_vocabulary",
    "ladder_ordering_violation",
    "ladder_meta_field_missing",
    "ladder_meta_field_invalid",
    "ladder_corpus_sha256_mismatch",
})


class LadderQaError(Exception):
    """Raised by ``validate_ladder_manifest`` with a typed-failure
    token in ``token`` so callers can dispatch on the reason
    without re-parsing the message."""

    def __init__(self, token: str, detail: str = "") -> None:
        if token not in LADDER_QA_TYPED_FAILURES:
            raise ValueError(
                f"unknown ladder QA token {token!r}; "
                f"valid tokens are {sorted(LADDER_QA_TYPED_FAILURES)}"
            )
        super().__init__(f"{token}: {detail}" if detail else token)
        self.token = token
        self.detail = detail


# Tuple variant kept for typed-vocabulary regression tests that
# want order-sensitive membership comparisons.
LADDER_STATES_TUPLE: Final[tuple[str, ...]] = tuple(sorted(_LADDER_STATES))
DIAGNOSTIC_TYPED_REASONS_TUPLE: Final[tuple[str, ...]] = tuple(
    sorted(_DIAGNOSTIC_TYPED_REASONS)
)
