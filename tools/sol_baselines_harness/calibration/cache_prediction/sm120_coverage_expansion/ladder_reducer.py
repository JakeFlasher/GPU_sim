"""Total reducer mapping per-cell diagnostics + per-slug env
memberships to exactly one ``_LADDER_STATES`` token.

Contract (coverage-readiness ladder + quarantine-sidecar
specification):

- Total over the 235-problem corpus: every slug → exactly one
  ``_LADDER_STATES`` member.
- Acyclic: typed-unavailable arch-incompat short-circuits
  everything; simulator-gap typed-unavailable short-circuits
  before any diagnostics check; quarantine / design-coverage
  blockers cap advancement at the cell's preceding
  pipeline-progress prefix (typically ``sol_execbench_correct``);
  normal-progression branch only fires when no blockers are
  present.

Caller contract: the reducer is invoked for every one of the 235
problem slugs (single source-of-truth pass). Slugs whose harvest
pipeline never touched them pass an empty ``diagnostics`` dict;
the reducer returns ``spec_only`` for those (slug not in
``env["cu_present"]`` falls through to the ``spec_only`` return).
There is NO separate caller pre-filter that emits ``spec_only``
directly — the reducer is total.
"""

from __future__ import annotations

from typing import Any, Mapping

from baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_states import (
    _LADDER_STATES,
)


def emit_ladder_state(
    slug: str,
    diagnostics: Mapping[str, Any],
    env: Mapping[str, set[str]],
) -> str:
    """Reduce per-cell diagnostics + env memberships → ladder state.

    Order of checks is load-bearing:

    1. Typed-unavailable arch-incompat (NVFP4 / TCGen05 / wgmma)
       short-circuits everything.
    2. Simulator-gap typed-unavailable (Extension A2 / B) short-
       circuits before any diagnostics signal.
    3. Compute blocker booleans (predictor-design-coverage missing,
       NCU CV > 0.05 quarantine, Tier-2 quarantine).
    4. GLOBAL ADVANCEMENT CAP: if any blocker is true, cap the
       state at the cell's preceding pipeline-progress prefix
       (``sol_execbench_correct`` if reached, else ``cu_compiles``
       if reached, else ``cu_compile_failed`` if a ``*.cu`` exists,
       else ``spec_only``).
    5. Normal progression: highest-first match (``mape_scored`` →
       ``predictor_prediction_available`` → ``tier2_trace_available``
       → ``ncu_label_available`` → ``sol_execbench_correct`` →
       ``cu_compiles`` → ``cu_compile_failed`` → ``spec_only``).

    Both ``ncu_label_available`` and ``tier2_trace_available``
    states require their respective quarantine flags clear; this
    is the contract that NCU-CV-quarantined or Tier-2-quarantined
    cells cannot advance past ``sol_execbench_correct``.

    Args:
        slug: the problem slug being reduced.
        diagnostics: per-cell diagnostic dict; absent keys default
            False (booleans) / None (scalars). Recognized keys
            include ``cu_compiles``, ``sol_execbench_correct``,
            ``ncu_attempted``, ``ncu_cv_observed``,
            ``ncu_label_available``, ``tier2_attempted``,
            ``tier2_quarantine_reason``, ``tier2_trace_available``,
            ``predictor_design_coverage_missing``,
            ``predictor_prediction_available``, ``mape_scored``.
        env: per-env-set slug memberships; missing keys default to
            empty set. Recognized keys: ``cu_present``,
            ``arch_incompat_nvfp4_tcgen05``, ``ext_a2_pending``,
            ``ext_b_pending``.

    Returns:
        Exactly one ``_LADDER_STATES`` member.
    """
    if slug in env.get("arch_incompat_nvfp4_tcgen05", set()):
        return "typed_unavailable_arch_nvfp4_tcgen05"
    if slug in env.get("ext_a2_pending", set()):
        return "typed_unavailable_simulator_gap_extension_a2"
    if slug in env.get("ext_b_pending", set()):
        return "typed_unavailable_simulator_gap_extension_b"

    pred_design_missing = bool(
        diagnostics.get("predictor_design_coverage_missing", False)
    )
    ncu_cv = diagnostics.get("ncu_cv_observed")
    ncu_quarantined = bool(
        diagnostics.get("ncu_attempted", False)
        and ncu_cv is not None
        and ncu_cv > 0.05
    )
    # tier2_quarantined fires only when the diagnostics row carries
    # a NON-EMPTY typed quarantine reason. The diagnostics schema's
    # success sentinel for ``tier2_quarantine_reason`` is the empty
    # string ``""`` (matching the closed-vocabulary success token
    # used elsewhere), NOT ``None`` — a clean ``tier2_attempted=true``
    # row with ``tier2_quarantine_reason=""`` MUST NOT be treated as
    # quarantined, otherwise every successful Tier-2 row caps at
    # ``sol_execbench_correct`` instead of advancing past it.
    tier2_reason = diagnostics.get("tier2_quarantine_reason")
    tier2_quarantined = bool(
        diagnostics.get("tier2_attempted", False)
        and tier2_reason is not None
        and tier2_reason != ""
    )
    ncu_admitted = (
        bool(diagnostics.get("ncu_label_available", False))
        and not ncu_quarantined
    )
    tier2_admitted = (
        bool(diagnostics.get("tier2_trace_available", False))
        and not tier2_quarantined
    )

    any_blocker = (
        pred_design_missing or ncu_quarantined or tier2_quarantined
    )
    if any_blocker:
        return _preceding_pipeline_prefix(slug, diagnostics, env)

    if (
        bool(diagnostics.get("mape_scored", False))
        and ncu_admitted
        and tier2_admitted
    ):
        return "mape_scored"
    if (
        bool(diagnostics.get("predictor_prediction_available", False))
        and ncu_admitted
        and tier2_admitted
    ):
        return "predictor_prediction_available"
    if tier2_admitted and ncu_admitted:
        return "tier2_trace_available"
    if ncu_admitted:
        return "ncu_label_available"
    if bool(diagnostics.get("sol_execbench_correct", False)):
        return "sol_execbench_correct"
    if bool(diagnostics.get("cu_compiles", False)):
        return "cu_compiles"
    if slug in env.get("cu_present", set()):
        return "cu_compile_failed"
    return "spec_only"


def _preceding_pipeline_prefix(
    slug: str,
    diagnostics: Mapping[str, Any],
    env: Mapping[str, set[str]],
) -> str:
    """Resolve the cell's deepest pipeline-progress prefix when a
    quarantine / design-coverage blocker caps advancement.

    The cap is set at ``sol_execbench_correct`` when correctness
    is observed, even if NCU / Tier-2 / predictor outputs are
    also recorded — those higher signals are voided by the
    blocker. Without correctness, fall back to ``cu_compiles``,
    then ``cu_compile_failed``, then ``spec_only``.
    """
    if bool(diagnostics.get("sol_execbench_correct", False)):
        return "sol_execbench_correct"
    if bool(diagnostics.get("cu_compiles", False)):
        return "cu_compiles"
    if slug in env.get("cu_present", set()):
        return "cu_compile_failed"
    return "spec_only"


def assert_state_in_vocabulary(state: str) -> None:
    """Defensive guard for callers that synthesize ladder states
    outside this reducer (e.g., emission scripts post-processing).
    Raises ``ValueError`` if ``state`` is not a ``_LADDER_STATES``
    member."""
    if state not in _LADDER_STATES:
        raise ValueError(
            f"ladder state {state!r} not in _LADDER_STATES; "
            f"valid members are {sorted(_LADDER_STATES)}"
        )
