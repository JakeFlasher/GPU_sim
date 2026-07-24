"""sm_120 coverage-expansion subpackage.

Owns the closed-vocabulary state machine, total reducer, schema
validator, disk-discovery probe, and emission script for the
`sm120_coverage_readiness_ladder_v1.toml` manifest.

The previously-fit calibrator + validation + exclusion-manifest
triple (`isotonic_params_sm120.toml`,
`sm120_validation_report_v1.toml`,
`sm120_simulator_unavailable_cells_v1.toml`,
`label_ledger_v1_sm120_partition.toml`,
`historical_label_split_v1_sm120.toml`) is the locked reference
baseline. Modules in this subpackage READ those artifacts to
build the ladder's diagnostics and env inputs; they do NOT
mutate any of them. Immutability is gated by
`scripts/baselines_harness/tests/test_baseline_artifacts_immutability.py`.

Importing this package does NOT touch disk, run NCU, or launch
any subprocess; emission is invoked explicitly through
`emit_ladder.emit_coverage_readiness_ladder(...)`.
"""

from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_states import (
    LADDER_QA_TYPED_FAILURES,
    LadderQaError,
    _DIAGNOSTIC_TYPED_REASONS,
    _LADDER_STATES,
    _VALID_SIMULATOR_OOD_TYPED_REASONS,
)
from sol_baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.ladder_reducer import (
    emit_ladder_state,
)

__all__ = [
    "LADDER_QA_TYPED_FAILURES",
    "LadderQaError",
    "_DIAGNOSTIC_TYPED_REASONS",
    "_LADDER_STATES",
    "_VALID_SIMULATOR_OOD_TYPED_REASONS",
    "emit_ladder_state",
]
