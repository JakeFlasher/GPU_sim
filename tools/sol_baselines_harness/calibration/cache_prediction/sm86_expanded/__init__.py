"""sm_86 expanded NCU read-only-sector ground-truth harvest.

Expands the frozen 5-cell pilot label ledger
(``label_ledger_v1_sm86_pilot_v1.toml``) to the ported classifiable
problem set by profiling the per-problem standalone probe binaries under
``nvbit_tracer/metric_extractor/tests/calibration_cohort_b/`` with NCU,
3 runs per cell, against the pinned read-only-sector metric family
(``ncu_harvester.NCU_REQUIRED_METRICS``) at ``--clock-control none``.

CLAUDE.md Rule 1: NCU is calibration-only on sm_86 / sm_120; this package
lives under ``calibration/cache_prediction/`` and is never on the deployed
or B200 path. Rule 2: every invocation pins ``--clock-control none`` and
carries no lock-clock flag (enforced by ``ncu_harvester._build_invocation``
and the argv-provenance guard test).
"""
