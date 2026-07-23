"""Policy debt gate for the eight Stage-7 SOL NVBit injectors."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUARANTINED = (
    "tools/sol_baselines_harness/surfaces/nvbit_metrics.py",
    "tools/sol_baselines_harness/metric_extractor/calibration_harness.py",
    "tools/sol_baselines_harness/calibration/cache_prediction/end_to_end_smoke.py",
    "tools/sol_baselines_harness/calibration/cache_prediction/sm120_coverage_expansion/harvest_tier2_pilot.py",
    "tools/sol_baselines_harness/calibration/cache_prediction/sm120_l2_associativity_probe/harvest_tier2_traces.py",
    "tools/sol_baselines_harness/calibration/cache_prediction/sm86_expanded/emit_expanded_raw_predictions.py",
    "tools/sol_baselines_harness/calibration/cache_prediction/sm86_geometry_sweep/harvest_sweep_traces.py",
    "tools/sol_baselines_harness/calibration/cache_prediction/sm86_pilot/emit_tier2_sidecars.py",
)


def test_stage7_quarantine_inventory_is_exact_and_present() -> None:
    assert len(QUARANTINED) == len(set(QUARANTINED)) == 8
    for relative in QUARANTINED:
        path = REPO_ROOT / relative
        assert path.is_file() and not path.is_symlink(), relative


def test_quarantined_injectors_do_not_claim_current_runtime_enforcement() -> None:
    forbidden = (
        "from tools.nvbit_tracer.runtime_env import",
        "import tools.nvbit_tracer.runtime_env",
        "tracer_runtime_env(",
        "universally guarded",
        "current runtime contract enforced",
    )
    for relative in QUARANTINED:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, (relative, token)
