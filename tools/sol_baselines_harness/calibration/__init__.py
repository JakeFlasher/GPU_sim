"""Calibration namespace for the Tier-2 cache predictor.

Only the sub-package `cache_prediction/` is PERMITTED to invoke
Nsight Compute (the calibration-only carveout per CLAUDE.md
Rule 1). Other modules directly under this package — including
this `__init__.py` itself — remain in the deployed framework
path and MUST NOT reference Nsight Compute, its CLI, its
metric names, or any module under `cache_prediction/`. The
quarantine static-guard at
`scripts/baselines_harness/tests/test_ncu_quarantine.py`
mechanically enforces this boundary.
"""
