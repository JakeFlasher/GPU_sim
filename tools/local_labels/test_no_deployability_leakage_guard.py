"""Tests for the no-deployability-leakage guard."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for p in (_HERE, _REPO_ROOT / "workloads" / "simbench_bw"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import no_deployability_leakage_guard as guard  # noqa: E402
from canonical_features import CANONICAL_FEATURE_NAMES  # noqa: E402

# The achieved-occupancy + IPC field names the guard must forbid as features.
_ACHIEVED_OCCUPANCY_FIELD = "occupancy_achieved_warps_pct"
_IPC_FIELD = "ipc_inst_per_cycle"
_ACHIEVED_OCCUPANCY_NCU = "sm__warps_active.avg.pct_of_peak_sustained_active"
_IPC_NCU = "sm__inst_executed.avg.per_cycle_elapsed"
_THEORETICAL_OCCUPANCY_FIELD = "occupancy_limit_warps_theoretical"


class LeakageGuardContractTest(unittest.TestCase):
    def test_achieved_occupancy_and_ipc_are_label_only_aliases(self) -> None:
        forbidden = guard.label_only_feature_aliases()
        for name in (
            _ACHIEVED_OCCUPANCY_FIELD,
            _IPC_FIELD,
            _ACHIEVED_OCCUPANCY_NCU,
            _IPC_NCU,
        ):
            self.assertIn(name, forbidden, msg=f"{name} must be a forbidden feature")

    def test_theoretical_occupancy_is_feature_eligible_not_forbidden(self) -> None:
        forbidden = guard.label_only_feature_aliases()
        eligible = guard.feature_eligible_aliases()
        self.assertIn(_THEORETICAL_OCCUPANCY_FIELD, eligible)
        self.assertNotIn(_THEORETICAL_OCCUPANCY_FIELD, forbidden)


class CanonicalFeaturesCleanTest(unittest.TestCase):
    """Positive control: the real deployable inference feature contract carries
    no achieved-occupancy / IPC leakage."""

    def test_canonical_features_have_no_label_only_leakage(self) -> None:
        leaks = guard.scan_features_for_leakage(CANONICAL_FEATURE_NAMES)
        self.assertEqual(leaks, [], msg=f"canonical features leaked: {leaks}")
        # And the assert form does not raise.
        guard.assert_no_deployability_leakage(CANONICAL_FEATURE_NAMES)

    def test_theoretical_occupancy_feature_is_allowed(self) -> None:
        features = list(CANONICAL_FEATURE_NAMES) + [_THEORETICAL_OCCUPANCY_FIELD]
        guard.assert_no_deployability_leakage(features)  # must not raise


class LeakageDetectedTest(unittest.TestCase):
    """Negative control: an inference feature surface containing achieved
    occupancy or IPC fails the guard."""

    def test_achieved_occupancy_field_in_features_fails(self) -> None:
        features = list(CANONICAL_FEATURE_NAMES) + [_ACHIEVED_OCCUPANCY_FIELD]
        with self.assertRaises(guard.DeployabilityLeakageError):
            guard.assert_no_deployability_leakage(features)

    def test_ipc_field_in_features_fails(self) -> None:
        features = list(CANONICAL_FEATURE_NAMES) + [_IPC_FIELD]
        with self.assertRaises(guard.DeployabilityLeakageError):
            guard.assert_no_deployability_leakage(features)

    def test_achieved_occupancy_ncu_name_in_features_fails(self) -> None:
        # Even if a feature surface references the raw NCU metric id rather than
        # the label-style field name, the guard catches it.
        with self.assertRaises(guard.DeployabilityLeakageError):
            guard.assert_no_deployability_leakage([_ACHIEVED_OCCUPANCY_NCU])

    def test_scan_reports_the_leaking_field_and_metric(self) -> None:
        leaks = guard.scan_features_for_leakage([_IPC_FIELD, "raw_prediction"])
        self.assertEqual(len(leaks), 1)
        self.assertEqual(leaks[0]["feature_field"], _IPC_FIELD)
        self.assertEqual(leaks[0]["label_only_metric_key"], "ipc_inst_per_cycle")


if __name__ == "__main__":
    unittest.main()
