"""sm_86 empirical L2-associativity geometry sweep (keystone predictor accuracy-gate path).

Mirrors the sm_120 `sm120_l2_associativity_probe/` precedent: re-harvest a
per-cell Tier-2 NVBit address trace once, replay the SetAssociativeSimulator at
each candidate `associativity_l2 in {4, 8, 16, 32, 64}`, and select the geometry
whose re-fit isotonic calibrator best satisfies the tiered MAPE accuracy gate.
NVBit is calibration-only (CLAUDE.md Rule 1); nothing here is on the deployed /
B200 path.
"""
