# SM86 scope-compatible DRAM residual ledger — 2026-10-15

## Analysis

H43 joins H42 scope-aware DRAM rows to H37 L2 route/fill-pressure and H38 L1
MSHR pressure observers. NCU remains a post-replay target only. The retained
full corpus provides 58 scope-compatible exact DRAM joins with mean APE 26.84%.

All 58 joins are in the balanced route stratum; concentrated/incomplete route
classes have no exact DRAM joins, so route timing cannot be identified. Pressure
strata are non-monotonic and confounded:

| partition | n | mean DRAM APE |
|---|---:|---:|
| low pending-merge pressure quartile | 15 | 28.69% |
| high pending-merge pressure quartile | 14 | 12.47% |
| low new-fill pressure quartile | 15 | 40.92% |
| high new-fill pressure quartile | 16 | 8.59% |
| zero/low MSHR-stall group | 39 | 28.59% |
| high MSHR-stall group | 19 | 23.25% |

The apparent improvement at high pressure is not a transferable service law:
pressure co-varies with workload family, launch shape, and demand volume, and
the exact joins are sparse. Public partitioned-DRAM queue semantics remain a
mechanism prior only.

## Decision

No partition/hash selection, queue delay, bandwidth multiplier, or NCU-derived
correction is promoted. The active 26.84% scope-compatible DRAM residual needs
a new locked probe varying one demand factor while independently accounting for
dirty writeback/coherence tails. The existing H41 compressed replay/report are
the retained evidence base.
