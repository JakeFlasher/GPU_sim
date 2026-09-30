# SM86 reuse-distance cache-phase projection (2026-09-30)

## Model

`tools/sm86_reuse_distance_phase_projection.py` implements a cache-pressure
alternative inspired by the public GPU cache-model source.  It reconstructs
only explicitly global 32-byte sectors from each retained threadblock trace and
uses a Fenwick tree to compute per-threadblock stack/reuse distances in
`O(N log N)`.  It emits access count, cold fraction, reuse quantiles, L1/L2
capacity-threshold fractions, and maximum reuse span.

The model fits `log(warm/cold)` endpoint ratios on paired cold/warm campaigns
from the opposite workload cohort.  Cold endpoints come from a
score-label-free frontier/archetype prediction report; score NCU observations
are consumed only after the warm prediction for APE reporting.  It is a cache
signature, not a directory, owner, read-from, eviction, or DRAM-completion
model.

## Holdout results

The current per-kernel-clock campaigns are the score set.  The ratio model is
trained on the 2026-09-29 paired refined campaigns, with base and large cohorts
held out in turn:

| score cohort | GPC MAPE | duration MAPE |
|---|---:|---:|
| warm / large | 20.32% | 19.65% |
| warm / base | 12.06% | 25.16% |

The all-row output covers 15/15 rows, but the shifted warm cache history is not
captured well enough for promotion.  Trace walking takes roughly 19 seconds
for the larger cohort, so this is also not yet a fast-path candidate.

## Interpretation

Reuse distance is useful as an uncertainty and phase descriptor, particularly
when paired with independent cache microbenchmarks, but it cannot substitute
for the proprietary GA102 ownership/read-from/flush state.  The negative result
is retained because it falsifies the simpler hypothesis that a per-block reuse
signature alone explains the warm endpoint shift.

The implementation follows the public [GPU cache model](https://github.com/tue-es/gpu-cache-model)'s
efficient reuse-distance/warp-pool approach, while the endpoint separation is
consistent with [GCStack+GCScaler](https://github.com/yonsei-hpcp/gcstack_gcscaler)
and the configurable cache abstractions documented by
[Accel-Sim/GPGPU-Sim](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md).
