# Warm-cache scenario interval (2026-09-29)

The warm clock campaign showed that a point warm/cold ratio is not portable
across families. `tools/sm86_warm_cache_scenario_interval.py` therefore emits a
point estimate plus a deliberately wide scenario envelope:

```text
warm point = cold endpoint projection × median(calibration warm/cold ratio)
interval   = cold endpoint × [min ratio - q90 residual, max ratio + q90 residual]
```

Calibration and score families are disjoint within the same high-clock/cache
regime. NCU score labels are loaded only after the point and interval are
formed. The interval is not a confidence interval and does not identify cache
ownership, read-from source, eviction, or DRAM completion.

## Live result

On the 15-row high-clock warm score:

| endpoint | point MAPE | interval coverage |
|---|---:|---:|
| GPC cycles | 16.016% | **93.33%** |
| duration | 16.748% | **93.33%** |

Model runtime is ~0.01 s. Reports are under
`results/alpha/sm86_warm_cache_scenario_goal_20260929/`.

The point error is intentionally not promoted; the value of this lane is an
explicit cache-phase uncertainty envelope with full workload coverage.
