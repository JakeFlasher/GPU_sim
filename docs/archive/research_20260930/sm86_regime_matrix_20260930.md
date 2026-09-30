# SM86 regime matrix (2026-09-30)

The consolidated matrix compares fixed route policies across fresh temporal
cells:

| phase | regime | route | GPC MAPE | duration MAPE |
|---|---|---|---:|---:|
| cold | low/unknown | neutral event/frontier | 8.36% | 8.08% |
| warm | low/unknown | neutral event/frontier | 8.02% | 7.90% |
| cold | high | clock elasticity | 9.88% | 9.67% |
| warm | high | clock elasticity | 9.36% | 9.76% |

This is a post-hoc comparison of already fixed reports, not a new fit or
route-selection procedure. High clock response is used only with independent
regime evidence; unknown regimes remain neutral.

A fractional blend using the observed high-burst fraction was also evaluated:
it reached 9.87%/10.23% cold and 9.37%/9.56% warm GPC/duration, essentially
matching the hard high-regime route. The hard route remains the simpler policy.

Artifact: `results/alpha/sm86_regime_matrix_goal_20260930/matrix.json`.
