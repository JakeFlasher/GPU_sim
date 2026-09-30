# Per-endpoint regime committee (2026-09-30)

`tools/sm86_endpoint_regime_committee.py` makes the endpoint-domain policy
explicit rather than routing both targets through one source:

| phase | GPC source | duration source |
|---|---|---|
| cold | clock elasticity | native-event/frontier crosswalk |
| warm | native-event/frontier crosswalk | same-regime native-event ratio |

The policy is fixed by endpoint and phase semantics before held-out NCU values
are loaded. All source models are family-disjoint and every row receives a
candidate/fallback interval.

## Live results

| phase | rows | GPC MAPE | duration MAPE |
|---|---:|---:|---:|
| cold | 15/15 | **7.49%** | **10.99%** |
| warm | 15/15 | **11.23%** | **9.62%** |

This is a stronger non-promoted endpoint committee than the earlier single
source router. It still does not establish a common NCU/CUDA-event/simulator
clock, physical pipe identity, or cache ownership/read-from semantics.

