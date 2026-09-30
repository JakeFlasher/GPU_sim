# Regime-aware endpoint router v2 (2026-09-30)

`tools/sm86_endpoint_regime_router_v2.py` requires an independent clock
observation before enabling the high-clock committee. If the observation is
absent or classifies the campaign as low/unknown, it routes both endpoints
through the neutral native-event/frontier crosswalk and records the uncertainty
explicitly. Held-out NCU frequencies never select the route.

On the fresh temporal campaigns, no external clock observation was supplied,
so the router selected `neutral_event_frontier_unknown_or_low_regime`:

| phase | rows | GPC MAPE | duration MAPE |
|---|---:|---:|---:|
| temporal cold | 15/15 | **8.36%** | **8.08%** |
| temporal warm | 15/15 | **8.02%** | **7.90%** |

This prevents the high-clock elasticity overcorrection observed when a 1.905-GHz
response model is applied to a roughly 1.343-GHz campaign.

The external-clock reducer was refined to classify high regime when at least
10% of profiler-window samples are above 1.15x the 1,350-MHz anchor and a high
burst is present. The boost cold/warm campaigns have 17.8%/18.6% such samples;
router-v2 high routes score 8.89%/8.31% cold and 9.64%/9.69% warm GPC/duration
on large-family holdouts. The median remains idle-diluted, so the classification
is explicitly a burst-fraction hypothesis, not a synchronized clock.
