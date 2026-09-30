# High-clock temporal cold/warm campaigns (2026-09-30)

The boost-requested pair completed 15/15 in both phases:

* cold: `full_3090ti_20260930_temporal_boost_cold_live3`
* warm/cache-none: `full_3090ti_20260930_temporal_boost_warm_live2`

Both used two NCU repeats, two native repeats, one replay, eight warmups, and a
concurrent 100-ms clock sampler. Per-kernel NCU-local frequencies were about
1.872 GHz cold and 1.870 GHz warm; the external sampler observed 1,890-MHz
bursts amid idle 1,350-MHz samples.

The prior high-clock elasticity route scored:

| phase | GPC MAPE | duration MAPE |
|---|---:|---:|
| cold | **9.88%** | **9.67%** |
| warm | **9.36%** | **9.76%** |

Neutral event/frontier scores were 15.63%/20.03% cold and 16.08%/19.06%
warm. This supports a regime-local clock/cache committee when an independent
clock witness is present; it does not establish a unified clock or cache
ownership/read-from semantics.

