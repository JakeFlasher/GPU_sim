# Temporal cold/warm phase campaigns (2026-09-30)

Two fresh 15-row campaigns were collected with the same bounded alpha path:

* cold: `results/alpha/full_3090ti_20260930_temporal_cold_live2/`
* warm/cache-none: `results/alpha/full_3090ti_20260930_temporal_warm_live2/`

Each used two NCU repeats, two native repeats, one replay, and eight warmups.
Both campaigns completed 15/15. Per-kernel NCU-local SM/GPC frequency was
stable near 1.343 GHz in both phases (cold median 1.343952 GHz, warm median
1.343429 GHz), while paired warm/cold ratios were:

| endpoint | median | range |
|---|---:|---:|
| GPC cycles | **0.833** | 0.717--0.972 |
| duration | **0.834** | 0.718--0.972 |

The prior refined campaign trained a neutral native-event/frontier crosswalk;
it scored the fresh temporal holdouts at:

| phase | GPC MAPE | duration MAPE |
|---|---:|---:|
| fresh cold | **8.36%** | **8.08%** |
| fresh warm | **8.02%** | **7.90%** |

This is strong endpoint evidence without a clock-regime confound, but it does
not identify cache-line ownership, read-from source, complete writeback, or a
unified NCU/CUDA-event/simulator endpoint.

Projection runtime was approximately 5--8 ms per family partition versus
11.3--11.5 s replay wall, or roughly **1,500--2,050x** projection-only speedup.

No test, hash, or verification workflow was run.
