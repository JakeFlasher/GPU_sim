# Temporal boost-requested cold campaign (2026-09-30)

`results/alpha/full_3090ti_20260930_temporal_boost_cold_live3/` completed all
15 workloads with two NCU repeats, two native repeats, one replay, and eight
warmups. The campaign requested `--clock-control boost` and ran a concurrent
100-ms `nvidia-smi` sampler; 1,890-MHz samples appeared during the run (the
sampler also captured idle 1,350-MHz samples).

The prior high-clock elasticity calibration was applied without using held-out
NCU labels for route selection. The elastic endpoint route scored:

* **9.88% GPC MAPE**
* **9.67% duration MAPE**

The neutral native-event/frontier route scored 15.63%/20.03% on the same
campaign. This is score-only high-regime evidence; the external sampler's
campaign-level sampling is not a launch-synchronized clock proof.

No test, hash, or verification workflow was run.

