# eval/baselines — comparison rigs (Phase 3B)

Stubs created in Phase 0; each becomes a runnable rig during Phase 3. Reporting
follows the Accel-Sim template + SIGMETRICS'18 critique: NRMSE + correlation per
metric class, MAE only for never-zero metrics, p50/p90/worst-case per kernel
family, leave-kernel-family-out + leave-SKU-out splits, sealed holdout touched
once (plan §3B; audit plan §6.2).

| Rig | Baseline | Notes |
|---|---|---|
| `accel_sim/` | Accel-Sim SM86_RTX3070 (the only upstream config overlapping our silicon) | shared NVBit traces; speed + per-metric accuracy vs our sm_86 labels |
| `upc_micro25/` | UPC MICRO'25 modern-core artifact (cycles) | ships Blackwell example traces; cycles-MAPE reference 13.45% Ampere / 17.41% Blackwell |
| `ppt_gpu/` | PPT-GPU | dormant + known memory-model bugs — document, don't just assert |
| `nugteren_rdh/` | in-repo `tools/baseline_runners/nugteren2014_rdh_baseline.py` | reuse-distance cache baseline |
| `roofline_solar/` | `tools/solar/` analytical SOL floor | SOL%-crosswalk baseline |
| `frozen_v5_heads/` | `data/local_labels/sm_86_v5_head.json` (6.4% L2 geomean) | the shipping learned-head baseline (insurance headline per D1) |
