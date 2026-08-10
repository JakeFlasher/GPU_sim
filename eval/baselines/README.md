# eval/baselines — contextual in-repo references

Current accuracy authority is silicon NCU, and current runtime evidence covers
only this repository's `sim-run`. External-simulator rigs are inactive preserved
placeholders: they must not be produced, consumed, promoted, or used as a gate
without explicit user reauthorization. The reporting methodology below remains
contextual: NRMSE + correlation per metric class, MAE only for never-zero metrics,
p50/p90/worst-case per kernel family, leave-kernel-family-out + leave-SKU-out
splits, and a sealed holdout touched once.

| Rig | Baseline | Notes |
|---|---|---|
| `accel_sim/` | Accel-Sim SM86_RTX3070 | **Inactive preserved placeholder; no current baseline, speed, accuracy, dependency, or gate authority** |
| `upc_micro25/` | UPC MICRO'25 modern-core artifact | Inactive literature reference; no current execution or gate |
| `ppt_gpu/` | PPT-GPU | Inactive literature reference with known memory-model bugs |
| `nugteren_rdh/` | in-repo `tools/baseline_runners/nugteren2014_rdh_baseline.py` | reuse-distance cache baseline |
| `roofline_solar/` | `tools/solar/` analytical SOL floor | SOL%-crosswalk baseline |
| `frozen_v5_heads/` | `data/local_labels/sm_86_v5_head.json` (6.4% L2 geomean) | the shipping learned-head baseline (insurance headline per D1) |
