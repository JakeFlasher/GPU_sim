# SM86 two-component clock elasticity model (2026-09-29)

## Motivation and research basis

The warm high-clock campaign falsified a single clock multiplier: cache-policy
and memory-wait effects changed alongside SM frequency.  The new model follows
the useful abstraction in frequency-scaling work: separate compute/core
response from memory/service response rather than forcing all time to scale
with frequency.  The source study reports separate core and memory domains,
queueing effects, and overlap between compute and memory; it also explicitly
shows that different kernels can be limited by different frequency domains
([Wang & Chu, arXiv:1701.05308](https://arxiv.org/abs/1701.05308),
[HTML model equations](https://arxiv.org/html/1701.05308v2)). Nsight Compute
also documents that cache control and clock control alter measurement state,
so the model keeps cache phase and clock regime separate
([NCU profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)).

## Model

For a low/high-clock paired calibration cohort, each workload contributes a
replay-derived semantic exposure vector:

```text
memory_wait_exposure_fraction =
  (L1/L2 dependence + async source exposure) / total readiness exposure
other_exposure_fraction = 1 - memory_wait_exposure_fraction
```

The model fits a nonnegative, [0,1]-bounded two-component response surface with
fixed ridge `0.01`, before scoring the prospective campaign.  Duration and GPC
are fit separately:

```text
duration response = r + (1-r) × memory_share
GPC response      = 1 + (1/r - 1) × memory_share
```

`r` is the low/high clock ratio.  The response-cone interval (all-core to
all-memory limiting) and a calibration residual interval are emitted.  NCU
score labels from the prospective campaign are never used to fit coefficients
or choose the response cone.

## Live result

Calibration:

* low-clock: `full_3090ti_20260929_refined_cold_live`
* high-clock: `full_3090ti_20260929_stage_clock_sampled_live2`
* base and large cohorts calibrated separately

Score: the independent high-clock campaign, all 15 rows.

| endpoint | frozen reference MAPE | full-core scaling | elasticity MAPE |
|---|---:|---:|---:|
| GPC cycles | 14.927% | 14.927% | **6.637%** |
| duration | 20.193% | 14.824% | **7.733%** |

All 15 rows receive candidates. The fit/adapter executes in under 0.05 s per
cohort. Reports are in
`results/alpha/sm86_clock_elasticity_goal_20260929/coverage_summary.json`.

The separate warm/cache-none high-clock campaign was intentionally not folded
into this cold calibration. A same-phase warm frontier transfer was negative
(19.50%/19.83% large and 38.68%/38.76% reverse base), demonstrating that cache
history is an additional latent state rather than a clock-elasticity term.

## Limits

This is a regime-transfer predictor, not a timing proof.  The exposure classes
are simulator readiness abstractions, not physical pipe or cache-owner tags;
GPC and duration remain distinct; cache phase is not inferred from frequency;
and the high-clock calibration is related-family evidence.  The frozen replay
and promoted default model remain unchanged.
