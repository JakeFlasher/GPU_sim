# NCU cold/warm phase-clock analysis (2026-09-30)

The paired fresh campaigns
`full_3090ti_20260930_clock_context_cold_live` and
`full_3090ti_20260930_clock_context_warm_live` collect per-kernel NCU GPC
frequency, duration, and GPC-cycle metrics under identical `--clock-control
none`/cache-policy settings except for the intended cache phase.

Across all 15 paired workloads:

| measure | result |
|---|---:|
| warm/cold NCU frequency ratio median | 0.9980 |
| frequency ratio range | 0.9589–1.0219 |
| warm/cold duration ratio range | 0.6148–1.0433 |
| warm/cold GPC-cycle ratio range | 0.6142–1.0458 |
| median absolute duration-vs-GPC/frequency residual | 0.0274 percentage points |
| maximum absolute residual | 0.1020 percentage points |

The last two rows use only NCU observations and establish an **NCU-local**
relation: warm/cold duration is explained by warm/cold GPC cycles divided by
the corresponding local frequency. They do not establish a common simulator,
CUDA-event, CUPTI, or host clock. The large spread in warm/cold ratios remains
workload-specific cache-phase behavior.

The observation report is
`results/alpha/sm86_ncu_phase_clock_analysis_goal_20260930/cold_warm.json`.
NCU labels remain score-only and are not simulator features or tuning inputs.
