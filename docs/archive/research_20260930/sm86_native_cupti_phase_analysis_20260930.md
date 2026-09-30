# Native CUPTI phase-state analysis (2026-09-30)

The paired cold/warm native CUPTI campaigns provide a phase-local observation
of PC/scheduler states without claiming a full-cache reset. Across 15 paired
workloads, median absolute CUPTI feature deltas were small: no-issue fraction
0.00683, aggregate pressure fraction 0.00540, and long-scoreboard fraction
0.00965. The p90 long-scoreboard delta was 0.01579.

In the same paired NCU campaigns, warm/cold ratios ranged from 0.614--1.046
(median 0.854 GPC and 0.844 duration). Thus the sampled CUPTI scheduler-state
surface is nearly phase-invariant while NCU endpoints vary materially. This
falsifies the stronger hypothesis that aggregate PC stall states alone encode
the entire warm-cache endpoint shift. It does not show that CUPTI is useless:
the surface remains an independent pressure descriptor for the non-promoted
committee model.

The report is
`results/alpha/sm86_native_cupti_surface_projection_goal_20260930/phase_analysis.json`.

The launch-local follow-up removes the first eight warmup buffers from the
warm campaign before aggregation. Its median deltas are 0.00879 no-issue,
0.00827 pressure, and 0.02165 long-scoreboard; the NCU ratios remain
0.614--1.046. The launch-local report is
`results/alpha/sm86_native_cupti_surface_projection_goal_20260930/launch_phase_analysis.json`.
Its largest simple association is IMC-miss delta versus duration ratio
(Pearson 0.498), insufficient to identify a cache owner or read-from source.

Using the same launch-split surface (`--discard-buffers 8`) in the fixed
four-feature projector gives warm MAPE **23.35% GPC / 23.96% duration** over
15 rows, a small improvement over the aggregate-warm route but still far above
the promoted endpoint projector.
