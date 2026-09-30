# SM86 endpoint-domain committee and clock response (2026-09-30)

## Fixed policy

`tools/sm86_endpoint_domain_committee.py` combines independent endpoint
approximations without selecting a route from score NCU values:

* GPC-cycle predictions remain on a replay/frontier or constrained external
  clock-response source.
* Duration predictions use a same-regime native CUDA-event-to-NCU ratio model.

The policy follows the measurement domain, not a claimed common clock.  NCU
targets are loaded only after routing for the final live error report.

## Current-clock live evidence

For the 2026-09-30 per-kernel-clock campaigns, the family-disjoint clock
elasticity response was fitted using the externally sampled high-clock cohorts
(about 1,905 MHz cold and 2,010 MHz warm) and replay readiness exposure.  It
does not consume the score rows' NCU frequency fields.

The resulting mixed route (`clock_committee/`) covers every row:

| phase | rows | GPC MAPE | duration MAPE |
|---|---:|---:|---:|
| cold, clock response + event ratio | 15 | **7.49%** | **12.38%** |
| warm, large clock response + base frontier + event ratio | 15 | **24.39%** | **9.62%** |

The warm route keeps the baseline for the base cohort because its independent
clock-fit residual is wider; this decision is made from calibration residuals,
before reading score labels.  The route is therefore an endpoint adapter, not a
native timing closure.

The conservative warm aggregate is retained in
`results/alpha/sm86_research_followup_20260930/clock_committee/warm_conservative_summary.json`.
For comparison, the fixed frontier/event committee without clock response is
16.54%/12.38% cold and 26.61%/9.62% warm (GPC/duration).  The clock response
substantially reduces current cold-cycle drift, while warm GPC remains cache-
phase sensitive.

This direction is consistent with the microbenchmark-driven analytical-model
approach in [Jarmusch and Chandrasekaran (2026)](https://arxiv.org/abs/2605.04178)
and the separate core/memory DVFS response model of
[Wang and Chu (2017)](https://arxiv.org/abs/1701.05308): recover bounded
response surfaces from controlled probes instead of assuming an undocumented
cycle mechanism.  The local model is deliberately weaker than those papers'
hardware-validation claims because the SM86 endpoint domains are not unified.

## Limits

The external active-clock sampler is campaign-level, not launch-local; NCU
per-kernel frequency is intentionally not used as a feature.  CUDA events,
CUPTI, NCU semaphore completion, device timers, and simulator frontiers remain
separate endpoints.  Exact pipe/slot tags, unrestricted ABI generation, cache
ownership/read-from/complete DRAM semantics, and a unified timer remain open.
