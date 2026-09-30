# SM86 clock-regime projection (2026-09-29)

## Motivation

The clock-sampled live campaign showed that the same replay workload can be
profiled at a substantially different active SM clock than the frozen 1,350
MHz anchor.  NCU's static device attribute is insufficient: the external
100-ms sampler recorded a 1,905 MHz median while the GPU was active.

The opt-in `tools/sm86_clock_regime_projection.py` keeps endpoint domains
separate and applies only the dimensional clock conversion that is justified
by the observation:

```text
duration_adjusted = duration_anchor × anchor_clock / active_clock
GPC_adjusted      = GPC_anchor       (unchanged)
```

The earlier external sampler reported 1,890--2,055 MHz, but it is not
launch-local. A dedicated interleaved NCU probe now reports per-kernel SM/GPC
medians of 1.546875 GHz under `base` and 1.867973 GHz under `boost`; see
`docs/sm86_profile_clock_context_20260930.md`. The adapter therefore retains
external samples as broad regime context and does not treat them as exact
launch clocks.

The adapter never asserts that NCU GPC cycles, CUDA events, CUPTI timestamps,
and simulator cycles share a clock.  It uses the external clock observation as
a runtime condition and reads NCU values only after prediction for comparison.
Per-launch clock variation remains an interval/unknown.

## Live score-only result

On `results/alpha/full_3090ti_20260929_stage_clock_sampled_live2/`, the
promoted frozen projection was compared to the independent NCU rows:

| endpoint | raw MAPE | clock-adjusted MAPE |
|---|---:|---:|
| duration | 23.595% | **12.413%** |
| GPC cycles | 13.073% | 13.073% (unchanged) |

Projection time was 0.0003 s.  The improvement is a timing-regime
normalization, not a claim that the four mechanisms are observable.  The
adjustment is retained as an opt-in committee lane and does not change the
promoted default.

## Boundary mapping

* Physical pipe/issue: untouched; GPC is not rescaled.
* CUDA ABI: untouched; ABI effects remain in the endpoint residual.
* Cache ownership/read-from: untouched; the clock factor does not identify
  cache state.
* Timing: partially mitigated for duration when an external active-clock
  observation exists; synchronized endpoint identity remains unresolved.

The source campaign and adjusted report are retained at
`results/alpha/full_3090ti_20260929_stage_clock_sampled_live2/`.
