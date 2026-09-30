# High-clock warm campaign (2026-09-29)

`results/alpha/full_3090ti_20260929_stage_clock_sampled_warm_live/` completed
all 15 workloads with two NCU repeats, `--cache-control none`, and an external
100-ms clock sampler.

| observation | value |
|---|---:|
| all clock samples | 1,763 |
| active-clock median | 2,010 MHz |
| large duration median | 6,720 ns |
| large GPC median | 13,090.5 cycles |

The high-clock warm lane is not a simple cold-to-warm rescaling: the existing
1350-MHz warm/cold ratios do not transfer to this regime.  A same-phase,
related-family frontier fit using high-clock warm base rows reaches 19.50% GPC
and 19.83% duration MAPE on high-clock warm large rows; the reverse base
holdout is 38.68%/38.76%.  It is retained as a negative result, not promoted.

The paired two-component elasticity model is also negative for warm GPC:
large holdout 25.15%/25.38% and reverse base 28.10%/28.45% GPC/duration.
Duration improves over the raw frozen projection but remains worse than simple
full-core normalization in this cache-none regime.

The two-component clock-elasticity model is intentionally fit and scored on
the high-clock **cold** pair only.  Warm cache history remains a separate
latent state; no mechanism closure is claimed.

No tests, hashing, or verification workflow was run.
