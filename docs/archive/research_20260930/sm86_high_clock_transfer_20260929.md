# High-clock cohort transfer projection (2026-09-29)

## Model

The first high-clock live campaign is used as an independent calibration
cohort, and the second independently profiled high-clock campaign is scored by
holdout family.  The existing frontier projector is run with:

```text
train = full_3090ti_20260929_stage_cold_live
score = full_3090ti_20260929_stage_clock_sampled_live2
features = sparse replay frontiers only
```

Only base-family NCU labels train the large-family score (and vice versa). The
model therefore learns a high-clock endpoint efficiency surface without using
the scored campaign's NCU labels. This is a related-family holdout, not an
independent workload-generalization proof.

## Live result

All 15 score rows receive a prediction; 14 are frontier-covered and one uses
the explicit fallback. Projection time is 0.011 s, about 1,029× the measured
replay wall for the combined score rows.

| scope | GPC MAPE | duration MAPE |
|---|---:|---:|
| large holdout | **6.513%** | **6.935%** |
| base holdout | 10.191% | 10.879% |
| combined 15 rows | **8.475%** | **9.038%** |

This is substantially better than applying the frozen 1,350 MHz model directly
to the shifted high-clock campaign (large duration MAPE 23.595%). It is not a
replacement for the frozen model: it requires a calibration cohort measured in
the same clock regime and does not identify why GPC cycles change under boost.

## Boundary mapping

* Pipe/issue: replay frontiers remain aggregate; no physical route tags.
* ABI: compiler and metadata effects remain in the learned residual.
* Cache: cache phase is not inferred from clock transfer.
* Timing: same-regime transfer improves prediction, but endpoint identity and
  clock synchronization remain unproved.

Reports are under
`results/alpha/sm86_high_clock_transfer_goal_20260929/`. The replay and
promoted 1,350 MHz defaults are unchanged.

An optional static-neighbor residual blend was selected using a train-cohort
leave-one-out sweep (`alpha_gpc=0.9`, `alpha_duration=0.4`).  It improves the
same 15-row high-regime holdout to **4.353% GPC / 6.386% duration MAPE**
(~58× projection speedup), with 14 frontier-covered rows and one explicit
fallback.  The alpha selection did not use score-campaign NCU labels.  This
variant remains a non-promoted, regime-local committee member.
