# SM86 per-request L2 frontier observer — H70 — 2026-10-16

## Implementation

The shared-L2 system now records bounded, timing-neutral per-request frontier
observers from the existing event path: issue-to-fill sum/max, issue-to-response
sum/max, and model service-delay sum/max, with sample count. These are observer
intervals over simulator event state; they are not hardware completion tokens,
queue capacities, or timing inputs. Existing L2 state, DRAM service, response
ordering, and all prior metrics remain unchanged.

## Full-first real-corpus evidence

Release `sim-trace`/`sim-run` builds passed. Complete replay ran full before fast
with exactly 12 workers and compressed artifacts:

`../sm86_replay_results/full_corpus_h70_l2_frontier_observer_20261016_v1/`

Coverage was 356 roots (286 `ok`, 62 controls, 8 failures) and 12,392 full
launches. The post-replay report is:
`../sm86_replay_results/metric_report_h70_l2_frontier_observer_20261016_v1/`.

A representative cold-read launch emits 32 frontier samples, 13,536 summed
issue-to-fill/response cycles (maximum 423), and a one-cycle aggregate service
interval. The new rows have no NCU aliases and are not scored or used for
selection.

## Decision

The observer is promoted into default full replay because it adds causal
visibility without changing behavior. Any future timing law must correlate
these per-request frontiers with a fixed-footprint dependent-vs-ILP probe and a
held-out NVBit trace before changing `DramBw::Carded` or L2 response timing.
