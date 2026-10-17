# SM86 IPC readiness concentration observer — H83 — 2026-10-16

## Implementation

Full replay now emits the maximum fraction (ppm) of exclusive readiness
limiting cycles attributable to one causal bucket, alongside the existing
issue/SM-active frontier interval and total limiting-cycle observer. The
calculation is saturating and report-only: overlapping exposure buckets remain
separate, and no readiness cause selects or alters the issue scheduler,
frontier, cache, DRAM, or IPC denominator.

## Full-first evidence

* replay: `../sm86_replay_results/full_corpus_h83_ipc_readiness_20261016_v1/`
* report: `../sm86_replay_results/metric_report_h83_ipc_readiness_20261016_v1/`
* coverage: 286 `ok`, 62 controls, 8 replay errors; 12,392 full launches with
  twelve workers.

The observer is present on 286 full launches: median dominant-cause fraction
425,114 ppm, P90 613,776 ppm, mean 442,454 ppm, with 103 launches above
500,000 ppm. Median total exclusive limiting cycles are 48,377.5 (P90
1,087,424). GPC-cycle and SM-active APE remain bit-equivalent to H82
(12.9008% and 18.0202%).

## Decision

The concentration is a useful causal decomposition, not evidence that a single
readiness bucket should be scaled into hardware IPC. The existing H71 EXIT and
H64 queue timing experiments provide negative promotion precedents; any timing
change still requires held-out endpoint-matched issue/readiness probes.
