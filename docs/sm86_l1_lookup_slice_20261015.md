# SM86 L1 tag-lookup observer — 2026-10-15

## Mechanism

The prior primary L1 lookup rows used only resident-data hits, even though the
sectored L1 already retained line-allocation and pending-merge provenance. H16
corrects the observer boundary without changing cache state, MSHR admission,
replacement, fill timing, L2 traffic, DRAM service, or warp readiness:

```
tag_lookup_hits = min(total_lookup_sectors,
                       resident_hits + merge_observations
                       - fresh_line_allocation_misses)
```

The subtraction is one tag miss per fresh line allocation, not three sector
misses. The result is explicitly a bounded profiler-like tag observation; the
modeled sector hit/miss split remains the timing state. Async-copy packetization
and hidden L1 hashing remain residuals for the next conditional arms.

## Representative full-first replay

Four real roots completed full replay 4/4 and complementary fast replay 4/4:
`fp32_fma_dep_chain`, `simbench_l1_global_vectorized_128b_copy`,
`simbench_l1_working_set_capacity_sweep`, and `cp_async_shared_tile_copy`.
Artifact:
`../sm86_replay_results/h16_l1_lookup_20261015_v2/full.jsonl.zst`.
The representative full report is
`../sm86_replay_results/metric_report_h16_l1_lookup_20261015_v1/`.

## Complete-corpus evidence

Full-first replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8
replay/input failures), 12,392 launches in 36.054 s with exactly 12 workers.
Both outputs are compressed:
`../sm86_replay_results/full_corpus_h16_l1_lookup_20261015_v1/full.jsonl.zst`
and `.../fast.jsonl.zst`.

## Exact residuals

| Metric | H15 mean APE | H16 mean APE | Delta |
|---|---:|---:|---:|
| L1 global-load lookup hits | 33.43% | 30.71% | −2.72 pp |
| L1 global-load lookup misses | 206.41% | 58.44% | −147.97 pp |
| L2 TEX source sectors | 28.24% | 28.24% | 0.00 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| SM active cycles | 18.01% | 18.01% | 0.00 pp |
| elapsed IPC | 12.31% | 12.31% | 0.00 pp |

The observer materially removes the previous L1 miss-counting defect but does
not close the sub-10% target. Remaining error is consistent with unresolved
SM86 set/way/hash behavior and async/same-request packetization, so H16 is a
promoted observer correction, not physical L1 closure. The next L1 arm is a
card-backed 256-set × 4-way effective state model with the hash retained as a
typed unknown/sensitivity interval.
