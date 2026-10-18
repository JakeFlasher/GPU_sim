# SM86 six-slice L2 routing observer — 2026-10-15

## Mechanism

H30 adds an observer-only six-slice request distribution to the effective SM86
L2 model. It uses the active card's six-slice geometry and selected effective
slice hash, but does not claim the proprietary GA102 crossbar hash, add queue
timing, or use NCU labels to select routing. Counts are sector requests by
slice and are transactionally carried through immediate/event L2 snapshots and
launch deltas.

## Real-trace evidence

Five high-stride/L2-thrash/working-set/SGEMM roots passed full-first replay:
`../sm86_replay_results/h30_l2_slice_observer_20261015_v1/full.jsonl.zst`.

The complete full-first replay then passed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 40.089 s with exactly 12
workers:

`../sm86_replay_results/full_corpus_h30_l2_slice_observer_20261015_v1/full.jsonl.zst`

Post-replay report:
`../sm86_replay_results/metric_report_h30_l2_slice_observer_20261015_v1/`.

## Slice distribution and exact non-regression

Across the full corpus, effective slice sectors are:

| Slice | Sectors |
|---:|---:|
| 0 | 1,718,579 |
| 1 | 1,653,401 |
| 2 | 1,717,493 |
| 3 | 1,652,955 |
| 4 | 1,717,655 |
| 5 | 1,653,918 |

The distribution is balanced under the active effective hash. H30 is bit-for-bit
neutral versus H29 for L1/L2/DRAM counters, IPC/frontiers, and APE. This is a
useful routing/queue research surface, not physical hash recovery or timing
closure; the next L2 arm requires independent slice-queue/service evidence.
