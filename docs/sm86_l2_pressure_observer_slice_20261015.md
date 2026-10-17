# SM86 L2 slice-pressure observer — 2026-10-15

## Mechanism

H31 extends H30's six-slice request counts with bounded per-launch pressure
observers: effective max-slice demand, minimum nonzero demand, spread, and
max/min imbalance ratio. These values are derived from the active effective
slice routing only; they do not recover a GA102 hash, model hidden queues, or
feed timing/NCU selection.

## Probe and real-trace evidence

The retained RTX 3090 Ti clock64 `l2_queue_frontier` probe was rebuilt and run
across unique/reuse modes, 1/2/4/8 resident warps, and 1/8/32-load bursts.
All smoke points passed checksums/timestamp bounds. Unique 32-load bursts rose
from roughly 8.9k cycles (1–4 warps) to 12.5k maximum cycles at 8 warps;
reuse bursts showed the same pressure growth. The differential is stable as an
observer signal but does not isolate a transferable fixed queue latency.

Five high-stride/L2-thrash roots passed representative full-first replay. The
complete full-first replay passed 356 roots (286 `ok`, 62 mixed-code controls,
8 replay/input failures), 12,392 launches in 39.887 s with exactly 12 workers:

`../sm86_replay_results/full_corpus_h31_l2_pressure_observer_20261015_v1/full.jsonl.zst`

Post-replay report:
`../sm86_replay_results/metric_report_h31_l2_pressure_observer_20261015_v1/`.

## Corpus distribution and exact comparison

The per-launch imbalance ratio has minimum 1.0, median 1.133, p90 3.0, and
maximum 38.0. Aggregate slice sectors remain:
`1,718,579 / 1,653,401 / 1,717,493 / 1,652,955 / 1,717,655 / 1,653,918`.
H31 is bit-for-bit neutral versus H30 for L1/L2/DRAM counters, IPC/frontiers,
and APE. No queue timing arm is promoted; fresh queue evidence remains an
observer/partitioning result only.
