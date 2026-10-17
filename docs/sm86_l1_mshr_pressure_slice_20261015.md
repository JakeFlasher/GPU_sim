# SM86 L1 MSHR-full pressure observer — 2026-10-15

## Mechanism

H38 instruments the existing finite L1 MSHR admission path. When a miss finds
the configured MSHR set full, the cache records one checked occupancy sample
and the number of MSHR entries in use. It records no delay, does not change
retry scheduling, and keeps the pressure fields separate from global/local
traffic partitions. Full replay emits sample count, occupancy sum, and a
bounded mean occupancy projection.

This is consistent with Huerta's finite memory-subcore queue/PRT mechanism and
the retained Ampere L1 MSHR behavior, but it does not claim that the carded
`mshr_count` is a proprietary GA102 PRT size. It is an observer for residual
partitioning, not an NCU fit or timing arm.

## Real-trace evidence

Five high-MSHR/high-stride/L2 roots passed representative full-first replay
(13 launches):
`../sm86_replay_results/h38_l1_mshr_pressure_20261015_v1/full.jsonl.zst`.

Complete full-first replay then complementary fast replay passed 356 roots
(286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 launches
in 40.241 s with exactly twelve workers:

- `../sm86_replay_results/full_corpus_h38_l1_mshr_pressure_20261015_v1/full.jsonl.zst`
- `../sm86_replay_results/full_corpus_h38_l1_mshr_pressure_20261015_v1/fast.jsonl.zst`

Post-replay report:
`../sm86_replay_results/metric_report_h38_l1_mshr_pressure_20261015_v1/`.

Across the full corpus, 953,785 full-set MSHR stall samples occur in 1,120
launches, with an occupancy sum of 30,978,868 entries (maximum per-launch
sample count 124,766; maximum sum 3,992,512). The representative streaming and
DSMEM roots each record 10,924 samples at occupancy 32; vendor SGEMM records
36 samples at occupancy 32; pointer-chase and working-set roots record none.

## Exact non-regression

A keyed H37 comparison found zero differences in every old counter, source
pressure field, metric prediction, DRAM field, IPC/frontier value, and trace
field after removing only the new MSHR observer rows and the nondeterministic
`sidecar_join_wall_ns`. H38 changes no cache state, service time, DRAM demand,
or scored APE. A future MSHR timing arm requires an independent locked probe
that separates occupancy from queue service and retirement tails.
