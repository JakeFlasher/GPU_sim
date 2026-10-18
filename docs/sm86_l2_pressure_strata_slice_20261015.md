# SM86 L2 pressure strata — 2026-10-15

H32 extends the H31 L2 observer with route-local pressure strata: active slice
count and maximum-slice demand fraction, alongside the existing max/min,
spread, and imbalance ratio. These are effective routing observers only; no
hidden GA102 hash, queue depth, or NCU-fitted latency is introduced.

The complete full-first replay passed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 40.152 s with 12
workers:

`../sm86_replay_results/full_corpus_h32_l2_pressure_strata_20261015_v1/full.jsonl.zst`

Report:
`../sm86_replay_results/metric_report_h32_l2_pressure_strata_20261015_v1/`.

Pressure distribution is unchanged from H31: imbalance ratio min 1.0, median
1.133, p90 3.0, max 38.0; max-slice fraction min 16.67%, median 18.75%, p90
50%, max 100%. Active-slice counts are 6 for 9,186 launches, 2 for 1,658,
3 for 488, 4 for 487, 1 for 358, and 5 for 215.

H32 is bit-for-bit neutral versus H31 for L1/L2/DRAM counters, IPC/frontiers,
and APE. The clock64 queue probe still supports pressure observation but not a
stable transferable service latency; timing remains unchanged.
