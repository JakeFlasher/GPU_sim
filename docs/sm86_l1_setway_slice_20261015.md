# SM86 effective L1 set/way state — 2026-10-15

## Mechanism and boundary

H17 adds a card-backed effective set-local LRU arm to the modeled L1 state:
256 modulo-indexed sets and a four-line per-set way bound from
`arch_profiles/cards/sm_86.json`. The p-chase active-capacity point (98,304 B)
remains the global capacity scenario; these two card sources are intentionally
recorded as separate effective assumptions. The proprietary GA102 hash is not
recovered: modulo indexing is a sensitivity scenario, not a physical claim.

Pending line/sector fills, generations, MSHR accounting, dirty safeguards, and
observer counters remain transactional. Fully associative state is still
available as the residual/sensitivity arm; H17 changes only the SM86 full-mode
carded path.

## Representative full-first replay

Five real roots completed full replay 5/5 and fast replay 5/5:
`simbench_l1_working_set_capacity_sweep`, `simbench_l1_l2_thrash_pointer_chase`,
`simbench_l1_global_vectorized_128b_copy`, `simbench_l1_vendor_sgemm_cublas`,
and `simbench_l1_vendor_sgemm_strided_batched_cublas`.
Artifact: `../sm86_replay_results/h17_l1_setway_20261015_v1/full.jsonl.zst`.
Report: `../sm86_replay_results/metric_report_h17_l1_setway_20261015_v1/`.

## Complete-corpus evidence

Full-first replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8
replay/input failures), 12,392 launches in 36.517 s with exactly 12 workers.
Compressed artifacts:
`../sm86_replay_results/full_corpus_h17_l1_setway_20261015_v1/full.jsonl.zst`
and `.../fast.jsonl.zst`.

## Exact residuals versus H16

| Metric | H16 mean APE | H17 mean APE | Delta |
|---|---:|---:|---:|
| L1 global-load lookup hits | 30.71% | 30.39% | −0.32 pp |
| L1 global-load lookup misses | 58.44% | 57.74% | −0.70 pp |
| L2 TEX source sectors | 28.24% | 32.43% | +4.19 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| GPC elapsed cycles | 12.91% | 12.75% | −0.16 pp |
| SM active cycles | 18.01% | 17.86% | −0.14 pp |
| SMSP active cycles | 17.50% | 17.36% | −0.14 pp |
| elapsed IPC | 12.31% | 12.12% | −0.19 pp |

The set/way arm is a modest broad IPC/L1 improvement, with a measurable L2
traffic regression. It is promoted as the default effective SM86 topology while
retaining the fully-associative arm for sensitivity and the unknown-hash
residual. It is not sub-10% closure; future work must address hash uncertainty,
L2 source packetization, and the source-qualified store/fence completion tail.
