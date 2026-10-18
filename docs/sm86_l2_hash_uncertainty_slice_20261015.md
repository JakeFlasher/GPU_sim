# SM86 L2 regression isolation / L1 hash uncertainty — 2026-10-15

## Finding

H17's fixed modulo 256-set × 4-way L1 arm created a reproducible high-stride
alias: real streaming roots went from 32,896 L2 lookup sectors to 82,048,
while the independent DRAM read stream stayed at 32,768 sectors. This is a
source-topology/hash residual, not a DRAM-rate problem and not evidence for
changing L2 service timing.

H18 adds a deterministic pressure-bounded unknown-hash scenario. It retains the
card modulo candidate until its set is full, then uses an XOR-fold candidate
only when that candidate has strictly more room. The decision uses only traced
line-domain occupancy and the card's set/way geometry; it never reads NCU
values or selects a hash from scored targets. Fixed modulo and fixed XOR remain
available as sensitivity interpretations, while the proprietary GA102 hash is
still unknown.

## Representative full-first evidence

Five real roots completed full replay 5/5:
`simbench_l1_dram_streaming_dot_product`,
`sm86_sub_simbench_cluster_remote_reduction`,
`sm86_sub_simbench_dsmem_ring_bandwidth`, `fp32_fma_dep_chain`, and
`simbench_l1_vendor_sgemm_cublas`.
Artifact: `../sm86_replay_results/h18_l1_hashfallback_20261015_v1/full.jsonl.zst`.
The streaming roots returned to 32,896 L2 lookup sectors (from H17's 82,048)
with 32,768 DRAM read sectors unchanged.

The final fallback code also completed the named topology holdout batch
(`simbench_l1_working_set_capacity_sweep`, `simbench_l1_l2_thrash_pointer_chase`,
`simbench_l1_global_vectorized_128b_copy`, and both vendor SGEMM roots) full
5/5. Artifact:
`../sm86_replay_results/h18_l1_hashfallback_20261015_v2_representative/full.jsonl.zst`.

## Complete-corpus evidence

Full-first replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8
replay/input failures), 12,392 launches in 36.751 s with exactly 12 workers:
`../sm86_replay_results/full_corpus_h18_l1_hashfallback_20261015_v2/full.jsonl.zst`.
Fast output is retained as the complementary stage at `.../fast.jsonl.zst`.
Report: `../sm86_replay_results/metric_report_h18_l1_hashfallback_20261015_v2/`.

## Exact residuals

| Metric | H17 modulo | H18 pressure-bounded hash | Delta |
|---|---:|---:|---:|
| L1 global-load lookup hits | 30.39% | 30.71% | +0.32 pp |
| L1 global-load lookup misses | 57.74% | 58.44% | +0.70 pp |
| L2 TEX source sectors | 32.43% | 28.24% | −4.19 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| GPC elapsed cycles | 12.75% | 12.91% | +0.16 pp |
| SM active cycles | 17.86% | 18.01% | +0.15 pp |
| elapsed IPC | 12.12% | 12.31% | +0.19 pp |

H18 is promoted as the default effective unknown-hash scenario because it
removes a demonstrable modulo alias regression without NCU fitting. It retains
the H17 set/way state and exposes the H17 modulo result as a sensitivity arm;
sub-10% L1/L2/IPC closure remains open.
