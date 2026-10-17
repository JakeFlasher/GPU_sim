# SM86 causal IPC frontier decomposition — 2026-10-15

## Mechanism

H21 adds explicit per-launch causal IPC-frontier rows to the native metric
surface: issue lower frontier, terminal lower frontier, and their width. These
are source-derived simulator boundaries, kept separate from NCU endpoint
predictions and from the typed writeback/coherence tails. This slice does not
silently select an endpoint or fit a residual; it makes the critical-path
uncertainty directly inspectable for the next timing arm.

## Full-first evidence

Five real critical-path roots completed full replay 5/5:
`simbench_l1_local_dynamic_index_array`, `simbench_l1_register_spill_local_mem`,
`simbench_l2_geglu_ffn_fused_bf16`, `simbench_l1_vendor_sgemm_cublas`, and
`lcd_flash_attn_bwd_bf16_sm80`. Artifact:
`../sm86_replay_results/h21_critical_frontier_20261015_v1/full.jsonl.zst`.

Complete default full-first replay completed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 38.560 s with 12 workers:
`../sm86_replay_results/full_corpus_h21_critical_frontier_20261015_v1/full.jsonl.zst`.
Report: `../sm86_replay_results/metric_report_h21_critical_frontier_20261015_v1/`.

## Exact residuals versus H20

| Metric | H20 mean APE | H21 mean APE | Delta |
|---|---:|---:|---:|
| L1 global-load lookup hits | 30.71% | 30.71% | 0.00 pp |
| L1 global-load lookup misses | 58.44% | 58.44% | 0.00 pp |
| L2 TEX source sectors | 28.24% | 28.24% | 0.00 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| SM active cycles | 18.01% | 18.01% | 0.00 pp |
| elapsed IPC | 12.31% | 12.31% | 0.00 pp |

This is a neutral but necessary critical-path observability slice: it narrows
future IPC work to explicit frontier/cause relations without relabeling an
unfinished timing model as accurate.
