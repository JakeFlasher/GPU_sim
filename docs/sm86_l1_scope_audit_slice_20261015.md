# SM86 L1 global-load scope audit — 2026-10-15

## Mechanism

H40 adds a post-replay scope diagnostic to the metric reducer. It compares
trace-side logical memory-instruction counts with independent NCU opcode
metrics and flags only gross disagreements (a fixed 4× ratio threshold). The
warning is machine-readable in compressed metric-comparison rows and does not
change replay, prediction selection, APE scoring, or NCU use as a model input.

## Full-corpus evidence

The complete full-first replay passed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 40.199 s with exactly
twelve workers:

- `../sm86_replay_results/full_corpus_h40_l1_scope_audit_20261015_v1/full.jsonl.zst`
- `../sm86_replay_results/full_corpus_h40_l1_scope_audit_20261015_v1/fast.jsonl.zst`
- `../sm86_replay_results/metric_report_h40_l1_scope_audit_20261015_v2/`

Exactly one full exact-identity join is flagged:
`sm86_adapted_l2_074_sam_iou_backward_sm120`. The trace reports 1,024 logical
global-load instructions and 4,352 replay sectors, while the independent NCU
opcode row reports 16 global-load instructions and 40 sectors (64× scope
ratio). The same row has exact warp-instruction agreement (32,736), so the
warning identifies metric-scope disagreement rather than trace truncation.

## Decision

No L1 sector/lookup correction is promoted. The simulator continues to report
its trace-derived sector domain, while the audit preserves the independent NCU
target and warns that this join is not a valid like-for-like opcode scope.
All old replay fields, timing/frontiers, cache/DRAM counters, and APE values are
exactly unchanged versus H38. Future corrections require a scope-matched NCU
metric set, not a corpus-wide multiplier.
