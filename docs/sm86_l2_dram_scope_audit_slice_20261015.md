# SM86 L2/DRAM scope audit — 2026-10-15

## Mechanism

H41 extends H40's fixed-threshold, post-replay scope diagnostics to L2
source-sector and DRAM read-sector domains. It compares simulator counters with
independent NCU operation metrics, retains machine-readable warnings, and never
changes replay, prediction selection, APE scoring, or NCU model inputs.

## Full-corpus evidence

Complete full-first replay passed 356 roots (286 `ok`, 62 mixed-code controls,
8 replay/input failures), 12,392 launches in 39.934 s with exactly twelve
workers:

- `../sm86_replay_results/full_corpus_h41_l2_dram_scope_audit_20261015_v1/full.jsonl.zst`
- `../sm86_replay_results/full_corpus_h41_l2_dram_scope_audit_20261015_v1/fast.jsonl.zst`
- `../sm86_replay_results/metric_report_h41_l2_dram_scope_audit_20261015_v1/`

Among full exact-identity joins, the audit emits 121 DRAM read-sector warnings,
3 L2 source-sector warnings, and the one H40 global-load opcode warning. The
warnings are gross fixed-ratio scope diagnostics, not model corrections. For
example, the SAM-IoU join has 2,464 modeled L2 source sectors versus 168 NCU
sectors and 1,076 modeled DRAM reads versus 564 NCU sectors; many streaming
roots show much larger DRAM scope ratios.

## Decision

No L2/DRAM multiplier or timing change is promoted. The simulator's typed
read-demand domain and independent NCU operation domain remain visible with
explicit scope warnings. A keyed H40 comparison found zero old replay-field
differences after excluding only the post-replay warning and
`sidecar_join_wall_ns`; old APE values remain unchanged. Future accuracy work
requires scope-matched NCU metric sets or independently justified operation
projections, not global balancing.
