# SM86 scope-aware NCU residual reporting — 2026-10-15

## Mechanism

H42 makes H40/H41 scope diagnostics operational in the post-replay reducer.
When a fixed-ratio trace-vs-NCU scope warning explicitly covers a metric, that
prediction remains in compressed machine-readable rows with status
`scope_mismatch`, but is excluded from raw/all-row comparison APE. Warning
counts and excluded rows are reported separately as a secondary like-for-like
view; no replay input, prediction selector, timing model, or NCU-derived
parameter changes.

## Evidence

H42 reduces the H41 complete full-first artifacts:

- Replay: `../sm86_replay_results/full_corpus_h41_l2_dram_scope_audit_20261015_v1/full.jsonl.zst`
- Fast: `../sm86_replay_results/full_corpus_h41_l2_dram_scope_audit_20261015_v1/fast.jsonl.zst`
- Scope-aware report: `../sm86_replay_results/metric_report_h42_scope_aware_20261015_v1/`

The retained H42 directory predates the H63 report-label correction and is
historical evidence. For machine-readable scope/raw view labels use the
regenerated H63 report:
`../sm86_replay_results/metric_report_h63_scope_label_audit_20261016_v1/`.

The corpus has 125 scope warnings and 67 explicitly excluded predictions in
the full mode. On 107 exact joins, the secondary scope-compatible mean APE
view is:

| metric | raw/all-row APE (authoritative) | secondary scope-compatible APE |
|---|---:|---:|
| L1 global-load sectors | 100.96% | 0.21% |
| L2 source sectors | 28.24% | 6.76% |
| DRAM read sectors | 40.91% | 26.84% |
| GPC elapsed cycles | 12.90% | 12.90% |

The large L1 reduction is driven by one 64× opcode-scope mismatch; L2 has
three scope warnings, while DRAM has 121. The independent NCU values remain
visible and are never used to alter simulator execution.

## Decision

Scope-aware APE is a **secondary like-for-like view**, with scope-mismatch
inventory shown separately. The all-row/raw H41 means remain the authoritative
overall residuals (L1 100.96%, L2 28.24%, DRAM 40.91%); excluded rows are
unresolved scope/model gaps, not completed mechanisms. The 26.84%
scope-compatible DRAM residual is still an active modeling target, and no
sub-10% overall closure claim is made.
