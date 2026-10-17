# SM86 LDG constant-form observer — 2026-10-16

## Mechanism

H48 adds static modifier-form counters for `LDG.*.CONSTANT` global loads:
`LTC128B`, `U16`, and other/plain. Huerta and modern-gpu-simulator source
distinguish global LDG traffic from explicit constant-space LDC/CBANK paths, so
the forms remain global traffic and are not rerouted to L0C/GCC. The counters
are source-qualified lookup-scope observers only; no timing, cache state,
lookup correction, or NCU fit is introduced.

## Full-corpus evidence

Complete full-first replay then complementary fast replay passed 356 roots
(286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 launches
in 40.899 s with exactly twelve workers:

- `../sm86_replay_results/full_corpus_h48_constant_form_observer_20261016_v1/full.jsonl.zst`
- `../sm86_replay_results/full_corpus_h48_constant_form_observer_20261016_v1/fast.jsonl.zst`
- `../sm86_replay_results/metric_report_h48_constant_form_observer_20261016_v1/`

Totals across full launches:

| form | instructions | sectors | nonzero launches |
|---|---:|---:|---:|
| LTC128B | 199,446 | 828,751 | 141 |
| U16 | 5,515,663 | 5,612,737 | 71 |
| other/plain | 1,092,487 | 4,375,547 | 688 |

The SAM-IoU global-load outlier is entirely LTC128B; RMSNorm/qkv forms are
U16, while streaming roots are other/plain. This explains why a universal
lookup correction is unsound.

## Exact non-regression

The observer changes no L1/L2/DRAM counters, IPC, timing/frontiers, or APE.
Static form labels do not establish a physical constant-cache client; a future
correction requires a locked form-specific probe varying reuse/working set with
NCU retained only as an independent target.
