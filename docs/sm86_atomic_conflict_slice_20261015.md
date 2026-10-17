# SM86 atomic/reduction conflict exposure — 2026-10-15

## Scope and evidence

H12 adds exact trace-sector conflict exposure for global atomic operations:
repeated same-sector aliases, conflict-bearing instruction count, and distinct
atomic instructions. It does not add per-alias latency or DRAM-sector
multipliers. Existing sector transaction vectors remain intact and no cache,
DRAM, or issue timing state changes.

Independent SM86 notes contain conflict-vs-distinct atomic ordering anchors,
but they are kernel-level durations rather than a safe per-transaction cycle
constant. The adversarial decision is therefore observer-only; timing-active
serialization remains a future locked microbenchmark arm.

## Representative full replay

Four real atomic roots completed full replay 4/4. The return-used root emitted
8 conflict-bearing instructions, 1 distinct instruction, and 24 same-sector
aliases; other selected histogram roots exposed no repeated-sector aliases
under their captured dynamic spans.

## Complete-corpus evidence

| Field | Result |
|---|---:|
| Roots discovered | 356 |
| `ok` roots | 286 |
| Mixed-code controls | 62 |
| Replay/input failures | 8 |
| Successful launches | 12,392 |
| Driver wall time | 35.578 s |
| Workers | 12 |
| NCU used as replay input | No |

Artifact: `../sm86_replay_results/full_corpus_h12_atomic_conflict_20261015_v1/full.jsonl.zst`.

Across successful launches H12 observed 58 conflict-bearing atomic
instructions, 622 distinct atomic instructions, and 198 alias sectors.

## Residuals and next arm

Exact DRAM/L1/L2/IPC residuals are unchanged from H10/H11 (DRAM 40.91%, active
IPC 18.19%, elapsed IPC 12.42% in the current effective-rate path). Report:
`../sm86_replay_results/metric_report_h12_atomic_conflict_20261015_v1/`.

H12 is conflict observability, not timing closure. A future timing arm requires
locked local atomic conflict microbenchmarks; do not infer a multiplier from
NCU targets or kernel-level duration anchors.
