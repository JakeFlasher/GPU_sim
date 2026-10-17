# SM86 control-flow/L1TEX observer — 2026-10-15

## Scope

H27 keeps the H26 branch-stream flush timing arm rejected, but adds a native
observer for dynamic non-fall-through control transfers at the instruction
front end. It records where branch-target path state intersects the large
L1TEX/front-end dependency residual without mutating stream buffers, cache
state, DRAM, or issue timing. Readiness schema 30 carries the field.

## Complete-corpus evidence

Five branch-heavy real roots completed representative full-first replay. The
complete full-first replay then completed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 40.452 s with 12
workers:

`../sm86_replay_results/full_corpus_h27_l1tex_observer_20261015_v1/full.jsonl.zst`

Post-replay report:
`../sm86_replay_results/metric_report_h27_l1tex_observer_20261015_v1/`.

The corpus contains 1,882,174 dynamic non-fall-through control transfers and
85,461,673 front-end fetches. H27 is bit-for-bit neutral versus H25 for all
L1/L2/DRAM counters, IPC/frontier values, and APE; no timing arm was enabled.

## Verdict and next seam

The observer confirms broad branch-target activity but does not identify a
safe universal stream-flush timing rule—the H26 A/B flush worsened GPC/SM/IPC
APE. Use this population to partition future front-end or L1TEX dependency
models by static path/target shape; do not fold the observer count into cycles.
