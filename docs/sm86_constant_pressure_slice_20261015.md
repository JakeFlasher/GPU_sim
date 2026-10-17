# SM86 constant-pressure L1TEX observer — 2026-10-15

## Mechanism

H28 makes the existing source-qualified SM86 constant-pressure path explicitly
observable in the memory-issue diagnostics. A static SASS classifier identifies
constant-load streams beyond the released five-entry Ampere dispatch window
(and the narrow U16/write-barrier pressure shape); the existing effective
operand frontier remains `11 + 39` cycles. H28 adds only a dynamic count of
instructions entering that class—no new latency, cache mutation, or NCU fit.

## Evidence

Representative full-first replay passed one direct streaming root and four
high-IMC/constant-pressure roots (1,156 full launches):
`../sm86_replay_results/h28_constant_pressure_20261015_v2/full.jsonl.zst`.

The complete full-first replay passed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 40.672 s with 12
workers:

`../sm86_replay_results/full_corpus_h28_constant_pressure_20261015_v1/full.jsonl.zst`

Post-replay report:
`../sm86_replay_results/metric_report_h28_constant_pressure_20261015_v1/`.

The corpus observes 6,336,451 constant-pressure instructions and 1,882,174
dynamic non-fall-through transfers. Readiness schema is 31. H28 is bit-for-bit
neutral versus H27 for all L1/L2/DRAM counters, IPC/frontiers, and APE.

## Verdict and next seam

The observer validates broad pressure coverage without claiming that the static
pressure boundary is a universal hardware queue threshold. A future timing arm
must be tied to an independent clock64 load-return/queue experiment; until then
the pressure count remains a mechanism-local partition for residual analysis.
