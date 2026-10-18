# SM86 static LDG.CONSTANT source slice — 2026-10-15

## Scope

H5 adds a static-opcode-qualified observer for `LDG.*.CONSTANT` global loads.
It is deliberately separate from H2's CBANK field-15 offsets: these records
carry ordinary traced global addresses and already use the existing L1/L2
constant-return timing route. H5 emits instruction, sector, per-instruction
128-B line-candidate, and CTA/warp-scoped unique-line counts without equating
them to GCC requests, L2 sectors, or DRAM service.

## Representative full replay

| Root | Constant-load instructions | Sectors | 128-B request candidates | Unique CTA/warp lines | NCU GCC requests |
|---|---:|---:|---:|---:|---:|
| FP16 HMMA | 96 | 768 | 768 | 96 | 16 |
| FP32 dependency chain | 1,024 | 8,704 | 8,704 | 512 | 64 |
| local/register spill | 260 | 4,112 | 1,028 | 16 | 16 |
| RMSNorm BF16 | 1,536 | 3,072 | 1,536 | 1,024 | 384 |
| cp.async tile copy | 0 | 0 | 0 | 0 | 12 |

The candidate scope is intentionally not treated as an NCU prediction: dynamic
instruction repetitions, warp/CTA aggregation, GCC cache residency, and the
128-B return path are distinct mechanisms. NCU values are post-replay only.

## Complete-corpus evidence

The full-first replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8
replay/input failures), 12,392 successful launches in 34.915 s with 12 workers.
Artifact:
`../sm86_replay_results/full_corpus_h5_ldg_constant_20261015_v1/full.jsonl.zst`.
Successful rows emitted 6,807,596 static constant-load instructions,
10,817,035 source sectors, 7,142,629 request candidates, and 1,039,257
CTA/warp-scoped unique-line candidates.

## Residual deltas

H5 is source attribution only; full exact-identity residuals remain unchanged
from H4/H3:

| Metric | Mean APE |
|---|---:|
| DRAM read sectors | 40.91% |
| `sm__cycles_active.avg` | 18.26% |
| `sm__inst_executed.avg.per_cycle_elapsed` | 12.53% |

Report: `../sm86_replay_results/metric_report_h5_ldg_constant_20261015_v1/`.

H5 closes a missing static source-class observation but is not an accuracy
improvement. The next arm must add an independently evidenced GCC/L1.5
response-scope discriminator or a bounded DRAM service relation, without fitting
NCU values from the scored workloads.
