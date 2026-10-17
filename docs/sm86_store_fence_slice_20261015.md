# SM86 source-qualified store/fence completion tail — 2026-10-15

## Mechanism

H19 extends the existing `MemoryAdmissionFeedback`/write-ack seam rather than
inventing a second queue. The default SM86 full replay now loads a bounded
candidate with carded Ampere priors: five waiting instructions per SMSP,
8-cycle admission visibility, rational L2 hit/miss sector service, and an
8-lane region completion endpoint. Static gating is strict: the candidate is
active only for a trace with global STORE/RED work and a device-scope
`MEMBAR.GL`, `MEMBAR.GPU`, or `MEMBAR.SC.GPU` (or explicit ERRBAR/CCTL marker).
Other traces retain the ordinary replay path. The tail is a simulator event
frontier and typed write acknowledgement; it does not claim dirty writeback,
coherence completion, or proprietary queue timing.

The candidate values are kept in
`arch_profiles/candidates/sm86_store_fence_candidate_v1.json` and were derived
from the independent clock64 store/fence topology probe in
`/home/jakeshea/GPU_sims_notes`; no NCU value is used as an input or selector.

## Representative full-first replay

The final candidate code completed full replay 5/5 on vendor SGEMM, strided
SGEMM, vector copy, streaming dot product, and implicit-GEMM convolution:
`../sm86_replay_results/h19_store_fence_candidate_20261015_v1/full.jsonl.zst`.
The feedback gate remained inactive on non-fence streaming/vector roots and
active on qualified store/fence kernels, producing 103,737 write acknowledgements
across the complete corpus path below.

## Complete-corpus evidence

Default full-first replay (the runner auto-selected the candidate) completed
356 roots (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392
launches in 36.577 s with exactly 12 workers:
`../sm86_replay_results/full_corpus_h19_store_fence_default_20261015_v2/full.jsonl.zst`.
Fast output is retained at `.../fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h19_store_fence_default_20261015_v2/`.
The final run recorded `memory_admission_feedback_defaulted=true` and
`ncu_used_as_input=false`.

## Exact residuals versus H18

| Metric | H18 mean APE | H19 mean APE | Delta |
|---|---:|---:|---:|
| L1 global-load lookup hits | 30.71% | 30.71% | 0.00 pp |
| L1 global-load lookup misses | 58.44% | 58.44% | 0.00 pp |
| L2 TEX source sectors | 28.24% | 28.24% | 0.00 pp |
| DRAM read sectors | 40.91% | 40.91% | 0.00 pp |
| GPC elapsed cycles | 12.91% | 12.91% | 0.00 pp |
| SM active cycles | 18.01% | 18.01% | 0.00 pp |
| elapsed IPC | 12.31% | 12.31% | 0.00 pp |

H19 closes the source-qualified completion plumbing and records real write/ack
work, but does not yet lower the corpus mean APE: qualified tails are not on the
current scored launch critical path. Preserve the candidate as the default
bounded effective tail and continue with a mechanism-local critical-path
refinement rather than fitting an NCU delta.
