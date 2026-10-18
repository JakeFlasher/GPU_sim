# SM86 L1 wavefront lookup semantics — P1 negative decision — 2026-10-16

## Independent same-binary probe

The notes probe `l1tex_arrival_replay_service` was rebuilt for SM86 with the
same host identity used by the corpus: GeForce RTX 3090 Ti, UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, driver
615.71.09. Correctness passed for line-layout load/store and word-layout load
controls. NCU application-replay observations for the cold load controls were:

| control | sectors | lookup hits | lookup misses | requests | output wavefronts |
|---|---:|---:|---:|---:|---:|
| load, line, W1, depth8 | 21,504 | 0 | 21,504 | 672 | 5,376 |
| load, word, W1, depth8 | 2,688 | 0 | 2,688 | 672 | 672 |
| store, line, W1, depth8 | 21,504 | 0 | 21,504 | 672 | 5,376 |

Raw NCU exports and the rebuilt binary are retained under
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/l1tex_arrival_replay_service/results/h65_l1_semantics_ncu_20261016_v1/`.
This confirms that cold same-request sector expansion must not automatically be
counted as a profiler lookup hit, but it does not identify one universal rule
for every client/packet class.

## Full-corpus A/B

A bounded experimental source change replaced the H16 projection
`hits + merges - line_alloc_misses` with `hits + pending_lookup_hits`, leaving
cache state, timing, and traffic unchanged. The complete real corpus was run
full-first with 12 workers and compressed output:

- Replay: `../sm86_replay_results/full_corpus_h65_l1_lookup_wavefront_v1/`
- Report: `../sm86_replay_results/metric_report_h65_l1_lookup_wavefront_20261016_v1/`
- Coverage: 356 roots, 286 `ok`, 62 controls, 8 failures, 12,392 full launches.

The experiment reduced zero-target false positives for lookup hits from 22/29
to 1/29, but worsened the nonzero-target residuals:

| Metric | H41/H63 baseline | Unqualified experiment |
|---|---:|---:|
| L1 global-load lookup-hit mean APE | 30.71% | 33.43% |
| L1 global-load lookup-miss mean APE | 58.44% | 110.05% |
| Scope-compatible lookup-hit zero-target false positives | 22/29 | 1/29 |

## Decision

The unqualified formula is rejected and the source has been restored to the
H16/H63 projection. The probe separates sector and output-wavefront endpoints,
but does not justify collapsing them or applying one client-independent rule.
Future work must gate any lookup correction by packet/client/form and preserve
sector conservation; wavefront counters should remain separate observer rows.
No timing, L1 state, L2 state, DRAM service, raw APE authority, or secondary
scope policy changed in the retained default.
