# SM86 full-replay implementation slice — 2026-10-15

## Scope

Architecture-neutral static metadata conversion and the SM86 complete-corpus
replay runner. NCU remained post-replay only; no profile value was used as a
replay input or parameter selector.

## Implementation

- `tools/sm120_replay_bench.py` now resolves architecture from explicit kernel
  metadata, accepts the stale SM86 `native_static_sm120_v3` producer label, and
  emits an architecture-correct control sidecar.
- `tools/sm120_corpus_replay.py` now defaults to the SM86 corpus, SM86 card and
  p-chase card; full mode translates to integrated `l1-l2`, enables carded DRAM
  by default for SM86, excludes `_diagnostics`, and preserves twelve-worker,
  bounded, compressed full-first replay behavior.

## Evidence

### Representative real batch

Roots: `fp32_fma_dep_chain`, `fp16_hmma_gemm_traceable`,
`cp_async_shared_tile_copy`.

| Mode | Result |
|---|---:|
| Full integrated L1/L2 + carded DRAM | 3/3 `ok` |
| Fast flat frontier | 3/3 `ok` |

Artifacts: `../sm86_replay_results/runner_batch_20261015_v1/{full,fast}.jsonl.zst`.

### Complete-corpus full replay

| Field | Result |
|---|---:|
| Roots discovered | 356 |
| Successful roots | 286 |
| Mixed-code control dispositions | 62 |
| Replay/input failures | 8 |
| Launches in successful rows | 12,392 |
| Driver wall time | 32.845 s |
| Workers | 12 |
| NCU used as replay input | No |

Artifact: `../sm86_replay_results/full_corpus_20261015_v2/full.jsonl.zst`.

The 62 mixed-code roots contain SM80 code objects executing on the SM86 host;
they remain explicit controls. The eight failures are retained with real
causes: unsupported sparse IMMA pipeline, trace/resource limits, local-memory
opcode mismatch, invalid grid range, and unresolved allocation identity.

## Interpretation and residuals

This slice establishes real-trace sidecar conversion, full-mode dispatch,
compressed artifact production, bounded concurrency, and corpus termination.
It does **not** establish sub-10% NCU accuracy. Successful full rows still
carry conditional dirty-writeback/coherence tails, and NCU joins/APE reduction
remain the next implementation slice.
