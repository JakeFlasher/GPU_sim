# SM86 integrated accuracy dashboard — H84 — 2026-10-16

## Evidence anchor

Latest full-first artifact/report:

* replay: `../sm86_replay_results/full_corpus_h83_ipc_readiness_20261016_v1/`
* report: `../sm86_replay_results/metric_report_h83_ipc_readiness_20261016_v1/`
* 356 roots: 286 `ok`, 62 `control_error`, 8 `replay_error`; 12,392 full
  launches; exactly 12 workers; compressed JSONL artifacts; NCU is post-replay
  comparison only.

## Current mechanism-local APE

Values below are mean relative error on the report's identity joins. Scope
exclusions remain explicit; raw/all-row authority is never replaced by the
secondary view.

| Endpoint | Mean APE | Compared | Status |
|---|---:|---:|---|
| L1 global-load sectors | 0.21% | 106/107 | H72 scope observer; one LTC-specific exclusion |
| L1 global-store sectors | 0.00% | 107/107 | baseline exact |
| L1 global lookup hit | 30.71% | 106/107 | H79 packet candidate remains observer-only because raw LTC row regresses |
| L1 global lookup miss | 32.39% | 106/107 | same boundary |
| Unified L1 hit rate | 29.04% | 107/107 | broader endpoint, not a physical hit law |
| L2 TEX sectors | 6.76% | 104/107 | three scope exclusions |
| L2 TEX lookup hits | 11.77% | 104/107 | write/other-client scope remains unresolved |
| L2 TEX lookup misses | 0.00% | 104/107 | exact on compatible rows |
| L2 op-write sectors | **9.37%** | 107/107 | H80 packet candidate selected |
| L2 op-write lookup hits | **9.36%** | 107/107 | H81 packet composition selected |
| L2 op-write lookup misses | 0.00% | 107/107 | all zero-target rows exact |
| DRAM read sectors | 26.84% | 58/107 | 49 scope mismatches; GCC/client demand unresolved |
| GPC elapsed cycles | 12.90% | 107/107 | no partition timing arm |
| SM active cycles | 18.02% | 107/107 | EXIT/queue timing arms rejected |
| SMSP active cycles | 17.51% | 107/107 | readiness decomposition only |

## Default composition and invariants

Full replay combines the implemented L1/L2 state, aggregate carded DRAM/read
service, scope observers, H80/H81 write operation endpoint, H82 route observer,
and H83 readiness observer. The following remain unchanged by the promoted
effective endpoints: L2 tag/dirty/pending state, DRAM read demand and service
frontier, write acknowledgements, SM/GPC frontiers, and IPC numerators/
denominators. Request-domain baseline rows are retained for differential audit.

## Closure gaps

The remaining >10% residuals are not addressed by relabelling: global L1 lookup
semantics need a form-qualified endpoint witness; DRAM reads need independent
GCC/instruction/maintenance completion ownership; and SM/SMSP/GPC timing needs
held-out issue/retirement probes. H71, H75, H77, H82, and H83 negative evidence
prevent speculative timing promotion. The next implementation seam must reduce
one of those residuals with a real-trace differential and full-first replay.
