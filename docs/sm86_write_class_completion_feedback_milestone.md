# Completed milestone: sm_86 write-class completion feedback

**Status:** complete 2026-08-19; schema-11 mechanism promoted opt-in, retirement incomplete  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[feedback miss-service rate and width](sm86_l1tex_feedback_service_rate_width_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_admission_backpressure_20260818_v1/analysis/write_completion_feedback.md`

## Goal

Separate admission/issue readiness from write-class completion. Schema 10
released an admission slot at service start but allowed `MEMBAR.SC.GPU` to pass
without observing service completion. Raw terminal therefore aliased last issue
for replyless STORE/RED, even though the live issue and GPC labels were distinct.

## Schema-11 execution contract

`membar_waits_for_service_completion: true` is an optional addition to one
schema-10 admission-feedback candidate. Existing candidates omit it and remain
schema 10 and byte compatible.

For the completion lane:

1. queue scope, capacity, hit/miss rate, and queue release are unchanged;
2. each priced matching STORE/RED updates only the issuing warp's completion
   frontier with its service-done timestamp;
3. `MEMBAR.SC.GPU` for that warp waits for its own frontier, not an SM-global
   completion tail or another warp's later work;
4. post-fence instrumentation stores cannot contaminate the frontier observed
   by the earlier MEMBAR; and
5. three CLOCKLO issues report first-to-second issue and second-to-third fence
   intervals separately.

The uncommon per-warp completion/CLOCK state lives in the already boxed
feedback lane. Ordinary schemas retain their prior inline scheduler footprint.
Readiness schema 11 adds `memory_write_completion` as a causal relation.

## Frozen campaign

Two candidates replay the predecessor configuration without refitting:

- STORE 201: per-SMSP capacity 16, hit/miss rate 1/10;
- RED 202: SM capacity 32, hit/miss rate 1/16.

Both run twice over all 100 cells. The live correctness, seven-run native CLOCK,
384 NCU profiles, SourceCounters, and 100 NVBit captures are unchanged.

## Strict holdout result

| tail | zero-fence MAE | schema-11 fence MAE | fence max | predecessor service/GPC MAE | schema-11 service/GPC MAE | GPC max |
|---|---:|---:|---:|---:|---:|---:|
| STORE | 2,860.9 | 496.4 | 2,456 | 1,127.2 | 1,124.8 | 3,452 |
| RED | 6,246.4 | 3,872.3 | 9,599 | 1,441.1 | 1,466.0 | 3,112 |

STORE fence transfer is strong across topology, CTA shape, and occupancy; its
largest residual is the spacing cell. RED moves in the correct direction but
deep burst, CTA, spacing, and sparse-SMSP residuals remain 2.4–8.1 thousand MAE
by stage. The existing RED service rate is not a RED RMW completion model.

The target-operation observer changes in zero RED cells. Aggregate Store-source
changes in six RED cells belong to later instrumentation stores. STORE target
and instrumentation remain the same operation, so its six observer/source
changes are not target-only cache evidence. Completion feedback preserves the
predecessor GPC completion scale rather than manufacturing a cache refit.

## Decision

Promoted:

- per-warp write-service completion frontiers;
- `MEMBAR.SC.GPU` readiness on that frontier;
- schema-11 `memory_write_completion` attribution;
- phase-specific three-CLOCK diagnostics; and
- boxed opt-in state with legacy serialization compatibility.

Not promoted:

- complete STORE visibility, dirty writeback, or coherence;
- RED read-modify-write completion;
- the RED fence residual as acceptable timing;
- a new rate, width, latency, or architecture default; or
- numerical GPC emission.

## Configured successor

Retain schema 11 as the fence/visibility skeleton. Next, evaluate independent
operation-specific completion candidates without changing admission timing:
STORE visibility/dirty-writeback latency and RED RMW completion work must be
separate dimensions. Calibrate on shallow fence cells and keep deep burst,
spacing, topology, CTA, and sparse-SMSP cells held out. Retirement stays typed
incomplete until those semantics, not merely their fitted timestamps, exist.

The retained limitations remain: NCU metrics are multi-pass, NVBit module origin
is unresolved, STORE target attribution shares an operation with instrumentation,
and this is retrospective mechanism development. No sudo or new GPU capture was
required.

This successor is complete in
[operation-specific write-completion latency](sm86_write_completion_latency_milestone.md).
Schema 12 cleanly tests a post-service tail, but every nonzero latency fails the
joint fence/GPC holdout requirement.
