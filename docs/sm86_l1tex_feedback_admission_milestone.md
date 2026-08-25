# Completed milestone: sm_86 feedback-coupled L1TEX admission

**Status:** complete 2026-08-18; feedback lane promoted opt-in, parameters withheld  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[finite L1TEX admission candidates](sm86_l1tex_finite_admission_candidates_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_admission_backpressure_20260818_v1/analysis/README.md`

## Goal

Close the causal defect exposed by schema 9: a queue-full delay must move later
warp demand and may change cache order. Post-hoc accumulation over fixed issue
timestamps cannot represent that feedback.

## Schema-10 execution contract

Direct replay accepts `--memory-admission-feedback` with a schema-1 document
containing exactly one finite admission candidate. The lane is launch-scoped,
requires full L1/L2 coverage, and is mutually exclusive with observational
service/admission candidate files.

For each matching effective STG or RED:

1. readiness includes the selected SM or SMSP queue;
2. the issue-cycle transaction capacity-filters up to four SMSP picks and a
   denied SMSP repicks other ready work rather than introducing a bubble;
3. accepted instructions reserve queue entries and emit compact group witnesses;
4. the SM yields after issue, scheduling every request before its next wake;
5. ordered L2 accesses reconstruct exact hit/miss work for the instruction;
6. the shared FIFO server assigns service start and completion;
7. service start releases the waiting-queue slot; and
8. a full queue parks later warps until the release frontier.

The pre-edge issue batch cannot consume a release created by another
instruction in the same unpriced batch; that release becomes visible at the
re-armed SM wake. Releases from already priced work may admit in the same
cycle. This keeps issue transactions deterministic without predicting an L2
outcome before the hierarchy computes it.

The candidate's `admission_latency_cycles` must equal the active issue model's
read-barrier latency. Schema 10 uses the existing architectural operand-read
timing source; a conflicting candidate is rejected rather than accepted as a
serialized no-op knob. The full candidate is also part of
`ReplayConfiguration` identity because this lane changes issue and cache order.
Reservation, service, and diagnostic counter updates preflight before commit.

Returning ATOMG is unchanged. Predicated-off records create neither a
reservation nor witness. Ordinary replay owns no feedback state and emits no
witnesses.

Schema 10 reports the selected candidate, admitted instructions, exact L2
hit/miss sectors, service work, maximum waiting occupancy, service completion,
and `memory_admission_queue` readiness cycles. The simulator still marks
STORE/RED retirement incomplete because dirty writeback, coherence, and RED
RMW completion remain outside the event stream.

## Candidate transfer

Three configurations frozen from schema-9 shallow calibration were replayed
twice over all 100 cells:

- STORE candidate 7: per-SMSP queue, capacity 16, hit/miss rates 1/10;
- RED candidate 12: SM queue, capacity 32, hit/miss rates 1/16; and
- RED candidate 14: per-SMSP queue, capacity 8, hit/miss rates 1/16.

All 300 pairs are byte deterministic after the adversarial scheduler repair.
The retained live GPU corpus supplies the unchanged correctness, native CLOCK,
NCU, SourceCounters, and NVBit labels. Selection used five shallow unique-line
depths; the remaining 20 cells per operation are strict selection holdouts.

| tail | candidate | calibration issue baseline → feedback MAE | held-out issue baseline → feedback MAE | held-out terminal baseline → feedback MAE | held-out issue max |
|---|---:|---:|---:|---:|---:|
| STORE | 7 | 1,639.4 → 774.6 | 9,761.5 → 3,446.4 | 7,623.9 → 1,336.7 | 23,227 |
| RED | 12 | 2,405.6 → 1,428.4 | 17,010.4 → 3,441.6 | 15,703.8 → 2,016.2 | 27,908 |
| RED | 14 | 2,405.6 → 1,388.0 | 17,010.4 → 3,499.5 | 15,703.8 → 1,969.8 | 28,009 |

These are like-for-like diagnostic-last-issue comparisons. The separate
held-out CLOCK-interval MAEs are 3,918.6 STORE and 3,441.2/3,497.9 RED. The
held-out result, rather than the earlier all-cell/CLOCK mixture, is the transfer
headline. Feedback closes most of the missing issue magnitude and substantially
improves the terminal lower bound, but does not identify a complete parameter.

## Corrected cache attribution

The aggregate Store-source lane changes in six STORE and seven/nine RED cells.
That does not mean the target operation's own hit/miss outcome changed:

- the same-ID schema-9 versus schema-10 RED observer changes in zero cells for
  both candidates; the seven/nine aggregate changes belong to later
  instrumentation stores whose order and inherited cache state changed;
- STORE candidate observations change in six cells, but target and
  instrumentation operations are both STORE. All six changes contain negative
  target-minus-control hit deltas, so they cannot be interpreted as physical
  target-only counts.

The supported conclusion is narrower: feedback executes a different downstream
cache order. This corpus does not identify a target STORE cache-outcome change
and shows no target RED cache-outcome change.

No frozen parameter transfers completely. Deep depth-128 issue remains
underpredicted by roughly 23–28 thousand cycles, and CTA/spacing residuals
remain material.

Repeat sensitivity also bounds the claim. NCU repeat counts are 3/5/7 in
69/20/11 cells, 31 target/control pairs have unequal counts, and using only the
common first three repeats preserves the held-out terminal direction. The
largest observed repeat-envelope widths are 28,121 GPC, 15,890 issue, and 5,058
fence cycles; these are sensitivity ranges, not confidence intervals.

## Decision

Promoted:

- the schema-10 feedback mechanism as an explicit research lane;
- transactional same-cycle capacity filtering;
- same-SMSP fallback selection after shared-capacity denial;
- issue-to-L2 yield and re-arm ordering;
- queue readiness as a causal scheduler constraint; and
- structurally identified timing-active replay configuration.

Not promoted:

- candidates 7, 12, or 14 as architecture defaults;
- a unique SM/per-SMSP queue scope or capacity;
- the 1/10 or 1/16 hit/miss rates;
- complete STORE/RED retirement; or
- numerical GPC emission.

## Configured successor

Hold the feedback machinery fixed and vary only downstream miss service rate or
service width. Calibrate on shallow unique-line depths; keep deep burst, reuse,
spacing, CTA transfer, and sparse-SMSP placement held out. A candidate must
improve both issue and terminal transfer, preserve deterministic executable
cache semantics, and avoid an operation-specific intercept.

Evidence limitations remain explicit: the 32 MiB eviction sweep is 5.33 times
reported L2 capacity but does not prove complete coldness; each NCU profile has
nine application-replay passes; all 100 NVBit module origins remain unresolved;
executed host binaries have no retained content identity; and the holdout is a
retrospective separation from candidate selection rather than prospectively
blinded data. No privileged command was required.

This successor is complete in
[feedback miss-service rate and width](sm86_l1tex_feedback_service_rate_width_milestone.md).
The sweep adds an explicit width quantum but promotes no rate or width; it
configures an independent write-class completion frontier next.
