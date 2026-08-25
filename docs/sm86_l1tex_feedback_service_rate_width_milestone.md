# Completed milestone: sm_86 feedback miss-service rate and width

**Status:** complete 2026-08-19; width mechanism retained opt-in, parameters withheld  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[feedback-coupled L1TEX admission](sm86_l1tex_feedback_admission_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_admission_backpressure_20260818_v1/analysis/feedback_service_sweep.md`

## Goal

Hold schema-10 queue feedback fixed and ask whether downstream L2-miss service
rate or service width explains the remaining STORE/RED transfer error. Queue
scope, queue capacity, hit service, operand-read latency, scheduler policy, and
the live GPU labels remain unchanged.

## Essential component

The inherited `(units, cycles)` formula is a fluid rational rate:

```text
fluid_cycles = ceil(work * cycles / units)
```

Scaled-equivalent pairs therefore cannot represent width. The admission
candidate now has an optional `l2_miss_service_width_sectors`. Zero preserves
the prior formula and is omitted from serialized legacy configurations. A
positive width charges fixed miss-sector quanta:

```text
round_cycles = ceil(width * cycles / units)
busy_cycles  = ceil(miss_sectors / width) * round_cycles
```

The final partial batch pays a complete round. Width is bounded to 256 sectors,
checked before replay, included in feedback configuration identity, and used by
both observational schema 9 and timing-active schema 10. Existing width-zero
results remain byte compatible.

## Frozen search and selection

The sweep contains 16 STORE candidates (rates 8/10/12/14 crossed with widths
1/2/4/8) and 20 RED candidates (rates 8/12/16/20/24 crossed with those widths).
STORE retains per-SMSP capacity 16; RED retains SM capacity 32. Hit service is
one sector/cycle and operand visibility is eight cycles throughout.

Five shallow unique-line cells are calibration. Among width-one candidates,
the selection retains the Pareto frontier over last-issue MAE and downstream
service-completion MAE. All widths at every Pareto rate advance together; the
remaining 20 cells per operation are holdout.

| tail | rate | calibration issue MAE | calibration service-completion MAE | Pareto |
|---|---:|---:|---:|---|
| STORE | 8 | 1,011.4 | 644.6 | no |
| STORE | 10 | 774.6 | 451.0 | yes |
| STORE | 12 | 943.8 | 1,025.8 | no |
| STORE | 14 | 1,347.0 | 1,768.2 | no |
| RED | 8 | 1,945.0 | 2,206.2 | no |
| RED | 12 | 1,475.2 | 845.0 | yes |
| RED | 16 | 1,428.4 | 969.0 | yes |
| RED | 20 | 2,222.0 | 2,556.2 | no |
| RED | 24 | 3,030.4 | 4,143.4 | no |

The retained sweep has 1,800 deterministic pairs: all 100 cells for 16 full
rate/width configurations, plus ten calibration target/control cells for 20
non-advanced configurations. An early exploratory reduction compared rate to
raw system terminal; after recognizing that raw terminal aliases last issue for
replyless writes, selection was corrected to the downstream service frontier.
Those already-run STORE-rate-12 results remain labeled sensitivity data and do
not enter selection. This is an honest retrospective development split, not a
prospectively blinded experiment.

## Strict holdout result

| tail/rate | width | issue MAE | raw-terminal MAE | service-completion MAE | issue max | completion max |
|---|---:|---:|---:|---:|---:|---:|
| STORE/10 | 1 | 3,446.4 | 1,336.7 | 1,127.2 | 23,227 | 3,452 |
| STORE/10 | 2 | 3,435.9 | 1,326.2 | 1,129.2 | 23,187 | 3,452 |
| STORE/10 | 4 | 3,666.1 | 1,515.8 | 1,508.9 | 23,027 | 4,114 |
| STORE/10 | 8 | 4,383.4 | 2,233.2 | 2,540.8 | 22,312 | 14,354 |
| RED/12 | 1 | 7,160.9 | 5,854.3 | 5,113.4 | 54,837 | 31,529 |
| RED/12 | 2 | 7,094.8 | 5,788.1 | 5,051.6 | 54,837 | 31,529 |
| RED/16 | 1 | 3,441.6 | 2,016.2 | 1,441.1 | 27,908 | 3,112 |
| RED/16 | 2 | 3,433.1 | 2,044.1 | 1,518.3 | 27,908 | 3,112 |

Widths four/eight fail address-topology transfer. At width two, STORE gains
only 10.5 issue-MAE cycles while losing 2.0 service-completion cycles; RED/16
gains 8.5 issue cycles while losing 77.2 completion cycles. These differences
are far below the retained repeat envelopes. No candidate improves both issue
and service completion over its width-one predecessor. RED rate 12 is
calibration-Pareto but fails holdout; rate 16 returns to the predecessor.

## Structural finding and decision

For all 20 predecessor holdout cells in both operations, raw modeled terminal
equals modeled last issue. Yet the two observed labels differ: the mean absolute
native-issue versus NCU-GPC gap is 2,569.8 STORE and 2,466.6 RED cycles, reaching
20,379/23,308. Any one shared prediction therefore has at least that joint-MAE
lower bound by the triangle inequality.

Promoted:

- an explicit, bounded, backward-compatible miss-service quantum mechanism;
- strict rate calibration followed by width transfer;
- downstream service completion as a separate diagnostic frontier; and
- exact replay/resume validation for all sweep configurations.

Not promoted:

- a new STORE or RED miss rate;
- widths 1, 2, 4, or 8 as an architecture default;
- the raw system terminal as write-class completion; or
- complete STORE/RED retirement or numerical GPC emission.

## Configured successor

Stop tuning one frontier. Keep admission feedback on the predecessor rates and
build an independent, opt-in write-class completion candidate. It must separate
queue-release/issue readiness from STORE visibility and RED RMW/fence
completion, score native fence and NCU GPC independently, and retain typed
incomplete retirement until dirty writeback, coherence, and RED RMW semantics
are actually modeled.

The inherited evidence limits remain: 32 MiB eviction is a strong but not
complete coldness proof, NCU metrics come from nine application-replay passes,
NVBit module origin is unresolved, and selection is retrospective. No sudo or
new GPU capture was required.

This successor is complete in
[write-class completion feedback](sm86_write_class_completion_feedback_milestone.md).
Schema 11 makes each warp's GPU MEMBAR observe its own write-service completion
frontier while leaving retirement capability gaps explicit.
