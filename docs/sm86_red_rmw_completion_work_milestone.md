# Completed milestone: sm_86 RED RMW completion work and parallelism

**Status:** complete 2026-08-19; schema-13 mechanism retained opt-in, no rate or lane count promoted  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[operation-specific write-completion latency](sm86_write_completion_latency_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_admission_backpressure_20260818_v1/analysis/red_rmw_completion_sweep.md`

## Goal and component

Test whether the remaining non-returning RED fence residual follows exact RMW
work rather than a constant post-service latency. Readiness schema 13 retains
two instruction-local witnesses from replay: L2 request count and unique
256-byte-region count. A feedback candidate may charge either or both witnesses
on one to 32 independent completion lanes.

Each job starts at the later of its admission-service completion and the
earliest lane-ready time. It occupies that lane for additive request/region
work, then optionally receives the schema-12 fixed tail without occupying the
lane. Admission server occupancy, queue release, and cache order do not change.
The resulting frontier remains per warp and delays only that warp's
`MEMBAR.SC.GPU`.

The implementation also exposes exact request, region, busy-cycle, and terminal
completion diagnostics. Work without MEMBAR feedback, work without a lane, more
than 32 lanes, and lanes without work are typed errors. Omitted work preserves
schema 11/12 serialization and behavior.

## Frozen experiment

The RED predecessor remains SM-scoped capacity 32 with an eight-cycle operand
floor and exact one-cycle-hit/16-cycle-miss sector service. Eighteen candidates
cross:

- request work at 2, 4, and 8 cycles per L2 request;
- region work at 4, 8, and 16 cycles per unique 256-byte region; and
- one, two, and four completion lanes.

Five shallow unique-line cells form the predeclared fence/GPC calibration set.
All 19 configurations, including the zero-work schema-11 predecessor, lie on
its Pareto frontier: more work makes a small fence improvement while increasing
GPC error. Line-unique calibration has two requests per region, so request rates
2/4/8 and region rates 4/8/16 are intentionally indistinguishable there. Lane
counts also tie. All 18 schema-13 candidates therefore advance without
refitting to the 20-cell topology, occupancy, spacing, deep-burst, sparse-SMSP,
and CTA-transfer holdout.

## Strict holdout

| candidate | basis/rate | lanes | fence MAE | GPC MAE | fence max | GPC max |
|---:|---|---:|---:|---:|---:|---:|
| 202 | no work | 0 | 3,872.3 | 1,466.0 | 9,599 | 3,112 |
| 214 | request / 4 cycles | 1 | 3,818.0 | 1,462.6 | 9,567 | 3,112 |
| 218 | request / 8 cycles | 2 | 3,797.2 | 1,477.2 | 9,535 | 3,112 |
| 220 | region / 4 cycles | 1 | 3,857.1 | 1,470.1 | 9,583 | 3,112 |
| 228 | region / 16 cycles | 4 | 3,822.3 | 1,492.2 | 9,535 | 3,112 |

Candidate 214 is the only central-label configuration that improves both
predecessor MAEs, by 54.3 fence cycles and 3.45 GPC cycles. Candidate 218 is the
other retrospective holdout-Pareto point. The apparent candidate-214 win is not
robust: it loses the GPC improvement under the observed lower repeat-envelope
label. Those envelopes are sensitivity bounds rather than confidence
intervals, but their multi-thousand-cycle scale is far larger than the central
gain. No candidate improves both targets across all retained repeat-sensitivity
labels.

The work model helps address-topology residuals while worsening several CTA,
spacing, sparse-SMSP, or occupancy groups. One-lane request/8 and
region/16 candidates also overload badly, whereas added lanes remove most of
that failure. This demonstrates that exact work and bounded completion state
are executable discriminators, but the existing evidence does not identify a
uniform rate, basis, or width.

## Decision and successor

Promoted only as explicit research machinery:

- schema-13 exact request/256-byte-region completion witnesses;
- bounded independent completion lanes and per-warp MEMBAR integration;
- transactional, overflow-checked completion accounting; and
- strict calibration/holdout plus repeat-sensitivity reduction.

No schema-13 candidate becomes an architecture default. STORE remains at the
schema-11 zero-tail frontier, and RED numerical completion remains typed
incomplete. A richer RED timing model would need direct evidence for how the
many-region network partitions work; this corpus does not justify inventing
that mapping.

The sweep retains 1,800 deterministic schema-13 replay pairs. The active replay
corpus is 5,519 pairs after final regeneration. It reuses the retained 384
multi-pass NCU profiles and 100 NVBit traces; no new GPU capture or sudo was
required. The next engineering milestone is bounded simulator event-throughput
profiling and optimization, while the many-region atomic/RMW mapping remains an
explicit future measurement problem.
