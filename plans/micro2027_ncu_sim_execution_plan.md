# Event-driven simulator execution roadmap

## Objective

Build a useful, fast `sm_86` trace-driven simulator, compare it with real NCU
measurements, then extend it to `sm_120`.

## Technical constraints

- NCU values are evaluation targets, not replay inputs.
- Keep the simulator event-driven and batch local actor progress.
- Keep architecture-dependent values in architecture profiles.
- Treat unknown behavior as a research question, not an inherited constant.
- Use direct GPU/toolchain access and focused tests throughout.
- Milestones organize work; they never prevent starting another sound task.

## Current nearest milestone — configured 2026-08-17

The **sm_86 causal warp readiness and async-copy/CTA-barrier handoff** milestone
is complete. It retained dependency origin, async source/local/SM-service
completion, CTA generations/arrival spread/rendezvous, and scheduler-selection
losses in readiness schema 1. Its 36-cell live campaign rejected a fixed CTA
latency because barrier-only cost and the coupled interaction do not transfer
from W4 to held-out W2/W8. No timing rule was promoted, and the six-workload
baseline remains exactly 33.403% active-SM MAPE. The full result is in
[`docs/sm86_causal_readiness_milestone.md`](../docs/sm86_causal_readiness_milestone.md).

The **sm_86 barrier pipeline and launch-chain cache handoff** milestone is also
complete. It adds transactional stream-L2 replay and one SM-shared
two-cycle-per-warp barrier-arrival service selected by a 33-cell W1-W16 sweep.
Coupled W2/W4/W8 warm-route and fixed-eight-warp multi-CTA campaigns transfer
the route and CTA-shape ordering without a fixed release, re-arm, route, or
CTA-count constant. Six-workload MAPE is 33.394%. The full result is in
[`docs/sm86_barrier_pipeline_cache_handoff_milestone.md`](../docs/sm86_barrier_pipeline_cache_handoff_milestone.md).

The **SM86 LDGSTS front-end retained-data discriminator** is complete. A
paper-derived LDGSTS-only composition of five effective outstanding positions,
4/2-cycle local/shared frontiers, and 13/39-cycle boundaries fails transfer,
guard, strict route-order, corrected-sign, and all-replay accounting gates.
The candidate surface is retired; production stays at the 33.128% baseline.
See the
[`not_retained` decision](../results/captures/sm86_ldgsts_front_end_transfer_20260828_v1/README.md).

The **sm_86 execution-port and control-pipeline closure** milestone is complete.
A 68-cell campaign rejects generic scalar initiation and dual issue, while an
exact predicated-forward-BRA rule reduces branch slope MAE from 5.033 to 0.033,
improves b11 skew, preserves all traffic, and improves production MAPE to
33.128%. See
[`docs/sm86_issue_port_control_pipeline_milestone.md`](../docs/sm86_issue_port_control_pipeline_milestone.md).

The **sm_86 SMSP scheduler and math-pipe contention** milestone is complete.
Its 99-cell campaign identifies a two-cycle high-occupancy ALU/bare-IMAD
service bound but rejects ten direct-port, FIFO/oldest, rational, bounded-queue,
and service-drain candidates. The best candidate fits W8 but changes W4,
overprices W12, and loses reversed-family order signs. Production is restored
byte-identically and remains at 33.128% MAPE. See
[`docs/sm86_scheduler_math_pipe_contention_milestone.md`](../docs/sm86_scheduler_math_pipe_contention_milestone.md).

The **sm_86 fixed-latency Control/Allocate and
register-file contention**, specified in
[`docs/sm86_control_allocate_register_file_milestone.md`](../docs/sm86_control_allocate_register_file_milestone.md).
That milestone is complete. The trusted SASS operand/reuse projection, 28-cell
exact-bank campaign, and 32-cell RAW-distance campaign are retained. Five
Control/Allocate/completion candidates fail complete transfer; the best
treatment-delta MAE is 2.906 cycles/generation but the prior scheduler MAE
barely changes. Production timing is restored byte-identically and remains at
33.128% MAPE.

The **sm_86 compiler-managed reuse cache and fixed-latency
result/writeback queue**, specified in
[`docs/sm86_reuse_result_queue_milestone.md`](../docs/sm86_reuse_result_queue_milestone.md),
has completed its RFC slice. Exact treatments support shared bank/source-
position entries tagged by warp/register, but the best timing composition
fails predecessor transfer and is inactive. Production remains byte-identical
at 33.128% MAPE.

Seven slices of **sm_86 fixed-latency result, bypass, and writeback queue**,
specified in
[`docs/sm86_result_writeback_queue_milestone.md`](../docs/sm86_result_writeback_queue_milestone.md),
are complete. The pacing pilot is retained as negative evidence. Balanced
producer order establishes an exact W4/W8 opposite-bank null and isolates a
separate W16 global phase. A compiler-native R4/R8 same-bank pair then proves
that two coincident fixed results are accommodated and bypassed without exposed
W4/W8 delay. Held-out W12 adds a 0.742-cycle/generation bank-specific onset,
but queue occupancy and RF retirement remain confounded. Result timing remains
inactive. A subsequent 7/8/9 issue-reservation campaign finds a W8 ninth-step
canary, but capacity-8/pop-1 changes no replay output and is rejected. The
fixed/L1TEX campaign then proves exact W1/W4 same-bank +1-cycle load deferral
with a null opposite-bank control and promotes a narrow HADD2-over-load rule.
It also corrects sm_86 STRONG.SM L1 routing, reduces five-width same-pair delta
MAE from 1.058 to 0.305 cycles/generation, and preserves 234 predecessor plus
12 production outputs byte-identically. A direct consumer campaign then
reproduces the paper's semantic boundary: fixed distance 3 fails and distance
4 is valid, while a global-address consumer faults at distance 4 and succeeds
at 5. Exact `IMAD.IADD` projection is retained; no timing scoreboard repairs
invalid schedules. The dense W4–W16 successor localizes the remaining shift to
a per-SMSP instruction-frontier phase beginning at W5 and rejects ten direct
IBuffer/fetch candidates despite substantial MAE improvement. The nearest work
was completed by a 96-cell U1/U2/U4/U8 footprint factorial. Exact direct NCU
counts identify three resolving cycles per uniform backward target; replay now
records that typed architecture-local state without adding delay beyond the
five-cycle BRA control. Twenty-four candidates still do not identify a
transferable IBuffer capacity. The paper-style branch-free successor is also
complete: capacity two wins both CLOCK objectives after correct in-flight
reservation, but incompatible instruction-cache parameters and less than 15%
exact event-order accuracy reject promotion. The atomic successor is now
complete through adversarial follow-up: alias-aware sectors, exact L2 request
topology, and CLOCK-boundary diagnostics are active. Partial aliases reject the
operation-only service candidate; 1–168 CTA placement rejects a per-SM queue,
ordinary cache hash, and portable spacing rule. Atomic timing was withheld for
a pairwise collision graph. That closure now promotes an exact aligned-256-byte
cross-warp contention region,
but shows that many-region scheduling is not a fixed retained hash; timing stays
inactive. W16 remains only a phase sentinel outside focused campaigns.

## Work sequence

### 1. Direct experiment path

- Provide direct NCU and NVBit commands.
- Make simulator inputs ordinary paths with useful validation errors.
- Remove legacy capture lifecycle wrappers and duplicated schemas.
- Produce one end-to-end example from CUDA workload to trace to prediction.

### 2. Cycle semantics

- Complete: ordinary NCU labels are per-kernel launch envelopes; Range Replay
  is a separate host/API range observable that SASS-only replay cannot derive.
- Complete: replay separates last issue, last processed system event, their raw
  maximum, and typed complete/incomplete retirement status.
- Complete: 80 cells, 240 kernel profiles, 240 range profiles, and 80 NVBit
  traces reject sum/max workload reducers and immediate numeric GPC emission.
- Next: model the measured L1TEX/request and store/RED retirement tails before
  calibrating a per-grid launch envelope; require timed command-stream input
  for any later workload-range prediction.
- Complete successor: a 102-cell depth/lane/layout/reuse/fence campaign rejects
  fixed retirement delays and promotes readiness schema 6 exact instruction,
  sector, ideal-wavefront, and issue-span diagnostics without timing changes.
- Next bounded discriminator: vary concurrent per-SM arrival pressure at fixed
  ideal wavefront work to separate L1TEX input service from internal output
  replay before completing store/RED retirement.
- Complete discriminator: 75 cells hold ideal work fixed across warp
  concurrency, CTA partition, skew, and line topology. Unique 128-byte lines
  and atomic requests are promoted into schema 6; fixed-rate, affine-work, and
  output-replay timing fail held-out transfer.
- Next prerequisite: evaluate candidate servers at actual per-instruction
  arrival timestamps through a bounded opt-in diagnostic before changing
  L1TEX issue or retirement timing.
- Complete prerequisite: readiness schema 7 evaluates 16 fixed-capacity
  SM/SMSP candidates online. Across 75 deterministic replay pairs, selected
  SM-shared line candidates reduce structurally held-out STORE/RED tail MAE to
  1,418.5/1,331.8 cycles while all stripped base outputs remain identical.
- Next prospective discriminator: identify finite L1TEX admission capacity and
  causal issue backpressure across STORE/RED/ATOMG burst, topology, sparse-SMSP,
  post-memory sentinel, and CTA-transfer axes. Do not promote the retrospective
  10/16-cycle line rates as completion events.
- Complete prospective discriminator: 100 cells, 384 NCU profiles, 16
  SourceCounters reports, and 100 NVBit traces identify STG/RED operand-read
  barriers, cold-preconditioned fence pressure by depth 2, STORE/RED issue
  pressure by depth 8/16, and an exact L2 reuse split. This is not an identified
  queue capacity. Frozen STORE/RED rates fail unseen reuse by 11,944/25,573
  cycles.
- Complete bounded candidate: schema 9 evaluates 16 SM/per-SMSP finite queues
  over exact per-instruction L2 hit/miss outcomes in 100 deterministic pairs.
  Shallow calibration does not transfer to deep burst; cumulative open-loop
  delay is rejected. The capacity-independent hit/miss service frontier lowers
  STORE/RED GPC MAE to 990.6/1,308.6 but remains observational.
- Complete feedback component: schema 10 parks full-queue warps, yields for
  exact L2 pricing, and reschedules later issue/cache order. Across the strict
  20-cell selection holdout, last-issue MAE drops from 9,762/17,010 to about
  3,446/3,442 cycles and terminal MAE to 1,337/1,970. Exact RED cache outcomes
  do not change; aggregate Store-source changes are later instrumentation-store
  effects. Deep burst still misses by 23–28k cycles.
- Complete rate/width discriminator: an explicit fixed miss-sector quantum is
  backward-compatible at width zero. STORE rate 10 and RED rates 12/16 survive
  shallow calibration, but rate 12 RED fails holdout, widths four/eight fail
  topology, and width two does not improve both issue and completion. No default
  changes.
- Complete write-class completion skeleton: schema 11 makes GPU MEMBAR wait on
  the issuing warp's service-done frontier and exposes separate CLOCK issue/fence
  phases. STORE fence holdout MAE falls to 496 cycles; RED improves but retains
  3,872-cycle MAE because RMW completion is absent.
- Complete fixed-tail discriminator: schema 12 adds non-occupying
  operation-specific completion latency. STORE zero dominates; RED latency
  improves fence only by degrading GPC, so no tail is promoted.
- Complete work/parallelism discriminator: schema 13 schedules exact L2-request
  or unique-256-byte-region work on one/two/four independent completion lanes.
  The only central-label joint improvement is 54 fence cycles and 3.5 GPC
  cycles and fails repeat-sensitivity transfer, so no basis, rate, or width is
  promoted. STORE remains schema 11.
- Next engineering milestone: profile event throughput and optimize the
  dominant simulator paths while requiring byte-identical modeled results.
- Complete event-cost milestone: 12 pinned-CPU cases now report wall time,
  memory, exact event composition, and decode/replay phases. Static CLOCK/MEMBAR
  classification plus transaction-plan feedback improves schema 10/11/13 by
  1.06×/1.39×/1.42× without changing JSON. Heap pop is not universally
  dominant.
- Next memory/engine milestone: batch or merge the 99.96%-same-cycle deep-write
  event stream to reduce the measured 695,520-entry queue while preserving
  exact `(time, insertion-sequence)` order.
- Complete same-cycle batching: narrow same-SM reply-less STORE/RED batches cut
  deep-write queue depth 96.8%, wall time about 10%, and RSS about 7%, with a
  0.5% maximum cohort regression and byte-identical outputs. Feedback remains
  on individual events.
- Next trace milestone: reduce full-snapshot decode/storage lifetime on the
  571 MiB gap-heavy case; preserve exact wire-admission failure precedence.
- Complete compact-lifetime milestone: direct replay releases raw threadblock
  bytes only after the unchanged full validation/compaction pass. Peak RSS falls
  5.2–29.3% across the cohort with 0.34% maximum timing regression; raw and
  campaign loaders remain unchanged.
- Optional next trace discriminator: stream direct validation one payload at a
  time only after proving exact structural/resource/semantic error precedence.
- Complete trace discriminator: exact-generation streaming is correct but
  improves gap-heavy RSS only 4.6% and remains opt-in. Hoisting validation-
  proven identity to each warp reduces `ValidatedInstruction` from 32 to 20
  bytes and all 12 cohort RSS values by 7.8–22.4% with byte-identical JSON.
- Complete atomic-network discriminator: a fixed-allocation 220-triplet and
  183-pair campaign identifies maximum per-LTS RED-sector occupancy as the
  exact local timing classifier. The locally exact address rule and all six
  ordinary hashes fail anchor collision transfer, so cross-SM atomic timing
  remains inactive.

### 3. SM and memory fidelity

- Validate compiler dependence counters, CTA barriers, replay, async-copy,
  atomic, and tensor paths. Do not introduce a generic register scoreboard for
  the modern NVIDIA path.
- Complete causal readiness attribution before calibrating issue ports or dual
  issue so dependency stalls and scheduler competition cannot be conflated.
- Complete: frozen CPU replay rejects the simplified LDGSTS front-end
  composite; no selector or timing state remains active.
- Next discriminator: a prospective barrier-free depth-1/4/5/6/8 LDGSTS burst
  crossed with SMSP placement and source route, holding destination work fixed,
  to distinguish acceptance release from route-sensitive backpressure.
- Calibrate L1/L2/DRAM geometry and latency from targeted microbenchmarks.
- Keep hypotheses exposed as configurable parameters until evidence supports a
  default.

### 4. Evaluation

- Join simulator outputs and NCU observations by explicit experiment keys.
- Report MAE, RMSE, MAPE where defined, per-workload residuals, and coverage.
- Investigate residual structure rather than fitting per-workload constants.
- Target cycle MAPE at or below 10%; report the achieved value honestly.

### 5. Performance

- Benchmark events per second, trace decode throughput, and memory use.
- Profile hot paths and reduce allocation, dispatch, and queue overhead.
- Preserve numerical behavior with focused regression tests.

### 6. Blackwell

- Capture `sm_120` traces and microbenchmarks directly.
- Add control decoding, cache policy, latency, and scheduling behavior from
  Blackwell-specific evidence.
- Reuse generic algorithms, not Ampere parameter values.

## Definition of useful progress

A change counts when it implements simulator behavior, enables an experiment,
fixes a demonstrated defect, improves measured speed, or explains a scientific
result. Producing more workflow documentation about whether work may proceed
does not count.
