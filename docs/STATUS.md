# Current status

The repository has functional Rust libraries for event scheduling, enhanced
NVBit trace decoding, warp/SM replay, compiler dependence controls, CTA
barriers, an SM-shared CTA-barrier arrival service, asynchronous-copy groups,
tensor initiation, a sectored L1/shared L2/DRAM hierarchy, transactional
same-stream L2 sessions, architecture-card loading, direct replay, and
numerical comparison with live NCU observations.

The 2026-08-17 RTX 3090 Ti campaign added exact cache-all LDGSTS destination
projection and an independently measured aggregate per-SM destination service
rate. The six-workload shared-L2 replay now has exact instruction cardinality,
stable modeled traffic, deterministic output, and 33.128% MAPE against median
`sm__cycles_active.max`. This is a live residual to explain, not a final cycle
accuracy claim.

## Current GPU-only live baseline (2026-08-29)

The active focus is now the event-driven GPU simulator on the local RTX 3090
Ti (`sm_86`); JADE/HETR and CPU-GPU co-simulation remain deferred. A fresh
six-cell cohort (cp.async canary, BF16/FP16 HMMA, INT8 IMMA, sparse BF16 HMMA,
and BF16 forward attention) has completed separate correctness, NCU, NVBit,
static-control, and release `sim-run` stages. Portable traces have one exact
accepted launch, full CTA coverage, exact reported-instruction counts, and
deterministic simulator twins. NCU observations remain held-out comparison
targets and are not simulator inputs. The machine-readable handoff is
[the 2026-08-29 SM86 baseline](../results/captures/sm86_simulator_live_baseline_20260829_v2/README.md).

The first portable NVBit invocation exposed an activation-contract trap:
`ACTIVE_FROM_START=0` can exit 0 while accepting zero launches when a binary
does not emit profiler markers. The canonical cohort command therefore uses
`ACTIVE_FROM_START=1`; zero-launch artifacts are rejected. The retained active
replay corpus initially reported 5,519 timestamp-stale outputs. Its simulator
matrices are now regenerated with the current release binary and twin-byte
checks; the authoritative validator reports 5,519 fresh, schema-matched,
byte-identical active pairs plus 2,070 known supplemental full-campaign
candidate directories. Fresh live residuals remain typed lower bounds with
dirty-writeback and coherence tails, so this round records a no-go for a
scalar timing correction. The closure report preserves independent
SM-completion, GPC-completion, and GPC-launch-envelope gaps and selects the
next discriminator by typed mechanism-local precedence;
the next discriminator is a store-free/barrier-free async-source pair with an
independent retirement observable. The durable per-cell manifest, compressed
raw evidence bundle, and 7,765-launch semantic replay report are linked from
the baseline handoff; the exact prospective follow-up contract is pinned in
`plans/sm86_ldgsts_burst_release_next_discriminator.json`; the validator now checks trace-derived instruction and
sector conservation plus readiness/frontier invariants for the active and
supplemental replay sets.

## SM86 LDGSTS burst-release discriminator (2026-08-30)

The standalone 40-cell barrier-free LDGSTS campaign has now run its repaired
build/static, correctness, seven-repetition direct, and 120-profile trusted-NCU
stages. The retained reduction is
[the v4 live report](../results/captures/sm86_ldgsts_burst_release_20260830_v4/analysis/README.md).
The gapped target was tightened to one inline-PTX copy/clock region so static
issue/gap/drain wait masks are zero and normal achieved short/long gaps land in
the frozen windows. A few raw scheduler pauses and three NCU LTS conservation
deviations remain explicitly retained, not clipped.

The measured issue frontier is flat at zero gap (`K=0` for L1, L2, and DRAM;
`R=0` across depths), while the independent post-issue completion frontier is
strongly route/depth ordered (L2-minus-L1 rises from roughly 179 to 267 cycles).
The separate LDG control has no positive knee (`P=4`, `S=4`, `K=0`), so the
frozen identification gate is `invalid_clock_assay`/no-model rather than a
timing candidate. The external fixed-clock lock was unavailable because
non-interactive sudo requires a password. The full 16,384-burst NVBit capture
timed out before a complete trace, while a marker-scoped 64-burst NCU-mode
smoke produced a complete trace and two byte-identical simulator replays;
those are pipeline-only evidence, not matrix timing results. No simulator
timing state was changed. JADE/HETR and CPU-GPU co-simulation remain deferred.

The 2026-08-17
[causal-readiness milestone](sm86_causal_readiness_milestone.md) is complete.
Reason-tagged dependency, async-copy, CTA-generation/arrival-spread, and
eligible-not-selected diagnostics are now part of direct replay. A 36-cell live
factorial (112 NCU profiles, seven SourceCounters profiles, and 36 independent
NVBit captures) falsified a fixed CTA-barrier latency: barrier-only cost and
the async/barrier interaction both change strongly with CTA width. No timing
constant was promoted, and all six production predictions remain byte-for-byte
unchanged at 33.403% active-SM MAPE.

The
[sm_86 barrier pipeline and launch-chain cache handoff milestone](sm86_barrier_pipeline_cache_handoff_milestone.md)
is complete. `StreamL2ReplaySession` transactionally carries only fully
drained L2 replacement/sector state across the exact launch order of one
traced device/stream. The direct runner exposes this as opt-in
`--cache-scope stream-l2`; the default remains cold per launch. Synthetic
coverage includes warm hits, partial sectors, replacement recency, rollback,
typed binding failures, counter reconciliation, and determinism.

A live two-launch RTX 3090 Ti NVBit capture then replayed a 2 MiB prewarm and
the unchanged W4 c10 target. Retaining L2 converted exactly 1,024 target
sectors from misses to hits, reduced target DRAM reads from 1,024 sectors to
zero, left target L1 counters unchanged, and moved terminal modeled time from
35,010 to 23,554 cycles. Both scopes replay byte-identically across repeats,
and all 72 saved outputs from the predecessor's 36 cold-launch cells remain
byte-for-byte unchanged. The live record is under
`results/captures/sm86_launch_chain_cache_20260817_v1`.

The 33-cell barrier-only campaign contains 99 NCU profiles, six
SourceCounters profiles, and 33 NVBit captures. W1-W8 follow an exact
+2-cycle-per-added-warp law. One SM-shared arrival service at two cycles per
warp predicts all ten W1-W16 widths with 0.103 cycles/generation MAE and the
five held-out widths with 0.091 MAE. Readiness schema 2 keeps SMSP dispatch,
service completion, CTA arrival, and other-participant rendezvous separate.
The b20 width slope transfers, but a constant 19.016-cycle residual is
confounded with four ptxas-scheduled ALU/control instructions, so no re-arm or
fixed release latency was added.

The coupled campaign adds 62 NCU profiles and 18 launch-chain NVBit captures.
Every L2 seed converts exactly 1,024 target sectors from miss to hit and removes
1,024 DRAM reads. Live L1 < L2 < DRAM ordering holds in all six W2/W4/W8
protocol groups; warm replay is nondecreasing in all six and strict in five.
The six-cell multi-CTA campaign holds eight warps and 96 KiB/SM constant while
varying 1×8, 2×4, and 4×2 CTA shapes. Its live and simulated shape ordering
matches with 0.032 cycles/generation slope MAE and a retained common
4.004-cycle offset. All 36 predecessor traces keep identical traffic; five of
six production predictions are unchanged, attention improves by eight cycles,
and production MAPE is 33.394%.

The
[sm_86 execution-port and control-pipeline milestone](sm86_issue_port_control_pipeline_milestone.md)
is complete. Its 68-cell campaign contains 204 NCU profiles, 14
SourceCounters profiles, 68 NVBit captures, and 5,136 decoded instructions per
binary. Compiler controls already predict dependent IADD/IMAD/LOP3 length
slopes. No profile shows an issue ratio above 1.0, and simple ALU/FMA or
carry-only initiation timelines fail cross-protocol and multi-CTA transfer, so
neither scalar initiation nor dual issue is promoted.

One exact resource survives: a predicated forward `BRA` with encoded stall 5
and yield set has a ten-cycle minimum next-issue frontier. Readiness schema 3
reports branch issue/resolution and limiting/exposure cycles. Branch slope MAE
falls from 5.033 to 0.033 cycles, b11 skew MAE falls from 16.228 to 5.718
cycles/generation, all cache/traffic counters remain unchanged, and production
MAPE improves to 33.128%.

The
[sm_86 SMSP scheduler and math-pipe contention milestone](sm86_scheduler_math_pipe_contention_milestone.md)
is complete. Its 99-cell campaign contains 693 correctness executions, 297
NCU profiles, 21 SourceCounters profiles, 99 NVBit captures, and 3,322,744
replayed warp instructions. Every profile has issue ratio 1.0 and no cell
exceeds 2% active-cycle CV. At W12/W16, active time is within 1.26
cycles/generation of the two-cycle executed-ALU service bound and within 4.41
cycles of the bare-IMAD bound.

Ten strict-port, rational, FIFO/oldest, bounded-queue, and decoupled-service
candidates were tested. The best reduces W8 same-family MAE from 29.14 to 2.28
cycles/generation, but changes the W4 no-contention relation, overprices W12,
and reduces reversed-family sign transfer from 13/18 to 10/18. No scheduler or
port relation is promoted. Candidate removal is verified by 198 restored
outputs byte-identical to the original baseline; production remains at 33.128%
MAPE and readiness stays schemas 1–3.

The
[sm_86 fixed-latency Control/Allocate and register-file contention milestone](sm86_control_allocate_register_file_milestone.md)
is complete. Its trusted SASS projection covers the retained corpus with zero
gaps. The first 28-cell exact-bank campaign measures
7.00/14.00/21.56/34.83 cycles/generation of W4/W8/W12/W16 bank cost.

The follow-on 32-cell completion-distance campaign adds 224 correctness
executions, 96 base-clock NCU profiles, 12 SourceCounters reports, 32 NVBit
captures, and 408,320 replay records. Its matched pair changes only two
source-register bytes per hot IMAD while holding values, destinations, carried
RAW edges, controls, reuse=0, and resources equal. W4 bank3 deltas across RAW
distance 1/2/4/8 are -0.03/2.00/8.00/7.00 cycles/generation; W16 deltas are
44.77/13.65/30.00/34.00. Maximum NCU CV is 1.58% and all issue ratios are 1.0.

Five staged-read/completion candidates were replayed over all cells. The best
has 2.906 cycles/generation treatment-delta MAE, but misses W8 distance-4 by
9.13 cycles and changes the prior 99-cell scheduler MAE only from 37.62 to
37.22. No timing state is promoted. Candidate removal is verified by 64
restored outputs byte-identical to production; the six-workload baseline stays
33.128% MAPE.

The compiler-managed RFC slice of
[sm_86 reuse cache and result/writeback queue](sm86_reuse_result_queue_milestone.md)
is complete. Its 40-cell exact campaign contains 280 correctness executions,
120 NCU profiles, 16 SourceCounters reports, 40 NVBit captures, and 518,400
replay records; maximum CV is 0.645%. The u10 no-hit control is exactly zero
across widths, while paired hits, retention, and source-position movement have
stable signed effects. Untagged and per-warp candidates fail decisively,
supporting shared bank/source-position entries tagged by warp/register.

The best active RFC composition reaches 1.451 cycles/generation treatment MAE,
but misses completion-distance width transfer and leaves the 99-cell scheduler
at 37.331 MAE. No timing is promoted. All 80 restored RFC outputs are
byte-identical to production, which remains at 33.128% MAPE.

Eight slices of
[sm_86 fixed-latency result, bypass, and writeback queue](sm86_result_writeback_queue_milestone.md)
are complete. Exact projection includes HADD2. The initial eight-cell pacing
pilot retained its negative scheduler/pipe crossover but did not identify a
queue.

The balanced opposite-bank order campaign then removed that confound: 14 cells
produced 98 correctness executions, 110 NCU profiles, eight SourceCounters
profiles, 14 NVBit captures, and 77,900 replay records. HADD2→FFMA coincident
completion versus FFMA→HADD2 separated completion is null at W4/W8 and nearly
null at W12. A W13-W16 adaptive sweep shows that the W16 slow regime requires
all four modeled scheduler partitions to be fully occupied; it is a global
phase sentinel, not a local queue-depth boundary.

The compiler-native same-bank pair has identical 16-register resources,
source-bank demands, controls, and timed consumers, with destination stream
R4/R8 in bank 0. Its eight cells add 56 correctness executions, 56 NCU
profiles, eight SourceCounters profiles, eight NVBit captures, and 37,920
replay records. Same-bank coincident completion is null to within 0.001
cycles/generation at W4/W8, identifying timing-neutral accommodation/bypass of
two same-cycle fixed results. Held-out W12 adds 0.742 bank-specific
cycles/generation beyond the opposite-bank pair, but queue occupancy and bank
retirement remain confounded. No result timing is promoted. The retained
projection corpus now has 56 sidecars, 39,848 records, 16,597 projections, and
zero gaps.

The capacity feasibility slice adds nine cells, 63 correctness executions, 63
NCU profiles, six SourceCounters profiles, nine NVBit captures, and 61,269
replay records. W1/W4 are null; W8 has `b8−b7=+0.164` and a dominant
`b9−b8=+1.126` cycles/generation while production predicts `−1.969` for the
b9 step. An experimental capacity-8/pop-1 model changes none of 18 outputs and
is removed; all restored outputs match production. Queue size remains unknown.
The retained corpus now has 64 sidecars, 40,424 records, 16,805 projections,
and zero gaps.

The fixed/L1TEX priority slice adds 20 cells, 140 correctness executions, 60
NCU profiles, eight SourceCounters profiles, 20 NVBit captures, and 88,396
replay records. Every profile has exactly 4,096 timed L1 hits and eight setup
misses per warp. W1/W4 same-bank collision-minus-control is exactly +1.000
driver cycles/generation; the complete R11→R14 opposite-bank control is exactly
zero. Replay now routes sm_86 `LDG.E.STRONG.SM` through L1 and promotes a narrow
SMSP/bank-local HADD2-over-load writeback rule with readiness schema 4. Raw
same-pair delta MAE across W1/W4/W8/W12/W16 improves from 1.058 to 0.305
cycles/generation. All 234 predecessor and 12 production outputs remain
byte-identical, so production remains at 33.128% MAPE. W8+ retains a separate
generic producer-shift phase; queue capacity is still unknown.

The bypass-frontier slice reproduces the fixed-versus-variable availability
boundary directly. Seven fixed distance-3 runs fail their checksum while
distance 4 is valid. Seven fresh global-address distance-4 contexts raise
`CUDA_ERROR_ILLEGAL_ADDRESS`; distance 5 passes seven trials. Its 20 valid
cells add 140 correctness executions, 60 NCU profiles, eight SourceCounters
profiles, 20 NVBit captures, and 83,558 replay records. W1/W4 d4/d5 and d5/d6
controls are null within 0.001 driver cycles/generation; W8+ returns to the
generic shift phase. Exact `IMAD.IADD` projection is promoted, expanding the
retained corpus to 87 sidecars, 42,256 records, and 17,078 projections with
zero gaps. No timing scoreboard is added for deliberately invalid schedules.

The dense producer-shift campaign adds 52 W4–W16 cells, 364 correctness
executions, 160 NCU profiles, 32 SourceCounters profiles, 52 NVBit captures,
and 264,940 replay records. The effect begins at W5 with only one two-warp
SMSP, not at the all-partition W8 boundary, and its sign changes between fixed
and global loops across later occupancy cohorts. Ten reference-inspired
one-fetch/decode-per-cycle IBuffer models were scored. Capacity four with
continue-through-branch fetch reduces driver delta MAE from 2.115 to 0.904 and
NCU-active MAE from 2.161 to 0.799, but fails fixed W9–W12 and global W13–W16
transfer. No front-end timing is promoted. The retained corpus now contains 95
sidecars, 43,088 records, and 17,182 projections with zero gaps.

The branch-frequency/static-footprint successor adds 96 U1/U2/U4/U8 ×
W4/W5/W8/W9/W13/W16 cells, 672 correctness executions, 320 general NCU
profiles, 96 direct front-end NCU profiles, 32 SourceCounters profiles, 96
NVBit captures, and 381,480 replay records. Eight paired cells were expanded
to seven NCU repeats; maximum CV is 4.090% because of discrete high-cycle
modes, while robust MAD stays below 1.889%. Forty spill-free cubins decode all
4,400 controls and project 1,190 fixed operands with zero gaps.

Direct counters scale to exactly 1,024/U uniform branch targets per warp and
measure 3.013–3.167 branch-resolving warp-cycles/target (median 3.052). Replay
now carries a distinct typed three-cycle uniform-backward-target capability;
its diagnostic count is exactly `64/U × warps`, and the existing five-cycle
BRA control stall remains the binding issue frontier, so no extra timing is
invented. A 24-candidate fetch/decode study selects stop-at-branch and three
resolution cycles with 0.0188 branch-counter MAE, but its best IBuffer capacity
is one rather than the paper/reference value three and still misses the large
U1 fixed phases. IBuffer capacity and fetch selection therefore remain
inactive. An interleaved 200-sample-per-arm host A/B finds no replay-time
regression. All six production predictions are unchanged at 33.128% MAPE. The
retained corpus now contains 135 sidecars, 47,488 records, and 18,372
projections with zero gaps.

The paper-style branch-free IBuffer successor is also complete. Its 32-cell
CLOCK timeline campaign contains 224 correctness executions, 144 NCU profiles,
seven SourceCounters profiles, 32 NVBit captures, 21,358 trace records, and 32
byte-identical replay pairs. The measured body is exactly eight groups of one
CS2R plus three NOPs; every legal pair changes one control byte. CLOCK timelines
are bit-stable in every cell except W5, whose maximum seven-trial range is six
cycles. Maximum NCU CV is 3.249% while robust MAD stays below 0.909%.

Correct in-flight fetch reservation changes the candidate result: both the
treatment-delta and absolute CLOCK objectives select capacity two, while the
paper/reference capacity-three/prefetch-eight shape is worse. The surviving
candidate is still rejected. Its treatment optimum requires a 40-cycle miss
and depth-four prefetch, its absolute optimum requires an 8-cycle miss and no
prefetch, and exact CLOCK event-order accuracy is at most 14.8%. No IBuffer or
instruction-cache timing is promoted. The retained corpus now has 153
sidecars, 48,784 records, and 18,426 projections with zero gaps.

The returning-atomic/RED successor is complete. Its 32 cells contain 224 long
and 224 trace-length correctness executions, 96 NCU profiles, eight
SourceCounters profiles, 32 NVBit captures, 101,587 trace records, and 32
byte-identical replay pairs. Exact NCU counts show that 32 lanes aliasing one
word generate 32 serialized atomic sectors, while 32 distinct words inside one
line generate four. Replay previously collapsed the aliasing case to one unique
sector; `sim-trace` now retains exact-address multiplicity and matches all 32
cells. Within one warp instruction, distinct-line and same-word arms are
timing-equivalent when both carry 32 sectors; later cross-SM work scopes that
null to intra-warp formation rather than a universal same-address rule.

Returning ATOMG stalls at its dependent IADD3 write-barrier consumer. RED's
sector-heavy stall is instead on the loop branch waiting for RED's read barrier.
The original offline sector server is now superseded and rejected by two
adversarial successors. The 138-cell runtime-address alias/width campaign shows
that identical sector counts can carry different L2 request counts. It promotes
exact compact L2 request topology and CLOCK-bounded issue diagnostics. A local
RED request+sector law is nearly exact, but the old operation-only sector model
rises to 14.903% MAPE and 45.635% maximum error.

The 106-cell multi-SM successor uses 1–168 one-warp CTAs and a 27-point spacing
sweep. B84 shared-word versus 4096-byte placement differs by 7.36× for ATOMG
and 17.15× for RED despite identical counts. Exact NVBit placement uses one CTA
on each of 84 SMs and two waves at B168. The data bounds about 23.52 effective
atomic units, but all 72 retained cache-hash candidates fail (best 42.1%
training and 73.7% transfer MAPE). A per-SM queue, ordinary L2 slice hash,
portable 128-byte rule, and GA102 page-offset penalty are rejected. No atomic
timing is promoted. A final 544-cell/3,808-execution pairwise oracle identifies
one exact topology rule with zero errors: cross-warp/CTA atomics contend iff
their words occupy the same aligned 256-byte region. This timing-free sm_86
capability is promoted; the many-region scheduling network remains unknown.
The retained corpus now has 181 sidecars, 49,968 records, and 18,562 projections
with zero gaps.

The
[sm_86 launch-cycle, completion, and range-semantics milestone](sm86_launch_cycle_semantics_milestone.md)
is complete. Its 80 cells contain 560 correct ordinary executions, 240
per-kernel NCU profiles (504 kernel rows), 96 Range Replay profiles, 48
Application Range Replay profiles, and 96 adversarial range profiles with the
final sync moved outside, plus 80 NVBit captures with 168 launches and
20,861,740 records, and 160 deterministic ordinary/carded replay pairs.

All seven GA102 GPC elapsed-counter instances observe the same temporal
envelope to within at most 17 cycles. Control `GPC elapsed - max SM active`
stays within 6 cycles while work grows from zero to 4,096 iterations, but the
envelope depends on grid size. Final store and RED tails add median 649 and
592 post-active GPC cycles that replay currently marks as unmodeled. A
discarded final load proves the existing terminal-event boundary is
directionally correct: it adds the modeled 420-cycle response beyond last
issue, but still misses request-service behavior.

`sim-sm` now exposes a typed `ModeledCompletionFrontier`: complete frontiers
carry a cycle value, while incomplete frontiers carry only a lower bound plus
the exact unmodeled tail flags. Direct JSON preserves last issue, the final
processed shared-system event, the raw maximum, and this status separately.
Launch aggregation retains one typed diagnostic per launch and still invents
no scalar time. Raw terminal plus a matched control envelope reaches only
30.880% single-launch GPC MAPE, so numeric emission remains unavailable.

Multi-launch sum/max hypotheses are rejected. Sum of ordinary per-kernel GPC
values has 81.161% MAPE against Range Replay and 97.280% against Application
Range Replay because those observables include untraced host/front-end gaps.
Moving the final synchronization outside changes those failures only to
84.473% and 97.220%; corresponding active-SM sums stay within 1.520% and
1.404%. An ordered cache session therefore does not imply a timed workload
range.

The
[sm_86 memory-retirement service milestone](sm86_memory_retirement_service_milestone.md)
is also complete. Its 102-cell depth/lane/layout/reuse/fence matrix contains
306 correct executions, 348 NCU profiles, 12 SourceCounters reports, 102 NVBit
captures with 19,385,695 records, and 204 deterministic ordinary/carded replay
runs. Eleven variable cells were expanded to five repeats and ten persistent
phase cells to seven; maximum retained GPC/SM-active CV is 7.229%/6.506%.

Fixed store and RED completion delays are rejected: B1-trained constants make
held-out MAE worse than the zero-tail baseline and miss by up to 1,069.5 and
3,992 cycles. B84 line streams expose nonlinear backlog and multimodal phases;
word-coalesced and same-word RED cells instead hide most drain during active
execution. Device fences shift backlog into SM-active time, while system
fences add broader work and one hidden store request per CTA.

Readiness schema 6 now reports active scalar-global instruction counts,
alias-aware sectors, ideal L1TEX wavefronts, and issue spans without changing
timing. Instruction/request and sector topology match every memory cell. Ideal
wavefronts match NCU output except B84 depth-64 line-stream store, which emits
1,391 additional internal-replay wavefronts. That measured replay and RED's
many-region phase stay explicit service gaps. Store/RED completion remains
typed incomplete; no campaign-specific tail penalty is active.

The
[sm_86 L1TEX arrival/output-replay milestone](sm86_l1tex_arrival_replay_service_milestone.md)
is complete. Its 75 cells contain 225 correct executions, 341 general NCU
profiles, 12 SourceCounters profiles, 20 focused L2-request profiles, 75 NVBit
captures with 159,553,212 records, and 150 deterministic ordinary/carded
replay runs. Twenty-eight line-stream phase cells remain above 2% CV after
seven repeats; maximum GPC/SM-active CV is 10.517%/8.634%.

At fixed 512 ideal wavefronts per SM, load time improves 2.79× from W1 to W8,
store time worsens 1.28×, and RED is nonmonotonic. CTA partition holds time
nearly stable while output replay changes. Arrival skew makes the rejection
decisive: W8 store falls from 21,873 to 2,047 cycles as extra replay rises from
247 to 1,738 wavefronts. Output replay is a phase response, not a scalar delay.

Schema 6 now also records unique 128-byte lines and exact atomic L2 requests.
Twenty focused NCU profiles match load/store unique lines and RED request
topology exactly. A host-only word-wavefront checksum error was caught during
adversarial reduction, corrected, rebuilt, and all correctness/static analyses
were regenerated; device SASS and measurements were unaffected.

Affine ideal-wavefront/unique-line surfaces fit their narrow store training
slice at 166.9-cycle MAE but fail held-out arrival/concurrency transfer at
4,617.6 MAE and 19,395.6 maximum error. Load held-out MAE is 6,882.5 with
23,632.1 maximum error. No L1TEX service or retirement timing is promoted.

The
[bounded sm_86 streaming memory-service candidate milestone](sm86_streaming_memory_service_candidates_milestone.md)
is complete. `sim-sm` now evaluates at most 16 opt-in SM-shared or SMSP-local
service hypotheses online at exact scalar-global issue timestamps. The state is
fixed-capacity, transactionally preflighted, adds no events, and cannot change
issue scheduling. Direct `l1-l2` replay accepts a strict schema-1 candidate
document and emits configuration plus observed work, busy time, queue exposure,
and maximum completion through readiness schema 7.

All 75 retained cells were replayed in 75 deterministic candidate pairs over
159,553,212 records. Stripping only candidate diagnostics restores every
ordinary output byte-for-byte. An SM-shared 10-cycle/unique-line STORE
hypothesis selected on eight work-scaling cells reaches 1,418.5-cycle MAE over
15 structurally held-out concurrency/CTA/skew/topology cells, versus 15,322.3
for the raw terminal. The corresponding 16-cycle RED hypothesis reaches
1,331.8 over six held-out cells, versus 32,059.7.

These rates remain inactive. The corpus is retrospective; the STORE word arm
still misses by 3,066 cycles; RED line and ideal work are collinear; and
rate-equivalent SMSP candidates fail W1–W4 occupancy transfer by tens of
thousands of cycles. Most measured STORE/RED cost is SM-active, so a
completion-only diagnostic can fit elapsed time while leaving issue
backpressure causally wrong. The evaluator mechanism is promoted, not a timing
constant or completion event.

The
[sm_86 L1TEX admission/backpressure and reuse milestone](sm86_l1tex_admission_backpressure_milestone.md)
is complete. Its prospective 100-cell STORE/RED/ATOMG matrix contains 300
fully checked native correctness executions, 700 independent
cold-preconditioned CLOCK executions, 384 NCU profiles after targeted repeat expansion,
16 SourceCounters profiles, 100 NVBit captures with 406,381,080 records, and
100 deterministic ordinary plus 100 frozen-candidate replay pairs. Four
spill-free functions contain 712 decoded instructions and zero unknown
controls.

Native CLOCK boundaries separate burst issue from a composite
`MEMBAR.SC.GPU; ERRBAR; CCTL.IVALL` fence sequence; that second interval is not
pure drain. STG/RED carry compiler read-dependence barrier 0 and a subsequent
address-register consumer waits on mask bit 0; returning ATOMG instead uses write barrier 2.
SourceCounters place pressure at the long-scoreboard consumer and compiler
ERRBAR/MEMBAR, not `stall_lg`. The original warm depth-4 conclusion was not
stable under a 32 MiB cold precondition: STORE/RED fence pressure appears by
depth 2, STORE issue delay by depth 8, and RED issue delay by depth 16. This is
a footprint/cache-regime transition, not an identified queue capacity. ATOMG
puts nearly all cost on its returning-value issue path.

Reuse is decisive. Line-unique and line-reuse arms preserve instructions,
L1TEX sectors/wavefronts, and LTS requests, but STORE/RED/ATOMG GPC deltas move
12,567→2,305, 25,049→939, and 33,527→11,423 cycles. Shared-L2 replay exactly
separates 172,032 cold miss sectors from the reused arm's 5,376 misses plus
166,656 hits. The frozen STORE/RED candidates therefore fail unseen reuse by
11,944/25,573 cycles and are rejected.

Readiness schema 8 now distinguishes operation-specific scalar-global
operand-read-barrier producers and consumer delay. Direct JSON exposes the
existing per-L2-source sector/hit split. Both are diagnostic-only; timing and
typed incomplete retirement for dynamically effective instructions are
unchanged.

Predicated-off memory records are retained as issued instructions but no
longer create memory traffic, result writeback, or dependency-barrier
producers. This closes the sparse-SMSP schema-8 overcount without changing
dynamically effective STORE/RED timing.

The adversarial integration pass also closes stale-result reuse. Direct JSON
now has root schema 1 plus simulator version, cache scope, carded-DRAM state,
and candidate count. The launch-cycle, retirement, arrival, and admission
campaigns build release `sim-run` before replay; cached outputs must parse,
match the exact root and launch identity plus every embedded candidate
configuration, and strictly postdate the simulator, trace, cards, control
sidecar, and candidate snapshot. NCU resume requires a complete report/export
pair and NVBit resume requires the complete trace/metadata/error-sidecar
bundle. All 3,719 active-chain replay pairs were regenerated and
byte-determinism checked. The final count, dependency freshness, schema/config
identity, and twin bytes are reproducible read-only with:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
.venv/bin/python workloads/research_probes/validate_active_replay_pairs.py
```

The default readiness hot path no longer embeds or copies the 16-entry
candidate arrays; opt-in state is boxed. Candidate replay now rejects sampled
coverage instead of silently dropping results. Dependency origin follows the
latest causal frontier, including pending L1TEX responses that share a barrier
index with later read or write producers.

After repeat expansion, equal-work one-per-SMSP versus two-SMSP placement is
near-invariant: STORE/RED/ATOMG differ by only 254/111/72 GPC cycles. The data
does not identify an SMSP-local input queue; SM-shared and per-SMSP admission
remain competing bounded candidates.

Readiness schema 9 now completes that bounded comparison. Sixteen finite
STORE/RED candidates cross SM/per-SMSP queues and capacities 4/8/16/32 while
one SM-shared FIFO server consumes exact per-instruction L2 hit/miss sectors.
All 100 deterministic candidate pairs preserve base replay. Shallow burst
calibration chooses per-SMSP capacity 16 for STORE and SM capacity 32 for RED
(the RED score ties per-SMSP capacity 8), but held-out deep-burst errors reach
23,166/31,250 cycles. Open-loop per-warp delay accumulation is decisively
invalid at 170,887/355,145-cycle MAE because it does not reschedule demand.

The hit/miss-aware downstream frontier is a useful partial result: STORE/RED
GPC MAE improves to 990.6/1,308.6 cycles from the frozen line-only
1,389.8/2,331.0. It is identical across all capacities and scopes, so no queue
parameter or timing rule advances. Schema 9 and exact matched L2-outcome
diagnostics advance; timing remains observational.

See the completed
[finite L1TEX admission-candidate milestone](sm86_l1tex_finite_admission_candidates_milestone.md)
for the candidate contract, exact reductions, and feedback-coupled successor.

That feedback successor is now complete. Readiness schema 10 runs one frozen
candidate at a time: matching issue reserves finite queue space, yields the SM
for exact L2 pricing, and turns queue release into a real readiness frontier.
An independent adversarial pass corrected the comparison to like-for-like
diagnostic last issue and separated the five calibration cells from 20 strict
selection holdouts. On those holdouts, STORE issue MAE falls from 9,761.5 to
3,446.4 cycles and RED from 17,010.4 to 3,441.6/3,499.5. Terminal GPC MAE falls
from 7,623.9 to 1,336.7 STORE and from 15,703.8 to 2,016.2/1,969.8 RED.

The earlier cache claim was too strong. The aggregate Store-source lane changes
in six STORE and seven/nine RED cells, but the same-candidate RED observer
changes in zero cells. STORE target attribution is unresolved because target
and instrumentation operations are both stores; six negative target-minus-
control hit deltas are not physical target-only counts. Feedback therefore
changes executable downstream instrumentation-store order, not an identified
target-operation cache outcome. Deep burst still misses by 23–28 thousand issue
cycles, so capacities 16/32/8 and the frozen 10/16-cycle miss rates remain
opt-in only. See
[feedback-coupled L1TEX admission](sm86_l1tex_feedback_admission_milestone.md).

The same pass repaired four execution/integration defects: an SM-wide capacity
denial now repicks ready work instead of creating a false SMSP bubble;
`admission_latency_cycles` must equal the active read-barrier latency; queue and
server counter updates are transactional; and the timing-active candidate is
part of `ReplayConfiguration` identity. Exact candidate configuration, launch
identity, strict JSON, and strictly-newer dependencies now govern replay reuse.

The downstream rate/width successor is also complete. The old rational rate
could not express width, so admission candidates now optionally charge fixed
miss-sector quanta while omitted/zero width remains byte-compatible fluid
service. Thirty-six configurations crossed STORE rates 8/10/12/14 and RED
rates 8/12/16/20/24 with widths 1/2/4/8 while keeping queue scope, capacity,
hit rate, and operand latency frozen.

Five shallow cells select STORE rate 10 and a RED Pareto pair at rates 12/16;
all widths transfer to 20 held-out cells. No candidate improves both issue and
downstream service-completion MAE over the predecessor. Width two changes MAE
by only 8–11 issue cycles while slightly worsening completion; widths four/eight
fail topology transfer. RED rate 12 fails holdout and rate 16 returns to the
predecessor. No rate or width becomes an architecture default.

The structural result is stronger: raw modeled terminal equals last issue in
all 40 predecessor holdout rows, while observed issue and GPC labels differ by
2,569.8 STORE and 2,466.6 RED cycles on average. One frontier cannot fit both.
See [feedback miss-service rate and width](sm86_l1tex_feedback_service_rate_width_milestone.md).

The write-class completion successor is complete. Schema 11 leaves admission
release at service start but makes each warp's `MEMBAR.SC.GPU` wait for that
warp's own service-done frontier. Three CLOCK reads now expose issue and fence
phases separately; uncommon phase/completion state stays boxed and schema 10 is
byte compatible when the option is absent.

Across 200 deterministic pairs, strict-holdout STORE fence MAE falls from the
zero-fence baseline 2,860.9 to 496.4 cycles while service-completion GPC MAE
stays 1,124.8 versus the predecessor 1,127.2. RED fence MAE falls from 6,246.4
to 3,872.3 and GPC MAE stays near the predecessor at 1,466.0, but deep/CTA/
spacing residuals remain large because RED RMW completion is still unmodeled.
See [write-class completion feedback](sm86_write_class_completion_feedback_milestone.md).

The operation-specific latency successor is complete. Schema 12 adds a
non-occupying post-service visibility/RMW tail without changing admission
service or queue release. STORE tests 0–2,048 cycles and RED 0–4,096 using a
five-cell fence/GPC Pareto calibration followed by strict holdout transfer.

STORE zero latency dominates every positive tail: its holdout fence/GPC MAE is
496.4/1,124.8 cycles. RED latency trades one target for the other; 3,072 cycles
reduces fence MAE from 3,872.3 to 2,512.8 while increasing GPC MAE from 1,466.0
to 3,534.7. No candidate improves both, so no latency becomes a default. See
[write-completion latency](sm86_write_completion_latency_milestone.md).

The work-dependent RED successor is complete. Schema 13 retains exact L2
request count and unique 256-byte-region count per priced instruction, then
schedules additive work on one to 32 independent completion lanes. Jobs begin
after admission service, update the issuing warp's MEMBAR frontier, and do not
change admission occupancy, queue release, or cache order.

Eighteen candidates cross request rates 2/4/8, region rates 4/8/16, and lane
counts 1/2/4. All advance from the five-cell calibration Pareto frontier to the
strict 20-cell holdout because line-unique calibration cannot distinguish the
two-requests-per-region bases or lane counts. Candidate 214 (four cycles per
request, one lane) is the only central-label joint improvement, moving fence
MAE 3,872.3→3,818.0 and GPC MAE 1,466.0→1,462.6 cycles. The 54/3.5-cycle gain
does not survive the observed repeat-sensitivity labels and is dwarfed by their
multi-thousand-cycle envelopes. Candidate 218 has the lowest fence MAE but
worsens GPC. No basis, rate, or width is promoted. See
[RED RMW completion work](sm86_red_rmw_completion_work_milestone.md).

The simulator-cost successor is complete. Twelve full-coverage retained traces
now have seven-run pinned-CPU wall time, peak RSS, exact event composition,
decode/replay/serialization phases, and sampled `perf` profiles. Optimized
absolute cost spans 0.222–3.131 seconds, 0.571–3.518 million modeled
instructions/s, 0.001–0.813 million events/s, and 41.1–582.7 MiB peak RSS.
Event throughput is workload-specific because one SM wake may batch more than a
thousand warp instructions while memory operations emit hierarchy events.

Profiles reject a queue-first assumption. Ordinary RED spends about 4.6% flat
self time in heap pop, while schema 11/13 spent 18–20% repeatedly parsing static
CLOCKLO/MEMBAR text and feedback replay cloned its boxed queues at every issue
transaction. Static phase classification and a checked four-SMSP reservation
plan remove those costs. Matched schema-10/11/13 speedups are 1.06×/1.39×/1.42×;
ordinary cases show no material median regression and normal JSON remains
byte-identical. See
[simulator event throughput](simulator_event_throughput_milestone.md).

After optimization, the complete Rust workspace, normal and instrumented
feature tests, 60 affected Python tests, and formatting pass. All 5,519 active
replay pairs were regenerated against the final default release binary and pass
freshness, exact schema/configuration identity, deterministic-twin, and byte
validation.

Exact counters expose the next target: ordinary deep RED reaches a 695,520-entry
queue and 99.96% of its processed events immediately follow another event at
the same modeled cycle. The successor should reduce that queue footprint with
deterministic same-time batching or ordered source merging before considering a
general timing wheel.

That bounded queue-footprint successor is complete. The shared driver now puts
consecutive same-SM/same-time reply-less STORE/RED requests from one actor
emission into one physical queue event and executes their existing L2
transactions in exact order. Response-bearing traffic, ATOMG, mixed timestamps,
cross-SM ordering, and timing-active feedback retain individual events. Batch
payloads are owned by live queue events rather than an append-only historical
arena, and each event reserves its represented logical insertion-sequence span.

Logical event counts are unchanged. Deep STORE/RED physical queue events and
maximum depth fall 96.8% (695,520→22,176 entries). Paired STORE/RED wall time
falls 10.9%/10.3% and RSS 9.8%/9.6%. Arrival STORE improves 8.5%/14.0%, reused
STORE 12.9%/19.0%, and RED gap64 1.9%/0.1%. The maximum matched cohort timing
regression is 0.48%; schema-10/11/13 feedback uses zero batches. Every
condition emits byte-identical normal JSON. See
[same-cycle event batching](simulator_same_cycle_event_batch_milestone.md).

The complete Rust workspace, normal and instrumented feature suites,
formatting, and 60 affected Python tests pass. All 5,519 active replay pairs
were regenerated once more against the final batching binary and pass exact
freshness, nested schema/configuration identity, deterministic-twin, byte, and
no-extra-output validation.

The direct trace-lifetime successor is complete. `sim-run` now uses a compact
direct loader that performs the unchanged descriptor-anchored read, wire
preflight, protobuf decode, allocation/semantic validation, and compaction pass,
then releases raw threadblock bytes before replay. Explicit raw-retaining and
campaign snapshot loaders are unchanged, and compact raw access fails with a
specific typed capability error.

Matched raw/compact runs preserve byte-identical JSON. Peak RSS falls in all 12
cases: gap64 571.2→402.6 MiB (-29.5%), ordinary RED 397.2→304.3 (-23.4%),
STORE 391.1→299.8 (-23.3%), ATOMG 346.8→250.4 (-27.8%), and schema 13
332.0→239.3 (-27.9%). Maximum paired cohort timing regression is 0.35%. See
[compact trace lifetime](simulator_compact_trace_lifetime_milestone.md).

The full Rust workspace, normal/feature and campaign-enabled trace suites,
formatting, and 60 affected Python tests pass. All 5,519 active pairs were
regenerated against the final compact-loader binary and pass exact freshness,
nested configuration, deterministic-twin, byte, and no-extra-output
validation.

A requested one-time independent adversarial pass then verified the core event
and loader invariants and repaired the evidence harness, paired estimator,
phase math, nested replay validation, analyzer freshness, surplus-result
detection, batch lifetime, and compact ordered API. The regenerated harness
rejects ordinary binaries whose feature controls are inactive and retains a
decodable canonical result for every case. See
[adversarial hardening](simulator_adversarial_hardening_20260820.md).

The compact-identity successor is complete. A descriptor-relative roster now
supports exact-generation one-file-at-a-time direct validation, but its
three-round pilot reduces the dominant gap-heavy peak by only 4.6%; streaming
is retained as a performance-build research option, not the default.

The actual peak was the validated instruction array. Function identity and
threadblock-warp identity are validation-proven invariants and now live once per
warp. Per-record SM and physical-warp values retain exact signed i32 payloads
plus an explicit presence byte. `ValidatedInstruction` falls from 32 to 20
bytes without changing replay accessors or normal JSON.

Seven-round matched ordinary binaries show RSS reductions in all 12 cases:
7.8–22.4% overall, 403.6→329.4 MiB on gap64, 304.2→259.5 on deep RED, and
239.2–239.4→194.6–194.7 across feedback schemas. Maximum paired timing
regression is 1.98%. See
[compact validated identity](simulator_compact_identity_milestone.md).

The sm_86 many-region atomic successor is complete. Its fixed-allocation driver
accepts arbitrary runtime region lists while holding SASS, 84 one-warp CTAs,
1,024 generations, and a 32 MiB backing allocation constant. The live campaign
contains 220 triplets/1,540 correctness executions, 183 single-pass LTS pair
profiles, 74 full NCU profiles, four SourceCounters reports, 14 NVBit traces,
and 14 byte-identical replay pairs.

Pair NCU maxima recover six exact local LTS-slice classes. Maximum regions on
one LTS perfectly separates all 220 triplets: 160 slice-distinct cells end by
958,949 CLOCK cycles, while 60 slice-colliding cells begin at 2,054,865. A
two-class oracle has 0.623% MAPE. SourceCounters keep the delay on RED's
read-barrier loop branch.

The locally exact low-bit mapping has 100% local balanced accuracy but only
49.1% anchor-transfer balanced accuracy and zero transfer collision recall;
the six ordinary hashes also fail. LTS counters are evaluation labels, not
simulator inputs. The probe and causal diagnosis advance, but no address hash,
counter-fed correction, or atomic timing queue is promoted. See
[many-region atomic/LTS scheduling](sm86_atomic_many_region_lts_milestone.md).

The GA102 address-to-LTS successor is complete. Because retained Nsight Compute
reports expose aggregate LTS reductions but zero per-slice instances, a 168-CTA
weighted oracle encodes six collision answers in one maximum. All-pairs
discovery over 256 cache lines yields 48 exact transitive classes at each of
three allocation prefixes. A dedicated 49-CTA successor then assigns the 48
representatives 0–47 RED generations and identifies one address per launch.

Nine official reports contain 196,096 one-pass NCU kernel actions. The dense
fit covers 65,535 lines and the untouched high holdout covers 32,768 through
region 262,139. Both independent oracles agree on all 4,096 dense overlaps and
all 2,400 available wide overlaps. Two low region bits are invariant, but only
one of three local linear parity relations survives high-address transfer.

Every simulator hash, IPOLY64, and SplitMix64 remains at chance. Even a
65,536-entry fit lookup retains only 13.37% of high-address collision pairs and
has 0.003% direct-label accuracy. This is consistent with hidden physical-page
placement participating in the mapping; virtual trace addresses alone do not
justify a GA102 slice rule. No hash or timing queue is promoted. See
[GA102 address-to-LTS mapping](sm86_atomic_lts_address_mapping_milestone.md).

The CUDA-VMM alias successor is complete. Seven 2 MiB-spaced virtual proxies
map one physical 2 MiB handle. The simultaneous read probe is rejected as a
slice oracle because even a single pointer spreads LTS read sectors across
instances. A sequential RED probe instead uses exactly one proxy per
synchronized kernel and first self-validates a complete 48-class reference
basis in every report.

Across eight physical offsets, 56 reports, and 3,080 RED actions, all 392 alias
observations retain the physical word's class despite a 14 MiB virtual span.
Classes are repeat-stable and change only with physical offset. A 16,432-action
dense successor maps every 128-byte line in the 2 MiB VMM allocation; its
48-way collision partition transfers exactly to the independent `cuMemAlloc`
map up to relabeling.

The extended campaign raises the official VMM total to 104,672 NCU actions.
Each 2 MiB granule in an exhaustive 8 MiB map transfers exactly, but different
granules require different permutations; one allocation-wide phase retains
only 55.51% of collisions. Eight distinct handles then produce eight stable
class maps, four of which merge the canonical representatives from 48 to 32
classes. Allocation ID plus relative offset is not a universal slice key.

The maps reduce to two partition families: standard granules alternate
`[0,1,0,1]`, while eight handles use `[0,1,0,1,1,0,1,0]`. Their arms score
100%/75% collision recall on the pair corpus and both classify all 220 local
triplets, but physical phase is absent from trace state. The two-family model is
retained only as an evaluator.

Compact global-atomic memory records now retain the already-validated
allocation ID in their previously unused route-disjoint projection word, with
checked registry relative-offset accessors and no size/timing change. No hash
or timing queue is promoted. The release corpus was refreshed and all 5,519
active pairs pass freshness, schema/configuration identity, and byte-identical
twin validation. See
[CUDA-VMM alias discriminator](sm86_atomic_vmm_alias_milestone.md).

The retained-data SM86 LDGSTS front-end test is complete. One frozen
LDGSTS-only composition of five effective outstanding positions, 4/2-cycle
local/shared frontiers, and 13/39-cycle boundaries worsens 30-cell transfer
MAE from 43.803 to 49.440 cycles/group and the W1-W8 guard from 14.377 to
15.037. It also fails one of six strict route orders and three of ten corrected
interaction signs. All nine coupled signs pass, the mechanism executes as
specified, and all 50 probe replays preserve traffic; production int8 changes
traffic, so all-replay accounting is not exact.

The candidate is rejected and its CLI, timing state, schema-14 diagnostics,
and derived binaries are retired. Production timing labels are descriptive:
their 32.901% candidate MAPE does not replace the current 33.128% baseline.
The exact result and restricted claim are in
[the LDGSTS front-end decision](../results/captures/sm86_ldgsts_front_end_transfer_20260828_v1/README.md).
A later prospective discriminator should use a barrier-free burst around depth
five to separate route-invariant acceptance release from route-sensitive
backpressure; the rejected composite is not retuned on the consumed labels.

The remaining technical sequence is:

1. add `sm_120` behavior only from architecture-appropriate evidence after the
   documented Ampere transition criterion is met; and
2. revisit atomic timing only with physical granule-phase evidence or a
   separately validated statistical phase model; both absolute VA and universal
   allocation-relative hashing are experimentally rejected.

The previous repository state contained extensive process machinery around
capture and evaluation. That machinery has been removed from the active
workflow. Test failures and measurement variance remain technical findings to
fix or document; neither pauses unrelated work.

The SOL-ExecBench and historical campaign implementations remain present for
future opt-in use, including their environments, datasets, schemas, and legacy
artifact formats. They no longer define the default simulator workflow.
