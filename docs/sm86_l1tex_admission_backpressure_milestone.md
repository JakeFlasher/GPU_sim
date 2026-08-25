# Completed milestone: sm_86 L1TEX admission, operand-read backpressure, and reuse

**Status:** complete 2026-08-18; schema-9 successor evaluated, timing model withheld  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[bounded streaming memory-service candidates](sm86_streaming_memory_service_candidates_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_admission_backpressure_20260818_v1/analysis/README.md`

## Goal

Prospectively test the predecessor's strongest SM-shared STORE/RED line rates,
separate issue admission from final drain, and identify why most measured cost
appears in SM-active time even though the candidate was completion-only.

NVIDIA defines LG throttle as waiting for the local/global L1 instruction queue
to become non-full, drain as post-EXIT outstanding-memory completion, and
L1TEX wavefronts as serialized work required to process a request. Those
definitions require separate admission, service, and retirement observations,
not one elapsed maximum. See the official
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
and [Python report stall-reason reference](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html).

## Probe construction

`workloads/research_probes/l1tex_admission_backpressure` contains spill-free
CONTROL, STORE, RED, and returning-ATOMG template kernels. Every selected warp
records three `CS2R ..., SR_CLOCKLO` boundaries:

1. before the target burst;
2. after every target memory instruction has issued; and
3. after the compiler's `MEMBAR.SC.GPU; ERRBAR; CCTL.IVALL` sequence.

The first interval measures instruction admission and dependency delay. The
second is a matched-control fence-sequence interval: it includes memory-barrier
drain plus ERRBAR and possible state-dependent CCTL invalidation, so it is not
promoted as pure retirement latency. NCU GPC and SM-active counters remain
separate launch-level observations. NCU application replay perturbs in-kernel
CLOCK timing, so seven independent native processes per cell supply CLOCK
labels; NCU supplies counters and stall attribution.

The 100-cell matrix varies:

- burst depth 1/2/4/8/16/32/64/128;
- W1/W2/W4/W8 occupancy;
- 0/1/4/16/64 dependent IMADs between memory instructions;
- unique line, unique word, reused line, reused word, same-word separated
  256-byte regions, and same-word shared 256-byte regions;
- all, one-warp-per-SMSP, two-warps-on-one-SMSP, and two-warps on each of two
  SMSPs; and
- 1×8, 2×4, and 4×2 CTA shapes at 256 selected warp-operations per nominal SM.

Eight active lanes keep the largest allocation bounded while preserving a
two-wavefront/eight-line line topology and one-wavefront/one-line word
topology. Every run validates CLOCK ordering and final STORE/RED/ATOM data.
Correctness-mode runs validate every selected lane/address (exact values except
deliberate same-word STORE races). A separate 32 MiB runtime memset after target
initialization provides the native timing follow-up's cold precondition.

## Complete evidence

- 100 cells and 300 native correct executions;
- 700 independent cold-preconditioned native timing executions;
- 384 NCU profiles after variance-driven five/seven-repeat expansion;
- 16 SourceCounters profiles;
- 100 NVBit captures with 406,381,080 exact records;
- empty unsupported, unmapped, and cross-allocation sidecars in every trace;
- four spill-free functions, 712 decoded instructions, and zero unknown
  controls;
- 100 deterministic ordinary and 100 frozen-candidate replay pairs; and
- zero candidate-induced base-output changes.

Every candidate replay embeds the configuration it evaluated; analysis
requires equality across all cells and against the retained metadata snapshot.
Resume validates parsed root configuration plus simulator, trace, card,
control-sidecar, and candidate modification times rather than trusting file
existence.

The cold native follow-up changed only the host preamble; all four target GPU
SASS functions are byte-identical to the retained NCU/NVBit target. Those
older stages are retained as a separate host-binary stage, not claimed to share
one executable identity. NVBit module origin remains unresolved, so successful
PC/control joins plus exact SASS equality mitigate but do not close binary
provenance.

Seven phase-sensitive cells retain GPC or SM-active CV above 5% after seven
repeats. Maximum CV is 12.563%/10.715%; medians are retained and the phase is
reported rather than erased by further averaging.
Native issue/fence-sequence CV reaches 10.011%/36.156%; the largest fence CVs
occur on small differences and preclude equivalence claims.

## Compiler barrier identifies admission, not retirement

Static SASS provides the key mechanism:

- `STG.E.STRONG.SM` and `RED.E.ADD.STRONG.GPU` carry read-dependence barrier 0;
- the subsequent address-register consumer/post-memory wait uses mask bit 0;
  and
- returning `ATOMG` instead carries write barrier 2 for its returned value.

The read barrier means the instruction's source operands cannot yet be reused:
it is an operand-consumption/admission frontier. SourceCounters show no
`stall_lg` samples. Dominant STORE/RED samples appear as long-scoreboard at the
post-memory wait and MEMBAR at the compiler `ERRBAR`; drain samples are
negligible because the explicit fence precedes EXIT. NCU's multi-pass
application replay collects metric groups on different executions, so
stall-versus-duration associations are replicated cell-level evidence rather
than co-temporal mediation.

`sim-sm` readiness schema 8 now records global LOAD/STORE/ATOMIC/RED operand-
read-barrier producer counts and attributes any resulting consumer delay by
operation. Counts and barrier effects require at least one dynamically
effective lane: a predicated-off memory record still consumes its issue slot
but creates no traffic, writeback, or dependency producer. Active instructions
retain the simulator's existing eight-cycle clear; the new data makes its
missing variable service explicit.

## Cold-preconditioned burst transition

Native matched-control medians show the same transition for STORE and RED:

| tail | depth | issue delta | composite-fence delta |
|---|---:|---:|---:|
| STORE | 2 | -908 | 2,003 |
| STORE | 4 | -884 | 4,380 |
| STORE | 8 | 557 | 5,284 |
| STORE | 16 | 5,046 | 3,570 |
| RED | 2 | -890 | 2,852 |
| RED | 4 | -846 | 6,320 |
| RED | 8 | -770 | 9,440 |
| RED | 16 | 8,529 | 8,566 |

Small negative issue deltas reflect whole-kernel matched-control code-shape
offsets, not negative memory service. Fence-sequence pressure is visible at
depth 2; STORE issue becomes positive at depth 8 and RED at depth 16. The old
warm claim that pressure began only at depth 8 is not stable under cold
preconditioning and is consistent with L2 priming. This transition does not by
itself identify a queue capacity. A fixed instruction latency or
completion-only terminal still cannot represent the observed regimes.

Returning ATOMG behaves differently: depth 32 adds 41,841 issue cycles but only
206 fence-sequence cycles. Its returned-value barrier serializes the loop before the
fence, confirming that ATOMG must not borrow the STG/RED operand-read model.

## Reuse falsifies the frozen line service

At W8 depth 32, line-unique and line-reuse have identical dynamic target
instructions, L1TEX sectors, ideal/output wavefronts, and LTS request counts.
Only inter-instruction address reuse changes:

| tail | unique GPC delta | reused GPC delta | unique L2 miss sectors | reused L2 miss/hit sectors |
|---|---:|---:|---:|---:|
| STORE | 12,567 | 2,305 | 172,032 | 5,376 / 166,656 |
| RED | 25,049 | 939 | 172,032 | 5,376 / 166,656 |
| ATOMG | 33,527 | 11,423 | 172,032 | 5,376 / 166,656 |

A dynamic-line, sector, wavefront, output-replay, or LTS-request server is
therefore insufficient. The existing shared-L2 replay already predicts the
exact cold/tag-hit partition. Direct JSON now exposes its per-source sectors
and hits, so future admission candidates can consume causal hierarchy state
instead of reconstructing it from total-counter subtraction.

The shared-256-byte STORE arm provides a second downstream discriminator:
21,504 dynamic line touches collapse to about 7,084 observed LTS writes. This
is write combining/coalescing, not a trace-topology mismatch.

## Spacing, SMSP placement, and CTA transfer

Adding 64 dependent IMADs between target instructions reduces STORE
issue+fence cost from 19,922 to 3,439 cycles and RED from 35,669 to 24. The
service drains during compute. ATOMG stays at 25,790 because each returned
value remains on the loop dependency path.
Gap IMADs also change the dependent STORE payload, so generic-compression
sensitivity remains a confound rather than a timing-only intervention.

Four selected warps one-per-SMSP and four on two SMSPs preserve total work.
STORE, RED, and ATOMG GPC deltas differ by only 254/111/72 cycles after repeat
expansion. The stabilized data does not identify an SMSP-local admission
scope; SM-shared and per-SMSP input queues remain competing candidates.

At fixed work, the 1×8, 2×4, and 4×2 CTA medians span 530 STORE, 408 RED, and
519 ATOMG GPC cycles. Phase variance makes this descriptive evidence of no
detected large CTA effect, not an equivalence bound.

## Prospective frozen-candidate result

The predecessor's 10-cycle STORE-line and 16-cycle RED-line candidates were
frozen before this corpus and replayed unchanged:

| tail | all-cell GPC MAE | maximum error | worst cell |
|---|---:|---:|---|
| STORE | 1,389.8 | 11,944 | line reuse |
| RED | 2,331.0 | 25,573 | line reuse |

They retain low error on cold unique-line depth/occupancy/CTA arms, but fail
the unseen reuse discriminator. Their earlier aggregate accuracy was driven by
a dominant cold-stream cohort rather than a transferable mechanism. Neither
rate is promoted.

The predecessor calibration could not identify work basis: STORE IDs 2/4/5
and RED IDs 10/12/13/15 tied exactly. Prospective topology gives STORE GPC MAE
1,389.8/1,981.6/1,422.4 and RED 2,331.0/12,658.2/7,087.6/4,663.4. STORE
line-only and hybrid remain close; RED topology favors line work over
ideal/hybrid, but reuse rejects every completion-only interpretation.

## Promotion decision

Promoted without timing changes:

- readiness schema 8 operation-specific operand-read-barrier producer and
  consumer-delay attribution;
- direct L2 per-source sector/hit diagnostics;
- the native CLOCK issue-versus-composite-fence measurement method; and
- the complete prospective corpus and reducer.

Rejected or still inactive:

- completion-only dynamic-line rates;
- a uniquely identified SM or SMSP admission scope;
- a fixed operand-read latency replacement;
- STORE/RED completion events;
- absolute GPC-cycle emission; and
- reinterpretation of persistent phase variance as measurement failure.

Matched target-control deltas remain whole-kernel marginal costs because
template specialization changes SASS/register counts. They do not isolate one
instruction in an otherwise payload-identical binary.

Ordinary STORE/RED completion remains typed incomplete.

## Successor design

The next bounded candidate connects existing causal pieces rather than adding
another scalar tail:

1. the compiler operand-read barrier waits on a bounded input/admission queue,
   with SM-shared and per-SMSP scopes tested as competing candidates;
2. queues feed an SM-shared downstream service;
3. service work distinguishes L2 tag hit from miss using the replay's exact
   per-source outcome;
4. admission and downstream-completion timestamps remain separate; and
5. returning ATOMG stays on its existing write-barrier response path.

Candidate capacities and hit/miss rates remain opt-in. A model may advance only
if one configuration jointly reproduces burst knee, reuse, compute spacing,
sparse SMSP placement, and CTA transfer without changing unrelated replay.

## Successor execution result: finite admission remains observational

The standalone handoff is
[finite L1TEX admission candidates](sm86_l1tex_finite_admission_candidates_milestone.md).

The configured successor is now implemented as readiness schema 9 and replayed
over all 100 cells in 100 deterministic pairs. Sixteen candidates cross
STORE/RED, SM/per-SMSP queue scope, and capacities 4/8/16/32. Each queue holds
waiting warp instructions, releases a slot when the separate SM-shared FIFO
server starts service, and permits same-cycle dequeue/admission. The server
prices exact per-instruction L2 outcomes at one cycle per hit sector and the
predeclared 10/16 cycles per STORE/RED miss sector. Evaluation retains actual
issue and cache order and changes zero stripped base outputs.

Shallow burst calibration selects STORE candidate 7 (per-SMSP, capacity 16)
and RED candidate 12 (SM, capacity 32; tied with per-SMSP capacity 8). Neither
transfers: maximum single-instruction admission-delay errors reach 23,166 and
31,250 cycles on held-out deep bursts. Summing open-loop delay per warp is
decisively invalid, with 170,887/355,145-cycle all-cell MAE, because delayed
demand was not rescheduled.

The L2-aware downstream frontier is nevertheless a useful narrower result. It
reduces all-cell GPC MAE to 990.6 STORE and 1,308.6 RED cycles, versus
1,389.8/2,331.0 for the frozen line-only candidates. That frontier is identical
across every tested capacity and scope, so it cannot identify the admission
queue. Schema 9 and exact matched instruction/hit/miss diagnostics are
promoted; capacity, scope, and timing remain inactive.

The next bounded component is an opt-in single-candidate feedback lane. It must
park a warp when the selected queue is full, let exact L2 outcomes advance the
downstream server, and reschedule later warp arrivals. Any resulting cache-order
change must be explicit rather than continuing to consume frozen outcomes.
Returning ATOMG remains on its separate result-writeback path.

This follows the retained local research notes: finite lower-memory request
queues, operation-specific return behavior, and scheduler/memory feedback are
separate mechanisms. The Ampere tuning guide's unified L1/Texture coalescing
role supplies context but no proprietary queue constant; see the official
[Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/11.0/ampere-tuning-guide/).

No privileged command was required.
