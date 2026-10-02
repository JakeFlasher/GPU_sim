# SM86 completion frontier and native-dispatch completion program

**Revision:** 2026-10-02
**Branch:** `GPUSim_main`
**Status:** active research plan; A8 is complete; the broad A9 campaign is
paused until this program closes the completion boundary.

## Why this is the next problem

The A8 vertical slice is useful but deliberately incomplete:

- the simulator reaches a modeled issue/terminal event;
- dirty writeback and coherence are marked unknown;
- the simulator upper is therefore open;
- finite interval coverage cannot be scored;
- lower-bound consistency can be perfect without proving an upper; and
- the real SM86 path still has zero native macro dispatches because no complete
  independent witness joins the response-owner and event-queue state.

The goal of this program is **not** to force every row to have a finite
interval. The goal is to earn a finite upper only for rows whose observable
completion obligations are closed, and to keep every other row open or exact
fallback. Native dispatch is a second, stricter outcome: it requires a real
production event-loop witness that includes all future-relevant completion
state.

## Research findings that constrain the design

### 1. The NCU target is a GPC-clock elapsed interval

Nsight Compute describes `gpc__cycles_elapsed.max` as the longest elapsed cycle
count across GPC instances between the metric's start and stop triggers. It is
the appropriate primary elapsed-cycle observation for the kernel, but it is not
itself a direct DRAM-service or cache-writeback counter ([NVIDIA metric
explanation](https://forums.developer.nvidia.com/t/what-is-gpc-cycles-elapsed-max/198128/2)).
The model must therefore name the observer endpoint explicitly rather than
equating a simulator timer, a CUDA event, a host timer, and the NCU value.

### 2. CUDA defines ordering and visibility, not a public per-store drain time

Global memory is persistent device memory, and a kernel's results are made
available through global-memory writes, but CUDA uses a weakly ordered memory
model. Fences and scoped atomics establish visibility/order at defined thread
scopes; they do not expose a proprietary per-line eviction schedule
([CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html),
[memory-fence guidance](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)).
The simulator may model a bounded visibility/service contract only when the
trace and experiment supply the required state. It must not invent a physical
writeback timestamp from a store count.

### 3. Asynchronous copies have explicit completion mechanisms

CUDA's asynchronous global-to-shared copies decouple initiation from
completion; `wait`/barrier operations are the defined point at which the data
may be consumed. The same documentation distinguishes copy initiation from
completion and describes the LDGSTS path relevant to SM86
([asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html),
[Cooperative Groups async copy](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)).
These completion tokens are observable candidates for a finite local tail, but
they do not close global dirty eviction or cross-SM coherence automatically.

### 4. NCU replay settings are part of the observation contract

Application replay requires deterministic application execution. Cache flush
and clock settings affect repeatability; compared runs must keep them stable
unless that setting is the declared intervention
([Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html),
[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
Every completion experiment therefore records target and mechanism metrics,
replay mode, cache policy, clock policy, launch selection, and device identity.

### 5. NVBit exposes dynamic SASS, not hidden cache state

NVBit dynamically instruments precompiled SASS and can inspect instruction-visible
state, addresses, masks, and launch activity, but it cannot by itself reveal
private cache ownership, eviction timing, or an internal coherence transaction
([NVBit paper](https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2019_NVBit.pdf),
[NVBit project](https://github.com/NVlabs/NVBit/blob/master/README.md?plain=1)).
Those quantities must stay black-box or be represented by an independently
validated interval model.

### 6. Validation must be counter-by-counter and mechanism-local

Accel-Sim's published methodology emphasizes broad workload coverage and
counter-by-counter comparison rather than one aggregate score ([Accel-Sim
ISCA 2020](https://ieeexplore.ieee.org/document/9138922)). This program follows
that standard: target-cycle intervals, mechanism counters, tail classifications,
and endpoint semantics are reported separately.

## Completion vocabulary

The simulator and reports must keep these frontiers distinct:

1. `T_issue`: last modeled issue/dispatch event.
2. `T_sm_active`: architecture-scoped SM-active observation frontier.
3. `T_visibility`: completion of defined memory-visibility obligations such as
   a fence, async-copy wait, or kernel boundary.
4. `T_writeback`: completion of dirty cache-sector service, if its queue,
   ownership, capacity, and service interval are modeled.
5. `T_coherence`: completion of cross-SM ownership/invalidation service, if its
   protocol state is known.
6. `T_ncu`: the declared NCU GPC elapsed-cycle observer endpoint.

No scalar is allowed to silently stand in for all six. A finite predicted upper
for `T_ncu` is permitted only if the declared mapping from the modeled frontier
to the observer endpoint is explicit and every active tail has a finite bound.

## Finite-upper admission conditions

A row may report a finite upper only when all of the following are true:

- the trace has complete unsampled launch/block coverage;
- every dynamic store/atomic/reduction has a decoded address, active/predicate
  mask, allocation identity, line/sector identity, and multiplicity;
- cache line generation and temporal/fence order are known wherever ownership
  is used;
- all cross-SM consumers/producers relevant to the line are represented, or a
  declared protocol envelope bounds the unseen participants;
- dirty-sector writeback has a typed queue, finite capacity, service interval,
  and completion event;
- coherence service has a typed ownership/invalidation state, finite capacity,
  service interval, and completion event;
- async-copy groups have source, destination, commit, wait, and completion
  tokens; and
- the event queue, response-owner ledger, write acknowledgements, dirty-sector
  queue, and coherence queue are all drained at the predicted boundary.

If any condition is false, the output must remain open/refused. A measured
NCU value may validate a prediction on a held-out row, but it cannot fill a
missing simulator state field for that same row.

## Stage ledger

| Stage | Focus | Exit evidence | Status |
|---|---|---|---|
| C0 | Endpoint and tail contract | A machine-readable contract separates `T_issue`, `T_visibility`, `T_writeback`, `T_coherence`, and `T_ncu`; finite-upper conditions are schema-checked | **complete** |
| C1 | Live tail microbenchmark matrix | Native correctness, NVBit trace, simulator replay, and repeated NCU observations exist for clean stores, dirty stores, atomics/reductions, cross-SM sharing, fences, and async-copy waits | **active** |
| C2 | Trace sufficiency and independent witness | NVBit-derived tail facts are independently extracted and every missing ownership/order/completion domain produces an explicit refusal | **planned** |
| C3 | Writeback/coherence service model | Model-local queues, capacities, service intervals, and completion tokens produce finite bounds only on closed cases and remain open otherwise | **planned** |
| C4 | Mechanism-local calibration | Service intervals are fitted on a disjoint calibration set using mechanism observations; scored rows never feed their own parameters | **planned** |
| C5 | Held-out finite-upper validation | Held-out NCU target intervals and mechanism counters validate finite rows; open rows are reported separately | **planned** |
| C6 | Production native witness | A real comparator-selected event-loop fixture dispatches one typed macro with tail state; a perturbed witness falls back before mutation | **planned** |
| C7 | Full 15-row campaign | The complete active roster runs only after C5/C6; each row has live correctness, NVBit, replay, prediction, NCU, and finite/open disposition | **planned** |
| C8 | Final claims and cleanup | Documentation, schemas, stale workloads, and remaining falsifiers are synchronized; no unsupported native or hardware claim remains | **planned** |

## C0 — endpoint and tail contract

### Work

- Add a contract document that declares the target endpoint as
  `ncu_grid_completion_semaphore` in GPC cycles and names all model-local
  frontiers separately.
- Define the difference between a kernel-boundary visibility guarantee and a
  completed dirty-eviction/coherence service.
- Add schema fields for `tail_status`, `finite_upper_reason`, and
  `open_dimensions`; keep open intervals first-class.
- Define the mechanism metric set:
  `gpc__cycles_elapsed.max`, `gpu__time_duration`, `sm__cycles_active`,
  `sm__inst_executed.sum`, L2/L1 sector counters, and DRAM read/write sectors
  where the NCU metric scope is valid.

### Falsifiers

- A purported finite upper is emitted while any tail flag or pending token is
  unresolved.
- The prediction/observation contract cannot state whether the endpoint is a
  GPC elapsed trigger, SM active time, or host synchronization.

## C1 — live tail microbenchmark matrix

Use small, medium, and bounded large shapes with the A8 workload launcher. No
case may allocate beyond its declared memory budget. Each case runs native
correctness, a fresh NVBit capture, simulator replay, prediction before NCU,
and five NCU application-replay repetitions.

### Required cases

1. read-only global stream (no dirty output);
2. coalesced global stores with a clean destination;
3. repeated stores that create dirty L2 residency;
4. cross-SM write/read sharing with explicit device-scope fence/atomic;
5. atomic add and reduction/RMW;
6. store plus `__threadfence()` and store plus block/device synchronization;
7. SM86 `cp.async`/LDGSTS producer, wait, shared use, and global store; and
8. the same cases with footprint, CTA count, cache operator, and cache-policy
   interventions.

### Required comparisons

- target GPC cycles and duration;
- DRAM/L2 read/write sectors and L1 request sectors;
- active cycles and executed instructions;
- trace-derived stores, atomics, sectors, fences, async groups, and waits;
- simulator `T_issue`, `T_visibility`, `T_writeback`, `T_coherence`, and
  `T_ncu` projections; and
- repetition dispersion and clock/cache settings.

The first positive finite-upper candidate should be a closed local visibility
case, not a cross-SM dirty-eviction case. Cross-SM/coherence rows remain open
until C3 closes their ownership state.

## C2 — trace sufficiency and independent witness

Extend the ordinary trace analysis (not a hidden execution gate) to produce
typed facts for:

- store/atomic/reduction identity and multiplicity;
- cache operator and memory scope when encoded in SASS;
- allocation, line, sector, and generation identity;
- warp/block/SM placement and temporal instrumentation order;
- fence, barrier, async commit, async wait, and kernel-boundary markers;
- producer/consumer and read-from candidates; and
- pending completion/acknowledgement tokens.

The tail witness must come from a separate extraction pass or independently
constructed fixture. A summary must never certify its own effects. If hardware
does not expose a required field, C2 records the missing domain and refuses a
finite upper rather than guessing it.

## C3 — writeback and coherence service model

Implement model-local typed services only after C2 identifies their inputs:

- `DirtyWritebackQueue`: sector demand, line grouping, capacity, admission,
  service interval, and completion token;
- `CoherenceService`: ownership transition, invalidation/ack multiplicity,
  cross-SM scope, service interval, and completion token; and
- `VisibilityBoundary`: fence/atomic/async-wait semantics with explicit scope.

Each service must expose lower and upper endpoints independently. Unknown
demand, unknown ownership, unknown order, or unknown capacity widens the upper
to open. Queue equations are written down and tested against adversarial
unknown tails before they are used in an observer projection.

The model must not infer writeback traffic from `dram__sectors_write.sum`
alone: that counter is an observation of traffic scope, not a complete
line-owner history. Likewise, a topology summary is not a coherence
read-from proof.

## C4 — mechanism-local calibration

Calibrate service intervals only from disjoint workloads/interventions:

- dependent chains for latency;
- independent streams for throughput;
- footprint/CTA-count interventions for capacity;
- fence/scope interventions for visibility; and
- sharing/owner interventions for coherence.

NCU mechanism values may describe calibration rows but may not be used as
features, implementation-selection criteria, or labels for the same scored
row. The supplied profile remains unvalidated until C4 produces independent
intervals and residual diagnostics.

## C5 — held-out finite-upper validation

Before collecting held-out NCU values, materialize predictions and record the
full source/launch/endpoint contract. Then collect five repetitions with stable
application replay, cache policy, and clock policy. Score:

- finite interval coverage only for finite uppers;
- lower-bound consistency for open rows;
- finite-row point error only when an upper exists;
- target/mechanism counter direction and dispersion; and
- refusal/open rates by tail class.

The stage is a failure if any row is labeled finite merely because its NCU
value is available. A finite upper must be earned from simulator state first.

## C6 — production native witness and dispatch

Only after C2/C3 have a closed case:

1. construct an independent pre/effect/post witness;
2. include writeback/coherence/visibility pending state and continuation;
3. compare it at the correct region-end event identity;
4. pass it through the real reference-feature event loop;
5. count one typed dispatch and resume a detailed suffix; and
6. perturb entry, effect, pending, owner, or completion state and require
   exact fallback before mutation.

The synthetic fixture is not sufficient. A real retained SM86 trace must
produce the positive dispatch or the result remains the explicit negative
boundary.

## C7 — full campaign

Run the 15 active portable-LCD workloads only after a vertical C5/C6 success.
For every row, preserve native correctness, NVBit trace, simulator replay,
pre-observation prediction, five NCU observations, mechanism counters, and
finite/open/refused status. OOM/OOD/timeout rows are recorded and excluded
from numeric accuracy denominators with their reasons; they are never silently
replaced.

## C8 — completion and stop rules

This program is complete only if at least one real row earns a finite upper,
the finite interval survives held-out NCU comparison, and the production path
has either one real tail-aware dispatch plus perturbation fallback or a precise
negative witness result. If no row satisfies those conditions, the supported
claim remains unknown-preserving exact fallback and all open dimensions are
listed explicitly.

No broad campaign result may be called accurate merely because the lower
bound is consistent. No model-local dispatch count is a runtime metric. No
simulator wall-clock or compression claim is made without a separate,
apples-to-apples measurement.

## Immediate next action

Begin C0 by adding the endpoint/tail contract and a machine-readable
completion-frontier report. Then implement the smallest closed visibility
microbenchmark and run it live through NVBit, replay, and NCU before attempting
dirty-writeback or coherence closure.

## Milestone log

### Research intake — complete

The official CUDA and Nsight Compute documentation, NVBit documentation and
paper, and Accel-Sim validation methodology were reviewed and mapped to the
open A8 observations. The result is a strict separation between issue,
visibility, writeback, coherence, and NCU elapsed-cycle frontiers. The broad
15-row campaign is intentionally deferred; no new measurements were collected
in this planning milestone.

### C0 — endpoint and tail contract — complete

The next implementation seam is the machine-readable completion-frontier
contract and report. Its first live falsifier is a small store/fence or
async-copy visibility case whose tail can be independently observed before any
attempt is made to claim dirty-writeback or coherence closure.

Implemented `docs/sm86_completion_frontier_contract_v1.json` and its schema,
plus `tools/report_sm86_completion_frontier.py`. The diagnostic report for the
A8 vertical simulator output identifies one open launch with
`dirty_writeback` and `coherence` dimensions; it does not consume NCU values
and does not treat the finite GPC envelope field as a finite completion upper.
Focused report tests pass (`3 passed`). C1 is now active.
