# Historical exact-cycle acceleration plan — 2026-09-12

> **Archive-only notice (2026-09-24).** This plan records an earlier
> acceleration study. Its SM86 event/dense observations remain useful context,
> but AMD/CDNA3, gem5, and any RTL/Verilator or certification language are not
> active requirements. The current completion bar is the software-only SM86
> NCU/profile boundary, one production resource intervention, and replay wall
> time; see `plans/cross_architecture_gpu_simulator_research_plan_20260923.md`.

**Decision.** Continue with an accelerated execution lane, but define its
contract as *model-relative observational equivalence*. The detailed SM86
executor remains the reference implementation. The accelerated lane may omit
host calls only when it can prove that the same modeled state, timestamped
events, ordering, and declared observations will be produced. Any unknown or
unclosed condition reifies the detailed path.

This plan is intentionally SM86-first. AMD CDNA3/gfx942 and gem5 Full System
are later validation targets behind a hardware/capture gate. No MI300X timing
claim is made in the current environment.

**Implementation status (2026-09-18).** The first slice is complete: the
`sim-semantics` crate contains a versioned observation contract, `sim-engine`
reports virtual clock gaps separately from handler events, schedule-derived
causal inputs reject barriers, memory roles and uncharacterized dispatch-only
entries, and the optional `reference` feature provides a dense local-cycle
executor for differential runs. On the SM86 canary, dense and event modes
produced identical serialized output (62 versus 378 local steps; 316 virtual
cycles skipped). The detailed replay remains the default. Runtime typed state
snapshots are now implemented through schema-15 boundaries, while the
contract-aware transformer remains intentionally limited to closed zero-length
identity proofs.

The retained 8-workload SM86 comparison is byte-identical in all 8 cases after
adding the demanded stream-buffer ready time to the wake calculation. After
the bounded-horizon clock fix, event mode performed 1,246,619 local cycle
steps and 219,112 idle jumps covering 3,771,866 virtual cycles; dense mode
performed 5,018,485 local steps. The comparison is automated by
`tools/semantic_events/sm86_reference_compare.py`. These are engineering
observations, not a silicon-speed claim. The retained seven-workload large
cohort also matches 7/7: event mode performed 2,069,235 local steps and
skipped 3,164,825 virtual cycles, while dense mode performed 5,234,060 local
steps. Across all 15 rows, event mode performed 3,315,854 local steps and
skipped 6,936,691 virtual cycles; dense mode performed 10,252,545 local
steps.

The optional `SIM_SM_REFERENCE_TRACE` path now emits post-event event ordering
and a schema-15 typed boundary root. The legacy
`state_projection_complete: false` header remains for old consumers, while the
typed root carries L1/L2 ownership, async groups, queue payloads, pipeline,
front-end, and activity-only power state. Unknown ownership/timing remains
explicitly incomplete and never certifies a summary.
The canary event and dense traces now match for all 13 processed events, in
addition to matching the final replay output.

The SM actor now commits its local clock to a bounded external horizon before
returning. Event mode counts that jump in its diagnostics; the dense lane walks
each locally idle cycle below the horizon and stops before processing the
external event itself. This prevents repeated front-end work after a wake and
keeps the differential oracle aligned with the event engine's ordering.

One bounded host run of the comparator showed median dense/event wall-time
ratios of 1.70x on the small cohort and 1.66x on the large cohort, with a few
workloads below 1x. These single-process timings are directional only; Phase 5
still requires repeated runs and planner-inclusive measurement before any
speedup claim.

## What the adversarial review changed

Three independent reviews covered the semantics, SM86/Accel-Sim integration,
and gem5/gfx942 feasibility. They agreed that the proposal is plausible as an
exact implementation technique, but not as unrestricted event deletion or as a
standalone novelty claim.

The review resolved these points:

* A scalar transition function is too coarse for GPU timing. The model needs a
  labeled transition system with explicit phase and same-time ordering. A skip
  must preserve `(cycle, phase, insertion-order)` behavior, not only endpoint
  cycles.
* Lookahead is global. `EventQueue::peek_time()` or “no ready warp” is not a
  proof when another SM, L2, DRAM channel, DMA path, host launch, retry, or
  periodic callback can arrive sooner. Unknown lookahead means no jump.
  Equal-time event classes and reserved sequence spans also need explicit
  representation; an anchor timestamp/sequence cannot stand in for a batch.
* “State-complete” is relative to a declared observation contract. It must
  include every modeled state that can influence future transitions and every
  metric promised by the paper; it need not claim knowledge of unmodeled
  silicon state.
* The current causal certificate proves only its narrow descriptor recurrence.
  It assumes one homogeneous pipeline, empty entry state, one active occupant,
  producer-supplied dependency/effect facts, and no cache, memory, barrier,
  atomic, or interconnect effects. Its event counts are nominal accounting, not
  measured production queue pops.
* The current `SmCore::advance` path is the correct starting point: it runs
  strictly before a typed horizon, tightens the horizon on reply-bearing
  requests, and parks at the boundary when a response has no computable due
  time. The `l1_miss_latency` bound must remain justified for every request
  class that uses it.
* The causal-frontier summary fails closed with a typed `StateMismatch` error
  whenever its independent exact and summarized model states differ; a
  terminal-state boolean cannot authorize an inaccurate skip.
* Summary eligibility requires an explicit complete resident-warp roster
  through `summarize_with_residency`; the descriptor-only entry point keeps
  residency unknown and reifies candidates instead of inferring SMSP
  exclusivity from the candidate list.
* Accel-Sim/GPGPU-Sim retain detailed cycles because shader scheduling,
  scoreboard countdowns, cache/MSHR queues, DRAM arbitration, power/statistics,
  and deadlock checks evolve even when no warp issues in the current slot.
* gem5’s current MI300X Full System configuration is a useful gfx942 software
  and timing reference, but its GPU configuration is minimally sized and its
  CPU normally uses KVM or Atomic execution. It is not a complete silicon
  model of an eight-XCD, 192 GiB MI300X.
* Generic adaptive fidelity is already occupied by temporal decoupling,
  sampling and clustering systems such as STEM+ROOT. The contribution must
  include GPU-specific state semantics and an architectural decision result,
  rather than only fewer events or a higher sampling rate.

The implementation therefore has four explicit modes:

1. **Detailed:** existing cycle-level SM86 path; the scientific reference.
2. **Exact accelerated:** conservative next-transition advancement and
   certified summaries with detailed fallback.
3. **Approximate sampled:** optional Photon/SIEVE/LPGSim-style sampling, with
   reported error and no exactness claim.
4. **Functional-only:** software/semantic validation without timing claims.

Only modes 1 and 2 may contribute to a “no timing trade-off” result.

## Scientific contract

### Transition and observation model

Represent a simulator state as

```text
S = (architectural state, pipeline state, queue state, resource state,
     pending events, counters, external-input frontier)
```

and each transition as

```text
S --(cycle, phase, sequence, event)--> S'
```

The detailed lane is the reference transition relation. An exact accelerated
operation `G` over an interval `[t, t+k)` is valid only if, for every allowed
external input stream in the declared environment,

```text
G(S_t, inputs[t:t+k)) = S_(t+k)
```

and the operation emits the same timestamped observations and same-time order
as the detailed lane. If the contract promises per-cycle counters, `G` must
emit equivalent samples or exact integrals. If it promises only instruction
completion and memory traffic, those are the required observations and power
state can be outside the contract.

### Exactness levels

Use precise language in code and reports:

* **Cycle-equivalent:** accelerated and detailed lanes have identical declared
  state/observation traces for a fixed simulator model and input stream.
* **Trace-preserving:** the comparison replays a fixed captured path. It does
  not prove that the program would take the same path after an architectural
  intervention.
* **Hardware-validated:** the simulator’s outputs agree with independent
  silicon measurements for a defined mechanism and workload set.

Event acceleration can establish the first property. It does not establish the
third property by itself. Accel-Sim’s trace-driven path is inherently subject
to the second property; gem5 Full System improves software-stack execution but
still has model-specific hardware limitations.

### Event-order contract

The current engine orders events by `(time, seq)`. Treat that as a public
contract until a measured need justifies a compatible extension. Same-cycle
handlers must not observe a later sequence’s state early. If a future component
requires explicit phases, add a backward-compatible `(time, phase, seq)` key or
per-phase queues after differential evidence shows that `(time, seq)` cannot
express the required behavior. Sequence-span reservation is only a safe
optimization inside a closed interval with no external same-cycle producer; it
does not replace phase ordering.

Never compare only final cycle numbers. Compare the event order at every
external interaction and the full boundary state used by the next transition.

## Baselines and evidence to preserve

The current repository already supplies the following seams:

* `sim/engine/src/lib.rs`: deterministic binary-heap event queue and sequence
  allocation.
* `sim/sm/src/issue.rs::SmCore::advance`: bounded issue execution, known-timer
  jump, response horizon and no-progress handling.
* `sim/sm/src/system.rs`: typed `SmWake`, L2 access/fill, memory response,
  atomic completion and write-ack events; per-SM unpriced response bounds.
* `sim/semantics/src/certificate.rs`: narrow interval certificate and explicit
  reification reasons.
* `sim/sm/src/causal_frontier.rs`: independent exact/summary recurrence and
  terminal-state comparison.
* `sim/trace/src/semantic.rs`: SM86 typed event adapter and loss reporting.
* Retained SM86 small/large campaigns and semantic round-trip probes documented
  in `docs/STATUS.md`.

The detailed lane must remain runnable without the accelerated lane. New
instrumentation is additive and must not change existing timing or output
schemas unless a versioned field is required.

## Phase 0 — Freeze the observation contract and baseline (3–5 days)

### Work

1. Write a versioned `ObservationContract` in `sim/semantics` with explicit
   booleans or enums for:
   * instruction issue, writeback and completion order;
   * warp/SM frontiers and launch/retirement envelopes;
   * L1/L2/MSHR/coalescer/DRAM requests and responses;
   * barriers, wait counters, atomics and asynchronous tokens;
   * queue occupancy/port utilization;
   * counters, power samples and statistics included in the study.
2. Add a parameter ledger that labels each timing value `documented`, `measured`,
   `inferred`, or `unknown`. NCU values remain validation observations and are
   never converted into simulator timing constants.
3. Define the closed-world assumption for each run: all host launches, DMA,
   peer traffic, periodic clocks, refreshes and external arrivals must either be
   represented in the engine or declared outside the contract.
4. Record the exact command, trace/control inputs, architecture card, revision,
   and output fields for the current detailed SM86 canaries.

### Completion bar

The contract and ledger are checked into `docs/` or `sim/semantics/`; a reviewer
can tell which observations exactness covers and which are intentionally out of
scope. Existing detailed canary outputs are unchanged.

### Stop condition

If mandatory fields depend on NCU labels, undocumented hardware guesses, or
unavailable producer data, narrow the contract before implementing skips.

## Phase 1 — Instrument the detailed reference lane (1–2 weeks)

### Work

Add low-overhead instrumentation behind an explicit run option. Record:

* event pushes, pops, same-cycle ties, sequence spans and handler phases;
* issue attempts, successful issues, scoreboard waits, port stalls and retries;
* per-component next-transition candidates and the reason each candidate was
  rejected;
* queue/stage occupancy and pending response counts;
* skipped-cycle opportunities observed by the detailed lane, without changing
  execution;
* snapshots at kernel launch, each external event boundary and completion.

Use a structural `BoundaryState` type for development comparisons. Do not use a
cryptographic digest as an authority or acceptance gate. A compact diagnostic
fingerprint may be printed for triage, but mismatch reporting must identify the
actual fields that differ.

### Required state inventory for SM86

The snapshot inventory must cover, at minimum:

* warp PC/cursor, active and predicate masks, register readiness and scoreboard
  counters;
* front-end buffers, instruction-cache/stream state and fetch port frontiers;
* issue arbitration state, round-robin pointers, execution-unit reservations,
  scalar/tensor/memory pipeline occupancy and release times;
* CTA/barrier/branch readiness, wait/reuse/yield controls and reconvergence;
* L1 tags, replacement metadata, dirty/sector masks, MSHRs, retries,
  fill-waiters and coalescer child ownership;
* L2 pending-sector generations, fills, evictions and cross-SM ordering;
* interconnect credits/packets, DRAM queue/bank/row/refresh state where modeled;
* async-copy groups, destination service, atomics, write acknowledgements and
  all response identity sets;
* diagnostics and counters included by `ObservationContract`.

### Completion bar

The detailed lane emits deterministic event and boundary traces for one small
stream, one memory stream, one barrier/async stream and one multi-SM trace. The
instrumentation changes no scored result. Run the existing SM86 small and large
canaries and retain their current outputs.

## Phase 2 — Make conservative next-transition advancement explicit (2–4 weeks)

This phase improves the exact accelerated lane without summarizing active
instruction regions.

### Component API

Introduce a private or crate-level interface with semantics equivalent to:

```rust
enum NextTransition {
    At { time: u64, reason: TransitionReason },
    Quiescent,
    Unknown { reason: UnknownReason },
}
```

and an `advance_before(horizon)` operation that executes all work strictly before
the horizon. Keep the public surface small. Do not add a generic callback or
dynamic event type to the hot path.

Each component may return a future time only after accounting for:

* internal countdowns and resource release times;
* queued messages, retry deadlines and port-free notifications;
* all external channels that can affect the component;
* periodic/statistics callbacks inside the observation contract; and
* same-cycle phase rules.

### SM86 implementation order

1. Reuse `SmCore::advance` and formalize its current `ReplayHorizon` behavior.
2. Make the response lower-bound proof explicit for L1 hits, merged pending
   fills, direct bypasses, atomics and write acknowledgements. Add a differential
   case that would fail if a response can occur before `l1_miss_latency`.
3. Keep `UnpricedBounds` and `response_due` as separate sources. The minimum
   horizon is the minimum across both plus issue, front-end and async timers.
4. At the system level, process every event at the current time before jumping.
   Never advance a core across a response or retry whose timestamp is only a
   lower bound.
5. Add wake hooks for every state mutation: memory responses, L2 fills, port
   free, async destination service, barrier updates, write acknowledgements,
   and host/launch events.
6. Continue ticking a component when a countdown, rate gate, arbitration pointer,
   power/statistics sample or queue transition changes. Sleeping is allowed only
   when the next wake is represented.

### Completion bar

For the closed SM86 component model, detailed and accelerated lanes produce the
same boundary snapshots, event sequence, frontiers and declared counters on all
Phase 1 streams. The accelerated lane reports `skipped_cycles`, `wake_reason`,
and `unknown_horizon` separately from event counts.

### Stop condition

If event bookkeeping costs as much as the detailed loop on dense workloads, keep
the mechanism as a correctness-preserving option and do not claim a speedup.
If any state mutation lacks a wake source, disable sleeping for that component.

## Phase 3 — State-complete certificates and exact fallback (3–5 weeks)

The existing interval certificate is a useful recurrence test. It is not yet a
native SM certificate. Extend it in layers rather than generalizing every GPU
mechanism at once.

### Certificate structure

Each certificate must identify:

* entry and exit cycle and phase;
* complete predecessor/successor frontiers;
* resident warp population and shared-resource ownership;
* conserved issue demand, reservations, releases and completion multiplicity;
* memory/cache/queue effects represented by the summary;
* barrier, async, atomic and visibility edges;
* counters or integrals updated by the summary;
* the external-input set for which the proof holds;
* a deterministic event-sequence reservation;
* `certify`, `reify`, or `unknown` decision and a precise reason.

The summary is a state transformer. It must return pending completion/release
events, not only a duration. If a cache or queue post-state cannot be projected,
the region is not eligible.

Formally, its input is `(local state, inbound-channel metadata, external-input
lookahead)` and its output is `(post-state, emitted events with cycle/phase/
sequence, observation deltas)`. A single sequence span cannot replace omitted
events that could collide with external events at multiple timestamps; such a
region must remain detailed or encode each externally visible boundary event.

### First eligible region

Start with a single SM86 partition containing a closed, homogeneous compute
pipeline and no memory or control effects. Require a complete resident-warp
set, not merely `active_per_smsp == 1` inferred from the candidate list. Require
that sequence numbers have a documented global admission meaning; otherwise
preserve per-SMSP ordering and refuse cross-SMSP summarization.

The certificate must reject:

* register RAW/WAR/WAW hazards, dynamic predicates and reconvergence;
* barriers, wait counters, yield/reuse controls and async groups;
* memory, atomics, cache/MSHR activity, queue occupancy or external arrivals;
* multiple resident contenders or shared pipeline ownership;
* mixed pipeline/timing parameters or unknown producer fields;
* arithmetic overflow and ambiguous same-cycle ordering.

The current SM86 semantic adapter does not observe complete register values or
cross-lane issue order. Until the native replay exposes a dynamic scoreboard
snapshot, dependency evidence is `unknown` and the corresponding interval is
reified; a missing field must never be interpreted as `dependency_free`.

### Independent checker

Implement a checker that reconstructs the eligible recurrence from the typed
certificate and compares the projected boundary state against a detailed run.
Run it in development and campaign validation; it is not a procedural commit
gate. Deliberately corrupt one field in a certificate and confirm that the
checker identifies the mismatch and reifies the region.

### Completion bar

Zero false certificates over adversarial tests; exact boundary-state equality
for every positive certificate; explicit reification for every negative case.
The native SM path remains unchanged when adaptive mode is disabled.

## Phase 4 — SM86 adversarial validation (2–4 weeks)

### Positive cases

Use the retained traces plus focused synthetic streams for:

* independent scalar/vector arithmetic with long idle gaps;
* homogeneous tensor or scalar issue spans where the full state is closed;
* one SM partition with no incoming work and no external memory effects.

Multiple-completion instructions remain a negative case in this phase: the
current certificate deliberately reifies them. A future summary class may add
completion multiplicity only after its release multiset and ownership state are
represented explicitly.

### Mandatory negative cases

Each case must prove that the selector falls back or parks at the boundary:

* same-cycle release versus issue tie;
* omitted resident warp and cross-SMSP contention;
* register dependency and predication/divergence;
* cache alias, MSHR merge, replacement and out-of-order child responses;
* queue full, port retry and backpressure;
* DRAM bank/row/refresh or unpriced response;
* barriers, wait counters, atomics and async-copy/TMA-like producer/consumer;
* host launch, DMA, TLB/page walk, interrupt or periodic callback;
* sequence ties that are not global admission order.

### Comparison outputs

Compare detailed and accelerated runs at every external boundary:

* issue, modeled-event, memory-return and completion frontiers separately;
* per-instruction issue/writeback/retirement where available;
* resource reservations and release multiplicities;
* cache/MSHR/L2/DRAM traffic and queue occupancy;
* barrier/async/atomic tokens and visibility order;
* declared counters and time-series samples;
* terminal state and deterministic output ordering.

### Completion bar

All retained SM86 canaries and adversarial streams pass with zero declared
observation mismatches. Any mismatch disables the affected certificate class
until the state projection or guard is corrected.

## Phase 5 — Measure whether acceleration is real (1–2 weeks)

Measure the full path, including instrumentation and planning:

```text
wall_time = input + planning + execution + output
```

Report event pushes/pops, actual component ticks, skipped virtual cycles,
certificate attempts, positive certificates, reifications, unknown horizons,
peak memory and wall time. Compare against the unchanged detailed lane on the
same host and build.

Partition results by event density and workload class. Dense HMMA/IMMA and
memory-contended kernels may have no profitable skip; long sparse waits may.
Do not aggregate mechanism-local frontiers into one score. Report speed and
fidelity by mechanism.

### Speedup bar

Claim a speedup only when the accelerated lane is faster after planning and
output costs on a held-out workload partition. A nominal event reduction or a
faster synthetic recurrence is not sufficient. If no positive speedup appears,
retain the implementation as a correctness and observability contribution and
state that result.

### Architecture-decision bar

Run controlled SM86 design sweeps over issue width, L1/MSHR capacity, memory
latency and warp occupancy. Verify that detailed and accelerated lanes preserve
the same ranking. A ranking change is a scientific failure until explained,
even if endpoint cycles happen to match on the original configuration.

## Phase 6 — Accel-Sim interoperability (3–6 weeks after SM86 proof)

Do not fork Accel-Sim’s detailed core first. Use its current cycle path as an
independent baseline and add an adapter at the trace/semantic boundary.

### Adapter

1. Import a compatible Accel-Sim/GPGPU-Sim trace or produce the existing SM86
   semantic IR from the same SASS capture.
2. Pin the Accel-Sim framework at `d930ad6d02c09bb56867132583735aba0389cff4` and its integrated `accel-sim/gpgpu-sim_distribution` tree at `91880c53383d5a6a6742bfb1be2c5f34e39c7871`; retain vanilla public GPGPU-Sim `a4ce3feac901c97a4b4601f679e43cf3589c79de` only as a separately named comparison. The framework helper script defaults to mutable branches and must not define the experiment identity.
3. Preserve one-to-one native instruction identity, active masks, registers,
   addresses, memory roles and replay markers.
   Do not use the semantic adapter’s serialization ordinal as a global issue
   order; cross-warp scheduling must come from dynamic scheduler state.
4. Mark TMA prefetch, unsupported mbarrier/SYNCS variants, unknown opcodes and
   missing control state as `unknown`; force detailed execution.
5. Keep fixed-trace results separate from claims about regenerated program
   control flow. Re-capture or functionally execute when an intervention can
   change addresses, predicates, atomics or synchronization.

The current SM86 semantic adapter does not observe all register values,
cross-lane issue order or silicon completion timestamps. Static constant-cache
records can also lack dynamic memory-class metadata. These are explicit losses,
not evidence that a dependency-free or cache-free interval is safe; a future
certificate must obtain the missing native replay state or reify the record.

### Comparison

Run the same trace through Accel-Sim’s detailed cycle loop and the project’s
semantic/detailed/accelerated lanes where the models expose comparable events.
Compare relative timing and mechanism counters, not raw cycles from different
architectural cards without a mapping.

### Completion bar

The adapter has no silent semantic losses on the chosen canaries, and all
unsupported features force detailed mode. The accelerated lane matches the
project’s detailed lane on declared observations. Accel-Sim is an independent
baseline, not a source of hidden timing constants.

## Phase 7 — Deferred gem5 gfx942/MI300X lane (only after hardware or captures)

The current gem5 documentation lists MI300X/gfx942 Full System support, but
requires an x86 `VEGA_X86` build, `GPU_VIPER`, a matching ROCm disk image, and
KVM or Atomic CPU execution. The supplied MI300X configuration is minimally
sized and is explicitly not intended to predict real hardware measurements.
KVM host execution may also use a coarse synchronization quantum, so an
end-to-end CPU/GPU timing claim needs a pinned host setup or must be limited to
GPU-model virtual time. Treat gem5 as a software-stack and model reference.

### Gate 7A — access and reproducibility

Do not start timing claims until one of these is available:

* an accessible MI300X host with pinned GPU identity, driver, ROCm version and
  device index; or
* retained HSACO/ROCr/gem5 captures with exact tool, kernel, card and workload
  metadata.

Before timing work, build the official gem5 Full System example and run a
small `gfx942` HIP kernel. Record kernel, disk-image, ROCm, gem5 revision,
configuration, CPU mode and simulated topology. Checkpoint only outside a GPU
kernel because gem5’s current documentation does not support in-kernel GPU
checkpoints.

### Gate 7B — functional semantic bridge

Add a CDNA3/gfx942 front end that emits the same typed event vocabulary while
retaining AMD identity. Start with wave64 vector operations, LDS/DS, MFMA,
wait counters, barriers, atomics, memory scopes and queue/dispatch events.
Unknown HSACO or runtime semantics remain explicit losses; they cannot become
generic scalar operations.

Use gem5 for functional driver/OS execution and event identity. Do not infer
undocumented MI300X latency, HBM arbitration, XCD placement or fabric policy
from the configuration script. Keep one-XCD/minimal-device results labeled as
such.

### Gate 7C — one-XCD timing reference

Instrument gem5’s Compute Unit and all delayed event sources before adding a
skip path. The inventory includes wave status, instruction/fetch buffers,
register and scoreboard readiness, WaitClass deadlines, sleep counters,
barriers, LDS returns, data/scalar ports, retries, TLB/SQC responses,
`ScheduleAdd`, dispatcher/WG completion and periodic statistics.

Implement detailed and smart-tick modes side by side. A CU may sleep only when
all internal and external wake sources have a known lower bound and every
declared per-cycle statistic has an exact update. Wake on every response,
retry, port-free, scheduled counter mutation, dispatch and WG completion.

### Gate 7D — topology and hardware validation

Only after one-XCD semantics and timing are stable, add eight-XCD/HBM/fabric
placement and collective traffic. Separate documented topology from measured
latency. Validate vector/LDS/MFMA, HBM locality, atomics, barriers and
collectives against independent hardware or retained captures. Until then,
report functional coverage and model uncertainty, not “cycle-accurate MI300X.”

## Phase 8 — Novelty and paper evaluation

### Baselines

Compare against:

* unchanged detailed SM86 project path;
* exact accelerated path with no region summaries;
* certified summaries plus fallback;
* approximate sampling/metadata compression where scope permits;
* Accel-Sim/GPGPU-Sim detailed execution;
* MGPUSim/Akita smart ticking as event-driven prior art;
* gem5 gfx942 detailed/smart-tick path once Gate 7 is open.

The paper must acknowledge that event-driven GPU simulation, smart ticking,
kernel/warp sampling, partial-order reduction, and variable-fidelity models all
predate this project. MGPUSim’s “no busy ticking,” NVArchSim’s event-based
components, ACALSim fast-forwarding, and recent Akita Smart Ticking make a
generic event queue an occupied contribution space.

### Required ablations

1. No conservative horizon: quantify stale-response errors.
2. No full resident-warp guard: expose contention errors.
3. No cache/MSHR state in the certificate: expose alias/merge errors.
4. No same-cycle sequence preservation: expose tie-order errors.
5. No planner-inclusive timing: show why nominal event reduction is inadequate.
6. No semantic loss reporting: show unsupported-feature misclassification.
7. Detailed versus approximate sampling: separate exactness from speed.

### Minimum publishable result

The project needs all of the following for a strong architecture submission:

* a formal GPU-specific state and wake contract;
* zero false positive certificates on adversarial tests;
* positive certificates across more than one mechanism;
* measured end-to-end speedup after planning overhead, or a documented result
  that the exact lane is neutral/slower and why;
* preserved per-mechanism observations and architecture rankings;
* independent SM86 validation against retained NCU/trace evidence;
* a nontrivial design conclusion enabled by the method;
* for MI300X, a separately validated gfx942/software-stack result rather than
  a guessed topology or latency table.

If certificate coverage is tiny or the speedup is negative, publish the work as
an exact observability and validation study or narrow the claim. Do not expand
the summary model merely to increase a headline skip percentage.

## Immediate next actions

1. Add the observation contract and parameter ledger without changing replay.
2. Instrument the detailed SM86 path and produce one boundary-state trace.
3. Prove the existing horizon assumptions with L1-hit, merged-fill, bypass,
   atomic and write-ack cases.
4. Add structural detailed-versus-accelerated comparison for one closed SM86
   partition.
5. Implement one certificate class with complete resident-warp and event-order
   guards; keep detailed fallback as the default.
6. Run the positive/negative matrix and measure planner-inclusive wall time.
7. Only if the SM86 result is exact and useful, begin the Accel-Sim adapter.
8. Keep MI300X/gem5 work gated until hardware or retained captures arrive.

## Primary references

1. gem5, “Full System AMD GPU model,” current documentation. It lists gfx942
   MI300X support, the VEGA_X86/GPU_VIPER requirements, KVM/Atomic CPU modes,
   and the minimally sized, non-hardware-measurement configuration caveat.
   [https://www.gem5.org/documentation/general_docs/gpu_models/gpufs](https://www.gem5.org/documentation/general_docs/gpu_models/gpufs)
2. Y. Sun et al., “MGPUSim: Enabling Multi-GPU Performance Modeling and
   Optimization,” ISCA 2019. Event-driven engine, conservative ordering and no
   busy ticking.
   [https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf)
3. Akita authors, “Akita: A Versatile Framework for Architectural Simulation,”
   arXiv 2604.28073. Smart Ticking and conservative wake-up behavior.
   [https://arxiv.org/abs/2604.28073](https://arxiv.org/abs/2604.28073)
4. Accel-Sim framework, current source and trace documentation.
   [https://github.com/accel-sim/accel-sim-framework](https://github.com/accel-sim/accel-sim-framework)
5. GPGPU-Sim distribution, current cycle-level source.
   [https://github.com/gpgpu-sim/gpgpu-sim_distribution](https://github.com/gpgpu-sim/gpgpu-sim_distribution)
6. O. Villa et al., “Need for Speed: Experiences Building a Trustworthy
   System-Level GPU Simulator,” HPCA 2021 (NVArchSim). Selective fidelity and
   event-based component modeling.
   [https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2021_NVArchSim.pdf](https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2021_NVArchSim.pdf)
7. ACALSim, arXiv 2605.22936. Event fast-forwarding with explicit separation of
   framework speed from timing-model accuracy.
   [https://arxiv.org/abs/2605.22936](https://arxiv.org/abs/2605.22936)
8. Photon authors, “Photon: A Fine-grained Sampled Simulation Methodology for
   GPU Workloads,” MICRO 2023.
   [https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf](https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf)
9. Local evidence: `/home/jakeshea/GPU_prototypes`, commit `40a331d`; current
   rebase files `sim/engine/src/lib.rs`, `sim/sm/src/issue.rs`,
   `sim/sm/src/system.rs`, `sim/semantics/src/certificate.rs`,
   `sim/sm/src/causal_frontier.rs`, and `sim/trace/src/semantic.rs`.
10. **STEM+ROOT**, MICRO 2025. Hierarchical clustering and execution-time
    distributions for large-scale GPU simulation; an approximate adaptive
    acceleration baseline. [https://seonjinna.github.io/assets/pdf/MICRO25_stem_root.pdf](https://seonjinna.github.io/assets/pdf/MICRO25_stem_root.pdf)
11. **Accel-Sim project**, current release overview. Modern H100/H200 features,
    GPUVision time-series counters and published validation claims that define
    the current comparison baseline. [https://accel-sim.github.io/](https://accel-sim.github.io/)
