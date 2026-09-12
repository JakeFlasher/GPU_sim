# Typed L2 ownership and power state for certified summary regions

## Decision

The remaining failure is a state-schema failure, not an L2 replacement-policy
failure or a missing scalar latency. The reference trace currently records a
single formatted L2 string and omits the requester-owned state that connects a
shared L2 allocation to an SM response, pending-sector merge, eviction, or
stale fill. It also has no simulator power state at all. A summary transformer
cannot be certified from that representation because it cannot prove that the
post-summary L2 ownership graph, pending response identities, and power
integration state are the same as the detailed lane.

The implementation should therefore proceed in this order:

1. Freeze a contract-relative typed boundary schema.
2. Replace the L2 debug string with a typed shared-cache snapshot plus a
   canonical requester/allocation ownership graph.
3. Add a fixed-clock, activity-based power state transformer. Keep hardware
   power measurements as independent validation data; do not turn NVML/NCU
   values into simulator timing or power coefficients.
4. Make every summary a transaction over typed entry state, typed emitted
   events, typed exit state, and observation deltas. Reify whenever any
   ownership, address domain, queue, or power premise is unknown.
5. Re-run dense/event differential traces and enable certified regions only
   after the typed checker produces zero false certificates on adversarial
   state transitions.

This resolves the stated incompleteness without claiming that a model-relative
power estimate is silicon truth. It also preserves a useful distinction:
power may be omitted from a performance-only observation contract, but it may
not be silently omitted from a contract that promises power or a complete
power-aware state transformer.

## Implementation status

The first executable slice is now present. `sim-semantics` contains versioned
typed L2, ownership, queue-payload, contract-completeness, and fixed-clock
activity-power records. `sim-mem` exports deterministic set/way/generation
projections while keeping the process-local L2 authority token private, and
exports a typed L1 fill/generation projection. `sim-sm` joins canonical
response owners and pending-fill ready times, emits typed pending events, and
records activity-only power at every reference-trace boundary. The typed root
record now assembles L1/L2, front-end, pipeline, async, queue, ownership, and
power state. The SM86 canary produces identical typed event records in event
and dense modes across all 13 events; the V2 typed root record is complete at
quiescent canary boundaries and explicitly reifies while an outstanding
response has no exact due time. The legacy V1
`state_projection_complete` flag remains false for compatibility with existing
diagnostic consumers; it is not the V2 typed-root completeness result.

The bounded executable check used the retained address-carry canary with
`--mode l1-l2`, `SIM_SM_REFERENCE_TRACE`, and both
`SIM_SM_REFERENCE_MODE=event` and `dense`. It produced 13 event records,
byte-identical replay JSON, and identical typed L2/power/queue records; the
terminal modeled event time was 785 cycles. Workspace tests passed after the
change, including 447 `sim-sm` tests in the reference-enabled build and the
full workspace suite.

The typed root now reports `typed_event_boundary_v2`. On the canary it returns
`incomplete(request_due_time_unknown)` while response-bearing work is still in
flight and `complete` for the six quiescent post-response boundaries. This is
the intended fail-closed transition rather than a guessed response timestamp.

## Evidence and diagnosis

The current schema is explicitly designed to refuse certification. `BoundarySnapshot`
has string fields for L1, the instruction front end, the memory issue
pipeline, the scalar pipeline, and async-copy slots, and only a free-form
`omitted_state` list can explain what is absent. Its `is_complete()` method is
therefore only an omission-list check, not a proof that the fields can be
transformed ([`sim/semantics/src/boundary.rs`](../sim/semantics/src/boundary.rs#L50-L91)).

The event trace writes `l2_state` using `L2Model::boundary_state()`, which is a
formatted debug string. It does not serialize the per-SM maps for outstanding
responses, fill waiters, fill masks, line folds, or retry state
([`sim/sm/src/system.rs`](../sim/sm/src/system.rs#L925-L953)). The SM projection
explicitly lists `l2_mshr_replacement_and_pending_sector_state` and
`per_cycle_power_state` as omitted ([`sim/sm/src/issue.rs`](../sim/sm/src/issue.rs#L9798-L9840)).

The L2 implementation already has most of the semantic data needed for a typed
projection: set/way tags, valid/dirty/pending masks, recency, allocation
generation, counters, pending-sector count, and transactional snapshots. It
also deliberately maintains a process-local `L2ModelOwner` inside fill tokens
and pending-sector keys ([`sim/mem/src/l2.rs`](../sim/mem/src/l2.rs#L462-L556),
[`sim/mem/src/l2.rs`](../sim/mem/src/l2.rs#L1016-L1256)). That owner is correct as
an authority token, but it is not a stable boundary identity and must never be
serialized as the meaning of the cache state.

The current accuracy contract declares issue, completion, memory transactions,
synchronization, resource state, and counters at per-cycle granularity, while
power is explicitly `Omitted` because no validated power transition system
exists ([`sim/semantics/src/observation.rs`](../sim/semantics/src/observation.rs#L36-L109)).
The existing retained validation is strong for the current differential path:
the status report records 13-event canary agreement and matching L2 strings,
while the semantic and package tests are green. That evidence establishes
deterministic replay and useful diagnostics. It does not establish that the
opaque strings are executable state transformers.

The trace side already provides allocation-relative identity when the
allocation registry is present. It validates allocation ranges and exposes
relative byte and 32-byte-sector offsets without inventing a physical address
([`sim/trace/src/reader.rs`](../sim/trace/src/reader.rs#L7585-L7657)). This is the
right basis for a typed address-domain field, but it is not evidence that a
virtual allocation offset equals a physical L2 slice selection.

The focused local checks confirm the starting point: `sim-semantics` passes 10
tests, `sim-mem` passes its 48 L2-related unit tests, and the focused
`sim-sm` run passes 5 selected tests. The full current package counts recorded
in `docs/STATUS.md` remain the reference campaign evidence. None of these
checks exercises a typed L2 ownership transformer or a power transition.

## Contract that the implementation must satisfy

Certification is relative to a declared observation contract. Define three
explicit modes rather than making one global completeness claim:

| Contract | L2 requirement | Power requirement | Permitted claim |
| --- | --- | --- | --- |
| `BoundaryDiagnosticV2` | Typed snapshot if available; unknown fields remain visible | Typed `omitted` status allowed | Differential diagnosis only |
| `CycleExactMemoryV2` | Complete shared L2, requester ownership, pending events, and address domain for every modeled memory transition | Power is outside the contract and is not used by the transformer | Model-relative issue/completion/memory/synchronization equivalence |
| `CycleExactPowerV2` | Same as `CycleExactMemoryV2` | Fixed-clock activity state, sample schedule, and all power-domain transitions are complete | Model-relative timing plus power-state equivalence; no silicon-power claim |

`BoundarySnapshot::is_complete()` should be replaced by a contract-aware
validation result, for example `Completeness::Complete`,
`Completeness::Incomplete(Vec<UnknownField>)`, or `Completeness::Refused`. A
typed field with `status = omitted` is valid only when the selected contract
omits that observation. A typed field with `status = unknown` must always force
reification.

The contract must also state the address domain. The current SM86 replay is a
fixed captured path and uses the L2 model's line identifiers. The schema should
distinguish:

```text
VirtualAbsoluteLine(line_id)
AllocationRelativeLine(allocation_id, line_offset)
PhysicalLine(physical_line_id)
UnknownAddressDomain
```

`CycleExactMemoryV2` may certify a model-relative replay over
`VirtualAbsoluteLine` if the architecture card explicitly declares that
interpretation. A physical-slice-sensitive summary must require either
`AllocationRelativeLine` plus a declared placement function or
`PhysicalLine`. It must refuse when the allocation registry is absent, an
address spans allocations, or the model would need an unobserved VA-to-PA
mapping. This carries forward the research review’s conclusion that L2 slice
mapping cannot be treated as solved merely because virtual addresses are
stable.

## Typed state schema

Add version 2 types to `sim-semantics` and keep version 1 deserialization only
for old diagnostic traces. Do not encode state with `format!("{state:?}")` or
use pointer/process identities in a serialized boundary.

The root event record should become conceptually:

```rust
pub struct EventBoundaryRecordV2 {
    pub schema_version: u32,
    pub contract: ObservationContract,
    pub event: TypedEventIdentity,
    pub queue: QueueBoundaryState,
    pub cores: Vec<SmBoundaryStateV2>,
    pub shared_l2: Option<L2BoundaryState>,
    pub power: PowerBoundaryState,
    pub completeness: Completeness,
}
```

The existing `BoundarySnapshot` fields should migrate into typed substructures
rather than remaining a collection of optional strings:

```rust
pub struct SmBoundaryStateV2 {
    pub sm_id: u32,
    pub scheduler: SchedulerState,
    pub warps: Vec<WarpBoundaryState>,
    pub blocks: Vec<BlockBoundaryState>,
    pub front_end: FrontEndState,
    pub memory_pipeline: MemoryIssuePipelineState,
    pub scalar_pipeline: ScalarExecutionPipelineState,
    pub async_groups: Vec<AsyncGroupState>,
    pub l1: Option<L1BoundaryState>,
    pub responses: ResponseOwnershipState,
}
```

The first patch can use private-to-public conversion functions and preserve the
old JSON fields as deprecated additive fields. The typed fields must be the
ones consumed by the checker. Once the V2 comparator is stable, remove the
string fields from the certification path while retaining a bounded legacy
diagnostic export if operators still need it.

Every typed state needs three properties:

* deterministic ordering of maps and sets before serialization;
* checked arithmetic and explicit overflow errors;
* a `project()`/`apply()` operation that is independent of debug formatting.

## L2 ownership normalization

### What must be represented

The shared L2 is one device-level state machine. It must not be copied into
each SM snapshot as if every SM owned a private cache. The typed boundary must
contain one `SharedL2State` and explicit requester ownership edges:

```rust
pub struct L2BoundaryState {
    pub geometry: L2GeometryState,
    pub address_domain: AddressDomain,
    pub clock: i64,
    pub bip_counter: u64,
    pub access_discipline: Option<L2AccessDisciplineState>,
    pub counters: L2CounterState,
    pub sets: Vec<L2SetState>,
    pub allocations: Vec<L2AllocationState>,
    pub request_owners: Vec<L2RequestOwner>,
    pub next_generation: GenerationState,
}

pub struct L2SetState {
    pub slice: u32,
    pub set: u64,
    pub ways: Vec<L2WayState>,
}

pub struct L2WayState {
    pub line: CanonicalLineKey,
    pub valid_mask: u8,
    pub dirty_mask: u8,
    pub pending_mask: u8,
    pub recency: i64,
    pub generation: Option<u64>,
}

pub struct L2AllocationState {
    pub allocation: CanonicalAllocationKey,
    pub generation: u64,
    pub sectors: Vec<L2SectorOwnership>,
}

pub struct L2SectorOwnership {
    pub sector: u8,
    pub status: PendingStatus,
    pub producer: Option<L2RequestKey>,
    pub consumers: Vec<L2RequestKey>,
    pub ready_at: Option<u64>,
}

pub struct L2RequestOwner {
    pub request: L2RequestKey,
    pub sm_id: u32,
    pub warp_slot: i32,
    pub instruction: u64,
    pub response: Option<ResponseKey>,
    pub line: CanonicalLineKey,
    pub sector_mask: u8,
    pub state: RequestState,
    pub due: Option<u64>,
}
```

The exact field names can change, but the ownership relations cannot be
dropped. For a pending sector, the cache allocation owns the fill capability;
one request may produce it, and multiple SM requests may consume/merge onto its
ready time. An eviction can make an old fill stale while a new allocation for
the same line gets a new generation. The typed state must preserve all three
facts.

### Canonical identity rules

1. Keep `L2ModelOwner` private. It is an authority token that prevents a fill
   from being applied to the wrong Rust model; it is not semantic state.
2. Preserve the allocation generation and the `next_generation` frontier. If
   the implementation remaps generation IDs, remap deterministically by sorted
   `(canonical line, old generation)` and preserve equality/inequality and ABA
   relationships. Never renumber two live generations into one.
3. Canonicalize response identity from stable simulator facts:
   `(sm_id, warp_slot, instruction_index, sector_or_response_index)` plus the
   response kind. The process-local `ResponseOwner` and pointer/debug text must
   not appear in JSON.
4. Normalize every pending map into sorted records. The state must include the
   current `fill_masks`, `outstanding_replies`, `outstanding_response_ids`,
   `fill_waiters`, `line_folds`, and retry entries held by
   `SharedL2State`. A scalar `outstanding_responses` count is not enough.
5. Label cache-line identity as virtual, allocation-relative, or physical. Do
   not call an allocation-relative offset a physical slice ID. The current
   architecture card’s six-slice geometry can validate shape and capacity, but
   a slice hash remains an architecture hypothesis unless the selected card
   explicitly supplies it.
6. Represent L2 topology and requester ownership separately. A line’s way and
   masks belong to the shared cache; a pending response belongs to the issuing
   SM; a merged request is a second consumer of the same allocation-sector
   owner. This prevents double-counting when several SMs touch one sector.

### Extraction and normalization API

Add a fallible API in `sim-mem`:

```rust
pub fn typed_boundary_state(
    &self,
    address_normalizer: &AddressNormalizer,
) -> Result<L2BoundaryState, L2ProjectionError>;
```

Add a corresponding `SharedL2State::boundary_state()` in `sim-sm` that joins
the L2 model state with per-SM requester maps. The join must be performed at
one system boundary, not by each SM independently. The API should expose an
opaque internal token-to-canonical-key map only long enough to build the typed
record; it must never make token construction public.

Add `L2BoundaryState::validate()` with these invariants:

* every way is in the set selected by its canonical line and declared hash;
* masks are disjoint where the model requires them (`pending & valid == 0` for
  read fills; dirty is a subset of valid or the explicitly documented write
  state);
* every pending sector has exactly one allocation generation;
* every producer/consumer request references a live allocation or is marked
  stale with the matching old generation;
* every response owner has one and only one owning SM;
* all pending event due times are represented in the queue projection;
* L2 counters and source splits still satisfy the existing checked
  conservation identities;
* no process-local owner, address, pointer, or unordered map order leaks into
  the serialized state.

## Power state

### Scope and non-leakage rule

The first implementation should be a fixed-clock, activity-based state model.
It is sufficient to make a power-aware summary transformer exact relative to
the simulator, while avoiding an unsupported claim about watts. It should not
consume the `power_w_median` values in the pointer-chase card or any scored NCU
row as a simulator parameter. Those values remain independent operating-point
observations.

The power schema should make the distinction explicit:

```rust
pub enum PowerModelStatus {
    Omitted,
    ActivityOnly,
    AnalyticCoefficients,
    Unknown,
}

pub struct PowerBoundaryState {
    pub schema_version: u32,
    pub status: PowerModelStatus,
    pub clock: PowerClockState,
    pub sample_period: Option<u64>,
    pub next_sample: Option<u64>,
    pub domains: Vec<PowerDomainState>,
    pub pending_transitions: Vec<PowerTransition>,
    pub cumulative_activity: ActivityTotals,
    pub samples: Vec<PowerSampleState>,
    pub omitted_observations: Vec<String>,
    pub provenance: PowerProvenance,
}
```

Each domain needs enough state to integrate exact activity over a skipped
interval:

* fixed frequency/clock-domain identity and whether DVFS is modeled;
* enabled/clock-gated/idle state;
* active-cycle and occupancy integrals;
* issue, completion, L1/L2 lookup, L2 fill, DRAM request, barrier, async, and
  queue-transition counts that the selected power model consumes;
* the last integration cycle and the next sampling deadline;
* any coefficient-set identity only when coefficients are an explicit,
  architecture-scoped simulator input.

`PowerState::advance_to(t)` must integrate the current domain state up to `t`
before applying an event at `t`. `PowerState::apply(delta)` then records the
typed activity caused by the event. A summary transformer must either emit the
same sample boundaries and exact integrals or refuse. It must never infer a
power sample from the terminal cycle alone.

Do not add DVFS transitions in this slice. The architecture cards already
distinguish a locked SM clock and NVML operating-point observations. A future
DVFS model needs typed frequency transitions, thermal state, and a wake/event
source; until then, any run that requests DVFS-aware power must return
`UnknownPowerTransition` and use the detailed path.

The recommended first useful power mode is `ActivityOnly`: it proves that the
summary preserves the explicitly modeled event-system activity integrals while
reporting omitted observations such as watts and per-instruction energy.
The second mode, `AnalyticCoefficients`, can be added after an independent
microbenchmark campaign establishes coefficients. The AccelWattch design is a
useful precedent: it separates constant, static, and dynamic power and couples
dynamic power to component activity; it does not treat a single measured power
number as hidden simulator state.[^1]

## Typed transformer architecture

The current interval certificate only transforms a narrow recurrence over warp
readiness, dispatch readiness, pipeline capacity, and completion times. It must
remain available for that narrow model, but it must not be upgraded by adding
more booleans. Add a typed transformer layer:

```rust
pub struct TransformerInput<'a> {
    pub entry: &'a BoundaryStateV2,
    pub external_frontier: &'a ExternalInputFrontier,
    pub contract: ObservationContract,
    pub interval: Interval,
}

pub struct TransformerOutput {
    pub exit: BoundaryStateV2,
    pub emitted_events: Vec<TypedEventTransition>,
    pub observation_delta: ObservationDelta,
    pub proof: SummaryProof,
}

pub enum TransformerDecision {
    Certify(TransformerOutput),
    Reify(ReifyReason),
}

pub trait StateTransformer {
    fn decide(&self, input: TransformerInput<'_>) -> Result<TransformerDecision, TransformError>;
}
```

The implementation should use staged copies or component transactions. Apply
the SM, L1, L2, queue, and power transforms to staged state; validate all
cross-component edges; commit only when every component succeeds. The existing
fallible `L2Model::try_snapshot()` is the correct pattern for cache staging.

The proof object must contain:

* canonical typed entry and exit state snapshots produced by deterministic
  serialization;
* the exact interval and same-time phase/sequence reservations;
* all emitted L2 allocation, merge, eviction, response, and stale-fill edges;
* power integration intervals and sample boundaries when power is declared;
* the external-input lookahead and closed-world declaration;
* the observation deltas covered by the selected contract;
* a precise refusal reason whenever any field is missing or unknown.

The proof is data consumed by the checker, not a process gate. A checker must
be able to recompute the typed exit state by applying the proof to the entry
state and compare it with a detailed replay at every external boundary.

## Implementation sequence

### Phase 0: freeze the contract and inventory (1–2 days)

Change:

* `sim/semantics/src/observation.rs`: add contract version 2, address-domain
  and power status enums, and contract-aware completeness validation.
* `sim/semantics/src/boundary.rs`: add `BoundarySnapshotV2`, typed event and
  queue state, and `UnknownField`/`Completeness` types. Keep V1 for reading
  existing diagnostic files.
* `docs/STATUS.md` and the active carry-on plan: state exactly which contracts
  omit power and which require it.

Acceptance:

* a serialized V2 header names the contract, schema, address domain, and power
  status;
* V1 traces still compare byte-for-byte through the existing path;
* `is_complete` cannot return true merely because an omission string is empty;
* no NCU or NVML observation is accepted as a simulator input through the new
  types.

### Phase 1: typed projection adapters (3–5 days)

Change:

* Convert the current L1 `boundary_state()` string into `L1BoundaryState`.
  Preserve sorted line stamps, sector masks, line/sector fills, generations,
  timing discipline, MSHR occupancy, and counters.
* Convert front-end, memory-issue, scalar-pipeline, and async debug strings
  into typed structs with explicit enum variants for unsupported states.
* Add typed event payloads to `sim-engine` or to the `sim-sm` trace adapter so
  pending-event payloads do not depend on `Debug` output.

Acceptance:

* V1 and V2 projections from the same canary compare equal after canonical
  normalization;
* map insertion order, process allocation order, and debug formatting do not
  change V2 JSON;
* malformed or unsupported state returns a typed error and keeps
  `Completeness::Incomplete`.

### Phase 2: shared L2 state and ownership (1–2 weeks)

Change:

* `sim/mem/src/l2.rs`: add typed extraction of geometry, occupied sets, ways,
  generations, counters, pending masks, and allocation discipline. Keep
  `L2ModelOwner` private. Add canonical generation normalization and checked
  validation.
* `sim/sm/src/issue.rs`: expose a typed snapshot of `SharedL2State`’s
  outstanding response IDs, fill masks, waiters, line folds, and retry state.
  Convert private response-owner identities into canonical `(sm, warp,
  instruction, sector)` keys at the projection boundary.
* `sim/sm/src/system.rs`: join one shared L2 snapshot with all SM requester
  snapshots and emit a single `L2BoundaryState` per event. Do not duplicate
  shared cache ways into per-core snapshots.
* `sim/trace` integration: select `AllocationRelativeLine` only when the
  allocation registry proves range and sector alignment; otherwise preserve
  `VirtualAbsoluteLine` and record the limitation.

Acceptance:

* cold line allocation, resident sector fill, pending merge, write-class tag
  hit, dirty eviction, same-tag reallocation, and stale completion each produce
  a typed state that round-trips through JSON;
* two SMs touching one pending allocation produce one producer and multiple
  consumers, never two physical L2 lines or two DRAM fills;
* an evicted generation’s fill is marked stale and cannot mutate the new
  generation;
* cloned/snapshotted models have equal typed state even though their private
  `L2ModelOwner` values differ;
* an unknown physical mapping refuses a physical-slice certificate but still
  permits the explicitly declared virtual replay diagnostic.

### Phase 3: fixed-clock power state (4–7 days)

Change:

* `sim/semantics/src/power.rs`: add typed power domains, activity deltas,
  sample boundaries, fixed-clock state, and provenance.
* `sim/sm/src/system.rs` and `sim/sm/src/issue.rs`: update power state at the
  same event boundaries that mutate issue, cache, queue, response, barrier,
  and async state. Integrate idle/gated intervals before each mutation.
* `sim/arch/src/card.rs`: do not add power coefficients in this phase. If a
  later card adds coefficients, make them a separate, architecture-bound,
  optional `PowerModelCard` with explicit provenance and validation.

Acceptance:

* fixed-clock activity-only runs produce identical per-domain integrals and
  samples in dense and event modes;
* crossing a power sample deadline without emitting its equivalent sample
  forces reification;
* a requested DVFS transition fails closed with a typed unknown-power reason;
* terminal cycle equality alone cannot make power state complete.

### Phase 4: state transformers and checker (1–2 weeks)

Change:

* add `sim/semantics/src/transform.rs` with the typed transformer interfaces,
  summary proof, and contract-aware refusal taxonomy;
* extend `sim/semantics/src/certificate.rs` so the narrow recurrence is one
  transformer implementation rather than the global definition of a
  certificate;
* implement the SM/L1/L2/queue/power staged application in `sim-sm`;
* replace `state_projection_complete: false` with a computed completeness
  object. A record is certifiable only when every field required by the
  selected contract is typed, validated, and transformed.

Acceptance:

* a deliberately corrupted L2 way, generation, requester owner, queue event,
  or power integral is detected by the checker;
* every unsupported or unknown field returns a stable `ReifyReason` and runs
  the detailed path;
* the detailed lane remains unchanged when accelerated mode is disabled.

### Phase 5: differential and adversarial validation (1–2 weeks)

Use the existing 13-event canary and dense reference as the first vertical
slice, then add focused traces:

| Fixture | Required assertion |
| --- | --- |
| Two SMs miss the same line/sector before fill | One L2 allocation owner; two response consumers; one lower-level fetch |
| Same-cycle access, fill, and eviction | `(time, sequence)` and phase ordering are preserved |
| Pending sector is evicted, then the same line is reallocated | Old fill becomes stale; new generation remains intact |
| Write/atomic hits a resident tag with an invalid sector | Tag-hit accounting and physical data-presence state remain distinct |
| Retained stream-L2 launch boundary | Persistent L2 state normalizes identically across launches; no pending state leaks |
| Allocation sidecar missing or unaligned | Physical-sensitive certification refuses; virtual diagnostic remains explicit |
| Long idle interval crossing a power sample | Exact activity integral and sample event are emitted, or the interval reifies |
| Fixed-clock idle/active/gated transitions | Dense and event power state is byte-identical |
| Synthetic unknown DVFS/periodic power event | Typed refusal; no guessed transition |

For each fixture compare at every event boundary, not just the terminal
cycle:

* queue depth, next event, and pending typed payloads;
* warp/block/front-end/pipeline/async state;
* L1 ways, fills, MSHR ownership, and counters;
* shared L2 ways, generations, allocation owners, request consumers, and
  post-event counters;
* power-domain state, activity deltas, sample deadlines, and provenance;
* issue, modeled-event, completion, launch, and typed-tail frontiers as
  separate evidence.

### Phase 6: enable certified regions (1–2 weeks)

Start with the existing narrow compute-only interval. Add memory-aware regions
only when all of these are true:

* shared L2 and all requester ownership is complete;
* no cache/MSHR/response/queue transition can occur inside the interval unless
  it is represented by the transformer;
* the address domain is admitted by the contract;
* power is either outside the contract or exactly transformed;
* external host, DMA, retry, refresh, periodic, and cross-SM arrivals are
  covered by the lookahead;
* same-time phase and sequence reservations are explicit.

The selector should return `CERTIFY`, `REIFY`, or `UNKNOWN`. `UNKNOWN` must be
treated as detailed execution, never as a successful but lower-confidence
summary. Keep certified regions mechanism-local; do not collapse independent
issue, completion, memory, launch, and power frontiers into one score.

## Validation and release bars

The work is complete only when all of the following hold:

1. V2 traces are typed end-to-end. No field used by the checker is a debug
   string, process-local owner token, pointer, or unordered map dump.
2. The 13-event canary and dense reference have equal typed state at every
   event and equal final replay output.
3. The retained small and large SM86 cohorts still match the existing replay
   counters and timing when V2 tracing is disabled.
4. Positive certified regions have exact typed boundary equality against the
   detailed lane, including L2 ownership and declared power state.
5. Negative cases reify for missing allocation identity, physical mapping,
   pending response ownership, queue lookahead, cache generation, sample
   deadline, or DVFS state.
6. The checker finds one-field corruptions in ways, masks, generations,
   requester identities, due times, and power integrals.
7. No NCU/NVML scored result changes simulator output when it is replaced,
   removed, or perturbed; those observations remain validation-only.
8. Any speed claim includes planner, typed projection, transformer, execution,
   and output costs. If the typed path is slower, retain it as a correctness
   and observability feature without claiming acceleration.

The first certifiable result should be stated narrowly as:

> For the declared SM86 model, fixed captured input stream, address domain,
> and observation contract, the typed summary transformer produces the same
> event-boundary state and declared observations as detailed replay.

It should not be stated as a claim of physical L2 slice ownership, silicon
power accuracy, or architecture-independent trace validity.

## Risks and explicit stop conditions

* **L2 state still cannot be normalized.** If the producer does not expose a
  stable requester identity or the trace cannot establish the selected address
  domain, keep L2-dependent regions detailed and report the missing field. Do
  not hash opaque debug strings into an apparent identity.
* **Power model lacks an independent input.** Keep `ActivityOnly`, omit watts,
  and exclude power from performance-only certification. Do not fit power from
  scored NCU rows.
* **Typed serialization dominates runtime.** Make V2 tracing opt-in and keep
  the detailed execution path unchanged; measure serialization separately from
  simulation.
* **L2 mapping ambiguity changes rankings.** Restrict the result to the
  virtual replay contract until an independent physical mapping study exists.
* **A summary changes same-time order.** Reify the region. A terminal-cycle
  match is insufficient if event sequence, ownership, or sample order differs.

## Sources

1. Vijay Kandiah et al., “AccelWattch: A Power Modeling Framework for Modern
   GPUs,” MICRO 2021. The project description separates constant, static, and
   dynamic power and derives dynamic activity from a performance model or
   hardware counters; it does not treat one measured power number as complete
   state. [AccelWattch project and paper](https://accel-sim.github.io/accelwattch.html).
2. NVIDIA, “Ampere GPU Architecture In-Depth,” GA102 whitepaper. The GA102
   memory system is partitioned into L2 slices paired with memory controllers,
   which supports representing shared slice geometry explicitly rather than
   inventing per-SM ownership. [NVIDIA whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf).
3. Tor M. Aamodt et al., GPGPU-Sim manual. Its L2 model is shared by SIMT
   cores and evolves through access, MSHR, fill, interconnect, and DRAM queues;
   this is the relevant prior for requiring queue and ownership state in a
   transformer. [GPGPU-Sim manual](https://gpgpu-sim.org/manual/index.php/Main_Page).
4. Sun et al., “MGPUSim: Enabling Multi-GPU Performance Modeling and
   Optimization,” ISCA 2019, and the pinned local mechanism memo. The useful
   transfer is event-driven advancement with explicit component wake state,
   while active compute and cache/MSHR state remain detailed.
   [`mgpusim_mechanism_memo.md`](../../GPU_sims_notes/upstream_research_20260901/mgpusim_mechanism_memo.md).
5. Local prototype assessment and source: deterministic `(time, sequence)`
   events, lane-aware scoreboard state, explicit resource ownership, and
   refusal of memory side effects in causal regions.
   [`GPU_prototypes/report-source.md`](../../GPU_prototypes/report-source.md),
   [`GPU_prototypes/src/mini_gpu.rs`](../../GPU_prototypes/src/mini_gpu.rs).
6. Local event-skipping research: exact summaries require a state-complete
   transformer and detailed fallback; blanket deletion of GPU events is not a
   sound contract.
   [`event_skipping_cycle_accuracy_research_20260912.md`](../docs/event_skipping_cycle_accuracy_research_20260912.md).

[^1]: The AccelWattch project page and paper are cited above. The local
     architecture card’s `power_w_median` fields are NVML covariates for probe
     provenance, not a simulator power model.
