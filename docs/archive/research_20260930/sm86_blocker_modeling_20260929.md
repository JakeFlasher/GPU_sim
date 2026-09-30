# SM86 blocker modeling: a boundary-constrained latent-event model

**Status:** research design and source-study artifact, 2026-09-29.  This is
not a claim that the four external blockers are closed, and it is not a new
accuracy score.  The model is deliberately conservative: the existing native
replay remains the reference path, and an unavailable latent field is emitted
as a typed unknown, interval, or candidate set rather than guessed.

## 1. Objective and completion bar

The resumed research pass has exercised every remaining public seam that was
reasonably reachable on the locked RTX 3090 Ti/SM86 host.  Four mechanisms are
still outside the public observation contract:

1. exact physical pipe, sub-unit, and issue-slot tags;
2. unrestricted independent SM86 SASS, scheduler/control-word, CUDA ELF, and
   kernel-ABI generation;
3. cache-line ownership, read-from source, and complete dirty writeback to
   DRAM;
4. one timing endpoint whose start, end, and included work are identical to
   NCU, CUPTI, CUDA events, device timers, and simulator completion.

The practical goal of this document is **not** to invent values for those
fields.  It is to define a useful model that can make progress on causal
questions while preserving the distinction between:

- what the trace or card states;
- what public hardware tools directly observe;
- what follows by a checked derivation;
- what is only a hypothesis;
- what is physically possible but currently unobservable.

The completion bar for a future implementation is therefore:

* the current default replay produces the same instruction counts, frontiers,
  cache state, and timing diagnostics when the new layer is disabled;
* each new result carries a typed identity, provenance, and capability scope;
* a value is reported as exact only when every admitted model world agrees;
* otherwise the result is a conservative interval, partial order, candidate set,
  or explicit `unknown`;
* no NCU/CUPTI value from a scored workload is used to select a simulator
  parameter for that workload;
* no bounded NAK bridge, cache service, or endpoint crosswalk is renamed as
  unrestricted native completion.

## 2. Current evidence boundary

The current local evidence is retained in the completion audit and deep
research report:

| blocker | strongest present evidence | still missing |
|---|---|---|
| physical pipes | known-population isolated-class NCU correspondences, aggregate pipe metrics, direct PMA PC/stall records | a dynamic `(instruction, warp, issue ordinal, physical sub-unit, issue cycle)` witness for mixed kernels |
| code generation/ABI | independent Mesa NAK SM86 bodies, seven operation families, a template NAK-to-CUDA-ELF bridge, NVIDIA PTX compiler/linker execution, bounded `denvdis` encodings | general relocatable CUDA ABI/QMD/control generation and complete SM86 form coverage from an independent stack |
| cache/DRAM | public RM full-cache writeback+invalidate accepted; NAK producer coupled to CPU-visible consumer and flush; typed L1/L2/writeback experiments | line owner, read-from source, dirty payload residency, bypass consumer, proof that all dirty lines reached DRAM |
| timing | locked-clock campaigns, pidfd host timing, CUPTI/CUDA-event crosswalk, warm/cold cache comparison | a public relation equating NCU WaitForIdle, CUPTI activity, CUDA events, native timers, simulator completion, and replay wall time |

The relevant in-tree seams are `sim/sm/src/issue.rs` and
`sim/sm/src/system.rs` (separate issue, response, launch, and typed-tail
frontiers), `sim/mem/src/l1.rs`, `l2.rs`, `coherence.rs`, `writeback.rs`, and
`payload.rs` (transactional cache and service experiments), and
`tools/nvptx_compile_sm86/` (official PTX paths plus the bounded NAK bridge).
The new model must sit beside these seams, not erase their explicit gaps.

## 3. Source and literature study

This section records the source-code seams that informed the design.  These
are clues and contrasts, not interchangeable SM86 ground truth.

### 3.1 Accel-Sim/GPGPU-Sim

The public Accel-Sim framework uses an NVBit SASS trace frontend, a detailed
GPGPU-Sim 4.x timing model, a hardware correlator, and a tuner.  The current
2.0 release extends the same approach to Hopper asynchronous operations,
chiplet memory, compressed traces, and CUPTI PM-sampling correlation.  Its
public claims are explicitly correlations against the target card, not a
per-instruction silicon issue trace.

Source inspection at the 2026-09-29 shallow clone found:

* `src/gpgpu-sim/shader.cc::shader_core_ctx::issue` rotates scheduler units;
  `scheduler_unit::cycle` walks prioritized warps, checks a scoreboard, and
  chooses a logical register-set/execution unit.  The source exposes the
  scheduler policy and modeled resource availability, not a physical GA102
  route tag.
* `src/gpgpu-sim/gpu-cache.{h,cc}` keeps valid/dirty tags, MSHR state,
  bandwidth ports, fill queues, and writeback requests.  This is a useful
  operational cache model, but its dirty bit and `mem_fetch` status are model
  state; they are not proof of the NVIDIA RM's physical line owner.
* `src/gpgpu-sim/l2cache.cc` and `dram_sched.cc` separate subpartition queues,
  DRAM credits, latency queues, bank arbitration, and FR-FCFS row selection.
  That separation motivates keeping issue, response, and DRAM service
  frontiers separate here.
* `util/tracer_nvbit/tracer_tool` records CTA, warp, SM, virtual PC, opcode,
  masks, addresses, register values, and instruction indices.  The trace is a
  strong semantic/replay input, but its fields do not include physical pipe
  selection or an ownership/read-from witness.
* `util/tuner` searches hardware-correlated configurations.  That is useful
  for a separately declared calibration campaign, but its configuration
  selection pattern is deliberately **not** adopted for same-workload NCU
  scoring in this repository.

The 2018 modern-memory study reports that adding detailed memory scheduling
and cache behavior can reduce counter and execution error dramatically.  The
lesson for this project is to add detail where the state is causally defined;
it is not permission to infer hidden silicon state from a matching aggregate.
The 2024 Accel-Sim hardware-modeling study likewise separates frontend,
result-bus, and memory-pipeline effects instead of treating one timing scalar
as the whole machine.

### 3.2 MGPUSim/Akita

The cloned MGPUSim source has both AMD timing code and a newer NVIDIA trace
lane.  `nvidia/smsp/scheduler.go` issues a trace instruction after a register
scoreboard and a `ResourcePool` check.  `nvidia/smsp/stage.go` maps an opcode
to a stage template, with a similarity fallback for unknown opcodes;
`pipelineinstance.go` keeps explicit non-memory and request/response stages;
`smspcontroller.go` joins memory responses by request ID.  The source is a
useful example of a typed pipeline and request identity, but the similarity
fallback is unsafe for this project: an unrecognized SM86 form must refuse or
remain unknown, not inherit a nearby opcode's timing.

`nvidia/gpu/builder.go` constructs per-SMSP L1 vector caches, banked writeback
L2s, and idealized DRAM controllers using Akita ports and events.  Akita's
`timing/serialengine.go` and queue/tracing packages provide deterministic event
ordering and per-component task traces.  They motivate a model with explicit
event ownership and response identities, while their generic components do
not supply GA102 ownership or RM telemetry.

### 3.3 gem5 AMD GPU

The cloned gem5 source is a strong reference for separating scheduling,
resource readiness, memory requests, and host/device protocol:

* `src/gpu-compute/scoreboard_check_stage.cc` checks wait counts, barriers,
  operand readiness, and matrix-resource availability and records the reason
  for every non-ready wavefront;
* `schedule_stage.cc` selects waves per execution resource, checks buses and
  resource readiness, and reserves resources with explicit conflict checks;
* `global_memory_pipeline.cc` tracks coalescer tokens, per-wave outstanding
  reads/writes, ordered response buffers, and round-trip statistics;
* `gpu_command_processor.cc` defers a kernel launch while L2 invalidations are
  outstanding and documents the race between host/BAR writes and GPU cache
  state.  The implementation can model this because its Ruby/full-system
  contract defines the endpoint; the same assumption cannot be imported into
  the NVIDIA path without a matching RM/cache witness.

gem5 is AMD GCN/CDNA-oriented, not a drop-in SM86 model.  Its contribution
here is the discipline of typed stall reasons, request ownership, and explicit
cache-invalidation completion.

### 3.4 Vortex

Vortex is an open RISC-V GPGPU with a complete ISA, RTL, runtime, and SimX
stack.  In `sim/simx/core.cpp`, issue width, per-warp instruction buffers,
scoreboard reservation, dispatcher pipelines, and commit are separate.  The
SimX cache (`cache_sim.cpp`) has bank/set/tag state, MSHRs, fill/replay events,
write-through/write-back modes, and dirty evictions; the memory coalescer
(`mem_coalescer.cpp`) keeps a request tag until every lane response returns.
This is a valuable complete-stack control experiment.  It also demonstrates
why an explicit ISA/ABI choice matters: a complete open Vortex stack is
possible because its hardware and software contracts were designed together,
not because an NVIDIA cubin metadata file can be reconstructed from a SASS
word.

The source snapshots inspected for this pass were shallow clones outside the
repository (so no third-party source is added to the working tree):

| source | inspected revision |
|---|---|
| Accel-Sim framework | `d930ad6` |
| GPGPU-Sim distribution | `91880c5` |
| MGPUSim | `adf7c16` |
| Akita | `011edf0` |
| Vortex | `83badaa` |
| gem5 | `f5c5a6e` |

These revisions are research inputs only; they are not repository dependencies
or promotion pins.

### 3.5 Public NVIDIA and independent compiler evidence

The installed NVIDIA `nvPTXCompiler` and `nvJitLink` paths provide a practical
PTX-to-cubin route, but they are proprietary lowerings and metadata producers.
Mesa NAK independently emits and executes native SM86 shader bodies; the
repository's bridge places those bodies inside NVIDIA-generated ELF metadata
for one fixed two-parameter layout.  `denvdis` and
`reoLantern/nvsass-disassembler` provide instruction-word encoding or
decode/re-encode experiments, not a relocatable compiler, scheduler/control
generator, or CUDA ABI.  The model below therefore treats **semantic
instruction**, **128-bit encoding**, **container metadata**, and **launch ABI**
as four separately certified contracts.

The goal continuation now profiles six retained bridged bodies directly with
NCU through the CUDA Driver API: `03_mul`, `04_fma_like`, `05_reciprocal`,
`06_shift`, `07_xor`, and `08_maximum`.  The matrix spans 2.400–2.624 us and
3,226–3,478 GPC cycles.  This is stronger evidence that the bounded bridge
executes under the CUDA launch path, but it does not generalize its reused
metadata or expose control-word/route/ABI coverage outside the template.
Warm-cache NCU profiles of the same six bodies produce median warm/cold ratios
of 0.95485 for duration and 0.95105 for GPC cycles, with unchanged SASS and
aggregate pipe populations. This adds cache-domain sensitivity without
identifying an owner, read-from source, or complete DRAM writeback. The raw
observations are retained in
`results/alpha/sm86_nak_cuda_elf_bridge_ncu_goal_20260929/`.

A fused mixed body combining IMAD, LOP3, and IMNMX was also bridged through a
larger template. It returned `7,11,8,13`; cold/warm NCU was 2.464/2.336 us and
3,284/3,090 GPC cycles, with 26 instructions and aggregate 18 ALU/3 LSU
counts. This supplies a mixed-operation aggregate observation while preserving
the route candidate set: no public field maps those dynamic PCs to physical
sub-units or issue cycles.
Five-repeat NCU follow-up medians are 2.464 us / 3,271 GPC cycles (`all`) and
2.336 us / 3,089 cycles (`none`), with stable 26-instruction, 18-ALU,
3-LSU populations. Timing variation without population variation reinforces
the model's separation of aggregate counts from latent route/cache state.

### 3.6 Memory-model and timing references

The PTX consistency specification defines reads as choosing a write from a
set of candidate writes constrained by program order, causality, scope, and
communication order.  Lustig, Sahasrabuddhe, and Giroux formalize that model
and machine-check its mapping.  This is directly useful: the model can keep a
candidate `read-from` set without pretending to know the physical cache owner.

The Nsight Compute guide states that NCU duration inserts a WaitForIdle and
semaphore before the grid and another WaitForIdle/semaphore after it, whereas
pre-Blackwell Nsight Systems/CUPTI uses a different prolog/end mechanism.  It
also documents replay serialization, cache flushing, and clock-control
differences.  CUPTI's range profiler can fix clocks and reset cache state, but
that improves reproducibility; it does not make its endpoint identical to
NCU or simulator completion.

## 4. Model foundation: a set of admissible worlds

### 4.1 Typed event identity

Every dynamic object entering the model is an `EventKey`:

```text
EventKey {
  launch_uid, function_uid, cta_uid, warp_uid, instruction_ordinal,
  pc, lane_or_sector, sm_id, origin
}
```

Fields are optional only when the producer contract says they are absent.  A
missing field cannot be regenerated from array order.  The key is separate
from content identity: alpha replay stays unbound, while a future bound
promotion may attach a content digest without changing event semantics.

### 4.2 Facts, bounds, candidates, and provenance

One observation is represented conceptually as:

```text
Observation {
  key: EventKey,
  fact: FactSet,                    // exact observed/derived fields
  interval: Map<Field, [lo, hi]>,  // checked bounds, inclusive
  candidates: Map<Field, Set<Value>>,
  order: PartialOrder,              // happens-before / no-later-than edges
  provenance: ObservationClass,
  contract: CapabilityScope,
}
```

`ObservationClass` is one of `DirectTrace`, `DirectHardware`, `Derived`,
`Hypothesis`, `Inference`, or `Unobservable`.  `CapabilityScope` names the
architecture, driver/tool version, operation family, ABI shape, and cache
policy for which the observation is valid.

For an admitted set of model worlds `W`, report:

```text
must(f) = intersection { f(w) | w in W }
may(f)  = union        { f(w) | w in W }
```

An exact value is legal only when `must(f) == may(f)` and the provenance is
not merely hypothetical.  A singleton hypothesis is still labeled a
hypothesis.  This lattice is the central anti-regression device: additional
evidence can shrink `may`, but an unsupported assumption cannot silently turn
an unknown into a fact.

### 4.3 Frontiers are a vector, never one score

The model keeps the existing frontiers and adds no scalar selection score:

```text
F = {
  issue, modeled_event, launch_envelope, typed_tail,
  visibility, writeback_service, endpoint_start, endpoint_end
}
```

Edges such as `issue <= modeled_event <= typed_tail` are explicit only when
the event contract establishes them.  Pipe route, cache owner, and endpoint
identity are orthogonal dimensions.  Two worlds are compared by an explicit
partial order or a mechanism-local discriminator, not a weighted “accuracy”
number.

## 5. Blocker A — latent physical-pipe and issue model

### 5.1 What is known

The current SM86 replay knows the dynamic PC/warp/mask stream, decoded control
bits, dependence barriers, logical operation class, sub-core placement,
resource availability, and model issue/response frontiers.  Isolated NCU
probes constrain aggregate class populations.  PMA contributes sampled PC,
stall, warp, and SM identity.  None of these fields is a universal physical
pipe or issue-slot tag.

### 5.2 Latent state

For each dynamic instruction `i`, retain:

```text
RouteState(i) = {
  logical_class,                 // exact classifier or typed gap
  candidate_pipes R_i,           // may be >1
  candidate_subcores S_i,
  issue_cycle interval T_i,
  issue_slot interval Q_i,
  aggregate_witnesses A_i,
  uniqueness: {Exact, Set, Unknown},
}
```

The physical assignment is a constrained matching problem, not a balancing
heuristic.  For candidate pipe `p`, introduce `x[i,p] in {0,1}`:

```text
sum_p x[i,p] = 1
x[i,p] = 0                  when p is not in R_i
c_p^- <= sum_i x[i,p] <= c_p^+   // only when a counter gives a bounded count
```

Issue cycles use the existing control-bit and dependency model to establish
the earliest legal issue.  If a public sample has no cycle resolution, it
constrains a distribution or an interval, never a particular instruction.
If the matching has multiple solutions, the emitted route is the candidate
set and the model timing remains the current logical timing.

### 5.3 Safe hypotheses

* An isolated class-to-aggregate correspondence can constrain the population
  for that class and card/driver configuration.
* A mixed-kernel aggregate counter cannot identify a route for a particular PC.
* PMA/CUPTI sampling is statistical unless a producer gives a dynamic ordinal
  join; a sampled PC is not an issued-instruction receipt.
* The existing SM86 control-word decoder can continue to drive dependency and
  stall semantics; it must not be interpreted as a physical-pipe tag.

### 5.4 Outputs and validation

The first implementation should emit a `route_envelope.v1` diagnostic with
candidate sets, count bounds, and unresolved fields.  It must not alter issue
cycles.  Validation consists of:

1. exact population conservation for the isolated kernels;
2. no route-set shrinkage for a mixed kernel without a new witness;
3. invariant-preserving projection: disabling the envelope produces byte-for-
   byte identical replay metrics;
4. synthetic matching tests with zero, one, and multiple feasible assignments;
5. a failure result, not an arbitrary route, when aggregate bounds are
   inconsistent.

Closure still requires a silicon/RM/firmware stream containing dynamic route
   and issue identity.  A feasible matching is not that stream.

## 6. Blocker B — independent ISA/ABI generation model

### 6.1 Four contracts, four certificates

The model treats compilation as a graph:

```text
Semantics -> Instruction encoding -> Control/scheduler metadata
          -> ELF/relocations/constants -> launch ABI/QMD -> execution
```

Each edge receives a separate capability certificate:

| certificate | example | does not imply |
|---|---|---|
| `SemanticSubset` | NAK add/mul/MUFU/SHF/LOP3 bodies | CUDA ABI or all SASS |
| `WordEncoding` | `denvdis` IADD3 word | linker, relocations, scheduling |
| `ContainerReuse` | NAK body in NVIDIA template | general ELF/QMD generation |
| `OfficialPtxLowering` | `nvPTXCompiler`/`nvJitLink` cubin | independent lowering |
| `LaunchObserved` | Driver API output 7..14 or 7..10 | all-form correctness |

Every generated artifact carries a `CoverageSet` of exact opcodes/forms,
register/addressing modes, metadata assumptions, and ABI parameter layouts.
An artifact outside that set is rejected, not assigned the nearest opcode's
timing.  In particular, MGPUSim's similarity fallback is useful for a
prototype but is not safe for a native SM86 claim.

### 6.2 Latent fields and hypotheses

The unresolved latent tuple is:

```text
AbiState = {
  register_contract, constant_layout, relocation_set,
  control_words, qmd_fields, stack/call contract,
  resource_usage, launch_parameter_shape,
}
```

NAK independently supplies a body and its own runtime/QMD contract.  The
current bridge supplies one hand-checked template and explicit constant
remaps.  It is valid to model `BodyEncoding = proven` and
`ContainerReuse = proven-for-template`; it is not valid to infer
`RelocatableCudaAbi = proven`.

### 6.3 Safe implementation seam

Add a capability ledger beside `tools/nvptx_compile_sm86/` with exact scopes:

```text
artifact_id, producer, arch, operation_forms, body_contract,
metadata_contract, relocations, parameter_layout, launch_observation,
unsupported_forms, provenance
```

The ledger is observation metadata, not a code-generation authority.  The
existing NVIDIA and NAK paths remain unchanged.  A future independent
compiler can add a new producer entry and be differentially checked against
the same semantic microkernels without changing native trace replay.

### 6.4 Validation and closure

For each admitted form, run: encode/decode round trip; `nvdisasm` decode;
semantic value test; launch under the producer's own ABI; and, only for a
declared bridge, launch under the bridged ABI.  Negative tests must include
wrong metadata size, wrong constant remap, missing relocation, unsupported
control word, and an operation outside the coverage set.  General closure
requires independently generated ELF/relocations/QMD/control plus a matrix
of ABI shapes, not one successful template launch.

## 7. Blocker C — cache ownership, read-from, and DRAM completion

### 7.1 Separate architectural visibility from physical residency

The model has two layers:

1. **Architectural candidate semantics.**  A byte read `r` has a candidate
   set `RF(r)` of writes (including initialization) allowed by PTX scope,
   program order, causality, and communication order.  This layer can prove
   value/ordering properties without naming an L1/L2 owner.
2. **Physical cache hypothesis.**  For each line/sector:

```text
PhysicalState = {
  must_owner, may_owners,
  valid_mask, dirty_mask,
  known_payload, may_payload,
  allocation_generation,
  pending_fill/writeback/coherence tokens,
  flush_coverage: Must/May/Unknown,
}
```

The current `sim/mem` L1/L2 generation-bound fills, transactional dirty plans,
payload checks, writeback queue, and ordered coherence service are excellent
bounded model states.  They are not promoted to `must_owner` or
`dram_persisted` without an independent observation.

### 7.2 Event transitions

* `GPUStore` adds a write to `RF` and marks a modeled dirty candidate only when
  the selected opt-in policy admits it.  It does not clear an older payload or
  claim DRAM visibility.
* `L2Fill` installs only the sectors authorized by its allocation-generation
  token.  A stale token is a no-op/error, never a resurrection.
* `Fence` adds an architectural ordering edge.  It may drain a modeled queue
  only when the carded experiment explicitly owns that queue.
* `PublicFlushAccepted` records RM packet acceptance/retirement and a typed
  service interval.  It does not by itself set `must_owner=DRAM`.
* `CPURead` can reduce `RF` candidates only when the mapping and read path are
  independently characterized.  Agreement before/after a flush is evidence
  of a value observation, not proof of the read source.
* `DRAMComplete` is emitted only in a hypothetical world whose protocol has a
  complete lower-level writeback acknowledgment and an independent consumer;
  otherwise the world ends at `WritebackServiceComplete` or
  `FlushProtocolComplete`.

### 7.3 Must/may cache results

For each line, expose:

```text
must_clean, may_dirty, must_resident, may_resident,
must_read_from, may_read_from, must_dram_persisted, may_dram_persisted
```

The public RM `0x38`/`0x3b` controls can add a `FlushProtocolComplete` fact.
The current NAK-to-CPU-visible coupling can add a value equality fact.  Neither
fact collapses the physical owner/read-from candidate sets.  The model must
therefore return `unknown` for `must_dram_persisted` on the current evidence.

### 7.4 Safety invariants

* A dirty payload cannot be discarded before a matching service plan is
  accepted and completed.
* A fill response cannot precede its L2 lower bound, registered pending-sector
  ready time, or allocation generation.
* A same-line later writeback cannot be cleared by an older completion token.
* An L2/L1 hit count is an observation-class counter, not a read-from proof.
* Cache policy, payload, and RM experiments are not used to select timing
  parameters for the same scored workload.

### 7.5 Experiments that would shrink the world set

The next useful probes are interventions, not more passive counters:

1. GPU producer store to a uniquely allocated line;
2. a consumer whose path is independently classified as cacheable or bypass;
3. a CPU/BAR1 or DMA read with a known mapping and ordering contract;
4. public flush/invalidate between producer and consumer;
5. repeated address-alias and partial-sector variants;
6. an oracle that reports owner/read-from or a complete writeback acknowledgment.

Items 1–5 can eliminate candidate worlds.  Only item 6 can close the physical
ownership/complete-DRAM boundary.

## 8. Blocker D — endpoint and timing model

### 8.1 Endpoint descriptors

Each timing source gets an explicit descriptor:

```text
Endpoint {
  name, clock_domain, start_trigger, end_trigger,
  included_work, excluded_work, serialization, cache_state,
  clock_control, uncertainty, provenance
}
```

The project currently has at least these endpoints:

| endpoint | semantic start/end (model description) |
|---|---|
| `issue_frontier` | last modeled issue decision; no cache tail |
| `modeled_event` | last deterministic simulator event |
| `launch_envelope` | grid submission/CTA distribution boundary |
| `typed_tail` | modeled response, visibility, writeback, or coherence tail |
| `NCU_wait_for_idle` | NCU semaphore/WaitForIdle envelope before and after grid |
| `CUPTI_activity` | CUPTI activity record span |
| `CUDA_event` | stream event timestamps around the launch |
| `globaltimer` | clock read by selected executing block/thread |
| `host_wall` | process and tool overhead inclusive wall time |

The model stores a partial order such as:

```text
issue_frontier <= modeled_event <= typed_tail
launch_start <= first_scheduled_CTA
last_scheduled_CTA <= launch_end
```

It does not add `NCU == CUPTI == CUDA_event` unless a separately validated
control establishes that exact relation.  A pairwise crosswalk is represented
as a bounded interval with a campaign identity, not a global conversion.

### 8.2 Clock relation hypothesis

For an endpoint pair with an independently controlled clock, a local relation
may be recorded as:

```text
t_endpoint = a * modeled_cycles + b + epsilon,
epsilon in [epsilon_lo, epsilon_hi]
```

`a`, `b`, and the error interval are observations of that crosswalk only.  They
are not fed back into the simulator for the same scored workload.  Warm/cold
cache state, profiler serialization, launch concurrency, and host overhead
must be separate dimensions.  The existing `timing_crosswalk_*` and cold/warm
campaigns are therefore retained as distinct endpoint observations.

### 8.3 Validation

Validate endpoint descriptors with ordering and metamorphic checks:

* adding a modeled writeback tail cannot make `typed_tail` earlier;
* disabling cache flush cannot be labeled an NCU-equivalent run;
* holding clocks fixed should reduce clock-domain variance without proving
  endpoint identity;
* changing host polling to pidfd must change `host_wall` overhead only, not
  simulator event cycles;
* a launch with no work must not fabricate an instruction-duration relation.

Unified timing closure requires a public endpoint contract with the same
triggers and included work.  A high correlation or a ratio near one is not
that contract.

## 9. Cross-blocker composition

The four models compose through `EventKey` and the frontier vector, not by
collapsing into one simulator score:

```text
trace/control ──> logical issue event ──> memory request event
       │                 │                         │
       │                 ├─ route candidates       ├─ L1/L2 candidate state
       │                 └─ issue interval          ├─ RF candidate writes
       │                                           └─ typed response/tail
compiler artifact ──> body/container/ABI scope ──> launch envelope
measurement descriptor ─────────────────────────> endpoint projection
```

An independent NAK body can enter the compiler ledger and produce a native
value observation.  It cannot silently become a scored native timing trace,
and its PMA samples cannot become physical route tags.  A public flush can
complete a protocol event and alter a candidate visibility relation; it cannot
silently close DRAM persistence.  A timing crosswalk can compare two endpoints
within a campaign; it cannot choose the route or cache owner.

## 10. No-regression implementation plan

The implementation should proceed in small, reversible slices:

### Phase A — schema and projection only

Add a versioned `boundary_model.v1` diagnostic schema (JSON or a Rust type in
`sim/semantics`) containing `EventKey`, provenance, intervals, candidate sets,
partial-order edges, and capability scopes.  No scheduler or cache behavior
changes.  A projection test must compare baseline replay output with the
model-disabled output.

### Phase B — route envelope

Add a read-only route-envelope builder in `sim/sm` that consumes existing
logical classifications and isolated-class evidence.  It may solve the
matching constraints above, but it must never feed a route guess back into
issue timing.  Use exact-form classification and typed refusal for unknown
operations.

### Phase C — compiler capability ledger

Record the existing NVIDIA, NAK, bridge, and `denvdis` artifacts with exact
coverage and ABI scopes.  Add negative coverage tests before adding another
code generator.  Keep all generated artifacts in the existing probe lane.

### Phase D — cache world observer

Wrap, rather than rewrite, the existing generation-bound L1/L2 and
transactional writeback/coherence APIs.  Emit must/may owner/read-from/
persisted fields.  Default native replay remains payload-agnostic and retains
the current explicit boundaries.

### Phase E — endpoint graph

Attach endpoint descriptors and partial-order edges to existing campaign
reports.  Keep NCU/CUPTI data as independent observations and never mutate
architecture cards from them.

### Phase F — differential and adversarial validation

Run the current representative SM86 canaries and the 15-workload campaign
with the new layer disabled and enabled.  Require identical baseline
instruction counts and frontiers; require monotonicity, generation, identity,
and resource-limit tests; use timeout-bounded real replay for event changes.

## 11. Acceptance and non-acceptance criteria

The model is useful when it can answer questions such as “which pipe classes
remain possible?”, “which writes may satisfy this read?”, or “which tails are
included in this endpoint?” without inventing a fact.  It is **not** a blocker
closure merely because:

* a maximum-flow assignment returns one solution after arbitrary tie-breaking;
* a formula balances aggregate pipe counts;
* a NAK body runs through a reused CUDA template;
* a public flush packet retires or a CPU value stays unchanged;
* a simulator cycle count is numerically close to NCU/CUPTI;
* a custom or PTX subset executor passes a functional test.

Closure of each blocker still requires the missing external witness described
in the opening table.  Until then, the model's strongest result is a
scientifically honest set of admissible explanations and conservative bounds.

## 12. References and inspected source

Primary papers and documentation:

* [Accel-Sim 2.0, arXiv:2608.22602](https://arxiv.org/abs/2608.22602)
* [Accel-Sim/GPGPU-Sim framework](https://github.com/accel-sim/accel-sim-framework)
* [GPGPU-Sim source](https://github.com/accel-sim/gpgpu-sim_distribution)
* [Exploring Modern GPU Memory System Design Challenges, arXiv:1810.07269](https://arxiv.org/abs/1810.07269)
* [Analyzing and Improving Hardware Modeling of Accel-Sim, arXiv:2401.10082](https://arxiv.org/abs/2401.10082)
* [Demystifying Nvidia Ampere, arXiv:2208.11174](https://arxiv.org/abs/2208.11174)
* [MGPUSim](https://github.com/sarchlab/mgpusim) and [Akita](https://github.com/sarchlab/akita)
* [Vortex source](https://github.com/comsys-lab/vortex) and [Vortex paper, arXiv:2110.10857](https://arxiv.org/abs/2110.10857)
* [gem5 GPU source](https://github.com/gem5/gem5) and [full-system GPU model notes](https://www.gem5.org/documentation/general_docs/gpu_models/gpufs)
* [A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model)
* [PTX memory consistency documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)
* [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
* [CUPTI documentation](https://docs.nvidia.com/cupti/main/main.html)
* [AMALi analytical GPU model source](https://github.com/ShihengCao/AMALi)
  (`ISA_parser/sass_parser.py`, `src/memory_model.py`, and
  `src/accelerators.py`), which keeps unit latency, reuse-profile cache
  response, and throughput/occupancy slopes separate.
* [Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures, arXiv:2605.04178](https://arxiv.org/abs/2605.04178)
* [GPGPU Performance Estimation with Core and Memory Frequency Scaling, arXiv:1701.05308](https://arxiv.org/abs/1701.05308)
* [GCStack+GCScaler source](https://github.com/yonsei-hpcp/gcstack_gcscaler)
* [GPU cache model source](https://github.com/tue-es/gpu-cache-model)
* [Ampere GPU memory model source](https://github.com/brucelee503jo3/gpu-memory-model)
* [Mesa NAK compiler](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/nouveau/compiler/nak)
* [redplait/denvdis](https://github.com/redplait/denvdis)
* [reoLantern/nvsass-disassembler](https://github.com/reoLantern/nvsass-disassembler) and [arXiv:2609.18662](https://arxiv.org/abs/2609.18662)

Local source and evidence:

* `docs/PROJECT_MEMORY.md`, `docs/STATUS.md`, `docs/sm86_deep_research_20260928.md`,
  and `docs/sm86_completion_audit_20260929.md`;
* `sim/sm/src/issue.rs`, `sim/sm/src/system.rs`,
  `sim/mem/src/{l1,l2,coherence,writeback,payload}.rs`;
* `results/alpha/sm86_pipe_metric_surface_20260929/`,
  `sm86_nak_pma_probe_20260929/`,
  `sm86_nak_cuda_elf_bridge_20260929/`,
  `rm_public_l2_flush_control_probe_20260929/`,
  `nak_shader_public_flush_coupling_20260929/`, and
  `timing_crosswalk_3090ti_20260929_current/`.
