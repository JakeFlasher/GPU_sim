# Remaining limitation completion plan

This plan preserves the original scope. It does not relabel bounded diagnostics
as native completion. Native traces, cards, control sidecars, and scored NCU
measurements remain separate evidence domains.

## Research conclusions

1. CUDA scopes have distinct coherency points: block scope at L1, device scope
   at L2, and system scope beyond the GPU. The simulator must retain scope,
   order, launch-boundary, and initial-state provenance rather than infer it
   from L2 misses.
2. PTX is a virtual ISA translated to target-specific instructions. A PTX
   subset or custom ITS executor can provide functional value semantics, but
   neither is native SASS regeneration.
3. Nsight pipeline metrics are estimated for instructions that may dynamically
   choose among multiple pipelines. Candidate domains and model-local
   scheduling are useful, but exact physical route IDs need an independent
   route source.
4. Current native protobuf traces do not carry complete registers, store
   payloads, memory return values, atomic results, or synchronization-object
   identities. The NVBit register-value example is therefore a separate
   partial-capture artifact.
5. The two blockers now prioritized are not solved by a larger heuristic:
   (a) native memory visibility/dirty persistence requires a value-bearing,
   scope-aware happens-before/read-from model and a globally ordered event
   stream; (b) the required external comparison is the published Accel-Sim
   aggregate accuracy-ceiling/throughput-speed-floor lane evaluated against
   local NCU ground truth. A
   matched architecture/configuration crosswalk and equal endpoint semantics
   are required only for the optional direct executable E10-P timing lane, not
   for registering or evaluating the literature targets.
6. Exact physical-pipe reconstruction is an inverse problem. On GA10x, public
   counters include aggregate/overlapping logical and physical sub-pipelines,
   and dynamic instruction routing can choose among candidates. Without an
   independently captured route witness, the scientifically valid result is a
   feasible set/range or refusal, not a chosen route ID.
7. “Value-dependent SASS regeneration” must be split into two deliverables:
   dynamic value-dependent execution-trace regeneration (closable for a finite
   admitted SASS subset with NVBit or an independent interpreter), and static
   SASS binary generation (a pinned NVIDIA compiler/toolchain artifact; runtime
   values do not rewrite the compiled instruction bytes).

Primary references: [CUDA thread scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/), and [NVBit](https://github.com/NVlabs/NVBit).

## Dependency order and acceptance bars

### 0. Alpha execution contract

The active branch is a fast, unbound development lane. It does not compute or
require SHA-256 artifact identities, verification manifests, freshness checks,
or test gates. Inputs are selected by explicit paths and operator intent;
outputs are development observations only. Preserve issue, event, completion,
launch, and typed-tail frontiers, but do not turn them into reproducibility or
silicon-validation claims. A later promotion lane may add stronger evidence
without changing this alpha path.

### 1. Functional state and ordered visibility

Extend native capture or a separate value contract with registers, predicates,
payloads, returned values, atomics, fences, synchronization objects, and
initial state. Keep accesses and fences ordered at equal cycles.

**Current:** ordered L1 access/fence stream, fence frontiers, pending-fill
tracking with generation-matched fill-visible callbacks, custom ITS v3
register/predicate trace, and bounded NVBit register-value adapter. Bypass
reads are now distinct from cacheable misses and do not publish phantom
sharers. Visibility access/fence/fill events now enter the global engine queue,
but actor batch-ahead interaction remains explicitly
`batch_ahead_visibility_unsupported`; full payload/read-from state and
unsupported past/future ordering remain typed refusal cases.

**Acceptance:** state-complete litmus traces, deterministic sequence checks,
explicit missing-state refusals, and reference/device differential results.

**Required state contract (promotion/native-memory track):** every admitted
  record must bind `(binary_sha256, function_uid, vpc, dynamic_ordinal, CTA,
warp, lane, SM)` to the authoritative SASS/control sidecar and carry active /
predicate masks, address space, byte width/mask, cache operator, source and
destination values, load return bytes, atomic old/new values, fence
scope/semantics, synchronization-object/read-from identity, initial bytes and
cache-residency provenance. Missing payload or synchronization identity is a
typed refusal, not a zero/default value.

### 2. Native coherence and dirty flush

Wire scope-aware release/acquire behavior, initial ownership, L1 invalidation,
dirty L1→L2→DRAM payload flow, eviction ordering, and host/launch boundaries.

**Current:** bounded owner/sharer queue, tokenized invalidation, typed fences,
resident-hit dirty masks, and capacity-bounded terminal flushing.

**Acceptance:** cross-SM litmus values, fence/atomic ordering, exact payload
conservation, zero residual dirty state, stale/capacity refusal tests, and
independent hardware value checks.

**Implementation order for the prioritized memory blocker:**

1. Replace the current mask-only observation with a versioned sidecar carrying
   byte payloads and an allocation/fill generation. Distinguish cacheable
   misses that will become resident from bypass/nonresident reads.
2. Route L1 access, fence, fill-visible, invalidation, and flush events through
   one global `(cycle, sequence)` event queue. A callback must carry the line
   generation and be rejected if the allocation has been evicted/reused.
   `CCTL.IVALL` now resets both cache state and requester-owned visibility
   state in the bounded event path; full hardware cache-control semantics
   remain outside the contract.
3. Implement PTX communication order and release/acquire/happens-before/read-
   from relations before assigning ownership. Unsynchronized conflicting
   accesses are data-race/allowed-outcome cases and must not be converted into
   unconditional invalidations. Include aliases/proxy fences and CTA/device/
   system participant identity; otherwise refuse.
4. Add byte-bearing L1→L2 merge and L2→DRAM state. Track known-byte masks,
   partial-sector writes, atomic results, ownership transitions, eviction
   order, and an explicit `FlushReason` (eviction, fence, kernel completion,
   stream/host synchronization). Clear dirty bits only after an acknowledged
   payload commit; unknown bytes or double ownership refuse.
5. Differentially test a randomized reference state machine and the following
   fixed litmus corpus: GPU-scope message passing, device atomics, mixed
   `.cta`/`.gpu` scopes, virtual aliases/proxy fence, partial-sector stores,
   pending-fill invalidation, and host D2H after stream synchronization.
   Compare returned values and every output byte, not only cache counters.

The current `L1VisibilityQueue`, resident-hit dirty mask, and writeback queue
are useful T0/T1 scaffolding and refusal tests. They do not satisfy this bar:
ATOM/RMW, RED, and asynchronous-copy paths are refused, fill generations/global
event order are now represented only at the bounded event/observation seam;
fences are timestamp frontiers rather than PTX causality, and no payload
reaches modeled
DRAM. `SectorPayloadV1` and L1 payload retention now prove transactional
known-byte updates through the cache flush-plan seam; they are not yet
connected to L2/DRAM ownership or the native flush boundary.

### 3. Finite compiler target

Select one explicit ISA/language/ABI/memory-model target. The current
`ptx_subset_v2`/`simt_its_v1` path is a bounded functional target, not arbitrary
PTX or SASS.

**Acceptance:** source corpus, parser/lowering, ABI rules, negative cases,
reference execution, and generated-artifact identity. Unsupported forms refuse.

Use two explicitly separate lanes: (i) SM86 native integration through pinned
`nvcc`/`ptxas` plus `cuobjdump`/`nvdisasm`, with optional bound identities for
PTX, cubin, static SASS/control and toolchain; and (ii) an open/custom target with a declared
grammar, register/predicate files, encoding, ABI, address spaces, memory model,
decoder/assembler and independent ISS. Grow `ptx_subset_v2` only as the latter
bounded target. A schema or opcode list alone is not arbitrary compiler
generation; calls, vectors, ABI, async/TMA, fences, and unsupported address
spaces must refuse.

### 4. Dynamic physical-pipe model

Use candidate domains, scoreboard/resource availability, initiation interval,
latency, and occupancy to compute model-local schedules. Preserve ambiguity;
do not use NCU as route labels or tuning data.

**Current:** candidate domains plus `candidate_earliest_completion_v1_model_only`
and model-local candidate issue/completion bounds. The output remains a
feasible-range diagnostic, not an observed route.

**Acceptance:** deterministic route/frontier output, uncertainty accounting,
held-out aggregate validation, and explicit refusal when route identity is not
observable. Exact silicon pipe IDs remain open without an independent source.

The next model version should emit a candidate set and earliest/latest feasible
issue/completion interval per dynamic instruction. Add SMSP/warp occupancy,
scoreboard readiness, control stall/yield/barrier/reuse constraints, memory
response events, initiation interval, in-flight capacity and issue-slot
constraints. If aggregate held-out counters leave multiple assignments, emit
`route_ambiguous` with bounds/entropy; if none fit, emit
`route_contract_mismatch`. NCU/CUPTI aggregate metrics may calibrate or score
resource pressure on disjoint microbenchmarks, never label a dynamic record.
Only a content-bound external `(function_uid, pc, warp, ordinal, route,
issue_cycle)` witness could close exact route identity.

### 5. External parser and timing comparison

Keep three evidence roles separate. **NCU measurements on the local RTX 3090
Ti are the detailed hardware ground truth** for this simulator. Published or
previously collected Accel-Sim results are an independent, aggregate target
lane: each row must retain its paper/source, workload, GPU,
simulator revision/configuration, metric name, endpoint, units, uncertainty,
and the accuracy error ceiling / throughput or speedup floor that this project
must meet or surpass. They are
not per-kernel ground-truth labels. A pinned Accel-Sim executable is useful for
an independent parser/representation smoke, but its card and cycle counter are
not a replacement for NCU ground truth.

**Current:** pinned Accel-Sim parser equality includes a source-compared,
zero-mask canonical external `EXIT`; the adapter rejects full-mask terminal
projection and the external reader rejects trailing record fields. The
common-input probe aligns nonterminal thread-instruction counts. Its optional
executable cycle output is retained only as parser/compatibility provenance;
the local NCU target remains the scoring endpoint. Literature aggregates are
not silently converted into a same-card cycle label.

**Acceptance:** canonical record equality and aligned instruction/termination
counts; a content-bound literature aggregate manifest with source/workload/
hardware/metric/units and explicit accuracy ceilings / throughput-speed floors;
and a separate NCU
campaign whose simulator output uses the same workload, metric domain, and
endpoint. A result is a valid win only when it meets or surpasses the declared
ceiling/floor in the corresponding metric domain; unmatched rows remain
`lower_bound_reference_unmatched`, not ground truth.

The first E10-L evaluator slice is now implemented at
`tools/evaluate_accelsim_literature_floor.py`. Against the retained development
holdout it reports `scoring=false` and `not_evaluable_*` for every target,
because the table has incomplete-tail status and the active clock protocol is
not admitted. This is a real refusal/attainment boundary, not a substitute
score.

**Remaining E10 parser/E10-P gates:** retain parser/executable provenance and typed refusals
for unknown opcodes, missing trace files/counters, nonzero stderr, terminal
projection, and header drift. Build the literature aggregate manifest from the
already-collected Accel-Sim papers/results and record the accuracy ceiling or
throughput/speed floor for every declared domain. Separately run the local NCU campaign
and compare simulator counters/frontiers to NCU under the verified clock
regime. Do not use an RTX3070 Accel-Sim cycle or a local model timestamp as a
direct NCU timing ratio.

### 6. Locked-clock validation

Independently record GPU name, UUID, compute capability, driver, device index,
clock/power/temperature state, trace/card/control/output digests, and repeated
measurements under explicitly locked clocks.

**Current:** the user supplied successful `-lgc 1350`/`-lmc 10501` command
acknowledgements; read-only verification currently observes the RTX 3090 Ti at
SM 1350 / memory 10501 in P0. A fresh 30-sample active CUDA poll held exactly
1350/10501 in P0, so the lock is now a campaign-ready prerequisite. A prior
44-sample ALU ramp nevertheless produced a real transient to memory 10251 MHz;
the readiness tool therefore remains conservative and the old run remains
non-scoring. Treat the operator acknowledgement as external evidence (the
agent cannot reproduce `sudo` noninteractively), and require at least 20 exact
active samples for every new profiled kernel, rejecting any drift.

The device-exposure prerequisite is now restored. A fresh query sees the
operator's RTX 3090 Ti (UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, SM86,
PCI `00000000:01:00.0`, driver 615.71.09) and observes SM 1350 MHz / memory
10501 MHz at idle. The first new active stress campaign is still **not** a
locked-clock validation: 150 samples held SM=1350, but 43 samples changed
memory to 10251 MHz and P-state P2 while the kernel was active (107 samples
were 10501/P0). This is evidence of an under-load clock/power-state transition,
not a reason to average the two regimes. Resolve or explicitly explain the
driver application-clock semantics, then rerun the active-window check and
reject any profiled kernel with a non-target sample. The raw poll is
`/tmp/locked_clock_campaign_1790583918638322726.csv` and the readiness probe
reports `target_clocks_not_stable` for this campaign.

An NCU probe with `--clock-control none` completed, but independent NVML
samples showed the profiled startup interval at memory 10251 MHz/P2 before
recovery to 10501/P0. Its cycles/time are therefore non-scoring. The next
campaign must warm up, wait for at least 20 consecutive exact active samples,
then monitor NVML through every profiled kernel and reject any drift.

**Acceptance:** verified lock acknowledgement, stable clock samples, repeated
held-out kernels, variance report, and independent simulator comparison.

## Current non-completion boundary

The repository currently has bounded mechanisms for every seam above, but the
native completion bars for payload values, full memory ordering, exact physical
routes, unrestricted compilation, locked-clock timing validation, and aligned external timing
are not met. The goal must remain active until those bars are proven or an
external prerequisite is formally resolved.
