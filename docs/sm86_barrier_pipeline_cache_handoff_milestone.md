# Completed milestone: sm_86 barrier pipeline and launch-chain cache handoff

**Status:** complete 2026-08-17; successor is execution-port/control-pipeline closure  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[causal warp readiness and async-barrier handoff](sm86_causal_readiness_milestone.md)  
**Final predecessor evidence:**
`results/captures/sm86_async_barrier_readiness_20260817_v4/analysis/README.md`

**Final milestone evidence:**
`results/captures/sm86_barrier_pipeline_20260817_v1/analysis/README.md`,
`results/captures/sm86_barrier_pipeline_coupled_20260817_v1/analysis/README.md`, and
`results/captures/sm86_barrier_pipeline_multicta_20260817_v1/analysis/README.md`

## Goal

Build the two missing mechanisms exposed by the predecessor campaign:

1. an architecture-local CTA-barrier issue/pipeline resource, distinct from
   compiler dependence counters and CTA rendezvous; and
2. an explicit launch-chain replay mode that can carry a fully drained L2
   cache state from a traced prewarm launch into a traced target launch.

Then repeat the async/barrier transfer test and add bounded multi-CTA scheduler
contention. Promote a timing rule only if the same mechanism predicts held-out
CTA widths, source routes, and block counts without reading NCU values or
campaign identities during replay.

## Why these are the next essential components

The final v4 live campaign established three facts that the current model
cannot represent with one latency constant:

- barrier-only cost is 14.00, 18.00, and 28.00 active-SM cycles per group for
  W2, W4, and W8 respectively;
- the DRAM async/barrier interaction is +1.78, +72.97, and +230.53
  cycles/group at those same widths, while the L1 interaction changes sign;
  and
- simulator CTA arrival spread predicts the direction of the producer/consumer
  effect, but target-only cold replay makes the L2 and DRAM trace cells
  identical even though live NCU traffic separates them.

The first two observations reject a fixed CTA-rendezvous latency. They instead
point to barrier issue throughput interacting with arrival skew and scheduler
occupancy. The third is a construction gap: hardware prewarm changes cache
state before the measured target, while `replay_trace_kernel` creates a new
`L2Model` for every launch.

The existing code already provides the right implementation seams:

- `simulate_kernel_shared_with_residency` accepts a caller-owned mutable
  `L2Model`, runs on a transactional snapshot, drains all events, and commits
  the final cache state only on success;
- `replay_kernel_with_schedule` currently constructs that model anew at the
  per-launch boundary;
- `PipelineClass` already gates Tensor-Core issue with a resource-ready
  timeline, while CTA barriers remain `DispatchOnly`; and
- readiness schema 1 now reports dependence origin, async completion arms,
  CTA generations/arrival spread/rendezvous, and eligible-not-selected delay.

This milestone extends those mechanisms rather than inventing a parallel
replay path.

## Scope and ordering

The work has three ordered stages:

1. **Launch-chain L2 handoff:** make warm-state experiments replayable and
   preserve the existing cold-launch default exactly.
2. **Barrier pipeline identification:** measure and implement only the
   architecture-local resource shape supported by the barrier-only sweep.
3. **Coupled validation:** rerun async/barrier cells with preserved warm state,
   then add multi-CTA contention to test scheduler selection.

The cache handoff is implemented before fitting the barrier resource because
the L2 and DRAM validation cells otherwise collapse to the same simulated
input.

## Completion summary

All ordered stages are complete.

- **Launch-chain cache state:** the transactional same-stream session retains
  only drained L2 state. A traced 2 MiB seed converts exactly 1,024 target
  sectors to hits and removes 1,024 DRAM reads while cold launch scope remains
  the default.
- **Barrier-only identification:** 33 cells, 99 NCU profiles, six
  SourceCounters profiles, and 33 NVBit captures select one SM-shared barrier
  arrival service at two cycles per warp. Full W1-W16 shape MAE is 0.103
  cycles/generation and held-out MAE is 0.091.
- **Consecutive generations:** the service predicts the complete width slope.
  A 19.016-cycle common residual cannot distinguish barrier re-arm from four
  ptxas-scheduled ALU/control instructions between BARs, so no re-arm or fixed
  release term is promoted.
- **Coupled warm routes:** 18 cells, 62 NCU profiles, 18 NVBit captures, and 36
  cold/warm replay pairs establish live L1 < L2 < DRAM at W2/W4/W8 and insert
  the modeled L2 middle route from traced state alone.
- **Multi-CTA transfer:** six cells hold eight warps and 96 KiB per SM constant
  while varying 1×8, 2×4, and 4×2 CTA shapes. Hardware and replay share the
  same ordering with 0.032 cycles/generation slope MAE; the common 4.004-cycle
  offset is retained.
- **Regression:** all 36 predecessor traces retain identical traffic and
  deterministic replay. Five production predictions are unchanged; attention
  moves +8 cycles toward observation, for 33.394% six-workload MAPE.

Readiness schema 2 reports barrier dispatch, service arrivals, accepted work,
busy cycles, first/last timestamp sums, issue/arrival spans, CTA rendezvous,
and scheduler selection loss with checked aggregation. `sm_120` remains a
typed capability gap. The configured successor is
[sm_86 execution-port and control-pipeline closure](sm86_issue_port_control_pipeline_milestone.md).

Final ordinary validation passes `cargo test --workspace --all-targets` (35
`sim-arch`, 8 `sim-engine`, 113 `sim-mem`, 324 `sim-sm`, 138 `sim-trace`, all
direct/campaign binaries and integrations), `cargo fmt --all --check`,
`git diff --check`, workspace Clippy with its documented advisory warnings, 12
focused Python parser/scoring tests, Python compilation of every campaign and
reducer, release replay of every retained cohort, and live CUDA correctness
smokes for both one-CTA and multi-CTA probes. No privileged command was needed.

## Stage 1 completion: launch-chain L2 handoff

Stage 1 is implemented and live-validated. `sim-sm` now exports an ordered
`StreamL2ReplaySession` bound to one exact trace root, replay configuration,
architecture-card placement, device, stream, and `(kernel_id,
function_unique_id)` sequence. Each launch is staged on an L2 snapshot and the
cache plus cursor commit together only after replay, drained-state validation,
checked counter-delta calculation, and cumulative reconciliation succeed.
Every ordinary one-launch entry point still creates a fresh L2.

`sim-run` exposes the required `launch` and `stream-l2` scopes. Stream-L2 mode
requires the shared hierarchy, rejects launch filtering and mixed
device/stream traces, emits explicit seed metadata, and reports cumulative
session L2 counters. No name, campaign key, NCU route, or observed hit rate is
read by replay.

Focused tests cover:

- cold-to-warm target conversion while rebuilding L1;
- partial-sector carry-over and LRU recency across a boundary;
- transactional failure/retry and typed order/configuration/root/stream
  mismatch;
- zero pending sectors, checked per-launch/cumulative counters, exhaustion;
  and
- repeated byte-identical sessions plus unchanged default serialization.

The live trace under
`results/captures/sm86_launch_chain_cache_20260817_v1` contains a 256-CTA 2 MiB
prewarm followed by the unchanged one-CTA W4 c10 target on device 0, stream 0.
The primary NVBit capture reports two accepted launches, 257 complete CTA
payloads, correct program output, and zero unsupported, unmapped, or
cross-allocation records. Relative to cold-per-launch replay, retained L2
changes the target from 54 hits / 1,042 misses / 1,024 DRAM read sectors to
1,078 hits / 18 misses / zero DRAM read sectors. Target L1 traffic is identical
in both modes. Terminal modeled time changes from 35,010 to 23,554 cycles; this
is a structural route-separation result, not a fitted cycle claim.

Both cache scopes reproduce byte-identically across two runs. Rebuilding the
release runner and replaying the predecessor's complete 36-cell matrix twice
also preserves all 72 prior JSON outputs byte-for-byte. The active work is now
Deliverables 2 and 3: identify the barrier issue-resource shape without using
the newly separated warm-cache result as a timing fit target.

### Explicitly excluded

- a fixed delay added at CTA release;
- reusing `barrier_latencies.read` or `.non_memory_write` for CTA barriers;
- importing NCU hit rates, stall ratios, or cycle labels into replay;
- retaining an undrained fill, DRAM request, MSHR, or dependency response
  across kernel boundaries;
- cross-stream concurrency or kernel overlap in the first launch-chain mode;
- assuming that `warpid_sm % 4` proves physical scheduler placement on the
  live GPU;
- preserving L1 state before a trace-supported SM-placement contract exists;
- a generic register scoreboard; and
- any `sm_120` default derived from Ampere measurements.

## Deliverable 1: explicit launch-chain L2 state

### Public semantic boundary

Add an ordinary replay-session API for a declared, ordered launch sequence.
Exact Rust names may change, but the ownership should be equivalent to:

```text
SharedHierarchySession {
    architecture,
    replay_configuration,
    trace/device/stream scope,
    l2_model,
    completed_launches,
}

replay_next(session, traced_launch) -> per_launch_result
```

The direct runner exposes an opt-in cache scope:

```text
--cache-scope launch       # existing behavior; default
--cache-scope stream-l2    # sequential same-device/same-stream L2 handoff
```

`stream-l2` accepts only the launch order present in one trace. The first
implementation does not infer cross-stream happens-before relations and does
not combine separate trace directories by path convention.

### State that crosses a launch boundary

Only fully committed L2 replacement/sector state and its deterministic
replacement-policy sequence cross the boundary. Before the next launch:

- the prior system event queue is empty;
- all L2 fills and response-bearing requests are complete;
- every per-SM L1, MSHR, dependence counter, async group, CTA, pipeline
  timeline, and scheduler state is destroyed;
- DRAM bandwidth service starts a new non-overlapping launch timeline; and
- L2 observer counters are reported as checked per-launch deltas even though
  physical cache state persists.

This gives the v4 prewarm experiment the state it actually needs without
claiming unsupported inter-kernel overlap or L1 residency.

### Required invariants

1. The session is bound to one `ArchId`, one exact `ReplayConfiguration`, one
   trace root, one device, and one stream.
2. Launches are consumed once in monotonically increasing traced order.
3. Architecture, cache geometry, replacement policy, address mapping, or
   stream mismatch fails with a typed error before cache mutation.
4. A failed launch leaves both the session's cache state and launch cursor
   unchanged.
5. Per-launch L2 counters are checked deltas; their sum equals the session
   cumulative delta and never underflows.
6. Default `launch` scope reproduces every current JSON result byte-for-byte.
7. Cache state contains no NCU-derived route label. Hits and misses arise only
   from traced addresses and the modeled replacement state.
8. Session metadata reports which earlier traced launch seeded a target, but
   kernel names and campaign keys are diagnostic identity, not timing inputs.

### Focused tests

- a two-launch synthetic trace where a prewarm converts target L2 misses to
  hits in `stream-l2` mode but not in `launch` mode;
- partial-sector prewarm followed by a mixed hit/fill target;
- replacement-pressure prewarm that proves recency state, not just valid bits,
  is carried;
- rollback after a deliberately malformed second launch;
- typed architecture/configuration/stream/order mismatch;
- zero in-flight L2 tokens at every successful boundary;
- checked per-launch/cumulative counter reconciliation; and
- repeated launch-chain replay with byte-identical output.

## Deliverable 2: CTA-barrier issue resource

### Separation from rendezvous

A barrier instruction has two distinct relations:

```text
warp ready for BAR
       |
       v
barrier issue resource available ----> actual CTA arrival timestamp
                                            |
all unfinished CTA warps have arrived ------+----> rendezvous release
```

The resource may delay the instruction's issue and therefore change arrival
spread. It must not be modeled as a post-rendezvous latency. Existing
`cta_barrier_ready = release_at` continues to represent release eligibility.

### Candidate resource shapes

The live sweep selects among, rather than presupposes, these architecture-local
hypotheses:

- dispatch-only baseline;
- one barrier issue timeline per SMSP;
- one barrier issue timeline shared by the SM; or
- a shared rational-throughput server with a finite initiation interval.

If two shapes are observationally equivalent in the bounded sweep, retain the
simpler one and label the unidentifiable distinction. Do not add a queue depth
or latency arm that the probe cannot identify.

Once selected, add a distinct architecture capability such as
`cta_barrier_issue`, supported only for `sm_86`. The schedule classifier may
map `OpClass::BlockBarrier` to a barrier pipeline class, but CTA semantics stay
owned by `OpClass`; pipeline classification must not erase the distinction
between arrival and release.

### Diagnostic extension

Readiness schema 2 adds, at minimum:

- barrier instructions issued;
- barrier-pipeline limiting cycles;
- barrier-pipeline constraint exposure;
- issue-server busy cycles or accepted work;
- per-generation first/last issue timestamps; and
- CTA arrival spread after resource gating.

Schema 1 remains readable. Aggregation uses checked arithmetic and stable SM
ordering. No per-cycle event log is introduced.

## Deliverable 3: live barrier-only identification sweep

### Kernel family

Extend `async_barrier_overlap` or add a sibling `barrier_pipeline` probe with
compile-time-specialized kernels:

| Protocol | Loop body | Purpose |
|---|---|---|
| `b00` | control loop, no barrier | dispatch/control subtraction |
| `b10` | one CTA barrier per generation | barrier throughput |
| `b20` | two consecutive CTA barriers | generation/re-arm pressure |
| `b11` | warp-dependent arithmetic before one barrier | controlled arrival skew |

Every kernel uses one CTA, no global memory in the measured interval, no async
copy, no atomics, and no spills. The long form uses enough generations to
dominate launch overhead; the NVBit form is bounded to 64 generations.

### Predeclared matrix

Use W1 through W8 plus W12 and W16. Identification widths are W1, W2, W4, W8,
and W16; W3, W5, W6, W7, and W12 are held out from resource selection.

| Stage | Protocols | Widths | Skew | Cells |
|---|---|---|---:|---:|
| Controls | `b00`, `b10` | 1-8, 12, 16 | 0 | 20 |
| Re-arm | `b20` | 1, 2, 4, 8, 16 | 0 | 5 |
| Arrival skew | `b11` | 2, 4, 8, 16 | 32, 64 | 8 |
| **Total** |  |  |  | **33** |

The skew chain's executed SASS and standalone cycle cost are measured. Warp
slots and CTA geometry are retained from NVBit. Their modulo distribution can
be compared with candidate SMSP models, but the report must call that a model
mapping, not an observed physical scheduler assignment.

For every cell:

1. run seven ordinary correctness/timing trials;
2. collect at least three foreground NCU repeats with base clocks and no cache
   control dependency;
3. increase to five repeats above 2% selected-cycle CV;
4. retain registers, spills, static SASS, executed barrier count, active/eligible
   warps, issue-active instructions, and barrier/wait/not-selected ratios;
5. run SourceCounters on W1/W4/W8/W16 `b10` and W4/W8 `b11` sentinels; and
6. capture/replay the reduced identification and held-out sentinels twice.

Primary observables are barrier-subtracted active cycles per generation,
incremental cost of the second consecutive barrier, and slope changes as CTA
width crosses the four-SMSP distribution. Sampled stall ratios support causal
interpretation but are not cycle targets.

## Deliverable 4: traced prewarm handoff experiment

Add a trace mode that includes both the prewarm launch and the 64-group target
launch in one same-stream capture. Live NCU continues to measure only the
target profiler range; NVBit must retain both launches for replay.

Use three target constructions with identical cache-all LDGSTS destination
work:

| Construction | Seed launch | Expected replay distinction |
|---|---|---|
| L1-sized reuse | traced small-region prewarm | target quickly self-warms; L2 seed is secondary |
| L2 rotation | traced 2 MiB prewarm | target source reads begin as modeled L2 hits |
| DRAM stream | no matching seed | target source reads begin as L2 misses |

Admission requires exact allocation/address continuity between seed and target,
zero unknown SASS encodings, correct payloads, and measured live L1/L2/DRAM
ordering. The target trace is unchanged between cold and warm simulator modes;
only the explicit presence of the traced seed launch changes initial L2 state.

The required first result is qualitative and structural: `stream-l2` must
separate the L2 and DRAM simulator routes and preserve exact destination work.
Absolute cycle fitting waits for that invariant.

## Deliverable 5: coupled and multi-CTA validation

After the first two components pass independently:

1. rerun c10/c11 for W2, W4, and W8 at L1/L2/DRAM with gap 0;
2. keep the predecessor W4 gap-64 cells as overlap checks;
3. add one, two, and four resident CTA constructions with fixed total active
   warps where traced physical warp-slot sets are disjoint; and
4. compare barrier-pipeline limiting cycles, CTA arrival spread, and
   eligible-not-selected delay with NCU ordering.

Multi-CTA replay remains a modeled co-residency claim, as documented by
`CoResidency::Modeled`; it is not relabeled as exact hardware block overlap.

## Model-promotion rules

A barrier resource enters `sm_86` timing only when:

- one predeclared candidate explains identification slopes without a
  workload-specific branch;
- predicted barrier-only cost has the correct ordering for every held-out CTA
  width and no held-out sign error;
- consecutive-generation behavior is consistent with the same resource;
- adding controlled skew changes modeled arrival spread in the live direction;
- warm-state replay separates L2 and DRAM from traced state alone;
- the coupled c10/c11 W2/W4/W8 interaction improves in sign and ordering;
- the six-workload production cohort and all prior contention probes are rerun;
  and
- `sm_120` reports a typed missing capability.

If no candidate transfers, publish the barrier sweep and retain the cache
session plus diagnostic schema 2. A failed timing hypothesis does not justify
discarding either essential mechanism.

## Acceptance criteria

### Simulator correctness

- Default per-launch replay is byte-identical to the current baseline.
- Same-stream launch-chain replay deterministically carries only drained L2
  state and reports checked per-launch counters.
- Session mismatch and rollback paths are typed and mutation-free.
- Barrier issue and CTA release are separate state transitions.
- Readiness accounting identifies barrier-pipeline delay without double-counting
  rendezvous or eligible-not-selected delay.
- No unbounded per-instruction log, per-stall-cycle event, or global cycle scan
  is added.

### Live evidence

- All retained cells pass correctness, SASS decode, spill, and trace-cardinality
  checks.
- Identification cells meet the repeat/CV rule or retain and explain their
  measured variance.
- Executed barrier counts and dynamic control work match the intended protocol.
- Traced prewarm and target launches share the validated allocation/address
  domain, and live hierarchy traffic establishes the intended route ordering.
- NCU and NVBit remain separate foreground processes.

### Scientific result

- The report publishes all 33 barrier cells, held-out predictions, repeat
  dispersion, and rejected candidate shapes.
- No NCU metric, campaign key, or workload name is read by simulator timing.
- Any architecture value is derived from barrier-only identification cells and
  tested unchanged on async and multi-CTA cells.
- Production scoring remains keyed and reports coverage, MAE, RMSE, and MAPE
  against the 33.403% active-SM baseline.

### Engineering cost

- Focused Rust/CUDA/Python checks and ordinary workspace tests pass.
- `cargo fmt --check` and `git diff --check` pass.
- Host replay is remeasured on the non-async and async sentinels; any regression
  above 3% is quantified in milliseconds and either removed or explicitly
  accepted as a measured research tradeoff.

## Expected artifacts

```text
workloads/research_probes/barrier_pipeline/
workloads/research_probes/async_barrier_overlap/
results/captures/sm86_barrier_pipeline_<date>_v1/
  metadata/
  ncu/
  ncu-source/
  nvbit/
  sass/
  simulator/
  analysis/README.md
results/captures/sm86_launch_chain_cache_<date>_v1/
  correctness/
  ncu/
  nvbit/
  simulator/
  analysis/README.md
```

No privileged command is required by the configured work. If NCU clock control
or device permissions later require operator action, stop at that exact step
and provide the user the complete `sudo ...` command rather than running it.
