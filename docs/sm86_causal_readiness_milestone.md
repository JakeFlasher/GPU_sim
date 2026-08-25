# Completed milestone: sm_86 causal warp readiness and async-barrier handoff

**Status:** completed 2026-08-17; diagnostics retained, fixed CTA-latency
hypothesis falsified  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Baseline campaign:**
`results/captures/sm86_cycle_residual_20260817_v1`  
**Final campaign:**
`results/captures/sm86_async_barrier_readiness_20260817_v4`  
**Reproducible reduction:**
`results/captures/sm86_async_barrier_readiness_20260817_v4/analysis/README.md`  
**Research basis:**
[GPU simulator notes synthesis](gpu_sims_notes_research_synthesis_20260817.md)

## Completed outcome

The milestone delivered versioned, reason-tagged readiness diagnostics without
changing replay timing. The simulator now preserves dependency origin, async
source versus local/SM destination completion, CTA barrier generations and
arrival spread, rendezvous exposure, and ready-but-not-selected delay. All 36
final traces replay twice to byte-identical results with exact instruction
cardinality and readiness schema 1.

The full live campaign comprised 36 correctness cells, 112 foreground NCU
profiles, seven SourceCounters sentinels, 36 separate NVBit captures, and two
fully decoded static SASS sidecars. Every correctness run passed; all NVBit
captures have exact tracer cardinality and zero unsupported, unmapped, or
cross-allocation records. The calibrated independent gap is 14 executed SASS
instructions and 63.9998 base-clock cycles per group.

The identification result does not transfer. Barrier-only cost is 14.00,
18.00, and 28.00 active-SM cycles/group at W2, W4, and W8. The corresponding
DRAM async/barrier interactions are +1.78, +72.97, and +230.53 cycles/group,
and the L1 interaction changes sign at W8. Therefore no fixed CTA timing value
was added to `sm_86`; the readiness component remains, and the falsification is
part of the result.

The six-workload production predictions remain exactly 1,092 / 8,906 / 8,906 /
5,846 / 2,365 / 12,448 cycles, preserving 6/6 coverage and 33.403% active-SM
MAPE. Median process-level host replay changed by +0.175 ms (+5.44%) on the
3.2 ms non-async sentinel and -0.256 ms (-1.66%) on the 15.4 ms async sentinel.
The small-control percentage exceeds the 3% target, but the representative
async path improves; the measured sub-millisecond diagnostic tradeoff is
accepted for this instrumentation milestone.

The successor is
[sm_86 barrier pipeline and launch-chain cache handoff](sm86_barrier_pipeline_cache_handoff_milestone.md).

## Goal

Build an event-driven causal readiness layer that can answer, for each warp and
CTA, why the next instruction was not issuable and which completion relation
made it issuable. Use a controlled live GPU experiment to separate:

1. L1TEX/async source completion;
2. LDGSTS destination delivery and aggregate service;
3. compiler dependence-counter waits;
4. CTA barrier rendezvous;
5. execution-pipeline availability; and
6. a ready warp losing scheduler selection.

Then promote only the smallest held-out-supported timing behavior into the
`sm_86` architecture model. No NCU target value becomes a replay input.

## Why this is the nearest milestone

The previous milestone closed exact LDGSTS destination projection and selected
an independently measured aggregate destination rate of two wavefronts per
three SM cycles. The resulting six-workload active-SM MAPE is 33.403%. The
largest remaining live signal is localized at `BAR.SYNC` instructions that
immediately follow async-copy waits, but NCU classifies those samples as
long-scoreboard rather than CTA-barrier stalls.

The current code contains all relevant timestamps but erases their identities:

```text
control/dependence ready ----+
async-group ready -----------+--> max --> earliest issue time
pipeline ready --------------+

CTA barrier arrival -----------------> no issue time until CTA release
```

Without a decomposition, adding an issue-port limit, a dual-issue rule, or a
fixed barrier latency would be confounded. Readiness attribution is therefore
both an essential component and a prerequisite for the next resource models.

## Scope

### Included

- reason-tagged warp readiness for known and unresolved completion conditions;
- repeated CTA-barrier generation, arrival, and release accounting;
- async-group source/destination limiting-cause accounting;
- ready-but-not-selected scheduler accounting;
- compact per-SM and per-launch diagnostic output;
- one controlled CUDA probe family with full NCU/NVBit/SASS coverage;
- a difference-in-differences analysis of async wait and CTA barrier coupling;
- architecture-card closure only for values independently supported by the
  experiment; and
- replay of the existing six-workload cohort after any model change.

### Excluded

- a generic register scoreboard;
- per-cycle global event generation or one queued event per stall cycle;
- a per-workload latency fit;
- changing the trace to reproduce counterfactual schedules;
- Tensor Core numerical/bit-accurate execution;
- broad dual-issue or all-pipeline calibration before readiness is observable;
- copying `sm_86` behavior into `sm_120`; and
- treating simulated causal intervals as numerically identical to NCU's
  sampled warp-stall counts.

## Design invariants

1. **Compiler controls remain authoritative.** Dependence is represented by
   the decoded SASS stall/set/wait controls already used by `sim-sm`.
2. **CTA and dependence barriers remain distinct.** A dependence counter waits
   for a producer completion; a CTA barrier waits for participating sibling
   warps.
3. **The event-driven shape is preserved.** Diagnostics accumulate at issue,
   completion, barrier, scheduler, and known-time-jump boundaries.
4. **Hot scheduler state stays compact.** New detailed state belongs in cold
   sidecars where practical, following the existing async-copy design.
5. **Attribution is deterministic.** Ties use an explicit stable rule and
   expose the tied causes; hash-map iteration cannot choose the result.
6. **Observed values remain outside replay.** NCU data selects and validates an
   architecture hypothesis; a trace replay never reads the campaign labels.
7. **Unknown behavior stays typed.** A missing `sm_120` mechanism is reported,
   not filled with an Ampere default.

## Deliverable 1: causal readiness representation

### Readiness constraints

The scheduler-facing query should retain the constraints it currently reduces
to one `Option<u64>`. Exact Rust names may change during implementation, but the
semantic result must distinguish:

| Constraint | Current source | Required diagnostic identity |
|---|---|---|
| Control eligibility | `WarpState::eligible` | control-stall/yield availability |
| Known dependence completion | `barrier_clear[]` | dependence-counter index and producer class |
| Unresolved dependence completion | `barrier_pending[]` | unresolved counter/source response |
| Async group wait | `async_wait_clear_time` | limiting group and source/destination arm |
| Tensor/resource availability | `tensor_pipe_ready[]` | pipeline class and ready time |
| CTA rendezvous | `at_block_barrier` | block and barrier generation |
| Scheduler competition | `pick_warps` | ready selection opportunity lost |

A suitable semantic shape is:

```text
WarpReadiness {
    terminal_state,
    control_until,
    dependence_constraints[],
    async_constraints[],
    pipeline_until,
    cta_barrier_generation?,
    earliest_known_issue_time?,
    limiting_causes[],
}
```

`earliest_known_issue_time` remains absent while a required source response or
CTA arrival is unknown. It is not replaced with an arbitrary large cycle.

### Cause vocabulary

The public aggregate should use a stable, architecture-neutral vocabulary:

- `control_stall`
- `dependence_counter_read`
- `dependence_counter_non_memory_write`
- `dependence_counter_shared_write`
- `dependence_counter_l1tex`
- `async_source`
- `async_destination_local`
- `async_destination_sm_service`
- `cta_barrier_rendezvous`
- `pipeline_tensor`
- `eligible_not_selected`
- `no_instruction` or terminal state, retained separately from a stall

If a producer cannot be classified from existing state, report a typed
`dependence_counter_unknown_origin` rather than guessing.

### Limiting and overlapping causes

One warp can have several simultaneous constraints. The implementation must
retain two different views:

1. **Constraint exposure:** timestamps/counts for every active relation. These
   may overlap and must not be summed as elapsed kernel time.
2. **Frontier-limiting delay:** an exclusive interval assigned to the
   constraint that advances the warp's issue frontier. Ties are retained as a
   tie count and resolved with a documented stable order for the exclusive
   total.

This avoids both common errors: double-counting overlapping waits and erasing
the fact that a non-limiting dependency was present.

### Event-driven accumulation

The actor already knows the next relevant issue/completion time. When it jumps
from `t` to `next_t`, diagnostics can charge `next_t - t` to the current
frontier-limiting state without stepping through every cycle. CTA intervals are
closed at the release event. Async intervals are closed when the final source
or destination-service completion is assigned. Scheduler competition is
counted at actual issue opportunities, not synthesized for skipped cycles.

The default output is aggregate. An unbounded per-instruction event log is not
part of this milestone.

## Deliverable 2: CTA barrier generations

`BlockState` currently retains only `waiting` and releases when it equals
`unfinished`. Extend its cold diagnostic state so repeated barriers cannot
collapse together:

- monotonically increasing barrier generation per resident CTA;
- first and last warp arrival times;
- arrival count and participating unfinished warp count;
- release time;
- arrival spread, `last_arrival - first_arrival`;
- per-warp rendezvous exposure, `release - warp_arrival`; and
- count of releases where warp retirement changed the participant set.

The existing one-cycle release transition remains the baseline execution
semantics. A new fixed CTA-barrier latency is added only if the factorial probe
finds a stable held-out residual that cannot be explained by arrival spread,
source completion, or scheduler competition.

## Deliverable 3: async completion decomposition

`AsyncCopyGroup` already stores `source_completion_time`, destination work,
aggregate destination completion, and final completion. Preserve these arms in
the wait diagnostic:

- groups committed;
- groups waited upon;
- source-pending response count;
- source-ready timestamp;
- warp-local destination-ready timestamp;
- aggregate SM destination-service timestamp;
- limiting arm at wait release; and
- latency hidden between commit and the first consuming wait.

For a wait allowing `N` pending groups, attribute only the committed prefix
that the instruction actually requires. Do not charge younger allowed groups.

## Deliverable 4: public diagnostic output

The direct runner should emit a versioned optional readiness section under the
launch timing result. The exact serialization can follow existing Rust types,
but it must provide at least:

```json
{
  "schema_version": 1,
  "limiting_warp_cycles": {
    "async_source": 0,
    "async_destination_sm_service": 0,
    "cta_barrier_rendezvous": 0,
    "eligible_not_selected": 0
  },
  "constraint_exposure": {},
  "selection_losses": 0,
  "async_groups": {
    "committed": 0,
    "source_limited": 0,
    "destination_limited": 0
  },
  "cta_barriers": {
    "generations": 0,
    "total_arrival_spread_cycles": 0,
    "max_arrival_spread_cycles": 0
  }
}
```

The example is a semantic minimum, not a frozen field layout. Every counter is
checked for overflow. Per-SM aggregation and launch folding must be independent
of map iteration order. Existing result fields and instruction/traffic counts
must remain unchanged for the same replay.

The documentation must call these values **simulator causal diagnostics**. They
are compared directionally with NCU stall categories, not asserted to equal
sampling counts or sampled warp-cycles.

## Deliverable 5: controlled live probe

### Kernel construction

Create `workloads/research_probes/async_barrier_overlap/` from the already
validated async-cross-warp probe shape. The timed loop should retain:

- cache-all non-bulk `cp.async` lowering to `LDGSTS`;
- a fixed number of copies and bytes per committed group;
- disjoint, valid shared-memory destinations;
- the same CTA shape within the identification set;
- runtime source stride/mask arguments so locality variants use the same
  binary and address-generation instructions;
- a correctness checksum outside the measured dependence chain; and
- a long live variant plus a 64-group NVBit trace variant.

The primary comparison must hold destination wavefront work, group count, CTA
shape, registers, and dynamic async instruction count fixed while changing
only source locality or the independent gap. Protocol controls without async
copy are expected to lack destination work and are used only in the factorial
subtraction.

### Source-latency conditions

Cache residency is an observed property, not a label assumed from allocation
size. Start with these constructions and retain a cell only if NCU traffic
confirms the intended ordering:

| Locality key | Construction | Required measured evidence |
|---|---|---|
| `l1_reuse` | repeatedly reuse a pre-warmed 512 B source region | high L1 hit/reuse, lowest lower-level sectors |
| `l2_rotation` | rotate through a region larger than effective L1 and smaller than 6 MiB L2 | more L1 misses, predominantly L2-resident source traffic |
| `dram_stream` | rotate/stream through a region larger than L2 without immediate reuse | highest DRAM sectors and source latency |

Initial region sizes are probe parameters, not architecture conclusions. If
the measured hierarchy does not separate monotonically, adjust the regions or
pre-warm strategy and report the failed construction.

### Protocol factors

Use a two-factor core plus one asymmetric stress case:

| Key | Async copy + required wait | CTA barrier | Purpose |
|---|---:|---:|---|
| `c00_baseline` | no | no | fixed loop/address/control overhead |
| `c01_barrier_only` | no | yes | CTA rendezvous without L1TEX source dependency |
| `c10_wait_only` | yes | no | async source/destination completion without CTA rendezvous |
| `c11_wait_barrier` | yes | yes | target coupled chain |
| `c12_producer_consumer` | producer warp only | yes, all warps | amplify arrival asymmetry; diagnostic, not fitted |

The independent-gap factor is inserted between async commit and the required
wait. Use `gap_0` and one instruction-stable arithmetic chain near 64 cycles.
Confirm its executed SASS count rather than relying on source-level operation
count.

### Configured matrix

The bounded identification matrix is:

| Stage | CTA warps | Protocols | Locality | Gap | Cells |
|---|---:|---|---|---|---:|
| Identification controls | 4 | `c00`, `c01` | n/a | 0, 64 | 4 |
| Identification async factorial | 4 | `c10`, `c11` | L1, L2, DRAM | 0, 64 | 12 |
| Asymmetric diagnosis | 4 | `c12` | L1, DRAM | 0, 64 | 4 |
| Held-out CTA shapes | 2 and 8 | `c00`, `c01` | n/a | 0 | 4 |
| Held-out async extremes | 2 and 8 | `c10`, `c11` | L1, DRAM | 0 | 8 |
| Held-out asymmetry | 2 and 8 | `c12` | L1, DRAM | 0 | 4 |
| **Total** |  |  |  |  | **36** |

The L2 cells, alternate CTA shapes, and producer/consumer cells are held out
from any timing-parameter selection. If the W3-W4 transition seen in the prior
destination-service probe remains unexplained, add a focused W1-W8 sweep only
for the two sentinel `c10`/`c11` cells; do not expand every factor.

### Live collection

For each runnable cell:

1. run the ordinary correctness check;
2. retain the launch arguments, source-region parameters, register/spill data,
   and static SASS;
3. collect three foreground NCU repeats with `--clock-control base`;
4. increase to five repeats for a cell whose selected cycle metric has CV above
   2%, and report the variability;
5. collect broad metrics and exact LDGSTS/source traffic;
6. collect SourceCounters for the sentinel L1/DRAM `c01`, `c10`, `c11`, and
   `c12` cells;
7. capture the reduced variant once with NVBit in a separate launch;
8. extract static control bits and require zero unknown encodings; and
9. replay each trace twice and require byte-identical output.

NCU metrics should include, where supported by the installed 2026.1 build:

- `sm__cycles_active.max`
- `gpc__cycles_elapsed.max`
- `smsp__warps_active.avg.per_cycle_active`
- `smsp__warps_eligible.avg.per_cycle_active`
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio`
- exact LDGSTS instruction, source-sector, and shared-wavefront counts;
- L1 source hit/miss sectors, L2 read sectors, and DRAM read sectors; and
- SourceCounters SASS attribution for the four stall classes above.

Metric availability is checked directly with NCU. A renamed or unavailable
metric is recorded and replaced only with an equivalent metric whose unit and
scope are explicit.

### SASS and trace invariants

Across locality cells of the same protocol, require:

- the same kernel binary;
- the same async-copy, commit, wait, and CTA-barrier instruction sequence;
- the same executed dynamic instruction count;
- the same LDGSTS destination wavefront count;
- the same CTA geometry and occupancy inputs; and
- no local-memory spill traffic.

The intended varying quantities are source addresses, measured cache/DRAM
route, and the explicitly selected independent-gap path.

## Analysis

Let `C[p, l, g, w]` be the median active-SM cycles per timed group for protocol
`p`, locality `l`, gap `g`, and CTA warp count `w`, after subtracting the fixed
prologue/epilogue measured by the in-kernel samples.

The non-additive async/barrier interaction is:

```text
I[l, g, w] =
    (C[c11, l, g, w] - C[c10, l, g, w])
  - (C[c01, -, g, w] - C[c00, -, g, w])
```

The source-locality effect without a CTA barrier is:

```text
S[l2, l1, g, w] = C[c10, l2, g, w] - C[c10, l1, g, w]
```

The overlap hidden by independent work is:

```text
H[p, l, w] = C[p, l, gap_0, w] - C[p, l, gap_64, w]
```

Compare these slopes with simulator deltas, CTA arrival spread, async
source-limited/destination-limited group counts, and scheduler selection
losses. Do not regress absolute hardware cycles directly onto a collection of
simulator stall counters and call the coefficients architecture parameters.

Parameter identification may use only the W4 L1/DRAM factorial cells. L2,
alternate CTA shapes, and asymmetric producer/consumer cells test transfer.

## Model-promotion rules

A timing behavior may enter the `sm_86` execution model when all of the
following hold:

- the causal mechanism is named and represented in simulator state;
- the corresponding live factor changes in the expected direction;
- traffic/SASS checks show that an unintended mechanism did not change;
- one architecture-local value or rule explains the identification cells;
- the rule predicts the sign and ordering of held-out deltas;
- it does not use workload identity, NCU cycles, or campaign keys at replay
  time; and
- it does not worsen the held-out probe aggregate while improving only the
  fitting cells.

If no new latency passes these conditions, keep the diagnostics and publish the
falsification. The milestone still delivers an essential observability
component and identifies scheduler/pipe work as the next residual.

Architecture-card resource closure is part of promotion. The existing
`barrier_latencies.read` and `.non_memory_write` fields describe compiler
dependence completions, not CTA rendezvous. Do not reuse either field for a CTA
barrier. Add a distinct architecture capability only if a distinct CTA
mechanism is measured.

## Implementation sequence

1. Add focused readiness snapshots and cause identities around
   `WarpState::ready_at` and `SmCore::issue_ready_at` without changing timing.
2. Add CTA generation/arrival/release state and unit tests for repeated
   barriers, retirement, and asymmetric arrivals.
3. Add async-group source/destination limiting-arm diagnostics and wait-prefix
   tests.
4. Add scheduler selection-loss accounting and deterministic aggregation.
5. Expose the optional versioned readiness diagnostics in `sim-direct-run`.
6. Build the CUDA probe and validate correctness, SASS, registers, spills, and
   traffic invariants.
7. Run the 36-cell NCU/NVBit campaign and analyze factorial/held-out deltas.
8. Implement only the supported sm_86 timing rule, if one survives.
9. Replay the six-workload residual cohort and report both old and new results.
10. Update architecture/component/status documentation with the measured
    conclusion.

Instrumentation-only changes are tested before live calibration so a timing
delta cannot be hidden inside the diagnostic implementation.

## Acceptance criteria

### Simulator correctness

- Every current readiness arm has a typed cause or explicit unknown-origin
  result.
- Repeated CTA barriers have distinct generations and correct arrival/release
  intervals.
- Async wait diagnostics honor `wait_prior`/maximum-pending-group semantics.
- Instrumentation-only replay preserves cycles, instruction cardinality,
  traffic, cache counters, and terminal event time exactly.
- Repeated replay produces byte-identical diagnostic output.
- Overflow and absent-ready-time paths fail with typed errors; they do not
  saturate silently.
- No per-cycle global scan or per-stall-cycle event is introduced.

### Live evidence

- All retained cells pass correctness and have zero unknown control encodings.
- Locality ordering is established from measured L1/L2/DRAM traffic.
- Identification cells have three repeats and no more than 2% cycle CV, or the
  higher measured variance is explicitly reported after five repeats.
- SourceCounters distinguish long-scoreboard, barrier, wait, and not-selected
  behavior on representative cells.
- NVBit traces have full block coverage, zero unsupported/unmapped records, and
  exact replay cardinality.

### Scientific result

- The report presents `S`, `H`, and `I` deltas with repeat dispersion.
- Simulator causal components predict the sign/order of held-out cells before
  any timing rule is promoted.
- Any promoted rule is architecture-local and workload-independent.
- The complete six-workload cohort is rerun and scored against the current
  33.403% active-SM MAPE baseline; improvement is a target, not a result to
  manufacture.
- A falsified CTA-latency hypothesis is reported as such and does not block
  retaining the readiness component.

### Engineering cost

- Focused Rust/Python/CUDA checks pass, followed by ordinary workspace tests.
- `cargo fmt --check` and `git diff --check` pass.
- Median host replay cost on one non-async and one async sentinel changes by no
  more than 3% unless a measured, explained tradeoff is accepted.
- `sm_120` continues to return a typed gap for architecture-local behavior not
  measured there.

## Expected output location

```text
workloads/research_probes/async_barrier_overlap/
results/captures/sm86_async_barrier_readiness_<date>_v1/
  metadata/
  ncu/
  nvbit/
  sass/
  simulator/
  analysis/README.md
```

No administrative command was required for this milestone. If a future
profiler or clock-control step is blocked by host permissions, give the
operator the exact `sudo ...` line to run rather than executing it implicitly.

## Completion verification

The frozen v4 reducer now derives and enforces the campaign-integrity claims
from the saved artifacts. It validates all 112 NCU capture contracts, 36
seven-trial correctness results, 36 one-CTA NVBit payloads and traces, all
three empty NVBit diagnostic sidecars per cell, two 504-instruction static
decodes, 36 byte-identical simulator replay pairs, and six byte-identical
production replay pairs. Every simulator result has full CTA coverage, exact
issued/record cardinality, and readiness schema 1. Per-cell evidence is in
`analysis/cells.csv`; aggregate predicates are in `analysis/summary.json`.

The complete feature-unified Rust workspace tests pass, as do Rust formatting,
12 focused Python parser/scoring tests, campaign/reducer bytecode compilation,
the final release replay sweep, and `git diff --check`. Workspace Clippy exits
successfully with existing advisory test/Rust-1.96 warnings and is not claimed
to be warning-free. No `sudo` command was needed.
