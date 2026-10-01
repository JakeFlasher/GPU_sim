# SM86 state-complete selective reification refactor

Updated 2026-10-01 from the supplied review findings. This is an implementation
plan, not an execution gate or authority record. It replaces the misleading
use of “causal” for descriptor self-equivalence with a staged, evidence-backed
reification contract. Native replay remains the detailed fallback; unknown
hardware internals remain black-box model variables.

**Phase 1 status (2026-10-01): complete.** The model-local scope is now
documented, abstract service demand composes into an opt-in service frontier,
unknown completion uppers propagate through observer projection, and elapsed
endpoint arithmetic is origin-invariant. Focused semantic/direct-run tests,
the workspace suite, the retained canary, and the descriptor probe pass.

**Phase 2 status (2026-10-01): conservative native-boundary fallback complete.**
The typed transformer now exposes a `TypedRegionContract` containing entry,
span, internal effects, pending completion tokens, observable labels, and exit.
`compare_native_region_contract` checks the contract against independently
observed native boundaries and returns `Preserve` only for complete exact typed
equality; a missing macro boundary, malformed state, omitted field, or unknown
future-relevant state returns a typed exact-replay fallback. The opt-in
reference writer records this decision beside each native boundary and does not
change default replay.

On the retained SM86 canary, dense and event reference traces reconstructed
identically at all 13 event boundaries, while all 13 typed native boundaries
required exact fallback. The reasons included
`shared_l2_transient_state`, `l1_fill_response_identity`, and
`unmodeled_completion_tails`. This is boundary/refusal evidence, not a claim
that the scalar descriptor recurrence is native-equivalent; a future preserve
result still requires a macro-produced complete typed exit.

**Phase 3 status (2026-10-01): complete.** Multi-warp candidate regions now
reify unless a future globally comparable issue witness is supplied; completion
cardinality greater than one is recorded as an explicit exact-path
`CompletionMultiplicity` reification; and fast-frontier output reports
trace/candidate/summarized/reified/uncovered record counts over the full trace.
Safe suffixes after an unmodelled prefix are retained as candidate spans, while
native entry-state closure remains a separate requirement. The descriptor API
also distinguishes residency-only input from an explicit independent-evidence
variant, so producer-supplied dependency booleans alone cannot certify a
region. Workspace tests, the release fast-frontier canary, and the semantic
reference comparator pass. The retained canary reports 54 trace records and
54 uncovered records (no false compression); this is expected because its
predicate population prevents scalar eligibility.

**Phase 4 status (2026-10-01): complete for the model/probe contract.** Service rates now record
whether capacity was already applied by a throughput probe, and typed
admission/service composition makes serial versus parallel overlap explicit.
The v2 fitter requires distinct dependent-latency, independent-throughput,
queue-capacity, occupancy, and contention modes; it refuses an aggregate-only
pool and leaves occupancy/contention unmodeled rather than inventing a hidden
parameter. The current observer path uses the declared serial composition and
unknown-tail propagation; a separate probe-only holdout scorer covers the
identifiable modes and reports occupancy/contention as unmodeled.

**Phase 5 status (2026-10-01): complete for the staged corpus.**
`CacheCandidateSummary` now carries separate temporal-order, fence-scope, and
line-generation domains in addition to topology, read-from, owner, and
completion. The compact trace can contribute only an observed warp-local
instrumentation order when decoded identities are complete; cross-warp/reordered topologies remain
indistinguishable without a supplied witness, and owner/read-from/completion
stay unknown. Generation/fence witnesses can be injected through the typed
constructor without promoting them to physical coherence claims. A separate
cache-history frontier removes a finite upper endpoint for unresolved
cross-SM history while adding no positive delay or coherence-bottleneck claim.

**Phase 6 status (2026-10-01): complete for the current staged corpus; result
is explicitly negative about hardware conclusions.** `tools/evaluate_reification_design.py`
compares detailed replay with service-profile variants, checks native terminal
invariance, ranks only the declared simulator elapsed lower bound, and reports
runtime/RSS. The descriptor probe reports the complementary model-local result:
a single-warp witnessed 16-instruction region collapses to one summary step,
while the three-warp case collapses zero regions and reifies all three.

Three current SM86 canary families have now been captured and predicted before
the NCU join: `atomic_histogram` (calibration), `gemm_hmma_bf16`
(development), and `addr_carry_canary` (held out). Independent NCU
`gpc__cycles_elapsed.max` values were 8,811 / 4,530 / 5,270 cycles; simulator
lower-bound proxies were 2,435 / 1,024 / 432 cycles. All three open intervals
cover their NCU value, with lower-bound-proxy APEs 72.36%, 77.40%, and 91.80%.
The corresponding NCU GPU-duration averages were 4,672 / 2,432 / 2,848 ns.
Runtime was 11.42 / 6.14 / 6.04 ms and peak RSS was 6.08 MiB / 16 KiB / 392
KiB. The post-prediction report includes split coverage, open-width status,
refusal/OOD, runtime, memory, and compression fields; these NCU values were
not calibration labels or selection features.
`tools/run_prediction_manifest.py` is the reproducible pre-NCU materialization
path for these family splits.
The prediction run used a profile emitted by the v2 independent-probe fitter;
the probe document contains latency, throughput, queue-capacity, occupancy,
and contention modes and contains no NCU rows.

Repeating that manifest for service scales 1/2/4 produced monotone model-local
rankings for every split: atomic 2,435/3,972/7,046; tensor async GEMM
1,024/1,040/1,072; and mixed-memory held-out 432/478/570 cycles. Each split
retained 100% open-interval coverage against its independent NCU observation.
Mean lower-bound-proxy APE across variants was 49.11% calibration, 76.92%
development, and 90.64% held out. Mean runtime/RSS were 13.33 ms/5.93 MiB,
6.19 ms/296 KiB, and 6.08 ms/16 KiB respectively. These rankings are
model-local service-capacity rankings; they are not silicon design rankings.

The conservative canary still has zero collapsed records and 54/54 uncovered,
so no hardware architectural conclusion is claimed. The model-local
single-warp probe provides the affordable selective-reification result (16
exact steps replaced by one summary step); real memory-heavy families remain
exact/refused. This is the final conclusion for the staged corpus, not a claim
of native compression or silicon scheduler causality.
The real-family fast-frontier checks were likewise refusal-only: atomic
histogram 4,480/4,480 uncovered, tensor async GEMM 51/51 uncovered, and the
mixed-memory canary 54/54 uncovered, each due to an unsupported predicate or
entry-state condition.
The model-local descriptor probe does provide a 16:1 instruction-to-summary
step ratio for its single-warp witnessed region; this ratio is not reported as
native compression because the real family traces above are refused by
predicate/entry-state guards.

The comparison was performed with the existing
`tools/semantic_events/sm86_reference_compare.py` boundary reconstruction:
`boundary_trace_equal=true`, 13 reconstructed event records, and no first
difference between dense and event replay on the retained canary.

## Research thesis

The defensible contribution is:

> Unknown-preserving, state-complete selective compression of GPU trace regions,
> with exact fallback whenever the boundary state, event order, completion
> effects, or observer endpoint is not closed.

This is a GPU-specific application of skipping/refinement ideas, not a claim of
new silicon scheduling or proprietary-mechanism recovery. A descriptor-only
summary is useful engineering evidence but is not a native-refinement or
hardware-causal certificate.

## Contracts and vocabulary

Keep three evidence levels separate:

1. **Model-local summary** — an independently coded recurrence and its compact
   macro-transition agree on the supplied descriptor model.
2. **Native state refinement** — a macro-transition preserves a declared
   projection of `SmCore`/system boundary state at region entry and exit, with
   all future-relevant fields represented or explicitly refused.
3. **Hardware comparison** — native replay or a reified model is compared with
   independent NCU measurements. This is not implied by (1) or (2).

Use `reification`/`summary` for the implementation surface. Reserve `causal`
for relations derived from an independently observed native order or state
transition. Keep the detailed native replay as the exact fallback.

## Phase 1 — sound model-local contract and interval semantics (complete)

### Deliverables

- Add this plan and update active documentation to state the three evidence
  levels above.
- Relabel the current causal-frontier report/contract as model-local
  descriptor equivalence; preserve the existing binary/API compatibility where
  practical, but make the serialized role explicit.
- Make frontier composition explicit and endpoint-specific.
- Propagate an unknown completion upper bound instead of taking a finite upper
  from unrelated known fields.
- Treat endpoint values as elapsed durations only when start/end share a clock
  and origin; otherwise return an unsupported/unknown projection.
- Add focused tests for unknown tails, origin shifts, service-rate sensitivity,
  and completion multiplicity.

### Completion evidence

- Descriptor-only tests still pass and report model-local status.
- A frontier with `typed_tail.upper = None` cannot produce a finite completion
  upper bound.
- Shifting endpoint start/end by a constant leaves duration unchanged.
- Changing an abstract service profile changes the composed candidate frontier,
  not merely a side-by-side diagnostic field.
- No native or silicon claim is emitted by Phase 1.

## Phase 2 — native boundary refinement and exact fallback

### Deliverables

- Define a typed region contract with entry boundary, trace span, internal
  effects, exit boundary, pending completion tokens, and observable labels.
- Reuse the existing typed/reference boundary projections, including
  `TypedCoreBoundaryState`, queue state, L1/L2 state, response owners, barriers,
  async-copy state, and explicit unknown fields.
- Add a native state-refinement checker that compares the detailed `SmCore`
  state with the reified macro state at region boundaries.
- Refuse a certificate when any future-relevant boundary field is omitted,
  unknown, or has an unmodeled tail.
- Keep reference tracing opt-in and diagnostic; it must not become an execution
  prerequisite.

### Completion evidence

- A real SM86 canary with a scalar region produces a native boundary comparison.
- A RAW dependency, memory response, barrier, or unmodeled tail forces exact
  fallback.
- Matching descriptor recurrences alone are reported as insufficient.

### Phase 2 implementation evidence

- `sim/semantics/src/transform.rs` defines and validates the typed region
  contract, exact native projection comparator, and structured fallback
  reasons.
- `sim/sm/src/system.rs` emits the comparator decision in opt-in reference
  traces. The current fast scalar path supplies no complete typed macro exit,
  so the canary is deliberately refused rather than certified.
- `tools/semantic_events/sm86_reference_compare.py` reconstructs the native
  stream and reports complete versus exact-fallback boundary counts without
  treating a trace digest or receipt as authority.

## Phase 3 — event order, completion effects, and full-span coverage

### Deliverables

- Replace per-warp local sequence sorting with one of: native issue-order
  witnesses, an explicit partial-order scheduler, or a conservative refusal.
- Define equal-time event phases and preserve the engine’s `(time, sequence)`
  order where it is part of the model contract.
- Replace `completion_units` as a count-only field with explicit completion
  effects/tokens; otherwise reify every multiplicity greater than one.
- Allow safe regions to begin after a noncandidate prefix once a valid entry
  boundary is supplied.
- Report candidate, reified, and uncovered spans against all trace records,
  including suffixes; do not report prefix-only collapse as launch coverage.

### Completion evidence

- Same-SMSP equal-sequence counterexamples cannot certify through synthetic
  lockstep ordering.
- Cross-warp synchronization and same-cycle contention have deterministic,
  explicit outcomes or are refused.
- Completion-token counts and pending state agree at region exit.
- Coverage denominators include every dynamic record.

## Phase 4 — connected abstract service and observer model

### Deliverables

- Replace aggregate `cycles/events` fitting with probe modes that separately
  identify dependent latency, independent initiation rate, queue capacity,
  occupancy hiding, and bank/contention effects.
- Make service parameters typed and interval-valued; do not divide by capacity
  twice or infer capacity from one aggregate slope.
- Compose demand into explicit admission/service/completion frontiers with
  overlap and queue state. Alternative candidate pools remain alternatives,
  not simultaneously attainable counts.
- Feed the composed completion frontier into the observer projection.
- Use a duration-aware endpoint contract with observer start/end nuisance terms,
  clock domain, cache policy, serialization, and included/excluded work.

### Completion evidence

- Independent probes identify only the parameters the design can distinguish.
- Service-rate changes affect the composed frontier and observer interval.
- Unknown queue/tail state widens or removes the upper bound.
- No scored workload’s NCU value is used for fitting or model selection.

## Phase 5 — cache candidate semantics and conservative traffic modeling

### Deliverables

- Keep topology overlap, model-cache replay, architectural candidate writes,
  physical owner/read-from state, and hardware completion as separate domains.
- Add temporal order/fence/generation witnesses only where available; never
  promote line overlap to owner, coherence traffic, or DRAM persistence.
- Propagate cache history and service uncertainty into traffic/frontier
  intervals without requiring proprietary routing knowledge.

### Completion evidence

- Two traces with identical topology but different temporal order remain
  distinguishable only when the supplied witness supports it.
- Missing ownership/read-from state remains explicitly unknown.
- Traffic and causal-order outputs are tested separately.

## Phase 6 — design-space conclusion and independent evaluation

### Deliverables

- Use a declared trace-safe design intervention (for example issue width or an
  abstract service capacity) and compare detailed versus reified execution.
- Demonstrate a design ranking or architectural conclusion that is unchanged
  when reification is disabled or when uncertain parameters vary within their
  calibrated intervals.
- Calibrate on independent probes, develop on separate workload families, and
  score untouched families with NCU joined only after prediction.
- Report error distributions, interval coverage/width, OOD/refusal behavior,
  compression ratio, replay/model runtime, and memory cost together.

### Completion evidence

- A nontrivial model-local design evaluation becomes affordable because a
  witnessed 16-instruction single-warp region replaces 16 exact descriptor
  steps with one summary step while preserving the same terminal envelope;
  native memory-heavy families remain exact/refused rather than being counted
  as compression.
- Exact fallback remains available for every unsupported or non-closed region.
- The paper can state precisely what is model-relative, native-refined, and
  hardware-compared.

## Cross-cutting invariants

- No scalar selector may collapse issue, service, completion, visibility,
  writeback, and observer frontiers.
- Unknown upper bounds remain unknown; no finite upper is manufactured from a
  different frontier.
- A producer cannot certify its own dependency-free claim without an
  independently derived witness or a conservative refusal.
- Native replay defaults remain unchanged by opt-in summary/prediction modes.
- Custom frontends never become evidence for native SASS timing.
- NCU remains an independent comparison target, never a same-workload feature,
  calibration label, or implementation-selection criterion.
- Local tests and reference traces are implementation evidence, not silicon
  validation or procedural approval artifacts.

## Primary prior-art anchors

- Genbrugge, Eyerman, and Eeckhout, *Interval Simulation* — mechanistic
  analytical intervals are driven by disruptive events, while detailed memory
  and branch components remain explicit:
  <https://users.elis.ugent.be/~leeckhou/papers/hpca10.pdf>
- Hong and Kim, *An Analytical Model for a GPU Architecture with Memory-Level
  and Thread-Level Parallelism Awareness* — MWP/CWP explicitly model warp
  overlap, occupancy, and memory service:
  <https://faculty.cc.gatech.edu/~hyesoon/hong_isca09.pdf>
- Jain and Manolios, *Skipping Refinement* — formalizes macro steps that skip
  multiple specification states and supplies local proof obligations:
  <https://arxiv.org/abs/1502.02942>
- Accel-Sim remains the native trace-driven baseline; current 2.0 release adds
  Hopper asynchronous features and marks Blackwell support experimental:
  <https://github.com/accel-sim/accel-sim-framework/releases>

## Second-round review queue

1. Obtain a native-preserved nontrivial region with a closed typed entry/exit.
2. Expand architecture-compatible traces beyond the three staged canaries.
3. Sweep independent service/cache uncertainty and test ranking stability.
4. Re-run the post-prediction NCU join on untouched workload families.
5. Preserve exact fallback for every newly exposed unsupported state.
