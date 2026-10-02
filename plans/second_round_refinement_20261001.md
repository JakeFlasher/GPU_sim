# Second-round refinement and reification program

**Revision basis:** second-round aggregated review at
`research_audit_20261001/second_round_aggregated_review_20261001.md`,
against commit `ca3867b0`.

**Purpose:** turn the review into a staged research and implementation program
without overstating what the current simulator proves. The program is ordered
by causal dependency: first make the existing interval and evaluation surfaces
sound, then build an executable macro transition, then test native refinement,
and only after that make any compression or architectural claim.

This document is a working plan, not an authority, approval, provenance, or
execution gate. It introduces no receipt, audit, hashing, admission, freeze,
or validation workflow. Native replay remains the exact fallback, and unknown
hardware state remains an explicit black-box variable.

## Research objective

Establish (or falsify) the following narrower thesis:

> A GPU trace region may be replaced by a machine-checkable macro transition
> only when its typed entry state, internal effects, completion state, event
> order, and observer projection are closed; every unsupported or unresolved
> case falls back to exact replay while retaining explicit unknowns.

The project must distinguish four evidence levels throughout:

1. **Descriptor equivalence:** two implementations agree on the supplied
   descriptor recurrence.
2. **Model transition equivalence:** an executable macro and detailed replay
   agree on the declared model state, internal events, and continuation.
3. **Native boundary refinement:** the macro preserves a declared projection of
   native `SmCore`/system state at entry, exit, and all future-relevant effects.
4. **Hardware comparison:** a prediction or intervention is compared with an
   independent NCU observation after prediction. Level 4 never upgrades levels
   1–3 by itself.

The current repository has useful evidence at level 1, conservative refusal at
level 3, and a lower-bound diagnostic at level 4. It has not yet demonstrated a
nonidentity executable macro or native compression.

## Stage map

| Stage | Focus | Main review findings | Exit evidence |
|---|---|---|---|
| R1 | Semantic closure and evidence hygiene | 4–9D, 10–14, evaluator/schema weaknesses | Edge-case tests pass; open/finite strata and identity metadata are explicit; no unsound finite upper or false compression ratio remains |
| R2 | Executable macro transition seam | Critical 0, Critical 1, Major 8, Major 10 | A nonidentity macro is installed by the engine, emits effects/tokens, and resumes detailed replay; macro-disabled and macro-enabled model states match |
| R3 | Independent witness and ordering closure | Critical 1, Major 3, event-order portion of 9D | Eligibility facts come from a separate witness path; false evidence and cross-warp/same-cycle counterexamples refuse or resolve explicitly |
| R4 | Service/queue/assignment semantics | Major 3, 6, 9, 9A–9C, 11, 14 | Demand has an explicit admission/service/completion interpretation; unknown/unlisted/optional pools widen or refuse; dependent/independent/contended probes affect declared parameters |
| R5 | Native state refinement | Critical 1, Major 7, Major 10 | One nontrivial native macro has closed typed entry, derived exit, event/effect span, pending-token mapping, and continuation proof; malformed/pending variants fall back |
| R6 | Independent evaluation and corpus expansion | Major 4–5, 12–13, second-round queue | Predictions are materialized before NCU join; finite intervals are scored separately from open refusals; split/hardware/endpoint identities are checked; real macro runtime and step reduction are measured |
| R7 | Robust design conclusion and paper/rebuttal | Novelty boundary and all remaining findings | Ranking is partial-order stable under independent uncertainty; claims are labeled model/native/hardware; a changed architectural conclusion is shown or explicitly ruled out |

Stages are deliberately sequential. A later stage may expose a need to return
to an earlier stage; that is a falsification or refinement result, not a reason
to relabel an unfinished mechanism as complete.

## Goal transition protocol

Only one stage is active at a time. At stage start, create one concrete goal
whose objective names the stage and its acceptance evidence. During the stage,
keep the changes limited to that seam and run a representative path before the
full suite. At stage completion, mark that goal complete only when its exit
evidence is present, record the evidence and remaining limitations in this
file and the continuation memory, then replace it with the next stage goal.
No goal completion is inferred from a green unit suite alone.

## R1 — semantic closure and evidence hygiene (complete)

### Goal

Remove known unsoundness from the existing model and evaluation surfaces before
building a macro path. Keep all changes model-relative and preserve default
native replay.

### Work packages

#### R1.1 Frontier and service arithmetic

- Make an all-unclassified or missing-pool demand refuse or retain an unknown
  service upper; it must never project to zero merely because no named pool is
  present.
- Do not add startup latency from an optional zero-demand pool to a lower bound.
- Define and encode whether service is a residual closure, a replacement for
  detailed timing, or an explicitly independent post-replay stage. Until an
  executable replacement exists, label the composition as a model-local
  intervention and never call it a detailed execution speedup.
- Treat candidate pool sets as alternatives or add an explicit assignment
  abstraction; do not present marginal pool maxima as a jointly feasible lower
  bound when one event can route to only one pool.
- Propagate unknown active service, visibility, writeback, cache-history, and
  typed-tail uppers uniformly through explicit endpoint pairs. Distinguish an
  inactive/unobserved field from an active unresolved tail in the type rather
  than relying on a few field-specific checks.

#### R1.2 Event and interval semantics

- Keep event time, insertion sequence, and logical event span as separate
  coordinates. Do not reject a valid span solely because a later-inserted event
  has an earlier time unless the producer supplies a causal sequence edge.
- Add adversarial tests for earlier-time insertion, equal-time phases, and
  internal spans with independent order witnesses.
- Compare declared native projection fields, not only one terminal modeled-event
  scalar, whenever an evaluator claims an invariant.

#### R1.3 Metrics and report vocabulary

- Report record coverage, exact/reified execution-step counts, and step
  reduction as separate fields. Never use a record ratio as a simulator-step
  compression ratio.
- Report finite interval coverage/width separately from open-interval refusal
  and unresolved-tail rates. Open intervals are conservative uncertainty, not
  calibrated accuracy.
- Use separate latency, throughput, queue, occupancy, and contention
  interventions in design-space reports; overlapping intervals produce an
  indeterminate tie rather than a lexical ranking.

#### R1.4 Evaluation input hygiene

- Validate manifest split vocabulary, family uniqueness, workload identity, and
  no family crossing calibration/development/held-out splits.
- Extend the NCU observation schema with endpoint descriptor, metric, GPU UUID,
  compute capability, SM clock, cache policy, launch identity, repetitions, and
  clock domain. Reject a join when the target metric or hardware identity does
  not match the prediction contract.
- Preserve pre-NCU prediction documents and post-NCU joins as separate files;
  never let target rows enter prediction materialization or model selection.

### R1 acceptance checks

1. `total_events=1, unclassified_events=1, pools={}` refuses or returns an
   open/unknown upper; it never returns zero service.
2. A zero-demand envelope has zero lower/upper service, and an optional pool
   with zero lower demand contributes no lower startup latency.
3. Active lower-only service/visibility/writeback/cache-history/typed-tail
   fields prevent a finite explicit endpoint upper; inactive fields are tested
   as a separate declared case.
4. An event inserted at time 5 after time 10 is represented without conflating
   insertion sequence with time order; a supplied causal order can still reject
   it conservatively.
5. The evaluator reports `finite_interval`, `open_interval`, and `refused` as
   disjoint strata and does not count open intervals as calibrated coverage.
6. A complete typed boundary with an explicit `reify`/fallback decision is
   counted as fallback by the Python summary.
7. A 16-record/one-step witness reports `record_coverage=16`,
   `exact_execution_steps=16`, `summarized_execution_steps=1`, and
   `step_reduction=16.0` with no ambiguous `compression_ratio` field.
8. All focused Rust/Python tests pass, plus the full workspace suite and JSON
   schema checks.

### R1 non-goals

- Do not claim a macro is executable yet.
- Do not use NCU observations to fit or select a scored workload.
- Do not solve occupancy, cache ownership, compiler routing, or proprietary
  hardware scheduling through guessed fields.

## R2 — executable macro transition seam (complete)

### Goal

Make a summary decision operational rather than diagnostic. A `Preserve` or
`Reify` decision must install a macro transition in a model executor, emit its
declared internal effects and pending completion tokens, and then resume exact
replay from the macro exit. The macro-disabled path remains the reference.

### Work packages

- Define a small executable `MacroTransition` interface with typed entry,
  internal event/effect span, typed exit, pending tokens, and continuation
  cursor.
- Implement one nonidentity scalar region whose exit is independently computed
  from the region entry and descriptor facts. Do not use the same detailed
  recurrence for both sides.
- Add an engine seam that consumes the reification decision; `fast_frontier` or
  report generation alone is insufficient.
- Run macro-enabled and macro-disabled execution from identical entry state and
  compare all declared model fields, not only terminal time.
- Refuse a macro when a response, barrier, unresolved tail, or pending token
  crosses the region boundary.

### Exit evidence

- The same trace executes through a real macro path and then a detailed suffix.
- Macro-disabled and macro-enabled terminal projections and continuation state
  match for the positive case.
- A pending-response/barrier adversarial case takes exact fallback.
- Runtime/step counters show actual macro dispatch, not report-only collapsing.

## R3 — independent witness and ordering closure (complete)

### Goal

Separate eligibility facts from producer assertions and make multi-warp order a
real witness or a conservative refusal.

### Work packages

- Extract dependency/effect/residency witnesses through a distinct trace/static
  analysis pass; do not derive evidence from the same `dependency_free` bit that
  admits the region.
- Add witness invalidation/perturbation tests: changing one dependency or effect
  must change eligibility or force fallback.
- Represent globally comparable issue order, or enumerate admissible orderings
  and require the summary to be invariant across them.
- Define equal-time event phases and preserve completion multiplicity as explicit
  effects/tokens.

### Exit evidence

- A caller cannot certify a false dependency-free region by supplying a matching
  range alone.
- Multi-warp/same-cycle counterexamples either have a declared order witness or
  refuse without synthetic lockstep.
- Completion-token count and pending state match at macro exit.

## R4 — service, queue, and assignment semantics (complete)

### Goal

Replace the current aggregate/max envelope with a declared abstract service
model whose demand, admission, queueing, completion, and observer composition
are distinguishable.

### Work packages

- Choose and document first-issue/last-issue conventions and whether service is
  residual or replacement timing.
- Add a minimal queue state: admission time, issue interval, latency, capacity,
  queue depth, and completion frontier.
- Model alternative resource assignments jointly; use interval assignment or
  explicit upper/lower relaxations rather than summing incompatible marginals.
- Fit intervals, not only median slopes; use occupancy/contention probes either
  in a parameterized model or label them explicitly unmodeled.
- Test one event, dependent pair, independent pair, ALU+LSU serial, and
  ALU+LSU overlap cases.

### Exit evidence

- Varying each probe mode changes only the parameter it identifies, or the
  report states that the dimension is unmodeled.
- Missing/slow candidate classes widen or refuse; they do not collapse to zero.
- The composed completion frontier is the only service quantity passed to the
  observer projection, with an explicit model policy.

## R5 — native state refinement (current stage)

### Goal

Demonstrate one native-preserved nontrivial region, or obtain a precise negative
result showing why the current typed boundary cannot close it.

### Work packages

- Extend the comparator from endpoint equality to transition/effect validation:
  internal event payloads, sequence/time spans, state deltas, pending-token
  mapping, and observable labels must be consumed by the proof.
- Derive the macro exit and effects independently from native replay; do not
  compare a typed boundary against itself as evidence.
- Require the Python reference summary to honor the explicit native decision;
  complete typed state with `reify` still counts as fallback.
- Exercise RAW, memory response, barrier, async-copy, and unmodeled-tail
  variants and preserve exact fallback.

### Exit evidence

- A scalar or otherwise nontrivial native macro has a closed typed entry, an
  independently derived exit, a checked internal trace, and a detailed
  continuation.
- A fabricated equal endpoint with empty effects is rejected.
- The native boundary report distinguishes preserved, exact-fallback, and
  incomplete cases without treating fallback as compression.

## R6 — independent evaluation and corpus expansion

### Goal

Measure the actual observer projection and real execution path over a broader,
identity-checked corpus.

### Work packages

- Materialize predictions before any NCU observation is read. Join observations
  only after prediction and retain all endpoint/hardware metadata.
- Add finite-width calibration/coverage/width metrics and separate open/refusal
  rates; report OOD behavior, runtime, memory, and actual macro step counts.
- Expand beyond the three canaries to scalar, tensor, memory, synchronization,
  and mixed families, with family-held-out splits checked by code.
- Compare macro-enabled versus detailed execution runtime on the same input and
  environment; report report-generation overhead separately.
- Sweep independent service/cache uncertainty and record partial-order ties.

### Exit evidence

- No scored NCU field appears in the pre-NCU prediction document.
- Every hardware join matches the declared metric, endpoint, GPU identity,
  clock/cache policy, and launch identity.
- Compression is a real execution-step/runtime measurement when claimed; refused
  regions remain in the denominator.

## R7 — robust conclusion and rebuttal framing

### Goal

Convert the evidence into a bounded scientific claim, including negative
results, and avoid novelty claims already covered by interval simulation,
GPU overlap models, skipping refinement, or native trace-driven simulators.

### Work packages

- Rank only interventions whose intervals are disjoint; report ties otherwise.
- Test whether any architectural conclusion changes when the macro is disabled,
  service/cache parameters vary independently, and unsupported regions fall
  back.
- State the contribution as a state-closure/refusal contract only if R2/R5
  succeed; otherwise frame the work as model-local scaffolding and a roadmap.
- Update README, status, project memory, review history, and rebuttal notes with
  observations, hypotheses, inferences, and unresolved limitations separated.

### Exit evidence

- A table maps every headline claim to evidence level 1–4 and its falsifier.
- The final paper/rebuttal does not call a lower-bound proxy an endpoint
  prediction, open coverage calibrated accuracy, or descriptor reduction native
  speedup.
- Remaining blockers are explicit and do not prevent useful model-local use.

## Current stage ledger

| Stage | Status | Evidence / next action |
|---|---|---|
| R1 | **complete (2026-10-01)** | Unknown/zero-demand/optional-pool arithmetic, active completion requirements, event-order semantics, v2 target contracts, finite/open evaluation strata, native projection completeness, and record-vs-step metrics are implemented and tested. |
| R2 | **complete (2026-10-01)** | `ModelExecutor::execute_decision` consumes Summary/Reify decisions, installs a nonidentity scalar macro or exact fallback, emits effects/pending completions, resumes detailed descriptors, and exposes macro dispatch counters. |
| R3 | **complete (2026-10-01)** | Eligibility requires a separate fact-complete witness with an allowed origin; stale/producer-premise facts reify, and multi-warp/same-cycle order remains explicit refusal or engine-order checked. |
| R4 | **complete (2026-10-01)** | Service roles, serial queue/frontier semantics, grouped assignment lower/upper envelopes, unknown/zero-demand handling, occupancy/contention labeling, and throughput interval endpoints are implemented and tested. Native metrics remain marginal where no per-record route witness exists. |
| R5 | **active** | Validate one nontrivial native macro boundary with internal effects/pending tokens and exact fallback. |
| R6 | queued | Depends on an actual macro path and observer contract. |
| R7 | queued | Final claim discipline after evidence is complete. |

## Continuity rules

- Preserve unrelated work and the pre-existing untracked research corpus.
- Run a representative executable or focused counterexample before broad tests.
- Keep default native replay unchanged unless a test explicitly selects the
  macro path.
- Do not convert a checksum, receipt, digest, or review artifact into an
  execution prerequisite.
- If a stage cannot produce its exit evidence, record the falsifying result and
  replace the goal with a narrower stage rather than declaring success.

## R1 implementation evidence

- `ResourceServiceProjection.upper_cycles` is now optional. Unclassified demand
  keeps the upper open; an all-unclassified/empty-pool envelope cannot project
  to zero unknown service. Zero-demand pools contribute zero startup latency.
- `FrontierVector.required_completion` distinguishes an active wholly-unknown
  completion field from an inactive/unprojected field. Explicit endpoint pairs
  propagate active unknown issue/service/tail/cache/visibility/writeback state.
- `TransformInterval.sequence_ordered` makes sequence monotonicity an explicit
  witness. Same-cycle effects still require engine `(time, phase, sequence)`
  order, while earlier-time insertion does not create a false global sequence
  edge.
- Native design evaluation now requires the declared timing projection to be
  present before reporting an invariant. Open intervals are refusal/uncertainty
  strata, not calibrated coverage; interval-overlap variants are indeterminate
  rather than lexically ranked. Record coverage and execution-step reduction
  are separate fields.
- Prediction manifests and post-NCU observations use v2 target contracts with
  metric, endpoint, clock domain, cache policy, launch identity, repetitions,
  and GPU identity. Family/split rosters are validated before any prediction
  process is launched. The reference summary honors explicit `reify` decisions
  even when a typed boundary shape is complete.
- Evidence: 119 `sim-semantics` tests, 64 `sim-run` tests, the full workspace
  suite (all non-ignored tests passed), 37 focused Python tests including the
  reference comparator, JSON/schema parsing, and a timed SM86 latent
  canary with `simulator_lower_cycles=432` and an open predicted upper. The
  unrelated PTX subset compiler test still has its pre-existing binary/schema
  mismatch and is outside R1.

## R2 implementation evidence (complete)

- `sim-sm::ModelExecutor` is now an explicit execution seam. It accepts
  detailed descriptors, consumes a `FidelityDecision::Summary` certificate,
  installs a typed `MacroTransition`, emits `MacroEffect` issue/completion
  effects and pending completion tokens, and resumes detailed execution.
- The transition projection is computed independently through the interval
  certificate's closed form before state installation. `ModelExecutionSnapshot`
  compares all declared model frontiers/effects between macro-enabled and
  detailed paths.
- The model-local probe now reports one macro dispatch for its 16-instruction
  witnessed region (`exact_execution_steps=16`, `causal_execution_steps=1`),
  while a pending predecessor rejects the macro with
  `IncomingPipelineWork`. This is a real model-executor path, not native
  `SmCore` execution. The fast-frontier surface aggregates `macro_dispatches`,
  and its v1 JSON schema requires that field. Native replay remains unchanged;
  R5 is still required for a native boundary-preservation claim.

## R3 implementation evidence (complete)

- `CausalFrontierWitness` facts are separate from descriptor booleans and carry
  an explicit origin. `CausalFrontierEvidence::from_independent_facts` requires
  contiguous fact coverage and rejects `ProducerPremise`; `covers` compares all
  dependency/effect/multiplicity fields before eligibility.
- The compact native fast path no longer manufactures `CompleteWithEvidence`
  from the candidate segment. It supplies residency only and therefore takes
  exact/fallback execution until a separate static/trace witness is provided.
- A stale witness whose dependency fact is perturbed reifies with
  `MissingIndependentEvidence`; multi-warp candidates still reify with
  `CrossWarpOrderUnknown`; same-cycle internal effects retain engine
  `(time, phase, sequence)` order.
- Evidence: 491 `sim-sm` tests (including adversarial witness/order cases),
  the model-local probe with one witnessed macro dispatch and three multi-warp
  refusals, plus the prior workspace and focused evaluation checks. This closes
  the model contract only; it does not infer a hardware scheduler order.

## R4 implementation evidence (complete)

- `ServiceStageRole` now distinguishes residual closure, replacement timing,
  and independent post-replay service. The current full-trace demand path is
  explicitly `IndependentPostReplay`, so it no longer silently adds a second
  cost to the observer endpoint. Replacement timing is a separate tested
  composition, not an implicit interpretation.
- `ServiceQueueProjection` reports a serial admission/frontier convention,
  queue depth, open completion uppers, and the unresolved marginal-pool
  assignment semantics. It is a model-local queue envelope, not a GPU
  scheduler trace.
- The v2 fitter now requires only latency/throughput/queue probes for the
  fitted rate equation. Occupancy/contention rows are retained as explicitly
  unmodeled dimensions and changing only those values leaves fitted rates
  unchanged; holdout reports them as unmodeled.
- Throughput profiles now retain a lower/upper initiation-interval envelope;
  the projection widens only the service upper under that uncertainty.
- Grouped alternative assignment domains now preserve lower/upper assignment
  envelopes when a producer supplies them; the native metrics path does not yet
  extract such groups, so its reports remain marginal. No service-rate or queue
  result is a silicon timing claim.

## R5 progress (active)

- `TypedRegionContract::validate` now validates every internal effect payload
  and rejects malformed sequence spans. The native comparator has a trace-aware
  entry point that compares internal effects and observable labels in addition
  to typed entry/exit projections.
- A `NativeBoundaryExecutionPath` converts the comparator decision into an
  explicit typed-macro versus exact-replay plan. The opt-in native writer
  records that plan, but still leaves replay unchanged because no native
  nonidentity macro transition is installed yet.
- Synthetic complete boundaries now preserve only when effect spans and labels
  match; endpoint-equal contracts with missing/duplicated/malformed effects
  force exact fallback. The retained SM86 canary remains fallback-only.
- `derive_typed_scalar_macro_region` / `execute_typed_scalar_macro` provide a
  strict typed-boundary executor for a closed single-warp fixture and are
  covered by preserve/fallback tests. A real SM86 pure-scalar probe now emits a
  complete typed pre/post boundary pair, but its cold front-end transition is
  not derivable by that rule and correctly remains exact replay.
- The typed executor explicitly refuses pending queue responses, dependency
  barriers, front-end/cache/pipeline state, and incomplete/unknown tails before
  deriving an exit.
- The reference writer now captures explicit pre/post typed boundaries and the
  issued-instruction effect span for each SM wake. The pure-scalar SM86 probe
  yields one complete pre/post pair with two detailed issue effects; an opt-in
  typed scalar spec still reifies because the cold front-end state is outside
  the strict macro closure (the bounded probe reports
  `future_relevant_warp_or_response_state_is_not_closed`). This is the required
  adversarial native check, not a preserve claim.
- The opt-in reference execution path now consumes
  `NativeBoundaryExecutionResult`: a matching transition selects the typed
  macro branch, while a rejected transition carries the detailed exit and
  exact-replay reason. `SmCore::execute_reference_macro_path` consumes that
  selected path and the native issue-model test compares macro entry/exit and
  detailed continuation state; the real probe selects the latter branch.
- The typed scalar rule now derives first issue from the entry warp readiness
  and can consume a closed front-end instruction-buffer prefix; it still
  refuses absent/noncontiguous buffers and all open pipeline/cache state.
- The decisive remaining R5 check is a real nonidentity native macro whose
  exit/effects/pending-token mapping are independently derived and consumed by
  continuation execution; until then this stage is not complete.
