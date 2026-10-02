# Latest-review refinement program

**Revision date:** 2026-10-02
**Branch:** `GPUSim_main`
**Input:** `research_audit_20261001/second_round_reviews/aggregated_review.md`
**Status:** staged execution; this document is the active plan for the next
revision after R7.

This plan treats the latest panel report as an implementation and claim
boundary, not as an approval gate. It does not add audit, provenance,
receipt, authority, sealing, freeze, or admission machinery. The project keeps
the permanent alpha rule: unsupported native or hardware behavior is refused
or represented as unknown rather than guessed.

## Working claim and evidence ladder

The only claim we should try to strengthen is:

> A GPU trace/model region may be replaced only when its typed timing,
> dependency/control, effect, pending, event-order, front-end, and continuation
> state is closed under an independently derived witness; otherwise detailed
> replay or an explicit open/refused result is retained.

Evidence is kept at four levels:

1. descriptor recurrence equivalence;
2. model-executor transition equivalence;
3. native `SmCore` boundary refinement and production dispatch;
4. independent observer comparison.

No later level is inferred from an earlier one. In particular, a model-local
16-to-1 dispatch count is not native speedup, an open lower-bound diagnostic is
not interval coverage, and a manifest-labeled NCU row is not an independently
verified device identity.

**Canonical status at synchronization:** R8.0/R8.1/R8.3/R8.4/R8.5/R8.6 are
complete. R8.2 is deliberately reopened: the test/helper seam is not
production event-loop evidence, and the bounded native canary fails at a
response-owner/queue join before any macro is consumed. The sections below
retain the staged work packages and falsifiers; this status line is authoritative
over earlier historical prose.

R9.0 is complete: the canary boundary now preserves unresolved owner/queue
joins as unknown and the bounded executable terminates with exact fallback.
R8.2 production macro dispatch remains open.

R9.1 and R9.2 are complete negative loops: the optional witness/dispatcher path
is exercised only when a complete independent typed boundary is supplied; the
ordinary native canary remains exact fallback with zero dispatches, and no
complete two-pass witness exists yet.

R10.0 live-campaign execution is complete for the bounded large development
probe: current-host CUDA/NCU/NVBit execution succeeded, profiler values remain
post-prediction observations, and the campaign role is development rather than
held-out accuracy.

R10.1 is the live writeback loop: dirty-stream footprints target the unresolved
terminal service tail, with fresh post-prediction NCU repetitions and no
writeback fitting.

R10.2 is complete for a two-case live cache-policy contrast: fresh
`cache-control=none` repetitions were collected after the all-cache
post-prediction condition; the contrast is observational only.

R11.0 is complete for the retained simulator cost loop: all 15 traces now have
detailed replay and fast-frontier measurements on identical inputs, with zero
accepted candidates and explicit refusal coverage.

## Stage ledger

| Stage | Focus | Exit evidence | Status |
|---|---|---|---|
| R8.0 | Review intake and executable plan | This plan maps every critical/major/open finding to a code seam, test, falsifier, and claim boundary | **complete** |
| R8.1 | Immediate soundness guards | Mismatched certificates, inconsistent grouped queues, missing observer projections, incomplete cache domains, and unchecked arithmetic refuse before mutation or finite projection | **complete** |
| R8.2 | Production native decision path | A comparator-selected closed fixture reaches the real event loop, increments a dispatch counter, installs effects/pending state, resumes a detailed suffix, and falls back before mutation when one field is perturbed | **reopened: production path unestablished** |
| R8.3 | Native state closure and independent refinement | Nonzero scalar latency/completion/front-end state is either consumed and compared against an independently captured boundary or conservatively refused; self-match tests are not the sole evidence | **complete** |
| R8.4 | Service/queue semantics and identifiability | Queue unknowns cannot close spuriously; residual/replacement roles have explicit equations and a held-out intervention, or the profile is clearly sidecar-only; latency, throughput, and queue parameters are separately identified | **complete** |
| R8.5 | Measurement identity and cost utility | NCU observations are explicitly verified or labeled as manifest-labeled historical data; source/trace/control/card/launch identity is bound; parser/planner/effect/compare/RSS/wall-time cost and accepted/refused coverage are reported without speedup overclaim | **complete** |
| R8.6 | Independent evaluation and paper synchronization | New tests/probes, negative results, novelty/prior-art boundary, histories, memory, status, schemas, and rebuttal all agree; full validation passes | **complete** |
| R9.0 | Native canary queue/response closure | The bounded reference canary terminates with explicit unknown owner/queue state; complete boundaries and macro dispatch remain zero unless independently closed | **complete** |
| R9.1 | Production witness dispatch attempt | A complete independent witness reaches the real event loop and proves continuation, or the canary records precise no-witness/refusal evidence without a native claim | **complete: no positive native fixture** |
| R9.2 | Two-pass witness/continuation audit | A complete independent pre/effect/post witness exists and drives one production dispatch, or the corpus records no eligible pair and exact fallback | **complete: no eligible pair** |
| R10.0 | Live SM86 campaign | A bounded live host run records host identity, native/NCU/trace/replay termination, prediction-before-post-observation artifacts, open/refused strata, and no unsupported speedup claim | **complete: 10-case large development probe** |
| R10.1 | Live writeback campaign | Large dirty streams complete native/trace/replay/NCU runs, preserve writeback tails as unknown/unavailable, and make no speedup or fitted-hardware claim | **complete: 2-case large writeback probe** |
| R10.2 | Live cache-policy contrast | Fresh none-cache repetitions compare against all-cache observations without fitting or cache-causality claims | **complete: 2-case contrast** |
| R11.0 | Full retained-trace simulator utility | Detailed versus analyzer wall/RSS/candidate/refusal measurements complete without native-speedup terminology | **complete: 15-workload cost campaign** |

## R8.1 — immediate soundness guards

### Work packages

- Bind a `MacroTransition` certificate to the consumed descriptor region:
  pipeline key, latency, initiation interval, capacity, instruction count,
  first/last sequence, and any incoming-state requirement. Validate all
  fields before changing executor state.
- In grouped service assignment, require classified counts to account for all
  events and propagate `unclassified_events > 0` to an open upper. Empty named
  pools and optional zero-demand pools must remain zero only when the demand is
  known zero.
- Make `observer_projection` mandatory in the Python contract at the same
  layer as the JSON schema; reject predictions that omit or contradict it.
- Require temporal order, fence scope, line generation, owner/read-from, and
  completion domains before an exact cache-history frontier is returned.
- Replace unchecked `+1`, group-count, queue-capacity, and assignment arithmetic
  with checked operations and reject invalid replacement admission.
- Make same-cycle external inputs require a supplied partial-order witness;
  otherwise preserve the open/refused result. State duplicate-effect
  multiplicity explicitly in the contract.

### Acceptance and falsifiers

- A latency-10 certificate consumed by a latency-1 descriptor returns a
  binding error and leaves the executor byte-for-byte unchanged.
- An envelope with `total_events=2`, `unclassified_events=1`, and a nominal
  two-event group cannot produce a finite queue or observer upper.
- A prediction row without `observer_projection` is rejected by both the
  runner-facing validator and the schema test.
- Cache owner/completion values without temporal/fence/generation values do
  not yield an exact history frontier.
- Overflow or replacement-without-admission is a typed error, not a wrapped or
  negative interval.

## R8.2 — production native decision path

### Work packages

- Identify the real `SmCore` event-loop boundary where the pre-event snapshot,
  independently derived region-end snapshot, and comparator decision coexist.
- Capture a distinct region-end boundary instead of comparing a nonzero derived
  exit to the same event identity. Preserve the event phase/sequence relation.
- Consume the selected `TypedMacro` in the production reference-feature loop;
  install every declared effect and pending token, advance the front-end and
  scheduler cursor exactly once, then resume detailed replay at the region end.
- Route every mismatch, unresolved tail, pending response, or unsupported
  event to exact replay before mutation. Expose a bounded dispatch/fallback
  counter in the reference result.
- Add a complete two-instruction production fixture and a field-perturbation
  differential. The fixture must execute through the same loop used by the
  opt-in writer, not a direct helper call.

### Acceptance and falsifiers

- A future production fixture must report one native typed-macro dispatch and a
  detailed suffix with the same declared continuation state as exact replay;
  the current helper fixture is not sufficient.
- Perturbing entry, effect, pending, exit, or continuation state selects exact
  fallback and does not mutate the native core.
- A real front-end-enabled SM86 canary remains exact fallback unless all
  future-relevant state is closed; no row is relabeled as native compression.

## R8.3 — native state closure and independent refinement

### Work packages

- Extend the typed scalar transition to model latency/release queues, local
  completion, capacity pressure, writeback, and pending tokens; compare those
  fields at the region end and at a dependent detailed suffix.
- If a field cannot be independently captured, make eligibility refuse before
  mutation rather than retaining a structurally complete but semantically
  partial scalar transition.
- Capture front-end cursor/fetch state and distinguish already-consumed
  buffers from a macro-owned prefix.
- Build an independent effect ledger (trace/static pass or separately built
  fixture) containing effect identity, payload class, multiplicity, span, and
  continuation obligations. Do not use a summary to certify its own effects.
- Bind all certificates to the exact region shape and continuation contract;
  preserve only when the independent ledger and native boundary agree.

### Acceptance and falsifiers

- A real carded scalar pipeline with latency >1 and a dependent suffix either
  matches release/completion/effect state or refuses.
- A fabricated equal endpoint with empty effects is rejected.
- A front-end cursor mismatch refuses before mutation.

## R8.2 attempted implementation and falsifying result (reopened 2026-10-02)

- The reference trace accepts an optional, separately produced typed witness
  (`SIM_SM_REFERENCE_TYPED_SCALAR_WITNESS`). At a pre-event boundary it runs
  `execute_typed_scalar_macro` against that witness's detailed effects/labels/
  exit, stores the resulting `TypedMacro` or `ExactReplay`, and exposes the
  decision in the trace. No witness means no macro plan.
- The first bounded real-SM canary exposed a response-owner/queue join failure
  before a production macro could be consumed. That real-corpus result remains
  exact fallback. The later P2 fixture wires a separately built complete
  witness through the same reference-feature event loop; it is synthetic
  differential evidence, not front-end/cache SM86 compression.
- The real native canary therefore has no dispatch counter or runtime benefit;
  its pre/post stream remains diagnostic exact fallback.

The retained SM86 front-end canary still selects exact fallback: its complete
fetch/cache/response state and independent witness are absent. The witness
contract is a research input for a closed-region experiment, not a mandatory
run artifact, target label, provenance gate, or silicon oracle.

## R8.4 — service/queue semantics and identifiability

### Work packages

- Fix grouped assignment and queue composition from R8.1, then expose the
  admission equation, serial/overlap convention, queue depth, and completion
  frontier as separate model-local fields.
- Add an explicit residual/replacement experiment on a held-out model trace;
  keep the detailed fallback and observer target separate. If replacement is
  not scientifically justified, keep the service profile sidecar-only and
  state that choice everywhere.
- Require dependent-chain latency observations, independent-stream
  throughput intervals, and queue-depth/capacity interventions. Do not derive
  latency from aggregate throughput when a dependent probe is absent.
- Fit lower/upper initiation intervals and score interval endpoints, while
  leaving occupancy/contention as an explicit unmodeled dimension.

### Acceptance and falsifiers

- An all-unclassified demand never closes a finite queue upper.
- A queue-depth intervention changes a declared queue equation or is reported
  as non-identifying; it must not silently produce the same “calibrated” rate.
- Changing an independent-post-replay profile changes only its sidecar unless
  a declared residual/replacement role is selected.

## R8.3 implementation evidence (complete 2026-10-02; conservative eligibility scope)

- The typed scalar eligibility rule now refuses latency >1, any pending queue
  token, front-end/scalar-pipeline state, and nonempty instruction buffers.
  These are state classes not represented by the issue-only transition. A
  latency-10 or otherwise longer completion tail is therefore a typed
  fallback, not a native transition.
- The optional witness carries a separately supplied detailed effect ledger,
  observable labels, and detailed exit. The comparator consumes those fields
  at the reference-feature event-loop seam; the positive synthetic fixture is
  built from an independently constructed entry/exit/effect region, while perturbed,
  empty-effect, pending/tail, and exact-replay cases refuse. This is not
  production dispatch evidence.
- Front-end buffer/cursor state remains outside the accepted issue-only rule.
  The retained front-end-enabled SM86 traces still lack a closed witness and
  remain exact fallback; no self-match or synthetic endpoint is called native
  completion.

This closes the omission boundary by refusal rather than pretending to model
latency, release queues, register values, memory responses, or barriers that
the actor does not consume. General native latency/completion modeling remains
an explicit future research stage, not a hidden claim.

## R8.5 — measurement identity and cost/utility

### Work packages

- Parse available NCU/device metadata where present. Where raw retained files
  cannot independently establish host identity, label rows
  `manifest_labeled_historical` and remove “independently checked” wording.
- Bind source trace, control/card, launch, cache policy, metric, endpoint,
  clock, repetition, and GPU identity semantically in prediction and
  observation contracts without introducing prohibited workflow machinery.
- Instrument separate detailed parse/replay, planning, effect-materialization,
  comparator, output, wall-time, and RSS costs. Do not call dispatch ratios
  speedup.
- Report accepted/refused/open coverage and false-summary counterexamples;
  compare methodology against interval/sampling/partial-execution prior art
  only where compatible measurements exist.

### Acceptance and falsifiers

- Forged manifest metadata is either rejected by metadata parsing or visibly
  labeled as historical/manifest-labeled and excluded from independent
  identity claims.
- A cost table includes the full analyzer path and never presents 16-to-1
  dispatch as CPU/runtime speedup.

## R8.4 implementation evidence (complete 2026-10-02; queue equation explicitly absent)

- The v2 fitter now requires explicit `dependent_latency` and
  `independent_throughput` modes. Omitting `latency_cycles` from a dependent
  probe is an error; aggregate `(cycles/events)` is never relabeled as chain
  latency. Queue-capacity/depth rows remain accepted interventions, but the
  profile declares `queue_depth` unmodeled because no queue-delay equation is
  fitted. Occupancy and contention remain unmodeled as before.
- Capacity-adjusted throughput slopes remain marked
  `capacity_already_applied`, with lower/upper initiation-interval endpoints.
  Queue-depth rows are retained in holdout diagnostics with an explicit
  `queue_depth_not_in_profile_equation` reason rather than being silently
  treated as calibrated queue state.
- `sim-run` now exposes `--service-stage-role` with explicit
  `independent_post_replay` (default), `residual_closure`, and
  `replacement_timing` compositions. Non-independent roles activate the
  resource-service completion requirement and therefore feed the declared
  model-local observer frontier only by explicit selection; the default
  sidecar does not alter native replay or the observer endpoint.
- Direct-run tests compare the sidecar and replacement roles on a held-out
  model fixture. The ordinary profile changes the declared resource frontier
  while the unresolved typed tail still dominates the observer candidate;
  a stressed replacement profile demonstrably moves the model-local lower
  endpoint. No NCU target is used, and no hardware prediction is implied.

This closes the review's queue-identifiability and disconnected-composition
wording. It does not recover a hardware scheduler queue or turn the abstract
service profile into a calibrated silicon model.

## R8.6 — evaluation, synchronization, and rebuttal

### Work packages

- Add adversarial and executable tests for every R8.1–R8.5 acceptance clause.
- Re-run representative model, native, service, and R6 artifact probes before
  the full workspace suite.
- Update the active plan, `docs/STATUS.md`, `docs/PROJECT_MEMORY.md`,
  `docs/REVIEW_HISTORY.md`, `docs/PREDICTIVE_CLOSURE.md`, README navigation,
  schemas, configs, and rebuttal with the actual result, including negative
  results and remaining falsifiers.
- Commit only coherent changes; leave the pre-existing untracked review corpus
  untouched.

## R8.5 implementation evidence (complete 2026-10-02; historical identity and full-cost scope)

- The default NCU CSV builder now emits `identity_status=manifest_labeled_historical`
  on the document and every row. It does not offer a caller-selected verified
  status or claim live-device proof; the standalone CSV adapter uses the more
  explicit `caller_labeled_unverified` status. The observation schema and join
  validator require status consistency.
- Pre-observation predictions now carry semantic source bindings for each
  workload's trace directory, control sidecar, architecture card, pchase card,
  and declared launch identity. These are ordinary input paths, not hashes,
  receipts, provenance gates, or execution prerequisites.
- `evaluate_reification_design.py` reports accepted/summarized versus
  reified/uncovered record fractions, child wall-clock/peak-RSS deltas, and an
  explicit cost-component scope covering parse, replay, planning, effect
  materialization, boundary comparison, and output. It sets
  `native_speedup_claimed=false`; dispatch/step ratios remain model-local
  counters rather than runtime speedups.
- The corrected R6 observation artifact is revalidated under the updated schema.
  A forged manifest identity is now visibly historical/unverified rather than
  being described as independent host verification.

This closes the identity wording and utility-accounting finding. It does not
create a live device verifier, and it does not compare against prior-art
simulator cost numbers that were not measured under the same input contract.

## Stop/claim discipline

If a stage cannot demonstrate its acceptance evidence, it is not complete.
Replace the active goal with a narrower goal that records the falsifying
result, keeps exact fallback, and lowers the paper claim. Never rename a
diagnostic sidecar, direct unit seam, or record ratio into native completion,
hardware causality, or speedup.

## R8.1 implementation evidence (complete 2026-10-02)

- `IntervalCertificate` now exposes its timing/shape key only through checked
  accessors, and `ModelExecutor::execute_macro_region` binds every descriptor
  in the consumed region to the certificate before calculating or mutating
  model state. The legacy single-descriptor API rejects multi-descriptor
  certificates; mismatched latency, pipeline, sequence, dependency, control,
  effect, or multiplicity leaves the executor unchanged with
  `CertificateBinding`.
- Grouped service assignments must not classify more events than
  `total_events - unclassified_events`; a classified group alongside an
  unknown tail falls back to the marginal projection and keeps the queue
  completion upper open. Replacement composition validates admission before
  constructing completion, and grouped service latency additions are checked.
- The Python prediction validator now requires a typed observer projection for
  every predicted row and checks that its lower/upper endpoints exactly match
  the prediction interval. This makes the validator agree with the v2 schema
  rather than relying on the normal runner to supply the field.
- Cache history requires temporal order, fence scope, line generation,
  owner/read-from, and hardware completion domains before an exact history
  frontier is returned. A known subset remains a lower-only frontier.
- The scalar transform's sequence arithmetic now checks the intermediate
  `instruction_count + 1` operation. New adversarial tests cover certificate
  mismatch/no-mutation, grouped unknown tails, observer projection binding,
  cache-history closure, and finite service arithmetic.

Focused evidence: `sim-semantics` 130 tests passed; reference `sim-sm` 503
tests passed; prediction/evaluation/schema Python tests 11 passed; formatting
and diff checks passed. This closes direct API/projection soundness findings,
not the production native event-loop, latency/completion closure, service
identifiability, or independent host-identity findings scheduled for R8.2/R8.5.

## R8.6 implementation evidence (complete 2026-10-02)

- All current docs agree that R8.2 real-corpus dispatch remains open after the
  bounded native queue/response-owner join failure; P2's synthetic fixture is
  separately labeled reference-feature differential evidence. R8.3 is typed eligibility/refusal, R8.4 is explicit
  service composition with an unmodeled queue equation, and R8.5 labels
  historical observation identity and full-cost scope.
- The committed R6 observation artifact is schema-valid with
  `manifest_labeled_historical`; the prediction artifact has semantic source
  bindings and no target values. Revised schemas and Python validators reject
  status/row mismatches and preserve open intervals.
- Validation passed: `cargo fmt --all -- --check`; full Cargo workspace (all
  tests passed, with only existing ignored machine-local tests); focused Python
  calibration/evaluation/schema/reference suites (41 passed); JSON Schema
  self-validation and committed prediction/observation artifact validation.
  The pre-existing untracked `research_audit_20261001/` directory was not
  staged.

The repository is synchronized for the next review. The highest-value open
research task remains a real production comparator-selected native fixture;
until its queue/response, completion, and front-end state close, exact replay
and explicit unknowns remain authoritative.

## R9.0 implementation evidence (complete 2026-10-02)

- The typed reference boundary now compares due response-owner keys with queued
  memory-response keys before declaring shared-L2 ownership complete. Any
  mismatch is recorded as `response_owner_queue_join_unknown` and widens the
  boundary instead of raising a structural error.
- The bounded SM86 `l1-l2` reference canary now exits successfully with a
  footer: 10,032 records, 5,016 pre/post pairs, zero complete typed boundaries,
  and exact fallback for all records. The prior failure at SM2/warp0/
  instruction17/sector0/due1686 is reproducibly gone.
- This is a termination and unknown-propagation fix. It does not select a
  native macro, reduce native steps, or close R8.2 production dispatch.

## R9.1 implementation evidence (complete 2026-10-02; negative native result)

- The reference feature accepts an optional typed witness and routes a
  comparator result through the `SmWake` dispatcher only when that witness
  matches the event. Exact replay remains the default; no witness is inferred
  from the same detailed boundary.
- The repaired 15-SM canary was rerun with no witness and completed with
  10,033 records before its footer, zero native macro dispatches, and exact
  fallback throughout. No complete typed boundary was observed.
- The positive requirement is therefore not met: there is no independently
  closed production fixture yet. The falsifier for the next loop is explicit:
  a complete witness matching a real event must produce one dispatch and an
  identical detailed continuation, while a perturbed witness must fallback
  before mutation.

## R9.2 implementation evidence (complete 2026-10-02; witness blocker)

- The bounded native canary contains no complete typed entry/effect/exit pair.
  Its front-end/cache/response state remains incomplete, so deriving a witness
  from the same stream would violate the independent-witness contract.
- The corrected canary terminates with 10,033 records including its footer,
  zero complete boundaries, zero dispatches, and exact fallback throughout.
- The exact positive falsifier is fixed: supply a separately produced complete
  pre-boundary, internal effect ledger, and region-end boundary whose
  continuation state matches the real `SmCore` event. One matching event must
  dispatch once; perturbing any entry/effect/pending/exit field must fallback
  before mutation. No current artifact satisfies that falsifier.

## Post-review completion program (started 2026-10-02)

The aggregate review is now treated as a set of falsifiable engineering and
research obligations, not as a request to make milestone prose look greener.
The remaining work is deliberately staged so that a later result cannot hide an
earlier unsound boundary.  A concern is considered resolved only when either
the implementation carries the missing state through the real path with a
focused differential, or the path refuses before mutation and the refusal is
observable in the contract and report.

### Concern-to-stage map

| Review concern | Current boundary | Planned resolution/evidence |
|---|---|---|
| Comparator plan is not consumed by production replay | `ReferenceTrace`/`SmCore` event loop; the real canary has no complete witness while the P2 synthetic fixture is now positive | P2's event-loop fixture counts one dispatch, resumes a detailed suffix, and proves field-perturbed fallback; real-corpus dispatch remains an explicit negative result |
| Same-event pre/post pairing cannot preserve a nonzero macro | `ReferenceTrace::record_event` records the same event identity before and after `run_sm` | P2 separates region-start, internal span, and region-end identities; a nonzero preserve decision must be reachable only with an independently captured end boundary |
| Scalar latency/completion state is omitted | typed scalar transform and native actor are issue-only | P3 either models release/completion/pending state end-to-end or rejects latency >1, open pipelines, front-end buffers, and dependent tails before mutation; the latter must be tested as the conservative contract |
| Certificate/descriptor binding | `ModelExecutor::execute_macro_region` and certificate accessors | P1 re-runs adversarial binding/no-mutation checks for pipeline, latency, II, capacity, shape, sequence, dependency, control, effect, and multiplicity fields |
| Unknown grouped queue upper can close | `ResourceDemandEnvelope::project_service_queue` | P1 proves inconsistent grouped counts and unknown tails remain open through queue and observer projections, including empty/zero-demand pools |
| Service profile is only an R6 sidecar | `IndependentPostReplay` is the default observer role | P4 keeps sidecar, residual, and replacement equations separate; a held-out replacement intervention must change only its declared model-local endpoint, never silently change the detailed path |
| Dispatch/step ratio is not runtime speedup | model executor counters and fast-frontier reports | P5 reports parser/planner/effect/compare/output, wall, RSS, accepted/refused/open coverage separately; no speedup term is emitted without a real end-to-end comparison |
| NCU identity is copied from manifest | observation builders and joins | P5 keeps `manifest_labeled_historical`/`caller_labeled_unverified` explicit, parses metadata where available, and binds source/trace/control/card/launch identities semantically |
| Queue/latency fitter is not identifying a queue | v2 probe fitter and profile schema | P4 requires explicit dependent-latency and independent-throughput modes, treats queue depth/capacity as interventions unless an equation is fit, and scores interval endpoints |
| Caller self-match/coarse native effects | typed transform/comparator and optional witness | P3 requires a separately produced effect ledger and continuation relation; equal endpoints alone or self-derived effects never certify native refinement |
| Empty native accepted set is vacuous | R6 corpus and native canary | P5 reports the negative boundary as the primary result, keeps refused regions in the denominator, and compares accepted/refused coverage and false-summary fixtures with compatible prior-art baselines |
| Evaluator/schema/source identity gaps | prediction contract, JSON schemas, joins | P1/P5 make observer projection, metric/endpoint/clock/cache/repetition/hardware, and source identities mandatory and test forged/mismatched rows |
| Cache order/generation/fence and same-cycle multiplicity | cache candidate/history frontiers and external-input transform | P1 requires all temporal/fence/generation/owner/read-from/completion domains or returns lower-only/open; P3 requires explicit same-cycle partial-order witnesses and states duplicate-effect multiplicity |
| Front-end closure and arithmetic | typed scalar transform and grouped service helpers | P1 adds checked arithmetic and front-end cursor/buffer mismatch refusal; P3 keeps front-end-enabled native traces exact fallback unless the cursor transition is independently closed |
| Novelty/utility over prior art | claims and evaluation narrative | P6 synchronizes the defensible unknown-preserving contract, negative boundary, prior-art/cost comparison, and remaining falsifiers across plans, status, memory, schemas, and review history |

### Stage ledger for the completion program

| Stage | Focus | Exit evidence | Goal state |
|---|---|---|---|
| P0 | Review intake and live gap audit | This map plus focused probes establish which findings are fixed, conservative refusals, or still open at the current head | **complete** |
| P1 | Contract and arithmetic closure | All direct API, queue, cache, observer-projection, schema, and checked-arithmetic adversaries pass with no mutation/finite-upper violations | **complete** |
| P2 | Production comparator path and boundary pairing | A real reference-feature event-loop fixture dispatches one complete macro and resumes detailed replay; perturbations fallback before mutation; otherwise a precise negative falsifier is retained | **complete** |
| P3 | Native state/effect/continuation closure | Latency/completion/front-end/pending state is either consumed and differentially checked or conservatively refused; independent witness/effect ledger is mandatory | **complete** |
| P4 | Service/queue semantics and identifiability | Residual/replacement/sidecar roles have explicit equations and held-out intervention; queue unknowns stay open and fitter modes are separated | **complete** |
| P5 | Independent evaluation and utility | Identity/source bindings, interval/open strata, accepted/refused coverage, full analyzer cost, and compatible baseline comparison are materialized before any post-observation join | **complete** |
| P6 | Synchronization and final falsification | All docs/tests/schemas/artifacts agree; full focused and workspace checks pass; every unresolved mechanism is named as an open falsifier | **complete** |

P0 through P6 are complete.  No later stage is active; the remaining native
and hardware items are explicit falsifiers rather than hidden work.

### P0 implementation evidence (complete 2026-10-02)

- The source audit confirmed that certificate binding, grouped unknown-tail
  propagation, mandatory observer projections, cache-history closure, checked
  scalar arithmetic, explicit service roles, historical observation labels, and
  full analyzer-cost accounting are present in the current head and covered by
  focused tests.
- The audit also confirmed the remaining open boundaries rather than inferring
  them from green tests: the live native canary has no complete independent
  witness and therefore records zero production macro dispatches; the
  reference writer still pairs the same event identity at pre/post; the
  issue-only scalar actor refuses latency >1/front-end/scalar-pipeline state;
  and the accepted native set is empty on the retained corpus.
- Representative checks passed before changing the goal: `sim-semantics` 131
  tests, reference `sim-sm` 503 tests, the focused prediction/evaluation/
  schema Python set 28 tests, and the full Cargo workspace (all tests passed,
  with only existing machine-local ignored tests). The review corpus remains
  the pre-existing untracked input and was not edited.

The next active goal is P1: close any remaining direct contract and arithmetic
holes before touching the production native event loop.

### P1 implementation evidence (complete 2026-10-02)

- A nonempty classified demand with no named pool or assignment witness now
  rejects during envelope validation instead of projecting a finite zero
  service interval. The fully unclassified case remains valid and keeps its
  service/queue upper open.
- Grouped assignment population uses checked accumulation in both validation
  and projection; overflow is a typed `ResourceDemandOverflow` rather than a
  release-build wraparound.
- `TypedRegionContract` now reserves internal event sequence spans and rejects
  duplicate or overlapping effect reservations before a macro can be
  compared. This makes effect multiplicity explicit instead of accepting two
  identical typed effects as one ambiguous transition.
- Focused evidence passed after the changes: `sim-semantics` 134 tests,
  reference `sim-sm` 503 tests, 28 prediction/evaluation/schema Python tests,
  and `cargo fmt --manifest-path sim/Cargo.toml --all -- --check`. The full
  workspace remains the next integration check after the production-path
  phase; no native event-loop behavior was changed in P1.

P1 is closed as a direct-contract hardening stage. The next active goal is P2:
make the comparator-selected boundary reachable through the real reference
event loop and repair the same-event pre/post pairing without weakening exact
fallback.

### P2 implementation evidence (complete 2026-10-02)

- The reference-feature event loop now consumes a validated `TypedMacro` only
  when its event sequence is immediately available, its entry has no pending
  queue work, and its continuation identity is the declared region-end
  sequence. Internal effect positions are explicitly reserved in
  `sim-engine::EventQueue` before the continuation wake is scheduled.
- The narrow scalar transform and native actor now agree on the accepted
  issue-only projection (warp hold, issue cursor, power event counters without
  manufactured completion tokens, and sequence frontier). A macro
  post-snapshot is not falsely paired with the
  original event: the continuation wake is the region-end boundary, and the
  complete typed exit is compared there before the trace can finish.
- A reference-feature integration fixture supplies an independently built
  complete typed witness, records one `native_macro_dispatches` entry, resumes
  one detailed suffix, and matches detailed cycles/instruction count. A second
  run perturbs the typed exit and records zero dispatches with unchanged
  detailed output; the queue/core remain unchanged before fallback.
- The retained native SM86 canary still has no complete independent witness
  and remains exact fallback/zero dispatch. The positive result is therefore a
  closed synthetic reference-feature fixture, not a claim that the retained
  front-end/cache traces are natively compressed.
- Focused evidence passed: reference `sim-sm` 506 tests (including the
  event-loop fixture), `sim-semantics` 134 tests, engine sequence tests, and
  formatting checks. The next goal is P3: close or conservatively refuse all
  remaining native state/effect/continuation classes.

### P3 implementation evidence (complete 2026-10-02)

- The accepted scalar transition remains explicitly latency-one and
  issue-only. Latency >1, live scalar/memory release queues, front-end state,
  pending responses, barriers, local completion history, unprojected timing
  tails, async-copy state, cache fills, and unmodeled tails refuse before a
  summary is constructed or dispatched.
- The native actor validates the entire macro descriptor and detailed
  continuation slot before mutating its warp, instruction buffer, counters, or
  scheduler hold. A malformed second descriptor now returns a typed refusal
  with a byte-for-byte unchanged boundary snapshot.
- The positive reference event-loop fixture now builds its witness from a
  separate detailed `SmCore` run (effects, typed exit, and continuation), then
  compares that ledger to the independently derived macro. The old
  summary-against-itself checks remain model tests only; they are not the native
  evidence claim.
- Focused evidence passed: `sim-semantics` 134 tests, reference `sim-sm` 506
  tests including descriptor no-mutation and independent-witness dispatch, the
  engine sequence tests, and formatting. The next goal is P4: make service,
  queue, and fitter semantics explicit and intervention-identifiable.

### P4 implementation evidence (complete 2026-10-02)

- The current sidecar/residual/replacement roles were rechecked against the
  actual direct-run fixture: independent post-replay never changes the
  observer endpoint, replacement is an explicit model-local intervention, and
  unknown typed/service tails keep uppers open. The stressed replacement
  profile changes the declared lower endpoint without reading an NCU target.
- The v1 compatibility fitter no longer infers dependent latency from
  aggregate cycles/events; like v2 it now requires explicit latency. Queue
  depth/capacity, occupancy, and contention remain interventions marked
  unmodeled because no queue-delay equation is fitted.
- Held-out independent-throughput scoring now uses the stored lower/upper
  initiation-interval endpoints, emits interval coverage, and reports distance
  to the interval rather than comparing only with the median point. A holdout
  on the upper endpoint is covered.
- Focused evidence passed: 30 Python fitter/prediction/evaluation/schema tests,
  `sim-run` 65 library + 14 CLI + 6 integration tests, and `sim-semantics` 134
  tests. The next goal is P5: source/observation identity, full cost/coverage,
  and compatible baseline comparisons.

### P5 implementation evidence (complete 2026-10-02)

- Prediction validation now requires a source binding for every workload and
  checks its trace, control, architecture-card, pchase-card, and launch
  identity against the declared target. Manifest-built observation rows retain
  the same binding when the roster supplies it; joins compare both maps and
  report `source_bindings_checked=true`. Caller-supplied standalone CSV rows
  remain explicitly unverified when no binding can be established.
- The committed R6 observation and evaluation artifacts were regenerated with
  these bindings and remain schema-valid. `manifest_labeled_historical` still
  means copied/document identity, not independently verified GPU UUID/driver
  proof.
- The design evaluator now reports accepted-region coverage, a primary
  negative-boundary result, typed-refusal-only false-summary protection, and an
  explicit `compatible_prior_art_comparison.performed=false` reason when prior
  art was not measured under the same trace/cost contract. Dispatch ratios stay
  outside speedup terminology.
- Focused evidence passed: 31 Python prediction/evaluation/schema tests plus
  the fitter suite, committed artifact schema validation, and the existing
  `sim-run`/semantics checks. P6 final synchronization and falsification is
  now complete below.

### P6 implementation evidence (complete 2026-10-02)

- The combined source/contract review was synchronized across this plan,
  `docs/STATUS.md`, `docs/PROJECT_MEMORY.md`, `docs/PREDICTIVE_CLOSURE.md`,
  `docs/REVIEW_HISTORY.md`, the prediction/observation/evaluation schemas, and
  the committed R6 artifacts. The artifacts validate with JSON Schema and the
  post-prediction join reports `source_bindings_checked=true`.
- Full Cargo workspace validation passed after all changes. Focused Python
  suites passed (31 tests for the affected builder/join/evaluator/fitter
  surfaces). The model-local `causal-frontier-probe` still reports 16→1 only
  for its witnessed model fixture, with multi-warp refusal; it is not native
  speedup evidence.
- A current release `sim-run --features reference` bounded real-SM canary
  completed in 47 seconds with 10,033 JSONL records including its footer,
  zero complete typed boundaries, zero native macro dispatches, and exact
  fallback. A shorter 45-second bound timed out before completion, so the
  longer bounded result is the retained termination evidence rather than an
  unqualified runtime claim.
- Full repository-wide `pytest -q` was not a valid project check: collection
  enters the untracked `research_audit_20261001` public-source corpus, whose
  imported test calls `SystemExit`; the broader tools collection also requires
  unavailable optional packages (`numpy`, `torch`, `flashinfer`, `click`). The
  affected project suites pass independently, and no source under the review
  corpus was edited.
- Final falsifiers remain explicit: a real retained SM86 trace still lacks a
  complete independent witness/closed front-end/cache/response state; NCU
  identity is historical/document-labeled; compatible prior-art cost numbers
  were not measured under this exact trace contract; and no native runtime
  speedup, finite hardware upper, scheduler causality, or SOTA claim is made.

## Accuracy-first NCU completion program (next workstream)

P0–P6 closed the reviewed logic, refusal, identity, and claim-discipline
surfaces. They did **not** close hardware accuracy. The next program therefore
has a different priority and vocabulary:

> Primary result: prediction agreement with independently collected NCU
> measurements under one declared endpoint contract. Secondary result:
> simulator wall-clock runtime. There is no hardware speedup or
> runtime/compression metric in this program.

This protocol follows the Nsight Compute guidance that replay mode and cache
control change reproducibility, that requested sections may require multiple
passes, and that profiler preparation/collection overhead is not part of the
reported workload duration ([Nsight Compute CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html),
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?highlight=instance)).
The validation bar is counter-by-counter correspondence, as in the Accel-Sim
methodology ([Accel-Sim ISCA paper](https://people.ece.ubc.ca/~aamodt/publications/papers/accelsim.isca2020.pdf)),
with NVBit/SASS traces treated as instrumentation inputs rather than hardware
truth ([NVBit](https://github.com/NVlabs/NVBit/blob/master/README.md?plain=1)).

### Accuracy workstream stage map

| Stage | Goal | Required work | Exit evidence / stop condition |
|---|---|---|---|
| A0 | Measurement contract | Declare metric(s), endpoint, clock domain, cache policy, replay mode, launch/kernel scope, clocks, device index, repetitions, build, trace, cards, and controls. Keep predictions separate from observations. | A row-level contract validator rejects metric/endpoint/clock/cache/replay/launch/device mismatches. No NCU value can enter prediction materialization. |
| A1 | Device and replay identity | Independently capture GPU name, UUID, compute capability, driver/CUDA/NCU versions, device index, clocks, power state, and application build. Query metric availability on the actual chip. Choose kernel replay for isolated kernels; application/range replay only when deterministic context/cache state is part of the target. | Each run is labeled independently verified or historical/document-labeled. A mismatch is excluded or relabeled, never silently joined. Record NCU pass counts and replay mode. |
| A2 | Deterministic baseline | Run correctness/native replay first, then collect NCU after predictions. Keep optimized builds, fixed launch inputs, warmup policy, synchronization, clock policy, and section/metric selection identical across repetitions. Use `cache-control=all` as the reproducible baseline; use `none` only as an explicit cache-state intervention. | Repeated runs have termination, kernel/launch identity, and dispersion reports. If replay is nondeterministic, stop accuracy scoring and repair the protocol. |
| A3 | Corpus and splits | Retain the 15-workload six-family roster, then add independent cases by family (scalar/normalization, tensor, attention, convolution, elementwise, memory, synchronization/mixed). Keep family-held-out splits and exact source bindings. Do not use scored NCU values for fitting, selection, or implementation choice. | Prediction artifact exists before each observation campaign; no family/workload crossing; OOD rows are reported separately. |
| A4 | Mechanism-local calibration | Fit only from independent probe data: dependent-chain latency, independent-stream throughput intervals, and explicit capacity interventions. Keep queue depth, occupancy, contention, writeback, cache ownership, and scheduler routing unmodeled unless a separate equation and intervention identify them. | Throughput lower/upper endpoints, dependent-latency intervals, and probe holdouts are scored. Any unidentifiable dimension remains open/refused. |
| A5 | Endpoint accuracy | Collect fresh NCU repetitions (minimum five for final strata; increase for high variance) using the same contract. Score finite interval coverage/width, open-upper lower consistency, point error, median/IQR, family-held-out error, and worst-case rows. Add mechanism counters only when their metric definitions and scopes match the simulator projection. | Accuracy claims require finite rows and stated coverage/error confidence. If all uppers remain open, report no finite accuracy; do not convert open lower consistency into coverage. |
| A6 | Intervention falsification | Run predeclared contrasts: cache all/none, clock regime, dirty/writeback footprint, occupancy/queue-depth probe, and selected memory/tensor/control families. Compare direction and mechanism counters after prediction; do not infer hidden hardware causality from one changed counter. | A wrong-direction or counter-inconsistent intervention blocks the affected mechanism claim and forces fallback/unknown. A passing self-authored test is not hardware evidence. |
| A7 | Real selective path | Supply a complete independently extracted witness for a real SM86 trace, or retain exact fallback. Validate entry/effects/pending/continuation at the distinct region-end event. The synthetic P2 fixture is only plumbing evidence. | One real dispatch with matching detailed continuation is required before any native selective-accuracy claim. Otherwise the negative boundary is the result. |
| A8 | Runtime context and final claims | Measure only simulator wall-clock runtime on identical trace/control/card inputs, with parser/replay/planning/output scope stated. Compare recorded traditional-simulator times only as contextual historical baselines unless the same workload/environment/scope is available. | Report runtime as a separate secondary table. Never call record collapse, dispatch count, or step ratio a speedup. Synchronize docs, artifacts, and remaining falsifiers. |

### NCU campaign recipe

1. Verify the device independently and save the ordinary run manifest.
2. Build the optimized target; record commit/build/card/trace/control paths.
3. Run correctness and simulator replay with a bounded timeout.
4. Materialize predictions and latent sidecars **before** starting NCU.
5. Profile the exact target with fixed `ncu` version, metric sections, replay
   mode, cache policy, clock policy, launch filters, and synchronization.
6. Repeat fresh measurements; report median, IQR, per-repetition values, and
   failures. Never merge across changed cache/replay/clock contracts.
7. Join only after prediction; score endpoint and mechanism strata separately.
8. Re-run the simulator with identical inputs for wall-clock runtime only.

### Accuracy stop rules

- No verified device/endpoint identity → historical label only; no hardware
  accuracy claim.
- Open predicted upper → open/refusal stratum; no finite coverage score.
- Nondeterministic replay or changed NCU settings → separate intervention,
  never pooled with baseline.
- Missing mechanism state → conservative fallback/unknown, not a fitted value.
- No real accepted witness → no native selective-accuracy or runtime claim;
  keep the synthetic fixture explicitly model/reference-level.

### A0 implementation evidence (complete 2026-10-02)

- `docs/sm86_ncu_accuracy_protocol_v1.json` declares the next campaign's
  exact endpoint (`gpc__cycles_elapsed.max` at
  `ncu_grid_completion_semaphore`), `gpc_cycles`, application replay,
  `cache-control=all`, `clock-control=none`, device index 0, five repetitions,
  three warmups, mechanism metrics, and the `sm86_accuracy_campaign_v1`
  launch identity.
- The protocol lists the independently required GPU/build fields and the
  mismatch policy, requires prediction-before-observation artifacts, forbids
  NCU values in predictions, and declares the six-family held-out split map.
- `tools/validate_sm86_ncu_accuracy_protocol.py` validates the contract and
  roster split closure. Its tests pass, the current historical 15-workload
  roster obeys the family mapping, and the protocol JSON passes Draft 2020-12
  schema validation. No new NCU measurement was collected in A0.

The next active goal is A1: independently verify the actual device/replay/build
identity and make the measurement runner emit the required row-level metadata
before the first new accuracy campaign.

### A1 implementation evidence (complete 2026-10-02)

- `tools/capture_sm86_ncu_identity.py` performs metadata-only queries:
  `nvidia-smi` for GPU name/UUID/compute capability/driver/device/clocks/P-state,
  `ncu --query-metrics --chip ga102` for target-metric availability, and
  `nvcc`, `ncu`, `rustc`, and Git/build/input checks for software and source
  identity. It never launches a workload or collects a profile.
- The identity artifact records the declared replay/cache/launch/device
  contract, all required hardware/build/input fields, metric availability, and
  mismatch reasons. `status=independently_verified` and
  `eligible_for_accuracy=true` occur only when every required comparison passes;
  a dirty worktree, wrong GPU, missing input, missing tool version, or missing
  metric is labeled `mismatch`/ineligible rather than silently used.
- `docs/sm86_ncu_identity_v1.schema.json` and five focused tests cover the
  valid fixture, compute-capability/metric mismatch, protocol contract echo,
  and schema shape. The current host metadata query succeeds, but the current
  checkout is correctly labeled ineligible because the worktree is dirty; no
  NCU workload measurement was collected.

The next active goal is A2: run the declared deterministic baseline protocol
and record repetition/termination dispersion before scoring accuracy.

### A2 implementation evidence (complete 2026-10-02)

- `tools/preflight_sm86_ncu_campaign.py` checks the declared protocol,
  independently verified identity artifact, clean/release build identity,
  fixed trace/control/card inputs, exact replay/cache/clock/launch/repetition
  contract, manifest split closure, prediction artifact validity, and absence
  of a pre-existing observation artifact. It reports mismatches and
  `ready_for_collection`; it does not launch NCU or block unrelated execution.
- `validate_run_records` defines the bounded campaign-run record shape:
  return code, timeout flag, elapsed seconds, and metric value. It rejects
  incomplete/failed/over-budget repetitions and summarizes all samples with
  median, quartiles, IQR, and min/max. The preflight report explicitly records
  `collection_performed=false`.
- The current historical manifest/identity is correctly **not ready** for the
  new protocol (dirty worktree, historical launch identity, and two rather
  than five repetitions). This is a protocol mismatch, not a new measurement.
- Focused A2 evidence passed: six preflight tests, five identity tests, five
  protocol tests, protocol/preflight/identity schema validation, and no NCU
  workload was collected. The next goal is A3: expand/confirm the accuracy
  corpus and split/source closure before any fresh campaign.

### A3 implementation evidence (complete 2026-10-02)

- The protocol now declares five expansion slots—scalar, tensor, memory,
  synchronization, and mixed—with required case types and held-out placement;
  these are planning slots only and contain no new measurements.
- `validate_manifest_sources` requires every roster workload to bind a unique
  trace, control sidecar, architecture card, and pchase card, and checks those
  paths before collection. The 15-workload historical roster passes source and
  family/split closure; its historical target-contract mismatch remains
  correctly reported by A2 preflight.
- A2 preflight now invokes source closure and requires a valid prediction
  artifact before reporting readiness. A missing prediction, crossed family,
  missing source path, or observation present before collection is a reported
  mismatch; no NCU data is collected or scored by these checks.
- Focused A3 evidence passed: seven preflight tests, six protocol/source tests,
  and protocol schema validation. The next goal is A4: independent probe
  calibration and mechanism-local prediction preparation.

### A4 implementation evidence (complete 2026-10-02)

- The declared `sm86_service_probe_v2` fitter/tests enforce explicit dependent
  latency and independent-throughput inputs, lower/upper throughput intervals,
  and explicit unmodeled queue-depth/occupancy/contention dimensions. No NCU
  value or workload identity enters the fitter.
- A protocol-aligned future campaign manifest and prediction artifact are now
  prepared at `docs/sm86_accuracy_campaign_v1_manifest.json` and
  `docs/sm86_accuracy_campaign_v1_predictions_pre_ncu.json`. All 15 rows are
  source-bound, pre-observation, five-repetition target rows with no NCU or
  observed fields. The service profile remains explicitly unvalidated and is
  used only as a model-local sidecar.
- Focused A4 evidence passed: 24 fitter/preflight/protocol/prediction tests;
  campaign manifest, prediction artifact, and service profile all pass schema
  validation. No NCU observations were collected or scored.

The next active goal is A5: collect the first deterministic NCU repetitions
only after A1/A2 identity and preflight are independently eligible.

### A5 preflight result (2026-10-02; collection not started)

The protocol-aligned manifest and prediction artifact are valid, but the
current identity/preflight report is not ready for collection: `working_tree_dirty`
is the sole current mismatch, so `identity_ineligible=true` and
`collection_performed=false`. This is the intended A1/A2 safety result; no NCU
workload command was launched. A5 remains active until a clean, independently
identified build can be supplied, after which the five-repetition application-
replay/all-cache/no-clock-control baseline can run.
