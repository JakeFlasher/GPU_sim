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
- A prototype was wired through the intended event-loop seam and passed a
  synthetic helper test, but the first bounded native canary exposed a
  response-owner/queue join failure before a production macro could be
  consumed. The prototype is not retained as production dispatch; the direct
  issue-model helper remains a test seam only.
- The native canary therefore has no dispatch counter or runtime benefit, and
  the pre/post reference pairing remains a diagnostic exact-fallback stream.

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
  at the model/native test seam; the positive helper fixture is built from an
  independently constructed entry/exit/effect region, while perturbed,
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

- All current docs agree that R8.2 production dispatch is reopened after the
  bounded native queue/response-owner join failure; the helper fixture is not
  production evidence. R8.3 is typed eligibility/refusal, R8.4 is explicit
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
