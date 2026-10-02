# Third-round rebuttal and revision record

**Revision date:** 2026-10-02
**Branch:** `GPUSim_main`
**Basis:** the second-round aggregated review supplied for this revision
**Status:** complete as a repository revision; remaining limitations are stated
as scientific boundaries, not hidden implementation tasks.

This document is a direct, evidence-bounded response to every second-round
finding. It is a research explanation, not an approval, audit, provenance,
sealing, admission, or execution gate. The alpha policy continues to prohibit
M5/M6 workflow machinery.

## Executive response

The review was correct that the earlier wording conflated four different
claims. The revision now separates them everywhere:

1. **Descriptor equivalence:** a closed-form summary agrees with the supplied
   descriptor recurrence.
2. **Model transition equivalence:** an executable model macro emits explicit
   effects/tokens and agrees with detailed model replay at the declared
   continuation boundary.
3. **Native boundary refinement:** a typed macro is consumed by a closed native
   `SmCore` issue-model seam only when its entry, effects, pending state, exit,
   and continuation are closed; otherwise the native path is exact replay.
4. **Independent hardware comparison:** predictions are materialized before an
   identity-checked NCU join. This compares a declared model-local observer
   candidate with measurements; it does not upgrade levels 1–3 into a silicon
   proof.

The strongest supported contribution is therefore:

> **An unknown-preserving, state-complete selective reification contract for GPU
> trace regions: a region may be replaced only when its typed boundary and
> future-relevant effects are closed; every unsupported or unresolved case
> falls back to exact replay.**

The revision deliberately withdraws claims of a new GPU scheduler, hardware
causal predictor, cache-coherence reconstruction, cycle-accurate NCU predictor,
or native speedup. The negative native result is part of the contribution:
front-end-enabled SM86 launches remain exact fallback rather than acquiring a
finite or precise summary by assumption.

## What changed in the implementation

- `sim/semantics` validates typed entry/exit state, internal effect payloads,
  observable labels, pending-token mapping, event spans, and explicit unknown
  upper bounds. Active unresolved service, visibility, writeback, cache, or
  retirement state cannot be hidden by a finite endpoint pair.
- `sim-sm::ModelExecutor` consumes a nonidentity scalar `MacroTransition`,
  emits `MacroEffect`/pending completion state, and resumes a detailed suffix.
  The model-only probe observes one macro dispatch for a 16-step witnessed
  region; this is not native timing.
- Native reference `SmCore` now consumes comparator-selected typed-macro versus
  exact-replay decisions at the closed issue-model seam. A two-instruction
  scalar macro matches detailed entry/exit and detailed suffix continuation.
  Real front-end-enabled probes select exact replay because fetch/cache and
  other future-relevant state is open.
- Eligibility requires independently originated contiguous witnesses. The
  compact native candidate path supplies residency only; it cannot certify its
  own dependency-free premise. Cross-warp order, same-cycle ambiguity, and
  unsupported completion multiplicity refuse conservatively.
- Service roles distinguish residual closure, replacement timing, and independent
  post-replay reporting. Queue and assignment envelopes are explicit
  model-local abstractions; occupancy/contention remain unmodeled. Capacity
  adjustment is not applied twice.
- Prediction/observation tooling now has explicit v1 manifest, v2 prediction,
  and v2 observation schemas. Predictions use the declared observer projection,
  never an NCU target. Row-level metric/endpoint/clock/hardware identity,
  repetition count, workload matching, and family-held-out split closure are
  checked before a join.
- R6 materializes a 15-workload, six-family corpus, two NCU repetitions per
  workload, pre-NCU predictions, post-NCU observations, and a joined report in
  `docs/review_round3_20261002/`. Open intervals are not counted as finite
  coverage; execution-step reduction is not inferred from record coverage.

## Response to the ranked findings

### Critical 0 — the evaluator did not execute a reified engine

**Response.** Agreed for the original evaluator. It now has three explicit
boundaries rather than one implied claim:

- R2 executes a nonidentity transition in `sim-sm::ModelExecutor` and resumes a
detailed suffix.
- R5 consumes a comparator-selected nonidentity typed macro in a closed native
  `SmCore` issue-model test and verifies detailed continuation state.
- The real front-end-enabled SM86 writer records exact fallback; R6 records
  `macro_execution_observed=false` and `step_reduction=null` for all 15 native
  rows.

The evaluator no longer reports wall-clock/report-generation cost as native
compression. A future native macro implementation must make the explicit macro
branch nonempty before runtime or step reduction is reported.

**Falsifier.** A native test that changes a declared entry/effect/pending/exit
field while the comparator still selects `Preserve`, or a corpus artifact with
`macro_execution_observed=true` without a dispatch counter, invalidates this
boundary.

### Critical 1 — “independent evidence” was a caller range

**Response.** R3 separates witness extraction from descriptor eligibility.
`CausalFrontierEvidence::from_independent_facts` requires contiguous coverage,
an allowed static/independent-trace origin, and equality for dependency,
control, side-effect, and multiplicity fields. Producer-premise-only facts and
stale/perturbed witnesses refuse. The compact native path no longer creates a
complete witness from the candidate segment itself.

**Limit.** This is independent evidence within the declared trace/static
contract, not a proof of proprietary hardware arbitration.

**Falsifier.** Perturbing one dependency/effect fact without changing the
witness must not preserve eligibility; if it does, the witness boundary is
unsound.

### Critical 2 / Major 9C — unknown tails acquired finite uppers

**Response.** `FrontierVector::required_completion` and
`NcuObserverProjection::from_frontiers` distinguish inactive fields from active
unknown requirements. Any active lower-only issue/service/visibility/writeback/
cache/typed tail makes the observer upper `None`, including explicit endpoint
pairs. The R6 native corpus exhibits the intended behavior: all 15 rows have
open uppers because retirement/cache state is unresolved.

`endpoint_start` and `endpoint_end` are treated as elapsed duration; shifting a
pair by a common origin does not change duration.

**Falsifier.** An active unknown frontier producing a finite upper, or an origin
shift changing elapsed duration, is a regression.

### Major 3 / Major 9 — service was a second detailed cost and lacked a queue

**Response.** The live full-trace role is explicitly
`independent_post_replay`; its demand does not silently alter the native
observer frontier. `ServiceQueueProjection` declares serial admission,
issue/completion frontiers, queue depth, capacity, and marginal assignment
semantics. Residual and replacement timing are separate typed roles and tests.
Grouped alternatives preserve lower/upper assignment envelopes when supplied.

**Limit.** This remains an analytical queue envelope, not a recovered SM
scheduler. Native per-record route witnesses are absent, so marginal pool
maxima are not called jointly feasible schedules.

**Falsifier.** A residual/replacement run whose detailed timing changes without
an explicit replacement role, or an all-unclassified demand that obtains a
finite zero service upper, invalidates the composition contract.

### Major 4 / Major 5 — lower-bound proxy and open coverage were presented as accuracy

**Response.** The prediction runner now uses `predicted_lower_cycles` and
`predicted_upper_cycles` from the declared observer projection and labels its
point basis. The evaluator reports `finite_interval`, `open_interval`, and
`refused` strata separately. Open rows have no finite coverage or width; their
lower-bound consistency is a diagnostic only.

R6 results are explicit:

| Split | Rows | Finite rows | Open rows | Finite coverage | Open lower consistency | Mean point APE |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 3 | 0 | 3 | not applicable | 0.667 | 14.640% |
| development | 4 | 0 | 4 | not applicable | 1.000 | 28.232% |
| held-out | 8 | 0 | 8 | not applicable | 0.750 | 25.938% |

These values are model-distance diagnostics against independent retained NCU
measurements, not calibrated interval accuracy.

**Falsifier.** A report that counts `upper=None` as covered finite accuracy, or
that puts an NCU value in `predictions_pre_ncu.json`, violates the revised
contract.

### Major 6 / Major 14 — probe modes did not identify occupancy/contention and slopes were collapsed

**Response.** The v2 fitter now requires separable dependent-latency,
independent-throughput, and queue-capacity probes. Throughput retains lower and
upper initiation-interval endpoints. Occupancy/contention rows are retained as
explicitly unmodeled dimensions; changing only those rows does not silently
change the fitted rate profile.

**Limit.** The checked-in profile is an unvalidated abstract example, not a
silicon-calibrated parameter set. One aggregate probe cannot identify latency,
initiation rate, queueing, startup, occupancy hiding, and overlap jointly.

**Falsifier.** If an occupancy/contention-only perturbation changes a fitted
parameter without a declared equation, or if upper slope uncertainty is
collapsed to a single exact upper, the calibration claim is too strong.

### Major 7 — Python summary ignored the native comparator decision

**Response.** The reference summary now honors explicit `preserve` versus
`reify` decisions. A complete typed boundary with `reify` is exact fallback,
not preservation eligibility. Synthetic comparator tests cover this case.

### Major 8 — record ratio was called compression

**Response.** Coverage and execution are separate fields. The model-local probe
reports `exact_execution_steps=16`, `summarized_execution_steps=1`, and one
macro dispatch. The R6 native corpus reports no macro dispatch and null step
reduction; 15 records are not relabeled as a native 16:1 speedup.

### Major 9A / 9B — unclassified and optional-zero demand arithmetic

**Response.** Unclassified demand retains an open service upper, including an
empty named-pool map. A zero-demand/optional pool contributes zero mandatory
startup latency. Adversarial tests cover both cases.

### Major 9D — synthetic insertion sequence order

**Response.** Event time, insertion sequence, and logical span are independent
coordinates. Sequence monotonicity is required only when a producer supplies a
causal order witness; equal-time phase/order checks remain explicit.

### Major 10 — one scalar native terminal invariant

**Response.** Native design evaluation compares the declared timing projection
(issue, modeled-system-event, terminal, completion, and readiness fields) and
requires all fields to be present before calling a variant invariant. The
closed native issue-model test additionally compares entry/exit and detailed
continuation state. A scalar terminal match alone is no longer a native claim.

### Minor/Open 11 — variant ranking conflated interventions

**Response.** Throughput, latency, and combined scales are separate CLI modes.
Only interval-disjoint pairs are ordered; overlapping or open pairs are
indeterminate. NCU error is never used for model-local ranking.

### Minor/Open 12 — family split was declarative

**Response.** Manifest and prediction validation reject family crossings,
workload split crossings, missing family labels, duplicate workload/intervention
identities, and malformed split names before prediction execution. The R6
manifest has six closed family assignments.

### Minor/Open 13 — NCU joins omitted endpoint/hardware identity

**Response.** `ncu_observation_v2` requires metric, endpoint, clock domain, and
hardware fields on every row. The join compares them to the target contract and
requires exact repetition counts, launch identity, cache policy, and GPU
identity. A duration CSV cannot silently join a GPC-cycle prediction.

### Major/Open 4–14 general limitation

The fixes above make the failure modes explicit; they do not turn an
unobservable internal into an observation. Physical routing, compiler choices,
cache ownership/read-from, cross-warp arbitration, occupancy hiding, and
hardware completion remain black-box variables. The correct response to those
unknowns is widening/refusal, not a guessed point estimate.

## Evidence-level claim map and falsifiers

| Claim | Evidence level | Current evidence | Falsifier / boundary |
|---|---|---|---|
| Closed descriptor summary agrees with its declared recurrence | 1: descriptor equivalence | R1/R2 semantic tests and model probe | Independent detailed model state diverges |
| Model macro emits effects/tokens and resumes detailed suffix | 2: model transition equivalence | `ModelExecutor` 16→1 probe; pending predecessor refuses | Macro-enabled model state differs at any declared field |
| A closed native issue-model scalar region can be consumed and continued | 3: native boundary refinement | `SmCore` two-instruction differential and comparator-selected path | Entry/effect/pending/exit/continuation mismatch |
| Real front-end-enabled traces are safe to preserve | Not claimed | R5/R6 exact fallback | A future complete front-end state/effect proof may upgrade this |
| Observer candidate can be compared after prediction | 4: independent hardware comparison | 15-workload, six-family R6 join with row-level identity | Any target leakage, identity mismatch, or split crossing |
| Current interval is calibrated hardware accuracy | Not claimed | All real uppers are open | Requires finite, independent, held-out intervals |
| Native macro gives runtime/step speedup | Not claimed | R6 reports no native dispatch | Requires measured macro-enabled native execution |
| Cache topology proves coherence/read-from or bottleneck | Not claimed | Candidate domains only | Requires temporal/fence/generation/owner/completion witnesses |

## Novelty and related-work boundary

Event skipping, max-frontier arithmetic, analytical service rates, and GPU
overlap models are established. The project does not claim novelty for those
mechanisms. The defensible distinction is the **machine-checkable reification
contract**: typed state closure, explicit internal effects and pending tokens,
independent witness origin, unknown-preserving observer composition, and exact
fallback at the native boundary. The closed native issue-model result and the
R6 negative front-end result make that boundary inspectable rather than
rhetorical.

This is a methodology/abstraction contribution, not a new hardware scheduler
or proprietary microarchitectural discovery. A paper should compare directly
against interval simulation, GPU overlap/MWP/CWP analytical models, skipping
refinement, and native trace-driven simulators, and should state that the
current revision does not yet establish a silicon speedup or cycle-accurate
endpoint predictor.

## Reviewer-facing conclusion

The third-round revision should be judged on the narrower question: **does the
implementation refuse exactly where the declared causal boundary is not
closed, and does it consume a closed transition without changing the declared
continuation state?** The answer is yes for the model executor and the closed
native issue-model seam, and no unsupported claim is made for full
front-end-enabled SM86 launches. The 15-workload R6 corpus demonstrates that
identity-checked independent evaluation can be run without target leakage, but
also demonstrates why the real native traces remain open/exact fallback.

## Latest-panel addendum (R8.1–R8.4)

The subsequent panel re-review identified additional executable boundaries,
which are now addressed in the active R8 plan:

- model certificates bind to the full consumed descriptor region before any
  state mutation; grouped unknown service tails, missing observer projections,
  and incomplete cache-history domains cannot acquire finite precision;
- an optional independently witnessed typed boundary is consumed by the real
  reference `SmWake` event branch. A validated fixture dispatches a two-
  instruction unit-tail macro and resumes a detailed suffix; invalid/empty
  effects, exact-replay decisions, pending tokens, non-unit completion tails,
  and untouched future state fall back before mutation;
- the service fitter requires explicit dependent latency and independent
  throughput. Queue-depth/capacity interventions are retained but declared
  unmodeled because no queue equation is fit. `sim-run` exposes explicit
  independent-sidecar, residual, and replacement roles; only the latter two
  activate the model-local resource completion requirement.

These changes strengthen refusal and composition semantics; they do not upgrade
the R6 rows to hardware accuracy, native speedup, or a recovered GPU scheduler.

That negative result is not relabeled as success. It is the current falsifiable
boundary and the appropriate rebuttal to the original overclaims.
