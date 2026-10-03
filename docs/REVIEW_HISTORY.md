# Review and refactor history

This is a narrative research history, not an approval, audit, provenance, or
execution gate. It records what the supplied review identified and what the
current worktree now demonstrates.

## Initial review — 2026-10-01

The review rejected five overclaims:

- descriptor recurrence equality was not native causal equivalence;
- service demand did not reach an observer endpoint;
- unknown completion tails were being given finite upper bounds;
- multi-warp order and completion multiplicity were synthetic/count-only;
- cache topology did not establish ownership, read-from, coherence, or DRAM
  causality.

The defensible contribution was reframed as unknown-preserving,
state-complete selective reification with exact fallback when the boundary is
not closed.

## Implemented refactor milestones

1. **Model-local frontier contract** — service composition, elapsed endpoint
   arithmetic, unknown-tail propagation, and explicit model-local vocabulary.
2. **Native boundary fallback** — typed region contracts, exact projection
   comparison, opt-in reference decisions, and structured reification reasons.
3. **Order/completion/coverage** — multi-warp refusal, independent dependency
   evidence, completion-multiplicity fallback, suffix entry-state refusal, and
   full-trace coverage counters.
4. **Service/observer composition** — capacity-aware probe profiles, v2
   identifiable probe modes, explicit admission/service overlap, and observer
   propagation.
5. **Cache witness separation** — warp-local instrumentation order only where
   supplied; fence, generation, owner/read-from, and hardware completion stay
   unknown; unresolved cache history widens a separate frontier.
6. **Staged evaluation** — service-capacity variants, family-held-out
   calibration/development/held-out prediction rows, post-prediction NCU join,
   interval/refusal/runtime/memory reporting, and a model-local compression
   probe whose execution status is explicit.

## Earlier staged evidence (retained, not the R6 headline)

- Three current SM86 families were captured and predicted before the NCU join:
  atomic histogram, tensor async GEMM, and the mixed-memory canary.
- NCU `gpc__cycles_elapsed.max` values were 8,811, 4,530, and 5,270 cycles;
  the earlier evaluator counted the open lower-bound intervals as covered, a
  convention later corrected by R1 to open/refusal reporting rather than finite
  interval accuracy.
- Lower-bound-proxy APEs were 72.36%, 77.40%, and 91.80%; these are comparison
  results, not calibration or selection features.
- A witnessed single-warp descriptor region reduced 16 exact model steps to
  one summary step. Real memory-heavy traces remained exact/refused.
- Full Rust workspace and focused reification/evaluation Python suites pass;
  the unrelated PTX subset compiler test still observes a pre-existing
  `ptx_subset_compiler_alpha_v1` versus `...artifact_v1` binary/schema mismatch.

The earlier three-canary numbers are retained for historical comparison only.
The R6 corpus below is the current evidence and deliberately does not count
open intervals as covered finite intervals.

## Second-round R1 hardening — 2026-10-01 (complete; R2 active)

The second-round review found several concrete soundness holes rather than
asking for more report volume. The active first stage now addresses those
holes before any executable macro work:

- unresolved/unclassified service demand produces an open upper, and optional
  zero-demand pools contribute no startup latency to the lower bound;
- explicit endpoint projections propagate every active lower-only completion
  frontier instead of checking only typed/cache tails;
- transform intervals separate event time from optional insertion-sequence
  order, so an earlier timestamp inserted later is not rejected by a synthetic
  monotonicity edge;
- design/evaluation summaries compare the declared timing projection, report
  finite interval coverage separately from open refusal, and expose partial
  order ties rather than lexical rankings;
- prediction/observation documents use a v2 target contract and reject metric,
  endpoint, clock, cache, launch, repetition, GPU, or family-split mismatches;
- record coverage is named separately from exact/summarized execution-step
  reduction; no record ratio is presented as compression.

These changes are still model/evaluation evidence. R2 is now complete at the
model-executor level; native `SmCore` remains untouched and no naming change is
treated as native completion.

## Second-round R2 execution seam — 2026-10-01 (complete; R3 active)

The descriptor analyzer now exposes an explicit model executor rather than only
returning report telemetry. `ModelExecutor::execute_macro` derives a
nonidentity scalar `MacroTransition` from the closed-form interval certificate,
installs its issue/completion effects and pending completion vector, and then
accepts a detailed continuation. The model-local probe observes one macro
dispatch for the 16-instruction witnessed region; a pending predecessor forces
`IncomingPipelineWork` fallback. Native `SmCore` still does not consume this
transition, so no native compression claim follows.
The fast-frontier output carries a `macro_dispatches` counter and the schema
requires it. R3 is now active: independently derive eligibility witnesses and
close or refuse global order and same-cycle effects.

## Second-round R3 witness/order closure — 2026-10-01 (complete; R4 active)

Eligibility now requires contiguous `CausalFrontierWitness` facts from an
allowed static/independent-trace origin. The witness is checked against every
dependency, control, side-effect, and completion-multiplicity field; stale
facts and producer-premise-only ranges reify. The compact native path no longer
creates evidence from its own candidate segment, so real traces remain exact or
refused until a separate witness source is supplied. Multi-warp order remains
an explicit `CrossWarpOrderUnknown` refusal, while same-cycle effects retain
the engine's time/phase/insertion-order check. This closes the descriptor
evidence boundary, not a hardware scheduler proof. R4 is now active on service
queue/assignment semantics.

## Second-round review questions

- Can a native typed macro exit be produced for a nontrivial region without
  leaving future-relevant state unknown?
- Can a larger, architecture-compatible trace corpus provide independent
  family splits and a finite-width interval rather than only open bounds?
- Does selective reification change an architectural conclusion, rather than
  only reducing model work, while preserving exact fallback?
- Which rankings remain stable when independent service and cache uncertainty
  intervals are swept?

These are research questions, not prerequisites enforced by repository
machinery.

## Second-round R4 service semantics — 2026-10-01 (complete; R5 active)

Service composition now carries an explicit role: residual closure,
replacement timing, or independent post-replay. Because current trace demand
counts the same full instruction population already represented by detailed
replay, the live path uses `independent_post_replay` and leaves the observer
frontier unchanged. A separate `ServiceQueueProjection` reports serial
admission, queue depth, open completion uppers, and unresolved marginal pool
 assignment. The v2 fitter now records occupancy/contention as unmodeled rather
than pretending their presence identifies a parameter, and retains a
lower/upper initiation-interval envelope. Grouped assignment envelopes are
preserved when supplied; the native metrics path remains marginal without a
per-record route witness. R5 is now active on native state refinement.

## Second-round R5 native refinement — 2026-10-01/02 (complete for closed issue-model seam; followed by R6)

The typed region validator now checks internal effect payload spans, and the
trace-aware native comparator checks effect/label sequences in addition to
entry/exit state. Comparator decisions are converted into an explicit
typed-macro versus exact-replay plan in the opt-in native writer. This improves
fallback soundness but does not yet consume a nonidentity native macro; the
retained SM86 canary remains exact-fallback evidence. A pure-scalar SM86 probe
now supplies a complete typed pre/post boundary pair; its cold front-end
transition is intentionally rejected by the strict typed scalar macro executor
instead of being claimed as preserved. The reference trace now retains paired
pre/post boundaries and explicit issue-effect spans, making that rejection a
transition-level check rather than endpoint-only telemetry. The reference
execution path consumes `NativeBoundaryExecutionResult`; the bounded probe
selects its exact-replay branch because the front-end state is not closed. The
native issue-model differential consumes the comparator-selected path for a
two-instruction scalar macro, matches detailed entry/exit state and terminal
cycles, resumes a detailed suffix, and verifies that exact replay is
non-mutating. Full front-end-enabled launches remain conservative fallback;
R6 therefore expands the independent corpus/observer evaluation rather than
weakening this boundary.

## Third-round R6 corpus and observer evaluation — 2026-10-02 (complete; followed by R7)

R6 materializes the requested independent evaluation without turning the
profiler target into a feature or execution selector:

- The checked-in manifest covers 15 retained SM86 workloads in six families;
  normalization is calibration, tensor GEMM is development, and attention,
  convolution, elementwise/reduction, and memory embedding are held out. The
  roster validator rejects workload or family split crossings before a
  simulator process starts.
- `run_prediction_manifest.py` writes the prediction document first, using the
  declared observer projection fields rather than the simulator-only field.
  The output records `native_detailed_replay_with_model_local_latent_sidecar`,
  `macro_execution_observed=false`, and null step reduction. No NCU value is
  present in the prediction document.
- `build_ncu_observation_set.py` then reads exactly two retained NCU CSV
  repetitions per workload. The post-prediction document carries row-level
  metric, endpoint, clock, cache, launch, repetition, and historical GPU
  identity. `evaluate_prediction_splits.py` rejects mismatched or unknown rows
  before joining.
- All 15 native rows have open uppers because unresolved retirement/cache tails
  remain active. Finite interval coverage is therefore not applicable; open
  lower-bound consistency is 0.667 (calibration), 1.000 (development), and
  0.750 (held-out). Mean lower-point APE is 14.640%, 28.232%, and 25.938%.
  These are negative/model-distance diagnostics, not calibrated interval or
  hardware accuracy. No native macro dispatch, runtime speedup, or step
  reduction is claimed.
- The complete artifacts and interpretation are in
  [`docs/review_round3_20261002/README.md`](review_round3_20261002/README.md).

R6 closes corpus, split, target-identity, and open/finite reporting hygiene. It
does not produce a front-end-enabled native macro or a finite silicon observer
upper. R7 now maps every paper/rebuttal claim to its evidence level and
falsifier.

## Latest panel re-review R8.0/R8.1/R8.2/R8.3 — 2026-10-02 (R8.2 reopened; R8.5 active)

The latest independent panel re-review found that the previous R7 wording was
still too strong in three places: the native decision was serialized but not
consumed by the production event loop, a nonzero post boundary was paired with
the same queue-event identity, and public/model transition APIs could accept a
mismatched certificate or close unknown queue/cache state. The active response
is recorded in [`latest_review_refinement_20261002.md`](../plans/latest_review_refinement_20261002.md).

R8.1 closed the directly demonstrated soundness holes. Certificates now bind
to every descriptor in the consumed region before mutation; grouped assignment
cannot classify unknown events; observer projection fields are required and
must equal the prediction interval; exact cache history requires temporal,
fence, generation, owner/read-from, and completion domains; replacement
admission and service arithmetic are checked. Adversarial tests verify
no-mutation refusal and open queue/cache uppers.

R8.2 attempted a narrow, optional independently witnessed native event-loop
seam. A synthetic helper fixture passed, but a bounded native canary exposed a
response-owner/queue join failure before a production macro could be consumed.
The prototype is not treated as production dispatch; the retained corpus
remains exact fallback and no native speedup is claimed. R8.3 instead closes
the typed eligibility boundary: latency >1, pending tokens,
front-end/scalar-pipeline state, and nonempty instruction buffers refuse before
an issue-only summary can be called native. Model-local latency summaries are
not native completion.

R8.4 closes the service/queue wording boundary. The v2 fitter now requires an
explicit dependent-chain latency and never infers it from aggregate
throughput. Queue-depth/capacity rows remain interventions but are explicitly
unmodeled because the current profile contains no queue-delay equation;
holdout output labels that reason. `sim-run` now exposes explicit independent-
post-replay, residual-closure, and replacement-timing roles. The default
sidecar leaves native/observer timing unchanged, while a selected replacement
role activates the resource-service completion requirement. A direct model
fixture shows both the tail-dominated unchanged candidate and a stressed
replacement lower-endpoint shift without reading NCU targets. R8.5 is active
on measurement identity and end-to-end cost.

R8.5 closes the measurement and utility wording boundary. The default CSV
observation builder labels copied campaign metadata
`manifest_labeled_historical`; the standalone adapter labels caller-supplied
identity `caller_labeled_unverified`. Neither claims live-device verification.
Prediction artifacts now retain semantic trace/control/card/launch bindings.
The design evaluator reports accepted/refused coverage and full child
wall-clock/peak-RSS scope (parse, replay, planning, effect materialization,
comparison, and output), while explicitly setting native speedup to false.

## R9.0 native canary queue/response closure — 2026-10-02 (complete; R8.2 remains open)

The bounded native reference canary exposed a real typed projection failure:
due response owners and queued memory responses could temporarily disagree
while L2 ownership was still unresolved. The boundary builder now treats that
owner/queue mismatch as unknown ownership and widens the typed completeness
result instead of aborting replay. The rerun completed with a footer: 10,032
reference records, 5,016 pre/post pairs, zero complete typed boundaries, and
exact fallback for every record. No native macro dispatch or speedup was
observed. This repairs termination/unknown propagation, not production macro
dispatch; R8.2 remains the next native research blocker.

R9.1 attempted the optional witness/dispatcher path after that repair. The
bounded native canary had no complete independent witness, so the guarded run
recorded zero native macro dispatches and exact fallback throughout. The real
SM86 production fixture remains open; P2 later supplies only a synthetic
reference-feature fixture as the matching falsifier.

R9.2 audited the two-pass witness requirement. No complete typed
pre/effect/post pair exists in the bounded canary, so constructing a witness
from the same detailed stream is rejected as self-certification. The positive
production falsifier remains explicit and unclaimed.

## P2 synthetic reference event-loop closure — 2026-10-02 (complete; real SM86 witness remains open)

The production reference-feature loop now has a bounded positive fixture that
supplies a complete independently built scalar witness. The comparator-selected
`TypedMacro` reaches the real `SmWake` dispatcher, reserves the logical effect
sequence positions, updates the narrow issue/power projection, schedules the
region-end continuation, and resumes a detailed suffix with the same cycles and
instruction count as detailed replay. The continuation boundary is compared at
its own event identity rather than paired with the original pre-event record.
A perturbed typed exit falls back before queue/core mutation. This is a
synthetic reference-feature differential, not evidence that the retained
front-end/cache SM86 corpus can be compressed; that corpus remains exact
fallback with zero native macro dispatches.

## P3 native state/effect closure — 2026-10-02 (complete; conservative scope)

The accepted native scalar seam now refuses every state class it does not
consume: non-unit latency and release/completion queues, front-end buffers and
cursors, pending responses, barriers, local-completion history, unprojected
timing tails, async/cache fills, and unmodeled tails. Descriptor and
continuation lookup are validated before any warp, counter, or queue mutation.
The positive synthetic witness is built from a separate detailed `SmCore`
effect/exit ledger; equal endpoints from the same transform remain model-only
tests. This closes the omission boundary by explicit refusal, not by claiming
native latency, cache, register, or scheduler equivalence.

## P4 service/queue identifiability — 2026-10-02 (complete)

The service layer now has an explicit admission/overlap/completion convention:
independent post-replay is a sidecar, while residual/replacement roles are
opt-in model-local interventions. Unknown demand keeps queue and observer
uppers open, and grouped assignment remains a marginal route envelope rather
than a hardware schedule. Both v1 compatibility and v2 fitting require an
explicit dependent-latency observation; queue depth/capacity, occupancy, and
contention remain unmodeled dimensions without a fitted delay equation.
Held-out throughput scoring uses lower/upper initiation-interval endpoints and
reports interval coverage rather than only median-point agreement. No queue
calibration or hardware scheduler causality is claimed.

## P5 evaluation/utility closure — 2026-10-02 (complete)

Prediction artifacts now require exact workload source bindings, and the
manifest-built observation set carries matching trace/control/card/launch
bindings when those fields are available. The post-prediction join compares
the binding maps and records whether that check ran; standalone caller-labeled
CSV observations remain explicitly unverified. The committed R6 artifacts were
regenerated and schema-validated. The design evaluator reports accepted-region
coverage, refusal/open strata, full child parse/replay/planning/effect/compare/
output cost, and a primary negative-boundary result. It explicitly records
that compatible prior-art cost comparison was not performed when inputs were
not measured under the same trace contract; no dispatch ratio is called a
speedup.

## P6 integration and final falsification — 2026-10-02 (complete)

The complete revision passes the full Cargo workspace and all affected Python
prediction/evaluation/schema/fitter suites. A current release reference canary
terminates in 47 seconds with 10,033 records, a footer, zero complete typed
boundaries, zero native macro dispatches, and exact fallback. The shorter
45-second probe timed out before completion, so no runtime claim is attached to
that bound. Repository-wide pytest collection remains unavailable because the
untracked review corpus exits during import and broader campaign tests require
optional packages. The final claim is therefore the unknown-preserving,
state-closure/refusal methodology plus the synthetic reference-feature
differential; real native compression, finite hardware prediction, scheduler
causality, and compatible prior-art speedup remain explicit falsifiers.

## R10.0 live SM86 development campaign — 2026-10-02 (complete)

The current RTX 3090 Ti host ran a fresh 168-block metric-domain campaign with
10 cases (read/atom/reduction/control), native correctness, NVBit traces,
simulator replay, analyzer coverage, and three fresh NCU repetitions per case.
The runner now completes the prediction pass before starting post-prediction
NCU collection. All 10 cases completed; all observer uppers remained open;
analyzer candidates and native macro dispatches were zero. Mean lower-point APE
against NCU medians was 43.793%, and open lower consistency was 1.0. This is a
live development/domain result, not held-out accuracy or speedup evidence.
The host changed P-state P0→P2 after the run while clocks remained 1350/10501
MHz; the limitation is retained in the campaign summary.

## R10.1 live writeback campaign — 2026-10-02 (complete)

Two large dirty-store footprints ran through native correctness, NVBit trace,
simulator replay, analyzer, and three fresh post-prediction NCU repetitions.
Both completed, but writeback completion remained explicitly unavailable and
observer uppers remained open. Mean lower-point APE was 32.936%, open
lower-bound consistency was 1.0, and native/analyzer dispatch counts were zero.
This live campaign strengthens the negative tail boundary without fitting a
writeback parameter or claiming hardware scheduler behavior.

## R10.2 live cache-policy contrast — 2026-10-02 (complete)

Two large development cases received three fresh `cache-control=none` NCU
repetitions after their all-cache post-prediction measurements. The observed
median shifts were -158 GPC cycles for the read case and +567 cycles for the
atomic case, with different repeatability ranges. This is an intervention
observation, not a cache-causality proof, model-fitting input, or speedup claim.

## R11.0 full retained-trace simulator cost campaign — 2026-10-02 (complete)

All 15 retained SM86 workloads ran detailed replay and fast-frontier analysis
on identical inputs. The campaign covered 5,130,860 records, accepted zero
candidate records, and observed zero native macro dispatches. Detailed replay
and analyzer wall/RSS costs are reported separately without calling the ratio
speedup. The result is utility/refusal evidence, not native acceleration.

## A0/A1 accuracy protocol and identity — 2026-10-02 (complete)

The accuracy-first NCU protocol now declares the exact target/replay/cache/
clock/device/repetition contract, required identity/build/input fields,
prediction-before-observation artifacts, and family-held-out splits. A metadata-
only capture path queries `nvidia-smi`, NCU metric availability, CUDA/NCU/Rust/
Git/build/input identity, and labels mismatches ineligible before any workload
profile. The current dirty checkout is correctly rejected for a new campaign;
no NCU workload data was collected in A0/A1.

## A4 mechanism calibration and prediction preparation — 2026-10-02 (complete)

The probe fitter and holdout checks enforce explicit dependent latency and
independent throughput intervals while leaving queue depth, occupancy, and
contention unmodeled without equations. A protocol-aligned 15-row prediction
artifact is prepared before observation collection, source-bound, and contains
no NCU values. The supplied service profile remains explicitly unvalidated;
no new NCU observations were collected or scored.

## A2/A3 deterministic preflight and corpus closure — 2026-10-02 (complete)

The campaign preflight now checks the declared replay/cache/clock/repetition
contract, identity/build eligibility, source paths, prediction-before-
observation ordering, and bounded repetition termination/dispersion metadata
without launching NCU. The historical 15-workload roster passes family/split
and source closure but is correctly ineligible for the new five-repetition
campaign contract. Five expansion slots (scalar, tensor, memory,
synchronization, mixed) are declared without collecting data.

## A5 first deterministic accuracy campaign — 2026-10-02 (complete)

After committing the coherent implementation and rebuilding the release
simulator, independent identity/preflight passed. Five fresh NCU application-
replay/all-cache/no-clock-control repetitions completed for all 15 workloads.
The first attempt retained the historical launch skip while reducing warmup
and produced no profiled kernels; its raw output is retained, and the corrected
skip-zero run produced 75 reports/CSV imports. Predictions remain separate and
pre-observation. All predicted uppers remain open, so finite coverage and
finite point-error scoring are not applicable; lower consistency is
0.667/1.000/0.750 for calibration/development/held-out. The next work is A6
intervention falsification, not a finite hardware-accuracy claim.

## A6 intervention falsification — 2026-10-02 (complete)

A predeclared cache all→none contrast produced five repetitions for all 15
workloads; a clock-control none→base contrast produced five repetitions for
two representative cases. Each kept separate prediction/observation/identity
artifacts and matching GPU-duration, GPC-cycle, SM-active, instruction, and
DRAM-sector counters. Cache-none target cycles decreased in 13/15 cases and
increased in two; clock-base target cycles increased in both. Simulator
projections were invariant and every upper remains open, so the result is an
observational direction/consistency check only. No causal hardware claim or
finite accuracy claim is made.

## A7 real selective path — 2026-10-02 (complete; negative boundary)

The current release reference path ran on a retained real SM86 embedding trace
under the verified A5 identity/endpoint contract without a self-derived witness.
It terminated cleanly with 10,033 records before the footer, 5,016 queue
events, zero complete typed boundaries, zero native macro dispatches, and exact
fallback. The result is retained in
`docs/sm86_accuracy_a7_real_selective_result_v1.json`. The final defensible
claim is unknown-preserving typed closure with exact fallback, independently
compared with NCU where the endpoint remains open. Native selective dispatch,
finite endpoint accuracy, and hardware-causal scheduling remain unclaimed.
A8 live-campaign foundation — 2026-10-02 (complete)
-------------------------------------------------------------------------

The accuracy-first live-campaign plan is recorded in
`plans/accuracy_first_live_campaign_a8_20261002.md`. A bounded workload
launcher now preserves stdout/stderr, enforces a wall-time/process-group
bound, checks free GPU memory, and distinguishes pass, OOM, OOD, timeout,
launcher, and ordinary runtime failures. Seven focused launcher tests pass.

All 15 active portable-LCD binaries were rebuilt from current source under
CUDA 13.4. Rebuilt binary help no longer exposes the retired
`--enable-cupti-activity` option, and the architecture/SASS verifier passes
all 15 rows. Sequential correctness smoke runs pass for all 15 rows with no
OOM/OOD/timeout or silent shape substitution.

The first fresh vertical slice uses `gemm_hmma_bf16_sm80`. Native correctness,
NVBit tracing, simulator replay, prediction-before-observation, independent
identity capture, and five NCU application-replay repetitions all complete.
The prediction has lower 3023 cycles and an open upper; NCU target values are
5684, 5490, 5514, 5524, and 5517 cycles (median 5517). The post-observation
join reports lower-bound consistency 1.0, finite interval coverage not
applicable, and no observation-driven selection. Dirty-writeback/coherence
remain open simulator tails and native macro dispatch remains zero. The
machine-readable slice summary is `docs/sm86_a8_vertical_slice_v1.json`.

A8.6 handoff synchronization is complete; A8 is complete and A9 is not
started.

The follow-on completion-frontier program is recorded in
`plans/sm86_completion_frontier_and_native_dispatch_20261002.md`. It defers
the broad roster campaign until a live tail-completion case can either earn a
finite upper or produce a precise refusal, and until the native witness path
has a real production dispatch falsifier.

C0 of the completion-frontier program is now implemented. The machine-readable
contract and diagnostic report keep dirty writeback/coherence open and do not
consume NCU targets; C1 begins the live tail microbenchmark matrix.

C1's live matrix is now complete: 13 small memory-domain/writeback rows
plus device-scope message passing and async-copy wait. All rows completed
native correctness, NVBit, replay, prediction-first ordering, and five NCU
repetitions, but every simulator upper remains open for dirty writeback or
coherence. The next experiment must identify a service/ownership interval;
the observed NCU values are not calibration inputs for the same rows. C3's
explicit service probe closes two model-local store frontiers but misses their
independent NCU medians, so the current padding/profile is falsified rather
than promoted. A disjoint store-2 holdout covers its median but not every raw
repetition (4/5), so repeat dispersion remains unresolved. The tail-witness
report also keeps ownership, generation, fence scope, and hardware completion
open for the same trace.

## Correction — 2026-10-03

C0/C1/C3 completion claims above are superseded. The tail-clearing heuristic
used a completion timestamp/topology counts instead of a complete relation.
It has been removed. Identical real store replay keeps queue events but the
observer upper is open again. The async trace/profile used 64 vs 16384 groups;
message passing included managed-memory migration and variable polling. The
1600-cycle padding was manual and the store-pattern variant was not a
family-held-out embedding workload. Raw data is retained, but those summaries
do not establish matched prediction accuracy, native closure, or calibration.
The revised plan repairs these defects before proceeding; C2/C3 are unfinished.

## C0-R correction — 2026-10-03

Removed the unsound replay tail-clearing heuristic. A live store-2 rerun with
explicit service parameters drains modeled writeback/coherence queues but keeps
semantic dirty-writeback/coherence tails and the observer upper open. Previous
finite-upper/C3 claims are withdrawn. C1-R is active for matched async/fence
recaptures; old raw results remain historical.

## C1-R matched recaptures — 2026-10-03

A same-binary 64-group async recapture matched native/NVBit/NCU launch shape and instruction count, but simulator lower 84866 exceeded NCU median 20946: a direct timing-model mismatch. A resident-device message-passing recapture had lower 1753 vs NCU median 4321 and remains open. Both are retained without tuning or finite-upper claims.
